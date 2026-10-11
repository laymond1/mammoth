"""Independent implementation of the official ToCom evaluation equations.

Reference: JieShibo/ToCom, models.py at OFFICIAL_COMMIT. Checkpoint names and
the reversed cumulative rank mask match that implementation. No upstream code
is vendored. The bank is inference-only; it never modifies a backbone weight
and works on any encoder whose blocks expose a joint ``attn.qkv`` projection.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .reducers import MAX_DEGREE, valid_degree

OFFICIAL_COMMIT = 'ee8b0d920f7456476d25387a67b3fba26477039a'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ProjectionBank(nn.Module):
    def __init__(self, down, up):
        super().__init__()
        self.register_buffer('down', down.detach().clone())
        self.register_buffer('up', up.detach().clone())

    def forward(self, x, source, target):
        # Official checkpoint packs 16 adjacent gaps from right to left.
        width = self.down.shape[0]
        index = torch.arange(width, device=x.device)
        src = index >= width - width // MAX_DEGREE * source
        tgt = index >= width - width // MAX_DEGREE * target
        mask = tgt.to(x.dtype) - src.to(x.dtype)
        return F.linear(F.linear(x, self.down) * mask, self.up)


class FrozenToCom(nn.Module):
    def __init__(self, layers, checkpoint_hash):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_hash = checkpoint_hash
        self._active = False

    @classmethod
    def from_checkpoint(cls, path, depth=12, embed_dim=768):
        """Accept official adapter-only or full `model` state dictionaries.

        Require all Q/V banks, exact shapes, finite floating weights, and a
        nonzero bank. Missing checkpoints never become identity substitutes.
        Loading is tensor-only: arbitrary checkpoint Python objects are refused.
        """
        path = Path(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError('ToCom checkpoint must contain a state dictionary')
        state = payload.get('model', payload)
        if not isinstance(state, dict):
            raise ValueError('ToCom checkpoint model must be a state dictionary')
        expected = {f'blocks.{i}.attn.{q}_tocom.adapter_{part}.weight'
                    for i in range(depth) for q in ('q', 'v') for part in ('down', 'up')}
        actual = {key for key in state if '_tocom' in key}
        if actual != expected:
            raise ValueError(f'Incomplete/incompatible ToCom checkpoint: missing={sorted(expected - actual)}, '
                             f'unexpected={sorted(actual - expected)}')
        layers, width, nonzero = [], None, False
        for i in range(depth):
            projections = {}
            for q in ('q', 'v'):
                prefix = f'blocks.{i}.attn.{q}_tocom.adapter_'
                down, up = state[prefix + 'down.weight'], state[prefix + 'up.weight']
                if any(not isinstance(v, torch.Tensor) or not v.is_floating_point()
                       or not torch.isfinite(v).all() for v in (down, up)):
                    raise ValueError('ToCom weights must be finite floating tensors')
                if down.ndim != 2 or down.shape[1] != embed_dim or down.shape[0] == 0 or down.shape[0] % MAX_DEGREE:
                    raise ValueError('ToCom down projection has incompatible dimensions')
                width = down.shape[0] if width is None else width
                if down.shape[0] != width or up.shape != (embed_dim, width):
                    raise ValueError('ToCom projection dimensions must agree across every layer')
                nonzero |= bool(torch.count_nonzero(down) and torch.count_nonzero(up))
                projections[q] = ProjectionBank(down.float(), up.float())
            layers.append(nn.ModuleDict(projections))
        if not nonzero:
            raise ValueError('All-zero/untrained ToCom checkpoint cannot represent a trained compensator')
        return cls(layers, file_sha256(path))

    @property
    def tensor_count(self):
        return sum(value.numel() for value in self.buffers())

    @contextmanager
    def applied(self, encoder, source, target, scale):
        """Temporarily correct Q/V outputs, leaving K and all weights untouched.

        ``encoder`` is one encoder, or a list of encoders that share this bank.
        ``source`` is the training degree, or a callable returning the degree
        of the pass being run when the passes of a model were trained differently.
        """
        encoders = list(encoder) if isinstance(encoder, (list, tuple)) else [encoder]
        per_pass = callable(source)
        if not per_pass:
            valid_degree(source)
        valid_degree(target)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('ToCom scale must be finite and nonnegative')
        if any(item.training for item in encoders) or torch.is_grad_enabled():
            raise RuntimeError('Frozen ToCom is restricted to eval + no_grad/inference_mode')
        if self._active:
            raise RuntimeError('Nested ToCom application is not supported')
        if any(len(item.blocks) != len(self.layers) for item in encoders):
            raise ValueError('ToCom and backbone depths differ')
        handles = []
        self._active = True
        try:
            if scale and (per_pass or source != target):
                for item in encoders:
                    for block, bank in zip(item.blocks, self.layers):
                        def correct(module, inputs, output, bank=bank):
                            current = valid_degree(source()) if per_pass else source
                            if current == target:
                                return None
                            q, k, v = output.chunk(3, dim=-1)
                            return torch.cat((q + scale * bank['q'](inputs[0], current, target),
                                              k, v + scale * bank['v'](inputs[0], current, target)), dim=-1)
                        handles.append(block.attn.qkv.register_forward_hook(correct))
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._active = False


class _Adapter(nn.Module):
    """Official ``Adapter(act=False)``: down, dropout, gap mask, zero-initialized up."""

    def __init__(self, embed_dim, width, dropout):
        super().__init__()
        self.adapter_down = nn.Linear(embed_dim, width, bias=False)
        self.adapter_up = nn.Linear(width, embed_dim, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source, target):
        width = self.adapter_down.out_features
        index = torch.arange(width, device=x.device)
        mask = ((index >= width - width // MAX_DEGREE * target).to(x.dtype)
                - (index >= width - width // MAX_DEGREE * source).to(x.dtype))
        return self.adapter_up(self.dropout(self.adapter_down(x)) * mask)


class _Pair(nn.Module):
    def __init__(self, embed_dim, width, dropout):
        super().__init__()
        self.q_tocom = _Adapter(embed_dim, width, dropout)
        self.v_tocom = _Adapter(embed_dim, width, dropout)


class _Layer(nn.Module):
    def __init__(self, embed_dim, width, dropout):
        super().__init__()
        self.attn = _Pair(embed_dim, width, dropout)


class TrainableToCom(nn.Module):
    """Bank under distillation; its state dictionary is a FrozenToCom checkpoint.

    ``pair`` is the (source, target) of the next student forward: the student
    runs at ``target`` and is pulled towards the plain model at ``source``.
    """

    def __init__(self, depth, embed_dim, rank=2, scale=.1, dropout=.1):
        super().__init__()
        self.blocks = nn.ModuleList(_Layer(embed_dim, rank * MAX_DEGREE, dropout) for _ in range(depth))
        self.scale = scale
        self.pair = None
        self._handles = []

    def install(self, encoder):
        if self._handles:
            raise RuntimeError('This bank is already installed')
        if len(encoder.blocks) != len(self.blocks):
            raise ValueError('ToCom and backbone depths differ')
        for block, layer in zip(encoder.blocks, self.blocks):
            def correct(module, inputs, output, adapters=layer.attn):
                if self.pair is None or self.pair[0] == self.pair[1]:
                    return None
                source, target = self.pair
                q, k, v = output.chunk(3, dim=-1)
                return torch.cat((q + self.scale * adapters.q_tocom(inputs[0], source, target),
                                  k, v + self.scale * adapters.v_tocom(inputs[0], source, target)), dim=-1)
            self._handles.append(block.attn.qkv.register_forward_hook(correct))
        return self

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
