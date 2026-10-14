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
# Degrees covered by the official bank and by every checkpoint that does not state its own range.
OFFICIAL_MAX_DEGREE = 16


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ProjectionBank(nn.Module):
    def __init__(self, down, up, max_degree=OFFICIAL_MAX_DEGREE):
        super().__init__()
        self.register_buffer('down', down.detach().clone())
        self.register_buffer('up', up.detach().clone())
        self.max_degree = max_degree

    def forward(self, x, source, target):
        # Official checkpoint packs its adjacent gaps (16 there) from right to left.
        width = self.down.shape[0]
        index = torch.arange(width, device=x.device)
        src = index >= width - width // self.max_degree * source
        tgt = index >= width - width // self.max_degree * target
        mask = tgt.to(x.dtype) - src.to(x.dtype)
        return F.linear(F.linear(x, self.down) * mask, self.up)


class FrozenToCom(nn.Module):
    def __init__(self, layers, checkpoint_hash, max_degree=OFFICIAL_MAX_DEGREE):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_hash = checkpoint_hash
        self.max_degree = max_degree
        self._active = False

    def covered(self, degree):
        if valid_degree(degree) > self.max_degree:
            raise ValueError(f'This ToCom bank covers degrees up to {self.max_degree}, not {degree}')
        return degree

    @classmethod
    def from_checkpoint(cls, path, depth=12, embed_dim=768):
        """Accept official adapter-only or full `model` state dictionaries.

        Require all Q/V banks, exact shapes, finite floating weights, and a
        nonzero bank. Missing checkpoints never become identity substitutes.
        Loading is tensor-only: arbitrary checkpoint Python objects are refused.
        A bank covers degrees 0 to ``max_degree`` of its checkpoint, or 0 to 16
        (the official range) when the checkpoint does not say.
        """
        path = Path(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError('ToCom checkpoint must contain a state dictionary')
        state = payload.get('model', payload)
        if not isinstance(state, dict):
            raise ValueError('ToCom checkpoint model must be a state dictionary')
        max_degree = payload.get('max_degree', OFFICIAL_MAX_DEGREE) if 'model' in payload else OFFICIAL_MAX_DEGREE
        if type(max_degree) is not int or not 1 <= max_degree <= MAX_DEGREE:
            raise ValueError(f'ToCom checkpoint must cover degrees up to an integer in [1, {MAX_DEGREE}]')
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
                if down.ndim != 2 or down.shape[1] != embed_dim or down.shape[0] == 0 or down.shape[0] % max_degree:
                    raise ValueError('ToCom down projection has incompatible dimensions')
                width = down.shape[0] if width is None else width
                if down.shape[0] != width or up.shape != (embed_dim, width):
                    raise ValueError('ToCom projection dimensions must agree across every layer')
                nonzero |= bool(torch.count_nonzero(down) and torch.count_nonzero(up))
                projections[q] = ProjectionBank(down.float(), up.float(), max_degree)
            layers.append(nn.ModuleDict(projections))
        if not nonzero:
            raise ValueError('All-zero/untrained ToCom checkpoint cannot represent a trained compensator')
        return cls(layers, file_sha256(path), max_degree)

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
            self.covered(source)
        self.covered(target)
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
                            current = self.covered(source()) if per_pass else source
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

    def __init__(self, embed_dim, rank, max_degree, dropout):
        super().__init__()
        self.rank = rank
        self.adapter_down = nn.Linear(embed_dim, rank * max_degree, bias=False)
        self.adapter_up = nn.Linear(rank * max_degree, embed_dim, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source, target):
        width = self.adapter_down.out_features
        index = torch.arange(width, device=x.device)
        mask = ((index >= width - self.rank * target).to(x.dtype)
                - (index >= width - self.rank * source).to(x.dtype))
        return self.adapter_up(self.dropout(self.adapter_down(x)) * mask)


class _Pair(nn.Module):
    def __init__(self, embed_dim, rank, max_degree, dropout):
        super().__init__()
        self.q_tocom = _Adapter(embed_dim, rank, max_degree, dropout)
        self.v_tocom = _Adapter(embed_dim, rank, max_degree, dropout)


class _Layer(nn.Module):
    def __init__(self, embed_dim, rank, max_degree, dropout):
        super().__init__()
        self.attn = _Pair(embed_dim, rank, max_degree, dropout)


class TrainableToCom(nn.Module):
    """Bank under distillation; its state dictionary is a FrozenToCom checkpoint.

    ``pair`` is the (source, target) of the next student forward: the student
    runs at ``target`` and is pulled towards the plain model at ``source``.
    The bank holds ``rank`` rows per adjacent pair of degrees up to
    ``max_degree``; a checkpoint must be saved with that ``max_degree``
    unless it is the official 16.
    """

    def __init__(self, depth, embed_dim, rank=2, scale=.1, dropout=.1, max_degree=MAX_DEGREE):
        super().__init__()
        self.max_degree = valid_degree(max_degree)
        self.blocks = nn.ModuleList(_Layer(embed_dim, rank, max_degree, dropout) for _ in range(depth))
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
