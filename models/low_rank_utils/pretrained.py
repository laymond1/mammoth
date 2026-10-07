"""Local AugReg loading shared with the independent source adapter.

Only the canonical encoder tensors are copied. Adapter weights, source heads,
task classifiers, scaling factors and CL block weights keep the source RNG
initialization. This module never constructs another model or fetches weights.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch


def canonical_pretrained_state(model):
    """Select the six stem tensors and twelve tensors per transformer block."""
    model = getattr(model, 'image_encoder', model)
    state = model.state_dict()
    names = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'norm.weight', 'norm.bias']
    for index in range(len(model.blocks)):
        names.extend(f'blocks.{index}.{suffix}' for suffix in (
            'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
            'attn.qkv.weight', 'attn.qkv.bias', 'attn.proj.weight', 'attn.proj.bias',
            'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'))
    return {name: state[name] for name in names}


def canonical_pretrained_digest(model):
    """Use the same tensor digest convention as the existing size protocols."""
    digest = hashlib.sha256()
    for name, tensor in sorted(canonical_pretrained_state(model).items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def load_pretrained_encoder(model, checkpoint, prefix=''):
    """Copy exact NPZ encoder weights after validating all shapes and values.

    The attention transforms match the pinned source loader. Deliberately no
    position interpolation, partial loading, head loading or automatic fetching
    is supported by the native B/S/T geometry.
    """
    model = getattr(model, 'image_encoder', model)
    path = Path(checkpoint).expanduser()
    if not path.is_file() or path.suffix != '.npz':
        raise FileNotFoundError('LoRA requires an existing local AugReg .npz checkpoint')
    targets = canonical_pretrained_state(model)
    tensors = {}
    with np.load(path, allow_pickle=False) as archive:
        prefix = prefix or ('opt/target/' if 'opt/target/embedding/kernel' in archive else '')

        def read(key):
            return archive[prefix + key]

        def assign(name, array):
            tensor = torch.from_numpy(np.array(array, copy=True))
            if tensor.shape != targets[name].shape or tensor.dtype != targets[name].dtype or not torch.isfinite(tensor).all():
                raise ValueError(f'LoRA checkpoint shape, dtype or values invalid for {name}')
            tensors[name] = tensor

        assign('patch_embed.proj.weight', read('embedding/kernel').transpose(3, 2, 0, 1))
        assign('patch_embed.proj.bias', read('embedding/bias'))
        assign('cls_token', read('cls'))
        assign('pos_embed', read('Transformer/posembed_input/pos_embedding'))
        assign('norm.weight', read('Transformer/encoder_norm/scale'))
        assign('norm.bias', read('Transformer/encoder_norm/bias'))
        for index, block in enumerate(model.blocks):
            name = f'blocks.{index}.'
            base = f'Transformer/encoderblock_{index}/'
            attention = base + 'MultiHeadDotProductAttention_1/'
            assign(name + 'norm1.weight', read(base + 'LayerNorm_0/scale'))
            assign(name + 'norm1.bias', read(base + 'LayerNorm_0/bias'))
            assign(name + 'norm2.weight', read(base + 'LayerNorm_2/scale'))
            assign(name + 'norm2.bias', read(base + 'LayerNorm_2/bias'))
            assign(name + 'attn.qkv.weight', np.concatenate([
                read(attention + part + '/kernel').reshape(model.embed_dim, -1).T for part in ('query', 'key', 'value')], axis=0))
            assign(name + 'attn.qkv.bias', np.concatenate([
                read(attention + part + '/bias').reshape(-1) for part in ('query', 'key', 'value')]))
            assign(name + 'attn.proj.weight', read(attention + 'out/kernel').transpose(2, 0, 1).reshape(model.embed_dim, -1))
            assign(name + 'attn.proj.bias', read(attention + 'out/bias'))
            for layer in range(2):
                dense = base + f'MlpBlock_3/Dense_{layer}/'
                assign(name + f'mlp.fc{layer + 1}.weight', read(dense + 'kernel').T)
                assign(name + f'mlp.fc{layer + 1}.bias', read(dense + 'bias'))
    for name, tensor in tensors.items():
        targets[name].copy_(tensor)


def learned_initial_state(network):
    """Export source-compatible state excluding canonical pretrained weights."""
    excluded = {'image_encoder.' + name for name in canonical_pretrained_state(network)}
    return {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items() if name not in excluded}


def restore_learned_initial_state(network, state):
    """Restore a matched learned initialization, failing on missing state."""
    expected = learned_initial_state(network)
    if set(state) != set(expected):
        raise ValueError('LoRA learned initialization keys do not match the constructed network')
    combined = network.state_dict()
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype or not torch.isfinite(tensor).all():
            raise ValueError(f'LoRA learned initialization invalid for {name}')
        combined[name] = tensor
    network.load_state_dict(combined, strict=True)
