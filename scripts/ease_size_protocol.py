"""Pinned, explicitly authorized EASE Small/Tiny protocols; no import side effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/cl_efficiency/ease_sizes/validation_plan.json'
REFERENCE_COMMIT = '6d67508ed8b7c4d23350ed4eb28c123914b40ae3'
VARIANTS = {
    'small_fixed64': ('small', 384, 6, 64),
    'tiny_fixed64': ('tiny', 192, 3, 64),
    'small_relative32': ('small', 384, 6, 32),
    'tiny_relative16': ('tiny', 192, 3, 16),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _require_values(actual, expected, scope):
    if any(type(actual.get(key)) is not type(value) or actual.get(key) != value for key, value in expected.items()):
        raise ValueError(f'EASE size {scope} is outside the authorized protocol')


def load_plan():
    plan = json.loads(PLAN.read_text())
    if (plan.get('training_authorized') is not True or plan.get('gpu') != '1'
            or plan.get('reference_commit') != REFERENCE_COMMIT
            or {item.get('name') for item in plan.get('variants', [])} != set(VARIANTS)
            or len(plan.get('variants', [])) != 4):
        raise ValueError('EASE size plan must retain the authorized four variants')
    _require_values(plan['validation'], {'tasks': 2, 'n_epochs_per_task': 20, 'batch_size': 48,
                    'seed': 1993, 'num_workers': 4, 'training_authorized': True}, 'validation')
    _require_values(plan['efficiency'], {'tasks': 1, 'n_epochs': 1, 'batch_size': 64, 'seed': 1993,
                    'num_workers': 4, 'measurement_repeats': 3, 'flops_runs': 1,
                    'training_authorized': True}, 'efficiency')
    _require_values(plan['validation']['accuracy_policy'], {'confirmed_by': 'user',
                    'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.0}, 'accuracy policy')
    return plan


def load_variant(name, verify_checkpoint=True):
    if name not in VARIANTS:
        raise ValueError('Unapproved EASE size variant')
    plan = load_plan()
    variant = next(item for item in plan['variants'] if item['name'] == name).copy()
    size, dimension, heads, adapter = VARIANTS[name]
    _require_values(variant, {'name': name, 'vit_size': size, 'feature_dim': dimension,
                    'num_heads': heads, 'adapter_size': adapter,
                    'checkpoint_tag': f'vit_{size}_patch16_224.augreg_in21k_ft_in1k'}, 'variant')
    for key in ('checkpoint_sha256', 'canonical_pretrained_sha256'):
        value = variant.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('EASE size checkpoints require SHA256 pins')
    checkpoint = Path(variant['checkpoint_file']).expanduser().resolve()
    expected_root = Path(plan['cache_root']).resolve() / 'torch/hub/checkpoints'
    if checkpoint.parent != expected_root or checkpoint.suffix != '.npz':
        raise ValueError('EASE size checkpoint must use the agreed shared NPZ cache')
    if verify_checkpoint and (not checkpoint.is_file() or file_hash(checkpoint) != variant['checkpoint_sha256']):
        raise ValueError('EASE size checkpoint changed or is missing')
    variant['checkpoint_file'] = str(checkpoint)
    return variant


def canonical_pretrained_state(model):
    """Select 150 frozen ViT tensors, normalizing upstream split QKV/MLP names."""
    import torch
    raw = model.state_dict()
    expected = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias',
                'norm.weight', 'norm.bias']
    for index in range(12):
        prefix = f'blocks.{index}.'
        expected.extend(prefix + suffix for suffix in (
            'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
            'attn.qkv.weight', 'attn.qkv.bias', 'attn.proj.weight', 'attn.proj.bias',
            'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'))
    state = {}
    for name in expected:
        if name in raw:
            state[name] = raw[name]
        elif '.attn.qkv.' in name:
            prefix, kind = name.split('qkv.')
            state[name] = torch.cat([raw[prefix + part + '_proj.' + kind].detach().cpu()
                                     for part in ('q', 'k', 'v')], dim=0)
        elif '.mlp.' in name and name.replace('.mlp.', '.') in raw:
            state[name] = raw[name.replace('.mlp.', '.')]
        else:
            raise RuntimeError(f'Missing EASE size pretrained tensor: {name}')
    return state


def canonical_pretrained_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(canonical_pretrained_state(model).items()):
        cpu = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tuple(cpu.shape)).encode())
        result.update(str(cpu.dtype).encode())
        result.update(cpu.numpy().tobytes())
    return result.hexdigest()
