"""Authorization and tensor normalization gates, without images or downloads."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from scripts import ease_size_protocol as protocol


@pytest.fixture
def pinned_plan(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    directory = cache / 'torch/hub/checkpoints'
    directory.mkdir(parents=True)
    checkpoint = directory / 'synthetic.npz'
    checkpoint.write_bytes(b'unit-test pinned bytes')
    variants = []
    for name, (size, dimension, heads, adapter) in protocol.VARIANTS.items():
        variants.append(dict(name=name, vit_size=size, feature_dim=dimension, num_heads=heads,
                             adapter_size=adapter, checkpoint_tag=f'vit_{size}_patch16_224.augreg_in21k_ft_in1k',
                             checkpoint_file=str(checkpoint), checkpoint_sha256=protocol.file_hash(checkpoint),
                             canonical_pretrained_sha256='a' * 64))
    plan = dict(training_authorized=True, gpu='1', reference_commit=protocol.REFERENCE_COMMIT,
                cache_root=str(cache), variants=variants,
                validation=dict(tasks=2, n_epochs_per_task=20, batch_size=48, seed=1993, num_workers=4,
                                training_authorized=True, accuracy_policy=dict(confirmed_by='user', pass_max_drop_pp=1.5, review_max_drop_pp=3.0)),
                efficiency=dict(tasks=1, n_epochs=1, batch_size=64, seed=1993, num_workers=4,
                                measurement_repeats=3, flops_runs=1, training_authorized=True))
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(protocol, 'PLAN', path)
    return SimpleNamespace(plan=plan, path=path, checkpoint=checkpoint)


@pytest.mark.parametrize('name', list(protocol.VARIANTS))
def test_each_authorized_variant_retains_geometry_and_exact_file(pinned_plan, name):
    variant = protocol.load_variant(name)
    assert variant['checkpoint_sha256'] == hashlib.sha256(pinned_plan.checkpoint.read_bytes()).hexdigest()
    assert variant['checkpoint_file'] == str(pinned_plan.checkpoint)
    assert variant['adapter_size'] == protocol.VARIANTS[name][3]


@pytest.mark.parametrize('section,key,value', [
    (None, 'training_authorized', False), (None, 'training_authorized', 1), (None, 'gpu', '0'),
    (None, 'reference_commit', 'b' * 40), ('validation', 'tasks', 1), ('validation', 'n_epochs_per_task', 1),
    ('validation', 'seed', 42), ('validation', 'batch_size', 64), ('efficiency', 'tasks', 2),
    ('efficiency', 'measurement_repeats', True), ('efficiency', 'measurement_repeats', 1),
    ('efficiency', 'flops_runs', 0), ('efficiency', 'num_workers', 0),
])
def test_wider_or_unapproved_scope_rejected(pinned_plan, section, key, value):
    target = pinned_plan.plan if section is None else pinned_plan.plan[section]
    target[key] = value
    pinned_plan.path.write_text(json.dumps(pinned_plan.plan))
    with pytest.raises(ValueError):
        protocol.load_variant('small_fixed64')


@pytest.mark.parametrize('key,value', [('feature_dim', 768), ('num_heads', 12), ('adapter_size', 16),
                                     ('checkpoint_tag', 'vit_small_patch16_224.augreg_in21k'),
                                     ('checkpoint_sha256', 'bad'), ('canonical_pretrained_sha256', 'G' * 64),
                                     ('checkpoint_file', '/tmp/outside-cache.npz')])
def test_wrong_architecture_or_unpinned_cache_rejected(pinned_plan, key, value):
    pinned_plan.plan['variants'][0][key] = value
    pinned_plan.path.write_text(json.dumps(pinned_plan.plan))
    with pytest.raises(ValueError):
        protocol.load_variant('small_fixed64')


def test_corrupt_checkpoint_fails_before_model_import(pinned_plan):
    pinned_plan.checkpoint.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        protocol.load_variant('small_fixed64')
    assert protocol.load_variant('small_fixed64', verify_checkpoint=False)['name'] == 'small_fixed64'


def state_model(split=False):
    state = {'cls_token': torch.zeros(1, 1, 2), 'pos_embed': torch.zeros(1, 197, 2),
             'patch_embed.proj.weight': torch.zeros(2, 3, 16, 16), 'patch_embed.proj.bias': torch.zeros(2),
             'norm.weight': torch.ones(2), 'norm.bias': torch.zeros(2), 'head.weight': torch.zeros(100, 2)}
    for i in range(12):
        prefix = f'blocks.{i}.'
        for name in ('norm1', 'norm2', 'attn.proj', 'mlp.fc1', 'mlp.fc2'):
            for kind in ('weight', 'bias'):
                state[prefix + name + '.' + kind] = torch.full((2,), float(i))
        for kind in ('weight', 'bias'):
            state[prefix + 'attn.qkv.' + kind] = torch.arange(6, dtype=torch.float32)
    if split:
        for name in list(state):
            if 'qkv.' in name:
                value = state.pop(name)
                for index, part in enumerate(('q', 'k', 'v')):
                    state[name.replace('qkv', part + '_proj')] = value[index * 2:(index + 1) * 2].clone()
            elif '.mlp.' in name:
                state[name.replace('.mlp.', '.')] = state.pop(name)
        state['cur_adapter.0.down_proj.weight'] = torch.randn(2, 2)
    return SimpleNamespace(state_dict=lambda: state)


def test_independent_split_reference_matches_native_canonical_digest():
    native, reference = state_model(), state_model(split=True)
    assert len(protocol.canonical_pretrained_state(native)) == len(protocol.canonical_pretrained_state(reference)) == 150
    assert protocol.canonical_pretrained_digest(native) == protocol.canonical_pretrained_digest(reference)
    reference.state_dict()['blocks.0.attn.k_proj.weight'][0] += 1
    assert protocol.canonical_pretrained_digest(native) != protocol.canonical_pretrained_digest(reference)


def test_missing_pretrained_tensor_is_not_silently_ignored():
    model = state_model()
    del model.state_dict()['norm.weight']
    with pytest.raises(RuntimeError, match='Missing'):
        protocol.canonical_pretrained_digest(model)
