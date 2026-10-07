"""HiDe size geometry and state contracts; CPU only, no pretrained downloads."""

from argparse import ArgumentParser
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.hide_prompt import HidePrompt
from models.hide_prompt_utils.network import HIDE_VIT_SIZES, HideNetwork, build_hide_network
from models.hide_prompt_utils.prompt import EPrompt
from models.hide_prompt_utils.statistics import ClassFeatureStatistics
from models.hide_prompt_utils.vit import HideVisionTransformer, MlpMapping, PrefixAttention


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('size,dimension,heads', [('base', 768, 12), ('small', 384, 6), ('tiny', 192, 3)])
def test_factory_keeps_patch16_depth12_and_ratio_based_tii(size, dimension, heads, tmp_path, monkeypatch):
    checkpoint = tmp_path / 'shared.npz'
    checkpoint.touch()
    loaded = []
    monkeypatch.setattr(HideVisionTransformer, 'load_pretrained', lambda subject, path: loaded.append(path))
    # Inspect full architecture without allocating frozen transformer storage.
    with torch.device('meta'):
        subject = build_hide_network(SimpleNamespace(hide_vit_size=size, hide_checkpoint=str(checkpoint)), 100, 10)
    assert HIDE_VIT_SIZES[size] == (dimension, heads)
    assert subject.feature_dim == dimension and subject.num_classes == 100
    assert loaded == [str(checkpoint), str(checkpoint)]
    assert subject.pretrained_model == f'vit_{size}_patch16_224.augreg_in21k_ft_in1k'
    assert subject.pretrained_checkpoint == str(checkpoint.resolve())
    for branch in (subject.tii, subject.prompt):
        assert len(branch.blocks) == 12
        assert branch.patch_embed.img_size == (224, 224)
        assert branch.patch_embed.patch_size == (16, 16)
        assert branch.pos_embed.shape == (1, 197, dimension)
        assert all(block.attn.num_heads == heads for block in branch.blocks)
        assert branch.head.weight.shape == (100, dimension)
    assert subject.tii.mlp.net[0][0].weight.shape == (2 * dimension, dimension)
    assert subject.tii.mlp.net[1][0].weight.shape == (dimension, 2 * dimension)
    assert isinstance(subject.prompt.mlp, nn.Identity)
    assert subject.prompt.e_prompt.prompt.shape == (5, 2, 10, 5, heads, 64)
    assert subject.prompt.prompt_layers == (0, 1, 2, 3, 4)
    assert subject.prompt.e_prompt.use_permute_fix is False


@pytest.mark.parametrize('size', ['small', 'tiny'])
def test_cli_keeps_size_and_shared_dataset_override_without_changing_schedule(monkeypatch, size):
    from main import parse_args
    import sys
    command = ['--model', 'hide-prompt', '--dataset', 'seq-cifar100-224', '--model_config', 'default',
               '--hide_vit_size', size, '--hide_checkpoint', '/tmp/shared-size.npz',
               '--dataset_config', 'ease', '--batch_size', '64', '--stop_after', '2', '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', ['main.py', *command])
    subject = parse_args(cmd=command)
    assert subject.hide_vit_size == size and subject.backbone == 'hide-vit'
    assert subject.dataset_config == 'ease' and subject.batch_size == 64
    assert (subject.hide_tii_epochs, subject.hide_prompt_epochs, subject.hide_ca_epochs, subject.n_epochs) == (20, 50, 30, 70)
    assert subject.hide_mlp_structure == [2]
    assert subject.hide_prompt_length == 5 and subject.hide_pool_size == 10


def test_parser_retains_base_default_and_rejects_unsupported_geometry():
    parser = HidePrompt.get_parser(ArgumentParser())
    assert parser.parse_args([]).hide_vit_size == 'base'
    with pytest.raises(SystemExit):
        parser.parse_args(['--hide_vit_size', 'large'])
    with pytest.raises(ValueError, match='base, small and tiny'):
        build_hide_network(SimpleNamespace(hide_vit_size='large'), 100, 10)


@pytest.mark.parametrize('dimension,heads', [(384, 6), (192, 3)])
@pytest.mark.parametrize('batch', [1, 2, 5])
def test_actual_head_geometry_preserves_author_legacy_prefix_and_attention(dimension, heads, batch):
    subject = EPrompt(5, dimension, 10, 5, heads)
    ids = (torch.arange(batch) % 2).view(-1, 1)
    inputs = torch.randn(batch, 3, dimension)
    prefixes = subject(inputs, prompt_idx=ids)['batched_prompt']
    # Author default deliberately reshapes the gathered [layer, K/V, batch, ...]
    # array directly, including odd heads and mixed per-example prompt IDs.
    expected = subject.prompt[:, :, ids].reshape(5, batch, 2, 5, heads, 64)
    torch.testing.assert_close(prefixes, expected, atol=0, rtol=0)
    attention = PrefixAttention(dimension, heads)
    query, key, value = nn.functional.linear(inputs, attention.qkv.weight, attention.qkv.bias).reshape(
        batch, 3, 3, heads, 64).permute(2, 0, 3, 1, 4)
    key = torch.cat((prefixes[0, :, 0].transpose(1, 2), key), dim=2)
    value = torch.cat((prefixes[0, :, 1].transpose(1, 2), value), dim=2)
    expected = ((query @ key.transpose(-2, -1) / 8).softmax(-1) @ value).transpose(1, 2).reshape(batch, 3, dimension)
    expected = nn.functional.linear(expected, attention.proj.weight, attention.proj.bias)
    torch.testing.assert_close(attention(inputs, prefixes[0]), expected)


@pytest.mark.parametrize('dimension,heads', [(384, 6), (192, 3)])
def test_size_phase_parameters_checkpoint_and_seen_class_routing(dimension, heads):
    # Preserve real feature/head geometry while shortening image/depth work.
    tii = HideVisionTransformer(100, img_size=16, patch_size=16, embed_dim=dimension,
                               depth=1, num_heads=heads, mlp_structure=(2,))
    prompt = HideVisionTransformer(100, img_size=16, patch_size=16, embed_dim=dimension,
                                  depth=1, num_heads=heads, pool_size=10, prompt_layers=(0,))
    subject = HideNetwork(tii, prompt, 10)
    subject.begin_task(0, 10)
    expected = {
        'tii': {f'tii.{name}' for name, _ in tii.named_parameters() if name.startswith(('mlp.', 'head.'))},
        'tii_alignment': {f'tii.{name}' for name, _ in tii.named_parameters() if name.startswith(('mlp.', 'head.'))},
        'prompt': {'prompt.e_prompt.prompt', 'prompt.head.weight', 'prompt.head.bias'},
        'prompt_alignment': {'prompt.head.weight', 'prompt.head.bias'},
        'eval': set(),
    }
    features = torch.randn(3, dimension)
    for phase, names in expected.items():
        subject.set_phase(phase)
        assert {name for name, parameter in subject.named_parameters() if parameter.requires_grad} == names
    assert subject.tii_outputs(features, fc_only=True)['logits'].shape == (3, 100)
    assert subject.prompt_outputs(features, fc_only=True)['logits'].shape == (3, 100)
    logits = torch.zeros(3, 100)
    logits[:, 99] = 1e6  # Future logits never select a future prompt.
    torch.testing.assert_close(subject.infer_prompt_ids(logits), torch.zeros(3, 1, dtype=torch.long))
    assert bool((subject.class_to_task[10:] == -1).all())
    saved = copy.deepcopy(subject.state_dict())
    subject.class_to_task[:10].fill_(9)
    subject.load_state_dict(saved, strict=True)
    assert subject.get_extra_state() == saved['_extra_state']
    assert torch.equal(subject.class_to_task[:10], torch.zeros(10, dtype=torch.long))
    statistics = ClassFeatureStatistics(100, dimension, n_centroids=10, seed=1993)
    assert statistics.means.shape == (100, 10, dimension)
    assert statistics.variances.shape == (100, 10, dimension)
    assert statistics.valid.count_nonzero() == 0
    assert bool((statistics.class_tasks == -1).all())


@pytest.mark.parametrize('dimension', [384, 192])
def test_tii_mapping_ratio_scales_hidden_width_and_keeps_residual_layernorm(dimension):
    subject = MlpMapping(dimension, (2,))
    features = torch.randn(2, dimension)
    expected = subject.net[1](subject.net[0](features)) + features
    expected = nn.functional.layer_norm(expected, (dimension,), subject.norm.weight, subject.norm.bias, 1e-5)
    torch.testing.assert_close(subject(features), expected)
