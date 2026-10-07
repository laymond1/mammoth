"""CPU synthetic HiDe-Prompt math and state checks; no datasets or downloads."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.hide_prompt_utils.network import HideNetwork, build_hide_network
from models.hide_prompt_utils.prompt import EPrompt
from models.hide_prompt_utils.vit import HideVisionTransformer, MlpMapping, PrefixAttention


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_network(momentum=.01, corrected=False):
    tii = HideVisionTransformer(6, img_size=8, patch_size=4, embed_dim=8,
                               depth=2, num_heads=2, mlp_structure=(2,))
    prompt = HideVisionTransformer(6, img_size=8, patch_size=4, embed_dim=8,
                                  depth=2, num_heads=2, pool_size=3, prompt_length=2,
                                  prompt_layers=(0, 1), use_permute_fix=corrected)
    return HideNetwork(tii, prompt, 3, momentum)


@pytest.mark.parametrize('batch', [1, 2, 5])
@pytest.mark.parametrize('momentum', [0., .01, .4])
def test_prefix_layout_and_active_gradients_match_source_formula(batch, momentum):
    prompt = EPrompt(length=2, embed_dim=8, pool_size=3, num_layers=2, num_heads=2)
    inputs = torch.randn(batch, 3, 8)
    idx = torch.full((batch, 1), 2, dtype=torch.long)
    actual = prompt(inputs, prompt_mask=idx, prompt_momentum=momentum)['batched_prompt']
    raw = prompt.prompt[:, :, idx]
    if momentum:
        previous = prompt.prompt[:, :, :2].detach().clone().mean(2, keepdim=True)
        previous = previous.unsqueeze(2).repeat(1, 1, batch, 1, 1, 1, 1)
        raw = (1 - momentum) * raw + momentum * previous
    expected = raw.reshape(2, batch, 2, 2, 2, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert prompt.prompt.grad[:, :, :2].count_nonzero() == 0
    assert prompt.prompt.grad[:, :, 2].count_nonzero() > 0


def test_named_prefix_fix_preserves_per_example_keys_and_values():
    original = EPrompt(1, 4, 3, 1, 2)
    corrected = EPrompt(1, 4, 3, 1, 2, use_permute_fix=True)
    corrected.load_state_dict(original.state_dict())
    idx = torch.tensor([[0], [2]])
    x = torch.zeros(2, 1, 4)
    fixed = corrected(x, prompt_idx=idx)['batched_prompt']
    torch.testing.assert_close(fixed[0, 0], original.prompt[0, :, 0], atol=0, rtol=0)
    torch.testing.assert_close(fixed[0, 1], original.prompt[0, :, 2], atol=0, rtol=0)
    assert not torch.equal(original(x, prompt_idx=idx)['batched_prompt'], fixed)


def test_prefix_attention_matches_independent_tensor_math():
    attention = PrefixAttention(8, 2)
    x = torch.randn(3, 4, 8)
    prefix = torch.randn(3, 2, 2, 2, 4)
    qkv = nn.functional.linear(x, attention.qkv.weight, attention.qkv.bias)
    q, k, v = qkv.reshape(3, 4, 3, 2, 4).permute(2, 0, 3, 1, 4)
    k = torch.cat((prefix[:, 0].transpose(1, 2), k), dim=2)
    v = torch.cat((prefix[:, 1].transpose(1, 2), v), dim=2)
    scores = torch.einsum('bhtd,bhsd->bhts', q, k) / 2
    expected = torch.einsum('bhts,bhsd->bhtd', scores.softmax(-1), v)
    expected = expected.transpose(1, 2).reshape(3, 4, 8)
    expected = nn.functional.linear(expected, attention.proj.weight, attention.proj.bias)
    torch.testing.assert_close(attention(x, prefix), expected)


def test_tii_mapping_keeps_source_residual_and_layernorm_epsilon():
    mapping = MlpMapping(8, (2,))
    x = torch.randn(5, 8)
    expected = mapping.net[1](mapping.net[0](x)) + x
    expected = nn.functional.layer_norm(expected, (8,), mapping.norm.weight, mapping.norm.bias, 1e-5)
    torch.testing.assert_close(mapping(x), expected)
    assert mapping.norm.eps == 1e-5


def test_phase_trainability_separates_tii_prompt_and_alignment():
    network = tiny_network()
    for phase, prefixes in [('tii', ('tii.mlp.', 'tii.head.')),
                            ('tii_alignment', ('tii.mlp.', 'tii.head.')),
                            ('prompt', ('prompt.e_prompt.prompt', 'prompt.head.')),
                            ('prompt_alignment', ('prompt.head.',)), ('eval', ())]:
        selected = network.set_phase(phase)
        trainable = {name for name, parameter in network.named_parameters() if parameter.requires_grad}
        assert trainable == {name for name, parameter in network.named_parameters() if name.startswith(prefixes)}
        assert {id(parameter) for parameter in selected} == {
            id(parameter) for parameter in network.parameters() if parameter.requires_grad}
    assert isinstance(network.prompt.mlp, nn.Identity)
    assert network.tii is not network.prompt
    assert network.tii.blocks[0].attn.qkv.weight.data_ptr() != network.prompt.blocks[0].attn.qkv.weight.data_ptr()


def test_prompt_transfer_momentum_and_inactive_restore():
    network = tiny_network(momentum=.2)
    assert network.begin_task(0, 2) == 0
    network.finalize_task()
    previous = network.prompt.e_prompt.prompt[:, :, 0].detach().clone()
    assert network.begin_task(2, 4) == 1
    torch.testing.assert_close(network.prompt.e_prompt.prompt[:, :, 1], previous, atol=0, rtol=0)
    snapshot = network.snapshot_inactive_prompts()
    with torch.no_grad():
        network.prompt.e_prompt.prompt.add_(2)
    network.restore_inactive_prompts(snapshot)
    torch.testing.assert_close(network.prompt.e_prompt.prompt[:, :, 0], snapshot[:, :, 0], atol=0, rtol=0)
    torch.testing.assert_close(network.prompt.e_prompt.prompt[:, :, 2], snapshot[:, :, 2], atol=0, rtol=0)
    current = network.prompt.e_prompt.prompt[:, :, 1].detach().clone()
    network.finalize_task()
    torch.testing.assert_close(network.prompt.e_prompt.prompt[:, :, 1], .8 * current + .2 * previous)
    torch.testing.assert_close(network.prompt.e_prompt.prompt[:, :, 0], previous, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='unfinalized'):
        network.finalize_task()


def test_routing_uses_tii_seen_class_to_task_map_without_oracle():
    network = tiny_network()
    network.begin_task(0, 2)
    network.finalize_task()
    network.begin_task(2, 4)
    logits = torch.tensor([[0., 5., 2., 1., 100., 100.], [0., 0., 1., 5., 100., 100.]])
    assert network.infer_prompt_ids(logits).tolist() == [[0], [1]]
    with pytest.raises(ValueError, match='unseen'):
        network.infer_prompt_ids(logits, 6)
    x = torch.randn(2, 3, 8, 8)
    network.tii_outputs = lambda input: {'logits': logits, 'pre_logits': torch.zeros(2, 8)}
    routed = network.routed_outputs(x)
    assert routed['prompt_idx'].tolist() == [[0], [1]]
    assert routed['logits'].shape == (2, 6)


def test_pre_logits_are_pre_mapping_and_fc_only_applies_mapping_once():
    network = tiny_network()
    network.begin_task(0, 2)
    x = torch.randn(3, 3, 8, 8)
    tii = network.tii_outputs(x)
    torch.testing.assert_close(network.tii_outputs(tii['pre_logits'], fc_only=True)['logits'], tii['logits'])
    output = network.prompt_outputs(x, train=True)
    torch.testing.assert_close(network.prompt_outputs(output['pre_logits'], fc_only=True)['logits'], output['logits'])
    assert output['prompt_idx'].tolist() == [[0], [0], [0]]


def test_checkpoint_restores_task_routing_and_phase_without_reinitialization():
    network = tiny_network()
    network.begin_task(0, 2)
    network.finalize_task()
    network.begin_task(2, 4)
    network.set_phase('prompt')
    state = copy.deepcopy(network.state_dict())
    restored = tiny_network()
    restored.load_state_dict(state)
    assert restored.phase == 'prompt'
    assert int(restored.active_task) == 1
    assert restored.seen_classes == 4
    assert restored.class_to_task.tolist() == [0, 0, 1, 1, -1, -1]
    assert restored.prompt.e_prompt.prompt.requires_grad
    assert not restored.tii.head.weight.requires_grad
    x = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(restored.prompt_outputs(x, train=True)['logits'], network.prompt_outputs(x, train=True)['logits'])
    wrong = tiny_network(momentum=.2)
    with pytest.raises(ValueError, match='configuration'):
        wrong.load_state_dict(state)


@pytest.mark.parametrize('returnt', ['out', 'features', 'both', 'full', 'all'])
def test_mammoth_backbone_return_conventions(returnt):
    network = tiny_network()
    network.begin_task(0, 2)
    network.finalize_task()
    output = network(torch.randn(2, 3, 8, 8), returnt=returnt)
    if returnt == 'out':
        assert output.shape == (2, 6)
    elif returnt == 'features':
        assert output.shape == (2, 8)
    elif returnt == 'both':
        assert [value.shape for value in output] == [(2, 6), (2, 8)]
    else:
        assert output['logits'].shape == (2, 6)
        assert output['features'].shape == (2, 8)


def test_factory_rejects_missing_weights_and_unapproved_sizes_before_construction(tmp_path):
    with pytest.raises(FileNotFoundError, match='hide_checkpoint'):
        build_hide_network(SimpleNamespace(), 100, 10)
    with pytest.raises(ValueError, match='base, small and tiny'):
        build_hide_network(SimpleNamespace(hide_vit_size='large'), 100, 10)


def test_npz_loader_maps_complete_synthetic_transformer_without_overwriting_head(tmp_path):
    model = HideVisionTransformer(3, img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2)
    initial_head = copy.deepcopy(model.head.state_dict())
    rng = np.random.default_rng(19)
    weights = {
        'embedding/kernel': rng.normal(size=(4, 4, 3, 8)).astype('float32'),
        'embedding/bias': rng.normal(size=8).astype('float32'),
        'cls': rng.normal(size=(1, 1, 8)).astype('float32'),
        'Transformer/posembed_input/pos_embedding': rng.normal(size=(1, 5, 8)).astype('float32'),
        'Transformer/encoder_norm/scale': rng.normal(size=8).astype('float32'),
        'Transformer/encoder_norm/bias': rng.normal(size=8).astype('float32'),
    }
    base = 'Transformer/encoderblock_0/'
    for norm in ('LayerNorm_0', 'LayerNorm_2'):
        for key in ('scale', 'bias'):
            weights[base + norm + '/' + key] = rng.normal(size=8).astype('float32')
    attn = base + 'MultiHeadDotProductAttention_1/'
    for name in ('query', 'key', 'value'):
        weights[attn + name + '/kernel'] = rng.normal(size=(8, 2, 4)).astype('float32')
        weights[attn + name + '/bias'] = rng.normal(size=(2, 4)).astype('float32')
    weights[attn + 'out/kernel'] = rng.normal(size=(2, 4, 8)).astype('float32')
    weights[attn + 'out/bias'] = rng.normal(size=8).astype('float32')
    for index, incoming, outgoing in ((0, 8, 32), (1, 32, 8)):
        prefix = base + f'MlpBlock_3/Dense_{index}/'
        weights[prefix + 'kernel'] = rng.normal(size=(incoming, outgoing)).astype('float32')
        weights[prefix + 'bias'] = rng.normal(size=outgoing).astype('float32')
    file = tmp_path / 'synthetic.npz'
    np.savez(file, **weights)
    model.load_pretrained(file)
    torch.testing.assert_close(model.patch_embed.proj.weight, torch.from_numpy(weights['embedding/kernel'].transpose(3, 2, 0, 1)))
    for offset, name in enumerate(('query', 'key', 'value')):
        torch.testing.assert_close(model.blocks[0].attn.qkv.weight[offset * 8:(offset + 1) * 8],
                                   torch.from_numpy(weights[attn + name + '/kernel'].reshape(8, 8).T))
    torch.testing.assert_close(model.blocks[0].attn.proj.weight, torch.from_numpy(weights[attn + 'out/kernel'].transpose(2, 0, 1).reshape(8, 8)))
    for key, value in initial_head.items():
        torch.testing.assert_close(model.head.state_dict()[key], value, atol=0, rtol=0)
    weights['embedding/bias'][0] = np.nan
    np.savez(file, **weights)
    with pytest.raises(ValueError, match='values invalid'):
        model.load_pretrained(file)


@pytest.mark.parametrize('corruption', ['future_mapping', 'noncontiguous_offsets', 'future_offsets', 'active_task'])
def test_checkpoint_rejects_inconsistent_or_future_task_routes(corruption):
    network = tiny_network()
    network.begin_task(0, 2)
    network.finalize_task()
    state = copy.deepcopy(network.state_dict())
    if corruption == 'future_mapping':
        state['class_to_task'][0] = 2
    elif corruption == 'noncontiguous_offsets':
        state['task_offsets'][0, 0] = 1
    elif corruption == 'future_offsets':
        state['task_offsets'][1] = torch.tensor([2, 4])
    else:
        state['active_task'].fill_(3)
    with pytest.raises(ValueError, match='checkpoint'):
        tiny_network().load_state_dict(state)
