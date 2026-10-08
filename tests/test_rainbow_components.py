"""CPU checks for RainbowPrompt source math, cache semantics and prefix layout."""
import ast
from copy import deepcopy
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.rainbow_prompt_utils.prompt import RainbowPrompt
from models.rainbow_prompt_utils.vit import PrefixAttention, RainbowVisionTransformer


SOURCE = Path('/tmp/cl-rainbow-official-20261008')
SOURCE_HASHES = {'prompt.py': 'f0411c0ae36af03baeeafc10971db7fc109494ee82e15baede3d7de118d92303',
    'attention.py': 'bed8a5f1dbc13717e7c0ae29ae9bab16d15592428d91e9547a4a25cf22f67e5c',
    'vision_transformer.py': '59c16c8929654768bdaf109024d710cbb7c306dbfc2f865e2f617041a6c09df8'}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_prompt(**overrides):
    options = dict(length=4, embed_dim=12, pool_size=3, n_tasks=3, num_layers=2, num_heads=3,
        prompt_tune_idx=(0, 1), self_attn_idx=(0,), D1=6, D2=5, KI_iter=2)
    options.update(overrides)
    return RainbowPrompt(**options)


def tiny_vit(**overrides):
    options = dict(num_classes=20, img_size=32, patch_size=16, embed_dim=12, depth=2, num_heads=3,
        prompt_length=4, pool_size=3, n_tasks=3, prompt_layers=(0, 1), self_attn_idx=(0,),
        D1=6, D2=5, KI_iter=2, warmup=1)
    options.update(overrides)
    return RainbowVisionTransformer(**options)


def source_classes():
    """Execute pinned author class ASTs without registries, downloads or trainers."""
    if any(not (SOURCE / filename).is_file() for filename in SOURCE_HASHES):
        pytest.skip('Optional pinned author checkout is absent')
    from functools import partial
    from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
    from timm.models._manipulate import named_apply
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'partial': partial, 'PatchEmbed': PatchEmbed,
                 'Mlp': Mlp, 'DropPath': DropPath, 'trunc_normal_': trunc_normal_, 'named_apply': named_apply,
                 'logger': logging.getLogger('rainbow_author_components')}
    selected = {'RainbowPrompt', 'PreT_Attention', 'Attention', 'LayerScale', 'Block', 'VisionTransformer',
                'get_init_weights_vit', 'init_weights_vit_timm'}
    for filename, expected in SOURCE_HASHES.items():
        data = (SOURCE / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        tree = ast.parse(data)
        tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
        exec(compile(tree, str(SOURCE / filename), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def source_prompt(official):
    return official.RainbowPrompt(length=4, embed_dim=12, embedding_key='cls', prompt_init='uniform',
        prompt_pool=True, prompt_key=True, pool_size=3, top_k=1, batchwise_prompt=False, prompt_key_init='uniform',
        num_layers=2, use_prefix_tune_for_e_prompt=True, num_heads=3, same_key_value=False,
        prompt_tune_idx=[0, 1], n_tasks=3, D1=6, D2=5, relation_type='attention', use_linear=True,
        KI_iter=2, self_attn_idx=[0])


def source_vit(official, prompted=True):
    return official.VisionTransformer(num_classes=20, img_size=32, patch_size=16, embed_dim=12, depth=2,
        num_heads=3, prompt_length=4, pool_size=3 if prompted else None, n_tasks=3,
        e_prompt_layer_idx=[0, 1], use_e_prompt=prompted, use_prefix_tune_for_e_prompt=prompted,
        prompt_pool=prompted, prompt_key=prompted, top_k=1, D1=6, D2=5,
        warm_up=1, KI_iter=2, self_attn_idx=[0], relation_type='attention', use_linear=True)


@pytest.mark.parametrize('layer,task,p_type', [(0, 0, 'Unique'), (1, 1, 'Unique'),
    (0, 0, 'Rainbow'), (0, 1, 'Rainbow'), (1, 0, 'Rainbow'), (1, 1, 'Rainbow'), (1, 2, 'Rainbow')])
def test_prompt_outputs_gradients_and_cache_match_pinned_author_source(layer, task, p_type):
    official = source_classes()
    torch.manual_seed(70)
    native = tiny_prompt()
    reference = source_prompt(official)
    reference.load_state_dict(native.state_dict(), strict=True)
    features = torch.randn(3, 12)
    probe = torch.randn(3, 4, 12)
    outputs = []
    for module in (native, reference):
        torch.manual_seed(108)
        result = module(torch.zeros(3, 5, 12), layer=layer, cls_features=features,
                        cur_id=task, train=True, p_type=p_type)
        prefix = torch.cat(result['batched_prompt'], dim=1)
        ((prefix * probe).sum() + result['sim_loss']).backward()
        outputs.append((prefix, result['sim_loss']))
    assert all(torch.equal(left, right) for left, right in zip(*outputs))
    left_parameters, right_parameters = dict(native.named_parameters()), dict(reference.named_parameters())
    assert left_parameters.keys() == right_parameters.keys()
    for key, value in left_parameters.items():
        other = right_parameters[key]
        assert (value.grad is None) == (other.grad is None)
        if value.grad is not None:
            assert torch.equal(value.grad, other.grad), key
    assert torch.equal(native.stored_rainbow_prompts, reference.stored_rainbow_prompts)


@pytest.mark.parametrize('prompted', [False, True])
def test_vit_initialization_parameter_names_and_values_match_author_named_apply(prompted):
    official = source_classes()
    torch.manual_seed(1993)
    native = tiny_vit(pool_size=3 if prompted else None)
    torch.manual_seed(1993)
    reference = source_vit(official, prompted)
    assert native.state_dict().keys() == reference.state_dict().keys()
    for name, value in native.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name


@pytest.mark.parametrize('train,epoch,task', [(True, 0, 0), (True, 1, 0), (True, 1, 1), (False, None, 2)])
def test_prompted_vit_forward_matches_pinned_source(train, epoch, task):
    official = source_classes()
    native = tiny_vit()
    native.rainbow_prompt.stored_rainbow_prompts.uniform_()
    reference = source_vit(official)
    reference.load_state_dict(native.state_dict(), strict=True)
    images, features = torch.randn(3, 3, 32, 32), torch.randn(3, 12)
    output = []
    for model in (native, reference):
        torch.manual_seed(91)
        output.append(model(images, task_id=2, learned_id=task, cls_features=features, train=train, epoch_info=epoch))
    for key in ('x', 'pre_logits', 'logits'):
        assert torch.equal(output[0][key], output[1][key]), key
    if train:
        assert torch.equal(output[0]['sim_loss'], output[1]['sim_loss'])
    else:
        assert 'sim_loss' not in output[0] and 'sim_loss' not in output[1]


def test_prefix_head_layout_matches_author_attention_for_asymmetric_key_value():
    official = source_classes()
    native = PrefixAttention(12, 3)
    reference = official.PreT_Attention(12, 3, qkv_bias=True)
    reference.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12) / 100
    key = torch.arange(2 * 2 * 12, dtype=torch.float32).reshape(2, 2, 12) / 10
    value = key.flip(-1) / 3
    assert torch.equal(native(tokens, [key, value]), reference(tokens, [key, value]))


def test_conditioning_is_source_uniform_token_average():
    prompt = tiny_prompt()
    knowledge = torch.randn(2, 4, 12)
    left, right = prompt.task_conditioning_step(knowledge, torch.randn(12)), prompt.task_conditioning_step(knowledge, torch.randn(12))
    assert torch.equal(left, right)
    torch.testing.assert_close(left, knowledge.mean(1, keepdim=True).expand_as(knowledge))


def test_warmup_unique_does_not_update_cache_or_evolution_parameters():
    prompt = tiny_prompt()
    result = prompt(torch.zeros(2, 5, 12), layer=1, cls_features=torch.randn(2, 12), cur_id=1, train=True, p_type='Unique')
    (torch.cat(result['batched_prompt'], 1).square().sum() + result['sim_loss']).backward()
    assert torch.count_nonzero(prompt.stored_rainbow_prompts) == 0
    assert prompt.base_knowledge_1.grad[1].abs().sum() > 0
    assert torch.count_nonzero(prompt.base_knowledge_1.grad[0]) == 0
    assert all(parameter.grad is None for parameter in prompt.query_matcher.parameters())
    assert all(parameter.requires_grad for parameter in prompt.parameters())


def test_old_knowledge_is_detached_and_cache_is_the_pre_optimizer_prompt():
    prompt = tiny_prompt()
    optimizer = torch.optim.SGD(prompt.parameters(), lr=.05)
    result = prompt(torch.zeros(2, 5, 12), layer=1, cls_features=torch.randn(2, 12), cur_id=1, train=True, p_type='Rainbow')
    prefix = torch.cat(result['batched_prompt'], 1)
    cache = prompt.stored_rainbow_prompts.clone()
    assert torch.equal(cache[1, 1], prefix[0].detach())
    (prefix * torch.randn_like(prefix)).sum().backward()
    assert torch.count_nonzero(prompt.base_knowledge_1.grad[0]) == 0
    assert torch.count_nonzero(prompt.base_knowledge_1.grad[2]) == 0
    assert prompt.base_knowledge_1.grad[1].abs().sum() > 0
    previous = prompt.base_knowledge_1.detach().clone()
    optimizer.step()
    assert not torch.equal(previous[1], prompt.base_knowledge_1[1])
    assert torch.equal(prompt.stored_rainbow_prompts, cache)
    assert all(parameter.requires_grad for parameter in prompt.parameters())


def test_evolution_dropout_remains_active_when_module_is_in_eval_mode():
    prompt = tiny_prompt().eval()
    prompt.task_id = 0
    current = torch.randn(1, 4, 12)
    values = []
    for seed in (20, 21):
        torch.manual_seed(seed)
        values.append(prompt.Prompt_Evolution(0, current, current, 12, 6))
    assert not torch.equal(*values)


def test_inference_routes_one_batch_mean_key_within_seen_tasks_without_oracle():
    prompt = tiny_prompt()
    with torch.no_grad():
        prompt.base_key.zero_()
        prompt.base_key[0, 0] = prompt.base_key[1, 1] = prompt.base_key[2, 2] = 1
        for task in range(3):
            prompt.stored_rainbow_prompts[task].fill_(task + 1.)
    features = torch.zeros(3, 12)
    features[0, 0] = 1
    features[1:, 1] = 1
    result = prompt(torch.zeros(3, 5, 12), layer=0, cls_features=features, task_id=0, cur_id=1, p_type='Rainbow')
    assert prompt.task_id == 1
    assert torch.all(torch.cat(result['batched_prompt'], 1) == 2)
    assert 'sim_loss' not in result
    result = prompt(torch.zeros(3, 5, 12), layer=0, cls_features=features, task_id=2, cur_id=0, p_type='Rainbow')
    assert prompt.task_id == 0
    assert torch.all(torch.cat(result['batched_prompt'], 1) == 1)


@pytest.mark.parametrize('options', [dict(length=3), dict(top_k=2), dict(pool_size=2), dict(KI_iter=0),
    dict(prompt_tune_idx=(1, 2)), dict(prompt_tune_idx=(0, 2)), dict(prompt_pool=False), dict(prompt_key=False),
    dict(use_prefix_tune_for_e_prompt=False), dict(use_linear=False), dict(relation_type='sigmoid'),
    dict(same_key_value=True), dict(prompt_init='zero'), dict(D1=True), dict(num_layers=1), dict(self_attn_idx=(2,))])
def test_inactive_or_invalid_prompt_options_fail_before_allocation(options):
    with pytest.raises(ValueError):
        tiny_prompt(**options)


def test_query_branch_has_no_prompt_and_keeps_full_feature_contract():
    model = tiny_vit(pool_size=None)
    result = model(torch.randn(2, 3, 32, 32))
    assert not hasattr(model, 'rainbow_prompt')
    assert result['logits'].shape == (2, 20) and result['pre_logits'].shape == (2, 12)
    assert torch.equal(result['pre_logits'], result['features'])
    assert 'sim_loss' not in result
    assert all(not hasattr(block.attn, 'key_') for block in model.blocks)


def test_prompted_attention_preserves_author_query_cache():
    model = tiny_vit()
    model(torch.randn(2, 3, 32, 32), learned_id=0, cls_features=torch.randn(2, 12))
    assert all(block.attn.key_.shape == (2, 3, 5, 4) for block in model.blocks)


@pytest.mark.parametrize('width,heads', [(192, 3), (384, 6), (768, 12)])
def test_all_vit_sizes_have_twelve_blocks_and_correct_prefix_heads_without_allocating_weights(width, heads):
    with torch.device('meta'):
        model = RainbowVisionTransformer(100, embed_dim=width, num_heads=heads, pool_size=10)
    assert len(model.blocks) == 12 and model.head.weight.shape == (100, width)
    assert model.rainbow_prompt.stored_rainbow_prompts.shape == (10, 12, 20, width)
    assert model.blocks[0].attn.num_heads == heads
    assert model.rainbow_prompt.query_matcher[0].weight.shape == (96, width)
    assert model.rainbow_prompt.fc1[0].weight.shape == (56, width)


def test_prompted_vit_requires_epoch_and_task_state_for_training():
    model = tiny_vit()
    image, feature = torch.randn(2, 3, 32, 32), torch.randn(2, 12)
    with pytest.raises(ValueError, match='epoch'):
        model(image, learned_id=0, cls_features=feature, train=True)
    with pytest.raises(ValueError, match='learned-task'):
        model(image, learned_id=-1, cls_features=feature, train=True, epoch_info=0)
