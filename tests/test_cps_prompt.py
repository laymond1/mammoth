"""CPU contract tests, unmodified author operators, and actual two-task training."""

import ast
import copy
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.coda_prompt_utils.vit import Attention, Block, VisionTransformer
from models.cps_prompt import CPSPrompt
from models.cps_prompt_utils.cps import CriticalPatchSampling
from models.cps_prompt_utils.model import PromptModel
from models.cps_prompt_utils.vit import CPSAttention
from models.low_rank_utils.factory import CHECKPOINTS, SIZES
from models.low_rank_utils.pretrained import canonical_pretrained_digest


@pytest.fixture(autouse=True)
def cpu_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def author_operators():
    """Execute only original definitions; no author imports or online fetching."""
    folder = Path(__file__).parent / 'fixtures/cps_author'
    namespace = {'torch': torch, 'Tuple': tuple, 'Attention': Attention, 'Block': Block, 'VisionTransformer': VisionTransformer}
    for filename, selected in [('cps.py', {'CriticalPatchSampling'}),
                               ('cps_vit.py', {'CPSAttention', 'CPSBlock', 'make_cps_class'})]:
        tree = ast.parse((folder / filename).read_text())
        body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(folder / filename), 'exec'), namespace)  # noqa: S102 -- pinned author operators only
    return namespace


@pytest.mark.parametrize('sampling', ('uniform', 'critical_score'))
def test_sampling_exact_author_selection_and_isolated_rng(sampling):
    native = CriticalPatchSampling(.4, sampling, .1, seed=1993)
    author = author_operators()['CriticalPatchSampling'](.4, sampling, False, .1)
    tokens = torch.arange(2 * 197 * 3, dtype=torch.float32).reshape(2, 197, 3)
    scores = torch.linspace(0, 1, 196).repeat(2, 1)
    torch.manual_seed(1993)
    expected = author(tokens, scores)
    global_before = torch.get_rng_state().clone()
    actual = native(tokens, scores)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 118, 3)
    assert torch.equal(torch.get_rng_state(), global_before)
    assert torch.equal(actual[:, 0], tokens[:, 0])
    assert (native.last_indices[:, 1:].diff(dim=1) > 0).all()


def test_sampling_checkpoint_restores_next_draw():
    sampler = CriticalPatchSampling(seed=7)
    tokens, scores = torch.randn(2, 197, 3), torch.rand(2, 196)
    sampler(tokens, scores)
    saved = copy.deepcopy(sampler.state_dict())
    expected = sampler(tokens, scores)
    restored = CriticalPatchSampling(seed=999)
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(restored(tokens, scores), expected, rtol=0, atol=0)


def test_attention_matches_unmodified_author_importance_output_and_gradients():
    torch.manual_seed(3)
    native = CPSAttention(24, num_heads=3, qkv_bias=True)
    source = author_operators()['CPSAttention'](24, num_heads=3, qkv_bias=True)
    source.load_state_dict(native.state_dict())
    x = torch.randn(2, 17, 24)
    actual, importance = native(x, importance=True)
    expected, source_importance = source(x, register_hook=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(importance, source_importance, rtol=0, atol=0)
    assert importance.shape == (2, 16)
    prefix = [torch.randn(2, 4, 24, requires_grad=True) for _ in range(2)]
    other = [tensor.detach().clone().requires_grad_() for tensor in prefix]
    native(x, prompt=prefix)[0].square().sum().backward()
    source(x, prompt=other)[0].square().sum().backward()
    for left, right in zip(prefix, other):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_full_sparse_network_matches_original_author_forward():
    torch.manual_seed(11)
    geometry = {'img_size': 64, 'patch_size': 16, 'embed_dim': 24, 'depth': 12, 'num_heads': 3}
    native = PromptModel(num_classes=20, n_tasks=2, pool_size=20, prompt_length=8,
                         pretrained=False, geometry=geometry, seed=1993)
    operators = author_operators()
    source = VisionTransformer(**geometry)
    source.__class__ = operators['make_cps_class'](VisionTransformer)
    for block in source.blocks:
        block.__class__ = operators['CPSBlock']
        block.attn.__class__ = operators['CPSAttention']
    source.load_state_dict(native.feat.state_dict())
    source.reduction_ratio, source.sampling, source.temperature = .4, 'critical_score', .1
    source.patchsampling = operators['CriticalPatchSampling'](.4, 'critical_score', False, .1)
    prompt = copy.deepcopy(native.prompt)
    head = copy.deepcopy(native.head)
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        query, _, scores = source(images)
    torch.manual_seed(1993)
    tokens, penalty, _ = source(images, prompt=prompt, q=query[:, 0], train=True, q_attn_scores=scores)
    expected = head(tokens[:, 0])
    actual, native_penalty = native(images, train=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(native_penalty, penalty, rtol=0, atol=0)
    assert native.diagnostics['query_tokens'] == 17
    assert native.diagnostics['prompted_tokens'] == 10


@pytest.mark.parametrize('size', ('tiny', 'small', 'base'))
def test_all_production_sizes_load_exact_canonical150_and_freeze(size):
    net = PromptModel(vit_size=size, checkpoint=CHECKPOINTS[size]['file'])
    dimension, heads = SIZES[size]
    assert net.feat.embed_dim == dimension and net.feat.num_heads == heads
    assert len(net.feat.blocks) == 12 and net.prompt.n_tasks == 10
    assert canonical_pretrained_digest(net.feat) == CHECKPOINTS[size]['canonical_sha256']
    assert all(not parameter.requires_grad for parameter in net.feat.parameters())
    assert all(parameter.requires_grad for parameter in net.prompt.parameters())
    assert net.head.in_features == dimension


def arguments():
    args = CPSPrompt.get_parser(ArgumentParser()).parse_args([])
    for key, value in {'n_epochs': 3, 'label_perc': 1., 'nowand': 1, 'device': 'cpu', 'lr': .001, 'optimizer': 'adam',
                           'optim_wd': 0., 'optim_mom': 0., 'optim_nesterov': False, 'fitting_mode': 'epochs',
                           'eval_epochs': None, 'lr_scheduler': None, 'scheduler_mode': 'epoch', 'debug_mode': False,
                           'code_optimization': 0, 'seed': 1993}.items():
        setattr(args, key, value)
    return args


def dataset():
    return SimpleNamespace(N_CLASSES=100, N_TASKS=10, N_CLASSES_PER_TASK=10, SETTING='class-il',
                           TEST_TRANSFORM=nn.Identity(), get_offsets=lambda task: (10 * task, 10 * (task + 1)))


def test_two_task_three_epoch_actual_meta_contract_phases_and_old_rows():
    torch.manual_seed(1993)
    data = dataset()
    model = CPSPrompt(nn.Identity(), nn.CrossEntropyLoss(), arguments(), None, data)
    images = torch.randn(2, 3, 224, 224)
    old_rows = None
    for task in range(2):
        model.meta_begin_task(data)
        assert model.custom_scheduler.optimizer is model.opt
        assert model.net.prompt.task_count == task and model.net.prompt.n_tasks == 10
        for epoch in range(3):
            model.meta_begin_epoch(epoch, data)
            before_prompt = model.net.prompt.e_p_0.detach().clone()
            before_head = model.net.head.weight.detach().clone()
            value = model.meta_observe(images, torch.tensor([10 * task, 10 * task + 1]), images, epoch=epoch)
            assert math.isfinite(value)
            assert all(parameter.grad is None for parameter in model.net.feat.parameters())
            assert not torch.equal(model.net.head.weight, before_head)
            if epoch == 0:
                assert not torch.equal(model.net.prompt.e_p_0, before_prompt)
                assert model.net.diagnostics['prompted_tokens'] == 118
                assert any(parameter.grad is not None for parameter in model.net.prompt.parameters())
            else:
                torch.testing.assert_close(model.net.prompt.e_p_0, before_prompt, rtol=0, atol=0)
                assert all(parameter.grad is None for parameter in model.net.prompt.parameters())
                assert model.net.diagnostics['prompted_tokens'] == 197
            model.meta_end_epoch(epoch, data)
            model.custom_scheduler.step()
        if old_rows is not None:
            torch.testing.assert_close(model.net.prompt.e_p_0[:10], old_rows, rtol=0, atol=0)
        old_rows = model.net.prompt.e_p_0[:10].detach().clone()
        model.eval()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (2, 10 * (task + 1)) and torch.isfinite(logits).all()
        model.train()
        model.meta_end_task(data)
    assert model.current_task == 2


def test_two_epochs_fail_before_allocation(monkeypatch):
    args = arguments()
    args.n_epochs = 2
    monkeypatch.setattr('models.cps_prompt.PromptModel', lambda **kwargs: pytest.fail('Allocated a model before phase validation'))
    with pytest.raises(ValueError, match='both phases'):
        CPSPrompt(nn.Identity(), nn.CrossEntropyLoss(), args, None, dataset())


def test_model_checkpoint_restores_task_rng_optimizer_and_next_update(monkeypatch):
    def synthetic(**options):
        options.update(pretrained=False, checkpoint=None,
                       geometry={'img_size': 64, 'patch_size': 16, 'embed_dim': 24, 'depth': 12, 'num_heads': 3})
        return PromptModel(**options)

    monkeypatch.setattr('models.cps_prompt.PromptModel', synthetic)
    torch.manual_seed(9)
    data = dataset()
    original = CPSPrompt(nn.Identity(), nn.CrossEntropyLoss(), arguments(), None, data)
    x = torch.randn(2, 3, 64, 64)
    original.meta_begin_task(data)
    original.meta_observe(x, torch.tensor([0, 1]), x, epoch=0)
    original.meta_end_task(data)
    original.meta_begin_task(data)
    original.meta_observe(x, torch.tensor([10, 11]), x, epoch=0)
    saved = copy.deepcopy(original.state_dict())
    optimizer = copy.deepcopy(original.opt.state_dict())
    restored = CPSPrompt(nn.Identity(), nn.CrossEntropyLoss(), arguments(), None, data)
    restored.load_state_dict(saved, strict=True)
    before = restored.net.prompt.e_p_0.detach().clone()
    restored.meta_begin_task(data)
    torch.testing.assert_close(restored.net.prompt.e_p_0, before, rtol=0, atol=0)
    assert restored.current_task == restored.net.prompt.task_count == 1
    restored.opt.load_state_dict(optimizer)
    expected = original.meta_observe(x, torch.tensor([10, 11]), x, epoch=0)
    actual = restored.meta_observe(x, torch.tensor([10, 11]), x, epoch=0)
    assert actual == expected
    for (_, left), (_, right) in zip(original.net.named_parameters(), restored.net.named_parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('method', ('cps-prompt', 'coda-frozen'))
def test_main_parser_and_common_preprocessing_config(monkeypatch, method):
    from main import parse_args
    from models import get_model_class
    monkeypatch.setattr('sys.argv', ['main.py', '--model', method, '--dataset', 'seq-cifar100-224',
                                    '--model_config', 'default',
                                    '--n_epochs', '3', '--batch_size', '64', '--backbone', 'resnet18'])
    args = parse_args()
    assert get_model_class(Namespace(model=method)).NAME == method
    assert (args.dataset_config, args.n_epochs, args.batch_size) == ('unified', 3, 64)
    if method == 'cps-prompt':
        assert (args.vit_type, args.reduction_ratio, args.temperature, args.phase_ratio) == ('tiny', .4, .1, .4)


def test_frozen_coda_control_does_not_change_legacy_coda():
    from models.coda_frozen import CodaFrozen
    from models.coda_prompt import CodaPrompt
    args = arguments()
    args.coda_vit_size, args.coda_checkpoint = 'tiny', CHECKPOINTS['tiny']['file']
    args.pool_size, args.prompt_len, args.ortho_mu, args.mu, args.virtual_bs_iterations = 100, 8, 0., 0., 1
    control = CodaFrozen(nn.Identity(), nn.CrossEntropyLoss(), args, None, dataset())
    assert all(not parameter.requires_grad for parameter in control.net.feat.parameters())
    control.meta_begin_task(dataset())
    x = torch.randn(2, 3, 224, 224)
    before = control.net.prompt.e_p_0.detach().clone()
    assert math.isfinite(control.meta_observe(x, torch.tensor([0, 1]), x, epoch=0))
    assert not torch.equal(control.net.prompt.e_p_0, before)
    del control
    baseline = CodaPrompt(nn.Identity(), nn.CrossEntropyLoss(), args, None, dataset())
    assert all(parameter.requires_grad for parameter in baseline.net.feat.parameters())
