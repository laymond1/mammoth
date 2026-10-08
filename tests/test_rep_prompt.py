"""REP reconstruction contracts: matching, prefix gradients, actual skips and state."""

import copy
import math
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.low_rank_utils.factory import CHECKPOINTS, SIZES
from models.low_rank_utils.pretrained import canonical_pretrained_digest
from models.rep_prompt import REPPrompt
from models.rep_prompt_utils.merge import progressive_merges, weighted_merge
from models.rep_prompt_utils.model import REPModel
from models.rep_prompt_utils.vit import REPAttention, keeping_probability


@pytest.fixture(autouse=True)
def cpu_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def synthetic(**options):
    defaults = {'num_classes': 100, 'n_tasks': 10, 'pool_size': 20, 'pretrained': False,
                    'geometry': {'img_size': 64, 'patch_size': 16, 'embed_dim': 24, 'depth': 12, 'num_heads': 3},
                    'query_geometry': {'img_size': 64, 'patch_size': 16, 'embed_dim': 12, 'depth': 12, 'num_heads': 3}}
    defaults.update(options)
    return REPModel(**defaults)


def test_matching_oracle_cls_mass_conservation_and_backpropagation():
    tokens = torch.tensor([100., 2., 4., 8., 10.]).reshape(1, 5, 1).requires_grad_()
    metric = torch.tensor([[[-1., -1.], [1., 0.], [1., 0.], [0., 1.], [-1., 0.]]])
    mass = torch.ones(1, 5, 1)
    output, sizes, removed = weighted_merge(tokens, metric, mass, 1)
    torch.testing.assert_close(output.flatten(), torch.tensor([100., 10., 3., 8.]), rtol=0, atol=0)
    torch.testing.assert_close(sizes.flatten(), torch.tensor([1., 1., 2., 1.]), rtol=0, atol=0)
    assert removed == 1 and output[0, 0, 0] == tokens[0, 0, 0]
    torch.testing.assert_close((output * sizes).sum(), (tokens * mass).sum(), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(tokens.grad.flatten(), torch.tensor([1., .5, .5, 1., 1.]), rtol=0, atol=0)


def test_zero_metrics_are_finite_and_progressive_counts_are_capped():
    assert progressive_merges(16, 12) == [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16]
    assert progressive_merges(16, 1) == [0]
    x = torch.randn(2, 5, 3)
    out, mass, removed = weighted_merge(x, torch.zeros_like(x), torch.ones(2, 5, 1), 100)
    assert removed == 2 and out.shape == (2, 3, 3)
    assert torch.isfinite(out).all() and (mass > 0).all()
    torch.testing.assert_close(out[:, 0], x[:, 0], rtol=0, atol=0)


def test_prefix_key_mass_bias_and_image_only_metric_match_closed_form():
    attention = REPAttention(6, num_heads=2, qkv_bias=True)
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.bias.zero_()
        attention.qkv.weight[12:].copy_(torch.eye(6))
        attention.proj.weight.copy_(torch.eye(6))
        attention.proj.bias.zero_()
    x = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    prefix = [torch.zeros(1, 2, 6, requires_grad=True), torch.ones(1, 2, 6, requires_grad=True)]
    mass = torch.tensor([1., 3., 5.]).reshape(1, 3, 1)
    output, metric = attention(x, prefix, mass=mass, need_metric=True)
    expected = ((x * mass).sum(dim=1) + prefix[1].sum(dim=1)) / 11
    torch.testing.assert_close(output, expected[:, None].expand_as(output), rtol=1e-6, atol=1e-6)
    assert metric.shape == (1, 3, 3)
    output.sum().backward()
    assert prefix[1].grad is not None and prefix[1].grad.abs().sum() > 0


def test_ald_equations_initialization_and_spatial_boundary():
    assert keeping_probability(0, 100) == 1.
    temporal = .5 * math.exp(-.01) + .5
    assert keeping_probability(10, 7) == pytest.approx(temporal)
    assert keeping_probability(10, 8) == pytest.approx(.9 * temporal)
    assert 0 < keeping_probability(100000, 8) < 1


@pytest.mark.parametrize('size', ('tiny', 'small', 'base'))
def test_production_encoder_query_and_persistent_dimension_correct_projection(size):
    model = REPModel(vit_size=size)
    dimension, heads = SIZES[size]
    assert model.feat.embed_dim == dimension and model.feat.num_heads == heads
    assert canonical_pretrained_digest(model.feat) == CHECKPOINTS[size]['canonical_sha256']
    assert canonical_pretrained_digest(model.query_encoder) == CHECKPOINTS['tiny']['canonical_sha256']
    assert model.projection.shape == (192, dimension)
    assert 'projection' in dict(model.named_buffers()) and not model.projection.requires_grad
    assert model.threshold == {'tiny': 8, 'small': 10, 'base': 12}[size]
    assert all(not parameter.requires_grad for parameter in model.feat.parameters())
    assert all(not parameter.requires_grad for parameter in model.query_encoder.parameters())


def test_disabled_components_reduce_to_frozen_coda_host():
    from models.coda_prompt_utils.model import Model
    torch.manual_seed(44)
    host = Model(num_classes=100, prompt_param=[10, [100, 8, 0]], vit_size='tiny')
    host.feat.requires_grad_(False)
    torch.manual_seed(44)
    rep = REPModel(pretrained=False, swift=False, atom=False, ald=False)
    torch.testing.assert_close(host.last.weight, rep.head.weight, rtol=0, atol=0)
    torch.testing.assert_close(host.prompt.e_p_0, rep.prompt.e_p_0, rtol=0, atol=0)
    images = torch.randn(2, 3, 224, 224)
    actual, penalty = rep(images, train=True)
    expected, host_penalty = host(images, train=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(penalty, host_penalty, rtol=0, atol=0)
    assert rep.query_encoder is None and rep.projection is None
    assert all(rep.diagnostics['executed']) and rep.diagnostics['output_tokens'] == [197] * 12


def test_layer_skips_are_real_and_do_not_advance_global_algorithm_rng():
    model = synthetic(seed=7)
    model.set_task(0)
    model.step = 1000
    calls = [0] * 12
    hooks = []
    for index, block in enumerate(model.feat.blocks):
        def record(module, inputs, output, index=index):
            calls[index] += 1
        hooks.append(block.attn.register_forward_hook(record))
    images = torch.randn(2, 3, 64, 64)
    global_before = torch.get_rng_state().clone()
    logits, _ = model(images, train=True)
    assert torch.equal(torch.get_rng_state(), global_before)
    trace = model.diagnostics
    assert [bool(count) for count in calls] == trace['executed']
    assert 0 < sum(calls) < 12
    assert all(count <= 1 for count in calls)
    assert sum(trace['merged_tokens']) == 17 - trace['output_tokens'][-1]
    assert torch.isfinite(logits).all()
    for hook in hooks:
        hook.remove()


def test_eval_is_deterministic_full_blocks_and_does_not_advance_ald_state():
    model = synthetic(seed=7)
    model.set_task(0)
    model.step = 1000
    images = torch.randn(2, 3, 64, 64)
    model(images, train=True)
    state = copy.deepcopy(model.rng.get_extra_state())
    model.eval()
    with torch.no_grad():
        first, second = model(images), model(images)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert all(model.diagnostics['executed']) and model.step == 1000
    assert model.diagnostics['output_tokens'][-1] < 17
    assert torch.equal(model.rng.get_extra_state()['states']['cpu'], state['states']['cpu'])


def arguments():
    args = REPPrompt.get_parser(ArgumentParser()).parse_args([])
    for key, value in {'n_epochs': 2, 'label_perc': 1., 'nowand': 1, 'device': 'cpu', 'lr': .001, 'optimizer': 'adam',
                           'optim_wd': 0., 'optim_mom': 0., 'optim_nesterov': False, 'fitting_mode': 'epochs',
                           'eval_epochs': None, 'lr_scheduler': None, 'scheduler_mode': 'epoch', 'debug_mode': False,
                           'code_optimization': 0, 'seed': 1993}.items():
        setattr(args, key, value)
    return args


def dataset():
    return SimpleNamespace(N_CLASSES=100, N_TASKS=10, N_CLASSES_PER_TASK=10, SETTING='class-il',
                           TEST_TRANSFORM=nn.Identity(), get_offsets=lambda task: (10 * task, 10 * (task + 1)))


def test_actual_tiny_two_task_training_mask_reset_and_frozen_encoders():
    torch.manual_seed(1993)
    data = dataset()
    args = arguments()
    args.rep_ald_gamma = 10.  # Explicit test setting exercises skips in a short run.
    model = REPPrompt(nn.Identity(), nn.CrossEntropyLoss(), args, None, data)
    images = torch.randn(2, 3, 224, 224)
    old = None
    skipped = False
    for task in range(2):
        model.meta_begin_task(data)
        assert model.net.step == 0 and model.net.prompt.task_count == task
        assert model.custom_scheduler.optimizer is model.opt
        for epoch in range(2):
            model.meta_begin_epoch(epoch, data)
            before = model.net.head.weight.detach().clone()
            value = model.meta_observe(images, torch.tensor([10 * task, 10 * task + 1]), images, epoch=epoch)
            assert math.isfinite(value) and not torch.equal(before, model.net.head.weight)
            assert model.net.step == epoch + 1
            assert all(parameter.grad is None for parameter in model.net.feat.parameters())
            assert all(parameter.grad is None for parameter in model.net.query_encoder.parameters())
            if epoch == 0:
                assert all(model.net.diagnostics['executed'])
                assert model.net.prompt.e_p_0.grad is not None
            skipped |= not all(model.net.diagnostics['executed'])
            model.meta_end_epoch(epoch, data)
            model.custom_scheduler.step()
        if old is not None:
            torch.testing.assert_close(model.net.prompt.e_p_0[:10], old, rtol=0, atol=0)
        old = model.net.prompt.e_p_0[:10].detach().clone()
        model.eval()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (2, 10 * (task + 1)) and torch.isfinite(logits).all()
        assert model.net.step == 2
        model.train()
        model.meta_end_task(data)
    assert skipped


def test_checkpoint_restores_projection_schedule_rng_and_next_update(monkeypatch):
    def network(**options):
        options.update(pool_size=20, checkpoint=None, query_checkpoint=None)
        return synthetic(**options)
    monkeypatch.setattr('models.rep_prompt.REPModel', network)
    data = dataset()
    original = REPPrompt(nn.Identity(), nn.CrossEntropyLoss(), arguments(), None, data)
    images = torch.randn(2, 3, 64, 64)
    original.meta_begin_task(data)
    original.meta_observe(images, torch.tensor([0, 1]), images, epoch=0)
    original.meta_end_task(data)
    original.meta_begin_task(data)
    original.meta_observe(images, torch.tensor([10, 11]), images, epoch=0)
    saved = copy.deepcopy(original.state_dict())
    optimizer = copy.deepcopy(original.opt.state_dict())
    restored = REPPrompt(nn.Identity(), nn.CrossEntropyLoss(), arguments(), None, data)
    restored.load_state_dict(saved, strict=True)
    assert restored.net.step == original.net.step == 1
    restored.meta_begin_task(data)
    assert restored.net.step == 1
    restored.opt.load_state_dict(optimizer)
    expected = original.meta_observe(images, torch.tensor([10, 11]), images, epoch=0)
    actual = restored.meta_observe(images, torch.tensor([10, 11]), images, epoch=0)
    assert actual == expected and original.net.diagnostics == restored.net.diagnostics
    for (_, left), (_, right) in zip(original.net.named_parameters(), restored.net.named_parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(original.net.projection, restored.net.projection, rtol=0, atol=0)


def test_checkpoint_rejects_inconsistent_task_state_before_mutation():
    model = synthetic()
    initial = copy.deepcopy(model.get_extra_state())
    with pytest.raises(ValueError, match='active task'):
        model.set_extra_state({'task_count': 1, 'step': 5, 'active_task': 0})
    assert model.get_extra_state() == initial
    with pytest.raises(ValueError, match='Unstarted'):
        model.set_extra_state({'task_count': 0, 'step': 5, 'active_task': None})
    assert model.get_extra_state() == initial


def test_parser_discovery_common_recipe_and_explicit_small(monkeypatch):
    from main import parse_args
    from models import get_model_class
    monkeypatch.setattr('sys.argv', ['main.py', '--model', 'rep-prompt', '--model_config', 'default',
                                    '--dataset', 'seq-cifar100-224', '--n_epochs', '3', '--batch_size', '64',
                                    '--rep_vit_size', 'small', '--rep_ald_threshold', '11'])
    args = parse_args()
    assert get_model_class(Namespace(model='rep-prompt')).NAME == 'rep-prompt'
    assert (args.dataset_config, args.rep_vit_size, args.n_epochs, args.batch_size) == ('unified', 'small', 3, 64)
    assert args.rep_ald_threshold == 11
    assert args.rep_swift == args.rep_atom == args.rep_ald == 1
