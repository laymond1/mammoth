"""RainbowPrompt CL wrapper checks on tiny synthetic image tensors, CPU only."""

from argparse import ArgumentParser, Namespace
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from models.rainbow_prompt import RainbowPrompt
from models.rainbow_prompt_utils.network import RainbowNetwork, RAINBOW_VIT_SIZES, build_rainbow_network
from models.rainbow_prompt_utils.vit import RainbowVisionTransformer


ROOT = Path(__file__).resolve().parents[1]


def args_for_test(**overrides):
    args = RainbowPrompt.get_parser(ArgumentParser()).parse_args([])
    for name, value in dict(n_epochs=2, rainbow_warmup_epochs=1, fitting_mode='epochs', eval_epochs=None,
                            label_perc=1., nowand=1, device='cpu', optim_mom=0.9,
                            optim_nesterov=False).items():
        setattr(args, name, value)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def make_model(monkeypatch, **overrides):
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    args = args_for_test(**overrides)
    common = dict(num_classes=4, img_size=8, patch_size=4, embed_dim=8, depth=2, num_heads=2)
    query = RainbowVisionTransformer(**common)
    prompt = RainbowVisionTransformer(**common, pool_size=2, n_tasks=2, prompt_length=4,
                                     prompt_layers=(0, 1), self_attn_idx=(0,), D1=3, D2=4,
                                     KI_iter=2, warmup=args.rainbow_warmup_epochs)
    net = RainbowNetwork(query, prompt, 2)
    net.pretrained_checkpoint = '/tmp/synthetic-unit-fixture.npz'
    dataset = SimpleNamespace(N_CLASSES=4, N_TASKS=2, N_CLASSES_PER_TASK=2, SETTING='class-il',
                              get_offsets=lambda task: (task * 2, (task + 1) * 2))
    model = RainbowPrompt(net, nn.CrossEntropyLoss(), args, None, dataset)
    model.to(torch.device('cpu'))
    return model, dataset


def samples(offset=0):
    return torch.randn(2, 3, 8, 8), torch.tensor([offset, offset + 1])


def step(model, offset=0, epoch=1):
    images, labels = samples(offset)
    return model.meta_observe(images, labels, images, epoch=epoch)


def test_discovery_parser_config_and_cli_precedence(monkeypatch):
    from models import get_model_class
    from backbone import get_backbone_class
    from main import parse_args
    assert get_model_class(Namespace(model='rainbow_prompt')) is RainbowPrompt
    assert get_backbone_class('rainbow-vit', return_args=True)[0] is not None
    defaults = RainbowPrompt.get_parser(ArgumentParser()).parse_args([])
    config = yaml.safe_load((ROOT / 'models/config/rainbow_prompt.yaml').read_text())['default']
    assert 'dataset_config' not in config
    assert (defaults.n_epochs, defaults.rainbow_warmup_epochs, defaults.batch_size) == (20, 5, 64)
    assert defaults.rainbow_prompt_length == 20 and defaults.rainbow_d1 == 56 and defaults.rainbow_d2 == 96
    monkeypatch.setattr('sys.argv', ['main.py', '--model', 'rainbow-prompt', '--model_config', 'default',
                                  '--dataset', 'seq-cifar100-224', '--dataset_config', 'l2p',
                                  '--n_epochs', '1', '--rainbow_warmup_epochs', '0', '--batch_size', '8',
                                  '--rainbow_vit_size', 'tiny'])
    args = parse_args()
    assert (args.dataset_config, args.n_epochs, args.rainbow_warmup_epochs, args.batch_size) == ('l2p', 1, 0, 8)
    assert args.backbone == 'rainbow-vit' and args.rainbow_vit_size == 'tiny'


def test_native_wrappers_freezing_optimizer_and_two_task_masks(monkeypatch):
    model, dataset = make_model(monkeypatch)
    model.train()
    assert not model.net.query.training
    frozen = {name: value.detach().clone() for name, value in model.net.named_parameters() if not value.requires_grad}
    assert model.args.lr == .03
    assert model.opt.param_groups[0]['lr'] == .0075
    assert model.opt.defaults['eps'] == 1e-8 and model.opt.defaults['betas'] == (.9, .999)
    model.meta_begin_task(dataset)
    assert math.isfinite(step(model, epoch=0))
    assert not bool(model.net.rainbow_ready[0])
    with pytest.raises(RuntimeError, match='cached'):
        model(samples()[0])
    assert isinstance(step(model), float)
    cache = model.net.prompt.rainbow_prompt.stored_rainbow_prompts.clone()
    assert bool(model.net.rainbow_ready[0])
    model.meta_end_task(dataset)
    torch.testing.assert_close(model.net.prompt.rainbow_prompt.stored_rainbow_prompts, cache, rtol=0, atol=0)
    assert model(samples()[0]).shape == (2, 2)
    old_optimizer = model.opt
    old_pool = {layer: getattr(model.net.prompt.rainbow_prompt, f'base_knowledge_{layer}')[0].detach().clone()
                for layer in (0, 1)}
    old_key = model.net.prompt.rainbow_prompt.base_key[0].detach().clone()
    model.meta_begin_task(dataset)
    assert model.opt is not old_optimizer and not model.opt.state and model.opt.param_groups[0]['lr'] == .0075
    assert math.isfinite(step(model, offset=2))
    model.meta_end_task(dataset)
    assert model(samples()[0]).shape == (2, 4)
    for layer in (0, 1):
        torch.testing.assert_close(getattr(model.net.prompt.rainbow_prompt, f'base_knowledge_{layer}')[0], old_pool[layer], rtol=0, atol=0)
    torch.testing.assert_close(model.net.prompt.rainbow_prompt.base_key[0], old_key, rtol=0, atol=0)
    for name, value in model.net.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
            assert value.grad is None
    assert all(value.requires_grad for value in model.net.prompt.rainbow_prompt.query_matcher.parameters())
    assert model.task_iteration == 1


def test_current_classes_ce_and_future_classifier_not_trained(monkeypatch):
    model, dataset = make_model(monkeypatch, rainbow_warmup_epochs=0, n_epochs=1)
    model.meta_begin_task(dataset)
    future = model.net.prompt.head.weight[2:].detach().clone()
    step(model, epoch=0)
    torch.testing.assert_close(model.net.prompt.head.weight[2:], future, rtol=0, atol=0)
    assert bool((model.net.prompt.head.weight.grad[2:] == 0).all())
    model.meta_end_task(dataset)
    model.meta_begin_task(dataset)
    with pytest.raises(ValueError, match='current-task'):
        step(model, epoch=0)


def test_source_ce_similarity_loss_backward_setting_and_partial_label_wrapper(monkeypatch):
    model, dataset = make_model(monkeypatch, rainbow_warmup_epochs=0, n_epochs=1)
    model.meta_begin_task(dataset)
    step(model, epoch=0)
    model.meta_end_task(dataset)
    model.meta_begin_task(dataset)
    observed = {}
    original_outputs = model.net.outputs
    def capture_outputs(*args, **kwargs):
        result = original_outputs(*args, **kwargs)
        observed['output'] = {name: result[name].detach().clone() for name in ('logits', 'sim_loss')}
        return result
    original_backward = torch.Tensor.backward
    def capture_backward(self, *args, **kwargs):
        observed['retain_graph'] = kwargs.get('retain_graph')
        return original_backward(self, *args, **kwargs)
    monkeypatch.setattr(model.net, 'outputs', capture_outputs)
    monkeypatch.setattr(torch.Tensor, 'backward', capture_backward)
    images = torch.randn(3, 3, 8, 8)
    labels = torch.tensor([2, -1, 3])
    actual = model.meta_observe(images, labels, images, epoch=0)
    logits = observed['output']['logits']
    assert logits.shape == (2, 4)  # meta_observe removes the unlabeled example.
    logits[:, :2] = -float('inf')
    expected = nn.functional.cross_entropy(logits, torch.tensor([2, 3])) - .01 * observed['output']['sim_loss']
    assert actual == pytest.approx(expected.item(), abs=1e-7)
    assert observed['retain_graph'] is True


def test_net_checkpoint_roundtrip_unknown_task_routing_and_return_types(monkeypatch):
    model, dataset = make_model(monkeypatch, rainbow_warmup_epochs=0, n_epochs=1)
    model.meta_begin_task(dataset)
    step(model, epoch=0)
    model.meta_end_task(dataset)
    model.eval()
    images = samples()[0]
    expected = model(images)
    restored, _ = make_model(monkeypatch, rainbow_warmup_epochs=0, n_epochs=1)
    restored.net.load_state_dict(copy.deepcopy(model.net.state_dict()))
    restored.eval()
    torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
    assert restored.net.get_extra_state()['learned_gate_active'] is False
    assert restored.net.get_extra_state()['paper_equivalent'] is False
    for returnt in ('out', 'features', 'both', 'full', 'all'):
        result = restored.net(images, returnt=returnt)
        assert isinstance(result, dict if returnt in ('full', 'all') else tuple if returnt == 'both' else torch.Tensor)
    with pytest.raises(ValueError, match='return type'):
        restored.net(images, returnt='bad')


def test_full_model_checkpoint_resumes_wrapper_task_counters(monkeypatch):
    model, dataset = make_model(monkeypatch, rainbow_warmup_epochs=0, n_epochs=1)
    model.meta_begin_task(dataset)
    step(model, epoch=0)
    model.meta_end_task(dataset)
    restored, restored_dataset = make_model(monkeypatch, rainbow_warmup_epochs=0, n_epochs=1)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    assert restored.current_task == 1 and restored.n_seen_classes == 2
    restored.meta_begin_task(restored_dataset)
    assert restored.n_past_classes == 2 and restored.n_seen_classes == 4
    assert math.isfinite(step(restored, offset=2, epoch=0))


@pytest.mark.parametrize('change', ['future', 'offsets', 'ready', 'protocol'])
def test_checkpoint_rejects_invalid_task_state(monkeypatch, change):
    model, dataset = make_model(monkeypatch)
    model.meta_begin_task(dataset)
    state = copy.deepcopy(model.net.state_dict())
    if change == 'future':
        state['active_task'].fill_(2)
    elif change == 'offsets':
        state['task_offsets'][0, 0] = 1
    elif change == 'ready':
        state['rainbow_ready'][1] = True
    else:
        state['_extra_state']['learned_gate_active'] = True
    with pytest.raises(ValueError):
        model.net.load_state_dict(state)


@pytest.mark.parametrize('overrides', [dict(n_epochs=1), dict(optimizer='sgd'), dict(optim_wd=.01),
                                      dict(eval_epochs=1), dict(enable_other_metrics=True),
                                      dict(lr_scheduler='cosine'), dict(lr=float('nan')),
                                      dict(rainbow_clip_grad=0), dict(rainbow_balance=-1),
                                      dict(start_from=1), dict(loadcheck='cached.pt'), dict(inference_only=True)])
def test_invalid_protocol_rejected(monkeypatch, overrides):
    with pytest.raises(ValueError):
        make_model(monkeypatch, **overrides)


@pytest.mark.parametrize('size', ['base', 'small', 'tiny'])
def test_factory_sizes_share_explicit_checkpoint_without_downloads(monkeypatch, tmp_path, size):
    import models.rainbow_prompt_utils.network as network_module
    calls = []
    class Stub:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def load_pretrained(self, checkpoint):
            calls.append(checkpoint)
    monkeypatch.setattr(network_module, 'RainbowVisionTransformer', Stub)
    monkeypatch.setattr(network_module, 'RainbowNetwork', lambda *a: SimpleNamespace())
    checkpoint = tmp_path / 'existing.npz'
    checkpoint.write_bytes(b'fixture')
    args = args_for_test(rainbow_vit_size=size, rainbow_checkpoint=str(checkpoint))
    result = build_rainbow_network(args, 100, 10)
    assert (calls[0]['embed_dim'], calls[0]['num_heads']) == RAINBOW_VIT_SIZES[size]
    assert calls[2] == calls[3] == str(checkpoint)
    assert result.pretrained_checkpoint == str(checkpoint)
    checkpoint.unlink()
    with pytest.raises(ValueError, match='no download'):
        build_rainbow_network(args, 100, 10)
