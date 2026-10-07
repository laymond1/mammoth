"""Native HiDe lifecycle/parser checks with tiny synthetic feature branches.

No CIFAR images, pretrained transformer inference, downloads or GPU execution.
Toy feature losses check optimization contracts without a CL experiment.
"""

import ast
from argparse import ArgumentParser, Namespace
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from datasets.utils.continual_dataset import MammothDatasetWrapper
from models.hide_prompt import HidePrompt
from models.hide_prompt_utils.network import HideNetwork
from models.hide_prompt_utils.prompt import EPrompt


ROOT = Path(__file__).resolve().parents[1]


def args_for_test(**overrides):
    args = HidePrompt.get_parser(ArgumentParser()).parse_args([])
    for name, value in dict(hide_tii_epochs=1, hide_prompt_epochs=1, hide_ca_epochs=1,
                            hide_n_centroids=1, n_epochs=2, num_workers=0, seed=1993,
                            fitting_mode='epochs', eval_epochs=None, label_perc=1., label_perc_by_class=1.,
                            optim_mom=0.9, optim_nesterov=False, device='cpu', nowand=1,
                            debug_mode=False, non_verbose=True).items():
        setattr(args, name, value)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


class Features(Dataset):
    def __init__(self, offset=0):
        self.data = torch.tensor([[1., 0.], [1.2, .1], [0., 1.], [.1, 1.2]])
        self.targets = torch.tensor([0, 0, 1, 1]) + offset
        self.transform = 'original'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.data[index]


class ToyBranch(nn.Module):
    def __init__(self, prompted=False):
        super().__init__()
        self.num_features, self.num_classes = 2, 4
        self.mlp = nn.Identity() if prompted else nn.Linear(2, 2)
        self.head = nn.Linear(2, 4)
        self.e_prompt = EPrompt(1, 2, 2, 1, 1) if prompted else None
        self.prompt_layers = (0,)

    def forward(self, inputs, task_id=-1, prompt_id=None, train=False, fc_only=False, prompt_momentum=0.):
        features = inputs
        if not fc_only and self.e_prompt is not None:
            indices = torch.full((len(inputs), 1), task_id, dtype=torch.long) if train else prompt_id
            prompt = self.e_prompt.prompt[:, :, indices].mean((0, 1, 3, 4, 5, 6))
            features = features + prompt[:, None]
        return {'pre_logits': features, 'logits': self.head(self.mlp(features)), 'prompt_idx': prompt_id}


def make_model(monkeypatch, **overrides):
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    network = HideNetwork(ToyBranch(), ToyBranch(True), 2)
    network.pretrained_checkpoint = '/tmp/toy-feature-fixture.npz'
    dataset = SimpleNamespace(N_CLASSES=4, N_TASKS=2, N_CLASSES_PER_TASK=2, SETTING='class-il',
                              get_offsets=lambda task: (task * 2, (task + 1) * 2),
                              TEST_TRANSFORM='deterministic',
                              get_normalization_transform=lambda: nn.Identity())
    dataset.train_loader = DataLoader(MammothDatasetWrapper(Features(), train=True), batch_size=2)
    subject = HidePrompt(network, nn.CrossEntropyLoss(), args_for_test(**overrides), None, dataset)
    subject.to(torch.device('cpu'))
    return subject, dataset


def test_discovery_registration_and_config_without_initialization():
    from models import get_model_class
    from backbone import get_backbone_class
    assert get_model_class(Namespace(model='hide-prompt')) is HidePrompt
    assert get_backbone_class('hide-vit', return_args=True)[0] is not None
    args = HidePrompt.get_parser(ArgumentParser()).parse_args([])
    assert (args.hide_vit_size, args.hide_tii_epochs, args.hide_prompt_epochs, args.n_epochs) == ('base', 20, 50, 70)
    assert args.hide_prompt_length == 5 and args.hide_reg == 0.1
    assert args.optim_wd == 0 and not args.hide_use_permute_fix
    with pytest.raises(SystemExit):
        HidePrompt.get_parser(ArgumentParser()).parse_args(['--hide_vit_size', 'large'])
    config = yaml.safe_load((ROOT / 'models/config/hide_prompt.yaml').read_text())['default']
    assert config['backbone'] == 'hide-vit' and 'dataset_config' not in config


@pytest.mark.parametrize('configuration', ['hide_prompt', 'ease'])
def test_native_cli_precedence_and_phase_epoch_counts(monkeypatch, configuration):
    from main import parse_args
    import sys
    command = ['--model', 'hide-prompt', '--dataset', 'seq-cifar100-224', '--model_config', 'default',
               '--dataset_config', configuration, '--hide_checkpoint', '/tmp/local-base.npz',
               '--hide_tii_epochs', '1', '--hide_prompt_epochs', '1', '--n_epochs', '2',
               '--batch_size', '64', '--stop_after', '1', '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', ['main.py', *command])
    args = parse_args(cmd=command)
    assert args.dataset_config == configuration
    assert (args.hide_tii_epochs, args.hide_prompt_epochs, args.n_epochs) == (1, 1, 2)
    assert args.hide_checkpoint == '/tmp/local-base.npz' and args.backbone == 'hide-vit'
    assert args.hide_prompt_length == 5 and args.optim_wd == 0.
    assert args.lr_scheduler is None and args.enable_other_metrics is False


@pytest.mark.parametrize('overrides,match', [
    ({'n_epochs': 1}, 'n_epochs must equal'), ({'hide_tii_epochs': 0}, 'positive'),
    ({'hide_prompt_epochs': 0}, 'positive'), ({'hide_ca_epochs': 0}, 'positive'),
    ({'hide_n_centroids': 0}, 'positive'), ({'hide_tii_lr': float('nan')}, 'positive'),
    ({'hide_reg': -1}, 'nonnegative'), ({'lr_scheduler': 'cosine'}, 'separate'),
    ({'eval_epochs': 1}, 'task-end'), ({'fitting_mode': 'iters'}, 'epoch fitting'),
    ({'enable_other_metrics': True}, 'task-end'), ({'label_perc': .5}, 'fully labeled'),
])
def test_invalid_protocol_is_rejected_before_image_inference(overrides, match):
    with pytest.raises(ValueError, match=match):
        HidePrompt._validate_protocol(args_for_test(**overrides))


def test_statistics_view_visits_tail_and_preserves_native_wrapper():
    wrapped = MammothDatasetWrapper(Features(), train=True)
    source = DataLoader(wrapped, batch_size=3, drop_last=True)
    dataset = SimpleNamespace(train_loader=source, TEST_TRANSFORM='deterministic')
    subject = SimpleNamespace(args=SimpleNamespace(batch_size=2, num_workers=0))
    loader = HidePrompt._statistics_loader(subject, dataset)
    assert loader.drop_last is False and len(loader.dataset) == 4
    assert isinstance(loader.dataset, MammothDatasetWrapper)
    assert loader.dataset.dataset is not wrapped.dataset
    assert loader.dataset.data is wrapped.data and loader.dataset.indexes is wrapped.indexes
    assert loader.dataset.transform == 'deterministic' and wrapped.transform == 'original'


def test_two_phases_optimizer_selection_and_current_class_mask(monkeypatch):
    subject, dataset = make_model(monkeypatch)
    subject.meta_begin_task(dataset)
    subject.meta_begin_epoch(0, dataset)
    assert subject._phase == 'tii'
    assert subject.opt.param_groups[0]['lr'] == pytest.approx(.0005 * 48 / 256)
    saved_prompt = subject.net.prompt.e_prompt.prompt.detach().clone()
    subject.observe(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]), None, epoch=0)
    torch.testing.assert_close(saved_prompt, subject.net.prompt.e_prompt.prompt)
    subject.meta_end_epoch(0, dataset)
    assert subject.tii_ready and not subject.prompt_ready
    subject.meta_begin_epoch(1, dataset)
    assert subject._phase == 'prompt'
    rates = [group['lr'] for group in subject.opt.param_groups]
    assert rates == pytest.approx([.03 * 48 / 256, .03 * 48 / 256 * .1])
    saved_tii = copy.deepcopy(subject.net.tii.state_dict())
    subject.observe(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]), None, epoch=1)
    for name, value in saved_tii.items():
        torch.testing.assert_close(value, subject.net.tii.state_dict()[name])
    subject.meta_end_epoch(1, dataset)
    subject.meta_end_task(dataset)
    assert subject.current_task == 1 and subject.prompt_ready
    assert [row['skipped'] for row in subject.alignment_history] == [True, True]
    assert subject.forward(torch.ones(2, 2)).shape == (2, 2)


def test_second_task_preserves_old_prompt_and_distributions_and_reloads(monkeypatch):
    subject, dataset = make_model(monkeypatch)
    subject.meta_begin_task(dataset)
    for epoch in range(2):
        subject.meta_begin_epoch(epoch, dataset)
        subject.observe(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]), None, epoch=epoch)
        subject.meta_end_epoch(epoch, dataset)
    subject.meta_end_task(dataset)
    old_prompt = subject.net.prompt.e_prompt.prompt[:, :, 0].detach().clone()
    old_stats = subject.prompt_statistics.means[:2].clone()
    dataset.train_loader = DataLoader(MammothDatasetWrapper(Features(2), train=True), batch_size=2)
    subject.meta_begin_task(dataset)
    for epoch in range(2):
        subject.meta_begin_epoch(epoch, dataset)
        subject.observe(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([2, 3]), None, epoch=epoch)
        subject.meta_end_epoch(epoch, dataset)
    subject.meta_end_task(dataset)
    torch.testing.assert_close(old_prompt, subject.net.prompt.e_prompt.prompt[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(old_stats, subject.prompt_statistics.means[:2], rtol=0, atol=0)
    assert subject.alignment_history[-2]['optimizer_steps'] == 2
    assert subject.alignment_history[-1]['optimizer_steps'] == 2
    assert subject.current_task == 2 and subject.net.class_to_task.tolist() == [0, 0, 1, 1]
    restored, _ = make_model(monkeypatch)
    restored.load_state_dict(subject.state_dict())
    assert restored.current_task == 2 and restored._phase == 'eval'
    assert restored.prompt_ready and restored.tii_ready
    for name, value in subject.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(subject.forward(torch.ones(2, 2)), restored.forward(torch.ones(2, 2)))


def test_stats_uses_pre_mlp_tii_features_and_finalized_prompt_without_momentum(monkeypatch):
    subject, dataset = make_model(monkeypatch)
    subject.meta_begin_task(dataset)
    subject._collect_statistics(dataset, 'tii')
    torch.testing.assert_close(subject.tii_statistics.means[0, 0], torch.tensor([1.1, .05], dtype=torch.float64))
    spy = Mock(wraps=subject.net.prompt_outputs)
    subject.net.prompt_outputs = spy
    subject._collect_statistics(dataset, 'prompt')
    assert all(call.kwargs['train'] is True and call.kwargs['prompt_momentum'] == 0. for call in spy.call_args_list)


def test_statistics_batches_each_class_before_feature_extraction(monkeypatch):
    subject, dataset = make_model(monkeypatch)
    dataset.train_loader.dataset.targets = torch.tensor([0, 1, 0, 1])
    subject.meta_begin_task(dataset)
    spy = Mock(wraps=subject.net.prompt_outputs)
    subject.net.prompt_outputs = spy
    subject._collect_statistics(dataset, 'prompt')
    expected = dataset.train_loader.dataset.data
    assert len(spy.call_args_list) == 2
    torch.testing.assert_close(spy.call_args_list[0].args[0], expected[[0, 2]])
    torch.testing.assert_close(spy.call_args_list[1].args[0], expected[[1, 3]])


def test_statistics_follow_device_selected_by_training_loop(monkeypatch):
    subject, dataset = make_model(monkeypatch)
    tii_move = Mock(wraps=subject.tii_statistics.to)
    prompt_move = Mock(wraps=subject.prompt_statistics.to)
    subject.tii_statistics.to = tii_move
    subject.prompt_statistics.to = prompt_move
    subject.meta_begin_task(dataset)
    tii_move.assert_called_once_with(subject.device)
    prompt_move.assert_called_once_with(subject.device)


def test_evaluation_and_phase_transitions_are_guarded(monkeypatch):
    subject, dataset = make_model(monkeypatch)
    subject.meta_begin_task(dataset)
    with pytest.raises(RuntimeError, match='evaluates after'):
        subject.forward(torch.ones(1, 2))
    with pytest.raises(RuntimeError, match='Finish current-task TII'):
        subject.meta_begin_epoch(1, dataset)
    with pytest.raises(RuntimeError, match='both full training phases'):
        subject.end_task(dataset)
    subject.meta_begin_epoch(0, dataset)
    with pytest.raises(RuntimeError, match='at least one optimizer update'):
        subject.meta_end_epoch(0, dataset)
    with pytest.raises(ValueError, match='current-task classes'):
        subject.observe(torch.ones(1, 2), torch.tensor([2]), None, epoch=0)
    with pytest.raises(ValueError, match='outside its two-stage'):
        subject.begin_epoch(2, dataset)


def test_step_schedule_matches_author_after_epoch_step_order(monkeypatch):
    subject, dataset = make_model(monkeypatch, hide_prompt_epochs=50, n_epochs=51)
    subject.meta_begin_task(dataset)
    subject.tii_ready.fill_(True)
    full_rate = .03 * 48 / 256
    for local, expected in ((0, full_rate), (29, full_rate), (30, full_rate), (31, full_rate * .1)):
        subject.begin_epoch(local + 1, dataset)
        assert subject.opt.param_groups[0]['lr'] == pytest.approx(expected)
        assert subject.opt.param_groups[1]['lr'] == pytest.approx(expected * .1)


def test_native_hooks_are_preserved_and_no_task_id_is_given_to_forward():
    tree = ast.parse((ROOT / 'models/hide_prompt.py').read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef))
    methods = {item.name: item for item in cls.body if isinstance(item, ast.FunctionDef)}
    assert [arg.arg for arg in methods['observe'].args.args] == ['self', 'inputs', 'labels', 'not_aug_inputs', 'epoch']
    assert [arg.arg for arg in methods['forward'].args.args] == ['self', 'inputs']
    assert not {'meta_begin_task', 'meta_observe', 'meta_end_task', 'meta_begin_epoch', 'meta_end_epoch'} & methods.keys()
