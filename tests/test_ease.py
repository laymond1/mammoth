"""EASE wiring/lifecycle checks without image inference or optimization."""

import ast
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from models.ease import Ease
from models.ease_utils.network import EaseNetwork
from datasets.utils.continual_dataset import MammothDatasetWrapper


ROOT = Path(__file__).resolve().parents[1]


def test_model_discovery_config_and_tiny_small_choices():
    from models import get_model_class
    from backbone import get_backbone_class
    assert get_model_class(Namespace(model='ease')) is Ease
    assert get_backbone_class('ease-vit', return_args=True)[0] is not None
    parser = Ease.get_parser(ArgumentParser())
    args = parser.parse_args([])
    assert (args.vit_size, args.ease_pretrain_type, args.adapter_size) == ('base', 'in21k_ft_in1k', 64)
    assert parser.parse_args(['--vit_size', 'tiny', '--adapter_size', '16']).adapter_size == 16
    with pytest.raises(SystemExit):
        parser.parse_args(['--vit_size', 'large'])
    defaults = yaml.safe_load((ROOT / 'models/config/ease.yaml').read_text())['default']
    assert defaults['backbone'] == 'ease-vit'
    assert defaults['ease_pretrain_type'] == 'in21k_ft_in1k'
    config = yaml.safe_load((ROOT / 'datasets/configs/seq-cifar100-224/ease.yaml').read_text())
    assert (config['N_TASKS'], config['N_CLASSES_PER_TASK']) == (10, 10)
    assert config['TEST_TRANSFORM'][0]['Resize']['size'] == 256


@pytest.mark.parametrize('size,bottleneck', [('base', 64), ('small', 64), ('tiny', 64),
                                            ('small', 32), ('tiny', 16)])
def test_native_config_and_cli_precedence_without_initialization(monkeypatch, size, bottleneck):
    import sys
    from main import parse_args
    command = ['--model', 'ease', '--dataset', 'seq-cifar100-224', '--model_config', 'default',
               '--vit_size', size, '--adapter_size', str(bottleneck), '--n_epochs', '1',
               '--batch_size', '64', '--stop_after', '1', '--device', 'cpu',
               '--ease_checkpoint', '/tmp/selected-base.npz']
    monkeypatch.setattr(sys, 'argv', ['main.py', *command])
    args = parse_args(cmd=command)
    assert args.backbone == 'ease-vit'
    assert (args.vit_size, args.adapter_size) == (size, bottleneck)
    assert args.ease_pretrain_type == 'in21k_ft_in1k'
    assert args.dataset_config == 'ease'
    assert args.pretrained is True
    assert args.ease_checkpoint == '/tmp/selected-base.npz'
    assert (args.n_epochs, args.batch_size, args.stop_after) == (1, 64, 1)
    assert args.lr_scheduler == 'cosine' and args.enable_other_metrics is False


class Records(Dataset):
    """Synthetic feature records, with no images or neural feature extraction."""
    def __init__(self):
        self.data = torch.tensor([[2., 0.], [4., 0.], [0., 6.]])
        self.targets = torch.tensor([0, 0, 1])
        self.transform = 'original'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.data[index]


def test_prototype_view_preserves_wrapper_and_covers_dropped_training_tail():
    records = Records()
    wrapped = MammothDatasetWrapper(records, train=True)
    training = DataLoader(wrapped, batch_size=2, shuffle=True, drop_last=True)
    dataset = SimpleNamespace(train_loader=training, TEST_TRANSFORM='deterministic')
    subject = SimpleNamespace(args=SimpleNamespace(batch_size=2, num_workers=0))
    loader = Ease._prototype_loader(subject, dataset)
    assert loader.drop_last is False
    assert len(loader.dataset) == 3
    assert isinstance(loader.dataset, MammothDatasetWrapper)
    assert loader.dataset.dataset is not records
    assert loader.dataset.data is records.data
    assert loader.dataset.transform == 'deterministic'
    assert records.transform == 'original'
    assert loader.dataset.indexes is wrapped.indexes


def test_end_task_averages_raw_training_features_and_counts_every_sample():
    records = Records()
    finalize = Mock()
    net = SimpleNamespace(eval=Mock(), classifier=SimpleNamespace(in_features=2),
                          backbone=SimpleNamespace(extract_all=lambda features: features),
                          finalize_task=finalize)
    subject = SimpleNamespace(net=net, device=torch.device('cpu'), n_seen_classes=2, n_past_classes=0,
                              _prototype_loader=lambda dataset: DataLoader(records, batch_size=2))
    Ease.end_task(subject, SimpleNamespace())
    means, past = finalize.call_args.args
    assert past == 0
    torch.testing.assert_close(means, torch.tensor([[3., 0.], [0., 6.]]))
    # Only a toy identity feature callback ran; no ViT or optimizer is involved.
    finalize.assert_called_once()


def test_missing_or_old_class_labels_stop_prototype_finalization():
    net = SimpleNamespace(eval=Mock(), classifier=SimpleNamespace(in_features=2),
                          backbone=SimpleNamespace(extract_all=lambda features: features), finalize_task=Mock())
    subject = SimpleNamespace(net=net, device=torch.device('cpu'), n_seen_classes=3, n_past_classes=0,
                              _prototype_loader=lambda dataset: DataLoader(Records(), batch_size=2))
    with pytest.raises(ValueError, match='Every new class'):
        Ease.end_task(subject, SimpleNamespace())
    net.finalize_task.assert_not_called()
    subject.n_past_classes = 1
    with pytest.raises(ValueError, match='current task'):
        Ease.end_task(subject, SimpleNamespace())


class ToyBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.cls_token = nn.Parameter(torch.zeros(1, 1, 2), requires_grad=False)
        self.feature_dim = 2
        self.head = nn.Identity()
        self.num_adapters = 0
        self.freeze_current = Mock()

    def start_task(self):
        self.num_adapters += 1
        return self.num_adapters - 1


def test_two_task_head_lifecycle_no_future_adapter_and_network_reload():
    network = EaseNetwork(ToyBank())
    network.begin_task(0, 2)
    first_head = network.proxy_head
    network.finalize_task(torch.tensor([[1., 0.], [0., 1.]]), 0)
    assert network.task_finalized.item() is True
    assert network.backbone.num_adapters == 1
    assert all(not p.requires_grad for p in first_head.parameters())
    network.begin_task(2, 3)
    assert network.proxy_head.out_features == 1
    assert network.proxy_head is not first_head
    assert all(p.requires_grad for p in network.proxy_head.parameters())
    assert network.classifier.weight.shape == (3, 4)
    network.finalize_task(torch.tensor([[1., 1., 2., 1.]]), 2)
    restored = EaseNetwork(ToyBank())
    restored.load_state_dict(network.state_dict())
    assert restored.task_finalized.item() is True
    assert restored.classifier.task_offsets == [(0, 2), (2, 3)]
    assert restored.proxy_head.out_features == 1
    torch.testing.assert_close(restored.classifier.weight, network.classifier.weight)


def test_invalid_task_transitions_preserve_existing_state():
    network = EaseNetwork(ToyBank())
    with pytest.raises(RuntimeError, match='active, unfinalized'):
        network.finalize_task(torch.empty(0, 0), 0)
    network.begin_task(0, 2)
    with pytest.raises(RuntimeError, match='Finalize the current'):
        network.begin_task(2, 3)
    assert network.backbone.num_adapters == 1
    network.finalize_task(torch.tensor([[1., 0.], [0., 1.]]), 0)
    saved = network.classifier.weight.clone()
    with pytest.raises(RuntimeError, match='active, unfinalized'):
        network.finalize_task(torch.tensor([[2., 0.], [0., 2.]]), 0)
    torch.testing.assert_close(network.classifier.weight, saved)


def test_native_hook_and_observe_contract():
    tree = ast.parse((ROOT / 'models/ease.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in methods['observe'].args.args] == ['self', 'inputs', 'labels', 'not_aug_inputs', 'epoch']
    assert 'meta_begin_task' not in methods and 'meta_end_task' not in methods
    assert 'meta_observe' not in methods
    assert Ease.COMPATIBILITY == ['class-il', 'task-il']


@pytest.mark.parametrize('eval_epochs', [-1, 0, 1])
def test_pre_finalization_epoch_evaluation_is_rejected_without_constructing_vit(eval_epochs):
    from backbone.ease_vit import EaseViT
    backbone = EaseViT.__new__(EaseViT)
    nn.Module.__init__(backbone)
    backbone.pretrained_loaded = True
    args = Namespace(ease_alpha=0.1, fitting_mode='epochs', eval_epochs=eval_epochs)
    with pytest.raises(ValueError, match='without per-epoch evaluation'):
        Ease(backbone, None, args, None)
