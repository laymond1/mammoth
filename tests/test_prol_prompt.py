"""PROL framework lifecycle checks on synthetic 8x8 tensors, CPU only.

These exercise the native Mammoth hooks and trainer, not a CIFAR experiment.
No checkpoint download, GPU computation or external logging is performed.
"""

from argparse import ArgumentParser, Namespace
import ast
import copy
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.cl_runtime import configure_native_environment

configure_native_environment()

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
import yaml

from models.prol_prompt import ProlPrompt
from models.prol_prompt_utils.network import ProlNetwork
from models.prol_prompt_utils.vit import ProlVisionTransformer


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_fixture(monkeypatch):
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def args_for_test(**overrides):
    args = ProlPrompt.get_parser(ArgumentParser()).parse_args([])
    for name, value in dict(n_epochs=1, batch_size=64, fitting_mode='epochs', eval_epochs=None,
                            label_perc=1., label_perc_by_class=1., nowand=1, device='cpu',
                            optim_mom=.9, optim_nesterov=False, num_workers=0,
                            debug_mode=False, non_verbose=True, seed=1993).items():
        setattr(args, name, value)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def toy_network():
    # Keep the source query's five task heads, even though its head is unused.
    common = dict(num_classes=20, img_size=8, patch_size=4, embed_dim=8,
                  depth=2, num_heads=2)
    query = ProlVisionTransformer(**common, n_tasks=5, prompted=False)
    prompt = ProlVisionTransformer(**common, n_tasks=2, prompted=True,
                                  prompt_length=5, prompt_layers=(0, 1))
    result = ProlNetwork(query, prompt, 2)
    result.pretrained_checkpoint = '/tmp/prol-synthetic-fixture.npz'
    return result


def make_model(**overrides):
    dataset = SimpleNamespace(N_CLASSES=20, N_TASKS=2, N_CLASSES_PER_TASK=10,
                              SETTING='class-il', get_offsets=lambda task: (task * 10, (task + 1) * 10))
    subject = ProlPrompt(toy_network(), nn.CrossEntropyLoss(), args_for_test(**overrides), None, dataset)
    subject.to(torch.device('cpu'))
    return subject, dataset


def images_and_labels(offset=0, count=4):
    generator = torch.Generator().manual_seed(1993 + offset)
    return torch.randn(count, 3, 8, 8, generator=generator), torch.arange(count) % 10 + offset


def complete_task(subject, dataset, offset=0):
    subject.meta_begin_task(dataset)
    subject.meta_begin_epoch(0, dataset)
    images, labels = images_and_labels(offset)
    loss = subject.meta_observe(images, labels, images, epoch=0)
    assert isinstance(loss, float) and math.isfinite(loss)
    subject.meta_end_epoch(0, dataset)
    subject.meta_end_task(dataset)
    return loss


def test_discovery_parser_config_and_cli_precedence(monkeypatch):
    from models import get_model_class
    from backbone import get_backbone_class
    from main import parse_args

    assert get_model_class(Namespace(model='prol_prompt')) is ProlPrompt
    assert get_model_class(Namespace(model='prol-prompt')) is ProlPrompt
    assert get_backbone_class('prol-vit', return_args=True)[0] is not None
    defaults = args_for_test()
    config = yaml.safe_load((ROOT / 'models/config/prol_prompt.yaml').read_text())['default']
    assert 'dataset_config' not in config
    assert defaults.n_epochs == 1 and defaults.batch_size == 64
    assert defaults.lr == .01 and defaults.prol_prompt_length == 5
    assert tuple(defaults.prol_prompt_layers) == tuple(range(7))
    assert defaults.prol_top_k_l == 1
    command = ['--model', 'prol-prompt', '--model_config', 'default', '--dataset', 'seq-cifar100-224',
               '--dataset_config', 'l2p', '--batch_size', '16', '--lr', '.05',
               '--prol_checkpoint', '/tmp/explicit-local.npz']
    monkeypatch.setattr('sys.argv', ['main.py', *command])
    args = parse_args(cmd=command)
    assert (args.dataset_config, args.batch_size, args.lr) == ('l2p', 16, .05)
    assert args.prol_checkpoint == '/tmp/explicit-local.npz' and args.backbone == 'prol-vit'


@pytest.mark.parametrize('overrides', [
    {'n_epochs': 2}, {'optimizer': 'sgd'}, {'optim_wd': .1}, {'eval_epochs': 1},
    {'enable_other_metrics': True}, {'lr_scheduler': 'cosine'}, {'fitting_mode': 'iters'},
    {'label_perc': .5}, {'label_perc_by_class': .5}, {'lr': float('nan')},
    {'lr': 0}, {'start_from': 1}, {'loadcheck': 'cached.pt'}, {'inference_only': True},
])
def test_invalid_protocol_is_rejected_before_training(overrides):
    with pytest.raises(ValueError):
        make_model(**overrides)


def test_native_lifecycle_trains_allowed_norms_and_freezes_old_head_and_generator():
    subject, dataset = make_model()
    frozen = {name: parameter.detach().clone() for name, parameter in subject.net.named_parameters()
              if not parameter.requires_grad}
    norm = subject.net.prompt.blocks[0].norm1.weight.detach().clone()
    assert subject.args.lr == .01  # Preserve nominal LR in configuration.
    complete_task(subject, dataset)
    assert subject.current_task == 1 and subject.n_seen_classes == 10
    assert subject.net.task_updates.tolist() == [1, 0]
    assert not torch.equal(subject.net.prompt.blocks[0].norm1.weight, norm)
    old_head = copy.deepcopy(subject.net.prompt.head.heads[0].state_dict())
    old_generators = {name: value.detach().clone() for name, value in subject.net.prompt.e_prompt.named_parameters()
                      if 'comp_gen' in name}
    old_optimizer = subject.opt
    subject.meta_begin_task(dataset)
    assert subject.opt is not old_optimizer and not subject.opt.state
    assert len(subject.net.prompt.head.heads) == 2
    assert all(not parameter.requires_grad for parameter in subject.net.prompt.head.heads[0].parameters())
    assert all(parameter.requires_grad for parameter in subject.net.prompt.head.heads[1].parameters())
    assert all(not parameter.requires_grad for name, parameter in subject.net.prompt.e_prompt.named_parameters()
               if 'comp_gen' in name)
    assert [group['lr'] for group in subject.opt.param_groups] == pytest.approx([.0025, 1.25e-6])
    subject.meta_begin_epoch(0, dataset)
    images, labels = images_and_labels(10)
    assert math.isfinite(subject.meta_observe(images, labels, images, epoch=0))
    subject.meta_end_epoch(0, dataset)
    subject.meta_end_task(dataset)
    assert subject.net.task_updates.tolist() == [1, 1]
    for name, value in old_head.items():
        torch.testing.assert_close(subject.net.prompt.head.heads[0].state_dict()[name], value, rtol=0, atol=0)
    for name, value in old_generators.items():
        torch.testing.assert_close(dict(subject.net.prompt.e_prompt.named_parameters())[name], value, rtol=0, atol=0)
    for name, parameter in subject.net.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
            assert parameter.grad is None


def test_unlabeled_samples_are_filtered_by_native_meta_observe():
    subject, dataset = make_model()
    subject.meta_begin_task(dataset)
    subject.meta_begin_epoch(0, dataset)
    original = subject.net.outputs
    capture = {}
    def capture_outputs(inputs, labels=None, train=False):
        capture['images'], capture['labels'] = len(inputs), labels.detach().clone()
        return original(inputs, labels=labels, train=train)
    subject.net.outputs = capture_outputs
    images, _ = images_and_labels(count=3)
    loss = subject.meta_observe(images, torch.tensor([0, -1, 1]), images, epoch=0, irrelevant_extra=True)
    assert math.isfinite(loss)
    assert capture['images'] == 2 and capture['labels'].tolist() == [0, 1]
    assert subject.task_iteration == 1 and subject.epoch_iteration == 1


def test_forward_uses_unknown_class_routing_without_task_or_label_oracle():
    subject, dataset = make_model()
    complete_task(subject, dataset)
    subject.eval()
    spy = Mock(wraps=subject.net.prompt.forwardA1)
    subject.net.prompt.forwardA1 = spy
    images, _ = images_and_labels()
    logits = subject(images)
    assert logits.shape == (4, 10)
    assert spy.call_args.args[1] is None
    assert spy.call_args.kwargs['task_id'] == -1
    assert spy.call_args.kwargs['train'] is False
    assert subject.net.query.training is True
    assert not subject.net.prompt.training
    assert list(inspect.signature(ProlPrompt.forward).parameters) == ['self', 'inputs']
    for returnt in ('out', 'features', 'both', 'full', 'all'):
        output = subject.net(images, returnt=returnt)
        assert isinstance(output, dict if returnt in ('full', 'all') else tuple if returnt == 'both' else torch.Tensor)
    with pytest.raises(ValueError, match='return type'):
        subject.net(images, returnt='invalid')


def test_dynamic_checkpoint_round_trip_restores_tasks_freezing_and_rng(tmp_path):
    subject, dataset = make_model()
    complete_task(subject, dataset)
    complete_task(subject, dataset, offset=10)
    subject.eval()
    images, _ = images_and_labels()
    expected = subject(images).detach().clone()  # Evaluation clamp is part of source behavior.
    checkpoint = tmp_path / 'synthetic_prol.pt'
    torch.save(subject.state_dict(), checkpoint)
    restored, _ = make_model()
    rng = torch.random.get_rng_state().clone()
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    restored.eval()
    assert restored.current_task == 2 and restored.n_seen_classes == 20
    assert len(restored.net.prompt.head.heads) == 2
    assert restored.net.task_updates.tolist() == [1, 1]
    assert restored.net.prompt.e_prompt.lmax_list == [10, 20]
    assert restored.net.get_extra_state()['paper_equivalent'] is False
    assert not any(parameter.requires_grad for parameter in restored.net.query.parameters())
    assert not any(parameter.requires_grad for parameter in restored.net.prompt.head.heads[0].parameters())
    assert not any(parameter.requires_grad for name, parameter in restored.net.prompt.e_prompt.named_parameters()
                   if 'comp_gen' in name)
    torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)


def test_task_boundary_checkpoint_preserves_native_wrapper_counters_for_next_task():
    subject, dataset = make_model()
    complete_task(subject, dataset)
    restored, restored_dataset = make_model()
    restored.load_state_dict(copy.deepcopy(subject.state_dict()))
    assert restored.current_task == 1 and restored.n_seen_classes == 10
    complete_task(restored, restored_dataset, offset=10)
    assert restored.current_task == 2 and restored.n_seen_classes == 20


def test_restoring_unlearned_checkpoint_clears_previous_task_state():
    subject, dataset = make_model()
    fresh, _ = make_model()
    fresh_state = copy.deepcopy(fresh.state_dict())
    complete_task(subject, dataset)
    complete_task(subject, dataset, offset=10)
    subject.load_state_dict(fresh_state)
    assert subject.current_task == 0 and subject.n_seen_classes == -1
    assert subject.n_past_classes == -1 and int(subject.net.active_task) == -1
    assert len(subject.net.prompt.head.heads) == 1
    assert subject.net.prompt.e_prompt.lmax_list == []
    assert all(parameter.requires_grad for name, parameter in subject.net.prompt.e_prompt.named_parameters()
               if 'comp_gen' in name)
    complete_task(subject, dataset)
    assert subject.current_task == 1 and subject.n_seen_classes == 10


@pytest.mark.parametrize('change', ['future_task', 'future_updates', 'offsets', 'protocol', 'prompt_count', 'head_count', 'query_head_count'])
def test_checkpoint_rejects_inconsistent_or_changed_protocol(change):
    subject, dataset = make_model()
    complete_task(subject, dataset)
    state = copy.deepcopy(subject.net.state_dict())
    if change == 'future_task':
        state['active_task'].fill_(2)
    elif change == 'future_updates':
        state['task_updates'][1] = 1
    elif change == 'offsets':
        state['task_offsets'][0, 0] = 1
    elif change == 'protocol':
        state['_extra_state']['paper_equivalent'] = True
    elif change == 'prompt_count':
        state['prompt.e_prompt._extra_state']['new_num_k'] = 2
    elif change == 'head_count':
        del state['prompt.head.heads.0.1.weight']
    else:
        for key, value in list(state.items()):
            if key.startswith('query.head.heads.0.'):
                state[key.replace('query.head.heads.0.', 'query.head.heads.1.')] = value.clone()
    with pytest.raises(ValueError):
        subject.net.load_state_dict(state)


@pytest.mark.parametrize('change', ['wrapper_task', 'seen_counter', 'counter_type'])
def test_full_checkpoint_rejects_wrapper_and_network_lifecycle_disagreement(change):
    subject, dataset = make_model()
    complete_task(subject, dataset)
    state = copy.deepcopy(subject.state_dict())
    if change == 'wrapper_task':
        state['_extra_state']['current_task'] = 0
    elif change == 'seen_counter':
        state['_extra_state']['class_counters']['_n_seen_classes'] = 20
    else:
        state['_extra_state']['class_counters']['_n_classes_current_task'] = True
    with pytest.raises(ValueError):
        subject.load_state_dict(state)


def test_framework_meta_hooks_are_inherited():
    tree = ast.parse((ROOT / 'models/prol_prompt.py').read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef))
    methods = {item.name: item for item in cls.body if isinstance(item, ast.FunctionDef)}
    assert not {'meta_begin_task', 'meta_observe', 'meta_end_task', 'meta_begin_epoch', 'meta_end_epoch'} & methods.keys()
    assert [arg.arg for arg in methods['observe'].args.args] == ['self', 'inputs', 'labels', 'not_aug_inputs', 'epoch']


def test_unused_epoch_copy_does_not_inflate_model_parameter_accounting():
    subject, dataset = make_model()
    subject.meta_begin_task(dataset)
    names = set(dict(subject.named_parameters()))
    count = sum(parameter.numel() for parameter in subject.parameters())
    subject.meta_begin_epoch(0, dataset)
    assert set(dict(subject.named_parameters())) == names
    assert sum(parameter.numel() for parameter in subject.parameters()) == count
    assert torch.equal(subject._old_epoch_key, subject.net.prompt.e_prompt.prompt_key)
    assert subject._old_epoch_key.data_ptr() != subject.net.prompt.e_prompt.prompt_key.data_ptr()
    assert '_old_epoch_key' not in subject.state_dict()
    images, labels = images_and_labels()
    subject.meta_observe(images, labels, images, epoch=0)
    subject.meta_end_epoch(0, dataset)
    assert subject._old_epoch_key is None


@pytest.mark.parametrize('size', ['base', 'small', 'tiny'])
def test_factory_retains_source_query_head_shape_and_requires_existing_local_weights(monkeypatch, tmp_path, size):
    import models.prol_prompt_utils.network as module
    calls = []
    class Branch:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def load_pretrained(self, checkpoint):
            calls.append(checkpoint)
    monkeypatch.setattr(module, 'ProlVisionTransformer', Branch)
    monkeypatch.setattr(module, 'ProlNetwork', lambda *args: SimpleNamespace())
    checkpoint = tmp_path / 'already_present.npz'
    checkpoint.write_bytes(b'synthetic factory contract fixture')
    args = args_for_test(prol_vit_size=size, prol_checkpoint=str(checkpoint))
    result = module.build_prol_network(args, 100, 10)
    assert (calls[0]['embed_dim'], calls[0]['num_heads']) == module.PROL_VIT_SIZES[size]
    assert calls[0]['n_tasks'] == 5 and calls[0]['prompted'] is False
    assert calls[1]['n_tasks'] == 10 and calls[1]['prompt_length'] == 5
    assert calls[2] == calls[3] == str(checkpoint)
    assert result.pretrained_checkpoint == str(checkpoint)
    checkpoint.unlink()
    with pytest.raises(ValueError, match='no download'):
        module.build_prol_network(args, 100, 10)


def test_actual_mammoth_trainer_completes_two_synthetic_tasks_and_visits_tails(monkeypatch, tmp_path):
    """Full native loop: 5000 train/1000 test per task, 8x8 images, batch64.

    This tests loop/hook/data-accounting correctness; accuracy is not an
    empirical CL performance result and no pretrained weights are used.
    """
    from main import parse_args
    from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper, store_masked_loaders
    from utils import training
    command = ['--model', 'prol-prompt', '--dataset', 'seq-cifar100-224', '--dataset_config', 'ease',
               '--model_config', 'default', '--n_epochs', '1', '--batch_size', '64', '--stop_after', '2',
               '--device', 'cpu', '--num_workers', '0', '--non_verbose', '1', '--debug_mode', '0']
    monkeypatch.setattr('sys.argv', ['main.py', *command])
    args = parse_args(cmd=command)
    args.nowand, args.results_path, args.permute_classes = 1, str(tmp_path / 'mammoth'), False
    class SyntheticImages(Dataset):
        def __init__(self, train):
            per_class = 500 if train else 100
            self.targets = np.repeat(np.arange(20), per_class)
            rng = np.random.default_rng(1993 + int(train))
            self.data = rng.normal(size=(len(self.targets), 3, 8, 8)).astype(np.float32)
            self.train, self.transform = train, None
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, index):
            image = torch.from_numpy(self.data[index].copy())
            label = int(self.targets[index])
            return (image, label, image.clone()) if self.train else (image, label)
    class SyntheticTasks(ContinualDataset):
        NAME, SETTING = 'synthetic-prol-cpu-contract', 'class-il'
        N_CLASSES, N_TASKS, N_CLASSES_PER_TASK, SIZE = 20, 2, 10, (8, 8)
        AVAIL_SCHEDS = []
        def __init__(self):
            super().__init__(args)
            self.load_count, self.logged = 0, []
        def get_data_loaders(self):
            self.load_count += 1
            return store_masked_loaders(SyntheticImages(True), SyntheticImages(False), self)
        def get_loss(self):
            return nn.CrossEntropyLoss()
        def log(self, args, logger, accuracies, task, setting, **kwargs):
            self.logged.append((task, accuracies))
            return None
    dataset = SyntheticTasks()
    subject = ProlPrompt(toy_network(), nn.CrossEntropyLoss(), args, None, dataset)
    subject.to(torch.device('cpu'))
    train_batches, eval_batches = [], []
    original_observe, original_forward = subject.meta_observe, subject.forward
    def observe(*inputs, **kwargs):
        train_batches.append((subject.current_task, len(inputs[0]), inputs[1].tolist()))
        torch.testing.assert_close(inputs[0], inputs[2], rtol=0, atol=0)
        return original_observe(*inputs, **kwargs)
    def forward(inputs):
        eval_batches.append(len(inputs))
        return original_forward(inputs)
    subject.meta_observe = Mock(side_effect=observe)
    subject.forward = forward
    for hook in ('meta_begin_task', 'meta_end_task', 'meta_begin_epoch', 'meta_end_epoch', 'meta_end_eval'):
        setattr(subject, hook, Mock(wraps=getattr(subject, hook)))
    training.train(subject, dataset, args)
    assert dataset.load_count == 2 and subject.current_task == 2
    assert subject.task_iteration == subject.epoch_iteration == 79
    assert subject.net.task_updates.tolist() == [79, 79]
    assert subject.meta_observe.call_count == 158
    assert all(getattr(subject, hook).call_count == 2 for hook in
               ('meta_begin_task', 'meta_end_task', 'meta_begin_epoch', 'meta_end_epoch', 'meta_end_eval'))
    assert sum(batch[1] for batch in train_batches) == 10000
    assert [batch[1] for batch in train_batches].count(8) == 2
    assert sum(eval_batches) == 3000 and len(eval_batches) == 48
    assert eval_batches.count(40) == 3
    assert all(task * 10 <= label < (task + 1) * 10 for task, _, labels in train_batches for label in labels)
    assert isinstance(dataset.train_loader.dataset, MammothDatasetWrapper)
    assert dataset.train_loader.drop_last is False
    assert len(subject.net.prompt.head.heads) == 2 and bool(subject.net.task_finalized)
    assert len(dataset.logged) == 2
