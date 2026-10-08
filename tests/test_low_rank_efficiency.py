"""Efficiency scope, accepted prerequisites, and complete native task-one costs."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import low_rank_protocol as validation
from scripts import run_low_rank_efficiency as runner


def spec(method='ewc-lora', pass_name='measurement'):
    return {
        'method': method, 'model': method, 'dataset': 'seq-cifar100-224',
        'mode': 'efficiency', 'tasks': 1,
        'protocol': runner.efficiency_protocol(method), 'pass': pass_name,
        'gpu': '1', 'checkpoint_file': validation.CHECKPOINT,
        'checkpoint_sha256': validation.CHECKPOINT_SHA256,
        'canonical_pretrained_sha256': validation.CANONICAL_PRETRAINED_SHA256,
        'reference_commit': validation.REFERENCE_COMMIT,
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'wandb_mode': 'online',
        'data_root': '/shared/data', 'output': '/unused/low-rank-pilot',
        'initial_state': '/unused/initial.pt', 'initial_state_sha256': 'a' * 64,
        'acceptance_summary': '/unused/acceptance.json', 'acceptance_summary_sha256': 'b' * 64,
    }


def accepted(method='ewc-lora'):
    return {
        'method': method, 'mode': 'validation', 'tasks': 2,
        'protocol': validation.protocol(method), 'implementation': 'mammoth_native',
        'status': 'completed', 'computation_status': 'completed',
        'wandb_logging_status': 'uploaded', 'acceptance_status': 'pass',
        'acceptance': {'status': 'pass'},
        'checkpoint_sha256': validation.CHECKPOINT_SHA256,
        'canonical_pretrained_sha256': validation.CANONICAL_PRETRAINED_SHA256,
        'reference_commit': validation.REFERENCE_COMMIT, 'precision': validation.PRECISION,
        'initial_state_sha256': 'a' * 64, 'initial_learned_sha256': 'c' * 64,
        'structural_checks': {name: True for name in validation.REQUIRED_STRUCTURAL_CHECKS},
    }


def completed(method='ewc-lora'):
    begin, end = runner.EXTRA_EXAMPLES[method]
    record = {
        **spec(method),
        'counts': {'observe_calls': 79, 'optimizer_updates': 79,
                   'task_begin_extra_samples': begin, 'task_end_extra_samples': end},
        'epoch_results': [{'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000, 'loss_mean': .5}],
        'task_results': [{'task': 1, 'seen_correct': 900, 'seen_samples': 1000,
                          'old_correct': 0, 'old_samples': 0, 'new_correct': 900, 'new_samples': 1000}],
        'prediction_files': [{'task': 1, 'path': 'task1_predictions.npz', 'sha256': 'd' * 64}],
        'first_batch_audits': [{'task': 1, 'epoch': 1, 'samples': 64,
                               'seed': validation.phase_seed(1993, 0, 'train'),
                               'optimizer_update_index': 1, 'input_sha256': 'e' * 64,
                               'labels_sha256': 'f' * 64, 'gradient_sha256': 'a' * 64}],
        'structural_checks': {name: True for name in runner.REQUIRED_CHECKS},
    }
    for name in ('training_protocol_complete', 'optimizer_updates_complete'):
        record['structural_checks'][name] = False
    recorder = SimpleNamespace(steps=79, training_examples=5000, evaluation_examples=1000, tasks_completed=1)
    return record, recorder


@pytest.mark.parametrize('method', validation.METHODS)
def test_protocol_retains_task_capacity_and_method_defaults_for_short_pilot(method):
    settings = runner.efficiency_protocol(method)
    assert settings['mode'] == 'efficiency'
    assert settings['tasks'] == settings['n_epochs'] == 1
    assert settings['dataset'] == 'seq-cifar100-224' and settings['dataset_config'] == 'ease'
    assert settings['dataset_total_tasks'] == 10 and settings['classes_per_task'] == 10
    assert settings['class_order'] == list(range(100))
    assert settings['vit_size'] == 'base' and settings['rank'] == 10
    assert settings['batch_size'] == 64 and settings['num_workers'] == 4 and settings['seed'] == 1993
    assert settings['precision'] == validation.PRECISION
    assert settings['hyperparameters'] == validation.protocol(method)['hyperparameters']
    assert settings['paper_reproduction'] is False and settings['efficiency_comparison'] is True
    command = runner.native_command(spec(method))
    for flag, value in (('--model', method), ('--stop_after', '1'), ('--n_epochs', '1'),
                        ('--dataset_config', 'ease'), ('--batch_size', '64'), ('--num_workers', '4')):
        assert command[command.index(flag) + 1] == value
    assert '--num_tasks' not in command and '--loadcheck' not in command


@pytest.mark.parametrize('method', validation.METHODS)
@pytest.mark.parametrize('pass_name', ['measurement', 'flops'])
def test_both_passes_require_the_same_frozen_native_scope(method, pass_name):
    runner.validate_spec(spec(method, pass_name))


@pytest.mark.parametrize('key,value', [
    ('method', 'sgd'), ('model', 'sd-lora'), ('dataset', 'seq-cifar10-224'),
    ('mode', 'validation'), ('tasks', 2), ('tasks', True),
    ('gpu', '0'), ('pass', 'warmup'), ('wandb_mode', 'offline'),
    ('wandb_entity', 'another-account'), ('wandb_project', 'another-project'),
    ('checkpoint_sha256', 'changed'), ('canonical_pretrained_sha256', 'changed'),
    ('reference_commit', 'changed'),
])
def test_changed_scope_or_evidence_cannot_launch_a_worker(key, value):
    current = spec()
    current[key] = value
    with pytest.raises(ValueError):
        runner.validate_spec(current)


@pytest.mark.parametrize('key,value', [
    ('tasks', 10), ('n_epochs', 20), ('n_epochs', True), ('batch_size', 128), ('num_workers', 0),
    ('seed', 0), ('rank', 4), ('dataset_config', 'default'), ('vit_size', 'small'),
    ('class_order', list(reversed(range(100)))), ('training_authorized', False),
])
def test_protocol_changes_cannot_hide_in_otherwise_valid_worker_spec(key, value):
    current = spec()
    current['protocol'][key] = value
    with pytest.raises(ValueError):
        runner.validate_spec(current)


@pytest.mark.parametrize('method', validation.METHODS)
def test_only_each_methods_completed_accepted_native_validation_is_eligible(method):
    runner.validate_acceptance(accepted(method), method)


@pytest.mark.parametrize('key,value', [
    ('status', 'logging_failed'), ('computation_status', 'failed'), ('acceptance_status', 'review'),
    ('wandb_logging_status', 'pending'), ('implementation', 'pinned_author_common_input_reference'),
    ('method', 'sd-lora'), ('mode', 'efficiency'), ('tasks', 1),
    ('checkpoint_sha256', 'changed'), ('canonical_pretrained_sha256', 'changed'),
    ('reference_commit', 'changed'),
])
def test_unfinished_unaccepted_or_different_method_validation_is_rejected(key, value):
    record = accepted()
    record[key] = value
    with pytest.raises(ValueError):
        runner.validate_acceptance(record, 'ewc-lora')


@pytest.mark.parametrize('gate', validation.REQUIRED_STRUCTURAL_CHECKS)
def test_every_required_two_task_validation_gate_is_needed_before_efficiency(gate):
    record = accepted()
    record['structural_checks'].pop(gate)
    with pytest.raises(ValueError):
        runner.validate_acceptance(record, 'ewc-lora')


@pytest.mark.parametrize('key', ['protocol', 'precision'])
def test_validation_protocol_and_effective_precision_are_pinned(key):
    record = accepted()
    record[key] = deepcopy(record[key])
    record[key]['n_epochs' if key == 'protocol' else 'tf32_allowed'] = 1 if key == 'protocol' else True
    with pytest.raises(ValueError):
        runner.validate_acceptance(record, 'ewc-lora')


def fingerprint():
    return {'n_samples': 5000, 'class_counts': {str(label): 500 for label in range(10)},
            'samples_ordered_sha256': 'a' * 64, 'samples_unordered_sha256': 'b' * 64,
            'global_ids_ordered_sha256': 'c' * 64, 'transform_repr_sha256': 'd' * 64}


def test_raw_validation_data_and_common_pilot_transforms_are_independent_requirements():
    actual, raw_reference, transform_reference = fingerprint(), fingerprint(), fingerprint()
    raw_reference['transform_repr_sha256'] = 'original-source-transform'
    transform_reference['samples_ordered_sha256'] = 'unused-transform-reference-data'
    assert runner.audit_fingerprints(actual, raw_reference, transform_reference) is True


@pytest.mark.parametrize('key', ['n_samples', 'class_counts', 'samples_ordered_sha256',
                                 'samples_unordered_sha256', 'global_ids_ordered_sha256'])
def test_missing_or_changed_raw_data_cannot_enter_the_common_comparison(key):
    actual, reference = fingerprint(), fingerprint()
    actual.pop(key)
    with pytest.raises(RuntimeError):
        runner.audit_fingerprints(actual, reference, reference)
    actual = fingerprint()
    actual[key] = 'changed'
    with pytest.raises(RuntimeError):
        runner.audit_fingerprints(actual, reference, reference)


def test_original_lora_transform_cannot_be_used_as_common_pilot_evidence():
    actual, raw_reference, transform_reference = fingerprint(), fingerprint(), fingerprint()
    actual['transform_repr_sha256'] = raw_reference['transform_repr_sha256'] = 'original-source-transform'
    with pytest.raises(RuntimeError):
        runner.audit_fingerprints(actual, raw_reference, transform_reference)
    actual.pop('transform_repr_sha256')
    with pytest.raises(RuntimeError):
        runner.audit_fingerprints(actual, raw_reference, transform_reference)


@pytest.mark.parametrize('method', validation.METHODS)
def test_one_task_completion_includes_each_full_statistics_or_prototype_pass(method):
    record, recorder = completed(method)
    runner.completion_checks(record, recorder)
    assert record['structural_checks']['training_protocol_complete'] is True
    assert record['structural_checks']['optimizer_updates_complete'] is True


@pytest.mark.parametrize('method', validation.METHODS)
@pytest.mark.parametrize('counter', ['task_begin_extra_samples', 'task_end_extra_samples'])
def test_skipped_or_extra_method_statistics_prevent_completion(method, counter):
    record, recorder = completed(method)
    record['counts'][counter] += 1
    with pytest.raises((ValueError, RuntimeError)):
        runner.completion_checks(record, recorder)


@pytest.mark.parametrize('corruption', [
    'dropped_tail', 'missing_epoch', 'duplicate_epoch', 'future_task', 'missing_update',
    'extra_update', 'boolean_updates', 'extra_evaluation', 'missing_task_evaluation',
    'recorder_steps', 'recorder_examples', 'observe_calls',
    'old_samples', 'future_predictions', 'missing_predictions', 'missing_batch_audit',
])
def test_completion_rejects_partial_repeated_or_unrecorded_work(corruption):
    record, recorder = completed('cl-lora')
    if corruption == 'dropped_tail':
        record['epoch_results'][0]['samples'] = 4992
    elif corruption == 'missing_epoch':
        record['epoch_results'].clear()
    elif corruption == 'duplicate_epoch':
        record['epoch_results'].append(deepcopy(record['epoch_results'][0]))
    elif corruption == 'future_task':
        record['epoch_results'][0]['task'] = 2
    elif corruption == 'missing_update':
        record['counts']['optimizer_updates'] = 78
    elif corruption == 'extra_update':
        record['counts']['optimizer_updates'] = 158
    elif corruption == 'boolean_updates':
        record['counts']['optimizer_updates'] = True
    elif corruption == 'extra_evaluation':
        recorder.evaluation_examples = 2000
    elif corruption == 'missing_task_evaluation':
        recorder.tasks_completed = 0
    elif corruption == 'recorder_steps':
        recorder.steps = 78
    elif corruption == 'recorder_examples':
        recorder.training_examples = 4992
    elif corruption == 'observe_calls':
        record['counts']['observe_calls'] = 80
    elif corruption == 'old_samples':
        record['task_results'][0]['old_samples'] = 100
    elif corruption == 'future_predictions':
        record['task_results'][0]['task'] = 2
    elif corruption == 'missing_predictions':
        record['prediction_files'].clear()
    else:
        record['first_batch_audits'].clear()
    with pytest.raises((ValueError, RuntimeError)):
        runner.completion_checks(record, recorder)


def test_failed_structural_audit_cannot_be_overwritten_by_valid_counts():
    record, recorder = completed()
    gate = next(name for name in runner.REQUIRED_CHECKS
                if name not in ('training_protocol_complete', 'optimizer_updates_complete'))
    record['structural_checks'][gate] = False
    with pytest.raises((ValueError, RuntimeError)):
        runner.completion_checks(record, recorder)


def test_common_preprocessing_replaces_source_crop_and_resize_on_this_instance_only():
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    class Dataset:
        TRANSFORM = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
        TEST_TRANSFORM = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    original_train, original_test = Dataset.TRANSFORM, Dataset.TEST_TRANSFORM
    dataset = Dataset()
    runner.apply_common_transforms(dataset)
    training, testing = dataset.TRANSFORM.transforms, dataset.TEST_TRANSFORM.transforms
    assert isinstance(training[0], transforms.RandomResizedCrop)
    assert training[0].scale == (.05, 1.) and training[0].interpolation == InterpolationMode.BILINEAR
    assert isinstance(testing[0], transforms.Resize) and testing[0].size == 256
    assert testing[0].interpolation == InterpolationMode.BICUBIC
    assert isinstance(testing[1], transforms.CenterCrop) and testing[1].size == (224, 224)
    assert list(training[-1].mean) == list(testing[-1].mean) == [0, 0, 0]
    assert list(training[-1].std) == list(testing[-1].std) == [1, 1, 1]
    assert Dataset.TRANSFORM is original_train and Dataset.TEST_TRANSFORM is original_test


@pytest.fixture
def small_audit(monkeypatch, tmp_path):
    """Use real CPU tensors while keeping the audit independent of a large ViT."""
    import numpy as np
    import torch
    from scripts import ease_size_protocol

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = torch.nn.Linear(1, 1, bias=False)
            self.image_encoder.requires_grad_(False)
            self.learned = torch.nn.Parameter(torch.ones(1))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Network()
            self.forward_calls = self.forward_examples = 0

        def forward(self, inputs):
            self.forward_calls += 1
            self.forward_examples += len(inputs)
            logits = torch.full((len(inputs), 10), -10.)
            return logits.scatter_(1, inputs.long(), 10.)

    # Canonical tensor selection has its own full-network tests; this test
    # observes hook behavior without constructing that unrelated encoder.
    monkeypatch.setattr(ease_size_protocol, 'canonical_pretrained_state', lambda encoder: encoder.state_dict())
    model = Model()
    dataset = SimpleNamespace(test_loaders=[])
    recorder = SimpleNamespace(output=tmp_path, snapshots={}, phase=lambda *_: nullcontext())
    record = {'structural_checks': {}}
    audit = runner.EfficiencyAudit(model, dataset, recorder, record, torch, np)
    return audit, model, dataset, recorder, record, torch, np


def test_prediction_audit_captures_one_actual_native_evaluation_before_inplace_masking(small_audit, monkeypatch):
    audit, model, dataset, recorder, record, torch, np = small_audit
    labels = torch.arange(10).repeat_interleave(100)
    source = torch.utils.data.TensorDataset(labels.float().unsqueeze(1), labels)
    source.targets, source.indexes = labels.numpy(), np.arange(1000)
    loader = torch.utils.data.DataLoader(source, batch_size=64, shuffle=False)
    dataset.test_loaders.append(loader)
    monkeypatch.setattr(model, 'forward', audit.forwarded(model.forward))
    evaluations = []

    def native_evaluate():
        evaluations.append(1)
        with torch.no_grad():
            for inputs, _ in loader:
                logits = model(inputs)
                # Mammoth masks the returned tensor in-place for task-IL.
                # Captured class-IL predictions must already be independent.
                logits.fill_(-100.)
        return [100.], [100.]

    result = audit.evaluated(native_evaluate)()
    assert result == ([100.], [100.]) and len(evaluations) == 1
    assert model.forward_calls == len(loader) and model.forward_examples == 1000
    assert record['task_results'][0]['seen_correct'] == 1000
    with np.load(recorder.output / 'task1_predictions.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['predictions'], labels.numpy())
        assert np.array_equal(saved['global_ids'], np.arange(1000))
    assert record['structural_checks']['evaluation_predictions_match_accuracy'] is True


def test_backward_audit_handles_fisher_autograd_and_rejects_finite_loss_with_bad_gradient(small_audit):
    audit, model, _, _, _, torch, _ = small_audit
    with torch.no_grad():
        with torch.enable_grad():
            loss = model.net.learned.square().sum()
            audit.backward(torch.Tensor.backward)(loss)
    assert torch.equal(model.net.learned.grad, torch.tensor([2.]))
    model.net.learned.grad = None

    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.sum()

        @staticmethod
        def backward(ctx, grad):
            return torch.full((1,), float('inf'))

    loss = BadGradient.apply(model.net.learned)
    assert torch.isfinite(loss)
    with pytest.raises(FloatingPointError):
        audit.backward(torch.Tensor.backward)(loss)


@pytest.mark.parametrize('shared_storage', [False, True])
def test_snapshots_count_unregistered_inf_covariance_storage_without_alias_duplication(small_audit, shared_storage):
    audit, model, _, recorder, _, torch, _ = small_audit
    encoder = model.net.image_encoder
    encoder.matrix = torch.zeros(2, 2)
    encoder.cur_matrix = encoder.matrix if shared_storage else torch.zeros(3, 3)
    audit.snapshot('covariances')
    snapshot = recorder.snapshots['covariances']
    assert snapshot['registered_buffer_storage_bytes'] == 0
    assert snapshot['unregistered_attention_state_storage_bytes'] == (4 if shared_storage else 13) * 4
    encoder.register_buffer('registered', torch.ones(2))
    audit.snapshot('with_buffer')
    assert recorder.snapshots['with_buffer']['registered_buffer_storage_bytes'] == 8
    assert recorder.snapshots['with_buffer']['unregistered_attention_state_storage_bytes'] == snapshot['unregistered_attention_state_storage_bytes']


def test_default_main_is_a_stdlib_only_dry_run_without_files_or_secrets(tmp_path):
    root = str(Path(runner.__file__).resolve().parents[1])
    code = '''
import importlib.abc
from pathlib import Path
import sys
sys.path.insert(0, ROOT)
class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'numpy', 'wandb', 'PIL', 'psutil'}:
            raise RuntimeError('Dry run attempted ML or sensor import: ' + fullname)
sys.meta_path.insert(0, RejectHeavyImports())
from scripts import run_low_rank_efficiency as runner
def forbidden(*args, **kwargs):
    raise RuntimeError('Dry run attempted to load credentials')
runner.configure_environment = forbidden
assert runner.main([]) == 0
assert not list(Path.cwd().iterdir())
assert not {'torch', 'torchvision', 'numpy', 'wandb'}.intersection(sys.modules)
'''.replace('ROOT', repr(root))
    result = subprocess.run([sys.executable, '-S', '-c', code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['execution'] is False
