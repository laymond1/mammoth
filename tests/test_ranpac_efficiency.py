"""RanPAC common-input evidence and native-worker delegation; no neural computation."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_ranpac_efficiency as runner
from scripts.cl_runtime import PUBLIC_DEFAULTS
from scripts.ease_reference_utils import dataset_fingerprint


def test_dry_run_has_no_ml_imports_credentials_or_filesystem_writes(tmp_path):
    command = ('import sys; from scripts import run_ranpac_efficiency as runner; '
               f'runner.main(["--output", {str(tmp_path)!r}]); '
               'assert "torch" not in sys.modules and "wandb" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', command], cwd=runner.ROOT,
                            capture_output=True, text=True, check=True)
    saved = json.loads(result.stdout)
    assert saved['experiment_executed'] is False
    assert saved['worker_spec']['pass'] == 'measurement'
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('pass_name', ('measurement', 'flops'))
def test_spec_pins_shared_npz_native_hyperparameters_and_complete_task(pass_name, tmp_path):
    spec = runner.make_spec(pass_name, tmp_path)
    flags = dict(zip(spec['main_args'][::2], spec['main_args'][1::2]))
    assert spec['pass'] == pass_name and spec['gpu'] == '1'
    for name, value in {'pretrained': '0', 'pretrained_checkpoint': runner.CHECKPOINT,
                        'dataset_config': 'ease', 'stop_after': '1', 'n_epochs': '1',
                        'seed': '1993', 'batch_size': '64', 'num_workers': '4',
                        'rp_size': '10000', 'optimizer': 'sgd', 'lr': '0.01',
                        'optim_mom': '0.9', 'optim_wd': '0.0005'}.items():
        assert flags['--' + name] == value


@pytest.mark.parametrize('changes', ({'gpu': '0'}, {'gpu': 'cpu'}, {'data_root': '/tmp/other'}))
def test_changed_device_or_data_root_rejected(changes, tmp_path):
    with pytest.raises(ValueError, match='GPU1'):
        runner.make_spec('measurement', tmp_path, dict(PUBLIC_DEFAULTS, **changes))


def test_invalid_pass_rejected(tmp_path):
    with pytest.raises(ValueError, match='only measurement'):
        runner.make_spec('validation', tmp_path)


def test_common_precision_adds_two_cpu_threads_without_replacing_seven_precision_settings(monkeypatch):
    precision = {'parameter_dtype': 'float32', 'autocast': False, 'matmul_precision': 'highest',
                 'tf32_allowed': False, 'cudnn_tf32_allowed': True,
                 'cudnn_deterministic': True, 'cudnn_benchmark': False}
    threads = []
    torch = SimpleNamespace(set_num_threads=lambda value: threads.append(value),
                            get_num_threads=lambda: threads[-1])
    monkeypatch.setattr(runner, '_NATIVE_PRECISION', lambda torch: precision.copy())
    assert runner.configure_precision(torch) == dict(precision, torch_cpu_threads=2)
    assert threads == [2]


def test_execution_pin_rejects_corrupted_local_weights_before_framework_import(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'checkpoint.npz'
    checkpoint.write_bytes(b'corrupted')
    monkeypatch.setattr(runner, 'CHECKPOINT', str(checkpoint))
    with pytest.raises(ValueError, match='exact local'):
        runner.execution_pin(runner.make_spec('measurement', tmp_path / 'run'))


class Transform:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value


@pytest.fixture
def input_audit():
    audit = object.__new__(runner.RanPACEfficiencyAudit)
    train = SimpleNamespace(data=np.zeros((5000, 1, 1, 3), dtype=np.uint8),
                            targets=np.arange(5000) % 10, indexes=np.arange(5000), transform=Transform('epoch augmentation'))
    test = SimpleNamespace(data=np.zeros((1000, 1, 1, 3), dtype=np.uint8),
                           targets=np.arange(1000) % 10, indexes=np.arange(1000), transform=Transform('common evaluation'))
    audit.dataset = SimpleNamespace(train_loader=SimpleNamespace(dataset=train),
                                    test_loaders=[SimpleNamespace(dataset=test)])
    audit.metadata = {'reference_data_snapshots': {
        f'task1_{name}': dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
        for name, source in (('train', train), ('test', test))}}
    audit.recorder = SimpleNamespace(phase=lambda name: nullcontext(), data_snapshots={'first_task': {}})
    audit.checks = {'effective_training_transforms_match_common': False}
    audit.training_snapshots = None
    audit.solve_calls = []
    return audit


@pytest.mark.parametrize('kind', ('transform', 'labels', 'raw_images'))
def test_actual_loader_changes_rejected_before_epoch(input_audit, kind):
    source = input_audit.dataset.train_loader.dataset
    if kind == 'transform':
        source.transform = Transform('different augmentation')
    elif kind == 'labels':
        source.targets[0] = 9
    else:
        source.data[0, 0, 0, 0] = 255
    with pytest.raises(RuntimeError, match='Actual RanPAC epoch loader'):
        input_audit.capture_training_preprocessing()


def test_epoch_guard_rejects_transform_override_after_begin_task(input_audit):
    input_audit.capture_training_preprocessing()
    input_audit.guard_epoch_transforms()
    input_audit.dataset.train_loader.dataset.transform = Transform('model override')
    with pytest.raises(RuntimeError, match='preprocessing changed'):
        input_audit.guard_epoch_transforms()


def test_native_finalization_test_transform_does_not_overwrite_epoch_evidence(input_audit, monkeypatch):
    audit = input_audit
    audit.capture_training_preprocessing()
    original = json.loads(json.dumps(audit.recorder.data_snapshots['epoch_inputs']))
    # Mimic the unchanged native end_task and the old common audit update.
    audit.dataset.train_loader.dataset.transform = audit.dataset.test_loaders[0].dataset.transform

    def native_end_task(self):
        for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[0])):
            source = loader.dataset
            self.recorder.data_snapshots['first_task'][split] = dataset_fingerprint(
                source.data, source.targets, source.indexes, source.transform)

    monkeypatch.setattr(runner.baseline_audit.BaselineAudit, 'end_task', native_end_task)
    audit.end_task()
    assert audit.recorder.data_snapshots['first_task'] == original
    assert audit.recorder.data_snapshots['task1_train'] == original['train']
    feature = audit.recorder.data_snapshots['task_end_feature_inputs']
    assert feature['transform_repr_sha256'] == original['test']['transform_repr_sha256']
    assert feature['transform_repr_sha256'] != original['train']['transform_repr_sha256']
    assert audit.dataset.train_loader.dataset.transform is audit.dataset.test_loaders[0].dataset.transform


@pytest.mark.parametrize('count', (0, 17, 19))
def test_missing_or_extra_native_ridge_solve_rejected(input_audit, count):
    input_audit.solve_calls = [{'matrix_shape': [10000, 10000], 'rhs_shape': [10000, 100],
                               'device': 'cpu', 'dtype': 'torch.float32'}] * count
    with pytest.raises(RuntimeError, match='all 17 candidate'):
        input_audit.complete()


def test_complete_preserves_all_eighteen_native_cpu_solve_calls(input_audit, monkeypatch):
    audit = input_audit
    audit.solve_calls = [{'matrix_shape': [10000, 10000], 'rhs_shape': [10000, 100],
                         'device': 'cpu', 'dtype': 'torch.float32'}] * 18
    completed = []
    monkeypatch.setattr(runner.baseline_audit.BaselineAudit, 'complete', lambda self: completed.append(True))
    audit.complete()
    assert completed == [True]
    assert audit.metadata['ridge_solver_audit']['observed_calls'] == 18
    assert audit.checks['native_ridge_solver_count'] is True


def test_instrumentation_calls_and_restores_native_hooks_and_solver(monkeypatch):
    calls = []
    model = SimpleNamespace(meta_begin_task=lambda: calls.append('begin_task'),
                            meta_begin_epoch=lambda: calls.append('begin_epoch'),
                            meta_observe=lambda: {'loss': 1.25})
    native_solver = lambda *args, **kwargs: 'native result'
    torch = SimpleNamespace(linalg=SimpleNamespace(solve=native_solver))
    audit = SimpleNamespace(model=model, torch=torch,
                            capture_training_preprocessing=lambda: calls.append('capture'),
                            guard_epoch_transforms=lambda: calls.append('guard'),
                            recorder=SimpleNamespace(active_phase='end_task'), solve_calls=[])
    monkeypatch.setattr(runner, '_NATIVE_INSTRUMENT', lambda audit: nullcontext())
    begin, epoch, observe = model.meta_begin_task, model.meta_begin_epoch, model.meta_observe
    with runner.instrument_ranpac(audit):
        model.meta_begin_task()
        model.meta_begin_epoch()
        audit.recorder.active_phase = 'train_epoch'
        assert model.meta_observe() == {'loss': 1.25}
        matrix = SimpleNamespace(shape=(10000, 10000), device='cpu', dtype='torch.float32')
        rhs = SimpleNamespace(shape=(10000, 100))
        assert torch.linalg.solve(matrix, rhs) == 'native result'
        assert audit.solve_calls == []
        audit.recorder.active_phase = 'end_task'
        assert torch.linalg.solve(matrix, rhs) == 'native result'
        assert len(audit.solve_calls) == 1
    assert calls == ['begin_task', 'capture', 'guard', 'begin_epoch', 'guard']
    assert model.meta_begin_task is begin and model.meta_begin_epoch is epoch and model.meta_observe is observe
    assert torch.linalg.solve is native_solver


def test_worker_reuses_common_native_benchmark_and_exposes_result_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'configure_environment', lambda: PUBLIC_DEFAULTS.copy())
    output = tmp_path / 'measurement1'
    calls = []

    def worker(spec_path):
        spec = json.loads(spec_path.read_text())
        assert runner.baseline_audit.BaselineAudit is runner.RanPACEfficiencyAudit
        assert runner.baseline_audit.execution_pin is runner.execution_pin
        assert runner.baseline_audit.instrument_baseline is runner.instrument_ranpac
        calls.append(spec['pass'])
        (output / 'metrics.json').write_text('{"status": "completed"}\n')
        return 0

    monkeypatch.setattr(runner.benchmark_cl, 'worker', worker)
    assert runner.execute('measurement', output) == 0
    assert calls == ['measurement']
    assert (output / 'result.json').read_bytes() == (output / 'metrics.json').read_bytes()
    with pytest.raises(ValueError, match='preserve prior results'):
        runner.execute('measurement', output)


def test_actual_native_dataset_loaders_match_saved_ease_and_hide_without_image_execution(monkeypatch):
    from main import parse_args
    from datasets import get_dataset
    from datasets.seq_cifar100_224 import SequentialCIFAR100224
    from torchvision.datasets import CIFAR100
    from utils.conf import base_path

    for name in ('TRANSFORM', 'TEST_TRANSFORM', 'MEAN', 'STD', 'SIZE', 'N_TASKS', 'N_CLASSES', 'N_CLASSES_PER_TASK'):
        monkeypatch.setattr(SequentialCIFAR100224, name, getattr(SequentialCIFAR100224, name))
    # Require existing cached files; never invoke torchvision's network download.
    def cached_only(self):
        if not self._check_integrity():
            pytest.fail('CPU loader check requires the existing shared CIFAR100 files')

    monkeypatch.setattr(CIFAR100, 'download', cached_only)
    spec = runner.make_spec('measurement', Path('/tmp/ranpac-loader-check'))
    args = parse_args(cmd=spec['main_args'], verbose=False)
    assert args.pretrained is False and args.dataset_config == 'ease'
    base_path(PUBLIC_DEFAULTS['data_root'])
    dataset = get_dataset(args)
    train, test = dataset.get_data_loaders()
    saved = json.loads((runner.ROOT / 'benchmarks/cl_efficiency/ease/efficiency_pilot_results.json').read_text())
    for split, loader in (('train', train), ('test', test)):
        source = loader.dataset
        actual = dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
        reference = saved['measurements'][0]['data_snapshots']['first_task'][split]
        assert all(actual[key] == reference[key] for key in (*runner.RAW_KEYS, 'transform_repr_sha256'))
        assert actual['transform_repr_sha256'] == runner.TRANSFORM_SHA[split]
    assert len(train.dataset) == 5000 and len(test.dataset) == 1000
