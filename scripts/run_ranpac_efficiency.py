"""Explicit task-one RanPAC measurement with common EASE input preprocessing.

The native end-task feature extraction changes the training dataset to the test
transform. Record that finalization input separately from the transform used by
the actual epoch. Default preparation imports no ML libraries or credentials.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import baseline_audit, benchmark_cl
from scripts.cl_runtime import PUBLIC_DEFAULTS, configure_environment

CHECKPOINT = '/mnt/sdab1/userHome/wslim/.cache/torch/hub/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
CHECKPOINT_SHA = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
TRANSFORM_SHA = {'train': '2b87a34cb83d20320607074f03b709cb68998ff546243dccb2b89d0e5670e412',
                 'test': 'c3425775bd0c9bad0848f553b08357b2a74f4ad5dcb661665ae199095869dc01'}
RAW_KEYS = ('n_samples', 'class_counts', 'samples_ordered_sha256',
            'samples_unordered_sha256', 'global_ids_ordered_sha256')
REQUIRED_CHECKS = (*baseline_audit.CHECKS, 'effective_training_transforms_match_common',
                   'native_ridge_solver_count')
_NATIVE_PRECISION = benchmark_cl._configure_ease_precision


def configure_precision(torch):
    precision = _NATIVE_PRECISION(torch)
    torch.set_num_threads(2)
    return dict(precision, torch_cpu_threads=torch.get_num_threads())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protocol():
    return {'model': 'ranpac', 'tasks': 1, 'dataset': 'seq-cifar100-224',
            'dataset_config': 'ease', 'dataset_total_tasks': 10, 'classes_per_task': 10,
            'n_epochs': 1, 'seed': 1993, 'batch_size': 64, 'num_workers': 4,
            'gpu': '1', 'rp_size': 10000, 'ridge_candidates': 17,
            'lr': 0.01, 'optimizer': 'sgd', 'optim_mom': 0.9, 'optim_wd': 0.0005,
            'torch_cpu_threads': 2,
            'checkpoint_sha256': CHECKPOINT_SHA,
            'preprocessing_sha256': TRANSFORM_SHA.copy(),
            'paper_reproduction': False, 'equal_compute_budget': False,
            'training_requires_explicit_execute': True}


def make_spec(pass_name, output, profile=None):
    if pass_name not in ('measurement', 'flops'):
        raise ValueError('RanPAC supports only measurement and separate FLOPs passes')
    profile = PUBLIC_DEFAULTS if profile is None else profile
    if str(profile['gpu']) != '1' or str(profile['data_root']) != PUBLIC_DEFAULTS['data_root']:
        raise ValueError('RanPAC common-input pilot requires GPU1 and the shared data root')
    output = Path(output).expanduser().resolve()
    args = ['--model', 'ranpac', '--dataset', 'seq-cifar100-224', '--model_config', 'default',
            '--dataset_config', 'ease', '--backbone', 'vit', '--pretrained', '0',
            '--pretrain_type', 'in21k-ft-in1k', '--pretrained_checkpoint', CHECKPOINT,
            '--stop_after', '1', '--n_epochs', '1', '--batch_size', '64', '--num_workers', '4',
            '--seed', '1993', '--rp_size', '10000', '--lr', '0.01', '--optimizer', 'sgd',
            '--optim_mom', '0.9', '--optim_wd', '0.0005', '--code_optimization', '0',
            '--permute_classes', '0', '--debug_mode', '0', '--enable_other_metrics', '0',
            '--eval_future', '0', '--disable_log', '0', '--non_verbose', '1',
            '--save_after_interrupt', '0', '--noise_rate', '0', '--label_perc', '1',
            '--label_perc_by_class', '1', '--validation', '0', '--joint', '0',
            '--fitting_mode', 'epochs', '--base_path', profile['data_root'], '--device', 'cuda:0']
    return {'model': 'ranpac', 'dataset': 'seq-cifar100-224', 'group': 'pretrained_224',
            'kind': 'training', 'baseline_pilot': True, 'ranpac_remeasurement': True,
            'pass': pass_name, 'output': str(output), 'main_args': args,
            'suite': 'ranpac-common-input-' + output.parent.name, 'gpu': '1',
            'data_root': profile['data_root'], 'sample_interval': 1.0,
            'wandb_mode': 'online', 'wandb_entity': profile['wandb_entity'],
            'wandb_project': profile['wandb_project']}


def execution_pin(spec):
    if (spec.get('model') != 'ranpac' or spec.get('gpu') != '1'
            or spec.get('ranpac_remeasurement') is not True or spec.get('pass') not in ('measurement', 'flops')):
        raise ValueError('RanPAC worker differs from the explicitly requested remeasurement')
    if not Path(CHECKPOINT).is_file() or file_hash(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('RanPAC requires the exact local accepted Base NPZ checkpoint')
    ease_path = ROOT / 'benchmarks/cl_efficiency/ease/efficiency_pilot_results.json'
    ease = json.loads(ease_path.read_text())
    if (ease.get('status') != 'completed' or ease.get('pilot_status') != 'pass'
            or ease.get('checkpoint_sha256') != CHECKPOINT_SHA):
        raise ValueError('Common-input evidence requires the completed accepted EASE pilot')
    reference = ease['measurements'][0]['data_snapshots']['first_task']
    hide_path = ROOT / 'benchmarks/cl_efficiency/hide_prompt/results.json'
    hide = json.loads(hide_path.read_text())
    hide_execution = Path(hide['sources']['efficiency']['file'])
    if file_hash(hide_execution) != hide['sources']['efficiency']['sha256']:
        raise ValueError('Saved HiDe-Prompt common-input evidence changed')
    hide_saved = json.loads(hide_execution.read_text())
    if hide_saved.get('status') != 'completed' or hide_saved.get('checkpoint_sha256') != CHECKPOINT_SHA:
        raise ValueError('Common-input evidence requires the completed HiDe-Prompt Base pilot')
    hide_record = hide_saved['measurements'][0]
    for split in ('train', 'test'):
        other = hide_record['data_snapshots'][f'task1_{split}']
        if (reference[split]['transform_repr_sha256'] != TRANSFORM_SHA[split]
                or other['transform_repr_sha256'] != TRANSFORM_SHA[split]
                or any(reference[split][key] != other[key] for key in RAW_KEYS)):
            raise ValueError('EASE and HiDe-Prompt common-input fingerprints disagree')
    files = ('scripts/run_ranpac_efficiency.py', 'scripts/benchmark_cl.py', 'scripts/baseline_audit.py',
             'scripts/cl_efficiency.py', 'scripts/cl_runtime.py', 'main.py', 'utils/training.py',
             'utils/evaluate.py', 'models/ranpac.py', 'backbone/vit.py', 'datasets/seq_cifar100_224.py',
             'datasets/utils/continual_dataset.py', 'datasets/configs/seq-cifar100-224/ease.yaml', 'uv.lock')
    paths = [ROOT / name for name in files]
    paths.extend(sorted((ROOT / 'models/ranpac_utils').glob('*.py')))
    return {'schema_version': 1, 'checkpoint_file': CHECKPOINT, 'checkpoint_sha256': CHECKPOINT_SHA,
            'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'canonical_pretrained_sha256_expected': baseline_audit.PTM_SHA,
            'protocol': protocol(), 'reference_data_snapshots': {
                f'task1_{split}': copy.deepcopy(reference[split]) for split in ('train', 'test')},
            'common_input_evidence': {'ease_results_sha256': file_hash(ease_path),
                                    'hide_report_sha256': file_hash(hide_path),
                                    'hide_execution_sha256': file_hash(hide_execution)},
            'execution_source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
            'minimum_accuracy_pct': 20.0,
            'accuracy_policy': 'Strict >20% gross-collapse guard only; this one-epoch pilot is not paper reproduction.',
            'preprocessing_record_policy': 'Actual epoch transforms are captured before training; native task-end feature transform is recorded separately.'}


class RanPACEfficiencyAudit(baseline_audit.BaselineAudit):
    def __init__(self, model, dataset, args, recorder, metadata, torch):
        expected = {'rp_size': 10000, 'lr': 0.01, 'optimizer': 'sgd', 'optim_mom': 0.9, 'optim_wd': 0.0005}
        if any(getattr(args, key, None) != value for key, value in expected.items()):
            raise ValueError('RanPAC must retain its native RP size and optimizer settings')
        super().__init__(model, dataset, args, recorder, metadata, torch)
        self.training_snapshots = None
        self.solve_calls = []
        self.checks.update(effective_training_transforms_match_common=False,
                           native_ridge_solver_count=False)

    def capture_training_preprocessing(self):
        from scripts.ease_reference_utils import dataset_fingerprint
        with self.recorder.phase('state_audit'):
            snapshots = {}
            for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[0])):
                source = loader.dataset
                actual = dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
                reference = self.metadata['reference_data_snapshots'][f'task1_{split}']
                if any(actual[key] != reference[key] for key in (*RAW_KEYS, 'transform_repr_sha256')):
                    raise RuntimeError('Actual RanPAC epoch loader differs from the common EASE/HiDe inputs')
                snapshots[split] = actual
            self.training_snapshots = copy.deepcopy(snapshots)
            self.recorder.data_snapshots['epoch_inputs'] = copy.deepcopy(snapshots)
            self.checks['effective_training_transforms_match_common'] = True

    def guard_epoch_transforms(self):
        if self.training_snapshots is None:
            raise RuntimeError('RanPAC loaders must be captured after native begin_task')
        for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[0])):
            actual = hashlib.sha256(repr(loader.dataset.transform).encode()).hexdigest()
            if actual != self.training_snapshots[split]['transform_repr_sha256']:
                raise RuntimeError('RanPAC preprocessing changed before the native training epoch')

    def end_task(self):
        self.guard_recorded_epoch_inputs()
        super().end_task()
        # RanPAC intentionally uses TEST_TRANSFORM for this extra feature pass.
        current = self.recorder.data_snapshots['first_task']['train']
        if current['transform_repr_sha256'] != self.training_snapshots['test']['transform_repr_sha256']:
            raise RuntimeError('RanPAC native task-end feature extraction lost the common test transform')
        self.recorder.data_snapshots['task_end_feature_inputs'] = copy.deepcopy(current)
        for split in ('train', 'test'):
            final = self.recorder.data_snapshots['first_task'][split]
            if any(final[key] != self.training_snapshots[split][key] for key in RAW_KEYS):
                raise RuntimeError('RanPAC task-end hook changed the raw task samples')
            self.recorder.data_snapshots['first_task'][split] = copy.deepcopy(self.training_snapshots[split])
            self.recorder.data_snapshots[f'task1_{split}'] = copy.deepcopy(self.training_snapshots[split])

    def guard_recorded_epoch_inputs(self):
        if self.training_snapshots is None or not self.checks['effective_training_transforms_match_common']:
            raise RuntimeError('RanPAC epoch input evidence is missing')

    def complete(self):
        if (len(self.solve_calls) != 18 or any(row != {'matrix_shape': [10000, 10000],
                'rhs_shape': [10000, 100], 'device': 'cpu', 'dtype': 'torch.float32'} for row in self.solve_calls)):
            raise RuntimeError('RanPAC must complete all 17 candidate and one final native CPU ridge solves')
        self.checks['native_ridge_solver_count'] = True
        self.metadata['ridge_solver_audit'] = {'observed_calls': len(self.solve_calls),
                                              'calls': self.solve_calls,
                                              'scope': 'Wrapped native torch.linalg.solve calls in end_task; no replacement or reduced solver.'}
        super().complete()


_NATIVE_INSTRUMENT = baseline_audit.instrument_baseline


@contextmanager
def instrument_ranpac(audit):
    def with_callback(original, callback):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            callback()
            return result
        return wrapped

    def guard_epoch(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            audit.guard_epoch_transforms()
            return original(*args, **kwargs)
        return wrapped

    def guard_observe(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if audit.recorder.active_phase == 'train_epoch':
                audit.guard_epoch_transforms()
            return original(*args, **kwargs)
        return wrapped

    original_solve = audit.torch.linalg.solve

    @functools.wraps(original_solve)
    def solve(matrix, rhs, *args, **kwargs):
        result = original_solve(matrix, rhs, *args, **kwargs)
        if audit.recorder.active_phase == 'end_task':
            audit.solve_calls.append({'matrix_shape': list(matrix.shape), 'rhs_shape': list(rhs.shape),
                                      'device': str(matrix.device), 'dtype': str(matrix.dtype)})
        return result

    with _NATIVE_INSTRUMENT(audit), \
            patch.object(audit.model, 'meta_begin_task', with_callback(audit.model.meta_begin_task, audit.capture_training_preprocessing)), \
            patch.object(audit.model, 'meta_begin_epoch', guard_epoch(audit.model.meta_begin_epoch)), \
            patch.object(audit.model, 'meta_observe', guard_observe(audit.model.meta_observe)), \
            patch.object(audit.torch.linalg, 'solve', solve):
        yield


def execute(pass_name, output):
    profile = configure_environment()
    spec = make_spec(pass_name, output, profile)
    output = Path(spec['output'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('RanPAC worker output must be empty; preserve prior results')
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / 'spec.json'
    benchmark_cl.write_json(spec_path, spec)
    updates = {'CL_EFFICIENCY_EXECUTE': '1', 'CUDA_VISIBLE_DEVICES': '1', 'WANDB_MODE': 'online',
               'WANDB_DIR': str(output), 'WANDB_RUN_GROUP': spec['suite']}
    with patch.dict(os.environ, updates), \
            patch.object(baseline_audit, 'execution_pin', execution_pin), \
            patch.object(baseline_audit, 'BaselineAudit', RanPACEfficiencyAudit), \
            patch.object(baseline_audit, 'instrument_baseline', instrument_ranpac), \
            patch.object(benchmark_cl, '_configure_ease_precision', configure_precision):
        for key in ('MAMMOTH_TEST', 'MAMMOTH_ARGS', 'MAMMOTH_DEVICE'):
            os.environ.pop(key, None)
        code = benchmark_cl.worker(spec_path)
    metrics_path = output / 'metrics.json'
    if metrics_path.is_file():
        (output / 'result.json').write_bytes(metrics_path.read_bytes())
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pass', dest='pass_name', choices=('measurement', 'flops'), default='measurement')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/peft_efficiency/runs/ranpac')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'experiment_executed': False, 'protocol': protocol(),
                          'worker_spec': make_spec(args.pass_name, args.output)}, indent=2))
        return 0
    return execute(args.pass_name, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
