"""Run the approved SD recovery gate, then three measurements and one profiler.

Preparation is read-only. Saved author evidence is reused without retraining;
failed or review gates stop the schedule. Existing pilot records stay intact.
"""
from __future__ import annotations

import argparse
import datetime
from fractions import Fraction
import json
import math
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.run_peft_efficiency import file_hash, gpu_occupancy, stop_launcher

DEFAULT_OUTPUT = ROOT / 'results/sd_recovery_20261008'
PLAN = ROOT / 'benchmarks/cl_efficiency/sd_recovery/plan.json'
THRESHOLDS = {'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.0, 'minimum_accuracy_pct': 90.0}


def load_plan():
    plan = json.loads(PLAN.read_text())
    expected = {'training_authorized': True, 'method': 'sd-lora', 'recipe': 'sd-common-lr001',
                'tasks': 1, 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                'gpu': '1', 'vit_size': 'base', 'lora_rank': 10, 'encoder_and_scale_lr': .001,
                'classifier_lr': .008, 'train_crop_min': .05, 'native_validation_runs': 1,
                'measurement_repeats': 3, 'flops_runs': 1, 'maximum_new_gpu_runs': 5,
                'author_retrained': False, **THRESHOLDS}
    if any(type(plan.get(key)) is not type(value) or plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Recovery plan differs from the approved recipe, gate or five-run limit')
    return plan


def accuracy_gate(native_pct, reference_pct=99.0, structural_checks_pass=True):
    if (type(native_pct) not in (int, float) or type(reference_pct) not in (int, float)
            or not math.isfinite(native_pct) or not math.isfinite(reference_pct)
            or not 0 <= native_pct <= 100 or not 0 <= reference_pct <= 100):
        raise ValueError('Accuracy must be a finite percentage')
    drop = float(Fraction(str(reference_pct)) - Fraction(str(native_pct)))
    status = ('fail' if structural_checks_pass is not True or native_pct < 90 or drop > 3
              else 'review' if drop > 1.5 else 'pass')
    return {'status': status, 'accuracy': {'reference_pct': reference_pct, 'native_pct': native_pct, 'drop_pp': drop},
            'thresholds': dict(THRESHOLDS), 'structural_checks_pass': structural_checks_pass is True,
            'measurement_aggregate_eligible': False}


def acceptance(directory):
    """Compare complete saved predictions, raw inputs and initialization."""
    import numpy as np
    from scripts import run_sd_recovery as worker
    from scripts.report_hide_results import RAW_KEYS, first_task_snapshot
    from scripts.run_low_rank_efficiency import REQUIRED_CHECKS
    directory = Path(directory).resolve()
    native_path, reference_path = directory / 'result.json', worker.AUTHOR_REFERENCE
    native, reference = json.loads(native_path.read_text()), json.loads(reference_path.read_text())
    validate_record(native, 'validation')
    worker.author_reference_pins()
    first = native['first_batch_audits'][0]
    other = reference['first_batch_audits'][0]
    checks = {key + '_matches': first.get(key) == other.get(key) for key in
              ('input_sha256', 'labels_sha256', 'gradient_sha256', 'seed', 'samples')}
    for key in ('initial_state_sha256', 'initial_learned_sha256', 'checkpoint_sha256'):
        checks[key + '_matches'] = native.get(key) == reference.get(key)
    checks['canonical_pretrained_sha256_matches'] = native['initial_canonical_pretrained_sha256'] == reference['canonical_pretrained_sha256']
    checks['precision_matches'] = native['precision'] == reference['precision']
    for split in ('train', 'test'):
        left, right = first_task_snapshot(native, split), first_task_snapshot(reference, split)
        checks[split + '_raw_and_transform_matches'] = all(left.get(key) is not None and left[key] == right.get(key) for key in RAW_KEYS)
    native_predictions, source_predictions = directory / 'task1_predictions.npz', reference_path.parent / 'task1_predictions.npz'
    with np.load(native_predictions, allow_pickle=False) as left, np.load(source_predictions, allow_pickle=False) as right:
        prediction_checks = {key + '_match': bool(np.array_equal(left[key], right[key])) for key in ('labels', 'global_ids')}
        prediction_checks['prediction_arrays_match'] = bool(np.array_equal(left['predictions'], right['predictions']))
        if len(left['predictions']) != 1000 or len(right['predictions']) != 1000:
            raise ValueError('Both validation predictions must cover exactly 1000 samples')
        native_pct = int((left['predictions'] == left['labels']).sum()) / 10.
        reference_pct = int((right['predictions'] == right['labels']).sum()) / 10.
    if native_pct != native['accuracy']['mean_accuracy_pct'] or reference_pct != reference['accuracy']['mean_accuracy_pct']:
        raise ValueError('Stored accuracy disagrees with prediction counts')
    structure = (all(native['structural_checks'].get(key) is True and reference['structural_checks'].get(key) is True
                     for key in REQUIRED_CHECKS) and all(checks.values())
                 and prediction_checks['labels_match'] and prediction_checks['global_ids_match'])
    result = accuracy_gate(native_pct, reference_pct, structure)
    result.update(first_batch_checks=checks, prediction_checks=prediction_checks,
        reference_result={'file': str(reference_path), 'sha256': file_hash(reference_path)},
        native_result={'file': str(native_path), 'sha256': file_hash(native_path)},
        source_predictions={'file': str(source_predictions), 'sha256': file_hash(source_predictions)},
        native_predictions={'file': str(native_predictions), 'sha256': file_hash(native_predictions)},
        reference_network_sha256=reference['state_audits']['network_sha256'],
        native_network_sha256=native['state_audits']['network_sha256'],
        network_sha256_matches=native['state_audits']['network_sha256'] == reference['state_audits']['network_sha256'],
        epoch_loss_mean_abs_difference=abs(native['epoch_results'][0]['loss_mean'] - reference['epoch_results'][0]['loss_mean']),
        native_wandb_url=native['wandb_url'], reference_original_logging_status=reference['wandb_logging_status'],
        author_retrained=False, scope='First task / one epoch; no task2 or full-sequence acceptance implied')
    write_json(directory / 'acceptance.json', result)
    return result


def validate_record(record, pass_kind):
    from scripts import run_sd_recovery as worker
    from scripts.run_low_rank_efficiency import REQUIRED_CHECKS, PRECISION
    worker.validate_recovery_record(record, pass_kind)
    worker.validate_spec(record)
    if (record.get('pass') != pass_kind or record.get('status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded' or record.get('precision') != PRECISION
            or any(record.get('structural_checks', {}).get(name) is not True for name in REQUIRED_CHECKS)
            or record.get('measurement_aggregate_eligible') is not (pass_kind == 'measurement')):
        raise ValueError('Recovery worker scope, structure or completed upload is invalid')
    expected = {'tasks_completed': 1, 'outer_observe_steps': 79, 'outer_training_stream_examples': 5000, 'evaluation_examples': 1000}
    if any(type(record.get('metrics', {}).get(k)) is not int or record['metrics'][k] != v for k, v in expected.items()):
        raise ValueError('Recovery worker must complete all 79 updates / 5000 train / 1000 test samples')
    if pass_kind != 'validation':
        from scripts.report_peft_efficiency import validate_new_record
        validate_new_record(record, 'sd-lora', pass_kind)
        observed = record['accuracy']['mean_accuracy_pct']
        if type(observed) not in (int, float) or observed < 90:
            raise ValueError('Recovery efficiency accuracy is below the approved 90% floor')
    return True


def jobs(output):
    return [{'pass': kind, 'directory': str((Path(output) / name).resolve()), 'status': 'pending',
             'command': [sys.executable, str(ROOT / 'scripts/run_sd_recovery.py'), '--pass', kind,
                         '--output', str((Path(output) / name).resolve()), '--execute']}
            for name, kind in [('validation', 'validation'), ('measurement1', 'measurement'),
                               ('measurement2', 'measurement'), ('measurement3', 'measurement'), ('flops', 'flops')]]


def execute(options):
    from scripts import run_sd_recovery as worker
    plan = load_plan()
    profile = configure_environment()
    if any(profile[k] != v for k, v in [('gpu', '1'), ('wandb_entity', 'laymond1'), ('wandb_project', 'cl-efficiency')]):
        raise ValueError('GPU or logging destination differs from authorization')
    output = (options.resume or options.output).resolve()
    sources = worker.execution_sources()
    if options.resume:
        saved = json.loads((output / 'execution.json').read_text())
        expected = jobs(output)
        if (saved['plan'] != plan or saved['execution_source_sha256'] != sources or len(saved['jobs']) != 5
                or any(any(a.get(k) != b[k] for k in ('pass', 'directory', 'command')) for a, b in zip(saved['jobs'], expected))):
            raise ValueError('Resume requires the same approved plan, sources and five-job schedule')
    else:
        output.mkdir(parents=True, exist_ok=False)
        saved = {'status': 'running', 'output': str(output), 'plan': plan, 'jobs': jobs(output),
                 'execution_source_sha256': sources, 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    process = None
    try:
        for index, job in enumerate(saved['jobs'], 1):
            if worker.execution_sources() != sources:
                raise ValueError('Frozen recovery sources changed; no worker started')
            if index > 1 and saved.get('acceptance', {}).get('status') != 'pass':
                raise ValueError('Efficiency measurements require a passed native validation')
            destination = Path(job['directory'])
            path = destination / 'result.json'
            if path.is_file():
                record = json.loads(path.read_text())
                validate_record(record, job['pass'])
                if job.get('result', {}).get('sha256', file_hash(path)) != file_hash(path):
                    raise ValueError('Saved completed result changed; no retraining')
            else:
                if destination.exists():
                    raise ValueError('Incomplete worker directory retained; no overwrite or automatic retry')
                job['gpu_occupancy_before_start'] = gpu_occupancy()
                job['status'] = 'running'
                write_json(output / 'execution.json', saved)
                logs = output / 'launcher_logs'
                logs.mkdir(exist_ok=True)
                print(f'[{index}/5] SD recovery {job["pass"]} starting', flush=True)
                with (logs / (destination.name + '.log')).open('w') as log:
                    process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    if process.wait() != 0:
                        raise RuntimeError('Recovery worker failed; retained evidence requires review')
                process = None
                record = json.loads(path.read_text())
                validate_record(record, job['pass'])
            job.update(status='completed', result={'file': str(path), 'sha256': file_hash(path), 'wandb_url': record['wandb_url']})
            if index == 1:
                saved['acceptance'] = acceptance(destination)
                if saved['acceptance']['status'] != 'pass':
                    saved['status'] = 'gate_' + saved['acceptance']['status']
                    saved['slack_completion_status'] = notify_slack('SD 회복 검증 ' + saved['acceptance']['status'] + ': 추가 측정 중단. ' + record['wandb_url'])
                    write_json(output / 'execution.json', saved)
                    return 1
            write_json(output / 'execution.json', saved)
        saved.update(status='completed', completed_gpu_runs=5, author_retrained=False)
        write_json(output / 'execution.json', saved)
        print(json.dumps({'status': 'completed', 'native_accuracy': saved['acceptance']['accuracy'], 'new_gpu_runs': 5}))
        return 0
    except BaseException as error:
        stop_launcher(process)
        saved.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(output / 'execution.json', saved)
        notify_slack('SD 회복 실행 중단: ' + type(error).__name__ + '; 로컬 증거 보존, 자동 재실행 없음.')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if options.execute:
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Recovery suite termination requested')
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            return execute(options)
        finally:
            signal.signal(signal.SIGTERM, previous)
    print(json.dumps({'execute': False, 'maximum_new_gpu_runs': 5, 'author_retrained': False,
                      'thresholds': THRESHOLDS, 'jobs': [item['pass'] for item in jobs(options.output)]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
