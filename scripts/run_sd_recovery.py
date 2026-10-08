"""Opt-in SD-LoRA first-task verification and efficiency at the recovered LR.

The original model, default recipe, and efficiency worker stay unchanged. This
separate wrapper uses their native trainer and audits with encoder/scale LR
0.001 and classifier LR 0.008. Preparation is dry by default.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_low_rank_efficiency as base
from scripts.cl_efficiency import notify_slack, valid_slack_webhook, write_json
from scripts.cl_runtime import configure_environment

RECIPE = 'sd-common-lr001'
PASSES = ('validation', 'measurement', 'flops')
AUTHOR_REFERENCE = ROOT / 'results/sd_cause_analysis_20261008/lower_lora_lr/result.json'
AUTHOR_RESULT_SHA256 = 'b0ed2393aaeba8c7168eb7bbb5dd1ccc0e0ee4a12b6196be0019e00d04c24685'
AUTHOR_DRIVER = ROOT / 'results/sd_cause_analysis_20261008/driver.py'
AUTHOR_DRIVER_SHA256 = 'f38dfeb32676b30a9d76390fa60ff93e85cb699674bbec73327b8bdfdb1243e1'
_BASE_PROTOCOL = base.efficiency_protocol
_BASE_COMMAND = base.native_command
_BASE_SOURCES = base.execution_sources
_BASE_BUILD_SPEC = base.build_spec
_BASE_AUDIT = base.EfficiencyAudit


def recovery_protocol(pass_kind='measurement'):
    if pass_kind not in PASSES:
        raise ValueError('SD recovery permits validation, measurement, or flops only')
    settings = deepcopy(_BASE_PROTOCOL('sd-lora'))
    settings['mode'] = 'validation' if pass_kind == 'validation' else 'efficiency'
    settings['hyperparameters']['lr'] = .001
    settings.update(recovery_recipe=RECIPE, encoder_and_scale_lr=.001,
                    validation_runs=1, pilot_min_accuracy_pct=90.,
                    efficiency_comparison=pass_kind != 'validation')
    return settings


def native_command(spec, output=None):
    command = _BASE_COMMAND(spec, output)
    command[command.index('--lr') + 1] = '0.001'
    command[command.index('--lora_fc_lr') + 1] = '0.008'
    return command


def execution_sources():
    result = _BASE_SOURCES()
    for relative in ('scripts/run_sd_recovery.py', 'scripts/run_sd_recovery_suite.py',
                     'benchmarks/cl_efficiency/sd_recovery/plan.json'):
        path = ROOT / relative
        if path.is_file():
            result[relative] = base.shared.file_hash(path)
    return result


def validate_spec(spec):
    pass_kind = spec.get('pass')
    expected_mode = 'validation' if pass_kind == 'validation' else 'efficiency'
    if (pass_kind not in PASSES or spec.get('method') != 'sd-lora' or spec.get('model') != 'sd-lora'
            or spec.get('dataset') != 'seq-cifar100-224' or spec.get('mode') != expected_mode
            or type(spec.get('tasks')) is not int or spec['tasks'] != 1
            or spec.get('protocol') != recovery_protocol(pass_kind) or spec.get('recovery_recipe') != RECIPE
            or spec.get('implementation') != 'mammoth_native' or spec.get('gpu') != '1'
            or spec.get('checkpoint_sha256') != base.CHECKPOINT_SHA256
            or spec.get('canonical_pretrained_sha256') != base.CANONICAL_PRETRAINED_SHA256
            or spec.get('reference_commit') != base.REFERENCE_COMMIT
            or spec.get('wandb_mode') != 'online' or spec.get('wandb_entity') != 'laymond1'
            or spec.get('wandb_project') != 'cl-efficiency'
            or spec.get('author_reference_sha256') != AUTHOR_RESULT_SHA256
            or type(spec.get('measurement_aggregate_eligible')) is not bool
            or spec['measurement_aggregate_eligible'] != (pass_kind == 'measurement')):
        raise ValueError('SD recovery must retain the approved one-task lower-LR GPU1 scope')
    for key in ('tasks', 'n_epochs', 'batch_size', 'num_workers', 'seed', 'rank', 'dataset_total_tasks', 'classes_per_task',
                'measurement_repeats', 'flops_runs', 'validation_runs'):
        if type(spec['protocol'].get(key)) is not int:
            raise ValueError('SD recovery protocol counts must be exact integers')


def author_reference_pins(reference=AUTHOR_REFERENCE):
    """Freeze the saved 99% author evidence without rerunning or publishing it."""
    reference = Path(reference).resolve()
    if base.shared.file_hash(reference) != AUTHOR_RESULT_SHA256:
        raise ValueError('Saved lower-LR author result differs from the approved evidence')
    record = json.loads(reference.read_text())
    hyper = record.get('protocol', {}).get('hyperparameters', {})
    if (record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('implementation') != 'immutable_author_sd' or record.get('case') != 'lower_lora_lr'
            or record.get('method') != 'sd-lora' or record.get('tasks') != 1
            or record.get('accuracy', {}).get('mean_accuracy_pct') != 99.
            or hyper != {'optimizer': 'adam', 'lr': .001, 'fc_lr': .008, 'weight_decay': 0.}
            or record.get('checkpoint_sha256') != base.CHECKPOINT_SHA256
            or record.get('canonical_pretrained_sha256') != base.CANONICAL_PRETRAINED_SHA256
            or record.get('reference_commit') != base.REFERENCE_COMMIT
            or record.get('precision') != base.PRECISION
            or record.get('counts') != {'observe_calls': 79, 'optimizer_updates': 79,
                                      'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}
            or any(record.get('structural_checks', {}).get(name) is not True for name in base.REQUIRED_CHECKS)):
        raise ValueError('Saved author evidence does not establish the approved full first-task contrast')
    if base.shared.file_hash(AUTHOR_DRIVER) != AUTHOR_DRIVER_SHA256:
        raise ValueError('Saved author diagnostic driver changed')
    paths = [reference, reference.parent / 'spec.json', reference.parent / 'task1_predictions.npz',
             reference.parent / 'comparison.json', AUTHOR_DRIVER]
    pins = {str(path.resolve()): base.shared.file_hash(path) for path in paths}
    for relative, digest in record['reference_source_sha256'].items():
        path = (Path(record['reference_source']) / relative).resolve()
        if base.shared.file_hash(path) != digest:
            raise ValueError('Immutable author source differs from the saved reference')
        pins[str(path)] = digest
    return pins


@contextmanager
def worker_overrides():
    """Install process-local adaptations and always restore the original helpers."""
    with ExitStack() as stack:
        for name, replacement in (('validate_spec', validate_spec), ('native_command', native_command),
                                  ('execution_sources', execution_sources), ('EfficiencyAudit', RecoveryAudit)):
            stack.enter_context(patch.object(base, name, replacement))
        yield


def build_spec(options, profile):
    pins = author_reference_pins(options.author_reference)
    with worker_overrides(), patch.object(base, 'efficiency_protocol', lambda method: recovery_protocol(options.pass_kind)), \
            patch.object(base, 'validate_spec', lambda spec: None):
        spec = _BASE_BUILD_SPEC(options, profile)
    mode = 'validation' if options.pass_kind == 'validation' else 'efficiency'
    name = f'sd-recovery-{options.pass_kind}-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec.update(name=name, suite=name, mode=mode, recovery_recipe=RECIPE, implementation='mammoth_native',
                author_reference=str(Path(options.author_reference).resolve()), author_reference_sha256=AUTHOR_RESULT_SHA256,
                recovery_reference_pins=pins, measurement_aggregate_eligible=options.pass_kind == 'measurement',
                efficiency_comparison=options.pass_kind != 'validation',
                validation_scope='New LR, first task and one epoch only; prior two-task default-LR validation remains separate')
    validate_spec(spec)
    return spec


def verify_reference_pins(spec):
    pins = spec.get('recovery_reference_pins', {})
    if not pins or pins.get(spec.get('author_reference')) != AUTHOR_RESULT_SHA256:
        raise ValueError('SD recovery requires the frozen author reference inputs')
    for path, digest in pins.items():
        if base.shared.file_hash(path) != digest:
            raise ValueError('SD recovery author evidence changed after preparation')
    if pins != author_reference_pins(spec['author_reference']):
        raise ValueError('SD recovery saved reference inputs are incomplete or changed')


def validate_recovery_record(record, pass_kind):
    """Validate complete native evidence; accuracy acceptance is the caller's gate.

    Validation runs have the same native workload and saved predictions, but are
    explicitly ineligible for the efficiency aggregate. Energy/FLOPs quality and
    remote W&B verification remain separate report/publication gates.
    """
    validate_spec(record)
    if (record['pass'] != pass_kind or record.get('status') != 'completed'
            or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded' or record.get('precision') != base.PRECISION
            or any(record.get('structural_checks', {}).get(name) is not True for name in base.REQUIRED_CHECKS)):
        raise ValueError('SD recovery requires completed, uploaded, structurally valid native evidence')
    reference = json.loads(Path(record['author_reference']).read_text())
    if (record.get('initial_state_sha256') != reference['initial_state_sha256']
            or record.get('initial_learned_sha256') != reference['initial_learned_sha256']
            or record.get('initial_canonical_pretrained_sha256') != base.CANONICAL_PRETRAINED_SHA256):
        raise ValueError('SD recovery initialization differs from saved author verification')
    expected_totals = {'outer_observe_steps': 79, 'outer_training_stream_examples': 5000,
                       'evaluation_examples': 1000, 'tasks_completed': 1}
    if any(type(record.get('metrics', {}).get(k)) is not int or record['metrics'][k] != v
           for k, v in expected_totals.items()):
        raise ValueError('SD recovery requires the complete first-task stream')
    base.completion_checks(record, SimpleNamespace(steps=79, training_examples=5000,
                                                  evaluation_examples=1000, tasks_completed=1))
    for split in ('task1_train', 'task1_test'):
        base.audit_fingerprints(record.get('data_snapshots', {}).get(split, {}),
                                record['common_reference_data_snapshots'][split],
                                record['common_transform_snapshots'][split])
    first = record['first_batch_audits'][0]
    if any(first.get(key) != reference['first_batch_audits'][0].get(key)
           for key in ('task', 'epoch', 'seed', 'samples', 'optimizer_update_index',
                       'input_sha256', 'labels_sha256', 'gradient_sha256')):
        raise ValueError('SD recovery first optimizer step differs from saved author evidence')
    groups = record.get('optimizer_group_audits', [])
    if (len(groups) != 2 or any(group.get('role') != role or group.get('lr') != rate
            or any(type(group.get(key)) is not int or group[key] != 1
                   for key in ('task', 'epoch', 'optimizer_update_index'))
            or type(group.get('parameters')) is not int or group['parameters'] <= 0
            for group, role, rate in zip(groups, ('encoder_and_scale', 'classifier'), (.001, .008)))):
        raise ValueError('SD recovery requires actual optimizer-group learning-rate evidence')
    prediction = record['prediction_files'][0]
    path = Path(prediction['path'])
    if base.shared.file_hash(path) != prediction['sha256']:
        raise ValueError('SD recovery saved predictions changed')
    import numpy as np
    with np.load(Path(record['author_reference']).parent / 'task1_predictions.npz', allow_pickle=False) as source:
        counts = base.shared.validate_prediction_archive(path, 1, source['global_ids'], source['labels'])
    accuracy = record.get('accuracy', {}).get('mean_accuracy_pct')
    if (type(accuracy) not in (int, float) or not math.isfinite(accuracy)
            or accuracy != counts['seen_correct'] / 10.
            or record['task_results'][0] != {'task': 1, **counts}):
        raise ValueError('SD recovery accuracy differs from the complete saved predictions')
    return record


class RecoveryAudit(_BASE_AUDIT):
    """Verify actual parameter-group LRs at the first native optimizer step."""
    def attach_optimizer(self, optimizer):
        already_hooked = optimizer is None or optimizer in self.hooked
        result = super().attach_optimizer(optimizer)
        if already_hooked:
            return result

        def audit_first_step(subject, arguments, keywords):
            if self.stage != 'train' or self.epoch['steps'] != 0:
                return
            groups = subject.param_groups
            encoder = {id(p) for p in self.model.net.image_encoder.parameters() if p.requires_grad}
            classifier = {id(p) for p in self.model.net.classifier_pool.parameters() if p.requires_grad}
            if (len(groups) != 2 or {id(p) for p in groups[0]['params']} != encoder
                    or {id(p) for p in groups[1]['params']} != classifier
                    or float(groups[0]['lr']) != .001 or float(groups[1]['lr']) != .008):
                raise ValueError('SD recovery actual LoRA/scale and classifier optimizer groups differ from the recipe')
            self.record['optimizer_group_audits'] = [
                {'task': 1, 'epoch': 1, 'optimizer_update_index': 1, 'role': role,
                 'lr': float(group['lr']), 'parameters': sum(p.numel() for p in group['params'])}
                for role, group in zip(('encoder_and_scale', 'classifier'), groups)]

        self.handles.append(optimizer.register_step_pre_hook(audit_first_step))
        return result


def worker(spec_path):
    if os.getenv('CL_SD_RECOVERY_EXECUTE') != '1':
        raise ValueError('SD recovery worker requires explicit parent execution authorization')
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    verify_reference_pins(spec)
    with worker_overrides():
        return base.worker(spec_path)


def execute(options, profile):
    spec = build_spec(options, profile)
    if not valid_slack_webhook():
        raise ValueError('Slack credentials are required before SD recovery execution')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        status = notify_slack(message)
        notifications.append({'message': message, 'status': status})
        write_json(output / 'notifications.json', notifications)
        return status

    if notify(f'SD-LoRA recovery {spec["pass"]} started: task1 epoch1 batch64 GPU1, LoRA/scale LR=.001 classifier LR=.008') != 'sent':
        raise RuntimeError('Required SD recovery start notification failed before training')
    env = os.environ.copy()
    env.pop('MAMMOTH_DEVICE', None)
    env.update(CUDA_VISIBLE_DEVICES='1', WANDB_MODE='online', WANDB_DIR=str(output), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', CL_LOW_RANK_EFFICIENCY_EXECUTE='1', CL_SD_RECOVERY_EXECUTE='1')
    from scripts.benchmark_cl import stop_worker
    started, heartbeat, timed_out = time.monotonic(), time.monotonic(), False
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
            cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= options.timeout:
                    timed_out = True
                    stop_worker(process)
                    break
                if time.monotonic() - heartbeat >= 600:
                    notify(f'SD-LoRA recovery {spec["pass"]} still running; local phase observations preserved')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify(f'SD-LoRA recovery {spec["pass"]} interrupted; local observations preserved')
            raise
    record = json.loads((output / 'result.json').read_text()) if (output / 'result.json').is_file() else {'status': 'failed'}
    status = 'timeout' if timed_out else record['status']
    if process.returncode != 0 and status == 'completed':
        status = 'failed'
    sent = notify(f'SD-LoRA recovery {spec["pass"]} {status}; W&B {record.get("wandb_url", "unavailable")}')
    write_json(output / 'completion.json', {'status': status, 'exit_code': process.returncode, 'timeout': timed_out,
        'wall_s': time.monotonic() - started, 'wandb_url': record.get('wandb_url'), 'slack_completion_status': sent})
    print(json.dumps({'status': status, 'result': str(output / 'result.json'), 'wandb_url': record.get('wandb_url')}))
    return 0 if process.returncode == 0 and status == 'completed' and sent == 'sent' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('sd-lora',), default='sd-lora')
    parser.add_argument('--pass', dest='pass_kind', choices=PASSES, default='validation')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/sd_recovery/prepared')
    parser.add_argument('--author-reference', type=Path, default=AUTHOR_REFERENCE)
    parser.add_argument('--acceptance-summary', type=Path)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(base.shared.CHECKPOINT))
    parser.add_argument('--timeout', type=float, default=3600.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker is not None:
        return worker(options.worker)
    if not options.execute:
        print(json.dumps({'method': 'sd-lora', 'pass': options.pass_kind, 'protocol': recovery_protocol(options.pass_kind),
                          'recovery_recipe': RECIPE, 'gpu': '1', 'execution': False}))
        return 0
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        return execute(options, configure_environment())
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())
