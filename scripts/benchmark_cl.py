"""Prepare or explicitly execute sequential, one-task Mammoth efficiency benchmarks.

The default is a dry run: no model imports, training, downloads, W&B writes or Slack posts.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import datetime
import functools
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COMMON_DATASET = 'seq-cifar100-224'
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, capture_accuracy, flatten, instrument, inventory, model_snapshot, notify_slack, readiness, write_json
from scripts.cl_runtime import configure_environment
from scripts.check_ease_runtime import file_hash


def ease_execution_pin():
    """Verify the local approved Base weights only inside an executing worker."""
    plan_path = ROOT / 'benchmarks/cl_efficiency/ease/validation_plan.json'
    plan = json.loads(plan_path.read_text())
    reference = plan['reference']
    if plan['efficiency']['training_authorized'] is not True:
        raise ValueError('The EASE efficiency protocol has not been authorized')
    checkpoint = Path(reference['checkpoint_file']).expanduser().resolve()
    expected_hash = reference['checkpoint_sha256']
    if (checkpoint.suffix != '.npz' or not checkpoint.is_file()
            or len(expected_hash) != 64 or any(c not in '0123456789abcdef' for c in expected_hash)):
        raise ValueError('EASE requires the pinned local Base NPZ checkpoint')
    if file_hash(checkpoint) != expected_hash:
        raise ValueError('EASE checkpoint differs from the approved SHA256')
    files = ('scripts/benchmark_cl.py', 'scripts/cl_efficiency.py', 'main.py', 'utils/training.py',
             'models/ease.py', 'backbone/ease_vit.py', 'models/ease_utils/vit.py',
             'models/ease_utils/network.py', 'models/ease_utils/classifier.py',
             'datasets/configs/seq-cifar100-224/ease.yaml', 'uv.lock')
    record = {'checkpoint_file': str(checkpoint), 'checkpoint_sha256': expected_hash,
              'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
              'validation_plan_sha256': file_hash(plan_path),
              'execution_source_sha256': {name: file_hash(ROOT / name) for name in files}}
    for name, section in (('official_reference', reference), ('native_validation', plan['validation'])):
        summary = Path(section['result_summary'])
        if not summary.is_absolute():
            summary = plan_path.parent / summary
        record[f'{name}_summary_sha256'] = file_hash(summary)
        if name == 'official_reference':
            saved = json.loads(summary.read_text())['official_reference']['data_snapshots']
            record['reference_data_snapshots'] = {key: saved[key] for key in ('task1_train', 'task1_test')}
        else:
            saved = json.loads(summary.read_text())
            validated = saved['native_validation']
            if (validated['status'] != 'completed' or validated['acceptance_status'] != 'pass'
                    or saved['acceptance']['status'] != 'pass'
                    or saved['checkpoint_sha256'] != expected_hash):
                raise ValueError('EASE requires the accepted native validation with the same pinned weights')
    return record


def _pinned_argument(command, flag, value):
    """Append an explicit value, rejecting conflicting separated/equal flags."""
    supplied = [command[index + 1] for index, arg in enumerate(command[:-1]) if arg == flag]
    supplied += [arg.split('=', 1)[1] for arg in command if arg.startswith(flag + '=')]
    if any(item != value for item in supplied) or (command and command[-1] == flag):
        raise ValueError(f'EASE execution requires {flag} {value}')
    if not supplied:
        command.extend([flag, value])


def _configure_ease_precision(torch):
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the accepted native validation; CUDA matmul TF32 remains disabled.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {'parameter_dtype': 'float32', 'autocast': False,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


class EaseAudit:
    """Observe the approved one-task pilot without additional image forwards."""

    def __init__(self, model, dataset, args, recorder, metadata, torch):
        expected = {'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                    'vit_size': 'base', 'adapter_size': 64, 'ease_pretrain_type': 'in21k_ft_in1k',
                    'dataset_config': 'ease', 'code_optimization': 0, 'drop_last': False,
                    'fitting_mode': 'epochs', 'permute_classes': False, 'joint': False,
                    'lr': 0.025, 'optim_mom': 0.9, 'optim_wd': 0.0005,
                    'optimizer': 'sgd', 'lr_scheduler': 'cosine', 'scheduler_mode': 'epoch'}
        if (any(getattr(args, key, None) != value for key, value in expected.items())
                or args.eval_epochs is not None or args.start_from is not None or args.loadcheck is not None):
            raise ValueError('Effective EASE arguments violate the approved one-epoch Base64 protocol')
        if str(Path(args.ease_checkpoint).resolve()) != metadata['checkpoint_file']:
            raise ValueError('Effective EASE checkpoint differs from the pinned local file')
        self.model, self.dataset, self.recorder, self.metadata, self.torch = model, dataset, recorder, metadata, torch
        self.ptm = model.net.backbone.backbone
        if (not model.net.backbone.pretrained_loaded or model.net.backbone.feature_dim != 768
                or any(p.requires_grad or p.dtype != torch.float32 for p in self.ptm.parameters())):
            raise ValueError('EASE pilot requires a frozen FP32 pretrained Base backbone')
        self.losses, self.predictions, self.epochs, self.prototype_examples = [], [], 0, 0
        self.checks = metadata['structural_checks'] = {
            'finite_loss_and_gradients': False, 'frozen_backbone_gradients_absent': False,
            'frozen_backbone_unchanged': False, 'finite_prototypes': False,
            'prototype_and_adapter_shapes': False, 'evaluation_predictions_match_accuracy': False,
            'dataset_matches_reference': False, 'training_protocol_complete': False}
        with recorder.phase('state_audit'):
            self.initial_hash = self.digest(self.ptm)
        metadata['initial_frozen_backbone_sha256'] = self.initial_hash

    def digest(self, module):
        import hashlib
        result = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            cpu = tensor.detach().cpu().contiguous()
            result.update(name.encode())
            result.update(str(tuple(cpu.shape)).encode())
            result.update(str(cpu.dtype).encode())
            result.update(cpu.numpy().tobytes())
        return result.hexdigest()

    def end_epoch(self):
        with self.recorder.phase('state_audit'):
            if any(p.grad is not None for p in self.ptm.parameters()):
                raise RuntimeError('Frozen EASE backbone received gradients')
            if any(not self.torch.isfinite(p).all().item()
                   or (p.grad is not None and not self.torch.isfinite(p.grad).all().item())
                   for p in self.model.net.parameters() if p.requires_grad):
                raise RuntimeError('Non-finite EASE adapter/proxy parameter or gradient')
        self.epochs += 1
        self.checks['finite_loss_and_gradients'] = bool(self.losses)
        self.checks['frozen_backbone_gradients_absent'] = True

    def end_task(self):
        with self.recorder.phase('state_audit'):
            net = self.model.net
            if (net.backbone.num_adapters != 1 or tuple(net.classifier.weight.shape) != (10, 768)
                    or not net.task_finalized.item() or any(p.requires_grad for p in net.backbone.parameters())):
                raise RuntimeError('EASE pilot adapter/prototype finalization is incomplete')
            if not self.torch.isfinite(net.classifier.weight).all().item():
                raise RuntimeError('EASE prototype table contains non-finite values')
            current_hash = self.digest(self.ptm)
            if current_hash != self.initial_hash:
                raise RuntimeError('Frozen EASE backbone changed during the pilot')
            from scripts.ease_reference_utils import dataset_fingerprint
            for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[0])):
                source = loader.dataset
                actual = dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
                self.recorder.data_snapshots['first_task'][split].update(actual)
                reference = self.metadata['reference_data_snapshots'][f'task1_{split}']
                keys = ('n_samples', 'class_counts', 'samples_ordered_sha256',
                        'samples_unordered_sha256', 'global_ids_ordered_sha256')
                if any(actual[key] != reference[key] for key in keys):
                    raise RuntimeError('EASE samples differ from the pinned official first task')
        self.metadata['final_frozen_backbone_sha256'] = current_hash
        self.metadata['prototype_adapter_image_pairs'] = self.prototype_examples
        self.checks.update(frozen_backbone_unchanged=True, finite_prototypes=True,
                           prototype_and_adapter_shapes=True, dataset_matches_reference=True)

    def complete(self):
        import numpy as np
        from torch.utils.data import SequentialSampler
        with self.recorder.phase('state_audit'):
            loader = self.dataset.test_loaders[0]
            if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
                raise RuntimeError('EASE saved predictions require the unchanged sequential test loader')
            labels, ids = np.asarray(loader.dataset.targets), np.asarray(loader.dataset.indexes)
            predicted = np.concatenate(self.predictions) if self.predictions else np.empty(0, dtype=np.int64)
            if len(predicted) != len(labels) or len(labels) != 1000:
                raise RuntimeError('EASE evaluation predictions are incomplete')
            accuracy = float((predicted == labels).mean() * 100)
            if abs(accuracy - self.metadata['accuracy'].get('mean_accuracy_pct', -1)) > 1e-9:
                raise RuntimeError('Saved EASE predictions disagree with the native class-IL accuracy')
            np.savez(self.recorder.output / 'task1_predictions.npz', predictions=predicted[:, None], labels=labels, global_ids=ids)
        self.checks['evaluation_predictions_match_accuracy'] = True
        valid = (self.epochs == 1 and len(self.losses) == 79 and self.prototype_examples == 5000
                 and self.recorder.steps == 79 and self.recorder.training_examples == 5000
                 and self.recorder.evaluation_examples == 1000 and self.recorder.tasks_completed == 1
                 and all(self.checks[name] for name in self.checks if name != 'training_protocol_complete'))
        if not valid:
            raise RuntimeError('EASE returned before the complete 79-step/5000-train/1000-eval pilot')
        self.checks['training_protocol_complete'] = True
        self.metadata['training_loss'] = {'batches': len(self.losses), 'mean': sum(self.losses) / len(self.losses),
                                          'minimum': min(self.losses), 'maximum': max(self.losses)}


@contextmanager
def instrument_ease(audit):
    """Chain the native meta wrappers and existing Recorder instrumentation."""
    from unittest.mock import patch
    model, recorder = audit.model, audit.recorder

    def observed(original):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if recorder.active_phase == 'train_epoch':
                loss = float(result['loss'] if isinstance(result, dict) else result)
                if not math.isfinite(loss):
                    raise RuntimeError('EASE loss is non-finite')
                audit.losses.append(loss)
            return result
        return wrapper

    def audited(original, callback):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            callback()
            return result
        return wrapper

    original_extract = model.net.backbone.extract_all

    @functools.wraps(original_extract)
    def extracted(inputs, *args, **kwargs):
        result = original_extract(inputs, *args, **kwargs)
        if recorder.active_phase == 'end_task':
            audit.prototype_examples += len(inputs)
        return result

    original_forward = model.forward

    @functools.wraps(original_forward)
    def forwarded(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if recorder.active_phase == 'evaluation':
            # Native evaluation masks logits in-place for task-IL after return.
            audit.predictions.append(result.detach().argmax(1).cpu().numpy())
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, 'meta_observe', observed(model.meta_observe)))
        stack.enter_context(patch.object(model, 'meta_end_epoch', audited(model.meta_end_epoch, audit.end_epoch)))
        stack.enter_context(patch.object(model, 'meta_end_task', audited(model.meta_end_task, audit.end_task)))
        stack.enter_context(patch.object(model.net.backbone, 'extract_all', extracted))
        stack.enter_context(patch.object(model, 'forward', forwarded))
        yield


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/cl_efficiency/manifest.json')
    result.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/prepared')
    result.add_argument('--models', default='all', help='all, or comma-separated model names; underscores and hyphens are equivalent')
    result.add_argument('--epochs', type=int, default=1, help='Outer-loop epochs; internal task-hook budgets remain in the manifest')
    result.add_argument('--batch-size', type=int, default=64)
    result.add_argument('--num-workers', type=int, default=4)
    result.add_argument('--seed', type=int, default=1993)
    result.add_argument('--gpu', default=os.getenv('CL_GPU', '0'), help='One physical NVIDIA index/UUID, or cpu; CUDA_VISIBLE_DEVICES is set for children')
    result.add_argument('--data-root', type=Path, default=Path(os.getenv('CL_DATA_ROOT', ROOT / 'data')),
                        help='Shared Mammoth dataset root; results remain in the run output directory')
    result.add_argument('--sample-interval', type=float, default=1.)
    result.add_argument('--timeout', type=float, default=7200., help='Timeout in seconds per pass')
    result.add_argument('--progress-interval', type=float, default=600., help='Slack heartbeat interval in seconds during a running pass')
    result.add_argument('--wandb-project', default=os.getenv('WANDB_PROJECT', 'cl-efficiency'))
    result.add_argument('--wandb-entity', default=os.getenv('WANDB_ENTITY'))
    result.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    result.add_argument('--slack', choices=('on', 'off'), default='on')
    result.add_argument('--flops', choices=('paired', 'off'), default='paired',
                        help='paired adds a separate diagnostic first-task pass; timings from that pass are excluded from comparison')
    result.add_argument('--execute', action='store_true', help='Actually launch experiments; absent by default')
    result.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    return result


def make_plan(args, entries):
    names = {entry['model'].replace('_', '-'): entry for entry in entries}
    requested = list(names) if args.models == 'all' else [name.strip().replace('_', '-') for name in args.models.split(',')]
    unknown = sorted(set(requested) - names.keys())
    if unknown:
        raise ValueError(f'Unknown models: {unknown}')
    jobs = []
    for name in dict.fromkeys(requested):
        entry = names[name]
        command = ['--model', entry['model'], '--dataset', entry['dataset'], '--model_config', 'default',
                   '--stop_after', '1', '--n_epochs', str(0 if entry['kind'] == 'inference' else args.epochs),
                   '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers), '--seed', str(args.seed),
                   '--permute_classes', '0', '--debug_mode', '0', '--enable_other_metrics', '0', '--eval_future', '0',
                   '--disable_log', '0', '--non_verbose', '1', '--save_after_interrupt', '0',
                   '--noise_rate', '0', '--label_perc', '1', '--label_perc_by_class', '1', '--validation', '0',
                   '--joint', '0', '--fitting_mode', 'epochs', '--base_path', str(args.data_root.resolve()),
                   '--device', 'cpu' if args.gpu == 'cpu' else 'cuda:0']
        for key, value in entry.get('args', {}).items():
            command.extend([f'--{key}', str(value)])
        jobs.append({**entry, 'main_args': command, 'seed': args.seed,
                     'passes': ['measurement', 'flops'] if args.flops == 'paired' else ['measurement'],
                     'status': 'deferred' if entry.get('deferred_reason') else 'planned_not_executed'})
    return jobs


def validate_manifest(entries):
    sources = {entry['model']: entry for entry in inventory(ROOT)}
    configured = {entry['model'].replace('_', '-'): entry for entry in entries}
    if len(configured) != len(entries) or set(sources) != set(configured):
        raise ValueError(f'Manifest must cover every source model once. Missing: {set(sources) - set(configured)}; extra: {set(configured) - set(sources)}')
    controlled = {'model', 'dataset', 'stop_after', 'start_from', 'n_epochs', 'batch_size', 'num_workers',
                  'seed', 'permute_classes', 'debug_mode', 'eval_future', 'disable_log', 'device', 'results_path',
                  'base_path', 'wandb_name', 'wandb_entity', 'wandb_project', 'enable_other_metrics', 'model_config',
                  'savecheck', 'save_after_interrupt', 'loadcheck'}
    controlled.update({'noise_rate', 'label_perc', 'label_perc_by_class', 'validation', 'joint', 'fitting_mode',
                       'custom_class_order', 'custom_task_order'})
    for entry in entries:
        if entry['dataset'] != COMMON_DATASET:
            raise ValueError(f'{entry["model"]}: all models must use {COMMON_DATASET}')
        collision = controlled.intersection(entry.get('args', {}))
        if collision:
            raise ValueError(f'{entry["model"]}: reserved benchmark arguments: {collision}')


def prepare(args):
    entries = json.loads(args.manifest.read_text())['models']
    validate_manifest(entries)
    jobs = make_plan(args, entries)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = readiness(args.gpu, args.wandb_mode, args.slack == 'on')
    devices['wandb_entity_configured'] = bool(args.wandb_entity)
    plan = {'schema_version': 1, 'experiment_executed': False,
            'protocol': {'dataset': COMMON_DATASET, 'tasks': 1, 'classes_per_task': 10, 'noise_rate': 0,
                         'outer_epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
                         'num_workers': args.num_workers, 'permute_classes': False,
                         'flops': args.flops, 'power_sample_interval_s': args.sample_interval,
                         'comparison_scope': 'Same CIFAR100 task/samples/labels; inspect raw sample hashes and effective_args. Backbones/pretraining/transforms differ.',
                         'forgetting': None, 'backward_transfer': None},
            'runtime': {'gpu': args.gpu, 'data_root': str(args.data_root.resolve()),
                        'cache_root': os.getenv('CL_CACHE_ROOT'), 'torch_home': os.getenv('TORCH_HOME'),
                        'hf_hub_cache': os.getenv('HF_HUB_CACHE')},
            'logging': {'wandb_project': args.wandb_project, 'wandb_entity': args.wandb_entity,
                        'wandb_mode': args.wandb_mode, 'slack': args.slack}, 'jobs': jobs}
    write_json(args.output / 'plan.json', plan)
    write_json(args.output / 'readiness.json', devices)
    write_json(args.output / 'inventory.json', inventory(ROOT))
    with (args.output / 'commands.txt').open('w') as file:
        for job in jobs:
            if job.get('deferred_reason'):
                file.write(f'# DEFERRED {job["model"]}: {job["deferred_reason"]}\n')
            else:
                file.write(shlex.join(['env', f'CL_GPU={args.gpu}', 'uv', 'run', '--no-sync', 'python', 'scripts/cl_runtime.py', '--',
                                       'python', 'main.py', *job['main_args']]) + '\n')
    return plan, devices


def execute_checks(args, ready):
    missing = [name for name in ('torch', 'torchvision', 'timm', 'open_clip', 'kornia', 'psutil') if not ready['packages'][name]]
    if missing:
        raise ValueError(f'Missing runtime packages: {missing}; use uv sync --locked --group dev --extra extra')
    if args.gpu != 'cpu' and ready['gpu_error']:
        raise ValueError('The selected NVIDIA GPU is unavailable; fix the driver/environment or explicitly select --gpu cpu')
    if args.wandb_mode != 'disabled' and (not ready['packages']['wandb'] or not args.wandb_entity):
        raise ValueError('W&B requires its SDK and --wandb-entity or WANDB_ENTITY. Cached login or WANDB_API_KEY is checked by W&B at execution.')
    if args.slack == 'on' and not ready['slack_webhook_valid']:
        raise ValueError('The worker environment needs a valid SLACK_WEBHOOK_URL, or explicitly select --slack off')


def stop_worker(process):
    """Terminate only this worker's new process group, including SDK subprocesses."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute(args, plan, ready):
    execute_checks(args, ready)
    # Never overwrite earlier runs. The dry-run plan is not a measured result.
    suite = 'cl-efficiency-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{os.getpid()}'
    output = args.output / suite
    output.mkdir()
    write_json(output / 'plan.json', plan)
    records, notification_status = [], []

    def notify(message):
        status = notify_slack(message, enabled=args.slack == 'on')
        notification_status.append({'message': message, 'status': status})
        write_json(output / 'notifications.json', notification_status)
        if status not in ('sent', 'disabled'):
            logging.warning('Slack notification status: %s', status)

    notify(f'CL benchmark started: {suite}; {len(plan["jobs"])} models; one task; FLOPs={args.flops}')
    try:
        for job in plan['jobs']:
            if job.get('deferred_reason'):
                records.append({'model': job['model'], 'group': job['group'], 'status': 'deferred', 'reason': job['deferred_reason']})
                continue
            result = {'model': job['model'], 'group': job['group'], 'dataset': job['dataset']}
            notify(f'CL {job["model"]}: starting first task ({job["dataset"]})')
            for pass_name in job['passes']:
                run_dir = output / job['model'] / pass_name
                run_dir.mkdir(parents=True)
                spec = {**job, 'pass': pass_name, 'output': str(run_dir), 'suite': suite,
                        'gpu': args.gpu, 'data_root': str(args.data_root.resolve()), 'sample_interval': args.sample_interval,
                        'wandb_mode': args.wandb_mode, 'wandb_entity': args.wandb_entity, 'wandb_project': args.wandb_project}
                write_json(run_dir / 'spec.json', spec)
                env = os.environ.copy()
                env.update(CL_EFFICIENCY_EXECUTE='1', CUDA_VISIBLE_DEVICES='' if args.gpu == 'cpu' else args.gpu,
                           WANDB_MODE=args.wandb_mode, WANDB_DIR=str(run_dir), WANDB_RUN_GROUP=suite)
                # Offline/local selection must win over any inherited W&B defaults.
                env.pop('MAMMOTH_TEST', None)
                env.pop('MAMMOTH_ARGS', None)
                env.pop('MAMMOTH_DEVICE', None)
                started = time.monotonic()
                timed_out = False
                with (run_dir / 'console.log').open('w') as console:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(run_dir / 'spec.json')],
                                               cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        next_progress = started + args.progress_interval
                        while process.poll() is None:
                            elapsed = time.monotonic() - started
                            if elapsed >= args.timeout:
                                timed_out = True
                                stop_worker(process)
                                break
                            if time.monotonic() >= next_progress:
                                notify(f'CL {job["model"]}: {pass_name} running, elapsed {elapsed / 60:.1f} min')
                                next_progress = time.monotonic() + args.progress_interval
                            time.sleep(1)
                    except BaseException:
                        stop_worker(process)
                        raise
                metrics_file = run_dir / 'metrics.json'
                record = json.loads(metrics_file.read_text()) if metrics_file.exists() else {'status': 'failed', 'reason': 'worker_did_not_write_metrics'}
                record.update(exit_code=process.returncode, timeout=timed_out, console_log=str(run_dir / 'console.log'))
                if timed_out:
                    record['status'] = 'timeout'
                elif process.returncode != 0 and record['status'] == 'completed':
                    record['status'] = 'failed'
                result[pass_name] = record
                if pass_name == 'measurement' and record['status'] != 'completed':
                    break
            result['status'] = result['measurement']['status']
            result['flops_status'] = result.get('flops', {}).get('status', 'not_collected')
            records.append(result)
            write_results(output, records)
            notify(f'CL {job["model"]}: {result["status"]}, FLOPs={result["flops_status"]}; W&B {result["measurement"].get("wandb_url") or "unavailable"}')
    except BaseException:
        write_results(output, records)
        notify(f'CL benchmark interrupted: {suite}; partial local results: {output}')
        raise
    write_results(output, records)
    failures = [r['model'] for r in records if r['status'] not in ('completed', 'deferred')
                or r.get('flops_status') not in (None, 'completed', 'not_collected')]
    notify(f'CL benchmark finished: {suite}; failures={failures}; local results: {output}')
    return 1 if failures else 0


def write_results(output, records):
    write_json(output / 'results.json', records)
    rows = []
    for result in records:
        row = {k: result.get(k) for k in ('model', 'group', 'dataset', 'status', 'reason', 'flops_status')}
        measured = result.get('measurement', {})
        row.update(flatten(measured.get('metrics', {}), 'efficiency'))
        row.update(flatten(measured.get('accuracy', {}), 'accuracy'))
        row.update(wandb_url=measured.get('wandb_url'), flops_partial=result.get('flops', {}).get('flops', {}).get('supported_ops_total'))
        first_task = measured.get('data_snapshots', {}).get('first_task', {})
        for split in ('train', 'test'):
            row[f'{split}_data_sha256'] = first_task.get(split, {}).get('indices_labels_sha256')
        rows.append(row)
    with (output / 'results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def finish_worker_logging(output, spec, record, wandb, real_finish, code):
    """Keep local computation results even when upload or SDK shutdown fails."""
    def failed(status):
        record['wandb_logging_status'] = status
        if record['computation_status'] == 'completed':
            record['status'] = 'logging_failed'

    if wandb is not None and wandb.run is not None:
        try:
            logged = {key: value for key, value in flatten({'efficiency': record['metrics'], 'phases': record['phases'],
                      'snapshots': record['model_snapshots'], 'flops': record.get('flops', {}), 'accuracy': record['accuracy']}).items()
                      if isinstance(value, (float, int)) and not isinstance(value, bool)}
            wandb.log(logged)
            wandb.run.summary.update({**logged, 'benchmark_status': record['status'], 'benchmark_pass': spec['pass']})
            artifact = wandb.Artifact(f'{spec["suite"]}-{spec["model"]}-{spec["pass"]}', type='cl-efficiency')
            for name in ('metrics.json', 'effective_args.json', 'telemetry.csv', 'phases.json', 'flops_ops.json',
                         'spec.json', 'task1_predictions.npz'):
                if (output / name).exists():
                    artifact.add_file(str(output / name), name=name)
            if (output / 'mammoth').exists():
                artifact.add_dir(str(output / 'mammoth'), name='mammoth')
            saved = wandb.run.log_artifact(artifact)
            if spec['wandb_mode'] == 'online':
                saved.wait()
                record['wandb_logging_status'] = 'uploaded'
            else:
                record['wandb_logging_status'] = 'offline_saved'
        except Exception as error:
            failed(f'failed:{type(error).__name__}')
            code = 1
        finally:
            wandb.finish = real_finish
            try:
                real_finish(exit_code=code)
            except Exception as error:
                failed(f'finish_failed:{type(error).__name__}')
                code = 1
    else:
        record['wandb_logging_status'] = 'disabled' if spec['wandb_mode'] == 'disabled' else 'run_not_initialized'
        if spec['wandb_mode'] != 'disabled':
            failed('run_not_initialized')
            code = 1
        if wandb is not None and real_finish is not None:
            wandb.finish = real_finish
    return code


def worker(spec_path):
    if os.getenv('CL_EFFICIENCY_EXECUTE') != '1':
        raise ValueError('Worker execution requires explicit benchmark --execute')
    spec = json.loads(spec_path.read_text())
    output = Path(spec['output'])
    recorder = Recorder(output, spec['gpu'], spec['sample_interval'])
    recorder.start()
    metadata = {'model': spec['model'], 'dataset': spec['dataset'], 'group': spec['group'], 'kind': spec['kind'],
                'pass': spec['pass'], 'suite': spec['suite'], 'status': 'failed', 'accuracy': {},
                'forgetting': None, 'backward_transfer': None, 'wandb_url': None}
    profiler, real_finish, wandb, patch_context = None, None, None, None
    profiler_closed = False
    ease_audit = None
    baseline_audit = None
    code = 1
    try:
        ease_pin = ease_execution_pin() if spec['model'] == 'ease' else None
        baseline_pin = None
        if spec.get('baseline_pilot'):
            from scripts.baseline_audit import execution_pin, BaselineAudit, instrument_baseline, require_initialization_args
            baseline_pin = execution_pin(spec)
            metadata.update(baseline_pin)
        if ease_pin is not None:
            if spec['gpu'] != '1':
                raise ValueError('The approved EASE pilot requires physical GPU1')
            metadata.update(ease_pin)
        import torch
        if ease_pin is not None or baseline_pin is not None:
            metadata['precision'] = _configure_ease_precision(torch)
        import main as mammoth
        import utils.training as training
        if spec['wandb_mode'] != 'disabled':
            import wandb
        recorder.torch = torch
        recorder.device = torch.device('cpu' if spec['gpu'] == 'cpu' else 'cuda:0')
        packages = ['torch', 'torchvision', 'timm'] + (['wandb'] if wandb is not None else [])
        metadata['versions'] = {package: importlib.metadata.version(package) for package in packages}
        metadata['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        metadata['git_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        command = [*spec['main_args'], '--results_path', str(output / 'mammoth'),
                   '--wandb_name', f'{spec["model"]}-{spec["pass"]}', '--wandb_project', spec['wandb_project']]
        if '--base_path' not in command and not any(arg.startswith('--base_path=') for arg in command):
            command.extend(['--base_path', str(spec.get('data_root') or os.getenv('CL_DATA_ROOT') or ROOT / 'data')])
        if spec['wandb_entity']:
            command.extend(['--wandb_entity', spec['wandb_entity']])
        if ease_pin is not None:
            _pinned_argument(command, '--ease_checkpoint', ease_pin['checkpoint_file'])
            _pinned_argument(command, '--code_optimization', '0')
        if baseline_pin is not None:
            _pinned_argument(command, '--pretrained_checkpoint', baseline_pin['checkpoint_file'])
            _pinned_argument(command, '--code_optimization', '0')
        # Mammoth checks sys.argv in several parser helpers, even when cmd is supplied.
        sys.argv = ['main.py', *command]
        args = mammoth.parse_args(cmd=command, verbose=True)
        if baseline_pin is not None:
            require_initialization_args(args, metadata)
        original_initialize = mammoth.initialize
        from unittest.mock import patch
        patch_context = ExitStack()

        original_write = training.Logger.write

        def write(logger, *args, **kwargs):
            metadata['accuracy'] = capture_accuracy(logger, metadata.get('compatibility'))
            return original_write(logger, *args, **kwargs)

        patch_context.enter_context(patch.object(training.Logger, 'write', write))

        def initialize(*args, **kwargs):
            nonlocal ease_audit, baseline_audit
            with recorder.phase('initialize'):
                model, dataset, effective_args = original_initialize(*args, **kwargs)
            if spec['wandb_mode'] == 'disabled':
                effective_args.nowand = 1
            elif effective_args.nowand:
                raise ValueError('W&B unexpectedly disabled; provide a project and entity')
            if effective_args.stop_after != 1 or effective_args.debug_mode or effective_args.eval_future:
                raise ValueError('Effective arguments violate the first-task measurement protocol')
            if (effective_args.dataset != COMMON_DATASET or dataset.NAME != COMMON_DATASET
                    or dataset.N_CLASSES != 100 or dataset.N_TASKS != 10 or dataset.N_CLASSES_PER_TASK != 10
                    or effective_args.noise_rate != 0 or effective_args.validation
                    or effective_args.label_perc != 1 or effective_args.label_perc_by_class != 1):
                raise ValueError('Effective dataset violates the shared CIFAR100 task/sample/label protocol')
            write_json(output / 'effective_args.json', vars(effective_args))
            metadata['effective_args'] = vars(effective_args).copy()
            metadata['compatibility'] = getattr(model, 'COMPATIBILITY', spec.get('compatibility'))
            recorder.snapshots['after_initialize'] = model_snapshot(model)
            patch_context.enter_context(instrument(recorder, model, dataset, training, spec.get('clean_true_labels', False)))
            if ease_pin is not None:
                ease_audit = EaseAudit(model, dataset, effective_args, recorder, metadata, torch)
                patch_context.enter_context(instrument_ease(ease_audit))
            if baseline_pin is not None:
                baseline_audit = BaselineAudit(model, dataset, effective_args, recorder, metadata, torch)
                patch_context.enter_context(instrument_baseline(baseline_audit))
            return model, dataset, effective_args

        patch_context.enter_context(patch.object(mammoth, 'initialize', initialize))
        # Keep the native run open until both native accuracy and telemetry are saved.
        if wandb is not None:
            real_finish = wandb.finish
            wandb.finish = lambda *args, **kwargs: None
        if spec['pass'] == 'flops':
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True)
            profiler.__enter__()
        try:
            mammoth.main(args)
        finally:
            patch_context.close()
            if profiler is not None:
                profiler.__exit__(None, None, None)
                profiler_closed = True
        if recorder.tasks_completed != 1:
            raise RuntimeError('First-task evaluation did not complete; run may have been interrupted')
        if ease_pin is not None and ease_audit is None:
            raise RuntimeError('EASE initialization did not install the required pilot audits')
        if ease_audit is not None:
            ease_audit.complete()
        if baseline_pin is not None and baseline_audit is None:
            raise RuntimeError('Baseline initialization did not install its required audits')
        if baseline_audit is not None:
            baseline_audit.complete()
        metadata['status'], code = 'completed', 0
    except BaseException as error:
        metadata['error_type'] = type(error).__name__
        # Full traceback stays in console.log; never serialize secret environment values.
        logging.exception('Benchmark worker failed')
    finally:
        if patch_context is not None:
            patch_context.close()
        if profiler is not None and profiler_closed:
            try:
                operations = [{'operation': event.key, 'count': event.count, 'flops': event.flops or 0}
                              for event in profiler.key_averages() if event.flops]
                metadata['flops'] = {'supported_ops_total': sum(op['flops'] for op in operations) or None,
                                     'scope': 'Partial supported PyTorch operator estimate over initialization, train forward/backward, task-end prototypes, evaluation and diagnostic audits in this separate pass.',
                                     'limitations': 'Unsupported/fused operators, optimizer and linear algebra may be omitted; phase FLOPs are not attributed. Not training-only FLOPs. No backward x3 extrapolation.'}
                write_json(output / 'flops_ops.json', operations)
            except Exception as error:
                metadata['flops_error'] = type(error).__name__
                metadata['status'], code = 'failed', 1
        if wandb is not None and wandb.run is not None:
            metadata['wandb_url'] = wandb.run.url if spec['wandb_mode'] == 'online' else None
        metadata['computation_status'] = metadata['status']
        record = recorder.save(metadata)
        code = finish_worker_logging(output, spec, record, wandb, real_finish, code)
        write_json(output / 'metrics.json', record)
    return code


def main(argv=None):
    configure_environment()
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.worker is not None:
        return worker(args.worker)
    if args.epochs < 1 or args.batch_size < 1 or args.sample_interval <= 0 or args.timeout <= 0 or args.progress_interval < 60:
        raise ValueError('Epochs/batch size/timing intervals must be positive; progress interval must be at least 60 seconds')
    plan, ready = prepare(args)
    logging.info('Prepared %d models (%d deferred) at %s; experiment execution=%s', len(plan['jobs']),
                 sum(bool(job.get('deferred_reason')) for job in plan['jobs']), args.output, args.execute)
    return execute(args, plan, ready) if args.execute else 0


if __name__ == '__main__':
    sys.exit(main())
