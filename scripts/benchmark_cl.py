"""Prepare or explicitly execute sequential, one-task Mammoth efficiency benchmarks.

The default is a dry run: no model imports, training, downloads, W&B writes or Slack posts.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.metadata
import json
import logging
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
    code = 1
    try:
        import torch
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
        # Mammoth checks sys.argv in several parser helpers, even when cmd is supplied.
        sys.argv = ['main.py', *command]
        args = mammoth.parse_args(cmd=command, verbose=True)
        original_initialize = mammoth.initialize
        from contextlib import ExitStack
        from unittest.mock import patch
        patch_context = ExitStack()

        original_write = training.Logger.write

        def write(logger, *args, **kwargs):
            metadata['accuracy'] = capture_accuracy(logger, metadata.get('compatibility'))
            return original_write(logger, *args, **kwargs)

        patch_context.enter_context(patch.object(training.Logger, 'write', write))

        def initialize(*args, **kwargs):
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
                                     'scope': 'Partial PyTorch operator estimate for this diagnostic pass; supported matrix multiplication/convolution only.',
                                     'limitations': 'Fused attention, unsupported operators, optimizer and linear algebra may be omitted. No backward x3 extrapolation.'}
                write_json(output / 'flops_ops.json', operations)
            except Exception as error:
                metadata['flops_error'] = type(error).__name__
                metadata['status'], code = 'failed', 1
        if wandb is not None and wandb.run is not None:
            metadata['wandb_url'] = wandb.run.url if spec['wandb_mode'] == 'online' else None
        record = recorder.save(metadata)
        if wandb is not None and wandb.run is not None:
            try:
                logged = {key: value for key, value in flatten({'efficiency': record['metrics'], 'phases': record['phases'],
                          'snapshots': record['model_snapshots'], 'flops': record.get('flops', {}), 'accuracy': record['accuracy']}).items()
                          if isinstance(value, (float, int)) and not isinstance(value, bool)}
                wandb.log(logged)
                wandb.run.summary.update({**logged, 'benchmark_status': record['status'], 'benchmark_pass': spec['pass']})
                artifact = wandb.Artifact(f'{spec["suite"]}-{spec["model"]}-{spec["pass"]}', type='cl-efficiency')
                for name in ('metrics.json', 'effective_args.json', 'telemetry.csv', 'phases.json', 'flops_ops.json', 'spec.json'):
                    if (output / name).exists():
                        artifact.add_file(str(output / name), name=name)
                if (output / 'mammoth').exists():
                    artifact.add_dir(str(output / 'mammoth'), name='mammoth')
                wandb.run.log_artifact(artifact)
                record['wandb_logging_status'] = 'offline_saved' if spec['wandb_mode'] == 'offline' else 'submitted'
            except Exception as error:
                record['wandb_logging_status'] = f'failed:{type(error).__name__}'
                code = 1
            finally:
                wandb.finish = real_finish
                try:
                    real_finish(exit_code=code)
                except Exception as error:
                    record['wandb_logging_status'] = f'finish_failed:{type(error).__name__}'
                    code = 1
        else:
            record['wandb_logging_status'] = 'disabled' if spec['wandb_mode'] == 'disabled' else 'run_not_initialized'
            if real_finish is not None:
                wandb.finish = real_finish
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
