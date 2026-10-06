"""Opt-in matched two-task official EASE reference, with local/W&B/Slack logs.

The reference algorithm is loaded from an immutable commit in a separate worker.
Only dataset location/order, workers and the exact pretrained file are adapted.
This calibration run is separate from the one-task efficiency protocol.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, flatten, model_snapshot, notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.check_ease_runtime import CHECKPOINT_NAME, CHECKPOINT_SHA256, file_hash

REFERENCE_COMMIT = '6d67508ed8b7c4d23350ed4eb28c123914b40ae3'


def isolate_reference_imports(source):
    """Select upstream namespace packages without native package collisions."""
    if any(name in sys.modules for name in ('utils', 'models', 'backbone')):
        raise RuntimeError('The reference requires a fresh worker without native framework imports')
    # Upstream directories are namespace packages without __init__.py. An
    # earlier namespace path does not win over a later native regular package;
    # remove every native root entry, including cwd's empty-string entry.
    sys.path[:] = [str(Path(source).resolve()), *[entry for entry in sys.path
                  if Path(entry or os.getcwd()).resolve() != ROOT]]


def stop(process):
    from scripts.benchmark_cl import stop_worker
    stop_worker(process)


def execute(options, profile):
    source = options.reference_source.resolve()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True,
                            text=True, check=True).stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError('Official reference source must match the reviewed commit')
    if subprocess.run(['git', 'status', '--porcelain'], cwd=source, capture_output=True,
                      text=True, check=True).stdout.strip():
        raise ValueError('Official reference source must be clean')
    checkpoint = Path(profile['cache_root']) / 'torch/hub/checkpoints' / CHECKPOINT_NAME
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Selected Base checkpoint does not match the pinned hash')
    name = 'ease-official-reference-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = options.output.resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    copied = output / 'source'
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns('.git', 'logs', '__pycache__'))
    config = json.loads((source / 'exps/ease_cifar.json').read_text())
    config.update(shuffle=False, init_cls=10, increment=10, device=['0'], seed=1993,
                  init_epochs=20, later_epochs=20, batch_size=48, ffn_num=64)
    spec = {'name': name, 'stage': 'official_matched_reference', 'source': str(copied),
            'reference_commit': commit, 'source_repository': 'https://github.com/sun-hailong/CVPR24-Ease',
            'output': str(output), 'gpu': profile['gpu'], 'data_root': profile['data_root'],
            'checkpoint': str(checkpoint), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'tasks': 2, 'num_workers': 4, 'config': config, 'paper_reproduction': False,
            'port_acceptance_evaluated': False, 'accuracy_margin_pp': None,
            'wandb_entity': profile['wandb_entity'], 'wandb_project': profile['wandb_project'],
            'allocator_environment': os.getenv('JE_ARROW_MALLOC_CONF'),
            'efficiency_protocol': False, 'flops_measured': False}
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        notifications.append({'message': message, 'status': notify_slack(message)})
        write_json(output / 'notifications.json', notifications)

    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=profile['gpu'], CL_EASE_REFERENCE_EXECUTE='1',
                       HF_HUB_OFFLINE='1', WANDB_MODE='online', WANDB_DIR=str(output))
    notify(f'EASE official reference started: Base/adapter64; first 2 tasks; 20 epochs/task; GPU {profile["gpu"]}')
    started, last_notification = time.monotonic(), time.monotonic()
    timed_out = False
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
                                   cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= options.timeout:
                    timed_out = True
                    stop(process)
                    notify('EASE official reference timed out; partial local results preserved')
                    break
                if time.monotonic() - last_notification >= 600:
                    progress_file = output / 'progress.json'
                    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
                    notify(f'EASE reference running: {progress.get("message", "initializing")}; elapsed {(time.monotonic() - started)/60:.1f} min')
                    last_notification = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop(process)
            notify('EASE official reference interrupted; partial local results preserved')
            raise
    metrics = json.loads((output / 'metrics.json').read_text()) if (output / 'metrics.json').exists() else {'status': 'failed'}
    if timed_out:
        metrics['status'] = 'timeout'
    elif process.returncode != 0 and metrics['status'] == 'completed':
        metrics['status'] = 'failed'
    write_json(output / 'metrics.json', metrics)
    notify(f'EASE official reference {metrics["status"]}; logging={metrics.get("wandb_logging_status", "unavailable")}; W&B {metrics.get("wandb_url") or "unavailable"}; local results preserved')
    write_json(output / 'completion.json', {'exit_code': process.returncode, 'status': metrics['status'],
                                           'timeout': timed_out, 'wandb_logging_status': metrics.get('wandb_logging_status'),
                                           'wall_s': time.monotonic() - started})
    print(json.dumps({'status': metrics['status'], 'output': str(output), 'wandb_url': metrics.get('wandb_url')}))
    return 0 if process.returncode == 0 and metrics['status'] == 'completed' else 1


def worker(spec_file):
    if os.getenv('CL_EASE_REFERENCE_EXECUTE') != '1':
        raise ValueError('Reference worker requires explicit --execute')
    spec = json.loads(spec_file.read_text())
    if spec['tasks'] != 2 or spec['config']['seed'] != 1993:
        raise ValueError('Reference protocol must remain the approved two-task seed1993 run')
    output = Path(spec['output'])
    recorder = Recorder(output, spec['gpu'])
    recorder.start()
    record = {**spec, 'status': 'failed', 'tasks_results': [], 'epoch_results': [], 'accuracy': {}}
    code, wandb, stack = 1, None, None
    try:
        import torch
        import numpy as np
        import timm
        import wandb
        from scripts.ease_reference_utils import install_reference_patches, dataset_fingerprint, shared_cifar_metadata
        # The two frameworks share top-level package names. This fresh worker
        # has imported no native models/backbone/utils before selecting source.
        isolate_reference_imports(spec['source'])
        from utils import data, data_manager
        from backbone import vit_ease
        from models import ease
        from trainer import _set_random, _set_device
        modules = {'timm': timm, 'data': data, 'data_manager': data_manager, 'ease': ease, 'vit': vit_ease}
        patch_records = {}
        stack = install_reference_patches(modules, spec['checkpoint'], spec['data_root'], patch_records)
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        torch.set_num_threads(2)
        config = spec['config'].copy()
        _set_random(config['seed'])
        _set_device(config)
        record['versions'] = {name: importlib.metadata.version(name) for name in
                              ('torch', 'torchvision', 'timm', 'numpy', 'pyarrow', 'easydict', 'wandb')}
        record['precision'] = {'matmul_precision': torch.get_float32_matmul_precision(),
                               'tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
                               'autocast': False, 'parameter_dtype': 'float32'}
        wandb.init(entity=spec['wandb_entity'], project=spec['wandb_project'], name=spec['name'],
                   group='ease-base-validation', job_type='official-reference', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        write_json(output / 'progress.json', {'message': 'loading official matched reference', 'wandb_url': record['wandb_url']})
        with recorder.phase('data_loading'):
            manager = data_manager.DataManager('cifar224', False, 1993, 10, 10, config)
        config['nb_classes'], config['nb_tasks'] = manager.nb_classes, manager.nb_tasks
        write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
        with recorder.phase('initialize'):
            learner = ease.Learner(config)
        assert learner._network.backbone.out_dim == 768
        record['runtime_patches'] = patch_records
        write_json(output / 'runtime_patches.json', patch_records)

        original_progress = ease.tqdm

        class EpochProgress:
            def __init__(self, iterable):
                self.progress = original_progress(iterable, disable=True)

            def __iter__(self):
                for epoch in self.progress:
                    with recorder.phase('train_epoch'):
                        yield epoch

            def set_description(self, message):
                self.progress.set_description(message)
                match = re.search(r'Task (\d+), Epoch (\d+)/(\d+) => Loss ([0-9.eE+-]+), Train_accy ([0-9.eE+-]+)', message)
                if match is None:
                    raise RuntimeError('Official epoch progress format changed')
                task, epoch, epochs = (int(match[index]) for index in (1, 2, 3))
                row = {'task': task + 1, 'epoch': epoch, 'epochs': epochs,
                       'loss': float(match[4]), 'train_accuracy_pct': float(match[5])}
                record['epoch_results'].append(row)
                write_json(output / 'epoch_results.json', record['epoch_results'])
                write_json(output / 'progress.json', {'message': message, **row, 'wandb_url': record['wandb_url']})
                wandb.log({'reference/train_loss': row['loss'], 'reference/train_accuracy_pct': row['train_accuracy_pct'],
                           'reference/task': row['task'], 'reference/epoch': epoch})
                print(message, flush=True)

        from unittest.mock import patch
        stack.enter_context(patch.object(ease, 'tqdm', EpochProgress))
        original_replace = learner.replace_fc

        def replace(loader):
            with recorder.phase('prototype_completion'):
                return original_replace(loader)

        learner.replace_fc = replace
        optimizer_slot = {}
        original_optimizer = learner.get_optimizer

        def get_optimizer(*arguments, **keywords):
            optimizer = original_optimizer(*arguments, **keywords)
            optimizer_slot['current'] = optimizer
            return optimizer

        learner.get_optimizer = get_optimizer

        def count_forward(module, arguments, keywords, result):
            if not keywords.get('test', False):
                recorder.training_examples += len(arguments[0])
                recorder.steps += 1

        handle = learner._network.register_forward_hook(count_forward, with_kwargs=True)
        original_eval = learner._eval_cnn
        latest_predictions = {}

        def evaluate_predictions(loader):
            predictions, labels = original_eval(loader)
            # Upstream calls .forward directly during evaluation, bypassing
            # nn.Module hooks. Count the returned labels from that same pass.
            recorder.evaluation_examples += len(labels)
            latest_predictions.update(predictions=predictions, labels=labels)
            return predictions, labels

        learner._eval_cnn = evaluate_predictions
        for task in range(2):
            with recorder.phase('incremental_train'):
                learner.incremental_train(manager)
            for name, dataset in [('train', learner.train_dataset), ('test', learner.test_dataset)]:
                recorder.data_snapshots[f'task{task+1}_{name}'] = dataset_fingerprint(
                    dataset.images, dataset.labels, dataset.global_ids, dataset.trsf)
            with recorder.phase('evaluation'):
                accuracy, _ = learner.eval_task()
            predictions, labels = latest_predictions['predictions'][:, 0], latest_predictions['labels']
            old, new = labels < task * 10, labels >= task * 10
            correct = predictions == labels
            row = {'task': task + 1, 'seen_classes': (task + 1) * 10,
                   'seen_accuracy_pct': float(correct.mean() * 100),
                   'old_accuracy_pct': float(correct[old].mean() * 100) if old.any() else None,
                   'new_accuracy_pct': float(correct[new].mean() * 100),
                   'official_accuracy': accuracy, 'prototype_shape': list(learner._network.fc.weight.shape),
                   'observed_adapters': len(learner._network.backbone.adapter_list) + 1}
            assert row['observed_adapters'] == task + 1
            assert row['prototype_shape'] == [(task + 1) * 10, (task + 1) * 768]
            record['tasks_results'].append(row)
            record['accuracy'] = row
            recorder.tasks_completed += 1
            from types import SimpleNamespace
            snapshot_subject = SimpleNamespace(parameters=learner._network.parameters,
                                               buffers=learner._network.buffers,
                                               optimizer=optimizer_slot['current'])
            recorder.snapshots[f'task{task+1}_before_after_task'] = model_snapshot(snapshot_subject)
            record.setdefault('unregistered_old_adapter_parameters', {})[f'task{task+1}'] = sum(
                p.numel() for adapter in learner._network.backbone.adapter_list for p in adapter.parameters())
            np.savez(output / f'task{task+1}_predictions.npz', predictions=latest_predictions['predictions'], labels=labels)
            # Preserve the reference's unregistered old adapters explicitly.
            torch.save({'model': learner._network.state_dict(), 'old_adapters': [a.state_dict() for a in
                        learner._network.backbone.adapter_list], 'task': task + 1, 'checkpoint_sha256': spec['checkpoint_sha256']},
                       output / f'task{task+1}_state.pt')
            write_json(output / 'task_results.json', record['tasks_results'])
            wandb.log({f'reference/{key}': value for key, value in row.items() if isinstance(value, (float, int))})
            status = notify_slack(f'EASE official reference task {task+1}/2 complete: seen accuracy {row["seen_accuracy_pct"]:.2f}%; W&B {record["wandb_url"]}')
            record.setdefault('task_notifications', []).append({'task': task + 1, 'status': status})
            if task == 0:
                with recorder.phase('after_task'):
                    learner.after_task()
        handle.remove()
        record.update(status='completed', acceptance_status='awaiting_accuracy_margin_decision',
                      accuracy_margin_pp=None, paper_reproduction=False)
        code = 0
    except BaseException as error:
        logging.exception('Official EASE reference failed')
        record['error_type'] = type(error).__name__
    finally:
        if stack is not None:
            stack.close()
        if 'runtime_patches' in record:
            write_json(output / 'runtime_patches.json', record['runtime_patches'])
        record['computation_status'] = record['status']
        record = recorder.save(record)
        if wandb is not None and wandb.run is not None:
            try:
                scalars = {key: value for key, value in flatten({'reference': record['accuracy'],
                           'reference_resources': record['metrics']}).items() if isinstance(value, (float, int))}
                wandb.run.summary.update({**scalars, 'reference_status': record['status'],
                                          'port_acceptance_evaluated': False, 'paper_reproduction': False})
                artifact = wandb.Artifact(spec['name'], type='cl-reference-calibration')
                for filename in ('spec.json', 'metrics.json', 'runtime_patches.json', 'epoch_results.json',
                                 'task_results.json', 'shared_data.json', 'telemetry.csv', 'phases.json',
                                 'task1_predictions.npz', 'task2_predictions.npz'):
                    if (output / filename).is_file():
                        artifact.add_file(str(output / filename), name=filename)
                wandb.run.log_artifact(artifact).wait()
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record['wandb_logging_status'] = f'failed:{type(error).__name__}'
                record['status'] = 'logging_failed'
                code = 1
            finally:
                try:
                    wandb.finish(exit_code=code)
                except Exception as error:
                    record['wandb_logging_status'] = f'finish_failed:{type(error).__name__}'
                    record['status'], code = 'logging_failed', 1
        write_json(output / 'metrics.json', record)
    return code


def main(argv=None):
    profile = configure_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reference-source', type=Path, default=Path('/tmp/cl-ease-official-20261007'))
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/ease/runs')
    parser.add_argument('--timeout', type=float, default=7200.)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if options.worker:
        return worker(options.worker)
    if not options.execute:
        print('Prepared official EASE Base two-task reference; execution=False')
        return 0
    return execute(options, profile)


if __name__ == '__main__':
    sys.exit(main())
