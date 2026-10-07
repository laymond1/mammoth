"""Explicitly gated first-two-task native validation through Mammoth's trainer.

Preparation imports only Python's standard library and starts no work. Execution
requires a complete independent author run, its learned initialization and the
approved local weights. Resource observations are validation provenance only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime
import functools
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from weakref import WeakKeyDictionary, WeakSet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, flatten, instrument, notify_slack, valid_slack_webhook, write_json
from scripts.cl_runtime import configure_environment
from scripts.hide_protocol import forbid_cifar_download, validate_shared_data
from scripts.low_rank_protocol import (CANONICAL_PRETRAINED_SHA256, CHECKPOINT, CHECKPOINT_SHA256,
    METHODS, PRECISION, REFERENCE_COMMIT, REQUIRED_STRUCTURAL_CHECKS, audit_method_state, canonical_pretrained_digest,
    completion_checks, configure_epoch_loader, configure_phase_loader_once, counts_from_predictions, evaluate_acceptance, file_hash,
    learned_state, load_initial_state, load_plan, native_command, protocol, seed_phase, source_hashes,
    tensor_digest, validate_prediction_archive)


def reference_record(path):
    document = json.loads(Path(path).read_text())
    return document.get('official_reference', document)


def update_wandb_summary(run, record):
    """Send summary mappings through the SDK's positional dictionary API."""
    values = {key: value for key, value in flatten({'native_counts': record['counts'],
              'native_acceptance': record.get('acceptance', {})}).items() if isinstance(value, (int, float, bool))}
    values.update(native_status=record['status'], acceptance_status=record['acceptance_status'], efficiency_comparison=False)
    for row in record['task_results']:
        values.update({f'task{row["task"]}/{key}': row[key]
                       for key in ('seen_correct', 'seen_samples', 'old_correct', 'old_samples', 'new_correct', 'new_samples')})
    run.summary.update(values)


def validate_spec(spec):
    method = spec.get('method')
    if (method not in METHODS or spec.get('protocol') != protocol(method) or spec.get('tasks') != 2
            or spec.get('mode') != 'validation' or spec.get('gpu') != '1'
            or spec.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or spec.get('canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256
            or spec.get('reference_commit') != REFERENCE_COMMIT
            or spec.get('wandb_entity') != 'laymond1' or spec.get('wandb_project') != 'cl-efficiency'
            or spec.get('wandb_mode') != 'online'):
        raise ValueError('Native low-rank settings differ from the approved first-two-task GPU1 protocol')


def validate_reference(reference, method, initial_hash):
    if (reference.get('status') != 'completed' or reference.get('method') != method
            or reference.get('protocol') != protocol(method) or reference.get('mode') != 'validation'
            or reference.get('reference_commit') != REFERENCE_COMMIT
            or reference.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or reference.get('canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256
            or reference.get('initial_state_sha256') != initial_hash
            or reference.get('precision') != PRECISION
            or reference.get('implementation') != 'pinned_author_common_input_reference'
            or reference.get('reference_source_sha256') != load_plan()['reference_source_sha256']
            or not reference.get('wandb_url') or reference.get('wandb_logging_status') != 'uploaded'
            or any(reference.get('structural_checks', {}).get(name) is not True for name in REQUIRED_STRUCTURAL_CHECKS)):
        raise ValueError('Native validation requires a passing independent reference with identical source, inputs and initialization')
    report = evaluate_acceptance(reference.get('task_results', []), reference.get('task_results', []),
                                 structural_checks=reference['structural_checks'])
    if report['status'] != 'pass':
        raise ValueError('Independent reference exact counts or structural gates do not pass')


def prediction_pins(reference, summary):
    files = reference.get('prediction_files', [])
    if len(files) != 2 or {item.get('task') for item in files} != {1, 2}:
        raise ValueError('Independent reference requires both full prediction archives')
    result = []
    for item in sorted(files, key=lambda row: row['task']):
        path = Path(item['path'])
        path = path if path.is_absolute() else Path(summary).parent / path
        if file_hash(path) != item.get('sha256'):
            raise ValueError('Independent reference predictions differ from their frozen hash pins')
        counts = validate_prediction_archive(path, item['task'])
        row = next(row for row in reference['task_results'] if row['task'] == item['task'])
        if any(row.get(key) != value for key, value in counts.items()):
            raise ValueError('Independent reference counts disagree with full saved predictions')
        result.append({'task': item['task'], 'path': str(path.resolve()), 'sha256': item['sha256']})
    return result


def build_spec(options, profile):
    load_plan()
    if (profile['gpu'] != '1' or profile['wandb_entity'] != 'laymond1'
            or profile['wandb_project'] != 'cl-efficiency'):
        raise ValueError('Low-rank validation requires GPU1 and the approved W&B destination')
    if options.reference_summary is None or options.initial_state is None:
        raise ValueError('Native validation requires --reference-summary and --initial-state')
    initial, summary, checkpoint = (Path(value).resolve() for value in
                                    (options.initial_state, options.reference_summary, options.checkpoint))
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Low-rank validation requires the pinned local Base AugReg NPZ')
    initial_hash = file_hash(initial)
    reference = reference_record(summary)
    validate_reference(reference, options.method, initial_hash)
    predictions = prediction_pins(reference, summary)
    data = validate_shared_data(profile['data_root'])
    name = options.method + '-native-first2-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (options.result_dir or options.output / name).resolve()
    spec = {'schema_version': 1, 'name': name, 'method': options.method, 'mode': 'validation', 'tasks': 2,
            'implementation': 'mammoth_native', 'protocol': protocol(options.method), 'gpu': '1',
            'checkpoint_file': str(checkpoint), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256, 'reference_commit': REFERENCE_COMMIT,
            'initial_state': str(initial), 'initial_state_sha256': initial_hash,
            'reference_summary': str(summary), 'reference_summary_sha256': file_hash(summary),
            'reference_prediction_files': predictions, 'data_root': profile['data_root'], 'data_files': data,
            'output': str(output), 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'wandb_mode': 'online',
            'execution_source_sha256': source_hashes(), 'paper_reproduction': False,
            'efficiency_comparison': False, 'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                capture_output=True, text=True, check=True).stdout.strip()}
    validate_spec(spec)
    return spec


def execute(options, profile):
    spec = build_spec(options, profile)
    if not valid_slack_webhook():
        raise ValueError('Slack credentials are required for low-rank validation')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        status = notify_slack(message)
        notifications.append({'message': message, 'status': status})
        write_json(output / 'notifications.json', notifications)
        return status

    if notify(f'{options.method} native validation started: first2 tasks × {spec["protocol"]["n_epochs"]} epochs; batch64; GPU1') != 'sent':
        raise RuntimeError('Required low-rank start notification failed before training')
    environment = os.environ.copy()
    environment.pop('MAMMOTH_DEVICE', None)
    environment.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', WANDB_MODE='online',
                       WANDB_DIR=str(output), CL_LOW_RANK_NATIVE_EXECUTE='1')
    from scripts.benchmark_cl import stop_worker
    start, heartbeat, timed_out = time.monotonic(), time.monotonic(), False
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
            cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - start >= options.timeout:
                    timed_out = True
                    stop_worker(process)
                    break
                if time.monotonic() - heartbeat >= 600:
                    progress = json.loads((output / 'progress.json').read_text()) if (output / 'progress.json').is_file() else {}
                    notify(f'{options.method} native validation running: task{progress.get("task", "?")}/2 epoch{progress.get("epoch", "?")}; W&B {progress.get("wandb_url", "initializing")}')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify(f'{options.method} native validation interrupted; local observations preserved')
            raise
    result = json.loads((output / 'result.json').read_text()) if (output / 'result.json').is_file() else {'status': 'failed'}
    status = 'timeout' if timed_out else result['status']
    if process.returncode != 0 and status == 'completed':
        status = 'failed'
    write_json(output / 'completion.json', {'status': status, 'exit_code': process.returncode, 'timeout': timed_out,
        'acceptance_status': result.get('acceptance_status'), 'wall_s': time.monotonic() - start,
        'wandb_url': result.get('wandb_url')})
    complete = notify(f'{options.method} native validation {status}: acceptance={result.get("acceptance_status", "not_evaluated")}; W&B {result.get("wandb_url", "unavailable")}')
    print(json.dumps({'status': status, 'acceptance_status': result.get('acceptance_status'),
                      'result': str(output / 'result.json'), 'wandb_url': result.get('wandb_url')}))
    return 0 if process.returncode == 0 and status == 'completed' and complete == 'sent' else 1


def worker(spec_path):
    if os.getenv('CL_LOW_RANK_NATIVE_EXECUTE') != '1':
        raise ValueError('Low-rank native worker requires explicit --execute parent authorization')
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    if os.getenv('CUDA_VISIBLE_DEVICES') != '1' or os.getenv('WANDB_MODE') != 'online':
        raise ValueError('Low-rank native worker requires GPU1 and online W&B')
    if source_hashes() != spec['execution_source_sha256']:
        raise ValueError('Native low-rank protected source changed after preparation')
    for path, pin in ((spec['checkpoint_file'], CHECKPOINT_SHA256),
                      (spec['initial_state'], spec['initial_state_sha256']),
                      (spec['reference_summary'], spec['reference_summary_sha256'])):
        if file_hash(path) != pin:
            raise ValueError('Low-rank pinned input artifact changed after preparation')
    reference = reference_record(spec['reference_summary'])
    validate_reference(reference, spec['method'], spec['initial_state_sha256'])
    if prediction_pins(reference, spec['reference_summary']) != spec['reference_prediction_files']:
        raise ValueError('Low-rank reference predictions changed after preparation')
    if validate_shared_data(spec['data_root']) != spec['data_files']:
        raise ValueError('Low-rank shared raw data changed after preparation')
    output, settings = Path(spec['output']), spec['protocol']
    recorder = Recorder(output, spec['gpu'])
    recorder.start()
    record = {**spec, 'status': 'failed', 'acceptance_status': 'not_evaluated', 'epoch_results': [],
        'task_results': [], 'first_batch_audits': [], 'prediction_files': [],
        'counts': {'observe_calls': 0, 'optimizer_updates': 0, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
        'structural_checks': {name: False for name in REQUIRED_STRUCTURAL_CHECKS}}
    checks, context, handles, code, wandb = record['structural_checks'], {}, [], 1, None
    try:
        import numpy as np
        import torch
        import wandb
        import main as mammoth
        from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
        from scripts.ease_size_protocol import canonical_pretrained_state
        from utils import training
        from utils.conf import base_path
        base_path(str(Path(spec['data_root']).resolve()))
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        torch.set_num_threads(2)
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_phase(1993, 0, 'task_begin')
        record['precision'] = {'matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_allowed': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'autocast': torch.is_autocast_enabled(), 'parameter_dtype': str(torch.get_default_dtype()).replace('torch.', ''),
            'torch_cpu_threads': torch.get_num_threads()}
        if record['precision'] != PRECISION:
            raise ValueError('Effective native precision differs from the frozen fp32 protocol')
        wandb.init(entity=spec['wandb_entity'], project=spec['wandb_project'], name=spec['name'], mode='online',
                   group=spec['method'] + '-first2-validation', job_type='native-acceptance', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        command = native_command(spec['method'], spec['checkpoint_file'], spec['data_root'], output)
        sys.argv = ['main.py', *command]
        with forbid_cifar_download():
            args = mammoth.parse_args(cmd=command)
            with recorder.phase('initialize'):
                model, dataset, args = mammoth.initialize(args)
        args.nowand = 1
        hyper = settings['hyperparameters']
        expected = {'stop_after': 2, 'n_epochs': settings['n_epochs'], 'batch_size': 64, 'num_workers': 4,
            'seed': 1993, 'lora_rank': 10, 'lora_vit_size': 'base', 'permute_classes': 0, 'noise_rate': 0,
            'validation': 0, 'label_perc': 1, 'label_perc_by_class': 1, 'debug_mode': 0,
            'enable_other_metrics': 0, 'eval_future': 0, 'eval_epochs': None, 'fitting_mode': 'epochs',
            'lr': hyper['lr'], 'lora_fc_lr': hyper['fc_lr'], 'optimizer': hyper['optimizer'],
            'optim_wd': hyper['weight_decay'], 'lr_scheduler': 'cosine', 'scheduler_mode': 'epoch'}
        expected.update({f'lora_{name}': value for name, value in hyper.items()
                         if name not in ('optimizer', 'lr', 'fc_lr', 'weight_decay', 'momentum')})
        if spec['method'] == 'cl-lora':
            expected.update(optim_mom=.9, optim_nesterov=0)
        if any(getattr(args, name) != value for name, value in expected.items()) or (dataset.N_TASKS, dataset.N_CLASSES_PER_TASK, dataset.N_CLASSES) != (10, 10, 100):
            raise ValueError('Effective native low-rank arguments or class partitions differ from the frozen protocol')
        catalog = torch.load(spec['initial_state'], map_location='cpu', weights_only=True)
        initial = load_initial_state(model.net, catalog, spec['method'])
        if initial != reference.get('initial_learned_sha256'):
            raise ValueError('Native learned initialization differs from the independent reference')
        frozen = canonical_pretrained_digest(model.net.image_encoder)
        if frozen != CANONICAL_PRETRAINED_SHA256:
            raise RuntimeError('Native 150 canonical pretrained tensors differ from the pinned Base NPZ')
        record.update(initial_learned_sha256=initial, initial_canonical_pretrained_sha256=frozen,
                      state_audits={'initial_frozen_backbone_sha256': frozen, 'tasks': []})
        checks.update({name: True for name in REQUIRED_STRUCTURAL_CHECKS})
        checks.update(training_protocol_complete=False, prediction_ids_and_counts_complete=False,
                      optimizer_updates_complete=False, method_state_valid=False, checkpoint_round_trip=False)
        write_json(output / 'effective_args.json', vars(args))
        write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'wandb')}
        frozen_names = {'image_encoder.' + name for name in canonical_pretrained_state(model.net.image_encoder)}
        epoch, hooked, configured = {}, WeakSet(), WeakKeyDictionary()

        def attach_optimizer(optimizer):
            if optimizer is None or optimizer in hooked:
                return optimizer
            hooked.add(optimizer)
            def before_step(subject, arguments, keywords):
                grads = {}
                for name, parameter in model.net.named_parameters():
                    if name in frozen_names and parameter.grad is not None:
                        checks['frozen_backbone_gradients_absent'] = False
                        raise RuntimeError('Frozen native pretrained parameter received a gradient')
                    if parameter.requires_grad:
                        if not bool(torch.isfinite(parameter).all()) or (parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())):
                            checks['finite_loss_and_gradients'] = False
                            raise FloatingPointError('Native low-rank learned parameter or gradient is nonfinite')
                        if parameter.grad is not None:
                            grads[name] = parameter.grad
                if context.get('stage') == 'train' and epoch.get('epoch') == 1 and epoch.get('steps') == 0:
                    context['batch_updates'] += 1
                    record['first_batch_audits'].append({'task': epoch['task'], 'epoch': 1,
                        'seed': epoch['seed'], 'optimizer_update_index': context['batch_updates'],
                        'samples': len(context['labels']), 'input_sha256': tensor_digest({'inputs': context['inputs']}),
                        'labels_sha256': tensor_digest({'labels': context['labels']}), 'gradient_sha256': tensor_digest(grads)})
            def after_step(subject, arguments, keywords):
                if context.get('stage') == 'train':
                    record['counts']['optimizer_updates'] += 1
            handles.extend((optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)))
            return optimizer

        def get_optimizer(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                return attach_optimizer(original(*args, **kwargs))
            return wrapped

        def loaders(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                task = len(dataset.test_loaders)
                source = dataset.train_loader.dataset
                raw = np.asarray(source.indexes)
                if len(source) != 5000 or not np.all(raw[1:] > raw[:-1]):
                    raise RuntimeError('Native training samples are incomplete or not in raw global index order')
                recorder.data_snapshots[f'task{task}_train'] = dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
                images = np.concatenate([loader.dataset.data for loader in dataset.test_loaders])
                labels = np.concatenate([loader.dataset.targets for loader in dataset.test_loaders])
                ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
                order = np.argsort(ids, kind='stable')
                recorder.data_snapshots[f'task{task}_test'] = dataset_fingerprint(images[order], labels[order], ids[order], dataset.TEST_TRANSFORM)
                for split in ('train', 'test'):
                    actual = recorder.data_snapshots[f'task{task}_{split}']
                    expected = reference['data_snapshots'][f'task{task}_{split}']
                    for key in ('n_samples', 'class_counts', 'samples_ordered_sha256', 'samples_unordered_sha256', 'global_ids_ordered_sha256', 'transform_repr_sha256'):
                        if actual[key] != expected[key]:
                            checks['dataset_matches_reference'] = False
                            raise RuntimeError('Native raw samples, IDs or preprocessing differ from the independent reference')
                return result
            return wrapped

        def task_phase(original, phase):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                task_id = len(dataset.test_loaders) - 1
                context['stage'] = phase
                seed_phase(1993, task_id, phase)
                configure_epoch_loader(dataset.train_loader, 1993, task_id, phase)
                configured[dataset.train_loader] = (1993, task_id, phase)
                result = original(*args, **kwargs)
                attach_optimizer(model.opt)
                context['stage'] = None
                return result
            return wrapped

        def begin_epoch(original):
            @functools.wraps(original)
            def wrapped(index, current_dataset):
                task_id = len(dataset.test_loaders) - 1
                selected = seed_phase(1993, task_id, 'train', index)
                configure_epoch_loader(dataset.train_loader, 1993, task_id, 'train', index)
                epoch.clear()
                epoch.update(task=task_id + 1, epoch=index + 1, seed=selected, steps=0, samples=0, loss_sum=0.)
                context.update(stage='train', batch_updates=0)
                return original(index, current_dataset)
            return wrapped

        def observed(original):
            @functools.wraps(original)
            def wrapped(inputs, labels, *args, **kwargs):
                context.update(inputs=inputs, labels=labels, batch_updates=0)
                result = original(inputs, labels, *args, **kwargs)
                loss = float(result['loss'] if isinstance(result, dict) else result)
                if not math.isfinite(loss):
                    checks['finite_loss_and_gradients'] = False
                    raise FloatingPointError('Native low-rank loss is nonfinite')
                if context.get('stage') == 'train':
                    epoch['steps'] += 1
                    epoch['samples'] += len(labels)
                    epoch['loss_sum'] += loss
                    record['counts']['observe_calls'] += 1
                return result
            return wrapped

        def end_epoch(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                row = {**epoch, 'loss_mean': epoch['loss_sum'] / max(1, epoch['steps'])}
                record['epoch_results'].append(row)
                wandb.log({f'native_train/{key}': value for key, value in row.items()})
                write_json(output / 'epoch_results.json', record['epoch_results'])
                write_json(output / 'progress.json', {**row, 'wandb_url': record['wandb_url']})
                context['stage'] = None
                return result
            return wrapped

        original_iter = torch.utils.data.DataLoader.__iter__
        def counted_iter(loader):
            stage = context.get('stage')
            if stage not in ('task_begin', 'task_end'):
                return original_iter(loader)
            configure_phase_loader_once(loader, 1993, len(dataset.test_loaders) - 1, stage, configured)
            def iterator():
                for batch in original_iter(loader):
                    record['counts'][stage + '_extra_samples'] += len(batch[0])
                    labels = batch[1]
                    task = len(dataset.test_loaders) - 1
                    if bool(((labels < task * 10) | (labels >= (task + 1) * 10)).any()):
                        checks['no_future_or_replay_samples'] = False
                        raise RuntimeError('Native task statistics read old or future samples')
                    yield batch
            return iterator()

        def evaluated(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                task = len(dataset.test_loaders)
                seed_phase(1993, task - 1, 'evaluation')
                predictions, labels, ids = [], [], []
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    for loader in dataset.test_loaders:
                        configure_epoch_loader(loader, 1993, task - 1, 'evaluation')
                        for inputs, targets, *rest in loader:
                            logits = model(inputs.to(model.device))
                            if logits.shape != (len(targets), task * 10) or not bool(torch.isfinite(logits).all()):
                                checks['nonoracle_seen_class_evaluation'] = False
                                raise RuntimeError('Native evaluation has nonfinite or incorrectly masked seen-class logits')
                            predictions.append(logits.argmax(1).cpu().numpy())
                            labels.append(targets.numpy())
                        ids.append(np.asarray(loader.dataset.indexes))
                model.train(was_training)
                predictions, labels, ids = map(np.concatenate, (predictions, labels, ids))
                order = np.argsort(ids, kind='stable')
                predictions, labels, ids = predictions[order], labels[order], ids[order]
                archive = output / f'task{task}_predictions.npz'
                np.savez_compressed(archive, predictions=predictions, labels=labels, global_ids=ids)
                reference_archive = next(item['path'] for item in spec['reference_prediction_files'] if item['task'] == task)
                with np.load(reference_archive, allow_pickle=False) as expected:
                    counts = validate_prediction_archive(archive, task, expected['global_ids'], expected['labels'])
                row = {'task': task, **counts}
                record['task_results'].append(row)
                record['prediction_files'].append({'task': task, 'path': str(archive), 'sha256': file_hash(archive)})
                write_json(output / 'task_results.json', record['task_results'])
                wandb.log({f'native_task/{key}': value for key, value in row.items()})
                return result
            return wrapped

        def finalized_eval(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                if spec['method'] == 'ewc-lora':
                    seed_phase(1993, len(dataset.test_loaders) - 1, 'task_end')
                result = original(*args, **kwargs)
                task = len(dataset.test_loaders)
                current = canonical_pretrained_digest(model.net.image_encoder)
                if current != frozen:
                    checks['frozen_backbone_unchanged'] = False
                    raise RuntimeError('Native canonical pretrained state changed during training')
                audit_method_state(model.net, model.algorithm, spec['method'], task)
                state = model.algorithm.state_dict()
                network_state = model.net.state_dict()
                if any(not bool(torch.isfinite(value).all()) for value in [*state.values(), *network_state.values()]):
                    raise FloatingPointError('Native low-rank task state is nonfinite')
                if model.net._cur_task != task - 1 or int(model._low_rank_state[3]) != task * 10:
                    raise RuntimeError('Native low-rank task counters disagree with the evaluated prefix')
                record['state_audits']['tasks'].append({'task': task, 'frozen_backbone_sha256': current,
                    'network_sha256': tensor_digest(network_state), 'algorithm_sha256': tensor_digest(state),
                    'network_task_id': model.net._cur_task, 'seen_classes': task * 10,
                    'algorithm_state_shapes': {name: list(value.shape) for name, value in state.items()}})
                destination = output / f'task{task}_state.pt'
                torch.save({'model': model.state_dict(), 'task': task, 'protocol': settings}, destination)
                saved = torch.load(destination, map_location='cpu', weights_only=True)
                if tensor_digest(saved['model']) != tensor_digest(model.state_dict()):
                    raise RuntimeError('Native low-rank task checkpoint round trip changed state')
                record['state_audits']['tasks'][-1]['checkpoint_sha256'] = file_hash(destination)
                row = record['task_results'][-1]
                notify_status = notify_slack(f'{spec["method"]} native task{task}/2 completed: seen={100 * row["seen_correct"] / row["seen_samples"]:.2f}%; W&B {record["wandb_url"]}')
                if notify_status != 'sent':
                    raise RuntimeError('Required low-rank task-completion notification failed')
                return result
            return wrapped

        with ExitStack() as stack:
            stack.enter_context(forbid_cifar_download())
            stack.enter_context(instrument(recorder, model, dataset, training))
            for target, name, wrapper in ((model, 'get_optimizer', get_optimizer),
                (dataset, 'get_data_loaders', loaders), (model, 'meta_begin_epoch', begin_epoch),
                (model, 'meta_observe', observed), (model, 'meta_end_epoch', end_epoch),
                (dataset, 'evaluate', evaluated), (model, 'meta_end_eval', finalized_eval)):
                stack.enter_context(patch.object(target, name, wrapper(getattr(target, name))))
            stack.enter_context(patch.object(model, 'meta_begin_task', task_phase(model.meta_begin_task, 'task_begin')))
            stack.enter_context(patch.object(model, 'meta_end_task', task_phase(model.meta_end_task, 'task_end')))
            stack.enter_context(patch.object(torch.utils.data.DataLoader, '__iter__', counted_iter))
            training.train(model, dataset, args)
        completion_checks(record, spec['method'])
        reference_audits = reference.get('first_batch_audits', [])
        if len(record['first_batch_audits']) != len(reference_audits):
            raise RuntimeError('Native and reference first-batch optimizer-update audit counts differ')
        for actual, expected in zip(record['first_batch_audits'], reference_audits):
            if any(actual[key] != expected.get(key) for key in ('task', 'epoch', 'seed', 'samples', 'optimizer_update_index', 'input_sha256', 'labels_sha256')):
                checks['first_batch_inputs_match'] = False
                raise RuntimeError('Native and independent reference first training inputs or labels differ')
        checks.update(training_protocol_complete=True, prediction_ids_and_counts_complete=True,
                      optimizer_updates_complete=True, method_state_valid=True, checkpoint_round_trip=True)
        report = evaluate_acceptance(record['task_results'], reference['task_results'], structural_checks=checks)
        record.update(status='completed', acceptance_status=report['status'], acceptance=report)
        write_json(output / 'acceptance.json', report)
        code = 0
    except Exception as error:
        record.update(error_type=type(error).__name__, error='Validation failed; inspect local structural audits', acceptance_status='fail')
        logging.error('Native low-rank validation failed: %s', type(error).__name__)
    finally:
        for handle in handles:
            handle.remove()
        record = recorder.save(record)
        record['computation_status'] = record['status']
        record['data_snapshots'] = recorder.data_snapshots
        write_json(output / 'result.json', record)
        if wandb is not None and wandb.run is not None:
            try:
                update_wandb_summary(wandb.run, record)
                write_json(output / 'state_audits.json', record.get('state_audits', {}))
                write_json(output / 'first_batch_audits.json', record['first_batch_audits'])
                artifact = wandb.Artifact(spec['name'], type='cl-native-acceptance')
                for filename in ('spec.json', 'result.json', 'metrics.json', 'acceptance.json', 'effective_args.json',
                                 'shared_data.json', 'epoch_results.json', 'task_results.json', 'state_audits.json', 'first_batch_audits.json',
                                 'phases.json', 'telemetry.csv',
                                 'task1_predictions.npz', 'task2_predictions.npz'):
                    if (output / filename).is_file():
                        artifact.add_file(str(output / filename))
                wandb.log_artifact(artifact).wait()
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record.update(wandb_logging_status='failed:' + type(error).__name__, status='logging_failed')
                logging.exception('Native low-rank evidence upload failed')
                code = 1
            finally:
                try:
                    wandb.finish(exit_code=code)
                except Exception as error:
                    record.update(wandb_logging_status='finish_failed:' + type(error).__name__, status='logging_failed')
                    code = 1
        else:
            record['wandb_logging_status'] = 'unavailable'
            code = 1
        write_json(output / 'result.json', record)
        write_json(output / 'metrics.json', record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--method', choices=tuple(METHODS), default='ewc-lora')
    parser.add_argument('--reference-summary', type=Path)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/low_rank/runs')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--timeout', type=float, default=21600.)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if not options.execute and options.worker is None:
        print(f'Prepared {options.method} native first2 validation; execution=False; protocol={json.dumps(protocol(options.method))}')
        return 0
    if options.worker is not None:
        return worker(options.worker)
    return execute(options, configure_environment(options.profile, options.credentials))


if __name__ == '__main__':
    raise SystemExit(main())
