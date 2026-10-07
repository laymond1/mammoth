"""Opt-in EASE shared-input validation and guarded ten-task native execution.

No credentials, model imports, GPU work, W&B or Slack calls occur in dry mode.
The native Mammoth train loop and all meta hooks remain the execution path.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cl_efficiency import Recorder, flatten, instrument, model_snapshot, notify_slack, write_json
from scripts.cl_runtime import PUBLIC_DEFAULTS, configure_environment, configure_native_environment
from scripts.ease_fullcl_protocol import (CANONICAL_PRETRAINED_SHA256, CHECKPOINT, CHECKPOINT_SHA256,
    PRECISION, STRUCTURAL_CHECKS, configure_epoch_loader, file_hash, load_native_initialization,
    native_command, phase_seed, prepare_native_args, protocol, seed_phase, source_hashes,
    validate_full_cl_acceptance, validate_reference)


def build_spec(options, profile=None):
    profile = dict(PUBLIC_DEFAULTS if profile is None else profile)
    if profile['gpu'] != '1' or profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('EASE shared-input execution requires the approved GPU1 and W&B target')
    settings = protocol(options.mode)
    if options.mode == 'full_cl':
        if options.acceptance_summary is None:
            raise ValueError('EASE full CL requires --acceptance-summary')
        accepted = validate_full_cl_acceptance(options.acceptance_summary, options.initial_state)
        reference_path = Path(accepted['reference_summary']).resolve()
        initial_state = Path(options.initial_state or accepted['initial_state']).resolve()
    else:
        if options.reference_summary is None:
            raise ValueError('EASE validation requires --reference-summary')
        reference_path = options.reference_summary.resolve()
        reference = validate_reference(json.loads(reference_path.read_text()), options.initial_state)
        initial_state = Path(options.initial_state or reference['initial_state']).resolve()
    reference = validate_reference(json.loads(reference_path.read_text()), initial_state)
    from scripts.run_ease_validation import reference_policy
    policy, pins = reference_policy(reference, {'unit': 'percentage_points',
                                               'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.})
    name = f'ease-native-{options.mode}-common-b64-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = options.result_dir.resolve() if options.result_dir else ROOT / 'benchmarks/cl_efficiency/ease_full_cl/runs' / name
    spec = {'schema_version': 1, 'name': name, 'stage': 'native_ease_shared_input', 'model': 'ease',
            'mode': options.mode, 'protocol': settings, 'tasks': settings['tasks'], 'output': str(output),
            'gpu': profile['gpu'], 'data_root': profile['data_root'], 'checkpoint_file': CHECKPOINT,
            'checkpoint_sha256': CHECKPOINT_SHA256, 'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
            'initial_state': str(initial_state), 'initial_state_sha256': file_hash(initial_state),
            'reference_summary': str(reference_path), 'reference_summary_sha256': file_hash(reference_path),
            'reference_wandb_url': reference['wandb_url'], 'reference_predictions_sha256': pins,
            'accuracy_policy': policy, 'precision': PRECISION.copy(),
            'wandb_entity': profile['wandb_entity'], 'wandb_project': profile['wandb_project'],
            'execution_source_sha256': source_hashes(),
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True, check=True).stdout.strip(),
            'paper_reproduction': False, 'equal_compute_budget': False, 'flops_measured': False}
    if options.mode == 'full_cl':
        spec.update(acceptance_summary=str(options.acceptance_summary.resolve()),
                    acceptance_summary_sha256=file_hash(options.acceptance_summary))
    return spec


def validate_spec(spec):
    if (spec.get('model') != 'ease' or spec.get('protocol') != protocol(spec.get('mode'))
            or spec.get('tasks') != spec['protocol']['tasks'] or spec.get('precision') != PRECISION
            or spec.get('gpu') != '1' or spec.get('checkpoint_file') != CHECKPOINT
            or spec.get('checkpoint_sha256') != CHECKPOINT_SHA256):
        raise ValueError('EASE native execution is outside the approved shared-input protocol')
    if spec.get('execution_source_sha256') != source_hashes():
        raise ValueError('EASE native execution sources changed after preparation')
    for key in ('checkpoint', 'initial_state', 'reference_summary'):
        path_key = 'checkpoint_file' if key == 'checkpoint' else key
        if file_hash(spec[path_key]) != spec[key + '_sha256']:
            raise ValueError('EASE pinned execution artifact changed after preparation')
    reference = validate_reference(json.loads(Path(spec['reference_summary']).read_text()), spec['initial_state'])
    from scripts.run_ease_validation import reference_policy
    policy, pins = reference_policy(reference, {'unit': 'percentage_points',
                                               'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.})
    if policy != spec['accuracy_policy'] or pins != spec['reference_predictions_sha256']:
        raise ValueError('EASE pinned reference predictions or exact accuracy policy changed')
    if spec['mode'] == 'full_cl':
        if file_hash(spec['acceptance_summary']) != spec['acceptance_summary_sha256']:
            raise ValueError('EASE accepted validation summary changed')
        validate_full_cl_acceptance(spec['acceptance_summary'], spec['initial_state'])
    return True


def compare_prefix_predictions(actual, expected):
    import numpy as np
    with np.load(actual, allow_pickle=False) as current, np.load(expected, allow_pickle=False) as accepted:
        if set(current.files) != set(accepted.files) or any(not np.array_equal(current[key], accepted[key]) for key in current.files):
            raise RuntimeError('EASE full CL prefix predictions differ from accepted native validation')
    return True


def completion_checks(record, recorder, model, settings):
    tasks, epochs, batches = settings['tasks'], settings['n_epochs'], math.ceil(5000 / settings['batch_size'])
    rows = record['epoch_results']
    return bool(len(rows) == tasks * epochs
                and {(row['task'], row['epoch']) for row in rows} == {(task, epoch) for task in range(1, tasks + 1) for epoch in range(1, epochs + 1)}
                and all(row['steps'] == batches and row['samples'] == 5000 for row in rows)
                and recorder.tasks_completed == tasks and recorder.steps == tasks * epochs * batches
                and recorder.training_examples == tasks * epochs * 5000
                and recorder.evaluation_examples == 1000 * tasks * (tasks + 1) // 2
                and record['counts']['prototype_examples'] == 5000 * tasks
                and record['counts']['prototype_adapter_image_pairs'] == 5000 * tasks * (tasks + 1) // 2
                and record['counts']['evaluation_adapter_image_pairs'] == 1000 * sum(task ** 2 for task in range(1, tasks + 1))
                and len(record['per_task_resources']) == tasks
                and model.net.backbone.num_adapters == tasks
                and list(model.net.classifier.weight.shape) == [10 * tasks, 768 * tasks]
                and bool(model.net.task_finalized))


def execute(options, spec):
    validate_spec(spec)
    profile = configure_environment()
    if any(profile[key] != spec[key] for key in ('gpu', 'wandb_entity', 'wandb_project', 'data_root')):
        raise ValueError('Effective runtime profile differs from the pinned EASE execution target')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        notifications.append({'message': message, 'status': notify_slack(message)})
        write_json(output / 'notifications.json', notifications)

    environment = os.environ.copy()
    environment.pop('MAMMOTH_DEVICE', None)
    environment.update(CUDA_VISIBLE_DEVICES='1', CL_EASE_FULLCL_EXECUTE='1', HF_HUB_OFFLINE='1',
                       WANDB_MODE='online', WANDB_DIR=str(output))
    configure_native_environment(environment)
    notify(f'EASE native {spec["mode"]} started: Base/adapter64, batch64, seed1993, {spec["tasks"]} tasks; GPU1')
    from scripts.benchmark_cl import stop_worker
    started, heartbeat, timed_out = time.monotonic(), time.monotonic(), False
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
                                   cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= options.timeout:
                    timed_out = True
                    stop_worker(process)
                    break
                if time.monotonic() - heartbeat >= 600:
                    path = output / 'progress.json'
                    progress = json.loads(path.read_text()) if path.exists() else {}
                    notify(f'EASE native {spec["mode"]}: task{progress.get("task", "?")}/{spec["tasks"]}, {progress.get("phase", "initializing")} epoch{progress.get("epoch", "?")}; W&B {progress.get("wandb_url") or "initializing"}')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify('EASE native interrupted; partial local results preserved')
            raise
    path = output / 'result.json'
    record = json.loads(path.read_text()) if path.exists() else {**spec, 'status': 'failed'}
    if timed_out:
        record['status'] = 'timeout'
    elif process.returncode != 0 and record.get('status') == 'completed':
        record['status'] = 'failed'
    write_json(path, record)
    write_json(output / 'metrics.json', record)
    notify(f'EASE native {spec["mode"]} {record["status"]}; acceptance={record.get("acceptance_status", "unavailable")}; W&B {record.get("wandb_url") or "unavailable"}')
    write_json(output / 'completion.json', {'status': record['status'], 'exit_code': process.returncode,
               'timeout': timed_out, 'wall_s': time.monotonic() - started, 'acceptance_status': record.get('acceptance_status')})
    print(json.dumps({'status': record['status'], 'acceptance_status': record.get('acceptance_status'),
                      'output': str(output), 'wandb_url': record.get('wandb_url')}), flush=True)
    return 0 if process.returncode == 0 and record['status'] == 'completed' else 1


def worker(spec_path):
    if os.getenv('CL_EASE_FULLCL_EXECUTE') != '1':
        raise ValueError('EASE native worker requires explicit --execute')
    spec = json.loads(spec_path.read_text())
    validate_spec(spec)
    settings, output = spec['protocol'], Path(spec['output'])
    configure_native_environment()
    recorder = Recorder(output, spec['gpu'])
    recorder.start()
    record = {**spec, 'status': 'failed', 'acceptance_status': 'not_evaluated',
              'task_results': [], 'epoch_results': [], 'first_batch_audits': [],
              'per_task_resources': [], 'structural_checks': {key: True for key in STRUCTURAL_CHECKS},
              'counts': {'prototype_examples': 0, 'prototype_adapter_image_pairs': 0,
                         'evaluation_adapter_image_pairs': 0}}
    record['structural_checks']['training_protocol_complete'] = False
    if spec['mode'] == 'full_cl':
        record['prefix_validation'] = {'status': 'pending', 'prediction_match': False, 'matched_tasks': [],
            'scope': 'First two tasks only; no independent ten-task performance reference'}
    code, wandb, resources, stack = 1, None, None, ExitStack()
    try:
        import torch
        import numpy as np
        import wandb
        from unittest.mock import patch
        from torch.utils.data import SequentialSampler
        import main as mammoth
        import utils.training as training
        from scripts.ease_acceptance import counts_from_predictions, evaluate_acceptance
        from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
        from scripts.ease_size_protocol import canonical_pretrained_digest
        from scripts.ease_common_reference_utils import first_batch_audit, validate_initialization_catalog
        from scripts.hide_protocol import forbid_cifar_download, validate_shared_data
        from scripts.run_hide_validation import TaskResources, tensor_digest
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        reference = json.loads(Path(spec['reference_summary']).read_text())
        accepted = json.loads(Path(spec['acceptance_summary']).read_text()) if spec['mode'] == 'full_cl' else None
        catalog = validate_initialization_catalog(torch.load(spec['initial_state'], map_location='cpu', weights_only=True))
        validate_shared_data(spec['data_root'])
        command = native_command(spec['mode'], spec['checkpoint_file'], spec['data_root'], output)
        args = prepare_native_args(mammoth.parse_args(cmd=command))
        record['native_cli'] = ['main.py', *command]
        record['initialization_overrides'] = {'pretrained': False,
            'dataset_config': 'hide_prompt', 'dataset_fields': 'Pinned uppercase input fields only; EASE20epoch schedule retained'}
        wandb.init(entity=spec['wandb_entity'], project=spec['wandb_project'], name=spec['name'],
                   group='ease-base-common-input-full-cl', job_type='native-' + spec['mode'], config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        stack.enter_context(forbid_cifar_download())
        with recorder.phase('initialize'):
            model, dataset, args = mammoth.initialize(args)
        args.nowand = 1
        expected = {'stop_after': settings['tasks'], 'n_epochs': 20, 'batch_size': 64, 'num_workers': 4,
            'seed': 1993, 'vit_size': 'base', 'adapter_size': 64, 'ease_pretrain_type': 'in21k_ft_in1k',
            'lr': .025, 'optim_mom': .9, 'optim_wd': .0005, 'optimizer': 'sgd', 'lr_scheduler': 'cosine',
            'scheduler_mode': 'epoch', 'code_optimization': 0, 'permute_classes': 0, 'noise_rate': 0,
            'validation': 0, 'label_perc': 1, 'label_perc_by_class': 1, 'eval_future': 0,
            'debug_mode': 0, 'dataset_config': 'hide_prompt', 'pretrained': False,
            'ease_alpha': .1, 'ease_recalc_sim': True, 'ease_use_reweight': True}
        if any(getattr(args, key) != value for key, value in expected.items()) or args.eval_epochs is not None:
            raise ValueError('Effective EASE arguments differ from the pinned shared-input protocol')
        if dataset.N_TASKS != 10 or dataset.N_CLASSES_PER_TASK != 10 or dataset.N_CLASSES != 100:
            raise ValueError('EASE native task partition differs from the pinned CIFAR100 stream')
        record['precision'] = {'matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_allowed': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'autocast': False, 'parameter_dtype': str(next(model.parameters()).dtype).split('.')[-1],
            'torch_cpu_threads': torch.get_num_threads()}
        if record['precision'] != PRECISION:
            raise ValueError('Effective EASE precision differs from the approved FP32 settings')
        write_json(output / 'effective_args.json', vars(args))
        write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'pyarrow', 'wandb')}
        checks = record['structural_checks']
        ptm = model.net.backbone.backbone
        with recorder.phase('state_audit'):
            frozen_pin = canonical_pretrained_digest(ptm)
        if frozen_pin != CANONICAL_PRETRAINED_SHA256 or frozen_pin != reference['initial_canonical_pretrained_sha256']:
            checks['canonical_pretrained_matches_pin'] = False
            raise RuntimeError('EASE canonical pretrained tensors differ from author/HiDe checkpoint')
        record['initial_canonical_pretrained_sha256'] = frozen_pin
        record['state_audits'] = {'initial_frozen_backbone_sha256': frozen_pin, 'tasks': []}
        adapter_pins, epoch, predictions, active = {}, {}, [], {'inputs': None, 'labels': None}
        resources = TaskResources(recorder)
        stack.enter_context(instrument(recorder, model, dataset, training))

        def audit_data(task):
            source = dataset.train_loader.dataset
            if (len(source.targets) != 5000 or sorted(set(map(int, source.targets))) != list(range((task - 1) * 10, task * 10))
                    or len(np.unique(source.indexes)) != 5000):
                raise RuntimeError('EASE training task includes missing, old or future samples')
            recorder.data_snapshots[f'task{task}_train'] = dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
            images = np.concatenate([loader.dataset.data for loader in dataset.test_loaders])
            labels = np.concatenate([loader.dataset.targets for loader in dataset.test_loaders])
            ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
            order = np.argsort(ids, kind='stable')
            recorder.data_snapshots[f'task{task}_test'] = dataset_fingerprint(images[order], labels[order], ids[order], dataset.test_loaders[0].dataset.transform)
            matched = reference if spec['mode'] == 'validation' else accepted
            if task <= 2:
                keys = ('n_samples', 'class_counts', 'samples_ordered_sha256', 'samples_unordered_sha256',
                        'global_ids_ordered_sha256', 'transform_repr_sha256')
                for split in ('train', 'test'):
                    actual, expected = recorder.data_snapshots[f'task{task}_{split}'], matched['data_snapshots'][f'task{task}_{split}']
                    if any(actual[key] != expected[key] for key in keys):
                        checks['dataset_matches_reference'] = False
                        raise RuntimeError('EASE actual samples or preprocessing differ from matched validation')

        def loaders(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                resources.start(len(dataset.test_loaders) + 1)
                return original(*arguments, **keywords)
            return wrapped

        def begin(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                task_id = len(dataset.test_loaders) - 1
                seed_phase(settings['seed'], task_id, 'task_begin')
                result = original(*arguments, **keywords)
                with recorder.phase('initialization_audit'):
                    load_native_initialization(model.net, catalog, task_id)
                    optimized = {id(parameter) for group in model.opt.param_groups for parameter in group['params']}
                    if any(parameter.requires_grad or id(parameter) in optimized for parameter in ptm.parameters()):
                        raise RuntimeError('EASE frozen pretrained parameters entered optimizer')
                    if any(parameter.requires_grad or id(parameter) in optimized for adapter in model.net.backbone.adapters[:-1] for parameter in adapter.parameters()):
                        raise RuntimeError('EASE old adapter parameters entered optimizer')
                    if ptm.training or any(adapter.training for adapter in model.net.backbone.adapters[:-1]):
                        raise RuntimeError('EASE frozen modules are in training mode')
                    if any(group['momentum'] != .9 or group['weight_decay'] != .0005 or group['lr'] != .025 or group['nesterov'] for group in model.opt.param_groups):
                        raise RuntimeError('EASE actual optimizer differs from the author schedule')
                    audit_data(task_id + 1)
                    recorder.snapshots[f'task{task_id + 1}_after_begin_task'] = model_snapshot(model)
                return result
            return wrapped

        def begin_epoch(original):
            @functools.wraps(original)
            def wrapped(index, current_dataset):
                task_id = model.net.backbone.num_adapters - 1
                selected = seed_phase(settings['seed'], task_id, 'train', index)
                configure_epoch_loader(dataset.train_loader, settings['seed'], task_id, 'train', index)
                epoch.clear()
                epoch.update(task=task_id + 1, epoch=index + 1, seed=selected, loss_sum=0., loss_weighted_sum=0., steps=0, samples=0, correct=0)
                return original(index, current_dataset)
            return wrapped

        def observed(original):
            @functools.wraps(original)
            def wrapped(inputs, labels, *arguments, **keywords):
                active.update(inputs=inputs, labels=labels)
                try:
                    result = original(inputs, labels, *arguments, **keywords)
                    loss = float(result['loss'] if isinstance(result, dict) else result)
                    if not math.isfinite(loss):
                        checks['finite_loss_and_gradients'] = False
                        raise FloatingPointError('EASE native loss is nonfinite')
                    epoch['loss_sum'] += loss
                    epoch['loss_weighted_sum'] += loss * len(labels)
                    epoch['steps'] += 1
                    epoch['samples'] += len(labels)
                    return result
                finally:
                    active.update(inputs=None, labels=None)
            return wrapped

        def logits(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                result = original(*arguments, **keywords)
                if active['labels'] is not None:
                    if result.shape[1] != 10 or not bool(torch.isfinite(result).all()):
                        raise FloatingPointError('EASE proxy logits are nonfinite or contain future classes')
                    epoch['correct'] += int((result.detach().argmax(1) == active['labels'] - model.n_past_classes).sum())
                return result
            return wrapped

        def audit_step(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                if epoch['epoch'] == 1 and epoch['steps'] == 0:
                    with recorder.phase('first_batch_audit'):
                        named = [(f'adapter.{name}', parameter) for name, parameter in model.net.backbone.adapters[-1].named_parameters()]
                        named += [(f'proxy_head.{name}', parameter) for name, parameter in model.net.proxy_head.named_parameters()]
                        row = first_batch_audit(active['inputs'], active['labels'], named,
                                                task=epoch['task'], epoch=epoch['epoch'], seed=epoch['seed'])
                        record['first_batch_audits'].append(row)
                        write_json(output / 'first_batch_audits.json', record['first_batch_audits'])
                        matched = reference if spec['mode'] == 'validation' else (accepted if epoch['task'] <= 2 else None)
                        if matched is not None:
                            expected = [saved for saved in matched['first_batch_audits'] if saved['task'] == epoch['task'] and saved['epoch'] == 1]
                            keys = ('seed', 'batch_size', 'inputs_sha256', 'labels_sha256')
                            if spec['mode'] == 'full_cl':
                                keys += ('gradient_sha256',)
                            if len(expected) != 1 or any(row[key] != expected[0][key] for key in keys):
                                checks['actual_inputs_match_reference'] = False
                                raise RuntimeError('EASE first actual images/labels/RNG differ from matched execution')
                return original(*arguments, **keywords)
            return wrapped

        def train_epoch(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                with patch.object(model.opt, 'step', audit_step(model.opt.step)):
                    result = original(*arguments, **keywords)
                with recorder.phase('epoch_audit'):
                    if any(parameter.grad is not None for parameter in ptm.parameters()):
                        checks['frozen_backbone_gradients_absent'] = False
                        raise RuntimeError('EASE frozen pretrained tensor received gradient')
                    if any(not bool(torch.isfinite(parameter).all()) or (parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()))
                           for parameter in model.net.parameters() if parameter.requires_grad):
                        checks['finite_loss_and_gradients'] = False
                        raise FloatingPointError('EASE trainable state or gradient is nonfinite')
                row = {key: epoch[key] for key in ('task', 'epoch', 'seed', 'steps', 'samples')}
                row.update(epochs=20, phase='train', loss=epoch['loss_sum'] / epoch['steps'],
                           loss_example_weighted=epoch['loss_weighted_sum'] / epoch['samples'],
                           train_accuracy_pct=100 * epoch['correct'] / epoch['samples'],
                           lr_after_scheduler=model.opt.param_groups[0]['lr'])
                record['epoch_results'].append(row)
                if accepted is not None and row['task'] <= 2:
                    expected = [saved for saved in accepted['epoch_results'] if saved['task'] == row['task'] and saved['epoch'] == row['epoch']]
                    if len(expected) != 1 or row != expected[0]:
                        raise RuntimeError('EASE full CL prefix epoch observations differ from accepted native execution')
                write_json(output / 'epoch_results.json', record['epoch_results'])
                write_json(output / 'progress.json', {**row, 'wandb_url': record['wandb_url']})
                wandb.log({f'native/{key}': value for key, value in row.items() if isinstance(value, (int, float))})
                print(f'EASE native task{row["task"]}/{settings["tasks"]} epoch{row["epoch"]}/20: loss={row["loss"]:.5f}, train={row["train_accuracy_pct"]:.2f}%', flush=True)
                return result
            return wrapped

        def prototype(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                task = model.net.backbone.num_adapters
                seed_phase(settings['seed'], task - 1, 'prototype')
                write_json(output / 'progress.json', {'task': task, 'phase': 'prototype_completion', 'wandb_url': record['wandb_url']})
                with recorder.phase('prototype_completion'):
                    return original(*arguments, **keywords)
            return wrapped

        def extracted(original):
            @functools.wraps(original)
            def wrapped(inputs, *arguments, **keywords):
                result = original(inputs, *arguments, **keywords)
                task = model.net.backbone.num_adapters
                if recorder.active_phase == 'prototype_completion':
                    record['counts']['prototype_examples'] += len(inputs)
                    record['counts']['prototype_adapter_image_pairs'] += len(inputs) * task
                elif recorder.active_phase == 'evaluation':
                    record['counts']['evaluation_adapter_image_pairs'] += len(inputs) * task
                return result
            return wrapped

        def forwarded(original):
            @functools.wraps(original)
            def wrapped(inputs, *arguments, **keywords):
                result = original(inputs, *arguments, **keywords)
                if recorder.active_phase == 'evaluation':
                    if result.shape[1] != model.n_seen_classes or not bool(torch.isfinite(result).all()):
                        checks['nonoracle_seen_class_evaluation'] = False
                        raise RuntimeError('EASE class-IL logits are nonfinite or contain future classes')
                    predictions.append(torch.argsort(result.detach(), dim=1, descending=True, stable=True)[:, :5].cpu().numpy())
                return result
            return wrapped

        def evaluated(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                task = model.net.backbone.num_adapters
                if keywords.get('last', False) or keywords.get('return_loss', False) or any(not isinstance(loader.sampler, SequentialSampler) or loader.drop_last for loader in dataset.test_loaders):
                    raise ValueError('EASE full evaluation requires all seen separate sequential test-task loaders')
                seed_phase(settings['seed'], task - 1, 'evaluation')
                predictions.clear()
                result = original(*arguments, **keywords)
                with recorder.phase('acceptance_audit'):
                    labels = np.concatenate([loader.dataset.targets for loader in dataset.test_loaders])
                    ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
                    predicted = np.concatenate(predictions)
                    if len(predicted) != task * 1000 or len(labels) != len(predicted):
                        raise RuntimeError('EASE actual evaluation stream is incomplete')
                    counts = counts_from_predictions(predicted[:, 0], labels, past_classes=(task - 1) * 10, seen_classes=task * 10)
                    row = {'task': task, 'seen_classes': task * 10, **counts,
                           'seen_accuracy_pct': 100 * counts['seen_correct'] / counts['seen_samples'],
                           'old_accuracy_pct': 100 * counts['old_correct'] / counts['old_samples'] if counts['old_samples'] else None,
                           'new_accuracy_pct': 100 * counts['new_correct'] / counts['new_samples'],
                           'native_class_il_per_task': result[0], 'native_task_il_per_task': result[1],
                           'prototype_shape': list(model.net.classifier.weight.shape), 'observed_adapters': task}
                    if row['prototype_shape'] != [task * 10, task * 768] or not bool(model.net.task_finalized):
                        checks['prototype_and_adapter_shapes'] = False
                        raise RuntimeError('EASE actual adapter/prototype lifecycle differs')
                    if any(abs(result[0][index] - 100 * (predicted[labels // 10 == index, 0] == labels[labels // 10 == index]).mean()) > 1e-9 for index in range(task)):
                        raise RuntimeError('EASE saved predictions disagree with native class-IL accuracy')
                    current_ptm = canonical_pretrained_digest(ptm)
                    if current_ptm != frozen_pin:
                        checks['frozen_backbone_unchanged'] = False
                        raise RuntimeError('EASE frozen pretrained tensors changed')
                    current = {str(index): tensor_digest(adapter.state_dict()) for index, adapter in enumerate(model.net.backbone.adapters)}
                    if any(current[key] != pin for key, pin in adapter_pins.items()):
                        checks['old_adapters_unchanged'] = False
                        raise RuntimeError('EASE previously finalized adapter changed')
                    adapter_pins.update(current)
                    record['state_audits']['tasks'].append({'task': task, 'backbone_sha256': current_ptm, 'adapter_sha256': current})
                    order = np.argsort(ids, kind='stable')
                    np.savez(output / f'task{task}_predictions.npz', predictions=predicted[order], labels=labels[order],
                             global_ids=ids[order], evaluation_global_ids=ids)
                    if accepted is not None and task <= 2:
                        compare_prefix_predictions(output / f'task{task}_predictions.npz', Path(accepted['output']) / f'task{task}_predictions.npz')
                        record['prefix_validation']['matched_tasks'].append(task)
                        record['prefix_validation']['prediction_match'] = True
                    with recorder.phase('checkpoint'):
                        state = {name: value.detach().cpu().clone() if torch.is_tensor(value) else value for name, value in model.net.state_dict().items()}
                        before = tensor_digest(state)
                        torch.save({'model': state, 'task': task, 'protocol': settings,
                                    'checkpoint_sha256': CHECKPOINT_SHA256}, output / f'task{task}_state.pt')
                        loaded = torch.load(output / f'task{task}_state.pt', map_location='cpu', weights_only=True)
                        model.net.load_state_dict(loaded['model'], strict=True)
                        if tensor_digest(model.net.state_dict()) != before or not bool(model.net.task_finalized):
                            checks['checkpoint_round_trip'] = False
                            raise RuntimeError('EASE native checkpoint round-trip changed state')
                        del state, loaded
                    recorder.snapshots[f'task{task}_after_end_task'] = model_snapshot(model)
                    record['task_results'].append(row)
                    write_json(output / 'task_results.json', record['task_results'])
                    if accepted is not None and task == 2:
                        prefix = evaluate_acceptance(record['task_results'], spec['accuracy_policy'],
                                                     structural_checks={**checks, 'training_protocol_complete': True})
                        if prefix['status'] != 'pass':
                            raise RuntimeError('EASE full CL first-two-task accuracy no longer passes')
                        record['prefix_validation'].update(status='pass', acceptance=prefix)
                    if accepted is not None:
                        write_json(output / 'prefix_validation.json', record['prefix_validation'])
                resource_row = resources.finish(task)
                record['per_task_resources'] = resources.rows
                write_json(output / 'per_task_resources.json', resources.rows)
                wandb.log({**flatten({f'task{task}_resources': resource_row['metrics']}),
                           f'task{task}_resources/wall_s': resource_row['wall_s']})
                wandb.log({f'native/{key}': value for key, value in row.items() if isinstance(value, (int, float))})
                record.setdefault('task_notifications', []).append({'task': task, 'status': notify_slack(
                    f'EASE native {spec["mode"]} task{task}/{settings["tasks"]}: seen={row["seen_accuracy_pct"]:.2f}%; W&B {record["wandb_url"]}')})
                return result
            return wrapped

        wrappers = [(dataset, 'get_data_loaders', loaders), (model, 'meta_begin_task', begin),
                    (model, 'meta_begin_epoch', begin_epoch), (model, 'meta_observe', observed),
                    (model.net, 'training_logits', logits), (training, 'train_single_epoch', train_epoch),
                    (model, 'end_task', prototype), (model.net.backbone, 'extract_all', extracted),
                    (model, 'forward', forwarded), (dataset, 'evaluate', evaluated)]
        for target, name, wrapper in wrappers:
            stack.enter_context(patch.object(target, name, wrapper(getattr(target, name))))
        training.train(model, dataset, args)
        checks['training_protocol_complete'] = completion_checks(record, recorder, model, settings)
        if not checks['training_protocol_complete']:
            raise RuntimeError('EASE native returned before all approved epochs/tasks/work completed')
        if accepted is not None and record['prefix_validation']['status'] != 'pass':
            raise RuntimeError('EASE full CL prefix validation is incomplete')
        if spec['mode'] == 'validation':
            acceptance = evaluate_acceptance(record['task_results'], spec['accuracy_policy'], structural_checks=checks)
            record.update(acceptance_status=acceptance['status'], acceptance=acceptance)
            write_json(output / 'acceptance.json', acceptance)
        else:
            record['acceptance_status'] = 'not_applicable'
        record['status'], code = 'completed', 0
    except BaseException as error:
        logging.exception('Native EASE shared-input execution failed')
        record['error_type'] = type(error).__name__
    finally:
        stack.close()
        record['tasks_results'] = record['task_results']
        record['computation_status'] = record['status']
        recorder.stop()
        recorder.samples.sort(key=lambda sample: sample['elapsed_s'])
        record = recorder.save(record)
        record['counts'].update(outer_observe_steps=recorder.steps, outer_training_stream_examples=recorder.training_examples,
                                evaluation_examples=recorder.evaluation_examples, tasks_completed=recorder.tasks_completed)
        if resources is not None:
            resources.capture_cuda()
            record['metrics'].update(resources.global_cuda)
            record['measurement_notes']['cuda_memory_scope'] = 'Per-task allocator resets; process maximum preserves initialization and all tasks.'
            record['measurement_notes']['task_resource_scope'] = 'Data loading through evaluation/checkpoint audit; excludes subsequent W&B and Slack calls.'
        record['measurement_notes']['adapter_image_pairs'] = 'Actual image exposures multiplied by traversed adapter spaces; work proxy, not FLOPs.'
        write_json(output / 'artifact_metrics.json', record)
        if wandb is not None and wandb.run is not None:
            try:
                wandb.run.summary.update({**{key: value for key, value in flatten({'native_resources': record['metrics'],
                    'native_counts': record['counts'], 'native_accuracy': record['task_results'][-1] if record['task_results'] else {},
                    'native_acceptance': record.get('acceptance', {})}).items() if isinstance(value, (int, float, bool))},
                    'native_status': record['status'], 'acceptance_status': record['acceptance_status'], 'paper_reproduction': False})
                artifact = wandb.Artifact(spec['name'], type='cl-ease-native-result')
                pins = {}
                for filename in ('spec.json', 'metrics.json', 'acceptance.json', 'effective_args.json', 'epoch_results.json',
                                 'task_results.json', 'shared_data.json', 'telemetry.csv', 'phases.json', 'per_task_resources.json',
                                 'prefix_validation.json', 'first_batch_audits.json',
                                 *(f'task{task}_predictions.npz' for task in range(1, settings['tasks'] + 1))):
                    local = output / ('artifact_metrics.json' if filename == 'metrics.json' else filename)
                    if local.is_file():
                        artifact.add_file(str(local), name=filename)
                        pins[filename] = {'local_file': local.name, 'sha256': file_hash(local)}
                record['wandb_artifact_files'] = pins
                write_json(output / 'artifact_files.json', pins)
                artifact.add_file(str(output / 'artifact_files.json'), name='artifact_files.json')
                logged = wandb.run.log_artifact(artifact)
                logged.wait()
                record['wandb_artifact'] = {'name': logged.qualified_name, 'state': logged.state}
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record['wandb_logging_status'] = 'failed:' + type(error).__name__
                record['status'], code = 'logging_failed', 1
            finally:
                try:
                    wandb.finish(exit_code=code)
                except Exception as error:
                    record['wandb_logging_status'] = 'finish_failed:' + type(error).__name__
                    record['status'], code = 'logging_failed', 1
        elif record['computation_status'] == 'completed':
            record['wandb_logging_status'] = 'run_missing_before_artifact_save'
            record['status'], code = 'logging_failed', 1
        write_json(output / 'result.json', record)
        write_json(output / 'metrics.json', record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('validation', 'full_cl'), default='validation')
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--reference-summary', type=Path)
    parser.add_argument('--acceptance-summary', type=Path)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=43200.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker:
        logging.basicConfig(level=logging.INFO)
        return worker(options.worker)
    if not options.execute:
        print(json.dumps({'experiment_executed': False, 'protocol': protocol(options.mode),
                          'required': '--reference-summary' if options.mode == 'validation' else '--acceptance-summary'}))
        return 0
    spec = build_spec(options)
    logging.basicConfig(level=logging.INFO)
    return execute(options, spec)


if __name__ == '__main__':
    sys.exit(main())
