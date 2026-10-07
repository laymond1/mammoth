"""Opt-in pinned-author EASE reference using the exact observed HiDe inputs.

Dry by default. CPU-only --prepare-initial-state exports ten task adapter/head
initializations. --execute explicitly enables the two-task GPU/W&B/Slack run.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, flatten, model_snapshot, notify_slack, write_json
from scripts.ease_fullcl_protocol import (CANONICAL_PRETRAINED_SHA256, CHECKPOINT,
    CHECKPOINT_SHA256, PRECISION, REFERENCE_COMMIT, configure_epoch_loader, file_hash,
    protocol, seed_phase)

REFERENCE_FILES = ('models/ease.py', 'models/base.py', 'utils/inc_net.py', 'utils/data.py',
                   'utils/data_manager.py', 'utils/toolkit.py', 'backbone/vit_ease.py',
                   'backbone/linears.py', 'trainer.py', 'exps/ease_cifar.json')
EXECUTION_FILES = ('scripts/run_ease_common_reference.py', 'scripts/ease_common_reference_utils.py',
                   'scripts/ease_reference_utils.py', 'scripts/ease_fullcl_protocol.py',
                   'scripts/ease_size_protocol.py', 'scripts/cl_efficiency.py',
                   'datasets/configs/seq-cifar100-224/hide_prompt.yaml', 'uv.lock')


def verify_reference_source(source):
    source = Path(source).resolve()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=source, capture_output=True,
                           text=True, check=True).stdout.strip()
    if commit != REFERENCE_COMMIT or dirty:
        raise ValueError('Independent EASE source must be clean at the pinned author commit')
    return {name: file_hash(source / name) for name in REFERENCE_FILES}


def author_config(source):
    config = json.loads((Path(source) / 'exps/ease_cifar.json').read_text())
    config.update(shuffle=False, init_cls=10, increment=10, device=['0'], seed=1993,
                  init_epochs=20, later_epochs=20, batch_size=64, ffn_num=64)
    return config


def isolate_author_imports(source):
    if any(name in sys.modules for name in ('utils', 'models', 'backbone')):
        raise RuntimeError('Independent author execution requires a fresh process')
    sys.path[:] = [str(Path(source).resolve()), *[entry for entry in sys.path
        if Path(entry or os.getcwd()).resolve() != ROOT]]


def build_spec(options, profile):
    pins = verify_reference_source(options.reference_source)
    if file_hash(options.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Reference checkpoint differs from the exact approved Base NPZ')
    if options.initial_state is None or not options.initial_state.is_file():
        raise ValueError('Reference requires a previously exported all-task initialization catalog')
    if profile['gpu'] != '1':
        raise ValueError('Reference requires the authorized GPU1')
    return {'schema_version': 1, 'model': 'ease', 'mode': 'reference',
            'implementation': 'pinned_author_common_input_reference', 'protocol': protocol('validation'),
            'reference_commit': REFERENCE_COMMIT, 'reference_source': str(options.reference_source.resolve()),
            'reference_source_sha256': pins, 'reference_repository': 'https://github.com/sun-hailong/CVPR24-Ease',
            'config': author_config(options.reference_source), 'tasks': 2,
            'checkpoint_file': str(options.checkpoint.resolve()), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
            'initial_state': str(options.initial_state.resolve()), 'initial_state_sha256': file_hash(options.initial_state),
            'gpu': profile['gpu'], 'data_root': profile['data_root'],
            'wandb_entity': profile['wandb_entity'], 'wandb_project': profile['wandb_project'],
            'paper_reproduction': False, 'flops_measured': False,
            'protocol_adaptation': 'Common HiDe preprocessing, original-index sample order, phase RNG and shared learnable initialization; original author algorithm unchanged',
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                capture_output=True, text=True, check=True).stdout.strip(),
            'execution_source_sha256': {name: file_hash(ROOT / name) for name in EXECUTION_FILES}}


def prepare_initial_state(options):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    verify_reference_source(options.reference_source)
    if file_hash(options.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Initialization export requires the pinned local Base NPZ')
    from scripts.ease_common_reference_utils import export_initialization_catalog, tensor_digest, validate_initialization_catalog
    import torch
    torch.set_num_threads(2)
    isolate_author_imports(options.reference_source)
    from backbone.vit_ease import Adapter
    from backbone.linears import CosineLinear
    catalog = export_initialization_catalog(
        lambda: Adapter(d_model=768, bottleneck=64, dropout=.1, init_option='lora',
                        adapter_scalar='0.1', adapter_layernorm_option='none'),
        lambda: CosineLinear(768, 10), options.prepare_initial_state,
        metadata={'reference_commit': REFERENCE_COMMIT, 'checkpoint_sha256': CHECKPOINT_SHA256})
    validate_initialization_catalog(catalog)
    sidecar = {'file': str(options.prepare_initial_state.resolve()),
               'file_sha256': file_hash(options.prepare_initial_state), **catalog['metadata'],
               'task_learned_sha256': {task: tensor_digest({
                   **{f'adapter.{name}': value for name, value in state['adapter'].items()},
                   **{f'proxy_head.{name}': value for name, value in state['proxy_head'].items()}})
                   for task, state in catalog['tasks'].items()},
               'training_executed': False, 'gpu_used': False, 'downloaded_weights': False}
    write_json(options.prepare_initial_state.with_suffix('.json'), sidecar)
    print(json.dumps(sidecar))
    return 0


def execute(options, profile):
    spec = build_spec(options, profile)
    output = options.result_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    name = 'ease-author-common-input-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    copied = output / 'source'
    shutil.copytree(options.reference_source, copied, ignore=shutil.ignore_patterns('.git', 'logs', '__pycache__'))
    spec.update(name=name, output=str(output), source=str(copied))
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        notifications.append({'message': message, 'status': notify_slack(message)})
        write_json(output / 'notifications.json', notifications)

    environment = os.environ.copy()
    environment.pop('MAMMOTH_DEVICE', None)
    environment.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', WANDB_MODE='online',
                       WANDB_DIR=str(output), CL_EASE_COMMON_REFERENCE_EXECUTE='1')
    from scripts.benchmark_cl import stop_worker
    started, heartbeat, timed_out = time.monotonic(), time.monotonic(), False
    notify('EASE author common-input reference started: Base/adapter64; 2 tasks × 20 epochs; batch64; GPU1')
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
                                   cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= options.timeout:
                    timed_out = True; stop_worker(process); break
                if time.monotonic() - heartbeat >= 600:
                    progress = json.loads((output / 'progress.json').read_text()) if (output / 'progress.json').exists() else {}
                    notify(f'EASE author common-input running: task{progress.get("task", "?")}/2 epoch{progress.get("epoch", "?")}/20; W&B {progress.get("wandb_url", "initializing")}')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify('EASE common-input reference interrupted; partial local observations preserved')
            raise
    result = json.loads((output / 'result.json').read_text()) if (output / 'result.json').exists() else {'status': 'failed'}
    status = 'timeout' if timed_out else result['status']
    if process.returncode != 0 and status == 'completed':
        status = 'failed'
    write_json(output / 'completion.json', {'status': status, 'exit_code': process.returncode,
        'timeout': timed_out, 'wall_s': time.monotonic() - started, 'wandb_url': result.get('wandb_url')})
    notify(f'EASE author common-input reference {status}; W&B {result.get("wandb_url", "unavailable")}')
    print(json.dumps({'status': status, 'result': str(output / 'result.json'), 'wandb_url': result.get('wandb_url')}))
    return 0 if process.returncode == 0 and status == 'completed' else 1


def worker(spec_file):
    if os.getenv('CL_EASE_COMMON_REFERENCE_EXECUTE') != '1':
        raise ValueError('The reference worker requires explicit execution authorization')
    spec = json.loads(spec_file.read_text())
    if spec['protocol'] != protocol('validation') or spec['config'] != author_config(spec['source']):
        raise ValueError('Reference worker protocol differs from the frozen approved settings')
    for name, pin in spec['execution_source_sha256'].items():
        if file_hash(ROOT / name) != pin:
            raise ValueError('Reference orchestration source changed after preparation')
    for name, pin in spec['reference_source_sha256'].items():
        if file_hash(Path(spec['source']) / name) != pin:
            raise ValueError('Copied author algorithm source differs from its reviewed pin')
    if file_hash(spec['checkpoint_file']) != CHECKPOINT_SHA256 or file_hash(spec['initial_state']) != spec['initial_state_sha256']:
        raise ValueError('Reference initialization or checkpoint changed after preparation')
    output = Path(spec['output'])
    recorder = Recorder(output, spec['gpu'])
    recorder.start()
    record = {**spec, 'status': 'failed', 'task_results': [], 'epoch_results': [],
              'first_batch_audits': [], 'prototype_adapter_image_pairs': 0, 'structural_checks': {}}
    stack, wandb, code, handles = None, None, 1, []
    try:
        import numpy as np
        import torch
        import timm
        import wandb
        from scripts.ease_common_reference_utils import (completion_checks, first_batch_audit,
            install_common_reference_patches, load_reference_task_initialization,
            tensor_digest, validate_initialization_catalog)
        from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
        from scripts.ease_size_protocol import canonical_pretrained_digest
        from scripts.ease_acceptance import counts_from_predictions
        isolate_author_imports(spec['source'])
        from utils import data, data_manager, inc_net
        from backbone import vit_ease
        from models import ease
        from trainer import _set_device
        modules = {'timm': timm, 'data': data, 'data_manager': data_manager,
                   'ease': ease, 'vit': vit_ease, 'inc_net': inc_net}
        patches = {}
        stack = install_common_reference_patches(modules, spec['checkpoint_file'], spec['data_root'], patches)
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
            'tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'autocast': False, 'parameter_dtype': 'float32', 'torch_cpu_threads': torch.get_num_threads()}
        if record['precision'] != PRECISION:
            raise RuntimeError('Author reference precision settings differ from the shared protocol')
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'wandb')}
        catalog = validate_initialization_catalog(torch.load(spec['initial_state'], map_location='cpu', weights_only=True))
        if (catalog['metadata'].get('reference_commit') != REFERENCE_COMMIT
                or catalog['metadata'].get('checkpoint_sha256') != CHECKPOINT_SHA256):
            raise ValueError('Reference initialization catalog has incompatible source/checkpoint metadata')
        config = spec['config'].copy()
        _set_device(config)
        wandb.init(entity=spec['wandb_entity'], project=spec['wandb_project'], name=spec['name'],
                   group='ease-base-common-input', job_type='author-reference', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        write_json(output / 'progress.json', {'message': 'initializing pinned EASE author reference', 'wandb_url': record['wandb_url']})
        write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
        with recorder.phase('data_loading'):
            manager = data_manager.DataManager('cifar224', False, 1993, 10, 10, config)
        config['nb_classes'], config['nb_tasks'] = manager.nb_classes, manager.nb_tasks
        with recorder.phase('initialize'):
            learner = ease.Learner(config)
        backbone = learner._network.backbone
        frozen = canonical_pretrained_digest(backbone)
        if frozen != CANONICAL_PRETRAINED_SHA256:
            raise RuntimeError('Author pretrained tensors differ from the canonical NPZ')
        record['initial_canonical_pretrained_sha256'] = frozen
        checks = record['structural_checks']
        checks.update(finite_loss_and_gradients=True, frozen_backbone_gradients_absent=True,
                      frozen_backbone_unchanged=True, old_adapters_unchanged=True,
                      prototype_and_adapter_shapes=True, training_protocol_complete=False,
                      learnable_initialization_matches=True, nonoracle_seen_class_evaluation=True,
                      canonical_pretrained_matches_pin=True)
        record['state_audits'] = {'initial_frozen_backbone_sha256': frozen, 'tasks': []}
        context, optimizer_slot, old_adapter_pins = {}, {}, {}

        original_update = learner._network.update_fc
        def update_fc(classes):
            task = learner._cur_task
            seed_phase(1993, task, 'task_begin')
            original_update(classes)
            learned_pin = load_reference_task_initialization(learner._network, catalog, task)
            record.setdefault('initialization_audits', []).append({'task': task + 1, 'learned_sha256': learned_pin})
        stack.enter_context(patch.object(learner._network, 'update_fc', update_fc))

        original_loader = ease.DataLoader
        class ObservedLoader:
            def __init__(self, loader):
                self.loader = loader
            def __len__(self):
                return len(self.loader)
            def __getattr__(self, name):
                return getattr(self.loader, name)
            def __iter__(self):
                for index, batch in enumerate(self.loader):
                    if getattr(self.dataset, '_ease_reference_mode', None) == 'train':
                        context.update(batch_index=index, inputs=batch[1], labels=batch[2])
                    yield batch
        def loader_factory(*args, **kwargs):
            return ObservedLoader(original_loader(*args, **kwargs))
        stack.enter_context(patch.object(ease, 'DataLoader', loader_factory))

        original_optimizer = learner.get_optimizer
        def get_optimizer(*args, **kwargs):
            optimizer = original_optimizer(*args, **kwargs)
            optimizer_slot['current'] = optimizer
            def before_step(subject, args, kwargs):
                named = [*(('adapter.' + name, parameter) for name, parameter in backbone.cur_adapter.named_parameters()),
                         *(('proxy_head.' + name, parameter) for name, parameter in learner._network.proxy_fc.named_parameters())]
                if any(parameter.grad is not None for name, parameter in backbone.named_parameters() if not name.startswith('cur_adapter.')):
                    checks['frozen_backbone_gradients_absent'] = False
                    raise RuntimeError('Frozen author backbone received gradients')
                if any(not bool(torch.isfinite(parameter).all()) or (parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()))
                       for name, parameter in named):
                    checks['finite_loss_and_gradients'] = False
                    raise FloatingPointError('Author parameters or gradients became nonfinite')
                if context['epoch'] == 0 and context['batch_index'] == 0:
                    audit = first_batch_audit(context['inputs'], context['labels'], named,
                        task=learner._cur_task + 1, epoch=1, seed=context['seed'])
                    record['first_batch_audits'].append(audit)
                    write_json(output / 'first_batch_audits.json', record['first_batch_audits'])
                recorder.steps += 1
            handles.append(optimizer.register_step_pre_hook(before_step))
            return optimizer
        stack.enter_context(patch.object(learner, 'get_optimizer', get_optimizer))

        original_ce = ease.F.cross_entropy
        def cross_entropy(*args, **kwargs):
            result = original_ce(*args, **kwargs)
            if recorder.active_phase == 'train_epoch':
                value = float(result.detach())
                if not math.isfinite(value):
                    raise FloatingPointError('Author loss became nonfinite')
                context['loss_sum'] += value
                context['loss_weighted_sum'] += value * len(args[1])
            return result
        stack.enter_context(patch.object(ease.F, 'cross_entropy', cross_entropy))

        def observed_forward(module, arguments, keywords, result):
            if not keywords.get('test', False):
                count = len(arguments[0])
                recorder.training_examples += count
                context['samples'] += count
                labels = context['labels'].to(result['logits'].device) - learner._known_classes
                context['correct'] += int((result['logits'].detach().argmax(1) == labels).sum())
                context['steps'] += 1
        handles.append(learner._network.register_forward_hook(observed_forward, with_kwargs=True))

        original_tqdm = ease.tqdm
        class EpochProgress:
            def __init__(self, iterable):
                self.progress = original_tqdm(iterable, disable=True)
            def __iter__(self):
                for epoch in self.progress:
                    task = learner._cur_task
                    selected = seed_phase(1993, task, 'train', epoch)
                    configure_epoch_loader(learner.train_loader.loader, 1993, task, 'train', epoch)
                    context.clear()
                    context.update(epoch=epoch, seed=selected, loss_sum=0., loss_weighted_sum=0., samples=0, steps=0, correct=0)
                    with recorder.phase('train_epoch'):
                        yield epoch
            def set_description(self, message):
                self.progress.set_description(message)
                row = {'task': learner._cur_task + 1, 'epoch': context['epoch'] + 1, 'epochs': 20,
                       'seed': context['seed'], 'loss': context['loss_sum'] / context['steps'],
                       'loss_example_weighted': context['loss_weighted_sum'] / context['samples'],
                       'train_accuracy_pct': 100 * context['correct'] / context['samples'],
                       'steps': context['steps'], 'samples': context['samples'],
                       'lr_after_scheduler': optimizer_slot['current'].param_groups[0]['lr']}
                record['epoch_results'].append(row)
                write_json(output / 'epoch_results.json', record['epoch_results'])
                write_json(output / 'progress.json', {**row, 'wandb_url': record['wandb_url']})
                wandb.log({f'reference/{name}': value for name, value in row.items()})
                print(message, flush=True)
        stack.enter_context(patch.object(ease, 'tqdm', EpochProgress))

        original_replace, original_proto = learner.replace_fc, backbone.forward_proto
        def forward_proto(inputs, *args, **kwargs):
            record['prototype_adapter_image_pairs'] += len(inputs)
            return original_proto(inputs, *args, **kwargs)
        stack.enter_context(patch.object(backbone, 'forward_proto', forward_proto))
        def replace(loader):
            seed_phase(1993, learner._cur_task, 'prototype')
            configure_epoch_loader(loader.loader, 1993, learner._cur_task, 'prototype')
            with recorder.phase('prototype_completion'):
                return original_replace(loader)
        stack.enter_context(patch.object(learner, 'replace_fc', replace))
        original_eval, latest = learner._eval_cnn, {}
        def evaluate(loader):
            seed_phase(1993, learner._cur_task, 'evaluation')
            configure_epoch_loader(loader.loader, 1993, learner._cur_task, 'evaluation')
            predictions, labels = original_eval(loader)
            recorder.evaluation_examples += len(labels)
            latest.update(predictions=predictions, labels=labels)
            return predictions, labels
        stack.enter_context(patch.object(learner, '_eval_cnn', evaluate))

        for task in range(2):
            with recorder.phase('incremental_train'):
                learner.incremental_train(manager)
            for split, dataset in (('train', learner.train_dataset), ('test', learner.test_dataset)):
                recorder.data_snapshots[f'task{task+1}_{split}'] = dataset_fingerprint(
                    dataset.images, dataset.labels, dataset.global_ids, dataset.trsf)
            with recorder.phase('evaluation'):
                author_accuracy, _ = learner.eval_task()
            predicted, labels = latest['predictions'], latest['labels']
            ids = learner.test_dataset.global_ids
            counts = counts_from_predictions(predicted[:, 0], labels, past_classes=task * 10, seen_classes=(task + 1) * 10)
            row = {'task': task + 1, 'seen_classes': (task + 1) * 10, **counts,
                   'seen_accuracy_pct': 100 * counts['seen_correct'] / counts['seen_samples'],
                   'old_accuracy_pct': 100 * counts['old_correct'] / counts['old_samples'] if counts['old_samples'] else None,
                   'new_accuracy_pct': 100 * counts['new_correct'] / counts['new_samples'],
                   'official_accuracy': author_accuracy, 'prototype_shape': list(learner._network.fc.weight.shape),
                   'observed_adapters': len(backbone.adapter_list) + 1}
            if row['prototype_shape'] != [(task + 1) * 10, (task + 1) * 768] or row['observed_adapters'] != task + 1:
                checks['prototype_and_adapter_shapes'] = False
                raise RuntimeError('Author adapter/prototype dimensions differ from the protocol')
            current_frozen = canonical_pretrained_digest(backbone)
            if current_frozen != frozen:
                checks['frozen_backbone_unchanged'] = False
                raise RuntimeError('Author frozen backbone changed during training')
            adapter_pins = {str(index): tensor_digest(adapter.state_dict()) for index, adapter in enumerate([*backbone.adapter_list, backbone.cur_adapter])}
            if any(adapter_pins[name] != pin for name, pin in old_adapter_pins.items()):
                checks['old_adapters_unchanged'] = False
                raise RuntimeError('A previously finalized author adapter changed')
            old_adapter_pins.update(adapter_pins)
            record['state_audits']['tasks'].append({'task': task + 1, 'backbone_sha256': frozen, 'adapter_sha256': adapter_pins})
            order = np.argsort(ids, kind='stable')
            np.savez(output / f'task{task+1}_predictions.npz', predictions=predicted[order], labels=labels[order],
                     global_ids=ids[order], evaluation_global_ids=ids)
            torch.save({'model': learner._network.state_dict(), 'old_adapters': [adapter.state_dict() for adapter in backbone.adapter_list],
                        'task': task + 1, 'checkpoint_sha256': CHECKPOINT_SHA256}, output / f'task{task+1}_state.pt')
            loaded = torch.load(output / f'task{task+1}_state.pt', map_location='cpu', weights_only=True)
            learner._network.load_state_dict(loaded['model'], strict=True)
            if tensor_digest(loaded['model']) != tensor_digest(learner._network.state_dict()):
                raise RuntimeError('Author task checkpoint did not round trip exactly')
            if any(tensor_digest(saved) != tensor_digest(adapter.state_dict())
                   for saved, adapter in zip(loaded['old_adapters'], backbone.adapter_list)):
                raise RuntimeError('Author checkpoint lost a previously finalized adapter')
            checks['checkpoint_round_trip'] = True
            del loaded
            record['task_results'].append(row)
            recorder.tasks_completed += 1
            from types import SimpleNamespace
            recorder.snapshots[f'task{task+1}'] = model_snapshot(SimpleNamespace(
                parameters=learner._network.parameters, buffers=learner._network.buffers, optimizer=optimizer_slot['current']))
            write_json(output / 'task_results.json', record['task_results'])
            wandb.log({f'reference/{name}': value for name, value in row.items() if isinstance(value, (int, float))})
            record.setdefault('task_notifications', []).append({'task': task + 1, 'status': notify_slack(
                f'EASE author common-input task{task+1}/2 complete: seen={row["seen_accuracy_pct"]:.2f}%; W&B {record["wandb_url"]}')})
            if task == 0:
                with recorder.phase('after_task'):
                    learner.after_task()
        checks['training_protocol_complete'] = completion_checks(record, recorder)
        record.update(status='completed', acceptance_status='reference_only', paper_reproduction=False)
        code = 0
    except BaseException as error:
        logging.exception('EASE common-input author reference failed')
        record['error_type'] = type(error).__name__
    finally:
        for handle in handles:
            handle.remove()
        if stack is not None:
            stack.close()
        if 'patches' in locals():
            record['runtime_patches'] = patches
            write_json(output / 'runtime_patches.json', patches)
        record['tasks_results'] = record['task_results']
        record['computation_status'] = record['status']
        record = recorder.save(record)
        write_json(output / 'artifact_metrics.json', record)
        if wandb is not None and wandb.run is not None:
            try:
                wandb.run.summary.update({**flatten({'reference_resources': record['metrics']}),
                    'reference_status': record['status'], 'paper_reproduction': False,
                    **{f'reference/{key}': value for key, value in (record['task_results'][-1] if record['task_results'] else {}).items()
                       if isinstance(value, (float, int))}})
                filenames = ['spec.json', 'runtime_patches.json', 'epoch_results.json', 'task_results.json',
                             'shared_data.json', 'first_batch_audits.json', 'telemetry.csv', 'phases.json',
                             'task1_predictions.npz', 'task2_predictions.npz']
                pins = {name: {'local_file': name, 'sha256': file_hash(output / name)} for name in filenames if (output / name).is_file()}
                pins['metrics.json'] = {'local_file': 'artifact_metrics.json', 'sha256': file_hash(output / 'artifact_metrics.json')}
                write_json(output / 'artifact_files.json', pins)
                artifact = wandb.Artifact(spec['name'], type='cl-ease-common-reference')
                for name, pin in pins.items():
                    artifact.add_file(str(output / pin['local_file']), name=name)
                artifact.add_file(str(output / 'artifact_files.json'), name='artifact_files.json')
                committed = wandb.run.log_artifact(artifact).wait()
                record['wandb_artifact'] = {'name': committed.qualified_name, 'state': committed.state}
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record['wandb_logging_status'] = f'failed:{type(error).__name__}'
                record['status'], code = 'logging_failed', 1
            finally:
                try:
                    wandb.finish(exit_code=code)
                except Exception as error:
                    record['wandb_logging_status'] = f'finish_failed:{type(error).__name__}'
                    record['status'], code = 'logging_failed', 1
        write_json(output / 'metrics.json', record)
        write_json(output / 'result.json', record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reference-source', type=Path, default=Path('/tmp/cl-ease-official-20261007'))
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--prepare-initial-state', type=Path)
    parser.add_argument('--result-dir', type=Path, default=ROOT / 'benchmarks/cl_efficiency/ease_full_cl/runs/author-reference')
    parser.add_argument('--timeout', type=float, default=7200.)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker:
        return worker(options.worker)
    if options.prepare_initial_state:
        if options.execute:
            raise ValueError('Prepare initialization in a separate CPU-only step')
        return prepare_initial_state(options)
    if not options.execute:
        print(json.dumps({'prepared': True, 'executed': False, 'protocol': protocol('validation')}))
        return 0
    from scripts.cl_runtime import configure_environment
    profile = configure_environment()
    logging.basicConfig(level=logging.INFO)
    return execute(options, profile)


if __name__ == '__main__':
    sys.exit(main())
