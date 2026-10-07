"""Independent immutable low-rank reference, with opt-in first-two-task runs.

Only local weight loading, loader order, RNG and instrumentation are adapted.
The source's learner, loss, optimizer steps and numerical forwards run directly
from the reviewed external checkout. A default invocation never trains.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, flatten, notify_slack, valid_slack_webhook, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.low_rank_protocol import (
    CHECKPOINT, CHECKPOINT_SHA256, CANONICAL_PRETRAINED_SHA256, METHODS,
    PRECISION, REFERENCE_COMMIT, REQUIRED_STRUCTURAL_CHECKS,
    audit_method_state, canonical_pretrained_digest, common_transforms, completion_checks,
    configure_epoch_loader, counts_from_predictions, file_hash, learned_state,
    protocol, seed_phase, source_hashes, tensor_digest, verify_reference_source,
)


def source_code(source, method, checkpoint):
    """Load original classes in an isolated worker's original namespaces."""
    verify_reference_source(source)
    import timm  # Load timm before replacing this worker's top-level packages.
    for name in ('models', 'methods', 'utils', 'dataloaders'):
        if name in sys.modules:
            raise RuntimeError('Independent reference cannot share native model namespaces')
        package = types.ModuleType(name)
        package.__path__ = [str(Path(source) / name)]
        sys.modules[name] = package
    module = importlib.import_module('methods.' + METHODS[method])
    network_module = importlib.import_module('models.net_' + METHODS[method])
    # This is the only shared numerical adapter: 150 canonical NPZ tensors.
    spec = importlib.util.spec_from_file_location(
        '_low_rank_local_npz', ROOT / 'models/low_rank_utils/pretrained.py')
    weights = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weights)

    def local_encoder(variant, pretrained=False, **kwargs):
        # The agreed IN21K->IN1K family has a 1000-way unused source head.
        kwargs['num_classes'] = 0 if method == 'cl-lora' else 1000
        encoder = network_module.ViT(**kwargs)
        weights.load_pretrained_encoder(encoder, checkpoint)
        return encoder

    network_module._create_vision_transformer = local_encoder
    name = {'ewc-lora': 'EWCLoRA', 'sd-lora': 'SDLoRA',
            'inf-lora': 'InfLoRA', 'cl-lora': 'CLLoRA'}[method]
    return module, getattr(module, name)


def author_args(source, method, device):
    settings = json.loads((Path(source) / f'configs/{METHODS[method]}/cifar100.json').read_text())
    settings.update(seed=[1993], batch_size=64, num_workers=4, device=[device],
                    data_path='/mnt/sdab1/share/data')
    return settings


def precision():
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_initial(options):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    configure_native_environment()
    import torch
    precision()
    if file_hash(options.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Independent reference requires the pinned local Base NPZ')
    seed_phase(1993, 0, 'task_begin')
    _, constructor = source_code(options.reference_source, options.method, options.checkpoint)
    learner = constructor(author_args(options.reference_source, options.method, torch.device('cpu')))
    canonical = canonical_pretrained_digest(learner.network.image_encoder)
    if canonical != CANONICAL_PRETRAINED_SHA256:
        raise RuntimeError('Independent source pretrained encoder differs from the pin')
    catalog = {'metadata': {'method': options.method, 'reference_commit': REFERENCE_COMMIT,
                           'checkpoint_sha256': CHECKPOINT_SHA256},
               'learned_state': {name: tensor.detach().cpu().clone()
                                 for name, tensor in learned_state(learner.network).items()}}
    destination = options.prepare_initial_state.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(catalog, destination)
    metadata = {**catalog['metadata'], 'file_sha256': file_hash(destination),
                'initial_learned_sha256': tensor_digest(catalog['learned_state']),
                'canonical_pretrained_sha256': canonical, 'training_executed': False,
                'image_forwards': 0, 'optimizer_steps': 0, 'gpu_used': False,
                'origin': 'Immutable common author constructors; canonical local IN21K->IN1K NPZ'}
    write_json(destination.with_suffix('.json'), metadata)
    print(json.dumps({'initial_state': str(destination), **metadata}))
    return 0


def run_spec(options, profile):
    if profile['gpu'] != '1' or options.initial_state is None:
        raise ValueError('Independent validation requires physical GPU1 and matched initial state')
    hashes = verify_reference_source(options.reference_source)
    if file_hash(options.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Independent validation requires the pinned Base checkpoint')
    name = options.method + '-reference-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (options.result_dir or options.output / name).resolve()
    return {'schema_version': 1, 'name': name, 'method': options.method, 'model': options.method,
            'mode': 'validation', 'implementation': 'pinned_author_common_input_reference',
            'protocol': protocol(options.method), 'precision': PRECISION,
            'reference_source': str(options.reference_source.resolve()),
            'reference_commit': REFERENCE_COMMIT, 'reference_source_sha256': hashes,
            'checkpoint_file': str(options.checkpoint.resolve()), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
            'initial_state': str(options.initial_state.resolve()),
            'initial_state_sha256': file_hash(options.initial_state),
            'execution_source_sha256': source_hashes(), 'output': str(output),
            'gpu': '1', 'data_root': profile['data_root'], 'wandb_entity': profile['wandb_entity'],
            'wandb_project': profile['wandb_project'], 'paper_reproduction': False,
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                          capture_output=True, check=True).stdout.strip(),
            'compatibility_adaptations': ['local canonical150 NPZ loader and unused1000-way head',
                'stable raw IDs and explicit phase RNG; no downloads',
                'CL pre-KD CE graph retained with allow_mutation_on_saved_tensors']}


def execute(options, profile):
    spec = run_spec(options, profile)
    if not valid_slack_webhook():
        raise ValueError('Independent validation requires Slack credentials')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        status = notify_slack(message)
        notifications.append({'message': message, 'status': status})
        write_json(output / 'notifications.json', notifications)
        return status

    environment = os.environ.copy()
    environment.pop('MAMMOTH_DEVICE', None)
    environment.update(CUDA_VISIBLE_DEVICES='1', CL_LOW_RANK_REFERENCE_EXECUTE='1',
                       HF_HUB_OFFLINE='1', WANDB_MODE='online', WANDB_DIR=str(output))
    if notify(f'{spec["method"]} independent reference started: GPU1, first2 tasks, batch64, {spec["protocol"]["n_epochs"]} epochs/task') != 'sent':
        raise RuntimeError('Reference start notification failed before training')
    start = heartbeat = time.monotonic()
    timed_out = False
    from scripts.benchmark_cl import stop_worker
    with (output / 'console.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
                                   cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - start >= options.timeout:
                    timed_out = True
                    stop_worker(process)
                    break
                if time.monotonic() - heartbeat >= 600:
                    notify(f'{spec["method"]} independent reference running: {(time.monotonic() - start) / 60:.0f} min; local progress.json')
                    heartbeat = time.monotonic()
                time.sleep(2)
        except BaseException:
            stop_worker(process)
            raise
    summary_file = output / 'summary.json'
    summary = json.loads(summary_file.read_text()) if summary_file.is_file() else {'status': 'failed'}
    result = {**summary, 'exit_code': process.returncode, 'timed_out': timed_out}
    write_json(output / 'execution.json', result)
    notified = notify(f'{spec["method"]} independent reference {summary.get("status")}: {summary.get("wandb_url", "local console.log")}')
    print(json.dumps({'output': str(output), 'status': summary.get('status'), 'wandb_url': summary.get('wandb_url')}))
    return 0 if process.returncode == 0 and summary.get('status') == 'completed' and notified == 'sent' else 1


def worker(spec_path):
    if os.environ.get('CL_LOW_RANK_REFERENCE_EXECUTE') != '1':
        raise ValueError('Reference worker needs an explicit authorized --execute invocation')
    spec = json.loads(Path(spec_path).read_text())
    method = spec['method']
    if spec['protocol'] != protocol(method) or spec['gpu'] != '1':
        raise ValueError('Independent reference protocol changed')
    if os.getenv('CUDA_VISIBLE_DEVICES') != '1' or os.getenv('WANDB_MODE') != 'online':
        raise ValueError('Independent reference requires physical GPU1 and online W&B')
    verify_reference_source(spec['reference_source'])
    if (file_hash(spec['checkpoint_file']) != CHECKPOINT_SHA256
            or file_hash(spec['initial_state']) != spec['initial_state_sha256']
            or any(file_hash(ROOT / name) != pin for name, pin in spec['execution_source_sha256'].items())):
        raise ValueError('Independent execution files changed after freezing')
    configure_native_environment()
    import numpy as np
    import torch
    import wandb
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import CIFAR100
    from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
    from scripts.low_rank_protocol import load_initial_state, validate_prediction_archive
    from scripts.hide_protocol import validate_shared_data
    from unittest.mock import patch
    validate_shared_data(spec['data_root'])
    precision()
    output = Path(spec['output'])
    recorder = Recorder(output, spec['gpu'])
    recorder.torch, recorder.device = torch, torch.device('cuda:0')
    recorder.start()
    record = {**spec, 'status': 'failed', 'structural_checks': {}, 'task_results': [],
              'epoch_results': [], 'first_batch_audits': [], 'data_snapshots': {},
              'counts': {'observe_calls': 0, 'optimizer_updates': 0,
                         'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
              'state_audits': {'tasks': []}}
    stack, code = ExitStack(), 1
    try:
        wandb.init(entity=spec['wandb_entity'], project=spec['wandb_project'], name=spec['name'],
                   group=method + '-validation', job_type='author-reference', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        seed_phase(1993, 0, 'task_begin')
        source_module, constructor = source_code(spec['reference_source'], method, spec['checkpoint_file'])
        learner = constructor(author_args(spec['reference_source'], method, torch.device('cuda:0')))
        catalog = torch.load(spec['initial_state'], map_location='cpu', weights_only=True)
        record['initial_learned_sha256'] = load_initial_state(learner.network, catalog, method)
        canonical = canonical_pretrained_digest(learner.network.image_encoder)
        from scripts.ease_size_protocol import canonical_pretrained_state
        frozen_names = {'image_encoder.' + name for name in canonical_pretrained_state(learner.network.image_encoder)}
        record['state_audits']['initial_frozen_backbone_sha256'] = canonical
        if canonical != CANONICAL_PRETRAINED_SHA256:
            raise RuntimeError('Author initialization pretrained tensors do not match the pin')
        checks = record['structural_checks']
        checks.update({name: True for name in REQUIRED_STRUCTURAL_CHECKS})
        checks['training_protocol_complete'] = False
        original_backward = torch.Tensor.backward

        def finite_backward(loss, *args, **kwargs):
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError('Author loss is nonfinite')
            return original_backward(loss, *args, **kwargs)

        stack.enter_context(patch.object(torch.Tensor, 'backward', finite_backward))
        training_transform, test_transform = common_transforms()
        raw = {split: CIFAR100(str(Path(spec['data_root']) / 'CIFAR100'), train=split == 'train', download=False)
               for split in ('train', 'test')}
        write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))

        class SourceDataset(Dataset):
            def __init__(self, classes, split, mode):
                original = raw[split]
                labels = np.asarray(original.targets, dtype=np.int64)
                self.global_ids = np.flatnonzero(np.isin(labels, classes)).astype(np.int64)
                self.data, self.targets = original.data[self.global_ids], labels[self.global_ids]
                self.transform = training_transform if mode == 'train' else test_transform

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, position):
                return int(self.global_ids[position]), self.transform(Image.fromarray(self.data[position])), int(self.targets[position])

        class DataManager:
            def get_task_size(self, task):
                return 10

            def get_dataset(self, classes, source, mode):
                return SourceDataset(classes, source, mode)

        dm = DataManager()
        current = {'epoch': None, 'stage': None, 'batch': None, 'batch_number': 0,
                   'update': 0, 'started': None, 'samples': 0}
        original_build_optimizer = learner.build_optimizer

        def optimizer(parameters):
            opt, scheduler = original_build_optimizer(parameters)

            def step_before(subject, args, kwargs):
                gradients = {}
                if any(parameter.requires_grad or parameter.grad is not None
                       for name, parameter in learner.network.named_parameters() if name in frozen_names):
                    raise RuntimeError('Author frozen canonical backbone acquired gradients')
                for name, parameter in learner.network.named_parameters():
                    if parameter.grad is None or not parameter.requires_grad:
                        continue
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError('Author nonfinite gradient')
                    gradients[name] = parameter.grad
                if not gradients:
                    raise RuntimeError('Author update lacks gradients')
                current['update'] += 1
                record['counts']['optimizer_updates'] += 1
                if current['epoch'] == 0 and current['batch_number'] == 1:
                    _, images, labels = current['batch']
                    record['first_batch_audits'].append({
                        'task': learner.cur_task + 1, 'epoch': 1,
                        'seed': seed_phase_value(learner.cur_task), 'samples': len(labels),
                        'optimizer_update_index': current['update'],
                        'input_sha256': tensor_digest({'inputs': images}),
                        'labels_sha256': tensor_digest({'labels': labels}),
                        'gradient_sha256': tensor_digest(gradients)})
            stack.callback(opt.register_step_pre_hook(step_before).remove)
            return opt, scheduler

        learner.build_optimizer = optimizer

        def seed_phase_value(task):
            from scripts.low_rank_protocol import phase_seed
            return phase_seed(1993, task, 'train', 0)

        class ObservedLoader:
            def __init__(self, loader):
                self.loader = loader

            def __len__(self):
                return len(self.loader)

            def __iter__(self):
                for batch in self.loader:
                    if current['epoch'] is not None:
                        current['batch'], current['update'] = batch, 0
                        current['batch_number'] += 1
                        current['samples'] += len(batch[2])
                        record['counts']['observe_calls'] += 1
                        recorder.steps += 1
                        recorder.training_examples += len(batch[2])
                    elif current['stage'] in ('task_begin', 'task_end'):
                        record['counts'][current['stage'] + '_extra_samples'] += len(batch[2])
                    if bool(((batch[2] < learner.cur_task * 10) | (batch[2] >= (learner.cur_task + 1) * 10)).any()):
                        raise RuntimeError('Source training/statistics read old or future samples')
                    yield batch

        def build_train(manager):
            view = manager.get_dataset(np.arange(learner.known_classes, learner.total_classes), 'train', 'train')
            snapshot = dataset_fingerprint(view.data, view.targets, view.global_ids, view.transform)
            record['data_snapshots'][f'task{learner.cur_task + 1}_train'] = snapshot
            loader = DataLoader(view, batch_size=64, shuffle=True, num_workers=4, drop_last=False)
            configure_epoch_loader(loader, 1993, learner.cur_task, 'task_begin')
            learner.train_loader = ObservedLoader(loader)

        learner.build_train_loader = build_train

        class EpochProgress:
            def __init__(self, values, **kwargs):
                self.values = values

            def __iter__(self):
                for epoch in self.values:
                    seed_phase(1993, learner.cur_task, 'train', epoch)
                    configure_epoch_loader(learner.train_loader.loader, 1993, learner.cur_task, 'train', epoch)
                    current.update(epoch=epoch, stage='train', batch_number=0, update=0,
                                   started=time.monotonic(), samples=0)
                    yield epoch
                    torch.cuda.synchronize()
                    row = {'task': learner.cur_task + 1, 'epoch': epoch + 1,
                           'steps': current['batch_number'], 'samples': current['samples'],
                           'wall_s': time.monotonic() - current['started']}
                    record['epoch_results'].append(row)
                    wandb.log({'task': row['task'], 'epoch': row['epoch'], 'epoch_wall_s': row['wall_s']})
                    write_json(output / 'progress.json', {'status': 'running', **row})
                    current['epoch'] = None
                    current['stage'] = None

            def set_description(self, text):
                print(text, flush=True)

        original_progress = source_module.tqdm
        source_module.tqdm = lambda values, **kwargs: EpochProgress(values, **kwargs) if isinstance(values, range) else original_progress(values, **kwargs)
        # Controlled phase seeding wraps source routines, never their math.
        if method == 'inf-lora':
            for name, phase in (('init_drm', 'task_begin'), ('psv_info', 'task_end')):
                original = getattr(learner, name)

                def seeded(loader, original=original, phase=phase):
                    seed_phase(1993, learner.cur_task, phase)
                    configure_epoch_loader(loader.loader, 1993, learner.cur_task, phase)
                    current['stage'] = phase
                    return original(loader)
                setattr(learner, name, seeded)
        if method == 'cl-lora':
            def prototypes(manager):
                seed_phase(1993, learner.cur_task, 'task_end')
                current['stage'] = 'task_end'
                view = manager.get_dataset(np.arange(learner.known_classes, learner.total_classes), 'train', 'test')
                loader = DataLoader(view, batch_size=64, shuffle=True, num_workers=4)
                configure_epoch_loader(loader, 1993, learner.cur_task, 'task_end')
                learner.train_loader_for_protonet = ObservedLoader(loader)
            learner.build_train_loader_for_protonet = prototypes

        for task in range(2):
            seed_phase(1993, task, 'task_begin')
            current['stage'] = 'task_begin'
            learner.before_task(dm)
            with recorder.phase(f'task{task + 1}_train'):
                if method == 'cl-lora':
                    # The source CE graph predates the separate KD update.
                    with torch.autograd.graph.allow_mutation_on_saved_tensors():
                        learner.incremental_train(dm)
                else:
                    learner.incremental_train(dm)
            if canonical_pretrained_digest(learner.network.image_encoder) != canonical:
                raise RuntimeError('Author pretrained backbone changed')
            seed_phase(1993, task, 'evaluation')
            view = dm.get_dataset(np.arange(0, (task + 1) * 10), 'test', 'test')
            record['data_snapshots'][f'task{task + 1}_test'] = dataset_fingerprint(view.data, view.targets, view.global_ids, view.transform)
            test = DataLoader(view, batch_size=64, shuffle=False, num_workers=4)
            configure_epoch_loader(test, 1993, task, 'evaluation')
            learner.network.eval()
            predictions, labels, ids = [], [], []
            with recorder.phase(f'task{task + 1}_evaluation'), torch.no_grad():
                for batch_ids, images, targets in test:
                    logits = learner.network.interface(images.to(learner.device))
                    if logits.shape != (len(targets), (task + 1) * 10) or not bool(torch.isfinite(logits).all()):
                        raise RuntimeError('Source inference must expose finite seen-class logits')
                    predictions.append(logits.argmax(1).cpu().numpy())
                    labels.append(targets.numpy())
                    ids.append(batch_ids.numpy())
                    recorder.evaluation_examples += len(targets)
            destination = output / f'task{task + 1}_predictions.npz'
            np.savez_compressed(destination, predictions=np.concatenate(predictions), labels=np.concatenate(labels), global_ids=np.concatenate(ids))
            counts = validate_prediction_archive(destination, task + 1)
            record['task_results'].append({'task': task + 1, **counts})
            record.setdefault('prediction_files', []).append({'task': task + 1, 'path': str(destination), 'sha256': file_hash(destination)})
            # The pinned source evaluates before after_task. EWC's separate
            # A/B accumulation changes logits after task2, so keep this order.
            seed_phase(1993, task, 'task_end')
            current['stage'] = 'task_end'
            configure_epoch_loader(learner.train_loader.loader, 1993, task, 'task_end')
            with recorder.phase(f'task{task + 1}_finalize'):
                learner.after_task()
            audit_method_state(learner.network, learner, method, task + 1)
            algorithm = {}
            for name in ('omega_W', 'feature_list', 'feature_mat'):
                for index, tensor in enumerate(getattr(learner, name, [])):
                    algorithm[f'{name}.{index}'] = torch.as_tensor(tensor)
            if any(not bool(torch.isfinite(value).all()) for value in algorithm.values()):
                raise RuntimeError('Author algorithm state is nonfinite')
            state = learner.network.state_dict()
            if any(not bool(torch.isfinite(value).all()) for value in state.values()):
                raise RuntimeError('Author network state is nonfinite')
            record['state_audits']['tasks'].append({'task': task + 1,
                'frozen_backbone_sha256': canonical_pretrained_digest(learner.network.image_encoder),
                'network_sha256': tensor_digest(state), 'algorithm_sha256': tensor_digest(algorithm),
                'algorithm_state_shapes': {name: list(value.shape) for name, value in algorithm.items()},
                'network_task_id': learner.network._cur_task, 'seen_classes': (task + 1) * 10})
            saved_state = {'network': {name: value.detach().cpu() for name, value in state.items()},
                        'algorithm': algorithm, 'project_type': getattr(learner, 'project_type', None),
                        'cur_task': learner.cur_task, 'known_classes': learner.known_classes}
            checkpoint_path = output / f'task{task + 1}_state.pt'
            torch.save(saved_state, checkpoint_path)
            restored = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if tensor_digest(restored['network']) != tensor_digest(saved_state['network']) or tensor_digest(restored['algorithm']) != tensor_digest(algorithm):
                raise RuntimeError('Source checkpoint serialization changed network or algorithm tensors')
            record['state_audits']['tasks'][-1].update(checkpoint_path=str(checkpoint_path), checkpoint_sha256=file_hash(checkpoint_path))
            recorder.tasks_completed += 1
            wandb.log({'task': task + 1, **flatten(counts, 'accuracy')})
            write_json(output / 'progress.json', {'status': 'running', 'task_completed': task + 1, 'counts': counts})
            if notify_slack(f'{method} independent reference task{task + 1}/2 completed: seen={100 * counts["seen_correct"] / counts["seen_samples"]:.2f}%; {record["wandb_url"]}') != 'sent':
                raise RuntimeError('Required reference task notification failed')
        completion_checks(record, method)
        checks['training_protocol_complete'] = True
        record['status'] = 'completed'
        code = 0
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stack.close()
        recorder.data_snapshots = record['data_snapshots']
        record = recorder.save(record)
        record['computation_status'] = record['status']
        write_json(output / 'summary.json', record)
        if wandb.run is not None:
            try:
                wandb.run.summary.update(flatten({'status': record['status'], 'counts': record['counts'], 'resources': record['metrics']}))
                for row in record['task_results']:
                    wandb.run.summary.update({f'task{row["task"]}/{name}': value
                                              for name, value in row.items() if name != 'task'})
                artifact = wandb.Artifact(spec['name'] + '-record', type='cl-validation', metadata={
                    'checkpoint_sha256': CHECKPOINT_SHA256, 'reference_commit': REFERENCE_COMMIT,
                    'local_checkpoint_sha256': {f'task{row["task"]}': row['checkpoint_sha256']
                                                for row in record['state_audits']['tasks']}})
                for name in ('summary.json', 'spec.json', 'progress.json', 'shared_data.json', 'metrics.json', 'phases.json', 'telemetry.csv'):
                    if (output / name).is_file():
                        artifact.add_file(str(output / name))
                for destination in output.glob('*predictions.npz'):
                    artifact.add_file(str(destination))
                wandb.log_artifact(artifact).wait()
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record.update(status='logging_failed', wandb_logging_status='failed:' + type(error).__name__)
                code = 1
            finally:
                try:
                    wandb.finish(exit_code=code)
                except Exception as error:
                    record.update(status='logging_failed', wandb_logging_status='finish_failed:' + type(error).__name__)
                    code = 1
        else:
            record['wandb_logging_status'] = 'unavailable'
            code = 1
        write_json(output / 'summary.json', record)
        write_json(output / 'metrics.json', record)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='ewc-lora')
    parser.add_argument('--reference-source', type=Path, default=Path('/tmp/cl-low-rank-official-20261008'))
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--prepare-initial-state', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/low_rank')
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--timeout', type=float, default=43200)
    options = parser.parse_args()
    if options.worker:
        return worker(options.worker)
    if options.prepare_initial_state:
        return prepare_initial(options)
    if not options.execute:
        print(json.dumps({'method': options.method, 'protocol': protocol(options.method), 'training_executed': False}, indent=2))
        return 0
    return execute(options, configure_environment(credentials_path=options.credentials))


if __name__ == '__main__':
    raise SystemExit(main())
