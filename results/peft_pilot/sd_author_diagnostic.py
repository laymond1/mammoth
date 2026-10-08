"""Opt-in immutable-author SD diagnostic for the existing common-EASE pilot.

One fresh first-task epoch only. This evidence is excluded from efficiency
aggregates. The author's learner, forward, loss, backward and Adam stay intact.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import datetime
import functools
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import low_rank_protocol as shared
from scripts.cl_efficiency import Recorder, flatten, notify_slack, valid_slack_webhook, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.run_low_rank_efficiency import REQUIRED_CHECKS, apply_common_transforms, audit_fingerprints, efficiency_protocol
from scripts.run_low_rank_reference import author_args, precision, source_code

DEFAULT_NATIVE = ROOT / 'results/peft_pilot/base-20261008/sd-lora/measurement1/result.json'
DEFAULT_SOURCE = Path('/tmp/cl-low-rank-official-20261008')


def diagnostic_protocol():
    value = efficiency_protocol('sd-lora')
    value.update(mode='diagnostic', measurement_repeats=0, flops_runs=0, efficiency_comparison=False)
    return value


def validate_spec(spec):
    expected = {'method': 'sd-lora', 'mode': 'diagnostic', 'tasks': 1, 'gpu': '1',
        'protocol': diagnostic_protocol(), 'precision': shared.PRECISION,
        'checkpoint_sha256': shared.CHECKPOINT_SHA256, 'reference_commit': shared.REFERENCE_COMMIT,
        'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
        'wandb_mode': 'online', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
        'measurement_aggregate_eligible': False, 'paper_reproduction': False}
    if any(spec.get(key) != value for key, value in expected.items()) or type(spec.get('tasks')) is not int:
        raise ValueError('The SD author diagnostic is restricted to the pinned one-task one-epoch common EASE recipe')
    for key in ('tasks', 'n_epochs', 'batch_size', 'num_workers', 'seed', 'rank'):
        if type(spec['protocol'].get(key)) is not int:
            raise ValueError('SD diagnostic scope requires exact integer counts')


def check_completion(record):
    expected_counts = {'observe_calls': 79, 'optimizer_updates': 79,
        'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}
    if record.get('counts') != expected_counts or any(type(record['counts'][key]) is not int for key in expected_counts):
        raise RuntimeError('SD source diagnostic requires all 79 ordinary updates and no hidden statistics passes')
    rows = record.get('epoch_results', [])
    if len(rows) != 1 or any(type(rows[0].get(k)) is not int or rows[0][k] != v
            for k, v in {'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000}.items()):
        raise RuntimeError('SD source diagnostic requires one complete first-task epoch')
    metrics = record.get('observed_totals', {})
    expected_totals = {'training_examples': 5000, 'evaluation_examples': 1000, 'steps': 79, 'tasks_completed': 1}
    if metrics != expected_totals or any(type(metrics.get(key)) is not int for key in expected_totals):
        raise RuntimeError('SD diagnostic must cover all 5000 train and 1000 test samples once')
    if (len(record.get('task_results', [])) != 1 or record['task_results'][0].get('seen_samples') != 1000
            or len(record.get('first_batch_audits', [])) != 1
            or any(record.get('structural_checks', {}).get(name) is not True for name in REQUIRED_CHECKS)):
        raise RuntimeError('SD diagnostic predictions, first update or structural evidence is incomplete')
    return True


def compare_records(source, native):
    first, expected = source['first_batch_audits'][0], native['first_batch_audits'][0]
    return {**{name + '_matches': first[name] == expected[name]
        for name in ('input_sha256', 'labels_sha256', 'gradient_sha256', 'seed', 'samples')},
        'counts_match': source['counts'] == native['counts'],
        'network_sha256_matches': source['state_audits']['network_sha256'] == native['state_audits']['network_sha256'],
        'loss_mean_abs_difference': abs(source['epoch_results'][0]['loss_mean'] - native['epoch_results'][0]['loss_mean']),
        'seen_accuracy_pp_difference': source['accuracy']['mean_accuracy_pct'] - native['accuracy']['mean_accuracy_pct']}


def cross_entropy_observer(original, record, current):
    @functools.wraps(original)
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        if current['stage'] == 'train':
            value = float(result.detach())
            if not math.isfinite(value):
                raise FloatingPointError('SD original source loss is nonfinite')
            current['loss_sum'] += value
            current['loss_calls'] += 1
        return result
    return observed


def prepare_spec(options, profile=None):
    native_path = options.native_result.resolve()
    native = json.loads(native_path.read_text())
    if (native.get('method') != 'sd-lora' or native.get('protocol') != efficiency_protocol('sd-lora')
            or native.get('status') != 'completed' or native.get('computation_status') != 'completed'
            or native.get('wandb_logging_status') != 'uploaded' or native.get('precision') != shared.PRECISION
            or any(native.get('structural_checks', {}).get(key) is not True for key in REQUIRED_CHECKS)):
        raise ValueError('SD diagnosis requires the intact completed native common-EASE pilot evidence')
    if profile is not None and any(profile[key] != native[key] for key in ('gpu', 'data_root', 'wandb_entity', 'wandb_project')):
        raise ValueError('SD diagnostic public runtime differs from the existing native pilot')
    sources = shared.verify_reference_source(options.reference_source)
    paths = {'native_result': native_path, 'native_predictions': Path(native['prediction_files'][0]['path']),
        'native_checkpoint': native_path.parent / 'task1_state.pt', 'initial_state': Path(native['initial_state']),
        'checkpoint_file': Path(native['checkpoint_file'])}
    pins = {key + '_sha256': shared.file_hash(path) for key, path in paths.items()}
    if pins['checkpoint_file_sha256'] != shared.CHECKPOINT_SHA256 or pins['initial_state_sha256'] != native['initial_state_sha256']:
        raise ValueError('SD diagnostic initial state or pretrained checkpoint differs from the native pilot')
    if (pins['native_predictions_sha256'] != native['prediction_files'][0]['sha256']
            or pins['native_checkpoint_sha256'] != native['state_audits']['checkpoint_sha256']):
        raise ValueError('SD diagnostic native predictions or saved checkpoint differ from the completed pilot')
    for entry in native['data_files']['files'].values():
        if shared.file_hash(entry['path']) != entry['sha256']:
            raise ValueError('SD diagnostic raw CIFAR100 files changed')
    execution = deepcopy(native['execution_source_sha256'])
    execution[str(Path(__file__).resolve().relative_to(ROOT))] = shared.file_hash(__file__)
    for name, digest in execution.items():
        if shared.file_hash(ROOT / name) != digest:
            raise ValueError('SD diagnostic source protection differs from the existing pilot freeze')
    name = 'sd-author-common-ease-diagnostic-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec = {'schema_version': 1, 'name': name, 'method': 'sd-lora', 'mode': 'diagnostic', 'tasks': 1,
        'implementation': 'immutable_author_sd', 'protocol': diagnostic_protocol(), 'precision': shared.PRECISION,
        'gpu': '1', 'wandb_mode': 'online', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
        'reference_source': str(options.reference_source.resolve()), 'reference_commit': shared.REFERENCE_COMMIT,
        'reference_source_sha256': sources, 'checkpoint_sha256': shared.CHECKPOINT_SHA256,
        'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
        'execution_source_sha256': execution, 'initial_learned_sha256': native['initial_learned_sha256'],
        'data_root': native['data_root'], 'data_files': native['data_files'],
        'common_reference_data_snapshots': native['common_reference_data_snapshots'],
        'common_transform_snapshots': native['common_transform_snapshots'],
        'measurement_aggregate_eligible': False, 'paper_reproduction': False, 'efficiency_comparison': False,
        'output': str(options.output.resolve()), **{key: str(path.resolve()) for key, path in paths.items()}, **pins}
    validate_spec(spec)
    return spec


def execute(options):
    spec = prepare_spec(options, configure_environment())
    if not valid_slack_webhook():
        raise ValueError('SD diagnostic requires the approved Slack credential')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []
    def notify(message):
        result = notify_slack(message)
        notifications.append({'message': message, 'status': result})
        write_json(output / 'notifications.json', notifications)
        return result
    if notify('SD immutable-author common EASE diagnostic started: first-task one epoch, GPU1, excluded from efficiency aggregates') != 'sent':
        raise RuntimeError('Required SD diagnostic start notification failed before training')
    environment = os.environ.copy()
    environment.pop('MAMMOTH_DEVICE', None)
    environment.update(CUDA_VISIBLE_DEVICES='1', WANDB_MODE='online', WANDB_DIR=str(output),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', CL_SD_AUTHOR_DIAGNOSTIC_EXECUTE='1')
    from scripts.benchmark_cl import stop_worker
    started, timed_out = time.monotonic(), False
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
            cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > options.timeout:
                    timed_out = True
                    stop_worker(process)
                    break
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            raise
    path = output / 'result.json'
    result = json.loads(path.read_text()) if path.is_file() else {'status': 'failed'}
    status = 'timeout' if timed_out else result['status']
    sent = notify(f'SD author diagnostic {status}: {result.get("diagnosis", "incomplete")}; W&B {result.get("wandb_url", "unavailable")}')
    write_json(output / 'completion.json', {'status': status, 'exit_code': process.returncode,
        'timeout': timed_out, 'wall_s': time.monotonic() - started, 'slack_completion_status': sent})
    print(json.dumps({'status': status, 'result': str(path), 'wandb_url': result.get('wandb_url')}))
    return 0 if process.returncode == 0 and status == 'completed' and sent == 'sent' else 1


def worker(spec_path):
    if (os.getenv('CL_SD_AUTHOR_DIAGNOSTIC_EXECUTE') != '1' or os.getenv('CUDA_VISIBLE_DEVICES') != '1'
            or os.getenv('WANDB_MODE') != 'online'):
        raise ValueError('SD diagnostic requires the explicit parent execution flag, GPU1 and online W&B')
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    if shared.verify_reference_source(spec['reference_source']) != spec['reference_source_sha256']:
        raise ValueError('SD immutable author source changed')
    for name, digest in spec['execution_source_sha256'].items():
        if shared.file_hash(ROOT / name) != digest:
            raise ValueError('SD diagnostic protected execution file changed')
    for name in ('native_result', 'native_predictions', 'native_checkpoint', 'initial_state', 'checkpoint_file'):
        if shared.file_hash(spec[name]) != spec[name + '_sha256']:
            raise ValueError('SD diagnostic pinned input artifact changed')
    for entry in spec['data_files']['files'].values():
        if shared.file_hash(entry['path']) != entry['sha256']:
            raise ValueError('SD diagnostic raw data changed')
    configure_native_environment()
    import numpy as np
    import torch
    import wandb
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import CIFAR100
    from types import SimpleNamespace
    from scripts.ease_reference_utils import dataset_fingerprint
    from scripts.ease_size_protocol import canonical_pretrained_state
    native = json.loads(Path(spec['native_result']).read_text())
    output = Path(spec['output'])
    precision()
    recorder = Recorder(output, '1')
    recorder.torch, recorder.device = torch, torch.device('cuda:0')
    recorder.start()
    record = {**spec, 'status': 'failed', 'structural_checks': {key: False for key in REQUIRED_CHECKS},
        'counts': {'observe_calls': 0, 'optimizer_updates': 0, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
        'first_batch_audits': [], 'epoch_results': [], 'task_results': [], 'accuracy': {}}
    code, handles = 1, []
    try:
        actual_precision = {'matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_allowed': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'autocast': torch.is_autocast_enabled(), 'parameter_dtype': str(torch.get_default_dtype()).removeprefix('torch.'),
            'torch_cpu_threads': torch.get_num_threads()}
        record['precision'] = actual_precision
        if actual_precision != shared.PRECISION:
            raise ValueError('SD source diagnostic actual precision differs from the native fp32 pin')
        wandb.init(entity='laymond1', project='cl-efficiency', mode='online', name=spec['name'],
            group='sd-author-collapse-diagnostic', job_type='non-aggregate-diagnostic', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        shared.seed_phase(1993, 0, 'task_begin')
        with recorder.phase('initialize'):
            source_module, constructor = source_code(spec['reference_source'], 'sd-lora', spec['checkpoint_file'])
            arguments = author_args(spec['reference_source'], 'sd-lora', torch.device('cuda:0'))
            arguments.update(epochs=1, batch_size=64, num_workers=4, data_path=spec['data_root'])
            expected = {'epochs': 1, 'batch_size': 64, 'num_workers': 4, 'rank': 10, 'init_cls': 10,
                'increment': 10, 'sessions': 10, 'shuffle': False, 'optimizer': 'adam', 'scheduler': 'cosine',
                'lrate': .008, 'fc_lrate': .008, 'weight_decay': 0.}
            if any(arguments.get(key) != value for key, value in expected.items()):
                raise ValueError('SD source arguments differ from the native common one-epoch recipe')
            write_json(output / 'effective_author_args.json', {k: v for k, v in arguments.items() if k != 'device'})
            learner = constructor(arguments)
            initial = shared.load_initial_state(learner.network, torch.load(spec['initial_state'], map_location='cpu', weights_only=True), 'sd-lora')
            if initial != spec['initial_learned_sha256'] or shared.canonical_pretrained_digest(learner.network.image_encoder) != shared.CANONICAL_PRETRAINED_SHA256:
                raise RuntimeError('SD source initialization or canonical pretrained tensors differ')
            if any(parameter.dtype != torch.float32 for parameter in learner.network.parameters()):
                raise ValueError('SD source diagnostic network parameters must all be fp32')
        checks = record['structural_checks']
        checks.update(learnable_initialization_matches=True, canonical_pretrained_matches_pin=True,
            finite_loss_and_gradients=True, frozen_backbone_gradients_absent=True, no_future_or_replay_samples=True)
        frozen = {'image_encoder.' + key for key in canonical_pretrained_state(learner.network.image_encoder)}
        common = SimpleNamespace()
        apply_common_transforms(common)
        raw = {split: CIFAR100(str(Path(spec['data_root']) / 'CIFAR100'), train=split == 'train', download=False) for split in ('train', 'test')}
        current = {'stage': 'task_begin', 'epoch': None, 'batch': None, 'batches': 0, 'samples': 0, 'loss_sum': 0., 'loss_calls': 0}
        class SourceDataset(Dataset):
            def __init__(self, classes, split, mode):
                if sorted(map(int, classes)) != list(range(10)):
                    raise RuntimeError('SD diagnostic requested a future task or different classes')
                labels = np.asarray(raw[split].targets, dtype=np.int64)
                self.global_ids = np.flatnonzero(np.isin(labels, classes)).astype(np.int64)
                self.data, self.targets = raw[split].data[self.global_ids], labels[self.global_ids]
                self.transform = common.TRANSFORM if mode == 'train' else common.TEST_TRANSFORM
                snapshot = dataset_fingerprint(self.data, self.targets, self.global_ids, self.transform)
                audit_fingerprints(snapshot, spec['common_reference_data_snapshots'][f'task1_{split}'], spec['common_transform_snapshots'][f'task1_{split}'])
                recorder.data_snapshots[f'task1_{split}'] = snapshot
            def __len__(self):
                return len(self.targets)
            def __getitem__(self, index):
                return int(self.global_ids[index]), self.transform(Image.fromarray(self.data[index])), int(self.targets[index])
        class DataManager:
            def get_task_size(self, task):
                if task != 0:
                    raise RuntimeError('SD diagnostic attempted a later task')
                return 10
            def get_dataset(self, classes, source, mode):
                return SourceDataset(classes, source, mode)
        class ObservedLoader:
            def __init__(self, loader):
                self.loader = loader
            def __len__(self):
                return len(self.loader)
            def __iter__(self):
                for batch in self.loader:
                    if bool(((batch[2] < 0) | (batch[2] >= 10)).any()):
                        raise RuntimeError('SD source diagnostic saw future or unlabeled samples')
                    current['batch'] = batch
                    if current['stage'] == 'train':
                        current['batches'] += 1
                        current['samples'] += len(batch[2])
                        record['counts']['observe_calls'] += 1
                        recorder.steps += 1
                        recorder.training_examples += len(batch[2])
                    yield batch
        def loader_adapter(view, **kwargs):
            loader = DataLoader(view, **kwargs)
            shared.configure_epoch_loader(loader, 1993, 0, current['stage'])
            return ObservedLoader(loader)
        original_optimizer = learner.build_optimizer
        def build_optimizer(parameters):
            optimizer, scheduler = original_optimizer(parameters)
            if any(group['lr'] != .008 or group['weight_decay'] != 0. for group in optimizer.param_groups):
                raise RuntimeError('SD source optimizer differs from the native Adam recipe')
            def before_step(subject, args, kwargs):
                gradients = {}
                for name, parameter in learner.network.named_parameters():
                    if name in frozen and (parameter.requires_grad or parameter.grad is not None):
                        raise RuntimeError('SD frozen source backbone acquired gradients')
                    if parameter.grad is not None:
                        if not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError('SD source gradient is nonfinite')
                        if parameter.requires_grad:
                            gradients[name] = parameter.grad
                if current['batches'] == 1:
                    _, images, labels = current['batch']
                    record['first_batch_audits'].append({'task': 1, 'epoch': 1, 'seed': 2993,
                        'samples': len(labels), 'optimizer_update_index': 1, 'input_sha256': shared.tensor_digest({'inputs': images}),
                        'labels_sha256': shared.tensor_digest({'labels': labels}), 'gradient_sha256': shared.tensor_digest(gradients)})
            def after_step(subject, args, kwargs):
                record['counts']['optimizer_updates'] += 1
            handles.extend((optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)))
            return optimizer, scheduler
        class EpochProgress:
            def __init__(self, values, **kwargs):
                self.values = values
            def __iter__(self):
                for epoch in self.values:
                    if epoch != 0:
                        raise RuntimeError('SD author diagnostic attempted another epoch')
                    shared.seed_phase(1993, 0, 'train', epoch)
                    shared.configure_epoch_loader(learner.train_loader.loader, 1993, 0, 'train', epoch)
                    current['stage'], current['epoch'] = 'train', epoch
                    with recorder.phase('train_epoch'):
                        yield epoch
                    if current['loss_calls'] != 79:
                        raise RuntimeError('SD original cross-entropy did not run exactly 79 times')
                    row = {'task': 1, 'epoch': 1, 'steps': current['batches'], 'samples': current['samples'],
                        'loss_sum': current['loss_sum'], 'loss_mean': current['loss_sum'] / 79.}
                    record['epoch_results'].append(row)
                    write_json(output / 'epoch_results.json', record['epoch_results'])
                    write_json(output / 'progress.json', row)
                    current['stage'], current['epoch'] = None, None
            def set_description(self, message):
                print(message, flush=True)
        predictions = []
        original_interface = learner.network.interface
        def interface(*args, **kwargs):
            logits = original_interface(*args, **kwargs)
            if logits.shape != (len(args[0]), 10) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError('SD source evaluation returned nonfinite or oracle logits')
            predictions.append(logits.argmax(1).detach().cpu().numpy())
            recorder.evaluation_examples += len(args[0])
            return logits
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys.modules['methods.base'], 'DataLoader', loader_adapter))
            stack.enter_context(patch.object(learner, 'build_optimizer', build_optimizer))
            stack.enter_context(patch.object(source_module, 'tqdm', EpochProgress))
            stack.enter_context(patch.object(source_module.F, 'cross_entropy', cross_entropy_observer(source_module.F.cross_entropy, record, current)))
            stack.enter_context(patch.object(learner.network, 'interface', interface))
            dm = DataManager()
            with recorder.phase('begin_task'):
                learner.before_task(dm)
            with recorder.phase('incremental_train'):
                learner.incremental_train(dm)
            current['stage'] = 'evaluation'
            shared.seed_phase(1993, 0, 'evaluation')
            with recorder.phase('evaluation'):
                accuracy = learner.incremental_test(dm)
            view = learner.test_loader.loader.dataset
            predicted = np.concatenate(predictions)
            destination = output / 'task1_predictions.npz'
            np.savez_compressed(destination, predictions=predicted, labels=view.targets, global_ids=view.global_ids)
            counts = shared.validate_prediction_archive(destination, 1, view.global_ids, view.targets)
            mean_accuracy = counts['seen_correct'] / 10.
            if abs(float(accuracy[0]['top1']) - mean_accuracy) > 1e-9:
                raise RuntimeError('SD actual source evaluator differs from captured predictions')
            record['task_results'] = [{'task': 1, **counts}]
            record['accuracy'] = {'mean_accuracy_pct': mean_accuracy}
            current['stage'] = 'task_end'
            shared.seed_phase(1993, 0, 'task_end')
            with recorder.phase('end_task'):
                learner.after_task()
        shared.audit_method_state(learner.network, learner, 'sd-lora', 1)
        state = learner.network.state_dict()
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise FloatingPointError('SD original source finalized state is nonfinite')
        canonical = shared.canonical_pretrained_digest(learner.network.image_encoder)
        if canonical != shared.CANONICAL_PRETRAINED_SHA256:
            raise RuntimeError('SD source canonical pretrained backbone changed')
        record['state_audits'] = {'network_sha256': shared.tensor_digest(state), 'frozen_backbone_sha256': canonical}
        checkpoint = output / 'task1_state.pt'
        with recorder.phase('checkpoint'):
            torch.save({'network': state, 'task': 1, 'protocol': spec['protocol']}, checkpoint)
            restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if shared.tensor_digest(restored['network']) != record['state_audits']['network_sha256']:
                raise RuntimeError('SD diagnostic state round trip changed tensors')
        record['state_audits']['checkpoint_sha256'] = shared.file_hash(checkpoint)
        checks.update({key: True for key in REQUIRED_CHECKS})
        recorder.tasks_completed = 1
        record['observed_totals'] = {key: getattr(recorder, key) for key in ('training_examples', 'evaluation_examples', 'steps', 'tasks_completed')}
        check_completion(record)
        comparisons = compare_records(record, native)
        with np.load(spec['native_predictions'], allow_pickle=False) as expected:
            comparisons['prediction_arrays_match'] = all(np.array_equal(value, expected[key])
                for key, value in {'predictions': predicted, 'labels': view.targets, 'global_ids': view.global_ids}.items())
        native_state = torch.load(spec['native_checkpoint'], map_location='cpu', weights_only=True)['model']
        native_network = {name[4:]: value for name, value in native_state.items() if name.startswith('net.')}
        comparisons['all_network_tensors_bitexact'] = set(native_network) == set(state) and all(torch.equal(value.cpu(), native_network[name]) for name, value in state.items())
        record['comparison'] = comparisons
        if not comparisons['input_sha256_matches'] or not comparisons['labels_sha256_matches']:
            raise RuntimeError('SD source diagnostic did not reproduce the native first batch')
        record['diagnosis'] = ('author_reproduces_native_feature_collapse' if comparisons['all_network_tensors_bitexact'] and comparisons['prediction_arrays_match'] and mean_accuracy == 10.
            else 'source_native_evidence_requires_review')
        write_json(output / 'comparison.json', {'diagnosis': record['diagnosis'], **comparisons})
        if notify_slack(f'SD author diagnostic task1 complete: {mean_accuracy:.2f}%; {record["diagnosis"]}; {record["wandb_url"]}') != 'sent':
            raise RuntimeError('SD required task notification failed')
        record['status'], code = 'completed', 0
    except Exception as error:
        record.update(error_type=type(error).__name__, error='SD author diagnostic failed; inspect local counts and structural evidence')
        logging.error('SD author diagnostic failed: %s', type(error).__name__)
    finally:
        for handle in handles:
            handle.remove()
        record['computation_status'] = record['status']
        record = recorder.save(record)
        write_json(output / 'result.json', record)
        if wandb.run is not None:
            try:
                wandb.run.summary.update({**flatten({'counts': record['counts'], 'accuracy': record['accuracy'],
                    'comparison': record.get('comparison', {}), 'structural_checks': record['structural_checks']}),
                    'diagnostic_status': record['status'], 'measurement_aggregate_eligible': False,
                    'diagnosis': record.get('diagnosis', 'incomplete')})
                artifact = wandb.Artifact(spec['name'], type='cl-diagnostic')
                files = [output / name for name in ('spec.json', 'result.json', 'comparison.json', 'task1_predictions.npz')
                         if (output / name).is_file()]
                manifest = {'files': {path.name: {'sha256': shared.file_hash(path), 'bytes': path.stat().st_size}
                    for path in files}, 'full_checkpoints_uploaded': False,
                    'local_checkpoint_sha256': record.get('state_audits', {}).get('checkpoint_sha256')}
                write_json(output / 'artifact_manifest.json', manifest)
                for path in [*files, output / 'artifact_manifest.json']:
                    artifact.add_file(str(path))
                wandb.run.log_artifact(artifact).wait()
                record['wandb_logging_status'] = 'uploaded'
            except Exception as error:
                record.update(status='logging_failed', wandb_logging_status='failed:' + type(error).__name__)
                code = 1
            try:
                wandb.finish(exit_code=code)
            except Exception as error:
                record.update(status='logging_failed', wandb_logging_status='finish_failed:' + type(error).__name__)
                code = 1
        else:
            record['wandb_logging_status'], code = 'run_not_initialized', 1
        write_json(output / 'result.json', record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-result', type=Path, default=DEFAULT_NATIVE)
    parser.add_argument('--reference-source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/peft_pilot/sd_author_common_ease_20261008')
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker is not None:
        return worker(options.worker)
    if options.preflight:
        spec = prepare_spec(options)
        print(json.dumps({'ready': True, 'method': 'sd-lora', 'tasks': 1, 'epochs': 1,
            'training_executed': False, 'output_created': False, 'checkpoint_sha256': spec['checkpoint_sha256'],
            'initial_state_sha256': spec['initial_state_sha256'], 'native_result_sha256': spec['native_result_sha256']}))
        return 0
    if not options.execute:
        print(json.dumps({'method': 'sd-lora', 'tasks': 1, 'epochs': 1, 'preprocessing': 'ease',
            'gpu': '1', 'execution': False, 'measurement_aggregate_eligible': False}))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())
