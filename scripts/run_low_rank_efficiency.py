"""Opt-in one-task LoRA efficiency worker with common EASE preprocessing.

Each invocation is one fresh measurement or separate partial-FLOPs pass. Dry
preparation imports only the standard library and performs no experiment work.
The accepted source-preprocessing implementation validation remains separate.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from copy import deepcopy
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
from scripts import low_rank_protocol as shared
from scripts.cl_efficiency import Recorder, flatten, instrument, model_snapshot, notify_slack, tensor_bytes, valid_slack_webhook, write_json
from scripts.cl_runtime import configure_environment
from scripts.hide_protocol import forbid_cifar_download, validate_shared_data

METHODS = tuple(shared.METHODS)
PRECISION = shared.PRECISION
CHECKPOINT_SHA256 = shared.CHECKPOINT_SHA256
CANONICAL_PRETRAINED_SHA256 = shared.CANONICAL_PRETRAINED_SHA256
REFERENCE_COMMIT = shared.REFERENCE_COMMIT
EXTRA_EXAMPLES = {'ewc-lora': (0, 5000), 'sd-lora': (0, 0), 'inf-lora': (5000, 5000), 'cl-lora': (0, 5000)}
REQUIRED_CHECKS = ('finite_loss_and_gradients', 'frozen_backbone_gradients_absent', 'frozen_backbone_unchanged',
    'canonical_pretrained_matches_pin', 'learnable_initialization_matches', 'dataset_matches_common_reference',
    'training_protocol_complete', 'evaluation_predictions_match_accuracy', 'prediction_ids_and_counts_complete',
    'method_state_valid', 'no_future_or_replay_samples', 'optimizer_updates_complete', 'checkpoint_round_trip')
RAW_FINGERPRINT_KEYS = ('n_samples', 'class_counts', 'samples_ordered_sha256', 'samples_unordered_sha256', 'global_ids_ordered_sha256')


def efficiency_protocol(method):
    settings = deepcopy(shared.protocol(method))
    settings.update(mode='efficiency', tasks=1, n_epochs=1, dataset_config='ease', efficiency_comparison=True,
        train_transforms=['RandomResizedCrop(224,scale=(0.05,1.0),bilinear)', 'RandomHorizontalFlip(0.5)',
                          'ToTensor', 'Normalize([0,0,0],[1,1,1])'],
        test_transforms=['Resize(256,bicubic)', 'CenterCrop(224)', 'ToTensor', 'Normalize([0,0,0],[1,1,1])'],
        normalization_source='Common EASE dataset configuration', measurement_repeats=3, flops_runs=1)
    settings.pop('pass_max_drop_pp', None)
    settings.pop('review_max_drop_pp', None)
    return settings


def native_command(spec, output=None):
    command = shared.native_command(spec['method'], spec['checkpoint_file'], spec['data_root'], output)
    for flag, value in (('--stop_after', '1'), ('--n_epochs', '1')):
        command[command.index(flag) + 1] = value
    command.extend(['--dataset_config', 'ease', '--disable_log', '0', '--save_after_interrupt', '0', '--joint', '0'])
    return command


def validate_spec(spec):
    method = spec.get('method')
    if (method not in METHODS or spec.get('model') != method or spec.get('dataset') != 'seq-cifar100-224'
            or spec.get('mode') != 'efficiency' or type(spec.get('tasks')) is not int or spec['tasks'] != 1
            or spec.get('protocol') != efficiency_protocol(method) or spec.get('pass') not in ('measurement', 'flops')
            or spec.get('gpu') != '1' or spec.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or spec.get('canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256
            or spec.get('reference_commit') != REFERENCE_COMMIT or spec.get('wandb_mode') != 'online'
            or spec.get('wandb_entity') != 'laymond1' or spec.get('wandb_project') != 'cl-efficiency'):
        raise ValueError('LoRA efficiency must remain the authorized one-task, one-epoch common EASE GPU1 protocol')
    for key in ('tasks', 'n_epochs', 'batch_size', 'num_workers', 'seed', 'rank', 'dataset_total_tasks', 'classes_per_task'):
        if type(spec['protocol'].get(key)) is not int:
            raise ValueError('LoRA efficiency protocol counts must be exact integers')


def validate_acceptance(record, method):
    if (record.get('method') != method or record.get('mode') != 'validation' or record.get('tasks') != 2
            or record.get('implementation') != 'mammoth_native' or record.get('protocol') != shared.protocol(method)
            or record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded' or record.get('acceptance_status') != 'pass'
            or record.get('acceptance', {}).get('status') != 'pass' or record.get('precision') != PRECISION
            or record.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or record.get('canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256
            or record.get('reference_commit') != REFERENCE_COMMIT
            or any(record.get('structural_checks', {}).get(name) is not True for name in shared.REQUIRED_STRUCTURAL_CHECKS)):
        raise ValueError('LoRA efficiency requires the completed and uploaded two-task native acceptance PASS')
    return record


def audit_fingerprints(actual, raw_reference, transform_reference):
    if (any(actual.get(key) != raw_reference.get(key) for key in RAW_FINGERPRINT_KEYS)
            or not actual.get('transform_repr_sha256')
            or actual['transform_repr_sha256'] != transform_reference.get('transform_repr_sha256')):
        raise RuntimeError('LoRA raw data, global IDs or actual preprocessing differ from the common EASE reference')
    return True


def completion_checks(record, recorder):
    method = record['method']
    begin, end = EXTRA_EXAMPLES[method]
    expected = {'observe_calls': 79, 'optimizer_updates': 79, 'task_begin_extra_samples': begin, 'task_end_extra_samples': end}
    if any(type(record.get('counts', {}).get(k)) is not int or record['counts'][k] != v for k, v in expected.items()):
        raise RuntimeError('LoRA efficiency observe/update counts or full task statistics passes are incomplete')
    expected_recorder = {'steps': 79, 'training_examples': 5000, 'evaluation_examples': 1000, 'tasks_completed': 1}
    if any(type(getattr(recorder, k, None)) is not int or getattr(recorder, k) != v for k, v in expected_recorder.items()):
        raise RuntimeError('LoRA efficiency requires exactly 79 steps, 5000 train and 1000 evaluation examples')
    rows = record.get('epoch_results', [])
    if len(rows) != 1 or any(type(rows[0].get(k)) is not int or rows[0][k] != v
                            for k, v in {'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000}.items()):
        raise RuntimeError('LoRA efficiency requires one complete first-task epoch')
    rows = record.get('task_results', [])
    if (len(rows) != 1 or any(type(rows[0].get(k)) is not int or rows[0][k] != v
            for k, v in {'task': 1, 'seen_samples': 1000, 'old_correct': 0, 'old_samples': 0, 'new_samples': 1000}.items())
            or type(rows[0].get('seen_correct')) is not int or not 0 <= rows[0]['seen_correct'] <= 1000
            or rows[0]['seen_correct'] != rows[0].get('new_correct')):
        raise RuntimeError('LoRA efficiency saved predictions do not cover the complete first task')
    if (len(record.get('prediction_files', [])) != 1 or record['prediction_files'][0].get('task') != 1
            or len(record.get('first_batch_audits', [])) != 1 or record['first_batch_audits'][0].get('task') != 1
            or record['first_batch_audits'][0].get('optimizer_update_index') != 1):
        raise RuntimeError('LoRA efficiency requires complete predictions and the first actual optimizer-step audit')
    pending = {'training_protocol_complete', 'optimizer_updates_complete'}
    if any(record.get('structural_checks', {}).get(name) is not True for name in REQUIRED_CHECKS if name not in pending):
        raise RuntimeError('LoRA efficiency structural audits are incomplete')
    record['structural_checks'].update(training_protocol_complete=True, optimizer_updates_complete=True)
    return True


def apply_common_transforms(dataset):
    """Install the agreed pipeline before LoRA builds its native transform."""
    from torchvision import transforms
    identity = transforms.Normalize([0, 0, 0], [1, 1, 1])
    dataset.TRANSFORM = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), identity])
    dataset.TEST_TRANSFORM = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), identity])
    dataset.MEAN, dataset.STD = [0, 0, 0], [1, 1, 1]
    dataset.get_normalization_transform = lambda: identity
    dataset.get_denormalization_transform = lambda: identity


def execution_sources():
    result = shared.source_hashes()
    for name in ('scripts/run_low_rank_efficiency.py', 'datasets/configs/seq-cifar100-224/ease.yaml',
                 'benchmarks/cl_efficiency/ease/validation_plan.json', 'scripts/run_peft_efficiency.py',
                 'benchmarks/cl_efficiency/peft_pilot/plan.json'):
        if (ROOT / name).is_file():
            result[name] = shared.file_hash(ROOT / name)
    return result


def build_spec(options, profile):
    if (profile['gpu'] != '1' or profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency'):
        raise ValueError('LoRA efficiency requires GPU1 and the approved W&B destination')
    accepted_path = (options.acceptance_summary or ROOT / f'results/low_rank/{options.method.split("-")[0]}_native_20261008/result.json').resolve()
    accepted = validate_acceptance(json.loads(accepted_path.read_text()), options.method)
    initial = Path(options.initial_state or accepted['initial_state']).resolve()
    checkpoint = Path(options.checkpoint).resolve()
    if shared.file_hash(initial) != accepted['initial_state_sha256'] or shared.file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('LoRA efficiency initialization or common pretrained checkpoint differs from acceptance')
    current = execution_sources()
    protected = ('models/low_rank_utils/', 'backbone/low_rank_vit.py')
    for name, digest in accepted['execution_source_sha256'].items():
        if name.startswith(protected) and current.get(name) != digest:
            raise ValueError('Accepted LoRA algorithm or backbone changed before efficiency preparation')
    from scripts.benchmark_cl import ease_execution_pin
    common = ease_execution_pin()
    plan_path = ROOT / 'benchmarks/cl_efficiency/ease/validation_plan.json'
    ease_plan = json.loads(plan_path.read_text())
    native_path = Path(ease_plan['validation']['result_summary'])
    if not native_path.is_absolute():
        native_path = plan_path.parent / native_path
    native_ease = json.loads(native_path.read_text())['native_validation']
    if common['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Common EASE weights differ from the accepted LoRA checkpoint')
    name = f'{options.method}-efficiency-{options.pass_kind}-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec = {'schema_version': 1, 'name': name, 'suite': name, 'model': options.method, 'method': options.method,
        'group': 'continual-low-rank', 'kind': 'training', 'mode': 'efficiency', 'tasks': 1, 'dataset': 'seq-cifar100-224',
        'pass': options.pass_kind, 'protocol': efficiency_protocol(options.method), 'gpu': '1', 'wandb_mode': 'online',
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'checkpoint_file': str(checkpoint),
        'checkpoint_sha256': CHECKPOINT_SHA256, 'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
        'reference_commit': REFERENCE_COMMIT, 'initial_state': str(initial), 'initial_state_sha256': shared.file_hash(initial),
        'initial_learned_sha256': accepted['initial_learned_sha256'], 'acceptance_summary': str(accepted_path),
        'acceptance_summary_sha256': shared.file_hash(accepted_path), 'accepted_validation_wandb_url': accepted['wandb_url'],
        'common_reference_data_snapshots': common['reference_data_snapshots'],
        'common_transform_snapshots': {key: native_ease['data_snapshots'][key] for key in ('task1_train', 'task1_test')},
        'common_reference_pins': {k: v for k, v in common.items() if k.endswith('_sha256')},
        'common_native_summary': str(native_path.resolve()), 'common_native_summary_sha256': shared.file_hash(native_path),
        'data_root': profile['data_root'], 'data_files': validate_shared_data(profile['data_root']),
        'execution_source_sha256': current, 'output': str(options.output.resolve()), 'paper_reproduction': False,
        'efficiency_comparison': True, 'measurement_aggregate_eligible': options.pass_kind == 'measurement',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    validate_spec(spec)
    return spec


def execute(options, profile):
    spec = build_spec(options, profile)
    if not valid_slack_webhook():
        raise ValueError('Slack credentials are required before LoRA efficiency execution')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []
    def notify(message):
        status = notify_slack(message)
        notifications.append({'message': message, 'status': status})
        write_json(output / 'notifications.json', notifications)
        return status
    if notify(f'{spec["method"]} efficiency {spec["pass"]} started: task1 epoch1 batch64 GPU1 common EASE preprocessing') != 'sent':
        raise RuntimeError('Required LoRA efficiency start notification failed before training')
    env = os.environ.copy()
    env.pop('MAMMOTH_DEVICE', None)
    env.update(CUDA_VISIBLE_DEVICES='1', WANDB_MODE='online', WANDB_DIR=str(output), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', CL_LOW_RANK_EFFICIENCY_EXECUTE='1')
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
                    notify(f'{spec["method"]} efficiency {spec["pass"]} still running; local phase observations preserved')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify(f'{spec["method"]} efficiency {spec["pass"]} interrupted; local observations preserved')
            raise
    record = json.loads((output / 'result.json').read_text()) if (output / 'result.json').is_file() else {'status': 'failed'}
    status = 'timeout' if timed_out else record['status']
    if process.returncode != 0 and status == 'completed':
        status = 'failed'
    sent = notify(f'{spec["method"]} efficiency {spec["pass"]} {status}; W&B {record.get("wandb_url", "unavailable")}')
    completion = {'status': status, 'exit_code': process.returncode, 'timeout': timed_out,
        'wall_s': time.monotonic() - started, 'wandb_url': record.get('wandb_url'), 'slack_completion_status': sent}
    write_json(output / 'completion.json', completion)
    print(json.dumps({'status': status, 'result': str(output / 'result.json'), 'wandb_url': record.get('wandb_url')}))
    return 0 if process.returncode == 0 and status == 'completed' and sent == 'sent' else 1


class EfficiencyAudit:
    """Read-only observations chained around native trainer/meta hooks."""
    def __init__(self, model, dataset, recorder, record, torch, np):
        self.model, self.dataset, self.recorder, self.record, self.torch, self.np = model, dataset, recorder, record, torch, np
        from scripts.ease_size_protocol import canonical_pretrained_state
        self.frozen_names = {'image_encoder.' + name for name in canonical_pretrained_state(model.net.image_encoder)}
        self.stage, self.epoch, self.predictions = None, {}, []
        self.configured, self.hooked, self.handles = WeakKeyDictionary(), WeakSet(), []

    def snapshot(self, name):
        value = model_snapshot(self.model)
        plain = {f'{index}.{attribute}': getattr(module, attribute) for index, module in enumerate(self.model.net.modules())
                 for attribute in ('matrix', 'cur_matrix') if self.torch.is_tensor(getattr(module, attribute, None))
                 and attribute not in module._buffers}
        value['unregistered_attention_state_storage_bytes'] = tensor_bytes(plain)
        self.recorder.snapshots[name] = value

    def finite(self):
        for name, parameter in self.model.net.named_parameters():
            if name in self.frozen_names and parameter.grad is not None:
                raise RuntimeError('Frozen LoRA pretrained parameter received a gradient')
            if parameter.requires_grad and (not bool(self.torch.isfinite(parameter).all())
                    or parameter.grad is not None and not bool(self.torch.isfinite(parameter.grad).all())):
                raise FloatingPointError('LoRA learned parameter or gradient is nonfinite')

    def attach_optimizer(self, optimizer):
        if optimizer is None or optimizer in self.hooked:
            return optimizer
        self.hooked.add(optimizer)
        def before_step(subject, arguments, keywords):
            self.finite()
            if self.stage == 'train' and self.epoch['steps'] == 0:
                grads = {name: p.grad for name, p in self.model.net.named_parameters() if p.requires_grad and p.grad is not None}
                self.record['first_batch_audits'].append({'task': 1, 'epoch': 1, 'optimizer_update_index': 1,
                    'seed': self.epoch['seed'], 'samples': len(self.labels),
                    'input_sha256': shared.tensor_digest({'inputs': self.inputs}),
                    'labels_sha256': shared.tensor_digest({'labels': self.labels}), 'gradient_sha256': shared.tensor_digest(grads)})
        def after_step(subject, arguments, keywords):
            if self.stage == 'train':
                self.record['counts']['optimizer_updates'] += 1
        self.handles.extend((optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)))
        return optimizer

    def phase(self, original, name):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            self.stage = name
            shared.seed_phase(1993, 0, name)
            shared.configure_epoch_loader(self.dataset.train_loader, 1993, 0, name)
            self.configured[self.dataset.train_loader] = (1993, 0, name)
            result = original(*args, **kwargs)
            self.attach_optimizer(self.model.opt)
            self.snapshot('after_' + name)
            self.stage = None
            return result
        return wrapped

    def get_optimizer(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return self.attach_optimizer(original(*args, **kwargs))
        return wrapped

    def loaders(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            from scripts.ease_reference_utils import dataset_fingerprint
            if len(self.dataset.test_loaders) != 1:
                raise RuntimeError('LoRA efficiency loaded an unauthorized later task')
            for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[0])):
                raw = loader.dataset
                ids, labels = self.np.asarray(raw.indexes), self.np.asarray(raw.targets)
                if not self.np.all(ids[1:] > ids[:-1]) or not self.np.all((labels >= 0) & (labels < 10)):
                    raise RuntimeError('LoRA task1 IDs or labels include future/reordered samples')
                actual = dataset_fingerprint(raw.data, labels, ids, raw.transform)
                audit_fingerprints(actual, self.record['common_reference_data_snapshots'][f'task1_{split}'],
                                   self.record['common_transform_snapshots'][f'task1_{split}'])
                self.recorder.data_snapshots[f'task1_{split}'] = actual
            self.record['structural_checks']['dataset_matches_common_reference'] = True
            return result
        return wrapped

    def begin_epoch(self, original):
        @functools.wraps(original)
        def wrapped(index, dataset):
            if index != 0:
                raise RuntimeError('LoRA efficiency attempted an unauthorized later epoch')
            seed = shared.seed_phase(1993, 0, 'train', index)
            shared.configure_epoch_loader(self.dataset.train_loader, 1993, 0, 'train', index)
            self.epoch = {'task': 1, 'epoch': 1, 'seed': seed, 'steps': 0, 'samples': 0, 'loss_sum': 0.}
            self.stage = 'train'
            return original(index, dataset)
        return wrapped

    def observed(self, original):
        @functools.wraps(original)
        def wrapped(inputs, labels, *args, **kwargs):
            self.inputs, self.labels = inputs, labels
            if bool(((labels < 0) | (labels >= 10)).any()):
                raise RuntimeError('LoRA observe received future or replay labels')
            result = original(inputs, labels, *args, **kwargs)
            values = result.values() if isinstance(result, dict) else [result]
            if any(not math.isfinite(float(value)) for value in values):
                raise FloatingPointError('LoRA named training loss is nonfinite')
            loss = float(result['loss'] if isinstance(result, dict) else result)
            self.epoch['steps'] += 1
            self.epoch['samples'] += len(labels)
            self.epoch['loss_sum'] += loss
            self.record['counts']['observe_calls'] += 1
            return result
        return wrapped

    def end_epoch(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            self.finite()
            self.record['epoch_results'].append({**self.epoch, 'loss_mean': self.epoch['loss_sum'] / max(1, self.epoch['steps'])})
            write_json(self.recorder.output / 'epoch_results.json', self.record['epoch_results'])
            write_json(self.recorder.output / 'progress.json', self.record['epoch_results'][-1])
            self.stage = None
            return result
        return wrapped

    def counted_iter(self, original):
        def wrapped(loader):
            if self.stage not in ('task_begin', 'task_end'):
                return original(loader)
            shared.configure_phase_loader_once(loader, 1993, 0, self.stage, self.configured)
            def iterator():
                for batch in original(loader):
                    labels = batch[1]
                    if bool(((labels < 0) | (labels >= 10)).any()):
                        raise RuntimeError('LoRA task statistics received future or replay labels')
                    self.record['counts'][self.stage + '_extra_samples'] += len(batch[0])
                    yield batch
            return iterator()
        return wrapped

    def forwarded(self, original):
        @functools.wraps(original)
        def wrapped(inputs, *args, **kwargs):
            logits = original(inputs, *args, **kwargs)
            if self.stage == 'evaluation':
                if logits.shape != (len(inputs), 10) or not bool(self.torch.isfinite(logits).all()):
                    raise RuntimeError('LoRA evaluation must return finite nonoracle ten-class logits')
                self.predictions.append(logits.argmax(1).detach().cpu().numpy())
            return logits
        return wrapped

    def evaluated(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            from torch.utils.data import SequentialSampler
            loader = self.dataset.test_loaders[0]
            if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
                raise RuntimeError('LoRA evaluation requires complete sequential test IDs')
            self.stage = 'evaluation'
            shared.seed_phase(1993, 0, 'evaluation')
            shared.configure_epoch_loader(loader, 1993, 0, 'evaluation')
            self.predictions = []
            result = original(*args, **kwargs)
            with self.recorder.phase('state_audit'):
                raw = loader.dataset
                predicted = self.np.concatenate(self.predictions) if self.predictions else self.np.empty(0, dtype=self.np.int64)
                labels, ids = self.np.asarray(raw.targets), self.np.asarray(raw.indexes)
                if len(predicted) != 1000 or len(labels) != 1000:
                    raise RuntimeError('LoRA native evaluation did not expose exactly 1000 predictions')
                counts = shared.counts_from_predictions(predicted, labels, past_classes=0, seen_classes=10)
                accuracy = counts['seen_correct'] / 10.
                if abs(float(result[0][0]) - accuracy) > 1e-9:
                    raise RuntimeError('LoRA saved predictions differ from native class-IL accuracy')
                order = self.np.argsort(ids, kind='stable')
                path = self.recorder.output / 'task1_predictions.npz'
                self.np.savez_compressed(path, predictions=predicted[order], labels=labels[order], global_ids=ids[order])
                shared.validate_prediction_archive(path, 1, ids[order], labels[order])
                self.record['prediction_files'] = [{'task': 1, 'path': str(path), 'sha256': shared.file_hash(path)}]
                self.record['task_results'] = [{'task': 1, **counts}]
                self.record['accuracy'] = {'setting': 'class-il', 'mean_accuracy_pct': accuracy,
                    'task_il_mean_accuracy_pct': float(result[1][0]), 'fullaccs': list(map(float, result[0])),
                    'fullaccs_mask_classes': list(map(float, result[1]))}
                self.record['structural_checks'].update(evaluation_predictions_match_accuracy=True, prediction_ids_and_counts_complete=True)
            self.stage = None
            return result
        return wrapped

    def finalized(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.recorder.phase('finalize_after_eval'):
                if self.record['method'] == 'ewc-lora':
                    shared.seed_phase(1993, 0, 'task_end')
                result = original(*args, **kwargs)
            with self.recorder.phase('state_audit'):
                current = shared.canonical_pretrained_digest(self.model.net.image_encoder)
                if current != CANONICAL_PRETRAINED_SHA256:
                    raise RuntimeError('Frozen LoRA canonical pretrained weights changed')
                shared.audit_method_state(self.model.net, self.model.algorithm, self.record['method'], 1)
                states = [self.model.net.state_dict(), self.model.algorithm.state_dict()]
                if any(not bool(self.torch.isfinite(v).all()) for state in states for v in state.values()):
                    raise FloatingPointError('LoRA finalized algorithm or network state is nonfinite')
                if self.model.net._cur_task != 0 or int(self.model._low_rank_state[3]) != 10:
                    raise RuntimeError('LoRA finalized native task counters are invalid')
                self.record['state_audits'] = {'task': 1, 'frozen_backbone_sha256': current,
                    'network_sha256': shared.tensor_digest(states[0]), 'algorithm_sha256': shared.tensor_digest(states[1]),
                    'algorithm_state_shapes': {name: list(value.shape) for name, value in states[1].items()}, 'seen_classes': 10}
                self.snapshot('after_finalization')
            with self.recorder.phase('checkpoint'):
                path = self.recorder.output / 'task1_state.pt'
                self.torch.save({'model': self.model.state_dict(), 'task': 1, 'protocol': self.record['protocol']}, path)
                saved = self.torch.load(path, map_location='cpu', weights_only=True)
                if shared.tensor_digest(saved['model']) != shared.tensor_digest(self.model.state_dict()):
                    raise RuntimeError('LoRA finalized checkpoint CPU round trip changed state')
                self.record['state_audits']['checkpoint_sha256'] = shared.file_hash(path)
            self.record['structural_checks'].update(frozen_backbone_unchanged=True, method_state_valid=True, checkpoint_round_trip=True)
            status = notify_slack(f'{self.record["method"]} efficiency {self.record["pass"]} task1 completed: seen={self.record["accuracy"]["mean_accuracy_pct"]:.2f}%; W&B {self.record["wandb_url"]}')
            self.record['task_notification_status'] = status
            if status != 'sent':
                raise RuntimeError('Required LoRA task-completion notification failed')
            return result
        return wrapped

    def backward(self, original):
        @functools.wraps(original)
        def wrapped(tensor, *args, **kwargs):
            if not bool(self.torch.isfinite(tensor).all()):
                raise FloatingPointError('LoRA backward loss is nonfinite')
            result = original(tensor, *args, **kwargs)
            self.finite()
            return result
        return wrapped

    def install(self, stack, training):
        stack.enter_context(instrument(self.recorder, self.model, self.dataset, training))
        wrappers = ((self.model, 'get_optimizer', self.get_optimizer), (self.dataset, 'get_data_loaders', self.loaders),
            (self.model, 'meta_begin_epoch', self.begin_epoch), (self.model, 'meta_observe', self.observed),
            (self.model, 'meta_end_epoch', self.end_epoch), (self.model, 'forward', self.forwarded),
            (self.dataset, 'evaluate', self.evaluated), (self.model, 'meta_end_eval', self.finalized))
        for target, name, wrapper in wrappers:
            stack.enter_context(patch.object(target, name, wrapper(getattr(target, name))))
        for name, phase in (('meta_begin_task', 'task_begin'), ('meta_end_task', 'task_end')):
            stack.enter_context(patch.object(self.model, name, self.phase(getattr(self.model, name), phase)))
        stack.enter_context(patch.object(self.torch.utils.data.DataLoader, '__iter__', self.counted_iter(self.torch.utils.data.DataLoader.__iter__)))
        stack.enter_context(patch.object(self.torch.Tensor, 'backward', self.backward(self.torch.Tensor.backward)))


def worker(spec_path):
    if os.getenv('CL_LOW_RANK_EFFICIENCY_EXECUTE') != '1':
        raise ValueError('LoRA efficiency worker requires explicit parent --execute authorization')
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    if os.getenv('CUDA_VISIBLE_DEVICES') != '1' or os.getenv('WANDB_MODE') != 'online':
        raise ValueError('LoRA efficiency worker requires GPU1 and online W&B')
    if execution_sources() != spec['execution_source_sha256']:
        raise ValueError('LoRA efficiency protected sources changed after preparation')
    for path, digest in ((spec['checkpoint_file'], CHECKPOINT_SHA256), (spec['initial_state'], spec['initial_state_sha256']),
                         (spec['acceptance_summary'], spec['acceptance_summary_sha256']),
                         (spec['common_native_summary'], spec['common_native_summary_sha256'])):
        if shared.file_hash(path) != digest:
            raise ValueError('LoRA efficiency pinned input evidence changed after preparation')
    accepted = validate_acceptance(json.loads(Path(spec['acceptance_summary']).read_text()), spec['method'])
    if validate_shared_data(spec['data_root']) != spec['data_files']:
        raise ValueError('Shared raw CIFAR100 data changed after LoRA efficiency preparation')
    output = Path(spec['output'])
    recorder = Recorder(output, '1')
    recorder.start()
    record = {**spec, 'status': 'failed', 'accuracy': {}, 'counts': {'observe_calls': 0, 'optimizer_updates': 0,
        'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}, 'epoch_results': [], 'task_results': [],
        'first_batch_audits': [], 'prediction_files': [], 'structural_checks': {name: False for name in REQUIRED_CHECKS}}
    code, wandb, audit = 1, None, None
    try:
        import numpy as np
        import torch
        import wandb
        import main as mammoth
        from utils import training
        from utils.conf import base_path
        from models.low_rank_utils import continual
        from scripts.benchmark_cl import _configure_ease_precision
        from scripts.ease_reference_utils import shared_cifar_metadata
        torch.set_num_threads(2)
        record['precision'] = {**_configure_ease_precision(torch), 'torch_cpu_threads': torch.get_num_threads()}
        record['precision']['autocast'] = torch.is_autocast_enabled()
        if record['precision'] != PRECISION:
            raise ValueError('LoRA efficiency precision differs from the common fp32 protocol')
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        base_path(str(Path(spec['data_root']).resolve()))
        shared.seed_phase(1993, 0, 'task_begin')
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'wandb')}
        wandb.init(entity='laymond1', project='cl-efficiency', name=spec['name'], mode='online', group='peft-common-efficiency',
                   job_type=spec['pass'], config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        command = native_command(spec, output)
        sys.argv = ['main.py', *command]
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) if spec['pass'] == 'flops' else None
        with profiler if profiler is not None else nullcontext():
            with forbid_cifar_download():
                args = mammoth.parse_args(cmd=command)
                with recorder.phase('initialize'), patch.object(continual, 'source_transforms', apply_common_transforms):
                    model, dataset, args = mammoth.initialize(args)
            args.nowand = 1
            expected = {'stop_after': 1, 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                'lora_rank': 10, 'lora_vit_size': 'base', 'dataset_config': 'ease', 'permute_classes': 0,
                'noise_rate': 0, 'validation': 0, 'label_perc': 1, 'label_perc_by_class': 1, 'debug_mode': 0,
                'enable_other_metrics': 0, 'eval_future': 0, 'eval_epochs': None, 'fitting_mode': 'epochs',
                'lr_scheduler': 'cosine', 'scheduler_mode': 'epoch', 'joint': 0, 'code_optimization': 0}
            hyper = spec['protocol']['hyperparameters']
            expected.update(lr=hyper['lr'], lora_fc_lr=hyper['fc_lr'], optimizer=hyper['optimizer'], optim_wd=hyper['weight_decay'])
            expected.update({f'lora_{k}': v for k, v in hyper.items() if k not in ('optimizer', 'lr', 'fc_lr', 'weight_decay', 'momentum')})
            if spec['method'] == 'cl-lora':
                expected.update(optim_mom=.9, optim_nesterov=0)
            if any(getattr(args, k, None) != v for k, v in expected.items()) or (dataset.N_TASKS, dataset.N_CLASSES_PER_TASK, dataset.N_CLASSES) != (10, 10, 100):
                raise ValueError('Effective LoRA efficiency arguments differ from the frozen workload')
            write_json(output / 'effective_args.json', vars(args))
            write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
            with recorder.phase('state_audit'):
                initial = shared.load_initial_state(model.net, torch.load(spec['initial_state'], map_location='cpu', weights_only=True), spec['method'])
                canonical = shared.canonical_pretrained_digest(model.net.image_encoder)
                if initial != accepted['initial_learned_sha256'] or canonical != CANONICAL_PRETRAINED_SHA256:
                    raise ValueError('LoRA efficiency initialization or pretrained state differs from acceptance')
            record.update(initial_learned_sha256=initial, initial_canonical_pretrained_sha256=canonical)
            checks = record['structural_checks']
            checks.update(canonical_pretrained_matches_pin=True, learnable_initialization_matches=True,
                          no_future_or_replay_samples=True, finite_loss_and_gradients=True, frozen_backbone_gradients_absent=True)
            audit = EfficiencyAudit(model, dataset, recorder, record, torch, np)
            audit.snapshot('after_initialize')
            with ExitStack() as stack:
                stack.enter_context(forbid_cifar_download())
                audit.install(stack, training)
                training.train(model, dataset, args)
            completion_checks(record, recorder)
            checks.update(training_protocol_complete=True, optimizer_updates_complete=True)
        if profiler is not None:
            operations = [{'operation': item.key, 'count': item.count, 'flops': item.flops}
                          for item in profiler.key_averages() if item.flops]
            total = sum(item['flops'] for item in operations)
            if not math.isfinite(total) or total <= 0:
                raise RuntimeError('Separate LoRA profiler pass did not produce a positive partial operator estimate')
            record['flops'] = {'supported_ops_total': total, 'partial': True, 'includes_backward_only_if_supported': True,
                'scope': 'Supported PyTorch operators across initialization, full begin/end hooks, training, evaluation and diagnostic audits in this separate pass',
                'limitations': 'NumPy DualGPM SVD, unsupported/fused operators, Torch linear algebra, optimizer, CPU/PIL preprocessing and serialization may be omitted; no backward x3 extrapolation or phase attribution',
                'measurement_aggregate_eligible': False}
            write_json(output / 'flops_ops.json', operations)
        record.update(status='completed', task1_scope='First-task cost only; later CL distillation/orthogonality and cumulative method state are outside this pilot')
        code = 0
    except Exception as error:
        record.update(error_type=type(error).__name__, error='LoRA efficiency failed; inspect local phase/count/structural audits')
        logging.error('LoRA efficiency worker failed: %s', type(error).__name__)
    finally:
        if audit is not None:
            for handle in audit.handles:
                handle.remove()
        record['computation_status'] = record['status']
        record = recorder.save(record)
        for name, content in (('result.json', record), ('state_audits.json', record.get('state_audits', {})),
                              ('first_batch_audits.json', record['first_batch_audits']), ('task_results.json', record['task_results'])):
            write_json(output / name, content)
        if wandb is not None and wandb.run is not None:
            try:
                logged = {k: v for k, v in flatten({'efficiency': record['metrics'], 'phases': record['phases'],
                    'snapshots': record['model_snapshots'], 'counts': record['counts'], 'flops': record.get('flops', {}),
                    'accuracy': record['accuracy'], 'structural_checks': record['structural_checks']}).items() if isinstance(v, (int, float, bool))}
                wandb.run.summary.update({**logged, 'benchmark_status': record['status'], 'benchmark_pass': spec['pass']})
                artifact = wandb.Artifact(spec['name'], type='cl-efficiency')
                for name in ('spec.json', 'result.json', 'metrics.json', 'effective_args.json', 'shared_data.json',
                    'epoch_results.json', 'task_results.json', 'state_audits.json', 'first_batch_audits.json',
                    'phases.json', 'telemetry.csv', 'task1_predictions.npz', 'flops_ops.json'):
                    if (output / name).is_file():
                        artifact.add_file(str(output / name))
                wandb.run.log_artifact(artifact).wait()
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
            record['wandb_logging_status'] = 'run_not_initialized'
            code = 1
        write_json(output / 'result.json', record)
        write_json(output / 'metrics.json', record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='ewc-lora')
    parser.add_argument('--pass', dest='pass_kind', choices=('measurement', 'flops'), default='measurement')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/peft_efficiency/prepared')
    parser.add_argument('--acceptance-summary', type=Path)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(shared.CHECKPOINT))
    parser.add_argument('--timeout', type=float, default=3600.)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker is not None:
        return worker(options.worker)
    if not options.execute:
        print(json.dumps({'method': options.method, 'pass': options.pass_kind, 'tasks': 1, 'epochs': 1,
            'batch_size': 64, 'workers': 4, 'seed': 1993, 'gpu': '1', 'preprocessing': 'ease', 'execution': False}))
        return 0
    return execute(options, configure_environment())


if __name__ == '__main__':
    sys.exit(main())
