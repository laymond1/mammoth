"""Opt-in pinned RainbowPrompt author/native validation and isolated cost pilots.

Importing and preparation start no training. Native execution always delegates to
Mammoth's real trainer and meta hooks; the independent runner compiles the
reviewed author classes and numerical epoch engine without importing the port.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import datetime
import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch
from weakref import WeakSet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import rainbow_protocol as shared
from scripts import rainbow_reference_utils as reference
from scripts.cl_efficiency import Recorder, flatten, instrument, model_snapshot, notify_slack, tensor_bytes, valid_slack_webhook, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.hide_protocol import forbid_cifar_download, validate_shared_data

REQUIRED_CHECKS = shared.REQUIRED_CHECKS
PRECISION = shared.PRECISION


def build_spec(options, profile):
    shared.load_run_plan()
    if any(profile[key] != value for key, value in {'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'}.items()):
        raise ValueError('RainbowPrompt requires the approved GPU1 and W&B destination')
    from scripts.benchmark_cl import ease_execution_pin
    common = ease_execution_pin()
    ease_plan_path = ROOT / 'benchmarks/cl_efficiency/ease/validation_plan.json'
    ease_plan = json.loads(ease_plan_path.read_text())
    native_path = Path(ease_plan['validation']['result_summary'])
    if not native_path.is_absolute():
        native_path = ease_plan_path.parent / native_path
    native = json.loads(native_path.read_text())['native_validation']
    initial = Path(options.initial_state).resolve()
    checkpoint = Path(getattr(options, 'checkpoint', shared.CHECKPOINT)).resolve()
    source = Path(getattr(options, 'source', shared.REFERENCE_SOURCE)).resolve()
    if shared.file_hash(checkpoint) != shared.CHECKPOINT_SHA256:
        raise ValueError('RainbowPrompt requires the common pinned local NPZ')
    pins = shared.verify_reference_source(source)
    reference.verify_author_source(source, pins)
    name = f'rainbow-{options.side}-{options.mode}-{options.pass_kind}-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec = {'schema_version': 1, 'name': name, 'suite': name, 'method': 'rainbow-prompt', 'model': 'rainbow-prompt',
        'side': options.side, 'mode': options.mode, 'pass': options.pass_kind,
        'implementation': 'mammoth_native' if options.side == 'native' else 'pinned_author_reference',
        'protocol': shared.protocol(options.mode), 'source_mode': 'official-active-path', 'learned_gate_active': False,
        'paper_equivalent': False, 'paper_reproduction': False, 'training_authorized': True,
        'measurement_aggregate_eligible': options.pass_kind == 'measurement', 'gpu': '1', 'wandb_mode': 'online',
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'checkpoint_file': str(checkpoint),
        'checkpoint_sha256': shared.CHECKPOINT_SHA256, 'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
        'reference_commit': shared.REFERENCE_COMMIT, 'reference_source': str(source), 'reference_source_sha256': pins,
        'initial_state': str(initial), 'initial_state_sha256': shared.file_hash(initial),
        'common_reference_data_snapshots': common['reference_data_snapshots'],
        'common_transform_snapshots': {key: native['data_snapshots'][key] for key in ('task1_train', 'task1_test')},
        'common_native_summary': str(native_path.resolve()), 'common_native_summary_sha256': shared.file_hash(native_path),
        'data_root': profile['data_root'], 'data_files': validate_shared_data(profile['data_root']),
        'execution_source_sha256': shared.execution_sources(), 'output': str(options.output.resolve()),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    shared.validate_spec(spec)
    return spec


def validate_saved_record(record):
    """Validate completed protocol evidence without constructing a model or using a GPU."""
    shared.validate_spec(record)
    p = record['protocol']
    tasks, epochs = p['tasks'], p['n_epochs']
    updates = tasks * epochs * 79
    if (record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('precision') != PRECISION or any(record.get('structural_checks', {}).get(key) is not True for key in REQUIRED_CHECKS)
            or record.get('counts') != {'observe_calls': updates, 'optimizer_updates': updates,
                                      'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}
            or len(record.get('epoch_results', [])) != tasks * epochs
            or len(record.get('first_batch_audits', [])) != tasks * epochs
            or len(record.get('task_results', [])) != tasks
            or len(record.get('prediction_files', [])) != tasks):
        raise ValueError('RainbowPrompt completed record has incomplete protocol or structural evidence')
    for row, task_epoch in zip(record['epoch_results'], ((task, epoch) for task in range(1, tasks + 1) for epoch in range(1, epochs + 1))):
        task, epoch = task_epoch
        if (any(type(row.get(key)) is not int or row[key] != value for key, value in
                {'task': task, 'epoch': epoch, 'steps': 79, 'samples': 5000,
                 'seed': shared.phase_seed(1993, task - 1, 'train', epoch - 1)}.items())
                or not math.isfinite(row.get('loss_mean', float('nan')))):
            raise ValueError('RainbowPrompt epoch evidence is incomplete or has the wrong phase RNG')
    for task, row in enumerate(record['task_results'], 1):
        for prefix, samples in (('seen', task * 1000), ('old', (task - 1) * 1000), ('new', 1000)):
            correct = row.get(prefix + '_correct')
            if (type(row.get(prefix + '_samples')) is not int or row[prefix + '_samples'] != samples
                    or type(correct) is not int or not 0 <= correct <= samples
                    or samples and row.get(prefix + '_accuracy_pct') != 100 * correct / samples):
                raise ValueError('RainbowPrompt exact saved prediction counts disagree with accuracy')
        if row.get('task') != task or row['seen_correct'] != row['old_correct'] + row['new_correct']:
            raise ValueError('RainbowPrompt task counts disagree')
    if record.get('accuracy', {}).get('mean_accuracy_pct') != record['task_results'][-1]['seen_accuracy_pct']:
        raise ValueError('RainbowPrompt final accuracy differs from saved counts')
    metrics = record.get('metrics', {})
    for key, value in {'outer_observe_steps': updates, 'outer_training_stream_examples': tasks * epochs * 5000,
                       'evaluation_examples': 1000 if tasks == 1 else 3000, 'tasks_completed': tasks}.items():
        if type(metrics.get(key)) is not int or metrics[key] != value:
            raise ValueError('RainbowPrompt resource metrics do not cover the complete workload')
    return True


def configure_precision(torch):
    from scripts.benchmark_cl import _configure_ease_precision
    torch.set_num_threads(2)
    value = {**_configure_ease_precision(torch), 'torch_cpu_threads': torch.get_num_threads(),
             'autocast': torch.is_autocast_enabled()}
    if value != PRECISION:
        raise ValueError('RainbowPrompt requires common FP32 precision')
    return value


def prepare_initial_state(path, source=shared.REFERENCE_SOURCE, checkpoint=shared.CHECKPOINT):
    """Explicit CPU-only preparation from seeded independent author initialization."""
    configure_native_environment()
    import torch
    if shared.file_hash(checkpoint) != shared.CHECKPOINT_SHA256:
        raise ValueError('RainbowPrompt initialization requires the pinned common NPZ')
    code = reference.load_author_code(source, shared.verify_reference_source(source))
    configure_precision(torch)
    torch.manual_seed(1993)
    branches = reference.build_author_branches(code, checkpoint)
    canonical = canonical_hashes(branches)
    if any(value != shared.CANONICAL_PRETRAINED_SHA256 for value in canonical.values()):
        raise ValueError('Independent RainbowPrompt initialization did not load all canonical pretrained tensors')
    metadata = reference.save_initial_state(branches, path, shared.CHECKPOINT_SHA256)
    return {**metadata, 'path': str(Path(path).resolve()), 'sha256': shared.file_hash(path), 'canonical_pretrained': canonical}


def canonical_hashes(branches):
    from scripts.ease_size_protocol import canonical_pretrained_digest
    return {name: canonical_pretrained_digest(module) for name, module in branches.items()}


def parameter_audit(branches):
    parameters = [parameter for module in branches.values() for parameter in module.parameters()]
    actual = {'trainable_parameters': sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
              'frozen_parameters': sum(parameter.numel() for parameter in parameters if not parameter.requires_grad),
              'stored_prompt_cache_bytes': tensor_bytes(branches['prompt'].rainbow_prompt.stored_rainbow_prompts)}
    if actual != {'trainable_parameters': 6521476, 'frozen_parameters': 171674212, 'stored_prompt_cache_bytes': 7372800}:
        raise ValueError('RainbowPrompt Base parameter counts or cache allocation differ from the pinned active architecture')
    return actual


def error_context(error, recorder, audit):
    """Keep code locations, never exception messages, environment or locals."""
    return {'error_type': type(error).__name__, 'phase': recorder.active_phase,
            'task': audit.task + 1 if audit is not None and audit.task >= 0 else None,
            'epoch': audit.epoch.get('epoch') if audit is not None else None,
            'frames': [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                       for frame in traceback.extract_tb(error.__traceback__)]}


def exact_counts(predicted, labels, task):
    import numpy as np
    predicted, labels = np.asarray(predicted), np.asarray(labels)
    if predicted.shape != labels.shape or labels.ndim != 1:
        raise ValueError('RainbowPrompt prediction shape mismatch')
    old, correct = labels < (task - 1) * 10, predicted == labels
    result = {'task': task}
    for prefix, mask in (('seen', np.ones(len(labels), dtype=bool)), ('old', old), ('new', ~old)):
        samples, count = int(mask.sum()), int(correct[mask].sum())
        result.update({prefix + '_samples': samples, prefix + '_correct': count,
                       prefix + '_accuracy_pct': 100 * count / samples if samples else None})
    return result


class RainbowAudit:
    """Read-only source/native observations around the actual numerical lifecycle."""
    def __init__(self, branches, dataset, recorder, record, torch, np, native=None):
        self.branches, self.dataset, self.recorder, self.record = branches, dataset, recorder, record
        self.torch, self.np, self.native = torch, np, native
        self.prompt = branches['prompt'].rainbow_prompt
        self.stage, self.task, self.epoch = None, -1, {}
        self.handles, self.optimizers, self.optimizer_ids = [], WeakSet(), []
        self.eval_predictions, self.eval_routes, self.eval_batches = [], [], []
        self.routing_layer_count, self.active_prefix_layers = 0, set()
        self.handles.append(self.prompt.register_forward_hook(self.prompt_output, with_kwargs=True))
        self.task_old = {}

    def frozen_finite(self):
        checks = []
        for branch, module in self.branches.items():
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise RuntimeError('Frozen RainbowPrompt pretrained or query tensor received a gradient')
                elif parameter.grad is not None:
                    checks.append(self.torch.isfinite(parameter.grad).all())
        if checks and not bool(self.torch.stack(checks).all()):
            raise FloatingPointError('RainbowPrompt learned gradient is nonfinite')

    def old_state(self):
        count = self.task
        result = {'cache': self.prompt.stored_rainbow_prompts[:count].detach().cpu().clone(),
                  'base_key': self.prompt.base_key[:count].detach().cpu().clone()}
        for layer in self.record['protocol']['prompt_layers']:
            result[f'base_knowledge_{layer}'] = getattr(self.prompt, f'base_knowledge_{layer}')[:count].detach().cpu().clone()
        for name, value in self.branches['prompt'].head.state_dict().items():
            result['head.' + name] = value[:count * 10].detach().cpu().clone()
        return result

    def assert_old_unchanged(self):
        current = self.old_state()
        if set(current) != set(self.task_old) or any(not self.torch.equal(current[key], self.task_old[key]) for key in current):
            raise RuntimeError('Previous-task RainbowPrompt cache, base pool, key or classifier rows changed')

    def snapshot(self, name):
        if self.native is not None:
            model = self.native
        else:
            model = self.torch.nn.Module()
            for key, value in self.branches.items():
                model.add_module(key, value)
            model.opt = self.optimizer
        value = model_snapshot(model)
        retained = {str(i): block.attn.key_ for i, block in enumerate(self.branches['prompt'].blocks) if hasattr(block.attn, 'key_')}
        value['unregistered_attention_state_storage_bytes'] = tensor_bytes(retained)
        self.recorder.snapshots[name] = value

    def load_task(self):
        from scripts.ease_reference_utils import dataset_fingerprint
        self.task = len(self.dataset.test_loaders) - 1
        if self.task not in range(self.record['protocol']['tasks']):
            raise RuntimeError('RainbowPrompt loaded an unauthorized task')
        for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[-1])):
            raw = loader.dataset
            ids, labels = self.np.asarray(raw.indexes), self.np.asarray(raw.targets)
            low, high = self.task * 10, (self.task + 1) * 10
            expected = 5000 if split == 'train' else 1000
            if (len(labels) != expected or len(self.np.unique(ids)) != expected or not self.np.all(ids[1:] > ids[:-1])
                    or not self.np.all((labels >= low) & (labels < high))):
                raise RuntimeError('RainbowPrompt task data include missing, future, replay or reordered samples')
            fingerprint = dataset_fingerprint(raw.data, labels, ids, raw.transform)
            if self.task == 0:
                from scripts.run_low_rank_efficiency import audit_fingerprints
                audit_fingerprints(fingerprint, self.record['common_reference_data_snapshots'][f'task1_{split}'],
                                   self.record['common_transform_snapshots'][f'task1_{split}'])
            self.recorder.data_snapshots[f'task{self.task + 1}_{split}'] = fingerprint
        self.task_old = self.old_state()
        self.record['structural_checks'].update(dataset_matches_common_reference=True, no_future_or_replay_samples=True)

    def attach_optimizer(self, optimizer):
        if optimizer in self.optimizers:
            return optimizer
        actual = {id(value) for group in optimizer.param_groups for value in group['params']}
        expected = {id(value) for value in self.branches['prompt'].parameters() if value.requires_grad}
        if (actual != expected or optimizer.state or len(optimizer.param_groups) != 1
                or optimizer.param_groups[0]['lr'] != .0075 or optimizer.param_groups[0]['weight_decay'] != 0.
                or optimizer.param_groups[0]['betas'] != (.9, .999) or optimizer.param_groups[0]['eps'] != 1e-8):
            raise RuntimeError('RainbowPrompt task-local Adam differs from the official approved optimizer')
        self.optimizers.add(optimizer)
        self.optimizer_ids.append(id(optimizer))
        self.optimizer = optimizer
        def before(subject, arguments, keywords):
            if self.stage != 'train':
                raise RuntimeError('RainbowPrompt optimizer update occurred outside ordinary training')
            self.frozen_finite()
            if self.epoch['steps'] == 78:
                self.last_batch_cache = self.prompt.stored_rainbow_prompts[self.task].detach().cpu().clone()
            if self.epoch['steps'] == 0 and self.epoch['epoch'] - 1 >= self.record['protocol']['warmup_epochs']:
                if self.active_prefix_layers != set(self.record['protocol']['prompt_layers']):
                    raise RuntimeError('RainbowPrompt did not write every active prefix cache before its first optimizer step')
                self.record['cache_audits'].append({'task': self.task + 1, 'epoch': self.epoch['epoch'],
                    'kind': 'first_rainbow_batch_pre_optimizer', 'layers': sorted(self.active_prefix_layers),
                    'prefix_matches_pre_update_cache': True,
                    'pre_update_cache_sha256': reference.state_digest({'cache': self.prompt.stored_rainbow_prompts[self.task]})})
        def after(subject, arguments, keywords):
            if self.epoch['steps'] == 78:
                if not self.torch.equal(self.last_batch_cache, self.prompt.stored_rainbow_prompts[self.task].detach().cpu()):
                    raise RuntimeError('RainbowPrompt optimizer step refreshed its stored pre-update cache')
                self.record['cache_audits'].append({'task': self.task + 1, 'epoch': self.epoch['epoch'],
                    'kind': 'last_batch_cache_timing', 'pre_post_optimizer_bitexact': True,
                    'cache_sha256': reference.state_digest({'cache': self.last_batch_cache})})
            self.record['counts']['optimizer_updates'] += 1
            self.epoch['steps'] += 1
        self.handles.extend((optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)))
        return optimizer

    def begin_epoch(self, index):
        if index not in range(self.record['protocol']['n_epochs']):
            raise RuntimeError('RainbowPrompt attempted an unauthorized epoch')
        seed = shared.seed_phase(1993, self.task, 'train', index)
        shared.configure_epoch_loader(self.dataset.train_loader, 1993, self.task, 'train', index)
        self.epoch = {'task': self.task + 1, 'epoch': index + 1, 'seed': seed, 'steps': 0, 'samples': 0, 'loss_sum': 0.}
        self.stage, self.active_prefix_layers = 'train', set()
        self.cache_before = self.prompt.stored_rainbow_prompts.detach().cpu().clone()
        self.last_batch_cache = None
        self.stream_ids, self.stream_labels = [], []

    def observed(self, inputs, labels):
        self.inputs, self.labels = inputs, labels
        if bool(((labels < self.task * 10) | (labels >= (self.task + 1) * 10)).any()):
            raise RuntimeError('RainbowPrompt training saw future or replay labels')
        self.epoch['samples'] += len(labels)
        self.record['counts']['observe_calls'] += 1
        self.stream_labels.extend(labels.detach().cpu().tolist())

    def backward(self, original):
        @functools.wraps(original)
        def wrapped(loss, *args, **kwargs):
            if self.stage == 'train' and (not kwargs.get('retain_graph', False) or not bool(self.torch.isfinite(loss).all())):
                raise RuntimeError('RainbowPrompt requires finite loss and source retain_graph=True')
            result = original(loss, *args, **kwargs)
            if self.stage == 'train':
                self.frozen_finite()
                self.epoch['loss_sum'] += float(loss.detach())
                if self.epoch['steps'] == 0:
                    gradients = {f'prompt.{name}': parameter.grad for name, parameter in self.branches['prompt'].named_parameters()
                                 if parameter.requires_grad and parameter.grad is not None}
                    self.record['first_batch_audits'].append({'task': self.task + 1, 'epoch': self.epoch['epoch'],
                        'seed': self.epoch['seed'], 'samples': len(self.labels),
                        'optimizer_update_index': self.record['counts']['optimizer_updates'] + 1,
                        'input_sha256': reference.state_digest({'inputs': self.inputs}),
                        'labels_sha256': reference.state_digest({'labels': self.labels}),
                        'gradient_sha256': reference.state_digest(gradients), 'gradient_timing': 'before_clip',
                        'effective_lr': self.optimizer.param_groups[0]['lr']})
            return result
        return wrapped

    def sampler_iter(self, original):
        def wrapped(sampler):
            if self.stage != 'train' or sampler is not self.dataset.train_loader.sampler:
                return original(sampler)
            def iterator():
                for index in original(sampler):
                    self.stream_ids.append(int(self.dataset.train_loader.dataset.indexes[index]))
                    yield index
            return iterator()
        return wrapped

    def prompt_output(self, module, arguments, keywords, result):
        train, layer = keywords.get('train', False), keywords['layer']
        if self.stage == 'train' and self.epoch['steps'] == 0 and keywords.get('p_type') == 'Rainbow':
            prefix = self.torch.cat(result['batched_prompt'], dim=1)
            cache = module.stored_rainbow_prompts[self.task, layer]
            if not self.torch.equal(prefix, cache.expand(prefix.shape[0], -1, -1)):
                raise RuntimeError('RainbowPrompt training cache does not match the returned pre-step prefix')
            self.active_prefix_layers.add(layer)
        if self.stage == 'evaluation':
            if train or keywords.get('task_id') not in (-1, None) or keywords.get('cur_id') != self.task:
                raise RuntimeError('RainbowPrompt evaluation received an oracle task label')
            features = keywords['cls_features']
            embedding = module.l2_normalize(features, dim=-1)
            scores = [self.torch.sum(module.l2_normalize(module.base_key[task], dim=-1) @ embedding.t()) / len(embedding)
                      for task in range(self.task + 1)]
            inferred = int(self.torch.stack(scores).argmax())
            selected = int(module.task_id)
            prefix = self.torch.cat(result['batched_prompt'], dim=1)
            if selected != inferred or not self.torch.equal(prefix, module.stored_rainbow_prompts[inferred, layer].expand(len(features), -1, -1)):
                raise RuntimeError('Unknown-task RainbowPrompt routing differs from one batch-mean selected cache')
            if layer == self.record['protocol']['prompt_layers'][0]:
                self.eval_routes.append({'test_task': self.eval_test_task + 1, 'samples': len(features), 'selected_task': selected,
                    'batch_mean_scores': [float(value.detach()) for value in scores], 'oracle_task_id': None})
                self.routing_layer_count = 0
            self.routing_layer_count += 1

    def end_epoch(self):
        self.frozen_finite()
        if self.epoch['steps'] != 79 or self.epoch['samples'] != 5000 or len(self.stream_ids) != 5000 or len(self.stream_labels) != 5000:
            raise RuntimeError('RainbowPrompt ordinary epoch stream is incomplete')
        if len(set(self.stream_ids)) != 5000 or set(self.stream_ids) != set(map(int, self.dataset.train_loader.dataset.indexes)):
            raise RuntimeError('RainbowPrompt sampler repeated or omitted current-task IDs')
        self.assert_old_unchanged()
        cache = self.prompt.stored_rainbow_prompts.detach().cpu()
        if self.last_batch_cache is None or not self.torch.equal(self.last_batch_cache, cache[self.task]):
            raise RuntimeError('RainbowPrompt epoch end changed the last pre-optimizer cached prefix')
        warmup = self.epoch['epoch'] - 1 < self.record['protocol']['warmup_epochs']
        if warmup and not self.torch.equal(cache, self.cache_before):
            raise RuntimeError('Unique warmup unexpectedly wrote the stored RainbowPrompt cache')
        if not warmup and (not bool(self.torch.isfinite(cache[self.task]).all()) or not bool(cache[self.task].abs().sum() > 0)):
            raise RuntimeError('RainbowPrompt active cache was not populated after warmup')
        row = {**self.epoch, 'loss_mean': self.epoch['loss_sum'] / self.epoch['steps']}
        self.record['epoch_results'].append(row)
        self.record['cache_audits'].append({'task': self.task + 1, 'epoch': self.epoch['epoch'],
            'kind': 'epoch_end', 'warmup_unique': warmup, 'cache_untouched_during_warmup': warmup,
            'cache_populated': not warmup, 'old_rows_unchanged': True,
            'cache_sha256': reference.state_digest({'cache': cache}),
            'old_state_sha256': reference.state_digest(self.old_state())})
        self.record['sample_stream_audits'].append({'task': self.task + 1, 'epoch': self.epoch['epoch'],
            'seed': self.epoch['seed'], 'samples': len(self.stream_ids),
            'global_ids_sha256': reference.state_digest({'ids': self.torch.tensor(self.stream_ids, dtype=self.torch.int64)}),
            'labels_sha256': reference.state_digest({'labels': self.torch.tensor(self.stream_labels, dtype=self.torch.int64)})})
        for filename, value in (('epoch_results.json', self.record['epoch_results']), ('progress.json', row)):
            write_json(self.recorder.output / filename, value)
        self.stage = None

    def begin_evaluation(self):
        from torch.utils.data import SequentialSampler
        self.stage, self.eval_predictions, self.eval_routes, self.eval_batches = 'evaluation', [], [], []
        shared.seed_phase(1993, self.task, 'evaluation')
        for loader in self.dataset.test_loaders:
            if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last or loader.batch_size != 64:
                raise RuntimeError('RainbowPrompt evaluation must preserve separate complete task batches of 64')
            shared.configure_epoch_loader(loader, 1993, self.task, 'evaluation')
        self.eval_test_task = 0

    def evaluation_output(self, inputs, logits):
        if logits.shape != (len(inputs), (self.task + 1) * 10) or not bool(self.torch.isfinite(logits).all()):
            raise RuntimeError('RainbowPrompt evaluation returned invalid seen-class logits')
        if self.routing_layer_count != len(self.record['protocol']['prompt_layers']):
            raise RuntimeError('RainbowPrompt evaluation did not route every prefix layer')
        self.eval_predictions.append(logits.argmax(1).detach().cpu().numpy())
        self.eval_batches.append(len(inputs))
        self.last_probe = (inputs.detach().clone(), logits.detach().clone())
        if sum(self.eval_batches) % 1000 == 0:
            self.eval_test_task += 1

    def end_evaluation(self, accuracy=None):
        predicted = self.np.concatenate(self.eval_predictions)
        labels = self.np.concatenate([self.np.asarray(loader.dataset.targets) for loader in self.dataset.test_loaders])
        ids = self.np.concatenate([self.np.asarray(loader.dataset.indexes) for loader in self.dataset.test_loaders])
        expected_batches = [64] * 15 + [40]
        if self.eval_batches != expected_batches * (self.task + 1) or len(predicted) != (self.task + 1) * 1000:
            raise RuntimeError('RainbowPrompt evaluation merged task loaders or dropped their routing tails')
        counts = exact_counts(predicted, labels, self.task + 1)
        if accuracy is not None and abs(sum(map(float, accuracy[0])) / len(accuracy[0]) - counts['seen_accuracy_pct']) > 1e-9:
            raise RuntimeError('Native RainbowPrompt accuracy disagrees with saved predictions')
        order = self.np.argsort(ids, kind='stable')
        path = self.recorder.output / f'task{self.task + 1}_predictions.npz'
        self.np.savez_compressed(path, predictions=predicted[order], labels=labels[order], global_ids=ids[order],
                                 evaluation_global_ids=ids, batch_sizes=self.np.asarray(self.eval_batches),
                                 inferred_tasks=self.np.asarray([row['selected_task'] for row in self.eval_routes]))
        self.record['prediction_files'].append({'task': self.task + 1, 'path': str(path), 'sha256': shared.file_hash(path)})
        self.record['task_results'].append(counts)
        self.record['accuracy'] = {'setting': 'class-il', 'mean_accuracy_pct': counts['seen_accuracy_pct']}
        self.record['routing_audits'].append({'task': self.task + 1, 'evaluation_global_ids_sha256': reference.state_digest({'ids': self.torch.from_numpy(ids)}),
                                             'batches': self.eval_routes, 'all_prefix_layers_match_selected_cache': True})
        self.record['structural_checks'].update(evaluation_predictions_match_accuracy=True, prediction_ids_and_counts_complete=True,
                                               unknown_task_batch_routing_preserved=True)
        self.stage = None
        self.finalized()
        status = notify_slack(f'RainbowPrompt {self.record["side"]} {self.record["mode"]} {self.record["pass"]}: task{self.task + 1} completed, seen={counts["seen_accuracy_pct"]:.2f}%; {self.record["wandb_url"]}')
        self.record.setdefault('task_notification_status', []).append(status)
        if status != 'sent':
            raise RuntimeError('Required RainbowPrompt task notification failed')
        floor = 90 if self.task == 0 else 20
        if counts['seen_accuracy_pct'] < floor or self.task > 0 and min(counts['old_accuracy_pct'], counts['new_accuracy_pct']) <= floor:
            raise RuntimeError('RainbowPrompt accuracy is abnormally low; further runs are blocked without recipe changes')

    def finalized(self):
        with self.recorder.phase('state_audit'):
            canonical = canonical_hashes(self.branches)
            if any(value != shared.CANONICAL_PRETRAINED_SHA256 for value in canonical.values()):
                raise RuntimeError('Frozen RainbowPrompt pretrained tensors changed')
            self.assert_old_unchanged()
            state = reference.learned_state(self.branches)
            if any(not bool(self.torch.isfinite(value).all()) for value in state.values()):
                raise FloatingPointError('Finalized RainbowPrompt learned/cache state is nonfinite')
            cache_before = self.prompt.stored_rainbow_prompts.detach().cpu().clone()
            lifecycle = None
            if self.native is not None:
                lifecycle = {key: value.detach().cpu().clone() for key, value in self.native.net.state_dict().items()
                             if key in ('active_task', 'task_offsets', 'task_finalized', 'rainbow_ready')}
                lifecycle['extra'] = self.native.get_extra_state()
                lifecycle['network_extra'] = self.native.net.get_extra_state()
                if int(self.native.net.active_task) != self.task or not bool(self.native.net.task_finalized) or not bool(self.native.net.rainbow_ready[:self.task + 1].all()):
                    raise RuntimeError('Finalized Mammoth RainbowPrompt lifecycle counters are invalid')
        with self.recorder.phase('checkpoint'):
            path = self.recorder.output / f'task{self.task + 1}_state.pt'
            payload = {'schema_version': 1, 'state': state, 'learned_sha256': reference.state_digest(state),
                'task': self.task + 1, 'seen_classes': (self.task + 1) * 10, 'lifecycle': lifecycle,
                'canonical_pretrained': canonical, 'protocol': self.record['protocol']}
            self.torch.save(payload, path)
            saved = self.torch.load(path, map_location='cpu', weights_only=True)
            if reference.state_digest(saved['state']) != payload['learned_sha256'] or saved['protocol'] != payload['protocol']:
                raise RuntimeError('RainbowPrompt compact checkpoint changed on disk')
            for branch, module in self.branches.items():
                selected = {key[len(branch) + 1:]: value for key, value in saved['state'].items() if key.startswith(branch + '.')}
                result = module.load_state_dict(selected, strict=False)
                if result.unexpected_keys:
                    raise RuntimeError('RainbowPrompt compact checkpoint has unexpected state')
            if self.native is not None:
                for name, value in saved['lifecycle'].items():
                    if name not in ('extra', 'network_extra'):
                        getattr(self.native.net, name).copy_(value.to(self.native.device))
                self.native.set_extra_state(saved['lifecycle']['extra'])
                self.native.net.set_extra_state(saved['lifecycle']['network_extra'])
            if reference.state_digest(reference.learned_state(self.branches)) != payload['learned_sha256']:
                raise RuntimeError('RainbowPrompt compact checkpoint did not restore all learned state')
        with self.recorder.phase('checkpoint_probe'), self.torch.no_grad():
            self.branches['query'].eval()
            self.branches['prompt'].eval()
            inputs, before = self.last_probe
            if self.native is not None:
                after = self.native.net.outputs(inputs, train=False)['logits'][:, :(self.task + 1) * 10]
            else:
                query = self.branches['query'](inputs)['pre_logits']
                after = self.branches['prompt'](inputs, task_id=-1, learned_id=self.task, cls_features=query, train=False)['logits'][:, :(self.task + 1) * 10]
            if not self.torch.equal(before, after) or not self.torch.equal(cache_before, self.prompt.stored_rainbow_prompts.detach().cpu()):
                raise RuntimeError('RainbowPrompt checkpoint changed deterministic outputs or recomputed its pre-update cache')
        self.record['state_audits'].append({'task': self.task + 1, 'canonical_pretrained': canonical,
            'learned_sha256': payload['learned_sha256'], 'checkpoint_sha256': shared.file_hash(path),
            'compact_checkpoint_bytes': path.stat().st_size, 'checkpoint_probe_examples': len(inputs),
            'checkpoint_outputs_bitexact': True, 'cached_pre_update_prompts_retained': True,
            'old_state_unchanged': True, 'task_local_optimizer_reset': True,
            'cache_populated_after_warmup': True, 'cache_pre_post_last_optimizer_bitexact': True,
            'cache_untouched_during_warmup': True, 'no_end_task_cache_refresh': True,
            'batch_mean_unknown_task_routing': True, 'phase_rng_epoch_seeds_verified': True})
        self.snapshot(f'after_task{self.task + 1}')
        self.record['structural_checks'].update(frozen_backbone_unchanged=True, method_state_valid=True,
            checkpoint_round_trip=True, stored_prompt_cache_populated_after_warmup=True, stored_prompt_cache_timing_matches=True)

    def end_task_cache_guard(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            self.stage = 'task_end'
            shared.seed_phase(1993, self.task, 'task_end')
            before = self.prompt.stored_rainbow_prompts.detach().clone()
            result = original(*args, **kwargs)
            if not self.torch.equal(before, self.prompt.stored_rainbow_prompts):
                raise RuntimeError('RainbowPrompt task end refreshed the official pre-update prompt cache')
            self.record['cache_audits'].append({'task': self.task + 1, 'kind': 'task_end',
                'no_cache_refresh': True, 'cache_sha256': reference.state_digest({'cache': before})})
            self.stage = None
            return result
        return wrapped

    def install_common(self, stack):
        stack.enter_context(patch.object(self.torch.Tensor, 'backward', self.backward(self.torch.Tensor.backward)))
        sampler = self.torch.utils.data.RandomSampler
        stack.enter_context(patch.object(sampler, '__iter__', self.sampler_iter(sampler.__iter__)))
        loader = self.torch.utils.data.DataLoader
        original_iter = loader.__iter__
        def counted(subject):
            if self.stage not in ('task_begin', 'task_end'):
                return original_iter(subject)
            def iterator():
                for batch in original_iter(subject):
                    self.record['counts'][self.stage + '_extra_samples'] += len(batch[0])
                    yield batch
            return iterator()
        stack.enter_context(patch.object(loader, '__iter__', counted))

    def install_native(self, stack, training):
        stack.enter_context(instrument(self.recorder, self.native, self.dataset, training))
        self.install_common(stack)
        def loaders(original):
            def wrapped(*args, **kwargs):
                shared.seed_phase(1993, len(self.dataset.test_loaders), 'task_begin')
                result = original(*args, **kwargs)
                self.load_task()
                return result
            return wrapped
        def begin_task(original):
            def wrapped(*args, **kwargs):
                self.stage = 'task_begin'
                shared.seed_phase(1993, self.task, 'task_begin')
                result = original(*args, **kwargs)
                self.stage = None
                return result
            return wrapped
        def optimizer(original):
            def wrapped(*args, **kwargs):
                return self.attach_optimizer(original(*args, **kwargs))
            return wrapped
        def begin_epoch(original):
            def wrapped(index, dataset):
                self.attach_optimizer(self.native.opt)
                self.begin_epoch(index)
                return original(index, dataset)
            return wrapped
        def observe(original):
            def wrapped(inputs, labels, *args, **kwargs):
                self.observed(inputs, labels)
                return original(inputs, labels, *args, **kwargs)
            return wrapped
        def end_epoch(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                self.end_epoch()
                return result
            return wrapped
        def forward(original):
            def wrapped(inputs, *args, **kwargs):
                logits = original(inputs, *args, **kwargs)
                if self.stage == 'evaluation':
                    if args or kwargs:
                        raise RuntimeError('Native RainbowPrompt evaluation used oracle arguments')
                    self.evaluation_output(inputs, logits)
                return logits
            return wrapped
        def evaluate(original):
            def wrapped(*args, **kwargs):
                self.begin_evaluation()
                result = original(*args, **kwargs)
                self.end_evaluation(result)
                return result
            return wrapped
        for target, name, wrapper in ((self.dataset, 'get_data_loaders', loaders), (self.native, 'meta_begin_task', begin_task),
                (self.native, 'get_optimizer', optimizer), (self.native, 'meta_begin_epoch', begin_epoch),
                (self.native, 'meta_observe', observe), (self.native, 'meta_end_epoch', end_epoch),
                (self.native, 'meta_end_task', self.end_task_cache_guard), (self.native, 'forward', forward),
                (self.dataset, 'evaluate', evaluate)):
            stack.enter_context(patch.object(target, name, wrapper(getattr(target, name))))

    def close(self):
        for handle in self.handles:
            handle.remove()


def effective_args(args, dataset, spec):
    p = spec['protocol']
    expected = {'stop_after': p['tasks'], 'n_epochs': p['n_epochs'], 'batch_size': 64, 'num_workers': 4,
        'seed': 1993, 'dataset_config': 'ease', 'permute_classes': 0, 'noise_rate': 0, 'validation': 0,
        'label_perc': 1, 'label_perc_by_class': 1, 'debug_mode': 0, 'enable_other_metrics': 0,
        'eval_future': 0, 'eval_epochs': None, 'fitting_mode': 'epochs', 'lr_scheduler': None,
        'joint': 0, 'code_optimization': 0, 'lr': .03, 'optim_wd': 0., 'optimizer': 'adam',
        'rainbow_vit_size': 'base', 'rainbow_warmup_epochs': p['warmup_epochs'], 'rainbow_prompt_length': 20,
        'rainbow_pool_size': 10, 'rainbow_d1': 56, 'rainbow_d2': 96, 'rainbow_ki_iter': 10,
        'rainbow_balance': .01, 'rainbow_clip_grad': 1.}
    if any(getattr(args, key, None) != value for key, value in expected.items()) or (dataset.N_TASKS, dataset.N_CLASSES_PER_TASK, dataset.N_CLASSES) != (10, 10, 100):
        raise ValueError('Effective RainbowPrompt arguments differ from the approved recipe')
    if list(args.rainbow_prompt_layers) != list(range(12)) or list(args.rainbow_self_attn_layers) != list(range(6)):
        raise ValueError('RainbowPrompt prefix or self-attention layers changed')
    return vars(args)


def train_author(code, branches, dataset, audit, record):
    torch, p = audit.torch, record['protocol']
    device = torch.device('cuda:0')
    source_args = SimpleNamespace(distributed=False, epochs=p['n_epochs'], print_freq=10000,
        train_mask=True, nb_classes=100, balancing=.01)
    class_mask = [list(range(task * 10, (task + 1) * 10)) for task in range(10)]
    for task in range(p['tasks']):
        shared.seed_phase(1993, task, 'task_begin')
        with audit.recorder.phase('data_loading'):
            dataset.get_data_loaders()
            audit.load_task()
        with audit.recorder.phase('begin_task'):
            optimizer = torch.optim.Adam((parameter for parameter in branches['prompt'].parameters() if parameter.requires_grad), lr=.0075, weight_decay=0.)
            audit.attach_optimizer(optimizer)
        for epoch in range(p['n_epochs']):
            audit.begin_epoch(epoch)
            def observed(inputs, labels):
                # Observe the original CPU batch; the immutable engine performs
                # its one host-to-device transfer, exactly as the source does.
                audit.observed(inputs, labels)
                audit.recorder.steps += 1
                audit.recorder.training_examples += len(labels)
            with audit.recorder.phase('train_epoch'):
                code.train_one_epoch(branches['prompt'], branches['query'], reference.SeenClassCriterion((task + 1) * 10),
                    reference.PairLoader(dataset.train_loader, observed), optimizer, device, epoch, max_norm=1.,
                    task_id=task, class_mask=class_mask, args=source_args)
            audit.end_epoch()
        with audit.recorder.phase('end_task'):
            shared.seed_phase(1993, task, 'task_end')
            audit.assert_old_unchanged()
            record['cache_audits'].append({'task': task + 1, 'kind': 'task_end', 'no_cache_refresh': True,
                'cache_sha256': reference.state_digest({'cache': audit.prompt.stored_rainbow_prompts})})
        with audit.recorder.phase('evaluation'), torch.no_grad():
            branches['query'].eval()
            branches['prompt'].eval()
            audit.begin_evaluation()
            for loader in dataset.test_loaders:
                for data in loader:
                    inputs = data[0].to(device)
                    features = branches['query'](inputs)['pre_logits']
                    logits = branches['prompt'](inputs, task_id=-1, learned_id=task, cls_features=features, train=False)['logits'][:, :(task + 1) * 10]
                    audit.evaluation_output(inputs, logits)
                    audit.recorder.evaluation_examples += len(inputs)
        audit.end_evaluation()
        audit.recorder.tasks_completed += 1


def complete_checks(record, recorder):
    p = record['protocol']
    expected = p['tasks'] * p['n_epochs'] * 79
    if (record['counts'] != {'observe_calls': expected, 'optimizer_updates': expected,
                            'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}
            or recorder.steps != expected or recorder.training_examples != p['tasks'] * p['n_epochs'] * 5000
            or recorder.evaluation_examples != (1000 if p['tasks'] == 1 else 3000) or recorder.tasks_completed != p['tasks']):
        raise RuntimeError('RainbowPrompt complete workload counters differ from the approved scope')
    checks = record['structural_checks']
    checks.update(training_protocol_complete=True, optimizer_updates_complete=True,
                  finite_loss_and_gradients=True, frozen_backbone_gradients_absent=True, phase_rng_matches_protocol=True)
    if not all(checks.values()):
        raise RuntimeError('RainbowPrompt did not establish every required structural check')


def upload_record(wandb, output, record):
    """Upload immutable successful bytes; any failed transfer blocks acceptance."""
    record['wandb_logging_status'] = 'uploaded'
    for name in ('result.json', 'metrics.json', 'artifact_metrics.json'):
        write_json(output / name, record)
    metrics = flatten({'efficiency': record['metrics'], 'phases': record['phases'], 'counts': record['counts'],
        'accuracy': record['accuracy'], 'structural_checks': record['structural_checks'], 'flops': record.get('flops', {})})
    wandb.run.summary.update({**{key: value for key, value in metrics.items() if isinstance(value, (int, float, bool))},
                             'benchmark_status': record['status'], 'benchmark_pass': record['pass']})
    artifact = wandb.Artifact(record['name'], type='cl-efficiency')
    names = ('spec.json', 'result.json', 'metrics.json', 'artifact_metrics.json', 'effective_args.json', 'shared_data.json',
        'epoch_results.json', 'task_results.json', 'first_batch_audits.json', 'state_audits.json', 'cache_audits.json',
        'routing_audits.json', 'sample_stream_audits.json', 'source_adaptations.json', 'telemetry.csv', 'phases.json', 'flops_ops.json')
    names += tuple(f'task{task}_predictions.npz' for task in range(1, record['protocol']['tasks'] + 1))
    for name in names:
        if (output / name).is_file():
            artifact.add_file(str(output / name))
    published = wandb.run.log_artifact(artifact)
    published.wait()
    receipt = {'run_id': wandb.run.id, 'wandb_url': record['wandb_url'], 'artifact_id': published.id,
               'artifact_state': str(published.state), 'result_sha256': shared.file_hash(output / 'result.json')}
    wandb.finish(exit_code=0 if record['status'] == 'completed' else 1)
    write_json(output / 'wandb_receipt.json', receipt)


def worker(spec_path):
    if os.getenv('CL_RAINBOW_EXECUTE') != '1' or os.getenv('CUDA_VISIBLE_DEVICES') != '1' or os.getenv('WANDB_MODE') != 'online':
        raise ValueError('RainbowPrompt worker requires explicit execution on GPU1 with online W&B')
    configure_native_environment()
    spec = json.loads(Path(spec_path).read_text())
    shared.validate_spec(spec)
    if shared.execution_sources() != spec['execution_source_sha256']:
        raise ValueError('RainbowPrompt execution sources changed after preparation')
    for path, digest in ((spec['checkpoint_file'], shared.CHECKPOINT_SHA256), (spec['initial_state'], spec['initial_state_sha256']),
                         (spec['common_native_summary'], spec['common_native_summary_sha256'])):
        if shared.file_hash(path) != digest:
            raise ValueError('RainbowPrompt pinned input evidence changed')
    reference.verify_author_source(spec['reference_source'], spec['reference_source_sha256'])
    if validate_shared_data(spec['data_root']) != spec['data_files']:
        raise ValueError('Shared CIFAR100 raw data changed')
    output = Path(spec['output'])
    recorder = Recorder(output, '1')
    recorder.start()
    record = {**spec, 'status': 'failed', 'accuracy': {}, 'counts': {'observe_calls': 0, 'optimizer_updates': 0,
        'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}, 'epoch_results': [], 'task_results': [],
        'first_batch_audits': [], 'prediction_files': [], 'state_audits': [], 'cache_audits': [], 'routing_audits': [],
        'sample_stream_audits': [], 'structural_checks': {key: False for key in REQUIRED_CHECKS}}
    audit, wandb, exit_code = None, None, 1
    try:
        import numpy as np
        import torch
        import wandb
        import main as mammoth
        from utils import training
        from utils.conf import base_path
        from scripts.ease_reference_utils import shared_cifar_metadata
        record['precision'] = configure_precision(torch)
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        base_path(str(Path(spec['data_root']).resolve()))
        shared.seed_phase(1993, 0, 'task_begin')
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'wandb')}
        wandb.init(entity='laymond1', project='cl-efficiency', name=spec['name'], mode='online',
                   group='rainbow-active-validation', job_type=f'{spec["side"]}-{spec["mode"]}-{spec["pass"]}', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        command = shared.native_command(spec, output)
        sys.argv = ['main.py', *command]
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) if spec['pass'] == 'flops' else None
        with profiler if profiler is not None else nullcontext():
            with recorder.phase('initialize'), forbid_cifar_download():
                args = mammoth.parse_args(cmd=command)
                if spec['side'] == 'native':
                    model, dataset, args = mammoth.initialize(args)
                    branches = {'query': model.net.query, 'prompt': model.net.prompt}
                    model.net.to(model.device)
                else:
                    from datasets import get_dataset
                    dataset = get_dataset(args)
                    code = reference.load_author_code(spec['reference_source'], spec['reference_source_sha256'])
                    branches = reference.build_author_branches(code, spec['checkpoint_file'], warmup=spec['protocol']['warmup_epochs'])
                    for module in branches.values():
                        module.to(recorder.device)
                    model = None
                    record['source_adaptations'] = code.adaptations
            args.nowand = 1
            write_json(output / 'effective_args.json', effective_args(args, dataset, spec))
            write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
            with recorder.phase('state_audit'):
                initial = reference.load_initial_state(branches, spec['initial_state'], shared.CHECKPOINT_SHA256)
                canonical = canonical_hashes(branches)
                if any(value != shared.CANONICAL_PRETRAINED_SHA256 for value in canonical.values()):
                    raise ValueError('RainbowPrompt canonical pretrained initialization differs from the pin')
            record.update(initial_learned_sha256=initial, initial_canonical_pretrained_sha256=shared.CANONICAL_PRETRAINED_SHA256,
                          initial_canonical_pretrained_branches=canonical, parameter_audits=parameter_audit(branches))
            record['structural_checks'].update(canonical_pretrained_matches_pin=True, learnable_initialization_matches=True)
            audit = RainbowAudit(branches, dataset, recorder, record, torch, np, model)
            with ExitStack() as stack:
                stack.enter_context(forbid_cifar_download())
                if spec['side'] == 'native':
                    audit.install_native(stack, training)
                    training.train(model, dataset, args)
                else:
                    audit.install_common(stack)
                    train_author(code, branches, dataset, audit, record)
            complete_checks(record, recorder)
        if profiler is not None:
            operations = [{'operation': item.key, 'count': item.count, 'flops': item.flops} for item in profiler.key_averages() if item.flops]
            total = sum(item['flops'] for item in operations)
            if not math.isfinite(total) or total <= 0:
                raise RuntimeError('Separate RainbowPrompt profiler produced no positive supported operator estimate')
            record['flops'] = {'supported_ops_total': total, 'partial': True, 'includes_backward_only_if_supported': True,
                'scope': 'Supported operators across initialization, full training, evaluation, live audits and checkpoint probes in this separate pass',
                'limitations': 'Unsupported or fused operators, optimizer, CPU/PIL transforms and serialization may be omitted; no backward multiplier or phase attribution',
                'measurement_aggregate_eligible': False}
            write_json(output / 'flops_ops.json', operations)
        record.update(status='completed', computation_status='completed')
        exit_code = 0
    except BaseException as error:
        record.update(error_type=type(error).__name__, error='RainbowPrompt worker failed; inspect local phase, count and structural audits', computation_status='failed')
        record['error_context'] = error_context(error, recorder, audit)
        write_json(output / 'error_context.json', record['error_context'])
        logging.error('RainbowPrompt worker failed: %s', type(error).__name__)
    finally:
        if audit is not None:
            audit.close()
        record = recorder.save(record)
        for name in ('epoch_results', 'task_results', 'first_batch_audits', 'state_audits', 'cache_audits', 'routing_audits', 'sample_stream_audits'):
            write_json(output / (name + '.json'), record[name])
        write_json(output / 'source_adaptations.json', record.get('source_adaptations', ['Native Mammoth train() and meta wrappers; common phase RNG and read-only source-equivalence audits']))
        record['measurement_notes']['diagnostic_scope'] = 'Matched live audits, compact checkpoints and deterministic 40-example checkpoint probes are included in cost; probes are excluded from ordinary evaluation counts'
        if record['status'] == 'completed':
            try:
                validate_saved_record(record)
            except Exception as error:
                record.update(status='failed', computation_status='failed', error_type=type(error).__name__)
                exit_code = 1
        write_json(output / 'result.json', record)
        if wandb is not None and wandb.run is not None:
            try:
                upload_record(wandb, output, record)
            except Exception as error:
                record.update(status='logging_failed', wandb_logging_status='failed', error_type=type(error).__name__)
                write_json(output / 'result.json', record)
                exit_code = 1
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass
    return exit_code


def execute(options, profile):
    spec = build_spec(options, profile)
    if not valid_slack_webhook():
        raise ValueError('Required Slack webhook is unavailable')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []
    def notify(message):
        status = notify_slack(message)
        notifications.append({'message': message, 'status': status})
        write_json(output / 'notifications.json', notifications)
        return status
    if notify(f'RainbowPrompt {spec["side"]} {spec["mode"]} {spec["pass"]} started: GPU1 batch64, official active path') != 'sent':
        raise RuntimeError('Required RainbowPrompt start notification failed before training')
    env = os.environ.copy()
    env.pop('MAMMOTH_DEVICE', None)
    env.update(CUDA_VISIBLE_DEVICES='1', WANDB_MODE='online', WANDB_DIR=str(output), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', CL_RAINBOW_EXECUTE='1')
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
                    notify(f'RainbowPrompt {spec["side"]} {spec["mode"]} {spec["pass"]} still running; local audits preserved')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify(f'RainbowPrompt {spec["side"]} interrupted; local results preserved')
            raise
    record = json.loads((output / 'result.json').read_text()) if (output / 'result.json').is_file() else {'status': 'failed'}
    status = 'timeout' if timed_out else record['status']
    if process.returncode != 0 and status == 'completed':
        status = 'failed'
    sent = notify(f'RainbowPrompt {spec["side"]} {spec["mode"]} {spec["pass"]} {status}; {record.get("wandb_url", "W&B unavailable")}')
    write_json(output / 'completion.json', {'status': status, 'exit_code': process.returncode, 'timeout': timed_out,
        'wall_s': time.monotonic() - started, 'wandb_url': record.get('wandb_url'), 'slack_completion_status': sent})
    print(json.dumps({'status': status, 'result': str(output / 'result.json'), 'wandb_url': record.get('wandb_url')}))
    return 0 if process.returncode == 0 and status == 'completed' and sent == 'sent' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=('author', 'native'), default='native')
    parser.add_argument('--mode', choices=shared.MODES, default='full_validation')
    parser.add_argument('--pass', dest='pass_kind', choices=('validation', 'measurement', 'flops'), default='validation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path(shared.CHECKPOINT))
    parser.add_argument('--source', type=Path, default=shared.REFERENCE_SOURCE)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--timeout', type=int, default=21600)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--prepare-initial-state', type=Path)
    options = parser.parse_args(argv)
    if options.worker:
        return worker(options.worker)
    if options.prepare_initial_state:
        print(json.dumps(prepare_initial_state(options.prepare_initial_state, options.source, options.checkpoint)))
        return 0
    if options.output is None or options.initial_state is None:
        parser.error('--output and --initial-state are required for preparation or execution')
    profile = configure_environment(options.profile, options.credentials)
    if options.execute:
        return execute(options, profile)
    print(json.dumps(build_spec(options, profile), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
