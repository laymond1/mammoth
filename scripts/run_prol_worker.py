"""Opt-in pinned PROL author/native validation and isolated cost pilots.

Importing and preparation start no training. Native execution always delegates to
Mammoth's real trainer and meta hooks; the independent runner compiles the
reviewed author classes and numerical epoch engine without importing the port.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
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
from scripts import prol_protocol as shared
from scripts import prol_reference_utils as reference
from scripts.cl_efficiency import Recorder, flatten, instrument, model_snapshot, notify_slack, tensor_bytes, valid_slack_webhook, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.hide_protocol import forbid_cifar_download, validate_shared_data

REQUIRED_CHECKS = shared.REQUIRED_CHECKS
PRECISION = shared.PRECISION


def build_spec(options, profile):
    shared.load_run_plan()
    if any(profile[key] != value for key, value in {'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'}.items()):
        raise ValueError('PROL requires the approved GPU1 and W&B destination')
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
        raise ValueError('PROL requires the common pinned local NPZ')
    pins = shared.verify_reference_source(source)
    reference.verify_author_source(source, pins)
    name = f'prol-{options.side}-{options.mode}-{options.pass_kind}-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec = {'schema_version': 1, 'name': name, 'suite': name, 'method': 'prol-prompt', 'model': 'prol-prompt',
        'side': options.side, 'mode': options.mode, 'pass': options.pass_kind,
        'implementation': 'mammoth_native' if options.side == 'native' else 'pinned_author_reference',
        'protocol': shared.protocol(options.mode), 'source_mode': 'official-active-path', 
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
        raise ValueError('PROL completed record has incomplete protocol or structural evidence')
    for row, task_epoch in zip(record['epoch_results'], ((task, epoch) for task in range(1, tasks + 1) for epoch in range(1, epochs + 1))):
        task, epoch = task_epoch
        if (any(type(row.get(key)) is not int or row[key] != value for key, value in
                {'task': task, 'epoch': epoch, 'steps': 79, 'samples': 5000,
                 'seed': shared.phase_seed(1993, task - 1, 'train', epoch - 1)}.items())
                or not math.isfinite(row.get('loss_mean', float('nan')))):
            raise ValueError('PROL epoch evidence is incomplete or has the wrong phase RNG')
    for task, row in enumerate(record['task_results'], 1):
        for prefix, samples in (('seen', task * 1000), ('old', (task - 1) * 1000), ('new', 1000)):
            correct = row.get(prefix + '_correct')
            if (type(row.get(prefix + '_samples')) is not int or row[prefix + '_samples'] != samples
                    or type(correct) is not int or not 0 <= correct <= samples
                    or samples and row.get(prefix + '_accuracy_pct') != 100 * correct / samples):
                raise ValueError('PROL exact saved prediction counts disagree with accuracy')
        if row.get('task') != task or row['seen_correct'] != row['old_correct'] + row['new_correct']:
            raise ValueError('PROL task counts disagree')
    if record.get('accuracy', {}).get('mean_accuracy_pct') != record['task_results'][-1]['seen_accuracy_pct']:
        raise ValueError('PROL final accuracy differs from saved counts')
    metrics = record.get('metrics', {})
    for key, value in {'outer_observe_steps': updates, 'outer_training_stream_examples': tasks * epochs * 5000,
                       'evaluation_examples': 1000 if tasks == 1 else 3000, 'tasks_completed': tasks}.items():
        if type(metrics.get(key)) is not int or metrics[key] != value:
            raise ValueError('PROL resource metrics do not cover the complete workload')
    return True


def configure_precision(torch):
    from scripts.benchmark_cl import _configure_ease_precision
    torch.set_num_threads(2)
    value = {**_configure_ease_precision(torch), 'torch_cpu_threads': torch.get_num_threads(),
             'autocast': torch.is_autocast_enabled()}
    if value != PRECISION:
        raise ValueError('PROL requires common FP32 precision')
    return value


def prepare_initial_state(path, source=shared.REFERENCE_SOURCE, checkpoint=shared.CHECKPOINT):
    """Explicit CPU-only preparation from seeded independent author initialization."""
    configure_native_environment()
    import torch
    if shared.file_hash(checkpoint) != shared.CHECKPOINT_SHA256:
        raise ValueError('PROL initialization requires the pinned common NPZ')
    code = reference.load_author_code(source, shared.verify_reference_source(source))
    configure_precision(torch)
    torch.manual_seed(1993)
    branches = reference.build_author_branches(code, checkpoint)
    canonical = canonical_hashes(branches)
    if any(value != shared.CANONICAL_PRETRAINED_SHA256 for value in canonical.values()):
        raise ValueError('Independent PROL initialization did not load all canonical pretrained tensors')
    metadata = reference.save_initial_state(branches, path, shared.CHECKPOINT_SHA256)
    return {**metadata, 'path': str(Path(path).resolve()), 'sha256': shared.file_hash(path), 'canonical_pretrained': canonical}


def canonical_hashes(branches):
    from scripts.ease_size_protocol import canonical_pretrained_digest
    return {name: canonical_pretrained_digest(module) for name, module in branches.items()}


def error_context(error, recorder, audit):
    """Keep code locations, never exception messages, environment or locals."""
    proof = audit.record.get('checkpoint_probe_audits', [])[-1:] if audit is not None else []
    return {'error_type': type(error).__name__, 'phase': recorder.active_phase,
            'last_checkpoint_check_phase': proof[0].get('phase') if proof else None,
            'task': audit.task + 1 if audit is not None and audit.task >= 0 else None,
            'epoch': audit.epoch.get('epoch') if audit is not None else None,
            'frames': [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                       for frame in traceback.extract_tb(error.__traceback__)]}


def exact_counts(predicted, labels, task):
    import numpy as np
    predicted, labels = np.asarray(predicted), np.asarray(labels)
    if predicted.shape != labels.shape or labels.ndim != 1:
        raise ValueError('PROL prediction shape mismatch')
    old, correct = labels < (task - 1) * 10, predicted == labels
    result = {'task': task}
    for prefix, mask in (('seen', np.ones(len(labels), dtype=bool)), ('old', old), ('new', ~old)):
        samples, count = int(mask.sum()), int(correct[mask].sum())
        result.update({prefix + '_samples': samples, prefix + '_correct': count,
                       prefix + '_accuracy_pct': 100 * count / samples if samples else None})
    return result



def parameter_audit(branches):
    parameters = [parameter for module in branches.values() for parameter in module.parameters()]
    return {'trainable_parameters': sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            'frozen_parameters': sum(parameter.numel() for parameter in parameters if not parameter.requires_grad),
            'total_parameters': sum(parameter.numel() for parameter in parameters),
            'generator_parameters': sum(parameter.numel() for name, parameter in branches['prompt'].e_prompt.named_parameters()
                                        if 'comp_gen' in name),
            'query_head_parameters': sum(parameter.numel() for parameter in branches['query'].head.parameters()),
            'prompt_head_count': len(branches['prompt'].head.heads)}


def rng_digest(torch):
    state = {'cpu': torch.random.get_rng_state()}
    if torch.cuda.is_initialized():
        state.update({f'cuda{index}': value for index, value in enumerate(torch.cuda.get_rng_state_all())})
    return reference.state_digest(state)


def frozen_encoder_hashes(branches):
    """Prompted LayerNorms are learned; only its other 100 canonical tensors freeze."""
    from scripts.ease_size_protocol import canonical_pretrained_state
    result = {}
    for branch, module in branches.items():
        state = canonical_pretrained_state(module)
        if branch == 'prompt':
            state = {key: value for key, value in state.items() if not reference.learned_name(branch, key)}
        result[branch] = {'sha256': reference.state_digest(state), 'tensors': len(state)}
    return result


def compact_clone(branches, saved, native=None):
    """Restore new heads/learned modules while sharing immutable encoder storage.

    No second 172M-parameter encoder allocation is necessary. Learned norms,
    heads and prompt modules have separate storage. Dynamic head restoration
    starts with one head and runs under a CPU/CUDA RNG fork.
    """
    import torch
    expected = reference.learned_state(branches)
    state = saved.get('state', {})
    if (set(state) != set(expected) or reference.state_digest(state) != saved.get('learned_sha256')
            or any(value.shape != expected[name].shape or value.dtype != expected[name].dtype
                   or not bool(torch.isfinite(value).all()) for name, value in state.items())):
        raise ValueError('PROL compact checkpoint learner catalogue is incomplete or corrupt')
    devices = [next(branches['prompt'].parameters()).device.index] if next(branches['prompt'].parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        clones = {}
        for branch, module in branches.items():
            cloned = copy.copy(module)
            cloned._modules = dict(module._modules)
            cloned._forward_hooks = OrderedDict()
            cloned._forward_pre_hooks = OrderedDict()
            if branch == 'prompt':
                cloned.e_prompt = copy.deepcopy(module.e_prompt, {id(module.e_prompt._forward_hooks): OrderedDict(),
                                                                  id(module.e_prompt._forward_pre_hooks): OrderedDict()})
                # Hooks belong to live audits, not the deterministic probe.
                cloned.e_prompt._forward_hooks.clear()
                cloned.e_prompt._forward_pre_hooks.clear()
                cloned.head = type(module.head)(module.num_features, module.num_classes // module.num_tasks)
                cloned.head.to(next(module.parameters()).device)
                cloned.norm = copy.deepcopy(module.norm)
                cloned.blocks = torch.nn.Sequential()
                for block in module.blocks:
                    child = copy.copy(block)
                    child._modules = dict(block._modules)
                    child.norm1, child.norm2 = copy.deepcopy(block.norm1), copy.deepcopy(block.norm2)
                    cloned.blocks.append(child)
            else:
                cloned.head = copy.deepcopy(module.head)
            clones[branch] = cloned
        if native is not None:
            from models.prol_prompt_utils.network import ProlNetwork
            network = ProlNetwork(clones['query'], clones['prompt'], native.net.total_tasks)
            network.pretrained_checkpoint = native.net.pretrained_checkpoint
            wrapper = copy.copy(native)
            wrapper._modules, wrapper._buffers = dict(native._modules), dict(native._buffers)
            wrapper.net = network
            full = dict(native.state_dict())
            # Only immutable pretrained tensors come from the live model.
            # Every learner/lifecycle value must be restored from disk bytes.
            for name, value in saved['state'].items():
                full['net.' + name] = value
            for name, value in saved['network_lifecycle'].items():
                full['net.' + name] = value
            full['_extra_state'] = saved['wrapper_extra']
            full['net._extra_state'] = saved['network_extra']
            full['net.prompt.e_prompt._extra_state'] = {'version': 1, **saved['prompt_lifecycle']}
            # The clone was reset to a single head; the native checkpoint loader
            # must reconstruct its later heads, lifecycle and generator freeze.
            wrapper.load_state_dict(full)
            return clones, wrapper
        prompt = clones['prompt']
        desired = saved['task']
        if next(prompt.parameters()).device.type == 'cpu':
            placement = reference.author_cpu_device()
        else:
            placement = nullcontext()
        with placement:
            while len(prompt.head.heads) < desired:
                prompt.head.update(prompt.num_classes // prompt.num_tasks)
        for key, value in saved['prompt_lifecycle'].items():
            setattr(prompt.e_prompt, key, copy.deepcopy(value))
        prompt.e_prompt.k_comp_gen.requires_grad_(desired < 2)
        prompt.e_prompt.v_comp_gen.requires_grad_(desired < 2)
        for branch, module in clones.items():
            selected = {key[len(branch) + 1:]: value for key, value in saved['state'].items() if key.startswith(branch + '.')}
            loaded = module.load_state_dict(selected, strict=False)
            if loaded.unexpected_keys:
                raise RuntimeError('PROL source clone could not restore its compact learner catalogue')
        clones['query'].requires_grad_(False)
        return clones, None


class ProlAudit:
    """Observe actual source/native forward, backward, scheduler and data streams."""
    def __init__(self, branches, dataset, recorder, record, torch, np, native=None):
        self.branches, self.dataset, self.recorder, self.record = branches, dataset, recorder, record
        self.torch, self.np, self.native = torch, np, native
        self.prompt = branches['prompt'].e_prompt
        self.stage, self.task, self.epoch = None, -1, {}
        self.handles, self.optimizers = [], WeakSet()
        self.handles.append(self.prompt.register_forward_hook(self.prompt_output, with_kwargs=True))
        self.task_old = {}
        self.initial_frozen = {f'{branch}.{name}': parameter for branch, module in branches.items()
                               for name, parameter in module.named_parameters() if not parameter.requires_grad}
        self.initial_frozen_digest = reference.state_digest(self.initial_frozen)
        self.initial_frozen_encoder = frozen_encoder_hashes(branches)
        self.record['initial_frozen_encoder'] = self.initial_frozen_encoder
        self.record['initial_frozen_parameters_sha256'] = self.initial_frozen_digest
        self.eval_predictions, self.eval_routes, self.eval_batches = [], [], []
        self.routing_layer_count = 0

    def frozen_finite(self):
        gradients = []
        for module in self.branches.values():
            for parameter in module.parameters():
                if not parameter.requires_grad and parameter.grad is not None:
                    # Source freezes generator tensors without clearing their
                    # last base-task grad. Adam.zero_grad clears these before
                    # the first task-two backward, when this check is applied.
                    raise RuntimeError('Frozen PROL tensor retained a gradient during an optimizer update')
                if parameter.grad is not None:
                    gradients.append(self.torch.isfinite(parameter.grad).all())
        if gradients and not bool(self.torch.stack(gradients).all()):
            raise FloatingPointError('PROL learned gradient is nonfinite')

    def old_state(self):
        state = {}
        for index in range(max(0, self.task)):
            for name, value in self.branches['prompt'].head.heads[index].state_dict().items():
                state[f'head.{index}.{name}'] = value.detach().cpu().clone()
        if self.task > 0:
            state.update({name: value.detach().cpu().clone() for name, value in self.prompt.state_dict().items()
                          if 'comp_gen' in name})
        return state

    def assert_old_unchanged(self):
        state = self.old_state()
        if state.keys() != self.task_old.keys() or any(not self.torch.equal(value, self.task_old[name]) for name, value in state.items()):
            raise RuntimeError('PROL frozen generator or previous classifier head changed')

    def snapshot(self, name):
        if self.native is not None:
            model = self.native
        else:
            model = self.torch.nn.Module()
            for key, value in self.branches.items():
                model.add_module(key, value)
            model.opt = self.optimizer
        self.recorder.snapshots[name] = model_snapshot(model)

    def load_task(self):
        from scripts.ease_reference_utils import dataset_fingerprint
        self.task = len(self.dataset.test_loaders) - 1
        if self.task not in range(self.record['protocol']['tasks']):
            raise RuntimeError('PROL loaded an unauthorized task')
        for split, loader in (('train', self.dataset.train_loader), ('test', self.dataset.test_loaders[-1])):
            raw = loader.dataset
            ids, labels = self.np.asarray(raw.indexes), self.np.asarray(raw.targets)
            expected = 5000 if split == 'train' else 1000
            if (len(labels) != expected or len(self.np.unique(ids)) != expected or not self.np.all(ids[1:] > ids[:-1])
                    or not self.np.all((labels >= self.task * 10) & (labels < (self.task + 1) * 10))):
                raise RuntimeError('PROL current-task split contains missing, future, replay or reordered samples')
            fingerprint = dataset_fingerprint(raw.data, labels, ids, raw.transform)
            if self.task == 0:
                from scripts.run_low_rank_efficiency import audit_fingerprints
                audit_fingerprints(fingerprint, self.record['common_reference_data_snapshots'][f'task1_{split}'],
                                   self.record['common_transform_snapshots'][f'task1_{split}'])
            self.recorder.data_snapshots[f'task{self.task + 1}_{split}'] = fingerprint
        self.task_old = self.old_state()
        self.record['structural_checks'].update(dataset_transforms_ids_match=True, no_future_or_replay_samples=True)

    def task_started(self):
        prompt = self.branches['prompt']
        task = self.task
        generators = [parameter for name, parameter in self.prompt.named_parameters() if 'comp_gen' in name]
        if (len(prompt.head.heads) != task + 1 or self.prompt.lmax_list != [10 * (index + 1) for index in range(task + 1)]
                or any(parameter.requires_grad != (task == 0) for parameter in generators)
                or any(parameter.requires_grad != (index == task) for index, head in enumerate(prompt.head.heads) for parameter in head.parameters())
                or any(parameter.requires_grad for parameter in self.branches['query'].parameters())):
            raise RuntimeError('PROL task head expansion, query freeze or generator freeze differs from the source')
        for name, parameter in prompt.named_parameters():
            if name.startswith(('blocks.', 'patch_embed.', 'cls_token', 'pos_embed', 'norm.', 'fc_norm.')):
                if parameter.requires_grad != reference.learned_name('prompt', name):
                    raise RuntimeError('PROL encoder trainable LayerNorm selection differs from the source')
        self.record['method_audits'].append({'task': task + 1, 'kind': 'task_begin',
            'head_count': len(prompt.head.heads), 'generator_trainable': task == 0,
            'previous_heads_frozen': True, 'old_key_change_allowed': True,
            'cpu_cuda_rng_after_boundary_sha256': rng_digest(self.torch), 'parameters': parameter_audit(self.branches)})
        self.record['structural_checks'].update(generator_frozen_from_task2=True, previous_classifier_heads_frozen=True,
                                               allowed_trainable_norms_match_source=True)

    def attach_optimizer(self, optimizer):
        if optimizer in self.optimizers:
            return optimizer
        prompt = self.branches['prompt']
        if self.task < 1:
            expected = [list(prompt.parameters())]
            rates = [.0025]
        else:
            normal, other = [], []
            for name, parameter in prompt.named_parameters():
                norm = 'norm1' in name or 'norm2' in name or name.startswith('norm') or 'fc_norm' in name
                (normal if norm else other).append(parameter)
            expected, rates = [other, normal], [.0025, 1.25e-6]
        actual = [group['params'] for group in optimizer.param_groups]
        if (len(actual) != len(expected) or any([id(value) for value in left] != [id(value) for value in right]
                                             for left, right in zip(actual, expected))
                or optimizer.state or [group['lr'] for group in optimizer.param_groups] != rates
                or any(group['weight_decay'] != 0. or group['betas'] != (.9, .999) or group['eps'] != 1e-8 for group in optimizer.param_groups)):
            raise RuntimeError('PROL source task-local Adam parameter order, rates or reset changed')
        self.optimizers.add(optimizer)
        self.optimizer = optimizer
        def before(subject, arguments, keywords):
            if self.stage != 'train':
                raise RuntimeError('PROL optimizer update occurred outside ordinary training')
            self.frozen_finite()
            self.lr_row = {'task': self.task + 1, 'epoch': 1, 'step': self.epoch['steps'] + 1,
                'pre_step_lrs': [group['lr'] for group in subject.param_groups],
                'loss_ce': self.loss_components['loss_ce'], 'threshold': .3,
                'threshold_triggered': self.reduce_lr}
        def after(subject, arguments, keywords):
            self.lr_row['post_optimizer_lrs'] = [group['lr'] for group in subject.param_groups]
            self.lr_row['post_inner_scheduler_lrs'] = list(self.lr_row['post_optimizer_lrs'])
            self.record['lr_audits'].append(self.lr_row)
            self.record['counts']['optimizer_updates'] += 1
            self.epoch['steps'] += 1
        self.handles.extend((optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)))
        return optimizer

    def begin_epoch(self, index):
        if index != 0:
            raise RuntimeError('PROL online recipe has exactly one epoch')
        seed = shared.seed_phase(1993, self.task, 'train', index)
        shared.configure_epoch_loader(self.dataset.train_loader, 1993, self.task, 'train', index)
        self.epoch = {'task': self.task + 1, 'epoch': 1, 'seed': seed, 'steps': 0, 'samples': 0, 'loss_sum': 0.}
        self.stage, self.reduce_lr = 'train', False
        self.stream_ids, self.stream_labels = [], []

    def observed(self, inputs, labels):
        self.inputs, self.labels = inputs, labels
        if bool(((labels < self.task * 10) | (labels >= (self.task + 1) * 10)).any()):
            raise RuntimeError('PROL training saw future or replay labels')
        self.epoch['samples'] += len(labels)
        self.record['counts']['observe_calls'] += 1
        self.stream_labels.extend(labels.detach().cpu().tolist())
        if self.epoch['steps'] == 0:
            self.first_rng = rng_digest(self.torch)

    def backward(self, original):
        @functools.wraps(original)
        def wrapped(loss, *args, **kwargs):
            if self.stage == 'train' and (kwargs.get('retain_graph', False) or not bool(self.torch.isfinite(loss).all())):
                raise RuntimeError('PROL active source backward must use finite loss without retain_graph')
            # Read only the actual numerical frame's scalars, never credentials
            # or unrelated stack locals. This avoids recomputing any loss math.
            frame = sys._getframe(1)
            if self.stage == 'train':
                if frame.f_code.co_name == 'observe':
                    components = frame.f_locals['components']
                    self.loss_components = {name: float(value.detach()) for name, value in components.items() if name != 'generalization_matrix'}
                elif frame.f_code.co_name == 'train_one_epoch_with_aux':
                    values = frame.f_locals
                    self.loss_components = {'loss_diagonal': float(values['Ldiag'].detach()),
                                            'loss_redundancy': float(values['Lrdn'].detach())}
                    # CE is captured directly at its threshold branch by the
                    # criterion observer below, so no inverse arithmetic drift.
                    self.loss_components['loss_ce'] = float(self.ce_values[0] * .001 + self.ce_values[1])
                else:
                    raise RuntimeError('PROL backward escaped the native observe or immutable author engine')
                self.reduce_lr = self.reduce_lr or self.loss_components['loss_ce'] < .3
            result = original(loss, *args, **kwargs)
            if self.stage == 'train':
                self.frozen_finite()
                self.epoch['loss_sum'] += float(loss.detach())
                if self.epoch['steps'] == 0:
                    gradients = {f'prompt.{name}': parameter.grad for name, parameter in self.branches['prompt'].named_parameters()
                                 if parameter.requires_grad and parameter.grad is not None}
                    self.record['first_batch_audits'].append({'task': self.task + 1, 'epoch': 1,
                        'seed': self.epoch['seed'], 'samples': len(self.labels), 'loss': float(loss.detach()),
                        **self.loss_components, 'optimizer_update_index': self.record['counts']['optimizer_updates'] + 1,
                        'input_sha256': reference.state_digest({'inputs': self.inputs}),
                        'labels_sha256': reference.state_digest({'labels': self.labels}),
                        'gradient_sha256': reference.state_digest(gradients), 'gradient_timing': 'before_clip',
                        'effective_lr': self.optimizer.param_groups[0]['lr'],
                        'effective_lrs': [group['lr'] for group in self.optimizer.param_groups],
                        'rng_sha256': self.first_rng})
                    path = self.recorder.output / f'task{self.task + 1}_first_gradients.npz'
                    self.np.savez_compressed(path, **{name: value.detach().cpu().numpy() for name, value in gradients.items()})
                    self.record['first_batch_audits'][-1].update(gradient_file=path.name,
                        gradient_file_sha256=shared.file_hash(path))
                    self.record['structural_checks']['first_actual_batch_inputs_gradients_lr_audited'] = True
            return result
        return wrapped

    def criterion(self, logits, labels):
        result = self.torch.nn.functional.cross_entropy(logits, labels)
        self.ce_values.append(result.detach())
        return result

    def scheduler_step(self, original):
        def wrapped(scheduler, *args, **kwargs):
            result = original(scheduler, *args, **kwargs)
            if scheduler.optimizer in self.optimizers and scheduler.last_epoch > 0:
                rates = [group['lr'] for group in scheduler.optimizer.param_groups]
                if scheduler.T_max == 20:
                    if not self.reduce_lr or not self.record['lr_audits'] or self.stage != 'train':
                        raise RuntimeError('PROL inner scheduler stepped before its sticky CE threshold')
                    self.record['lr_audits'][-1].update(post_inner_scheduler_lrs=rates,
                                                       inner_scheduler_step=scheduler.last_epoch)
                elif scheduler.T_max == 1:
                    self.record['method_audits'].append({'task': self.task + 1, 'kind': 'outer_scheduler',
                        'last_epoch': scheduler.last_epoch, 'effective_lrs': rates})
                    if any(rate != 0. for rate in rates):
                        raise RuntimeError('PROL epoch-end outer scheduler did not reach zero')
                else:
                    raise RuntimeError('PROL scheduler configuration changed')
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
        layer = keywords['layer_num']
        labels = arguments[1] if len(arguments) > 1 else keywords.get('y')
        indices = self.torch.topk(result['similarity'], k=1, dim=1).indices[:, 0]
        if self.stage == 'train' and self.epoch['steps'] == 0 and layer == self.record['protocol']['prompt_layers'][0]:
            unique = int(indices.unique().numel())
            self.record['method_audits'].append({'task': self.task + 1, 'kind': 'first_batch_routing',
                'cross_batch_similarity_assignment': True, 'samples': len(indices),
                'unique_index_count': unique, 'duplicate_index_count': len(indices) - unique,
                'selected_local_indices_sha256': reference.state_digest({'indices': indices}),
                'column0_selected_count': int((indices == 0).sum())})
        if self.stage == 'evaluation':
            if labels is not None or keywords.get('task_id', -1) != -1 or module.training:
                raise RuntimeError('PROL evaluation used a task or label oracle')
            if result['similarity'].shape != (len(indices), (self.task + 1) * 10):
                raise RuntimeError('PROL unknown-class routing considered future keys')
            if layer == self.record['protocol']['prompt_layers'][0]:
                self.routing_layer_count = 0
                self.eval_routes.append({'test_task': self.eval_test_task + 1, 'samples': len(indices),
                    'oracle_task_id': None, 'labels_provided': False, 'routing_key_count': (self.task + 1) * 10,
                    'selected_local_indices_sha256': reference.state_digest({'indices': indices}),
                    'unique_index_count': int(indices.unique().numel()),
                    'duplicate_index_count': len(indices) - int(indices.unique().numel()),
                    'column0_selected_count': int((indices == 0).sum())})
            self.routing_layer_count += 1
            for name, lower, upper in (('scaler', .999, 1.001), ('scaler_v', .999, 1.001),
                                       ('shifter', -.0001, .0001), ('shifter_v', -.0001, .0001)):
                parameter = getattr(module, name)
                if not bool(((parameter >= lower) & (parameter <= upper)).all()):
                    raise RuntimeError('PROL evaluation-time affine clamp differs from the source')

    def end_epoch(self):
        self.frozen_finite()
        if self.epoch['steps'] != 79 or self.epoch['samples'] != 5000 or len(self.stream_ids) != 5000 or len(self.stream_labels) != 5000:
            raise RuntimeError('PROL ordinary online stream is incomplete')
        if len(set(self.stream_ids)) != 5000 or set(self.stream_ids) != set(map(int, self.dataset.train_loader.dataset.indexes)):
            raise RuntimeError('PROL sampler repeated or omitted current-task IDs')
        self.assert_old_unchanged()
        rows = [row for row in self.record['lr_audits'] if row['task'] == self.task + 1]
        triggered = False
        for step, row in enumerate(rows, 1):
            triggered = triggered or row['loss_ce'] < .3
            if row['step'] != step or row['threshold_triggered'] != triggered or bool(row.get('inner_scheduler_step', False)) != triggered:
                raise RuntimeError('PROL sticky threshold or inner scheduler trace is incomplete')
        row = {**self.epoch, 'loss_mean': self.epoch['loss_sum'] / self.epoch['steps'],
               'inner_scheduler_steps': sum('inner_scheduler_step' in value for value in rows),
               'cpu_cuda_rng_after_epoch_sha256': rng_digest(self.torch)}
        self.record['epoch_results'].append(row)
        self.record['sample_stream_audits'].append({'task': self.task + 1, 'epoch': 1,
            'seed': self.epoch['seed'], 'samples': len(self.stream_ids),
            'global_ids_sha256': reference.state_digest({'ids': self.torch.tensor(self.stream_ids, dtype=self.torch.int64)}),
            'labels_sha256': reference.state_digest({'labels': self.torch.tensor(self.stream_labels, dtype=self.torch.int64)})})
        write_json(self.recorder.output / 'progress.json', row)
        self.record['structural_checks']['scheduler_trigger_and_all_groups_match'] = True
        self.stage = None

    def begin_evaluation(self):
        from torch.utils.data import SequentialSampler
        self.stage, self.eval_predictions, self.eval_routes, self.eval_batches = 'evaluation', [], [], []
        shared.seed_phase(1993, self.task, 'evaluation')
        for loader in self.dataset.test_loaders:
            if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last or loader.batch_size != 64:
                raise RuntimeError('PROL evaluation must keep complete separate task batches of64')
            shared.configure_epoch_loader(loader, 1993, self.task, 'evaluation')
        self.eval_test_task = 0

    def evaluation_output(self, inputs, logits):
        if logits.shape != (len(inputs), (self.task + 1) * 10) or not bool(self.torch.isfinite(logits).all()):
            raise RuntimeError('PROL evaluation returned invalid seen logits')
        if self.routing_layer_count != len(self.record['protocol']['prompt_layers']):
            raise RuntimeError('PROL evaluation did not route all active prefix layers')
        self.eval_predictions.append(logits.argmax(1).detach().cpu().numpy())
        self.eval_batches.append(len(inputs))
        self.last_probe = (inputs.detach().clone(), logits.detach().clone())
        if sum(self.eval_batches) % 1000 == 0:
            self.eval_test_task += 1

    def end_evaluation(self, accuracy=None):
        predicted = self.np.concatenate(self.eval_predictions)
        labels = self.np.concatenate([self.np.asarray(loader.dataset.targets) for loader in self.dataset.test_loaders])
        ids = self.np.concatenate([self.np.asarray(loader.dataset.indexes) for loader in self.dataset.test_loaders])
        if self.eval_batches != ([64] * 15 + [40]) * (self.task + 1) or len(predicted) != (self.task + 1) * 1000:
            raise RuntimeError('PROL evaluation omitted samples or merged task routing tails')
        counts = exact_counts(predicted, labels, self.task + 1)
        if accuracy is not None and abs(sum(map(float, accuracy[0])) / len(accuracy[0]) - counts['seen_accuracy_pct']) > 1e-9:
            raise RuntimeError('Native PROL accuracy differs from saved exact predictions')
        order = self.np.argsort(ids, kind='stable')
        path = self.recorder.output / f'task{self.task + 1}_predictions.npz'
        self.np.savez_compressed(path, predictions=predicted[order], labels=labels[order], global_ids=ids[order],
                                 evaluation_global_ids=ids, batch_sizes=self.np.asarray(self.eval_batches))
        self.record['prediction_files'].append({'task': self.task + 1, 'path': str(path), 'sha256': shared.file_hash(path)})
        self.record['task_results'].append(counts)
        self.record['evaluated_tasks_completed'] = self.task + 1
        self.record['accuracy'] = {'setting': 'class-il', 'mean_accuracy_pct': counts['seen_accuracy_pct']}
        self.record['routing_audits'].append({'task': self.task + 1,
            'evaluation_global_ids_sha256': reference.state_digest({'ids': self.torch.from_numpy(ids)}),
            'batches': self.eval_routes, 'all_active_prefix_layers_routed': True,
            'cross_batch_similarity_assignment': True, 'evaluation_affine_clamp_preserved': True})
        self.record['structural_checks'].update(seen_class_counts_and_predictions_match=True, source_routing_and_clamps_preserved=True)
        self.stage = None
        self.finalized()
        status = notify_slack(f'PROL {self.record["side"]} {self.record["mode"]} {self.record["pass"]}: task{self.task + 1} completed, seen={counts["seen_accuracy_pct"]:.2f}%; {self.record["wandb_url"]}')
        self.record.setdefault('task_notification_status', []).append(status)
        if status != 'sent':
            raise RuntimeError('Required PROL task notification failed')
        floor = 90 if self.task == 0 else 20
        if counts['seen_accuracy_pct'] < floor:
            self.recorder.tasks_completed = self.task + 1
            self.record['accuracy_floor_failure'] = {'task': self.task + 1, 'floor_pct': floor, 'seen_accuracy_pct': counts['seen_accuracy_pct']}
            raise RuntimeError('PROL accuracy below the approved absolute floor; later jobs are blocked')

    def finalized(self):
        proof = {'task': self.task + 1, 'checks': {}}
        self.record.setdefault('checkpoint_probe_audits', []).append(proof)
        def save_proof():
            proof['phase'] = self.recorder.active_phase
            write_json(self.recorder.output / 'checkpoint_probe_audits.json', self.record['checkpoint_probe_audits'])
        with self.recorder.phase('state_audit'):
            frozen = frozen_encoder_hashes(self.branches)
            proof['checks'].update(frozen_encoder_unchanged=frozen == self.initial_frozen_encoder,
                initial_frozen_parameters_unchanged=reference.state_digest(self.initial_frozen) == self.initial_frozen_digest)
            save_proof()
            if not all(proof['checks'].values()):
                raise RuntimeError('PROL frozen query or nonnorm encoder tensors changed')
            try:
                self.assert_old_unchanged()
                proof['checks']['old_heads_and_frozen_generator_unchanged'] = True
            except RuntimeError:
                proof['checks']['old_heads_and_frozen_generator_unchanged'] = False
                save_proof()
                raise
            state = reference.learned_state(self.branches)
            proof['checks']['learned_state_finite'] = all(bool(self.torch.isfinite(value).all()) for value in state.values())
            save_proof()
            if not proof['checks']['learned_state_finite']:
                raise FloatingPointError('Finalized PROL learned state is nonfinite')
            if self.native is not None and (int(self.native.net.active_task) != self.task
                    or not bool(self.native.net.task_finalized) or self.native.net.task_updates[:self.task + 1].tolist() != [79] * (self.task + 1)):
                raise RuntimeError('Finalized PROL Mammoth lifecycle counters are invalid')
            lifecycle = {name: copy.deepcopy(getattr(self.prompt, name)) for name in
                ('old_num_k', 'new_num_k', 'old_num_c', 'new_num_c', 'kmax_list', 'lmax_list')}
            proof.update(frozen_encoder=frozen, learned_sha256=reference.state_digest(state))
            proof['checks']['learned_state_finite'] = True
            save_proof()
        with self.recorder.phase('checkpoint'):
            path = self.recorder.output / f'task{self.task + 1}_state.pt'
            payload = {'schema_version': 1, 'state': state, 'learned_sha256': reference.state_digest(state),
                'task': self.task + 1, 'seen_classes': (self.task + 1) * 10, 'prompt_lifecycle': lifecycle,
                'frozen_encoder': frozen, 'protocol': self.record['protocol']}
            if self.native is not None:
                payload['network_lifecycle'] = {key: value.detach().cpu().clone() for key, value in self.native.net.state_dict().items()
                                               if key in ('active_task', 'task_offsets', 'task_finalized', 'task_updates')}
                payload['wrapper_extra'] = self.native.get_extra_state()
                payload['network_extra'] = self.native.net.get_extra_state()
            self.torch.save(payload, path)
            saved = self.torch.load(path, map_location='cpu', weights_only=True)
            proof['checks']['disk_catalogue_matches'] = reference.state_digest(saved['state']) == payload['learned_sha256'] and saved['protocol'] == payload['protocol']
            save_proof()
            if not proof['checks']['disk_catalogue_matches']:
                raise RuntimeError('PROL compact checkpoint changed on disk')
            proof['checks']['disk_catalogue_matches'] = True
            proof['checkpoint_sha256'] = shared.file_hash(path)
            save_proof()
            rng = rng_digest(self.torch)
            clones, native_clone = compact_clone(self.branches, saved, self.native)
            restored_digest = reference.state_digest(reference.learned_state(clones))
            proof['checks']['clone_catalogue_matches'] = restored_digest == payload['learned_sha256']
            proof.update(clone_learned_sha256=restored_digest, clone_rng_before_sha256=rng,
                         clone_rng_after_sha256=rng_digest(self.torch))
            proof['checks']['clone_rng_unchanged'] = proof['clone_rng_after_sha256'] == rng
            save_proof()
            if not proof['checks']['clone_catalogue_matches']:
                raise RuntimeError('PROL checkpoint clone did not restore every learned norm, prompt or head')
            if not proof['checks']['clone_rng_unchanged']:
                raise RuntimeError('PROL dynamic checkpoint restoration advanced experiment RNG')
        with self.recorder.phase('checkpoint_probe'), self.torch.no_grad():
            inputs, before = self.last_probe
            if native_clone is not None:
                after = native_clone.net.outputs(inputs, train=False)['logits'][:, :(self.task + 1) * 10]
            else:
                clones['query'].train(True)
                clones['prompt'].eval()
                features = clones['query'](inputs)['pre_logits']
                after = clones['prompt'].forwardA1(inputs, None, task_id=-1, cls_features=features, train=False)['logits'][:, :(self.task + 1) * 10]
            proof.update(checkpoint_probe_examples=len(inputs), input_sha256=reference.state_digest({'inputs': inputs}),
                probe_rng_before_sha256=rng, probe_rng_after_sha256=rng_digest(self.torch),
                before_logits_sha256=reference.state_digest({'logits': before}),
                after_logits_sha256=reference.state_digest({'logits': after}),
                max_absolute_logit_difference=float((before - after).abs().max()),
                predictions_changed=int((before.argmax(1) != after.argmax(1)).sum()))
            proof['checks'].update(probe_outputs_bitexact=self.torch.equal(before, after),
                                   probe_rng_unchanged=proof['probe_rng_after_sha256'] == rng)
            save_proof()
            if not proof['checks']['probe_outputs_bitexact'] or not proof['checks']['probe_rng_unchanged']:
                # Preserve each failure condition independently. Raw images are
                # identified by a digest; small logits/predictions are uploadable.
                self.np.savez_compressed(self.recorder.output / f'task{self.task + 1}_checkpoint_probe_failure.npz',
                    before_logits=before.detach().cpu().numpy(), after_logits=after.detach().cpu().numpy(),
                    before_predictions=before.argmax(1).detach().cpu().numpy(),
                    after_predictions=after.argmax(1).detach().cpu().numpy())
                raise RuntimeError('PROL checkpoint clone changed clamped outputs or RNG')
        del clones, native_clone
        self.record['state_audits'].append({'task': self.task + 1, 'frozen_encoder': frozen,
            'query_canonical_sha256': frozen['query']['sha256'], 'frozen_digest': self.initial_frozen_digest,
            'frozen_non_norm_unchanged': True, 'query_unchanged': True, 'generator_unchanged_after_freeze': True,
            'frozen_parameters_sha256': self.initial_frozen_digest,
            'learned_sha256': payload['learned_sha256'], 'checkpoint_sha256': shared.file_hash(path),
            'compact_checkpoint_bytes': path.stat().st_size, 'checkpoint_probe_examples': len(inputs),
            'checkpoint_outputs_bitexact': True, 'checkpoint_rng_unchanged': True,
            'checkpoint_dynamic_head_reset_restored': True, 'checkpoint_evaluation_clamp_preserved': True,
            'old_heads_unchanged': True, 'generator_frozen_from_task2': True, 'old_key_change_allowed': True,
            'task_local_optimizer_reset': True, 'allowed_trainable_norms_match_source': True,
            'phase_rng_epoch_seeds_verified': True, 'dynamic_head_count': self.task + 1})
        self.snapshot(f'after_task{self.task + 1}')
        self.record['structural_checks'].update(frozen_non_norm_encoder_unchanged=True, query_frozen_and_unchanged=True,
                                               checkpoint_round_trip=True)

    def install_common(self, stack):
        stack.enter_context(patch.object(self.torch.Tensor, 'backward', self.backward(self.torch.Tensor.backward)))
        sampler = self.torch.utils.data.RandomSampler
        stack.enter_context(patch.object(sampler, '__iter__', self.sampler_iter(sampler.__iter__)))
        scheduler = self.torch.optim.lr_scheduler.CosineAnnealingLR
        stack.enter_context(patch.object(scheduler, 'step', self.scheduler_step(scheduler.step)))
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
                self.record['method_audits'].append({'task': self.task + 1, 'kind': 'boundary_rng_before',
                    'cpu_cuda_rng_sha256': rng_digest(self.torch)})
                result = original(*args, **kwargs)
                self.task_started()
                self.stage = None
                return result
            return wrapped
        def optimizer(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                # Mammoth reloads an optimizer before begin_task updates the
                # network's task and grows its classifier. That discarded
                # prelude optimizer is outside the author's task Adam path.
                if int(self.native.net.active_task) != self.task:
                    return result
                return self.attach_optimizer(result)
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
                self.ce_values = []
                return original(inputs, labels, *args, **kwargs)
            return wrapped
        def end_epoch(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                self.end_epoch()
                return result
            return wrapped
        def end_task(original):
            def wrapped(*args, **kwargs):
                self.stage = 'task_end'
                shared.seed_phase(1993, self.task, 'task_end')
                result = original(*args, **kwargs)
                self.assert_old_unchanged()
                self.stage = None
                return result
            return wrapped
        def forward(original):
            def wrapped(inputs, *args, **kwargs):
                logits = original(inputs, *args, **kwargs)
                if self.stage == 'evaluation':
                    if args or kwargs:
                        raise RuntimeError('PROL native evaluation received oracle arguments')
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
                (self.native, 'meta_end_task', end_task), (self.native, 'forward', forward), (self.dataset, 'evaluate', evaluate)):
            stack.enter_context(patch.object(target, name, wrapper(getattr(target, name))))

    def close(self):
        for handle in self.handles:
            handle.remove()

def effective_args(args, dataset, spec):
    p = spec['protocol']
    expected = {'stop_after': p['tasks'], 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4,
        'seed': 1993, 'dataset_config': 'ease', 'permute_classes': 0, 'noise_rate': 0, 'validation': 0,
        'label_perc': 1, 'label_perc_by_class': 1, 'debug_mode': 0, 'enable_other_metrics': 0,
        'eval_future': 0, 'eval_epochs': None, 'fitting_mode': 'epochs', 'lr_scheduler': None,
        'joint': 0, 'code_optimization': 0, 'lr': .01, 'optim_wd': 0., 'optimizer': 'adam',
        'prol_vit_size': 'base', 'prol_prompt_length': 5, 'prol_num_tasks': 10,
        'prol_top_k_l': 1, 'prol_clip_grad': 1., 'prol_intertask_coeff': .001,
        'prol_pull_coeff': 1., 'prol_pull_coeff2': 1., 'prol_loss_threshold': .3,
        'prol_norm_lr_ratio': .0005, 'prol_cos_tmax': 20, 'prol_cos_eta_min': .0005}
    if any(getattr(args, key, None) != value for key, value in expected.items()) or (dataset.N_TASKS, dataset.N_CLASSES_PER_TASK, dataset.N_CLASSES) != (10, 10, 100):
        raise ValueError('Effective PROL arguments differ from the approved recipe')
    if list(args.prol_prompt_layers) != list(range(7)):
        raise ValueError('PROL active prefix layers changed')
    return vars(args)


def train_author(code, branches, dataset, audit, record):
    torch, p = audit.torch, record['protocol']
    device = next(branches['prompt'].parameters()).device
    source_args = reference.author_args(code, scale_lr=True, batch_size=64, lr=.01, epochs=1,
        print_freq=10000, num_tasks=10)
    class_mask = [list(range(task * 10, (task + 1) * 10)) for task in range(10)]
    for task in range(p['tasks']):
        shared.seed_phase(1993, task, 'task_begin')
        with audit.recorder.phase('data_loading'):
            dataset.get_data_loaders()
            audit.load_task()
        with audit.recorder.phase('begin_task'):
            audit.stage = 'task_begin'
            shared.seed_phase(1993, task, 'task_begin')
            record['method_audits'].append({'task': task + 1, 'kind': 'boundary_rng_before',
                'cpu_cuda_rng_sha256': rng_digest(torch)})
            optimizer, scheduler = reference.begin_author_task(code, branches, source_args, task, class_mask, task)
            audit.attach_optimizer(optimizer)
            audit.task_started()
            audit.stage = None
        audit.begin_epoch(0)
        def observed(inputs, labels):
            audit.observed(inputs, labels)
            audit.ce_values = []
            audit.recorder.steps += 1
            audit.recorder.training_examples += len(labels)
        with audit.recorder.phase('train_epoch'):
            reference.run_author_epoch(code, branches, reference.PairLoader(dataset.train_loader, observed),
                optimizer, source_args, task_id=task, class_mask=class_mask, old_num_k=task, epoch=0,
                criterion=audit.criterion)
        scheduler.step()
        audit.end_epoch()
        with audit.recorder.phase('end_task'):
            shared.seed_phase(1993, task, 'task_end')
            audit.assert_old_unchanged()
        with audit.recorder.phase('evaluation'), torch.no_grad():
            branches['query'].train(True)
            branches['prompt'].eval()
            audit.begin_evaluation()
            for loader in dataset.test_loaders:
                for data in loader:
                    inputs = data[0].to(device)
                    features = branches['query'](inputs)['pre_logits']
                    logits = branches['prompt'].forwardA1(inputs, None, task_id=-1, cls_features=features, train=False)['logits'][:, :(task + 1) * 10]
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
        raise RuntimeError('PROL complete workload counters differ from the approved scope')
    checks = record['structural_checks']
    checks.update(complete_stream_and_optimizer_updates=True, finite_loss_and_gradients=True,
                  frozen_non_norm_encoder_gradients_absent=True, initial_learned_state_and_phase_rng_match=True)
    if not all(checks.values()):
        raise RuntimeError('PROL did not establish every required structural check')


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
        'epoch_results.json', 'task_results.json', 'first_batch_audits.json', 'state_audits.json', 'method_audits.json',
        'routing_audits.json', 'sample_stream_audits.json', 'lr_audits.json', 'source_adaptations.json', 'telemetry.csv', 'phases.json', 'flops_ops.json')
    names += ('checkpoint_probe_audits.json', 'error_context.json')
    names += tuple(f'task{task}_predictions.npz' for task in range(1, record['protocol']['tasks'] + 1))
    names += tuple(f'task{task}_first_gradients.npz' for task in range(1, record['protocol']['tasks'] + 1))
    names += tuple(f'task{task}_checkpoint_probe_failure.npz' for task in range(1, record['protocol']['tasks'] + 1))
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
    if os.getenv('CL_PROL_EXECUTE') != '1' or os.getenv('CUDA_VISIBLE_DEVICES') != '1' or os.getenv('WANDB_MODE') != 'online':
        raise ValueError('PROL worker requires explicit execution on GPU1 with online W&B')
    configure_native_environment()
    spec = json.loads(Path(spec_path).read_text())
    shared.validate_spec(spec)
    if shared.execution_sources() != spec['execution_source_sha256']:
        raise ValueError('PROL execution sources changed after preparation')
    for path, digest in ((spec['checkpoint_file'], shared.CHECKPOINT_SHA256), (spec['initial_state'], spec['initial_state_sha256']),
                         (spec['common_native_summary'], spec['common_native_summary_sha256'])):
        if shared.file_hash(path) != digest:
            raise ValueError('PROL pinned input evidence changed')
    reference.verify_author_source(spec['reference_source'], spec['reference_source_sha256'])
    if validate_shared_data(spec['data_root']) != spec['data_files']:
        raise ValueError('Shared CIFAR100 raw data changed')
    output = Path(spec['output'])
    recorder = Recorder(output, '1')
    recorder.start()
    record = {**spec, 'status': 'failed', 'accuracy': {}, 'counts': {'observe_calls': 0, 'optimizer_updates': 0,
        'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}, 'epoch_results': [], 'task_results': [],
        'first_batch_audits': [], 'prediction_files': [], 'state_audits': [], 'method_audits': [], 'routing_audits': [], 'lr_audits': [],
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
                   group='prol-active-validation', job_type=f'{spec["side"]}-{spec["mode"]}-{spec["pass"]}', config=spec, dir=str(output))
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
                    branches = reference.build_author_branches(code, spec['checkpoint_file'])
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
                    raise ValueError('PROL canonical pretrained initialization differs from the pin')
            record.update(initial_learned_sha256=initial, initial_canonical_pretrained_sha256=shared.CANONICAL_PRETRAINED_SHA256,
                          initial_canonical_pretrained_branches=canonical, parameter_audits=parameter_audit(branches))
            expected_counts = {'trainable_parameters': 127098, 'frozen_parameters': 171575828,
                'total_parameters': 171702926, 'generator_parameters': 672, 'query_head_parameters': 16916,
                'prompt_head_count': 1}
            if record['parameter_audits'] != expected_counts:
                raise ValueError('PROL Base initial parameter count differs from the reviewed source')
            record['structural_checks'].update(canonical_pretrained_matches_pin=True, initial_learned_state_and_phase_rng_match=True)
            audit = ProlAudit(branches, dataset, recorder, record, torch, np, model)
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
                raise RuntimeError('Separate PROL profiler produced no positive supported operator estimate')
            record['flops'] = {'supported_ops_total': total, 'partial': True, 'includes_backward_only_if_supported': True,
                'scope': 'Supported operators across initialization, full training, evaluation, live audits and checkpoint probes in this separate pass',
                'limitations': 'Unsupported or fused operators, optimizer, CPU/PIL transforms and serialization may be omitted; no backward multiplier or phase attribution',
                'measurement_aggregate_eligible': False}
            write_json(output / 'flops_ops.json', operations)
        record.update(status='completed', computation_status='completed')
        exit_code = 0
    except BaseException as error:
        record.update(error_type=type(error).__name__, error='PROL worker failed; inspect local phase, count and structural audits', computation_status='failed')
        record['error_context'] = error_context(error, recorder, audit)
        write_json(output / 'error_context.json', record['error_context'])
        logging.error('PROL worker failed: %s', type(error).__name__)
    finally:
        if audit is not None:
            audit.close()
        recorder.tasks_completed = max(recorder.tasks_completed, record.get('evaluated_tasks_completed', 0))
        record = recorder.save(record)
        for name in ('epoch_results', 'task_results', 'first_batch_audits', 'state_audits', 'method_audits', 'routing_audits', 'sample_stream_audits', 'lr_audits'):
            write_json(output / (name + '.json'), record[name])
        write_json(output / 'source_adaptations.json', record.get('source_adaptations', ['Native Mammoth train() and meta wrappers; common phase RNG and read-only source-equivalence audits']))
        record['measurement_notes']['diagnostic_scope'] = 'Matched live audits, compact checkpoints and deterministic 40-example checkpoint probes are included in cost; probes are excluded from ordinary evaluation counts'
        record['measurement_notes']['tasks_completed_scope'] = 'Ordinary task evaluations completed, including a task later rejected by the accuracy floor; status separately records workload acceptance'
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
    if notify(f'PROL {spec["side"]} {spec["mode"]} {spec["pass"]} started: GPU1 batch64, official active path') != 'sent':
        raise RuntimeError('Required PROL start notification failed before training')
    env = os.environ.copy()
    env.pop('MAMMOTH_DEVICE', None)
    env.update(CUDA_VISIBLE_DEVICES='1', WANDB_MODE='online', WANDB_DIR=str(output), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', CL_PROL_EXECUTE='1')
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
                    notify(f'PROL {spec["side"]} {spec["mode"]} {spec["pass"]} still running; local audits preserved')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify(f'PROL {spec["side"]} interrupted; local results preserved')
            raise
    record = json.loads((output / 'result.json').read_text()) if (output / 'result.json').is_file() else {'status': 'failed'}
    status = 'timeout' if timed_out else record['status']
    if process.returncode != 0 and status == 'completed':
        status = 'failed'
    sent = notify(f'PROL {spec["side"]} {spec["mode"]} {spec["pass"]} {status}; {record.get("wandb_url", "W&B unavailable")}')
    write_json(output / 'completion.json', {'status': status, 'exit_code': process.returncode, 'timeout': timed_out,
        'wall_s': time.monotonic() - started, 'wandb_url': record.get('wandb_url'), 'slack_completion_status': sent})
    print(json.dumps({'status': status, 'result': str(output / 'result.json'), 'wandb_url': record.get('wandb_url')}))
    return 0 if process.returncode == 0 and status == 'completed' and sent == 'sent' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=('author', 'native'), default='native')
    parser.add_argument('--mode', choices=shared.MODES, default='validation')
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
