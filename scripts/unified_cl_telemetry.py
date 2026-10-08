"""Opt-in, task-preserving observations for the fresh CL comparison.

This module starts no sampler, process, GPU work or service connection. The
caller owns Recorder lifetime. Wrappers call the ordinary Mammoth meta hooks;
prediction capture uses the existing evaluation forwards, before task masks.
"""
from contextlib import contextmanager
import functools
import hashlib
import inspect
import math
import time
import weakref

import numpy as np
import torch
from torch.optim import Optimizer
import torch.optim.optimizer as optimizer_hooks

from scripts.cl_efficiency import integrate_power, prepare_loader_metadata


SCHEMA = 'unified-cl-telemetry-v1'
COUNTERS = ('outer_observe_calls', 'outer_current_image_visits',
            'optimizer_step_attempts', 'optimizer_steps', 'replay_image_visits',
            'feature_image_visits', 'feature_vector_visits', 'encoder_image_visits',
            'evaluation_image_visits')
_SKIP_ATTRIBUTES = {'args', 'dataset', 'train_loader', 'test_loaders', 'transform',
                    '_forward_hooks', '_forward_pre_hooks', '_backward_hooks',
                    '_state_dict_hooks', '_load_state_dict_pre_hooks',
                    '_load_state_dict_post_hooks', '_optimizer_step_pre_hooks',
                    '_optimizer_step_post_hooks'}


def union_intervals(intervals):
    """Disjoint union: inclusive nested phases must never be added twice."""
    merged = []
    for lower, upper in sorted(intervals):
        if not all(math.isfinite(v) for v in (lower, upper)) or upper < lower:
            raise ValueError('Invalid phase interval')
        if upper == lower:
            continue
        if merged and lower <= merged[-1][1]:
            merged[-1][1] = max(upper, merged[-1][1])
        else:
            merged.append([float(lower), float(upper)])
    return merged


def phase_cost(phases, samples, *, interval=1.):
    spans = union_intervals([(row['start_s'], row['end_s']) for row in phases])
    wall = sum(upper - lower for lower, upper in spans)
    energy, covered = integrate_power(samples, spans, max(6., interval * 3))
    return {'wall_s': wall, 'inclusive_calls': len(phases), 'union_intervals_s': spans,
            'gpu_board_energy_j': energy, 'gpu_energy_coverage_s': covered,
            'gpu_energy_coverage_fraction': covered / wall if wall else None}


def _storage(value):
    if torch.is_tensor(value):
        if value.layout != torch.strided:
            return None
        storage = value.untyped_storage()
        if not storage.nbytes():
            return None
        return (str(value.device), storage.data_ptr()), storage.nbytes()
    if isinstance(value, np.ndarray):
        base = value
        visited = set()
        while getattr(base, 'base', None) is not None and id(base) not in visited:
            visited.add(id(base))
            if torch.is_tensor(base.base):
                return _storage(base.base)
            if not isinstance(base.base, np.ndarray):
                break
            base = base.base
        if not base.nbytes or base.dtype.hasobject:
            return None
        return ('cpu', int(base.__array_interface__['data'][0])), int(base.nbytes)
    return None


def persistent_state_snapshot(model, *, optimizers=()):
    """Allocated tensor/NumPy storage, including unregistered CL state.

    Categories form a disjoint allocation accounting in priority order. Storage
    shared by views, teachers or NumPy/CPU tensors is charged once. Python object
    overhead and external dataset/cache files are explicitly outside the scope.
    Gradients have a separate allocation count and are not persistent state.
    """
    seen, objects, sizes = {}, set(), {}
    unavailable = set()

    def walk(value, category):
        allocation = _storage(value)
        if allocation is not None:
            key, size = allocation
            if key not in seen:
                seen[key] = category
                sizes[category] = sizes.get(category, 0) + size
            return
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            if value.numel() if torch.is_tensor(value) else value.size:
                unavailable.add(type(value).__name__)
            return
        if value is None or isinstance(value, (str, bytes, bool, int, float)) or callable(value) and not isinstance(value, torch.nn.Module):
            return
        if id(value) in objects:
            return
        objects.add(id(value))
        if isinstance(value, dict):
            for item in value.values():
                walk(item, category)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item, category)
        elif isinstance(value, Optimizer):
            walk(value.state, 'optimizer_state')
        elif hasattr(value, '__dict__'):
            # Dataset/sampler objects are not algorithm persistent model state.
            if isinstance(value, (torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.Sampler)):
                return
            for name, item in vars(value).items():
                if name not in _SKIP_ATTRIBUTES and not name.startswith('_unified_'):
                    walk(item, category)

    parameters = list(model.parameters())
    buffers = list(model.buffers())
    gradients = [p.grad for p in parameters if p.grad is not None]
    walk(parameters, 'registered_parameters')
    walk(buffers, 'registered_buffers')
    for opt in optimizers:
        walk(opt.state, 'optimizer_state')
    walk(model, 'auxiliary_unregistered')
    persistent_total = sum(sizes.values())
    before = persistent_total
    walk(gradients, 'gradients')
    return {'persistent_storage_bytes': persistent_total,
            'persistent_bytes_by_category': {key: sizes.get(key, 0) for key in
                ('registered_parameters', 'registered_buffers', 'optimizer_state', 'auxiliary_unregistered')},
            'additional_gradient_storage_bytes': sum(sizes.values()) - before,
            'unique_persistent_allocations': sum(category != 'gradients' for category in seen.values()),
            'unsupported_storage_types': sorted(unavailable),
            'scope': 'Unique allocated tensor and numeric NumPy storage reachable from model and observed optimizers; excludes dataset, Python overhead and files.'}


def registered_state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            # Extra-state can be heterogeneous; do not pretend it was hashed.
            return None
        value = value.detach().contiguous().cpu()
        digest.update(name.encode() + b'\0')
        digest.update(str(tuple(value.shape)).encode() + str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _owned_parameters(model):
    return {id(parameter): (name, parameter) for name, parameter in model.named_parameters()}


def _original_indexes(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        parent = _original_indexes(dataset.dataset)
        return None if parent is None else parent[np.asarray(dataset.indices, dtype=np.int64)]
    indexes = getattr(dataset, 'indexes', None)
    return None if indexes is None else np.asarray(indexes, dtype=np.int64)


class _PredictionLoader:
    def __init__(self, native, owner, evaluated_task):
        self.native, self.owner, self.evaluated_task = native, owner, evaluated_task

    def __len__(self):
        return len(self.native)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __iter__(self):
        offset, ordered_ids = 0, None
        sampler = getattr(self.native, 'sampler', None)
        indexes = _original_indexes(self.native.dataset)
        if indexes is not None:
            if isinstance(sampler, torch.utils.data.SequentialSampler):
                ordered_ids = indexes
            elif type(sampler).__module__ == 'scripts.unified_cl_inputs' and callable(getattr(sampler, 'indices', None)):
                ordered_ids = indexes[sampler.indices()]
        fields = list(getattr(self.native.dataset, 'extra_return_fields', ()))
        for batch in self.native:
            count = len(batch[1])
            ids = None
            if 'original_ids' in fields:
                ids = batch[2 + fields.index('original_ids')]
            elif ordered_ids is not None:
                ids = ordered_ids[offset:offset + count]
            self.owner._evaluation_batch = (self.evaluated_task, batch[1], ids)
            offset += count
            try:
                yield batch
            finally:
                self.owner._evaluation_batch = None


class UnifiedTelemetry:
    """A scoped observer, retaining task-keyed lifecycle, state and predictions.

    ``optimizer_steps`` counts completed real optimizer.step calls; gradient
    hooks distinguish received gradients from requires_grad and optimizer
    membership. Feature/replay counters stay None unless their actual events
    have been attached or explicitly supplied through ``add_counter``.
    """
    def __init__(self, recorder, model, dataset, training, *, canonical_digest=None,
                 loader_fingerprint=None):
        self.recorder, self.model, self.dataset, self.training = recorder, model, dataset, training
        self.canonical_digest, self.loader_fingerprint = canonical_digest, loader_fingerprint
        self.record = {'schema': SCHEMA, 'tasks': {}, 'initial': None,
                       'phase_timeline': [], 'counter_sources': {}, 'errors': []}
        self.task, self.epoch, self.phase = None, None, 'framework'
        # Holding an old optimizer alive would itself inflate measured memory.
        self.optimizers, self._handles, self._restored = weakref.WeakValueDictionary(), [], []
        self._available_counters = {}
        self._gradient_seen, self._stepped_with_gradient = {}, {}
        self._gradient_hook_ids = weakref.WeakValueDictionary()
        self._pending_steps = {}
        self._evaluation_batch, self._evaluation_record = None, None
        self._active = False

    def task_record(self, task=None):
        task = self.task if task is None else task
        if task is None:
            raise RuntimeError('A task must be active for task counters')
        key = f'task{int(task) + 1}'
        return self.record['tasks'].setdefault(key, {'task_id': int(task), 'data_snapshots': {},
            'model_snapshots': {}, 'lifecycle': [], 'counters': {name: None for name in COUNTERS},
            'optimizer_events': {}, 'evaluations': [], 'gradient_parameters': {},
            'stepped_gradient_parameters': {}})

    def add_counter(self, name, count, *, source):
        if name not in COUNTERS or not isinstance(count, int) or isinstance(count, bool) or count < 0 or not source:
            raise ValueError('Counter needs an approved name, nonnegative integer and event source')
        counters = self.task_record()['counters']
        counters[name] = (counters[name] or 0) + count
        sources = self.record['counter_sources'].setdefault(name, [])
        if source not in sources:
            sources.append(source)

    @contextmanager
    def phase_context(self, name):
        previous = self.phase
        self.phase = name
        start = time.perf_counter() - self.recorder.started
        task, epoch = self.task, self.epoch
        try:
            with self.recorder.phase(name):
                yield
        finally:
            end = time.perf_counter() - self.recorder.started
            spans = self.recorder.phases
            if spans and spans[-1]['phase'] == name:
                # Recorder owns the CUDA synchronization boundaries.
                start, end = spans[-1]['start_s'], spans[-1]['end_s']
            self.record['phase_timeline'].append({'phase': name, 'task_id': task,
                'epoch': epoch, 'start_s': start, 'end_s': end, 'wall_s': end - start})
            self.phase = previous

    def _parameters(self):
        parameters = _owned_parameters(self.model)
        # New task heads/pools can replace Parameter objects.
        attached = self._gradient_hook_ids
        if self._active:
            for identity, (name, parameter) in parameters.items():
                if parameter.requires_grad and attached.get(identity) is not parameter:
                    # A Tensor captured by its own hook creates a cycle and can
                    # retain a replaced CODA pool until GC, distorting peaks.
                    size = parameter.numel()
                    def received(gradient, name=name, size=size):
                        self._gradient_seen[name] = size
                        if self.task is not None:
                            self.task_record()['gradient_parameters'][name] = size
                        return gradient
                    self._handles.append(parameter.register_hook(received))
                    attached[identity] = parameter
        return parameters

    def snapshot(self):
        owned = self._parameters()
        for value in vars(self.model).values():
            if isinstance(value, Optimizer):
                self.optimizers[id(value)] = value
        optimizer_parameters = {id(p): p for opt in self.optimizers.values()
            for group in opt.param_groups for p in group['params'] if id(p) in owned}
        return {'parameters_total': sum(p.numel() for _, p in owned.values()),
                'parameters_requires_grad': sum(p.numel() for _, p in owned.values() if p.requires_grad),
                'parameters_in_observed_optimizers': sum(p.numel() for p in optimizer_parameters.values()),
                'parameters_receiving_gradient_so_far': sum(self._gradient_seen.values()),
                'parameters_stepped_with_gradient_so_far': sum(self._stepped_with_gradient.values()),
                'registered_state_sha256': registered_state_digest(self.model),
                'canonical_encoder_sha256': self.canonical_digest(self.model) if self.canonical_digest else None,
                **persistent_state_snapshot(self.model, optimizers=self.optimizers.values())}

    def _optimizer_pre(self, opt, args, kwargs):
        owned = self._parameters()
        selected = {id(p): p for group in opt.param_groups for p in group['params'] if id(p) in owned}
        if not selected or self.task is None:
            return
        self.optimizers[id(opt)] = opt
        self.add_counter('optimizer_step_attempts', 1, source='torch global optimizer step pre hook')
        event = self.task_record()['optimizer_events'].setdefault(self.phase, {'attempts': 0, 'completed': 0, 'groups': []})
        event['attempts'] += 1
        groups = [{'lr': float(group['lr']), 'weight_decay': float(group.get('weight_decay', 0)),
                   'parameter_numel': sum(p.numel() for p in group['params'] if id(p) in owned)} for group in opt.param_groups]
        if not event['groups'] or event['groups'][-1] != groups:
            event['groups'].append(groups)
        self._pending_steps[id(opt)] = {identity: (owned[identity][0], p.numel())
            for identity, p in selected.items() if p.grad is not None}

    def _optimizer_post(self, opt, args, kwargs):
        if id(opt) not in self._pending_steps or self.task is None:
            return
        self.add_counter('optimizer_steps', 1, source='torch global optimizer step post hook')
        self.task_record()['optimizer_events'][self.phase]['completed'] += 1
        for identity, (name, size) in self._pending_steps.pop(id(opt), {}).items():
            self._stepped_with_gradient[name] = size
            self.task_record()['stepped_gradient_parameters'][name] = size

    def watch_module(self, module, *, counter='encoder_image_visits', role='encoder'):
        """Attach once at an actual encoder root; never infer hidden passes."""
        if not self._active:
            raise RuntimeError('Module observations must be scoped to the active context')
        if counter not in COUNTERS:
            raise ValueError('Unknown module counter')
        source = f'forward pre hook:{role}'
        self._available_counters[counter] = source
        if self.task is not None:
            self.add_counter(counter, 0, source=source)
        def before(module, args):
            # Internal phases can unfreeze a head just before its first forward.
            self._parameters()
            if args and torch.is_tensor(args[0]) and self.task is not None:
                self.add_counter(counter, len(args[0]), source=f'forward pre hook:{role}')
        self._handles.append(module.register_forward_pre_hook(before))

    def watch_replay_buffer(self, buffer):
        """Count actual get_data returns; augmentation/selection remain original."""
        self._available_counters['replay_image_visits'] = 'buffer.get_data return'
        if self.task is not None:
            self.add_counter('replay_image_visits', 0, source='buffer.get_data return')
        def decorate(original):
            signature = inspect.signature(original)
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                index = 1 if bound.arguments.get('return_index', False) else 0
                self.add_counter('replay_image_visits', len(result[index]), source='buffer.get_data return')
                return result
            return wrapped
        self._patch(buffer, 'get_data', decorate)

    def _patch(self, target, name, decorate):
        original, owned = getattr(target, name), name in vars(target)
        self._restored.append((target, name, original, owned))
        setattr(target, name, decorate(original))

    def _timed(self, name, boundary=None):
        def decorate(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                if name == 'begin_task':
                    self.task = int(getattr(self.model, 'current_task', len(self.record['tasks'])))
                    self.epoch = None
                    self.task_record()
                    for key in ('outer_observe_calls', 'outer_current_image_visits', 'optimizer_step_attempts', 'optimizer_steps'):
                        self.add_counter(key, 0, source='native lifecycle initialization')
                    for key, source in self._available_counters.items():
                        self.add_counter(key, 0, source=source)
                if name == 'begin_epoch':
                    self.epoch = int(args[0] if args else kwargs['epoch'])
                with self.phase_context(name):
                    result = original(*args, **kwargs)
                if boundary and self.task is not None:
                    self.task_record()['model_snapshots'][boundary] = self.snapshot()
                    self.recorder.snapshots[f'task{self.task+1}/{boundary}'] = self.task_record()['model_snapshots'][boundary]
                if self.task is not None:
                    self.task_record()['lifecycle'].append({'event': name, 'epoch': self.epoch})
                return result
            return wrapped
        return decorate

    def _loaders(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            self.task = int(getattr(self.model, 'current_task', len(self.record['tasks'])))
            self.epoch = None
            with self.phase_context('data_loading'):
                result = original(*args, **kwargs)
            task = int(getattr(self.model, 'current_task', len(self.record['tasks'])))
            metadata = {name: (self.loader_fingerprint(loader) if self.loader_fingerprint else prepare_loader_metadata(loader))
                        for name, loader in zip(('train', 'test'), result)}
            self.task_record(task)['data_snapshots']['loaders'] = metadata
            self.recorder.data_snapshots[f'task{task+1}/loaders'] = metadata
            return result
        return wrapped

    def _observed(self, original):
        @functools.wraps(original)
        def wrapped(inputs, *args, **kwargs):
            self._parameters()
            result = original(inputs, *args, **kwargs)
            self.add_counter('outer_observe_calls', 1, source='native meta_observe return')
            self.add_counter('outer_current_image_visits', len(inputs), source='native meta_observe inputs')
            self.recorder.steps += 1
            self.recorder.training_examples += len(inputs)
            return result
        return wrapped

    def _forward(self, original):
        @functools.wraps(original)
        def wrapped(inputs, *args, **kwargs):
            output = original(inputs, *args, **kwargs)
            if self._evaluation_record is not None and self._evaluation_batch is not None:
                evaluated_task, labels, ids = self._evaluation_batch
                classes = self._evaluation_record['seen_classes']
                if not torch.is_tensor(output) or output.ndim != 2 or len(output) != len(labels) or output.shape[1] < classes:
                    raise ValueError('Unexpected native evaluation output')
                prediction = output[:, :classes].detach().argmax(1).cpu().tolist()
                row = self._evaluation_record['predictions'].setdefault(str(evaluated_task), {'labels': [], 'predictions': [], 'original_ids': []})
                row['labels'].extend(torch.as_tensor(labels).cpu().tolist())
                row['predictions'].extend(prediction)
                if ids is None:
                    row['original_ids'] = None
                elif row['original_ids'] is not None:
                    row['original_ids'].extend(torch.as_tensor(ids).cpu().tolist())
                self.add_counter('evaluation_image_visits', len(labels), source='ordinary evaluation forward')
                self.recorder.evaluation_examples += len(labels)
            return output
        return wrapped

    def _evaluated(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            native = self.dataset.test_loaders
            row = {'seen_classes': int(self.dataset.get_offsets()[1]), 'predictions': {}, 'complete': False, 'accuracy': None}
            self._evaluation_record = row
            self.dataset.test_loaders = [_PredictionLoader(loader, self, task) for task, loader in enumerate(native)]
            self.add_counter('evaluation_image_visits', 0, source='ordinary evaluation forward')
            try:
                with self.phase_context('evaluation'):
                    result = original(*args, **kwargs)
                row['accuracy'] = [[float(value) for value in result[0]], [float(value) for value in result[1]]]
                row['complete'] = True
                self._prediction_metrics(row)
                return result
            finally:
                self.dataset.test_loaders = native
                self._evaluation_batch, self._evaluation_record = None, None
                self.task_record()['evaluations'].append(row)
        return wrapped

    def _prediction_metrics(self, row):
        predictions = row['predictions']
        def accuracy(keys):
            pairs = [(a, b) for key in keys for a, b in zip(predictions[key]['labels'], predictions[key]['predictions'])]
            return 100 * sum(a == b for a, b in pairs) / len(pairs) if pairs else None
        keys = sorted(predictions, key=int)
        row['metrics'] = {'seen_accuracy_pct': accuracy(keys),
                          'old_accuracy_pct': accuracy([key for key in keys if int(key) < self.task]),
                          'new_accuracy_pct': accuracy([str(self.task)]) if str(self.task) in predictions else None,
                          'examples': sum(len(predictions[key]['labels']) for key in keys)}

    @contextmanager
    def context(self):
        if self._active:
            raise RuntimeError('Telemetry context cannot be nested')
        self._active = True
        try:
            self._handles.extend([optimizer_hooks.register_optimizer_step_pre_hook(self._optimizer_pre),
                                  optimizer_hooks.register_optimizer_step_post_hook(self._optimizer_post)])
            for value in vars(self.model).values():
                if isinstance(value, Optimizer):
                    self.optimizers[id(value)] = value
            self.record['initial'] = self.snapshot()
            self._patch(self.dataset, 'get_data_loaders', self._loaders)
            self._patch(self.dataset, 'evaluate', self._evaluated)
            for name, phase, boundary in (
                ('meta_begin_task', 'begin_task', 'after_begin_task'),
                ('meta_end_task', 'end_task', 'after_end_task'),
                ('meta_end_eval', 'end_eval', 'after_end_eval'),
                ('meta_begin_epoch', 'begin_epoch', None),
                ('meta_end_epoch', 'end_epoch', None)):
                self._patch(self.model, name, self._timed(phase, boundary))
            self._patch(self.model, 'meta_observe', self._observed)
            self._patch(self.model, 'forward', self._forward)
            self._patch(self.training, 'train_single_epoch', self._timed('train_epoch'))
            yield self
        finally:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            self._gradient_hook_ids.clear()
            for target, name, original, owned in reversed(self._restored):
                if owned:
                    setattr(target, name, original)
                else:
                    delattr(target, name)
            self._restored.clear()
            self._pending_steps.clear()
            self._active = False

    def finalize(self):
        """Summarize only observed spans; raw inclusive timeline remains intact."""
        timeline, samples = self.record['phase_timeline'], self.recorder.samples
        self.record['cost'] = phase_cost(timeline, samples, interval=self.recorder.interval)
        self.record['phase_costs'] = {name: phase_cost([row for row in timeline if row['phase'] == name], samples,
                                                      interval=self.recorder.interval)
                                     for name in sorted({row['phase'] for row in timeline})}
        for row in self.record['tasks'].values():
            row['cost'] = phase_cost([phase for phase in timeline if phase['task_id'] == row['task_id']], samples,
                                      interval=self.recorder.interval)
            row['parameters_receiving_gradient'] = sum(row['gradient_parameters'].values())
            row['parameters_stepped_with_gradient'] = sum(row['stepped_gradient_parameters'].values())
        self.record['notes'] = {'cost_scope': 'Union of observed framework spans; whole-process Recorder cost is separate and includes initialization/audits.',
            'phase_scope': 'Phase costs are inclusive. Do not sum different phase groups.',
            'missing_counters': 'None means no attached observable event, never an estimated zero.',
            'optimizer_scope': 'Completed calls and gradient-bearing parameter participation, not proof every value changed.',
            'parameter_gradient_scope': 'Unique registered parameter names, using the latest observed numel for replaced parameters; task-keyed catalogs retained.',
            'prediction_scope': 'Existing class-IL forwards before native task masking; no extra inference.',
            'measurement_overhead': 'Task snapshots/hash copies, gradient/module hooks and prediction CPU copies are included in whole observed cost.'}
        return self.record
