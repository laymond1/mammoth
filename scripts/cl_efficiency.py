"""Opt-in telemetry for the CL benchmark. Importing this module starts no work."""

from __future__ import annotations

import ast
import csv
import functools
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item
    return result


def inventory(root):
    """Inspect source without importing models, downloading weights, or using a GPU."""
    result = []
    for path in sorted((Path(root) / 'models').glob('*.py')):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields = {}
            for assignment in node.body:
                if isinstance(assignment, ast.Assign):
                    for target in assignment.targets:
                        if isinstance(target, ast.Name) and target.id in ('NAME', 'COMPATIBILITY'):
                            fields[target.id] = ast.literal_eval(assignment.value)
            if 'NAME' not in fields:
                continue
            result.append({
                'model': fields['NAME'].replace('_', '-'), 'source': str(path.relative_to(root)),
                'line': node.lineno, 'class': node.name, 'compatibility': fields.get('COMPATIBILITY', []),
                'config_exists': (Path(root) / 'models' / 'config' / (path.stem + '.yaml')).exists(),
                'status': 'source_present_runtime_unverified',
            })
    return result


def query_gpu(gpu):
    completed = subprocess.run([
        'nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,name,memory.used,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, timeout=5, check=True)
    rows = list(csv.reader(completed.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError('Expected exactly one physical GPU')
    uuid, name, memory, utilization, power = [item.strip() for item in rows[0]]

    def number(value):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) and parsed >= 0 else None
        except ValueError:
            return None

    return {'gpu_uuid': uuid, 'gpu_name': name, 'gpu_device_memory_mib': number(memory),
            'gpu_utilization_percent': number(utilization), 'gpu_power_w': number(power)}


def readiness(gpu, wandb_mode, slack):
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ('torch', 'torchvision', 'timm', 'wandb', 'psutil', 'open_clip', 'clip', 'kornia', 'xitorch', 'sklearn')}
    try:
        device = query_gpu(gpu) if gpu != 'cpu' else {'gpu': 'cpu', 'gpu_power_w': None}
        gpu_error = None
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        device, gpu_error = None, type(error).__name__
    return {
        'python': platform.python_version(), 'packages': packages, 'device': device,
        'gpu_error': gpu_error, 'wandb_mode': wandb_mode,
        'wandb_api_key_in_environment': bool(os.getenv('WANDB_API_KEY')),
        'wandb_auth_note': 'Cached SDK login is also supported; no network authentication was attempted.',
        'wandb_entity_configured': bool(os.getenv('WANDB_ENTITY')),
        'slack_requested': slack, 'slack_webhook_configured': bool(os.getenv('SLACK_WEBHOOK_URL')),
        'slack_webhook_valid': valid_slack_webhook(),
        'experiment_executed': False,
    }


def valid_slack_webhook():
    try:
        parsed = urlparse(os.getenv('SLACK_WEBHOOK_URL', ''))
    except ValueError:
        return False
    return (parsed.scheme == 'https' and parsed.hostname in ('hooks.slack.com', 'hooks.slack-gov.com')
            and not parsed.username and not parsed.password and parsed.path.startswith('/services/'))


def notify_slack(message, enabled=True):
    """Never put the secret webhook URL or HTTP error body in logs/results."""
    if not enabled:
        return 'disabled'
    webhook = os.getenv('SLACK_WEBHOOK_URL', '')
    if not valid_slack_webhook():
        return 'missing_or_invalid_webhook'
    request = Request(webhook, data=json.dumps({'text': message}).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=10) as response:
            return 'sent' if response.status == 200 else 'http_error'
    except Exception as error:
        return f'failed:{type(error).__name__}'


def capture_accuracy(logger, compatibility=None):
    """Read the native logger in memory; its repr logs may contain NumPy constructors."""
    result = {'setting': logger.setting}
    for name, key in (('accs', 'mean_accuracy_pct'), ('accs_mask_classes', 'task_il_mean_accuracy_pct')):
        values = getattr(logger, name, [])
        if len(values):
            value = float(values[-1])
            result[key] = value if math.isfinite(value) else None
    for name in ('fullaccs', 'fullaccs_mask_classes'):
        values = getattr(logger, name, [])
        if len(values):
            result[name] = [float(value) if math.isfinite(float(value)) else None for value in values[-1]]
    if (logger.setting == 'class-il' and compatibility is not None
            and not {'class-il', 'general-continual'}.intersection(compatibility)):
        # Mammoth returns a zero placeholder here for task-only methods such as PNN.
        result['mean_accuracy_pct'] = None
        result['fullaccs'] = None
        result['class_il_status'] = 'unsupported_by_model'
    return result


def prepare_loader_metadata(loader, clean_true_labels=False):
    """Use the native wrapper's extra-field API without changing samples or labels."""
    dataset = getattr(loader, 'dataset', None)
    if dataset is None or not hasattr(dataset, 'targets'):
        return {}
    if clean_true_labels:
        if 'true_labels' in dataset.extra_return_fields:
            raise ValueError('Clean-label benchmark must not inherit noisy true_labels')
        dataset.add_extra_return_field('true_labels', dataset.targets.copy())
    labels = [int(value) for value in dataset.targets]
    indices = [int(value) for value in dataset.indexes]
    pairs = sorted(zip(indices, labels))
    return {
        'samples': len(labels), 'classes': sorted(set(labels)),
        'indices_labels_sha256': hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest(),
        'extra_return_fields': list(dataset.extra_return_fields),
        'transform': repr(getattr(dataset, 'transform', None)),
        'clean_true_labels_added': clean_true_labels,
    }


def integrate_power(samples, intervals, max_gap):
    """Integrate only observed, finite power; gaps and missing sensors remain missing."""
    energy, coverage = 0., 0.
    for previous, current in zip(samples, samples[1:]):
        start, end = previous['elapsed_s'], current['elapsed_s']
        p0, p1 = previous.get('gpu_power_w'), current.get('gpu_power_w')
        if end <= start or end - start > max_gap or p0 is None or p1 is None:
            continue
        if not all(math.isfinite(p) and p >= 0 for p in (p0, p1)):
            continue
        for lower, upper in intervals:
            left, right = max(start, lower), min(end, upper)
            if right <= left:
                continue
            slope = (p1 - p0) / (end - start)
            energy += (p0 + slope * ((left + right) / 2 - start)) * (right - left)
            coverage += right - left
    return (energy if coverage > 0 else None), coverage


def tensor_bytes(value, seen=None):
    """Count tensor storage once, including allocated (rather than filled) buffer slots."""
    seen = set() if seen is None else seen
    if hasattr(value, 'untyped_storage'):
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(tensor_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v, seen) for v in value)
    return 0


def model_snapshot(model):
    parameters = list(model.parameters())
    trainable = [p for p in parameters if p.requires_grad]
    optimizers, states, optimizer_params = [], [], {}
    for value in vars(model).values():
        if hasattr(value, 'param_groups') and hasattr(value, 'state') and id(value) not in optimizers:
            optimizers.append(id(value))
            states.append(value.state)
            for group in value.param_groups:
                optimizer_params.update({id(p): p for p in group['params']})
    replay = {name: tensor_bytes(vars(value)) for name, value in vars(model).items()
              if 'buffer' in name.lower() and hasattr(value, '__dict__')}
    auxiliary = {name: value for name, value in vars(model).items()
                 if hasattr(value, 'untyped_storage') or isinstance(value, (dict, tuple, list))}
    return {
        'parameters_total': sum(p.numel() for p in parameters),
        'parameters_trainable': sum(p.numel() for p in trainable),
        'parameters_in_optimizer': sum(p.numel() for p in optimizer_params.values()),
        'parameter_storage_bytes': tensor_bytes(parameters),
        'registered_buffer_storage_bytes': tensor_bytes(list(model.buffers())),
        'optimizer_state_storage_bytes': tensor_bytes(states),
        'replay_allocated_tensor_bytes': sum(replay.values()), 'replay_buffers': replay,
        'auxiliary_tensor_bytes': tensor_bytes(auxiliary),
    }


class Recorder:
    """Per-process monitor plus synchronized phase timing. No extra training forwards."""

    def __init__(self, output, gpu='0', interval=1.):
        self.output, self.gpu, self.interval = Path(output), gpu, interval
        self.samples, self.phases, self.snapshots, self.data_snapshots = [], [], {}, {}
        self.active_phase = 'framework'
        self.torch, self.device, self.psutil_process = None, None, None
        self.training_examples, self.evaluation_examples, self.steps, self.tasks_completed = 0, 0, 0, 0
        self.errors = set()
        self.stop_event = threading.Event()
        self.started = time.perf_counter()
        self.stopped = None
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            import psutil
            self.psutil_process = psutil.Process()
        except ImportError:
            self.errors.add('psutil_missing_rss_unavailable')

    def sample(self):
        sample_start = time.perf_counter()
        row = {'phase': self.active_phase}
        if self.psutil_process is not None:
            try:
                processes = [self.psutil_process] + self.psutil_process.children(recursive=True)
                row['process_tree_rss_mib'] = sum(p.memory_info().rss for p in processes if p.is_running()) / 2 ** 20
            except Exception:
                self.errors.add('rss_sample_failed')
        if self.gpu != 'cpu':
            try:
                row.update(query_gpu(self.gpu))
            except (OSError, ValueError, subprocess.SubprocessError):
                self.errors.add('gpu_sensor_unavailable')
        row['elapsed_s'] = (sample_start + time.perf_counter()) / 2 - self.started
        self.samples.append(row)

    def start(self):
        def poll():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.sample()
        self.thread = threading.Thread(target=poll, name='cl-efficiency-sampler', daemon=True)
        self.thread.start()

    def stop(self):
        if self.stopped is not None:
            return
        self.stop_event.set()
        self.thread.join(timeout=6)
        self.sample()
        self.stopped = time.perf_counter() - self.started

    def synchronize(self):
        if self.torch is not None and self.device is not None and self.device.type == 'cuda' and self.torch.cuda.is_initialized():
            self.torch.cuda.synchronize(self.device)

    def cuda_memory(self):
        if self.torch is None or self.device is None or self.device.type != 'cuda' or not self.torch.cuda.is_initialized():
            return {'cuda_peak_allocated_mib': None, 'cuda_peak_reserved_mib': None}
        try:
            return {'cuda_peak_allocated_mib': self.torch.cuda.max_memory_allocated(self.device) / 2 ** 20,
                    'cuda_peak_reserved_mib': self.torch.cuda.max_memory_reserved(self.device) / 2 ** 20}
        except RuntimeError:
            self.errors.add('cuda_allocator_stats_unavailable')
            return {'cuda_peak_allocated_mib': None, 'cuda_peak_reserved_mib': None}

    @contextmanager
    def phase(self, name):
        self.synchronize()
        previous = self.active_phase
        self.active_phase = name
        start = time.perf_counter() - self.started
        annotation = self.torch.profiler.record_function(f'efficiency/{name}') if self.torch is not None else nullcontext()
        try:
            with annotation:
                yield
        finally:
            self.synchronize()
            end = time.perf_counter() - self.started
            self.phases.append({'phase': name, 'start_s': start, 'end_s': end, 'wall_s': end - start})
            self.active_phase = previous

    def save(self, metadata):
        self.stop()
        metrics = {'process_wall_s': self.stopped, 'outer_training_stream_examples': self.training_examples,
                   'evaluation_examples': self.evaluation_examples, 'outer_observe_steps': self.steps,
                   'tasks_completed': self.tasks_completed, **self.cuda_memory()}
        for key in ('process_tree_rss_mib', 'gpu_device_memory_mib', 'gpu_utilization_percent', 'gpu_power_w'):
            values = [s[key] for s in self.samples if s.get(key) is not None]
            metrics[f'peak_{key}'] = max(values) if values else None
        energy, coverage = integrate_power(self.samples, [(0., self.stopped)], max(6., self.interval * 3))
        metrics.update(gpu_board_energy_j=energy, gpu_energy_coverage_s=coverage,
                       gpu_energy_coverage_fraction=coverage / self.stopped if self.stopped else 0.,
                       gpu_board_energy_wh=energy / 3600 if energy is not None else None)
        phases = {}
        for name in sorted({p['phase'] for p in self.phases}):
            spans = [(p['start_s'], p['end_s']) for p in self.phases if p['phase'] == name]
            duration = sum(end - start for start, end in spans)
            joules, covered = integrate_power(self.samples, spans, max(6., self.interval * 3))
            phases[name] = {'wall_s': duration, 'calls': len(spans), 'gpu_board_energy_j': joules,
                            'gpu_energy_coverage_s': covered}
        # Inclusive phases may nest: do not sum them to estimate process wall time/energy.
        training = phases.get('train_epoch', {}).get('wall_s', 0.)
        evaluation = phases.get('evaluation', {}).get('wall_s', 0.)
        metrics['outer_training_examples_per_s'] = self.training_examples / training if training else None
        metrics['evaluation_examples_per_s'] = self.evaluation_examples / evaluation if evaluation else None
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        metrics['process_tree_cpu_s'] = cpu.ru_utime + cpu.ru_stime + children.ru_utime + children.ru_stime
        record = {**metadata, 'metrics': metrics, 'phases': phases, 'model_snapshots': self.snapshots,
                  'data_snapshots': self.data_snapshots,
                  'measurement_notes': {
                      'gpu_energy_scope': 'Selected physical GPU board; includes other users and idle power, excludes CPU/RAM/wall power.',
                      'gpu_energy_method': 'Trapezoidal integration of sampled nvidia-smi power; no extrapolation across missing samples.',
                      'phase_times': 'CUDA synchronized at phase boundaries; inclusive when phases nest.',
                      'cuda_memory_scope': 'PyTorch allocator high water since process startup, including initialization; excludes non-PyTorch allocations.',
                      'sampled_rss_scope': 'Sum of process and live descendant RSS; shared pages can be counted more than once.',
                      'example_count_scope': 'Outer meta_observe inputs only; excludes replay and hidden training inside task hooks.',
                      'snapshots_scope': 'Registered parameters/buffers plus explicitly reported auxiliary and replay tensors; categories can overlap.',
                  }, 'sensor_errors': sorted(self.errors)}
        write_json(self.output / 'metrics.json', record)
        write_json(self.output / 'phases.json', self.phases)
        fields = sorted({key for sample in self.samples for key in sample})
        with (self.output / 'telemetry.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)
        return record


@contextmanager
def instrument(recorder, model, dataset, training, clean_true_labels=False):
    """Wrap, and always call, the framework meta hooks; restore every patched method."""
    restored = []

    def patch(target, name, wrapper):
        original = getattr(target, name)
        owned = name in vars(target)
        restored.append((target, name, original, owned))
        setattr(target, name, wrapper(original))

    def timed(name, snapshot=None):
        def decorate(original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                with recorder.phase(name):
                    result = original(*args, **kwargs)
                if snapshot:
                    recorder.snapshots[snapshot] = model_snapshot(model)
                    if hasattr(dataset, 'train_loader') and dataset.train_loader is not None:
                        recorder.data_snapshots[snapshot] = {
                            'train_transform': repr(dataset.train_loader.dataset.transform),
                            'test_transform': repr(dataset.test_loaders[-1].dataset.transform),
                        }
                return result
            return wrapper
        return decorate

    def loaders(original):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            with recorder.phase('data_loading'):
                train, test = original(*args, **kwargs)
                recorder.data_snapshots['first_task'] = {
                    'train': prepare_loader_metadata(train, clean_true_labels),
                    'test': prepare_loader_metadata(test),
                }
                return train, test
        return wrapper

    def observed(original):
        @functools.wraps(original)
        def wrapper(inputs, *args, **kwargs):
            result = original(inputs, *args, **kwargs)
            # Count only the ordinary epoch loop, not calls nested in begin/end_task.
            if recorder.active_phase == 'train_epoch':
                recorder.training_examples += len(inputs)
                recorder.steps += 1
            return result
        return wrapper

    def forwarded(original):
        @functools.wraps(original)
        def wrapper(inputs, *args, **kwargs):
            result = original(inputs, *args, **kwargs)
            if recorder.active_phase == 'evaluation':
                recorder.evaluation_examples += len(inputs)
            return result
        return wrapper

    def evaluated(original):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            recorder.tasks_completed += 1
            return result
        return wrapper

    try:
        patch(dataset, 'get_data_loaders', loaders)
        patch(dataset, 'evaluate', timed('evaluation'))
        patch(model, 'meta_begin_task', timed('begin_task', 'after_begin_task'))
        patch(model, 'meta_end_task', timed('end_task', 'after_end_task'))
        patch(model, 'meta_begin_epoch', timed('begin_epoch'))
        patch(model, 'meta_end_epoch', timed('end_epoch'))
        patch(model, 'meta_observe', observed)
        patch(model, 'forward', forwarded)
        patch(model, 'meta_end_eval', evaluated)
        patch(training, 'train_single_epoch', timed('train_epoch'))
        yield
    finally:
        for target, name, original, owned in reversed(restored):
            if owned:
                setattr(target, name, original)
            else:
                delattr(target, name)
