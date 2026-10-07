"""Opt-in native HiDe execution using Mammoth's unchanged trainer and meta hooks.

Full validation requires a completed independent reference. Smoke results are
labeled separately. Efficiency requires a passing full-validation result.
Full CL runs the accepted implementation over all ten native task partitions.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import Recorder, flatten, instrument, integrate_power, model_snapshot, notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.hide_protocol import (CHECKPOINT, CHECKPOINT_SHA256, CANONICAL_PRETRAINED_SHA256,
                                   REFERENCE_COMMIT, configure_epoch_loader, file_hash,
                                   native_command as _native_command, protocol, seed_phase,
                                   source_hashes, validate_efficiency_acceptance,
                                   validate_full_cl_acceptance, FULL_CL_PROTECTED_FILES, FULL_CL_PRECISION,
                                   forbid_cifar_download, validate_shared_data)


def native_command(spec, output):
    return _native_command(spec['mode'], spec['checkpoint_file'], spec['data_root'], output)


def validate_spec(spec):
    if (spec.get('mode') not in ('validation', 'smoke', 'efficiency', 'full_cl')
            or spec.get('protocol') != protocol(spec['mode']) or spec.get('gpu') != '1'
            or spec.get('model') != 'hide-prompt'
            or spec.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or spec.get('pass') not in ('measurement', 'flops')
            or (spec['pass'] == 'flops' and spec['mode'] != 'efficiency')):
        raise ValueError('Native HiDe settings differ from the approved GPU1 protocol')


def reference_record(path):
    document = json.loads(Path(path).read_text())
    return document.get('official_reference', document)


def validate_reference(reference, initial_state_hash):
    if (reference.get('status') != 'completed' or reference.get('mode') != 'validation'
            or reference.get('protocol') != protocol('validation')
            or reference.get('reference_commit') != REFERENCE_COMMIT
            or reference.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or reference.get('initial_state_sha256') != initial_state_hash):
        raise ValueError('HiDe requires a completed reference with identical full protocol and learned initialization')
    from scripts.hide_acceptance import reference_policy
    reference_policy(reference.get('task_results', reference.get('tasks_results', [])))


def build_spec(options, profile):
    if profile['gpu'] != '1':
        raise ValueError('Approved HiDe execution requires physical GPU1')
    mode = options.mode
    initial = Path(options.initial_state).resolve() if options.initial_state else None
    if initial is None or not initial.is_file():
        raise ValueError('Native HiDe requires --initial-state exported by the independent reference')
    checkpoint = Path(getattr(options, 'checkpoint', CHECKPOINT) or CHECKPOINT).resolve()
    if file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Native HiDe requires the pinned local Base checkpoint')
    validate_shared_data(profile['data_root'])
    name = 'hide-native-' + mode + '-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (options.result_dir or options.output / name).resolve()
    spec = {'schema_version': 1, 'name': name, 'model': 'hide-prompt', 'mode': mode,
            'pass': options.pass_kind, 'protocol': protocol(mode), 'tasks': protocol(mode)['tasks'],
            'output': str(output), 'gpu': profile['gpu'], 'data_root': profile['data_root'],
            'checkpoint_file': str(checkpoint), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
            'initial_state': str(initial), 'initial_state_sha256': file_hash(initial),
            'wandb_entity': profile['wandb_entity'], 'wandb_project': profile['wandb_project'],
            'paper_reproduction': False, 'reference_commit': REFERENCE_COMMIT,
            'execution_source_sha256': source_hashes(),
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                         text=True, check=True).stdout.strip()}
    if mode == 'validation':
        if options.reference_summary is None:
            raise ValueError('Full HiDe validation requires --reference-summary')
        path = Path(options.reference_summary).resolve()
        reference = reference_record(path)
        validate_reference(reference, spec['initial_state_sha256'])
        spec.update(reference_summary=str(path), reference_summary_sha256=file_hash(path))
    if mode in ('efficiency', 'full_cl'):
        if options.acceptance_summary is None:
            raise ValueError(f'HiDe {mode} requires --acceptance-summary')
        path = Path(options.acceptance_summary).resolve()
        prerequisite = (validate_full_cl_acceptance if mode == 'full_cl' else validate_efficiency_acceptance)(path, initial_state=initial)
        spec.update(acceptance_summary=str(path), acceptance_summary_sha256=file_hash(path))
        if mode == 'full_cl':
            spec['protected_prerequisite_sources'] = {source: prerequisite['execution_source_sha256'][source]
                                                       for source in FULL_CL_PROTECTED_FILES}
            spec['prefix_validation_tasks'] = [1, 2]
    validate_spec(spec)
    return spec


def execute(options, profile):
    spec = build_spec(options, profile)
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    notifications = []

    def notify(message):
        notifications.append({'message': message, 'status': notify_slack(message)})
        write_json(output / 'notifications.json', notifications)

    environment = os.environ.copy()
    environment.pop('MAMMOTH_DEVICE', None)
    environment.update(CUDA_VISIBLE_DEVICES=profile['gpu'], CL_HIDE_NATIVE_EXECUTE='1',
                       HF_HUB_OFFLINE='1', WANDB_MODE='online', WANDB_DIR=str(output))
    notify(f'HiDe native {spec["mode"]}/{spec["pass"]} started: GPU1; TII{spec["protocol"]["tii_epochs"]}+prompt{spec["protocol"]["prompt_epochs"]}; tasks{spec["tasks"]}')
    started, heartbeat = time.monotonic(), time.monotonic()
    timed_out = False
    from scripts.benchmark_cl import stop_worker
    with (output / 'console.log').open('w') as console:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(output / 'spec.json')],
                                   cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= options.timeout:
                    timed_out = True
                    stop_worker(process)
                    break
                if time.monotonic() - heartbeat >= 600:
                    path = output / 'progress.json'
                    progress = json.loads(path.read_text()) if path.exists() else {}
                    notify(f'HiDe native {spec["mode"]}: task{progress.get("task", "?")}/{spec["tasks"]}, {progress.get("phase", "initializing")} epoch{progress.get("epoch", "?")}; W&B {progress.get("wandb_url") or "initializing"}')
                    heartbeat = time.monotonic()
                time.sleep(1)
        except BaseException:
            stop_worker(process)
            notify('HiDe native interrupted; partial local results preserved')
            raise
    path = output / 'result.json'
    record = json.loads(path.read_text()) if path.exists() else {**spec, 'status': 'failed'}
    if timed_out:
        record['status'] = 'timeout'
    elif process.returncode != 0 and record['status'] == 'completed':
        record['status'] = 'failed'
    write_json(path, record)
    write_json(output / 'metrics.json', record)
    notify(f'HiDe native {record["status"]}: acceptance={record.get("acceptance_status", "not_evaluated")}; W&B {record.get("wandb_url") or "unavailable"}')
    write_json(output / 'completion.json', {'status': record['status'], 'exit_code': process.returncode,
               'timeout': timed_out, 'wall_s': time.monotonic() - started,
               'acceptance_status': record.get('acceptance_status')})
    print(json.dumps({'status': record['status'], 'acceptance_status': record.get('acceptance_status'),
                      'output': str(output), 'wandb_url': record.get('wandb_url')}), flush=True)
    return 0 if process.returncode == 0 and record['status'] == 'completed' else 1


def tensor_digest(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not hasattr(tensor, 'detach'):
            continue
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(tuple(value.shape)).encode())
        result.update(str(value.dtype).encode()); result.update(value.numpy().tobytes())
    return result.hexdigest()


def completion_checks(record, recorder, model, settings):
    """Require each individual phase/epoch and its full stream, not only totals."""
    rows = record['epoch_results']
    batches = math.ceil(5000 / settings['batch_size'])
    coordinates = {(row['task'], row['phase'], row['epoch']) for row in rows}
    expected = {(task, phase, epoch) for task in range(1, settings['tasks'] + 1)
                for phase, count in (('tii_train', settings['tii_epochs']), ('prompt_train', settings['prompt_epochs']))
                for epoch in range(1, count + 1)}
    return bool(len(rows) == len(expected) and coordinates == expected
                and all(row['steps'] == batches and row['samples'] == 5000 for row in rows)
                and recorder.tasks_completed == settings['tasks']
                and recorder.steps == batches * settings['n_epochs'] * settings['tasks']
                and recorder.training_examples == 5000 * settings['n_epochs'] * settings['tasks']
                and recorder.evaluation_examples == sum(task * 1000 for task in range(1, settings['tasks'] + 1))
                and record['counts']['tii_training_steps'] == batches * settings['tii_epochs'] * settings['tasks']
                and record['counts']['prompt_training_steps'] == batches * settings['prompt_epochs'] * settings['tasks']
                and record['counts']['tii_statistics_examples'] == 5000 * settings['tasks']
                and record['counts']['prompt_statistics_examples'] == 5000 * settings['tasks']
                and record['counts']['routing_evaluation_examples'] == recorder.evaluation_examples
                and bool(model.net.task_finalized) and int(model.net.active_task) == settings['tasks'] - 1
                and len(model.alignment_history) == 2 * settings['tasks'])


def audit_finite_learned_state(module):
    """Inspect the actual alignment gradients and updated trainable tensors."""
    import torch
    for parameter in module.parameters():
        if parameter.requires_grad and (not bool(torch.isfinite(parameter).all())
                or (parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()))):
            raise FloatingPointError('HiDe classifier alignment state or gradient is nonfinite')


def compare_prefix_predictions(actual, expected):
    """Use full saved prediction tensors, IDs and TII routing for the two-task gate."""
    import numpy as np
    with np.load(actual, allow_pickle=False) as current, np.load(expected, allow_pickle=False) as accepted:
        keys = ('predictions', 'labels', 'global_ids', 'tii_predictions', 'evaluation_global_ids')
        if set(current.files) != set(accepted.files) or any(not np.array_equal(current[key], accepted[key]) for key in keys):
            raise RuntimeError('HiDe full CL prefix predictions differ from the accepted independent validation')
    return True


class TaskResources:
    """Per-task intervals with real sensors, preserving the process CUDA high water.

    This observer calls no model forward and consumes no random numbers. Task
    intervals include data loading, training, end-task work and checkpoint audit.
    Phase times are inclusive; they must not be added to estimate task wall time.
    """

    def __init__(self, recorder):
        self.recorder = recorder
        self.rows = []
        self.active = None
        self.global_cuda = recorder.cuda_memory()

    @staticmethod
    def cpu_seconds():
        import resource
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime

    def capture_cuda(self):
        current = self.recorder.cuda_memory()
        for key, value in current.items():
            if value is not None:
                self.global_cuda[key] = max(value, self.global_cuda.get(key) or 0.)
        return current

    def start(self, task):
        if self.active is not None or task != len(self.rows) + 1:
            raise RuntimeError('HiDe task resource intervals must be sequential and nonoverlapping')
        recorder = self.recorder
        recorder.synchronize()
        self.capture_cuda()
        if recorder.torch is not None and recorder.device is not None and recorder.device.type == 'cuda':
            recorder.torch.cuda.reset_peak_memory_stats(recorder.device)
        recorder.sample()
        self.active = {'task': task, 'start_s': time.perf_counter() - recorder.started,
                       'phase_offset': len(recorder.phases), 'cpu_s': self.cpu_seconds(),
                       'steps': recorder.steps, 'train_examples': recorder.training_examples,
                       'eval_examples': recorder.evaluation_examples}

    def finish(self, task):
        if self.active is None or self.active['task'] != task:
            raise RuntimeError('HiDe task resources ended outside the active task')
        recorder, active = self.recorder, self.active
        recorder.synchronize()
        end = time.perf_counter() - recorder.started
        cuda = self.capture_cuda()
        recorder.sample()
        # A boundary sample can finish just before the background monitor's
        # earlier sample; timestamp order is the integration order.
        samples = sorted(recorder.samples, key=lambda sample: sample['elapsed_s'])
        maximum_gap = max(6., recorder.interval * 3)
        joules, coverage = integrate_power(samples, [(active['start_s'], end)], maximum_gap)
        metrics = {**cuda, 'gpu_board_energy_j': joules,
                   'gpu_board_energy_wh': joules / 3600 if joules is not None else None,
                   'gpu_energy_coverage_s': coverage,
                   'gpu_energy_coverage_fraction': coverage / (end - active['start_s']),
                   'process_tree_cpu_s': self.cpu_seconds() - active['cpu_s'],
                   'outer_observe_steps': recorder.steps - active['steps'],
                   'outer_training_stream_examples': recorder.training_examples - active['train_examples'],
                   'evaluation_examples': recorder.evaluation_examples - active['eval_examples']}
        task_samples = [sample for sample in samples if active['start_s'] <= sample['elapsed_s'] <= end]
        for key in ('process_tree_rss_mib', 'gpu_device_memory_mib', 'gpu_utilization_percent', 'gpu_power_w'):
            values = [sample[key] for sample in task_samples if sample.get(key) is not None]
            metrics['peak_' + key] = max(values) if values else None
        selected = recorder.phases[active['phase_offset']:]
        phases = {}
        for name in sorted({phase['phase'] for phase in selected}):
            spans = [(phase['start_s'], phase['end_s']) for phase in selected if phase['phase'] == name]
            energy, covered = integrate_power(samples, spans, maximum_gap)
            phases[name] = {'wall_s': sum(last - first for first, last in spans), 'calls': len(spans),
                            'gpu_board_energy_j': energy, 'gpu_energy_coverage_s': covered}
        row = {'task': task, 'start_s': active['start_s'], 'end_s': end,
               'wall_s': end - active['start_s'], 'metrics': metrics, 'phases': phases}
        self.rows.append(row)
        self.active = None
        return row


def worker(spec_path):
    if os.getenv('CL_HIDE_NATIVE_EXECUTE') != '1':
        raise ValueError('HiDe worker requires explicit --execute')
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    if file_hash(spec['checkpoint_file']) != spec['checkpoint_sha256'] or file_hash(spec['initial_state']) != spec['initial_state_sha256']:
        raise ValueError('HiDe pinned local initial states changed')
    if any(file_hash(ROOT / source) != pin for source, pin in spec['execution_source_sha256'].items()):
        raise ValueError('HiDe execution source changed after preparation')
    reference, accepted = None, None
    if spec['mode'] == 'validation':
        if file_hash(spec['reference_summary']) != spec['reference_summary_sha256']:
            raise ValueError('HiDe reference summary changed')
        reference = reference_record(spec['reference_summary'])
        validate_reference(reference, spec['initial_state_sha256'])
    if spec['mode'] in ('efficiency', 'full_cl'):
        if file_hash(spec['acceptance_summary']) != spec['acceptance_summary_sha256']:
            raise ValueError('HiDe acceptance summary changed')
        accepted = (validate_full_cl_acceptance if spec['mode'] == 'full_cl' else validate_efficiency_acceptance)(
            spec['acceptance_summary'], initial_state=spec['initial_state'])
    output = Path(spec['output'])
    settings = spec['protocol']
    recorder = Recorder(output, spec['gpu'])
    recorder.start()
    record = {**spec, 'status': 'failed', 'acceptance_status': 'not_evaluated',
              'task_results': [], 'epoch_results': [], 'structural_checks': {},
              'counts': {'tii_training_steps': 0, 'prompt_training_steps': 0,
                         'tii_statistics_examples': 0, 'prompt_statistics_examples': 0,
                         'prompt_training_tii_examples': 0, 'routing_evaluation_examples': 0}}
    code, wandb, stack, task_resources = 1, None, ExitStack(), None
    try:
        import numpy as np
        import torch
        import wandb
        from unittest.mock import patch
        from torch.utils.data import SequentialSampler
        import main as mammoth
        import utils.training as training
        from scripts.ease_reference_utils import dataset_fingerprint, shared_cifar_metadata
        from scripts.ease_size_protocol import canonical_pretrained_digest, canonical_pretrained_state
        from scripts.hide_reference_utils import load_initial_state, learned_state_digest
        from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS, counts_from_predictions, evaluate_acceptance, reference_policy
        stack.enter_context(forbid_cifar_download())
        validate_shared_data(spec['data_root'])
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
        command = native_command(spec, output)
        sys.argv = ['main.py', *command]
        args = mammoth.parse_args(cmd=command)
        wandb.init(entity=spec['wandb_entity'], project=spec['wandb_project'], name=spec['name'],
                   group={'efficiency': 'hide-base-efficiency', 'full_cl': 'hide-base-full-cl'}.get(spec['mode'], 'hide-base-validation'),
                   job_type=f'native-{spec["mode"]}-{spec["pass"]}', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        with recorder.phase('initialize'):
            model, dataset, args = mammoth.initialize(args)
            initial_metadata = load_initial_state({'tii': model.net.tii, 'prompt': model.net.prompt}, spec['initial_state'])
        # extend_args rewrites nowand from environment; take ownership afterwards.
        args.nowand = 1
        expected = {'stop_after': settings['tasks'], 'n_epochs': settings['n_epochs'],
                    'batch_size': settings['batch_size'], 'seed': settings['seed'],
                    'num_workers': settings['num_workers'], 'hide_tii_epochs': settings['tii_epochs'],
                    'hide_prompt_epochs': settings['prompt_epochs'], 'hide_ca_epochs': settings['ca_epochs'],
                    'dataset_config': settings['dataset_config'], 'optimizer': 'adam', 'optim_wd': 0.,
                    'permute_classes': 0, 'noise_rate': 0, 'validation': 0, 'label_perc': 1,
                    'label_perc_by_class': 1, 'debug_mode': 0, 'hide_use_permute_fix': False}
        if any(getattr(args, key) != value for key, value in expected.items()) or args.lr_scheduler is not None or args.eval_epochs is not None:
            raise ValueError('Effective HiDe arguments differ from the matched protocol')
        if dataset.N_TASKS != 10 or dataset.N_CLASSES_PER_TASK != 10 or dataset.N_CLASSES != 100:
            raise ValueError('HiDe must keep all ten class partitions and stop after approved tasks')
        record['native_cli'] = ['main.py', *command]
        record['initialization_metadata'] = initial_metadata
        record['initial_learned_sha256'] = {name: learned_state_digest(module, name) for name, module in
                                          (('tii', model.net.tii), ('prompt', model.net.prompt))}
        if reference is not None and record['initial_learned_sha256'] != reference.get('initial_learned_sha256'):
            raise ValueError('Native learned tensors differ from the independent reference initialization')
        if spec['mode'] == 'full_cl' and record['initial_learned_sha256'] != accepted.get('initial_learned_sha256'):
            raise ValueError('Full CL learned tensors differ from the accepted validation initialization')
        record['initial_canonical_pretrained_sha256'] = {name: canonical_pretrained_digest(module) for name, module in
                                                        (('tii', model.net.tii), ('prompt', model.net.prompt))}
        if any(value != CANONICAL_PRETRAINED_SHA256 for value in record['initial_canonical_pretrained_sha256'].values()):
            raise ValueError('Both native pretrained branches must match the pinned canonical tensors')
        checks = record['structural_checks']
        checks.update({name: True for name in REQUIRED_STRUCTURAL_CHECKS})
        checks['training_protocol_complete'] = False
        checks['dataset_matches_reference'] = reference is not None or spec['mode'] != 'validation'
        write_json(output / 'effective_args.json', vars(args))
        write_json(output / 'shared_data.json', shared_cifar_metadata(spec['data_root']))
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'wandb', 'scikit-learn')}
        record['precision'] = {'matmul_precision': 'highest', 'tf32_allowed': False, 'cudnn_tf32_allowed': True,
                               'cudnn_deterministic': True, 'cudnn_benchmark': False,
                               'autocast': False, 'parameter_dtype': 'float32', 'torch_cpu_threads': torch.get_num_threads()}
        if spec['mode'] == 'full_cl':
            if record['precision'] != FULL_CL_PRECISION:
                raise ValueError('Full CL effective precision differs from the accepted validation')
            task_resources = TaskResources(recorder)
            record['task_resources'] = task_resources.rows
            record['prefix_validation'] = {'status': 'pending', 'matched_tasks': [], 'prediction_match': False}
        record['protocol_differences'] = ['Shared native dataset wrapper and transforms; independent author model/engines.',
            'Explicit exported learned initialization and phase RNG replace constructor-dependent RNG.',
            'Statistics use stable per-class batches; evaluation keeps separate test-task batch boundaries.',
            'Native trainer transfers non_augmented inputs and records system statistics.',
            'This runner owns W&B; native W&B autologging is disabled after initialization.']
        old_prompt_digests = {}
        epoch = {}
        predictions, tii_predictions = [], []
        active_labels = None
        stack.enter_context(instrument(recorder, model, dataset, training))

        def loaders(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                if task_resources is not None:
                    task_resources.start(len(dataset.test_loaders) + 1)
                result = original(*arguments, **keywords)
                task = len(dataset.test_loaders)
                with recorder.phase('data_audit'):
                    source = dataset.train_loader.dataset
                    expected_labels = {str(label): 500 for label in range((task - 1) * 10, task * 10)}
                    recorder.data_snapshots[f'task{task}_train'] = dataset_fingerprint(source.data, source.targets, source.indexes, source.transform)
                    if recorder.data_snapshots[f'task{task}_train']['class_counts'] != expected_labels:
                        checks['no_future_or_replay_samples'] = False
                        raise RuntimeError('HiDe training loader contains missing, old or future raw samples')
                    images = np.concatenate([loader.dataset.data for loader in dataset.test_loaders])
                    labels = np.concatenate([loader.dataset.targets for loader in dataset.test_loaders])
                    ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
                    order = np.argsort(ids, kind='stable')
                    recorder.data_snapshots[f'task{task}_test'] = dataset_fingerprint(images[order], labels[order], ids[order], dataset.TEST_TRANSFORM)
                    matched = reference if reference is not None else accepted
                    if matched is not None and (spec['mode'] != 'full_cl' or task <= 2):
                        keys = ('n_samples', 'class_counts', 'samples_ordered_sha256', 'samples_unordered_sha256', 'global_ids_ordered_sha256', 'transform_repr_sha256')
                        if spec['mode'] == 'efficiency':
                            keys = keys[:-1]  # Approved efficiency uses the common ease transform.
                        for split in ('train', 'test'):
                            actual = recorder.data_snapshots[f'task{task}_{split}']
                            expected = matched['data_snapshots'][f'task{task}_{split}']
                            if any(actual[key] != expected[key] for key in keys):
                                checks['dataset_matches_reference'] = False
                                raise RuntimeError('Native samples/preprocessing differ from the independent reference')
                return result
            return wrapped

        def begin_epoch(original):
            @functools.wraps(original)
            def wrapped(index, current_dataset):
                phase = 'tii_train' if index < settings['tii_epochs'] else 'prompt_train'
                local = index if phase == 'tii_train' else index - settings['tii_epochs']
                selected = seed_phase(settings['seed'], model.current_task, phase, local)
                configure_epoch_loader(dataset.train_loader, settings['seed'], model.current_task, phase, local)
                epoch.clear()
                epoch.update(phase=phase, epoch=local + 1, outer_epoch=index + 1,
                             task=model.current_task + 1, seed=selected, loss_sum=0., loss_weighted_sum=0.,
                             steps=0, samples=0, correct=0)
                result = original(index, current_dataset)
                recorder.snapshots[f'task{model.current_task + 1}_{phase}_epoch{local + 1}'] = model_snapshot(model)
                return result
            return wrapped

        def observed(original):
            @functools.wraps(original)
            def wrapped(inputs, labels, *arguments, **keywords):
                nonlocal active_labels
                active_labels = labels
                try:
                    result = original(inputs, labels, *arguments, **keywords)
                    loss = float(result['loss'] if isinstance(result, dict) else result)
                    if not math.isfinite(loss):
                        checks['finite_loss_and_gradients'] = False
                        raise FloatingPointError('Native HiDe loss is nonfinite')
                    epoch['loss_sum'] += loss
                    epoch['loss_weighted_sum'] += loss * len(labels)
                    epoch['steps'] += 1
                    epoch['samples'] += len(labels)
                    record['counts'][('tii' if model._phase == 'tii' else 'prompt') + '_training_steps'] += 1
                    if epoch['epoch'] == 1 and epoch['steps'] == 1:
                        with recorder.phase('first_batch_audit'):
                            gradients = {name: parameter.grad for name, parameter in model.net.named_parameters()
                                         if parameter.requires_grad and parameter.grad is not None}
                            record.setdefault('first_batch_audits', []).append({
                                'task': model.current_task + 1, 'phase': epoch['phase'],
                                'loss': loss, 'samples': len(labels), 'seed': epoch['seed'],
                                'input_sha256': tensor_digest({'inputs': inputs}),
                                'labels_sha256': tensor_digest({'labels': labels}),
                                'clipped_gradient_sha256': tensor_digest(gradients)})
                            write_json(output / 'first_batch_audits.json', record['first_batch_audits'])
                            matched = reference if reference is not None else (accepted if spec['mode'] == 'full_cl' and model.current_task < 2 else None)
                            if matched is not None:
                                matching = [row for row in matched.get('first_batch_audits', [])
                                            if row['task'] == model.current_task + 1 and row['phase'] == epoch['phase']]
                                current = record['first_batch_audits'][-1]
                                compared_keys = ('samples', 'seed', 'input_sha256', 'labels_sha256')
                                if spec['mode'] == 'full_cl':
                                    compared_keys += ('clipped_gradient_sha256',)
                                if len(matching) != 1 or any(current[key] != matching[0][key] for key in compared_keys):
                                    checks['matched_phase_rng_and_batching'] = False
                                    raise RuntimeError('HiDe actual first-batch images/labels differ from the independent reference')
                    return result
                finally:
                    active_labels = None
            return wrapped

        def branch_forward(original, branch):
            @functools.wraps(original)
            def wrapped(inputs, *arguments, **keywords):
                result = original(inputs, *arguments, **keywords)
                if recorder.active_phase == branch + '_statistics':
                    record['counts'][branch + '_statistics_examples'] += len(inputs)
                if active_labels is not None:
                    if branch == model._phase:
                        logits = result['logits'][:, model.n_past_classes:model.n_seen_classes]
                        epoch['correct'] += int((logits.detach().argmax(1) + model.n_past_classes == active_labels).sum())
                    elif branch == 'tii' and model._phase == 'prompt':
                        record['counts']['prompt_training_tii_examples'] += len(inputs)
                return result
            return wrapped

        def train_epoch(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                with recorder.phase(epoch['phase']):
                    result = original(*arguments, **keywords)
                with recorder.phase('epoch_audit'):
                    for branch in (model.net.tii, model.net.prompt):
                        frozen = canonical_pretrained_state(branch)
                        names = dict(branch.named_parameters())
                        if any(names[name].grad is not None for name in frozen):
                            checks['frozen_backbone_gradients_absent'] = False
                            raise RuntimeError('Frozen HiDe pretrained tensor received a gradient')
                    if any(not bool(torch.isfinite(parameter).all()) or (parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()))
                           for parameter in model.net.parameters() if parameter.requires_grad):
                        checks['finite_loss_and_gradients'] = False
                        raise FloatingPointError('Nonfinite native HiDe learnable state/gradient')
                    for index, pin in old_prompt_digests.items():
                        if tensor_digest({'prompt': model.net.prompt.e_prompt.prompt[:, :, int(index)]}) != pin:
                            checks['old_prompt_slots_unchanged'] = False
                            raise RuntimeError('Previously finalized HiDe prompt changed')
                row = {key: epoch[key] for key in ('task', 'phase', 'epoch', 'outer_epoch', 'seed', 'steps', 'samples')}
                row.update(loss=epoch['loss_sum'] / epoch['steps'],
                           loss_example_weighted=epoch['loss_weighted_sum'] / epoch['samples'],
                           train_accuracy_pct=epoch['correct'] / epoch['samples'] * 100,
                           learning_rates=[group['lr'] for group in model.opt.param_groups])
                record['epoch_results'].append(row)
                write_json(output / 'epoch_results.json', record['epoch_results'])
                write_json(output / 'progress.json', {**row, 'wandb_url': record['wandb_url']})
                wandb.log({f'native/{key}': value for key, value in row.items() if isinstance(value, (int, float))})
                print(f'Native task{row["task"]}/{settings["tasks"]} {row["phase"]} epoch{row["epoch"]}: loss={row["loss"]:.5f}, train={row["train_accuracy_pct"]:.2f}%', flush=True)
                return result
            return wrapped

        def statistics(original):
            @functools.wraps(original)
            def wrapped(current_dataset, branch):
                seed_phase(settings['seed'], model.current_task, branch + '_statistics')
                if spec['mode'] == 'full_cl':
                    write_json(output / 'progress.json', {'task': model.current_task + 1, 'phase': branch + '_statistics',
                               'wandb_url': record['wandb_url']})
                with recorder.phase(branch + '_statistics'):
                    result = original(current_dataset, branch)
                stats = model.tii_statistics if branch == 'tii' else model.prompt_statistics
                seen = model.n_seen_classes
                if (list(stats.means.shape) != [100, 10, 768] or not bool(stats.valid[:seen].all())
                        or bool(stats.valid[seen:].any()) or bool((stats.class_tasks[seen:] != -1).any())
                        or not bool(torch.isfinite(stats.means).all()) or not bool(torch.isfinite(stats.variances).all())):
                    checks['statistics_shapes_and_seen_classes'] = False
                    raise RuntimeError('HiDe statistics shapes/seen-class state are invalid')
                return result
            return wrapped

        def alignment(original):
            @functools.wraps(original)
            def wrapped(branch):
                seed_phase(settings['seed'], model.current_task, branch + '_alignment')
                if spec['mode'] == 'full_cl':
                    write_json(output / 'progress.json', {'task': model.current_task + 1, 'phase': branch + '_alignment',
                               'epochs': settings['ca_epochs'], 'wandb_url': record['wandb_url']})
                hooks = []
                original_sgd = torch.optim.SGD
                module = model.net.tii if branch == 'tii' else model.net.prompt

                def finite_audit(*ignored):
                    with recorder.phase(branch + '_alignment_audit'):
                        try:
                            audit_finite_learned_state(module)
                        except FloatingPointError:
                            checks['finite_loss_and_gradients'] = False
                            raise

                def observed_sgd(*arguments, **keywords):
                    subject = original_sgd(*arguments, **keywords)
                    hooks.append(subject.register_step_pre_hook(finite_audit))
                    hooks.append(subject.register_step_post_hook(finite_audit))
                    return subject

                try:
                    with patch.object(torch.optim, 'SGD', observed_sgd), recorder.phase(branch + '_alignment'):
                        result = original(branch)
                    finite_audit()
                finally:
                    for hook in hooks:
                        hook.remove()
                metric = model.alignment_history[-1]
                expected_steps = 0 if model.current_task == 0 else model.n_past_classes * settings['ca_epochs']
                if metric['optimizer_steps'] != expected_steps or metric['skipped'] != (model.current_task == 0):
                    checks['alignment_schedule_complete'] = False
                    raise RuntimeError('HiDe classifier alignment did not complete its source schedule')
                return result
            return wrapped

        def finalize(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                with recorder.phase('prompt_finalize'):
                    return original(*arguments, **keywords)
            return wrapped

        def routed(original):
            @functools.wraps(original)
            def wrapped(inputs, *arguments, **keywords):
                result = original(inputs, *arguments, **keywords)
                if recorder.active_phase == 'evaluation':
                    tii_predictions.append(result['tii_logits'][:, :model.n_seen_classes].detach().argmax(1).cpu().numpy())
                    record['counts']['routing_evaluation_examples'] += len(inputs)
                return result
            return wrapped

        def forwarded(original):
            @functools.wraps(original)
            def wrapped(inputs, *arguments, **keywords):
                if arguments or keywords:
                    checks['routing_without_task_oracle'] = False
                    raise RuntimeError('HiDe class-IL forward received an oracle task argument')
                result = original(inputs)
                if recorder.active_phase == 'evaluation':
                    if not bool(torch.isfinite(result).all()):
                        checks['finite_loss_and_gradients'] = False
                        raise FloatingPointError('HiDe seen-class evaluation logits are nonfinite')
                    predictions.append(torch.argsort(result.detach(), dim=1, descending=True, stable=True)[:, :5].cpu().numpy())
                return result
            return wrapped

        def evaluated(original):
            @functools.wraps(original)
            def wrapped(*arguments, **keywords):
                predictions.clear(); tii_predictions.clear()
                task = int(model.net.active_task) + 1
                seed_phase(settings['seed'], task - 1, 'evaluation')
                if keywords.get('last', False) or keywords.get('return_loss', False):
                    raise ValueError('HiDe validation must evaluate every seen task at task end')
                if any(not isinstance(loader.sampler, SequentialSampler) or loader.drop_last for loader in dataset.test_loaders):
                    raise ValueError('HiDe evaluation must preserve test-task batch boundaries')
                result = original(*arguments, **keywords)
                with recorder.phase('acceptance_audit'):
                    labels = np.concatenate([loader.dataset.targets for loader in dataset.test_loaders])
                    ids = np.concatenate([loader.dataset.indexes for loader in dataset.test_loaders])
                    predicted = np.concatenate(predictions)
                    tii = np.concatenate(tii_predictions)
                    counts = counts_from_predictions(predicted[:, 0], labels, past_classes=(task - 1) * 10, seen_classes=task * 10)
                    row = {'task': task, 'seen_classes': task * 10, **counts,
                           'seen_accuracy_pct': 100 * counts['seen_correct'] / counts['seen_samples'],
                           'old_accuracy_pct': 100 * counts['old_correct'] / counts['old_samples'] if counts['old_samples'] else None,
                           'new_accuracy_pct': 100 * counts['new_correct'] / counts['new_samples'],
                           'tii_seen_accuracy_pct': float(100 * (tii == labels).mean()),
                           'tii_task_routing_accuracy_pct': float(100 * (tii // 10 == labels // 10).mean()),
                           'native_class_il_per_task': result[0], 'native_task_il_per_task': result[1]}
                    if len(labels) != task * 1000 or any(abs(result[0][index] - 100 * (predicted[labels // 10 == index, 0] == labels[labels // 10 == index]).mean()) > 1e-9 for index in range(task)):
                        raise RuntimeError('Saved class-IL predictions disagree with native evaluation')
                    current = {name: canonical_pretrained_digest(module) for name, module in (('tii', model.net.tii), ('prompt', model.net.prompt))}
                    if current != record['initial_canonical_pretrained_sha256']:
                        checks['frozen_backbones_unchanged'] = False
                        raise RuntimeError('HiDe frozen pretrained branch changed')
                    bank = model.net.prompt.e_prompt.prompt
                    for index, pin in old_prompt_digests.items():
                        if tensor_digest({'prompt': bank[:, :, int(index)]}) != pin:
                            checks['old_prompt_slots_unchanged'] = False
                            raise RuntimeError('Previously finalized prompt changed after task end')
                    old_prompt_digests.update({str(index): tensor_digest({'prompt': bank[:, :, index]}) for index in range(task)})
                    order = np.argsort(ids, kind='stable')
                    np.savez(output / f'task{task}_predictions.npz', predictions=predicted[order], labels=labels[order],
                             global_ids=ids[order], tii_predictions=tii[order], evaluation_global_ids=ids)
                    if spec['mode'] == 'full_cl' and task <= 2:
                        compare_prefix_predictions(output / f'task{task}_predictions.npz',
                                                   Path(spec['acceptance_summary']).parent / f'task{task}_predictions.npz')
                        record['prefix_validation']['matched_tasks'].append(task)
                        record['prefix_validation']['prediction_match'] = True
                    with recorder.phase('checkpoint'):
                        state = {name: value.detach().cpu().clone() if torch.is_tensor(value) else value for name, value in model.state_dict().items()}
                        before = tensor_digest(state)
                        torch.save({'model': state, 'task': task, 'protocol': settings,
                                    'checkpoint_sha256': CHECKPOINT_SHA256}, output / f'task{task}_state.pt')
                        loaded = torch.load(output / f'task{task}_state.pt', map_location='cpu', weights_only=True)
                        model.load_state_dict(loaded['model'], strict=True)
                        if tensor_digest(model.state_dict()) != before:
                            checks['checkpoint_round_trip'] = False
                            raise RuntimeError('HiDe statistics/routing changed across native checkpoint reload')
                        del state, loaded
                    recorder.snapshots[f'task{task}_after_end_task'] = model_snapshot(model)
                    record['task_results'].append(row)
                    write_json(output / 'task_results.json', record['task_results'])
                    if spec['mode'] == 'full_cl' and task == 2:
                        prefix_checks = {**checks, 'training_protocol_complete': True}
                        prefix = evaluate_acceptance(record['task_results'], reference_policy(accepted['task_results']),
                                                     structural_checks=prefix_checks)
                        if prefix['status'] != 'pass':
                            raise RuntimeError('HiDe full CL first-two-task accuracy failed accepted validation')
                        record['prefix_validation'].update(status='pass', acceptance=prefix,
                                                          scope='First two tasks only; no independent ten-task performance reference')
                    if spec['mode'] == 'full_cl':
                        write_json(output / 'prefix_validation.json', record['prefix_validation'])
                if task_resources is not None:
                    resource_row = task_resources.finish(task)
                    write_json(output / 'per_task_resources.json', task_resources.rows)
                    wandb.log({**flatten({f'task{task}_resources': resource_row['metrics']}),
                               f'task{task}_resources/wall_s': resource_row['wall_s']})
                wandb.log({f'native/{key}': value for key, value in row.items() if isinstance(value, (int, float))})
                record.setdefault('task_notifications', []).append({'task': task, 'status': notify_slack(
                    f'HiDe native {spec["mode"]} task{task}/{settings["tasks"]}: seen={row["seen_accuracy_pct"]:.2f}%, routing={row["tii_task_routing_accuracy_pct"]:.2f}%; W&B {record["wandb_url"]}')})
                return result
            return wrapped

        wrappers = [(dataset, 'get_data_loaders', loaders), (model, 'meta_begin_epoch', begin_epoch),
                    (model, 'meta_observe', observed), (training, 'train_single_epoch', train_epoch),
                    (model, '_collect_statistics', statistics), (model, '_align', alignment),
                    (model.net, 'finalize_task', finalize),
                    (model.net, 'tii_outputs', lambda original: branch_forward(original, 'tii')),
                    (model.net, 'prompt_outputs', lambda original: branch_forward(original, 'prompt')),
                    (model.net, 'routed_outputs', routed), (model, 'forward', forwarded),
                    (dataset, 'evaluate', evaluated)]
        for target, name, wrapper in wrappers:
            stack.enter_context(patch.object(target, name, wrapper(getattr(target, name))))
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) if spec['pass'] == 'flops' else None
        with profiler if profiler is not None else nullcontext():
            # All native task/epoch hooks and meta_observe remain in this loop.
            training.train(model, dataset, args)
        if profiler is not None:
            operations = [{'operation': item.key, 'count': item.count, 'flops': item.flops} for item in profiler.key_averages() if item.flops]
            record['flops'] = {'supported_ops_total': sum(item['flops'] for item in operations) or None,
                              'scope': 'Torch-profiler supported matrix multiply/convolution operators across complete one-task execution',
                              'partial': True, 'includes_backward_only_if_supported': True,
                              'excludes': ['unsupported operators', 'CPU sklearn clustering', 'data decoding', 'checkpoint/logging'],
                              'measurement_aggregate_eligible': False}
            write_json(output / 'flops_ops.json', operations)
        complete = completion_checks(record, recorder, model, settings)
        checks['training_protocol_complete'] = bool(complete)
        if not complete:
            raise RuntimeError('Native HiDe returned before every approved task/phase completed')
        if spec['mode'] == 'full_cl' and (record['prefix_validation']['status'] != 'pass'
                or len(task_resources.rows) != 10 or sum(row['optimizer_steps'] for row in model.alignment_history) != 27000
                or any(row['metrics']['outer_observe_steps'] != 5530 or row['metrics']['outer_training_stream_examples'] != 350000
                       or row['metrics']['evaluation_examples'] != 1000 * row['task'] for row in task_resources.rows)):
            raise RuntimeError('Native HiDe full CL prefix, task resources or alignment counts are incomplete')
        record['alignment'] = [{**row, 'task': row['task_id'] + 1,
                                'status': 'not_applicable' if row['skipped'] else 'completed',
                                'steps': row['optimizer_steps']} for row in model.alignment_history]
        record['alignment_history'] = record['alignment']
        record['model_phase_history'] = model.phase_history
        if spec['mode'] == 'validation':
            report = evaluate_acceptance(record['task_results'], reference_policy(reference['task_results']), structural_checks=checks)
            record.update(acceptance_status=report['status'], acceptance=report)
            write_json(output / 'acceptance.json', report)
        else:
            record['acceptance_status'] = 'not_applicable'
        record['status'], code = 'completed', 0
    except BaseException as error:
        logging.exception('Native HiDe execution failed')
        record['error_type'] = type(error).__name__
    finally:
        stack.close()
        record['tasks_results'] = record['task_results']
        record['computation_status'] = record['status']
        if task_resources is not None:
            recorder.stop()
            recorder.samples.sort(key=lambda sample: sample['elapsed_s'])
        record = recorder.save(record)
        if task_resources is not None:
            task_resources.capture_cuda()
            record['metrics'].update(task_resources.global_cuda)
            record['measurement_notes']['cuda_memory_scope'] = 'Per-task allocator high water resets at data loading; process maximum preserves initialization and all tasks.'
            record['measurement_notes']['task_resource_scope'] = 'Data loading through evaluation and checkpoint audit; excludes subsequent W&B and Slack calls. CPU includes own process and reaped descendants.'
        # Keep the uploaded measurement bytes immutable when the final local
        # result subsequently adds upload/finish status and artifact identity.
        write_json(output / 'artifact_metrics.json', record)
        if wandb is not None and wandb.run is not None:
            try:
                wandb.run.summary.update({**{key: value for key, value in flatten({'native_resources': record['metrics'],
                    'native_counts': record['counts'], 'native_acceptance': record.get('acceptance', {})}).items() if isinstance(value, (int, float, bool))},
                    'native_status': record['status'], 'acceptance_status': record['acceptance_status'], 'paper_reproduction': False})
                artifact = wandb.Artifact(spec['name'], type='cl-hide-native-result')
                artifact_files = {}
                for filename in ('spec.json', 'metrics.json', 'acceptance.json', 'effective_args.json', 'epoch_results.json',
                                 'task_results.json', 'shared_data.json', 'telemetry.csv', 'phases.json',
                                 'per_task_resources.json', 'prefix_validation.json', 'first_batch_audits.json', 'flops_ops.json',
                                 *(f'task{task}_predictions.npz' for task in range(1, settings['tasks'] + 1))):
                    local_file = output / ('artifact_metrics.json' if filename == 'metrics.json' else filename)
                    if local_file.is_file():
                        artifact.add_file(str(local_file), name=filename)
                        artifact_files[filename] = {'local_file': local_file.name, 'sha256': file_hash(local_file)}
                record['wandb_artifact_files'] = artifact_files
                write_json(output / 'artifact_files.json', artifact_files)
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
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('validation', 'smoke', 'efficiency', 'full_cl'), default='validation')
    parser.add_argument('--pass', dest='pass_kind', choices=('measurement', 'flops'), default='measurement')
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--reference-summary', type=Path)
    parser.add_argument('--acceptance-summary', type=Path)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/hide_prompt/runs')
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--timeout', type=float, default=21600.)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if not options.execute and options.worker is None:
        print(f'Prepared HiDe native {options.mode}; execution=False; protocol={json.dumps(protocol(options.mode))}')
        return 0
    if options.worker is not None:
        return worker(options.worker)
    return execute(options, configure_environment())


if __name__ == '__main__':
    sys.exit(main())
