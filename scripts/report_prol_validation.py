"""Audit saved PROL evidence; no model construction, credentials or networking.

The six-run stage remains separate from the prior nine-method comparison.
Guarded or failed execution produces a partial report without cost averages.
"""
from __future__ import annotations

import argparse
import csv
import datetime
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import integrate_power, write_json
from scripts.report_hide_results import file_hash, method_summary, RAW_KEYS, read_json
from scripts import prol_protocol as shared
from scripts.low_rank_protocol import phase_seed

DEFAULT_RUN_ROOT = ROOT / 'results/prol_validation_20261008'
DEFAULT_OUTPUT = ROOT / 'benchmarks/cl_efficiency/prol_validation'
VALIDATION_PLAN = ROOT / 'benchmarks/cl_efficiency/prol_prompt/validation_plan.json'
REPORT_FILES = ('results.json', 'results.csv', 'RESULTS.md')
JOB_NAMES = shared.JOB_NAMES
SOURCE_COMMIT = shared.REFERENCE_COMMIT
WAND_PREFIX = 'https://wandb.ai/laymond1/cl-efficiency/runs/'


def pin(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError('Required PROL evidence is missing or empty')
    return {'file': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def check_pin(proof, path):
    expected = pin(path)
    if (not isinstance(proof, dict) or Path(proof.get('file', '')).resolve() != Path(expected['file'])
            or proof.get('sha256') != expected['sha256']
            or 'bytes' in proof and (type(proof['bytes']) is not int or proof['bytes'] != expected['bytes'])):
        raise ValueError('PROL evidence fingerprint changed')
    return expected


def strict_equal(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(strict_equal(value[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(strict_equal(a, b) for a, b in zip(value, expected))
    return value == expected


def _integer(values, key, expected):
    if type(values.get(key)) is not int or values[key] != expected:
        raise ValueError('PROL requires exact complete stream and prediction counts')


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def _finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)


def _gradient_evidence(record, result_path):
    import numpy as np
    evidence = []
    for task, row in enumerate(record.get('first_batch_audits', []), 1):
        path = Path(result_path).resolve().parent / f'task{task}_first_gradients.npz'
        if row.get('gradient_file') != path.name or row.get('gradient_file_sha256') != file_hash(path):
            raise ValueError('PROL actual first-gradient artifact changed or is missing')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        if (not arrays or any(not name.startswith('prompt.') or value.dtype != np.dtype('float32')
                              or not np.isfinite(value).all() for name, value in arrays.items())):
            raise ValueError('PROL first-gradient arrays must contain finite FP32 actual gradients')
        digest = hashlib.sha256()
        for name, value in sorted(arrays.items()):
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(b'torch.float32')
            digest.update(np.ascontiguousarray(value).tobytes())
        if digest.hexdigest() != row.get('gradient_sha256'):
            raise ValueError('PROL gradient SHA does not describe the actual saved tensors')
        evidence.append((arrays, pin(path)))
    return evidence


def _method_evidence(record, tasks):
    rows = record.get('method_audits', [])
    for task in range(1, tasks + 1):
        selected = {}
        for kind in ('boundary_rng_before', 'task_begin', 'first_batch_routing', 'outer_scheduler'):
            matches = [row for row in rows if row.get('task') == task and row.get('kind') == kind]
            if len(matches) != 1:
                raise ValueError('PROL task-boundary RNG, head growth or outer scheduler evidence is incomplete')
            selected[kind] = matches[0]
        before, begin = selected['boundary_rng_before'], selected['task_begin']
        if (not _digest(before.get('cpu_cuda_rng_sha256')) or not _digest(begin.get('cpu_cuda_rng_after_boundary_sha256'))
                or type(begin.get('head_count')) is not int or begin['head_count'] != task
                or begin.get('generator_trainable') is not (task == 1) or begin.get('previous_heads_frozen') is not True
                or begin.get('old_key_change_allowed') is not True):
            raise ValueError('PROL task-boundary RNG or trainability audit is invalid')
        expected = {'trainable_parameters': 127098 if task == 1 else 126426,
                    'frozen_parameters': 171575828 if task == 1 else 171585726,
                    'total_parameters': 171702926 if task == 1 else 171712152,
                    'generator_parameters': 672, 'query_head_parameters': 16916, 'prompt_head_count': task}
        for key, value in expected.items():
            _integer(begin.get('parameters', {}), key, value)
        routing, outer = selected['first_batch_routing'], selected['outer_scheduler']
        unique = routing.get('unique_index_count')
        if (routing.get('cross_batch_similarity_assignment') is not True or routing.get('samples') != 64
                or type(unique) is not int or not 1 <= unique <= 10
                or type(routing.get('duplicate_index_count')) is not int or routing['duplicate_index_count'] != 64 - unique
                or type(routing.get('column0_selected_count')) is not int or not 0 <= routing['column0_selected_count'] <= 64
                or not _digest(routing.get('selected_local_indices_sha256'))
                or type(outer.get('last_epoch')) is not int or outer['last_epoch'] != 1
                or not strict_equal(outer.get('effective_lrs'), [0.] * (1 if task == 1 else 2))):
            raise ValueError('PROL source first-batch routing or epoch cosine evidence is invalid')


def exact_task_metrics(record):
    rows = record.get('task_results', [])
    if len(rows) not in (1, 2):
        raise ValueError('PROL requires one or two complete saved tasks')
    result = {}
    for task, row in enumerate(rows, 1):
        for key, expected in {'task': task, 'seen_samples': task * 1000,
                              'old_samples': (task - 1) * 1000, 'new_samples': 1000}.items():
            _integer(row, key, expected)
        if 'seen_classes' in row:
            _integer(row, 'seen_classes', task * 10)
        for prefix in ('seen', 'old', 'new'):
            correct, samples = row.get(prefix + '_correct'), row[prefix + '_samples']
            if type(correct) is not int or not 0 <= correct <= samples:
                raise ValueError('PROL prediction counts are invalid')
            if samples:
                percent = Fraction(100 * correct, samples)
                if prefix + '_accuracy_pct' in row and row[prefix + '_accuracy_pct'] != float(percent):
                    raise ValueError('PROL task accuracy disagrees with exact counts')
                result[f'task{task}_{prefix}_accuracy'] = percent
        if row['seen_correct'] != row['old_correct'] + row['new_correct']:
            raise ValueError('PROL seen/old/new prediction totals disagree')
    final = result[f'task{len(rows)}_seen_accuracy']
    if record.get('accuracy', {}).get('mean_accuracy_pct') != float(final):
        raise ValueError('PROL reported accuracy disagrees with final saved counts')
    return result


def _predictions(record, result_path):
    import numpy as np
    directory = Path(result_path).resolve().parent
    saved_files, rows = record.get('prediction_files', []), record['task_results']
    if len(saved_files) != len(rows):
        raise ValueError('Every PROL task requires a pinned prediction artifact')
    predictions = []
    for task, row in enumerate(rows, 1):
        path, saved = directory / f'task{task}_predictions.npz', saved_files[task - 1]
        if (type(saved.get('task')) is not int or saved['task'] != task
                or Path(saved.get('path', saved.get('file', ''))).resolve() != path
                or saved.get('sha256') != file_hash(path)):
            raise ValueError('PROL prediction artifact fingerprint changed')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in ('predictions', 'labels', 'global_ids',
                                                          'evaluation_global_ids', 'batch_sizes')}
        size = task * 1000
        if (any(arrays[key].shape != (size,) or not np.issubdtype(arrays[key].dtype, np.integer)
                for key in ('predictions', 'labels', 'global_ids', 'evaluation_global_ids'))
                or len(np.unique(arrays['global_ids'])) != size
                or np.any(arrays['global_ids'][1:] <= arrays['global_ids'][:-1])
                or not np.array_equal(np.sort(arrays['global_ids']), np.sort(arrays['evaluation_global_ids']))
                or np.any(arrays['labels'] < 0) or np.any(arrays['labels'] >= task * 10)
                or np.any(arrays['predictions'] < 0) or np.any(arrays['predictions'] >= task * 10)
                or np.any(np.bincount(arrays['labels'], minlength=task * 10) != 100)
                or not np.issubdtype(arrays['batch_sizes'].dtype, np.integer)
                or not np.array_equal(arrays['batch_sizes'], np.asarray(([64] * 15 + [40]) * task))):
            raise ValueError('PROL saved prediction inputs, order or task tails are incomplete')
        routing = record.get('routing_audits', [])
        if len(routing) != len(rows):
            raise ValueError('PROL unknown-class routing evidence is incomplete')
        route = routing[task - 1]
        batches = route.get('batches', [])
        if (type(route.get('task')) is not int or route['task'] != task
                or not _digest(route.get('evaluation_global_ids_sha256')) or len(batches) != task * 16
                or any(route.get(key) is not True for key in ('all_active_prefix_layers_routed',
                    'cross_batch_similarity_assignment', 'evaluation_affine_clamp_preserved'))):
            raise ValueError('PROL source routing or affine-clamp audit is invalid')
        for index, (batch, length) in enumerate(zip(batches, arrays['batch_sizes'])):
            unique, duplicate = batch.get('unique_index_count'), batch.get('duplicate_index_count')
            if (type(batch.get('samples')) is not int or batch['samples'] != int(length)
                    or type(batch.get('test_task')) is not int or batch['test_task'] != index // 16 + 1
                    or batch.get('oracle_task_id', 'missing') is not None or batch.get('labels_provided') is not False
                    or type(batch.get('routing_key_count')) is not int or batch['routing_key_count'] != task * 10
                    or not _digest(batch.get('selected_local_indices_sha256'))
                    or type(unique) is not int or not 1 <= unique <= min(int(length), task * 10)
                    or type(duplicate) is not int or duplicate != int(length) - unique
                    or type(batch.get('column0_selected_count')) is not int or not 0 <= batch['column0_selected_count'] <= int(length)):
                raise ValueError('PROL routing uses incomplete class keys, task oracle or changed batch tails')
        correct, old = arrays['predictions'] == arrays['labels'], arrays['labels'] < (task - 1) * 10
        for key, value in {'seen_correct': int(correct.sum()), 'old_correct': int(correct[old].sum()),
                           'old_samples': int(old.sum()), 'new_correct': int(correct[~old].sum()),
                           'new_samples': int((~old).sum())}.items():
            _integer(row, key, value)
        predictions.append((arrays, pin(path)))
    return predictions


def _stream_evidence(record, tasks):
    batches, streams = record.get('first_batch_audits', []), record.get('sample_stream_audits', [])
    if len(batches) != tasks or len(streams) != tasks:
        raise ValueError('PROL actual first batches and stream RNG evidence are incomplete')
    for task, (batch, stream) in enumerate(zip(batches, streams), 1):
        for row in (batch, stream):
            for key, value in {'task': task, 'epoch': 1, 'seed': phase_seed(1993, task - 1, 'train', 0)}.items():
                _integer(row, key, value)
        _integer(batch, 'samples', 64)
        _integer(batch, 'optimizer_update_index', (task - 1) * 79 + 1)
        _integer(stream, 'samples', 5000)
        expected_lrs = [.0025] if task == 1 else [.0025, .00000125]
        if (batch.get('effective_lr') != .0025 or batch.get('gradient_timing') != 'before_clip'
                or not strict_equal(batch.get('effective_lrs'), expected_lrs)
                or any(not _digest(batch.get(key)) for key in ('input_sha256', 'labels_sha256', 'gradient_sha256', 'rng_sha256'))
                or any(not _digest(stream.get(key)) for key in ('global_ids_sha256', 'labels_sha256'))):
            raise ValueError('PROL actual batch LR groups, gradients or RNG hashes are invalid')
    lr_rows = record.get('lr_audits', [])
    if len(lr_rows) != tasks * 79:
        raise ValueError('PROL every-step scheduler evidence is incomplete')
    previous_rates, triggered, scheduler_steps = None, False, 0
    for index, row in enumerate(lr_rows):
        task = index // 79 + 1
        local_step = index % 79 + 1
        if local_step == 1:
            previous_rates = [.0025] if task == 1 else [.0025, .00000125]
            triggered, scheduler_steps = False, 0
        if (row.get('task') != task or type(row.get('task')) is not int
                or type(row.get('step')) is not int or row['step'] != local_step
                or type(row.get('epoch')) is not int or row['epoch'] != 1):
            raise ValueError('PROL scheduler evidence task sequence is invalid')
        for key in ('pre_step_lrs', 'post_optimizer_lrs', 'post_inner_scheduler_lrs'):
            rates = row.get(key)
            if (not isinstance(rates, list) or len(rates) != (1 if task == 1 else 2)
                    or any(not _finite(value) or value < 0 for value in rates)):
                raise ValueError('PROL scheduler group rates are invalid')
        if row.get('pre_step_lrs') != row.get('post_optimizer_lrs'):
            raise ValueError('PROL optimizer unexpectedly changed LR before scheduler')
        if not _finite(row.get('loss_ce')) or row.get('threshold') != .3:
            raise ValueError('PROL scheduler threshold operand is missing')
        triggered = triggered or row['loss_ce'] < .3
        if row.get('threshold_triggered') is not triggered or bool(row.get('inner_scheduler_step', False)) is not triggered:
            raise ValueError('PROL scheduler sticky threshold or stepping differs from the source')
        if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(row['pre_step_lrs'], previous_rates)):
            raise ValueError('PROL scheduler trace has discontinuous LR groups')
        scheduler_steps += int(triggered)
        base = [.0025] if task == 1 else [.0025, .00000125]
        expected = [.0005 + (rate - .0005) * (1 + math.cos(math.pi * scheduler_steps / 20)) / 2 for rate in base]
        after = row['post_inner_scheduler_lrs']
        if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(after, expected)):
            raise ValueError('PROL inner cosine scheduler rates differ from the source')
        previous_rates = after


def _state_evidence(record, tasks):
    canonical = read_json(VALIDATION_PLAN)['common']['checkpoint']['canonical_pretrained_sha256']
    if (record.get('initial_canonical_pretrained_sha256') != canonical
            or record.get('initial_canonical_pretrained_branches') != {'query': canonical, 'prompt': canonical}):
        raise ValueError('PROL initial canonical branch tensors differ from the pin')
    states = record.get('state_audits', [])
    if len(states) != tasks:
        raise ValueError('PROL finalized task state audits are incomplete')
    initial_frozen = record.get('initial_frozen_encoder', {})
    frozen_parameters = record.get('initial_frozen_parameters_sha256')
    if (not _digest(frozen_parameters) or initial_frozen.get('query') != {'sha256': canonical, 'tensors': 150}
            or type(initial_frozen.get('prompt', {}).get('tensors')) is not int
            or initial_frozen['prompt']['tensors'] != 100 or not _digest(initial_frozen['prompt'].get('sha256'))):
        raise ValueError('PROL initial frozen encoder catalogue is incomplete')
    for task, row in enumerate(states, 1):
        _integer(row, 'task', task)
        _integer(row, 'dynamic_head_count', task)
        _integer(row, 'checkpoint_probe_examples', 40)
        if (row.get('query_canonical_sha256') != canonical
                or not strict_equal(row.get('frozen_encoder'), initial_frozen) or row.get('frozen_digest') != frozen_parameters
                or any(row.get(key) is not True for key in ('checkpoint_outputs_bitexact', 'checkpoint_rng_unchanged',
                        'frozen_non_norm_unchanged', 'query_unchanged', 'old_heads_unchanged',
                        'generator_frozen_from_task2', 'generator_unchanged_after_freeze', 'task_local_optimizer_reset'))
                or any(not _digest(row.get(key)) for key in ('learned_sha256', 'checkpoint_sha256', 'frozen_digest'))
                or type(row.get('compact_checkpoint_bytes')) is not int or row['compact_checkpoint_bytes'] <= 0):
            raise ValueError('PROL frozen state, task growth or checkpoint evidence is invalid')


def _cost_evidence(record, result_path, *, enforce_coverage=True):
    metrics = record['metrics']
    coverage = metrics.get('gpu_energy_coverage_fraction')
    if (not _finite(coverage) or coverage < 0 or coverage > 1.000001
            or enforce_coverage and (coverage < .99 or record.get('sensor_errors'))):
        raise ValueError('PROL cost samples require clean sensors and 99% energy coverage')
    for key in ('process_wall_s', 'process_tree_cpu_s', 'gpu_board_energy_wh', 'gpu_board_energy_j',
                'gpu_energy_coverage_s', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib'):
        if not _finite(metrics.get(key), positive=True):
            raise ValueError('PROL resource observation is missing or invalid')
    if not math.isclose(metrics['gpu_board_energy_j'] / 3600, metrics['gpu_board_energy_wh'], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('PROL energy J and Wh units disagree')
    if result_path is None:
        return
    path = Path(result_path).resolve().parent / 'telemetry.csv'
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    uuid = read_json(VALIDATION_PLAN)['common']['gpu_uuid']
    if len(rows) < 2 or any(row.get('gpu_uuid') != uuid for row in rows):
        raise ValueError('PROL telemetry physical GPU identity is incomplete or changed')
    samples = []
    for row in rows:
        try:
            elapsed, power = float(row['elapsed_s']), float(row['gpu_power_w'])
        except (KeyError, ValueError, TypeError):
            raise ValueError('PROL raw power observations are missing') from None
        if not _finite(elapsed) or elapsed < 0 or not _finite(power) or power < 0:
            raise ValueError('PROL raw power observations are invalid')
        samples.append({'elapsed_s': elapsed, 'gpu_power_w': power})
    if any(b['elapsed_s'] <= a['elapsed_s'] for a, b in zip(samples, samples[1:])):
        raise ValueError('PROL telemetry times must increase')
    interval = record.get('sampling_interval_s', 1.)
    if not _finite(interval, positive=True):
        raise ValueError('PROL sampling interval is invalid')
    energy, observed = integrate_power(samples, [(0., metrics['process_wall_s'])], max(6., interval * 3))
    if (energy is None or not math.isclose(energy, metrics['gpu_board_energy_j'], rel_tol=1e-9, abs_tol=1e-8)
            or not math.isclose(observed, metrics['gpu_energy_coverage_s'], rel_tol=1e-9, abs_tol=1e-8)
            or not math.isclose(observed / metrics['process_wall_s'], coverage, rel_tol=1e-9, abs_tol=1e-8)):
        raise ValueError('PROL recorded energy or coverage disagrees with raw power CSV')


def validate_record(record, name, *, expected_protocol=None, result_path=None):
    tasks, mode = (2, 'validation') if name.startswith('validation_') else (1, 'efficiency')
    side = 'author' if name.endswith('_author') else 'native'
    kind = 'flops' if name == 'flops' else 'measurement' if name.startswith('measurement') else 'validation'
    if (name not in JOB_NAMES or record.get('method') != 'prol-prompt' or record.get('model') != 'prol-prompt'
            or record.get('side') != side or record.get('mode') != mode or record.get('pass') != kind
            or record.get('implementation') != ('pinned_author_reference' if side == 'author' else 'mammoth_native')
            or record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded'
            or not str(record.get('wandb_url', '')).startswith(WAND_PREFIX)
            or record.get('measurement_aggregate_eligible') is not (kind == 'measurement')):
        raise ValueError('PROL worker identity, completion or upload is invalid')
    expected_protocol = shared.protocol(mode) if expected_protocol is None else expected_protocol
    if not strict_equal(record.get('protocol'), expected_protocol):
        raise ValueError('PROL worker protocol differs from the approved exact recipe')
    for key, value in {'tasks': tasks, 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                       'dataset_total_tasks': 10, 'classes_per_task': 10}.items():
        _integer(record['protocol'], key, value)
    checks = record.get('structural_checks', {})
    if any(checks.get(key) is not True for key in shared.REQUIRED_CHECKS) or any(value is not True for value in checks.values()):
        raise ValueError('All PROL structural checks must pass')
    common = read_json(VALIDATION_PLAN)['common']
    if (not strict_equal(record.get('precision'), common['precision'])
            or record.get('checkpoint_sha256') != common['checkpoint']['sha256']
            or record.get('canonical_pretrained_sha256') != common['checkpoint']['canonical_pretrained_sha256']
            or record.get('reference_commit') != SOURCE_COMMIT):
        raise ValueError('PROL source, checkpoint or explicit FP32 policy changed')
    for key, value in {'tasks_completed': tasks, 'outer_observe_steps': tasks * 79,
                       'outer_training_stream_examples': tasks * 5000, 'evaluation_examples': 3000 if tasks == 2 else 1000}.items():
        _integer(record.get('metrics', {}), key, value)
    for key in ('observe_calls', 'optimizer_updates'):
        _integer(record.get('counts', {}), key, tasks * 79)
    for key in ('task_begin_extra_samples', 'task_end_extra_samples'):
        _integer(record.get('counts', {}), key, 0)
    epochs = record.get('epoch_results', [])
    if len(epochs) != tasks or len(record.get('task_results', [])) != tasks:
        raise ValueError('PROL epoch or task stream is incomplete')
    for task, row in enumerate(epochs, 1):
        for key, value in {'task': task, 'epoch': 1, 'steps': 79, 'samples': 5000,
                           'seed': phase_seed(1993, task - 1, 'train', 0)}.items():
            _integer(row, key, value)
        if not _finite(row.get('loss_mean')):
            raise ValueError('PROL epoch loss must be finite')
    for task in range(1, tasks + 1):
        for split, size, per_class in (('train', 5000, 500), ('test', 1000, 100)):
            snapshot = record.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            _integer(snapshot, 'n_samples', size)
            counts = {str(label): per_class for label in range((task - 1) * 10, task * 10)}
            if not strict_equal(snapshot.get('class_counts'), counts) or any(not snapshot.get(key) for key in RAW_KEYS):
                raise ValueError('PROL raw data fingerprints are incomplete')
    observed = exact_task_metrics(record)
    if observed['task1_seen_accuracy'] < 90 or tasks == 2 and observed['task2_seen_accuracy'] < 20:
        raise ValueError('PROL accuracy floor failed; investigate implementation')
    if any(not _digest(record.get(key)) for key in ('initial_state_sha256', 'initial_learned_sha256')):
        raise ValueError('PROL initial-state fingerprints are invalid')
    expected_source = read_json(ROOT / 'benchmarks/cl_efficiency/prol_prompt/source_review.json')['reference_source_sha256']
    if record.get('reference_source_sha256') != expected_source:
        raise ValueError('PROL independent author source fingerprints differ from the reviewed pin')
    required_sources = ('scripts/prol_protocol.py', 'scripts/run_prol_worker.py', 'scripts/prol_reference_utils.py',
                        'scripts/run_prol_suite.py', 'scripts/report_prol_validation.py',
                        'benchmarks/cl_efficiency/prol_prompt/run_plan.json')
    execution_sources = record.get('execution_source_sha256', {})
    if any(source not in execution_sources for source in required_sources):
        raise ValueError('PROL execution source fingerprints are incomplete')
    for source, digest in execution_sources.items():
        path = (ROOT / source).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path) != digest:
            raise ValueError('PROL execution source changed')
    _stream_evidence(record, tasks)
    _state_evidence(record, tasks)
    _method_evidence(record, tasks)
    if result_path is not None:
        _predictions(record, result_path)
        _gradient_evidence(record, result_path)
    if kind == 'measurement':
        _cost_evidence(record, result_path)
    if kind == 'flops':
        value = record.get('flops', {}).get('supported_ops_total')
        if not _finite(value, positive=True) or record['flops'].get('partial') is not True:
            raise ValueError('PROL needs a separate positive partial FLOPs diagnostic')
        if result_path is not None:
            operations = json.loads((Path(result_path).parent / 'flops_ops.json').read_text())
            if (not isinstance(operations, list) or not operations
                    or any(not _finite(row.get('flops'), positive=True) for row in operations)
                    or sum(row['flops'] for row in operations) != value):
                raise ValueError('PROL partial FLOPs total disagrees with supported operator evidence')
    return observed


def compare_pair(reference, native, reference_path, native_path, minimum_accuracy_pct=90.):
    """Pure acceptance receipt from exact counts and actual saved evidence."""
    import numpy as np
    left, right = exact_task_metrics(reference), exact_task_metrics(native)
    tasks = len(reference['task_results'])
    if tasks != 2 or len(native['task_results']) != tasks:
        raise ValueError('PROL implementation validation requires both first two tasks')
    selected = ['task1_seen_accuracy', 'task2_seen_accuracy', 'task2_old_accuracy', 'task2_new_accuracy']
    metrics = []
    for key in selected:
        gap = left[key] - right[key]
        status = 'pass' if gap <= Fraction(3, 2) else 'review' if gap <= 3 else 'fail'
        if key == 'task1_seen_accuracy' and min(left[key], right[key]) < Fraction(str(minimum_accuracy_pct)):
            status = 'fail'
        if key == 'task2_seen_accuracy' and min(left[key], right[key]) < 20:
            status = 'fail'
        metrics.append({'metric': key, 'reference_pct': float(left[key]), 'native_pct': float(right[key]),
                        'drop_pp': float(gap), 'drop_exact': {'numerator': gap.numerator, 'denominator': gap.denominator},
                        'status': status})
    structural = all(record.get('structural_checks', {}).get(key) is True
                     for record in (reference, native) for key in shared.REQUIRED_CHECKS) and all(
                         value is True for record in (reference, native) for value in record.get('structural_checks', {}).values())
    checks = {key + '_matches': bool(reference.get(key)) and strict_equal(reference[key], native.get(key))
              for key in ('initial_state_sha256', 'initial_learned_sha256', 'data_files', 'sample_stream_audits')}
    batches_a, batches_b = reference.get('first_batch_audits', []), native.get('first_batch_audits', [])
    batch_keys = ('task', 'epoch', 'seed', 'samples', 'optimizer_update_index', 'input_sha256', 'labels_sha256',
                  'effective_lr', 'effective_lrs', 'rng_sha256', 'gradient_timing')
    checks['all_epoch_first_batch_inputs_match'] = len(batches_a) == 2 and len(batches_b) == 2 and all(
        all(a.get(key) is not None and strict_equal(a[key], b.get(key)) for key in batch_keys) for a, b in zip(batches_a, batches_b))
    for task in (1, 2):
        for kind in ('boundary_rng_before', 'task_begin'):
            left_boundary = [row for row in reference.get('method_audits', []) if row.get('task') == task and row.get('kind') == kind]
            right_boundary = [row for row in native.get('method_audits', []) if row.get('task') == task and row.get('kind') == kind]
            checks[f'task{task}_{kind}_matches'] = len(left_boundary) == len(right_boundary) == 1 and strict_equal(left_boundary, right_boundary)
    first_gradient = bool(batches_a) and bool(batches_b) and _digest(batches_a[0].get('gradient_sha256')) and (
        batches_a[0]['gradient_sha256'] == batches_b[0].get('gradient_sha256'))
    gradients = [{'task': a.get('task'), 'epoch': a.get('epoch'),
                  'gradient_sha256_matches': a.get('gradient_sha256') == b.get('gradient_sha256'),
                  'reference_gradient_sha256': a.get('gradient_sha256'), 'native_gradient_sha256': b.get('gradient_sha256')}
                 for a, b in zip(batches_a, batches_b)]
    for diagnostic, ((a, apin), (b, bpin)) in zip(gradients, zip(_gradient_evidence(reference, reference_path), _gradient_evidence(native, native_path))):
        matched_layout = set(a) == set(b) and all(a[key].shape == b[key].shape for key in a)
        diagnostic.update(reference_gradients=apin, native_gradients=bpin, tensor_layout_matches=matched_layout,
                          maximum_absolute_difference=max(float(np.max(np.abs(a[key].astype(np.float64) - b[key]))) for key in a) if matched_layout else None,
                          differing_elements=sum(int(np.count_nonzero(a[key] != b[key])) for key in a) if matched_layout else None)
        checks[f'task{diagnostic["task"]}_gradient_layout_matches'] = matched_layout
    for task in range(1, 3):
        for split in ('train', 'test'):
            a = reference.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            b = native.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            checks[f'task{task}_{split}_matches'] = all(a.get(key) is not None and strict_equal(a[key], b.get(key)) for key in RAW_KEYS)
    predictions = {}
    for task, ((a, apin), (b, bpin)) in enumerate(zip(_predictions(reference, reference_path), _predictions(native, native_path)), 1):
        row = {key + '_match': bool(np.array_equal(a[key], b[key])) for key in ('labels', 'global_ids', 'evaluation_global_ids', 'batch_sizes')}
        row.update(prediction_arrays_match=bool(np.array_equal(a['predictions'], b['predictions'])),
                   reference_predictions=apin, native_predictions=bpin)
        predictions[f'task{task}'] = row
        checks[f'task{task}_prediction_inputs_match'] = all(row[key + '_match'] for key in ('labels', 'global_ids', 'evaluation_global_ids', 'batch_sizes'))
    status = 'fail' if not structural or not all(checks.values()) or any(row['status'] == 'fail' for row in metrics) else (
        'review' if not first_gradient or any(row['status'] == 'review' for row in metrics) else 'pass')
    return {'status': status, 'accuracy_metrics': metrics,
            'thresholds': {'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.,
                           'first_task_minimum_accuracy_pct': minimum_accuracy_pct, 'task2_collapse_floor_pct': 20.},
            'structural_checks_pass': structural, 'input_checks': checks, 'prediction_checks': predictions,
            'reference_result': pin(reference_path), 'native_result': pin(native_path),
            'initial_first_batch_gradient_matches': first_gradient, 'gradient_diagnostics': gradients,
            'gradient_mismatch_action': 'review and diagnose before any further GPU runs',
            'final_learned_state_matches': [a.get('learned_sha256') == b.get('learned_sha256')
                for a, b in zip(reference.get('state_audits', []), native.get('state_audits', []))],
            'normality_review_required': True, 'measurement_aggregate_eligible': False}


def normality_diagnostics(reference, native):
    """Observed accuracy/loss evidence; 20% collapse guard is not adequacy."""
    evidence = {}
    for side, record in (('author', reference), ('native', native)):
        values = exact_task_metrics(record)
        if 'task2_seen_accuracy' not in values:
            raise ValueError('PROL normality review requires both tasks')
        evidence[side] = {'accuracy_pct': {key: float(value) for key, value in values.items()},
                          'old_accuracy_drop_after_task2_pp': float(values['task1_seen_accuracy'] - values['task2_old_accuracy']),
                          'epoch_results': record.get('epoch_results', []),
                          'first_batch_losses': [{key: batch[key] for key in ('task', 'loss', 'loss_ce',
                              'loss_diagonal', 'loss_redundancy') if key in batch} for batch in record.get('first_batch_audits', [])],
                          'lr_audits': record.get('lr_audits', []),
                          'source_mode': 'official-active-path', 'non_diagonal_formula': 'GM - GM @ I',
                          'cross_batch_similarity_assignment_preserved': True}
    return evidence


def review_normality(reference, native, reference_path, native_path, *, status, rationale, reviewed_by='codex'):
    if status not in ('pass', 'review', 'fail') or not isinstance(rationale, str) or not rationale.strip() or not reviewed_by:
        raise ValueError('PROL normality review requires an explicit decision and rationale')
    return {'schema_version': 1, 'status': status, 'reviewed_by': reviewed_by, 'rationale': rationale,
            'reference_result': pin(reference_path), 'native_result': pin(native_path),
            'diagnostics': normality_diagnostics(reference, native),
            'collapse_floor_is_not_normality_criterion': True,
            'review_scope': 'Assess old/new accuracy, loss, LR transitions and preserved source issues; no new accuracy floors'}


def validate_normality_review(saved, reference, native, reference_path, native_path):
    expected = review_normality(reference, native, reference_path, native_path, status=saved.get('status'),
                               rationale=saved.get('rationale'), reviewed_by=saved.get('reviewed_by'))
    if not strict_equal(saved, expected):
        raise ValueError('PROL normality review changed or does not match actual saved evidence')
    return expected


def _partial_report(run_root, execution):
    sources, diagnostics = [], []
    urls = set()
    expected_source = read_json(ROOT / 'benchmarks/cl_efficiency/prol_prompt/source_review.json')['reference_source_sha256']
    common = read_json(VALIDATION_PLAN)['common']
    for name in JOB_NAMES:
        path = run_root / name / 'result.json'
        if not path.is_file():
            continue
        record = read_json(path)
        mode, side = ('validation' if name.startswith('validation_') else 'efficiency'), ('author' if name.endswith('_author') else 'native')
        kind = 'flops' if name == 'flops' else 'measurement' if name.startswith('measurement') else 'validation'
        if (record.get('method') != 'prol-prompt' or record.get('model') != 'prol-prompt'
                or not str(record.get('wandb_url', '')).startswith(WAND_PREFIX) or record['wandb_url'] in urls
                or record.get('mode') != mode or record.get('side') != side or record.get('pass') != kind
                or record.get('status') not in ('completed', 'failed', 'stopped')
                or not strict_equal(record.get('protocol'), shared.protocol(mode))
                or record.get('reference_commit') != SOURCE_COMMIT or record.get('reference_source_sha256') != expected_source
                or record.get('checkpoint_sha256') != common['checkpoint']['sha256']
                or record.get('canonical_pretrained_sha256') != common['checkpoint']['canonical_pretrained_sha256']
                or not strict_equal(record.get('precision'), common['precision'])
                or Path(record.get('output', '')).resolve() != path.parent):
            raise ValueError('PROL partial evidence has invalid worker identity')
        urls.add(record['wandb_url'])
        pins = record.get('execution_source_sha256', {})
        if not pins or not strict_equal(pins, execution.get('execution_source_sha256')):
            raise ValueError('PROL partial execution source fingerprints are incomplete')
        for source, digest in pins.items():
            source_path = (ROOT / source).resolve()
            if not source_path.is_relative_to(ROOT) or not source_path.is_file() or file_hash(source_path) != digest:
                raise ValueError('PROL partial execution source changed')
        if record.get('initial_state_sha256') != execution.get('initial_state_sha256'):
            raise ValueError('PROL partial execution initialization changed')
        scheduled = [job for job in execution.get('jobs', []) if job.get('name') == name]
        if len(scheduled) != 1:
            raise ValueError('PROL partial worker is not part of the approved schedule')
        if scheduled[0].get('result'):
            check_pin(scheduled[0]['result'], path)
        observed = None
        if record.get('task_results'):
            observed = {key: float(value) for key, value in exact_task_metrics(record).items()}
            _predictions(record, path)
        metrics = record.get('metrics', {})
        for key in ('tasks_completed', 'outer_observe_steps', 'outer_training_stream_examples', 'evaluation_examples'):
            value = metrics.get(key)
            maximum = {'tasks_completed': record['protocol']['tasks'], 'outer_observe_steps': record['protocol']['tasks'] * 79,
                       'outer_training_stream_examples': record['protocol']['tasks'] * 5000,
                       'evaluation_examples': 3000 if mode == 'validation' else 1000}[key]
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError('PROL partial stream counts exceed the approved scope')
        if len(record.get('task_results', [])) > record['protocol']['tasks']:
            raise ValueError('PROL partial evaluated task scope exceeds the approved recipe')
        try:
            _cost_evidence(record, path, enforce_coverage=False)
            cost_audit = {'status': 'raw_power_verified', 'failed_or_incomplete_run': True,
                          'complete_measurement_eligible': False}
        except (ValueError, FileNotFoundError, KeyError):
            cost_audit = {'status': 'unverified_or_incomplete_sensor_evidence', 'failed_or_incomplete_run': True,
                          'complete_measurement_eligible': False}
        sources.append({'name': name, 'method': 'prol-prompt', 'pass': record['pass'], 'mode': record['mode'],
                        'side': record['side'], **pin(path), 'wandb_url': record['wandb_url'],
                        'execution_source_sha256': record.get('execution_source_sha256', {})})
        diagnostics.append({'name': name, 'status': record.get('status'), 'computation_status': record.get('computation_status'),
                            'accuracy_pct': observed, 'structural_checks': record.get('structural_checks', {}),
                            'error_type': record.get('error_type'), 'error_context': record.get('error_context'),
                            'metrics': metrics, 'phases': record.get('phases', {}),
                            'sensor_errors': record.get('sensor_errors', []), 'cost_audit': cost_audit,
                            'epoch_results': record.get('epoch_results', []),
                            'measurement_aggregate_eligible': False})
    if not sources:
        raise ValueError('PROL partial report requires saved worker evidence')
    receipts = {}
    for name in ('acceptance', 'validation_acceptance', 'normality_review'):
        path = run_root / (name + '.json')
        if path.is_file():
            receipts[name] = {'record': read_json(path), 'file': pin(path)}
    return {'schema_version': 1, 'status': 'stopped', 'publication_gate': 'blocked', 'method': 'prol-prompt',
            'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_root': str(run_root), 'new_gpu_training_runs': len(sources),
            'new_validation_runs': sum(source['pass'] == 'validation' for source in sources),
            'new_measurement_runs': sum(source['pass'] == 'measurement' for source in sources),
            'new_flops_runs': sum(source['pass'] == 'flops' for source in sources), 'sources': sources,
            'validation': {}, 'pilot': None, 'diagnostics': diagnostics, 'receipts': receipts,
            'execution': pin(run_root / 'execution.json'), 'report_generator': pin(__file__),
            'structural_check_count': len(shared.REQUIRED_CHECKS),
            'scope': {'source_mode': 'official-active-path', 'source_commit': SOURCE_COMMIT,
                      'paper_reproduction': False, 'broad_comparison': False, 'prior_nine_method_report_changed': False,
                      'failed_results_are_not_efficiency_averages': True,
                      'stop_reason': execution.get('stop_reason', execution.get('error_type', execution.get('status')))}}


def build_report(run_root=DEFAULT_RUN_ROOT):
    run_root = Path(run_root).resolve()
    execution_path = run_root / 'execution.json'
    execution = read_json(execution_path)
    if execution.get('status') != 'completed':
        return _partial_report(run_root, execution)
    plan = shared.load_run_plan()
    if (type(execution.get('completed_gpu_runs')) is not int or execution['completed_gpu_runs'] != 6
            or len(execution.get('jobs', [])) != 6 or execution.get('plan') != plan
            or Path(execution.get('output', '')).resolve() != run_root):
        raise ValueError('PROL report requires the exact approved six-run stage')
    from scripts.run_prol_suite import jobs
    first = execution['jobs'][0].get('command', [])
    if first.count('--initial-state') != 1:
        raise ValueError('PROL scheduled initialization is missing or ambiguous')
    try:
        initial_path = Path(first[first.index('--initial-state') + 1]).resolve()
    except (TypeError, IndexError):
        raise ValueError('PROL scheduled initialization is invalid') from None
    expected_jobs = jobs(run_root, initial_path)
    records, sources, urls = {}, [], set()
    for name, job, expected_job in zip(JOB_NAMES, execution['jobs'], expected_jobs):
        path = run_root / name / 'result.json'
        record = read_json(path)
        validate_record(record, name, result_path=path)
        if (job.get('status') != 'completed' or Path(job.get('directory', '')).resolve() != path.parent
                or Path(record.get('output', '')).resolve() != path.parent or record['wandb_url'] in urls
                or any(job.get(key) != expected_job[key] for key in ('name', 'side', 'mode', 'pass', 'directory', 'command'))
                or record['execution_source_sha256'] != execution.get('execution_source_sha256')
                or record['initial_state_sha256'] != execution.get('initial_state_sha256')):
            raise ValueError('PROL workers need distinct scheduled runs and exact frozen source/init')
        check_pin(job.get('result'), path)
        if job['result'].get('wandb_url') != record['wandb_url']:
            raise ValueError('PROL scheduled W&B identity changed')
        urls.add(record['wandb_url'])
        records[name] = record
        sources.append({'name': name, 'method': 'prol-prompt', 'pass': record['pass'], 'mode': record['mode'],
                        'side': record['side'], **pin(path), 'wandb_url': record['wandb_url'],
                        'execution_source_sha256': record['execution_source_sha256']})
    reference_path, native_path = run_root / 'validation_author/result.json', run_root / 'validation_native/result.json'
    receipt_path = run_root / 'acceptance.json'
    if not receipt_path.is_file():
        receipt_path = run_root / 'validation_acceptance.json'
    saved = read_json(receipt_path)
    expected = compare_pair(records['validation_author'], records['validation_native'], reference_path, native_path)
    if not strict_equal(saved, expected) or expected['status'] != 'pass':
        raise ValueError('PROL acceptance must match independently passing saved evidence')
    execution_acceptance = execution.get('acceptance', execution.get('validation_acceptance'))
    if not strict_equal(execution_acceptance, saved):
        raise ValueError('PROL execution and acceptance receipts disagree')
    normality_path = run_root / 'normality_review.json'
    normality = validate_normality_review(read_json(normality_path), records['validation_author'], records['validation_native'],
                                          reference_path, native_path)
    if normality['status'] != 'pass':
        raise ValueError('PROL low-performance normality review did not pass')
    first = records['validation_native']
    pilot_diagnostics = []
    for name in JOB_NAMES[2:]:
        record = records[name]
        for key in ('initial_state_sha256', 'initial_learned_sha256', 'data_files'):
            if not strict_equal(record.get(key), first.get(key)):
                raise ValueError('PROL efficiency changed common initialization or data')
        batch_keys = ('task', 'epoch', 'seed', 'samples', 'optimizer_update_index', 'input_sha256', 'labels_sha256',
                      'effective_lr', 'effective_lrs', 'rng_sha256', 'gradient_timing')
        observed_batch, expected_batch = record['first_batch_audits'][0], first['first_batch_audits'][0]
        if (not all(strict_equal(observed_batch.get(key), expected_batch.get(key)) for key in batch_keys)
                or not strict_equal(record.get('sample_stream_audits'), first.get('sample_stream_audits', [])[:1])):
            raise ValueError('PROL pilot changed first-task actual inputs, RNG or LR')
        for kind in ('boundary_rng_before', 'task_begin'):
            expected_boundary = [row for row in first['method_audits'] if row.get('task') == 1 and row.get('kind') == kind]
            observed_boundary = [row for row in record['method_audits'] if row.get('task') == 1 and row.get('kind') == kind]
            if not strict_equal(expected_boundary, observed_boundary):
                raise ValueError('PROL pilot changed first-task boundary RNG or parameter initialization')
        pilot_diagnostics.append({'name': name,
            'first_gradient_sha256_matches_validation_native': observed_batch['gradient_sha256'] == expected_batch['gradient_sha256'],
            'first_loss_matches_validation_native': observed_batch.get('loss') == expected_batch.get('loss'),
            'first_loss': observed_batch.get('loss'), 'validation_native_first_loss': expected_batch.get('loss'),
            'first_task_learned_state_matches_validation_native': record['state_audits'][0]['learned_sha256'] == first['state_audits'][0]['learned_sha256'],
            'interpretation': 'Numerical diagnostic under preserved duplicate CUDA indexing; exact input/RNG/LR evidence remains mandatory'})
        for split in ('train', 'test'):
            if not strict_equal(record['data_snapshots'][f'task1_{split}'], first['data_snapshots'][f'task1_{split}']):
                raise ValueError('PROL pilot changed first-task transforms or sample order')
    pilot = method_summary('prol-prompt', [records[name] for name in JOB_NAMES[2:5]], records['flops'],
                           protocol=records['measurement1']['protocol'])
    pilot.update(label='PROL (official active path)', changed_recipe=False, n_epochs=1, broad_comparison=False,
                 numerical_diagnostics=pilot_diagnostics)
    validation = {'online': {'acceptance': saved, 'acceptance_file': pin(receipt_path),
                             'normality_review': normality, 'normality_review_file': pin(normality_path),
                             'reference': sources[0], 'native': sources[1], 'aggregate_excluded': True}}
    return {'schema_version': 1, 'status': 'completed', 'publication_gate': 'pass', 'method': 'prol-prompt',
            'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'run_root': str(run_root),
            'new_gpu_training_runs': 6, 'new_validation_runs': 2, 'new_measurement_runs': 3, 'new_flops_runs': 1,
            'sources': sources, 'validation': validation, 'pilot': pilot, 'execution': pin(execution_path),
            'report_generator': pin(__file__), 'structural_check_count': len(shared.REQUIRED_CHECKS),
            'scope': {'source_mode': 'official-active-path', 'source_commit': SOURCE_COMMIT,
                      'paper_reproduction': False, 'trainable_backbone_norms': True,
                      'source_non_diagonal_formula': 'GM - GM @ I', 'source_cross_batch_assignment_preserved': True,
                      'source_eval_clamps_preserved': True, 'validation_tasks': 2, 'validation_epochs_per_task': 1,
                      'pilot_tasks': 1, 'pilot_epochs': 1, 'changed_pilot_recipe': False,
                      'broad_comparison': False, 'prior_nine_method_report_changed': False, 'equal_compute_budget': False,
                      'full_ten_tasks': False, 'same_seed_repeats': True, 'seed_uncertainty_estimated': False,
                      'validation_and_profiler_costs_excluded': True, 'full_machine_energy_measured': False,
                      'flops': 'partial supported-operator estimates'}}


def csv_text(report):
    rows = []
    for kind, validation in report['validation'].items():
        rows.extend({'kind': kind + '_validation', **row} for row in validation['acceptance']['accuracy_metrics'])
    if report['pilot'] is not None:
        for key, value in report['pilot']['resources'].items():
            rows.append({'kind': 'pilot_resource', 'metric': key, **value})
        rows.append({'kind': 'pilot_accuracy', 'metric': 'accuracy_pct', **report['pilot']['accuracy_pct']})
        for phase, values in report['pilot']['phases'].items():
            rows.extend({'kind': 'pilot_phase', 'phase': phase, 'metric': key, **stats} for key, stats in values.items())
        rows.append({'kind': 'separate_partial_flops', 'metric': 'supported_ops_total', 'value': report['pilot']['flops']['supported_ops_total']})
    else:
        rows.extend({'kind': 'stopped_diagnostic', **row} for row in report['diagnostics'])
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)
    return stream.getvalue()


def markdown(report):
    lines = ['# PROL 구현 검증과 첫 태스크 효율성 pilot', '',
             '고정한 공식 활성 코드를 공통 조건으로 검증합니다. LayerNorm 학습, 배치 간 indexing, 평가 중 clamp와 '
             '`GM - GM @ I` 비대각 항을 보존하며 논문 전체 성능 재현을 주장하지 않습니다.', '',
             'CIFAR100-224/ease, ViT-B/16 AugReg IN21K→IN1K, GPU1, batch64, seed1993, FP32입니다. '
             '검증과 pilot 모두 1epoch이며 nominal LR 0.01을 batch scaling하여 시작 LR 0.0025를 사용합니다.', '']
    if report['pilot'] is None:
        lines += ['**실행을 중단했습니다. 효율성 평균과 통과 판정을 생성하지 않습니다.**', '',
                  '| 실행 | 상태 | 관측 정확도 |', '|---|---|---|']
        lines.extend(f'| {row["name"]} | {row["status"]} | {json.dumps(row["accuracy_pct"], ensure_ascii=False)} |' for row in report['diagnostics'])
        lines += ['', '완료 또는 실패한 관측을 진단 자료로 보존했습니다. 추가 GPU 실행·재학습·LR 변경은 포함하지 않습니다.']
    else:
        lines += ['| 검증 | 지표 | 저자 % | native % | 하락 %p | 판정 |', '|---|---|---:|---:|---:|---|']
        for kind, validation in report['validation'].items():
            for row in validation['acceptance']['accuracy_metrics']:
                lines.append(f'| {kind} | {row["metric"]} | {row["reference_pct"]:.2f} | {row["native_pct"]:.2f} | {row["drop_pp"]:.2f} | {row["status"]} |')
        lines += ['', '저장된 예측과 정답 수의 정확한 분수로 정확도를 계산했습니다. 하락≤1.5%p pass, 1.5–3.0%p review, '
                  '3.0%p 초과 fail입니다. 첫 태스크 90% 하한·태스크2 seen 20% 붕괴 하한을 적용하고, '
                  'old/new 정확도·loss·LR를 별도로 점검했습니다. 20%는 정상 성능의 충분조건이 아닙니다.', '',
                  '| 첫 태스크 측정 | 평균 | 표본 SD |', '|---|---:|---:|']
        resource = report['pilot']['resources']
        for key in ('process_wall_s', 'process_tree_cpu_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib'):
            values = resource[key]
            lines.append(f'| {key} | {values["mean"]:.4f} | {values["std_sample"]:.4f} |')
        acc = report['pilot']['accuracy_pct']
        lines += [f'| accuracy_pct | {acc["mean"]:.2f} | {acc["std_sample"]:.2f} |', '',
                  f'별도 지원 연산 부분 FLOPs: **{report["pilot"]["flops"]["supported_ops_total"] / 1e12:.3f} TFLOPs**.', '',
                  '측정 3회의 평균·표본 SD에는 저자/native 검증 2회와 FLOPs 실행 비용을 포함하지 않았습니다. '
                  '동일 seed의 비용 변동이며 seed 간 성능 불확실성이 아닙니다.']
    lines += ['', 'GPU 보드 에너지는 idle 전력을 포함하며 CPU/RAM/전체 머신 에너지를 제외합니다. RSS는 부모·자식의 '
              '합으로 공유 페이지를 중복 집계할 수 있고 CUDA peak는 PyTorch allocator 범위입니다. '
              '부분 FLOPs는 미지원 연산·CPU 작업을 누락할 수 있습니다. 학습/평가/상태 점검 phase는 중첩될 수 있습니다.', '',
              '첫 태스크 pilot은 전체 CL 성능·망각이나 동일 계산 예산 비교를 입증하지 않습니다. '
              '기존 9방법 보고서와 비교 확대는 변경하지 않았습니다.', '', '| 실행 | W&B |', '|---|---|']
    lines.extend(f'| {source["name"]} | [run]({source["wandb_url"]}) |' for source in report['sources'])
    return '\n'.join(lines) + '\n'


def write_report(report, output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', report)
    (output / 'results.csv').write_text(csv_text(report))
    (output / 'RESULTS.md').write_text(markdown(report))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args(argv)
    report = build_report(options.run_root)
    if options.write:
        write_report(report, options.output)
    print(json.dumps({'publication_gate': report['publication_gate'], 'new_gpu_training_runs': report['new_gpu_training_runs'], 'written': options.write}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
