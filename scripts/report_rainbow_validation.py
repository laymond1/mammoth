"""Audit saved RainbowPrompt validation and its separate short-recipe pilot.

Reads small saved evidence only: no model construction, training, credentials,
or networking. This eight-run stage never widens the prior nine-method report.
"""
from __future__ import annotations

import argparse
import csv
import datetime
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.report_hide_results import (file_hash, method_summary,
                                         RAW_KEYS, read_json)
from scripts import rainbow_protocol as shared

DEFAULT_RUN_ROOT = ROOT / 'results/rainbow_validation_20261008'
DEFAULT_OUTPUT = ROOT / 'benchmarks/cl_efficiency/rainbow_validation'
VALIDATION_PLAN = ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/validation_plan.json'
REPORT_FILES = ('results.json', 'results.csv', 'RESULTS.md')
JOB_NAMES = ('full_author', 'full_native', 'short_author', 'short_native',
             'measurement1', 'measurement2', 'measurement3', 'flops')
SOURCE_COMMIT = '0b84d3e3738669fe4bb516ccc1c526a535576ad3'
WAND_PREFIX = 'https://wandb.ai/laymond1/cl-efficiency/runs/'


def pin(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError('Required RainbowPrompt evidence is missing or empty')
    return {'file': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def check_pin(proof, path):
    expected = pin(path)
    if (not isinstance(proof, dict) or Path(proof.get('file', '')).resolve() != Path(expected['file'])
            or proof.get('sha256') != expected['sha256']
            or 'bytes' in proof and (type(proof['bytes']) is not int or proof['bytes'] != expected['bytes'])):
        raise ValueError('RainbowPrompt evidence fingerprint changed')
    return expected


def _integer(values, key, expected):
    if type(values.get(key)) is not int or values[key] != expected:
        raise ValueError('RainbowPrompt requires exact complete stream and prediction counts')


def strict_equal(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(strict_equal(value[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(strict_equal(a, b) for a, b in zip(value, expected))
    return value == expected


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def exact_task_metrics(record):
    """Derive accuracy from exact saved integer counts, never rounded scores."""
    rows = record.get('task_results', [])
    if len(rows) not in (1, 2):
        raise ValueError('RainbowPrompt requires one or two complete saved tasks')
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
                raise ValueError('RainbowPrompt prediction counts are invalid')
            if samples:
                percent = Fraction(100 * correct, samples)
                if prefix + '_accuracy_pct' in row and row[prefix + '_accuracy_pct'] != float(percent):
                    raise ValueError('RainbowPrompt task accuracy disagrees with exact counts')
                result[f'task{task}_{prefix}_accuracy'] = percent
        if row['seen_correct'] != row['old_correct'] + row['new_correct']:
            raise ValueError('RainbowPrompt seen/old/new prediction totals disagree')
    final = result[f'task{len(rows)}_seen_accuracy']
    if record.get('accuracy', {}).get('mean_accuracy_pct') != float(final):
        raise ValueError('RainbowPrompt reported accuracy disagrees with final saved counts')
    return result


def _predictions(record, result_path):
    import numpy as np
    directory = Path(result_path).resolve().parent
    predictions = []
    saved_files = record.get('prediction_files', [])
    if len(saved_files) != len(record['task_results']):
        raise ValueError('Every RainbowPrompt task requires a pinned prediction artifact')
    for task, row in enumerate(record['task_results'], 1):
        path = directory / f'task{task}_predictions.npz'
        saved = saved_files[task - 1]
        if (saved.get('task') != task or type(saved.get('task')) is not int
                or Path(saved.get('path', saved.get('file', ''))).resolve() != path
                or saved.get('sha256') != file_hash(path)):
            raise ValueError('RainbowPrompt prediction artifact fingerprint changed')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in ('predictions', 'labels', 'global_ids', 'evaluation_global_ids',
                                                         'batch_sizes', 'inferred_tasks')}
        size = task * 1000
        if (any(arrays[key].shape != (size,) or not np.issubdtype(arrays[key].dtype, np.integer)
                for key in ('predictions', 'labels', 'global_ids', 'evaluation_global_ids'))
                or len(np.unique(arrays['global_ids'])) != size
                or np.any(arrays['global_ids'][1:] <= arrays['global_ids'][:-1])
                or not np.array_equal(np.sort(arrays['global_ids']), np.sort(arrays['evaluation_global_ids']))
                or np.any(arrays['labels'] < 0) or np.any(arrays['labels'] >= task * 10)
                or np.any(arrays['predictions'] < 0) or np.any(arrays['predictions'] >= task * 10)
                or np.any(np.bincount(arrays['labels'], minlength=task * 10) != 100)):
            raise ValueError('RainbowPrompt saved predictions or routing sample order are incomplete')
        batch_sizes = np.asarray(([64] * 15 + [40]) * task)
        if (not np.issubdtype(arrays['batch_sizes'].dtype, np.integer)
                or not np.array_equal(arrays['batch_sizes'], batch_sizes)
                or arrays['inferred_tasks'].shape != (16 * task,)
                or not np.issubdtype(arrays['inferred_tasks'].dtype, np.integer)
                or np.any(arrays['inferred_tasks'] < 0) or np.any(arrays['inferred_tasks'] >= task)):
            raise ValueError('RainbowPrompt evaluation batches changed task boundaries or routing tails')
        routing = record.get('routing_audits', [])
        if len(routing) != len(record['task_results']):
            raise ValueError('RainbowPrompt actual routing evidence is incomplete')
        route = routing[task - 1]
        batches = route.get('batches', [])
        if (route.get('task') != task or type(route.get('task')) is not int
                or route.get('all_prefix_layers_match_selected_cache') is not True
                or not _digest(route.get('evaluation_global_ids_sha256')) or len(batches) != 16 * task):
            raise ValueError('RainbowPrompt routing cache evidence is invalid')
        for index, (batch, length, inferred) in enumerate(zip(batches, batch_sizes, arrays['inferred_tasks'])):
            scores = batch.get('batch_mean_scores', [])
            if (type(batch.get('samples')) is not int or batch['samples'] != int(length)
                    or type(batch.get('selected_task')) is not int or batch['selected_task'] != int(inferred)
                    or type(batch.get('test_task')) is not int or batch['test_task'] != index // 16 + 1
                    or batch.get('oracle_task_id', 'missing') is not None or len(scores) != task
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores)
                    or int(np.argmax(scores)) != int(inferred)):
                raise ValueError('RainbowPrompt routing does not establish batch-mean selection without oracle labels')
        correct = arrays['predictions'] == arrays['labels']
        old = arrays['labels'] < (task - 1) * 10
        for key, value in {'seen_correct': int(correct.sum()), 'old_correct': int(correct[old].sum()),
                           'old_samples': int(old.sum()), 'new_correct': int(correct[~old].sum()),
                           'new_samples': int((~old).sum())}.items():
            _integer(row, key, value)
        predictions.append((arrays, pin(path)))
    return predictions


def _state_evidence(record, tasks, epochs):
    canonical = shared.CANONICAL_PRETRAINED_SHA256
    if (record.get('initial_canonical_pretrained_sha256') != canonical
            or record.get('initial_canonical_pretrained_branches') != {'query': canonical, 'prompt': canonical}):
        raise ValueError('RainbowPrompt initial query/prompt canonical tensors differ from the pin')
    states = record.get('state_audits', [])
    if len(states) != tasks:
        raise ValueError('RainbowPrompt finalized task state audits are incomplete')
    for task, row in enumerate(states, 1):
        if (row.get('task') != task or type(row.get('task')) is not int
                or row.get('canonical_pretrained') != {'query': canonical, 'prompt': canonical}
                or any(row.get(key) is not True for key in ('checkpoint_outputs_bitexact', 'cached_pre_update_prompts_retained',
                    'old_state_unchanged', 'task_local_optimizer_reset', 'cache_populated_after_warmup',
                    'cache_pre_post_last_optimizer_bitexact', 'cache_untouched_during_warmup',
                    'no_end_task_cache_refresh', 'batch_mean_unknown_task_routing', 'phase_rng_epoch_seeds_verified'))
                or type(row.get('checkpoint_probe_examples')) is not int or row['checkpoint_probe_examples'] != 40
                or not _digest(row.get('learned_sha256')) or not _digest(row.get('checkpoint_sha256'))
                or type(row.get('compact_checkpoint_bytes')) is not int or row['compact_checkpoint_bytes'] <= 0):
            raise ValueError('RainbowPrompt frozen encoders, cache retention or checkpoint probes are invalid')
    batches, streams = record.get('first_batch_audits', []), record.get('sample_stream_audits', [])
    if len(batches) != tasks * epochs or len(streams) != tasks * epochs:
        raise ValueError('RainbowPrompt per-epoch actual batches and stream RNG evidence are incomplete')
    cache = record.get('cache_audits', [])
    warmup = record['protocol']['warmup_epochs']
    expected_cache = []
    for task in range(1, tasks + 1):
        for epoch in range(1, epochs + 1):
            if epoch > warmup:
                expected_cache.append((task, epoch, 'first_rainbow_batch_pre_optimizer'))
            expected_cache.append((task, epoch, 'last_batch_cache_timing'))
            expected_cache.append((task, epoch, 'epoch_end'))
        expected_cache.append((task, None, 'task_end'))
    if len(cache) != len(expected_cache):
        raise ValueError('RainbowPrompt warmup and pre-optimizer cache audits are incomplete')
    for row, (task, epoch, kind) in zip(cache, expected_cache):
        expected_identity = {'task': task, 'kind': kind}
        if epoch is not None:
            expected_identity['epoch'] = epoch
        if any(row.get(key) != value or type(row.get(key)) is not type(value) for key, value in expected_identity.items()):
            raise ValueError('RainbowPrompt cache lifecycle order differs from the source')
        if kind == 'first_rainbow_batch_pre_optimizer':
            if (row.get('layers') != list(range(12)) or row.get('prefix_matches_pre_update_cache') is not True
                    or not _digest(row.get('pre_update_cache_sha256'))):
                raise ValueError('RainbowPrompt cache was not recorded before the actual optimizer step')
        elif kind == 'last_batch_cache_timing':
            if row.get('pre_post_optimizer_bitexact') is not True or not _digest(row.get('cache_sha256')):
                raise ValueError('RainbowPrompt last optimizer step refreshed the stored prompt cache')
        elif kind == 'task_end':
            if row.get('no_cache_refresh') is not True or not _digest(row.get('cache_sha256')):
                raise ValueError('RainbowPrompt task end changed the stored prompt cache')
        elif (row.get('warmup_unique') is not (epoch <= warmup)
                or row.get('cache_untouched_during_warmup') is not (epoch <= warmup)
                or row.get('cache_populated') is not (epoch > warmup) or row.get('old_rows_unchanged') is not True
                or not _digest(row.get('cache_sha256')) or not _digest(row.get('old_state_sha256'))):
            raise ValueError('RainbowPrompt cache warmup/population evidence differs from the source')
    for index, (batch, stream) in enumerate(zip(batches, streams)):
        task, epoch = index // epochs + 1, index % epochs + 1
        seed = shared.phase_seed(1993, task - 1, 'train', epoch - 1)
        for row in (batch, stream):
            for key, value in {'task': task, 'epoch': epoch, 'seed': seed}.items():
                _integer(row, key, value)
        _integer(batch, 'samples', 64)
        _integer(batch, 'optimizer_update_index', index * 79 + 1)
        _integer(stream, 'samples', 5000)
        if (batch.get('effective_lr') != .0075 or type(batch.get('effective_lr')) is not float
                or batch.get('gradient_timing') != 'before_clip'
                or any(not _digest(batch.get(key)) for key in ('input_sha256', 'labels_sha256', 'gradient_sha256'))
                or any(not _digest(stream.get(key)) for key in ('global_ids_sha256', 'labels_sha256'))):
            raise ValueError('RainbowPrompt actual batch LR, gradients or phase stream hashes are invalid')


def validate_record(record, name, *, expected_protocol=None):
    plan = read_json(VALIDATION_PLAN)
    full = name.startswith('full_')
    tasks, epochs = (2, 20) if full else (1, 1)
    mode = 'full_validation' if full else 'short_validation' if name.startswith('short_') else 'efficiency'
    side = 'author' if name.endswith('_author') else 'native'
    pass_kind = 'flops' if name == 'flops' else 'measurement' if name.startswith('measurement') else 'validation'
    if (name not in JOB_NAMES or record.get('method') != 'rainbow-prompt' or record.get('model') != 'rainbow-prompt'
            or record.get('side') != side or record.get('mode') != mode or record.get('pass') != pass_kind
            or record.get('implementation') != ('pinned_author_reference' if side == 'author' else 'mammoth_native')
            or record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded'
            or not str(record.get('wandb_url', '')).startswith(WAND_PREFIX)
            or record.get('measurement_aggregate_eligible') is not (pass_kind == 'measurement')):
        raise ValueError('RainbowPrompt worker identity, completion or upload is invalid')
    expected_protocol = shared.protocol(mode) if expected_protocol is None else expected_protocol
    if not strict_equal(record.get('protocol'), expected_protocol):
        raise ValueError('RainbowPrompt worker protocol differs from the approved exact recipe')
    protocol = record.get('protocol', {})
    for key, value in {'tasks': tasks, 'n_epochs': epochs, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                       'dataset_total_tasks': 10, 'classes_per_task': 10}.items():
        _integer(protocol, key, value)
    checks = record.get('structural_checks', {})
    required = shared.REQUIRED_CHECKS
    if any(checks.get(key) is not True for key in required) or any(value is not True for value in checks.values()):
        raise ValueError('All common and source-specific RainbowPrompt structural checks must pass')
    common = plan['common']
    checkpoint = common['checkpoint']
    precision = record.get('precision', {})
    if (set(precision) != set(common['precision'])
            or any(type(precision.get(key)) is not type(value) or precision[key] != value for key, value in common['precision'].items())
            or record.get('checkpoint_sha256') != checkpoint['sha256']
            or record.get('canonical_pretrained_sha256') != checkpoint['canonical_pretrained_sha256']
            or record.get('reference_commit') != SOURCE_COMMIT):
        raise ValueError('RainbowPrompt source, checkpoint or explicit FP32 policy changed')
    for key, value in {'tasks_completed': tasks, 'outer_observe_steps': tasks * epochs * 79,
                       'outer_training_stream_examples': tasks * epochs * 5000,
                       'evaluation_examples': 3000 if full else 1000}.items():
        _integer(record.get('metrics', {}), key, value)
    for key in ('observe_calls', 'optimizer_updates'):
        _integer(record.get('counts', {}), key, tasks * epochs * 79)
    for key in ('task_begin_extra_samples', 'task_end_extra_samples'):
        _integer(record.get('counts', {}), key, 0)
    rows = record.get('epoch_results', [])
    if len(rows) != tasks * epochs or len(record.get('task_results', [])) != tasks:
        raise ValueError('RainbowPrompt epoch or task stream is incomplete')
    for index, row in enumerate(rows):
        for key, value in {'task': index // epochs + 1, 'epoch': index % epochs + 1, 'steps': 79, 'samples': 5000}.items():
            _integer(row, key, value)
        _integer(row, 'seed', shared.phase_seed(1993, index // epochs, 'train', index % epochs))
        if type(row.get('loss_mean')) not in (int, float) or not math.isfinite(row['loss_mean']):
            raise ValueError('RainbowPrompt epoch loss must be finite')
    for task in range(1, tasks + 1):
        for split, size, per_class in (('train', 5000, 500), ('test', 1000, 100)):
            snapshot = record.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            _integer(snapshot, 'n_samples', size)
            expected_classes = {str(label): per_class for label in range((task - 1) * 10, task * 10)}
            if not strict_equal(snapshot.get('class_counts'), expected_classes) or any(not snapshot.get(key) for key in RAW_KEYS):
                raise ValueError('RainbowPrompt raw data class counts or fingerprints are incomplete')
    observed = exact_task_metrics(record)
    if observed['task1_seen_accuracy'] < 90 or any(value <= 20 for value in observed.values()):
        raise ValueError('RainbowPrompt accuracy is abnormally close to chance; investigate implementation')
    for key in ('initial_state_sha256', 'initial_learned_sha256', 'execution_source_sha256'):
        if not record.get(key):
            raise ValueError('RainbowPrompt initialization or execution provenance is missing')
    if any(not _digest(record.get(key)) for key in ('initial_state_sha256', 'initial_learned_sha256')):
        raise ValueError('RainbowPrompt initial-state fingerprints are invalid')
    expected_source = read_json(ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/source_review.json')['reference_source_sha256']
    if record.get('reference_source_sha256') != expected_source:
        raise ValueError('RainbowPrompt independent author source fingerprints differ from the reviewed pin')
    _state_evidence(record, tasks, epochs)
    required_sources = ('scripts/rainbow_protocol.py', 'scripts/run_rainbow_worker.py',
                        'scripts/rainbow_reference_utils.py', 'scripts/run_rainbow_suite.py',
                        'scripts/report_rainbow_validation.py', 'benchmarks/cl_efficiency/rainbow_prompt/run_plan.json')
    if any(source not in record['execution_source_sha256'] for source in required_sources):
        raise ValueError('RainbowPrompt execution source fingerprints are incomplete')
    for source, digest in record['execution_source_sha256'].items():
        path = (ROOT / source).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path) != digest:
            raise ValueError('RainbowPrompt execution source changed')
    if pass_kind == 'measurement':
        metrics = record['metrics']
        coverage = metrics.get('gpu_energy_coverage_fraction')
        if type(coverage) not in (int, float) or not math.isfinite(coverage) or coverage < .99 or record.get('sensor_errors'):
            raise ValueError('RainbowPrompt cost samples require clean sensors and 99% energy coverage')
        for key in ('process_wall_s', 'process_tree_cpu_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib'):
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('RainbowPrompt resource observation is missing or invalid')
    if pass_kind == 'flops':
        value = record.get('flops', {}).get('supported_ops_total')
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or record['flops'].get('partial') is not True:
            raise ValueError('RainbowPrompt needs a separate positive partial FLOPs diagnostic')
    return observed


def compare_pair(reference, native, reference_path, native_path, minimum_accuracy_pct=90.):
    """Pure receipt derivation from raw counts and artifacts; never writes it."""
    import numpy as np
    left, right = exact_task_metrics(reference), exact_task_metrics(native)
    tasks = len(reference['task_results'])
    if len(native['task_results']) != tasks:
        raise ValueError('RainbowPrompt author/native task scope differs')
    selected = ['task1_seen_accuracy'] + (['task2_seen_accuracy', 'task2_old_accuracy', 'task2_new_accuracy'] if tasks == 2 else [])
    metrics = []
    for key in selected:
        gap = left[key] - right[key]
        status = 'pass' if gap <= Fraction(3, 2) else 'review' if gap <= 3 else 'fail'
        floor = Fraction(str(minimum_accuracy_pct)) if key == 'task1_seen_accuracy' else Fraction(20)
        if min(left[key], right[key]) < floor or key != 'task1_seen_accuracy' and min(left[key], right[key]) == floor:
            status = 'fail'
        metrics.append({'metric': key, 'reference_pct': float(left[key]), 'native_pct': float(right[key]),
                        'drop_pp': float(gap), 'status': status})
    required = shared.REQUIRED_CHECKS
    structural = all(record.get('structural_checks', {}).get(key) is True for record in (reference, native) for key in required)
    checks = {}
    for key in ('initial_state_sha256', 'initial_learned_sha256', 'data_files', 'sample_stream_audits'):
        checks[key + '_matches'] = bool(reference.get(key)) and reference[key] == native.get(key)
    batches_a, batches_b = reference.get('first_batch_audits', []), native.get('first_batch_audits', [])
    batch_keys = ('task', 'epoch', 'seed', 'samples', 'optimizer_update_index', 'input_sha256', 'labels_sha256',
                  'effective_lr', 'gradient_timing')
    checks['all_epoch_first_batch_inputs_match'] = bool(batches_a) and len(batches_a) == len(batches_b) and all(
        all(a.get(key) is not None and strict_equal(a[key], b.get(key)) for key in batch_keys) for a, b in zip(batches_a, batches_b))
    checks['initial_first_batch_gradient_matches'] = bool(batches_a) and bool(batches_b) and (
        batches_a[0].get('task') == 1 and batches_a[0].get('epoch') == 1
        and batches_a[0].get('gradient_sha256') is not None
        and batches_a[0]['gradient_sha256'] == batches_b[0].get('gradient_sha256'))
    gradient_diagnostics = [{'task': a.get('task'), 'epoch': a.get('epoch'),
        'gradient_sha256_matches': a.get('gradient_sha256') == b.get('gradient_sha256')}
        for a, b in zip(batches_a, batches_b)]
    for task in range(1, tasks + 1):
        for split in ('train', 'test'):
            a = reference.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            b = native.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            checks[f'task{task}_{split}_matches'] = all(a.get(key) is not None and a[key] == b.get(key) for key in RAW_KEYS)
    predictions = {}
    source_artifacts, native_artifacts = _predictions(reference, reference_path), _predictions(native, native_path)
    for task, ((a, apin), (b, bpin)) in enumerate(zip(source_artifacts, native_artifacts), 1):
        row = {key + '_match': bool(np.array_equal(a[key], b[key])) for key in ('labels', 'global_ids', 'evaluation_global_ids')}
        row['prediction_arrays_match'] = bool(np.array_equal(a['predictions'], b['predictions']))
        row.update(reference_predictions=apin, native_predictions=bpin)
        predictions[f'task{task}'] = row
        checks[f'task{task}_prediction_inputs_match'] = all(row[key + '_match'] for key in ('labels', 'global_ids', 'evaluation_global_ids'))
    status = 'fail' if not structural or not all(checks.values()) or any(row['status'] == 'fail' for row in metrics) else (
        'review' if any(row['status'] == 'review' for row in metrics) else 'pass')
    return {'status': status, 'accuracy_metrics': metrics,
            'thresholds': {'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.,
                           'first_task_minimum_accuracy_pct': minimum_accuracy_pct, 'task2_collapse_floor_pct': 20.},
            'structural_checks_pass': structural, 'input_checks': checks, 'prediction_checks': predictions,
            'reference_result': pin(reference_path), 'native_result': pin(native_path),
            'gradient_diagnostics': gradient_diagnostics,
            'final_learned_state_matches': [a.get('learned_sha256') == b.get('learned_sha256')
                for a, b in zip(reference.get('state_audits', []), native.get('state_audits', []))],
            'measurement_aggregate_eligible': False}


def build_report(run_root=DEFAULT_RUN_ROOT):
    run_root = Path(run_root).resolve()
    execution_path = run_root / 'execution.json'
    execution = read_json(execution_path)
    plan = shared.load_run_plan()
    if (execution.get('status') != 'completed' or type(execution.get('completed_gpu_runs')) is not int
            or execution['completed_gpu_runs'] != 8 or len(execution.get('jobs', [])) != 8):
        raise ValueError('RainbowPrompt report requires the complete approved eight-run stage')
    if (execution.get('plan') != plan or Path(execution.get('output', '')).resolve() != run_root
            or not execution.get('initial_state_sha256') or not execution.get('execution_source_sha256')):
        raise ValueError('RainbowPrompt execution authorization or common initial/source evidence changed')
    from scripts.run_rainbow_suite import jobs
    first_command = execution['jobs'][0].get('command', [])
    if first_command.count('--initial-state') != 1:
        raise ValueError('RainbowPrompt scheduled initialization is missing or ambiguous')
    try:
        initial_path = Path(first_command[first_command.index('--initial-state') + 1]).resolve()
    except (IndexError, TypeError):
        raise ValueError('RainbowPrompt scheduled initialization is invalid') from None
    expected_jobs = jobs(run_root, initial_path)
    records, sources, urls = {}, [], set()
    for name, job, expected_job in zip(JOB_NAMES, execution['jobs'], expected_jobs):
        path = run_root / name / 'result.json'
        record = read_json(path)
        validate_record(record, name)
        if (job.get('status') != 'completed' or Path(job.get('directory', '')).resolve() != path.parent
                or Path(record.get('output', '')).resolve() != path.parent or record['wandb_url'] in urls
                or any(job.get(key) != expected_job[key] for key in ('name', 'side', 'mode', 'pass', 'directory'))
                or job.get('command') != expected_job['command']
                or record['execution_source_sha256'] != execution['execution_source_sha256']
                or record['initial_state_sha256'] != execution['initial_state_sha256']):
            raise ValueError('RainbowPrompt workers must have distinct scheduled runs and exact output directories')
        check_pin(job.get('result'), path)
        if job['result'].get('wandb_url') != record['wandb_url']:
            raise ValueError('RainbowPrompt scheduled W&B identity changed')
        urls.add(record['wandb_url'])
        _predictions(record, path)
        records[name] = record
        sources.append({'name': name, 'method': 'rainbow-prompt', 'pass': record['pass'], 'mode': record['mode'],
                        'side': record['side'], **pin(path), 'wandb_url': record['wandb_url'],
                        'execution_source_sha256': record['execution_source_sha256']})
    validation = {}
    for kind in ('full', 'short'):
        receipt_path = run_root / f'{kind}_acceptance.json'
        saved = read_json(receipt_path)
        expected = compare_pair(records[kind + '_author'], records[kind + '_native'],
                                run_root / (kind + '_author') / 'result.json', run_root / (kind + '_native') / 'result.json',
                                minimum_accuracy_pct=90.)
        if not strict_equal(saved, expected) or expected['status'] != 'pass':
            raise ValueError('RainbowPrompt saved acceptance must match independent passing evidence')
        if not strict_equal(execution.get(kind + '_acceptance'), saved):
            raise ValueError('RainbowPrompt execution and saved acceptance receipts disagree')
        validation[kind] = {'acceptance': saved, 'acceptance_file': pin(receipt_path),
                            'reference': next(source for source in sources if source['name'] == kind + '_author'),
                            'native': next(source for source in sources if source['name'] == kind + '_native'),
                            'aggregate_excluded': True}
    short = records['short_native']
    for name in JOB_NAMES[4:]:
        record = records[name]
        for key in ('protocol', 'initial_state_sha256', 'initial_learned_sha256', 'data_files', 'data_snapshots', 'first_batch_audits'):
            if key != 'protocol' and record.get(key) != short.get(key):
                raise ValueError('RainbowPrompt cost workers changed short-recipe inputs or initialization')
    pilot = method_summary('rainbow-prompt', [records[name] for name in JOB_NAMES[4:7]], records['flops'],
                           protocol=records['measurement1']['protocol'])
    pilot.update(label='RainbowPrompt (official active path; warm-up 0)', changed_recipe=True,
                 n_epochs=1, warmup_epochs=0, broad_comparison=False)
    return {'schema_version': 1, 'status': 'completed', 'publication_gate': 'pass', 'method': 'rainbow-prompt',
            'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_root': str(run_root), 'new_gpu_training_runs': 8, 'new_validation_runs': 4,
            'new_measurement_runs': 3, 'new_flops_runs': 1, 'sources': sources, 'validation': validation,
            'pilot': pilot, 'execution': pin(execution_path), 'report_generator': pin(__file__),
            'structural_check_count': len(shared.REQUIRED_CHECKS),
            'scope': {'source_mode': 'official-active-path', 'source_commit': SOURCE_COMMIT,
                      'learned_gate_active': False, 'gate_sparsity_loss_active': False, 'paper_reproduction': False,
                      'validation_tasks': 2, 'validation_epochs_per_task': 20, 'validation_warmup_epochs': 5,
                      'pilot_tasks': 1, 'pilot_epochs': 1, 'pilot_warmup_epochs': 0, 'changed_pilot_recipe': True,
                      'reference_recipe_adaptations': {'batch_size': 64, 'seed': 1993,
                          'seen_class_evaluation': True, 'common_actual_transforms': True},
                      'broad_comparison': False, 'prior_nine_method_report_changed': False,
                      'equal_compute_budget': False, 'full_ten_tasks': False, 'same_seed_repeats': True,
                      'seed_uncertainty_estimated': False, 'validation_and_profiler_costs_excluded': True,
                      'full_machine_energy_measured': False, 'flops': 'partial supported-operator estimates'}}


def csv_text(report):
    rows = []
    for kind, validation in report['validation'].items():
        rows.extend({'kind': kind + '_validation', **row} for row in validation['acceptance']['accuracy_metrics'])
    for key, value in report['pilot']['resources'].items():
        rows.append({'kind': 'pilot_resource', 'metric': key, **value})
    rows.append({'kind': 'pilot_accuracy', 'metric': 'accuracy_pct', **report['pilot']['accuracy_pct']})
    for phase, values in report['pilot']['phases'].items():
        for key, stats in values.items():
            rows.append({'kind': 'pilot_phase', 'phase': phase, 'metric': key, **stats})
    rows.append({'kind': 'separate_partial_flops', 'metric': 'supported_ops_total',
                 'value': report['pilot']['flops']['supported_ops_total']})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)
    return stream.getvalue()


def markdown(report):
    lines = ['# RainbowPrompt 구현 검증과 첫 태스크 효율성 pilot', '',
             '고정한 공식 코드의 활성 경로를 검증했습니다. 학습 가능한 gate와 sparsity loss는 연결되어 있지 않으며 논문 전체 성능 재현을 주장하지 않습니다.', '',
             'CIFAR100-224/ease, ViT-B/16 AugReg IN21K→IN1K, GPU 1, batch64, seed1993, FP32입니다. '
             '첫 2태스크 검증은 공식 20epochs·warm-up 5 일정을 유지하면서 batch·seed·전처리·seen-class 평가를 공통 조건으로 맞췄습니다. '
             '단축 첫 태스크와 pilot은 **1epoch·warm-up 0**인 별도 recipe입니다.', '',
             '| 검증 | 지표 | 저자 % | native % | 하락 %p | 판정 |', '|---|---|---:|---:|---:|---|']
    for kind, validation in report['validation'].items():
        for row in validation['acceptance']['accuracy_metrics']:
            lines.append(f'| {kind} | {row["metric"]} | {row["reference_pct"]:.2f} | {row["native_pct"]:.2f} | {row["drop_pp"]:.2f} | {row["status"]} |')
    resource = report['pilot']['resources']
    lines += ['', '정확도는 저장된 정답 수와 샘플 수의 정확한 분수로 계산했습니다. 하락≤1.5%p pass, 1.5–3.0%p review, '
              f'3.0%p 초과 fail이며 {report["structural_check_count"]}구조 검사·초기 상태·실제 배치·평가 batch 순서도 대조했습니다. '
              '첫 태스크의 저자/native 정확도 90% 하한과 태스크2의 20% 붕괴 탐지 하한을 적용했습니다. '
              '태스크2 하한은 충분한 CL 성능을 보증하는 기준이 아닙니다.', '',
              '| 첫 태스크 측정 | 평균 | 표본 SD |', '|---|---:|---:|']
    for key in ('process_wall_s', 'process_tree_cpu_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib'):
        values = resource[key]
        lines.append(f'| {key} | {values["mean"]:.4f} | {values["std_sample"]:.4f} |')
    lines += [f'| accuracy_pct | {report["pilot"]["accuracy_pct"]["mean"]:.2f} | {report["pilot"]["accuracy_pct"]["std_sample"]:.2f} |', '',
              f'별도 지원 연산 부분 FLOPs: **{report["pilot"]["flops"]["supported_ops_total"] / 1e12:.3f} TFLOPs**.', '',
              '측정 3회의 평균·표본 SD에 저자/native 검증 4회와 FLOPs 실행의 비용을 포함하지 않았습니다. '
              '동일 seed의 비용 변동이며 seed 간 성능 불확실성이 아닙니다. GPU 보드 에너지는 CPU/RAM/전체 머신 에너지를 포함하지 않습니다. '
              '부분 FLOPs는 미지원 연산과 CPU 작업을 누락할 수 있습니다.', '',
              '첫 태스크 pilot은 전체 CL 성능·망각이나 동일 계산 예산 비교를 입증하지 않습니다. '
              '신규 방법 구현 이후로 미룬 비교 확대와 기존 9방법 보고서는 이번 단계에 포함하지 않습니다.', '',
              '| 실행 | W&B |', '|---|---|']
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
    print(json.dumps({'publication_gate': report['publication_gate'], 'new_gpu_training_runs': 8, 'written': options.write}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
