"""Self-contained saved-evidence tests; no model, data download or GPU work."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_rainbow_validation as report


@pytest.fixture
def stage(tmp_path, monkeypatch):
    run_root = tmp_path / 'runs'
    source_root = tmp_path / 'source'
    sources = {}
    for name in ('scripts/rainbow_protocol.py', 'scripts/run_rainbow_worker.py',
                 'scripts/rainbow_reference_utils.py', 'scripts/run_rainbow_suite.py',
                 'scripts/report_rainbow_validation.py', 'benchmarks/cl_efficiency/rainbow_prompt/run_plan.json'):
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic saved evidence fixture\n')
        sources[name] = report.file_hash(path)
    review = report.read_json(report.shared.REVIEW)
    review_path = source_root / 'benchmarks/cl_efficiency/rainbow_prompt/source_review.json'
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(report, 'ROOT', source_root)
    monkeypatch.setattr(report.shared, 'load_run_plan', lambda: {})
    common = report.read_json(report.VALIDATION_PLAN)['common']
    from scripts.run_rainbow_suite import jobs as make_jobs
    records, jobs = {}, make_jobs(run_root, tmp_path / 'initial.pt')
    for index, name in enumerate(report.JOB_NAMES):
        directory = run_root / name
        directory.mkdir(parents=True)
        full = name.startswith('full_')
        tasks, epochs = (2, 20) if full else (1, 1)
        mode = 'full_validation' if full else 'short_validation' if name.startswith('short_') else 'efficiency'
        side = 'author' if name.endswith('_author') else 'native'
        kind = 'flops' if name == 'flops' else 'measurement' if name.startswith('measurement') else 'validation'
        rows, predictions = [], []
        for task in range(1, tasks + 1):
            labels = np.repeat(np.arange(task * 10), 100)
            guessed = labels.copy()
            for partition in range(task):
                start = partition * 1000
                guessed[start:start + 10] = (guessed[start:start + 10] + 1) % (task * 10)
            global_ids = np.arange(task * 1000)
            path = directory / f'task{task}_predictions.npz'
            np.savez_compressed(path, predictions=guessed, labels=labels,
                                global_ids=global_ids, evaluation_global_ids=global_ids,
                                batch_sizes=np.asarray(([64] * 15 + [40]) * task), inferred_tasks=np.zeros(16 * task, dtype=np.int64))
            correct = int((guessed == labels).sum())
            row = {'task': task, 'seen_classes': task * 10, 'seen_correct': correct, 'seen_samples': task * 1000,
                   'old_correct': (task - 1) * 990, 'old_samples': (task - 1) * 1000,
                   'new_correct': 990, 'new_samples': 1000, 'seen_accuracy_pct': 99.,
                   'old_accuracy_pct': 99. if task > 1 else None, 'new_accuracy_pct': 99.}
            rows.append(row)
            predictions.append({'task': task, 'path': str(path), 'sha256': report.file_hash(path)})
        snapshots = {f'task{task}_{split}': {key: (5000 if split == 'train' else 1000) if key == 'n_samples'
                     else {str(label): 500 if split == 'train' else 100 for label in range((task - 1) * 10, task * 10)}
                     if key == 'class_counts' else key + str(task)
                     for key in report.RAW_KEYS} for task in range(1, tasks + 1) for split in ('train', 'test')}
        cost = (999999., 888888., 777777., 666666., 10., 20., 30., 555555.)[index]
        canonical = common['checkpoint']['canonical_pretrained_sha256']
        sha = '0' * 64
        cache_audits = []
        for task in range(1, tasks + 1):
            for epoch in range(1, epochs + 1):
                warmup = epoch <= (5 if full else 0)
                if not warmup:
                    cache_audits.append({'task': task, 'epoch': epoch, 'kind': 'first_rainbow_batch_pre_optimizer',
                                        'layers': list(range(12)), 'prefix_matches_pre_update_cache': True, 'pre_update_cache_sha256': sha})
                cache_audits.extend([{'task': task, 'epoch': epoch, 'kind': 'last_batch_cache_timing',
                                     'pre_post_optimizer_bitexact': True, 'cache_sha256': sha},
                                    {'task': task, 'epoch': epoch, 'kind': 'epoch_end', 'warmup_unique': warmup,
                                     'cache_untouched_during_warmup': warmup, 'cache_populated': not warmup,
                                     'old_rows_unchanged': True, 'cache_sha256': sha, 'old_state_sha256': sha}])
            cache_audits.append({'task': task, 'kind': 'task_end', 'no_cache_refresh': True, 'cache_sha256': sha})
        epoch_identity = [{'task': task, 'epoch': epoch, 'seed': report.shared.phase_seed(1993, task - 1, 'train', epoch - 1)}
                          for task in range(1, tasks + 1) for epoch in range(1, epochs + 1)]
        record = {'model': 'rainbow-prompt', 'method': 'rainbow-prompt', 'side': side, 'mode': mode, 'pass': kind,
                  'implementation': 'pinned_author_reference' if side == 'author' else 'mammoth_native',
                  'status': 'completed', 'computation_status': 'completed', 'wandb_logging_status': 'uploaded',
                  'wandb_url': report.WAND_PREFIX + name, 'output': str(directory),
                  'protocol': report.shared.protocol(mode), 'measurement_aggregate_eligible': kind == 'measurement',
                  'checkpoint_sha256': common['checkpoint']['sha256'],
                  'canonical_pretrained_sha256': common['checkpoint']['canonical_pretrained_sha256'],
                  'reference_commit': report.SOURCE_COMMIT, 'precision': common['precision'],
                  'initial_state_sha256': sha, 'initial_learned_sha256': '1' * 64,
                  'initial_canonical_pretrained_sha256': canonical,
                  'initial_canonical_pretrained_branches': {'query': canonical, 'prompt': canonical},
                  'reference_source_sha256': review['reference_source_sha256'],
                  'execution_source_sha256': sources, 'data_files': {'train': 'same', 'test': 'same'},
                  'data_snapshots': snapshots, 'first_batch_audits': [{**identity, 'samples': 64,
                    'input_sha256': sha, 'labels_sha256': sha, 'gradient_sha256': sha, 'gradient_timing': 'before_clip',
                    'effective_lr': .0075, 'optimizer_update_index': index * 79 + 1} for index, identity in enumerate(epoch_identity)],
                  'sample_stream_audits': [{**identity, 'samples': 5000, 'global_ids_sha256': sha,
                                           'labels_sha256': sha} for identity in epoch_identity],
                  'cache_audits': cache_audits,
                  'routing_audits': [{'task': task, 'evaluation_global_ids_sha256': sha, 'all_prefix_layers_match_selected_cache': True,
                    'batches': [{'test_task': index // 16 + 1, 'samples': length, 'selected_task': 0,
                                'batch_mean_scores': [1.] + [0.] * (task - 1), 'oracle_task_id': None}
                               for index, length in enumerate(([64] * 15 + [40]) * task)]} for task in range(1, tasks + 1)],
                  'structural_checks': {key: True for key in report.shared.REQUIRED_CHECKS},
                  'state_audits': [{'task': task, 'canonical_pretrained': {'query': canonical, 'prompt': canonical},
                    'learned_sha256': sha, 'checkpoint_sha256': sha, 'compact_checkpoint_bytes': 100,
                    'checkpoint_probe_examples': 40, **{key: True for key in ('checkpoint_outputs_bitexact',
                    'cached_pre_update_prompts_retained', 'old_state_unchanged', 'task_local_optimizer_reset',
                    'cache_populated_after_warmup', 'cache_pre_post_last_optimizer_bitexact', 'cache_untouched_during_warmup',
                    'no_end_task_cache_refresh', 'batch_mean_unknown_task_routing', 'phase_rng_epoch_seeds_verified')}} for task in range(1, tasks + 1)],
                  'counts': {'observe_calls': tasks * epochs * 79, 'optimizer_updates': tasks * epochs * 79,
                             'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}, 'accuracy': {'mean_accuracy_pct': 99.},
                  'task_results': rows, 'prediction_files': predictions,
                  'epoch_results': [{**identity, 'steps': 79, 'samples': 5000, 'loss_mean': 1.} for identity in epoch_identity],
                  'metrics': {'tasks_completed': tasks, 'outer_observe_steps': tasks * epochs * 79,
                    'outer_training_stream_examples': tasks * epochs * 5000, 'evaluation_examples': 3000 if full else 1000,
                    'process_wall_s': cost, 'process_tree_cpu_s': cost * 2, 'gpu_board_energy_wh': cost / 10,
                    'gpu_energy_coverage_fraction': 1., 'cuda_peak_allocated_mib': cost * 3,
                    'peak_process_tree_rss_mib': cost * 4}, 'phases': {'train_epoch': {'wall_s': cost - 1}},
                  'flops': {'supported_ops_total': 123456789., 'partial': True} if kind == 'flops' else {}}
        path = directory / 'result.json'
        report.write_json(path, record)
        for filename in ('spec.json', 'artifact_metrics.json', 'phases.json', 'epoch_results.json',
                         'task_results.json', 'first_batch_audits.json', 'state_audits.json', 'flops_ops.json',
                         'cache_audits.json', 'routing_audits.json', 'sample_stream_audits.json', 'effective_args.json',
                         'shared_data.json', 'source_adaptations.json'):
            (directory / filename).write_text(json.dumps(record if filename == 'artifact_metrics.json' else {}))
        (directory / 'telemetry.csv').write_text('elapsed_s,power_w\n0,100\n')
        jobs[index].update(status='completed', result={**report.pin(path), 'wandb_url': record['wandb_url']})
        records[name] = record
    acceptances = {}
    for kind in ('full', 'short'):
        acceptance = report.compare_pair(records[kind + '_author'], records[kind + '_native'],
            run_root / (kind + '_author') / 'result.json', run_root / (kind + '_native') / 'result.json')
        report.write_json(run_root / f'{kind}_acceptance.json', acceptance)
        acceptances[kind + '_acceptance'] = acceptance
    execution = {'status': 'completed', 'completed_gpu_runs': 8, 'output': str(run_root), 'jobs': jobs,
                 'execution_source_sha256': sources, 'initial_state_sha256': '0' * 64, 'plan': {}, **acceptances}
    report.write_json(run_root / 'execution.json', execution)
    return run_root, records, execution


def rewrite(stage, name, mutate):
    root, records, execution = stage
    mutate(records[name])
    path = root / name / 'result.json'
    report.write_json(path, records[name])
    execution['jobs'][report.JOB_NAMES.index(name)]['result'].update(report.pin(path))
    report.write_json(root / 'execution.json', execution)


def test_cost_statistics_exclude_all_validation_and_profiler_runs(stage):
    saved = report.build_report(stage[0])
    assert saved['new_gpu_training_runs'] == 8 and len(saved['sources']) == 8
    assert saved['pilot']['resources']['process_wall_s']['values'] == [10., 20., 30.]
    assert saved['pilot']['resources']['process_wall_s']['mean'] == 20.
    assert saved['pilot']['resources']['process_wall_s']['std_sample'] == 10.
    assert saved['pilot']['flops']['supported_ops_total'] == 123456789.
    assert saved['scope']['changed_pilot_recipe'] is True and saved['scope']['broad_comparison'] is False
    assert saved['scope']['learned_gate_active'] is False
    assert len(saved['validation']['full']['acceptance']['accuracy_metrics']) == 4
    assert len(saved['validation']['short']['acceptance']['accuracy_metrics']) == 1
    assert 'warm-up 0' in report.markdown(saved)
    assert 'pilot_resource' in report.csv_text(saved)


@pytest.mark.parametrize('key,value', [('tasks_completed', True), ('outer_observe_steps', 78),
                                      ('outer_training_stream_examples', 4992), ('evaluation_examples', 999),
                                      ('gpu_energy_coverage_fraction', .98), ('process_wall_s', 0.),
                                      ('gpu_board_energy_wh', -1.)])
def test_incomplete_stream_or_invalid_cost_cannot_enter_mean(stage, key, value):
    rewrite(stage, 'measurement1', lambda row: row['metrics'].update({key: value}))
    with pytest.raises(ValueError):
        report.build_report(stage[0])


def test_nonfinite_observation_is_rejected_before_aggregation(stage):
    record = deepcopy(stage[1]['measurement1'])
    record['metrics']['gpu_board_energy_wh'] = float('nan')
    with pytest.raises(ValueError):
        report.validate_record(record, 'measurement1')


@pytest.mark.parametrize('mutation', ['precision_type', 'missing_check', 'gate', 'warmup_bool', 'initial',
                                    'first_batch', 'source', 'upload', 'duplicate_run', 'flops_not_partial'])
def test_forged_recipe_provenance_or_gate_is_rejected(stage, mutation):
    def change(row):
        if mutation == 'precision_type':
            row['precision']['autocast'] = 0
        elif mutation == 'missing_check':
            row['structural_checks'].pop(report.shared.REQUIRED_CHECKS[-1])
        elif mutation == 'gate':
            row['protocol']['learned_gate_active'] = True
        elif mutation == 'warmup_bool':
            row['protocol']['warmup_epochs'] = False
        elif mutation == 'initial':
            row['initial_learned_sha256'] = 'wrong'
        elif mutation == 'first_batch':
            row['first_batch_audits'][0]['input_sha256'] = 'wrong'
        elif mutation == 'source':
            row['execution_source_sha256']['scripts/run_rainbow_worker.py'] = 'wrong'
        elif mutation == 'upload':
            row['wandb_logging_status'] = 'failed'
        elif mutation == 'duplicate_run':
            row['wandb_url'] = report.WAND_PREFIX + 'measurement2'
        else:
            row['flops']['partial'] = False
    rewrite(stage, 'flops' if mutation == 'flops_not_partial' else 'measurement1', change)
    with pytest.raises(ValueError):
        report.build_report(stage[0])


@pytest.mark.parametrize('field', ['status', 'accuracy_metrics', 'input_checks', 'thresholds'])
def test_saved_acceptance_cannot_override_independent_gate(stage, field):
    path = stage[0] / 'full_acceptance.json'
    gate = report.read_json(path)
    if field == 'status':
        gate[field] = 'review'
    elif field == 'accuracy_metrics':
        gate[field][0]['drop_pp'] = 1.
    elif field == 'input_checks':
        gate[field]['initial_state_sha256_matches'] = False
    else:
        gate[field]['first_task_minimum_accuracy_pct'] = 20.
    report.write_json(path, gate)
    with pytest.raises(ValueError):
        report.build_report(stage[0])


def test_prediction_integrity_and_evaluation_batch_order_are_required(stage):
    path = stage[0] / 'measurement1/task1_predictions.npz'
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays['evaluation_global_ids'][0] = arrays['evaluation_global_ids'][1]
    np.savez_compressed(path, **arrays)
    rewrite(stage, 'measurement1', lambda row: row['prediction_files'][0].update(sha256=report.file_hash(path)))
    with pytest.raises(ValueError):
        report.build_report(stage[0])


@pytest.mark.parametrize('mutation', ['canonical', 'warmup', 'pre_optimizer', 'last_optimizer', 'task_end', 'phase_seed', 'route'])
def test_true_structural_flags_cannot_hide_false_saved_state_evidence(stage, mutation):
    def change(row):
        if mutation == 'canonical':
            row['state_audits'][0]['canonical_pretrained']['query'] = 'wrong'
        elif mutation == 'warmup':
            next(item for item in row['cache_audits'] if item['kind'] == 'epoch_end')['cache_populated'] = True
        elif mutation == 'pre_optimizer':
            next(item for item in row['cache_audits'] if item['kind'] == 'first_rainbow_batch_pre_optimizer')['prefix_matches_pre_update_cache'] = False
        elif mutation == 'last_optimizer':
            next(item for item in row['cache_audits'] if item['kind'] == 'last_batch_cache_timing')['pre_post_optimizer_bitexact'] = False
        elif mutation == 'task_end':
            next(item for item in row['cache_audits'] if item['kind'] == 'task_end')['no_cache_refresh'] = False
        elif mutation == 'phase_seed':
            row['epoch_results'][0]['seed'] += 1
        else:
            row['routing_audits'][1]['batches'][0]['batch_mean_scores'] = [0., 1.]
    rewrite(stage, 'full_native', change)
    with pytest.raises(ValueError):
        report.build_report(stage[0])


@pytest.mark.parametrize('index,expected', [(0, 'fail'), (1, 'pass')])
def test_only_initial_gradient_is_a_gate_and_later_drift_remains_visible(stage, index, expected):
    root, records, _ = stage
    native = deepcopy(records['full_native'])
    native['first_batch_audits'][index]['gradient_sha256'] = 'f' * 64
    path = root / 'full_native/result.json'
    report.write_json(path, native)
    gate = report.compare_pair(records['full_author'], native, root / 'full_author/result.json', path)
    assert gate['status'] == expected
    assert gate['gradient_diagnostics'][index]['gradient_sha256_matches'] is False
    assert gate['input_checks']['initial_first_batch_gradient_matches'] is (index != 0)


@pytest.mark.parametrize('native_correct,expected', [(975, 'pass'), (974, 'review'), (960, 'review'), (959, 'fail')])
def test_exact_fraction_gates_have_inclusive_boundaries(stage, native_correct, expected):
    root, records, _ = stage
    native = deepcopy(records['short_native'])
    path = root / 'short_native/task1_predictions.npz'
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays['predictions'] = arrays['labels'].copy()
    wrong = 1000 - native_correct
    arrays['predictions'][:wrong] = (arrays['labels'][:wrong] + 1) % 10
    np.savez_compressed(path, **arrays)
    native['prediction_files'][0]['sha256'] = report.file_hash(path)
    native['task_results'][0].update(seen_correct=native_correct, new_correct=native_correct,
                                    seen_accuracy_pct=native_correct / 10, new_accuracy_pct=native_correct / 10)
    native['accuracy']['mean_accuracy_pct'] = native_correct / 10
    report.write_json(root / 'short_native/result.json', native)
    gate = report.compare_pair(records['short_author'], native,
                              root / 'short_author/result.json', root / 'short_native/result.json')
    assert gate['status'] == expected
