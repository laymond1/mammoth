"""Saved-evidence validation tests: no models, downloads or GPU execution."""
from copy import deepcopy
import csv
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import report_prol_validation as report


@pytest.fixture
def stage(tmp_path, monkeypatch):
    run_root, source_root = tmp_path / 'runs', tmp_path / 'source'
    sources = {}
    for name in ('scripts/prol_protocol.py', 'scripts/run_prol_worker.py', 'scripts/prol_reference_utils.py',
                 'scripts/run_prol_suite.py', 'scripts/report_prol_validation.py',
                 'benchmarks/cl_efficiency/prol_prompt/run_plan.json'):
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic saved source evidence\n')
        sources[name] = report.file_hash(path)
    review = report.read_json(report.shared.REVIEW)
    review_path = source_root / 'benchmarks/cl_efficiency/prol_prompt/source_review.json'
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(report, 'ROOT', source_root)
    monkeypatch.setattr(report.shared, 'load_run_plan', lambda: {})
    def make_jobs(root, initial):
        return [{'name': name, 'side': 'author' if index == 0 else 'native',
                 'mode': 'validation' if index < 2 else 'efficiency',
                 'pass': 'validation' if index < 2 else 'measurement' if index < 5 else 'flops',
                 'directory': str(root / name), 'command': ['worker', '--initial-state', str(initial), '--name', name]}
                for index, name in enumerate(report.JOB_NAMES)]
    monkeypatch.setitem(sys.modules, 'scripts.run_prol_suite', SimpleNamespace(jobs=make_jobs))
    common = report.read_json(report.VALIDATION_PLAN)['common']
    records, jobs = {}, make_jobs(run_root, tmp_path / 'initial.pt')
    for index, name in enumerate(report.JOB_NAMES):
        directory = run_root / name
        directory.mkdir(parents=True)
        tasks = 2 if index < 2 else 1
        mode = 'validation' if index < 2 else 'efficiency'
        side, kind = jobs[index]['side'], jobs[index]['pass']
        rows, predictions = [], []
        for task in range(1, tasks + 1):
            labels = np.repeat(np.arange(task * 10), 100)
            guessed = labels.copy()
            for partition in range(task):
                start = partition * 1000
                guessed[start:start + 10] = (guessed[start:start + 10] + 1) % (task * 10)
            path = directory / f'task{task}_predictions.npz'
            global_ids = np.arange(task * 1000)
            np.savez_compressed(path, predictions=guessed, labels=labels, global_ids=global_ids,
                                evaluation_global_ids=global_ids, batch_sizes=np.asarray(([64] * 15 + [40]) * task))
            rows.append({'task': task, 'seen_classes': task * 10, 'seen_correct': task * 990,
                         'seen_samples': task * 1000, 'old_correct': (task - 1) * 990,
                         'old_samples': (task - 1) * 1000, 'new_correct': 990, 'new_samples': 1000,
                         'seen_accuracy_pct': 99., 'old_accuracy_pct': 99. if task > 1 else None, 'new_accuracy_pct': 99.})
            predictions.append({'task': task, 'path': str(path), 'sha256': report.file_hash(path)})
        snapshots = {f'task{task}_{split}': {key: (5000 if split == 'train' else 1000) if key == 'n_samples'
                     else {str(label): 500 if split == 'train' else 100 for label in range((task - 1) * 10, task * 10)}
                     if key == 'class_counts' else key + str(task) for key in report.RAW_KEYS}
                     for task in range(1, tasks + 1) for split in ('train', 'test')}
        cost = (99999., 88888., 10., 20., 30., 77777.)[index]
        sha, canonical = '0' * 64, common['checkpoint']['canonical_pretrained_sha256']
        frozen = {'query': {'sha256': canonical, 'tensors': 150}, 'prompt': {'sha256': sha, 'tensors': 100}}
        gradient = np.asarray([1.], dtype=np.float32)
        digest = hashlib.sha256()
        digest.update(b'prompt.synthetic')
        digest.update(str(tuple(gradient.shape)).encode())
        digest.update(b'torch.float32')
        digest.update(gradient.tobytes())
        gradient_sha = digest.hexdigest()
        for task in range(1, tasks + 1):
            np.savez_compressed(directory / f'task{task}_first_gradients.npz', **{'prompt.synthetic': gradient})
        methods = []
        for task in range(1, tasks + 1):
            params = {'trainable_parameters': 127098 if task == 1 else 126426,
                      'frozen_parameters': 171575828 if task == 1 else 171585726,
                      'total_parameters': 171702926 if task == 1 else 171712152,
                      'generator_parameters': 672, 'query_head_parameters': 16916, 'prompt_head_count': task}
            methods.extend([{'task': task, 'kind': 'boundary_rng_before', 'cpu_cuda_rng_sha256': sha},
                            {'task': task, 'kind': 'task_begin', 'head_count': task, 'generator_trainable': task == 1,
                             'previous_heads_frozen': True, 'old_key_change_allowed': True,
                             'cpu_cuda_rng_after_boundary_sha256': sha, 'parameters': params},
                            {'task': task, 'kind': 'first_batch_routing', 'samples': 64,
                             'cross_batch_similarity_assignment': True, 'unique_index_count': 1,
                             'duplicate_index_count': 63, 'column0_selected_count': 64, 'selected_local_indices_sha256': sha},
                            {'task': task, 'kind': 'outer_scheduler', 'last_epoch': 1,
                             'effective_lrs': [0.] * (1 if task == 1 else 2)}])
        identities = [{'task': task, 'epoch': 1, 'seed': report.phase_seed(1993, task - 1, 'train', 0)} for task in range(1, tasks + 1)]
        lr_audits = [{'task': task, 'epoch': 1, 'step': step, 'loss_ce': 1., 'threshold': .3, 'threshold_triggered': False, 'pre_step_lrs': [.0025] if task == 1 else [.0025, .00000125],
                      'post_optimizer_lrs': [.0025] if task == 1 else [.0025, .00000125],
                      'post_inner_scheduler_lrs': [.0025] if task == 1 else [.0025, .00000125]}
                     for task in range(1, tasks + 1) for step in range(1, 80)]
        states = [{'task': task, 'dynamic_head_count': task, 'checkpoint_probe_examples': 40,
                   'frozen_encoder': frozen, 'query_canonical_sha256': canonical, 'learned_sha256': sha, 'checkpoint_sha256': sha,
                   'frozen_digest': sha, 'compact_checkpoint_bytes': 100,
                   **{key: True for key in ('checkpoint_outputs_bitexact', 'checkpoint_rng_unchanged',
                       'frozen_non_norm_unchanged', 'query_unchanged', 'old_heads_unchanged',
                       'generator_frozen_from_task2', 'generator_unchanged_after_freeze', 'task_local_optimizer_reset')}}
                  for task in range(1, tasks + 1)]
        record = {'method': 'prol-prompt', 'model': 'prol-prompt', 'side': side, 'mode': mode, 'pass': kind,
                  'implementation': 'pinned_author_reference' if side == 'author' else 'mammoth_native',
                  'status': 'completed', 'computation_status': 'completed', 'wandb_logging_status': 'uploaded',
                  'wandb_url': report.WAND_PREFIX + name, 'output': str(directory), 'protocol': report.shared.protocol(mode),
                  'measurement_aggregate_eligible': kind == 'measurement',
                  'checkpoint_sha256': common['checkpoint']['sha256'], 'canonical_pretrained_sha256': canonical,
                  'reference_commit': report.SOURCE_COMMIT, 'precision': common['precision'],
                  'initial_state_sha256': sha, 'initial_learned_sha256': '1' * 64,
                  'initial_frozen_encoder': frozen, 'initial_frozen_parameters_sha256': sha,
                  'initial_canonical_pretrained_sha256': canonical,
                  'initial_canonical_pretrained_branches': {'query': canonical, 'prompt': canonical},
                  'reference_source_sha256': review['reference_source_sha256'], 'execution_source_sha256': sources,
                  'data_files': {'train': 'same', 'test': 'same'}, 'data_snapshots': snapshots,
                  'first_batch_audits': [{**identity, 'samples': 64, 'input_sha256': sha, 'labels_sha256': sha,
                       'gradient_sha256': gradient_sha, 'rng_sha256': sha,
                       'gradient_file': f'task{identity["task"]}_first_gradients.npz',
                       'gradient_file_sha256': report.file_hash(directory / f'task{identity["task"]}_first_gradients.npz'), 'gradient_timing': 'before_clip', 'effective_lr': .0025,
                       'effective_lrs': [.0025] if identity['task'] == 1 else [.0025, .00000125],
                       'optimizer_update_index': i * 79 + 1, 'loss': 1., 'loss_ce': 1.}
                       for i, identity in enumerate(identities)],
                  'sample_stream_audits': [{**identity, 'samples': 5000, 'global_ids_sha256': sha, 'labels_sha256': sha}
                                          for identity in identities],
                  'state_audits': states, 'lr_audits': lr_audits, 'method_audits': methods, 'routing_audits': [{'task': task, 'evaluation_global_ids_sha256': sha,
                       'all_active_prefix_layers_routed': True, 'cross_batch_similarity_assignment': True,
                       'evaluation_affine_clamp_preserved': True,
                       'batches': [{'test_task': i // 16 + 1, 'samples': length, 'oracle_task_id': None,
                         'labels_provided': False, 'routing_key_count': task * 10, 'selected_local_indices_sha256': sha,
                         'unique_index_count': 1, 'duplicate_index_count': length - 1, 'column0_selected_count': length}
                         for i, length in enumerate(([64] * 15 + [40]) * task)]} for task in range(1, tasks + 1)],
                  'structural_checks': {key: True for key in report.shared.REQUIRED_CHECKS},
                  'counts': {'observe_calls': tasks * 79, 'optimizer_updates': tasks * 79,
                             'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
                  'accuracy': {'mean_accuracy_pct': 99.}, 'task_results': rows, 'prediction_files': predictions,
                  'epoch_results': [{**identity, 'steps': 79, 'samples': 5000, 'loss_mean': 1.} for identity in identities],
                  'metrics': {'tasks_completed': tasks, 'outer_observe_steps': tasks * 79,
                       'outer_training_stream_examples': tasks * 5000, 'evaluation_examples': 3000 if tasks == 2 else 1000,
                       'process_wall_s': cost, 'process_tree_cpu_s': cost * 2, 'gpu_board_energy_wh': cost / 10,
                       'gpu_board_energy_j': cost * 360, 'gpu_energy_coverage_s': cost, 'gpu_energy_coverage_fraction': 1.,
                       'cuda_peak_allocated_mib': cost * 3, 'peak_process_tree_rss_mib': cost * 4},
                  'phases': {'train_epoch': {'wall_s': cost - 1}}, 'sensor_errors': [],
                  'flops': {'supported_ops_total': 123456789., 'partial': True} if kind == 'flops' else {}}
        path = directory / 'result.json'
        report.write_json(path, record)
        for filename in ('spec.json', 'artifact_metrics.json', 'phases.json', 'epoch_results.json', 'task_results.json',
                         'first_batch_audits.json', 'state_audits.json', 'lr_audits.json', 'method_audits.json',
                         'routing_audits.json', 'sample_stream_audits.json', 'effective_args.json', 'shared_data.json', 'source_adaptations.json'):
            (directory / filename).write_text(json.dumps(record if filename == 'artifact_metrics.json' else {}))
        (directory / 'flops_ops.json').write_text(json.dumps([{'operation': 'aten::mm', 'count': 1, 'flops': 123456789.}]))
        # Only timing workers need realistic power trace; large validation costs
        # deliberately test exclusion from every efficiency mean.
        with (directory / 'telemetry.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['elapsed_s', 'gpu_power_w', 'gpu_uuid'])
            writer.writeheader()
            for elapsed in range(int(cost) + 1 if kind == 'measurement' else 2):
                writer.writerow({'elapsed_s': elapsed, 'gpu_power_w': 360., 'gpu_uuid': common['gpu_uuid']})
        jobs[index].update(status='completed', result={**report.pin(path), 'wandb_url': record['wandb_url']})
        records[name] = record
    reference_path, native_path = run_root / 'validation_author/result.json', run_root / 'validation_native/result.json'
    acceptance = report.compare_pair(records['validation_author'], records['validation_native'], reference_path, native_path)
    normality = report.review_normality(records['validation_author'], records['validation_native'], reference_path, native_path,
                                      status='pass', rationale='Synthetic fixture has 99% old/new and finite loss with complete LR audits.')
    report.write_json(run_root / 'acceptance.json', acceptance)
    report.write_json(run_root / 'normality_review.json', normality)
    execution = {'status': 'completed', 'completed_gpu_runs': 6, 'output': str(run_root), 'jobs': jobs,
                 'execution_source_sha256': sources, 'initial_state_sha256': '0' * 64, 'plan': {}, 'acceptance': acceptance}
    report.write_json(run_root / 'execution.json', execution)
    return run_root, records, execution


def rewrite(stage, name, mutate):
    root, records, execution = stage
    mutate(records[name])
    path = root / name / 'result.json'
    report.write_json(path, records[name])
    execution['jobs'][report.JOB_NAMES.index(name)]['result'].update(report.pin(path))
    report.write_json(root / 'execution.json', execution)


def test_only_three_native_measurements_enter_cost_averages(stage):
    saved = report.build_report(stage[0])
    assert saved['new_gpu_training_runs'] == 6 and len(saved['sources']) == 6
    assert saved['pilot']['resources']['process_wall_s']['values'] == [10., 20., 30.]
    assert saved['pilot']['resources']['process_wall_s']['mean'] == 20.
    assert saved['pilot']['resources']['process_wall_s']['std_sample'] == 10.
    assert saved['pilot']['flops']['supported_ops_total'] == 123456789.
    assert saved['scope']['changed_pilot_recipe'] is False and saved['scope']['broad_comparison'] is False
    assert len(saved['validation']['online']['acceptance']['accuracy_metrics']) == 4
    assert '20%' in report.markdown(saved)
    assert 'pilot_resource' in report.csv_text(saved)


@pytest.mark.parametrize('key,value', [('tasks_completed', True), ('outer_observe_steps', 78),
    ('outer_training_stream_examples', 4992), ('evaluation_examples', 999), ('gpu_energy_coverage_fraction', .98),
    ('process_wall_s', 0.), ('gpu_board_energy_wh', -1.), ('gpu_board_energy_j', -1.)])
def test_invalid_counts_resources_or_sensors_cannot_enter_mean(stage, key, value):
    rewrite(stage, 'measurement1', lambda row: row['metrics'].update({key: value}))
    with pytest.raises(ValueError):
        report.build_report(stage[0])


@pytest.mark.parametrize('mutation', ['precision', 'missing_check', 'source', 'rng', 'lrs', 'upload', 'duplicate', 'flops'])
def test_changed_policy_source_first_batch_or_upload_blocks_report(stage, mutation):
    def change(row):
        if mutation == 'precision':
            row['precision']['autocast'] = 0
        elif mutation == 'missing_check':
            row['structural_checks'].pop(report.shared.REQUIRED_CHECKS[-1])
        elif mutation == 'source':
            row['execution_source_sha256']['scripts/run_prol_worker.py'] = 'wrong'
        elif mutation == 'rng':
            row['first_batch_audits'][0]['rng_sha256'] = '2' * 64
        elif mutation == 'lrs':
            row['first_batch_audits'][0]['effective_lrs'] = [.01]
        elif mutation == 'upload':
            row['wandb_logging_status'] = 'failed'
        elif mutation == 'duplicate':
            row['wandb_url'] = report.WAND_PREFIX + 'measurement2'
        else:
            row['flops']['partial'] = False
    rewrite(stage, 'flops' if mutation == 'flops' else 'measurement1', change)
    with pytest.raises(ValueError):
        report.build_report(stage[0])


@pytest.mark.parametrize('mutation', ['power', 'uuid', 'time', 'energy_units'])
def test_raw_csv_energy_and_gpu_identity_are_verified(stage, mutation):
    path = stage[0] / 'measurement1/telemetry.csv'
    if mutation == 'energy_units':
        rewrite(stage, 'measurement1', lambda row: row['metrics'].update(gpu_board_energy_wh=2.))
    else:
        text = path.read_text()
        if mutation == 'power':
            text = text.replace('360.0', '361.0')
        elif mutation == 'uuid':
            text = text.replace('GPU-', 'OTHER-')
        else:
            text = text.replace('1,360.0', '0,360.0', 1)
        path.write_text(text)
    with pytest.raises(ValueError):
        report.build_report(stage[0])


def test_flops_sum_cannot_be_manufactured(stage):
    path = stage[0] / 'flops/flops_ops.json'
    path.write_text(json.dumps([{'flops': 1.}]))
    with pytest.raises(ValueError, match='FLOPs total'):
        report.build_report(stage[0])


def test_normality_review_is_explicit_and_pinned(stage):
    path = stage[0] / 'normality_review.json'
    saved = report.read_json(path)
    saved['diagnostics']['native']['old_accuracy_drop_after_task2_pp'] = 0.1
    report.write_json(path, saved)
    with pytest.raises(ValueError, match='normality review'):
        report.build_report(stage[0])


def test_normality_review_cannot_claim20_floor_is_adequacy(stage):
    path = stage[0] / 'normality_review.json'
    saved = report.read_json(path)
    saved['status'] = 'review'
    report.write_json(path, saved)
    with pytest.raises(ValueError, match='normality review did not pass'):
        report.build_report(stage[0])


def test_first_gradient_difference_requires_review_not_silent_pass(stage):
    left, right = stage[1]['validation_author'], deepcopy(stage[1]['validation_native'])
    gradient = np.asarray([2.], dtype=np.float32)
    path = stage[0] / 'validation_native/task1_first_gradients.npz'
    np.savez_compressed(path, **{'prompt.synthetic': gradient})
    digest = hashlib.sha256()
    digest.update(b'prompt.synthetic')
    digest.update(str(tuple(gradient.shape)).encode())
    digest.update(b'torch.float32')
    digest.update(gradient.tobytes())
    right['first_batch_audits'][0].update(gradient_sha256=digest.hexdigest(), gradient_file_sha256=report.file_hash(path))
    receipt = report.compare_pair(left, right, stage[0] / 'validation_author/result.json', stage[0] / 'validation_native/result.json')
    assert receipt['status'] == 'review'
    assert receipt['initial_first_batch_gradient_matches'] is False
    assert receipt['input_checks']['all_epoch_first_batch_inputs_match'] is True


def test_first_rng_difference_is_hard_failure(stage):
    left, right = stage[1]['validation_author'], deepcopy(stage[1]['validation_native'])
    right['first_batch_audits'][0]['rng_sha256'] = '2' * 64
    receipt = report.compare_pair(left, right, stage[0] / 'validation_author/result.json', stage[0] / 'validation_native/result.json')
    assert receipt['status'] == 'fail'


@pytest.mark.parametrize('native_correct,expected', [(975, 'pass'), (974, 'review'), (960, 'review'), (959, 'fail')])
def test_exact_thresholds_do_not_use_rounding(stage, native_correct, expected):
    left, right = deepcopy(stage[1]['validation_author']), deepcopy(stage[1]['validation_native'])
    # Compare exact metrics directly by matching first-task raw prediction counts.
    directory = stage[0] / 'validation_native'
    path = directory / 'task1_predictions.npz'
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays['predictions'] = arrays['labels'].copy()
    wrong = 1000 - native_correct
    arrays['predictions'][:wrong] = (arrays['predictions'][:wrong] + 1) % 10
    np.savez_compressed(path, **arrays)
    right['prediction_files'][0]['sha256'] = report.file_hash(path)
    right['task_results'][0].update(seen_correct=native_correct, new_correct=native_correct,
                                         seen_accuracy_pct=native_correct / 10, new_accuracy_pct=native_correct / 10)
    receipt = report.compare_pair(left, right, stage[0] / 'validation_author/result.json', stage[0] / 'validation_native/result.json')
    assert receipt['accuracy_metrics'][0]['status'] == expected
    assert receipt['status'] == expected


def test_20_percent_seen_is_only_collapse_floor_not_partition_floor(stage):
    # The strict floor applies to seen accuracy; no invented old/new cutoffs.
    left, right = deepcopy(stage[1]['validation_author']), deepcopy(stage[1]['validation_native'])
    for record, name in ((left, 'validation_author'), (right, 'validation_native')):
        path = stage[0] / name / 'task2_predictions.npz'
        with np.load(path) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays['predictions'] = (arrays['labels'] + 1) % 20
        arrays['predictions'][1000:1400] = arrays['labels'][1000:1400]
        np.savez_compressed(path, **arrays)
        record['prediction_files'][1]['sha256'] = report.file_hash(path)
        record['task_results'][1].update(seen_correct=400, old_correct=0, new_correct=400,
                                          seen_accuracy_pct=20., old_accuracy_pct=0., new_accuracy_pct=40.)
        record['accuracy']['mean_accuracy_pct'] = 20.
    receipt = report.compare_pair(left, right, stage[0] / 'validation_author/result.json', stage[0] / 'validation_native/result.json')
    assert receipt['status'] == 'pass'
    assert receipt['normality_review_required'] is True
    assert report.normality_diagnostics(left, right)['native']['old_accuracy_drop_after_task2_pp'] == 99.


def test_partial_guard_report_keeps_diagnostics_without_efficiency_mean(stage):
    root, records, execution = stage
    execution.update(status='stopped', stop_reason='author first task below 90%', completed_gpu_runs=1)
    report.write_json(root / 'execution.json', execution)
    for name in report.JOB_NAMES[1:]:
        (root / name / 'result.json').unlink()
    rewrite(stage, 'validation_author', lambda row: row.update(status='failed', computation_status='failed'))
    saved = report.build_report(root)
    assert saved['status'] == 'stopped' and saved['publication_gate'] == 'blocked'
    assert saved['pilot'] is None and saved['new_gpu_training_runs'] == 1
    assert 'stopped_diagnostic' in report.csv_text(saved)
    assert '효율성 평균' in report.markdown(saved)


@pytest.mark.parametrize('kind,key', [('boundary_rng_before', 'cpu_cuda_rng_sha256'),
                                      ('task_begin', 'cpu_cuda_rng_after_boundary_sha256')])
def test_task2_boundary_rng_difference_cannot_be_masked_by_train_reseed(stage, kind, key):
    left, right = stage[1]['validation_author'], deepcopy(stage[1]['validation_native'])
    row = next(row for row in right['method_audits'] if row['task'] == 2 and row['kind'] == kind)
    row[key] = '2' * 64
    receipt = report.compare_pair(left, right, stage[0] / 'validation_author/result.json', stage[0] / 'validation_native/result.json')
    assert receipt['status'] == 'fail'
    assert receipt['input_checks']['all_epoch_first_batch_inputs_match'] is True


def test_missing_head_boundary_audit_blocks_report(stage):
    rewrite(stage, 'measurement1', lambda row: row.update(method_audits=[]))
    with pytest.raises(ValueError, match='boundary RNG'):
        report.build_report(stage[0])


def test_scheduler_sticky_trigger_is_checked_from_actual_ce(stage):
    rewrite(stage, 'measurement1', lambda row: row['lr_audits'][5].update(loss_ce=.1))
    with pytest.raises(ValueError, match='sticky threshold'):
        report.build_report(stage[0])


def test_frozen_state_hash_cannot_be_waived_by_boolean_checks(stage):
    rewrite(stage, 'measurement1', lambda row: row['state_audits'][0].update(frozen_digest='2' * 64))
    with pytest.raises(ValueError, match='frozen state'):
        report.build_report(stage[0])


def test_gradient_arrays_must_match_actual_record_digest(stage):
    path = stage[0] / 'measurement1/task1_first_gradients.npz'
    np.savez_compressed(path, **{'prompt.synthetic': np.asarray([9.], dtype=np.float32)})
    rewrite(stage, 'measurement1', lambda row: row['first_batch_audits'][0].update(gradient_file_sha256=report.file_hash(path)))
    with pytest.raises(ValueError, match='gradient SHA'):
        report.build_report(stage[0])


def test_pilot_loss_difference_is_explicit_diagnostic_under_identical_recipe(stage):
    rewrite(stage, 'measurement1', lambda row: row['first_batch_audits'][0].update(loss=1.1))
    saved = report.build_report(stage[0])
    assert saved['pilot']['changed_recipe'] is False
    assert saved['pilot']['numerical_diagnostics'][0]['first_loss_matches_validation_native'] is False
