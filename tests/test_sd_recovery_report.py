"""Recovery integration gates using locally preserved experiment evidence.

A clean checkout without the untracked diagnostic and worker artifacts skips
these integration cases; the separate worker/suite tests are self-contained.
"""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from scripts import report_sd_recovery as report
from scripts import run_sd_recovery as worker
from scripts import run_sd_recovery_suite as suite
from scripts.cl_efficiency import write_json


def require_local_artifacts(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        pytest.skip('Recovery report integration requires local experiment artifacts: ' + ', '.join(missing[:3]))


@pytest.fixture
def recovery(tmp_path):
    require_local_artifacts((report.DEFAULT_REFERENCE, report.DEFAULT_REFERENCE.parent / 'task1_predictions.npz',
                             report.DEFAULT_PRIOR))
    reference = report.full.read_json(report.DEFAULT_REFERENCE)
    old = report.full.read_json(report.DEFAULT_PRIOR)
    paths = [old['execution']['file'], old['failed_observations'][0]['file'],
             *(source['file'] for source in old['sources']),
             *(source['file'] for source in old['historical_compatibility'].values())]
    diagnostic = old.get('diagnostic_evidence')
    if diagnostic is not None:
        paths.append(diagnostic['file'])
    require_local_artifacts(paths)
    execution = report.full.read_json(old['execution']['file'])
    require_local_artifacts((execution['partial_continuation']['driver_file'],))
    if diagnostic is not None:
        saved_diagnostic = report.full.read_json(diagnostic['file'])
        require_local_artifacts([saved_diagnostic[key] for key in
                                 ('reference_source', 'native_predictions', 'native_checkpoint',
                                  'initial_state', 'checkpoint_file')])
        require_local_artifacts((report.ROOT / 'results/peft_pilot/sd_author_diagnostic.py',))
    failed = report.full.read_json(old['failed_observations'][0]['file'])
    source_hashes = worker.execution_sources()
    jobs = suite.jobs(tmp_path)
    results = []
    for index, job in enumerate(jobs):
        record = deepcopy(failed)
        directory = Path(job['directory'])
        directory.mkdir()
        predictions = directory / 'task1_predictions.npz'
        shutil.copyfile(report.DEFAULT_REFERENCE.parent / 'task1_predictions.npz', predictions)
        record.update(mode='validation' if index == 0 else 'efficiency',
                      implementation='mammoth_native', recovery_recipe=worker.RECIPE,
                      author_reference=str(report.DEFAULT_REFERENCE),
                      author_reference_sha256=worker.AUTHOR_RESULT_SHA256,
                      protocol=worker.recovery_protocol(job['pass']), **{'pass': job['pass']},
                      measurement_aggregate_eligible=job['pass'] == 'measurement',
                      efficiency_comparison=job['pass'] != 'validation', output=str(directory),
                      execution_source_sha256=source_hashes,
                      task_results=deepcopy(reference['task_results']),
                      accuracy=deepcopy(reference['accuracy']))
        record['optimizer_group_audits'] = [{'role': role, 'lr': lr, 'parameters': 1, 'task': 1,
                                            'epoch': 1, 'optimizer_update_index': 1}
                                           for role, lr in [('encoder_and_scale', .001), ('classifier', .008)]]
        record['epoch_results'][0]['loss_mean'] = reference['epoch_results'][0]['loss_mean']
        record['state_audits']['network_sha256'] = reference['state_audits']['network_sha256']
        record['wandb_url'] = f'https://wandb.ai/laymond1/cl-efficiency/runs/recovery{index}'
        record['prediction_files'] = [{'task': 1, 'path': str(predictions), 'sha256': report.shared.file_hash(predictions)}]
        record['metrics']['process_wall_s'] = (999999., 10., 20., 30., 10000.)[index]
        if job['pass'] == 'flops':
            record['flops'] = {'supported_ops_total': 1234567890123., 'partial': True}
        result_path = directory / 'result.json'
        write_json(result_path, record)
        job.update(status='completed', result={'file': str(result_path),
                                              'sha256': report.shared.file_hash(result_path),
                                              'wandb_url': record['wandb_url']})
        results.append(record)
    acceptance = suite.accuracy_gate(99.)
    acceptance.update(first_batch_checks={key + '_matches': True for key in
                                         ('input_sha256', 'labels_sha256', 'gradient_sha256', 'seed', 'samples')},
                      prediction_checks={'labels_match': True, 'global_ids_match': True, 'prediction_arrays_match': True},
                      reference_result=report._pin(report.DEFAULT_REFERENCE),
                      native_result=report._pin(tmp_path / 'validation/result.json'),
                      native_predictions=report._pin(tmp_path / 'validation/task1_predictions.npz'),
                      source_predictions=report._pin(report.DEFAULT_REFERENCE.parent / 'task1_predictions.npz'),
                      reference_network_sha256=reference['state_audits']['network_sha256'],
                      native_network_sha256=reference['state_audits']['network_sha256'],
                      network_sha256_matches=True)
    write_json(tmp_path / 'validation/acceptance.json', acceptance)
    saved = {'status': 'completed', 'completed_gpu_runs': 5, 'author_retrained': False,
             'output': str(tmp_path), 'jobs': jobs, 'acceptance': acceptance,
             'execution_source_sha256': source_hashes}
    write_json(tmp_path / 'execution.json', saved)
    return tmp_path, saved, results


def rewrite(recovery, index, change):
    root, execution, records = recovery
    change(records[index])
    path = root / Path(execution['jobs'][index]['directory']).name / 'result.json'
    write_json(path, records[index])
    execution['jobs'][index]['result']['sha256'] = report.shared.file_hash(path)
    write_json(root / 'execution.json', execution)


def test_nine_methods_preserve_eight_summaries_and_exclude_other_costs(recovery):
    root, _, _ = recovery
    saved = report.build_report(root)
    old = report.full.read_json(report.DEFAULT_PRIOR)
    assert saved['publication_gate'] == 'pass'
    assert saved['method_count'] == 9
    assert saved['new_gpu_training_runs'] == 5
    assert saved['new_validation_runs'] == 1
    assert saved['new_measurement_runs'] == 3
    assert saved['new_flops_runs'] == 1
    assert len(saved['sources']) == 20
    assert saved['sources'][:16] == old['sources']
    for name in report.OLD_METHODS:
        assert saved['methods'][name] == old['methods'][name]
    sd = saved['methods']['sd-lora']
    assert sd['resources']['process_wall_s']['values'] == [10., 20., 30.]
    assert sd['resources']['process_wall_s']['mean'] == 20.
    assert sd['resources']['process_wall_s']['std_sample'] == 10.
    assert sd['flops']['supported_ops_total'] == 1234567890123.
    assert saved['validation']['aggregate_excluded'] is True
    assert saved['author_reference']['aggregate_excluded'] is True
    assert saved['author_reference']['original_wandb_logging_status'] == 'disabled_side_conversation'
    assert saved['failed_observations'] == old['failed_observations']
    assert saved['original_deferred_jobs'] == old['deferred_jobs']
    assert saved['scope']['ease_cpu_threads'] == 'not_recorded'
    assert 'sd-lora' in report.csv_text(saved)
    text = report.markdown(saved)
    assert '99.0%' in text
    assert 'LR만 0.001' in text
    assert '13구조 검사' in text


@pytest.mark.parametrize('corruption', ['count_bool', 'short_stream', 'task_samples', 'wrong_accuracy', 'precision',
    'gradient', 'missing_check', 'train_input', 'seed', 'initial', 'lr', 'head_lr', 'rank', 'tasks',
    'upload', 'sensor', 'energy', 'collapse', 'missing_wrapper_pin', 'source'])
def test_invalid_new_worker_cannot_enter_cost_aggregates(recovery, corruption):
    def change(record):
        if corruption == 'count_bool':
            record['metrics']['tasks_completed'] = True
        elif corruption == 'short_stream':
            record['metrics']['outer_training_stream_examples'] = 4992
        elif corruption == 'task_samples':
            record['task_results'][0]['seen_samples'] = 999
        elif corruption == 'wrong_accuracy':
            record['accuracy']['mean_accuracy_pct'] = 98.9
        elif corruption == 'precision':
            record['precision']['autocast'] = 0
        elif corruption == 'gradient':
            record['structural_checks']['finite_loss_and_gradients'] = False
        elif corruption == 'missing_check':
            record['structural_checks'].pop('checkpoint_round_trip')
        elif corruption == 'train_input':
            record['data_snapshots']['task1_train']['transform_repr_sha256'] = 'wrong'
        elif corruption == 'seed':
            record['first_batch_audits'][0]['seed'] += 1
        elif corruption == 'initial':
            record['initial_learned_sha256'] = 'wrong'
        elif corruption == 'lr':
            record['protocol']['hyperparameters']['lr'] = .008
        elif corruption == 'head_lr':
            record['protocol']['hyperparameters']['fc_lr'] = .001
        elif corruption == 'rank':
            record['protocol']['rank'] = 20
        elif corruption == 'tasks':
            record['protocol']['tasks'] = 2
        elif corruption == 'upload':
            record['wandb_logging_status'] = 'failed'
        elif corruption == 'sensor':
            record['metrics']['gpu_energy_coverage_fraction'] = .98
        elif corruption == 'energy':
            record['metrics']['gpu_board_energy_wh'] = 0.
        elif corruption == 'collapse':
            record['accuracy']['mean_accuracy_pct'] = 89.9
            record['task_results'][0].update(seen_correct=899, new_correct=899)
        elif corruption == 'missing_wrapper_pin':
            record['execution_source_sha256'].pop('scripts/run_sd_recovery.py')
        else:
            record['execution_source_sha256']['models/sd_lora.py'] = 'wrong'
    rewrite(recovery, 1, change)
    with pytest.raises(ValueError):
        report.build_report(recovery[0])


@pytest.mark.parametrize('corruption', ['review', 'fail', 'wrong_gap', 'structural', 'first_batch',
    'reference_pin', 'native_pin', 'prediction_pin', 'network', 'aggregation'])
def test_validation_receipt_cannot_bypass_independent_gates(recovery, corruption):
    root, saved, _ = recovery
    acceptance = deepcopy(saved['acceptance'])
    if corruption in ('review', 'fail'):
        acceptance['status'] = corruption
    elif corruption == 'wrong_gap':
        acceptance['accuracy']['drop_pp'] = 1.4
    elif corruption == 'structural':
        acceptance['structural_checks_pass'] = False
    elif corruption == 'first_batch':
        acceptance['first_batch_checks']['gradient_sha256_matches'] = False
    elif corruption == 'reference_pin':
        acceptance['reference_result']['sha256'] = 'wrong'
    elif corruption == 'native_pin':
        acceptance['native_result']['sha256'] = 'wrong'
    elif corruption == 'prediction_pin':
        acceptance['source_predictions']['sha256'] = 'wrong'
    elif corruption == 'network':
        acceptance['native_network_sha256'] = 'wrong'
    else:
        acceptance['measurement_aggregate_eligible'] = True
    saved['acceptance'] = acceptance
    write_json(root / 'validation/acceptance.json', acceptance)
    write_json(root / 'execution.json', saved)
    with pytest.raises(ValueError):
        report.build_report(root)


@pytest.mark.parametrize('corruption', ['duplicate_url', 'profile_missing', 'worker_pending', 'schedule', 'six_runs', 'unfinished'])
def test_incomplete_repeated_or_widened_execution_is_rejected(recovery, corruption):
    root, saved, _ = recovery
    if corruption == 'duplicate_url':
        rewrite(recovery, 2, lambda record: record.update(wandb_url='https://wandb.ai/laymond1/cl-efficiency/runs/recovery1'))
    elif corruption == 'profile_missing':
        rewrite(recovery, 4, lambda record: record.update(flops={}))
    elif corruption == 'worker_pending':
        saved['jobs'][2]['status'] = 'pending'
    elif corruption == 'schedule':
        saved['jobs'][1]['command'].extend(['--tasks', '2'])
    elif corruption == 'six_runs':
        saved['completed_gpu_runs'] = 6
    else:
        saved['status'] = 'running'
    write_json(root / 'execution.json', saved)
    with pytest.raises(ValueError):
        report.build_report(root)


def test_modified_prior_summary_cannot_be_treated_as_raw_evidence(recovery, tmp_path):
    prior = report.full.read_json(report.DEFAULT_PRIOR)
    prior['methods']['ease']['accuracy_pct']['mean'] = 100.
    path = tmp_path / 'modified_prior.json'
    write_json(path, prior)
    with pytest.raises(ValueError, match='Prior report'):
        report.build_report(recovery[0], prior_report_path=path)


def test_author_reference_must_be_saved_evidence_without_retraining(recovery, tmp_path):
    reference = report.full.read_json(report.DEFAULT_REFERENCE)
    reference['accuracy']['mean_accuracy_pct'] = 100.
    path = tmp_path / 'author_result.json'
    write_json(path, reference)
    with pytest.raises(ValueError, match='saved 99% author reference changed'):
        report.build_report(recovery[0], reference_path=path)


def test_modified_prediction_arrays_cannot_satisfy_accuracy_gate(recovery):
    import numpy as np
    root, _, records = recovery
    path = root / 'measurement2/task1_predictions.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['predictions'][0] = (arrays['labels'][0] + 1) % 10
    np.savez(path, **arrays)
    rewrite(recovery, 2, lambda record: record['prediction_files'][0].update(sha256=report.shared.file_hash(path)))
    with pytest.raises(ValueError, match='prediction arrays'):
        report.build_report(root)


def test_report_preparation_never_executes_or_publishes(recovery, monkeypatch, capsys):
    monkeypatch.setattr(report, 'DEFAULT_RUN_ROOT', recovery[0])
    monkeypatch.setattr(suite, 'execute', lambda options: pytest.fail('reader executed GPU work'))
    assert report.main(['--run-root', str(recovery[0])]) == 0
    assert '"written": false' in capsys.readouterr().out
