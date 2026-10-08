"""Closed partial scope, rejected SD evidence and independent cost repeats."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import report_peft_partial as report
from scripts.cl_efficiency import write_json
from scripts.run_low_rank_efficiency import REQUIRED_CHECKS as LORA_CHECKS
from scripts.run_ranpac_efficiency import REQUIRED_CHECKS as RANPAC_CHECKS


@pytest.fixture
def partial_execution(tmp_path, monkeypatch):
    """Synthetic saved workers exercise reporting gates without model execution."""
    baseline = deepcopy(report.full.read_json(report.full.BASELINE_FILE)['models']['l2p']['measurements'][0])
    baseline['method'] = 'ewc-lora'
    baseline['model'] = 'ewc-lora'
    baseline['computation_status'] = 'completed'
    baseline['structural_checks'].update({name: True for name in LORA_CHECKS})
    baseline['task_results'] = [{'task': 1, 'seen_correct': 960, 'seen_samples': 1000}]
    jobs = report.controller.jobs(tmp_path)
    for index, job in enumerate(jobs, 1):
        if index in (6, 7, 8):
            job.update(status='deferred', deferred_reason='SD collapse under investigation')
            continue
        record = deepcopy(baseline)
        record.update(method=job['method'], model=job['method'], **{'pass': job['pass']})
        record['wandb_url'] = f'https://wandb.ai/laymond1/cl-efficiency/runs/partial{index}'
        if job['method'] == 'ranpac':
            record['structural_checks'] = {name: True for name in RANPAC_CHECKS}
        if job['pass'] == 'flops':
            record['flops'] = {'supported_ops_total': 1e14, 'partial': True}
            record['metrics']['process_wall_s'] = 10000.
        else:
            record['metrics']['process_wall_s'] = (10., 20., 30.)[(index - 1) % 4]
        directory = Path(job['directory'])
        directory.mkdir(parents=True)
        path = directory / 'result.json'
        if index == 5:
            record['accuracy']['mean_accuracy_pct'] = 10.
            record['task_results'][0]['seen_correct'] = 100
            record['metrics']['process_wall_s'] = 999999.
            job['status'] = 'quarantined'
        else:
            job['status'] = 'completed'
        write_json(path, record)
        pin = {'file': str(path), 'sha256': report.controller.file_hash(path), 'wandb_url': record['wandb_url']}
        if index == 5:
            quarantine = {**pin, 'bytes': path.stat().st_size, 'accuracy_pct': 10., 'accepted': False,
                          'reason': 'SD collapse under investigation'}
            job['quarantine'] = quarantine
        else:
            job['result'] = pin
    driver = tmp_path / 'controller.py'
    driver.write_text('# test-side controller fingerprint\n')
    saved = {'status': 'partial_completed', 'output': str(tmp_path), 'plan': report.controller.load_plan(),
             'jobs': jobs, 'execution_source_sha256': report.controller.source_hashes(),
             'sd_quarantine': quarantine,
             'failures': [{'type': 'ValueError', 'message': 'Investigate abnormal one-epoch accuracy before accepting the worker'}],
             'partial_continuation': {'driver_file': str(driver), 'driver_sha256': report.controller.file_hash(driver),
                 'frozen_commit': report.FROZEN_COMMIT, 'only_original_job_indices': list(range(9, 21)),
                 'remaining_methods': ['inf-lora', 'cl-lora', 'ranpac'], 'full_twenty_workers_completed': False,
                 'original_execution_sha256': 'a' * 64, 'unaffected_workers_completed': 12,
                 'accepted_workers_including_ewc': 16, 'sd_quarantined_workers': 1, 'sd_deferred_workers': 3}}
    execution = tmp_path / 'execution.json'
    write_json(execution, saved)
    monkeypatch.setattr(report, 'DEFAULT_DIAGNOSTIC', tmp_path / 'absent-diagnostic.json')
    return execution, saved


def rewrite_saved(execution, saved):
    write_json(execution, saved)


def rewrite_record(saved, index, update):
    job = saved['jobs'][index - 1]
    path = Path(job['directory']) / 'result.json'
    record = report.full.read_json(path)
    update(record)
    write_json(path, record)
    if index == 5:
        saved['sd_quarantine'].update(sha256=report.controller.file_hash(path), bytes=path.stat().st_size,
                                      wandb_url=record['wandb_url'])
        job['quarantine'] = deepcopy(saved['sd_quarantine'])
    else:
        job['result'].update(sha256=report.controller.file_hash(path), wandb_url=record['wandb_url'])


def test_exact_partial_counts_and_costs_exclude_sd_and_profilers(partial_execution):
    execution, saved = partial_execution
    result = report.build_report(execution)
    assert result['status'] == 'partial_completed'
    assert result['publication_gate'] == 'pass_for_accepted_subset'
    assert result['all_requested_methods_passed'] is False
    assert result['method_count'] == 8
    assert result['new_accepted_workers'] == 16
    assert result['new_measurement_runs'] == 12
    assert result['new_flops_runs'] == 4
    assert result['new_quarantined_workers'] == 1
    assert result['new_deferred_workers'] == 3
    assert len(result['sources']) == 16
    assert result['sources'][4]['original_job_index'] == 9
    assert 'sd-lora' not in result['methods']
    assert result['failed_observations'][0]['accuracy_pct'] == 10.
    assert result['failed_observations'][0]['aggregate_excluded'] is True
    assert result['failed_observations'][0]['observed_resources']['process_wall_s'] == 999999.
    for name in report.ACCEPTED_NEW_METHODS:
        values = result['methods'][name]['resources']['process_wall_s']
        assert values['mean'] == 20.
        assert values['std_sample'] == 10.
        assert result['methods'][name]['flops']['supported_ops_total'] == 1e14
    assert result['execution']['execution_source_sha256'] == saved['execution_source_sha256']
    assert result['report_generator']['sha256'] == report.controller.file_hash(report.__file__)
    text = report.markdown(result)
    assert '원인은 아직 확정되지 않았으며' in text
    assert '전체9방법·신규20실행이 모두 통과한 결과가 아닙니다' in text
    assert 'sd-lora' not in report.csv_text(result)


@pytest.mark.parametrize('status', ['completed', 'failed', 'interrupted', 'running'])
def test_nonfinal_status_cannot_be_published(partial_execution, status):
    execution, saved = partial_execution
    saved['status'] = status
    rewrite_saved(execution, saved)
    with pytest.raises(ValueError, match='partial_completed'):
        report.build_report(execution)


@pytest.mark.parametrize('corruption', ['accepted_sd', 'missing_quarantine', 'sd_count', 'source', 'driver', 'scope', 'failure_history'])
def test_partial_gate_rejects_arbitrary_sd_exclusion_and_changed_provenance(partial_execution, corruption):
    execution, saved = partial_execution
    if corruption == 'accepted_sd':
        saved['jobs'][4]['status'] = 'completed'
    elif corruption == 'missing_quarantine':
        saved.pop('sd_quarantine')
    elif corruption == 'sd_count':
        saved['partial_continuation']['sd_deferred_workers'] = 2
    elif corruption == 'source':
        saved['execution_source_sha256']['models/sd_lora.py'] = 'wrong'
    elif corruption == 'driver':
        Path(saved['partial_continuation']['driver_file']).write_text('# changed\n')
    elif corruption == 'scope':
        saved['jobs'][8]['command'].extend(['--tasks', '2'])
    else:
        saved['failures'] = []
    rewrite_saved(execution, saved)
    with pytest.raises(ValueError):
        report.build_report(execution)


@pytest.mark.parametrize('status', ['failed', 'deferred', 'running', 'pending'])
def test_other_methods_failures_cannot_be_silently_omitted(partial_execution, status):
    execution, saved = partial_execution
    saved['jobs'][8]['status'] = status
    rewrite_saved(execution, saved)
    with pytest.raises(ValueError, match='unaffected'):
        report.build_report(execution)


@pytest.mark.parametrize('corruption', ['accuracy_passes', 'raw', 'sensor', 'count', 'structural', 'deferred_executed'])
def test_sd_failure_evidence_still_has_strict_nonaccuracy_gates(partial_execution, corruption):
    execution, saved = partial_execution
    if corruption == 'deferred_executed':
        directory = Path(saved['jobs'][5]['directory'])
        directory.mkdir(parents=True)
        (directory / 'spec.json').write_text('{}')
    else:
        def change(record):
            if corruption == 'accuracy_passes':
                record['accuracy']['mean_accuracy_pct'] = 96.
            elif corruption == 'raw':
                record['data_snapshots']['first_task']['train']['transform_repr_sha256'] = 'different'
            elif corruption == 'sensor':
                record['metrics']['gpu_energy_coverage_fraction'] = .5
            elif corruption == 'count':
                record['task_results'][0]['seen_correct'] = 99
            else:
                record['structural_checks']['finite_loss_and_gradients'] = False
        rewrite_record(saved, 5, change)
    rewrite_saved(execution, saved)
    with pytest.raises(ValueError):
        report.build_report(execution)


@pytest.mark.parametrize('duplicate', ['url', 'path', 'changed_record'])
def test_repeated_or_changed_evidence_is_not_a_new_cost_observation(partial_execution, duplicate):
    execution, saved = partial_execution
    if duplicate == 'url':
        rewrite_record(saved, 2, lambda record: record.update(wandb_url=saved['jobs'][0]['result']['wandb_url']))
    elif duplicate == 'path':
        saved['jobs'][1]['result'] = deepcopy(saved['jobs'][0]['result'])
    else:
        path = Path(saved['jobs'][1]['result']['file'])
        record = report.full.read_json(path)
        record['metrics']['process_wall_s'] = 55.
        write_json(path, record)
    rewrite_saved(execution, saved)
    with pytest.raises(ValueError, match='repeat|fingerprints'):
        report.build_report(execution)


def test_ongoing_draft_is_explicitly_blocked_and_excludes_incomplete_groups(partial_execution):
    execution, saved = partial_execution
    saved['status'] = 'running'
    for job in saved['jobs'][12:]:
        job['status'] = 'pending'
        job.pop('result')
    rewrite_saved(execution, saved)
    result = report.build_report(execution, draft=True)
    assert result['status'] == 'draft'
    assert result['publication_gate'] == 'blocked'
    assert result['method_count'] == 6  # four historical + EWC + InfLoRA
    assert result['new_accepted_workers'] == 8
    assert result['draft_incomplete_method_groups'] == ['cl-lora', 'ranpac']
    assert '초안' in report.markdown(result)


def test_draft_cannot_hide_a_failed_unaffected_method(partial_execution):
    execution, saved = partial_execution
    saved['status'] = 'running'
    saved['jobs'][12]['status'] = 'failed'
    rewrite_saved(execution, saved)
    with pytest.raises(ValueError, match='unaffected'):
        report.build_report(execution, draft=True)


@pytest.fixture
def diagnostic_execution(partial_execution, tmp_path, monkeypatch):
    execution, saved = partial_execution
    initial = tmp_path / 'initial.pt'
    initial.write_bytes(b'initialization test evidence')
    native_directory = Path(saved['sd_quarantine']['file']).parent
    checkpoint = native_directory / 'task1_state.pt'
    checkpoint.write_bytes(b'finalized native test evidence')
    predictions = native_directory / 'task1_predictions.npz'
    predictions.write_bytes(b'prediction archive test provenance')
    def native_fields(native):
        native.update(initial_state=str(initial), initial_state_sha256=report.low_rank.file_hash(initial),
                      initial_learned_sha256='b'*64, execution_source_sha256=report.controller.source_hashes(),
                      prediction_files=[{'path': str(predictions), 'sha256': report.low_rank.file_hash(predictions)}])
    rewrite_record(saved, 5, native_fields)
    rewrite_saved(execution, saved)
    native = report.full.read_json(saved['sd_quarantine']['file'])
    diagnostic_directory = tmp_path / 'diagnostic'
    diagnostic_directory.mkdir()
    diagnostic = diagnostic_directory / 'result.json'
    protocol = report.efficiency_protocol('sd-lora')
    protocol.update(mode='diagnostic', measurement_repeats=0, flops_runs=0, efficiency_comparison=False)
    sources = {**native['execution_source_sha256'], 'results/peft_pilot/sd_author_diagnostic.py':
               report.low_rank.file_hash(report.ROOT / 'results/peft_pilot/sd_author_diagnostic.py')}
    record = {'method': 'sd-lora', 'mode': 'diagnostic', 'status': 'completed', 'computation_status': 'completed',
              'wandb_logging_status': 'uploaded', 'checkpoint_sha256': report.full.CHECKPOINT_SHA256,
              'canonical_pretrained_sha256': report.low_rank.CANONICAL_PRETRAINED_SHA256,
              'reference_commit': report.low_rank.REFERENCE_COMMIT, 'reference_source': str(tmp_path / 'author'),
              'reference_source_sha256': {'author.py': 'source-pin'}, 'execution_source_sha256': sources,
              'measurement_aggregate_eligible': False, 'efficiency_comparison': False, 'paper_reproduction': False,
              'precision': report.low_rank.PRECISION, 'protocol': protocol, 'output': str(diagnostic_directory),
              'diagnosis': 'source_native_evidence_requires_review',
              'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/diagnostic',
              'counts': {'observe_calls': 79, 'optimizer_updates': 79, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
              'observed_totals': {'training_examples': 5000, 'evaluation_examples': 1000, 'steps': 79, 'tasks_completed': 1},
              'structural_checks': {name: True for name in LORA_CHECKS},
              'epoch_results': [{'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000}],
              'first_batch_audits': [{'task': 1}], 'task_results': native['task_results'], 'accuracy': native['accuracy'],
              'data_snapshots': native['data_snapshots'], 'data_files': native.get('data_files'),
              'state_audits': {'frozen_backbone_sha256': report.low_rank.CANONICAL_PRETRAINED_SHA256},
              'initial_learned_sha256': native['initial_learned_sha256']}
    paths = {'native_result': Path(saved['sd_quarantine']['file']), 'native_predictions': predictions,
             'native_checkpoint': checkpoint, 'initial_state': initial, 'checkpoint_file': Path(report.low_rank.CHECKPOINT)}
    actual_hash = report.low_rank.file_hash
    def no_weight_read(path):
        return report.low_rank.CHECKPOINT_SHA256 if Path(path) == Path(report.low_rank.CHECKPOINT) else actual_hash(path)
    monkeypatch.setattr(report.low_rank, 'file_hash', no_weight_read)
    monkeypatch.setattr(report.low_rank, 'verify_reference_source', lambda path: {'author.py': 'source-pin'})
    for key, path in paths.items():
        record[key], record[key + '_sha256'] = str(path), report.low_rank.file_hash(path)
    write_json(diagnostic, record)
    return execution, diagnostic, record


def test_diagnostic_is_provenance_only_and_does_not_accept_sd(diagnostic_execution):
    execution, diagnostic, _ = diagnostic_execution
    result = report.build_report(execution, diagnostic_path=diagnostic)
    assert result['method_count'] == 8
    assert len(result['sources']) == 16
    assert result['diagnostic_evidence']['sha256'] == report.controller.file_hash(diagnostic)
    assert result['diagnostic_evidence']['aggregate_excluded'] is True
    assert result['diagnosis_status'] == 'cause_not_confirmed'
    assert result['scope']['sd_accepted'] is False
    assert result['new_measurement_runs'] == 12
    write_json(diagnostic, {'method': 'sd-lora', 'mode': 'efficiency', 'measurement_aggregate_eligible': True})
    with pytest.raises(ValueError, match='separate'):
        report.build_report(execution, diagnostic_path=diagnostic)


@pytest.mark.parametrize('corruption', ['status', 'upload', 'protocol', 'precision', 'count', 'totals', 'checks',
                                      'source', 'reference', 'initial', 'checkpoint', 'input', 'artifact'])
def test_incomplete_or_changed_diagnostic_cannot_be_published(diagnostic_execution, corruption):
    execution, diagnostic, record = diagnostic_execution
    if corruption == 'status':
        record['status'] = 'failed'
    elif corruption == 'upload':
        record['wandb_logging_status'] = 'failed'
    elif corruption == 'protocol':
        record['protocol']['n_epochs'] = 2
    elif corruption == 'precision':
        record['precision'] = {**record['precision'], 'autocast': True}
    elif corruption == 'count':
        record['counts']['optimizer_updates'] = 78
    elif corruption == 'totals':
        record['observed_totals']['training_examples'] = 4992
    elif corruption == 'checks':
        record['structural_checks'].pop('frozen_backbone_unchanged')
    elif corruption == 'source':
        record['execution_source_sha256']['models/sd_lora.py'] = 'changed'
    elif corruption == 'reference':
        record['reference_source_sha256']['author.py'] = 'changed'
    elif corruption == 'initial':
        record['initial_state_sha256'] = 'changed'
    elif corruption == 'checkpoint':
        record['checkpoint_file_sha256'] = 'changed'
    elif corruption == 'input':
        record['data_snapshots']['first_task']['train']['transform_repr_sha256'] = 'changed'
    else:
        Path(record['native_predictions']).write_bytes(b'changed evidence')
    write_json(diagnostic, record)
    with pytest.raises(ValueError):
        report.build_report(execution, diagnostic_path=diagnostic)


def test_original_full_report_still_rejects_partial_execution(partial_execution):
    execution, _ = partial_execution
    with pytest.raises(ValueError, match='complete approved'):
        report.full.build_report(execution)


def test_draft_cannot_write_to_public_benchmarks(partial_execution, monkeypatch):
    execution, _ = partial_execution
    with pytest.raises(ValueError, match='Draft'):
        report.main(['--execution', str(execution), '--draft', '--write',
                     '--output', str(report.ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial')])
