"""Bounds, evidence completeness, historical reuse and profiler separation."""
from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import report_peft_efficiency as report
from scripts import run_peft_efficiency as runner
from scripts.cl_efficiency import write_json


@pytest.fixture
def measurement():
    saved = report.read_json(report.BASELINE_FILE)['models']['l2p']['measurements'][0]
    record = deepcopy(saved)
    record['model'] = 'ewc-lora'
    from scripts.run_low_rank_efficiency import REQUIRED_CHECKS
    record['structural_checks'].update({key: True for key in REQUIRED_CHECKS})
    return record


def test_schedule_has_exactly_twenty_separate_fresh_workers(tmp_path):
    jobs = runner.jobs(tmp_path)
    assert len(jobs) == 20
    assert len({job['directory'] for job in jobs}) == 20
    for method in runner.METHODS:
        selected = [job for job in jobs if job['method'] == method]
        assert [job['pass'] for job in selected] == ['measurement'] * 3 + ['flops']
        assert all('--execute' in job['command'] for job in selected)
        assert all('run_ranpac_efficiency.py' in ' '.join(job['command']) for job in selected) == (method == 'ranpac')


@pytest.mark.parametrize('change', ['command', 'repeat'])
def test_resume_rejects_widened_or_duplicate_schedule(tmp_path, change):
    saved = {'output': str(tmp_path), 'jobs': runner.jobs(tmp_path)}
    if change == 'command':
        saved['jobs'][0]['command'].extend(['--tasks', '2'])
    else:
        saved['jobs'][1] = deepcopy(saved['jobs'][0])
    with pytest.raises(ValueError, match='schedule'):
        runner.validate_schedule(saved)


def test_cancellation_reaches_launcher_cleanup(tmp_path):
    ready, cleaned = tmp_path / 'ready', tmp_path / 'cleaned'
    code = '''
import pathlib, signal, sys, time
def cleanup(sig, frame):
    pathlib.Path(sys.argv[2]).write_text('cleaned')
    sys.exit(0)
signal.signal(signal.SIGINT, cleanup)
pathlib.Path(sys.argv[1]).write_text('ready')
while True: time.sleep(.01)
'''
    process = subprocess.Popen([sys.executable, '-S', '-c', code, str(ready), str(cleaned)], start_new_session=True)
    try:
        import time
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        runner.stop_launcher(process)
        assert process.poll() == 0
        assert cleaned.read_text() == 'cleaned'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_plan_rejects_silent_wider_scope(monkeypatch, tmp_path):
    saved = runner.load_plan()
    saved['tasks'] = 2
    path = tmp_path / 'plan.json'
    write_json(path, saved)
    monkeypatch.setattr(runner, 'PLAN', path)
    with pytest.raises(ValueError, match='approved'):
        runner.load_plan()


@pytest.mark.parametrize('corruption', ['input', 'precision', 'count', 'gradient', 'accuracy', 'energy', 'upload'])
def test_invalid_worker_cannot_enter_aggregation(measurement, corruption):
    if corruption == 'input':
        measurement['data_snapshots']['first_task']['train']['transform_repr_sha256'] = 'wrong'
    elif corruption == 'precision':
        measurement['precision']['autocast'] = True
    elif corruption == 'count':
        measurement['metrics']['outer_training_stream_examples'] = 4992
    elif corruption == 'gradient':
        measurement['structural_checks']['finite_loss_and_gradients'] = False
    elif corruption == 'accuracy':
        measurement['accuracy']['mean_accuracy_pct'] = 20.
    elif corruption == 'energy':
        measurement['metrics']['gpu_energy_coverage_fraction'] = .98
    else:
        measurement['wandb_logging_status'] = 'failed'
    with pytest.raises(ValueError):
        report.validate_new_record(measurement, 'ewc-lora', 'measurement')


def test_missing_frozen_weight_evidence_fails(measurement):
    measurement['structural_checks'].pop('frozen_backbone_unchanged')
    with pytest.raises(ValueError, match='structural'):
        report.validate_new_record(measurement, 'ewc-lora', 'measurement')


def test_historical_metadata_stays_unknown(measurement):
    history = report.historical_inputs()
    assert set(history) == {'ease', 'l2p', 'dualprompt', 'hide-prompt'}
    for name, (document, records, _, _) in history.items():
        proof = report.compatible_history(document, records, measurement)
        assert proof['status'] == 'compatible', (name, proof)
        assert proof['cpu_thread_evidence']['status'] == ('metadata_missing' if name == 'ease' else 'matched')


def test_profile_resources_do_not_enter_measurement_statistics(measurement):
    measurements = [deepcopy(measurement) for _ in range(3)]
    for row, wall in zip(measurements, (10., 20., 30.)):
        row['metrics']['process_wall_s'] = wall
    diagnostic = deepcopy(measurement)
    diagnostic['pass'] = 'flops'
    diagnostic['metrics']['process_wall_s'] = 10000.
    diagnostic['flops'] = {'supported_ops_total': 123456.}
    summary = report.method_summary('ewc-lora', measurements, diagnostic)
    assert summary['resources']['process_wall_s']['mean'] == 20.
    assert summary['resources']['process_wall_s']['std_sample'] == 10.
    assert summary['flops']['supported_ops_total'] == 123456.


def test_preparation_does_not_execute_workers(capsys, monkeypatch):
    monkeypatch.setattr(runner, 'execute', lambda options: pytest.fail('dry run executed experiments'))
    assert runner.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['execute'] is False
    assert len(result['schedule']) == 20


@pytest.mark.parametrize('state,count', [('running', 1), ('finished', True), ('finished', 2)])
def test_remote_incomplete_or_wrong_counts_fail_before_artifact_read(measurement, tmp_path, state, count):
    from scripts import publish_peft_efficiency as publisher
    path = tmp_path / 'result.json'
    write_json(path, measurement)
    summary = {'benchmark_status': 'completed', 'benchmark_pass': 'measurement',
               'efficiency/tasks_completed': count}
    api = SimpleNamespace(run=lambda name: SimpleNamespace(state=state, summary=summary))
    with pytest.raises(ValueError):
        publisher.verify_worker(api, {'file': str(path), 'sha256': runner.file_hash(path),
                                'method': 'ewc-lora', 'pass': 'measurement'}, tmp_path / 'readback')
