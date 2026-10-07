"""Efficiency approval, accounting and failure tests using saved CPU fixtures."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_hide_efficiency as pilot


def record(pass_name='measurement', value=1.):
    return {
        'pass': pass_name, 'status': 'completed',
        'metrics': {**dict.fromkeys(pilot.RESOURCE_KEYS, value),
                    'gpu_energy_coverage_fraction': .999, **pilot.PROTOCOL_COUNTS},
        'counts': dict(pilot.METHOD_COUNTS),
        'phases': {phase: {'wall_s': value, 'gpu_board_energy_j': value,
                          'gpu_energy_coverage_s': value} for phase in pilot.PHASE_KEYS},
        'accuracy': {'seen_correct': 900, 'seen_samples': 1000},
        'structural_checks': dict.fromkeys(pilot.REQUIRED_CHECKS, True),
        'alignment_history': [{'task_id': 0, 'branch': branch, 'status': 'not_applicable', 'steps': 0}
                              for branch in ('tii', 'prompt')],
        'sensor_errors': [], 'flops': {'supported_ops_total': 123456, 'scope': 'partial'},
        'wandb_url': 'https://wandb.ai/fixture/cl-efficiency/runs/fixture',
    }


def plan():
    return {
        'common': {'dataset': 'seq-cifar100-224', 'dataset_total_tasks': 10,
                   'classes_per_task': 10, 'seed': 1993, 'num_workers': 4, 'gpu': '1',
                   'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k'},
        'efficiency': {'tasks': 1, 'dataset_config': 'ease', 'tii_epochs': 1,
                       'prompt_epochs': 1, 'outer_epochs_per_task': 2, 'ca_epochs': 30,
                       'batch_size': 64, 'training_authorized': True,
                       'measurement_repeats': 3, 'flops_runs': 1, 'ca_runs_task1': False},
    }


def acceptance():
    return {'status': 'completed', 'acceptance_status': 'pass',
            'structural_checks': dict.fromkeys(pilot.REQUIRED_CHECKS, True)}


def test_preparation_reads_no_credentials_or_native_runner(monkeypatch, capsys):
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('Unexpected credential read'))
    monkeypatch.setattr(pilot, 'execute', lambda *args: pytest.fail('Unexpected experiment'))
    monkeypatch.setattr(pilot, 'run_native', lambda *args: pytest.fail('Unexpected native import'))
    assert pilot.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out


def test_default_runs_with_stdlib_only():
    result = subprocess.run([sys.executable, '-S', str(Path(pilot.__file__).resolve())],
                            capture_output=True, text=True, check=True)
    assert 'execution=False' in result.stdout


def test_profiler_resource_costs_cannot_enter_measurement_distributions():
    summary = pilot.summarize([record(value=value) for value in (30., 10., 20.)], record('flops', 99999.))
    assert summary['resources']['process_wall_s'] == {
        'median': 20., 'min': 10., 'max': 30., 'values': [30., 10., 20.],
    }
    assert summary['phases']['tii_statistics']['wall_s']['median'] == 20.
    assert summary['phases']['prompt_statistics']['wall_s']['median'] == 20.
    assert summary['alignment']['status'] == 'not_applicable'
    assert summary['seed_variation_estimated'] is False
    assert summary['flops']['supported_ops_total'] == 123456


def test_phase_energy_without_observed_coverage_remains_missing():
    measurements = [record() for _ in range(3)]
    for saved in measurements:
        saved['phases']['prompt_finalize'].update(gpu_board_energy_j=None, gpu_energy_coverage_s=0.)
    energy = pilot.summarize(measurements, record('flops'))['phases']['prompt_finalize']['gpu_board_energy_j']
    assert energy == {'median': None, 'min': None, 'max': None,
                      'values': [None, None, None], 'observed_count': 0}


@pytest.mark.parametrize('field,value', [('tasks_completed', 0), ('outer_observe_steps', 157),
                                       ('outer_training_stream_examples', 9992), ('evaluation_examples', 999),
                                       ('tasks_completed', True)])
def test_incomplete_native_protocol_blocks_aggregation(field, value):
    measurements = [record() for _ in range(3)]
    measurements[1]['metrics'][field] = value
    with pytest.raises(ValueError, match='Incomplete'):
        pilot.summarize(measurements, record('flops'))


@pytest.mark.parametrize('field', tuple(pilot.METHOD_COUNTS))
def test_hidden_statistics_or_training_work_cannot_be_omitted(field):
    measurements = [record() for _ in range(3)]
    del measurements[0]['counts'][field]
    with pytest.raises(ValueError, match='statistics passes'):
        pilot.summarize(measurements, record('flops'))


@pytest.mark.parametrize('mutation', ['wrong_pass', 'failed', 'missing_check', 'false_check',
                                     'nonboolean_check', 'sensor_error', 'low_coverage', 'nan_time',
                                     'negative_energy', 'missing_phase', 'alignment_omitted',
                                     'alignment_executed', 'alignment_zero_cost_claim'])
def test_invalid_measurement_blocks_aggregation(mutation):
    measurements = [record() for _ in range(3)]
    saved = measurements[0]
    if mutation == 'wrong_pass':
        saved['pass'] = 'flops'
    elif mutation == 'failed':
        saved['status'] = 'failed'
    elif mutation == 'missing_check':
        del saved['structural_checks']['statistics_shapes_and_seen_classes']
    elif mutation == 'false_check':
        saved['structural_checks']['finite_loss_and_gradients'] = False
    elif mutation == 'nonboolean_check':
        saved['structural_checks']['training_protocol_complete'] = 1
    elif mutation == 'sensor_error':
        saved['sensor_errors'] = ['gpu_sensor_unavailable']
    elif mutation == 'low_coverage':
        saved['metrics']['gpu_energy_coverage_fraction'] = .98
    elif mutation == 'nan_time':
        saved['metrics']['process_wall_s'] = float('nan')
    elif mutation == 'negative_energy':
        saved['metrics']['gpu_board_energy_j'] = -1
    elif mutation == 'missing_phase':
        del saved['phases']['tii_statistics']
    elif mutation == 'alignment_omitted':
        saved['alignment_history'] = []
    elif mutation == 'alignment_executed':
        saved['alignment_history'][0]['steps'] = 10
    else:
        saved['alignment_history'][0]['status'] = 'completed'
    with pytest.raises(ValueError):
        pilot.summarize(measurements, record('flops'))


@pytest.mark.parametrize('value', [None, 0, -1, float('inf'), True])
def test_partial_flops_must_be_available_without_fabricated_full_coverage(value):
    diagnostic = record('flops')
    diagnostic['flops']['supported_ops_total'] = value
    with pytest.raises(ValueError, match='partial operator estimate'):
        pilot.summarize([record() for _ in range(3)], diagnostic)


def test_profiler_pass_requires_all_method_work():
    diagnostic = record('flops')
    diagnostic['counts']['prompt_statistics_examples'] = 0
    with pytest.raises(ValueError, match='statistics passes'):
        pilot.summarize([record() for _ in range(3)], diagnostic)


@pytest.mark.parametrize('section,field,value', [
    ('common', 'gpu', '0'), ('common', 'seed', 42), ('common', 'dataset_total_tasks', 2),
    ('efficiency', 'tasks', 2), ('efficiency', 'tii_epochs', 20),
    ('efficiency', 'prompt_epochs', 50), ('efficiency', 'measurement_repeats', 4),
    ('efficiency', 'flops_runs', True), ('efficiency', 'dataset_config', 'hide_prompt'),
    ('efficiency', 'training_authorized', False),
])
def test_outside_approved_scope_cannot_execute(section, field, value):
    protocol = plan()
    protocol[section][field] = value
    with pytest.raises(ValueError, match='authorized'):
        pilot.require_approved_scope(protocol, acceptance())


@pytest.mark.parametrize('state', ['review', 'fail', 'not_evaluated'])
def test_native_reference_review_and_failure_block_efficiency(state):
    approved = acceptance()
    approved['acceptance_status'] = state
    with pytest.raises(ValueError, match='two-task acceptance'):
        pilot.require_approved_scope(plan(), approved)


def test_committed_execution_plan_matches_the_efficiency_contract():
    pilot.require_approved_scope(json.loads(pilot.PLAN_PATH.read_text()), acceptance())


@pytest.fixture
def execution_fixture(monkeypatch, tmp_path):
    protocol = plan()
    checkpoint = tmp_path / 'checkpoint.npz'
    checkpoint.write_bytes(b'CPU test fixture, not model weights')
    protocol['common'].update(checkpoint_file=str(checkpoint), checkpoint_sha256=pilot.file_hash(checkpoint))
    protocol_file = tmp_path / 'execution_plan.json'
    protocol_file.write_text(json.dumps(protocol))
    accepted = tmp_path / 'acceptance.json'
    accepted.write_text(json.dumps(acceptance()))
    initial_state = tmp_path / 'initial.pt'
    initial_state.write_bytes(b'CPU state fixture')
    monkeypatch.setattr(pilot, 'PLAN_PATH', protocol_file)
    monkeypatch.setattr(pilot, 'check_acceptance', lambda *args: json.loads(accepted.read_text()))
    monkeypatch.setattr(pilot, 'configure_environment', lambda: {'gpu': '1'})
    calls = []

    def native(arguments):
        pass_name = arguments[arguments.index('--pass') + 1]
        output = Path(arguments[arguments.index('--result-dir') + 1])
        output.mkdir(parents=True, exist_ok=False)
        pilot.write_json(output / 'result.json', record(pass_name))
        pilot.write_json(output / 'completion.json', {'status': 'completed', 'wall_s': 7.})
        calls.append((pass_name, arguments))
        return 0

    monkeypatch.setattr(pilot, 'run_native', native)
    return accepted, initial_state, tmp_path / 'pilot', calls


def test_execution_uses_four_separate_workers_and_exact_accepted_initial_state(execution_fixture):
    accepted, initial_state, output, calls = execution_fixture
    assert pilot.main(['--execute', '--acceptance-summary', str(accepted),
                       '--initial-state', str(initial_state), '--result-dir', str(output)]) == 0
    assert [name for name, _ in calls] == ['measurement'] * 3 + ['flops']
    for _, arguments in calls:
        assert arguments[arguments.index('--mode') + 1] == 'efficiency'
        assert arguments[arguments.index('--initial-state') + 1] == str(initial_state)
        assert arguments[arguments.index('--acceptance-summary') + 1] == str(accepted)
    execution = json.loads((output / 'execution.json').read_text())
    assert execution['status'] == 'completed'
    assert len(execution['measurements']) == 3
    assert len(execution['workers']) == 4
    assert all(len(worker['result_sha256']) == 64 for worker in execution['workers'])
    assert (output / 'summary.json').is_file()
    assert execution['summary']['launcher_wall_s_including_logging']['values'] == [7.] * 3


def test_rejected_acceptance_has_no_credentials_output_or_gpu_work(execution_fixture, monkeypatch):
    accepted, initial_state, output, calls = execution_fixture
    saved = acceptance()
    saved['acceptance_status'] = 'review'
    accepted.write_text(json.dumps(saved))
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('Credentials before acceptance'))
    with pytest.raises(ValueError, match='two-task acceptance'):
        pilot.main(['--execute', '--acceptance-summary', str(accepted),
                    '--initial-state', str(initial_state), '--result-dir', str(output)])
    assert not calls and not output.exists()


def test_failed_worker_preserves_evidence_and_stops_later_workers(execution_fixture, monkeypatch):
    accepted, initial_state, output, calls = execution_fixture
    original = pilot.run_native

    def failed(arguments):
        original(arguments)
        return 1

    monkeypatch.setattr(pilot, 'run_native', failed)
    with pytest.raises(RuntimeError, match='partial local results'):
        pilot.main(['--execute', '--acceptance-summary', str(accepted),
                    '--initial-state', str(initial_state), '--result-dir', str(output)])
    assert len(calls) == 1
    execution = json.loads((output / 'execution.json').read_text())
    assert execution['status'] == 'failed'
    assert len(execution['measurements']) == 1
    assert execution['flops'] is None
    assert not (output / 'summary.json').exists()


def test_changed_pretrained_file_blocks_workers_and_output(execution_fixture, monkeypatch):
    accepted, initial_state, output, calls = execution_fixture
    protocol = json.loads(pilot.PLAN_PATH.read_text())
    Path(protocol['common']['checkpoint_file']).write_bytes(b'changed cached weights')
    with pytest.raises(ValueError, match='checkpoint changed'):
        pilot.main(['--execute', '--acceptance-summary', str(accepted),
                    '--initial-state', str(initial_state), '--result-dir', str(output)])
    assert not calls and not output.exists()


def test_execution_without_acceptance_files_has_no_side_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('Credentials without approval'))
    monkeypatch.setattr(pilot, 'check_acceptance', lambda *args: pytest.fail('Missing path should be rejected first'))
    output = tmp_path / 'pilot'
    with pytest.raises(ValueError, match='acceptance-summary and --initial-state'):
        pilot.main(['--execute', '--result-dir', str(output)])
    assert not output.exists()
