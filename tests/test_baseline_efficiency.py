"""Approve only complete baseline pilots and exclude profiler overhead."""

import copy
import json
from types import SimpleNamespace

import pytest

from scripts import run_baseline_efficiency as pilot


def plan():
    return {'training_authorized': True, 'models': list(pilot.MODELS),
            'checkpoint': {'file': pilot.CHECKPOINT, 'sha256': pilot.CHECKPOINT_SHA256, 'tag': pilot.CHECKPOINT_TAG},
            'ease_reference': {'file': pilot.EASE_RESULTS, 'sha256': pilot.EASE_RESULTS_SHA256, 'reuse_completed_measurements': True},
            'efficiency': {'tasks': 1, 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4,
                           'seed': 1993, 'measurement_repeats': 3, 'flops_runs': 1}}


def record(pass_name='measurement', value=1.):
    return {'pass': pass_name, 'status': 'completed',
            'metrics': {**dict.fromkeys(pilot.RESOURCE_KEYS, value),
                        'gpu_energy_coverage_fraction': .999,
                        'tasks_completed': 1, 'outer_observe_steps': 79,
                        'outer_training_stream_examples': 5000, 'evaluation_examples': 1000},
            'phases': {phase: {'wall_s': value, 'gpu_board_energy_j': value,
                              'gpu_energy_coverage_s': value} for phase in pilot.PHASE_KEYS},
            'accuracy': {'mean_accuracy_pct': 90.},
            'structural_checks': {**dict.fromkeys(pilot.REQUIRED_CHECKS, True), 'method_specific_final_state': True},
            'sensor_errors': [], 'flops': {'supported_ops_total': 123456, 'scope': 'partial'}}


def test_default_does_not_read_credentials_or_execute(monkeypatch, capsys):
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('No credentials in preparation'))
    monkeypatch.setattr(pilot, 'execute', lambda *args: pytest.fail('No experiments in preparation'))
    assert pilot.main([]) == 0
    output = capsys.readouterr().out
    assert 'execution=False' in output and 'EASE reused' in output


def test_prepared_plan_matches_authorized_scope():
    source = pilot.ROOT / 'benchmarks/cl_efficiency/baselines/validation_plan.json'
    pilot.require_approved_scope(json.loads(source.read_text()))


@pytest.mark.parametrize('field,value', [('tasks', 2), ('n_epochs', 20), ('batch_size', 48),
                                       ('num_workers', 0), ('seed', 42), ('measurement_repeats', 4),
                                       ('flops_runs', 0), ('tasks', True)])
def test_scope_rejects_changed_protocol(field, value):
    invalid = plan()
    invalid['efficiency'][field] = value
    with pytest.raises(ValueError, match='authorized'):
        pilot.require_approved_scope(invalid)


@pytest.mark.parametrize('mutation', ['unauthorized', 'extra_method', 'missing_method', 'wrong_file',
                                     'wrong_hash', 'wrong_tag', 'changed_reference', 'reference_rerun'])
def test_scope_rejects_unauthorized_method_or_weights(mutation):
    invalid = plan()
    if mutation == 'unauthorized':
        invalid['training_authorized'] = False
    elif mutation == 'extra_method':
        invalid['models'].append('ease')
    elif mutation == 'missing_method':
        invalid['models'].pop()
    elif mutation == 'wrong_file':
        invalid['checkpoint']['file'] = '/tmp/other.npz'
    elif mutation == 'wrong_hash':
        invalid['checkpoint']['sha256'] = '0' * 64
    elif mutation == 'wrong_tag':
        invalid['checkpoint']['tag'] = 'vit_base_patch16_224.augreg_in21k'
    elif mutation == 'changed_reference':
        invalid['ease_reference']['sha256'] = '0' * 64
    else:
        invalid['ease_reference']['reuse_completed_measurements'] = False
    with pytest.raises(ValueError):
        pilot.require_approved_scope(invalid)


def test_pilot_manifest_preserves_inventory_and_method_hyperparameters():
    source = json.loads((pilot.ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())
    original = copy.deepcopy(source)
    updated = pilot.make_pilot_manifest(source)
    assert source == original
    assert len(updated['models']) == len(source['models']) == 71
    for before, after in zip(source['models'], updated['models']):
        if before['model'] not in pilot.MODELS:
            assert after == before
        else:
            assert after['baseline_pilot'] is True
            assert after['args']['lr'] == before['args']['lr']
            assert after['args']['dataset_config'] == 'ease'
            assert after['args']['backbone'] == 'vit'
            assert after['args']['pretrained'] == 0
            assert after['args']['pretrained_checkpoint'] == pilot.CHECKPOINT
            if before['model'] in ('l2p', 'dualprompt'):
                assert after['args']['batchwise_prompt'] == before['args']['batchwise_prompt'] == 0


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_dataset', 'deferred'])
def test_pilot_manifest_cannot_skip_or_substitute_method(mutation):
    source = json.loads((pilot.ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())
    entry = next(item for item in source['models'] if item['model'] == 'l2p')
    if mutation == 'missing':
        source['models'].remove(entry)
    elif mutation == 'duplicate':
        source['models'].append(copy.deepcopy(entry))
    elif mutation == 'wrong_dataset':
        entry['dataset'] = 'seq-cifar10'
    else:
        entry['deferred_reason'] = 'unready'
    with pytest.raises(ValueError):
        pilot.make_pilot_manifest(source)


def test_profiler_time_memory_cpu_and_energy_never_enter_measurement_statistics():
    summary = pilot.summarize([record(value=value) for value in (30., 10., 20.)], record('flops', value=99999.))
    expected = {'median': 20., 'min': 10., 'max': 30., 'values': [30., 10., 20.]}
    assert summary['resources']['process_wall_s'] == expected
    assert summary['resources']['process_tree_cpu_s'] == expected
    assert summary['resources']['cuda_peak_allocated_mib'] == expected
    assert summary['resources']['gpu_board_energy_j'] == expected
    assert summary['phases']['end_task']['wall_s'] == expected
    assert summary['phases']['end_task']['gpu_board_energy_j'] == expected
    assert summary['flops']['supported_ops_total'] == 123456
    assert summary['seed_variation_estimated'] is False


@pytest.mark.parametrize('count', [0, 1, 2, 4])
def test_incomplete_repeat_count_fails(count):
    with pytest.raises(ValueError, match='three measurements'):
        pilot.summarize([record() for _ in range(count)], record('flops'))


@pytest.mark.parametrize('field,value', [('tasks_completed', 0), ('outer_observe_steps', 78),
                                       ('outer_training_stream_examples', 4992), ('evaluation_examples', 999)])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_incomplete_protocol_fails_for_measurements_and_diagnostic(field, value, diagnostic):
    measurements, flops = [record() for _ in range(3)], record('flops')
    (flops if diagnostic else measurements[1])['metrics'][field] = value
    with pytest.raises(ValueError, match='Incomplete'):
        pilot.summarize(measurements, flops)


@pytest.mark.parametrize('mutation', ['wrong_pass', 'failed', 'missing_checks', 'missing_required', 'missing_method_final',
                                     'structural_failure', 'method_specific_failure', 'sensor_error',
                                     'low_coverage', 'nan_time', 'nan_cpu'])
def test_invalid_measurement_fails(mutation):
    measurements = [record() for _ in range(3)]
    bad = measurements[0]
    if mutation == 'wrong_pass':
        bad['pass'] = 'flops'
    elif mutation == 'failed':
        bad['status'] = 'failed'
    elif mutation == 'missing_checks':
        bad['structural_checks'] = {}
    elif mutation == 'missing_required':
        del bad['structural_checks']['pretrained_backbones_match_checkpoint']
    elif mutation == 'missing_method_final':
        del bad['structural_checks']['method_specific_final_state']
    elif mutation == 'structural_failure':
        bad['structural_checks']['frozen_backbone_unchanged'] = False
    elif mutation == 'method_specific_failure':
        bad['structural_checks']['method_specific_final_state'] = False
    elif mutation == 'sensor_error':
        bad['sensor_errors'] = ['missing_sensor']
    elif mutation == 'low_coverage':
        bad['metrics']['gpu_energy_coverage_fraction'] = .98
    else:
        bad['metrics']['process_tree_cpu_s' if mutation == 'nan_cpu' else 'process_wall_s'] = float('nan')
    with pytest.raises(ValueError):
        pilot.summarize(measurements, record('flops'))


@pytest.mark.parametrize('accuracy', [None, -1., 19.9, 20., 100.1, float('nan'), True])
def test_accuracy_collapse_or_invalid_value_fails(accuracy):
    measurements = [record() for _ in range(3)]
    measurements[0]['accuracy']['mean_accuracy_pct'] = accuracy
    with pytest.raises(ValueError, match='collapse check'):
        pilot.summarize(measurements, record('flops'))


def test_greater_than_twenty_percent_is_only_a_collapse_check():
    measurements = [record() for _ in range(3)]
    measurements[0]['accuracy']['mean_accuracy_pct'] = 20.1
    assert pilot.summarize(measurements, record('flops'))['status'] == 'pass'


@pytest.mark.parametrize('value', [None, 0, -1, float('inf'), True])
def test_positive_partial_flops_is_required(value):
    flops = record('flops')
    flops['flops']['supported_ops_total'] = value
    with pytest.raises(ValueError, match='partial operator estimate'):
        pilot.summarize([record() for _ in range(3)], flops)


def test_missing_flops_is_incomplete():
    with pytest.raises(ValueError, match='three measurements'):
        pilot.summarize([record() for _ in range(3)], None)


def test_ease_reference_is_reused_without_mutating_input():
    path = pilot.ROOT / pilot.EASE_RESULTS
    before = path.read_bytes()
    comparison = pilot._read_ease_reference()
    assert comparison['rerun'] is False
    assert comparison['summary']['resources']['process_tree_cpu_s']['median'] > 0.
    assert path.read_bytes() == before


def test_collection_preserves_failed_worker_record_and_suite_path(tmp_path):
    directory = tmp_path / 'ranpac' / 'repeat1'
    suite = directory / 'cl-efficiency-test'
    suite.mkdir(parents=True)
    failed = {'model': 'ranpac', 'measurement': {'status': 'failed', 'reason': 'test_failure'}}
    (suite / 'results.json').write_text(json.dumps([failed]))
    execution = {'suites': [], 'models': {'ranpac': {'measurements': [], 'flops': None}}}
    result = pilot._collect_suite(directory, set(), 'ranpac', execution, tmp_path)
    assert result == failed
    saved = json.loads((tmp_path / 'execution.json').read_text())
    assert saved['models']['ranpac']['measurements'] == [failed['measurement']]
    assert saved['suites'] == ['ranpac/repeat1/cl-efficiency-test']


def test_gpu_guard_fails_before_any_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'configure_environment', lambda: {'gpu': '0'})
    monkeypatch.setattr(pilot, 'make_pilot_manifest', lambda *args: pytest.fail('No experiment before GPU guard'))
    with pytest.raises(ValueError, match='physical GPU1'):
        pilot.execute(SimpleNamespace(output=tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.fixture
def mocked_execution(tmp_path, monkeypatch):
    """Stand in for worker processes; no real benchmark execute is permitted."""
    checkpoint = tmp_path / 'test.npz'
    checkpoint.write_bytes(b'test-only')
    monkeypatch.setattr(pilot, 'CHECKPOINT', str(checkpoint))
    monkeypatch.setattr(pilot, 'CHECKPOINT_SHA256', pilot.hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    # Scope validation has dedicated cases above; this fixture replaces only
    # filesystem/GPU/external dependencies to inspect parent orchestration.
    monkeypatch.setattr(pilot, 'require_approved_scope', lambda plan: None)
    monkeypatch.setattr(pilot, '_read_ease_reference', lambda: {'rerun': False, 'summary': {}})
    monkeypatch.setattr(pilot, '_source_hashes', lambda: {})
    monkeypatch.setattr(pilot, 'configure_environment', lambda: {
        'gpu': '1', 'data_root': str(tmp_path), 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    invocations = []

    def prepare(args):
        args.output.mkdir(parents=True)
        return {}, {}

    def execute(args, prepared, ready):
        invocations.append((args.models, args.flops))
        suite = args.output / 'cl-efficiency-mocked'
        suite.mkdir()
        result = {'model': args.models, 'measurement': record(value=len(invocations))}
        if args.flops == 'paired':
            result['flops'] = record('flops', value=99999.)
        (suite / 'results.json').write_text(json.dumps([result]))
        return 0

    monkeypatch.setattr(pilot.benchmark_cl, 'prepare', prepare)
    monkeypatch.setattr(pilot.benchmark_cl, 'execute', execute)
    return tmp_path / 'outputs', invocations


def test_parent_orchestrates_exactly_three_measurements_and_one_diagnostic_per_method(mocked_execution):
    output, invocations = mocked_execution
    assert pilot.execute(SimpleNamespace(output=output)) == 0
    assert invocations == [(method, mode) for method in pilot.MODELS for mode in ('off', 'off', 'paired')]
    suites = list(output.glob('baseline-efficiency-pilot-*'))
    assert len(suites) == 1
    saved = json.loads((suites[0] / 'execution.json').read_text())
    assert saved['status'] == 'completed'
    assert saved['summary']['new_measurement_runs'] == 9
    assert saved['summary']['new_flops_runs'] == 3
    assert saved['ease_reference']['rerun'] is False
    for name in pilot.MODELS:
        assert len(saved['models'][name]['measurements']) == 3
        assert saved['models'][name]['flops']['pass'] == 'flops'
        assert saved['models'][name]['summary']['resources']['process_wall_s']['max'] < 99999.


def test_parent_preserves_failure_and_stops_before_further_workers(mocked_execution, monkeypatch):
    output, invocations = mocked_execution
    successful = pilot.benchmark_cl.execute

    def failed(args, prepared, ready):
        successful(args, prepared, ready)
        file = args.output / 'cl-efficiency-mocked/results.json'
        result = json.loads(file.read_text())
        result[0]['measurement']['status'] = 'failed'
        result[0]['measurement']['reason'] = 'mock_failure'
        file.write_text(json.dumps(result))
        return 1

    monkeypatch.setattr(pilot.benchmark_cl, 'execute', failed)
    with pytest.raises(RuntimeError, match='partial results are preserved'):
        pilot.execute(SimpleNamespace(output=output))
    assert invocations == [('ranpac', 'off')]
    suite = next(output.glob('baseline-efficiency-pilot-*'))
    saved = json.loads((suite / 'execution.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['models']['ranpac']['measurements'][0]['reason'] == 'mock_failure'
    assert not saved['models']['l2p']['measurements']


def test_parent_records_interrupted_suite_without_losing_partial_results(mocked_execution, monkeypatch):
    output, invocations = mocked_execution
    successful = pilot.benchmark_cl.execute

    def interrupted(args, prepared, ready):
        successful(args, prepared, ready)
        raise KeyboardInterrupt()

    monkeypatch.setattr(pilot.benchmark_cl, 'execute', interrupted)
    with pytest.raises(KeyboardInterrupt):
        pilot.execute(SimpleNamespace(output=output))
    assert invocations == [('ranpac', 'off')]
    suite = next(output.glob('baseline-efficiency-pilot-*'))
    saved = json.loads((suite / 'execution.json').read_text())
    assert saved['status'] == 'failed' and saved['failure_type'] == 'KeyboardInterrupt'
    assert saved['invocations'][0]['status'] == 'interrupted'
    assert saved['invocations'][0]['partial_suites']
    assert len(saved['models']['ranpac']['measurements']) == 1
