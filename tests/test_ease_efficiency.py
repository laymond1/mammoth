"""Reject incomplete pilots and prevent profiling overhead in measurement totals."""

import copy

import pytest

from scripts import run_ease_efficiency as pilot


def record(pass_name='measurement', value=1.):
    return {'pass': pass_name, 'status': 'completed',
            'metrics': {**dict.fromkeys(pilot.RESOURCE_KEYS, value),
                        'gpu_energy_coverage_fraction': 0.999,
                        'tasks_completed': 1, 'outer_observe_steps': 79,
                        'outer_training_stream_examples': 5000, 'evaluation_examples': 1000},
            'phases': {phase: {'wall_s': value, 'gpu_board_energy_j': value,
                              'gpu_energy_coverage_s': value} for phase in pilot.PHASE_KEYS},
            'accuracy': {'class_il_pct': 90.},
            'structural_checks': dict.fromkeys(pilot.REQUIRED_CHECKS, True), 'sensor_errors': [],
            'flops': {'supported_ops_total': 123456, 'scope': 'partial'}}


def test_default_does_not_read_credentials_or_execute(monkeypatch, capsys):
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('No credentials in preparation'))
    monkeypatch.setattr(pilot, 'execute', lambda *args: pytest.fail('No experiment in preparation'))
    assert pilot.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out


def test_separate_profile_never_enters_efficiency_aggregates():
    measurements = [record(value=value) for value in (30., 10., 20.)]
    summary = pilot.summarize(measurements, record('flops', value=99999.))
    assert summary['status'] == 'pass'
    assert summary['resources']['process_wall_s'] == {'median': 20., 'min': 10., 'max': 30., 'values': [30., 10., 20.]}
    assert summary['phases']['end_task']['wall_s']['median'] == 20.
    assert summary['flops']['supported_ops_total'] == 123456
    assert summary['seed_variation_estimated'] is False


@pytest.mark.parametrize('field,value', [('tasks_completed', 0), ('outer_observe_steps', 78),
                                       ('outer_training_stream_examples', 4992), ('evaluation_examples', 999)])
def test_incomplete_protocol_blocks_aggregation(field, value):
    measurements = [record() for _ in range(3)]
    measurements[1]['metrics'][field] = value
    with pytest.raises(ValueError, match='Incomplete'):
        pilot.summarize(measurements, record('flops'))


@pytest.mark.parametrize('mutation', ['wrong_pass', 'failed', 'structural_failure', 'missing_checks',
                                     'missing_required_check', 'sensor_error', 'low_coverage', 'nan_time'])
def test_invalid_measurement_blocks_aggregation(mutation):
    measurements = [record() for _ in range(3)]
    bad = measurements[0]
    if mutation == 'wrong_pass':
        bad['pass'] = 'flops'
    elif mutation == 'failed':
        bad['status'] = 'failed'
    elif mutation == 'structural_failure':
        bad['structural_checks']['finite_loss_and_gradients'] = False
    elif mutation == 'missing_checks':
        bad['structural_checks'] = {}
    elif mutation == 'missing_required_check':
        del bad['structural_checks']['training_protocol_complete']
    elif mutation == 'sensor_error':
        bad['sensor_errors'] = ['missing_sensor']
    elif mutation == 'low_coverage':
        bad['metrics']['gpu_energy_coverage_fraction'] = 0.98
    else:
        bad['metrics']['process_wall_s'] = float('nan')
    with pytest.raises(ValueError):
        pilot.summarize(measurements, record('flops'))


def test_diagnostic_protocol_and_structure_must_also_complete():
    diagnostic = record('flops')
    diagnostic['metrics']['outer_observe_steps'] = 78
    with pytest.raises(ValueError, match='Incomplete'):
        pilot.summarize([record() for _ in range(3)], diagnostic)


@pytest.mark.parametrize('value', [None, 0, -1, float('inf'), True])
def test_missing_partial_flops_blocks_aggregation(value):
    diagnostic = record('flops')
    diagnostic['flops']['supported_ops_total'] = value
    with pytest.raises(ValueError, match='partial operator estimate'):
        pilot.summarize([record() for _ in range(3)], diagnostic)


@pytest.mark.parametrize('count', [0, 1, 2, 4])
def test_wrong_repeat_count_blocks_aggregation(count):
    with pytest.raises(ValueError, match='three measurements'):
        pilot.summarize([record() for _ in range(count)], record('flops'))


def test_scope_requires_user_authorization_and_native_acceptance():
    efficiency = dict(dataset='seq-cifar100-224', tasks=1, classes_per_task=10,
                      n_epochs=1, batch_size=64, seed=1993, num_workers=4,
                      measurement_repeats=3, flops_runs=1, training_authorized=True)
    valid = {'efficiency': efficiency, 'validation': {'acceptance_status': 'pass'}}
    pilot.require_approved_scope(valid)
    for field, value in [('tasks', 2), ('n_epochs', 20), ('seed', 42),
                         ('measurement_repeats', 4), ('training_authorized', False)]:
        invalid = copy.deepcopy(valid)
        invalid['efficiency'][field] = value
        with pytest.raises(ValueError, match='authorized'):
            pilot.require_approved_scope(invalid)
    invalid = copy.deepcopy(valid)
    invalid['validation']['acceptance_status'] = 'review'
    with pytest.raises(ValueError, match='two-task acceptance'):
        pilot.require_approved_scope(invalid)


def test_size_aggregation_keeps_variant_identity_and_its_canonical_pretraining():
    variant = dict(name='tiny_relative16', vit_size='tiny', feature_dim=192, num_heads=3,
                   adapter_size=16, checkpoint_sha256='a' * 64, canonical_pretrained_sha256='b' * 64)
    measurements = [record() for _ in range(3)]
    diagnostic = record('flops')
    for saved in [*measurements, diagnostic]:
        saved.update({key: variant[key] for key in ('vit_size', 'feature_dim', 'num_heads', 'adapter_size',
                                                    'checkpoint_sha256', 'canonical_pretrained_sha256')},
                     ease_variant=variant['name'], initial_canonical_pretrained_sha256='b' * 64)
        saved['structural_checks']['canonical_pretrained_matches_checkpoint'] = True
    summary = pilot.summarize(measurements, diagnostic, variant=variant)
    assert summary['status'] == 'pass'
    assert 'twenty-epoch' in summary['accuracy_scope']
    diagnostic['ease_variant'] = 'small_relative32'
    with pytest.raises(ValueError, match='selected accepted size variant'):
        pilot.summarize(measurements, diagnostic, variant=variant)


def test_size_aggregation_cannot_omit_the_canonical_checkpoint_check():
    variant = dict(name='small_fixed64', vit_size='small', feature_dim=384, num_heads=6,
                   adapter_size=64, checkpoint_sha256='a' * 64, canonical_pretrained_sha256='b' * 64)
    saved = record()
    saved.update({key: variant[key] for key in ('vit_size', 'feature_dim', 'num_heads', 'adapter_size',
                                              'checkpoint_sha256', 'canonical_pretrained_sha256')},
                 ease_variant=variant['name'], initial_canonical_pretrained_sha256='b' * 64)
    diagnostic = copy.deepcopy(saved)
    diagnostic['pass'] = 'flops'
    with pytest.raises(ValueError, match='selected accepted size variant'):
        pilot.summarize([copy.deepcopy(saved) for _ in range(3)], diagnostic, variant=variant)
