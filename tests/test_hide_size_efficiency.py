"""Own-size acceptance and four fresh pilot workers using CPU saved fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import hide_protocol
from scripts import hide_size_protocol
from scripts import run_hide_efficiency as pilot
from tests.test_hide_efficiency import record as base_record


@pytest.fixture(params=['small', 'tiny'])
def size_fixture(request, tmp_path, monkeypatch):
    name = request.param
    selected = hide_size_protocol.load_variant(name, verify_checkpoint=False)
    plan = hide_size_protocol.load_plan(name)
    checkpoint, initial = tmp_path / 'pretrained.npz', tmp_path / 'initial.pt'
    checkpoint.write_bytes(name.encode())
    initial.write_bytes(b'CPU fixture, not model weights')
    selected.update(checkpoint_file=str(checkpoint), checkpoint_sha256=pilot.file_hash(checkpoint))
    plan['common'].update(selected)
    monkeypatch.setattr(hide_size_protocol, 'load_variant', lambda variant, verify_checkpoint=True: dict(selected))
    monkeypatch.setattr(hide_size_protocol, 'load_plan', lambda variant=None: deepcopy(plan))
    accepted = {'status': 'completed', 'model': 'hide-prompt', 'mode': 'validation',
                'variant': name, 'acceptance_status': 'pass',
                'protocol': hide_protocol.protocol('validation', variant=name),
                'checkpoint_sha256': selected['checkpoint_sha256'],
                'canonical_pretrained_sha256': selected['canonical_pretrained_sha256'],
                'initial_learned_sha256': {'tii': 'a' * 64, 'prompt': 'b' * 64},
                'execution_source_sha256': hide_protocol.source_hashes(variant=name),
                'structural_checks': dict.fromkeys(pilot.REQUIRED_CHECKS, True)}
    accepted_path = tmp_path / 'accepted.json'
    accepted_path.write_text(json.dumps(accepted))
    checks, calls = [], []

    def acceptance(path, state, variant=None):
        checks.append((path, state, variant))
        return json.loads(accepted_path.read_text())

    monkeypatch.setattr(pilot, 'check_acceptance', acceptance)
    monkeypatch.setattr(pilot, 'configure_environment', lambda: {'gpu': '1'})

    def observation(pass_name='measurement', value=1.):
        saved = base_record(pass_name, value)
        saved.update(variant=name, protocol=hide_protocol.protocol('efficiency', variant=name),
                     model='hide-prompt', gpu='1', checkpoint_sha256=selected['checkpoint_sha256'],
                     canonical_pretrained_sha256=selected['canonical_pretrained_sha256'],
                     precision=dict(hide_protocol.FULL_CL_PRECISION),
                     wandb_logging_status='uploaded', wandb_artifact={'state': 'COMMITTED'},
                     initial_state_sha256=pilot.file_hash(initial),
                     initial_learned_sha256=accepted['initial_learned_sha256'],
                     acceptance_summary_sha256=pilot.file_hash(accepted_path),
                     execution_source_sha256=accepted['execution_source_sha256'],
                     **{key: selected[key] for key in ('vit_size', 'feature_dim', 'num_heads', 'checkpoint_tag')})
        return saved

    def native(arguments):
        pass_name = arguments[arguments.index('--pass') + 1]
        output = Path(arguments[arguments.index('--result-dir') + 1])
        output.mkdir(parents=True, exist_ok=False)
        pilot.write_json(output / 'result.json', observation(pass_name))
        pilot.write_json(output / 'completion.json', {'status': 'completed', 'wall_s': 7.})
        calls.append(arguments)
        return 0

    monkeypatch.setattr(pilot, 'run_native', native)
    command = ['--execute', '--variant', name, '--acceptance-summary', str(accepted_path),
               '--initial-state', str(initial), '--result-dir', str(tmp_path / 'pilot')]
    return name, selected, accepted, accepted_path, initial, checks, calls, observation, command


@pytest.mark.parametrize('name', ['small', 'tiny'])
def test_size_pilot_dry_preparation_has_no_ML_import_or_external_actions(name):
    script = ('import sys; from scripts.run_hide_efficiency import main; '
              f'main(["--variant", "{name}"]); '
              'assert "torch" not in sys.modules; assert "wandb" not in sys.modules')
    result = subprocess.run([sys.executable, '-S', '-c', script], cwd=pilot.ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert f'HiDe-Prompt {name.capitalize()}' in result.stdout and 'execution=False' in result.stdout


def test_pilot_forwards_own_size_and_pins_into_four_fresh_workers(size_fixture):
    name, info, _, _, _, checks, calls, _, command = size_fixture
    assert pilot.main(command) == 0
    assert checks[0][2] == name and len(calls) == 4
    assert [args[args.index('--pass') + 1] for args in calls] == ['measurement'] * 3 + ['flops']
    assert len({args[args.index('--result-dir') + 1] for args in calls}) == 4
    for args in calls:
        assert args[args.index('--variant') + 1] == name
        assert args[args.index('--checkpoint') + 1] == info['checkpoint_file']
    output = Path(command[command.index('--result-dir') + 1])
    saved = json.loads((output / 'execution.json').read_text())
    assert saved['variant'] == saved['summary']['variant'] == name and saved['status'] == 'completed'
    assert all(source in saved['execution_source_sha256'] for source in hide_size_protocol.SOURCE_FILES)
    assert saved['summary']['resources']['process_wall_s']['values'] == [1.] * 3


def test_other_size_acceptance_is_rejected_before_credentials_or_output(size_fixture, monkeypatch):
    name, _, accepted, path, _, _, calls, _, command = size_fixture
    accepted['variant'] = 'tiny' if name == 'small' else 'small'
    path.write_text(json.dumps(accepted))
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('Credential read before same-size gate'))
    with pytest.raises(ValueError, match='own matched two-task'):
        pilot.main(command)
    assert calls == [] and not Path(command[-1]).exists()


def test_failed_size_worker_stops_later_workers_and_preserves_result(size_fixture, monkeypatch):
    _, _, _, _, _, _, calls, _, command = size_fixture
    original = pilot.run_native

    def mismatched(arguments):
        original(arguments)
        path = Path(arguments[arguments.index('--result-dir') + 1]) / 'result.json'
        saved = json.loads(path.read_text())
        saved['initial_state_sha256'] = 'c' * 64
        path.write_text(json.dumps(saved))
        return 0

    monkeypatch.setattr(pilot, 'run_native', mismatched)
    with pytest.raises(ValueError, match='changed accepted initialization'):
        pilot.main(command)
    output = Path(command[-1])
    assert len(calls) == 1
    saved = json.loads((output / 'execution.json').read_text())
    assert saved['status'] == 'failed' and len(saved['workers']) == 1
    assert (output / 'measurement1/result.json').is_file() and not (output / 'summary.json').exists()


def test_profiler_costs_excluded_and_variant_inferred_from_saved_records(size_fixture):
    name, _, _, _, _, _, _, observation, _ = size_fixture
    summary = pilot.summarize([observation(value=v) for v in (1., 2., 3.)], observation('flops', 999.))
    assert summary['variant'] == name
    assert summary['resources']['process_wall_s']['values'] == [1., 2., 3.]
    assert summary['resources']['gpu_board_energy_wh']['median'] == 2.
    assert summary['same_seed_repeat'] is True and summary['seed_variation_estimated'] is False


@pytest.mark.parametrize('field', ['initial_state_sha256', 'execution_source_sha256', 'initial_learned_sha256'])
def test_mixed_initialization_or_source_snapshots_cannot_be_aggregated(size_fixture, field):
    _, _, _, _, _, _, _, observation, _ = size_fixture
    measurements = [observation() for _ in range(3)]
    measurements[1][field] = 'changed'
    with pytest.raises(ValueError, match='share accepted initialization'):
        pilot.summarize(measurements, observation('flops'))


def test_mixed_base_and_size_measurements_are_rejected(size_fixture):
    _, _, _, _, _, _, _, observation, _ = size_fixture
    with pytest.raises(ValueError, match='Mixed backbone sizes'):
        pilot.summarize([base_record(), observation(), observation()], observation('flops'))


@pytest.mark.parametrize('field,value', [('wandb_logging_status', 'failed'), ('precision', {}),
                                      ('feature_dim', 768), ('gpu', '0')])
def test_incomplete_or_wrong_recipe_size_worker_cannot_enter_summary(size_fixture, field, value):
    name, _, _, _, _, _, _, observation, _ = size_fixture
    bad = observation()
    bad[field] = value
    with pytest.raises(ValueError, match='Size pilot worker'):
        pilot.summarize([bad, observation(), observation()], observation('flops'), variant=name)
