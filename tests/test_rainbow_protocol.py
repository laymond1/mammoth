"""Rainbow execution scope, source integrity and guarded staged schedule."""

import copy
import json

import pytest

from scripts import rainbow_protocol as protocol
from scripts import run_rainbow_suite as suite


def spec(mode='full_validation', side='author', kind='validation'):
    return {'method': 'rainbow-prompt', 'mode': mode, 'side': side, 'pass': kind,
            'protocol': protocol.protocol(mode), 'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
            'checkpoint_sha256': protocol.CHECKPOINT_SHA256, 'canonical_pretrained_sha256': protocol.CANONICAL_PRETRAINED_SHA256,
            'reference_commit': protocol.REFERENCE_COMMIT, 'training_authorized': True,
            'source_mode': 'official-active-path', 'learned_gate_active': False, 'paper_equivalent': False,
            'measurement_aggregate_eligible': kind == 'measurement'}


def test_scopes_and_source_pin():
    assert protocol.load_run_plan()['maximum_new_gpu_runs'] == 8
    full, short = protocol.protocol('full_validation'), protocol.protocol('short_validation')
    assert (full['tasks'], full['n_epochs'], full['warmup_epochs']) == (2, 20, 5)
    assert (short['tasks'], short['n_epochs'], short['warmup_epochs']) == (1, 1, 0)
    assert full['effective_lr'] == .0075
    if protocol.REFERENCE_SOURCE.is_dir():
        assert len(protocol.verify_reference_source()) == 21
    assert len(protocol.REQUIRED_CHECKS) == 17


@pytest.mark.parametrize('change,value', [('gpu', '0'), ('learned_gate_active', True), ('training_authorized', False),
                                        ('paper_equivalent', True), ('measurement_aggregate_eligible', True),
                                        ('source_mode', 'paper')])
def test_spec_rejects_changed_scope(change, value):
    value_spec = spec()
    protocol.validate_spec(value_spec)
    value_spec[change] = value
    with pytest.raises(ValueError):
        protocol.validate_spec(value_spec)


def test_flops_is_a_separate_ineligible_native_run():
    value = spec('efficiency', 'native', 'flops')
    assert protocol.validate_spec(value)
    value['side'] = 'author'
    with pytest.raises(ValueError):
        protocol.validate_spec(value)


def test_job_order_and_dry_default(tmp_path, capsys):
    schedule = suite.jobs(tmp_path, tmp_path / 'initial.pt')
    assert [item['name'] for item in schedule] == list(protocol.JOB_NAMES)
    assert len({item['directory'] for item in schedule}) == 8
    assert schedule[-1]['pass'] == 'flops' and schedule[-2]['pass'] == 'measurement'
    assert suite.main(['--output', str(tmp_path / 'unused')]) == 0
    assert '"execute": false' in capsys.readouterr().out
    assert not (tmp_path / 'unused').exists()


def test_plan_rejects_extra_runs_or_changed_warmup(tmp_path, monkeypatch):
    plan = protocol.load_run_plan()
    monkeypatch.setattr(protocol, 'PLAN', tmp_path / 'plan.json')
    for change in ('budget', 'recipe'):
        altered = copy.deepcopy(plan)
        if change == 'budget':
            altered['maximum_new_gpu_runs'] = 9
        else:
            altered['protocols']['short_validation']['warmup_epochs'] = 5
        protocol.PLAN.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            protocol.load_run_plan()


def test_native_command_has_effective_scope_and_no_legacy_nowand_flag():
    args = protocol.native_command(spec('short_validation', 'native'))
    assert args[args.index('--n_epochs') + 1] == '1'
    assert args[args.index('--rainbow_warmup_epochs') + 1] == '0'
    assert args[args.index('--validation') + 1] == '0'
    assert '--nowand' not in args


@pytest.mark.parametrize('gate', ['review', 'fail'])
def test_suite_stops_before_short_runs_on_full_gate(tmp_path, monkeypatch, gate):
    from types import SimpleNamespace
    initial = tmp_path / 'initial.pt'
    initial.write_bytes(b'synthetic-initial')
    monkeypatch.setattr(protocol, 'execution_sources', lambda: {'synthetic_source': 'unchanged'})
    monkeypatch.setattr(protocol, 'verify_reference_source', lambda: {})
    monkeypatch.setattr(suite, 'configure_environment', lambda: {'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(suite, 'gpu_occupancy', lambda: {'gpu_uuid': 'GPU-1fb8c05f-24e7-75d1-5cf9-0f47df8e46ad'})
    monkeypatch.setattr(suite, 'notify_slack', lambda *args, **kwargs: 'sent')
    calls = []
    class CompletedWorker:
        def __init__(self, command, **kwargs):
            name = command[command.index('--output') + 1]
            directory = suite.Path(name)
            directory.mkdir()
            calls.append(directory.name)
            record = {'status': 'completed', 'wandb_logging_status': 'uploaded', 'side': 'author' if directory.name.endswith('_author') else 'native',
                      'mode': 'full_validation', 'pass': 'validation', 'protocol': protocol.protocol('full_validation'),
                      'precision': protocol.PRECISION, 'structural_checks': dict.fromkeys(protocol.REQUIRED_CHECKS, True),
                      'metrics': {'outer_observe_steps': 3160, 'outer_training_stream_examples': 200000,
                                  'tasks_completed': 2, 'evaluation_examples': 3000},
                      'task_results': [{'seen_accuracy_pct': 99}], 'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/synthetic'}
            (directory / 'result.json').write_text(json.dumps(record))
        def wait(self):
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(suite.subprocess, 'Popen', CompletedWorker)
    monkeypatch.setattr(suite, 'pair_acceptance', lambda *args: {'status': gate})
    options = SimpleNamespace(initial_state=initial, output=tmp_path / 'runs', resume=None, through='flops')
    assert suite.execute(options) == 1
    assert calls == ['full_author', 'full_native']
    saved = json.loads((options.output / 'execution.json').read_text())
    assert saved['status'] == 'full_gate_' + gate
    assert all(job['status'] == 'pending' for job in saved['jobs'][2:])
