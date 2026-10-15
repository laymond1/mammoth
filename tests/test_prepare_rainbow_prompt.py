"""Prepared Rainbow plans stay unexecuted and record the pending pilot decision."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import prepare_rainbow_prompt as preparer


def test_plan_preparation_scope_and_shared_conditions(tmp_path):
    environment = dict(os.environ)
    plan = preparer.prepare(tmp_path)
    assert dict(os.environ) == environment
    assert json.loads((tmp_path / 'plan.json').read_text()) == plan
    assert plan['training_authorized'] is False
    for name in ('executed', 'experiment_executed', 'training_executed', 'gpu_computation_executed',
                 'downloads_executed', 'wandb_messages_sent', 'slack_messages_sent'):
        assert plan[name] is False
    assert plan['pilot']['selected_option'] is None
    assert plan['reference_runner']['status'] == 'planned_not_implemented'
    assert plan['method_settings']['effective_lr'] == .0075
    assert plan['method_settings']['learned_gate_active'] is False
    argv = plan['candidate_native_argv']
    for key, value in [('--n_epochs', '20'), ('--batch_size', '64'), ('--stop_after', '2'),
                       ('--rainbow_warmup_epochs', '5'), ('--wandb_entity', 'laymond1')]:
        assert argv[argv.index(key) + 1] == value
    assert 'https://hooks.slack.com' not in json.dumps(plan)
    from scripts.cl_runtime import public_setting
    assert plan['common']['data_root'] == public_setting('data_root')
    assert plan['common']['gpu'] == '1' and plan['common']['wandb_entity'] == 'laymond1'


def test_preparer_standard_library_only_and_no_execution_flag(tmp_path):
    result = subprocess.run([sys.executable, '-S', str(Path(preparer.__file__)), '--output', str(tmp_path)],
                            check=True, capture_output=True, text=True)
    assert 'executed=False' in result.stdout
    with pytest.raises(SystemExit):
        preparer.main(['--execute'])


def write_protocols(tmp_path):
    execution = copy.deepcopy(json.loads(preparer.PROTOCOL.read_text()))
    validation = copy.deepcopy(json.loads(preparer.VALIDATION.read_text()))
    first, second = tmp_path / 'execution.json', tmp_path / 'validation.json'
    return execution, validation, first, second


def test_existing_checkpoint_hash_is_read_only(tmp_path):
    execution, validation, first, second = write_protocols(tmp_path)
    checkpoint = tmp_path / 'local.npz'
    checkpoint.write_bytes(b'local-checkpoint-fixture')
    for plan in (execution, validation):
        plan['common']['checkpoint'].update(file=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    first.write_text(json.dumps(execution))
    second.write_text(json.dumps(validation))
    plan = preparer.prepare(tmp_path / 'prepared', first, second, check_local=True)
    assert plan['local_checkpoint']['checked'] is True
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA-256'):
        preparer.prepare(tmp_path / 'bad', first, second, check_local=True)
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('change', ['authorization', 'gate', 'warmup', 'runner', 'selected', 'mismatch'])
def test_preparation_rejects_protocol_drift(tmp_path, change):
    execution, validation, first, second = write_protocols(tmp_path)
    if change == 'authorization':
        execution['training_authorized'] = True
    elif change == 'runner':
        execution['reference_runner']['status'] = 'ready'
    elif change == 'selected':
        execution['pilot']['selected_option'] = 'one_epoch_warmup0'
    elif change == 'mismatch':
        execution['common']['batch_size'] = 32
    else:
        for plan in (execution, validation):
            plan['method_settings']['learned_gate_active' if change == 'gate' else 'warmup_epochs'] = True if change == 'gate' else 20
    first.write_text(json.dumps(execution))
    second.write_text(json.dumps(validation))
    with pytest.raises(ValueError):
        preparer.prepare(tmp_path / 'bad', first, second)
    assert not (tmp_path / 'bad').exists()
