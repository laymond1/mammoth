"""Preparation preserves phase budgets and has no experiment side effects."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import prepare_hide_prompt


def test_prepare_records_pending_two_stage_protocol(tmp_path):
    before = dict(os.environ)
    plan = prepare_hide_prompt.prepare(tmp_path)
    assert dict(os.environ) == before
    assert json.loads((tmp_path / 'plan.json').read_text()) == plan
    for key in ('executed', 'experiment_executed', 'training_executed', 'image_inference_executed',
                'gpu_computation_executed', 'downloads_executed', 'wandb_messages_sent', 'slack_messages_sent'):
        assert plan[key] is False
    assert plan['execution_status'] == 'pending_next_stage_authorization'
    assert plan['reference']['commit'] == '0ed433334ff2cfb206c5e2899651c3044ec85f92'
    assert plan['reference']['runner_implemented'] is False
    assert plan['validation']['tii_epochs'] == 20
    assert plan['validation']['prompt_epochs'] == 50
    assert plan['validation']['ca_epochs'] == 30
    assert plan['efficiency']['tii_epochs'] == plan['efficiency']['prompt_epochs'] == 1
    assert plan['efficiency']['ca_epochs'] == 30
    assert plan['efficiency']['equal_compute_budget_to_existing_one_epoch'] is False
    assert plan['validation']['dataset_config'] == 'hide_prompt'
    assert plan['efficiency']['dataset_config'] == 'ease'
    for command in plan['candidate_commands']:
        argv = command['native_main_argv']
        assert argv[argv.index('--n_epochs') + 1] == ('70' if command['name'] == 'validation' else '2')
        assert argv[argv.index('--nowand') + 1] == '1'
        assert command['training_authorized'] is False
        assert command['reference_runner_implemented'] is False
    assert plan['common']['runtime_environment']['CUDA_VISIBLE_DEVICES'] == '1'
    assert plan['logging']['entity'] == 'laymond1'
    assert 'https://hooks.slack.com' not in json.dumps(plan)


def test_preparer_runs_without_site_packages_or_ml_imports(tmp_path):
    script = Path(prepare_hide_prompt.__file__)
    code = ("import runpy,sys;sys.argv=['prepare_hide_prompt.py','--output',sys.argv[1]];"
            f"runpy.run_path({str(script)!r},run_name='__main__')")
    result = subprocess.run([sys.executable, '-S', '-c', code, str(tmp_path)],
                            check=True, capture_output=True, text=True)
    assert 'executed=False' in result.stdout
    assert (tmp_path / 'plan.json').is_file()


def test_local_checkpoint_hashing_never_downloads(tmp_path):
    protocol = json.loads(prepare_hide_prompt.PROTOCOL.read_text())
    checkpoint = tmp_path / 'cached.npz'
    checkpoint.write_bytes(b'test-local-checkpoint')
    protocol['checkpoint'].update(file=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    plan = prepare_hide_prompt.prepare(tmp_path / 'prepared', protocol_path=path, check_local=True)
    assert plan['local_checkpoint']['checked'] is True
    assert plan['local_checkpoint']['download_attempted'] is False
    checkpoint.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='SHA-256'):
        prepare_hide_prompt.prepare(tmp_path / 'bad', protocol_path=path, check_local=True)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        prepare_hide_prompt.prepare(tmp_path / 'missing', protocol_path=path, check_local=True)
    assert not (tmp_path / 'missing' / 'plan.json').exists()


@pytest.mark.parametrize('change', ['training_authorized', 'runner_implemented'])
def test_preparer_rejects_scope_drift(tmp_path, change):
    protocol = copy.deepcopy(json.loads(prepare_hide_prompt.PROTOCOL.read_text()))
    if change == 'training_authorized':
        protocol[change] = True
    else:
        protocol['reference'][change] = True
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        prepare_hide_prompt.prepare(tmp_path / 'prepared', protocol_path=path)
    assert not (tmp_path / 'prepared').exists()


def test_execution_flag_is_not_an_available_option():
    with pytest.raises(SystemExit):
        prepare_hide_prompt.main(['--execute'])
