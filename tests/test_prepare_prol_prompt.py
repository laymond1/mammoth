"""Preparation is strictly local and never authorizes proposed GPU work."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.prepare_prol_prompt import prepare
from scripts.prol_protocol import JOB_NAMES, PLAN, load_preparation_plan, native_command, verify_reference_source


def test_preparation_six_jobs_and_core_cli(tmp_path):
    output = tmp_path / 'plan.json'
    record = prepare(output)
    assert json.loads(output.read_text()) == record
    assert not record['training_authorized'] and not record['experiment_executed']
    assert [job['name'] for job in record['jobs']] == list(JOB_NAMES)
    assert sum(job['aggregate_eligible'] for job in record['jobs']) == 3
    assert sum(job['profile_flops'] for job in record['jobs']) == 1
    assert record['jobs'][0]['native_training_core_command'] is None
    assert [job['tasks'] for job in record['jobs']] == [2, 2, 1, 1, 1, 1]
    assert all(job['n_epochs_per_task'] == 1 for job in record['jobs'])
    assert all('pending' in job['runner_status'] for job in record['jobs'])
    command = native_command(load_preparation_plan(), 2)
    for flag, value in (('--batch_size', '64'), ('--lr', '0.01'), ('--n_epochs', '1'),
                        ('--stop_after', '2'), ('--nowand', '1'), ('--dataset_config', 'ease')):
        assert command[command.index(flag) + 1] == value
    assert not record['services']['credentials_read']
    assert not record['services']['remote_writes_executed']


@pytest.mark.parametrize('key,value', [('training_authorized', True), ('training_authorized', 0),
    ('experiment_executed', True), ('reference_commit', 'wrong'), ('maximum_proposed_gpu_runs', 8),
    ('job_names', []), ('source_mode', 'paper-corrected')])
def test_reject_mutated_scope(tmp_path, key, value):
    plan = json.loads(PLAN.read_text())
    plan[key] = value
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        prepare(tmp_path / 'output.json', path)
    assert not (tmp_path / 'output.json').exists()


@pytest.mark.parametrize('group,key,value', [
    ('common', 'batch_size', 10), ('common', 'gpu', '0'), ('common', 'dataset_config', 'l2p'),
    ('common', 'class_order', list(reversed(range(100)))), ('common', 'seed', 0),
    ('common', 'wandb_entity', 'different'), ('method_settings', 'nominal_lr', .05),
    ('method_settings', 'backbone_norms_trainable', False), ('method_settings', 'n_epochs_per_task', 20),
    ('validation', 'required_structural_checks', []),
])
def test_reject_changed_conditions(tmp_path, group, key, value):
    plan = copy.deepcopy(json.loads(PLAN.read_text()))
    plan[group][key] = value
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_preparation_plan(path)


def test_import_has_no_ml_or_service_side_effects():
    code = ('import sys; import scripts.prepare_prol_prompt; '
            'assert not any(x in sys.modules for x in ("torch", "wandb", "requests", "scripts.cl_runtime"))')
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[1])


def test_no_execution_option(tmp_path):
    result = subprocess.run([sys.executable, 'scripts/prepare_prol_prompt.py', '--execute',
                             '--output', str(tmp_path / 'never.json')],
                            capture_output=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode != 0 and not (tmp_path / 'never.json').exists()


def test_reference_hashes_match_all_tracked_files():
    plan = load_preparation_plan()
    if not Path(plan['reference_source']).is_dir():
        pytest.skip('Pinned author source is not present')
    hashes = verify_reference_source(plan['reference_source'])
    assert len(hashes) == 22 and 'engine_prol.py' in hashes and 'configs/cifar100_prol.py' in hashes
