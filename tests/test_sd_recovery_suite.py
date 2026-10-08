"""The recovery gate must stop on review, collapse or structural failures."""
import pytest

from scripts import run_sd_recovery_suite as suite
from scripts.run_sd_recovery_suite import accuracy_gate, jobs, main


@pytest.mark.parametrize('accuracy,status', [(100, 'pass'), (99, 'pass'), (97.5, 'pass'),
                                           (97.4, 'review'), (96, 'review'), (95.9, 'fail'), (10, 'fail')])
def test_relative_accuracy_boundaries(accuracy, status):
    assert accuracy_gate(accuracy)['status'] == status


def test_absolute_accuracy_floor_and_structural_checks():
    assert accuracy_gate(89.9, 90)['status'] == 'fail'
    assert accuracy_gate(90, 90)['status'] == 'pass'
    assert accuracy_gate(99, structural_checks_pass=False)['status'] == 'fail'


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1, 101])
def test_invalid_accuracy(value):
    with pytest.raises(ValueError):
        accuracy_gate(value)


def test_distinct_five_job_schedule_and_dry_default(tmp_path, capsys):
    schedule = jobs(tmp_path)
    assert [job['pass'] for job in schedule] == ['validation', 'measurement', 'measurement', 'measurement', 'flops']
    assert len({job['directory'] for job in schedule}) == 5
    assert main(['--output', str(tmp_path / 'unused')]) == 0
    assert '"execute": false' in capsys.readouterr().out
    assert not (tmp_path / 'unused').exists()


def test_plan_rejects_expanded_budget_or_changed_recipe(tmp_path, monkeypatch):
    import json
    original = suite.load_plan()
    monkeypatch.setattr(suite, 'PLAN', tmp_path / 'plan.json')
    for key, value in [('maximum_new_gpu_runs', 6), ('encoder_and_scale_lr', .008), ('tasks', 2), ('training_authorized', False)]:
        suite.PLAN.write_text(json.dumps({**original, key: value}))
        with pytest.raises(ValueError):
            suite.load_plan()
