"""Execution tools must preserve the explicitly approved stage boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import check_ease_runtime, run_ease_reference


def test_smoke_without_execute_does_not_load_credentials_or_start_compute(monkeypatch, capsys):
    monkeypatch.setattr(check_ease_runtime, 'configure_environment', lambda: pytest.fail('No execution authorized'))
    assert check_ease_runtime.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out


def test_reference_without_execute_does_not_launch_worker(monkeypatch, capsys):
    monkeypatch.setattr(run_ease_reference, 'configure_environment', lambda: {})
    monkeypatch.setattr(run_ease_reference, 'execute', lambda *args: pytest.fail('No execution authorized'))
    assert run_ease_reference.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out


def test_reference_worker_requires_explicit_parent_authorization(monkeypatch, tmp_path):
    monkeypatch.delenv('CL_EASE_REFERENCE_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit --execute'):
        run_ease_reference.worker(tmp_path / 'absent.json')


@pytest.mark.parametrize('tasks,seed', [(3, 1993), (2, 42)])
def test_reference_worker_rejects_expansion_beyond_approved_protocol(monkeypatch, tmp_path, tasks, seed):
    monkeypatch.setenv('CL_EASE_REFERENCE_EXECUTE', '1')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'tasks': tasks, 'config': {'seed': seed}}))
    monkeypatch.setattr(run_ease_reference, 'Recorder', lambda *args: pytest.fail('Compute must remain blocked'))
    with pytest.raises(ValueError, match='approved two-task seed1993'):
        run_ease_reference.worker(spec)


def test_reference_namespace_packages_win_over_native_regular_packages(tmp_path):
    source = tmp_path / 'reference'
    (source / 'utils').mkdir(parents=True)
    (source / 'utils' / 'data.py').write_text('ORIGIN = "official_fixture"\n')
    command = ('from scripts.run_ease_reference import isolate_reference_imports; '
               f'isolate_reference_imports({str(source)!r}); '
               'from utils import data; assert data.ORIGIN == "official_fixture"; '
               'print(data.__file__)')
    result = subprocess.run([sys.executable, '-c', command], cwd=run_ease_reference.ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert str(source / 'utils' / 'data.py') in result.stdout


def test_native_validation_preparation_cannot_start_credentials_or_compute(monkeypatch, capsys):
    from scripts import run_ease_validation
    monkeypatch.setattr(run_ease_validation, 'configure_environment', lambda: pytest.fail('Preparation must not read credentials'))
    monkeypatch.setattr(run_ease_validation, 'execute', lambda *args: pytest.fail('Preparation must not execute'))
    assert run_ease_validation.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out


def test_native_validation_worker_requires_execute(monkeypatch, tmp_path):
    from scripts import run_ease_validation
    monkeypatch.delenv('CL_EASE_VALIDATION_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit --execute'):
        run_ease_validation.worker(tmp_path / 'absent.json')


def test_size_validation_preparation_cannot_read_plan_credentials_or_start_compute(monkeypatch, capsys):
    from scripts import run_ease_validation
    monkeypatch.setattr(run_ease_validation, 'load_size_variant', lambda *args, **kwargs: pytest.fail('Dry run must not read size plan'))
    monkeypatch.setattr(run_ease_validation, 'configure_environment', lambda: pytest.fail('No credentials during preparation'))
    monkeypatch.setattr(run_ease_validation, 'execute', lambda *args: pytest.fail('No worker during preparation'))
    assert run_ease_validation.main(['--variant', 'small_fixed64', '--reference-summary', 'absent.json']) == 0
    assert 'small_fixed64' in capsys.readouterr().out


@pytest.mark.parametrize('field,value', [('tasks', 3), ('seed', 42), ('n_epochs_per_task', 1),
                                        ('batch_size', 64), ('training_authorized', False)])
def test_native_validation_rejects_unapproved_protocol(field, value):
    from scripts import run_ease_validation
    spec = {'stage': 'native_two_task_acceptance', 'tasks': 2,
            'protocol': {'seed': 1993, 'n_epochs_per_task': 20, 'batch_size': 48,
                         'initial_variant': 'base_fixed64', 'training_authorized': True}}
    if field == 'tasks':
        spec[field] = value
    else:
        spec['protocol'][field] = value
    with pytest.raises(ValueError, match='authorized Base two-task'):
        run_ease_validation.validate_spec(spec)
