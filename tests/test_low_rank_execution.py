"""Training authorization boundaries and immutable reference prerequisites."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from scripts import low_rank_protocol as protocol
from scripts import run_low_rank_validation as runner


def valid_spec():
    return {'method': 'ewc-lora', 'mode': 'validation', 'tasks': 2, 'protocol': protocol.protocol('ewc-lora'),
            'gpu': '1', 'checkpoint_sha256': protocol.CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': protocol.CANONICAL_PRETRAINED_SHA256,
            'reference_commit': protocol.REFERENCE_COMMIT, 'wandb_entity': 'laymond1',
            'wandb_project': 'cl-efficiency', 'wandb_mode': 'online'}


def test_dry_run_never_loads_credentials_or_launches_training(monkeypatch, capsys):
    monkeypatch.setattr(runner, 'configure_environment', lambda *args: pytest.fail('dry-run credentials'))
    monkeypatch.setattr(runner, 'execute', lambda *args: pytest.fail('dry-run training'))
    assert runner.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out


def test_dry_run_is_stdlib_only():
    command = 'import sys; from scripts.run_low_rank_validation import main; main([]); assert "torch" not in sys.modules; assert "wandb" not in sys.modules; assert "numpy" not in sys.modules'
    result = subprocess.run([sys.executable, '-S', '-c', command], cwd=runner.ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_worker_refuses_missing_parent_execution_guard(monkeypatch, tmp_path):
    monkeypatch.delenv('CL_LOW_RANK_NATIVE_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit --execute'):
        runner.worker(tmp_path / 'missing.json')


@pytest.mark.parametrize('key,value', [('tasks', 10), ('gpu', '0'), ('mode', 'efficiency'), ('method', 'sgd'),
                                     ('wandb_mode', 'offline'), ('wandb_entity', 'other'), ('checkpoint_sha256', 'other')])
def test_spec_refuses_scope_expansion_and_changed_destination(key, value):
    spec = valid_spec()
    spec[key] = value
    with pytest.raises(ValueError, match='approved first-two-task'):
        runner.validate_spec(spec)


@pytest.mark.parametrize('key,value', [('n_epochs', 1), ('batch_size', 128), ('seed', 0), ('num_workers', 16), ('rank', 4)])
def test_spec_refuses_protocol_changes(key, value):
    spec = valid_spec()
    spec['protocol'][key] = value
    with pytest.raises(ValueError, match='approved first-two-task'):
        runner.validate_spec(spec)


def test_prediction_reference_counts_are_verified_against_saved_predictions(tmp_path):
    import numpy as np
    paths, rows = [], []
    for task in (1, 2):
        path = tmp_path / f'task{task}.npz'
        labels = np.repeat(np.arange(task * 10), 100)
        np.savez(path, predictions=labels, labels=labels, global_ids=np.arange(task * 1000))
        paths.append({'task': task, 'path': path.name, 'sha256': protocol.file_hash(path)})
        rows.append({'task': task, **protocol.counts_from_predictions(labels, labels, past_classes=(task - 1) * 10, seen_classes=task * 10)})
    reference = {'prediction_files': paths, 'task_results': rows}
    assert len(runner.prediction_pins(reference, tmp_path / 'summary.json')) == 2
    reference['task_results'][1]['old_correct'] -= 1
    with pytest.raises(ValueError, match='disagree'):
        runner.prediction_pins(reference, tmp_path / 'summary.json')


def test_worker_rejects_wrong_gpu_before_loading_ml(monkeypatch, tmp_path):
    monkeypatch.setenv('CL_LOW_RANK_NATIVE_EXECUTE', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('WANDB_MODE', 'online')
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(valid_spec()))
    monkeypatch.setattr(runner, 'source_hashes', lambda: pytest.fail('must reject device before source reads'))
    with pytest.raises(ValueError, match='GPU1'):
        runner.worker(path)


def independent_reference():
    spec = valid_spec()
    rows = [{'task': 1, 'seen_correct': 900, 'seen_samples': 1000, 'old_correct': 0,
             'old_samples': 0, 'new_correct': 900, 'new_samples': 1000},
            {'task': 2, 'seen_correct': 1800, 'seen_samples': 2000, 'old_correct': 900,
             'old_samples': 1000, 'new_correct': 900, 'new_samples': 1000}]
    return {**spec, 'status': 'completed', 'implementation': 'pinned_author_common_input_reference',
            'initial_state_sha256': 'initial-pin', 'precision': protocol.PRECISION,
            'reference_source_sha256': protocol.load_plan()['reference_source_sha256'],
            'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/verified', 'wandb_logging_status': 'uploaded',
            'structural_checks': {name: True for name in protocol.REQUIRED_STRUCTURAL_CHECKS}, 'task_results': rows}


def test_reference_must_pass_all_structures_before_native_launch():
    reference = independent_reference()
    runner.validate_reference(reference, 'ewc-lora', 'initial-pin')
    reference['structural_checks']['method_state_valid'] = False
    with pytest.raises(ValueError, match='passing independent reference'):
        runner.validate_reference(reference, 'ewc-lora', 'initial-pin')


@pytest.mark.parametrize('key,value', [('status', 'failed'), ('method', 'sd-lora'),
                                     ('initial_state_sha256', 'changed'), ('wandb_logging_status', 'offline'),
                                     ('reference_commit', 'changed'), ('implementation', 'mammoth_native')])
def test_reference_rejects_unfinished_mismatched_or_nonindependent_results(key, value):
    reference = independent_reference()
    reference[key] = value
    with pytest.raises(ValueError, match='passing independent reference'):
        runner.validate_reference(reference, 'ewc-lora', 'initial-pin')
