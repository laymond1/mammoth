"""Inference-only authorization and immutable original-run provenance."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import prol_diagnostic_protocol as shared
from scripts import run_prol_diagnostic as launcher


def spec():
    return {'schema_version': 1, 'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic',
            'training_authorized': False, 'diagnostic_authorized': True,
            'source_mode': 'official-active-path', 'paper_equivalent': False,
            'gpu': '1', 'gpu_uuid': shared.GPU_UUID, 'wandb_mode': 'online',
            'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'protocol': shared.protocol(),
            'checkpoint_sha256': shared.CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
            'reference_commit': shared.REFERENCE_COMMIT, 'origin_execution_commit': shared.ORIGIN_EXECUTION_COMMIT,
            'prior_result_sha256': shared.PRIOR_PINS['result.json'],
            'prior_predictions_sha256': shared.PRIOR_PINS['task1_predictions.npz'],
            'prior_checkpoint_sha256': shared.PRIOR_PINS['task1_state.pt'],
            'measurement_aggregate_eligible': False}


def test_exact_single_process_inference_matrix():
    plan = shared.load_plan()
    value = spec()
    assert shared.validate_spec(value)
    p = value['protocol']
    assert plan['maximum_new_gpu_processes'] == 1
    assert len(p['model_sides']) * len(p['hook_modes']) * p['repetitions_per_cell'] == p['model_evaluations'] == 40
    assert p['encoder_forward_calls'] == 80 and p['diagnostic_image_visits'] == 1600
    assert p['paired_reference_clone_comparisons'] == 20
    assert p['training_examples'] == p['training_updates'] == p['optimizer_calls'] == p['backward_calls'] == 0
    assert not p['original_live_logits_available'] and not p['new_cl_accuracy']


@pytest.mark.parametrize('mutate', [lambda s: s.update(training_authorized=True),
    lambda s: s.update(diagnostic_authorized=1), lambda s: s.update(gpu='0'),
    lambda s: s.update(measurement_aggregate_eligible=True), lambda s: s.update(paper_equivalent=True),
    lambda s: s.update(wandb_mode='offline'), lambda s: s.update(prior_checkpoint_sha256='0' * 64),
    lambda s: s['protocol'].update(repetitions_per_cell=3), lambda s: s['protocol'].update(training_updates=1),
    lambda s: s['protocol'].update(gradient_enabled=True), lambda s: s['protocol'].update(model_evaluations=80),
    lambda s: s['protocol'].update(original_live_logits_available=True), lambda s: s['protocol'].update(batch_size=True)])
def test_scope_changes_and_bool_integer_substitution_rejected(mutate):
    value = spec()
    mutate(value)
    with pytest.raises(ValueError):
        shared.validate_spec(value)


@pytest.mark.parametrize('key,value', [('maximum_new_gpu_processes', 2), ('training_authorized', True),
    ('automatic_retries_authorized', True), ('subsequent_training_authorized', True),
    ('gate_changes_authorized', True), ('paper_correction_authorized', True)])
def test_plan_does_not_authorize_retraining_or_looser_gates(monkeypatch, tmp_path, key, value):
    plan = shared.load_plan()
    plan[key] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(shared, 'PLAN', path)
    with pytest.raises(ValueError):
        shared.load_plan()


def test_original_git_bytes_remain_verifiable_after_new_local_edits(monkeypatch):
    original = b'original frozen execution code\n'
    digest = hashlib.sha256(original).hexdigest()
    calls = []
    def git_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=original)
    monkeypatch.setattr(shared.subprocess, 'run', git_run)
    record = {'git_commit': shared.ORIGIN_EXECUTION_COMMIT, 'execution_source_sha256': {'scripts/old.py': digest}}
    assert shared.historical_sources(record) == record['execution_source_sha256']
    assert calls == [['git', 'show', shared.ORIGIN_EXECUTION_COMMIT + ':scripts/old.py']]
    record['execution_source_sha256']['scripts/old.py'] = '0' * 64
    with pytest.raises(ValueError, match='fingerprint'):
        shared.historical_sources(record)


@pytest.mark.parametrize('path', ['/tmp/secret.py', '../outside.py'])
def test_historical_paths_cannot_escape_repository(monkeypatch, path):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid provenance paths must be rejected before invoking Git')
    monkeypatch.setattr(shared.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='source path'):
        shared.historical_sources({'git_commit': shared.ORIGIN_EXECUTION_COMMIT,
                                  'execution_source_sha256': {path: '0' * 64}})


def test_dry_launcher_reads_no_credentials_and_starts_no_process(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry plan must not call services or spawn a worker')
    for name in ('configure_environment', 'notify_slack', 'gpu_occupancy'):
        monkeypatch.setattr(launcher, name, forbidden)
    monkeypatch.setattr(launcher.subprocess, 'Popen', forbidden)
    assert launcher.main([]) == 0


def test_existing_directory_is_preserved_without_second_gpu_process(monkeypatch, tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'sentinel').write_text('retained failure')
    monkeypatch.setattr(launcher, 'configure_native_environment', lambda: None)
    monkeypatch.setattr(launcher, 'configure_environment', lambda: {})
    monkeypatch.setattr(launcher, 'build_spec', lambda *args: {'output': str(output)})
    monkeypatch.setattr(launcher, 'gpu_occupancy', lambda: {'gpu_uuid': shared.GPU_UUID})
    def forbidden(*args, **kwargs):
        raise AssertionError('Existing evidence must not be overwritten or retried')
    monkeypatch.setattr(launcher, 'notify_slack', forbidden)
    monkeypatch.setattr(launcher.subprocess, 'Popen', forbidden)
    with pytest.raises(FileExistsError):
        launcher.execute(SimpleNamespace(output=output, timeout=900))
    assert (output / 'sentinel').read_text() == 'retained failure'


def test_single_worker_failure_records_one_attempt_and_never_retries(monkeypatch, tmp_path):
    output = tmp_path / 'one-attempt'
    sources = {'frozen.py': 'a' * 64}
    monkeypatch.setattr(launcher, 'configure_native_environment', lambda: None)
    monkeypatch.setattr(launcher, 'configure_environment', lambda: {})
    monkeypatch.setattr(launcher, 'build_spec', lambda *args: {'output': str(output), 'execution_source_sha256': sources})
    monkeypatch.setattr(launcher, 'gpu_occupancy', lambda: {'gpu_uuid': shared.GPU_UUID})
    monkeypatch.setattr(shared, 'execution_sources', lambda: sources)
    notifications = []
    monkeypatch.setattr(launcher, 'notify_slack', lambda message: notifications.append(message) or 'sent')
    attempts = []
    class FailedWorker:
        returncode = 1
        def __init__(self, command, **kwargs):
            attempts.append(command)
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '1'
            assert kwargs['env']['CL_PROL_DIAGNOSTIC_EXECUTE'] == '1'
            assert kwargs['env']['WANDB_MODE'] == 'online'
        def poll(self):
            return self.returncode
    monkeypatch.setattr(launcher.subprocess, 'Popen', FailedWorker)
    with pytest.raises(RuntimeError, match='preserve partial evidence'):
        launcher.execute(SimpleNamespace(output=output, timeout=900))
    saved = json.loads((output / 'execution.json').read_text())
    assert len(attempts) == saved['new_gpu_processes'] == 1
    assert saved['status'] == 'failed' and saved['new_training'] is False
    assert '--worker' in attempts[0] and not any('main.py' in arg for arg in attempts[0])
    assert saved['slack_failure_status'] == 'sent' and len(notifications) == 2
