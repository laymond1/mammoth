"""Authorization, RNG and orchestration boundaries for six PROL jobs."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prol_protocol as shared
from scripts import run_prol_suite as suite


def spec(mode='validation', side='native', kind='validation'):
    return {'mode': mode, 'side': side, 'pass': kind, 'protocol': shared.protocol(mode),
            'method': 'prol-prompt', 'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
            'checkpoint_sha256': shared.CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
            'reference_commit': shared.REFERENCE_COMMIT, 'training_authorized': True,
            'source_mode': 'official-active-path', 'paper_equivalent': False,
            'measurement_aggregate_eligible': kind == 'measurement'}


@pytest.mark.parametrize('mode,side,kind', [('validation', 'author', 'validation'),
    ('validation', 'native', 'validation'), ('efficiency', 'native', 'measurement'), ('efficiency', 'native', 'flops')])
def test_scope(mode, side, kind):
    assert shared.validate_spec(spec(mode, side, kind))


@pytest.mark.parametrize('change', [lambda s: s.update(side='author', mode='efficiency', **{'pass': 'measurement'}),
    lambda s: s.update(gpu='0'), lambda s: s.update(training_authorized=1), lambda s: s.update(paper_equivalent=True),
    lambda s: s.update(wandb_project='other'), lambda s: s.update(reference_commit='0' * 40),
    lambda s: s['protocol'].update(n_epochs=2), lambda s: s['protocol'].update(effective_lr=.0125),
    lambda s: s.update(measurement_aggregate_eligible=True), lambda s: s['protocol'].update(tasks=10)])
def test_changed_scope_rejected(change):
    value = spec()
    change(value)
    with pytest.raises(ValueError):
        shared.validate_spec(value)


@pytest.mark.parametrize('key,value', [('training_authorized', False), ('maximum_new_gpu_runs', 7),
    ('automatic_retries_authorized', True), ('normality_review_required', False),
    ('first_task_minimum_accuracy_pct', 20.), ('job_names', ['measurement1']), ('protocols', {})])
def test_changed_run_plan_rejected(monkeypatch, tmp_path, key, value):
    plan = shared.load_run_plan()
    plan[key] = value
    path = tmp_path / 'tampered.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(shared, 'RUN_PLAN', path)
    with pytest.raises(ValueError):
        shared.load_run_plan()


def test_common_profile_cannot_change_independently(monkeypatch, tmp_path):
    plan = shared.load_run_plan()
    plan['common']['checkpoint']['file'] = '/tmp/wrong.npz'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(shared, 'RUN_PLAN', path)
    with pytest.raises(ValueError):
        shared.load_run_plan()


def test_six_jobs_only_and_no_flops_in_timing_repeats(tmp_path):
    jobs = suite.jobs(tmp_path, tmp_path / 'initial.pt')
    assert len(jobs) == 6 and [job['name'] for job in jobs] == list(shared.JOB_NAMES)
    assert [(job['side'], job['mode']) for job in jobs[:2]] == [('author', 'validation'), ('native', 'validation')]
    assert sum(job['pass'] == 'measurement' for job in jobs) == 3
    assert jobs[-1]['pass'] == 'flops'
    assert all(job['status'] == 'pending' for job in jobs)
    for job in jobs:
        assert job['command'][-1] == '--execute'


def test_epoch_boundaries_and_explicit_native_cli():
    command = shared.native_command(spec(), Path('/tmp/prol-results'))
    for flag, value in (('--n_epochs', '1'), ('--stop_after', '2'), ('--batch_size', '64'),
                        ('--lr', '0.01'), ('--dataset_config', 'ease'), ('--device', 'cuda:0'),
                        ('--eval_future', '0'), ('--enable_other_metrics', '0'), ('--joint', '0')):
        assert command[command.index(flag) + 1] == value
    assert command[command.index('--results_path') + 1] == '/tmp/prol-results/mammoth'
    assert shared.native_command(spec('efficiency', kind='measurement'))[command.index('--stop_after') + 1] == '1'


def test_dry_suite_does_not_read_credentials_or_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry planning must not call services or start a process')
    monkeypatch.setattr(suite, 'configure_environment', forbidden)
    monkeypatch.setattr(suite, 'notify_slack', forbidden)
    monkeypatch.setattr(suite.subprocess, 'Popen', forbidden)
    assert suite.main([]) == 0


def test_phase_rng_and_loader_generator_match_previous_common_protocol():
    import torch
    import numpy as np
    from scripts import low_rank_protocol
    for task in (0, 1):
        for phase in ('task_begin', 'train', 'task_end', 'evaluation'):
            assert shared.phase_seed(1993, task, phase) == low_rank_protocol.phase_seed(1993, task, phase)
            shared.seed_phase(1993, task, phase)
            first = torch.randn(3), np.random.random(3)
            shared.seed_phase(1993, task, phase)
            assert torch.equal(first[0], torch.randn(3)) and np.array_equal(first[1], np.random.random(3))
    with pytest.raises(ValueError):
        shared.phase_seed(1993, 2, 'train')


@pytest.mark.parametrize('status', ['failed', 'running', 'validation_gate_review', 'validation_gate_fail'])
def test_suite_never_resumes_failed_or_unreviewed_work(monkeypatch, tmp_path, status):
    initial = tmp_path / 'initial.pt'
    initial.write_bytes(b'initial evidence')
    output = tmp_path / 'runs'
    output.mkdir()
    sources = {'frozen.py': 'a' * 64}
    saved = {'plan': shared.load_run_plan(), 'execution_source_sha256': sources,
             'initial_state_sha256': shared.file_hash(initial), 'jobs': suite.jobs(output, initial), 'status': status}
    (output / 'execution.json').write_text(json.dumps(saved))
    monkeypatch.setattr(suite, 'configure_native_environment', lambda: None)
    monkeypatch.setattr(suite, 'configure_environment', lambda: {'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(shared, 'execution_sources', lambda: sources)
    monkeypatch.setattr(shared, 'verify_reference_source', lambda: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Rejected resume must not contact services or start another worker')
    monkeypatch.setattr(suite, 'notify_slack', forbidden)
    monkeypatch.setattr(suite.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='no automatic retry'):
        suite.execute(SimpleNamespace(initial_state=initial, resume=output, output=output, through='flops'))
    assert json.loads((output / 'execution.json').read_text()) == saved


def test_pilot_requires_a_pinned_human_readable_normality_decision(tmp_path, monkeypatch):
    from scripts import report_prol_validation as report
    for name in shared.JOB_NAMES[:2]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='loss-trend normality review'):
        suite.require_normality_review(tmp_path)
    receipt = {'status': 'review', 'rationale': 'Old-class accuracy needs diagnosis'}
    path = tmp_path / 'normality_review.json'
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(report, 'validate_normality_review', lambda *args: receipt)
    with pytest.raises(ValueError, match='did not pass'):
        suite.require_normality_review(tmp_path)
    receipt['status'] = 'pass'
    path.write_text(json.dumps(receipt))
    assert suite.require_normality_review(tmp_path) == {'file': str(path.resolve()), 'sha256': shared.file_hash(path), 'status': 'pass'}
