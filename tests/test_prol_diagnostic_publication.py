"""CPU-only publication proofs for the single inference-only PROL diagnosis."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.cl_runtime import configure_native_environment
configure_native_environment()

from scripts import publish_prol_diagnostic as publisher
from scripts import report_prol_diagnostic as report_module
from scripts.report_low_rank_results import run_path
from tests.test_prol_diagnostic_report import stage


class SavedArtifact:
    """Exercise real manifest checks without accessing W&B or the network."""
    def __init__(self, files, record, name):
        self.state = 'COMMITTED'
        self.qualified_name = 'laymond1/cl-efficiency/' + name + ':v0'
        self.digest = 'synthetic-artifact-digest'
        self.record = deepcopy(record)
        self.manifest = SimpleNamespace(entries={name: SimpleNamespace(
            digest=hashlib.md5(Path(proof['file']).read_bytes(), usedforsecurity=False).hexdigest(),
            size=proof['bytes'], ref=None) for name, proof in files.items()})

    def get_path(self, name):
        assert name == 'artifact_metrics.json'
        def download(root):
            path = Path(root) / name
            report_module.write_json(path, self.record)
            return str(path)
        return SimpleNamespace(download=download)


class SavedRun:
    def __init__(self, summary, artifact, state='finished'):
        self.state, self.summary = state, deepcopy(summary)
        self.artifacts = [artifact]

    def logged_artifacts(self):
        return self.artifacts


class SavedAPI:
    def __init__(self, runs):
        self.runs = runs
        self.run_calls = []
        self.artifacts = {}

    def run(self, path):
        self.run_calls.append(path)
        return self.runs[path]

    def artifact(self, name):
        return self.artifacts[name]

    def flush(self):
        pass


@pytest.fixture
def publication(stage, tmp_path):
    root, record, prior, execution = stage
    report = report_module.build_report(root)
    directory = report_module.write_report(report, tmp_path / 'report')
    old_files = publisher.historical_publisher.worker_files(report['historical_failed_source'])
    old_summary = {'benchmark_status': 'failed', 'benchmark_pass': 'validation',
        'accuracy/mean_accuracy_pct': prior['accuracy']['mean_accuracy_pct'],
        **{'efficiency/' + name: prior['metrics'][name] for name in
            ('tasks_completed', 'outer_observe_steps', 'outer_training_stream_examples', 'evaluation_examples')}}
    old_artifact = SavedArtifact(old_files, prior, 'original-failed-author')
    old_run = SavedRun(old_summary, old_artifact, state='failed')
    new_files = report_module.worker_files(record, root)
    new_summary = {'diagnostic_status': record['status'], 'new_cl_accuracy': False,
        'measurement_aggregate_eligible': False, **{'counts/' + name: value for name, value in record['counts'].items()}}
    new_artifact = SavedArtifact(new_files, record, 'single-diagnostic')
    new_run = SavedRun(new_summary, new_artifact)
    api = SavedAPI({run_path(prior['wandb_url']): old_run, run_path(record['wandb_url']): new_run})
    return SimpleNamespace(directory=directory, report=report, root=root, record=record, prior=prior,
        api=api, old_run=old_run, new_run=new_run, old_artifact=old_artifact, new_artifact=new_artifact)


def test_default_dry_checks_saved_evidence_without_credentials_services_or_weights(publication, monkeypatch, tmp_path, capsys):
    denied = Mock(side_effect=AssertionError('Dry preparation must not access credentials or services'))
    monkeypatch.setattr(publisher, 'configure_environment', denied)
    monkeypatch.setattr(publisher, 'notify_slack', denied)
    monkeypatch.setattr(publisher, 'publish', denied)
    (publication.root / 'unlisted_model.pt').write_bytes(b'local-only weights')
    report, files = publisher.prepare(publication.directory)
    assert report['status'] == 'completed'
    assert files['evidence/execution.json']['sha256'] == report['execution']['sha256']
    assert 'diagnostic/forward_outputs.npz' in files
    assert len([name for name in files if name.startswith('diagnostic/trace_')]) == 20
    assert not any('tail_inputs' in name or Path(name).suffix == '.pt' for name in files)
    output = tmp_path / 'no-publication-during-dry'
    assert publisher.main(['--report-dir', str(publication.directory), '--output', str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)['execute'] is False
    denied.assert_not_called()


@pytest.mark.parametrize('secret', [{'slack_webhook_url': 'private-value'},
    {'note': 'https://hooks.slack.com/services/private-value'}])
def test_prepare_rejects_credentials_without_echoing_them(publication, secret):
    path = Path(publication.report['historical_failed_source']['file']).parent / 'effective_args.json'
    report_module.write_json(path, secret)
    with pytest.raises(ValueError, match='private credential') as error:
        publisher.prepare(publication.directory)
    assert 'private-value' not in str(error.value)


def test_scalar_summary_keeps_diagnosis_separate_from_training_and_acceptance(publication):
    summary = publisher.scalar_summary(publication.report)
    assert summary['new_gpu_processes'] == summary['diagnostic_worker_count'] == 1
    assert summary['new_training_runs'] == summary['counts/train_steps'] == 0
    assert summary['counts/model_evaluations'] == 40
    assert summary['counts/encoder_forward_calls'] == 80
    for field in ('new_training', 'new_cl_accuracy', 'measurement_aggregate_eligible',
                  'checkpoint_issue_resolved', 'implementation_acceptance_changed', 'subsequent_training_authorized'):
        assert summary[field] is False
    assert summary['historical_failed_worker_count'] == 1


def test_source_readback_verifies_both_immutable_terminal_runs_and_exact_counts(publication, tmp_path):
    original = Path(publication.report['historical_failed_source']['file']).read_bytes()
    proof = publisher.verify_sources(publication.api, publication.report, tmp_path / 'readback')
    assert publisher.validate_source_proofs(proof, publication.report) is None
    assert proof['diagnostic_worker_count'] == proof['historical_failed_worker_count'] == 1
    old, new = proof['workers']
    assert old['local_status'] == old['remote_status'] == 'failed'
    assert old['run_state'] == 'failed'
    assert new['local_status'] == new['remote_status'] == 'completed'
    assert new['run_state'] == 'finished'
    assert new['counts_summary_verified'] and new['new_cl_accuracy'] is False
    assert all(item['matches'] is True for worker in proof['workers'] for item in worker['files'].values())
    assert Path(publication.report['historical_failed_source']['file']).read_bytes() == original
    assert publication.api.run_calls == [run_path(publication.prior['wandb_url']), run_path(publication.record['wandb_url'])]


@pytest.mark.parametrize('field,value', [('counts/model_evaluations', 41), ('counts/train_steps', True),
    ('new_cl_accuracy', True), ('measurement_aggregate_eligible', True), ('diagnostic_status', 'pass')])
def test_remote_summary_cannot_change_workload_or_acceptance(publication, tmp_path, field, value):
    publication.new_run.summary[field] = value
    with pytest.raises(ValueError):
        publisher.verify_diagnostic(publication.api, publication.report, tmp_path / 'readback')


@pytest.mark.parametrize('change', ['running', 'uncommitted', 'duplicate-artifact', 'changed-digest', 'remote-ref'])
def test_diagnostic_requires_one_terminal_run_and_one_matching_direct_manifest(publication, tmp_path, change):
    artifact = publication.new_artifact
    if change == 'running':
        publication.new_run.state = 'running'
    elif change == 'uncommitted':
        artifact.state = 'PENDING'
    elif change == 'duplicate-artifact':
        publication.new_run.artifacts.append(artifact)
    elif change == 'changed-digest':
        artifact.manifest.entries['forward_outputs.npz'].digest = '0' * 32
    else:
        artifact.manifest.entries['forward_outputs.npz'].ref = 'https://untrusted.invalid/evidence'
    with pytest.raises(ValueError):
        publisher.verify_diagnostic(publication.api, publication.report, tmp_path / 'readback')


@pytest.mark.parametrize('field', ['counts', 'precision', 'matrix'])
def test_manifest_match_alone_does_not_accept_changed_remote_semantics(publication, tmp_path, field):
    if field == 'matrix':
        publication.new_artifact.record[field][0]['rng_unchanged'] = False
    elif field == 'counts':
        publication.new_artifact.record[field]['optimizer_updates'] = 1
    else:
        publication.new_artifact.record[field] = {'dtype': 'float16'}
    with pytest.raises(ValueError, match='record semantics'):
        publisher.verify_diagnostic(publication.api, publication.report, tmp_path / 'readback')


def test_historical_failed_run_cannot_be_relabelled_completed(publication, tmp_path):
    publication.old_run.summary['benchmark_status'] = 'completed'
    with pytest.raises(ValueError, match='completion state'):
        publisher.verify_sources(publication.api, publication.report, tmp_path / 'readback')
    publication.old_run.summary['benchmark_status'] = 'failed'
    publication.old_run.state = 'finished'
    assert publisher.verify_sources(publication.api, publication.report, tmp_path / 'second')['workers'][0]['remote_status'] == 'failed'


def test_terminal_failed_diagnostic_publishes_evidence_without_completion_or_training_claim(publication, tmp_path):
    publication.record.update(status='failed', computation_status='failed', error_type='KeyboardInterrupt')
    report_module.write_json(publication.root / 'result.json', publication.record)
    report_module.write_json(publication.root / 'artifact_metrics.json', publication.record)
    execution = report_module.read_json(publication.root / 'execution.json')
    execution.update(status='failed', result=publisher.pin(publication.root / 'result.json'))
    report_module.write_json(publication.root / 'execution.json', execution)
    report = report_module.build_report(publication.root)
    report_module.write_report(report, publication.directory)
    artifact = SavedArtifact(report_module.worker_files(publication.record, publication.root),
                             publication.record, 'failed-single-diagnostic')
    publication.new_run.artifacts = [artifact]
    publication.new_run.state = 'failed'
    publication.new_run.summary['diagnostic_status'] = 'failed'
    assert publisher.prepare(publication.directory)[0]['publication_gate'] == 'partial_evidence_only'
    proof = publisher.verify_sources(publication.api, report, tmp_path / 'readback')
    assert proof['workers'][1]['local_status'] == proof['workers'][1]['remote_status'] == 'failed'
    assert publisher.scalar_summary(report)['checkpoint_issue_resolved'] is False
    assert publisher.scalar_summary(report)['subsequent_training_authorized'] is False
    publication.new_run.state = 'running'
    with pytest.raises(ValueError, match='remote run must be finished'):
        publisher.verify_diagnostic(publication.api, report, tmp_path / 'running')


@pytest.mark.parametrize('change', ['duplicated-source', 'wrong-count', 'old-status', 'new-state', 'unmatched-file'])
def test_receipt_cannot_substitute_duplicate_or_unverified_sources(publication, tmp_path, change):
    proof = publisher.verify_sources(publication.api, publication.report, tmp_path / 'readback')
    if change == 'duplicated-source':
        proof['workers'][1]['wandb_url'] = proof['workers'][0]['wandb_url']
    elif change == 'wrong-count':
        proof['diagnostic_worker_count'] = True
    elif change == 'old-status':
        proof['workers'][0]['remote_status'] = 'completed'
    elif change == 'new-state':
        proof['workers'][1]['run_state'] = 'running'
    else:
        next(iter(proof['workers'][1]['files'].values()))['matches'] = False
    with pytest.raises(ValueError):
        publisher.validate_source_proofs(proof, publication.report)


def stage_pending(publication, output):
    """Save an already uploaded report; finalization must not create another run."""
    output.mkdir()
    proof = publisher.verify_sources(publication.api, publication.report, output / 'initial_readbacks')
    report_module.write_json(output / 'source_readback.json', proof)
    _, originals = publisher.prepare(publication.directory)
    files = {}
    for name, value in originals.items():
        target = output / 'report_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(value['file']).read_bytes())
        files[name] = publisher.pin(target)
    url = report_module.WAND_PREFIX + 'report-only'
    artifact = SavedArtifact(files, {}, 'report-only')
    publication.api.runs[run_path(url)] = SavedRun(publisher.scalar_summary(publication.report), artifact)
    publication.api.artifacts[artifact.qualified_name] = artifact
    report_module.write_json(output / 'publication_pending.json', {'wandb_url': url, 'artifact': artifact.qualified_name})
    return files


def test_slack_finalization_retries_same_snapshot_and_then_blocks_duplicate(publication, tmp_path, monkeypatch):
    output = tmp_path / 'pending'
    stage_pending(publication, output)
    slack = Mock(side_effect=['failed', 'sent'])
    monkeypatch.setattr(publisher, 'notify_slack', slack)
    assert publisher.finalize(publication.directory, output, publication.api) == 1
    receipt = report_module.read_json(output / 'publication.json')
    assert receipt['status'] == 'notification_failed'
    assert receipt['subsequent_training_authorized'] is False
    assert publisher.finalize(publication.directory, output, publication.api) == 0
    receipt = report_module.read_json(output / 'publication.json')
    assert receipt['status'] == 'completed' and receipt['slack_completion_status'] == 'sent'
    assert receipt['remote_readback']['artifact_state'] == 'COMMITTED'
    assert slack.call_count == 2
    with pytest.raises(ValueError, match='already finalized'):
        publisher.finalize(publication.directory, output, publication.api)
    assert slack.call_count == 2
    assert json.loads(Path(publication.report['historical_failed_source']['file']).read_text())['status'] == 'failed'
    assert '원인해결·구현통과판정 변경없음' in slack.call_args.args[0]


def test_notification_retry_rejects_changed_immutable_snapshot(publication, tmp_path, monkeypatch):
    output = tmp_path / 'pending'
    stage_pending(publication, output)
    slack = Mock(return_value='failed')
    monkeypatch.setattr(publisher, 'notify_slack', slack)
    assert publisher.finalize(publication.directory, output, publication.api) == 1
    (output / 'report_snapshot' / 'RESULTS.md').write_text('changed after upload')
    with pytest.raises(ValueError, match='snapshot changed'):
        publisher.finalize(publication.directory, output, publication.api)
    assert slack.call_count == 1
