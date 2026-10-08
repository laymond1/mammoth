"""Publication gate tests with synthetic saved evidence and fake remote APIs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import publish_rainbow_validation as publisher
from tests.test_rainbow_validation_report import stage


def fixture_proofs():
    sources = [{'name': name, 'method': 'rainbow-prompt', 'mode': 'full_validation' if index < 2 else
                'short_validation' if index < 4 else 'efficiency', 'side': 'author' if index in (0, 2) else 'native',
                'pass': 'validation' if index < 4 else 'measurement' if index < 7 else 'flops',
                'wandb_url': publisher.report_module.WAND_PREFIX + name, 'sha256': name + '-hash'}
               for index, name in enumerate(publisher.report_module.JOB_NAMES)]
    workers = [{**source, 'source_sha256': source['sha256'], 'run_state': 'finished', 'artifact_state': 'COMMITTED',
                'record_semantics_verified': True, 'accuracy_summary_verified': True,
                'count_summary_verified': {'efficiency/tasks_completed': 1},
                'files': {'artifact_metrics.json': {'matches': True}}} for source in sources]
    return {'status': 'pass', 'worker_count': 8, 'workers': workers}, sources


def test_all_eight_workers_must_have_exact_readback_identity():
    proof, sources = fixture_proofs()
    publisher.validate_source_proofs(proof, sources)


@pytest.mark.parametrize('field,value', [('name', 'other'), ('mode', 'efficiency'), ('side', 'native'),
                                      ('source_sha256', 'wrong'), ('run_state', 'running'),
                                      ('artifact_state', 'PENDING'), ('record_semantics_verified', False),
                                      ('accuracy_summary_verified', False), ('count_summary_verified', {}),
                                      ('files', {'artifact_metrics.json': {'matches': False}})])
def test_incomplete_or_changed_source_readback_cannot_finalize(field, value):
    proof, sources = fixture_proofs()
    proof['workers'][0][field] = value
    with pytest.raises(ValueError):
        publisher.validate_source_proofs(proof, sources)


def test_repeated_url_or_boolean_count_cannot_satisfy_readback():
    proof, sources = fixture_proofs()
    proof['workers'][1]['wandb_url'] = proof['workers'][0]['wandb_url']
    with pytest.raises(ValueError):
        publisher.validate_source_proofs(proof, sources)
    proof, sources = fixture_proofs()
    proof['worker_count'] = True
    with pytest.raises(ValueError):
        publisher.validate_source_proofs(proof, sources)


def test_prepare_never_reads_credentials_or_uploads_weights(stage, tmp_path, monkeypatch):
    report = publisher.report_module.build_report(stage[0])
    directory = tmp_path / 'report'
    publisher.report_module.write_report(report, directory)
    for name in publisher.report_module.JOB_NAMES:
        (stage[0] / name / 'initial_state.pt').write_bytes(b'synthetic forbidden weight artifact')
        (stage[0] / name / 'final_network.pt').write_bytes(b'synthetic forbidden weight artifact')
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('dry preparation read credentials'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda message: pytest.fail('dry preparation sent Slack'))
    saved, files = publisher.prepare(directory)
    assert len(saved['sources']) == 8
    assert all(Path(name).suffix in ('.json', '.md', '.csv', '.npz') for name in files)
    assert not any('.pt' in name for name in files)
    assert publisher.main(['--report-dir', str(directory)]) == 0


def test_edited_markdown_is_not_publishable(stage, tmp_path):
    directory = tmp_path / 'report'
    publisher.report_module.write_report(publisher.report_module.build_report(stage[0]), directory)
    (directory / 'RESULTS.md').write_text('misleading full-paper reproduction claim')
    with pytest.raises(ValueError, match='Markdown'):
        publisher.prepare(directory)


@pytest.mark.parametrize('private', [{'api_key': 'private-value'}, {'SLACK_WEBHOOK_URL': 'private-value'},
                                   {'runtime': ['https://hooks.slack.com/services/private-value']}])
def test_private_fields_are_rejected_without_echoing_their_contents(stage, tmp_path, private):
    directory = tmp_path / 'report'
    publisher.report_module.write_report(publisher.report_module.build_report(stage[0]), directory)
    (stage[0] / 'measurement1/model_snapshots.json').write_text(json.dumps(private))
    with pytest.raises(ValueError, match='private credential') as error:
        publisher.prepare(directory)
    assert 'private-value' not in str(error.value)


def test_remote_record_semantics_are_compared_to_local_saved_evidence(stage, tmp_path, monkeypatch):
    saved = publisher.report_module.build_report(stage[0])
    source = saved['sources'][0]
    local = publisher.report_module.read_json(source['file'])
    remote = deepcopy(local)
    remote['protocol']['learned_gate_active'] = True
    remote_file = tmp_path / 'remote.json'
    remote_file.write_text(json.dumps(remote))
    artifact = SimpleNamespace(state='COMMITTED', qualified_name='laymond1/cl-efficiency/fixture:v0', digest='digest',
        manifest=SimpleNamespace(entries=publisher.worker_files(source)),
        get_path=lambda name: SimpleNamespace(download=lambda root: str(remote_file)))
    summary = {'benchmark_status': 'completed', 'benchmark_pass': source['pass'],
               'accuracy/mean_accuracy_pct': local['accuracy']['mean_accuracy_pct']}
    summary.update({'efficiency/' + key: local['metrics'][key] for key in
                    ('tasks_completed', 'outer_observe_steps', 'outer_training_stream_examples', 'evaluation_examples')})
    run = SimpleNamespace(state='finished', summary=summary, logged_artifacts=lambda: [artifact])
    monkeypatch.setattr(publisher, 'manifest_matches', lambda artifact, files: {name: {'matches': True} for name in files})
    with pytest.raises(ValueError, match='semantics'):
        publisher.verify_worker(SimpleNamespace(run=lambda path: run), source, tmp_path / 'readback')


def test_failed_slack_retry_rechecks_all_sources_and_report_before_send(stage, tmp_path, monkeypatch):
    report = publisher.report_module.build_report(stage[0])
    directory = tmp_path / 'report'
    publisher.report_module.write_report(report, directory)
    report, original = publisher.prepare(directory)
    output = tmp_path / 'publication'
    for name, proof in original.items():
        target = output / 'report_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(proof['file']).read_bytes())
    proofs, _ = fixture_proofs()
    proofs['workers'] = [{**proof, **source, 'source_sha256': source['sha256']}
                        for proof, source in zip(proofs['workers'], report['sources'])]
    publisher.write_json(output / 'source_readback.json', proofs)
    publisher.write_json(output / 'publication_pending.json', {'wandb_url': publisher.report_module.WAND_PREFIX + 'report',
                                                             'artifact': 'laymond1/cl-efficiency/report:v0'})
    checked, notified = [], []
    def verify(api, source, target):
        checked.append(source['name'])
        return proofs['workers'][publisher.report_module.JOB_NAMES.index(source['name'])]
    def notify(message):
        notified.append(message)
        assert len(checked) == len(notified) * 8
        return 'failed' if len(notified) == 1 else 'sent'
    monkeypatch.setattr(publisher, 'verify_worker', verify)
    monkeypatch.setattr(publisher, 'notify_slack', notify)
    monkeypatch.setattr(publisher, 'verify_report_publication', lambda *args: {'status': 'pass', 'run_state': 'finished', 'artifact_state': 'COMMITTED'})
    assert publisher.finalize(directory, output, object()) == 1
    assert publisher.finalize(directory, output, object()) == 0
    with pytest.raises(ValueError, match='Already finalized'):
        publisher.finalize(directory, output, object())
    assert len(checked) == 16 and len(notified) == 2
