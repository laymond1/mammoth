"""Reject incomplete remote evidence before finalizing recovery publication."""
from copy import deepcopy

import pytest

from scripts.publish_sd_recovery import validate_source_proofs


def fixture_proofs():
    observations = [{'wandb_url': f'https://wandb.ai/laymond1/cl-efficiency/runs/{index}'} for index in range(22)]
    workers = [{'wandb_url': item['wandb_url'], 'run_state': 'finished', 'artifact_state': 'COMMITTED',
                'record_semantics_verified': True, 'role': 'accepted_cost' if index < 20 else 'validation' if index == 20 else 'preserved_failure'}
               for index, item in enumerate(observations)]
    return {'status': 'pass', 'worker_count': 22, 'workers': workers}, observations


def test_complete_readback():
    proof, observations = fixture_proofs()
    validate_source_proofs(proof, observations)


@pytest.mark.parametrize('field,value', [('run_state', 'running'), ('artifact_state', 'PENDING'),
                                       ('record_semantics_verified', False), ('role', 'accepted_cost'),
                                       ('wandb_url', 'https://wandb.ai/laymond1/cl-efficiency/runs/another')])
def test_changed_source_proof(field, value):
    proof, observations = fixture_proofs()
    altered = deepcopy(proof)
    altered['workers'][20][field] = value
    with pytest.raises(ValueError):
        validate_source_proofs(altered, observations)


def test_count_alone_cannot_finalize():
    proof, observations = fixture_proofs()
    proof['workers'].pop()
    with pytest.raises(ValueError):
        validate_source_proofs(proof, observations)


def test_failed_slack_can_retry_after_remote_reverification(tmp_path, monkeypatch):
    import json
    from scripts import publish_sd_recovery as publisher
    output, directory, run_root = tmp_path / 'publication', tmp_path / 'report', tmp_path / 'runs'
    (output / 'report_snapshot').mkdir(parents=True)
    directory.mkdir()
    (run_root / 'validation').mkdir(parents=True)
    (output / 'report_snapshot/results.json').write_text('{}')
    original = {'results.json': publisher.pin(output / 'report_snapshot/results.json')}
    report = {'execution': {'file': str(run_root / 'execution.json')},
              'methods': {'sd-lora': {'accuracy_pct': {'mean': 99.}}}}
    (run_root / 'validation/acceptance.json').write_text(json.dumps({'accuracy': {'native_pct': 99., 'reference_pct': 99.}}))
    (output / 'source_readback.json').write_text('{}')
    (output / 'publication_pending.json').write_text(json.dumps({'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/report',
                                                               'artifact': 'laymond1/cl-efficiency/report:v0'}))
    monkeypatch.setattr(publisher, 'prepare', lambda path: (report, original))
    monkeypatch.setattr(publisher, 'publication_sources', lambda value: [])
    monkeypatch.setattr(publisher, 'validate_source_proofs', lambda *args: None)
    monkeypatch.setattr(publisher, 'scalar_summary', lambda value: {})
    verified, notified = [], []

    def verify(*args):
        verified.append(True)
        return {'status': 'pass', 'run_state': 'finished', 'artifact_state': 'COMMITTED'}

    def notify(message):
        notified.append(message)
        return 'failed' if len(notified) == 1 else 'sent'

    monkeypatch.setattr(publisher, 'verify_report_publication', verify)
    monkeypatch.setattr(publisher, 'notify_slack', notify)
    assert publisher.finalize(directory, output, object()) == 1
    assert publisher.finalize(directory, output, object()) == 0
    assert len(verified) == len(notified) == 2
    with pytest.raises(ValueError, match='Already finalized'):
        publisher.finalize(directory, output, object())
    assert len(notified) == 2
