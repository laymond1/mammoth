"""Report publication and verified Slack gates using saved CPU fixtures/stubs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import publish_hide_report as publisher
from scripts.cl_efficiency import write_json
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS
from scripts.hide_protocol import protocol
from scripts.report_hide_results import build_report


@pytest.fixture
def reports(tmp_path):
    reference = {'status': 'completed', 'mode': 'validation', 'protocol': protocol('validation'),
        'checkpoint_sha256': 'a' * 64, 'initial_state_sha256': 'b' * 64,
        'initial_learned_sha256': {'tii': 'c' * 64, 'prompt': 'd' * 64},
        'initial_canonical_pretrained_sha256': {'tii': 'e' * 64, 'prompt': 'e' * 64},
        'task_results': [
            {'task': 1, 'seen_correct': 990, 'seen_samples': 1000, 'old_correct': 0,
             'old_samples': 0, 'new_correct': 990, 'new_samples': 1000},
            {'task': 2, 'seen_correct': 1940, 'seen_samples': 2000, 'old_correct': 965,
             'old_samples': 1000, 'new_correct': 975, 'new_samples': 1000}],
        'first_batch_audits': [{'task': task, 'phase': phase, 'samples': 64, 'seed': 1993,
                               'input_sha256': 'f' * 64, 'labels_sha256': 'a' * 64,
                               'loss': 1., 'clipped_gradient_sha256': 'b' * 64}
                              for task in (1, 2) for phase in ('tii_train', 'prompt_train')]}
    native = copy.deepcopy(reference)
    native.update(acceptance_status='pass', structural_checks=dict.fromkeys(REQUIRED_STRUCTURAL_CHECKS, True))
    reference_file, native_file = tmp_path / 'reference.json', tmp_path / 'native.json'
    write_json(reference_file, reference)
    write_json(native_file, native)
    directory = tmp_path / 'report'
    directory.mkdir()
    report = build_report(reference, native)
    report['sources'] = {name: {'file': str(path), 'sha256': publisher.file_hash(path)}
                         for name, path in (('reference', reference_file), ('native', native_file))}
    write_json(directory / 'results.json', report)
    for filename in publisher.REPORT_FILES[1:]:
        (directory / filename).write_bytes(b'Explicit CPU test artifact; no experiment result')
    return directory, tmp_path / 'publication'


@pytest.fixture
def fake_sdk(monkeypatch):
    events = []
    run = SimpleNamespace(url='https://wandb.ai/laymond1/cl-efficiency/runs/reportfixture', summary={})

    class Artifact:
        def __init__(self, name, **kwargs):
            self.state = 'COMMITTED'
            self.qualified_name = 'laymond1/cl-efficiency/' + name + ':v0'
            self.files = {}

        def add_file(self, path, name):
            self.files[name] = Path(path).read_bytes()

        def wait(self):
            events.append(('artifact_wait', self.state))

    def log_artifact(artifact):
        events.append(('artifact', artifact))
        return artifact

    run.log_artifact = log_artifact
    sdk = SimpleNamespace(run=run, Artifact=Artifact)

    def initialize(**kwargs):
        events.append(('init', kwargs))
        return run

    def finish(**kwargs):
        events.append(('finish', kwargs))
        sdk.run = None

    sdk.init, sdk.finish = initialize, finish
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: {
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('No completion before readback'))
    return sdk, events, run


def arguments(reports):
    directory, output = reports
    return ['--report-dir', str(directory), '--output', str(output)]


def test_dry_run_is_read_only_and_has_no_credentials_network_or_notifications(reports, monkeypatch, capsys):
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Credential read in preparation'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Network send in preparation'))
    assert publisher.main(arguments(reports)) == 0
    assert not reports[1].exists()
    assert 'execution=False' in capsys.readouterr().out


def test_preparation_works_with_stdlib_only(reports):
    result = subprocess.run([sys.executable, '-S', str(Path(publisher.__file__).resolve()), *arguments(reports)],
                            capture_output=True, text=True, check=True)
    assert 'execution=False' in result.stdout
    assert not reports[1].exists()


@pytest.mark.parametrize('filename', publisher.REPORT_FILES)
def test_missing_report_artifact_blocks_authentication(reports, filename, monkeypatch):
    (reports[0] / filename).unlink()
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before local validation'))
    with pytest.raises(ValueError, match='Missing or empty'):
        publisher.main([*arguments(reports), '--execute'])
    assert not reports[1].exists()


def test_changed_provenance_blocks_publication(reports, monkeypatch):
    report = publisher.read_json(reports[0] / 'results.json')
    Path(report['sources']['reference']['file']).write_text('{}')
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before provenance validation'))
    with pytest.raises(ValueError, match='provenance'):
        publisher.main([*arguments(reports), '--execute'])
    assert not reports[1].exists()


def test_forged_accuracy_row_cannot_override_the_pinned_input_counts(reports):
    report = publisher.read_json(reports[0] / 'results.json')
    report['validation']['accuracy_metrics'][0]['native_accuracy_pct'] = 100.
    write_json(reports[0] / 'results.json', report)
    with pytest.raises(ValueError, match='disagree with pinned inputs'):
        publisher.prepare(reports[0])


def test_completed_efficiency_is_rejected_without_passed_validation(reports):
    report = publisher.read_json(reports[0] / 'results.json')
    report['efficiency']['status'] = 'completed'
    report['validation']['report_gate'] = 'blocked'
    write_json(reports[0] / 'results.json', report)
    with pytest.raises(ValueError, match='passing accuracy'):
        publisher.prepare(reports[0])


def test_publishing_uploads_immutable_snapshots_and_finishes_before_success(reports, fake_sdk):
    _, events, run = fake_sdk
    assert publisher.main([*arguments(reports), '--execute']) == 0
    output = reports[1]
    record = publisher.read_json(output / 'result.json')
    assert record['status'] == 'completed'
    assert record['slack_completion_status'] == 'deferred_until_readback'
    assert record['new_training'] is False and record['new_gpu_computation'] is False
    pins = publisher.read_json(output / 'artifact_files.json')
    assert set(pins) == {*publisher.REPORT_FILES, 'artifact_metadata.json'}
    artifact = next(value for name, value in events if name == 'artifact')
    assert set(artifact.files) == {*pins, 'artifact_files.json'}
    for filename, pin in pins.items():
        assert publisher.file_hash(output / pin['local_file']) == pin['sha256']
        assert artifact.files[filename] == (output / pin['local_file']).read_bytes()
    (reports[0] / 'report.md').write_text('Later human edit')
    assert (output / 'snapshots/report.md').read_bytes() == artifact.files['report.md']
    assert events[-1] == ('finish', {'exit_code': 0})
    assert run.summary['report_only'] is True


def test_failed_artifact_upload_has_no_success_or_secret_in_receipt(reports, fake_sdk, capsys):
    _, _, run = fake_sdk

    def failed(artifact):
        raise RuntimeError('Private credential marker must not be serialized')

    run.log_artifact = failed
    assert publisher.main([*arguments(reports), '--execute']) == 1
    record = publisher.read_json(reports[1] / 'result.json')
    assert record['status'] == 'logging_failed'
    assert record['error_type'] == 'RuntimeError'
    assert 'Private credential marker' not in (reports[1] / 'result.json').read_text()
    assert 'Private credential marker' not in capsys.readouterr().out


def readback_proof(output):
    record = publisher.read_json(output / 'result.json')
    pins = publisher.read_json(output / 'artifact_files.json')
    return {'status': 'pass', 'wandb_url': record['wandb_url'], 'run_state': 'finished',
            'artifact': record['wandb_artifact']['name'], 'artifact_state': 'COMMITTED',
            'result_sha256': publisher.file_hash(output / 'result.json'),
            'artifact_files': {name: {'sha256': pin['sha256'], 'matches_uploaded_pin': True,
                                     'local_snapshot_matches': True} for name, pin in pins.items()}}


def test_slack_completion_requires_verified_artifact_and_is_idempotent(reports, fake_sdk, monkeypatch):
    publisher.main([*arguments(reports), '--execute'])
    output = reports[1]
    proof_path = output / 'readback.json'
    write_json(proof_path, readback_proof(output))
    sends = []
    monkeypatch.setattr(publisher, 'notify_slack', lambda message: sends.append(message) or 'sent')
    cli = [*arguments(reports), '--execute', '--notify-after-readback', str(proof_path)]
    assert publisher.main(cli) == 0
    assert publisher.main(cli) == 0
    assert len(sends) == 1 and '결과 저장·검증 완료' in sends[0]
    assert '태스크2 정확도 97.00%' in sends[0]
    assert '최대 저자−구현 차이 0.00%p' in sends[0]
    assert publisher.read_json(output / 'notification.json')['status'] == 'sent'


def test_korean_completion_includes_actual_summary_means_and_worst_gap():
    record = {'validation_acceptance': 'pass', 'implementation_gate': 'pass',
              'efficiency_status': 'completed',
              'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/fixture',
              'summary': {'validation/task2_seen/native_accuracy_pct': 98.25,
                          'validation/task1_seen/reference_minus_native_pp': -.2,
                          'validation/task2_seen/reference_minus_native_pp': .05,
                          'validation/task2_old/reference_minus_native_pp': .1,
                          'validation/task2_new/reference_minus_native_pp': 0.,
                          'efficiency/hide-prompt/process_wall_s/mean': 70.25,
                          'efficiency/hide-prompt/gpu_board_energy_wh/mean': 4.25,
                          'efficiency/hide-prompt/cuda_peak_allocated_mib/mean': 6144.}}
    message = publisher.completion_message(record)
    assert '태스크2 정확도 98.25%' in message
    assert '최대 저자−구현 차이 0.10%p' in message
    assert '효율성 3회 평균: 70.2초, GPU 보드 4.250Wh, CUDA peak 6.00GiB' in message


@pytest.mark.parametrize('mutation', ['not_finished', 'uncommitted', 'wrong_result_hash', 'missing_file', 'failed_file'])
def test_incomplete_readback_blocks_credentials_and_slack(reports, fake_sdk, monkeypatch, mutation):
    publisher.main([*arguments(reports), '--execute'])
    proof = readback_proof(reports[1])
    if mutation == 'not_finished':
        proof['run_state'] = 'running'
    elif mutation == 'uncommitted':
        proof['artifact_state'] = 'PENDING'
    elif mutation == 'wrong_result_hash':
        proof['result_sha256'] = '0' * 64
    elif mutation == 'missing_file':
        del proof['artifact_files']['results.csv']
    else:
        proof['artifact_files']['report.md']['local_snapshot_matches'] = False
    proof_path = reports[1] / 'readback.json'
    write_json(proof_path, proof)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Credentials before readback gate'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Slack before readback gate'))
    with pytest.raises(ValueError):
        publisher.main([*arguments(reports), '--execute', '--notify-after-readback', str(proof_path)])
    assert not (reports[1] / 'notification.json').exists()
