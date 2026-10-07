"""Upload-only recovery using saved CPU fixtures and an in-memory W&B API."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import recover_low_rank_upload as recovery
from scripts import report_low_rank_results as reporter
from scripts.cl_efficiency import write_json
from scripts.low_rank_protocol import evaluate_acceptance
from tests.test_low_rank_report import saved_reports


@pytest.fixture
def failed_upload(saved_reports):
    manifest = reporter.read_json(saved_reports.inputs)
    path = Path(manifest['ewc-lora']['native'])
    directory = path.parent
    record = reporter.read_json(path)
    reference = Path(manifest['ewc-lora']['reference'])
    record.update(name='ewc-native-fixture', implementation='mammoth_native',
                  status='logging_failed', computation_status='completed', wandb_logging_status='failed:TypeError',
                  reference_summary=str(reference), reference_summary_sha256=reporter.file_hash(reference),
                  execution_source_sha256={'original_worker.py': 'd' * 64},
                  phases={'observe': {'wall_s': 120.}}, measurement_notes={'scope': 'unchanged original validation'},
                  sensor_errors=[], precision=record['protocol']['precision'])
    record['acceptance'] = evaluate_acceptance(record['task_results'], reporter.read_json(reference)['task_results'],
                                               structural_checks=record['structural_checks'])
    spec_keys = ('name', 'method', 'implementation', 'protocol', 'reference_summary', 'reference_summary_sha256',
                 'reference_commit', 'checkpoint_sha256', 'canonical_pretrained_sha256', 'initial_state_sha256',
                 'execution_source_sha256')
    write_json(directory / 'spec.json', {key: record[key] for key in spec_keys})
    for name in ('result.json', 'metrics.json'):
        write_json(directory / name, record)
    completion = {'status': 'logging_failed', 'exit_code': 1, 'timeout': False, 'acceptance_status': 'pass',
                  'wall_s': 125.5, 'wandb_url': record['wandb_url']}
    write_json(directory / 'completion.json', completion)
    for name, value in (('acceptance.json', record['acceptance']), ('epoch_results.json', record['epoch_results']),
                        ('task_results.json', record['task_results']), ('effective_args.json', {'n_epochs': 20}),
                        ('shared_data.json', {'dataset': 'cifar100'}), ('phases.json', [{'name': 'observe', 'wall_s': 120.}])):
        write_json(directory / name, value)
    (directory / 'telemetry.csv').write_text('seconds,power_w\n0,300\n')
    originals = {name: (directory / name).read_bytes() for name in recovery.ORIGINAL_FILES}
    return SimpleNamespace(directory=directory, record=record, completion=completion, originals=originals)


@pytest.fixture
def fake_recovery_sdk(failed_upload, monkeypatch):
    events, artifacts = [], {}
    path = reporter.run_path(failed_upload.record['wandb_url'])

    class Summary(dict):
        def update(self, values):
            events.append(('summary_dict', copy.deepcopy(values)))
            super().update(values)

    run = SimpleNamespace(id=path.rsplit('/', 1)[1], url=failed_upload.record['wandb_url'], state='failed',
                          summary=Summary(), logged_artifacts=lambda: list(artifacts.values()))

    class Artifact:
        def __init__(self, name, **kwargs):
            self.qualified_name = f'laymond1/cl-efficiency/{name}:v0'
            self.metadata = kwargs.get('metadata', {})
            self.state, self.digest = 'COMMITTED', 'fixture-artifact-digest'
            self.files = {}
            self.manifest = SimpleNamespace(entries={})

        def add_file(self, path, name):
            content = Path(path).read_bytes()
            self.files[name] = content
            self.manifest.entries[name] = SimpleNamespace(
                digest=base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode(),
                size=len(content), ref=None)

        def wait(self):
            events.append(('wait', self.qualified_name))

        def get_path(self, name):
            def download(root):
                destination = Path(root) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(self.files[name])
                return str(destination)
            return SimpleNamespace(download=download)

    class Api:
        def __init__(self, **kwargs):
            events.append(('api', kwargs))

        def run(self, requested):
            assert requested == path
            events.append(('read_run', requested))
            return run

        def artifact(self, name):
            events.append(('read_artifact', name))
            return artifacts[name]

        def flush(self):
            pass

    sdk = SimpleNamespace(Api=Api, Artifact=Artifact, run=None)

    def initialize(**kwargs):
        events.append(('resume', kwargs))
        run.state = 'running'
        sdk.run = run
        return run

    def upload(artifact):
        events.append(('upload', list(artifact.files)))
        artifacts[artifact.qualified_name] = artifact
        return artifact

    def finish(exit_code):
        events.append(('finish', exit_code))
        run.state = 'finished' if exit_code == 0 else 'failed'
        sdk.run = None

    run.log_artifact, sdk.init, sdk.finish = upload, initialize, finish
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(recovery, 'configure_environment', lambda **kw: {'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(recovery, 'notify_slack', lambda message: events.append(('slack', message)) or 'sent')
    return SimpleNamespace(events=events, sdk=sdk, run=run, artifacts=artifacts)


def arguments(fixture):
    return ['--result-dir', str(fixture.directory), '--execute']


def test_actual_format_dry_run_is_stdlib_only_read_only_and_no_credentials(failed_upload, monkeypatch):
    monkeypatch.setattr(recovery, 'configure_environment', lambda **kw: pytest.fail('Dry run read credentials'))
    monkeypatch.setattr(recovery, 'notify_slack', lambda *args: pytest.fail('Dry run sent Slack'))
    assert recovery.main(['--result-dir', str(failed_upload.directory)]) == 0
    result = subprocess.run([sys.executable, '-S', recovery.__file__, '--result-dir', str(failed_upload.directory)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['credentials_read'] is False
    assert not (failed_upload.directory / 'upload_recovery').exists()
    assert not (failed_upload.directory / 'state_audits.json').exists()
    for name, content in failed_upload.originals.items():
        assert (failed_upload.directory / name).read_bytes() == content


@pytest.mark.parametrize('mutation', ['status', 'computation', 'gate', 'acceptance', 'spec', 'reference',
                                    'metrics', 'epochs', 'prediction', 'checkpoint', 'completion', 'first_batch'])
def test_preflight_blocks_invalid_saved_evidence_before_auth(failed_upload, mutation, monkeypatch):
    directory, value = failed_upload.directory, copy.deepcopy(failed_upload.record)
    if mutation == 'status':
        value['status'] = 'failed'
    elif mutation == 'computation':
        value['computation_status'] = 'failed'
    elif mutation == 'gate':
        value['structural_checks']['checkpoint_round_trip'] = False
    elif mutation == 'acceptance':
        value['acceptance_status'] = 'review'
    elif mutation == 'spec':
        value['execution_source_sha256'] = {'new_worker.py': 'e' * 64}
    elif mutation == 'reference':
        Path(value['reference_summary']).write_text('{}')
    elif mutation == 'metrics':
        write_json(directory / 'metrics.json', {})
    elif mutation == 'epochs':
        value['epoch_results'][0]['samples'] = 4999
    elif mutation in ('prediction', 'checkpoint'):
        (directory / ('task1_predictions.npz' if mutation == 'prediction' else 'task1_state.pt')).write_bytes(b'changed')
    elif mutation == 'completion':
        completion = copy.deepcopy(failed_upload.completion)
        completion['timeout'] = True
        write_json(directory / 'completion.json', completion)
    elif mutation == 'first_batch':
        value['first_batch_audits'][0]['input_sha256'] = 'e' * 64
    write_json(directory / 'result.json', value)
    monkeypatch.setattr(recovery, 'configure_environment', lambda **kw: pytest.fail('Auth before saved evidence passed'))
    with pytest.raises((ValueError, RuntimeError)):
        recovery.main(arguments(failed_upload))
    assert not (directory / 'upload_recovery').exists()


def test_existing_run_resume_dict_summary_fourteen_files_readback_then_commit_and_slack(failed_upload, fake_recovery_sdk):
    assert recovery.main(arguments(failed_upload)) == 0
    events = fake_recovery_sdk.events
    resume = next(value for name, value in events if name == 'resume')
    assert resume['id'] == fake_recovery_sdk.run.id and resume['resume'] == 'must'
    artifact = next(iter(fake_recovery_sdk.artifacts.values()))
    assert set(artifact.files) == set(recovery.EVIDENCE_FILES) and len(artifact.files) == 14
    assert not any(name.endswith('.pt') for name in artifact.files)
    restored = reporter.read_json(failed_upload.directory / 'result.json')
    assert restored['status'] == 'completed' and restored['wandb_logging_status'] == 'uploaded'
    assert reporter.read_json(failed_upload.directory / 'metrics.json') == restored
    assert all(restored[key] == value for key, value in failed_upload.record.items() if key not in recovery.RECOVERABLE_FIELDS)
    completion = reporter.read_json(failed_upload.directory / 'completion.json')
    assert completion['status'] == 'completed' and completion['exit_code'] == 0
    assert completion['wall_s'] == failed_upload.completion['wall_s']
    metadata = restored['upload_recovery']
    for name, content in failed_upload.originals.items():
        assert Path(metadata['original_files'][name]['file']).read_bytes() == content
    proof = reporter.read_json(metadata['remote_upload_receipt'])
    assert proof['run_state'] == 'finished' and proof['artifact_state'] == 'COMMITTED'
    assert len(proof['all_evidence_files']) == 14 and proof['full_checkpoints_uploaded'] is False
    assert proof['count_summary_verified']['native_counts/observe_calls'] == 3160
    assert proof['count_summary_verified']['task2/old_samples'] == 1000
    assert fake_recovery_sdk.run.summary == recovery.summary_values(restored)
    assert artifact.metadata['local_checkpoint_sha256'] == proof['local_checkpoint_sha256']
    names = [name for name, _ in events]
    assert names.index('finish') < names.index('read_run') < names.index('slack')
    assert reporter.read_json(failed_upload.directory / 'state_audits.json') == failed_upload.record['state_audits']
    assert json.loads((failed_upload.directory / 'first_batch_audits.json').read_text()) == failed_upload.record['first_batch_audits']


@pytest.mark.parametrize('fault', ['id', 'summary', 'manifest', 'record', 'ambiguous', 'artifact'])
def test_remote_failures_preserve_original_failed_files_and_never_notify(failed_upload, fake_recovery_sdk, fault):
    sdk = fake_recovery_sdk.sdk
    if fault == 'id':
        fake_recovery_sdk.run.id = 'wrong-id'
    else:
        finish = sdk.finish
        def broken_finish(exit_code):
            finish(exit_code)
            if exit_code == 0:
                artifact = next(iter(fake_recovery_sdk.artifacts.values()))
                if fault == 'summary':
                    fake_recovery_sdk.run.summary['native_counts/optimizer_updates'] = 3159
                elif fault == 'manifest':
                    artifact.manifest.entries['task1_predictions.npz'].digest = 'incorrect'
                elif fault == 'record':
                    uploaded = json.loads(artifact.files['result.json'])
                    uploaded['status'] = 'failed'
                    artifact.files['result.json'] = json.dumps(uploaded).encode()
                    content = artifact.files['result.json']
                    artifact.manifest.entries['result.json'].digest = base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode()
                    artifact.manifest.entries['result.json'].size = len(content)
                elif fault == 'ambiguous':
                    clone = copy.deepcopy(artifact)
                    clone.qualified_name = 'laymond1/cl-efficiency/duplicate:v0'
                    fake_recovery_sdk.artifacts[clone.qualified_name] = clone
                elif fault == 'artifact':
                    artifact.state = 'PENDING'
        sdk.finish = broken_finish
    options = arguments(failed_upload) + ['--readback-timeout', '0.001']
    assert recovery.main(options) == 1
    for name, content in failed_upload.originals.items():
        assert (failed_upload.directory / name).read_bytes() == content
    assert not any(name == 'slack' for name, _ in fake_recovery_sdk.events)
    receipt = next((failed_upload.directory / 'upload_recovery').glob('*/result.json'))
    body = reporter.read_json(receipt)
    assert body['status'] == 'upload_recovery_failed' and 'error_type' in body
    assert 'error' not in body


def test_notification_failure_keeps_verified_computation_and_upload_completed(failed_upload, fake_recovery_sdk, monkeypatch):
    monkeypatch.setattr(recovery, 'notify_slack', lambda message: 'failed:OSError')
    assert recovery.main(arguments(failed_upload)) == 1
    record = reporter.read_json(failed_upload.directory / 'result.json')
    assert record['status'] == 'completed' and record['wandb_logging_status'] == 'uploaded'
    receipt = next((failed_upload.directory / 'upload_recovery').glob('*/result.json'))
    assert reporter.read_json(receipt)['status'] == 'notification_failed'
