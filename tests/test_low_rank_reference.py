"""Independent reference preparation and guards; no training or external writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import low_rank_protocol as protocol
from scripts import run_low_rank_reference as runner


SOURCE = Path('/tmp/cl-low-rank-official-20261008')


def test_reference_dry_run_is_stdlib_only_and_ignores_missing_inputs(tmp_path):
    command = '''
import sys
from scripts.run_low_rank_reference import main
assert main() == 0
assert not {'torch', 'numpy', 'timm', 'wandb', 'models', 'methods', 'utils'}.intersection(sys.modules)
'''
    result = subprocess.run([
        sys.executable, '-S', '-c', command, '--method', 'inf-lora',
        '--reference-source', str(tmp_path / 'absent-source'),
        '--checkpoint', str(tmp_path / 'absent-checkpoint'),
        '--initial-state', str(tmp_path / 'absent-initial'),
        '--credentials', str(tmp_path / 'absent-credentials'),
    ], cwd=runner.ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document['method'] == 'inf-lora'
    assert document['training_executed'] is False
    assert document['protocol'] == protocol.protocol('inf-lora')
    assert not list(tmp_path.iterdir())


def test_reference_dry_run_never_reads_credentials_or_prepares_models(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['run_low_rank_reference.py'])
    monkeypatch.setattr(runner, 'configure_environment', lambda **kwargs: pytest.fail('dry-run credentials'))
    monkeypatch.setattr(runner, 'source_code', lambda *args: pytest.fail('dry-run source model import'))
    monkeypatch.setattr(runner, 'execute', lambda *args: pytest.fail('dry-run worker launch'))
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)['training_executed'] is False


def test_reference_worker_refuses_missing_execution_guard_before_reading_spec(monkeypatch, tmp_path):
    monkeypatch.delenv('CL_LOW_RANK_REFERENCE_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit authorized --execute'):
        runner.worker(tmp_path / 'absent.json')


@pytest.mark.parametrize('device,wandb', [('0', 'online'), ('1', 'offline')])
def test_reference_worker_refuses_changed_device_or_logging_before_ml_imports(monkeypatch, tmp_path, device, wandb):
    monkeypatch.setenv('CL_LOW_RANK_REFERENCE_EXECUTE', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', device)
    monkeypatch.setenv('WANDB_MODE', wandb)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps({'method': 'ewc-lora', 'protocol': protocol.protocol('ewc-lora'), 'gpu': '1'}))
    monkeypatch.setattr(runner, 'verify_reference_source', lambda *args: pytest.fail('must reject environment first'))
    with pytest.raises(ValueError, match='physical GPU1 and online'):
        runner.worker(path)


def test_reference_worker_rejects_changed_execution_file_before_configuring_ml(monkeypatch, tmp_path):
    checkpoint, initial, implementation = [tmp_path / name for name in ('checkpoint.npz', 'initial.pt', 'source.py')]
    checkpoint.write_bytes(b'checkpoint')
    initial.write_bytes(b'initial')
    implementation.write_bytes(b'implementation drift')
    spec = {'method': 'ewc-lora', 'protocol': protocol.protocol('ewc-lora'), 'gpu': '1',
            'reference_source': str(SOURCE), 'checkpoint_file': str(checkpoint),
            'initial_state': str(initial), 'initial_state_sha256': protocol.file_hash(initial),
            'execution_source_sha256': {str(implementation): '0' * 64}}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    monkeypatch.setenv('CL_LOW_RANK_REFERENCE_EXECUTE', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setenv('WANDB_MODE', 'online')
    monkeypatch.setattr(runner, 'CHECKPOINT_SHA256', protocol.file_hash(checkpoint))
    monkeypatch.setattr(runner, 'verify_reference_source', lambda *args: {})
    monkeypatch.setattr(runner, 'configure_native_environment', lambda: pytest.fail('must reject source drift before ML'))
    with pytest.raises(ValueError, match='files changed after freezing'):
        runner.worker(path)


def test_reference_source_verification_detects_tensor_code_drift_even_with_clean_git_status(monkeypatch):
    plan = protocol.load_plan()
    pins = plan['reference_source_sha256']
    original = protocol.file_hash

    def altered_hash(path):
        return '0' * 64 if Path(path).name == 'vit_ewclora.py' else original(path)

    if not SOURCE.is_dir():
        pytest.skip('Pinned reference source is not installed locally')
    assert protocol.verify_reference_source(SOURCE) == pins
    monkeypatch.setattr(protocol, 'file_hash', altered_hash)
    with pytest.raises(ValueError, match='clean at the reviewed pinned revision'):
        protocol.verify_reference_source(SOURCE)


def fake_launch_spec(tmp_path):
    return {'method': 'ewc-lora', 'protocol': protocol.protocol('ewc-lora'),
            'output': str(tmp_path / 'run')}


def test_reference_missing_slack_credentials_rejects_before_output_or_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'run_spec', lambda *args: fake_launch_spec(tmp_path))
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.setattr(runner, 'valid_slack_webhook', lambda: False)
    monkeypatch.setattr(runner, 'notify_slack', lambda *args: pytest.fail('credential failure sent notification'))
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('credential failure launched worker'))
    with pytest.raises(ValueError, match='Slack credentials'):
        runner.execute(SimpleNamespace(), {'gpu': '1'})
    assert not (tmp_path / 'run').exists()


def test_reference_start_notification_failure_prevents_worker_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'run_spec', lambda *args: fake_launch_spec(tmp_path))
    monkeypatch.setenv('WANDB_API_KEY', 'test-only-placeholder')
    monkeypatch.setattr(runner, 'valid_slack_webhook', lambda: True)
    monkeypatch.setattr(runner, 'notify_slack', lambda message: 'failed:TimeoutError')
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('notification failure launched worker'))
    with pytest.raises(RuntimeError, match='start notification failed before training'):
        runner.execute(SimpleNamespace(), {'gpu': '1'})
    notifications = json.loads((tmp_path / 'run/notifications.json').read_text())
    assert notifications[0]['status'] == 'failed:TimeoutError'


def test_reference_completion_notification_failure_returns_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'run_spec', lambda *args: fake_launch_spec(tmp_path))
    monkeypatch.setenv('WANDB_API_KEY', 'test-only-placeholder')
    monkeypatch.setattr(runner, 'valid_slack_webhook', lambda: True)
    messages = iter(['sent', 'failed:TimeoutError'])
    monkeypatch.setattr(runner, 'notify_slack', lambda message: next(messages))

    class FinishedWorker:
        returncode = 0

        def poll(self):
            return 0

    def launch(*args, **kwargs):
        # No process is launched; represent a successfully uploaded worker.
        (tmp_path / 'run/summary.json').write_text(json.dumps({
            'status': 'completed', 'wandb_logging_status': 'uploaded',
            'wandb_url': 'https://wandb.ai/test-only'}))
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '1'
        assert kwargs['env']['CL_LOW_RANK_REFERENCE_EXECUTE'] == '1'
        assert kwargs['env']['WANDB_MODE'] == 'online'
        return FinishedWorker()

    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    assert runner.execute(SimpleNamespace(timeout=10), {'gpu': '1'}) == 1
    notifications = json.loads((tmp_path / 'run/notifications.json').read_text())
    assert [row['status'] for row in notifications] == ['sent', 'failed:TimeoutError']


def test_real_cpu_author_initial_catalog_matches_native_without_source_changes(tmp_path):
    checkpoint = Path(protocol.CHECKPOINT)
    if not SOURCE.is_dir() or not checkpoint.is_file():
        pytest.skip('Pinned source and shared Base checkpoint are not installed locally')
    before = protocol.verify_reference_source(SOURCE)
    destination = tmp_path / 'ewc-initial.pt'
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', WANDB_MODE='disabled',
                       JE_ARROW_MALLOC_CONF='background_thread:false')
    environment.pop('WANDB_API_KEY', None)
    environment.pop('SLACK_WEBHOOK_URL', None)
    result = subprocess.run([
        sys.executable, str(Path(runner.__file__)), '--method', 'ewc-lora',
        '--reference-source', str(SOURCE), '--checkpoint', str(checkpoint),
        '--prepare-initial-state', str(destination),
    ], cwd=runner.ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(destination.with_suffix('.json').read_text())
    assert metadata['training_executed'] is metadata['gpu_used'] is False
    assert metadata['image_forwards'] == metadata['optimizer_steps'] == 0
    assert metadata['canonical_pretrained_sha256'] == protocol.CANONICAL_PRETRAINED_SHA256
    assert protocol.verify_reference_source(SOURCE) == before
    import torch
    from models.low_rank_utils.factory import build_network

    catalog = torch.load(destination, map_location='cpu', weights_only=True)
    protocol.validate_initial_state(catalog, 'ewc-lora')
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            protocol.seed_phase(1993, 0, 'task_begin')
            native = build_network('ewc', runner.author_args(SOURCE, 'ewc-lora', torch.device('cpu')))
        native_state = protocol.learned_state(native)
        assert set(native_state) == set(catalog['learned_state'])
        for name, value in native_state.items():
            torch.testing.assert_close(value.cpu(), catalog['learned_state'][name], atol=0, rtol=0)
        assert protocol.tensor_digest(native_state) == metadata['initial_learned_sha256']
        assert protocol.canonical_pretrained_digest(native.image_encoder) == protocol.CANONICAL_PRETRAINED_SHA256
    finally:
        torch.set_num_threads(previous_threads)
