"""Environment/connection preparation tests: all external services are mocked."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import check_cl_environment as checker


@pytest.fixture(autouse=True)
def isolate_checker_environment(monkeypatch):
    for name in ('CUDA_VISIBLE_DEVICES', 'WANDB_SILENT', 'WANDB_CONSOLE',
                 'WANDB_DISABLE_CODE', 'WANDB_DISABLE_GIT', 'WANDB_CONFIG_PATHS'):
        monkeypatch.setenv(name, os.environ.get(name, ''))


@pytest.fixture
def profile(tmp_path):
    return {'gpu': '1', 'data_root': str(tmp_path / 'data'), 'cache_root': str(tmp_path / 'cache'),
            'wandb_entity': 'public-user', 'wandb_project': 'cl-efficiency'}


def fake_sdk(profile, fail_at=None):
    run = SimpleNamespace(id='test123', log=Mock(), summary={}, finish=Mock())
    artifact = SimpleNamespace(add_file=Mock(), add_dir=Mock(), wait=Mock())
    run.log_artifact = Mock(return_value=artifact)
    remote = SimpleNamespace(summary=dict(checker.CHECK_METRICS))
    api = SimpleNamespace(run=Mock(return_value=remote))
    sdk = SimpleNamespace(Settings=Mock(return_value='private-settings'), init=Mock(return_value=run),
                          Artifact=Mock(return_value=artifact), Api=Mock(return_value=api))
    if fail_at == 'init':
        sdk.init.side_effect = RuntimeError('secret API key and https://hooks.slack.com/services/private/token')
    elif fail_at == 'artifact':
        artifact.wait.side_effect = OSError('secret API key and private webhook')
    elif fail_at == 'verify':
        remote.summary = {}
    return sdk, run, artifact, api


def basic_report(profile, status='passed'):
    return {'profile': dict(profile), 'environment': {'status': status},
            'training_executed': False, 'inference_executed': False, 'experiment_executed': False,
            'wandb': {'status': 'not_requested'}, 'slack': {'status': 'not_requested'}}


def test_default_never_tests_external_connections(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(checker, 'configure_environment', lambda: {**profile, 'SLACK_WEBHOOK_URL': 'secret'})

    def inspect(effective):
        assert effective == profile
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
        return {'status': 'passed'}

    monkeypatch.setattr(checker, 'inspect_environment', inspect)
    wandb = Mock(side_effect=AssertionError('No online runs without a flag'))
    slack = Mock(side_effect=AssertionError('No posts without a flag'))
    monkeypatch.setattr(checker, 'check_wandb', wandb)
    monkeypatch.setattr(checker, 'notify_slack', slack)
    assert checker.main(['--output', str(tmp_path / 'out')]) == 0
    report = json.loads((tmp_path / 'out/report.json').read_text())
    assert report['profile'] == profile
    assert report['wandb']['status'] == report['slack']['status'] == 'not_requested'
    assert report['training_executed'] is report['inference_executed'] is False
    wandb.assert_not_called()
    slack.assert_not_called()


def test_wandb_artifact_contains_only_synthetic_connection_markers(tmp_path, monkeypatch, profile):
    sdk, run, artifact, api = fake_sdk(profile)
    monkeypatch.setattr(checker.importlib, 'import_module', lambda name: sdk)
    # Private files coexist with the report but must never be added to an artifact.
    (tmp_path / 'wandb.conf').write_text('private webhook/API key')
    monkeypatch.setenv('WANDB_CONFIG_PATHS', str(tmp_path / 'wandb.conf'))
    report = basic_report(profile)
    result = checker.check_wandb(profile, report, tmp_path)
    assert result['status'] == 'passed'
    assert result['server_summary_verified'] is result['artifact_uploaded'] is True
    artifact.add_file.assert_called_once_with(str(tmp_path / 'connectivity_report.json'),
                                              name='connectivity_report.json')
    artifact.add_dir.assert_not_called()
    artifact.wait.assert_called_once_with(timeout=30)
    run.finish.assert_called_once_with(exit_code=0)
    api.run.assert_called_once_with('public-user/cl-efficiency/test123')
    run.log.assert_called_once_with(checker.CHECK_METRICS)
    config = sdk.init.call_args.kwargs['config']
    assert config == checker.CHECK_METRICS
    assert sdk.init.call_args.kwargs['job_type'] == 'connectivity-test'
    assert sdk.init.call_args.kwargs['mode'] == 'online'
    settings = sdk.Settings.call_args.kwargs
    assert settings['disable_code'] is settings['disable_git'] is settings['silent'] is True
    assert settings['disable_job_creation'] is settings['x_disable_stats'] is settings['x_disable_meta'] is True
    assert settings['finish_timeout'] == 30
    assert settings['finish_timeout_raises'] is True
    assert settings['console'] == 'off'
    assert 'WANDB_CONFIG_PATHS' not in os.environ
    assert settings['config_paths'] is None
    public_file = (tmp_path / 'connectivity_report.json').read_text()
    assert 'private webhook' not in public_file
    assert json.loads(public_file) == {'schema_version': 1, **checker.CHECK_METRICS}
    for forbidden in ('profile', 'data_root', 'cache_root', 'packages', 'gpu', 'run_url', str(tmp_path)):
        assert forbidden not in public_file
    assert result['run_url'] == 'https://wandb.ai/public-user/cl-efficiency/runs/test123'


def test_sdk_diagnostics_and_exception_text_are_discarded(tmp_path, monkeypatch, profile, capsys):
    sdk, run, artifact, api = fake_sdk(profile)

    def fail(**kwargs):
        print('private credential')
        raise RuntimeError('private credential')

    sdk.init.side_effect = fail
    monkeypatch.setattr(checker.importlib, 'import_module', lambda name: sdk)
    result = checker.check_wandb(profile, basic_report(profile), tmp_path)
    assert result['error_type'] == 'RuntimeError'
    assert 'private credential' not in json.dumps(result)
    assert capsys.readouterr() == ('', '')


def test_server_verification_occurs_after_artifact_and_finish(tmp_path, monkeypatch, profile):
    sdk, run, artifact, api = fake_sdk(profile)
    events = []
    artifact.wait.side_effect = lambda **kwargs: events.append('artifact_uploaded')
    run.finish.side_effect = lambda **kwargs: events.append('run_finished')

    def verify(path):
        assert events == ['artifact_uploaded', 'run_finished']
        events.append('server_read')
        return SimpleNamespace(summary=dict(checker.CHECK_METRICS))

    api.run.side_effect = verify
    monkeypatch.setattr(checker.importlib, 'import_module', lambda name: sdk)
    assert checker.check_wandb(profile, basic_report(profile), tmp_path)['status'] == 'passed'
    assert events == ['artifact_uploaded', 'run_finished', 'server_read']


@pytest.mark.parametrize('failure', ['init', 'artifact', 'verify'])
def test_wandb_failures_never_expose_exception_contents(tmp_path, monkeypatch, profile, capsys, failure):
    sdk, run, artifact, api = fake_sdk(profile, fail_at=failure)
    monkeypatch.setattr(checker.importlib, 'import_module', lambda name: sdk)
    result = checker.check_wandb(profile, basic_report(profile), tmp_path)
    serialized = json.dumps(result)
    assert result['status'] == 'failed'
    assert result['error_type'] in {'RuntimeError', 'OSError', 'ValueError'}
    assert 'secret' not in serialized
    assert 'hooks.slack.com' not in serialized
    assert capsys.readouterr() == ('', '')
    if failure == 'artifact':
        run.finish.assert_called_once_with(exit_code=1)
        api.run.assert_not_called()
    if failure == 'verify':
        run.finish.assert_called_once_with(exit_code=0)
        assert result['server_summary_verified'] is False


def test_failed_gpu_still_checks_both_requested_connections(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(checker, 'configure_environment', lambda: profile)
    monkeypatch.setattr(checker, 'inspect_environment', lambda _: {'status': 'failed', 'gpu': {'status': 'failed'}})
    sdk, run, artifact, api = fake_sdk(profile)
    monkeypatch.setattr(checker.importlib, 'import_module', lambda name: sdk)
    slack = Mock(return_value='sent')
    monkeypatch.setattr(checker, 'notify_slack', slack)
    assert checker.main(['--output', str(tmp_path), '--check-wandb', '--check-slack']) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['environment']['status'] == 'failed'
    assert report['wandb']['status'] == report['slack']['status'] == 'passed'
    slack.assert_called_once()
    message = slack.call_args.args[0]
    assert 'W&B 검사=passed' in message
    assert report['wandb']['run_url'] not in message
    assert 'https://' not in message
    assert str(tmp_path) not in message
    assert '환경=failed' not in message
    assert report['training_executed'] is report['inference_executed'] is False


def test_wandb_failure_does_not_skip_slack(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(checker, 'configure_environment', lambda: profile)
    monkeypatch.setattr(checker, 'inspect_environment', lambda _: {'status': 'passed'})
    sdk, run, artifact, api = fake_sdk(profile, fail_at='init')
    monkeypatch.setattr(checker.importlib, 'import_module', lambda name: sdk)
    slack = Mock(return_value='sent')
    monkeypatch.setattr(checker, 'notify_slack', slack)
    assert checker.main(['--output', str(tmp_path), '--check-wandb', '--check-slack']) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['wandb']['status'] == 'failed'
    assert report['slack']['status'] == 'passed'
    slack.assert_called_once()


def test_slack_flag_alone_sends_exactly_one_message(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(checker, 'configure_environment', lambda: profile)
    monkeypatch.setattr(checker, 'inspect_environment', lambda _: {'status': 'passed'})
    monkeypatch.setattr(checker, 'check_wandb', Mock(side_effect=AssertionError('W&B not requested')))
    slack = Mock(return_value='sent')
    monkeypatch.setattr(checker, 'notify_slack', slack)
    assert checker.main(['--output', str(tmp_path), '--check-slack']) == 0
    slack.assert_called_once()
    assert '학습·추론 실행 없음' in slack.call_args.args[0]


def test_slack_exception_uses_only_error_type(monkeypatch, profile):
    monkeypatch.setattr(checker, 'notify_slack', Mock(side_effect=OSError('private webhook URL')))
    assert checker.check_slack(basic_report(profile)) == {'status': 'failed', 'error_type': 'OSError'}


def test_cuda_inspection_does_not_construct_tensors_or_models(monkeypatch, profile):
    forbidden = Mock(side_effect=AssertionError('Tensor/model computation prohibited'))
    cuda = SimpleNamespace(is_available=Mock(return_value=True), init=Mock(),
                           get_device_properties=Mock(return_value=SimpleNamespace(
                               name='Mock GPU', total_memory=1024, major=8, minor=6, multi_processor_count=10)))
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='12.8'), tensor=forbidden, empty=forbidden,
                            nn=SimpleNamespace(Module=forbidden))
    monkeypatch.setattr(checker.importlib, 'import_module', lambda name: torch if name == 'torch' else SimpleNamespace())
    monkeypatch.setattr(checker.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(checker, 'query_gpu', Mock(return_value={'gpu_name': 'Mock GPU'}))
    monkeypatch.setattr(Path, 'exists', lambda self: True)
    monkeypatch.setattr(Path, 'is_dir', lambda self: True)
    monkeypatch.setattr(checker.os, 'access', lambda *args: True)
    report = checker.inspect_environment(profile)
    assert report['status'] == 'passed'
    assert report['gpu']['cuda']['logical_device'] == 0
    cuda.init.assert_called_once()
    cuda.get_device_properties.assert_called_once_with(0)
    forbidden.assert_not_called()
