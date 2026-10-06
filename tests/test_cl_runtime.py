"""Shared runtime tests using synthetic credentials and no external operations."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts import cl_runtime


@pytest.fixture
def public_profile(tmp_path):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps({
        'gpu': '1', 'wandb_entity': 'profile-team', 'wandb_project': 'profile-project',
        'data_root': '/shared/data', 'cache_root': '/shared/cache',
    }))
    return path


@pytest.fixture
def credentials(tmp_path):
    path = tmp_path / 'private.conf'
    path.write_text(json.dumps({
        'entity': 'credential-team', 'project': 'credential-project',
        'api_key': 'synthetic-api-key', 'slack_webhook_url': 'https://example.invalid/synthetic-hook',
        'HOME': '/must-not-apply', 'unrelated_secret': 'must-not-return',
    }))
    return path


def test_shared_paths_and_private_whitelist(public_profile, credentials):
    env = {'HOME': '/existing/home', 'CODEX_HOME': '/existing/codex', 'CUDA_VISIBLE_DEVICES': '7'}
    profile = cl_runtime.configure_environment(public_profile, credentials, env)
    assert profile == {'gpu': '1', 'wandb_entity': 'profile-team', 'wandb_project': 'profile-project',
                       'data_root': '/shared/data', 'cache_root': '/shared/cache'}
    assert env['CL_GPU'] == '1' and env['CL_DATA_ROOT'] == '/shared/data' and env['CL_CACHE_ROOT'] == '/shared/cache'
    assert env['TORCH_HOME'] == '/shared/cache/torch'
    assert env['XDG_CACHE_HOME'] == '/shared/cache'
    assert env['HF_HOME'] == '/shared/cache/huggingface'
    assert env['HF_HUB_CACHE'] == '/shared/cache/huggingface/hub'
    assert env['HF_DATASETS_CACHE'] == '/shared/cache/huggingface/datasets'
    assert env['WANDB_API_KEY'] == 'synthetic-api-key'
    assert env['SLACK_WEBHOOK_URL'] == 'https://example.invalid/synthetic-hook'
    assert env['HOME'] == '/existing/home' and env['CODEX_HOME'] == '/existing/codex'
    assert env['CUDA_VISIBLE_DEVICES'] == '7'
    assert 'unrelated_secret' not in env and 'HOME' not in profile
    assert 'synthetic-api-key' not in json.dumps(profile)
    assert 'synthetic-hook' not in json.dumps(profile)
    assert env['JE_ARROW_MALLOC_CONF'] == 'background_thread:false'


@pytest.mark.parametrize('existing', [None, '', 'dirty_decay_ms:0'])
def test_native_environment_disables_only_unconfigured_arrow_thread(existing):
    env = {'HOME': '/existing/home', 'CUDA_VISIBLE_DEVICES': '7', 'MALLOC_CONF': 'unrelated-setting'}
    if existing is not None:
        env['JE_ARROW_MALLOC_CONF'] = existing
    result = cl_runtime.configure_native_environment(env)
    expected = 'dirty_decay_ms:0,background_thread:false' if existing else 'background_thread:false'
    assert result == {'JE_ARROW_MALLOC_CONF': expected}
    assert env['JE_ARROW_MALLOC_CONF'] == expected
    assert env['HOME'] == '/existing/home' and env['CUDA_VISIBLE_DEVICES'] == '7'
    assert env['MALLOC_CONF'] == 'unrelated-setting'


@pytest.mark.parametrize('existing', ['background_thread:false,dirty_decay_ms:0', 'background_thread:true'])
def test_native_environment_preserves_explicit_allocator_choice(existing):
    env = {'JE_ARROW_MALLOC_CONF': existing}
    assert cl_runtime.configure_native_environment(env) == env == {'JE_ARROW_MALLOC_CONF': existing}


def test_native_import_regression_in_fresh_process():
    # No model creation, image inference, downloads, or CUDA initialization.
    # This ordering reproduced exit139 in Arrow's jemalloc_bg_thd before the fix.
    env = os.environ.copy()
    env.pop('JE_ARROW_MALLOC_CONF', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = ('from scripts.cl_runtime import configure_native_environment; '
               'configure_native_environment(); import timm, kornia; '
               'from transformers import CLIPTextModel; import pyarrow; '
               'print("native imports completed")')
    result = subprocess.run([sys.executable, '-c', command], cwd=cl_runtime.ROOT,
                            env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0
    assert 'native imports completed' in result.stdout


def test_existing_environment_overrides_profile_and_secrets(public_profile, credentials):
    env = {'CL_GPU': 'cpu', 'CL_DATA_ROOT': '/override/data', 'CL_CACHE_ROOT': '/override/cache',
           'WANDB_ENTITY': 'environment-team', 'WANDB_PROJECT': 'environment-project',
           'WANDB_API_KEY': 'existing-api-key', 'SLACK_WEBHOOK_URL': 'existing-webhook',
           'TORCH_HOME': '/stale/cache', 'HF_HUB_CACHE': '/stale/hub'}
    profile = cl_runtime.configure_environment(public_profile, credentials, env)
    assert profile['gpu'] == 'cpu' and profile['wandb_entity'] == 'environment-team'
    assert profile['wandb_project'] == 'environment-project'
    assert env['TORCH_HOME'] == '/override/cache/torch'
    assert env['HF_HUB_CACHE'] == '/override/cache/huggingface/hub'
    assert env['WANDB_API_KEY'] == 'existing-api-key' and env['SLACK_WEBHOOK_URL'] == 'existing-webhook'


def test_credentials_public_aliases_fill_missing_profile_values(tmp_path, credentials):
    profile_path = tmp_path / 'partial.json'
    profile_path.write_text('{}')
    profile = cl_runtime.configure_environment(profile_path, credentials, {})
    assert profile['wandb_entity'] == 'credential-team'
    assert profile['wandb_project'] == 'credential-project'


@pytest.mark.parametrize('content', [
    '{\nwandb_entity: fixture-team\nAPI_KEY: fixture-key\nSLACK_WEBHOOK_URL: https://example.invalid/fixture\n}',
    'ENTITY=fixture-team\nwandb_api_key="fixture-key"\nslack_webhook_url=\'https://example.invalid/fixture\'',
    '{"ENTITY": "fixture-team", "API_KEY": "fixture-key", "SLACK_WEBHOOK_URL": "https://example.invalid/fixture"}',
    '# literal configuration\n"wandb_entity": "fixture-team",\napi_key: fixture-key\nslack_webhook_url: https://example.invalid/fixture',
])
def test_json_and_flat_literal_formats(tmp_path, content):
    path = tmp_path / 'fixture.conf'
    path.write_text(content)
    assert cl_runtime.read_credentials(path) == {'wandb_entity': 'fixture-team', 'WANDB_API_KEY': 'fixture-key',
                                                'SLACK_WEBHOOK_URL': 'https://example.invalid/fixture'}


@pytest.mark.parametrize('content', [
    '{\napi_key: sensitive-fixture\nmalformed sensitive-fixture\n}',
    '{"api_key": "sensitive-fixture", "slack_webhook_url":',
    'api_key: "sensitive-fixture',
    '{\napi_key: sensitive-fixture',
    '{"api_key": ["sensitive-fixture"]}',
    'api_key: sensitive-fixture\nAPI_KEY: duplicate',
    'api_key: "sensitive-fixture\\nextra"',
])
def test_malformed_configuration_never_echoes_secret(tmp_path, content):
    path = tmp_path / 'fixture.conf'
    path.write_text(content)
    with pytest.raises(ValueError) as caught:
        cl_runtime.read_credentials(path)
    assert 'sensitive-fixture' not in str(caught.value)
    assert caught.value.__cause__ is None


def test_shell_substitution_remains_literal(tmp_path):
    path = tmp_path / 'literal.conf'
    path.write_text('api_key: $(do-not-execute)\nslack_webhook_url: `do-not-execute`')
    assert cl_runtime.read_credentials(path)['WANDB_API_KEY'] == '$(do-not-execute)'


def test_credential_file_selection_order(tmp_path, monkeypatch):
    explicit, environment, git, default = [tmp_path / name for name in ('explicit.conf', 'env.conf', 'git.conf', 'wandb.conf')]
    for path in (explicit, environment, git, default):
        path.write_text('{}')
    monkeypatch.setattr(cl_runtime, 'ROOT', tmp_path)
    lookup = Mock(return_value=str(git))
    monkeypatch.setattr(cl_runtime, '_git_credentials_file', lookup)
    assert cl_runtime._credentials_path(explicit, {'CL_CREDENTIALS_FILE': str(environment)}) == explicit
    assert cl_runtime._credentials_path(None, {'CL_CREDENTIALS_FILE': str(environment)}) == environment
    lookup.assert_not_called()
    assert cl_runtime._credentials_path(None, {}) == git
    lookup.return_value = None
    assert cl_runtime._credentials_path(None, {}) == default
    default.unlink()
    assert cl_runtime._credentials_path(None, {}) is None


def test_git_selection_uses_local_config_without_logging(tmp_path, monkeypatch):
    execute = Mock(return_value=type('Result', (), {'returncode': 0, 'stdout': '/private/credentials.conf\n'})())
    monkeypatch.setattr(cl_runtime.subprocess, 'run', execute)
    assert cl_runtime._git_credentials_file() == '/private/credentials.conf'
    assert execute.call_args.args[0] == ['git', 'config', '--local', '--get', 'cl.credentialsFile']


@pytest.mark.parametrize('command', [
    ['python', 'main.py', '--model', 'sgd'],
    ['/venv/bin/python3.12', '-u', '/checkout/main.py'],
    ['uv', 'run', '--no-sync', 'python', 'main.py'],
    ['main.py'],
])
def test_mammoth_command_receives_shared_defaults(command):
    profile = {'data_root': '/shared/data', 'gpu': '1'}
    prepared = cl_runtime.prepare_command(command, profile)
    assert prepared == command + ['--base_path', '/shared/data', '--device', 'cuda:0']
    assert command != prepared  # the original argv is retained


def test_explicit_mammoth_args_are_preserved():
    command = ['python', 'main.py', '--base_path=/explicit/data', '--device', 'cpu']
    assert cl_runtime.prepare_command(command, {'data_root': '/shared/data', 'gpu': '1'}) == command
    assert cl_runtime.prepare_command(['python', 'main.py'], {'data_root': '/shared/data', 'gpu': 'cpu'})[-2:] == ['--device', 'cpu']


@pytest.mark.parametrize('command', [
    ['pytest', 'tests'], ['python', '-c', 'print(1)'], ['cat', 'main.py'],
    ['python', '-c', 'main.py'], ['python', 'another_script.py', 'main.py'],
    ['uv', 'run', 'python', '-m', 'module', 'main.py'],
])
def test_generic_command_argv_is_unchanged(command):
    assert cl_runtime.prepare_command(command, {'data_root': '/shared/data', 'gpu': '1'}) == command


@pytest.mark.parametrize(('gpu', 'visible', 'device'), [('1', '1', 'cuda:0'), ('cpu', '', 'cpu')])
def test_wrapper_executes_with_gpu_mapping_only_after_configuration(monkeypatch, gpu, visible, device):
    profile = {'data_root': '/shared/data', 'gpu': gpu}
    configure = Mock(return_value=profile)
    execute = Mock()
    monkeypatch.setattr(cl_runtime, 'configure_environment', configure)
    monkeypatch.setattr(cl_runtime.os, 'execvpe', execute)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    cl_runtime.main(['--profile', '/fixture/public.json', '--credentials', '/fixture/private.conf', '--', 'python', 'main.py'])
    configure.assert_called_once_with(Path('/fixture/public.json'), Path('/fixture/private.conf'))
    assert execute.call_args.args[:2] == ('python', ['python', 'main.py', '--base_path', '/shared/data', '--device', device])
    assert execute.call_args.args[2]['CUDA_VISIBLE_DEVICES'] == visible


def test_import_has_no_configuration_or_environment_side_effects(monkeypatch):
    read = Mock(side_effect=AssertionError('Import must not read private configuration'))
    execute = Mock(side_effect=AssertionError('Import must not invoke Git or run commands'))
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(cl_runtime.subprocess, 'run', execute)
    before = dict(cl_runtime.os.environ)
    spec = importlib.util.spec_from_file_location('_cl_runtime_import_check', cl_runtime.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert dict(cl_runtime.os.environ) == before
    read.assert_not_called()
    execute.assert_not_called()


def test_invalid_public_profile_does_not_modify_environment(tmp_path, credentials):
    profile_path = tmp_path / 'invalid.json'
    profile_path.write_text('{"cache_root": "relative/cache"}')
    env = {'HOME': '/existing/home'}
    with pytest.raises(ValueError, match='absolute'):
        cl_runtime.configure_environment(profile_path, credentials, env)
    assert env == {'HOME': '/existing/home'}
