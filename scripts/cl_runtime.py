"""Load shared CL paths and private credentials, or run a command in that environment.

Importing this module does not read configuration or modify the environment. Public
profiles may be committed; credential files must remain private and untracked.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / 'benchmarks/cl_efficiency/runtime.json'
PUBLIC_DEFAULTS = {
    'gpu': '1',
    'wandb_entity': 'laymond1',
    'wandb_project': 'cl-efficiency',
    'data_root': '/mnt/sdab1/share/data',
    'cache_root': '/mnt/sdab1/userHome/wslim/.cache',
}
PUBLIC_ENV = {
    'gpu': 'CL_GPU',
    'wandb_entity': 'WANDB_ENTITY',
    'wandb_project': 'WANDB_PROJECT',
    'data_root': 'CL_DATA_ROOT',
    'cache_root': 'CL_CACHE_ROOT',
}
_CREDENTIAL_ALIASES = {
    'entity': 'wandb_entity',
    'wandb_entity': 'wandb_entity',
    'project': 'wandb_project',
    'wandb_project': 'wandb_project',
    'api_key': 'WANDB_API_KEY',
    'wandb_api_key': 'WANDB_API_KEY',
    'slack_webhook_url': 'SLACK_WEBHOOK_URL',
}


def _read_text(path, kind):
    try:
        return Path(path).read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        # Exception messages deliberately omit contents and private file paths.
        raise ValueError(f'Unable to read {kind} configuration') from None


def _flat_credentials(text):
    """Parse only flat literal key:value/key=value lines; never execute content."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip() == '{':
        if len(lines) < 2 or lines[-1].strip() != '}':
            raise ValueError('Malformed credentials configuration')
        lines = lines[1:-1]
    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r"(?:([A-Za-z_][A-Za-z0-9_]*)|\"([A-Za-z_][A-Za-z0-9_]*)\"|'([A-Za-z_][A-Za-z0-9_]*)')\s*[:=]\s*(.+)", line)
        if not match:
            raise ValueError('Malformed credentials configuration')
        key = next(group for group in match.groups()[:3] if group is not None)
        value = match.group(4).strip()
        if value.endswith(','):
            value = value[:-1].rstrip()
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                raise ValueError('Malformed credentials configuration') from None
            if not isinstance(value, str):
                raise ValueError('Malformed credentials configuration')
        elif value.startswith("'"):
            if len(value) < 2 or not value.endswith("'") or "'" in value[1:-1]:
                raise ValueError('Malformed credentials configuration')
            value = value[1:-1]
        elif not value or any(char in value for char in '{}[]\"\''):
            raise ValueError('Malformed credentials configuration')
        if key.lower() in values:
            raise ValueError('Duplicate credentials setting')
        values[key.lower()] = value
    return values


def read_credentials(path):
    """Read private settings, retaining only supported public and secret keys.

    Accept strict JSON objects or flat literal lines, optionally surrounded by
    braces on separate lines. Error messages never include file contents.
    """
    text = _read_text(path, 'credentials')
    try:
        values = json.loads(text)
    except ValueError:
        values = None
    # Parse outside the JSON exception handler so malformed flat credentials do
    # not retain a JSONDecodeError (and its private input) as exception context.
    if values is None:
        values = _flat_credentials(text)
    if not isinstance(values, dict):
        raise ValueError('Credentials configuration must contain flat settings')
    filtered = {}
    for key, value in values.items():
        alias = _CREDENTIAL_ALIASES.get(str(key).lower())
        if alias is None:
            continue
        if not isinstance(value, str) or '\n' in value or '\r' in value or '\0' in value:
            raise ValueError('Credentials configuration contains an invalid setting')
        if alias in filtered:
            raise ValueError('Duplicate credentials setting')
        filtered[alias] = value.strip()
    return filtered


def _git_credentials_file():
    try:
        result = subprocess.run(['git', 'config', '--local', '--get', 'cl.credentialsFile'],
                                cwd=ROOT, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _credentials_path(explicit, environ):
    candidate = explicit or environ.get('CL_CREDENTIALS_FILE') or _git_credentials_file()
    if candidate:
        path = Path(candidate).expanduser()
        return path if path.is_absolute() else ROOT / path
    default = ROOT / 'wandb.conf'
    return default if default.is_file() else None


def _public_profile(path):
    if path is None and not DEFAULT_PROFILE.is_file():
        return {}
    text = _read_text(path or DEFAULT_PROFILE, 'public profile')
    try:
        values = json.loads(text)
    except ValueError:
        raise ValueError('Malformed public profile configuration') from None
    if not isinstance(values, dict):
        raise ValueError('Public profile configuration must contain flat settings')
    return {key: value for key, value in values.items() if key in PUBLIC_DEFAULTS}


def configure_environment(profile_path=None, credentials_path=None, environ=None):
    """Apply shared paths and credentials; return only the effective public profile.

    Precedence for public values is defaults, credential entity/project, public
    profile, then existing CL_*/WANDB_* environment overrides. Existing nonempty
    environment secrets win. Credential selection is explicit path, environment
    path, local Git configuration, then the current worktree's wandb.conf.
    HOME, CODEX_HOME and CUDA_VISIBLE_DEVICES are never changed here.
    """
    environ = os.environ if environ is None else environ
    profile = dict(PUBLIC_DEFAULTS)
    path = _credentials_path(credentials_path, environ)
    credentials = read_credentials(path) if path is not None else {}
    for key in ('wandb_entity', 'wandb_project'):
        if credentials.get(key):
            profile[key] = credentials[key]
    profile.update(_public_profile(profile_path))
    for key, name in PUBLIC_ENV.items():
        if environ.get(name):
            profile[key] = environ[name]
    if any(not isinstance(value, str) or not value or any(char in value for char in '\n\r\0') for value in profile.values()):
        raise ValueError('Public profile contains an invalid setting')
    if not re.fullmatch(r'cpu|[0-9]+|GPU-[A-Za-z0-9-]+', profile['gpu']):
        raise ValueError('Select one physical GPU index, UUID, or cpu')
    if not all(Path(profile[key]).is_absolute() for key in ('data_root', 'cache_root')):
        raise ValueError('Shared data and cache roots must be absolute paths')
    for key, name in PUBLIC_ENV.items():
        environ[name] = profile[key]
    cache = Path(profile['cache_root'])
    environ.update({
        'XDG_CACHE_HOME': str(cache),
        'TORCH_HOME': str(cache / 'torch'),
        'HF_HOME': str(cache / 'huggingface'),
        'HF_HUB_CACHE': str(cache / 'huggingface/hub'),
        'HF_DATASETS_CACHE': str(cache / 'huggingface/datasets'),
    })
    for name in ('WANDB_API_KEY', 'SLACK_WEBHOOK_URL'):
        if not environ.get(name) and credentials.get(name):
            environ[name] = credentials[name]
    return profile


def prepare_command(command, profile):
    """Inject Mammoth path/device defaults while preserving explicit arguments."""
    command = list(command)
    if not command:
        raise ValueError('A command is required after --')
    python_name = r'python(?:[0-9]+(?:\.[0-9]+)*)?|pypy(?:[0-9]+)?'
    executable = Path(command[0]).name
    if executable == 'main.py':
        script = 0
    else:
        python_index = 0 if re.fullmatch(python_name, executable) else None
        if executable == 'uv' and len(command) > 1 and command[1] == 'run':
            python_index = next((index for index, arg in enumerate(command[2:], 2)
                                 if re.fullmatch(python_name, Path(arg).name)), None)
        script = None
        if python_index is not None:
            index = python_index + 1
            while index < len(command):
                arg = command[index]
                if arg in ('-c', '-m'):
                    break
                if arg in ('-W', '-X'):
                    index += 2
                    continue
                if arg == '--':
                    index += 1
                    if index < len(command) and Path(command[index]).name == 'main.py':
                        script = index
                    break
                if not arg.startswith('-'):
                    script = index if Path(arg).name == 'main.py' else None
                    break
                index += 1
    if script is None:
        return command
    args = command[script + 1:]
    if not any(arg == '--base_path' or arg.startswith('--base_path=') for arg in args):
        command.extend(['--base_path', profile['data_root']])
    if not any(arg == '--device' or arg.startswith('--device=') for arg in args):
        command.extend(['--device', 'cpu' if profile['gpu'] == 'cpu' else 'cuda:0'])
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    try:
        profile = configure_environment(args.profile, args.credentials)
        command = prepare_command(command, profile)
    except ValueError as error:
        parser.error(str(error))
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if profile['gpu'] == 'cpu' else profile['gpu']
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
