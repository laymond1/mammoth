"""Check shared CL prerequisites and explicitly requested service connections.

This command never constructs a model, loads a dataset, or runs training/inference.
The default makes no W&B run or Slack post; each connection test requires its flag.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import datetime
import importlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
from urllib.parse import quote
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, query_gpu, write_json
from scripts.cl_runtime import configure_environment


PUBLIC_KEYS = ('gpu', 'data_root', 'cache_root', 'wandb_entity', 'wandb_project')
PACKAGES = {
    # In this environment, open_clip first succeeds; torch/torchvision/timm first
    # followed by open_clip terminates with SIGSEGV during dependency inspection.
    'open_clip': 'open-clip-torch',
    'torch': 'torch', 'torchvision': 'torchvision', 'timm': 'timm',
    'psutil': 'psutil', 'clip': 'clip', 'kornia': 'kornia',
    'xitorch': 'xitorch', 'sklearn': 'scikit-learn',
    'wandb': 'wandb',
}
CHECK_METRICS = {
    'connectivity/test': 1, 'training_executed': False, 'inference_executed': False,
    'experiment_executed': False, 'check_kind': 'environment-and-connectivity',
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', type=Path, help='Directory for this preparation report and SDK files')
    result.add_argument('--check-wandb', action='store_true', help='Create one online connectivity run and verify its saved summary')
    result.add_argument('--check-slack', action='store_true', help='Send one connection test notification')
    return result


@contextmanager
def quiet_sdk():
    """Discard SDK diagnostics; exception strings may contain private values."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def inspect_environment(profile):
    """Import dependencies and inspect CUDA properties without tensor operations."""
    packages, torch_module = {}, None
    for module, distribution in PACKAGES.items():
        try:
            # Visibility is configured by main() before this first torch import.
            with quiet_sdk():
                imported = importlib.import_module(module)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
            packages[module] = {'status': 'passed', 'version': version}
            if module == 'torch':
                torch_module = imported
        except Exception as error:
            packages[module] = {'status': 'failed', 'error_type': type(error).__name__}

    cache = Path(profile['cache_root'])
    paths = {
        'data_root': Path(profile['data_root']), 'cache_root': cache,
        'torch_checkpoints': cache / 'torch/hub/checkpoints',
        'huggingface_hub': cache / 'huggingface/hub', 'clip_cache': cache / 'clip',
    }
    path_status = {name: {'path': str(path), 'exists': path.exists(), 'is_directory': path.is_dir(),
                          'readable': os.access(path, os.R_OK), 'writable': os.access(path, os.W_OK)}
                   for name, path in paths.items()}
    gpu = {'physical_device': profile['gpu']}
    if profile['gpu'] == 'cpu':
        gpu.update(status='cpu_selected', cuda_status='not_requested', nvidia_status='not_requested')
    else:
        try:
            gpu['nvidia'] = query_gpu(profile['gpu'])
            gpu['nvidia_status'] = 'passed'
        except Exception as error:
            gpu.update(nvidia_status='failed', nvidia_error_type=type(error).__name__)
        try:
            if torch_module is None:
                raise ImportError('Torch is unavailable')
            with quiet_sdk():
                if not torch_module.cuda.is_available():
                    raise RuntimeError('CUDA is unavailable')
                torch_module.cuda.init()
                properties = torch_module.cuda.get_device_properties(0)
            gpu['cuda'] = {'logical_device': 0, 'name': properties.name,
                           'total_memory_bytes': int(properties.total_memory),
                           'compute_capability': [int(properties.major), int(properties.minor)],
                           'multiprocessors': int(properties.multi_processor_count),
                           'torch_cuda_version': torch_module.version.cuda}
            gpu['cuda_status'] = 'passed'
        except Exception as error:
            gpu.update(cuda_status='failed', cuda_error_type=type(error).__name__)
        gpu['status'] = 'passed' if gpu['cuda_status'] == gpu['nvidia_status'] == 'passed' else 'failed'
    required_paths = all(path_status[key]['is_directory'] and path_status[key]['readable']
                         for key in ('data_root', 'cache_root'))
    status = 'passed' if (required_paths and all(item['status'] == 'passed' for item in packages.values())
                          and gpu['status'] != 'failed') else 'failed'
    return {'status': status, 'python': platform.python_version(), 'packages': packages,
            'paths': path_status, 'gpu': gpu}


def check_wandb(profile, report, output):
    """Send only synthetic connection markers and read back server metrics.

    Detailed environment information stays in the local report.json. The sole
    artifact contains no profile, paths, package/GPU details, or run URL.
    """
    profile = {key: profile[key] for key in PUBLIC_KEYS}
    result = {'status': 'failed', 'run_url': None, 'server_summary_verified': False,
              'artifact_uploaded': False}
    run, finished = None, False
    os.environ.update({'WANDB_SILENT': 'true', 'WANDB_CONSOLE': 'off',
                       'WANDB_DISABLE_CODE': 'true', 'WANDB_DISABLE_GIT': 'true'})
    os.environ.pop('WANDB_CONFIG_PATHS', None)
    try:
        with quiet_sdk():
            wandb = importlib.import_module('wandb')
            sdk_dir = output / 'wandb_sdk'
            sdk_dir.mkdir(parents=True, exist_ok=True)
            settings = wandb.Settings(silent=True, disable_git=True, disable_code=True,
                                      disable_job_creation=True, console='off', save_code=False,
                                      config_paths=None,
                                      x_disable_stats=True, x_disable_meta=True, init_timeout=30,
                                      finish_timeout=30, finish_timeout_raises=True,
                                      ignore_globs=['**/wandb.conf', '**/.env', '**/.env.*'])
            test_id = uuid.uuid4().hex[:12]
            run = wandb.init(entity=profile['wandb_entity'], project=profile['wandb_project'],
                             mode='online', job_type='connectivity-test', tags=['no-training', 'environment-check'],
                             id=test_id, resume='never', name='connectivity-' + test_id, dir=str(sdk_dir),
                             config=dict(CHECK_METRICS), settings=settings)
            if run is None or not re.fullmatch(r'[A-Za-z0-9_-]+', str(run.id)):
                raise ValueError('Invalid online run')
            run_path = f"{profile['wandb_entity']}/{profile['wandb_project']}/{run.id}"
            result['run_url'] = (f"https://wandb.ai/{quote(profile['wandb_entity'], safe='')}/"
                                 f"{quote(profile['wandb_project'], safe='')}/runs/{run.id}")
            run.log(dict(CHECK_METRICS))
            run.summary.update(dict(CHECK_METRICS))
            # Never add_dir(), log_code(), or pass a private configuration file.
            public_file = output / 'connectivity_report.json'
            write_json(public_file, {'schema_version': 1, **CHECK_METRICS})
            artifact = wandb.Artifact('environment-check-' + str(run.id), type='connectivity-check')
            artifact.add_file(str(public_file), name='connectivity_report.json')
            logged = run.log_artifact(artifact)
            logged.wait(timeout=30)
            result['artifact_uploaded'] = True
            run.finish(exit_code=0)
            finished = True
            remote = wandb.Api(timeout=30).run(run_path)
            if not all(remote.summary.get(key) == value for key, value in CHECK_METRICS.items()):
                raise ValueError('Server summary verification failed')
            result.update(status='passed', server_summary_verified=True)
    except Exception as error:
        result['error_type'] = type(error).__name__
    finally:
        if run is not None and not finished:
            try:
                with quiet_sdk():
                    run.finish(exit_code=1)
            except Exception as error:
                result['finish_error_type'] = type(error).__name__
    return result


def check_slack(report):
    """Send one synthetic test message without environment information or URLs."""
    message = ('CL 환경 연결 테스트: 학습·추론 실행 없음. '
               f"W&B 검사={report['wandb']['status']}.")
    try:
        status = notify_slack(message)
        if status not in ('sent', 'disabled', 'missing_or_invalid_webhook', 'http_error') and not re.fullmatch(r'failed:[A-Za-z][A-Za-z0-9_]*', status):
            status = 'unexpected_delivery_status'
        return {'status': 'passed' if status == 'sent' else 'failed', 'delivery_status': status}
    except Exception as error:
        return {'status': 'failed', 'error_type': type(error).__name__}


def main(argv=None):
    args = parser().parse_args(argv)
    now = datetime.datetime.now(datetime.timezone.utc)
    output = (args.output or ROOT / 'benchmarks/cl_efficiency/environment_checks' /
              (now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])).resolve()
    report = {'schema_version': 1, 'check_kind': 'environment-and-connectivity',
              'created_at_utc': now.isoformat(), 'experiment_executed': False,
              'training_executed': False, 'inference_executed': False,
              'wandb': {'status': 'not_requested'}, 'slack': {'status': 'not_requested'}}
    try:
        configured = configure_environment()
        profile = {key: configured[key] for key in PUBLIC_KEYS}
        report['profile'] = profile
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if profile['gpu'] == 'cpu' else str(profile['gpu'])
        report['environment'] = inspect_environment(profile)
    except Exception as error:
        report['environment'] = {'status': 'failed', 'error_type': type(error).__name__}
    write_json(output / 'report.json', report)
    if args.check_wandb:
        if 'profile' in report:
            report['wandb'] = check_wandb(report['profile'], report, output)
        else:
            report['wandb'] = {'status': 'failed', 'error_type': 'ConfigurationUnavailable'}
        write_json(output / 'report.json', report)
    if args.check_slack:
        report['slack'] = check_slack(report)
        write_json(output / 'report.json', report)
    passed = all(report[key]['status'] in ('passed', 'not_requested') for key in ('environment', 'wandb', 'slack'))
    report['status'] = 'passed' if passed else 'failed'
    write_json(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'environment': report['environment']['status'],
                      'wandb': report['wandb']['status'], 'slack': report['slack']['status'],
                      'run_url': report['wandb'].get('run_url'), 'report': str(output / 'report.json'),
                      'training_executed': False, 'inference_executed': False}, ensure_ascii=False))
    return int(not passed)


if __name__ == '__main__':
    sys.exit(main())
