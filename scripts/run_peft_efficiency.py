"""Approved, sequential Base pilot: four LoRA methods and a fresh RanPAC.

Default operation prints the scope without imports of ML libraries, credentials,
network calls or experiments. --execute runs exactly 15 measurement workers and
five separate partial-FLOPs workers. --resume preserves completed workers.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.low_rank_protocol import CHECKPOINT, CHECKPOINT_SHA256

METHODS = ('ewc-lora', 'sd-lora', 'inf-lora', 'cl-lora', 'ranpac')
PLAN = ROOT / 'benchmarks/cl_efficiency/peft_pilot/plan.json'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_plan():
    plan = json.loads(PLAN.read_text())
    expected = {'dataset': 'seq-cifar100-224', 'dataset_config': 'ease', 'tasks': 1,
                'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                'gpu': '1', 'vit_size': 'base', 'lora_rank': 10,
                'measurement_repeats': 3, 'flops_runs_per_method': 1,
                'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'}
    if (plan.get('training_authorized') is not True or plan.get('methods') != list(METHODS)
            or any(type(plan.get(key)) is not type(value) or plan.get(key) != value
                   for key, value in expected.items())
            or plan.get('checkpoint') != {'file': CHECKPOINT, 'sha256': CHECKPOINT_SHA256}
            or plan.get('new_measurement_runs') != 15 or plan.get('new_flops_runs') != 5):
        raise ValueError('Plan differs from the approved five-method, first-task scope')
    return plan


def jobs(output):
    result = []
    for method in METHODS:
        for index, mode in enumerate(('measurement', 'measurement', 'measurement', 'flops'), 1):
            directory = Path(output) / method / (f'measurement{index}' if mode == 'measurement' else 'flops')
            script = 'run_ranpac_efficiency.py' if method == 'ranpac' else 'run_low_rank_efficiency.py'
            command = [sys.executable, str(ROOT / 'scripts' / script), '--pass', mode,
                       '--output', str(directory.resolve()), '--execute']
            if method != 'ranpac':
                command.extend(['--method', method])
            result.append({'method': method, 'pass': mode, 'directory': str(directory.resolve()),
                           'command': command, 'status': 'pending'})
    return result


def source_hashes():
    names = ['scripts/run_peft_efficiency.py', 'scripts/report_peft_efficiency.py',
             'scripts/run_low_rank_efficiency.py', 'scripts/run_ranpac_efficiency.py',
             'scripts/run_low_rank_validation.py', 'scripts/run_low_rank_reference.py',
             'scripts/low_rank_protocol.py', 'scripts/baseline_audit.py',
             'scripts/benchmark_cl.py', 'scripts/cl_efficiency.py', 'scripts/cl_runtime.py',
             'scripts/report_hide_results.py', 'main.py', 'utils/training.py', 'utils/evaluate.py',
             'backbone/low_rank_vit.py', 'backbone/vit.py', 'uv.lock',
             'datasets/configs/seq-cifar100-224/ease.yaml', 'datasets/seq_cifar100_224.py',
             'benchmarks/cl_efficiency/peft_pilot/plan.json']
    for method in METHODS:
        name = method.replace('-', '_')
        names.append(f'models/{name}.py')
        if method != 'ranpac' or (ROOT / f'models/config/{name}.yaml').is_file():
            names.append(f'models/config/{name}.yaml')
    names.extend(str(path.relative_to(ROOT)) for directory in
                 ('models/low_rank_utils', 'models/ranpac_utils')
                 for path in (ROOT / directory).rglob('*.py'))
    return {name: file_hash(ROOT / name) for name in sorted(set(names))}


def verified_record(job):
    path = Path(job['directory']) / 'result.json'
    record = json.loads(path.read_text())
    from scripts.report_peft_efficiency import validate_new_record
    validate_new_record(record, job['method'], job['pass'])
    return {'file': str(path), 'sha256': file_hash(path), 'wandb_url': record['wandb_url']}


def validate_schedule(saved):
    expected = jobs(Path(saved['output']))
    actual = saved.get('jobs', [])
    immutable = ('method', 'pass', 'directory', 'command')
    if (len(actual) != 20 or any(any(left.get(key) != right[key] for key in immutable)
                               for left, right in zip(actual, expected))):
        raise ValueError('Saved schedule differs from the exact approved twenty-worker order and commands')
    return True


def gpu_occupancy():
    gpu = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True).strip()
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    active = []
    for row in rows:
        fields = [item.strip() for item in row.split(',')]
        if len(fields) == 2 and fields[0] == gpu and fields[1].isdigit():
            pid = int(fields[1])
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            active.append(pid)
    if active:
        raise RuntimeError('GPU1 has other live compute processes; no worker started')
    return {'gpu_uuid': gpu, 'active_compute_pids': active}


def stop_launcher(process):
    """Give the LoRA parent a chance to clean up its separate worker group."""
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        import psutil
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait(timeout=10)


def execute(options):
    plan = load_plan()
    profile = configure_environment()
    if any(profile[key] != value for key, value in
           (('gpu', '1'), ('wandb_entity', 'laymond1'), ('wandb_project', 'cl-efficiency'))):
        raise ValueError('Effective GPU or destination differs from the approved scope')
    if file_hash(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('Shared pretrained checkpoint changed')
    if options.resume:
        directory = options.resume.resolve()
        saved = json.loads((directory / 'execution.json').read_text())
        if saved.get('plan') != plan or saved.get('execution_source_sha256') != source_hashes():
            raise ValueError('Resume requires the same frozen protocol and source files')
        validate_schedule(saved)
    else:
        directory = options.output.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        saved = {'schema_version': 1, 'status': 'running', 'plan': plan,
                 'output': str(directory), 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'execution_source_sha256': source_hashes(), 'jobs': jobs(directory),
                 'notifications': [], 'failures': []}
    notify = lambda message: saved['notifications'].append({'message': message,
                              'status': notify_slack(message)})
    saved['status'] = 'running'
    notify('Base PEFT 효율성 pilot 시작/재개: LoRA 4종 + RanPAC, 첫 태스크, 측정15회 + 별도 FLOPs5회. GPU1; W&B laymond1/cl-efficiency.')
    write_json(directory / 'execution.json', saved)
    process = None
    try:
        for index, job in enumerate(saved['jobs'], 1):
            if source_hashes() != saved['execution_source_sha256']:
                raise ValueError('Source changed after the pilot was frozen')
            if job['status'] == 'completed':
                if verified_record(job) != job['result']:
                    raise ValueError('Completed worker evidence changed before resume')
                continue
            job_directory = Path(job['directory'])
            if (job_directory / 'result.json').is_file():
                # A parent interruption after a successful worker does not retrain it.
                job['result'] = verified_record(job)
                job['status'] = 'completed'
                write_json(directory / 'execution.json', saved)
                continue
            if job_directory.exists() and any(job_directory.iterdir()):
                raise ValueError('Incomplete worker directory retained; diagnose and archive it before retrying')
            job['gpu_occupancy_before_start'] = gpu_occupancy()
            log_directory = directory / 'launcher_logs'
            log_directory.mkdir(exist_ok=True)
            log_path = log_directory / f'{job["method"]}-{job_directory.name}.log'
            job['launcher_log'] = str(log_path)
            job.update(status='running', started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(directory / 'execution.json', saved)
            print(f'[{index}/20] {job["method"]} {job["pass"]} starting', flush=True)
            notify(f'PEFT pilot {index}/20 시작: {job["method"]} {job["pass"]}; GPU1, 첫 태스크.')
            with log_path.open('a', encoding='utf-8') as log:
                process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                worker_started = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - worker_started > 3600:
                        raise TimeoutError('First-task worker exceeded its one-hour execution limit')
                    job['worker_launcher_pid'] = process.pid
                    write_json(directory / 'execution.json', saved)
                    time.sleep(5)
                job['exit_code'] = process.returncode
            if process.returncode:
                job['status'] = 'failed'
                raise RuntimeError(f'{job["method"]} {job["pass"]} failed; see {log_path}')
            job['result'] = verified_record(job)
            job.update(status='completed', completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(directory / 'execution.json', saved)
            print(f'[{index}/20] completed {job["result"]["wandb_url"]}', flush=True)
            notify(f'PEFT pilot {index}/20 완료: {job["method"]} {job["pass"]}; W&B {job["result"]["wandb_url"]}')
        saved.update(status='completed', completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(directory / 'execution.json', saved)
        print(f'All 20 workers completed. Local execution: {directory / "execution.json"}', flush=True)
        return 0
    except BaseException as error:
        stop_launcher(process)
        saved['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        saved['failures'].append({'type': type(error).__name__, 'message': str(error),
                                  'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        notify(f'PEFT pilot 중단: {type(error).__name__}. 완료 기록과 실패 로그를 보존하고 원인을 점검합니다.')
        write_json(directory / 'execution.json', saved)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/peft_pilot/base-20261008')
    parser.add_argument('--resume', type=Path)
    options = parser.parse_args(argv)
    if not options.execute:
        print(json.dumps({'execute': False, 'plan': load_plan(), 'schedule': jobs(options.output)}, ensure_ascii=False))
        return 0
    previous = signal.getsignal(signal.SIGTERM)
    def cancelled(signum, frame):
        raise KeyboardInterrupt('Pilot controller termination requested')
    signal.signal(signal.SIGTERM, cancelled)
    try:
        return execute(options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
