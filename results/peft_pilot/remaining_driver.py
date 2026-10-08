"""Resume only InfLoRA, CL-LoRA and RanPAC after quarantining the SD collapse.

This result-side controller does not change the frozen experiment sources. The
default is a read-only preflight. --execute reuses the original twenty-job
schedule, launches only jobs 9..20, and ends with partial_completed. No SD run
is repeated or accepted. Completed EWC and remaining workers are never retrained.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import run_peft_efficiency as original
from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.report_peft_efficiency import read_json, validate_new_record

FROZEN_COMMIT = '04559dbdc8c7e68adb96da7871d9d31c2fc5365e'
DEFAULT_EXECUTION = ROOT / 'results/peft_pilot/base-20261008/execution.json'
REMAINING_METHODS = ('inf-lora', 'cl-lora', 'ranpac')
DEFERRED_REASON = 'SD first-task accuracy collapsed; investigate before any further SD execution'


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_frozen(saved):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if commit != FROZEN_COMMIT or saved.get('execution_source_sha256') != original.source_hashes():
        raise ValueError('Remaining workers require the original frozen commit and exact source fingerprints')
    if saved.get('plan') != original.load_plan():
        raise ValueError('Approved first-task experiment plan changed')
    original.validate_schedule(saved)


def sd_quarantine(saved):
    job = saved['jobs'][4]
    path = Path(job['directory']) / 'result.json'
    record = read_json(path)
    accuracy = record.get('accuracy', {}).get('mean_accuracy_pct')
    if (job['method'] != 'sd-lora' or job['pass'] != 'measurement'
            or record.get('method', record.get('model')) != 'sd-lora'
            or record.get('pass') != 'measurement' or record.get('status') != 'completed'
            or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded'
            or isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
            or not math.isfinite(accuracy) or accuracy != 10.0):
        raise ValueError('Expected saved SD 10% collapse evidence is unavailable or changed')
    try:
        validate_new_record(record, 'sd-lora', 'measurement')
    except ValueError as error:
        if str(error) != 'Investigate abnormal one-epoch accuracy before accepting the worker':
            raise ValueError('SD failed a different gate; do not silently quarantine unrelated failures') from None
    else:
        raise ValueError('SD collapse unexpectedly passed the original accuracy gate')
    proof = {'file': str(path.resolve()), 'sha256': original.file_hash(path),
             'bytes': path.stat().st_size, 'wandb_url': record['wandb_url'],
             'accuracy_pct': accuracy, 'computation_status': record['computation_status'],
             'wandb_logging_status': record['wandb_logging_status'], 'accepted': False,
             'reason': DEFERRED_REASON}
    old = saved.get('sd_quarantine')
    if old is not None and old != proof:
        raise ValueError('Quarantined SD evidence changed after the partial continuation started')
    return proof


def preflight(execution):
    execution = Path(execution).resolve()
    saved = read_json(execution)
    if Path(saved.get('output', '')).resolve() != execution.parent:
        raise ValueError('Execution state must stay in its original output directory')
    if saved.get('status') not in ('failed', 'interrupted', 'running', 'partial_completed'):
        raise ValueError('Only the interrupted SD pilot or this partial continuation can be resumed')
    check_frozen(saved)
    for job in saved['jobs'][:4]:
        if job.get('status') != 'completed' or original.verified_record(job) != job.get('result'):
            raise ValueError('All four already completed EWC workers must remain verified and unchanged')
    proof = sd_quarantine(saved)
    for job in saved['jobs'][5:8]:
        if job.get('status') not in ('pending', 'deferred'):
            raise ValueError('Additional SD workers already changed state; partial scope needs review')
        directory = Path(job['directory'])
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Deferred SD output contains unexpected evidence; preserve it for review')
    for index, job in enumerate(saved['jobs'][8:], 9):
        if job['method'] not in REMAINING_METHODS or job.get('status') == 'deferred':
            raise ValueError('Remaining schedule differs from the twelve approved unaffected workers')
        if job.get('status') == 'completed' and original.verified_record(job) != job.get('result'):
            raise ValueError(f'Previously completed worker {index} evidence changed')
    return saved, proof


def execute(execution):
    execution = Path(execution).resolve()
    saved, proof = preflight(execution)
    profile = configure_environment()
    if any(profile[key] != value for key, value in
           (('gpu', '1'), ('wandb_entity', 'laymond1'), ('wandb_project', 'cl-efficiency'))):
        raise ValueError('Effective GPU or logging destination differs from the approved scope')
    if original.file_hash(original.CHECKPOINT) != original.CHECKPOINT_SHA256:
        raise ValueError('Shared pretrained checkpoint changed')
    previous = saved.get('partial_continuation')
    if previous is None:
        saved['partial_continuation'] = {
            'driver_file': str(Path(__file__).resolve()), 'driver_sha256': original.file_hash(__file__),
            'original_execution_sha256': original.file_hash(execution), 'frozen_commit': FROZEN_COMMIT,
            'started_at_utc': utc_now(), 'only_original_job_indices': list(range(9, 21)),
            'remaining_methods': list(REMAINING_METHODS), 'full_twenty_workers_completed': False}
    elif (previous.get('driver_sha256') != original.file_hash(__file__)
          or previous.get('only_original_job_indices') != list(range(9, 21))
          or previous.get('frozen_commit') != FROZEN_COMMIT):
        raise ValueError('Durable remaining-controller scope or source changed before resume')
    saved['sd_quarantine'] = proof
    saved['jobs'][4].update(status='quarantined', quarantine=proof)
    for job in saved['jobs'][5:8]:
        job.update(status='deferred', deferred_reason=DEFERRED_REASON)
    saved['status'] = 'running'
    write_json(execution, saved)

    def notify(message):
        saved.setdefault('notifications', []).append({'message': message, 'status': notify_slack(message)})
        write_json(execution, saved)

    process, active_job = None, None
    try:
        notify('PEFT pilot 부분 재개: SD10% 결과 격리 및 SD3회 보류. 완료 EWC4회는 재사용하며 '
               '기존 InfLoRA·CL-LoRA·RanPAC12개 실행만 GPU1에서 진행합니다.')
        for index, job in enumerate(saved['jobs'][8:], 9):
            check_frozen(saved)
            sd_quarantine(saved)
            active_job = job
            if job.get('status') == 'completed':
                if original.verified_record(job) != job.get('result'):
                    raise ValueError('Completed remaining worker changed; no retraining allowed')
                continue
            directory = Path(job['directory'])
            if (directory / 'result.json').is_file():
                job['result'] = original.verified_record(job)
                job.update(status='completed', adopted_after_interruption=True, completed_at_utc=utc_now())
                write_json(execution, saved)
                continue
            if directory.exists():
                if any(directory.iterdir()):
                    raise ValueError('Incomplete worker output retained; diagnose and archive it before retrying')
                # The original standalone worker creates its own fresh directory.
                directory.rmdir()
            job['gpu_occupancy_before_start'] = original.gpu_occupancy()
            log_directory = execution.parent / 'remaining_launcher_logs'
            log_directory.mkdir(exist_ok=True)
            log_path = log_directory / f'{job["method"]}-{directory.name}.log'
            job.update(status='running', started_at_utc=utc_now(), launcher_log=str(log_path))
            write_json(execution, saved)
            print(f'[{index}/20; partial] {job["method"]} {job["pass"]} starting', flush=True)
            notify(f'PEFT 부분 pilot 원래 {index}/20 시작: {job["method"]} {job["pass"]}; SD 격리 유지.')
            with log_path.open('a', encoding='utf-8') as log:
                process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                job['worker_launcher_pid'] = process.pid
                started = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - started > 3600:
                        raise TimeoutError('Remaining first-task worker exceeded the original one-hour limit')
                    write_json(execution, saved)
                    time.sleep(5)
                job['exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError(f'{job["method"]} {job["pass"]} failed; preserved log: {log_path}')
            job['result'] = original.verified_record(job)
            job.update(status='completed', completed_at_utc=utc_now())
            write_json(execution, saved)
            notify(f'PEFT 부분 pilot 원래 {index}/20 완료: {job["method"]} {job["pass"]}; '
                   f'W&B {job["result"]["wandb_url"]}')
        if any(job['status'] != 'completed' for job in saved['jobs'][8:]):
            raise ValueError('Partial continuation is missing an unaffected worker')
        saved.update(status='partial_completed', partial_completed_at_utc=utc_now())
        saved['partial_continuation']['unaffected_workers_completed'] = 12
        saved['partial_continuation']['accepted_workers_including_ewc'] = 16
        saved['partial_continuation']['sd_quarantined_workers'] = 1
        saved['partial_continuation']['sd_deferred_workers'] = 3
        write_json(execution, saved)
        notify('PEFT 부분 pilot 완료: EWC4 + InfLoRA4 + CL-LoRA4 + RanPAC4의16개 승인 기록 완료. '
               'SD10% 1개는 격리, SD3개는 보류; 전체20개 완료가 아닙니다.')
        print(json.dumps({'status': 'partial_completed', 'accepted_workers': 16,
                          'sd_quarantined': 1, 'sd_deferred': 3, 'execution': str(execution)}), flush=True)
        return 0
    except BaseException as error:
        original.stop_launcher(process)
        if active_job is not None and active_job.get('status') == 'running':
            active_job['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        saved['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        saved.setdefault('failures', []).append({'type': type(error).__name__, 'message': str(error),
                                               'at_utc': utc_now(), 'controller': 'remaining_driver'})
        write_json(execution, saved)
        notify(f'PEFT 부분 pilot 중단: {type(error).__name__}; 완료 기록·SD 격리·실패 로그 보존.')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if not options.execute:
        saved, proof = preflight(options.execution)
        print(json.dumps({'execute': False, 'frozen_commit': FROZEN_COMMIT, 'sd_quarantine': proof,
                          'would_run_only_original_jobs': [
                              {'original_index': index, **job}
                              for index, job in enumerate(saved['jobs'][8:], 9)
                              if job.get('status') != 'completed'],
                          'final_status': 'partial_completed'}, ensure_ascii=False))
        return 0
    previous = signal.getsignal(signal.SIGTERM)
    def cancelled(signum, frame):
        raise KeyboardInterrupt('Remaining pilot controller termination requested')
    signal.signal(signal.SIGTERM, cancelled)
    try:
        return execute(options.execution)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
