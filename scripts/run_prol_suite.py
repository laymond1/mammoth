"""Six approved PROL runs; pair verification and normality review gate pilots."""

import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prol_protocol as shared
from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.run_peft_efficiency import gpu_occupancy, stop_launcher

DEFAULT_OUTPUT = ROOT / 'results/prol_validation_20261008'
DEFAULT_INITIAL = ROOT / 'results/prol_initial_20261008/initial_state.pt'


def jobs(output, initial_state=DEFAULT_INITIAL):
    result = []
    for name in shared.JOB_NAMES:
        side = 'author' if name == 'validation_author' else 'native'
        mode = 'validation' if name.startswith('validation') else 'efficiency'
        kind = 'flops' if name == 'flops' else 'measurement' if name.startswith('measurement') else 'validation'
        directory = (Path(output) / name).resolve()
        result.append({'name': name, 'side': side, 'mode': mode, 'pass': kind, 'directory': str(directory),
                       'status': 'pending', 'command': [sys.executable, str(ROOT / 'scripts/run_prol_worker.py'),
                           '--side', side, '--mode', mode, '--pass', kind, '--output', str(directory),
                           '--initial-state', str(Path(initial_state).resolve()), '--execute']})
    return result


def validate_record(record, job):
    from scripts.run_prol_worker import validate_saved_record
    validate_saved_record(record)
    if (record.get('wandb_logging_status') != 'uploaded' or record.get('mode') != job['mode']
            or record.get('side') != job['side'] or record.get('pass') != job['pass']):
        raise ValueError('PROL scope, completion or upload is inconsistent')
    tasks = record['task_results']
    if tasks[0]['seen_accuracy_pct'] < 90. or len(tasks) == 2 and tasks[1]['seen_accuracy_pct'] < 20.:
        raise ValueError('PROL abnormally low accuracy stops subsequent workers')
    return True


def pair_acceptance(output):
    from scripts.report_prol_validation import compare_pair
    paths = [Path(output) / name / 'result.json' for name in ('validation_author', 'validation_native')]
    reference, native = [json.loads(path.read_text()) for path in paths]
    receipt = compare_pair(reference, native, *paths, minimum_accuracy_pct=90.)
    write_json(Path(output) / 'validation_acceptance.json', receipt)
    return receipt


def require_normality_review(output):
    from scripts.report_prol_validation import validate_normality_review
    paths = [Path(output) / name / 'result.json' for name in ('validation_author', 'validation_native')]
    reference, native = [json.loads(path.read_text()) for path in paths]
    path = Path(output) / 'normality_review.json'
    if not path.is_file():
        raise ValueError('PROL pilots require saved old/new accuracy and loss-trend normality review')
    receipt = json.loads(path.read_text())
    validate_normality_review(receipt, reference, native, *paths)
    if receipt.get('status') != 'pass':
        raise ValueError('PROL normality review did not pass; pilot work stopped')
    return {'file': str(path.resolve()), 'sha256': shared.file_hash(path), 'status': 'pass'}


def execute(options):
    plan = shared.load_run_plan()
    configure_native_environment()
    profile = configure_environment()
    if any(profile[key] != value for key, value in [('gpu', '1'), ('wandb_entity', 'laymond1'), ('wandb_project', 'cl-efficiency')]):
        raise ValueError('PROL GPU/logging destination differs from authorization')
    if not options.initial_state.is_file():
        raise FileNotFoundError('Prepare pinned independent author initialization before execution')
    initial_sha, sources = shared.file_hash(options.initial_state), shared.execution_sources()
    shared.verify_reference_source()
    output = (options.resume or options.output).resolve()
    schedule = jobs(output, options.initial_state)
    if options.resume:
        saved = json.loads((output / 'execution.json').read_text())
        if (saved['plan'] != plan or saved['execution_source_sha256'] != sources
                or saved['initial_state_sha256'] != initial_sha or len(saved['jobs']) != 6
                or any(any(old[key] != new[key] for key in ('name', 'side', 'mode', 'pass', 'directory', 'command'))
                       for old, new in zip(saved['jobs'], schedule))):
            raise ValueError('Resume requires the same six jobs, code pins and initialization')
        if saved['status'] not in ('stage_completed', 'completed'):
            raise ValueError('Failed or incomplete PROL work is retained; no automatic retry')
    else:
        output.mkdir(parents=True, exist_ok=False)
        saved = {'status': 'running', 'plan': plan, 'jobs': schedule, 'output': str(output),
                 'initial_state_sha256': initial_sha, 'execution_source_sha256': sources,
                 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    process = None
    write_json(output / 'execution.json', saved)
    saved['slack_start_status'] = notify_slack('PROL 검증 시작: Base, batch64, GPU1, nominal LR0.01. 검증·성능 정상성 확인 후 효율성 측정.')
    try:
        if saved['slack_start_status'] != 'sent':
            raise RuntimeError('PROL Slack start notification failed; no worker started')
        for index, job in enumerate(saved['jobs']):
            if index > shared.JOB_NAMES.index(options.through):
                break
            if shared.execution_sources() != sources:
                raise ValueError('Frozen PROL experiment sources changed; next worker stopped')
            if index >= 2:
                if saved.get('validation_acceptance', {}).get('status') != 'pass':
                    raise ValueError('PROL pilots require passed independent author/native verification')
                saved['normality_review'] = require_normality_review(output)
            destination, path = Path(job['directory']), Path(job['directory']) / 'result.json'
            if path.is_file():
                if job.get('status') != 'completed' or job.get('result', {}).get('sha256') != shared.file_hash(path):
                    raise ValueError('Incomplete or altered PROL evidence retained; no retry')
                record = json.loads(path.read_text())
                validate_record(record, job)
            else:
                if destination.exists():
                    raise ValueError('Incomplete PROL worker directory retained; no overwrite or retry')
                job['gpu_occupancy_before_start'] = gpu_occupancy()
                if job['gpu_occupancy_before_start']['gpu_uuid'] != plan['gpu_uuid']:
                    raise ValueError('Physical GPU1 identity changed')
                job['status'] = 'running'
                write_json(output / 'execution.json', saved)
                logs = output / 'launcher_logs'
                logs.mkdir(exist_ok=True)
                print(f'[{index + 1}/6] PROL {job["name"]} starting', flush=True)
                with (logs / (job['name'] + '.log')).open('w') as log:
                    process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    if process.wait() != 0:
                        job['status'] = 'failed'
                        if path.is_file():
                            job['result'] = {'file': str(path), 'sha256': shared.file_hash(path)}
                        raise RuntimeError('PROL worker failed; saved evidence retained')
                process = None
                record = json.loads(path.read_text())
                validate_record(record, job)
            job.update(status='completed', result={'file': str(path), 'sha256': shared.file_hash(path),
                                                   'wandb_url': record['wandb_url']})
            if notify_slack('PROL 실행 완료: ' + job['name'] + '. ' + record['wandb_url']) != 'sent':
                raise RuntimeError('PROL completion notification failed; later workers stopped')
            if index == 1:
                receipt = pair_acceptance(output)
                saved['validation_acceptance'] = receipt
                if receipt['status'] != 'pass':
                    saved.update(status='validation_gate_' + receipt['status'], completed_gpu_runs=2)
                    saved['slack_completion_status'] = notify_slack('PROL 검증 ' + receipt['status'] + ': 후속 실행 중단, 원인 점검 필요.')
                    write_json(output / 'execution.json', saved)
                    return 1
            write_json(output / 'execution.json', saved)
        complete = sum(job['status'] == 'completed' for job in saved['jobs'])
        saved.update(status='completed' if complete == 6 else 'stage_completed', completed_gpu_runs=complete,
                     completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        saved['slack_completion_status'] = notify_slack('PROL 단계 완료: ' + str(complete) + '/6회. 로컬 검증·효율성 증거 보존.')
        write_json(output / 'execution.json', saved)
        if saved['slack_completion_status'] != 'sent':
            raise RuntimeError('PROL stage completion notification failed')
        print(json.dumps({'status': saved['status'], 'completed_gpu_runs': complete}))
        return 0
    except BaseException as error:
        stop_launcher(process)
        # Configuration/HTTP exceptions can embed private values. Save only
        # error types and job coordinates, never arbitrary exception strings.
        saved.update(status='failed', error_type=type(error).__name__,
                     completed_gpu_runs=sum(job['status'] == 'completed' for job in saved['jobs']))
        saved['slack_failure_status'] = notify_slack('PROL 실행 중단: ' + type(error).__name__ + '. 로컬 증거 보존, 자동 재실행 없음.')
        write_json(output / 'execution.json', saved)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--initial-state', type=Path, default=DEFAULT_INITIAL)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--through', choices=shared.JOB_NAMES, default='flops')
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if not options.execute:
        shared.load_run_plan()
        print(json.dumps({'execute': False, 'job_names': list(shared.JOB_NAMES), 'maximum_new_gpu_runs': 6,
                          'validation': shared.protocol(), 'efficiency': shared.protocol('efficiency'),
                          'normality_review_required_before_pilots': True}))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt('PROL suite termination requested')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
