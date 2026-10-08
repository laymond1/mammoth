"""Eight approved RainbowPrompt runs, with independent full/short acceptance gates."""

import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import rainbow_protocol as protocol
from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.run_peft_efficiency import gpu_occupancy, stop_launcher

DEFAULT_OUTPUT = ROOT / 'results/rainbow_validation_20261008'
DEFAULT_INITIAL = ROOT / 'results/rainbow_initial_20261008/initial_state.pt'


def jobs(output, initial_state=DEFAULT_INITIAL):
    result = []
    for name in protocol.JOB_NAMES:
        side = 'author' if name.endswith('_author') else 'native'
        mode = 'full_validation' if name.startswith('full_') else 'short_validation' if name.startswith('short_') else 'efficiency'
        kind = 'flops' if name == 'flops' else 'measurement' if name.startswith('measurement') else 'validation'
        directory = (Path(output) / name).resolve()
        result.append({'name': name, 'side': side, 'mode': mode, 'pass': kind, 'directory': str(directory),
                       'status': 'pending', 'command': [sys.executable, str(ROOT / 'scripts/run_rainbow_worker.py'),
                           '--side', side, '--mode', mode, '--pass', kind, '--output', str(directory),
                           '--initial-state', str(Path(initial_state).resolve()), '--execute']})
    return result


def validate_record(record, job):
    if (record.get('status') != 'completed' or record.get('wandb_logging_status') != 'uploaded'
            or record.get('mode') != job['mode'] or record.get('side') != job['side'] or record.get('pass') != job['pass']
            or record.get('protocol') != protocol.protocol(job['mode']) or record.get('precision') != protocol.PRECISION
            or any(record.get('structural_checks', {}).get(name) is not True for name in protocol.REQUIRED_CHECKS)):
        raise ValueError('RainbowPrompt worker scope, completion, upload or structural checks failed')
    settings = record['protocol']
    tasks = record.get('task_results', [])
    if not tasks or tasks[0].get('seen_accuracy_pct', -1) < 90:
        raise ValueError('RainbowPrompt first-task accuracy is abnormally low; later runs stopped')
    updates = settings['tasks'] * settings['n_epochs'] * 79
    for name, value in {'outer_observe_steps': updates, 'outer_training_stream_examples': updates // 79 * 5000,
                        'tasks_completed': settings['tasks'], 'evaluation_examples': 3000 if settings['tasks'] == 2 else 1000}.items():
        if type(record.get('metrics', {}).get(name)) is not int or record['metrics'][name] != value:
            raise ValueError('RainbowPrompt worker did not complete the full approved training/evaluation stream')
    return True


def pair_acceptance(output, prefix):
    from scripts.report_rainbow_validation import compare_pair
    reference_path = Path(output) / (prefix + '_author') / 'result.json'
    native_path = Path(output) / (prefix + '_native') / 'result.json'
    reference, native = (json.loads(path.read_text()) for path in (reference_path, native_path))
    receipt = compare_pair(reference, native, reference_path, native_path, minimum_accuracy_pct=90.)
    write_json(Path(output) / (prefix + '_acceptance.json'), receipt)
    return receipt


def execute(options):
    plan = protocol.load_run_plan()
    configure_native_environment()
    profile = configure_environment()
    if any(profile[key] != value for key, value in [('gpu', '1'), ('wandb_entity', 'laymond1'), ('wandb_project', 'cl-efficiency')]):
        raise ValueError('GPU/logging destination differs from authorization')
    if not options.initial_state.is_file():
        raise FileNotFoundError('Prepare the matched independent author initialization before GPU execution')
    initial_sha = protocol.file_hash(options.initial_state)
    sources = protocol.execution_sources()
    protocol.verify_reference_source()
    output = (options.resume or options.output).resolve()
    schedule = jobs(output, options.initial_state)
    if options.resume:
        saved = json.loads((output / 'execution.json').read_text())
        if (saved['plan'] != plan or saved['execution_source_sha256'] != sources
                or saved['initial_state_sha256'] != initial_sha or len(saved['jobs']) != 8
                or any(any(old[key] != new[key] for key in ('name', 'side', 'mode', 'pass', 'directory', 'command'))
                       for old, new in zip(saved['jobs'], schedule))):
            raise ValueError('Resume requires the same eight-job schedule, source pins and initialization')
    else:
        output.mkdir(parents=True, exist_ok=False)
        saved = {'status': 'running', 'plan': plan, 'jobs': schedule, 'output': str(output),
                 'initial_state_sha256': initial_sha, 'execution_source_sha256': sources,
                 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    process = None
    write_json(output / 'execution.json', saved)
    saved['slack_start_status'] = notify_slack('RainbowPrompt 검증 시작: Base, batch64, GPU1. 원형·단축 검증 통과 후 효율성 측정. learned gate 미포함.')
    try:
        if saved['slack_start_status'] != 'sent':
            raise RuntimeError('RainbowPrompt Slack start notification was not delivered; no GPU worker started')
        for index, job in enumerate(saved['jobs']):
            if index > protocol.JOB_NAMES.index(options.through):
                break
            if protocol.execution_sources() != sources:
                raise ValueError('Frozen RainbowPrompt experiment sources changed; no next worker started')
            if index >= 2 and saved.get('full_acceptance', {}).get('status') != 'pass':
                raise ValueError('Short verification requires passed full author/native acceptance')
            if index >= 4 and saved.get('short_acceptance', {}).get('status') != 'pass':
                raise ValueError('Efficiency measurements require passed short author/native acceptance')
            destination = Path(job['directory'])
            path = destination / 'result.json'
            if path.is_file():
                if job.get('status') != 'completed' or job.get('result', {}).get('sha256') != protocol.file_hash(path):
                    raise ValueError('Incomplete or altered worker evidence retained; no automatic retry')
                record = json.loads(path.read_text())
                validate_record(record, job)
            else:
                if destination.exists():
                    raise ValueError('Incomplete worker directory retained; no overwrite or automatic retry')
                job['gpu_occupancy_before_start'] = gpu_occupancy()
                if job['gpu_occupancy_before_start']['gpu_uuid'] != 'GPU-1fb8c05f-24e7-75d1-5cf9-0f47df8e46ad':
                    raise ValueError('Physical GPU1 identity changed')
                job['status'] = 'running'
                write_json(output / 'execution.json', saved)
                logs = output / 'launcher_logs'
                logs.mkdir(exist_ok=True)
                print(f'[{index + 1}/8] RainbowPrompt {job["name"]} starting', flush=True)
                with (logs / (job['name'] + '.log')).open('w') as log:
                    process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    if process.wait() != 0:
                        raise RuntimeError('RainbowPrompt worker failed; evidence retained for diagnosis')
                process = None
                record = json.loads(path.read_text())
                validate_record(record, job)
            job.update(status='completed', result={'file': str(path), 'sha256': protocol.file_hash(path),
                                                   'wandb_url': record['wandb_url']})
            notify_slack('RainbowPrompt 실행 완료: ' + job['name'] + '. ' + record['wandb_url'])
            if index in (1, 3):
                prefix = 'full' if index == 1 else 'short'
                receipt = pair_acceptance(output, prefix)
                saved[prefix + '_acceptance'] = receipt
                if receipt['status'] != 'pass':
                    saved['status'] = prefix + '_gate_' + receipt['status']
                    saved['slack_completion_status'] = notify_slack('RainbowPrompt ' + prefix + ' 검증 ' + receipt['status'] + ': 후속 실행 중단, 원인 점검 필요.')
                    write_json(output / 'execution.json', saved)
                    return 1
            write_json(output / 'execution.json', saved)
        complete = sum(job['status'] == 'completed' for job in saved['jobs'])
        saved.update(status='completed' if complete == 8 else 'stage_completed', completed_gpu_runs=complete,
                     completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        saved['slack_completion_status'] = notify_slack('RainbowPrompt 단계 완료: ' + str(complete) + '/8회. 구현 검증·성능 증거 점검 후 승인된 다음 단계 진행.')
        write_json(output / 'execution.json', saved)
        print(json.dumps({'status': saved['status'], 'completed_gpu_runs': complete}))
        return 0
    except BaseException as error:
        stop_launcher(process)
        saved.update(status='failed', error_type=type(error).__name__, error=str(error))
        saved['slack_failure_status'] = notify_slack('RainbowPrompt 실행 중단: ' + type(error).__name__ + '. 로컬 증거 보존, 자동 재실행 없음.')
        write_json(output / 'execution.json', saved)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--initial-state', type=Path, default=DEFAULT_INITIAL)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--through', choices=protocol.JOB_NAMES, default='flops',
                        help='Stop at an approved job boundary to inspect saved validation evidence')
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if not options.execute:
        print(json.dumps({'execute': False, 'job_names': list(protocol.JOB_NAMES), 'maximum_new_gpu_runs': 8,
                          'full': protocol.protocol('full_validation'), 'short': protocol.protocol('short_validation')}))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt('RainbowPrompt suite termination requested')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
