"""Execute the approved Small then Tiny chain; every dependency fails closed.

This stage-specific coordinator does not modify training sources or settings.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UV = ['uv', 'run', '--no-sync', '--offline', 'python']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def emit(value):
    print(json.dumps(value), flush=True)


def state(size, field, value, output=None):
    path = PUB / 'execution_plan.json'
    plan = read(path)
    plan['status'] = size + '_' + field + '_' + value
    plan['variants'][size][field] = value
    if output is not None:
        plan['variants'][size][field + '_output'] = str(output)
    write(path, plan)
    emit({'variant': size, 'stage': field, 'status': value, 'output': str(output) if output else None})


def idle_gpu():
    for _ in range(30):
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        busy = any('GPU-1fb8c05f-24e7-75d1-5cf9-0f47df8e46ad' in line for line in result.stdout.splitlines())
        if not busy:
            return
        time.sleep(1)
    raise RuntimeError('GPU1 is occupied; no additional GPU job was launched')


def progress(output):
    path = output / 'epoch_results.json'
    if path.is_file():
        try:
            rows = read(path)
            if rows:
                return rows[-1]
        except json.JSONDecodeError:
            pass
    return None


def run(stage, arguments, output=None, gpu=False):
    if gpu:
        idle_gpu()
    log = PUB / 'prepared' / (stage + '.log')
    record_path = PUB / 'environment/command_log.json'
    record = read(record_path)
    row = {'stage': stage, 'argv': UV + arguments,
           'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'log': str(log)}
    record['commands'].append(row)
    write(record_path, record)
    with log.open('w') as stream:
        process = subprocess.Popen(UV + arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        last = 0
        try:
            while process.poll() is None:
                if time.monotonic() - last >= 30:
                    emit({'stage': stage, 'status': 'running', 'pid': process.pid,
                          'latest_epoch': progress(output) if output else None})
                    last = time.monotonic()
                time.sleep(1)
        except BaseException:
            process.send_signal(2)
            process.wait(timeout=120)
            raise
    record = read(record_path)
    record['commands'][-1].update(exit_code=process.returncode,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(record_path, record)
    if process.returncode:
        raise RuntimeError(stage + ' failed; original log and partial outputs were preserved')
    emit({'stage': stage, 'status': 'completed'})


def verify(stage, output):
    run(stage + '_readback', ['scripts/verify_hide_results.py', str(output / 'result.json'),
                             '--output', str(PUB / 'prepared/readbacks' / stage), '--execute'])


def validation_gate(size, reference, native, initial):
    from scripts.hide_protocol import validate_efficiency_acceptance
    from scripts.report_hide_results import exact_validation
    import numpy as np
    expected, observed = read(reference / 'result.json'), read(native / 'result.json')
    validate_efficiency_acceptance(native / 'result.json', initial_state=initial, variant=size)
    if exact_validation(expected, observed)['report_gate'] != 'pass':
        raise ValueError('Exact accuracy, structural or actual first-batch gate failed')
    for output, record in ((reference, expected), (native, observed)):
        for task in (1, 2):
            with np.load(output / ('task%d_predictions.npz' % task), allow_pickle=False) as arrays:
                labels, predictions = arrays['labels'], arrays['predictions'][:, 0]
                row = record['task_results'][task - 1]
                for partition in ('seen', 'old', 'new'):
                    mask = np.ones(len(labels), dtype=bool) if partition == 'seen' else labels < (task - 1) * 10 if partition == 'old' else labels >= (task - 1) * 10
                    if row[partition + '_samples'] != int(mask.sum()) or row[partition + '_correct'] != int((labels[mask] == predictions[mask]).sum()):
                        raise ValueError('Saved validation counts disagree with actual prediction arrays')
    emit({'variant': size, 'validation_gate': 'pass', 'task2_accuracy_pct': observed['task_results'][-1]['seen_accuracy_pct']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-reference', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    from scripts.hide_size_protocol import load_plan
    load_plan()
    if not options.execute:
        emit({'prepared': True, 'executed': False, 'variant_order': ['small', 'tiny'], 'full_cl': False})
        return
    os.environ.update(UV_PROJECT_ENVIRONMENT=str(ROOT.parent.parent / '.venv'), UV_CACHE_DIR='/tmp/hide-sizes-uvcache',
                      HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      MPLCONFIGDIR='/tmp/hide-sizes-matplotlib')
    # ROOT is nested under the foundation checkout's .worktrees directory.
    for size in ('small', 'tiny'):
        initial = PUB / ('prepared/' + size + '_author_initial_state.pt')
        if size == 'small':
            reference = options.small_reference.resolve()
            start = time.monotonic()
            while not (reference / 'result.json').is_file():
                if time.monotonic() - start > 10800:
                    raise TimeoutError('Existing Small reference did not finish')
                emit({'stage': 'small_reference', 'status': 'running', 'latest_epoch': progress(reference)})
                time.sleep(30)
        else:
            state(size, 'reference', 'running')
            before = set((PUB / 'runs').glob('hide-author-tiny-validation-*'))
            run('tiny_reference', ['scripts/run_hide_reference.py', '--variant', size, '--mode', 'validation',
                                  '--initial-state', str(initial), '--output', str(PUB / 'runs'), '--execute'], gpu=True)
            created = set((PUB / 'runs').glob('hide-author-tiny-validation-*')) - before
            if len(created) != 1:
                raise ValueError('Tiny reference output is ambiguous')
            reference = created.pop()
        if read(reference / 'result.json')['status'] != 'completed':
            raise ValueError(size + ' author reference failed')
        verify(size + '_reference', reference)
        state(size, 'reference', 'completed', reference)
        native = PUB / ('runs/hide-native-' + size + '-validation')
        state(size, 'native_validation', 'running', native)
        run(size + '_native', ['scripts/run_hide_validation.py', '--execute', '--variant', size, '--mode', 'validation',
                              '--initial-state', str(initial), '--reference-summary', str(reference / 'result.json'),
                              '--result-dir', str(native)], output=native, gpu=True)
        verify(size + '_native', native)
        validation_gate(size, reference, native, initial)
        state(size, 'native_validation', 'pass', native)
        pilot = PUB / ('runs/hide-efficiency-' + size)
        state(size, 'efficiency', 'running', pilot)
        run(size + '_efficiency', ['scripts/run_hide_efficiency.py', '--execute', '--variant', size,
                                  '--acceptance-summary', str(native / 'result.json'), '--initial-state', str(initial),
                                  '--result-dir', str(pilot)], output=pilot, gpu=True)
        if read(pilot / 'execution.json')['status'] != 'completed':
            raise ValueError(size + ' pilot is incomplete')
        for worker in ('measurement1', 'measurement2', 'measurement3', 'flops'):
            verify(size + '_' + worker, pilot / worker)
        state(size, 'efficiency', 'completed', pilot)
    path = PUB / 'execution_plan.json'
    plan = read(path)
    plan['status'] = 'experiments_completed_pending_report'
    write(path, plan)
    emit({'status': plan['status']})


if __name__ == '__main__':
    main()
