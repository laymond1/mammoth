"""Launch the approved single inference-only PROL diagnostic process."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prol_diagnostic_protocol as shared
from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.hide_protocol import validate_shared_data
from scripts.run_peft_efficiency import gpu_occupancy, stop_launcher


def build_spec(output, profile):
    shared.load_plan()
    if any(profile.get(key) != value for key, value in {'gpu': '1', 'wandb_entity': 'laymond1',
            'wandb_project': 'cl-efficiency', 'data_root': '/mnt/sdab1/share/data',
            'cache_root': '/mnt/sdab1/userHome/wslim/.cache'}.items()):
        raise ValueError('PROL diagnostic runtime differs from the shared selected profile')
    prior = shared.verify_prior_inputs()
    if shared.file_hash(shared.CHECKPOINT) != shared.CHECKPOINT_SHA256:
        raise ValueError('PROL diagnostic requires the common pinned local NPZ')
    source_pins = shared.verify_reference_source()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    spec = {'schema_version': 1, 'name': 'prol-checkpoint-diagnostic-' + stamp,
            'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic', 'source_mode': 'official-active-path',
            'paper_equivalent': False, 'training_authorized': False, 'diagnostic_authorized': True,
            'protocol': shared.protocol(), 'gpu': '1', 'gpu_uuid': shared.GPU_UUID, 'wandb_mode': 'online',
            'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
            'checkpoint_file': shared.CHECKPOINT, 'checkpoint_sha256': shared.CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
            'reference_commit': shared.REFERENCE_COMMIT, 'reference_source': str(shared.REFERENCE_SOURCE),
            'reference_source_sha256': source_pins, 'origin_execution_commit': shared.ORIGIN_EXECUTION_COMMIT,
            'original_execution_source_sha256': prior['execution_source_sha256'],
            'original_wandb_url': prior['wandb_url'], 'prior_result': str(shared.PRIOR_DIRECTORY / 'result.json'),
            'prior_result_sha256': shared.PRIOR_PINS['result.json'],
            'prior_predictions': str(shared.PRIOR_DIRECTORY / 'task1_predictions.npz'),
            'prior_predictions_sha256': shared.PRIOR_PINS['task1_predictions.npz'],
            'prior_checkpoint': str(shared.PRIOR_DIRECTORY / 'task1_state.pt'),
            'prior_checkpoint_sha256': shared.PRIOR_PINS['task1_state.pt'],
            'data_root': profile['data_root'], 'data_files': validate_shared_data(profile['data_root']),
            'execution_source_sha256': shared.execution_sources(), 'output': str(Path(output).resolve()),
            'measurement_aggregate_eligible': False,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    shared.validate_spec(spec)
    return spec


def execute(options):
    plan = shared.load_plan()
    configure_native_environment()
    spec = build_spec(options.output, configure_environment())
    occupancy = gpu_occupancy()
    if occupancy['gpu_uuid'] != shared.GPU_UUID:
        raise ValueError('Physical GPU1 identity differs from the approved device')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    saved = {'status': 'prepared', 'plan': plan, 'spec_sha256': shared.file_hash(output / 'spec.json'),
             'execution_source_sha256': spec['execution_source_sha256'], 'gpu_occupancy': occupancy,
             'new_gpu_processes': 0, 'new_training': False,
             'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    process = None
    started = time.monotonic()
    try:
        saved['slack_start_status'] = notify_slack('PROL checkpoint 추론 진단 시작: GPU1, 4조건×10회, 동일40샘플. 학습·판정 기준 변경 없음.')
        write_json(output / 'execution.json', saved)
        if saved['slack_start_status'] != 'sent':
            raise RuntimeError('Required diagnostic start notification failed')
        if shared.execution_sources() != spec['execution_source_sha256']:
            raise ValueError('Diagnostic code changed before launch')
        env = os.environ.copy()
        env.pop('MAMMOTH_DEVICE', None)
        env.update(CUDA_VISIBLE_DEVICES='1', CL_PROL_DIAGNOSTIC_EXECUTE='1', WANDB_MODE='online',
                   WANDB_DIR=str(output), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        command = [sys.executable, str(ROOT / 'scripts/diagnose_prol_checkpoint.py'), '--worker', str(output / 'spec.json')]
        saved.update(status='running', command=command, new_gpu_processes=1)
        write_json(output / 'execution.json', saved)
        with (output / 'console.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if time.monotonic() - started >= options.timeout:
                    stop_launcher(process)
                    raise TimeoutError('Diagnostic timeout; no retry')
                time.sleep(1)
        record_path = output / 'result.json'
        if record_path.is_file():
            saved['result'] = {'file': str(record_path), 'sha256': shared.file_hash(record_path)}
        if process.returncode != 0:
            raise RuntimeError('Diagnostic failed; preserve partial evidence and stop')
        record = json.loads(record_path.read_text())
        from scripts.diagnose_prol_checkpoint import validate_saved_record
        validate_saved_record(record)
        if record.get('wandb_logging_status') != 'uploaded':
            raise ValueError('Diagnostic small evidence was not uploaded')
        saved.update(status='completed', launcher_wall_s=time.monotonic() - started,
                     wandb_url=record['wandb_url'], completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        saved['slack_completion_status'] = notify_slack('PROL 추론 진단 40평가 완료. 재학습·후속 실험 없이 원자료 검토 및 보고서 게시. ' + record['wandb_url'])
        write_json(output / 'execution.json', saved)
        if saved['slack_completion_status'] != 'sent':
            raise RuntimeError('Required diagnostic completion notification failed')
        print(json.dumps({'status': 'completed', 'model_evaluations': 40, 'new_training': False,
                          'wandb_url': record['wandb_url']}))
        return 0
    except BaseException as error:
        stop_launcher(process)
        saved.update(status='failed', error_type=type(error).__name__, launcher_wall_s=time.monotonic() - started)
        record_path = output / 'result.json'
        if record_path.is_file():
            saved['result'] = {'file': str(record_path), 'sha256': shared.file_hash(record_path)}
        saved['slack_failure_status'] = notify_slack('PROL 추론 진단 중단: ' + type(error).__name__ + '. 증거 보존, 자동 재실행 없음.')
        write_json(output / 'execution.json', saved)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=shared.DEFAULT_OUTPUT)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if not options.execute:
        shared.load_plan()
        print(json.dumps({'execute': False, 'maximum_new_gpu_processes': 1, 'new_training': False,
                          'protocol': shared.protocol()}))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Diagnostic termination requested')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
