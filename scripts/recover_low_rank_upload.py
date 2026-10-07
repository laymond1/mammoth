"""Recover a completed native LoRA upload on its existing W&B run.

Default is a read-only, stdlib-only evidence check. --execute resumes the saved
run with resume='must', uploads 14 small evidence files, verifies remote finished
counts/artifact hashes, and sends the authorized recovery notification. Saved
training measurements/source pins remain unchanged. No Torch, GPU or training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import flatten, notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.low_rank_protocol import REQUIRED_STRUCTURAL_CHECKS, evaluate_acceptance
from scripts.publish_low_rank_results import exact_remote_count, manifest_matches, verify_run
from scripts.report_low_rank_results import (
    COUNT_KEYS, artifact_inputs, file_hash, read_json, record_body, run_path, validate_pair,
)

EVIDENCE_FILES = ('spec.json', 'result.json', 'metrics.json', 'acceptance.json', 'effective_args.json',
                  'shared_data.json', 'epoch_results.json', 'task_results.json', 'state_audits.json',
                  'first_batch_audits.json', 'phases.json', 'telemetry.csv',
                  'task1_predictions.npz', 'task2_predictions.npz')
ORIGINAL_FILES = ('result.json', 'metrics.json', 'completion.json')
RECOVERABLE_FIELDS = ('status', 'wandb_logging_status', 'upload_recovery')


def json_value(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(result_dir):
    """Use the original saved specification, never today's changed source files."""
    directory = Path(result_dir).resolve()
    record, spec = (read_json(directory / name) for name in ('result.json', 'spec.json'))
    if (record.get('implementation') != 'mammoth_native' or record.get('status') != 'logging_failed'
            or record.get('computation_status') != 'completed'
            or not str(record.get('wandb_logging_status', '')).startswith(('failed:', 'finish_failed:'))
            or record.get('acceptance_status') != 'pass'):
        raise ValueError('Recovery requires failed logging after completed, passing native computation')
    if any(record.get('structural_checks', {}).get(key) is not True for key in REQUIRED_STRUCTURAL_CHECKS):
        raise ValueError('Recovery requires all 14 verified structural checks')
    if any(record.get(key) != value for key, value in spec.items()):
        raise ValueError('Saved native record differs from its original specification pins')
    source = Path(spec.get('reference_summary', ''))
    if not source.is_absolute() or not source.is_file() or file_hash(source) != spec.get('reference_summary_sha256'):
        raise ValueError('Saved author reference summary is missing or changed')
    reference = record_body(read_json(source))
    candidate = deepcopy(record)
    candidate.update(status='completed', wandb_logging_status='uploaded')
    pair = validate_pair(record['method'], reference, candidate)
    acceptance = evaluate_acceptance(record['task_results'], reference['task_results'], structural_checks=record['structural_checks'])
    if pair['publication_gate'] != 'pass' or record.get('acceptance') != acceptance:
        raise ValueError('Saved exact-count acceptance or first-batch comparison does not pass')
    artifacts = artifact_inputs(record, directory / 'result.json')
    completion = read_json(directory / 'completion.json')
    if (completion.get('status') != 'logging_failed' or completion.get('acceptance_status') != 'pass'
            or completion.get('timeout') is not False or completion.get('wandb_url') != record['wandb_url']):
        raise ValueError('Saved worker completion does not confirm passing computation and failed logging')
    if json_value(directory / 'metrics.json') != record:
        raise ValueError('Saved failed metrics and result records disagree')
    expected = {'acceptance.json': record['acceptance'], 'epoch_results.json': record['epoch_results'],
                'task_results.json': record['task_results'],
                'state_audits.json': record['state_audits'], 'first_batch_audits.json': record['first_batch_audits']}
    pins = {}
    for name in (*EVIDENCE_FILES, 'completion.json'):
        path = directory / name
        if name in ('state_audits.json', 'first_batch_audits.json') and not path.exists():
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Recovery evidence is missing or empty')
        if name in expected and json_value(path) != expected[name]:
            raise ValueError('Saved evidence differs from the completed computation record')
        pins[name] = {'file': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
    path = run_path(record['wandb_url'])
    return {'directory': directory, 'record': record, 'completion': completion, 'pins': pins,
            'artifacts': artifacts, 'run_path': path, 'run_id': path.rsplit('/', 1)[1],
            'reference_file': str(source), 'reference_sha256': file_hash(source)}


def summary_values(record):
    values = {key: value for key, value in flatten({'native_counts': record['counts'],
        'native_acceptance': record['acceptance']}).items() if isinstance(value, (int, float, bool))}
    values.update(native_status='completed', acceptance_status='pass', efficiency_comparison=False)
    for row in record['task_results']:
        values.update({f'task{row["task"]}/{key}': row[key] for key in COUNT_KEYS})
    return values


def check_pins(pins):
    for pin in pins.values():
        path = Path(pin['file'])
        if not path.is_file() or path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
            raise ValueError('Original computation evidence changed during recovery')


def verify_recovery(api, prepared, candidate, artifact_name, files, output, timeout):
    deadline = time.monotonic() + timeout
    while True:
        api.flush()
        run, artifact = api.run(prepared['run_path']), api.artifact(artifact_name)
        if run.state == 'finished' and artifact.state == 'COMMITTED':
            expected = summary_values(candidate)
            summary = dict(run.summary)
            for name, value in expected.items():
                actual = summary.get(name)
                if (not exact_remote_count(actual, value) if isinstance(value, (int, float)) and not isinstance(value, bool)
                        else type(actual) is not type(value) or actual != value):
                    raise ValueError('Recovery remote summary differs from exact completed counts/gates')
            all_files = manifest_matches(artifact, files)
            proof = verify_run(api, candidate['method'], 'native', candidate,
                               {'artifacts': prepared['artifacts']}, output)
            if proof['artifact'] != artifact.qualified_name:
                raise ValueError('Recovery evidence artifact is ambiguous')
            return {**proof, 'status': 'pass', 'all_evidence_files': all_files,
                    'summary_verified': expected, 'source_run_preserved': True,
                    'new_training': False, 'new_gpu_computation': False}
        if time.monotonic() >= deadline:
            raise TimeoutError('Recovered run/artifact did not reach finished/COMMITTED')
        time.sleep(min(5., max(0., deadline - time.monotonic())))


def recover(options):
    prepared = prepare(options.result_dir)
    profile = configure_environment(credentials_path=options.credentials)
    if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('Recovery requires laymond1/cl-efficiency')
    import wandb
    directory = prepared['directory']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = directory / 'upload_recovery' / stamp
    original, evidence = output / 'original', output / 'evidence'
    original.mkdir(parents=True, exist_ok=False)
    evidence.mkdir()
    check_pins(prepared['pins'])
    originals = {}
    for name in ORIGINAL_FILES:
        destination = original / name
        shutil.copyfile(directory / name, destination)
        originals[name] = {'file': str(destination), 'sha256': file_hash(destination), 'bytes': destination.stat().st_size}
        if originals[name]['sha256'] != prepared['pins'][name]['sha256']:
            raise ValueError('Failed completion snapshot changed while copying')
    recovery = {'kind': 'upload_only', 'started_at': stamp, 'run_id': prepared['run_id'],
                'source_run_preserved': True, 'new_training': False, 'new_gpu_computation': False,
                'original_status': prepared['record']['status'],
                'original_wandb_logging_status': prepared['record']['wandb_logging_status'],
                'original_files': originals, 'script_sha256': file_hash(Path(__file__)),
                'remote_upload_receipt': str(output / 'readback.json')}
    candidate = deepcopy(prepared['record'])
    candidate.update(status='completed', wandb_logging_status='uploaded', upload_recovery=recovery)
    files = {}
    generated = {'result.json': candidate, 'metrics.json': candidate,
                 'state_audits.json': candidate['state_audits'], 'first_batch_audits.json': candidate['first_batch_audits']}
    for name in EVIDENCE_FILES:
        destination = evidence / name
        if name in generated:
            write_json(destination, generated[name])
        else:
            shutil.copyfile(directory / name, destination)
            if file_hash(destination) != prepared['pins'][name]['sha256']:
                raise ValueError('Evidence changed while creating upload snapshot')
        files[name] = {'file': str(destination), 'sha256': file_hash(destination), 'bytes': destination.stat().st_size}
    receipt = {'status': 'failed', **recovery, 'wandb_url': candidate['wandb_url'],
               'evidence_file_count': len(files), 'full_checkpoints_uploaded': False,
               'remote_readback_status': 'pending', 'slack_completion_status': 'not_sent'}
    write_json(output / 'result.json', receipt)
    try:
        run = wandb.init(entity='laymond1', project='cl-efficiency', id=prepared['run_id'], resume='must',
                         dir=str(output), mode='online')
        if run.id != prepared['run_id'] or run_path(run.url) != prepared['run_path']:
            raise ValueError('W&B recovery must preserve the exact original run ID')
        run.summary.update(summary_values(candidate))
        metadata = {'upload_recovery': True, 'new_training': False, 'new_gpu_computation': False,
                    'local_checkpoint_sha256': {f'task{task}': pin['sha256']
                                               for task, pin in prepared['artifacts']['local_checkpoints'].items()},
                    'original_record_sha256': prepared['pins']['result.json']['sha256']}
        artifact = wandb.Artifact(candidate['name'], type='cl-native-acceptance', metadata=metadata)
        for name, pin in files.items():
            artifact.add_file(pin['file'], name=name)
        logged = run.log_artifact(artifact)
        logged.wait()
        if logged.state != 'COMMITTED':
            raise ValueError('Recovery evidence artifact did not reach COMMITTED')
        receipt['wandb_artifact'] = {'name': logged.qualified_name, 'digest': logged.digest}
        wandb.finish(exit_code=0)
        api = wandb.Api(timeout=60)
        proof = verify_recovery(api, prepared, candidate, logged.qualified_name, files,
                                output / 'remote_record', options.readback_timeout)
        write_json(output / 'readback.json', proof)
        receipt.update(status='completed', remote_readback_status='pass', wandb_logging_status='uploaded')
        check_pins(prepared['pins'])
        recovery.update(remote_readback_sha256=file_hash(output / 'readback.json'),
                        artifact=receipt['wandb_artifact'], remote_readback_status='pass')
        candidate['upload_recovery'] = recovery
        # Only upload status/recovery provenance change; numerical/source/resource data stay exact.
        if any(candidate.get(key) != value for key, value in prepared['record'].items() if key not in RECOVERABLE_FIELDS):
            raise ValueError('Upload recovery changed original computation evidence')
        write_json(directory / 'state_audits.json', candidate['state_audits'])
        write_json(directory / 'first_batch_audits.json', candidate['first_batch_audits'])
        write_json(directory / 'result.json', candidate)
        write_json(directory / 'metrics.json', candidate)
        completion = {**prepared['completion'], 'status': 'completed', 'exit_code': 0, 'upload_recovery': recovery}
        write_json(directory / 'completion.json', completion)
        write_json(output / 'result.json', receipt)
    except Exception as error:
        receipt.update(status='upload_recovery_failed', error_type=type(error).__name__)
        if getattr(wandb, 'run', None) is not None:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
        write_json(output / 'result.json', receipt)
        print(json.dumps({'status': receipt['status'], 'error_type': receipt['error_type'], 'receipt': str(output / 'result.json')}))
        return 1
    message = (f'{candidate["method"]} 기존 W&B 실행의 결과 업로드 복구 완료: 14개 증거 파일·정확한 정답 수·finished 상태 확인. '
               f'재학습·GPU 계산 없이 기존 검증 측정치를 보존했습니다. W&B {candidate["wandb_url"]}')
    sent = notify_slack(message)
    receipt['slack_completion_status'] = sent
    if sent != 'sent':
        receipt['status'] = 'notification_failed'
    write_json(output / 'notification.json', {'status': sent, 'message': message, 'readback_sha256': file_hash(output / 'readback.json')})
    write_json(output / 'result.json', receipt)
    print(json.dumps({'status': receipt['status'], 'wandb_url': candidate['wandb_url'], 'receipt': str(output / 'result.json')}))
    return 0 if receipt['status'] == 'completed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', required=True, type=Path)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--readback-timeout', type=float, default=120.)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if not 0 < options.readback_timeout <= 600:
        raise ValueError('Recovery readback timeout must be positive and bounded')
    if not options.execute:
        prepared = prepare(options.result_dir)
        print(json.dumps({'execute': False, 'credentials_read': False, 'external_actions': False,
                          'new_training': False, 'new_gpu_computation': False, 'run_id': prepared['run_id'],
                          'evidence_file_count': len(EVIDENCE_FILES), 'acceptance_status': 'pass'}))
        return 0
    return recover(options)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'recovery_preflight_failed', 'error_type': type(error).__name__}))
        raise SystemExit(1)
