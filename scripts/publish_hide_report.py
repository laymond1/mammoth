"""Publish a completed HiDe report; no new training, GPU work or measurements.

The default validates local report files without credentials, network or writes.
--execute uploads immutable snapshots and finishes a report-only W&B run.
Completion Slack is a separate --execute --notify-after-readback operation and
requires the existing verifier's successful finished/COMMITTED/hash receipt.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment

REPORT_FILES = ('results.json', 'results.csv', 'report.md',
                'efficiency_comparison.png', 'efficiency_comparison.pdf')
ACCURACY_METRICS = {'task1_seen', 'task2_seen', 'task2_old', 'task2_new'}
RESOURCE_KEYS = ('process_wall_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib',
                 'peak_process_tree_rss_mib', 'process_tree_cpu_s')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Saved report or receipt must be a JSON object')
    return value


def prepare(report_dir):
    """Fail closed on missing outputs or changed provenance, before authentication."""
    report_dir = Path(report_dir).resolve()
    files = {}
    for filename in REPORT_FILES:
        path = report_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty report artifact: {filename}')
        files[filename] = file_hash(path)
    report = read_json(report_dir / 'results.json')
    validation = report.get('validation', {})
    status = validation.get('acceptance', {}).get('status')
    rows = validation.get('accuracy_metrics', [])
    if (report.get('status') != 'completed' or report.get('model') != 'hide-prompt'
            or status not in ('pass', 'review', 'fail') or len(rows) != 4
            or {row.get('metric') for row in rows} != ACCURACY_METRICS
            or any(row.get('status') not in ('pass', 'review', 'fail') for row in rows)):
        raise ValueError('Report completion and exact four-metric validation are inconsistent')
    efficiency = report.get('efficiency', {})
    methods = efficiency.get('methods', [])
    if efficiency.get('status') not in ('completed', 'not_executed'):
        raise ValueError('Efficiency reporting must be completed or explicitly not executed')
    if efficiency['status'] == 'completed':
        if status != 'pass' or validation.get('report_gate') != 'pass' or validation.get('first_batch_comparison', {}).get('status') != 'pass':
            raise ValueError('Published efficiency requires passing accuracy, structure and first-batch gates')
        if (not methods or sum(method.get('model') == 'hide-prompt' for method in methods) != 1
                or len({method.get('model') for method in methods}) != len(methods)
                or any(method.get('measurement_count') != 3 for method in methods)):
            raise ValueError('Efficiency methods must contain three accepted measurements and one HiDe entry')
        for method in methods:
            estimate = method.get('flops', {}).get('supported_ops_total')
            if isinstance(estimate, bool) or not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate <= 0:
                raise ValueError('Completed efficiency must include the separate partial FLOPs diagnostic')
    elif methods:
        raise ValueError('Unexecuted efficiency cannot contain measured method entries')
    sources = report.get('sources', {})
    if not {'reference', 'native'}.issubset(sources):
        raise ValueError('Report requires reference/native provenance')
    for source in sources.values():
        path = Path(source.get('file', ''))
        pin = source.get('sha256', '')
        if not path.is_absolute() or not path.is_file() or not re.fullmatch(r'[0-9a-f]{64}', pin) or file_hash(path) != pin:
            raise ValueError('Report provenance is missing or changed')
    # Recompute from the actual pinned records; status strings cannot authorize
    # a report whose published counts or distributions were altered afterwards.
    from scripts.report_hide_results import build_report
    reference = read_json(sources['reference']['file'])
    reference = reference.get('official_reference', reference)
    native = read_json(sources['native']['file'])
    efficiency_source = sources.get('efficiency_execution_companion', sources.get('efficiency'))
    historical_source = sources.get('historical')
    expected = build_report(reference, native,
                            efficiency_source['file'] if efficiency_source is not None else None,
                            historical_source['file'] if historical_source is not None else None)
    if report['validation'] != expected['validation'] or report['efficiency'] != expected['efficiency']:
        raise ValueError('Report counts, acceptance or efficiency distributions disagree with pinned inputs')
    return report, files


def scalar_summary(report):
    result = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'validation_acceptance': report['validation']['acceptance']['status'],
              'implementation_gate': report['validation'].get('report_gate', 'blocked'),
              'first_batch_match': report['validation'].get('first_batch_comparison', {}).get('status', 'unavailable'),
              'efficiency_status': report['efficiency']['status'],
              'equal_compute_budget': False, 'seed_variation_estimated': False}
    for row in report['validation']['accuracy_metrics']:
        for key in ('reference_accuracy_pct', 'native_accuracy_pct', 'reference_minus_native_pp'):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                result[f'validation/{row["metric"]}/{key}'] = value
    for method in report['efficiency']['methods']:
        name = method['model']
        for metric in RESOURCE_KEYS:
            for statistic in ('mean', 'std_sample', 'median', 'min', 'max'):
                value = method.get('resources', {}).get(metric, {}).get(statistic)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    result[f'efficiency/{name}/{metric}/{statistic}'] = value
        for statistic in ('mean', 'std_sample'):
            value = method.get('accuracy_pct', {}).get(statistic)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                result[f'efficiency/{name}/accuracy_pct/{statistic}'] = value
        result[f'efficiency/{name}/partial_supported_ops_total'] = method['flops']['supported_ops_total']
    return result


def publish(options):
    report, pins = prepare(options.report_dir)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'snapshots'
    snapshot.mkdir()
    for filename, pin in pins.items():
        shutil.copyfile(options.report_dir / filename, snapshot / filename)
        if file_hash(snapshot / filename) != pin:
            raise ValueError('Report changed while taking the immutable upload snapshot')
    name = 'hide-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    metadata = {'schema_version': 1, 'name': name, 'model': 'hide-prompt',
                'report_only': True, 'new_training': False, 'new_gpu_computation': False,
                'report_dir': str(options.report_dir.resolve()), 'report_files_sha256': pins,
                'validation_acceptance': report['validation']['acceptance']['status'],
                'implementation_gate': report['validation'].get('report_gate', 'blocked'),
                'first_batch_match': report['validation'].get('first_batch_comparison', {}).get('status', 'unavailable'),
                'efficiency_status': report['efficiency']['status'],
                'provenance': report['sources'], 'summary': scalar_summary(report)}
    write_json(output / 'artifact_metadata.json', metadata)
    artifact_files = {filename: {'local_file': f'snapshots/{filename}', 'sha256': pin} for filename, pin in pins.items()}
    artifact_files['artifact_metadata.json'] = {'local_file': 'artifact_metadata.json',
                                               'sha256': file_hash(output / 'artifact_metadata.json')}
    write_json(output / 'artifact_files.json', artifact_files)
    record = {**metadata, 'status': 'failed', 'wandb_logging_status': 'not_started',
              'readback_status': 'pending', 'slack_completion_status': 'deferred_until_readback'}
    write_json(output / 'result.json', record)
    wandb = None
    try:
        profile = configure_environment()
        if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
            raise ValueError('HiDe report publication requires the configured laymond1/cl-efficiency project')
        import wandb
        run = wandb.init(entity='laymond1', project='cl-efficiency', name=name,
                         group='hide-base-results', job_type='report-only',
                         tags=['report-only', 'hide-prompt', 'no-new-training'], config=metadata,
                         dir=str(output), mode='online')
        record['wandb_url'] = run.url
        run.summary.update(metadata['summary'])
        artifact = wandb.Artifact(name, type='cl-hide-report', metadata={
            'report_only': True, 'validation_acceptance': metadata['validation_acceptance'],
            'efficiency_status': metadata['efficiency_status']})
        for filename, pin in artifact_files.items():
            artifact.add_file(str(output / pin['local_file']), name=filename)
        artifact.add_file(str(output / 'artifact_files.json'), name='artifact_files.json')
        logged = run.log_artifact(artifact)
        logged.wait()
        if logged.state != 'COMMITTED':
            raise ValueError('Report artifact did not reach COMMITTED')
        record['wandb_artifact'] = {'name': logged.qualified_name, 'state': logged.state}
        record['wandb_logging_status'] = 'uploaded'
        wandb.finish(exit_code=0)
        record['status'] = 'completed'
    except Exception as error:
        record['error_type'] = type(error).__name__
        record['status'] = 'logging_failed'
        record['wandb_logging_status'] = 'failed:' + type(error).__name__
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
    write_json(output / 'result.json', record)
    print(json.dumps({'status': record['status'], 'wandb_url': record.get('wandb_url'),
                      'result': str(output / 'result.json'), 'readback': 'pending'}))
    return 0 if record['status'] == 'completed' else 1


def completion_message(record):
    summary = record.get('summary', {})
    parts = [f'HiDe-Prompt 결과 저장·검증 완료: 정확도 {record["validation_acceptance"]}, 구현 {record["implementation_gate"]}.']
    task2 = summary.get('validation/task2_seen/native_accuracy_pct')
    gaps = [summary.get(f'validation/{metric}/reference_minus_native_pp') for metric in ACCURACY_METRICS]
    gaps = [value for value in gaps if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)]
    accuracy = []
    if isinstance(task2, (int, float)) and not isinstance(task2, bool) and math.isfinite(task2):
        accuracy.append(f'태스크2 정확도 {task2:.2f}%')
    if gaps:
        accuracy.append(f'최대 저자−구현 차이 {max(gaps):.2f}%p')
    if accuracy:
        parts.append(', '.join(accuracy) + '.')
    keys = ('process_wall_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib')
    values = [summary.get(f'efficiency/hide-prompt/{key}/mean') for key in keys]
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values):
        parts.append(f'효율성 3회 평균: {values[0]:.1f}초, GPU 보드 {values[1]:.3f}Wh, CUDA peak {values[2] / 1024:.2f}GiB.')
    else:
        parts.append('효율성 미실행.' if record['efficiency_status'] == 'not_executed' else f'효율성 {record["efficiency_status"]}.')
    parts.append(f'W&B {record["wandb_url"]}')
    return '\n'.join(parts)


def notify_verified(options):
    output = options.output.resolve()
    result_path = output / 'result.json'
    record, proof = read_json(result_path), read_json(options.notify_after_readback)
    pins = read_json(output / 'artifact_files.json')
    checks = proof.get('artifact_files', {})
    artifact = record.get('wandb_artifact', {})
    if (record.get('status') != 'completed' or record.get('wandb_logging_status') != 'uploaded'
            or proof.get('status') != 'pass' or proof.get('run_state') != 'finished'
            or proof.get('artifact_state') != 'COMMITTED' or proof.get('wandb_url') != record.get('wandb_url')
            or proof.get('artifact') != artifact.get('name')
            or proof.get('result_sha256') != file_hash(result_path) or set(checks) != set(pins)):
        raise ValueError('Slack completion requires matching finished/COMMITTED/hash readback proof')
    for filename, pin in pins.items():
        checked = checks[filename]
        if (checked.get('matches_uploaded_pin') is not True or checked.get('local_snapshot_matches') is not True
                or checked.get('sha256') != pin['sha256'] or file_hash(output / pin['local_file']) != pin['sha256']):
            raise ValueError('Slack completion artifact proof differs from immutable report snapshots')
    receipt = output / 'notification.json'
    if receipt.is_file() and read_json(receipt).get('status') == 'sent':
        print('Verified HiDe report completion was already sent.')
        return 0
    configure_environment()
    message = completion_message(record)
    status = notify_slack(message)
    write_json(receipt, {'status': status, 'message': message, 'result_sha256': file_hash(result_path),
                         'readback_sha256': file_hash(options.notify_after_readback)})
    print(json.dumps({'slack_completion_status': status, 'receipt': str(receipt)}))
    return 0 if status == 'sent' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Fresh publication receipt directory; reused only for verified notification')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--notify-after-readback', type=Path)
    options = parser.parse_args(argv)
    if not options.execute:
        prepare(options.report_dir)
        print('Completed HiDe report prepared; no credentials, network, uploads or Slack sends. execution=False')
        return 0
    if options.notify_after_readback is not None:
        return notify_verified(options)
    return publish(options)


if __name__ == '__main__':
    sys.exit(main())
