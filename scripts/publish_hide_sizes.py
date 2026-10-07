"""Publish verified HiDe Small/Tiny validation and first-task efficiency reports.

Preparation rechecks saved inputs without authentication or writes. --execute
uploads immutable report snapshots; completion Slack requires a separate
finished/COMMITTED/hash readback receipt. No training or GPU work is performed.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS
from scripts.publish_hide_report import ACCURACY_METRICS, REPORT_FILES, file_hash, read_json
from scripts.report_hide_results import METRICS

SOURCE_KEYS = {'small_reference', 'small_native', 'small_efficiency',
               'tiny_reference', 'tiny_native', 'tiny_efficiency', 'base_efficiency'}
VARIANTS = ('base', 'small', 'tiny')


def prepare(directory):
    """Require actual passing results and recompute the entire pinned payload."""
    directory = Path(directory).resolve()
    pins = {}
    for filename in REPORT_FILES:
        path = directory / filename
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing or empty size report artifact: ' + filename)
        pins[filename] = file_hash(path)
    report = read_json(directory / 'results.json')
    if (report.get('status') != 'completed' or report.get('model') != 'hide-prompt'
            or report.get('stage') != 'hide_sizes' or set(report.get('validation', {})) != {'small', 'tiny'}):
        raise ValueError('Publication requires completed Small/Tiny validation and pilot observations')
    for variant in ('small', 'tiny'):
        validation = report['validation'][variant]
        acceptance, batches = validation.get('acceptance', {}), validation.get('first_batch_comparison', {})
        rows, checks = validation.get('accuracy_metrics', []), acceptance.get('structural_checks', {})
        if (validation.get('report_gate') != 'pass' or acceptance.get('status') != 'pass'
                or any(checks.get(name) is not True for name in REQUIRED_STRUCTURAL_CHECKS)
                or batches.get('status') != 'pass' or len(batches.get('rows', [])) != 4
                or any(row.get('status') != 'pass' for row in batches['rows'])
                or len(rows) != 4 or {row.get('metric') for row in rows} != ACCURACY_METRICS
                or any(row.get('status') != 'pass' for row in rows)):
            raise ValueError('Both sizes require passing accuracy, structure and first-batch gates')
    methods = report.get('efficiency', {}).get('methods', [])
    if (report.get('efficiency', {}).get('status') != 'completed' or len(methods) != 3
            or {method.get('variant') for method in methods} != set(VARIANTS)
            or any(method.get('measurement_count') != 3 for method in methods)):
        raise ValueError('Size comparison requires Base/Small/Tiny with three measurements each')
    for method in methods:
        estimate = method.get('flops', {}).get('supported_ops_total')
        if type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate <= 0:
            raise ValueError('Each size requires its separate partial FLOPs diagnostic')
    sources = report.get('sources', {})
    if set(sources) != SOURCE_KEYS:
        raise ValueError('Size report requires exactly seven pinned run sources')
    inputs = {}
    for name, source in sources.items():
        path, pin = Path(source.get('file', '')), source.get('sha256', '')
        if not path.is_absolute() or not path.is_file() or not re.fullmatch(r'[0-9a-f]{64}', pin) or file_hash(path) != pin:
            raise ValueError('Size report provenance is missing or changed')
        inputs[name] = path
    from scripts.report_hide_sizes import build_report, csv_fields, csv_rows, markdown
    expected = build_report(inputs)
    stable = lambda value: {key: item for key, item in value.items() if key != 'generated_at'}
    if stable(report) != stable(expected):
        raise ValueError('Size report payload disagrees with actual pinned observations')
    fields = csv_fields(expected)
    expected_rows = [{key: (json.dumps(row[key], sort_keys=True)
                           if isinstance(row.get(key), (dict, list)) else
                           str(row[key]) if row.get(key) is not None else '')
                      for key in fields} for row in csv_rows(expected)]
    with (directory / 'results.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields or list(reader) != expected_rows:
            raise ValueError('Size report CSV disagrees with actual pinned observations')
    if (directory / 'report.md').read_text() != markdown(expected):
        raise ValueError('Size report Markdown disagrees with actual pinned observations')
    return report, pins


def scalar_summary(report):
    result = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'paper_reproduction': False, 'full_cl_included': False,
              'seed_variation_estimated': False, 'equal_compute_budget': False,
              'efficiency_status': 'completed', 'measurement_repeats': 3, 'partial_flops_runs_per_size': 1}
    for variant in ('small', 'tiny'):
        result['validation/' + variant + '/implementation_gate'] = report['validation'][variant]['report_gate']
        for row in report['validation'][variant]['accuracy_metrics']:
            for key in ('reference_accuracy_pct', 'native_accuracy_pct', 'reference_minus_native_pp'):
                value = row.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    result[f'validation/{variant}/{row["metric"]}/{key}'] = value
    for method in report['efficiency']['methods']:
        variant = method['variant']
        for metric in METRICS:
            for stat in ('mean', 'std_sample', 'median', 'min', 'max'):
                value = method.get('resources', {}).get(metric, {}).get(stat)
                if type(value) in (int, float) and math.isfinite(value):
                    result[f'efficiency/{variant}/{metric}/{stat}'] = value
        for stat in ('mean', 'std_sample'):
            value = method.get('accuracy_pct', {}).get(stat)
            if type(value) in (int, float) and math.isfinite(value):
                result[f'efficiency/{variant}/pilot_accuracy_pct/{stat}'] = value
        result[f'efficiency/{variant}/partial_supported_ops_total'] = method['flops']['supported_ops_total']
    return result


def tables(report):
    validation = [[variant, row['metric'], row['reference_accuracy_pct'], row['native_accuracy_pct'],
                   row['reference_minus_native_pp'], row['status']]
                  for variant in ('small', 'tiny') for row in report['validation'][variant]['accuracy_metrics']]
    columns = ['variant', 'measurement_count', 'partial_supported_ops_total']
    columns.extend(metric + '/' + stat for metric in METRICS for stat in ('mean', 'std_sample'))
    resources, phases, parameters = [], [], []
    for method in report['efficiency']['methods']:
        resources.append([method['variant'], 3, method['flops']['supported_ops_total']] +
                         [method.get('resources', {}).get(metric, {}).get(stat) for metric in METRICS for stat in ('mean', 'std_sample')])
        for name, metrics in method.get('phases', {}).items():
            phases.append([method['variant'], name, metrics.get('wall_s', {}).get('mean'),
                           metrics.get('wall_s', {}).get('std_sample'), metrics.get('gpu_board_energy_j', {}).get('mean')])
        for phase, metrics in method.get('parameters', {}).items():
            for name, stats in metrics.items():
                parameters.append([method['variant'], phase, name, stats.get('mean'), stats.get('std_sample')])
    return {'validation/accuracy': (['variant', 'metric', 'reference_accuracy_pct', 'native_accuracy_pct',
                                     'reference_minus_native_pp', 'status'], validation),
            'efficiency/resources': (columns, resources),
            'efficiency/phases': (['variant', 'phase', 'wall_s_mean', 'wall_s_std_sample', 'gpu_board_energy_j_mean'], phases),
            'efficiency/parameters': (['variant', 'phase', 'metric', 'mean', 'std_sample'], parameters)}


def require_project(profile):
    if profile.get('wandb_entity') != 'laymond1' or profile.get('wandb_project') != 'cl-efficiency':
        raise ValueError('HiDe size publication requires configured laymond1/cl-efficiency')


def publish(options):
    report, pins = prepare(options.report_dir)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'snapshots'
    snapshot.mkdir()
    for name, pin in pins.items():
        shutil.copyfile(options.report_dir / name, snapshot / name)
        if file_hash(snapshot / name) != pin:
            raise ValueError('Size report changed while taking immutable snapshots')
    snapshot_report, snapshot_pins = prepare(snapshot)
    if snapshot_report != report or snapshot_pins != pins:
        raise ValueError('Size report changed before publication')
    name = 'hide-sizes-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    metadata = {'schema_version': 1, 'name': name, 'model': 'hide-prompt', 'stage': 'hide_sizes',
                'report_only': True, 'new_training': False, 'new_gpu_computation': False,
                'wandb_group': 'hide-size-results', 'report_dir': str(options.report_dir.resolve()),
                'report_files_sha256': pins, 'sources': report['sources'], 'summary': scalar_summary(report),
                'publication_source_sha256': {source: file_hash(ROOT / source) for source in
                    ('scripts/publish_hide_sizes.py', 'scripts/report_hide_sizes.py', 'scripts/publish_hide_report.py')}}
    write_json(output / 'artifact_metadata.json', metadata)
    artifacts = {name: {'local_file': 'snapshots/' + name, 'sha256': pin} for name, pin in pins.items()}
    artifacts['artifact_metadata.json'] = {'local_file': 'artifact_metadata.json', 'sha256': file_hash(output / 'artifact_metadata.json')}
    write_json(output / 'artifact_files.json', artifacts)
    record = {**metadata, 'status': 'failed', 'wandb_logging_status': 'not_started',
              'readback_status': 'pending', 'slack_completion_status': 'deferred_until_readback'}
    write_json(output / 'result.json', record)
    wandb = None
    try:
        require_project(configure_environment())
        import wandb
        run = wandb.init(entity='laymond1', project='cl-efficiency', name=name,
                         group='hide-size-results', job_type='report-only',
                         tags=['report-only', 'hide-prompt', 'small', 'tiny', 'single-seed'],
                         config=metadata, dir=str(output), mode='online')
        record['wandb_url'] = run.url
        run.summary.update(metadata['summary'])
        run.log({key: wandb.Table(columns=columns, data=data) for key, (columns, data) in tables(report).items()})
        artifact = wandb.Artifact(name, type='cl-hide-size-report', metadata={
            'report_only': True, 'stage': 'hide_sizes', 'validation_tasks': 2, 'pilot_tasks': 1, 'seed': 1993})
        for name, pin in artifacts.items():
            artifact.add_file(str(output / pin['local_file']), name=name)
        artifact.add_file(str(output / 'artifact_files.json'), name='artifact_files.json')
        logged = run.log_artifact(artifact)
        logged.wait()
        if logged.state != 'COMMITTED':
            raise ValueError('Size report artifact did not reach COMMITTED')
        record['wandb_artifact'] = {'name': logged.qualified_name, 'state': logged.state}
        record['wandb_logging_status'] = 'uploaded'
        wandb.finish(exit_code=0)
        record['status'] = 'completed'
    except Exception as error:
        record.update(status='logging_failed', error_type=type(error).__name__,
                      wandb_logging_status='failed:' + type(error).__name__)
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
    write_json(output / 'result.json', record)
    print('HiDe size report publication status: ' + record['status'])
    return 0 if record['status'] == 'completed' else 1


def completion_message(record):
    summary = record['summary']
    parts = ['HiDe Small→Tiny 결과 저장·원격 검증 완료 (GPU1, batch64, seed1993).']
    for variant in ('small', 'tiny'):
        accuracy = summary[f'validation/{variant}/task2_seen/native_accuracy_pct']
        wall = summary[f'efficiency/{variant}/process_wall_s/mean']
        energy = summary[f'efficiency/{variant}/gpu_board_energy_wh/mean']
        allocated = summary[f'efficiency/{variant}/cuda_peak_allocated_mib/mean']
        device = summary[f'efficiency/{variant}/peak_gpu_device_memory_mib/mean']
        parts.append(f'{variant.capitalize()}: 검증 태스크2 정확도 {accuracy:.2f}%; '
                     f'첫 태스크 TII1+P1 계측3회 평균 {wall:.1f}초, GPU 보드 {energy:.3f}Wh, '
                     f'CUDA allocated {allocated / 1024:.2f}GiB, NVML {device / 1024:.2f}GiB.')
    parts.append('W&B ' + record['wandb_url'])
    return '\n'.join(parts)


def notify_verified(options):
    output = options.output.resolve()
    result_path = output / 'result.json'
    record, proof = read_json(result_path), read_json(options.notify_after_readback)
    pins = read_json(output / 'artifact_files.json')
    checks, artifact = proof.get('artifact_files', {}), record.get('wandb_artifact', {})
    url = urlparse(record.get('wandb_url', ''))
    parts = url.path.strip('/').split('/')
    if (record.get('status') != 'completed' or record.get('model') != 'hide-prompt' or record.get('stage') != 'hide_sizes'
            or record.get('wandb_logging_status') != 'uploaded' or artifact.get('state') != 'COMMITTED'
            or record.get('report_only') is not True or record.get('new_training') is not False or record.get('new_gpu_computation') is not False
            or record.get('wandb_group') != 'hide-size-results' or url.scheme != 'https' or url.hostname != 'wandb.ai'
            or len(parts) != 4 or parts[:3] != ['laymond1', 'cl-efficiency', 'runs']
            or proof.get('status') != 'pass' or proof.get('run_state') != 'finished'
            or proof.get('artifact_state') != 'COMMITTED' or proof.get('wandb_url') != record.get('wandb_url')
            or proof.get('artifact') != artifact.get('name') or proof.get('result_sha256') != file_hash(result_path)
            or set(pins) != {*REPORT_FILES, 'artifact_metadata.json'} or set(checks) != set(pins)
            or set(record.get('report_files_sha256', {})) != set(REPORT_FILES)):
        raise ValueError('Slack completion requires matching finished/COMMITTED/hash size-report readback')
    for name, pin in pins.items():
        path = (output / pin.get('local_file', '')).resolve()
        checked = checks[name]
        if (not path.is_relative_to(output) or not path.is_file()
                or checked.get('matches_uploaded_pin') is not True or checked.get('local_snapshot_matches') is not True
                or checked.get('sha256') != pin.get('sha256') or file_hash(path) != pin.get('sha256')
                or name in REPORT_FILES and pin['sha256'] != record['report_files_sha256'][name]):
            raise ValueError('Slack completion proof differs from immutable size-report snapshots')
    metadata = read_json(output / pins['artifact_metadata.json']['local_file'])
    if any(record.get(key) != value for key, value in metadata.items()):
        raise ValueError('Publication record differs from uploaded size-report metadata')
    receipt = output / 'notification.json'
    if receipt.is_file() and read_json(receipt).get('status') == 'sent':
        if read_json(receipt).get('result_sha256') != file_hash(result_path):
            raise ValueError('Existing completion notification belongs to another publication result')
        print('Verified HiDe size completion was already sent.')
        return 0
    message = completion_message(record)
    require_project(configure_environment())
    status = notify_slack(message)
    write_json(receipt, {'status': status, 'message': message, 'result_sha256': file_hash(result_path),
                         'readback_sha256': file_hash(options.notify_after_readback)})
    print('HiDe size Slack completion status: ' + status)
    return 0 if status == 'sent' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--notify-after-readback', type=Path)
    options = parser.parse_args(argv)
    if not options.execute:
        prepare(options.report_dir)
        print('HiDe size report prepared; execution=False')
        return 0
    if options.notify_after_readback is not None:
        return notify_verified(options)
    return publish(options)


if __name__ == '__main__':
    sys.exit(main())
