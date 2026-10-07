"""Publish completed EASE full-CL observations with immutable artifact pins.

Default preparation is read-only and never authenticates. --execute uploads a
report-only run. --execute --notify-after-readback sends completion only after
the existing verifier confirms finished/COMMITTED and all saved file hashes.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.report_ease_fullcl import REPORT_FILES, RESOURCE_KEYS, audit_prediction_counts, audit_prefix_predictions, build_report, file_hash, read_json, verify_saved_predictions


def prepare(directory):
    directory = Path(directory).resolve()
    pins = {}
    for filename in REPORT_FILES:
        path = directory / filename
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing or empty full-CL report artifact: ' + filename)
        pins[filename] = file_hash(path)
    report = read_json(directory / 'results.json')
    sources = report.get('sources', {})
    expected_sources = {'native', 'reference', 'accepted_native', 'native_artifact_manifest',
                        'reference_artifact_manifest', 'accepted_native_artifact_manifest'}
    if report.get('comparison', {}).get('status') != 'not_provided':
        expected_sources.update({'hide_native', 'hide_artifact_manifest'})
    if set(sources) != expected_sources:
        raise ValueError('Full-CL report requires pinned run, validation and prediction provenance')
    for source in sources.values():
        path = Path(source.get('file', ''))
        if not path.is_absolute() or not path.is_file() or file_hash(path) != source.get('sha256'):
            raise ValueError('Full-CL report provenance is missing or changed')
    hide = read_json(sources['hide_native']['file']) if 'hide_native' in sources else None
    expected = build_report(*(read_json(sources[name]['file']) for name in ('native', 'reference', 'accepted_native')), hide)
    runs = [('native', 10), ('reference', 2), ('accepted_native', 2)]
    if hide is not None:
        runs.append(('hide', 10))
    for name, tasks in runs:
        key = 'hide_native' if name == 'hide' else name
        if str(verify_saved_predictions(sources[key]['file'], tasks)) != sources[name + '_artifact_manifest']['file']:
            raise ValueError('Full-CL prediction manifest does not belong to the pinned run')
        audit_prediction_counts(sources[key]['file'], tasks)
    audit_prefix_predictions(sources['native']['file'], sources['accepted_native']['file'])
    if {key: value for key, value in report.items() if key != 'sources'} != expected:
        raise ValueError('Full-CL report metrics disagree with pinned inputs')
    return report, pins


def scalar_summary(report):
    accuracy = report['accuracy']
    result = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'paper_reproduction': False, 'seed_variation_estimated': False,
              'equal_compute_budget': False, 'implementation_prefix_gate': 'pass',
              'full_ten_task_accuracy_acceptance': 'not_assessed_by_matched_ten_task_reference',
              'hide_comparison_status': report['comparison']['status']}
    for key in ('final_accuracy_pct', 'average_incremental_accuracy_pct', 'forgetting_pp'):
        result['full_cl/' + key] = accuracy[key]
    for key, value in report['resources']['process'].items():
        if type(value) in (int, float):
            result['resources/' + key] = value
    for key, value in report['work_counts'].items():
        if type(value) is int:
            result['work/' + key] = value
    return result


def tables(report):
    columns = ['task', 'seen_accuracy_pct', 'old_accuracy_pct', 'new_accuracy_pct', 'wall_s', *RESOURCE_KEYS]
    costs = report['resources']['tasks']
    data = [[row[key] for key in columns[:4]] + [cost['wall_s']] + [cost['metrics'][key] for key in RESOURCE_KEYS]
            for row, cost in zip(report['accuracy']['trajectory'], costs)]
    matrix = [[time + 1, task + 1, value]
              for time, row in enumerate(report['accuracy']['class_il_accuracy_matrix_pct'])
              for task, value in enumerate(row) if value is not None]
    phases = [[row['task'], name, phase['wall_s'], phase.get('gpu_board_energy_j')]
              for row in costs for name, phase in row['phases'].items()]
    result = {'full_cl/task_metrics': (columns, data),
            'full_cl/accuracy_matrix': (['tasks_learned', 'evaluated_task', 'class_il_accuracy_pct'], matrix),
            'full_cl/task_phases': (['task', 'phase', 'wall_s', 'gpu_board_energy_j'], phases)}
    if report['comparison']['status'] == 'compatible':
        result['comparison/methods'] = (
            ['model', 'final_accuracy_pct', 'average_incremental_accuracy_pct', 'forgetting_pp',
             'process_wall_s', 'gpu_board_energy_wh', 'equal_compute_budget'],
            [[row['model'], row['accuracy']['final_accuracy_pct'], row['accuracy']['average_incremental_accuracy_pct'],
              row['accuracy']['forgetting_pp'], row['metrics'].get('process_wall_s'),
              row['metrics'].get('gpu_board_energy_wh'), False] for row in report['comparison']['methods']])
    return result


def publish(options):
    report, pins = prepare(options.report_dir)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'snapshots').mkdir()
    for filename, pin in pins.items():
        shutil.copyfile(options.report_dir / filename, output / 'snapshots' / filename)
        if file_hash(output / 'snapshots' / filename) != pin:
            raise ValueError('Full-CL report changed while taking upload snapshots')
    name = 'ease-fullcl-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    metadata = {'schema_version': 1, 'name': name, 'model': 'ease', 'mode': 'full_cl',
                'report_only': True, 'new_training': False, 'new_gpu_computation': False,
                'report_dir': str(options.report_dir.resolve()), 'report_files_sha256': pins,
                'sources': report['sources'], 'summary': scalar_summary(report)}
    write_json(output / 'artifact_metadata.json', metadata)
    artifact_files = {filename: {'local_file': 'snapshots/' + filename, 'sha256': pin}
                      for filename, pin in pins.items()}
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
            raise ValueError('Full-CL publication requires configured laymond1/cl-efficiency')
        import wandb
        run = wandb.init(entity='laymond1', project='cl-efficiency', name=name,
                         group='ease-base-full-cl', job_type='report-only',
                         tags=['report-only', 'ease', 'full-cl', 'single-seed'],
                         config=metadata, dir=str(output), mode='online')
        record['wandb_url'] = run.url
        run.summary.update(metadata['summary'])
        run.log({name: wandb.Table(columns=columns, data=data) for name, (columns, data) in tables(report).items()})
        artifact = wandb.Artifact(name, type='cl-ease-fullcl-report', metadata={
            'report_only': True, 'tasks': 10, 'seed': 1993, 'implementation_prefix_gate': 'pass'})
        for filename, pin in artifact_files.items():
            artifact.add_file(str(output / pin['local_file']), name=filename)
        artifact.add_file(str(output / 'artifact_files.json'), name='artifact_files.json')
        logged = run.log_artifact(artifact)
        logged.wait()
        if logged.state != 'COMMITTED':
            raise ValueError('Full-CL report artifact did not reach COMMITTED')
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
    print(json.dumps({'status': record['status'], 'wandb_url': record.get('wandb_url'),
                      'result': str(output / 'result.json'), 'readback': 'pending'}))
    return 0 if record['status'] == 'completed' else 1


def notify_verified(options):
    output = options.output.resolve()
    result_path = output / 'result.json'
    record, proof = read_json(result_path), read_json(options.notify_after_readback)
    pins = read_json(output / 'artifact_files.json')
    checks = proof.get('artifact_files', {})
    if (record.get('status') != 'completed' or record.get('wandb_logging_status') != 'uploaded'
            or proof.get('status') != 'pass' or proof.get('run_state') != 'finished'
            or proof.get('artifact_state') != 'COMMITTED' or proof.get('wandb_url') != record.get('wandb_url')
            or proof.get('artifact') != record.get('wandb_artifact', {}).get('name')
            or proof.get('result_sha256') != file_hash(result_path) or set(checks) != set(pins)):
        raise ValueError('Slack completion requires matching finished/COMMITTED/hash readback proof')
    for filename, pin in pins.items():
        checked = checks[filename]
        if (checked.get('matches_uploaded_pin') is not True or checked.get('local_snapshot_matches') is not True
                or checked.get('sha256') != pin['sha256'] or file_hash(output / pin['local_file']) != pin['sha256']):
            raise ValueError('Slack completion proof differs from immutable full-CL report snapshots')
    receipt = output / 'notification.json'
    if receipt.is_file() and read_json(receipt).get('status') == 'sent':
        print('Verified EASE full-CL completion was already sent.')
        return 0
    summary = record['summary']
    message = (f'EASE Base 전체10태스크 결과 저장·검증 완료 (GPU1, batch64, seed1993).\n'
               f'최종 정확도 {summary["full_cl/final_accuracy_pct"]:.2f}%, '
               f'평균 증분 정확도 {summary["full_cl/average_incremental_accuracy_pct"]:.2f}%, '
               f'과거9태스크 망각 {summary["full_cl/forgetting_pp"]:.2f}%p.\nW&B {record["wandb_url"]}')
    configure_environment()
    status = notify_slack(message)
    write_json(receipt, {'status': status, 'message': message, 'result_sha256': file_hash(result_path),
                         'readback_sha256': file_hash(options.notify_after_readback)})
    print(json.dumps({'slack_completion_status': status, 'receipt': str(receipt)}))
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
        print('Full-CL report prepared; no credentials, network or notifications. execution=False')
        return 0
    if options.notify_after_readback is not None:
        return notify_verified(options)
    return publish(options)


if __name__ == '__main__':
    sys.exit(main())
