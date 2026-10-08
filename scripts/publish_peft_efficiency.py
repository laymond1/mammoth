"""Verify twenty saved W&B workers and publish the nine-method pilot report.

Default operation audits local records only. --execute reads completed remote
runs and their committed artifacts, publishes small report files, verifies them
and sends the final Slack completion. This tool never starts training.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.publish_low_rank_results import manifest_matches, verify_publication
from scripts.report_low_rank_results import run_path
from scripts.report_peft_efficiency import REPORT_FILES, build_report, csv_text, markdown, read_json
from scripts.run_peft_efficiency import file_hash


def pins(paths):
    return {Path(path).name: {'file': str(Path(path).resolve()), 'sha256': file_hash(path),
                              'bytes': Path(path).stat().st_size} for path in paths}


def prepare(directory):
    directory = Path(directory).resolve()
    report = read_json(directory / 'results.json')
    expected = build_report(report['execution']['file'])
    if any(report.get(key) != value for key, value in expected.items() if key != 'generated_at_utc'):
        raise ValueError('Published report differs from the complete saved observations')
    if (directory / 'RESULTS.md').read_text() != markdown(report) or (directory / 'results.csv').read_text() != csv_text(report):
        raise ValueError('Markdown or CSV differs from the report values')
    paths = [directory / name for name in REPORT_FILES] + [directory / 'plan.json']
    if any(not path.is_file() or not path.stat().st_size for path in paths):
        raise ValueError('Report is missing a required nonempty artifact file')
    return report, pins(paths)


def verify_worker(api, source, output):
    local = read_json(source['file'])
    if file_hash(source['file']) != source['sha256']:
        raise ValueError('Local worker changed before remote verification')
    run = api.run(run_path(local['wandb_url']))
    summary = dict(run.summary)
    if (run.state != 'finished' or summary.get('benchmark_status') != 'completed'
            or summary.get('benchmark_pass') != source['pass']):
        raise ValueError('Every fresh worker must be finished and remotely completed')
    checked = {}
    for key in ('tasks_completed', 'outer_observe_steps', 'outer_training_stream_examples', 'evaluation_examples'):
        name = 'efficiency/' + key
        value = summary.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != local['metrics'][key]:
            raise ValueError('Remote worker update/sample counts disagree with saved observations')
        checked[name] = local['metrics'][key]
    expected_accuracy = local['accuracy']['mean_accuracy_pct']
    value = summary.get('accuracy/mean_accuracy_pct')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(value, expected_accuracy, rel_tol=0, abs_tol=1e-10):
        raise ValueError('Remote worker accuracy disagrees with saved predictions')
    directory = Path(source['file']).parent
    required = ['spec.json', 'task1_predictions.npz', 'telemetry.csv', 'phases.json']
    if source['pass'] == 'flops':
        required.append('flops_ops.json')
    if source['method'] != 'ranpac':
        required.extend(('epoch_results.json', 'task_results.json', 'first_batch_audits.json', 'state_audits.json'))
    candidates = [artifact for artifact in run.logged_artifacts()
                  if artifact.state == 'COMMITTED' and set(required + ['metrics.json']).issubset(artifact.manifest.entries)]
    if len(candidates) != 1:
        raise ValueError('Each fresh run requires one unambiguous committed evidence artifact')
    artifact = candidates[0]
    proofs = manifest_matches(artifact, pins(directory / name for name in required))
    destination = Path(output) / (source['method'] + '-' + directory.name)
    destination.mkdir(parents=True, exist_ok=True)
    remote = read_json(artifact.get_path('metrics.json').download(root=str(destination)))
    fields = ('model', 'method', 'pass', 'status', 'computation_status', 'protocol',
              'checkpoint_sha256', 'canonical_pretrained_sha256', 'execution_source_sha256',
              'precision', 'counts', 'accuracy', 'metrics', 'structural_checks', 'data_snapshots',
              'task_results', 'initial_state_sha256', 'flops')
    if any(remote.get(key) != local.get(key) for key in fields):
        raise ValueError('Remote evidence semantics differ from the completed local worker')
    return {'method': source['method'], 'pass': source['pass'], 'wandb_url': local['wandb_url'],
            'run_state': run.state, 'artifact_state': artifact.state,
            'artifact': artifact.qualified_name, 'artifact_digest': artifact.digest,
            'count_summary_verified': checked, 'accuracy_summary_verified': True,
            'record_semantics_verified': True, 'files': proofs}


def scalar_summary(report):
    result = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'new_source_run_count': 20, 'method_count': 9, 'new_measurement_runs': 15,
              'new_flops_runs': 5, 'publication_gate': 'pass', 'equal_compute_budget': False,
              'paper_reproduction': False, 'first_task_only': True}
    for name, entry in report['methods'].items():
        result[name + '/accuracy_pct'] = entry['accuracy_pct']['mean']
        for key in ('process_wall_s', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib',
                    'process_tree_cpu_s', 'gpu_board_energy_wh'):
            result[name + '/' + key + '_mean'] = entry['resources'][key]['mean']
        result[name + '/partial_flops'] = entry['flops']['supported_ops_total']
    return result


def publish(options):
    directory = options.report_dir.resolve()
    report, files = prepare(directory)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = configure_environment()
    if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('Publication destination differs from the approved project')
    import wandb
    api = wandb.Api(timeout=60)
    proofs = []
    for index, source in enumerate(report['sources'], 1):
        proofs.append(verify_worker(api, source, output / 'readbacks'))
        write_json(output / 'source_readback.json', {'status': 'running', 'workers': proofs})
        print(f'W&B worker evidence verified {index}/20', flush=True)
    source_readback = {'status': 'pass', 'worker_count': 20, 'workers': proofs}
    write_json(output / 'source_readback.json', source_readback)
    # Publication uses a separate immutable snapshot of only public reports.
    snapshot = output / 'report_snapshot'
    snapshot.mkdir()
    for filename, pin in files.items():
        (snapshot / filename).write_bytes(Path(pin['file']).read_bytes())
    snapshot_files = pins(snapshot / name for name in files)
    if any(snapshot_files[name]['sha256'] != pin['sha256'] or snapshot_files[name]['bytes'] != pin['bytes']
           for name, pin in files.items()):
        raise ValueError('Public report changed during source readback; no publication started')
    summary = scalar_summary(report)
    name = 'peft-base-task1-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = wandb.init(entity='laymond1', project='cl-efficiency', mode='online', name=name,
                     group='peft-common-efficiency', job_type='report', dir=str(output),
                     config={'report_only': True, 'new_training': False, 'protocol': report['protocol'],
                             'report_sha256': files['results.json']['sha256']})
    try:
        run.summary.update(summary)
        artifact = wandb.Artifact(name, type='cl-efficiency-report', metadata={'files': {
            filename: {'sha256': pin['sha256'], 'bytes': pin['bytes']} for filename, pin in snapshot_files.items()}})
        for pin in snapshot_files.values():
            artifact.add_file(pin['file'])
        logged = run.log_artifact(artifact)
        logged.wait()
        url, artifact_name = run.url, logged.qualified_name
        write_json(output / 'publication_pending.json', {'wandb_url': url, 'artifact': artifact_name})
    finally:
        run.finish()
    api.flush()
    remote = verify_publication(api, url, artifact_name, snapshot_files, summary, 180)
    write_json(output / 'publication_readback.json', remote)
    status = notify_slack('Base 첫 태스크 PEFT 효율성 pilot 완료: LoRA4종 + RanPAC 신규20실행 finished, '
                         'artifact COMMITTED 확인. 기존4방법 재사용하여 총9방법 보고서 작성. '
                         '동일 seed 비용 반복이며 전체 CL/동일 예산 비교가 아닙니다. W&B ' + url)
    receipt = {'status': 'completed' if status == 'sent' else 'notification_failed',
               'wandb_url': url, 'artifact': artifact_name, 'slack_completion_status': status,
               'source_worker_count': 20, 'report_files': snapshot_files, 'remote_readback': remote,
               'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / 'publication.json', receipt)
    for filename in ('publication.json', 'source_readback.json', 'publication_readback.json'):
        (directory / filename).write_bytes((output / filename).read_bytes())
    print(json.dumps({'status': receipt['status'], 'wandb_url': url, 'slack': status}))
    return 0 if status == 'sent' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'benchmarks/cl_efficiency/peft_pilot')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/peft_pilot/publication-20261008')
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if options.execute:
        return publish(options)
    report, files = prepare(options.report_dir)
    print(json.dumps({'execute': False, 'method_count': report['method_count'], 'files': list(files)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
