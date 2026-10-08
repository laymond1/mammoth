"""Publish the accepted pilot subset with the SD failure kept explicit.

Read-only by default. --execute verifies remote source evidence, uploads an
immutable report snapshot, verifies its manifest and sends the Slack result.
This script never trains a model or accepts a quarantined result.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.publish_low_rank_results import manifest_matches
from scripts.publish_peft_efficiency import pins, verify_worker
from scripts.report_low_rank_results import run_path
from scripts.report_peft_efficiency import csv_text, read_json
from scripts.report_peft_partial import REPORT_FILES, build_report, markdown
from scripts.run_peft_efficiency import file_hash


def prepare(directory):
    directory = Path(directory).resolve()
    report = read_json(directory / 'results.json')
    evidence = report.get('diagnostic_evidence')
    expected = build_report(report['execution']['file'], diagnostic_path=evidence['file'] if evidence else None)
    if any(report.get(key) != value for key, value in expected.items() if key != 'generated_at_utc'):
        raise ValueError('Partial report differs from saved accepted and quarantined observations')
    if ((directory / 'RESULTS.md').read_text() != markdown(report)
            or (directory / 'results.csv').read_text() != csv_text(report)):
        raise ValueError('Partial Markdown or CSV differs from its observations')
    paths = [directory / name for name in REPORT_FILES] + [directory / 'plan.json']
    if any(not path.is_file() or not path.stat().st_size for path in paths):
        raise ValueError('Missing nonempty partial report artifact')
    return report, pins(paths)


def verify_diagnostic(api, source, output):
    if file_hash(source['file']) != source['sha256']:
        raise ValueError('SD author diagnostic changed before remote verification')
    local = read_json(source['file'])
    run = api.run(run_path(local['wandb_url']))
    summary = dict(run.summary)
    if (run.state != 'finished' or summary.get('diagnostic_status') != 'completed'
            or summary.get('measurement_aggregate_eligible') is not False
            or summary.get('diagnosis') != local['diagnosis']):
        raise ValueError('SD diagnostic run must be finished and excluded from aggregates')
    for key, value in local['counts'].items():
        actual = summary.get('counts/' + key)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != value:
            raise ValueError('SD remote diagnostic update counts differ')
    if summary.get('accuracy/mean_accuracy_pct') != local['accuracy']['mean_accuracy_pct']:
        raise ValueError('SD remote diagnostic accuracy differs')
    directory = Path(source['file']).parent
    required = ('spec.json', 'comparison.json', 'task1_predictions.npz', 'artifact_manifest.json')
    candidates = [artifact for artifact in run.logged_artifacts()
                  if artifact.state == 'COMMITTED' and set((*required, 'result.json')).issubset(artifact.manifest.entries)]
    if len(candidates) != 1:
        raise ValueError('SD diagnostic requires one unambiguous committed evidence artifact')
    artifact = candidates[0]
    proofs = manifest_matches(artifact, pins(directory / name for name in required))
    destination = Path(output) / 'sd-author-diagnostic'
    destination.mkdir(parents=True, exist_ok=True)
    remote_path = Path(artifact.get_path('result.json').download(root=str(destination)))
    manifest = read_json(directory / 'artifact_manifest.json')
    pin = manifest.get('files', {}).get('result.json', {})
    if (manifest.get('full_checkpoints_uploaded') is not False
            or file_hash(remote_path) != pin.get('sha256') or remote_path.stat().st_size != pin.get('bytes')):
        raise ValueError('SD downloaded diagnostic differs from the pinned pre-upload result bytes')
    remote = read_json(remote_path)
    fields = ('method', 'mode', 'status', 'computation_status', 'protocol', 'reference_commit',
              'reference_source_sha256', 'execution_source_sha256', 'checkpoint_sha256',
              'initial_state_sha256', 'initial_learned_sha256', 'canonical_pretrained_sha256',
              'native_result_sha256', 'native_predictions_sha256', 'native_checkpoint_sha256',
              'precision', 'counts', 'accuracy', 'structural_checks',
              'comparison', 'diagnosis', 'state_audits', 'first_batch_audits', 'epoch_results',
              'task_results', 'observed_totals', 'measurement_aggregate_eligible')
    if any(remote.get(key) != local.get(key) for key in fields):
        raise ValueError('SD diagnostic remote semantics differ from saved observations')
    return {'method': 'sd-lora', 'kind': 'author_diagnostic', 'accepted': False,
            'wandb_url': local['wandb_url'], 'run_state': run.state,
            'artifact': artifact.qualified_name, 'artifact_state': artifact.state,
            'artifact_digest': artifact.digest, 'files': proofs,
            'record_semantics_verified': True, 'uploaded_record_sha256': file_hash(remote_path),
            'uploaded_record_bytes_verified': True, 'measurement_aggregate_eligible': False}


def scalar_summary(report):
    result = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'accepted_source_run_count': len(report['sources']), 'method_count': report['method_count'],
              'new_measurement_runs': report['new_measurement_runs'], 'new_flops_runs': report['new_flops_runs'],
              'publication_gate': report['publication_gate'], 'all_requested_methods_passed': False,
              'status': 'partial_completed', 'quarantined_workers': 1, 'deferred_workers': 3,
              'equal_compute_budget': False, 'paper_reproduction': False, 'first_task_only': True,
              'sd-lora/accepted': False, 'sd-lora/observed_accuracy_pct': 10.}
    evidence = report.get('diagnostic_evidence')
    if evidence:
        result['sd-lora/diagnosis'] = evidence['diagnosis']
        result['sd-lora/diagnostic_wandb_url'] = evidence['wandb_url']
    for name, entry in report['methods'].items():
        result[name + '/accuracy_pct'] = entry['accuracy_pct']['mean']
        for key in ('process_wall_s', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib',
                    'process_tree_cpu_s', 'gpu_board_energy_wh'):
            result[name + '/' + key + '_mean'] = entry['resources'][key]['mean']
        result[name + '/partial_flops'] = entry['flops']['supported_ops_total']
    return result


def summary_value_matches(actual, expected):
    """JSON transport may round floats; counts and booleans remain exact."""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return not isinstance(actual, bool) and isinstance(actual, (int, float)) and actual == expected
    if isinstance(expected, float):
        return (not isinstance(actual, bool) and isinstance(actual, (int, float))
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10))
    return actual == expected


def verify_report_publication(api, url, artifact_name, files, summary, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        api.flush()
        run = api.run(run_path(url))
        if run.state == 'finished':
            artifact = api.artifact(artifact_name)
            if artifact.state == 'COMMITTED':
                if any(not summary_value_matches(dict(run.summary).get(key), value) for key, value in summary.items()):
                    raise ValueError('Finished partial report summary differs from saved observations')
                return {'status': 'pass', 'run_state': run.state, 'artifact_state': artifact.state,
                        'artifact': artifact.qualified_name, 'artifact_digest': artifact.digest,
                        'files': manifest_matches(artifact, files), 'summary_verified': True,
                        'summary_numeric_policy': {'counts_and_booleans': 'exact',
                                                   'float_relative_tolerance': 1e-12,
                                                   'float_absolute_tolerance': 1e-10}}
        if time.monotonic() >= deadline:
            raise TimeoutError('Partial report did not reach finished/COMMITTED')
        time.sleep(min(5., max(0., deadline - time.monotonic())))


def finalize_saved_publication(directory, output, api):
    """Finish a saved upload without training or creating another W&B run."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    report, original_files = prepare(directory)
    if (output / 'publication.json').exists():
        raise ValueError('Publication already finalized; no duplicate Slack notification sent')
    source_readback = read_json(output / 'source_readback.json')
    sources = [*report['sources'], *report['failed_observations']]
    if report.get('diagnostic_evidence'):
        sources.append(report['diagnostic_evidence'])
    proofs = source_readback.get('workers', [])
    if (source_readback.get('status') != 'pass' or source_readback.get('accepted_worker_count') != 16
            or source_readback.get('quarantined_worker_count') != 1
            or len(proofs) != len(sources)
            or {proof['wandb_url'] for proof in proofs} != {source['wandb_url'] for source in sources}
            or sum(proof.get('accepted') is True for proof in proofs) != 16
            or any(proof.get('run_state') != 'finished' or proof.get('artifact_state') != 'COMMITTED'
                   or proof.get('record_semantics_verified') is not True for proof in proofs)):
        raise ValueError('Saved source verification is incomplete or differs from the partial report')
    snapshot = output / 'report_snapshot'
    snapshot_files = pins(snapshot / name for name in original_files)
    if any(snapshot_files[name]['sha256'] != pin['sha256'] or snapshot_files[name]['bytes'] != pin['bytes']
           for name, pin in original_files.items()):
        raise ValueError('Saved publication snapshot differs from the current immutable report')
    pending = read_json(output / 'publication_pending.json')
    url, artifact_name = pending['wandb_url'], pending['artifact']
    if (not url.startswith('https://wandb.ai/laymond1/cl-efficiency/runs/')
            or not artifact_name.startswith('laymond1/cl-efficiency/')):
        raise ValueError('Saved publication destination differs from the approved project')
    remote = verify_report_publication(api, url, artifact_name, snapshot_files, scalar_summary(report))
    write_json(output / 'publication_readback.json', remote)
    message = ('Base 첫 태스크 PEFT 부분 결과: EWC·InfLoRA·CL-LoRA·RanPAC의16개 승인 실행 완료, '
               '기존4방법과 총8방법 보고서 저장. SD10% 관측1개는 집계 제외, SD3개 보류. '
               '원격 finished/artifact COMMITTED 검증 완료. 전체20개 완료가 아닙니다. W&B ' + url)
    status = notify_slack(message)
    receipt = {'status': 'partial_completed' if status == 'sent' else 'notification_failed',
               'wandb_url': url, 'artifact': artifact_name, 'slack_completion_status': status,
               'accepted_source_worker_count': 16, 'source_evidence_count': len(proofs),
               'quarantined_workers': 1, 'deferred_workers': 3, 'all_requested_methods_passed': False,
               'publisher_file': str(Path(__file__).resolve()), 'publisher_sha256': file_hash(__file__),
               'report_files': snapshot_files, 'remote_readback': remote,
               'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / 'publication.json', receipt)
    for filename in ('publication.json', 'source_readback.json', 'publication_readback.json'):
        (directory / filename).write_bytes((output / filename).read_bytes())
    print(json.dumps({'status': receipt['status'], 'wandb_url': url, 'slack': status}))
    return 0 if status == 'sent' else 1


def publish(options):
    directory = options.report_dir.resolve()
    report, files = prepare(directory)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = configure_environment()
    if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('Partial publication destination differs from the approved project')
    import wandb
    api = wandb.Api(timeout=60)
    proofs = []
    observations = [*report['sources'], *report['failed_observations']]
    for index, source in enumerate(observations, 1):
        proof = verify_worker(api, source, output / 'readbacks')
        proof['accepted'] = source in report['sources']
        proofs.append(proof)
        write_json(output / 'source_readback.json', {'status': 'running', 'workers': proofs})
        print(f'W&B pilot evidence verified {index}/{len(observations)}', flush=True)
    if report.get('diagnostic_evidence'):
        proofs.append(verify_diagnostic(api, report['diagnostic_evidence'], output / 'readbacks'))
    readback = {'status': 'pass', 'accepted_worker_count': len(report['sources']),
                'quarantined_worker_count': 1, 'diagnostic_worker_count': int(bool(report.get('diagnostic_evidence'))),
                'workers': proofs}
    write_json(output / 'source_readback.json', readback)
    snapshot = output / 'report_snapshot'
    snapshot.mkdir()
    for filename, pin in files.items():
        (snapshot / filename).write_bytes(Path(pin['file']).read_bytes())
    snapshot_files = pins(snapshot / name for name in files)
    if any(snapshot_files[name]['sha256'] != pin['sha256'] or snapshot_files[name]['bytes'] != pin['bytes']
           for name, pin in files.items()):
        raise ValueError('Partial report changed during evidence readback; no publication started')
    summary = scalar_summary(report)
    name = 'peft-base-task1-partial-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = wandb.init(entity='laymond1', project='cl-efficiency', mode='online', name=name,
                     group='peft-common-efficiency', job_type='partial-report', dir=str(output),
                     config={'report_only': True, 'new_training': False, 'all_requested_methods_passed': False,
                             'protocol': report['protocol'], 'report_sha256': files['results.json']['sha256']})
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
    return finalize_saved_publication(directory, output, api)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/peft_pilot/partial-publication-20261008')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--finalize-existing-output', type=Path,
                        help='Verify a previously uploaded report and send completion without creating another run')
    options = parser.parse_args(argv)
    if options.execute:
        if options.finalize_existing_output is not None:
            profile = configure_environment()
            if (profile['wandb_entity'], profile['wandb_project']) != ('laymond1', 'cl-efficiency'):
                raise ValueError('Finalization destination differs from the approved project')
            import wandb
            return finalize_saved_publication(options.report_dir, options.finalize_existing_output, wandb.Api(timeout=60))
        return publish(options)
    report, files = prepare(options.report_dir)
    print(json.dumps({'execute': False, 'method_count': report['method_count'],
                      'status': report['status'], 'files': list(files)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
