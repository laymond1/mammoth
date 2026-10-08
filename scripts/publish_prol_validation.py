"""Publish an audited six-run PROL stage, only with --execute.

Default operation checks local evidence without reading credentials or making
network requests. Publication verifies every terminal worker and COMMITTED
artifact before uploading immutable small results and sending Slack completion.
No training, raw checkpoint upload, or broad method comparison is performed.
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
from scripts.publish_low_rank_results import manifest_matches
from scripts.publish_peft_partial import verify_report_publication
from scripts.report_low_rank_results import run_path
from scripts import report_prol_validation as report_module

DEFAULT_REPORT = report_module.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / 'results/prol_validation_publication_20261008'
pin = report_module.pin


def ensure_no_secrets(value):
    """Fail without echoing the value if small records contain credentials."""
    private_keys = {'api_key', 'wandb_api_key', 'slack_webhook_url', 'authorization',
                    'access_token', 'refresh_token', 'password', 'secret'}
    if isinstance(value, dict):
        if any(str(key).lower() in private_keys for key in value):
            raise ValueError('PROL publication evidence contains a private credential field')
        for item in value.values():
            ensure_no_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            ensure_no_secrets(item)
    elif isinstance(value, str) and ('hooks.slack.com/services/' in value
                                     or value.lower().startswith(('--wandb_api_key', '--slack_webhook_url'))):
        raise ValueError('PROL publication evidence contains a private credential value')


def worker_files(source):
    """Explicit small evidence allowlist; initial/final model files stay local."""
    directory = Path(source['file']).resolve().parent
    record = report_module.read_json(source['file'])
    required = ['spec.json', 'artifact_metrics.json', 'telemetry.csv', 'phases.json',
                'epoch_results.json', 'task_results.json', 'first_batch_audits.json', 'state_audits.json',
                'method_audits.json', 'lr_audits.json', 'routing_audits.json', 'sample_stream_audits.json',
                'effective_args.json', 'shared_data.json', 'source_adaptations.json']
    if record.get('status') != 'completed':
        # Early guards can run before model/data construction. Preserve every
        # available allowlisted audit, while requiring the immutable main record.
        required = ['spec.json', 'artifact_metrics.json'] + [name for name in required[2:] if (directory / name).is_file()]
    required.extend(f'task{task}_predictions.npz' for task in range(1, len(record.get('task_results', [])) + 1))
    required.extend(row['gradient_file'] for row in record.get('first_batch_audits', []))
    if record['pass'] == 'flops' and record.get('status') == 'completed':
        required.append('flops_ops.json')
    optional = ('model_snapshots.json', 'optimizer_group_audits.json', 'data_snapshots.json',
                'structural_checks.json', 'reference_source_sha256.json', 'parameter_audits.json', 'error_context.json')
    required.extend(name for name in optional if (directory / name).is_file())
    return {name: pin(directory / name) for name in required}


def prepare(directory):
    directory = Path(directory).resolve()
    report = report_module.read_json(directory / 'results.json')
    expected = report_module.build_report(report['run_root'])
    if (set(report) != set(expected) or any(report[key] != value for key, value in expected.items() if key != 'generated_at_utc')):
        raise ValueError('PROL report differs from the saved accepted stage')
    if ((directory / 'RESULTS.md').read_text() != report_module.markdown(report)
            or (directory / 'results.csv').read_text() != report_module.csv_text(report)):
        raise ValueError('PROL Markdown or CSV differs from saved observations')
    files = {name: pin(directory / name) for name in report_module.REPORT_FILES}
    files['evidence/execution.json'] = pin(report['execution']['file'])
    for kind, validation in report['validation'].items():
        files[f'evidence/{kind}_acceptance.json'] = pin(validation['acceptance_file']['file'])
        files['evidence/normality_review.json'] = pin(validation['normality_review_file']['file'])
    for name, receipt in report.get('receipts', {}).items():
        files[f'evidence/{name}.json'] = pin(receipt['file']['file'])
    for source in report['sources']:
        name = source['name']
        files[f'evidence/{name}/result.json'] = pin(source['file'])
        files.update({f'evidence/{name}/{filename}': proof for filename, proof in worker_files(source).items()})
    if any(proof['bytes'] > 16 * 1024 * 1024 for proof in files.values()):
        raise ValueError('PROL publication includes an unexpectedly large results artifact')
    for name, proof in files.items():
        if Path(name).suffix == '.json':
            ensure_no_secrets(json.loads(Path(proof['file']).read_text()))
    return report, files


def scalar_summary(report):
    result = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'method': 'prol-prompt', 'source_mode': 'official-active-path', 'publication_gate': report['publication_gate'],
              'stage_status': report['status'], 'source_worker_count': len(report['sources']),
              'new_validation_runs': report['new_validation_runs'], 'new_measurement_runs': report['new_measurement_runs'],
              'new_flops_runs': report['new_flops_runs'],
              'paper_reproduction': False, 'broad_comparison': False,
              'changed_pilot_recipe': False, 'pilot_epochs': 1,
              'validation_costs_aggregate_eligible': False, 'flops_costs_aggregate_eligible': False,
              'equal_compute_budget': False, 'full_machine_energy_measured': False}
    for kind, validation in report['validation'].items():
        result[kind + '/acceptance'] = validation['acceptance']['status']
        for metric in validation['acceptance']['accuracy_metrics']:
            for key in ('reference_pct', 'native_pct', 'drop_pp'):
                result[f'{kind}/{metric["metric"]}/{key}'] = metric[key]
    pilot = report['pilot']
    if pilot is None:
        result.update(efficiency_averages_available=False, failed_results_aggregate_eligible=False)
        for row in report['diagnostics']:
            result[row['name'] + '/status'] = row['status']
            result[row['name'] + '/cost_audit'] = row['cost_audit']['status']
            for key, value in (row.get('accuracy_pct') or {}).items():
                result[row['name'] + '/' + key] = value
            for key, value in row.get('metrics', {}).items():
                if type(value) in (int, float) and math.isfinite(value):
                    result['observed_failed_or_incomplete/' + row['name'] + '/' + key] = value
        return result
    result['efficiency_averages_available'] = True
    result['pilot/accuracy_pct_mean'] = pilot['accuracy_pct']['mean']
    for key, values in pilot['resources'].items():
        if values['mean'] is not None:
            result['pilot/' + key + '_mean'] = values['mean']
        if values['std_sample'] is not None:
            result['pilot/' + key + '_std_sample'] = values['std_sample']
    result['pilot/partial_supported_ops_total'] = pilot['flops']['supported_ops_total']
    return result


def verify_worker(api, source, output):
    local = report_module.read_json(source['file'])
    report_module.check_pin(source, source['file'])
    run = api.run(run_path(source['wandb_url']))
    summary = dict(run.summary)
    completed = local.get('status') == 'completed'
    states = ('finished',) if completed else ('failed', 'finished')
    if (run.state not in states or summary.get('benchmark_status') != local.get('status')
            or summary.get('benchmark_pass') != source['pass']):
        raise ValueError('Every PROL worker must be terminal and match its local completion state')
    checked = {}
    for key in ('tasks_completed', 'outer_observe_steps', 'outer_training_stream_examples', 'evaluation_examples'):
        value = summary.get('efficiency/' + key)
        if type(value) not in (int, float) or value != local['metrics'][key]:
            raise ValueError('Remote PROL stream counts differ from local evidence')
        checked['efficiency/' + key] = local['metrics'][key]
    value, expected_accuracy = summary.get('accuracy/mean_accuracy_pct'), local.get('accuracy', {}).get('mean_accuracy_pct')
    if (expected_accuracy is None and value is not None or expected_accuracy is not None
            and (type(value) not in (int, float) or not math.isclose(value, expected_accuracy, rel_tol=0, abs_tol=1e-10))):
        raise ValueError('Remote PROL accuracy differs from local predictions')
    files = worker_files(source)
    artifacts = [artifact for artifact in run.logged_artifacts()
                 if artifact.state == 'COMMITTED' and set(files).issubset(artifact.manifest.entries)]
    if len(artifacts) != 1:
        raise ValueError('PROL worker requires one unambiguous COMMITTED evidence artifact')
    artifact = artifacts[0]
    proofs = manifest_matches(artifact, files)
    destination = Path(output) / source['name']
    destination.mkdir(parents=True, exist_ok=True)
    remote = report_module.read_json(artifact.get_path('artifact_metrics.json').download(root=str(destination)))
    fields = ('model', 'method', 'side', 'mode', 'pass', 'implementation', 'status', 'computation_status',
              'protocol', 'checkpoint_sha256', 'canonical_pretrained_sha256', 'reference_commit',
              'execution_source_sha256', 'precision', 'counts', 'accuracy', 'metrics', 'structural_checks',
              'data_snapshots', 'task_results', 'epoch_results', 'initial_state_sha256', 'initial_learned_sha256',
              'first_batch_audits', 'state_audits', 'method_audits', 'lr_audits', 'routing_audits', 'sample_stream_audits',
              'reference_source_sha256', 'initial_canonical_pretrained_branches',
              'initial_canonical_pretrained_sha256', 'initial_frozen_encoder', 'initial_frozen_parameters_sha256',
              'measurement_aggregate_eligible', 'flops',
              'error_type', 'error_context', 'parameter_audits', 'source_adaptations')
    if any(not report_module.strict_equal(remote.get(key), local.get(key)) for key in fields):
        raise ValueError('Remote PROL evidence semantics differ from the completed local worker')
    return {'name': source['name'], 'method': source['method'], 'pass': source['pass'],
            'mode': source['mode'], 'side': source['side'], 'source_sha256': source['sha256'],
            'wandb_url': source['wandb_url'], 'run_state': run.state, 'local_status': local.get('status'),
            'remote_status': summary.get('benchmark_status'), 'artifact_state': artifact.state,
            'artifact': artifact.qualified_name, 'artifact_digest': artifact.digest,
            'count_summary_verified': checked, 'accuracy_summary_verified': True,
            'record_semantics_verified': True, 'files': proofs}


def validate_source_proofs(proofs, sources):
    workers = proofs.get('workers', [])
    if (proofs.get('status') != 'pass' or type(proofs.get('worker_count')) is not int
            or not sources or len(sources) > 6 or proofs['worker_count'] != len(sources) or len(workers) != len(sources)
            or len({item.get('wandb_url') for item in workers}) != len(sources)):
        raise ValueError('PROL source readback is incomplete or repeated')
    for source, proof in zip(sources, workers):
        if (any(proof.get(key) != source.get(key) for key in ('name', 'method', 'mode', 'pass', 'side', 'wandb_url'))
                or proof.get('source_sha256') != source['sha256']
                or proof.get('local_status') not in ('completed', 'failed', 'stopped')
                or proof.get('remote_status') != proof.get('local_status')
                or proof.get('run_state') not in (('finished',) if proof.get('local_status') == 'completed' else ('finished', 'failed'))
                or proof.get('artifact_state') != 'COMMITTED' or proof.get('record_semantics_verified') is not True
                or proof.get('accuracy_summary_verified') is not True or not proof.get('count_summary_verified')
                or not proof.get('files') or any(item.get('matches') is not True for item in proof['files'].values())):
            raise ValueError('PROL source readback differs from the audited observations')


def finalize(directory, output, api):
    report, originals = prepare(directory)
    output = Path(output).resolve()
    prior_path = output / 'publication.json'
    if prior_path.exists():
        prior = report_module.read_json(prior_path)
        if prior.get('status') != 'notification_failed' or prior.get('slack_completion_status') == 'sent':
            raise ValueError('Already finalized; duplicate notification avoided')
    source_readback = report_module.read_json(output / 'source_readback.json')
    validate_source_proofs(source_readback, report['sources'])
    # Recheck finished source runs on a finalization retry; do not trust stale
    # local proof alone when retrying a previously failed Slack completion.
    refreshed = [verify_worker(api, source, output / 'final_readbacks') for source in report['sources']]
    validate_source_proofs({'status': 'pass', 'worker_count': len(report['sources']), 'workers': refreshed}, report['sources'])
    files = {name: pin(output / 'report_snapshot' / name) for name in originals}
    if any(proof['sha256'] != originals[name]['sha256'] or proof['bytes'] != originals[name]['bytes'] for name, proof in files.items()):
        raise ValueError('Immutable PROL publication snapshot changed')
    pending = report_module.read_json(output / 'publication_pending.json')
    url, artifact = pending['wandb_url'], pending['artifact']
    if not url.startswith(report_module.WAND_PREFIX) or not artifact.startswith('laymond1/cl-efficiency/'):
        raise ValueError('Saved PROL publication destination changed')
    remote = verify_report_publication(api, url, artifact, files, scalar_summary(report))
    write_json(output / 'publication_readback.json', remote)
    if report['pilot'] is None:
        message = (f'PROL 검증 중단: {len(report["sources"])}회 실행의 진단 자료를 보존했습니다. '
                   '효율성 평균 없음, 추가 GPU 실행 없음. terminal worker / artifact COMMITTED 확인. W&B ' + url)
    else:
        message = (f'PROL 구현 검증 완료: 첫2태스크1epoch 저자/native 대조 및 old/new·loss 정상성 점검 통과. '
                   f'첫태스크 측정3회 평균 정확도 {report["pilot"]["accuracy_pct"]["mean"]:.2f}%, '
                   '별도부분FLOPs1회, 총6실행 finished / artifact COMMITTED 확인. 공식 활성 경로. W&B ' + url)
    status = notify_slack(message)
    receipt = {'status': 'completed' if status == 'sent' else 'notification_failed', 'wandb_url': url,
               'artifact': artifact, 'slack_completion_status': status, 'source_worker_count': len(report['sources']),
               'method': 'prol-prompt', 'stage_status': report['status'], 'publication_gate': report['publication_gate'],
               'broad_comparison': False, 'report_files': files,
               'remote_readback': remote, 'publisher_file': str(Path(__file__).resolve()),
               'publisher_sha256': report_module.file_hash(__file__),
               'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / 'publication.json', receipt)
    for name in ('publication.json', 'source_readback.json', 'publication_readback.json'):
        (Path(directory) / name).write_bytes((output / name).read_bytes())
    print(json.dumps({'status': receipt['status'], 'wandb_url': url, 'slack': status}))
    return 0 if status == 'sent' else 1


def publish(options):
    report, files = prepare(options.report_dir)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = configure_environment()
    if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('PROL destination differs from the approved project')
    import wandb
    api = wandb.Api(timeout=60)
    proofs = []
    for source in report['sources']:
        proofs.append(verify_worker(api, source, output / 'readbacks'))
        write_json(output / 'source_readback.json', {'status': 'running', 'workers': proofs})
    readback = {'status': 'pass', 'worker_count': len(report['sources']), 'workers': proofs}
    validate_source_proofs(readback, report['sources'])
    write_json(output / 'source_readback.json', readback)
    snapshot = output / 'report_snapshot'
    for name, proof in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(proof['file']).read_bytes())
        if report_module.file_hash(target) != proof['sha256']:
            raise ValueError('PROL evidence changed during publication snapshot')
    snapshot_files = {name: pin(snapshot / name) for name in files}
    name = 'prol-validation-stage-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = wandb.init(entity='laymond1', project='cl-efficiency', mode='online', name=name,
                     group='prol-validation', job_type='validation-report', dir=str(output),
                     config={'report_only': True, 'new_training': False, 'source_mode': 'official-active-path',
                             'changed_pilot_recipe': False, 'report_sha256': files['results.json']['sha256']})
    try:
        run.summary.update(scalar_summary(report))
        artifact = wandb.Artifact(name, type='cl-validation-report', metadata={'files': {
            key: {'sha256': proof['sha256'], 'bytes': proof['bytes']} for key, proof in snapshot_files.items()}})
        for key, proof in snapshot_files.items():
            artifact.add_file(proof['file'], name=key)
        logged = run.log_artifact(artifact)
        logged.wait()
        write_json(output / 'publication_pending.json', {'wandb_url': run.url, 'artifact': logged.qualified_name})
    finally:
        run.finish()
    return finalize(options.report_dir, output, api)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--finalize-existing-output', type=Path)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if options.execute and options.finalize_existing_output:
        configure_environment()
        import wandb
        return finalize(options.report_dir, options.finalize_existing_output, wandb.Api(timeout=60))
    if options.execute:
        return publish(options)
    report, files = prepare(options.report_dir)
    print(json.dumps({'execute': False, 'source_worker_count': len(report['sources']), 'small_results_files': len(files), 'new_training': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
