"""Publish the inference-only PROL diagnosis after immutable evidence readback.

Dry mode validates local observations without credentials or network access.
Execution verifies the original failed run and the single diagnostic run before
one report-only W&B publication and Slack notification; model weights stay local.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import report_prol_diagnostic as report_module
from scripts import publish_prol_validation as historical_publisher
from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.publish_low_rank_results import manifest_matches
from scripts.publish_peft_partial import verify_report_publication, summary_value_matches
from scripts.report_low_rank_results import run_path

DEFAULT_REPORT = report_module.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / 'results/prol_diagnostic_publication_20261008'
pin = report_module.pin
ensure_no_secrets = historical_publisher.ensure_no_secrets


def prepare(directory):
    directory = Path(directory).resolve()
    report = report_module.read_json(directory / 'results.json')
    expected = report_module.build_report(report['run_root'])
    if set(report) != set(expected) or any(not report_module.strict_equal(report[key], value) for key, value in expected.items() if key != 'generated_at_utc'):
        raise ValueError('PROL diagnostic report differs from saved raw observations')
    if ((directory / 'RESULTS.md').read_text() != report_module.markdown(report)
            or (directory / 'results.csv').read_text() != report_module.csv_text(report)):
        raise ValueError('PROL diagnostic Markdown or CSV changed')
    files = {name: pin(directory / name) for name in report_module.REPORT_FILES}
    files['evidence/execution.json'] = pin(report['execution']['file'])
    record = report_module.read_json(report['source']['file'])
    files['diagnostic/result.json'] = pin(report['source']['file'])
    files.update({f'diagnostic/{name}': proof for name, proof in report_module.worker_files(record, report['run_root']).items()})
    historical = report['historical_failed_source']
    files['historical/result.json'] = pin(historical['file'])
    files.update({f'historical/{name}': proof for name, proof in historical_publisher.worker_files(historical).items()})
    if any(proof['bytes'] > 16 * 1024 * 1024 for proof in files.values()):
        raise ValueError('PROL diagnostic publication contains oversized evidence')
    for name, proof in files.items():
        if Path(name).suffix == '.json':
            ensure_no_secrets(json.loads(Path(proof['file']).read_text()))
    if any(Path(name).suffix not in ('.json', '.md', '.csv', '.npz') or 'tail_inputs' in name for name in files):
        raise ValueError('PROL diagnostic publication must exclude images and model weights')
    return report, files


def scalar_summary(report):
    result = {'report_only': True, 'new_gpu_computation': False, 'new_training': False,
              'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic', 'status': report['status'],
              'new_gpu_processes': 1, 'new_training_runs': 0, 'diagnostic_worker_count': 1,
              'historical_failed_worker_count': 1, 'publication_gate': report['publication_gate'],
              'checkpoint_issue_resolved': False, 'implementation_acceptance_changed': False,
              'new_cl_accuracy': False, 'measurement_aggregate_eligible': False,
              'subsequent_training_authorized': False, 'broad_comparison': False}
    result.update({'counts/' + key: value for key, value in report['counts'].items()})
    result.update({'diagnosis/' + key: value for key, value in report['summary'].items()
                   if type(value) in (int, float, bool, str)})
    return result


def verify_diagnostic(api, report, output):
    source = report['source']
    report_module.check_pin(source, source['file'])
    local = report_module.read_json(source['file'])
    run = api.run(run_path(source['wandb_url']))
    summary = dict(run.summary)
    states = ('finished',) if local['status'] == 'completed' else ('failed', 'finished')
    if (run.state not in states or summary.get('diagnostic_status') != local['status']
            or summary.get('measurement_aggregate_eligible') is not False or summary.get('new_cl_accuracy') is not False):
        raise ValueError('PROL diagnostic remote run must be finished and excluded from CL/efficiency acceptance')
    for key, value in local['counts'].items():
        if not summary_value_matches(summary.get('counts/' + key), value):
            raise ValueError('PROL diagnostic remote computation counts changed')
    files = report_module.worker_files(local, report['run_root'])
    artifacts = [artifact for artifact in run.logged_artifacts()
                 if artifact.state == 'COMMITTED' and set(files).issubset(artifact.manifest.entries)]
    if len(artifacts) != 1:
        raise ValueError('PROL diagnostic needs one unambiguous COMMITTED small evidence artifact')
    artifact = artifacts[0]
    proofs = manifest_matches(artifact, files)
    output = Path(output) / 'diagnostic'
    output.mkdir(parents=True, exist_ok=True)
    remote = report_module.read_json(artifact.get_path('artifact_metrics.json').download(root=str(output)))
    fields = ('method', 'stage', 'status', 'computation_status', 'protocol', 'precision', 'counts',
              'checkpoint_sha256', 'canonical_pretrained_sha256', 'reference_commit', 'origin_execution_commit',
              'prior_result_sha256', 'prior_predictions_sha256', 'prior_checkpoint_sha256',
              'execution_source_sha256', 'reference_source_sha256', 'input_sha256', 'matrix', 'output_files',
              'restoration', 'input_evidence', 'metrics', 'phases', 'measurement_aggregate_eligible',
              'error_type', 'error_context', 'versions')
    if any(not report_module.strict_equal(remote.get(key), local.get(key)) for key in fields):
        raise ValueError('PROL diagnostic remote record semantics differ from local evidence')
    return {'kind': 'inference_diagnostic', 'wandb_url': source['wandb_url'], 'source_sha256': source['sha256'],
            'run_state': run.state, 'local_status': local['status'], 'remote_status': summary.get('diagnostic_status'),
            'artifact_state': artifact.state, 'artifact': artifact.qualified_name,
            'artifact_digest': artifact.digest, 'record_semantics_verified': True, 'counts_summary_verified': True,
            'files': proofs, 'measurement_aggregate_eligible': False, 'new_cl_accuracy': False}


def verify_sources(api, report, output):
    historical = historical_publisher.verify_worker(api, report['historical_failed_source'], Path(output) / 'historical')
    if historical.get('local_status') != 'failed' or historical.get('remote_status') != 'failed':
        raise ValueError('The original failed PROL validation must remain failed')
    diagnostic = verify_diagnostic(api, report, output)
    receipt = {'status': 'pass', 'diagnostic_worker_count': 1, 'historical_failed_worker_count': 1,
               'workers': [historical, diagnostic]}
    validate_source_proofs(receipt, report)
    return receipt


def validate_source_proofs(receipt, report):
    workers = receipt.get('workers', [])
    if (receipt.get('status') != 'pass' or type(receipt.get('diagnostic_worker_count')) is not int
            or receipt['diagnostic_worker_count'] != 1 or type(receipt.get('historical_failed_worker_count')) is not int
            or receipt['historical_failed_worker_count'] != 1 or len(workers) != 2
            or len({proof.get('wandb_url') for proof in workers}) != 2):
        raise ValueError('PROL diagnostic source readback is incomplete or repeated')
    old, new = workers
    if (old.get('wandb_url') != report['historical_failed_source']['wandb_url']
            or old.get('source_sha256') != report['historical_failed_source']['sha256']
            or old.get('local_status') != 'failed' or old.get('remote_status') != 'failed'
            or old.get('run_state') not in ('failed', 'finished')
            or new.get('wandb_url') != report['source']['wandb_url'] or new.get('source_sha256') != report['source']['sha256']
            or new.get('local_status') != report['status'] or new.get('remote_status') != report['status']
            or new.get('run_state') not in (('finished',) if report['status'] == 'completed' else ('failed', 'finished'))
            or new.get('new_cl_accuracy') is not False
            or new.get('counts_summary_verified') is not True
            or any(proof.get('artifact_state') != 'COMMITTED' or proof.get('record_semantics_verified') is not True
                   or not proof.get('files') or any(item.get('matches') is not True for item in proof['files'].values())
                   for proof in workers)):
        raise ValueError('PROL diagnostic source proofs differ from their immutable failed/diagnostic runs')


def finalize(directory, output, api):
    report, originals = prepare(directory)
    output = Path(output).resolve()
    prior = output / 'publication.json'
    if prior.is_file():
        saved = report_module.read_json(prior)
        if saved.get('status') != 'notification_failed' or saved.get('slack_completion_status') == 'sent':
            raise ValueError('Publication already finalized; duplicate notification avoided')
    validate_source_proofs(report_module.read_json(output / 'source_readback.json'), report)
    refreshed = verify_sources(api, report, output / 'final_readbacks')
    validate_source_proofs(refreshed, report)
    files = {name: pin(output / 'report_snapshot' / name) for name in originals}
    if any(proof['sha256'] != originals[name]['sha256'] or proof['bytes'] != originals[name]['bytes'] for name, proof in files.items()):
        raise ValueError('PROL diagnostic immutable publication snapshot changed')
    pending = report_module.read_json(output / 'publication_pending.json')
    url, artifact = pending['wandb_url'], pending['artifact']
    if not url.startswith(report_module.WAND_PREFIX) or not artifact.startswith('laymond1/cl-efficiency/'):
        raise ValueError('PROL diagnostic publication destination changed')
    remote = verify_report_publication(api, url, artifact, files, scalar_summary(report))
    write_json(output / 'publication_readback.json', remote)
    summary = report['summary']
    if report['status'] == 'failed':
        message = (f'PROL 추론 진단 중단: {len(report["matrix"])}개 평가 자료 보존, 단일프로세스, 학습0회. '
                   '원인해결·구현통과판정 변경없음, 추가GPU·자동재실행 없음. 실패증거artifact COMMITTED 확인. W&B ' + url)
    else:
        message = (f'PROL 추론 진단 완료: 40 model evaluations / 80 encoder forwards, 40개이미지 반복, 학습0회. '
                   f'A/B20쌍중logit불일치 {summary["non_bitexact_reference_clone_comparisons"]}쌍. '
                   f'{summary["diagnosis"]}; 원인해결·구현통과판정 변경없음, 후속학습중단 유지. '
                   'original failed run + diagnostic finished / artifact COMMITTED 확인. W&B ' + url)
    slack = notify_slack(message)
    receipt = {'status': 'completed' if slack == 'sent' else 'notification_failed', 'wandb_url': url,
               'artifact': artifact, 'slack_completion_status': slack, 'report_files': files,
               'remote_readback': remote, 'diagnostic_worker_count': 1, 'historical_failed_worker_count': 1,
               'new_training': False, 'checkpoint_issue_resolved': False, 'subsequent_training_authorized': False,
               'publisher_file': str(Path(__file__).resolve()), 'publisher_sha256': report_module.shared.file_hash(__file__),
               'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / 'publication.json', receipt)
    for name in ('publication.json', 'source_readback.json', 'publication_readback.json'):
        (Path(directory) / name).write_bytes((output / name).read_bytes())
    print(json.dumps({'status': receipt['status'], 'wandb_url': url, 'slack': slack}))
    return 0 if slack == 'sent' else 1


def publish(options):
    report, files = prepare(options.report_dir)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = configure_environment()
    if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('PROL diagnostic publication destination differs from approval')
    import wandb
    api = wandb.Api(timeout=60)
    readback = verify_sources(api, report, output / 'readbacks')
    write_json(output / 'source_readback.json', readback)
    snapshot = output / 'report_snapshot'
    for name, proof in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(proof['file']).read_bytes())
        if report_module.shared.file_hash(target) != proof['sha256']:
            raise ValueError('PROL diagnostic evidence changed during snapshot')
    pinned = {name: pin(snapshot / name) for name in files}
    name = 'prol-checkpoint-diagnostic-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = wandb.init(entity='laymond1', project='cl-efficiency', mode='online', name=name,
                     group='prol-checkpoint-diagnostic', job_type='diagnostic-report', dir=str(output),
                     config={'report_only': True, 'new_training': False, 'new_gpu_computation': False,
                             'report_sha256': files['results.json']['sha256'], 'checkpoint_issue_resolved': False})
    try:
        run.summary.update(scalar_summary(report))
        artifact = wandb.Artifact(name, type='cl-diagnostic-report', metadata={'files': {
            key: {'sha256': proof['sha256'], 'bytes': proof['bytes']} for key, proof in pinned.items()}})
        for key, proof in pinned.items():
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
    print(json.dumps({'execute': False, 'small_evidence_files': len(files), 'new_training': False,
                      'diagnostic_source_count': 1, 'historical_failed_source_count': 1}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
