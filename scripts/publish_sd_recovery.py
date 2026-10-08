"""Publish the verified recovery comparison and archive existing author evidence.

No training is performed. A separate snapshot contains the nine-method report,
the native gate, preserved failure evidence, and the originally local-only
author result. Every source run and committed artifact is read back first.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.publish_peft_efficiency import verify_worker
from scripts.publish_peft_partial import verify_report_publication
from scripts.run_peft_efficiency import file_hash

DEFAULT_REPORT = ROOT / 'benchmarks/cl_efficiency/sd_recovery'
DEFAULT_OUTPUT = ROOT / 'results/sd_recovery_publication_20261008'


def pin(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def prepare(directory):
    from scripts import report_sd_recovery as report_module
    from scripts import run_sd_recovery as worker
    directory = Path(directory).resolve()
    report = json.loads((directory / 'results.json').read_text())
    expected = report_module.build_report(Path(report['execution']['file']).parent)
    if any(report.get(key) != value for key, value in expected.items() if key != 'generated_at_utc'):
        raise ValueError('Recovery report differs from the saved accepted observations')
    if ((directory / 'RESULTS.md').read_text() != report_module.markdown(report)
            or (directory / 'results.csv').read_text() != report_module.csv_text(report)):
        raise ValueError('Recovery Markdown or CSV differs from saved observations')
    files = {name: pin(directory / name) for name in (*report_module.REPORT_FILES, 'plan.json', 'checks.json', 'README.md')}
    run_root = Path(report['execution']['file']).parent
    for name, path in {'native/acceptance.json': run_root / 'validation/acceptance.json',
                       'native/execution.json': run_root / 'execution.json',
                       'prior/results.json': ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial/results.json',
                       'prior/publication.json': ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial/publication.json',
                       'prior/source_readback.json': ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial/source_readback.json',
                       'prior/publication_readback.json': ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial/publication_readback.json',
                       'prior/failed_sd_result.json': ROOT / 'results/peft_pilot/base-20261008/sd-lora/measurement1/result.json',
                       'cause/summary.json': ROOT / 'results/sd_cause_analysis_20261008/summary.json',
                       'cause/RESULTS.md': ROOT / 'results/sd_cause_analysis_20261008/RESULTS.md',
                       'cause/original_crop_result.json': ROOT / 'results/sd_cause_analysis_20261008/original_crop/result.json'}.items():
        files['evidence/' + name] = pin(path)
    reference = json.loads(worker.AUTHOR_REFERENCE.read_text())
    for name in ('effective_author_args.json', 'epoch_results.json', 'adapter_analysis.json', 'training_trace.json'):
        files['evidence/author/' + name] = pin(worker.AUTHOR_REFERENCE.parent / name)
    for path, digest in worker.author_reference_pins().items():
        path = Path(path)
        if path.parent == worker.AUTHOR_REFERENCE.parent:
            name = 'evidence/author/' + path.name
        elif path == worker.AUTHOR_DRIVER:
            name = 'evidence/author/driver.py'
        else:
            name = 'evidence/author/source/' + str(path.relative_to(reference['reference_source']))
        files[name] = pin(path)
        if files[name]['sha256'] != digest:
            raise ValueError('Author evidence changed during archival preparation')
    if any(item['bytes'] <= 0 for item in files.values()):
        raise ValueError('Required recovery publication evidence is empty')
    return report, files


def scalar_summary(report):
    result = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'method_count': 9, 'publication_gate': 'pass', 'status': 'completed',
              'new_sd_measurement_runs': 3, 'new_sd_flops_runs': 1, 'new_native_validation_runs': 1,
              'author_retrained': False, 'author_reference_originally_local_only': True,
              'archived_existing_author_result': True, 'native_validation_aggregate_eligible': False,
              'author_diagnostic_aggregate_eligible': False, 'equal_compute_budget': False,
              'paper_reproduction': False, 'first_task_only': True,
              'sd-lora/encoder_and_scale_lr': .001, 'sd-lora/classifier_lr': .008,
              'sd-lora/old_failed_accuracy_pct': 10., 'sd-lora/old_failed_aggregate_eligible': False}
    for name, entry in report['methods'].items():
        result[name + '/accuracy_pct'] = entry['accuracy_pct']['mean']
        for key in ('process_wall_s', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib',
                    'process_tree_cpu_s', 'gpu_board_energy_wh'):
            result[name + '/' + key + '_mean'] = entry['resources'][key]['mean']
        result[name + '/partial_flops'] = entry['flops']['supported_ops_total']
    gate = json.loads((Path(report['execution']['file']).parent / 'validation/acceptance.json').read_text())
    result.update({'sd-lora/native_validation_accuracy_pct': gate['accuracy']['native_pct'],
                   'sd-lora/author_accuracy_pct': gate['accuracy']['reference_pct'],
                   'sd-lora/validation_drop_pp': gate['accuracy']['drop_pp'],
                   'sd-lora/network_sha256_matches': gate['network_sha256_matches']})
    return result


def publication_sources(report):
    observations = list(report['sources'])
    root = Path(report['execution']['file']).parent
    for path, pass_kind in [(root / 'validation/result.json', 'validation'),
                            (ROOT / 'results/peft_pilot/base-20261008/sd-lora/measurement1/result.json', 'measurement')]:
        local = json.loads(path.read_text())
        observations.append({'method': 'sd-lora', 'pass': pass_kind, 'file': str(path),
                             'sha256': file_hash(path), 'wandb_url': local['wandb_url']})
    if len(observations) != 22 or len({item['wandb_url'] for item in observations}) != 22:
        raise ValueError('Publication requires 22 distinct saved source runs')
    return observations


def validate_source_proofs(proofs, observations):
    workers = proofs.get('workers', [])
    if (proofs.get('status') != 'pass' or proofs.get('worker_count') != 22 or len(workers) != 22
            or {proof.get('wandb_url') for proof in workers} != {item['wandb_url'] for item in observations}
            or sum(proof.get('role') == 'accepted_cost' for proof in workers) != 20
            or sum(proof.get('role') == 'validation' for proof in workers) != 1
            or sum(proof.get('role') == 'preserved_failure' for proof in workers) != 1
            or any(proof.get('run_state') != 'finished' or proof.get('artifact_state') != 'COMMITTED'
                   or proof.get('record_semantics_verified') is not True for proof in workers)):
        raise ValueError('Recovery source readback is incomplete or differs from saved observations')


def verify_recovery_semantics(api, source, proof, output):
    """The shared verifier predates recovery recipe and actual LR-group audits."""
    local = json.loads(Path(source['file']).read_text())
    if local.get('recovery_recipe') != 'sd-common-lr001':
        return proof
    artifact = api.artifact(proof['artifact'])
    destination = Path(output) / ('recovery-extra-' + Path(source['file']).parent.name)
    destination.mkdir(parents=True, exist_ok=True)
    remote = json.loads(Path(artifact.get_path('metrics.json').download(root=str(destination))).read_text())
    fields = ('recovery_recipe', 'implementation', 'author_reference_sha256', 'recovery_reference_pins',
              'initial_learned_sha256', 'initial_canonical_pretrained_sha256', 'state_audits',
              'first_batch_audits', 'optimizer_group_audits', 'measurement_aggregate_eligible')
    if any(remote.get(key) != local.get(key) for key in fields):
        raise ValueError('Remote SD recovery recipe, LR-group or initialization evidence differs')
    proof['recovery_semantics_verified'] = True
    return proof


def finalize(directory, output, api):
    report, original = prepare(directory)
    output = Path(output).resolve()
    if (output / 'publication.json').exists():
        prior = json.loads((output / 'publication.json').read_text())
        if prior.get('status') != 'notification_failed' or prior.get('slack_completion_status') == 'sent':
            raise ValueError('Already finalized; duplicate notification avoided')
    proofs = json.loads((output / 'source_readback.json').read_text())
    validate_source_proofs(proofs, publication_sources(report))
    snapshot = output / 'report_snapshot'
    files = {name: pin(snapshot / name) for name in original}
    if any(files[name]['sha256'] != source['sha256'] or files[name]['bytes'] != source['bytes'] for name, source in original.items()):
        raise ValueError('Immutable publication snapshot differs from the current report')
    pending = json.loads((output / 'publication_pending.json').read_text())
    url, artifact = pending['wandb_url'], pending['artifact']
    if (not url.startswith('https://wandb.ai/laymond1/cl-efficiency/runs/')
            or not artifact.startswith('laymond1/cl-efficiency/')):
        raise ValueError('Saved publication destination changed')
    readback = verify_report_publication(api, url, artifact, files, scalar_summary(report))
    write_json(output / 'publication_readback.json', readback)
    gate = json.loads((Path(report['execution']['file']).parent / 'validation/acceptance.json').read_text())
    sd = report['methods']['sd-lora']['accuracy_pct']['mean']
    status = notify_slack(f'SD-LoRA 회복 완료: native {gate["accuracy"]["native_pct"]:.1f}% / 저자 {gate["accuracy"]["reference_pct"]:.1f}%, '
                         f'pilot {sd:.1f}%. 검증1+측정3+별도FLOPs1 완료. 9방법 보고서와 기존 저자 증거 보관, '
                         'finished / artifact COMMITTED 검증 완료. 첫 태스크·1epoch 결과입니다. W&B ' + url)
    receipt = {'status': 'completed' if status == 'sent' else 'notification_failed', 'wandb_url': url,
               'artifact': artifact, 'slack_completion_status': status, 'source_worker_count': 22,
               'method_count': 9, 'new_gpu_runs': 5, 'author_retrained': False,
               'report_files': files, 'remote_readback': readback, 'author_reference_originally_local_only': True,
               'publisher_file': str(Path(__file__).resolve()), 'publisher_sha256': file_hash(__file__),
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
        raise ValueError('Recovery publication destination differs from the approved project')
    import wandb
    api = wandb.Api(timeout=60)
    observations = publication_sources(report)
    proofs = []
    for index, source in enumerate(observations, 1):
        proof = verify_worker(api, source, output / 'readbacks')
        proof = verify_recovery_semantics(api, source, proof, output / 'readbacks')
        proof['role'] = 'accepted_cost' if index <= 20 else 'validation' if index == 21 else 'preserved_failure'
        proofs.append(proof)
        write_json(output / 'source_readback.json', {'status': 'running', 'workers': proofs})
        print(f'W&B source verified {index}/22', flush=True)
    write_json(output / 'source_readback.json', {'status': 'pass', 'worker_count': 22, 'accepted_cost_worker_count': 20,
                                               'native_validation_count': 1, 'preserved_failure_count': 1, 'workers': proofs})
    snapshot = output / 'report_snapshot'
    for name, source in files.items():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source['file']).read_bytes())
        if file_hash(destination) != source['sha256']:
            raise ValueError('Recovery publication evidence changed during snapshot')
    snapshot_files = {name: pin(snapshot / name) for name in files}
    name = 'sd-recovery-nine-method-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = wandb.init(entity='laymond1', project='cl-efficiency', mode='online', name=name,
                     group='peft-common-efficiency', job_type='recovery-report', dir=str(output),
                     config={'report_only': True, 'new_training': False, 'author_retrained': False,
                             'report_sha256': files['results.json']['sha256']})
    try:
        run.summary.update(scalar_summary(report))
        artifact = wandb.Artifact(name, type='cl-efficiency-report', metadata={'files': {
            key: {'sha256': value['sha256'], 'bytes': value['bytes']} for key, value in snapshot_files.items()}})
        for key, value in snapshot_files.items():
            artifact.add_file(value['file'], name=key)
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
    print(json.dumps({'execute': False, 'method_count': report['method_count'], 'archival_files': len(files), 'new_training': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
