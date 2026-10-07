"""Verify eight completed LoRA runs and publish a report, only with --execute.

Default execution checks local saved evidence without credentials or networking.
Publication verifies finished runs, exact remote counts, committed prediction /
spec manifests, semantic source records and checkpoint digest metadata. It then
uploads immutable small report snapshots, verifies their remote readback, and
sends one final Slack completion. No model, GPU or training is used.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment
from scripts.report_low_rank_results import (
    COUNT_KEYS, REPORT_FILES, artifact_inputs, build_report, csv_text,
    file_hash, markdown, read_json, record_body, run_path,
)


def prepare(report_dir):
    """Recompute the complete report and verify its immutable local inputs."""
    directory = Path(report_dir).resolve()
    pins = {}
    for filename in REPORT_FILES:
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty report artifact: {filename}')
        pins[filename] = file_hash(path)
    report = read_json(directory / 'results.json')
    sources, pairs = report.get('sources', {}), {}
    if len(sources) != 4:
        raise ValueError('Report needs provenance for four reference/native pairs')
    verified = {}
    for method, sides in sources.items():
        if set(sides) != {'reference', 'native'}:
            raise ValueError('Report provenance must contain both sides for each method')
        pairs[method], verified[method] = {}, {}
        for side, source in sides.items():
            path = Path(source.get('file', ''))
            if not path.is_absolute() or not path.is_file() or file_hash(path) != source.get('sha256'):
                raise ValueError('Report provenance file is missing or changed')
            body = record_body(read_json(path))
            pairs[method][side] = body
            verified[method][side] = {'file': str(path), 'sha256': file_hash(path),
                                       'artifacts': artifact_inputs(body, path)}
    expected = build_report(pairs, verified)
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('Report counts, gates or digest provenance differ from pinned input records')
    if report['publication_gate'] != 'pass':
        raise ValueError('Final publication requires all four accuracy/structural/input gates to pass')
    if (directory / 'RESULTS.md').read_text(encoding='utf-8') != markdown(report):
        raise ValueError('Markdown report differs from the exact pinned count rendering')
    if (directory / 'results.csv').read_text(encoding='utf-8') != csv_text(report).replace('\r\n', '\n'):
        raise ValueError('CSV report differs from the exact pinned counts')
    return report, pins, pairs


def file_md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest(), base64.b64encode(digest.digest()).decode('ascii')


def manifest_matches(artifact, inputs):
    entries = artifact.manifest.entries
    proofs = {}
    for filename, pin in inputs.items():
        entry = entries.get(filename)
        if entry is None or getattr(entry, 'ref', None):
            raise ValueError('Remote artifact lacks a directly uploaded required evidence file')
        local = Path(pin['file'])
        if file_hash(local) != pin['sha256'] or local.stat().st_size != pin['bytes']:
            raise ValueError('Local evidence changed during remote verification')
        hexadecimal, encoded = file_md5(local)
        if entry.digest not in (hexadecimal, encoded) or entry.size != pin['bytes']:
            raise ValueError('Remote artifact file digest or size differs from the local evidence')
        proofs[filename] = {'sha256': pin['sha256'], 'wandb_manifest_digest': entry.digest,
                            'bytes': entry.size, 'matches': True}
    return proofs


def exact_remote_count(value, expected):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value == expected


def verify_run(api, method, side, local, provenance, output):
    """Read-only W&B API verification, called solely by authorized publication."""
    path = run_path(local['wandb_url'])
    run = api.run(path)
    if run.state != 'finished':
        raise ValueError('Every source W&B run must be finished before final publication')
    summary = dict(run.summary)
    status_key = 'status' if side == 'reference' else 'native_status'
    if summary.get(status_key) != 'completed':
        raise ValueError('Remote W&B computation status is not completed')
    if side == 'native' and summary.get('acceptance_status') != 'pass':
        raise ValueError('Remote native acceptance is not pass')
    prefix = 'counts' if side == 'reference' else 'native_counts'
    checked_counts = {}
    for key in ('observe_calls', 'optimizer_updates'):
        name = f'{prefix}/{key}'
        if not exact_remote_count(summary.get(name), local['counts'][key]):
            raise ValueError('Remote observe/update summary counts differ from completed local records')
        checked_counts[name] = local['counts'][key]
    for row in local['task_results']:
        for key in COUNT_KEYS:
            name = f'task{row["task"]}/{key}'
            if not exact_remote_count(summary.get(name), row[key]):
                raise ValueError('Remote task accuracy count differs from the exact local count')
            checked_counts[name] = row[key]
    artifacts = provenance['artifacts']
    record_name = artifacts['remote_record_file']
    required = {*artifacts['files'], record_name}
    candidates = [artifact for artifact in run.logged_artifacts()
                  if artifact.state == 'COMMITTED' and required.issubset(artifact.manifest.entries)]
    if len(candidates) != 1:
        raise ValueError('Each run needs one unambiguous committed validation evidence artifact')
    artifact = candidates[0]
    files = manifest_matches(artifact, artifacts['files'])
    destination = Path(output) / (method + '-' + side)
    destination.mkdir(parents=True, exist_ok=True)
    remote_record = record_body(read_json(artifact.get_path(record_name).download(root=str(destination))))
    core = ('method', 'mode', 'status', 'protocol', 'reference_commit', 'checkpoint_sha256',
            'canonical_pretrained_sha256', 'initial_state_sha256', 'initial_learned_sha256',
            'reference_source_sha256', 'execution_source_sha256', 'implementation', 'precision',
            'counts', 'task_results', 'structural_checks')
    if any(remote_record.get(key) != local.get(key) for key in core):
        raise ValueError('Uploaded validation record count/source/gate fields differ from local completion')
    if side == 'native' and remote_record.get('acceptance_status') != local.get('acceptance_status'):
        raise ValueError('Uploaded native record acceptance differs from local completion')
    expected_checkpoints = {f'task{task}': value['sha256']
                            for task, value in artifacts['local_checkpoints'].items()}
    remote_audits = {f'task{row.get("task")}': row.get('checkpoint_sha256')
                     for row in remote_record.get('state_audits', {}).get('tasks', [])}
    metadata = artifact.metadata or {}
    metadata_pins = metadata.get('local_checkpoint_sha256', {})
    catalogs = [catalog for catalog in (remote_audits, metadata_pins) if catalog]
    if not catalogs or any(catalog != expected_checkpoints for catalog in catalogs):
        raise ValueError('Remote validation artifact lacks matching local checkpoint SHA-256 metadata')
    return {'method': method, 'side': side, 'wandb_url': local['wandb_url'], 'run_state': run.state,
            'count_summary_verified': checked_counts, 'artifact': artifact.qualified_name,
            'artifact_state': artifact.state, 'artifact_digest': artifact.digest,
            'files': files, 'record_semantics_verified': True,
            'local_checkpoint_sha256': expected_checkpoints, 'full_checkpoints_uploaded': False}


def scalar_summary(report):
    summary = {'report_only': True, 'new_training': False, 'new_gpu_computation': False,
               'source_run_count': 8, 'validation_method_count': 4,
               'publication_gate': report['publication_gate'], 'efficiency_comparison': False,
               'paper_reproduction': False}
    for method, entry in report['methods'].items():
        summary[f'{method}/acceptance'] = entry['acceptance_status']
        for row in entry['accuracy_metrics']:
            for key in ('reference_correct', 'reference_samples', 'native_correct', 'native_samples',
                        'reference_accuracy_pct', 'native_accuracy_pct', 'drop_pp'):
                summary[f'{method}/{row["metric"]}/{key}'] = row[key]
    return summary


def completion_message(report, url):
    lines = ['Continual LoRA 4개 방법 검증 완료: 저자·구현 실행 8개 finished, 정확도·구조 검사 모두 pass.']
    for entry in report['methods'].values():
        rows = entry['accuracy_metrics']
        task2 = next(row for row in rows if row['metric'] == 'task2_seen')
        lines.append(f'{entry["label"]}: 태스크2 {task2["native_correct"]}/{task2["native_samples"]} '
                     f'({task2["native_accuracy_pct"]:.2f}%), 최대 저자−구현 {max(row["drop_pp"] for row in rows):+.2f}%p.')
    lines += ['공통 저자 코드 변형의 첫 2태스크 검증이며 논문 수치 재현·효율성 파일럿은 아닙니다.', f'W&B {url}']
    return '\n'.join(lines)


def verify_publication(api, run_url, artifact_name, files, expected_summary, timeout):
    deadline = time.monotonic() + timeout
    while True:
        api.flush()
        run = api.run(run_path(run_url))
        if run.state == 'finished':
            artifact = api.artifact(artifact_name)
            if artifact.state == 'COMMITTED':
                if any(dict(run.summary).get(key) != value for key, value in expected_summary.items()):
                    raise ValueError('Finished report run summary differs from exact local counts')
                return {'status': 'pass', 'run_state': run.state, 'artifact_state': artifact.state,
                        'artifact': artifact.qualified_name, 'artifact_digest': artifact.digest,
                        'files': manifest_matches(artifact, files), 'summary_verified': True}
        if time.monotonic() >= deadline:
            raise TimeoutError('Final report run/artifact readback did not reach finished/COMMITTED')
        time.sleep(min(5., max(0., deadline - time.monotonic())))


def publish(options):
    report, pins, pairs = prepare(options.report_dir)
    profile = configure_environment(credentials_path=options.credentials)
    if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('Final LoRA report requires laymond1/cl-efficiency')
    import wandb
    api = wandb.Api(timeout=60)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'failed', 'report_only': True, 'new_training': False, 'new_gpu_computation': False,
              'slack_completion_status': 'not_sent', 'remote_readback_status': 'pending',
              'report_files_sha256': pins, 'summary': scalar_summary(report)}
    write_json(output / 'result.json', record)
    try:
        source_proofs = []
        for method, sides in pairs.items():
            for side, local in sides.items():
                source_proofs.append(verify_run(api, method, side, local, report['sources'][method][side], output / 'readback'))
        write_json(output / 'source_readback.json', {'status': 'pass', 'run_count': 8, 'runs': source_proofs})
        snapshot = output / 'snapshots'
        snapshot.mkdir()
        files = {}
        for filename, pin in pins.items():
            path = snapshot / filename
            shutil.copyfile(options.report_dir / filename, path)
            if file_hash(path) != pin:
                raise ValueError('Report changed during immutable publication snapshot')
            files[filename] = {'file': str(path), 'sha256': pin, 'bytes': path.stat().st_size}
        metadata = {'report_only': True, 'source_run_count': 8, 'efficiency_comparison': False,
                    'paper_reproduction': False, 'report_files_sha256': pins,
                    'reference_commit': report['reference_commit'], 'checkpoint_sha256': report['checkpoint_sha256'],
                    'local_checkpoint_sha256': {method: {side: proof['artifacts']['local_checkpoints']
                                                         for side, proof in sources.items()}
                                                for method, sources in report['sources'].items()}}
        write_json(output / 'artifact_metadata.json', metadata)
        for filename in ('artifact_metadata.json', 'source_readback.json'):
            path = output / filename
            files[filename] = {'file': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
        name = 'low-rank-validation-report-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run = wandb.init(entity='laymond1', project='cl-efficiency', name=name, group='low-rank-first2-results',
                         job_type='report-only', tags=['report-only', 'low-rank', 'first2-validation'],
                         config=metadata, dir=str(output), mode='online')
        record['wandb_url'] = run.url
        run.summary.update(record['summary'])
        artifact = wandb.Artifact(name, type='cl-low-rank-report', metadata=metadata)
        for filename, pin in files.items():
            artifact.add_file(pin['file'], name=filename)
        logged = run.log_artifact(artifact)
        logged.wait()
        if logged.state != 'COMMITTED':
            raise ValueError('Final report artifact did not reach COMMITTED')
        record['wandb_artifact'] = {'name': logged.qualified_name, 'digest': logged.digest}
        wandb.finish(exit_code=0)
        proof = verify_publication(api, record['wandb_url'], logged.qualified_name, files, record['summary'], options.readback_timeout)
        write_json(output / 'publication_readback.json', proof)
        record.update(status='completed', remote_readback_status='pass', wandb_logging_status='uploaded')
        write_json(output / 'result.json', record)
        message = completion_message(report, record['wandb_url'])
        sent = notify_slack(message)
        record['slack_completion_status'] = sent
        write_json(output / 'notification.json', {'status': sent, 'message': message,
                   'publication_readback_sha256': file_hash(output / 'publication_readback.json')})
        if sent != 'sent':
            record['status'] = 'notification_failed'
    except Exception as error:
        record.update(status='publication_failed', error_type=type(error).__name__)
        if getattr(wandb, 'run', None) is not None:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
    write_json(output / 'result.json', record)
    print(json.dumps({'status': record['status'], 'wandb_url': record.get('wandb_url'),
                      'slack_completion_status': record['slack_completion_status'], 'receipt': str(output / 'result.json')}))
    return 0 if record['status'] == 'completed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--readback-timeout', type=float, default=120.)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args(argv)
    if not options.execute:
        report, _, _ = prepare(options.report_dir)
        print(json.dumps({'publication_gate': report['publication_gate'], 'execute': False,
                          'credentials_read': False, 'external_actions': False}))
        return 0
    if not 0 < options.readback_timeout <= 600:
        raise ValueError('Publication readback timeout must be positive and bounded')
    return publish(options)


if __name__ == '__main__':
    raise SystemExit(main())
