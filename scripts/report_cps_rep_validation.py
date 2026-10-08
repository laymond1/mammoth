"""Audit local five-run evidence; optionally verify/publish W&B and Slack.

No training or GPU work. --publish is explicit; default only writes local reports.
"""
import argparse
import base64
import csv
import datetime
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import notify_slack, write_json
from scripts.cl_runtime import configure_environment, configure_native_environment
from scripts.cps_author_reference import file_hash
from scripts.cps_rep_protocol import RUNS, compare_records, protocol

configure_native_environment()


def read_json(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}


def build_report(root):
    import numpy as np
    root = Path(root).resolve()
    records, rows = {}, []
    for name in RUNS:
        directory = root / name
        record = read_json(directory / 'result.json')
        if (record['status'] != 'completed' or record['structural_checks_passed'] is not True
                or record['protocol'] != protocol() or record['cpu_smoke'] is not False
                or len(record['task_results']) != 2 or len(record['epoch_results']) != 6
                or len(record['epoch_inputs']) != 6):
            raise ValueError(f'Incomplete or non-approved GPU validation evidence: {name}')
        metrics = record['metrics']
        for key, expected in (('tasks_completed', 2), ('outer_observe_steps', 474),
                              ('outer_training_stream_examples', 30000), ('evaluation_examples', 3000)):
            if metrics[key] != expected:
                raise ValueError('Full stream counts differ from approved execution')
        for task in record['task_results']:
            with np.load(directory / f'task{task["task"]}_predictions.npz', allow_pickle=False) as predictions:
                labels, predicted = predictions['labels'], predictions['predictions']
                old = labels < (task['task'] - 1) * 10
                if not np.array_equal(predicted, predictions['logits'].argmax(1)) or len(labels) != task['task'] * 1000:
                    raise ValueError('Saved prediction logits or counts disagree')
                actual = {'seen_accuracy_pct': float(100 * (predicted == labels).mean()),
                          'new_accuracy_pct': float(100 * (predicted[~old] == labels[~old]).mean()),
                          'old_accuracy_pct': float(100 * (predicted[old] == labels[old]).mean()) if old.any() else None}
                if any(task[key] != value for key, value in actual.items()):
                    raise ValueError('Accuracy report differs from actual saved predictions')
        records[name] = record
        rows.append({'run': name, 'task1_accuracy_pct': record['task_results'][0]['seen_accuracy_pct'],
                     'task2_seen_accuracy_pct': record['task_results'][1]['seen_accuracy_pct'],
                     'task2_old_accuracy_pct': record['task_results'][1]['old_accuracy_pct'],
                     'task2_new_accuracy_pct': record['task_results'][1]['new_accuracy_pct'],
                     'validation_wall_s': metrics['process_wall_s'],
                     'train_epochs_wall_s': record['phases']['train_epoch']['wall_s'],
                     'cuda_peak_allocated_mib': metrics['cuda_peak_allocated_mib'],
                     'cuda_peak_reserved_mib': metrics['cuda_peak_reserved_mib'],
                     'gpu_board_energy_j': metrics['gpu_board_energy_j'],
                     'gpu_energy_coverage_fraction': metrics['gpu_energy_coverage_fraction'],
                     'peak_process_tree_rss_mib': metrics['peak_process_tree_rss_mib'],
                     'wandb_url': record['wandb_url'], 'result_pin': pin(directory / 'result.json')})
    comparisons = {'cps-native': ('cps-author', False), 'rep-off': ('coda-frozen', False), 'rep-on': ('coda-frozen', True)}
    gates = {name: compare_records(records[name], records[reference], decline_only=decline)
             for name, (reference, decline) in comparisons.items()}
    paired_predictions = {}
    for name, reference in (('cps-native', 'cps-author'), ('rep-off', 'coda-frozen')):
        audits = []
        for task in (1, 2):
            with np.load(root / name / f'task{task}_predictions.npz') as actual, np.load(root / reference / f'task{task}_predictions.npz') as expected:
                audits.append({'task': task, 'identical_predictions': bool(np.array_equal(actual['predictions'], expected['predictions'])),
                               'maximum_absolute_logit_difference': float(np.abs(actual['logits'] - expected['logits']).max())})
        paired_predictions[name] = audits
    summary = read_json(root / 'summary.json')
    if summary['completed_runs'] != 5 or summary['acceptance'] != gates:
        raise ValueError('Controller summary differs from recomputed acceptance')
    return {'schema': 'cps-rep-verification-report-v1', 'protocol': protocol(), 'run_root': str(root),
            'rows': rows, 'acceptance': gates, 'paired_prediction_audits': paired_predictions,
            'rep_diagnostic': {
                'status': 'hold' if gates['rep-on']['status'] == 'hold' else 'completed',
                'cause_isolated': False,
                'facts': ['REP OFF has identical logits and predictions to frozen-CODA on both tasks.',
                          'REP ON uses the same complete input stream, initial prompt/head and canonical host/query weights.',
                          'Finite loss/state/gradient, old prompt row protection and checkpoint assertions passed.',
                          'All three REP components are enabled together, so this stage cannot assign the decline to one component.'],
                'epoch_observations': [{'task': row['task'], 'epoch': row['epoch'], 'loss': row['loss'],
                                        'executed_blocks_mean': row['diagnostics']['executed'],
                                        'merged_tokens_mean': row['diagnostics']['merged_tokens'],
                                        'ald_step_mean': row['diagnostics']['step']}
                                       for row in records['rep-on']['epoch_results']],
                'next_recommendation': 'Review paper-to-code interpretations and test Swift/AToM/ALD separately on the same Tiny two-task protocol before accepting REP or launching its efficiency pilots.',
                'additional_training_executed': False,
            },
            'source_summary': pin(root / 'summary.json'),
            'initial_manifest': read_json(root / 'initial_manifest.json'),
            'structural_checks': ['full two-task sample and update counts', 'all six epoch input hashes and raw dataset fingerprints',
                                  'same prompt/head initialization and paired physical GPU', 'finite loss/state/gradients',
                                  'frozen encoders unchanged', 'previous prompt rows unchanged', 'checkpoint round trip',
                                  'saved class-IL logits/predictions equal native evaluation'],
            'limitations': ['One seed and three epochs per task; implementation verification, not paper accuracy reproduction.',
                            'Resource costs include loader worker startup, complete input hashing and validation audits; not an efficiency pilot.',
                            'Runs on GPU0 and GPU1 overlap; CPU contention and GPU-board idle power are included.',
                            'GPU board energy excludes CPU/RAM/wall power. No FLOPs profiler or whole-machine energy measurement was performed.',
                            'REP is a documented paper-v6 reconstruction, without an independent full author runner.',
                            'Swift uses a separate Tiny query even for a Tiny host; there is no smaller-query FLOP saving at this size.',
                            '20epoch pilots, Small/Base and batch16/64 expansion, and the held11method comparison were not launched.']}


def write_report(report, output):
    output = Path(output)
    write_json(output / 'results.json', report)
    buffer = io.StringIO(newline='')
    rows = [{key: value for key, value in row.items() if key != 'result_pin'} for row in report['rows']]
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (output / 'results.csv').write_text(buffer.getvalue())
    lines = ['# CPS-Prompt / REP Tiny 구현 검증', '',
             'CIFAR100-224 unified · Tiny AugReg IN21K→IN1K · batch64 · seed1993 · FP32 · 첫2태스크 × 3epochs.', '',
             '| 실행 | Task1 정확도 | Task2 seen | Task2 old | Task2 new |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| [{row["run"]}]({row["wandb_url"]}) | {row["task1_accuracy_pct"]:.2f}% | {row["task2_seen_accuracy_pct"]:.2f}% | '
                     f'{row["task2_old_accuracy_pct"]:.2f}% | {row["task2_new_accuracy_pct"]:.2f}% |')
    lines.extend(['', ('CPS native↔저자 및 REP OFF↔frozen-CODA는 1.5%p 이내 통과, 1.5~3%p 점검, 3%p 초과 실패. '
                  'REP ON은 seen·old·new 정확도에서 frozen-CODA보다 3%p 초과 하락 시 후속 진행을 보류한다.'), '',
                  '| 판정 | 상태 | 최대 정확도 차이/하락 |', '| --- | --- | ---: |'])
    for name, gate in report['acceptance'].items():
        value = max((max(0., -row['delta_pp']) if name == 'rep-on' else abs(row['delta_pp'])) for row in gate.get('metrics', [])) if gate.get('metrics') else None
        lines.append(f'| {name} | {gate["status"]} | {value:.2f}%p |' if value is not None else f'| {name} | {gate["status"]} | 구조/조건 불일치 |')
    deltas = ', '.join(f'task{row["task"]} {row["metric"].removesuffix("_accuracy_pct")} {row["delta_pp"]:+.2f}%p'
                       for row in report['acceptance']['rep-on'].get('metrics', []) if row['metric'] != 'new_accuracy_pct' or row['task'] != 1)
    observations = report['rep_diagnostic']['epoch_observations']
    executed = [row['executed_blocks_mean'] for row in observations]
    merged = [row['merged_tokens_mean'] for row in observations]
    lines.extend(['', (f'REP ON의 차이는 {deltas}이며 판정은 {report["acceptance"]["rep-on"]["status"]}다. '
                  '공통 입력·초기화·고정 encoder·유한 상태·과거 prompt 보존 검증은 통과했다.'), '',
                  (f'REP ON의 평균 실행 블록 수는 {min(executed):.2f}~{max(executed):.2f}/12이며, '
                   f'평균 병합 수는 {min(merged):.2f}~{max(merged):.2f} tokens다. '
                   '세 구성요소를 함께 변경했고 3 epochs만 검증했으므로 하락 원인은 분리되지 않았다. '
                   '논문 해석/연결을 재점검하고 Swift·AToM·ALD별 검증을 먼저 하는 것이 다음 추천 단계다. 추가 학습은 실행하지 않았다.')])
    lines.extend(['', '다음 값은 이번 검증 실행의 관측치이며 효율성 pilot 결과가 아니다. 프로세스 시간은 모델 초기화 이후부터 검증·기록을 포함한다.', '',
                  '| 실행 | 검증 시간(s) | 학습 epoch 시간(s) | CUDA peak allocated(MiB) | GPU 보드 에너지(kJ) |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for row in rows:
        energy = f'{row["gpu_board_energy_j"] / 1000:.2f}' if row['gpu_board_energy_j'] is not None else '미측정'
        lines.append(f'| {row["run"]} | {row["validation_wall_s"]:.1f} | {row["train_epochs_wall_s"]:.1f} | {row["cuda_peak_allocated_mib"]:.1f} | {energy} |')
    lines.extend(['', '## 범위와 제한', '', *['- ' + value for value in report['limitations']], '',
                  '전체 입력·예측·설정·전력 샘플·태스크별 체크포인트는 실행별 로컬 폴더와 W&B artifact에 저장한다.', ''])
    (output / 'RESULTS.md').write_text('\n'.join(lines))


def verify_manifest(artifact, files):
    proofs = {}
    for name, proof in files.items():
        entry = artifact.manifest.entries[name]
        digest = hashlib.md5(usedforsecurity=False)
        with Path(proof['file']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(2 ** 20), b''):
                digest.update(chunk)
        if entry.size != proof['bytes'] or entry.digest not in (digest.hexdigest(), base64.b64encode(digest.digest()).decode()):
            raise ValueError('Committed artifact file differs from local evidence')
        proofs[name] = {'sha256': proof['sha256'], 'bytes': proof['bytes'], 'remote_digest': entry.digest, 'matches': True}
    return proofs


def verify_workers(api, report, output):
    proofs = []
    root = Path(report['run_root'])
    files = ('spec.json', 'epoch_results.json', 'epoch_inputs.json', 'task_results.json', 'data_fingerprints.json',
             'metrics.json', 'phases.json', 'task1_predictions.npz', 'task2_predictions.npz', 'task1_state.pt', 'task2_state.pt')
    fields = ('run', 'protocol', 'status', 'initial_learned_sha256', 'physical_gpu_uuid', 'task_results', 'epoch_results',
              'epoch_inputs', 'data_fingerprints', 'task_initialization', 'metrics', 'sources', 'author_sources', 'structural_checks_passed')
    for row in report['rows']:
        directory = root / row['run']
        local = read_json(directory / 'result.json')
        run = api.run(row['wandb_url'].split('wandb.ai/', 1)[1].replace('/runs/', '/'))
        summary = dict(run.summary)
        if run.state != 'finished' or summary.get('status') != 'completed' or summary.get('structural_checks_passed') is not True:
            raise ValueError('Worker is not remotely finished and structurally completed')
        for key in ('tasks_completed', 'outer_observe_steps', 'outer_training_stream_examples', 'evaluation_examples'):
            if summary.get('resources/' + key) != local['metrics'][key]:
                raise ValueError('Remote stream summary count differs from local result')
        if summary.get('task/seen_accuracy_pct') != local['task_results'][-1]['seen_accuracy_pct']:
            raise ValueError('Remote accuracy summary differs from saved predictions')
        candidates = [artifact for artifact in run.logged_artifacts()
                      if artifact.state == 'COMMITTED' and {*files, 'result.json'}.issubset(artifact.manifest.entries)]
        if len(candidates) != 1:
            raise ValueError('Expected one unambiguous committed worker evidence artifact')
        artifact = candidates[0]
        file_proofs = verify_manifest(artifact, {name: pin(directory / name) for name in files})
        remote = read_json(artifact.get_path('result.json').download(root=str(Path(output) / row['run'])))
        # Final Slack completion is appended after artifact upload; compare the
        # immutable computation/evidence fields, not that notification receipt.
        if any(remote.get(key) != local.get(key) for key in fields):
            raise ValueError('Committed worker record semantics differ from local completion')
        proofs.append({'run': row['run'], 'wandb_url': row['wandb_url'], 'run_state': run.state,
                       'artifact': artifact.qualified_name, 'artifact_state': artifact.state,
                       'record_semantics_match': True, 'files': file_proofs})
    return proofs


def publish(report, output):
    profile = configure_environment()
    if profile['wandb_entity'] != 'laymond1' or profile['wandb_project'] != 'cl-efficiency':
        raise ValueError('Publication destination differs from approved project')
    import wandb
    api = wandb.Api(timeout=45)
    directory = Path(report['run_root']) / 'publication'
    directory.mkdir(parents=True, exist_ok=True)
    proofs = verify_workers(api, report, directory / 'readbacks')
    write_json(output / 'worker_readback.json', {'status': 'pass', 'workers': proofs})
    with wandb.init(entity='laymond1', project='cl-efficiency', name='cps-rep-validation-report-20261009',
                    group='cps-rep-' + Path(report['run_root']).name, job_type='validation-report', mode='online',
                    dir=str(directory), config={'report_only': True, 'new_training': False, 'protocol': protocol()}) as run:
        url = run.url
        run.summary.update({'status': 'completed', 'verified_worker_count': 5, 'paper_reproduction': False,
                            'efficiency_pilot': False, **{name + '/acceptance': gate['status'] for name, gate in report['acceptance'].items()}})
        run.log({'results': wandb.Table(columns=['run', 'task1_pct', 'task2_seen_pct', 'task2_old_pct', 'task2_new_pct'],
            data=[[row['run'], row['task1_accuracy_pct'], row['task2_seen_accuracy_pct'], row['task2_old_accuracy_pct'], row['task2_new_accuracy_pct']]
                  for row in report['rows']])})
        artifact = wandb.Artifact('cps-rep-validation-report-20261009', type='validation-report')
        files = {name: pin(Path(output) / name) for name in ('results.json', 'results.csv', 'RESULTS.md', 'worker_readback.json')}
        # Publish metrics and provenance only. Initial parameter payload remains
        # local; its digest is recorded in initial_manifest.json and results.json.
        files.update({name: pin(Path(report['run_root']) / name)
                      for name in ('initial_manifest.json', 'summary.json')})
        # The first worker version saved telemetry.csv locally but its upload
        # allowlist called it samples.csv. Preserve the completed worker
        # artifacts and publish all five raw sensor streams in this report.
        files.update({f'evidence/{name}/telemetry.csv': pin(Path(report['run_root']) / name / 'telemetry.csv')
                      for name in RUNS})
        files.update({f'evidence/{name}/notifications.json': pin(Path(report['run_root']) / name / 'notifications.json')
                      for name in RUNS})
        for name, proof in files.items():
            artifact.add_file(proof['file'], name=name)
        uploaded = run.log_artifact(artifact).wait()
        qualified_name = uploaded.qualified_name
    api = wandb.Api(timeout=45)
    remote = api.artifact(qualified_name)
    file_proofs = verify_manifest(remote, files)
    run_state = api.run(url.split('wandb.ai/', 1)[1].replace('/runs/', '/')).state
    if remote.state != 'COMMITTED' or run_state != 'finished':
        raise ValueError('Published report is not finished/committed')
    message = f'CPS/REP 검증 5/5 완료 및 W&B artifact 검증 완료. 판정: { {name: gate["status"] for name, gate in report["acceptance"].items()} }. 후속 효율성 실험은 아직 실행하지 않았습니다. 보고서 {url}'
    slack = notify_slack(message)
    receipt = {'status': 'completed' if slack == 'sent' else 'notification_failed', 'wandb_url': url,
               'artifact': qualified_name, 'artifact_state': remote.state, 'run_state': run_state,
               'verified_worker_count': 5, 'report_files': file_proofs, 'slack_status': slack,
               'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(output / 'publication.json', receipt)
    return receipt


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'results/cps_rep_gpu_validation_20261009')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/cps_rep/gpu_validation')
    parser.add_argument('--publish', action='store_true')
    options = parser.parse_args()
    receipt_path = options.output / 'publication.json'
    if options.publish and receipt_path.is_file():
        previous = read_json(receipt_path)
        if previous.get('status') == 'completed':
            print(json.dumps({'status': 'already_published', 'wandb_url': previous['wandb_url']}))
            return
    report = build_report(options.run_root)
    write_report(report, options.output)
    receipt = publish(report, options.output) if options.publish else {'status': 'local_only'}
    print(json.dumps({'acceptance': {name: gate['status'] for name, gate in report['acceptance'].items()}, 'publication': receipt}, ensure_ascii=False))


if __name__ == '__main__':
    main_cli()
