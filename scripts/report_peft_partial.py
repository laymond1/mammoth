"""Render the accepted subset of the frozen PEFT pilot without accepting SD.

The original full-report gate stays unchanged. Final publication requires the
exact original twenty-job schedule: sixteen accepted workers, the preserved SD
collapse, and three deferred SD jobs. --draft can inspect an ongoing continuation
but always blocks publication. No training, ML imports or networking are used.
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

from scripts import report_peft_efficiency as full
from scripts import run_peft_efficiency as controller
from scripts import low_rank_protocol as low_rank
from scripts.cl_efficiency import write_json
from scripts.report_hide_results import accuracy, first_task_snapshot, METRICS, RAW_KEYS
from scripts.run_low_rank_efficiency import efficiency_protocol, REQUIRED_CHECKS as LORA_CHECKS

REPORT_FILES = full.REPORT_FILES
ACCEPTED_NEW_METHODS = ('ewc-lora', 'inf-lora', 'cl-lora', 'ranpac')
HISTORICAL_METHODS = ('ease', 'l2p', 'dualprompt', 'hide-prompt')
FROZEN_COMMIT = '04559dbdc8c7e68adb96da7871d9d31c2fc5365e'
DEFAULT_DIAGNOSTIC = ROOT / 'results/peft_pilot/sd_author_common_ease_20261008/result.json'
DEFAULT_EXECUTION = ROOT / 'results/peft_pilot/base-20261008/execution.json'
csv_text = full.csv_text
plot = full.plot


def _failed_observation(saved):
    """Verify the preserved failure without changing the accuracy threshold."""
    job, proof = saved['jobs'][4], saved.get('sd_quarantine', {})
    path = Path(job['directory']).resolve() / 'result.json'
    if (job.get('status') != 'quarantined' or job.get('quarantine') != proof
            or proof.get('accepted') is not False or Path(proof.get('file', '')).resolve() != path
            or not path.is_file() or controller.file_hash(path) != proof.get('sha256')
            or path.stat().st_size != proof.get('bytes')):
        raise ValueError('Partial report requires the exact preserved SD quarantine evidence')
    record = full.read_json(path)
    observed = accuracy(record)
    if (type(observed) not in (int, float) or not math.isfinite(observed)
            or observed != 10.0 or observed != proof.get('accuracy_pct')
            or record.get('computation_status') != 'completed'
            or record.get('wandb_url') != proof.get('wandb_url')):
        raise ValueError('The excluded SD observation must remain the original 10% collapse')
    try:
        full.validate_new_record(record, 'sd-lora', 'measurement')
    except ValueError as error:
        if str(error) != 'Investigate abnormal one-epoch accuracy before accepting the worker':
            raise ValueError('SD failed a different gate; arbitrary failures cannot be excluded') from None
    else:
        raise ValueError('The original SD accuracy gate unexpectedly passed')
    # The full validator stops at accuracy. Audit the subsequent input/sensor
    # conditions independently, without manufacturing an accepted SD record.
    target = full.read_json(full.EASE_FILE)['measurements'][0]
    for split in ('train', 'test'):
        current, expected = first_task_snapshot(record, split), first_task_snapshot(target, split)
        if any(current.get(key) is None or current[key] != expected.get(key) for key in RAW_KEYS):
            raise ValueError('Quarantined SD raw data or transforms differ from the common input')
    metrics = record['metrics']
    coverage = metrics.get('gpu_energy_coverage_fraction')
    if (type(coverage) not in (int, float) or not math.isfinite(coverage)
            or coverage < .99 or record.get('sensor_errors')):
        raise ValueError('Quarantined SD sensor evidence is incomplete')
    for key in ('process_wall_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib'):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Quarantined SD resource evidence is incomplete')
    rows = record.get('task_results', [])
    if (len(rows) != 1 or type(rows[0].get('seen_correct')) is not int
            or type(rows[0].get('seen_samples')) is not int
            or rows[0]['seen_correct'] != 100 or rows[0]['seen_samples'] != 1000):
        raise ValueError('SD failure accuracy must agree with the saved exact prediction counts')
    if not any(row.get('message') == 'Investigate abnormal one-epoch accuracy before accepting the worker'
               for row in saved.get('failures', [])):
        raise ValueError('Original SD failure history must be preserved')
    return {'method': 'sd-lora', 'pass': 'measurement', 'original_job_index': 5,
            'file': str(path), 'sha256': proof['sha256'], 'bytes': proof['bytes'],
            'wandb_url': record['wandb_url'], 'accuracy_pct': observed,
            'seen_correct': 100, 'seen_samples': 1000,
            'computation_status': record['computation_status'],
            'wandb_logging_status': record['wandb_logging_status'],
            'acceptance_status': 'failed_accuracy_gate', 'aggregate_excluded': True,
            'reason': proof.get('reason'), 'cause_status': 'under_investigation',
            'observed_resources': {key: metrics.get(key) for key in METRICS}}


def _diagnostic_evidence(path, failure):
    if path is None:
        path = DEFAULT_DIAGNOSTIC if DEFAULT_DIAGNOSTIC.is_file() else None
    if path is None:
        return None
    path = Path(path).resolve()
    record = full.read_json(path)
    expected_protocol = efficiency_protocol('sd-lora')
    expected_protocol.update(mode='diagnostic', measurement_repeats=0, flops_runs=0, efficiency_comparison=False)
    if (record.get('method') != 'sd-lora' or record.get('mode') != 'diagnostic'
            or record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded'
            or not str(record.get('wandb_url', '')).startswith('https://wandb.ai/laymond1/cl-efficiency/runs/')
            or record.get('measurement_aggregate_eligible') is not False
            or record.get('efficiency_comparison') is not False
            or record.get('paper_reproduction') is not False
            or record.get('checkpoint_sha256') != full.CHECKPOINT_SHA256
            or record.get('canonical_pretrained_sha256') != low_rank.CANONICAL_PRETRAINED_SHA256
            or record.get('reference_commit') != low_rank.REFERENCE_COMMIT
            or record.get('protocol') != expected_protocol or record.get('precision') != low_rank.PRECISION):
        raise ValueError('Optional SD diagnostic must be completed, pinned and separate from efficiency observations')
    expected_counts = {'observe_calls': 79, 'optimizer_updates': 79,
                       'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}
    expected_totals = {'training_examples': 5000, 'evaluation_examples': 1000, 'steps': 79, 'tasks_completed': 1}
    for observed, expected in ((record.get('counts', {}), expected_counts),
                               (record.get('observed_totals', {}), expected_totals)):
        if set(observed) != set(expected) or any(type(observed[key]) is not int or observed[key] != value
                                               for key, value in expected.items()):
            raise ValueError('SD diagnostic must retain all actual update, sample and task counts')
    checks = record.get('structural_checks', {})
    if any(checks.get(name) is not True for name in LORA_CHECKS) or any(value is not True for value in checks.values()):
        raise ValueError('SD diagnostic structural checks must all pass')
    epochs = record.get('epoch_results', [])
    if (len(epochs) != 1 or any(type(epochs[0].get(key)) is not int or epochs[0][key] != value
                              for key, value in {'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000}.items())
            or len(record.get('first_batch_audits', [])) != 1
            or len(record.get('task_results', [])) != 1
            or type(record['task_results'][0].get('seen_samples')) is not int
            or record['task_results'][0]['seen_samples'] != 1000):
        raise ValueError('SD diagnostic first update, complete epoch or prediction evidence is incomplete')
    native = full.read_json(failure['file'])
    for split in ('train', 'test'):
        current, expected = first_task_snapshot(record, split), first_task_snapshot(native, split)
        if any(current.get(key) is None or current[key] != expected.get(key) for key in RAW_KEYS):
            raise ValueError('SD diagnostic actual raw inputs or transforms differ from the native pilot')
    row = record['task_results'][0]
    observed_accuracy = accuracy(record)
    if (type(row.get('seen_correct')) is not int or not 0 <= row['seen_correct'] <= 1000
            or type(observed_accuracy) not in (int, float) or not math.isfinite(observed_accuracy)
            or observed_accuracy != row['seen_correct'] / 10.):
        raise ValueError('SD diagnostic accuracy must agree with exact prediction counts')
    if (Path(record.get('output', '')).resolve() != path.parent
            or record.get('state_audits', {}).get('frozen_backbone_sha256') != low_rank.CANONICAL_PRETRAINED_SHA256
            or record.get('data_files') != native.get('data_files')):
        raise ValueError('SD diagnostic finalized backbone, data provenance or output directory changed')
    expected_paths = {'native_result': Path(failure['file']),
                      'native_predictions': Path(native['prediction_files'][0]['path']),
                      'native_checkpoint': Path(failure['file']).parent / 'task1_state.pt',
                      'initial_state': Path(native['initial_state']),
                      'checkpoint_file': Path(low_rank.CHECKPOINT)}
    if record.get('initial_learned_sha256') != native.get('initial_learned_sha256'):
        raise ValueError('SD diagnostic learnable initialization differs from the quarantined native pilot')
    for name, expected in expected_paths.items():
        if (Path(record.get(name, '')).resolve() != expected.resolve()
                or not expected.is_file() or low_rank.file_hash(expected) != record.get(name + '_sha256')):
            raise ValueError('SD diagnostic pinned input artifact changed or differs from the quarantined native pilot')
    if (record.get('native_result_sha256') != failure['sha256']
            or record.get('checkpoint_file_sha256') != low_rank.CHECKPOINT_SHA256
            or record.get('initial_state_sha256') != native.get('initial_state_sha256')):
        raise ValueError('SD diagnostic checkpoint or native-evidence provenance is inconsistent')
    if low_rank.verify_reference_source(record.get('reference_source', '')) != record.get('reference_source_sha256'):
        raise ValueError('SD diagnostic immutable author-source fingerprints changed')
    diagnostic_driver = ROOT / 'results/peft_pilot/sd_author_diagnostic.py'
    expected_sources = {**native['execution_source_sha256'],
                        str(diagnostic_driver.relative_to(ROOT)): low_rank.file_hash(diagnostic_driver)}
    if record.get('execution_source_sha256') != expected_sources:
        raise ValueError('SD diagnostic protected execution-source provenance differs from its native pilot')
    for name, digest in expected_sources.items():
        if low_rank.file_hash(ROOT / name) != digest:
            raise ValueError('SD diagnostic protected source fingerprint changed')
    return {'method': 'sd-lora', 'mode': 'diagnostic', 'file': str(path),
            'sha256': controller.file_hash(path), 'bytes': path.stat().st_size,
            'output': str(Path(record.get('output', path.parent)).resolve()),
            'wandb_url': record.get('wandb_url'), 'status': record.get('status'),
            'computation_status': record.get('computation_status'),
            'wandb_logging_status': record.get('wandb_logging_status'),
            'diagnosis': record.get('diagnosis'), 'comparison': record.get('comparison', {}),
            'measurement_aggregate_eligible': False, 'aggregate_excluded': True,
            'source_fingerprints': record.get('execution_source_sha256', {}),
            'interpretation': 'Diagnostic evidence is reported separately; it does not accept SD or resolve the cause automatically.'}


def build_report(execution_path, *, diagnostic_path=None, draft=False):
    """Accept precisely the unaffected subset while retaining every failed job."""
    execution_path = Path(execution_path).resolve()
    saved = full.read_json(execution_path)
    expected_statuses = ('partial_completed', 'running') if draft else ('partial_completed',)
    if saved.get('status') not in expected_statuses or saved.get('plan') != controller.load_plan():
        raise ValueError('Final partial reporting requires partial_completed; ongoing execution is draft-only')
    if Path(saved.get('output', '')).resolve() != execution_path.parent:
        raise ValueError('Execution evidence must remain in its original output directory')
    controller.validate_schedule(saved)
    if saved.get('execution_source_sha256') != controller.source_hashes():
        raise ValueError('Original measurement source fingerprints changed')
    continuation = saved.get('partial_continuation', {})
    if (continuation.get('frozen_commit') != FROZEN_COMMIT
            or continuation.get('only_original_job_indices') != list(range(9, 21))
            or continuation.get('remaining_methods') != ['inf-lora', 'cl-lora', 'ranpac']
            or continuation.get('full_twenty_workers_completed') is not False):
        raise ValueError('Partial continuation scope or original frozen source provenance changed')
    driver = Path(continuation.get('driver_file', ''))
    if not driver.is_file() or controller.file_hash(driver) != continuation.get('driver_sha256'):
        raise ValueError('Partial controller fingerprint changed')
    if saved['status'] == 'partial_completed':
        expected_counts = {'unaffected_workers_completed': 12, 'accepted_workers_including_ewc': 16,
                           'sd_quarantined_workers': 1, 'sd_deferred_workers': 3}
        if any(type(continuation.get(key)) is not int or continuation[key] != value
               for key, value in expected_counts.items()):
            raise ValueError('Partial controller completion counts are inconsistent')
    failure = _failed_observation(saved)
    deferred = []
    for index, job in enumerate(saved['jobs'][5:8], 6):
        directory = Path(job['directory'])
        if (job.get('status') != 'deferred' or job.get('result') is not None
                or not job.get('deferred_reason')
                or directory.exists() and any(directory.iterdir())):
            raise ValueError('Exactly the three unexecuted SD jobs must remain deferred')
        deferred.append({'method': job['method'], 'pass': job['pass'], 'original_job_index': index,
                         'directory': job['directory'], 'reason': job['deferred_reason']})
    records = {name: {'measurement': [], 'flops': []} for name in ACCEPTED_NEW_METHODS}
    sources, seen_paths, seen_urls = [], {failure['file']}, {failure['wandb_url']}
    for index, job in enumerate(saved['jobs'], 1):
        if job['method'] == 'sd-lora':
            continue
        if job.get('status') != 'completed':
            if draft and saved['status'] == 'running' and job.get('status') in ('pending', 'running'):
                continue
            raise ValueError('An unaffected worker failed or is incomplete; it cannot be silently excluded')
        source = job['result']
        path = Path(source['file']).resolve()
        record = full.read_json(path)
        if (path != Path(job['directory']).resolve() / 'result.json' or str(path) in seen_paths
                or source['wandb_url'] != record.get('wandb_url') or source['wandb_url'] in seen_urls
                or controller.file_hash(path) != source['sha256']):
            raise ValueError('Accepted worker files, run IDs or source fingerprints changed or repeat')
        full.validate_new_record(record, job['method'], job['pass'])
        seen_paths.add(str(path))
        seen_urls.add(source['wandb_url'])
        records[job['method']][job['pass']].append(record)
        sources.append({'method': job['method'], 'pass': job['pass'], 'original_job_index': index, **source})
    target = records['ewc-lora']['measurement'][0] if records['ewc-lora']['measurement'] else None
    if target is None or len(records['ewc-lora']['measurement']) != 3 or len(records['ewc-lora']['flops']) != 1:
        raise ValueError('The already completed EWC workers cannot be missing from a partial report')
    methods, history_proofs = {}, {}
    for name, (document, measurements, flops, path) in full.historical_inputs().items():
        proof = full.compatible_history(document, measurements, target)
        if proof['status'] != 'compatible':
            raise ValueError(f'Historical {name} cannot be reused: {proof["reasons"]}')
        methods[name] = full.method_summary(name, measurements, flops, protocol=document['protocol'], historical=True)
        history_proofs[name] = {**proof, 'file': str(path), 'sha256': controller.file_hash(path)}
    missing_groups = []
    for name, item in records.items():
        if len(item['measurement']) != 3 or len(item['flops']) != 1:
            if draft and saved['status'] == 'running':
                missing_groups.append(name)
                continue
            raise ValueError('Every accepted method requires three measurements and one separate profiler')
        methods[name] = full.method_summary(name, item['measurement'], item['flops'][0], protocol=saved['plan'])
        methods[name]['label'] = full.LABELS.get(name, methods[name]['label'])
        methods[name]['method_diagnostics'] = [{key: record.get(key) for key in
            ('counts', 'extra_samples', 'method_state_storage', 'unregistered_statistics_storage',
             'task_end_training_image_presentations', 'ridge_candidate_count', 'ridge_solve_count',
             'ridge_solver_audit', 'rp_size', 'auxiliary_storage', 'state_audits')}
            for record in item['measurement']]
    measurements = sum(source['pass'] == 'measurement' for source in sources)
    profilers = sum(source['pass'] == 'flops' for source in sources)
    is_final = saved['status'] == 'partial_completed' and not draft
    if is_final and (len(methods) != 8 or len(sources) != 16 or measurements != 12 or profilers != 4):
        raise ValueError('Final partial report must contain eight accepted methods and exactly sixteen fresh workers')
    diagnostic = _diagnostic_evidence(diagnostic_path, failure)
    generated = Path(__file__).resolve()
    return {'schema_version': 1, 'status': 'partial_completed' if is_final else 'draft',
            'publication_gate': 'pass_for_accepted_subset' if is_final else 'blocked',
            'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'protocol': saved['plan'], 'method_count': len(methods), 'expected_accepted_method_count': 8,
            'requested_method_count': 9, 'all_requested_methods_passed': False,
            'new_measurement_runs': measurements, 'new_flops_runs': profilers,
            'new_accepted_workers': len(sources), 'new_quarantined_workers': 1, 'new_deferred_workers': 3,
            'planned_new_workers': 20, 'historical_reused_methods': list(HISTORICAL_METHODS),
            'methods': methods, 'historical_compatibility': history_proofs,
            'sources': sources, 'failed_observations': [failure], 'deferred_jobs': deferred,
            'diagnostic_evidence': diagnostic, 'diagnosis_status': 'cause_not_confirmed',
            'draft_incomplete_method_groups': missing_groups,
            'execution': {'file': str(execution_path), 'sha256': controller.file_hash(execution_path),
                          'execution_source_sha256': saved['execution_source_sha256'],
                          'frozen_commit': FROZEN_COMMIT,
                          'original_execution_sha256': continuation.get('original_execution_sha256'),
                          'partial_continuation': continuation, 'failures': saved['failures']},
            'report_generator': {'file': str(generated), 'sha256': controller.file_hash(generated)},
            'scope': {'first_task_only': True, 'paper_reproduction': False, 'equal_compute_budget': False,
                      'same_seed_repeats': True, 'seed_uncertainty_estimated': False,
                      'full_machine_energy_measured': False, 'accepted_subset_only': True,
                      'sd_accepted': False, 'sd_and_diagnostic_costs_in_aggregates': False,
                      'historical_measurement_dates_and_source_snapshots_differ': True,
                      'same_augmentation_recipe_not_identical_random_crop_realizations': True,
                      'ease_cpu_threads': 'not_recorded', 'flops': 'partial supported-operator estimates'}}


def markdown(report):
    rows = ['# Base 첫 태스크: 승인된 방법의 부분 효율성 보고서', '',
            f'보고서 상태: **{report["status"]}**. 현재 승인된 비교는 {report["method_count"]}방법입니다. '
            '요청한 전체9방법·신규20실행이 모두 통과한 결과가 아닙니다.', '',
            f'신규 승인 기록은 측정{report["new_measurement_runs"]}회와 별도 부분 FLOPs'
            f'{report["new_flops_runs"]}회입니다. EASE·L2P·DualPrompt·HiDe-Prompt는 '
            '원본 데이터·전처리·가중치·precision을 대조한 기존 기록을 재사용했습니다.', '',
            'SD-LoRA의 첫 측정은100/1000(10.0%)로 비정상 정확도 검사를 통과하지 못했습니다. '
            '해당 관측은 격리하고 나머지 SD측정2회·FLOPs1회를 보류했습니다. 원인은 아직 확정되지 않았으며 '
            '별도 진단 증거를 검토합니다. SD 관측과 진단 실행은 평균·표준편차·비교 그림에 포함하지 않습니다.', '',
            'CIFAR100 첫10클래스(train5000/test1000), ViT-B/16 AugReg IN21K→IN1K, batch64, '
            'seed1993, GPU1, FP32입니다. 신규 방법은1epoch, HiDe는TII1+prompt1입니다. '
            '방법별 task-begin/end 작업을 포함하므로 동일 계산 예산 비교가 아닙니다.', '',
            '| 승인된 방법 | 정확도 % | 학습 시간 s | 벽시간 s, 평균±SD | CUDA peak GiB | RSS peak GiB | CPU s | GPU 보드 Wh, 평균±SD | 부분 TFLOPs |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in report['methods'].values():
        r = item['resources']
        training_time = item['phases']['train_epoch']['wall_s']['mean']
        rows.append(f'| {item["label"]} | {item["accuracy_pct"]["mean"]:.2f} | {training_time:.2f} | '
                    f'{r["process_wall_s"]["mean"]:.2f} ± {r["process_wall_s"]["std_sample"]:.2f} | '
                    f'{r["cuda_peak_allocated_mib"]["mean"] / 1024:.3f} | '
                    f'{r["peak_process_tree_rss_mib"]["mean"] / 1024:.3f} | '
                    f'{r["process_tree_cpu_s"]["mean"]:.2f} | '
                    f'{r["gpu_board_energy_wh"]["mean"]:.3f} ± {r["gpu_board_energy_wh"]["std_sample"]:.3f} | '
                    f'{item["flops"]["supported_ops_total"] / 1e12:.3f} |')
    rows += ['', '학습 시간은 `train_epoch` 관측 구간의 평균입니다. HiDe는 TII와 prompt 학습 구간의 합계입니다. '
             '아래 태스크 경계·평가 구간은 별도로 기록했으며 중첩된 statistics 구간을 다시 더하지 않습니다.', '',
             '| 승인된 방법 | begin_task s | end_epoch s | end_task s | evaluation s |',
             '|---|---:|---:|---:|---:|']
    for item in report['methods'].values():
        values = [item['phases'].get(phase, {}).get('wall_s', {}).get('mean')
                  for phase in ('begin_task', 'end_epoch', 'end_task', 'evaluation')]
        rows.append('| ' + item['label'] + ' | ' + ' | '.join('미관측' if value is None else f'{value:.2f}' for value in values) + ' |')
    rows += ['', '![승인된 방법의 효율성 비교](efficiency_comparison.png)', '', '[PDF 그림](efficiency_comparison.pdf)', '',
             '평균±표본 SD는 동일 seed의 실행 비용 변동입니다. 별도 profiler 실행의 시간·메모리·에너지는 '
             '평균에서 제외했습니다. GPU 보드 에너지는 CPU/RAM/전체 머신 에너지를 포함하지 않습니다. '
             '부분 FLOPs는 CPU 선형대수·미지원 연산 등 전체 연산을 보장하지 않습니다.', '',
             '기존 기록은 측정 날짜·소스가 다르며 EASE의 Torch CPU thread 수는 미기록입니다. '
             '같은 augmentation recipe라도 방법별 RNG 경로 때문에 실제 crop은 달라질 수 있습니다. '
             '전체10태스크 정확도·망각·동일 예산 성능을 추정하지 않습니다.', '',
             '첫 태스크에서는 EWC의 과거 지식 penalty, CL의 KD/과거 branch 학습, Inf의 과거 subspace '
             '상호작용 비용이 나타나지 않습니다. Fisher·feature/SVD/GPM·프로토타입·RanPAC RP/ridge는 '
             '실제 수행한 범위로 포함합니다. RanPAC은 실제 학습 전처리와 task-end 특징 추출 전처리를 별도로 기록합니다.', '',
             '| 승인된 방법 | 측정 실행 | 별도 FLOPs |', '|---|---|---|']
    for item in report['methods'].values():
        links = ' · '.join(f'[측정{i}]({url})' for i, url in enumerate(item['wandb_urls'], 1))
        rows.append(f'| {item["label"]} | {links} | [진단]({item["flops_wandb_url"]}) |')
    rows += ['', '| 격리 관측 | 실제 정확도 | 판정 | 실행 기록 |', '|---|---:|---|---|']
    for observation in report['failed_observations']:
        rows.append(f'| SD-LoRA 측정1 | {observation["accuracy_pct"]:.1f}% '
                    f'({observation["seen_correct"]}/{observation["seen_samples"]}) | 정확도 검사 실패·집계 제외 | '
                    f'[W&B]({observation["wandb_url"]}) |')
    diagnostic = report.get('diagnostic_evidence')
    if diagnostic is None:
        rows += ['', '별도 SD 진단 결과는 아직 이 보고서에 포함되지 않았습니다. 원인은 미확정입니다.']
    else:
        rows += ['', f'별도 SD 진단 상태: `{diagnostic.get("status")}`. '
                 f'관측된 진단 분류: `{diagnostic.get("diagnosis")}`. '
                 '이는 효율성 집계와 SD 승인 판정에 영향을 주지 않으며 원인 검토를 위한 별도 증거입니다.']
        comparison = diagnostic.get('comparison', {})
        if (diagnostic.get('diagnosis') == 'author_reproduces_native_feature_collapse'
                and all(comparison.get(key) is True for key in
                        ('input_sha256_matches', 'labels_sha256_matches', 'gradient_sha256_matches',
                         'all_network_tensors_bitexact', 'prediction_arrays_match'))):
            rows += ['', '동일 초기값·첫 배치·79회 업데이트에서 저자 코드도10.0%를 기록했습니다. '
                     '첫 그래디언트·최종 모델의 모든 텐서·1000개 예측이 현재 구현과 완전히 일치했습니다. '
                     '이 조건의 원형 일치는 확인됐으며, 공통 전처리와 기존 LR0.008 조합의 학습 안정성은 '
                     '추가 검토가 필요합니다. 전처리나 학습률 중 어느 요인이 붕괴를 일으켰는지는 이 대조만으로 확정하지 않습니다.']
        if diagnostic.get('wandb_url'):
            rows += ['', f'[별도 진단 W&B]({diagnostic["wandb_url"]})']
    if report['publication_gate'] == 'blocked':
        rows += ['', '**초안: 진행 중인 실행을 포함하므로 최종 게시할 수 없습니다.**']
    return '\n'.join(rows) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial')
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args(argv)
    report = build_report(options.execution, diagnostic_path=options.diagnostic, draft=options.draft)
    if options.write:
        if options.draft and options.output.resolve().is_relative_to(ROOT / 'benchmarks'):
            raise ValueError('Draft reports may be written only outside the public benchmarks directory')
        options.output.mkdir(parents=True, exist_ok=True)
        write_json(options.output / 'results.json', report)
        write_json(options.output / 'plan.json', report['protocol'])
        (options.output / 'results.csv').write_text(csv_text(report))
        (options.output / 'RESULTS.md').write_text(markdown(report))
        plot(report, options.output)
    print(json.dumps({'status': report['status'], 'publication_gate': report['publication_gate'],
                      'method_count': report['method_count'], 'accepted_workers': report['new_accepted_workers'],
                      'written': options.write}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
