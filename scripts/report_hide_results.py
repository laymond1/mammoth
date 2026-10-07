"""Build local HiDe-Prompt reports from completed, saved observations only.

No model, data loader, GPU, credentials or network service is accessed. The
optional Matplotlib import only renders scientific PNG/PDF artifacts. Historical
baselines are reused when their data, preprocessing, weights and protocol match;
their different phase budgets and measurement times remain explicit.
"""
from __future__ import annotations

import argparse
import csv
import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.hide_acceptance import evaluate_acceptance, reference_policy

METRICS = (
    'process_wall_s', 'process_tree_cpu_s', 'gpu_board_energy_wh',
    'gpu_board_energy_j', 'gpu_energy_coverage_fraction',
    'cuda_peak_allocated_mib', 'cuda_peak_reserved_mib',
    'peak_process_tree_rss_mib', 'peak_gpu_device_memory_mib',
    'outer_training_examples_per_s', 'evaluation_examples_per_s',
)
PHASE_METRICS = ('wall_s', 'gpu_board_energy_j', 'gpu_energy_coverage_s')
RAW_KEYS = ('n_samples', 'class_counts', 'samples_ordered_sha256',
            'samples_unordered_sha256', 'global_ids_ordered_sha256',
            'transform_repr_sha256')
PRECISION_FIELDS = ('parameter_dtype', 'autocast', 'matmul_precision',
                    'tf32_allowed', 'cudnn_tf32_allowed', 'cudnn_deterministic',
                    'cudnn_benchmark')
LABELS = {'ease': 'EASE', 'ranpac': 'RanPAC', 'l2p': 'L2P',
          'dualprompt': 'DualPrompt', 'hide-prompt': 'HiDe-Prompt'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Saved result must be a JSON object')
    return value


def descriptive(values):
    """Sample SD describes repeated execution costs, never seed uncertainty."""
    values = list(values)
    observed = [value for value in values if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in observed):
        raise ValueError('Report observations must be finite numbers or explicitly missing')
    return {'values': values, 'available_count': len(observed),
            'mean': statistics.mean(observed) if observed else None,
            'std_sample': statistics.stdev(observed) if len(observed) > 1 else None,
            'median': statistics.median(observed) if observed else None,
            'min': min(observed) if observed else None,
            'max': max(observed) if observed else None}


def accuracy(record):
    value = record.get('accuracy')
    if isinstance(value, dict):
        for key in ('mean_accuracy_pct', 'class_il_pct', 'seen_accuracy_pct'):
            if key in value:
                return value[key]
    rows = record.get('task_results', record.get('tasks_results', []))
    if len(rows) == 1 and rows[0].get('seen_samples'):
        return 100 * rows[0]['seen_correct'] / rows[0]['seen_samples']
    return None


def first_batches(reference, native):
    expected = {(task, phase) for task in (1, 2) for phase in ('tii_train', 'prompt_train')}
    by_side = []
    for record in (reference, native):
        rows = record.get('first_batch_audits', [])
        indexed = {(row.get('task'), row.get('phase')): row for row in rows}
        by_side.append(indexed if len(indexed) == len(rows) else {})
    rows = []
    for coordinate in sorted(expected):
        left, right = (side.get(coordinate) for side in by_side)
        present = left is not None and right is not None
        keys = ('samples', 'seed', 'input_sha256', 'labels_sha256')
        match = present and all(left.get(key) == right.get(key) and left.get(key) is not None for key in keys)
        row = {'task': coordinate[0], 'phase': coordinate[1],
               'status': 'pass' if match else ('missing' if not present else 'fail'),
               'input_labels_seed_and_samples_match': match}
        if present:
            row.update(reference_loss=left.get('loss'), native_loss=right.get('loss'),
                       clipped_gradient_sha256_matches=left.get('clipped_gradient_sha256') == right.get('clipped_gradient_sha256'))
            if isinstance(left.get('loss'), (int, float)) and isinstance(right.get('loss'), (int, float)):
                row['native_minus_reference_loss'] = right['loss'] - left['loss']
        rows.append(row)
    return {'status': 'pass' if all(row['status'] == 'pass' for row in rows) else 'fail',
            'rows': rows,
            'scope': 'Exact actual input/label hashes, phase seed and sample count. Gradient SHA and loss gaps are diagnostics; masked-CE floating reductions need not be bitwise equal.'}


def exact_validation(reference, native):
    policy = reference_policy(reference.get('task_results', reference.get('tasks_results', [])))
    evaluated = evaluate_acceptance(native.get('task_results', native.get('tasks_results', [])), policy,
                                   structural_checks=native.get('structural_checks', {}))
    expected_status = native.get('acceptance_status')
    if expected_status in ('pass', 'review', 'fail') and expected_status != evaluated['status']:
        raise ValueError('Saved native acceptance disagrees with the exact saved counts and structural gates')
    reference_rows = {row['task']: row for row in reference.get('task_results', reference.get('tasks_results', []))}
    native_rows = {row['task']: row for row in native.get('task_results', native.get('tasks_results', []))}
    metrics = []
    for task, partition in ((1, 'seen'), (2, 'seen'), (2, 'old'), (2, 'new')):
        name = f'task{task}_{partition}'
        ref = reference_rows[task]
        observed = native_rows.get(task, {})
        correct_key, samples_key = partition + '_correct', partition + '_samples'
        correct, samples = ref[correct_key], ref[samples_key]
        native_correct, native_samples = observed.get(correct_key), observed.get(samples_key)
        reference_accuracy = Fraction(correct * 100, samples)
        valid = (type(native_correct) is int and type(native_samples) is int
                 and native_samples == samples and 0 <= native_correct <= native_samples)
        native_accuracy = Fraction(native_correct * 100, native_samples) if valid else None
        gap = reference_accuracy - native_accuracy if valid else None
        status = ('fail' if not valid or gap > Fraction(3) else
                  'review' if gap > Fraction(3, 2) else 'pass')
        metrics.append({'metric': name, 'reference_correct': correct, 'reference_samples': samples,
                        'native_correct': native_correct, 'native_samples': native_samples,
                        'reference_accuracy_pct': float(reference_accuracy),
                        'native_accuracy_pct': float(native_accuracy) if valid else None,
                        'reference_minus_native_pp': float(gap) if valid else None,
                        'gap_exact': {'numerator': gap.numerator, 'denominator': gap.denominator} if valid else None,
                        'status': status})
    batches = first_batches(reference, native)
    return {'acceptance': evaluated, 'accuracy_metrics': metrics,
            'first_batch_comparison': batches,
            'report_gate': 'pass' if evaluated['status'] == 'pass' and batches['status'] == 'pass'
                          and native.get('status') == 'completed' else 'blocked',
            'reference': {key: reference.get(key) for key in ('status', 'wandb_url', 'mode', 'protocol', 'metrics', 'phases', 'alignment_results', 'initial_state_sha256', 'initial_learned_sha256')},
            'native': {key: native.get(key) for key in ('status', 'wandb_url', 'mode', 'protocol', 'metrics', 'phases', 'alignment_history', 'counts', 'initial_state_sha256', 'initial_learned_sha256')}}


def first_task_snapshot(record, split):
    snapshots = record.get('data_snapshots', {})
    return snapshots.get(f'task1_{split}', snapshots.get('first_task', {}).get(split, {}))


def cpu_thread_metadata(record):
    """Preserve missing observations; accept either serialized field location."""
    precision = record.get('precision') or {}
    values = {name: value for name, value in (
        ('record', record.get('torch_cpu_threads')),
        ('precision', precision.get('torch_cpu_threads')),
    ) if value is not None}
    invalid = any(type(value) is not int or value < 1 for value in values.values())
    conflict = bool(values) and any(value != next(iter(values.values())) for value in values.values())
    return {'value': next(iter(values.values())) if values and not invalid and not conflict else None,
            'sources': values, 'status': 'invalid' if invalid or conflict else
            'recorded' if values else 'not_recorded'}


def compatible_history(document, records, target):
    reasons = []
    if document.get('status') != 'completed' or document.get('pilot_status') != 'pass':
        reasons.append('historical pilot is not completed/pass')
    if document.get('checkpoint_sha256') != target.get('checkpoint_sha256'):
        reasons.append('checkpoint fingerprint differs or is unavailable')
    protocol = document.get('protocol', {})
    efficiency = protocol.get('efficiency', protocol)
    historical_gpu, target_gpu = protocol.get('gpu'), target.get('gpu')
    if historical_gpu is not None and target_gpu is not None and str(historical_gpu) != str(target_gpu):
        reasons.append('physical GPU selection differs')
    for key, expected in (('dataset', 'seq-cifar100-224'), ('dataset_config', 'ease')):
        if protocol.get(key) != expected:
            reasons.append(f'historical {key} differs')
    for key, expected in (('tasks', 1), ('batch_size', 64), ('seed', 1993), ('num_workers', 4)):
        if efficiency.get(key) != expected:
            reasons.append(f'historical {key} differs')
    if len(records) != 3 or any(record.get('pass') != 'measurement' or record.get('status') != 'completed' for record in records):
        reasons.append('three completed measurement observations are required')
    target_precision = target.get('precision') or {}
    target_threads = cpu_thread_metadata(target)
    historical_threads = []
    for record in records:
        if record.get('metrics', {}).get('evaluation_examples') != 1000:
            reasons.append('historical evaluation sample count differs')
        for split in ('train', 'test'):
            left, right = first_task_snapshot(record, split), first_task_snapshot(target, split)
            for key in RAW_KEYS:
                if left.get(key) is None or right.get(key) is None or left.get(key) != right.get(key):
                    reasons.append(f'historical {split} {key} differs or lacks evidence')
        historical_precision = record.get('precision') or {}
        for key in PRECISION_FIELDS:
            left, right = historical_precision.get(key), target_precision.get(key)
            if left is None or right is None or type(left) is not type(right) or left != right:
                reasons.append(f'precision policy {key} differs or lacks required evidence')
        threads = cpu_thread_metadata(record)
        historical_threads.append(threads)
        if threads['status'] == 'invalid' or target_threads['status'] == 'invalid':
            reasons.append('CPU thread metadata is invalid or conflicting within a record')
        elif threads['value'] is not None and target_threads['value'] is not None and threads['value'] != target_threads['value']:
            reasons.append('recorded CPU thread count differs')
    threads_available = (target_threads['status'] == 'recorded'
                         and all(entry['status'] == 'recorded' for entry in historical_threads))
    threads_match = threads_available and all(entry['value'] == target_threads['value'] for entry in historical_threads)
    return {'status': 'compatible' if not reasons else 'excluded', 'reasons': sorted(set(reasons)),
            'required_precision_fields': list(PRECISION_FIELDS),
            'cpu_thread_evidence': {'status': 'matched' if threads_match else
                                    'metadata_missing' if target_threads['status'] == 'not_recorded' or
                                    any(entry['status'] == 'not_recorded' for entry in historical_threads) else 'mismatch_or_invalid',
                                    'historical': historical_threads, 'target': target_threads,
                                    'scope': 'Only serialized CPU thread metadata is used. Missing values remain unknown; source defaults are not substituted.'},
            'physical_gpu_evidence': 'matched' if historical_gpu is not None and target_gpu is not None and str(historical_gpu) == str(target_gpu) else 'not_available_in_both_serialized_records',
            'scope': 'Requires matched raw first-task data, transforms, Base checkpoint, batch64, seed1993, workers and seven explicit precision policies. Known CPU thread counts must match; missing metadata is disclosed. Different phase budgets, source snapshots and execution dates remain.'}


def method_summary(name, measurements, flops, *, protocol=None, historical=False):
    if len(measurements) != 3 or any(record.get('status') != 'completed' or record.get('pass') != 'measurement' for record in measurements):
        raise ValueError('Efficiency reports require three complete measurement records')
    resources = {key: descriptive(record.get('metrics', {}).get(key) for record in measurements) for key in METRICS}
    phases = {}
    for phase in sorted(set().union(*(record.get('phases', {}) for record in measurements))):
        phases[phase] = {key: descriptive(record.get('phases', {}).get(phase, {}).get(key) for record in measurements)
                         for key in PHASE_METRICS}
    snapshots = {}
    for phase in sorted(set().union(*(record.get('model_snapshots', {}) for record in measurements))):
        snapshots[phase] = {key: descriptive(record.get('model_snapshots', {}).get(phase, {}).get(key) for record in measurements)
                            for key in ('parameters_total', 'parameters_trainable', 'parameters_in_optimizer',
                                        'parameter_storage_bytes', 'registered_buffer_storage_bytes', 'optimizer_state_storage_bytes')}
    diagnostic = flops.get('flops', flops) if isinstance(flops, dict) else {}
    alignment_skipped = (name == 'hide-prompt' and all(
        len(record.get('alignment_history', [])) == 2
        and all(row.get('status') == 'not_applicable' for row in record['alignment_history'])
        for record in measurements))
    return {'model': name, 'label': LABELS.get(name, name), 'historical_reuse': historical,
            'protocol': protocol, 'measurement_count': 3, 'resources': resources,
            'phases': phases, 'parameters': snapshots,
            'not_applicable_phases': ['tii_alignment', 'prompt_alignment'] if alignment_skipped else [],
            'accuracy_pct': descriptive(accuracy(record) for record in measurements),
            'counts': [record.get('counts', {}) for record in measurements],
            'flops': diagnostic, 'flops_wandb_url': flops.get('wandb_url') if isinstance(flops, dict) else None,
            'wandb_urls': [record.get('wandb_url') for record in measurements]}


def summary_only(document):
    """A saved summary can be rendered, without manufacturing raw-data evidence."""
    if document.get('status') != 'pass' or document.get('measurement_count') != 3 or document.get('flops_count') != 1:
        raise ValueError('Summary-only reporting requires the complete passing pilot summary')
    records = []
    for index in range(3):
        records.append({'status': 'completed', 'pass': 'measurement',
            'metrics': {key: document.get('resources', {}).get(key, {}).get('values', [None] * 3)[index] for key in METRICS},
            'phases': {phase: {key: entry.get(key, {}).get('values', [None] * 3)[index] for key in PHASE_METRICS}
                       for phase, entry in document.get('phases', {}).items()},
            'model_snapshots': document.get('parameter_snapshots', [{}, {}, {}])[index],
            'task_results': document.get('task_diagnostics', [[], [], []])[index] or [],
            'wandb_url': document.get('wandb_urls', [None] * 3)[index]})
    return method_summary('hide-prompt', records, document.get('flops', {})), records


def load_efficiency(path):
    value = read_json(path)
    companion = Path(path).with_name('execution.json')
    if 'measurements' not in value and companion.is_file() and companion.resolve() != Path(path).resolve():
        value = read_json(companion)
    if 'measurements' not in value:
        summary, records = summary_only(value)
        return summary, records, value
    if value.get('status') != 'completed' or value.get('summary', {}).get('status') != 'pass':
        raise ValueError('Efficiency execution is incomplete or did not pass measurement checks')
    from scripts.run_hide_efficiency import summarize
    summarize(value['measurements'], value['flops'])
    return method_summary('hide-prompt', value['measurements'], value['flops'], protocol=value.get('protocol')), value['measurements'], value


def build_report(reference, native, efficiency_path=None, historical_path=None):
    if reference.get('status') != 'completed' or reference.get('mode') != 'validation':
        raise ValueError('Report requires the completed full author-code validation reference')
    if native.get('mode') != 'validation':
        raise ValueError('Smoke or efficiency accuracy cannot replace full native validation')
    from scripts.hide_protocol import protocol
    if reference.get('protocol') != protocol('validation') or native.get('protocol') != reference.get('protocol'):
        raise ValueError('Reference/native report inputs must use the same approved full validation protocol')
    for key in ('checkpoint_sha256', 'initial_state_sha256', 'initial_learned_sha256',
                'initial_canonical_pretrained_sha256'):
        if not reference.get(key) or reference[key] != native.get(key):
            raise ValueError(f'Reference/native {key} is missing or differs')
    validation = exact_validation(reference, native)
    methods, exclusions = [], []
    efficiency = {'status': 'not_executed', 'methods': methods, 'historical_exclusions': exclusions}
    if efficiency_path is not None:
        if validation['report_gate'] != 'pass':
            raise ValueError('Efficiency observations cannot be reported as accepted after failed validation/first-batch gates')
        current, records, raw = load_efficiency(efficiency_path)
        if historical_path is not None:
            historical = read_json(historical_path)
            entries = historical.get('models', {})
            if not entries and 'measurements' in historical:
                entries = {'ease': {'measurements': historical['measurements'], 'flops': historical.get('flops_diagnostic', {})}}
            for name in ('ease', 'ranpac', 'l2p', 'dualprompt'):
                if name not in entries:
                    continue
                entry = entries[name]
                matching = compatible_history(historical, entry.get('measurements', []), records[0])
                if matching['status'] != 'compatible':
                    exclusions.append({'model': name, **matching})
                    continue
                saved = method_summary(name, entry['measurements'], entry.get('flops', {}),
                                       protocol=historical.get('protocol'), historical=True)
                saved['compatibility'] = matching
                methods.append(saved)
        methods.append(current)
        efficiency.update(status='completed', execution_status=raw.get('status'),
                          phase_budget='HiDe: TII1+prompt1, two full statistics passes, inferred routing, task1 alignment N/A. Historical methods: one outer epoch and their native task hooks.',
                          comparison_is_equal_compute_budget=False,
                          seed_variation_estimated=False, std_definition='Sample standard deviation (ddof=1) across three executions with the same seed; not uncertainty across seeds.')
    return {'schema_version': 1, 'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': 'completed', 'model': 'hide-prompt', 'validation': validation, 'efficiency': efficiency,
            'measurement_scope': {
                'energy': 'Trapezoidal integration of sampled selected physical GPU-board power; excludes CPU/RAM/whole-machine energy and includes idle/other-user board activity.',
                'time': 'Recorder process and inclusive synchronized phase times. Final W&B artifact upload/finish is outside Recorder sampling.',
                'memory': 'CUDA allocator peak and sampled sum of process-tree RSS; shared RSS pages can be counted multiple times.',
                'flops': 'One separate partial supported-operator diagnostic per method; unsupported/fused operators and CPU clustering may be omitted. Diagnostic resource costs are excluded.',
                'comparison': 'Historical observations were measured at different times and source snapshots. Matching inputs and settings does not establish equal phase budgets or eliminate measurement drift.',
                'accuracy': 'Two-task matched-author correctness and first-task short-pilot accuracy are separate; a first-task pilot cannot establish forgetting or full-sequence CL performance.'}}


def format_number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def mean_sd(value, factor=1.):
    if value['mean'] is None:
        return '—'
    mean = format_number(value['mean'] * factor)
    return mean if value['std_sample'] is None else f'{mean} ± {format_number(value["std_sample"] * factor)}'


def markdown(report):
    validation = report['validation']
    lines = ['# HiDe-Prompt 실행 결과', '',
             f'정확도 판정: **{validation["acceptance"]["status"]}**. 실제 첫 배치 입력·라벨·seed 대조: **{validation["first_batch_comparison"]["status"]}**.', '',
             '검증 조건은 Base ViT-B/16, 동일 IN21K→IN1K 가중치, CIFAR100 첫 2태스크, seed1993, batch64입니다. 태스크마다 TII20 + prompt50, 태스크2에서 두 정렬 각각 30epoch을 계획한 독립 저자 코드 비교이며, 실제 완료 상태와 비용은 아래에 표시합니다. 논문 수치 재현과 구분합니다.', '',
             '| 정확도 지표 | 저자 코드 (%) | 현재 구현 (%) | 저자−현재 (%p) | 판정 |',
             '|---|---:|---:|---:|---|']
    for row in validation['accuracy_metrics']:
        lines.append(f'| {row["metric"]} | {format_number(row["reference_accuracy_pct"])} | {format_number(row["native_accuracy_pct"])} | {format_number(row["reference_minus_native_pp"])} | {row["status"]} |')
    lines.extend(['', '판정은 저장된 정답 수/샘플 수의 정확한 분수로 계산했습니다. 1.5%p 이내 통과, 1.5~3.0%p 추가 점검, 3.0%p 초과 실패이며 구조 검증 실패도 통과를 차단합니다.', '',
                  '| 검증 실행 | 상태 | W&B | 벽시간 (s) | GPU 보드 에너지 (Wh) | CUDA peak (MiB) |',
                  '|---|---|---|---:|---:|---:|'])
    for name in ('reference', 'native'):
        side = validation[name]
        metric = side.get('metrics') or {}
        url = side.get('wandb_url')
        lines.append(f'| {name} | {side.get("status", "unknown")} | {"[run](" + url + ")" if url else "—"} | {format_number(metric.get("process_wall_s"))} | {format_number(metric.get("gpu_board_energy_wh"))} | {format_number(metric.get("cuda_peak_allocated_mib"))} |')
    lines.extend(['', '### 검증 단계 비용', '',
                  '두 태스크의 누적 단계 비용입니다. 정렬은 첫 태스크에서 해당 없음이며 실제 정렬 작업은 태스크2에만 발생합니다. 타이머가 중첩되므로 단계 합계는 총 비용과 다릅니다.', '',
                  '| 단계 | 저자 코드 시간 (s) | 현재 구현 시간 (s) | 현재 구현 GPU 보드 (J) |',
                  '|---|---:|---:|---:|'])
    reference_phases = validation['reference'].get('phases') or {}
    native_phases = validation['native'].get('phases') or {}
    for phase in ('tii_train', 'tii_statistics', 'tii_alignment', 'prompt_train', 'prompt_finalize',
                  'prompt_statistics', 'prompt_alignment', 'evaluation', 'checkpoint'):
        left, right = reference_phases.get(phase, {}), native_phases.get(phase, {})
        lines.append(f'| {phase} | {format_number(left.get("wall_s"))} | {format_number(right.get("wall_s"))} | {format_number(right.get("gpu_board_energy_j"))} |')
    lines.extend(['', '## 첫 태스크 효율성', ''])
    methods = report['efficiency']['methods']
    if not methods:
        lines.append('효율성 실험은 실행하지 않았거나 검증 통과를 기다리고 있습니다. 측정값을 추정하지 않습니다.')
    else:
        lines.extend(['HiDe는 TII1+prompt1, batch64, 공통 `ease` 전처리로 측정했습니다. 두 통계 추출과 실제 TII routing 비용을 포함하며 첫 태스크 정렬은 해당 없음입니다. 기존 방법은 1 outer epoch이므로 동일 계산 예산 비교가 아닙니다.', '',
                      '아래 수치는 같은 seed의 측정 3회 **평균 ± 표본 표준편차**입니다. 별도 FLOPs 실행의 시간·메모리·에너지는 집계에서 제외했습니다.', '',
                      '| 방법 | 정확도 (%) | 벽시간 (s) | GPU 보드 (Wh) | CUDA peak (GiB) | RSS peak (GiB) | 부분 FLOPs (TF) |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for method in methods:
            resource = method['resources']
            estimate = method['flops'].get('supported_ops_total')
            lines.append(f'| {method["label"]}{" (기존 측정 재사용)" if method["historical_reuse"] else " (TII1+P1)"} | {mean_sd(method["accuracy_pct"])} | {mean_sd(resource["process_wall_s"])} | {mean_sd(resource["gpu_board_energy_wh"])} | {mean_sd(resource["cuda_peak_allocated_mib"], 1/1024)} | {mean_sd(resource["peak_process_tree_rss_mib"], 1/1024)} | {format_number(estimate / 1e12 if estimate is not None else None)} |')
        current = next(method for method in methods if method['model'] == 'hide-prompt')
        lines.extend(['', '### HiDe 단계 비용', '',
                      '단계 타이머는 포함 관계가 있으므로 합산하여 총 비용으로 해석하지 않습니다. 짧은 단계에서 power sample이 겹치지 않으면 에너지는 미측정으로 남깁니다.', '',
                      '| 단계 | 시간 (s) | GPU 보드 에너지 (J) | 에너지 관측 시간 (s) |',
                      '|---|---:|---:|---:|'])
        for phase, values in current['phases'].items():
            if phase in current['not_applicable_phases']:
                lines.append(f'| {phase} (해당 없음; hook 오버헤드는 총 비용에 포함) | N/A | N/A | N/A |')
            else:
                lines.append(f'| {phase} | {mean_sd(values["wall_s"])} | {mean_sd(values["gpu_board_energy_j"])} | {mean_sd(values["gpu_energy_coverage_s"])} |')
        lines.extend(['', '### 파라미터 수', '', '| 관측 시점 | 전체 | 학습 가능 | optimizer 포함 |', '|---|---:|---:|---:|'])
        for phase, values in current['parameters'].items():
            cells = ['—' if values[key]['mean'] is None else f'{values[key]["mean"]:,.0f}'
                     for key in ('parameters_total', 'parameters_trainable', 'parameters_in_optimizer')]
            lines.append('| ' + phase + ' | ' + ' | '.join(cells) + ' |')
        historical_methods = [method for method in methods if method['historical_reuse']]
        reused = ', '.join(method['label'] for method in historical_methods) or '없음'
        lines.extend(['', '![효율성 비교](efficiency_comparison.png)', '', '[PDF 그림](efficiency_comparison.pdf)', '',
                      f'기존 측정 재사용: **{reused}**. 데이터·전처리·가중치·batch·seed·precision 7개 필드가 일치할 때만 비교에 포함하며, 실행 시각·소스 차이와 방법별 학습/태스크 종료 작업 차이는 유지됩니다. CPU thread 수는 top-level 또는 precision 필드에 저장된 실제 메타데이터를 읽고, 양쪽에 있으면 일치를 요구합니다.', '',
                      '| 기존 방법 | CPU thread 기록 |', '|---|---|'])
        for method in historical_methods:
            evidence = method['compatibility']['cpu_thread_evidence']
            if evidence['status'] == 'matched':
                note = f'양쪽 {evidence["target"]["value"]}개 일치'
            else:
                unknown = []
                if any(entry['status'] == 'not_recorded' for entry in evidence['historical']):
                    unknown.append('기존 측정 미기록')
                if evidence['target']['status'] == 'not_recorded':
                    unknown.append('HiDe 측정 미기록')
                note = ' · '.join(unknown) + '; 일치 여부 미확인, 값 추정 없음'
            lines.append(f'| {method["label"]} | {note} |')
        exclusions = report['efficiency'].get('historical_exclusions', [])
        if exclusions:
            lines.extend(['', '다음 기존 측정은 조건 차이 또는 필수 증거 부족으로 표·그림의 동일 입력 비교에서 제외했습니다.', '',
                          '| 제외한 방법 | 제외 이유 |', '|---|---|'])
            for entry in exclusions:
                lines.append(f'| {LABELS.get(entry["model"], entry["model"])} | {"; ".join(entry["reasons"])} |')
        lines.extend(['',
                      '| 방법 | 측정 W&B | 별도 FLOPs W&B |', '|---|---|---|'])
        for method in methods:
            links = ' · '.join(f'[run{i}]({url})' for i, url in enumerate(method['wandb_urls'], 1) if url)
            flops_url = method.get('flops_wandb_url')
            lines.append(f'| {method["label"]} | {links or "—"} | {"[run](" + flops_url + ")" if flops_url else "—"} |')
    lines.extend(['', 'GPU 보드 에너지는 전체 머신 에너지가 아닙니다. seed1993 반복의 표준편차는 실행 비용 변동이며 seed 간 불확실성이 아닙니다. CUDA allocator peak와 device memory, 프로세스 RSS는 범위가 다릅니다. Recorder는 최종 W&B artifact upload/finish 전까지 측정합니다.', '',
                  '[JSON 결과](results.json) · [CSV 결과](results.csv)', ''])
    return '\n'.join(lines)


def csv_rows(report):
    rows = []
    for row in report['validation']['accuracy_metrics']:
        rows.append({'kind': 'accuracy_validation', 'model': 'hide-prompt', 'metric': row['metric'], **row})
    for method in report['efficiency']['methods']:
        context = {'model': method['model'], 'historical_reuse': method['historical_reuse']}
        for key, values in method['resources'].items():
            rows.append({'kind': 'efficiency_resource', **context, 'metric': key, **values})
        rows.append({'kind': 'efficiency_accuracy', **context, 'metric': 'accuracy_pct', **method['accuracy_pct']})
        for phase, values in method['phases'].items():
            for key, observed in values.items():
                skipped = phase in method['not_applicable_phases']
                rows.append({'kind': 'alignment_hook_overhead' if skipped else 'efficiency_phase', **context,
                             'phase': phase, 'status': 'not_applicable' if skipped else 'measured',
                             'metric': key, **observed})
        for phase, values in method['parameters'].items():
            for key, observed in values.items():
                rows.append({'kind': 'parameters', **context, 'phase': phase, 'metric': key, **observed})
        rows.append({'kind': 'flops_diagnostic', **context, 'metric': 'partial_supported_ops_total',
                     'value': method['flops'].get('supported_ops_total'), 'measurement_aggregate_eligible': False})
    return rows


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = report['efficiency']['methods']
    if not methods:
        figure, axis = plt.subplots(figsize=(7.2, 3.2), layout='constrained')
        metrics = report['validation']['accuracy_metrics']
        positions = list(range(len(metrics)))
        axis.bar([index - .18 for index in positions], [row['reference_accuracy_pct'] for row in metrics], width=.36, label='Pinned author reference', color='#71869b')
        native = [row['native_accuracy_pct'] if row['native_accuracy_pct'] is not None else float('nan') for row in metrics]
        axis.bar([index + .18 for index in positions], native, width=.36, label='Native HiDe-Prompt', color='#176385')
        axis.set_xticks(positions, [row['metric'].replace('_', '\n') for row in metrics])
        axis.set_ylabel('Accuracy (%)'); axis.set_ylim(0, 105)
        axis.set_title('HiDe-Prompt correctness; efficiency not executed')
        axis.legend(frameon=False)
    else:
        figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), layout='constrained')
        labels = [method['label'] + ('\nTII1+P1' if method['model'] == 'hide-prompt' else '\n1 epoch') for method in methods]
        colors = ['#176385' if method['model'] == 'hide-prompt' else '#879bab' for method in methods]
        panels = [('process_wall_s', 'Process wall time (s)', 1.),
                  ('gpu_board_energy_wh', 'GPU-board energy (Wh)', 1.),
                  ('cuda_peak_allocated_mib', 'CUDA allocator peak (GiB)', 1/1024),
                  ('peak_process_tree_rss_mib', 'Process-tree RSS peak (GiB)', 1/1024),
                  ('accuracy', 'Short-pilot accuracy (%)', 1.),
                  ('flops', 'Partial supported operators (TFLOPs)', 1e-12)]
        for axis, (key, title, factor) in zip(axes.flat, panels):
            if key == 'flops':
                means = [method['flops'].get('supported_ops_total') for method in methods]
                errors = None
            else:
                stats = [method['accuracy_pct'] if key == 'accuracy' else method['resources'][key] for method in methods]
                means = [values['mean'] for values in stats]
                errors = [values['std_sample'] * factor if values['std_sample'] is not None else 0 for values in stats]
            axis.bar(range(len(methods)), [value * factor if value is not None else float('nan') for value in means],
                     color=colors, yerr=errors, capsize=3, width=.68,
                     error_kw={'linewidth': 1, 'capthick': 1})
            axis.set_xticks(range(len(methods)), labels, fontsize=8)
            axis.set_title(title, fontsize=10, loc='left')
            axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
            axis.spines[['top', 'right']].set_visible(False)
            if key == 'accuracy':
                axis.set_ylim(0, 105)
        figure.suptitle('First-task efficiency: mean ± sample SD of 3 same-seed executions\nHiDe uses two training phases; historical methods use one outer epoch', fontsize=11)
        notes = ['GPU-board energy excludes CPU/RAM/whole-machine power. FLOPs come from a separate partial-operator pass.']
        missing_threads = [method['label'] for method in methods if method['historical_reuse']
                           and method.get('compatibility', {}).get('cpu_thread_evidence', {}).get('status') == 'metadata_missing']
        exclusions = report['efficiency'].get('historical_exclusions', [])
        metadata_notes = []
        if missing_threads:
            metadata_notes.append('Historical CPU-thread metadata missing: ' + ', '.join(missing_threads))
        if exclusions:
            metadata_notes.append('Excluded from matched inputs: ' + ', '.join(LABELS.get(entry['model'], entry['model']) for entry in exclusions))
        if metadata_notes:
            notes.append('. '.join(metadata_notes) + '.')
        figure.supxlabel('\n'.join(notes), fontsize=8)
    figure.savefig(output / 'efficiency_comparison.png', dpi=240)
    figure.savefig(output / 'efficiency_comparison.pdf')
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--efficiency', type=Path)
    parser.add_argument('--historical', type=Path, default=ROOT / 'benchmarks/cl_efficiency/baselines/efficiency_pilot_results.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/hide_prompt')
    options = parser.parse_args(argv)
    reference = read_json(options.reference)
    reference = reference.get('official_reference', reference)
    native = read_json(options.native)
    historical = options.historical if options.historical.is_file() else None
    report = build_report(reference, native, options.efficiency, historical)
    report['sources'] = {name: {'file': str(path.resolve()), 'sha256': file_hash(path)}
                         for name, path in (('reference', options.reference), ('native', options.native),
                                            ('efficiency', options.efficiency), ('historical', historical)) if path is not None}
    if options.efficiency is not None:
        companion = options.efficiency.with_name('execution.json')
        if 'measurements' not in read_json(options.efficiency) and companion.is_file() and companion.resolve() != options.efficiency.resolve():
            report['sources']['efficiency_execution_companion'] = {'file': str(companion.resolve()), 'sha256': file_hash(companion)}
    report['report_source_sha256'] = file_hash(__file__)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', report)
    rows = csv_rows(report)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)
    (output / 'report.md').write_text(markdown(report))
    plot_report(report, output)
    print(json.dumps({'status': 'completed', 'output': str(output),
                      'accuracy_acceptance': report['validation']['acceptance']['status'],
                      'efficiency': report['efficiency']['status']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
