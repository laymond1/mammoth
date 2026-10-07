"""Build EASE shared-input full-CL reports from immutable saved observations.

No model, training, GPU, credentials or network services are accessed. Optional
HiDe comparisons require observed inputs and precision to match; method phase
budgets remain different. Matplotlib only renders standalone report figures.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.ease_acceptance import evaluate_acceptance
from scripts.ease_fullcl_protocol import PRECISION, STRUCTURAL_CHECKS, protocol
from scripts.report_hide_fullcl import RESOURCE_KEYS, file_hash, finite, read_json, task_observations

REPORT_FILES = ('results.json', 'results.csv', 'report.md', 'accuracy_matrix.png',
                'accuracy_matrix.pdf', 'task_resources.png', 'task_resources.pdf')
RAW_FIELDS = ('n_samples', 'class_counts', 'samples_ordered_sha256', 'samples_unordered_sha256',
              'global_ids_ordered_sha256', 'transform_repr_sha256')
PHASE_GROUPS = {'adapter_training': ('train_epoch',),
                'prototype_completion': ('prototype_completion',), 'evaluation': ('evaluation',)}


def completed(record, mode, *, native=True):
    if (record.get('status') != 'completed' or record.get('model') != 'ease'
            or record.get('mode') != mode
            or record.get('protocol') != protocol('validation' if mode == 'reference' else mode)
            or record.get('wandb_logging_status') != 'uploaded'
            or record.get('wandb_artifact', {}).get('state') != 'COMMITTED'):
        raise ValueError('EASE requires completed uploaded COMMITTED records for the approved recipe')
    precision = record.get('precision', {})
    if set(precision) != set(PRECISION) or any(type(precision[key]) is not type(value)
                                            or precision[key] != value for key, value in PRECISION.items()):
        raise ValueError('EASE saved precision differs from the matched recipe')
    checks = record.get('structural_checks', {})
    if native and any(checks.get(key) is not True for key in STRUCTURAL_CHECKS):
        raise ValueError('EASE native structural gates must all pass')
    if not native and (not checks or any(value is not True for value in checks.values())):
        raise ValueError('EASE independent reference structural gates must all pass')


def matched_validation(reference, native):
    """Recompute four exact-count margins instead of trusting saved status text."""
    rows = reference.get('task_results', reference.get('tasks_results', []))
    if len(rows) != 2 or [row.get('task') for row in rows] != [1, 2]:
        raise ValueError('Matched EASE reference must contain exactly two ordered tasks')
    policy = {'unit': 'percentage_points', 'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3., 'metrics': []}
    for task, partition in ((1, 'seen'), (2, 'seen'), (2, 'old'), (2, 'new')):
        row = rows[task - 1]
        correct, samples = row.get(partition + '_correct'), row.get(partition + '_samples')
        if type(correct) is not int or type(samples) is not int or samples < 1 or not 0 <= correct <= samples:
            raise ValueError('Matched reference accuracy counts are invalid')
        policy['metrics'].append({'metric': f'task{task}_{partition}', 'task': task,
                                  'accuracy_key': partition + '_accuracy_pct',
                                  'reference_correct': correct, 'reference_samples': samples})
    observed = native.get('task_results', native.get('tasks_results', []))[:2]
    acceptance = evaluate_acceptance(observed, policy, structural_checks=native.get('structural_checks', {}))
    by_side = [{row.get('task'): row for row in record.get('first_batch_audits', []) if row.get('epoch', 1) == 1}
               for record in (reference, native)]
    batches = []
    for task in (1, 2):
        left, right = (indexed.get(task, {}) for indexed in by_side)
        keys = ('seed', 'batch_size', 'inputs_sha256', 'labels_sha256')
        match = all(left.get(key) is not None and left[key] == right.get(key) for key in keys)
        batches.append({'task': task, 'status': 'pass' if match else 'fail',
                        'gradient_sha256_matches': left.get('gradient_sha256') is not None
                        and left.get('gradient_sha256') == right.get('gradient_sha256')})
    metrics = []
    for task, partition in ((1, 'seen'), (2, 'seen'), (2, 'old'), (2, 'new')):
        left, right = rows[task - 1], observed[task - 1] if len(observed) >= task else {}
        samples, correct = right.get(partition + '_samples'), right.get(partition + '_correct')
        reference_accuracy = Fraction(100 * left[partition + '_correct'], left[partition + '_samples'])
        valid = type(samples) is int and type(correct) is int and samples == left[partition + '_samples'] and 0 <= correct <= samples
        native_accuracy = Fraction(100 * correct, samples) if valid else None
        gap = reference_accuracy - native_accuracy if valid else None
        status = 'fail' if gap is None or gap > 3 else 'review' if gap > Fraction(3, 2) else 'pass'
        metrics.append({'metric': f'task{task}_{partition}', 'reference_accuracy_pct': float(reference_accuracy),
                        'native_accuracy_pct': float(native_accuracy) if valid else None,
                        'reference_minus_native_pp': float(gap) if valid else None, 'status': status,
                        'gap_exact': {'numerator': gap.numerator, 'denominator': gap.denominator} if valid else None})
    return {'acceptance': acceptance, 'accuracy_metrics': metrics, 'first_batches': batches,
            'report_gate': 'pass' if acceptance['status'] == 'pass' and all(row['status'] == 'pass' for row in batches) else 'blocked',
            'scope': 'Independent author/native first-two-task counts and actual first-batch inputs/labels/seed/batch64. Gradient hashes are diagnostics.'}


def resource_observations(record):
    rows = record.get('per_task_resources', [])
    if len(rows) != 10 or [row.get('task') for row in rows] != list(range(1, 11)):
        raise ValueError('Full EASE requires ten ordered measured task intervals')
    tasks, previous_end = [], None
    for task, row in enumerate(rows, 1):
        start = finite(row.get('start_s'), 'task start', minimum=0)
        end = finite(row.get('end_s'), 'task end', minimum=start)
        wall = finite(row.get('wall_s'), 'task wall time', minimum=0)
        if abs(end - start - wall) > 1e-6 or previous_end is not None and start < previous_end - 1e-6:
            raise ValueError('EASE task intervals overlap or are inconsistent')
        previous_end = end
        metrics = row.get('metrics', {})
        for key, expected in (('outer_observe_steps', 1580), ('outer_training_stream_examples', 100000),
                              ('evaluation_examples', task * 1000)):
            if type(metrics.get(key)) is not int or metrics[key] != expected:
                raise ValueError('EASE task sample/step counts disagree with the recipe')
        measured = {}
        for key in RESOURCE_KEYS:
            value = metrics.get(key)
            measured[key] = finite(value, key, minimum=0, maximum=1 if key == 'gpu_energy_coverage_fraction' else None) if value is not None else None
        phases, groups = row.get('phases', {}), {}
        for name, members in PHASE_GROUPS.items():
            if any(member not in phases for member in members):
                raise ValueError('EASE task resources omit a measured method phase')
            energy = [phases[member].get('gpu_board_energy_j') for member in members]
            groups[name] = {'wall_s': sum(finite(phases[member].get('wall_s'), 'phase time', minimum=0) for member in members),
                            'gpu_board_energy_wh': sum(finite(value, 'phase GPU energy', minimum=0) for value in energy) / 3600
                            if all(value is not None for value in energy) else None}
        tasks.append({'task': task, 'wall_s': wall, 'metrics': measured, 'phases': phases, 'phase_groups': groups})
    return {'process': record['metrics'], 'tasks': tasks,
            'scope': 'Whole Recorder includes preparation, training, prototypes, evaluation and audits; final artifact upload/finish excluded. Task interval ends after evaluation/checkpoint audit before task Slack/logging. Phase times are inclusive.',
            'energy_scope': 'Selected GPU-board sampled power integral; CPU/RAM and total-machine energy are unmeasured.',
            'memory_scope': 'Per-task CUDA peaks are reset; process peak preserves initialization and all tasks. NVML/RSS peaks are sampled; summed process-tree RSS can duplicate shared pages.',
            'missing_values': 'null is unavailable, never imputed.'}


def input_compatibility(ease, hide):
    reasons = []
    if (hide.get('status') != 'completed' or hide.get('model') != 'hide-prompt' or hide.get('mode') != 'full_cl'
            or hide.get('wandb_logging_status') != 'uploaded' or hide.get('wandb_artifact', {}).get('state') != 'COMMITTED'):
        reasons.append('HiDe full10 is not completed/uploaded/COMMITTED')
    for key in ('checkpoint_sha256', 'canonical_pretrained_sha256'):
        if ease.get(key) is None or ease.get(key) != hide.get(key):
            reasons.append(key + ' differs or is unavailable')
    for key in ('dataset', 'dataset_config', 'tasks', 'dataset_total_tasks', 'classes_per_task',
                'vit_size', 'batch_size', 'seed', 'num_workers', 'drop_last', 'shuffle_training'):
        left, right = ease.get('protocol', {}).get(key), hide.get('protocol', {}).get(key)
        if left is None or type(left) is not type(right) or left != right:
            reasons.append('protocol.' + key + ' differs or is unavailable')
    if ease.get('gpu') is None or str(ease['gpu']) != str(hide.get('gpu')):
        reasons.append('physical GPU differs or is unavailable')
    for key in PRECISION:
        left, right = ease.get('precision', {}).get(key), hide.get('precision', {}).get(key)
        if left is None or type(left) is not type(right) or left != right:
            reasons.append('precision.' + key + ' differs or is unavailable')
    for task in range(1, 11):
        for split in ('train', 'test'):
            name = f'task{task}_{split}'
            left, right = (record.get('data_snapshots', {}).get(name, {}) for record in (ease, hide))
            for field in RAW_FIELDS:
                if left.get(field) is None or type(left[field]) is not type(right.get(field)) or left[field] != right.get(field):
                    reasons.append(name + '.' + field + ' differs or is unavailable')
    ease_batches = {row.get('task'): row for row in ease.get('first_batch_audits', []) if row.get('epoch', 1) == 1}
    hide_batches = {row.get('task'): row for row in hide.get('first_batch_audits', []) if row.get('phase') == 'tii_train'}
    for task in range(1, 11):
        left, right = ease_batches.get(task, {}), hide_batches.get(task, {})
        for source, target in (('inputs_sha256', 'input_sha256'), ('labels_sha256', 'labels_sha256'), ('seed', 'seed'), ('batch_size', 'samples')):
            if left.get(source) is None or left[source] != right.get(target):
                reasons.append(f'task{task} actual first-batch {source} differs or is unavailable')
    return {'status': 'compatible' if not reasons else 'excluded', 'reasons': sorted(set(reasons)),
            'scope': 'All ten raw train/test populations, observed post-begin-task transforms, first training batches, checkpoint, seed/batch/workers/GPU and eight recorded precision policies. The EASE train RNG matches HiDe TII; HiDe prompt-stage augmentation and all phase budgets remain method-specific.'}


def build_report(native, reference, accepted_native, hide=None):
    completed(native, 'full_cl'); completed(reference, 'reference', native=False); completed(accepted_native, 'validation')
    for key, expected in (('outer_observe_steps', 15800), ('outer_training_stream_examples', 1000000),
                          ('evaluation_examples', 55000), ('tasks_completed', 10)):
        if type(native.get('metrics', {}).get(key)) is not int or native['metrics'][key] != expected:
            raise ValueError('Full EASE aggregate sample/step counts are incomplete')
    epochs = native.get('epoch_results', [])
    if (len(epochs) != 200 or {(row.get('task'), row.get('epoch')) for row in epochs}
            != {(task, epoch) for task in range(1, 11) for epoch in range(1, 21)}
            or any(row.get('steps') != 79 or row.get('samples') != 5000 for row in epochs)):
        raise ValueError('Full EASE requires 200 completed native training epochs')
    for key in ('checkpoint_sha256', 'canonical_pretrained_sha256', 'initial_state_sha256', 'initial_canonical_pretrained_sha256'):
        if native.get(key) is None or native[key] != accepted_native.get(key) or native[key] != reference.get(key):
            raise ValueError('EASE checkpoint or author initialization changed')
    pins = native.get('execution_source_sha256', {})
    if not pins or pins != accepted_native.get('execution_source_sha256'):
        raise ValueError('EASE full10 sources differ from accepted validation')
    accepted, prefix = matched_validation(reference, accepted_native), matched_validation(reference, native)
    saved_prefix = native.get('prefix_validation', {})
    if (accepted_native.get('acceptance_status') != 'pass' or accepted['report_gate'] != 'pass'
            or prefix['report_gate'] != 'pass' or saved_prefix.get('status') != 'pass'
            or saved_prefix.get('matched_tasks') != [1, 2]
            or saved_prefix.get('prediction_match') is not True):
        raise ValueError('EASE matched validation and full10 prediction-prefix gates must pass')
    accuracy, resources = task_observations(native), resource_observations(native)
    observed = native.get('counts', {})
    work = {'outer_optimizer_steps': native['metrics']['outer_observe_steps'],
            'outer_image_training_examples': native['metrics']['outer_training_stream_examples'],
            'evaluation_image_examples': native['metrics']['evaluation_examples']}
    for key in ('prototype_examples', 'prototype_adapter_image_pairs', 'evaluation_adapter_image_pairs'):
        value = observed.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('EASE observed adapter-forward counts are invalid')
        work[key] = value
    work['scope'] = 'Saved image exposure and adapter-image pair counters; a pair counts one image through one adapter subspace. Repeated frozen-backbone passes are method work, not unique images, FLOPs or equal compute budgets.'
    comparison = {'status': 'not_provided', 'equal_compute_budget': False}
    if hide is not None:
        compatibility = input_compatibility(native, hide)
        comparison = {'status': compatibility['status'], 'compatibility': compatibility,
                      'equal_compute_budget': False, 'hide_wandb_url': hide.get('wandb_url'),
                      'schedule_scope': 'EASE20 epochs/task vs HiDe TII20+prompt50 and CA30/task2 onward; single seed and different optimization work.'}
        if compatibility['status'] == 'compatible':
            hide_accuracy = task_observations(hide)
            comparison['methods'] = [
                {'model': 'ease', 'accuracy': accuracy, 'metrics': native['metrics']},
                {'model': 'hide-prompt', 'accuracy': hide_accuracy, 'metrics': hide['metrics']}]
    return {'schema_version': 1, 'status': 'completed', 'model': 'ease', 'mode': 'full_cl',
            'protocol': native['protocol'], 'native_wandb_url': native.get('wandb_url'),
            'validation': {'accepted_two_task': accepted, 'current_prefix': prefix, 'prefix_prediction_match': saved_prefix,
                           'full_ten_task_accuracy_acceptance': 'not_assessed_by_matched_ten_task_reference'},
            'accuracy': accuracy, 'resources': resources, 'work_counts': work, 'comparison': comparison,
            'limitations': ['One native full10 run with seed1993; no seed variance or confidence interval.',
                            'Independent author/native acceptance covers two tasks, not a full10 paper-table reproduction.',
                            'Abnormally low accuracy is an implementation issue to investigate.',
                            'Different method schedules; no equal-compute-budget or full10 FLOPs comparison.']}


def verify_saved_predictions(result_file, tasks):
    directory = Path(result_file).resolve().parent
    manifest_file = directory / 'artifact_files.json'
    manifest = read_json(manifest_file)
    for task in range(1, tasks + 1):
        pin = manifest.get(f'task{task}_predictions.npz', {})
        path = Path(pin.get('local_file', ''))
        path = path if path.is_absolute() else directory / path
        if not path.is_file() or file_hash(path) != pin.get('sha256'):
            raise ValueError('Saved predictions differ from the immutable artifact manifest')
    return manifest_file


def audit_prediction_counts(result_file, tasks):
    """Recompute accuracy from all saved top-1 predictions on CPU only."""
    import numpy as np
    from scripts.ease_acceptance import counts_from_predictions
    result_file = Path(result_file).resolve()
    record = read_json(result_file)
    rows = record.get('task_results', record.get('tasks_results', []))
    if len(rows) != tasks or [row.get('task') for row in rows] != list(range(1, tasks + 1)):
        raise ValueError('Prediction audit requires complete ordered task result rows')
    verify_saved_predictions(result_file, tasks)
    for task, row in enumerate(rows, 1):
        with np.load(result_file.parent / f'task{task}_predictions.npz', allow_pickle=False) as archive:
            predictions, labels, ids, evaluation_ids = (archive[key] for key in
                                                        ('predictions', 'labels', 'global_ids', 'evaluation_global_ids'))
            if (predictions.ndim != 2 or predictions.shape[0] != task * 1000 or predictions.shape[1] < 1
                    or labels.shape != (task * 1000,) or ids.shape != labels.shape
                    or not np.issubdtype(predictions.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer)
                    or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids)
                    or evaluation_ids.shape != ids.shape or not np.issubdtype(evaluation_ids.dtype, np.integer)
                    or not np.array_equal(np.sort(evaluation_ids), np.sort(ids))
                    or np.any(ids < 0) or np.any(ids >= 10000)
                    or any(np.sum(labels == label) != 100 for label in range(task * 10))):
                raise ValueError('Prediction archive sample counts, classes or raw IDs are invalid')
            counts = counts_from_predictions(predictions[:, 0], labels, past_classes=(task - 1) * 10, seen_classes=task * 10)
            if any(row.get(key) != value for key, value in counts.items()):
                raise ValueError('Saved result accuracy counts disagree with raw prediction archives')
            matrix = row.get('native_class_il_per_task')
            if matrix is not None:
                measured = [100 * np.sum((predictions[:, 0] == labels) & (labels // 10 == index)) / 1000 for index in range(task)]
                if len(matrix) != task or any(abs(left - right) > 1e-8 for left, right in zip(matrix, measured)):
                    raise ValueError('Saved accuracy matrix disagrees with raw prediction archives')


def audit_prefix_predictions(native_file, accepted_file):
    import numpy as np
    for task in (1, 2):
        with np.load(Path(native_file).resolve().parent / f'task{task}_predictions.npz', allow_pickle=False) as full, \
                np.load(Path(accepted_file).resolve().parent / f'task{task}_predictions.npz', allow_pickle=False) as accepted:
            if any(key not in full or key not in accepted or not np.array_equal(full[key], accepted[key])
                   for key in ('predictions', 'labels', 'global_ids', 'evaluation_global_ids')):
                raise ValueError('Full EASE first-two-task prediction arrays differ from accepted validation')


def render_markdown(report):
    accuracy, process, work = report['accuracy'], report['resources']['process'], report['work_counts']
    def display(value, digits=2):
        return 'N/A' if value is None else f'{value:.{digits}f}'
    lines = ['# EASE Base 공통 입력 전체 10태스크 결과', '',
             f'최종 class-IL 정확도 **{accuracy["final_accuracy_pct"]:.2f}%**, 평균 증분 정확도 **{accuracy["average_incremental_accuracy_pct"]:.2f}%**, 과거9태스크 평균 망각 **{accuracy["forgetting_pp"]:.2f}%p**입니다.', '',
             'GPU1·batch64·workers4·seed1993, ViT-B/16 AugReg IN21K → IN1K와 adapter64를 사용했습니다. CIFAR100 고정 순서10클래스/태스크이며 실제 학습·평가 전처리는 HiDe 공통 입력 조건입니다. 태스크마다20epoch를 학습했습니다.', '',
             '새 batch64·공통 입력의 독립 저자/native 첫2태스크에서 정확도 하락1.5%p 이내 통과, 1.5~3.0%p 추가 점검, 3.0%p 초과 실패 기준을 적용했습니다. Full10의 첫2태스크 예측도 accepted native와 일치했습니다. 전체10태스크 독립 기준선이나 논문 표 재현으로 판정하지 않습니다. 비정상적으로 낮은 성능은 구현 문제로 조사합니다.', '',
             f'[Native W&B]({report.get("native_wandb_url")})', '',
             '| 전체 실행 지표 | 관측값 |', '|---|---:|']
    for key, label in (('process_wall_s', 'Recorder 실행 시간(s)'), ('gpu_board_energy_wh', 'GPU 보드 에너지(Wh)'),
                       ('process_tree_cpu_s', '프로세스·자식 CPU 시간(s)'), ('cuda_peak_allocated_mib', 'CUDA allocated peak(MiB)'),
                       ('cuda_peak_reserved_mib', 'CUDA reserved peak(MiB)'), ('peak_gpu_device_memory_mib', 'NVML device peak(MiB)'),
                       ('peak_process_tree_rss_mib', '프로세스·자식 RSS peak(MiB)')):
        lines.append('| ' + label + ' | ' + display(process.get(key)) + ' |')
    coverage = process.get('gpu_energy_coverage_fraction')
    lines += ['| GPU 에너지 sampling coverage(%) | ' + display(coverage * 100 if coverage is not None else None, 4) + ' |', '',
              f'실제 optimizer update **{work["outer_optimizer_steps"]:,}회**, 외부 학습 이미지 노출 **{work["outer_image_training_examples"]:,}개**, 누적 평가 노출 **{work["evaluation_image_examples"]:,}개**입니다. Prototype/evaluation의 adapter-image pair는 한 이미지를 하나의 adapter subspace에 통과시킨 관측 횟수로, 고유 이미지 수나 FLOPs가 아닙니다.', '',
              '| 태스크 | 관측 정확도(%) | 과거(%) | 신규(%) | 시간(s) | GPU 보드(Wh) | allocated(MiB) | reserved(MiB) | NVML(MiB) | RSS(MiB) |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row, cost in zip(accuracy['trajectory'], report['resources']['tasks']):
        values = (row['seen_accuracy_pct'], row['old_accuracy_pct'], row['new_accuracy_pct'], cost['wall_s'],
                  *(cost['metrics'][key] for key in ('gpu_board_energy_wh', 'cuda_peak_allocated_mib', 'cuda_peak_reserved_mib',
                                                    'peak_gpu_device_memory_mib', 'peak_process_tree_rss_mib')))
        lines.append('| ' + str(row['task']) + ' | ' + ' | '.join(display(value) for value in values) + ' |')
    comparison = report['comparison']
    lines += ['', '망각은 각 과거 태스크의 최종 태스크 이전 최고 정확도에서 최종 정확도를 빼고9개 태스크에 대해 평균합니다. 개선되면 음수를 유지합니다. 미래 행렬 항목은 null입니다. 평균 증분 정확도는 관측 클래스 정확도의10회 평균입니다.', '']
    if comparison['status'] == 'compatible':
        lines += ['실제10태스크 데이터·전처리·첫 배치·가중치·batch/seed/workers/GPU/precision을 대조해 HiDe와 공통 입력 비교에 포함했습니다. EASE20epoch/task와 HiDe TII20 + prompt50 + 이후CA30은 학습 작업량이 다르며 동일 계산 예산 비교가 아닙니다.', '',
                  '| 방법 | 최종 정확도(%) | 평균 증분(%) | 망각(%p) | Recorder 시간(s) | GPU 보드(Wh) |', '|---|---:|---:|---:|---:|---:|']
        for method in comparison['methods']:
            row = method['accuracy']; metric = method['metrics']
            lines.append('| ' + method['model'] + ' | ' + ' | '.join(display(value) for value in
                         (row['final_accuracy_pct'], row['average_incremental_accuracy_pct'], row['forgetting_pp'],
                          metric.get('process_wall_s'), metric.get('gpu_board_energy_wh'))) + ' |')
    elif comparison['status'] == 'excluded':
        lines += ['HiDe 비교에서 제외했습니다: ' + '; '.join(comparison['compatibility']['reasons']) + '.', '']
    else:
        lines += ['HiDe 비교 원본이 제공되지 않았습니다.', '']
    lines += ['', 'GPU 에너지는 선택 보드의 power sampling 적분이며 전체 머신 전력이 아닙니다. CUDA peak는 태스크마다 reset하며 전체 peak는 초기화와 모든 태스크 최대값을 보존합니다. NVML/RSS는 sampling 최대값이고 RSS는 공유 페이지를 중복 합산할 수 있습니다. 전체 Recorder는 마지막 artifact upload/finish를 제외하고, 태스크 interval은 태스크별 Slack/logging도 제외합니다. Phase 시간은 inclusive이므로 중첩 항목을 합산하지 않습니다.', '',
              '한 seed의 한 실행입니다. Tiny·Small 및 BudgetCL/aL-SAR 계산 예산 비교는 후속 단계입니다.', '']
    return '\n'.join(lines)


def render_plots(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    accuracy = report['accuracy']; tasks = report['resources']['tasks']; x = list(range(1, 11))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    matrix = np.array([[np.nan if value is None else value for value in row] for row in accuracy['class_il_accuracy_matrix_pct']])
    image = axes[0].imshow(matrix, vmin=0, vmax=100, cmap='viridis')
    axes[0].set(xlabel='Evaluated task', ylabel='Tasks learned', xticks=range(10), xticklabels=x,
                yticks=range(10), yticklabels=x, title='EASE class-IL accuracy (%)')
    figure.colorbar(image, ax=axes[0])
    axes[1].plot(x, [row['seen_accuracy_pct'] for row in accuracy['trajectory']], marker='o', label='EASE seen classes')
    if report['comparison']['status'] == 'compatible':
        other = report['comparison']['methods'][1]['accuracy']
        axes[1].plot(x, [row['seen_accuracy_pct'] for row in other['trajectory']], marker='s', label='HiDe seen classes')
    axes[1].set(xlabel='Tasks learned', ylabel='Accuracy (%)', xticks=x, ylim=(0, 100), title='Different method schedules, one seed')
    axes[1].legend(); axes[1].grid(alpha=.2)
    for extension in ('png', 'pdf'):
        figure.savefig(directory / ('accuracy_matrix.' + extension), dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in PHASE_GROUPS:
        axes[0, 0].plot(x, [row['phase_groups'][name]['wall_s'] for row in tasks], marker='o', label=name)
    axes[0, 0].plot(x, [row['wall_s'] for row in tasks], linestyle='--', color='black', label='Task interval')
    axes[0, 0].set(ylabel='Wall time (s)', title='EASE phase and task costs'); axes[0, 0].legend()
    for key, label in (('cuda_peak_allocated_mib', 'CUDA allocated'), ('cuda_peak_reserved_mib', 'CUDA reserved'),
                       ('peak_gpu_device_memory_mib', 'NVML device')):
        axes[1, 0].plot(x, [row['metrics'][key] / 1024 if row['metrics'][key] is not None else None for row in tasks], marker='o', label=label)
    axes[1, 0].set(ylabel='Memory peak (GiB)', title='GPU memory footprint', ylim=(0, None)); axes[1, 0].legend()
    for axis, key, label in ((axes[0, 1], 'gpu_board_energy_wh', 'GPU board energy (Wh)'),
                            (axes[1, 1], 'peak_process_tree_rss_mib', 'Process-tree RSS peak (GiB)')):
        values = [row['metrics'][key] / (1024 if key.endswith('_mib') else 1) if row['metrics'][key] is not None else None for row in tasks]
        axis.plot(x, values, marker='o'); axis.set(ylabel=label, title=label)
    for axis in axes.flat:
        axis.set(xlabel='Task', xticks=x); axis.grid(alpha=.2)
    for extension in ('png', 'pdf'):
        figure.savefig(directory / ('task_resources.' + extension), dpi=160)
    plt.close(figure)


def save_report(native_file, reference_file, accepted_native_file, directory, hide_file=None):
    paths = {name: Path(path).resolve() for name, path in
             (('native', native_file), ('reference', reference_file), ('accepted_native', accepted_native_file))}
    hide = read_json(hide_file) if hide_file is not None else None
    report = build_report(*(read_json(paths[name]) for name in ('native', 'reference', 'accepted_native')), hide)
    for name, tasks in (('native', 10), ('reference', 2), ('accepted_native', 2)):
        paths[name + '_artifact_manifest'] = verify_saved_predictions(paths[name], tasks)
        audit_prediction_counts(paths[name], tasks)
    audit_prefix_predictions(paths['native'], paths['accepted_native'])
    if hide_file is not None:
        paths['hide_native'] = Path(hide_file).resolve()
        paths['hide_artifact_manifest'] = verify_saved_predictions(paths['hide_native'], 10)
        audit_prediction_counts(paths['hide_native'], 10)
    report['sources'] = {name: {'file': str(path), 'sha256': file_hash(path)} for name, path in paths.items()}
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'results.json', report)
    with (directory / 'results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('task', 'seen_accuracy_pct', 'old_accuracy_pct', 'new_accuracy_pct', 'wall_s', *RESOURCE_KEYS))
        for row, cost in zip(report['accuracy']['trajectory'], report['resources']['tasks']):
            writer.writerow((row['task'], row['seen_accuracy_pct'], row['old_accuracy_pct'], row['new_accuracy_pct'],
                             cost['wall_s'], *(cost['metrics'][key] for key in RESOURCE_KEYS)))
    (directory / 'report.md').write_text(render_markdown(report)); render_plots(report, directory)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'reference', 'accepted-native', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--hide-native', type=Path)
    options = parser.parse_args(argv)
    report = save_report(options.native, options.reference, options.accepted_native, options.output, options.hide_native)
    print(json.dumps({'status': report['status'], 'output': str(options.output.resolve()),
                      'final_accuracy_pct': report['accuracy']['final_accuracy_pct'], 'hide_comparison': report['comparison']['status']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
