"""Summarize a completed ten-task HiDe run from saved observations only.

No training, GPU, credentials or network are accessed. Future accuracy matrix
entries remain missing. Forgetting uses the best *previous* accuracy, allowing
negative values when the final model improves an older task.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS
from scripts.hide_protocol import FULL_CL_PRECISION, protocol
from scripts.report_hide_results import exact_validation, file_hash, read_json

REPORT_FILES = ('results.json', 'results.csv', 'report.md',
                'accuracy_matrix.png', 'accuracy_matrix.pdf',
                'task_resources.png', 'task_resources.pdf')
PHASE_GROUPS = {'training': ('tii_train', 'prompt_train'),
                'statistics': ('tii_statistics', 'prompt_statistics'),
                'alignment': ('tii_alignment', 'prompt_alignment'),
                'evaluation': ('evaluation',)}
RESOURCE_KEYS = ('gpu_board_energy_wh', 'gpu_energy_coverage_fraction',
                 'cuda_peak_allocated_mib', 'cuda_peak_reserved_mib',
                 'peak_gpu_device_memory_mib', 'peak_process_tree_rss_mib',
                 'process_tree_cpu_s')


def finite(value, name, *, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f'Invalid saved {name}')
    return value


def task_observations(record):
    """Audit all lower-triangle entries against exact aggregate sample counts."""
    rows = record.get('task_results', [])
    if len(rows) != 10 or [row.get('task') for row in rows] != list(range(1, 11)):
        raise ValueError('Completed full CL requires exactly ten ordered task results')
    matrix, trajectory = [], []
    for task, row in enumerate(rows, 1):
        values = row.get('native_class_il_per_task', [])
        if len(values) != task or row.get('seen_samples') != 1000 * task:
            raise ValueError('Accuracy matrix or seen sample counts are incomplete')
        correct = []
        for value in values:
            value = finite(value, 'class-IL accuracy', minimum=0, maximum=100)
            count = round(value * 10)
            if abs(value * 10 - count) > 1e-7:
                raise ValueError('Per-task accuracy is incompatible with 1000 test samples')
            correct.append(count)
        count_keys = ('seen_correct', 'seen_samples', 'new_correct', 'new_samples', 'old_correct', 'old_samples')
        if (any(type(row.get(key)) is not int for key in count_keys) or row['seen_correct'] != sum(correct)
                or row.get('new_samples') != 1000 or row.get('new_correct') != correct[-1]
                or row.get('old_samples') != (task - 1) * 1000
                or row.get('old_correct') != sum(correct[:-1])):
            raise ValueError('Matrix and saved seen/old/new correct counts disagree')
        seen = 100 * sum(correct) / (task * 1000)
        if abs(finite(row.get('seen_accuracy_pct'), 'seen accuracy') - seen) > 1e-8:
            raise ValueError('Saved seen accuracy disagrees with exact correct counts')
        matrix.append(list(values) + [None] * (10 - task))
        trajectory.append({'task': task, 'seen_accuracy_pct': seen,
                           'old_accuracy_pct': 100 * sum(correct[:-1]) / ((task - 1) * 1000) if task > 1 else None,
                           'new_accuracy_pct': values[-1],
                           'tii_seen_accuracy_pct': row.get('tii_seen_accuracy_pct'),
                           'tii_task_routing_accuracy_pct': row.get('tii_task_routing_accuracy_pct')})
    forgetting = [{'task': index + 1,
                   'best_previous_accuracy_pct': max(matrix[time][index] for time in range(index, 9)),
                   'final_accuracy_pct': matrix[9][index],
                   'forgetting_pp': max(matrix[time][index] for time in range(index, 9)) - matrix[9][index]}
                  for index in range(9)]
    return {'class_il_accuracy_matrix_pct': matrix, 'trajectory': trajectory,
            'final_accuracy_pct': trajectory[-1]['seen_accuracy_pct'],
            'average_incremental_accuracy_pct': sum(row['seen_accuracy_pct'] for row in trajectory) / 10,
            'forgetting_pp': sum(row['forgetting_pp'] for row in forgetting) / 9,
            'forgetting_by_task': forgetting,
            'forgetting_definition': 'Mean over old tasks 1..9 of max accuracy after acquisition through task9 minus accuracy after task10; improvements remain negative.',
            'future_matrix_entries': 'null: unobserved, never zero',
            'test_samples_per_task': 1000, 'cumulative_evaluation_examples': 55000}


def resource_observations(record):
    rows = record.get('task_resources', [])
    if len(rows) != 10 or [row.get('task') for row in rows] != list(range(1, 11)):
        raise ValueError('Completed full CL requires ten measured task intervals')
    observed, previous_end = [], None
    for task, row in enumerate(rows, 1):
        start = finite(row.get('start_s'), 'task start', minimum=0)
        end = finite(row.get('end_s'), 'task end', minimum=start)
        wall = finite(row.get('wall_s'), 'task wall time', minimum=0)
        if abs(end - start - wall) > 1e-6 or (previous_end is not None and start < previous_end - 1e-6):
            raise ValueError('Task resource intervals are overlapping or inconsistent')
        previous_end = end
        metrics = row.get('metrics', {})
        for key, expected in (('outer_observe_steps', 5530), ('outer_training_stream_examples', 350000),
                              ('evaluation_examples', task * 1000)):
            if metrics.get(key) != expected:
                raise ValueError('Task resource sample/step counts disagree with full recipe')
        resources = {}
        for key in RESOURCE_KEYS:
            value = metrics.get(key)
            resources[key] = (finite(value, key, minimum=0,
                                     maximum=1 if key == 'gpu_energy_coverage_fraction' else None)
                              if value is not None else None)
        phases = row.get('phases', {})
        grouped = {}
        for name, members in PHASE_GROUPS.items():
            if any(member not in phases for member in members):
                raise ValueError('Task resources omit a measured method phase')
            wall_s = sum(finite(phases[member].get('wall_s'), 'phase wall time', minimum=0) for member in members)
            energies = [phases[member].get('gpu_board_energy_j') for member in members]
            grouped[name] = {'wall_s': wall_s,
                             'gpu_board_energy_wh': sum(finite(value, 'phase GPU energy', minimum=0) for value in energies) / 3600
                             if all(value is not None for value in energies) else None}
        observed.append({'task': task, 'wall_s': wall, 'metrics': resources,
                         'phase_groups': grouped, 'phases': phases})
    return {'process': record.get('metrics', {}), 'tasks': observed,
            'phase_groups': {key: list(value) for key, value in PHASE_GROUPS.items()},
            'scope': 'Task interval starts before loading and ends after evaluation/prediction/checkpoint audit; per-task Slack/logging and final artifact upload are excluded. Phase times are inclusive; do not sum nested phases.',
            'energy_scope': 'Sampled power integral for the selected GPU board; CPU/RAM and total machine energy are unmeasured.',
            'memory_scope': 'CUDA allocator peak is reset per task; NVML device and summed process-tree RSS are sampled peaks. RSS can count shared fork pages more than once.',
            'missing_resource_values': 'null means unavailable; values are never imputed.'}


def work_observations(record):
    """Preserve actual optimizer and generated-feature counts, without FLOPs."""
    alignments = []
    for row in record['alignment_history']:
        populations = row.get('sampled_examples_per_epoch')
        if (not isinstance(populations, list) or len(populations) != (0 if row['skipped'] else 30)
                or any(type(value) is not int or value < 1 for value in populations)
                or row.get('old_classes') != 10 * (row['task'] - 1)
                or row.get('samples_per_centroid') != (64 if row['branch'] == 'tii' else 320)):
            raise ValueError('Alignment generated-feature observations are incomplete')
        alignments.append({'task': row['task'], 'branch': row['branch'],
                           'optimizer_steps': row['optimizer_steps'],
                           'old_classes': row['old_classes'],
                           'samples_per_centroid': row['samples_per_centroid'],
                           'sampled_examples_per_epoch': populations,
                           'synthetic_feature_examples_generated': sum(populations),
                           'skipped': row['skipped']})
    outer_steps = record['metrics']['outer_observe_steps']
    alignment_steps = sum(row['optimizer_steps'] for row in alignments)
    extra_tii = record.get('counts', {}).get('prompt_training_tii_examples')
    if extra_tii is not None and (type(extra_tii) is not int or extra_tii < 0):
        raise ValueError('Observed additional TII forward image count is invalid')
    return {'outer_image_training_examples': record['metrics']['outer_training_stream_examples'],
            'evaluation_image_examples': record['metrics']['evaluation_examples'],
            'prompt_training_additional_tii_forward_image_examples': extra_tii,
            'outer_optimizer_steps': outer_steps,
            'alignment_optimizer_steps': alignment_steps,
            'total_optimizer_steps': outer_steps + alignment_steps,
            'synthetic_feature_examples_generated': sum(row['synthetic_feature_examples_generated'] for row in alignments),
            'synthetic_feature_examples_generated_by_branch': {
                branch: sum(row['synthetic_feature_examples_generated'] for row in alignments if row['branch'] == branch)
                for branch in ('tii', 'prompt')},
            'alignment_observations': alignments,
            'scope': 'Counts come from saved outer-loop metrics, counts.prompt_training_tii_examples and alignment history. Additional TII forward image examples reuse the prompt-stage images in a second frozen encoder; they are neither additional outer-loop images nor unique images. A missing counter remains null. Generated synthetic features are a sampled population, not raw images or classifier-consumed training examples. Optimizer steps cover heterogeneous operations and do not estimate FLOPs or equal compute budgets.'}


def build_report(native, reference, accepted_native):
    settings = native.get('protocol', {})
    if (native.get('status') != 'completed' or native.get('model') != 'hide-prompt'
            or native.get('mode') != 'full_cl' or settings != protocol('full_cl')):
        raise ValueError('Report requires the approved completed HiDe Base full-CL recipe')
    if native.get('wandb_logging_status') != 'uploaded' or native.get('wandb_artifact', {}).get('state') != 'COMMITTED':
        raise ValueError('Completed full CL requires its uploaded COMMITTED native artifact')
    if any(type(native.get('precision', {}).get(key)) is not type(value)
           or native['precision'][key] != value for key, value in FULL_CL_PRECISION.items()):
        raise ValueError('Full CL precision differs from accepted validation')
    for key, expected in (('outer_observe_steps', 55300), ('outer_training_stream_examples', 3500000),
                          ('evaluation_examples', 55000), ('tasks_completed', 10)):
        if type(native.get('metrics', {}).get(key)) is not int or native['metrics'][key] != expected:
            raise ValueError('Full CL aggregate execution counts are incomplete')
    alignments = native.get('alignment_history', [])
    coordinates = {(row.get('task'), row.get('branch')) for row in alignments}
    if len(alignments) != 20 or coordinates != {(task, branch) for task in range(1, 11) for branch in ('tii', 'prompt')}:
        raise ValueError('Full CL alignment history is incomplete')
    if any(row.get('optimizer_steps') != 300 * (row['task'] - 1)
           or row.get('skipped') is not (row['task'] == 1) for row in alignments):
        raise ValueError('Full CL alignment optimizer steps differ from the source schedule')
    checks = native.get('structural_checks', {})
    if any(checks.get(key) is not True for key in REQUIRED_STRUCTURAL_CHECKS):
        raise ValueError('Full CL structural checks must all pass')
    accepted = exact_validation(reference, accepted_native)
    prefix = dict(native, task_results=native.get('task_results', [])[:2])
    prefix.pop('acceptance_status', None)
    matched_prefix = exact_validation(reference, prefix)
    saved_prefix = native.get('prefix_validation', {})
    if (accepted['report_gate'] != 'pass' or matched_prefix['report_gate'] != 'pass'
            or saved_prefix.get('status') != 'pass' or saved_prefix.get('prediction_match') is not True):
        raise ValueError('Full CL requires the accepted matched two-task and prediction-prefix gates')
    if any(native.get(key) != accepted_native.get(key) for key in
           ('checkpoint_sha256', 'initial_state_sha256', 'initial_learned_sha256', 'initial_canonical_pretrained_sha256')):
        raise ValueError('Full CL checkpoint or learned initialization differs from accepted validation')
    return {'schema_version': 1, 'status': 'completed', 'model': 'hide-prompt',
            'mode': 'full_cl', 'protocol': settings, 'native_wandb_url': native.get('wandb_url'),
            'validation': {'accepted_two_task': accepted, 'current_prefix': matched_prefix,
                           'prefix_prediction_match': saved_prefix,
                           'full_ten_task_accuracy_acceptance': 'not_assessed_by_matched_ten_task_reference'},
            'accuracy': task_observations(native), 'resources': resource_observations(native),
            'work_counts': work_observations(native),
            'limitations': ['Single seed1993; no seed variation estimate or confidence interval.',
                            'Native full10 run; matched independent implementation validation covers only the first two tasks.',
                            'The author-code protocol and IN21K→IN1K weights are explicit adaptations; this is not paper-table reproduction.',
                            'Abnormally low accuracy is an implementation issue to investigate; no paper threshold is applied across different recipes.',
                            'No equal-compute-budget comparison or new full10 FLOPs profiling was performed.']}


def verify_saved_predictions(native_file):
    """Pin all ten prediction files through the worker's immutable manifest."""
    directory = Path(native_file).resolve().parent
    manifest_file = directory / 'artifact_files.json'
    manifest = read_json(manifest_file)
    for task in range(1, 11):
        filename = f'task{task}_predictions.npz'
        pin = manifest.get(filename, {})
        path = Path(pin.get('local_file', ''))
        path = path if path.is_absolute() else directory / path
        if not path.is_file() or file_hash(path) != pin.get('sha256'):
            raise ValueError('Full CL saved prediction differs from its immutable artifact pin')
    return manifest_file


def render_markdown(report):
    accuracy, resources = report['accuracy'], report['resources']
    work = report['work_counts']
    def formatted(value, digits=2):
        return 'N/A' if value is None else f'{value:.{digits}f}'
    process = resources['process']
    lines = ['# HiDe-Prompt Base 전체 10태스크 결과', '',
             f'최종 class-IL 정확도 **{accuracy["final_accuracy_pct"]:.2f}%**, 평균 증분 정확도 **{accuracy["average_incremental_accuracy_pct"]:.2f}%**, 과거 9태스크 평균 망각 **{accuracy["forgetting_pp"]:.2f}%p**입니다.', '',
             'GPU1·batch64·seed1993, CIFAR100 고정 순서 10클래스/태스크, ViT-B/16 AugReg IN21K → IN1K 가중치를 사용했습니다. 각 태스크는 TII20 + prompt50epoch이며, 태스크2부터 두 분류기에 CA30epoch를 적용했습니다.', '',
             '첫2태스크는 기존 독립 저자 기준선과 정확도·구조·첫 배치를 대조하고 저장 예측까지 일치했습니다. 전체10태스크에 대한 독립 기준선은 실행하지 않았으며 논문 표 재현으로 판정하지 않습니다. 비정상적으로 낮은 성능은 구현 문제로 조사합니다.', '',
             f'[Native W&B]({report.get("native_wandb_url")})', '',
             '| 전체 실행 지표 | 관측값 |', '|---|---:|',
             '| Recorder 실행 시간(s) | ' + formatted(process.get('process_wall_s')) + ' |',
             '| GPU 보드 에너지(Wh) | ' + formatted(process.get('gpu_board_energy_wh')) + ' |',
             '| CUDA allocated peak(MiB) | ' + formatted(process.get('cuda_peak_allocated_mib')) + ' |',
             '| CUDA reserved peak(MiB) | ' + formatted(process.get('cuda_peak_reserved_mib')) + ' |',
             '| GPU device memory peak(MiB) | ' + formatted(process.get('peak_gpu_device_memory_mib')) + ' |',
             '| 프로세스·자식 RSS peak(MiB) | ' + formatted(process.get('peak_process_tree_rss_mib')) + ' |',
             '| GPU 에너지 sampling coverage(%) | ' + formatted(process['gpu_energy_coverage_fraction'] * 100
                                                                               if process.get('gpu_energy_coverage_fraction') is not None else None, 4) + ' |', '',
             f'실제 optimizer update는 외부 TII/prompt 학습 **{work["outer_optimizer_steps"]:,}회**와 classifier alignment **{work["alignment_optimizer_steps"]:,}회**, 합계 **{work["total_optimizer_steps"]:,}회**입니다. '
             f'외부 학습 이미지 노출은 **{work["outer_image_training_examples"]:,}개**, 누적 평가 이미지 노출은 **{work["evaluation_image_examples"]:,}개**입니다. '
             f'Alignment가 생성한 synthetic feature는 저장된 매 epoch population의 합 **{work["synthetic_feature_examples_generated"]:,}개**이며 raw image 수와 별개입니다. 생성 population 전체를 classifier가 학습에 소비한 것은 아닙니다. '
             '각 optimizer step의 연산은 다르며 이 횟수를 FLOPs나 동일 계산 예산으로 해석하지 않습니다.', '',
             'Prompt 학습 중 별도로 수행한 고정 TII encoder forward의 이미지 노출은 **'
             + (f'{work["prompt_training_additional_tii_forward_image_examples"]:,}개'
                if work['prompt_training_additional_tii_forward_image_examples'] is not None else '미기록')
             + '**입니다. 이는 prompt 단계의 같은 이미지를 두 번째 encoder에도 통과시킨 관측 횟수이며 외부 학습 이미지·synthetic feature 수와 구분합니다. 학습 가능한 파라미터 수만으로는 이 추가 backbone 연산을 설명하지 못합니다.', '',
             '| 학습 완료 태스크 | 관측 클래스 정확도(%) | 과거 클래스(%) | 신규 클래스(%) | 시간(s) | GPU 보드(Wh) | CUDA peak(MiB) | RSS peak(MiB) |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row, cost in zip(accuracy['trajectory'], resources['tasks']):
        values = (row['seen_accuracy_pct'], row['old_accuracy_pct'], row['new_accuracy_pct'],
                  cost['wall_s'], cost['metrics']['gpu_board_energy_wh'],
                  cost['metrics']['cuda_peak_allocated_mib'], cost['metrics']['peak_process_tree_rss_mib'])
        lines.append('| ' + str(row['task']) + ' | ' + ' | '.join(formatted(value) for value in values) + ' |')
    lines += ['', '망각은 태스크1..9 각각에 대해 최종 태스크 이전의 최고 정확도에서 최종 정확도를 빼고 평균합니다. 후반 학습으로 개선되면 음수를 유지합니다. 정확도 행렬의 미래 태스크는 평가하지 않았으므로 null입니다. 평균 증분 정확도는 각 태스크 종료 시 관측 클래스 정확도의 10회 평균입니다.', '',
              'GPU 에너지는 선택된 GPU 보드의 power sampling 적분이며 전체 머신 전력은 아닙니다. CUDA peak는 태스크별 allocator 최대값이며 전체 실행 peak는 초기화와 모든 태스크의 최대값을 보존합니다. Device memory·RSS는 샘플 최대값입니다. RSS는 자식 프로세스의 공유 페이지를 중복 합산할 수 있습니다. Recorder 전체 시간은 초기화·전처리·학습·통계·정렬·평가·checkpoint·audit를 포함하며 마지막 artifact upload/finish는 제외합니다. 태스크 시간은 데이터 로딩부터 평가·예측·checkpoint 검사까지이며 태스크별 Slack/logging 및 마지막 artifact 업로드는 제외합니다. 중첩 phase 시간은 inclusive이므로 단순 합산하지 않습니다.', '',
              '한 seed의 한 실행이며 신뢰구간을 추정하지 않습니다. Tiny·Small 및 계산 예산을 맞춘 BudgetCL/aL-SAR 비교는 후속 단계입니다.', '']
    return '\n'.join(lines)


def render_plots(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    accuracy = report['accuracy']
    matrix = np.array([[np.nan if value is None else value for value in row]
                       for row in accuracy['class_il_accuracy_matrix_pct']])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    painted = axes[0].imshow(matrix, vmin=0, vmax=100, cmap='viridis')
    axes[0].set(xticks=range(10), xticklabels=range(1, 11), yticks=range(10), yticklabels=range(1, 11),
                xlabel='Evaluated task', ylabel='Tasks learned', title='Class-IL accuracy (%)')
    figure.colorbar(painted, ax=axes[0], label='Accuracy (%)')
    x = [row['task'] for row in accuracy['trajectory']]
    for key, label in (('seen_accuracy_pct', 'All seen classes'), ('old_accuracy_pct', 'Old classes'),
                       ('new_accuracy_pct', 'New classes')):
        axes[1].plot(x, [row[key] for row in accuracy['trajectory']], marker='o', label=label)
    axes[1].set(xlabel='Tasks learned', ylabel='Accuracy (%)', xticks=x, ylim=(0, 100), title='Single-seed trajectory')
    axes[1].legend(); axes[1].grid(alpha=.2)
    for extension in ('png', 'pdf'):
        figure.savefig(directory / ('accuracy_matrix.' + extension), dpi=160)
    plt.close(figure)
    tasks = report['resources']['tasks']
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name in PHASE_GROUPS:
        axes[0, 0].plot(x, [row['phase_groups'][name]['wall_s'] for row in tasks], marker='o', label=name)
    axes[0, 0].plot(x, [row['wall_s'] for row in tasks], color='black', linestyle='--', label='Task interval')
    axes[0, 0].set(ylabel='Wall time (s)', title='Task and method phase costs'); axes[0, 0].legend()
    for key, label in (('cuda_peak_allocated_mib', 'CUDA allocated'),
                       ('cuda_peak_reserved_mib', 'CUDA reserved'),
                       ('peak_gpu_device_memory_mib', 'NVML device')):
        values = [row['metrics'][key] / 1024 if row['metrics'][key] is not None else None for row in tasks]
        axes[1, 0].plot(x, values, marker='o', label=label)
    axes[1, 0].set(ylabel='Memory peak (GiB)', title='GPU memory footprint', ylim=(0, None))
    axes[1, 0].legend()
    for axis, key, label in ((axes[0, 1], 'gpu_board_energy_wh', 'GPU board energy (Wh)'),
                            (axes[1, 1], 'peak_process_tree_rss_mib', 'Process-tree RSS peak (GiB)')):
        values = [row['metrics'][key] for row in tasks]
        if key.endswith('_mib'):
            values = [value / 1024 if value is not None else None for value in values]
        axis.plot(x, values, marker='o'); axis.set(ylabel=label, title=label)
    for axis in axes.flat:
        axis.set(xlabel='Task', xticks=x); axis.grid(alpha=.2)
    for extension in ('png', 'pdf'):
        figure.savefig(directory / ('task_resources.' + extension), dpi=160)
    plt.close(figure)


def save_report(native_file, reference_file, accepted_native_file, directory):
    paths = {name: Path(path).resolve() for name, path in
             (('native', native_file), ('reference', reference_file), ('accepted_native', accepted_native_file))}
    report = build_report(*(read_json(paths[name]) for name in ('native', 'reference', 'accepted_native')))
    paths['native_artifact_manifest'] = verify_saved_predictions(paths['native'])
    report['sources'] = {name: {'file': str(path), 'sha256': file_hash(path)} for name, path in paths.items()}
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'results.json', report)
    with (directory / 'results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('task', 'seen_accuracy_pct', 'old_accuracy_pct', 'new_accuracy_pct',
                         'wall_s', *RESOURCE_KEYS, *(name + '_wall_s' for name in PHASE_GROUPS)))
        for row, cost in zip(report['accuracy']['trajectory'], report['resources']['tasks']):
            writer.writerow((row['task'], row['seen_accuracy_pct'], row['old_accuracy_pct'], row['new_accuracy_pct'],
                             cost['wall_s'], *(cost['metrics'][key] for key in RESOURCE_KEYS),
                             *(cost['phase_groups'][name]['wall_s'] for name in PHASE_GROUPS)))
    (directory / 'report.md').write_text(render_markdown(report))
    render_plots(report, directory)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--accepted-native', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args(argv)
    report = save_report(options.native, options.reference, options.accepted_native, options.output)
    print(json.dumps({'status': report['status'], 'output': str(options.output.resolve()),
                      'final_accuracy_pct': report['accuracy']['final_accuracy_pct']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
