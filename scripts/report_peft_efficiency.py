"""Read saved pilots, audit common inputs, and render a nine-method comparison.

No training, credentials or networking. Historical results retain their actual
metadata and measurement dates. --write produces local JSON/CSV/Markdown and
scientific PNG/PDF figures; profiling passes never enter cost distributions.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.low_rank_protocol import CHECKPOINT_SHA256
from scripts.report_hide_results import (compatible_history, cpu_thread_metadata,
    first_task_snapshot, method_summary, PRECISION_FIELDS, RAW_KEYS)
from scripts.run_peft_efficiency import file_hash, load_plan, METHODS, validate_schedule

REPORT_FILES = ('RESULTS.md', 'results.json', 'results.csv',
                'efficiency_comparison.png', 'efficiency_comparison.pdf')
LABELS = {'ewc-lora': 'EWC-LoRA', 'sd-lora': 'SD-LoRA', 'inf-lora': 'InfLoRA', 'cl-lora': 'CL-LoRA'}
BASELINE_FILE = ROOT / 'benchmarks/cl_efficiency/baselines/efficiency_pilot_results.json'
HIDE_FILE = ROOT / 'benchmarks/cl_efficiency/hide_prompt/results.json'
EASE_FILE = ROOT / 'benchmarks/cl_efficiency/ease/efficiency_pilot_results.json'


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Saved evidence must be a JSON object')
    return value


def historical_inputs():
    """Require raw completed records rather than constructing them from a summary."""
    baseline, ease, hide = (read_json(path) for path in (BASELINE_FILE, EASE_FILE, HIDE_FILE))
    source = hide['sources']['efficiency']
    hide_path = Path(source['file'])
    if file_hash(hide_path) != source['sha256']:
        raise ValueError('Saved HiDe efficiency source changed')
    raw_hide = read_json(hide_path)
    hide_gpus = {record.get('gpu') for record in raw_hide['measurements']}
    if len(hide_gpus) != 1 or None in hide_gpus:
        raise ValueError('HiDe historical physical GPU metadata is missing or inconsistent')
    result = {'ease': (ease, ease['measurements'], ease['flops_diagnostic'], EASE_FILE),
              'hide-prompt': ({'status': raw_hide['status'], 'pilot_status': raw_hide['summary']['status'],
                              'checkpoint_sha256': raw_hide['checkpoint_sha256'],
                              'protocol': {**raw_hide['protocol'], 'gpu': next(iter(hide_gpus))}},
                             raw_hide['measurements'], raw_hide['flops'], hide_path)}
    for name in ('l2p', 'dualprompt'):
        entry = baseline['models'][name]
        result[name] = (baseline, entry['measurements'], entry['flops'], BASELINE_FILE)
    return result


def validate_new_record(record, method, pass_name):
    """Fail closed on scope, precision, input, structure, collapse and sensors."""
    if (method not in METHODS or pass_name not in ('measurement', 'flops')
            or record.get('method', record.get('model')) != method
            or record.get('pass') != pass_name or record.get('status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded'
            or not str(record.get('wandb_url', '')).startswith('https://wandb.ai/laymond1/cl-efficiency/runs/')
            or record.get('checkpoint_sha256') != CHECKPOINT_SHA256):
        raise ValueError('Worker identity, completed upload or checkpoint is invalid')
    expected = {'tasks_completed': 1, 'outer_observe_steps': 79,
                'outer_training_stream_examples': 5000, 'evaluation_examples': 1000}
    if any(type(record.get('metrics', {}).get(key)) is not int or record['metrics'][key] != value
           for key, value in expected.items()):
        raise ValueError('Worker stream, update or evaluation counts are incomplete')
    precision = load_plan()['precision']
    if any(record.get('precision', {}).get(key) != precision[key] for key in PRECISION_FIELDS):
        raise ValueError('Worker precision differs from common FP32 policy')
    if cpu_thread_metadata(record)['value'] != 2:
        raise ValueError('New workers must record two Torch CPU threads')
    checks = record.get('structural_checks', {})
    if method == 'ranpac':
        from scripts.run_ranpac_efficiency import REQUIRED_CHECKS
    else:
        from scripts.run_low_rank_efficiency import REQUIRED_CHECKS
    if (any(checks.get(key) is not True for key in REQUIRED_CHECKS)
            or any(value is not True for value in checks.values())):
        raise ValueError('Every worker structural check must pass')
    from scripts.report_hide_results import accuracy
    observed = accuracy(record)
    if isinstance(observed, bool) or not isinstance(observed, (float, int)) or not math.isfinite(observed) or not 20 < observed <= 100:
        raise ValueError('Investigate abnormal one-epoch accuracy before accepting the worker')
    target = read_json(EASE_FILE)['measurements'][0]
    for split in ('train', 'test'):
        current, historical = first_task_snapshot(record, split), first_task_snapshot(target, split)
        if any(current.get(key) is None or current[key] != historical.get(key) for key in RAW_KEYS):
            raise ValueError(f'Actual {split} raw data or transform differs from the common EASE input')
    if pass_name == 'measurement':
        coverage = record['metrics'].get('gpu_energy_coverage_fraction')
        if not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or coverage < .99 or record.get('sensor_errors'):
            raise ValueError('Measurements require clean GPU telemetry and at least99% energy coverage')
        for key in ('process_wall_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib'):
            value = record['metrics'].get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Missing or invalid required efficiency measurement')
    else:
        value = record.get('flops', {}).get('supported_ops_total')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Separate profiler worker requires a positive supported-operator FLOPs estimate')
    return True


def build_report(execution_path):
    saved = read_json(execution_path)
    plan = load_plan()
    if saved.get('status') != 'completed' or saved.get('plan') != plan or len(saved.get('jobs', [])) != 20:
        raise ValueError('Comparison requires the complete approved 20-worker execution')
    validate_schedule(saved)
    records, provenance = {name: {'measurement': [], 'flops': []} for name in METHODS}, []
    seen_paths, seen_urls = set(), set()
    for job in saved['jobs']:
        if job.get('status') != 'completed':
            raise ValueError('A scheduled worker is incomplete')
        source = job['result']
        path = Path(source['file'])
        if (path.resolve() != Path(job['directory']).resolve() / 'result.json'
                or str(path.resolve()) in seen_paths or source['wandb_url'] in seen_urls):
            raise ValueError('Repeated runs require twenty distinct worker files and W&B run IDs')
        seen_paths.add(str(path.resolve()))
        seen_urls.add(source['wandb_url'])
        if file_hash(path) != source['sha256']:
            raise ValueError('Completed worker record changed')
        record = read_json(path)
        validate_new_record(record, job['method'], job['pass'])
        records[job['method']][job['pass']].append(record)
        provenance.append({'method': job['method'], 'pass': job['pass'], **source})
    target = records['ewc-lora']['measurement'][0]
    methods, history_proofs = {}, {}
    for name, (document, measurements, flops, path) in historical_inputs().items():
        proof = compatible_history(document, measurements, target)
        if proof['status'] != 'compatible':
            raise ValueError(f'Historical {name} cannot be reused: {proof["reasons"]}')
        methods[name] = method_summary(name, measurements, flops, protocol=document['protocol'], historical=True)
        history_proofs[name] = {**proof, 'file': str(path), 'sha256': file_hash(path)}
    for name in METHODS:
        item = records[name]
        if len(item['measurement']) != 3 or len(item['flops']) != 1:
            raise ValueError('Each new method requires three measurements and one separate profiler')
        methods[name] = method_summary(name, item['measurement'], item['flops'][0], protocol=plan)
        methods[name]['label'] = LABELS.get(name, methods[name]['label'])
        methods[name]['method_diagnostics'] = [{key: record.get(key) for key in
            ('counts', 'extra_samples', 'method_state_storage', 'unregistered_statistics_storage',
             'task_end_training_image_presentations', 'ridge_candidate_count', 'ridge_solve_count',
             'rp_size', 'auxiliary_storage', 'state_audits')} for record in item['measurement']]
    return {'schema_version': 1, 'status': 'completed', 'publication_gate': 'pass',
            'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'protocol': plan, 'method_count': 9, 'new_measurement_runs': 15, 'new_flops_runs': 5,
            'historical_reused_methods': ['ease', 'l2p', 'dualprompt', 'hide-prompt'],
            'methods': methods, 'historical_compatibility': history_proofs,
            'sources': provenance, 'execution': {'file': str(Path(execution_path).resolve()),
                                                'sha256': file_hash(execution_path),
                                                'execution_source_sha256': saved['execution_source_sha256']},
            'scope': {'first_task_only': True, 'paper_reproduction': False, 'equal_compute_budget': False,
                      'same_seed_repeats': True, 'seed_uncertainty_estimated': False,
                      'full_machine_energy_measured': False,
                      'historical_measurement_dates_and_source_snapshots_differ': True,
                      'same_augmentation_recipe_not_identical_random_crop_realizations': True,
                      'ease_cpu_threads': 'not_recorded', 'flops': 'partial supported-operator estimates'}}


def csv_text(report):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(('method', 'historical_reuse', 'accuracy_pct', 'wall_s_mean', 'wall_s_sd',
                     'cuda_allocated_peak_gib_mean', 'rss_peak_gib_mean', 'cpu_s_mean',
                     'gpu_board_wh_mean', 'gpu_board_wh_sd', 'partial_tflops', 'measurement_urls'))
    for name, method in report['methods'].items():
        resources = method['resources']
        writer.writerow((name, method['historical_reuse'], method['accuracy_pct']['mean'],
            resources['process_wall_s']['mean'], resources['process_wall_s']['std_sample'],
            resources['cuda_peak_allocated_mib']['mean'] / 1024,
            resources['peak_process_tree_rss_mib']['mean'] / 1024,
            resources['process_tree_cpu_s']['mean'], resources['gpu_board_energy_wh']['mean'],
            resources['gpu_board_energy_wh']['std_sample'], method['flops']['supported_ops_total'] / 1e12,
            ' '.join(method['wandb_urls'])))
    return stream.getvalue()


def markdown(report):
    rows = ['# Base 첫 태스크: 9방법 효율성 pilot', '',
        'LoRA4종과 RanPAC의 신규 측정15회·별도 부분 FLOPs5회가 완료됐습니다. EASE·L2P·DualPrompt·HiDe-Prompt는 실제 원본 데이터·전처리·가중치·precision을 대조한 기존 기록을 재사용했습니다.', '',
        'CIFAR100 첫10클래스(train5000/test1000), ViT-B/16 AugReg IN21K→IN1K, batch64, seed1993, GPU1, FP32입니다. 신규 방법은1epoch, HiDe는TII1+prompt1입니다. 각 방법 고유 task-begin/end 작업을 포함하므로 동일 계산 예산 비교가 아닙니다.', '',
        '| 방법 | 정확도 % | 벽시간 s, 평균±SD | CUDA peak GiB | RSS peak GiB | CPU s | GPU 보드 Wh, 평균±SD | 부분 TFLOPs |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in report['methods'].values():
        r = method['resources']
        rows.append(f'| {method["label"]} | {method["accuracy_pct"]["mean"]:.2f} | '
                    f'{r["process_wall_s"]["mean"]:.2f} ± {r["process_wall_s"]["std_sample"]:.2f} | '
                    f'{r["cuda_peak_allocated_mib"]["mean"] / 1024:.3f} | '
                    f'{r["peak_process_tree_rss_mib"]["mean"] / 1024:.3f} | '
                    f'{r["process_tree_cpu_s"]["mean"]:.2f} | '
                    f'{r["gpu_board_energy_wh"]["mean"]:.3f} ± {r["gpu_board_energy_wh"]["std_sample"]:.3f} | '
                    f'{method["flops"]["supported_ops_total"] / 1e12:.3f} |')
    rows += ['', '![효율성 비교](efficiency_comparison.png)', '', '[PDF 그림](efficiency_comparison.pdf)', '',
        '평균±표본 SD는 동일 seed의 실행 비용 변동입니다. 별도 profiler 실행의 시간·메모리·에너지는 평균에 넣지 않았습니다. GPU 에너지는 CPU/RAM/전체 머신 에너지를 포함하지 않습니다. 부분 FLOPs는 CPU 선형대수·미지원 연산 등 전체 연산을 보장하지 않습니다.', '',
        '기존 기록은 측정 날짜·소스가 다릅니다. EASE의 Torch CPU thread 수는 미기록으로 남깁니다. 공통 augmentation recipe와 원본 데이터가 같아도 방법별 RNG 경로에 따라 실제 crop은 달라질 수 있습니다. 전체10태스크 정확도·망각·동일 예산 성능을 추정하지 않습니다.', '',
        '첫 태스크에서는 EWC의 과거 지식 penalty, CL의 KD/과거 branch 학습, Inf의 과거 subspace 상호작용 비용이 나타나지 않습니다. Fisher·feature 수집/SVD/GPM·프로토타입·RanPAC RP/ridge 비용은 실제 수행한 범위로 포함합니다.', '',
        'RanPAC 과거 기록의 학습 전처리 hash는 task-end 특징 추출 전처리로 덮어써졌습니다. 새 실행은 실제 학습 전처리를 보존하고 task-end 전처리를 별도로 기록합니다.', '',
        '| 방법 | 측정 실행 | 별도 FLOPs |', '|---|---|---|']
    for method in report['methods'].values():
        links = ' · '.join(f'[측정{i}]({url})' for i, url in enumerate(method['wandb_urls'], 1))
        rows.append(f'| {method["label"]} | {links} | [진단]({method["flops_wandb_url"]}) |')
    return '\n'.join(rows) + '\n'


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    items = list(report['methods'].values())
    labels = [method['label'] for method in items]
    colors = ['#7b8794' if item['historical_reuse'] else '#267fa7' for item in items]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    specs = [('accuracy', 'First-task accuracy (%)', 1), ('process_wall_s', 'Wall time (s)', 1),
             ('cuda_peak_allocated_mib', 'CUDA allocated peak (GiB)', 1024),
             ('gpu_board_energy_wh', 'GPU board energy (Wh)', 1)]
    for ax, (key, title, scale) in zip(axes.flat, specs):
        entries = [method['accuracy_pct'] if key == 'accuracy' else method['resources'][key] for method in items]
        ax.bar(labels, [entry['mean'] / scale for entry in entries], color=colors,
               yerr=[(entry['std_sample'] or 0) / scale for entry in entries], capsize=3)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if key == 'accuracy':
            ax.set_ylim(0, 105)
    fig.suptitle('Base ViT, CIFAR100 first task, batch64; mean +/- sample SD of three same-seed runs\nGray: historical reuse; blue: new measurements. HiDe has TII1 + prompt1; compute budgets differ.', fontsize=12)
    fig.savefig(Path(output) / 'efficiency_comparison.png', dpi=220)
    fig.savefig(Path(output) / 'efficiency_comparison.pdf')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, default=ROOT / 'results/peft_pilot/base-20261008/execution.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/peft_pilot')
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args(argv)
    report = build_report(options.execution)
    if options.write:
        options.output.mkdir(parents=True, exist_ok=True)
        write_json(options.output / 'results.json', report)
        (options.output / 'results.csv').write_text(csv_text(report))
        (options.output / 'RESULTS.md').write_text(markdown(report))
        plot(report, options.output)
    print(json.dumps({'status': report['publication_gate'], 'methods': list(report['methods']),
                      'new_workers': 20, 'written': options.write}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
