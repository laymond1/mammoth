"""Aggregate four saved two-task LoRA validation pairs; default is read-only.

--write produces RESULTS.md, results.json and results.csv from real completed
records. No credentials, GPU, model, network client or training are imported.
Resource observations describe these validation runs, not efficiency pilots.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.low_rank_protocol import (
    CANONICAL_PRETRAINED_SHA256, CHECKPOINT_SHA256, METHODS, REFERENCE_COMMIT,
    REQUIRED_STRUCTURAL_CHECKS, completion_checks, evaluate_acceptance, file_hash,
    protocol,
)

LABELS = {'ewc-lora': 'EWC-LoRA', 'sd-lora': 'SD-LoRA', 'inf-lora': 'InfLoRA', 'cl-lora': 'CL-LoRA'}
COORDINATES = ((1, 'seen'), (2, 'seen'), (2, 'old'), (2, 'new'))
RESOURCE_KEYS = ('process_wall_s', 'process_tree_cpu_s', 'gpu_board_energy_wh',
                 'gpu_board_energy_j', 'gpu_energy_coverage_fraction',
                 'cuda_peak_allocated_mib', 'cuda_peak_reserved_mib',
                 'peak_process_tree_rss_mib', 'peak_gpu_device_memory_mib',
                 'outer_training_examples_per_s', 'evaluation_examples_per_s')
REPORT_FILES = ('RESULTS.md', 'results.json', 'results.csv')
COUNT_KEYS = tuple(f'{partition}_{name}' for partition in ('seen', 'old', 'new') for name in ('correct', 'samples'))


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('A saved low-rank record must be a JSON object')
    return value


def run_path(url):
    match = re.fullmatch(r'https://wandb\.ai/(laymond1/cl-efficiency/runs/([A-Za-z0-9_-]+))/?', str(url))
    if not match:
        raise ValueError('Every validation run must have a laymond1/cl-efficiency W&B URL')
    return 'laymond1/cl-efficiency/' + match[2]


def record_body(value):
    return value.get('official_reference', value)


def first_batches(reference, native, method):
    expected = {(1, 1), (2, 1)} | ({(2, 2)} if method == 'cl-lora' else set())
    fields = ('task', 'epoch', 'seed', 'samples', 'optimizer_update_index', 'input_sha256', 'labels_sha256')
    indexed = []
    for record in (reference, native):
        rows = record.get('first_batch_audits', [])
        coordinates = {(row.get('task'), row.get('optimizer_update_index')): row for row in rows}
        if len(rows) != len(expected) or set(coordinates) != expected:
            raise ValueError('First-batch audits must include each source optimizer update exactly once')
        indexed.append(coordinates)
    rows = []
    for coordinate in sorted(expected):
        left, right = (entries[coordinate] for entries in indexed)
        match = all(left.get(key) is not None and left.get(key) == right.get(key) for key in fields)
        if left.get('samples') != 64 or left.get('epoch') != 1:
            raise ValueError('First-batch validation must use a full batch64 in epoch1')
        rows.append({'task': coordinate[0], 'optimizer_update_index': coordinate[1], 'inputs_match': match,
                     'gradient_sha256_matches': left.get('gradient_sha256') == right.get('gradient_sha256')})
    return {'status': 'pass' if all(row['inputs_match'] for row in rows) else 'fail', 'rows': rows,
            'gradient_hash_scope': 'Recorded diagnostic only; accuracy acceptance uses exact prediction counts.'}


def resource_diagnostics(record):
    metrics = record.get('metrics', {})
    values = {key: metrics.get(key) for key in RESOURCE_KEYS}
    for value in values.values():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError('Resource observations must be finite nonnegative numbers or missing')
    return {'observation_count': 1, 'metrics': values, 'phases': record.get('phases', {}),
            'scope': 'One full validation run. Includes statistics/prototypes/evaluation/instrumentation; no efficiency pilot, repeats, FLOPs or energy ranking.'}


def validate_pair(method, reference, native):
    for side, record in (('reference', reference), ('native', native)):
        if (record.get('method') != method or record.get('mode') != 'validation'
                or record.get('status') != 'completed' or record.get('wandb_logging_status') != 'uploaded'
                or record.get('protocol') != protocol(method)):
            raise ValueError(f'{method} {side} must be a completed uploaded full two-task validation')
        run_path(record.get('wandb_url'))
        completion_checks(record, method)
        if any(record.get('structural_checks', {}).get(name) is not True for name in REQUIRED_STRUCTURAL_CHECKS):
            raise ValueError(f'{method} {side} is missing a verified structural gate')
        if record.get('checkpoint_sha256') != CHECKPOINT_SHA256 or record.get('reference_commit') != REFERENCE_COMMIT:
            raise ValueError('Report must use the pinned author revision and Base checkpoint')
        if record.get('canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256:
            raise ValueError('Report canonical pretrained tensors differ from the pin')
    for key in ('initial_state_sha256', 'initial_learned_sha256'):
        if (not re.fullmatch(r'[0-9a-f]{64}', str(reference.get(key, '')))
                or reference.get(key) != native.get(key)):
            raise ValueError('Native and reference learned initialization is missing or differs')
    acceptance = evaluate_acceptance(native['task_results'], reference['task_results'],
                                     structural_checks=native['structural_checks'])
    if native.get('acceptance_status') != acceptance['status']:
        raise ValueError('Saved acceptance differs from exact count and structural-gate recomputation')
    batches = first_batches(reference, native, method)
    rows = []
    for task, partition in COORDINATES:
        name = f'task{task}_{partition}'
        rows.append({'metric': name, **acceptance['metrics'][name]})
    return {'method': method, 'label': LABELS[method], 'protocol': protocol(method),
            'acceptance_status': acceptance['status'], 'acceptance': acceptance,
            'publication_gate': 'pass' if acceptance['status'] == batches['status'] == 'pass' else 'blocked',
            'accuracy_metrics': rows, 'first_batch_comparison': batches,
            'runs': {side: {'wandb_url': record['wandb_url'], 'run_path': run_path(record['wandb_url']),
                            'counts': record['counts'], 'task_results': record['task_results'],
                            'state_audits': record.get('state_audits', {}),
                            'resource_diagnostics': resource_diagnostics(record)}
                     for side, record in (('reference', reference), ('native', native))}}


def artifact_inputs(record, record_path):
    """Keep large checkpoints local; upload their hashes rather than weights."""
    parent = record_path.parent
    files = {}
    predictions = record.get('prediction_files', [])
    prediction_pins = {row.get('task'): row.get('sha256') for row in predictions}
    if len(predictions) != 2 or set(prediction_pins) != {1, 2}:
        raise ValueError('Each completed run needs two recorded prediction archive pins')
    for name in ('spec.json', 'task1_predictions.npz', 'task2_predictions.npz'):
        path = parent / name
        if not path.is_file():
            raise ValueError(f'Missing validation artifact input: {name}')
        files[name] = {'file': str(path.resolve()), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
        if name.startswith('task') and files[name]['sha256'] != prediction_pins[int(name[4])]:
            raise ValueError('Prediction archive differs from the saved validation pin')
    checkpoints = {}
    audits = {row.get('task'): row for row in record.get('state_audits', {}).get('tasks', [])}
    for task in (1, 2):
        filename = f'task{task}_state.pt'
        path = parent / filename
        if not path.is_file():
            raise ValueError('Each validation run needs its local task checkpoint')
        pin = file_hash(path)
        recorded = audits.get(task, {}).get('checkpoint_sha256')
        if recorded != pin:
            raise ValueError('Local checkpoint differs from the saved state-audit pin')
        checkpoints[str(task)] = {'file': str(path.resolve()), 'relative_file': filename,
                                  'sha256': pin, 'bytes': path.stat().st_size, 'uploaded': False}
    return {'files': files, 'local_checkpoints': checkpoints,
            'remote_record_file': 'summary.json' if record_path.name == 'summary.json' else 'result.json'}


def load_inputs(path):
    manifest_path = Path(path).resolve()
    document = read_json(manifest_path)
    entries = document.get('methods', document)
    if set(entries) != set(METHODS):
        raise ValueError('Input manifest must contain all four approved methods exactly once')
    pairs, sources = {}, {}
    for method in METHODS:
        if set(entries[method]) != {'reference', 'native'}:
            raise ValueError('Every method needs exactly one reference and native record')
        pairs[method], sources[method] = {}, {}
        for side in ('reference', 'native'):
            source = Path(entries[method][side]).expanduser()
            source = (source if source.is_absolute() else manifest_path.parent / source).resolve()
            body = record_body(read_json(source))
            pairs[method][side] = body
            sources[method][side] = {'file': str(source), 'sha256': file_hash(source),
                                     'artifacts': artifact_inputs(body, source)}
    return pairs, sources


def build_report(pairs, sources=None):
    if set(pairs) != set(METHODS):
        raise ValueError('Report requires four completed validation pairs')
    methods = {method: validate_pair(method, **pairs[method]) for method in METHODS}
    run_paths = [run['run_path'] for method in methods.values() for run in method['runs'].values()]
    if len(set(run_paths)) != 8:
        raise ValueError('The eight method/reference/native W&B run IDs must be distinct')
    return {'schema_version': 1, 'status': 'completed', 'report_kind': 'four_method_two_task_validation',
            'publication_gate': 'pass' if all(entry['publication_gate'] == 'pass' for entry in methods.values()) else 'blocked',
            'reference_commit': REFERENCE_COMMIT, 'checkpoint_sha256': CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
            'source_variant': 'Immutable common author repository with matched shared input/initialization/phase RNG.',
            'paper_reproduction': False, 'efficiency_comparison': False, 'new_training': False,
            'statistical_tolerance_estimated': False, 'methods': methods, 'sources': sources or {}}


def csv_text(report):
    stream = io.StringIO(newline='')
    fields = ('method', 'metric', 'reference_correct', 'reference_samples', 'native_correct', 'native_samples',
              'reference_accuracy_pct', 'native_accuracy_pct', 'drop_pp', 'status')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for method, entry in report['methods'].items():
        for row in entry['accuracy_metrics']:
            writer.writerow({'method': method, **{key: row[key] for key in fields if key != 'method'}})
    return stream.getvalue()


def markdown(report):
    lines = ['# Continual LoRA 검증 결과', '',
             'CIFAR100 첫 2개 태스크의 저자 코드·Mammoth 구현 검증 8개 실행입니다. '
             'Base ViT, rank10, batch64, seed1993, 원본 클래스 순서를 사용했습니다. '
             'EWC/SD/Inf는 태스크당 20epoch, CL은 30epoch입니다.', '',
             f'공통 저자 저장소 `{REFERENCE_COMMIT[:12]}`의 입력·초기값·RNG를 맞춘 변형입니다. '
             '논문 수치 재현이나 효율성 파일럿 결과가 아닙니다. '
             '판정은 반올림 전 정답/표본 수와 구조 검사로 계산합니다. '
             '저자−구현 차이가 1.5%p 이하면 pass, 3%p 이하면 review입니다.', '',
             '| 방법 | 지표 | 저자 정답/표본 (정확도) | 구현 정답/표본 (정확도) | 저자−구현 %p | 판정 |',
             '|---|---|---:|---:|---:|---|']
    metric_labels = {'task1_seen': '태스크1 전체', 'task2_seen': '태스크2 전체',
                     'task2_old': '태스크2 이전', 'task2_new': '태스크2 신규'}
    for entry in report['methods'].values():
        for row in entry['accuracy_metrics']:
            lines.append(f'| {entry["label"]} | {metric_labels[row["metric"]]} | '
                         f'{row["reference_correct"]}/{row["reference_samples"]} ({row["reference_accuracy_pct"]:.2f}%) | '
                         f'{row["native_correct"]}/{row["native_samples"]} ({row["native_accuracy_pct"]:.2f}%) | '
                         f'{row["drop_pp"]:+.2f} | {row["status"]} |')
    lines += ['', '| 방법 | 구조 검사 | 첫 배치 입력 | 최종 판정 |', '|---|---|---|---|']
    for entry in report['methods'].values():
        gates = entry['acceptance']['structural_checks']
        lines.append(f'| {entry["label"]} | {sum(value is True for value in gates.values())}/{len(gates)} | '
                     f'{entry["first_batch_comparison"]["status"]} | {entry["acceptance_status"]} |')
    lines += ['', '실행별 자원 진단은 각 1회 관측이며 통계·프로토타입·평가·계측 비용을 포함합니다. '
              '반복 효율성 측정, FLOPs 진단, 동일 예산 비교를 수행하지 않아 방법 간 효율성 순위를 매기지 않습니다.', '',
              '| 방법/실행 | 전체 시간 s | CUDA peak GiB | GPU 보드 Wh | 에너지 관측 비율 |',
              '|---|---:|---:|---:|---:|']
    def display(value, scale=1., decimals=2):
        return '미기록' if value is None else f'{value / scale:.{decimals}f}'
    for entry in report['methods'].values():
        for side, run in entry['runs'].items():
            values = run['resource_diagnostics']['metrics']
            label = '저자' if side == 'reference' else '구현'
            lines.append(f'| {entry["label"]}/{label} | {display(values["process_wall_s"], decimals=1)} | '
                         f'{display(values["cuda_peak_allocated_mib"], 1024)} | '
                         f'{display(values["gpu_board_energy_wh"], decimals=3)} | '
                         f'{display(values["gpu_energy_coverage_fraction"])} |')
    lines += ['', '학습 체크포인트는 로컬에 보관하고 SHA-256을 결과에 기록합니다. '
              'W&B에는 예측·설정·검사 결과와 체크포인트 해시를 저장합니다.', '']
    for entry in report['methods'].values():
        lines.append(f'- {entry["label"]}: [저자 실행]({entry["runs"]["reference"]["wandb_url"]}), '
                     f'[구현 실행]({entry["runs"]["native"]["wandb_url"]})')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path, help='JSON method -> {reference, native} record paths')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/low_rank')
    parser.add_argument('--write', action='store_true', help='Write the three local report artifacts')
    options = parser.parse_args(argv)
    pairs, sources = load_inputs(options.inputs)
    report = build_report(pairs, sources)
    if options.write:
        options.output.mkdir(parents=True, exist_ok=True)
        report['generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(options.output / 'results.json', report)
        (options.output / 'results.csv').write_text(csv_text(report), encoding='utf-8')
        (options.output / 'RESULTS.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'publication_gate': report['publication_gate'],
                      'methods': list(report['methods']), 'writes': options.write,
                      'external_actions': False, 'efficiency_comparison': False}))
    return 0 if report['publication_gate'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
