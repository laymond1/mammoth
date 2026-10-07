"""Recompute HiDe size correctness and first-task costs from local evidence.

This report reads saved JSON/NPZ files only. It never initializes a model,
starts a data loader, contacts W&B, or executes training or CUDA operations.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.hide_acceptance import counts_from_predictions
from scripts.hide_protocol import (CANONICAL_PRETRAINED_SHA256, CHECKPOINT_SHA256,
                                   FULL_CL_PRECISION, REFERENCE_COMMIT, protocol)
from scripts.hide_size_protocol import load_variant
from scripts.report_hide_results import (RAW_KEYS, accuracy, cpu_thread_metadata,
                                         csv_rows as single_csv_rows, descriptive,
                                         exact_validation, file_hash, first_task_snapshot,
                                         format_number, load_efficiency, mean_sd, read_json)

INPUT_KEYS = ('small_reference', 'small_native', 'small_efficiency',
              'tiny_reference', 'tiny_native', 'tiny_efficiency', 'base_efficiency')
REPORT_FILES = ('results.json', 'results.csv', 'report.md',
                'efficiency_comparison.png', 'efficiency_comparison.pdf')


def variant_info(size):
    if size != 'base':
        return load_variant(size, verify_checkpoint=False)
    return {'vit_size': 'base', 'feature_dim': 768, 'num_heads': 12,
            'checkpoint_tag': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'checkpoint_sha256': CHECKPOINT_SHA256,
            'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256}


def expected_protocol(mode, size):
    return protocol(mode, variant=None if size == 'base' else size)


def git_blob_hash(root, commit, name):
    if not isinstance(commit, str) or len(commit) != 40 or any(letter not in '0123456789abcdef' for letter in commit):
        raise ValueError('Historical execution requires its immutable git commit')
    result = subprocess.run(['git', 'show', commit + ':' + name], cwd=root,
                            capture_output=True, check=True)
    return hashlib.sha256(result.stdout).hexdigest()


class Evidence:
    """Retain actual file provenance, including automatically followed inputs."""
    def __init__(self):
        self.files = {}
        self.sources = {}

    def pin(self, path, expected=None):
        path = Path(path).expanduser().resolve()
        observed = file_hash(path)
        if expected is not None and (not expected or observed != expected):
            raise ValueError(f'Saved evidence SHA256 differs: {path}')
        if path.as_posix() in self.files and self.files[path.as_posix()] != observed:
            raise ValueError('Evidence changed during report generation')
        self.files[path.as_posix()] = observed
        return path

    def execution_sources(self, record, historical=False):
        pins = record.get('execution_source_sha256', {})
        if not pins:
            raise ValueError('Every execution needs saved source SHA256 provenance')
        location = Path(record['output']).resolve()
        root = next((candidate for candidate in (location, *location.parents)
                     if (candidate / 'main.py').is_file()), None)
        if root is None:
            raise ValueError('Saved execution source worktree is unavailable')
        for name, pin in pins.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Execution source pins must be relative worktree paths')
            if historical:
                if git_blob_hash(root, record.get('git_commit'), name) != pin:
                    raise ValueError('Historical source SHA256 differs from its immutable git blob')
            else:
                self.pin(root / relative, pin)
        key = str(root) + ('@' + record['git_commit'] if historical else '@current_files')
        entry = self.sources.setdefault(key, {'worktree': str(root),
            'verification': 'immutable_git_blobs' if historical else 'current_files',
            'git_commit': record.get('git_commit'), 'files': {}})
        previous = entry['files']
        if any(name in previous and previous[name] != pin for name, pin in pins.items()):
            raise ValueError('Executions disagree on source hashes within one worktree')
        previous.update(pins)


def record_pins(record, size, evidence):
    info = variant_info(size)
    if (record.get('status') != 'completed' or record.get('model') != 'hide-prompt'
            or record.get('checkpoint_sha256') != info['checkpoint_sha256']
            or record.get('canonical_pretrained_sha256') != info['canonical_pretrained_sha256']
            or record.get('wandb_logging_status') != 'uploaded'
            or record.get('wandb_artifact', {}).get('state') != 'COMMITTED'
            or record.get('reference_commit') != REFERENCE_COMMIT):
        raise ValueError('Record is incomplete or differs from its size-specific weight/source pins')
    if size != 'base' and (record.get('vit_size', record.get('protocol', {}).get('vit_size')) != size
            or any(record.get(key) != value for key, value in
            (('variant', size), ('feature_dim', info['feature_dim']),
             ('num_heads', info['num_heads']), ('checkpoint_tag', info['checkpoint_tag'])))):
        raise ValueError('Size record geometry or checkpoint tag differs')
    for key in ('initial_learned_sha256', 'initial_canonical_pretrained_sha256'):
        values = record.get(key)
        if not isinstance(values, dict) or set(values) != {'tii', 'prompt'}:
            raise ValueError('Both initialized branch fingerprints are required')
        if any(not isinstance(value, str) or len(value) != 64 or any(letter not in '0123456789abcdef' for letter in value)
               for value in values.values()):
            raise ValueError('Branch fingerprints must be SHA256 strings')
    if set(record['initial_canonical_pretrained_sha256'].values()) != {info['canonical_pretrained_sha256']}:
        raise ValueError('Initialized frozen branch tensors differ from the size pin')
    if not isinstance(record.get('initial_state_sha256'), str) or len(record['initial_state_sha256']) != 64:
        raise ValueError('Learned initialization needs its immutable saved file SHA256')
    evidence.pin(record['checkpoint_file'], record['checkpoint_sha256'])
    evidence.pin(record['initial_state'], record['initial_state_sha256'])
    evidence.execution_sources(record, historical=size == 'base')
    if record.get('reference_source_sha256'):
        for name, pin in record['reference_source_sha256'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Author source pins must be relative paths')
            evidence.pin(Path(record['reference_source']) / relative, pin)


def prediction_audit(record, tasks, evidence):
    """Recount top-1, including legacy native [N,5] and author [N,1] arrays."""
    import numpy as np
    rows = record.get('task_results', [])
    if len(rows) != tasks or {row.get('task') for row in rows} != set(range(1, tasks + 1)):
        raise ValueError('Prediction audit requires the complete approved task sequence')
    audited = []
    for row in sorted(rows, key=lambda item: item['task']):
        task = row['task']
        filename = f'task{task}_predictions.npz'
        pin = record.get('wandb_artifact_files', {}).get(filename, {})
        path = Path(record['output']) / pin.get('local_file', filename)
        expected = pin.get('sha256', row.get('predictions_sha256'))
        if not expected:
            raise ValueError('Prediction arrays need an immutable saved SHA256 pin')
        path = evidence.pin(path, expected)
        with np.load(path, allow_pickle=False) as archive:
            predictions, labels, ids = (archive[key] for key in ('predictions', 'labels', 'global_ids'))
        seen = task * 10
        count = task * 1000
        if (predictions.ndim != 2 or not 1 <= predictions.shape[1] <= seen
                or predictions.shape[0] != count or labels.shape != (count,) or ids.shape != (count,)
                or any(not np.issubdtype(value.dtype, np.integer) for value in (predictions, labels, ids))
                or bool(((predictions < 0) | (predictions >= seen)).any())
                or bool(((labels < 0) | (labels >= seen)).any())
                or bool(((ids < 0) | (ids >= 10000)).any()) or not bool((np.diff(ids) > 0).all())):
            raise ValueError('Prediction arrays expose missing, duplicate, future or invalid samples')
        expected_classes = {str(label): 100 for label in range(seen)}
        class_counts = {str(int(label)): int(number) for label, number in zip(*np.unique(labels, return_counts=True))}
        snapshot = record.get('data_snapshots', {}).get(f'task{task}_test', {})
        ids_sha = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
        if (class_counts != expected_classes or snapshot.get('n_samples') != count
                or snapshot.get('class_counts') != expected_classes
                or snapshot.get('global_ids_ordered_sha256') != ids_sha):
            raise ValueError('Prediction sample identities do not match the recorded test snapshot')
        counts = counts_from_predictions(predictions[:, 0].tolist(), labels.tolist(),
                                        past_classes=(task - 1) * 10, seen_classes=seen)
        if any(type(row.get(key)) is not int or row[key] != value for key, value in counts.items()):
            raise ValueError('Saved accuracy counts disagree with the actual NPZ predictions')
        audited.append({'task': task, 'file': str(path), 'sha256': expected, **counts,
                        'seen_accuracy_pct': 100 * counts['seen_correct'] / counts['seen_samples'],
                        'global_ids_ordered_sha256': ids_sha,
                        'prediction_columns': predictions.shape[1], 'accuracy_column': 0})
    return audited


def full_validation(reference, native, size, reference_path, evidence):
    expected = expected_protocol('validation', size)
    for record in (reference, native):
        if record.get('mode') != 'validation' or record.get('protocol') != expected:
            raise ValueError('Two-task validation must use its own size-specific author/native protocol')
        record_pins(record, size, evidence)
        counts = {'tasks_completed': 2, 'outer_observe_steps': 11060,
                  'outer_training_stream_examples': 700000, 'evaluation_examples': 3000}
        if any(type(record.get('metrics', {}).get(key)) is not int or record['metrics'][key] != value
               for key, value in counts.items()):
            raise ValueError('Validation did not complete the two-task training/evaluation schedule')
        coordinates = {(task, phase, epoch) for task in (1, 2)
                       for phase, count in (('tii_train', 20), ('prompt_train', 50))
                       for epoch in range(1, count + 1)}
        epochs = record.get('epoch_results', [])
        if len(epochs) != 140 or {(row.get('task'), row.get('phase'), row.get('epoch')) for row in epochs} != coordinates:
            raise ValueError('Validation needs every approved TII/prompt epoch exactly once')
        batches = record.get('first_batch_audits', [])
        if len(batches) != 4 or {(row.get('task'), row.get('phase')) for row in batches} != {(task, phase) for task in (1, 2) for phase in ('tii_train', 'prompt_train')}:
            raise ValueError('Validation first-batch evidence must cover exactly two tasks and both phases')
    for row in native['epoch_results']:
        seed = 1993 + 100000 * (row['task'] - 1) + (1000 if row['phase'] == 'prompt_train' else 0) + row['epoch'] - 1
        if row.get('steps') != 79 or row.get('samples') != 5000 or row.get('seed') != seed:
            raise ValueError('Native validation epoch counts or phase RNG differs')
    for key in ('checkpoint_sha256', 'canonical_pretrained_sha256', 'initial_state_sha256',
                'initial_learned_sha256', 'initial_canonical_pretrained_sha256'):
        if reference.get(key) != native.get(key):
            raise ValueError(f'Same-size author/native {key} differs')
    if not native.get('reference_summary_sha256'):
        raise ValueError('Native validation needs the immutable reference summary SHA256')
    evidence.pin(reference_path, native['reference_summary_sha256'])
    if Path(native['reference_summary']).resolve() != Path(reference_path).resolve():
        raise ValueError('Native validation refers to a different independent reference')
    prediction_checks = {'reference': prediction_audit(reference, 2, evidence),
                         'native': prediction_audit(native, 2, evidence)}
    for task in (1, 2):
        for split in ('train', 'test'):
            left = reference.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            right = native.get('data_snapshots', {}).get(f'task{task}_{split}', {})
            if any(left.get(key) is None or left.get(key) != right.get(key) for key in RAW_KEYS):
                raise ValueError('Same-size author/native raw samples or transforms differ')
    result = exact_validation(reference, native)
    if result['report_gate'] != 'pass':
        raise ValueError('Same-size correctness or first-batch gate blocks efficiency reporting')
    result['prediction_recount'] = prediction_checks
    result['protocol'] = expected
    result['weight_pins'] = {key: native[key] for key in ('checkpoint_sha256', 'canonical_pretrained_sha256',
                                                        'initial_state_sha256', 'initial_learned_sha256')}
    return result


def common_pilot_inputs(records):
    """Require observed common inputs; weights intentionally differ by size."""
    first = records[0]
    for record in records:
        settings = record.get('protocol', {})
        common = {'dataset': 'seq-cifar100-224', 'dataset_config': 'ease', 'tasks': 1,
                  'batch_size': 64, 'seed': 1993, 'num_workers': 4,
                  'tii_epochs': 1, 'prompt_epochs': 1, 'n_epochs': 2, 'ca_epochs': 30}
        if any(type(settings.get(key)) is not type(value) or settings.get(key) != value for key, value in common.items()) or record.get('gpu') != '1':
            raise ValueError('All pilots must keep the same data, phase budgets, batch, seed, workers and GPU')
        for key, value in FULL_CL_PRECISION.items():
            if key == 'torch_cpu_threads':
                threads = cpu_thread_metadata(record)
                valid = threads['status'] == 'recorded' and threads['value'] == value
            else:
                actual = record.get('precision', {}).get(key)
                valid = type(actual) is type(value) and actual == value
            if not valid:
                raise ValueError('Pilots need matching recorded FP32 and CPU thread policies')
        for split in ('train', 'test'):
            left, right = first_task_snapshot(first, split), first_task_snapshot(record, split)
            if any(left.get(key) is None or left.get(key) != right.get(key) for key in RAW_KEYS):
                raise ValueError('Common pilot raw samples, identities or actual transforms differ')
        batches = {(row.get('task'), row.get('phase')): row for row in record.get('first_batch_audits', [])}
        target = {(row.get('task'), row.get('phase')): row for row in first.get('first_batch_audits', [])}
        if len(batches) != 2 or set(batches) != {(1, 'tii_train'), (1, 'prompt_train')}:
            raise ValueError('Every pilot needs both actual first-batch audits')
        for coordinate, row in batches.items():
            if any(row.get(key) is None or row.get(key) != target.get(coordinate, {}).get(key)
                   for key in ('samples', 'seed', 'input_sha256', 'labels_sha256')):
                raise ValueError('Pilot first-batch inputs, labels, phase seeds or sample counts differ')
    return {'status': 'pass', 'observations': len(records), 'raw_snapshot_fields': list(RAW_KEYS),
            'precision': FULL_CL_PRECISION, 'first_batch_inputs_labels_seeds_match': True,
            'weights_equal_across_sizes': False,
            'scope': 'Actual saved first-task samples/transforms, FP32/CPU2, batch64, seed1993, workers4 and GPU1; size-specific weights and execution times differ.'}


def build_report(inputs):
    """Public deterministic payload builder used again by the report publisher."""
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError('Size report requires exactly the seven named raw input files')
    inputs = {key: Path(path).expanduser().resolve() for key, path in inputs.items()}
    evidence = Evidence()
    sources = {key: {'file': str(evidence.pin(path)), 'sha256': file_hash(path)} for key, path in inputs.items()}
    validation, native_records = {}, {}
    for size in ('small', 'tiny'):
        reference = read_json(inputs[size + '_reference'])
        reference = reference.get('official_reference', reference)
        native = read_json(inputs[size + '_native'])
        validation[size] = full_validation(reference, native, size, inputs[size + '_reference'], evidence)
        native_records[size] = native
    methods, all_pilots = [], []
    for size in ('base', 'small', 'tiny'):
        path = inputs[size + '_efficiency']
        raw = read_json(path)
        if 'measurements' not in raw:
            raise ValueError('Every size, including historical Base, requires raw execution records')
        summary, measurements, raw = load_efficiency(path)
        coordinator = raw.get('protocol', {})
        common = {'tasks': 1, 'dataset_config': 'ease', 'batch_size': 64,
                  'tii_epochs': 1, 'prompt_epochs': 1, 'n_epochs': 2, 'ca_epochs': 30,
                  'outer_epochs_per_task': 2, 'measurement_repeats': 3, 'flops_runs': 1,
                  'ca_runs_task1': False}
        if any(type(coordinator.get(key)) is not type(value) or coordinator.get(key) != value
               for key, value in common.items()):
            raise ValueError('Pilot coordinator protocol differs from the approved size recipe')
        if size != 'base' and raw.get('variant') != size:
            raise ValueError('Pilot coordinator variant differs from its requested size')
        if not raw.get('acceptance_summary_sha256'):
            raise ValueError('Pilot coordinator needs its immutable acceptance summary SHA256')
        native = native_records.get(size)
        if native is None:
            accepted_path = evidence.pin(raw['acceptance_summary'], raw.get('acceptance_summary_sha256'))
            native = read_json(accepted_path)
            reference_path = evidence.pin(native['reference_summary'], native.get('reference_summary_sha256'))
            reference = read_json(reference_path)
            reference = reference.get('official_reference', reference)
            base_validation = full_validation(reference, native, size, reference_path, evidence)
            summary['historical_acceptance_report_gate'] = base_validation['report_gate']
        else:
            accepted_path = evidence.pin(raw['acceptance_summary'], raw.get('acceptance_summary_sha256'))
            if accepted_path != inputs[size + '_native']:
                raise ValueError('Pilot acceptance is not the supplied size-specific native validation')
        for key in ('checkpoint_sha256', 'initial_state_sha256'):
            if raw.get(key) != native.get(key):
                raise ValueError('Pilot coordinator weights or initialization differ from accepted validation')
        if size != 'base' and raw.get('canonical_pretrained_sha256') != native.get('canonical_pretrained_sha256'):
            raise ValueError('Pilot coordinator canonical weights differ from accepted validation')
        if size != 'base' and raw.get('execution_source_sha256'):
            evidence.execution_sources(raw)
        recounted = []
        for record in [*measurements, raw['flops']]:
            record_pins(record, size, evidence)
            if record.get('mode') != 'efficiency' or record.get('protocol') != expected_protocol('efficiency', size):
                raise ValueError('Pilot worker differs from the size-specific short recipe')
            for key in ('initial_state_sha256', 'initial_learned_sha256',
                        'initial_canonical_pretrained_sha256', 'execution_source_sha256'):
                if record.get(key) != native.get(key):
                    raise ValueError('Pilot initialization or execution sources differ from its accepted validation')
            if (Path(record['acceptance_summary']).resolve() != accepted_path
                    or record.get('acceptance_summary_sha256') != file_hash(accepted_path)):
                raise ValueError('Pilot worker acceptance file changed')
            counts = prediction_audit(record, 1, evidence)[0]
            if record['pass'] == 'measurement':
                recounted.append(counts['seen_accuracy_pct'])
            if accuracy(record) != counts['seen_accuracy_pct']:
                raise ValueError('Pilot accuracy field differs from recounted predictions')
            all_pilots.append(record)
        info = variant_info(size)
        summary.update(variant=size, label='HiDe-' + size.title(), historical_reuse=size == 'base',
                       feature_dim=info['feature_dim'], num_heads=info['num_heads'],
                       checkpoint_tag=info['checkpoint_tag'], checkpoint_sha256=info['checkpoint_sha256'],
                       canonical_pretrained_sha256=info['canonical_pretrained_sha256'],
                       initial_state_sha256=native['initial_state_sha256'],
                       precision=measurements[0]['precision'],
                       accuracy_pct=descriptive(recounted), alignment_status='not_applicable')
        methods.append(summary)
    compatibility = common_pilot_inputs(all_pilots)
    report_sources = {str(path): file_hash(ROOT / path) for path in
                      ('scripts/report_hide_sizes.py', 'scripts/report_hide_results.py',
                       'scripts/hide_acceptance.py', 'scripts/hide_size_protocol.py',
                       'scripts/hide_protocol.py', 'scripts/run_hide_efficiency.py')}
    return {'schema_version': 1, 'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': 'completed', 'model': 'hide-prompt', 'stage': 'hide_sizes',
            'validation': validation, 'efficiency': {'status': 'completed', 'methods': methods,
                'compatibility': compatibility, 'seed_variation_estimated': False,
                'std_definition': 'Sample SD (ddof=1) across three same-seed executions; excludes the fourth partial-FLOPs diagnostic.'},
            'full10_executed': False, 'paper_reproduction': False, 'sources': sources,
            'provenance': {'evidence_files': evidence.files, 'execution_sources': evidence.sources,
                           'report_source_sha256': report_sources},
            'measurement_scope': {
                'accuracy': 'Same-size independent author/native two-task checks and separate first-task TII1+prompt1 pilot accuracy. No full-sequence accuracy or forgetting is inferred.',
                'budgets': 'Same input and phase/epoch budgets across sizes; FLOPs and parameter counts differ. TII mapping ratio2 is retained. Task1 classifier alignment is not applicable.',
                'pretraining': 'AugReg IN21K to IN1K fine-tuned family. Selected checkpoint filenames identify Base aug_medium/weight_decay0.1, Small aug_light1/weight_decay0.03 and Tiny aug_none/weight_decay0.03; size comparisons include those recipe effects.',
                'time': 'Recorder process and inclusive synchronized phase wall time; final W&B upload/finish lies outside Recorder sampling.',
                'energy': 'Sampled physical GPU-board power integration; excludes CPU/RAM/whole-machine energy and may include other board activity.',
                'memory': 'CUDA allocator/device peaks and process-tree RSS have different scopes; shared RSS pages can be counted multiple times.',
                'flops': 'One separate partial supported-operator pass per size; omitted/unsupported operators and CPU clustering prevent a complete-method estimate.',
                'history': 'Base reuses historical observations from its preserved source worktree. Different execution dates and source snapshots may introduce timing drift.'}}


def csv_rows(report):
    rows = []
    for size in ('small', 'tiny'):
        partial = {'validation': report['validation'][size], 'efficiency': {'methods': []}}
        rows.extend({'variant': size, **row} for row in single_csv_rows(partial))
    for method in report['efficiency']['methods']:
        partial = {'validation': {'accuracy_metrics': []}, 'efficiency': {'methods': [method]}}
        rows.extend({'variant': method['variant'], **row} for row in single_csv_rows(partial))
    return rows


def csv_fields(report):
    return list(dict.fromkeys(key for row in csv_rows(report) for key in row))


def markdown(report):
    lines = ['# HiDe-Prompt 크기 확장 결과', '',
             'Small→Tiny 각각 동일 크기의 독립 저자 코드와 첫 2태스크를 검증했습니다. 논문 수치 재현이나 전체 10태스크 결과로 해석하지 않습니다.', '',
             '| 크기 | 지표 | 저자 (%) | native (%) | 저자−native (%p) | 판정 |',
             '|---|---|---:|---:|---:|---|']
    for size, validation in report['validation'].items():
        for row in validation['accuracy_metrics']:
            lines.append(f'| {size.title()} | {row["metric"]} | {format_number(row["reference_accuracy_pct"])} | {format_number(row["native_accuracy_pct"])} | {format_number(row["reference_minus_native_pp"])} | {row["status"]} |')
    lines.extend(['', '허용 하락: 1.5%p 이내 통과, 1.5~3.0%p 추가 점검, 3.0%p 초과 실패. 실제 NPZ의 top1 정답 수/샘플 수를 재계산하고 입력·초기화·구조·첫 배치 조건도 통과해야 합니다. 서로 다른 크기 간 정확도 차이를 구현 실패로 판정하지 않습니다.', '',
                  '첫 태스크 pilot은 공통 `ease` 전처리, batch64, seed1993, workers4, GPU1, FP32/CPU2, TII1+prompt1입니다. TII MLP 비율은 모두 2이며 prefix 길이5·첫 5개 layer를 유지합니다. 첫 태스크 CA는 해당 없음입니다. Base는 보존한 이전 pilot을 재사용하므로 실행 시점·소스 차이를 포함합니다.', '',
                  '| 크기 | pilot 정확도 (%) | 벽시간 (s) | GPU 보드 (Wh) | CUDA peak (GiB) | RSS peak (GiB) | 부분 FLOPs (TF) |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for method in report['efficiency']['methods']:
        resources = method['resources']
        flops = method['flops']['supported_ops_total']
        lines.append(f'| {method["variant"].title()}{" (기존)" if method["historical_reuse"] else ""} | {mean_sd(method["accuracy_pct"])} | {mean_sd(resources["process_wall_s"])} | {mean_sd(resources["gpu_board_energy_wh"])} | {mean_sd(resources["cuda_peak_allocated_mib"], 1/1024)} | {mean_sd(resources["peak_process_tree_rss_mib"], 1/1024)} | {format_number(flops / 1e12)} |')
    lines.extend(['', '평균 ± 표본 표준편차는 동일 seed의 측정 3회 실행 비용 변동입니다. 별도 부분 FLOPs 1회의 비용을 제외했으며, seed 간 불확실성이나 전체 CL 성능·망각을 추정하지 않습니다.', '',
                  '![크기별 효율성](efficiency_comparison.png)', '', '[PDF 그림](efficiency_comparison.pdf)', '',
                  '| 크기 | 측정 W&B | 별도 FLOPs W&B |', '|---|---|---|'])
    for method in report['efficiency']['methods']:
        links = ' · '.join(f'[run{index}]({url})' for index, url in enumerate(method['wandb_urls'], 1) if url)
        flops_url = method.get('flops_wandb_url')
        lines.append(f'| {method["variant"].title()} | {links or "—"} | {"[run](" + flops_url + ")" if flops_url else "—"} |')
    lines.extend(['', 'GPU 보드 에너지는 전체 머신 에너지와 다릅니다. 단계 타이머는 중첩되어 총 비용으로 합산할 수 없습니다. 부분 FLOPs는 미지원 연산과 CPU clustering 등을 누락할 수 있습니다. IN21K→IN1K 계열은 같지만 선택한 파일명 기준 Base는 aug_medium·weight decay0.1, Small은 aug_light1·0.03, Tiny는 aug_none·0.03입니다. 크기 비교에는 이 사전학습 recipe 차이도 포함됩니다.', '',
                  '[JSON 결과](results.json) · [CSV 결과](results.csv)', ''])
    return '\n'.join(lines)


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = report['efficiency']['methods']
    labels = [method['variant'].title() + ('\nHistorical' if method['historical_reuse'] else '') for method in methods]
    figure, axes = plt.subplots(2, 3, figsize=(10.3, 6.2), layout='constrained')
    panels = [('process_wall_s', 'Process wall time (s)', 1.),
              ('gpu_board_energy_wh', 'GPU-board energy (Wh)', 1.),
              ('cuda_peak_allocated_mib', 'CUDA allocator peak (GiB)', 1 / 1024),
              ('peak_process_tree_rss_mib', 'Process-tree RSS peak (GiB)', 1 / 1024),
              ('accuracy', 'Short-pilot accuracy (%)', 1.),
              ('flops', 'Partial supported operators (TFLOPs)', 1e-12)]
    for axis, (key, title, factor) in zip(axes.flat, panels):
        if key == 'flops':
            values = [method['flops']['supported_ops_total'] * factor for method in methods]
            errors = None
        else:
            values = [method['accuracy_pct'] if key == 'accuracy' else method['resources'][key] for method in methods]
            errors = [(value['std_sample'] or 0) * factor for value in values]
            values = [value['mean'] * factor for value in values]
        axis.bar(range(3), values, yerr=errors, capsize=3, color=['#8498ac', '#176385', '#47a09d'])
        axis.set_xticks(range(3), labels)
        axis.set_title(title, fontsize=10, loc='left')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        if key == 'accuracy':
            axis.set_ylim(0, 105)
    figure.suptitle('HiDe-Prompt size comparison: first task, TII1 + prompt1\nMean ± sample SD of three same-seed executions', fontsize=12)
    figure.supxlabel('Base is historical. Task1 alignment N/A. GPU-board energy excludes CPU/RAM. FLOPs use a separate partial-operator pass.', fontsize=8)
    figure.savefig(output / 'efficiency_comparison.png', dpi=240)
    figure.savefig(output / 'efficiency_comparison.pdf')
    plt.close(figure)


def write_report(report, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', report)
    rows = csv_rows(report)
    fields = csv_fields(report)
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)
    (output / 'report.md').write_text(markdown(report))
    plot_report(report, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in INPUT_KEYS:
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args(argv)
    report = build_report({key: getattr(options, key) for key in INPUT_KEYS})
    write_report(report, options.output)
    print(json.dumps({'status': report['status'], 'output': str(options.output.resolve()),
                      'variants': [method['variant'] for method in report['efficiency']['methods']], 'full10_executed': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
