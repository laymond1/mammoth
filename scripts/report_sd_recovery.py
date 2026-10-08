"""Audit the separate SD learning-rate recovery and render nine saved pilots.

This reader performs no training or networking. It preserves the original
eight-method report, failed SD observation and deferred jobs. Only the three
fresh SD measurements contribute to new cost statistics; native validation,
the saved author diagnostic and the separate profiler remain outside them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import datetime
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import report_peft_efficiency as full
from scripts import report_peft_partial as partial
from scripts import low_rank_protocol as shared
from scripts.cl_efficiency import write_json
from scripts.report_hide_results import accuracy, cpu_thread_metadata, first_task_snapshot, RAW_KEYS
from scripts.run_low_rank_efficiency import REQUIRED_CHECKS

REPORT_FILES = full.REPORT_FILES
DEFAULT_RUN_ROOT = ROOT / 'results/sd_recovery_20261008'
DEFAULT_PRIOR = ROOT / 'benchmarks/cl_efficiency/peft_pilot_partial/results.json'
DEFAULT_REFERENCE = ROOT / 'results/sd_cause_analysis_20261008/lower_lora_lr/result.json'
DEFAULT_OUTPUT = ROOT / 'benchmarks/cl_efficiency/sd_recovery'
REFERENCE_SHA256 = 'b0ed2393aaeba8c7168eb7bbb5dd1ccc0e0ee4a12b6196be0019e00d04c24685'
OLD_METHODS = ('ease', 'hide-prompt', 'l2p', 'dualprompt', 'ewc-lora', 'inf-lora', 'cl-lora', 'ranpac')


def csv_text(report):
    """Retain original metadata while making this stage's reuse explicit."""
    rows = list(csv.reader(io.StringIO(full.csv_text(report))))
    rows[0].extend(('reused_in_recovery_report', 'recovery_recipe', 'train_epoch_s_mean'))
    for row in rows[1:]:
        item = report['methods'][row[0]]
        row.extend((row[0] != 'sd-lora', item.get('recipe', 'original_accepted_recipe'),
                    item['phases']['train_epoch']['wall_s']['mean']))
    output = io.StringIO(newline='')
    csv.writer(output, lineterminator='\n').writerows(rows)
    return output.getvalue()


def plot(report, output):
    # The old summaries retain their original reuse flag. For this stage's
    # figure all eight prior methods are reused, and only SD is newly measured.
    styled = deepcopy(report)
    for name, method in styled['methods'].items():
        method['historical_reuse'] = name != 'sd-lora'
    full.plot(styled, output)


def _pin(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Required recovery evidence is missing')
    return {'file': str(path), 'sha256': shared.file_hash(path), 'bytes': path.stat().st_size}


def _check_pin(proof, path, label):
    expected = _pin(path)
    if (not isinstance(proof, dict) or Path(proof.get('file', '')).resolve() != Path(expected['file'])
            or proof.get('sha256') != expected['sha256']
            or 'bytes' in proof and (type(proof['bytes']) is not int or proof['bytes'] != expected['bytes'])):
        raise ValueError(f'{label} evidence fingerprint changed')
    return expected


def _exact_counts(record):
    expected_metrics = {'tasks_completed': 1, 'outer_observe_steps': 79,
                        'outer_training_stream_examples': 5000, 'evaluation_examples': 1000}
    expected_counts = {'observe_calls': 79, 'optimizer_updates': 79,
                       'task_begin_extra_samples': 0, 'task_end_extra_samples': 0}
    for values, expected in ((record.get('metrics', {}), expected_metrics),
                             (record.get('counts', {}), expected_counts)):
        if any(type(values.get(key)) is not int or values[key] != value for key, value in expected.items()):
            raise ValueError('SD recovery requires the complete 79-update, 5000/1000 first-task stream')
    epochs, tasks = record.get('epoch_results', []), record.get('task_results', [])
    if (len(epochs) != 1 or any(type(epochs[0].get(key)) is not int or epochs[0][key] != value
                              for key, value in {'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000}.items())
            or len(tasks) != 1):
        raise ValueError('SD recovery requires one complete epoch and one complete task result')
    task = tasks[0]
    expected = {'task': 1, 'seen_samples': 1000, 'old_correct': 0, 'old_samples': 0, 'new_samples': 1000}
    if (any(type(task.get(key)) is not int or task[key] != value for key, value in expected.items())
            or type(task.get('seen_correct')) is not int or not 0 <= task['seen_correct'] <= 1000
            or type(task.get('new_correct')) is not int or task['new_correct'] != task['seen_correct']):
        raise ValueError('SD recovery prediction counts are invalid or incomplete')
    observed = accuracy(record)
    if type(observed) not in (int, float) or not math.isfinite(observed) or observed != task['seen_correct'] / 10:
        raise ValueError('SD recovery reported accuracy disagrees with exact saved prediction counts')
    return Fraction(task['seen_correct'], 10)


def _precision_and_structure(record):
    precision = record.get('precision', {})
    if (set(precision) != set(shared.PRECISION)
            or any(type(precision.get(key)) is not type(value) or precision[key] != value
                   for key, value in shared.PRECISION.items())
            or cpu_thread_metadata(record)['value'] != 2):
        raise ValueError('SD recovery must retain the explicit common FP32 policy and two CPU threads')
    checks = record.get('structural_checks', {})
    if any(checks.get(name) is not True for name in REQUIRED_CHECKS) or any(value is not True for value in checks.values()):
        raise ValueError('All thirteen SD recovery structural checks must pass')
    if (record.get('checkpoint_sha256') != shared.CHECKPOINT_SHA256
            or record.get('canonical_pretrained_sha256') != shared.CANONICAL_PRETRAINED_SHA256
            or record.get('reference_commit') != shared.REFERENCE_COMMIT
            or record.get('state_audits', {}).get('frozen_backbone_sha256') != shared.CANONICAL_PRETRAINED_SHA256):
        raise ValueError('SD recovery checkpoint, author source or frozen backbone pin changed')


def _common_inputs(record, target):
    for split in ('train', 'test'):
        observed, expected = first_task_snapshot(record, split), first_task_snapshot(target, split)
        if any(observed.get(key) is None or observed[key] != expected.get(key) for key in RAW_KEYS):
            raise ValueError('SD recovery raw data, order or actual preprocessing differ from the common input')
    for key in ('initial_state_sha256', 'initial_learned_sha256', 'data_files'):
        if record.get(key) is None or record[key] != target.get(key):
            raise ValueError('SD recovery initialization or shared data provenance changed')
    left, right = record.get('first_batch_audits', []), target.get('first_batch_audits', [])
    if len(left) != 1 or len(right) != 1:
        raise ValueError('SD recovery requires the complete first actual optimizer-step audit')
    keys = ('task', 'epoch', 'seed', 'samples', 'optimizer_update_index',
            'input_sha256', 'labels_sha256', 'gradient_sha256')
    if any(left[0].get(key) is None or left[0][key] != right[0].get(key) for key in keys):
        raise ValueError('SD recovery first actual batch, initialization gradients or phase seed changed')


def _source_fingerprints(record, protected):
    observed = record.get('execution_source_sha256', {})
    required = ('scripts/run_sd_recovery.py', 'scripts/run_sd_recovery_suite.py',
                'benchmarks/cl_efficiency/sd_recovery/plan.json')
    if not observed or any(name not in observed for name in required):
        raise ValueError('SD recovery execution source fingerprints are missing')
    # Experiment wrappers and plans differ by design. Runtime model, data,
    # training, measurement and environment sources must remain identical.
    runtime_prefixes = ('models/', 'backbone/', 'datasets/', 'utils/')
    runtime_files = {'main.py', 'uv.lock', 'scripts/cl_efficiency.py', 'scripts/cl_runtime.py',
                     'scripts/low_rank_protocol.py', 'scripts/run_low_rank_efficiency.py'}
    for name, digest in protected.items():
        if (name.startswith(runtime_prefixes) or name in runtime_files) and observed.get(name) != digest:
            raise ValueError('SD recovery changed protected model or common measurement runtime sources')
    for name, digest in observed.items():
        path = (ROOT / name).resolve()
        if (not path.is_relative_to(ROOT) or not path.is_file() or type(digest) is not str
                or shared.file_hash(path) != digest):
            raise ValueError('SD recovery execution source fingerprint changed')
    return {'status': 'pass', 'protected_runtime_unchanged': True,
            'execution_source_sha256': observed}


def _reference(path):
    proof = _pin(path)
    if proof['sha256'] != REFERENCE_SHA256:
        raise ValueError('The saved 99% author reference changed; do not substitute a retrained reference')
    record = full.read_json(path)
    if (record.get('method') != 'sd-lora' or record.get('mode') != 'diagnostic'
            or record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'disabled_side_conversation'
            or record.get('measurement_aggregate_eligible') is not False
            or record.get('efficiency_comparison') is not False or record.get('paper_reproduction') is not False
            or record.get('protocol', {}).get('hyperparameters') != {'optimizer': 'adam', 'lr': .001, 'fc_lr': .008, 'weight_decay': 0.}
            or _exact_counts(record) != 99):
        raise ValueError('The independent author reference must remain the original local-only 99% diagnostic')
    _precision_and_structure(record)
    return record, {**proof, 'accuracy_pct': 99., 'source_kind': 'saved_independent_author_diagnostic',
                    'original_wandb_logging_status': record['wandb_logging_status'],
                    'logging_scope': 'local_only', 'aggregate_excluded': True,
                    'measurement_aggregate_eligible': False}


def validate_sd_record(record, pass_kind, reference, protected_sources):
    from scripts.run_sd_recovery import recovery_protocol, validate_spec
    validate_spec(record)
    if (pass_kind not in ('validation', 'measurement', 'flops')
            or record.get('method') != 'sd-lora' or record.get('model') != 'sd-lora'
            or record.get('pass') != pass_kind or record.get('status') != 'completed'
            or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded'
            or not str(record.get('wandb_url', '')).startswith('https://wandb.ai/laymond1/cl-efficiency/runs/')
            or record.get('protocol') != recovery_protocol(pass_kind)
            or record.get('mode') != ('validation' if pass_kind == 'validation' else 'efficiency')
            or record.get('initial_canonical_pretrained_sha256') != shared.CANONICAL_PRETRAINED_SHA256
            or record.get('measurement_aggregate_eligible') is not (pass_kind == 'measurement')):
        raise ValueError('SD recovery worker identity, exact recipe, completion or upload is invalid')
    for key in ('tasks', 'n_epochs', 'batch_size', 'num_workers', 'seed', 'rank', 'dataset_total_tasks', 'classes_per_task'):
        if type(record['protocol'].get(key)) is not int:
            raise ValueError('SD recovery protocol counts must be exact integers')
    observed = _exact_counts(record)
    if observed < 90:
        raise ValueError('SD recovery pilot accuracy is below the approved 90% minimum')
    _precision_and_structure(record)
    _common_inputs(record, reference)
    groups = record.get('optimizer_group_audits', [])
    if (len(groups) != 2 or any(group.get('role') != role or group.get('lr') != rate
            or any(type(group.get(key)) is not int or group[key] != 1
                   for key in ('task', 'epoch', 'optimizer_update_index'))
            or type(group.get('parameters')) is not int or group['parameters'] <= 0
            for group, role, rate in zip(groups, ('encoder_and_scale', 'classifier'), (.001, .008)))):
        raise ValueError('SD recovery requires actual optimizer-group learning-rate evidence')
    source_proof = _source_fingerprints(record, protected_sources)
    if pass_kind in ('validation', 'measurement'):
        coverage = record.get('metrics', {}).get('gpu_energy_coverage_fraction')
        if type(coverage) not in (int, float) or not math.isfinite(coverage) or coverage < .99 or record.get('sensor_errors'):
            raise ValueError('SD recovery measurements require clean sensors and at least 99% energy coverage')
        for key in ('process_wall_s', 'gpu_board_energy_wh', 'cuda_peak_allocated_mib', 'peak_process_tree_rss_mib'):
            value = record['metrics'].get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('SD recovery resource observations are missing or invalid')
    else:
        value = record.get('flops', {}).get('supported_ops_total')
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('SD recovery requires a positive separate supported-operator FLOPs estimate')
    return source_proof


def _predictions(record, path):
    """Audit the serialized predictions without restoring or evaluating a model."""
    import numpy as np
    path = Path(path).resolve()
    if record.get('implementation') == 'mammoth_native':
        saved = record.get('prediction_files', [])
        if (len(saved) != 1 or type(saved[0].get('task')) is not int or saved[0]['task'] != 1
                or Path(saved[0].get('path', '')).resolve() != path
                or saved[0].get('sha256') != shared.file_hash(path)):
            raise ValueError('Native recovery prediction artifact fingerprint changed')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('predictions', 'labels', 'global_ids')}
    if (any(value.shape != (1000,) or not np.issubdtype(value.dtype, np.integer) for value in arrays.values())
            or len(np.unique(arrays['global_ids'])) != 1000
            or np.any(arrays['labels'] < 0) or np.any(arrays['labels'] >= 10)
            or np.any(arrays['predictions'] < 0) or np.any(arrays['predictions'] >= 10)
            or int(np.sum(arrays['predictions'] == arrays['labels'])) != record['task_results'][0]['seen_correct']):
        raise ValueError('Recovery prediction arrays do not establish the complete saved task accuracy')
    return arrays


def _prior(path):
    prior = full.read_json(path)
    if (prior.get('status') != 'partial_completed' or prior.get('publication_gate') != 'pass_for_accepted_subset'
            or prior.get('method_count') != 8 or set(prior.get('methods', {})) != set(OLD_METHODS)):
        raise ValueError('Recovery requires the immutable accepted eight-method partial report')
    reconstructed = partial.build_report(prior['execution']['file'])
    for key in ('methods', 'sources', 'protocol', 'execution', 'historical_compatibility',
                'failed_observations', 'deferred_jobs', 'diagnostic_evidence', 'report_generator'):
        if prior.get(key) != reconstructed.get(key):
            raise ValueError('Prior report summaries or raw provenance changed')
    return prior, _pin(path)


def validate_acceptance(acceptance, native, reference, native_path, reference_path):
    native_pct, reference_pct = _exact_counts(native), _exact_counts(reference)
    gap = reference_pct - native_pct
    expected = {'reference_pct': float(reference_pct), 'native_pct': float(native_pct), 'drop_pp': float(gap)}
    if (acceptance.get('status') != 'pass' or gap > Fraction(3, 2) or native_pct < 90
            or acceptance.get('accuracy') != expected
            or acceptance.get('thresholds') != {'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3., 'minimum_accuracy_pct': 90.}
            or acceptance.get('structural_checks_pass') is not True
            or acceptance.get('measurement_aggregate_eligible') is not False):
        raise ValueError('Native recovery validation must pass the independent 1.5pp and 90% gates')
    batches = acceptance.get('first_batch_checks', {})
    if any(batches.get(key) is not True for key in ('input_sha256_matches', 'labels_sha256_matches',
                                                  'gradient_sha256_matches', 'seed_matches', 'samples_matches')) or any(
            value is not True for value in batches.values()):
        raise ValueError('Native recovery acceptance must retain matching first-batch evidence')
    predictions = acceptance.get('prediction_checks', {})
    if any(predictions.get(key) is not True for key in ('labels_match', 'global_ids_match')):
        raise ValueError('Native recovery acceptance prediction labels and IDs must match the author reference')
    _check_pin(acceptance.get('native_result'), native_path, 'Native validation')
    _check_pin(acceptance.get('reference_result'), reference_path, 'Author reference')
    native_predictions, source_predictions = Path(native_path).parent / 'task1_predictions.npz', Path(reference_path).parent / 'task1_predictions.npz'
    _check_pin(acceptance.get('native_predictions'), native_predictions, 'Native predictions')
    _check_pin(acceptance.get('source_predictions'), source_predictions, 'Author predictions')
    left, right = _predictions(native, native_predictions), _predictions(reference, source_predictions)
    import numpy as np
    expected_predictions = {key + '_match': bool(np.array_equal(left[key], right[key])) for key in ('labels', 'global_ids')}
    expected_predictions['prediction_arrays_match'] = bool(np.array_equal(left['predictions'], right['predictions']))
    if predictions != expected_predictions:
        raise ValueError('Native recovery acceptance prediction comparison disagrees with saved arrays')
    for key, record in (('native', native), ('reference', reference)):
        value = acceptance.get(key + '_network_sha256')
        if not value or value != record.get('state_audits', {}).get('network_sha256'):
            raise ValueError('Native recovery final network evidence is incomplete')
    if acceptance.get('network_sha256_matches') is not (
            acceptance['native_network_sha256'] == acceptance['reference_network_sha256']):
        raise ValueError('Native recovery network equality diagnostic disagrees with saved states')
    return True


def build_report(run_root=DEFAULT_RUN_ROOT, *, prior_report_path=DEFAULT_PRIOR, reference_path=DEFAULT_REFERENCE):
    run_root = Path(run_root).resolve()
    prior, prior_proof = _prior(prior_report_path)
    reference, reference_proof = _reference(reference_path)
    # The prior controller pins RanPAC's independent worker too. Use the
    # preserved SD worker's complete runtime footprint for the SD comparison;
    # the prior-report reconstruction already checks the full old controller.
    protected = full.read_json(prior['failed_observations'][0]['file'])['execution_source_sha256']
    validation_path = run_root / 'validation/result.json'
    native = full.read_json(validation_path)
    validate_sd_record(native, 'validation', reference, protected)
    acceptance_path = run_root / 'validation/acceptance.json'
    acceptance = full.read_json(acceptance_path)
    validate_acceptance(acceptance, native, reference, validation_path, reference_path)
    records, sources = [], []
    seen_paths, seen_urls = {str(validation_path)}, {native['wandb_url']}
    seen_urls.update(source['wandb_url'] for source in prior['sources'])
    for directory, kind in (('measurement1', 'measurement'), ('measurement2', 'measurement'),
                            ('measurement3', 'measurement'), ('flops', 'flops')):
        path = run_root / directory / 'result.json'
        record = full.read_json(path)
        validate_sd_record(record, kind, reference, protected)
        proof = _pin(path)
        if (str(path) in seen_paths or record['wandb_url'] in seen_urls
                or Path(record.get('output', '')).resolve() != path.parent):
            raise ValueError('Recovery requires four distinct fresh workers in their exact output directories')
        seen_paths.add(str(path))
        seen_urls.add(record['wandb_url'])
        left = _predictions(record, path.parent / 'task1_predictions.npz')
        right = _predictions(reference, Path(reference_path).parent / 'task1_predictions.npz')
        import numpy as np
        if any(not np.array_equal(left[key], right[key]) for key in ('labels', 'global_ids')):
            raise ValueError('Recovery cost workers prediction labels or IDs differ from the author reference')
        records.append(record)
        sources.append({'method': 'sd-lora', 'pass': kind, **proof, 'wandb_url': record['wandb_url'],
                        'execution_source_sha256': record['execution_source_sha256']})
    execution_path = run_root / 'execution.json'
    execution = full.read_json(execution_path)
    from scripts.run_sd_recovery_suite import jobs
    expected_jobs = jobs(run_root)
    if (execution.get('status') != 'completed' or execution.get('completed_gpu_runs') != 5
            or type(execution.get('completed_gpu_runs')) is not int or execution.get('author_retrained') is not False
            or Path(execution.get('output', '')).resolve() != run_root
            or len(execution.get('jobs', [])) != 5 or execution.get('acceptance') != acceptance):
        raise ValueError('Recovery report requires the complete approved five-run execution')
    for job, expected, record in zip(execution['jobs'], expected_jobs, [native, *records]):
        if (job.get('status') != 'completed' or any(job.get(key) != expected[key] for key in ('pass', 'directory', 'command'))
                or job.get('result', {}).get('wandb_url') != record['wandb_url']
                or record['execution_source_sha256'] != execution.get('execution_source_sha256')):
            raise ValueError('Recovery execution schedule, source snapshot or worker identity changed')
        _check_pin(job.get('result'), Path(job['directory']) / 'result.json', 'Scheduled recovery worker')
    methods = deepcopy(prior['methods'])
    sd = full.method_summary('sd-lora', records[:3], records[3], protocol=records[0]['protocol'])
    sd.update(label='SD-LoRA (LR 0.001)', recipe='sd-common-lr001',
              recipe_change={'lora_and_scale_lr': .001, 'classifier_lr': .008, 'train_crop_min': .05,
                             'original_lora_and_scale_lr': .008, 'default_configuration_changed': False})
    sd['method_diagnostics'] = [{key: record.get(key) for key in ('counts', 'state_audits', 'structural_checks')}
                                for record in records[:3]]
    methods['sd-lora'] = sd
    from scripts.run_sd_recovery import recovery_protocol
    return {'schema_version': 1, 'status': 'completed', 'publication_gate': 'pass',
            'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'protocol': recovery_protocol('measurement'), 'method_count': 9,
            'all_requested_methods_passed': True, 'new_measurement_runs': 3, 'new_flops_runs': 1,
            'new_validation_runs': 1, 'new_gpu_training_runs': 5, 'reused_method_count': 8,
            'methods': methods, 'sources': [*deepcopy(prior['sources']), *sources], 'recovery_sources': sources,
            'historical_reused_methods': list(prior['historical_reused_methods']),
            'historical_compatibility': deepcopy(prior['historical_compatibility']),
            'prior_accepted_report': prior_proof, 'prior_execution': deepcopy(prior['execution']),
            'execution': {**_pin(execution_path), 'execution_source_sha256': execution['execution_source_sha256']},
            'failed_observations': deepcopy(prior['failed_observations']),
            'original_deferred_jobs': deepcopy(prior['deferred_jobs']),
            'original_diagnostic_evidence': deepcopy(prior['diagnostic_evidence']),
            'validation': {**_pin(validation_path), 'wandb_url': native['wandb_url'],
                           'acceptance_file': _pin(acceptance_path), 'acceptance': acceptance,
                           'aggregate_excluded': True, 'measurement_aggregate_eligible': False},
            'author_reference': reference_proof,
            'report_generator': _pin(__file__),
            'scope': {'first_task_only': True, 'paper_reproduction': False, 'equal_compute_budget': False,
                      'same_seed_repeats': True, 'seed_uncertainty_estimated': False,
                      'full_machine_energy_measured': False, 'sd_tuned_recipe': True,
                      'sd_default_configuration_changed': False, 'old_failed_sd_aggregate_excluded': True,
                      'validation_author_diagnostic_and_profiler_resources_excluded': True,
                      'historical_measurement_dates_and_source_snapshots_differ': True,
                      'same_augmentation_recipe_not_identical_random_crop_realizations': True,
                      'ease_cpu_threads': 'not_recorded', 'flops': 'partial supported-operator estimates'}}


def markdown(report):
    validation = report['validation']['acceptance']['accuracy']
    rows = ['# Base 첫 태스크: SD-LoRA 회복 후 9방법 효율성 pilot', '',
            'SD-LoRA는 공통 crop 최소값 0.05를 유지하고 LoRA·scale LR만 0.001로 낮췄습니다. '
            'classifier LR은 0.008이며 모델 기본 설정과 기존 결과는 그대로 보존했습니다.', '',
            f'저장된 독립 저자 결과 {validation["reference_pct"]:.1f}%와 native 첫 태스크 '
            f'{validation["native_pct"]:.1f}%를 대조했습니다. 저자 대비 하락폭은 '
            f'{validation["drop_pp"]:.1f}%p로 1.5%p 통과 기준과 정확도 90% 하한을 통과했습니다. '
            'native 검증 1회와 저자 진단은 효율성 평균에 포함하지 않습니다.', '',
            '신규 SD 측정 3회·별도 부분 FLOPs 1회를 기존 승인된 8방법 기록과 합쳤습니다. '
            '기존 SD 10.0% 관측은 격리된 실패 증거로 유지하며 보류했던 원래 3실행은 재작성하지 않습니다.', '',
            'CIFAR100 첫 10클래스(train 5000/test 1000), ViT-B/16 AugReg IN21K→IN1K, '
            'batch 64, seed 1993, GPU 1, FP32입니다. 첫 태스크의 전체 79업데이트와 13구조 검사를 확인했습니다. '
            'HiDe는 TII 1epoch+prompt 1epoch이며 방법별 task-begin/end 연산을 포함합니다.', '',
            '| 방법 | 정확도 % | 학습 시간 s | 벽시간 s, 평균±SD | CUDA peak GiB | RSS peak GiB | CPU s | GPU 보드 Wh, 평균±SD | 부분 TFLOPs |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in report['methods'].values():
        r = item['resources']
        rows.append(f'| {item["label"]} | {item["accuracy_pct"]["mean"]:.2f} | '
                    f'{item["phases"]["train_epoch"]["wall_s"]["mean"]:.2f} | '
                    f'{r["process_wall_s"]["mean"]:.2f} ± {r["process_wall_s"]["std_sample"]:.2f} | '
                    f'{r["cuda_peak_allocated_mib"]["mean"] / 1024:.3f} | '
                    f'{r["peak_process_tree_rss_mib"]["mean"] / 1024:.3f} | '
                    f'{r["process_tree_cpu_s"]["mean"]:.2f} | '
                    f'{r["gpu_board_energy_wh"]["mean"]:.3f} ± {r["gpu_board_energy_wh"]["std_sample"]:.3f} | '
                    f'{item["flops"]["supported_ops_total"] / 1e12:.3f} |')
    rows += ['', '학습 시간은 train_epoch 구간의 평균이며 HiDe는 TII와 prompt 구간의 합계입니다. '
             '벽시간에는 초기화·데이터 로딩·태스크 경계·평가·감사·체크포인트 저장을 포함하고 최종 W&B 업로드는 제외합니다.', '',
             '![9방법 효율성 비교](efficiency_comparison.png)', '', '[PDF 그림](efficiency_comparison.pdf)', '',
             '평균±표본 SD는 동일 seed를 사용한 실행 비용의 변동입니다. 검증·저자 진단·별도 profiler의 '
             '시간·메모리·에너지는 평균에서 제외했습니다. GPU 보드 Wh는 CPU/RAM/전체 머신 에너지를 포함하지 않습니다. '
             '부분 FLOPs는 지원 연산의 추정이며 CPU 선형대수와 미지원 연산 등 모든 비용을 포함하지 않습니다.', '',
             'SD는 안정성 확인을 거친 별도 LR 설정입니다. 이 비교는 동일 계산 예산, 전체 10태스크 성능·망각, '
             '여러 seed의 성능 재현을 평가하지 않습니다. 첫 태스크에는 EWC의 과거 지식 penalty, CL의 과거 branch/KD, '
             'Inf의 과거 subspace 상호작용이 나타나지 않습니다. 측정된 Fisher·feature/SVD/GPM·프로토타입·RP/ridge 비용은 포함했습니다.', '',
             '기존 기록은 날짜·소스·단계 예산이 다릅니다. EASE의 Torch CPU thread 수는 미기록으로 남깁니다. '
             '같은 원본 데이터·augmentation recipe라도 방법별 RNG 경로에 따라 실제 crop은 달라질 수 있습니다.', '',
             '| 방법 | 측정 실행 | 별도 FLOPs |', '|---|---|---|']
    for item in report['methods'].values():
        links = ' · '.join(f'[측정{i}]({url})' for i, url in enumerate(item['wandb_urls'], 1))
        rows.append(f'| {item["label"]} | {links} | [진단]({item["flops_wandb_url"]}) |')
    rows += ['', f'[SD native 검증 W&B]({report["validation"]["wandb_url"]})', '',
             '저자 99.0% 기록은 기존 로컬 진단의 원본 SHA와 함께 별도 보존합니다. '
             '최종 보고서에 보관하더라도 당시 학습이 W&B 온라인으로 실행된 것으로 바꾸지 않습니다.', '',
             '| 격리 관측 | 정확도 | 처리 |', '|---|---:|---|']
    for failed in report['failed_observations']:
        rows.append(f'| [기존 SD-LoRA]({failed["wandb_url"]}) | {failed["accuracy_pct"]:.1f}% | 원본 보존·집계 제외 |')
    return '\n'.join(rows) + '\n'


def write_report(report, output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', report)
    (output / 'results.csv').write_text(csv_text(report))
    (output / 'RESULTS.md').write_text(markdown(report))
    plot(report, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--prior-report', type=Path, default=DEFAULT_PRIOR)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args(argv)
    report = build_report(options.run_root, prior_report_path=options.prior_report, reference_path=options.reference)
    if options.write:
        write_report(report, options.output)
    print(json.dumps({'publication_gate': report['publication_gate'], 'method_count': report['method_count'],
                      'new_gpu_training_runs': report['new_gpu_training_runs'], 'written': options.write}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
