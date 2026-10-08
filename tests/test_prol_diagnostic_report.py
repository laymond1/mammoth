"""Synthetic saved-array evidence tests; no model, GPU or network work."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import report_prol_diagnostic as report


@pytest.fixture
def stage(tmp_path, monkeypatch):
    root, source_root, prior_root = tmp_path / 'run', tmp_path / 'source', tmp_path / 'prior'
    root.mkdir()
    prior_root.mkdir()
    sources = {}
    for name in ('scripts/prol_diagnostic_protocol.py', 'scripts/run_prol_diagnostic.py', 'scripts/diagnose_prol_checkpoint.py',
                 'scripts/report_prol_diagnostic.py', 'scripts/publish_prol_diagnostic.py',
                 'benchmarks/cl_efficiency/prol_diagnostic/run_plan.json'):
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic immutable execution source\n')
        sources[name] = report.shared.file_hash(path)
    monkeypatch.setattr(report, 'ROOT', source_root)
    monkeypatch.setattr(report.shared, 'PRIOR_DIRECTORY', prior_root)
    monkeypatch.setattr(report.shared, 'load_plan', lambda: {})
    monkeypatch.setattr(report, 'historical_callback_hash', lambda: '0' * 64)
    transform = 'synthetic deterministic CIFAR100 test preprocessing'
    transform_hash = hashlib.sha256(transform.encode()).hexdigest()
    prior_labels = np.repeat(np.arange(10), 100).astype(np.int64)
    prior_predictions = prior_labels.copy()
    prior_predictions[:7] = 1
    prior_ids = np.arange(1000, dtype=np.int64)
    np.savez_compressed(prior_root / 'task1_predictions.npz', predictions=prior_predictions, labels=prior_labels,
                        global_ids=prior_ids, evaluation_global_ids=prior_ids, batch_sizes=np.asarray([64] * 15 + [40]))
    prior = {'method': 'prol-prompt', 'model': 'prol-prompt', 'mode': 'validation', 'side': 'author', 'pass': 'validation',
             'status': 'failed', 'computation_status': 'failed', 'git_commit': report.shared.ORIGIN_EXECUTION_COMMIT,
             'wandb_url': report.WAND_PREFIX + 'old-failed', 'execution_source_sha256': {'scripts/old.py': '0' * 64},
             'task_results': [{'task': 1, 'seen_correct': 993, 'seen_samples': 1000}],
             'counts': {'optimizer_updates': 79}, 'accuracy': {'mean_accuracy_pct': 99.3},
             'metrics': {'tasks_completed': 1, 'outer_observe_steps': 79, 'outer_training_stream_examples': 5000, 'evaluation_examples': 1000},
             'data_snapshots': {'task1_test': {'transform_repr': transform, 'transform_repr_sha256': transform_hash}},
             'first_batch_audits': [], 'prediction_files': []}
    report.write_json(prior_root / 'result.json', prior)
    (prior_root / 'task1_state.pt').write_bytes(b'synthetic local-only old checkpoint')
    prior_pins = {name: report.shared.file_hash(prior_root / name) for name in ('result.json', 'task1_predictions.npz', 'task1_state.pt')}
    monkeypatch.setattr(report.shared, 'PRIOR_PINS', prior_pins)
    monkeypatch.setattr(report.shared, 'verify_prior_inputs', lambda: deepcopy(prior))
    for name in ('spec.json', 'artifact_metrics.json', 'phases.json', 'epoch_results.json', 'task_results.json',
                 'first_batch_audits.json', 'state_audits.json', 'method_audits.json', 'lr_audits.json',
                 'routing_audits.json', 'sample_stream_audits.json', 'effective_args.json', 'shared_data.json', 'source_adaptations.json'):
        report.write_json(prior_root / name, prior if name == 'artifact_metrics.json' else {})
    (prior_root / 'telemetry.csv').write_text('elapsed_s,gpu_power_w\n0,100\n')
    ids, labels, old_predictions = prior_ids[-40:], prior_labels[-40:], prior_predictions[-40:]
    inputs = np.zeros((40, 3, 224, 224), dtype=np.float32)
    np.savez_compressed(root / 'tail_inputs.npz', inputs=inputs, labels=labels, global_ids=ids)
    input_hash = report.array_digest('inputs', inputs)
    evidence = {'examples': 40, 'global_ids': ids.tolist(), 'labels': labels.tolist(),
        'original_tail_predictions': old_predictions.tolist(), 'global_ids_sha256': report.array_digest('ids', ids),
        'labels_sha256': report.array_digest('labels', labels), 'input_sha256': input_hash,
        'input_shape': [40, 3, 224, 224], 'input_dtype': 'torch.float32',
        'transform_repr': transform, 'transform_repr_sha256': transform_hash, 'raw_test_sha256': '1' * 64,
        'original_live_logits_available': False, 'local_tail_inputs': {
            'name': 'tail_inputs.npz', 'sha256': report.shared.file_hash(root / 'tail_inputs.npz'),
            'bytes': (root / 'tail_inputs.npz').stat().st_size, 'upload': False}}
    frozen = {'query': {'sha256': report.shared.CANONICAL_PRETRAINED_SHA256, 'tensors': 150},
              'prompt': {'sha256': report.SAVED_FROZEN_PROMPT_SHA256, 'tensors': 100}}
    lifecycle = {'old_num_k': 0, 'new_num_k': 1, 'old_num_c': 0, 'new_num_c': 10, 'kmax_list': [1], 'lmax_list': [10]}
    restoration = {'saved_learned_sha256': report.SAVED_LEARNED_SHA256,
                   'author_restored_learned_sha256': report.SAVED_LEARNED_SHA256,
                   'compact_clone_learned_sha256': report.SAVED_LEARNED_SHA256, 'frozen_encoder': frozen,
                   'checkpoint_digest_matches': True, 'all_saved_learners_restored': True, 'clone_rng_unchanged': True,
                   'query_initial_canonical_matches_pin': True, 'prompt_initial_canonical_matches_pin': True,
                   'clone_rng_before_sha256': '0' * 64, 'clone_rng_after_sha256': '0' * 64,
                   'historical_audit_callback_ast_sha256': '0' * 64,
                   'historical_audit_callback_commit': report.shared.ORIGIN_EXECUTION_COMMIT,
                   'learner_catalogue_tensor_count': 399, 'saved_learned_tensor_count': 399,
                   'author_restored_learned_tensor_count': 399, 'compact_clone_learned_tensor_count': 399,
                   'prompt_lifecycle': lifecycle, 'author_restored_prompt_lifecycle': lifecycle,
                   'compact_clone_prompt_lifecycle': lifecycle}
    arrays = {'logits': np.zeros((40, 40, 10), dtype=np.float32), 'query_features': np.zeros((40, 40, 768), dtype=np.float32),
              'predictions': np.broadcast_to(labels, (40, 40)).copy(), 'global_ids': ids, 'labels': labels,
              'row_indices': np.arange(40, dtype=np.int64),
              **{name: np.zeros((40, 8), dtype=np.uint8) for name in ('rng_cpu_before', 'rng_cpu_after', 'rng_cuda0_before', 'rng_cuda0_after')}}
    arrays['logits'][:, np.arange(40), labels] = 1.
    rows = []
    trace = {}
    for layer in range(7):
        trace[f'layer{layer}_similarity_new'] = np.zeros((40, 10), dtype=np.float32)
        trace[f'layer{layer}_indices'] = np.zeros((40, 1), dtype=np.int64)
        trace[f'layer{layer}_prefix'] = np.zeros((40, 2, 5, 12, 64), dtype=np.float32)
        for when in ('pre_affine', 'post_affine'):
            for affine in ('scaler', 'scaler_v', 'shifter', 'shifter_v'):
                trace[f'layer{layer}_{when}_{affine}'] = (np.ones if affine.startswith('scaler') else np.zeros)((100, 5), dtype=np.float32)
    for index, (side, mode, repeat) in enumerate(itertools.product(report.shared.MODEL_SIDES, report.shared.HOOK_MODES, range(1, 11))):
        row = {'row_index': index, 'branch': side, 'audit_mode': mode, 'audit': mode == 'on', 'cell_name': f'{side}_{mode}',
               'repeat': repeat, 'examples': 40, 'input_sha256': input_hash,
               'rng_before_sha256': report.rng_digest(arrays, index, 'before'),
               'rng_after_sha256': report.rng_digest(arrays, index, 'after'),
               'learned_before_sha256': report.SAVED_LEARNED_SHA256, 'learned_after_sha256': report.SAVED_LEARNED_SHA256,
               'frozen_before': frozen, 'frozen_after': frozen, 'rng_unchanged': True, 'learned_unchanged': True,
               'frozen_unchanged': True, 'finite': True, 'gradients_absent': True,
               'query_training_mode': True, 'prompt_training_mode': False, 'labels_provided': False, 'oracle_task_id': None,
               'original_tail_predictions_match': True,
               **{key + '_sha256': report.array_digest(key, arrays[key][index]) for key in ('logits', 'predictions', 'query_features')}}
        if mode == 'on':
            name = f'trace_{index:02d}.npz'
            np.savez_compressed(root / name, **trace)
            row.update(trace_file=name, trace_sha256=report.shared.file_hash(root / name), active_prefix_layers_audited=7,
                       audit_routing=[{'samples': 40, 'routing_key_count': 10, 'labels_provided': False, 'oracle_task_id': None}])
        rows.append(row)
    spec = {'schema_version': 1, 'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic', 'training_authorized': False,
            'diagnostic_authorized': True, 'source_mode': 'official-active-path', 'paper_equivalent': False,
            'gpu': '1', 'gpu_uuid': report.shared.GPU_UUID, 'wandb_mode': 'online', 'wandb_entity': 'laymond1',
            'wandb_project': 'cl-efficiency', 'protocol': report.shared.protocol(),
            'checkpoint_sha256': report.shared.CHECKPOINT_SHA256, 'canonical_pretrained_sha256': report.shared.CANONICAL_PRETRAINED_SHA256,
            'reference_commit': report.shared.REFERENCE_COMMIT, 'origin_execution_commit': report.shared.ORIGIN_EXECUTION_COMMIT,
            'prior_result_sha256': prior_pins['result.json'], 'prior_predictions_sha256': prior_pins['task1_predictions.npz'],
            'prior_checkpoint_sha256': prior_pins['task1_state.pt'], 'measurement_aggregate_eligible': False,
            'execution_source_sha256': sources, 'output': str(root),
            'data_files': {'files': {'test': {'sha256': '1' * 64}}}}
    record = {**spec, 'status': 'completed', 'computation_status': 'completed', 'wandb_logging_status': 'uploaded',
              'wandb_url': report.WAND_PREFIX + 'diagnostic', 'precision': report.shared.PRECISION,
              'input_sha256': input_hash, 'matrix': rows, 'input_evidence': evidence, 'restoration': restoration,
              'matrix_seed': 2993, 'matrix_rng_initial_sha256': rows[0]['rng_before_sha256'],
              'counts': {'train_steps': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'query_forwards': 40,
                         'prompt_forwards': 40, 'model_evaluations': 40, 'encoder_forward_calls': 80, 'diagnostic_image_visits': 1600},
              'metrics': {}, 'phases': {}}
    report.write_json(root / 'spec.json', spec)
    report.write_json(root / 'restoration.json', restoration)
    report.write_json(root / 'input_evidence.json', evidence)
    report.write_json(root / 'matrix.json', rows)
    np.savez_compressed(root / 'forward_outputs.npz', **arrays)
    (root / 'telemetry.csv').write_text('elapsed_s,gpu_power_w\n0,100\n')
    report.write_json(root / 'phases.json', {})
    record['output_files'] = [{'name': path.name, 'sha256': report.shared.file_hash(path), 'bytes': path.stat().st_size}
                              for path in root.iterdir() if path.is_file() and path.name != 'tail_inputs.npz']
    report.write_json(root / 'result.json', record)
    report.write_json(root / 'artifact_metrics.json', record)
    execution = {'status': 'completed', 'new_gpu_processes': 1, 'new_training': False, 'plan': {},
                 'spec_sha256': report.shared.file_hash(root / 'spec.json'), 'execution_source_sha256': sources,
                 'wandb_url': record['wandb_url'], 'gpu_occupancy': {'gpu_uuid': report.shared.GPU_UUID},
                 'command': [sys.executable, str(source_root / 'scripts/diagnose_prol_checkpoint.py'), '--worker', str(root / 'spec.json')],
                 'result': report.pin(root / 'result.json')}
    report.write_json(root / 'execution.json', execution)
    return root, record, prior, execution


def rewrite(stage, *, arrays=None):
    root, record, prior, execution = stage
    if arrays is not None:
        np.savez_compressed(root / 'forward_outputs.npz', **arrays)
    report.write_json(root / 'matrix.json', record['matrix'])
    for row in record['output_files']:
        path = root / row['name']
        row.update(sha256=report.shared.file_hash(path), bytes=path.stat().st_size)
    report.write_json(root / 'result.json', record)
    report.write_json(root / 'artifact_metrics.json', record)
    execution['result'] = report.pin(root / 'result.json')
    report.write_json(root / 'execution.json', execution)


def test_saved_complete_matrix_is_inference_evidence_not_training_acceptance(stage):
    saved = report.build_report(stage[0])
    assert saved['counts']['model_evaluations'] == 40 and saved['counts']['encoder_forward_calls'] == 80
    assert saved['summary']['reference_clone_comparisons'] == 20
    assert saved['summary']['within_cell_comparisons'] == 180
    assert saved['summary']['hook_comparisons'] == 20
    assert saved['summary']['checkpoint_issue_resolved'] is False
    assert saved['summary']['implementation_acceptance_changed'] is False
    assert saved['scope']['new_cl_accuracy'] is False
    assert saved['trace_evidence']['trace_count'] == 20
    assert len(saved['trace_comparisons']['comparisons']) == 28
    assert saved['trace_comparisons']['generated_prefix_is_not_effective_block_input'] is True
    assert '원인이 해결' in report.markdown(saved)


@pytest.mark.parametrize('field,value', [('model_evaluations', 41), ('encoder_forward_calls', 81), ('diagnostic_image_visits', 1599),
                                       ('backward_calls', 1), ('optimizer_updates', True), ('train_steps', 1)])
def test_wrong_count_or_training_work_cannot_claim_completed_matrix(stage, field, value):
    stage[1]['counts'][field] = value
    rewrite(stage)
    with pytest.raises(ValueError):
        report.build_report(stage[0])


def test_missing_condition_cannot_be_filled_by_repeated_row(stage):
    stage[1]['matrix'][39]['repeat'] = 9
    rewrite(stage)
    with pytest.raises(ValueError, match='condition/repeat'):
        report.build_report(stage[0])


def test_wrong_output_digest_is_rejected(stage):
    stage[1]['matrix'][0]['logits_sha256'] = '0' * 64
    rewrite(stage)
    with pytest.raises(ValueError, match='output hashes'):
        report.build_report(stage[0])


def test_raw_rng_byte_changes_remain_findings_and_do_not_accept_training(stage):
    root, record, prior, execution = stage
    with np.load(root / 'forward_outputs.npz') as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['rng_cpu_after'][0, 0] = 1
    arrays['rng_cpu_before'][1:, 0] = 1
    arrays['rng_cpu_after'][1:, 0] = 1
    for index, row in enumerate(record['matrix']):
        row['rng_before_sha256'] = report.rng_digest(arrays, index, 'before')
        row['rng_after_sha256'] = report.rng_digest(arrays, index, 'after')
        row['rng_unchanged'] = row['rng_before_sha256'] == row['rng_after_sha256']
    rewrite(stage, arrays=arrays)
    saved = report.build_report(root)
    assert saved['summary']['rng_changed_evaluations'] == 1
    assert saved['summary']['diagnosis'].endswith('_with_rng_changes')
    assert saved['summary']['checkpoint_issue_resolved'] is False


def test_raw_rng_disagrees_with_boolean_only_claim(stage):
    stage[1]['matrix'][0]['rng_unchanged'] = False
    rewrite(stage)
    with pytest.raises(ValueError, match='hashes disagree'):
        report.build_report(stage[0])


def test_local_transformed_input_must_match_hash(stage):
    path = stage[0] / 'tail_inputs.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['inputs'][0, 0, 0, 0] = 1.
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='actual local transformed'):
        report.build_report(stage[0])


def test_restoration_lifecycle_is_verified(stage):
    stage[1]['restoration']['prompt_lifecycle']['new_num_c'] = 20
    report.write_json(stage[0] / 'restoration.json', stage[1]['restoration'])
    rewrite(stage)
    with pytest.raises(ValueError, match='lifecycle'):
        report.build_report(stage[0])


def test_trace_arrays_are_linked_and_seven_layer_complete(stage):
    path = stage[0] / 'trace_10.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if not key.startswith('layer6_')}
    np.savez_compressed(path, **arrays)
    stage[1]['matrix'][10]['trace_sha256'] = report.shared.file_hash(path)
    rewrite(stage)
    with pytest.raises(ValueError, match='trace omits'):
        report.build_report(stage[0])


def test_actual_single_process_journal_is_required(stage):
    stage[3]['new_gpu_processes'] = 2
    report.write_json(stage[0] / 'execution.json', stage[3])
    with pytest.raises(ValueError, match='single approved'):
        report.build_report(stage[0])


def test_numeric_output_difference_is_recomputed_without_accuracy_gate(stage):
    root, record, prior, execution = stage
    with np.load(root / 'forward_outputs.npz') as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['logits'][20, 0, 9] -= np.float32(.0001)
    record['matrix'][20]['logits_sha256'] = report.array_digest('logits', arrays['logits'][20])
    rewrite(stage, arrays=arrays)
    saved = report.build_report(root)
    assert saved['summary']['non_bitexact_reference_clone_comparisons'] == 1
    assert saved['summary']['reference_clone_max_abs_difference'] > 0
    assert saved['comparisons']['reference_clone_pairs'][0]['predictions_match_count'] == 40
    assert saved['summary']['implementation_acceptance_changed'] is False


def test_learner_mutation_is_preserved_as_diagnostic_finding(stage):
    record = stage[1]
    for index, row in enumerate(record['matrix'][:20]):
        row['learned_after_sha256'] = '1' * 64
        row['learned_unchanged'] = index > 0
        if index > 0:
            row['learned_before_sha256'] = '1' * 64
    rewrite(stage)
    saved = report.build_report(stage[0])
    assert saved['summary']['learned_state_changed_evaluations'] == 1
    assert saved['summary']['evaluation_start_differs_from_restored_learners'] is True
    assert saved['summary']['checkpoint_issue_resolved'] is False


def test_raw_generated_prefix_difference_is_localized_without_recomputing_forward(stage):
    path = stage[0] / 'trace_30.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['layer2_prefix'][0, 0, 0, 0, 0] = 1.
    np.savez_compressed(path, **arrays)
    stage[1]['matrix'][30]['trace_sha256'] = report.shared.file_hash(path)
    rewrite(stage)
    saved = report.build_report(stage[0])
    comparison = saved['trace_comparisons']['comparisons'][0]
    assert comparison['first_observed_internal_difference_layer'] == 2
    assert comparison['layers'][2]['prefix_max_abs_difference'] == 1.
    assert saved['trace_comparisons']['association_is_not_causal_proof'] is True


def test_incomplete_diagnostic_stays_failed_and_preserves_individual_evidence(stage):
    root, record, prior, execution = stage
    record.update(status='failed', computation_status='failed', error_type='RuntimeError')
    execution['status'] = 'failed'
    rewrite(stage)
    saved = report.build_report(root)
    assert saved['status'] == 'failed'
    assert saved['publication_gate'] == 'partial_evidence_only'
    assert saved['summary']['diagnosis'] == 'diagnostic_incomplete'
    assert not saved['comparisons']
    assert 'incomplete_diagnostic' in report.csv_text(saved)
