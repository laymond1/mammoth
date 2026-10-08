"""Read-only evidence report for the approved 40-evaluation PROL diagnosis.

A restored reference is a newly reconstructed model, not the old live object.
No training, model construction, credentials or network calls are performed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prol_diagnostic_protocol as shared
from scripts.cl_efficiency import write_json
from scripts.report_hide_results import read_json
from scripts.report_prol_validation import pin, check_pin, strict_equal, _digest, _finite, _cost_evidence

DEFAULT_RUN_ROOT = shared.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / 'benchmarks/cl_efficiency/prol_diagnostic'
REPORT_FILES = ('results.json', 'results.csv', 'RESULTS.md')
WAND_PREFIX = 'https://wandb.ai/laymond1/cl-efficiency/runs/'
SAVED_LEARNED_SHA256 = 'd505f62e3f9c2a48f59314f67e5f51956927cc8fbd3c4bb47641b9b0a267ef04'
SAVED_FROZEN_PROMPT_SHA256 = 'bb72833cf571cbb39a57bde79dbb2232bd0fead52f3927d67166f42fedf03bec'


def array_digest(name, value):
    """Match the pinned reference's named tensor digest without importing torch."""
    import numpy as np
    dtype = {np.dtype('float32'): 'torch.float32', np.dtype('int64'): 'torch.int64', np.dtype('uint8'): 'torch.uint8'}.get(value.dtype)
    if dtype is None:
        raise ValueError('PROL diagnostic arrays require exact FP32/int64/uint8 types')
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(dtype.encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def rng_digest(arrays, row, when):
    import numpy as np
    digest = hashlib.sha256()
    for name in ('cpu', 'cuda0'):
        value = arrays[f'rng_{name}_{when}'][row]
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b'torch.uint8')
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _integer(values, key, expected):
    if type(values.get(key)) is not int or values[key] != expected:
        raise ValueError('PROL diagnostic scope requires exact inference-only counts')


def _outputs(record, run_root, *, complete=True):
    """Every raw output byte is immutable; checkpoint/images remain local."""
    files = {}
    for row in record.get('output_files', []):
        name = row.get('name')
        if (not isinstance(name, str) or Path(name).name != name or name in files
                or type(row.get('bytes')) is not int or row['bytes'] <= 0 or not _digest(row.get('sha256'))):
            raise ValueError('PROL diagnostic output manifest is invalid')
        proof = pin(run_root / name)
        if row['sha256'] != proof['sha256'] or row['bytes'] != proof['bytes']:
            raise ValueError('PROL diagnostic raw evidence changed')
        files[name] = proof
    required = {'matrix.json', 'forward_outputs.npz', 'restoration.json', 'input_evidence.json', 'telemetry.csv', 'phases.json'}
    if complete and not required.issubset(files):
        raise ValueError('PROL diagnostic output evidence is incomplete')
    forbidden = {'tail_inputs.npz', 'task1_state.pt', 'initial_state.pt'}
    if forbidden.intersection(files) or any(Path(name).suffix not in ('.json', '.csv', '.npz') for name in files):
        raise ValueError('PROL diagnostic manifest must exclude raw images and model checkpoints')
    return files


def _historical_input(record):
    old = shared.verify_prior_inputs()
    if record.get('prior_result_sha256') != shared.PRIOR_PINS['result.json']:
        raise ValueError('PROL diagnosis substituted its historical failed run')
    return old, {'name': 'original_failed_author', 'method': 'prol-prompt', 'side': 'author', 'mode': 'validation',
                 'pass': 'validation', **pin(shared.PRIOR_DIRECTORY / 'result.json'), 'wandb_url': old['wandb_url'],
                 'git_commit': old['git_commit'], 'historical_execution_sources': old['execution_source_sha256']}


def _source_pins(record):
    hashes = record.get('execution_source_sha256', {})
    required = ('scripts/prol_diagnostic_protocol.py', 'scripts/run_prol_diagnostic.py', 'scripts/diagnose_prol_checkpoint.py',
                'scripts/report_prol_diagnostic.py', 'scripts/publish_prol_diagnostic.py',
                'benchmarks/cl_efficiency/prol_diagnostic/run_plan.json')
    if not hashes or any(name not in hashes for name in required):
        raise ValueError('PROL diagnostic execution source fingerprints are incomplete')
    for name, digest in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or shared.file_hash(path) != digest:
            raise ValueError('PROL diagnostic execution source changed')
    return hashes


def historical_callback_hash():
    source = subprocess.run(['git', 'show', f'{shared.ORIGIN_EXECUTION_COMMIT}:scripts/run_prol_worker.py'],
                            cwd=shared.ROOT, capture_output=True, check=True).stdout.decode()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ProlAudit')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'prompt_output')
    return hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()


def restoration_evidence(record, run_root, rows):
    saved = read_json(run_root / 'restoration.json')
    if not strict_equal(record.get('restoration'), saved):
        raise ValueError('PROL restoration file and worker record disagree')
    learner = saved.get('saved_learned_sha256')
    if (learner != SAVED_LEARNED_SHA256 or saved.get('author_restored_learned_sha256') != learner
            or saved.get('compact_clone_learned_sha256') != learner
            or any(saved.get(key) is not True for key in ('checkpoint_digest_matches', 'all_saved_learners_restored',
                'clone_rng_unchanged', 'query_initial_canonical_matches_pin', 'prompt_initial_canonical_matches_pin'))
            or not _digest(saved.get('clone_rng_before_sha256'))
            or saved['clone_rng_before_sha256'] != saved.get('clone_rng_after_sha256')
            or saved.get('historical_audit_callback_commit') != shared.ORIGIN_EXECUTION_COMMIT
            or saved.get('historical_audit_callback_ast_sha256') != historical_callback_hash()):
        raise ValueError('PROL reference/clone restoration or historical audit callback evidence is invalid')
    _integer(saved, 'learner_catalogue_tensor_count', 399)
    for key in ('saved_learned_tensor_count', 'author_restored_learned_tensor_count', 'compact_clone_learned_tensor_count'):
        _integer(saved, key, 399)
    lifecycle = {'old_num_k': 0, 'new_num_k': 1, 'old_num_c': 0, 'new_num_c': 10, 'kmax_list': [1], 'lmax_list': [10]}
    if any(not strict_equal(saved.get(key), lifecycle) for key in ('prompt_lifecycle',
            'author_restored_prompt_lifecycle', 'compact_clone_prompt_lifecycle')):
        raise ValueError('PROL source checkpoint lifecycle changed')
    frozen = saved.get('frozen_encoder', {})
    if (frozen.get('query') != {'sha256': shared.CANONICAL_PRETRAINED_SHA256, 'tensors': 150}
            or frozen.get('prompt') != {'sha256': SAVED_FROZEN_PROMPT_SHA256, 'tensors': 100}):
        raise ValueError('PROL restored frozen encoder pin changed')
    if rows[0]['learned_before_sha256'] != learner or not strict_equal(rows[0]['frozen_before'], frozen):
        raise ValueError('PROL first inference did not begin from its saved/restored checkpoint')
    return saved


def input_evidence(record, run_root, arrays, old):
    import numpy as np
    saved = read_json(run_root / 'input_evidence.json')
    if not strict_equal(saved, record.get('input_evidence')):
        raise ValueError('PROL diagnostic input file and worker record disagree')
    _integer(saved, 'examples', 40)
    if (saved.get('global_ids') != arrays['global_ids'].tolist() or saved.get('labels') != arrays['labels'].tolist()
            or saved.get('global_ids_sha256') != array_digest('ids', arrays['global_ids'])
            or saved.get('labels_sha256') != array_digest('labels', arrays['labels'])
            or saved.get('input_shape') != [40, 3, 224, 224] or saved.get('input_dtype') != 'torch.float32'
            or saved.get('input_sha256') != record.get('input_sha256')
            or saved.get('original_live_logits_available') is not False):
        raise ValueError('PROL diagnostic sample identity or transformed input metadata changed')
    with np.load(run_root / 'tail_inputs.npz', allow_pickle=False) as archive:
        inputs, ids, labels = (archive[key].copy() for key in ('inputs', 'global_ids', 'labels'))
    if (inputs.shape != (40, 3, 224, 224) or inputs.dtype != np.dtype('float32') or not np.isfinite(inputs).all()
            or not np.array_equal(ids, arrays['global_ids']) or not np.array_equal(labels, arrays['labels'])
            or array_digest('inputs', inputs) != record['input_sha256']):
        raise ValueError('PROL diagnostic hashes disagree with actual local transformed image tensors')
    local_proof = pin(run_root / 'tail_inputs.npz')
    local = saved.get('local_tail_inputs', {})
    if (local.get('name') != 'tail_inputs.npz' or local.get('sha256') != local_proof['sha256']
            or type(local.get('bytes')) is not int or local['bytes'] != local_proof['bytes']
            or local.get('upload') is not False):
        raise ValueError('PROL diagnostic local image artifact fingerprint or upload policy changed')
    snapshot = old['data_snapshots']['task1_test']
    if (saved.get('transform_repr') != snapshot['transform_repr']
            or saved.get('transform_repr_sha256') != snapshot['transform_repr_sha256']
            or hashlib.sha256(saved['transform_repr'].encode()).hexdigest() != saved['transform_repr_sha256']
            or saved.get('raw_test_sha256') != record.get('data_files', {}).get('files', {}).get('test', {}).get('sha256')):
        raise ValueError('PROL diagnostic changed the original deterministic test preprocessing or raw data pin')
    return saved, local_proof


def trace_evidence(rows, run_root, files):
    import numpy as np
    names = {f'trace_{index:02d}.npz' for index, row in enumerate(rows) if row['audit']}
    if names != {name for name in files if name.startswith('trace_')}:
        raise ValueError('PROL diagnostic must retain exactly the twenty hook-ON traces')
    for index, row in enumerate(rows):
        if not row['audit']:
            if any(key in row for key in ('trace_file', 'trace_sha256', 'audit_routing', 'active_prefix_layers_audited')):
                raise ValueError('PROL hook-OFF evaluations unexpectedly recorded internal audit hooks')
            continue
        name = f'trace_{index:02d}.npz'
        if row.get('trace_file') != name or row.get('trace_sha256') != files[name]['sha256']:
            raise ValueError('PROL hook-ON trace is not linked to its actual evaluation row')
        _integer(row, 'active_prefix_layers_audited', 7)
        routes = row.get('audit_routing', [])
        if (len(routes) != 1 or routes[0].get('samples') != 40 or routes[0].get('routing_key_count') != 10
                or routes[0].get('labels_provided') is not False or routes[0].get('oracle_task_id', 'missing') is not None):
            raise ValueError('PROL diagnostic hook routing used incomplete keys or an oracle')
        expected = {}
        for layer in range(7):
            expected[f'layer{layer}_similarity_new'] = ((40, 10), np.dtype('float32'))
            expected[f'layer{layer}_indices'] = ((40, 1), np.dtype('int64'))
            expected[f'layer{layer}_prefix'] = ((40, 2, 5, 12, 64), np.dtype('float32'))
            for when in ('pre_affine', 'post_affine'):
                for affine in ('scaler', 'scaler_v', 'shifter', 'shifter_v'):
                    expected[f'layer{layer}_{when}_{affine}'] = ((100, 5), np.dtype('float32'))
        with np.load(run_root / name, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError('PROL hook-ON trace omits source layers or operands')
            for key, (shape, dtype) in expected.items():
                value = archive[key]
                if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                    raise ValueError('PROL source trace shape, precision or finiteness changed')
                if key.endswith('_indices') and (np.any(value < 0) or np.any(value >= 10)):
                    raise ValueError('PROL source trace selected future class keys')
    return {'trace_count': 20, 'active_layers_per_trace': 7,
            'prefix_meaning': 'Raw generated e_prompt returned by source compute_att_over_prompt; later blocks also combine adjacent prefixes',
            'source_operands_captured_without_recomputation': True,
            'causal_proof_of_duplicate_write_winner': False}


def trace_comparisons(rows, run_root, conditions):
    """Compare saved actual operands; do not recompute routing or run models."""
    import numpy as np
    observed = {}
    for index, row in enumerate(rows):
        if row['audit']:
            with np.load(run_root / row['trace_file'], allow_pickle=False) as archive:
                observed[index] = {key: archive[key].copy() for key in archive.files}
    pairs = [('reference_clone', repeat, conditions[('author_restored', 'on', repeat)], conditions[('compact_clone', 'on', repeat)])
             for repeat in range(1, 11)]
    pairs.extend((side + '_repeat_vs_first', repeat, conditions[(side, 'on', 1)], conditions[(side, 'on', repeat)])
                 for side in shared.MODEL_SIDES for repeat in range(2, 11))
    results = []
    for kind, repeat, left, right in pairs:
        layers = []
        for layer in range(7):
            values = {'layer': layer}
            for name in ('indices', 'similarity_new', 'prefix'):
                a, b = observed[left][f'layer{layer}_{name}'], observed[right][f'layer{layer}_{name}']
                values[name + '_bitexact'] = bool(np.array_equal(a, b))
                if name != 'indices':
                    values[name + '_max_abs_difference'] = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
            affine_matches, clamps_valid = True, True
            for when in ('pre_affine', 'post_affine'):
                for affine in ('scaler', 'scaler_v', 'shifter', 'shifter_v'):
                    a, b = observed[left][f'layer{layer}_{when}_{affine}'], observed[right][f'layer{layer}_{when}_{affine}']
                    affine_matches = affine_matches and bool(np.array_equal(a, b))
                    if when == 'post_affine':
                        lower, upper = (.999, 1.001) if affine.startswith('scaler') else (-.0001, .0001)
                        clamps_valid = clamps_valid and bool(((a >= lower) & (a <= upper)).all() and ((b >= lower) & (b <= upper)).all())
            values.update(pre_post_affine_arrays_match=affine_matches, post_affine_clamp_bounds_valid=clamps_valid)
            layers.append(values)
        changed = [row['layer'] for row in layers if any(row[name + '_bitexact'] is False for name in ('indices', 'similarity_new', 'prefix'))
                   or row['pre_post_affine_arrays_match'] is False]
        results.append({'kind': kind, 'repeat': repeat, 'left_row_index': left, 'right_row_index': right,
                        'first_observed_internal_difference_layer': min(changed) if changed else None,
                        'layers': layers})
    return {'comparisons': results, 'within_cell_scope': 'Each ON repeat2..10 versus its first repeat',
            'generated_prefix_is_not_effective_block_input': True, 'association_is_not_causal_proof': True}


def _matrix(record, run_root, old):
    import numpy as np
    rows = record.get('matrix', [])
    saved_rows = json.loads((run_root / 'matrix.json').read_text())
    if len(rows) != 40 or not strict_equal(rows, saved_rows):
        raise ValueError('PROL diagnostic requires every one of the forty saved evaluations')
    with np.load(run_root / 'forward_outputs.npz', allow_pickle=False) as archive:
        keys = ('logits', 'query_features', 'predictions', 'global_ids', 'labels', 'row_indices',
                'rng_cpu_before', 'rng_cpu_after', 'rng_cuda0_before', 'rng_cuda0_after')
        if set(archive.files) != set(keys):
            raise ValueError('PROL raw output and single-GPU RNG byte catalogue changed')
        arrays = {key: archive[key].copy() for key in keys}
    shapes = {'logits': (40, 40, 10), 'query_features': (40, 40, 768), 'predictions': (40, 40),
              'global_ids': (40,), 'labels': (40,), 'row_indices': (40,)}
    for key, value in arrays.items():
        if key.startswith('rng_'):
            if value.ndim != 2 or value.shape[0] != 40 or value.shape[1] < 1 or value.dtype != np.dtype('uint8'):
                raise ValueError('PROL diagnostic requires raw CPU/CUDA RNG bytes per evaluation')
            continue
        dtype = np.dtype('float32') if key in ('logits', 'query_features') else np.dtype('int64')
        if value.shape != shapes[key] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError('PROL diagnostic raw output shapes, precision or finiteness are invalid')
    if (len(np.unique(arrays['global_ids'])) != 40 or not np.array_equal(arrays['row_indices'], np.arange(40))
            or np.any(arrays['labels'] < 0) or np.any(arrays['labels'] >= 10)
            or not np.array_equal(arrays['predictions'], arrays['logits'].argmax(2))):
        raise ValueError('PROL diagnostic IDs, row order or argmax predictions are invalid')
    with np.load(shared.PRIOR_DIRECTORY / 'task1_predictions.npz', allow_pickle=False) as archive:
        prior_arrays = {key: archive[key].copy() for key in ('predictions', 'labels', 'global_ids', 'evaluation_global_ids')}
    if not np.array_equal(arrays['global_ids'], prior_arrays['evaluation_global_ids'][-40:]):
        raise ValueError('PROL diagnostic must use the exact old final forty images in original order')
    locations = np.searchsorted(prior_arrays['global_ids'], arrays['global_ids'])
    if (np.any(locations >= len(prior_arrays['global_ids']))
            or not np.array_equal(prior_arrays['global_ids'][locations], arrays['global_ids'])
            or not np.array_equal(prior_arrays['labels'][locations], arrays['labels'])):
        raise ValueError('PROL diagnostic tail labels or sample identity changed')
    prior_predictions = prior_arrays['predictions'][locations]
    condition_rows = {}
    for index, row in enumerate(rows):
        _integer(row, 'row_index', index)
        key = (row.get('branch'), row.get('audit_mode'), row.get('repeat'))
        if (key[0] not in shared.MODEL_SIDES or key[1] not in shared.HOOK_MODES
                or type(key[2]) is not int or not 1 <= key[2] <= 10 or key in condition_rows
                or row.get('audit') is not (key[1] == 'on') or row.get('cell_name') != f'{key[0]}_{key[1]}'):
            raise ValueError('PROL diagnostic condition/repeat matrix changed')
        condition_rows[key] = index
        _integer(row, 'examples', 40)
        if (row.get('query_training_mode') is not True or row.get('prompt_training_mode') is not False
                or row.get('labels_provided') is not False or row.get('oracle_task_id', 'missing') is not None):
            raise ValueError('PROL diagnostic model mode or inference oracle contract changed')
        if row.get('input_sha256') != record.get('input_sha256') or not _digest(row.get('input_sha256')):
            raise ValueError('PROL diagnostic evaluations changed the common transformed inputs')
        for prefix in ('rng', 'learned'):
            before, after = row.get(prefix + '_before_sha256'), row.get(prefix + '_after_sha256')
            if (not _digest(before) or not _digest(after)
                    or row.get(prefix + '_unchanged') is not (before == after)):
                raise ValueError('PROL inference RNG/learned-state hashes disagree with reported observations')
        if (row['rng_before_sha256'] != rng_digest(arrays, index, 'before')
                or row['rng_after_sha256'] != rng_digest(arrays, index, 'after')):
            raise ValueError('PROL diagnostic RNG hashes disagree with saved actual state bytes')
        if (row.get('frozen_unchanged') is not strict_equal(row.get('frozen_before'), row.get('frozen_after'))
                or row.get('finite') is not True or row.get('gradients_absent') is not True):
            raise ValueError('PROL diagnostic frozen state, finite outputs or gradients are invalid')
        for when in ('before', 'after'):
            frozen = row['frozen_' + when]
            if (set(frozen) != {'query', 'prompt'} or type(frozen['query'].get('tensors')) is not int
                    or frozen['query']['tensors'] != 150 or not _digest(frozen['query'].get('sha256'))
                    or type(frozen['prompt'].get('tensors')) is not int or frozen['prompt']['tensors'] != 100
                    or not _digest(frozen['prompt'].get('sha256'))):
                raise ValueError('PROL diagnostic frozen encoder catalogue is incomplete')
        for key in ('logits', 'predictions', 'query_features'):
            if row.get(key + '_sha256') != array_digest(key, arrays[key][index]):
                raise ValueError('PROL diagnostic output hashes do not describe saved actual arrays')
        if row.get('original_tail_predictions_match') is not bool(np.array_equal(arrays['predictions'][index], prior_predictions)):
            raise ValueError('PROL diagnostic saved-tail prediction agreement is inconsistent')
    expected_conditions = set(itertools.product(shared.MODEL_SIDES, shared.HOOK_MODES, range(1, 11)))
    if set(condition_rows) != expected_conditions:
        raise ValueError('PROL diagnostic matrix omits an approved condition')
    return rows, arrays, condition_rows, prior_predictions


def numeric_comparison(arrays, left, right):
    import numpy as np
    logits_left, logits_right = arrays['logits'][left], arrays['logits'][right]
    difference = np.abs(logits_left.astype(np.float64) - logits_right.astype(np.float64))
    query_difference = np.abs(arrays['query_features'][left].astype(np.float64) - arrays['query_features'][right].astype(np.float64))
    return {'left_row_index': left, 'right_row_index': right,
            'logits_bitexact': bool(np.array_equal(logits_left, logits_right)),
            'logits_max_abs_difference': float(difference.max()), 'logits_mean_abs_difference': float(difference.mean()),
            'logits_differing_elements': int(np.count_nonzero(logits_left != logits_right)),
            'predictions_match_count': int(np.count_nonzero(arrays['predictions'][left] == arrays['predictions'][right])),
            'predictions_total': 40, 'query_features_bitexact': bool(np.array_equal(arrays['query_features'][left], arrays['query_features'][right])),
            'query_features_max_abs_difference': float(query_difference.max())}


def comparisons(arrays, conditions, prior_predictions):
    import numpy as np
    paired = [{'audit_mode': mode, 'repeat': repeat,
               **numeric_comparison(arrays, conditions[('author_restored', mode, repeat)], conditions[('compact_clone', mode, repeat)])}
              for mode in shared.HOOK_MODES for repeat in range(1, 11)]
    repetitions = [{'branch': side, 'audit_mode': mode, 'left_repeat': left, 'right_repeat': right,
                    **numeric_comparison(arrays, conditions[(side, mode, left)], conditions[(side, mode, right)])}
                   for side in shared.MODEL_SIDES for mode in shared.HOOK_MODES for left, right in itertools.combinations(range(1, 11), 2)]
    hooks = [{'branch': side, 'repeat': repeat,
              **numeric_comparison(arrays, conditions[(side, 'off', repeat)], conditions[(side, 'on', repeat)])}
             for side in shared.MODEL_SIDES for repeat in range(1, 11)]
    agreement = [{'branch': side, 'audit_mode': mode, 'repeat': repeat, 'row_index': index,
                  'saved_prediction_agreement_count': int(np.count_nonzero(arrays['predictions'][index] == prior_predictions)),
                  'compared_examples': 40, 'new_cl_accuracy': False}
                 for (side, mode, repeat), index in conditions.items()]
    return {'reference_clone_pairs': paired, 'within_cell_repetitions': repetitions, 'hook_on_off': hooks,
            'saved_tail_prediction_agreement': agreement}


def evidence_summary(values):
    pairs, repeats, hooks = (values[key] for key in ('reference_clone_pairs', 'within_cell_repetitions', 'hook_on_off'))
    reproduced = any(not row['logits_bitexact'] for row in pairs)
    repeat_variation = any(not row['logits_bitexact'] for row in repeats)
    hook_difference = any(not row['logits_bitexact'] for row in hooks)
    return {'reference_clone_comparisons': len(pairs), 'non_bitexact_reference_clone_comparisons': sum(not row['logits_bitexact'] for row in pairs),
            'within_cell_comparisons': len(repeats), 'non_bitexact_within_cell_comparisons': sum(not row['logits_bitexact'] for row in repeats),
            'hook_comparisons': len(hooks), 'non_bitexact_hook_comparisons': sum(not row['logits_bitexact'] for row in hooks),
            'reference_clone_max_abs_difference': max(row['logits_max_abs_difference'] for row in pairs),
            'within_cell_max_abs_difference': max(row['logits_max_abs_difference'] for row in repeats),
            'hook_max_abs_difference': max(row['logits_max_abs_difference'] for row in hooks),
            'diagnosis': 'reference_clone_output_difference_observed' if reproduced else 'reference_clone_output_difference_not_observed',
            'within_cell_variation_observed': repeat_variation, 'hook_association_observed': hook_difference,
            'checkpoint_issue_resolved': False, 'original_live_probe_reconstructed': False,
            'implementation_acceptance_changed': False, 'training_retry_authorized': False}


def worker_files(record, run_root):
    files = _outputs(record, Path(run_root).resolve(), complete=record.get('status') == 'completed')
    files['spec.json'] = pin(Path(run_root) / 'spec.json')
    files['artifact_metrics.json'] = pin(Path(run_root) / 'artifact_metrics.json')
    # A failed worker must upload its sanitized failure context as part of the
    # same immutable small artifact. Never silently omit a local failure proof.
    if (Path(run_root) / 'error_context.json').is_file():
        files['error_context.json'] = pin(Path(run_root) / 'error_context.json')
    return files


def execution_evidence(record, run_root, sources, plan):
    journal = read_json(run_root / 'execution.json')
    spec_path = run_root / 'spec.json'
    spec = read_json(spec_path)
    expected_status = 'completed' if record.get('status') == 'completed' else 'failed'
    if (journal.get('status') != expected_status or type(journal.get('new_gpu_processes')) is not int
            or journal['new_gpu_processes'] != 1 or journal.get('new_training') is not False
            or not strict_equal(journal.get('plan'), plan) or journal.get('spec_sha256') != shared.file_hash(spec_path)
            or not strict_equal(journal.get('execution_source_sha256'), sources)
            or expected_status == 'completed' and journal.get('wandb_url') != record.get('wandb_url')
            or journal.get('gpu_occupancy', {}).get('gpu_uuid') != shared.GPU_UUID):
        raise ValueError('PROL diagnostic launcher journal does not prove the single approved GPU process')
    command = journal.get('command', [])
    if (len(command) != 4 or Path(command[1]).resolve() != ROOT / 'scripts/diagnose_prol_checkpoint.py'
            or command[2] != '--worker' or Path(command[3]).resolve() != spec_path):
        raise ValueError('PROL diagnostic journal scheduled an unexpected worker command')
    if any(not strict_equal(record.get(key), value) for key, value in spec.items()):
        raise ValueError('PROL diagnostic worker record changed its pinned launch specification')
    check_pin(journal.get('result'), run_root / 'result.json')
    return pin(run_root / 'execution.json')


def partial_report(record, run_root, plan):
    counts = record.get('counts', {})
    limits = {'train_steps': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'query_forwards': 40,
              'prompt_forwards': 40, 'model_evaluations': 40, 'encoder_forward_calls': 80, 'diagnostic_image_visits': 1600}
    if any(type(counts.get(key)) is not int or not 0 <= counts[key] <= maximum for key, maximum in limits.items()):
        raise ValueError('PROL incomplete diagnostic exceeded its approved inference-only scope')
    sources = _source_pins(record)
    execution = execution_evidence(record, run_root, sources, plan)
    old, historical = _historical_input(record)
    files = _outputs(record, run_root, complete=False)
    rows = record.get('matrix', [])
    if not isinstance(rows, list) or len(rows) > 40:
        raise ValueError('PROL incomplete diagnostic matrix exceeded its approved scope')
    return {'schema_version': 1, 'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic', 'status': 'failed',
            'publication_gate': 'partial_evidence_only', 'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_root': str(run_root), 'new_gpu_processes': 1, 'new_training_runs': 0, 'counts': counts,
            'protocol': shared.protocol(), 'source': {**pin(run_root / 'result.json'), 'wandb_url': record.get('wandb_url'),
                'execution_source_sha256': sources}, 'historical_failed_source': historical, 'raw_files': files,
            'matrix': rows, 'comparisons': {}, 'summary': {'diagnosis': 'diagnostic_incomplete',
                'recorded_evaluations': len(rows), 'checkpoint_issue_resolved': False, 'implementation_acceptance_changed': False,
                'error_type': record.get('error_type')}, 'error_context': record.get('error_context'),
            'execution': execution, 'observed_metrics': record.get('metrics', {}), 'observed_phases': record.get('phases', {}),
            'report_generator': pin(__file__), 'scope': {'inference_only': True, 'new_cl_accuracy': False,
                'measurement_aggregate_eligible': False, 'original_failed_validation_preserved': True,
                'checkpoint_issue_resolved': False, 'subsequent_training_authorized': False}}


def build_report(run_root=DEFAULT_RUN_ROOT):
    import numpy as np
    run_root = Path(run_root).resolve()
    record = read_json(run_root / 'result.json')
    plan = shared.load_plan()
    shared.validate_spec(record)
    if record.get('status') == 'failed':
        return partial_report(record, run_root, plan)
    if (record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('wandb_logging_status') != 'uploaded' or not str(record.get('wandb_url', '')).startswith(WAND_PREFIX)
            or Path(record.get('output', '')).resolve() != run_root):
        raise ValueError('PROL diagnostic worker must have completed its exact approved matrix')
    if not strict_equal(record.get('precision'), shared.PRECISION):
        raise ValueError('PROL diagnostic precision changed')
    counts = record.get('counts', {})
    expected_counts = {'train_steps': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'query_forwards': 40,
                       'prompt_forwards': 40, 'model_evaluations': 40, 'encoder_forward_calls': 80, 'diagnostic_image_visits': 1600}
    for key, value in expected_counts.items():
        _integer(counts, key, value)
    if any('train' in key or 'backward' in key or 'optimizer' in key for key in counts
           if key not in expected_counts and counts[key] != 0):
        raise ValueError('PROL diagnostic unexpectedly recorded additional training work')
    sources = _source_pins(record)
    execution = execution_evidence(record, run_root, sources, plan)
    old, historical = _historical_input(record)
    files = _outputs(record, run_root)
    rows, arrays, conditions, prior_predictions = _matrix(record, run_root, old)
    traces = trace_evidence(rows, run_root, files)
    values = comparisons(arrays, conditions, prior_predictions)
    summary = evidence_summary(values)
    inputs, local_inputs = input_evidence(record, run_root, arrays, old)
    restoration = restoration_evidence(record, run_root, rows)
    _integer(record, 'matrix_seed', 2993)
    if inputs.get('original_tail_predictions') != prior_predictions.tolist():
        raise ValueError('PROL diagnostic old forty-image prediction reference changed')
    if record.get('matrix_rng_initial_sha256') != rows[0]['rng_before_sha256']:
        raise ValueError('PROL diagnostic initial RNG does not match the first actual forward')
    for previous, current in zip(rows, rows[1:]):
        if previous['rng_after_sha256'] != current['rng_before_sha256']:
            raise ValueError('PROL diagnostic changed RNG outside its recorded inference operations')
    summary.update(rng_changed_evaluations=sum(not row['rng_unchanged'] for row in rows),
                   learned_state_changed_evaluations=sum(not row['learned_unchanged'] for row in rows),
                   frozen_state_changed_evaluations=sum(not row['frozen_unchanged'] for row in rows),
                   learned_start_distinct_digest_count=len({row['learned_before_sha256'] for row in rows}),
                   evaluation_start_differs_from_restored_learners=any(row['learned_before_sha256'] != restoration['saved_learned_sha256'] for row in rows))
    baselines = {}
    for side in shared.MODEL_SIDES:
        first = next(row for row in rows if row['branch'] == side)
        baselines[side] = {'first_row_index': first['row_index'],
                           'learned_start_matches_restored_checkpoint': first['learned_before_sha256'] == restoration['saved_learned_sha256'],
                           'frozen_start_matches_restored_checkpoint': strict_equal(first['frozen_before'], restoration['frozen_encoder'])}
    if summary['rng_changed_evaluations']:
        summary['diagnosis'] += '_with_rng_changes'
    if summary['learned_state_changed_evaluations'] or summary['frozen_state_changed_evaluations']:
        summary['diagnosis'] += '_with_state_changes'
    try:
        _cost_evidence(record, run_root / 'result.json', enforce_coverage=False)
        cost_audit = {'status': 'raw_power_verified', 'efficiency_comparison_eligible': False}
    except (ValueError, FileNotFoundError, KeyError):
        cost_audit = {'status': 'unverified_or_incomplete_sensor_evidence', 'efficiency_comparison_eligible': False}
    return {'schema_version': 1, 'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic', 'status': 'completed',
            'publication_gate': 'evidence_verified', 'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_root': str(run_root), 'new_gpu_processes': 1, 'new_training_runs': 0,
            'counts': counts, 'protocol': shared.protocol(), 'source': {**pin(run_root / 'result.json'),
                'wandb_url': record['wandb_url'], 'execution_source_sha256': sources},
            'historical_failed_source': historical, 'raw_files': files, 'matrix': rows,
            'comparisons': values, 'summary': summary, 'input_evidence': inputs, 'restoration': restoration,
            'local_transformed_inputs': local_inputs, 'trace_evidence': traces,
            'trace_comparisons': trace_comparisons(rows, run_root, conditions), 'baseline_observations': baselines,
            'cost_audit': cost_audit, 'execution': execution,
            'observed_metrics': record.get('metrics', {}), 'observed_phases': record.get('phases', {}),
            'report_generator': pin(__file__), 'scope': {'inference_only': True, 'unique_examples': 40,
                'training_updates': 0, 'original_failed_validation_preserved': True,
                'new_cl_accuracy': False, 'measurement_aggregate_eligible': False,
                'original_live_logits_available': False, 'checkpoint_issue_resolved': False,
                'causal_proof_of_duplicate_write_winner': False, 'broad_comparison': False,
                'prior_nine_method_report_changed': False, 'subsequent_training_authorized': False}}


def csv_text(report):
    rows = [{'kind': kind, **row} for kind, values in report['comparisons'].items() for row in values]
    if not rows:
        rows = [{'kind': 'incomplete_diagnostic', **report['summary']}]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def markdown(report):
    summary = report['summary']
    if report['status'] == 'failed':
        return ('# PROL checkpoint 추론 진단 중단\n\n'
                f'한 번의 승인된 프로세스에서 {len(report["matrix"])}개 평가 자료를 기록한 후 중단했습니다. '
                '40개 전체 행렬이 완료되지 않았으므로 완전한 A/B 비교나 원인 해결을 주장하지 않습니다.\n\n'
                '저장된 오류 문맥과 개별 관측을 보존했으며 학습·자동 재실행·구현 통과 판정 변경은 없습니다.\n')
    lines = ['# PROL checkpoint 추론 진단', '',
             '저장된 첫 태스크 checkpoint에서 새로 복원한 저자 모델 A와 compact clone B를 비교했습니다. '
             '각 모델에서 audit ON/OFF를 10회씩 평가하여 **40회 모델 평가·80회 encoder forward·20개 A/B 쌍**을 관측했습니다.', '',
             '입력은 기존 평가 마지막 **40개 이미지**이며 동일 이미지를 반복하여 1,600번 방문했습니다. '
             '학습·backward·optimizer update는 모두 0회입니다. 별도의 CL 정확도나 효율성 pilot 결과가 아닙니다.', '',
             '| 비교 | 전체 쌍 | bitexact 불일치 | logit 최대 절대 차이 |', '|---|---:|---:|---:|',
             f'| A/B | 20 | {summary["non_bitexact_reference_clone_comparisons"]} | {summary["reference_clone_max_abs_difference"]:.9g} |',
             f'| 같은 조건 반복 | 180 | {summary["non_bitexact_within_cell_comparisons"]} | {summary["within_cell_max_abs_difference"]:.9g} |',
             f'| audit ON/OFF | 20 | {summary["non_bitexact_hook_comparisons"]} | {summary["hook_max_abs_difference"]:.9g} |', '',
             f'관측 분류: **{summary["diagnosis"]}**.', '',
             '**이번 행렬에서 재현되지 않아도 원인이 해결되었다고 판정하지 않습니다.** '
             '원래 학습 프로세스의 probe logits/RNG가 저장되지 않았으므로 당시 live 객체를 그대로 재현할 수 없습니다. '
             'audit ON은 원래 hook 연산에 내부 trace와 CPU 복사를 추가해 실행 순서에 영향을 줄 수 있습니다. '
             'hook과 출력 변동의 관계는 CUDA 중복 쓰기 승자의 인과적 증명이 아닙니다.', '',
             '저장된 기존 40개 예측과의 일치 수는 진단 자료이며 새 태스크 정확도가 아닙니다. '
             '기존 checkpoint 검사와 구현 통과 판정은 변경하지 않았고, 후속 학습·재시도는 승인되지 않았습니다.', '',
             f'[원래 실패한 저자 실행]({report["historical_failed_source"]["wandb_url"]}) · '
             f'[이번 추론 진단]({report["source"]["wandb_url"]})']
    return '\n'.join(lines) + '\n'


def write_report(report, output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', report)
    (output / 'results.csv').write_text(csv_text(report))
    (output / 'RESULTS.md').write_text(markdown(report))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args(argv)
    report = build_report(options.run_root)
    if options.write:
        write_report(report, options.output)
    print(json.dumps({'status': report['status'], 'diagnosis': report['summary']['diagnosis'], 'written': options.write,
                      'checkpoint_issue_resolved': False, 'new_training_runs': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
