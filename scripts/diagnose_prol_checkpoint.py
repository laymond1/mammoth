"""Explicit inference-only diagnosis of the preserved PROL checkpoint failure.

Four conditions each receive ten model evaluations on the same forty images.
Each model evaluation runs exactly one query and one prompted encoder forward.
Imports and dry preparation start no model, GPU, logging service or experiment.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager, nullcontext
import copy
import functools
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prol_diagnostic_protocol as shared
from scripts import prol_protocol as base
from scripts import prol_reference_utils as reference
from scripts import run_prol_worker as training_worker
from scripts.cl_efficiency import Recorder, flatten, notify_slack, write_json
from scripts.cl_runtime import configure_native_environment


SMALL_FILES = ('spec.json', 'result.json', 'metrics.json', 'artifact_metrics.json', 'matrix.json',
               'restoration.json', 'input_evidence.json', 'forward_outputs.npz', 'telemetry.csv', 'phases.json',
               'error_context.json') + tuple(f'trace_{index:02d}.npz' for index in range(40))


def historical_callback():
    """Compile the exact old audit method; no replacement numerical operations."""
    path = 'scripts/run_prol_worker.py'
    source = subprocess.run(['git', 'show', f'{shared.ORIGIN_EXECUTION_COMMIT}:{path}'], cwd=ROOT,
                            capture_output=True, check=True).stdout.decode()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ProlAudit')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'prompt_output')
    namespace = {'reference': reference}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, f'git:{shared.ORIGIN_EXECUTION_COMMIT}:{path}', 'exec'), namespace)
    return namespace['prompt_output'], hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()


def tensor_hash(name, value):
    return reference.state_digest({name: value})


def rng_state(torch):
    result = {'cpu': torch.random.get_rng_state().clone()}
    if torch.cuda.is_initialized():
        result.update({f'cuda{index}': value.clone() for index, value in enumerate(torch.cuda.get_rng_state_all())})
    return result


def gradients_absent(branches):
    return all(parameter.grad is None for module in branches.values() for parameter in module.parameters())


@contextmanager
def forbid_training(torch, counts):
    """Fail before any unintended backward or optimizer execution."""
    def backward(*args, **kwargs):
        counts['backward_calls'] += 1
        raise RuntimeError('PROL diagnosis forbids backward computation')
    def optimizer(*args, **kwargs):
        counts['optimizer_updates'] += 1
        raise RuntimeError('PROL diagnosis forbids optimizer updates')
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, 'backward', backward))
        stack.enter_context(patch.object(torch.autograd, 'backward', backward))
        stack.enter_context(patch.object(torch.autograd, 'grad', backward))
        for name in ('Adam', 'AdamW', 'SGD'):
            stack.enter_context(patch.object(getattr(torch.optim, name), 'step', optimizer))
        yield


def reconstruct_tail(spec, prior):
    """Rebuild the exact old forty-image batch from pinned raw data and order."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from scripts.hide_protocol import validate_shared_data
    if validate_shared_data(spec['data_root']) != spec['data_files']:
        raise ValueError('Diagnostic shared raw CIFAR100 files changed')
    with np.load(spec['prior_predictions'], allow_pickle=False) as saved:
        ids = saved['evaluation_global_ids'][-40:].astype(np.int64)
        sorted_ids, sorted_labels, sorted_predictions = saved['global_ids'], saved['labels'], saved['predictions']
        if saved['batch_sizes'].tolist() != [64] * 15 + [40]:
            raise ValueError('Original PROL evaluation did not end in the approved forty-image batch')
        mapping = {int(index): (int(label), int(prediction)) for index, label, prediction in zip(sorted_ids, sorted_labels, sorted_predictions)}
    if len(ids) != 40 or len(set(map(int, ids))) != 40 or any(int(index) not in mapping for index in ids):
        raise ValueError('PROL diagnosis requires forty distinct original evaluation IDs')
    raw_path = Path(spec['data_root']) / 'CIFAR100/cifar-100-python/test'
    with raw_path.open('rb') as stream:
        raw = pickle.load(stream, encoding='latin1')
    data = np.asarray(raw['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    targets = np.asarray(raw['fine_labels'], dtype=np.int64)
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0, 0, 0], [1, 1, 1])])
    transform_digest = hashlib.sha256(repr(transform).encode()).hexdigest()
    expected = prior['data_snapshots']['task1_test']
    if transform_digest != expected['transform_repr_sha256'] or repr(transform) != expected['transform_repr']:
        raise ValueError('PROL diagnostic test preprocessing differs from the original task1 evaluation')
    labels = targets[ids]
    if not np.array_equal(labels, np.asarray([mapping[int(index)][0] for index in ids], dtype=np.int64)) or np.any(labels >= 10):
        raise ValueError('PROL diagnostic raw labels or task scope differ from the saved predictions')
    inputs = torch.stack([transform(Image.fromarray(data[int(index)])) for index in ids])
    original_predictions = np.asarray([mapping[int(index)][1] for index in ids], dtype=np.int64)
    evidence = {'examples': 40, 'global_ids': ids.tolist(), 'labels': labels.tolist(),
        'original_tail_predictions': original_predictions.tolist(), 'global_ids_sha256': tensor_hash('ids', torch.from_numpy(ids)),
        'input_sha256': tensor_hash('inputs', inputs), 'input_shape': list(inputs.shape), 'input_dtype': str(inputs.dtype),
        'transform_repr': repr(transform), 'transform_repr_sha256': transform_digest,
        'raw_test_sha256': spec['data_files']['files']['test']['sha256'],
        'source': 'Original task1 evaluation_global_ids final40, original order, pinned raw CIFAR100 test and exact deterministic transform',
        'original_live_logits_available': False,
        'original_last_routing': prior['routing_audits'][0]['batches'][-1]}
    return inputs, labels, ids, original_predictions, evidence


def restore_reference(branches, saved):
    """Restore every stored source learner and its task lifecycle, with no forward."""
    import torch
    expected = reference.learned_state(branches)
    state = saved.get('state', {})
    if (saved.get('task') != 1 or saved.get('seen_classes') != 10 or set(state) != set(expected)
            or reference.state_digest(state) != saved.get('learned_sha256')
            or any(value.shape != expected[name].shape or value.dtype != expected[name].dtype
                   or not bool(torch.isfinite(value).all()) for name, value in state.items())):
        raise ValueError('PROL diagnostic source checkpoint catalogue is invalid')
    for branch, module in branches.items():
        selected = {key[len(branch) + 1:]: value for key, value in state.items() if key.startswith(branch + '.')}
        loaded = module.load_state_dict(selected, strict=False)
        if loaded.unexpected_keys or any(branch + '.' + key in expected for key in loaded.missing_keys):
            raise ValueError('PROL diagnostic did not restore every persisted learner')
    required = {'old_num_k': 0, 'new_num_k': 1, 'old_num_c': 0, 'new_num_c': 10, 'kmax_list': [1], 'lmax_list': [10]}
    if saved.get('prompt_lifecycle') != required:
        raise ValueError('PROL diagnostic checkpoint task lifecycle differs from completed task1')
    for name, value in required.items():
        setattr(branches['prompt'].e_prompt, name, copy.deepcopy(value))
    branches['query'].requires_grad_(False).train(True)
    branches['prompt'].eval()
    for generators in (branches['prompt'].e_prompt.k_comp_gen, branches['prompt'].e_prompt.v_comp_gen):
        generators.requires_grad_(True)
    if reference.state_digest(reference.learned_state(branches)) != saved['learned_sha256']:
        raise ValueError('PROL diagnostic source learner restore changed stored values')


@contextmanager
def audited_forward(branches, recorder, record, torch, np, callback, audit_protocol):
    """Keep the old audit operations; retain trace references for later CPU copies."""
    audit = SimpleNamespace(record={'protocol': audit_protocol}, torch=torch, np=np,
        task=0, stage='evaluation', eval_test_task=0, eval_routes=[], routing_layer_count=0)
    trace = []
    module = branches['prompt'].e_prompt
    original = module.compute_att_over_prompt
    handle = module.register_forward_hook(functools.partial(callback, audit), with_kwargs=True)
    def compute(*args, **kwargs):
        # All captured tensors are actual source operands and outputs. Holding
        # references changes allocator lifetime and Python scheduling; there is
        # no recomputation, scatter substitution or extra GPU tracing kernel.
        pre = {name: getattr(module, name).detach() for name in ('scaler', 'scaler_v', 'shifter', 'shifter_v')}
        output = original(*args, **kwargs)
        post = {name: getattr(module, name).detach() for name in pre}
        trace.append({'layer': int(args[3]), 'similarity_new': args[4].detach(), 'indices': args[5].detach(),
                      'prefix': output.detach(), 'pre_affine': pre, 'post_affine': post})
        return output
    try:
        with patch.object(module, 'compute_att_over_prompt', compute):
            yield audit, trace
    finally:
        handle.remove()


def evaluate_pair(branches, inputs, counts):
    counts['query_forwards'] += 1
    features = branches['query'](inputs)['pre_logits']
    counts['prompt_forwards'] += 1
    output = branches['prompt'].forwardA1(inputs, None, task_id=-1, cls_features=features, train=False)
    counts['model_evaluations'] += 1
    counts['encoder_forward_calls'] += 2
    counts['diagnostic_image_visits'] += len(inputs)
    return output['logits'][:, :10], features


def collect_matrix(branches_by_side, inputs, labels, ids, original_predictions, output, recorder, record,
                   callback, audit_protocol=None):
    """Run exactly the approved forty evaluations; variability is a finding."""
    import numpy as np
    import torch
    output = Path(output)
    audit_protocol = base.protocol('validation') if audit_protocol is None else audit_protocol
    input_digest = tensor_hash('inputs', inputs)
    matrices, logits_arrays, query_arrays, prediction_arrays, rng_arrays = [], [], [], [], {}
    counts = record['counts']
    with forbid_training(torch, counts), torch.no_grad():
        for side in shared.MODEL_SIDES:
            branches = branches_by_side[side]
            branches['query'].train(True)
            branches['prompt'].eval()
            for mode in shared.HOOK_MODES:
                for repetition in range(1, 11):
                    index = len(matrices)
                    with recorder.phase('evidence_before'):
                        before_rng = rng_state(torch)
                        learned_before = reference.state_digest(reference.learned_state(branches))
                        frozen_before = training_worker.frozen_encoder_hashes(branches)
                        if not gradients_absent(branches):
                            raise RuntimeError('PROL diagnostic reference unexpectedly has gradients')
                    hook = audited_forward(branches, recorder, record, torch, np, callback, audit_protocol) if mode == 'on' else nullcontext((None, []))
                    with hook as (audit, trace):
                        # No RNG/state copies or extra GPU trace operations are
                        # performed inside an OFF evaluation's forward phase.
                        with recorder.phase('diagnostic_forward'):
                            logits, features = evaluate_pair(branches, inputs, counts)
                        after_rng = rng_state(torch)
                        with recorder.phase('evidence_after'):
                            learned_after = reference.state_digest(reference.learned_state(branches))
                            frozen_after = training_worker.frozen_encoder_hashes(branches)
                            cpu_logits = logits.detach().cpu()
                            cpu_features = features.detach().cpu()
                            predictions = cpu_logits.argmax(1)
                            incomplete = len(trace) != len(audit_protocol['prompt_layers']) if mode == 'on' else bool(trace)
                            if incomplete:
                                raise RuntimeError('PROL diagnostic internal trace has an incomplete layer scope')
                            if mode == 'on':
                                tensors = {}
                                for layer in trace:
                                    prefix = f'layer{layer["layer"]}_'
                                    for name in ('similarity_new', 'indices', 'prefix'):
                                        tensors[prefix + name] = layer[name].cpu().numpy()
                                    for when in ('pre_affine', 'post_affine'):
                                        for name, value in layer[when].items():
                                            tensors[prefix + when + '_' + name] = value.cpu().numpy()
                                np.savez_compressed(output / f'trace_{index:02d}.npz', **tensors)
                                del layer, value, tensors
                    row = {'row_index': index, 'branch': side, 'audit_mode': mode, 'audit': mode == 'on',
                        'cell_name': f'{side}_{mode}', 'repeat': repetition, 'examples': len(inputs),
                        'input_sha256': input_digest, 'rng_before_sha256': reference.state_digest(before_rng),
                        'rng_after_sha256': reference.state_digest(after_rng),
                        'learned_before_sha256': learned_before, 'learned_after_sha256': learned_after,
                        'frozen_before': frozen_before, 'frozen_after': frozen_after,
                        'rng_unchanged': reference.state_digest(before_rng) == reference.state_digest(after_rng),
                        'learned_unchanged': learned_before == learned_after, 'frozen_unchanged': frozen_before == frozen_after,
                        'logits_sha256': tensor_hash('logits', cpu_logits), 'predictions_sha256': tensor_hash('predictions', predictions),
                        'query_features_sha256': tensor_hash('query_features', cpu_features),
                        'finite': bool(torch.isfinite(cpu_logits).all() and torch.isfinite(cpu_features).all()),
                        'gradients_absent': gradients_absent(branches),
                        'original_tail_predictions_match': bool(np.array_equal(predictions.numpy(), original_predictions)),
                        'query_training_mode': branches['query'].training, 'prompt_training_mode': branches['prompt'].training,
                        'labels_provided': False, 'oracle_task_id': None}
                    if mode == 'on':
                        row['audit_routing'] = audit.eval_routes
                        row['active_prefix_layers_audited'] = audit.routing_layer_count
                        row['trace_file'] = f'trace_{index:02d}.npz'
                        row['trace_sha256'] = shared.file_hash(output / row['trace_file'])
                        row['trace_prefix_semantics'] = 'Raw generated prefix per layer; the source block adds the preceding raw prefix after layer0'
                    matrices.append(row)
                    logits_arrays.append(cpu_logits.numpy())
                    query_arrays.append(cpu_features.numpy())
                    prediction_arrays.append(predictions.numpy())
                    for when, state in (('before', before_rng), ('after', after_rng)):
                        for name, value in state.items():
                            rng_arrays.setdefault(f'rng_{name}_{when}', []).append(value.numpy())
                    record['matrix'] = matrices
                    write_json(output / 'matrix.json', matrices)
                    np.savez_compressed(output / 'forward_outputs.npz', logits=np.stack(logits_arrays),
                        query_features=np.stack(query_arrays), predictions=np.stack(prediction_arrays), global_ids=ids,
                        labels=labels, row_indices=np.arange(len(matrices), dtype=np.int64),
                        **{name: np.stack(values) for name, values in rng_arrays.items()})
                    del trace, logits, features, cpu_logits, cpu_features, before_rng, after_rng
    expected = {'train_steps': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'query_forwards': 40,
        'prompt_forwards': 40, 'model_evaluations': 40, 'encoder_forward_calls': 80, 'diagnostic_image_visits': 1600}
    if counts != expected or len(matrices) != 40:
        raise RuntimeError('PROL diagnostic performed a different workload than the approved matrix')
    return matrices


def upload_record(wandb, output, record):
    record['wandb_logging_status'] = 'uploaded'
    record['output_files'] = [{'name': name, 'sha256': shared.file_hash(output / name), 'bytes': (output / name).stat().st_size}
                              for name in SMALL_FILES if (output / name).is_file() and name not in ('result.json', 'metrics.json', 'artifact_metrics.json')]
    for name in ('result.json', 'metrics.json', 'artifact_metrics.json'):
        write_json(output / name, record)
    metrics = flatten({'counts': record['counts'], 'efficiency': record['metrics']})
    wandb.run.summary.update({**{key: value for key, value in metrics.items() if isinstance(value, (int, float, bool))},
        'diagnostic_status': record['status'], 'training_updates': 0, 'measurement_aggregate_eligible': False,
        'new_cl_accuracy': False})
    artifact = wandb.Artifact(record['name'], type='cl-diagnostic')
    for name in SMALL_FILES:
        if (output / name).is_file():
            artifact.add_file(str(output / name))
    published = wandb.run.log_artifact(artifact)
    published.wait()
    receipt = {'run_id': wandb.run.id, 'wandb_url': record['wandb_url'], 'artifact_id': published.id,
        'artifact_state': str(published.state), 'result_sha256': shared.file_hash(output / 'result.json')}
    wandb.finish(exit_code=0 if record['status'] == 'completed' else 1)
    write_json(output / 'wandb_receipt.json', receipt)


def validate_saved_record(record):
    """Check completed matrix scope without GPU/model construction or network."""
    shared.validate_spec(record)
    expected = {'train_steps': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'query_forwards': 40,
        'prompt_forwards': 40, 'model_evaluations': 40, 'encoder_forward_calls': 80, 'diagnostic_image_visits': 1600}
    if (record.get('status') != 'completed' or record.get('computation_status') != 'completed'
            or record.get('counts') != expected or record.get('precision') != shared.PRECISION
            or len(record.get('matrix', [])) != 40):
        raise ValueError('PROL diagnostic matrix workload or completion evidence is incomplete')
    rows = record['matrix']
    expected_cells = {(side, mode, repeat) for side in shared.MODEL_SIDES for mode in shared.HOOK_MODES for repeat in range(1, 11)}
    cells = {(row.get('branch'), row.get('audit_mode'), row.get('repeat')) for row in rows}
    if cells != expected_cells or any(row.get('row_index') != index for index, row in enumerate(rows)):
        raise ValueError('PROL diagnostic matrix conditions or repetition order changed')
    for row in rows:
        if (row.get('input_sha256') != record.get('input_sha256') or row.get('finite') is not True
                or row.get('gradients_absent') is not True or row.get('labels_provided') is not False
                or row.get('oracle_task_id') is not None or row.get('query_training_mode') is not True
                or row.get('prompt_training_mode') is not False):
            raise ValueError('PROL diagnostic inference boundaries or model modes changed')
        for prefix in ('rng', 'learned'):
            same = row.get(prefix + '_before_sha256') == row.get(prefix + '_after_sha256')
            if row.get(prefix + '_unchanged') is not same:
                raise ValueError('PROL diagnostic before/after equality flags disagree with their hashes')
        if row.get('frozen_unchanged') is not (row.get('frozen_before') == row.get('frozen_after')):
            raise ValueError('PROL diagnostic frozen state equality flag differs from the evidence')
    return True


def worker(spec_path):
    if os.getenv('CL_PROL_DIAGNOSTIC_EXECUTE') != '1' or os.getenv('CUDA_VISIBLE_DEVICES') != '1' or os.getenv('WANDB_MODE') != 'online':
        raise ValueError('PROL diagnostic requires explicit GPU1 inference-only execution and online W&B')
    configure_native_environment()
    spec = json.loads(Path(spec_path).read_text())
    shared.validate_spec(spec)
    prior = shared.verify_prior_inputs()
    if shared.execution_sources() != spec['execution_source_sha256']:
        raise ValueError('PROL diagnostic execution code changed after preparation')
    for path, expected in ((spec['checkpoint_file'], shared.CHECKPOINT_SHA256), (spec['prior_result'], spec['prior_result_sha256']),
        (spec['prior_predictions'], spec['prior_predictions_sha256']), (spec['prior_checkpoint'], spec['prior_checkpoint_sha256'])):
        if shared.file_hash(path) != expected:
            raise ValueError('PROL diagnostic input evidence changed')
    reference.verify_author_source(spec['reference_source'], spec['reference_source_sha256'])
    output = Path(spec['output'])
    recorder = Recorder(output, gpu='1')
    recorder.start()
    record = {**spec, 'status': 'failed', 'computation_status': 'failed', 'matrix': [],
        'counts': {'train_steps': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'query_forwards': 0,
                   'prompt_forwards': 0, 'model_evaluations': 0, 'encoder_forward_calls': 0, 'diagnostic_image_visits': 0}}
    wandb, exit_code = None, 1
    try:
        import numpy as np
        import torch
        import wandb
        record['precision'] = training_worker.configure_precision(torch)
        recorder.torch, recorder.device = torch, torch.device('cuda:0')
        record['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'wandb')}
        wandb.init(entity='laymond1', project='cl-efficiency', name=spec['name'], mode='online', group='prol-checkpoint-diagnostic',
            job_type='inference-only-matrix', config=spec, dir=str(output))
        record['wandb_url'] = wandb.run.url
        with recorder.phase('input_reconstruction'):
            inputs, labels, ids, original_predictions, evidence = reconstruct_tail(spec, prior)
            np.savez_compressed(output / 'tail_inputs.npz', inputs=inputs.numpy(), labels=labels, global_ids=ids)
            evidence['local_tail_inputs'] = {'name': 'tail_inputs.npz', 'sha256': shared.file_hash(output / 'tail_inputs.npz'),
                'bytes': (output / 'tail_inputs.npz').stat().st_size, 'upload': False}
            evidence['labels_sha256'] = tensor_hash('labels', torch.from_numpy(labels))
            record['input_sha256'] = evidence['input_sha256']
            record['input_evidence'] = evidence
            write_json(output / 'input_evidence.json', evidence)
        with recorder.phase('initialize_and_restore'):
            code = reference.load_author_code(spec['reference_source'], spec['reference_source_sha256'])
            torch.manual_seed(1993)
            branches = reference.build_author_branches(code, spec['checkpoint_file'])
            if any(value != shared.CANONICAL_PRETRAINED_SHA256 for value in training_worker.canonical_hashes(branches).values()):
                raise ValueError('PROL diagnostic initial canonical encoders differ from the agreed NPZ')
            saved = torch.load(spec['prior_checkpoint'], map_location='cpu', weights_only=True)
            restore_reference(branches, saved)
            for module in branches.values():
                module.to(recorder.device)
            rng = training_worker.rng_digest(torch)
            clones, _ = training_worker.compact_clone(branches, saved)
            frozen = training_worker.frozen_encoder_hashes(branches)
            if (frozen != saved['frozen_encoder'] or training_worker.frozen_encoder_hashes(clones) != frozen
                    or reference.state_digest(reference.learned_state(clones)) != saved['learned_sha256']
                    or training_worker.rng_digest(torch) != rng):
                raise ValueError('PROL diagnostic frozen/learned catalogue or compact clone RNG restoration failed')
            callback, callback_sha = historical_callback()
            restoration = {'saved_learned_sha256': saved['learned_sha256'], 'author_restored_learned_sha256': reference.state_digest(reference.learned_state(branches)),
                'compact_clone_learned_sha256': reference.state_digest(reference.learned_state(clones)),
                'frozen_encoder': frozen, 'checkpoint_digest_matches': True, 'all_saved_learners_restored': True,
                'clone_rng_unchanged': True, 'query_initial_canonical_matches_pin': True, 'prompt_initial_canonical_matches_pin': True,
                'prompt_trainable_norm_changes_expected': True, 'historical_audit_callback_ast_sha256': callback_sha,
                'historical_audit_callback_commit': shared.ORIGIN_EXECUTION_COMMIT,
                'prompt_lifecycle': saved['prompt_lifecycle'], 'learner_catalogue_tensor_count': len(saved['state']),
                'saved_learned_tensor_count': len(saved['state']),
                'author_restored_learned_tensor_count': len(reference.learned_state(branches)),
                'compact_clone_learned_tensor_count': len(reference.learned_state(clones)),
                'author_restored_prompt_lifecycle': {name: copy.deepcopy(getattr(branches['prompt'].e_prompt, name)) for name in saved['prompt_lifecycle']},
                'compact_clone_prompt_lifecycle': {name: copy.deepcopy(getattr(clones['prompt'].e_prompt, name)) for name in saved['prompt_lifecycle']},
                'clone_rng_before_sha256': rng, 'clone_rng_after_sha256': training_worker.rng_digest(torch),
                'reference_meaning': 'Author reference A is reconstructed from saved checkpoint in a new process; original live outputs unavailable'}
            record['restoration'] = restoration
            write_json(output / 'restoration.json', restoration)
            inputs = inputs.to(recorder.device)
            # This is a selected diagnostic seed. The old process's final RNG
            # bytes were not saved and are not reconstructed by this reset.
            matrix_seed = base.seed_phase(1993, 0, 'train')
            record['matrix_seed'] = matrix_seed
            record['matrix_rng_initial_sha256'] = training_worker.rng_digest(torch)
        collect_matrix({'author_restored': branches, 'compact_clone': clones}, inputs, labels, ids,
                       original_predictions, output, recorder, record, callback)
        record.update(status='completed', computation_status='completed')
        validate_saved_record(record)
        exit_code = 0
    except BaseException as error:
        record.update(status='failed', computation_status='failed', error_type=type(error).__name__,
                      error='PROL diagnostic failed; inspect sanitized local evidence')
        record['error_context'] = training_worker.error_context(error, recorder, None)
        write_json(output / 'error_context.json', record['error_context'])
        logging.error('PROL diagnostic failed: %s', type(error).__name__)
    finally:
        record = recorder.save(record)
        record['measurement_notes']['scope'] = 'Inference-only matrix diagnostic; initialization, state hashing, ON trace retention and all reporting preparation included; no CL training or efficiency pilot'
        record['measurement_notes']['observer_effect'] = 'ON preserves exact historical audit callback; additional source-operand references change allocator lifetime and Python scheduling. CPU materialization occurs after forward. OFF has no internal hooks or wrappers.'
        record['measurement_notes']['historical_limit'] = 'Original before/after logits and RNG distinction were not saved; current reconstructed A/B evidence cannot establish original process outputs'
        write_json(output / 'result.json', record)
        if wandb is not None and wandb.run is not None:
            try:
                upload_record(wandb, output, record)
            except Exception as error:
                record.update(status='logging_failed', error_type=type(error).__name__, wandb_logging_status='failed')
                write_json(output / 'result.json', record)
                exit_code = 1
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    options = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Diagnostic termination requested')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return worker(options.worker)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
