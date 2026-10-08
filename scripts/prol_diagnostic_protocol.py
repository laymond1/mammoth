"""Pinned inference-only PROL checkpoint diagnosis; importing starts no work."""

import hashlib
import json
from pathlib import Path
import subprocess

from scripts import prol_protocol as base

ROOT = base.ROOT
PLAN = ROOT / 'benchmarks/cl_efficiency/prol_diagnostic/run_plan.json'
DEFAULT_OUTPUT = ROOT / 'results/prol_diagnostic_20261008'
ORIGIN_EXECUTION_COMMIT = '697036abe847c0beb269675d336eb8b7251f102e'
PRIOR_DIRECTORY = ROOT / 'results/prol_validation_20261008/validation_author'
PRIOR_PINS = {
    'result.json': 'f077b107bbec8f1d26ec7d0b85680c00a499cc360cb4bcad8a233a03c2568630',
    'task1_predictions.npz': '8afda9bbc6859863811de0e86c4280403eab5ad4f8930905518137f24ff4c797',
    'task1_state.pt': '331407ee874b305702164e98c7530220c5f5ecac55d7eb055bfb4fb94bfed3e4',
}
CHECKPOINT = base.CHECKPOINT
CHECKPOINT_SHA256 = base.CHECKPOINT_SHA256
CANONICAL_PRETRAINED_SHA256 = base.CANONICAL_PRETRAINED_SHA256
REFERENCE_SOURCE = base.REFERENCE_SOURCE
REFERENCE_COMMIT = base.REFERENCE_COMMIT
GPU_UUID = 'GPU-1fb8c05f-24e7-75d1-5cf9-0f47df8e46ad'
PRECISION = base.PRECISION
MODEL_SIDES = ('author_restored', 'compact_clone')
HOOK_MODES = ('off', 'on')
CELL_NAMES = tuple(f'{side}_{hook}' for side in MODEL_SIDES for hook in HOOK_MODES)
file_hash = base.file_hash
tensor_digest = base.tensor_digest
state_digest = base.state_digest
verify_reference_source = base.verify_reference_source


def strict_equal(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(strict_equal(value[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(strict_equal(left, right) for left, right in zip(value, expected))
    return value == expected


def protocol():
    return {'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic', 'source_mode': 'official-active-path',
            'dataset': 'seq-cifar100-224', 'dataset_config': 'ease', 'vit_size': 'base',
            'pretrained_family': 'AugReg IN21K -> IN1K fine-tuned', 'seen_classes': 10,
            'prior_task': 1, 'original_training_batch_size': 64, 'batch_size': 40, 'examples_per_evaluation': 40,
            'prompt_length': 5, 'prompt_layers': list(range(7)), 'top_k_l': 1,
            'input_selection': 'original task1 evaluation_global_ids final40 in original order',
            'model_sides': list(MODEL_SIDES), 'hook_modes': list(HOOK_MODES),
            'repetitions_per_cell': 10, 'model_evaluations': 40, 'encoder_forward_calls': 80,
            'paired_reference_clone_comparisons': 20, 'diagnostic_image_visits': 1600,
            'unique_examples': 40, 'seed': 1993, 'matrix_seed': 2993,
            'rng_seed_strategy': 'set phase-derived diagnostic seed2993 once before matrix; original probe RNG unavailable',
            'query_training_mode': True, 'prompt_training_mode': False,
            'gradient_enabled': False, 'autocast': False, 'precision': dict(PRECISION),
            'optimizer_calls': 0, 'backward_calls': 0, 'training_updates': 0, 'training_examples': 0,
            'extra_model_evaluations': 0, 'task_oracle': False, 'labels_provided_to_model': False,
            'source_eval_affine_clamp_preserved': True, 'source_cross_batch_assignment_preserved': True,
            'measurement_aggregate_eligible': False, 'new_cl_accuracy': False,
            'original_live_logits_available': False, 'root_cause_claim_without_evidence': False}


def load_plan():
    plan = json.loads(PLAN.read_text())
    expected = {'schema_version': 1, 'method': 'prol-prompt', 'training_authorized': False,
                'diagnostic_authorized': True, 'maximum_new_gpu_processes': 1, 'gpu': '1',
                'gpu_uuid': GPU_UUID, 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
                'automatic_retries_authorized': False, 'subsequent_training_authorized': False,
                'gate_changes_authorized': False, 'paper_correction_authorized': False,
                'origin_execution_commit': ORIGIN_EXECUTION_COMMIT, 'protocol': protocol(),
                'prior_pins': PRIOR_PINS}
    if any(not strict_equal(plan.get(key), value) for key, value in expected.items()):
        raise ValueError('PROL diagnosis differs from the approved inference-only scope')
    return plan


def historical_sources(record):
    """Verify the old run's immutable Git bytes even after approved new edits."""
    commit = record.get('git_commit')
    expected = record.get('execution_source_sha256', {})
    if commit != ORIGIN_EXECUTION_COMMIT or not isinstance(expected, dict) or not expected:
        raise ValueError('Original PROL execution provenance is missing')
    verified = {}
    for name, digest in expected.items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Original PROL source path is invalid')
        saved = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT, capture_output=True, check=True).stdout
        actual = hashlib.sha256(saved).hexdigest()
        if actual != digest:
            raise ValueError('Original PROL Git source fingerprint changed')
        verified[name] = actual
    return verified


def verify_prior_inputs():
    for name, digest in PRIOR_PINS.items():
        if file_hash(PRIOR_DIRECTORY / name) != digest:
            raise ValueError('The original PROL failed-run evidence changed')
    record = json.loads((PRIOR_DIRECTORY / 'result.json').read_text())
    if (record.get('status') != 'failed' or record.get('side') != 'author'
            or record.get('wandb_url') != 'https://wandb.ai/laymond1/cl-efficiency/runs/2qvxzkht'
            or len(record.get('task_results', [])) != 1 or record['task_results'][0].get('seen_correct') != 993
            or record.get('counts', {}).get('optimizer_updates') != 79):
        raise ValueError('The diagnostic input is not the approved failed task1 record')
    historical_sources(record)
    return record


def validate_spec(spec):
    expected = {'schema_version': 1, 'method': 'prol-prompt', 'stage': 'checkpoint-diagnostic',
                'training_authorized': False, 'diagnostic_authorized': True,
                'source_mode': 'official-active-path', 'paper_equivalent': False,
                'gpu': '1', 'gpu_uuid': GPU_UUID, 'wandb_mode': 'online',
                'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'protocol': protocol(),
                'checkpoint_sha256': CHECKPOINT_SHA256,
                'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
                'reference_commit': REFERENCE_COMMIT, 'origin_execution_commit': ORIGIN_EXECUTION_COMMIT,
                'prior_result_sha256': PRIOR_PINS['result.json'],
                'prior_predictions_sha256': PRIOR_PINS['task1_predictions.npz'],
                'prior_checkpoint_sha256': PRIOR_PINS['task1_state.pt'],
                'measurement_aggregate_eligible': False}
    if any(not strict_equal(spec.get(key), value) for key, value in expected.items()):
        raise ValueError('PROL diagnostic specification exceeds authorization')
    return True


def execution_sources():
    pins = base.execution_sources()
    for name in ('scripts/prol_diagnostic_protocol.py', 'scripts/run_prol_diagnostic.py',
                 'scripts/diagnose_prol_checkpoint.py', 'scripts/report_prol_diagnostic.py',
                 'scripts/publish_prol_diagnostic.py', 'benchmarks/cl_efficiency/prol_diagnostic/run_plan.json'):
        pins[name] = file_hash(ROOT / name)
    return dict(sorted(pins.items()))
