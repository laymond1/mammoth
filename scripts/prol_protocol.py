"""PROL CPU preparation and proposed GPU workloads; importing executes no work."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/cl_efficiency/prol_prompt/validation_plan.json'
REVIEW = ROOT / 'benchmarks/cl_efficiency/prol_prompt/source_review.json'
REFERENCE_SOURCE = Path('/tmp/cl-prol-official-20261008')
REFERENCE_COMMIT = 'bfff8418a4f603a24ae578f1e108bfac89af1e18'
JOB_NAMES = ('validation_author', 'validation_native', 'measurement1', 'measurement2', 'measurement3', 'flops')
METHOD_SETTINGS = {
    'source_mode': 'official-active-path', 'paper_equivalent': False,
    'n_epochs_per_task': 1, 'prompt_length': 5, 'prompt_layers': list(range(7)),
    'top_k_l': 1, 'prefix_tuning': True, 'optimizer': 'adam', 'weight_decay': 0.,
    'optimizer_eps': 1e-8, 'optimizer_betas': [0.9, 0.999], 'clip_grad_norm': 1.,
    'nominal_lr': .01, 'effective_lr': .0025, 'lr_scaling': 'nominal_lr * batch_size / 256',
    'later_task_norm_lr_ratio': .0005, 'later_task_initial_norm_lr': .00000125,
    'intertask_coeff': .001, 'generalization_coeff': .1,
    'pull_constraint_coeff': 1., 'pull_constraint_coeff2': 1.,
    'loss_threshold': .3, 'threshold_operand': 'current_task_CE + intertask_coeff * seen_class_CE',
    'batch_cosine_T_max': 20, 'batch_cosine_eta_min': .0005,
    'batch_cosine_trigger': 'sticky within epoch once threshold operand < loss_threshold',
    'epoch_cosine_T_max': 1, 'epoch_cosine_eta_min': 0.,
    'generator_freeze': 'beginning of second task', 'old_heads_frozen': True,
    'backbone_norms_trainable': True, 'query_frozen': True,
    'source_non_diagonal_formula': 'GM - GM @ I',
    'source_cross_batch_similarity_assignment': True,
    'source_current_task_scaler_index_local': True, 'source_eval_clamps_scaler_shifter': True,
    'class_il_evaluation': 'seen classes only; no task oracle',
}
REQUIRED_CHECKS = (
    'finite_loss_and_gradients', 'frozen_non_norm_encoder_gradients_absent',
    'frozen_non_norm_encoder_unchanged', 'query_frozen_and_unchanged',
    'allowed_trainable_norms_match_source', 'canonical_pretrained_matches_pin',
    'initial_learned_state_and_phase_rng_match', 'dataset_transforms_ids_match',
    'first_actual_batch_inputs_gradients_lr_match', 'complete_stream_and_optimizer_updates',
    'generator_frozen_from_task2', 'previous_classifier_heads_frozen',
    'source_routing_and_clamps_preserved', 'scheduler_trigger_and_all_groups_match',
    'checkpoint_round_trip', 'seen_class_counts_and_predictions_match', 'no_future_or_replay_samples',
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_reference_source(source=REFERENCE_SOURCE):
    source = Path(source).resolve()
    expected = json.loads(REVIEW.read_text())['reference_source_sha256']
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, check=True,
                          capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=source, check=True,
                             capture_output=True).stdout.decode().split('\0')
    tracked = [name for name in tracked if name]
    actual = {name: file_hash(source / name) for name in tracked}
    if head != REFERENCE_COMMIT or actual != expected:
        raise ValueError('Official PROL source differs from the pinned reviewed snapshot')
    return actual


def load_preparation_plan(path=PLAN):
    plan = json.loads(Path(path).read_text())
    expected = {'schema_version': 1, 'method': 'prol-prompt', 'source_mode': 'official-active-path',
                'training_authorized': False, 'experiment_executed': False,
                'maximum_proposed_gpu_runs': 6, 'reference_commit': REFERENCE_COMMIT}
    if (any(type(plan.get(key)) is not type(value) or plan[key] != value for key, value in expected.items())
            or plan.get('method_settings') != METHOD_SETTINGS or plan.get('job_names') != list(JOB_NAMES)
            or plan.get('validation', {}).get('required_structural_checks') != list(REQUIRED_CHECKS)):
        raise ValueError('PROL preparation must remain an unexecuted, unapproved six-run proposal')
    common = plan.get('common', {})
    required = {'dataset': 'seq-cifar100-224', 'dataset_config': 'ease', 'dataset_total_tasks': 10,
                'classes_per_task': 10, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                'vit_size': 'base', 'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'}
    if any(type(common.get(key)) is not type(value) or common[key] != value for key, value in required.items()):
        raise ValueError('PROL preparation differs from the selected common conditions')
    if common.get('class_order') != list(range(100)):
        raise ValueError('PROL preparation requires the selected original class order')
    return plan


def native_command(plan, tasks):
    if tasks not in (1, 2):
        raise ValueError('PROL proposed scope is first one or two tasks')
    common, method = plan['common'], plan['method_settings']
    return ['uv', 'run', '--no-sync', '--offline', 'python', 'main.py',
            '--model', 'prol-prompt', '--model_config', 'default', '--dataset', common['dataset'],
            '--dataset_config', common['dataset_config'], '--backbone', 'prol-vit',
            '--prol_checkpoint', common['checkpoint']['file'], '--prol_vit_size', 'base',
            '--batch_size', '64', '--num_workers', '4', '--seed', '1993', '--n_epochs', '1',
            '--lr', str(method['nominal_lr']), '--stop_after', str(tasks), '--permute_classes', '0',
            '--enable_other_metrics', '0', '--nowand', '1', '--non_verbose', '1']
