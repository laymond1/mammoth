"""PROL CPU preparation and proposed GPU workloads; importing executes no work."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/cl_efficiency/prol_prompt/validation_plan.json'
RUN_PLAN = ROOT / 'benchmarks/cl_efficiency/prol_prompt/run_plan.json'
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
PREPARATION_REQUIRED_CHECKS = REQUIRED_CHECKS
REQUIRED_CHECKS = tuple('first_actual_batch_inputs_gradients_lr_audited' if name ==
                        'first_actual_batch_inputs_gradients_lr_match' else name for name in REQUIRED_CHECKS)
MODES = ('validation', 'efficiency')
_CHECKPOINT_FILE = 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
CHECKPOINT_SHA256 = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
CANONICAL_PRETRAINED_SHA256 = 'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300'
PRECISION = {'parameter_dtype': 'float32', 'autocast': False, 'matmul_precision': 'highest',
             'tf32_allowed': False, 'cudnn_tf32_allowed': True, 'cudnn_deterministic': True,
             'cudnn_benchmark': False, 'torch_cpu_threads': 2}


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
            or plan.get('validation', {}).get('required_structural_checks') != list(PREPARATION_REQUIRED_CHECKS)):
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


def _default_data_root():
    from scripts.cl_runtime import public_setting
    return public_setting('data_root')


def _default_checkpoint():
    # Resolved when used: importing this module must not load scripts.cl_runtime.
    from scripts.cl_runtime import checkpoint_path
    return checkpoint_path(_CHECKPOINT_FILE)


def __getattr__(name):
    if name == 'CHECKPOINT':
        return _default_checkpoint()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def native_command(plan, tasks=None, output=None):
    # Preserve the historical preparation API; a worker specification uses the
    # second path below, with explicit shared data/device and zero extra work.
    if 'mode' in plan:
        settings = protocol(plan['mode'])
        command = ['--model', 'prol-prompt', '--model_config', 'default', '--dataset', 'seq-cifar100-224',
                   '--dataset_config', 'ease', '--backbone', 'prol-vit', '--prol_vit_size', 'base',
                   '--prol_checkpoint', str(plan.get('checkpoint_file', _default_checkpoint())),
                   '--base_path', str(plan.get('data_root') or _default_data_root()), '--device', 'cuda:0',
                   '--n_epochs', '1', '--stop_after', str(settings['tasks']), '--batch_size', '64', '--num_workers', '4',
                   '--seed', '1993', '--permute_classes', '0', '--validation', '0', '--lr', '0.01',
                   '--optimizer', 'adam', '--optim_wd', '0', '--enable_other_metrics', '0', '--eval_future', '0',
                   '--debug_mode', '0', '--non_verbose', '1', '--save_after_interrupt', '0', '--joint', '0',
                   '--disable_log', '0', '-O', '0']
        destination = output if output is not None else tasks
        if destination is not None:
            command.extend(['--results_path', str(Path(destination) / 'mammoth')])
        return command
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


def protocol(mode='validation'):
    if mode not in MODES:
        raise ValueError('Unknown PROL workload')
    return {'method': 'prol-prompt', 'source_mode': 'official-active-path', 'paper_equivalent': False,
            'dataset': 'seq-cifar100-224', 'dataset_config': 'ease', 'dataset_total_tasks': 10,
            'classes_per_task': 10, 'class_order': list(range(100)), 'tasks': 2 if mode == 'validation' else 1,
            'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993, 'vit_size': 'base',
            'prompt_length': 5, 'prompt_layers': list(range(7)), 'top_k_l': 1,
            'optimizer': 'adam', 'nominal_lr': .01, 'effective_lr': .0025, 'weight_decay': 0., 'clip_grad': 1.,
            'norm_lr_ratio': .0005, 'loss_threshold': .3, 'cos_tmax': 20, 'cos_eta_min': .0005,
            'intertask_coeff': .001, 'pull_coeff': 1., 'pull_coeff2': 1.,
            'backward_retain_graph': False, 'seen_class_evaluation': True,
            'trainable_encoder_norms': True, 'generator_freezes_task2': True,
            'eval_affine_clamps': True, 'source_cross_batch_similarity_assignment': True,
            'measurement_aggregate_eligible': mode == 'efficiency'}


def load_run_plan():
    plan = json.loads(RUN_PLAN.read_text())
    expected = {'training_authorized': True, 'method': 'prol-prompt', 'source_mode': 'official-active-path',
                'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
                'maximum_new_gpu_runs': 6, 'measurement_repeats': 3, 'flops_runs': 1,
                'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.,
                'first_task_minimum_accuracy_pct': 90., 'task2_collapse_floor_pct': 20.,
                'normality_review_required': True, 'automatic_retries_authorized': False,
                'automatic_lr_changes_authorized': False, 'paper_correction_authorized': False}
    if (any(type(plan.get(key)) is not type(value) or plan[key] != value for key, value in expected.items())
            or plan.get('protocols') != {mode: protocol(mode) for mode in MODES}
            or plan.get('job_names') != list(JOB_NAMES) or plan.get('method_settings') != METHOD_SETTINGS
            or plan.get('required_structural_checks') != list(REQUIRED_CHECKS)
            or plan.get('common') != load_preparation_plan()['common']):
        raise ValueError('PROL execution differs from the approved six-run workloads')
    return plan


def validate_spec(spec):
    mode, side, kind = spec.get('mode'), spec.get('side'), spec.get('pass')
    if (mode not in MODES or side not in ('author', 'native') or kind not in ('validation', 'measurement', 'flops')
            or mode != 'efficiency' and kind != 'validation' or mode == 'efficiency' and (side != 'native' or kind == 'validation')
            or spec.get('protocol') != protocol(mode) or spec.get('method') != 'prol-prompt'
            or spec.get('gpu') != '1' or spec.get('wandb_entity') != 'laymond1' or spec.get('wandb_project') != 'cl-efficiency'
            or spec.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or spec.get('canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256
            or spec.get('reference_commit') != REFERENCE_COMMIT or spec.get('training_authorized') is not True
            or spec.get('source_mode') != 'official-active-path' or spec.get('paper_equivalent') is not False
            or spec.get('measurement_aggregate_eligible') is not (kind == 'measurement')):
        raise ValueError('PROL worker specification exceeds the authorized workloads')
    return True


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(tensor_digest(value).encode() if hasattr(value, 'detach') else
                      json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return result.hexdigest()


def execution_sources():
    names = ('scripts/prol_protocol.py', 'scripts/run_prol_worker.py', 'scripts/prol_reference_utils.py',
             'scripts/run_prol_suite.py', 'scripts/cl_efficiency.py', 'scripts/cl_runtime.py',
             'scripts/hide_protocol.py', 'scripts/low_rank_protocol.py', 'scripts/ease_reference_utils.py',
             'scripts/ease_size_protocol.py', 'scripts/report_prol_validation.py', 'scripts/report_hide_results.py',
             'scripts/publish_prol_validation.py', 'scripts/benchmark_cl.py', 'scripts/run_peft_efficiency.py',
             'scripts/run_low_rank_efficiency.py', 'scripts/publish_low_rank_results.py',
             'scripts/publish_peft_partial.py', 'scripts/report_low_rank_results.py',
             'models/prol_prompt.py', 'models/config/prol_prompt.yaml', 'backbone/prol_vit.py',
             'main.py', 'utils/training.py', 'utils/evaluate.py', 'models/utils/continual_model.py',
             'datasets/utils/continual_dataset.py', 'datasets/seq_cifar100_224.py', 'datasets/seq_cifar100.py',
             'datasets/configs/seq-cifar100-224/ease.yaml', 'benchmarks/cl_efficiency/prol_prompt/run_plan.json',
             'benchmarks/cl_efficiency/prol_prompt/source_review.json', 'benchmarks/cl_efficiency/ease/validation_plan.json',
             'benchmarks/cl_efficiency/ease/execution_results.json', 'benchmarks/cl_efficiency/ease/native_validation_results.json', 'uv.lock')
    names = set(names) | {str(path.relative_to(ROOT)) for path in (ROOT / 'models/prol_prompt_utils').glob('*.py')}
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def phase_seed(seed, task_id, phase, epoch=0):
    from scripts.low_rank_protocol import phase_seed as choose
    return choose(seed, task_id, phase, epoch)


def seed_phase(seed, task_id, phase, epoch=0):
    from scripts.low_rank_protocol import seed_phase as apply
    return apply(seed, task_id, phase, epoch)


def configure_epoch_loader(loader, seed, task_id, phase, epoch=0):
    from scripts.low_rank_protocol import configure_epoch_loader as configure
    return configure(loader, seed, task_id, phase, epoch)
