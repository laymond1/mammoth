"""Pinned RainbowPrompt validation and pilot workloads; importing starts no work."""

import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cl_runtime import checkpoint_path, public_setting


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/run_plan.json'
REVIEW = ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/source_review.json'
REFERENCE_SOURCE = Path('/tmp/cl-rainbow-official-20261008')
REFERENCE_COMMIT = '0b84d3e3738669fe4bb516ccc1c526a535576ad3'
CHECKPOINT = checkpoint_path('B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
CHECKPOINT_SHA256 = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
CANONICAL_PRETRAINED_SHA256 = 'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300'
MODES = ('full_validation', 'short_validation', 'efficiency')
REQUIRED_CHECKS = ('finite_loss_and_gradients', 'frozen_backbone_gradients_absent', 'frozen_backbone_unchanged',
    'canonical_pretrained_matches_pin', 'learnable_initialization_matches', 'dataset_matches_common_reference',
    'training_protocol_complete', 'evaluation_predictions_match_accuracy', 'prediction_ids_and_counts_complete',
    'method_state_valid', 'no_future_or_replay_samples', 'optimizer_updates_complete', 'checkpoint_round_trip',
    'stored_prompt_cache_populated_after_warmup', 'stored_prompt_cache_timing_matches',
    'unknown_task_batch_routing_preserved', 'phase_rng_matches_protocol')
PRECISION = {'parameter_dtype': 'float32', 'autocast': False, 'matmul_precision': 'highest',
             'tf32_allowed': False, 'cudnn_tf32_allowed': True, 'cudnn_deterministic': True,
             'cudnn_benchmark': False, 'torch_cpu_threads': 2}
JOB_NAMES = ('full_author', 'full_native', 'short_author', 'short_native',
             'measurement1', 'measurement2', 'measurement3', 'flops')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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
        if hasattr(value, 'detach'):
            result.update(tensor_digest(value).encode())
        else:
            result.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return result.hexdigest()


def protocol(mode='full_validation'):
    if mode not in MODES:
        raise ValueError('Unknown RainbowPrompt workload')
    return {'method': 'rainbow-prompt', 'source_mode': 'official-active-path', 'learned_gate_active': False,
            'paper_equivalent': False, 'dataset': 'seq-cifar100-224', 'dataset_config': 'ease',
            'dataset_total_tasks': 10, 'classes_per_task': 10, 'tasks': 2 if mode == 'full_validation' else 1,
            'n_epochs': 20 if mode == 'full_validation' else 1,
            'warmup_epochs': 5 if mode == 'full_validation' else 0,
            'batch_size': 64, 'num_workers': 4, 'seed': 1993, 'vit_size': 'base',
            'prompt_length': 20, 'pool_size': 10, 'D1': 56, 'D2': 96, 'KI_iter': 10,
            'prompt_layers': list(range(12)), 'self_attn_layers': list(range(6)),
            'balancing': .01, 'optimizer': 'adam', 'nominal_lr': .03, 'effective_lr': .0075,
            'weight_decay': 0., 'clip_grad': 1., 'backward_retain_graph': True,
            'seen_class_evaluation': True, 'batch_mean_routing': True,
            'measurement_aggregate_eligible': mode == 'efficiency'}


def load_run_plan():
    plan = json.loads(PLAN.read_text())
    expected = {'training_authorized': True, 'method': 'rainbow-prompt', 'source_mode': 'official-active-path',
                'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
                'pilot_recipe': 'one_epoch_warmup0', 'maximum_new_gpu_runs': 8,
                'measurement_repeats': 3, 'flops_runs': 1, 'pass_max_drop_pp': 1.5,
                'review_max_drop_pp': 3.0, 'first_task_minimum_accuracy_pct': 90.0,
                'task2_collapse_floor_pct': 20.0, 'gate_restore_authorized': False}
    if (any(type(plan.get(key)) is not type(value) or plan[key] != value for key, value in expected.items())
            or plan.get('protocols') != {mode: protocol(mode) for mode in MODES}
            or plan.get('job_names') != list(JOB_NAMES)):
        raise ValueError('RainbowPrompt execution differs from the approved eight-run workloads')
    return plan


def verify_reference_source(source=REFERENCE_SOURCE):
    source = Path(source).resolve()
    expected = json.loads(REVIEW.read_text())['reference_source_sha256']
    actual = {name: file_hash(source / name) for name in expected}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    if head != REFERENCE_COMMIT or actual != expected:
        raise ValueError('Official RainbowPrompt source differs from the pinned reviewed snapshot')
    return actual


def execution_sources():
    names = ('scripts/rainbow_protocol.py', 'scripts/run_rainbow_worker.py', 'scripts/rainbow_reference_utils.py',
             'scripts/run_rainbow_suite.py', 'scripts/cl_efficiency.py', 'scripts/cl_runtime.py',
             'scripts/hide_protocol.py', 'scripts/low_rank_protocol.py', 'scripts/ease_reference_utils.py',
             'scripts/ease_size_protocol.py', 'scripts/report_rainbow_validation.py',
             'scripts/report_hide_results.py', 'scripts/benchmark_cl.py',
             'scripts/run_low_rank_efficiency.py', 'scripts/run_peft_efficiency.py',
             'models/rainbow_prompt.py', 'models/config/rainbow_prompt.yaml', 'backbone/rainbow_vit.py',
             'models/hide_prompt_utils/vit.py', 'main.py', 'utils/training.py', 'utils/evaluate.py',
             'models/utils/continual_model.py', 'datasets/utils/continual_dataset.py',
             'datasets/seq_cifar100_224.py', 'datasets/seq_cifar100.py',
             'datasets/configs/seq-cifar100-224/ease.yaml',
             'benchmarks/cl_efficiency/rainbow_prompt/run_plan.json',
             'benchmarks/cl_efficiency/rainbow_prompt/validation_plan.json',
             'benchmarks/cl_efficiency/rainbow_prompt/source_review.json',
             'benchmarks/cl_efficiency/ease/validation_plan.json',
             'benchmarks/cl_efficiency/ease/execution_results.json',
             'benchmarks/cl_efficiency/ease/native_validation_results.json', 'uv.lock')
    names = set(names) | {str(path.relative_to(ROOT)) for path in (ROOT / 'models/rainbow_prompt_utils').glob('*.py')}
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def validate_spec(spec):
    mode, side, kind = spec.get('mode'), spec.get('side'), spec.get('pass')
    if (mode not in MODES or side not in ('author', 'native') or kind not in ('validation', 'measurement', 'flops')
            or mode != 'efficiency' and kind != 'validation' or mode == 'efficiency' and (side != 'native' or kind == 'validation')
            or spec.get('protocol') != protocol(mode) or spec.get('method') != 'rainbow-prompt'
            or spec.get('gpu') != '1' or spec.get('wandb_entity') != 'laymond1' or spec.get('wandb_project') != 'cl-efficiency'
            or spec.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or spec.get('canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256
            or spec.get('reference_commit') != REFERENCE_COMMIT or spec.get('training_authorized') is not True
            or spec.get('source_mode') != 'official-active-path' or spec.get('learned_gate_active') is not False
            or spec.get('paper_equivalent') is not False
            or spec.get('measurement_aggregate_eligible') is not (kind == 'measurement')):
        raise ValueError('RainbowPrompt worker specification exceeds the authorized workloads')
    return True


def native_command(spec, output=None):
    settings = protocol(spec['mode'])
    command = ['--model', 'rainbow-prompt', '--model_config', 'default', '--dataset', 'seq-cifar100-224',
               '--dataset_config', 'ease', '--backbone', 'rainbow-vit', '--rainbow_vit_size', 'base',
               '--rainbow_checkpoint', str(spec.get('checkpoint_file', CHECKPOINT)),
               '--base_path', str(spec.get('data_root') or public_setting('data_root')), '--device', 'cuda:0',
               '--n_epochs', str(settings['n_epochs']), '--rainbow_warmup_epochs', str(settings['warmup_epochs']),
               '--stop_after', str(settings['tasks']), '--batch_size', '64', '--num_workers', '4',
               '--seed', '1993', '--permute_classes', '0', '--validation', '0',
               '--lr', '0.03', '--optimizer', 'adam', '--optim_wd', '0',
               '--enable_other_metrics', '0', '--eval_future', '0', '--debug_mode', '0', '--non_verbose', '1',
               '--save_after_interrupt', '0', '--joint', '0', '--disable_log', '0', '-O', '0']
    if output is not None:
        command.extend(['--results_path', str(Path(output) / 'mammoth')])
    return command


def phase_seed(seed, task_id, phase, epoch=0):
    from scripts.low_rank_protocol import phase_seed as choose
    return choose(seed, task_id, phase, epoch)


def seed_phase(seed, task_id, phase, epoch=0):
    from scripts.low_rank_protocol import seed_phase as apply
    return apply(seed, task_id, phase, epoch)


def configure_epoch_loader(loader, seed, task_id, phase, epoch=0):
    from scripts.low_rank_protocol import configure_epoch_loader as configure
    return configure(loader, seed, task_id, phase, epoch)
