"""Approved comparison plan and argv generation; stdlib only, no execution.

The old Base-only experiment adapters are not used here. Commands describe the
native training entrypoint; an execution recorder must still supply the pinned
input/RNG audits, task evidence, sensors and publication before a run is ready.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/cl_efficiency/unified_comparison/protocol.json'
DATA_ROOT = '/mnt/sdab1/share/data'
CACHE_ROOT = '/mnt/sdab1/userHome/wslim/.cache'
GENERIC_METHODS = ('sgd', 'er', 'derpp', 'ewc_on', 'si', 'lwf_mc')
PEFT_METHODS = ('l2p', 'dualprompt', 'coda-prompt', 'hide-prompt', 'rainbow-prompt',
                'ease', 'ranpac', 'ewc-lora', 'sd-lora', 'inf-lora', 'cl-lora')
METHODS = GENERIC_METHODS + PEFT_METHODS
SIZES = ('tiny', 'small', 'base')
BATCHES = (16, 64)
MODES = ('validation', 'measurement', 'flops')
DIMENSIONS = {'tiny': (192, 3), 'small': (384, 6), 'base': (768, 12)}
CHECKPOINTS = {
    'tiny': {
        'file': CACHE_ROOT + '/torch/hub/checkpoints/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        'sha256': '0c5c4818ecf1049ae34194223c908172a2967334bab6d86670351b59d608b5dd',
        'canonical_sha256': '3258efdf29eb69379b559f8884b31134e5f299a2e6cd682f08eb9483e9c8e1cb',
        'pretraining_recipe_note': 'AugReg aug_none, weight decay 0.03; size recipes differ',
    },
    'small': {
        'file': CACHE_ROOT + '/torch/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        'sha256': '545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3',
        'canonical_sha256': '1bc2129fcd9c44c0104385f3db58a422595d205fcfff592392124b5f2bf7b1a5',
        'pretraining_recipe_note': 'AugReg aug_light1, weight decay 0.03; size recipes differ',
    },
    'base': {
        'file': CACHE_ROOT + '/torch/hub/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        'sha256': 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124',
        'canonical_sha256': 'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300',
        'pretraining_recipe_note': 'AugReg aug_medium1, weight decay 0.1; size recipes differ',
    },
}


def _method_recipes():
    recipes = {method: {'optimizer': 'adamw', 'lr': 0.0001, 'optim_wd': 0.0,
                       'lr_scheduler': None, 'lr_scaling': 'none',
                       'recipe_label': 'new benchmark starting point; not an official reproduction'}
               for method in GENERIC_METHODS}
    recipes['sgd']['method_label'] = 'Sequential fine-tuning; optimizer is explicitly AdamW'
    recipes['derpp'].update(alpha=0.3, beta=0.5)
    recipes['ewc_on'].update(e_lambda=10.0, gamma=1.0)
    recipes['si'].update(c=0.5, xi=1.0)
    recipes['lwf_mc'].update(wd_reg=0.0)
    for method in ('l2p', 'dualprompt', 'hide-prompt', 'rainbow-prompt'):
        recipes[method] = {'optimizer': 'adam', 'lr': 0.03, 'optim_wd': 0.0,
                           'lr_scheduler': None, 'lr_scaling': 'nominal_lr * current_batch / 256'}
    recipes['l2p'].update(pool_size_l2p=10, length=5, top_k=5, batchwise_prompt=0,
                          pull_constraint_coeff=0.5)
    recipes['dualprompt'].update(size=10, length=5, top_k=1, g_prompt_length=5,
                                 g_prompt_layer_idx=[0, 1], e_prompt_layer_idx=[2, 3, 4], use_permute_fix=0)
    recipes['coda-prompt'] = {'optimizer': 'adam', 'lr': 0.001, 'optim_wd': 0.0,
                             'lr_scheduler': None, 'lr_scaling': 'none', 'pool_size': 100,
                             'prompt_len': 8, 'mu': 0.0, 'ortho_mu': 0.0, 'virtual_bs_iterations': 1,
                             'owned_scheduler': 'CosineSchedule, recreated each task'}
    recipes['hide-prompt'].update(hide_pool_size=10, hide_prompt_length=5,
                                 hide_prompt_layers=[0, 1, 2, 3, 4], hide_mlp_structure=[2],
                                 hide_ca_epochs=30, hide_tii_lr=0.0005, hide_ca_lr=0.005,
                                 hide_head_lr_ratio=0.1, hide_reg=0.1, hide_n_centroids=10,
                                 hide_use_permute_fix=0)
    recipes['rainbow-prompt'].update(rainbow_pool_size=10, rainbow_prompt_length=20,
                                    rainbow_prompt_layers=list(range(12)), rainbow_self_attn_layers=list(range(6)),
                                    rainbow_d1=56, rainbow_d2=96, rainbow_ki_iter=10,
                                    rainbow_balance=0.01, rainbow_clip_grad=1.0)
    for method in ('ease', 'ranpac'):
        recipes[method] = {'optimizer': 'sgd', 'lr': 0.025, 'optim_wd': 0.0005,
                           'optim_mom': 0.9, 'optim_nesterov': 0,
                           'lr_scheduler': 'cosine', 'lr_scaling': 'none', 'adapter_size': 64}
    recipes['ranpac'].update(rp_size=10000, lr_scheduler=None,
                             owned_scheduler='Task1 source cosine; task2 performs no optimizer update')
    recipes['ewc-lora'] = {'optimizer': 'adam', 'lr': 0.0005, 'optim_wd': 0.0,
                           'lr_scheduler': 'cosine', 'lr_scaling': 'none', 'lora_rank': 10,
                           'lora_fc_lr': 0.005, 'lora_gamma': 1.0, 'lora_ewc_lambda': 10000000.0}
    recipes['sd-lora'] = {'optimizer': 'adam', 'lr': 0.001, 'optim_wd': 0.0,
                          'lr_scheduler': 'cosine', 'lr_scaling': 'none', 'lora_rank': 10,
                          'lora_fc_lr': 0.008, 'recipe_label': 'accepted Base recovery LR; new sizes require stability checks'}
    recipes['inf-lora'] = {'optimizer': 'adam', 'lr': 0.0005, 'optim_wd': 0.0,
                           'lr_scheduler': 'cosine', 'lr_scaling': 'none', 'lora_rank': 10,
                           'lora_fc_lr': 0.0005, 'lora_lamb': 0.95, 'lora_lame': 1.0}
    recipes['cl-lora'] = {'optimizer': 'sgd', 'lr': 0.03, 'optim_wd': 0.0001,
                          'optim_mom': 0.9, 'optim_nesterov': 0, 'lr_scheduler': 'cosine',
                          'lr_scaling': 'none', 'lora_rank': 10, 'lora_fc_lr': 0.03,
                          'lora_kd_ratio': 5.0, 'lora_temperature': 2.0, 'lora_msa': [1, 0, 1],
                          'lora_shared_pos': list(range(6)), 'lora_specific_pos': list(range(6, 12))}
    return recipes


METHOD_RECIPES = _method_recipes()
EXTRA_PHASES = {
    'sgd': ['No additional CL mechanism'],
    'er': ['Reservoir replay begins within task1; transformed replay concatenated with current batch'],
    'derpp': ['Two independently sampled replay forwards: stored-logit MSE and label CE; active within task1'],
    'ewc_on': ['End of each task: 5000 individual-example Fisher forward/backward calls, no optimizer steps',
               'Task2: previous-weight/Fisher penalty; source normalization uses len(loader)*batch'],
    'si': ['Per-update parameter/gradient vector tracking', 'Task end importance update; task2 importance penalty'],
    'lwf_mc': ['Task-end teacher copy', 'Task2 old-teacher forward plus student forward on each current batch'],
    'l2p': ['Frozen query forward plus prompted forward', 'Seen-class evaluation and non-batchwise prompt routing'],
    'dualprompt': ['Frozen query forward plus prompted forward', 'Task2 prompt transfer and task-independent evaluation'],
    'coda-prompt': ['Query forward plus prompted forward', 'Task2 Gram-Schmidt Parameter replacement and optimizer recreation',
                    'Record optimizer-updated, requires-grad and gradient-bearing parameters separately'],
    'hide-prompt': ['Each task: separate TII and prompt training stages, each uses the nominal epoch count',
                    'Each task: per-class statistics collection for both branches',
                    'Task2: TII classifier alignment30 epochs and prompt classifier alignment30 epochs'],
    'rainbow-prompt': ['Query plus prompt forwards, retain_graph=True, pre-last-update cached prompt preserved',
                       'Official active path without learned gate', 'Validation warmup0; measurement/FLOPs warmup5'],
    'ease': ['Task-specific adapter and proxy head', 'Task-end prototype extraction/completion and task2 expanded inference'],
    'ranpac': ['Task1 only: adapter training', 'Task2 observe returns0: no optimizer updates; source training loader loops remain',
               'Each task: feature extraction, random projection, ridge selection/solve and classifier replacement'],
    'ewc-lora': ['Each task: prototypes and empirical Fisher extra pass', 'Task2 EWC penalty; post-evaluation source finalize'],
    'sd-lora': ['Prototypes and source scale updates', 'Task2 accumulated task-specific low-rank branches'],
    'inf-lora': ['Begin-task activation pass and SVD', 'Task-end prototypes and DualGPM update', 'Task2 past-subspace projection'],
    'cl-lora': ['Task1 one optimizer update per observe', 'Task2 KD update followed by CE update: two optimizer updates per observe',
                'Task-end prototypes and old/shared low-rank snapshots'],
}


def approved_protocol():
    """Return an independent JSON-compatible copy of the authorized plan."""
    return {
        'schema_version': 1, 'stage': 'unified_comparison', 'approved_on': '2026-10-08',
        'training_authorized': True, 'experiments_executed': False,
        'approval_scope': 'User approved all17 methods, all3 sizes, both batches, fresh two-task comparison',
        'runtime_ready': False, 'entrypoint': 'main.py', 'old_base_only_runners_reused': False,
        'methods': list(METHODS), 'generic_methods': list(GENERIC_METHODS), 'peft_methods': list(PEFT_METHODS),
        'primary_size': 'tiny', 'sizes': list(SIZES), 'batch_sizes': list(BATCHES),
        'dataset': {'name': 'seq-cifar100-224', 'config': 'unified', 'setting': 'class-il',
                    'total_tasks': 10, 'stop_after': 2, 'classes_per_task': 10, 'num_classes': 100,
                    'class_order': list(range(100)), 'train_samples_per_task': 5000,
                    'test_samples_per_task': 1000, 'shuffle_training': True, 'drop_last': False,
                    'train_transform': ['RandomResizedCrop(224,scale=(0.08,1),ratio=(0.75,4/3),bicubic,antialias=True)',
                                        'RandomHorizontalFlip(0.5)', 'ToTensor', 'Normalize(0.5,0.5)'],
                    'test_transform': ['Resize(256,bicubic,antialias=True)', 'CenterCrop(224)',
                                       'ToTensor', 'Normalize(0.5,0.5)'],
                    'replay_and_auxiliary_transform_policy': 'Audit actual replay/statistics/herding paths, including Kornia conversion; never assume matching config implies bit-identical crops'},
        'environment': {'gpu': '1', 'device': 'cuda:0', 'gpu_processes': 1, 'seed': 1993,
                        'num_workers': 4, 'torch_cpu_threads': 2, 'data_root': DATA_ROOT, 'cache_root': CACHE_ROOT,
                        'torch_checkpoints': CACHE_ROOT + '/torch/hub/checkpoints',
                        'huggingface_hub': CACHE_ROOT + '/huggingface/hub', 'clip_cache': CACHE_ROOT + '/clip'},
        'precision': {'parameter_dtype': 'float32', 'autocast': False, 'matmul_precision': 'highest',
                      'tf32_allowed': False, 'cudnn_tf32_allowed': False,
                      'cudnn_deterministic': True, 'cudnn_benchmark': False, 'code_optimization': 0},
        'replay': {'buffer_size': 500, 'minibatch_policy': 'equal_current_batch',
                   'decision': 'New approved decision replaces previously proposed fixed replay64',
                   'scope': ['er', 'derpp'], 'budget_unit': 'examples, not total auxiliary bytes'},
        'schedule': {'validation_epochs_per_task': 2, 'measurement_epochs_per_task': 20,
                     'measurement_repeats': 3, 'flops_runs_per_cell': 1,
                     'flops_epochs_per_task': 20, 'flops_excluded_from_resource_statistics': True,
                     'repeat_policy': 'Fresh same-seed processes; cost variation, not accuracy seed uncertainty',
                     'hide_epoch_interpretation': 'Nominal count applies to each TII and prompt stage; CA30/task2 retained',
                     'rainbow_validation_warmup': 0, 'rainbow_measurement_warmup': 5,
                     'lr_policy': 'Preserve method-specific batch scaling; generic6 fixed starting LR; no added automatic scaling',
                     'counts': {'cells': 102, 'validation_runs': 102, 'measurement_runs': 306,
                                'flops_runs': 102, 'total_planned_processes': 510},
                     'ordering': 'All validation cells first in tiny/small/base and16/64 order; measurement/FLOPs require corresponding cell acceptance',
                     'failure_policy': 'Stop and report unresolved failures; no automatic retry or silent recipe/batch substitution'},
        'checkpoints': deepcopy(CHECKPOINTS), 'pretraining_family': 'augreg_in21k_ft_in1k',
        'method_recipes': deepcopy(METHOD_RECIPES), 'extra_phases': deepcopy(EXTRA_PHASES),
        'accuracy': {'metrics': ['task1_seen', 'task2_seen', 'task2_old', 'task2_new', 'task1_minus_task2_old_pp'],
                     'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.0,
                     'gap_policy': 'Only matched independent-reference recipes qualify for the existing1.5/3pp source/native policy',
                     'abnormal_policy': 'Investigate abnormal quality using loss/state/updates/matched evidence; no universal90% Tiny/generic floor',
                     'validation_label': 'New common-input two-epoch structural validation; not paper-quality reproduction',
                     'measurement_label': 'Two-task20-epoch recipe; extra phases and different update counts remain visible',
                     'paper_reproduction': False, 'equal_compute_budget': False},
        'measurement': {'whole_observation': 'Initialization,data loading,begin/train/end,method extra phases,evaluation,audits/checkpoints; final publication excluded',
                        'phase_costs': ['initialization', 'data_loading', 'task_begin', 'train', 'task_end', 'evaluation',
                                        'Fisher', 'SVD_or_GPM', 'statistics', 'classifier_alignment', 'ridge_solve', 'checkpoint'],
                        'overlapping_phases_are_not_summed': True,
                        'cost_metrics': ['wall_s', 'cpu_s', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes',
                                         'nvml_peak_device_bytes', 'rss_peak_bytes', 'gpu_board_j', 'power_coverage',
                                         'partial_supported_flops', 'optimizer_updates', 'backward_calls',
                                         'encoder_calls', 'image_visits', 'parameter_and_auxiliary_unique_storage_bytes'],
                        'gpu_energy_is_not_whole_machine_energy': True,
                        'partial_flops_are_not_total_compute': True, 'inference_benchmark_calls_authorized_here': 0,
                        'inference_policy': 'Reuse ordinary task-end evaluation timings; standalone inference workload needs its own explicit plan',
                        'historical_results': 'Preserve as references, do not pool into fresh unified measurements'},
        'publication': {'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'wandb_mode': 'online',
                        'slack_notifications': True, 'secrets_source': 'Existing environment or local runtime config; never serialized',
                        'local_results_root': str(ROOT / 'results/unified_comparison'),
                        'checkpoints_stay_local': True},
        'required_runtime_evidence': ['checkpoint_file_and_canonical_tensor_pins', 'effective_args_and_actual_replay_batch',
                                      'common_input_ids_transforms_and_task_rng', 'method_lifecycle_and_extra_phase_audits',
                                      'finite_loss_gradients_and_quality_review', 'cell_validation_acceptance',
                                      'source_commit_and_file_hashes', 'complete_task_and_resource_records'],
        'excluded': ['PROL', 'LW2G', 'MIR_new_implementation', 'BudgetCL_or_aL_SAR_budget_controller',
                     'full10_tasks', 'extra_seeds', 'rank_or_prompt_sweeps', 'automatic_retries', 'remote_git_push'],
        'eta': {'measurement_gpu_hours_range': [45, 110], 'validation_and_full_recipe_flops_gpu_hours_range': [65, 160],
                'user_facing_calendar_days_range': [4, 8],
                'evidence': ['GPU1 historical Base EASE20epoch/task two-task observation approximately565s',
                             'GPU1 Base EWC/SD/Inf-LoRA two-task20epoch observations734-942s',
                             'GPU1 Base CL-LoRA two-task30epoch1219s, linear20epoch extrapolation approximately813s',
                             'EASE train seconds/epoch Base13.3411,Small5.6880,Tiny3.1364 at batch64'],
                'limitations': 'Rough planning range, not a measured new-protocol forecast; generic6,CODA,batch16,new input and Fisher cost unknown. Development/failure repair, external GPU contention and network delay unbounded. Re-estimate after successful actual cells.'},
    }


def _strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_strict_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_strict_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def validate_protocol(protocol):
    """Reject scope/recipe drift without echoing unknown fields or their values."""
    if not _strict_equal(protocol, approved_protocol()):
        raise ValueError('Unified comparison protocol differs from the approved plan')
    return deepcopy(protocol)


def load_protocol(path=PLAN):
    with Path(path).open(encoding='utf-8') as stream:
        return validate_protocol(json.load(stream))


def verify_checkpoints(protocol=None):
    """Optional read-only CPU hashing; never downloads or constructs a model."""
    protocol = load_protocol() if protocol is None else validate_protocol(protocol)
    observed = {}
    for size, checkpoint in protocol['checkpoints'].items():
        digest = hashlib.sha256()
        with Path(checkpoint['file']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(2 ** 20), b''):
                digest.update(chunk)
        if digest.hexdigest() != checkpoint['sha256']:
            raise ValueError('Unified comparison checkpoint file changed')
        observed[size] = digest.hexdigest()
    return observed


def _make_spec(method, size, batch_size, mode, repeat):
    if method not in METHODS or size not in SIZES or type(batch_size) is not int or batch_size not in BATCHES or mode not in MODES:
        raise ValueError('Unsupported unified comparison cell')
    max_repeat = 3 if mode == 'measurement' else 1
    if type(repeat) is not int or not 1 <= repeat <= max_repeat:
        raise ValueError('Unsupported unified comparison repetition')
    nominal_epochs = 2 if mode == 'validation' else 20
    epochs = nominal_epochs * 2 if method == 'hide-prompt' else nominal_epochs
    per_epoch = math.ceil(5000 / batch_size)
    per_task = per_epoch * epochs
    updates = [per_task, per_task]
    if method == 'cl-lora':
        updates[1] *= 2
    elif method == 'ranpac':
        updates[1] = 0
    run_id = f'unified-{method}-{size}-b{batch_size}-{mode}-r{repeat}'
    dimension, heads = DIMENSIONS[size]
    return {'schema_version': 1, 'run_id': run_id, 'method': method, 'vit_size': size,
            'batch_size': batch_size, 'minibatch_size': batch_size, 'buffer_size': 500,
            'mode': mode, 'repeat': repeat, 'nominal_epochs_per_task': nominal_epochs, 'n_epochs': epochs,
            'tasks': 2, 'dataset_total_tasks': 10, 'classes_per_task': 10, 'num_classes': 100,
            'dataset': 'seq-cifar100-224', 'dataset_config': 'unified', 'seed': 1993,
            'num_workers': 4, 'torch_cpu_threads': 2, 'gpu': '1', 'feature_dim': dimension, 'num_heads': heads,
            'checkpoint': deepcopy(CHECKPOINTS[size]), 'method_recipe': deepcopy(METHOD_RECIPES[method]),
            'rainbow_warmup_epochs': 0 if mode == 'validation' else 5,
            'extra_phases': list(EXTRA_PHASES[method]),
            'expected_training': {'loader_batches_per_epoch': per_epoch, 'observe_calls_per_task': [per_task, per_task],
                                  'main_optimizer_steps_per_task': updates,
                                  'fisher_backward_calls_per_task': [5000, 5000] if method == 'ewc_on' else None,
                                  'counts_note': 'Main-loop counts only; method-specific extra passes and alignment are separately audited'},
            'resource_statistics_eligible': mode == 'measurement',
            'requires_validation_run_id': None if mode == 'validation' else f'unified-{method}-{size}-b{batch_size}-validation-r1',
            'result_dir': str(ROOT / 'results/unified_comparison' / run_id),
            'training_authorized': True, 'runtime_ready': False, 'executed': False}


def validate_run_spec(spec):
    try:
        expected = _make_spec(spec['method'], spec['vit_size'], spec['batch_size'], spec['mode'], spec['repeat'])
    except (KeyError, TypeError) as error:
        raise ValueError('Malformed unified comparison run spec') from error
    if not _strict_equal(spec, expected):
        raise ValueError('Unified comparison run spec differs from the approved cell')
    return deepcopy(spec)


def build_run_specs(protocol=None, modes=MODES):
    """Return the bounded510-run plan. This function does not start any jobs."""
    protocol = load_protocol() if protocol is None else validate_protocol(protocol)
    if (not isinstance(modes, (tuple, list)) or not modes
            or any(type(mode) is not str or mode not in MODES for mode in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError('Modes must be a nonempty unique subset of approved modes')
    return [_make_spec(method, size, batch, mode, repeat)
            for mode in modes for size in protocol['sizes'] for batch in protocol['batch_sizes']
            for method in protocol['methods'] for repeat in range(1, 4 if mode == 'measurement' else 2)]


def build_method_args(spec):
    """Native argument values, including every method's explicit size/weights."""
    spec = validate_run_spec(spec)
    method, size, checkpoint = spec['method'], spec['vit_size'], spec['checkpoint']['file']
    args = {'model': method, 'model_config': 'default', 'dataset': 'seq-cifar100-224', 'dataset_config': 'unified',
            'stop_after': 2, 'n_epochs': spec['n_epochs'], 'batch_size': spec['batch_size'],
            'num_workers': 4, 'seed': 1993, 'drop_last': 0, 'permute_classes': 0,
            'fitting_mode': 'epochs', 'scheduler_mode': 'epoch', 'enable_other_metrics': 0,
            'eval_future': 0, 'joint': 0, 'non_verbose': 1, 'code_optimization': 0,
            'base_path': DATA_ROOT, 'device': 'cuda:0', 'backbone': 'vit-augreg', 'vit_size': size,
            'pretrained': 1, 'vit_checkpoint': checkpoint, 'lr_scheduler': None,
            'results_path': str(Path(spec['result_dir']) / 'native_results'),
            'checkpoint_path': str(Path(spec['result_dir']) / 'checkpoints'),
            'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency', 'wandb_name': spec['run_id']}
    excluded = {'lr_scaling', 'recipe_label', 'method_label', 'owned_scheduler'}
    args.update({key: deepcopy(value) for key, value in spec['method_recipe'].items() if key not in excluded})
    if method in ('er', 'derpp'):
        args.update(buffer_size=500, minibatch_size=spec['batch_size'])
    if method in ('l2p', 'dualprompt', 'coda-prompt'):
        args['pretrained'] = 0
        args.pop('vit_checkpoint')
        if method == 'coda-prompt':
            args.update(coda_vit_size=size, coda_checkpoint=checkpoint)
        else:
            args.update(pretrained_checkpoint=checkpoint)
            args['l2p_vit_size' if method == 'l2p' else 'dualprompt_vit_size'] = size
    elif method == 'hide-prompt':
        args.update(backbone='hide-vit', hide_vit_size=size, hide_checkpoint=checkpoint,
                    hide_tii_epochs=spec['nominal_epochs_per_task'], hide_prompt_epochs=spec['nominal_epochs_per_task'])
        args.pop('vit_size')
        args.pop('vit_checkpoint')
        args.pop('pretrained')
    elif method == 'rainbow-prompt':
        args.update(backbone='rainbow-vit', rainbow_vit_size=size, rainbow_checkpoint=checkpoint,
                    rainbow_warmup_epochs=spec['rainbow_warmup_epochs'])
        args.pop('vit_size')
        args.pop('vit_checkpoint')
        args.pop('pretrained')
    elif method == 'ease':
        args.update(backbone='ease-vit', ease_checkpoint=checkpoint, ease_pretrain_type='in21k_ft_in1k')
        args.pop('vit_checkpoint')
    elif method == 'ranpac':
        args.update(pretrained=0, ranpac_vit_size=size, pretrained_checkpoint=checkpoint)
        args.pop('vit_checkpoint')
        args.pop('adapter_size')  # Source RanPAC's bottleneck64 is internal, not a parser argument.
    elif method.endswith('-lora'):
        args.update(backbone='low-rank-vit', lora_vit_size=size, lora_checkpoint=checkpoint)
        args.pop('vit_size')
        args.pop('vit_checkpoint')
        args.pop('pretrained')
    return args


def expected_initial_learning_rates(spec):
    """Planned initial optimizer rates; actual runtime rates must be recorded."""
    spec = validate_run_spec(spec)
    method, recipe, batch = spec['method'], spec['method_recipe'], spec['batch_size']
    rate = recipe['lr'] * batch / 256 if recipe['lr_scaling'] != 'none' else recipe['lr']
    if method == 'hide-prompt':
        return {'tii': recipe['hide_tii_lr'] * batch / 256, 'prompt': rate,
                'prompt_head': rate * recipe['hide_head_lr_ratio'],
                'tii_alignment': recipe['hide_ca_lr'] * batch / 256, 'prompt_alignment': recipe['hide_ca_lr']}
    if method.endswith('-lora'):
        return {'main': rate, 'head': recipe['lora_fc_lr']}
    return {'main': rate}


def build_command(spec, executable='uv'):
    """An argv list, never a shell command. Does not invoke the process."""
    if type(executable) is not str or not executable:
        raise ValueError('An executable name is required')
    command = [executable, 'run', 'python', 'main.py']
    for key, value in build_method_args(spec).items():
        if value is None:
            continue  # Explicit model/default recipe sets scheduler=None; no invalid CLI None literal.
        command.append('--' + key)
        if isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


def build_environment(spec):
    """Public environment overlay only; never reads credentials or os.environ."""
    validate_run_spec(spec)
    return {'CUDA_VISIBLE_DEVICES': '1', 'MAMMOTH_DEVICE': 'cuda:0', 'CL_GPU': '1',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
            'CL_DATA_ROOT': DATA_ROOT, 'CL_CACHE_ROOT': CACHE_ROOT,
            'XDG_CACHE_HOME': CACHE_ROOT, 'TORCH_HOME': CACHE_ROOT + '/torch',
            'HF_HOME': CACHE_ROOT + '/huggingface', 'HF_HUB_CACHE': CACHE_ROOT + '/huggingface/hub',
            'WANDB_ENTITY': 'laymond1', 'WANDB_PROJECT': 'cl-efficiency', 'WANDB_MODE': 'online'}
