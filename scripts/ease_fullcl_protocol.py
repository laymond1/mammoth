"""Approved EASE Base comparison on the observed HiDe input protocol.

Imports perform no training, tensor imports, credential reads or downloads.
Historical EASE validation and its batch48 results are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = '6d67508ed8b7c4d23350ed4eb28c123914b40ae3'
CHECKPOINT = ('/mnt/sdab1/userHome/wslim/.cache/torch/hub/checkpoints/'
              'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
CHECKPOINT_SHA256 = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
CANONICAL_PRETRAINED_SHA256 = 'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300'
PHASES = ('train', 'prototype', 'evaluation', 'task_begin')
PRECISION = {'matmul_precision': 'highest', 'tf32_allowed': False, 'cudnn_tf32_allowed': True,
             'cudnn_deterministic': True, 'cudnn_benchmark': False, 'autocast': False,
             'parameter_dtype': 'float32', 'torch_cpu_threads': 2}
SOURCE_FILES = ('scripts/run_ease_fullcl.py', 'scripts/ease_fullcl_protocol.py',
                'scripts/ease_common_reference_utils.py', 'scripts/ease_acceptance.py',
                'scripts/ease_reference_utils.py', 'scripts/ease_size_protocol.py', 'scripts/run_ease_validation.py',
                'scripts/run_hide_validation.py', 'scripts/hide_protocol.py', 'scripts/cl_efficiency.py',
                'models/ease.py', 'backbone/ease_vit.py', 'models/ease_utils/network.py',
                'models/ease_utils/classifier.py', 'models/ease_utils/vit.py', 'models/config/ease.yaml',
                'datasets/configs/seq-cifar100-224/hide_prompt.yaml',
                'datasets/configs/seq-cifar100-224/ease.yaml', 'datasets/seq_cifar100_224.py',
                'datasets/utils/continual_dataset.py', 'utils/training.py', 'utils/schedulers.py',
                'utils/conf.py', 'main.py', 'uv.lock')
STRUCTURAL_CHECKS = ('finite_loss_and_gradients', 'dataset_matches_reference',
                     'frozen_backbone_unchanged', 'old_adapters_unchanged',
                     'prototype_and_adapter_shapes', 'training_protocol_complete',
                     'frozen_backbone_gradients_absent', 'checkpoint_round_trip',
                     'learnable_initialization_matches', 'actual_inputs_match_reference',
                     'nonoracle_seen_class_evaluation', 'canonical_pretrained_matches_pin')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def protocol(mode='validation'):
    if mode not in ('validation', 'full_cl'):
        raise ValueError('EASE shared-input mode must be validation or full_cl')
    return {'schema_version': 1, 'mode': mode, 'model': 'ease',
            'training_authorized': True, 'approval': 'User: 좋아. 이대로 진행. after EASE shared-input recommendations',
            'tasks': 2 if mode == 'validation' else 10, 'n_epochs': 20,
            'batch_size': 64, 'seed': 1993, 'num_workers': 4,
            'dataset': 'seq-cifar100-224', 'dataset_config': 'hide_prompt',
            'dataset_total_tasks': 10, 'classes_per_task': 10,
            'vit_size': 'base', 'feature_dim': 768, 'adapter_size': 64,
            'pretrain_family': 'augreg_in21k_ft_in1k', 'optimizer': 'sgd',
            'lr': .025, 'momentum': .9, 'weight_decay': .0005,
            'lr_scheduler': 'cosine', 'scheduler_mode': 'epoch',
            'ease_alpha': .1, 'ease_recalc_sim': True, 'ease_use_reweight': True,
            'drop_last': False, 'shuffle_training': True,
            'phase_rng': 'base_seed + 100000*task_id + 1000*phase_index + local_epoch',
            'initialization': 'Pinned author adapter/proxy catalog for all ten task coordinates',
            'prototype_batching': 'stable raw-index order, sequential, no dropped tail',
            'evaluation_batching': 'separate native test-task loaders, stable raw-index order',
            'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.,
            'paper_reproduction': False, 'equal_compute_budget': False}


def phase_seed(seed, task_id, phase, epoch=0):
    if (phase not in PHASES or type(task_id) is not int or task_id < 0
            or type(epoch) is not int or not 0 <= epoch < 1000):
        raise ValueError('Invalid EASE phase RNG coordinates')
    return (int(seed) + 100000 * task_id + 1000 * PHASES.index(phase) + epoch) % 2 ** 32


def seed_phase(seed, task_id, phase, epoch=0):
    import numpy as np
    import torch
    selected = phase_seed(seed, task_id, phase, epoch)
    random.seed(selected)
    np.random.seed(selected)
    torch.manual_seed(selected)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(selected)
    return selected


def worker_seed(worker_id):
    import numpy as np
    import torch
    selected = torch.initial_seed() % 2 ** 32
    random.seed(selected)
    np.random.seed(selected)


def configure_epoch_loader(loader, seed, task_id, phase, epoch=0):
    import torch
    if loader.drop_last or getattr(loader, 'persistent_workers', False):
        raise ValueError('EASE matched input requires fresh workers and no dropped tail')
    selected = phase_seed(seed, task_id, phase, epoch)
    generator = torch.Generator().manual_seed(selected)
    loader.generator = generator
    if hasattr(loader.sampler, 'generator'):
        loader.sampler.generator = generator
    loader.worker_init_fn = worker_seed
    return selected


def native_command(mode, checkpoint=CHECKPOINT, data_root='/mnt/sdab1/share/data', output=None):
    settings = protocol(mode)
    command = ['--model', 'ease', '--dataset', settings['dataset'], '--model_config', 'default',
               '--dataset_config', 'hide_prompt', '--ease_checkpoint', str(checkpoint),
               '--vit_size', 'base', '--adapter_size', '64', '--ease_pretrain_type', 'in21k_ft_in1k',
               '--stop_after', str(settings['tasks']), '--n_epochs', '20', '--batch_size', '64',
               '--num_workers', '4', '--seed', '1993', '--permute_classes', '0',
               '--lr', '.025', '--optim_mom', '.9', '--optim_wd', '.0005', '--optimizer', 'sgd',
               '--lr_scheduler', 'cosine', '--scheduler_mode', 'epoch', '--code_optimization', '0',
               '--noise_rate', '0', '--validation', '0', '--label_perc', '1', '--label_perc_by_class', '1',
               '--eval_future', '0', '--enable_other_metrics', '0', '--debug_mode', '0',
               '--non_verbose', '1', '--base_path', str(data_root), '--device', 'cuda:0']
    if output is not None:
        command.extend(['--results_path', str(Path(output) / 'mammoth'),
                        '--checkpoint_path', str(Path(output) / 'checkpoints')])
    return command


def prepare_native_args(args):
    """Apply only the pinned input configuration after legacy model-config parsing.

    EASE's model config overrides --dataset_config during parsing. Applying the
    uppercase dataset fields explicitly avoids inheriting HiDe's 70 epochs or
    batch48 while preserving the native dataset constructor and splitter.
    """
    import yaml
    from datasets import get_dataset_class
    config = yaml.safe_load((ROOT / 'datasets/configs/seq-cifar100-224/hide_prompt.yaml').read_text())
    dataset_fields = {key: value for key, value in config.items() if key.isupper()}
    get_dataset_class(args).set_default_from_config(dataset_fields, argparse.ArgumentParser())
    args.dataset_config = 'hide_prompt'
    args.pretrained = False
    return args


def dataset_args(mode='validation', checkpoint=CHECKPOINT, data_root='/mnt/sdab1/share/data'):
    import main as mammoth
    return prepare_native_args(mammoth.parse_args(cmd=native_command(mode, checkpoint, data_root)))


def create_dataset(mode='validation', checkpoint=CHECKPOINT, data_root='/mnt/sdab1/share/data'):
    from scripts.hide_protocol import forbid_cifar_download, validate_shared_data
    from utils.conf import base_path
    from datasets import get_dataset
    validate_shared_data(data_root)
    base_path(str(Path(data_root).resolve()))
    args = dataset_args(mode, checkpoint, data_root)
    with forbid_cifar_download():
        return get_dataset(args)


def source_hashes():
    return {source: file_hash(ROOT / source) for source in SOURCE_FILES}


def load_native_initialization(network, catalog, task_id):
    """Copy pinned author learnable tensors after the native lifecycle hook."""
    import torch
    if catalog.get('schema_version') != 1 or set(catalog.get('tasks', {})) != {str(task) for task in range(10)}:
        raise ValueError('EASE initialization must contain all ten authorized tasks')
    row = catalog['tasks'][str(task_id)]
    adapter = network.backbone.adapters[task_id]
    for target, key in ((adapter, 'adapter'), (network.proxy_head, 'proxy_head')):
        expected, actual = target.state_dict(), row[key]
        if set(expected) != set(actual) or any(not torch.is_tensor(value)
                or value.shape != expected[name].shape or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all()) for name, value in actual.items()):
            raise ValueError('EASE author initialization tensor shape, dtype or finiteness differs')
        target.load_state_dict(actual, strict=True)
        if any(not torch.equal(value.detach().cpu(), actual[name]) for name, value in target.state_dict().items()):
            raise RuntimeError('EASE learnable initialization failed exact tensor transfer')
    return True


def validate_reference(record, initial_state=None):
    if (record.get('status') != 'completed' or record.get('model') != 'ease'
            or record.get('protocol') != protocol('validation') or record.get('mode') != 'reference'
            or record.get('reference_commit') != REFERENCE_COMMIT
            or record.get('precision') != PRECISION
            or record.get('wandb_logging_status') != 'uploaded'
            or record.get('wandb_artifact', {}).get('state') != 'COMMITTED'):
        raise ValueError('EASE native requires a completed uploaded matching shared-input author reference')
    if record.get('checkpoint_sha256') != CHECKPOINT_SHA256 or file_hash(record['checkpoint_file']) != CHECKPOINT_SHA256:
        raise ValueError('EASE reference pretrained checkpoint differs')
    initial_state = Path(initial_state or record['initial_state']).resolve()
    if file_hash(initial_state) != record.get('initial_state_sha256'):
        raise ValueError('EASE author initialization catalog changed')
    if record.get('initial_canonical_pretrained_sha256') != CANONICAL_PRETRAINED_SHA256:
        raise ValueError('EASE reference canonical pretrained tensors differ')
    if not record.get('structural_checks') or any(value is not True for value in record['structural_checks'].values()):
        raise ValueError('EASE reference structural checks did not all pass')
    return record


def validate_full_cl_acceptance(path, initial_state=None):
    from scripts.ease_acceptance import evaluate_acceptance
    from scripts.run_ease_validation import reference_policy
    record = json.loads(Path(path).read_text())
    if (record.get('status') != 'completed' or record.get('mode') != 'validation'
            or record.get('model') != 'ease' or record.get('protocol') != protocol('validation')
            or record.get('acceptance_status') != 'pass' or record.get('precision') != PRECISION
            or record.get('wandb_logging_status') != 'uploaded'
            or record.get('wandb_artifact', {}).get('state') != 'COMMITTED'):
        raise ValueError('EASE full CL requires uploaded passing shared-input validation')
    checks = record.get('structural_checks', {})
    if set(checks) != set(STRUCTURAL_CHECKS) or any(value is not True for value in checks.values()):
        raise ValueError('EASE accepted structural gates are incomplete')
    if record.get('execution_source_sha256') != source_hashes():
        raise ValueError('EASE accepted execution sources changed')
    reference_path = Path(record['reference_summary'])
    if file_hash(reference_path) != record['reference_summary_sha256']:
        raise ValueError('EASE accepted reference summary changed')
    reference = validate_reference(json.loads(reference_path.read_text()), initial_state)
    policy, pins = reference_policy(reference, {'unit': 'percentage_points',
                                               'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.})
    if pins != record.get('reference_predictions_sha256'):
        raise ValueError('EASE accepted reference predictions changed')
    recalculated = evaluate_acceptance(record['task_results'], policy, structural_checks=checks)
    if recalculated.get('status') != 'pass' or recalculated != record.get('acceptance'):
        raise ValueError('EASE accepted exact-count accuracy gate no longer passes')
    selected = Path(initial_state or record['initial_state'])
    if file_hash(selected) != record['initial_state_sha256']:
        raise ValueError('EASE accepted initialization catalog changed')
    for task in (1, 2):
        filename = f'task{task}_predictions.npz'
        pin = record.get('wandb_artifact_files', {}).get(filename, {})
        if file_hash(Path(record['output']) / filename) != pin.get('sha256'):
            raise ValueError('EASE accepted native predictions changed')
    return record
