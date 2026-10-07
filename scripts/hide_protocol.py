"""Approved HiDe execution settings and reproducible shared data conventions.

Importing this module performs no ML imports, credential reads or execution.
The historical CPU-only validation plan is retained unchanged.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = '0ed433334ff2cfb206c5e2899651c3044ec85f92'
CHECKPOINT_SHA256 = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
CANONICAL_PRETRAINED_SHA256 = 'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300'
CHECKPOINT = ('/mnt/sdab1/userHome/wslim/.cache/torch/hub/checkpoints/'
              'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
PHASES = ('tii_train', 'prompt_train', 'tii_statistics', 'prompt_statistics',
          'tii_alignment', 'prompt_alignment', 'evaluation')
SOURCE_FILES = ('scripts/run_hide_validation.py', 'scripts/hide_protocol.py',
                'scripts/hide_acceptance.py', 'scripts/hide_reference_utils.py',
                'models/hide_prompt.py', 'backbone/hide_vit.py',
                'models/hide_prompt_utils/network.py', 'models/hide_prompt_utils/prompt.py',
                'models/hide_prompt_utils/vit.py', 'models/hide_prompt_utils/statistics.py',
                'models/config/hide_prompt.yaml', 'datasets/configs/seq-cifar100-224/hide_prompt.yaml',
                'datasets/configs/seq-cifar100-224/ease.yaml', 'scripts/cl_efficiency.py',
                'utils/training.py', 'main.py', 'uv.lock')


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def protocol(mode='validation'):
    if mode not in ('validation', 'smoke', 'efficiency'):
        raise ValueError('HiDe mode must be validation, smoke or efficiency')
    validation = mode == 'validation'
    return {'schema_version': 1, 'mode': mode, 'model': 'hide-prompt',
            'training_authorized': True, 'approval': 'User: 좋아. 이대로 진행. after the HiDe GPU execution recommendations',
            'tasks': 1 if mode == 'efficiency' else 2,
            'tii_epochs': 20 if validation else 1, 'prompt_epochs': 50 if validation else 1,
            'n_epochs': 70 if validation else 2, 'ca_epochs': 1 if mode == 'smoke' else 30,
            'batch_size': 64, 'seed': 1993, 'num_workers': 4,
            'dataset': 'seq-cifar100-224', 'dataset_config': 'ease' if mode == 'efficiency' else 'hide_prompt',
            'dataset_total_tasks': 10, 'classes_per_task': 10, 'vit_size': 'base',
            'optimizer': 'adam', 'weight_decay': 0., 'tii_lr': .0005, 'prompt_lr': .03,
            'ca_lr': .005, 'head_lr_ratio': .1, 'reg': .1, 'centroids': 10,
            'prefix_permute_fix': False, 'drop_last': False, 'shuffle_training': True,
            'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.,
            'paper_reproduction': False, 'accuracy_acceptance_eligible': validation,
            'phase_rng': 'base_seed + 100000*task_id + 1000*phase_index + local_epoch',
            'statistics_batching': 'per-class, stable raw-index order, sequential, no dropped tail',
            'evaluation_batching': 'separate native test-task loaders, stable raw-index order',
            'equal_compute_budget_to_existing_one_epoch': False}


def phase_seed(seed, task_id, phase, epoch=0):
    if phase not in PHASES or type(task_id) is not int or task_id < 0 or type(epoch) is not int or not 0 <= epoch < 1000:
        raise ValueError('Invalid HiDe phase RNG coordinates')
    return (int(seed) + task_id * 100000 + PHASES.index(phase) * 1000 + epoch) % (2 ** 32)


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
    """Match both runners' Python/NumPy augmentations to Torch's worker seed."""
    import numpy as np
    import torch
    selected = torch.initial_seed() % (2 ** 32)
    random.seed(selected)
    np.random.seed(selected)


def configure_epoch_loader(loader, seed, task_id, phase, epoch=0):
    import torch
    selected = phase_seed(seed, task_id, phase, epoch)
    generator = torch.Generator().manual_seed(selected)
    loader.generator = generator
    if hasattr(loader.sampler, 'generator'):
        loader.sampler.generator = generator
    loader.worker_init_fn = worker_seed
    if loader.drop_last or getattr(loader, 'persistent_workers', False):
        raise ValueError('Matched HiDe training requires fresh workers and no dropped samples')
    return selected


def native_command(mode, checkpoint=CHECKPOINT, data_root='/mnt/sdab1/share/data', output=None):
    settings = protocol(mode)
    command = ['--model', 'hide-prompt', '--dataset', settings['dataset'], '--model_config', 'default',
               '--dataset_config', settings['dataset_config'], '--hide_checkpoint', str(checkpoint),
               '--hide_vit_size', 'base', '--hide_tii_epochs', str(settings['tii_epochs']),
               '--hide_prompt_epochs', str(settings['prompt_epochs']), '--hide_ca_epochs', str(settings['ca_epochs']),
               '--n_epochs', str(settings['n_epochs']), '--stop_after', str(settings['tasks']),
               '--batch_size', str(settings['batch_size']), '--num_workers', str(settings['num_workers']),
               '--seed', str(settings['seed']), '--permute_classes', '0', '--lr', '.03', '--optim_wd', '0',
               '--optimizer', 'adam', '--code_optimization', '0', '--noise_rate', '0', '--validation', '0',
               '--label_perc', '1', '--label_perc_by_class', '1', '--eval_future', '0',
               '--enable_other_metrics', '0', '--debug_mode', '0', '--non_verbose', '1',
               '--base_path', str(data_root), '--device', 'cuda:0']
    if output is not None:
        command.extend(['--results_path', str(Path(output) / 'mammoth'),
                        '--checkpoint_path', str(Path(output) / 'checkpoints')])
    return command


def dataset_args(mode='validation', checkpoint=CHECKPOINT, data_root='/mnt/sdab1/share/data'):
    """Share parser/dataset configuration only; never initialize a native model."""
    import main as mammoth
    return mammoth.parse_args(cmd=native_command(mode, checkpoint, data_root))


def create_dataset(mode='validation', checkpoint=CHECKPOINT, data_root='/mnt/sdab1/share/data'):
    # Parsing alone does not call main.extend_args, which normally sets this
    # process-wide path before dataset/class-name construction.
    validate_shared_data(data_root)
    from utils.conf import base_path
    base_path(str(Path(data_root).resolve()))
    from datasets import get_dataset
    args = dataset_args(mode, checkpoint, data_root)
    with forbid_cifar_download():
        return get_dataset(args)


def validate_shared_data(data_root):
    """Verify existing official raw files before any download-enabled constructor."""
    from torchvision.datasets import CIFAR100
    root = Path(data_root).resolve() / 'CIFAR100' / CIFAR100.base_folder
    expected = dict(CIFAR100.train_list + CIFAR100.test_list)
    expected[CIFAR100.meta['filename']] = CIFAR100.meta['md5']
    files = {}
    for name, expected_md5 in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError('Shared CIFAR100 raw files are missing; downloads are disabled')
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                md5.update(block); sha256.update(block)
        if md5.hexdigest() != expected_md5:
            raise ValueError('Shared CIFAR100 raw file integrity failed; downloads are disabled')
        files[name] = {'path': str(path), 'md5': expected_md5, 'sha256': sha256.hexdigest()}
    return {'root': str(root), 'files': files, 'downloads_allowed': False}


@contextmanager
def forbid_cifar_download():
    """Cover dataset and class-name helpers that call download=True internally."""
    from unittest.mock import patch
    from torchvision.datasets import CIFAR100

    def existing_only(dataset):
        if not dataset._check_integrity():
            raise FileNotFoundError('CIFAR100 integrity failed; this execution forbids downloads')

    with patch.object(CIFAR100, 'download', existing_only):
        yield


no_cifar_downloads = forbid_cifar_download


def source_hashes():
    return {source: file_hash(ROOT / source) for source in SOURCE_FILES}


def validate_efficiency_acceptance(path, plan=None, initial_state=None):
    """Fail closed before credentials or workers; reviewed accuracy cannot pass."""
    import json
    record = json.loads(Path(path).read_text())
    from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS
    if (record.get('status') != 'completed' or record.get('acceptance_status') != 'pass'
            or record.get('mode') != 'validation' or record.get('model') != 'hide-prompt'
            or record.get('protocol') != protocol('validation')
            or record.get('wandb_logging_status') != 'uploaded'):
        raise ValueError('HiDe efficiency requires completed, uploaded, passing full validation')
    checks = record.get('structural_checks', {})
    if any(checks.get(key) is not True for key in REQUIRED_STRUCTURAL_CHECKS) or any(value is not True for value in checks.values()):
        raise ValueError('HiDe acceptance structural gates did not all pass')
    pins = record.get('execution_source_sha256', {})
    if not pins or any(file_hash(ROOT / source) != pin for source, pin in pins.items()):
        raise ValueError('HiDe accepted execution sources have changed')
    if record.get('checkpoint_sha256') != CHECKPOINT_SHA256 or file_hash(record['checkpoint_file']) != CHECKPOINT_SHA256:
        raise ValueError('HiDe accepted checkpoint differs from the approved pretrained file')
    selected = Path(initial_state or record['initial_state'])
    if file_hash(selected) != record.get('initial_state_sha256'):
        raise ValueError('HiDe efficiency requires the same exported learned initialization')
    return record
