"""Pinned common-input low-rank acceptance protocol; imports are stdlib only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess

from scripts.hide_protocol import CHECKPOINT, CHECKPOINT_SHA256, CANONICAL_PRETRAINED_SHA256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/cl_efficiency/low_rank/validation_plan.json'
REFERENCE_COMMIT = 'd007ef5c13a6d8a7d7db3eedf93f5e796e2ce429'
METHODS = {'ewc-lora': 'ewclora', 'sd-lora': 'sdlora', 'inf-lora': 'inflora', 'cl-lora': 'cllora'}
METHOD_SETTINGS = {
    'ewc-lora': {'optimizer': 'adam', 'lr': .0005, 'fc_lr': .005, 'weight_decay': 0., 'gamma': 1., 'ewc_lambda': 1e7},
    'sd-lora': {'optimizer': 'adam', 'lr': .008, 'fc_lr': .008, 'weight_decay': 0.},
    'inf-lora': {'optimizer': 'adam', 'lr': .0005, 'fc_lr': .0005, 'weight_decay': 0., 'lamb': .95, 'lame': 1.},
    'cl-lora': {'optimizer': 'sgd', 'lr': .03, 'fc_lr': .03, 'weight_decay': .0001, 'momentum': .9,
                'kd_ratio': 5., 'temperature': 2., 'msa': [1, 0, 1], 'shared_pos': list(range(6)),
                'specific_pos': list(range(6, 12))},
}
PHASES = ('task_begin', 'train', 'task_end', 'evaluation')
PRECISION = {'matmul_precision': 'highest', 'tf32_allowed': False, 'cudnn_tf32_allowed': True,
             'cudnn_deterministic': True, 'cudnn_benchmark': False, 'autocast': False,
             'parameter_dtype': 'float32', 'torch_cpu_threads': 2}
REQUIRED_STRUCTURAL_CHECKS = (
    'finite_loss_and_gradients', 'dataset_matches_reference', 'frozen_backbone_unchanged',
    'frozen_backbone_gradients_absent', 'canonical_pretrained_matches_pin',
    'learnable_initialization_matches', 'training_protocol_complete',
    'nonoracle_seen_class_evaluation', 'prediction_ids_and_counts_complete',
    'method_state_valid', 'no_future_or_replay_samples', 'first_batch_inputs_match',
    'optimizer_updates_complete', 'checkpoint_round_trip',
)
REFERENCE_FILES = ('main.py', 'trainer.py', 'methods/base.py', 'utils/factory.py',
                   'utils/function.py', 'utils/toolkit.py', 'dataloaders/data.py',
                   'dataloaders/data_manager.py', 'models/modules/linears.py',
                   *tuple(f'{directory}/{prefix}{method}.{extension}'
                          for method in METHODS.values()
                          for directory, prefix, extension in (('methods', '', 'py'), ('models', 'net_', 'py'),
                                                                ('models', 'vit_', 'py'), ('configs', '', 'unused'))
                          if extension != 'unused'),
                   *tuple(f'configs/{method}/cifar100.json' for method in METHODS.values()))
SOURCE_FILES = ('scripts/low_rank_protocol.py', 'scripts/run_low_rank_validation.py',
                'scripts/run_low_rank_reference.py', 'scripts/cl_efficiency.py', 'scripts/cl_runtime.py',
                'scripts/hide_protocol.py', 'scripts/ease_size_protocol.py', 'scripts/ease_acceptance.py',
                'scripts/ease_reference_utils.py', 'benchmarks/cl_efficiency/low_rank/validation_plan.json',
                'backbone/low_rank_vit.py', 'models/utils/continual_model.py',
                'datasets/seq_cifar100_224.py', 'datasets/seq_cifar100.py',
                'datasets/utils/continual_dataset.py',
                'utils/training.py', 'utils/evaluate.py', 'main.py', 'uv.lock')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def protocol(method):
    if method not in METHODS:
        raise ValueError('Low-rank validation requires one of the four approved methods')
    return {'schema_version': 1, 'mode': 'validation', 'method': method, 'author_method': METHODS[method],
            'tasks': 2, 'dataset': 'seq-cifar100-224', 'dataset_total_tasks': 10,
            'classes_per_task': 10, 'class_order': list(range(100)), 'training_authorized': True,
            'n_epochs': 30 if method == 'cl-lora' else 20, 'batch_size': 64, 'num_workers': 4,
            'seed': 1993, 'vit_size': 'base', 'rank': 10, 'drop_last': False, 'shuffle_training': True,
            'precision': deepcopy(PRECISION), 'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.,
            'hyperparameters': deepcopy(METHOD_SETTINGS[method]), 'lr_scheduler': 'cosine', 'scheduler_mode': 'epoch',
            'phase_rng': 'base_seed + 100000*task_id + 1000*phase_index + local_epoch',
            'sample_order': 'stable original raw global index order before seeded shuffle',
            'train_transforms': ['RandomResizedCrop(224,bilinear)', 'RandomHorizontalFlip(0.5)',
                                 'ToTensor', 'Normalize((0,0,0),(1,1,1))'],
            'test_transforms': ['Resize(224,bilinear)', 'ToTensor',
                                'Normalize((0,0,0),(1,1,1))'],
            'normalization_source': 'Pinned author dataloaders/data.py iCIFAR100 identity normalization',
            'paper_reproduction': False, 'efficiency_comparison': False, 'synthetic_training': False}


def load_plan():
    plan = json.loads(PLAN.read_text())
    if (plan.get('reference_commit') != REFERENCE_COMMIT or plan.get('gpu') != '1'
            or plan.get('training_authorized') is not True or plan.get('wandb_mode') != 'online'
            or plan.get('wandb_entity') != 'laymond1' or plan.get('wandb_project') != 'cl-efficiency'
            or plan.get('protocols') != {method: protocol(method) for method in METHODS}
            or set(plan.get('reference_source_sha256', {})) != set(REFERENCE_FILES)):
        raise ValueError('Low-rank validation plan differs from the approved first-two-task protocol')
    return plan


def verify_reference_source(source):
    source = Path(source).resolve()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True,
                          text=True, check=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=source,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    dirty = [line for line in status if not (line.startswith('?? ') and '/__pycache__/' in line
                                            and line.endswith('.pyc'))]
    expected = load_plan()['reference_source_sha256']
    actual = {name: file_hash(source / name) for name in REFERENCE_FILES}
    if head != REFERENCE_COMMIT or dirty or actual != expected:
        raise ValueError('Independent low-rank source must be clean at the reviewed pinned revision')
    return actual


def source_hashes():
    names = set(SOURCE_FILES)
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / 'models/low_rank_utils').rglob('*.py'))
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / 'models/low_rank_utils').rglob('*.json'))
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / 'models/low_rank_utils').rglob('*.yaml'))
    for method in METHODS:
        for name in (f'models/{method.replace("-", "_")}.py', f'models/config/{method.replace("-", "_")}.yaml'):
            if (ROOT / name).is_file():
                names.add(name)
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def phase_seed(seed, task_id, phase, epoch=0):
    if (phase not in PHASES or type(task_id) is not int or not 0 <= task_id < 2
            or type(epoch) is not int or not 0 <= epoch < 1000):
        raise ValueError('Low-rank RNG coordinates exceed the approved two-task stream')
    return (int(seed) + 100000 * task_id + 1000 * PHASES.index(phase) + epoch) % (2 ** 32)


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
    selected = torch.initial_seed() % (2 ** 32)
    random.seed(selected)
    np.random.seed(selected)


def configure_epoch_loader(loader, seed, task_id, phase, epoch=0):
    import torch
    if loader.drop_last or getattr(loader, 'persistent_workers', False):
        raise ValueError('Shared low-rank loaders require fresh workers and no dropped samples')
    selected = phase_seed(seed, task_id, phase, epoch)
    generator = torch.Generator().manual_seed(selected)
    loader.generator = generator
    if hasattr(loader.sampler, 'generator'):
        loader.sampler.generator = generator
    loader.worker_init_fn = worker_seed
    return selected


def configure_phase_loader_once(loader, seed, task_id, phase, configured):
    """Retain generator advancement when a statistics loader is read repeatedly."""
    coordinates = (seed, task_id, phase)
    if configured.get(loader) == coordinates:
        return False
    configure_epoch_loader(loader, seed, task_id, phase)
    configured[loader] = coordinates
    return True


def common_transforms():
    from torchvision import transforms
    train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), transforms.Normalize((0., 0., 0.), (1., 1., 1.))])
    test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize((0., 0., 0.), (1., 1., 1.))])
    return train, test


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_pretrained_digest(encoder):
    from scripts.ease_size_protocol import canonical_pretrained_digest as digest
    return digest(encoder)


def audit_method_state(network, algorithm, method, task):
    """Validate source algorithm state after its task-finalization hook."""
    import torch
    if method not in METHODS or task not in (1, 2) or network._cur_task != task - 1:
        raise RuntimeError('Low-rank finalized task state has an invalid method or task ID')
    dimension = network.image_encoder.embed_dim
    if dimension != 768 or any(not bool(torch.isfinite(value).all()) for value in network.state_dict().values()):
        raise RuntimeError('Low-rank finalized network state has invalid dimensions or nonfinite values')
    if method == 'ewc-lora':
        values = list(algorithm.omega_W)
        if (int(algorithm.count_updates) != task or len(values) != 24
                or any(tuple(value.shape) != (dimension, dimension) or not bool(torch.isfinite(torch.as_tensor(value)).all())
                       or bool((torch.as_tensor(value) < 0).any()) for value in values)):
            raise RuntimeError('EWCLoRA finalized Fisher count, shapes or values are invalid')
    elif method == 'inf-lora':
        values, kinds = list(algorithm.feature_list), list(algorithm.project_type)
        if (len(values) != 12 or len(kinds) != 12 or any(kind not in ('remove', 'retain') for kind in kinds)
                or any(len(value.shape) != 2 or value.shape[0] != dimension or value.shape[1] > dimension // 2
                       or not bool(torch.isfinite(torch.as_tensor(value)).all()) for value in values)):
            raise RuntimeError('InfLoRA finalized DualGPM bases or projection types are invalid')
    elif method == 'cl-lora':
        if (tuple(network.fc.weight.shape) != (10 * task, dimension * task)
                or tuple(network.proxy_fc.weight.shape) != (10, dimension)
                or bool((network.fc.weight.detach().norm(dim=1) == 0).any())):
            raise RuntimeError('CLLoRA finalized prototype or proxy classifier shapes/values are invalid')
    elif method == 'sd-lora':
        if len(network.image_encoder.scaling_factor) != 1 or len(network.image_encoder.scaling_factor_prev) != 20:
            raise RuntimeError('SDLoRA finalized historical scale catalog is invalid')
    return True


def learned_state(network):
    """Every noncanonical parameter/buffer is shared, including all ten task slots."""
    from scripts.ease_size_protocol import canonical_pretrained_state
    canonical = set(canonical_pretrained_state(network.image_encoder))
    return {name: value for name, value in network.state_dict().items()
            if not (name.startswith('image_encoder.') and name[len('image_encoder.'):] in canonical)}


def validate_initial_state(catalog, method):
    import torch
    metadata = catalog.get('metadata', {}) if isinstance(catalog, dict) else {}
    state = catalog.get('learned_state', {}) if isinstance(catalog, dict) else {}
    if (metadata.get('method') != method or metadata.get('reference_commit') != REFERENCE_COMMIT
            or metadata.get('checkpoint_sha256') != CHECKPOINT_SHA256 or not state
            or any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
                   or not bool(torch.isfinite(value).all()) for name, value in state.items())):
        raise ValueError('Low-rank learned initialization catalog is incompatible or nonfinite')
    return catalog


def load_initial_state(network, catalog, method):
    validate_initial_state(catalog, method)
    actual = learned_state(network)
    expected = catalog['learned_state']
    if set(actual) != set(expected) or any(actual[name].shape != expected[name].shape for name in actual):
        raise ValueError('Native and author initialization state names or shapes differ')
    network.load_state_dict(expected, strict=False)
    pin = tensor_digest(learned_state(network))
    if pin != tensor_digest(expected):
        raise RuntimeError('Native learnable initialization did not transfer exactly')
    return pin


def counts_from_predictions(predictions, labels, *, past_classes, seen_classes):
    from scripts.ease_acceptance import counts_from_predictions as count
    return count(predictions, labels, past_classes=past_classes, seen_classes=seen_classes)


def reference_policy(rows):
    from scripts.hide_acceptance import reference_policy as policy
    return policy(rows)


def evaluate_acceptance(rows, reference_rows, *, structural_checks):
    """Exact four-count gate, with independent method-specific structural checks."""
    from scripts.ease_acceptance import REQUIRED_STRUCTURAL_CHECKS as helper_gates
    from scripts.ease_acceptance import evaluate_acceptance as exact
    checks = deepcopy(structural_checks) if isinstance(structural_checks, dict) else {}
    failures = [name for name in REQUIRED_STRUCTURAL_CHECKS if checks.get(name) is not True]
    helper = {name: True for name in helper_gates}
    helper.update(checks)
    report = exact(rows, reference_policy(reference_rows), structural_checks=helper)
    report['structural_checks'] = checks
    report['structural_failures'] = list(dict.fromkeys(failures + [name for name, value in checks.items() if value is not True]))
    report['validation_errors'] = [error for error in report['validation_errors']
                                   if not error.startswith('missing required structural check')]
    if report['structural_failures']:
        report['status'] = 'fail'
        report['validation_errors'].append('Missing or failed low-rank structural gates')
    return report


def validate_prediction_archive(path, task, expected_ids=None, expected_labels=None):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'predictions', 'labels', 'global_ids'}:
            raise ValueError('Prediction archive must contain predictions, labels and global_ids')
        predictions, labels, ids = (archive[name] for name in ('predictions', 'labels', 'global_ids'))
        count = task * 1000
        if (task not in (1, 2) or any(value.shape != (count,) or value.dtype.kind not in 'iu'
                                    for value in (predictions, labels, ids))
                or len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= 10000)
                or not np.all(ids[1:] > ids[:-1])
                or any(int(np.count_nonzero(labels == label)) != 100 for label in range(task * 10))
                or (expected_ids is not None and not np.array_equal(ids, expected_ids))
                or (expected_labels is not None and not np.array_equal(labels, expected_labels))):
            raise ValueError('Prediction IDs, stable order or full CIFAR100 class counts are invalid')
        return counts_from_predictions(predictions, labels, past_classes=(task - 1) * 10,
                                       seen_classes=task * 10)


def native_command(method, checkpoint=CHECKPOINT, data_root='/mnt/sdab1/share/data', output=None):
    settings = protocol(method)
    hyper = settings['hyperparameters']
    command = ['--model', method, '--dataset', settings['dataset'], '--model_config', 'default',
               '--backbone', 'low-rank-vit', '--lora_vit_size', 'base', '--lora_checkpoint', str(checkpoint),
               '--lora_rank', '10', '--n_epochs', str(settings['n_epochs']), '--stop_after', '2',
               '--batch_size', '64', '--num_workers', '4', '--seed', '1993', '--permute_classes', '0',
               '--device', 'cuda:0', '--base_path', str(data_root), '--non_verbose', '1', '--debug_mode', '0',
               '--enable_other_metrics', '0', '--eval_future', '0', '--code_optimization', '0',
               '--noise_rate', '0', '--validation', '0', '--label_perc', '1', '--label_perc_by_class', '1',
               '--lr', str(hyper['lr']), '--lora_fc_lr', str(hyper['fc_lr']), '--optimizer', hyper['optimizer'],
               '--optim_wd', str(hyper['weight_decay']), '--lr_scheduler', 'cosine', '--scheduler_mode', 'epoch',
               '--wandb_entity', 'laymond1', '--wandb_project', 'cl-efficiency']
    if output is not None:
        command.extend(['--results_path', str(Path(output) / 'native_logs')])
    return command


def completion_checks(record, method):
    settings = protocol(method)
    rows = record.get('epoch_results', [])
    expected = {(task, epoch) for task in (1, 2) for epoch in range(1, settings['n_epochs'] + 1)}
    if (len(rows) != len(expected) or {(row.get('task'), row.get('epoch')) for row in rows} != expected
            or any(row.get('steps') != 79 or row.get('samples') != 5000 for row in rows)):
        raise RuntimeError('Low-rank epochs do not cover the full approved two-task stream')
    counts = record.get('counts', {})
    observes = settings['n_epochs'] * 79 * 2
    updates = settings['n_epochs'] * 79 * (3 if method == 'cl-lora' else 2)
    if counts.get('observe_calls') != observes or counts.get('optimizer_updates') != updates:
        raise RuntimeError('Low-rank observe calls or optimizer updates are incomplete or double counted')
    expected_extra = {'ewc-lora': (0, 10000), 'sd-lora': (0, 0),
                      'inf-lora': (10000, 10000), 'cl-lora': (0, 15000)}[method]
    if (counts.get('task_begin_extra_samples'), counts.get('task_end_extra_samples')) != expected_extra:
        raise RuntimeError('Low-rank source task statistics or prototype data passes are incomplete or repeated')
    if len(record.get('task_results', [])) != 2 or len(record.get('first_batch_audits', [])) != (3 if method == 'cl-lora' else 2):
        raise RuntimeError('Low-rank full predictions or first-batch update audits are missing')
    return True
