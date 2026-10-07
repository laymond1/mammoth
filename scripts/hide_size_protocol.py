"""Explicit Small/Tiny HiDe protocols using existing pinned shared weights.

Import and dry preparation use only the standard library. The configuration
plan stays immutable during execution; progress lives in execution_plan.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/cl_efficiency/hide_sizes/validation_plan.json'
SOURCE_FILES = ('scripts/hide_size_protocol.py', 'benchmarks/cl_efficiency/hide_sizes/validation_plan.json')
VARIANTS = {'small': (384, 6), 'tiny': (192, 3)}
REFERENCE_COMMIT = '0ed433334ff2cfb206c5e2899651c3044ec85f92'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require(actual, expected, scope):
    if any(type(actual.get(key)) is not type(value) or actual.get(key) != value for key, value in expected.items()):
        raise ValueError('HiDe size ' + scope + ' differs from the authorized protocol')


def load_plan(variant=None):
    plan = json.loads(PLAN.read_text())
    _require(plan, {'schema_version': 1, 'training_authorized': True, 'model': 'hide-prompt',
                    'reference_commit': REFERENCE_COMMIT, 'full_cl_authorized': False,
                    'additional_seeds_authorized': False, 'variant_order': ['small', 'tiny']}, 'approval')
    if [row.get('name') for row in plan.get('variants', [])] != ['small', 'tiny']:
        raise ValueError('HiDe size plan requires exactly Small then Tiny')
    _require(plan['common'], {'dataset': 'seq-cifar100-224', 'dataset_total_tasks': 10,
                             'classes_per_task': 10, 'seed': 1993, 'num_workers': 4,
                             'batch_size': 64, 'gpu': '1', 'permute_classes': False,
                             'class_order': list(range(100)), 'prompt_length': 5,
                             'prompt_layers': [0, 1, 2, 3, 4], 'mlp_structure': [2],
                             'centroids': 10, 'prompt_momentum': .01,
                             'data_root': '/mnt/sdab1/share/data',
                             'cache_root': '/mnt/sdab1/userHome/wslim/.cache',
                             'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'}, 'common settings')
    from scripts.hide_protocol import FULL_CL_PRECISION
    _require(plan['common'], {'precision': FULL_CL_PRECISION}, 'precision')
    _require(plan['validation'], {'tasks': 2, 'dataset_config': 'hide_prompt', 'tii_epochs': 20,
                                 'prompt_epochs': 50, 'n_epochs': 70, 'ca_epochs': 30,
                                 'batch_size': 64, 'training_authorized': True,
                                 'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.}, 'validation')
    _require(plan['efficiency'], {'tasks': 1, 'dataset_config': 'ease', 'tii_epochs': 1,
                                 'prompt_epochs': 1, 'outer_epochs_per_task': 2, 'n_epochs': 2,
                                 'ca_epochs': 30, 'batch_size': 64, 'training_authorized': True,
                                 'measurement_repeats': 3, 'flops_runs': 1, 'ca_runs_task1': False}, 'pilot')
    if variant is not None:
        if variant not in VARIANTS:
            raise ValueError('Unapproved HiDe size variant')
        selected = next(row for row in plan['variants'] if row['name'] == variant)
        plan['common'].update(selected)
        plan['variant'] = variant
    return plan


def load_variant(name, verify_checkpoint=True):
    if name not in VARIANTS:
        raise ValueError('Unapproved HiDe size variant')
    plan = load_plan()
    result = next(row for row in plan['variants'] if row['name'] == name).copy()
    dimension, heads = VARIANTS[name]
    _require(result, {'name': name, 'vit_size': name, 'feature_dim': dimension, 'num_heads': heads,
                      'depth': 12, 'patch_size': 16,
                      'checkpoint_tag': f'vit_{name}_patch16_224.augreg_in21k_ft_in1k'}, 'geometry')
    for key in ('checkpoint_sha256', 'canonical_pretrained_sha256'):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(letter not in '0123456789abcdef' for letter in value):
            raise ValueError('HiDe size pretrained weights require SHA256 pins')
    checkpoint = Path(result['checkpoint_file']).expanduser().resolve()
    expected_root = Path(plan['common']['cache_root']).resolve() / 'torch/hub/checkpoints'
    if checkpoint.parent != expected_root or checkpoint.suffix != '.npz':
        raise ValueError('HiDe size checkpoint must reuse the agreed shared NPZ cache')
    if verify_checkpoint and (not checkpoint.is_file() or file_hash(checkpoint) != result['checkpoint_sha256']):
        raise ValueError('HiDe size checkpoint is missing or changed')
    result['checkpoint_file'] = str(checkpoint)
    return result


def size_protocol(mode, name):
    if mode not in ('validation', 'smoke', 'efficiency'):
        raise ValueError('HiDe size approval covers validation and first-task pilot only')
    from scripts.hide_protocol import protocol
    settings = protocol(mode)
    selected = load_variant(name, verify_checkpoint=False)
    settings.update(variant=name, vit_size=name, feature_dim=selected['feature_dim'],
                    num_heads=selected['num_heads'], prompt_length=5, prompt_layers=[0, 1, 2, 3, 4],
                    mlp_structure=[2], approval='User: Small then Tiny, first two-task validation and first-task efficiency pilot only.',
                    paper_reproduction=False)
    return settings
