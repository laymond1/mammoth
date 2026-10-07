"""Meaningful synthetic checks for the independent pinned-author runner."""
from argparse import Namespace
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.hide_reference_utils import (PairLoader, build_author_branches, learned_state_digest,
    load_author_code, load_initial_state, save_initial_state, set_author_phase, verify_author_source)
from scripts.run_hide_reference import author_args, observed_optimizer, per_class_loaders

SOURCE = Path('/tmp/cl-hide-official-20261007')


@pytest.fixture(scope='module')
def author():
    if not SOURCE.is_dir():
        pytest.skip('Pinned author checkout unavailable; runtime refuses another source')
    verify_author_source(SOURCE)
    return load_author_code(SOURCE)


def small_branches(author):
    return build_author_branches(author, embed_dim=8, depth=2, num_heads=2, img_size=16, patch_size=8)


def test_author_independent_core_and_training_loss(author):
    branches = small_branches(author)
    assert type(branches['tii']).__module__ == 'hide_pinned_author'
    assert type(branches['prompt'].e_prompt).__module__ == 'hide_pinned_author'
    set_author_phase(branches, 'tii')
    frozen = {name: value.detach().clone() for name, value in branches['tii'].named_parameters()
              if not value.requires_grad}
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 3])
    loader = PairLoader(DataLoader(TensorDataset(images, labels), batch_size=2))
    optimizer = torch.optim.Adam([p for p in branches['tii'].parameters() if p.requires_grad], lr=.001)
    checks = {}
    args = author_args({'seed': 1993, 'batch_size': 2, 'tii_epochs': 1, 'prompt_epochs': 1, 'ca_epochs': 1}, 'tii')
    with observed_optimizer(optimizer, branches, checks):
        result = author.engines['tii']['train_one_epoch'](branches['tii'], torch.nn.CrossEntropyLoss(),
            loader, optimizer, torch.device('cpu'), 0, 1., task_id=0, class_mask=[list(range(10))], args=args)
    assert np.isfinite(result['Loss'])
    assert checks == {'finite_loss_and_gradients': True, 'frozen_backbone_gradients_absent': True}
    assert all(torch.equal(value, dict(branches['tii'].named_parameters())[name]) for name, value in frozen.items())
    set_author_phase(branches, 'prompt')
    assert all(p.grad is None for p in branches['tii'].parameters())
    logits = branches['prompt'](images, task_id=0, train=True)['logits'][:, :10]
    loss = torch.nn.functional.cross_entropy(logits, labels) + author.engines['prompt']['orth_loss'](
        branches['prompt'](images, task_id=0, train=True)['pre_logits'], labels, torch.device('cpu'), args)
    loss.backward()
    assert branches['prompt'].e_prompt.prompt.grad is not None
    assert all(p.grad is None for p in branches['prompt'].blocks.parameters())


def test_matched_learned_state_round_trip(author, tmp_path):
    branches = small_branches(author)
    path = tmp_path / 'matched.pt'
    saved = save_initial_state(branches, path, 'a' * 64)
    expected = saved['learned_sha256']
    other = small_branches(author)
    assert learned_state_digest(other['tii'], 'tii') != expected['tii']
    result = load_initial_state(other, path)
    assert result == saved
    assert {name: learned_state_digest(module, name) for name, module in other.items()} == expected
    with pytest.raises(FileExistsError):
        save_initial_state(other, path, 'a' * 64)


@pytest.mark.parametrize('field', ['names', 'hash', 'shape', 'nan'])
def test_matched_state_rejects_invalid_tensors(author, tmp_path, field):
    branches = small_branches(author)
    path = tmp_path / 'matched.pt'
    save_initial_state(branches, path, 'a' * 64)
    state = torch.load(path, weights_only=True)
    if field == 'names':
        state['branches']['tii'].pop('head.bias')
    elif field == 'hash':
        state['learned_sha256']['tii'] = 'b' * 64
    elif field == 'shape':
        state['branches']['tii']['head.bias'] = torch.zeros(4)
    else:
        state['branches']['tii']['head.bias'][0] = float('nan')
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_initial_state(branches, path)


def test_pair_loader_limits_batches_without_overconsuming():
    subject = types.SimpleNamespace(steps=0, training_examples=0)
    loader = DataLoader(TensorDataset(torch.arange(8), torch.arange(8), torch.arange(8)), batch_size=2)
    pairs = PairLoader(loader, subject, max_batches=2)
    assert len(pairs) == 2
    assert len(list(pairs)) == 2
    assert subject.steps == 2 and subject.training_examples == 4


def test_reference_dry_run_has_no_ml_or_credentials():
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts/run_hide_reference.py'
    result = subprocess.run([sys.executable, '-S', str(script)], text=True, capture_output=True, check=True)
    import json
    assert json.loads(result.stdout)['executed'] is False


def test_source_pin_hashes_and_ast_no_native_algorithm(author):
    hashes = verify_author_source(SOURCE)
    assert len(hashes) == 15
    assert author.engines['tii']['train_one_epoch'].__code__.co_filename.startswith(str(SOURCE))
    assert author.engines['prompt']['orth_loss'].__code__.co_filename.startswith(str(SOURCE))
    assert author.VisionTransformer.forward.__code__.co_filename.startswith(str(SOURCE))


def test_independent_per_class_statistics_are_pinned(author):
    module = small_branches(author)['tii']
    images = torch.randn(20, 3, 16, 16)
    labels = torch.zeros(20, dtype=torch.long)
    loader = PairLoader(DataLoader(TensorDataset(images, labels), batch_size=7, shuffle=False))
    args = author_args({'seed': 1993, 'batch_size': 7, 'tii_epochs': 1, 'prompt_epochs': 1, 'ca_epochs': 1}, 'tii')
    args.n_centroids = 3
    engine = author.engines['tii']
    engine['cls_mean'].clear()
    engine['cls_cov'].clear()
    engine['_compute_mean'](module, {0: {'train': loader}}, torch.device('cpu'), [0], args)
    first = [value.clone() for value in engine['cls_mean'][0]]
    torch.manual_seed(8124)
    np.random.seed(7421)
    engine['_compute_mean'](module, {0: {'train': loader}}, torch.device('cpu'), [0], args)
    assert all(torch.equal(a, b) for a, b in zip(first, engine['cls_mean'][0]))
    assert len(engine['cls_cov'][0]) == 3
    engine['cls_mean'].clear()
    engine['cls_cov'].clear()


def test_author_callback_observes_actual_first_batch_before_update(author):
    branches = small_branches(author)
    set_author_phase(branches, 'tii')
    images, labels = torch.randn(2, 3, 16, 16), torch.tensor([0, 1])
    loader = PairLoader(DataLoader(TensorDataset(images, labels), batch_size=2))
    optimizer = torch.optim.Adam([p for p in branches['tii'].parameters() if p.requires_grad], lr=.001)
    args = author_args({'seed': 1993, 'batch_size': 2, 'tii_epochs': 1, 'prompt_epochs': 1, 'ca_epochs': 1}, 'tii')
    rows = []
    initial = learned_state_digest(branches['tii'], 'tii')
    def callback(inputs, targets, loss, module, subject):
        rows.append({'input': torch.equal(inputs, images), 'labels': torch.equal(targets, labels),
                     'finite_loss': bool(torch.isfinite(loss)), 'before': learned_state_digest(module, 'tii'),
                     'has_gradient': any(p.grad is not None for p in module.parameters())})
    engine = author.engines['tii']
    engine['_audit_batch'] = callback
    try:
        engine['train_one_epoch'](branches['tii'], torch.nn.CrossEntropyLoss(), loader, optimizer,
            torch.device('cpu'), 0, 1., task_id=0, class_mask=[list(range(10))], args=args)
    finally:
        engine['_audit_batch'] = lambda *items: None
    assert rows == [{'input': True, 'labels': True, 'finite_loss': True, 'before': initial, 'has_gradient': True}]
    assert learned_state_digest(branches['tii'], 'tii') != initial
