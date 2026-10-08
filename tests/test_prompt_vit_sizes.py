"""Local L2P/DualPrompt size ports, including actual pinned CPU weights.

No downloads, CUDA initialization, W&B or Slack. Cost/accuracy experiments are
separate; the synthetic smoke only verifies the Mammoth two-task lifecycle.
"""
from argparse import ArgumentParser
import gc
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from models.dualprompt import DualPrompt
from models.dualprompt_utils import model as dual_model
from models.dualprompt_utils import vision_transformer as dual_vit
from models.l2p import L2P
from models.l2p_utils import l2p_model, pretrained
from models.l2p_utils import vit_prompt as l2p_vit
from models.low_rank_utils.factory import CHECKPOINTS, SIZES


def arguments(method, size, checkpoint=None):
    cls = L2P if method == 'l2p' else DualPrompt
    args = cls.get_parser(ArgumentParser()).parse_args([])
    setattr(args, f'{method}_vit_size', size)
    args.pretrained_checkpoint = checkpoint
    args.pretrained = False
    return args


@pytest.fixture(autouse=True)
def native_cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Size preparation tests must not fetch weights or use CUDA')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)
    gc.collect()


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
def test_parser_size_defaults_keep_base_and_prompt_recipe(method):
    args = arguments(method, 'base')
    assert getattr(args, f'{method}_vit_size') == 'base'
    assert args.length == 5
    if method == 'l2p':
        assert (args.pool_size_l2p, args.top_k, args.head_type) == (10, 5, 'prompt')
    else:
        assert (args.size, args.top_k, args.g_prompt_length) == (10, 1, 5)
        assert args.g_prompt_layer_idx == [0, 1] and args.e_prompt_layer_idx == [2, 3, 4]


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
@pytest.mark.parametrize('size', ['tiny', 'small', 'base'])
def test_factory_geometry_keeps_depth_and_head_dimension(monkeypatch, method, size):
    module = l2p_vit if method == 'l2p' else dual_vit
    calls = []
    monkeypatch.setattr(module, 'create_vision_transformer', lambda *a, **k: calls.append((a, k)))
    factory = module.vit_base_patch16_224_l2p if method == 'l2p' else module.vit_base_patch16_224_dualprompt
    factory(pretrained=False, vit_size=size, num_classes=100)
    width, heads = SIZES[size]
    positional, kwargs = calls[0]
    assert (kwargs['embed_dim'], kwargs['num_heads'], kwargs['depth'], kwargs['patch_size']) == (width, heads, 12, 16)
    assert width // heads == 64 and kwargs['num_classes'] == 100
    expected_variant = 'vit_base_patch16_224_in21k_fn_in1k_old' if size == 'base' else f'vit_{size}_patch16_224.augreg_in21k_ft_in1k'
    assert positional[0] == expected_variant and kwargs['pretrained'] is False


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
@pytest.mark.parametrize('size', ['tiny', 'small'])
def test_non_base_network_requires_explicit_local_weights_before_construction(monkeypatch, method, size):
    module = l2p_model if method == 'l2p' else dual_model
    def forbidden(*args, **kwargs):
        pytest.fail('No model or dataset should be allocated before the missing checkpoint rejection')
    monkeypatch.setattr(module, 'vit_base_patch16_224_l2p' if method == 'l2p' else 'vit_base_patch16_224_dualprompt', forbidden)
    if method == 'l2p':
        monkeypatch.setattr(module, 'get_dataset', forbidden)
    with pytest.raises(ValueError, match='local pretrained_checkpoint'):
        module.L2PModel(arguments(method, size)) if method == 'l2p' else module.Model(arguments(method, size), 100)


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
def test_unpinned_local_weights_fail_before_encoder_construction(monkeypatch, tmp_path, method):
    checkpoint = tmp_path / 'incorrect.npz'
    checkpoint.write_bytes(b'wrong checkpoint bytes')
    module = l2p_model if method == 'l2p' else dual_model
    monkeypatch.setattr(module, 'vit_base_patch16_224_l2p' if method == 'l2p' else 'vit_base_patch16_224_dualprompt',
                        lambda *a, **k: pytest.fail('Unpinned weights must fail before construction'))
    with pytest.raises(ValueError, match='pinned AugReg'):
        module.L2PModel(arguments(method, 'base', str(checkpoint))) if method == 'l2p' else module.Model(arguments(method, 'base', str(checkpoint)), 100)


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
@pytest.mark.parametrize('size', ['tiny', 'small', 'base'])
def test_actual_local_weights_match_both_encoders_without_prompt_changes(monkeypatch, method, size):
    checkpoint = CHECKPOINTS[size]['file']
    assert Path(checkpoint).is_file(), 'Shared size checkpoint is required for this CPU preparation test'
    args = arguments(method, size, checkpoint)
    if method == 'l2p':
        monkeypatch.setattr(l2p_model, 'get_dataset', lambda args: SimpleNamespace(N_CLASSES=100))
        network = l2p_model.L2PModel(args)
    else:
        network = dual_model.Model(args, 100)
    width, heads = SIZES[size]
    assert network.vit_size == size
    assert network.initial_canonical_pretrained_branches == {'query': CHECKPOINTS[size]['canonical_sha256'],
                                                           'prompt': CHECKPOINTS[size]['canonical_sha256']}
    assert network.initial_canonical_pretrained_sha256 == CHECKPOINTS[size]['canonical_sha256']
    assert network.pretrained_sha256 == CHECKPOINTS[size]['sha256']
    for branch in (network.original_model, network.model):
        assert len(branch.blocks) == 12 and (branch.embed_dim, branch.num_heads) == (width, heads)
        assert branch.head.weight.shape == (100, width)
    assert not any(p.requires_grad for p in network.original_model.parameters())
    assert not any(p.requires_grad for p in network.model.blocks.parameters())
    if method == 'l2p':
        assert network.model.prompt.prompt.shape == (10, 5, width)
        assert network.model.prompt.prompt_key.shape == (10, width)
        assert network.model.pos_embed.shape == (1, 222, width)
        assert torch.equal(network.model.pos_embed[:, 1:26], network.original_model.pos_embed[:, :1].expand(-1, 25, -1))
    else:
        assert network.model.pos_embed.shape == (1, 197, width)
        assert network.model.g_prompt.shape == (2, 2, 5, heads, 64)
        assert network.model.e_prompt.prompt.shape == (3, 2, 10, 5, heads, 64)
        assert network.model.e_prompt.prompt_key.shape == (10, width)
    network.eval()
    with torch.no_grad():
        images = torch.zeros((1, 3, 224, 224))
        logits = network(images) if method == 'l2p' else network(images, task_id=-1)
    assert logits.shape == (1, 100) and torch.isfinite(logits).all()
    assert pretrained.verify_pretrained_branches(network.original_model, network.model, CHECKPOINTS[size], l2p=method == 'l2p') == network.initial_canonical_pretrained_branches
    # Check that acceptance compares both branches, rather than only one hash.
    with torch.no_grad():
        network.model.blocks[0].norm1.bias[0].add_(0.25)
    with pytest.raises(ValueError, match='Query or prompt encoder'):
        pretrained.verify_pretrained_branches(network.original_model, network.model, CHECKPOINTS[size], l2p=method == 'l2p')
    del network


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
@pytest.mark.parametrize('size', ['tiny', 'small'])
def test_factory_non_base_pretrained_cannot_download(method, size):
    factory = l2p_vit.vit_base_patch16_224_l2p if method == 'l2p' else dual_vit.vit_base_patch16_224_dualprompt
    with pytest.raises(ValueError, match='explicit local NPZ'):
        factory(pretrained=True, vit_size=size)


def native_arguments(monkeypatch, method, size):
    import main
    command = ['--model', method, '--dataset', 'seq-cifar100-224', '--backbone', 'vit',
               '--model_config', 'default', '--dataset_config', 'ease', '--pretrain_type', 'in21k-ft-in1k',
               '--pretrained', '0', '--pretrained_checkpoint', CHECKPOINTS[size]['file'],
               f'--{method}_vit_size', size, '--batchwise_prompt', '0',
               '--lr', '.03', '--n_epochs', '2', '--batch_size', '2', '--num_workers', '0',
               '--stop_after', '2', '--device', 'cpu', '--seed', '1993']
    monkeypatch.setattr(sys, 'argv', ['main.py', *command])
    args = main.parse_args(cmd=command)
    args.nowand = True  # This workspace derives nowand at runtime; it is not a CLI argument.
    return args


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
@pytest.mark.parametrize('size', ['tiny', 'small', 'base'])
def test_staged_cli_preserves_explicit_size_and_shared_local_checkpoint(monkeypatch, method, size):
    import main
    monkeypatch.setattr(main, 'initialize', lambda *a, **k: pytest.fail('Staged parse must not initialize a model'))
    args = native_arguments(monkeypatch, method, size)
    assert getattr(args, f'{method}_vit_size') == size
    assert args.pretrained is False and args.pretrained_checkpoint == CHECKPOINTS[size]['file']
    assert args.batchwise_prompt is False and args.dataset_config == 'ease'
    assert args.n_epochs == 2 and args.stop_after == 2


class SyntheticTasks:
    """No dataset I/O: two tiny batches exercise the actual Mammoth wrappers."""
    N_CLASSES = 100
    N_TASKS = 10
    N_CLASSES_PER_TASK = 10
    SETTING = 'class-il'

    def get_offsets(self, task):
        return task * 10, (task + 1) * 10


@pytest.mark.parametrize('method', ['l2p', 'dualprompt'])
def test_tiny_cpu_two_task_wrapper_smoke_preserves_encoder_and_task_transition(monkeypatch, method):
    args = native_arguments(monkeypatch, method, 'tiny')
    dataset = SyntheticTasks()
    monkeypatch.setattr(l2p_model, 'get_dataset', lambda args: dataset)
    torch.manual_seed(1993)
    cls = L2P if method == 'l2p' else DualPrompt
    model = cls(None, torch.nn.functional.cross_entropy, args, None, dataset=dataset).to('cpu')
    assert args.lr == pytest.approx(.03 * 2 / 256)
    initial = model.net.initial_canonical_pretrained_branches
    generator = torch.Generator().manual_seed(2993)
    # Synthetic, fixed inputs with common mean/std .5; this is no accuracy or cost experiment.
    images = (torch.rand((2, 3, 224, 224), generator=generator) - .5) / .5
    for task in range(2):
        model.train()
        previous_slot = None
        if method == 'dualprompt' and task:
            previous_slot = model.net.model.e_prompt.prompt[:, :, :1].detach().clone()
        model.meta_begin_task(dataset)
        model.meta_begin_epoch(0, dataset)
        assert (model.current_task, model.n_past_classes, model.n_seen_classes) == (task, task * 10, (task + 1) * 10)
        assert not model.net.original_model.training
        if previous_slot is not None:
            assert torch.equal(model.net.model.e_prompt.prompt[:, :, 1:2], previous_slot)
        before_head = model.net.model.head.weight.detach().clone()
        prompt = model.net.model.prompt.prompt if method == 'l2p' else model.net.model.e_prompt.prompt
        before_prompt = prompt.detach().clone()
        loss = model.meta_observe(images, torch.tensor([task * 10, task * 10 + 1]), images, epoch=0)
        assert isinstance(loss, float) and math.isfinite(loss)
        assert model._task_iteration == 1 and model._epoch_iteration == 1
        assert not torch.equal(before_head, model.net.model.head.weight)
        assert not torch.equal(before_prompt, prompt)
        assert all(parameter.grad is None for parameter in model.net.original_model.parameters())
        assert all(parameter.grad is None for parameter in model.net.model.blocks.parameters())
        model.meta_end_epoch(0, dataset)
        model.eval()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (2, (task + 1) * 10) and torch.isfinite(logits).all()
        assert pretrained.verify_pretrained_branches(model.net.original_model, model.net.model, CHECKPOINTS['tiny'], l2p=method == 'l2p') == initial
        model.meta_end_task(dataset)
    assert model.current_task == 2
