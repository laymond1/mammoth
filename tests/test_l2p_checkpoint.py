"""Local L2P weight selection and prompt positions, without image computation."""

from argparse import ArgumentParser
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from models.l2p import L2P
from models.l2p_utils import l2p_model
from models.l2p_utils.vit_prompt import resize_pos_embed


def arguments(**updates):
    values = dict(
        length=5, embedding_key='cls', prompt_key_init='uniform', prompt_pool=True,
        prompt_key=True, pool_size_l2p=10, top_k=5, batchwise_prompt=False,
        head_type='prompt', use_prompt_mask=False, use_original_ckpt=False,
        freeze=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'],
        pretrained_checkpoint=None,
    )
    values.update(updates)
    return SimpleNamespace(**values)


class NoForwardViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)
        self.prompt = nn.Linear(2, 2)
        self.load_pretrained = Mock()

    def forward(self, *args, **kwargs):
        pytest.fail('Checkpoint tests must not run image forwards')


@pytest.fixture
def constructors(monkeypatch):
    created = []
    calls = []

    def factory(**kwargs):
        model = NoForwardViT()
        calls.append(kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(l2p_model, 'get_dataset', Mock(return_value=SimpleNamespace(N_CLASSES=100)))
    monkeypatch.setattr(l2p_model, 'vit_base_patch16_224_l2p', factory)
    return calls, created


def test_local_checkpoint_loads_both_backbones_and_preserves_freezing(tmp_path, constructors):
    checkpoint = tmp_path / 'shared.npz'
    checkpoint.touch()
    calls, created = constructors
    net = l2p_model.L2PModel(arguments(pretrained_checkpoint=str(checkpoint)))
    assert [call['pretrained'] for call in calls] == [False, False]
    assert all(call['num_classes'] == 100 for call in calls)
    for model in created:
        model.load_pretrained.assert_called_once_with(str(checkpoint))
    assert net.pretrained_checkpoint == str(checkpoint)
    assert not net.original_model.training
    assert not any(parameter.requires_grad for parameter in net.original_model.parameters())
    assert not any(parameter.requires_grad for parameter in net.model.blocks.parameters())
    assert all(parameter.requires_grad for parameter in net.model.head.parameters())
    assert all(parameter.requires_grad for parameter in net.model.prompt.parameters())
    assert calls[1]['prompt_length'] == 5 and calls[1]['top_k'] == 5


@pytest.mark.parametrize('legacy_arguments', [False, True])
def test_default_path_preserves_existing_pretrained_behavior(constructors, legacy_arguments):
    args = arguments()
    if legacy_arguments:
        del args.pretrained_checkpoint
    calls, created = constructors
    net = l2p_model.L2PModel(args)
    assert [call['pretrained'] for call in calls] == [True, True]
    assert net.pretrained_checkpoint is None
    assert all(not model.load_pretrained.called for model in created)


def test_local_loader_failure_propagates_without_pretrained_fallback(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'malformed.npz'
    checkpoint.touch()
    created = [NoForwardViT(), NoForwardViT()]
    created[0].load_pretrained.side_effect = ValueError('invalid NPZ')
    factory = Mock(side_effect=created)
    monkeypatch.setattr(l2p_model, 'vit_base_patch16_224_l2p', factory)
    monkeypatch.setattr(l2p_model, 'get_dataset', lambda args: SimpleNamespace(N_CLASSES=100))
    with pytest.raises(ValueError, match='invalid NPZ'):
        l2p_model.L2PModel(arguments(pretrained_checkpoint=str(checkpoint)))
    assert factory.call_count == 2
    assert all(call.kwargs['pretrained'] is False for call in factory.call_args_list)
    created[1].load_pretrained.assert_not_called()


def test_conflicting_original_checkpoint_fails_before_model_creation(constructors):
    with pytest.raises(ValueError, match='cannot be combined'):
        l2p_model.L2PModel(arguments(pretrained_checkpoint='local.npz', use_original_ckpt=True))
    assert constructors == ([], [])
    l2p_model.get_dataset.assert_not_called()


@pytest.mark.parametrize('checkpoint_kind', ['missing', 'directory', 'non_npz'])
def test_invalid_local_checkpoint_cannot_fall_back_to_download(tmp_path, constructors, checkpoint_kind):
    checkpoint = tmp_path / 'invalid.npz'
    error = FileNotFoundError
    if checkpoint_kind == 'directory':
        checkpoint.mkdir()
    elif checkpoint_kind == 'non_npz':
        checkpoint = tmp_path / 'invalid.pt'
        checkpoint.touch()
        error = ValueError
    with pytest.raises(error):
        l2p_model.L2PModel(arguments(pretrained_checkpoint=str(checkpoint)))
    assert constructors == ([], [])
    l2p_model.get_dataset.assert_not_called()


def test_parser_exposes_optional_checkpoint_without_changing_default():
    parser = L2P.get_parser(ArgumentParser())
    assert parser.parse_args([]).pretrained_checkpoint is None
    assert parser.parse_args(['--pretrained_checkpoint', '/tmp/shared.npz']).pretrained_checkpoint == '/tmp/shared.npz'


@pytest.mark.parametrize('prompt_tokens', [0, 25])
def test_position_expansion_preserves_pretrained_class_and_patch_positions(prompt_tokens):
    positions = torch.arange(197 * 3, dtype=torch.float32).reshape(1, 197, 3)
    target = torch.empty(1, 197 + prompt_tokens, 3)
    expanded = resize_pos_embed(positions, target, num_prefix_tokens=1, gs_new=(14, 14))
    assert torch.equal(expanded[:, :1], positions[:, :1])
    assert torch.equal(expanded[:, -196:], positions[:, 1:])
    assert torch.equal(expanded[:, :1 + prompt_tokens], positions[:, :1].expand(-1, 1 + prompt_tokens, -1))
