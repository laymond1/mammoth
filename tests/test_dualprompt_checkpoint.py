"""DualPrompt local checkpoint contracts, without inference or downloads."""

from argparse import ArgumentParser, Namespace

import pytest
from torch import nn

from models.dualprompt import DualPrompt
from models.dualprompt_utils import model as network_module
from models.dualprompt_utils import vision_transformer as vit_module


class CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)
        self.checkpoints = []

    def load_pretrained(self, checkpoint):
        self.checkpoints.append(checkpoint)

    def forward(self, *args, **kwargs):
        pytest.fail('Checkpoint checks must not execute image inference')


@pytest.fixture
def checkpoint_factory(monkeypatch):
    calls = []

    def create_model(variant, **kwargs):
        model = CheckpointModel()
        calls.append((variant, kwargs, model))
        return model

    monkeypatch.setattr(vit_module, 'create_vision_transformer', create_model)
    return calls


@pytest.mark.parametrize('pretrained', [True, False])
def test_local_checkpoint_prevents_remote_loading(checkpoint_factory, tmp_path, pretrained):
    checkpoint = tmp_path / 'weights.npz'
    checkpoint.touch()
    model = vit_module.vit_base_patch16_224_dualprompt(pretrained=pretrained, pretrained_checkpoint=checkpoint, num_classes=100)
    _, kwargs, _ = checkpoint_factory[0]
    assert kwargs['pretrained'] is False
    assert 'pretrained_checkpoint' not in kwargs
    assert kwargs['num_classes'] == 100
    assert model.checkpoints == [str(checkpoint.resolve())]


@pytest.mark.parametrize('pretrained', [True, False])
def test_unspecified_checkpoint_preserves_legacy_loading(checkpoint_factory, pretrained):
    model = vit_module.vit_base_patch16_224_dualprompt(pretrained=pretrained)
    variant, kwargs, _ = checkpoint_factory[0]
    assert variant == 'vit_base_patch16_224_in21k_fn_in1k_old'
    assert kwargs['pretrained'] is pretrained
    assert model.checkpoints == []


def test_missing_local_checkpoint_fails_before_model_or_download(checkpoint_factory, tmp_path):
    with pytest.raises(FileNotFoundError, match='checkpoint does not exist'):
        vit_module.vit_base_patch16_224_dualprompt(pretrained=True, pretrained_checkpoint=tmp_path / 'missing.npz')
    assert checkpoint_factory == []


@pytest.mark.parametrize('suffix', ['.pt', '.safetensors', '.txt'])
def test_unsupported_checkpoint_rejected_before_model_creation(checkpoint_factory, tmp_path, suffix):
    checkpoint = tmp_path / f'weights{suffix}'
    checkpoint.touch()
    with pytest.raises(ValueError, match='local ViT NPZ'):
        vit_module.vit_base_patch16_224_dualprompt(pretrained=True, pretrained_checkpoint=checkpoint)
    assert checkpoint_factory == []


def _model_arguments(checkpoint=True):
    parser = DualPrompt.get_parser(ArgumentParser())
    args = parser.parse_args([])
    if checkpoint:
        args.pretrained_checkpoint = '/local/shared/weights.npz'
    return args


def test_parser_accepts_local_checkpoint_without_changing_defaults():
    parser = DualPrompt.get_parser(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.pretrained_checkpoint is None
    assert defaults.pretrained == 1
    args = parser.parse_args(['--pretrained', '0', '--pretrained_checkpoint', '/local/weights.npz'])
    assert args.pretrained is False
    assert args.pretrained_checkpoint == '/local/weights.npz'


@pytest.mark.parametrize('pretrained', ['0', '1'])
def test_native_parser_preserves_explicit_checkpoint_and_protocol(monkeypatch, pretrained):
    import sys
    from main import parse_args
    command = ['--model', 'dualprompt', '--dataset', 'seq-cifar100-224',
               '--model_config', 'default', '--dataset_config', 'ease', '--lr', '0.03',
               '--pretrained', pretrained,
               '--pretrained_checkpoint', '/shared/selected-base.npz',
               '--n_epochs', '1', '--batch_size', '64', '--num_workers', '4',
               '--stop_after', '1', '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', ['main.py', *command])
    args = parse_args(cmd=command, verbose=False)
    assert args.pretrained_checkpoint == '/shared/selected-base.npz'
    assert args.pretrained == bool(int(pretrained))
    assert args.optimizer == 'adam' and args.lr == 0.03
    assert args.dataset_config == 'ease'
    assert (args.n_epochs, args.batch_size, args.num_workers, args.stop_after) == (1, 64, 4, 1)


@pytest.mark.parametrize('has_checkpoint_argument', [True, False])
def test_query_and_prompt_backbones_receive_same_checkpoint(monkeypatch, has_checkpoint_argument):
    calls = []

    monkeypatch.setattr(network_module, 'checkpoint_spec', lambda size, path: {
        'file': str(path), 'sha256': 'a' * 64, 'canonical_sha256': 'b' * 64})
    monkeypatch.setattr(network_module, 'verify_pretrained_branches', lambda *args, **kwargs: {
        'query': 'b' * 64, 'prompt': 'b' * 64})

    def factory(**kwargs):
        calls.append(kwargs)
        return CheckpointModel()

    monkeypatch.setattr(network_module, 'vit_base_patch16_224_dualprompt', factory)
    args = _model_arguments()
    if not has_checkpoint_argument:
        del args.pretrained_checkpoint
    model = network_module.Model(args, 100)
    assert len(calls) == 2
    expected = '/local/shared/weights.npz' if has_checkpoint_argument else None
    assert all(call['pretrained_checkpoint'] == expected for call in calls)
    assert all(call['pretrained'] == 1 for call in calls)
    assert calls[0]['num_classes'] == calls[1]['num_classes'] == 100
    assert calls[1]['use_prefix_tune_for_g_prompt'] == 1
    assert calls[1]['use_prefix_tune_for_e_prompt'] == 1
    assert not model.original_model.training
    assert all(not parameter.requires_grad for parameter in model.original_model.parameters())
    assert not model.model.blocks.weight.requires_grad
    assert model.model.head.weight.requires_grad


def test_prefix_prompts_do_not_extend_image_position_table():
    # A narrow CPU model verifies architecture only; no image forward/backward.
    model = vit_module.VisionTransformer(
        img_size=224, patch_size=16, embed_dim=12, depth=1, num_heads=3,
        num_classes=10, args=Namespace(use_permute_fix=False),
        prompt_length=5, prompt_pool=True, prompt_key=True, pool_size=10, top_k=1,
        use_g_prompt=True, g_prompt_length=5, g_prompt_layer_idx=[0],
        use_prefix_tune_for_g_prompt=True, use_e_prompt=True,
        e_prompt_layer_idx=[0], use_prefix_tune_for_e_prompt=True,
    )
    assert model.pos_embed.shape == (1, 197, 12)
    assert model.total_prompt_len == 0
    assert model.g_prompt.shape == (1, 2, 5, 3, 4)
    assert model.e_prompt.prompt.shape == (1, 2, 10, 5, 3, 4)
