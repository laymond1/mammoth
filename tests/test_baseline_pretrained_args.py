"""Exercise staged native parsing so pretrained=0 cannot become bool('0')."""

import sys

import pytest

import main
from models.l2p import L2P
from models.ranpac import RanPAC
from models.dualprompt import DualPrompt


@pytest.fixture
def no_initialization(monkeypatch):
    """These are real CLI parses; model construction and weight I/O are forbidden."""
    import backbone
    import backbone.vit
    import numpy as np
    import timm
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail('Parser regression must not construct models, read weights, or download')

    monkeypatch.setattr(main, 'initialize', forbidden)
    monkeypatch.setattr(backbone, 'get_backbone', forbidden)
    monkeypatch.setattr(backbone.vit, 'create_vision_transformer', forbidden)
    monkeypatch.setattr(timm, 'create_model', forbidden)
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    for model in (L2P, RanPAC, DualPrompt):
        monkeypatch.setattr(model, '__init__', forbidden)


def parse(monkeypatch, model, pretrained=None):
    command = ['--model', model, '--dataset', 'seq-cifar100-224', '--backbone', 'vit',
               '--model_config', 'default', '--dataset_config', 'ease', '--pretrain_type', 'in21k-ft-in1k',
               '--pretrained_checkpoint', '/tmp/parser-only-local.npz', '--lr', '.03',
               '--n_epochs', '1', '--batch_size', '64', '--num_workers', '4',
               '--stop_after', '1', '--device', 'cpu']
    if pretrained is not None:
        command.extend(['--pretrained', pretrained])
    monkeypatch.setattr(sys, 'argv', ['main.py', *command])
    return main.parse_args(cmd=command)


@pytest.mark.parametrize('model', ['ranpac', 'l2p', 'dualprompt'])
@pytest.mark.parametrize('value,expected', [('0', False), ('1', True)])
def test_native_cli_binary_pretrained_retains_boolean_type_and_local_checkpoint(monkeypatch, no_initialization, model, value, expected):
    args = parse(monkeypatch, model, value)
    assert args.pretrained is expected
    assert args.pretrained_checkpoint == '/tmp/parser-only-local.npz'
    assert args.backbone == 'vit'
    assert args.pretrain_type == 'in21k-ft-in1k'
    assert args.dataset_config == 'ease'


@pytest.mark.parametrize('model', ['ranpac', 'l2p', 'dualprompt'])
def test_native_pretrained_default_is_still_true(monkeypatch, no_initialization, model):
    assert parse(monkeypatch, model).pretrained is True
