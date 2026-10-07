"""RanPAC registration, local checkpoint mapping, and device checks without images."""

from argparse import ArgumentParser, Namespace

import pytest
import torch
from torch import nn

from backbone.vit import VisionTransformer
from models.ranpac import RanPAC
from models.ranpac_utils.inc_net import CosineLinear, RanPACNet
from models.ranpac_utils.ranpac import RanPAC_Model
from models.utils.continual_model import ContinualModel


def test_parser_keeps_original_defaults_and_accepts_explicit_checkpoint():
    parser = RanPAC.get_parser(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.rp_size == 10000
    assert defaults.pretrain_type == 'in21k'
    assert defaults.pretrained_checkpoint is None
    assert parser.parse_args(['--pretrained_checkpoint', '/local/checkpoint.npz']).pretrained_checkpoint == '/local/checkpoint.npz'


class CheckpointBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded = None

    def load_pretrained(self, path):
        self.loaded = path


@pytest.fixture
def construction_calls(monkeypatch):
    import models.ranpac as ranpac_module

    calls = []

    class Wrapper(nn.Module):
        def __init__(self, backbone, args):
            super().__init__()
            calls.append(backbone.loaded)
            self._network = backbone

    def base_init(self, backbone, loss, args, transform, dataset=None):
        nn.Module.__init__(self)
        self.net = backbone

    monkeypatch.setattr(ranpac_module, 'RanPAC_Model', Wrapper)
    monkeypatch.setattr(ContinualModel, '__init__', base_init)
    monkeypatch.setattr(ranpac_module, 'get_device', lambda: torch.device('cpu'))
    return calls


def test_explicit_checkpoint_is_loaded_before_adapters_are_constructed(tmp_path, construction_calls):
    checkpoint = tmp_path / 'weights.npz'
    checkpoint.write_bytes(b'local-checkpoint-fixture')
    backbone = CheckpointBackbone()
    model = RanPAC(backbone, None, Namespace(pretrained_checkpoint=str(checkpoint)), None)
    assert backbone.loaded == str(checkpoint)
    assert construction_calls == [str(checkpoint)]
    assert model.net._network is backbone
    assert 'W_rand' in model._buffers
    assert model.W_rand is None


def test_checkpoint_option_is_backward_compatible_when_absent(construction_calls):
    backbone = CheckpointBackbone()
    RanPAC(backbone, None, Namespace(), None)
    assert backbone.loaded is None
    assert construction_calls == [None]


def test_missing_local_checkpoint_fails_before_network_construction(tmp_path, construction_calls):
    with pytest.raises(FileNotFoundError, match='pretrained checkpoint does not exist'):
        RanPAC(CheckpointBackbone(), None, Namespace(pretrained_checkpoint=str(tmp_path / 'missing.npz')), None)
    assert construction_calls == []


def test_local_checkpoint_requires_supported_backbone(tmp_path, construction_calls):
    checkpoint = tmp_path / 'weights.npz'
    checkpoint.write_bytes(b'fixture')
    with pytest.raises(TypeError, match='load_pretrained'):
        RanPAC(nn.Linear(2, 2), None, Namespace(pretrained_checkpoint=str(checkpoint)), None)
    assert construction_calls == []


def test_qkv_and_mlp_conversion_preserves_loaded_values_and_freezes_only_ptm(monkeypatch):
    import models.ranpac_utils.inc_net as inc_net

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
            self.head = nn.Linear(2, 3)
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(2, 2304)
            block.mlp = nn.Module()
            block.mlp.fc1 = nn.Linear(2, 2)
            block.mlp.fc2 = nn.Linear(2, 2)
            self.blocks = nn.ModuleList([block])
            with torch.no_grad():
                block.attn.qkv.weight.copy_(torch.arange(4608).reshape(2304, 2))
                block.attn.qkv.bias.copy_(torch.arange(2304))

    class Receiver(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.adaptmlp = nn.Linear(2, 2, bias=False)
            self.frozen = nn.Linear(2, 2, bias=False)

        def load_state_dict(self, state, strict=False):
            self.loaded = state
            return ['adaptmlp.weight'], []

    monkeypatch.setattr(inc_net, 'VisionTransformer', Receiver)
    backbone = Backbone()
    qkv_weight = backbone.blocks[0].attn.qkv.weight.detach().clone()
    qkv_bias = backbone.blocks[0].attn.qkv.bias.detach().clone()
    fc1_weight = backbone.blocks[0].mlp.fc1.weight.detach().clone()
    network = RanPACNet(backbone)
    state = network.convnet.loaded
    for index, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
        torch.testing.assert_close(state[f'blocks.0.attn.{name}.weight'], qkv_weight[index * 768:(index + 1) * 768])
        torch.testing.assert_close(state[f'blocks.0.attn.{name}.bias'], qkv_bias[index * 768:(index + 1) * 768])
    torch.testing.assert_close(state['blocks.0.fc1.weight'], fc1_weight)
    assert not any('qkv' in key or 'mlp.fc' in key or 'head.' in key for key in state)
    assert network.convnet.adaptmlp.weight.requires_grad
    assert not network.convnet.frozen.weight.requires_grad
    assert not network.convnet.training


@pytest.fixture
def registered_metadata_network(monkeypatch):
    # Full native architecture on meta: no downloads, storage, inference, or training.
    linspace = torch.linspace
    monkeypatch.setattr(torch, 'linspace', lambda *args, **kwargs: linspace(*args, **dict(kwargs, device='cpu')))
    with torch.device('meta'):
        backbone = VisionTransformer(patch_size=16, embed_dim=768, depth=12,
                                     num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=100)
        network = RanPAC_Model(backbone, Namespace())
    return network


def test_wrapper_registers_real_architecture_and_trainable_adapter_parameters(registered_metadata_network):
    wrapper = registered_metadata_network
    assert isinstance(wrapper, nn.Module)
    assert wrapper._modules['_network'] is wrapper._network
    names = dict(wrapper.named_parameters())
    assert '_network.convnet.blocks.0.attn.q_proj.weight' in names
    assert '_network.convnet.blocks.11.adaptmlp.up_proj.weight' in names
    assert sum(parameter.numel() for parameter in names.values() if parameter.requires_grad) == 1189632
    assert all('adaptmlp' in name for name, parameter in names.items() if parameter.requires_grad)
    assert sum(parameter.numel() for parameter in names.values()) == 86988288
    assert set(wrapper.state_dict()) == set(names)


def test_recursive_device_move_updates_task_head_device(registered_metadata_network):
    wrapper = registered_metadata_network
    root = nn.Module()
    root.net = wrapper
    assert root.to('meta') is root
    assert wrapper._network.device.type == 'meta'
    wrapper._network.update_fc(10)
    assert wrapper._network.fc.weight.device.type == 'meta'
    assert wrapper._network.fc.weight.shape == (10, 768)


def test_wrapper_train_eval_and_to_follow_standard_module_contract(registered_metadata_network):
    wrapper = registered_metadata_network
    assert wrapper.train() is wrapper
    assert wrapper.training and wrapper._network.training and wrapper._network.convnet.training
    assert wrapper.eval() is wrapper
    assert not wrapper.training and not wrapper._network.training and not wrapper._network.convnet.training
    assert wrapper.to('meta') is wrapper
    assert wrapper._network.device.type == 'meta'


def test_random_projection_is_registered_saved_and_moved_with_head():
    head = CosineLinear(3, 2)
    projection = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    head.W_rand = projection
    head.weight = nn.Parameter(torch.zeros(2, 4))
    head.use_RP = True
    assert dict(head.named_buffers())['W_rand'] is projection
    torch.testing.assert_close(head.state_dict()['W_rand'], projection)
    assert 'W_rand' not in dict(head.named_parameters())
    head.to(dtype=torch.float64)
    assert head.W_rand.dtype == head.weight.dtype == torch.float64


def test_root_projection_copy_is_registered(construction_calls):
    model = RanPAC(CheckpointBackbone(), None, Namespace(), None)
    model.W_rand = torch.ones(2, 3)
    assert dict(model.named_buffers())['W_rand'] is model.W_rand
    assert 'W_rand' in model.state_dict()
    model.to(torch.device('meta'))
    assert model.W_rand.device.type == 'meta'
