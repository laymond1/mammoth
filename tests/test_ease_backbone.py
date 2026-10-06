"""EASE architecture and checkpoint checks, without model computation or downloads."""

import inspect

import pytest
import timm
import torch
from torch import nn

from backbone import get_backbone_class
from backbone.ease_vit import EaseViT, ease_vit
from models.ease_utils.network import EaseNetwork
from models.ease_utils.vit import resolve_vit_name


class MetadataViT(nn.Module):
    """Only shapes and registration; forward is forbidden in these checks."""

    def __init__(self, width):
        super().__init__()
        self.num_features = width
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.blocks = nn.ModuleList([nn.Module() for _ in range(12)])
        for block in self.blocks:
            block.attn = nn.Linear(width, width)
            block.attn.fused_attn = True

    def forward(self, *args, **kwargs):
        pytest.fail("These checks must not execute neural inference")


@pytest.fixture
def metadata_backbones(monkeypatch):
    calls = []

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        width = {"tiny": 192, "small": 384, "base": 768}[name.split("_")[1]]
        return MetadataViT(width)

    monkeypatch.setattr(timm, "create_model", create_model)
    # All initialization is metadata only, including checkpoint tensor shapes.
    with torch.device("meta"):
        yield calls


@pytest.mark.parametrize("size,width", [("tiny", 192), ("small", 384), ("base", 768)])
def test_three_sizes_choose_confirmed_augreg_family(metadata_backbones, size, width):
    net = ease_vit(100, vit_size=size)
    name, kwargs = metadata_backbones[-1]
    assert name == f"vit_{size}_patch16_224.augreg_in21k_ft_in1k"
    assert kwargs["pretrained"] is True
    assert kwargs["num_classes"] == 0
    assert kwargs["drop_path_rate"] == 0.0
    assert net.feature_dim == width
    assert net.adapter_size == 64
    assert net.num_adapters == 0
    assert not any(parameter.requires_grad for parameter in net.parameters())
    assert all(not block.attn.fused_attn for block in net.backbone.blocks)


@pytest.mark.parametrize("size", ["tiny", "small", "base"])
def test_in21k_remains_explicit_option(metadata_backbones, size):
    net = ease_vit(100, vit_size=size, ease_pretrain_type="in21k")
    assert net.model_name == f"vit_{size}_patch16_224.augreg_in21k"


def test_factory_registration_and_defaults():
    factory, arguments = get_backbone_class("ease-vit", return_args=True)
    assert factory is ease_vit
    assert "adapter_size" in arguments and "vit_size" in arguments
    assert inspect.signature(factory).parameters["ease_pretrain_type"].default == "in21k_ft_in1k"
    assert resolve_vit_name() == "vit_base_patch16_224.augreg_in21k_ft_in1k"


def test_adapter_lifecycle_and_registered_state(metadata_backbones):
    net = ease_vit(100, pretrained=False)
    assert net.pretrained_loaded is False
    assert net.current_adapter_parameters() == []
    assert net.start_task() == 0
    net.train(True)
    assert net.adapters[0].training
    assert not net.backbone.training
    assert net.start_task() == 1
    assert not net.adapters[0].training
    assert not any(parameter.requires_grad for parameter in net.adapters[0].parameters())
    assert all(parameter.requires_grad for parameter in net.adapters[1].parameters())
    assert len(net.adapters[1]) == 12
    assert net.adapters[1][0].down_proj.out_features == 64
    assert net.adapters[1][0].scale == 0.1 and net.adapters[1][0].dropout == 0.1
    state = net.state_dict()
    assert "adapters.0.0.down_proj.weight" in state
    assert "adapters.1.11.up_proj.bias" in state
    net.freeze_current()
    assert net.current_adapter_parameters() == []
    assert not any(parameter.requires_grad for parameter in net.parameters())


@pytest.mark.parametrize("saved_count,existing_count", [(0, 2), (1, 0), (2, 0), (2, 1), (1, 3)])
@pytest.mark.parametrize("finalized", [False, True])
def test_reload_reconstructs_bank_and_lifecycle(metadata_backbones, saved_count, existing_count, finalized):
    source = ease_vit(100, vit_size="tiny")
    for _ in range(saved_count):
        source.start_task()
    source.train(True)
    if finalized:
        source.freeze_current()
    target = ease_vit(100, vit_size="tiny")
    for _ in range(existing_count):
        target.start_task()
    target.load_state_dict(source.state_dict())
    assert target.num_adapters == saved_count
    assert target._current_trainable == bool(saved_count and not finalized)
    assert target.training
    assert not target.backbone.training
    assert not any(parameter.requires_grad for group in target.adapters[:-1] for parameter in group.parameters())
    if saved_count:
        assert all(parameter.requires_grad == (not finalized) for parameter in target.adapters[-1].parameters())
        assert target.adapters[-1].training == (not finalized)


def test_tensor_only_legacy_reload_is_frozen(metadata_backbones):
    source = ease_vit(100, vit_size="tiny")
    source.start_task()
    state = source.state_dict()
    state.pop("_extra_state")
    target = ease_vit(100, vit_size="tiny")
    target.load_state_dict(state)
    assert target.num_adapters == 1
    assert not target.training
    assert not any(parameter.requires_grad for parameter in target.parameters())


@pytest.mark.parametrize("mismatch", ["model_name", "feature_dim", "adapter_size", "num_adapters"])
def test_reload_rejects_recipe_metadata_mismatch(metadata_backbones, mismatch):
    source = ease_vit(100, vit_size="tiny")
    source.start_task()
    state = source.state_dict()
    state["_extra_state"][mismatch] = "wrong" if mismatch == "model_name" else 999
    target = ease_vit(100, vit_size="tiny")
    with pytest.raises(RuntimeError, match="EASE checkpoint validation failed"):
        target.load_state_dict(state)
    assert target.num_adapters == 0


@pytest.mark.parametrize("kind", ["gap", "negative", "noncanonical", "layer", "missing", "shape"])
def test_reload_rejects_malformed_adapter_bank(metadata_backbones, kind):
    source = ease_vit(100, vit_size="tiny")
    source.start_task()
    state = source.state_dict()
    key = "adapters.0.0.down_proj.weight"
    if kind == "missing":
        state.pop(key)
    elif kind == "shape":
        state[key] = torch.empty(64, 193)
    else:
        replacement = {"gap": "adapters.2.0.down_proj.weight", "negative": "adapters.-1.0.down_proj.weight",
                       "noncanonical": "adapters.00.0.down_proj.weight", "layer": "adapters.0.12.down_proj.weight"}[kind]
        state[replacement] = state.pop(key)
    target = ease_vit(100, vit_size="tiny")
    with pytest.raises(RuntimeError, match="EASE checkpoint validation failed"):
        target.load_state_dict(state)
    assert target.num_adapters == 0


@pytest.mark.parametrize("size,pretrain", [("large", "in21k"), ("base", "in1k"), ("base", "random")])
def test_unknown_checkpoint_family_is_rejected(size, pretrain):
    with pytest.raises(ValueError):
        resolve_vit_name(size, pretrain)


def _network_with_real_adapter_bank(n_tasks, finalized=True, **configuration):
    """Use real lifecycle modules, metadata ViT weights, and tiny CPU tables."""
    bank = ease_vit(100, vit_size="tiny")
    with torch.device("cpu"):
        network = EaseNetwork(bank, **configuration)
    for task in range(n_tasks):
        past, seen = task * 2, (task + 1) * 2
        network.begin_task(past, seen)
        if task < n_tasks - 1 or finalized:
            # Explicit synthetic means exercise finalization and freezing;
            # no input image or feature-extraction call is involved.
            means = torch.ones(2, network.classifier.in_features, device="cpu")
            network.finalize_task(means, past)
    return network


@pytest.mark.parametrize("source_count,target_count", [(0, 2), (1, 0), (2, 0), (2, 1), (1, 3)])
@pytest.mark.parametrize("finalized", [False, True])
def test_network_reload_restores_real_ease_bank_and_can_add_next_task(metadata_backbones, source_count, target_count, finalized):
    source = _network_with_real_adapter_bank(source_count, finalized)
    target = _network_with_real_adapter_bank(target_count)
    target.load_state_dict(source.state_dict())
    assert target.backbone.num_adapters == source_count
    assert target.classifier.n_adapters == source_count
    assert target.classifier.task_offsets == [(task * 2, (task + 1) * 2) for task in range(source_count)]
    assert target.classifier.weight.device.type == "cpu"
    assert target.backbone._current_trainable == bool(source_count and not finalized)
    assert target.task_finalized.item() == bool(source_count and finalized)
    assert all(not parameter.requires_grad for group in target.backbone.adapters[:-1] for parameter in group.parameters())
    if source_count:
        assert target.proxy_head.out_features == 2
        assert target.proxy_head.weight.device.type == "meta"
        assert all(parameter.requires_grad == (not finalized) for parameter in target.proxy_head.parameters())
        assert all(parameter.device.type == "meta" for parameter in target.backbone.adapters[-1].parameters())
        torch.testing.assert_close(target.classifier.weight, source.classifier.weight)
    else:
        assert target.proxy_head is None
    if not source_count or finalized:
        target.begin_task(source_count * 2, (source_count + 1) * 2)
        assert target.backbone.num_adapters == source_count + 1
        assert target.classifier.n_adapters == source_count + 1
        assert target.backbone.current_adapter_index == source_count
        assert target.proxy_head.out_features == 2


@pytest.mark.parametrize("setting,value", [("alpha", 0.3), ("recalc_sim", False), ("use_reweight", False)])
def test_network_reload_rejects_changed_scoring_configuration_before_mutation(metadata_backbones, setting, value):
    source = _network_with_real_adapter_bank(2)
    target = _network_with_real_adapter_bank(1, **{setting: value})
    old_proxy = target.proxy_head
    with pytest.raises(RuntimeError, match="configuration does not match"):
        target.load_state_dict(source.state_dict())
    assert target.backbone.num_adapters == 1
    assert target.classifier.n_adapters == 1
    assert target.proxy_head is old_proxy


@pytest.mark.parametrize("kind", ["adapter_count", "proxy_rows", "proxy_sigma", "finalized", "active", "offsets"])
def test_network_reload_rejects_cross_component_mismatch_before_mutation(metadata_backbones, kind):
    source = _network_with_real_adapter_bank(2)
    state = source.state_dict()
    if kind == "adapter_count":
        state = {key: value for key, value in state.items() if not key.startswith("backbone.adapters.1.")}
        state["backbone._extra_state"]["num_adapters"] = 1
    elif kind == "proxy_rows":
        state["proxy_head.weight"] = torch.empty(3, 192)
    elif kind == "proxy_sigma":
        state["proxy_head.sigma"] = torch.empty(2)
    elif kind == "finalized":
        state["backbone._extra_state"]["current_trainable"] = True
    elif kind == "active":
        state["task_finalized"] = torch.tensor(False, device="cpu")
    else:
        state["classifier._task_offsets"] = torch.tensor([[0, 2], [3, 4]], device="cpu")
    target = _network_with_real_adapter_bank(1)
    old_proxy = target.proxy_head
    old_adapters = target.backbone.adapters
    with pytest.raises(RuntimeError, match="EASE network checkpoint validation failed"):
        target.load_state_dict(state)
    assert target.backbone.num_adapters == 1
    assert target.classifier.n_adapters == 1
    assert target.proxy_head is old_proxy
    assert target.backbone.adapters is old_adapters


def test_legacy_network_checkpoint_reloads_finalized_bank_frozen(metadata_backbones):
    source = _network_with_real_adapter_bank(2)
    state = source.state_dict()
    state.pop("_extra_state")
    state.pop("backbone._extra_state")
    target = _network_with_real_adapter_bank(0)
    target.load_state_dict(state)
    assert target.backbone.num_adapters == target.classifier.n_adapters == 2
    assert target.task_finalized.item()
    assert not any(parameter.requires_grad for parameter in target.parameters())


def test_legacy_active_network_checkpoint_requires_lifecycle_metadata(metadata_backbones):
    source = _network_with_real_adapter_bank(1, finalized=False)
    state = source.state_dict()
    state.pop("backbone._extra_state")
    target = _network_with_real_adapter_bank(0)
    with pytest.raises(RuntimeError, match="trainability and task finalization disagree"):
        target.load_state_dict(state)
    assert target.backbone.num_adapters == target.classifier.n_adapters == 0
