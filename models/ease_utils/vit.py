"""EASE's parallel AdaptFormer branch and explicit pretrained weight selection.

The adapter placement and initialization follow the CVPR 2024 EASE reference
implementation (sun-hailong/CVPR24-Ease, backbone/vit_ease.py). The frozen
transformer itself is provided by the repository's pinned timm version.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


VIT_DIMENSIONS = {"tiny": 192, "small": 384, "base": 768}
PRETRAIN_TYPES = ("in21k", "in21k_ft_in1k")


def resolve_vit_name(vit_size="base", pretrain_type="in21k_ft_in1k"):
    """Select a tagged weight family, avoiding timm's mutable default aliases."""
    if vit_size not in VIT_DIMENSIONS:
        raise ValueError("EASE vit_size must be one of: tiny, small, base")
    if pretrain_type not in PRETRAIN_TYPES:
        raise ValueError("EASE ease_pretrain_type must be in21k or in21k_ft_in1k")
    return f"vit_{vit_size}_patch16_224.augreg_{pretrain_type}"


class ParallelAdapter(nn.Module):
    """One task-specific EASE adapter, without layer normalization or residual."""

    def __init__(self, feature_dim, adapter_size=64):
        super().__init__()
        if adapter_size < 1:
            raise ValueError("EASE adapter_size must be positive")
        self.down_proj = nn.Linear(feature_dim, adapter_size)
        self.up_proj = nn.Linear(adapter_size, feature_dim)
        self.scale = 0.1
        self.dropout = 0.1
        # Match the reference's LoRA-style zero initial contribution.
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        x = F.relu(self.down_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.up_proj(x) * self.scale


def adapter_block_forward(block, x, adapter=None):
    """Insert the adapter after attention, in parallel with the frozen FFN.

    In particular, the adapter consumes the post-attention residual stream,
    before norm2. Calling the frozen FFN with gradients enabled is essential:
    its input can depend on adapters in earlier blocks.
    """
    x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
    adapter_output = adapter(x) if adapter is not None else None
    branch = block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    if adapter_output is not None:
        branch = branch + adapter_output
    return x + branch


def load_local_weights(model, checkpoint):
    """Load an entire frozen backbone; only a pretrained classifier is ignored.

    PyTorch loading explicitly restricts deserialization to weights. Missing
    transformer parameters and unrelated keys are errors rather than a random
    backbone silently passing as pretrained. timm's original AugReg NPZ files
    are also supported through its dedicated NumPy loader.
    """
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"EASE checkpoint does not exist: {path}")
    if path.suffix == ".npz":
        model.load_pretrained(str(path))
        return
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("EASE checkpoint must contain a model state dictionary")
    for key in ("state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if key not in ("head.weight", "head.bias")}
    model.load_state_dict(state, strict=True)
