"""Frozen ViT with independently registered task adapter spaces for EASE."""

import torch
from torch import nn

from backbone import MammothBackbone, register_backbone


class EaseViT(MammothBackbone):
    """Share one frozen ViT and train one independent adapter set per task.

    ``feature_dim`` is the width of one adapter space. ``extract_all`` returns
    the concatenated spaces used by EASE's expanded prototype classifier. The
    Mammoth ``forward`` contract uses the current space and a frozen placeholder
    head; the continual method owns its task heads and prototype classifier.
    """

    def __init__(self, num_classes, vit_size="base", pretrained=True,
                 ease_pretrain_type="in21k_ft_in1k", ease_checkpoint=None, adapter_size=64):
        super().__init__()
        import timm

        # Import lazily: importing models while backbone discovery is running
        # would enter ContinualModel/dataset discovery before registration ends.
        from models.ease_utils.vit import load_local_weights, resolve_vit_name

        if adapter_size < 1:
            raise ValueError("EASE adapter_size must be positive")
        self.model_name = resolve_vit_name(vit_size, ease_pretrain_type)
        self.vit_size = vit_size
        self.pretrain_type = ease_pretrain_type
        self.adapter_size = adapter_size
        self.backbone = timm.create_model(
            self.model_name, pretrained=bool(pretrained and not ease_checkpoint),
            num_classes=0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        )
        if ease_checkpoint:
            load_local_weights(self.backbone, ease_checkpoint)
        self.pretrained_loaded = bool(pretrained or ease_checkpoint)
        self.checkpoint_path = str(ease_checkpoint) if ease_checkpoint else None
        self.feature_dim = self.backbone.num_features
        self.num_features = self.feature_dim
        self.out_dim = self.feature_dim
        self.adapters = nn.ModuleList()
        self._current_trainable = False
        self.head = nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(False)
        # The EASE reference computes attention explicitly, without fused SDPA.
        for block in self.backbone.blocks:
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False
        self.train(False)

    @property
    def num_adapters(self):
        return len(self.adapters)

    @property
    def current_adapter_index(self):
        return self.num_adapters - 1 if self.num_adapters else None

    def start_task(self):
        """Freeze the previous task and create a newly initialized adapter set."""
        from models.ease_utils.vit import ParallelAdapter

        self.freeze_current()
        anchor = self.backbone.cls_token
        adapters = nn.ModuleList([
            ParallelAdapter(self.feature_dim, self.adapter_size)
            for _ in self.backbone.blocks
        ]).to(device=anchor.device, dtype=anchor.dtype)
        self.adapters.append(adapters)
        self._current_trainable = True
        adapters.train(self.training)
        return self.current_adapter_index

    def freeze_current(self):
        if self.num_adapters:
            self.adapters[-1].requires_grad_(False)
            self.adapters[-1].eval()
        self._current_trainable = False

    def current_adapter_parameters(self):
        if not self.num_adapters:
            return []
        return [parameter for parameter in self.adapters[-1].parameters() if parameter.requires_grad]

    def get_extra_state(self):
        """Record the adapter recipe and lifecycle alongside registered weights."""
        return {
            "version": 1,
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "adapter_size": self.adapter_size,
            "num_adapters": self.num_adapters,
            "current_trainable": self._current_trainable,
            "training": self.training,
            "pretrained_loaded": self.pretrained_loaded,
        }

    def _validate_extra_state(self, state):
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("unsupported EASE checkpoint metadata")
        recipe = {"model_name": self.model_name, "feature_dim": self.feature_dim, "adapter_size": self.adapter_size}
        if any(state.get(key) != value for key, value in recipe.items()):
            raise ValueError("EASE checkpoint backbone or adapter recipe does not match")
        count = state.get("num_adapters")
        if type(count) is not int or count < 0:
            raise ValueError("invalid EASE checkpoint adapter count")
        for flag in ("current_trainable", "training", "pretrained_loaded"):
            if type(state.get(flag)) is not bool:
                raise ValueError("invalid EASE checkpoint lifecycle flag")
        if count == 0 and state["current_trainable"]:
            raise ValueError("an empty EASE checkpoint cannot have a trainable adapter")

    def set_extra_state(self, state):
        self._validate_extra_state(state)
        if state["num_adapters"] != self.num_adapters:
            raise ValueError("EASE checkpoint metadata and adapter bank disagree")
        self.adapters.requires_grad_(False)
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(False)
        self.freeze_current()
        self._current_trainable = state["current_trainable"]
        if self.num_adapters:
            self.adapters[-1].requires_grad_(self._current_trainable)
        self.pretrained_loaded = state["pretrained_loaded"]
        self.train(state["training"])

    def _checkpoint_adapter_count(self, state_dict, prefix):
        """Validate every saved adapter tensor before allocating task modules."""
        adapter_prefix = prefix + "adapters."
        shapes = {
            ("down_proj", "weight"): (self.adapter_size, self.feature_dim),
            ("down_proj", "bias"): (self.adapter_size,),
            ("up_proj", "weight"): (self.feature_dim, self.adapter_size),
            ("up_proj", "bias"): (self.feature_dim,),
        }
        tasks = {}
        for key, tensor in state_dict.items():
            if not key.startswith(adapter_prefix):
                continue
            parts = key[len(adapter_prefix):].split(".")
            if len(parts) != 4 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError("malformed EASE checkpoint adapter key")
            task, layer = int(parts[0]), int(parts[1])
            if parts[:2] != [str(task), str(layer)] or layer >= len(self.backbone.blocks):
                raise ValueError("invalid EASE checkpoint adapter index")
            parameter = (parts[2], parts[3])
            if parameter not in shapes or not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shapes[parameter]:
                raise ValueError("EASE checkpoint adapter tensor shape does not match the recipe")
            tasks.setdefault(task, set()).add((layer, *parameter))
        if sorted(tasks) != list(range(len(tasks))):
            raise ValueError("EASE checkpoint adapter indices must be contiguous")
        expected = {(layer, *parameter) for layer in range(len(self.backbone.blocks)) for parameter in shapes}
        if any(parameters != expected for parameters in tasks.values()):
            raise ValueError("EASE checkpoint contains an incomplete adapter set")
        return len(tasks)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        extra_key = prefix + "_extra_state"
        try:
            count = self._checkpoint_adapter_count(state_dict, prefix)
            metadata = state_dict.get(extra_key)
            if metadata is not None:
                self._validate_extra_state(metadata)
                if metadata["num_adapters"] != count:
                    raise ValueError("EASE checkpoint metadata and adapter bank disagree")
            else:
                # Older tensor-only checkpoints have no training lifecycle.
                # Restore them frozen and use their strictly validated shapes.
                metadata = self.get_extra_state()
                metadata.update(num_adapters=count, current_trainable=False, training=False)
                state_dict[extra_key] = metadata
        except ValueError as error:
            error_msgs.append(f"{prefix}EASE checkpoint validation failed: {error}")
            return
        if count != self.num_adapters:
            self.adapters = nn.ModuleList()
            self._current_trainable = False
            for _ in range(count):
                self.start_task()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.head.eval()
        for index, adapters in enumerate(self.adapters):
            adapters.train(bool(mode and self._current_trainable and index == self.current_adapter_index))
        return self

    def _embed_tokens(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        return self.backbone.norm_pre(x)

    def _run_adapter_space(self, tokens, adapter_index):
        from models.ease_utils.vit import adapter_block_forward

        if adapter_index < -1 or adapter_index >= self.num_adapters:
            raise IndexError("EASE adapter index is outside the registered task spaces")
        adapters = self.adapters[adapter_index] if adapter_index >= 0 else None
        x = tokens
        for index, block in enumerate(self.backbone.blocks):
            x = adapter_block_forward(block, x, adapters[index] if adapters is not None else None)
        return self.backbone.norm(x)[:, 0]

    def extract_features(self, x, adapter_index):
        """Return CLS features in one task space, or unadapted features for -1."""
        return self._run_adapter_space(self._embed_tokens(x), adapter_index)

    def extract_all(self, x):
        if not self.num_adapters:
            raise RuntimeError("EASE requires start_task before expanded feature extraction")
        # Match the reference: patch embedding is shared by all independent
        # passes. The residual additions do not mutate this token tensor.
        tokens = self._embed_tokens(x)
        return torch.cat([self._run_adapter_space(tokens, index) for index in range(self.num_adapters)], dim=1)

    def forward(self, x, returnt="out"):
        features = self.extract_features(x, self.current_adapter_index if self.num_adapters else -1)
        if returnt == "features":
            return features
        logits = self.head(features)
        if returnt == "out":
            return logits
        if returnt in ("both", "full", "all"):
            return logits, features
        raise ValueError(f"Unknown return type: {returnt}")


@register_backbone("ease-vit")
def ease_vit(num_classes, vit_size="base", pretrained=True, ease_pretrain_type="in21k_ft_in1k",
             ease_checkpoint=None, adapter_size=64):
    """Parser-friendly EASE factory for Base, Small and Tiny patch-16 ViTs."""
    return EaseViT(num_classes, vit_size, pretrained, ease_pretrain_type, ease_checkpoint, adapter_size)
