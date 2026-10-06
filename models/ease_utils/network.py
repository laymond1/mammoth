"""Task lifecycle for EASE, independent of Mammoth's training loop."""

import math

import torch
from torch import nn

from models.ease_utils.classifier import CosineHead, PrototypeClassifier, _validate_offsets


class EaseNetwork(nn.Module):
    def __init__(self, backbone, alpha=0.1, recalc_sim=True, use_reweight=True):
        super().__init__()
        self.backbone = backbone
        # The method owns both heads; the backbone's generic head is unused.
        self.backbone.head = nn.Identity()
        self.classifier = PrototypeClassifier(backbone.feature_dim)
        self.proxy_head = None
        self.alpha = alpha
        self.recalc_sim = recalc_sim
        self.use_reweight = use_reweight
        self.register_buffer('task_finalized', torch.tensor(False))

    def begin_task(self, past_classes, seen_classes):
        ranges = self.classifier.task_offsets
        if ranges and not self.task_finalized.item():
            raise RuntimeError('Finalize the current EASE task before beginning another')
        expected_past = ranges[-1][1] if ranges else 0
        if past_classes != expected_past or seen_classes <= past_classes:
            raise ValueError('EASE tasks must add a nonempty, contiguous class interval')
        task = self.backbone.start_task()
        if task != len(ranges):
            raise RuntimeError('EASE adapter bank and class intervals are inconsistent')
        anchor = self.backbone.backbone.cls_token
        self.proxy_head = CosineHead(self.backbone.feature_dim, seen_classes - past_classes).to(
            device=anchor.device, dtype=anchor.dtype)
        self.classifier.expand(seen_classes, self.backbone.num_adapters,
                               [*ranges, (past_classes, seen_classes)])
        self.task_finalized.fill_(False)

    def training_logits(self, inputs):
        if self.proxy_head is None or self.task_finalized.item():
            raise RuntimeError('Begin an EASE task before training its proxy head')
        features = self.backbone.extract_features(inputs, self.backbone.current_adapter_index)
        return self.proxy_head(features)

    def finalize_task(self, prototypes, past_classes):
        if self.proxy_head is None or self.task_finalized.item():
            raise RuntimeError('EASE finalization requires one active, unfinalized task')
        self.classifier.update_current(prototypes, past_classes)
        self.classifier.complete_old_prototypes(self.recalc_sim)
        self.backbone.freeze_current()
        self.proxy_head.requires_grad_(False)
        self.proxy_head.eval()
        self.task_finalized.fill_(True)

    def forward(self, inputs):
        if not self.task_finalized.item():
            raise RuntimeError('EASE prototype evaluation requires end_task to complete first')
        return self.classifier(self.backbone.extract_all(inputs),
                               use_reweight=self.use_reweight, alpha=self.alpha)

    def get_extra_state(self):
        return {'version': 1, 'feature_dim': self.backbone.feature_dim,
                'alpha': self.alpha, 'recalc_sim': self.recalc_sim,
                'use_reweight': self.use_reweight}

    def _validate_extra_state(self, state):
        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError('unsupported EASE network checkpoint metadata')
        alpha = state.get('alpha')
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha) or alpha < 0):
            raise ValueError('invalid EASE checkpoint reweighting coefficient')
        if any(type(state.get(flag)) is not bool for flag in ('recalc_sim', 'use_reweight')):
            raise ValueError('invalid EASE checkpoint configuration flag')
        expected = self.get_extra_state()
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('EASE checkpoint scoring or prototype-completion configuration does not match')

    def set_extra_state(self, state):
        self._validate_extra_state(state)

    def _validate_checkpoint_components(self, state_dict, prefix):
        """Cross-check the task lifecycle before any child module loads weights."""
        weight = state_dict.get(prefix + 'classifier.weight')
        offsets = state_dict.get(prefix + 'classifier._task_offsets')
        finalized = state_dict.get(prefix + 'task_finalized')
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                or not isinstance(offsets, torch.Tensor) or offsets.ndim != 2
                or offsets.shape[1] != 2 or offsets.dtype != torch.long
                or offsets.device.type == 'meta'):
            raise ValueError('invalid EASE checkpoint prototype table or task offsets')
        if (not isinstance(finalized, torch.Tensor) or finalized.ndim != 0
                or finalized.dtype != torch.bool or finalized.device.type == 'meta'):
            raise ValueError('invalid EASE checkpoint task-finalization flag')
        count = offsets.shape[0]
        if weight.shape[1] != count * self.backbone.feature_dim:
            raise ValueError('EASE checkpoint prototype dimensions disagree with task offsets')
        ranges = _validate_offsets([tuple(pair) for pair in offsets.detach().cpu().tolist()],
                                   weight.shape[0], count)
        is_finalized = finalized.item()
        if not count and is_finalized:
            raise ValueError('an empty EASE checkpoint cannot have a finalized task')
        proxy_weight = state_dict.get(prefix + 'proxy_head.weight')
        proxy_sigma = state_dict.get(prefix + 'proxy_head.sigma')
        if count:
            expected_shape = (ranges[-1][1] - ranges[-1][0], self.backbone.feature_dim)
            if (not isinstance(proxy_weight, torch.Tensor) or tuple(proxy_weight.shape) != expected_shape
                    or not isinstance(proxy_sigma, torch.Tensor) or tuple(proxy_sigma.shape) != (1,)):
                raise ValueError('EASE checkpoint proxy head disagrees with the current class interval')
        elif proxy_weight is not None or proxy_sigma is not None:
            raise ValueError('an empty EASE checkpoint cannot contain a task proxy head')

        from backbone.ease_vit import EaseViT
        if isinstance(self.backbone, EaseViT):
            backbone_prefix = prefix + 'backbone.'
            adapter_count = self.backbone._checkpoint_adapter_count(state_dict, backbone_prefix)
            metadata = state_dict.get(backbone_prefix + '_extra_state')
            if metadata is not None:
                self.backbone._validate_extra_state(metadata)
                if metadata['num_adapters'] != adapter_count:
                    raise ValueError('EASE checkpoint adapter metadata and bank disagree')
                current_trainable = metadata['current_trainable']
            else:
                # Tensor-only legacy backbones are restored frozen; an active
                # training task cannot be resumed without its lifecycle state.
                current_trainable = False
            if adapter_count != count:
                raise ValueError('EASE checkpoint adapter count and classifier task offsets disagree')
            if count and current_trainable != (not is_finalized):
                raise ValueError('EASE checkpoint adapter trainability and task finalization disagree')
        return count, is_finalized

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        extra_key = prefix + '_extra_state'
        metadata = state_dict.get(extra_key)
        try:
            if metadata is not None:
                self._validate_extra_state(metadata)
            count, finalized = self._validate_checkpoint_components(state_dict, prefix)
        except ValueError as error:
            # Returning after appending an error would still let PyTorch load
            # children and resize adapter banks. Stop before those mutations.
            raise RuntimeError(f'{prefix}EASE network checkpoint validation failed: {error}') from error
        if metadata is None:
            # Older network states inherit the caller's explicit configuration.
            state_dict[extra_key] = self.get_extra_state()
        if count:
            weight = state_dict[prefix + 'proxy_head.weight']
            anchor = self.backbone.backbone.cls_token
            self.proxy_head = CosineHead(self.backbone.feature_dim, weight.shape[0]).to(
                device=anchor.device, dtype=anchor.dtype)
            if finalized:
                self.proxy_head.requires_grad_(False)
                self.proxy_head.eval()
        else:
            self.proxy_head = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
