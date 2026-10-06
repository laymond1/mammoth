"""Cosine heads and semantic-guided prototype completion for EASE.

The completion and block reweighting follow the official CVPR24-EASE code
(`models/ease.py` and `backbone/linears.py`, revision 6d67508). Task boundaries
are explicit so the initial task and later increments need not have equal size.
"""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_offsets(task_offsets: Sequence[tuple[int, int]], n_classes: int,
                      n_adapters: int) -> list[tuple[int, int]]:
    if len(task_offsets) != n_adapters:
        raise ValueError("EASE requires one adapter and one class interval per task")
    offsets = []
    previous_end = 0
    for interval in task_offsets:
        if len(interval) != 2:
            raise ValueError("Every task interval must contain a start and an end")
        start, end = interval
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or start != previous_end or end <= start):
            raise ValueError("Task intervals must be contiguous, nonempty, and start at zero")
        offsets.append((start, end))
        previous_end = end
    if previous_end != n_classes:
        raise ValueError("Task intervals must cover exactly the seen classes")
    return offsets


class CosineHead(nn.Module):
    """Trainable current-task proxy head, with the official initialization."""

    def __init__(self, feature_dim: int, n_classes: int):
        super().__init__()
        self.in_features = _positive_int(feature_dim, "feature_dim")
        self.out_features = _positive_int(n_classes, "n_classes")
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.sigma = nn.Parameter(torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.sigma.fill_(1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.in_features:
            raise ValueError("Current-task features must have shape (batch, feature_dim)")
        return self.sigma * F.linear(F.normalize(features, p=2, dim=1),
                                     F.normalize(self.weight, p=2, dim=1))


class PrototypeClassifier(nn.Module):
    """Nontrainable class means across all observed adapter feature spaces.

    For task ``t`` the weight has shape ``(seen_classes, (t + 1) * D)``.
    Rows belonging to the current task are measured using its training images
    in every observed adapter space. Older rows in newer feature spaces are
    completed from semantic similarities; no older images are required.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = _positive_int(feature_dim, "feature_dim")
        self.register_buffer("weight", torch.empty(0, 0))
        # The official prototype classifier is never optimized: its scale is 1.
        self.register_buffer("sigma", torch.ones(1))
        self.register_buffer("_task_offsets", torch.empty(0, 2, dtype=torch.long))

    @property
    def task_offsets(self) -> list[tuple[int, int]]:
        return [tuple(interval) for interval in self._task_offsets.detach().cpu().tolist()]

    @property
    def n_adapters(self) -> int:
        return self._task_offsets.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @torch.no_grad()
    def expand(self, n_classes: int, n_adapters: int,
               task_offsets: Sequence[tuple[int, int]]):
        """Expand the table while retaining previously observed class blocks."""
        _positive_int(n_classes, "n_classes")
        _positive_int(n_adapters, "n_adapters")
        offsets = _validate_offsets(task_offsets, n_classes, n_adapters)
        old_offsets = self.task_offsets
        if (n_classes < self.out_features or n_adapters < self.n_adapters
                or offsets[:len(old_offsets)] != old_offsets):
            raise ValueError("Prototype expansion cannot remove or change an observed task")
        expanded = self.weight.new_zeros(n_classes, n_adapters * self.feature_dim)
        expanded[:self.out_features, :self.in_features].copy_(self.weight)
        self.weight = expanded
        self._task_offsets = torch.tensor(offsets, dtype=torch.long, device=self.weight.device)

    @torch.no_grad()
    def update_current(self, prototypes: torch.Tensor, past_classes: int):
        """Copy current-class means in every observed adapter feature space."""
        if not self.n_adapters or past_classes != self.task_offsets[-1][0]:
            raise ValueError("past_classes must match the current task's class boundary")
        expected = (self.out_features - past_classes, self.in_features)
        if prototypes.ndim != 2 or tuple(prototypes.shape) != expected:
            raise ValueError(f"Current prototypes must have shape {expected}")
        if not prototypes.is_floating_point() or not torch.isfinite(prototypes).all():
            raise ValueError("Current prototypes must contain finite floating-point means")
        self.weight[past_classes:].copy_(prototypes)

    @staticmethod
    def _semantic_transfer(old_means: torch.Tensor, reference_old: torch.Tensor,
                           reference_new: torch.Tensor) -> torch.Tensor:
        # cosine_similarity in the official nested loops uses eps=1e-8.
        similarities = F.normalize(old_means, p=2, dim=1, eps=1e-8) @ F.normalize(
            reference_old, p=2, dim=1, eps=1e-8).transpose(0, 1)
        return F.softmax(similarities, dim=1) @ reference_new

    @torch.no_grad()
    def complete_old_prototypes(self, recalc_sim: bool = True):
        """Complete old rows with the official semantic-guided interpolation.

        With ``recalc_sim=True``, every old missing block is recomputed using all
        observed classes from the target feature space's task onward. Otherwise
        only the newly added feature space is filled, using current-task classes.
        """
        if not self.n_adapters:
            raise ValueError("Expand the classifier before completing prototypes")
        offsets = self.task_offsets
        current_task = self.n_adapters - 1
        for old_task in range(current_task):
            old_start, old_end = offsets[old_task]
            source_start = old_task * self.feature_dim
            source_end = source_start + self.feature_dim
            target_tasks = range(old_task + 1, current_task + 1) if recalc_sim else (current_task,)
            for target_task in target_tasks:
                reference_start = offsets[target_task if recalc_sim else current_task][0]
                target_start = target_task * self.feature_dim
                target_end = target_start + self.feature_dim
                completed = self._semantic_transfer(
                    self.weight[old_start:old_end, source_start:source_end],
                    self.weight[reference_start:, source_start:source_end],
                    self.weight[reference_start:, target_start:target_end],
                )
                self.weight[old_start:old_end, target_start:target_end].copy_(completed)

    def forward(self, features: torch.Tensor, use_reweight: bool = True,
                alpha: float = 0.1) -> torch.Tensor:
        if not self.n_adapters:
            raise ValueError("Expand the classifier before evaluation")
        if features.ndim != 2 or features.shape[1] != self.in_features:
            raise ValueError("Features must concatenate every observed adapter feature space")
        if not use_reweight:
            return self.sigma * F.linear(F.normalize(features, p=2, dim=1),
                                         F.normalize(self.weight, p=2, dim=1))
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite, nonnegative reweighting coefficient")

        # Each class retains weight 1 in its own task's adapter feature space.
        # The remaining spaces share total coefficient alpha, as in the paper.
        off_diagonal = alpha / (self.n_adapters - 1) if self.n_adapters > 1 else 0.0
        coefficients = self.weight.new_full((self.out_features, self.n_adapters), off_diagonal)
        for task, (start, end) in enumerate(self.task_offsets):
            coefficients[start:end, task] = 1.0
        logits = features.new_zeros(features.shape[0], self.out_features)
        for task in range(self.n_adapters):
            start = task * self.feature_dim
            end = start + self.feature_dim
            similarities = F.linear(F.normalize(features[:, start:end], p=2, dim=1),
                                    F.normalize(self.weight[:, start:end], p=2, dim=1))
            logits = logits + similarities * coefficients[:, task]
        return self.sigma * logits

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Tables expand at task boundaries. Restore their registered shapes even
        # when loading into a freshly constructed, empty classifier.
        weight = state_dict.get(prefix + "weight")
        offsets = state_dict.get(prefix + "_task_offsets")
        if weight is not None and offsets is not None:
            valid_shape = (weight.ndim == 2 and offsets.ndim == 2 and offsets.shape[1] == 2
                           and weight.shape[1] == offsets.shape[0] * self.feature_dim)
            try:
                if not valid_shape:
                    raise ValueError("Prototype checkpoint has incompatible feature dimensions")
                _validate_offsets([tuple(pair) for pair in offsets.detach().cpu().tolist()],
                                  weight.shape[0], offsets.shape[0])
            except ValueError as exc:
                error_msgs.append(f"{prefix}{exc}")
            else:
                self.weight = self.weight.new_empty(weight.shape)
                self._task_offsets = self._task_offsets.new_empty(offsets.shape)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
