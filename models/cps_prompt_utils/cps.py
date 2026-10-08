# Adapted from CPS-Prompt, copyright (c) 2026 Wonseon Lim, Jaesung Lee, Dae-Won Kim.
# MIT license and pinned source attribution: see SOURCE.json and UPSTREAM_LICENSE.
"""Attention-weighted sampling without replacement; CLS and positions survive."""

import math

import torch
from torch import nn

from models.resource_prompt_utils.state import CheckpointRNG


class CriticalPatchSampling(nn.Module):
    def __init__(self, reduction_ratio=.4, sampling='critical_score', temperature=.1, seed=1993):
        super().__init__()
        if not math.isfinite(reduction_ratio) or not 0 <= reduction_ratio < 1:
            raise ValueError('reduction_ratio must be in [0,1)')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        if sampling not in ('critical_score', 'uniform'):
            raise ValueError('Unknown CPS sampling strategy')
        self.reduction_ratio = reduction_ratio
        self.sampling = sampling
        self.temperature = temperature
        self.rng = CheckpointRNG(seed)
        self.last_indices = None

    def forward(self, tokens, scores=None):
        batch, count, width = tokens.shape
        keep = int((count - 1) * (1 - self.reduction_ratio))
        if keep < 1:
            raise ValueError('CPS must retain at least one image patch')
        if self.reduction_ratio == 0:
            self.last_indices = torch.arange(count, device=tokens.device).expand(batch, -1)
            return tokens
        generator = self.rng.generator(tokens.device)
        if self.sampling == 'uniform':
            indices = torch.rand(batch, count - 1, device=tokens.device, generator=generator).argsort(dim=1)[:, :keep] + 1
        else:
            if scores is None or scores.shape != (batch, count - 1):
                raise ValueError('CPS requires one query importance per image patch')
            if not torch.isfinite(scores).all():
                raise ValueError('CPS importance scores must be finite')
            probabilities = (scores.detach() / self.temperature).softmax(dim=1)
            indices = torch.multinomial(probabilities, keep, replacement=False, generator=generator) + 1
        indices = indices.sort(dim=1).values
        indices = torch.cat((torch.zeros(batch, 1, dtype=torch.long, device=tokens.device), indices), dim=1)
        self.last_indices = indices.detach()
        return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, width))
