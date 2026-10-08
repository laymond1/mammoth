# Copyright (c) 2026 Wonseon Lim, Jaesung Lee, Dae-Won Kim
# Licensed under the MIT License (see LICENSE in the project root).
"""
Critical Patch Sampling (CPS).

Implements the patch-selection module used during the prompt-training phase
of CPS-Prompt. Given a sequence of patch tokens (with the [CLS] token at
position 0), the module keeps a fraction ``1 - reduction_ratio`` of the patch
tokens and discards the rest. The [CLS] token is always retained.

Two sampling strategies are supported:

* ``uniform``         — patches are sampled uniformly at random.
* ``critical_score``  — patches are sampled (without replacement) from a
  multinomial distribution whose probabilities are obtained by applying a
  temperature-scaled softmax to per-patch attention scores. The attention
  scores come from a query forward pass through the ViT *without* prompts.
"""

import torch


class CriticalPatchSampling(torch.nn.Module):
    """Sparse patch selection used during prompt training.

    Args:
        reduction_ratio: Fraction of patch tokens to drop. Must lie in
            ``[0, 1)``. ``0`` keeps all patches.
        sampling: Either ``"uniform"`` or ``"critical_score"``.
        token_shuffling: If ``False`` (default), kept patch indices are
            re-sorted so that positional ordering is preserved.
        temperature: Temperature applied to the attention scores before the
            softmax used by ``critical_score`` sampling.
    """

    def __init__(self, reduction_ratio=0.5, sampling="uniform", token_shuffling=False, temperature=1.0):
        super().__init__()
        assert 0 <= reduction_ratio < 1, "The reduction_ratio must be in [0,1)"

        self.reduction_ratio = reduction_ratio
        self.sampling = sampling
        self.token_shuffling = token_shuffling
        self.temperature = temperature

    def forward(self, x, attn_scores=None, force_drop=True):
        """Drop a subset of patch tokens.

        Args:
            x: Token tensor of shape ``(N, L, D)``. The [CLS] token is
                assumed to be at position 0 along the length dimension.
            attn_scores: Per-patch attention scores of shape ``(N, L - 1)``.
                Required when ``sampling == "critical_score"``.
            force_drop: If ``True``, dropping is applied even when the
                module is in eval mode. (Currently always required.)
        """
        if not force_drop:
            return x
        if self.reduction_ratio == 0:
            return x

        # batch, length, dim
        N, L, D = x.shape
        
        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x, attn_scores)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    def get_mask(self, x, attn_scores=None):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        elif self.sampling == "critical_score":
            return self.critical_score_mask(x, attn_scores)
        else:
            raise NotImplementedError(f"CPS does not support {self.sampling} sampling")

    def uniform_mask(self, x):
        """Return a kept-patch index mask using uniform random sampling."""
        N, L, D = x.shape
        _L = L - 1  # number of patch tokens (excluding CLS)

        keep = int(_L * (1 - self.reduction_ratio))
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask
    
    def critical_score_mask(self, x, attn_scores):
        """Return a kept-patch index mask sampled from attention scores."""
        N, L, D = x.shape
        _L = L - 1
        keep = int(_L * (1 - self.reduction_ratio))

        # Temperature-scaled softmax over per-patch attention scores yields a
        # probability distribution from which kept patches are sampled
        # without replacement.
        attn_scores = attn_scores / self.temperature
        attn_scores = torch.softmax(attn_scores, dim=1)

        patch_mask = torch.multinomial(attn_scores, num_samples=keep, replacement=False)  # (B, keep)
        # Shift indices by +1 to account for the [CLS] token at position 0.
        patch_mask = patch_mask + 1

        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]

        return patch_mask