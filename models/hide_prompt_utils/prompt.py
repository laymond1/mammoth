"""HiDe-Prompt's task-indexed prefix bank.

Ported from HiDe-Prompt commit 0ed433334ff2cfb206c5e2899651c3044ec85f92,
peft/prompt/hide_prompt.py. The default intentionally retains the author's
prefix reshape ordering; use_permute_fix is an explicit alternative recipe.
"""

import math

import torch
from torch import nn


class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, pool_size=10, num_layers=5,
                 num_heads=12, prompt_init='uniform', use_permute_fix=False):
        super().__init__()
        if min(length, embed_dim, pool_size, num_layers, num_heads) < 1 or embed_dim % num_heads:
            raise ValueError('HiDe-Prompt prefix dimensions must be positive and divisible by heads')
        if prompt_init not in ('uniform', 'zero'):
            raise ValueError('HiDe-Prompt prompt_init must be uniform or zero')
        self.length = length
        self.pool_size = pool_size
        self.top_k = 1
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_permute_fix = use_permute_fix
        shape = (num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads)
        if prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(shape))
        else:
            # Keep the source's random draw before its uniform initialization.
            self.prompt = nn.Parameter(torch.randn(shape))
            nn.init.uniform_(self.prompt, -1, 1)

    def forward(self, x_embed, prompt_mask=None, prompt_idx=None, prompt_weight=None,
                prompt_momentum=0):
        if not math.isfinite(prompt_momentum) or not 0 <= prompt_momentum <= 1:
            raise ValueError('HiDe-Prompt prompt_momentum must lie in [0, 1]')
        idx = prompt_mask if prompt_mask is not None else prompt_idx
        if prompt_weight is not None:
            if prompt_weight.shape != (x_embed.shape[0], self.pool_size):
                raise ValueError('HiDe-Prompt prompt weights must be batch by pool size')
            raw = torch.einsum('bp,ndplhe->ndblhe', prompt_weight, self.prompt).unsqueeze(3)
        else:
            if (idx is None or idx.dtype != torch.long or idx.ndim != 2
                    or idx.shape != (x_embed.shape[0], 1)):
                raise ValueError('HiDe-Prompt requires one integer prompt index per example')
            if bool(((idx < 0) | (idx >= self.pool_size)).any()):
                raise ValueError('HiDe-Prompt index is outside the prompt bank')
            raw = self.prompt[:, :, idx]
            if prompt_momentum > 0 and prompt_mask is not None:
                first = int(idx[0, 0])
                if first < 1 or not bool((idx == first).all()):
                    raise ValueError('Prompt momentum requires a single noninitial training task')
                with torch.no_grad():
                    previous = self.prompt[:, :, :first].detach().clone().mean(2, keepdim=True)
                    previous = previous.unsqueeze(2).repeat(1, 1, idx.shape[0], 1, 1, 1, 1)
                raw = (1 - prompt_momentum) * raw + prompt_momentum * previous
        layers, dual, batch, top_k, length, heads, head_dim = raw.shape
        if self.use_permute_fix:
            raw = raw.permute(0, 2, 1, 3, 4, 5, 6)
        batched = raw.reshape(layers, batch, dual, top_k * length, heads, head_dim)
        result = {'batched_prompt': batched}
        if idx is not None:
            result['prompt_idx'] = idx
        return result
