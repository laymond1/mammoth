"""AToM between attention and MLP; ALD skips whole blocks before execution."""

import math

import torch
from torch.nn import functional as F

from models.coda_prompt_utils.vit import Attention, VisionTransformer

from .merge import progressive_merges, weighted_merge


def keeping_probability(step, merged, *, minimum=.5, gamma=.001, alpha=.9, threshold=8):
    """REP Eq5/6; full first update follows the paper's initialization text."""
    if step == 0:
        return 1.
    temporal = (1 - minimum) * math.exp(-gamma * step) + minimum
    return temporal * (alpha if merged >= threshold else 1.)


class REPAttention(Attention):
    def forward(self, x, prompt=None, mass=None, need_metric=False):
        batch, count, width = x.shape
        qkv = self.qkv(x).reshape(batch, count, 3, self.num_heads, width // self.num_heads).permute(2, 0, 3, 1, 4)
        q, image_k, v = qkv.unbind(0)
        metric = image_k.mean(dim=1) if need_metric else None
        k = image_k
        if prompt is not None:
            pk, pv = [value.reshape(batch, -1, self.num_heads, width // self.num_heads).transpose(1, 2) for value in prompt]
            k, v = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
            if mass is not None:
                # Every prefix key has unit mass; image masses track prior merges.
                mass = torch.cat((mass.new_ones(batch, pk.shape[2], 1), mass), dim=1)
        bias = None if mass is None else mass.log()[:, None, None, :, 0]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale,
                                             dropout_p=self.attn_drop.p if self.training else 0.)
        out = out.transpose(1, 2).reshape(batch, count, width)
        return self.proj_drop(self.proj(out)), metric


class REPVisionTransformer(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.blocks:
            block.attn.__class__ = REPAttention

    def forward(self, images, prompt, query, *, train=False, atom=True, ald=True, maximum=16,
                step=0, rng=None, minimum=.5, gamma=.001, alpha=.9, threshold=8):
        x = self.patch_embed(images)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :x.shape[1]])
        mass = x.new_ones(x.shape[0], x.shape[1], 1) if atom else None
        schedule = progressive_merges(maximum, len(self.blocks)) if atom else [0] * len(self.blocks)
        draws = torch.rand(len(self.blocks), generator=rng.generator('cpu')) if train and ald else None
        loss = x.new_zeros(1)
        trace = {'step': step, 'input_tokens': [], 'output_tokens': [], 'merged_tokens': [],
                     'executed': [], 'keeping_probabilities': [], 'prefix_tokens': []}
        for index, block in enumerate(self.blocks):
            before = x.shape[1]
            # Prospective local merge count avoids a circular decision: dropping
            # the block also skips its attention/merging. Actual counts are logged.
            planned = min(schedule[index], (before - 1) // 2)
            probability = keeping_probability(step, planned, minimum=minimum, gamma=gamma,
                                               alpha=alpha, threshold=threshold) if train and ald else 1.
            execute = draws is None or draws[index].item() < probability
            prefix_length, merged = 0, 0
            if execute:
                prefix, penalty, x = prompt(query, index, x, train=train)
                loss = loss + penalty
                prefix_length = 0 if prefix is None else prefix[0].shape[1]
                attention, metric = block.attn(block.norm1(x), prompt=prefix, mass=mass, need_metric=planned > 0)
                x = x + block.drop_path(attention)
                if planned:
                    x, mass, merged = weighted_merge(x, metric, mass, planned)
                x = x + block.drop_path(block.mlp(block.norm2(x)))
            trace['input_tokens'].append(before)
            trace['output_tokens'].append(x.shape[1])
            trace['merged_tokens'].append(merged)
            trace['executed'].append(bool(execute))
            trace['keeping_probabilities'].append(probability)
            trace['prefix_tokens'].append(prefix_length)
        return self.norm(x), loss, trace
