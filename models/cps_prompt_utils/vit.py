# Adapted from CPS-Prompt, copyright (c) 2026 Wonseon Lim, Jaesung Lee, Dae-Won Kim.
# MIT license and pinned source attribution: see SOURCE.json and UPSTREAM_LICENSE.
"""Explicit author attention with CLS-attention/value-norm importance."""

import torch

from models.coda_prompt_utils.vit import Attention, Block, VisionTransformer


class CPSAttention(Attention):
    def forward(self, x, prompt=None, importance=False):
        batch, count, width = x.shape
        qkv = self.qkv(x).reshape(batch, count, 3, self.num_heads, width // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if prompt is not None:
            pk, pv = [value.reshape(batch, -1, self.num_heads, width // self.num_heads).transpose(1, 2) for value in prompt]
            k, v = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
        attention = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        out = (attention @ v).transpose(1, 2).reshape(batch, count, width)
        out = self.proj_drop(self.proj(out))
        scores = None
        if importance:
            value_norm = torch.linalg.vector_norm(v.transpose(1, 2).reshape(batch, v.shape[2], width), dim=2)
            scores = (attention[:, :, 0].sum(dim=1) * value_norm)[:, 1:]
        return out, scores


class CPSBlock(Block):
    def forward(self, x, prompt=None, importance=False):
        attention, scores = self.attn(self.norm1(x), prompt=prompt, importance=importance)
        x = x + self.drop_path(attention)
        return x + self.drop_path(self.mlp(self.norm2(x))), scores


class CPSVisionTransformer(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.blocks:
            block.__class__ = CPSBlock
            block.attn.__class__ = CPSAttention

    def forward(self, x, prompt=None, q=None, train=False, task_id=None, sampler=None, scores=None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :x.shape[1]])
        if prompt is not None and train:
            x = sampler(x, scores)
        token_count = x.shape[1]
        loss = x.new_zeros(1)
        importance = None
        for index, block in enumerate(self.blocks):
            prefix = None
            if prompt is not None:
                prefix, penalty, x = prompt(q, index, x, train=train, task_id=task_id)
                loss = loss + penalty
            x, importance = block(x, prompt=prefix, importance=prompt is None and index == len(self.blocks) - 1)
        if prompt is not None:
            loss = loss / len(prompt.e_layers)
        return self.norm(x), loss, importance, token_count
