"""CODA-Prompt encoder: key/value prefixes inside attention, two passes per input."""

import torch
from torch.nn import functional as F

from models.coda_prompt_utils import vit as coda_vit

from .adapters import _Compressed
from .reducers import Signals


class CodaBlock(_Compressed, coda_vit.Block):
    def forward(self, x, prompt=None):
        controller, count = self._enter(x.shape[1])
        if count == 0 and controller.size is None:
            return super().forward(x, prompt=prompt)
        return self._reduced(x, controller, count, prompt)

    def _reduced(self, x, controller, count, prompt=None):
        attn = self.attn
        y = self.norm1(x)
        batch, tokens, width = y.shape
        qkv = attn.qkv(y).reshape(batch, tokens, 3, attn.num_heads, width // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        size, prefix = controller.size, 0
        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(batch, -1, attn.num_heads, width // attn.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(batch, -1, attn.num_heads, width // attn.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
            prefix = pk.shape[2]
            if size is not None:
                # Prefixes are not image tokens: unit size, never merged or pruned.
                size = torch.cat((size.new_ones(batch, prefix, 1), size), dim=1)
        bias = self._bias(size, q)
        # Same call as coda_prompt_utils.vit.Attention, plus the size bias.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=attn.scale, dropout_p=attn.attn_drop.p)
        y = attn.proj_drop(attn.proj(y.transpose(1, 2).reshape(batch, tokens, width)))
        x = x + self.drop_path(y)
        if count:
            signals = Signals(q, k, attn.scale, bias, prefix=prefix, block=self._tc_index, depth=controller.depth)
            x, controller.size = controller.operator(x, controller.size, count, signals)
        return x + self.drop_path(self.mlp(self.norm2(x)))


FAMILY = ('feat', coda_vit.VisionTransformer, coda_vit.Block, coda_vit.Attention, CodaBlock)
