"""CPS-Prompt encoder: the CODA prefixes with explicit attention and a patch-importance output.

CPS samples patches once at the input during its joint phase; that sampling
stays its own. Per-block reduction is meant for inference: during training
the importance scores must cover every patch, which they do only at degree 0.
"""

import torch

from models.cps_prompt_utils import vit as cps_vit

from .adapters import _Compressed
from .reducers import Signals


class CPSBlock(_Compressed, cps_vit.CPSBlock):
    def forward(self, x, prompt=None, importance=False):
        controller, count = self._enter(x.shape[1])
        if count == 0 and controller.size is None:
            return super().forward(x, prompt=prompt, importance=importance)
        return self._reduced(x, controller, count, prompt, importance)

    def _reduced(self, x, controller, count, prompt=None, importance=False):
        attn = self.attn
        y = self.norm1(x)
        batch, tokens, width = y.shape
        qkv = attn.qkv(y).reshape(batch, tokens, 3, attn.num_heads, width // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        size, prefix = controller.size, 0
        if prompt is not None:
            pk, pv = [value.reshape(batch, -1, attn.num_heads, width // attn.num_heads).transpose(1, 2) for value in prompt]
            k, v = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
            prefix = pk.shape[2]
            if size is not None:
                # Prefixes are not image tokens: unit size, never merged or pruned.
                size = torch.cat((size.new_ones(batch, prefix, 1), size), dim=1)
        bias = self._bias(size, q)
        # Same attention as cps_prompt_utils.vit.CPSAttention, plus the size bias.
        attention = (q @ k.transpose(-2, -1)) * attn.scale
        if bias is not None:
            attention = attention + bias
        attention = attn.attn_drop(attention.softmax(dim=-1))
        y = attn.proj_drop(attn.proj((attention @ v).transpose(1, 2).reshape(batch, tokens, width)))
        scores = None
        if importance:
            # One score per token entering this block; unused outside CPS training.
            value_norm = torch.linalg.vector_norm(v.transpose(1, 2).reshape(batch, v.shape[2], width), dim=2)
            scores = (attention[:, :, 0].sum(dim=1) * value_norm)[:, 1:]
        x = x + self.drop_path(y)
        if count:
            signals = Signals(q, k, attn.scale, bias, prefix=prefix, block=self._tc_index, depth=controller.depth)
            x, controller.size = controller.operator(x, controller.size, count, signals)
        return x + self.drop_path(self.mlp(self.norm2(x))), scores


FAMILY = ('feat', cps_vit.CPSVisionTransformer, cps_vit.CPSBlock, cps_vit.CPSAttention, CPSBlock)
