"""EASE encoder: a frozen timm ViT run once per task adapter by its backbone wrapper.

The wrapper loops over the blocks with ``adapter_block_forward`` instead of
calling them, so it is patched too. Every adapter pass starts from the full
token sequence and is reduced on its own.
"""

from timm.models import vision_transformer as timm_vit

from backbone import ease_vit
from models.ease_utils.vit import adapter_block_forward

from .adapters import _Compressed
from .reducers import Signals


class EaseBlock(_Compressed, timm_vit.Block):
    def forward(self, x):
        return self.adapted(x)

    def adapted(self, x, adapter=None):
        controller, count = self._enter(x.shape[1])
        if count == 0 and controller.size is None:
            return adapter_block_forward(self, x, adapter)
        return self._reduced(x, controller, count, adapter)

    def _reduced(self, x, controller, count, adapter=None):
        attn = self.attn
        y = self.norm1(x)
        batch, tokens, width = y.shape
        qkv = attn.qkv(y).reshape(batch, tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        # Same explicit attention as timm with fused_attn off, plus the size bias.
        scores = (q * attn.scale) @ k.transpose(-2, -1)
        bias = self._bias(controller.size, q)
        if bias is not None:
            scores = scores + bias
        scores = attn.attn_drop(scores.softmax(dim=-1))
        y = attn.proj_drop(attn.proj((scores @ v).transpose(1, 2).reshape(batch, tokens, width)))
        x = x + self.drop_path1(self.ls1(y))
        if count:
            signals = Signals(q, k, attn.scale, bias, block=self._tc_index, depth=controller.depth)
            x, controller.size = controller.operator(x, controller.size, count, signals)
        # As in adapter_block_forward: the adapter reads the post-attention stream, in parallel with the MLP.
        adapter_output = adapter(x) if adapter is not None else None
        branch = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if adapter_output is not None:
            branch = branch + adapter_output
        return x + branch


class EaseHost(ease_vit.EaseViT):
    def _run_adapter_space(self, tokens, adapter_index):
        if adapter_index < -1 or adapter_index >= self.num_adapters:
            raise IndexError("EASE adapter index is outside the registered task spaces")
        adapters = self.adapters[adapter_index] if adapter_index >= 0 else None
        x = tokens
        for index, block in enumerate(self.backbone.blocks):
            x = block.adapted(x, adapters[index] if adapters is not None else None)
        return self.backbone.norm(x)[:, 0]


FAMILY = ('backbone.backbone', timm_vit.VisionTransformer, timm_vit.Block, timm_vit.Attention, EaseBlock,
          ('backbone', ease_vit.EaseViT, EaseHost))
