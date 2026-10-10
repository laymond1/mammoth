"""SD-LoRA encoder: per-task K/V LoRA added in place to the joint projection."""

import torch

from models.low_rank_utils import net_sdlora, vit_sdlora

from .adapters import _Compressed
from .reducers import Signals


class SDLoRABlock(_Compressed, vit_sdlora.Block):
    def forward(self, x, task_id, register_hook=False):
        controller, count = self._enter(x.shape[1])
        if count == 0 and controller.size is None:
            return super().forward(x, task_id, register_hook=register_hook)
        if register_hook:
            raise NotImplementedError('Attention-map hooks are not supported with token compression')
        return self._reduced(x, controller, count, task_id)

    def _reduced(self, x, controller, count, task_id):
        attn = self.attn
        y = self.norm1(x)
        batch, tokens, width = y.shape
        qkv = attn.qkv(y)
        # Same K/V update as vit_sdlora.Attention_LoRA.
        new_k, new_v = 0, 0
        for i in range(task_id):
            new_k += attn.scaling_factor_prev[i](attn.lora_B_k[i](attn.lora_A_k[i](y)) / (torch.norm(attn.lora_A_k[i].weight) * torch.norm(attn.lora_B_k[i].weight)))
            new_v += attn.scaling_factor_prev[i](attn.lora_B_v[i](attn.lora_A_v[i](y)) / (torch.norm(attn.lora_A_v[i].weight) * torch.norm(attn.lora_B_v[i].weight)))
        new_k += attn.scaling_factor[0](attn.lora_B_k[task_id](attn.lora_A_k[task_id](y)))
        new_v += attn.scaling_factor[0](attn.lora_B_v[task_id](attn.lora_A_v[task_id](y)))
        qkv[:, :, attn.dim: 2 * attn.dim] += new_k
        qkv[:, :, 2 * attn.dim:] += new_v
        qkv = qkv.reshape(batch, tokens, 3, attn.num_heads, width // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) * attn.scale
        bias = self._bias(controller.size, q)
        if bias is not None:
            scores = scores + bias
        scores = attn.attn_drop(scores.softmax(dim=-1))
        y = attn.proj_drop(attn.proj((scores @ v).transpose(1, 2).reshape(batch, tokens, width)))
        x = x + self.drop_path1(self.ls1(y))
        if count:
            # The metric is the key attention really uses, LoRA update included.
            signals = Signals(q, k, attn.scale, bias, block=self._tc_index, depth=controller.depth)
            x, controller.size = controller.operator(x, controller.size, count, signals)
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


FAMILY = ('image_encoder', net_sdlora.ViT, vit_sdlora.Block, vit_sdlora.Attention_LoRA, SDLoRABlock)
