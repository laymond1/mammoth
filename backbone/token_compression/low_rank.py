"""InfLoRA and EWC-LoRA encoders: K/V low-rank updates folded into the projection of every block."""

import torch
from torch.nn import functional as F

from models.low_rank_utils import net_ewclora, net_inflora, vit_ewclora, vit_inflora

from .adapters import _Compressed
from .reducers import Signals


class _LowRank(_Compressed):
    def _attend(self, x, controller, count, q, k, v):
        """Explicit attention of the source blocks with the size bias, then reduction and MLP."""
        attn = self.attn
        batch, tokens, width = x.shape
        scores = (q @ k.transpose(-2, -1)) * attn.scale
        bias = self._bias(controller.size, q)
        if bias is not None:
            scores = scores + bias
        scores = attn.attn_drop(scores.softmax(dim=-1))
        y = attn.proj_drop(attn.proj((scores @ v).transpose(1, 2).reshape(batch, tokens, width)))
        x = x + self.drop_path1(self.ls1(y))
        if count:
            # The metric is the key attention really uses, low-rank update included.
            signals = Signals(q, k, attn.scale, bias, block=self._tc_index, depth=controller.depth)
            x, controller.size = controller.operator(x, controller.size, count, signals)
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class InfLoRABlock(_LowRank, vit_inflora.Block):
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        controller, count = self._enter(x.shape[1])
        if count == 0 and controller.size is None:
            return super().forward(x, task, register_hook=register_hook, get_feat=get_feat, get_cur_feat=get_cur_feat)
        if register_hook or get_feat:
            raise NotImplementedError('Attention-map hooks and get_feat are not supported with token compression')
        return self._reduced(x, controller, count, task, get_cur_feat)

    def _reduced(self, x, controller, count, task, get_cur_feat=False):
        attn = self.attn
        y = self.norm1(x)
        batch, tokens, width = y.shape
        if get_cur_feat:
            # Same running input covariance as vit_inflora.Attention_LoRA, over the tokens this block receives.
            attn.cur_matrix = (attn.cur_matrix * attn.n_cur_matrix
                               + torch.bmm(y.detach().permute(0, 2, 1), y.detach()).sum(dim=0).cpu()
                               ) / (attn.n_cur_matrix + batch * tokens)
            attn.n_cur_matrix += batch * tokens
        qkv = attn.qkv(y).reshape(batch, tokens, 3, attn.num_heads, width // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if task > -0.5:
            weight_k = torch.stack([torch.mm(attn.lora_B_k[t].weight, attn.lora_A_k[t].weight) for t in range(task + 1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(attn.lora_B_v[t].weight, attn.lora_A_v[t].weight) for t in range(task + 1)], dim=0).sum(dim=0)
            k = k + F.linear(y, weight_k).reshape(batch, tokens, attn.num_heads, width // attn.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(y, weight_v).reshape(batch, tokens, attn.num_heads, width // attn.num_heads).permute(0, 2, 1, 3)
        return self._attend(x, controller, count, q, k, v)


class EWCLoRABlock(_LowRank, vit_ewclora.Block):
    def forward(self, x, use_new=True, register_hook=False):
        controller, count = self._enter(x.shape[1])
        if count == 0 and controller.size is None:
            return super().forward(x, use_new=use_new, register_hook=register_hook)
        return self._reduced(x, controller, count, use_new, register_hook)

    def _reduced(self, x, controller, count, use_new=True, register_hook=False):
        attn = self.attn
        y = self.norm1(x)
        batch, tokens, width = y.shape
        qkv = attn.qkv(y)
        # Same accumulated and new-task updates as vit_ewclora.Attention_LoRA.
        qkv[:, :, attn.dim: 2 * attn.dim] += attn.lora_B_k(attn.lora_A_k(y))
        qkv[:, :, 2 * attn.dim:] += attn.lora_B_v(attn.lora_A_v(y))
        if use_new:
            delta_w_k_new = attn.lora_new_B_k.weight @ attn.lora_new_A_k.weight
            delta_w_v_new = attn.lora_new_B_v.weight @ attn.lora_new_A_v.weight
            qkv[:, :, attn.dim: 2 * attn.dim] += y @ delta_w_k_new.t()
            qkv[:, :, 2 * attn.dim:] += y @ delta_w_v_new.t()
            if register_hook:
                # The Fisher estimate reads these weight-space gradients.
                delta_w_k_new.register_hook(attn.save_grad('delta_w_k_new'))
                delta_w_v_new.register_hook(attn.save_grad('delta_w_v_new'))
        qkv = qkv.reshape(batch, tokens, 3, attn.num_heads, width // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return self._attend(x, controller, count, q, k, v)


INF_LORA = ('image_encoder', net_inflora.ViT, vit_inflora.Block, vit_inflora.Attention_LoRA, InfLoRABlock)
EWC_LORA = ('image_encoder', net_ewclora.ViT, vit_ewclora.Block, vit_ewclora.Attention_LoRA, EWCLoRABlock)
