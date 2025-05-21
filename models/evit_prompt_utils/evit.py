# This code is a reimplementation based on the EViT methodology.
# This code has been modified for on-device continual learning by Wonseon Lim.

from typing import Tuple

import math
import torch
import torch.nn.functional as F

from models.prompt_utils.vit import Attention, Block, VisionTransformer
from models.evit_prompt_utils.utils import complement_idx


class EViTBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def forward(self, x, register_hook=False, prompt=None, keep_rate=None, tokens=None, get_idx=False, query=False):
        # Query Forward
        if query:
            attn_out = self.attn(self.norm1(x), register_hook=register_hook, query=query)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, None, None
        
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        B, N, C = x.shape

        attn_out, index, idx, cls_attn, left_tokens = self.attn(
            self.norm1(x),
            register_hook=register_hook,
            prompt=prompt,
            keep_rate=keep_rate,
            tokens=tokens
        )
        x = x + self.drop_path(attn_out)

        if index is not None:
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, idx
        return x, n_tokens, None


class EViTAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(self, x, register_hook=False, prompt=None, keep_rate=None, tokens=None, query=False):
        # Query Forward
        if query:
            return super().forward(x, register_hook=register_hook)

        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # import ipdb; ipdb.set_trace()
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if keep_rate is None:
            return x, None, None, None, None

        left_tokens = N - 1
        if keep_rate < 1 or tokens is not None:  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            if prompt is not None:
                cls_attn = attn[:, :, 0, 5:]  # [B, H, N-1]
            else:
                cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens

        return  x, None, None, None, left_tokens


def make_evit_class(transformer_class):
    class EVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, keep_rate=None, tokens=None, get_idx=False):
            # forward features
            B, _, h, w = x.shape
            if not isinstance(keep_rate, (tuple, list)):
                keep_rate = (keep_rate, ) * len(self.blocks)
            if not isinstance(tokens, (tuple, list)):
                tokens = (tokens, ) * len(self.blocks)
            assert len(keep_rate) == len(self.blocks)
            assert len(tokens) == len(self.blocks)
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
            
            # for input with another resolution, interpolate the positional embedding.
            # used for finetining a ViT on images with larger size.
            pos_embed = self.pos_embed
            if x.shape[1] != pos_embed.shape[1]:
                assert h == w  # for simplicity assume h == w
                real_pos = pos_embed[:, self.num_tokens:]
                hw = int(math.sqrt(real_pos.shape[1]))
                true_hw = int(math.sqrt(x.shape[1] - self.num_tokens))
                real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
                new_pos = F.interpolate(real_pos, size=true_hw, mode='bicubic', align_corners=False)
                new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
                pos_embed = torch.cat([pos_embed[:, :self.num_tokens], new_pos], dim=1)

            x = self.pos_drop(x + pos_embed)

            prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)

            left_tokens = []
            idxs = []

            for i, blk in enumerate(self.blocks):
                if prompt is not None:
                    if train:
                        p_list, loss, x = prompt.forward(q, i, x, train=True)
                        prompt_loss += loss
                    else:
                        p_list, _, x = prompt.forward(q, i, x, train=False)
                else:
                    p_list = None

                # Query forward
                if q is None and not self.query_merge:
                    x, left_token, idx = blk(
                        x,
                        register_hook=(register_blk == i),
                        query=True
                    )
                else:
                    x, left_token, idx = blk(
                        x,
                        register_hook=(register_blk == i),
                        prompt=p_list,
                        keep_rate=keep_rate[i],
                        tokens=tokens[i],
                        get_idx=get_idx  # Always get idx for better tracking
                    )
                    
                left_tokens.append(left_token)
                if idx is not None:
                    idxs.append(idx)

            x = self.norm(x)

            if prompt is not None:
                prompt_loss /= len(prompt.e_layers)

            return x, prompt_loss, left_tokens, idxs


    return EVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, 
    keep_rate: list = None, fuse_token: bool = False
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._evit_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    EVisionTransformer = make_evit_class(model.__class__)

    model.__class__ = EVisionTransformer
    model.keep_rate = keep_rate
    model.fuse_token = fuse_token
    model.query_merge = False
    model._evit_info = {
        "keep_rate": model.keep_rate,
        "fuse_token": model.fuse_token,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._evit_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = EViTBlock
            module._evit_info = model._evit_info
        elif isinstance(module, Attention):
            module.__class__ = EViTAttention

    # set keep rate for each block
    for i, block in enumerate(model.blocks):
        if isinstance(block, EViTBlock):
            print(f"Block {i} keep rate: {model._evit_info['keep_rate'][i]:.2f}")
            block.keep_rate = model._evit_info["keep_rate"][i]
            block.fuse_token = model._evit_info["fuse_token"]
