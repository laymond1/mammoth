from typing import Tuple

import torch
from timm.models.layers import DropPath
from models.prompt_utils.vit import Attention, Block, VisionTransformer
from models.rep_utils.merge import bipartite_soft_matching, merge_source, merge_wavg
from models.rep_utils.utils import parse_r, AToM_parse_r, parse_theta


class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt, size=attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # import ipdb; ipdb.set_trace()
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            if prompt is not None:
                attn[:, :, :, 4:] = attn[:, :, :, 4:] + size.log()[:, None, None, :, 0]
            else:
                attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            # attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if prompt is not None:
            k = k[:, :, 4:, :]
        # Return k as well here
        return x, k.mean(1)


def make_rep_class(transformer_class):
    class REPVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, feat=False) -> torch.Tensor:
            if self._tome_info["tome_type"] == 'tome':
                self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            elif self._tome_info["tome_type"] == 'atom':
                self._tome_info["r"] = AToM_parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._pld_info["step"] += 1
            self._pld_info["theta"] = parse_theta(self._tome_info["r"], **self._pld_info)

            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
    
            x = x + self.pos_embed[:,:x.size(1),:]
            x = self.pos_drop(x)

            prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)

            theta = self._pld_info["theta"]  # keep probs per layer
            for i, blk in enumerate(self.blocks):

                # Prompt
                if prompt is not None:
                    if train:
                        p_list, loss, x = prompt.forward(q, i, x, train=True)
                        prompt_loss += loss
                    else:
                        p_list, _, x = prompt.forward(q, i, x, train=False)

                else:
                    p_list = None

                blk.drop_path.drop_prob = 1-theta[i]
                x = blk(x, register_blk==i, prompt=p_list)

            x = self.norm(x)

            if prompt is not None:
                prompt_loss /= len(prompt.e_layers)

            return x, prompt_loss

        
        def get_step(self):
            return self._step
        
        def update_step(self):
            self._step += 1

    return REPVisionTransformer


def apply_patch_layer(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True,
    tome_type: str = "tome", use_ald: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    REPVisionTransformer = make_rep_class(model.__class__)

    model.__class__ = REPVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "tome_type": tome_type
    }
    model.gamma = 0.001 # initial gamma following deepspeed hyp
    model.theta_min = 0.5 # minimum probability
    model.tau = 12 # 12 (Base) / 16 (Large)
    model._step = 0
    model._pld_info = {
        "gamma": model.gamma,
        "step": model._step,
        "theta": None,
        "theta_min": model.theta_min,
        "tau": model.tau,
        "use_ald": use_ald
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module.drop_path = DropPath(model.theta_min) if use_ald else DropPath(0.0)
            module._tome_info = model._tome_info
            module._pld_info = model._pld_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
