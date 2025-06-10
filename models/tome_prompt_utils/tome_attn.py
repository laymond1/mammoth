# This code is a reimplementation based on the ToMe methodology.
# This code has been modified for on-device continual learning by Wonseon Lim.

from typing import Tuple

import torch
from models.prompt_utils.vit import Attention, Block, VisionTransformer
from models.tome_prompt_utils.merge import bipartite_soft_matching, merge_source, merge_wavg
from models.tome_prompt_utils.utils import parse_r


def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


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

    def forward(self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None, sparse: bool = True) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # Full Token Forward
        if not sparse:
            x_attn = self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt, sparse=sparse)
            metric = None
            x = x + self._drop_path1(x_attn)
            x = x + self._drop_path2(self.mlp(self.norm2(x)))
            return x
        # Sparse Token Forward
        else:
            B, N, C = x.shape
            x_attn, cls_attn, indices, idx, r = self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt)
            x = x + self._drop_path1(x_attn)

            if r > 0:
                # Apply ToMe here
                non_cls = x[:, 1:, :]  # Exclude class token
                x_others = torch.gather(non_cls, dim=1, index=indices)
                
                compl = complement_idx(idx, N - 1)
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)

            x = x + self._drop_path2(self.mlp(self.norm2(x)))
            return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None, sparse: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Full Token Forward
        if not sparse:
            return super().forward(x, register_hook=register_hook, prompt=prompt)

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
            # attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if prompt is not None:
            cls_attn = attn[:, :, 0, 5:]
        else:
            cls_attn = attn[:, :, 0, 1:]
        cls_attn = cls_attn.mean(dim=1) # [B, N-1]

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # We can only reduce by a maximum of 50% tokens
            protected = 5 if prompt is not None else 1
            r = min(r, (N - protected) // 2)
            _, idx = torch.topk(cls_attn, N-r, dim=1, largest=True, sorted=True)
            indices = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        return x, cls_attn, indices, idx, r


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, feat=False) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
    
            x = x + self.pos_embed[:,:x.size(1),:]
            x = self.pos_drop(x)

            prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)

            for i,blk in enumerate(self.blocks):

                if prompt is not None:
                    if train:
                        p_list, loss, x = prompt.forward(q, i, x, train=True)
                        prompt_loss += loss
                    else:
                        p_list, _, x = prompt.forward(q, i, x, train=False)
                
                else:
                    p_list = None

                # Forward for prompt 
                if prompt is not None:
                    if train:
                        # Prompt Forward for Prompt Training
                        if self.prompt_prompt_sparse:
                            # Sparse Token Forward (Train for Prompt)
                            x = blk(x, register_blk==i, prompt=p_list)
                        else:
                            # Full Token Forward (Train for Prompt)
                            x = blk(x, register_blk==i, prompt=p_list, sparse=False)
                    else:
                        # Prompt Forward for Inference or Classifier Training
                        if self.head_prompt_sparse and feat:
                            # Sparse Token Forward (Train for Classifier, feat=True)
                            x = blk(x, register_blk==i, prompt=p_list)
                        elif self.test_prompt_sparse:
                            # Sparse Token Forward (Inference)
                            x = blk(x, register_blk==i, prompt=p_list)
                        else:
                            # Full Token Forward (Inference)
                            x = blk(x, register_blk==i, prompt=p_list, sparse=False)
                # Forward for query
                else:
                    if train:
                        # Query Forward for Prompt Training
                        if self.prompt_query_sparse:
                            # Sparse Token Forward (Train for Prompt)
                            x = blk(x, register_blk==i, prompt=p_list)
                        else:
                            # Full Token Forward (Train for Prompt)
                            x = blk(x, register_blk==i, prompt=p_list, sparse=False)
                    else:
                        # Query Forward for Inference or Classifier Training
                        if self.head_query_sparse and feat:
                            # Sparse Token Forward (Train for Classifier, feat=True)
                            x = blk(x, register_blk==i, prompt=p_list)
                        elif self.test_query_sparse:
                            # Sparse Token Forward (Inference)
                            x = blk(x, register_blk==i, prompt=p_list)
                        else:
                            # Full Token Forward (Inference)
                            x = blk(x, register_blk==i, prompt=p_list, sparse=False)

            x = self.norm(x)

            if prompt is not None:
                prompt_loss /= len(prompt.e_layers)
            
            return x, prompt_loss

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model.prompt_prompt_sparse = False
    model.head_prompt_sparse = False
    model.test_prompt_sparse = False
    model.prompt_query_sparse = False
    model.head_query_sparse = False
    model.test_query_sparse = False
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
            module._tome_info = model._tome_info
