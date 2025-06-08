# This code is a reimplementation based on the ToPrune methodology.
# This code has been modified for on-device continual learning by Wonseon Lim.

from typing import Tuple

import torch
from models.prompt_utils.vit import Attention, Block, VisionTransformer
from models.toprune_prompt_utils.patchdropout import PatchDropout
from models.tome_prompt_utils.merge import bipartite_soft_matching, merge_source, merge_wavg
from models.tome_prompt_utils.utils import parse_r


class PatchDropout(torch.nn.Module):
    """
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    Code: https://github.com/yueliukth/PatchDropout
    """
    def __init__(self, keep_rate=0.5, sampling="uniform", token_shuffling=False):
        super().__init__()
        assert 0 < keep_rate <=1, "The keep_rate must be in (0,1]"
        
        self.keep_rate = keep_rate
        self.sampling = sampling
        self.token_shuffling = token_shuffling

    def forward(self, x, force_drop=True):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        # if not self.training and not force_drop: return x        
        if not force_drop: return x        
        if self.keep_rate == 1: return x

        # batch, length, dim
        N, L, D = x.shape
        
        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    def get_mask(self, x):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        else:
            return NotImplementedError(f"PatchDropout does ot support {self.sampling} sampling")
    
    def uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        keep = int(_L * self.keep_rate)
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask


class DropToMeBlock(Block):
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
        attn_size = self._dropmerge_info["size"] if self._dropmerge_info["prop_attn"] else None
        # Full Token Forward
        if not sparse:
            x_attn = self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt, size=attn_size, sparse=sparse)
            metric = None
            x = x + self._drop_path1(x_attn)
            x = x + self._drop_path2(self.mlp(self.norm2(x)))
            return x
        # Sparse Token Forward
        else:
            x_attn, metric = self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt, size=attn_size)
            x = x + self._drop_path1(x_attn)

            r = self._dropmerge_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._dropmerge_info["class_token"],
                    self._dropmerge_info["distill_token"],
                )
                if self._dropmerge_info["trace_source"]:
                    self._dropmerge_info["source"] = merge_source(
                        merge, x, self._dropmerge_info["source"]
                    )
                x, self._dropmerge_info["size"] = merge_wavg(merge, x, self._dropmerge_info["size"])

            x = x + self._drop_path2(self.mlp(self.norm2(x)))
            return x


class DropToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None, size: torch.Tensor = None, sparse: bool = True
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


def make_dropmerge_class(transformer_class):
    class DropMergeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, feat=False) -> torch.Tensor:
            # self.patchdrop = PatchDropout(
            #     keep_rate=1-self.drop_rate, 
            #     sampling=self.sampling, 
            #     token_shuffling=self.token_shuffling
            # )
            if prompt is not None:
                self._dropmerge_info["r"] = parse_r(len(self.blocks), self.prompt_r)
            else:
                self._dropmerge_info["r"] = parse_r(len(self.blocks), self.query_r)
            self._dropmerge_info["size"] = None
            self._dropmerge_info["source"] = None


            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
    
            x = x + self.pos_embed[:,:x.size(1),:]
            x = self.pos_drop(x)

            # Forward for prompt 
            if prompt is not None:
                if train:
                    # Prompt Forward for Prompt Training
                    if self.prompt_prompt_sparse:
                        # Sparse Token Forward (Train for Prompt)
                        x = PatchDropout(
                                keep_rate=1-self.drop_rate, 
                                sampling=self.sampling, 
                                token_shuffling=self.token_shuffling
                            )(x)
                    else:
                        # Full Token Forward (Train for Prompt)
                        pass
                else:
                    # Query Forward for Inference or Classifier Training
                    if self.head_prompt_sparse and feat:
                        # Sparse Token Forward (Train for Classifier, feat=True)
                        x = PatchDropout(
                                keep_rate=1-self.drop_rate, 
                                sampling=self.sampling, 
                                token_shuffling=self.token_shuffling
                            )(x)
                    elif self.test_prompt_sparse:
                        # Sparse Token Forward (Inference)
                        x = PatchDropout(
                                keep_rate=1-self.drop_rate, 
                                sampling=self.sampling, 
                                token_shuffling=self.token_shuffling
                            )(x)
                    else:
                        # Full Token Forward (Inference)
                        pass
            # Forward for query
            else:
                if train:
                    # Query Forward for Prompt Training
                    if self.prompt_query_sparse:
                        # Sparse Token Forward (Train for Prompt)
                        x = PatchDropout(
                                keep_rate=1-self.drop_rate, 
                                sampling=self.sampling, 
                                token_shuffling=self.token_shuffling
                            )(x)
                    else:
                        # Full Token Forward (Train for Prompt)
                        pass
                else:
                    # Query Forward for Inference or Classifier Training
                    if self.head_query_sparse and feat:
                        # Sparse Token Forward (Train for Classifier, feat=True)
                        x = PatchDropout(
                                keep_rate=1-self.drop_rate, 
                                sampling=self.sampling, 
                                token_shuffling=self.token_shuffling
                            )(x)
                    elif self.test_query_sparse:
                        # Sparse Token Forward (Inference)
                        x = PatchDropout(
                                keep_rate=1-self.drop_rate, 
                                sampling=self.sampling, 
                                token_shuffling=self.token_shuffling
                            )(x)
                    else:
                        # Full Token Forward (Inference)
                        pass

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
                if prompt is not None and not self.late_merge:
                    if train:
                        # Prompt Forward for Prompt Training
                        if self.prompt_prompt_sparse:
                            # Sparse Token Forward (Train for Prompt)
                            x = blk(x, register_blk==i, prompt=p_list)
                        else:
                            # Full Token Forward (Train for Prompt)
                            x = blk(x, register_blk==i, prompt=p_list, sparse=False)
                    else:
                        # Query Forward for Inference or Classifier Training
                        if self.head_prompt_sparse and feat:
                            # Sparse Token Forward (Train for Classifier, feat=True)
                            x = blk(x, register_blk==i, prompt=p_list)
                        elif self.test_prompt_sparse:
                            # Sparse Token Forward (Inference)
                            x = blk(x, register_blk==i, prompt=p_list)
                        else:
                            # Full Token Forward (Inference)
                            x = blk(x, register_blk==i, prompt=p_list, sparse=False)
                # Late Merge Forward
                elif prompt is not None and self.late_merge:
                    if p_list is not None:
                        # Full Token Forward
                        x = blk(x, register_blk==i, prompt=p_list, sparse=False)
                    else:
                        # Sparse Token Forward
                        x = blk(x, register_blk==i, prompt=p_list)
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

    return DropMergeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToPrune to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._dropmerge_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    DropMergeVisionTransformer = make_dropmerge_class(model.__class__)

    model.__class__ = DropMergeVisionTransformer
    model.drop_rate = 0.0
    model.late_merge = False
    model.prompt_r = 0
    model.query_r = 0
    model.patchdrop = None
    model.sampling = "uniform"
    model.token_shuffling = False
    model.prompt_prompt_sparse = False
    model.head_prompt_sparse = False
    model.test_prompt_sparse = False
    model.prompt_query_sparse = False
    model.head_query_sparse = False
    model.test_query_sparse = False
    model._dropmerge_info = {
        "drop_rate": model.drop_rate,
        "prompt_r": model.prompt_r,
        "query_r": model.query_r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._dropmerge_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = DropToMeBlock
            module._dropmerge_info = model._dropmerge_info
        elif isinstance(module, Attention):
            module.__class__ = DropToMeAttention