# This code is a reimplementation based on the ToMe methodology.
# This code has been modified for on-device continual learning by Wonseon Lim.

from typing import Tuple

import torch
from models.prompt_utils.vit import Attention, Block, VisionTransformer


class PatchSampling(torch.nn.Module):
    """
    This is based on the PatchDropout implementation.
    PatchDropout: https://arxiv.org/abs/2208.07220
    Code: https://github.com/yueliukth/PatchDropout
    """
    def __init__(self, keep_rate=0.5, sampling="uniform", token_shuffling=False, temperature=1.0):
        super().__init__()
        assert 0 < keep_rate <=1, "The keep_rate must be in (0,1]"
        
        self.keep_rate = keep_rate
        self.sampling = sampling
        self.token_shuffling = token_shuffling
        self.temperature = temperature

    def forward(self, x, attn_scores=None, force_drop=True):
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
        patch_mask = self.get_mask(x, attn_scores)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    def get_mask(self, x, attn_scores=None):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        elif self.sampling == "attention":
            return self.attention_mask(x, attn_scores)
        elif self.sampling == "significance_score":
            return self.significance_score_mask(x, attn_scores)
        else:
            return NotImplementedError(f"PatchSampling does ot support {self.sampling} sampling")
    
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

    def attention_mask(self, x, attn_logits):
        """
        Returns an id-mask using attention scores
        """
        N, L, D = x.shape
        _L = L -1
        keep = int(_L * self.keep_rate)

        # Apply temperature scaling
        cls_attn_logits = attn_logits[:, :, 0, 1:].mean(dim=1)  # shape: (B, 196)
        scaled_logits = cls_attn_logits / self.temperature  # shape: (B, 196)

        # Convert to probabilities (softmax)
        probs = torch.nn.functional.softmax(scaled_logits, dim=1)  # (B, 196)

        # Sample 'keep' tokens from 196 using attention probs
        patch_mask = torch.multinomial(probs, num_samples=keep, replacement=False)  # (B, keep)
        patch_mask = patch_mask + 1  # shift by 1 to exclude CLS token

        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]

        return patch_mask
    
    def significance_score_mask(self, x, attn_scores):
        """
        Returns an id-mask using attention scores
        """
        N, L, D = x.shape
        _L = L - 1
        keep = int(_L * self.keep_rate)

        # Apply temperature scaling
        attn_scores = attn_scores / self.temperature
        attn_scores = torch.softmax(attn_scores, dim=1)

        # Sample 'keep' tokens from 196 using attention probs
        patch_mask = torch.multinomial(attn_scores, num_samples=keep, replacement=False)  # (B, keep)
        patch_mask = patch_mask + 1
        
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]

        return patch_mask


class PaSamplingBlock(Block):
    """
    Modifications:
     - Apply PaSampling between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None, sparse: bool = True) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # Full Token Forward
        x_attn, attn_scores = self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt)
        x = x + self._drop_path1(x_attn)
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        
        return x, attn_scores


class PaSamplingAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None, sparse: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Full Token Forward
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

        attn_logit = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_logit.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # if register_hook:
        #     self.save_attention_map(attn)
            # attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
    
        if register_hook:
            if self._pasampling_info['sampling'] == 'attention':
                return x, attn_logit
            elif self._pasampling_info['sampling'] == 'significance_score':
                v_norm = torch.linalg.norm(
                    v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
                )  # value norm of size [B x T]
                significance_score = attn[:, :, 0].sum(
                    dim=1
                )  # attention weights of CLS token of size [B x T]
                significance_score = significance_score * v_norm  # [B x T]
                significance_score = significance_score[:, 1:]  # [B x T-1]

                # significance_score = significance_score / significance_score.sum(
                #     dim=1, keepdim=True
                # )  # [B x T-1]

                return x, significance_score
            else:
                return x, None
        else:
            return x, None


def make_pasampling_class(transformer_class):
    class PaSamplingVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, feat=False, q_attn_scores=None) -> torch.Tensor:
            self.patchsampling = PatchSampling(keep_rate=self.keep_rate, sampling=self.sampling, token_shuffling=False, temperature=self.temperature)
            self.query_patchsampling = PatchSampling(keep_rate=self.keep_rate, sampling="uniform", token_shuffling=False)

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
                        x = self.patchsampling(x, q_attn_scores)
                    else:
                        # Full Token Forward (Train for Prompt)
                        pass
                else:
                    # Prompt Forward for Inference or Classifier Training
                    if self.head_prompt_sparse and feat:
                        # Sparse Token Forward (Train for Classifier, feat=True)
                        x = self.patchsampling(x, q_attn_scores)
                    elif self.test_prompt_sparse:
                        # Sparse Token Forward (Inference)
                        x = self.patchsampling(x, q_attn_scores)
                    else:
                        # Full Token Forward (Inference)
                        pass
            # Forward for query
            else:
                if train:
                    # Query Forward for Prompt Training
                    if self.prompt_query_sparse:
                        # Sparse Token Forward (Train for Prompt)
                        x = self.query_patchsampling(x)
                    else:
                        # Full Token Forward (Train for Prompt)
                        pass
                else:
                    # Query Forward for Inference or Classifier Training
                    if self.head_query_sparse and feat:
                        # Sparse Token Forward (Train for Classifier, feat=True)
                        x = self.query_patchsampling(x)
                    elif self.test_query_sparse:
                        # Sparse Token Forward (Inference)
                        x = self.query_patchsampling(x)
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

                if prompt is not None:
                    x, attn_scores = blk(x, register_hook=(i == register_blk), prompt=p_list) # attn_scores is None
                else:
                    x, attn_scores = blk(x, register_hook=(i == 11), prompt=p_list) # attn_scores is only for the last block

            x = self.norm(x)

            if prompt is not None:
                prompt_loss /= len(prompt.e_layers)
            
            return x, prompt_loss, attn_scores

    return PaSamplingVisionTransformer


def apply_patch(
    model: VisionTransformer, sampling: str = 'uniform', trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies PatchSampling to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._pasampling_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    PaSamplingVisionTransformer = make_pasampling_class(model.__class__)

    model.__class__ = PaSamplingVisionTransformer
    model.keep_rate = 0.0
    model.temperature = 1.0
    model.sampling = sampling
    model.prompt_prompt_sparse = False
    model.head_prompt_sparse = False
    model.test_prompt_sparse = False
    model.prompt_query_sparse = False
    model.head_query_sparse = False
    model.test_query_sparse = False
    model._pasampling_info = {
        "sampling": model.sampling,
        "token_shuffling": False,
        "temperature": model.temperature,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._pasampling_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = PaSamplingBlock
        elif isinstance(module, Attention):
            module.__class__ = PaSamplingAttention
            module._pasampling_info = model._pasampling_info


def exponential_temperature(epoch, total_epochs, start_temp=0.1, end_temp=5.0):
    ratio = epoch / total_epochs
    return start_temp * ((end_temp / start_temp) ** ratio)