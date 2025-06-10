# This code is a reimplementation based on the ToPrune methodology.
# This code has been modified for on-device continual learning by Wonseon Lim.

from typing import Tuple

import torch
from models.prompt_utils.vit import Attention, Block, VisionTransformer
from models.toprune_prompt_utils.patchdropout import PatchDropout


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


def make_patchdrop_class(transformer_class):
    class PatchDropVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, feat=False) -> torch.Tensor:
            self.patchdrop = PatchDropout(
                keep_rate=self.keep_rate, 
                sampling=self.sampling, 
                token_shuffling=self.token_shuffling
            )

            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
    
            x = x + self.pos_embed[:,:x.size(1),:]
            x = self.pos_drop(x)

            if train:
                x = self.patchdrop(x)
            else:
                x = self.patchdrop(x, force_drop=False)

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

                x = blk(x, register_blk==i, prompt=p_list)

            x = self.norm(x)

            if prompt is not None:
                prompt_loss /= len(prompt.e_layers)
            
            return x, prompt_loss

    return PatchDropVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToPrune to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._patchdrop_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    PatchDropVisionTransformer = make_patchdrop_class(model.__class__)

    model.__class__ = PatchDropVisionTransformer
    model.keep_rate = 0.0
    model.patchdrop = None
    model.sampling = "uniform"
    model.token_shuffling = False
    model._patchdrop_info = {
        "keep_rate": model.keep_rate,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._patchdrop_info["distill_token"] = True