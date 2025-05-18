from typing import Tuple

import torch
from models.prompt_utils.vit import Attention, Block, VisionTransformer
from .patch.vit import ToMeBlock, ToMeAttention
from .utils import parse_r, AToM_parse_r, parse_theta


def make_tome_class(transformer_class):
    class REPVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            if self._tome_info["tome_type"] == 'tome':
                self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            elif self._tome_info["tome_type"] == 'atom':
                self._tome_info["r"] = AToM_parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._pld_info["step"] += 1
            self._pld_info["theta"] = parse_theta(self._tome_info["r"], **self._pld_info)

            if not self._pld_info["use_ald"]:
                # filter register_blk from kwdargs
                if "register_blk" in kwdargs:
                    del kwdargs["register_blk"]
                return super().forward(*args, **kwdargs)
            else:
                # Extract x and optional args
                if len(args) > 0:
                    x = args[0]
                    other_args = args[1:]
                else:
                    x = kwdargs.get("x", None)
                    other_args = []

                # Get optional arguments
                # register_blk = kwdargs.get("register_blk", -1)
                prompt = kwdargs.get("prompt", None)
                q = kwdargs.get("q", None)
                train = kwdargs.get("train", False)

                B = x.shape[0]
                x = self.patch_embed(x)

                cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_tokens, x), dim=1)
        
                x = x + self.pos_embed[:,:x.size(1),:]
                x = self.pos_drop(x)

                prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)

                theta = self._pld_info["theta"]  # keep probs per layer
                for i, blk in enumerate(self.blocks):

                    # Block-level keep/drop decision
                    drop_block = False
                    if train:
                        prob = theta[i]
                        keep = torch.rand(1).item() < prob  # single decision per block
                        drop_block = not keep

                    if not train or not drop_block:
                        # Prompt
                        if prompt is not None:
                            if train:
                                p_list, loss, x_tmp = prompt.forward(q, i, x, train=True)
                                prompt_loss += loss
                            else:
                                p_list, _, x_tmp = prompt.forward(q, i, x, train=False)
                        else:
                            p_list = None
                            x_tmp = x

                        x = blk(x_tmp, prompt=p_list)
                    else:
                        # Block dropped, skip blk
                        continue

                x = self.norm(x)

                if prompt is not None:
                    prompt_loss /= len(prompt.e_layers)

                return x, prompt_loss

        
        def get_step(self):
            return self._step
        
        def update_step(self):
            self._step += 1

    return REPVisionTransformer


def apply_patch(
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
    REPVisionTransformer = make_tome_class(model.__class__)

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
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
