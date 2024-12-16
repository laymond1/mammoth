# This code is a reimplementation based on the OVOR methodology.
# This code has been modified for online continual learning by Wonseon Lim.

import torch
import torch.nn as nn

from models.prompt_utils.vit import VisionTransformer
from models.prompt_utils.prompt import L2P, DualPrompt, CodaPrompt, MVPPrompt, OnePrompt

class PromptModel(nn.Module):
    def __init__(self, args, num_classes=10, pretrained=False, prompt_flag=False, prompt_param=None):
        super(PromptModel, self).__init__()

        self.args = args
        # select prompt method
        self.num_classes = num_classes
        self.prompt_flag = prompt_flag

        # get feature encoder
        if pretrained:
            self.feat = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models.vision_transformer import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            if 'head.weight' in load_dict:
                del load_dict['head.weight']
                del load_dict['head.bias']
            missing, unexpected = self.feat.load_state_dict(load_dict, strict=False)
            assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
            # grad false
            self.feat.requires_grad_(False)

        # classifier
        self.head = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(args, 768, prompt_param) # prompt_param: 30 20 -1
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(args, 768, prompt_param) # prompt_param: 10 40 10
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(args, 768, prompt_param) # prompt_param: 100 8 0.0
        elif self.prompt_flag == 'mvp':
            self.prompt = MVPPrompt(args, 768, prompt_param) # prompt_param: 10 40 10
        elif self.prompt_flag == 'oneprompt':
            self.prompt = OnePrompt(args, 768, prompt_param)
        else:
            self.prompt = None

    def forward_features(self, x, train=False, last=False, warmup=False,  **kwargs):
        if self.prompt is not None:
            if self.prompt_flag  == 'oneprompt':
                out, prompt_loss = self.feat(
                    x, prompt=self.prompt,
                    q=torch.zeros(x.size(0), 768, device=x.device), train=train
                )
            elif self.prompt_flag == 'mvp':
                with torch.no_grad():
                    q, _ = self.feat(x)
                    q = q[:, 0, :]
                out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train)
                top_k = extract_topk_key(q, self.prompt, top_k=self.args.top_k)
                mask = self.prompt.mask[top_k].mean(1).squeeze().clone()
                mask = torch.sigmoid(mask)*2.
            else:
                with torch.no_grad():
                    q, _ = self.feat(x)
                    q = q[:, 0, :]
                out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
            if warmup:
                prompt_loss = torch.zeros_like(prompt_loss)
                out = out.detach()
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        feature = out.view(out.size(0), -1)

        if hasattr(self.prompt, 'use_mask') and self.prompt.use_mask:
            return feature, prompt_loss, mask
        else:
            return feature
        
    def forward_head(self, feature):
        return self.head(feature)

    def forward(self, x, train=False, last=False, warmup=False, feat=False, **kwargs):
        if last:
            return self.head(x)

        if self.prompt is not None:
            if self.prompt_flag  == 'oneprompt':
                out, prompt_loss = self.feat(
                    x, prompt=self.prompt,
                    q=torch.zeros(x.size(0), 768, device=x.device), train=train
                )
            elif self.prompt_flag == 'mvp':
                with torch.no_grad():
                    q, _ = self.feat(x)
                    q = q[:, 0, :]
                out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train)
                top_k = extract_topk_key(q, self.prompt, top_k=self.args.top_k)
                mask = self.prompt.mask[top_k].mean(1).squeeze().clone()
                mask = torch.sigmoid(mask)*2.
            else:
                with torch.no_grad():
                    q, _ = self.feat(x)
                    q = q[:, 0, :]
                out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
            if warmup:
                prompt_loss = torch.zeros_like(prompt_loss)
                out = out.detach()
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)

        if feat:
            return out
        out = self.head(out)
        if hasattr(self.prompt, 'use_mask') and self.prompt.use_mask:
            out = out * mask
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
        
def extract_topk_key(query, prompt, top_k=1):
    K = prompt.e_k
    # cosine similarity to match keys/querries
    n_K = nn.functional.normalize(K, dim=1)
    q = nn.functional.normalize(query, dim=1).detach()
    cos_sim = torch.einsum('bj,kj->bk', q, n_K)

    # top-k 
    distance = 1 - cos_sim
    _, top_k = torch.topk(distance, top_k, dim=1, largest=False)
    return top_k