# This code is a reimplementation based on the OVOR methodology.
# This code has been modified for on-device continual learning by Wonseon Lim.

import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms

from models.sparsevit_prompt_utils.vit import VisionTransformer
from models.sparsevit_prompt_utils.prompt import L2P, DualPrompt, CodaPrompt, MVPPrompt, OnePrompt, OSPrompt


vit_config = {
    'tiny':  {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
    'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base':  {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
}


class PromptModel(nn.Module):
    def __init__(self, args, num_classes=10, pretrained=False, prompt_flag=False, prompt_param=None):
        super(PromptModel, self).__init__()

        self.args = args
        # select prompt method
        self.num_classes = num_classes
        self.prompt_flag = prompt_flag
        # select vit type
        vit_type = getattr(args, 'vit_type', 'base')
        if vit_type not in vit_config:
            raise ValueError(f"Unknown ViT type: {vit_type}")

        cfg = vit_config[vit_type]
        self.embed_dim = cfg['embed_dim']

        # get feature encoder
        if pretrained:
            self.feat = VisionTransformer(img_size=224, patch_size=16,
                                        embed_dim=cfg['embed_dim'],
                                        depth=cfg['depth'],
                                        num_heads=cfg['num_heads'],
                                        ckpt_layer=0, drop_path_rate=0)

            pretrained_model = timm.create_model(f'vit_{vit_type}_patch16_224', pretrained=True)
            load_dict = pretrained_model.state_dict()
            if 'head.weight' in load_dict:
                del load_dict['head.weight']
                del load_dict['head.bias']
            missing, unexpected = self.feat.load_state_dict(load_dict, strict=False)
            assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
            # grad false
            self.feat.requires_grad_(False)

        # Sparse Prompt
        self.feat.query_merge = args.query_merge

        # classifier
        self.head = nn.Linear(self.embed_dim, num_classes)

        # create prompting module
        if self.prompt_flag == 'sparse':
            self.prompt = CodaPrompt(args, self.embed_dim, prompt_param, self.embed_dim) # prompt_param: 100 8 0.0
        else:
            self.prompt = None

    def forward(self, x, y=None, q=None, train=False, last=False, warmup=False, feat=False, **kwargs):
        if last:
            return self.head(x)

        if self.prompt is not None:
            with torch.no_grad():
                if q is None:
                    layer_idx = len(self.feat.blocks) - 1
                    q, _ = self.feat(x, register_blk=layer_idx)
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, drop_rate=self.args.drop_rate, sparse_type=self.args.sparse_type)
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

        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out