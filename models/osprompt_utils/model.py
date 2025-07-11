# This code is a reimplementation based on the OSPrompt methodology.
# This code has been modified for continual learning by Wonseon Lim.

import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms

from models.osprompt_utils.vit import VisionTransformer
from models.prompt_utils.prompt import OSPrompt, OSPromptPP


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
            # load query model
            if self.args.query == 'vit':
                zoo_model_query = VisionTransformer(img_size=224, patch_size=16,
                                                    embed_dim=cfg['embed_dim'],
                                                    depth=cfg['depth'],
                                                    num_heads=cfg['num_heads'],
                                                    ckpt_layer=0, drop_path_rate=0)
                load_dict = timm.create_model(f'vit_{vit_type}_patch16_224', pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model_query.load_state_dict(load_dict)
            elif self.args.query == 'poolformer':
                print( "Load poolformer fine-tuned on in1k ...")
                zoo_model_query = timm.create_model('poolformerv2_m36.sail_in1k', pretrained=True, features_only=True)
            elif self.args.query == 'None':
                zoo_model_query = None
            else:
                NotImplementedError
            self.feat_query = zoo_model_query
            if self.feat_query is not None:
                self.feat_query.requires_grad_(False)

            # load prompt model
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

        # classifier
        self.head = nn.Linear(self.embed_dim, num_classes)

        # create prompting module
        if self.prompt_flag == 'os':
            self.prompt = OSPrompt(args, self.embed_dim, prompt_param, self.embed_dim) # prompt_param: 100 8 1e-4
        elif self.prompt_flag == 'ospp':
            self.prompt = OSPromptPP(args, self.embed_dim, prompt_param, self.embed_dim) # prompt_param: 100 8 1e-4
        else:
            self.prompt = None

        # this is for OS-Prompt
        self.dset_mean = (0.0, 0.0, 0.0)
        self.dset_std = (1.0, 1.0, 1.0)

        if self.args.query == 'vit':
            self.dset_mean_q = (0.0,0.0,0.0)
            self.dset_std_q = (1.0,1.0,1.0)
        elif self.args.query == 'None':
            self.dset_mean_q = (0.0,0.0,0.0)
            self.dset_std_q = (1.0,1.0,1.0)
        else:
            self.dset_mean_q  = timm.data.resolve_model_data_config(zoo_model_query)['mean']
            self.dset_std_q  = timm.data.resolve_model_data_config(zoo_model_query)['std']
            raise NotImplementedError(f"Input data is already normalized")

        print ('norm for query: {} /{}'.format(self.dset_mean_q, self.dset_std_q ))
        
    def forward(self, x, y=None, q=None, train=False, last=False, feat=False, **kwargs):
        if last:
            return self.head(x)

        if self.args.query == 'vit':
            x_backbone = x
            x_query = x.clone()
        elif self.args.query in ['poolformer', 'swin']: # but not used and data is already normalized
            x_backbone = transforms.Normalize(self.dset_mean, self.dset_std)(x)
            x_query = transforms.Normalize(self.dset_mean_q, self.dset_std_q)(x)
            raise NotImplementedError(f"Input data is already normalized")
        elif self.args.query == 'None':
            x_backbone = x
            x_query = x.clone()

        if self.prompt is not None:
            with torch.no_grad():
                if self.args.query == 'vit':
                    q, _ = self.feat_query(x_query)
                    q = q[:,0,:]
                elif self.args.query in ['poolformer', 'swin']:
                    q = self.feat_query(x_query)
                    q = q[-1].mean(-2).mean(-1)
                elif self.args.query == 'None':
                    q = None
                else:
                    q = self.feat_query(x_query)
            # Forward with prompt
            out, prompt_loss = self.feat(x_backbone, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
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
