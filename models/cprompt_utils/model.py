# This code is a reimplementation based on the CPrompt methodology.
# This code has been modified for continual learning by Wonseon Lim.

import copy
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms

from models.cprompt_utils.vit import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.cprompt_utils.head import SimpleLinear
from models.cprompt_utils.vitprompt import ViT_KPrompts, tensor_prompt


vit_config = {
    'tiny':  {'patch_size':16, 'embed_dim': 192, 'depth': 12, 'num_heads': 3},
    'small': {'patch_size':16, 'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base':  {'patch_size':16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'patch_size':16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
}


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    default_num_classes = pretrained_cfg.num_classes
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_KPrompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        # pretrained_custom_load='npz' in pretrained_cfg.url,
        **kwargs)
    return model


class PromptModel(nn.Module):
    def __init__(self, args, num_classes=10, pretrained=False):
        super(PromptModel, self).__init__()

        self.args = args
        # select prompt method
        self.num_classes = num_classes
        self.clas_w = nn.ModuleList()
        self.ts_prompts_1 = nn.ModuleList()
        self.ts_prompts_2 = nn.ModuleList()
        # select vit type
        vit_type = getattr(args, 'vit_type', 'base')
        if vit_type not in vit_config:
            raise ValueError(f"Unknown ViT type: {vit_type}")

        model_kwargs = vit_config[vit_type]
        self.embed_dim = model_kwargs['embed_dim']

        # get feature encoder
        self.feat = _create_vision_transformer(f'vit_{vit_type}_patch16_224', pretrained=True, **model_kwargs)
        # grad false
        self.feat.requires_grad_(False)
        if getattr(args, 'use_grad_checkpoint', False):
            self.feat.set_grad_checkpointing(True)

        self.task_tokens = copy.deepcopy(self.feat.cls_token) # not used
        self.keys = tensor_prompt(self.num_classes, self.feat.embed_dim, ortho=True)
    
    def update_fc(self, nb_classes, cur_task_nbclasses):
        self.aux_cla = self.generate_fc(self.feat.embed_dim, cur_task_nbclasses)

        cla_w = self.generate_fc(self.feat.embed_dim, cur_task_nbclasses)
        self.clas_w.append(cla_w)
 
        vitprompt_1 = nn.Linear(self.feat.embed_dim, 50, bias=False)
        
        self.ts_prompts_1.append(vitprompt_1)
        vitprompt_2 = nn.Linear(self.feat.embed_dim, 50, bias=False)
        self.ts_prompts_2.append(vitprompt_2)

        if len(self.clas_w)>1:
            self.ts_prompts_1[-1].load_state_dict(self.ts_prompts_1[-2].state_dict())
            self.ts_prompts_2[-1].load_state_dict(self.ts_prompts_2[-2].state_dict())
        
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc
        
    def aux_forward(self, image):
        i = len(self.clas_w)-1
        image_features = self.feat(image, instance_tokens=self.ts_prompts_1[i].weight,second_pro=self.ts_prompts_2[i].weight, returnbeforepool=True, )
        feature = image_features[:,0,:]
        logits = self.aux_cla(feature)['logits']
        return logits, feature

    def forward(self, image, gen_p, train):
        image_features = self.feat(image, instance_tokens=gen_p[0], second_pro=gen_p[1], returnbeforepool=True)
        feature = image_features[:,0,:]
        if train:
            i=len(self.clas_w)-1
            return self.clas_w[i](feature)['logits']
        for i in range(len(self.clas_w)):
            if i==0:
                logits=self.clas_w[i](feature)['logits']
            else:
                logit=self.clas_w[i](feature)['logits']
                logits=torch.cat((logits, logit),1)
        return logits
    
    def fix_branch_layer(self):
        for param in self.clas_w.parameters():
            param.requires_grad=False
            param.grad=None

        for param in self.ts_prompts_1.parameters():
            param.requires_grad=False
            param.grad=None
            
        for param in self.ts_prompts_2.parameters():
            param.requires_grad=False
            param.grad=None



    # def forward(self, x, y=None, q=None, train=False, last=False, feat=False, **kwargs):
    #     if last:
    #         return self.head(x)

    #     if self.args.query == 'vit':
    #         x_backbone = x
    #         x_query = x.clone()
    #     elif self.args.query in ['poolformer', 'swin']: # but not used and data is already normalized
    #         x_backbone = transforms.Normalize(self.dset_mean, self.dset_std)(x)
    #         x_query = transforms.Normalize(self.dset_mean_q, self.dset_std_q)(x)
    #         raise NotImplementedError(f"Input data is already normalized")
    #     elif self.args.query == 'None':
    #         x_backbone = x
    #         x_query = x.clone()

    #     if self.prompt is not None:
    #         with torch.no_grad():
    #             if self.args.query == 'vit':
    #                 q, _ = self.feat_query(x_query)
    #                 q = q[:,0,:]
    #             elif self.args.query in ['poolformer', 'swin']:
    #                 q = self.feat_query(x_query)
    #                 q = q[-1].mean(-2).mean(-1)
    #             elif self.args.query == 'None':
    #                 q = None
    #             else:
    #                 q = self.feat_query(x_query)
    #         # Forward with prompt
    #         out, prompt_loss = self.feat(x_backbone, prompt=self.prompt, q=q, train=train)
    #         out = out[:, 0, :]
    #     else:
    #         out, _ = self.feat(x)
    #         out = out[:, 0, :]
    #     out = out.view(out.size(0), -1)

    #     if feat:
    #         return out
    #     out = self.head(out)
    #     if self.prompt is not None and train:
    #         return out, prompt_loss
    #     else:
    #         return out
