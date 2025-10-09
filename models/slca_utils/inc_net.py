import copy
import os
import sys
import torch
import timm
from torch import nn
import torch.nn.functional as F
from backbone.ResNetBlock import resnet18, resnet34
from backbone.ResNetBottleneck import resnet50
from backbone.vit import vit_base_patch16_224_prompt_prototype
from models.slca_utils.convs.cifar_resnet import resnet32
from backbone.vit import VisionTransformer
from models.slca_utils.convs.linears import SimpleContinualLinear


vit_config = {
    'tiny':  {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
    'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base':  {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
}


def get_convnet(feature_extractor_type, pretrained=False):
    name = feature_extractor_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet18_cifar':
        return resnet18(pretrained=pretrained, cifar=True)
    elif name == 'resnet18_cifar_cos':
        return resnet18(pretrained=pretrained, cifar=True, no_last_relu=True)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name in ['tiny', 'small', 'base']:
        if name not in vit_config:
            raise ValueError(f"Unknown ViT type: {name}")
        return _create_vit_model(name, vit_config[name])
    elif name == 'vit-b-p16':
        print("Using ViT-B/16 pretrained on ImageNet21k (NO FINETUNE ON IN1K)")
        model = vit_base_patch16_224_prompt_prototype(pretrained=pretrained, pretrain_type='in21k', num_classes=0)
        model.norm = nn.LayerNorm(model.embed_dim)  # from the original implementation
        return model
    elif name == 'vit-b-p16-mocov3':
        model = vit_base_patch16_224_prompt_prototype(pretrained=pretrained, pretrain_type='in21k', num_classes=0)

        del model.head
        if not os.path.exists('mocov3-vit-base-300ep.pth'):
            print("Cannot find the pretrained model for MoCoV3-ViT-B/16")
            print("Please download the model from https://drive.google.com/file/d/1bshDu4jEKztZZvwpTVXSAuCsDoXwCkfy/view?usp=share_link")
            sys.exit(1)

        ckpt = torch.load('mocov3-vit-base-300ep.pth', map_location='cpu', weights_only=True)['model']  # from the original implementation
        state_dict = model.state_dict()
        state_dict.update(ckpt)
        model.load_state_dict(state_dict)
        del model.norm
        model.norm = nn.LayerNorm(model.embed_dim)
        return model
    else:
        raise NotImplementedError('Unknown type {}'.format(feature_extractor_type))


def _create_vit_model(name, config):
    """Helper function to create ViT model with pretrained weights."""
    model = VisionTransformer(
        img_size=224, 
        patch_size=16,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        drop_path_rate=0
    )
    
    pretrained_model = timm.create_model(f'vit_{name}_patch16_224', pretrained=True)
    load_dict = pretrained_model.state_dict()
    
    # Remove head weights if they exist
    if 'head.weight' in load_dict:
        del load_dict['head.weight']
        del load_dict['head.bias']
    
    missing, unexpected = model.load_state_dict(load_dict, strict=False)
    assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    
    return model


class BaseNet(nn.Module):

    def __init__(self, feature_extractor_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(feature_extractor_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x, returnt='features')

    def forward(self, x):
        x = self.convnet(x, returnt='features')
        out = self.fc(x)
        '''
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        '''
        out.update({'features': x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class FinetuneIncrementalNet(BaseNet):

    def __init__(self, feature_extractor_type, pretrained, fc_with_ln=False):
        super().__init__(feature_extractor_type, pretrained)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln

    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.convnet.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleContinualLinear(in_dim, out_dim)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.convnet(x, returnt='features')
        else:
            x = self.convnet(x, returnt='features')
        out = self.fc(x)
        out.update({'features': x})

        return out
