import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms

from models.prompt_utils.vit import VisionTransformer


vit_config = {
    'tiny':  {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
    'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'base':  {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
}


class ViT(nn.Module):
    def __init__(self, args, num_classes=10, pretrained=False):
        super(ViT, self).__init__()

        self.args = args
        # select prompt method
        self.num_classes = num_classes
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
            # # grad true
            # self.feat.requires_grad_(true)

        # classifier
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, feat=False):
        """
        Forward pass through the ViT model.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Output tensor after passing through the feature extractor and classifier.
        """
        out, _ = self.feat(x)
        out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        
        if feat:
            return out

        out = self.head(out)
        
        return out