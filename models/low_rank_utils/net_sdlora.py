# Ported from CL-LoRA d007ef5c13a6d8a7d7db3eedf93f5e796e2ce429.
import torch
import torch.nn as nn

from .vit_sdlora import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from .pretrained import load_pretrained_encoder
from .output import backbone_output


def _create_vision_transformer(variant, pretrained=False, checkpoint=None, **kwargs):
    """Construct locally, retaining source initialization before loading weights."""
    if kwargs.pop('features_only', False):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    if pretrained and checkpoint is None:
        raise FileNotFoundError('LoRA pretrained loading requires an explicit local NPZ checkpoint')
    kwargs.setdefault('num_classes', 1000)
    model = ViT(**kwargs)
    if pretrained:
        load_pretrained_encoder(model, checkpoint)
    return model


class ViT(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)

    def forward(self, x, task_id, register_blk=-1):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i)

        x = self.norm(x)

        return x


class Net(nn.Module):
    def __init__(self, args, *, geometry=None, checkpoint=None):
        super(Net, self).__init__()

        geometry = geometry or dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        model_kwargs = dict(patch_size=16, embed_dim=768,
                            depth=12, num_heads=12,
                            n_tasks=args["sessions"], rank=args["rank"])

        model_kwargs.update(geometry)
        self.image_encoder = _create_vision_transformer(args.get("load"), pretrained=checkpoint is not None, checkpoint=checkpoint, **model_kwargs)
        self.class_num = args["init_cls"]

        # Linear classifier for each task
        self.classifier_pool = nn.ModuleList([
            nn.Linear(self.image_encoder.out_dim, self.class_num, bias=True)
            for i in range(args["sessions"])
        ])

        for module in self.image_encoder.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self._cur_task = -1

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features = self.image_encoder(image, self._cur_task)
        else:
            image_features = self.image_encoder(image, task)

        image_features = image_features[:,0,:]  # [128,768]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, fc_only=False, *, returnt=None):
        if fc_only:
            fc_outs = []
            for ti in range(self._cur_task + 1):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            result = {'logits': torch.cat(fc_outs, dim=1), 'features': image}
            return result['logits'] if returnt is None else backbone_output(result, returnt)

        logits = []
        image_features = self.image_encoder(image, task_id=self._cur_task)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        for classifier in [self.classifier_pool[self._cur_task]]:
            logits.append(classifier(image_features))

        return backbone_output({
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }, returnt)

    def interface(self, image, task_id=None):
        logits = []
        image_features = self.image_encoder(image, task_id=self._cur_task if task_id is None else task_id)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        for classifier in self.classifier_pool[: self._cur_task+1]:
            logits.append(classifier(image_features))

        logits = torch.cat(logits, dim=1)
        return logits

    def update_fc(self, nb_classes):
        self._cur_task +=1
