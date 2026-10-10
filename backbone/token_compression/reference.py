"""ImageNet-1K reference classifiers for training and checking compensator banks.

The encoder is ``backbone.vit`` with the same pinned tensors the CL methods
load. Unlike those methods, the pretrained 1000-class head is kept, because a
bank is distilled on ImageNet logits.
"""

import numpy as np
import torch

from backbone.vit import VisionTransformer

NORMALIZATION = {'augreg': ((.5, .5, .5), (.5, .5, .5)),
                 'deit': ((.485, .456, .406), (.229, .224, .225))}


def _finish(model, weight, bias, family):
    from models.low_rank_utils.pretrained import canonical_pretrained_digest
    head = model.head.heads[0]
    if weight.shape != head.weight.shape or bias.shape != head.bias.shape:
        raise ValueError('Pretrained ImageNet head does not fit the reference classifier')
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    model.canonical_pretrained_sha256 = canonical_pretrained_digest(model)
    model.normalization = NORMALIZATION[family]
    return model.eval()


def augreg_reference(vit_size='tiny', checkpoint=None, drop_path_rate=0.):
    """Pinned AugReg IN21k->IN1k weights, the ones behind ``--backbone vit-augreg``."""
    from models.low_rank_utils.factory import SIZES, checkpoint_spec
    from models.low_rank_utils.pretrained import load_pretrained_encoder
    spec = checkpoint_spec(vit_size, checkpoint)
    dimension, heads = SIZES[vit_size]
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=dimension, depth=12, num_heads=heads,
                              num_classes=1000, drop_path_rate=drop_path_rate)
    load_pretrained_encoder(model, spec['file'])
    with np.load(spec['file'], allow_pickle=False) as archive:
        prefix = 'opt/target/' if 'opt/target/embedding/kernel' in archive else ''
        weight = torch.from_numpy(np.array(archive[prefix + 'head/kernel'], copy=True)).T
        bias = torch.from_numpy(np.array(archive[prefix + 'head/bias'], copy=True))
    model = _finish(model, weight, bias, 'augreg')
    if model.canonical_pretrained_sha256 != spec['canonical_sha256']:
        raise ValueError('Reference encoder differs from pinned AugReg tensors')
    return model


def deit_reference(checkpoint, drop_path_rate=0.):
    """Official DeiT ``.pth`` weights; kept to re-check banks trained before the plug-in."""
    state = dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    weight, bias = state.pop('head.weight'), state.pop('head.bias')
    dimension = state['cls_token'].shape[-1]
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=dimension, depth=12, num_heads=dimension // 64,
                              num_classes=1000, drop_path_rate=drop_path_rate)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {'head.heads.0.weight', 'head.heads.0.bias'}:
        raise ValueError(f'Incompatible DeiT checkpoint: missing={missing}, unexpected={unexpected}')
    return _finish(model, weight, bias, 'deit')
