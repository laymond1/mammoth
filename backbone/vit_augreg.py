"""Full fine-tuning ViTs with explicit, locally pinned AugReg weights.

This backbone is separate from prompt/adapter networks: all encoder and head
parameters remain trainable. Existing ``vit`` defaults are unchanged.
"""

from backbone import register_backbone
from backbone.vit import VisionTransformer


@register_backbone('vit-augreg')
def vit_augreg(num_classes: int, vit_size: str = 'tiny',
               pretrained: int = 1, vit_checkpoint: str = ''):
    """Build Tiny/Small/Base with the Mammoth forward and gradient contracts.

    Pretrained mode verifies the shared NPZ file and all canonical encoder
    tensors. The dataset classifier is initialized independently and no model
    or dataset download is performed.
    """
    from models.low_rank_utils.factory import SIZES, checkpoint_spec
    from models.low_rank_utils.pretrained import canonical_pretrained_digest, load_pretrained_encoder

    if vit_size not in SIZES:
        raise ValueError('vit_size must be tiny, small or base')
    if pretrained not in (0, 1):
        raise ValueError('pretrained must be 0 or 1')
    if num_classes <= 0:
        raise ValueError('The full fine-tuning backbone requires a positive num_classes')
    if not pretrained and vit_checkpoint:
        raise ValueError('vit_checkpoint requires pretrained=True')
    # Check the complete file before allocating a network.
    spec = checkpoint_spec(vit_size, vit_checkpoint or None) if pretrained else None
    dimension, heads = SIZES[vit_size]
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=dimension,
                              depth=12, num_heads=heads, num_classes=num_classes)
    if spec is not None:
        load_pretrained_encoder(model, spec['file'])
        actual = canonical_pretrained_digest(model)
        if actual != spec['canonical_sha256']:
            raise ValueError('Full fine-tuning encoder differs from pinned AugReg tensors')
        model.pretrained_checkpoint = spec['file']
        model.pretrained_sha256 = spec['sha256']
        model.canonical_pretrained_sha256 = actual
    model.pretrained_model = f'vit_{vit_size}_patch16_224.augreg_in21k_ft_in1k'
    model.vit_size = vit_size
    return model
