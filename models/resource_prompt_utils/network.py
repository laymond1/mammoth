"""Reuse CODA pool semantics and the pinned canonical150 encoder loader."""

from models.coda_prompt_utils.model import CodaPrompt
from models.coda_prompt_utils.vit import VisionTransformer
from models.low_rank_utils.factory import SIZES, checkpoint_spec
from models.low_rank_utils.pretrained import (
    canonical_pretrained_digest,
    load_pretrained_encoder,
)


class LayeredCodaPrompt(CodaPrompt):
    def __init__(self, dimension, n_tasks, prompt_param, layers=(0, 1, 2, 3, 4)):
        object.__setattr__(self, '_requested_layers', tuple(layers))
        super().__init__(dimension, n_tasks, prompt_param, key_dim=dimension)

    def _init_smart(self, dimension, prompt_param):
        super()._init_smart(dimension, prompt_param)
        self.e_layers = list(self._requested_layers)


def build_encoder(vit_size='tiny', checkpoint=None, *, encoder_class=VisionTransformer,
                  pretrained=True, geometry=None):
    if vit_size not in SIZES:
        raise ValueError('Resource prompt vit_size must be tiny, small or base')
    if geometry is not None and pretrained:
        raise ValueError('Synthetic geometry cannot load a production checkpoint')
    if checkpoint is not None and not pretrained:
        raise ValueError('A checkpoint requires pretrained=True')
    # Fail before encoder allocation and initialization RNG consumption.
    spec = checkpoint_spec(vit_size, checkpoint) if pretrained else None
    dimension, heads = SIZES[vit_size]
    options = {'img_size': 224, 'patch_size': 16, 'embed_dim': dimension, 'depth': 12,
                   'num_heads': heads, 'drop_path_rate': 0}
    options.update(geometry or {})
    encoder = encoder_class(**options)
    if spec is not None:
        load_pretrained_encoder(encoder, spec['file'])
        actual = canonical_pretrained_digest(encoder)
        if actual != spec['canonical_sha256']:
            raise ValueError('Resource prompt encoder differs from the pinned canonical150 tensors')
        encoder.pretrained_checkpoint = spec['file']
        encoder.pretrained_sha256 = spec['sha256']
        encoder.canonical_pretrained_sha256 = actual
    encoder.requires_grad_(False)
    encoder.vit_size = vit_size
    return encoder
