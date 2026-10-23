"""Pinned AugReg ViT-Ti/S/B weights (ImageNet-21k pretraining, ImageNet-1k fine-tuning at 224 px).

``convert`` runs once on the server, where timm 0.9.8 reads the original ``.npz``; it writes a
plain ``.npz`` of timm-named float32 arrays that every device loads with numpy alone. The
source files and the encoder digest are those pinned in ``models/low_rank_utils/factory.py``
of ``codex/cps-rep``, so the converted encoder is checked against the repository's own pin.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SOURCE_DIR = Path('/mnt/sdab1/userHome/wslim/.cache/torch/hub/checkpoints')
CACHE = Path('/mnt/sdab1/userHome/wslim/.cache/token_efficiency/weights')
PINNED = {
    'tiny': ('Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
             '0c5c4818ecf1049ae34194223c908172a2967334bab6d86670351b59d608b5dd',
             '3258efdf29eb69379b559f8884b31134e5f299a2e6cd682f08eb9483e9c8e1cb'),
    'small': ('S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
              '545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3',
              '1bc2129fcd9c44c0104385f3db58a422595d205fcfff592392124b5f2bf7b1a5'),
    'base': ('B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
             'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124',
             'cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300'),
}
# AugReg normalization (Inception style).
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_digest(state):
    """Digest convention of ``canonical_pretrained_digest``: six stem tensors and twelve per block, no head."""
    names = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'norm.weight', 'norm.bias']
    blocks = sorted({int(name.split('.')[1]) for name in state if name.startswith('blocks.')})
    for index in blocks:
        names.extend(f'blocks.{index}.{suffix}' for suffix in (
            'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'attn.qkv.weight', 'attn.qkv.bias',
            'attn.proj.weight', 'attn.proj.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'))
    digest = hashlib.sha256()
    for name in sorted(names):
        tensor = torch.as_tensor(state[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def path(size, root=CACHE):
    return Path(root) / f'augreg-{size}.npz'


def convert(size, out=CACHE):
    """Read the pinned original with timm and write the plain array file and its manifest."""
    from timm.models.vision_transformer import VisionTransformer as TimmViT, _load_weights

    from .vit import SIZES

    name, source_sha256, canonical = PINNED[size]
    source = SOURCE_DIR / name
    if file_sha256(source) != source_sha256:
        raise ValueError(f'{source} differs from the pinned file')
    width, heads = SIZES[size]
    model = TimmViT(embed_dim=width, depth=12, num_heads=heads, num_classes=1000)
    _load_weights(model, str(source))
    state = {key: value.detach().float().contiguous() for key, value in model.state_dict().items()}
    if encoder_digest(state) != canonical:
        raise ValueError('Converted encoder does not match the repository pin')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = path(size, out)
    np.savez(target, **{key: value.numpy() for key, value in state.items()})
    manifest = {'size': size, 'source': name, 'source_sha256': source_sha256, 'encoder_digest': canonical,
                'file_sha256': file_sha256(target), 'tensors': len(state),
                'parameters': int(sum(value.numel() for value in state.values()))}
    target.with_suffix('.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def load(size, root=CACHE, verify=True):
    """Return the state dictionary of ``size`` as numpy arrays, checking the file against its manifest."""
    target = path(size, root)
    if verify:
        manifest = json.loads(target.with_suffix('.json').read_text())
        if file_sha256(target) != manifest['file_sha256']:
            raise ValueError(f'{target} does not match its manifest')
    with np.load(target) as arrays:
        return {name: arrays[name] for name in arrays.files}
