"""One forward per implementation of the same token-reduction operator.

study     ``scripts.token_efficiency``: own ViT, the code behind the five-device 1.8-4.1x figures
plugin    ``backbone.token_compression.attach`` on ``backbone.vit`` (the CL plug-in)
official  facebookresearch/ToMe (``tome.patch.timm`` on timm's ViT); needs ``tome`` on ``PYTHONPATH``

Every arm loads the same pinned AugReg tensors and returns ``(forward, info)`` like
``token_efficiency.methods.build_forward``. ``dense`` is the unpatched model of that arm.
``official`` with degree 0 and ``patched=True`` is the patched model at r=0, whose attention is
the manual softmax path and not the fused one.
"""
import os

import torch

from scripts.token_efficiency import methods, weights
from scripts.token_efficiency.vit import SIZES, build as build_study

ARMS = ('study', 'plugin', 'official')
DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def _gemm_patch_embed(embed):
    """Run a timm ``PatchEmbed`` as one matrix product (the study's arithmetic), only when asked for."""
    weight, bias = embed.proj.weight, embed.proj.bias

    def forward(images):
        batch, _, height, width = images.shape
        rows, cols = height // 16, width // 16
        patches = images.reshape(batch, 3, rows, 16, cols, 16).permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * cols, 768)
        return torch.nn.functional.linear(patches, weight.reshape(weight.shape[0], 768), bias)
    embed.forward = forward


def _finish(model, device, dtype):
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if os.environ.get('TOME_LATENCY_GEMM_PATCH') == '1':
        _gemm_patch_embed(model.patch_embed)
    return model


def _plugin_model(size, state):
    from backbone.vit import VisionTransformer
    width, heads = SIZES[size]
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=width, depth=12, num_heads=heads, num_classes=1000)
    mapped = {name.replace('head.', 'head.heads.0.', 1) if name.startswith('head.') else name: torch.as_tensor(value)
              for name, value in state.items()}
    result = model.load_state_dict(mapped, strict=False)
    if result.unexpected_keys or result.missing_keys:
        raise ValueError(f'Mammoth ViT does not take the pinned tensors: {result}')
    return model


def _official_model(size, state):
    from timm.models.vision_transformer import VisionTransformer
    width, heads = SIZES[size]
    model = VisionTransformer(embed_dim=width, depth=12, num_heads=heads, num_classes=1000)
    model.load_state_dict({name: torch.as_tensor(value) for name, value in state.items()}, strict=True)
    return model


class _Trace:
    """Placeholder for arms that keep no token trace."""

    def __init__(self):
        self.trace = []


def build(arm, size, dtype, device, operator, degree, weights_root, patched=False):
    """Return ``(forward, info)``; ``operator`` is ``dense`` or one of the four operator names."""
    state = weights.load(size, weights_root)
    dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
    device = torch.device(device)
    if arm == 'study':
        model = build_study(size, state, device, dtype)
        spec = {'kind': 'dense'} if operator == 'dense' else {'kind': 'inblock', 'operator': operator, 'degree': degree}
        return methods.build_forward(model, spec)
    if arm == 'plugin':
        model = _finish(_plugin_model(size, state), device, dtype)
        info = {'capturable': True}
        if operator != 'dense':
            from backbone.token_compression import attach
            info['state'] = attach(model, operator, degree=degree)
        return (lambda images: model(images)), info
    if arm == 'official':
        model = _finish(_official_model(size, state), device, dtype)
        info = {'capturable': True}
        if operator != 'dense' or patched:
            if operator not in ('tome', 'dense'):
                raise ValueError('The official package has ToMe only')
            import tome
            tome.patch.timm(model)
            model.r = degree
            info['state'] = _Trace()
        return (lambda images: model(images)), info
    raise ValueError(f'Unknown arm {arm}')
