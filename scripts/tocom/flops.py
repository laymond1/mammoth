"""Forward FLOPs of token compression on ViT-Tiny, and the CPS-Prompt and REP settings that equal ToMe.

CPS-Prompt and REP compress tokens with their own algorithm in training. To set
them next to models trained with the plug-in, their setting is chosen so that
the FLOPs of the prompted pass equal those of the same pass under ToMe at a
training degree. This script measures that on the real modules (random weights,
one 224x224 image, CPU) and prints the closest settings; ``run_cl.OWN`` holds them.

FLOPs are multiply-accumulates of one forward pass, attention products and the
matching products of ToMe included (the fvcore convention). torch's counter
reports two per multiply-add and skips fused attention, hence the explicit
attention and the halving below.

    uv run --no-sync python -m scripts.tocom.flops
"""

from scripts.cl_runtime import configure_native_environment

configure_native_environment()

from contextlib import contextmanager  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from backbone.token_compression import attach, detach  # noqa: E402

PATCHES = 196


def _explicit_attention(query, key, value, attn_mask=None, dropout_p=0., is_causal=False, scale=None):
    weights = (query @ key.transpose(-2, -1)) * (query.shape[-1] ** -.5 if scale is None else scale)
    if attn_mask is not None:
        weights = weights + attn_mask
    return weights.softmax(dim=-1) @ value


@contextmanager
def _counted():
    fused = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _explicit_attention
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            yield counter
    finally:
        F.scaled_dot_product_attention = fused


def flops(call):
    with _counted() as counter:
        call()
    return counter.get_total_flops() / 2


def image():
    return torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))


def plain_reduction(degrees, operator='tome'):
    """FLOPs reduction of one plain ViT-Tiny pass at each degree."""
    from backbone.vit import VisionTransformer
    net = VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=100).eval()
    x, controller = image(), attach(net, operator)
    base = flops(lambda: net(x))
    result = {}
    for degree in degrees:
        with controller.degree(degree):
            result[degree] = 1 - flops(lambda: net(x)) / base
    return result


def cps_reductions(degrees, keeps):
    """Prompted pass of CPS-Prompt: under ToMe at ``degrees``, and with ``keeps`` sampled patches."""
    from models.cps_prompt_utils.model import PromptModel
    net = PromptModel(num_classes=100, n_tasks=10, vit_size='tiny', pretrained=False).eval()
    x = image()
    with torch.no_grad():
        query, _, scores, _ = net.feat(x)

    def prompted(train):
        return flops(lambda: net.feat(x, prompt=net.prompt, q=query[:, 0], train=train, sampler=net.sampler, scores=scores))

    base = prompted(False)
    controller = attach(net, 'tome')
    targets = {}
    for degree in degrees:
        with controller.degree(degree):
            targets[degree] = 1 - prompted(False) / base
    detach(net)
    net.train()
    sampled = {}
    for keep in keeps:
        # Any ratio in this interval keeps exactly ``keep`` patches; its middle is robust to rounding.
        net.sampler.reduction_ratio = round(1 - (keep + .5) / PATCHES, 3)
        if int(PATCHES * (1 - net.sampler.reduction_ratio)) != keep:
            raise ValueError(f'No three-decimal ratio keeps {keep} patches')
        sampled[keep] = (net.sampler.reduction_ratio, 1 - prompted(True) / base)
    return targets, sampled


def rep_reductions(degrees, maxima):
    """Prompted pass of REP: under ToMe at ``degrees``, and with its own merging up to ``maxima``."""
    from models.rep_prompt_utils.model import REPModel
    net = REPModel(num_classes=100, n_tasks=10, vit_size='tiny', pretrained=False, atom=True, ald=False).eval()
    x = image()
    with torch.no_grad():
        query = net.query_encoder(x)[0][:, 0] @ net.projection

    def prompted(atom, maximum=0):
        return flops(lambda: net.feat(x, net.prompt, query, train=False, atom=atom, ald=False, maximum=maximum))

    base = prompted(False)
    own = {maximum: 1 - prompted(True, maximum) / base for maximum in maxima}
    controller = attach(net, 'tome')
    targets = {}
    for degree in degrees:
        with controller.degree(degree):
            targets[degree] = 1 - prompted(False) / base
    return targets, own


def closest(targets, candidates):
    return {degree: min(candidates, key=lambda key: abs(candidates[key] - target)) for degree, target in targets.items()}


def main():
    # Degree 24 needs 71 patches for CPS-Prompt and a maximum of 187 for REP: REP's merges per block are
    # clamped to half of the tokens, so the maximum has to be far above the per-block counts that happen.
    degrees = (8, 16, 24)
    report = {'plain': {operator: plain_reduction((0, 4, 8, 10, 13, 16, 24), operator) for operator in ('tome', 'evit')}}
    targets, sampled = cps_reductions(degrees, range(40, PATCHES + 1))
    report['cps-prompt'] = {degree: {'target': targets[degree], 'kept_patches': keep, 'reduction_ratio': sampled[keep][0],
                                     'reduction': sampled[keep][1]}
                            for degree, keep in closest(targets, {keep: value[1] for keep, value in sampled.items()}).items()}
    targets, own = rep_reductions(degrees, range(0, 200))
    report['rep-prompt'] = {degree: {'target': targets[degree], 'rep_merge_max': maximum, 'reduction': own[maximum]}
                            for degree, maximum in closest(targets, own).items()}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
