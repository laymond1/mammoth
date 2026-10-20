"""Every token-reduction path of the study as ``spec -> forward(images) -> logits``.

A spec is a small dict, so a case can travel as JSON to any device:

    {'kind': 'dense'}
    {'kind': 'inblock', 'operator': 'tome' | 'pitome' | 'evit' | 'cls-prune', 'degree': r}
    {'kind': 'kept', 'selector': 'random' | 'center' | 'variance' | 'cps', 'keep': k, 'layer': L}
    {'kind': 'continue', 'layer': L, 'keep': k}
    {'kind': 'resolution', 'pixels': p}

``kept`` restarts from the patch embedding with the class token and ``keep`` patches; with the
``cps`` selector the patches are the top-``keep`` critical scores read at ``layer`` of the same
frozen model (L = 12 is CPS-Prompt's query pass, L = 2 or 3 the early-block selection).
``continue`` keeps the layer-L hidden states of the chosen tokens instead of restarting.
"""
import torch
from torch.nn import functional as F

from . import flops, operators
from .vit import PATCHES, Reduction

GRID = 14


def spec_id(spec):
    kind = spec['kind']
    if kind == 'dense':
        return 'dense'
    if kind == 'inblock':
        return f"{spec['operator']}-r{spec['degree']}"
    if kind == 'scheduled':
        value = spec['value']
        return f"{spec['operator']}-p{round(100 * value)}" if value < 1 else f"{spec['operator']}-r{value}"
    if kind == 'kept':
        if spec['selector'] == 'cross':
            return f"kept-cross{spec['scorer']}-L{spec['layer']}-k{spec['keep']}"
        layer = f"-L{spec['layer']}" if spec['selector'] == 'cps' else ''
        return f"kept-{spec['selector']}{layer}-k{spec['keep']}"
    if kind == 'continue':
        return f"continue-L{spec['layer']}-k{spec['keep']}"
    if kind == 'resolution':
        return f"res{spec['pixels']}"
    if kind == 'ats':
        return f"ats-p{round(100 * spec['rho'])}"
    raise ValueError(f'Unknown method kind {kind}')


def center_order():
    """Patch indices from the most to the least central (distance of patch centres to the image centre)."""
    rows, cols = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing='ij')
    distance = (rows - (GRID - 1) / 2) ** 2 + (cols - (GRID - 1) / 2) ** 2
    # Ties are broken by index so the order is the same on every device.
    return torch.as_tensor(sorted(range(PATCHES), key=lambda i: (distance.flatten()[i].item(), i)))


def patch_variance(images):
    """Pixel variance of every 16x16 patch over its three channels: (batch, 196)."""
    batch = images.shape[0]
    patches = images.reshape(batch, 3, GRID, 16, GRID, 16).permute(0, 2, 4, 1, 3, 5).reshape(batch, PATCHES, -1)
    return patches.float().var(dim=-1)


def resized_pos_embed(model, pixels):
    """Bicubic resize of the 14x14 position grid, as timm does for other input sizes."""
    side = pixels // 16
    pos = model.pos_embed
    grid = pos[:, 1:].reshape(1, GRID, GRID, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(side, side), mode='bicubic', align_corners=False).to(pos.dtype)
    return torch.cat((pos[:, :1], grid.permute(0, 2, 3, 1).reshape(1, side * side, -1)), dim=1)


def build_forward(model, spec, scorer=None):
    """Return ``(forward, info)``; ``info`` holds static facts such as whether CUDA graphs can capture it.

    ``scorer`` is the smaller frozen ViT whose layer-L scores choose patches for ``cross`` selection.
    """
    kind = spec['kind']
    info = {'capturable': True}
    if kind == 'dense':
        return model, info
    if kind == 'inblock':
        state = Reduction(spec['operator'], spec['degree'])
        info['state'] = state
        return (lambda images: model(images, state)), info
    if kind == 'scheduled':
        state = Reduction(spec['operator'], schedule=operators.schedule(spec['operator'], spec['value']))
        info['state'] = state
        # PPT splits the batch by policy with a host synchronisation, which CUDA graphs cannot capture.
        info['capturable'] = spec['operator'] != 'ppt'
        return (lambda images: model(images, state)), info
    if kind == 'kept':
        keep, selector = spec['keep'], spec['selector']
        if selector == 'center':
            order = center_order()[:keep].to(model.pos_embed.device)

            def choose(images, embedded):
                return order.expand(len(images), -1)
        elif selector == 'random':
            def choose(images, embedded):
                return torch.rand(len(images), PATCHES, device=images.device).topk(keep, dim=1).indices
        elif selector == 'variance':
            def choose(images, embedded):
                return patch_variance(images).topk(keep, dim=1).indices
        elif selector == 'cps':
            layer = spec['layer']

            def choose(images, embedded):
                return model.scores(images, layer, embedded)[0].topk(keep, dim=1).indices
        elif selector == 'cross':
            layer = spec['layer']

            def choose(images, embedded):
                return scorer.scores(images, layer)[0].topk(keep, dim=1).indices
        else:
            raise ValueError(f'Unknown selector {selector}')

        def forward(images):
            embedded = model.embed(images)
            return model.forward_kept(images, choose(images, embedded), embedded)
        return forward, info
    if kind == 'continue':
        layer, keep = spec['layer'], spec['keep']

        def forward(images):
            score, state = model.scores(images, layer)
            return model.forward_continue(layer, state, score.topk(keep, dim=1).indices)
        return forward, info
    if kind == 'ats':
        from . import ats
        info['capturable'] = False  # the kept count depends on the images
        return (lambda images: ats.forward(model, images, spec['rho'])), info
    if kind == 'resolution':
        pixels = spec['pixels']
        pos = resized_pos_embed(model, pixels)

        def forward(images):
            small = F.interpolate(images, size=(pixels, pixels), mode='bicubic', align_corners=False, antialias=True)
            x = model.patch_embed(small)
            x = torch.cat((model.cls_token.expand(len(x), -1, -1), x), dim=1) + pos
            return model.classify(model.run(x))
        return forward, info
    raise ValueError(f'Unknown method kind {kind}')


def ats_flops(model, spec, images, batch=64):
    """Mean per-image FLOPs of ATS on ``images`` (normalized, on the model's device) and the mean kept counts."""
    from . import ats
    counts = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            trace = []
            ats.forward(model, images[start:start + batch], spec['rho'], trace=trace)
            counts.append(torch.stack(trace, dim=1).cpu())
    counts = torch.cat(counts).numpy()
    per_image = [flops.ats(model.size, row) for row in counts]
    return float(sum(per_image) / len(per_image)), counts


def analytic_flops(size, spec):
    """(total FLOPs per image, of which the selection or merging overhead)."""
    kind = spec['kind']
    if kind == 'dense':
        return flops.dense(size), 0
    if kind == 'inblock':
        return flops.inblock(size, spec['operator'], spec['degree'])
    if kind == 'scheduled':
        return flops.scheduled(size, spec['operator'], operators.schedule(spec['operator'], spec['value']))
    if kind == 'kept':
        if spec['selector'] == 'cross':
            overhead = flops.embed(flops.width_of(spec['scorer'])) + flops.score_at(spec['scorer'], spec['layer'])
            return flops.kept(size, spec['keep'])[0] + overhead, overhead
        return flops.kept(size, spec['keep'], spec['selector'], spec.get('layer'))
    if kind == 'continue':
        return flops.continued(size, spec['layer'], spec['keep'])
    if kind == 'resolution':
        return flops.resolution(size, spec['pixels'])
    if kind == 'ats':
        return None, None  # depends on the images: measured per image (accuracy.py, bench.py)
    raise ValueError(f'Unknown method kind {kind}')


def matched_specs(size, degree, layers=(2, 3)):
    """Input-level specs whose FLOPs match the ToMe schedule at ``degree``, selection cost included.

    A selector whose cost alone exceeds the target is left out; CPS-Prompt's 12-block query pass
    is such a case below the dense budget, so it is compared at the kept counts of the early layers.
    """
    target = flops.inblock(size, 'tome', degree)[0]
    specs = []
    for selector, layer in [('random', None), ('center', None), ('variance', None)] + [('cps', layer) for layer in layers]:
        keep = flops.keep_for(size, target, selector, layer)
        if keep:
            spec = {'kind': 'kept', 'selector': selector, 'keep': keep}
            if layer:
                spec['layer'] = layer
            specs.append(spec)
    if size != 'tiny':
        # A ViT-Ti scorer chooses for the larger model (question 5, small-model draft).
        for layer in (3, 12):
            overhead = flops.embed(flops.width_of('tiny')) + flops.score_at('tiny', layer)
            keep = flops.keep_for(size, target - overhead)
            if keep:
                specs.append({'kind': 'kept', 'selector': 'cross', 'scorer': 'tiny', 'layer': layer, 'keep': keep})
    for layer in layers:
        keep = flops.keep_for_continue(size, target, layer)
        if keep:
            specs.append({'kind': 'continue', 'layer': layer, 'keep': keep})
    # Added in-block methods, at the official layers, with the per-layer amount that meets the budget.
    for operator, values in (('ppt', range(1, 99)), ('repshift', [v / 100 for v in range(1, 51)])):
        fitting = [v for v in values if flops.scheduled(size, operator, operators.schedule(operator, v))[0] <= target]
        if fitting:
            # FLOPs fall as the amount grows: the smallest amount within the budget comes closest to it.
            specs.append({'kind': 'scheduled', 'operator': operator, 'value': min(fitting)})
    pixels = max((p for p in range(32, 225, 16) if flops.resolution(size, p)[0] <= target), default=None)
    if pixels:
        specs.append({'kind': 'resolution', 'pixels': pixels})
    return specs
