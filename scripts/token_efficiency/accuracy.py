"""ImageNet-val top-1 of every path, once, on a server GPU: accuracy does not depend on the device.

Besides top-1, every image's predicted class, top-1 probability and top-1 minus top-2 probability
are stored, so the confidence-gated cascades of question 5 can be evaluated afterwards for any
threshold without running the models again.

    python -m scripts.token_efficiency.accuracy grid --out results/token_efficiency/accuracy
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data, flops, methods, weights
from .vit import build

DEGREES = (4, 8, 12, 16, 20, 24)
OPERATORS = ('tome', 'pitome', 'evit', 'cls-prune')


def evaluate(size, spec, model=None, batch=500, device='cuda', limit=None, seed=0):
    images = np.load(data.CACHE / 'images.npy', mmap_mode='r')
    labels = np.load(data.CACHE / 'labels.npy').astype(np.int64)
    count = len(labels) if limit is None else limit
    model = model or build(size, weights.load(size), device)
    scorer = build(spec['scorer'], weights.load(spec['scorer']), device) if spec.get('selector') == 'cross' else None
    forward, _ = methods.build_forward(model, spec, scorer)
    torch.manual_seed(seed)
    predicted, top1, margin, kept = [], [], [], []
    with torch.inference_mode():
        for start in range(0, count, batch):
            x = data.normalize(images[start:min(start + batch, count)], device)
            if spec['kind'] == 'ats':
                from . import ats
                trace = []
                logits = ats.forward(model, x, spec['rho'], trace=trace)
                kept.append(torch.stack(trace, dim=1).cpu())
            else:
                logits = forward(x)
            probability = logits.float().softmax(dim=1)
            best = probability.topk(2, dim=1)
            predicted.append(best.indices[:, 0].cpu())
            top1.append(best.values[:, 0].cpu())
            margin.append((best.values[:, 0] - best.values[:, 1]).cpu())
    predicted = torch.cat(predicted).numpy().astype(np.int16)
    result = {'predicted': predicted, 'top1_probability': torch.cat(top1).numpy().astype(np.float32),
              'margin': torch.cat(margin).numpy().astype(np.float32),
              'accuracy': float((predicted == labels[:count]).mean()), 'images': count}
    if kept:
        result['kept'] = torch.cat(kept).numpy().astype(np.int16)
        result['per_image_flops'] = np.array([flops.ats(size, row) for row in result['kept']], dtype=np.float64)
    return result


def score_mass(size, layers=(2, 3), batch=500, device='cuda', out=None):
    """Per image: share of the layer-L critical-score mass held by its top-k patches, for every k (question 5, S2)."""
    images = np.load(data.CACHE / 'images.npy', mmap_mode='r')
    model = build(size, weights.load(size), device)
    result = {}
    with torch.inference_mode():
        for layer in layers:
            shares = []
            for start in range(0, len(images), batch):
                score = model.scores(data.normalize(images[start:start + batch], device), layer)[0]
                ordered = score.sort(dim=1, descending=True).values
                shares.append((ordered.cumsum(dim=1) / ordered.sum(dim=1, keepdim=True)).half().cpu())
            result[f'layer{layer}'] = torch.cat(shares).numpy()
    if out:
        np.savez(out, **result)
    return result


def grid_specs(size):
    specs = [{'kind': 'dense'}]
    specs += [{'kind': 'inblock', 'operator': op, 'degree': r} for op in OPERATORS for r in DEGREES]
    seen = {methods.spec_id(spec) for spec in specs}
    # ATS: the release (rho 1), the paper's reduced budgets (0.87, 0.72) and two more.
    specs += [{'kind': 'ats', 'rho': rho} for rho in (1.0, 0.87, 0.72, 0.6, 0.5)]
    for degree in DEGREES:
        matched = methods.matched_specs(size, degree)
        # CPS-Prompt's 12-block score, random and centre patches at the kept counts of the early-layer
        # selections, to compare which patches are chosen at an equal count (question 4).
        extra = [{'kind': 'kept', 'selector': selector, 'keep': spec['keep'], **({'layer': 12} if selector == 'cps' else {})}
                 for spec in matched if spec['kind'] == 'kept' and spec.get('layer') in (2, 3)
                 for selector in ('cps', 'random', 'center')]
        for spec in matched + extra:
            if methods.spec_id(spec) not in seen:
                seen.add(methods.spec_id(spec))
                specs.append(spec)
    return specs


def run_grid(out, sizes=('tiny', 'small', 'base'), device='cuda'):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(out)
    manifest = json.loads((data.CACHE / 'manifest.json').read_text())
    for size in sizes:
        model = build(size, weights.load(size), device)
        folder = out / size
        folder.mkdir(parents=True, exist_ok=True)
        for spec in grid_specs(size):
            name = methods.spec_id(spec)
            if (folder / f'{name}.json').exists():
                continue
            start = time.time()
            result = evaluate(size, spec, model, device=device)
            total, overhead = methods.analytic_flops(size, spec)
            extra = {}
            if 'kept' in result:
                total, overhead = float(result['per_image_flops'].mean()), 0
                extra = {'kept': result['kept']}
            np.savez(folder / f'{name}.npz', predicted=result['predicted'], top1_probability=result['top1_probability'],
                     margin=result['margin'], **extra)
            record = {'size': size, 'spec': spec, 'id': name, 'accuracy': result['accuracy'], 'images': result['images'],
                      'flops': total, 'flops_ratio': total / flops.dense(size), 'overhead_flops': overhead,
                      'seconds': time.time() - start, 'cache_images_sha256': manifest['images_sha256'],
                      'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if device == 'cuda' else device}
            (folder / f'{name}.json').write_text(json.dumps(record, indent=1))
            print(json.dumps({key: record[key] for key in ('size', 'id', 'accuracy', 'flops_ratio', 'seconds')}), flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['grid', 'mass'])
    parser.add_argument('--out', default='results/token_efficiency/accuracy')
    parser.add_argument('--sizes', nargs='+', default=['tiny', 'small', 'base'])
    args = parser.parse_args()
    if args.command == 'mass':
        torch.backends.cuda.matmul.allow_tf32 = False
        for size in args.sizes:
            score_mass(size, out=Path(args.out) / size / 'score_mass.npz')
        return
    run_grid(args.out, args.sizes)


if __name__ == '__main__':
    main()
