"""Question 5: speculative-decoding-style designs, evaluated from stored per-image predictions.

S1 draft -> check -> re-run.  A cheap path answers; if its top-1 probability (or top-1 minus top-2)
   is below a threshold, the image is re-run densely. With a ``continue`` draft at layer L the re-run
   reuses blocks 1..L-1 and the attention of block L (the KV-cache analogue), so it only pays the MLP
   of block L and blocks L+1..12 on all tokens; the re-run's answer is exactly the dense answer.
S2 the draft picks the budget.  The layer-L scores decide per image how many patches to keep:
   images whose score mass is concentrated keep few, the others keep more.
S3 small-model draft.  ViT-Ti answers; when unsure, a larger ViT runs.

All accuracies are on the 50,000 ImageNet-val images; costs are analytic FLOPs per image.
"""
import json
from pathlib import Path

import numpy as np

from . import data, flops, methods

THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)


def load_run(folder, size, spec):
    name = methods.spec_id(spec)
    arrays = np.load(Path(folder) / size / f'{name}.npz')
    record = json.loads((Path(folder) / size / f'{name}.json').read_text())
    return {key: arrays[key] for key in arrays.files} | {'record': record}


def rerun_flops(size, spec):
    """FLOPs of the dense re-run after a draft: shared work is reused only for ``continue`` drafts."""
    width = flops.width_of(size)
    tokens = flops.PATCHES + 1
    if spec['kind'] == 'continue':
        layer = spec['layer']
        return flops.mlp(tokens, width) + (flops.DEPTH - layer) * flops.block(tokens, width) + flops.head(width)
    return flops.dense(size)


def curve(draft, dense, labels, draft_flops, rerun, gate='top1_probability'):
    """Accuracy, acceptance, agreement with the dense answer and mean FLOPs at each threshold."""
    confidence = draft[gate]
    draft_right = draft['predicted'] == labels
    dense_right = dense['predicted'] == labels
    same = draft['predicted'] == dense['predicted']
    rows = []
    for threshold in THRESHOLDS:
        accept = confidence >= threshold
        rate = float(accept.mean())
        rows.append({'threshold': float(threshold), 'acceptance': rate,
                     'accuracy': float(np.where(accept, draft_right, dense_right).mean()),
                     'agreement_with_dense': float(np.where(accept, same, True).mean()),
                     'flops': draft_flops + (1 - rate) * rerun})
    return rows


def s1(folder, size, spec, gate='top1_probability'):
    labels = np.load(data.CACHE / 'labels.npy').astype(np.int64)
    draft, dense = load_run(folder, size, spec), load_run(folder, size, {'kind': 'dense'})
    draft_flops = methods.analytic_flops(size, spec)[0]
    rows = curve(draft, dense, labels, draft_flops, rerun_flops(size, spec), gate)
    base = flops.dense(size)
    for row in rows:
        row['flops_ratio'] = row['flops'] / base
    return {'size': size, 'draft': methods.spec_id(spec), 'gate': gate, 'dense_accuracy': float((dense['predicted'] == labels).mean()),
            'draft_accuracy': float((draft['predicted'] == labels).mean()), 'draft_flops_ratio': draft_flops / base,
            'rerun_flops_ratio': rerun_flops(size, spec) / base, 'rows': rows}


def s3(folder, small, large, gate='top1_probability'):
    """ViT-``small`` answers; below the threshold the dense ViT-``large`` runs from scratch."""
    labels = np.load(data.CACHE / 'labels.npy').astype(np.int64)
    draft, target = load_run(folder, small, {'kind': 'dense'}), load_run(folder, large, {'kind': 'dense'})
    rows = curve(draft, target, labels, flops.dense(small), flops.dense(large), gate)
    for row in rows:
        row['flops_ratio'] = row['flops'] / flops.dense(large)
    return {'draft': small, 'target': large, 'gate': gate, 'rows': rows,
            'target_accuracy': float((target['predicted'] == labels).mean()),
            'draft_accuracy': float((draft['predicted'] == labels).mean())}


def pareto(points, x='flops_ratio', y='accuracy'):
    """Points not beaten by another point with no more cost and no less accuracy."""
    best, front = -1.0, []
    for point in sorted(points, key=lambda p: (p[x], -p[y])):
        if point[y] > best:
            front.append(point)
            best = point[y]
    return front


def s2(folder, size, layer, coverages=np.round(np.linspace(0.3, 0.99, 70), 3)):
    """Per image, keep the fewest patches (among the stored ``continue`` runs) holding ``coverage`` of the score mass."""
    labels = np.load(data.CACHE / 'labels.npy').astype(np.int64)
    runs = sorted((json.loads(path.read_text()) for path in (Path(folder) / size).glob(f'continue-L{layer}-k*.json')),
                  key=lambda record: record['spec']['keep'])
    keeps = [record['spec']['keep'] for record in runs]
    right = np.stack([load_run(folder, size, record['spec'])['predicted'] == labels for record in runs])
    cost = np.array([record['flops'] for record in runs])
    mass = np.load(Path(folder) / size / 'score_mass.npz')[f'layer{layer}'].astype(np.float32)
    held = mass[:, np.array(keeps) - 1]  # share of the mass inside the top-k, for every stored k
    rows = []
    for coverage in coverages:
        choice = np.where(held >= coverage, np.arange(len(keeps))[None], len(keeps) - 1).min(axis=1)
        rows.append({'coverage': float(coverage), 'accuracy': float(right[choice, np.arange(len(labels))].mean()),
                     'flops_ratio': float(cost[choice].mean() / flops.dense(size)),
                     'mean_keep': float(np.array(keeps)[choice].mean())})
    static = [{'keep': k, 'accuracy': float(r.mean()), 'flops_ratio': float(c / flops.dense(size))} for k, r, c in zip(keeps, right, cost)]
    return {'size': size, 'layer': layer, 'keeps': keeps, 'rows': rows, 'static': static}
