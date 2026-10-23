"""Stage 1 grids: eager timing of ToMe-family operators, and timing controls that split the cost.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.stage1 --grid eager --label cal07-rtx4090-gpu1 --results DIR

``eager``: fp32 and fp16, batch 1, 16 and 64, degree 0 (dense) and 8, 16, 24 of tome, cls-prune, evit and pitome
in the study ViT and in the CL plug-in, and of tome in the official package (unpatched dense, patched r=0).
``ablation``: the study ViT with the real merge, with the merge replaced by dropping the last tokens
(shrink), and with the size bias removed from attention (unmasked), to separate what each part costs.
"""
import argparse
from pathlib import Path

from scripts.token_efficiency import weights

from .arms import VARIANTS
from .grid import run_grid
from .run_unit import unit_id

OPERATORS = ('tome', 'cls-prune', 'evit', 'pitome')
DEGREES = (8, 16, 24)
BATCHES = (1, 16, 64)
SETTINGS = {'warmup_iters': 10, 'warmup_seconds': 2.0, 'min_iters': 30, 'min_seconds': 3.0, 'profile_batches': [1, 16],
            'energy_batches': [], 'graph': True}


def grid_units(root, grid='eager', device='cuda', dtypes=('fp32', 'fp16'), size='tiny', arms=('study', 'plugin', 'official')):
    common = {'size': size, 'device': device, 'batches': list(BATCHES), 'settings': SETTINGS,
              'weights': str(Path(root) / 'weights'), 'images': str(Path(root) / 'bench_images.npy')}
    units = []
    for dtype in dtypes:
        if grid == 'ablation':
            units.append(dict(common, arm='study', dtype=dtype, operator='dense', degree=0))
            units.extend(dict(common, arm='study', dtype=dtype, operator='tome', degree=degree, variant=variant)
                         for degree in DEGREES for variant in VARIANTS)
            continue
        for arm in ('study', 'plugin'):
            if arm not in arms:
                continue
            units.append(dict(common, arm=arm, dtype=dtype, operator='dense', degree=0))
            for operator in OPERATORS:
                units.extend(dict(common, arm=arm, dtype=dtype, operator=operator, degree=degree) for degree in DEGREES)
        if 'official' in arms:
            units.append(dict(common, arm='official', dtype=dtype, operator='dense', degree=0))
            units.append(dict(common, arm='official', dtype=dtype, operator='tome', degree=0, patched=True))
            units.extend(dict(common, arm='official', dtype=dtype, operator='tome', degree=degree) for degree in DEGREES)
    return units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', choices=['eager', 'ablation'], default='eager')
    parser.add_argument('--label', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--root', default=str(weights.CACHE.parent), help='folder with weights/ and bench_images.npy')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only', help='substring of the unit id')
    parser.add_argument('--arms', nargs='+', default=['study', 'plugin', 'official'], choices=['study', 'plugin', 'official'])
    parser.add_argument('--dtypes', nargs='+', default=['fp32', 'fp16'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cpu-guard', action='store_true', help='edge boards: also wait while another process keeps a core busy')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    units = grid_units(args.root, args.grid, args.device, tuple(args.dtypes), arms=tuple(args.arms))
    if args.only:
        units = [unit for unit in units if args.only in unit_id(unit)]
    units = units[:args.limit]
    if args.list:
        for unit in units:
            print(unit_id(unit))
        return
    run_grid(units, unit_id, 'scripts.tome_latency.run_unit', Path(args.results) / args.label, cpu_guard=args.cpu_guard)


if __name__ == '__main__':
    main()
