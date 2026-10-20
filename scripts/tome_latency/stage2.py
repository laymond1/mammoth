"""Stage 2 grid: eager, manual CUDA graph and torch.compile modes of the study ViT, per degree and batch.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.stage2 --label cal07-rtx4090-gpu1 --results DIR

fp32: degree 0, 8, 16, 24 at batch 1 and 16 (``max-autotune`` only at degree 16, where it is slow to build);
fp16: degree 0 and 16 at batch 1 and 16.
"""
import argparse
from pathlib import Path

from scripts.token_efficiency import weights

from .compile_unit import unit_id
from .grid import run_grid

SETTINGS = {'warmup_iters': 10, 'warmup_seconds': 2.0, 'min_iters': 30, 'min_seconds': 3.0}
BASE_MODES = ('eager', 'cuda-graph', 'compile-default', 'compile-reduce-overhead')


def grid_units(root, size='tiny'):
    common = {'size': size, 'settings': SETTINGS, 'weights': str(Path(root) / 'weights'), 'images': str(Path(root) / 'bench_images.npy')}
    units = []
    for dtype, degrees in (('fp32', (0, 8, 16, 24)), ('fp16', (0, 16))):
        for degree in degrees:
            for batch in (1, 16):
                modes = BASE_MODES + (('compile-max-autotune-no-cudagraphs',) if dtype == 'fp32' and degree == 16 else ())
                units.append(dict(common, dtype=dtype, degree=degree, batch=batch, modes=list(modes)))
    return units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--root', default=str(weights.CACHE.parent))
    parser.add_argument('--only', help='substring of the unit id')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    units = [unit for unit in grid_units(args.root) if not args.only or args.only in unit_id(unit)]
    if args.list:
        for unit in units:
            print(unit_id(unit), unit['modes'])
        return
    run_grid(units, unit_id, 'scripts.tome_latency.compile_unit', Path(args.results) / args.label, timeout=5400)


if __name__ == '__main__':
    main()
