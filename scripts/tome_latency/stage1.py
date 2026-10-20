"""Stage 1 grid: eager timing of ToMe-family operators in three implementations, one process per unit.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.stage1 --label cal07-rtx4090-gpu1 --results DIR

Grid: fp32 and fp16, batch 1, 16 and 64, degree 0 (dense) and 8, 16, 24 of tome, cls-prune, evit and pitome
in the study ViT and in the CL plug-in, and of tome in the official package (unpatched dense, patched r=0).
A unit waits while another process uses any GPU of the machine and is discarded if one appeared during it.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.token_efficiency import measure, weights

from .run_unit import REPOSITORY, unit_id

OPERATORS = ('tome', 'cls-prune', 'evit', 'pitome')
DEGREES = (8, 16, 24)
BATCHES = (1, 16, 64)
SETTINGS = {'warmup_iters': 10, 'warmup_seconds': 2.0, 'min_iters': 30, 'min_seconds': 3.0, 'profile_batches': [1, 16],
            'energy_batches': [], 'graph': True}


def grid_units(root, device='cuda', dtypes=('fp32', 'fp16'), size='tiny'):
    common = {'size': size, 'device': device, 'batches': list(BATCHES), 'settings': SETTINGS,
              'weights': str(Path(root) / 'weights'), 'images': str(Path(root) / 'bench_images.npy')}
    units = []
    for dtype in dtypes:
        for arm in ('study', 'plugin'):
            units.append(dict(common, arm=arm, dtype=dtype, operator='dense', degree=0))
            for operator in OPERATORS:
                units.extend(dict(common, arm=arm, dtype=dtype, operator=operator, degree=degree) for degree in DEGREES)
        units.append(dict(common, arm='official', dtype=dtype, operator='dense', degree=0))
        units.append(dict(common, arm='official', dtype=dtype, operator='tome', degree=0, patched=True))
        units.extend(dict(common, arm='official', dtype=dtype, operator='tome', degree=degree) for degree in DEGREES)
    return units


def run_grid(label, results, root, limit=None, retries=2, max_wait=1200, only=None):
    folder = Path(results) / label
    folder.mkdir(parents=True, exist_ok=True)
    units = grid_units(root)
    if only:
        units = [unit for unit in units if only in unit_id(unit)]
    units = units[:limit]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONFAULTHANDLER='1', JE_ARROW_MALLOC_CONF='background_thread:false')
    for index, unit in enumerate(units):
        out = folder / f'{unit_id(unit)}.json'
        if out.exists():
            continue
        for attempt in range(retries + 1):
            waited = 0
            while measure.other_gpu_processes(all_gpus=True) and waited < max_wait:
                if waited % 300 == 0:
                    print(f'GPU busy (pids {measure.other_gpu_processes(all_gpus=True)}), waiting', flush=True)
                time.sleep(30)
                waited += 30
            if measure.other_gpu_processes(all_gpus=True):
                print(f'[{index + 1}/{len(units)}] {unit_id(unit)} skipped: GPU still busy after {max_wait}s', flush=True)
                break
            start = time.time()
            done = subprocess.run([sys.executable, '-m', 'scripts.tome_latency.run_unit', '--unit', json.dumps(unit), '--out', str(out)],
                                  cwd=str(REPOSITORY), env=env, capture_output=True, text=True)
            if done.returncode == 0 and out.exists():
                intruders = measure.other_gpu_processes(all_gpus=True)
                if intruders:
                    out.rename(folder / f'{unit_id(unit)}.contaminated{attempt}.json')
                    print(f'[{index + 1}/{len(units)}] {unit_id(unit)} discarded: pids {intruders} started during it', flush=True)
                    continue
                print(f'[{index + 1}/{len(units)}] {unit_id(unit)} {time.time() - start:.0f}s', flush=True)
                break
            (folder / f'{unit_id(unit)}.attempt{attempt}.log').write_text(done.stdout[-20000:] + '\n--- stderr ---\n' + done.stderr[-20000:])
            print(f'[{index + 1}/{len(units)}] {unit_id(unit)} failed (exit {done.returncode}), attempt {attempt}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--root', default=str(weights.CACHE.parent), help='folder with weights/ and bench_images.npy')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only', help='substring of the unit id')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        for unit in grid_units(args.root):
            print(unit_id(unit))
        return
    run_grid(args.label, args.results, args.root, args.limit, only=args.only)


if __name__ == '__main__':
    main()
