"""Does one torch.profiler session slow down the eager launches that follow it in the same process?

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.profiler_residue --out FILE

The grid times batch 1, profiles it, and then times batch 16 in the same process. If the profiler leaves its
CUPTI callbacks on, every later kernel launch pays a little, and a launch-bound call with hundreds of kernels
pays a lot. Each case is timed (a) in a fresh process state, (b) after a profile at batch 1, (c) after
a second profile; ``fresh`` is run in its own process so (a) is never preceded by a profile.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.token_efficiency import data, measure, weights

from . import arms

SETTINGS = (10, 2.0, 30, 3.0)


def eager_ms(forward, x, device):
    measure.run_for(forward, x, device, SETTINGS[0], SETTINGS[1])
    return measure.summary(measure.run_for(forward, x, device, SETTINGS[2], SETTINGS[3]))['median_ms']


def run_case(dtype, operator, degree, root, profile_first):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    forward, info = arms.build('study', 'tiny', dtype, device, operator, degree, str(Path(root) / 'weights'))
    images = np.load(Path(root) / 'bench_images.npy', mmap_mode='r')
    x1, x16 = (data.normalize(images[:n], device, arms.DTYPES[dtype]) for n in (1, 16))
    result = {'dtype': dtype, 'operator': operator, 'degree': degree, 'profile_first': profile_first}
    with torch.inference_mode():
        result['b1_before'] = eager_ms(forward, x1, device)
        if profile_first:
            result['kernels_b1'] = measure.kernel_profile(forward, x1)['kernels']
        result['b16'] = eager_ms(forward, x16, device)
        result['b1_after'] = eager_ms(forward, x1, device)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(weights.CACHE.parent))
    parser.add_argument('--out')
    parser.add_argument('--case', help='JSON of one case (internal)')
    args = parser.parse_args()
    if args.case:
        case = json.loads(args.case)
        print(json.dumps(run_case(case['dtype'], case['operator'], case['degree'], args.root, case['profile_first'])))
        return
    rows = []
    for dtype in ('fp32', 'fp16'):
        for operator, degree in (('dense', 0), ('tome', 16)):
            for profile_first in (False, True):
                case = {'dtype': dtype, 'operator': operator, 'degree': degree, 'profile_first': profile_first}
                done = subprocess.run([sys.executable, '-m', 'scripts.tome_latency.profiler_residue', '--root', args.root, '--case', json.dumps(case)],
                                      capture_output=True, text=True)
                rows.append(json.loads(done.stdout.strip().splitlines()[-1]))
                print(rows[-1], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    print('\nbatch 16 eager ms without / with a batch-1 profile before it')
    for dtype in ('fp32', 'fp16'):
        for operator in ('dense', 'tome'):
            pair = {r['profile_first']: r for r in rows if r['dtype'] == dtype and r['operator'] == operator}
            print(f"  {dtype} {operator:<6} {pair[False]['b16']:.3f} / {pair[True]['b16']:.3f}   (batch 1: {pair[False]['b1_before']:.3f} -> after profile {pair[True]['b1_after']:.3f})")


if __name__ == '__main__':
    main()
