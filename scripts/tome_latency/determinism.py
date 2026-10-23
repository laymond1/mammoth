"""How much the ToMe output changes from one run to the next, on the same inputs, before any compiler is involved.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.determinism --out FILE

``scatter_add`` sums with atomics, so the order of additions can differ between launches. A rounding difference
can flip a near-tie in the next block's matching and change a logit by 1e-2. This measures that baseline:
eager against its own first pass, and CUDA-graph replays against eager, over the same 256 images.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.token_efficiency import weights

from . import arms
from .compile_unit import agreement, chunks, graphed


def run(dtype, degree, batch, repeats, root, count=256):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    forward, _ = arms.build('study', 'tiny', dtype, device, 'tome', degree, str(Path(root) / 'weights'))
    inputs = chunks(np.load(Path(root) / 'bench_images.npy', mmap_mode='r'), device, arms.DTYPES[dtype], batch, count)
    result = {'dtype': dtype, 'degree': degree, 'batch': batch, 'repeats': repeats, 'eager_vs_first_eager': [], 'graph_vs_first_eager': []}
    with torch.inference_mode():
        reference = [forward(chunk).clone() for chunk in inputs]
        for _ in range(repeats):
            result['eager_vs_first_eager'].append(agreement(reference, [forward(chunk).clone() for chunk in inputs], dtype))
        graph, static, output = graphed(forward, inputs[0])
        for _ in range(repeats):
            outputs = []
            for chunk in inputs:
                static.copy_(chunk)
                graph.replay()
                outputs.append(output.clone())
            result['graph_vs_first_eager'].append(agreement(reference, outputs, dtype))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(weights.CACHE.parent))
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cases = [run(dtype, degree, batch, args.repeats, args.root) for dtype in ('fp32', 'fp16') for degree in (8, 16) for batch in (1, 16)]
    Path(args.out).write_text(json.dumps(cases, indent=1))
    for case in cases:
        for kind in ('eager_vs_first_eager', 'graph_vs_first_eager'):
            beyond = [entry['share_beyond_tolerance'] for entry in case[kind]]
            print(f"{case['dtype']} r{case['degree']} b{case['batch']:<3} {kind:<22} images beyond tolerance per repeat: "
                  f"mean {100 * sum(beyond) / len(beyond):.2f}%  max {100 * max(beyond):.2f}%  worst logit diff {max(e['max_abs_diff'] for e in case[kind]):.1e}")


if __name__ == '__main__':
    main()
