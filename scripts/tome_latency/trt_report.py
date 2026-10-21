"""Engine ratio against eager and compile ratios, with the verdict rule fixed before the engines ran (stage 3).

    python -m scripts.tome_latency.trt_report --trt FILE --eager STAGE1_DIR --compile STAGE2_DIR [--json FILE]

For every dtype, batch and degree: ``R_trt`` is the ToMe engine time over the degree-0 engine time of the same
precision and batch, i.e. a ratio inside one runtime. ``R_eager`` and ``R_compile`` are the same ratio in PyTorch eager and in
``torch.compile`` with CUDA graphs (``reduce-overhead``). The excess of a ratio is ``R - 1``.

Verdict (agreed before the engines were built): the engine excess is at most 30 % of the eager excess, so the slowness
belongs to the PyTorch execution; at least 70 %, so it belongs to the merge operator itself; in between it is mixed.
A ratio of 1.1 or less in eager has no slowness to explain.
"""
import argparse
import glob
import json
from pathlib import Path

from .report import load, ratio

LOW, HIGH = 0.30, 0.70


def verdict(r_trt, r_eager):
    if r_eager is None or r_eager <= 1.1:
        return 'no slowness in eager'
    excess = (r_trt - 1) / (r_eager - 1)
    if excess <= LOW:
        return f'execution ({excess:.0%} of eager excess)'
    if excess >= HIGH:
        return f'merge operator itself ({excess:.0%})'
    return f'mixed ({excess:.0%})'


def compile_ratios(folder):
    """ToMe / dense in ``compile-reduce-overhead`` per (dtype, degree, batch)."""
    times = {}
    for path in glob.glob(str(Path(folder) / '*.json')):
        record = json.loads(Path(path).read_text())
        unit = record['unit']
        for mode in record['modes']:
            if mode.get('mode') == 'compile-reduce-overhead' and 'timing' in mode:
                times[(unit['dtype'], unit['degree'], unit['batch'])] = mode['timing']['median_ms']
    return {key: value / times[(key[0], 0, key[2])] for key, value in times.items() if key[1] and (key[0], 0, key[2]) in times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trt', required=True)
    parser.add_argument('--eager', required=True, help='stage 1 result folder of the same device')
    parser.add_argument('--compile', required=True, help='stage 2 result folder of the same device')
    parser.add_argument('--json')
    args = parser.parse_args()
    engines = json.loads(Path(args.trt).read_text())['records']
    eager = load(args.eager)
    compiled = compile_ratios(args.compile)
    table = []
    print(f"{'dtype':<5}{'batch':>5}{'deg':>4} | {'dense ms':>8} {'tome ms':>8} | {'R_trt':>6} {'R_eager':>8} {'R_compile':>9} | verdict | logits vs eager fp32 (beyond tol / top1)")
    for record in sorted(engines.values(), key=lambda r: (r['precision'], r['batch'], r['degree'])):
        if not record['degree']:
            continue
        dense = engines.get(f"vit-tiny_dense_b{record['batch']}_{record['precision']}")
        if not dense:
            continue
        r_trt = record['timing']['median_ms'] / dense['timing']['median_ms']
        r_eager = ratio(eager, ('study', record['precision'], record['operator'], record['degree'], 'real', False),
                        ('study', record['precision'], 'dense', 0, 'real', False), record['batch'])
        r_compile = compiled.get((record['precision'], record['degree'], record['batch']))
        agree = record['vs_eager_fp32']
        row = {'precision': record['precision'], 'batch': record['batch'], 'degree': record['degree'], 'dense_ms': dense['timing']['median_ms'],
               'tome_ms': record['timing']['median_ms'], 'r_trt': r_trt, 'r_eager': r_eager, 'r_compile': r_compile,
               'verdict': verdict(r_trt, r_eager), 'beyond_tol': agree['share_beyond_tolerance'], 'top1': agree['top1_agreement']}
        table.append(row)
        print(f"{row['precision']:<5}{row['batch']:>5}{row['degree']:>4} | {row['dense_ms']:>8.3f} {row['tome_ms']:>8.3f} | {r_trt:>6.2f} "
              f"{r_eager if r_eager is None else round(r_eager, 2)!s:>8} {r_compile if r_compile is None else round(r_compile, 2)!s:>9} | {row['verdict']} | "
              f"{100 * row['beyond_tol']:.1f}% / {100 * row['top1']:.1f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))


if __name__ == '__main__':
    main()
