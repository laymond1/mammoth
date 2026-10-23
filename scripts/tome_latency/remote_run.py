"""Clean re-measurement of one device: eager grid (study ViT), timing controls, then compile modes.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.remote_run --label cal08-rtx2080ti-gpu1 \\
        --results ~/tome_latency_results/cal08 --root ~/.cache/token_efficiency [--cpu-guard] [--no-compile]

Run it from the folder that holds ``scripts/`` (a copy of one commit). Units run one process at a time and wait
while anything else uses the machine (``grid.py``); profiling always comes after the last timing.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--root', required=True, help='folder with weights/ and bench_images.npy')
    parser.add_argument('--cpu-guard', action='store_true', help='edge boards: also wait while another process keeps a core busy')
    parser.add_argument('--no-compile', action='store_true', help='no triton on this machine: eager and manual CUDA graph only')
    args = parser.parse_args()
    results = Path(args.results).expanduser()
    results.mkdir(parents=True, exist_ok=True)
    guard = ['--cpu-guard'] if args.cpu_guard else []
    env = dict(os.environ, JE_ARROW_MALLOC_CONF='background_thread:false', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.getcwd())
    common = ['--label', args.label, '--root', str(Path(args.root).expanduser())]
    steps = [
        ('eager grid', 'stage1.log', ['scripts.tome_latency.stage1', '--grid', 'eager', '--arms', 'study', '--results', str(results / 'stage1'), *common, *guard]),
        ('ablation', 'ablation.log', ['scripts.tome_latency.stage1', '--grid', 'ablation', '--results', str(results / 'ablation'), *common, *guard]),
        ('stage 2', 'stage2.log', ['scripts.tome_latency.stage2', '--no-compile' if args.no_compile else '--no-autotune',
                                   '--results', str(results / 'stage2'), *common, *guard]),
    ]
    for name, log, argv in steps:
        print(time.strftime('%H:%M:%S'), name, flush=True)
        with open(results / log, 'w') as stream:
            subprocess.run([sys.executable, '-m', *argv], stdout=stream, stderr=subprocess.STDOUT, env=env)
    print(time.strftime('%H:%M:%S'), 'done', flush=True)


if __name__ == '__main__':
    main()
