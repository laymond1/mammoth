"""Stage 3 on one device from ONNX files that are already exported: build the engines, time them, check ImageNet top-1.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.trt_stage --onnx DIR --out DIR --root ~/.cache/token_efficiency \\
        [--accuracy] [--workspace-gb 4]

Run it from the folder that holds ``scripts/`` (a copy of one commit), on a machine nothing else is using: the engines are built
by timing tactics on the GPU, and the engine timings are taken next to nothing. Writes ``plans/``, ``trt_run.json`` and, with
``--accuracy``, ``trt_accuracy.json`` into ``--out``. ``trt_report`` then sets the engine ratios next to the eager and compile ones.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', required=True, help='folder with weights/, bench_images.npy and (for --accuracy) imagenet-val/')
    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--workspace-gb', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JE_ARROW_MALLOC_CONF='background_thread:false', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.getcwd())
    root = str(Path(args.root).expanduser())
    steps = [('build', ['scripts.tome_latency.trt_build', '--onnx', str(Path(args.onnx).expanduser()), '--out', str(out / 'plans'), '--workspace-gb', str(args.workspace_gb)]),
             ('run', ['scripts.tome_latency.trt_run', '--plans', str(out / 'plans'), '--out', str(out / 'trt_run.json'), '--root', root])]
    if args.accuracy:
        steps.append(('accuracy', ['scripts.tome_latency.trt_accuracy', '--plans', str(out / 'plans'), '--out', str(out / 'trt_accuracy.json'), '--root', root]))
    for name, argv in steps:
        print(time.strftime('%H:%M:%S'), name, flush=True)
        with open(out / f'{name}.log', 'w') as stream:
            done = subprocess.run([sys.executable, '-m', *argv], stdout=stream, stderr=subprocess.STDOUT, env=env)
        if done.returncode:
            print(f'{name} failed with exit code {done.returncode}; see {out / (name + ".log")}', flush=True)
            sys.exit(done.returncode)
    print(time.strftime('%H:%M:%S'), 'done', flush=True)


if __name__ == '__main__':
    main()
