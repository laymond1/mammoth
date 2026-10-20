"""Measure one (arm, dtype, operator, degree) unit at every batch size, in its own process.

    python -m scripts.tome_latency.run_unit --unit '{...}' --out FILE      (called by ``stage1``)

Timing rules are those of ``token_efficiency.bench.measure_batch``: warm-up, one device synchronisation
per call, ``inference_mode``, median of the timed calls, TF32 off, and a CUDA graph replay of the same call.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.token_efficiency import bench, measure

from . import arms

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[1]


def unit_id(unit):
    operator = unit['operator']
    name = 'dense' if operator == 'dense' else f"{operator}-r{unit['degree']}"
    variant = unit.get('variant', 'real')
    return (f"{unit['arm']}__{unit['size']}__{unit['dtype']}__{name}{'-patched' if unit.get('patched') else ''}"
            f"{'' if variant == 'real' else '__' + variant}")


def code_state():
    """Digest of both packages and the git commit they come from, so a result names the code that made it."""
    digest = hashlib.sha256()
    for path in sorted([*PACKAGE.glob('*.py'), *bench.PACKAGE.glob('*.py')]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=REPOSITORY, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {'sha256': digest.hexdigest(), 'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def cpu_clocks():
    """Current clock of every core in MHz (min, mean, max): batch 1 is bound by how fast the CPU issues kernels."""
    values = []
    for entry in sorted(Path('/sys/devices/system/cpu').glob('cpu[0-9]*')):
        try:
            values.append(int((entry / 'cpufreq' / 'scaling_cur_freq').read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return {'min': min(values), 'mean': sum(values) / len(values), 'max': max(values)} if values else None


def run_unit(unit, out):
    device = torch.device(unit['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    base = {'unit': unit, 'unit_id': unit_id(unit), 'code': code_state(), 'environment': measure.environment(device),
            'state_before': measure.device_state(device), 'cpu_clocks_before': cpu_clocks(),
            'env': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'TOME_LATENCY_GEMM_PATCH', 'PYTHONPATH')}}
    start = time.perf_counter()
    forward, info = arms.build(unit['arm'], unit['size'], unit['dtype'], device, unit['operator'], unit.get('degree', 0),
                               unit['weights'], patched=unit.get('patched', False), variant=unit.get('variant', 'real'))
    base['build_seconds'] = time.perf_counter() - start
    images = np.load(unit['images'], mmap_mode='r')
    rows = []
    for batch in unit['batches']:
        row = bench.measure_batch(None, forward, info, images, unit, batch, device)
        row['cpu_clocks_after'] = cpu_clocks()
        row['status'] = 'ok'
        rows.append(row)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    base['rows'] = rows
    Path(out).write_text(json.dumps(base, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run_unit(json.loads(args.unit), args.out)


if __name__ == '__main__':
    main()
