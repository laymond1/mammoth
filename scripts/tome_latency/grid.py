"""Run units one process at a time on a machine that nothing else is using.

A unit waits while another process uses any GPU of the machine and is discarded if one appeared during it,
so a timing is never taken next to somebody else's job. A unit whose process dies is retried.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.token_efficiency import measure

from .run_unit import REPOSITORY


def run_grid(units, ident, module, folder, retries=2, max_wait=1200, timeout=3600, cpu_guard=False):
    """``units`` are dicts; ``module`` is run as ``python -m module --unit JSON --out FILE``; ``ident(unit)`` names the output.

    ``cpu_guard`` (edge boards): also wait while another process keeps a CPU core more than half busy, or the
    Jetson GPU load is above 10 %, as the five-device study did.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONFAULTHANDLER='1', JE_ARROW_MALLOC_CONF='background_thread:false')

    def busy():
        found = measure.other_gpu_processes(all_gpus=True)
        if cpu_guard:
            found += measure.busy_cpu_processes()
            load = measure.jetson_gpu_load()
            if load is not None and load > 10:
                found.append(f'GR3D {load:.0f}%')
        return found

    for index, unit in enumerate(units):
        name = ident(unit)
        out = folder / f'{name}.json'
        if out.exists():
            continue
        for attempt in range(retries + 1):
            waited = 0
            while busy() and waited < max_wait:
                if waited % 300 == 0:
                    print(f'machine busy ({busy()}), waiting', flush=True)
                time.sleep(30)
                waited += 30
            if busy():
                print(f'[{index + 1}/{len(units)}] {name} skipped: machine still busy after {max_wait}s', flush=True)
                break
            start = time.time()
            try:
                done = subprocess.run([sys.executable, '-m', module, '--unit', json.dumps(unit), '--out', str(out)],
                                      cwd=str(REPOSITORY), env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                (folder / f'{name}.attempt{attempt}.log').write_text(f'timeout after {timeout}s\n{(error.stdout or b"")[-5000:]}')
                print(f'[{index + 1}/{len(units)}] {name} timed out after {timeout}s, attempt {attempt}', flush=True)
                continue
            if done.returncode == 0 and out.exists():
                intruders = busy()
                if intruders:
                    out.rename(folder / f'{name}.contaminated{attempt}.json')
                    print(f'[{index + 1}/{len(units)}] {name} discarded: pids {intruders} started during it', flush=True)
                    continue
                print(f'[{index + 1}/{len(units)}] {name} {time.time() - start:.0f}s', flush=True)
                break
            (folder / f'{name}.attempt{attempt}.log').write_text(done.stdout[-20000:] + '\n--- stderr ---\n' + done.stderr[-20000:])
            print(f'[{index + 1}/{len(units)}] {name} failed (exit {done.returncode}), attempt {attempt}', flush=True)
