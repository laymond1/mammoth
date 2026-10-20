"""Device measurements: one process per (model, precision, method), every batch size inside it.

    python -m token_efficiency.bench grid --grid regime --label cal07-rtx4090 --results DIR
    python -m token_efficiency.bench unit --unit '{...}' --out FILE      (called by ``grid``)

Batch sizes run in increasing order, so the process's peak resident memory after each batch is the
peak of that batch. A unit whose process dies (out of memory, segfault) is retried once; a batch that
runs out of memory is recorded as such and the larger batches are skipped.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data, flops, measure, methods, weights
from .vit import build

PACKAGE = Path(__file__).resolve().parent
# Folder that holds the top-level package: the repository root here, the copy's parent on a device.
TOP = PACKAGE.parents[__package__.count('.')]
DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def unit_id(unit):
    kind = f"__{unit['train']}" if unit.get('train') else ''
    return f"{unit['size']}{kind}__{unit['dtype']}__{methods.spec_id(unit['spec'])}"


def measure_batch(model, forward, info, images, unit, batch, device):
    dtype = DTYPES[unit['dtype']]
    x = data.normalize(images[:batch], device, dtype)
    settings = unit['settings']
    record = {'batch': batch}
    with torch.inference_mode():
        measure.memory_reset(device)
        warm = measure.run_for(forward, x, device, settings['warmup_iters'], settings['warmup_seconds'])
        record['warmup'] = {'iterations': len(warm), 'first_ms': warm[0] * 1e3}
        record['eager'] = measure.summary(measure.run_for(forward, x, device, settings['min_iters'], settings['min_seconds']))
        record['memory'] = measure.memory_peak(device)
        state = info.get('state')
        if state is not None:
            record['token_trace'] = list(state.trace)
        if device.type == 'cuda':
            if info['capturable'] and settings.get('graph', True):
                try:
                    graph = measure.capture_graph(forward, x)
                    replay = lambda _: graph.replay()
                    measure.run_for(replay, x, device, 10, 0.2)
                    record['graph'] = measure.summary(measure.run_for(replay, x, device, settings['min_iters'], settings['min_seconds']))
                    del graph
                except RuntimeError as error:
                    record['graph'] = {'error': str(error)[:300]}
            if batch in settings.get('profile_batches', (1,)):
                record['profile'] = measure.kernel_profile(forward, x)
        if batch in settings.get('energy_batches', ()):
            record['energy'] = measure.measure_energy(forward, x, device, settings['energy_seconds'], batch)
        record['state_after'] = measure.device_state(device)
    return record


def run_unit(unit, out):
    device = torch.device(unit['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if unit.get('threads'):
        torch.set_num_threads(unit['threads'])
    torch.manual_seed(0)
    base = {'unit': unit, 'unit_id': unit_id(unit), 'code_sha256': code_digest(), 'environment': measure.environment(device),
            'state_before': measure.device_state(device), 'flops': dict(zip(('total', 'overhead'), methods.analytic_flops(unit['size'], unit['spec']))),
            'dense_flops': flops.dense(unit['size'])}
    load_start = time.perf_counter()
    model = build(unit['size'], weights.load(unit['size'], unit['weights']), device, DTYPES[unit['dtype']])
    base['load_seconds'] = time.perf_counter() - load_start
    base['memory_after_load'] = measure.memory_peak(device)
    base['weight_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
    spec = unit['spec']
    scorer = None
    if spec.get('selector') == 'cross':
        scorer = build(spec['scorer'], weights.load(spec['scorer'], unit['weights']), device, DTYPES[unit['dtype']])
        base['weight_bytes'] += sum(p.numel() * p.element_size() for p in scorer.parameters())
    forward, info = methods.build_forward(model, spec, scorer)
    images = np.load(unit['images'], mmap_mode='r')
    if spec['kind'] == 'ats':
        mean, counts = methods.ats_flops(model, spec, data.normalize(images, device, DTYPES[unit['dtype']]))
        base['flops'] = {'total': mean, 'overhead': 0, 'kept_mean': counts.mean(axis=0).tolist()}
    rows = []
    for batch in unit['batches']:
        try:
            row = measure_batch(model, forward, info, images, unit, batch, device)
        except torch.cuda.OutOfMemoryError as error:
            rows.append({'batch': batch, 'status': 'oom', 'error': str(error)[:300]})
            break
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower():
                raise
            rows.append({'batch': batch, 'status': 'oom', 'error': str(error)[:300]})
            break
        row['status'] = 'ok'
        rows.append(row)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    base['rows'] = rows
    Path(out).write_text(json.dumps(base, indent=1))


def run_train_unit(unit, out):
    """Training step cost for one (model, training kind, precision, reduction) at each batch size."""
    from .train import Trainer
    device = torch.device(unit['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if unit.get('threads'):
        torch.set_num_threads(unit['threads'])
    torch.manual_seed(0)
    settings = unit['settings']
    base = {'unit': unit, 'unit_id': unit_id(unit), 'code_sha256': code_digest(), 'environment': measure.environment(device),
            'state_before': measure.device_state(device)}
    images = np.load(unit['images'], mmap_mode='r')
    amp = unit['dtype'] == 'amp16'
    rows = []
    for batch in unit['batches']:
        try:
            model = build(unit['size'], weights.load(unit['size'], unit['weights']), device)
            trainer = Trainer(model, unit['train'], unit['spec'])
            x = data.normalize(images[:batch], device)
            labels = torch.randint(0, 100, (batch,), device=device)
            scaler = torch.amp.GradScaler('cuda') if amp else None
            autocast = (lambda: torch.autocast('cuda', dtype=torch.float16)) if amp else (lambda: None)
            step = lambda _: trainer.step(x, labels, scaler, autocast())
            measure.memory_reset(device)
            measure.run_for(step, x, device, settings['warmup_iters'], settings['warmup_seconds'])
            row = {'batch': batch, 'trainable': trainer.trainable,
                   'step': measure.summary(measure.run_for(step, x, device, settings['min_iters'], settings['min_seconds'])),
                   'memory': measure.memory_peak(device)}
            if batch in settings.get('energy_batches', ()):
                row['energy'] = measure.measure_energy(step, x, device, settings['energy_seconds'], batch)
            row['state_after'] = measure.device_state(device)
            row['status'] = 'ok'
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if isinstance(error, RuntimeError) and 'out of memory' not in str(error).lower():
                raise
            rows.append({'batch': batch, 'status': 'oom', 'error': str(error)[:300]})
            break
        rows.append(row)
        del trainer, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    base['rows'] = rows
    Path(out).write_text(json.dumps(base, indent=1))


# -------------------------------------------------------------------------------------------- grids
def regime_specs(size):
    """Question 1: where reduction pays off. The ToMe r=16 budget for every family."""
    target = flops.inblock(size, 'tome', 16)[0]
    specs = [{'kind': 'dense'}] + [{'kind': 'inblock', 'operator': op, 'degree': 16} for op in ('tome', 'pitome', 'evit', 'cls-prune')]
    specs.append({'kind': 'kept', 'selector': 'center', 'keep': flops.keep_for(size, target, 'center')})
    specs.append({'kind': 'continue', 'layer': 3, 'keep': flops.keep_for_continue(size, target, 3)})
    specs.append({'kind': 'kept', 'selector': 'cps', 'layer': 3, 'keep': flops.keep_for(size, target, 'cps', 3)})
    return specs


def sweep_specs(size, degrees=(4, 8, 12, 16, 20, 24)):
    """Question 6: every family over the compression degrees."""
    specs, seen = [{'kind': 'dense'}], {'dense'}
    specs += [{'kind': 'ats', 'rho': rho} for rho in (1.0, 0.72, 0.5)]
    seen.update(methods.spec_id(spec) for spec in specs)
    for degree in degrees:
        candidates = [{'kind': 'inblock', 'operator': op, 'degree': degree} for op in ('tome', 'pitome', 'evit', 'cls-prune')]
        matched = methods.matched_specs(size, degree)
        candidates += matched
        candidates += [{'kind': 'kept', 'selector': 'cps', 'layer': 12, 'keep': spec['keep']}
                       for spec in matched if spec['kind'] == 'kept' and spec.get('layer') == 3]
        for spec in candidates:
            if methods.spec_id(spec) not in seen:
                seen.add(methods.spec_id(spec))
                specs.append(spec)
    return specs


def train_specs(size):
    """Training: the ToMe r=16 FLOPs budget for each family, and CPS-Prompt's 12-layer score at the L3 count."""
    target = flops.inblock(size, 'tome', 16)[0]
    keep3 = flops.keep_for(size, target, 'cps', 3)
    return [{'kind': 'dense'}, {'kind': 'inblock', 'operator': 'tome', 'degree': 16}, {'kind': 'inblock', 'operator': 'evit', 'degree': 16},
            {'kind': 'kept', 'selector': 'random', 'keep': flops.keep_for(size, target, 'random')},
            {'kind': 'kept', 'selector': 'cps', 'layer': 3, 'keep': keep3},
            {'kind': 'kept', 'selector': 'cps', 'layer': 12, 'keep': keep3},
            {'kind': 'continue', 'layer': 3, 'keep': flops.keep_for_continue(size, target, 3)}]


PROFILES = {
    # name: (device, dtypes, regime batches, sweep batches, threads, sweep degrees)
    'gpu': ('cuda', ('fp32', 'fp16'), (1, 4, 16, 64, 256), (1, 64), None, (4, 8, 12, 16, 20, 24)),
    'jetson': ('cuda', ('fp32', 'fp16'), (1, 4, 16, 64), (1, 16), None, (8, 16, 24)),
    'cpu': ('cpu', ('fp32',), (1, 4, 16), (1, 16), 4, (8, 16, 24)),
}


def grid_units(grid, profile, root):
    device, dtypes, regime_batches, sweep_batches, threads, degrees = PROFILES[profile]
    settings = {'warmup_iters': 10, 'warmup_seconds': 2.0, 'min_iters': 30, 'min_seconds': 3.0,
                'energy_seconds': 10.0 if device == 'cuda' else 20.0, 'profile_batches': (1, 64)}
    if device == 'cpu':
        settings.update(warmup_iters=3, min_iters=10)
    units = []
    if grid == 'train':
        train_dtypes = ('fp32', 'amp16') if device == 'cuda' else ('fp32',)
        train_settings = dict(settings, warmup_iters=5, warmup_seconds=2.0, min_iters=20, min_seconds=3.0, energy_batches=(16,))
        train_sizes = ('tiny', 'small', 'base')
        if device == 'cpu':
            # A ViT-B step takes tens of seconds on four A76 cores: fewer steps, and ViT-Ti/S only.
            train_settings.update(warmup_iters=2, warmup_seconds=0.0, min_iters=5, min_seconds=0.0)
            train_sizes = ('tiny', 'small')
        for size in train_sizes:
            for kind in ('full', 'lora', 'prompt'):
                for dtype in train_dtypes:
                    for spec in train_specs(size):
                        units.append({'size': size, 'train': kind, 'dtype': dtype, 'spec': spec, 'device': device, 'batches': [16],
                                      'threads': threads, 'settings': train_settings, 'weights': str(Path(root) / 'weights'),
                                      'images': str(Path(root) / 'bench_images.npy')})
        return units
    for size in ('tiny', 'small', 'base'):
        for dtype in dtypes:
            if grid == 'regime':
                specs, batches = regime_specs(size), regime_batches
                settings_here = dict(settings, energy_batches=(1, batches[-2]))
            else:
                specs, batches = sweep_specs(size, degrees), sweep_batches
                settings_here = dict(settings, energy_batches=(1,), graph=False, warmup_seconds=1.0, min_seconds=2.0,
                                     energy_seconds=5.0 if profile == 'gpu' else 10.0)
                if device == 'cpu':
                    settings_here.update(min_iters=5)
            for spec in specs:
                units.append({'size': size, 'dtype': dtype, 'spec': spec, 'device': device, 'batches': list(batches),
                              'threads': threads, 'settings': settings_here, 'weights': str(Path(root) / 'weights'),
                              'images': str(Path(root) / 'bench_images.npy')})
    return units


def run_grid(grid, profile, label, results, root, limit=None, retries=2, share_machine=False, cpu_guard=False):
    folder = Path(results) / label / grid
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(PROFILES[profile][0])
    idle = folder / 'idle_power.json'
    units = grid_units(grid, profile, root)[:limit]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONFAULTHANDLER='1')
    for index, unit in enumerate(units):
        out = folder / f'{unit_id(unit)}.json'
        if out.exists():
            continue
        # Timing needs the machine to itself: wait while another job runs on any of its GPUs
        # (or only on ours with --share-machine), and measure again if one started meanwhile.
        # Edge boards add a CPU check (any other process above half a core) and the Jetson GPU load.
        def busy():
            found = measure.other_gpu_processes(all_gpus=not share_machine)
            if cpu_guard:
                found += measure.busy_cpu_processes()
                load = measure.jetson_gpu_load()
                if load is not None and load > 10:
                    found.append(f'GR3D {load:.0f}%')
            return found
        for attempt in range(retries + 1):
            waited = 0
            while busy():
                if waited % 1800 == 0:
                    print(f'GPU busy (pids {busy()}), waiting', flush=True)
                time.sleep(60)
                waited += 60
            if not idle.exists():
                # Idle board power once per grid, taken while nothing else runs.
                idle.write_text(json.dumps({'before': measure.idle_power(device), 'state': measure.device_state(device),
                                            'environment': measure.environment(device)}, indent=1))
            start = time.time()
            done = subprocess.run([sys.executable, '-m', f'{__package__}.bench', 'unit', '--unit', json.dumps(unit), '--out', str(out)],
                                  cwd=str(TOP), env=env, capture_output=True, text=True)
            log = folder / f'{unit_id(unit)}.attempt{attempt}.log'
            if done.returncode == 0 and out.exists():
                intruders = busy()
                if intruders:
                    out.rename(folder / f'{unit_id(unit)}.contaminated{attempt}.json')
                    print(f'[{index + 1}/{len(units)}] {unit_id(unit)} discarded: pids {intruders} started during it', flush=True)
                    continue
                print(f'[{index + 1}/{len(units)}] {unit_id(unit)} {time.time() - start:.0f}s', flush=True)
                break
            log.write_text(done.stdout[-20000:] + '\n--- stderr ---\n' + done.stderr[-20000:])
            print(f'[{index + 1}/{len(units)}] {unit_id(unit)} failed (exit {done.returncode}), attempt {attempt}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['unit', 'grid', 'list'])
    parser.add_argument('--unit')
    parser.add_argument('--out')
    parser.add_argument('--grid', choices=['regime', 'sweep', 'train'])
    parser.add_argument('--profile', choices=sorted(PROFILES), default='gpu')
    parser.add_argument('--label')
    parser.add_argument('--results', default='results/token_efficiency/devices')
    parser.add_argument('--root', default=str(weights.CACHE.parent), help='folder with weights/ and bench_images.npy')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--share-machine', action='store_true', help='only require our own GPU to be free')
    parser.add_argument('--cpu-guard', action='store_true', help='also wait while another process keeps a CPU core busy (edge boards)')
    args = parser.parse_args()
    if args.command == 'unit':
        unit = json.loads(args.unit)
        (run_train_unit if unit.get('train') else run_unit)(unit, args.out)
    elif args.command == 'list':
        for unit in grid_units(args.grid, args.profile, args.root)[:args.limit]:
            print(unit_id(unit), unit['batches'])
    else:
        run_grid(args.grid, args.profile, args.label, args.results, args.root, args.limit, share_machine=args.share_machine, cpu_guard=args.cpu_guard)


if __name__ == '__main__':
    main()
