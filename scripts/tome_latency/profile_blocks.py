"""Kernels and GPU time of one forward pass, split by block and by part of the block (study ViT).

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.profile_blocks --dtype fp32 --batch 1 --operator tome --degree 16 --out FILE

Parts of a block: ``qkv`` (the projection), ``attn`` (scaled-dot-product attention and output projection),
``merge`` (the token-reduction operator), ``mlp`` (the MLP) and ``rest`` (norms, residual adds, size bias).
``record_function`` ranges mark the parts; a kernel belongs to the innermost range that contains the CPU-side
launch of that kernel, found through the correlation id in the Chrome trace. Ranges add CPU time, so the
CPU durations are only comparable within one profile; kernel counts and GPU time are not changed by them.
"""
import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.token_efficiency import data, measure, weights
from scripts.token_efficiency.vit import Attention, Block, Mlp, VisionTransformer

from . import arms

PARTS = ('qkv', 'attn', 'merge', 'mlp')
DEVICE_CATEGORIES = ('kernel', 'gpu_memcpy', 'gpu_memset')
LAUNCH_CATEGORIES = ('cuda_runtime', 'cuda_driver')


def attribute(events):
    """Per-block, per-part ``{'kernels', 'gpu_us', 'cpu_us'}`` from Chrome-trace events (all iterations summed)."""
    ranges = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
              and (e['name'].startswith('blk') or e['name'] in PARTS)]
    # cuBLAS launches its GEMMs through the driver API, and sorts issue memsets: count both, as the grid's
    # profile does, so the parts add up to the same number of device operations.
    launch = {e['args']['correlation']: e for e in events
              if e.get('cat') in LAUNCH_CATEGORIES and e.get('ph') == 'X' and 'correlation' in e.get('args', {})}
    table = defaultdict(lambda: {'kernels': 0, 'gpu_us': 0.0, 'cpu_us': 0.0})
    block_ranges = [r for r in ranges if r['name'].startswith('blk')]
    part_ranges = [r for r in ranges if r['name'] in PARTS]

    def inside(candidates, call):
        hits = [r for r in candidates if r['pid'] == call['pid'] and r['tid'] == call['tid'] and r['ts'] <= call['ts'] <= r['ts'] + r['dur']]
        return min(hits, key=lambda r: r['dur']) if hits else None

    for kernel in (e for e in events if e.get('cat') in DEVICE_CATEGORIES and e.get('ph') == 'X'):
        call = launch.get(kernel.get('args', {}).get('correlation'))
        block = inside(block_ranges, call) if call else None
        part = inside(part_ranges, call) if call and block else None
        key = (block['name'] if block else 'outside', part['name'] if part else ('rest' if block else 'embed-classify'))
        table[key]['kernels'] += 1
        table[key]['gpu_us'] += kernel['dur']
    for part in part_ranges:
        block = inside(block_ranges, part)
        table[(block['name'] if block else 'outside', part['name'])]['cpu_us'] += part['dur']
    for block in block_ranges:
        table[(block['name'], 'block')]['cpu_us'] += block['dur']
    return table


class _Marked:
    """The operator wrapped in a ``merge`` range."""

    def __init__(self, operator):
        self.operator = operator

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def __call__(self, *args):
        from torch.profiler import record_function
        with record_function('merge'):
            return self.operator(*args)


def instrument():
    """Wrap the study ViT's block parts in named ranges (process-wide, for a profiling process only)."""
    from torch.profiler import record_function

    def wrap(owner, name, label):
        original = getattr(owner, name)

        def marked(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)
        setattr(owner, name, marked)

    reduced = Block.reduced

    def reduced_marked(self, x, state, index):
        with record_function(f'blk{index:02d}'):
            return reduced(self, x, state, index)
    Block.reduced = reduced_marked

    # The dense path calls ``Block.forward`` in order; the counter restarts at every embedding.
    counter = [0]
    embed = VisionTransformer.embed
    forward = Block.forward

    def embed_marked(self, images):
        counter[0] = 0
        return embed(self, images)

    def forward_marked(self, x):
        index, counter[0] = counter[0], counter[0] + 1
        with record_function(f'blk{index:02d}'):
            return forward(self, x)
    VisionTransformer.embed, Block.forward = embed_marked, forward_marked
    wrap(Attention, 'heads', 'qkv')
    wrap(Attention, 'mix', 'attn')
    wrap(Mlp, 'forward', 'mlp')


def profile_forward(unit, iterations=20):
    from torch.profiler import ProfilerActivity, profile
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    forward, info = arms.build('study', unit['size'], unit['dtype'], device, unit['operator'], unit.get('degree', 0),
                               str(Path(unit['root']) / 'weights'), variant=unit.get('variant', 'real'))
    state = info.get('state')
    if state is not None:
        state.operator = _Marked(state.operator)
    instrument()
    images = np.load(Path(unit['root']) / 'bench_images.npy', mmap_mode='r')
    x = data.normalize(images[:unit['batch']], device, arms.DTYPES[unit['dtype']])
    with torch.inference_mode():
        for _ in range(10):
            forward(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iterations):
                forward(x)
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trace.json'
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text())['traceEvents']
    table = attribute(events)
    rows = [{'block': block, 'part': part, 'kernels': value['kernels'] / iterations, 'gpu_us': value['gpu_us'] / iterations,
             'cpu_us': value['cpu_us'] / iterations} for (block, part), value in sorted(table.items())]
    return {'unit': unit, 'iterations': iterations, 'rows': rows, 'environment': measure.environment(device),
            'token_trace': list(state.trace) if state is not None else None}


def summarize(record):
    """Totals per part over all blocks, and the share of each part."""
    totals = defaultdict(lambda: {'kernels': 0.0, 'gpu_us': 0.0, 'cpu_us': 0.0})
    block_cpu = 0.0
    for row in record['rows']:
        if row['part'] == 'block':
            block_cpu += row['cpu_us']
            continue
        for key in ('kernels', 'gpu_us', 'cpu_us'):
            totals[row['part']][key] += row[key]
    # ``rest`` has no range of its own: its CPU time is what the block spends outside the named parts.
    totals['rest']['cpu_us'] = max(0.0, block_cpu - sum(totals[part]['cpu_us'] for part in PARTS))
    whole = {key: sum(value[key] for value in totals.values()) for key in ('kernels', 'gpu_us')}
    cpu_whole = sum(value['cpu_us'] for part, value in totals.items() if part != 'embed-classify')
    return {part: dict(value, kernel_share=value['kernels'] / whole['kernels'] if whole['kernels'] else 0,
                       gpu_share=value['gpu_us'] / whole['gpu_us'] if whole['gpu_us'] else 0,
                       cpu_share_of_blocks=value['cpu_us'] / cpu_whole if cpu_whole and part != 'embed-classify' else 0)
            for part, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', default='fp32')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--operator', default='tome')
    parser.add_argument('--degree', type=int, default=16)
    parser.add_argument('--variant', default='real')
    parser.add_argument('--size', default='tiny')
    parser.add_argument('--root', default=str(weights.CACHE.parent))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    unit = {'size': args.size, 'dtype': args.dtype, 'batch': args.batch, 'operator': args.operator, 'degree': args.degree,
            'variant': args.variant, 'root': args.root}
    record = profile_forward(unit)
    record['summary'] = summarize(record)
    Path(args.out).write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
