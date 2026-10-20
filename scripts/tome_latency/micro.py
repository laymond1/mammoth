"""Microbenchmarks at the shapes of every block of ToMe r=16: the merge operator alone, and attention with and without the size bias.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.micro --out FILE

For each dtype and batch, and each block of the schedule: eager time (one synchronisation per call, median),
CUDA-graph replay time, kernel count and GPU time of the call, and which attention kernel ran. The sum over
the twelve blocks is what the merge operator and the masked attention add to one forward pass.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.token_efficiency import measure
from scripts.token_efficiency.reducers import OPERATORS, Signals, planned, schedule

from .arms import DTYPES

WIDTH, HEADS = 192, 3
SETTINGS = dict(warmup=(10, 1.0), timed=(100, 1.5))


def timed(call, device):
    """Eager and CUDA-graph medians (ms) of ``call()``. No profiler runs here: it would slow the launches of every later call."""
    x = torch.zeros(1, device=device)
    run = lambda _: call()
    with torch.inference_mode():
        measure.run_for(run, x, device, *SETTINGS['warmup'])
        eager = measure.summary(measure.run_for(run, x, device, *SETTINGS['timed']))
        graph = measure.capture_graph(run, x)
        replay = lambda _: graph.replay()
        measure.run_for(replay, x, device, 10, 0.2)
        replayed = measure.summary(measure.run_for(replay, x, device, *SETTINGS['timed']))
    return {'eager_ms': eager['median_ms'], 'graph_ms': replayed['median_ms']}


def profiled(call, device):
    """Kernel count, GPU time and attention kernels of one ``call()``; run only after every timing is done."""
    x = torch.zeros(1, device=device)
    with torch.inference_mode():
        profile = measure.kernel_profile(lambda _: call(), x)
    return {'kernels': profile['kernels'], 'gpu_us': profile['gpu_us'], 'attention_kernels': profile['attention_kernels']}


def merge_case(operator, tokens, batch, dtype, device, block, depth, generator):
    count = planned(16, tokens)
    x = torch.randn(batch, tokens, WIDTH, device=device, dtype=dtype, generator=generator)
    q = torch.randn(batch, HEADS, tokens, WIDTH // HEADS, device=device, dtype=dtype, generator=generator)
    k = torch.randn(batch, HEADS, tokens, WIDTH // HEADS, device=device, dtype=dtype, generator=generator)
    size = torch.ones(batch, tokens, 1, device=device, dtype=dtype) if block else None
    signals = Signals(q, k, (WIDTH // HEADS) ** -0.5, None, block=block, depth=depth)
    return lambda: operator(x, size, count, signals)


def attention_case(tokens, batch, dtype, device, masked, generator):
    q, k, v = (torch.randn(batch, HEADS, tokens, WIDTH // HEADS, device=device, dtype=dtype, generator=generator) for _ in range(3))
    bias = torch.zeros(batch, 1, 1, tokens, device=device, dtype=dtype) if masked else None
    return lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=(WIDTH // HEADS) ** -0.5)


def block_calls(operator, trace, batch, dtype, device, generator):
    for block, tokens in enumerate(trace):
        yield block, tokens, {
            'merge': merge_case(OPERATORS[operator](), tokens, batch, dtype, device, block, len(trace), generator),
            'attention': attention_case(tokens, batch, dtype, device, False, generator),
            'attention_masked': attention_case(tokens, batch, dtype, device, True, generator)}


def run(dtypes, batches, degree=16, operator='tome'):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    trace = schedule(degree, 197, 12)
    result = {'degree': degree, 'operator': operator, 'token_trace': trace, 'cases': [], 'environment': measure.environment(device)}
    for phase in ('timing', 'profile'):
        generator = torch.Generator(device=device).manual_seed(0)
        for dtype_name in dtypes:
            dtype = DTYPES[dtype_name]
            for batch in batches:
                if phase == 'timing':
                    entry = {'dtype': dtype_name, 'batch': batch, 'blocks': []}
                    result['cases'].append(entry)
                else:
                    entry = next(c for c in result['cases'] if (c['dtype'], c['batch']) == (dtype_name, batch))
                for block, tokens, calls in block_calls(operator, trace, batch, dtype, device, generator):
                    if phase == 'timing':
                        entry['blocks'].append({'block': block, 'tokens': tokens, **{key: timed(call, device) for key, call in calls.items()}})
                    else:
                        for key, call in calls.items():
                            entry['blocks'][block][key].update(profiled(call, device))
    for entry in result['cases']:
        for key in ('merge', 'attention', 'attention_masked'):
            fields = ('eager_ms', 'graph_ms', 'kernels', 'gpu_us')
            entry[f'{key}_total'] = {field: sum(b[key][field] for b in entry['blocks']) for field in fields}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtypes', nargs='+', default=['fp32', 'fp16'])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 16, 64])
    parser.add_argument('--operator', default='tome')
    parser.add_argument('--degree', type=int, default=16)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(run(args.dtypes, args.batches, args.degree, args.operator), indent=1))


if __name__ == '__main__':
    main()
