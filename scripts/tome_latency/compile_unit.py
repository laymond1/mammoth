"""Stage 2 unit: one (dtype, degree, batch) of the study ViT in every execution mode, each checked against eager.

    python -m scripts.tome_latency.compile_unit --unit '{...}' --out FILE      (called by ``stage2``)

Modes: ``eager``; ``cuda-graph`` (manual capture of the whole forward); ``compile-default``,
``compile-reduce-overhead`` (inductor plus CUDA graphs) and ``compile-max-autotune-no-cudagraphs``
(``torch.compile`` with static shapes). A mode that fails is recorded with its error and the others go on.
The inductor and Triton caches live in a fresh folder, so compile times are cold-start times.
"""
import os
import shutil
import tempfile

_CACHE = tempfile.mkdtemp(prefix='tome-latency-cache-')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(_CACHE, 'inductor')
os.environ['TRITON_CACHE_DIR'] = os.path.join(_CACHE, 'triton')

import argparse  # noqa: E402
import atexit  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.token_efficiency import data, measure  # noqa: E402

from . import arms  # noqa: E402
from .run_unit import code_state, cpu_clocks  # noqa: E402

atexit.register(shutil.rmtree, _CACHE, ignore_errors=True)
TOLERANCE = {'fp32': 1e-3, 'fp16': 5e-2}
MODES = ('eager', 'cuda-graph', 'compile-default', 'compile-reduce-overhead', 'compile-max-autotune-no-cudagraphs')


def unit_id(unit):
    name = 'dense' if unit['degree'] == 0 else f"tome-r{unit['degree']}"
    return f"compile__{unit['dtype']}__{name}__b{unit['batch']}"


def chunks(images, device, dtype, batch, count=256):
    return [data.normalize(images[start:start + batch], device, dtype) for start in range(0, count, batch)]


def agreement(reference, outputs, dtype):
    """How far a mode's logits are from eager's over the same images: worst, mean, share beyond tolerance, top-1."""
    diff = (torch.cat(outputs).float() - torch.cat(reference).float()).abs()
    worst = diff.amax(dim=1)
    same_top1 = (torch.cat(outputs).argmax(1) == torch.cat(reference).argmax(1)).float().mean().item()
    return {'images': len(worst), 'max_abs_diff': worst.max().item(), 'mean_max_abs_diff': worst.mean().item(),
            'share_beyond_tolerance': (worst > TOLERANCE[dtype]).float().mean().item(), 'tolerance': TOLERANCE[dtype],
            'top1_agreement': same_top1}


def time_calls(call, x, device, settings):
    measure.run_for(call, x, device, settings['warmup_iters'], settings['warmup_seconds'])
    return measure.summary(measure.run_for(call, x, device, settings['min_iters'], settings['min_seconds']))


def graphed(forward, x):
    """Capture the forward once; returns the graph and the static output it writes."""
    static = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward(static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward(static)
    return graph, static, output


def run_mode(mode, forward, inputs, reference, unit, device):
    x = inputs[0]
    settings = unit['settings']
    record = {'mode': mode}
    with torch.inference_mode():
        if mode == 'eager':
            call, outputs = forward, [forward(chunk).clone() for chunk in inputs]
        elif mode == 'cuda-graph':
            graph, static, output = graphed(forward, x)
            outputs = []
            for chunk in inputs:
                static.copy_(chunk)
                graph.replay()
                outputs.append(output.clone())
            record['bitwise_equal_to_eager'] = all(torch.equal(a, b) for a, b in zip(outputs, reference))
            call = lambda _: graph.replay()
        else:
            options = {'mode': {'compile-default': 'default', 'compile-reduce-overhead': 'reduce-overhead',
                                'compile-max-autotune-no-cudagraphs': 'max-autotune-no-cudagraphs'}[mode]}
            torch._dynamo.reset()
            compiled = torch.compile(forward, dynamic=False, fullgraph=False, **options)
            mark = torch.compiler.cudagraph_mark_step_begin if mode == 'compile-reduce-overhead' else (lambda: None)

            def call(images):
                mark()
                return compiled(images)
            start = time.perf_counter()
            call(x)
            torch.cuda.synchronize()
            record['compile_seconds'] = time.perf_counter() - start
            outputs = [call(chunk).clone() for chunk in inputs]
        record['agreement'] = agreement(reference, outputs, unit['dtype'])
        record['timing'] = time_calls(lambda images: call(images), x, device, settings)
        try:
            record['profile'] = {key: value for key, value in measure.kernel_profile(lambda images: call(images), x).items()
                                 if key in ('kernels', 'gpu_us')}
        except Exception as error:  # the profiler cannot always see inside a replayed graph
            record['profile'] = {'error': str(error)[:200]}
    return record


def run_unit(unit, out):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dtype = arms.DTYPES[unit['dtype']]
    base = {'unit': unit, 'unit_id': unit_id(unit), 'code': code_state(), 'environment': measure.environment(device),
            'state_before': measure.device_state(device), 'cpu_clocks_before': cpu_clocks(),
            'env': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'TORCHINDUCTOR_CACHE_DIR')}}
    forward, info = arms.build('study', unit['size'], unit['dtype'], device, 'dense' if unit['degree'] == 0 else 'tome',
                               unit['degree'], unit['weights'])
    inputs = chunks(np.load(unit['images'], mmap_mode='r'), device, dtype, unit['batch'])
    with torch.inference_mode():
        reference = [forward(chunk).clone() for chunk in inputs]
    base['token_trace'] = list(info['state'].trace) if info.get('state') is not None else None
    base['modes'] = []
    for mode in unit['modes']:
        try:
            base['modes'].append(run_mode(mode, forward, inputs, reference, unit, device))
        except Exception as error:
            base['modes'].append({'mode': mode, 'error': f'{type(error).__name__}: {error}'[:600], 'trace': traceback.format_exc()[-1500:]})
        torch.cuda.empty_cache()
    base['state_after'] = measure.device_state(device)
    Path(out).write_text(json.dumps(base, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run_unit(json.loads(args.unit), args.out)


if __name__ == '__main__':
    main()
