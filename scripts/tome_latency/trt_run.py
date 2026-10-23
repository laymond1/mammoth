"""Time TensorRT engines and check their output against PyTorch eager (stage 3).

    CUDA_VISIBLE_DEVICES=1 python -m scripts.tome_latency.trt_run --plans DIR --out FILE [--only SUBSTRING]

An engine call is timed like an eager call: inputs already on the GPU, one device synchronisation per call, median of the
timed calls (warm-up 10 calls and 2 s, timed 30 calls and 3 s). The same 256 benchmark images then go through the
engine and through the PyTorch eager model of the same degree, in fp32 and in fp16, to see how far the logits move.
The batch is fixed by the engine, so the images go in chunks of that size.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from scripts.token_efficiency import data, measure, weights

from . import arms
from .compile_unit import agreement
from .run_unit import code_state

PLAN = re.compile(r'vit-tiny_(?P<name>dense|[a-z-]+-r(?P<degree>\d+))_b(?P<batch>\d+)_(?P<precision>fp32|fp16)\.plan')
SETTINGS = (10, 2.0, 30, 3.0)


class Engine:
    """A deserialised engine with its input and output bound to preallocated CUDA tensors."""

    def __init__(self, plan_path, device):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')  # engines with the ScatterReduction plugin need the registry at load time
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.tensors = {}
        for name in names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}[self.engine.get_tensor_dtype(name)]
            self.tensors[name] = torch.empty(shape, dtype=dtype, device=device)
            self.context.set_tensor_address(name, self.tensors[name].data_ptr())
        self.input = next(self.tensors[n] for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output = next(self.tensors[n] for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    def enqueue(self):
        # A non-default stream: on the default stream TensorRT adds a synchronisation to every call.
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')

    def __call__(self, images):
        """Copy in, run, and leave the result in ``self.output`` once the default stream has waited for the engine's."""
        self.input.copy_(images)
        self.stream.wait_stream(torch.cuda.current_stream())
        self.enqueue()
        torch.cuda.current_stream().wait_stream(self.stream)
        return self.output


def chunks(images, device, batch, count=256):
    return [data.normalize(images[start:start + batch], device, torch.float32) for start in range(0, count, batch)]


def eager_reference(operator, degree, dtype, device, root, inputs):
    forward, _ = arms.build('study', 'tiny', dtype, device, 'dense' if degree == 0 else operator, degree, str(Path(root) / 'weights'))
    with torch.inference_mode():
        return [forward(chunk.to(arms.DTYPES[dtype])).clone() for chunk in inputs]


def bench_plan(plan, root, device):
    match = PLAN.fullmatch(plan.name)
    degree, batch, precision = int(match['degree'] or 0), int(match['batch']), match['precision']
    operator = match['name'].rsplit('-r', 1)[0] if degree else 'dense'
    engine = Engine(plan, device)
    images = np.load(Path(root) / 'bench_images.npy', mmap_mode='r')
    inputs = chunks(images, device, batch)
    record = {'plan': plan.name, 'operator': operator, 'degree': degree, 'batch': batch, 'precision': precision,
              'tensorrt': engine.trt.__version__, 'engine_layers': engine.engine.num_layers}
    call = lambda _: engine.enqueue()
    with torch.inference_mode():
        engine.input.copy_(inputs[0])
        torch.cuda.synchronize(device)
        measure.run_for(call, inputs[0], device, SETTINGS[0], SETTINGS[1])
        record['timing'] = measure.summary(measure.run_for(call, inputs[0], device, SETTINGS[2], SETTINGS[3]))
        outputs = [engine(chunk).clone() for chunk in inputs]
    for dtype in ('fp32', 'fp16'):
        reference = eager_reference(operator, degree, dtype, device, root, inputs)
        # The fp32 tolerance only applies when both sides are fp32; any fp16 side moves logits by fp16 rounding.
        record[f'vs_eager_{dtype}'] = agreement(reference, outputs, 'fp32' if precision == dtype == 'fp32' else 'fp16')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plans', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default=str(weights.CACHE.parent))
    parser.add_argument('--only')
    args = parser.parse_args()
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    result = json.loads(out.read_text()) if out.exists() else {'code': code_state(), 'environment': measure.environment(device), 'records': {}}
    for plan in sorted(Path(args.plans).glob('*.plan')):
        if args.only and args.only not in plan.name or plan.stem in result['records']:
            continue
        record = bench_plan(plan, args.root, device)
        record['state_after'] = measure.device_state(device)
        result['records'][plan.stem] = record
        print(plan.stem, f"{record['timing']['median_ms']:.3f} ms", flush=True)
        out.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
