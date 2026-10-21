"""Build TensorRT engines from the static ONNX files of ``trt_export`` (stage 3).

    python -m scripts.tome_latency.trt_build --onnx DIR --out DIR [--precisions fp32 fp16]

fp32 engines turn TF32 off, so they compute like the PyTorch eager runs (TF32 off); fp16 engines set the FP16
builder flag on the same fp32 ONNX. Parser errors, the TensorRT version, the flags and the build time are written to
``build_report.json`` next to the plans. Engines are tied to the device and the TensorRT version that built them.
"""
import argparse
import json
import time
from pathlib import Path


def make_logger(trt, messages):
    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, message):
            if severity <= trt.ILogger.WARNING:
                messages.append(f'{str(severity).split(".")[-1]}: {message}')
    return Logger()


def build_engine(onnx_path, plan_path, precision='fp32', workspace_gb=4):
    """Parse and build one engine; returns a record with ``ok`` and the reasons when it is not."""
    import tensorrt as trt
    messages = []
    logger = make_logger(trt, messages)
    # ONNX ScatterElements with a reduction is imported as the ScatterReduction plugin, which lives in libnvinfer_plugin.
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    record = {'onnx': Path(onnx_path).name, 'precision': precision, 'tensorrt': trt.__version__}
    if not parser.parse_from_file(str(onnx_path)):
        record.update(ok=False, stage='parse', errors=[str(parser.get_error(i)) for i in range(parser.num_errors)], messages=messages[-20:])
        return record
    layers = {}
    for index in range(network.num_layers):
        kind = str(network.get_layer(index).type).split('.')[-1]
        layers[kind] = layers.get(kind, 0) + 1
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    record['flags'] = {'fp16': precision == 'fp16', 'tf32': False, 'workspace_gb': workspace_gb, 'optimization_level': config.builder_optimization_level}
    record['network_layers'] = {'total': network.num_layers, 'by_type': layers}
    start = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    record['build_seconds'] = time.perf_counter() - start
    if blob is None:
        record.update(ok=False, stage='build', messages=messages[-20:])
        return record
    Path(plan_path).write_bytes(bytes(blob))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(blob))
    record.update(ok=True, plan=Path(plan_path).name, plan_mb=len(bytes(blob)) / 2 ** 20, engine_layers=engine.num_layers, messages=messages[-20:])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--precisions', nargs='+', default=['fp32', 'fp16'])
    parser.add_argument('--only', help='substring of the ONNX file name')
    parser.add_argument('--workspace-gb', type=int, default=4)
    args = parser.parse_args()
    import tensorrt as trt
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'build_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'tensorrt': trt.__version__, 'builds': {}}
    for onnx in sorted(Path(args.onnx).glob('*.onnx')):
        if args.only and args.only not in onnx.name:
            continue
        for precision in args.precisions:
            key = f'{onnx.stem}_{precision}'
            plan = out / f'{key}.plan'
            if plan.exists() and report['builds'].get(key, {}).get('ok'):
                continue
            report['builds'][key] = build_engine(onnx, plan, precision, args.workspace_gb)
            state = report['builds'][key]
            print(key, 'ok' if state['ok'] else f"FAILED at {state['stage']}", f"{state.get('build_seconds', 0):.0f}s", flush=True)
            report_path.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
