"""Static ONNX export of the study ViT-Ti with ToMe at one degree and batch (stage 3).

    python -m scripts.tome_latency.trt_export --out DIR [--degrees 0 8 16 24] [--batches 1 16]

Every file has fixed shapes: the token schedule of a degree is static, so each degree and batch is its own model.
The ONNX holds fp32 weights; fp16 is a builder option of the engine. The token schedule and the operator are not
changed.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from scripts.token_efficiency import weights
from scripts.token_efficiency.vit import Reduction, build


class Export(nn.Module):
    """The model at one degree with the reduction state created inside ``forward``, so tracing sees no outside state."""

    def __init__(self, model, operator, degree):
        super().__init__()
        self.model, self.operator, self.degree = model, operator, degree

    def forward(self, images):
        if self.degree == 0:
            return self.model(images)
        return self.model(images, Reduction(self.operator, self.degree))


def onnx_name(operator, degree, batch):
    return f"vit-tiny_{'dense' if degree == 0 else f'{operator}-r{degree}'}_b{batch}.onnx"


def export_one(path, operator, degree, batch, root, opset=17, legacy=True):
    model = build('tiny', weights.load('tiny', Path(root) / 'weights'), 'cpu', torch.float32)
    module = Export(model, operator, degree).eval()
    example = torch.zeros(batch, 3, 224, 224)
    kwargs = {'input_names': ['images'], 'output_names': ['logits'], 'opset_version': opset}
    with torch.inference_mode():
        if legacy:
            torch.onnx.export(module, (example,), str(path), dynamo=False, **kwargs)
        else:
            torch.onnx.export(module, (example,), str(path), dynamo=True, **kwargs)


def inspect(path):
    """Operator histogram of the file, and whether the checker accepts it."""
    import onnx
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return {'ops': dict(Counter(node.op_type for node in model.graph.node)), 'opset': [o.version for o in model.opset_import if o.domain == ''][0],
            'nodes': len(model.graph.node), 'size_mb': Path(path).stat().st_size / 2 ** 20}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default=str(weights.CACHE.parent))
    parser.add_argument('--operator', default='tome')
    parser.add_argument('--degrees', nargs='+', type=int, default=[0, 8, 16, 24])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 16])
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--dynamo', action='store_true', help='use the dynamo exporter instead of the TorchScript one')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {'torch': torch.__version__, 'opset': args.opset, 'exporter': 'dynamo' if args.dynamo else 'torchscript', 'files': {}}
    for degree in args.degrees:
        for batch in args.batches:
            path = out / onnx_name(args.operator, degree, batch)
            try:
                export_one(path, args.operator, degree, batch, args.root, args.opset, legacy=not args.dynamo)
                report['files'][path.name] = inspect(path)
                print(path.name, report['files'][path.name]['nodes'], 'nodes', flush=True)
            except Exception as error:
                report['files'][path.name] = {'error': f'{type(error).__name__}: {error}'[:800]}
                print(path.name, 'FAILED', str(error)[:200], flush=True)
    (out / 'export_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
