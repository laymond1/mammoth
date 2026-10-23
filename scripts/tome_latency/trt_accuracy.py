"""ImageNet top-1 of the TensorRT engines next to PyTorch eager, on the 50,000 cached validation crops (stage 3).

    python -m scripts.tome_latency.trt_accuracy --plans DIR --out FILE [--val DIR] [--limit N]

Every model sees each batch of 16 once: the engines at degree 0 and 16 (fp32 and fp16), and PyTorch eager at both
precisions, with the fp16 ToMe model built twice. The two fp16 eager copies show how much top-1 moves from run to run
with no engine involved; an fp16 engine is judged against that spread. Agreement is the share of images where a model
predicts the same class as the fp32 eager ToMe model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.token_efficiency import data, weights

from . import arms
from .run_unit import code_state
from .trt_run import Engine

DEGREE = 16
BATCH = 16


def models_for(plans, device, root):
    models = {}
    for precision in ('fp32', 'fp16'):
        for degree in (0, DEGREE):
            name = 'dense' if degree == 0 else f'tome-r{degree}'
            plan = Path(plans) / f'vit-tiny_{name}_b{BATCH}_{precision}.plan'
            if plan.exists():
                engine = Engine(plan, device)
                models[f'trt-{precision}:{name}'] = lambda x, engine=engine: engine(x)
    for dtype in ('fp32', 'fp16'):
        for degree in (0, DEGREE):
            name = 'dense' if degree == 0 else f'tome-r{degree}'
            for copy in ((1, 2) if dtype == 'fp16' and degree else (1,)):
                forward, _ = arms.build('study', 'tiny', dtype, device, 'dense' if degree == 0 else 'tome', degree, str(Path(root) / 'weights'))
                models[f"eager-{dtype}:{name}{'' if copy == 1 else ' (second copy)'}"] = lambda x, f=forward, d=arms.DTYPES[dtype]: f(x.to(d))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plans', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default=str(weights.CACHE.parent))
    parser.add_argument('--val', default=None, help='folder with images.npy and labels.npy (default: <root>/imagenet-val)')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    val = Path(args.val or Path(args.root) / 'imagenet-val')
    images = np.load(val / 'images.npy', mmap_mode='r')
    labels = np.load(val / 'labels.npy').astype(np.int64)
    count = min(len(images), args.limit or len(images))
    models = models_for(args.plans, device, args.root)
    predictions = {name: np.empty(count, dtype=np.int16) for name in models}
    with torch.inference_mode():
        for start in range(0, count - count % BATCH, BATCH):
            x = data.normalize(images[start:start + BATCH], device, torch.float32)
            for name, model in models.items():
                predictions[name][start:start + BATCH] = model(x).argmax(1).cpu().numpy()
    done = count - count % BATCH
    reference = predictions['eager-fp32:tome-r16'][:done]
    rows = {name: {'top1': float((p[:done] == labels[:done]).mean()), 'agreement_with_eager_fp32_tome': float((p[:done] == reference).mean())}
            for name, p in predictions.items()}
    Path(args.out).write_text(json.dumps({'code': code_state(), 'images': done, 'batch': BATCH, 'degree': DEGREE, 'rows': rows}, indent=1))
    print(f'{done} images')
    for name, row in rows.items():
        print(f"  {name:<34} top-1 {100 * row['top1']:.2f}%   same class as eager fp32 ToMe: {100 * row['agreement_with_eager_fp32_tome']:.2f}%")


if __name__ == '__main__':
    main()
