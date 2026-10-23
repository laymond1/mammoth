"""Known-answer check of token reduction and a ToCom bank on ImageNet-1K validation.

Without ``--bank`` it reports top-1 of the pretrained classifier at every
degree, i.e. the compression gap of the backbone under one operator. With a
bank it also reports, per (source, target) pair, how far the compensated
model at ``target`` moves back to the same model at ``source``: accuracy,
KL to the source-degree output and top-1 agreement. A working bank lowers the
KL in every pair. Read-only apart from the optional ``--output`` file.

With ``--shapes R:D ...`` it reports top-1 at those input shapes instead (the 224-pixel centre crop resized to
R pixels, then D tokens removed per block with ``--operator``), without a bank. ``--latency B ...`` adds the
median time of one forward pass at every shape and batch size B, on 224-pixel inputs (the resize included).

Run from the repository root:
    uv run --no-sync python -m scripts.tocom.imagenet_check --weights augreg --operator tome
"""

from scripts.cl_runtime import DEFAULT_PROFILE, configure_native_environment

configure_native_environment()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402

from backbone.token_compression import (OPERATORS, FrozenToCom, attach, compensate, parse_shape,  # noqa: E402
                                        shape_cost, shape_label)
from backbone.token_compression.reference import augreg_reference, deit_reference  # noqa: E402

DEGREES = (0, 4, 8, 12, 16)
SOURCES = (0, 8, 16)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', choices=('augreg', 'deit'), required=True)
    parser.add_argument('--vit-size', choices=('tiny', 'small', 'base'), default='tiny', help='AugReg size')
    parser.add_argument('--pretrained', help='DeiT .pth file, or an AugReg .npz overriding the pinned path')
    parser.add_argument('--operator', choices=tuple(OPERATORS), default='tome')
    parser.add_argument('--bank', help='ToCom checkpoint trained for these weights and this operator')
    parser.add_argument('--scale', type=float, default=.1)
    parser.add_argument('--degrees', type=int, nargs='+', default=list(DEGREES), help='Inference degrees')
    parser.add_argument('--sources', type=int, nargs='+', default=list(SOURCES),
                        help='With a bank: degrees the compensated model is pulled back to')
    parser.add_argument('--shapes', nargs='+', default=[], metavar='R:D',
                        help='Report top-1 at these input shapes (resolution:degree) instead, without a bank')
    parser.add_argument('--latency', type=int, nargs='+', default=[], metavar='BATCH',
                        help='With --shapes: also time one forward pass per shape at these batch sizes')
    parser.add_argument('--data', help='ImageNet root containing val/<synset>/; default from the runtime profile')
    parser.add_argument('--stride', type=int, default=1, help='Evaluate every n-th validation image')
    parser.add_argument('--batch-size', type=int, default=250)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--output', type=Path, help='Write the JSON result here')
    args = parser.parse_args(argv)
    if args.weights == 'deit' and not args.pretrained:
        parser.error('--weights deit requires --pretrained')
    if min(args.stride, args.batch_size) < 1 or args.workers < 0:
        parser.error('stride and batch size must be positive')
    args.sources = sorted(set(args.sources))
    args.degrees = sorted({*args.degrees, *args.sources})
    try:
        args.shapes = [parse_shape(text) for text in args.shapes]
    except ValueError as error:
        parser.error(str(error))
    if args.shapes and args.bank:
        parser.error('--shapes is evaluated without a bank')
    if args.latency and (not args.shapes or min(args.latency) < 1):
        parser.error('--latency takes positive batch sizes and goes with --shapes')
    return args


def kl(student, teacher):
    return F.kl_div(F.log_softmax(student, dim=1), F.log_softmax(teacher, dim=1),
                    reduction='sum', log_target=True).item()


@torch.no_grad()
def check(model, loader, controller, degrees, pairs, scale, device):
    correct = {degree: 0 for degree in degrees}
    stats = {pair: dict(correct=0, kl_off=0., kl_on=0., agree_off=0, agree_on=0) for pair in pairs}
    total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        plain = {}
        for degree in degrees:
            with controller.degree(degree):
                plain[degree] = model(images).float()
                controller.verify_pass()
            correct[degree] += int((plain[degree].argmax(1) == labels).sum())
        for source, target in pairs:
            with compensate(model, source, target, scale):
                corrected = model(images).float()
                controller.verify_pass()
            entry, reference = stats[source, target], plain[source].argmax(1)
            entry['correct'] += int((corrected.argmax(1) == labels).sum())
            entry['kl_off'] += kl(plain[target], plain[source])
            entry['kl_on'] += kl(corrected, plain[source])
            entry['agree_off'] += int((plain[target].argmax(1) == reference).sum())
            entry['agree_on'] += int((corrected.argmax(1) == reference).sum())
        total += len(labels)
    accuracy = {degree: 100 * correct[degree] / total for degree in degrees}
    rows = [{'source': source, 'target': target, 'acc_source': accuracy[source], 'acc_off': accuracy[target],
             'acc_on': 100 * entry['correct'] / total, 'kl_off': entry['kl_off'] / total,
             'kl_on': entry['kl_on'] / total, 'agree_off': 100 * entry['agree_off'] / total,
             'agree_on': 100 * entry['agree_on'] / total}
            for (source, target), entry in stats.items()]
    return total, accuracy, rows


@torch.no_grad()
def check_shapes(model, loader, controller, shapes, device):
    correct, total = {shape: 0 for shape in shapes}, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        for shape in shapes:
            with controller.shape(*shape):
                outputs = model(images)
                controller.verify_pass()
            correct[shape] += int((outputs.argmax(1) == labels).sum())
        total += len(labels)
    return total, {shape_label(*shape): 100 * correct[shape] / total for shape in shapes}


@torch.no_grad()
def shape_latency(model, controller, shapes, batches, device, size=224, rounds=5, repeats=40, warmup=10):
    """Median milliseconds of one forward pass per shape and batch size, on ``size``-pixel inputs (the resize to
    the shape's resolution included). Each pass is timed alone, from a synchronized device to a synchronized
    device. The shapes take turns over several rounds, so that a slow drift of the device favours none of them."""
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    timings = {}
    for batch in batches:
        images = torch.randn(batch, 3, size, size, generator=torch.Generator().manual_seed(0)).to(device)
        samples = {shape: [] for shape in shapes}
        for _ in range(rounds):
            for shape in shapes:
                with controller.shape(*shape):
                    for index in range(warmup + repeats):
                        synchronize()
                        started = time.perf_counter()
                        model(images)
                        synchronize()
                        if index >= warmup:
                            samples[shape].append((time.perf_counter() - started) * 1000)
                    controller.verify_pass()
        for shape, values in samples.items():
            timings.setdefault(shape_label(*shape), {})[str(batch)] = statistics.median(values)
    return timings


def main(argv=None):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = (augreg_reference(args.vit_size, args.pretrained) if args.weights == 'augreg'
             else deit_reference(args.pretrained))
    bank = None
    if args.bank:
        bank = FrozenToCom.from_checkpoint(args.bank, len(model.blocks), model.embed_dim)
    controller = attach(model, args.operator, bank=bank)
    model.to(device)

    data = Path(args.data or Path(json.loads(DEFAULT_PROFILE.read_text())['data_root']) / 'imagenet')
    mean, std = model.normalization
    dataset = ImageFolder(str(data / 'val'), transform=transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean, std)]))
    if len(dataset.classes) != 1000 or len(dataset) != 50000:
        raise ValueError(f'Expected the 50,000-image ImageNet-1K validation set under {data / "val"}')
    loader = DataLoader(Subset(dataset, range(0, len(dataset), args.stride)), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    pairs = [(source, target) for source in args.sources for target in args.degrees if source != target] if bank else []

    started = time.time()
    if args.shapes:
        total, by_shape = check_shapes(model, loader, controller, args.shapes, device)
        costs = {shape_label(*shape): shape_cost(*shape, model.embed_dim, len(model.blocks)) for shape in args.shapes}
        result = {'weights': args.weights, 'vit_size': args.vit_size if args.weights == 'augreg' else None,
                  'backbone_canonical_sha256': model.canonical_pretrained_sha256, 'operator': args.operator,
                  'images': total, 'seconds': round(time.time() - started, 1), 'torch': torch.__version__,
                  'accuracy_by_shape': by_shape, 'cost_by_shape': costs}
        if args.latency:
            result['device'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
            result['latency_ms'] = shape_latency(model, controller, args.shapes, args.latency, device)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'{args.weights} {args.operator}: images={total} seconds={result["seconds"]}')
        for label, value in by_shape.items():
            times = ''.join(f'  b{batch} {ms:.2f} ms' for batch, ms in result.get('latency_ms', {}).get(label, {}).items())
            print(f'  {label:>7}  cost {costs[label]:.3f}  top-1 {value:.2f}{times}')
        return result
    total, accuracy, rows = check(model, loader, controller, args.degrees, pairs, args.scale, device)
    result = {'weights': args.weights, 'vit_size': args.vit_size if args.weights == 'augreg' else None,
              'backbone_canonical_sha256': model.canonical_pretrained_sha256, 'operator': args.operator,
              'bank_sha256': bank.checkpoint_hash if bank else None, 'scale': args.scale if bank else None,
              'images': total, 'seconds': round(time.time() - started, 1), 'torch': torch.__version__,
              'accuracy_by_degree': accuracy, 'pairs': rows}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    print(f'{args.weights} {args.operator}: images={total} seconds={result["seconds"]}')
    print('top-1 by degree:', {degree: round(value, 2) for degree, value in accuracy.items()})
    if rows:
        print(f'{"src":>3} {"tgt":>3} | {"acc@src":>7} {"off":>6} {"on":>6} {"gain":>6} | '
              f'{"KL off":>7} {"KL on":>7} {"cut":>6} | {"agree off":>9} {"agree on":>8}')
        for row in rows:
            cut = 100 * (1 - row['kl_on'] / row['kl_off']) if row['kl_off'] else float('nan')
            print(f'{row["source"]:>3} {row["target"]:>3} | {row["acc_source"]:>7.2f} {row["acc_off"]:>6.2f} '
                  f'{row["acc_on"]:>6.2f} {row["acc_on"] - row["acc_off"]:>+6.2f} | {row["kl_off"]:>7.4f} '
                  f'{row["kl_on"]:>7.4f} {cut:>5.1f}% | {row["agree_off"]:>9.2f} {row["agree_on"]:>8.2f}')
    return result


if __name__ == '__main__':
    main()
