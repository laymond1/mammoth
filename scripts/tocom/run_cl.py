"""Run one continual-learning method with token compression and evaluate the ToCom grid.

The method code is untouched: the launcher builds the model through Mammoth's
own entry points, patches the encoder blocks, and wraps ``meta_end_eval`` so
that after every task the model is also evaluated at each inference degree,
with and without the compensator bank of each operator.

Example (from the repository root):
    uv run --no-sync python -m scripts.tocom.run_cl --method er --operator tome --degree 8 \\
        --banks tome=<bank dir> cls-prune=<bank dir> --seed 1993 --output results/tocom-cl-pilot
"""

from scripts.cl_runtime import DEFAULT_PROFILE, configure_native_environment

configure_native_environment()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import main as mammoth  # noqa: E402
from backbone.token_compression import OPERATORS, FrozenToCom, attach, find_encoder  # noqa: E402
from backbone.token_compression.compensator import OFFICIAL_COMMIT, file_sha256  # noqa: E402
from backbone.token_compression.evaluation import evaluate_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Method recipes of the unified Tiny comparison (scripts/unified_cl_protocol.py).
COMMON = dict(dataset='seq-cifar100-224', dataset_config='unified', model_config='default', num_workers=4,
              drop_last=0, permute_classes=0, fitting_mode='epochs', scheduler_mode='epoch',
              enable_other_metrics=0, eval_future=0, joint=0, non_verbose=1, code_optimization=0)


def method_args(method, size, checkpoint, batch_size):
    if method == 'er':
        return dict(model='er', backbone='vit-augreg', vit_size=size, pretrained=1, vit_checkpoint=checkpoint,
                    optimizer='adamw', lr=0.0001, optim_wd=0.0, buffer_size=500, minibatch_size=batch_size)
    if method == 'coda-prompt':
        return dict(model='coda-prompt', backbone='vit-augreg', vit_size=size, pretrained=0, coda_vit_size=size,
                    coda_checkpoint=checkpoint, optimizer='adam', lr=0.001, optim_wd=0.0, pool_size=100,
                    prompt_len=8, mu=0.0, ortho_mu=0.0, virtual_bs_iterations=1)
    if method == 'sd-lora':
        # LoRA learning rate accepted for Base in the unified study; other sizes need a stability check.
        return dict(model='sd-lora', backbone='low-rank-vit', lora_vit_size=size, lora_checkpoint=checkpoint,
                    optimizer='adam', lr=0.001, optim_wd=0.0, lr_scheduler='cosine', lora_rank=10, lora_fc_lr=0.008)
    raise ValueError(f'No recipe for {method}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=('er', 'coda-prompt', 'sd-lora'), required=True)
    parser.add_argument('--vit-size', choices=('tiny', 'small', 'base'), default='tiny')
    parser.add_argument('--operator', choices=tuple(OPERATORS), default='tome', help='Reduction used in training')
    parser.add_argument('--degree', type=int, default=0, help='Training degree: tokens removed per block')
    parser.add_argument('--targets', type=int, nargs='+', default=[0, 4, 8, 12, 16], help='Inference degrees')
    parser.add_argument('--eval-operators', nargs='+', choices=tuple(OPERATORS),
                        help='Operators to infer with; default: all at degree 0, else the training operator')
    parser.add_argument('--banks', nargs='*', default=[], metavar='OPERATOR=DIR',
                        help='Bank directories holding checkpoint.pth and provenance.json')
    parser.add_argument('--allow-partial-banks', action='store_true', help='Accept unfinished banks (smoke runs)')
    parser.add_argument('--no-grid', action='store_true', help='Train only; used to prove the grid has no side effects')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic algorithms: reproducible weights')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--tasks', type=int, help='Stop after this many tasks (default: all)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True, help='Experiment directory; one sub-directory per run')
    parser.add_argument('--run-id', help='Default: <method>-<operator>-r<degree>-seed<seed>')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='After --: extra Mammoth arguments')
    args = parser.parse_args(argv)
    args.extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    if args.degree not in args.targets:
        parser.error('Inference degrees must include the training degree')
    if args.eval_operators is None:
        args.eval_operators = sorted(OPERATORS) if args.degree == 0 else [args.operator]
    elif args.degree and args.eval_operators != [args.operator]:
        parser.error('A model trained with reduction is inferred with its own operator')
    return args


def state_digest(net):
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_banks(specifications, digest, depth, width, allow_partial):
    """Banks bound to these pretrained weights and to the operator they are named for."""
    banks, records, scales = {}, {}, set()
    for specification in specifications:
        name, _, directory = specification.partition('=')
        directory = Path(directory)
        if name not in OPERATORS or name in banks or not directory.is_dir():
            raise ValueError(f'Bank must be OPERATOR=DIR with a distinct known operator: {specification}')
        manifest = json.loads((directory / 'provenance.json').read_text())
        checkpoint = directory / 'checkpoint.pth'
        if manifest.get('operator') != name:
            raise ValueError(f'Bank in {directory} was trained for {manifest.get("operator")}, not {name}')
        if manifest.get('backbone', {}).get('canonical_sha256') != digest:
            raise ValueError(f'Bank in {directory} was trained for other pretrained weights')
        if manifest.get('checkpoint_sha256') != file_sha256(checkpoint):
            raise ValueError(f'Bank checkpoint in {directory} differs from its provenance')
        if not (manifest.get('complete') or allow_partial):
            raise ValueError(f'Bank in {directory} is unfinished')
        banks[name] = FrozenToCom.from_checkpoint(checkpoint, depth, width)
        records[name] = manifest
        scales.add(manifest['scale'])
    if len(scales) > 1:
        raise ValueError('Banks were trained with different scales')
    return banks, records, scales.pop() if scales else None


def main(argv=None):
    args = parse_args(argv)
    if args.deterministic:
        # Must precede the first CUDA call; token merging scatters into shared destinations.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    from models.low_rank_utils.factory import checkpoint_spec
    from models.low_rank_utils.pretrained import canonical_pretrained_digest
    from utils.training import train

    run = args.output / (args.run_id or f'{args.method}-{args.operator}-r{args.degree}-seed{args.seed}')
    run.mkdir(parents=True, exist_ok=False)
    spec = checkpoint_spec(args.vit_size)
    values = dict(COMMON, **method_args(args.method, args.vit_size, spec['file'], args.batch_size),
                  n_epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device,
                  base_path=json.loads(DEFAULT_PROFILE.read_text())['data_root'] + '/',
                  results_path=str(run / 'native_results'), checkpoint_path=str(run / 'checkpoints'))
    if args.tasks is not None:
        values['stop_after'] = args.tasks
    arguments = [item for key, value in values.items() for item in (f'--{key}', str(value))] + args.extra
    model, dataset, native = mammoth.initialize(mammoth.parse_args(arguments))
    if native.code_optimization != 0 or native.distributed != 'no':
        raise ValueError('Use FP32 (--code_optimization 0) on one device')
    if args.method.endswith('-lora'):
        # The low-rank models install their source pipeline with unnormalized inputs; the study and the
        # banks use the common recipe, as the unified comparison does.
        from scripts.unified_cl_inputs import common_transforms
        dataset.TRANSFORM, dataset.TEST_TRANSFORM = common_transforms()
        normalize = dataset.TEST_TRANSFORM.transforms[-1]
        dataset.get_normalization_transform = lambda: normalize

    family, encoder = find_encoder(model.net)
    digest = canonical_pretrained_digest(encoder)
    if digest != spec['canonical_sha256']:
        raise ValueError('Encoder differs from the pinned pretrained weights before training')
    banks, bank_records, scale = load_banks(args.banks, digest, len(encoder.blocks), encoder.embed_dim,
                                            args.allow_partial_banks)
    attach(model.net, args.operator, degree=args.degree, bank=banks.get(args.operator))

    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
    config = {'method': args.method, 'vit_size': args.vit_size, 'operator': args.operator, 'degree': args.degree,
              'targets': sorted(set(args.targets)), 'eval_operators': args.eval_operators, 'scale': scale,
              'seed': args.seed, 'grid': not args.no_grid, 'deterministic': args.deterministic,
              'native_arguments': arguments}
    (run / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (run / 'provenance.json').write_text(json.dumps({
        'commit': commit, 'dirty': dirty, 'family': family, 'backbone_canonical_sha256': digest,
        'pretrained_file_sha256': spec['sha256'], 'tocom_official_commit': OFFICIAL_COMMIT,
        'banks': bank_records, 'torch': str(torch.__version__), 'device': str(model.device),
        'started_at_unix': time.time()}, indent=2) + '\n')

    records = []
    native_end_eval = model.meta_end_eval

    def meta_end_eval(evaluated, accs):
        native_end_eval(evaluated, accs)
        # The weight digest lets a run with the grid be compared to one without it.
        record = {'task': len(records), 'classes': evaluated.get_offsets()[1], 'class_il': list(accs[0]),
                  'task_il': list(accs[1]), 'state_sha256': state_digest(model.net)}
        if not args.no_grid:
            started = time.perf_counter()
            grid = evaluate_grid(model, evaluated.test_loaders, record['classes'], args.degree,
                                 args.targets, args.eval_operators, banks, scale)
            if grid['source'] != record['class_il']:
                raise RuntimeError(f'Grid at the training degree differs from Mammoth: {grid["source"]} vs {accs[0]}')
            record.update(grid, grid_seconds=time.perf_counter() - started)
        records.append(record)
        temporary = run / 'metrics.json.tmp'
        temporary.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, run / 'metrics.json')

    model.meta_end_eval = meta_end_eval
    started = time.time()
    train(model, dataset, native)
    (run / 'summary.json').write_text(json.dumps({
        'execution_status': 'completed', 'tasks': len(records), 'seconds': time.time() - started,
        'final_class_il': records[-1]['class_il'] if records else None}, indent=2) + '\n')


if __name__ == '__main__':
    main()
