"""Run one continual-learning method with token compression and evaluate the ToCom grid.

The method code is untouched: the launcher builds the model through Mammoth's
own entry points, patches the encoder blocks, and wraps ``meta_end_eval`` so
that after every task the model is also evaluated at each inference degree,
with and without the compensator bank of each operator.

CPS-Prompt and REP reduce tokens with their own algorithm in training. They take
``--matched-degree`` instead of a training degree: the ToMe degree whose FLOPs their
own setting equals. Their banks correct each encoder pass from the degree that
pass was trained at. On a budget path (``--schedule``) each task takes the own setting of
the matched degree its path gives it; the grid then holds, for each inference operator in
``--eval-operators``, the bank of that operator from every ``--sources`` degree (``src<degree>``,
the prompted pass trained at that degree, the query pass on every token), and the source row is the model as a
fixed run evaluates it.

Training uses batches of 16 and every evaluation batches of 256, unless told otherwise;
Mammoth alone would evaluate with the training batch size.

``--schedule`` trains each task at its own degree, a budget path: the task about to be learned and the
evaluation after it run at that task's degree, and the grid records the bank of the training operator
from every ``--sources`` degree, so that bank policies can be compared afterwards per test set.

Two references for budget paths: ``--joint`` learns every class as one task (Mammoth's joint setting) at
one degree, the usual upper bound of a continual method; ``--mix`` trains every step at a degree drawn
uniformly from a list, one model for any budget, and is evaluated like a budget path. With ``--schedule``,
``--mix`` draws every step from the degrees of the list that cost no more than the task's own (the degree
or more): training at mixed degrees within the budget of the moment. ``--mix-no-return`` trains a task whose
budget came back (its degree is below an earlier task's) at its own degree only, and ``--mix-own SHARE`` draws the
degree the step would otherwise run at with that probability and the others uniformly.

Every run keeps its records in its own directory. ``--wandb ENTITY/PROJECT`` also
mirrors them to Weights & Biases; Mammoth's own logging there stays off.

Examples (from the repository root):
    uv run --no-sync python -m scripts.tocom.run_cl --method er --operator tome --degree 8 \\
        --banks tome=<bank dir> evit=<bank dir> --seed 1993 --output results/tocom-cl-main
    uv run --no-sync python -m scripts.tocom.run_cl --method rep-prompt --matched-degree 16 \\
        --eval-operators tome evit --banks tome=<bank dir> evit=<bank dir> --seed 1993 --output results/tocom-cl-main
    uv run --no-sync python -m scripts.tocom.run_cl --method coda-prompt --operator tome --path P5 \\
        --schedule 0 0 0 0 0 24 24 24 24 24 --banks tome=<bank dir> --targets 0 4 8 12 16 20 24 --seed 1993 \\
        --output results/tocom-switch
    uv run --no-sync python -m scripts.tocom.run_cl --method inf-lora --operator evit --mix 0 8 16 24 \\
        --banks evit=<bank dir> --targets 0 4 8 12 16 20 24 --seed 1993 --output results/tocom-bounds
    uv run --no-sync python -m scripts.tocom.run_cl --method cps-prompt --path P5 \\
        --schedule 0 0 0 0 0 24 24 24 24 24 --eval-operators tome evit --banks tome=<bank dir> evit=<bank dir> \\
        --targets 0 4 8 12 16 20 24 --seed 1993 --output results/scenario-cps-rep-paths
"""

from scripts.cl_runtime import DEFAULT_PROFILE, configure_native_environment

configure_native_environment()

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

import main as mammoth  # noqa: E402
from backbone.token_compression import OPERATORS, FrozenToCom, attach, encoders, valid_degree  # noqa: E402
from backbone.token_compression.compensator import OFFICIAL_COMMIT, file_sha256  # noqa: E402
from backbone.token_compression.evaluation import evaluate_grid  # noqa: E402
from scripts.tocom.tracking import Mirror, Progress  # noqa: E402
from utils.conf import create_seeded_dataloader  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# Method recipes of the unified Tiny comparison (scripts/unified_cl_protocol.py), except where a method notes
# a value set for this study (ViT-Tiny, batch 16; results/tocom-recipe-diagnosis-20261012/HANDOFF.md).
COMMON = dict(dataset='seq-cifar100-224', dataset_config='unified', model_config='default', num_workers=4,
              drop_last=0, permute_classes=0, fitting_mode='epochs', scheduler_mode='epoch',
              enable_other_metrics=0, eval_future=0, joint=0, non_verbose=1, code_optimization=0)
# The method's own compression setting at which the FLOPs of its prompted pass equal ToMe at a training
# degree, on ViT-Tiny (scripts/tocom/flops.py): the ratio of patches CPS-Prompt samples away in its
# joint phase, and the largest per-block merge of REP's progressive schedule with layer dropping off.
# Degree 24 is 71 patches kept for CPS-Prompt; REP's schedule is clamped to half of the tokens in every
# block, so 187 is the smallest maximum that reaches the FLOPs of ToMe 24 (184 to 186 fall .003 short).
OWN = {'cps-prompt': {0: 0., 8: .237, 16: .477, 24: .635}, 'rep-prompt': {0: 0, 8: 27, 16: 71, 24: 187}}


def method_args(method, size, checkpoint, batch_size, matched=None):
    if method == 'er':
        # Unified has AdamW at 1e-4, which at batch 16 moves the whole backbone too far (31% after ten tasks).
        # SGD at 0.001 was chosen over AdamW at 1e-5 on validation data: 49.5% against 25.1%.
        return dict(model='er', backbone='vit-augreg', vit_size=size, pretrained=1, vit_checkpoint=checkpoint,
                    optimizer='sgd', lr=0.001, optim_mom=0.9, optim_wd=0.0, buffer_size=500, minibatch_size=batch_size)
    if method == 'coda-prompt':
        return dict(model='coda-prompt', backbone='vit-augreg', vit_size=size, pretrained=0, coda_vit_size=size,
                    coda_checkpoint=checkpoint, optimizer='adam', lr=0.001, optim_wd=0.0, pool_size=100,
                    prompt_len=8, mu=0.0, ortho_mu=0.0, virtual_bs_iterations=1)
    if method == 'sd-lora':
        # Unified has lr 0.001 and classifier lr 0.008. At batch 16 on Tiny the classifier of each new task then
        # pushes the old tasks out, so both are a quarter of that here.
        return dict(model='sd-lora', backbone='low-rank-vit', lora_vit_size=size, lora_checkpoint=checkpoint,
                    optimizer='adam', lr=0.00025, optim_wd=0.0, lr_scheduler='cosine', lora_rank=10, lora_fc_lr=0.002)
    if method == 'lwf-mc':
        # As for ER: SGD at 0.001 instead of unified's AdamW at 1e-4 (55.4% against 52.9% for AdamW at 1e-5
        # on validation data, 43% with the unified setting).
        return dict(model='lwf_mc', backbone='vit-augreg', vit_size=size, pretrained=1, vit_checkpoint=checkpoint,
                    optimizer='sgd', lr=0.001, optim_mom=0.9, optim_wd=0.0, wd_reg=0.0)
    if method == 'inf-lora':
        return dict(model='inf-lora', backbone='low-rank-vit', lora_vit_size=size, lora_checkpoint=checkpoint,
                    optimizer='adam', lr=0.0005, optim_wd=0.0, lr_scheduler='cosine', lora_rank=10,
                    lora_fc_lr=0.0005, lora_lamb=0.95, lora_lame=1.0)
    if method == 'ewc-lora':
        # Unified follows the released code: factors merged separately and penalty weight 1e7, under which the
        # update of every task after the first is zero. Here the merge of the paper is used, and the weight was
        # chosen on validation data at batch 16 (the Fisher estimate grows as the batch shrinks).
        return dict(model='ewc-lora', backbone='low-rank-vit', lora_vit_size=size, lora_checkpoint=checkpoint,
                    optimizer='adam', lr=0.0005, optim_wd=0.0, lr_scheduler='cosine', lora_rank=10,
                    lora_fc_lr=0.005, lora_gamma=1.0, lora_ewc_lambda=1000.0, lora_ewc_merge='product')
    if method == 'ease':
        return dict(model='ease', backbone='ease-vit', vit_size=size, pretrained=1, ease_checkpoint=checkpoint,
                    ease_pretrain_type='in21k_ft_in1k', adapter_size=64, optimizer='sgd', lr=0.025,
                    optim_wd=0.0005, optim_mom=0.9, optim_nesterov=0, lr_scheduler='cosine')
    # The prompt methods below build their own frozen encoder; the backbone argument is a placeholder.
    if method == 'cps-prompt':
        # Author recipe except for the ratio (0.4 there); the joint phase is the first 40% of the epochs.
        return dict(model='cps-prompt', backbone='resnet18', vit_type=size, cps_checkpoint=checkpoint,
                    optimizer='adam', lr=0.001, optim_wd=0.0, e_prompt_pool_size=100, e_prompt_length=8,
                    reduction_ratio=OWN[method][matched], temperature=0.1, phase_ratio=0.4)
    if method == 'rep-prompt':
        # REP merges on its own progressive schedule and keeps its swift query. Layer dropping is off:
        # it acts in training only, so with it the FLOPs of training and inference could not both match.
        maximum = OWN[method][matched]
        return dict(model='rep-prompt', backbone='resnet18', rep_vit_size=size, rep_checkpoint=checkpoint,
                    rep_query_checkpoint=checkpoint, optimizer='adam', lr=0.001, optim_wd=0.0, pool_size=100,
                    prompt_len=8, rep_swift=1, rep_atom=int(maximum > 0), rep_ald=0, rep_merge_max=maximum)
    raise ValueError(f'No recipe for {method}')


def joint_args(method):
    """Native arguments of joint training: one task of every class. The low-rank methods keep one classifier
    head per task, sized by ``--lora_classes_per_task``, so for them that head must hold all hundred classes."""
    return dict(joint=1, **({'lora_classes_per_task': 100} if method.endswith('-lora') else {}))


def mix_suffix(config):
    """Name of a path's run with mixed degrees: ``M``, then ``R`` without mixing after a return, ``W`` with a share."""
    get = config.get if isinstance(config, dict) else lambda name: getattr(config, name)
    return 'M' + ('R' if get('mix_no_return') else '') + ('W' if get('mix_own') is not None else '')


def grid_plan(method, degree, matched, bank_names):
    """Source row and bank conditions of the grid, as ``evaluate_grid`` takes them.

    A method trained with the plug-in is corrected from its training degree by every bank. CPS-Prompt
    and REP run two encoder passes trained differently: the query pass always sees every token, the
    prompted pass is compressed by the method itself, to the FLOPs of the matched degree. Each bank
    corrects the passes from (0, matched); as the match is one of FLOPs, not of token schedules, the
    inference operator's own bank is also tried from (0, 0).
    """
    if method not in OWN:
        return degree, None
    compensations = {name: (name, (0, matched), None) for name in bank_names}
    if matched:
        compensations.update({f'{name}@0': (name, (0, 0), [name]) for name in bank_names})
    return own_source(method), compensations


def own_source(method):
    """Source row of a model that compresses on its own. CPS-Prompt infers with every token, which is the
    plug-in at degree 0. REP's own inference merges on its schedule, which is no plug-in degree: its source
    row is the model as it is."""
    return 0 if method == 'cps-prompt' else None


def own_switch_plan(sources):
    """Grid conditions of CPS-Prompt or REP trained on a budget path, as ``evaluate_grid`` takes them.

    The bank of the inference operator (no name: each operator takes its own) corrects the two encoder passes
    from (0, degree): the query pass saw every token, the prompted pass was trained at the matched degree.
    The labels are those of a plug-in model on a path, ``src<degree>``; ``src0`` is the ``@0`` row of a fixed run.
    """
    return {f'src{degree}': (None, (0, degree), None) for degree in sorted(set(sources))}


def apply_own_setting(model, method, setting):
    """Give a built CPS-Prompt or REP model the own compression setting of the task about to be learned.

    Both read their setting from a plain attribute at every forward call, so a budget path changes it between
    tasks and a constant path is the fixed run: the ratio of patches CPS-Prompt's sampler removes, and REP's
    largest per-block merge with its merging switched on or off.
    """
    if method == 'cps-prompt':
        if not 0 <= setting < 1:
            raise ValueError('reduction_ratio must be in [0,1)')
        model.net.sampler.reduction_ratio = float(setting)
    elif method == 'rep-prompt':
        if type(setting) is not int or setting < 0:
            raise ValueError('rep_merge_max must be a nonnegative integer')
        model.net.maximum, model.net.atom = setting, setting > 0
    else:
        raise ValueError(f'{method} has no own compression setting')


def own_setting(model, method):
    """The own compression setting a built CPS-Prompt or REP model holds now (what ``apply_own_setting`` sets)."""
    return model.net.sampler.reduction_ratio if method == 'cps-prompt' else model.net.maximum


def switch_plan(operator, sources):
    """Grid conditions of a model trained on a budget path, as ``evaluate_grid`` takes them.

    Such a model has no single training degree. The bank of its operator corrects from each source degree,
    labelled ``src<degree>``; the source row of every evaluation is the degree of the task just learned.
    Bank policies (one fixed source, the last training degree, each test set's own training degree) are
    read from these rows afterwards.
    """
    return {f'src{degree}': (operator, (degree,), None) for degree in sorted(set(sources))}


METHODS = ('er', 'coda-prompt', 'sd-lora', 'lwf-mc', 'inf-lora', 'ewc-lora', 'ease', 'cps-prompt', 'rep-prompt')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--vit-size', choices=('tiny', 'small', 'base'), default='tiny')
    parser.add_argument('--operator', choices=tuple(OPERATORS), default='tome', help='Reduction used in training')
    parser.add_argument('--degree', type=int, default=0, help='Training degree: tokens removed per block')
    parser.add_argument('--schedule', type=int, nargs='+', metavar='DEGREE',
                        help='Training degree of every task in order, a budget path; replaces --degree. CPS-Prompt and '
                             'REP take the matched degree of every task (0, 8, 16, 24) and train it with their own setting')
    parser.add_argument('--path', help='With --schedule: name of the budget path (letters and digits), used in the run id')
    parser.add_argument('--sources', type=int, nargs='+', default=[0, 8, 16, 24],
                        help='With --schedule or --mix: degrees the bank of the training operator corrects from in the grid')
    parser.add_argument('--mix', type=int, nargs='+', metavar='DEGREE',
                        help='Train every step at a degree drawn uniformly from these; everything else (task '
                             'boundaries, evaluation) runs at --degree. With --schedule: from those at least the '
                             "task's degree, and everything else at the task's degree")
    parser.add_argument('--mix-own', type=float, metavar='SHARE',
                        help='With --mix: draw the degree the step would otherwise run at (--degree, or the degree of the '
                             'task on a path) with this probability, and the other allowed degrees uniformly')
    parser.add_argument('--mix-no-return', action='store_true',
                        help='With --mix and --schedule: a task whose degree is below that of an earlier task (the budget '
                             'came back) trains at its own degree only')
    parser.add_argument('--joint', action='store_true',
                        help='Learn every class as one task at --degree (Mammoth --joint 1): the joint upper bound')
    parser.add_argument('--matched-degree', type=int, choices=(0, 8, 16, 24),
                        help='CPS-Prompt and REP: the ToMe training degree their own compression equals in FLOPs '
                             '(with --schedule, every task has its own)')
    parser.add_argument('--targets', type=int, nargs='+', default=[0, 4, 8, 10, 13, 16], help='Inference degrees')
    parser.add_argument('--eval-operators', nargs='+', choices=tuple(OPERATORS),
                        help='Operators to infer with; default: all at degree 0, else the training operator')
    parser.add_argument('--banks', nargs='*', default=[], metavar='OPERATOR=DIR',
                        help='Bank directories holding checkpoint.pth and provenance.json')
    parser.add_argument('--allow-partial-banks', action='store_true', help='Accept unfinished banks (smoke runs)')
    parser.add_argument('--no-grid', action='store_true', help='Train only; used to prove the grid has no side effects')
    parser.add_argument('--save-weights', action='store_true',
                        help='Keep the evaluated model of every task, for scripts.tocom.reevaluate')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic algorithms: reproducible weights')
    parser.add_argument('--wandb', metavar='ENTITY/PROJECT', help='Mirror the records of the run to Weights & Biases')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--tasks', type=int, help='Stop after this many tasks (default: all)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch (and replay batch of ER)')
    parser.add_argument('--eval-batch-size', type=int, default=256, help='Batch of every evaluation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True, help='Experiment directory; one sub-directory per run')
    parser.add_argument('--run-id', help='Default: <method>-<operator>-r<degree>-seed<seed>, or '
                                         '<method>-own-m<matched degree>-seed<seed>, or on a path of CPS-Prompt and REP '
                                         '<method>-own-<path>-seed<seed>')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='After --: extra Mammoth arguments')
    args = parser.parse_args(argv)
    args.extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    if args.wandb and (args.wandb.count('/') != 1 or not all(args.wandb.split('/'))):
        parser.error('--wandb takes ENTITY/PROJECT')
    if args.schedule is not None:
        if args.degree:
            parser.error('--schedule replaces --degree')
        if not args.path or not re.fullmatch(r'[A-Za-z0-9]+', args.path):
            parser.error('--schedule needs --path, a name of letters and digits')
        try:
            for degree in (*args.schedule, *args.sources):
                valid_degree(degree)
        except ValueError as error:
            parser.error(str(error))
        if not set(args.schedule) <= set(args.targets):
            parser.error('Inference degrees must include every degree of the path')
        if args.method in OWN:
            # No training operator: each task trains with the method's own setting, the plug-in stays at degree 0,
            # and the inference operators are the caller's (each with its bank).
            if not set(args.schedule) <= set(OWN[args.method]):
                parser.error(f'{args.method} has its own setting at degrees {sorted(OWN[args.method])}: '
                             'a budget path may use only these')
            if args.matched_degree is not None:
                parser.error('--matched-degree is for a fixed budget: a path gives every task its own')
        else:
            if args.eval_operators not in (None, [args.operator]):
                parser.error('A model trained on a budget path is inferred with its own operator')
            args.eval_operators, args.degree = [args.operator], args.schedule[0]
    elif args.path is not None:
        parser.error('--path goes with --schedule')
    if args.mix is None and (args.mix_own is not None or args.mix_no_return):
        parser.error('--mix-own and --mix-no-return go with --mix')
    if args.joint and (args.schedule is not None or args.mix or args.method in OWN or args.tasks is not None):
        parser.error('--joint trains one task at one plug-in degree: not with --schedule, --mix, --tasks or CPS/REP')
    if args.mix is not None:
        if args.method in OWN:
            parser.error('--mix is for methods trained with the plug-in')
        if args.mix_own is not None and not 0 < args.mix_own < 1:
            parser.error('--mix-own is a probability between 0 and 1')
        if args.mix_no_return and args.schedule is None:
            parser.error('--mix-no-return goes with --schedule')
        if args.schedule is not None and not set(args.schedule) <= set(args.mix):
            parser.error('With --schedule, --mix must include every degree of the path')
        try:
            for degree in (*args.mix, *args.sources):
                valid_degree(degree)
        except ValueError as error:
            parser.error(str(error))
        if args.degree not in args.mix or not set(args.mix) <= set(args.targets):
            parser.error('--degree and the inference degrees must include the degrees of --mix')
        if args.eval_operators not in (None, [args.operator]):
            parser.error('A model trained with mixed degrees is inferred with its own operator')
        args.eval_operators = [args.operator]
    if args.degree not in args.targets:
        parser.error('Inference degrees must include the training degree')
    if (args.method in OWN) != (args.matched_degree is not None) and not (args.method in OWN and args.schedule is not None):
        parser.error('--matched-degree is required for CPS-Prompt and REP, and is only theirs')
    if args.method in OWN and (args.degree or args.vit_size != 'tiny'):
        parser.error('CPS-Prompt and REP compress with their own algorithm in training (degree 0); '
                     'their settings are matched on ViT-Tiny only')
    if args.eval_operators is None:
        args.eval_operators = sorted(OPERATORS) if args.degree == 0 else [args.operator]
    elif args.degree and args.eval_operators != [args.operator]:
        parser.error('A model trained with reduction is inferred with its own operator')
    return args


def state_digest(net):
    digest = hashlib.sha256()

    def update(value):
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            # Extra state of a module: lifecycle counters and generator states.
            for key in sorted(value):
                digest.update(str(key).encode())
                update(value[key])
        else:
            digest.update(repr(value).encode())

    for name, value in sorted(net.state_dict().items()):
        digest.update(name.encode())
        update(value)
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


def fixed_evaluation_batch(dataset, size):
    """Give the test loaders batches of ``size``; Mammoth builds them with the training batch size.

    The loader of each task is built again the way Mammoth built it (own seeded generator, no shuffling),
    so neither the training nor the order of the test samples changes.
    """
    native = dataset.get_data_loaders

    def get_data_loaders():
        train, test = native()
        if test.batch_size != size:
            test = create_seeded_dataloader(dataset.args, test.dataset, non_verbose=True, batch_size=size, shuffle=False)
            dataset.test_loaders[-1] = test
        return train, test

    dataset.get_data_loaders = get_data_loaders


def build(arguments, method, eval_batch=None):
    """Mammoth model and dataset for these native arguments, with the inputs of the study.

    ``eval_batch`` fixes the batch size of the test loaders; None leaves them as Mammoth builds them.
    """
    # Mammoth turns its own Weights & Biases logging on from these variables, and that logging runs every
    # training step under a profiler. The launcher mirrors the run itself (scripts/tocom/tracking.py).
    for name in ('WANDB_ENTITY', 'WANDB_PROJECT'):
        os.environ.pop(name, None)
    model, dataset, native = mammoth.initialize(mammoth.parse_args(arguments))
    if not native.nowand:
        raise ValueError("Mammoth's own Weights & Biases logging must stay off")
    if native.code_optimization != 0 or native.distributed != 'no':
        raise ValueError('Use FP32 (--code_optimization 0) on one device')
    if method.endswith('-lora') or method == 'ease':
        # The low-rank models and EASE install their source pipeline with unnormalized inputs; the study
        # and the banks use the common recipe, as the unified comparison does.
        from scripts.unified_cl_inputs import common_transforms
        dataset.TRANSFORM, dataset.TEST_TRANSFORM = common_transforms()
        normalize = dataset.TEST_TRANSFORM.transforms[-1]
        dataset.get_normalization_transform = lambda: normalize
    for transform in (dataset.TRANSFORM, dataset.TEST_TRANSFORM):
        normalize = transform.transforms[-1]
        if tuple(normalize.mean) != (.5, .5, .5) or tuple(normalize.std) != (.5, .5, .5):
            raise ValueError('The pretrained weights and the banks expect inputs normalized with 0.5/0.5')
    if eval_batch:
        fixed_evaluation_batch(dataset, eval_batch)
    return model, dataset, native


def experiment_name(output):
    """Name of the experiment a run belongs to: its output directory, as seen from ``results/``."""
    output = output.resolve()
    try:
        return '-'.join(output.relative_to(ROOT / 'results').parts)
    except ValueError:
        return output.name


def pinned_digest(net, spec):
    """Digest of the pretrained weights every compressed encoder of ``net`` starts from."""
    from models.low_rank_utils.pretrained import canonical_pretrained_digest
    family, members = encoders(net)
    if any(canonical_pretrained_digest(member) != spec['canonical_sha256'] for member in members):
        raise ValueError('An encoder differs from the pinned pretrained weights before training')
    return family, members[0], spec['canonical_sha256']


def main(argv=None):
    args = parse_args(argv)
    if args.deterministic:
        # Must precede the first CUDA call; token merging scatters into shared destinations.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    from models.low_rank_utils.factory import checkpoint_spec
    from utils.training import train

    own, switching, mixing = args.method in OWN, args.schedule is not None, args.mix is not None
    # Absolute, because Mammoth resolves a relative results path under its data root.
    default_id = (f'{args.method}-own-{args.path}-seed{args.seed}' if own and switching else
                  f'{args.method}-own-m{args.matched_degree}-seed{args.seed}' if own else
                  f'{args.method}-{args.operator}-{args.path}{mix_suffix(args)}-seed{args.seed}' if switching and mixing else
                  f'{args.method}-{args.operator}-{args.path}-seed{args.seed}' if switching else
                  f'{args.method}-{args.operator}-MIX-seed{args.seed}' if mixing else
                  f'{args.method}-{args.operator}-J{args.degree}-seed{args.seed}' if args.joint else
                  f'{args.method}-{args.operator}-r{args.degree}-seed{args.seed}')
    run = (args.output / (args.run_id or default_id)).resolve()
    run.mkdir(parents=True, exist_ok=False)
    spec = checkpoint_spec(args.vit_size)
    # CPS-Prompt and REP on a budget path are built with the setting of the first task.
    matched = args.schedule[0] if own and switching else args.matched_degree
    values = dict(COMMON, **method_args(args.method, args.vit_size, spec['file'], args.batch_size, matched),
                  n_epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device,
                  base_path=json.loads(DEFAULT_PROFILE.read_text())['data_root'] + '/',
                  results_path=str(run / 'native_results'), checkpoint_path=str(run / 'checkpoints'))
    if args.tasks is not None:
        values['stop_after'] = args.tasks
    if args.joint:
        values.update(joint_args(args.method))
    arguments = [item for key, value in values.items() for item in (f'--{key}', str(value))] + args.extra
    model, dataset, native = build(arguments, args.method, args.eval_batch_size)
    if switching and len(args.schedule) != dataset.N_TASKS:
        raise ValueError(f'The budget path has {len(args.schedule)} degrees for {dataset.N_TASKS} tasks')

    family, encoder, digest = pinned_digest(model.net, spec)
    banks, bank_records, scale = load_banks(args.banks, digest, len(encoder.blocks), encoder.embed_dim,
                                            args.allow_partial_banks)
    controller = attach(model.net, args.operator, degree=args.degree, bank=banks.get(args.operator))
    if switching or mixing:
        if not args.no_grid:
            # CPS-Prompt and REP are inferred with every operator named, each with its own bank.
            needed = args.eval_operators if own else [args.operator]
            if any(name not in banks for name in needed):
                raise ValueError('A budget path is evaluated with the bank of each inference operator: pass it with --banks')
            # Checked now rather than at the first evaluation, hours later.
            for degree in (*(args.schedule or ()), *(args.mix or ()), *args.sources, *args.targets):
                for name in needed:
                    banks[name].covered(degree)
        if own:
            source, compensations = own_source(args.method), own_switch_plan(args.sources)
        else:
            source, compensations = (None if switching else args.degree), switch_plan(args.operator, args.sources)
    else:
        source, compensations = grid_plan(args.method, args.degree, args.matched_degree, list(banks))

    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
    config = {'method': args.method, 'vit_size': args.vit_size, 'operator': args.operator,
              'degree': None if switching else args.degree, 'schedule': args.schedule, 'path': args.path,
              'sources': sorted(set(args.sources)) if switching or mixing else None,
              'mix': sorted(set(args.mix)) if mixing else None, 'mix_own': args.mix_own, 'mix_no_return': args.mix_no_return,
              'joint': args.joint,
              'matched_degree': args.matched_degree,
              'own_setting': None if not own else ([OWN[args.method][degree] for degree in args.schedule] if switching
                                                   else OWN[args.method][args.matched_degree]),
              'targets': sorted(set(args.targets)), 'eval_operators': args.eval_operators, 'scale': scale,
              'source': source, 'compensations': compensations, 'save_weights': args.save_weights,
              'seed': args.seed, 'grid': not args.no_grid, 'deterministic': args.deterministic,
              'batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size, 'native_arguments': arguments}
    (run / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (run / 'provenance.json').write_text(json.dumps({
        'commit': commit, 'dirty': dirty, 'family': family, 'backbone_canonical_sha256': digest,
        'pretrained_file_sha256': spec['sha256'], 'tocom_official_commit': OFFICIAL_COMMIT,
        'banks': bank_records, 'torch': str(torch.__version__), 'device': str(model.device),
        'started_at_unix': time.time()}, indent=2) + '\n')
    mirror = Mirror(run, args.wandb, experiment_name(args.output), config, {
        'epochs': args.epochs, 'batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size,
        'commit': commit, 'dirty': dirty,
        'encoder_family': family, 'backbone_canonical_sha256': digest,
        'bank_sha256': {name: record['checkpoint_sha256'] for name, record in bank_records.items()},
        'native': values})

    records = []
    native_observe, native_end_epoch, native_end_eval = model.meta_observe, model.meta_end_epoch, model.meta_end_eval
    native_begin_task = model.meta_begin_task

    def meta_begin_task(evaluated):
        # A budget path: the task about to be learned (with whatever its method runs when a task begins)
        # and the evaluation after it run at that task's degree, until the next task begins. CPS-Prompt and REP
        # train it with their own setting of that matched degree, the plug-in staying at degree 0.
        degree = valid_degree(args.schedule[len(records)])
        if own:
            apply_own_setting(model, args.method, OWN[args.method][degree])
        else:
            controller.default_degree = degree
        native_begin_task(evaluated)

    # Mixed degrees: a generator of its own, so that the draws leave every other random stream as it was.
    draws, steps = random.Random(args.seed), collections.Counter()

    def meta_observe(*inputs, **keywords):
        # Read-only: the loss Mammoth gets back, and the learning rate the step has just used.
        base = controller.default_degree
        if mixing:
            # On a budget path, only degrees that cost no more than the task's own.
            allowed = [d for d in args.mix if d >= base] if switching else list(args.mix)
            if args.mix_no_return and base < max(args.schedule[:len(records) + 1]):
                allowed = [base]
            if args.mix_own is not None and len(allowed) > 1:
                others = [d for d in allowed if d != base]
                controller.default_degree = base if draws.random() < args.mix_own else draws.choice(others)
            else:
                controller.default_degree = draws.choice(allowed)
            steps[controller.default_degree] += 1
        try:
            loss = native_observe(*inputs, **keywords)
        finally:
            if mixing:
                controller.default_degree = base
        optimizer = getattr(model, 'opt', None)
        progress.step(loss, optimizer.param_groups[0]['lr'] if optimizer is not None else None)
        return loss

    def meta_end_epoch(epoch, trained):
        native_end_epoch(epoch, trained)
        mirror.epoch(progress.end_epoch(len(records), epoch))

    def meta_end_eval(evaluated, accs):
        # The grid reads the model Mammoth has just evaluated, before a method's own end-of-evaluation
        # step (EWC-LoRA merges its new update there). The weight digest lets a run with the grid be
        # compared to one without it.
        record = {'task': len(records), 'classes': evaluated.get_offsets()[1], 'class_il': list(accs[0]),
                  'task_il': list(accs[1]), 'state_sha256': state_digest(model.net)}
        if switching:
            record['degree'] = args.schedule[len(records)] if own else controller.default_degree
        if switching and own:
            # Read from the model, not from the schedule: the setting its task was trained with.
            record['own_setting'] = own_setting(model, args.method)
            if record['own_setting'] != OWN[args.method][record['degree']]:
                raise RuntimeError(f'Task {record["task"]} was not trained with the own setting of degree {record["degree"]}')
        if mixing:
            record['mix_steps'] = {str(degree): steps[degree] for degree in sorted(set(args.mix))}
            steps.clear()
        if args.save_weights:
            # The whole method state, so that the model can be rebuilt and inferred again; no optimizer
            # and no replay buffer, so training cannot resume from it.
            path = run / 'weights' / f'task{record["task"]:02d}.pt'
            path.parent.mkdir(exist_ok=True)
            torch.save({'model': model.state_dict(), 'task': record['task'], 'classes': record['classes']}, path)
            record['weights'] = str(path.relative_to(run))
        if not args.no_grid:
            started = time.perf_counter()
            grid = evaluate_grid(model, evaluated.test_loaders, record['classes'],
                                 record['degree'] if switching and not own else source,
                                 args.targets, args.eval_operators, banks, scale, compensations)
            if grid['source'] != record['class_il']:
                raise RuntimeError(f'Grid source row differs from Mammoth: {grid["source"]} vs {accs[0]}')
            record.update(grid, grid_seconds=time.perf_counter() - started)
        native_end_eval(evaluated, accs)
        record['seconds'] = progress.end_task()
        records.append(record)
        temporary = run / 'metrics.json.tmp'
        temporary.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, run / 'metrics.json')
        mirror.task(record)

    model.meta_observe, model.meta_end_epoch, model.meta_end_eval = meta_observe, meta_end_epoch, meta_end_eval
    if switching:
        model.meta_begin_task = meta_begin_task
    progress = Progress(run / 'progress.jsonl')
    started = time.time()
    train(model, dataset, native)
    # Mammoth returns quietly when a signal stops it, so completion is judged by the tasks evaluated.
    expected = dataset.N_TASKS if args.tasks is None else args.tasks
    completed = len(records) == expected
    summary = {'execution_status': 'completed' if completed else 'interrupted', 'tasks': len(records),
               'expected_tasks': expected, 'seconds': time.time() - started,
               'final_class_il': records[-1]['class_il'] if records else None}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    mirror.finish(records[-1] if records else None, summary, completed)
    if not completed:
        raise SystemExit(f'Stopped after {len(records)} of {expected} tasks')


if __name__ == '__main__':
    main()
