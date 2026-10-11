"""Evaluate the saved weights of a finished run again, without training.

A run started with ``--save-weights`` keeps, for every task, the model that was
evaluated. This script rebuilds the method, restores that model and repeats the
grid at other inference degrees, operators or banks. With ``--verify`` it only
checks that the restored weights and their accuracy equal what the run recorded.

Examples (from the repository root):
    uv run --no-sync python -m scripts.tocom.reevaluate --run <run directory> --verify
    uv run --no-sync python -m scripts.tocom.reevaluate --run <run directory> --tasks 9 --targets 0 2 6 \\
        --banks tome=<bank dir> evit=<bank dir> --output <json file>
"""

from scripts.cl_runtime import configure_native_environment

configure_native_environment()

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402

from backbone.token_compression import OPERATORS, attach  # noqa: E402
from backbone.token_compression.evaluation import evaluate_grid  # noqa: E402
from scripts.tocom.run_cl import build, grid_plan, load_banks, pinned_digest, state_digest  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='Run directory holding config.json and weights/')
    parser.add_argument('--tasks', type=int, nargs='+', help='Tasks to evaluate, counted from 0 (default: all)')
    parser.add_argument('--verify', action='store_true', help='Only compare the restored models with the record')
    parser.add_argument('--targets', type=int, nargs='+', help='Inference degrees (default: those of the run)')
    parser.add_argument('--eval-operators', nargs='+', choices=tuple(OPERATORS), help='Default: those of the run')
    parser.add_argument('--banks', nargs='*', default=[], metavar='OPERATOR=DIR')
    parser.add_argument('--device', help='Default: the device of the run')
    parser.add_argument('--output', type=Path, help='Result file (default: <run>/reevaluation.json)')
    return parser.parse_args(argv)


def replaced(arguments, **values):
    """The native arguments of the run with some values replaced."""
    arguments = list(arguments)
    for key, value in values.items():
        arguments[arguments.index(f'--{key}') + 1] = str(value)
    return arguments


def restore(model, dataset, state, task, method):
    """Put a freshly built model in the state it was evaluated in after ``task``."""
    if getattr(model, 'old_net', 0) is None and any(key.startswith('old_net.') for key in state):
        # LwF-MC registers its previous network as a module only once a task has ended.
        model.old_net = copy.deepcopy(model.net)
    model.load_state_dict(state)
    if method == 'coda-prompt':
        # CODA-Prompt keeps these counters outside its state dictionary; begin_task sets them.
        model.net.prompt.task_count = task
        model.net.task_id = task or None
        model.offset_1, model.offset_2 = dataset.get_offsets(task)
    model.to(model.device)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads((args.run / 'config.json').read_text())
    records = json.loads((args.run / 'metrics.json').read_text())
    if not all('weights' in record for record in records):
        raise ValueError('This run did not save its weights')
    from models.low_rank_utils.factory import checkpoint_spec

    with tempfile.TemporaryDirectory() as scratch:
        values = dict(results_path=f'{scratch}/results', checkpoint_path=f'{scratch}/checkpoints')
        if args.device:
            values['device'] = args.device
        model, dataset, _ = build(replaced(config['native_arguments'], **values), config['method'])
        model.to(model.device)
        family, encoder, digest = pinned_digest(model.net, checkpoint_spec(config['vit_size']))
        banks, _, scale = load_banks(args.banks, digest, len(encoder.blocks), encoder.embed_dim, False)
        attach(model.net, config['operator'], degree=config['degree'], bank=banks.get(config['operator']))
        source, compensations = grid_plan(config['method'], config['degree'], config['matched_degree'], list(banks))
        targets = args.targets or config['targets']
        operators = args.eval_operators or config['eval_operators']
        if source is not None and source not in targets:
            targets = sorted({*targets, source})

        results = []
        for record in records:
            # Test loaders of the tasks seen so far are built task by task, as in training.
            dataset.get_data_loaders()
            if args.tasks is not None and record['task'] not in args.tasks:
                continue
            state = torch.load(args.run / record['weights'], map_location='cpu', weights_only=True)
            restore(model, dataset, state['model'], record['task'], config['method'])
            model.net.eval()
            class_il, task_il = dataset.evaluate(model, dataset)
            result = {'task': record['task'], 'classes': record['classes'], 'class_il': list(class_il),
                      'state_matches': state_digest(model.net) == record['state_sha256'],
                      'class_il_matches': list(class_il) == record['class_il'],
                      'task_il_matches': list(task_il) == record['task_il']}
            if not args.verify:
                result.update(evaluate_grid(model, dataset.test_loaders, record['classes'], source, targets,
                                            operators, banks, scale or config['scale'], compensations))
            results.append(result)
            print(json.dumps({key: result[key] for key in ('task', 'state_matches', 'class_il_matches', 'task_il_matches')}))
    if args.verify:
        failed = [result['task'] for result in results
                  if not (result['state_matches'] and result['class_il_matches'] and result['task_il_matches'])]
        if failed:
            raise SystemExit(f'Restored models differ from the record at tasks {failed}')
        print(f'verified {len(results)} tasks of {args.run.name}')
        return
    output = args.output or args.run / 'reevaluation.json'
    output.write_text(json.dumps({'run': args.run.name, 'targets': targets, 'eval_operators': operators,
                                  'records': results}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
