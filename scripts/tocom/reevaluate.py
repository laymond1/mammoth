"""Evaluate the saved weights of a finished run again, without training.

A run started with ``--save-weights`` keeps, for every task, the model that was
evaluated. This script rebuilds the method, restores that model and repeats the
grid at other inference degrees, operators or banks. With ``--verify`` it only
checks that the restored weights and their accuracy equal what the run recorded.

Examples (from the repository root):
    uv run --no-sync python -m scripts.tocom.reevaluate --run <run directory> --verify
    uv run --no-sync python -m scripts.tocom.reevaluate --run <run directory> --tasks 9 --targets 0 2 6 \\
        --banks tome=<bank dir> evit=<bank dir> --output <json file>

``--eval-shapes R:D ...`` also evaluates the restored models at those input shapes (resolution:degree) with
``--shapes-operator`` (default: the run's training operator) and no bank. A run trained at another resolution
is restored and evaluated at that resolution.

``--predictions FILE`` also keeps, for every test sample and grid condition, the class-IL prediction, the
prediction among the classes of the sample's own task (task-IL) and the largest logit of each task's classes
(see ``Predictions``), and checks that they give the accuracy of the grid.
"""

from scripts.cl_runtime import configure_native_environment

configure_native_environment()

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from backbone.token_compression import OPERATORS, attach, parse_shape, shape_label  # noqa: E402
from backbone.token_compression.evaluation import evaluate_grid, evaluate_shapes  # noqa: E402
from scripts.tocom.run_cl import (budget_shape, build, grid_plan, load_banks, pinned_digest, state_digest,  # noqa: E402
                                  switch_plan)


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
    parser.add_argument('--predictions', type=Path, help='Also keep per-sample predictions in this .npz file')
    parser.add_argument('--source-only', action='store_true',
                        help='Evaluate the model as it was evaluated in the run (the source row) and no other condition')
    parser.add_argument('--eval-shapes', nargs='+', default=[], metavar='R:D',
                        help='Also evaluate at these input shapes (resolution:degree), without a bank')
    parser.add_argument('--shapes-operator', choices=tuple(OPERATORS), help="Default: the run's training operator")
    args = parser.parse_args(argv)
    try:
        args.eval_shapes = [parse_shape(text) for text in args.eval_shapes]
    except ValueError as error:
        parser.error(str(error))
    return args


class Predictions:
    """Per-sample outputs of the grid conditions, as ``evaluate_grid`` passes them to its sink.

    For a condition and a sample: ``pred``, the class-IL prediction (argmax over the classes seen); ``within``,
    the argmax among the classes of the sample's own task, which is the task-IL prediction; ``task_max``, the
    largest logit of each task's classes. A score offset per task moves ``task_max`` and leaves ``within``, so
    the two give class-IL under any such offset. Conditions are named ``source`` or
    ``<operator>/<bank label or off>/<target>``; with several tasks, ``t<task>:`` comes first.
    """

    def __init__(self, width):
        self.width, self.prefix, self.arrays = width, '', {}

    @staticmethod
    def name(condition):
        if condition is None:
            return 'source'
        operator, label, target = condition
        return f'{operator}/{label or "off"}/{target}'

    def __call__(self, index, condition, logits, labels):
        blocks = logits.shape[1] // self.width
        if blocks * self.width != logits.shape[1]:
            raise ValueError('The classes seen are not a whole number of tasks')
        tasks = logits.view(len(logits), blocks, self.width)
        own = labels // self.width
        values = {'pred': logits.argmax(1), 'within': tasks[torch.arange(len(logits)), own].argmax(1) + own * self.width,
                  'task_max': tasks.amax(2)}
        if condition is None:
            values.update(label=labels, loader=torch.full_like(labels, index))
        for key, value in values.items():
            name = f'{self.prefix}{key}' if key in ('label', 'loader') else f'{self.prefix}{self.name(condition)}|{key}'
            self.arrays.setdefault(name, []).append(value.cpu())

    def check(self, grid):
        """The accuracy of every condition, recomputed from the predictions, equals the grid's."""
        labels, loaders = (torch.cat(self.arrays[f'{self.prefix}{key}']) for key in ('label', 'loader'))

        def accuracy(name):
            hits = torch.cat(self.arrays[f'{self.prefix}{name}|pred']) == labels
            return [hits[loaders == index].sum().item() / (loaders == index).sum().item() * 100
                    for index in range(len(grid['source']))]

        if accuracy('source') != grid['source']:
            raise RuntimeError('Predictions of the source row differ from the grid')
        for operator, entry in grid['operators'].items():
            for label, targets in (('off', entry['off']), *entry['on'].items()):
                for target, values in targets.items():
                    if accuracy(f'{operator}/{label}/{target}') != values:
                        raise RuntimeError(f'Predictions of {operator}/{label}/{target} differ from the grid')

    def save(self, path):
        arrays = {name: torch.cat(values).numpy() for name, values in self.arrays.items()}
        np.savez_compressed(path, **{name: value.astype(np.int16) if value.dtype == np.int64 else value
                                     for name, value in arrays.items()})


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
    if config.get('deterministic'):
        # As in the run, and before the first CUDA call: token merging scatters into shared destinations,
        # and without this a merged condition can differ from the record by an image.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if not all('weights' in record for record in records):
        raise ValueError('This run did not save its weights')
    from models.low_rank_utils.factory import checkpoint_spec

    with tempfile.TemporaryDirectory() as scratch:
        values = dict(results_path=f'{scratch}/results', checkpoint_path=f'{scratch}/checkpoints')
        if args.device:
            values['device'] = args.device
        # A run from before the evaluation batch was fixed is evaluated as it was, with its training batch.
        model, dataset, _ = build(replaced(config['native_arguments'], **values), config['method'],
                                  config.get('eval_batch_size'))
        model.to(model.device)
        family, encoder, digest = pinned_digest(model.net, checkpoint_spec(config['vit_size']))
        banks, _, scale = load_banks(args.banks, digest, len(encoder.blocks), encoder.embed_dim, False)
        schedule = config.get('schedule')
        controller = attach(model.net, config['operator'], degree=schedule[0] if schedule else config['degree'],
                            bank=banks.get(config['operator']))
        # A run trained at another input resolution was evaluated at it.
        controller.default_resolution = controller.check_resolution(config.get('resolution'))
        if schedule:
            # A budget path: each task is evaluated at its own degree, with the bank from every source of the run.
            source, compensations = None, switch_plan(config['operator'], config['sources'])
        elif config.get('mix'):
            source, compensations = config['degree'], switch_plan(config['operator'], config['sources'])
        else:
            source, compensations = grid_plan(config['method'], config['degree'], config['matched_degree'], list(banks))
        targets = args.targets or config['targets']
        operators = args.eval_operators or config['eval_operators']
        if args.source_only:
            targets, operators, compensations = [], [], {}
        if source is not None and source not in targets:
            targets = sorted({*targets, source})
        if schedule and not args.source_only:
            targets = sorted({*targets, *schedule})

        results = []
        # Classes per task of the dataset as defined; joint training gives the instance one task of all classes.
        predictions = Predictions(type(dataset).N_CLASSES_PER_TASK) if args.predictions else None
        chosen = [record['task'] for record in records if args.tasks is None or record['task'] in args.tasks]
        for record in records:
            # Test loaders of the tasks seen so far are built task by task, as in training.
            dataset.get_data_loaders()
            if args.tasks is not None and record['task'] not in args.tasks:
                continue
            state = torch.load(args.run / record['weights'], map_location='cpu', weights_only=True)
            restore(model, dataset, state['model'], record['task'], config['method'])
            if schedule:
                # The task's budget, at (224, budget) or the shape the run gave that budget.
                shapes = {int(budget): parse_shape(text) for budget, text in (config.get('budget_shapes') or {}).items()}
                resolution, controller.default_degree = budget_shape(shapes, schedule[record['task']])
                controller.default_resolution = controller.check_resolution(resolution)
            model.net.eval()
            class_il, task_il = dataset.evaluate(model, dataset)
            result = {'task': record['task'], 'classes': record['classes'], 'class_il': list(class_il),
                      'state_matches': state_digest(model.net) == record['state_sha256'],
                      'class_il_matches': list(class_il) == record['class_il'],
                      'task_il_matches': list(task_il) == record['task_il']}
            if not args.verify:
                if predictions is not None:
                    predictions.prefix = f't{record["task"]}:' if len(chosen) > 1 else ''
                result.update(evaluate_grid(model, dataset.test_loaders, record['classes'],
                                            schedule[record['task']] if schedule else source, targets,
                                            operators, banks, scale or config['scale'], compensations, predictions))
                if predictions is not None:
                    predictions.check(result)
                if args.eval_shapes:
                    result['shapes'] = evaluate_shapes(model, dataset.test_loaders, record['classes'], args.eval_shapes,
                                                       args.shapes_operator or config['operator'])
            results.append(result)
            print(json.dumps({key: result[key] for key in ('task', 'state_matches', 'class_il_matches', 'task_il_matches')}))
    if args.verify:
        failed = [result['task'] for result in results
                  if not (result['state_matches'] and result['class_il_matches'] and result['task_il_matches'])]
        if failed:
            raise SystemExit(f'Restored models differ from the record at tasks {failed}')
        print(f'verified {len(results)} tasks of {args.run.name}')
        return
    if predictions is not None:
        predictions.save(args.predictions)
    output = args.output or args.run / 'reevaluation.json'
    output.write_text(json.dumps({'run': args.run.name, 'targets': targets, 'eval_operators': operators,
                                  'eval_shapes': [shape_label(*shape) for shape in args.eval_shapes],
                                  'shapes_operator': (args.shapes_operator or config['operator']) if args.eval_shapes else None,
                                  'records': results}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
