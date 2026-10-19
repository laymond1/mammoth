"""Records of one run: the training progress as a local file, and a mirror on Weights & Biases.

The files of the run directory stay the reference. The mirror only reads what the launcher has already
recorded, so it cannot change the training, and a failure of the service is logged and ignored.
"""

from contextlib import contextmanager
import json
import logging
import re
import time

FAMILIES = {'er': 'replay', 'lwf-mc': 'regularization', 'coda-prompt': 'prompt', 'cps-prompt': 'prompt',
            'rep-prompt': 'prompt', 'ease': 'adapter', 'sd-lora': 'lora', 'inf-lora': 'lora', 'ewc-lora': 'lora'}
# Runs without a dataset in their config are on this one; runs on another are told apart by name and group.
DATASET = 'seq-cifar100-224'
UPLOADED = ('config.json', 'provenance.json', 'metrics.json', 'summary.json', 'progress.jsonl')
FINISH_TIMEOUT = 900.


def identity(config):
    """How a run was trained, in the fields runs are filtered and grouped by.

    A run on a budget path has kind ``switch`` and the name of its path as training degree; a run with mixed
    degrees has kind ``mix`` and training degree ``MIX``, one with mixed degrees within the budget of a path kind
    ``switch-mix`` and the path's name followed by ``M`` (``MS`` for mixed shapes); a path's run at the shapes of
    its budgets has the path's name followed by ``S``; joint training has kind ``joint`` and its degree. A run
    at another input resolution than 224 pixels has kind ``shape``, its degree and ``train_resolution``. A run on
    another dataset than seq-cifar100-224 also has ``dataset``.
    """
    fields = _identity(config)
    if config.get('dataset', DATASET) != DATASET:
        fields['dataset'] = config['dataset']
    return fields


def _identity(config):
    if config.get('schedule') or config.get('mix'):
        pool = {'grid-levels': 'L', 'menu': 'U', 'grid': 'G'}.get(config.get('shape_pool'), '')
        suffix = ('M' + ('S' + pool if config.get('mix_shapes') else '') + ('R' if config.get('mix_no_return') else '')
                  + ('W' if config.get('mix_own') is not None else ''))
        kind, degree = (('switch-mix', f'{config["path"]}{suffix}') if config.get('mix') else
                        ('switch', config['path'] + ('S' if config.get('budget_shapes') else ''))) \
            if config.get('schedule') else ('mix', 'MIX')
        return {'method': config['method'], 'family': FAMILIES[config['method']], 'kind': kind,
                'train_operator': config['operator'], 'train_degree': degree, 'seed': config['seed']}
    if config.get('joint'):
        return {'method': config['method'], 'family': FAMILIES[config['method']], 'kind': 'joint',
                'train_operator': config['operator'], 'train_degree': config['degree'], 'seed': config['seed']}
    if config.get('resolution'):
        return {'method': config['method'], 'family': FAMILIES[config['method']], 'kind': 'shape',
                'train_operator': config['operator'] if config['degree'] else 'none', 'train_degree': config['degree'],
                'train_resolution': config['resolution'], 'seed': config['seed']}
    own = config['matched_degree'] is not None
    degree = config['matched_degree'] if own else config['degree']
    kind = 'none' if not degree else 'own' if own else 'plugin'
    return {'method': config['method'], 'family': FAMILIES[config['method']], 'kind': kind,
            'train_operator': config['operator'] if kind == 'plugin' else kind, 'train_degree': degree,
            'seed': config['seed']}


def _mean(values):
    return sum(values) / len(values)


def _conditions(record):
    """Every grid condition of a task record: (operator, kind, degree, accuracy over the tasks seen).

    ``off`` is the inference degree as it is and ``on`` adds the bank of the inference operator. ``on0`` is
    that bank correcting from degree 0 (CPS-Prompt and REP), ``src<degree>`` the bank of a run on a budget
    path correcting from that degree, and ``cross-<bank>`` another operator's bank.
    """
    for name, entry in record.get('operators', {}).items():
        for target, values in entry['off'].items():
            yield name, 'off', int(target), _mean(values)
        for label, targets in entry['on'].items():
            kind = ('on' if label == name else 'on0' if label == f'{name}@0' else
                    label if re.fullmatch(r'src\d+', label) else f'cross-{label}')
            for target, values in targets.items():
                yield name, kind, int(target), _mean(values)


def task_metrics(record):
    """The scalars of one task record under the names they are plotted by, against the tasks learned."""
    metrics = {'task': record['task'] + 1, 'acc/class_il': _mean(record['class_il']),
               'acc/task_il': _mean(record['task_il'])}
    metrics.update({f'task_acc/t{index + 1:02d}': value for index, value in enumerate(record['class_il'])})
    for key, name in (('seconds', 'task_minutes'), ('grid_seconds', 'grid_minutes')):
        if key in record:
            metrics[f'time/{name}'] = record[key] / 60
    for label, accuracies in record.get('shapes', {}).items():
        resolution, _, degree = label.partition(':')
        metrics[f'shape/{resolution}px-r{int(degree):02d}'] = _mean(accuracies)
    values = {(name, kind, degree): value for name, kind, degree, value in _conditions(record)}
    for (name, kind, degree), value in values.items():
        metrics[f'{name}-{kind}/r{degree:02d}'] = value
        if kind in ('on', 'on0'):
            # What the bank adds at this inference degree, in percentage points.
            metrics[f'{name}-gain{kind[2:]}/r{degree:02d}'] = value - values[name, 'off', degree]
    return metrics


def final_curves(record):
    """Per inference operator, accuracy by inference degree for every condition of the last task."""
    curves = {}
    for name, kind, degree, value in _conditions(record):
        curves.setdefault(name, {}).setdefault(kind, {})[degree] = value
    return curves


class Progress:
    """Loss, learning rate and duration of every epoch, appended to ``progress.jsonl`` as the run goes."""

    def __init__(self, path):
        self.path, self.losses, self.rate = path, [], None
        self.mark = self.task_mark = time.time()

    def step(self, loss, rate):
        self.losses.append(float(loss))
        self.rate = rate

    def end_epoch(self, task, epoch):
        now = time.time()
        entry = {'task': task, 'epoch': epoch, 'steps': len(self.losses),
                 'loss': _mean(self.losses) if self.losses else None, 'lr': self.rate,
                 'seconds': now - self.mark, 'time': now}
        with self.path.open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        self.losses, self.mark = [], now
        return entry

    def end_task(self):
        """Seconds since the previous task was recorded; the next epoch is timed from here."""
        now = time.time()
        seconds, self.mark, self.task_mark = now - self.task_mark, now, now
        return seconds


@contextmanager
def _same_random_state():
    """Leave the generators the training draws from exactly as they were."""
    import random
    import numpy as np
    import torch
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


class Mirror:
    """The Weights & Biases run of one training run; does nothing without a target ``entity/project``."""

    def __init__(self, directory, target, experiment, config, extra):
        self.directory, self.run, self.epochs, self.sections = directory, None, 0, set()
        if not target:
            return
        entity, _, project = target.partition('/')
        fields = identity(config)
        # Runs of one setting share a group across seeds; on another dataset the dataset joins the group.
        group = directory.name.rsplit('-seed', 1)[0] + (f'@{fields["dataset"]}' if 'dataset' in fields else '')
        options = dict(
            entity=entity, project=project, name=directory.name, group=group,
            job_type='train', tags=[experiment, fields['kind'], fields['family'], *[fields[key] for key in ('dataset',) if key in fields]],
            dir=str(directory),
            config={'experiment': experiment, **fields, **extra,
                    **{key: config[key] for key in ('vit_size', 'own_setting', 'targets', 'eval_operators', 'scale',
                                                    'deterministic', 'save_weights')},
                    # Runs that evaluate other input shapes name them and their own cost.
                    **{key: config[key] for key in ('eval_shapes', 'cost') if config.get('eval_shapes')}})
        try:
            import wandb
            # No console capture, code or diff upload: the run directory holds the log and the provenance.
            # A run that ends while the service is unreachable waits this long, then leaves the rest to
            # ``wandb sync``, so that an outage cannot hold a queue worker.
            options['settings'] = wandb.Settings(console='off', save_code=False, disable_git=True, silent=True,
                                                 finish_timeout=FINISH_TIMEOUT)
            with _same_random_state():
                try:
                    self.run = wandb.init(**options)
                except Exception as error:
                    logging.warning(f'Weights & Biases is unreachable ({type(error).__name__}); recording offline')
                    self.run = wandb.init(mode='offline', **options)
            self.run.define_metric('epoch')
            self.run.define_metric('train/*', step_metric='epoch')
            self.run.define_metric('task')
            (directory / 'tracking.json').write_text(json.dumps({
                'entity': entity, 'project': project, 'id': self.run.id, 'name': self.run.name,
                'group': options['group'], 'url': self.run.url, 'offline': self.run.url is None}, indent=2) + '\n')
        except Exception as error:
            self._failed(error)

    def _failed(self, error):
        logging.warning(f'Weights & Biases mirror stopped ({type(error).__name__}: {error}); the run goes on')
        self.run = None

    def epoch(self, entry):
        if self.run is None:
            return
        try:
            self.epochs += 1
            values = {'epoch': self.epochs, 'train/task': entry['task'] + 1, 'train/loss': entry['loss'],
                      'train/lr': entry['lr'], 'train/epoch_seconds': entry['seconds']}
            self.run.log({key: value for key, value in values.items() if value is not None})
        except Exception as error:
            self._failed(error)

    def task(self, record):
        if self.run is None:
            return
        try:
            metrics = task_metrics(record)
            for section in {key.split('/')[0] for key in metrics if '/' in key} - self.sections:
                self.run.define_metric(f'{section}/*', step_metric='task')
                self.sections.add(section)
            self.run.log(metrics)
        except Exception as error:
            self._failed(error)

    def finish(self, record, summary, completed):
        """Final curves, the summary and the record files; a run stopped early is marked as failed."""
        if self.run is None:
            return
        try:
            import wandb
            self.run.summary.update({'status': summary['execution_status'], 'tasks_done': summary['tasks'],
                                     'hours': summary['seconds'] / 3600})
            for name, kinds in final_curves(record or {}).items():
                degrees = sorted(kinds['off'])
                self.run.log({f'final/curve-{name}': wandb.plot.line_series(
                    xs=degrees, ys=[[kinds[kind].get(degree) for degree in degrees] for kind in kinds],
                    keys=list(kinds), title=f'{name}: final accuracy by inference degree',
                    xname='tokens removed per block at inference')})
            for name in UPLOADED:
                if (self.directory / name).is_file():
                    self.run.save(str(self.directory / name), base_path=str(self.directory), policy='now')
            self.run.finish(exit_code=0 if completed else 1)
        except Exception as error:
            self._failed(error)
