"""Paired evaluation of one model over inference degrees, with and without compensation.

Every condition sees the same batches and only ``model(inputs)`` is called, so
the grid needs no knowledge of the continual-learning method.
"""

from contextlib import contextmanager
import random

import numpy as np
import torch

from .adapters import compensate, find_encoder
from .reducers import OPERATORS, valid_degree
from .shapes import shape_label


@contextmanager
def preserved_state(model, loaders):
    """Evaluate in eval mode, then restore module modes, global RNG and loader generators."""
    flags = [(module, module.training) for module in model.modules()]
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    generators = {}
    for loader in loaders:
        for owner in (loader, getattr(loader, 'sampler', None)):
            generator = getattr(owner, 'generator', None)
            if generator is not None:
                generators[generator] = generator.get_state()
    try:
        model.eval()
        yield
    finally:
        for generator, state in generators.items():
            generator.set_state(state)
        torch.set_rng_state(rng[0])
        np.random.set_state(rng[1])
        random.setstate(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        for module, training in flags:
            module.training = training


@torch.no_grad()
def evaluate_grid(model, loaders, n_classes, source, targets, operators, banks, scale, compensations=None, sink=None):
    """Class-IL accuracy per loader for the source row and every inference condition.

    ``source`` is the plug-in degree the model was trained at; the source row
    runs the model at that degree. It is None for a model whose own inference
    is not a plug-in degree (REP merging on its own schedule): the source row
    is then the model as it is, and every target is run.

    ``operators`` are the reduction operators to infer with. For each, ``off``
    runs every target degree as is, and ``on[label]`` adds a bank.
    ``compensations`` maps a label to ``(bank name, training degrees, operators
    or None)``: the degrees are one value, or one per encoder pass (see
    ``compensate``), and the last item restricts the label to some inference
    operators. A bank name of None is the bank of the inference operator
    itself, whichever it is. By default every bank corrects from ``source``
    under its own name, which also tests a bank on another operator.

    A condition that cannot differ from the source row (the target is the
    source degree and the bank has nothing to correct) reports that row.
    Accuracy follows ``utils.evaluate``: argmax over the first ``n_classes`` logits.

    ``sink``, when given, is called for every batch and condition as
    ``sink(loader index, condition, logits, labels)``, with the first ``n_classes``
    logits; the condition is None for the source row and ``(operator, label or
    None, target)`` otherwise, conditions that report the source row included.
    It lets a caller keep per-sample predictions; nothing else changes.
    """
    if source is not None:
        valid_degree(source)
    targets = sorted({valid_degree(target) for target in targets})
    if compensations is None:
        if source is None:
            raise ValueError('A model without a plug-in training degree needs explicit compensations')
        compensations = {name: (name, (source,), None) for name in banks}
    compensations = {label: (bank, (degrees,) if isinstance(degrees, int) else tuple(degrees), only)
                     for label, (bank, degrees, only) in compensations.items()}
    for bank, degrees, only in compensations.values():
        names = [bank] if bank is not None else [name for name in operators if only is None or name in only]
        if not degrees or any(name not in banks for name in names):
            raise ValueError('Every compensation needs a loaded bank and its training degrees')
        for name in names:
            for degree in (*degrees, *targets):
                banks[name].covered(degree)
    _, encoder = find_encoder(model.net)
    controller = encoder.token_compression
    trained_operator, trained_bank = controller.operator, controller.bank

    def is_source(target, degrees=()):
        return source is not None and target == source and all(degree == target for degree in degrees)

    labels = {name: [label for label, (_, _, only) in compensations.items() if only is None or name in only]
              for name in operators}
    every = [(name, label, target) for name in operators for target in targets for label in (None, *labels[name])]
    conditions = [condition for condition in every if not is_source(condition[2], compensations[condition[1]][1] if condition[1] else ())]
    reported = [condition for condition in every if condition not in conditions]
    instances = {name: OPERATORS[name]() for name in operators}
    result = {'source': [], 'samples': [], 'operators': {
        name: {'off': {str(t): [] for t in targets}, 'on': {label: {str(t): [] for t in targets} for label in labels[name]}}
        for name in operators}}

    def correct(outputs, labels, index, condition):
        if outputs.shape[1] < n_classes or not torch.isfinite(outputs).all():
            raise ValueError('Invalid evaluation logits')
        if sink is not None:
            for each in ((None, *reported) if condition is None else (condition,)):
                sink(index, each, outputs[:, :n_classes], labels)
        return torch.sum(outputs[:, :n_classes].argmax(1) == labels).item()

    try:
        with preserved_state(model, loaders):
            for index, loader in enumerate(loaders):
                counts, source_count, total = {condition: 0. for condition in conditions}, 0., 0.
                for batch in loader:
                    inputs, targets_ = batch[0].to(model.device), batch[1].to(model.device)
                    controller.operator, controller.bank = trained_operator, trained_bank
                    if source is None:
                        source_count += correct(model(inputs), targets_, index, None)
                    else:
                        with controller.degree(source):
                            source_count += correct(model(inputs), targets_, index, None)
                    for name, label, target in conditions:
                        controller.operator = instances[name]
                        if label is None:
                            with controller.degree(target):
                                counts[name, label, target] += correct(model(inputs), targets_, index, (name, label, target))
                                controller.verify_pass()
                        else:
                            bank, degrees, _ = compensations[label]
                            controller.bank = banks[name if bank is None else bank]
                            with compensate(model.net, degrees, target, scale):
                                counts[name, label, target] += correct(model(inputs), targets_, index, (name, label, target))
                                controller.verify_pass()
                    total += targets_.shape[0]
                if not total:
                    raise ValueError('Empty evaluation loader')
                # Same expression as utils.evaluate, so the source row can be compared exactly.
                result['source'].append(source_count / total * 100)
                result['samples'].append(int(total))
                for name in operators:
                    entry = result['operators'][name]
                    for target in targets:
                        entry['off'][str(target)].append(result['source'][-1] if is_source(target)
                                                         else counts[name, None, target] / total * 100)
                        for label in labels[name]:
                            same = is_source(target, compensations[label][1])
                            entry['on'][label][str(target)].append(result['source'][-1] if same
                                                                   else counts[name, label, target] / total * 100)
    finally:
        controller.operator, controller.bank = trained_operator, trained_bank
    return result


@torch.no_grad()
def evaluate_shapes(model, loaders, n_classes, shapes, operator):
    """Class-IL accuracy per loader at every input shape, without a bank.

    ``shapes`` are ``(resolution, degree)`` pairs, resolution None for the native one; reductions use
    ``operator``. Batches and accuracy are those of ``evaluate_grid``, and the model, the random state and
    the loaders are left as they were. Returns ``{"resolution:degree": [accuracy per loader]}``.
    """
    _, encoder = find_encoder(model.net)
    controller = encoder.token_compression
    shapes = [(controller.native if resolution is None else resolution, valid_degree(degree)) for resolution, degree in shapes]
    for resolution, _ in shapes:
        controller.check_resolution(resolution)
    labels = [shape_label(resolution, degree) for resolution, degree in shapes]
    if len(set(labels)) != len(labels):
        raise ValueError('Every shape is evaluated once')
    trained_operator, instance = controller.operator, OPERATORS[operator]()
    result = {label: [] for label in labels}
    try:
        with preserved_state(model, loaders):
            for loader in loaders:
                counts, total = dict.fromkeys(labels, 0.), 0.
                for batch in loader:
                    inputs, targets = batch[0].to(model.device), batch[1].to(model.device)
                    controller.operator = instance
                    for (resolution, degree), label in zip(shapes, labels):
                        with controller.shape(resolution, degree):
                            outputs = model(inputs)
                            controller.verify_pass()
                        if outputs.shape[1] < n_classes or not torch.isfinite(outputs).all():
                            raise ValueError('Invalid evaluation logits')
                        counts[label] += torch.sum(outputs[:, :n_classes].argmax(1) == targets).item()
                    total += targets.shape[0]
                if not total:
                    raise ValueError('Empty evaluation loader')
                for label in labels:
                    result[label].append(counts[label] / total * 100)
    finally:
        controller.operator = trained_operator
    return result
