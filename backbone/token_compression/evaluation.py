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
def evaluate_grid(model, loaders, n_classes, source, targets, operators, banks, scale):
    """Class-IL accuracy per loader for the source degree and every inference condition.

    ``operators`` are the reduction operators to infer with. For each, ``off``
    runs every target degree as is, and ``on[name]`` adds the bank ``name``
    from ``banks`` (its own bank, or another operator's to test transfer).
    A target equal to ``source`` is the source row itself. Accuracy follows
    ``utils.evaluate``: argmax over the first ``n_classes`` logits.
    """
    valid_degree(source)
    targets = sorted({valid_degree(target) for target in targets})
    _, encoder = find_encoder(model.net)
    controller = encoder.token_compression
    trained_operator, trained_bank = controller.operator, controller.bank
    conditions = [(name, bank, target) for name in operators for target in targets if target != source
                  for bank in (None, *banks)]
    instances = {name: OPERATORS[name]() for name in operators}
    result = {'source': [], 'samples': [], 'operators': {
        name: {'off': {str(t): [] for t in targets}, 'on': {bank: {str(t): [] for t in targets} for bank in banks}}
        for name in operators}}

    def correct(outputs, labels):
        if outputs.shape[1] < n_classes or not torch.isfinite(outputs).all():
            raise ValueError('Invalid evaluation logits')
        return torch.sum(outputs[:, :n_classes].argmax(1) == labels).item()

    try:
        with preserved_state(model, loaders):
            for loader in loaders:
                counts, source_count, total = {condition: 0. for condition in conditions}, 0., 0.
                for batch in loader:
                    inputs, labels = batch[0].to(model.device), batch[1].to(model.device)
                    controller.operator, controller.bank = trained_operator, trained_bank
                    with controller.degree(source):
                        source_count += correct(model(inputs), labels)
                    for name, bank, target in conditions:
                        controller.operator = instances[name]
                        if bank is None:
                            with controller.degree(target):
                                counts[name, bank, target] += correct(model(inputs), labels)
                                controller.verify_pass()
                        else:
                            controller.bank = banks[bank]
                            with compensate(model.net, source, target, scale):
                                counts[name, bank, target] += correct(model(inputs), labels)
                                controller.verify_pass()
                    total += labels.shape[0]
                if not total:
                    raise ValueError('Empty evaluation loader')
                # Same expression as utils.evaluate, so the source row can be compared exactly.
                result['source'].append(source_count / total * 100)
                result['samples'].append(int(total))
                for name in operators:
                    entry = result['operators'][name]
                    for target in targets:
                        same = target == source
                        entry['off'][str(target)].append(result['source'][-1] if same
                                                         else counts[name, None, target] / total * 100)
                        for bank in banks:
                            entry['on'][bank][str(target)].append(result['source'][-1] if same
                                                                  else counts[name, bank, target] / total * 100)
    finally:
        controller.operator, controller.bank = trained_operator, trained_bank
    return result
