"""Reversible common-input protocol adapters for the pinned EASE author code.

The author's forward, optimization, prototypes and semantic completion functions
remain unchanged. Only input, initialization and RNG conventions are adapted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
from unittest.mock import patch

from scripts.ease_reference_utils import install_reference_patches


def tensor_digest(tensors):
    import torch
    result = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        if not torch.is_tensor(value):
            raise TypeError('Audited state must contain only tensors')
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(str(value.dtype).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def first_batch_audit(inputs, labels, named_parameters, *, task, epoch, seed):
    """Read the actual batch and the actual gradients immediately before SGD."""
    import torch
    gradients = {}
    for name, parameter in named_parameters:
        if parameter.grad is not None:
            if not parameter.requires_grad or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError('A frozen or nonfinite parameter gradient was observed')
            gradients[name] = parameter.grad
    if not gradients:
        raise ValueError('The first minibatch has no observed gradients')
    return {'task': task, 'epoch': epoch, 'seed': seed, 'batch_size': len(labels),
            'inputs_sha256': tensor_digest({'inputs': inputs}),
            'labels_sha256': tensor_digest({'labels': labels}),
            'gradient_sha256': tensor_digest(gradients),
            'gradient_parameter_count': len(gradients)}


def common_transforms():
    """Exact HiDe full-stream transforms, including bicubic train interpolation."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    common = [transforms.ToTensor(), transforms.Normalize(
        mean=[.5071, .4867, .4408], std=[.2675, .2565, .2761])]
    train = [transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC,
                                         scale=(.08, 1.), ratio=(.75, 4 / 3)),
             transforms.RandomHorizontalFlip(.5), transforms.ColorJitter(brightness=63 / 255)]
    test = [transforms.Resize(256, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224)]
    return train, test, common


def install_common_reference_patches(modules, checkpoint, data_root, records):
    """Layer reversible transforms/sequential statistics on the existing adapters."""
    stack = install_reference_patches(modules, checkpoint, data_root, records)
    data, ease = modules['data'], modules['ease']
    original_init, original_loader = data.iCIFAR224.__init__, ease.DataLoader

    def initialize(subject, *args, **kwargs):
        original_init(subject, *args, **kwargs)
        subject.train_trsf, subject.test_trsf, subject.common_trsf = common_transforms()

    def loader(dataset, *args, **kwargs):
        source = getattr(dataset, '_ease_reference_source', None)
        mode = getattr(dataset, '_ease_reference_mode', None)
        # The author prototype calculation is unchanged. Stable sequential
        # observations remove a reduction-order confound against native.
        if source == 'train' and mode == 'test':
            if args:
                raise ValueError('Author loaders must provide named options')
            kwargs['shuffle'] = False
        if kwargs.get('drop_last', False) or kwargs.get('persistent_workers', False):
            raise ValueError('Matched EASE loaders require fresh workers and every sample')
        return original_loader(dataset, *args, **kwargs)

    try:
        stack.enter_context(patch.object(data.iCIFAR224, '__init__', initialize))
        stack.enter_context(patch.object(ease, 'DataLoader', loader))
        records['common_input_protocol'] = 'Exact HiDe full10 hide_prompt.yaml transforms'
        records['patches'].extend([
            'HiDe common train/test transforms; augmentation/normalization adaptation, not paper reproduction',
            'prototype samples sequential in split-global order; original averaging/completion functions unchanged'])
        records['loader_seed_policy']['prototype_order_difference'] = 'both reference and native sequential'
        records['loader_seed_policy'].update(
            generator='Phase-seeded fresh torch.Generator for each train epoch/prototype/evaluation pass',
            worker_python_numpy_seed='torch.initial_seed() modulo 2**32',
            epoch_seed='1993 + 100000*zero_based_task + 1000*phase_index + zero_based_epoch')
    except BaseException:
        stack.close()
        raise
    return stack


def export_initialization_catalog(adapter_factory, proxy_factory, path, *, metadata=None):
    """Save ten original-author learnable initializations without any forwards."""
    import numpy as np
    import torch
    from scripts.ease_fullcl_protocol import phase_seed
    path = Path(path)
    if path.exists():
        raise FileExistsError('Initialization export must not overwrite a previous artifact')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    states = {}
    try:
        with torch.random.fork_rng(devices=[]):
            for task in range(10):
                selected = phase_seed(1993, task, 'task_begin', 0)
                random.seed(selected); np.random.seed(selected); torch.manual_seed(selected)
                adapters = torch.nn.ModuleList([adapter_factory() for _ in range(12)])
                proxy = proxy_factory()
                states[str(task)] = {'adapter': {key: tensor.detach().cpu().clone() for key, tensor in adapters.state_dict().items()},
                                     'proxy_head': {key: tensor.detach().cpu().clone() for key, tensor in proxy.state_dict().items()},
                                     'seed': selected}
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
    catalog = {'schema_version': 1, 'tasks': states, 'metadata': {
        'feature_dim': 768, 'adapter_size': 64, 'num_blocks': 12, 'classes_per_task': 10,
        'seed': 1993, 'origin': 'Unmodified pinned-author Adapter and CosineLinear constructors',
        'image_forwards': 0, 'optimizer_steps': 0, **(metadata or {})}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(catalog, path)
    return catalog


def validate_initialization_catalog(catalog):
    import torch
    from scripts.ease_fullcl_protocol import phase_seed
    if catalog.get('schema_version') != 1 or set(catalog.get('tasks', {})) != {str(task) for task in range(10)}:
        raise ValueError('Initialization requires all ten fixed task coordinates')
    expected = {}
    for index in range(12):
        expected.update({f'{index}.down_proj.weight': (64, 768), f'{index}.down_proj.bias': (64,),
                         f'{index}.up_proj.weight': (768, 64), f'{index}.up_proj.bias': (768,)})
    for key, row in catalog['tasks'].items():
        if row['seed'] != phase_seed(1993, int(key), 'task_begin', 0) or set(row['adapter']) != set(expected):
            raise ValueError('Initialization adapter keys or seed differ from the approved protocol')
        if set(row['proxy_head']) != {'weight', 'sigma'}:
            raise ValueError('Initialization proxy keys differ from the cosine head')
        shapes = {**expected, 'proxy_head.weight': (10, 768), 'proxy_head.sigma': (1,)}
        tensors = {**row['adapter'], **{f'proxy_head.{name}': value for name, value in row['proxy_head'].items()}}
        if any(tuple(value.shape) != shapes[name] or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
               for name, value in tensors.items()):
            raise ValueError('Initialization tensors have incorrect shape, dtype or finiteness')
    return catalog


def load_reference_task_initialization(network, catalog, task):
    row = catalog['tasks'][str(task)]
    network.backbone.cur_adapter.load_state_dict(row['adapter'], strict=True)
    network.proxy_fc.load_state_dict(row['proxy_head'], strict=True)
    expected = tensor_digest({**{f'adapter.{key}': value for key, value in row['adapter'].items()},
                              **{f'proxy_head.{key}': value for key, value in row['proxy_head'].items()}})
    actual = tensor_digest({**{f'adapter.{key}': value for key, value in network.backbone.cur_adapter.state_dict().items()},
                            **{f'proxy_head.{key}': value for key, value in network.proxy_fc.state_dict().items()}})
    if actual != expected:
        raise RuntimeError('Author learnable initialization did not transfer exactly')
    return actual


def completion_checks(record, recorder):
    expected = {(task, epoch) for task in (1, 2) for epoch in range(1, 21)}
    rows = record['epoch_results']
    if (len(rows) != 40 or {(row['task'], row['epoch']) for row in rows} != expected
            or any(row['steps'] != 79 or row['samples'] != 5000 or row['epochs'] != 20 for row in rows)
            or recorder.tasks_completed != 2 or recorder.steps != 3160
            or recorder.training_examples != 200000 or recorder.evaluation_examples != 3000
            or record.get('prototype_adapter_image_pairs') != 15000
            or len(record.get('first_batch_audits', [])) != 2):
        raise RuntimeError('Author common-input reference did not complete the exact two-task protocol')
    return True
