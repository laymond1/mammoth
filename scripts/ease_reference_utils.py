"""Local, reversible integration patches for the unmodified EASE reference.

These helpers do not run a model, download data, or send external messages.
The orchestration script owns the authorized task loop and device selection.
"""

import ast
from contextlib import ExitStack
from functools import partial
import hashlib
import inspect
from pathlib import Path
import pickle
import random
import struct
import textwrap
from unittest.mock import patch

import numpy as np


BASE_TAG = 'vit_base_patch16_224.augreg_in21k_ft_in1k'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_vector(values, name, count):
    vector = np.asarray(values)
    if vector.ndim != 1 or len(vector) != count or vector.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain one integer per sample')
    return vector.astype(np.int64, copy=False)


def dataset_fingerprint(images, labels, global_ids, transform=None):
    """Hash raw HWC images, mapped labels and original split-global identities.

    Sorting the complete records by their unique global IDs preserves sample
    multiplicity while removing iteration-order differences. Transform repr is
    recorded separately: an identity Normalize can change repr without pixels.
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError('CIFAR fingerprints require uint8 HWC RGB image arrays')
    labels = _integer_vector(labels, 'labels', len(images))
    global_ids = _integer_vector(global_ids, 'global_ids', len(images))
    if (global_ids < 0).any() or len(np.unique(global_ids)) != len(global_ids):
        raise ValueError('CIFAR global sample IDs must be nonnegative and unique')
    if (labels < 0).any():
        raise ValueError('CIFAR fingerprint labels must be fully labeled')
    header = struct.pack('<QQQQ', len(images), *images.shape[1:])
    records = []
    for image, label, global_id in zip(images, labels, global_ids):
        digest = hashlib.sha256(struct.pack('<qq', int(global_id), int(label)))
        digest.update(np.ascontiguousarray(image).tobytes())
        records.append(digest.digest())
    ordered = hashlib.sha256(header)
    unordered = hashlib.sha256(header)
    for record in records:
        ordered.update(record)
    for index in np.argsort(global_ids, kind='stable'):
        unordered.update(records[int(index)])
    class_labels, counts = np.unique(labels, return_counts=True)
    transform_repr = repr(transform)
    return {
        'n_samples': len(images),
        'image_shape': list(images.shape[1:]),
        'image_dtype': str(images.dtype),
        'class_counts': {str(int(label)): int(count) for label, count in zip(class_labels, counts)},
        'samples_ordered_sha256': ordered.hexdigest(),
        'samples_unordered_sha256': unordered.hexdigest(),
        'global_ids_ordered_sha256': hashlib.sha256(global_ids.astype('<i8').tobytes()).hexdigest(),
        'first_global_ids': global_ids[:10].tolist(),
        'transform_repr': transform_repr,
        'transform_repr_sha256': hashlib.sha256(transform_repr.encode()).hexdigest(),
    }


def shared_cifar_metadata(data_root):
    """Describe the existing official CIFAR100 files without downloading them."""
    root = Path(data_root).expanduser().resolve() / 'CIFAR100' / 'cifar-100-python'
    files = {}
    for name in ('train', 'test', 'meta'):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing shared CIFAR100 file: {path}')
        files[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
        # CIFAR100's trusted torchvision distribution uses this pickle format.
        with path.open('rb') as stream:
            content = pickle.load(stream, encoding='latin1')
        if name != 'meta':
            labels = np.asarray(content['fine_labels'], dtype=np.int64)
            data = np.asarray(content['data'])
            if data.dtype != np.uint8 or data.shape != (len(labels), 3072):
                raise ValueError('Shared CIFAR100 raw sample layout is incompatible')
            classes, counts = np.unique(labels, return_counts=True)
            files[name].update(n_samples=len(labels),
                               class_counts={str(int(label)): int(count) for label, count in zip(classes, counts)})
        else:
            files[name]['fine_label_names'] = list(content['fine_label_names'])
    return {'dataset': 'CIFAR100', 'root': str(root.parent), 'files': files,
            'global_id_policy': 'zero-based original index within each train/test split'}


def _module(stack, name):
    return stack[name] if isinstance(stack, dict) else getattr(stack, name)


def _load_pinned_weights(model, checkpoint):
    if checkpoint.suffix == '.npz':
        model.load_pretrained(str(checkpoint))
        return
    if checkpoint.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device='cpu')
    else:
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Reference checkpoint must contain a model state dictionary')
    for key in ('state_dict', 'model'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if state and all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if key not in ('head.weight', 'head.bias')}
    model.load_state_dict(state, strict=True)


def _native_worker_seed(worker_id, num_workers, seed):
    """Match Mammoth's callback while retaining PyTorch's worker Torch seed."""
    worker_seed = num_workers + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _size_constructor(original, feature_dim, num_heads=None):
    """Specialize the author's geometry constants without replacing its code.

    The upstream factory has fixed 768-wide QKV conversion slices as well as
    architecture arguments. Specializing those same dimension constants keeps
    its original conversion, initialization and freezing behavior intact. The
    classifier reweight method also has a fixed 768-wide default argument;
    specializing that default leaves its complete calculation body intact.
    """
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    changes = {'feature_dim_constants': 0, 'attention_head_arguments': 0}

    class SizeConstants(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 768:
                changes['feature_dim_constants'] += 1
                return ast.copy_location(ast.Constant(feature_dim), node)
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'VisionTransformer' and num_heads is not None:
                for keyword in node.keywords:
                    if keyword.arg == 'num_heads' and isinstance(keyword.value, ast.Constant) and keyword.value.value == 12:
                        changes['attention_head_arguments'] += 1
                        keyword.value = ast.copy_location(ast.Constant(num_heads), keyword.value)
            return node

    tree = ast.fix_missing_locations(SizeConstants().visit(tree))
    if not changes['feature_dim_constants'] or (num_heads is not None and changes['attention_head_arguments'] != 1):
        raise RuntimeError('Reviewed upstream size constructor no longer has the expected dimension constants')
    namespace = dict(original.__globals__)
    exec(compile(tree, inspect.getsourcefile(original) or '<reference-constructor>', 'exec'), namespace)
    specialized = namespace[original.__name__]
    specialized.__module__, specialized.__qualname__ = original.__module__, original.__qualname__
    changes['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    changes['specialized_ast_sha256'] = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    return specialized, changes


def install_reference_patches(stack, checkpoint, data_root, records, seed=1993, variant=None):
    """Install protocol adapters; return an ExitStack that restores each patch.

    ``stack`` provides the official ``data``, ``data_manager``, ``ease`` and
    ``vit`` modules plus installed ``timm``. ``records`` is a public mutable dict.
    All learning/prototype code remains the pinned upstream implementation.
    An explicit size variant specializes two upstream constructors and the
    classifier's block-width default; absent-variant Base uses their originals.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('The reference loader seed must be a nonnegative integer')
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing exact shared EASE checkpoint: {checkpoint}')
    cifar_root = Path(data_root).expanduser().resolve() / 'CIFAR100'
    if not (cifar_root / 'cifar-100-python' / 'train').is_file():
        raise FileNotFoundError('Shared CIFAR100 data must already exist')
    timm = _module(stack, 'timm')
    data = _module(stack, 'data')
    manager_module = _module(stack, 'data_manager')
    ease = _module(stack, 'ease')
    vit = _module(stack, 'vit')
    original_create = timm.create_model
    original_get_dataset = manager_module.DataManager.get_dataset
    original_load = vit.VisionTransformer.load_state_dict
    original_loader = ease.DataLoader
    loader_signature = inspect.signature(original_loader)
    tag = BASE_TAG if variant is None else variant['checkpoint_tag']
    size = 'Base' if variant is None else variant['vit_size'].title()
    records['checkpoint'] = {'path': str(checkpoint), 'sha256': file_sha256(checkpoint),
                             'bytes': checkpoint.stat().st_size, 'tag': tag}
    if variant is not None:
        architecture = {'base': (768, 12), 'small': (384, 6), 'tiny': (192, 3)}
        if (architecture.get(variant['vit_size']) != (variant['feature_dim'], variant['num_heads'])
                or Path(variant['checkpoint_file']).resolve() != checkpoint
                or records['checkpoint']['sha256'] != variant['checkpoint_sha256']):
            raise ValueError('Reference variant architecture or local checkpoint differs from its pin')
        records['variant'] = dict(variant)
        records['reference_kind'] = 'author-code-size-variant'
    records['patches'] = [
        f'exact tagged {size} checkpoint loaded locally; pretrained downloads disabled',
        'shared CIFAR100 root with download=False',
        'plain CIFAR samples reordered by original split-global IDs',
        'strict reference pretrained key audit; only cur_adapter.* may be missing',
        'reference data-loader num_workers=4',
        'native-matched loader generator and Python/NumPy worker seeding',
    ]
    records['loader_seed_policy'] = {
        'seed': seed,
        'generator': 'independent torch.Generator.manual_seed(seed) for each loader',
        'worker_python_numpy_seed': 'num_workers + worker_id + seed',
        'worker_torch_seed': 'standard PyTorch initialization; callback does not modify it',
        'prototype_order_difference': 'reference prototype loader keeps shuffle=True; native prototype loader uses shuffle=False',
    }

    def pinned_create(model_name, *args, **kwargs):
        # The author's original factory retains its Base alias; the selected
        # constructor dimension and tagged local checkpoint are both explicit.
        if model_name not in ('vit_base_patch16_224', tag, tag.split('.', 1)[0]):
            raise ValueError(f'Reference may construct only the selected {size} checkpoint family')
        if kwargs.get('num_classes', 0) != 0 or kwargs.get('checkpoint_path'):
            raise ValueError('Reference checkpoint factory requires a classifier-free frozen backbone')
        kwargs['pretrained'] = False
        kwargs['num_classes'] = 0
        model = original_create(tag, *args, **kwargs)
        _load_pinned_weights(model, checkpoint)
        records.setdefault('checkpoint_loads', []).append({'requested_alias': model_name, 'resolved_tag': tag})
        return model

    def shared_download(subject):
        train = data.datasets.cifar.CIFAR100(str(cifar_root), train=True, download=False)
        test = data.datasets.cifar.CIFAR100(str(cifar_root), train=False, download=False)
        subject.train_data, subject.train_targets = train.data, np.asarray(train.targets)
        subject.test_data, subject.test_targets = test.data, np.asarray(test.targets)
        records['dataset_root'] = str(cifar_root)

    def audited_load(subject, state_dict, strict=True, assign=False):
        result = original_load(subject, state_dict, strict=strict, assign=assign)
        if not strict:
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            if unexpected or any(not key.startswith('cur_adapter.') for key in missing):
                raise RuntimeError('Reference pretrained conversion contains missing or unexpected backbone keys')
            records.setdefault('converted_checkpoint_audits', []).append({'missing_adapter_keys': missing,
                                                                         'unexpected_keys': unexpected})
        return result

    def aligned_get_dataset(subject, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        if subject.dataset_name.lower() != 'cifar224':
            raise ValueError('The matched reference protocol supports only plain CIFAR100-224')
        if (appendent is not None and len(appendent) != 0) or m_rate is not None:
            raise ValueError('The matched reference protocol forbids replay and sample removal')
        indices = np.asarray(list(indices))
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(np.unique(indices)) != len(indices):
            raise ValueError('Reference class indices must be unique integers')
        result = original_get_dataset(subject, indices, source, mode, appendent=appendent,
                                      ret_data=ret_data, m_rate=m_rate)
        dataset = result[2] if ret_data else result
        if dataset.use_path:
            raise ValueError('Reference CIFAR samples must be raw arrays, not paths')
        if source == 'train':
            images, labels = subject._train_data, subject._train_targets
        elif source == 'test':
            images, labels = subject._test_data, subject._test_targets
        else:
            raise ValueError('Reference source must be train or test')
        images, labels = np.asarray(images), np.asarray(labels)
        global_ids = np.flatnonzero(np.isin(labels, indices))
        original_ids = np.concatenate([np.flatnonzero(labels == label) for label in indices])
        if (not np.array_equal(dataset.images, images[original_ids])
                or not np.array_equal(dataset.labels, labels[original_ids])):
            raise RuntimeError('Reference sample-order adapter may only reorder the original selected samples')
        original_fingerprint = dataset_fingerprint(dataset.images, dataset.labels, original_ids, dataset.trsf)
        dataset.images = images[global_ids]
        dataset.labels = labels[global_ids]
        dataset.global_ids = global_ids
        dataset._ease_reference_source = source
        dataset._ease_reference_mode = mode
        aligned_fingerprint = dataset_fingerprint(dataset.images, dataset.labels, global_ids, dataset.trsf)
        if original_fingerprint['samples_unordered_sha256'] != aligned_fingerprint['samples_unordered_sha256']:
            raise RuntimeError('Reference ordering adapter changed the selected sample multiset')
        records.setdefault('datasets', []).append({'source': source, 'mode': mode, 'classes': indices.tolist(),
                                                   'original_ordered_sha256': original_fingerprint['samples_ordered_sha256'],
                                                   **aligned_fingerprint})
        return (dataset.images, dataset.labels, dataset) if ret_data else dataset

    def seeded_loader(dataset, *args, **kwargs):
        import torch
        options = loader_signature.bind_partial(dataset, *args, **kwargs).arguments
        workers = options.get('num_workers', 0)
        if 'generator' not in options:
            kwargs['generator'] = torch.Generator().manual_seed(seed)
        if 'worker_init_fn' not in options:
            kwargs['worker_init_fn'] = partial(_native_worker_seed, num_workers=workers, seed=seed)
        source = getattr(dataset, '_ease_reference_source', None)
        mode = getattr(dataset, '_ease_reference_mode', None)
        records.setdefault('loaders', []).append({
            'source': source, 'mode': mode, 'num_workers': workers,
            'shuffle': options.get('shuffle', False),
            'drop_last': options.get('drop_last', False),
            'generator_seed': seed if 'generator' not in options else 'caller_provided',
            'worker_callback': 'native_python_numpy' if 'worker_init_fn' not in options else 'caller_provided',
        })
        return original_loader(dataset, *args, **kwargs)

    restoration = ExitStack()
    try:
        if variant is not None:
            inc_net = _module(stack, 'inc_net')
            factory, factory_audit = _size_constructor(vit.vit_base_patch16_224_ease, variant['feature_dim'], variant['num_heads'])
            get_backbone, backbone_audit = _size_constructor(inc_net.get_backbone, variant['feature_dim'])
            classifier = inc_net.CosineLinear
            reweight, reweight_audit = _size_constructor(classifier.forward_reweight, variant['feature_dim'])
            restoration.enter_context(patch.object(vit, 'vit_base_patch16_224_ease', factory))
            restoration.enter_context(patch.object(inc_net, 'get_backbone', get_backbone))
            restoration.enter_context(patch.object(classifier, 'forward_reweight', reweight))
            records['constructor_specialization'] = {'vit_factory': factory_audit, 'get_backbone': backbone_audit,
                                                       'classifier_reweight': reweight_audit}
            records['patches'].append('author constructors and classifier block-width default specialized for feature dimension, attention heads and configured adapter width; classifier calculation unchanged')
        restoration.enter_context(patch.object(timm, 'create_model', pinned_create))
        restoration.enter_context(patch.object(data.iCIFAR224, 'download_data', shared_download))
        restoration.enter_context(patch.object(manager_module.DataManager, 'get_dataset', aligned_get_dataset))
        restoration.enter_context(patch.object(vit.VisionTransformer, 'load_state_dict', audited_load))
        restoration.enter_context(patch.object(ease, 'num_workers', 4))
        restoration.enter_context(patch.object(ease, 'DataLoader', seeded_loader))
    except BaseException:
        restoration.close()
        raise
    return restoration
