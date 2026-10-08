"""Explicit input streams for the fresh common-protocol comparison.

All coordinates are zero-based. The runner changes ``phase`` before every
main, replay, statistics or SVD pass. This module never reseeds the algorithm:
per-image augmentation temporarily saves/restores only CPU Python/NumPy/Torch
RNG, and each sampler/DataLoader uses its own CPU generator. Outputs of replay
transforms stay on CPU; the caller controls device transfers.

Only already masked MammothDatasetWrapper datasets are accepted. Rewrapping
retains original indexes, class mapping, non-augmented images and extra fields.
Methods which construct their own loaders must explicitly use these APIs;
config equality alone does not prove that an internal pass is isolated.
"""
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import numbers
import pickle
import random
import threading

from scripts.cl_runtime import configure_native_environment

configure_native_environment()

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import transforms


INPUT_SCHEMA = 'unified-cl-inputs-v1'
_CPU_RNG_LOCK = threading.RLock()


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


@dataclass(frozen=True)
class InputPhase:
    task: int
    epoch: int
    phase: str
    pass_id: int = 0

    def __post_init__(self):
        for name in ('task', 'epoch', 'pass_id'):
            object.__setattr__(self, name, _nonnegative_integer(getattr(self, name), name))
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError('phase must be an explicit nonempty name')


def stable_seed(seed, *keys):
    payload = json.dumps([INPUT_SCHEMA, _nonnegative_integer(seed, 'seed'), *keys],
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2 ** 63)


def capture_cpu_rng():
    """No CUDA state access, initialization or deferred CUDA seed operations."""
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.default_generator.get_state().clone()}


def cpu_rng_digest(state=None):
    state = capture_cpu_rng() if state is None else state
    digest = hashlib.sha256()
    digest.update(pickle.dumps(state['python'], protocol=4))
    numpy_state = state['numpy']
    digest.update(numpy_state[0].encode())
    digest.update(numpy_state[1].tobytes())
    digest.update(pickle.dumps(numpy_state[2:], protocol=4))
    digest.update(state['torch'].cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def isolated_cpu_rng(seed):
    """Temporarily seed CPU augmentation and restore even when it raises.

    torch.default_generator.manual_seed deliberately avoids the CUDA side
    effects of the global seeding convenience function. Not a model-RNG reset.
    """
    seed = _nonnegative_integer(seed, 'seed')
    with _CPU_RNG_LOCK:
        before = capture_cpu_rng()
        try:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.default_generator.manual_seed(seed)
            yield
        finally:
            torch.default_generator.set_state(before['torch'])
            np.random.set_state(before['numpy'])
            random.setstate(before['python'])


def common_transforms():
    """The approved TorchVision PIL recipe, including replay augmentation."""
    interpolation = transforms.InterpolationMode.BICUBIC
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.08, 1.), ratio=(.75, 4. / 3.),
                                     interpolation=interpolation, antialias=True),
        transforms.RandomHorizontalFlip(.5), transforms.ToTensor(), normalize])
    evaluation = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    return train, evaluation


def describe_transform(transform):
    """Audit actual supported operations; unknown/Kornia routes fail visibly."""
    if transform is None:
        return None
    if isinstance(transform, ReplayTransform):
        return {'operation': 'KeyedReplayTransform', 'recipe': describe_transform(transform.streams.train_transform),
                'original_ids_required': True, 'device': 'cpu'}
    if isinstance(transform, transforms.Compose):
        return {'operation': 'Compose', 'transforms': [describe_transform(item) for item in transform.transforms]}
    common = {'operation': type(transform).__name__}
    if isinstance(transform, transforms.RandomResizedCrop):
        return dict(common, size=list(transform.size), scale=list(transform.scale), ratio=list(transform.ratio),
                    interpolation=transform.interpolation.name, antialias=transform.antialias)
    if isinstance(transform, transforms.RandomHorizontalFlip):
        return dict(common, p=transform.p)
    if isinstance(transform, transforms.Resize):
        return dict(common, size=transform.size if isinstance(transform.size, int) else list(transform.size),
                    interpolation=transform.interpolation.name, max_size=transform.max_size, antialias=transform.antialias)
    if isinstance(transform, transforms.CenterCrop):
        return dict(common, size=list(transform.size))
    if isinstance(transform, transforms.Normalize):
        return dict(common, mean=list(transform.mean), std=list(transform.std), inplace=transform.inplace)
    if isinstance(transform, transforms.ToTensor):
        return common
    if isinstance(transform, transforms.ToPILImage):
        return dict(common, mode=transform.mode)
    raise ValueError(f'Unaudited input transform: {type(transform).__module__}.{type(transform).__name__}')


def transform_fingerprint(transform):
    description = describe_transform(transform)
    encoded = json.dumps(description, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'description': description, 'sha256': hashlib.sha256(encoded).hexdigest()}


def _original_ids(values, *, count=None, unique=False):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or bool((ids < 0).any()):
        raise ValueError('Original indexes must be a nonnegative integer vector')
    if bool((ids > np.iinfo(np.int64).max).any()):
        raise ValueError('Original indexes exceed int64')
    ids = ids.astype(np.int64, copy=True)
    if count is not None and len(ids) != count:
        raise ValueError('Original index count differs from image count')
    if unique and len(np.unique(ids)) != len(ids):
        raise ValueError('Dataset original indexes must be unique, including duplicate-image rows')
    return ids


def _array_digest(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _pil_image(image):
    if isinstance(image, Image.Image):
        return image
    if torch.is_tensor(image):
        image = image.detach().cpu()
        if image.ndim != 3:
            raise ValueError('Replay images must be raw CHW tensors, not normalized batches')
        if image.is_floating_point() and (not torch.isfinite(image).all() or bool((image < 0).any()) or bool((image > 1).any())):
            raise ValueError('Replay image tensors must contain raw [0,1] pixels')
        return transforms.ToPILImage()(image)
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3:
        raise ValueError('Raw array images must be HWC uint8')
    return Image.fromarray(array)


class ReplayTransform:
    """Callable for Buffer/apply_transform, requiring explicitly bound IDs.

    Assign to model.transform after construction to avoid Kornia conversion.
    The runner keeps Buffer sampling/reservoir RNG natural and binds the chosen
    slots' original dataset IDs. Never infer identity from image pixel hashes.
    """
    def __init__(self, streams):
        self.streams = streams
        self._binding = None

    def __call__(self, image):
        if self._binding is None:
            raise RuntimeError('Replay original IDs are missing; use bind_replay with the actual selected IDs')
        ids, phase, offset = self._binding
        if offset >= len(ids):
            raise RuntimeError('Replay transform received more images than bound original IDs')
        result = self.streams.transform_one(image, ids[offset], transform_kind='train', phase=phase)
        self._binding = (ids, phase, offset + 1)
        return result


class CommonInputStreams:
    def __init__(self, seed=1993):
        self.seed = _nonnegative_integer(seed, 'seed')
        self.train_transform, self.eval_transform = common_transforms()
        self._phase = None
        self.replay_transform = ReplayTransform(self)

    @property
    def phase(self):
        if self._phase is None:
            raise RuntimeError('Input phase is missing; explicitly coordinate task/epoch/phase/pass before iteration')
        return self._phase

    def set_phase(self, task, epoch, phase, pass_id=0):
        self._phase = InputPhase(task, epoch, phase, pass_id)
        return self._phase

    @contextmanager
    def phase_context(self, task, epoch, phase, pass_id=0):
        before = self._phase
        self.set_phase(task, epoch, phase, pass_id)
        try:
            yield self.phase
        finally:
            self._phase = before

    def transform_one(self, image, original_id, *, transform_kind='train', split='train', phase=None):
        phase = self.phase if phase is None else phase
        if not isinstance(phase, InputPhase):
            raise ValueError('Explicit augmentation coordinates must be an InputPhase')
        original_id = _nonnegative_integer(original_id, 'original_id')
        transform = self.train_transform if transform_kind == 'train' else self.eval_transform if transform_kind == 'eval' else None
        if transform is None or split not in ('train', 'test'):
            raise ValueError('Transform kind must be train/eval and split must be train/test')
        seed = stable_seed(self.seed, 'augmentation', asdict(phase), split, transform_kind, original_id)
        with isolated_cpu_rng(seed):
            return transform(_pil_image(image))

    def transform_batch(self, raw_images, original_ids, *, transform_kind='train', split='train', phase=None):
        """Canonical CPU pixels; callers move the returned batch to a device."""
        ids = _original_ids(original_ids, count=len(raw_images))
        if not len(ids):
            raise ValueError('Cannot transform an empty image batch')
        return torch.stack([self.transform_one(image, original_id, transform_kind=transform_kind, split=split, phase=phase)
                            for image, original_id in zip(raw_images, ids)])

    @contextmanager
    def bind_replay(self, original_ids, *, phase=None):
        ids = _original_ids(original_ids)
        phase = self.phase if phase is None else phase
        if not isinstance(phase, InputPhase):
            raise ValueError('Replay phase must be an InputPhase')
        before = self.replay_transform._binding
        self.replay_transform._binding = (ids, phase, 0)
        try:
            yield self.replay_transform
            if self.replay_transform._binding[2] != len(ids):
                raise RuntimeError('Replay transform consumed fewer images than bound original IDs')
        finally:
            self.replay_transform._binding = before

    def wrap_dataset(self, dataset, *, split, transform_kind=None, include_original_ids=False):
        from datasets.utils.continual_dataset import MammothDatasetWrapper
        if not isinstance(dataset, MammothDatasetWrapper):
            raise TypeError('Common inputs require the native masked MammothDatasetWrapper')
        if isinstance(dataset.dataset, _KeyedDataset):
            raise ValueError('Dataset already has a keyed input adapter; do not wrap it twice')
        if split not in ('train', 'test'):
            raise ValueError('Dataset split must be train/test')
        transform_kind = ('train' if dataset.train else 'eval') if transform_kind is None else transform_kind
        if transform_kind not in ('train', 'eval'):
            raise ValueError('Dataset transform kind must be train/eval')
        ids = _original_ids(dataset.indexes, count=len(dataset), unique=True)
        source = copy.copy(dataset.dataset)
        source.transform = self.train_transform if transform_kind == 'train' else self.eval_transform
        adapter = _KeyedDataset(source, ids, self, split, dataset.train)
        wrapped = MammothDatasetWrapper(adapter, train=dataset.train)
        wrapped.indexes = ids.copy()
        for name in dataset.extra_return_fields:
            wrapped.add_extra_return_field(name, getattr(dataset, name))
        if include_original_ids:
            if 'original_ids' in wrapped.extra_return_fields:
                raise ValueError('original_ids extra field already exists')
            wrapped.add_extra_return_field('original_ids', ids.copy())
        return wrapped


class _KeyedDataset(Dataset):
    def __init__(self, source, original_ids, streams, split, train):
        self._source = source
        self._original_ids = original_ids
        self._streams = streams
        self._split = split
        self._train = train

    def __getattr__(self, name):
        source = self.__dict__.get('_source')
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    def __setattr__(self, name, value):
        source = self.__dict__.get('_source')
        if not name.startswith('_') and source is not None and hasattr(source, name):
            setattr(source, name, value)
        else:
            object.__setattr__(self, name, value)

    def __copy__(self):
        # Native EASE/HiDe stats views shallow-copy wrapper.dataset and change
        # transform; isolate their dataset metadata, while sharing image arrays.
        return type(self)(copy.copy(self._source), self._original_ids.copy(), self._streams, self._split, self._train)

    def __len__(self):
        return len(self._source)

    def __getitem__(self, index):
        index = _nonnegative_integer(index, 'sample_index')
        if index >= len(self):
            raise IndexError(index)
        phase = self._streams.phase
        actual = transform_fingerprint(self._source.transform)['sha256']
        kind = None
        for candidate, transform in (('train', self._streams.train_transform), ('eval', self._streams.eval_transform)):
            if actual == transform_fingerprint(transform)['sha256']:
                kind = candidate
                break
        if kind is None:
            raise ValueError('Dataset changed to an unaudited common transform')
        seed = stable_seed(self._streams.seed, 'augmentation', asdict(phase), self._split, kind, int(self._original_ids[index]))
        with isolated_cpu_rng(seed):
            result = self._source[index]
        if not isinstance(result, (tuple, list)) or len(result) < (3 if self._train else 2):
            raise ValueError('Native dataset train/evaluation return contract was not preserved')
        return result


def _native_view(dataset):
    from datasets.utils.continual_dataset import MammothDatasetWrapper
    if isinstance(dataset, Subset):
        native, positions = _native_view(dataset.dataset)
        selected = _original_ids(dataset.indices, count=len(dataset), unique=True)
        if len(selected) and selected.max() >= len(positions):
            raise ValueError('Subset indexes exceed the native dataset view')
        return native, positions[selected]
    if not isinstance(dataset, MammothDatasetWrapper):
        raise TypeError('Input views must retain the native MammothDatasetWrapper')
    return dataset, np.arange(len(dataset), dtype=np.int64)


def _keyed_view(dataset):
    native, positions = _native_view(dataset)
    if not isinstance(native.dataset, _KeyedDataset):
        raise TypeError('Common-input loaders require a keyed native dataset')
    ids = _original_ids(native.indexes, count=len(native), unique=True)
    if not np.array_equal(ids, native.dataset._original_ids):
        raise ValueError('Wrapper original indexes diverged from the keyed adapter')
    return native, positions


class PhaseSampler(Sampler):
    """Fresh CPU generator per coordinate; batch size never enters the seed."""
    def __init__(self, dataset, *, shuffle):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        _keyed_view(dataset)

    def __len__(self):
        return len(self.dataset)

    def indices(self):
        native, _ = _keyed_view(self.dataset)
        adapter = native.dataset
        phase = adapter._streams.phase
        if not self.shuffle:
            return list(range(len(self)))
        generator = torch.Generator(device='cpu')
        generator.manual_seed(stable_seed(adapter._streams.seed, 'sampler', asdict(phase), adapter._split))
        return torch.randperm(len(self), generator=generator).tolist()

    def __iter__(self):
        return iter(self.indices())


def _worker_cpu_threads(worker_id):
    # Keep PyTorch's native worker default; the runner separately configures
    # the model process to two CPU threads. No extra worker thread pools.
    torch.set_num_threads(1)


def build_loader(dataset, *, batch_size, num_workers=0, shuffle=True, pin_memory=False, collate_fn=None,
                 multiprocessing_context='spawn', timeout=0):
    """Nonpersistent workers and an isolated iterator/base-seed generator.

    Construct after set_phase, and coordinate again before each epoch. Spawned
    workers avoid inheriting active native/OpenMP pools from the two-thread
    model process. The sampler follows the current coordinate. Loader base seeds need no algorithm
    RNG; they may advance across iterations, because keyed augmentation ignores
    worker seeds. Internal method-created loaders require the same treatment.
    """
    batch_size = _nonnegative_integer(batch_size, 'batch_size')
    num_workers = _nonnegative_integer(num_workers, 'num_workers')
    if not batch_size:
        raise ValueError('batch_size must be positive')
    sampler = PhaseSampler(dataset, shuffle=shuffle)
    native, _ = _keyed_view(dataset)
    adapter = native.dataset
    generator = torch.Generator(device='cpu')
    generator.manual_seed(stable_seed(adapter._streams.seed, 'loader', asdict(adapter._streams.phase), adapter._split))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      drop_last=False, pin_memory=pin_memory, persistent_workers=False,
                      generator=generator, worker_init_fn=_worker_cpu_threads, collate_fn=collate_fn,
                      multiprocessing_context=multiprocessing_context if num_workers else None,
                      timeout=timeout if num_workers else 0)


def fingerprint_batch(inputs, labels, original_ids):
    ids = _original_ids(original_ids, count=len(inputs))
    if len(labels) != len(inputs):
        raise ValueError('Batch label count differs from inputs')
    return {'n_samples': len(ids), 'inputs_sha256': _array_digest(inputs), 'labels_sha256': _array_digest(labels),
            'original_ids_sha256': _array_digest(ids), 'original_ids': ids.tolist()}


def fingerprint_dataset(dataset):
    native, positions = _native_view(dataset)
    ids = _original_ids(native.indexes, count=len(native), unique=True)[positions]
    if isinstance(native.dataset, _KeyedDataset):
        _keyed_view(dataset)
    return {'schema': INPUT_SCHEMA, 'n_samples': len(dataset), 'train_contract': bool(native.train),
            'native_wrapper_preserved': True, 'dataset_view': 'subset' if isinstance(dataset, Subset) else 'native',
            'raw_images_sha256': _array_digest(native.data[positions]), 'labels_sha256': _array_digest(np.asarray(native.targets)[positions]),
            'original_ids_sha256': _array_digest(ids), 'original_ids': ids.tolist(),
            'extra_return_fields': list(native.extra_return_fields),
            'transform': transform_fingerprint(native.transform),
            'not_aug_transform': transform_fingerprint(getattr(native, 'not_aug_transform', None)),
            'target_transform': transform_fingerprint(getattr(native, 'target_transform', None))}


def fingerprint_loader(loader, *, require_isolated=True):
    """Inspect actual loader evidence without visiting or augmenting examples."""
    try:
        native, positions = _keyed_view(loader.dataset)
        keyed = True
    except TypeError:
        keyed = False
    dedicated_generator = loader.generator is not None and loader.generator is not torch.default_generator and loader.generator.device.type == 'cpu'
    isolated = keyed and isinstance(loader.sampler, PhaseSampler) and loader.sampler.dataset is loader.dataset and dedicated_generator and not loader.persistent_workers
    if require_isolated and not isolated:
        raise ValueError('Loader lacks explicit common-input sampler/generator/phase isolation')
    result = {'dataset': fingerprint_dataset(loader.dataset), 'batch_size': loader.batch_size,
              'num_workers': loader.num_workers, 'drop_last': loader.drop_last,
              'persistent_workers': loader.persistent_workers, 'isolated': bool(isolated),
              'model_cpu_threads': torch.get_num_threads(),
              'worker_cpu_threads': 1 if loader.num_workers and loader.worker_init_fn is _worker_cpu_threads else None,
              'worker_start_method': loader.multiprocessing_context.get_start_method() if loader.multiprocessing_context else None}
    if keyed:
        adapter = native.dataset
        result['coordinate'] = asdict(adapter._streams.phase)
        result['input_seed'] = adapter._streams.seed
    if isinstance(loader.sampler, PhaseSampler):
        positions = loader.sampler.indices()
        native, view_positions = _keyed_view(loader.dataset)
        ids = np.asarray(native.indexes, dtype=np.int64)[view_positions[positions]]
        result.update(ordered_original_ids=ids.tolist(), ordered_original_ids_sha256=_array_digest(ids),
                      shuffled=loader.sampler.shuffle)
    return result
