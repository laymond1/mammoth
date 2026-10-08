"""CPU-only evidence for common indexed inputs, not accuracy experiments."""
import copy
import gc
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from scripts.unified_cl_inputs import (CommonInputStreams, InputPhase, build_loader,
                                      capture_cpu_rng, common_transforms, cpu_rng_digest,
                                      describe_transform, fingerprint_batch, fingerprint_dataset,
                                      fingerprint_loader, isolated_cpu_rng, stable_seed,
                                      transform_fingerprint)
from datasets.utils.continual_dataset import MammothDatasetWrapper, store_masked_loaders
from utils.augmentations import apply_transform


class SyntheticCIFAR(Dataset):
    def __init__(self, count=73, train=True):
        self.data = np.random.default_rng(921).integers(0, 256, size=(count, 32, 32, 3), dtype=np.uint8)
        self.data[1] = self.data[0]  # Identity must remain the original index.
        self.targets = np.arange(count, dtype=np.int64) % 20
        self.transform = transforms.ToTensor()
        self.not_aug_transform = transforms.ToTensor()
        self.target_transform = None
        self.train = train
        self.consume_source_rng = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        augmented = self.transform(image)
        if self.consume_source_rng:
            random.random()
            np.random.rand()
            torch.rand(1)
        label = self.targets[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        result = (augmented, label, self.not_aug_transform(image)) if self.train else (augmented, label)
        if hasattr(self, 'logits'):
            result += (self.logits[index],)
        return result


@pytest.fixture(autouse=True)
def cpu_native_policy(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Input preparation must not initialize CUDA or download anything')
    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    before_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before_threads)
    gc.collect()


def native_wrapper(count=73, *, train=True):
    return MammothDatasetWrapper(SyntheticCIFAR(count, train=train), train=train)


def keyed_loader(streams, *, batch_size=16, num_workers=0, count=73, train=True):
    wrapped = streams.wrap_dataset(native_wrapper(count, train=train), split='train' if train else 'test',
                                   include_original_ids=True)
    return build_loader(wrapped, batch_size=batch_size, num_workers=num_workers, shuffle=train, timeout=20)


def collect(loader):
    images, labels, ids = [], [], []
    for batch in loader:
        images.append(batch[0])
        labels.append(batch[1])
        ids.append(batch[-1])
    return torch.cat(images), torch.cat(labels), torch.cat(ids)


def restore_cpu_rng(snapshot):
    random.setstate(snapshot['python'])
    np.random.set_state(snapshot['numpy'])
    torch.default_generator.set_state(snapshot['torch'])


def draw_algorithm():
    return random.random(), float(np.random.rand()), torch.rand(3)


def test_isolated_rng_preserves_all_cpu_states_and_never_touches_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Only the CPU default_generator may be seeded/accessed')
    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    for name in ('manual_seed', 'manual_seed_all', 'seed', 'seed_all', 'get_rng_state', 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    before = cpu_rng_digest()
    with isolated_cpu_rng(1234):
        first = draw_algorithm()
    assert cpu_rng_digest() == before
    with isolated_cpu_rng(1234):
        second = draw_algorithm()
        inside = cpu_rng_digest()
        with isolated_cpu_rng(9876):
            draw_algorithm()
        assert cpu_rng_digest() == inside
    assert first[:2] == second[:2] and torch.equal(first[2], second[2])
    assert cpu_rng_digest() == before
    with pytest.raises(RuntimeError, match='source failed'):
        with isolated_cpu_rng(55):
            draw_algorithm()
            raise RuntimeError('source failed')
    assert cpu_rng_digest() == before


@pytest.mark.parametrize('workers', [0, 2])
def test_batch_16_64_and_worker_count_have_identical_order_and_pixels(workers):
    streams = CommonInputStreams()
    streams.set_phase(0, 0, 'train')
    before = cpu_rng_digest()
    reference = collect(keyed_loader(streams, batch_size=16, num_workers=0))
    candidate = collect(keyed_loader(streams, batch_size=64, num_workers=workers))
    assert all(torch.equal(left, right) for left, right in zip(reference, candidate))
    assert cpu_rng_digest() == before
    assert len(reference[2]) == 73 and len(set(reference[2].tolist())) == 73
    assert not torch.equal(reference[2], torch.arange(73))
    # Repeated iterators also keep exactly the same sample order/pixels.
    repeated = collect(keyed_loader(streams, batch_size=16, num_workers=workers))
    assert all(torch.equal(left, right) for left, right in zip(reference, repeated))


@pytest.mark.parametrize('workers', [0, 2])
def test_algorithm_rng_advances_naturally_between_batches(workers):
    streams = CommonInputStreams()
    streams.set_phase(1, 3, 'train')
    wrapped = streams.wrap_dataset(native_wrapper(19), split='train', include_original_ids=True)
    wrapped.dataset.consume_source_rng = True
    loader = build_loader(wrapped, batch_size=16, num_workers=workers, timeout=20)
    before = capture_cpu_rng()
    expected = [draw_algorithm() for _ in range(len(loader))]
    expected_final = cpu_rng_digest()
    restore_cpu_rng(before)
    actual = [draw_algorithm() for batch in loader]
    assert cpu_rng_digest() == expected_final
    for left, right in zip(expected, actual):
        assert left[:2] == right[:2] and torch.equal(left[2], right[2])


@pytest.mark.parametrize('workers', [0, 2])
def test_svd_auxiliary_pass_does_not_advance_main_stream_or_algorithm_rng(workers):
    streams = CommonInputStreams()
    streams.set_phase(0, 2, 'train')
    main = keyed_loader(streams, count=19, num_workers=workers)
    first = collect(main)
    before_rng = cpu_rng_digest()
    with streams.phase_context(0, 2, 'svd', pass_id=7):
        auxiliary = collect(keyed_loader(streams, count=19, num_workers=workers))
        assert fingerprint_loader(main)['coordinate']['phase'] == 'svd'
    assert streams.phase == InputPhase(0, 2, 'train', 0)
    after = collect(main)
    assert all(torch.equal(left, right) for left, right in zip(first, after))
    assert cpu_rng_digest() == before_rng
    assert not torch.equal(first[0][torch.argsort(first[2])], auxiliary[0][torch.argsort(auxiliary[2])])


def test_explicit_coordinates_are_required_and_task_epoch_pass_change_seeds():
    streams = CommonInputStreams()
    dataset = streams.wrap_dataset(native_wrapper(3), split='train')
    with pytest.raises(RuntimeError, match='phase is missing'):
        build_loader(dataset, batch_size=2)
    with pytest.raises(RuntimeError, match='phase is missing'):
        dataset[0]
    seeds = {stable_seed(1993, 'augmentation', {'task': t, 'epoch': e, 'phase': p, 'pass_id': n}, 1)
             for t, e, p, n in [(0, 0, 'train', 0), (1, 0, 'train', 0), (0, 1, 'train', 0), (0, 0, 'svd', 0), (0, 0, 'train', 1)]}
    assert len(seeds) == 5


def test_phase_coordination_preserves_natural_algorithm_rng_progression():
    streams = CommonInputStreams()
    streams.set_phase(0, 0, 'train')
    before = capture_cpu_rng()
    expected = draw_algorithm()
    expected_after = cpu_rng_digest()
    restore_cpu_rng(before)
    with streams.phase_context(0, 0, 'svd'):
        actual = draw_algorithm()  # SVD/model randomness stays on the natural algorithm stream.
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert cpu_rng_digest() == expected_after and expected_after != cpu_rng_digest(before)


def test_native_subset_views_keep_original_ids_and_explicit_statistics_phase():
    streams = CommonInputStreams()
    streams.set_phase(1, 0, 'statistics')
    wrapped = streams.wrap_dataset(native_wrapper(19), split='train', transform_kind='eval', include_original_ids=True)
    view = Subset(Subset(wrapped, [3, 8, 11, 16]), [1, 2])
    loader = build_loader(view, batch_size=16, shuffle=False)
    before = cpu_rng_digest()
    output = collect(loader)
    evidence = fingerprint_loader(loader)
    assert output[2].tolist() == evidence['ordered_original_ids'] == [8, 11]
    assert evidence['dataset']['dataset_view'] == 'subset' and evidence['dataset']['native_wrapper_preserved']
    assert evidence['dataset']['transform'] == transform_fingerprint(streams.eval_transform)
    assert cpu_rng_digest() == before
    wrapped.indexes = wrapped.indexes + 100
    with pytest.raises(ValueError, match='diverged'):
        fingerprint_loader(loader)


def test_duplicate_images_are_keyed_by_original_indexes_and_replay_pixels_match():
    streams = CommonInputStreams()
    streams.set_phase(0, 0, 'train')
    dataset = streams.wrap_dataset(native_wrapper(3), split='train', include_original_ids=True)
    left, right = dataset[0], dataset[1]
    assert torch.equal(left[2], right[2]) and left[-1] != right[-1]
    assert not torch.equal(left[0], right[0])
    before = cpu_rng_digest()
    replay = streams.transform_batch(torch.stack([left[2], right[2]]), [0, 1])
    assert replay.device.type == 'cpu' and torch.equal(replay, torch.stack([left[0], right[0]]))
    assert cpu_rng_digest() == before
    # Reordered replay selection keeps identity with its actual chosen slot.
    with streams.bind_replay([1, 0]) as transform:
        reverse = apply_transform(torch.stack([right[2], left[2]]), transform)
    assert torch.equal(reverse, replay.flip(0)) and cpu_rng_digest() == before


def test_replay_rejects_missing_ids_and_incomplete_or_excess_binding():
    streams = CommonInputStreams()
    streams.set_phase(0, 0, 'replay')
    image = torch.zeros(3, 32, 32)
    with pytest.raises(RuntimeError, match='IDs are missing'):
        streams.replay_transform(image)
    with pytest.raises(RuntimeError, match='fewer images'):
        with streams.bind_replay([2, 3]) as transform:
            transform(image)
    with pytest.raises(RuntimeError, match='more images'):
        with streams.bind_replay([2]) as transform:
            transform(image)
            transform(image)
    with pytest.raises(RuntimeError, match='IDs are missing'):
        streams.replay_transform(image)
    with streams.bind_replay([2]) as outer:
        with streams.bind_replay([3]) as inner:
            inner(image)
        outer(image)


def test_native_wrapper_preserves_extra_fields_labels_indexes_non_aug_and_logits():
    source = SyntheticCIFAR(30)
    source.logits = torch.arange(30 * 4).reshape(30, 4)
    native = MammothDatasetWrapper(source, train=True)
    original_labels = np.arange(30)
    native.add_extra_return_field('true_labels', original_labels)
    positions = np.array([3, 8, 11])
    native.data = native.data[positions]
    native.targets = native.targets[positions] + 40
    native.logits = native.logits[positions]
    native.indexes = native.indexes[positions]
    streams = CommonInputStreams()
    streams.set_phase(1, 0, 'train')
    wrapped = streams.wrap_dataset(native, split='train', include_original_ids=True)
    assert isinstance(wrapped, MammothDatasetWrapper) and wrapped.train
    assert wrapped.extra_return_fields == ('true_labels', 'original_ids')
    assert np.array_equal(wrapped.indexes, positions)
    result = wrapped[1]
    assert len(result) == 6
    assert result[1] == native.targets[1] and result[4] == 8 and result[5] == 8
    assert torch.equal(result[2], transforms.ToTensor()(Image.fromarray(source.data[1])))
    assert torch.equal(result[3], source.logits[1])
    assert native.transform is not wrapped.transform  # Original metadata remains untouched.


def test_real_store_masked_loaders_class_permutation_and_task_split_are_retained():
    args = SimpleNamespace(permute_classes=True, class_order=np.roll(np.arange(20), 3), validation=False,
                           validation_mode='current', noise_rate=0, label_perc=1, label_perc_by_class=1,
                           batch_size=16, num_workers=0, seed=1993, drop_last=False)
    setting = SimpleNamespace(SETTING='class-il', c_task=-1, args=args, N_CLASSES=20,
                              N_CLASSES_PER_TASK=10, test_loaders=[], get_offsets=lambda: (0, 10))
    train, test = store_masked_loaders(SyntheticCIFAR(40), SyntheticCIFAR(40, train=False), setting)
    streams = CommonInputStreams()
    streams.set_phase(0, 0, 'train')
    wrapped = streams.wrap_dataset(train.dataset, split='train', include_original_ids=True)
    assert np.array_equal(wrapped.indexes, train.dataset.indexes)
    assert torch.equal(wrapped.targets, train.dataset.targets)
    assert bool(((wrapped.targets >= 0) & (wrapped.targets < 10)).all())
    assert collect(build_loader(wrapped, batch_size=16))[2].sort().values.tolist() == sorted(train.dataset.indexes.tolist())
    with streams.phase_context(0, 0, 'evaluation'):
        eval_wrapper = streams.wrap_dataset(test.dataset, split='test', include_original_ids=True)
        assert not eval_wrapper.train and len(eval_wrapper[0]) == 3


def test_statistics_shallow_copy_does_not_change_main_transform_or_images():
    streams = CommonInputStreams()
    streams.set_phase(0, 0, 'train')
    wrapped = streams.wrap_dataset(native_wrapper(4), split='train')
    original = wrapped[0][0]
    view = copy.copy(wrapped)
    object.__setattr__(view, 'dataset', copy.copy(wrapped.dataset))
    view.transform = streams.eval_transform
    assert wrapped.transform is streams.train_transform and view.transform is streams.eval_transform
    assert view.data is wrapped.data
    with streams.phase_context(0, 0, 'statistics'):
        eval_image = view[0][0]
        assert fingerprint_dataset(view)['transform'] == transform_fingerprint(streams.eval_transform)
    assert torch.equal(wrapped[0][0], original)
    assert not torch.equal(eval_image, original)


def test_fingerprints_audit_actual_all_transform_paths_without_consuming_rng():
    streams = CommonInputStreams()
    streams.set_phase(1, 4, 'train')
    loader = keyed_loader(streams, count=19)
    before = cpu_rng_digest()
    evidence = fingerprint_loader(loader)
    replay = transform_fingerprint(streams.replay_transform)
    assert cpu_rng_digest() == before
    assert evidence['isolated'] and evidence['drop_last'] is False and evidence['persistent_workers'] is False
    assert evidence['coordinate'] == {'task': 1, 'epoch': 4, 'phase': 'train', 'pass_id': 0}
    assert len(evidence['ordered_original_ids']) == 19
    assert replay['description']['recipe'] == evidence['dataset']['transform']['description']
    assert replay['description']['original_ids_required'] is True
    batch = next(iter(loader))
    actual = fingerprint_batch(batch[0], batch[1], batch[-1])
    assert actual['original_ids'] == evidence['ordered_original_ids'][:16]
    changed = batch[0].clone()
    changed[0, 0, 0, 0] += .25
    assert fingerprint_batch(changed, batch[1], batch[-1])['inputs_sha256'] != actual['inputs_sha256']
    class Unrecorded:
        def __call__(self, image):
            return image
    with pytest.raises(ValueError, match='Unaudited input transform'):
        describe_transform(Unrecorded())
    loader.dataset.transform = transforms.Compose([transforms.ToTensor()])
    with pytest.raises(ValueError, match='unaudited common transform'):
        loader.dataset[0]
    unisolated = DataLoader(loader.dataset, batch_size=16, shuffle=False)
    with pytest.raises(ValueError, match='lacks explicit'):
        fingerprint_loader(unisolated)
    assert fingerprint_loader(unisolated, require_isolated=False)['isolated'] is False


@pytest.mark.parametrize('ids', [[0, 0, 1], [-1, 1, 2], [0., 1., 2.]])
def test_invalid_dataset_original_indexes_cannot_claim_keyed_stream(ids):
    streams = CommonInputStreams()
    wrapped = native_wrapper(3)
    wrapped.indexes = np.array(ids)
    with pytest.raises(ValueError, match='Original indexes|original indexes'):
        streams.wrap_dataset(wrapped, split='train')


def test_common_recipe_is_exactly_approved_and_plain_torchvision():
    train, evaluation = common_transforms()
    description = describe_transform(train)['transforms']
    assert [item['operation'] for item in description] == ['RandomResizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
    assert description[0] == {'operation': 'RandomResizedCrop', 'size': [224, 224], 'scale': [.08, 1.],
                              'ratio': [.75, 4. / 3.], 'interpolation': 'BICUBIC', 'antialias': True}
    assert description[1]['p'] == .5
    assert description[-1]['mean'] == [.5, .5, .5] and description[-1]['std'] == [.5, .5, .5]
    assert [item['operation'] for item in describe_transform(evaluation)['transforms']] == ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']
    with pytest.raises(TypeError, match='native masked'):
        CommonInputStreams().wrap_dataset(SyntheticCIFAR(3), split='train')
