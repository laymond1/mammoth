"""Reference integration checks using fake modules and tiny raw image arrays."""

import importlib.util
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


_PATH = Path(__file__).resolve().parents[1] / 'scripts/ease_reference_utils.py'
_SPEC = importlib.util.spec_from_file_location('_ease_reference_helpers', _PATH)
helpers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helpers)


@pytest.fixture
def fake_reference(tmp_path):
    data_root = tmp_path / 'data'
    cifar = data_root / 'CIFAR100' / 'cifar-100-python'
    cifar.mkdir(parents=True)
    (cifar / 'train').write_bytes(b'existing CIFAR fixture')
    checkpoint = tmp_path / 'checkpoint.pth'
    torch.save({'cls_token': torch.tensor([2.0]), 'head.weight': torch.tensor([99.0])}, checkpoint)
    images = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    labels = np.array([1, 0, 1, 0])

    class FakeCifar:
        def download_data(self):
            pytest.fail('The reference original downloader must not run')

    class FakeManager:
        dataset_name = 'cifar224'

        def __init__(self):
            self._train_data = images
            self._train_targets = labels
            self._test_data = images + 1
            self._test_targets = labels

        def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
            data = self._train_data if source == 'train' else self._test_data
            targets = self._train_targets if source == 'train' else self._test_targets
            ids = np.concatenate([np.flatnonzero(targets == label) for label in indices])
            result = SimpleNamespace(images=data[ids], labels=targets[ids],
                                     trsf=f'{source}/{mode}', use_path=False)
            return (result.images, result.labels, result) if ret_data else result

    class FakeTransformer:
        def __init__(self, missing=(), unexpected=()):
            self.missing = list(missing)
            self.unexpected = list(unexpected)

        def load_state_dict(self, state, strict=True, assign=False):
            return SimpleNamespace(missing_keys=self.missing, unexpected_keys=self.unexpected)

    pretrained = SimpleNamespace(load_state_dict=Mock(), load_pretrained=Mock())
    create = Mock(return_value=pretrained)
    cifar_loader = Mock(side_effect=lambda root, train, download:
                        SimpleNamespace(data=images if train else images + 1, targets=labels.tolist()))
    stack = {
        'timm': SimpleNamespace(create_model=create),
        'data': SimpleNamespace(iCIFAR224=FakeCifar,
                                datasets=SimpleNamespace(cifar=SimpleNamespace(CIFAR100=cifar_loader))),
        'data_manager': SimpleNamespace(DataManager=FakeManager),
        'vit': SimpleNamespace(VisionTransformer=FakeTransformer),
        'ease': SimpleNamespace(num_workers=8, DataLoader=torch.utils.data.DataLoader),
    }
    return SimpleNamespace(stack=stack, checkpoint=checkpoint, data_root=data_root, images=images,
                           labels=labels, pretrained=pretrained, create=create, cifar_loader=cifar_loader)


def test_exact_tagged_checkpoint_factory_disables_download_and_loads_full_backbone(fake_reference):
    fixture = fake_reference
    records = {}
    original_factory = fixture.stack['timm'].create_model
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, records):
        result = fixture.stack['timm'].create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        assert result is fixture.pretrained
        fixture.create.assert_called_once_with(helpers.BASE_TAG, pretrained=False, num_classes=0)
        weights = fixture.pretrained.load_state_dict.call_args.args[0]
        assert set(weights) == {'cls_token'}
        torch.testing.assert_close(weights['cls_token'], torch.tensor([2.0]))
        assert fixture.pretrained.load_state_dict.call_args.kwargs == {'strict': True}
        assert records['checkpoint']['sha256'] == helpers.file_sha256(fixture.checkpoint)
        assert records['checkpoint_loads'] == [{'requested_alias': 'vit_base_patch16_224', 'resolved_tag': helpers.BASE_TAG}]
    assert fixture.stack['timm'].create_model is original_factory


def test_shared_cifar_loading_uses_existing_dataset_and_restores_worker_settings(fake_reference):
    fixture = fake_reference
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, {}):
        subject = fixture.stack['data'].iCIFAR224()
        subject.download_data()
        assert fixture.stack['ease'].num_workers == 4
        assert fixture.cifar_loader.call_count == 2
        first, second = fixture.cifar_loader.call_args_list
        assert first.args == (str(fixture.data_root / 'CIFAR100'),)
        assert first.kwargs == {'train': True, 'download': False}
        assert second.kwargs == {'train': False, 'download': False}
        np.testing.assert_array_equal(subject.train_data, fixture.images)
        np.testing.assert_array_equal(subject.test_data, fixture.images + 1)
        np.testing.assert_array_equal(subject.train_targets, fixture.labels)
    assert fixture.stack['ease'].num_workers == 8
    assert fixture.stack['ease'].DataLoader is torch.utils.data.DataLoader


@pytest.mark.parametrize('drop_last', [False, True])
def test_reference_shuffled_batches_match_native_seeded_loader_across_epochs(fake_reference, drop_last):
    from utils.conf import create_seeded_dataloader

    fixture = fake_reference
    args = SimpleNamespace(seed=1993, num_workers=0)
    native = create_seeded_dataloader(args, list(range(23)), batch_size=5, shuffle=True,
                                     drop_last=drop_last, non_verbose=True)
    records = {}
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, records):
        reference = fixture.stack['ease'].DataLoader(list(range(23)), batch_size=5,
                                                     shuffle=True, num_workers=0, drop_last=drop_last)
        observed = []
        for _ in range(2):
            native_batches = [batch.tolist() for batch in native]
            reference_batches = [batch.tolist() for batch in reference]
            assert reference_batches == native_batches
            assert sum(map(len, reference_batches)) == (20 if drop_last else 23)
            observed.append(reference_batches)
        assert observed[0] != observed[1]
        assert reference.drop_last is drop_last
        assert records['loaders'][0]['shuffle'] is True
        assert records['loaders'][0]['drop_last'] is drop_last
        assert records['loader_seed_policy']['seed'] == 1993


def test_worker_python_numpy_seeding_matches_native_without_changing_torch_seed(fake_reference):
    from utils.conf import worker_init_fn

    fixture = fake_reference
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, {}):
            # Constructing a loader does not spawn its workers or run an image.
            reference = fixture.stack['ease'].DataLoader([0], num_workers=4)
            reference.worker_init_fn(2)
            ref_python, ref_numpy = random.random(), np.random.random(3)
            worker_init_fn(2, num_workers=4, seed=1993)
            assert random.random() == ref_python
            np.testing.assert_array_equal(np.random.random(3), ref_numpy)
            assert ref_python == random.Random(1999).random()
            np.testing.assert_array_equal(ref_numpy, np.random.RandomState(1999).random_sample(3))
            torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def test_reference_prototype_loader_keeps_original_shuffle_and_tail_defaults(fake_reference):
    fixture = fake_reference
    records = {}
    class PrototypeRecords(list):
        _ease_reference_source = 'train'
        _ease_reference_mode = 'test'
    dataset = PrototypeRecords([0, 1])
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, records):
        prototype = fixture.stack['ease'].DataLoader(dataset, batch_size=48, shuffle=True, num_workers=4)
        assert isinstance(prototype.sampler, torch.utils.data.RandomSampler)
        assert prototype.drop_last is False
        assert records['loaders'][0]['source'] == 'train' and records['loaders'][0]['mode'] == 'test'
        assert records['loaders'][0]['shuffle'] is True and records['loaders'][0]['drop_last'] is False
        assert 'shuffle=True' in records['loader_seed_policy']['prototype_order_difference']


@pytest.mark.parametrize('source,mode,ret_data', [('train', 'train', False), ('train', 'test', True), ('test', 'test', True)])
def test_class_major_dataset_is_reordered_to_native_global_ids_without_changing_samples(fake_reference, source, mode, ret_data):
    fixture = fake_reference
    records = {}
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, records):
        manager = fixture.stack['data_manager'].DataManager()
        result = manager.get_dataset([0, 1], source, mode, ret_data=ret_data)
        dataset = result[2] if ret_data else result
        expected_images = fixture.images if source == 'train' else fixture.images + 1
        np.testing.assert_array_equal(dataset.images, expected_images)
        np.testing.assert_array_equal(dataset.labels, fixture.labels)
        np.testing.assert_array_equal(dataset.global_ids, [0, 1, 2, 3])
        assert dataset.trsf == f'{source}/{mode}' and dataset.use_path is False
        if ret_data:
            assert result[0] is dataset.images and result[1] is dataset.labels
        report = records['datasets'][0]
        assert report['source'] == source and report['mode'] == mode
        assert report['original_ordered_sha256'] != report['samples_ordered_sha256']
        native = helpers.dataset_fingerprint(expected_images, fixture.labels, [0, 1, 2, 3], dataset.trsf)
        assert report['samples_ordered_sha256'] == native['samples_ordered_sha256']
        assert report['samples_unordered_sha256'] == native['samples_unordered_sha256']


def test_selected_classes_retain_original_global_indices(fake_reference):
    fixture = fake_reference
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, {}):
        dataset = fixture.stack['data_manager'].DataManager().get_dataset([0], 'train', 'test')
        np.testing.assert_array_equal(dataset.images, fixture.images[[1, 3]])
        np.testing.assert_array_equal(dataset.labels, [0, 0])
        np.testing.assert_array_equal(dataset.global_ids, [1, 3])


@pytest.mark.parametrize('kwargs', [{'appendent': (np.zeros((1, 2, 2, 3)), [0])}, {'m_rate': 0.0}, {'m_rate': 0.5}])
def test_reference_protocol_rejects_replay_or_sample_removal(fake_reference, kwargs):
    fixture = fake_reference
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, {}):
        with pytest.raises(ValueError, match='forbids replay'):
            fixture.stack['data_manager'].DataManager().get_dataset([0], 'train', 'train', **kwargs)


def test_reference_pretrained_conversion_accepts_only_missing_adapter_keys(fake_reference):
    fixture = fake_reference
    records = {}
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, records):
        model = fixture.stack['vit'].VisionTransformer(missing=['cur_adapter.0.down_proj.weight'])
        result = model.load_state_dict({}, strict=False)
        assert result.missing_keys == ['cur_adapter.0.down_proj.weight']
        assert records['converted_checkpoint_audits'][0]['unexpected_keys'] == []


@pytest.mark.parametrize('missing,unexpected', [(['blocks.0.norm1.weight'], []), ([], ['bad.weight'])])
def test_reference_pretrained_conversion_rejects_missing_frozen_or_unexpected_keys(fake_reference, missing, unexpected):
    fixture = fake_reference
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, {}):
        model = fixture.stack['vit'].VisionTransformer(missing=missing, unexpected=unexpected)
        with pytest.raises(RuntimeError, match='backbone keys'):
            model.load_state_dict({}, strict=False)


def test_reference_factory_rejects_other_family_and_classifier_weights(fake_reference):
    fixture = fake_reference
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, {}):
        with pytest.raises(ValueError, match='selected Base'):
            fixture.stack['timm'].create_model('vit_base_patch16_224_in21k', pretrained=True)
        with pytest.raises(ValueError, match='classifier-free'):
            fixture.stack['timm'].create_model('vit_base_patch16_224', num_classes=1000)
        fixture.create.assert_not_called()


def test_fingerprints_separate_sample_order_from_selected_multiset():
    images = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    labels, global_ids = np.array([0, 1, 0]), np.array([7, 2, 4])
    first = helpers.dataset_fingerprint(images, labels, global_ids, 'test-transform')
    order = np.array([2, 0, 1])
    second = helpers.dataset_fingerprint(images[order], labels[order], global_ids[order], 'test-transform')
    assert first['samples_ordered_sha256'] != second['samples_ordered_sha256']
    assert first['samples_unordered_sha256'] == second['samples_unordered_sha256']
    assert first['class_counts'] == {'0': 2, '1': 1}
    assert first['transform_repr'] == repr('test-transform')


@pytest.mark.parametrize('changed', ['image', 'label', 'global_id'])
def test_fingerprint_detects_image_label_or_identity_changes(changed):
    images = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    labels, ids = np.array([0, 1]), np.array([1, 2])
    original = helpers.dataset_fingerprint(images, labels, ids)
    if changed == 'image':
        images[0, 0, 0, 0] = 1
    elif changed == 'label':
        labels[0] = 2
    else:
        ids[0] = 3
    modified = helpers.dataset_fingerprint(images, labels, ids)
    assert original['samples_ordered_sha256'] != modified['samples_ordered_sha256']
    assert original['samples_unordered_sha256'] != modified['samples_unordered_sha256']


def test_duplicate_global_ids_are_rejected():
    with pytest.raises(ValueError, match='unique'):
        helpers.dataset_fingerprint(np.zeros((2, 2, 2, 3), dtype=np.uint8), [0, 0], [1, 1])


def test_raw_shared_cifar_metadata_records_file_hashes_counts_and_class_names(tmp_path):
    root = tmp_path / 'CIFAR100' / 'cifar-100-python'
    root.mkdir(parents=True)
    for name, labels in [('train', [0, 1, 0]), ('test', [0, 1])]:
        with (root / name).open('wb') as stream:
            pickle.dump({'data': np.zeros((len(labels), 3072), dtype=np.uint8), 'fine_labels': labels}, stream)
    with (root / 'meta').open('wb') as stream:
        pickle.dump({'fine_label_names': ['first', 'second']}, stream)
    record = helpers.shared_cifar_metadata(tmp_path)
    assert record['dataset'] == 'CIFAR100'
    assert record['files']['train']['n_samples'] == 3
    assert record['files']['train']['class_counts'] == {'0': 2, '1': 1}
    assert record['files']['test']['n_samples'] == 2
    assert record['files']['meta']['fine_label_names'] == ['first', 'second']
    for name in ('train', 'test', 'meta'):
        assert record['files'][name]['sha256'] == helpers.file_sha256(root / name)
    assert 'original index' in record['global_id_policy']


@pytest.mark.parametrize('size,dimension,heads,width', [('small', 384, 6, 64), ('tiny', 192, 3, 64),
                                                      ('small', 384, 6, 32), ('tiny', 192, 3, 16)])
def test_explicit_size_patch_wires_the_local_tag_and_restores_both_author_constructors(fake_reference, monkeypatch, size, dimension, heads, width):
    fixture = fake_reference
    original_factory, original_get_backbone = Mock(), Mock()
    factory, get_backbone, reweight = Mock(), Mock(), Mock()

    class Classifier:
        def forward_reweight(self, features, cur_task, out_dim=768):
            return features

    original_reweight = Classifier.forward_reweight
    fixture.stack['vit'].vit_base_patch16_224_ease = original_factory
    fixture.stack['inc_net'] = SimpleNamespace(get_backbone=original_get_backbone, CosineLinear=Classifier)
    specialize = Mock(side_effect=[(factory, {'feature_dim_constants': 9}), (get_backbone, {'feature_dim_constants': 5}),
                                  (reweight, {'feature_dim_constants': 1})])
    monkeypatch.setattr(helpers, '_size_constructor', specialize)
    variant = {'name': f'{size}_{width}', 'vit_size': size, 'feature_dim': dimension, 'num_heads': heads,
               'adapter_size': width, 'checkpoint_file': str(fixture.checkpoint),
               'checkpoint_sha256': helpers.file_sha256(fixture.checkpoint),
               'checkpoint_tag': f'vit_{size}_patch16_224.augreg_in21k_ft_in1k', 'canonical_pretrained_sha256': '0' * 64}
    records = {}
    with helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, records, variant=variant):
        assert fixture.stack['vit'].vit_base_patch16_224_ease is factory
        assert fixture.stack['inc_net'].get_backbone is get_backbone
        assert Classifier.forward_reweight is reweight
        fixture.stack['timm'].create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        fixture.create.assert_called_once_with(variant['checkpoint_tag'], pretrained=False, num_classes=0)
        assert records['reference_kind'] == 'author-code-size-variant'
        assert records['variant']['adapter_size'] == width
        assert records['checkpoint']['tag'] == variant['checkpoint_tag']
        assert records['constructor_specialization']['classifier_reweight']['feature_dim_constants'] == 1
        with pytest.raises(ValueError, match=f'selected {size.title()}'):
            fixture.stack['timm'].create_model('vit_base_patch16_224_in21k', pretrained=True)
    assert fixture.stack['vit'].vit_base_patch16_224_ease is original_factory
    assert fixture.stack['inc_net'].get_backbone is original_get_backbone
    assert Classifier.forward_reweight is original_reweight
    assert specialize.call_args_list[0].args == (original_factory, dimension, heads)
    assert specialize.call_args_list[1].args == (original_get_backbone, dimension)
    assert specialize.call_args_list[2].args == (original_reweight, dimension)


def test_explicit_reference_size_rejects_mismatched_checkpoint_before_patching(fake_reference):
    fixture = fake_reference
    variant = {'name': 'tiny_relative16', 'vit_size': 'tiny', 'feature_dim': 192, 'num_heads': 3, 'adapter_size': 16,
               'checkpoint_file': str(fixture.checkpoint), 'checkpoint_sha256': '0' * 64,
               'checkpoint_tag': 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'}
    original = fixture.stack['timm'].create_model
    with pytest.raises(ValueError, match='differs from its pin'):
        helpers.install_reference_patches(fixture.stack, fixture.checkpoint, fixture.data_root, {}, variant=variant)
    assert fixture.stack['timm'].create_model is original
