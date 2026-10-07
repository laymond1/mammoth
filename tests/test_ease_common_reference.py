"""Common-input reference guards, actual batches and immutable initialization."""
import json
from contextlib import ExitStack
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from scripts import ease_common_reference_utils as helpers
from scripts import run_ease_common_reference as runner
from scripts.ease_fullcl_protocol import configure_epoch_loader, phase_seed


def test_dry_default_does_not_read_credentials_or_launch(monkeypatch, capsys):
    monkeypatch.setattr(runner, 'execute', lambda *args: pytest.fail('Dry run executed'))
    assert runner.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['executed'] is False
    assert result['protocol']['batch_size'] == 64


def test_prepare_and_execute_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match='separate CPU'):
        runner.main(['--execute', '--prepare-initial-state', str(tmp_path / 'init.pt')])


def test_worker_cannot_bypass_execute(tmp_path, monkeypatch):
    monkeypatch.delenv('CL_EASE_COMMON_REFERENCE_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit execution'):
        runner.worker(tmp_path / 'absent.json')


@pytest.mark.parametrize('train', [False, True])
def test_common_transform_pixels_exactly_match_pinned_hide_yaml(train):
    import yaml
    from datasets.utils.continual_dataset import build_torchvision_transform
    config = yaml.safe_load((runner.ROOT / 'datasets/configs/seq-cifar100-224/hide_prompt.yaml').read_text())
    native = build_torchvision_transform(config['TRANSFORM' if train else 'TEST_TRANSFORM'])
    train_list, test_list, common = helpers.common_transforms()
    reference = transforms.Compose((train_list if train else test_list) + common)
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    torch.manual_seed(1993)
    actual = reference(image)
    torch.manual_seed(1993)
    expected = native(image)
    assert torch.equal(actual, expected)
    assert repr(reference) == repr(native)


def test_seeded_common_batches_keep_tail_and_repeat_actual_augmentations():
    class Images(torch.utils.data.Dataset):
        def __init__(self):
            self.transform = transforms.Compose(sum((helpers.common_transforms()[0], helpers.common_transforms()[2]), []))
        def __len__(self):
            return 11
        def __getitem__(self, index):
            image = Image.fromarray(np.full((32, 32, 3), index * 17, dtype=np.uint8))
            return self.transform(image), index
    first = torch.utils.data.DataLoader(Images(), batch_size=4, shuffle=True, num_workers=0)
    second = torch.utils.data.DataLoader(Images(), batch_size=4, shuffle=True, num_workers=0)
    configure_epoch_loader(first, 1993, 0, 'train', 0)
    configure_epoch_loader(second, 1993, 0, 'train', 0)
    torch.manual_seed(1993)
    left = list(first)
    torch.manual_seed(1993)
    right = list(second)
    assert [len(batch[1]) for batch in left] == [4, 4, 3]
    assert all(torch.equal(a, b) for left_batch, right_batch in zip(left, right) for a, b in zip(left_batch, right_batch))


def tiny_catalog(tmp_path):
    from models.ease_utils.vit import ParallelAdapter
    from models.ease_utils.classifier import CosineHead
    return helpers.export_initialization_catalog(lambda: ParallelAdapter(768, 64),
        lambda: CosineHead(768, 10), tmp_path / 'init.pt')


def test_catalog_all_tasks_reproducible_and_preserves_outer_rng(tmp_path):
    random.seed(41); np.random.seed(42); torch.manual_seed(43)
    python, numpy, tensor = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    catalog = tiny_catalog(tmp_path)
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.random.get_rng_state(), tensor)
    assert helpers.validate_initialization_catalog(catalog) is catalog
    assert len(catalog['tasks']) == 10
    assert catalog['tasks']['0']['seed'] == phase_seed(1993, 0, 'task_begin')
    assert not torch.equal(catalog['tasks']['0']['adapter']['0.down_proj.weight'],
                           catalog['tasks']['1']['adapter']['0.down_proj.weight'])
    loaded = torch.load(tmp_path / 'init.pt', map_location='cpu', weights_only=True)
    assert helpers.tensor_digest(loaded['tasks']['9']['adapter']) == helpers.tensor_digest(catalog['tasks']['9']['adapter'])


@pytest.mark.parametrize('corruption', ['missing_task', 'wrong_seed', 'wrong_shape', 'nonfinite', 'wrong_dtype', 'unknown_head'])
def test_catalog_corruption_fails_closed(tmp_path, corruption):
    catalog = tiny_catalog(tmp_path)
    row = catalog['tasks']['0']
    if corruption == 'missing_task':
        catalog['tasks'].pop('9')
    elif corruption == 'wrong_seed':
        row['seed'] += 1
    elif corruption == 'wrong_shape':
        row['adapter']['0.up_proj.weight'] = torch.zeros(768, 63)
    elif corruption == 'nonfinite':
        row['proxy_head']['weight'][0, 0] = float('nan')
    elif corruption == 'wrong_dtype':
        row['proxy_head']['sigma'] = row['proxy_head']['sigma'].double()
    elif corruption == 'unknown_head':
        row['proxy_head']['bias'] = torch.zeros(10)
    with pytest.raises(ValueError, match='Initialization'):
        helpers.validate_initialization_catalog(catalog)


def test_actual_firstbatch_audit_changes_with_gradient_or_pixels():
    head = torch.nn.Linear(3, 2)
    inputs, labels = torch.tensor([[1., 2., 3.]]), torch.tensor([1])
    torch.nn.functional.cross_entropy(head(inputs), labels).backward()
    audit = helpers.first_batch_audit(inputs, labels, head.named_parameters(), task=1, epoch=1, seed=1993)
    assert audit['gradient_parameter_count'] == 2
    assert audit['batch_size'] == 1
    head.weight.grad[0, 0] += .01
    changed = helpers.first_batch_audit(inputs, labels, head.named_parameters(), task=1, epoch=1, seed=1993)
    assert changed['gradient_sha256'] != audit['gradient_sha256']
    assert changed['inputs_sha256'] == audit['inputs_sha256']
    inputs[0, 0] += 1
    assert helpers.first_batch_audit(inputs, labels, head.named_parameters(), task=1, epoch=1, seed=1993)['inputs_sha256'] != audit['inputs_sha256']


@pytest.mark.parametrize('nonfinite', [False, True])
def test_firstbatch_audit_rejects_absent_or_invalid_gradients(nonfinite):
    parameter = torch.nn.Parameter(torch.ones(1))
    if nonfinite:
        parameter.grad = torch.tensor([float('inf')])
    with pytest.raises((ValueError, FloatingPointError)):
        helpers.first_batch_audit(torch.ones(1, 1), torch.zeros(1, dtype=torch.long), [('weight', parameter)],
                                  task=1, epoch=1, seed=1993)


def complete_record():
    return {'epoch_results': [{'task': task, 'epoch': epoch, 'epochs': 20, 'steps': 79, 'samples': 5000}
            for task in (1, 2) for epoch in range(1, 21)], 'prototype_adapter_image_pairs': 15000,
            'first_batch_audits': [{}, {}]}


def test_complete_reference_uses_actual_3160_steps():
    recorder = SimpleNamespace(tasks_completed=2, steps=3160, training_examples=200000, evaluation_examples=3000)
    assert helpers.completion_checks(complete_record(), recorder)


@pytest.mark.parametrize('corruption', ['batch48_steps', 'dropped_tail', 'duplicate_epoch', 'missing_audit', 'wrong_prototypes'])
def test_reference_count_gate_rejects_incomplete_or_historical_protocol(corruption):
    record = complete_record()
    recorder = SimpleNamespace(tasks_completed=2, steps=3160, training_examples=200000, evaluation_examples=3000)
    if corruption == 'batch48_steps':
        recorder.steps = 4200
    elif corruption == 'dropped_tail':
        record['epoch_results'][0]['samples'] = 4992
    elif corruption == 'duplicate_epoch':
        record['epoch_results'][-1]['epoch'] = 19
    elif corruption == 'missing_audit':
        record['first_batch_audits'].pop()
    elif corruption == 'wrong_prototypes':
        record['prototype_adapter_image_pairs'] -= 5000
    with pytest.raises(RuntimeError, match='exact two-task'):
        helpers.completion_checks(record, recorder)


def test_source_verification_rejects_dirty_or_other_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kwargs:
        SimpleNamespace(stdout='unexpected' if 'rev-parse' in command else ''))
    with pytest.raises(ValueError, match='clean at the pinned'):
        runner.verify_reference_source(tmp_path)


def test_author_config_changes_only_approved_protocol_fields(tmp_path):
    source = tmp_path / 'source'
    (source / 'exps').mkdir(parents=True)
    original = {'seed': [7], 'device': [0], 'shuffle': True, 'init_cls': 5, 'increment': 5,
                'init_epochs': 20, 'later_epochs': 20, 'batch_size': 48, 'ffn_num': 64,
                'alpha': .1, 'use_reweight': True, 'recalc_sim': True}
    (source / 'exps/ease_cifar.json').write_text(json.dumps(original))
    adapted = runner.author_config(source)
    assert adapted['batch_size'] == 64
    assert adapted['seed'] == 1993
    assert adapted['alpha'] == original['alpha'] and adapted['use_reweight'] is True and adapted['recalc_sim'] is True


def test_reference_task_initialization_roundtrip(tmp_path):
    from models.ease_utils.vit import ParallelAdapter
    from models.ease_utils.classifier import CosineHead
    catalog = tiny_catalog(tmp_path)
    network = SimpleNamespace(backbone=SimpleNamespace(cur_adapter=torch.nn.ModuleList([ParallelAdapter(768) for _ in range(12)])),
                              proxy_fc=CosineHead(768, 10))
    pin = helpers.load_reference_task_initialization(network, catalog, 2)
    assert torch.equal(network.proxy_fc.weight, catalog['tasks']['2']['proxy_head']['weight'])
    assert pin == helpers.tensor_digest({**{f'adapter.{name}': value for name, value in network.backbone.cur_adapter.state_dict().items()},
        **{f'proxy_head.{name}': value for name, value in network.proxy_fc.state_dict().items()}})


def test_common_patch_restores_author_class_and_loader_and_keeps_training_shuffle(monkeypatch):
    class AuthorData:
        def __init__(self):
            self.train_trsf = ['old train']
            self.test_trsf = ['old test']
            self.common_trsf = ['old common']
    class Records(list):
        _ease_reference_source = 'train'
        _ease_reference_mode = 'test'
    ease = SimpleNamespace(DataLoader=torch.utils.data.DataLoader)
    modules = {'data': SimpleNamespace(iCIFAR224=AuthorData), 'ease': ease}
    original_init, original_loader = AuthorData.__init__, ease.DataLoader
    def base_patches(modules, checkpoint, data_root, records):
        records.update(patches=[], loader_seed_policy={})
        return ExitStack()
    monkeypatch.setattr(helpers, 'install_reference_patches', base_patches)
    records = {}
    with helpers.install_common_reference_patches(modules, 'unused', 'unused', records):
        subject = AuthorData()
        assert isinstance(subject.train_trsf[0], transforms.RandomResizedCrop)
        prototype = ease.DataLoader(Records(range(9)), batch_size=4, shuffle=True)
        assert isinstance(prototype.sampler, torch.utils.data.SequentialSampler)
        train_data = Records(range(9))
        train_data._ease_reference_mode = 'train'
        train = ease.DataLoader(train_data, batch_size=4, shuffle=True)
        assert isinstance(train.sampler, torch.utils.data.RandomSampler)
        assert [len(batch) for batch in prototype] == [4, 4, 1]
        with pytest.raises(ValueError, match='every sample'):
            ease.DataLoader(train_data, batch_size=4, drop_last=True)
    assert AuthorData.__init__ is original_init
    assert ease.DataLoader is original_loader
    assert AuthorData().train_trsf == ['old train']
