"""Independent immutable-source size variants and matched CPU two-task oracles."""
from argparse import ArgumentParser
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from scripts import unified_lora_reference as reference
from scripts.low_rank_protocol import seed_phase, tensor_digest, verify_reference_source
from models.low_rank_utils.pretrained import canonical_pretrained_digest, canonical_pretrained_state
from models.low_rank_utils.continual import LowRankContinualModel
from utils.schedulers import get_scheduler


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    monkeypatch.setattr('socket.socket.connect', lambda *_a, **_k: pytest.fail('Network forbidden in CPU reference validation'))
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1993)
        yield
    torch.set_num_threads(old)


def test_default_cli_is_stdlib_only_and_does_not_read_source_or_credentials(tmp_path):
    script = ('import sys; from scripts import unified_lora_reference as r; '
              'r.main(["--method","sd-lora","--vit-size","small","--batch-size","64",'
              f'"--reference-source",{str(tmp_path / "missing-source")!r}]); '
              'assert "torch" not in sys.modules and "wandb" not in sys.modules and "timm" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', script], cwd=reference.ROOT, capture_output=True, text=True, check=True)
    document = json.loads(result.stdout)
    assert document['model_constructed'] is document['experiment_executed'] is False
    assert document['reference']['vit_size'] == 'small' and document['reference']['batch_size'] == 64
    assert document['reference']['recipe']['lr'] == .001 and document['reference']['recipe']['lora_fc_lr'] == .008
    assert not list(tmp_path.iterdir())


def test_direct_script_cli_works_outside_repository_without_constructing_models(tmp_path):
    result = subprocess.run([sys.executable, str(reference.ROOT / 'scripts/unified_lora_reference.py'),
                             '--method', 'inf-lora', '--vit-size', 'tiny', '--batch-size', '64'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    document = json.loads(result.stdout)
    assert document['reference']['method'] == 'inf-lora'
    assert document['reference']['batch_size'] == 64 and document['model_constructed'] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('method', tuple(reference.METHODS))
@pytest.mark.parametrize('size,batch', [('tiny', 16), ('small', 64), ('base', 16)])
def test_fresh_recipe_mapping_preserves_two_tasks_ten_task_capacity_and_size_pins(method, size, batch):
    spec = reference.reference_spec(method, size, batch)
    description = reference.reference_description(spec)
    settings = reference.author_args(spec)
    assert description['implementation'] == 'pinned_author_size_variant_common_input_reference'
    assert (settings['sessions'], settings['epochs'], settings['batch_size']) == (10, 2, batch)
    assert settings['rank'] == 10 and settings['init_cls'] == settings['increment'] == 10
    assert settings['lrate'] == spec['method_recipe']['lr']
    assert settings['fc_lrate'] == spec['method_recipe']['lora_fc_lr']
    assert description['normalization_mean'] == description['normalization_std'] == [.5] * 3
    assert not description['paper_reproduction'] and not description['resource_statistics_eligible']


def test_reference_rejects_recipe_scope_drift():
    spec = reference.reference_spec('sd-lora')
    spec['method_recipe']['lr'] = .008
    with pytest.raises(ValueError, match='approved cell'):
        reference.reference_description(spec)
    with pytest.raises(ValueError, match='supported size'):
        reference.reference_spec('sd-lora', 'tiny', 32)


def test_common_image_pipeline_uses_bicubic_crop_and_mean_half():
    from PIL import Image
    training, testing = reference.common_transforms()
    assert training.transforms[0].interpolation is transforms.InterpolationMode.BICUBIC
    assert training.transforms[0].scale == (.08, 1.) and training.transforms[0].ratio == (.75, 4 / 3)
    assert training.transforms[-1].mean == testing.transforms[-1].mean == (.5,) * 3
    image = Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8))
    output = testing(image)
    assert output.shape == (3, 224, 224)
    torch.testing.assert_close(output, torch.full_like(output, (127 / 255 - .5) / .5))


@pytest.mark.parametrize('method', tuple(reference.METHODS))
@pytest.mark.parametrize('size', ('tiny', 'small'))
def test_actual_author_full_size_network_canonical150_and_native_catalog(method, size):
    """Read actual local NPZ, all12 blocks; no forward or optimizer step."""
    before = verify_reference_source(reference.DEFAULT_SOURCE)
    spec = reference.reference_spec(method, size, 16)
    seed_phase(1993, 0, 'task_begin')
    session = reference.build_reference(spec)
    assert session.learner.network.image_encoder.embed_dim == spec['feature_dim']
    assert session.learner.network.image_encoder.blocks[0].attn.num_heads == spec['num_heads']
    assert len(canonical_pretrained_state(session.learner.network)) == 150
    assert canonical_pretrained_digest(session.learner.network) == spec['checkpoint']['canonical_sha256']
    assert session.learner.network.__class__.__module__.startswith('_unified_lora_author_')
    assert sys.modules['models'].__file__.startswith(str(reference.ROOT))
    patched = session.evidence['loaded_source_adaptations'][f'models/net_{reference.METHODS[method]}.py']
    assert patched['geometry_patches'] == 1
    assert patched['classifier_patches'] == (0 if method == 'cl-lora' else 1)
    catalog = reference.initial_catalog(session)
    from models.low_rank_utils.factory import build_network
    native_args = reference.author_args(spec)
    seed_phase(1993, 0, 'task_begin')
    native = build_network(method, native_args, vit_size=size)
    assert reference.restore_initial_catalog(native, catalog, session) == catalog['initial_learned_sha256']
    assert tensor_digest(native.state_dict()) == tensor_digest(session.learner.network.state_dict())
    assert verify_reference_source(reference.DEFAULT_SOURCE) == before


class ToyTasks:
    NAME, SETTING = 'synthetic-unified-low-rank', 'class-il'
    N_CLASSES, N_TASKS, N_CLASSES_PER_TASK = 20, 10, 2
    TEST_TRANSFORM = transforms.Lambda(lambda value: value)

    def __init__(self):
        self.images = [torch.rand(8, 3, 8, 8) * 2 - 1 for _ in range(2)]
        self.labels = [torch.tensor([0, 1] * 4) + task * 2 for task in range(2)]
        self.loaders = [DataLoader(TensorDataset(images, labels, images), batch_size=4, shuffle=False,
                                   generator=torch.Generator().manual_seed(1993)) for images, labels in zip(self.images, self.labels)]
        self.set_task(0)

    def set_task(self, task):
        self.train_loader = self.loaders[task]

    def get_offsets(self, task):
        return task * 2, (task + 1) * 2

    def inputs(self, task, phase, epoch):
        if phase == 'evaluation':
            return [(torch.cat(self.images[:task + 1]), torch.cat(self.labels[:task + 1]), torch.cat(self.images[:task + 1]))]
        return self.loaders[task]


def make_toy(method):
    spec = reference.reference_spec(method)
    geometry = dict(img_size=8, patch_size=4, embed_dim=8, depth=2, num_heads=2)
    overrides = dict(init_cls=2, increment=2, rank=2, num_workers=0, batch_size=4,
                     shared_pos=[0], specific_pos=[1])
    seed_phase(1993, 0, 'task_begin')
    session = reference.build_reference(spec, synthetic_geometry=geometry, synthetic_args=overrides)
    cls = type('ToyNative', (LowRankContinualModel,), {'NAME': method, 'METHOD': reference.METHODS[method]})
    args = cls.get_parser(ArgumentParser()).parse_args([])
    args.lr, args.lora_fc_lr = spec['method_recipe']['lr'], spec['method_recipe']['lora_fc_lr']
    args.n_epochs = 2
    args.label_perc = 1.
    args.nowand, args.num_workers, args.batch_size = 1, 0, 4
    args.fitting_mode, args.eval_epochs = 'epochs', None
    args.loadcheck = args.start_from = None
    args.lora_rank, args.lora_shared_pos, args.lora_specific_pos = 2, [0], [1]
    dataset = ToyTasks()
    source_settings = reference.author_args(spec)
    source_settings.update(overrides)
    seed_phase(1993, 0, 'task_begin')
    native_net = import_module(f'models.low_rank_utils.net_{reference.METHODS[method]}').Net(source_settings, geometry=geometry)
    # The toy's random encoder stands in for the independently pinned NPZ.
    native_net.load_state_dict(session.learner.network.state_dict(), strict=True)
    catalog = reference.initial_catalog(session)
    reference.restore_initial_catalog(native_net, catalog, session)
    return session, cls(native_net, F.cross_entropy, args, None, dataset=dataset), dataset


@pytest.mark.parametrize('method', tuple(reference.METHODS))
def test_actual_source_vs_native_two_task_updates_boundaries_and_seen_old_new_logits(method):
    session, native, dataset = make_toy(method)
    for task in range(2):
        dataset.set_task(task)
        seed_phase(1993, task, 'task_begin')
        native.meta_begin_task(dataset)
        scheduler = get_scheduler(native, native.args, reload_optim=False)
        gradients, lrs, losses = [], [], []

        def before_step(subject, args, kwargs):
            if native.epoch_iteration == 0:
                gradients.append({name: value.grad.detach().clone() for name, value in native.net.named_parameters()
                                  if value.requires_grad and value.grad is not None})
                lrs.append([group['lr'] for group in subject.param_groups])

        handle = native.opt.register_step_pre_hook(before_step)
        for epoch in range(2):
            seed_phase(1993, task, 'train', epoch)
            native.meta_begin_epoch(epoch, dataset)
            native.net.train()
            for images, labels, original in dataset.train_loader:
                losses.append(native.meta_observe(images, labels, original, epoch=epoch))
            native.meta_end_epoch(epoch, dataset)
            scheduler.step()
        handle.remove()
        seed_phase(1993, task, 'task_end')
        native.meta_end_task(dataset)
        seed_phase(1993, task, 'evaluation')
        native.eval()
        with torch.no_grad():
            native_logits = native(torch.cat(dataset.images[:task + 1])).clone()
        seed_phase(1993, task, 'task_end')
        native.meta_end_eval(dataset, ([], []))
        row = session.run_task(dataset.inputs, phase_rng=lambda t, p, e: seed_phase(1993, t, p, e))
        assert all(np.isfinite(loss) for loss in losses)
        assert row['observe_calls'] == 4
        assert row['optimizer_updates'] == (8 if method == 'cl-lora' and task == 1 else 4)
        assert row['known_classes'] == (task + 1) * 2 and row['network_task'] == task
        torch.testing.assert_close(row['logits'], native_logits, rtol=1e-5, atol=1e-6)
        # Both old and new classifier segments are compared, not just argmax.
        if task:
            torch.testing.assert_close(row['logits'][:, :2], native_logits[:, :2], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(row['logits'][:, 2:], native_logits[:, 2:], rtol=1e-5, atol=1e-6)
        first_steps = 2 if method == 'cl-lora' and task else 1
        assert len(row['first_gradients']) == first_steps
        assert len(gradients) == first_steps * 2
        for source_grads, native_grads, audit, native_lr in zip(row['first_gradients'], gradients, row['first_batch_audits'], lrs):
            assert source_grads.keys() == native_grads.keys()
            for name in source_grads:
                torch.testing.assert_close(source_grads[name], native_grads[name], rtol=1e-5, atol=1e-6)
            assert audit['lrs'] == native_lr
        for name, value in session.learner.network.state_dict().items():
            torch.testing.assert_close(value, native.net.state_dict()[name], rtol=1e-5, atol=1e-6)
        if method == 'ewc-lora':
            for author_fisher, native_fisher in zip(session.learner.omega_W, native.algorithm.omega_W):
                torch.testing.assert_close(author_fisher, native_fisher, rtol=1e-5, atol=1e-6)
        if method == 'inf-lora':
            assert session.learner.project_type == native.algorithm.project_type
            for author_basis, native_basis in zip(session.learner.feature_list, native.algorithm.feature_list):
                np.testing.assert_allclose(author_basis, native_basis.numpy(), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match='two tasks'):
        session.run_task(dataset.inputs, phase_rng=lambda *_: pytest.fail('extra task started'))


def test_catalog_identity_or_tensor_corruption_is_rejected():
    session, native, _ = make_toy('sd-lora')
    catalog = reference.initial_catalog(session)
    catalog['metadata']['vit_size'] = 'small'
    with pytest.raises(ValueError, match='identity/hash'):
        reference.restore_initial_catalog(native.net, catalog, session)
    catalog = reference.initial_catalog(session)
    with torch.no_grad():
        native.net.image_encoder.patch_embed.proj.weight.add_(1)
    with pytest.raises(ValueError, match='identical frozen'):
        reference.restore_initial_catalog(native.net, catalog, session)
    catalog = reference.initial_catalog(session)
    next(iter(catalog['learned_state'].values())).add_(1)
    with pytest.raises(ValueError, match='identity/hash'):
        reference.restore_initial_catalog(native.net, catalog, session)
