"""Pinned-author HiDe geometry extensions without GPU training or network I/O."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts import run_hide_reference as runner
from scripts.hide_reference_utils import (build_author_branches, learned_state,
    learned_state_digest, load_author_code, load_initial_state, save_initial_state,
    set_author_phase, verify_author_source)

SOURCE = Path('/tmp/cl-hide-official-20261007')


@pytest.fixture(scope='module')
def author():
    if not SOURCE.is_dir():
        pytest.skip('Pinned author checkout absent')
    verify_author_source(SOURCE)
    return load_author_code(SOURCE)


@pytest.fixture
def pins(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'small.npz'
    checkpoint.write_bytes(b'local pinned checkpoint fixture')
    variant = {'name': 'small', 'vit_size': 'small', 'feature_dim': 384, 'num_heads': 6,
               'checkpoint_file': str(checkpoint), 'checkpoint_sha256': runner.file_hash(checkpoint),
               'canonical_pretrained_sha256': 'a' * 64,
               'checkpoint_tag': 'vit_small_patch16_224.augreg_in21k_ft_in1k'}
    module = ModuleType('scripts.hide_size_protocol')
    module.load_variant = lambda name, verify_checkpoint=False: variant.copy()
    module.SOURCE_FILES = ('scripts/hide_size_protocol.py', 'benchmarks/cl_efficiency/hide_sizes/validation_plan.json')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return checkpoint, variant


def test_base_geometry_and_checkpoint_defaults_remain_unchanged():
    options = Namespace(checkpoint=Path(runner.CHECKPOINT))
    assert runner.reference_variant(options) is None
    assert runner.author_geometry() == {}
    assert runner.checkpoint_pins(options) == (runner.CHECKPOINT_SHA256, runner.CANONICAL_PRETRAINED_SHA256)


def test_reference_variant_binds_shared_cache_path_and_geometry(pins):
    checkpoint, variant = pins
    options = Namespace(variant='small', checkpoint=checkpoint)
    assert runner.reference_variant(options, True) == variant
    assert runner.author_geometry(variant) == {'embed_dim': 384, 'num_heads': 6}
    assert runner.checkpoint_pins(options, variant) == (variant['checkpoint_sha256'], variant['canonical_pretrained_sha256'])


def test_reference_rejects_checkpoint_from_another_size(pins, tmp_path):
    checkpoint, variant = pins
    other = tmp_path / 'other.npz'
    other.write_bytes(checkpoint.read_bytes())
    with pytest.raises(ValueError, match='path differs'):
        runner.reference_variant(Namespace(variant='small', checkpoint=other), True)


def test_variant_spec_pins_immutable_plan_and_exact_checkpoint(pins, tmp_path, monkeypatch):
    checkpoint, variant = pins
    initial = tmp_path / 'initial.pt'
    initial.write_bytes(b'matched learned initialization')
    source = tmp_path / 'author'
    source.mkdir()
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'verify_author_source', lambda path: {'model.py': 'b' * 64})
    monkeypatch.setattr(runner, 'protocol', lambda mode, variant=None: {'mode': mode, 'vit_size': variant,
                                                                    'accuracy_acceptance_eligible': True})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='a' * 40))
    names = ('scripts/run_hide_reference.py', 'scripts/hide_reference_utils.py', 'scripts/hide_protocol.py',
             'scripts/cl_efficiency.py', 'scripts/hide_size_protocol.py',
             'benchmarks/cl_efficiency/hide_sizes/validation_plan.json')
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    options = Namespace(variant='small', checkpoint=checkpoint, reference_source=source, initial_state=initial, mode='validation')
    spec = runner.source_spec(options)
    assert spec['variant'] == 'small' and spec['feature_dim'] == 384 and spec['num_heads'] == 6
    assert spec['variant_config'] == variant
    assert spec['checkpoint_sha256'] == variant['checkpoint_sha256']
    assert spec['canonical_pretrained_sha256'] == variant['canonical_pretrained_sha256']
    assert set(spec['execution_source_sha256']) == set(names)
    assert not any('execution_plan.json' in name for name in spec['execution_source_sha256'])


def test_variant_spec_rejects_changed_npz_before_initialization(pins, tmp_path, monkeypatch):
    checkpoint, variant = pins
    checkpoint.write_bytes(b'corrupted checkpoint')
    monkeypatch.setattr(runner, 'protocol', lambda mode, variant=None: {})
    monkeypatch.setattr(runner, 'verify_author_source', lambda path: {})
    with pytest.raises(ValueError, match='NPZ differs'):
        runner.source_spec(Namespace(variant='small', checkpoint=checkpoint, reference_source=tmp_path,
                                     initial_state=tmp_path / 'not-needed.pt', mode='validation'))


def test_variant_cpu_export_uses_original_geometry_and_metadata(pins, tmp_path, monkeypatch):
    checkpoint, variant = pins
    monkeypatch.setattr(runner, 'verify_author_source', lambda path: {})
    monkeypatch.setattr(runner, 'load_author_code', lambda path: 'immutable-author-code')
    subject = torch.nn.Module()
    subject.head = torch.nn.Linear(4, 100)
    branches = {'tii': subject, 'prompt': subject}
    observed = {}
    def build(code, path, **kwargs):
        observed.update(code=code, checkpoint=path, geometry=kwargs)
        return branches
    monkeypatch.setattr(runner, 'build_author_branches', build)
    from scripts import ease_size_protocol
    monkeypatch.setattr(ease_size_protocol, 'canonical_pretrained_digest', lambda module: variant['canonical_pretrained_sha256'])
    output = tmp_path / 'author-initial.pt'
    assert runner.prepare_initial(Namespace(variant='small', checkpoint=checkpoint, reference_source=tmp_path,
                                            prepare_initial_state=output)) == 0
    assert observed['geometry'] == {'embed_dim': 384, 'num_heads': 6}
    metadata = json.loads(output.with_suffix('.json').read_text())
    state = torch.load(output, map_location='cpu', weights_only=True)
    assert metadata['variant'] == 'small' and metadata['variant_config'] == variant
    assert state['checkpoint_sha256'] == variant['checkpoint_sha256']
    assert metadata['canonical_pretrained_sha256'] == {'tii': variant['canonical_pretrained_sha256'], 'prompt': variant['canonical_pretrained_sha256']}
    assert metadata['image_forwards'] == metadata['optimizer_steps'] == 0
    assert metadata['gpu_used'] is False


@pytest.mark.parametrize('size,dimension,heads', [('small', 384, 6), ('tiny', 192, 3)])
def test_actual_author_size_architecture_and_learned_shapes(author, size, dimension, heads):
    branches = build_author_branches(author, embed_dim=dimension, num_heads=heads)
    tii, prompt = branches['tii'], branches['prompt']
    assert type(tii).__module__ == type(prompt).__module__ == 'hide_pinned_author'
    assert len(tii.blocks) == len(prompt.blocks) == 12
    assert all(block.attn.num_heads == heads for module in branches.values() for block in module.blocks)
    assert tuple(tii.patch_embed.proj.weight.shape) == (dimension, 3, 16, 16)
    assert tuple(tii.head.weight.shape) == tuple(prompt.head.weight.shape) == (100, dimension)
    assert tuple(prompt.e_prompt.prompt.shape) == (5, 2, 10, 5, heads, dimension // heads)
    assert tuple(tii.mlp.net[0][0].weight.shape) == (2 * dimension, dimension)
    assert tuple(tii.mlp.net[1][0].weight.shape) == (dimension, 2 * dimension)
    assert all(not parameter.requires_grad for module in branches.values() for parameter in module.parameters())
    for branch in ('tii', 'prompt'):
        assert all(value.dtype == torch.float32 for value in learned_state(branches[branch], branch).values())


def test_original_author_reduced_geometry_gradients_and_phase_switch(author):
    branches = build_author_branches(author, embed_dim=12, depth=2, num_heads=3, img_size=16, patch_size=8)
    images, labels = torch.randn(2, 3, 16, 16), torch.tensor([0, 1])
    set_author_phase(branches, 'tii')
    loss = torch.nn.functional.cross_entropy(branches['tii'](images)['logits'][:, :10], labels)
    loss.backward()
    assert any(parameter.grad is not None for parameter in branches['tii'].mlp.parameters())
    assert all(parameter.grad is None for parameter in branches['tii'].blocks.parameters())
    set_author_phase(branches, 'prompt')
    assert all(parameter.grad is None for parameter in branches['tii'].parameters())
    result = branches['prompt'](images, task_id=0, train=True)
    torch.nn.functional.cross_entropy(result['logits'][:, :10], labels).backward()
    assert branches['prompt'].e_prompt.prompt.grad is not None
    assert all(parameter.grad is None for parameter in branches['prompt'].blocks.parameters())
    set_author_phase(branches, 'eval')
    assert all(parameter.grad is None and not parameter.requires_grad for module in branches.values() for parameter in module.parameters())


def test_variant_checkpoint_roundtrip_uses_selected_npz_pin(author, tmp_path):
    branches = build_author_branches(author, embed_dim=12, depth=2, num_heads=3, img_size=16, patch_size=8)
    for namespace in author.engines.values():
        namespace['cls_mean'].clear(); namespace['cls_cov'].clear()
        namespace['cls_mean'][0] = [torch.ones(12)]
        namespace['cls_cov'][0] = [torch.full((12,), .1)]
    pin = 'c' * 64
    result = runner.checkpoint_round_trip(branches, author, 0, tmp_path, checkpoint_sha256=pin)
    state = torch.load(result['file'], map_location='cpu', weights_only=True)
    assert state['checkpoint_sha256'] == pin
    assert result['learned_sha256'] == {name: learned_state_digest(module, name) for name, module in branches.items()}
    for namespace in author.engines.values():
        namespace['cls_mean'].clear(); namespace['cls_cov'].clear()


@pytest.mark.parametrize('variant', ['small', 'tiny'])
def test_size_dry_cli_without_ml_or_credentials(variant):
    result = subprocess.run([sys.executable, '-S', str(runner.ROOT / 'scripts/run_hide_reference.py'),
                             '--variant', variant], text=True, capture_output=True, check=True)
    record = json.loads(result.stdout)
    assert record['executed'] is False
    assert record['protocol']['vit_size'] == variant
    assert record['protocol']['batch_size'] == 64


@pytest.mark.parametrize('variant', ['small', 'tiny'])
def test_actual_shared_size_npz_canonical_and_learned_roundtrip(author, variant, tmp_path):
    from scripts.ease_size_protocol import canonical_pretrained_digest
    from scripts.hide_size_protocol import load_variant
    selected = load_variant(variant, verify_checkpoint=True)
    torch.manual_seed(1993)
    branches = build_author_branches(author, selected['checkpoint_file'], **runner.author_geometry(selected))
    assert {canonical_pretrained_digest(module) for module in branches.values()} == {selected['canonical_pretrained_sha256']}
    path = tmp_path / f'{variant}-initial.pt'
    saved = save_initial_state(branches, path, selected['checkpoint_sha256'])
    with torch.no_grad():
        branches['tii'].head.weight.add_(.01)
    loaded = load_initial_state(branches, path)
    assert loaded == saved
    assert learned_state_digest(branches['tii'], 'tii') == saved['learned_sha256']['tii']
    assert saved['checkpoint_sha256'] == selected['checkpoint_sha256']


def test_size_author_args_preserve_alignment_schedule():
    settings = {'seed': 1993, 'batch_size': 64, 'tii_epochs': 20, 'prompt_epochs': 50, 'ca_epochs': 30, 'vit_size': 'tiny'}
    tii, prompt = runner.author_args(settings, 'tii'), runner.author_args(settings, 'prompt')
    assert tii.model == prompt.model == 'vit_tiny_patch16_224'
    assert tii.ca_lr == .00125 and prompt.ca_lr == .005
    assert tii.n_centroids == prompt.n_centroids == 10
    assert tii.crct_epochs == prompt.crct_epochs == 30
