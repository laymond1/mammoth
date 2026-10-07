"""CPU constructor/protocol checks; no image forwards, downloads or experiments."""

import ast
import functools
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from scripts import ease_reference_utils as helpers
from scripts import ease_size_protocol as protocol
from scripts import run_ease_reference as runner


SOURCE = Path('/tmp/cl-ease-official-20261007')


def upstream_function(path, name, namespace):
    """Compile just a reviewed upstream function, without importing its modules."""
    if not path.is_file():
        pytest.skip('The read-only reviewed EASE source is not available')
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('dimension,heads,adapter_size', [(384, 6, 64), (192, 3, 64), (384, 6, 32), (192, 3, 16)])
def test_reviewed_constructor_specialization_preserves_qkv_conversion_and_freezing(dimension, heads, adapter_size):
    parameters = {'cls_token': torch.nn.Parameter(torch.zeros(1)),
                  'cur_adapter.0.down_proj.weight': torch.nn.Parameter(torch.zeros(1))}

    class Transformer:
        def __init__(self, **kwargs):
            self.options = kwargs

        def load_state_dict(self, state, strict=True):
            self.loaded = state
            assert strict is False
            return SimpleNamespace(missing_keys=['cur_adapter.0.down_proj.weight'])

        def named_parameters(self):
            return parameters.items()

    qkv = torch.arange(3 * dimension, dtype=torch.float32)
    pretrained = SimpleNamespace(state_dict=lambda: {'blocks.0.attn.qkv.weight': qkv[:, None].clone(),
                                                       'blocks.0.attn.qkv.bias': qkv.clone(),
                                                       'blocks.0.mlp.fc1.weight': torch.ones(1)})
    create = Mock(return_value=pretrained)
    original = upstream_function(SOURCE / 'backbone/vit_ease.py', 'vit_base_patch16_224_ease',
                                 {'__name__': '_author_factory', 'VisionTransformer': Transformer,
                                  'timm': SimpleNamespace(create_model=create),
                                  'partial': functools.partial, 'nn': torch.nn})
    specialized, audit = helpers._size_constructor(original, dimension, heads)
    result = specialized(num_classes=0, tuning_config=SimpleNamespace(ffn_num=adapter_size))
    assert result.options['embed_dim'] == dimension and result.options['num_heads'] == heads
    assert result.options['depth'] == 12 and result.options['patch_size'] == 16
    assert result.options['tuning_config'].ffn_num == adapter_size
    for index, part in enumerate(('q', 'k', 'v')):
        torch.testing.assert_close(result.loaded[f'blocks.0.attn.{part}_proj.bias'], qkv[index * dimension:(index + 1) * dimension])
        assert result.loaded[f'blocks.0.attn.{part}_proj.weight'].shape == (dimension, 1)
    assert 'blocks.0.fc1.weight' in result.loaded
    assert not parameters['cls_token'].requires_grad
    assert parameters['cur_adapter.0.down_proj.weight'].requires_grad
    assert audit['feature_dim_constants'] == 9 and audit['attention_head_arguments'] == 1
    assert original.__code__ is not specialized.__code__


@pytest.mark.parametrize('dimension,adapter_size', [(384, 64), (192, 64), (384, 32), (192, 16)])
def test_reviewed_get_backbone_uses_the_same_adapter_recipe_and_selected_width(dimension, adapter_size):
    factory = Mock(return_value=SimpleNamespace(eval=lambda: 'constructed author backbone'))
    upstream_backbone = ModuleType('backbone')
    upstream_backbone.vit_ease = SimpleNamespace(vit_base_patch16_224_ease=factory)
    original = upstream_function(SOURCE / 'utils/inc_net.py', 'get_backbone', {'__name__': '_author_inc_net'})
    specialized, audit = helpers._size_constructor(original, dimension)
    with patch.dict(sys.modules, {'backbone': upstream_backbone}):
        result = specialized({'backbone_type': 'vit_base_patch16_224_ease', 'ffn_num': adapter_size,
                              'model_name': 'ease', 'device': ['cpu']})
    assert result == 'constructed author backbone'
    config = factory.call_args.kwargs['tuning_config']
    assert config.d_model == dimension and config.ffn_num == adapter_size
    assert config.ffn_option == 'parallel' and config.ffn_adapter_scalar == '0.1'
    assert config.ffn_adapter_layernorm_option == 'none' and config.ffn_adapter_init_option == 'lora'
    assert config.vpt_on is False and config._device == 'cpu'
    assert factory.return_value.out_dim == dimension
    assert audit['feature_dim_constants'] == 5 and audit['attention_head_arguments'] == 0


def test_constructor_specialization_refuses_a_changed_unreviewed_shape():
    def changed_constructor():
        return 384
    with pytest.raises(RuntimeError, match='expected dimension constants'):
        helpers._size_constructor(changed_constructor, 192, 3)


def test_variant_cli_is_dry_by_default_and_does_not_verify_or_load_weights(monkeypatch, capsys):
    execute = Mock(side_effect=AssertionError('No experiments'))
    load = Mock(side_effect=AssertionError('No checkpoint access'))
    monkeypatch.setattr(runner, 'execute', execute)
    monkeypatch.setattr(runner, 'configure_environment', lambda: {})
    monkeypatch.setattr(protocol, 'load_variant', load)
    assert runner.main(['--variant', 'tiny_relative16']) == 0
    assert 'execution=False' in capsys.readouterr().out
    execute.assert_not_called()
    load.assert_not_called()


def test_base_selection_keeps_the_original_checkpoint_and_has_no_variant(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'file_hash', lambda path: runner.CHECKPOINT_SHA256)
    load = Mock(side_effect=AssertionError('Base must not use the size plan'))
    monkeypatch.setattr(protocol, 'load_variant', load)
    variant, checkpoint, digest = runner.reference_selection(SimpleNamespace(variant=None), {'cache_root': str(tmp_path)})
    assert variant is None and checkpoint.name == runner.CHECKPOINT_NAME and digest == runner.CHECKPOINT_SHA256
    load.assert_not_called()


@pytest.fixture
def size_spec(monkeypatch, tmp_path):
    variant = {'name': 'tiny_relative16', 'vit_size': 'tiny', 'feature_dim': 192, 'num_heads': 3,
               'adapter_size': 16, 'checkpoint_file': str(tmp_path / 'pinned.npz'),
               'checkpoint_sha256': '1' * 64, 'canonical_pretrained_sha256': '2' * 64,
               'checkpoint_tag': 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'}
    monkeypatch.setattr(protocol, 'load_variant', lambda name: variant.copy())
    config = {'backbone_type': 'vit_base_patch16_224_ease', 'model_name': 'ease', 'use_init_ptm': False,
              'use_reweight': True, 'recalc_sim': True, 'alpha': .1, 'use_diagonal': False,
              'use_old_data': False, 'moni_adam': False, 'adapter_num': -1, 'memory_size': 0,
              'init_lr': .025, 'later_lr': .025, 'weight_decay': .0005, 'min_lr': 0,
              'optimizer': 'sgd', 'scheduler': 'cosine', 'beta': 0}
    config.update(init_epochs=20, later_epochs=20, seed=1993, shuffle=False, init_cls=10, increment=10,
                  ffn_num=16, batch_size=48)
    return {'variant': variant['name'], 'variant_config': variant, 'config': config, 'gpu': '1', 'num_workers': 4,
            'tasks': 2, 'checkpoint': variant['checkpoint_file'], 'checkpoint_sha256': variant['checkpoint_sha256'],
            'checkpoint_tag': variant['checkpoint_tag'], 'reference_commit': runner.REFERENCE_COMMIT}


def test_fresh_reference_worker_revalidates_the_entire_size_selection(size_spec):
    assert runner.validate_reference_spec(size_spec) == size_spec['variant_config']


@pytest.mark.parametrize('field,value', [('ffn_num', 64), ('init_epochs', 1), ('use_old_data', True), ('alpha', .5), ('moni_adam', True)])
def test_reference_worker_rejects_shape_or_algorithm_protocol_changes(size_spec, field, value):
    size_spec['config'][field] = value
    with pytest.raises(ValueError, match='approved architecture/protocol/checkpoint'):
        runner.validate_reference_spec(size_spec)


def test_reference_worker_rejects_changed_checkpoint_pin(size_spec):
    size_spec['checkpoint_sha256'] = '3' * 64
    with pytest.raises(ValueError, match='approved architecture/protocol/checkpoint'):
        runner.validate_reference_spec(size_spec)


@pytest.fixture
def metadata_backbone(size_spec):
    variant = size_spec['variant_config']
    width, adapter = variant['feature_dim'], variant['adapter_size']
    with torch.device('meta'):
        backbone = torch.nn.Module()
        backbone.out_dim = backbone.embed_dim = width
        backbone.blocks = torch.nn.ModuleList()
        backbone.cur_adapter = torch.nn.ModuleList()
        for _ in range(12):
            block = torch.nn.Module()
            block.attn = SimpleNamespace(num_heads=variant['num_heads'])
            backbone.blocks.append(block)
            layer = torch.nn.Module()
            layer.down_proj = torch.nn.Linear(width, adapter)
            layer.up_proj = torch.nn.Linear(adapter, width)
            backbone.cur_adapter.append(layer)
        backbone.register_parameter('cls_token', torch.nn.Parameter(torch.empty(1, 1, width), requires_grad=False))
    return backbone


def test_reference_checks_meta_architecture_and_canonical_pin_without_an_image_forward(size_spec, metadata_backbone, monkeypatch):
    variant = size_spec['variant_config']
    digest = Mock(return_value=variant['canonical_pretrained_sha256'])
    monkeypatch.setattr(protocol, 'canonical_pretrained_digest', digest)
    assert runner.audit_variant_initialization(metadata_backbone, variant) == variant['canonical_pretrained_sha256']
    digest.assert_called_once_with(metadata_backbone)
    digest.return_value = 'wrong'
    with pytest.raises(RuntimeError, match='canonical pin'):
        runner.audit_variant_initialization(metadata_backbone, variant)


@pytest.mark.parametrize('changed', ['heads', 'width', 'trainable_ptm'])
def test_reference_rejects_wrong_architecture_before_digest(size_spec, metadata_backbone, monkeypatch, changed):
    if changed == 'heads':
        metadata_backbone.blocks[0].attn.num_heads = 12
    elif changed == 'width':
        metadata_backbone.out_dim = 768
    else:
        metadata_backbone.cls_token.requires_grad_(True)
    digest = Mock(side_effect=AssertionError('Do not read a mismatched model'))
    monkeypatch.setattr(protocol, 'canonical_pretrained_digest', digest)
    with pytest.raises(RuntimeError, match='wrong architecture'):
        runner.audit_variant_initialization(metadata_backbone, size_spec['variant_config'])
    digest.assert_not_called()


@pytest.mark.parametrize('incomplete', [None, 'epochs', 'steps', 'evaluation_examples'])
def test_size_reference_requires_all_twenty_epochs_and_all_samples(incomplete):
    record = {'epoch_results': [{'task': task, 'epoch': epoch, 'epochs': 20} for task in (1, 2) for epoch in range(1, 21)]}
    recorder = SimpleNamespace(tasks_completed=2, steps=4200, training_examples=200000, evaluation_examples=3000)
    if incomplete is None:
        runner.audit_variant_completion(record, recorder)
    else:
        if incomplete == 'epochs':
            record['epoch_results'].pop()
        else:
            setattr(recorder, incomplete, getattr(recorder, incomplete) - 1)
        with pytest.raises(RuntimeError, match='20 epochs|full matched'):
            runner.audit_variant_completion(record, recorder)


def test_reference_source_does_not_import_the_native_algorithm():
    for file in ('scripts/run_ease_reference.py', 'scripts/ease_reference_utils.py'):
        source = (runner.ROOT / file).read_text()
        assert 'models.ease_utils' not in source and 'backbone.ease_vit' not in source
    tree = ast.parse((runner.ROOT / 'scripts/run_ease_reference.py').read_text())
    imports = [(node.module, tuple(alias.name for alias in node.names)) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert ('backbone', ('vit_ease',)) in imports and ('models', ('ease',)) in imports
