"""CPU checks for the native pinned LoRA networks, with no downloads or datasets."""
import ast
import copy
import logging
import math
from collections import OrderedDict
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath, Mlp, PatchEmbed, lecun_normal_, trunc_normal_
from timm.models._manipulate import checkpoint_seq, named_apply

from models.low_rank_utils.factory import CHECKPOINTS, SIZES, build_network, checkpoint_spec, method_name
from models.low_rank_utils.modules.linears import CosineLinear
from models.low_rank_utils.pretrained import (
    canonical_pretrained_digest, canonical_pretrained_state, learned_initial_state,
    load_pretrained_encoder, restore_learned_initial_state,
)


SOURCE = Path('/tmp/cl-low-rank-official-20261008/models')
GEOMETRY = dict(img_size=8, patch_size=4, embed_dim=8, depth=2, num_heads=2)
ARGS = dict(init_cls=2, increment=2, sessions=3, rank=2, device=['cpu'], msa=[1, 0, 1], shared_pos=[0], specific_pos=[1])
METHODS = ('ewclora', 'sdlora', 'inflora', 'cllora')


@pytest.fixture(autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def network(method):
    return import_module('models.low_rank_utils.net_' + method).Net(ARGS, geometry=GEOMETRY)


def source_network(method):
    """Execute pinned source classes without legacy timm registration/fetching.

    Only the test geometry and classifier input width change; source attention,
    initialization, network forwards and lifecycle functions execute unchanged.
    """
    if not (SOURCE / f'vit_{method}.py').is_file():
        pytest.skip('Pinned author source is not installed locally')
    scope = dict(math=math, logging=logging, partial=partial, OrderedDict=OrderedDict,
                 torch=torch, nn=nn, F=F, np=np, PatchEmbed=PatchEmbed, Mlp=Mlp,
                 DropPath=DropPath, trunc_normal_=trunc_normal_, lecun_normal_=lecun_normal_,
                 named_apply=named_apply, checkpoint_seq=checkpoint_seq, CosineLinear=CosineLinear)
    path = SOURCE / 'modules/linears.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(tree, str(path), 'exec'), scope)
    selected = {'init_weights_vit_timm', 'init_weights_vit_jax', 'init_weights_vit_moco', 'get_init_weights_vit'}
    path = SOURCE / f'vit_{method}.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 or isinstance(node, ast.FunctionDef) and node.name in selected]
    exec(compile(tree, str(path), 'exec'), scope)
    path = SOURCE / f'net_{method}.py'
    tree = ast.parse(path.read_text().replace('nn.Linear(768, self.class_num, bias=True)', 'nn.Linear(8, self.class_num, bias=True)'))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(tree, str(path), 'exec'), scope)

    def create(variant, pretrained=False, **kwargs):
        kwargs.update(GEOMETRY)
        kwargs['num_classes'] = 0 if method == 'cllora' else 1000
        return scope['ViT'](**kwargs)

    scope['_create_vision_transformer'] = create
    args = dict(ARGS, load='synthetic-local-test')
    return scope['Net'](args)


def output(net, method, images, task=0):
    if method == 'ewclora':
        return net(images, use_new=True, register_hook=True)
    if method == 'cllora':
        return net(images, task)
    if method == 'inflora':
        return net(images, get_feat=True, get_cur_feat=True)
    return net(images)


@pytest.mark.parametrize('method', METHODS)
def test_seeded_source_initialization_forward_gradients_and_task_boundary(method):
    torch.manual_seed(1993)
    native = network(method)
    native_rng = torch.get_rng_state().clone()
    torch.manual_seed(1993)
    source = source_network(method)
    assert torch.equal(torch.get_rng_state(), native_rng)
    assert set(native.state_dict()) == set(source.state_dict())
    for name, tensor in native.state_dict().items():
        torch.testing.assert_close(tensor, source.state_dict()[name], atol=0, rtol=0)
    native.update_fc(2)
    update_rng = torch.get_rng_state().clone()
    # CL creates random proxy heads at each update; repeat from the same stream.
    if method == 'cllora':
        torch.manual_seed(41)
        native = network(method)
        before = torch.get_rng_state().clone()
        native.update_fc(2)
        update_rng = torch.get_rng_state().clone()
        torch.manual_seed(41)
        source = source_network(method)
        assert torch.equal(torch.get_rng_state(), before)
    source.update_fc(2)
    assert torch.equal(torch.get_rng_state(), update_rng)
    # Exercise nonzero adapter paths, including SD norm divisions at task two.
    for name, param in native.named_parameters():
        if 'lora' in name and ('B' in name or 'old' in name):
            with torch.no_grad():
                param.fill_(.03)
    source.load_state_dict(native.state_dict(), strict=True)
    images = torch.randn(2, 3, 8, 8)
    actual, expected = output(native, method, images), output(source, method, images)
    for name in ('logits', 'features'):
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    actual['logits'].square().sum().backward()
    expected['logits'].square().sum().backward()
    for (name, param), (_, reference) in zip(native.named_parameters(), source.named_parameters()):
        assert (param.grad is None) == (reference.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, reference.grad, atol=0, rtol=0)
    if method == 'ewclora':
        for native_block, source_block in zip(native.image_encoder.blocks, source.image_encoder.blocks):
            for kind in ('k', 'v'):
                torch.testing.assert_close(getattr(native_block.attn, f'delta_w_{kind}_new_grad'),
                                           getattr(source_block.attn, f'delta_w_{kind}_new_grad'), atol=0, rtol=0)
        before = torch.get_rng_state().clone()
        native.accumulate_and_reset_lora()
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        source.accumulate_and_reset_lora()
        assert torch.equal(after, torch.get_rng_state())
    elif method == 'inflora':
        for native_block, source_block in zip(native.image_encoder.blocks, source.image_encoder.blocks):
            assert native_block.attn.n_cur_matrix == source_block.attn.n_cur_matrix == 10
            torch.testing.assert_close(native_block.attn.matrix, source_block.attn.matrix, atol=0, rtol=0)
            torch.testing.assert_close(native_block.attn.cur_matrix, source_block.attn.cur_matrix, atol=0, rtol=0)
    elif method == 'cllora':
        native.save_old_shared_lora()
        source.save_old_shared_lora()
    for name, tensor in native.state_dict().items():
        torch.testing.assert_close(tensor, source.state_dict()[name], atol=0, rtol=0)
    before = torch.get_rng_state().clone()
    native.update_fc(4)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    source.update_fc(4)
    assert torch.equal(after, torch.get_rng_state())
    actual, expected = output(native, method, images, task=1), output(source, method, images, task=1)
    torch.testing.assert_close(actual['logits'], expected['logits'], atol=0, rtol=0)
    if method == 'cllora':
        labels = torch.tensor([2, 3])
        loader = [(None, images, labels)]
        native.replace_fc(loader)
        source.replace_fc(loader)
        torch.testing.assert_close(native.fc.weight, source.fc.weight, atol=0, rtol=0)
    torch.testing.assert_close(native.interface(images), source.interface(images), atol=0, rtol=0)


@pytest.mark.parametrize('method', METHODS)
def test_backbone_return_variants_share_one_original_forward(method):
    net = network(method)
    net.update_fc(2)
    net.eval()
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        original = net(images)
    assert set(original) == {'logits', 'features'}
    calls = []
    handle = net.image_encoder.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.no_grad():
            for returnt in ('out', 'features', 'both', 'full', 'all'):
                before = len(calls)
                result = net(images, returnt=returnt)
                assert len(calls) == before + 1
                if returnt == 'out':
                    torch.testing.assert_close(result, original['logits'], atol=0, rtol=0)
                elif returnt == 'features':
                    torch.testing.assert_close(result, original['features'], atol=0, rtol=0)
                elif returnt == 'both':
                    torch.testing.assert_close(result[0], original['logits'], atol=0, rtol=0)
                    torch.testing.assert_close(result[1], original['features'], atol=0, rtol=0)
                else:
                    for name, tensor in original.items():
                        torch.testing.assert_close(result[name], tensor, atol=0, rtol=0)
        with pytest.raises(ValueError, match='return type'):
            net(images, returnt='unsupported')
    finally:
        handle.remove()


def test_cl_cosine_classifier_uses_only_selected_branch_and_task_classes():
    classifier = CosineLinear(16, 4)
    with torch.no_grad():
        classifier.weight.copy_(torch.arange(64).reshape(4, 16) / 64)
        classifier.sigma.fill_(2.5)
    features = torch.randn(3, 8)
    expected = 2.5 * F.linear(F.normalize(features), F.normalize(classifier.weight[2:4, 8:16]))
    actual = classifier.forward_all(features, task_id=1, inc=2, feature_dim=8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def save_npz_encoder(net, path, *, prefixed=False):
    """Create source-layout NPZ tensors from a synthetic encoder independently."""
    encoder = net.image_encoder
    array = lambda tensor: tensor.detach().cpu().numpy().copy()
    weights = {'embedding/kernel': array(encoder.patch_embed.proj.weight).transpose(2, 3, 1, 0),
               'embedding/bias': array(encoder.patch_embed.proj.bias), 'cls': array(encoder.cls_token),
               'Transformer/posembed_input/pos_embedding': array(encoder.pos_embed),
               'Transformer/encoder_norm/scale': array(encoder.norm.weight),
               'Transformer/encoder_norm/bias': array(encoder.norm.bias)}
    dimension, heads = encoder.embed_dim, GEOMETRY['num_heads']
    for index, block in enumerate(encoder.blocks):
        base = f'Transformer/encoderblock_{index}/'
        for norm, name in ((block.norm1, 'LayerNorm_0'), (block.norm2, 'LayerNorm_2')):
            weights[base + name + '/scale'] = array(norm.weight)
            weights[base + name + '/bias'] = array(norm.bias)
        attention = base + 'MultiHeadDotProductAttention_1/'
        for part, weight, bias in zip(('query', 'key', 'value'), block.attn.qkv.weight.chunk(3), block.attn.qkv.bias.chunk(3)):
            weights[attention + part + '/kernel'] = array(weight).T.reshape(dimension, heads, dimension // heads)
            weights[attention + part + '/bias'] = array(bias).reshape(heads, dimension // heads)
        weights[attention + 'out/kernel'] = array(block.attn.proj.weight).T.reshape(heads, dimension // heads, dimension)
        weights[attention + 'out/bias'] = array(block.attn.proj.bias)
        for layer in range(2):
            dense = base + f'MlpBlock_3/Dense_{layer}/'
            module = getattr(block.mlp, f'fc{layer + 1}')
            weights[dense + 'kernel'] = array(module.weight).T
            weights[dense + 'bias'] = array(module.bias)
    # These unrelated state names must never overwrite learned initialization.
    weights['head/kernel'] = np.full((dimension, 1000), 99., dtype=np.float32)
    weights['head/bias'] = np.full((1000,), 99., dtype=np.float32)
    np.savez(path, **{('opt/target/' if prefixed else '') + name: value for name, value in weights.items()})


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('prefixed', [False, True])
def test_npz_loading_preserves_rng_and_learned_initialization(tmp_path, method, prefixed):
    pretrained, target = network('ewclora'), network(method)
    path = tmp_path / 'synthetic.npz'
    save_npz_encoder(pretrained, path, prefixed=prefixed)
    before_learned = learned_initial_state(target)
    before_rng = torch.get_rng_state().clone()
    load_pretrained_encoder(target, path)
    assert torch.equal(torch.get_rng_state(), before_rng)
    for name, tensor in canonical_pretrained_state(target).items():
        torch.testing.assert_close(tensor, canonical_pretrained_state(pretrained)[name], atol=0, rtol=0)
    for name, tensor in learned_initial_state(target).items():
        torch.testing.assert_close(tensor, before_learned[name], atol=0, rtol=0)
    assert len(canonical_pretrained_state(target)) == 30
    assert canonical_pretrained_digest(target) == canonical_pretrained_digest(pretrained)


def test_malformed_checkpoint_fails_before_partial_mutation(tmp_path):
    pretrained, target = network('ewclora'), network('ewclora')
    path = tmp_path / 'invalid.npz'
    save_npz_encoder(pretrained, path)
    with np.load(path) as archive:
        values = dict(archive)
    values['Transformer/encoderblock_1/MlpBlock_3/Dense_1/bias'][:] = np.nan
    np.savez(path, **values)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='invalid'):
        load_pretrained_encoder(target, path)
    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


@pytest.mark.parametrize('method', METHODS)
def test_learned_initialization_roundtrip_preserves_pretrained_state(method):
    original, target = network(method), network(method)
    pretrained = canonical_pretrained_digest(target)
    initial = learned_initial_state(original)
    restore_learned_initial_state(target, initial)
    assert canonical_pretrained_digest(target) == pretrained
    for name, tensor in learned_initial_state(target).items():
        torch.testing.assert_close(tensor, initial[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match='keys'):
        restore_learned_initial_state(target, {})


def test_cl_classifier_checkpoint_reconstruction_keeps_rng_and_old_prototypes():
    original, target = network('cllora'), network('cllora')
    original.update_fc(2)
    with torch.no_grad():
        original.fc.weight.fill_(.2)
        original.fc.sigma.fill_(3.)
    old = original.fc.weight.detach().clone()
    original.update_fc(4)
    torch.testing.assert_close(original.fc.weight[:2, :8], old, atol=0, rtol=0)
    assert original.fc.weight[:2, 8:].count_nonzero() == 0
    assert original.fc.sigma.item() == 3.
    before = torch.get_rng_state().clone()
    target.load_state_dict(original.state_dict(), strict=True)
    assert torch.equal(torch.get_rng_state(), before)
    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, original.state_dict()[name], atol=0, rtol=0)


def test_pinned_factory_fails_locally_without_fetching(tmp_path):
    assert SIZES == {'base': (768, 12), 'small': (384, 6), 'tiny': (192, 3)}
    assert method_name('ewc-lora') == 'ewclora'
    assert method_name('sd') == 'sdlora'
    assert method_name('inflora') == 'inflora'
    assert method_name('cl_lora') == 'cllora'
    with pytest.raises(ValueError, match='Unknown'):
        method_name('vanilla')
    with pytest.raises(ValueError, match='vit_size'):
        checkpoint_spec('large')
    with pytest.raises(FileNotFoundError, match='downloads are disabled'):
        build_network('ewc', ARGS, checkpoint=tmp_path / 'absent.npz')
    path = tmp_path / 'wrong.npz'
    np.savez(path, unrelated=np.zeros(1))
    with pytest.raises(ValueError, match='pinned'):
        build_network('sd', ARGS, checkpoint=path)


@pytest.mark.parametrize('vit_size', ['tiny', 'small'])
def test_existing_pinned_sizes_initialize_and_forward_on_cpu(vit_size):
    if not Path(CHECKPOINTS[vit_size]['file']).is_file():
        pytest.skip('Shared pinned AugReg checkpoint is not installed locally')
    net = build_network('ewc', dict(init_cls=10, increment=10, sessions=10, rank=10), vit_size)
    assert net.feature_dim == SIZES[vit_size][0]
    assert len(net.image_encoder.blocks) == 12
    assert len(canonical_pretrained_state(net)) == 150
    assert canonical_pretrained_digest(net) == CHECKPOINTS[vit_size]['canonical_sha256']
    net.update_fc(10)
    net.eval()
    with torch.no_grad():
        result = net(torch.zeros(1, 3, 224, 224), use_new=True)
    assert result['logits'].shape == (1, 10)
    assert result['features'].shape == (1, SIZES[vit_size][0])
    assert torch.isfinite(result['logits']).all()


@pytest.mark.parametrize('cli_name,source_method', [
    ('ewc-lora', 'ewclora'), ('sd-lora', 'sdlora'), ('inf-lora', 'inflora'), ('cl-lora', 'cllora')])
def test_native_discovery_config_and_dynamic_backbone_dispatch(monkeypatch, cli_name, source_method):
    root = Path(__file__).resolve().parents[1]
    if not (root / 'models' / (cli_name.replace('-', '_') + '.py')).is_file():
        pytest.skip('This native wrapper is not active in the current staged integration')
    import main
    from backbone import get_backbone
    from models import get_model_class, get_model_names
    from models.low_rank_utils import factory

    args = main.parse_args(cmd=[
        '--model', cli_name, '--dataset', 'seq-cifar100-224', '--model_config', 'default',
        '--lora_vit_size', 'tiny', '--lora_rank', '3', '--lora_checkpoint', '/tmp/explicit-local.npz',
        '--lr', '.0123', '--batch_size', '2', '--num_workers', '0', '--device', 'cpu'])
    assert cli_name in get_model_names()
    cls = get_model_class(args)
    assert cls.NAME == cli_name and cls.METHOD == source_method
    assert args.backbone == 'low-rank-vit'
    assert args.low_rank_method == source_method
    assert args.lora_rank == 3 and args.lr == .0123 and args.lora_vit_size == 'tiny'
    assert args.lora_checkpoint == '/tmp/explicit-local.npz'
    assert args.batch_size == 2 and args.num_workers == 0
    # Exercise actual registered constructor dispatch while intercepting only
    # expensive encoder construction. Parsing and dispatch must never fetch.
    captured = {}
    sentinel = nn.Identity()

    def build(method, source_args, vit_size='base', checkpoint=None):
        captured.update(method=method, args=source_args, size=vit_size, checkpoint=checkpoint)
        return sentinel

    monkeypatch.setattr(factory, 'build_network', build)
    args.num_classes = 100
    assert get_backbone(args) is sentinel
    assert captured['method'] == source_method
    assert captured['size'] == 'tiny' and captured['checkpoint'] == '/tmp/explicit-local.npz'
    assert captured['args']['rank'] == 3
    assert captured['args']['init_cls'] == captured['args']['increment'] == 10
    assert captured['args']['sessions'] == 10
