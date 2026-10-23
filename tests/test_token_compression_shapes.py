"""Input shapes: images resized before the patch embedding, position embedding resampled, and their cost."""

import copy
from types import SimpleNamespace

import pytest
import torch
from timm.layers import resample_abs_pos_embed
from torch.nn import functional as F

from backbone.token_compression import (FrozenToCom, attach, compensate, detach, find_encoder, parse_shape, shape_cost,
                                        shape_label, valid_resolution)
from backbone.token_compression.adapters import _resampled
from backbone.token_compression.evaluation import evaluate_shapes
from backbone.token_compression.reducers import OPERATORS
from backbone.vit import VisionTransformer as MammothViT
from models.coda_prompt_utils import vit as coda_vit
from models.low_rank_utils import net_inflora, net_sdlora

GEOMETRY = dict(patch_size=4, embed_dim=48, depth=4, num_heads=3)
NATIVE, GRID = 32, 8

# Encoders that take another resolution, built for a given one, and the network that holds each.
BUILD = {'mammoth-vit': lambda size: MammothViT(num_classes=10, img_size=size, **GEOMETRY),
         'coda-vit': lambda size: coda_vit.VisionTransformer(num_classes=10, img_size=size, **GEOMETRY),
         'inf-lora-vit': lambda size: net_inflora.ViT(num_classes=0, n_tasks=3, rank=4, img_size=size, **GEOMETRY)}
HOLD = {'mammoth-vit': lambda encoder: encoder, 'coda-vit': lambda encoder: SimpleNamespace(feat=encoder),
        'inf-lora-vit': lambda encoder: SimpleNamespace(image_encoder=encoder)}


@pytest.fixture(autouse=True)
def cpu_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def perturbed(module, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn_like(parameter) * .05)
    return module.double().eval()


def network(family, seed=0):
    encoder = perturbed(BUILD[family](NATIVE), seed)
    return HOLD[family](encoder), encoder


def features(family, encoder, x):
    if family == 'mammoth-vit':
        return encoder.forward_features(x)
    if family == 'coda-vit':
        return encoder(x)[0]
    return encoder(x, task_id=0)


def images(count=2, seed=1):
    return torch.randn(count, 3, NATIVE, NATIVE, generator=torch.Generator().manual_seed(seed)).double()


def resampled_reference(table, grid):
    """timm's resampling of a position embedding with a class token (bicubic, antialiased), in double precision."""
    patches = table[:, 1:].reshape(1, GRID, GRID, -1).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=(grid, grid), mode='bicubic', antialias=True)
    return torch.cat((table[:, :1], patches.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)), dim=1)


def bank_for(tmp_path, depth=4, dim=48, width=32):
    generator = torch.Generator().manual_seed(5)
    state = {f'blocks.{i}.attn.{q}_tocom.adapter_{part}.weight':
             torch.randn((width, dim) if part == 'down' else (dim, width), generator=generator) * .05
             for i in range(depth) for q in ('q', 'v') for part in ('down', 'up')}
    torch.save({'model': state}, tmp_path / 'bank.pt')
    return FrozenToCom.from_checkpoint(tmp_path / 'bank.pt', depth=depth, embed_dim=dim).double()


@pytest.mark.parametrize('family', list(BUILD))
def test_native_inputs_run_as_before_and_detach_restores_the_embedding(family):
    net, encoder = network(family)
    x = images()
    with torch.no_grad():
        before = features(family, encoder, x)
    embed = encoder.patch_embed
    hooks, keys = (len(embed._forward_pre_hooks), len(embed._forward_hooks)), list(encoder.state_dict())
    controller = attach(net, 'tome')
    assert (controller.native, controller.patch, controller.resizable, controller.resolution) == (NATIVE, 4, True, None)
    assert list(encoder.state_dict()) == keys and not any(module is encoder for module in embed.modules())
    with torch.no_grad():
        assert torch.equal(features(family, encoder, x), before)
        for resolution in (None, NATIVE):
            with controller.shape(resolution, 0):
                assert torch.equal(features(family, encoder, x), before)
        # The embedding no longer checks the size itself; the inputs it is given are still checked.
        with pytest.raises(ValueError):
            features(family, encoder, x[..., :24, :24])
    detach(net)
    assert (len(embed._forward_pre_hooks), len(embed._forward_hooks)) == hooks and embed.strict_img_size
    assert not hasattr(embed, '_tc_owner') and '_pos_embed' not in encoder.__dict__
    with torch.no_grad():
        assert torch.equal(features(family, encoder, x), before)


@pytest.mark.parametrize('family', list(BUILD))
@pytest.mark.parametrize('resolution', [24, 16])
def test_another_resolution_equals_an_encoder_built_for_it(family, resolution):
    net, encoder = network(family)
    x, grid = images(), resolution // 4
    reference = BUILD[family](resolution).double().eval()
    state = encoder.state_dict()
    state['pos_embed'] = resampled_reference(state['pos_embed'], grid)
    reference.load_state_dict(state)
    small = F.interpolate(x, size=(resolution, resolution), mode='bicubic', align_corners=False, antialias=True)
    controller = attach(net, 'tome')
    with torch.no_grad():
        expected = features(family, reference, small)
        assert expected.shape[1] == grid * grid + 1
        with controller.shape(resolution, 0):
            assert torch.allclose(features(family, encoder, x), expected, rtol=0, atol=1e-12)
        # A run at that resolution: the default of every pass, which a shape can still override.
        controller.default_resolution = resolution
        assert torch.allclose(features(family, encoder, x), expected, rtol=0, atol=1e-12)
        with controller.shape(None, 0):
            assert features(family, encoder, x).shape[1] == GRID * GRID + 1
        # Reduction runs on the tokens of the smaller grid, on its schedule.
        with controller.shape(resolution, 2):
            assert features(family, encoder, x).shape[1] < grid * grid + 1
            controller.verify_pass()
        assert controller.resolution == resolution and controller.current == 0


@pytest.mark.parametrize('family', list(BUILD))
def test_the_gradient_reaches_every_native_position_without_an_interpolation(family):
    # CUDA has no deterministic backward of an interpolation, so none may lie between the loss and the table;
    # the gradient still reaches the native positions the resampled grid is made of, not only the leading rows.
    net, encoder = network(family)
    controller = attach(net, 'tome')
    with controller.shape(24, 0):
        loss = features(family, encoder, images()).square().mean()
    names, seen, stack = set(), set(), [loss.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.add(type(node).__name__)
        stack.extend(child for child, _ in node.next_functions)
    assert not any('Upsample' in name for name in names)
    loss.backward()
    assert encoder.pos_embed.grad[0, 1 + 36:].abs().sum(1).gt(0).all()


def test_the_resampling_is_timms_as_a_fixed_linear_map():
    _, encoder = network('mammoth-vit')
    table, prefix = _resampled(encoder, 6)
    assert prefix == 1 and torch.allclose(table, resampled_reference(encoder.pos_embed, 6), rtol=0, atol=1e-12)
    timm = resample_abs_pos_embed(encoder.pos_embed.float(), (6, 6), old_size=(GRID, GRID), num_prefix_tokens=1)
    assert torch.allclose(table.float(), timm, rtol=0, atol=1e-6)
    upstream = torch.randn_like(table)
    mine = torch.autograd.grad((_resampled(encoder, 6)[0] * upstream).sum(), encoder.pos_embed)[0]
    reference = torch.autograd.grad((resampled_reference(encoder.pos_embed, 6) * upstream).sum(), encoder.pos_embed)[0]
    assert torch.allclose(mine, reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize('family', list(BUILD))
def test_a_copied_encoder_resamples_its_own_position_embedding(family):
    net, encoder = network(family)
    controller = attach(net, 'tome')
    copied = find_encoder(copy.deepcopy(net))[1]
    assert copied.patch_embed._tc_owner is copied and copied.token_compression is controller
    x = images()
    with torch.no_grad(), controller.shape(24, 0):
        before = features(family, copied, x)
        assert torch.equal(features(family, encoder, x), before)
        # Not a constant: the layer norms would remove a shift of every channel.
        encoder.pos_embed.add_(torch.randn_like(encoder.pos_embed))
        assert torch.equal(features(family, copied, x), before)
        assert not torch.allclose(features(family, encoder, x), before)


def test_other_families_keep_their_native_resolution():
    encoder = perturbed(net_sdlora.ViT(num_classes=0, n_tasks=3, rank=4, img_size=NATIVE, **GEOMETRY), 2)
    controller = attach(SimpleNamespace(image_encoder=encoder), 'tome')
    assert not controller.resizable and controller.check_resolution(None) is None
    with pytest.raises(ValueError):
        controller.check_resolution(24)
    with pytest.raises(ValueError):
        with controller.shape(24, 0):
            pass
    assert not hasattr(encoder.patch_embed, '_tc_owner')


def test_banks_correct_at_the_native_resolution_only(tmp_path):
    net, encoder = network('coda-vit')
    controller = attach(net, 'tome', bank=bank_for(tmp_path))
    with controller.shape(24, 0), pytest.raises(ValueError, match='native resolution'):
        with compensate(net, 0, 4, .1):
            pass
    controller.default_resolution = 24
    with pytest.raises(ValueError, match='native resolution'):
        with compensate(net, 0, 4, .1):
            pass
    controller.default_resolution = None
    with torch.no_grad(), compensate(net, 0, 4, .1):
        features('coda-vit', encoder, images())
        controller.verify_pass()


class Learner(torch.nn.Module):
    """The part of a continual model the evaluation uses."""

    def __init__(self, net):
        super().__init__()
        self.net, self.device = net, torch.device('cpu')

    def forward(self, x):
        return self.net(x)


def test_shapes_are_evaluated_as_single_conditions_and_leave_the_model_as_it_was():
    net, _ = network('mammoth-vit')
    model = Learner(net)
    controller = attach(net, 'tome', degree=2)
    controller.default_resolution = 24
    generator = torch.Generator().manual_seed(3)
    loaders = [[(torch.randn(4, 3, NATIVE, NATIVE, generator=generator).double(), torch.randint(0, 6, (4,), generator=generator))
                for _ in range(2)] for _ in range(2)]
    shapes = [(None, 0), (24, 4), (16, 0)]
    model.eval()
    expected, operator = {}, OPERATORS['evit']()
    with torch.no_grad():
        for resolution, degree in shapes:
            values = expected[shape_label(resolution or NATIVE, degree)] = []
            for loader in loaders:
                hits = total = 0
                for inputs, labels in loader:
                    controller.operator = operator
                    with controller.shape(resolution, degree):
                        hits += (model(inputs)[:, :6].argmax(1) == labels).sum().item()
                    total += len(labels)
                values.append(hits / total * 100)
    controller.operator = trained = OPERATORS['tome']()
    model.train()
    state = torch.get_rng_state()
    result = evaluate_shapes(model, loaders, 6, shapes, 'evit')
    assert result == expected and list(result) == ['32:0', '24:4', '16:0']
    assert controller.operator is trained and model.training and torch.equal(torch.get_rng_state(), state)
    assert (controller.resolution, controller.current) == (24, 2)
    for broken in ([(24, 4), (24, 4)], [(18, 0)], [(24, 25)]):
        with pytest.raises(ValueError):
            evaluate_shapes(model, loaders, 6, broken, 'evit')


def test_shapes_are_checked_and_cost_what_the_budget_allows():
    assert parse_shape('176:8') == (176, 8) and shape_label(176, 8) == '176:8'
    assert valid_resolution(24, patch=4, native=NATIVE) == 24
    for broken in ('176', '176:x', '100:8', '240:0', '16:0', '176:25', '176:-1'):
        with pytest.raises(ValueError):
            parse_shape(broken)
    for broken in (24., True, 4):
        with pytest.raises(ValueError):
            valid_resolution(broken, patch=4, native=NATIVE)
    # ViT-Ti block cost relative to 224 pixels without reduction: shapes of the three budgets.
    costs = {(224, 24): .3460, (176, 8): .3471, (192, 12): .3483, (160, 4): .3587, (208, 16): .3706, (144, 0): .3815,
             (128, 0): .2983, (224, 16): .4958, (160, 0): .4770, (224, 8): .7373, (208, 4): .7166, (192, 0): .7083}
    assert all(abs(shape_cost(*shape) - value) < 5e-5 for shape, value in costs.items())
    assert shape_cost(224, 0) == shape_cost(224, 0, dim=768) == 1 and shape_cost(224, 24, dim=768) < shape_cost(224, 24)
    # At 224 pixels the cost is the FLOPs a degree saves, as measured for the plug-in.
    stated = {0: 0, 4: 13, 8: 26, 10: 33, 13: 41, 16: 50}
    assert all(abs((1 - shape_cost(224, degree)) * 100 - percent) < 1 for degree, percent in stated.items())


def test_latency_is_timed_per_shape_and_batch_and_leaves_the_controller_as_it_was():
    from scripts.tocom.imagenet_check import parse_args, shape_latency
    net, _ = network('mammoth-vit')
    controller = attach(net.float(), 'tome', degree=2)
    timings = shape_latency(net, controller, [(NATIVE, 0), (24, 4)], [1, 3], torch.device('cpu'), size=NATIVE,
                            rounds=2, repeats=3, warmup=1)
    assert list(timings) == ['32:0', '24:4'] and all(list(value) == ['1', '3'] for value in timings.values())
    assert all(ms > 0 for value in timings.values() for ms in value.values())
    assert (controller.resolution, controller.current) == (None, 2)
    assert parse_args(['--weights', 'augreg', '--shapes', '176:8', '--latency', '1', '16']).latency == [1, 16]
    for broken in (['--latency', '1'], ['--shapes', '176:8', '--latency', '0'], ['--shapes', '176:8', '--bank', 'x']):
        with pytest.raises(SystemExit):
            parse_args(['--weights', 'augreg', *broken])
