"""Token compression on the CODA-Prompt and SD-LoRA encoders: identity, parity, prefixes and LoRA."""

import copy
from types import SimpleNamespace

import pytest
import torch

from backbone.token_compression import FrozenToCom, attach, compensate, find_encoder
from backbone.token_compression.reducers import merge_tokens
from backbone.vit import VisionTransformer as MammothViT
from models.coda_prompt_utils import vit as coda_vit
from models.low_rank_utils import net_sdlora, vit_sdlora
from models.low_rank_utils.pretrained import canonical_pretrained_state

GEOMETRY = dict(img_size=32, patch_size=4, embed_dim=48, depth=4, num_heads=3)


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


def mammoth():
    return perturbed(MammothViT(num_classes=10, **GEOMETRY), 0)


def coda():
    return SimpleNamespace(feat=perturbed(coda_vit.VisionTransformer(num_classes=10, **GEOMETRY), 1))


def sd_lora():
    return SimpleNamespace(image_encoder=perturbed(net_sdlora.ViT(num_classes=0, n_tasks=3, rank=4, **GEOMETRY), 2))


def share(source, target):
    """Give ``target`` the canonical encoder tensors of ``source``."""
    tensors = canonical_pretrained_state(find_encoder(target)[1])
    with torch.no_grad():
        for name, tensor in canonical_pretrained_state(find_encoder(source)[1]).items():
            tensors[name].copy_(tensor)


def cls_feature(net, x):
    family, encoder = find_encoder(net)
    if family == 'mammoth-vit':
        return encoder.forward_features(x)[:, 0]
    if family == 'coda-vit':
        return encoder(x)[0][:, 0]
    return encoder(x, task_id=0)[:, 0]


def prefix(batch=2, length=4, width=48):
    return torch.randn(batch, length, width).double(), torch.randn(batch, length, width).double()


def test_coda_is_identity_at_degree_zero():
    net, x = coda(), torch.randn(2, 3, 32, 32).double()
    encoder = net.feat
    with torch.no_grad():
        before = encoder(x)[0]
    keys = list(encoder.state_dict())
    controller = attach(net, 'tome')
    with torch.no_grad():
        assert torch.equal(encoder(x)[0], before)
        hidden = encoder.pos_drop(torch.cat((encoder.cls_token.expand(2, -1, -1), encoder.patch_embed(x)), dim=1)
                                  + encoder.pos_embed)
        for index, block in enumerate(encoder.blocks):
            for prompt in (None, prefix()):
                expected = coda_vit.Block.forward(block, hidden, prompt=prompt)
                assert torch.equal(block._reduced(hidden, controller, 0, prompt), expected)
            hidden = coda_vit.Block.forward(block, hidden)
    assert list(encoder.state_dict()) == keys


def test_sd_lora_is_identity_at_degree_zero():
    net, x = sd_lora(), torch.randn(2, 3, 32, 32).double()
    encoder = net.image_encoder
    names = [(name, parameter.requires_grad) for name, parameter in encoder.named_parameters()]
    with torch.no_grad():
        before = [encoder(x, task_id=task) for task in range(3)]
    controller = attach(net, 'tome')
    with torch.no_grad():
        for task in range(3):
            assert torch.equal(encoder(x, task_id=task), before[task])
        hidden = encoder.pos_drop(torch.cat((encoder.cls_token.expand(2, -1, -1), encoder.patch_embed(x)), dim=1)
                                  + encoder.pos_embed)
        for block in encoder.blocks:
            # Task 2 uses the normalized directions of tasks 0 and 1 plus its own update.
            expected = vit_sdlora.Block.forward(block, hidden, 2)
            assert torch.equal(block._reduced(hidden, controller, 0, 2), expected)
            hidden = expected
        with controller.degree(8), pytest.raises(NotImplementedError):
            encoder.blocks[0](hidden, 0, register_hook=True)
    assert [(name, parameter.requires_grad) for name, parameter in encoder.named_parameters()] == names


@pytest.mark.parametrize('operator', ['tome', 'cls-prune'])
def test_families_agree_on_shared_weights(operator):
    reference, x = mammoth(), torch.randn(2, 3, 32, 32).double()
    nets = [reference, coda(), sd_lora()]
    for net in nets[1:]:
        share(reference, net)
    # Neutral SD-LoRA: a zero current-task update leaves the projection untouched.
    with torch.no_grad():
        for block in nets[2].image_encoder.blocks:
            block.attn.lora_B_k[0].weight.zero_()
            block.attn.lora_B_v[0].weight.zero_()
    controllers = [attach(net, operator) for net in nets]
    with torch.no_grad():
        for degree in (0, 8, 16):
            outputs = []
            for net, controller in zip(nets, controllers):
                with controller.degree(degree):
                    outputs.append(cls_feature(net, x))
                    controller.verify_pass()
            assert controllers[0].trace == controllers[1].trace == controllers[2].trace
            for output in outputs[1:]:
                torch.testing.assert_close(output, outputs[0], atol=1e-10, rtol=0)


def test_coda_prefix_matches_rep_attention():
    from models.rep_prompt_utils.vit import REPAttention
    net = coda()
    controller = attach(net, 'tome', degree=8)
    block = net.feat.blocks[1]
    hidden, prompt = torch.randn(2, 65, 48).double(), prefix()
    size = torch.randint(1, 4, (2, 65, 1)).double()
    with torch.no_grad():
        # Independent prefix-aware proportional attention, then the same merge.
        attention, metric = REPAttention.forward(block.attn, block.norm1(hidden), prompt=prompt, mass=size,
                                                 need_metric=True)
        expected, expected_size = merge_tokens(hidden + attention, metric, size, 8)
        expected = expected + block.mlp(block.norm2(expected))
        controller.begin_pass()
        controller.size = size
        actual = block._reduced(hidden, controller, 8, prompt)
    assert metric.shape == (2, 65, 16) and actual.shape == (2, 57, 48)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert torch.equal(controller.size, expected_size)


def test_coda_prunes_image_tokens_only():
    net = coda()
    controller = attach(net, 'cls-prune', degree=8)
    block = net.feat.blocks[0]
    hidden, prompt = torch.randn(2, 65, 48).double(), prefix()
    with torch.no_grad():
        controller.begin_pass()
        with_prefix = block._reduced(hidden, controller, 8, prompt)
        both = net.feat(torch.randn(2, 3, 32, 32).double())[0]
        controller.verify_pass()
    assert with_prefix.shape == (2, 57, 48) and torch.isfinite(with_prefix).all()
    # Four blocks leave 65 - 4 * 8 tokens; pruning never creates sizes.
    assert both.shape == (2, 33, 48) and controller.size is None


def bank_for(tmp_path, depth=4, dim=48, width=32):
    generator = torch.Generator().manual_seed(5)
    state = {f'blocks.{i}.attn.{q}_tocom.adapter_{part}.weight':
             torch.randn((width, dim) if part == 'down' else (dim, width), generator=generator) * .05
             for i in range(depth) for q in ('q', 'v') for part in ('down', 'up')}
    torch.save({'model': state}, tmp_path / 'bank.pt')
    return FrozenToCom.from_checkpoint(tmp_path / 'bank.pt', depth=depth, embed_dim=dim).double()


def test_sd_lora_equals_a_plain_encoder_with_merged_lora(tmp_path):
    reference, net, x = mammoth(), sd_lora(), torch.randn(2, 3, 32, 32).double()
    share(reference, net)
    merged = copy.deepcopy(reference)
    with torch.no_grad():
        for block, plain in zip(net.image_encoder.blocks, merged.blocks):
            attn = block.attn
            factor = float(next(attn.scaling_factor[0].parameters()))
            plain.attn.qkv.weight[48:96] += factor * attn.lora_B_k[0].weight @ attn.lora_A_k[0].weight
            plain.attn.qkv.weight[96:] += factor * attn.lora_B_v[0].weight @ attn.lora_A_v[0].weight
    bank = bank_for(tmp_path)
    sd, plain = attach(net, 'tome', bank=bank), attach(merged, 'tome', bank=bank)
    with torch.no_grad():
        for degree in (0, 8, 16):
            with sd.degree(degree), plain.degree(degree):
                # The merge metric must be the key after the LoRA update, as on the merged model.
                torch.testing.assert_close(cls_feature(net, x), cls_feature(merged, x), atol=1e-10, rtol=0)
                assert sd.trace == plain.trace
            with compensate(net, 0, degree, .1):
                corrected = cls_feature(net, x)
            with compensate(merged, 0, degree, .1):
                torch.testing.assert_close(corrected, cls_feature(merged, x), atol=1e-10, rtol=0)


def test_sd_lora_gradients_reach_the_current_task():
    net = sd_lora()
    encoder = net.image_encoder.train()
    attach(net, 'tome', degree=8)
    encoder(torch.randn(2, 3, 32, 32).double(), task_id=1).sum().backward()
    attn = encoder.blocks[0].attn
    assert attn.lora_B_k[1].weight.grad.abs().sum() > 0 and attn.lora_A_v[1].weight.grad.abs().sum() > 0
    assert attn.lora_B_k[2].weight.grad is None
