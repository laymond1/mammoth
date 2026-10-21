"""Token compression on the InfLoRA, EWC-LoRA, CL-LoRA, CPS-Prompt, REP and EASE encoders."""

import copy
from types import SimpleNamespace

import pytest
import torch

from backbone.token_compression import FrozenToCom, attach, compensate, detach, find_encoder, schedule
from backbone.vit import VisionTransformer as MammothViT
from models.coda_prompt_utils import vit as coda_vit
from models.low_rank_utils import net_cllora, net_ewclora, net_inflora, vit_cllora, vit_ewclora, vit_inflora
from models.low_rank_utils.pretrained import canonical_pretrained_state

GEOMETRY = dict(img_size=32, patch_size=4, embed_dim=48, depth=6, num_heads=3)
TINY = dict(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3)


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


def mammoth(geometry=GEOMETRY):
    return perturbed(MammothViT(num_classes=10, **geometry), 0)


def inf_lora():
    return SimpleNamespace(image_encoder=perturbed(net_inflora.ViT(num_classes=0, n_tasks=3, rank=4, **GEOMETRY), 1))


def ewc_lora():
    return SimpleNamespace(image_encoder=perturbed(net_ewclora.ViT(num_classes=0, n_tasks=3, rank=4, **GEOMETRY), 2))


def cl_lora():
    # Shared blocks first, then task-specific ones, as CL-LoRA places them (here 3 and 3 of 6).
    return SimpleNamespace(image_encoder=perturbed(net_cllora.ViT(num_classes=0, n_tasks=3, rank=4, msa=[1, 0, 1],
                                                                  shared_pos=[0, 1, 2], specific_pos=[3, 4, 5],
                                                                  **GEOMETRY), 6))


def cps():
    from models.cps_prompt_utils.model import PromptModel
    return perturbed(PromptModel(num_classes=10, n_tasks=2, pool_size=4, prompt_length=4, layers=(0, 1, 2),
                                 pretrained=False, geometry=GEOMETRY), 3)


def rep(**options):
    from models.rep_prompt_utils.model import REPModel
    return perturbed(REPModel(num_classes=10, n_tasks=2, pool_size=4, prompt_length=4, atom=False,
                              pretrained=False, geometry=GEOMETRY, query_geometry=GEOMETRY, **options), 4)


def ease(adapters=2):
    from backbone.ease_vit import EaseViT
    from models.ease_utils.network import EaseNetwork
    net = EaseNetwork(EaseViT(0, vit_size='tiny', pretrained=False, adapter_size=8))
    for _ in range(adapters):
        net.backbone.start_task()
    return perturbed(net, 5)


def share(source, target):
    """Give ``target`` the canonical encoder tensors of ``source``."""
    tensors = canonical_pretrained_state(find_encoder(target)[1])
    with torch.no_grad():
        for name, tensor in canonical_pretrained_state(find_encoder(source)[1]).items():
            tensors[name].copy_(tensor)


def embedded(encoder, x):
    return encoder.pos_drop(torch.cat((encoder.cls_token.expand(x.shape[0], -1, -1), encoder.patch_embed(x)), dim=1)
                            + encoder.pos_embed)


def bank_for(tmp_path, depth=6, dim=48, width=32):
    generator = torch.Generator().manual_seed(5)
    state = {f'blocks.{i}.attn.{q}_tocom.adapter_{part}.weight':
             torch.randn((width, dim) if part == 'down' else (dim, width), generator=generator) * .05
             for i in range(depth) for q in ('q', 'v') for part in ('down', 'up')}
    torch.save({'model': state}, tmp_path / 'bank.pt')
    return FrozenToCom.from_checkpoint(tmp_path / 'bank.pt', depth=depth, embed_dim=dim).double()


def test_every_family_is_told_apart():
    from models.low_rank_utils import net_sdlora
    nets = {'mammoth-vit': mammoth(), 'inf-lora-vit': inf_lora(), 'ewc-lora-vit': ewc_lora(), 'cl-lora-vit': cl_lora(),
            'cps-vit': cps(),
            'rep-vit': rep(), 'ease-vit': ease(1),
            'coda-vit': SimpleNamespace(feat=coda_vit.VisionTransformer(num_classes=10, **GEOMETRY)),
            'sd-lora-vit': SimpleNamespace(image_encoder=net_sdlora.ViT(num_classes=0, n_tasks=3, rank=4, **GEOMETRY))}
    for name, net in nets.items():
        assert find_encoder(net)[0] == name
        attach(net, 'tome')
        # The patched host classes are still recognised, and detaching restores the originals.
        assert find_encoder(net)[0] == name
        detach(net)
        assert find_encoder(net)[0] == name
    with pytest.raises(TypeError):
        find_encoder(SimpleNamespace(feat=torch.nn.Identity()))


@pytest.mark.parametrize('build, block_class, arguments', [(inf_lora, vit_inflora.Block, (2,)),
                                                           (ewc_lora, vit_ewclora.Block, (True,)),
                                                           (ewc_lora, vit_ewclora.Block, (False,))])
def test_low_rank_is_identity_at_degree_zero(build, block_class, arguments):
    net, x = build(), torch.randn(2, 3, 32, 32).double()
    encoder = net.image_encoder
    names = [(name, parameter.requires_grad) for name, parameter in encoder.named_parameters()]
    with torch.no_grad():
        before = encoder(x, *arguments)
    controller = attach(net, 'tome')
    with torch.no_grad():
        assert torch.equal(encoder(x, *arguments), before)
        hidden = embedded(encoder, x)
        for block in encoder.blocks:
            expected = block_class.forward(block, hidden, *arguments)
            assert torch.equal(block._reduced(hidden, controller, 0, *arguments), expected)
            hidden = expected
    assert [(name, parameter.requires_grad) for name, parameter in encoder.named_parameters()] == names


@pytest.mark.parametrize('family', ['inf', 'ewc'])
def test_low_rank_equals_a_plain_encoder_with_merged_updates(tmp_path, family):
    reference, x = mammoth(), torch.randn(2, 3, 32, 32).double()
    net = inf_lora() if family == 'inf' else ewc_lora()
    share(reference, net)
    merged = copy.deepcopy(reference)
    with torch.no_grad():
        for block, plain in zip(net.image_encoder.blocks, merged.blocks):
            attn = block.attn
            if family == 'inf':
                # Tasks 0 and 1 are active at task id 1.
                update_k = sum(attn.lora_B_k[t].weight @ attn.lora_A_k[t].weight for t in range(2))
                update_v = sum(attn.lora_B_v[t].weight @ attn.lora_A_v[t].weight for t in range(2))
            else:
                update_k = attn.lora_B_k.weight @ attn.lora_A_k.weight + attn.lora_new_B_k.weight @ attn.lora_new_A_k.weight
                update_v = attn.lora_B_v.weight @ attn.lora_A_v.weight + attn.lora_new_B_v.weight @ attn.lora_new_A_v.weight
            plain.attn.qkv.weight[48:96] += update_k
            plain.attn.qkv.weight[96:] += update_v
    bank = bank_for(tmp_path)

    def feature():
        return net.image_encoder(x, 1)[:, 0] if family == 'inf' else net.image_encoder(x, True)[:, 0]

    for operator in ('tome', 'evit'):
        low, plain = attach(net, operator, bank=bank), attach(merged, operator, bank=bank)
        with torch.no_grad():
            for degree in (0, 8, 16):
                with low.degree(degree), plain.degree(degree):
                    # The reduction signals must come from the keys after the low-rank update.
                    torch.testing.assert_close(feature(), merged.forward_features(x)[:, 0], atol=1e-10, rtol=0)
                    low.verify_pass()
                    assert low.trace == plain.trace == schedule(degree, 65, 6)
                with compensate(net, 0, degree, .1):
                    corrected = feature()
                with compensate(merged, 0, degree, .1):
                    torch.testing.assert_close(corrected, merged.forward_features(x)[:, 0], atol=1e-10, rtol=0)
        detach(net)
        detach(merged)


def cl_lora_blocks(encoder, task, use_new):
    """The block arguments CL-LoRA's forward_feature gives each block."""
    for index, block in enumerate(encoder.blocks):
        weight = encoder.block_weights[task][encoder.specific_pos.index(index)] if index in encoder.specific_pos else None
        yield block, (task, weight, use_new)


@pytest.mark.parametrize('task, use_new', [(1, True), (2, False)])
def test_cl_lora_is_identity_at_degree_zero(task, use_new):
    net, x = cl_lora(), torch.randn(2, 3, 32, 32).double()
    encoder = net.image_encoder
    names = [(name, parameter.requires_grad) for name, parameter in encoder.named_parameters()]
    with torch.no_grad():
        before, shared = encoder(x, task, use_new=use_new), encoder(x, shared=True, use_new=use_new)
    controller = attach(net, 'tome')
    with torch.no_grad():
        assert torch.equal(encoder(x, task, use_new=use_new), before)
        assert torch.equal(encoder(x, shared=True, use_new=use_new), shared)
        hidden = embedded(encoder, x)
        for block, arguments in cl_lora_blocks(encoder, task, use_new):
            expected = vit_cllora.Block.forward(block, hidden, *arguments)
            assert torch.equal(block._reduced(hidden, controller, 0, *arguments), expected)
            hidden = expected
    assert [(name, parameter.requires_grad) for name, parameter in encoder.named_parameters()] == names


@pytest.mark.parametrize('use_new', [True, False])
def test_cl_lora_equals_a_plain_encoder_with_merged_updates(tmp_path, use_new):
    reference, x = mammoth(), torch.randn(2, 3, 32, 32).double()
    net, task = cl_lora(), 1
    share(reference, net)
    encoder, merged = net.image_encoder, copy.deepcopy(reference)
    with torch.no_grad():
        for (block, (_, weight, _)), plain in zip(cl_lora_blocks(encoder, task, use_new), merged.blocks):
            attn = block.attn
            if attn.is_shared:
                q, v = (attn.lora_q, attn.lora_v) if use_new else (attn.lora_q_old, attn.lora_v_old)
                scale_q = scale_v = 1.
            else:
                q, v = attn.lora_q[task], attn.lora_v[task]
                scale_q, scale_v = weight[0], weight[2]
            plain.attn.qkv.weight[:48] += scale_q * q.lora_B.weight @ q.lora_A.weight
            plain.attn.qkv.weight[96:] += scale_v * v.lora_B.weight @ v.lora_A.weight
    bank = bank_for(tmp_path)
    for operator in ('tome', 'evit'):
        low, plain = attach(net, operator, bank=bank), attach(merged, operator, bank=bank)
        with torch.no_grad():
            for degree in (0, 8, 16):
                with low.degree(degree), plain.degree(degree):
                    # The reduction signals must come from the queries and keys after the low-rank updates.
                    torch.testing.assert_close(encoder(x, task, use_new=use_new)[:, 0], merged.forward_features(x)[:, 0],
                                               atol=1e-10, rtol=0)
                    low.verify_pass()
                    assert low.trace == plain.trace == schedule(degree, 65, 6)
                with compensate(net, 0, degree, .1):
                    corrected = encoder(x, task, use_new=use_new)[:, 0]
                with compensate(merged, 0, degree, .1):
                    torch.testing.assert_close(corrected, merged.forward_features(x)[:, 0], atol=1e-10, rtol=0)
        detach(net)
        detach(merged)


def test_cl_lora_distillation_passes_reduce_the_shared_blocks_and_train():
    net, x = cl_lora(), torch.randn(2, 3, 32, 32).double()
    encoder = net.image_encoder.train()
    controller = attach(net, 'tome', degree=8)
    # forward_kd runs the shared blocks only, once with the new shared updates and once with the old ones.
    new, old = encoder(x, shared=True, use_new=True), encoder(x, shared=True, use_new=False)
    assert controller.trace == schedule(8, 65, 6)[:3] and new.shape == old.shape == (2, 65 - 3 * 8, 48)
    new.sum().backward()
    shared = encoder.blocks[0].attn
    assert shared.lora_q.lora_B.weight.grad.abs().sum() > 0 and shared.lora_q_old.lora_B.weight.grad is None
    encoder(x, 2, use_new=True).sum().backward()
    assert encoder.blocks[3].attn.lora_v[2].lora_A.weight.grad.abs().sum() > 0 and encoder.block_weights[2].grad.abs().sum() > 0


def test_inf_lora_gathers_covariances_of_the_tokens_it_sees():
    net, x = inf_lora(), torch.randn(2, 3, 32, 32).double()
    encoder = net.image_encoder
    controller = attach(net, 'tome', degree=8)
    for block in encoder.blocks:
        block.attn.cur_matrix = block.attn.cur_matrix.double()
    with torch.no_grad():
        encoder(x, 0, get_cur_feat=True)
        controller.verify_pass()
        first = encoder.blocks[0].norm1(embedded(encoder, x))
    # Block 0 still receives every token; later blocks receive the reduced sequence.
    assert [block.attn.n_cur_matrix for block in encoder.blocks] == [2 * tokens for tokens in controller.trace]
    expected = torch.bmm(first.permute(0, 2, 1), first).sum(dim=0) / (2 * 65)
    torch.testing.assert_close(encoder.blocks[0].attn.cur_matrix, expected, atol=1e-12, rtol=0)
    with pytest.raises(NotImplementedError):
        encoder.blocks[0](embedded(encoder, x), 0, get_feat=True)


def test_ewc_lora_keeps_fisher_hooks_and_gradients_under_reduction():
    net = ewc_lora()
    encoder = net.image_encoder.train()
    attach(net, 'evit', degree=8)
    encoder(torch.randn(2, 3, 32, 32).double(), True, register_hook=True).sum().backward()
    for block in encoder.blocks:
        attn = block.attn
        assert attn.delta_w_k_new_grad.shape == (48, 48) and attn.delta_w_v_new_grad.abs().sum() > 0
        assert attn.lora_new_B_k.weight.grad.abs().sum() > 0 and attn.lora_new_A_v.weight.grad.abs().sum() > 0


def test_cps_is_identity_at_degree_zero_and_matches_coda():
    from backbone.token_compression.coda import CodaBlock
    net, x = cps(), torch.randn(2, 3, 32, 32).double()
    reference = SimpleNamespace(feat=perturbed(coda_vit.VisionTransformer(num_classes=10, **GEOMETRY), 9))
    share(net, reference)
    with torch.no_grad():
        before = net(x)
    controller, plain = attach(net, 'tome'), attach(reference, 'tome')
    prompt = (torch.randn(2, 2, 48).double(), torch.randn(2, 2, 48).double())
    with torch.no_grad():
        assert torch.equal(net(x), before)
        for degree in (8, 16):
            with controller.degree(degree), plain.degree(degree):
                tokens, _, scores, count = net.feat(x)
                controller.verify_pass()
                torch.testing.assert_close(tokens, reference.feat(x)[0], atol=1e-10, rtol=0)
                assert controller.trace == plain.trace and count == 65
                # One importance score per patch token entering the last block.
                assert scores.shape == (2, controller.trace[-1] - 1) and torch.isfinite(scores).all()
        # Prefix handling is the CODA patch's: explicit attention against the fused call.
        hidden, size = torch.randn(2, 65, 48).double(), torch.randint(1, 4, (2, 65, 1)).double()
        for state in (None, size):
            controller.begin_pass()
            plain.begin_pass()
            controller.size = plain.size = state
            actual, _ = net.feat.blocks[1]._reduced(hidden, controller, 8, prompt)
            expected = CodaBlock._reduced(reference.feat.blocks[1], hidden, plain, 8, prompt)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
            assert torch.equal(controller.size, plain.size)


def test_cps_inference_runs_both_passes_reduced_and_training_needs_every_patch():
    net, x = cps(), torch.randn(2, 3, 32, 32).double()
    controller = attach(net, 'evit')
    with torch.no_grad():
        full = net(x)
        with controller.degree(8):
            reduced = net(x)
            controller.verify_pass()
            assert net.diagnostics['query_tokens'] == net.diagnostics['prompted_tokens'] == 65
    assert reduced.shape == full.shape and not torch.equal(reduced, full)
    net.train()
    logits, _ = net(x, train=True)
    # The joint phase samples 60% of the 64 patches once, then runs unreduced at degree 0.
    assert net.diagnostics['prompted_tokens'] == 39 and torch.isfinite(logits).all()
    with controller.degree(8), pytest.raises(ValueError):
        net(x, train=True)


@pytest.mark.parametrize('train', [False, True])
def test_rep_keeps_its_own_schedule_until_a_degree_is_requested(train):
    x = torch.randn(2, 3, 32, 32).double()
    plain = rep(minimum=.3, gamma=.5)
    merging = copy.deepcopy(plain)
    merging.atom, merging.maximum = True, 12
    for net in (plain, merging):
        reference = copy.deepcopy(net)
        controller = attach(net, 'tome')
        for model in (net, reference):
            model.train(train)
            model.step = 7
        with torch.set_grad_enabled(train):
            actual, expected = net(x, train=train), reference(x, train=train)
        # Without a degree REP runs as it is: its own merging and layer dropping, an unreduced query.
        assert torch.equal(actual[0] if train else actual, expected[0] if train else expected)
        assert net.diagnostics == reference.diagnostics and net.diagnostics['query_tokens'] == 65
        assert controller.trace == reference.diagnostics['input_tokens']
    assert sum(merging.diagnostics['merged_tokens']) > 0 == sum(plain.diagnostics['merged_tokens'])
    if train:
        assert not all(merging.diagnostics['executed'])
        return
    # A requested degree replaces REP's merging by the uniform reduction, whatever rep_atom says.
    with torch.no_grad():
        for degree in (0, 8):
            with merging.feat.token_compression.degree(degree), plain.feat.token_compression.degree(degree):
                assert torch.equal(merging(x), plain(x))
                assert merging.diagnostics['input_tokens'] == schedule(degree, 65, 6)


def test_rep_cannot_merge_twice_in_training():
    net, x = rep(), torch.randn(2, 3, 32, 32).double()
    net.atom = True
    attach(net, 'tome', degree=8)
    with torch.no_grad(), pytest.raises(RuntimeError):
        net(x)
    with pytest.raises(TypeError):
        # The swift query encoder must exist: REP's host query bypasses the patched blocks.
        attach(rep(swift=False), 'tome')


@pytest.mark.parametrize('train', [False, True])
def test_rep_matches_its_native_merging_on_a_uniform_schedule(monkeypatch, train):
    from models.rep_prompt_utils import vit as rep_vit
    x = torch.randn(2, 3, 32, 32).double()
    plugin = rep()
    native = copy.deepcopy(plugin)
    monkeypatch.setattr(rep_vit, 'progressive_merges', lambda maximum, depth: [maximum] * depth)
    # The plug-in as the trainer: a default degree, REP's own merging off. Same query for both.
    controller = attach(plugin, 'tome', degree=8)
    with torch.no_grad():
        query = native.query_encoder(x)[0][:, 0] @ native.projection
    outputs = []
    for net, atom in ((native, True), (plugin, False)):
        net.train(train)
        with torch.set_grad_enabled(train):
            outputs.append(net.feat(x, net.prompt, query, train=train, atom=atom, ald=True, maximum=8, step=7,
                                    rng=net.rng, minimum=.3, gamma=.5, alpha=.9, threshold=8))
    (left, _, expected), (right, _, trace) = outputs
    torch.testing.assert_close(right, left, atol=1e-10, rtol=0)
    assert trace == expected
    if train:
        # Layer dropping skipped blocks, which neither reduce tokens nor follow the full schedule.
        assert not all(trace['executed'])
        right.sum().backward()
        # Only the prompts of executed blocks receive gradients.
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in plugin.prompt.parameters())
    else:
        controller.verify_pass()
        assert controller.trace == schedule(8, 65, 6)


def test_rep_reduces_its_query_encoder_at_a_requested_degree():
    net, x = rep(), torch.randn(2, 3, 32, 32).double()
    reference = SimpleNamespace(feat=copy.deepcopy(net.query_encoder))
    controller, plain = attach(net, 'evit'), attach(reference, 'evit')
    with torch.no_grad():
        net(x)
        assert net.diagnostics['query_tokens'] == 65
        with controller.degree(8), plain.degree(8):
            logits = net(x)
            controller.verify_pass()
            # The query encoder is an ordinary frozen encoder under the same controller.
            assert torch.equal(net.query_encoder(x)[0], reference.feat(x)[0])
    assert torch.isfinite(logits).all() and controller.size is None
    assert net.diagnostics['query_tokens'] == 65 - 6 * 8 == net.diagnostics['output_tokens'][-1]
    detach(net)
    assert type(net.query_encoder.blocks[0]) is coda_vit.Block and not hasattr(net.query_encoder, 'token_compression')


def test_cps_bank_corrects_each_pass_from_its_own_training_degree(tmp_path):
    net, x = cps(), torch.randn(2, 3, 32, 32).double()
    controller = attach(net, 'tome', bank=bank_for(tmp_path))
    with torch.no_grad():
        # Piece by piece, one training degree per call: the query pass from 0, the prompted pass from 8.
        with compensate(net, 0, 16, .1):
            query, _, scores, _ = net.feat(x)
        with compensate(net, 8, 16, .1):
            tokens = net.feat(x, prompt=net.prompt, q=query[:, 0], train=False, sampler=net.sampler, scores=scores)[0]
        expected = net.head(tokens[:, 0])
        with compensate(net, (0, 8), 16, .1):
            actual = net(x)
            controller.verify_pass()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        others = {}
        for sources in ((0, 0), (8, 8), (8, 0)):
            with compensate(net, sources, 16, .1):
                others[sources] = net(x)
        with compensate(net, 0, 16, .1):
            assert torch.equal(others[0, 0], net(x))
        assert all(not torch.allclose(actual, other) for other in others.values())
        # One forward per context: a third pass has no training degree, and one pass leaves one unused.
        with pytest.raises(RuntimeError), compensate(net, (0, 8), 16, .1):
            net(x)
            net.feat(x)
        with pytest.raises(RuntimeError), compensate(net, (0, 8), 16, .1):
            net.feat(x)
    assert controller.sources is None and not controller.overridden


def test_rep_bank_corrects_the_query_encoder_and_the_host_separately(tmp_path):
    net, x = rep(), torch.randn(2, 3, 32, 32).double()
    controller = attach(net, 'tome', bank=bank_for(tmp_path))
    with torch.no_grad():
        with compensate(net, 0, 16, .1):
            query = net.query_encoder(x)[0][:, 0] @ net.projection
        with compensate(net, 8, 16, .1):
            tokens = net.feat(x, net.prompt, query, train=False, atom=False, ald=False)[0]
        expected = net.head(tokens[:, 0])
        with compensate(net, (0, 8), 16, .1):
            actual = net(x)
            controller.verify_pass()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        with compensate(net, (8, 8), 16, .1):
            # Only the query encoder differs here, so the bank does reach it.
            assert not torch.allclose(actual, net(x))
        with compensate(net, (0, 0), 16, .1):
            assert not torch.allclose(actual, net(x))


def test_ease_is_identity_at_degree_zero():
    from models.ease_utils.vit import adapter_block_forward
    net, x = ease(), torch.randn(1, 3, 224, 224).double()
    backbone = net.backbone
    keys = list(net.state_dict())
    with torch.no_grad():
        before = backbone.extract_all(x)
    controller = attach(net, 'tome')
    with torch.no_grad():
        assert torch.equal(backbone.extract_all(x), before)
        assert torch.equal(backbone(x, returnt='features'), before[:, 192:])
        hidden = backbone._embed_tokens(x)
        for index, block in enumerate(backbone.backbone.blocks):
            for adapter in (None, backbone.adapters[1][index]):
                expected = adapter_block_forward(block, hidden, adapter)
                assert torch.equal(block._reduced(hidden, controller, 0, adapter), expected)
            hidden = adapter_block_forward(block, hidden, backbone.adapters[0][index])
    assert list(net.state_dict()) == keys and before.shape == (1, 384)


@pytest.mark.parametrize('operator', ['tome', 'evit'])
def test_ease_matches_the_plain_encoder_and_reduces_every_adapter_pass(tmp_path, operator):
    net, x = ease(), torch.randn(1, 3, 224, 224).double()
    reference = mammoth(TINY)
    share(net, reference)
    backbone = net.backbone
    bank = bank_for(tmp_path, depth=12, dim=192)
    controller, plain = attach(net, operator, bank=bank), attach(reference, operator, bank=bank)
    with torch.no_grad():
        for degree in (8, 16):
            with controller.degree(degree), plain.degree(degree):
                # Without an adapter the encoder is the pretrained ViT itself.
                torch.testing.assert_close(backbone.extract_features(x, -1), reference.forward_features(x)[:, 0],
                                           atol=1e-9, rtol=0)
                assert controller.trace == plain.trace == schedule(degree, 197, 12)
                separate = [backbone.extract_features(x, index) for index in range(2)]
                torch.testing.assert_close(backbone.extract_all(x), torch.cat(separate, dim=1), atol=0, rtol=0)
                controller.verify_pass()
            with compensate(net, 0, degree, .1):
                corrected = backbone.extract_features(x, -1)
            with compensate(reference, 0, degree, .1):
                torch.testing.assert_close(corrected, reference.forward_features(x)[:, 0], atol=1e-9, rtol=0)
    assert not torch.allclose(separate[0], separate[1])


@pytest.mark.parametrize('degree', [0, 8])
def test_ease_takes_another_resolution_as_the_plain_encoder_does(degree):
    net, x = ease(), torch.randn(1, 3, 224, 224).double()
    reference = mammoth(TINY)
    share(net, reference)
    backbone = net.backbone
    controller, plain = attach(net, 'tome'), attach(reference, 'tome')
    assert controller.resizable
    with torch.no_grad():
        with controller.shape(160, degree), plain.shape(160, degree):
            torch.testing.assert_close(backbone.extract_features(x, -1), reference.forward_features(x)[:, 0], atol=1e-9, rtol=0)
            assert controller.trace == plain.trace == schedule(degree, 101, 12)
            # Every adapter pass runs at that shape too.
            assert backbone.extract_all(x).shape == (1, 384)
            controller.verify_pass()
        torch.testing.assert_close(backbone.extract_features(x, -1), reference.forward_features(x)[:, 0], atol=1e-9, rtol=0)
        assert len(controller.trace) == 12 and controller.trace[0] == 197
    detach(net)
    assert '_pos_embed' not in backbone.backbone.__dict__


def test_ease_adapters_train_under_reduction():
    net = ease()
    backbone = net.backbone
    backbone.adapters[-1].requires_grad_(True)
    attach(net, 'tome', degree=8)
    backbone.extract_features(torch.randn(1, 3, 224, 224).double(), 1).sum().backward()
    assert backbone.adapters[1][0].down_proj.weight.grad.abs().sum() > 0
    assert backbone.adapters[0][0].down_proj.weight.grad is None


def trained_twice():
    """An EWC-LoRA encoder as it is before the second merge: one update already merged, a new one trained."""
    net = ewc_lora()
    return net, net.image_encoder


@pytest.mark.parametrize('operator, degree', [('tome', 0), ('tome', 2), ('evit', 2)])
def test_ewc_lora_product_merge_keeps_the_function(operator, degree):
    from models.low_rank_utils.ewc import merge_products
    net, encoder = trained_twice()
    x = torch.randn(2, 3, 32, 32).double()
    attach(net, operator, degree=degree)
    keys = {name: value.shape for name, value in encoder.state_dict().items()}
    accumulated = [copy.deepcopy(block.attn.lora_A_k.weight) for block in encoder.blocks]
    with torch.no_grad():
        before = encoder(x, True)
        merge_products(net)
        # After the merge the new factors contribute nothing, with or without them in the forward pass.
        assert torch.allclose(encoder(x, True), before, atol=1e-10) and torch.allclose(encoder(x, False), before, atol=1e-10)
    assert {name: value.shape for name, value in encoder.state_dict().items()} == keys
    for block, old in zip(encoder.blocks, accumulated):
        assert torch.equal(block.attn.lora_A_k.weight, old) and not block.attn.lora_new_B_k.weight.any()


def test_ewc_lora_merges_differ_only_from_the_second_merge_on():
    from models.low_rank_utils.ewc import EWCLifecycle, merge_products
    x = torch.randn(2, 3, 32, 32).double()

    def merged(mode, first):
        net, encoder = trained_twice()
        with torch.no_grad():
            if first:
                # Nothing merged yet: the accumulated factors are zero, as at the end of the first task.
                for block in encoder.blocks:
                    for layer in (*block.attn.lora_A, *block.attn.lora_B):
                        layer.weight.zero_()
            before = encoder(x, True)
            life = EWCLifecycle()
            life.pending_merge.fill_(True)
            torch.manual_seed(11)
            life.finalize_task(SimpleNamespace(net=SimpleNamespace(
                image_encoder=encoder, accumulate_and_reset_lora=lambda: [
                    module.accumulate_and_reset_lora() for module in encoder.modules()
                    if isinstance(module, vit_ewclora.Attention_LoRA)]), args=SimpleNamespace(lora_ewc_merge=mode)))
            assert not bool(life.pending_merge)
            return before, encoder(x, False), [block.attn.lora_new_A_v.weight.clone() for block in encoder.blocks]

    for first in (True, False):
        before, product, fresh = merged('product', first)
        released, factors, same = merged('factors', first)
        assert torch.allclose(product, before, atol=1e-10)
        # The released merge is exact only while nothing has been accumulated; afterwards it adds cross terms.
        assert torch.allclose(factors, released, atol=1e-10) == first
        # Both modes draw the next task's factors from the same random stream.
        assert all(torch.equal(a, b) for a, b in zip(fresh, same))
    # An argument namespace from before the option existed means the released merge.
    life, (net, encoder) = EWCLifecycle(), trained_twice()
    life.pending_merge.fill_(True)
    called = []
    life.finalize_task(SimpleNamespace(net=SimpleNamespace(accumulate_and_reset_lora=lambda: called.append(1)), args=SimpleNamespace()))
    assert called == [1]
