"""Token-compression plug-in contracts: operators, identity at degree 0 and the ToMe reference."""

import copy

import pytest
import torch

from backbone.token_compression import attach, detach, schedule
from backbone.token_compression.reducers import merge_tokens, prune_tokens
from backbone.vit import Attention, Block, VisionTransformer


@pytest.fixture(autouse=True)
def cpu_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def small_vit(seed=0):
    """65 tokens, depth 4; every tensor perturbed so outputs are informative."""
    torch.manual_seed(seed)
    net = VisionTransformer(img_size=32, patch_size=4, embed_dim=48, depth=4, num_heads=3, num_classes=10)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(torch.randn_like(parameter) * .05)
    return net.eval()


def features(net, x):
    return net.forward_head(net.forward_features(x), pre_logits=True)


def test_merge_matches_rep_operator():
    from models.rep_prompt_utils.merge import weighted_merge
    torch.manual_seed(1)
    tokens, metric = torch.randn(3, 65, 48), torch.randn(3, 65, 16)
    size = torch.randint(1, 5, (3, 65, 1)).float()
    for count in (0, 1, 8, 16, 40):
        merged, merged_size = merge_tokens(tokens, metric, size, count)
        expected, expected_size, _ = weighted_merge(tokens, metric, size, count)
        assert torch.equal(merged, expected) and torch.equal(merged_size, expected_size)
    # Without sizes every token starts with unit size.
    merged, merged_size = merge_tokens(tokens, metric, None, 8)
    expected, expected_size, _ = weighted_merge(tokens, metric, torch.ones(3, 65, 1), 8)
    assert torch.equal(merged, expected) and torch.equal(merged_size, expected_size)
    assert merge_tokens(tokens, metric, None, 0) == (tokens, None)


def test_merge_matches_official_tome():
    merge = pytest.importorskip('tome.merge')
    torch.manual_seed(2)
    tokens, metric = torch.randn(2, 65, 48).double(), torch.randn(2, 65, 16).double()
    size = torch.randint(1, 5, (2, 65, 1)).double()
    for count in (1, 8, 16, 40):
        matching, _ = merge.bipartite_soft_matching(metric, count, class_token=True)
        expected, expected_size = merge.merge_wavg(matching, tokens, size)
        merged, merged_size = merge_tokens(tokens, metric, size, count)
        torch.testing.assert_close(merged, expected, atol=1e-12, rtol=0)
        assert torch.equal(merged_size, expected_size)


def test_prune_keeps_cls_and_order():
    tokens = torch.arange(2 * 7 * 1, dtype=torch.float).reshape(2, 7, 1)
    scores = torch.tensor([[.1, .9, .2, .8, .3, .7], [.6, .5, .4, .3, .2, .1]])
    size = torch.arange(1., 8.).reshape(1, 7, 1).repeat(2, 1, 1)
    pruned, pruned_size = prune_tokens(tokens, scores, size, 2)
    # Lowest two scores are patches 1 and 3 (row 0), 5 and 6 (row 1); CLS stays first.
    assert pruned[..., 0].tolist() == [[0., 2., 4., 5., 6.], [7., 8., 9., 10., 11.]]
    assert pruned_size[..., 0].tolist() == [[1., 3., 5., 6., 7.], [1., 2., 3., 4., 5.]]
    assert prune_tokens(tokens, scores, None, 2)[1] is None
    # At most (tokens - 1) // 2 tokens leave.
    assert prune_tokens(tokens, scores, None, 16)[0].shape[1] == 4
    with pytest.raises(ValueError):
        prune_tokens(tokens, scores[:, :-1], None, 1)


def test_attach_is_identity_at_degree_zero():
    net, x = small_vit(), torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = net(x)
    keys, parameters = list(net.state_dict()), [id(p) for p in net.parameters()]
    controller = attach(net, 'tome')
    with torch.no_grad():
        assert torch.equal(net(x), before)
        # The written-out attention path must not drift from the original block.
        hidden = net.norm_pre(net._pos_embed(net.patch_embed(x)))
        for block in net.blocks:
            expected = Block.forward(block, hidden)
            assert torch.equal(block._reduced(hidden, controller, 0), expected)
            hidden = expected
    assert list(net.state_dict()) == keys and [id(p) for p in net.parameters()] == parameters
    assert controller.trace == [65] * 4 and controller.size is None
    detach(net)
    assert all(type(block) is Block for block in net.blocks) and not hasattr(net, 'token_compression')
    with torch.no_grad():
        assert torch.equal(net(x), before)


@pytest.mark.parametrize('operator', ['tome', 'cls-prune'])
def test_schedule_degree_context_and_gradients(operator):
    net, x = small_vit(), torch.randn(2, 3, 32, 32)
    controller = attach(net, operator, degree=8)
    with torch.no_grad():
        compressed = net(x)
        assert controller.trace == schedule(8, 65, 4) == [65, 57, 49, 41]
        controller.verify_pass()
        # Only merging tracks sizes; pruned sequences keep unit tokens.
        assert (controller.size is None) == (operator == 'cls-prune')
        with controller.degree(16):
            assert controller.current == 16
            net(x)
            # The clamp halves the sequence once fewer than 2r + 1 tokens remain.
            assert controller.trace == [65, 49, 33, 17]
            controller.verify_pass()
        with controller.degree(0):
            full = net(x)
        assert controller.current == 8
    assert compressed.shape == full.shape == (2, 10) and torch.isfinite(compressed).all()
    assert not torch.equal(compressed, full)
    net.train()
    net(x).sum().backward()
    assert all(block.attn.qkv.weight.grad.abs().sum() > 0 for block in net.blocks)
    with pytest.raises(ValueError):
        with controller.degree(17):
            pass
    controller.trace = [65, 65, 65, 65]
    with pytest.raises(RuntimeError):
        controller.verify_pass()


def test_matches_official_tome_patch_on_timm():
    tome = pytest.importorskip('tome')
    from timm.models.vision_transformer import VisionTransformer as TimmViT
    net = small_vit().double()
    reference = TimmViT(img_size=32, patch_size=4, embed_dim=48, depth=4, num_heads=3, num_classes=0).double().eval()
    missing, unexpected = reference.load_state_dict(net.state_dict(), strict=False)
    assert not missing and all(key.startswith('head.') for key in unexpected)
    tome.patch.timm(reference, prop_attn=True)
    controller = attach(net, 'tome')
    x = torch.randn(2, 3, 32, 32).double()
    with torch.no_grad():
        for degree in (0, 4, 8, 16):
            reference.r = degree
            with controller.degree(degree):
                torch.testing.assert_close(features(net, x), reference(x), atol=1e-10, rtol=0)


def test_deepcopy_shares_the_controller():
    net = small_vit()
    controller = attach(net, 'tome', degree=4)
    clone = copy.deepcopy(net)
    assert clone.token_compression is controller and all(block._tc is controller for block in clone.blocks)
    assert all(a is not b for a, b in zip(clone.parameters(), net.parameters()))
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad(), controller.degree(8):
        assert torch.equal(clone(x), net(x))


def test_attach_rejects_unknown_structures():
    class OtherAttention(Attention):
        pass

    net = small_vit()
    net.blocks[1].attn.__class__ = OtherAttention
    with pytest.raises(TypeError):
        attach(net, 'tome')
    assert all(type(block) is Block for block in net.blocks)
    net.blocks[1].attn.__class__ = Attention
    with pytest.raises(ValueError):
        attach(net, 'unknown')
    attach(net, 'tome')
    with pytest.raises(RuntimeError):
        attach(net, 'tome')
    with pytest.raises(TypeError):
        attach(torch.nn.Linear(2, 2), 'tome')
    with pytest.raises(NotImplementedError):
        with torch.no_grad(), net.token_compression.degree(8):
            net.blocks[0](torch.randn(1, 65, 48), AB={})


def bank_state(depth=4, dim=48, width=32, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return {f'blocks.{i}.attn.{q}_tocom.adapter_{part}.weight':
            torch.randn((width, dim) if part == 'down' else (dim, width), generator=generator) * .05
            for i in range(depth) for q in ('q', 'v') for part in ('down', 'up')}


def attach_with_bank(tmp_path, net, operator='tome'):
    from backbone.token_compression import FrozenToCom
    state = bank_state()
    torch.save({'model': state}, tmp_path / 'bank.pt')
    bank = FrozenToCom.from_checkpoint(tmp_path / 'bank.pt', depth=4, embed_dim=48)
    return attach(net, operator, bank=bank.double()), state


def test_compensation_equals_merged_weights(tmp_path):
    from backbone.token_compression import compensate
    net, x = small_vit().double(), torch.randn(2, 3, 32, 32).double()
    controller, state = attach_with_bank(tmp_path, net)
    index = torch.arange(32)
    with torch.no_grad():
        for source, target in ((0, 16), (16, 0), (4, 12), (12, 4), (8, 9)):
            # Paper Eq. 5-7: add (or subtract) the products of the gaps between both degrees.
            mask = ((index >= 32 - 2 * target).double() - (index >= 32 - 2 * source).double())
            merged = copy.deepcopy(net)
            for i, block in enumerate(merged.blocks):
                for part, rows in (('q', slice(0, 48)), ('v', slice(96, 144))):
                    down, up = [state[f'blocks.{i}.attn.{part}_tocom.adapter_{name}.weight'].double()
                                for name in ('down', 'up')]
                    block.attn.qkv.weight[rows] += .1 * (up * mask) @ down
            with compensate(net, source, target, .1):
                corrected = net(x)
                controller.verify_pass()
            with controller.degree(target):
                torch.testing.assert_close(corrected, merged(x), atol=1e-10, rtol=0)
                assert not torch.equal(corrected, net(x))


def test_compensation_identity_cleanup_and_guards(tmp_path):
    from backbone.token_compression import compensate
    net, x = small_vit().double(), torch.randn(1, 3, 32, 32).double()
    controller, _ = attach_with_bank(tmp_path, net)
    with torch.no_grad():
        with controller.degree(8):
            plain = net(x)
        for scale in (.1, 0.):
            with compensate(net, 8 if scale else 0, 8, scale):
                assert torch.equal(net(x), plain)
        with compensate(net, 0, 8, .1):
            assert controller.current == 8
            with pytest.raises(RuntimeError):
                with compensate(net, 0, 16, .1):
                    pass
        assert controller.current == 0
        assert all(not block.attn.qkv._forward_hooks for block in net.blocks)
        with pytest.raises(ValueError):
            with compensate(net, 0, 17, .1):
                pass
        with pytest.raises(ValueError):
            with compensate(net, 0, 8, -1.):
                pass
    with pytest.raises(RuntimeError):
        with compensate(net, 0, 8, .1):
            pass
    net.train()
    with torch.no_grad(), pytest.raises(RuntimeError):
        with compensate(net, 0, 8, .1):
            pass
    assert all(not block.attn.qkv._forward_hooks for block in net.blocks) and not controller.bank._active


def test_bank_validation(tmp_path):
    from backbone.token_compression import FrozenToCom, compensate
    state = bank_state()
    cases = {'missing': {k: v for k, v in state.items() if 'blocks.3.attn.v_tocom.adapter_up' not in k},
             'shape': {k: (v[:, :-1] if k.endswith('q_tocom.adapter_down.weight') else v) for k, v in state.items()},
             'zero': {k: torch.zeros_like(v) for k, v in state.items()},
             'nan': {k: v * float('nan') for k, v in state.items()}}
    for name, broken in cases.items():
        torch.save({'model': broken}, tmp_path / f'{name}.pt')
        with pytest.raises(ValueError):
            FrozenToCom.from_checkpoint(tmp_path / f'{name}.pt', depth=4, embed_dim=48)
    torch.save({'model': state}, tmp_path / 'bank.pt')
    bank = FrozenToCom.from_checkpoint(tmp_path / 'bank.pt', depth=4, embed_dim=48)
    assert bank.tensor_count == 4 * 2 * 2 * 48 * 32 and len(bank.checkpoint_hash) == 64
    with pytest.raises(ValueError):
        FrozenToCom.from_checkpoint(tmp_path / 'bank.pt', depth=12, embed_dim=48)
    net = small_vit()
    net.blocks = net.blocks[:3]
    with pytest.raises(ValueError):
        attach(net, 'tome', bank=bank)
    net = small_vit()
    attach(net, 'tome')
    with torch.no_grad(), pytest.raises(RuntimeError):
        with compensate(net, 0, 8, .1):
            pass


class Toy(torch.nn.Module):
    """The minimal surface the grid needs from a continual model."""

    def __init__(self, net):
        super().__init__()
        self.net, self.device = net, torch.device('cpu')

    def forward(self, x):
        return self.net(x)


def test_grid_matches_single_conditions_and_restores_state(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from backbone.token_compression import OPERATORS, FrozenToCom, compensate
    from backbone.token_compression.evaluation import evaluate_grid
    net = small_vit()
    torch.save({'model': bank_state()}, tmp_path / 'tome.pt')
    torch.save({'model': bank_state(seed=9)}, tmp_path / 'prune.pt')
    banks = {'tome': FrozenToCom.from_checkpoint(tmp_path / 'tome.pt', depth=4, embed_dim=48),
             'cls-prune': FrozenToCom.from_checkpoint(tmp_path / 'prune.pt', depth=4, embed_dim=48)}
    controller = attach(net, 'tome', degree=8, bank=banks['tome'])
    model = Toy(net).train()
    torch.manual_seed(4)
    loaders = [DataLoader(TensorDataset(torch.randn(24, 3, 32, 32), torch.randint(low, low + 5, (24,))), batch_size=10)
               for low in (0, 5)]
    operator, state = controller.operator, torch.get_rng_state()
    grid = evaluate_grid(model, loaders, 10, 8, [0, 8, 16], ['tome', 'cls-prune'], banks, .1)
    assert torch.equal(torch.get_rng_state(), state) and model.training and net.training
    assert controller.operator is operator and controller.bank is banks['tome'] and controller.current == 8
    assert grid['samples'] == [24, 24]

    def accuracy(loader, context):
        hits = 0
        with torch.no_grad(), context:
            for images, labels in loader:
                hits += int((net(images)[:, :10].argmax(1) == labels).sum())
        return hits / 24 * 100

    net.eval()
    assert grid['source'] == [accuracy(loader, controller.degree(8)) for loader in loaders]
    for name in ('tome', 'cls-prune'):
        entry = grid['operators'][name]
        assert entry['off']['8'] == entry['on']['tome']['8'] == entry['on']['cls-prune']['8'] == grid['source']
        controller.operator = OPERATORS[name]()
        for target in (0, 16):
            assert entry['off'][str(target)] == [accuracy(loader, controller.degree(target)) for loader in loaders]
            for bank in banks:
                # A bank of another operator is a transfer condition, evaluated the same way.
                controller.bank = banks[bank]
                assert entry['on'][bank][str(target)] == [
                    accuracy(loader, compensate(net, 8, target, .1)) for loader in loaders]


def test_grid_runs_every_target_for_a_model_with_its_own_inference(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from backbone.token_compression import FrozenToCom, compensate
    from backbone.token_compression.evaluation import evaluate_grid
    net = small_vit()
    torch.save({'model': bank_state()}, tmp_path / 'tome.pt')
    banks = {'tome': FrozenToCom.from_checkpoint(tmp_path / 'tome.pt', depth=4, embed_dim=48)}
    controller = attach(net, 'tome', degree=8, bank=banks['tome'])
    model = Toy(net).eval()
    torch.manual_seed(4)
    loaders = [DataLoader(TensorDataset(torch.randn(24, 3, 32, 32), torch.randint(0, 10, (24,))), batch_size=10)]
    # No plug-in source degree: the source row is the model as it stands, here at its default degree.
    compensations = {'tome': ('tome', 8, None), 'tome@0': ('tome', (0,), ['tome'])}
    grid = evaluate_grid(model, loaders, 10, None, [0, 8], ['tome', 'cls-prune'], banks, .1, compensations)

    def accuracy(context=None):
        hits = 0
        with torch.no_grad(), context or controller.degree(8):
            for images, labels in loaders[0]:
                hits += int((net(images)[:, :10].argmax(1) == labels).sum())
        return [hits / 24 * 100]

    tome, prune = grid['operators']['tome'], grid['operators']['cls-prune']
    assert grid['source'] == accuracy() and set(tome['on']) == {'tome', 'tome@0'} and set(prune['on']) == {'tome'}
    # Every target is run, also the one that happens to equal the model's own inference.
    assert tome['off']['8'] == tome['on']['tome']['8'] == grid['source']
    assert tome['off']['0'] == accuracy(controller.degree(0))
    assert tome['on']['tome']['0'] == accuracy(compensate(net, 8, 0, .1))
    assert tome['on']['tome@0']['8'] == accuracy(compensate(net, 0, 8, .1))
    assert tome['on']['tome@0']['0'] == tome['off']['0']
    with pytest.raises(ValueError):
        evaluate_grid(model, loaders, 10, None, [0, 8], ['tome'], banks, .1)
    with pytest.raises(ValueError):
        evaluate_grid(model, loaders, 10, 8, [0, 8], ['tome'], banks, .1, {'x': ('evit', 8, None)})


def test_microbatches_follow_the_official_repeated_augmentation_replicas():
    from timm.data.distributed_sampler import RepeatAugSampler
    from scripts.tocom.train_bank import MICROBATCH, REPLICAS, epoch_microbatches
    length, epoch = 20000, 3
    steps = epoch_microbatches(length, epoch)
    assert len(steps) == length // 256 * 256 // REPLICAS // MICROBATCH and all(len(step) == REPLICAS for step in steps)
    for rank in range(REPLICAS):
        sampler = RepeatAugSampler(range(length), num_replicas=REPLICAS, rank=rank, num_repeats=3)
        sampler.set_epoch(epoch)
        expected = list(sampler)
        expected = [int(index) for index in expected[:len(expected) // MICROBATCH * MICROBATCH]]
        assert [index for step in steps for index in step[rank]] == expected
    assert epoch_microbatches(length, 4) != steps


def official_evit_block(x, cls_attn, left_tokens):
    """Token reorganization of the official EViT ``Block.forward`` with ``fuse_token`` (Apache-2.0)."""
    def complement_idx(idx, dim):
        a = torch.arange(dim, device=idx.device)
        ndim = idx.ndim
        dims = idx.shape
        n_idx = dims[-1]
        dims = dims[:-1] + (-1, )
        for i in range(1, ndim):
            a = a.unsqueeze(0)
        a = a.expand(*dims)
        masked = torch.scatter(a, -1, idx, 0)
        compl, _ = torch.sort(masked, dim=-1, descending=False)
        compl = compl.permute(-1, *tuple(range(ndim - 1)))
        compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
        return compl

    B, N, C = x.shape
    _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
    index = idx.unsqueeze(-1).expand(-1, -1, C)
    non_cls = x[:, 1:]
    x_others = torch.gather(non_cls, dim=1, index=index)
    compl = complement_idx(idx, N - 1)
    non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
    extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
    return torch.cat([x[:, 0:1], x_others, extra_token], dim=1)


def test_evit_matches_the_official_block():
    from backbone.token_compression.reducers import Signals, fuse_prune_tokens
    torch.manual_seed(6)
    x = torch.randn(3, 65, 48).double()
    q, k = torch.randn(3, 3, 65, 16).double(), torch.randn(3, 3, 65, 16).double()
    signals = Signals(q, k, 16 ** -.5)
    # Official score: CLS row of the softmaxed attention without its own column, averaged over heads.
    attention = ((q @ k.transpose(-2, -1)) * 16 ** -.5).softmax(dim=-1)
    cls_attn = attention[:, :, 0, 1:].mean(dim=1)
    torch.testing.assert_close(signals.cls_attention(), cls_attn, atol=1e-14, rtol=0)
    for count in (1, 8, 16, 40):
        removed = min(count, 32)
        fused, size = fuse_prune_tokens(x, cls_attn, None, count)
        # Removing `removed` tokens net keeps 64 - removed - 1 patches plus the fused token.
        expected = official_evit_block(x, cls_attn, 64 - removed - 1)
        assert size is None and fused.shape == (3, 65 - removed, 48)
        torch.testing.assert_close(fused, expected, atol=1e-12, rtol=0)
    assert fuse_prune_tokens(x, cls_attn, None, 0)[0] is x
    with pytest.raises(ValueError):
        fuse_prune_tokens(x, cls_attn, torch.ones(3, 65, 1).double(), 4)
    weights = cls_attn.clone().requires_grad_()
    fuse_prune_tokens(x, weights, None, 8)[0].sum().backward()
    assert weights.grad.abs().sum() > 0


def test_evit_runs_through_every_family_schedule():
    net, x = small_vit(), torch.randn(2, 3, 32, 32)
    controller = attach(net, 'evit', degree=8)
    with torch.no_grad():
        compressed = net(x)
        controller.verify_pass()
        # The fused token replaces one pruned token, so the schedule equals the other operators'.
        assert controller.trace == [65, 57, 49, 41] and controller.size is None
        with controller.degree(0):
            assert not torch.equal(net(x), compressed)
