"""CPU-only geometry, checkpoint and actual Mammoth two-task training checks."""

from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.cl_runtime import configure_native_environment

configure_native_environment()

from backbone.vit import VisionTransformer
from models.coda_prompt import CodaPrompt
from models.coda_prompt_utils.model import Model
from models.low_rank_utils.factory import CHECKPOINTS, SIZES
from models.low_rank_utils.pretrained import canonical_pretrained_digest
from models.ranpac import RanPAC
from models.ranpac_utils.inc_net import RanPACNet
from utils.training import train_single_epoch
from utils.schedulers import get_scheduler


@pytest.fixture(autouse=True)
def cpu_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('method,size_arg,checkpoint_arg', [
    (CodaPrompt, 'coda_vit_size', 'coda_checkpoint'),
    (RanPAC, 'ranpac_vit_size', 'pretrained_checkpoint'),
])
def test_parser_legacy_defaults_and_three_sizes(method, size_arg, checkpoint_arg):
    parser = method.get_parser(ArgumentParser())
    defaults = parser.parse_args([])
    assert getattr(defaults, size_arg) == 'base'
    assert getattr(defaults, checkpoint_arg) is None
    for size in ('tiny', 'small', 'base'):
        args = parser.parse_args([f'--{size_arg}', size, f'--{checkpoint_arg}', '/shared/weights.npz'])
        assert getattr(args, size_arg) == size
        assert getattr(args, checkpoint_arg) == '/shared/weights.npz'
    with pytest.raises(SystemExit):
        parser.parse_args([f'--{size_arg}', 'large'])
    if method is RanPAC:
        assert defaults.rp_size == 10000 and defaults.pretrain_type == 'in21k'
        assert parser.parse_args(['--ranpac_checkpoint', '/shared/alias.npz']).pretrained_checkpoint == '/shared/alias.npz'


@pytest.mark.parametrize('method,size_arg,checkpoint_arg', [
    ('coda_prompt', 'coda_vit_size', 'coda_checkpoint'),
    ('ranpac', 'ranpac_vit_size', 'pretrained_checkpoint'),
])
def test_framework_parser_discovery_and_explicit_cli_precedence(monkeypatch, method, size_arg, checkpoint_arg):
    from main import parse_args
    from models import get_model_class

    monkeypatch.setattr('sys.argv', ['main.py', '--model', method, '--model_config', 'default',
                                  '--dataset', 'seq-cifar100-224', '--dataset_config', 'unified',
                                  '--backbone', 'vit-augreg', '--vit_size', 'tiny', '--vit_checkpoint', CHECKPOINTS['tiny']['file'],
                                  f'--{size_arg}', 'tiny', f'--{checkpoint_arg}', CHECKPOINTS['tiny']['file'],
                                  '--n_epochs', '2', '--batch_size', '16', '--lr', '.001'])
    args = parse_args()
    assert get_model_class(Namespace(model=method)).NAME == method
    assert getattr(args, size_arg) == 'tiny'
    assert getattr(args, checkpoint_arg) == CHECKPOINTS['tiny']['file']
    assert (args.backbone, args.vit_size, args.dataset_config, args.n_epochs, args.batch_size) == ('vit-augreg', 'tiny', 'unified', 2, 16)


def meta_context(monkeypatch):
    linspace = torch.linspace
    monkeypatch.setattr(torch, 'linspace', lambda *args, **kwargs: linspace(*args, **dict(kwargs, device='cpu')))
    return torch.device('meta')


@pytest.mark.parametrize('size', ('tiny', 'small', 'base'))
def test_coda_full_geometry_and_task_capacity(monkeypatch, size):
    dimension, heads = SIZES[size]
    with meta_context(monkeypatch):
        model = Model(num_classes=100, prompt_param=[10, [100, 8, 0]], vit_size=size)
    assert model.feat.embed_dim == dimension and model.feat.num_heads == heads
    assert len(model.feat.blocks) == 12 and model.last.in_features == dimension
    assert model.prompt.n_tasks == 10 and model.prompt.key_d == dimension
    assert model.prompt.e_layers == [0, 1, 2, 3, 4]
    for layer in model.prompt.e_layers:
        assert getattr(model.prompt, f'e_k_{layer}').shape == (100, dimension)
        assert getattr(model.prompt, f'e_a_{layer}').shape == (100, dimension)
        assert getattr(model.prompt, f'e_p_{layer}').shape == (100, 8, dimension)
    assert all(parameter.requires_grad for parameter in model.feat.parameters())


@pytest.mark.parametrize('size', ('tiny', 'small', 'base'))
def test_ranpac_full_geometry_and_only_adapters_trainable(monkeypatch, size):
    dimension, heads = SIZES[size]
    with meta_context(monkeypatch):
        backbone = VisionTransformer(patch_size=16, embed_dim=dimension, depth=12, num_heads=heads,
                                     mlp_ratio=4, qkv_bias=True, num_classes=100)
        net = RanPACNet(backbone)
        net.to('meta')
        net.update_fc(10)
    assert net.convnet.embed_dim == dimension and len(net.convnet.blocks) == 12
    assert net.convnet.blocks[0].attn.num_heads == heads
    assert net.fc.weight.shape == (10, dimension)
    assert net.convnet.blocks[0].adaptmlp.down_proj.weight.shape == (64, dimension)
    assert net.convnet.blocks[0].adaptmlp.up_proj.weight.shape == (dimension, 64)
    trainable = {name for name, parameter in net.convnet.named_parameters() if parameter.requires_grad}
    assert trainable and all('adaptmlp' in name for name in trainable)
    assert sum(parameter.numel() for parameter in net.convnet.parameters() if parameter.requires_grad) == 12 * (129 * dimension + 64)


@pytest.mark.parametrize('size', ('tiny', 'small', 'base'))
def test_ranpac_qkv_conversion_slices_each_dimension_exactly(monkeypatch, size):
    import models.ranpac_utils.inc_net as module

    dimension, heads = SIZES[size]

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
            self.embed_dim, self.num_heads = dimension, heads
            self.head = nn.Linear(2, 3)
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(2, 3 * dimension)
            block.mlp = nn.Module()
            block.mlp.fc1 = nn.Linear(2, 2)
            self.blocks = nn.ModuleList([block])
            with torch.no_grad():
                block.attn.qkv.weight.copy_(torch.arange(6 * dimension).reshape(3 * dimension, 2))
                block.attn.qkv.bias.copy_(torch.arange(3 * dimension))

    class Receiver(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.options = kwargs
            self.adaptmlp = nn.Linear(2, 2)

        def load_state_dict(self, state, strict=False):
            self.loaded = state
            return ['adaptmlp.weight', 'adaptmlp.bias'], []

    monkeypatch.setattr(module, 'VisionTransformer', Receiver)
    backbone = Backbone()
    weight, bias = backbone.blocks[0].attn.qkv.weight.clone(), backbone.blocks[0].attn.qkv.bias.clone()
    converted = RanPACNet(backbone).convnet
    assert converted.options['embed_dim'] == dimension and converted.options['num_heads'] == heads
    assert converted.options['tuning_config'].d_model == dimension
    for part, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
        torch.testing.assert_close(converted.loaded[f'blocks.0.attn.{name}.weight'], weight[part * dimension:(part + 1) * dimension], rtol=0, atol=0)
        torch.testing.assert_close(converted.loaded[f'blocks.0.attn.{name}.bias'], bias[part * dimension:(part + 1) * dimension], rtol=0, atol=0)


def test_coda_pinned_tiny_encoder_exact_and_bad_file_rejected_before_allocation(monkeypatch, tmp_path):
    model = Model(num_classes=100, prompt_param=[10, [100, 8, 0]], vit_size='tiny', checkpoint=CHECKPOINTS['tiny']['file'])
    assert canonical_pretrained_digest(model.feat) == CHECKPOINTS['tiny']['canonical_sha256']
    assert model.pretrained_sha256 == CHECKPOINTS['tiny']['sha256']
    assert all(parameter.requires_grad for parameter in model.feat.parameters())
    del model
    monkeypatch.setattr('models.coda_prompt_utils.model.VisionTransformer', lambda **kwargs: pytest.fail('allocated before file validation'))
    bad = tmp_path / 'wrong.npz'
    bad.write_bytes(b'wrong pinned file')
    with pytest.raises(ValueError, match='pinned'):
        Model(prompt_param=[10, [100, 8, 0]], vit_size='tiny', checkpoint=bad)
    with pytest.raises(ValueError, match='pinned'):
        Model(prompt_param=[10, [100, 8, 0]], vit_size='small', checkpoint=CHECKPOINTS['tiny']['file'])


class SyntheticSamples(Dataset):
    extra_return_fields = []

    def __init__(self, offset):
        self.images = torch.randn(4, 3, 224, 224)
        self.labels = torch.tensor([offset, offset + 1, offset, offset + 1])
        self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.images[index]


def arguments(method):
    args = method.get_parser(ArgumentParser()).parse_args([])
    defaults = dict(n_epochs=2, label_perc=1., nowand=1, device='cpu', lr=.001, optimizer='sgd',
                    optim_wd=0., optim_mom=0., optim_nesterov=False, fitting_mode='epochs',
                    eval_epochs=None, lr_scheduler=None, scheduler_mode='epoch', debug_mode=False, code_optimization=0)
    for name, value in defaults.items():
        setattr(args, name, value)
    return args


@pytest.mark.parametrize('method', (CodaPrompt, RanPAC))
def test_actual_tiny_two_task_two_epoch_meta_training_and_scheduler(monkeypatch, method):
    """Actual 12-block training and task end, with four synthetic examples/task."""
    torch.manual_seed(1993)
    dataset = SimpleNamespace(N_CLASSES=100, N_TASKS=10, N_CLASSES_PER_TASK=10, SETTING='class-il',
                              TEST_TRANSFORM=nn.Identity(), get_offsets=lambda task: (10 * task, 10 * (task + 1)))
    args = arguments(method)
    if method is CodaPrompt:
        args.coda_vit_size, args.coda_checkpoint = 'tiny', CHECKPOINTS['tiny']['file']
        monkeypatch.setattr('models.coda_prompt.get_dataset', lambda *_: pytest.fail('provided dataset was not reused'))
        model = method(nn.Identity(), nn.CrossEntropyLoss(), args, None, dataset)
        encoder = model.net.feat
        expected = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
    else:
        args.ranpac_vit_size, args.pretrained_checkpoint, args.rp_size = 'tiny', CHECKPOINTS['tiny']['file'], 2
        backbone = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=100)
        model = method(backbone, nn.CrossEntropyLoss(), args, None, dataset)
        encoder = model.net._network.convnet
        expected = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters() if not parameter.requires_grad}
        assert encoder.canonical_pretrained_sha256 == CHECKPOINTS['tiny']['canonical_sha256']
    model.to(torch.device('cpu'))
    previous_optimizer = None
    solve_calls = []
    original_solve = torch.linalg.solve

    def solve(matrix, rhs, *a, **kw):
        solve_calls.append((tuple(matrix.shape), tuple(rhs.shape)))
        return original_solve(matrix, rhs, *a, **kw)

    monkeypatch.setattr(torch.linalg, 'solve', solve)
    old_prompt = None
    for task in range(2):
        dataset.train_loader = DataLoader(SyntheticSamples(task * 10), batch_size=2, shuffle=False)
        model.meta_begin_task(dataset)
        # This is the same branch used by utils.training.train after meta_begin_task.
        scheduler = get_scheduler(model, args, reload_optim=True) if not hasattr(model, 'scheduler') else model.custom_scheduler
        assert scheduler is model.custom_scheduler and scheduler.optimizer is model.opt
        if task == 1 and method is CodaPrompt:
            assert model.opt is not previous_optimizer
            assert model.net.prompt.task_count == 1 and model.net.prompt.n_tasks == 10
        old_last_epoch = scheduler.last_epoch
        old_lr = model.opt.param_groups[0]['lr']
        step_calls = []
        original_step = model.opt.step

        def step(*a, _native=original_step, _calls=step_calls, **kw):
            _calls.append(1)
            return _native(*a, **kw)

        monkeypatch.setattr(model.opt, 'step', step)
        for epoch in range(2):
            model.meta_begin_epoch(epoch, dataset)
            with tqdm(disable=True) as progress:
                train_single_epoch(model, dataset.train_loader, args, epoch, progress, system_tracker=lambda: None, scheduler=scheduler)
            model.meta_end_epoch(epoch, dataset)
        assert scheduler.last_epoch == old_last_epoch + 2
        assert model.opt.param_groups[0]['lr'] != old_lr
        assert model.task_iteration == 4
        assert len(step_calls) == (0 if method is RanPAC and task == 1 else 4)
        if method is CodaPrompt:
            assert encoder.blocks[0].attn.qkv.weight.requires_grad
            assert encoder.blocks[0].attn.qkv.weight.grad is not None
            optimized = {id(parameter) for group in model.opt.param_groups for parameter in group['params']}
            assert not any(id(parameter) in optimized for parameter in encoder.parameters())
        previous_optimizer = model.opt
        model.meta_end_task(dataset)
        assert model.current_task == task + 1
        if method is CodaPrompt:
            if old_prompt is not None:
                for name, tensor in old_prompt.items():
                    torch.testing.assert_close(dict(model.net.prompt.named_parameters())[name][:10], tensor, rtol=0, atol=0)
            old_prompt = {name: parameter[:10].detach().clone() for name, parameter in model.net.prompt.named_parameters()}
        with torch.no_grad():
            output = model(dataset.train_loader.dataset.images[:1])
        assert output.shape == (1, (task + 1) * 10) and torch.isfinite(output).all()
        if method is RanPAC:
            assert len(solve_calls) == (task + 1) * 18
            assert solve_calls[-18:] == [((2, 2), (2, 100))] * 18
            assert not any(parameter.requires_grad for parameter in encoder.parameters())
            assert model.net._network.fc.W_rand.shape == (192, 2)
            assert model.G.shape == (2, 2) and model.Q.shape == (2, 100)
    for name, tensor in expected.items():
        torch.testing.assert_close(dict(encoder.named_parameters())[name], tensor, rtol=0, atol=0)


def test_ranpac_size_mismatch_fails_before_loading_or_adapters(monkeypatch):
    backbone = SimpleNamespace(embed_dim=192, num_heads=3)
    monkeypatch.setattr('models.ranpac.RanPAC_Model', lambda *_: pytest.fail('constructed mismatched adapters'))
    with pytest.raises(ValueError, match='matching --vit_size'):
        RanPAC(backbone, None, Namespace(ranpac_vit_size='small'), None)
