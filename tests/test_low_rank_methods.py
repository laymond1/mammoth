"""Native algorithm checks using 8-dimensional synthetic CPU networks only."""

import copy
from argparse import ArgumentParser
from importlib import import_module
import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from models.low_rank_utils.continual import LowRankContinualModel, source_transforms
from models.low_rank_utils.ewc import compute_fisher
from models.low_rank_utils.inf import InfLifecycle, attention_modules
from utils.conf import get_device
from utils.schedulers import get_scheduler


METHODS = ('ewclora', 'sdlora', 'inflora', 'cllora')


@pytest.fixture(autouse=True)
def synthetic_cpu_environment(monkeypatch):
    monkeypatch.setattr(get_device, 'device', torch.device('cpu'), raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1993)
        yield
    torch.set_num_threads(threads)


class TinyTasks:
    NAME = 'synthetic-low-rank'
    SETTING = 'class-il'
    N_CLASSES = 4
    N_TASKS = 2
    N_CLASSES_PER_TASK = 2
    TEST_TRANSFORM = transforms.Lambda(lambda value: value)

    def __init__(self):
        self.images = torch.rand(8, 3, 8, 8)
        self.set_task(0)

    def set_task(self, task):
        labels = torch.tensor([0, 1] * 4) + task * 2
        data = TensorDataset(self.images, labels, self.images)
        self.train_loader = DataLoader(data, batch_size=4, shuffle=False)

    def get_offsets(self, task):
        return task * 2, (task + 1) * 2


def make_model(method, dataset=None):
    cls = type('TinyLowRank', (LowRankContinualModel,), {'NAME': method, 'METHOD': method})
    parser = cls.get_parser(ArgumentParser())
    args = parser.parse_args([])
    args.label_perc = args.label_perc_by_class = 1.
    args.nowand = 1
    args.num_workers = 0
    args.fitting_mode = 'epochs'
    args.eval_epochs = None
    args.loadcheck = args.start_from = None
    args.lora_rank = 2
    args.lora_shared_pos, args.lora_specific_pos = [0], [1]
    source_args = dict(init_cls=2, increment=2, sessions=2, rank=2, device=['cpu'],
                       msa=args.lora_msa, shared_pos=[0], specific_pos=[1])
    network = import_module(f'models.low_rank_utils.net_{method}').Net(
        source_args, geometry=dict(img_size=8, patch_size=4, embed_dim=8, depth=2, num_heads=2))
    dataset = dataset or TinyTasks()
    return cls(network, F.cross_entropy, args, None, dataset=dataset), dataset


def observe_batch(model, dataset):
    batch = next(iter(dataset.train_loader))
    model.net.train()
    model.meta_begin_epoch(0, dataset)
    loss = model.meta_observe(*batch, epoch=0)
    assert np.isfinite(loss)
    return batch


@pytest.mark.parametrize('method', METHODS)
def test_native_wrappers_reset_task_optimizer_and_use_source_cosine(method):
    model, dataset = make_model(method)
    assert model.opt is None
    model.meta_begin_task(dataset)
    assert (model.current_task, model.n_past_classes, model.n_seen_classes) == (0, 0, 2)
    expected = torch.optim.SGD if method == 'cllora' else torch.optim.Adam
    assert isinstance(model.opt, expected)
    groups = model.opt.param_groups
    assert [group['lr'] for group in groups] == ([model.args.lr, model.args.lora_fc_lr]
                                                if method in ('ewclora', 'sdlora') else [model.args.lr])
    assert {id(p) for group in groups for p in group['params']} == {id(p) for p in model.get_parameters()}
    frozen = {name: p.detach().clone() for name, p in model.net.named_parameters() if not p.requires_grad}
    scheduler = get_scheduler(model, model.args)
    observe_batch(model, dataset)
    scheduler.step()
    assert scheduler.T_max == (30 if method == 'cllora' else 20)
    assert model.task_iteration == 1 and model.epoch_iteration == 1
    for name, value in frozen.items():
        torch.testing.assert_close(dict(model.net.named_parameters())[name], value, rtol=0, atol=0)
    model.meta_end_task(dataset)
    model.eval()
    assert model(dataset.images).shape == (8, 2)
    model.meta_end_eval(dataset, ([], []))
    old_optimizer = model.opt
    dataset.set_task(1)
    model.meta_begin_task(dataset)
    assert (model.current_task, model.n_past_classes, model.n_seen_classes) == (1, 2, 4)
    assert model.opt is not old_optimizer
    assert [group['lr'] for group in model.opt.param_groups] == [group['initial_lr'] for group in scheduler.optimizer.param_groups]


def test_ewc_fisher_uses_delta_weight_hooks_and_batch_weighted_average():
    model, dataset = make_model('ewclora')
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    network = model.net
    expected = [torch.zeros(8, 8) for _ in range(4)]
    samples = 0
    # Independent expression of the reference's empirical batch-mean Fisher.
    network.eval()
    for batch in dataset.train_loader:
        network.zero_grad()
        F.cross_entropy(network(batch[0], use_new=True, register_hook=True)['logits'], batch[1]).backward()
        gradients = []
        for module in network.modules():
            if hasattr(module, 'delta_w_k_new_grad'):
                gradients += [module.delta_w_k_new_grad, module.delta_w_v_new_grad]
        for index, gradient in enumerate(gradients):
            expected[index] += gradient.detach().square() * len(batch[1])
        samples += len(batch[1])
    actual = compute_fisher(network, dataset.train_loader, 0, torch.device('cpu'))
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target / samples, rtol=0, atol=0)
        assert value.shape == (8, 8) and torch.isfinite(value).all() and (value >= 0).all()
    model.meta_end_task(dataset)
    assert int(model.algorithm.count_updates) == 1
    for value, target in zip(model.algorithm.omega_W, actual):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    assert bool(model.algorithm.pending_merge)
    model.meta_end_eval(dataset, ([], []))
    for parameter in network.parameters():
        if getattr(parameter, '_is_new_b', False):
            assert torch.count_nonzero(parameter) == 0


def test_ewc_merge_accumulates_both_factors_and_penalizes_weight_delta():
    model, dataset = make_model('ewclora')
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    previous = {name: parameter.detach().clone() for name, parameter in model.net.named_parameters()}
    model.meta_end_task(dataset)
    model.meta_end_eval(dataset, ([], []))
    for name, parameter in model.net.named_parameters():
        if '.lora_A_' in name or '.lora_B_' in name:
            torch.testing.assert_close(parameter, previous[name] + previous[name.replace('.lora_', '.lora_new_')])
    dataset.set_task(1)
    model.meta_begin_task(dataset)
    with torch.no_grad():
        for parameter in model.net.parameters():
            if getattr(parameter, '_is_new_b', False):
                parameter.fill_(0.02)
    inputs, labels, unaugmented = next(iter(dataset.train_loader))
    logits = model.net(inputs, use_new=True)['logits']
    expected = F.cross_entropy(logits, labels - 2)
    a = [p for p in model.net.parameters() if getattr(p, '_is_new_a', False)]
    b = [p for p in model.net.parameters() if getattr(p, '_is_new_b', False)]
    expected += model.args.lora_ewc_lambda / 2 * sum(torch.sum(omega * (pb @ pa).square())
                                                  for omega, pa, pb in zip(model.algorithm.omega_W, a, b))
    assert model.observe(inputs, labels, unaugmented) == pytest.approx(expected.item())


def test_ewc_evaluation_precedes_non_equivalent_merge_and_pending_checkpoint_resumes():
    model, dataset = make_model('ewclora')
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    model.meta_end_task(dataset)
    model.meta_end_eval(dataset, ([], []))
    dataset.set_task(1)
    model.meta_begin_task(dataset)
    with torch.no_grad():
        for parameter in model.net.parameters():
            if getattr(parameter, '_is_new_b', False):
                parameter.fill_(0.1)
    model.eval()
    reference_logits = model(dataset.images).detach()
    model.meta_end_task(dataset)
    assert bool(model.algorithm.pending_merge)
    torch.testing.assert_close(model(dataset.images), reference_logits, rtol=0, atol=0)
    restored, _ = make_model('ewclora', dataset)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored.eval()
    assert bool(restored.algorithm.pending_merge)
    torch.testing.assert_close(restored(dataset.images), reference_logits, rtol=0, atol=0)
    # This source merge adds A and B independently; the cross terms alter the
    # output, which is why simply merging in native end_task is incorrect.
    model.meta_end_eval(dataset, ([], []))
    restored.algorithm.begin_task(restored, dataset)
    assert not bool(model.algorithm.pending_merge)
    assert not bool(restored.algorithm.pending_merge)
    assert not torch.allclose(model(dataset.images), reference_logits, rtol=0, atol=1e-7)


def test_inflora_initializes_fixed_a_from_svd_and_keeps_old_task_factors():
    model, dataset = make_model('inflora')
    network = model.net
    network.update_fc(2)
    with torch.no_grad():
        for batch in dataset.train_loader:
            network(batch[0], get_cur_feat=True)
        expected = []
        for module in attention_modules(network):
            u, _, _ = torch.linalg.svd(module.cur_matrix)
            expected.append(u[:, :2].T / np.sqrt(3))
            module.cur_matrix.zero_()
            module.n_cur_matrix = 0
    network._cur_task = -1
    model.meta_begin_task(dataset)
    for module, target in zip(attention_modules(network), expected):
        torch.testing.assert_close(module.lora_A_k[0].weight, target, rtol=0, atol=0)
        torch.testing.assert_close(module.lora_A_v[0].weight, target, rtol=0, atol=0)
        assert not module.lora_A_k[0].weight.requires_grad
    observe_batch(model, dataset)
    model.meta_end_task(dataset)
    assert len(model.algorithm.feature_list) == 2
    old = {name: p.detach().clone() for name, p in network.named_parameters()
           if any(key in name for key in ('lora_A_k.0', 'lora_B_k.0', 'lora_A_v.0', 'lora_B_v.0', 'classifier_pool.0'))}
    dataset.set_task(1)
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    for name, value in old.items():
        torch.testing.assert_close(dict(network.named_parameters())[name], value, rtol=0, atol=0)


def test_dualgpm_remove_retain_and_complement_threshold_arithmetic():
    algorithm = InfLifecycle()
    algorithm.update_dualgpm([np.diag(np.array([10., 2., 1., 0.5], dtype=np.float32))], 0.9)
    assert algorithm.project_type == ['remove']
    np.testing.assert_array_equal(algorithm.feature_list[0], np.eye(4, dtype=np.float32)[:, :1])
    algorithm.update_dualgpm([np.eye(4, dtype=np.float32)], 0.95)
    assert algorithm.project_type == ['retain']
    assert algorithm.feature_list[0].shape == (4, 0)
    # A retained subspace loses the captured directions under Eq. (9).
    algorithm.feature_list[0] = torch.eye(4)[:, :2]
    algorithm.update_dualgpm([np.diag(np.array([4., 2., 1., 1.], dtype=np.float32))], 0.9)
    assert algorithm.feature_list[0].shape[1] <= 2


def test_cl_two_optimizer_updates_match_independent_pre_kd_graph_reference():
    model, dataset = make_model('cllora')
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    model.meta_end_task(dataset)
    dataset.set_task(1)
    model.meta_begin_task(dataset)
    reference = copy.deepcopy(model.net)
    optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad],
                                lr=model.args.lr, momentum=0.9, weight_decay=0.0001)
    inputs, labels, unaugmented = next(iter(dataset.train_loader))
    with torch.autograd.graph.allow_mutation_on_saved_tensors():
        loss = F.cross_entropy(reference(inputs, 1)['logits'], labels - 2)
        pred, soft = reference.forward_kd(inputs)
        kd = -(torch.softmax(soft / 2, dim=1) * torch.log_softmax(pred / 2, dim=1)).sum() / len(labels) * 5
        optimizer.zero_grad()
        kd.backward()
        with torch.no_grad():
            for projection in ('q', 'v'):
                attention = reference.image_encoder.blocks[0].attn
                old_b = getattr(attention, f'lora_{projection}_old').lora_B.weight
                new_b = getattr(attention, f'lora_{projection}').lora_B.weight
                norms = old_b.detach().norm(dim=1)
                new_b.grad *= (len(norms) * norms / norms.sum()).unsqueeze(1)
        optimizer.step()
        old, new = reference.image_encoder.block_weights
        ortho = ((old.flatten() / (old.norm() + 1e-8)) * (new.flatten() / (new.norm() + 1e-8))).sum().abs()
        loss += 0.0001 * ortho
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.opt.step = Mock(wraps=model.opt.step)
    actual = model.meta_observe(inputs, labels, unaugmented)
    assert actual == pytest.approx(loss.item())
    assert model.opt.step.call_count == 2
    for (_, value), (_, expected) in zip(model.net.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert torch.isfinite(value).all()


def test_cl_prototypes_use_each_branch_raw_mean_and_preserve_old_rows():
    model, dataset = make_model('cllora')
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    model.meta_end_task(dataset)
    old = model.net.fc.weight.detach().clone()
    dataset.set_task(1)
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    expected = torch.empty(2, 16)
    with torch.no_grad():
        model.net.image_encoder.eval()
        for task in range(2):
            features = model.net.image_encoder(dataset.images, task, use_new=True)[:, 0, :]
            expected[:, task * 8:(task + 1) * 8] = torch.stack([features[::2].mean(0), features[1::2].mean(0)])
    model.meta_end_task(dataset)
    torch.testing.assert_close(model.net.fc.weight[2:], expected)
    torch.testing.assert_close(model.net.fc.weight[:2, :8], old, rtol=0, atol=0)
    assert torch.count_nonzero(model.net.fc.weight[:2, 8:]) == 0
    attention = model.net.image_encoder.blocks[0].attn
    for projection in ('q', 'v'):
        current = getattr(attention, f'lora_{projection}')
        previous = getattr(attention, f'lora_{projection}_old')
        for value, target in zip(previous.parameters(), current.parameters()):
            torch.testing.assert_close(value, target, rtol=0, atol=0)
            assert not value.requires_grad


@pytest.mark.parametrize('method', METHODS)
def test_completed_task_state_checkpoint_restores_predictions_and_next_update(method):
    model, dataset = make_model(method)
    model.meta_begin_task(dataset)
    observe_batch(model, dataset)
    model.meta_end_task(dataset)
    model.eval()
    model.meta_end_eval(dataset, ([], []))
    before = model(dataset.images).detach()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    restored, _ = make_model(method, dataset)
    rng = torch.random.get_rng_state()
    restored.load_state_dict(state)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    restored.eval()
    assert restored.current_task == 1 and restored.net._cur_task == 0
    torch.testing.assert_close(restored(dataset.images), before, rtol=0, atol=0)
    if method == 'ewclora':
        assert int(restored.algorithm.count_updates) == 1
        assert len(restored.algorithm.omega_W) == 4
    if method == 'inflora':
        assert restored.algorithm.project_type == model.algorithm.project_type
        for value, expected in zip(restored.algorithm.feature_list, model.algorithm.feature_list):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    if method == 'cllora':
        assert not restored.net.fc.weight.requires_grad
    dataset.set_task(1)
    restored.meta_begin_task(dataset)
    observe_batch(restored, dataset)


def test_source_cifar_pipeline_uses_bilinear_and_identity_normalization_only_on_instance():
    class Cifar:
        NAME = 'seq-cifar100-224'
        TRANSFORM = 'unchanged-class-transform'
    dataset = Cifar()
    source_transforms(dataset)
    assert Cifar.TRANSFORM == 'unchanged-class-transform'
    assert isinstance(dataset.TRANSFORM.transforms[0], transforms.RandomResizedCrop)
    assert dataset.TRANSFORM.transforms[0].interpolation == transforms.InterpolationMode.BILINEAR
    assert dataset.TEST_TRANSFORM.transforms[0].interpolation == transforms.InterpolationMode.BILINEAR
    image = torch.tensor([[[0.2]], [[0.5]], [[0.9]]])
    torch.testing.assert_close(dataset.get_normalization_transform()(image), image, rtol=0, atol=0)
