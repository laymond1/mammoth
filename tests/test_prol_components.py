"""CPU numerical oracle against PROL's actual pinned classes and engine."""
from argparse import ArgumentParser
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.cl_runtime import configure_native_environment
configure_native_environment()

import torch
from torch import nn

from models.prol_prompt_utils.head import SimpleContinualLinear
from models.prol_prompt import ProlPrompt as NativeProlMethod
from models.prol_prompt_utils.network import ProlNetwork
from models.prol_prompt_utils.prompt import ProlPrompt
from models.prol_prompt_utils.vit import PrefixAttention, ProlVisionTransformer
from scripts.prol_reference_utils import (SOURCE_HASHES, author_args, author_cpu_device,
    begin_author_task, build_author_branches, load_author_code, run_author_epoch, verify_author_source)


SOURCE = Path('/tmp/cl-prol-official-20261008')


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def official():
    if not SOURCE.is_dir():
        pytest.skip('Optional pinned PROL checkout is absent')
    return load_author_code(SOURCE)


def tiny_prompt(cls=ProlPrompt):
    return cls(length=5, embed_dim=12, num_tasks=2, num_classes=20,
        embedding_key='cls', prompt_pool=True, prompt_key=True, top_k=1,
        top_k_l=1, num_layers=2, use_prefix_tune_for_e_prompt=True, num_heads=3)


def tiny_vit(prompted=True):
    return ProlVisionTransformer(num_classes=20, img_size=32, embed_dim=12,
        depth=2, num_heads=3, n_tasks=2 if prompted else 5,
        prompted=prompted, prompt_layers=(0, 1))


def source_vit(official, prompted=True):
    options = dict(num_classes=20, img_size=32, embed_dim=12, depth=2,
                   num_heads=3, num_tasks=2 if prompted else 5)
    if prompted:
        options.update(prompt_length=5, prompt_pool=True, prompt_key=True,
            embedding_key='cls', top_k=1, top_k_l=1, use_e_prompt=True,
            e_prompt_layer_idx=[0, 1], use_prefix_tune_for_e_prompt=True,
            use_prefix_tune_for_g_prompt=True, g_prompt_layer_idx=[])
    return official.VisionTransformerL(**options)


def tensor_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()
            if isinstance(value, torch.Tensor)}


def assert_tensor_state_equal(left, right):
    left, right = tensor_state(left), tensor_state(right)
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def assert_gradients_equal(left, right):
    left, right = dict(left.named_parameters()), dict(right.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].requires_grad == right[name].requires_grad, name
        assert (left[name].grad is None) == (right[name].grad is None), name
        if left[name].grad is not None:
            assert torch.equal(left[name].grad, right[name].grad), name


def assert_output_equal(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, right[key]), key
        else:
            assert value == right[key], key


def test_author_pins_reject_forged_class_hashes(official):
    pins = dict(SOURCE_HASHES)
    pins['promptL.py'] = '0' * 64
    with pytest.raises(ValueError, match='pins differ'):
        verify_author_source(SOURCE, pins)
    assert official.LPrompt.__module__ == 'prol_pinned_author'
    assert Path(official.train_one_epoch_with_aux.__code__.co_filename) == SOURCE / 'engine_prol.py'


@pytest.mark.parametrize('prompted', [False, True])
def test_complete_initialization_and_unused_query_head_match_source(official, prompted):
    torch.manual_seed(1993)
    native = tiny_vit(prompted)
    native_rng = torch.get_rng_state()
    torch.manual_seed(1993)
    reference = source_vit(official, prompted)
    assert torch.equal(torch.get_rng_state(), native_rng)
    assert_tensor_state_equal(native, reference)
    assert native.head.heads[0][1].out_features == (10 if prompted else 4)
    assert native.head.heads[0][0].eps == 1e-5
    assert native.blocks[0].norm1.eps == 1e-6


@pytest.mark.parametrize('train,task,layer', [(train, task, layer)
    for train in (True, False) for task in (0, 1) for layer in (0, 1)])
def test_all_prompt_outputs_and_gradients_match_actual_source(official, train, task, layer):
    torch.manual_seed(20)
    native, reference = tiny_prompt(), tiny_prompt(official.LPrompt)
    reference.load_state_dict(tensor_state(native), strict=True)
    for index in range(task + 1):
        native.process_new_task(index, index + 1)
        reference.process_new_task(index, index + 1)
    tokens, query = torch.randn(3, 5, 12), torch.randn(3, 12)
    labels = torch.tensor([0, 2, 8]) + task * 10
    results = []
    probe = torch.randn(3, 2, 5, 3, 4)
    for module in (native, reference):
        module.train(train)
        result = module(tokens, labels if train else None, task_id=task if train else -1,
                        layer_num=layer, cls_features=query)
        loss = (result['batched_prompt'] * probe).sum()
        for key in ('reduce_sim', 'reduce_sim2'):
            if key in result:
                loss = loss + result[key]
        loss.backward()
        results.append(result)
    assert_output_equal(*results)
    assert_gradients_equal(native, reference)
    assert_tensor_state_equal(native, reference)


@pytest.mark.parametrize('train,task', [(True, 0), (True, 1), (False, 0), (False, 1)])
def test_query_and_full_prompted_forwardA1_match_source_without_oracle(official, train, task):
    torch.manual_seed(56)
    native = {'query': tiny_vit(False), 'prompt': tiny_vit()}
    reference = build_author_branches(official, num_classes=20, embed_dim=12, depth=2,
        num_heads=3, img_size=32, prompt_layers=(0, 1), n_tasks=2)
    for branch in native:
        reference[branch].load_state_dict(tensor_state(native[branch]), strict=True)
    for index in range(task + 1):
        for branches in (native, reference):
            if index:
                with author_cpu_device():
                    branches['prompt'].head.update(10)
            branches['prompt'].e_prompt.process_new_task(index, index + 1)
    # Independently appended source heads use the same saved RNG at the call boundary.
    reference['prompt'].load_state_dict(tensor_state(native['prompt']), strict=True)
    images, labels = torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2]) + task * 10
    results = []
    query_results = []
    for branches in (native, reference):
        branches['prompt'].train(train)
        query_output = branches['query'](images)
        query_results.append(query_output)
        results.append(branches['prompt'].forwardA1(images, labels if train else None,
            task_id=task if train else -1, cls_features=query_output['pre_logits'], train=train))
    assert_output_equal(*query_results)
    assert_output_equal(*results)
    assert results[0]['logits'].shape == (3, (task + 1) * 10)


def test_asymmetric_prefix_attention_and_input_gradients_match_source(official):
    native, reference = PrefixAttention(12, 3), official.PreT_Attention(12, 3, qkv_bias=True)
    reference.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.randn(3, 5, 12)
    prefix = torch.randn(3, 2, 5, 3, 4)
    outputs, gradients = [], []
    for module in (native, reference):
        x, prompt = tokens.clone().requires_grad_(), prefix.clone().requires_grad_()
        output = module(x, prompt)
        output.square().sum().backward()
        outputs.append(output)
        gradients.append((x.grad, prompt.grad))
    assert torch.equal(*outputs)
    assert all(torch.equal(left, right) for left, right in zip(*gradients))
    assert_gradients_equal(native, reference)


def test_actual_source_head_append_freezes_old_head_and_matches_rng(official):
    native, reference = SimpleContinualLinear(12, 10), official.SimpleContinualLinear(12, 10)
    reference.load_state_dict(native.state_dict(), strict=True)
    torch.manual_seed(40)
    native.update(10)
    native_rng = torch.get_rng_state()
    torch.manual_seed(40)
    with author_cpu_device():
        reference.update(10)
    assert torch.equal(torch.get_rng_state(), native_rng)
    assert_tensor_state_equal(native, reference)
    assert not any(parameter.requires_grad for parameter in native.heads[0].parameters())
    assert all(parameter.requires_grad for parameter in native.heads[1].parameters())


def test_source_cross_batch_assignment_and_zero_generator_when_key_zero_absent(official):
    for cls in (ProlPrompt, official.LPrompt):
        prompt = tiny_prompt(cls)
        prompt.process_new_task(0, 1)
        prompt.eval()
        with torch.no_grad():
            prompt.prompt_key.zero_()
            prompt.prompt_key[:, 0] = -1
            prompt.prompt_key[0, 0] = 1
            prompt.prompt_key[1, 0] = 0
            prompt.prompt_key[1, 1] = 1
        query = torch.zeros(2, 12)
        query[0, 0] = 1
        query[1, :2] = torch.tensor([.6, .8])
        captured = []
        original = prompt.compute_att_over_prompt

        def capture(raw, start, stop, layer, similarity, indices):
            captured.append((similarity.clone(), indices.clone()))
            return original(raw, start, stop, layer, similarity, indices)

        prompt.compute_att_over_prompt = capture
        prompt(torch.zeros(2, 5, 12), None, layer_num=0, cls_features=query)
        similarity, indices = captured[-1]
        assert indices[:, 0].tolist() == [0, 1]
        assert torch.equal(similarity[:, 0], torch.ones(2))
        torch.testing.assert_close(similarity[:, 1], torch.full((2,), .8), rtol=0, atol=0)
        query[:, 0] = 0
        query[:, 1] = 1
        result = prompt(torch.zeros(2, 5, 12), None, layer_num=0, cls_features=query)
        assert captured[-1][1][:, 0].tolist() == [1, 1]
        assert torch.count_nonzero(result['batched_prompt']) == 0


def test_second_task_scaler_uses_local_old_row_and_old_key_receives_gradient(official):
    for cls in (ProlPrompt, official.LPrompt):
        prompt = tiny_prompt(cls)
        prompt.process_new_task(0, 1)
        prompt.process_new_task(1, 2)
        with torch.no_grad():
            prompt.prompt_key.zero_()
            prompt.prompt_key[:, 0] = -1
            prompt.prompt_key[10, 0] = 1
            prompt.prompt_key[0, :2] = torch.tensor([1., 1.])
            prompt.scaler[0, 1:] = 1.001
            prompt.scaler[10, 1:] = .999
            prompt.shifter.zero_()
        query = torch.zeros(3, 12)
        query[:, 0] = 1
        output = prompt(torch.zeros(3, 5, 12), torch.full((3,), 10), task_id=1, layer_num=0, cls_features=query)
        prefix = output['batched_prompt']
        torch.testing.assert_close(prefix[:, 0, 1], prefix[:, 0, 0] * 1.001, rtol=0, atol=0)
        output['reduce_sim2'].backward()
        assert prompt.prompt_key.grad[0].abs().sum() > 0
        assert prompt.prompt_key.grad[10].abs().sum() > 0
        assert not any(parameter.requires_grad for parameter in prompt.k_comp_gen.parameters())


def test_source_inference_clamps_all_affine_rows_including_unseen_rows(official):
    for cls in (ProlPrompt, official.LPrompt):
        prompt = tiny_prompt(cls)
        prompt.process_new_task(0, 1)
        prompt.eval()
        with torch.no_grad():
            prompt.scaler.fill_(2)
            prompt.scaler_v.fill_(0)
            prompt.shifter.fill_(1)
            prompt.shifter_v.fill_(-1)
            prompt(torch.zeros(3, 5, 12), None, layer_num=0, cls_features=torch.randn(3, 12))
        assert torch.all(prompt.scaler == 1.001)
        assert torch.all(prompt.scaler_v == .999)
        assert torch.all(prompt.shifter == .0001)
        assert torch.all(prompt.shifter_v == -.0001)


def test_actual_author_two_task_engine_matches_native_model_gradients_steps_and_freezes(official):
    torch.manual_seed(1993)
    native = {'query': tiny_vit(False), 'prompt': tiny_vit()}
    official.apply_author_freeze(author_args(official), native['query'], native['prompt'])
    torch.manual_seed(1993)
    reference = build_author_branches(official, num_classes=20, embed_dim=12, depth=2,
        num_heads=3, img_size=32, prompt_layers=(0, 1), n_tasks=2)
    args = author_args(official, num_tasks=2, lr=.0025, loss_thres=100., lrcos_tmax=3)
    mask = [list(range(10)), list(range(10, 20))]
    images = [torch.randn(3, 3, 32, 32) for _ in range(2)]
    before = tensor_state(native['prompt'])
    for task in range(2):
        optimizers, metrics, gradients, rng = [], [], [], []
        batches = [(x, torch.tensor([0, 2, 7]) + 10 * task) for x in images]
        for branches in (native, reference):
            torch.manual_seed(70 + task)
            optimizer, _ = begin_author_task(official, branches, args, task, mask, task)
            optimizers.append(optimizer)
            captured = []

            def capture_step(opt, positional, keywords):
                captured.append({name: parameter.grad.detach().clone() if parameter.grad is not None else None
                                 for name, parameter in branches['prompt'].named_parameters()})

            handle = optimizer.register_step_pre_hook(capture_step)
            metrics.append(run_author_epoch(official, branches, batches, optimizer, args,
                                           task_id=task, class_mask=mask, old_num_k=task))
            handle.remove()
            gradients.append(captured)
            rng.append(torch.get_rng_state())
        assert metrics[0] == metrics[1]
        assert torch.equal(*rng)
        assert [group['lr'] for group in optimizers[0].param_groups] == [group['lr'] for group in optimizers[1].param_groups]
        assert len(gradients[0]) == len(gradients[1]) == 2
        for left, right in zip(*gradients):
            assert left.keys() == right.keys()
            for name in left:
                assert (left[name] is None) == (right[name] is None), name
                if left[name] is not None:
                    assert torch.equal(left[name], right[name]), name
        assert_tensor_state_equal(native['query'], reference['query'])
        assert_tensor_state_equal(native['prompt'], reference['prompt'])
        assert_gradients_equal(native['prompt'], reference['prompt'])
        assert native['prompt'].e_prompt.lmax_list == reference['prompt'].e_prompt.lmax_list
    prompt = native['prompt']
    assert len(prompt.head.heads) == 2
    assert not any(parameter.requires_grad for parameter in prompt.head.heads[0].parameters())
    assert not any(parameter.requires_grad for parameter in prompt.e_prompt.k_comp_gen.parameters())
    assert all(parameter.requires_grad for parameter in prompt.blocks[0].norm1.parameters())
    assert not torch.equal(prompt.blocks[0].norm1.weight, before['blocks.0.norm1.weight'])
    assert torch.equal(prompt.blocks[0].attn.qkv.weight, before['blocks.0.attn.qkv.weight'])
    assert prompt.e_prompt.prompt_key.grad[:10].abs().sum() > 0


def test_actual_engine_non_diagonal_is_zero_and_ce_trigger_precedes_pull_loss(official):
    branches = build_author_branches(official, num_classes=20, embed_dim=12, depth=2,
        num_heads=3, img_size=32, prompt_layers=(0, 1), n_tasks=2)
    args = author_args(official, num_tasks=2, lr=.0025, loss_thres=3., pull_constraint_coeff=100.)
    mask = [list(range(10)), list(range(10, 20))]
    optimizer, _ = begin_author_task(official, branches, args, 0, mask, 0)
    images, labels = torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])
    with torch.no_grad():
        branches['prompt'].e_prompt.prompt_key[:3].copy_(-branches['query'](images)['pre_logits'])
    captured = {}

    def trace(frame, event, arg):
        if frame.f_code is official.train_one_epoch_with_aux.__code__ and event == 'return':
            for key in ('NonD', 'Lrdn', 'GM', 'Ldiag', 'reduce_lr', 'loss', 'lr_scheduler'):
                value = frame.f_locals[key]
                captured[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
        return trace

    original_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        run_author_epoch(official, branches, [(images, labels)],
                         optimizer, args, class_mask=mask)
    finally:
        sys.settrace(original_trace)
    assert torch.count_nonzero(captured['GM'] - torch.diag_embed(torch.diagonal(captured['GM']))) > 0
    assert torch.count_nonzero(captured['NonD']) == 0
    assert captured['Lrdn'] == 0
    assert captured['Ldiag'] > 0
    assert captured['loss'] > args.loss_thres
    assert captured['reduce_lr'] is True
    assert captured['lr_scheduler'].last_epoch == 1


def test_lr_scaling_is_source_batch_over_256_even_when_unscale_flag_true(official):
    assert author_args(official, scale_lr=True, lr=.01, batch_size=64, unscale_lr=True).lr == .0025
    assert author_args(official, scale_lr=True, lr=.05, batch_size=64, unscale_lr=True).lr == .0125
    assert author_args(official, scale_lr=True, lr=.01, batch_size=64, world_size=2, unscale_lr=False).lr == .005


def native_method_for_oracle(nominal_lr, threshold):
    args = NativeProlMethod.get_parser(ArgumentParser()).parse_args([])
    for name, value in dict(lr=nominal_lr, prol_loss_threshold=threshold, prol_cos_tmax=3,
        n_epochs=1, batch_size=64, fitting_mode='epochs', eval_epochs=None,
        label_perc=1., label_perc_by_class=1., nowand=1, device='cpu', optim_mom=.9,
        optim_nesterov=False, num_workers=0, debug_mode=False, non_verbose=True, seed=1993).items():
        setattr(args, name, value)
    dataset = SimpleNamespace(N_CLASSES=20, N_TASKS=2, N_CLASSES_PER_TASK=10,
        SETTING='class-il', get_offsets=lambda task: (task * 10, (task + 1) * 10))
    network = ProlNetwork(tiny_vit(False), tiny_vit(), 2)
    network.pretrained_checkpoint = '/tmp/prol-synthetic-oracle-fixture.npz'
    subject = NativeProlMethod(network, nn.CrossEntropyLoss(), args, None, dataset)
    return subject.to(torch.device('cpu')), dataset


@pytest.mark.parametrize('nominal_lr,threshold,controlled', [(.01, .3, False), (.05, .3, False),
    (.01, 100., False), (.05, 100., False), (.01, .3, True)])
def test_actual_native_meta_training_matches_independent_author_engine(official, nominal_lr, threshold, controlled):
    """Native observe/loss/schedulers, compared with the untouched author engine."""
    torch.manual_seed(1993)
    subject, dataset = native_method_for_oracle(nominal_lr, threshold)
    initial_rng = torch.get_rng_state()
    torch.manual_seed(1993)
    reference = build_author_branches(official, num_classes=20, embed_dim=12, depth=2,
        num_heads=3, img_size=32, prompt_layers=(0, 1), n_tasks=2)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert_tensor_state_equal(subject.net.query, reference['query'])
    assert_tensor_state_equal(subject.net.prompt, reference['prompt'])
    args = author_args(official, scale_lr=True, num_tasks=2, batch_size=64,
        lr=nominal_lr, loss_thres=threshold, lrcos_tmax=3)
    mask = [list(range(10)), list(range(10, 20))]
    generator = torch.Generator().manual_seed(812)
    images = [torch.randn(3, 3, 32, 32, generator=generator) for _ in range(3)]

    class ThresholdProbeCriterion(nn.Module):
        """Keep actual CE gradients while forcing first-small, later-large CE."""
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, logits, labels):
            actual = nn.functional.cross_entropy(logits, labels)
            desired = .1 if self.calls // 2 == 0 else 1.
            self.calls += 1
            return actual - actual.detach() + desired

    for task in range(2):
        batches = [(image, torch.tensor([0, 2, 7]) + 10 * task) for image in images]
        task_rng = torch.get_rng_state()
        native_gradients, native_rates, native_losses = [], [], []
        if controlled:
            subject.loss = ThresholdProbeCriterion()
        subject.meta_begin_task(dataset)
        subject.meta_begin_epoch(0, dataset)

        def capture_native_step(opt, positional, keywords):
            native_gradients.append({name: parameter.grad.detach().clone() if parameter.grad is not None else None
                                     for name, parameter in subject.net.prompt.named_parameters()})
            native_rates.append([group['lr'] for group in opt.param_groups])

        handle = subject.opt.register_step_pre_hook(capture_native_step)
        for image, labels in batches:
            native_losses.append(subject.meta_observe(image, labels, image, epoch=0))
        handle.remove()
        native_inner_rates = [group['lr'] for group in subject.opt.param_groups]
        subject.meta_end_epoch(0, dataset)
        native_outer_rates = [group['lr'] for group in subject.opt.param_groups]
        subject.meta_end_task(dataset)
        completed_rng = torch.get_rng_state()
        torch.set_rng_state(task_rng)
        optimizer, outer_scheduler = begin_author_task(official, reference, args, task, mask, task)
        author_gradients, author_rates = [], []

        def capture_author_step(opt, positional, keywords):
            author_gradients.append({name: parameter.grad.detach().clone() if parameter.grad is not None else None
                                     for name, parameter in reference['prompt'].named_parameters()})
            author_rates.append([group['lr'] for group in opt.param_groups])

        handle = optimizer.register_step_pre_hook(capture_author_step)
        author_metrics = run_author_epoch(official, reference, batches, optimizer, args,
                                         task_id=task, class_mask=mask, old_num_k=task,
                                         criterion=ThresholdProbeCriterion() if controlled else None)
        handle.remove()
        assert sum(native_losses) / len(native_losses) == author_metrics['Loss']
        assert native_rates == author_rates
        if controlled:
            assert subject.reduce_lr is True
            assert subject.inner_scheduler.last_epoch == len(batches)
            assert native_rates[0][0] != native_rates[1][0] != native_rates[2][0]
        assert native_inner_rates == [group['lr'] for group in optimizer.param_groups]
        outer_scheduler.step()
        assert native_outer_rates == [group['lr'] for group in optimizer.param_groups]
        assert torch.equal(torch.get_rng_state(), completed_rng)
        for left, right in zip(native_gradients, author_gradients):
            assert left.keys() == right.keys()
            for name in left:
                assert (left[name] is None) == (right[name] is None), name
                if left[name] is not None:
                    assert torch.equal(left[name], right[name]), name
        assert_tensor_state_equal(subject.net.query, reference['query'])
        assert_tensor_state_equal(subject.net.prompt, reference['prompt'])
        assert_gradients_equal(subject.net.prompt, reference['prompt'])
        assert subject.net.prompt.e_prompt.lmax_list == reference['prompt'].e_prompt.lmax_list
        assert subject.net.query.training is True
        images = [image.flip(-1) for image in images]
