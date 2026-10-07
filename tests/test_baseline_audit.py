"""Baseline pilot audits with synthetic CPU tensors; no image inference."""

import copy
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import SequentialSampler
from timm.models.vision_transformer import VisionTransformer

from scripts import baseline_audit as module
from scripts.cl_efficiency import Recorder
from scripts.ease_reference_utils import dataset_fingerprint


def reference_vit():
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=12, depth=12,
                              num_heads=3, num_classes=100)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class StateOnly:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


def test_digest_matches_original_ease_byte_order_and_detects_changes():
    state = OrderedDict(z=torch.arange(6, dtype=torch.float32).reshape(2, 3),
                        a=torch.tensor([1, 2], dtype=torch.int64))
    expected = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        expected.update(name.encode())
        expected.update(str(tuple(tensor.shape)).encode())
        expected.update(str(tensor.dtype).encode())
        expected.update(tensor.numpy().tobytes())
    assert module.digest(state) == expected.hexdigest()
    assert module.digest(dict(reversed(list(state.items())))) == expected.hexdigest()
    changed = {**state, 'z': state['z'].clone()}
    changed['z'][0, 0] += 1
    assert module.digest(changed) != expected.hexdigest()
    assert module.digest({'z': state['z'].double(), 'a': state['a']}) != expected.hexdigest()


def test_l2p_prefix_normalization_preserves_all_150_pretrained_tensors():
    reference = reference_vit().state_dict()
    raw = dict(reference)
    raw['pos_embed'] = torch.cat((raw['pos_embed'][:, :1].repeat(1, 6, 1), raw['pos_embed'][:, 1:]), dim=1)
    raw['prompt.prompt'] = torch.ones(10, 5, 12)
    raw['prompt.prompt_key'] = torch.ones(10, 12)
    canonical = module.canonical_state(StateOnly(raw), 'l2p')
    expected = {name: value for name, value in reference.items() if not name.startswith('head.')}
    assert len(canonical) == 150
    assert set(canonical) == set(expected)
    assert canonical['pos_embed'].shape == (1, 197, 12)
    assert all(torch.equal(canonical[name], value) for name, value in expected.items())
    assert module.digest(canonical) == module.digest(expected)


def split_ranpac_state(state):
    raw = {}
    for name, value in state.items():
        if '.attn.qkv.' in name:
            for part, chunk in zip(('q', 'k', 'v'), value.chunk(3, dim=0)):
                raw[name.replace('.attn.qkv.', f'.attn.{part}_proj.')] = chunk.clone()
        else:
            raw[name.replace('.mlp.fc1.', '.fc1.').replace('.mlp.fc2.', '.fc2.')] = value
    for index in range(12):
        raw[f'blocks.{index}.adaptmlp.up_proj.weight'] = torch.zeros(12, 2)
    return raw


def test_ranpac_split_qkv_and_mlp_normalization_reconstructs_150_tensors():
    reference = reference_vit().state_dict()
    canonical = module.canonical_state(StateOnly(split_ranpac_state(reference)), 'ranpac')
    expected = {name: value for name, value in reference.items() if not name.startswith('head.')}
    assert len(canonical) == 150 and set(canonical) == set(expected)
    assert all(torch.equal(canonical[name], value) for name, value in expected.items())
    assert module.digest(canonical) == module.digest(expected)


@pytest.mark.parametrize('change', ['missing', 'extra'])
def test_incomplete_or_unexpected_backbone_tensors_are_rejected(change):
    raw = dict(reference_vit().state_dict())
    if change == 'missing':
        raw.pop('blocks.11.mlp.fc2.bias')
    else:
        raw['unrecognized_backbone'] = torch.ones(1)
    with pytest.raises(RuntimeError, match='150 canonical'):
        module.canonical_state(StateOnly(raw), 'dualprompt')


class HookModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.NAME = method
        self.native_calls = []
        self.loss = 1.25

    def meta_observe(self, inputs, **kwargs):
        self.native_calls.append(('observe', inputs, kwargs))
        return self.loss

    def meta_end_epoch(self, *args, **kwargs):
        self.native_calls.append(('end_epoch', args, kwargs))
        return 'native_epoch'

    def meta_end_task(self, *args, **kwargs):
        self.native_calls.append(('end_task', args, kwargs))
        return 'native_task'

    def forward(self, logits):
        self.native_calls.append(('forward', logits))
        return logits


def split_ranpac_module(model):
    for block in model.blocks:
        original = block.attn.qkv
        del block.attn.qkv
        for part, weight, bias in zip(('q', 'k', 'v'), original.weight.chunk(3, dim=0), original.bias.chunk(3, dim=0)):
            projection = nn.Linear(12, 12)
            with torch.no_grad():
                projection.weight.copy_(weight)
                projection.bias.copy_(bias)
            projection.requires_grad_(False)
            setattr(block.attn, f'{part}_proj', projection)
        block.fc1, block.fc2 = block.mlp.fc1, block.mlp.fc2
        del block.mlp
        block.adaptmlp = nn.Linear(12, 2)


@pytest.fixture
def audit_factory(tmp_path, monkeypatch):
    def create(method='dualprompt'):
        query = reference_vit()
        expected = module.digest(module.canonical_state(query, 'dualprompt'))
        monkeypatch.setattr(module, 'PTM_SHA', expected)
        model = HookModel(method)
        if method == 'ranpac':
            split_ranpac_module(query)
            network = nn.Module()
            network.convnet = query
            network.fc = nn.Linear(4, 10)
            network.fc.use_RP = True
            model.net = nn.Module()
            model.net._network = network
            for name, shape in {'G': (4, 4), 'Q': (4, 100), 'W_rand': (768, 4)}.items():
                model.register_buffer(name, torch.zeros(shape))
            network.fc.register_buffer('W_rand', model.W_rand)
        else:
            prompted = copy.deepcopy(query)
            prompted.register_parameter('g_prompt', nn.Parameter(torch.ones(2)))
            prompted.head.requires_grad_(True)
            model.net = nn.Module()
            model.net.original_model, model.net.model = query, prompted
        train = SimpleNamespace(data=np.zeros((5000, 1, 1, 3), dtype=np.uint8),
                                targets=np.arange(5000) % 10, indexes=np.arange(5000), transform='synthetic static transform')
        test = SimpleNamespace(data=np.zeros((1000, 1, 1, 3), dtype=np.uint8),
                               targets=np.arange(1000) % 10, indexes=np.arange(1000), transform='synthetic static transform')
        dataset = SimpleNamespace(train_loader=SimpleNamespace(dataset=train),
                                  test_loaders=[SimpleNamespace(dataset=test, sampler=SequentialSampler(range(1000)), drop_last=False)])
        args = SimpleNamespace(n_epochs=1, batch_size=64, num_workers=4, seed=1993,
                               dataset_config='ease', code_optimization=0, drop_last=False, fitting_mode='epochs',
                               permute_classes=False, joint=False, eval_epochs=None, loadcheck=None, start_from=None,
                               pretrained_checkpoint=str(tmp_path / 'synthetic.npz'), rp_size=4)
        model.args = args
        metadata = {'checkpoint_file': args.pretrained_checkpoint, 'minimum_accuracy_pct': 20.,
                    'accuracy': {'mean_accuracy_pct': 100.},
                    'reference_data_snapshots': {f'task1_{split}': dataset_fingerprint(source.data, source.targets, source.indexes)
                                                for split, source in (('train', train), ('test', test))}}
        recorder = Recorder(tmp_path, gpu='cpu')
        recorder.data_snapshots['first_task'] = {'train': {}, 'test': {}}
        audit = module.BaselineAudit(model, dataset, args, recorder, metadata, torch)
        return audit
    return create


def finalize(audit):
    audit.losses = [1.25] * 79
    audit.end_epoch()
    audit.prototype_examples = 5000 if audit.method == 'ranpac' else 0
    audit.end_task()
    audit.predictions = [audit.dataset.test_loaders[0].dataset.targets.copy()]
    audit.recorder.steps, audit.recorder.training_examples = 79, 5000
    audit.recorder.evaluation_examples, audit.recorder.tasks_completed = 1000, 1


@pytest.mark.parametrize('method', module.MODELS)
def test_three_methods_require_complete_structural_and_prediction_evidence(audit_factory, method):
    audit = audit_factory(method)
    finalize(audit)
    audit.complete()
    assert set(audit.checks) == set(module.CHECKS) and all(audit.checks.values())
    assert audit.metadata['initial_frozen_backbone_sha256'] == audit.metadata['final_frozen_backbone_sha256']
    assert all(value == module.PTM_SHA for value in audit.metadata['canonical_pretrained_sha256'].values())
    with np.load(audit.recorder.output / 'task1_predictions.npz') as saved:
        assert saved['predictions'].shape == (1000, 1)
        assert np.array_equal(saved['predictions'][:, 0], saved['labels'])
        assert np.array_equal(saved['global_ids'], np.arange(1000))
    if method == 'ranpac':
        assert audit.metadata['task_end_training_image_presentations'] == 5000
        assert audit.metadata['ridge_candidate_count'] == 17
        assert audit.metadata['ridge_solve_count'] == 18
        storage = audit.metadata['method_state_storage']
        assert storage['cpu_gram_G_bytes'] == 4 * 4 * 4
        assert storage['cpu_cross_product_Q_bytes'] == 4 * 100 * 4
        assert storage['projection_shared_storage'] is True


@pytest.mark.parametrize('field,value', [('n_epochs', 2), ('batch_size', 48), ('num_workers', 0), ('seed', 1994),
                                       ('dataset_config', 'l2p'), ('code_optimization', 1), ('drop_last', True),
                                       ('fitting_mode', 'iters'), ('permute_classes', True), ('joint', True),
                                       ('eval_epochs', 1), ('loadcheck', '/tmp/other.pt'), ('start_from', 1),
                                       ('pretrained_checkpoint', '/tmp/other.npz')])
def test_effective_argument_changes_rejected_before_completion(audit_factory, field, value):
    audit = audit_factory()
    setattr(audit.model.args, field, value)
    with pytest.raises(ValueError, match='shared pilot protocol'):
        module.BaselineAudit(audit.model, audit.dataset, audit.model.args, audit.recorder, audit.metadata, torch)


@pytest.mark.parametrize('change', ['canonical_value', 'unfrozen', 'non_fp32'])
def test_initial_backbone_checkpoint_or_freeze_violation_rejected(audit_factory, change):
    audit = audit_factory()
    backbone = audit.model.net.original_model
    if change == 'canonical_value':
        backbone.pos_embed.data[0, 0, 0] += 1
    elif change == 'unfrozen':
        backbone.cls_token.requires_grad_(True)
    else:
        backbone.cls_token.data = backbone.cls_token.data.double()
    with pytest.raises(RuntimeError, match='pretrained tensors|frozen FP32'):
        module.BaselineAudit(audit.model, audit.dataset, audit.model.args, audit.recorder, audit.metadata, torch)


@pytest.mark.parametrize('change', ['frozen_gradient', 'adaptation_parameter', 'adaptation_gradient'])
def test_epoch_rejects_frozen_gradients_and_nonfinite_adaptation(audit_factory, change):
    audit = audit_factory()
    if change == 'frozen_gradient':
        parameter = audit.model.net.original_model.cls_token
        parameter.grad = torch.zeros_like(parameter)
    else:
        parameter = audit.model.net.model.g_prompt
        if change == 'adaptation_parameter':
            parameter.data[0] = float('nan')
        else:
            parameter.grad = torch.full_like(parameter, float('inf'))
    with pytest.raises(RuntimeError, match='received gradients|non-finite'):
        audit.end_epoch()
    assert audit.epochs == 0


@pytest.mark.parametrize('change', ['frozen_weight', 'raw_image', 'raw_label', 'raw_id', 'head_shape', 'query_unfrozen'])
def test_task_end_rejects_frozen_changes_and_data_or_prompt_contract_corruption(audit_factory, change):
    audit = audit_factory()
    if change == 'frozen_weight':
        audit.model.net.model.pos_embed.data[0, 0, 0] += 1
    elif change == 'raw_image':
        audit.dataset.train_loader.dataset.data[0, 0, 0, 0] = 1
    elif change == 'raw_label':
        audit.dataset.train_loader.dataset.targets[0] = 1
    elif change == 'raw_id':
        audit.dataset.test_loaders[0].dataset.indexes[[0, 1]] = [1, 0]
    elif change == 'head_shape':
        audit.model.net.model.head = nn.Linear(12, 99)
    else:
        audit.model.net.original_model.head.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match='backbone changed|raw samples|freeze contract'):
        audit.end_task()
    assert not audit.checks['method_specific_final_state']


@pytest.mark.parametrize('field,shape', [('G', (3, 4)), ('Q', (4, 99)), ('W_rand', (767, 4))])
def test_ranpac_task_end_rejects_incomplete_ridge_shapes(audit_factory, field, shape):
    audit = audit_factory('ranpac')
    setattr(audit.model, field, torch.zeros(shape))
    audit.prototype_examples = 5000
    with pytest.raises(RuntimeError, match='RP/ridge state'):
        audit.end_task()


@pytest.mark.parametrize('change', ['nan_ridge', 'nan_classifier', 'rp_disabled', 'classifier_shape', 'incomplete_features'])
def test_ranpac_task_end_requires_finite_complete_final_classifier(audit_factory, change):
    audit = audit_factory('ranpac')
    audit.prototype_examples = 5000
    if change == 'nan_ridge':
        audit.model.G[0, 0] = float('nan')
    elif change == 'nan_classifier':
        audit.model.net._network.fc.weight.data[0, 0] = float('nan')
    elif change == 'rp_disabled':
        audit.model.net._network.fc.use_RP = False
    elif change == 'classifier_shape':
        audit.model.net._network.fc.weight = nn.Parameter(torch.ones(9, 4))
    else:
        audit.prototype_examples = 4992
    with pytest.raises(RuntimeError, match='RP/ridge state|classifier completion'):
        audit.end_task()


@pytest.mark.parametrize('change', ['gram_dtype', 'projection_dtype', 'projection_unshared'])
def test_ranpac_ridge_dtype_and_registered_projection_sharing_are_required(audit_factory, change):
    audit = audit_factory('ranpac')
    audit.prototype_examples = 5000
    if change == 'gram_dtype':
        audit.model.G = audit.model.G.double()
    elif change == 'projection_dtype':
        audit.model.net._network.fc.W_rand = audit.model.W_rand.double()
    else:
        audit.model.net._network.fc.W_rand = audit.model.W_rand.clone()
    with pytest.raises(RuntimeError, match='dtype/device changed|registered shared projection'):
        audit.end_task()


@pytest.mark.parametrize('field,value', [('steps', 78), ('training_examples', 4992), ('evaluation_examples', 999), ('tasks_completed', 0)])
def test_incomplete_outer_native_protocol_cannot_pass(audit_factory, field, value):
    audit = audit_factory()
    finalize(audit)
    setattr(audit.recorder, field, value)
    with pytest.raises(RuntimeError, match='full first-task pilot'):
        audit.complete()
    assert not audit.checks['training_protocol_complete']


@pytest.mark.parametrize('change', ['epochs', 'loss_count', 'method_specific_check'])
def test_incomplete_audit_lifecycle_cannot_pass(audit_factory, change):
    audit = audit_factory()
    finalize(audit)
    if change == 'epochs':
        audit.epochs = 0
    elif change == 'loss_count':
        audit.losses.pop()
    else:
        audit.checks['method_specific_final_state'] = False
    with pytest.raises(RuntimeError, match='full first-task pilot'):
        audit.complete()


@pytest.mark.parametrize('change', ['missing_predictions', 'native_accuracy', 'random_sampler', 'drop_last'])
def test_prediction_evidence_cannot_disagree_with_native_evaluation(audit_factory, change):
    audit = audit_factory()
    finalize(audit)
    if change == 'missing_predictions':
        audit.predictions[0] = audit.predictions[0][:-1]
    elif change == 'native_accuracy':
        audit.metadata['accuracy']['mean_accuracy_pct'] = 99.9
    elif change == 'random_sampler':
        audit.dataset.test_loaders[0].sampler = object()
    else:
        audit.dataset.test_loaders[0].drop_last = True
    with pytest.raises(RuntimeError, match='incomplete|disagree|sequential'):
        audit.complete()


@pytest.mark.parametrize('correct', [100, 200])
def test_gross_accuracy_collapse_fails_even_when_native_accuracy_matches(audit_factory, correct):
    audit = audit_factory()
    finalize(audit)
    labels = audit.dataset.test_loaders[0].dataset.targets
    predicted = (labels + 1) % 10
    predicted[:correct] = labels[:correct]
    audit.predictions = [predicted]
    audit.metadata['accuracy']['mean_accuracy_pct'] = correct / 10
    with pytest.raises(RuntimeError, match='accuracy collapsed'):
        audit.complete()


def test_wrappers_chain_existing_meta_calls_capture_predictions_and_restore_on_exception(audit_factory):
    audit = audit_factory()
    originals = {name: getattr(audit.model, name) for name in ('meta_observe', 'meta_end_epoch', 'meta_end_task', 'forward')}
    audit.end_epoch, audit.end_task = Mock(), Mock()
    with pytest.raises(ValueError, match='synthetic interruption'):
        with module.instrument_baseline(audit):
            with audit.recorder.phase('train_epoch'):
                assert audit.model.meta_observe([1, 2], epoch=0) == 1.25
            with audit.recorder.phase('evaluation'):
                logits = torch.tensor([[2., 1.], [0., 1.]])
                returned = audit.model(logits)
                returned.fill_(float('-inf'))
            assert audit.model.meta_end_epoch(0) == 'native_epoch'
            assert audit.model.meta_end_task(audit.dataset) == 'native_task'
            raise ValueError('synthetic interruption')
    assert audit.losses == [1.25]
    assert audit.predictions[0].tolist() == [0, 1]
    assert [call[0] for call in audit.model.native_calls] == ['observe', 'forward', 'end_epoch', 'end_task']
    audit.end_epoch.assert_called_once_with()
    audit.end_task.assert_called_once_with()
    assert all(getattr(audit.model, name) == original for name, original in originals.items())


@pytest.mark.parametrize('loss', [float('nan'), {'loss': float('inf')}])
def test_nonfinite_native_loss_fails_during_observe(audit_factory, loss):
    audit = audit_factory()
    audit.model.loss = loss
    with module.instrument_baseline(audit), audit.recorder.phase('train_epoch'):
        with pytest.raises(RuntimeError, match='loss is non-finite'):
            audit.model.meta_observe([0])
    assert audit.losses == []


def test_nonfinite_native_evaluation_logits_fail_without_saved_predictions(audit_factory):
    audit = audit_factory()
    with module.instrument_baseline(audit), audit.recorder.phase('evaluation'):
        with pytest.raises(RuntimeError, match='logits are non-finite'):
            audit.model(torch.tensor([[float('nan'), 1.]]))
    assert audit.predictions == []


def test_ranpac_feature_hook_counts_only_existing_task_end_calls_and_is_removed(audit_factory):
    audit = audit_factory('ranpac')
    backbone = audit.model.net._network.convnet
    # Replace image forward with an identity over synthetic feature records.
    # This checks callback counting and never runs the ViT image pathway.
    backbone.forward = Mock(side_effect=lambda value: value)
    with module.instrument_baseline(audit):
        assert len(backbone._forward_hooks) == 1
        with audit.recorder.phase('train_epoch'):
            backbone([1, 2])
        with audit.recorder.phase('end_task'):
            backbone([1, 2, 3])
    assert audit.prototype_examples == 3
    assert len(backbone._forward_hooks) == 0
    assert backbone.forward.call_count == 2


@pytest.fixture
def pin_files(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    plan_dir = tmp_path / 'benchmarks/cl_efficiency/baselines'
    plan_dir.mkdir(parents=True)
    ease_dir = tmp_path / 'benchmarks/cl_efficiency/ease'
    ease_dir.mkdir(parents=True)
    checkpoint = tmp_path / 'shared.npz'
    checkpoint.write_bytes(b'synthetic checkpoint metadata only; never loaded')
    plan = {'training_authorized': True, 'models': list(module.MODELS),
            'efficiency': {'tasks': 1, 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4, 'seed': 1993,
                           'measurement_repeats': 3, 'flops_runs': 1},
            'checkpoint': {'file': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'tag': 'explicit-test-tag'}}
    (plan_dir / 'validation_plan.json').write_text(json.dumps(plan))
    (ease_dir / 'validation_plan.json').write_text(json.dumps({'reference': {
        'checkpoint_file': str(checkpoint), 'checkpoint_sha256': plan['checkpoint']['sha256']}}))
    reference = {'official_reference': {'data_snapshots': {'task1_train': {'identity': 'train'}, 'task1_test': {'identity': 'test'}}}}
    (ease_dir / 'execution_results.json').write_text(json.dumps(reference))
    sources = ['scripts/benchmark_cl.py', 'scripts/baseline_audit.py', 'scripts/run_baseline_efficiency.py',
               'scripts/cl_efficiency.py', 'scripts/cl_runtime.py', 'main.py', 'utils/training.py',
               'backbone/vit.py', 'datasets/configs/seq-cifar100-224/ease.yaml', 'uv.lock']
    sources.extend(f'models/{method}.py' for method in module.MODELS)
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# public synthetic source fingerprint\n')
    return SimpleNamespace(root=tmp_path, plan=plan, checkpoint=checkpoint, plan_file=plan_dir / 'validation_plan.json')


def test_execution_pin_records_checkpoint_sources_and_original_data_reference(pin_files):
    pinned = module.execution_pin({'model': 'dualprompt', 'gpu': '1'})
    assert pinned['checkpoint_file'] == str(pin_files.checkpoint)
    assert pinned['checkpoint_sha256'] == hashlib.sha256(pin_files.checkpoint.read_bytes()).hexdigest()
    assert pinned['reference_data_snapshots'] == {'task1_train': {'identity': 'train'}, 'task1_test': {'identity': 'test'}}
    assert pinned['execution_source_sha256']['main.py'] == hashlib.sha256((pin_files.root / 'main.py').read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['authorization', 'model_set', 'selected_model', 'gpu', 'repeat', 'epochs',
                                   'bool_tasks', 'float_epochs', 'checkpoint_bytes', 'ease_checkpoint_file', 'ease_checkpoint_hash'])
def test_execution_pin_rejects_scope_or_checkpoint_changes(pin_files, change):
    spec = {'model': 'dualprompt', 'gpu': '1'}
    if change == 'authorization':
        pin_files.plan['training_authorized'] = False
    elif change == 'model_set':
        pin_files.plan['models'].append('ease')
    elif change == 'selected_model':
        spec['model'] = 'ease'
    elif change == 'gpu':
        spec['gpu'] = '0'
    elif change == 'repeat':
        pin_files.plan['efficiency']['measurement_repeats'] = 1
    elif change == 'epochs':
        pin_files.plan['efficiency']['n_epochs'] = 20
    elif change == 'bool_tasks':
        pin_files.plan['efficiency']['tasks'] = True
    elif change == 'float_epochs':
        pin_files.plan['efficiency']['n_epochs'] = 1.0
    elif change == 'checkpoint_bytes':
        pin_files.checkpoint.write_bytes(b'changed synthetic bytes')
    else:
        ease_file = pin_files.root / 'benchmarks/cl_efficiency/ease/validation_plan.json'
        ease_plan = json.loads(ease_file.read_text())
        if change == 'ease_checkpoint_file':
            ease_plan['reference']['checkpoint_file'] = str(pin_files.root / 'different.npz')
        else:
            ease_plan['reference']['checkpoint_sha256'] = '0' * 64
        ease_file.write_text(json.dumps(ease_plan))
    pin_files.plan_file.write_text(json.dumps(pin_files.plan))
    with pytest.raises(ValueError, match='scope|checkpoint changed|exact accepted EASE'):
        module.execution_pin(spec)


def test_general_worker_without_baseline_pilot_never_installs_baseline_audits(tmp_path, monkeypatch):
    import sys
    from scripts import benchmark_cl
    forbidden_pin = Mock(side_effect=AssertionError('Generic workers must not load pilot pin'))
    monkeypatch.setattr(module, 'execution_pin', forbidden_pin)
    spec = {'output': str(tmp_path), 'gpu': 'cpu', 'sample_interval': 1., 'model': 'dualprompt',
            'dataset': 'seq-cifar100-224', 'group': 'vit', 'kind': 'prompt', 'pass': 'measurement',
            'suite': 'synthetic', 'wandb_mode': 'disabled', 'wandb_project': 'synthetic',
            'wandb_entity': None, 'main_args': [], 'data_root': str(tmp_path)}
    spec_file = tmp_path / 'spec.json'
    spec_file.write_text(json.dumps(spec))
    recorder = SimpleNamespace(start=Mock(), save=lambda metadata: metadata)
    monkeypatch.setattr(benchmark_cl, 'Recorder', lambda *args, **kwargs: recorder)
    monkeypatch.setenv('CL_EFFICIENCY_EXECUTE', '1')
    fake_main = SimpleNamespace(parse_args=Mock(side_effect=RuntimeError('synthetic stop before initialization')))
    monkeypatch.setitem(sys.modules, 'main', fake_main)
    assert benchmark_cl.worker(spec_file) == 1
    forbidden_pin.assert_not_called()
    fake_main.parse_args.assert_called_once()
    assert json.loads((tmp_path / 'metrics.json').read_text())['error_type'] == 'RuntimeError'
