"""Size efficiency gates use matched evidence and static CPU tensors only."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import benchmark_cl, ease_size_protocol, run_ease_efficiency as pilot
from scripts.cl_efficiency import Recorder


@pytest.fixture
def size_files(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_cl, 'ROOT', tmp_path)
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    names = ('scripts/benchmark_cl.py', 'scripts/cl_efficiency.py', 'main.py', 'utils/training.py',
             'models/ease.py', 'backbone/ease_vit.py', 'models/ease_utils/vit.py',
             'models/ease_utils/network.py', 'models/ease_utils/classifier.py',
             'datasets/configs/seq-cifar100-224/ease.yaml', 'models/config/ease.yaml',
             'uv.lock', 'scripts/ease_size_protocol.py',
             'scripts/run_ease_efficiency.py')
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('public source fixture')
    monkeypatch.setattr(pilot, '__file__', str(tmp_path / 'scripts/run_ease_efficiency.py'))
    plan = {'dataset': 'seq-cifar100-224', 'classes_per_task': 10,
            'efficiency': dict(dataset='seq-cifar100-224', tasks=1, classes_per_task=10,
                               n_epochs=1, batch_size=64, seed=1993, num_workers=4,
                               measurement_repeats=3, flops_runs=1, training_authorized=True)}
    plan_path = tmp_path / 'benchmarks/cl_efficiency/ease_sizes/validation_plan.json'
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan))
    variants, summaries = {}, {}
    for name, (size, dimension, heads, adapter) in ease_size_protocol.VARIANTS.items():
        checkpoint = tmp_path / f'{size}.npz'
        checkpoint.write_bytes(f'public {size} fixture; never loaded'.encode())
        variant = dict(name=name, vit_size=size, feature_dim=dimension, num_heads=heads, adapter_size=adapter,
                       checkpoint_tag=f'vit_{size}_patch16_224.augreg_in21k_ft_in1k',
                       checkpoint_file=str(checkpoint), checkpoint_sha256=benchmark_cl.file_hash(checkpoint),
                       canonical_pretrained_sha256='a' * 64)
        variants[name] = variant
        summary = {'variant': name, 'checkpoint_sha256': variant['checkpoint_sha256'],
                   'official_reference': {'status': 'completed', 'variant': name,
                                          'checkpoint_sha256': variant['checkpoint_sha256'],
                                          'initial_canonical_pretrained_sha256': 'a' * 64,
                                          'data_snapshots': {'task1_train': {'n_samples': 5000},
                                                             'task1_test': {'n_samples': 1000}}},
                   'native_validation': {'status': 'completed', 'variant': name,
                                         'checkpoint_sha256': variant['checkpoint_sha256'], 'acceptance_status': 'pass'},
                   'acceptance': {'status': 'pass'}}
        path = tmp_path / f'{name}-accepted.json'
        path.write_text(json.dumps(summary))
        summaries[name] = path

    def load_variant(name, verify_checkpoint=True):
        if name not in variants:
            raise ValueError('Unapproved EASE size variant')
        variant = copy.deepcopy(variants[name])
        if verify_checkpoint and benchmark_cl.file_hash(variant['checkpoint_file']) != variant['checkpoint_sha256']:
            raise ValueError('EASE size checkpoint changed or is missing')
        return variant

    monkeypatch.setattr(ease_size_protocol, 'load_variant', load_variant)
    monkeypatch.setattr(ease_size_protocol, 'load_plan', lambda: copy.deepcopy(plan))
    return SimpleNamespace(root=tmp_path, variants=variants, summaries=summaries, plan=plan)


@pytest.mark.parametrize('name', pilot.SIZE_VARIANTS)
def test_each_size_pin_uses_its_accepted_weights_and_dimensions(size_files, name):
    variant = size_files.variants[name]
    pin = benchmark_cl.ease_execution_pin({'ease_variant': name,
                                          'ease_acceptance_summary': str(size_files.summaries[name])})
    assert pin['ease_variant'] == name
    assert pin['feature_dim'] == variant['feature_dim']
    assert pin['adapter_size'] == variant['adapter_size']
    assert pin['checkpoint_sha256'] == variant['checkpoint_sha256']
    assert 'models/config/ease.yaml' in pin['execution_source_sha256']
    assert 'datasets/configs/seq-cifar100-224/ease.yaml' in pin['execution_source_sha256']


@pytest.mark.parametrize('mutation', ['wrapper_variant', 'wrapper_hash', 'review', 'native_variant',
                                      'native_hash', 'reference_variant', 'reference_hash',
                                      'reference_canonical', 'reference_incomplete', 'partial_samples'])
def test_wrong_size_or_unaccepted_evidence_blocks_cost(size_files, mutation):
    name = 'tiny_relative16'
    path = size_files.summaries[name]
    evidence = json.loads(path.read_text())
    if mutation == 'wrapper_variant':
        evidence['variant'] = 'base_fixed64'
    elif mutation == 'wrapper_hash':
        evidence['checkpoint_sha256'] = 'b' * 64
    elif mutation == 'review':
        evidence['native_validation']['acceptance_status'] = 'review'
    elif mutation == 'reference_canonical':
        evidence['official_reference']['initial_canonical_pretrained_sha256'] = 'b' * 64
    elif mutation == 'reference_incomplete':
        evidence['official_reference']['status'] = 'running'
    elif mutation == 'partial_samples':
        evidence['official_reference']['data_snapshots']['task1_train']['n_samples'] = 4992
    else:
        role, field = mutation.split('_')
        role = 'native_validation' if role == 'native' else 'official_reference'
        evidence[role]['variant' if field == 'variant' else 'checkpoint_sha256'] = 'wrong'
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        benchmark_cl.ease_execution_pin(name, acceptance_summary=path)


def test_acceptance_snapshot_and_checkpoint_must_remain_unchanged(size_files):
    name = 'small_relative32'
    summary = size_files.summaries[name]
    spec = dict(ease_variant=name, ease_acceptance_summary=str(summary),
                ease_acceptance_summary_sha256=benchmark_cl.file_hash(summary))
    summary.write_text(summary.read_text() + '\n')
    with pytest.raises(ValueError, match='summary changed'):
        benchmark_cl.ease_execution_pin(spec)
    spec.pop('ease_acceptance_summary_sha256')
    Path(size_files.variants[name]['checkpoint_file']).write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='checkpoint changed'):
        benchmark_cl.ease_execution_pin(spec)


def test_size_pin_needs_explicit_accepted_evidence(size_files):
    with pytest.raises(ValueError, match='acceptance summary'):
        benchmark_cl.ease_execution_pin('small_fixed64')
    size_files.plan['efficiency']['training_authorized'] = False
    with pytest.raises(ValueError, match='not been authorized'):
        benchmark_cl.ease_execution_pin('small_fixed64', size_files.summaries['small_fixed64'])


def test_variant_metadata_survives_manifest_job_spec_preparation(size_files):
    entry = {'model': 'ease', 'dataset': 'seq-cifar100-224', 'kind': 'training',
             'args': {'vit_size': 'tiny', 'adapter_size': 16}, 'ease_variant': 'tiny_relative16',
             'ease_acceptance_summary': str(size_files.summaries['tiny_relative16'])}
    args = benchmark_cl.parser().parse_args(['--models', 'ease', '--gpu', '1'])
    job = benchmark_cl.make_plan(args, [entry])[0]
    spec = {**job, 'pass': 'measurement'}
    assert spec['ease_variant'] == entry['ease_variant']
    assert spec['ease_acceptance_summary'] == entry['ease_acceptance_summary']
    assert '--ease_variant' not in job['main_args']


def test_size_dry_run_never_loads_weights_evidence_or_credentials(monkeypatch, capsys):
    monkeypatch.setattr(ease_size_protocol, 'load_variant', lambda *a, **kw: pytest.fail('Dry run must not load weights'))
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('Dry run must not load secrets'))
    monkeypatch.setattr(pilot, 'execute', lambda *a: pytest.fail('Dry run must not execute'))
    assert pilot.main(['--variant', 'small_fixed64']) == 0
    assert 'small_fixed64' in capsys.readouterr().out


def static_canonical_names():
    names = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'norm.weight', 'norm.bias']
    for index in range(12):
        names.extend(f'blocks.{index}.{suffix}' for suffix in (
            'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias', 'attn.qkv.weight', 'attn.qkv.bias',
            'attn.proj.weight', 'attn.proj.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'))
    return names


@pytest.fixture(params=pilot.SIZE_VARIANTS)
def size_audit(request, tmp_path):
    import numpy as np
    import torch
    from torch.utils.data import SequentialSampler
    from scripts.ease_reference_utils import dataset_fingerprint

    name = request.param
    size, dimension, heads, adapter = ease_size_protocol.VARIANTS[name]
    ptm = torch.nn.Module()
    ptm.register_parameter('anchor', torch.nn.Parameter(torch.ones(1), requires_grad=False))
    ptm.num_features = dimension
    ptm.blocks = [SimpleNamespace(attn=SimpleNamespace(num_heads=heads)) for _ in range(12)]
    ptm.state_dict = lambda: dict.fromkeys(static_canonical_names(), ptm.anchor)
    backbone = torch.nn.Module()
    backbone.backbone = ptm
    backbone.register_parameter('adapter', torch.nn.Parameter(torch.ones(1)))
    backbone.pretrained_loaded, backbone.feature_dim, backbone.num_adapters = True, dimension, 1
    backbone.adapter_size = adapter
    backbone.adapters = [[SimpleNamespace(down_proj=SimpleNamespace(in_features=dimension, out_features=adapter),
                                         up_proj=SimpleNamespace(in_features=adapter, out_features=dimension))
                          for _ in range(12)]]
    net = torch.nn.Module()
    net.backbone = backbone
    net.classifier = torch.nn.Module()
    net.classifier.register_buffer('weight', torch.zeros(10, dimension))
    net.register_buffer('task_finalized', torch.tensor(True))
    model = SimpleNamespace(net=net)
    train = SimpleNamespace(data=np.zeros((5000, 1, 1, 3), dtype=np.uint8), targets=np.arange(5000) % 10,
                            indexes=np.arange(5000), transform='synthetic transform')
    test = SimpleNamespace(data=np.zeros((1000, 1, 1, 3), dtype=np.uint8), targets=np.arange(1000) % 10,
                           indexes=np.arange(1000), transform='synthetic transform')
    dataset = SimpleNamespace(train_loader=SimpleNamespace(dataset=train),
                              test_loaders=[SimpleNamespace(dataset=test, sampler=SequentialSampler(range(1000)), drop_last=False)])
    args = SimpleNamespace(n_epochs=1, batch_size=64, num_workers=4, seed=1993, vit_size=size, adapter_size=adapter,
                           ease_pretrain_type='in21k_ft_in1k', dataset_config='ease', code_optimization=0, drop_last=False,
                           fitting_mode='epochs', permute_classes=False, joint=False, lr=.025, optim_mom=.9, optim_wd=.0005,
                           optimizer='sgd', lr_scheduler='cosine', scheduler_mode='epoch', eval_epochs=None,
                           start_from=None, loadcheck=None, ease_checkpoint=str(tmp_path / 'fixture.npz'))
    metadata = dict(ease_variant=name, vit_size=size, feature_dim=dimension, adapter_size=adapter, num_heads=heads,
                    checkpoint_file=args.ease_checkpoint,
                    canonical_pretrained_sha256=ease_size_protocol.canonical_pretrained_digest(ptm),
                    accuracy={'mean_accuracy_pct': 100.},
                    reference_data_snapshots={f'task1_{split}': dataset_fingerprint(source.data, source.targets, source.indexes)
                                              for split, source in (('train', train), ('test', test))})
    recorder = Recorder(tmp_path, gpu='cpu')
    recorder.data_snapshots['first_task'] = {'train': {}, 'test': {}}
    audit = benchmark_cl.EaseAudit(model, dataset, args, recorder, metadata, torch)
    return SimpleNamespace(audit=audit, args=args, torch=torch)


def test_dynamic_prototype_and_adapter_shapes_complete_all_four_variants(size_audit):
    audit = size_audit.audit
    audit.losses = [1.] * 79
    audit.end_epoch()
    audit.model.net.backbone.adapter.requires_grad_(False)
    audit.prototype_examples = 5000
    audit.end_task()
    audit.predictions = [audit.dataset.test_loaders[0].dataset.targets.copy()]
    audit.recorder.steps, audit.recorder.training_examples = 79, 5000
    audit.recorder.evaluation_examples, audit.recorder.tasks_completed = 1000, 1
    audit.complete()
    assert all(audit.checks.values())
    assert audit.metadata['canonical_pretrained_tensor_count'] == 150


@pytest.mark.parametrize('mutation', ['base_dimension', 'wrong_rank', 'canonical', 'heads'])
def test_size_audit_rejects_implicit_base_or_wrong_pretrained_recipe(size_audit, mutation):
    audit = size_audit.audit
    if mutation == 'base_dimension':
        audit.model.net.backbone.feature_dim = 768
    elif mutation == 'wrong_rank':
        size_audit.args.adapter_size += 1
    elif mutation == 'canonical':
        audit.metadata['canonical_pretrained_sha256'] = 'b' * 64
    else:
        audit.ptm.blocks[0].attn.num_heads += 1
    with pytest.raises(ValueError):
        benchmark_cl.EaseAudit(audit.model, audit.dataset, size_audit.args,
                               audit.recorder, audit.metadata, size_audit.torch)


def test_size_audit_rejects_wrong_final_prototype_or_adapter_dimensions(size_audit):
    audit = size_audit.audit
    audit.model.net.backbone.adapter.requires_grad_(False)
    audit.model.net.backbone.adapters[0][0].up_proj.in_features += 1
    with pytest.raises(RuntimeError, match='adapter shapes'):
        audit.end_task()


def test_size_efficiency_scope_rejects_other_variant_acceptance(size_files):
    accepted = json.loads(size_files.summaries['small_fixed64'].read_text())
    pilot.require_approved_scope(size_files.plan, 'small_fixed64', accepted)
    with pytest.raises(ValueError, match='same size variant'):
        pilot.require_approved_scope(size_files.plan, 'tiny_fixed64', accepted)


def test_size_execution_stops_before_credentials_when_acceptance_needs_review(size_files, monkeypatch):
    name = 'tiny_fixed64'
    path = size_files.summaries[name]
    evidence = json.loads(path.read_text())
    evidence['native_validation']['acceptance_status'] = 'review'
    path.write_text(json.dumps(evidence))
    monkeypatch.setattr(pilot, 'configure_environment', lambda: pytest.fail('Rejected evidence must not read credentials'))
    with pytest.raises(ValueError, match='two-task acceptance'):
        pilot.execute(SimpleNamespace(variant=name, acceptance_summary=path, output=size_files.root / 'outputs'))


@pytest.mark.parametrize('name', pilot.SIZE_VARIANTS)
def test_size_runner_propagates_acceptance_and_runs_exactly_three_measurements_one_diagnostic(size_files, monkeypatch, name):
    from tests.test_ease_efficiency import record

    variant = size_files.variants[name]
    manifest_path = size_files.root / 'benchmarks/cl_efficiency/manifest.json'
    manifest_path.write_text(json.dumps({'models': [{'model': 'ease', 'dataset': 'seq-cifar100-224',
                                                    'kind': 'training', 'args': {}}]}))
    profile = Mock(return_value={'gpu': '1', 'data_root': str(size_files.root / 'data'),
                                 'wandb_entity': 'test-entity', 'wandb_project': 'test-project'})
    monkeypatch.setattr(pilot, 'configure_environment', profile)
    modes = []

    def prepare(args):
        args.output.mkdir(parents=True)
        entries = json.loads(args.manifest.read_text())['models']
        return {'jobs': benchmark_cl.make_plan(args, entries)}, {}

    def saved_record(pass_name):
        saved = record(pass_name)
        saved.update({key: variant[key] for key in ('vit_size', 'feature_dim', 'num_heads', 'adapter_size',
                                                    'checkpoint_sha256', 'canonical_pretrained_sha256')},
                     ease_variant=name, initial_canonical_pretrained_sha256='a' * 64)
        saved['structural_checks']['canonical_pretrained_matches_checkpoint'] = True
        return saved

    def execute(args, plan, ready):
        assert args.execute is True and args.gpu == '1'
        job = plan['jobs'][0]
        assert job['ease_variant'] == name
        assert job['ease_acceptance_summary'] == str(size_files.summaries[name])
        assert job['ease_acceptance_summary_sha256'] == benchmark_cl.file_hash(size_files.summaries[name])
        assert job['args']['vit_size'] == variant['vit_size']
        assert job['args']['adapter_size'] == variant['adapter_size']
        assert job['args']['ease_checkpoint'] == variant['checkpoint_file']
        modes.append(args.flops)
        directory = args.output / 'cl-efficiency-fixture'
        directory.mkdir()
        result = {'model': 'ease', 'measurement': saved_record('measurement')}
        if args.flops == 'paired':
            result['flops'] = saved_record('flops')
        (directory / 'results.json').write_text(json.dumps([result]))
        return 0

    monkeypatch.setattr(benchmark_cl, 'prepare', prepare)
    monkeypatch.setattr(benchmark_cl, 'execute', execute)
    output = size_files.root / 'outputs'
    assert pilot.execute(SimpleNamespace(variant=name, acceptance_summary=size_files.summaries[name], output=output)) == 0
    assert modes == ['off', 'off', 'paired']
    execution_path, = output.glob('*/execution.json')
    execution = json.loads(execution_path.read_text())
    assert execution['status'] == 'completed' and execution['variant']['name'] == name
    assert execution['summary']['measurement_count'] == 3
    assert execution['summary']['flops_count'] == 1
    assert len(execution['measurements']) == 3
    assert execution['flops']['pass'] == 'flops'
