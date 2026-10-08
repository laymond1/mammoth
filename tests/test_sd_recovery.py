"""A separate SD recovery recipe cannot alter or bypass the accepted protocol."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_sd_recovery as runner


def spec(pass_kind='validation'):
    return {'method': 'sd-lora', 'model': 'sd-lora', 'dataset': 'seq-cifar100-224',
        'mode': 'validation' if pass_kind == 'validation' else 'efficiency', 'tasks': 1, 'pass': pass_kind,
        'protocol': runner.recovery_protocol(pass_kind), 'recovery_recipe': runner.RECIPE,
        'implementation': 'mammoth_native', 'gpu': '1', 'wandb_mode': 'online',
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
        'checkpoint_file': runner.base.shared.CHECKPOINT,
        'checkpoint_sha256': runner.base.CHECKPOINT_SHA256,
        'canonical_pretrained_sha256': runner.base.CANONICAL_PRETRAINED_SHA256,
        'reference_commit': runner.base.REFERENCE_COMMIT, 'author_reference_sha256': runner.AUTHOR_RESULT_SHA256,
        'author_reference': str(runner.AUTHOR_REFERENCE), 'data_root': '/shared/data',
        'measurement_aggregate_eligible': pass_kind == 'measurement'}


@pytest.mark.parametrize('pass_kind', runner.PASSES)
def test_all_recovery_passes_keep_the_same_native_workload_and_separate_validation(pass_kind):
    current = spec(pass_kind)
    runner.validate_spec(current)
    protocol = current['protocol']
    assert protocol['hyperparameters'] == {'optimizer': 'adam', 'lr': .001, 'fc_lr': .008, 'weight_decay': 0.}
    assert protocol['train_transforms'][0] == 'RandomResizedCrop(224,scale=(0.05,1.0),bilinear)'
    assert (protocol['tasks'], protocol['n_epochs'], protocol['batch_size'], protocol['num_workers'],
            protocol['rank'], protocol['dataset_total_tasks'], protocol['seed']) == (1, 1, 64, 4, 10, 10, 1993)
    assert protocol['efficiency_comparison'] is (pass_kind != 'validation')
    assert current['measurement_aggregate_eligible'] is (pass_kind == 'measurement')
    command = runner.native_command(current)
    for flag, expected in (('--lr', '0.001'), ('--lora_fc_lr', '0.008'), ('--stop_after', '1'),
                           ('--n_epochs', '1'), ('--batch_size', '64'), ('--dataset_config', 'ease')):
        assert command[command.index(flag) + 1] == expected
    assert '--num_tasks' not in command and '--loadcheck' not in command


def test_recovery_does_not_change_original_recipe_or_helpers():
    before = runner.base.efficiency_protocol('sd-lora')
    command_before = runner.base.native_command(spec('measurement'))
    with runner.worker_overrides():
        assert runner.base.native_command(spec('measurement'))[command_before.index('--lr') + 1] == '0.001'
        assert runner.base.EfficiencyAudit is runner.RecoveryAudit
    assert runner.base.efficiency_protocol('sd-lora') == before
    assert before['hyperparameters']['lr'] == .008
    assert runner.base.native_command(spec('measurement')) == command_before
    assert runner.base.EfficiencyAudit is runner._BASE_AUDIT


def test_process_local_adaptations_restore_helpers_on_failure():
    original = runner.base.validate_spec
    with pytest.raises(RuntimeError):
        with runner.worker_overrides():
            raise RuntimeError('diagnostic failure')
    assert runner.base.validate_spec is original


@pytest.mark.parametrize('key,value', [
    ('method', 'ewc-lora'), ('model', 'inf-lora'), ('dataset', 'seq-cifar10-224'), ('tasks', 2), ('tasks', True),
    ('mode', 'efficiency'), ('pass', 'warmup'), ('gpu', '0'), ('wandb_mode', 'offline'),
    ('wandb_entity', 'other'), ('wandb_project', 'other'), ('checkpoint_sha256', 'changed'),
    ('canonical_pretrained_sha256', 'changed'), ('reference_commit', 'changed'), ('implementation', 'author'),
    ('recovery_recipe', 'default'), ('author_reference_sha256', 'changed'), ('measurement_aggregate_eligible', True),
    ('measurement_aggregate_eligible', 0),
])
def test_changed_scope_cannot_launch_recovery(key, value):
    current = spec()
    current[key] = value
    with pytest.raises(ValueError):
        runner.validate_spec(current)


@pytest.mark.parametrize('key,value', [
    ('batch_size', 128), ('num_workers', 0), ('seed', 0), ('tasks', 10), ('n_epochs', 20), ('n_epochs', True),
    ('rank', 4), ('vit_size', 'small'), ('dataset_config', 'default'), ('dataset_total_tasks', 1),
    ('encoder_and_scale_lr', .008), ('validation_runs', 2), ('measurement_repeats', True), ('flops_runs', 2),
])
def test_protocol_changes_cannot_hide_inside_recovery_spec(key, value):
    current = spec()
    current['protocol'][key] = value
    with pytest.raises(ValueError):
        runner.validate_spec(current)


@pytest.mark.parametrize('key,value', [('lr', .008), ('fc_lr', .001), ('optimizer', 'sgd'), ('weight_decay', .1)])
def test_only_encoder_and_scale_lr_can_change_from_the_original_recipe(key, value):
    current = spec()
    current['protocol']['hyperparameters'][key] = value
    with pytest.raises(ValueError):
        runner.validate_spec(current)


def test_dry_preparation_never_loads_credentials_or_starts_execution(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry preparation attempted execution')
    monkeypatch.setattr(runner, 'configure_environment', forbidden)
    monkeypatch.setattr(runner, 'execute', forbidden)
    monkeypatch.setattr(runner, 'author_reference_pins', forbidden)
    assert runner.main([]) == 0
    assert json.loads(capsys.readouterr().out)['execution'] is False


def test_recovery_worker_requires_its_own_explicit_execution_flag(monkeypatch, tmp_path):
    monkeypatch.delenv('CL_SD_RECOVERY_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit parent'):
        runner.worker(tmp_path / 'unused.json')


def test_recovery_worker_routes_to_original_native_worker_with_scoped_overrides(monkeypatch, tmp_path):
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec('measurement')))
    monkeypatch.setenv('CL_SD_RECOVERY_EXECUTE', '1')
    monkeypatch.setattr(runner, 'verify_reference_pins', lambda spec: None)
    original_validation = runner.base.validate_spec
    def fake_worker(spec_path):
        assert spec_path == path
        assert runner.base.validate_spec is runner.validate_spec
        assert runner.base.native_command is runner.native_command
        assert runner.base.execution_sources is runner.execution_sources
        assert runner.base.EfficiencyAudit is runner.RecoveryAudit
        return 0
    monkeypatch.setattr(runner.base, 'worker', fake_worker)
    assert runner.worker(path) == 0
    assert runner.base.validate_spec is original_validation


def test_saved_author_evidence_is_pinned_including_source_and_predictions():
    if not runner.AUTHOR_REFERENCE.is_file() or not runner.AUTHOR_DRIVER.is_file():
        pytest.skip('Optional local author evidence is not present in this checkout')
    evidence = json.loads(runner.AUTHOR_REFERENCE.read_text())
    if any(not (Path(evidence['reference_source']) / relative).is_file()
           for relative in evidence['reference_source_sha256']):
        pytest.skip('Optional immutable author source checkout is not present')
    pins = runner.author_reference_pins()
    assert pins[str(runner.AUTHOR_REFERENCE)] == runner.AUTHOR_RESULT_SHA256
    assert pins[str(runner.AUTHOR_DRIVER)] == runner.AUTHOR_DRIVER_SHA256
    assert str(runner.AUTHOR_REFERENCE.parent / 'task1_predictions.npz') in pins
    assert len([path for path in pins if path.startswith('/tmp/cl-low-rank-official-20261008/')]) == 25
    current = spec()
    current['recovery_reference_pins'] = pins
    runner.verify_reference_pins(current)


def test_changed_saved_author_result_is_rejected_before_training(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='approved evidence'):
        runner.author_reference_pins(path)


def test_reference_pin_set_cannot_drop_predictions(monkeypatch):
    current = spec()
    pins = {'/reference/result.json': runner.AUTHOR_RESULT_SHA256, '/reference/predictions.npz': 'a' * 64}
    current['author_reference'] = '/reference/result.json'
    current['recovery_reference_pins'] = {current['author_reference']: runner.AUTHOR_RESULT_SHA256}
    monkeypatch.setattr(runner.base.shared, 'file_hash', lambda path: pins[str(path)])
    monkeypatch.setattr(runner, 'author_reference_pins', lambda path: pins)
    with pytest.raises(ValueError, match='incomplete'):
        runner.verify_reference_pins(current)


def test_build_spec_preserves_old_acceptance_provenance_and_adds_separate_recovery(monkeypatch, tmp_path):
    options = SimpleNamespace(method='sd-lora', pass_kind='validation', author_reference=Path('/reference/result.json'))
    previous_protocol = runner.base.efficiency_protocol
    def old_builder(current_options, profile):
        assert runner.base.efficiency_protocol('sd-lora') == runner.recovery_protocol('validation')
        current = spec('validation')
        current.update(acceptance_summary='/old-two-task/result.json', acceptance_summary_sha256='a' * 64,
                       accepted_validation_wandb_url='https://wandb.ai/old-validation')
        return current
    monkeypatch.setattr(runner, '_BASE_BUILD_SPEC', old_builder)
    monkeypatch.setattr(runner, 'author_reference_pins', lambda path: {'/reference/result.json': runner.AUTHOR_RESULT_SHA256})
    current = runner.build_spec(options, {})
    assert current['mode'] == 'validation' and current['measurement_aggregate_eligible'] is False
    assert current['acceptance_summary'] == '/old-two-task/result.json'
    assert current['author_reference'] == '/reference/result.json'
    assert runner.base.efficiency_protocol is previous_protocol


def record_fixture(tmp_path):
    """Small synthetic evidence keeps core unit tests independent of raw runs."""
    import numpy as np
    labels = np.repeat(np.arange(10, dtype=np.int64), 100)
    predicted = labels.copy()
    predicted[:10] = 1
    predictions = {'predictions': predicted, 'labels': labels, 'global_ids': np.arange(1000, dtype=np.int64)}
    reference_path = tmp_path / 'reference.json'
    np.savez_compressed(tmp_path / 'task1_predictions.npz', **predictions)
    fingerprints = {}
    for split, samples, digest in [('task1_train', 5000, 'd'), ('task1_test', 1000, 'e')]:
        fingerprints[split] = {'n_samples': samples,
            'class_counts': {str(label): samples // 10 for label in range(10)},
            'samples_ordered_sha256': digest * 64, 'samples_unordered_sha256': digest * 64,
            'global_ids_ordered_sha256': digest * 64, 'transform_repr_sha256': 'f' * 64}
    source = {'initial_state_sha256': 'a' * 64, 'initial_learned_sha256': 'b' * 64,
        'counts': {'observe_calls': 79, 'optimizer_updates': 79, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
        'epoch_results': [{'task': 1, 'epoch': 1, 'steps': 79, 'samples': 5000, 'loss_mean': .63}],
        'task_results': [{'task': 1, 'seen_correct': 990, 'seen_samples': 1000, 'old_correct': 0,
                         'old_samples': 0, 'new_correct': 990, 'new_samples': 1000}],
        'first_batch_audits': [{'task': 1, 'epoch': 1, 'seed': 2993, 'samples': 64, 'optimizer_update_index': 1,
            'input_sha256': 'a' * 64, 'labels_sha256': 'b' * 64, 'gradient_sha256': 'c' * 64}],
        'accuracy': {'mean_accuracy_pct': 99.}, 'common_reference_data_snapshots': deepcopy(fingerprints),
        'common_transform_snapshots': deepcopy(fingerprints), 'data_snapshots': deepcopy(fingerprints)}
    reference_path.write_text(json.dumps(source))
    path = tmp_path / 'predictions.npz'
    np.savez_compressed(path, **predictions)
    record = spec()
    record.update(author_reference=str(reference_path), status='completed', computation_status='completed', wandb_logging_status='uploaded',
        precision=runner.base.PRECISION, structural_checks={key: True for key in runner.base.REQUIRED_CHECKS},
        initial_state_sha256=source['initial_state_sha256'], initial_learned_sha256=source['initial_learned_sha256'],
        initial_canonical_pretrained_sha256=runner.base.CANONICAL_PRETRAINED_SHA256,
        counts=source['counts'], epoch_results=source['epoch_results'], task_results=source['task_results'],
        first_batch_audits=source['first_batch_audits'], accuracy=source['accuracy'],
        prediction_files=[{'task': 1, 'path': str(path), 'sha256': runner.base.shared.file_hash(path)}],
        metrics={'outer_observe_steps': 79, 'outer_training_stream_examples': 5000, 'evaluation_examples': 1000, 'tasks_completed': 1},
        common_reference_data_snapshots=source['common_reference_data_snapshots'],
        common_transform_snapshots=source['common_transform_snapshots'], data_snapshots=source['data_snapshots'],
        optimizer_group_audits=[{'task': 1, 'epoch': 1, 'optimizer_update_index': 1,
            'role': role, 'lr': lr, 'parameters': count}
            for role, lr, count in [('encoder_and_scale', .001, 368641), ('classifier', .008, 7680)]])
    return deepcopy(record)


def test_complete_record_validation_recomputes_accuracy_from_saved_predictions(tmp_path):
    record = record_fixture(tmp_path)
    assert runner.validate_recovery_record(record, 'validation') is record


@pytest.mark.parametrize('corruption', [
    'status', 'upload', 'precision', 'structural_check', 'initial', 'incomplete_stream', 'boolean_steps',
    'missing_updates', 'missing_prediction', 'changed_prediction_hash', 'reported_accuracy', 'first_gradient',
    'first_seed', 'group_lr', 'missing_group', 'boolean_group_count', 'different_transform', 'future_samples',
])
def test_unfinished_or_unfaithful_record_is_rejected(corruption, tmp_path):
    record = record_fixture(tmp_path)
    if corruption == 'status': record['status'] = 'failed'
    elif corruption == 'upload': record['wandb_logging_status'] = 'pending'
    elif corruption == 'precision': record['precision']['autocast'] = True
    elif corruption == 'structural_check': record['structural_checks']['frozen_backbone_unchanged'] = False
    elif corruption == 'initial': record['initial_learned_sha256'] = 'changed'
    elif corruption == 'incomplete_stream': record['metrics']['outer_training_stream_examples'] = 4992
    elif corruption == 'boolean_steps': record['metrics']['outer_observe_steps'] = True
    elif corruption == 'missing_updates': record['counts']['optimizer_updates'] = 78
    elif corruption == 'missing_prediction': record['prediction_files'] = []
    elif corruption == 'changed_prediction_hash': record['prediction_files'][0]['sha256'] = 'changed'
    elif corruption == 'reported_accuracy': record['accuracy']['mean_accuracy_pct'] = 10.
    elif corruption == 'first_gradient': record['first_batch_audits'][0]['gradient_sha256'] = 'changed'
    elif corruption == 'first_seed': record['first_batch_audits'][0]['seed'] = 1993
    elif corruption == 'group_lr': record['optimizer_group_audits'][0]['lr'] = .008
    elif corruption == 'missing_group': record['optimizer_group_audits'].pop()
    elif corruption == 'boolean_group_count': record['optimizer_group_audits'][0]['parameters'] = True
    elif corruption == 'different_transform': record['data_snapshots']['task1_train']['transform_repr_sha256'] = 'changed'
    elif corruption == 'future_samples': record['task_results'][0]['seen_samples'] = 2000
    with pytest.raises((ValueError, RuntimeError)):
        runner.validate_recovery_record(record, 'validation')


def test_actual_first_step_optimizer_groups_must_match_parameter_roles():
    import torch
    encoder = torch.nn.Linear(2, 2)
    classifier = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam([{'params': list(encoder.parameters()), 'lr': .001},
                                {'params': list(classifier.parameters()), 'lr': .008}])
    audit = object.__new__(runner.RecoveryAudit)
    from weakref import WeakSet
    audit.hooked, audit.handles = WeakSet(), []
    audit.stage, audit.epoch = 'train', {'steps': 0, 'seed': 2993}
    audit.model = SimpleNamespace(net=SimpleNamespace(image_encoder=encoder, classifier_pool=classifier))
    audit.record = {'first_batch_audits': [], 'counts': {'optimizer_updates': 0}}
    audit.inputs, audit.labels = torch.ones(1, 2), torch.zeros(1, dtype=torch.long)
    audit.finite = lambda: None
    # The base first-gradient hook expects named_parameters on a real network.
    audit.model.net.named_parameters = lambda: [('encoder.' + n, p) for n, p in encoder.named_parameters()] + list(classifier.named_parameters())
    try:
        audit.attach_optimizer(optimizer)
        optimizer.step()
        assert [group['lr'] for group in audit.record['optimizer_group_audits']] == [.001, .008]
        assert [group['role'] for group in audit.record['optimizer_group_audits']] == ['encoder_and_scale', 'classifier']
        optimizer.param_groups[0]['lr'] = .008
        with pytest.raises(ValueError, match='actual LoRA/scale'):
            optimizer.step()
    finally:
        for handle in audit.handles:
            handle.remove()
