"""Exact low-rank accuracy gates, real stream completeness, and source pins."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import low_rank_protocol as protocol


def rows(correct=900):
    return [{'task': 1, 'seen_correct': correct, 'seen_samples': 1000, 'old_correct': 0,
             'old_samples': 0, 'new_correct': correct, 'new_samples': 1000},
            {'task': 2, 'seen_correct': 2 * correct, 'seen_samples': 2000, 'old_correct': correct,
             'old_samples': 1000, 'new_correct': correct, 'new_samples': 1000}]


def checks():
    return {name: True for name in protocol.REQUIRED_STRUCTURAL_CHECKS}


@pytest.mark.parametrize('method,epochs', [('ewc-lora', 20), ('sd-lora', 20), ('inf-lora', 20), ('cl-lora', 30)])
def test_exact_first_two_protocol_and_native_command(method, epochs):
    settings = protocol.protocol(method)
    assert settings['tasks'] == 2 and settings['n_epochs'] == epochs
    assert settings['batch_size'] == 64 and settings['seed'] == 1993 and settings['num_workers'] == 4
    assert settings['class_order'] == list(range(100))
    assert settings['dataset_total_tasks'] == 10 and settings['rank'] == 10
    assert settings['train_transforms'][-1] == 'Normalize((0,0,0),(1,1,1))'
    assert settings['test_transforms'][0] == 'Resize(224,bilinear)'
    assert settings['precision'] == protocol.PRECISION
    command = protocol.native_command(method)
    assert command[command.index('--stop_after') + 1] == '2'
    assert '--num_tasks' not in command and command[command.index('--debug_mode') + 1] == '0'
    assert '--nowand' not in command  # The framework sets this internally after initialization.


@pytest.mark.parametrize('correct,status', [(885, 'pass'), (884, 'review'), (870, 'review'), (869, 'fail'), (950, 'pass')])
def test_exact_percentage_point_boundary(correct, status):
    result = protocol.evaluate_acceptance(rows(correct), rows(), structural_checks=checks())
    assert result['status'] == status
    assert set(result['metrics']) == {'task1_seen', 'task2_seen', 'task2_old', 'task2_new'}


def test_old_class_failure_is_not_hidden_by_seen_average():
    actual = rows()
    actual[1].update(old_correct=869, seen_correct=1769)
    result = protocol.evaluate_acceptance(actual, rows(), structural_checks=checks())
    assert result['metrics']['task2_seen']['status'] == 'review'
    assert result['metrics']['task2_old']['status'] == 'fail'
    assert result['status'] == 'fail'


@pytest.mark.parametrize('gate', protocol.REQUIRED_STRUCTURAL_CHECKS)
def test_each_structural_gate_blocks_numerical_pass(gate):
    current = checks()
    current.pop(gate)
    report = protocol.evaluate_acceptance(rows(), rows(), structural_checks=current)
    assert report['status'] == 'fail' and gate in report['structural_failures']


def test_incomplete_predictions_cannot_pass(tmp_path):
    destination = tmp_path / 'predictions.npz'
    labels = np.repeat(np.arange(10), 100)
    np.savez(destination, labels=labels, predictions=labels, global_ids=np.arange(1000))
    assert protocol.validate_prediction_archive(destination, 1)['seen_correct'] == 1000
    for corruption in ('missing', 'duplicate_id', 'wrong_class_counts', 'future_prediction', 'wrong_order'):
        ids, actual, predictions = np.arange(1000), labels.copy(), labels.copy()
        if corruption == 'missing':
            ids, actual, predictions = ids[:-1], actual[:-1], predictions[:-1]
        elif corruption == 'duplicate_id':
            ids[-1] = ids[-2]
        elif corruption == 'wrong_class_counts':
            actual[-1] = 0
        elif corruption == 'future_prediction':
            predictions[-1] = 10
        elif corruption == 'wrong_order':
            ids = ids[::-1]
        np.savez(destination, labels=actual, predictions=predictions, global_ids=ids)
        with pytest.raises(ValueError):
            protocol.validate_prediction_archive(destination, 1)


def completed(method):
    epochs = protocol.protocol(method)['n_epochs']
    return {'epoch_results': [{'task': task, 'epoch': epoch, 'steps': 79, 'samples': 5000}
                             for task in (1, 2) for epoch in range(1, epochs + 1)],
            'counts': {'observe_calls': 2 * epochs * 79,
                       'optimizer_updates': (3 if method == 'cl-lora' else 2) * epochs * 79,
                       'task_begin_extra_samples': 10000 if method == 'inf-lora' else 0,
                       'task_end_extra_samples': {'ewc-lora': 10000, 'sd-lora': 0, 'inf-lora': 10000, 'cl-lora': 15000}[method]},
            'task_results': rows(), 'first_batch_audits': [{}] * (3 if method == 'cl-lora' else 2)}


@pytest.mark.parametrize('method', protocol.METHODS)
def test_counts_observe_separately_from_actual_optimizer_updates(method):
    record = completed(method)
    assert protocol.completion_checks(record, method)
    record['counts']['optimizer_updates'] -= 1
    with pytest.raises(RuntimeError, match='optimizer updates'):
        protocol.completion_checks(record, method)


@pytest.mark.parametrize('method', protocol.METHODS)
def test_statistics_and_prototypes_require_actual_full_source_passes(method):
    record = completed(method)
    record['counts']['task_end_extra_samples'] += 1
    with pytest.raises(RuntimeError, match='task statistics or prototype'):
        protocol.completion_checks(record, method)


@pytest.mark.parametrize('corruption', ['dropped_tail', 'missing_epoch', 'duplicate_epoch', 'wrong_task'])
def test_stream_gate_rejects_partial_or_repeated_training(corruption):
    record = completed('ewc-lora')
    if corruption == 'dropped_tail':
        record['epoch_results'][-1]['samples'] = 4992
    elif corruption == 'missing_epoch':
        record['epoch_results'].pop()
    elif corruption == 'duplicate_epoch':
        record['epoch_results'][-1] = record['epoch_results'][0].copy()
    else:
        record['epoch_results'][-1]['task'] = 3
    with pytest.raises(RuntimeError, match='full approved'):
        protocol.completion_checks(record, 'ewc-lora')


def test_phase_rng_never_expands_to_future_tasks():
    values = [protocol.phase_seed(1993, task, phase, epoch) for task in (0, 1)
              for phase in protocol.PHASES for epoch in (0, 1, 2)]
    assert len(set(values)) == len(values)
    assert protocol.phase_seed(1993, 1, 'train', 2) == 102995
    with pytest.raises(ValueError):
        protocol.phase_seed(1993, 2, 'train')


def test_shared_epoch_loader_ignores_unrelated_constructor_rng():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.arange(30)), shuffle=True, batch_size=7)
    protocol.configure_epoch_loader(loader, 1993, 0, 'train')
    first = torch.cat([batch[0] for batch in loader])
    torch.rand(200)
    protocol.configure_epoch_loader(loader, 1993, 0, 'train')
    assert torch.equal(first, torch.cat([batch[0] for batch in loader]))
    loader = DataLoader(TensorDataset(torch.arange(30)), shuffle=True, batch_size=7, drop_last=True)
    with pytest.raises(ValueError, match='no dropped'):
        protocol.configure_epoch_loader(loader, 1993, 0, 'train')


def test_repeated_prototype_reads_advance_one_phase_generator():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    source = TensorDataset(torch.arange(100))
    loader = DataLoader(source, shuffle=True, batch_size=17)
    registry = {}
    assert protocol.configure_phase_loader_once(loader, 1993, 1, 'task_end', registry)
    first = torch.cat([batch[0] for batch in loader])
    assert not protocol.configure_phase_loader_once(loader, 1993, 1, 'task_end', registry)
    second = torch.cat([batch[0] for batch in loader])
    assert not torch.equal(first, second)
    matched = DataLoader(source, shuffle=True, batch_size=17)
    protocol.configure_epoch_loader(matched, 1993, 1, 'task_end')
    assert torch.equal(first, torch.cat([batch[0] for batch in matched]))
    assert torch.equal(second, torch.cat([batch[0] for batch in matched]))


def test_algorithm_state_gate_rejects_missing_fisher_or_dualgpm():
    import torch
    network = SimpleNamespace(_cur_task=0, image_encoder=SimpleNamespace(embed_dim=768), state_dict=lambda: {'x': torch.ones(1)})
    with pytest.raises(RuntimeError, match='Fisher'):
        protocol.audit_method_state(network, SimpleNamespace(omega_W=[], count_updates=1), 'ewc-lora', 1)
    with pytest.raises(RuntimeError, match='DualGPM'):
        protocol.audit_method_state(network, SimpleNamespace(feature_list=[], project_type=[]), 'inf-lora', 1)


def test_plan_protects_all_author_method_files():
    plan = protocol.load_plan()
    assert plan['protocols'] == {method: protocol.protocol(method) for method in protocol.METHODS}
    for method in protocol.METHODS.values():
        assert all(name in plan['reference_source_sha256'] for name in
                   (f'methods/{method}.py', f'models/net_{method}.py', f'models/vit_{method}.py', f'configs/{method}/cifar100.json'))
    assert plan['full_ten_task_training_authorized'] is False
