"""Shared-input EASE execution gates, without authorized GPU experiments."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import ease_fullcl_protocol as protocol
from scripts import run_ease_fullcl as runner


def completed_fixture(mode):
    settings = protocol.protocol(mode)
    tasks = settings['tasks']
    rows = [{'task': task, 'epoch': epoch, 'steps': 79, 'samples': 5000}
            for task in range(1, tasks + 1) for epoch in range(1, 21)]
    record = {'epoch_results': rows, 'counts': {'prototype_examples': 5000 * tasks,
              'prototype_adapter_image_pairs': 5000 * tasks * (tasks + 1) // 2,
              'evaluation_adapter_image_pairs': 1000 * sum(task ** 2 for task in range(1, tasks + 1))},
              'per_task_resources': [{'task': task} for task in range(1, tasks + 1)]}
    recorder = SimpleNamespace(tasks_completed=tasks, steps=1580 * tasks, training_examples=100000 * tasks,
                               evaluation_examples=1000 * tasks * (tasks + 1) // 2)
    model = SimpleNamespace(net=SimpleNamespace(backbone=SimpleNamespace(num_adapters=tasks),
                            classifier=SimpleNamespace(weight=SimpleNamespace(shape=(10 * tasks, 768 * tasks))), task_finalized=True))
    return record, recorder, model, settings


@pytest.mark.parametrize('mode', ['validation', 'full_cl'])
def test_recipe_retains_ease_schedule_and_hide_actual_input(mode):
    settings = protocol.protocol(mode)
    assert settings['n_epochs'] == 20 and settings['batch_size'] == 64
    assert settings['adapter_size'] == 64 and settings['vit_size'] == 'base'
    assert settings['dataset_total_tasks'] == 10 and settings['classes_per_task'] == 10
    assert settings['dataset_config'] == 'hide_prompt' and settings['equal_compute_budget'] is False
    command = protocol.native_command(mode)
    assert command[command.index('--stop_after') + 1] == str(settings['tasks'])
    assert command[command.index('--batch_size') + 1] == '64'
    assert command[command.index('--n_epochs') + 1] == '20'
    assert '--nowand' not in command
    assert '--num_tasks' not in command


@pytest.mark.parametrize('mode', ['validation', 'full_cl'])
def test_completion_requires_all_outer_and_expanded_forward_work(mode):
    record, recorder, model, settings = completed_fixture(mode)
    assert runner.completion_checks(record, recorder, model, settings)
    assert recorder.steps == (3160 if mode == 'validation' else 15800)
    assert recorder.training_examples == (200000 if mode == 'validation' else 1000000)
    assert recorder.evaluation_examples == (3000 if mode == 'validation' else 55000)
    record['epoch_results'][-1]['samples'] = 4999
    assert not runner.completion_checks(record, recorder, model, settings)


@pytest.mark.parametrize('corruption', ['duplicate_epoch', 'short_epoch', 'observe_steps', 'train_examples',
                                     'evaluation_examples', 'tasks', 'prototype_examples', 'prototype_pairs',
                                     'evaluation_pairs', 'adapter_count', 'shape', 'finalized', 'resources'])
def test_full_completion_fails_closed_for_each_missing_work_category(corruption):
    record, recorder, model, settings = completed_fixture('full_cl')
    if corruption == 'duplicate_epoch': record['epoch_results'][-1] = deepcopy(record['epoch_results'][0])
    if corruption == 'short_epoch': record['epoch_results'][-1]['steps'] -= 1
    if corruption == 'observe_steps': recorder.steps -= 1
    if corruption == 'train_examples': recorder.training_examples -= 1
    if corruption == 'evaluation_examples': recorder.evaluation_examples -= 1
    if corruption == 'tasks': recorder.tasks_completed -= 1
    if corruption == 'prototype_examples': record['counts']['prototype_examples'] -= 1
    if corruption == 'prototype_pairs': record['counts']['prototype_adapter_image_pairs'] -= 1
    if corruption == 'evaluation_pairs': record['counts']['evaluation_adapter_image_pairs'] -= 1
    if corruption == 'adapter_count': model.net.backbone.num_adapters -= 1
    if corruption == 'shape': model.net.classifier.weight.shape = (100, 768)
    if corruption == 'finalized': model.net.task_finalized = False
    if corruption == 'resources': record['per_task_resources'].pop()
    assert not runner.completion_checks(record, recorder, model, settings)


def test_dry_cli_remains_stdlib_only_and_no_credentials():
    command = ('import sys; from scripts.run_ease_fullcl import main; main(["--mode", "full_cl"]); '
               'assert "torch" not in sys.modules; assert "wandb" not in sys.modules')
    result = subprocess.run([sys.executable, '-S', '-c', command], cwd=runner.ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '"experiment_executed": false' in result.stdout and '"tasks": 10' in result.stdout


def test_worker_cannot_run_without_authorized_parent(monkeypatch, tmp_path):
    monkeypatch.delenv('CL_EASE_FULLCL_EXECUTE', raising=False)
    monkeypatch.setattr(runner, 'Recorder', lambda *args: pytest.fail('Must not construct observer'))
    with pytest.raises(ValueError, match='explicit --execute'):
        runner.worker(tmp_path / 'missing.json')


def test_full_execution_requires_accepted_validation_before_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'configure_environment', lambda: pytest.fail('Must not read credentials'))
    options = SimpleNamespace(mode='full_cl', acceptance_summary=None, initial_state=None,
                              reference_summary=None, result_dir=tmp_path)
    with pytest.raises(ValueError, match='acceptance-summary'):
        runner.build_spec(options)


def test_phase_rng_matches_hide_tii_training_and_resets_all_coordinates():
    from scripts.hide_protocol import phase_seed as hide_seed
    for task in range(10):
        for epoch in range(20):
            assert protocol.phase_seed(1993, task, 'train', epoch) == hide_seed(1993, task, 'tii_train', epoch)
    assert len({protocol.phase_seed(1993, task, phase, epoch) for task in range(10)
                for phase in protocol.PHASES for epoch in range(20)}) == 800
    with pytest.raises(ValueError): protocol.phase_seed(1993, -1, 'train', 0)
    with pytest.raises(ValueError): protocol.phase_seed(1993, 0, 'unknown', 0)
    with pytest.raises(ValueError): protocol.phase_seed(1993, 0, 'train', 1000)


def test_epoch_loader_reseeds_sampler_and_requires_fresh_nondropped_workers():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.arange(9)), batch_size=4, shuffle=True, num_workers=0)
    protocol.configure_epoch_loader(loader, 1993, 0, 'train', 0)
    first = torch.cat([batch[0] for batch in loader])
    protocol.configure_epoch_loader(loader, 1993, 0, 'train', 0)
    assert torch.equal(first, torch.cat([batch[0] for batch in loader]))
    assert loader.generator is loader.sampler.generator
    assert len(first) == 9 and loader.worker_init_fn is protocol.worker_seed
    with pytest.raises(ValueError, match='no dropped tail'):
        protocol.configure_epoch_loader(DataLoader(TensorDataset(torch.arange(9)), batch_size=4, drop_last=True), 1993, 0, 'train')


@pytest.mark.parametrize('key', ['predictions', 'labels', 'global_ids', 'evaluation_global_ids'])
def test_prefix_predicate_checks_all_saved_arrays(tmp_path, key):
    import numpy as np
    values = {'predictions': np.arange(15).reshape(3, 5), 'labels': np.arange(3),
              'global_ids': np.arange(3), 'evaluation_global_ids': np.arange(3)}
    actual, accepted = tmp_path / 'actual.npz', tmp_path / 'accepted.npz'
    np.savez(actual, **values)
    np.savez(accepted, **values)
    assert runner.compare_prefix_predictions(actual, accepted)
    values[key].flat[0] += 1
    np.savez(actual, **values)
    with pytest.raises(RuntimeError, match='prefix predictions differ'):
        runner.compare_prefix_predictions(actual, accepted)


def test_initializer_transfers_all_tensor_values_and_rejects_bad_recipe():
    import torch
    network = SimpleNamespace(backbone=SimpleNamespace(adapters=torch.nn.ModuleList([torch.nn.Linear(3, 2)])),
                              proxy_head=torch.nn.Linear(3, 2))
    state = {'adapter': {key: value.clone().fill_(.25) for key, value in network.backbone.adapters[0].state_dict().items()},
             'proxy_head': {key: value.clone().fill_(.5) for key, value in network.proxy_head.state_dict().items()}}
    catalog = {'schema_version': 1, 'tasks': {str(task): deepcopy(state) for task in range(10)}}
    assert protocol.load_native_initialization(network, catalog, 0)
    assert torch.equal(network.proxy_head.weight, torch.full_like(network.proxy_head.weight, .5))
    catalog['tasks']['0']['adapter']['weight'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finiteness'):
        protocol.load_native_initialization(network, catalog, 0)


def test_native_dataset_override_uses_hide_transform_without_schedule_leak(monkeypatch):
    import main as mammoth
    from datasets import get_dataset_class
    args = mammoth.parse_args(cmd=protocol.native_command('validation'))
    klass = get_dataset_class(args)
    fields = {name: getattr(klass, name) for name in ('TRANSFORM', 'TEST_TRANSFORM', 'N_TASKS', 'N_CLASSES_PER_TASK', 'N_CLASSES', 'SIZE', 'MEAN', 'STD')}
    for name, value in fields.items(): monkeypatch.setattr(klass, name, value)
    args = protocol.prepare_native_args(args)
    assert args.dataset_config == 'hide_prompt' and args.n_epochs == 20 and args.batch_size == 64
    assert args.pretrained is False
    assert 'Normalize' in repr(klass.TRANSFORM) and 'ColorJitter' in repr(klass.TRANSFORM)
    assert '(0.08, 1.0)' in repr(klass.TRANSFORM)
    assert 'Normalize' in repr(klass.TEST_TRANSFORM)


def test_reference_gate_rejects_mismatched_precision_before_tensor_imports(tmp_path):
    record = {'status': 'completed', 'model': 'ease', 'mode': 'reference',
              'protocol': protocol.protocol('validation'), 'reference_commit': protocol.REFERENCE_COMMIT,
              'precision': {**protocol.PRECISION, 'tf32_allowed': True},
              'wandb_logging_status': 'uploaded', 'wandb_artifact': {'state': 'COMMITTED'}}
    with pytest.raises(ValueError, match='matching shared-input author reference'):
        protocol.validate_reference(record)


def test_source_manifest_protects_method_trainer_input_and_measurement():
    sources = set(protocol.SOURCE_FILES)
    assert {'models/ease.py', 'backbone/ease_vit.py', 'utils/training.py', 'main.py',
            'scripts/cl_efficiency.py', 'scripts/run_hide_validation.py',
            'datasets/configs/seq-cifar100-224/hide_prompt.yaml'} <= sources
    assert len(sources) == len(protocol.SOURCE_FILES)


def accepted_fixture(tmp_path, monkeypatch):
    import numpy as np
    from scripts.ease_acceptance import counts_from_predictions, evaluate_acceptance
    from scripts.run_ease_validation import reference_policy
    initial, checkpoint = tmp_path / 'initial.pt', tmp_path / 'weights.npz'
    initial.write_bytes(b'catalog')
    checkpoint.write_bytes(b'pretrained')
    reference_dir, native_dir = tmp_path / 'reference', tmp_path / 'native'
    reference_dir.mkdir(); native_dir.mkdir()
    original_hash = protocol.file_hash
    monkeypatch.setattr(protocol, 'file_hash', lambda path: protocol.CHECKPOINT_SHA256
                        if Path(path) == checkpoint else original_hash(path))
    rows, native_pins = [], {}
    for task in (1, 2):
        labels = np.repeat(np.arange(task * 10), 100)
        data = {'predictions': np.stack([labels] * 5, axis=1), 'labels': labels,
                'global_ids': np.arange(task * 1000), 'evaluation_global_ids': np.arange(task * 1000)}
        name = f'task{task}_predictions.npz'
        np.savez(reference_dir / name, **data); np.savez(native_dir / name, **data)
        native_pins[name] = {'local_file': name, 'sha256': original_hash(native_dir / name)}
        counts = counts_from_predictions(labels, labels, past_classes=(task - 1) * 10, seen_classes=task * 10)
        rows.append({'task': task, **counts, 'seen_accuracy_pct': 100.,
                     'old_accuracy_pct': None if task == 1 else 100., 'new_accuracy_pct': 100.})
    reference = {'status': 'completed', 'model': 'ease', 'mode': 'reference', 'protocol': protocol.protocol('validation'),
                 'reference_commit': protocol.REFERENCE_COMMIT, 'precision': protocol.PRECISION,
                 'wandb_logging_status': 'uploaded', 'wandb_artifact': {'state': 'COMMITTED'},
                 'checkpoint_file': str(checkpoint), 'checkpoint_sha256': protocol.CHECKPOINT_SHA256,
                 'initial_state': str(initial), 'initial_state_sha256': original_hash(initial),
                 'initial_canonical_pretrained_sha256': protocol.CANONICAL_PRETRAINED_SHA256,
                 'structural_checks': {'training_protocol_complete': True}, 'output': str(reference_dir),
                 'task_results': rows, 'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/referencefixture'}
    reference_path = reference_dir / 'result.json'
    reference_path.write_text(json.dumps(reference))
    policy, prediction_pins = reference_policy(reference, {'unit': 'percentage_points', 'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.})
    checks = {key: True for key in protocol.STRUCTURAL_CHECKS}
    accepted = {'status': 'completed', 'model': 'ease', 'mode': 'validation',
                'protocol': protocol.protocol('validation'), 'precision': protocol.PRECISION,
                'acceptance_status': 'pass', 'acceptance': evaluate_acceptance(rows, policy, structural_checks=checks),
                'wandb_logging_status': 'uploaded', 'wandb_artifact': {'state': 'COMMITTED'},
                'execution_source_sha256': protocol.source_hashes(), 'structural_checks': checks,
                'reference_summary': str(reference_path), 'reference_summary_sha256': original_hash(reference_path),
                'reference_predictions_sha256': prediction_pins,
                'initial_state': str(initial), 'initial_state_sha256': original_hash(initial),
                'task_results': rows, 'wandb_artifact_files': native_pins, 'output': str(native_dir)}
    path = native_dir / 'result.json'
    path.write_text(json.dumps(accepted))
    return accepted, path, initial


def test_full_gate_recomputes_exact_accepted_counts_and_source_provenance(tmp_path, monkeypatch):
    record, path, initial = accepted_fixture(tmp_path, monkeypatch)
    assert protocol.validate_full_cl_acceptance(path, initial)['task_results'] == record['task_results']


@pytest.mark.parametrize('source', protocol.SOURCE_FILES)
def test_full_gate_rejects_each_changed_execution_source(tmp_path, monkeypatch, source):
    record, path, initial = accepted_fixture(tmp_path, monkeypatch)
    record['execution_source_sha256'][source] = 'changed'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='execution sources changed'):
        protocol.validate_full_cl_acceptance(path, initial)


@pytest.mark.parametrize('corruption', ['mode', 'precision', 'missing_check', 'false_check', 'upload',
                                     'artifact', 'init', 'reference', 'reference_prediction',
                                     'native_prediction', 'accuracy', 'acceptance', 'review'])
def test_full_gate_rejects_each_missing_or_corrupted_prerequisite(tmp_path, monkeypatch, corruption):
    record, path, initial = accepted_fixture(tmp_path, monkeypatch)
    if corruption == 'mode': record['mode'] = 'efficiency'
    if corruption == 'precision': record['precision'] = {**record['precision'], 'torch_cpu_threads': 4}
    if corruption == 'missing_check': record['structural_checks'].pop('checkpoint_round_trip')
    if corruption == 'false_check': record['structural_checks']['old_adapters_unchanged'] = False
    if corruption == 'upload': record['wandb_logging_status'] = 'failed'
    if corruption == 'artifact': record['wandb_artifact']['state'] = 'PENDING'
    if corruption == 'init': initial.write_bytes(b'changed')
    if corruption == 'reference': Path(record['reference_summary']).write_text('{}')
    if corruption == 'reference_prediction': (Path(record['reference_summary']).parent / 'task2_predictions.npz').write_bytes(b'changed')
    if corruption == 'native_prediction': (Path(record['output']) / 'task2_predictions.npz').write_bytes(b'changed')
    if corruption == 'accuracy': record['task_results'][0]['seen_correct'] -= 1
    if corruption == 'acceptance': record['acceptance']['status'] = 'fail'
    if corruption == 'review': record['acceptance_status'] = 'review'
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, RuntimeError)):
        protocol.validate_full_cl_acceptance(path, initial)
