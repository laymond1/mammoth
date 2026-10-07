"""Execution boundaries, matched streams, and exact HiDe acceptance gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from scripts import hide_acceptance as acceptance
from scripts import hide_protocol as protocol
from scripts import run_hide_validation as runner


def reference_rows(correct=900):
    return [{'task': 1, 'seen_correct': correct, 'seen_samples': 1000, 'old_correct': 0,
             'old_samples': 0, 'new_correct': correct, 'new_samples': 1000},
            {'task': 2, 'seen_correct': 2 * correct, 'seen_samples': 2000,
             'old_correct': correct, 'old_samples': 1000, 'new_correct': correct, 'new_samples': 1000}]


def checks():
    return {name: True for name in acceptance.REQUIRED_STRUCTURAL_CHECKS}


@pytest.mark.parametrize('mode,epochs,tasks,config', [('validation', 70, 2, 'hide_prompt'),
                                                  ('smoke', 2, 2, 'hide_prompt'),
                                                  ('efficiency', 2, 1, 'ease')])
def test_exact_approved_protocol(mode, epochs, tasks, config):
    settings = protocol.protocol(mode)
    assert settings['n_epochs'] == epochs and settings['tasks'] == tasks
    assert settings['dataset_config'] == config and settings['batch_size'] == 64
    assert settings['dataset_total_tasks'] == 10 and settings['classes_per_task'] == 10
    assert settings['training_authorized'] is True
    command = protocol.native_command(mode)
    assert command[command.index('--stop_after') + 1] == str(tasks)
    assert '--nowand' not in command and '--num_tasks' not in command


def test_phase_rng_coordinates_are_distinct_and_stable():
    coordinates = [protocol.phase_seed(1993, task, phase, epoch)
                   for task in (0, 1) for phase in protocol.PHASES for epoch in (0, 1, 2)]
    assert len(set(coordinates)) == len(coordinates)
    assert protocol.phase_seed(1993, 1, 'prompt_train', 3) == 102996
    with pytest.raises(ValueError):
        protocol.phase_seed(1993, 0, 'unknown')


def test_shared_epoch_sampler_and_augmentations_ignore_constructor_rng():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.arange(30)), batch_size=7, shuffle=True)
    protocol.seed_phase(1993, 0, 'tii_train')
    protocol.configure_epoch_loader(loader, 1993, 0, 'tii_train')
    first = torch.cat([batch[0] for batch in loader])
    torch.rand(150)
    protocol.seed_phase(1993, 0, 'tii_train')
    protocol.configure_epoch_loader(loader, 1993, 0, 'tii_train')
    assert torch.equal(first, torch.cat([batch[0] for batch in loader]))
    protocol.configure_epoch_loader(loader, 1993, 0, 'prompt_train')
    assert not torch.equal(first, torch.cat([batch[0] for batch in loader]))


def test_loader_rejects_dropped_tail():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.arange(30)), batch_size=7, shuffle=True, drop_last=True)
    with pytest.raises(ValueError, match='no dropped'):
        protocol.configure_epoch_loader(loader, 1993, 0, 'tii_train')


@pytest.mark.parametrize('correct,status,drop', [(885, 'pass', 1.5), (884, 'review', 1.6),
                                               (870, 'review', 3.), (869, 'fail', 3.1), (950, 'pass', -5.)])
def test_four_counts_exact_threshold_and_improvements(correct, status, drop):
    report = acceptance.evaluate_acceptance(reference_rows(correct), acceptance.reference_policy(reference_rows()), structural_checks=checks())
    assert report['status'] == status
    assert report['metrics']['task2_seen']['drop_pp'] == drop
    assert report['structural_checks'] == checks()
    assert 'old_adapters_unchanged' not in report['structural_checks']


@pytest.mark.parametrize('gate', acceptance.REQUIRED_STRUCTURAL_CHECKS)
def test_each_required_gate_blocks_accuracy_only_pass(gate):
    gates = checks()
    gates.pop(gate)
    report = acceptance.evaluate_acceptance(reference_rows(), acceptance.reference_policy(reference_rows()), structural_checks=gates)
    assert report['status'] == 'fail' and gate in report['structural_failures']


def test_reference_full_evaluation_and_count_partitions_required():
    rows = reference_rows()
    rows[1]['seen_samples'] = 1999
    with pytest.raises(ValueError, match='every seen'):
        acceptance.reference_policy(rows)
    rows = reference_rows()
    rows[1]['old_correct'] -= 1
    with pytest.raises(ValueError, match='partition'):
        acceptance.evaluate_acceptance(reference_rows(), acceptance.reference_policy(rows), structural_checks=checks())


def test_native_preparation_never_reads_credentials_or_starts_work(monkeypatch, capsys):
    monkeypatch.setattr(runner, 'configure_environment', lambda: pytest.fail('Credentials during dry run'))
    monkeypatch.setattr(runner, 'execute', lambda *args: pytest.fail('Unapproved worker'))
    assert runner.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out


def test_worker_rejects_absent_parent_authorization(monkeypatch, tmp_path):
    monkeypatch.delenv('CL_HIDE_NATIVE_EXECUTE', raising=False)
    with pytest.raises(ValueError, match='explicit --execute'):
        runner.worker(tmp_path / 'missing.json')


def test_dry_run_is_stdlib_only():
    command = 'import sys; from scripts.run_hide_validation import main; main([]); assert "torch" not in sys.modules; assert "wandb" not in sys.modules'
    completed = subprocess.run([sys.executable, '-S', '-c', command], cwd=runner.ROOT, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('field,value', [('gpu', '0'), ('model', 'ease'), ('mode', 'unknown'),
                                      ('pass', 'flops'), ('checkpoint_sha256', 'wrong')])
def test_strict_spec_rejects_expanded_or_mismatched_settings(field, value):
    spec = {'gpu': '1', 'model': 'hide-prompt', 'mode': 'validation', 'pass': 'measurement',
            'protocol': protocol.protocol('validation'), 'checkpoint_sha256': protocol.CHECKPOINT_SHA256}
    spec[field] = value
    with pytest.raises(ValueError, match='approved GPU1'):
        runner.validate_spec(spec)


def test_effective_cli_retains_batch64_and_source_transforms(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    args = protocol.dataset_args()
    assert args.batch_size == 64 and args.n_epochs == 70
    assert args.hide_tii_epochs == 20 and args.hide_prompt_epochs == 50 and args.hide_ca_epochs == 30
    assert args.dataset_config == 'hide_prompt' and args.stop_after == 2 and args.permute_classes == 0


def completed_fixture(mode='efficiency'):
    settings = protocol.protocol(mode)
    rows = [{'task': task, 'phase': phase, 'epoch': epoch, 'steps': 79, 'samples': 5000}
            for task in range(1, settings['tasks'] + 1)
            for phase, count in (('tii_train', settings['tii_epochs']), ('prompt_train', settings['prompt_epochs']))
            for epoch in range(1, count + 1)]
    record = {'epoch_results': rows,
              'counts': {'tii_training_steps': 79 * settings['tii_epochs'] * settings['tasks'],
                         'prompt_training_steps': 79 * settings['prompt_epochs'] * settings['tasks'],
                         'tii_statistics_examples': 5000 * settings['tasks'],
                         'prompt_statistics_examples': 5000 * settings['tasks'],
                         'routing_evaluation_examples': sum(task * 1000 for task in range(1, settings['tasks'] + 1))}}
    recorder = SimpleNamespace(tasks_completed=settings['tasks'], steps=79 * settings['n_epochs'] * settings['tasks'],
        training_examples=5000 * settings['n_epochs'] * settings['tasks'],
        evaluation_examples=sum(task * 1000 for task in range(1, settings['tasks'] + 1)))
    model = SimpleNamespace(net=SimpleNamespace(task_finalized=True, active_task=settings['tasks'] - 1),
                            alignment_history=[{}] * (2 * settings['tasks']))
    return record, recorder, model, settings


@pytest.mark.parametrize('mode', ['validation', 'smoke', 'efficiency'])
def test_completion_requires_full_phase_streams(mode):
    record, recorder, model, settings = completed_fixture(mode)
    assert runner.completion_checks(record, recorder, model, settings)
    if mode == 'validation':
        assert recorder.steps == 11060 and recorder.training_examples == 700000


@pytest.mark.parametrize('corruption', ['duplicate_epoch', 'missing_sample', 'missing_tii_step', 'missing_statistics',
                                      'extra_evaluation', 'missing_alignment', 'unfinished_task'])
def test_completion_rejects_incomplete_or_double_counted_execution(corruption):
    record, recorder, model, settings = completed_fixture()
    if corruption == 'duplicate_epoch': record['epoch_results'][1] = deepcopy(record['epoch_results'][0])
    if corruption == 'missing_sample': record['epoch_results'][0]['samples'] -= 1
    if corruption == 'missing_tii_step': record['counts']['tii_training_steps'] -= 1
    if corruption == 'missing_statistics': record['counts']['prompt_statistics_examples'] -= 1
    if corruption == 'extra_evaluation': recorder.evaluation_examples += 1
    if corruption == 'missing_alignment': model.alignment_history.pop()
    if corruption == 'unfinished_task': model.net.task_finalized = False
    assert not runner.completion_checks(record, recorder, model, settings)


def test_efficiency_gate_requires_pass_before_source_or_checkpoint_reads(tmp_path):
    path = tmp_path / 'result.json'
    for status in ('review', 'fail', 'not_evaluated'):
        path.write_text(json.dumps({'status': 'completed', 'acceptance_status': status}))
        with pytest.raises(ValueError, match='passing full validation'):
            protocol.validate_efficiency_acceptance(path)


def test_create_dataset_sets_shared_base_path_before_parser_and_construction(monkeypatch, tmp_path):
    import datasets
    from utils.conf import base_path
    monkeypatch.setattr(base_path, 'path', './sentinel/', raising=False)
    validated = []
    monkeypatch.setattr(protocol, 'validate_shared_data', lambda path: validated.append(path))

    def parse(*args):
        assert base_path() == str(tmp_path.resolve()) + '/'
        return SimpleNamespace(dataset='seq-cifar100-224')

    monkeypatch.setattr(protocol, 'dataset_args', parse)
    monkeypatch.setattr(datasets, 'get_dataset', lambda args: args)
    assert protocol.create_dataset(data_root=str(tmp_path)).dataset == 'seq-cifar100-224'
    assert validated == [str(tmp_path)]


def test_shared_data_validator_blocks_missing_and_corrupt_files_before_parser(monkeypatch, tmp_path):
    from torchvision.datasets import CIFAR100
    raw = tmp_path / 'CIFAR100' / CIFAR100.base_folder
    raw.mkdir(parents=True)
    payloads = {'train': b'local_train', 'test': b'local_test', 'meta': b'local_meta'}
    monkeypatch.setattr(CIFAR100, 'train_list', [['train', hashlib.md5(payloads['train']).hexdigest()]])
    monkeypatch.setattr(CIFAR100, 'test_list', [['test', hashlib.md5(payloads['test']).hexdigest()]])
    monkeypatch.setattr(CIFAR100, 'meta', {'filename': 'meta', 'md5': hashlib.md5(payloads['meta']).hexdigest()})
    monkeypatch.setattr(protocol, 'dataset_args', lambda *args: pytest.fail('Parsing before existing raw integrity'))
    with pytest.raises(FileNotFoundError, match='downloads are disabled'):
        protocol.create_dataset(data_root=str(tmp_path))
    for name, content in payloads.items():
        (raw / name).write_bytes(content)
    result = protocol.validate_shared_data(tmp_path)
    assert result['downloads_allowed'] is False and set(result['files']) == set(payloads)
    assert result['files']['train']['sha256'] == hashlib.sha256(payloads['train']).hexdigest()
    (raw / 'meta').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity failed'):
        protocol.validate_shared_data(tmp_path)


@pytest.mark.parametrize('valid', [True, False])
def test_download_guard_covers_inherited_dataset_calls_and_restores_method(monkeypatch, valid):
    from torchvision.datasets import CIFAR100
    from datasets.seq_cifar100 import MyCIFAR100
    original = CIFAR100.download
    monkeypatch.setattr(CIFAR100, '_check_integrity', lambda subject: valid)
    subject = object.__new__(MyCIFAR100)
    with protocol.forbid_cifar_download():
        assert CIFAR100.download is not original
        if valid:
            assert subject.download() is None
        else:
            with pytest.raises(FileNotFoundError, match='forbids downloads'):
                subject.download()
    assert CIFAR100.download is original


@pytest.mark.parametrize('corruption', ['parameter', 'gradient'])
def test_alignment_finite_audit_covers_actual_updated_state_and_gradients(corruption):
    import torch
    module = torch.nn.Linear(4, 2)
    module(torch.ones(3, 4)).sum().backward()
    runner.audit_finite_learned_state(module)
    if corruption == 'parameter':
        with torch.no_grad():
            module.weight[0, 0] = float('inf')
    else:
        module.weight.grad[0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='alignment state or gradient'):
        runner.audit_finite_learned_state(module)
