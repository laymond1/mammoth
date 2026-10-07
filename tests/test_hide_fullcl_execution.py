"""Full-stream execution gates and measurement boundaries without GPU training."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import hide_protocol as protocol
from scripts import run_hide_validation as runner
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS
from tests.test_hide_execution import completed_fixture, reference_rows


def accepted_fixture(tmp_path, monkeypatch):
    initial = tmp_path / 'initial.pt'
    checkpoint = tmp_path / 'pretrained.npz'
    reference = tmp_path / 'reference.json'
    hashes = {initial: 'learned-init', checkpoint: protocol.CHECKPOINT_SHA256}
    reference.write_text(json.dumps({'status': 'completed', 'protocol': protocol.protocol('validation'),
                                     'reference_commit': protocol.REFERENCE_COMMIT,
                                     'checkpoint_sha256': protocol.CHECKPOINT_SHA256,
                                     'initial_state_sha256': 'learned-init', 'task_results': reference_rows()}))
    original_hash = protocol.file_hash
    source_pins = {source: original_hash(protocol.ROOT / source) for source in protocol.SOURCE_FILES}
    source_pins['scripts/run_hide_validation.py'] = 'historical-accepted-runner'
    source_pins['scripts/hide_protocol.py'] = 'historical-accepted-protocol'
    record = {'status': 'completed', 'acceptance_status': 'pass', 'mode': 'validation', 'model': 'hide-prompt',
              'protocol': protocol.protocol('validation'), 'wandb_logging_status': 'uploaded',
              'wandb_artifact': {'state': 'COMMITTED'},
              'structural_checks': {key: True for key in REQUIRED_STRUCTURAL_CHECKS},
              'execution_source_sha256': source_pins, 'precision': protocol.FULL_CL_PRECISION,
              'checkpoint_file': str(checkpoint), 'checkpoint_sha256': protocol.CHECKPOINT_SHA256,
              'canonical_pretrained_sha256': protocol.CANONICAL_PRETRAINED_SHA256,
              'initial_state': str(initial), 'initial_state_sha256': 'learned-init',
              'reference_summary': str(reference), 'reference_summary_sha256': original_hash(reference),
              'task_results': reference_rows(), 'wandb_artifact_files': {}}
    for task in (1, 2):
        name = f'task{task}_predictions.npz'
        (tmp_path / name).write_bytes(bytes([task]))
        record['wandb_artifact_files'][name] = {'local_file': name, 'sha256': original_hash(tmp_path / name)}
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr(protocol, 'file_hash', lambda path: hashes.get(Path(path)) or original_hash(path))
    return record, path, initial


def test_full_cl_exact_recipe_and_native_cli_preserve_all_class_partitions():
    settings = protocol.protocol('full_cl')
    assert settings['tasks'] == settings['dataset_total_tasks'] == 10
    assert settings['tii_epochs'] == 20 and settings['prompt_epochs'] == 50 and settings['ca_epochs'] == 30
    assert settings['n_epochs'] == 70 and settings['batch_size'] == 64 and settings['seed'] == 1993
    assert settings['accuracy_acceptance_eligible'] is False and settings['paper_reproduction'] is False
    command = protocol.native_command('full_cl')
    assert command[command.index('--stop_after') + 1] == '10'
    assert command[command.index('--dataset_config') + 1] == 'hide_prompt'
    assert '--num_tasks' not in command and '--nowand' not in command
    runner.validate_spec({'mode': 'full_cl', 'protocol': settings, 'gpu': '1', 'model': 'hide-prompt',
                          'checkpoint_sha256': protocol.CHECKPOINT_SHA256, 'pass': 'measurement'})


def test_full_cl_completion_checks_all_700_epochs_and_55k_evaluations():
    record, recorder, model, settings = completed_fixture('full_cl')
    assert runner.completion_checks(record, recorder, model, settings)
    assert len(record['epoch_results']) == 700 and recorder.steps == 55300
    assert recorder.training_examples == 3500000 and recorder.evaluation_examples == 55000
    record['epoch_results'][-1]['samples'] -= 1
    assert not runner.completion_checks(record, recorder, model, settings)


def test_full_cl_dry_run_remains_stdlib_only_and_does_not_read_credentials():
    script = ('import sys; from scripts.run_hide_validation import main; main(["--mode", "full_cl"]); '
              'assert "torch" not in sys.modules; assert "wandb" not in sys.modules')
    completed = subprocess.run([sys.executable, '-S', '-c', script], cwd=runner.ROOT,
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert 'execution=False' in completed.stdout and '"tasks": 10' in completed.stdout


def test_full_cl_allows_only_explicit_instrumentation_source_changes(tmp_path, monkeypatch):
    record, path, initial = accepted_fixture(tmp_path, monkeypatch)
    assert protocol.validate_full_cl_acceptance(path, initial_state=initial)['task_results'] == record['task_results']
    with pytest.raises(ValueError, match='accepted execution sources have changed'):
        protocol.validate_efficiency_acceptance(path, initial_state=initial)


@pytest.mark.parametrize('source', protocol.FULL_CL_PROTECTED_FILES)
def test_each_protected_source_blocks_changed_implementation(tmp_path, monkeypatch, source):
    record, path, initial = accepted_fixture(tmp_path, monkeypatch)
    record['execution_source_sha256'][source] = 'changed'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='protected implementation sources'):
        protocol.validate_full_cl_acceptance(path, initial_state=initial)


@pytest.mark.parametrize('corruption', ['source_pin_missing', 'precision', 'checkpoint', 'canonical', 'initial',
                                      'reference', 'prediction', 'structure', 'uploaded', 'artifact', 'accuracy'])
def test_full_cl_prerequisite_rejects_missing_or_changed_validation_evidence(tmp_path, monkeypatch, corruption):
    record, path, initial = accepted_fixture(tmp_path, monkeypatch)
    if corruption == 'source_pin_missing': record['execution_source_sha256'].pop('scripts/run_hide_validation.py')
    if corruption == 'precision': record['precision'] = {**record['precision'], 'autocast': True}
    if corruption == 'checkpoint': record['checkpoint_sha256'] = 'other'
    if corruption == 'canonical': record['canonical_pretrained_sha256'] = 'other'
    if corruption == 'initial': record['initial_state_sha256'] = 'other'
    if corruption == 'reference': record['reference_summary_sha256'] = 'other'
    if corruption == 'prediction': (tmp_path / 'task2_predictions.npz').write_bytes(b'changed')
    if corruption == 'structure': record['structural_checks']['routing_without_task_oracle'] = False
    if corruption == 'uploaded': record['wandb_logging_status'] = 'failed'
    if corruption == 'artifact': record['wandb_artifact']['state'] = 'PENDING'
    if corruption == 'accuracy': record['task_results'] = reference_rows(850)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        protocol.validate_full_cl_acceptance(path, initial_state=initial)


@pytest.mark.parametrize('key', ['predictions', 'labels', 'global_ids', 'tii_predictions', 'evaluation_global_ids'])
def test_prefix_comparison_checks_all_prediction_and_routing_arrays(tmp_path, key):
    import numpy as np
    data = {'predictions': np.arange(15).reshape(3, 5), 'labels': np.arange(3),
            'global_ids': np.arange(3), 'tii_predictions': np.arange(3), 'evaluation_global_ids': np.arange(3)}
    accepted, actual = tmp_path / 'accepted.npz', tmp_path / 'actual.npz'
    np.savez(accepted, **data)
    np.savez(actual, **data)
    assert runner.compare_prefix_predictions(actual, accepted)
    changed = deepcopy(data)
    changed[key].flat[0] += 1
    np.savez(actual, **changed)
    with pytest.raises(RuntimeError, match='prefix predictions differ'):
        runner.compare_prefix_predictions(actual, accepted)


class ResourceFixture:
    def __init__(self):
        self.started = 0.
        self.interval = 1.
        self.torch, self.device = None, None
        self.samples, self.phases = [], []
        self.steps, self.training_examples, self.evaluation_examples = 0, 0, 0
        self.peak = 100.
        self.clock = 10.

    def synchronize(self):
        pass

    def cuda_memory(self):
        return {'cuda_peak_allocated_mib': self.peak, 'cuda_peak_reserved_mib': self.peak + 5.}

    def sample(self):
        self.samples.append({'elapsed_s': self.clock, 'gpu_power_w': 200.,
                             'process_tree_rss_mib': 1024., 'gpu_device_memory_mib': 400.})


def test_task_resources_integrate_disjoint_intervals_and_preserve_global_cuda_peak(monkeypatch):
    recorder = ResourceFixture()
    monkeypatch.setattr(runner.time, 'perf_counter', lambda: recorder.clock)
    resources = runner.TaskResources(recorder)
    monkeypatch.setattr(resources, 'cpu_seconds', lambda: recorder.clock * .5)
    resources.start(1)
    recorder.clock, recorder.peak = 12., 350.
    recorder.sample()
    recorder.phases.extend([{'phase': 'tii_train', 'start_s': 10., 'end_s': 12.},
                            {'phase': 'evaluation', 'start_s': 12., 'end_s': 14.}])
    recorder.clock = 14.
    recorder.samples.reverse()  # Boundary and monitor samples may append out of timestamp order.
    recorder.steps, recorder.training_examples, recorder.evaluation_examples = 5530, 350000, 1000
    row = resources.finish(1)
    assert row['wall_s'] == 4. and row['metrics']['gpu_board_energy_j'] == 800.
    assert row['metrics']['gpu_energy_coverage_fraction'] == 1.
    assert row['metrics']['cuda_peak_allocated_mib'] == 350.
    assert row['metrics']['process_tree_cpu_s'] == 2.
    assert row['phases']['tii_train']['gpu_board_energy_j'] == 400.
    recorder.clock, recorder.peak = 20., 200.
    resources.start(2)
    recorder.clock, recorder.peak = 22., 250.
    recorder.steps, recorder.training_examples, recorder.evaluation_examples = 11060, 700000, 3000
    row = resources.finish(2)
    assert row['metrics']['outer_observe_steps'] == 5530
    assert row['metrics']['outer_training_stream_examples'] == 350000
    assert row['metrics']['evaluation_examples'] == 2000
    assert row['metrics']['cuda_peak_allocated_mib'] == 250.
    assert resources.global_cuda['cuda_peak_allocated_mib'] == 350.


def test_task_resources_missing_power_remains_unavailable_and_invalid_order_fails(monkeypatch):
    recorder = ResourceFixture()
    monkeypatch.setattr(runner.time, 'perf_counter', lambda: recorder.clock)
    resources = runner.TaskResources(recorder)
    with pytest.raises(RuntimeError, match='sequential'):
        resources.start(2)
    resources.start(1)
    with pytest.raises(RuntimeError, match='nonoverlapping'):
        resources.start(1)
    recorder.clock = 12.
    recorder.samples[0]['gpu_power_w'] = None
    row = resources.finish(1)
    assert row['metrics']['gpu_board_energy_wh'] is None
    assert row['metrics']['gpu_energy_coverage_fraction'] == 0.
    with pytest.raises(RuntimeError, match='outside the active'):
        resources.finish(1)
