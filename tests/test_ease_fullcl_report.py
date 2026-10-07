"""Saved observations, exact acceptance and fail-closed full-CL publication."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import publish_ease_fullcl as publisher
from scripts import report_ease_fullcl as reporter
from scripts.cl_efficiency import write_json
from scripts.ease_fullcl_protocol import PRECISION, STRUCTURAL_CHECKS, protocol
from scripts.hide_protocol import protocol as hide_protocol


@pytest.fixture
def records():
    rows = []
    for task in range(1, 11):
        values = [99.4] if task == 1 else [98.9, 97.] if task == 2 else [90.] * task
        if task == 10:
            values = [99.6] + [93.] * 9
        counts = [round(value * 10) for value in values]
        rows.append({'task': task, 'seen_correct': sum(counts), 'seen_samples': task * 1000,
                     'old_correct': sum(counts[:-1]), 'old_samples': (task - 1) * 1000,
                     'new_correct': counts[-1], 'new_samples': 1000,
                     'seen_accuracy_pct': sum(values) / task, 'native_class_il_per_task': values})
    base = {'status': 'completed', 'model': 'ease', 'mode': 'validation', 'protocol': protocol('validation'),
            'precision': PRECISION.copy(), 'structural_checks': dict.fromkeys(STRUCTURAL_CHECKS, True),
            'wandb_logging_status': 'uploaded', 'wandb_artifact': {'name': 'fixture:v0', 'state': 'COMMITTED'},
            'checkpoint_sha256': 'a' * 64, 'canonical_pretrained_sha256': 'b' * 64,
            'initial_state_sha256': 'c' * 64, 'initial_canonical_pretrained_sha256': 'b' * 64,
            'execution_source_sha256': {'protected.py': 'd' * 64}, 'gpu': '1',
            'first_batch_audits': [{'task': task, 'epoch': 1, 'seed': 1993 + (task - 1) * 100000,
                                   'batch_size': 64, 'inputs_sha256': 'e' * 64,
                                   'labels_sha256': 'f' * 64, 'gradient_sha256': 'a' * 64} for task in range(1, 11)]}
    reference = dict(copy.deepcopy(base), mode='reference', task_results=copy.deepcopy(rows[:2]))
    accepted = dict(copy.deepcopy(base), task_results=copy.deepcopy(rows[:2]), acceptance_status='pass')
    native = dict(copy.deepcopy(base), mode='full_cl', protocol=protocol('full_cl'), task_results=rows,
                  prefix_validation={'status': 'pass', 'prediction_match': True, 'matched_tasks': [1, 2]},
                  metrics={'outer_observe_steps': 15800, 'outer_training_stream_examples': 1000000,
                           'evaluation_examples': 55000, 'tasks_completed': 10, 'process_wall_s': 1000.,
                           'cuda_peak_reserved_mib': 20000., 'gpu_board_energy_wh': 100.},
                  epoch_results=[{'task': task, 'epoch': epoch, 'steps': 79, 'samples': 5000}
                                 for task in range(1, 11) for epoch in range(1, 21)],
                  counts={'prototype_examples': 50000, 'prototype_adapter_image_pairs': 275000,
                          'evaluation_adapter_image_pairs': 385000},
                  wandb_url='https://wandb.ai/laymond1/cl-efficiency/runs/easefixture')
    phases = {name: {'wall_s': 1., 'gpu_board_energy_j': 360., 'calls': 1}
              for group in reporter.PHASE_GROUPS.values() for name in group}
    native['per_task_resources'] = [
        {'task': task, 'start_s': (task - 1) * 10., 'end_s': task * 10., 'wall_s': 10.,
         'metrics': {'outer_observe_steps': 1580, 'outer_training_stream_examples': 100000,
                     'evaluation_examples': task * 1000, **dict.fromkeys(reporter.RESOURCE_KEYS, 100.),
                     'gpu_energy_coverage_fraction': 1.}, 'phases': copy.deepcopy(phases)} for task in range(1, 11)]
    native['data_snapshots'] = {
        f'task{task}_{split}': dict.fromkeys(reporter.RAW_FIELDS, 'observed-fixture')
        for task in range(1, 11) for split in ('train', 'test')}
    hide = dict(copy.deepcopy(native), model='hide-prompt', protocol=hide_protocol('full_cl'))
    hide['first_batch_audits'] = [
        {'task': row['task'], 'phase': 'tii_train', 'seed': row['seed'], 'samples': row['batch_size'],
         'input_sha256': row['inputs_sha256'], 'labels_sha256': row['labels_sha256']} for row in native['first_batch_audits']]
    return native, reference, accepted, hide


def test_exact_matrix_forgetting_work_and_matched_different_schedule_comparison(records):
    report = reporter.build_report(*records)
    accuracy = report['accuracy']
    assert accuracy['class_il_accuracy_matrix_pct'][0] == [99.4] + [None] * 9
    assert accuracy['final_accuracy_pct'] == pytest.approx(93.66)
    assert accuracy['average_incremental_accuracy_pct'] == pytest.approx((99.4 + 97.95 + 7 * 90 + 93.66) / 10)
    assert accuracy['forgetting_by_task'][0]['forgetting_pp'] == pytest.approx(-.2)
    assert report['validation']['current_prefix']['report_gate'] == 'pass'
    assert report['comparison']['status'] == 'compatible'
    assert report['comparison']['equal_compute_budget'] is False
    assert report['work_counts']['prototype_adapter_image_pairs'] == 275000
    assert len(publisher.tables(report)['comparison/methods'][1]) == 2


@pytest.mark.parametrize('mutation', ['counts', 'epochs', 'precision', 'initialization', 'source', 'structure', 'prefix', 'upload', 'interval', 'phase'])
def test_incomplete_or_changed_full_run_cannot_claim_completed(records, mutation):
    native = records[0]
    if mutation == 'counts':
        native['metrics']['outer_observe_steps'] -= 1
    elif mutation == 'epochs':
        native['epoch_results'].pop()
    elif mutation == 'precision':
        native['precision']['torch_cpu_threads'] = 4
    elif mutation == 'initialization':
        native['initial_state_sha256'] = 'f' * 64
    elif mutation == 'source':
        native['execution_source_sha256']['protected.py'] = 'a' * 64
    elif mutation == 'structure':
        native['structural_checks']['old_adapters_unchanged'] = False
    elif mutation == 'prefix':
        native['prefix_validation']['prediction_match'] = False
    elif mutation == 'upload':
        native['wandb_artifact']['state'] = 'PENDING'
    elif mutation == 'interval':
        native['per_task_resources'][1].update(start_s=9., wall_s=11.)
    else:
        native['per_task_resources'][1]['phases'].pop('prototype_completion')
    with pytest.raises(ValueError):
        reporter.build_report(*records)


@pytest.mark.parametrize('mutation', ['late_train_transform', 'late_test_transform', 'batch', 'first_batch', 'dtype', 'threads', 'canonical'])
def test_hide_input_comparison_excludes_even_late_task_observed_mismatches(records, mutation):
    hide = records[3]
    if mutation in ('late_train_transform', 'late_test_transform'):
        split = 'train' if mutation == 'late_train_transform' else 'test'
        hide['data_snapshots'][f'task10_{split}']['transform_repr_sha256'] = 'different'
    elif mutation == 'batch':
        hide['protocol']['batch_size'] = 48
    elif mutation == 'first_batch':
        hide['first_batch_audits'][9]['input_sha256'] = 'different'
    elif mutation == 'dtype':
        hide['precision']['autocast'] = 0  # False cannot substitute for integer zero.
    elif mutation == 'threads':
        hide['precision'].pop('torch_cpu_threads')
    else:
        hide['canonical_pretrained_sha256'] = 'different'
    report = reporter.build_report(*records)
    assert report['comparison']['status'] == 'excluded'
    assert 'methods' not in report['comparison']
    assert 'comparison/methods' not in publisher.tables(report)
    assert 'HiDe 비교에서 제외' in reporter.render_markdown(report)


@pytest.mark.parametrize('drop', [15, 16, 30, 31])
def test_accuracy_margin_boundary_is_exact_counts_not_rounded_display(records, drop):
    native = records[2]
    for key in ('seen_correct', 'new_correct'):
        native['task_results'][0][key] -= drop
    native['task_results'][0]['seen_accuracy_pct'] = 99.4  # Display is not authoritative.
    validation = reporter.matched_validation(records[1], native)
    expected = 'pass' if drop <= 15 else 'review' if drop <= 30 else 'fail'
    assert validation['accuracy_metrics'][0]['status'] == expected
    assert validation['acceptance']['status'] == expected


def saved_run(directory, record):
    directory.mkdir()
    result = directory / 'result.json'
    write_json(result, record)
    pins = {}
    for task, row in enumerate(record['task_results'], 1):
        labels = np.repeat(np.arange(task * 10, dtype=np.int64), 100)
        predictions = labels.copy()
        for index, accuracy in enumerate(row['native_class_il_per_task']):
            wrong = 1000 - round(accuracy * 10)
            start = index * 1000
            predictions[start:start + wrong] = (predictions[start:start + wrong] + 1) % (task * 10)
        path = directory / f'task{task}_predictions.npz'
        np.savez(path, predictions=predictions[:, None], labels=labels, global_ids=np.arange(len(labels)), evaluation_global_ids=np.arange(len(labels)))
        pins[path.name] = {'local_file': path.name, 'sha256': reporter.file_hash(path)}
    write_json(directory / 'artifact_files.json', pins)
    return result


@pytest.fixture
def reports(records, tmp_path):
    native, reference, accepted, hide = (saved_run(tmp_path / name, record) for name, record in
                                        zip(('native', 'reference', 'accepted', 'hide'), records))
    directory = tmp_path / 'report'
    reporter.save_report(native, reference, accepted, directory, hide)
    return directory, tmp_path / 'publication'


def test_saved_raw_prediction_audit_and_exported_scientific_artifacts(reports):
    directory, _ = reports
    assert (directory / 'accuracy_matrix.png').read_bytes().startswith(b'\x89PNG')
    assert (directory / 'task_resources.pdf').read_bytes().startswith(b'%PDF')
    assert b'\r\n' not in (directory / 'results.csv').read_bytes()
    assert len((directory / 'results.csv').read_text().splitlines()) == 11
    assert publisher.prepare(directory)[0]['comparison']['status'] == 'compatible'


def test_dry_publication_reads_saved_outputs_without_auth_network_or_writes(reports, monkeypatch, capsys):
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth in preparation'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Webhook in preparation'))
    assert publisher.main(['--report-dir', str(reports[0]), '--output', str(reports[1])]) == 0
    assert not reports[1].exists()
    assert 'execution=False' in capsys.readouterr().out


@pytest.mark.parametrize('mutation', ['source', 'matrix', 'prediction', 'manifest', 'labels', 'prefix_identity'])
def test_changed_provenance_or_raw_predictions_block_publication_before_auth(reports, monkeypatch, mutation):
    directory, output = reports
    report = reporter.read_json(directory / 'results.json')
    native = Path(report['sources']['native']['file'])
    if mutation == 'source':
        native.write_text('{}')
    elif mutation == 'matrix':
        report['accuracy']['class_il_accuracy_matrix_pct'][9][0] = 100.
        write_json(directory / 'results.json', report)
    elif mutation == 'prediction':
        (native.parent / 'task7_predictions.npz').write_bytes(b'changed')
    elif mutation == 'manifest':
        Path(report['sources']['native_artifact_manifest']['file']).write_text('{}')
    else:
        path = native.parent / ('task1_predictions.npz' if mutation == 'prefix_identity' else 'task7_predictions.npz')
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if mutation == 'prefix_identity':
            arrays['global_ids'][[0, 1]] = arrays['global_ids'][[1, 0]]
        else:
            arrays['labels'][0] = 99
        np.savez(path, **arrays)
        manifest_file = native.parent / 'artifact_files.json'
        manifest = reporter.read_json(manifest_file)
        manifest[path.name]['sha256'] = reporter.file_hash(path)
        write_json(manifest_file, manifest)
        report['sources']['native_artifact_manifest']['sha256'] = reporter.file_hash(manifest_file)
        write_json(directory / 'results.json', report)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before local gates'))
    with pytest.raises(ValueError):
        publisher.main(['--report-dir', str(directory), '--output', str(output), '--execute'])
    assert not output.exists()


def test_publication_snapshots_waits_for_commit_and_finishes_before_success(reports, monkeypatch):
    events = []
    run = SimpleNamespace(url='https://wandb.ai/laymond1/cl-efficiency/runs/reportfixture', summary={})
    class Artifact:
        state = 'COMMITTED'; qualified_name = 'laymond1/cl-efficiency/easereportfixture:v0'
        def __init__(self, *args, **kwargs):
            self.files = {}
        def add_file(self, path, name):
            self.files[name] = Path(path).read_bytes()
        def wait(self):
            events.append('committed')
    run.log = lambda table: events.append(('tables', table))
    run.log_artifact = lambda artifact: events.append(('artifact', artifact)) or artifact
    sdk = SimpleNamespace(run=run, init=lambda **kwargs: run, Artifact=Artifact, Table=lambda **kwargs: kwargs,
                          finish=lambda **kwargs: events.append(('finish', kwargs)))
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: {'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('No Slack before readback'))
    assert publisher.main(['--report-dir', str(reports[0]), '--output', str(reports[1]), '--execute']) == 0
    record = reporter.read_json(reports[1] / 'result.json')
    assert record['status'] == 'completed'
    assert record['readback_status'] == 'pending'
    assert events[-1] == ('finish', {'exit_code': 0})
    pins = reporter.read_json(reports[1] / 'artifact_files.json')
    assert set(pins) == {*reporter.REPORT_FILES, 'artifact_metadata.json'}
    for pin in pins.values():
        assert reporter.file_hash(reports[1] / pin['local_file']) == pin['sha256']


def test_memory_plot_contains_observed_allocator_reserved_device_series(records, tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    for task, row in enumerate(records[0]['per_task_resources'], 1):
        row['metrics'].update(cuda_peak_allocated_mib=6 * 1024 + task,
                              cuda_peak_reserved_mib=20 * 1024 + task,
                              peak_gpu_device_memory_mib=23 * 1024 + task)
    report = reporter.build_report(*records)
    captured = {}
    def capture(figure, path, **kwargs):
        if Path(path).stem == 'task_resources':
            axis = figure.axes[2]
            captured['series'] = {line.get_label(): list(line.get_ydata()) for line in axis.lines}
            captured['minimum'] = axis.get_ylim()[0]
    monkeypatch.setattr(Figure, 'savefig', capture)
    reporter.render_plots(report, tmp_path)
    assert captured['series'] == {'CUDA allocated': [6 + task / 1024 for task in range(1, 11)],
                                 'CUDA reserved': [20 + task / 1024 for task in range(1, 11)],
                                 'NVML device': [23 + task / 1024 for task in range(1, 11)]}
    assert captured['minimum'] == 0
