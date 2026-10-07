"""Saved-fixture scientific calculations and publication gates, with no GPU."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import publish_hide_fullcl as publisher
from scripts import report_hide_fullcl as reporter
from scripts.cl_efficiency import write_json
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS
from scripts.hide_protocol import FULL_CL_PRECISION, protocol


@pytest.fixture
def records():
    reference = {'status': 'completed', 'mode': 'validation', 'protocol': protocol('validation'),
                 'checkpoint_sha256': 'a' * 64, 'initial_state_sha256': 'b' * 64,
                 'initial_learned_sha256': {'tii': 'c' * 64, 'prompt': 'd' * 64},
                 'initial_canonical_pretrained_sha256': {'tii': 'e' * 64, 'prompt': 'e' * 64},
                 'structural_checks': dict.fromkeys(REQUIRED_STRUCTURAL_CHECKS, True),
                 'first_batch_audits': [{'task': task, 'phase': phase, 'samples': 64, 'seed': 1993,
                                        'input_sha256': 'f' * 64, 'labels_sha256': 'a' * 64}
                                       for task in (1, 2) for phase in ('tii_train', 'prompt_train')]}
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
    reference['task_results'] = copy.deepcopy(rows[:2])
    accepted = copy.deepcopy(reference)
    accepted['acceptance_status'] = 'pass'
    native = copy.deepcopy(reference)
    native.update(model='hide-prompt', mode='full_cl', protocol=protocol('full_cl'),
                  task_results=rows, precision=FULL_CL_PRECISION.copy(),
                  prefix_validation={'status': 'pass', 'prediction_match': True},
                  metrics={'outer_observe_steps': 55300, 'outer_training_stream_examples': 3500000,
                           'evaluation_examples': 55000, 'tasks_completed': 10},
                  counts={'prompt_training_tii_examples': 2500000},
                  alignment_history=[{'task': task, 'branch': branch, 'optimizer_steps': 300 * (task - 1),
                                      'skipped': task == 1, 'old_classes': 10 * (task - 1),
                                      'samples_per_centroid': 64 if branch == 'tii' else 320,
                                      'sampled_examples_per_epoch': [] if task == 1 else
                                      [10 * task * 10 * (64 if branch == 'tii' else 320)] * 30}
                                     for task in range(1, 11) for branch in ('tii', 'prompt')],
                  wandb_logging_status='uploaded', wandb_artifact={'name': 'fixture:v0', 'state': 'COMMITTED'},
                  wandb_url='https://wandb.ai/laymond1/cl-efficiency/runs/fullfixture')
    phases = {name: {'wall_s': 1., 'gpu_board_energy_j': 360., 'calls': 1}
              for group in reporter.PHASE_GROUPS.values() for name in group}
    native['task_resources'] = [
        {'task': task, 'start_s': (task - 1) * 10., 'end_s': task * 10., 'wall_s': 10.,
         'metrics': {'outer_observe_steps': 5530, 'outer_training_stream_examples': 350000,
                     'evaluation_examples': task * 1000,
                     **dict.fromkeys(reporter.RESOURCE_KEYS, 100.), 'gpu_energy_coverage_fraction': 1.},
         'phases': copy.deepcopy(phases)} for task in range(1, 11)]
    return native, reference, accepted


def test_accuracy_matrix_exact_counts_incremental_average_and_negative_forgetting(records):
    report = reporter.build_report(*records)
    accuracy = report['accuracy']
    assert accuracy['class_il_accuracy_matrix_pct'][0] == [99.4] + [None] * 9
    assert accuracy['final_accuracy_pct'] == pytest.approx(93.66)
    assert accuracy['average_incremental_accuracy_pct'] == pytest.approx((99.4 + 97.95 + 7 * 90 + 93.66) / 10)
    assert accuracy['forgetting_by_task'][0]['forgetting_pp'] == pytest.approx(-.2)
    assert accuracy['forgetting_by_task'][1]['forgetting_pp'] == 4.
    assert accuracy['forgetting_pp'] == pytest.approx((-.2 + 4 - 7 * 3) / 9)
    assert 'not_assessed' in report['validation']['full_ten_task_accuracy_acceptance']


def test_work_counts_separate_observed_synthetic_populations_from_raw_images(records):
    report = reporter.build_report(*records)
    work = report['work_counts']
    assert work['outer_optimizer_steps'] == 55300
    assert work['alignment_optimizer_steps'] == 27000
    assert work['total_optimizer_steps'] == 82300
    assert work['outer_image_training_examples'] == 3500000
    assert work['synthetic_feature_examples_generated'] == 62208000
    # Zero-variance centroids can reduce the generated population; use the
    # actual returned observation, never a classes*centroids plan estimate.
    records[0]['alignment_history'][2]['sampled_examples_per_epoch'][0] -= 64
    altered = reporter.build_report(*records)
    assert altered['work_counts']['synthetic_feature_examples_generated'] == 62208000 - 64
    assert altered['work_counts']['total_optimizer_steps'] == 82300
    assert '82,300' in reporter.render_markdown(report)
    assert publisher.scalar_summary(report)['work/alignment_optimizer_steps'] == 27000


def test_extra_tii_forward_uses_observed_counter_and_preserves_missing_values(records):
    records[0]['counts']['prompt_training_tii_examples'] = 12345
    report = reporter.build_report(*records)
    assert report['work_counts']['prompt_training_additional_tii_forward_image_examples'] == 12345
    assert report['work_counts']['outer_image_training_examples'] == 3500000
    assert publisher.scalar_summary(report)['work/prompt_training_additional_tii_forward_image_examples'] == 12345
    assert '12,345개' in reporter.render_markdown(report)
    del records[0]['counts']['prompt_training_tii_examples']
    report = reporter.build_report(*records)
    assert report['work_counts']['prompt_training_additional_tii_forward_image_examples'] is None
    assert '**미기록**' in reporter.render_markdown(report)


@pytest.mark.parametrize('field', ['seen_correct', 'new_correct', 'old_correct', 'seen_samples'])
def test_disagreeing_saved_counts_are_rejected(records, field):
    records[0]['task_results'][5][field] += 1
    with pytest.raises(ValueError, match='counts|samples'):
        reporter.build_report(*records)


@pytest.mark.parametrize('mutation', ['duplicate_task', 'nan_accuracy', 'resource_overlap', 'missing_phase',
                                      'missing_prefix', 'precision', 'epochs', 'steps', 'alignment', 'upload'])
def test_incomplete_or_changed_execution_cannot_be_published_as_completed(records, mutation):
    native = records[0]
    if mutation == 'duplicate_task':
        native['task_results'][9]['task'] = 9
    elif mutation == 'nan_accuracy':
        native['task_results'][9]['native_class_il_per_task'][0] = float('nan')
    elif mutation == 'resource_overlap':
        native['task_resources'][1].update(start_s=9., wall_s=11.)
    elif mutation == 'missing_phase':
        native['task_resources'][3]['phases'].pop('prompt_alignment')
    elif mutation == 'missing_prefix':
        native['prefix_validation']['prediction_match'] = False
    elif mutation == 'precision':
        native['precision']['tf32_allowed'] = True
    elif mutation == 'epochs':
        native['protocol']['tii_epochs'] = 19
    elif mutation == 'steps':
        native['metrics']['outer_observe_steps'] -= 1
    elif mutation == 'alignment':
        native['alignment_history'][-1]['optimizer_steps'] -= 1
    else:
        native['wandb_artifact']['state'] = 'PENDING'
    with pytest.raises(ValueError):
        reporter.build_report(*records)


def test_missing_sensor_observation_is_preserved_not_imputed(records):
    records[0]['task_resources'][0]['metrics']['gpu_board_energy_wh'] = None
    report = reporter.build_report(*records)
    assert report['resources']['tasks'][0]['metrics']['gpu_board_energy_wh'] is None
    assert '| N/A |' in reporter.render_markdown(report)


@pytest.fixture
def reports(records, tmp_path):
    paths = {}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    for name, record in zip(('native', 'reference', 'accepted_native'), records):
        path = run_dir / (name + '.json')
        write_json(path, record)
        paths[name] = {'file': str(path), 'sha256': reporter.file_hash(path)}
    manifest = {}
    for task in range(1, 11):
        filename = f'task{task}_predictions.npz'
        path = run_dir / filename
        path.write_bytes(b'Explicit CPU fixture, not model predictions')
        manifest[filename] = {'local_file': filename, 'sha256': reporter.file_hash(path)}
    manifest_file = run_dir / 'artifact_files.json'
    write_json(manifest_file, manifest)
    paths['native_artifact_manifest'] = {'file': str(manifest_file), 'sha256': reporter.file_hash(manifest_file)}
    directory = tmp_path / 'report'
    directory.mkdir()
    report = reporter.build_report(*records)
    report['sources'] = paths
    write_json(directory / 'results.json', report)
    for filename in reporter.REPORT_FILES[1:]:
        (directory / filename).write_bytes(b'Explicit CPU fixture, not an experiment artifact')
    return directory, tmp_path / 'publication'


def test_publication_default_is_stdlib_only_read_only(reports):
    command = [sys.executable, '-S', publisher.__file__, '--report-dir', str(reports[0]), '--output', str(reports[1])]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert 'execution=False' in result.stdout
    assert not reports[1].exists()


@pytest.mark.parametrize('mutation', ['source', 'matrix', 'prediction', 'manifest'])
def test_report_and_prediction_provenance_is_fail_closed_before_authentication(reports, mutation, monkeypatch):
    directory, output = reports
    report = reporter.read_json(directory / 'results.json')
    if mutation == 'source':
        Path(report['sources']['native']['file']).write_text('{}')
    elif mutation == 'matrix':
        report['accuracy']['class_il_accuracy_matrix_pct'][9][0] = 100.
        write_json(directory / 'results.json', report)
    elif mutation == 'prediction':
        (Path(report['sources']['native']['file']).parent / 'task7_predictions.npz').write_bytes(b'changed')
    else:
        Path(report['sources']['native_artifact_manifest']['file']).write_text('{}')
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before local gates'))
    with pytest.raises(ValueError):
        publisher.main(['--report-dir', str(directory), '--output', str(output), '--execute'])
    assert not output.exists()


def test_wandb_tables_contain_only_observed_matrix_cells(records):
    report = reporter.build_report(*records)
    table = publisher.tables(report)
    assert len(table['full_cl/accuracy_matrix'][1]) == 55
    assert all(row[2] is not None for row in table['full_cl/accuracy_matrix'][1])
    assert len(table['full_cl/task_metrics'][1]) == 10


def test_publication_waits_for_commit_finishes_and_preserves_immutable_pins(reports, monkeypatch):
    events = []
    run = SimpleNamespace(url='https://wandb.ai/laymond1/cl-efficiency/runs/reportfixture', summary={})
    class Artifact:
        state = 'COMMITTED'
        qualified_name = 'laymond1/cl-efficiency/reportfixture:v0'
        def __init__(self, *args, **kwargs):
            self.files = {}
        def add_file(self, path, name):
            self.files[name] = Path(path).read_bytes()
        def wait(self):
            events.append('committed')
    run.log = lambda table: events.append(('tables', table))
    run.log_artifact = lambda artifact: events.append(('artifact', artifact)) or artifact
    sdk = SimpleNamespace(run=run, init=lambda **kwargs: run, Artifact=Artifact,
                          Table=lambda **kwargs: kwargs, finish=lambda **kwargs: events.append(('finish', kwargs)))
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: {'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('No Slack before readback'))
    assert publisher.main(['--report-dir', str(reports[0]), '--output', str(reports[1]), '--execute']) == 0
    record = reporter.read_json(reports[1] / 'result.json')
    assert record['status'] == 'completed'
    assert record['readback_status'] == 'pending'
    assert events[-1] == ('finish', {'exit_code': 0})
    pins = reporter.read_json(reports[1] / 'artifact_files.json')
    artifact = next(event[1] for event in events if isinstance(event, tuple) and event[0] == 'artifact')
    assert set(pins) == {*reporter.REPORT_FILES, 'artifact_metadata.json'}
    assert set(artifact.files) == {*pins, 'artifact_files.json'}
    (reports[0] / 'report.md').write_text('later edit')
    for filename, pin in pins.items():
        assert reporter.file_hash(reports[1] / pin['local_file']) == pin['sha256']


def test_local_report_writes_scientific_png_pdf_and_newline_csv(reports, tmp_path):
    report = reporter.read_json(reports[0] / 'results.json')
    sources = report['sources']
    output = tmp_path / 'rendered'
    reporter.save_report(*(sources[name]['file'] for name in ('native', 'reference', 'accepted_native')), output)
    assert (output / 'accuracy_matrix.png').read_bytes().startswith(b'\x89PNG')
    assert (output / 'task_resources.pdf').read_bytes().startswith(b'%PDF')
    assert b'\r\n' not in (output / 'results.csv').read_bytes()
    assert len((output / 'results.csv').read_text().splitlines()) == 11


def test_completion_notification_requires_finished_hash_readback_before_webhook(reports, monkeypatch):
    output = reports[1]
    output.mkdir()
    write_json(output / 'result.json', {'status': 'completed', 'wandb_logging_status': 'uploaded'})
    write_json(output / 'artifact_files.json', {})
    proof = output / 'proof.json'
    write_json(proof, {'status': 'pass', 'run_state': 'running'})
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before readback gates'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Webhook before readback gates'))
    with pytest.raises(ValueError, match='readback proof'):
        publisher.main(['--report-dir', str(reports[0]), '--output', str(output),
                        '--execute', '--notify-after-readback', str(proof)])
