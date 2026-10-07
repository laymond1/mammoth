"""Read-only publication gates and SDK/Slack stubs; no experiment or network."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import publish_hide_sizes as publisher
from scripts import report_hide_sizes as reporter
from scripts.cl_efficiency import write_json
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS
from scripts.report_hide_results import exact_validation, method_summary


def reference_record(size):
    return {'status': 'completed', 'mode': 'validation', 'variant': size,
            'task_results': [
                {'task': 1, 'seen_correct': 990, 'seen_samples': 1000,
                 'old_correct': 0, 'old_samples': 0, 'new_correct': 990, 'new_samples': 1000},
                {'task': 2, 'seen_correct': 1940, 'seen_samples': 2000,
                 'old_correct': 965, 'old_samples': 1000, 'new_correct': 975, 'new_samples': 1000}],
            'first_batch_audits': [
                {'task': task, 'phase': phase, 'samples': 64, 'seed': 1993,
                 'input_sha256': 'a' * 64, 'labels_sha256': 'b' * 64,
                 'loss': 1., 'clipped_gradient_sha256': 'c' * 64}
                for task in (1, 2) for phase in ('tii_train', 'prompt_train')]}


def pilot_record(size, repeat, pass_name='measurement'):
    scale = {'base': 3., 'small': 2., 'tiny': 1.}[size]
    metrics = dict.fromkeys(publisher.METRICS, scale * repeat)
    metrics.update(process_wall_s=30. * scale + repeat,
                   gpu_board_energy_wh=scale + repeat / 100,
                   cuda_peak_allocated_mib=scale * 1024,
                   cuda_peak_reserved_mib=(scale + 1) * 1024,
                   peak_gpu_device_memory_mib=(scale + 2) * 1024,
                   gpu_energy_coverage_fraction=.999)
    return {'status': 'completed', 'pass': pass_name, 'metrics': metrics,
            'task_results': [{'task': 1, 'seen_correct': 900, 'seen_samples': 1000}],
            'counts': {'outer_observe_steps': 158, 'outer_training_stream_examples': 10000},
            'phases': {'tii_train': {'wall_s': 10. * scale + repeat,
                                     'gpu_board_energy_j': scale * 1200,
                                     'gpu_energy_coverage_s': 10. * scale + repeat}},
            'model_snapshots': {'tii_train': {'parameters_total': int(scale * 1000),
                                               'parameters_trainable': int(scale * 100)}},
            'alignment_history': [{'status': 'not_applicable'}, {'status': 'not_applicable'}],
            'flops': {'supported_ops_total': scale * 1e12, 'scope': 'partial'},
            'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/fixture'}


def saved_fixture_report(inputs):
    """Stub the separate scientific evidence loader, retaining real metric math.

    Reporter tests exercise native NPZ/source/precision gates. These fixtures
    exercise the publisher's binding to saved evidence and transport receipts.
    """
    records = {key: publisher.read_json(path) for key, path in inputs.items()}
    validation = {size: exact_validation(records[size + '_reference'], records[size + '_native'])
                  for size in ('small', 'tiny')}
    methods = []
    for size in publisher.VARIANTS:
        saved = records[size + '_efficiency']
        method = method_summary('hide-prompt', saved['measurements'], saved['flops'],
                                historical=size == 'base')
        method.update(variant=size, feature_dim={'base': 768, 'small': 384, 'tiny': 192}[size])
        methods.append(method)
    return {'schema_version': 1, 'generated_at': 'CPU-fixture-only', 'status': 'completed',
            'model': 'hide-prompt', 'stage': 'hide_sizes', 'validation': validation,
            'efficiency': {'status': 'completed', 'methods': methods,
                           'seed_variation_estimated': False},
            'sources': {key: {'file': str(path.resolve()), 'sha256': publisher.file_hash(path)}
                        for key, path in inputs.items()},
            'provenance': {'scope': 'Saved CPU fixture, not an experiment'},
            'full10_executed': False, 'paper_reproduction': False}


def save_csv(directory, report):
    fields = reporter.csv_fields(report)
    with (directory / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True)
                         if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in reporter.csv_rows(report))


@pytest.fixture
def reports(tmp_path, monkeypatch):
    inputs = {}
    for size in ('small', 'tiny'):
        reference = reference_record(size)
        native = copy.deepcopy(reference)
        native.update(acceptance_status='pass', structural_checks=dict.fromkeys(REQUIRED_STRUCTURAL_CHECKS, True))
        for side, record in (('reference', reference), ('native', native)):
            key = size + '_' + side
            inputs[key] = tmp_path / (key + '.json')
            write_json(inputs[key], record)
    for size in publisher.VARIANTS:
        key = size + '_efficiency'
        inputs[key] = tmp_path / (key + '.json')
        write_json(inputs[key], {'measurements': [pilot_record(size, repeat) for repeat in (1, 2, 3)],
                                 'flops': pilot_record(size, 10000, 'flops')})
    monkeypatch.setattr(reporter, 'build_report', saved_fixture_report)
    directory = tmp_path / 'report'
    directory.mkdir()
    report = saved_fixture_report(inputs)
    write_json(directory / 'results.json', report)
    save_csv(directory, report)
    (directory / 'report.md').write_text(reporter.markdown(report))
    for name in publisher.REPORT_FILES[3:]:
        (directory / name).write_bytes(b'Explicit saved CPU test artifact; no experiment result')
    return directory, tmp_path / 'publication'


@pytest.fixture
def fake_sdk(monkeypatch):
    events = []
    run = SimpleNamespace(url='https://wandb.ai/laymond1/cl-efficiency/runs/sizesfixture', summary={})

    class Artifact:
        def __init__(self, name, **kwargs):
            self.state = 'COMMITTED'
            self.qualified_name = 'laymond1/cl-efficiency/' + name + ':v0'
            self.files = {}
            self.settings = kwargs

        def add_file(self, path, name):
            self.files[name] = Path(path).read_bytes()

        def wait(self):
            events.append(('artifact_wait', self.state))

    def initialize(**kwargs):
        events.append(('init', kwargs))
        sdk.run = run
        return run

    def finish(**kwargs):
        events.append(('finish', kwargs))
        sdk.run = None

    def log_artifact(artifact):
        events.append(('artifact', artifact))
        return artifact

    run.log_artifact = log_artifact
    run.log = lambda value: events.append(('tables', value))
    sdk = SimpleNamespace(run=run, Artifact=Artifact, init=initialize, finish=finish,
                          Table=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: {
        'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Slack before verified readback'))
    return sdk, events, run


def arguments(reports):
    return ['--report-dir', str(reports[0]), '--output', str(reports[1])]


def readback_proof(output):
    record = publisher.read_json(output / 'result.json')
    pins = publisher.read_json(output / 'artifact_files.json')
    return {'status': 'pass', 'wandb_url': record['wandb_url'], 'run_state': 'finished',
            'artifact': record['wandb_artifact']['name'], 'artifact_state': 'COMMITTED',
            'result_sha256': publisher.file_hash(output / 'result.json'),
            'artifact_files': {name: {'sha256': pin['sha256'], 'matches_uploaded_pin': True,
                                     'local_snapshot_matches': True} for name, pin in pins.items()}}


def notification_arguments(reports, proof):
    path = reports[1] / 'readback.json'
    write_json(path, proof)
    return [*arguments(reports), '--execute', '--notify-after-readback', str(path)]


def test_default_cli_reads_saved_observations_without_credentials_or_writes(reports, monkeypatch, capsys):
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Credentials read in preparation'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Network send in preparation'))
    assert publisher.main(arguments(reports)) == 0
    assert not reports[1].exists()
    assert 'execution=False' in capsys.readouterr().out


def test_cli_import_and_help_need_no_model_or_site_packages():
    result = subprocess.run([sys.executable, '-S', str(Path(publisher.__file__).resolve()), '--help'],
                            capture_output=True, text=True, check=True)
    assert '--notify-after-readback' in result.stdout


def test_prepare_integrates_real_saved_npz_counts_protocol_and_dependent_evidence(tmp_path, monkeypatch):
    from tests.test_report_hide_sizes import saved_inputs
    saved = saved_inputs.__wrapped__(tmp_path, monkeypatch)

    def saved_graphs(report, output):
        # Plot rendering is independently tested; keep this provenance check light.
        for filename in ('efficiency_comparison.png', 'efficiency_comparison.pdf'):
            (output / filename).write_bytes(b'CPU fixture graph')

    monkeypatch.setattr(reporter, 'plot_report', saved_graphs)
    directory = tmp_path / 'scientific-report'
    report = reporter.build_report(saved.inputs)
    reporter.write_report(report, directory)
    actual, pins = publisher.prepare(directory)
    assert actual['validation']['small']['accuracy_metrics'][0]['native_correct'] == 1000
    assert actual['validation']['tiny']['accuracy_metrics'][1]['native_samples'] == 2000
    assert actual['efficiency']['methods'][1]['resources']['process_wall_s']['mean'] == 60.
    assert set(pins) == set(publisher.REPORT_FILES)
    # Inputs' JSON bytes stay pinned, but changing their dependent predictions
    # must still block publication before credentials or writes.
    native = publisher.read_json(saved.inputs['tiny_native'])
    prediction = Path(native['output']) / 'task1_predictions.npz'
    prediction.write_bytes(prediction.read_bytes() + b'Corrupted dependent evidence')
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before NPZ provenance'))
    with pytest.raises(ValueError, match='SHA256 differs'):
        publisher.main(['--report-dir', str(directory), '--output', str(tmp_path / 'blocked-upload'), '--execute'])
    assert not (tmp_path / 'blocked-upload').exists()


@pytest.mark.parametrize('filename', publisher.REPORT_FILES)
def test_missing_artifact_blocks_authentication_and_publication_directory(reports, monkeypatch, filename):
    (reports[0] / filename).unlink()
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Credentials before local gates'))
    with pytest.raises(ValueError, match='Missing or empty'):
        publisher.main([*arguments(reports), '--execute'])
    assert not reports[1].exists()


@pytest.mark.parametrize('mutation', ['changed_source', 'missing_source', 'relative_source', 'extra_source'])
def test_seven_pinned_sources_are_required_before_authentication(reports, monkeypatch, mutation):
    report = publisher.read_json(reports[0] / 'results.json')
    if mutation == 'changed_source':
        Path(report['sources']['small_reference']['file']).write_text('{}')
    elif mutation == 'missing_source':
        del report['sources']['tiny_native']
    elif mutation == 'relative_source':
        report['sources']['base_efficiency']['file'] = 'relative.json'
    else:
        report['sources']['unexpected'] = report['sources']['tiny_native']
    write_json(reports[0] / 'results.json', report)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Credentials before provenance'))
    with pytest.raises(ValueError, match='provenance|seven'):
        publisher.main([*arguments(reports), '--execute'])
    assert not reports[1].exists()


@pytest.mark.parametrize('mutation', ['accuracy', 'resource', 'counts', 'source_scope', 'precision_claim'])
def test_entire_payload_is_recomputed_instead_of_trusting_selected_scalars(reports, mutation):
    report = publisher.read_json(reports[0] / 'results.json')
    if mutation == 'accuracy':
        report['validation']['small']['accuracy_metrics'][0]['native_accuracy_pct'] = 100.
    elif mutation == 'resource':
        report['efficiency']['methods'][1]['resources']['gpu_board_energy_wh']['mean'] = .001
    elif mutation == 'counts':
        report['efficiency']['methods'][2]['counts'][0]['outer_training_stream_examples'] = 1
    elif mutation == 'source_scope':
        report['provenance']['scope'] = 'Forged source verification'
    else:
        report['efficiency']['seed_variation_estimated'] = True
    write_json(reports[0] / 'results.json', report)
    with pytest.raises(ValueError, match='payload disagrees'):
        publisher.prepare(reports[0])


def test_generation_timestamp_is_the_only_ignored_payload_field(reports):
    report = publisher.read_json(reports[0] / 'results.json')
    report['generated_at'] = 'Later local rendering timestamp'
    write_json(reports[0] / 'results.json', report)
    assert publisher.prepare(reports[0])[0]['generated_at'] == report['generated_at']


def test_markdown_cannot_claim_different_cost_or_validation_results(reports):
    path = reports[0] / 'report.md'
    path.write_text(path.read_text().replace('97.00', '100.00'))
    with pytest.raises(ValueError, match='Markdown disagrees'):
        publisher.prepare(reports[0])


@pytest.mark.parametrize('mutation', ['missing_row', 'changed_mean', 'changed_variant', 'wrong_header'])
def test_csv_must_match_every_saved_accuracy_resource_and_phase_observation(reports, mutation):
    path = reports[0] / 'results.csv'
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if mutation == 'missing_row':
        rows.pop()
    elif mutation == 'changed_mean':
        next(row for row in rows if row['metric'] == 'gpu_board_energy_wh')['mean'] = '0.001'
    elif mutation == 'changed_variant':
        rows[0]['variant'] = 'base'
    else:
        fields.reverse()
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='CSV disagrees'):
        publisher.prepare(reports[0])


@pytest.mark.parametrize('size,mutation', [('small', 'blocked'), ('tiny', 'structure'),
                                         ('small', 'batch'), ('tiny', 'review')])
def test_both_sizes_must_pass_accuracy_structural_and_actual_first_batch_gates(reports, size, mutation):
    report = publisher.read_json(reports[0] / 'results.json')
    validation = report['validation'][size]
    if mutation == 'blocked':
        validation['report_gate'] = 'blocked'
    elif mutation == 'structure':
        validation['acceptance']['structural_checks'][REQUIRED_STRUCTURAL_CHECKS[0]] = False
    elif mutation == 'batch':
        validation['first_batch_comparison']['rows'][0]['status'] = 'fail'
    else:
        validation['accuracy_metrics'][0]['status'] = 'review'
    write_json(reports[0] / 'results.json', report)
    with pytest.raises(ValueError, match='Both sizes'):
        publisher.prepare(reports[0])


@pytest.mark.parametrize('mutation', ['duplicate_size', 'two_measurements', 'no_flops', 'boolean_flops'])
def test_separate_three_measurements_and_partial_flops_are_required_for_each_size(reports, mutation):
    report = publisher.read_json(reports[0] / 'results.json')
    method = report['efficiency']['methods'][2]
    if mutation == 'duplicate_size':
        method['variant'] = 'small'
    elif mutation == 'two_measurements':
        method['measurement_count'] = 2
    elif mutation == 'no_flops':
        method['flops']['supported_ops_total'] = None
    else:
        method['flops']['supported_ops_total'] = True
    write_json(reports[0] / 'results.json', report)
    with pytest.raises(ValueError, match='measurements|FLOPs'):
        publisher.prepare(reports[0])


def test_publish_uploads_immutable_snapshots_tables_and_only_finishes_then_succeeds(reports, fake_sdk):
    _, events, run = fake_sdk
    assert publisher.main([*arguments(reports), '--execute']) == 0
    output = reports[1]
    record = publisher.read_json(output / 'result.json')
    assert record['status'] == 'completed'
    assert record['slack_completion_status'] == 'deferred_until_readback'
    assert record['new_training'] is False and record['new_gpu_computation'] is False
    settings = next(value for event, value in events if event == 'init')
    assert (settings['entity'], settings['project'], settings['group']) == ('laymond1', 'cl-efficiency', 'hide-size-results')
    artifact = next(value for event, value in events if event == 'artifact')
    assert artifact.settings['type'] == 'cl-hide-size-report'
    pins = publisher.read_json(output / 'artifact_files.json')
    assert set(pins) == {*publisher.REPORT_FILES, 'artifact_metadata.json'}
    assert set(artifact.files) == {*pins, 'artifact_files.json'}
    for name, pin in pins.items():
        assert publisher.file_hash(output / pin['local_file']) == pin['sha256']
        assert artifact.files[name] == (output / pin['local_file']).read_bytes()
    (reports[0] / 'report.md').write_text('Later human edit')
    assert (output / 'snapshots/report.md').read_bytes() == artifact.files['report.md']
    assert events[-1] == ('finish', {'exit_code': 0})
    assert run.summary['full_cl_included'] is False and run.summary['equal_compute_budget'] is False
    assert run.summary['efficiency/small/gpu_board_energy_wh/mean'] == pytest.approx(2.02)
    tables = next(value for event, value in events if event == 'tables')
    assert len(tables['validation/accuracy']['data']) == 8
    assert len(tables['efficiency/resources']['data']) == 3
    assert len(tables['efficiency/phases']['data']) == 3
    assert len(tables['efficiency/parameters']['data']) == 18


def test_copy_race_is_rejected_before_authentication(reports, monkeypatch):
    copyfile = publisher.shutil.copyfile

    def altered(source, destination):
        copyfile(source, destination)
        if Path(source).name == 'report.md':
            Path(destination).write_text('Altered during snapshot')

    monkeypatch.setattr(publisher.shutil, 'copyfile', altered)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before snapshot gate'))
    with pytest.raises(ValueError, match='immutable snapshots'):
        publisher.main([*arguments(reports), '--execute'])
    assert not (reports[1] / 'result.json').exists()


@pytest.mark.parametrize('failure', ['upload', 'uncommitted', 'finish'])
def test_failed_upload_commit_or_finish_never_claims_success_or_serializes_secret_text(reports, fake_sdk, capsys, failure):
    sdk, _, run = fake_sdk

    def secret_failure(*args, **kwargs):
        raise RuntimeError('Private credential marker must not appear')

    if failure == 'upload':
        run.log_artifact = secret_failure
    elif failure == 'finish':
        sdk.finish = secret_failure
    else:
        original = run.log_artifact
        def not_committed(artifact):
            artifact.state = 'PENDING'
            return original(artifact)
        run.log_artifact = not_committed
    assert publisher.main([*arguments(reports), '--execute']) == 1
    record = publisher.read_json(reports[1] / 'result.json')
    assert record['status'] == 'logging_failed' and record['wandb_logging_status'].startswith('failed:')
    assert record['slack_completion_status'] == 'deferred_until_readback'
    assert 'Private credential marker' not in (reports[1] / 'result.json').read_text()
    assert 'Private credential marker' not in capsys.readouterr().out


def test_wrong_project_never_initializes_sdk(reports, fake_sdk, monkeypatch):
    _, events, _ = fake_sdk
    monkeypatch.setattr(publisher, 'configure_environment', lambda: {'wandb_entity': 'other', 'wandb_project': 'cl-efficiency'})
    assert publisher.main([*arguments(reports), '--execute']) == 1
    assert not events


def test_verified_notification_contains_actual_means_and_is_idempotent(reports, fake_sdk, monkeypatch):
    assert publisher.main([*arguments(reports), '--execute']) == 0
    proof = readback_proof(reports[1])
    cli = notification_arguments(reports, proof)
    sends = []
    monkeypatch.setattr(publisher, 'notify_slack', lambda message: sends.append(message) or 'sent')
    assert publisher.main(cli) == 0
    # A newly saved, valid readback receipt must not repeat the same publication.
    proof['additional_readback_note'] = 'Second verification'
    write_json(reports[1] / 'readback.json', proof)
    assert publisher.main(cli) == 0
    assert len(sends) == 1
    assert 'Small: 검증 태스크2 정확도 97.00%' in sends[0]
    assert '62.0초, GPU 보드 2.020Wh, CUDA allocated 2.00GiB, NVML 4.00GiB' in sends[0]
    assert 'Tiny: 검증 태스크2 정확도 97.00%' in sends[0]
    assert '32.0초, GPU 보드 1.020Wh, CUDA allocated 1.00GiB, NVML 3.00GiB' in sends[0]
    receipt = publisher.read_json(reports[1] / 'notification.json')
    assert receipt['status'] == 'sent' and receipt['result_sha256'] == proof['result_sha256']


@pytest.mark.parametrize('mutation', ['running', 'uncommitted', 'wrong_result', 'wrong_artifact',
                                     'other_run', 'missing_file', 'failed_file', 'wrong_file_hash'])
def test_incomplete_or_mismatched_readback_blocks_credentials_and_slack(reports, fake_sdk, monkeypatch, mutation):
    publisher.main([*arguments(reports), '--execute'])
    proof = readback_proof(reports[1])
    if mutation == 'running':
        proof['run_state'] = 'running'
    elif mutation == 'uncommitted':
        proof['artifact_state'] = 'PENDING'
    elif mutation == 'wrong_result':
        proof['result_sha256'] = '0' * 64
    elif mutation == 'wrong_artifact':
        proof['artifact'] = 'laymond1/cl-efficiency/another:v0'
    elif mutation == 'other_run':
        proof['wandb_url'] = 'https://wandb.ai/laymond1/cl-efficiency/runs/other'
    elif mutation == 'missing_file':
        del proof['artifact_files']['results.csv']
    elif mutation == 'failed_file':
        proof['artifact_files']['report.md']['local_snapshot_matches'] = False
    else:
        proof['artifact_files']['results.json']['sha256'] = '0' * 64
    cli = notification_arguments(reports, proof)
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before readback gates'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Slack before readback gates'))
    with pytest.raises(ValueError):
        publisher.main(cli)
    assert not (reports[1] / 'notification.json').exists()


@pytest.mark.parametrize('mutation', ['snapshot', 'outside_path', 'record_summary', 'record_pin', 'project_url'])
def test_notification_binds_local_snapshots_uploaded_metadata_and_current_record(reports, fake_sdk, monkeypatch, mutation):
    publisher.main([*arguments(reports), '--execute'])
    output = reports[1]
    if mutation == 'snapshot':
        (output / 'snapshots/results.json').write_text('{}')
    elif mutation == 'outside_path':
        pins = publisher.read_json(output / 'artifact_files.json')
        external = output.parent / 'external-report.md'
        external.write_bytes((output / 'snapshots/report.md').read_bytes())
        pins['report.md']['local_file'] = str(external)
        write_json(output / 'artifact_files.json', pins)
    else:
        record = publisher.read_json(output / 'result.json')
        if mutation == 'record_summary':
            record['summary']['efficiency/tiny/gpu_board_energy_wh/mean'] = .001
        elif mutation == 'record_pin':
            record['report_files_sha256']['results.csv'] = '0' * 64
        else:
            record['wandb_url'] = 'https://wandb.ai/other/cl-efficiency/runs/sizesfixture'
        write_json(output / 'result.json', record)
    cli = notification_arguments(reports, readback_proof(output))
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth before local proof binding'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Slack before local proof binding'))
    with pytest.raises(ValueError):
        publisher.main(cli)
    assert not (output / 'notification.json').exists()


def test_failed_slack_can_be_retried_once_then_idempotently_ignored(reports, fake_sdk, monkeypatch):
    publisher.main([*arguments(reports), '--execute'])
    cli = notification_arguments(reports, readback_proof(reports[1]))
    responses, sends = iter(['failed:network', 'sent']), []
    monkeypatch.setattr(publisher, 'notify_slack', lambda message: sends.append(message) or next(responses))
    assert publisher.main(cli) == 1
    assert publisher.main(cli) == 0
    assert publisher.main(cli) == 0
    assert len(sends) == 2


def test_existing_sent_receipt_for_another_publication_is_rejected(reports, fake_sdk, monkeypatch):
    publisher.main([*arguments(reports), '--execute'])
    cli = notification_arguments(reports, readback_proof(reports[1]))
    write_json(reports[1] / 'notification.json', {'status': 'sent', 'result_sha256': '0' * 64})
    monkeypatch.setattr(publisher, 'configure_environment', lambda: pytest.fail('Auth for stale notification'))
    with pytest.raises(ValueError, match='another publication'):
        publisher.main(cli)
