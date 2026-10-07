"""Saved fixture aggregation/publication checks; no experiments or networking."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import publish_low_rank_results as publisher
from scripts import report_low_rank_results as reporter
from scripts.cl_efficiency import write_json
from scripts.low_rank_protocol import (
    CANONICAL_PRETRAINED_SHA256, CHECKPOINT_SHA256, METHODS, REFERENCE_COMMIT,
    REQUIRED_STRUCTURAL_CHECKS, phase_seed, protocol,
)


@pytest.fixture
def saved_reports(tmp_path):
    manifest = {}
    for index, method in enumerate(METHODS):
        settings = protocol(method)
        reference = {'method': method, 'status': 'completed', 'mode': 'validation', 'protocol': settings,
            'wandb_logging_status': 'uploaded', 'reference_commit': REFERENCE_COMMIT,
            'checkpoint_sha256': CHECKPOINT_SHA256, 'canonical_pretrained_sha256': CANONICAL_PRETRAINED_SHA256,
            'initial_state_sha256': str(index + 1) * 64, 'initial_learned_sha256': str(index + 5) * 64,
            'structural_checks': dict.fromkeys(REQUIRED_STRUCTURAL_CHECKS, True),
            'task_results': [
                {'task': 1, 'seen_correct': 900, 'seen_samples': 1000, 'old_correct': 0,
                 'old_samples': 0, 'new_correct': 900, 'new_samples': 1000},
                {'task': 2, 'seen_correct': 1750, 'seen_samples': 2000, 'old_correct': 850,
                 'old_samples': 1000, 'new_correct': 900, 'new_samples': 1000}],
            'counts': {'observe_calls': settings['n_epochs'] * 79 * 2,
                       'optimizer_updates': settings['n_epochs'] * 79 * (3 if method == 'cl-lora' else 2)},
            'epoch_results': [{'task': task, 'epoch': epoch, 'steps': 79, 'samples': 5000}
                              for task in (1, 2) for epoch in range(1, settings['n_epochs'] + 1)],
            'first_batch_audits': [{'task': task, 'epoch': 1, 'seed': phase_seed(1993, task - 1, 'train'),
                'optimizer_update_index': update, 'samples': 64, 'input_sha256': 'a' * 64,
                'labels_sha256': 'b' * 64, 'gradient_sha256': 'c' * 64}
                for task, update in ([(1, 1), (2, 1)] + ([(2, 2)] if method == 'cl-lora' else []))],
            'metrics': {'process_wall_s': 120., 'cuda_peak_allocated_mib': 1024.,
                        'gpu_board_energy_wh': 1., 'gpu_energy_coverage_fraction': 0.9}}
        extra_begin, extra_end = {'ewc-lora': (0, 10000), 'sd-lora': (0, 0),
                                  'inf-lora': (10000, 10000), 'cl-lora': (0, 15000)}[method]
        reference['counts'].update(task_begin_extra_samples=extra_begin, task_end_extra_samples=extra_end)
        native = copy.deepcopy(reference)
        native.update(acceptance_status='pass')
        manifest[method] = {}
        for side, record in (('reference', reference), ('native', native)):
            directory = tmp_path / method / side
            directory.mkdir(parents=True)
            record['wandb_url'] = f'https://wandb.ai/laymond1/cl-efficiency/runs/{index}{side}'
            write_json(directory / 'spec.json', {'method': method, 'protocol': settings, 'reference_commit': REFERENCE_COMMIT})
            audits = []
            record['prediction_files'] = []
            for task in (1, 2):
                predictions = directory / f'task{task}_predictions.npz'
                predictions.write_bytes(f'Explicit CPU fixture {method}/{side}/{task}'.encode())
                record['prediction_files'].append({'task': task, 'path': str(predictions), 'sha256': reporter.file_hash(predictions)})
                checkpoint = directory / f'task{task}_state.pt'
                checkpoint.write_bytes(f'Explicit CPU checkpoint fixture {method}/{side}/{task}'.encode())
                audits.append({'task': task, 'checkpoint_sha256': reporter.file_hash(checkpoint)})
            record['state_audits'] = {'tasks': audits}
            path = directory / ('summary.json' if side == 'reference' else 'result.json')
            write_json(path, record)
            manifest[method][side] = str(path)
    path = tmp_path / 'inputs.json'
    write_json(path, manifest)
    output = tmp_path / 'report'
    assert reporter.main(['--inputs', str(path), '--output', str(output), '--write']) == 0
    return SimpleNamespace(inputs=path, report_dir=output, publish_dir=tmp_path / 'publication')


def test_exact_four_metric_aggregation_and_korean_validation_scope(saved_reports):
    report = reporter.read_json(saved_reports.report_dir / 'results.json')
    assert report['publication_gate'] == 'pass'
    assert report['paper_reproduction'] is False and report['efficiency_comparison'] is False
    for entry in report['methods'].values():
        assert len(entry['accuracy_metrics']) == 4
        assert all(row['drop_pp_exact'] == {'numerator': 0, 'denominator': 1} for row in entry['accuracy_metrics'])
        assert entry['runs']['native']['resource_diagnostics']['observation_count'] == 1
        assert entry['runs']['native']['resource_diagnostics']['metrics']['process_tree_cpu_s'] is None
    text = (saved_reports.report_dir / 'RESULTS.md').read_text()
    assert '900/1000 (90.00%)' in text and '1750/2000 (87.50%)' in text
    assert '효율성 파일럿 결과가 아닙니다' in text and '효율성 순위를 매기지 않습니다' in text
    assert text.count('https://wandb.ai/') == 8


def test_dry_runs_are_stdlib_only_without_credentials_external_actions_or_writes(saved_reports, monkeypatch):
    monkeypatch.setattr(publisher, 'configure_environment', lambda **kw: pytest.fail('Dry run read credentials'))
    monkeypatch.setattr(publisher, 'notify_slack', lambda *args: pytest.fail('Dry run sent notification'))
    fresh = saved_reports.publish_dir
    assert reporter.main(['--inputs', str(saved_reports.inputs), '--output', str(fresh)]) == 0
    assert publisher.main(['--report-dir', str(saved_reports.report_dir), '--output', str(fresh)]) == 0
    assert not fresh.exists()
    for script, arguments in ((reporter, ['--inputs', str(saved_reports.inputs)]),
                              (publisher, ['--report-dir', str(saved_reports.report_dir), '--output', str(fresh)])):
        process = subprocess.run([sys.executable, '-S', script.__file__, *arguments], capture_output=True, text=True, check=True)
        assert 'false' in process.stdout.lower()
    assert not fresh.exists()


def test_exact_count_boundary_and_saved_acceptance_consistency(saved_reports):
    pairs, sources = reporter.load_inputs(saved_reports.inputs)
    native = pairs['ewc-lora']['native']
    native['task_results'][0].update(seen_correct=885, new_correct=885)
    report = reporter.build_report(pairs, sources)
    assert report['methods']['ewc-lora']['accuracy_metrics'][0]['drop_pp_exact'] == {'numerator': 3, 'denominator': 2}
    native['task_results'][0].update(seen_correct=884, new_correct=884)
    with pytest.raises(ValueError, match='Saved acceptance'):
        reporter.build_report(pairs, sources)
    native['acceptance_status'] = 'review'
    assert reporter.build_report(pairs, sources)['publication_gate'] == 'blocked'


@pytest.mark.parametrize('mutation', ['counts', 'markdown', 'csv', 'checkpoint', 'predictions', 'source', 'first_batch'])
def test_changed_counts_rendering_or_provenance_blocks_before_auth(saved_reports, mutation, monkeypatch):
    report = reporter.read_json(saved_reports.report_dir / 'results.json')
    if mutation == 'counts':
        report['methods']['ewc-lora']['accuracy_metrics'][0]['native_correct'] = 1000
        write_json(saved_reports.report_dir / 'results.json', report)
    elif mutation in ('markdown', 'csv'):
        (saved_reports.report_dir / ('RESULTS.md' if mutation == 'markdown' else 'results.csv')).write_text('forged')
    elif mutation == 'checkpoint':
        Path(report['sources']['ewc-lora']['native']['artifacts']['local_checkpoints']['1']['file']).write_bytes(b'changed')
    elif mutation == 'predictions':
        Path(report['sources']['ewc-lora']['native']['artifacts']['files']['task1_predictions.npz']['file']).write_bytes(b'changed')
    else:
        source = Path(report['sources']['ewc-lora']['native']['file'])
        value = reporter.read_json(source)
        if mutation == 'source':
            value['protocol']['n_epochs'] = 1
        else:
            value['first_batch_audits'][0]['input_sha256'] = 'd' * 64
        write_json(source, value)
    monkeypatch.setattr(publisher, 'configure_environment', lambda **kw: pytest.fail('Auth before local validation'))
    with pytest.raises(ValueError):
        publisher.main(['--report-dir', str(saved_reports.report_dir), '--output', str(saved_reports.publish_dir), '--execute'])
    assert not saved_reports.publish_dir.exists()


@pytest.fixture
def fake_sdk(saved_reports, monkeypatch):
    pairs, sources = reporter.load_inputs(saved_reports.inputs)
    events, runs, artifacts = [], {}, {}

    class Artifact:
        def __init__(self, name, **kwargs):
            self.qualified_name = f'laymond1/cl-efficiency/{name}:v0'
            self.state, self.digest = 'COMMITTED', 'fixture-logical-digest'
            self.metadata = kwargs.get('metadata', {})
            self.files = {}
            self.manifest = SimpleNamespace(entries={})

        def add_bytes(self, name, content):
            self.files[name] = content
            self.manifest.entries[name] = SimpleNamespace(
                digest=base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode(), size=len(content), ref=None)

        def add_file(self, path, name):
            self.add_bytes(name, Path(path).read_bytes())

        def wait(self):
            events.append(('wait', self.qualified_name))

        def get_path(self, name):
            def download(root):
                destination = Path(root) / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(self.files[name])
                return str(destination)
            return SimpleNamespace(download=download)

    for method, sides in pairs.items():
        for side, local in sides.items():
            artifact = Artifact(method + '-' + side)
            source = sources[method][side]['artifacts']
            for name, pin in source['files'].items():
                artifact.add_file(pin['file'], name)
            uploaded = copy.deepcopy(local)
            uploaded.pop('wandb_logging_status')
            artifact.add_bytes(source['remote_record_file'], json.dumps(uploaded).encode())
            artifacts[artifact.qualified_name] = artifact
            prefix = 'counts' if side == 'reference' else 'native_counts'
            summary = {f'{prefix}/{key}': local['counts'][key] for key in ('observe_calls', 'optimizer_updates')}
            summary['status' if side == 'reference' else 'native_status'] = 'completed'
            if side == 'native':
                summary['acceptance_status'] = 'pass'
            for row in local['task_results']:
                summary.update({f'task{row["task"]}/{key}': row[key] for key in reporter.COUNT_KEYS})
            runs[reporter.run_path(local['wandb_url'])] = SimpleNamespace(state='finished', summary=summary, logged_artifacts=lambda artifact=artifact: [artifact])

    class Api:
        def __init__(self, **kwargs):
            events.append(('api', kwargs))

        def run(self, path):
            events.append(('read_run', path))
            return runs[path]

        def artifact(self, name):
            return artifacts[name]

        def flush(self):
            pass

    sdk = SimpleNamespace(Api=Api, Artifact=Artifact, run=None)
    def initialize(**kwargs):
        events.append(('init', kwargs))
        run = SimpleNamespace(state='running', summary={}, url='https://wandb.ai/laymond1/cl-efficiency/runs/finalreport')
        def logged(artifact):
            events.append(('upload', artifact.qualified_name))
            artifacts[artifact.qualified_name] = artifact
            return artifact
        run.log_artifact = logged
        runs['laymond1/cl-efficiency/finalreport'] = run
        sdk.run = run
        return run

    def finish(exit_code):
        events.append(('finish', exit_code))
        sdk.run.state = 'finished' if exit_code == 0 else 'failed'
        sdk.run = None
    sdk.init, sdk.finish = initialize, finish
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setattr(publisher, 'configure_environment', lambda **kw: {'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency'})
    monkeypatch.setattr(publisher, 'notify_slack', lambda message: events.append(('slack', message)) or 'sent')
    return SimpleNamespace(events=events, runs=runs, artifacts=artifacts, sdk=sdk)


def publish_arguments(saved_reports):
    return ['--report-dir', str(saved_reports.report_dir), '--output', str(saved_reports.publish_dir), '--execute']


def test_publish_verifies_eight_finished_remote_counts_artifacts_and_final_readback_before_slack(saved_reports, fake_sdk):
    assert publisher.main(publish_arguments(saved_reports)) == 0
    record = reporter.read_json(saved_reports.publish_dir / 'result.json')
    assert record['status'] == 'completed' and record['slack_completion_status'] == 'sent'
    proof = reporter.read_json(saved_reports.publish_dir / 'source_readback.json')
    assert proof['status'] == 'pass' and proof['run_count'] == len(proof['runs']) == 8
    assert all(row['full_checkpoints_uploaded'] is False for row in proof['runs'])
    final = reporter.read_json(saved_reports.publish_dir / 'publication_readback.json')
    assert final['status'] == 'pass' and final['summary_verified'] is True
    events = [event[0] for event in fake_sdk.events]
    assert events.index('init') > max(index for index, name in enumerate(events[:events.index('init')]) if name == 'read_run')
    assert events.index('slack') > events.index('finish')
    artifact = fake_sdk.artifacts[record['wandb_artifact']['name']]
    assert set(artifact.files) == {*reporter.REPORT_FILES, 'artifact_metadata.json', 'source_readback.json'}
    assert not any(name.endswith('.pt') for name in artifact.files)
    assert '논문 수치 재현·효율성 파일럿은 아닙니다' in fake_sdk.events[-1][1]


@pytest.mark.parametrize('mutation', ['running', 'counts', 'acceptance', 'missing_artifact', 'digest', 'semantics', 'checkpoint_metadata'])
def test_remote_incomplete_or_tampered_source_blocks_report_upload_and_slack(saved_reports, fake_sdk, mutation):
    run = fake_sdk.runs['laymond1/cl-efficiency/0native']
    artifact = run.logged_artifacts()[0]
    if mutation == 'running':
        run.state = 'running'
    elif mutation == 'counts':
        run.summary['task2/old_correct'] -= 1
    elif mutation == 'acceptance':
        run.summary['acceptance_status'] = 'review'
    elif mutation == 'missing_artifact':
        del artifact.manifest.entries['spec.json']
    elif mutation == 'digest':
        artifact.manifest.entries['task1_predictions.npz'].digest = 'wrong'
    else:
        remote = json.loads(artifact.files['result.json'])
        if mutation == 'semantics':
            remote['task_results'][0]['seen_correct'] -= 1
        else:
            remote['state_audits']['tasks'][0]['checkpoint_sha256'] = '0' * 64
        artifact.add_bytes('result.json', json.dumps(remote).encode())
    assert publisher.main(publish_arguments(saved_reports)) == 1
    assert not any(event[0] in ('init', 'upload', 'slack') for event in fake_sdk.events)
    assert reporter.read_json(saved_reports.publish_dir / 'result.json')['status'] == 'publication_failed'


def test_final_remote_readback_failure_blocks_slack_without_exposing_error_contents(saved_reports, fake_sdk, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('private credential content')
    monkeypatch.setattr(publisher, 'verify_publication', failed)
    assert publisher.main(publish_arguments(saved_reports)) == 1
    text = (saved_reports.publish_dir / 'result.json').read_text()
    assert 'private credential' not in text
    assert not any(event[0] == 'slack' for event in fake_sdk.events)
