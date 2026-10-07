"""Saved-file size report audits, including corrupted predictions and gates."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import hide_size_protocol
from scripts import report_hide_sizes as report
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS, counts_from_predictions
from scripts.run_hide_efficiency import METHOD_COUNTS, PHASE_KEYS, PROTOCOL_COUNTS, RESOURCE_KEYS


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def saved_inputs(tmp_path, monkeypatch):
    (tmp_path / 'main.py').write_text('public source fixture\n')
    sources = {'main.py': report.file_hash(tmp_path / 'main.py')}
    infos = {}
    for size, dimension, heads in (('base', 768, 12), ('small', 384, 6), ('tiny', 192, 3)):
        checkpoint = tmp_path / size / 'shared.npz'
        initial = tmp_path / size / 'initial.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(('public weight fixture ' + size).encode())
        initial.write_bytes(('public initialization fixture ' + size).encode())
        infos[size] = {'name': size, 'vit_size': size, 'feature_dim': dimension, 'num_heads': heads,
                       'checkpoint_tag': f'vit_{size}_patch16_224.augreg_in21k_ft_in1k',
                       'checkpoint_file': str(checkpoint), 'checkpoint_sha256': report.file_hash(checkpoint),
                       'canonical_pretrained_sha256': digest('canonical-' + size), 'initial': initial}
    monkeypatch.setattr(report, 'variant_info', lambda size: copy.deepcopy(infos[size]))
    monkeypatch.setattr(hide_size_protocol, 'load_variant', lambda size, verify_checkpoint=True: copy.deepcopy(infos[size]))
    # Historical source commits are read-only git blobs in production. A source
    # fixture supplies the blob bytes without creating commits in unit tests.
    monkeypatch.setattr(report, 'git_blob_hash', lambda root, commit, name: report.file_hash(Path(root) / name))

    def record(size, side, mode='validation', wall=60.):
        info = infos[size]
        tasks = 2 if mode == 'validation' else 1
        location = tmp_path / size / side
        location.mkdir()
        protocol = report.expected_protocol(mode, size)
        value = {'status': 'completed', 'model': 'hide-prompt', 'mode': mode,
                 'pass': 'measurement', 'protocol': protocol, 'output': str(location), 'gpu': '1',
                 'reference_commit': report.REFERENCE_COMMIT, 'git_commit': 'a' * 40,
                 'checkpoint_file': info['checkpoint_file'], 'checkpoint_sha256': info['checkpoint_sha256'],
                 'canonical_pretrained_sha256': info['canonical_pretrained_sha256'],
                 'initial_state': str(info['initial']), 'initial_state_sha256': report.file_hash(info['initial']),
                 'initial_learned_sha256': {branch: digest(size + branch) for branch in ('tii', 'prompt')},
                 'initial_canonical_pretrained_sha256': {branch: info['canonical_pretrained_sha256'] for branch in ('tii', 'prompt')},
                 'execution_source_sha256': sources.copy(), 'precision': report.FULL_CL_PRECISION.copy(),
                 'wandb_logging_status': 'uploaded', 'wandb_artifact': {'state': 'COMMITTED'},
                 'wandb_url': 'https://wandb.ai/public/fixture/runs/' + size + '-' + side,
                 'acceptance_status': 'pass' if mode == 'validation' else 'not_applicable',
                 'structural_checks': dict.fromkeys(REQUIRED_STRUCTURAL_CHECKS, True),
                 'task_results': [], 'epoch_results': [], 'first_batch_audits': [],
                 'wandb_artifact_files': {}, 'data_snapshots': {}, 'sensor_errors': [],
                 'counts': METHOD_COUNTS.copy(), 'metrics': {key: 10. for key in RESOURCE_KEYS},
                 'phases': {phase: {'wall_s': 1., 'gpu_board_energy_j': 2., 'gpu_energy_coverage_s': 1.} for phase in PHASE_KEYS},
                 'alignment_history': [{'branch': branch, 'task_id': 0, 'status': 'not_applicable', 'steps': 0} for branch in ('tii', 'prompt')],
                 'model_snapshots': {'task1_tii_train_epoch1': {'parameters_total': dimension * 100,
                      'parameters_trainable': dimension * 20, 'parameters_in_optimizer': dimension * 20,
                      'parameter_storage_bytes': dimension * 400, 'registered_buffer_storage_bytes': 1000,
                      'optimizer_state_storage_bytes': 0}}}
        if size != 'base':
            value.update(variant=size, vit_size=size, feature_dim=info['feature_dim'], num_heads=info['num_heads'], checkpoint_tag=info['checkpoint_tag'])
        value['metrics'].update(process_wall_s=wall, gpu_board_energy_wh=wall / 10,
                                gpu_energy_coverage_fraction=1., **PROTOCOL_COUNTS)
        if mode == 'validation':
            value['metrics'].update(tasks_completed=2, outer_observe_steps=11060,
                                    outer_training_stream_examples=700000, evaluation_examples=3000)
        for task in range(1, tasks + 1):
            labels = np.repeat(np.arange(task * 10, dtype=np.int64), 100)
            ids = np.arange(task * 1000, dtype=np.int64)
            predictions = labels[:, None].repeat(1 if side == 'reference' else 5, axis=1)
            npz = location / f'task{task}_predictions.npz'
            np.savez(npz, predictions=predictions, labels=labels, global_ids=ids)
            value['wandb_artifact_files'][npz.name] = {'local_file': npz.name, 'sha256': report.file_hash(npz)}
            counts = counts_from_predictions(labels.tolist(), labels.tolist(), past_classes=(task - 1) * 10, seen_classes=task * 10)
            value['task_results'].append({'task': task, **counts})
            for split in ('train', 'test'):
                classes = range((task - 1) * 10, task * 10) if split == 'train' else range(task * 10)
                value['data_snapshots'][f'task{task}_{split}'] = {
                    'n_samples': 5000 if split == 'train' else task * 1000,
                    'class_counts': {str(label): 500 if split == 'train' else 100 for label in classes},
                    'samples_ordered_sha256': digest(f'ordered-{task}-{split}'),
                    'samples_unordered_sha256': digest(f'unordered-{task}-{split}'),
                    'global_ids_ordered_sha256': digest('train-ids') if split == 'train' else hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                    'transform_repr_sha256': digest(protocol['dataset_config'] + split)}
            for phase, count in (('tii_train', protocol['tii_epochs']), ('prompt_train', protocol['prompt_epochs'])):
                seed = 1993 + (task - 1) * 100000 + (1000 if phase == 'prompt_train' else 0)
                value['first_batch_audits'].append({'task': task, 'phase': phase, 'seed': seed, 'samples': 64,
                    'input_sha256': digest(f'{protocol["dataset_config"]}-{task}-{phase}-inputs'),
                    'labels_sha256': digest(f'{task}-{phase}-labels')})
                for epoch in range(1, count + 1):
                    value['epoch_results'].append({'task': task, 'phase': phase, 'epoch': epoch,
                                                   'steps': 79, 'samples': 5000, 'seed': seed + epoch - 1})
        return value

    inputs = {}
    for size in ('base', 'small', 'tiny'):
        reference = record(size, 'reference')
        reference_path = Path(reference['output']) / 'result.json'
        save(reference_path, reference)
        native = record(size, 'native')
        native.update(reference_summary=str(reference_path), reference_summary_sha256=report.file_hash(reference_path))
        native_path = Path(native['output']) / 'result.json'
        save(native_path, native)
        measurements = []
        for index in range(3):
            measurement = record(size, f'measurement{index + 1}', 'efficiency', 50. + index * 10)
            measurement.update(acceptance_summary=str(native_path), acceptance_summary_sha256=report.file_hash(native_path))
            measurements.append(measurement)
        flops = record(size, 'flops', 'efficiency', 900.)
        flops.update(acceptance_summary=str(native_path), acceptance_summary_sha256=report.file_hash(native_path),
                     **{'pass': 'flops', 'flops': {'supported_ops_total': 1e12}})
        settings = report.expected_protocol('efficiency', size)
        settings.update(outer_epochs_per_task=2, measurement_repeats=3, flops_runs=1, ca_runs_task1=False)
        execution = {'status': 'completed', 'model': 'hide-prompt', 'protocol': settings,
                     'acceptance_summary': str(native_path), 'acceptance_summary_sha256': report.file_hash(native_path),
                     'checkpoint_sha256': native['checkpoint_sha256'], 'initial_state_sha256': native['initial_state_sha256'],
                     'measurements': measurements, 'flops': flops, 'summary': {'status': 'pass'}}
        if size != 'base':
            execution['variant'] = size
            execution['canonical_pretrained_sha256'] = native['canonical_pretrained_sha256']
        efficiency_path = tmp_path / size / 'execution.json'
        save(efficiency_path, execution)
        inputs[size + '_efficiency'] = efficiency_path
        if size != 'base':
            inputs[size + '_reference'], inputs[size + '_native'] = reference_path, native_path
    return SimpleNamespace(inputs=inputs, root=tmp_path, infos=infos)


def test_build_recounts_and_excludes_flops_costs_with_deterministic_payload(saved_inputs):
    result = report.build_report(saved_inputs.inputs)
    repeated = report.build_report(saved_inputs.inputs)
    assert {key: value for key, value in result.items() if key != 'generated_at'} == {key: value for key, value in repeated.items() if key != 'generated_at'}
    assert result['stage'] == 'hide_sizes' and result['full10_executed'] is False
    assert set(result['validation']) == {'small', 'tiny'}
    assert set(result['sources']) == set(report.INPUT_KEYS)
    methods = result['efficiency']['methods']
    assert [entry['variant'] for entry in methods] == ['base', 'small', 'tiny']
    for entry in methods:
        assert entry['resources']['process_wall_s']['values'] == [50., 60., 70.]
        assert entry['resources']['process_wall_s']['mean'] == 60.
        assert entry['resources']['process_wall_s']['std_sample'] == 10.
        assert entry['accuracy_pct']['values'] == [100., 100., 100.]
        assert entry['alignment_status'] == 'not_applicable'
    assert result['efficiency']['compatibility']['observations'] == 12
    assert any(value['verification'] == 'immutable_git_blobs' for value in result['provenance']['execution_sources'].values())
    assert result['validation']['small']['prediction_recount']['native'][0]['prediction_columns'] == 5


@pytest.mark.parametrize('mutation', ['mixed_variant', 'weight', 'initialization', 'precision', 'transform',
                                     'first_batch', 'missing_snapshot', 'threads', 'source', 'coverage',
                                     'sensor', 'incomplete', 'alignment', 'logging', 'acceptance_pin'])
def test_bad_pilot_evidence_blocks_report(saved_inputs, mutation):
    path = saved_inputs.inputs['small_efficiency']
    value = report.read_json(path)
    row = value['measurements'][0]
    if mutation == 'mixed_variant': row['variant'] = 'tiny'
    elif mutation == 'weight': row['checkpoint_sha256'] = 'b' * 64
    elif mutation == 'initialization': row['initial_learned_sha256']['tii'] = 'b' * 64
    elif mutation == 'precision': row['precision']['autocast'] = True
    elif mutation == 'transform': row['data_snapshots']['task1_train']['transform_repr_sha256'] = 'b' * 64
    elif mutation == 'first_batch': row['first_batch_audits'][0]['input_sha256'] = 'b' * 64
    elif mutation == 'missing_snapshot': row['data_snapshots'].pop('task1_test')
    elif mutation == 'threads': row['torch_cpu_threads'] = 4
    elif mutation == 'source': (saved_inputs.root / 'main.py').write_text('changed source\n')
    elif mutation == 'coverage': row['metrics']['gpu_energy_coverage_fraction'] = .98
    elif mutation == 'sensor': row['sensor_errors'] = ['fixture sensor error']
    elif mutation == 'incomplete': value['measurements'].pop()
    elif mutation == 'alignment': row['alignment_history'][0]['status'] = 'completed'
    elif mutation == 'logging': row['wandb_artifact']['state'] = 'PENDING'
    elif mutation == 'acceptance_pin': row['acceptance_summary_sha256'] = 'b' * 64
    save(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        report.build_report(saved_inputs.inputs)


@pytest.mark.parametrize('mutation', ['bytes', 'counts', 'future', 'duplicates', 'sample_ids', 'missing_class'])
def test_prediction_corruption_and_misreported_accuracy_are_rejected(saved_inputs, mutation):
    path = saved_inputs.inputs['small_native']
    value = report.read_json(path)
    npz = Path(value['output']) / 'task1_predictions.npz'
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if mutation == 'counts': value['task_results'][0]['seen_correct'] = 999
    elif mutation == 'future': arrays['predictions'][0, 0] = 10
    elif mutation == 'duplicates': arrays['global_ids'][1] = arrays['global_ids'][0]
    elif mutation == 'sample_ids': arrays['global_ids'] += 1
    elif mutation == 'missing_class': arrays['labels'][arrays['labels'] == 9] = 8
    else: arrays['predictions'][0, 0] = 1
    np.savez(npz, **arrays)
    if mutation != 'bytes':
        value['wandb_artifact_files'][npz.name]['sha256'] = report.file_hash(npz)
    save(path, value)
    with pytest.raises(ValueError):
        report.build_report(saved_inputs.inputs)


@pytest.mark.parametrize('mistakes,status', [(15, 'pass'), (16, 'review'), (31, 'fail')])
def test_exact_boundary_uses_recounted_same_size_author_accuracy(saved_inputs, mistakes, status):
    path = saved_inputs.inputs['small_native']
    native = report.read_json(path)
    reference_path = saved_inputs.inputs['small_reference']
    reference = report.read_json(reference_path)
    npz = Path(native['output']) / 'task1_predictions.npz'
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['predictions'][:mistakes, 0] = (arrays['labels'][:mistakes] + 1) % 10
    np.savez(npz, **arrays)
    native['wandb_artifact_files'][npz.name]['sha256'] = report.file_hash(npz)
    native['task_results'][0]['seen_correct'] = native['task_results'][0]['new_correct'] = 1000 - mistakes
    native['acceptance_status'] = status
    if status == 'pass':
        result = report.full_validation(reference, native, 'small', reference_path, report.Evidence())
        assert result['accuracy_metrics'][0]['reference_minus_native_pp'] == 1.5
    else:
        with pytest.raises(ValueError, match='correctness or first-batch'):
            report.full_validation(reference, native, 'small', reference_path, report.Evidence())


def test_report_renders_five_artifacts_and_publisher_can_recompute_csv(saved_inputs, tmp_path):
    value = report.build_report(saved_inputs.inputs)
    output = tmp_path / 'rendered'
    report.write_report(value, output)
    assert {path.name for path in output.iterdir()} == set(report.REPORT_FILES)
    with (output / 'results.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == report.csv_fields(value)
        actual = list(reader)
    expected = [{key: '' if row.get(key) is None else json.dumps(row[key], sort_keys=True)
                 if isinstance(row[key], (dict, list)) else str(row[key])
                 for key in report.csv_fields(value)} for row in report.csv_rows(value)]
    assert actual == expected
    assert (output / 'efficiency_comparison.png').stat().st_size > 1000
    assert (output / 'efficiency_comparison.pdf').read_bytes().startswith(b'%PDF')
    markdown = (output / 'report.md').read_text()
    assert '첫 태스크 CA는 해당 없음' in markdown and '1.5%p' in markdown


def test_cli_help_requires_no_ml_imports_or_execution():
    result = subprocess.run([sys.executable, '-S', str(Path(report.__file__)), '--help'], capture_output=True, text=True, check=True)
    assert '--small-reference' in result.stdout and '--base-efficiency' in result.stdout


def test_actual_historical_base_is_verified_against_immutable_source_and_predictions():
    worktree = report.ROOT.parent / 'hide-prompt'
    path = worktree / 'benchmarks/cl_efficiency/hide_prompt/runs/efficiency-pilot-b64-20261007/execution.json'
    if not path.is_file():
        pytest.skip('Preserved historical Base worktree is unavailable')
    raw = report.read_json(path)
    native = report.read_json(raw['acceptance_summary'])
    reference_path = Path(native['reference_summary'])
    reference = report.read_json(reference_path)
    evidence = report.Evidence()
    assert report.full_validation(reference, native, 'base', reference_path, evidence)['report_gate'] == 'pass'
    summary, measurements, execution = report.load_efficiency(path)
    for row in [*measurements, execution['flops']]:
        report.record_pins(row, 'base', evidence)
        report.prediction_audit(row, 1, evidence)
    assert report.common_pilot_inputs([*measurements, execution['flops']])['status'] == 'pass'
    assert summary['measurement_count'] == 3
    assert all(entry['verification'] == 'immutable_git_blobs' for entry in evidence.sources.values())
