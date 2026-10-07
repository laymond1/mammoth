"""Size-native execution guards and exact saved reference counts, without training."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_ease_validation as native


GEOMETRIES = {
    'small_fixed64': ('small', 384, 6, 64),
    'tiny_fixed64': ('tiny', 192, 3, 64),
    'small_relative32': ('small', 384, 6, 32),
    'tiny_relative16': ('tiny', 192, 3, 16),
}


@pytest.fixture(params=GEOMETRIES)
def size_case(request, tmp_path, monkeypatch):
    name = request.param
    size, dimension, heads, adapter = GEOMETRIES[name]
    checkpoint = tmp_path / f'{size}-shared.npz'
    checkpoint.write_bytes(f'{size} local fixture checkpoint'.encode())
    variant = {'name': name, 'vit_size': size, 'feature_dim': dimension,
               'num_heads': heads, 'adapter_size': adapter,
               'checkpoint_tag': f'vit_{size}_patch16_224.augreg_in21k_ft_in1k',
               'checkpoint_file': str(checkpoint.resolve()),
               'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'canonical_pretrained_sha256': ('1' if size == 'small' else '2') * 64}
    def load(selected, verify_checkpoint=False):
        if selected != name:
            raise ValueError('Unapproved EASE size variant')
        return dict(variant)
    monkeypatch.setattr(native, 'load_size_variant', load)
    policy = {'confirmed_by': 'user', 'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3.0}
    protocol = {'tasks': 2, 'seed': 1993, 'n_epochs_per_task': 20, 'batch_size': 48,
                'num_workers': 4, 'initial_variant': name, 'training_authorized': True,
                'accuracy_policy': policy}
    spec = {'stage': 'native_size_two_task_acceptance', 'tasks': 2, 'variant': name,
            'variant_config': dict(variant), 'protocol': protocol, 'data_root': str(tmp_path / 'data'),
            'output': str(tmp_path / 'native'), 'gpu': '1',
            **{key: value for key, value in variant.items() if key != 'name'}}
    output = tmp_path / 'official'
    output.mkdir()
    for task in (1, 2):
        labels = np.repeat(np.arange(task * 10), 100)
        predicted = labels.copy()
        if task == 1:
            predicted[:23] = (predicted[:23] + 1) % 10
        else:
            predicted[:31] = (predicted[:31] + 1) % 20
            predicted[1000:1017] = (predicted[1000:1017] + 1) % 20
        np.savez(output / f'task{task}_predictions.npz', predictions=predicted[:, None], labels=labels)
    reference = {'status': 'completed', 'stage': 'author_code_size_variant_reference',
                 'reference_kind': 'author-code-size-variant', 'variant': name,
                 'variant_config': dict(variant), 'feature_dim': dimension, 'adapter_size': adapter,
                 'checkpoint': variant['checkpoint_file'], 'checkpoint_sha256': variant['checkpoint_sha256'],
                 'checkpoint_tag': variant['checkpoint_tag'],
                 'canonical_pretrained_sha256': variant['canonical_pretrained_sha256'],
                 'initial_canonical_pretrained_sha256': variant['canonical_pretrained_sha256'],
                 'config': {'seed': 1993, 'init_epochs': 20, 'later_epochs': 20, 'batch_size': 48,
                            'init_cls': 10, 'increment': 10, 'shuffle': False},
                 'tasks': 2, 'output': str(output), 'wandb_url': 'https://example.invalid/reference'}
    return SimpleNamespace(name=name, variant=variant, spec=spec, policy=policy, reference=reference, output=output)


def test_each_geometry_uses_the_approved_shape_and_local_pretrained_flag(size_case):
    case = size_case
    native.validate_spec(case.spec)
    assert native.variant_config(case.spec) == case.variant
    command = native.native_command(case.spec, Path(case.spec['output']))
    get = lambda flag: command[command.index(flag) + 1]
    assert get('--vit_size') == case.variant['vit_size']
    assert int(get('--adapter_size')) == case.variant['adapter_size']
    assert get('--pretrained') == '0'
    assert get('--ease_checkpoint') == case.variant['checkpoint_file']
    assert (get('--n_epochs'), get('--batch_size'), get('--stop_after'), get('--seed')) == ('20', '48', '2', '1993')


def test_existing_base_command_remains_base64_and_does_not_add_pretrained_override(tmp_path):
    spec = {'stage': 'native_two_task_acceptance', 'checkpoint_file': '/shared/base.npz', 'data_root': '/shared/data'}
    command = native.native_command(spec, tmp_path)
    assert command[command.index('--vit_size') + 1] == 'base'
    assert command[command.index('--adapter_size') + 1] == '64'
    assert '--pretrained' not in command
    assert native.variant_config(spec)['feature_dim'] == 768


def test_actual_native_parser_retains_local_flag_and_geometry_without_model_construction(size_case, monkeypatch):
    import sys
    import timm
    import main as mammoth
    monkeypatch.setattr(timm, 'create_model', lambda *args, **kwargs: pytest.fail('Parser must not construct pretrained models'))
    monkeypatch.setattr(mammoth, 'initialize', lambda *args, **kwargs: pytest.fail('Parser must not initialize models or loaders'))
    command = native.native_command(size_case.spec, Path(size_case.spec['output']))
    monkeypatch.setattr(sys, 'argv', ['main.py', *command])
    args = native.prepare_native_args(mammoth.parse_args(cmd=command, verbose=False), size_case.spec)
    assert args.pretrained is False
    assert args.vit_size == size_case.variant['vit_size']
    assert args.adapter_size == size_case.variant['adapter_size']
    assert args.ease_checkpoint == size_case.variant['checkpoint_file']
    assert (args.n_epochs, args.batch_size, args.stop_after) == (20, 48, 2)


def test_preinitialization_local_checkpoint_guard_blocks_wrong_path(size_case):
    args = SimpleNamespace(ease_checkpoint='/other/weights.npz', pretrained=True)
    with pytest.raises(ValueError, match='pinned local checkpoint'):
        native.prepare_native_args(args, size_case.spec)
    assert args.pretrained is True


def test_existing_base_preinitialization_keeps_backbone_pretrained_flag():
    args = SimpleNamespace(pretrained=True)
    assert native.prepare_native_args(args, {'stage': 'native_two_task_acceptance'}) is args
    assert args.pretrained is True


def test_preinitialization_rejects_missing_pinned_local_file(size_case):
    Path(size_case.spec['checkpoint_file']).unlink()
    args = SimpleNamespace(ease_checkpoint=size_case.spec['checkpoint_file'], pretrained=True)
    with pytest.raises(ValueError, match='pinned local checkpoint'):
        native.prepare_native_args(args, size_case.spec)
    assert args.pretrained is True


@pytest.mark.parametrize('field,value', [('feature_dim', 768), ('adapter_size', 128), ('num_heads', 12),
                                        ('checkpoint_sha256', 'f' * 64), ('canonical_pretrained_sha256', 'f' * 64),
                                        ('checkpoint_tag', 'wrong-tag'), ('checkpoint_file', '/different/file.npz')])
def test_size_spec_rejects_wrong_geometry_or_pretrained_pin(size_case, field, value):
    size_case.spec[field] = value
    with pytest.raises(ValueError, match='approved variant'):
        native.validate_spec(size_case.spec)


def test_size_spec_rejects_metadata_from_another_variant(size_case):
    size_case.spec['variant_config']['name'] = 'another-variant'
    with pytest.raises(ValueError, match='configuration'):
        native.validate_spec(size_case.spec)


@pytest.mark.parametrize('key,value', [('seed', 42), ('n_epochs_per_task', 1), ('batch_size', 64),
                                     ('training_authorized', False), ('initial_variant', 'base_fixed64')])
def test_size_spec_rejects_wrong_learning_scope(size_case, key, value):
    size_case.spec['protocol'][key] = value
    with pytest.raises(ValueError, match='authorized size two-task'):
        native.validate_spec(size_case.spec)


def test_matching_reference_is_required_before_native_training(size_case):
    # The pinned author config has separate first/later schedules, no `epochs`.
    assert 'epochs' not in size_case.reference['config']
    native.validate_size_reference(size_case.reference, size_case.variant)


@pytest.mark.parametrize('key', ['init_epochs', 'later_epochs'])
def test_reference_requires_both_author_epoch_schedules(size_case, key):
    size_case.reference['config'][key] = 1
    with pytest.raises(ValueError, match='matched two-task learning protocol'):
        native.validate_size_reference(size_case.reference, size_case.variant)


def test_size_validation_rejects_a_different_physical_gpu(size_case):
    size_case.spec['gpu'] = '0'
    with pytest.raises(ValueError, match='authorized GPU1'):
        native.validate_spec(size_case.spec)


def test_variant_wandb_group_and_tags_match_reference_group(size_case):
    metadata = native.wandb_run_context(size_case.variant)
    assert metadata['group'] == f'ease-size-{size_case.name}-validation'
    assert size_case.name in metadata['tags']
    assert f'adapter-{size_case.variant["adapter_size"]}' in metadata['tags']
    assert native.wandb_run_context() == {'group': 'ease-base-validation', 'job_type': 'native-acceptance'}


@pytest.mark.parametrize('key,value', [('variant', 'base_fixed64'), ('feature_dim', 768), ('adapter_size', 128),
                                     ('checkpoint_sha256', 'f' * 64), ('canonical_pretrained_sha256', 'f' * 64),
                                     ('initial_canonical_pretrained_sha256', 'f' * 64), ('checkpoint', '/wrong.npz'),
                                     ('status', 'running'), ('stage', 'official_base_reference')])
def test_wrong_reference_identity_or_initial_weights_block_native(size_case, key, value):
    size_case.reference[key] = value
    with pytest.raises(ValueError):
        native.validate_size_reference(size_case.reference, size_case.variant)


def test_reference_policy_uses_variant_npz_counts_and_exact_thresholds(size_case):
    template = deepcopy(size_case.policy)
    policy, pins = native.reference_policy(size_case.reference, template)
    assert template == size_case.policy
    assert policy['unit'] == 'percentage_points'
    assert [(m['metric'], m['reference_correct'], m['reference_samples']) for m in policy['metrics']] == [
        ('task1_seen', 977, 1000), ('task2_seen', 1952, 2000), ('task2_old', 969, 1000), ('task2_new', 983, 1000)]
    assert policy['metrics'][0]['pass_min_accuracy_pct'] == 96.2
    assert policy['metrics'][1]['review_min_accuracy_pct'] == 94.6
    assert len(pins) == 2 and all(native.file_hash(Path(path)) == digest for path, digest in pins.items())
    # The former Base counts (996/1944/966/978) must never be inherited.
    assert [m['reference_correct'] for m in policy['metrics']] != [996, 1944, 966, 978]
    from scripts.ease_acceptance import REQUIRED_STRUCTURAL_CHECKS, evaluate_acceptance
    rows = [{'task': 1, 'seen_correct': 977, 'seen_samples': 1000, 'old_correct': 0, 'old_samples': 0, 'new_correct': 977, 'new_samples': 1000},
            {'task': 2, 'seen_correct': 1952, 'seen_samples': 2000, 'old_correct': 969, 'old_samples': 1000, 'new_correct': 983, 'new_samples': 1000}]
    assert evaluate_acceptance(rows, policy, structural_checks=dict.fromkeys(REQUIRED_STRUCTURAL_CHECKS, True))['status'] == 'pass'


def test_mutated_reference_predictions_change_count_and_file_pins(size_case):
    policy, old_pins = native.reference_policy(size_case.reference, size_case.policy)
    labels = np.repeat(np.arange(10), 100)
    np.savez(size_case.output / 'task1_predictions.npz', predictions=labels[:, None], labels=labels)
    changed, new_pins = native.reference_policy(size_case.reference, size_case.policy)
    assert old_pins != new_pins
    assert policy['metrics'][0]['reference_correct'] == 977
    assert changed['metrics'][0]['reference_correct'] == 1000


def test_incomplete_reference_archive_blocks_policy(size_case):
    np.savez(size_case.output / 'task1_predictions.npz', predictions=np.zeros((999, 1), dtype=int), labels=np.zeros(999, dtype=int))
    with pytest.raises(ValueError, match='full evaluation set'):
        native.reference_policy(size_case.reference, size_case.policy)


def test_variant_cli_retains_reference_and_size_output_directory(monkeypatch, size_case):
    def capture(options, profile):
        assert options.variant == size_case.name and options.reference_summary == Path('reference.json')
        assert options.output == native.ROOT / 'benchmarks/cl_efficiency/ease_sizes/runs'
        return 0
    monkeypatch.setattr(native, 'configure_environment', lambda: {})
    monkeypatch.setattr(native, 'execute', capture)
    assert native.main(['--execute', '--variant', size_case.name, '--reference-summary', 'reference.json']) == 0


def test_execute_rejects_missing_variant_reference_before_notifications_or_launch(size_case, monkeypatch, tmp_path):
    from scripts import ease_size_protocol
    monkeypatch.setattr(ease_size_protocol, 'load_plan', lambda: {'validation': size_case.spec['protocol']})
    monkeypatch.setattr(native, 'notify_slack', lambda *args: pytest.fail('Invalid preparation must not notify'))
    options = SimpleNamespace(variant=size_case.name, reference_summary=None, output=tmp_path, timeout=1)
    with pytest.raises(ValueError, match='reference-summary'):
        native.execute(options, {})


def test_execute_builds_matching_dynamic_spec_before_mocked_worker(size_case, monkeypatch, tmp_path):
    from scripts import ease_size_protocol
    monkeypatch.setattr(ease_size_protocol, 'load_plan', lambda: {'validation': size_case.spec['protocol']})
    wrapper = tmp_path / 'reference-summary.json'
    wrapper.write_text(json.dumps({'official_reference': size_case.reference}))
    notifications = []
    monkeypatch.setattr(native, 'notify_slack', lambda message: notifications.append(message) or 'mocked')
    monkeypatch.setattr(native.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='fixture-commit\n'))
    captured = []
    class FinishedWorker:
        returncode = 0

        def __init__(self, command, **kwargs):
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '1'
            assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
            spec = json.loads(Path(command[-1]).read_text())
            captured.append(spec)
            native.write_json(Path(spec['output']) / 'metrics.json',
                              {'status': 'completed', 'acceptance_status': 'pass', 'wandb_logging_status': 'mocked'})

        def poll(self):
            return 0

    monkeypatch.setattr(native.subprocess, 'Popen', FinishedWorker)
    options = SimpleNamespace(variant=size_case.name, reference_summary=wrapper, output=tmp_path / 'native-results', timeout=1)
    profile = {'gpu': '1', 'data_root': '/shared/data', 'wandb_entity': 'fixture', 'wandb_project': 'fixture'}
    assert native.execute(options, profile) == 0
    spec = captured[0]
    assert spec['stage'] == 'native_size_two_task_acceptance'
    assert spec['variant'] == size_case.name and spec['feature_dim'] == size_case.variant['feature_dim']
    assert spec['adapter_size'] == size_case.variant['adapter_size']
    assert spec['canonical_pretrained_sha256'] == size_case.variant['canonical_pretrained_sha256']
    assert [metric['reference_correct'] for metric in spec['protocol']['accuracy_policy']['metrics']] == [977, 1952, 969, 983]
    assert len(spec['reference_predictions_sha256']) == 2
    assert 'scripts/ease_size_protocol.py' in spec['execution_source_sha256']
    assert notifications and all('Base/adapter64' not in message for message in notifications)
