"""Size-specific native gates and real CPU statistics; no training or services."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import hide_protocol
from scripts import hide_size_protocol
from scripts import run_hide_validation as runner
from tests.test_hide_execution import completed_fixture, reference_rows


def reference(name, initial_hash='initial'):
    info = hide_size_protocol.load_variant(name, verify_checkpoint=False)
    return {'status': 'completed', 'mode': 'validation', 'variant': name,
            'protocol': hide_protocol.protocol('validation', variant=name),
            'reference_commit': hide_protocol.REFERENCE_COMMIT,
            'checkpoint_sha256': info['checkpoint_sha256'],
            'canonical_pretrained_sha256': info['canonical_pretrained_sha256'],
            'initial_state_sha256': initial_hash, 'task_results': reference_rows()}


@pytest.mark.parametrize('name,dim,heads', [('small', 384, 6), ('tiny', 192, 3)])
@pytest.mark.parametrize('mode', ['validation', 'smoke', 'efficiency'])
def test_size_native_command_and_stream_counts(name, dim, heads, mode):
    settings = hide_protocol.protocol(mode, variant=name)
    info = hide_size_protocol.load_variant(name, verify_checkpoint=False)
    spec = {'variant': name, 'mode': mode, 'gpu': '1', 'model': 'hide-prompt',
            'pass': 'measurement', 'protocol': settings, **{key: info[key] for key in
                ('vit_size', 'feature_dim', 'num_heads', 'checkpoint_tag',
                 'checkpoint_file', 'checkpoint_sha256', 'canonical_pretrained_sha256')},
            'data_root': '/mnt/sdab1/share/data'}
    runner.validate_spec(spec)
    command = runner.native_command(spec, Path('/tmp/prepared-hide-size'))
    assert command[command.index('--hide_vit_size') + 1] == name
    assert command[command.index('--hide_checkpoint') + 1] == info['checkpoint_file']
    assert command[command.index('--batch_size') + 1] == '64'
    assert command[command.index('--stop_after') + 1] == str(settings['tasks'])
    assert (settings['feature_dim'], settings['num_heads']) == (dim, heads)
    record, recorder, model, _ = completed_fixture(mode)
    assert runner.completion_checks(record, recorder, model, settings)
    if mode == 'validation':
        assert recorder.steps == 11060 and recorder.training_examples == 700000
        assert recorder.evaluation_examples == 3000 and len(record['epoch_results']) == 140


@pytest.mark.parametrize('name', ['small', 'tiny'])
def test_size_dry_native_is_stdlib_only_and_never_executes(name):
    script = ('import sys; from scripts.run_hide_validation import main; '
              f'main(["--variant", "{name}"]); '
              'assert "torch" not in sys.modules; assert "wandb" not in sys.modules')
    result = subprocess.run([sys.executable, '-S', '-c', script], cwd=runner.ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'execution=False' in result.stdout and f'"variant": "{name}"' in result.stdout


@pytest.mark.parametrize('name', ['small', 'tiny'])
def test_unapproved_full_cl_rejected_before_credentials(name, monkeypatch):
    monkeypatch.setattr(runner, 'configure_environment', lambda: pytest.fail('Unexpected credential read'))
    with pytest.raises(ValueError, match='excludes full_cl'):
        runner.main(['--variant', name, '--mode', 'full_cl', '--execute'])


@pytest.mark.parametrize('name', ['small', 'tiny'])
def test_native_reference_requires_own_size_not_base_or_other_size(name):
    saved = reference(name)
    runner.validate_reference(saved, 'initial', variant=name)
    other = reference('tiny' if name == 'small' else 'small')
    with pytest.raises(ValueError, match='identical full protocol'):
        runner.validate_reference(other, 'initial', variant=name)
    saved['canonical_pretrained_sha256'] = hide_protocol.CANONICAL_PRETRAINED_SHA256
    with pytest.raises(ValueError, match='canonical pretrained'):
        runner.validate_reference(saved, 'initial', variant=name)


@pytest.fixture(params=['small', 'tiny'])
def native_fixture(request, tmp_path, monkeypatch):
    name = request.param
    info = hide_size_protocol.load_variant(name, verify_checkpoint=False)
    initial, checkpoint = tmp_path / 'initial.pt', tmp_path / 'pretrained.npz'
    initial.write_bytes(b'CPU metadata fixture, not model weights')
    checkpoint.write_bytes(name.encode())
    info.update(checkpoint_file=str(checkpoint), checkpoint_sha256=runner.file_hash(checkpoint))
    monkeypatch.setattr(runner, 'checkpoint_info', lambda variant=None: dict(info))
    monkeypatch.setattr(runner, 'validate_shared_data', lambda *args: None)
    saved = reference(name, runner.file_hash(initial))
    saved['checkpoint_sha256'] = info['checkpoint_sha256']
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(saved))
    options = SimpleNamespace(mode='validation', variant=name, initial_state=initial,
                              checkpoint=None, result_dir=tmp_path / 'native', output=tmp_path,
                              pass_kind='measurement', reference_summary=path, acceptance_summary=None)
    profile = {'gpu': '1', 'data_root': str(tmp_path), 'wandb_entity': 'fixture', 'wandb_project': 'cl-efficiency'}
    return name, info, options, profile


def test_spec_selects_size_checkpoint_default_and_freezes_immutable_plan(native_fixture):
    name, info, options, profile = native_fixture
    spec = runner.build_spec(options, profile)
    assert spec['variant'] == name and spec['checkpoint_file'] == info['checkpoint_file']
    assert spec['checkpoint_sha256'] == info['checkpoint_sha256']
    assert spec['canonical_pretrained_sha256'] == info['canonical_pretrained_sha256']
    assert all(source in spec['execution_source_sha256'] for source in hide_size_protocol.SOURCE_FILES)
    assert not any(source.endswith('execution_plan.json') for source in spec['execution_source_sha256'])
    assert not options.result_dir.exists()


def test_wrong_pretrained_override_cannot_substitute_base_weights(native_fixture, tmp_path, monkeypatch):
    _, _, options, profile = native_fixture
    changed = tmp_path / 'base.npz'
    changed.write_bytes(b'wrong backbone weights')
    options.checkpoint = changed
    monkeypatch.setattr(runner, 'validate_shared_data', lambda *args: pytest.fail('Data read before checkpoint validation'))
    with pytest.raises(ValueError, match='pinned local checkpoint'):
        runner.build_spec(options, profile)
    assert not options.result_dir.exists()


def test_variant_spec_cannot_claim_base_dimensions(native_fixture):
    _, _, options, profile = native_fixture
    spec = runner.build_spec(options, profile)
    for key, value in [('feature_dim', 768), ('num_heads', 12), ('vit_size', 'base')]:
        altered = deepcopy(spec)
        altered[key] = value
        with pytest.raises(ValueError, match='geometry'):
            runner.validate_spec(altered)


@pytest.mark.parametrize('dim', [192, 384, 768])
def test_real_statistics_accept_size_geometry_and_reject_future_or_nonfinite_values(dim):
    from models.hide_prompt_utils.statistics import ClassFeatureStatistics
    subject = ClassFeatureStatistics(100, dim)
    subject.valid[:20] = True
    subject.class_tasks[:10] = 0
    subject.class_tasks[10:20] = 1
    runner.audit_statistics(subject, 20, dim)
    if dim != 768:
        with pytest.raises(RuntimeError, match='statistics shapes'):
            runner.audit_statistics(subject, 20)
    subject.valid[20, 0] = True
    with pytest.raises(RuntimeError, match='seen-class'):
        runner.audit_statistics(subject, 20, dim)
    subject.valid[20, 0] = False
    subject.variances[0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='statistics shapes'):
        runner.audit_statistics(subject, 20, dim)
