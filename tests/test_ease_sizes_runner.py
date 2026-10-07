"""Stage ordering, reuse and failure gates for the authorized four-size pipeline."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import run_ease_sizes as pipeline
from scripts import run_ease_reference, run_ease_validation, run_ease_efficiency


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    variants = {name: {'name': name, 'vit_size': size, 'feature_dim': d, 'num_heads': h,
                      'adapter_size': r, 'checkpoint_sha256': 'a' * 64,
                      'canonical_pretrained_sha256': 'b' * 64}
                for name, (size, d, h, r) in pipeline.VARIANTS.items()}
    base = tmp_path / 'historical-base.json'
    base.write_text('{"historical":"immutable"}')
    plan = {'data_root': str(tmp_path / 'data'), 'cache_root': str(tmp_path / 'cache'),
            'base_reference': {'file': base.name, 'sha256': pipeline.file_hash(base)}}
    profile = {'gpu': '1', **{key: plan[key] for key in ('data_root', 'cache_root')}}
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'load_plan', lambda: plan)
    monkeypatch.setattr(pipeline, 'load_variant', lambda name: variants[name])
    monkeypatch.setattr(pipeline, 'configure_environment', lambda: profile)
    paths = ['scripts/run_ease_sizes.py', 'scripts/ease_size_protocol.py', 'scripts/run_ease_reference.py',
             'scripts/ease_reference_utils.py', 'scripts/run_ease_validation.py', 'scripts/run_ease_efficiency.py',
             'scripts/benchmark_cl.py', 'scripts/cl_efficiency.py', 'models/ease.py', 'backbone/ease_vit.py',
             'main.py', 'utils/training.py', 'uv.lock', 'models/ease_utils/__init__.py',
             'models/config/ease.yaml', 'datasets/configs/seq-cifar100-224/ease.yaml']
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic pinned source')
    calls = []
    state = SimpleNamespace(variants=variants, calls=calls, output=tmp_path / 'runs', plan=plan,
                            reference_error=False, native_acceptance='pass', efficiency_error=False)

    def reference(options, profile):
        key = options.variant
        calls.append((key, 'reference'))
        directory = options.output / 'matched-reference'
        directory.mkdir()
        record = {'status': 'completed', 'wandb_logging_status': 'uploaded', 'variant': key,
                  'checkpoint_sha256': variants[key]['checkpoint_sha256'],
                  'initial_canonical_pretrained_sha256': variants[key]['canonical_pretrained_sha256'],
                  'wandb_url': 'https://example.test/reference'}
        (directory / 'metrics.json').write_text(json.dumps(record))
        return 1 if state.reference_error else 0

    def native(options, profile):
        key = options.variant
        calls.append((key, 'native'))
        reference = json.loads(options.reference_summary.read_text())
        assert reference['variant'] == key
        directory = options.output / 'native'
        directory.mkdir()
        record = {'status': 'completed', 'wandb_logging_status': 'uploaded', 'variant': key,
                  'checkpoint_sha256': variants[key]['checkpoint_sha256'],
                  'acceptance_status': state.native_acceptance, 'acceptance': {'status': state.native_acceptance},
                  'wandb_url': 'https://example.test/native'}
        (directory / 'metrics.json').write_text(json.dumps(record))
        return 0

    def efficiency(argv):
        key = argv[argv.index('--variant') + 1]
        calls.append((key, 'efficiency'))
        acceptance = json.loads(Path(argv[argv.index('--acceptance-summary') + 1]).read_text())
        assert acceptance['variant'] == key and acceptance['native_validation']['acceptance_status'] == 'pass'
        parent = Path(argv[argv.index('--output') + 1])
        directory = parent / 'efficiency'
        directory.mkdir()
        (directory / 'execution.json').write_text(json.dumps({'status': 'completed', 'summary': {'status': 'pass'}}))
        return 1 if state.efficiency_error else 0

    monkeypatch.setattr(run_ease_reference, 'execute', reference)
    monkeypatch.setattr(run_ease_validation, 'execute', native)
    monkeypatch.setattr(run_ease_efficiency, 'main', efficiency)
    return state


def options(prepared):
    return SimpleNamespace(output=prepared.output, reference_source=Path('/tmp/synthetic-source'), timeout=60.)


def execution(prepared):
    return json.loads(next(prepared.output.glob('*/execution.json')).read_text())


def test_four_variants_validate_before_cost_and_reuse_base(prepared):
    assert pipeline.execute(options(prepared)) == 0
    assert prepared.calls == [(name, stage) for name in pipeline.VARIANTS for stage in ('reference', 'native', 'efficiency')]
    result = execution(prepared)
    assert result['status'] == 'completed' and result['base_reused'] is True
    assert result['completed_counts'] == {'reference': 4, 'native_validation': 4, 'measurement': 12, 'flops': 4, 'total': 24}
    assert len(result['variants']) == 4 and len(result['invocations']) == 12
    assert all(value['status'] == 'completed' for value in result['variants'].values())


@pytest.mark.parametrize('acceptance', ['review', 'fail', 'not_evaluated'])
def test_review_or_failure_stops_cost_and_preserves_reference_native(prepared, acceptance):
    prepared.native_acceptance = acceptance
    with pytest.raises(RuntimeError, match='Resolve'):
        pipeline.execute(options(prepared))
    assert prepared.calls == [('small_fixed64', 'reference'), ('small_fixed64', 'native')]
    result = execution(prepared)
    assert result['status'] == 'failed'
    assert result['variants']['small_fixed64']['status'] == 'acceptance_blocked'
    assert Path(result['variants']['small_fixed64']['reference_summary']).is_file()


def test_reference_failure_preserves_partial_directory_and_stops(prepared):
    prepared.reference_error = True
    with pytest.raises(RuntimeError, match='invocation failed'):
        pipeline.execute(options(prepared))
    result = execution(prepared)
    assert result['status'] == 'failed' and len(prepared.calls) == 1
    assert result['invocations'][0]['status'] == 'failed'
    assert Path(result['invocations'][0]['run_directories'][0], 'metrics.json').is_file()


def test_cost_failure_cannot_start_next_variant(prepared):
    prepared.efficiency_error = True
    with pytest.raises(RuntimeError, match='efficiency execution failed'):
        pipeline.execute(options(prepared))
    result = execution(prepared)
    assert result['status'] == 'failed' and len(prepared.calls) == 3
    assert result['invocations'][-1]['status'] == 'failed'


def test_default_scope_does_not_read_credentials_or_execute(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('Default preparation must not execute or read credentials')
    monkeypatch.setattr(pipeline, 'configure_environment', forbidden)
    monkeypatch.setattr(pipeline, 'execute', forbidden)
    assert pipeline.main([]) == 0
    assert 'execution=False' in capsys.readouterr().out
