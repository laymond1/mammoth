"""Keep partial completion and failed SD evidence explicit in publication."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import publish_peft_partial as publisher
from scripts.cl_efficiency import write_json
from scripts.run_peft_efficiency import file_hash


def test_partial_summary_does_not_claim_all_requested_methods_passed():
    report = {'sources': [{}] * 16, 'method_count': 8, 'new_measurement_runs': 12,
              'new_flops_runs': 4, 'publication_gate': 'pass_for_accepted_subset', 'methods': {}}
    summary = publisher.scalar_summary(report)
    assert summary['accepted_source_run_count'] == 16
    assert summary['all_requested_methods_passed'] is False
    assert summary['sd-lora/accepted'] is False
    assert summary['sd-lora/observed_accuracy_pct'] == 10.
    assert summary['quarantined_workers'] == 1 and summary['deferred_workers'] == 3
    assert summary['publication_gate'] != 'pass'


@pytest.mark.parametrize('actual,expected,matches', [
    (95.4, 95.39999999999999, True),
    (31.6064549240012, 31.606454924001202, True),
    (31.6065, 31.606454924001202, False),
    (79.00000000001, 79, False),
    (True, 1, False),
    (0, False, False),
    (False, False, True),
    (float('nan'), 2.94, False),
    (None, 2.94, False),
])
def test_json_float_rounding_never_relaxes_integer_counts_or_boolean_gates(actual, expected, matches):
    assert publisher.summary_value_matches(actual, expected) is matches


def test_report_summary_rounding_still_requires_an_exact_artifact_manifest(tmp_path):
    path = tmp_path / 'results.json'
    path.write_bytes(b'report')
    files = publisher.pins([path])
    run = SimpleNamespace(state='finished', summary={'accuracy': 95.4, 'accepted_workers': 16})
    artifact = SimpleNamespace(state='COMMITTED', qualified_name='test/report:v0', digest='digest',
                               manifest=SimpleNamespace(entries={}))
    api = SimpleNamespace(flush=lambda: None, run=lambda key: run, artifact=lambda key: artifact)
    with pytest.raises(ValueError, match='required evidence'):
        publisher.verify_report_publication(api, 'https://wandb.ai/laymond1/cl-efficiency/runs/id', artifact.qualified_name,
                                            files, {'accuracy': 95.39999999999999, 'accepted_workers': 16})
    artifact.manifest.entries['results.json'] = SimpleNamespace(digest='wrong', size=6, ref=None)
    with pytest.raises(ValueError, match='digest or size'):
        publisher.verify_report_publication(api, 'https://wandb.ai/laymond1/cl-efficiency/runs/id', artifact.qualified_name,
                                            files, {'accuracy': 95.39999999999999, 'accepted_workers': 16})


@pytest.fixture
def diagnostic(tmp_path):
    local = {'wandb_url': 'https://wandb.ai/laymond1/cl-efficiency/runs/diagnostic',
             'diagnosis': 'author_reproduces_native_feature_collapse',
             'counts': {'observe_calls': 79, 'optimizer_updates': 79},
             'accuracy': {'mean_accuracy_pct': 10.}}
    path = tmp_path / 'result.json'
    write_json(path, local)
    summary = {'diagnostic_status': 'completed', 'measurement_aggregate_eligible': False,
               'diagnosis': local['diagnosis'], 'counts/observe_calls': 79,
               'counts/optimizer_updates': 79, 'accuracy/mean_accuracy_pct': 10.}
    return {'file': str(path), 'sha256': file_hash(path)}, summary


@pytest.mark.parametrize('change', ['unfinished', 'aggregate', 'count', 'accuracy', 'diagnosis'])
def test_bad_diagnostic_cannot_enter_remote_publication(diagnostic, tmp_path, change):
    source, original = diagnostic
    summary = deepcopy(original)
    state = 'finished'
    if change == 'unfinished':
        state = 'running'
    elif change == 'aggregate':
        summary['measurement_aggregate_eligible'] = True
    elif change == 'count':
        summary['counts/observe_calls'] = True
    elif change == 'accuracy':
        summary['accuracy/mean_accuracy_pct'] = 99.
    else:
        summary['diagnosis'] = 'different'
    run = SimpleNamespace(state=state, summary=summary,
                          logged_artifacts=lambda: pytest.fail('Read artifact before validating diagnostic'))
    api = SimpleNamespace(run=lambda path: run)
    with pytest.raises(ValueError):
        publisher.verify_diagnostic(api, source, tmp_path / 'readback')


def test_changed_diagnostic_rejected_before_remote_api(diagnostic, tmp_path):
    source, _ = diagnostic
    source['sha256'] = 'changed'
    api = SimpleNamespace(run=lambda path: pytest.fail('Remote call before checking local pin'))
    with pytest.raises(ValueError, match='changed'):
        publisher.verify_diagnostic(api, source, tmp_path / 'readback')


def test_committed_small_diagnostic_manifest_and_semantics_match(diagnostic, tmp_path):
    from scripts.publish_low_rank_results import file_md5
    source, summary = diagnostic
    local = publisher.read_json(source['file'])
    required = ('spec.json', 'comparison.json', 'task1_predictions.npz', 'artifact_manifest.json')
    for name in required:
        (tmp_path / name).write_bytes(b'public evidence')
    write_json(tmp_path / 'artifact_manifest.json', {'full_checkpoints_uploaded': False,
        'files': {'result.json': {'sha256': file_hash(source['file']), 'bytes': Path(source['file']).stat().st_size}}})
    entries = {name: SimpleNamespace(digest=file_md5(tmp_path / name)[1],
                                    size=(tmp_path / name).stat().st_size, ref=None)
               for name in (*required, 'result.json')}
    artifact = SimpleNamespace(state='COMMITTED', qualified_name='test/diagnostic:v0', digest='manifest',
                               manifest=SimpleNamespace(entries=entries),
                               get_path=lambda name: SimpleNamespace(download=lambda root: str(tmp_path / name)))
    run = SimpleNamespace(state='finished', summary=summary, logged_artifacts=lambda: [artifact])
    api = SimpleNamespace(run=lambda path: run)
    proof = publisher.verify_diagnostic(api, source, tmp_path / 'readback')
    assert proof['accepted'] is False
    assert proof['measurement_aggregate_eligible'] is False
    assert proof['record_semantics_verified'] is True
    assert set(proof['files']) == set(required)
    assert not any('state.pt' in name for name in proof['files'])
    # The artifact body is downloaded separately and its semantic gate matters.
    remote_path = tmp_path / 'remote.json'
    write_json(remote_path, {**local, 'comparison': {'input_sha256_matches': False}})
    artifact.get_path = lambda name: SimpleNamespace(download=lambda root: str(remote_path))
    with pytest.raises(ValueError, match='pinned pre-upload'):
        publisher.verify_diagnostic(api, source, tmp_path / 'readback')
