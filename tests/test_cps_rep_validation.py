"""Independent-source and bounded-run acceptance contracts; no GPU/services."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.cps_author_reference import SOURCE, author_classes, verify_source
from scripts.cps_rep_protocol import (
    accuracy_status,
    compare_records,
    learned_state,
    native_arguments,
    restore_learned,
    tensor_digest,
)


def test_reference_uses_pinned_author_bodies_and_matches_native_forward():
    from models.cps_prompt_utils.model import PromptModel

    torch.set_num_threads(2)
    classes = author_classes()
    assert len(verify_source()) == 6
    assert classes.model.observe.__code__.co_filename == str(SOURCE / 'method.py')
    assert classes.prompt['CodaPrompt'].forward.__code__.co_filename == str(SOURCE / 'prompt.py')
    assert classes.patched['CPSAttention'].forward.__code__.co_filename == str(SOURCE / 'cps_vit.py')
    args = SimpleNamespace(n_tasks=10, vit_type='tiny', e_prompt_pool_size=100,
                           e_prompt_length=8, e_prompt_layer_idx=[0, 1, 2, 3, 4], ortho_mu=0.,
                           reduction_ratio=.4, temperature=.1, sampling='critical_score')
    with torch.random.fork_rng():
        torch.default_generator.manual_seed(1993)
        source = classes.network(args, num_classes=100, pretrained=True)
        native = PromptModel()
        restore_learned(native, learned_state(source))
        images = torch.randn(2, 3, 224, 224)
        left, _ = source(images, train=True)
        right, _ = native(images, train=True)
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
        for network in (source, native):
            torch.default_generator.manual_seed(7281)
            network.prompt.process_task_count()
        assert tensor_digest(learned_state(source)) == tensor_digest(learned_state(native))
        with torch.no_grad():
            torch.testing.assert_close(source(images), native(images), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(('delta', 'status'), [(1.5, 'pass'), (-1.5, 'pass'), (1.5001, 'review'),
                                             (3., 'review'), (-3.0001, 'fail')])
def test_accuracy_boundaries(delta, status):
    assert accuracy_status(delta) == status


def _completed():
    return {'status': 'completed', 'structural_checks_passed': True, 'protocol': {'tasks': 2},
            'physical_gpu_uuid': 'GPU-matched', 'initial_learned_sha256': 'matched',
            'epoch_inputs': [{'task': 1, 'batches_sha256': 'actual-inputs'}], 'data_fingerprints': {'train': 'matched'},
            'task_results': [{'task': 1, 'seen_accuracy_pct': 80., 'old_accuracy_pct': None, 'new_accuracy_pct': 80.},
                             {'task': 2, 'seen_accuracy_pct': 70., 'old_accuracy_pct': 60., 'new_accuracy_pct': 80.}]}


@pytest.mark.parametrize('field', ['protocol', 'physical_gpu_uuid', 'initial_learned_sha256', 'epoch_inputs', 'data_fingerprints'])
def test_comparison_rejects_actual_evidence_mismatch(field):
    source, native = _completed(), _completed()
    native[field] = 'different'
    assert compare_records(native, source)['status'] == 'fail'


def test_rep_full_gate_is_gross_decline_and_includes_old_task():
    control, rep = _completed(), _completed()
    rep['task_results'][1]['old_accuracy_pct'] -= 3.
    assert compare_records(rep, control, decline_only=True)['status'] == 'pass'
    rep['task_results'][1]['old_accuracy_pct'] -= .01
    assert compare_records(rep, control, decline_only=True)['status'] == 'hold'
    rep = copy.deepcopy(control)
    rep['task_results'][1]['new_accuracy_pct'] += 10.
    assert compare_records(rep, control, decline_only=True)['status'] == 'pass'


@pytest.mark.parametrize('run', ['cps-author', 'cps-native', 'coda-frozen', 'rep-off', 'rep-on'])
def test_bounded_arguments_keep_two_tasks_three_epochs_batch64(run, tmp_path):
    args = native_arguments(run, tmp_path)
    for name, value in (('--stop_after', '2'), ('--n_epochs', '3'), ('--batch_size', '64'),
                        ('--code_optimization', '0'), ('--seed', '1993')):
        assert args[args.index(name) + 1] == value
    assert '--debug_mode' not in args
    if run.startswith('rep-'):
        assert all(args[args.index(name) + 1] == ('1' if run == 'rep-on' else '0')
                   for name in ('--rep_swift', '--rep_atom', '--rep_ald'))
