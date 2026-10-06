"""Exact EASE acceptance boundaries using saved-count fixtures, without compute."""

from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path

import pytest

from scripts.ease_acceptance import REQUIRED_STRUCTURAL_CHECKS, counts_from_predictions, evaluate_acceptance


PLAN = Path(__file__).resolve().parents[1] / 'benchmarks/cl_efficiency/ease/validation_plan.json'


@pytest.fixture
def policy():
    return json.loads(PLAN.read_text())['validation']['accuracy_policy']


def rows(task1=996, old=966, new=978):
    return [
        {'task': 1, 'seen_correct': task1, 'seen_samples': 1000,
         'old_correct': 0, 'old_samples': 0, 'new_correct': task1, 'new_samples': 1000},
        {'task': 2, 'seen_correct': old + new, 'seen_samples': 2000,
         'old_correct': old, 'old_samples': 1000, 'new_correct': new, 'new_samples': 1000},
    ]


CHECKS = {'finite_loss_and_gradients': True, 'dataset_matches_reference': True,
          'frozen_backbone_unchanged': True, 'old_adapters_unchanged': True,
          'prototype_and_adapter_shapes': True, 'training_protocol_complete': True,
          'frozen_backbone_gradients_absent': True}


def assess(policy, task_results=None, checks=None):
    return evaluate_acceptance(rows() if task_results is None else task_results, policy,
                               structural_checks=CHECKS if checks is None else checks)


def test_prediction_counts_partition_saved_top1_labels():
    counts = counts_from_predictions([0, 1, 1, 3], [0, 2, 1, 3], past_classes=2, seen_classes=4)
    assert counts == {'seen_correct': 3, 'seen_samples': 4, 'old_correct': 2,
                      'old_samples': 2, 'new_correct': 1, 'new_samples': 2}
    first = counts_from_predictions(iter([0, 1]), iter([0, 0]), past_classes=0, seen_classes=2)
    assert first == {'seen_correct': 1, 'seen_samples': 2, 'old_correct': 0,
                     'old_samples': 0, 'new_correct': 1, 'new_samples': 2}


@pytest.mark.parametrize('predictions,labels,past,seen', [
    ([], [], 0, 2), ([0], [0, 1], 0, 2), ([[0, 1]], [0], 0, 2),
    ([True], [0], 0, 2), ([0.0], [0], 0, 2), ([0], [float('nan')], 0, 2),
    ([2], [0], 0, 2), ([0], [-1], 0, 2), ([0], [2], 0, 2), ([0], [0], 2, 2),
])
def test_invalid_saved_prediction_inputs_are_rejected(predictions, labels, past, seen):
    with pytest.raises(ValueError):
        counts_from_predictions(predictions, labels, past_classes=past, seen_classes=seen)


def test_reference_counts_pass_and_report_is_json_safe_without_mutation(policy):
    native = rows()
    before = deepcopy((native, policy, CHECKS))
    result = assess(policy, native)
    assert result['status'] == 'pass'
    assert set(result['metrics']) == {'task1_seen', 'task2_seen', 'task2_old', 'task2_new'}
    assert all(metric['drop_pp_exact'] == {'numerator': 0, 'denominator': 1} for metric in result['metrics'].values())
    assert not result['validation_errors'] and not result['structural_failures']
    assert (native, policy, CHECKS) == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('task1,old,new,status,drop', [
    (981, 951, 963, 'pass', Fraction(3, 2)),
    (980, 950, 962, 'review', Fraction(8, 5)),
    (966, 936, 948, 'review', Fraction(3)),
    (965, 935, 947, 'fail', Fraction(31, 10)),
])
def test_exact_thresholds_classify_each_metric(policy, task1, old, new, status, drop):
    result = assess(policy, rows(task1, old, new))
    assert result['status'] == status
    for metric in result['metrics'].values():
        assert metric['status'] == status
        assert metric['drop_pp_exact'] == {'numerator': drop.numerator, 'denominator': drop.denominator}


def test_display_rounding_does_not_turn_just_below_boundary_into_pass(policy):
    native = rows(old=951, new=962)
    # 1913/2000 is 95.65%; a one-decimal display can look like the 95.7% boundary.
    native[1]['seen_accuracy_pct'] = 95.7
    result = assess(policy, native)
    seen = result['metrics']['task2_seen']
    assert seen['status'] == 'review'
    assert seen['native_accuracy_pct'] == 95.65
    assert seen['drop_pp_exact'] == {'numerator': 31, 'denominator': 20}
    assert result['metrics']['task2_old']['status'] == 'pass'


def test_decimal_and_fraction_limits_preserve_exact_boundaries(policy):
    policy['pass_max_drop_pp'] = Decimal('1.5')
    policy['review_max_drop_pp'] = Fraction(3)
    assert assess(policy, rows(981, 951, 963))['status'] == 'pass'


def test_improvements_pass_with_negative_drops(policy):
    result = assess(policy, rows(1000, 1000, 1000))
    assert result['status'] == 'pass'
    assert all(metric['drop_pp'] < 0 for metric in result['metrics'].values())
    assert result['metrics']['task1_seen']['drop_pp_exact'] == {'numerator': -2, 'denominator': 5}


def test_one_metric_failure_overrides_passing_average(policy):
    result = assess(policy, rows(old=935, new=1000))
    assert result['metrics']['task2_seen']['status'] == 'pass'
    assert result['metrics']['task2_new']['status'] == 'pass'
    assert result['metrics']['task2_old']['status'] == 'fail'
    assert result['status'] == 'fail'


def test_one_review_metric_overrides_pass(policy):
    result = assess(policy, rows(old=950, new=978))
    assert result['metrics']['task2_seen']['status'] == 'pass'
    assert result['metrics']['task2_old']['status'] == 'review'
    assert result['status'] == 'review'


def test_structural_failure_overrides_accuracy_and_review(policy):
    checks = dict(CHECKS, old_adapters_unchanged=False)
    result = assess(policy, rows(980, 950, 962), checks)
    assert result['status'] == 'fail'
    assert result['structural_failures'] == ['old_adapters_unchanged']


@pytest.mark.parametrize('checks', [{}, {'finite_loss_and_gradients': None}, {'dataset_matches_reference': 'True'}, {1: True}])
def test_missing_or_invalid_structural_results_fail(policy, checks):
    result = assess(policy, checks=checks)
    assert result['status'] == 'fail'
    assert result['validation_errors']


@pytest.mark.parametrize('name', REQUIRED_STRUCTURAL_CHECKS)
def test_omitting_any_required_structural_check_blocks_acceptance(policy, name):
    checks = dict(CHECKS)
    del checks[name]
    result = assess(policy, checks=checks)
    assert result['status'] == 'fail'
    assert f'missing:{name}' in result['structural_failures']


def test_missing_task_fails_instead_of_using_float_accuracy_fallback(policy):
    native = rows()[:1]
    result = assess(policy, native)
    assert result['status'] == 'fail'
    assert result['metrics']['task1_seen']['status'] == 'pass'
    assert all(result['metrics'][name]['status'] == 'fail' for name in ('task2_seen', 'task2_old', 'task2_new'))
    assert result['validation_errors']


def test_missing_metric_count_fails_even_if_display_accuracy_exists(policy):
    native = rows()
    del native[1]['new_correct']
    native[1]['new_accuracy_pct'] = 97.8
    assert assess(policy, native)['status'] == 'fail'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 1001, True, 996.0, None])
def test_nonfinite_or_invalid_native_counts_fail(policy, value):
    native = rows()
    native[0]['seen_correct'] = value
    assert assess(policy, native)['status'] == 'fail'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), None])
def test_nonfinite_display_metrics_fail_even_with_valid_counts(policy, value):
    native = rows()
    native[1]['seen_accuracy_pct'] = value
    assert assess(policy, native)['status'] == 'fail'


def test_sample_count_mismatch_blocks_comparison(policy):
    native = rows()
    native[1].update(seen_samples=1999, old_samples=999)
    result = assess(policy, native)
    assert result['status'] == 'fail'
    assert any('sample count 1999 differs from reference 2000' in error for error in result['validation_errors'])


def test_partition_inconsistency_is_validation_failure(policy):
    native = rows()
    native[1]['seen_correct'] += 1
    result = assess(policy, native)
    assert result['status'] == 'fail'
    assert any('do not partition' in error for error in result['validation_errors'])


@pytest.mark.parametrize('invalid_rows', [None, [], [None], rows() + [rows()[0]], rows() + [{'task': 3}]])
def test_invalid_or_unapproved_task_results_fail(policy, invalid_rows):
    result = evaluate_acceptance(invalid_rows, policy, structural_checks=CHECKS)
    assert result['status'] == 'fail'


@pytest.mark.parametrize('change', [
    {'unit': 'percent'}, {'pass_max_drop_pp': float('nan')}, {'review_max_drop_pp': float('inf')},
    {'pass_max_drop_pp': -1}, {'review_max_drop_pp': 1}, {'metrics': []},
])
def test_malformed_policy_is_rejected_instead_of_silent_defaults(policy, change):
    policy.update(change)
    with pytest.raises(ValueError):
        assess(policy)


def test_stale_display_hardcoded_threshold_is_rejected(policy):
    policy['metrics'][0]['pass_min_accuracy_pct'] = 98.6
    with pytest.raises(ValueError, match='disagrees'):
        assess(policy)


def test_reference_counts_and_partitions_must_be_consistent(policy):
    policy['metrics'][0]['reference_correct'] = 997
    with pytest.raises(ValueError, match='disagrees'):
        assess(policy)
