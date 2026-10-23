"""Controlled axis sampling: approved shapes, finite cost, parser and restoration."""

import random

import numpy as np
import pytest
import torch

from backbone.token_compression.controller import CompressionController
from backbone.token_compression.reducers import OPERATORS
from scripts.tocom import run_cl
from scripts.tocom.matched_axes import AXES, LEVELS, MatchedAxesPlan, axis_rules, macs
from scripts.tocom.tracking import identity


@pytest.fixture(autouse=True)
def cpu_policy(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')


def consume(plan):
    for size in plan.sizes:
        plan.take(size)
    return plan.finish()


def test_policy_matches_the_reviewed_design_without_accuracy_tuning():
    expected = {
        'degree': [((224, level), (224, level)) for level in LEVELS],
        'resolution': [((224, 0), (224, 0)), ((192, 0), (208, 0)),
                       ((160, 0), (176, 0)), ((128, 0), (144, 0))],
        'joint': [((224, 0), (224, 0)), ((208, 4), (208, 3)),
                  ((176, 3), (208, 11)), ((208, 18), (176, 8))],
    }
    for axis in AXES:
        for level, (lo_expected, hi_expected) in zip(LEVELS, expected[axis]):
            lo, hi, probability = axis_rules(axis)[level]
            assert (lo, hi) == (lo_expected, hi_expected)
            assert (1 - probability) * macs(lo) + probability * macs(hi) == macs((224, level))


@pytest.mark.parametrize('seed', [1993, 1994, 1995])
def test_pilot_cost_is_sample_weighted_and_native_rngs_are_untouched(seed):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    records = {}
    for axis in AXES:
        plan = MatchedAxesPlan(axis, seed, 0, 0, 5000, 20, 16)
        assert len(plan.sizes) == 6260 and sum(plan.sizes) == 100000
        assert all(macs(shape) <= macs((224, 0)) for shape in plan.shapes)
        records[axis] = consume(plan)
    assert len({r['level_sequence_sha256'] for r in records.values()}) == 1
    assert len({r['target_sample_weighted_macs'] for r in records.values()}) == 1
    for record in records.values():
        assert record['sample_weighted_level_counts'] == {level: 25000 for level in LEVELS}
        assert abs(record['actual_sample_weighted_macs'] - record['target_sample_weighted_macs']) <= record['absolute_error_bound_macs']
    for a in AXES:
        for b in AXES:
            if a == b:
                continue
            target = records[a]['target_sample_weighted_macs']
            gap = abs(records[a]['actual_sample_weighted_macs'] - records[b]['actual_sample_weighted_macs'])
            bound = records[a]['absolute_error_bound_macs'] + records[b]['absolute_error_bound_macs']
            assert gap <= bound and bound / target < .0002
    assert random.getstate() == python_state
    assert numpy_state[0] == np.random.get_state()[0] and np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_low_budget_uses_exactly_the_same_shape_and_cost_for_all_arms():
    records = []
    for axis in AXES:
        plan = MatchedAxesPlan(axis, 1993, 2, 24, 5000, 20, 16)
        assert set(plan.shapes) == {(224, 24)}
        record = consume(plan)
        assert record['shape_steps'] == {'224:24': 6260} and record['relative_error'] == 0
        records.append(record)
    assert len({r['actual_sample_weighted_macs'] for r in records}) == 1
    assert len({r['shape_sequence_sha256'] for r in records}) == 1


def test_mismatched_or_truncated_loops_cannot_emit_a_completed_ledger():
    plan = MatchedAxesPlan('joint', 1, 0, 0, 10, 4, 3)
    with pytest.raises(RuntimeError, match='batch mismatch'):
        plan.take(1)
    assert plan.cursor == 0
    with pytest.raises(RuntimeError, match='0 of 16'):
        plan.finish()
    consume(plan)
    with pytest.raises(RuntimeError, match='more updates'):
        plan.take(3)
    for kwargs in ({'budget': 8}, {'axis': 'other'}, {'epochs': 1}, {'examples': 0}):
        values = dict(axis='joint', seed=1, task=0, budget=0, examples=10, epochs=4, batch_size=3)
        values.update(kwargs)
        with pytest.raises(ValueError):
            MatchedAxesPlan(**values)


def test_native_observe_keeps_its_contract_and_restores_defaults_on_failure():
    controller = CompressionController(OPERATORS['tome'](), 12, 0)
    controller.native, controller.patch, controller.resizable = 224, 16, True
    plan = MatchedAxesPlan('joint', 1, 0, 0, 8, 1, 2)
    before = controller.default_resolution, controller.default_degree
    x, y = torch.zeros(2, 3, 224, 224), torch.zeros(2, dtype=torch.long)
    seen = []

    def native(inputs, labels, not_aug_inputs, epoch=None):
        seen.append((controller.resolution or controller.native, controller.current))
        assert inputs is x and labels is y and not_aug_inputs is x and epoch == 3
        return {'loss': 1.25}

    assert plan.observe(controller, native, x, y, x, epoch=3) == {'loss': 1.25}
    assert seen == [plan.shapes[0]]
    assert (controller.default_resolution, controller.default_degree) == before

    def failed(*args, **kwargs):
        raise RuntimeError('native step failed')

    with pytest.raises(RuntimeError, match='native step failed'):
        plan.observe(controller, failed, x, y, x)
    assert (controller.default_resolution, controller.default_degree) == before


def arguments(axis):
    return ['--method', 'coda-prompt', '--seed', '1993', '--output', 'unused',
            '--path', 'P3', '--schedule', *map(str, [0, 0, 24, 24, 24, 24, 24, 0, 0, 0]),
            '--targets', '0', '8', '16', '24', '--no-grid',
            '--eval-shapes', '224:0', '224:8', '224:16', '224:24', '--matched-axis', axis]


@pytest.mark.parametrize('axis,suffix', [('degree', 'MCD'), ('resolution', 'MCR'), ('joint', 'MCJ')])
def test_parser_and_record_identity_keep_the_three_arms_distinct(axis, suffix):
    args = run_cl.parse_args(arguments(axis))
    assert args.matched_axis == axis and run_cl.path_suffix(args) == suffix
    fields = identity(vars(args))
    assert fields['kind'] == 'switch-mix' and fields['train_degree'] == 'P3' + suffix
    assert fields['matched_axis'] == axis


@pytest.mark.parametrize('extra', [
    ['--method', 'inf-lora'], ['--dataset', 'seq-imagenet-r'], ['--vit-size', 'small'],
    ['--mix', '0', '8', '16', '24'], ['--budget-shapes', '24=176:8'],
    ['--operator', 'evit'], ['--resolution', '176'], ['--epochs', '0'],
])
def test_incompatible_recipes_are_rejected_before_training(extra):
    with pytest.raises(SystemExit):
        run_cl.parse_args(arguments('joint') + extra)
