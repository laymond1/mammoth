"""Cost-matched high-budget axis comparison; low-budget tasks share (224, 24).

The scalar cost is the existing bare ViT-Tiny encoder MAC proxy. Balanced
sampling controls its expectation and bounds finite sample-weighted error;
it is not an independent loss-gradient sampler or a full training cost model.
"""

from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import random

from backbone.token_compression.shapes import block_macs


AXES = ('degree', 'resolution', 'joint')
LEVELS = (0, 8, 16, 24)
SUFFIXES = {'degree': 'MCD', 'resolution': 'MCR', 'joint': 'MCJ'}


@lru_cache(maxsize=None)
def macs(shape):
    return block_macs(*shape)


def _rng(*keys):
    digest = hashlib.sha256('|'.join(map(str, keys)).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], 'big'))


def _bracket(points, target):
    exact = [a for a in points if macs(a) == target]
    if exact:
        a = min(exact)
        return a, a, Fraction(0)
    lower = [a for a in points if macs(a) < target]
    upper = [a for a in points if macs(a) > target]
    if not lower or not upper:
        raise ValueError('The axis has no shapes bracketing the target MAC cost')
    lo = max(lower, key=lambda a: (macs(a), a))
    hi = min(upper, key=lambda a: (macs(a), a))
    return lo, hi, Fraction(target - macs(lo), macs(hi) - macs(lo))


def axis_rules(axis):
    """Closest shapes around each target; no accuracy-dependent selection."""
    if axis not in AXES:
        raise ValueError(f'Unknown matched axis: {axis}')
    rules = {}
    for level in LEVELS:
        points = ([(224, level)] if axis == 'degree' or level == 0 else
                  [(r, 0) for r in range(128, 225, 16)] if axis == 'resolution' else
                  [(r, d) for r in range(128, 224, 16) for d in range(1, 25)])
        rules[level] = _bracket(points, macs((224, level)))
    return rules


class MatchedAxesPlan:
    """One task's fixed shape sequence, consumed only at meta_observe.

    Every arm shares cost-stratum order, including a separate balance for the
    smaller last batch. Separate local RNGs leave all native RNG streams alone.
    The plan checks actual batch sizes and complete consumption rather than
    silently accepting truncated/debug or overridden training loops.
    """

    def __init__(self, axis, seed, task, budget, examples, epochs, batch_size):
        if budget not in (0, 24):
            raise ValueError('Matched axes supports task budgets 0 and 24 only')
        if min(examples, epochs, batch_size) <= 0:
            raise ValueError('Matched axes requires positive examples, epochs and batch size')
        rules = axis_rules(axis)
        self.axis, self.task, self.budget = axis, task, budget
        sizes = [batch_size] * (examples // batch_size)
        if examples % batch_size:
            sizes.append(examples % batch_size)
        sizes *= epochs
        positions = defaultdict(list)
        for index, size in enumerate(sizes):
            positions[size].append(index)
        levels = [24] * len(sizes)
        if budget == 0:
            for size, indices in positions.items():
                if len(indices) % len(LEVELS):
                    raise ValueError('Matched axes needs each batch-size count divisible by four across a task')
                assigned = list(LEVELS) * (len(indices) // len(LEVELS))
                _rng('levels', seed, task, size).shuffle(assigned)
                for index, level in zip(indices, assigned):
                    levels[index] = level
        groups = defaultdict(list)
        for index, (level, size) in enumerate(zip(levels, sizes)):
            groups[level, size].append(index)
        self.shapes = [(224, 24)] * len(sizes)
        self.error_bound_macs = 0
        if budget == 0:
            for (level, size), indices in groups.items():
                lo, hi, probability = rules[level]
                real_count = len(indices) * probability
                count_hi = real_count.numerator // real_count.denominator
                if _rng('round', seed, task, level, size, axis).random() < real_count - count_hi:
                    count_hi += 1
                assigned = [hi] * count_hi + [lo] * (len(indices) - count_hi)
                _rng('shapes', seed, task, level, size, axis).shuffle(assigned)
                for index, shape in zip(indices, assigned):
                    self.shapes[index] = shape
                self.error_bound_macs += size * (macs(hi) - macs(lo))
        self.sizes, self.levels, self.cursor = sizes, levels, 0
        self.target_macs = sum(size * macs((224, level)) for size, level in zip(sizes, levels))
        self.actual_macs = sum(size * macs(a) for size, a in zip(sizes, self.shapes))
        if abs(self.actual_macs - self.target_macs) > self.error_bound_macs:
            raise RuntimeError('Matched-axis plan exceeds its MAC rounding bound')
        if any(macs(a) > macs((224, budget)) for a in self.shapes):
            raise RuntimeError('Matched-axis shape exceeds the nominal task budget')

    def take(self, batch_size):
        if self.cursor >= len(self.shapes):
            raise RuntimeError('Matched-axis plan received more updates than scheduled')
        if batch_size != self.sizes[self.cursor]:
            raise RuntimeError(f'Matched-axis batch mismatch: expected {self.sizes[self.cursor]}, got {batch_size}')
        shape = self.shapes[self.cursor]
        self.cursor += 1
        return shape

    def observe(self, controller, native_observe, *inputs, **keywords):
        """Change only update-time defaults, restoring them even on a failed step."""
        resolution, degree = self.take(len(inputs[0]))
        before = controller.default_resolution, controller.default_degree
        try:
            controller.default_resolution = controller.check_resolution(resolution)
            controller.default_degree = degree
            return native_observe(*inputs, **keywords)
        finally:
            controller.default_resolution, controller.default_degree = before

    def finish(self):
        if self.cursor != len(self.shapes):
            raise RuntimeError(f'Matched-axis task used {self.cursor} of {len(self.shapes)} scheduled updates')
        counts = Counter(f'{r}:{d}' for r, d in self.shapes)
        histogram = Counter((level, size, *a) for level, size, a in zip(self.levels, self.sizes, self.shapes))
        return {
            'axis': self.axis, 'task': self.task, 'budget': self.budget,
            'default_shape': [224, self.budget], 'updates': self.cursor,
            'images_with_repeats': sum(self.sizes), 'shape_steps': dict(sorted(counts.items())),
            'actual_sample_weighted_macs': self.actual_macs, 'target_sample_weighted_macs': self.target_macs,
            'relative_error': self.actual_macs / self.target_macs - 1,
            'absolute_error_bound_macs': self.error_bound_macs,
            'level_sequence_sha256': hashlib.sha256(json.dumps(self.levels).encode()).hexdigest(),
            'shape_sequence_sha256': hashlib.sha256(json.dumps(self.shapes).encode()).hexdigest(),
            'sample_weighted_level_counts': {level: sum(size for size, l in zip(self.sizes, self.levels) if l == level)
                                             for level in sorted(set(self.levels))},
            'histogram': {':'.join(map(str, k)): count for k, count in sorted(histogram.items())},
        }
