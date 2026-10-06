"""Exact, computation-free acceptance of the approved EASE two-task port.

Only saved top-1 predictions and labels are needed. No model, tensor library,
credentials, or runtime configuration is imported. Display-rounded accuracies
never determine acceptance.
"""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from numbers import Integral


_METRICS = {
    'task1_seen': (1, 'seen'),
    'task2_seen': (2, 'seen'),
    'task2_old': (2, 'old'),
    'task2_new': (2, 'new'),
}
_PARTITIONS = ('seen', 'old', 'new')
REQUIRED_STRUCTURAL_CHECKS = (
    'finite_loss_and_gradients',
    'dataset_matches_reference',
    'frozen_backbone_unchanged',
    'old_adapters_unchanged',
    'prototype_and_adapter_shapes',
    'training_protocol_complete',
    'frozen_backbone_gradients_absent',
)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer count')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _decimal_fraction(value, name):
    """Read JSON decimal values without introducing binary-float rounding."""
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite decimal number')
    if isinstance(value, Fraction):
        return value
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f'{name} must be a finite decimal number') from None
    if not decimal.is_finite():
        raise ValueError(f'{name} must be a finite decimal number')
    return Fraction(decimal)


def _encoded_fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def counts_from_predictions(predictions, labels, *, past_classes, seen_classes):
    """Count one saved evaluation using flat top-1 class IDs.

    For an N x K predictions archive, pass its first column explicitly. Labels
    below ``past_classes`` are old; the remainder are new. Returned keys can be
    added to a task-results row with ``{'task': 1 or 2, **counts}``.
    """
    past_classes = _integer(past_classes, 'past_classes')
    seen_classes = _integer(seen_classes, 'seen_classes', minimum=1)
    if past_classes >= seen_classes:
        raise ValueError('past_classes must be smaller than seen_classes')
    try:
        predictions, labels = list(predictions), list(labels)
    except TypeError:
        raise ValueError('predictions and labels must be flat iterables') from None
    if not labels or len(predictions) != len(labels):
        raise ValueError('predictions and labels need equal nonzero sample counts')
    counts = {f'{group}_{key}': 0 for group in _PARTITIONS for key in ('correct', 'samples')}
    for index, (prediction, label) in enumerate(zip(predictions, labels)):
        prediction = _integer(prediction, f'prediction[{index}]')
        label = _integer(label, f'label[{index}]')
        if prediction >= seen_classes or label >= seen_classes:
            raise ValueError(f'sample {index} contains a class outside the seen classes')
        group = 'old' if label < past_classes else 'new'
        correct = int(prediction == label)
        counts['seen_samples'] += 1
        counts['seen_correct'] += correct
        counts[f'{group}_samples'] += 1
        counts[f'{group}_correct'] += correct
    return counts


def _validated_policy(policy):
    if not isinstance(policy, Mapping) or policy.get('unit') != 'percentage_points':
        raise ValueError('policy must use percentage_points')
    pass_limit = _decimal_fraction(policy.get('pass_max_drop_pp'), 'pass_max_drop_pp')
    review_limit = _decimal_fraction(policy.get('review_max_drop_pp'), 'review_max_drop_pp')
    if not 0 <= pass_limit < review_limit:
        raise ValueError('policy requires 0 <= pass_max_drop_pp < review_max_drop_pp')
    references = {}
    entries = policy.get('metrics')
    if not isinstance(entries, (list, tuple)):
        raise ValueError('policy metrics must be a list of the four approved accuracy metrics')
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get('metric') not in _METRICS:
            raise ValueError('policy contains an unsupported accuracy metric')
        name = entry['metric']
        if name in references:
            raise ValueError(f'policy contains duplicate metric {name}')
        task, partition = _METRICS[name]
        if _integer(entry.get('task'), f'{name}.task', minimum=1) != task or entry.get('accuracy_key') != f'{partition}_accuracy_pct':
            raise ValueError(f'policy task/accuracy_key mismatch for {name}')
        samples = _integer(entry.get('reference_samples'), f'{name}.reference_samples', minimum=1)
        correct = _integer(entry.get('reference_correct'), f'{name}.reference_correct')
        if correct > samples:
            raise ValueError(f'{name}.reference_correct exceeds reference_samples')
        accuracy = Fraction(100 * correct, samples)
        for field, expected in (
            ('reference_accuracy_pct', accuracy),
            ('pass_min_accuracy_pct', accuracy - pass_limit),
            ('review_min_accuracy_pct', accuracy - review_limit),
        ):
            if field in entry and _decimal_fraction(entry[field], f'{name}.{field}') != expected:
                raise ValueError(f'{name}.{field} disagrees with exact reference counts and policy')
        references[name] = {'correct': correct, 'samples': samples, 'accuracy': accuracy}
    if set(references) != set(_METRICS):
        raise ValueError('policy must contain exactly the four approved accuracy metrics')
    if references['task2_seen']['samples'] != sum(references[name]['samples'] for name in ('task2_old', 'task2_new')):
        raise ValueError('policy reference old/new sample counts do not partition task2_seen')
    if references['task2_seen']['correct'] != sum(references[name]['correct'] for name in ('task2_old', 'task2_new')):
        raise ValueError('policy reference old/new correct counts do not partition task2_seen')
    return pass_limit, review_limit, references


def _validated_task_counts(row, task):
    counts = {}
    for partition in _PARTITIONS:
        minimum = 0 if task == 1 and partition == 'old' else 1
        samples = _integer(row.get(f'{partition}_samples'), f'{partition}_samples', minimum=minimum)
        correct = _integer(row.get(f'{partition}_correct'), f'{partition}_correct')
        if correct > samples:
            raise ValueError(f'{partition}_correct exceeds {partition}_samples')
        counts[partition] = (correct, samples)
        accuracy_key = f'{partition}_accuracy_pct'
        if accuracy_key in row:
            if not (partition == 'old' and task == 1 and row[accuracy_key] is None):
                _decimal_fraction(row[accuracy_key], accuracy_key)
    if task == 1 and counts['old'] != (0, 0):
        raise ValueError('task 1 must not contain old-class samples')
    for column, name in ((0, 'correct'), (1, 'samples')):
        if counts['seen'][column] != counts['old'][column] + counts['new'][column]:
            raise ValueError(f'old/new {name} counts do not partition seen counts')
    return counts


def evaluate_acceptance(task_results, policy, *, structural_checks):
    """Return a JSON-safe report; malformed native results always fail.

    ``task_results`` contains exactly two rows with ``task`` and all six count
    keys returned by :func:`counts_from_predictions`. Optional display accuracy
    fields may be rounded, but must be finite. ``policy`` is the approved
    ``validation.accuracy_policy`` object. A malformed policy raises ValueError
    instead of silently selecting different margins.

    ``structural_checks`` maps all ``REQUIRED_STRUCTURAL_CHECKS`` names to bool,
    where True means verified/pass. Additional checks are allowed. Failure names
    in the policy are descriptive, not positive-check names. The caller must
    actually perform the checks; this function cannot verify a model from
    predictions. Missing, nonboolean or failed checks block passing.
    """
    pass_limit, review_limit, references = _validated_policy(policy)
    errors, tasks, bad_tasks = [], {}, set()
    try:
        rows = list(task_results)
    except TypeError:
        rows = []
        errors.append('task_results must be an iterable of task count mappings')
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append('each task result must be a mapping')
            continue
        try:
            task = _integer(row.get('task'), 'task', minimum=1)
        except ValueError as error:
            errors.append(str(error))
            continue
        if task not in (1, 2):
            errors.append(f'unapproved task {task}; only tasks 1 and 2 are accepted')
            continue
        if task in tasks or task in bad_tasks:
            bad_tasks.add(task)
            tasks.pop(task, None)
            errors.append(f'duplicate task {task}')
            continue
        try:
            tasks[task] = _validated_task_counts(row, task)
        except ValueError as error:
            bad_tasks.add(task)
            errors.append(f'task {task}: {error}')

    checks, structural_failures = {}, []
    if not isinstance(structural_checks, Mapping) or not structural_checks:
        errors.append('structural_checks must be a nonempty mapping of verified booleans')
        structural_failures.append('missing_structural_checks')
    else:
        for name, passed in structural_checks.items():
            if not isinstance(name, str) or not name or not isinstance(passed, bool):
                errors.append('structural check names must be nonempty strings and values must be bool')
                structural_failures.append('invalid_structural_check')
                continue
            checks[name] = passed
            if not passed:
                structural_failures.append(name)
    for name in REQUIRED_STRUCTURAL_CHECKS:
        if name not in checks:
            errors.append(f'missing required structural check {name}')
            structural_failures.append(f'missing:{name}')

    metrics = {}
    for name, (task, partition) in _METRICS.items():
        reference = references[name]
        metric = {
            'task': task, 'status': 'fail',
            'reference_correct': reference['correct'], 'reference_samples': reference['samples'],
            'reference_accuracy_pct': float(reference['accuracy']),
            'native_correct': None, 'native_samples': None, 'native_accuracy_pct': None,
            'drop_pp': None, 'drop_pp_exact': None, 'validation_errors': [],
        }
        if task not in tasks or task in bad_tasks:
            metric['validation_errors'].append(f'task {task} is missing or invalid')
        else:
            correct, samples = tasks[task][partition]
            metric.update(native_correct=correct, native_samples=samples)
            if samples != reference['samples']:
                metric['validation_errors'].append(f'{name}: native sample count {samples} differs from reference {reference["samples"]}')
            else:
                accuracy = Fraction(100 * correct, samples)
                drop = reference['accuracy'] - accuracy
                status = 'pass' if drop <= pass_limit else ('review' if drop <= review_limit else 'fail')
                metric.update(status=status, native_accuracy_pct=float(accuracy), drop_pp=float(drop),
                              drop_pp_exact=_encoded_fraction(drop))
        errors.extend(metric['validation_errors'])
        metrics[name] = metric

    statuses = {metric['status'] for metric in metrics.values()}
    status = 'fail' if errors or structural_failures or 'fail' in statuses else ('review' if 'review' in statuses else 'pass')
    return {
        'status': status,
        'pass_max_drop_pp': float(pass_limit), 'pass_max_drop_pp_exact': _encoded_fraction(pass_limit),
        'review_max_drop_pp': float(review_limit), 'review_max_drop_pp_exact': _encoded_fraction(review_limit),
        'metrics': metrics, 'structural_checks': checks, 'structural_failures': structural_failures,
        'validation_errors': errors,
        'comparison_basis': 'Exact correct/sample counts; displayed accuracy and drop values do not decide acceptance.',
        'statistical_tolerance_estimated': False,
    }
