"""HiDe's exact four-metric accuracy comparison and verified structural gates."""
from copy import deepcopy

from scripts.ease_acceptance import counts_from_predictions
from scripts.ease_acceptance import evaluate_acceptance as _evaluate_exact
from scripts.ease_acceptance import REQUIRED_STRUCTURAL_CHECKS as _EXACT_HELPER_GATES

REQUIRED_STRUCTURAL_CHECKS = (
    'finite_loss_and_gradients', 'dataset_matches_reference', 'frozen_backbones_unchanged',
    'frozen_backbone_gradients_absent', 'old_prompt_slots_unchanged', 'no_future_or_replay_samples',
    'statistics_shapes_and_seen_classes', 'alignment_schedule_complete', 'routing_without_task_oracle',
    'checkpoint_round_trip', 'matched_learned_initialization', 'matched_phase_rng_and_batching',
    'training_protocol_complete',
)


def reference_policy(task_results):
    policy = {'unit': 'percentage_points', 'pass_max_drop_pp': 1.5, 'review_max_drop_pp': 3., 'metrics': []}
    rows = {row['task']: row for row in task_results}
    if set(rows) != {1, 2} or len(task_results) != 2:
        raise ValueError('HiDe reference must contain exactly two tasks')
    for task, partition in ((1, 'seen'), (2, 'seen'), (2, 'old'), (2, 'new')):
        row = rows[task]
        expected = 2000 if task == 2 and partition == 'seen' else 1000
        if row.get(partition + '_samples') != expected:
            raise ValueError('HiDe reference must evaluate every seen CIFAR100 sample')
        policy['metrics'].append({'metric': f'task{task}_{partition}', 'task': task,
                                 'accuracy_key': partition + '_accuracy_pct',
                                 'reference_correct': row[partition + '_correct'],
                                 'reference_samples': row[partition + '_samples']})
    return policy


def evaluate_acceptance(task_results, policy, *, structural_checks):
    """Reuse the established rational count evaluator with HiDe-specific gates.

    The EASE-only gate aliases exist inside the mathematical helper invocation;
    they are removed from the public HiDe report and never claim adapter checks.
    """
    valid = isinstance(structural_checks, dict)
    checks = deepcopy(structural_checks) if valid else {}
    missing = [name for name in REQUIRED_STRUCTURAL_CHECKS if checks.get(name) is not True]
    helper_checks = {name: not missing for name in _EXACT_HELPER_GATES}
    helper_checks.update(checks)
    report = _evaluate_exact(task_results, policy, structural_checks=helper_checks)
    report['structural_checks'] = checks
    report['structural_failures'] = [name for name in REQUIRED_STRUCTURAL_CHECKS if checks.get(name) is not True]
    report['structural_failures'].extend(name for name, value in checks.items()
                                         if value is not True and name not in report['structural_failures'])
    report['validation_errors'] = [error for error in report['validation_errors']
                                   if not error.startswith('missing required structural check')]
    if missing:
        report['status'] = 'fail'
        report['validation_errors'].append('Missing or failed HiDe structural gates: ' + ', '.join(missing))
    report['method'] = 'hide-prompt'
    return report
