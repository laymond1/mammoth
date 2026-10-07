"""Run the accepted HiDe-Prompt task-one efficiency pilot in fresh workers.

Three measurement workers use TII1 + prompt1 with common EASE preprocessing.
A fourth worker diagnoses partial supported-operator FLOPs; its resource costs
are excluded from all measurement distributions. Preparation imports no ML
library, reads no credentials, and starts no experiment.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cl_efficiency import write_json
from scripts.cl_runtime import configure_environment
from scripts.hide_acceptance import REQUIRED_STRUCTURAL_CHECKS

PLAN_PATH = ROOT / 'benchmarks/cl_efficiency/hide_prompt/execution_plan.json'
RESOURCE_KEYS = (
    'process_wall_s', 'process_tree_cpu_s', 'cuda_peak_allocated_mib',
    'cuda_peak_reserved_mib', 'peak_process_tree_rss_mib',
    'peak_gpu_device_memory_mib', 'gpu_board_energy_j', 'gpu_board_energy_wh',
    'gpu_energy_coverage_fraction', 'outer_training_examples_per_s',
    'evaluation_examples_per_s',
)
PHASE_KEYS = (
    'initialize', 'data_loading', 'begin_task', 'tii_train', 'prompt_train',
    'tii_statistics', 'prompt_finalize', 'prompt_statistics', 'evaluation',
    'checkpoint',
)
REQUIRED_CHECKS = REQUIRED_STRUCTURAL_CHECKS
PROTOCOL_COUNTS = {
    'tasks_completed': 1, 'outer_observe_steps': 158,
    'outer_training_stream_examples': 10000, 'evaluation_examples': 1000,
}
METHOD_COUNTS = {
    'tii_training_steps': 79, 'prompt_training_steps': 79,
    'tii_statistics_examples': 5000, 'prompt_statistics_examples': 5000,
    'prompt_training_tii_examples': 5000, 'routing_evaluation_examples': 1000,
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_approved_scope(plan, acceptance, variant=None):
    """Require the selected size's acceptance before any pilot worker."""
    common = plan.get('common', {})
    expected_common = {
        'dataset': 'seq-cifar100-224', 'dataset_total_tasks': 10,
        'classes_per_task': 10, 'seed': 1993, 'num_workers': 4, 'gpu': '1',
    }
    efficiency = plan.get('efficiency', {})
    expected = {
        'tasks': 1, 'dataset_config': 'ease', 'tii_epochs': 1,
        'prompt_epochs': 1, 'outer_epochs_per_task': 2, 'ca_epochs': 30,
        'batch_size': 64, 'training_authorized': True,
        'measurement_repeats': 3, 'flops_runs': 1, 'ca_runs_task1': False,
    }
    for actual, desired in ((common, expected_common), (efficiency, expected)):
        if any(type(actual.get(key)) is not type(value) or actual.get(key) != value
               for key, value in desired.items()):
            raise ValueError('Pilot differs from the authorized TII1/prompt1 three-measurement/one-FLOPs protocol')
    if variant is None:
        if common.get('checkpoint_tag') != 'vit_base_patch16_224.augreg_in21k_ft_in1k':
            raise ValueError('The approved HiDe-Prompt efficiency pilot uses the Base checkpoint')
    else:
        from scripts.hide_size_protocol import load_variant
        from scripts.hide_protocol import protocol
        info = load_variant(variant, verify_checkpoint=False)
        if any(common.get(key) != info[key] for key in ('vit_size', 'feature_dim', 'num_heads',
                    'checkpoint_file', 'checkpoint_sha256', 'canonical_pretrained_sha256', 'checkpoint_tag')):
            raise ValueError('Pilot differs from the selected size checkpoint and geometry')
        if (acceptance.get('variant') != variant or acceptance.get('protocol') != protocol('validation', variant=variant)
                or acceptance.get('checkpoint_sha256') != info['checkpoint_sha256']
                or acceptance.get('canonical_pretrained_sha256') != info['canonical_pretrained_sha256']):
            raise ValueError('The selected size must have its own matched two-task acceptance')
    checks = acceptance.get('structural_checks', {})
    if (acceptance.get('status') != 'completed' or acceptance.get('acceptance_status') != 'pass'
            or not checks or any(value is not True for value in checks.values())):
        raise ValueError('Matched native/reference two-task acceptance must pass before efficiency execution')


def distribution(values, *, allow_missing=False):
    values = list(values)
    observed = [value for value in values if value is not None]
    if (not values or (not allow_missing and len(observed) != len(values))
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in observed)):
        raise ValueError('Resource aggregation requires finite nonnegative numerical observations')
    result = {'median': statistics.median(observed) if observed else None,
              'min': min(observed) if observed else None,
              'max': max(observed) if observed else None, 'values': values}
    if allow_missing:
        result['observed_count'] = len(observed)
    return result


def validate_record(record, pass_name, variant=None):
    if record.get('pass') != pass_name or record.get('status') != 'completed':
        raise ValueError('Only completed workers from the requested pass may enter aggregation')
    if any(type(record.get('metrics', {}).get(key)) is not int or record['metrics'][key] != value
           for key, value in PROTOCOL_COUNTS.items()):
        raise ValueError('Incomplete first-task protocol cannot enter aggregation')
    if any(type(record.get('counts', {}).get(key)) is not int or record['counts'][key] != value
           for key, value in METHOD_COUNTS.items()):
        raise ValueError('Both training phases and full statistics passes must complete')
    checks = record.get('structural_checks', {})
    if any(checks.get(key) is not True for key in REQUIRED_CHECKS) or any(value is not True for value in checks.values()):
        raise ValueError('Structural checks must pass before aggregation')
    alignment = record.get('alignment_history', [])
    if (len(alignment) != 2 or {row.get('branch') for row in alignment} != {'tii', 'prompt'}
            or any(row.get('task_id') != 0 or row.get('status') != 'not_applicable'
                   or row.get('steps') != 0 for row in alignment)):
        raise ValueError('Both first-task classifier alignments must be recorded as not applicable')
    variant = variant if variant is not None else record.get('variant')
    if variant is not None:
        from scripts.hide_size_protocol import load_variant
        from scripts.hide_protocol import protocol, FULL_CL_PRECISION
        info = load_variant(variant, verify_checkpoint=False)
        if (record.get('variant') != variant or record.get('protocol') != protocol('efficiency', variant=variant)
                or record.get('model') != 'hide-prompt' or record.get('gpu') != '1'
                or record.get('checkpoint_sha256') != info['checkpoint_sha256']
                or record.get('canonical_pretrained_sha256') != info['canonical_pretrained_sha256']
                or record.get('precision') != FULL_CL_PRECISION
                or any(type(record.get('precision', {}).get(key)) is not type(value) for key, value in FULL_CL_PRECISION.items())
                or record.get('wandb_logging_status') != 'uploaded'
                or record.get('wandb_artifact', {}).get('state') != 'COMMITTED'):
            raise ValueError('Size pilot worker differs from the selected recipe or uploaded result')
        if any(type(record.get(key)) is not type(info[key]) or record[key] != info[key]
               for key in ('vit_size', 'feature_dim', 'num_heads', 'checkpoint_tag')):
            raise ValueError('Size pilot worker checkpoint and geometry metadata disagree')
    return record


def summarize(measurements, flops, variant=None):
    """Aggregate measurement costs only, preserving missing phase energy."""
    if len(measurements) != 3:
        raise ValueError('A complete pilot requires exactly three measurements and one FLOPs pass')
    variant = variant if variant is not None else measurements[0].get('variant')
    if any(record.get('variant') != variant for record in [*measurements, flops]):
        raise ValueError('Mixed backbone sizes cannot enter one pilot aggregation')
    for record in measurements:
        validate_record(record, 'measurement', variant=variant)
    validate_record(flops, 'flops', variant=variant)
    if variant is not None:
        for key in ('initial_state_sha256', 'initial_learned_sha256', 'acceptance_summary_sha256', 'execution_source_sha256'):
            identity = measurements[0].get(key)
            if not identity or any(record.get(key) != identity for record in [*measurements[1:], flops]):
                raise ValueError('Size pilot workers must share accepted initialization and execution source pins')
    resources = {key: distribution(record['metrics'].get(key) for record in measurements)
                 for key in RESOURCE_KEYS}
    if any(record.get('sensor_errors') for record in measurements):
        raise ValueError('Resolve sensor errors before accepting the efficiency pilot')
    if resources['gpu_energy_coverage_fraction']['min'] < 0.99:
        raise ValueError('GPU energy coverage must reach 99% for every measurement')
    phases = {}
    for phase in PHASE_KEYS:
        if any(phase not in record.get('phases', {}) for record in measurements):
            raise ValueError(f'Missing measured method phase: {phase}')
        phases[phase] = {
            key: distribution((record['phases'][phase].get(key) for record in measurements),
                              allow_missing=key == 'gpu_board_energy_j')
            for key in ('wall_s', 'gpu_board_energy_j', 'gpu_energy_coverage_s')
        }
    estimate = flops.get('flops', {}).get('supported_ops_total')
    if (isinstance(estimate, bool) or not isinstance(estimate, (int, float))
            or not math.isfinite(estimate) or estimate <= 0):
        raise ValueError('The separate FLOPs pass must provide a positive partial operator estimate')
    result = {
        'status': 'pass', 'measurement_count': 3, 'flops_count': 1,
        'resources': resources, 'phases': phases,
        'accuracy': [record.get('accuracy', record.get('task_results')) for record in measurements],
        'task_diagnostics': [record.get('task_results') for record in measurements],
        'parameter_snapshots': [record.get('model_snapshots', {}) for record in measurements],
        'flops': flops['flops'], 'same_seed_repeat': True,
        'seed_variation_estimated': False,
        'alignment': {'status': 'not_applicable', 'steps': 0,
                      'reason': 'The pinned method starts classifier alignment at task two.'},
        'aggregation_scope': 'Three measurement workers only; profiler time, memory and energy excluded.',
        'logging_scope': 'Recorder totals include method execution, audits, checkpoints and inline logging; final W&B artifact upload and finish occur after sampling stops.',
        'phase_scope': 'Inclusive CUDA-synchronized phase costs cannot be summed to recover process totals.',
        'energy_scope': 'Sampled selected-GPU board energy, excluding CPU/RAM/whole-system energy.',
        'flops_scope': 'Partial supported operators during this separate worker; unsupported operators and CPU clustering are not a complete-method FLOPs estimate.',
        'accuracy_scope': 'Short TII1/prompt1 pilot with common EASE preprocessing; correctness margins do not apply.',
        'comparison_scope': 'Two training phases plus two statistics passes and inferred routing; unequal compute to one-epoch pilots.',
        'wandb_urls': [record.get('wandb_url') for record in measurements],
        'flops_wandb_url': flops.get('wandb_url'),
    }
    if variant is not None:
        result['variant'] = variant
        result['protocol'] = measurements[0]['protocol']
    return result


def run_native(arguments):
    # Native main owns fresh subprocess creation, telemetry, W&B and Slack.
    # Keeping this import inside execution preserves preparation's stdlib scope.
    from scripts import run_hide_validation
    return run_hide_validation.main(arguments)


def check_acceptance(path, initial_state, variant=None):
    from scripts.hide_protocol import validate_efficiency_acceptance
    if variant is not None:
        return validate_efficiency_acceptance(path, initial_state=initial_state, variant=variant)
    return validate_efficiency_acceptance(path, initial_state=initial_state)


def execute(options):
    if options.acceptance_summary is None or options.initial_state is None:
        raise ValueError('Efficiency execution requires --acceptance-summary and --initial-state')
    acceptance_path = options.acceptance_summary.expanduser().resolve()
    initial_state = options.initial_state.expanduser().resolve()
    variant = getattr(options, 'variant', None)
    # Check the condition before reading credentials or creating output folders.
    if variant is None:
        acceptance = check_acceptance(acceptance_path, initial_state)
        plan = json.loads(PLAN_PATH.read_text())
    else:
        from scripts.hide_size_protocol import load_plan
        acceptance = check_acceptance(acceptance_path, initial_state, variant=variant)
        plan = load_plan(variant)
    require_approved_scope(plan, acceptance, variant=variant)
    profile = configure_environment()
    if profile['gpu'] != '1':
        raise ValueError('The approved HiDe-Prompt efficiency pilot uses physical GPU1')
    checkpoint = Path(plan['common']['checkpoint_file'])
    if file_hash(checkpoint) != plan['common']['checkpoint_sha256']:
        raise ValueError('The pinned pretrained checkpoint changed after native acceptance')
    name = 'hide-efficiency-pilot-' + (variant + '-' if variant is not None else '') + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (options.result_dir if options.result_dir is not None else options.output / name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    execution = {
        'status': 'running', 'model': 'hide-prompt', 'name': name, 'output': str(output),
        'protocol': plan['efficiency'], 'acceptance_summary': str(acceptance_path),
        'acceptance_summary_sha256': file_hash(acceptance_path),
        'initial_state': str(initial_state), 'initial_state_sha256': file_hash(initial_state),
        'checkpoint_sha256': plan['common']['checkpoint_sha256'],
        'measurements': [], 'flops': None, 'workers': [],
        'execution_source_sha256': {str(Path(__file__).relative_to(ROOT)): file_hash(__file__)},
    }
    if variant is not None:
        from scripts.hide_size_protocol import SOURCE_FILES
        execution.update(variant=variant, checkpoint_tag=plan['common']['checkpoint_tag'],
                         canonical_pretrained_sha256=plan['common']['canonical_pretrained_sha256'])
        execution['execution_source_sha256'].update({source: file_hash(ROOT / source) for source in SOURCE_FILES})
    write_json(output / 'execution.json', execution)
    try:
        for index, pass_name in enumerate(('measurement', 'measurement', 'measurement', 'flops'), start=1):
            directory = output / (f'measurement{index}' if pass_name == 'measurement' else 'flops')
            arguments = ['--execute', '--mode', 'efficiency', '--pass', pass_name,
                         '--result-dir', str(directory), '--initial-state', str(initial_state),
                         '--acceptance-summary', str(acceptance_path), '--timeout', str(options.timeout)]
            if variant is not None:
                arguments.extend(['--variant', variant, '--checkpoint', str(checkpoint)])
            code = run_native(arguments)
            result = directory / 'result.json'
            record = json.loads(result.read_text()) if result.is_file() else {'status': 'failed', 'pass': pass_name}
            completion_path = directory / 'completion.json'
            completion = json.loads(completion_path.read_text()) if completion_path.is_file() else {}
            execution['workers'].append({'directory': str(directory.relative_to(output)), 'exit_code': code,
                                         'result_sha256': file_hash(result) if result.is_file() else None,
                                         'completion': completion})
            if pass_name == 'measurement':
                execution['measurements'].append(record)
            else:
                execution['flops'] = record
            write_json(output / 'execution.json', execution)
            if code != 0:
                raise RuntimeError('Native efficiency worker failed; partial local results are preserved')
            validate_record(record, pass_name, variant=variant)
            if variant is not None and (record.get('initial_state_sha256') != execution['initial_state_sha256']
                    or record.get('acceptance_summary_sha256') != execution['acceptance_summary_sha256']
                    or record.get('initial_learned_sha256') != acceptance.get('initial_learned_sha256')
                    or record.get('execution_source_sha256') != acceptance.get('execution_source_sha256')):
                raise ValueError('Size pilot worker changed accepted initialization or source pins')
            print(f'HiDe-Prompt {pass_name} saved: {record.get("wandb_url") or directory}', flush=True)
        execution['summary'] = summarize(execution['measurements'], execution['flops'], variant=variant)
        execution['summary']['launcher_wall_s_including_logging'] = distribution(
            (worker['completion'].get('wall_s') for worker in execution['workers'][:3]), allow_missing=True)
        execution['summary']['launcher_wall_scope'] = (
            'Native launcher timer includes worker startup, computation, final W&B upload/finish and completion notification; '
            'this interval has no corresponding sampled GPU-energy estimate.')
        execution['status'] = 'completed'
        write_json(output / 'summary.json', execution['summary'])
        write_json(output / 'execution.json', execution)
        print(json.dumps({'status': 'completed', 'output': str(output),
                          'measurement_runs': 3, 'flops_runs': 1}), flush=True)
        return 0
    except BaseException:
        execution['status'] = 'failed'
        write_json(output / 'execution.json', execution)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--variant', choices=('small', 'tiny'), default=None)
    parser.add_argument('--acceptance-summary', type=Path)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/hide_prompt/runs')
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=7200., help='Timeout in seconds per fresh worker')
    options = parser.parse_args(argv)
    if not options.execute:
        print(f'Prepared HiDe-Prompt {options.variant.capitalize() if options.variant is not None else "Base"}: seed1993, task1, TII1+prompt1, ease transforms, batch64; '
              'measurement=3, partial-FLOPs=1; execution=False')
        return 0
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        raise ValueError('Timeout must be a positive finite number')
    return execute(options)


if __name__ == '__main__':
    sys.exit(main())
