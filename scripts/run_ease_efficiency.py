"""Opt-in EASE pilot: three measurements followed by one FLOPs pass.

Each worker starts from the pinned pretrained checkpoint with seed1993. The
default prints the scope without reading credentials or starting experiments.
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

from scripts import benchmark_cl
from scripts.cl_efficiency import write_json
from scripts.cl_runtime import configure_environment

RESOURCE_KEYS = (
    'process_wall_s', 'cuda_peak_allocated_mib', 'cuda_peak_reserved_mib',
    'peak_process_tree_rss_mib', 'peak_gpu_device_memory_mib',
    'gpu_board_energy_j', 'gpu_board_energy_wh', 'gpu_energy_coverage_fraction',
    'outer_training_examples_per_s', 'evaluation_examples_per_s',
)
PHASE_KEYS = ('initialize', 'data_loading', 'begin_task', 'train_epoch', 'end_task', 'evaluation')
REQUIRED_CHECKS = ('finite_loss_and_gradients', 'frozen_backbone_gradients_absent',
                   'frozen_backbone_unchanged', 'finite_prototypes', 'prototype_and_adapter_shapes',
                   'training_protocol_complete', 'evaluation_predictions_match_accuracy',
                   'dataset_matches_reference')
SIZE_VARIANTS = ('small_fixed64', 'tiny_fixed64', 'small_relative32', 'tiny_relative16')


def require_approved_scope(plan, variant=None, acceptance=None):
    efficiency = plan['efficiency']
    if variant is not None:
        efficiency = {'dataset': plan.get('dataset', 'seq-cifar100-224'),
                      'classes_per_task': plan.get('classes_per_task', 10), **efficiency}
    expected = {'dataset': 'seq-cifar100-224', 'tasks': 1, 'classes_per_task': 10,
                'n_epochs': 1, 'batch_size': 64, 'seed': 1993, 'num_workers': 4,
                'measurement_repeats': 3, 'flops_runs': 1, 'training_authorized': True}
    if any(efficiency.get(key) != value for key, value in expected.items()):
        raise ValueError('Pilot must remain the authorized three-measurement/one-FLOPs protocol')
    if variant is not None:
        if (variant not in SIZE_VARIANTS or not acceptance
                or acceptance.get('variant') != variant
                or acceptance.get('acceptance', {}).get('status') != 'pass'
                or acceptance.get('native_validation', {}).get('acceptance_status') != 'pass'):
            raise ValueError('Native two-task acceptance must pass for the same size variant before the pilot')
    elif plan['validation']['acceptance_status'] != 'pass':
        raise ValueError('Native two-task acceptance must pass before the pilot')


def distribution(values):
    values = list(values)
    if not values or any(isinstance(value, bool) or not isinstance(value, (int, float))
                         or not math.isfinite(value) for value in values):
        raise ValueError('Resource aggregation requires finite numerical observations')
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'values': values}


def summarize(measurements, flops, variant=None):
    """Aggregate measurements only; profiling overhead cannot enter their totals."""
    if len(measurements) != 3 or flops.get('pass') != 'flops' or flops.get('status') != 'completed':
        raise ValueError('A complete pilot requires three measurements and one completed FLOPs pass')
    if any(record.get('pass') != 'measurement' or record.get('status') != 'completed' for record in measurements):
        raise ValueError('Only completed measurement passes may enter efficiency aggregation')
    for record in [*measurements, flops]:
        metrics = record['metrics']
        expected = {'tasks_completed': 1, 'outer_observe_steps': 79,
                    'outer_training_stream_examples': 5000, 'evaluation_examples': 1000}
        if any(metrics.get(key) != value for key, value in expected.items()):
            raise ValueError('Incomplete first-task protocol cannot enter aggregation')
        checks = record.get('structural_checks', {})
        if any(checks.get(key) is not True for key in REQUIRED_CHECKS) or any(value is not True for value in checks.values()):
            raise ValueError('Structural checks must pass before aggregation')
        if variant is not None:
            expected = {key: variant[key] for key in ('vit_size', 'feature_dim', 'num_heads', 'adapter_size',
                                                      'checkpoint_sha256', 'canonical_pretrained_sha256')}
            if (record.get('ease_variant') != variant['name']
                    or any(record.get(key) != value for key, value in expected.items())
                    or checks.get('canonical_pretrained_matches_checkpoint') is not True
                    or record.get('initial_canonical_pretrained_sha256') != variant['canonical_pretrained_sha256']):
                raise ValueError('Efficiency observations must match the selected accepted size variant')
    resources = {key: distribution(record['metrics'][key] for record in measurements) for key in RESOURCE_KEYS}
    phases = {phase: {
        key: distribution(record['phases'][phase][key] for record in measurements)
        for key in ('wall_s', 'gpu_board_energy_j', 'gpu_energy_coverage_s')}
        for phase in PHASE_KEYS}
    if any(record.get('sensor_errors') for record in measurements):
        raise ValueError('Resolve sensor errors before accepting the efficiency pilot')
    if resources['gpu_energy_coverage_fraction']['min'] < 0.99:
        raise ValueError('GPU energy coverage must reach 99% for every measurement')
    partial = flops.get('flops', {}).get('supported_ops_total')
    if not isinstance(partial, (int, float)) or isinstance(partial, bool) or not math.isfinite(partial) or partial <= 0:
        raise ValueError('The separate FLOPs pass must provide a positive partial operator estimate')
    return {'status': 'pass', 'measurement_count': 3, 'flops_count': 1,
            'resources': resources, 'phases': phases,
            'accuracy': [record['accuracy'] for record in measurements],
            'flops': flops['flops'], 'same_seed_repeat': True,
            'seed_variation_estimated': False,
            'aggregation_scope': 'Three measurement passes only; FLOPs-pass time, memory and energy excluded.',
            'accuracy_scope': 'One-epoch pilot accuracy; the matched twenty-epoch 1.5/3.0pp margins do not apply.'}


def execute(options):
    variant_name = getattr(options, 'variant', None)
    acceptance = None
    if variant_name is not None:
        from scripts.ease_size_protocol import load_plan, load_variant
        variant = load_variant(variant_name, verify_checkpoint=True)
        protocol = load_plan()
        if not options.acceptance_summary:
            raise ValueError('EASE size efficiency requires --acceptance-summary')
        acceptance = benchmark_cl.read_ease_size_acceptance(options.acceptance_summary, variant)
        require_approved_scope(protocol, variant_name, acceptance)
        acceptance_path = options.acceptance_summary.expanduser().resolve()
        acceptance_hash = benchmark_cl.file_hash(acceptance_path)
    else:
        if getattr(options, 'acceptance_summary', None) is not None:
            raise ValueError('--acceptance-summary requires --variant')
        plan_path = ROOT / 'benchmarks/cl_efficiency/ease/validation_plan.json'
        protocol = json.loads(plan_path.read_text())
        require_approved_scope(protocol)
        variant = {'name': 'base_fixed64', 'vit_size': 'base', 'adapter_size': 64,
                   **{key: protocol['reference'][key] for key in ('checkpoint_file', 'checkpoint_sha256')}}
    profile = configure_environment()
    if profile['gpu'] != '1':
        raise ValueError('The approved pilot uses physical GPU1')
    checkpoint = Path(variant['checkpoint_file'])
    if benchmark_cl.file_hash(checkpoint) != variant['checkpoint_sha256']:
        raise ValueError('The pretrained checkpoint changed after native acceptance')
    prefix = f'ease-efficiency-{variant_name}-' if variant_name else 'ease-efficiency-pilot-'
    name = prefix + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = options.output.resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())
    entry = next(item for item in manifest['models'] if item['model'] == 'ease')
    entry['args'].update(vit_size=variant['vit_size'], adapter_size=variant['adapter_size'], ease_checkpoint=str(checkpoint),
                         ease_pretrain_type='in21k_ft_in1k', code_optimization=0)
    if variant_name is not None:
        entry.update(ease_variant=variant_name, ease_acceptance_summary=str(acceptance_path),
                     ease_acceptance_summary_sha256=acceptance_hash)
    manifest_path = output / 'manifest.json'
    write_json(manifest_path, manifest)
    execution = {'status': 'running', 'name': name, 'output': str(output),
                 'protocol': protocol['efficiency'],
                 'native_acceptance': (acceptance['native_validation'].get('wandb_url') if acceptance is not None
                                       else protocol['validation']['wandb_url']),
                 'measurements': [], 'flops': None, 'suites': [],
                 'execution_source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__), ROOT / 'scripts/benchmark_cl.py', ROOT / 'scripts/cl_efficiency.py',
                                  ROOT / 'models/ease.py', ROOT / 'backbone/ease_vit.py', ROOT / 'uv.lock')}}
    if variant_name is not None:
        execution.update(variant=variant, acceptance_summary=str(acceptance_path),
                         acceptance_summary_sha256=acceptance_hash)
        execution['execution_source_sha256']['scripts/ease_size_protocol.py'] = benchmark_cl.file_hash(
            ROOT / 'scripts/ease_size_protocol.py')
    write_json(output / 'execution.json', execution)
    try:
        for repeat, flops_mode in enumerate(('off', 'off', 'paired'), start=1):
            directory = output / f'repeat{repeat}'
            arguments = ['--manifest', str(manifest_path), '--models', 'ease', '--output', str(directory),
                         '--epochs', '1', '--batch-size', '64', '--num-workers', '4', '--seed', '1993',
                         '--gpu', profile['gpu'], '--data-root', profile['data_root'],
                         '--wandb-entity', profile['wandb_entity'], '--wandb-project', profile['wandb_project'],
                         '--flops', flops_mode, '--execute']
            args = benchmark_cl.parser().parse_args(arguments)
            prepared, ready = benchmark_cl.prepare(args)
            before = set(directory.glob('cl-efficiency-*'))
            code = benchmark_cl.execute(args, prepared, ready)
            suites = set(directory.glob('cl-efficiency-*')) - before
            if len(suites) != 1:
                raise RuntimeError('A common runner invocation must create exactly one new suite')
            suite = suites.pop()
            results = json.loads((suite / 'results.json').read_text())
            if len(results) != 1 or results[0]['model'] != 'ease':
                raise RuntimeError('Pilot runner executed an unexpected model')
            result = results[0]
            execution['suites'].append(str(suite.relative_to(output)))
            execution['measurements'].append(result['measurement'])
            if 'flops' in result:
                execution['flops'] = result['flops']
            write_json(output / 'execution.json', execution)
            if code != 0:
                raise RuntimeError('Common runner failed; partial results are preserved')
            print(f'EASE efficiency measurement {repeat}/3 saved: {result["measurement"].get("wandb_url")}', flush=True)
        execution['summary'] = summarize(execution['measurements'], execution['flops'],
                                         variant=variant if variant_name is not None else None)
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
    parser.add_argument('--variant', choices=SIZE_VARIANTS)
    parser.add_argument('--acceptance-summary', type=Path)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args(argv)
    if options.output is None:
        options.output = ROOT / 'benchmarks/cl_efficiency' / ('ease_sizes' if options.variant else 'ease') / 'runs'
    if not options.execute:
        name = options.variant or 'Base64'
        print(f'Prepared EASE {name}: seed1993, task1, epoch1, batch64; measurement=3, FLOPs=1; execution=False')
        return 0
    return execute(options)


if __name__ == '__main__':
    sys.exit(main())
