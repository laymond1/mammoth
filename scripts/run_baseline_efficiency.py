"""Opt-in RanPAC/L2P/DualPrompt pilot, reusing the completed EASE measurements.

Each method gets three fresh measurement workers and one separate FLOPs worker.
The default prints the approved scope without credentials or model computation.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_cl
from scripts.cl_efficiency import write_json
from scripts.cl_runtime import checkpoint_path, configure_environment
from scripts.run_ease_efficiency import distribution, RESOURCE_KEYS as EASE_RESOURCE_KEYS, PHASE_KEYS

MODELS = ('ranpac', 'l2p', 'dualprompt')
CHECKPOINT = checkpoint_path('B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
CHECKPOINT_SHA256 = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
CHECKPOINT_TAG = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
EASE_RESULTS = 'benchmarks/cl_efficiency/ease/efficiency_pilot_results.json'
EASE_RESULTS_SHA256 = 'dc84f34f585ebd93e2add569acecd379170cd4a48c9c5779808201c401f41aa9'
RESOURCE_KEYS = (*EASE_RESOURCE_KEYS, 'process_tree_cpu_s')
REQUIRED_CHECKS = (
    'finite_loss_and_gradients', 'frozen_backbone_gradients_absent',
    'frozen_backbone_unchanged', 'training_protocol_complete',
    'evaluation_predictions_match_accuracy', 'dataset_matches_reference',
    'minimum_accuracy_not_collapsed', 'pretrained_backbones_match_checkpoint',
    'method_specific_final_state',
)


def require_approved_scope(plan):
    """Reject wider runs, different pretrained weights, and missing authorization."""
    expected = {'tasks': 1, 'n_epochs': 1, 'batch_size': 64, 'num_workers': 4,
                'seed': 1993, 'measurement_repeats': 3, 'flops_runs': 1}
    if plan.get('training_authorized') is not True or plan.get('models') != list(MODELS):
        raise ValueError('Pilot requires the authorized RanPAC/L2P/DualPrompt scope')
    efficiency = plan.get('efficiency', {})
    if any(type(efficiency.get(key)) is not int or efficiency[key] != value for key, value in expected.items()):
        raise ValueError('Pilot must retain the authorized task1/epoch1/seed1993 three-measurement protocol')
    checkpoint = plan.get('checkpoint', {})
    if checkpoint != {'file': CHECKPOINT, 'sha256': CHECKPOINT_SHA256, 'tag': CHECKPOINT_TAG}:
        raise ValueError('Pilot must use the exact shared EASE pretrained checkpoint')
    reference = plan.get('ease_reference', {})
    if reference != {'file': EASE_RESULTS, 'sha256': EASE_RESULTS_SHA256, 'reuse_completed_measurements': True}:
        raise ValueError('Pilot must reuse the immutable completed EASE results')


def make_pilot_manifest(manifest):
    """Preserve the complete common inventory and each method's unique settings."""
    selected = copy.deepcopy(manifest)
    entries = selected['models']
    for name in MODELS:
        matches = [entry for entry in entries if entry['model'] == name]
        if len(matches) != 1:
            raise ValueError(f'The common manifest must contain exactly one {name} entry')
        entry = matches[0]
        if entry['dataset'] != 'seq-cifar100-224' or entry.get('deferred_reason'):
            raise ValueError(f'{name} must be ready for the common dataset')
        entry['baseline_pilot'] = True
        entry['args'].update(backbone='vit', pretrained=0, pretrain_type='in21k-ft-in1k',
                             pretrained_checkpoint=CHECKPOINT, dataset_config='ease', code_optimization=0)
    return selected


def summarize(measurements, flops):
    """Use measurement observations only; profiling cost never enters statistics."""
    if len(measurements) != 3 or not isinstance(flops, dict) or flops.get('pass') != 'flops' or flops.get('status') != 'completed':
        raise ValueError('A complete pilot requires three measurements and one completed FLOPs pass')
    if any(record.get('pass') != 'measurement' or record.get('status') != 'completed' for record in measurements):
        raise ValueError('Only completed measurement passes may enter aggregation')
    for record in [*measurements, flops]:
        metrics = record.get('metrics', {})
        expected = {'tasks_completed': 1, 'outer_observe_steps': 79,
                    'outer_training_stream_examples': 5000, 'evaluation_examples': 1000}
        if any(metrics.get(key) != value for key, value in expected.items()):
            raise ValueError('Incomplete first-task protocol cannot enter aggregation')
        checks = record.get('structural_checks', {})
        if any(checks.get(key) is not True for key in REQUIRED_CHECKS) or any(value is not True for value in checks.values()):
            raise ValueError('All required structural checks must pass before aggregation')
        accuracy = record.get('accuracy', {}).get('mean_accuracy_pct')
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 20. < accuracy <= 100.:
            raise ValueError('One-epoch accuracy must clear the 20% collapse check; this is not paper reproduction')
    resources = {key: distribution(record['metrics'][key] for record in measurements) for key in RESOURCE_KEYS}
    phases = {phase: {key: distribution(record['phases'][phase][key] for record in measurements)
                      for key in ('wall_s', 'gpu_board_energy_j', 'gpu_energy_coverage_s')}
              for phase in PHASE_KEYS}
    if any(record.get('sensor_errors') for record in measurements):
        raise ValueError('Resolve sensor errors before accepting efficiency measurements')
    if resources['gpu_energy_coverage_fraction']['min'] < .99:
        raise ValueError('GPU energy coverage must reach 99% in every measurement')
    partial = flops.get('flops', {}).get('supported_ops_total')
    if isinstance(partial, bool) or not isinstance(partial, (int, float)) or not math.isfinite(partial) or partial <= 0:
        raise ValueError('The separate FLOPs pass must provide a positive partial operator estimate')
    return {'status': 'pass', 'measurement_count': 3, 'flops_count': 1,
            'resources': resources, 'phases': phases,
            'accuracy': [record['accuracy'] for record in measurements], 'flops': flops['flops'],
            'same_seed_repeat': True, 'seed_variation_estimated': False,
            'aggregation_scope': 'Three measurements only; FLOPs-pass time, memory, CPU time and energy excluded.',
            'accuracy_scope': 'One-epoch accuracy; >20% is a collapse check, not the 1.5/3.0pp reproduction policy.',
            'cpu_time_scope': 'resource.getrusage user+system CPU seconds for the process and terminated children; live-child CPU time may be omitted. Can exceed elapsed time with multiple threads/processes; not CPU energy.',
            'task_end_scope': 'Includes native method finalization; RanPAC random projection/ridge work is included without per-solver FLOPs attribution.'}


def _source_hashes():
    paths = [Path(__file__), ROOT / 'scripts/benchmark_cl.py', ROOT / 'scripts/baseline_audit.py',
             ROOT / 'scripts/cl_efficiency.py', ROOT / 'main.py', ROOT / 'utils/training.py',
             ROOT / 'backbone/vit.py', ROOT / 'uv.lock']
    for name in MODELS:
        paths.append(ROOT / f'models/{name}.py')
        paths.extend(sorted((ROOT / f'models/{name}_utils').glob('*.py')))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _read_ease_reference():
    path = ROOT / EASE_RESULTS
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EASE_RESULTS_SHA256:
        raise ValueError('The immutable EASE reference changed')
    result = json.loads(raw)
    if result.get('status') != 'completed' or result.get('pilot_status') != 'pass' or result.get('checkpoint_sha256') != CHECKPOINT_SHA256:
        raise ValueError('The reused EASE pilot must have passed with the shared checkpoint')
    # Supplement its published summary without modifying the original input.
    summary = copy.deepcopy(result['summary'])
    summary['resources']['process_tree_cpu_s'] = distribution(record['metrics']['process_tree_cpu_s'] for record in result['measurements'])
    return {'file': str(path), 'sha256': EASE_RESULTS_SHA256, 'rerun': False, 'summary': summary}


def _collect_suite(directory, before, name, execution, output):
    suites = set(directory.glob('cl-efficiency-*')) - before
    if len(suites) != 1:
        raise RuntimeError('A common runner invocation must create exactly one new suite')
    suite = suites.pop()
    execution['suites'].append(str(suite.relative_to(output)))
    records = json.loads((suite / 'results.json').read_text())
    if len(records) != 1 or records[0]['model'] != name:
        raise RuntimeError('Pilot runner executed an unexpected model')
    result = records[0]
    execution['models'][name]['measurements'].append(result['measurement'])
    if 'flops' in result:
        execution['models'][name]['flops'] = result['flops']
    write_json(output / 'execution.json', execution)
    return result


def execute(options):
    protocol = json.loads((ROOT / 'benchmarks/cl_efficiency/baselines/validation_plan.json').read_text())
    require_approved_scope(protocol)
    ease = _read_ease_reference()
    profile = configure_environment()
    if profile['gpu'] != '1':
        raise ValueError('The approved pilot uses physical GPU1')
    if hashlib.sha256(Path(CHECKPOINT).read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('The shared pretrained checkpoint changed')
    name = 'baseline-efficiency-pilot-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = options.output.resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = make_pilot_manifest(json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text()))
    manifest_path = output / 'manifest.json'
    write_json(manifest_path, manifest)
    execution = {'status': 'running', 'name': name, 'output': str(output), 'protocol': protocol,
                 'ease_reference': ease, 'models': {model: {'measurements': [], 'flops': None} for model in MODELS},
                 'suites': [], 'invocations': [], 'execution_source_sha256': _source_hashes()}
    write_json(output / 'execution.json', execution)
    try:
        for model in MODELS:
            for repeat, flops_mode in enumerate(('off', 'off', 'paired'), start=1):
                directory = output / model / f'repeat{repeat}'
                arguments = ['--manifest', str(manifest_path), '--models', model, '--output', str(directory),
                             '--epochs', '1', '--batch-size', '64', '--num-workers', '4', '--seed', '1993',
                             '--gpu', profile['gpu'], '--data-root', profile['data_root'],
                             '--wandb-entity', profile['wandb_entity'], '--wandb-project', profile['wandb_project'],
                             '--flops', flops_mode, '--execute']
                args = benchmark_cl.parser().parse_args(arguments)
                prepared, ready = benchmark_cl.prepare(args)
                before = set(directory.glob('cl-efficiency-*'))
                invocation = {'model': model, 'repeat': repeat, 'flops': flops_mode,
                              'directory': str(directory.relative_to(output)), 'status': 'running'}
                execution['invocations'].append(invocation)
                write_json(output / 'execution.json', execution)
                try:
                    code = benchmark_cl.execute(args, prepared, ready)
                except BaseException:
                    invocation['status'] = 'interrupted'
                    invocation['partial_suites'] = [str(path.relative_to(output)) for path in sorted(set(directory.glob('cl-efficiency-*')) - before)]
                    # Common-runner partial logs remain in these directories even
                    # if an interrupt occurred before a complete results.json.
                    for path in invocation['partial_suites']:
                        if (output / path / 'results.json').is_file():
                            try:
                                _collect_suite(directory, before, model, execution, output)
                            except (KeyError, ValueError, OSError, RuntimeError):
                                pass
                    write_json(output / 'execution.json', execution)
                    raise
                result = _collect_suite(directory, before, model, execution, output)
                invocation.update(status='completed' if code == 0 else 'failed', exit_code=code)
                write_json(output / 'execution.json', execution)
                if code != 0:
                    raise RuntimeError('Common runner failed; partial results are preserved')
                print(f'{model} measurement {repeat}/3 saved: {result["measurement"].get("wandb_url")}', flush=True)
            records = execution['models'][model]
            records['summary'] = summarize(records['measurements'], records['flops'])
            write_json(output / 'execution.json', execution)
        execution['summary'] = {'status': 'pass', 'new_measurement_runs': 9, 'new_flops_runs': 3,
                                'ease_rerun': False, 'models': {model: execution['models'][model]['summary'] for model in MODELS},
                                'ease_reference': ease}
        execution['status'] = 'completed'
        write_json(output / 'summary.json', execution['summary'])
        write_json(output / 'execution.json', execution)
        print(json.dumps({'status': 'completed', 'output': str(output), 'measurement_runs': 9, 'flops_runs': 3, 'ease_rerun': False}), flush=True)
        return 0
    except BaseException as error:
        execution.update(status='failed', failure_type=type(error).__name__)
        write_json(output / 'execution.json', execution)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/baselines/runs')
    options = parser.parse_args(argv)
    if not options.execute:
        print('Prepared RanPAC/L2P/DualPrompt: GPU1, seed1993, task1, epoch1, batch64, workers4; measurements=9, FLOPs=3; EASE reused; execution=False')
        return 0
    return execute(options)


if __name__ == '__main__':
    sys.exit(main())
