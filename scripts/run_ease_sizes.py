"""Run the authorized four EASE size variants, gated by matched native acceptance.

The default prints scope only. Every computation requires --execute. Each variant
uses an independent author-code reference, native validation, then three fresh
measurements and one separate FLOPs worker. Previously accepted Base is reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cl_efficiency import write_json
from scripts.cl_runtime import configure_environment
from scripts.ease_size_protocol import VARIANTS, file_hash, load_plan, load_variant


def _completed_run(parent, before):
    created = {p for p in parent.iterdir() if p.is_dir()} - before
    if len(created) != 1:
        raise RuntimeError('Each variant invocation must produce exactly one run directory')
    run = created.pop()
    record = json.loads((run / 'metrics.json').read_text())
    if record.get('status') != 'completed' or record.get('wandb_logging_status') != 'uploaded':
        raise RuntimeError('Variant computation and remote logging must both complete')
    return run, record


def _call(parent, invocation, operation, execution, output):
    parent.mkdir(parents=True, exist_ok=True)
    before = {p for p in parent.iterdir() if p.is_dir()}
    execution['invocations'].append(invocation)
    write_json(output / 'execution.json', execution)
    try:
        code = operation()
        invocation['exit_code'] = code
        invocation['status'] = 'completed' if code == 0 else 'failed'
        created = {p for p in parent.iterdir() if p.is_dir()} - before
        invocation['run_directories'] = [str(p.resolve()) for p in sorted(created)]
        write_json(output / 'execution.json', execution)
        if code != 0:
            raise RuntimeError('Variant invocation failed; raw and partial logs are preserved')
        return _completed_run(parent, before)
    except BaseException as error:
        invocation['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        invocation['failure_type'] = type(error).__name__
        invocation['run_directories'] = [str(p.resolve()) for p in sorted({p for p in parent.iterdir() if p.is_dir()} - before)]
        write_json(output / 'execution.json', execution)
        raise


def execute(options):
    from scripts import run_ease_reference, run_ease_validation, run_ease_efficiency
    plan = load_plan()
    profile = configure_environment()
    if (profile['gpu'] != '1' or profile['data_root'] != plan['data_root']
            or profile['cache_root'] != plan['cache_root']):
        raise ValueError('EASE size stage must reuse the agreed GPU/data/cache profile')
    base_result = ROOT / plan['base_reference']['file']
    if file_hash(base_result) != plan['base_reference']['sha256']:
        raise ValueError('The reused Base efficiency results changed')
    variants = [load_variant(name) for name in VARIANTS]
    name = 'ease-size-pilot-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = options.output.resolve() / name
    output.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT / 'scripts/run_ease_sizes.py', ROOT / 'scripts/ease_size_protocol.py',
        ROOT / 'scripts/run_ease_reference.py', ROOT / 'scripts/ease_reference_utils.py',
        ROOT / 'scripts/run_ease_validation.py', ROOT / 'scripts/run_ease_efficiency.py',
        ROOT / 'scripts/benchmark_cl.py', ROOT / 'scripts/cl_efficiency.py',
        ROOT / 'models/ease.py', ROOT / 'backbone/ease_vit.py', ROOT / 'main.py',
        ROOT / 'utils/training.py', ROOT / 'uv.lock', ROOT / 'models/config/ease.yaml',
        ROOT / 'datasets/configs/seq-cifar100-224/ease.yaml']
    source_files += sorted((ROOT / 'models/ease_utils').glob('*.py'))
    execution = {'status': 'running', 'name': name, 'output': str(output), 'protocol': plan,
                 'base_reused': True, 'variants': {}, 'invocations': [],
                 'execution_source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in source_files}}
    write_json(output / 'execution.json', execution)
    try:
        for variant in variants:
            key = variant['name']
            saved = execution['variants'][key] = {'variant_config': variant, 'status': 'running'}
            directory = output / key
            ref_parent = directory / 'reference'
            reference_options = SimpleNamespace(variant=key, reference_source=options.reference_source,
                                                output=ref_parent, timeout=options.timeout)
            ref_dir, reference = _call(ref_parent, {'variant': key, 'stage': 'reference', 'status': 'running'},
                                      lambda: run_ease_reference.execute(reference_options, profile), execution, output)
            if (reference.get('variant') != key or reference.get('checkpoint_sha256') != variant['checkpoint_sha256']
                    or reference.get('initial_canonical_pretrained_sha256') != variant['canonical_pretrained_sha256']):
                raise RuntimeError('Completed reference does not match the pinned variant')
            reference_summary = directory / 'reference_summary.json'
            write_json(reference_summary, {'variant': key, 'variant_config': variant, 'official_reference': reference,
                                          'checkpoint_sha256': variant['checkpoint_sha256']})
            saved.update(reference_run=str(ref_dir), reference_summary=str(reference_summary), reference_wandb=reference['wandb_url'])
            write_json(output / 'execution.json', execution)
            native_parent = directory / 'native_validation'
            native_options = SimpleNamespace(variant=key, reference_summary=reference_summary,
                                             output=native_parent, timeout=options.timeout)
            native_dir, native = _call(native_parent, {'variant': key, 'stage': 'native_validation', 'status': 'running'},
                                      lambda: run_ease_validation.execute(native_options, profile), execution, output)
            saved.update(native_run=str(native_dir), native_wandb=native.get('wandb_url'), acceptance_status=native.get('acceptance_status'))
            if (native.get('variant') != key or native.get('checkpoint_sha256') != variant['checkpoint_sha256']
                    or native.get('acceptance_status') != 'pass' or native.get('acceptance', {}).get('status') != 'pass'):
                saved['status'] = 'acceptance_blocked'
                write_json(output / 'execution.json', execution)
                raise RuntimeError('Resolve same-variant acceptance before efficiency measurements')
            acceptance_summary = directory / 'acceptance_summary.json'
            write_json(acceptance_summary, {'variant': key, 'variant_config': variant,
                'checkpoint_sha256': variant['checkpoint_sha256'], 'official_reference': reference,
                'native_validation': native, 'acceptance': native['acceptance']})
            saved['acceptance_summary'] = str(acceptance_summary)
            write_json(output / 'execution.json', execution)
            efficient_parent = directory / 'efficiency'
            efficient_parent.mkdir(parents=True, exist_ok=True)
            before = {p for p in efficient_parent.iterdir() if p.is_dir()}
            invocation = {'variant': key, 'stage': 'efficiency', 'status': 'running'}
            execution['invocations'].append(invocation)
            write_json(output / 'execution.json', execution)
            code = run_ease_efficiency.main(['--variant', key, '--acceptance-summary', str(acceptance_summary),
                                              '--output', str(efficient_parent), '--execute'])
            created = {p for p in efficient_parent.iterdir() if p.is_dir()} - before
            invocation.update(status='completed' if code == 0 else 'failed', exit_code=code,
                              run_directories=[str(p.resolve()) for p in sorted(created)])
            if code != 0 or len(created) != 1:
                raise RuntimeError('Variant efficiency execution failed; partial logs preserved')
            efficient_dir = created.pop()
            efficient = json.loads((efficient_dir / 'execution.json').read_text())
            if efficient.get('status') != 'completed' or efficient.get('summary', {}).get('status') != 'pass':
                raise RuntimeError('Variant efficiency statistics are incomplete')
            saved.update(status='completed', efficiency_run=str(efficient_dir), efficiency=efficient)
            write_json(output / 'execution.json', execution)
            print(f'EASE {key}: matched validation passed; measurements=3, FLOPs=1 saved.', flush=True)
        execution['status'] = 'completed'
        execution['completed_counts'] = {'reference': 4, 'native_validation': 4, 'measurement': 12, 'flops': 4, 'total': 24}
        write_json(output / 'execution.json', execution)
        print(json.dumps({'status': 'completed', 'output': str(output), 'new_runs': 24, 'base_reused': True}), flush=True)
        return 0
    except BaseException as error:
        execution.update(status='failed', failure_type=type(error).__name__)
        if execution['invocations'] and execution['invocations'][-1]['status'] == 'running':
            execution['invocations'][-1].update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', failure_type=type(error).__name__)
        write_json(output / 'execution.json', execution)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/ease_sizes/runs')
    parser.add_argument('--reference-source', type=Path, default=Path('/tmp/cl-ease-official-20261007'))
    parser.add_argument('--timeout', type=float, default=7200.)
    options = parser.parse_args(argv)
    if not options.execute:
        print('Prepared EASE Small64/Tiny64/Small32/Tiny16: reference4 + native4 + measurement12 + FLOPs4; GPU1; Base reused; execution=False')
        return 0
    return execute(options)


if __name__ == '__main__':
    sys.exit(main())
