"""Prepare EASE size/adapter efficiency plans without launching any experiments.

Generated manifests keep the complete source inventory required by the common
runner. Only EASE is selected in each plan. This tool has no execution option.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_cl
from scripts.cl_efficiency import write_json
from scripts.cl_runtime import configure_environment


def prepare(output):
    """Generate five instrumented runner plans; no model imports or messages."""
    output = Path(output).resolve()
    protocol = json.loads((ROOT / 'benchmarks/cl_efficiency/ease/validation_plan.json').read_text())
    canonical = json.loads((ROOT / 'benchmarks/cl_efficiency/manifest.json').read_text())
    profile = configure_environment()
    commands = []
    for variant in protocol['variants']:
        directory = output / variant['name']
        manifest = copy.deepcopy(canonical)
        entry = next(item for item in manifest['models'] if item['model'] == 'ease')
        entry['args'].update(vit_size=variant['vit_size'], adapter_size=variant['adapter_size'],
                             ease_pretrain_type='in21k_ft_in1k')
        manifest_path = directory / 'manifest.json'
        write_json(manifest_path, manifest)
        options = ['--manifest', str(manifest_path), '--models', 'ease', '--output', str(directory),
                   '--gpu', profile['gpu'], '--data-root', profile['data_root'],
                   '--wandb-entity', profile['wandb_entity'], '--wandb-project', profile['wandb_project']]
        args = benchmark_cl.parser().parse_args(options)
        plan, _ = benchmark_cl.prepare(args)
        commands.append({'variant': variant['name'], 'experiment_executed': False,
                         'instrumented_runner_argv': ['uv', 'run', '--no-sync', 'python',
                                                      'scripts/benchmark_cl.py', *options],
                         'execute_option_present': args.execute,
                         'plan': str(directory / 'plan.json'),
                         'passes': plan['jobs'][0]['passes']})
    write_json(output / 'runner_commands.json', commands)
    (output / 'runner_commands.txt').write_text(''.join(
        shlex.join(item['instrumented_runner_argv']) + '\n' for item in commands))
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/prepared/ease')
    args = parser.parse_args(argv)
    commands = prepare(args.output)
    print(f'Prepared {len(commands)} EASE variants at {args.output.resolve()}; experiment execution=False')
    return 0


if __name__ == '__main__':
    sys.exit(main())
