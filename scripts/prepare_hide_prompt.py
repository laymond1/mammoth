"""Write HiDe-Prompt candidate protocols without importing ML or running them.

This preparer has no execution option. It does not inspect credentials, mutate
the environment, contact a service, download files, or sample hardware sensors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'benchmarks/cl_efficiency/hide_prompt/validation_plan.json'


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _native_argv(protocol, candidate, output):
    common = protocol['common']
    args = ['uv', 'run', '--no-sync', 'python', 'main.py', '--model', 'hide-prompt',
            '--model_config', 'default', '--dataset', common['dataset'],
            '--dataset_config', candidate['dataset_config'],
            '--stop_after', str(candidate['tasks']),
            '--n_epochs', str(candidate['tii_epochs'] + candidate['prompt_epochs']),
            '--hide_tii_epochs', str(candidate['tii_epochs']),
            '--hide_prompt_epochs', str(candidate['prompt_epochs']),
            '--hide_ca_epochs', str(candidate['ca_epochs']),
            '--batch_size', str(candidate['batch_size']),
            '--num_workers', str(common['num_workers']), '--seed', str(common['seed']),
            '--permute_classes', '0', '--lr', str(common['prompt_lr_nominal']),
            '--hide_tii_lr', str(common['tii_lr_nominal']),
            '--hide_ca_lr', str(common['ca_lr_nominal']),
            '--hide_prompt_length', '5', '--hide_prompt_momentum', '0.01',
            '--hide_reg', '0.1', '--hide_n_centroids', '10',
            '--optim_wd', '0', '--hide_checkpoint', protocol['checkpoint']['file'],
            '--base_path', common['data_root'], '--device', 'cuda:0',
            '--nowand', '1', '--non_verbose', '1', '--debug_mode', '0',
            '--results_path', str(output / 'mammoth')]
    return args


def prepare(output, protocol_path=PROTOCOL, check_local=False):
    """Persist a fresh, unexecuted plan; optionally hash an existing local NPZ."""
    output = Path(output).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get('training_authorized') is not False:
        raise ValueError('This stage is preparation only; training_authorized must be false')
    if protocol.get('reference', {}).get('runner_implemented') is not False:
        raise ValueError('The independent reference runner is a pending next-stage deliverable')
    checkpoint = protocol['checkpoint']
    local = {'checked': False, 'download_attempted': False}
    if check_local:
        path = Path(checkpoint['file'])
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        observed = digest.hexdigest()
        if observed != checkpoint['sha256']:
            raise ValueError('Existing local checkpoint SHA-256 does not match the pinned file')
        local.update(checked=True, file=str(path), sha256=observed)

    plan = copy.deepcopy(protocol)
    plan.update(executed=False, experiment_executed=False, training_executed=False,
                image_inference_executed=False, gpu_computation_executed=False,
                downloads_executed=False, wandb_messages_sent=False, slack_messages_sent=False,
                execution_status='pending_next_stage_authorization',
                local_checkpoint=local)
    plan['candidate_commands'] = []
    for name in ('validation', 'efficiency'):
        candidate = plan[name]
        plan['candidate_commands'].append({
            'name': name, 'executed': False, 'training_authorized': False,
            'purpose': 'Configuration inspection only; not an instrumented experiment runner',
            'native_main_argv': _native_argv(plan, candidate, output / name),
            'proposed_runtime_environment': plan['common']['runtime_environment'],
            'logging_status': 'Main command has nowand=1; future approved runner must attach W&B, Slack, phase timers and resource sensors',
            'reference_runner_implemented': False})
    _write_json(output / 'plan.json', plan)
    (output / 'candidate_commands.md').write_text(
        '# HiDe-Prompt candidate configurations — not executed\n\n'
        'Training requires the next-stage decision and a verified independent reference runner. '
        'These commands only document native configuration; they do not measure the full protocol.\n\n'
        + ''.join(f"## {item['name']}\n\n```bash\n{shlex.join(item['native_main_argv'])}\n```\n\n"
                  for item in plan['candidate_commands']))
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'benchmarks/cl_efficiency/prepared/hide_prompt')
    parser.add_argument('--check-local', action='store_true',
                        help='Read and hash the pinned existing NPZ; never download a missing file')
    args = parser.parse_args(argv)
    plan = prepare(args.output, check_local=args.check_local)
    print(f"Prepared HiDe-Prompt at {args.output.resolve()}; executed={plan['executed']}; training authorization=pending")
    return 0


if __name__ == '__main__':
    sys.exit(main())
