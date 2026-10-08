"""Prepare RainbowPrompt protocols using only the standard library.

No execution option, credential reads, ML imports, service writes or downloads.
The independent author GPU runner and pilot selection remain future work.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/execution_plan.json'
VALIDATION = ROOT / 'benchmarks/cl_efficiency/rainbow_prompt/validation_plan.json'


def prepare(output, protocol_path=PROTOCOL, validation_path=VALIDATION, check_local=False):
    protocol = json.loads(Path(protocol_path).read_text())
    validation = json.loads(Path(validation_path).read_text())
    for plan in (protocol, validation):
        if (plan.get('training_authorized') is not False or plan.get('experiment_executed') is not False
                or plan.get('method') != 'rainbow-prompt' or plan.get('source_mode') != 'official-active-path'):
            raise ValueError('RainbowPrompt preparation requires an unexecuted official-active-path protocol')
    if protocol['common'] != validation['common'] or protocol['method_settings'] != validation['method_settings']:
        raise ValueError('Validation and execution conditions disagree')
    if (protocol['reference_runner']['status'] != 'planned_not_implemented'
            or protocol['reference_runner']['executable_command'] is not None
            or protocol['pilot']['selected_option'] is not None):
        raise ValueError('The independent GPU runner and pilot decision remain pending')
    method = protocol['method_settings']
    if method['learned_gate_active'] or method['warmup_epochs'] >= method['n_epochs_per_task']:
        raise ValueError('Invalid active-path gate or unpopulated Rainbow cache configuration')
    checkpoint = protocol['common']['checkpoint']
    local = {'checked': False, 'download_attempted': False}
    if check_local:
        digest = hashlib.sha256()
        with Path(checkpoint['file']).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != checkpoint['sha256']:
            raise ValueError('Existing local checkpoint SHA-256 does not match the pin')
        local.update(checked=True, file=checkpoint['file'], sha256=digest.hexdigest())
    output = Path(output).resolve()
    args = list(protocol['native_validation']['arguments'])
    args.extend(['--base_path', protocol['common']['data_root'], '--device', 'cuda:0',
                 '--wandb_entity', protocol['common']['wandb_entity'],
                 '--wandb_project', protocol['common']['wandb_project'],
                 '--non_verbose', '1', '--results_path', str(output / 'future_native')])
    manifest = copy.deepcopy(protocol)
    manifest.update(executed=False, experiment_executed=False, training_executed=False,
                    gpu_computation_executed=False, downloads_executed=False,
                    wandb_messages_sent=False, slack_messages_sent=False,
                    local_checkpoint=local, execution_status='pending_gpu_recipe_decision',
                    validation_plan_sha256=hashlib.sha256(Path(validation_path).read_bytes()).hexdigest(),
                    execution_plan_sha256=hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest())
    manifest['candidate_native_argv'] = ['uv', 'run', '--no-sync', 'python', 'main.py', *args]
    manifest['candidate_command_limit'] = (
        'Configuration inspection only. This main command is not an instrumented experiment runner; '
        'the future runner must provide precision settings, shared caches, independent comparison, '
        'W&B, Slack, energy/memory/timers, and separate FLOPs.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'plan.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    (output / 'candidate_command.md').write_text(
        '# RainbowPrompt candidate native configuration — not executed\n\n'
        + manifest['candidate_command_limit'] + '\n\n```bash\n'
        + shlex.join(manifest['candidate_native_argv']) + '\n```\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/prepared/rainbow_prompt')
    parser.add_argument('--check-local', action='store_true', help='Hash the existing local NPZ; never download')
    args = parser.parse_args(argv)
    plan = prepare(args.output, check_local=args.check_local)
    print(f"Prepared RainbowPrompt at {args.output.resolve()}; executed={plan['executed']}; GPU recipe pending")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
