"""Prepare PROL commands and local pins; no training, credentials or service calls."""

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prol_protocol import (JOB_NAMES, PLAN, file_hash, load_preparation_plan,
                                   native_command, verify_reference_source)


def prepare(output, plan_path=PLAN, check_local=False):
    plan = load_preparation_plan(plan_path)
    if check_local:
        verify_reference_source(plan['reference_source'])
        checkpoint = plan['common']['checkpoint']
        if file_hash(checkpoint['file']) != checkpoint['sha256']:
            raise ValueError('Shared local checkpoint does not match its pinned hash')
        if not Path(plan['common']['data_root']).is_dir():
            raise ValueError('Shared data root is missing')
    jobs = []
    for name in JOB_NAMES:
        side = 'author' if name == 'validation_author' else 'native'
        tasks = 2 if name.startswith('validation') else 1
        jobs.append({'name': name, 'side': side, 'tasks': tasks, 'n_epochs_per_task': 1,
                     'status': 'proposed_not_executed',
                     'aggregate_eligible': name.startswith('measurement'),
                     'profile_flops': name == 'flops',
                     'native_training_core_command': shlex.join(native_command(plan, tasks)) if side == 'native' else None,
                     'runner_status': 'independent_cpu_oracle_only_gpu_worker_pending' if side == 'author'
                                      else 'Mammoth_core_available_instrumented_gpu_worker_pending'})
    record = {'schema_version': 1, 'method': 'prol-prompt', 'training_authorized': False,
              'experiment_executed': False, 'local_pins_checked': check_local,
              'jobs': jobs, 'environment': {key: plan['common'][key] for key in
                  ('gpu', 'gpu_uuid', 'data_root', 'cache_root', 'torch_checkpoints', 'huggingface_hub', 'clip_cache')},
              'services': {'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
                           'slack': 'existing SLACK_WEBHOOK_URL / private wandb.conf via cl_runtime',
                           'credentials_read': False, 'remote_writes_executed': False},
              'next_stage': 'review recipe; authorize GPU stage; implement and CPU-test instrumented workers before running',
              'cost_scope': plan['efficiency']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/cl_efficiency/prol_prompt/execution_plan.json')
    parser.add_argument('--check-local', action='store_true')
    args = parser.parse_args(argv)
    record = prepare(args.output, check_local=args.check_local)
    print(json.dumps({'status': 'prepared', 'jobs': len(record['jobs']), 'training_authorized': False,
                      'experiment_executed': False, 'local_pins_checked': args.check_local}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
