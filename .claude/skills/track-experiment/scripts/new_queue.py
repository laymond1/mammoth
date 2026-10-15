"""Prepare the queue scripts of a new experiment from the latest one. Nothing is started and nothing is posted.

Copies settings.sh, start.sh, worker.sh, run_job.sh, watch.sh and jobs.txt with the experiment path replaced,
and writes estimates.json from the durations the source experiment measured (the mean per
``method condition value``), keeping the source's own estimate where no run of a setting finished.
Results, logs, report state and one-off scripts (launch, preflight, gates) are not copied.

Run it from the root of the worktree that holds the experiments:
    python new_queue.py --source results/tocom-cl-main-v4 --target results/tocom-cl-main-v5
"""

import argparse
import json
from pathlib import Path
import statistics

SCRIPTS = ('settings.sh', 'start.sh', 'worker.sh', 'run_job.sh', 'watch.sh')


def run_id(method, condition, value, seed):
    """The run directory of a job line, as scripts.tocom.report names it."""
    return f'{method}-own-m{value}-seed{seed}' if condition == 'own' else f'{method}-{condition}-r{value}-seed{seed}'


def measured(source):
    """Mean seconds of the completed runs of every setting, and how many runs each rests on."""
    times = {}
    for line in (source / 'jobs.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            state = json.loads((source / 'reports' / 'runs' / f'{run_id(*fields)}.json').read_text())
        except (OSError, ValueError):
            continue
        if state.get('status') == 'completed' and state.get('seconds'):
            times.setdefault(' '.join(fields[:3]), []).append(state['seconds'])
    return {key: (round(statistics.mean(values)), len(values)) for key, values in times.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True, help='Experiment directory to copy from')
    parser.add_argument('--target', type=Path, required=True, help='New experiment directory; must not exist')
    args = parser.parse_args(argv)
    source, target = args.source, args.target
    if not (source / 'jobs.txt').is_file():
        parser.error(f'{source} holds no jobs.txt')
    if target.exists():
        parser.error(f'{target} already exists; an experiment directory is never reused')

    missing = [name for name in SCRIPTS if not (source / name).is_file()]
    if missing:
        parser.error(f'{source} lacks {", ".join(missing)}')
    old, new = source.as_posix().rstrip('/'), target.as_posix().rstrip('/')
    target.mkdir(parents=True)
    for name in SCRIPTS:
        text = (source / name).read_text()
        if name != 'settings.sh' and old not in text:
            print(f'warning: {name} does not mention {old}; check its paths by hand')
        (target / name).write_text(text.replace(old, new))
    (target / 'jobs.txt').write_text((source / 'jobs.txt').read_text())

    try:
        estimates = json.loads((source / 'estimates.json').read_text())
    except (OSError, ValueError):
        estimates = {}
    times = measured(source)
    estimates.update({key: seconds for key, (seconds, _) in times.items()})
    (target / 'estimates.json').write_text(json.dumps(dict(sorted(estimates.items())), indent=2) + '\n')

    keys = {' '.join(line.split()[:3]) for line in (target / 'jobs.txt').read_text().splitlines() if line.strip()}
    print(f'{target}: {", ".join(SCRIPTS)}, jobs.txt ({len((target / "jobs.txt").read_text().splitlines())} runs), estimates.json')
    print(f'estimates: {len(times)} settings measured in {source.name}, '
          f'{len(estimates) - len(times)} kept from its estimates.json')
    unknown = sorted(keys - set(estimates))
    if unknown:
        print(f'no expected time for: {", ".join(unknown)}')
    skipped = sorted(path.name for path in source.iterdir()
                     if path.is_file() and path.suffix in ('.sh', '.py') and path.name not in SCRIPTS)
    if skipped:
        print(f'not copied (specific to {source.name}): {", ".join(skipped)}')
    print('next: edit jobs.txt and the training arguments in run_job.sh, then preview the kickoff with --dry-run')


if __name__ == '__main__':
    main()
