"""Report the progress of a queue of runs to Slack.

An experiment directory holds ``jobs.txt`` with one run per line, ``method condition value seed`` (the
condition is ``tome`` or ``evit`` with a training degree, or ``own`` with the matched degree), and one
sub-directory per run as ``scripts.tocom.run_cl`` writes it. A queue worker calls ``start`` and ``finish``
around every run, which keeps the state of the queue under ``reports/``, and one ``watch`` process runs
beside the workers. What is posted:

    kickoff   the whole queue and when it should end; it must be delivered before anything starts
    finish    at once: a run that failed (how far it got, the end of its log), and the tables of the
              finished queue
    watch     at once: a run none of whose records has grown for too long, and a queue left with no worker;
              every two hours: a digest with the state of the queue, the final class-IL mean of the runs
              finished since the last one next to that of the same method trained without reduction, and
              where the running ones are
    say       a free line, for the preflight and for a worker that stops
    status    the state of the queue on the terminal; nothing is posted

``kickoff --per-run`` also posts the start (content, expected time) and the end (duration, headline) of
every run, which suits a short queue; ``--seed-tables`` a table whenever a seed is finished.

Expected times start from ``estimates.json`` (seconds per ``method condition value``) and follow the runs
that have finished. The webhook is read from a private credentials file by this process only: it is never
printed or recorded and no training process receives it. ``--dry-run`` prints instead of posting.

Examples (from the repository root):
    uv run --no-sync python -m scripts.tocom.report --experiment <dir> --credentials <file> kickoff --workers 6
    uv run --no-sync python -m scripts.tocom.report --experiment <dir> --credentials <file> watch
    uv run --no-sync python -m scripts.tocom.report --experiment <dir> status
"""

import argparse
from collections import namedtuple
from contextlib import contextmanager
import csv
import fcntl
import heapq
import json
import os
from pathlib import Path
import re
import statistics
import time
import urllib.request

from scripts.cl_runtime import read_credentials
from scripts.tocom.tracking import identity

NAMES = {'er': 'ER', 'lwf-mc': 'LwF-MC', 'coda-prompt': 'CODA-Prompt', 'cps-prompt': 'CPS-Prompt', 'rep-prompt': 'REP',
         'ease': 'EASE', 'sd-lora': 'SD-LoRA', 'inf-lora': 'InfLoRA', 'ewc-lora': 'EWC-LoRA'}
SHORT = {'coda-prompt': 'CODA', 'cps-prompt': 'CPS'}
OPERATORS = {'tome': 'ToMe', 'evit': 'EViT', 'cls-prune': 'cls-prune', 'pitome': 'PiToMe'}
# Four characters each, for the fixed-width tables.
CODES = {'tome': 'ToMe', 'evit': 'EViT', 'cls-prune': 'Prun', 'pitome': 'PiTM'}
# Operators a model trained without the plug-in is inferred with, and the table columns of a queue that trains
# with no operator.
GRID = ('tome', 'evit')
WEBHOOK = 'https://hooks.slack.com/'
# A run is marked in the digest once it has taken this many times its expected duration, plus a margin.
LATE, MARGIN = 1.5, 600
# A run is stalled when none of its records has grown for this long. A long run gets a share of its expected
# duration instead: the evaluation grid after its last tasks is silent for longer.
STALL, STALL_SHARE = 1800, .3
# No run is running although some are left: said once this has lasted, as starting workers look the same.
IDLE = 600
ROWS = 40


class Job(namedtuple('Job', 'method condition value seed')):
    """One run of a queue: ``value`` is a training degree, the name of a budget path (``P1``) for a run
    that trains each task at its own degree, ``J<degree>`` for joint training at that degree, or ``MIX``
    for training at mixed degrees."""

    @classmethod
    def parse(cls, fields):
        method, condition, value, seed = fields
        named = re.fullmatch(r'P\d+|J\d+|MIX', value) is not None
        if method not in NAMES or condition not in ('own', *OPERATORS) or (named and condition == 'own'):
            raise ValueError(f'Unknown job: {" ".join(fields)}')
        return cls(method, condition, value if named else int(value), int(seed))

    @property
    def id(self):
        if self.condition == 'own':
            return f'{self.method}-own-m{self.value}-seed{self.seed}'
        if self.kind in ('switch', 'joint', 'mix'):
            return f'{self.method}-{self.condition}-{self.value}-seed{self.seed}'
        return f'{self.method}-{self.condition}-r{self.value}-seed{self.seed}'

    @property
    def key(self):
        return f'{self.method} {self.condition} {self.value}'

    @property
    def kind(self):
        if isinstance(self.value, str):
            return {'J': 'joint', 'M': 'mix'}.get(self.value[0], 'switch')
        return 'none' if not self.value else 'own' if self.condition == 'own' else 'plugin'

    @property
    def title(self):
        training = {'none': '압축 없이 학습', 'own': f'자체 압축으로 학습(ToMe {self.value}과 같은 FLOPs)',
                    'plugin': f'{OPERATORS.get(self.condition)} 학습 압축도 {self.value}',
                    'switch': f'{OPERATORS.get(self.condition)} 예산 경로 {self.value}',
                    'joint': f'{OPERATORS.get(self.condition)} {str(self.value)[1:]}로 joint 학습',
                    'mix': f'{OPERATORS.get(self.condition)} 0~24 섞어 학습'}[self.kind]
        return f'{NAMES[self.method]} · {training} · seed {self.seed}'

    @property
    def train(self):
        """The training condition in at most seven plain characters, for fixed-width tables."""
        return {'none': 'none', 'own': f'own {self.value}', 'plugin': f'{CODES.get(self.condition)} {self.value}',
                'switch': f'{CODES.get(self.condition)} {self.value}', 'joint': f'{CODES.get(self.condition)} {self.value}',
                'mix': f'{CODES.get(self.condition)} MIX'}[self.kind]

    @property
    def brief(self):
        return {'none': '압축 없음', 'own': f'자체 {self.value}', 'plugin': f'{OPERATORS.get(self.condition)} {self.value}',
                'switch': f'{OPERATORS.get(self.condition)} 경로 {self.value}',
                'joint': f'{OPERATORS.get(self.condition)} joint {str(self.value)[1:]}', 'mix': f'{OPERATORS.get(self.condition)} 섞음'}[self.kind]

    @property
    def inference(self):
        # A model trained with the plug-in is inferred with its own operator; the others with every operator.
        own = self.kind in ('plugin', 'switch', 'mix') or (self.kind == 'joint' and self.value != 'J0')
        return OPERATORS[self.condition] if own else '·'.join(OPERATORS[name] for name in GRID)


def duration(seconds):
    minutes = max(round(seconds / 60), 1)
    return f'{minutes}분' if minutes < 60 else f'{minutes // 60}시간 {minutes % 60:02d}분'


def clock(moment, now):
    day = '' if time.localtime(moment)[:3] == time.localtime(now)[:3] else '%m/%d '
    return time.strftime(f'{day}%H:%M', time.localtime(moment))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


class Queue:
    """The jobs of an experiment directory and what is known about each: started, finished, how long."""

    def __init__(self, root):
        self.root, self.reports = Path(root), Path(root) / 'reports'
        self.jobs = {job.id: job for job in (Job.parse(line.split()) for line in
                                             (self.root / 'jobs.txt').read_text().splitlines() if line.strip())}
        self.priors = read_json(self.root / 'estimates.json', {})
        self.plan = read_json(self.reports / 'kickoff.json', {})

    @contextmanager
    def locked(self):
        (self.reports / 'runs').mkdir(parents=True, exist_ok=True)
        with (self.reports / 'lock').open('w') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            self.states = {path.stem: json.loads(path.read_text()) for path in (self.reports / 'runs').glob('*.json')}
            yield self

    def save(self, name):
        path = self.reports / 'runs' / f'{name}.json'
        path.with_suffix('.tmp').write_text(json.dumps(self.states[name], indent=2) + '\n')
        os.replace(path.with_suffix('.tmp'), path)

    def once(self, marker):
        """True the first time it is asked for this marker; call it with the lock held."""
        path = self.reports / f'sent-{marker}'
        if path.exists():
            return False
        path.write_text(f'{time.time()}\n')
        return True

    def of(self, status):
        return [name for name, state in self.states.items() if state.get('status') == status]

    def expected(self, job):
        """Seconds a run should take and what the figure rests on, or (None, None)."""
        done = [(self.jobs[name], self.states[name]['seconds']) for name in self.of('completed') if name in self.jobs]
        same = [seconds for other, seconds in done if other.key == job.key]
        if same:
            return statistics.mean(same), '같은 설정 실측'
        prior = self.priors.get(job.key)
        if prior is None:
            return None, None
        ratios = [seconds / self.priors[other.key] for other, seconds in done if self.priors.get(other.key)]
        return (prior * statistics.median(ratios), '파일럿 기준, 실측 반영') if ratios else (prior, '파일럿 기준')

    def end(self, now):
        """When the queue should be empty, or None while some run has no expected time.

        The waiting runs go, in queue order, to whichever worker is free first, and the queue ends with the
        last of them. Dividing the remaining work by the number of workers would end too early: the workers
        do not finish together.
        """
        running = self.of('running')
        slots = [max(self.states[name]['expected'] - (now - self.states[name]['started']), 60)
                 if self.states[name].get('expected') else None for name in running]
        waiting = [self.expected(job)[0] for name, job in self.jobs.items() if name not in self.states]
        if None in slots or None in waiting:
            return None
        slots += [0.] * max((self.plan.get('workers') or 1) - len(slots), 0)
        heapq.heapify(slots)
        for seconds in waiting:
            heapq.heappush(slots, heapq.heappop(slots) + seconds)
        return now + max(slots)

    def footer(self, now):
        end = self.end(now)
        counts = (f'완료 {len(self.of("completed"))} · 실행 중 {len(self.of("running"))} · '
                  f'대기 {len(self.jobs) - len(self.states)} · 실패 {len(self.of("failed"))}')
        return f'진행: {counts}' + (f' · 전체 예상 종료 {clock(end, now)}' if end else '')

    def run(self, name):
        """Config and last task record of a run directory, or (None, None) if it has no evaluated task."""
        config, records = read_json(self.root / name / 'config.json'), read_json(self.root / name / 'metrics.json')
        return (config, records[-1]) if config and records else (None, None)

    def position(self, name, brief=False):
        """How far a run has got: the tasks it has evaluated and the last epoch it has recorded."""
        records = read_json(self.root / name / 'metrics.json', [])
        try:
            last = json.loads((self.root / name / 'progress.jsonl').read_text().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            return '기록 없음' if brief else f'평가한 태스크 {len(records)}개'
        where = f'태스크 {last["task"] + 1} 에포크 {last["epoch"] + 1}'
        return where if brief else f'평가한 태스크 {len(records)}개, 마지막 기록 {where}'


def mean(values):
    return sum(values) / len(values)


def shifts(config, record):
    """Per inference operator, what the two outermost inference degrees do: (operator, degree, off, on)."""
    targets = config['targets']
    for name in config['eval_operators']:
        entry = record.get('operators', {}).get(name)
        if not entry or name not in entry['on']:
            continue
        trained = (config['compensations'] or {}).get(name, (None, (config['degree'],)))[1]
        for target in (max(targets), min(targets)):
            # Skip the degree the model was trained at: there the bank has nothing to correct.
            if not all(degree == target for degree in trained):
                yield name, target, mean(entry['off'][str(target)]), mean(entry['on'][name][str(target)])


def headline(config, record):
    parts = [f'최종 정확도 {mean(record["class_il"]):.1f}%']
    several = len(config['eval_operators']) > 1
    for name, target, off, on in shifts(config, record):
        label = f'{OPERATORS.get(name, name)} ' if several else ''
        parts.append(f'{label}추론 {target}: {off:.1f} → 보정 {on:.1f} ({on - off:+.1f})')
    return ' · '.join(parts)


def tables(queue, seeds):
    """Two fixed-width tables over the completed runs of these seeds, averaged over the seeds.

    Each row is a method. The first table infers a model trained without reduction at the highest degree,
    the second infers a model trained at the highest degree without reduction: the accuracy as trained,
    then without and with the bank of the inference operator. Rows stay under 40 characters for a phone,
    so a queue that trains with more than two operators gets the tables once per pair of operators.
    """
    def cells(job, pick):
        values = []
        for seed in seeds:
            name = job(seed).id
            config, record = queue.run(name)
            if config and queue.states.get(name, {}).get('status') == 'completed':
                try:
                    values.append(pick(record))
                except KeyError:
                    pass
        return [mean(column) for column in zip(*values)] if values else None

    def pick(name, target):
        return lambda record: (mean(record['class_il']), mean(record['operators'][name]['off'][target]),
                               mean(record['operators'][name]['on'][name][target]))

    if any(job.kind in ('switch', 'joint', 'mix') for job in queue.jobs.values()):
        # Runs on budget paths and their references are not tabulated by training degree this way.
        return ''
    high = max((job.value for job in queue.jobs.values()), default=0)
    trained = [name for name in OPERATORS if any(job.condition == name for job in queue.jobs.values())] or list(GRID)
    text = ''
    for names in (trained[index:index + 2] for index in range(0, len(trained), 2)):
        forward = ['method   base ' + ' '.join(f'{CODES[name]} {high}'.ljust(9) for name in names)]
        backward = ['         ' + ' '.join(CODES[name].ljust(14) for name in names),
                    'method   ' + ' '.join('base off>on'.ljust(14) for _ in names)]
        for method in NAMES:
            mine = [job for job in queue.jobs.values() if job.method == method]
            if not mine:
                continue
            own = mine[0].condition == 'own'
            label = SHORT.get(method, NAMES[method]).ljust(8)
            plain = [cells(lambda seed: Job(method, 'own' if own else 'tome', 0, seed), pick(name, str(high)))
                     for name in names]
            if any(plain):
                base = next(row[0] for row in plain if row)
                forward.append(f'{label} {base:4.1f} ' + ' '.join(
                    f'{row[1]:4.1f}>{row[2]:4.1f}' if row else '-'.center(9) for row in plain))
            reduced = [cells(lambda seed: Job(method, 'own' if own else name, high, seed), pick(name, '0'))
                       for name in names]
            if any(reduced):
                backward.append(f'{label} ' + ' '.join(
                    f'{row[0]:4.1f} {row[1]:4.1f}>{row[2]:4.1f}' if row else '-'.center(14) for row in reduced))
        if len(forward) > 1:
            text += (f'압축 없이 학습 → 추론 {high} (base: 학습한 그대로, 보정 끔>켬)\n```\n' + '\n'.join(forward)
                     + '\n```\n')
        if len(backward) > 2:
            text += (f'압축도 {high}으로 학습 → 추론 0 (base: 학습한 그대로, 보정 끔>켬)\n```\n' + '\n'.join(backward)
                     + '\n```\n')
    return text


def final_rows(queue):
    """Every condition of the last task of every completed run, one row each, for the tables of the analysis."""
    for name in sorted(queue.of('completed')):
        config, record = queue.run(name)
        if not config:
            continue
        fields = identity(config)
        source = mean(record['class_il'])
        for operator, entry in record.get('operators', {}).items():
            for label, targets in (('off', entry['off']), *entry['on'].items()):
                for target, values in targets.items():
                    yield {**fields, 'infer_operator': operator, 'infer_degree': int(target),
                           'bank': '' if label == 'off' else label, 'accuracy': mean(values), 'as_trained': source}


def upload_final(queue, path, rows):
    """One Weights & Biases run holding the table of the whole queue; returns its address or None."""
    target = queue.plan.get('wandb')
    if not target or not rows:
        return None
    try:
        import wandb
        entity, _, project = target.partition('/')
        with wandb.init(entity=entity, project=project, name=f'{queue.root.name}-summary', job_type='summary',
                        dir=str(queue.reports), config={'experiment': queue.root.name, 'runs': len(queue.of('completed'))},
                        settings=wandb.Settings(console='off', save_code=False, disable_git=True, silent=True)) as run:
            run.log({'final_grid': wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])})
            run.save(str(path), base_path=str(path.parent), policy='now')
            return run.url
    except Exception as error:
        print(f'summary upload failed: {type(error).__name__}')
        return None


def post(webhook, text):
    """Deliver one message; retried, and silent about the address whatever happens."""
    request = urllib.request.Request(webhook, data=json.dumps({'text': text}).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status == 200:
                    return True
        except Exception:
            # Not reported: an HTTP error can carry the address.
            pass
        if attempt < 2:
            time.sleep(2 ** (attempt + 1))
    return False


def deliver(queue, messages, credentials, dry_run):
    """Post the messages in order and keep a record of each; False if any was not delivered."""
    webhook = None
    if messages and not dry_run:
        try:
            webhook = (read_credentials(credentials) if credentials else {}).get('SLACK_WEBHOOK_URL', '')
        except ValueError:
            webhook = ''
        if not webhook.startswith(WEBHOOK):
            webhook = None
    delivered = True
    for event, text in messages:
        if dry_run:
            print(f'--- {event}\n{text}')
            continue
        sent = bool(webhook) and post(webhook, text)
        delivered &= sent
        with (queue.reports / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'event': event, 'delivered': sent, 'text': text},
                                    ensure_ascii=False) + '\n')
    return delivered


def results(queue, names):
    """Fixed-width rows of finished runs: the final class-IL mean, and beside it that of the run of the same
    method and seed trained without reduction."""
    def accuracy(name):
        config, record = queue.run(name)
        return mean(record['class_il']) if config and queue.states.get(name, {}).get('status') == 'completed' else None

    methods, order = list(NAMES), list(queue.jobs)
    rows = ['method   train   seed  acc none']
    for name in sorted(names, key=lambda name: (methods.index(queue.jobs[name].method), queue.jobs[name].seed,
                                                order.index(name))):
        job = queue.jobs[name]
        plain = next((other.id for other in queue.jobs.values()
                      if (other.method, other.seed, other.value) == (job.method, job.seed, 0)), None)
        rows.append(f'{SHORT.get(job.method, NAMES[job.method]):8s} {job.train:7s} {job.seed:4d} ' + ' '.join(
            '   -' if value is None else f'{value:4.1f}' for value in (accuracy(name), plain and accuracy(plain))))
    return rows


def digest(queue, since, now):
    """The state of the queue, the runs finished since the last digest, and where the running ones are."""
    recent = {status: [name for name in queue.of(status) if queue.states[name]['finished'] > since and name in queue.jobs]
              for status in ('completed', 'failed')}
    lines = [f'📈 *{queue.root.name} 진행* · {duration(now - queue.plan.get("started", since))} 경과', queue.footer(now)]
    if recent['completed']:
        rows = results(queue, recent['completed'])
        more = f'\n외 {len(rows) - 1 - ROWS}개' if len(rows) - 1 > ROWS else ''
        lines.append(f'지난 {duration(now - since)} 동안 끝난 실행 {len(recent["completed"])}개 · 최종 class-IL 평균(%), '
                     'none은 같은 방법과 seed를 압축 없이 학습한 실행\n```\n' + '\n'.join(rows[:ROWS + 1]) + f'\n```{more}')
    else:
        lines.append(f'지난 {duration(now - since)} 동안 끝난 실행은 없습니다.')
    if recent['failed']:
        lines.append(f'그동안 실패 {len(recent["failed"])}개(이미 알림): ' + ', '.join(sorted(recent['failed'])))
    for name in sorted(queue.of('running'), key=lambda name: queue.states[name]['started']):
        state, elapsed = queue.states[name], now - queue.states[name]['started']
        expected = state.get('expected')
        over = ' · ⚠️ 예상 초과' if expected and elapsed > LATE * expected + MARGIN else ''
        job = queue.jobs.get(name)
        title = f'{NAMES[job.method]} · {job.brief} · seed {job.seed}' if job else name
        lines.append(f'• GPU {state["gpu"]} · {title} · {duration(elapsed)} 경과'
                     + (f'(예상 {duration(expected)})' if expected else '') + f' · {queue.position(name, brief=True)}{over}')
    return '\n'.join(lines)


def stalled(queue, now):
    """One alert for every running run none of whose records has grown for too long."""
    messages = []
    for name in queue.of('running'):
        state, folder = queue.states[name], queue.root / name
        signs = (folder / 'progress.jsonl', folder / 'metrics.json', queue.root / 'logs' / f'{name}.log')
        changed = max([state['started'], *(path.stat().st_mtime for path in signs if path.exists())])
        limit = max(STALL, STALL_SHARE * (state.get('expected') or 0))
        if now - changed > limit and state.get('stalled') != changed and name in queue.jobs:
            state['stalled'] = changed
            queue.save(name)
            expected = f' (예상 {duration(state["expected"])})' if state.get('expected') else ''
            messages.append((f'stalled {name}', f'🛑 *멈춤* · {queue.jobs[name].title} · GPU {state["gpu"]}\n'
                                               f'기록이 {duration(now - changed)}째 늘지 않습니다 · 시작한 지 '
                                               f'{duration(now - state["started"])}{expected}\n{queue.position(name)}'))
    return messages


def watch(queue, now):
    """One periodic check: the messages it calls for, and whether nothing is left to watch."""
    state = read_json(queue.reports / 'watch.json', {})
    messages = stalled(queue, now)
    left = [name for name in queue.jobs if not queue.states.get(name, {}).get('finished')]
    over = not left
    if left and not queue.of('running'):
        state.setdefault('idle', now)
        if now - state['idle'] >= IDLE:
            messages.append(('idle', f'⛔ *대기열 멈춤* · 실행 중인 것이 없는데 {len(left)}개가 남았습니다. '
                                     f'작업자가 모두 멈춘 상태입니다.\n{queue.footer(now)}'))
            over = True
    else:
        state.pop('idle', None)
    since = state.setdefault('digest', queue.plan.get('started', now))
    if not over and now - since >= queue.plan.get('digest_hours', 2) * 3600:
        messages.append(('digest', digest(queue, since, now)))
        state['digest'] = now
    (queue.reports / 'watch.json').write_text(json.dumps(state) + '\n')
    return messages, over


def kickoff(queue, args, now):
    queue.plan = {'workers': args.workers, 'wandb': args.wandb, 'started': now, 'digest_hours': args.digest_hours,
                  'per_run': args.per_run, 'seed_tables': args.seed_tables}
    (queue.reports / 'kickoff.json').write_text(json.dumps(queue.plan, indent=2) + '\n')
    (queue.reports / 'watch.json').unlink(missing_ok=True)
    jobs = list(queue.jobs.values())
    kinds = [sum(job.kind == kind for job in jobs) for kind in ('none', 'plugin', 'own', 'switch', 'joint', 'mix')]
    seeds = sorted({job.seed for job in jobs})
    end = queue.end(now)
    cadence = [f'{args.digest_hours:g}시간마다 진행 요약', *(['실행마다 시작과 완료'] if args.per_run else []),
               *(['seed 하나가 끝나면 요약표'] if args.seed_tables else [])]
    lines = [f'🚀 *{queue.root.name} 시작* · 실행 {len(jobs)}개',
             (' · '.join(text for count, text in (
                 (kinds[3], f'예산 경로로 학습 {kinds[3]}(경로 {len({job.value for job in jobs if job.kind == "switch"})}개)'),
                 (kinds[4], f'joint 학습 {kinds[4]}'), (kinds[5], f'제거량 섞어 학습 {kinds[5]}')) if count) + ' '
              if any(kinds[3:]) else
              f'압축 없이 학습 {kinds[0]} · plug-in으로 줄여 학습 {kinds[1]} · 자체 압축으로 학습 {kinds[2]} ')
             + f'(방법 {len({job.method for job in jobs})}개 × seed {"·".join(map(str, seeds))})',
             f'동시에 {args.workers}개' + (f' · 예상 {duration(end - now)} → {clock(end, now)}쯤 끝남' if end else ''),
             f'보고: {", ".join(cadence)}. 실패와 멈춤은 바로, 끝나면 전체 요약']
    if args.wandb:
        lines.append(f'W&B: https://wandb.ai/{args.wandb}')
    return [('kickoff', '\n'.join(lines))]


def start(queue, args, now):
    job = Job.parse(args.job)
    expected, basis = queue.expected(job)
    queue.states[job.id] = {'status': 'running', 'gpu': args.gpu, 'started': now, 'expected': expected,
                            'attempt': queue.states.get(job.id, {}).get('attempt', 0) + 1}
    queue.save(job.id)
    if not queue.plan.get('per_run'):
        return []
    lines = [f'▶️ *시작 {len(queue.states)}/{len(queue.jobs)}* · {job.title} · GPU {args.gpu}',
             f'추론: {job.inference}' + (f' · 예상 {duration(expected)} → {clock(now + expected, now)}쯤 끝남 ({basis})'
                                        if expected else ' · 예상 시간 미정')]
    return [(f'start {job.id}', '\n'.join(lines))]


def finish(queue, args, now):
    job = Job.parse(args.job)
    state = queue.states.setdefault(job.id, {'started': now, 'gpu': '?'})
    summary = read_json(queue.root / job.id / 'summary.json', {})
    completed = args.exit_code == 0 and summary.get('execution_status') == 'completed'
    state.update(status='completed' if completed else 'failed', finished=now, seconds=now - state['started'])
    queue.save(job.id)
    tracking = read_json(queue.root / job.id / 'tracking.json', {})
    link = f'\nW&B: {tracking["url"]}' if tracking.get('url') else ''
    took = duration(state['seconds']) + (f' (예상 {duration(state["expected"])})' if state.get('expected') else '')
    config, record = queue.run(job.id)
    messages = []
    if not (completed and config):
        try:
            lines = [line for line in (queue.root / 'logs' / f'{job.id}.log').read_text(errors='replace').splitlines()
                     if line.strip()]
            tail = f'\n마지막 로그: `{lines[-1][-300:]}`' if lines else ''
        except OSError:
            tail = ''
        messages.append((f'failed {job.id}', f'❌ *실패* · {job.title} · GPU {state["gpu"]} · {took} 뒤 종료 코드 '
                                             f'{args.exit_code}\n{queue.position(job.id)}{tail}\n'
                                             f'이 실행은 건너뛰고 대기열은 계속됩니다.\n{queue.footer(now)}{link}'))
    elif queue.plan.get('per_run'):
        messages.append((f'finish {job.id}', f'✅ *완료 {len(queue.of("completed"))}/{len(queue.jobs)}* · {job.title} · '
                                             f'{took}\n{headline(config, record)}\n{queue.footer(now)}{link}'))

    seed_jobs = [name for name, other in queue.jobs.items() if other.seed == job.seed]
    if queue.plan.get('seed_tables') and all(queue.states.get(name, {}).get('finished') for name in seed_jobs) \
            and queue.once(f'seed-{job.seed}'):
        good = sum(queue.states[name]['status'] == 'completed' for name in seed_jobs)
        messages.append((f'seed {job.seed}', f'📊 *seed {job.seed} 끝남* · {len(seed_jobs)}개 중 완료 {good}, 실패 '
                                             f'{len(seed_jobs) - good} · 최종 정확도(%)\n{tables(queue, [job.seed])}'))
    if all(queue.states.get(name, {}).get('finished') for name in queue.jobs) and queue.once('all'):
        rows = list(final_rows(queue))
        path = queue.reports / 'final_grid.csv'
        if rows:
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        address = None if args.dry_run else upload_final(queue, path, rows)
        seeds = sorted({other.seed for other in queue.jobs.values()})
        failed = ', '.join(sorted(queue.of('failed')))
        messages.append(('all', f'🏁 *{queue.root.name} 끝남* · 완료 {len(queue.of("completed"))} · 실패 '
                                f'{len(queue.of("failed"))} · {duration(now - queue.plan.get("started", now))}\n'
                                + (f'실패한 실행: {failed}\n' if failed else '')
                                + f'seed {len(seeds)}개 평균, 최종 정확도(%)\n{tables(queue, seeds)}'
                                + f'결과: {queue.root}' + (f'\nW&B 집계: {address}' if address else '')))
    return messages


def status(queue, now):
    print(queue.footer(now))
    for name in sorted(queue.of('running'), key=lambda name: queue.states[name]['started']):
        state = queue.states[name]
        expected = f' / 예상 {duration(state["expected"])}' if state.get('expected') else ''
        print(f'  GPU {state["gpu"]}  {name}: {duration(now - state["started"])} 경과{expected} · {queue.position(name)}')
    for name in sorted(queue.of('failed')):
        print(f'  실패  {name}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', type=Path, required=True, help='Directory holding jobs.txt and the runs')
    parser.add_argument('--credentials', type=Path, help='Private file holding slack_webhook_url')
    parser.add_argument('--dry-run', action='store_true', help='Print the messages instead of posting them')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('kickoff')
    command.add_argument('--workers', type=int, required=True)
    command.add_argument('--wandb', metavar='ENTITY/PROJECT')
    command.add_argument('--digest-hours', type=float, default=2, help='Hours between two progress digests')
    command.add_argument('--per-run', action='store_true', help='Also post the start and the end of every run')
    command.add_argument('--seed-tables', action='store_true', help='Also post a table whenever a seed is finished')
    for name in ('start', 'finish'):
        command = commands.add_parser(name)
        command.add_argument('job', nargs=4, metavar='FIELD', help='method condition value seed')
        command.add_argument('--gpu' if name == 'start' else '--exit-code', required=True,
                             type=str if name == 'start' else int)
    commands.add_parser('say').add_argument('text')
    commands.add_parser('status')
    command = commands.add_parser('watch')
    command.add_argument('--interval', type=float, default=300, help='Seconds between two checks')
    command.add_argument('--once', action='store_true', help='Check once and return')
    args = parser.parse_args(argv)

    while args.command == 'watch':
        with Queue(args.experiment).locked() as queue:
            messages, over = watch(queue, time.time())
        deliver(queue, messages, args.credentials, args.dry_run)
        if over or args.once:
            return
        time.sleep(args.interval)
    now = time.time()
    with Queue(args.experiment).locked() as queue:
        if args.command == 'status':
            return status(queue, now)
        messages = [('say', f'*{queue.root.name}* · {args.text}')] if args.command == 'say' else \
            {'kickoff': kickoff, 'start': start, 'finish': finish}[args.command](queue, args, now)
    # Posted with the lock released, so a slow delivery does not hold the other workers.
    if not deliver(queue, messages, args.credentials, args.dry_run) and args.command == 'kickoff':
        raise SystemExit('The kickoff message was not delivered; nothing should start')


if __name__ == '__main__':
    main()
