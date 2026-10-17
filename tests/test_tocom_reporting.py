"""The records a run mirrors to Weights & Biases and the progress messages of a queue."""

import json
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tocom import report  # noqa: E402
from scripts.tocom.tracking import Mirror, Progress, final_curves, identity, task_metrics  # noqa: E402

TARGETS = (0, 8, 16)


def config_of(method, operator='tome', degree=0, matched=None, seed=1):
    own = matched is not None
    labels = {name: [name, [0, matched], None] for name in ('tome', 'evit')} if own else None
    if matched:
        labels.update({f'{name}@0': [name, [0, 0], [name]] for name in ('tome', 'evit')})
    return {'method': method, 'vit_size': 'tiny', 'operator': operator, 'degree': degree, 'matched_degree': matched,
            'own_setting': 71 if matched else None, 'targets': list(TARGETS), 'scale': 0.1,
            'eval_operators': ['tome', 'evit'] if own or not degree else [operator],
            'source': degree if not own else None, 'compensations': labels, 'save_weights': True, 'seed': seed,
            'deterministic': True}


def record_of(config, base=60.):
    """One task record of two tasks: every lost token costs accuracy and a bank gives a part of it back."""
    record = {'task': 1, 'class_il': [base, base - 10], 'task_il': [90., 80.], 'seconds': 120., 'grid_seconds': 30.,
              'operators': {}}
    trained = config['matched_degree'] if config['matched_degree'] is not None else config['degree']
    for name in config['eval_operators']:
        off = {str(target): [base - abs(target - trained) / 4] * 2 for target in TARGETS}
        labels = list(config['compensations'] or ('tome', 'evit'))
        labels = [label for label in labels if '@' not in label or label.startswith(name)]
        record['operators'][name] = {'off': off, 'on': {
            label: {target: [value + (1 if label == name else .5) for value in values] for target, values in off.items()}
            for label in labels}}
    return record


def test_identity_names_the_three_training_kinds():
    assert identity(config_of('er'))['kind'] == 'none'
    assert identity(config_of('rep-prompt', matched=0))['train_operator'] == 'none'
    plugged = identity(config_of('sd-lora', 'evit', 16))
    assert (plugged['kind'], plugged['train_operator'], plugged['train_degree'], plugged['family']) == \
        ('plugin', 'evit', 16, 'lora')
    own = identity(config_of('cps-prompt', matched=8))
    assert (own['kind'], own['train_operator'], own['train_degree']) == ('own', 'own', 8)


def test_task_metrics_name_every_condition_and_its_gain():
    config = config_of('rep-prompt', matched=16)
    metrics = task_metrics(record_of(config))
    assert metrics['task'] == 2 and metrics['acc/class_il'] == 55 and metrics['task_acc/t02'] == 50
    assert metrics['time/task_minutes'] == 2 and metrics['time/grid_minutes'] == .5
    assert metrics['tome-off/r16'] == 60 and metrics['tome-on/r16'] == 61 and metrics['tome-gain/r16'] == 1
    assert metrics['tome-on0/r00'] == 56.5 and metrics['tome-gain0/r00'] == .5
    assert metrics['tome-cross-evit/r08'] == 58.5 and 'tome-cross-evit@0/r08' not in metrics
    sections = {key.split('/')[0] for key in metrics if '/' in key}
    assert sections == {'acc', 'task_acc', 'time', *(f'{name}-{kind}' for name in ('tome', 'evit')
                                                      for kind in ('off', 'on', 'gain', 'on0', 'gain0'))} | \
        {'tome-cross-evit', 'evit-cross-tome'}
    curves = final_curves(record_of(config))
    assert sorted(curves['evit']) == ['cross-tome', 'off', 'on', 'on0'] and curves['evit']['off'][0] == 56


def test_progress_writes_one_line_per_epoch(tmp_path):
    progress = Progress(tmp_path / 'progress.jsonl')
    for loss in (2., 4.):
        progress.step(torch.tensor(loss), .01)
    first = progress.end_epoch(0, 0)
    second = progress.end_epoch(0, 1)
    lines = [json.loads(line) for line in (tmp_path / 'progress.jsonl').read_text().splitlines()]
    assert lines == [first, second]
    assert (first['loss'], first['steps'], first['lr']) == (3., 2, .01) and second['loss'] is None
    assert progress.end_task() >= 0


def test_mirror_without_a_target_does_nothing(tmp_path):
    mirror = Mirror(tmp_path, None, 'test', config_of('er'), {})
    mirror.epoch({})
    mirror.task({})
    mirror.finish(None, {}, True)
    assert mirror.run is None and not list(tmp_path.iterdir())


def test_offline_mirror_records_the_run_and_keeps_the_random_state(tmp_path, monkeypatch):
    pytest.importorskip('wandb')
    monkeypatch.setenv('WANDB_MODE', 'offline')
    run = tmp_path / 'er-tome-r8-seed1'
    run.mkdir()
    config = config_of('er', degree=8)
    (run / 'metrics.json').write_text('[]')
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    expected = random.random(), np.random.rand(), torch.rand(1).item()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    mirror = Mirror(run, 'someone/some-project', 'test', config, {'epochs': 2})
    mirror.epoch({'task': 0, 'epoch': 0, 'loss': 1., 'lr': .1, 'seconds': 1.})
    mirror.task(record_of(config))
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected
    tracking = json.loads((run / 'tracking.json').read_text())
    assert tracking['group'] == 'er-tome-r8' and tracking['offline'] and mirror.sections >= {'acc', 'tome-gain'}
    mirror.finish(record_of(config), {'execution_status': 'completed', 'tasks': 2, 'seconds': 7200.}, True)
    assert mirror.run is not None and list((run / 'wandb').glob('offline-run-*/*.wandb'))


def test_mirror_never_stops_a_run(tmp_path, monkeypatch):
    wandb = pytest.importorskip('wandb')

    def refuse(**_):
        raise RuntimeError('no service')
    monkeypatch.setattr(wandb, 'init', refuse)
    mirror = Mirror(tmp_path, 'someone/some-project', 'test', config_of('er'), {})
    assert mirror.run is None
    mirror.task(record_of(config_of('er')))
    mirror.finish(None, {}, False)


# ----------------------------------------------------------------------------- queue reports

JOBS = [(method, condition, value) for method, conditions in (
    ('er', (('tome', 0), ('tome', 16), ('evit', 16))), ('rep-prompt', (('own', 0), ('own', 16))))
    for condition, value in conditions]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / 'tocom-test'
    (root / 'logs').mkdir(parents=True)
    (root / 'jobs.txt').write_text(''.join(f'{method} {condition} {value} {seed}\n'
                                           for seed in (1, 2) for method, condition, value in JOBS))
    (root / 'estimates.json').write_text(json.dumps({f'{m} {c} {v}': 1800 for m, c, v in JOBS}))
    now = [1_800_000_000.]
    monkeypatch.setattr(report.time, 'time', lambda: now[0])

    def call(*arguments, dry=True):
        report.main(['--experiment', str(root), *(['--dry-run'] if dry else []), *map(str, arguments)])

    def run(job, seed, seconds=1200, code=0, base=60.):
        """Start a job, write what the launcher would, and finish it ``seconds`` later."""
        method, condition, value = job
        call('start', method, condition, value, seed, '--gpu', 0)
        name = report.Job(method, condition, value, seed).id
        config = config_of(method, condition if condition != 'own' else 'tome', value if condition != 'own' else 0,
                           value if condition == 'own' else None, seed)
        if code == 0:
            (root / name).mkdir()
            (root / name / 'config.json').write_text(json.dumps(config))
            (root / name / 'metrics.json').write_text(json.dumps([record_of(config, base)]))
            (root / name / 'summary.json').write_text(json.dumps({'execution_status': 'completed'}))
        else:
            (root / 'logs' / f'{name}.log').write_text('Traceback\nRuntimeError: out of memory\n\n')
        now[0] += seconds
        call('finish', method, condition, value, seed, '--exit-code', code)

    def watch():
        """One periodic check; returns whether the watch is over. Its messages go to the output."""
        with report.Queue(root).locked() as queue:
            messages, over = report.watch(queue, now[0])
        report.deliver(queue, messages, None, True)
        return over

    call.watch = watch
    return root, now, call, run


def test_kickoff_says_what_runs_for_how_long_and_how_it_is_reported(experiment, capsys):
    root, _, call, run = experiment
    call('kickoff', '--workers', 2, '--wandb', 'someone/some-project')
    text = capsys.readouterr().out
    assert '실행 10개' in text and '압축 없이 학습 4 · plug-in으로 줄여 학습 4 · 자체 압축으로 학습 2' in text
    # Ten runs of 30 minutes on two workers.
    assert '예상 2시간 30분' in text and 'https://wandb.ai/someone/some-project' in text
    assert '보고: 2시간마다 진행 요약. 실패와 멈춤은 바로, 끝나면 전체 요약' in text
    # By default a run that goes well is not posted, and neither is a finished seed; the state is still kept.
    for job in JOBS:
        run(job, 1)
    assert capsys.readouterr().out == ''
    states = [json.loads(path.read_text()) for path in (root / 'reports/runs').glob('*.json')]
    assert len(states) == 5 and all(state['status'] == 'completed' and state['seconds'] == 1200 for state in states)


def test_the_end_of_the_queue_is_the_end_of_its_slowest_worker(tmp_path):
    root = tmp_path / 'queue'
    (root / 'reports').mkdir(parents=True)
    (root / 'jobs.txt').write_text('er tome 0 1\ner tome 8 1\ner tome 16 1\n')
    (root / 'estimates.json').write_text(json.dumps({'er tome 0': 4 * 3600, 'er tome 8': 3600, 'er tome 16': 3600}))
    (root / 'reports/kickoff.json').write_text(json.dumps({'workers': 2}))
    with report.Queue(root).locked() as queue:
        # One worker takes the long run, the other the two short ones: four hours, not six hours halved.
        assert queue.end(1000.) == 1000 + 4 * 3600
        queue.states['er-tome-r0-seed1'] = {'status': 'running', 'started': 1000., 'expected': 4 * 3600}
        assert queue.end(1000. + 1800) == 1000 + 4 * 3600
        queue.states['er-tome-r8-seed1'] = {'status': 'completed', 'started': 1000., 'finished': 8200., 'seconds': 7200.}
        # The short runs take twice their estimate, so the one still waiting is expected to as well.
        assert queue.end(8200.) == 8200 + 7200
        queue.priors.pop('er tome 16')
        assert queue.end(8200.) is None


def test_with_per_run_a_start_says_what_runs_and_for_how_long(experiment, capsys):
    root, _, call, _ = experiment
    call('kickoff', '--workers', 2, '--per-run', '--seed-tables')
    assert '보고: 2시간마다 진행 요약, 실행마다 시작과 완료, seed 하나가 끝나면 요약표.' in capsys.readouterr().out
    call('start', 'er', 'evit', 16, 1, '--gpu', 1)
    text = capsys.readouterr().out
    assert '시작 1/10' in text and 'ER · EViT 학습 압축도 16 · seed 1 · GPU 1' in text
    assert '추론: EViT · 예상 30분' in text and '파일럿 기준' in text
    call('start', 'rep-prompt', 'own', 16, 1, '--gpu', 0)
    assert '자체 압축으로 학습(ToMe 16과 같은 FLOPs)' in capsys.readouterr().out
    assert json.loads((root / 'reports/runs/er-evit-r16-seed1.json').read_text())['status'] == 'running'


def test_with_per_run_a_finish_gives_the_headline_and_the_state_of_the_queue(experiment, capsys):
    _, _, call, run = experiment
    call('kickoff', '--workers', 2, '--per-run')
    run(JOBS[0], 1)
    text = capsys.readouterr().out
    assert '완료 1/10' in text and 'ER · 압축 없이 학습 · seed 1 · 20분 (예상 30분)' in text
    # Trained at 0 and inferred at 16: four points lost, one given back by the bank of each operator.
    assert '최종 정확도 55.0%' in text and 'ToMe 추론 16: 56.0 → 보정 57.0 (+1.0)' in text and 'EViT 추론 16' in text
    assert '추론 0:' not in text
    assert '진행: 완료 1 · 실행 중 0 · 대기 9 · 실패 0 · 전체 예상 종료' in text
    run(JOBS[1], 1)
    text = capsys.readouterr().out
    assert '추론 0: 56.0 → 보정 57.0 (+1.0)' in text and '추론 16:' not in text and 'ToMe 추론' not in text


def test_expected_time_follows_the_runs_that_finished(experiment, capsys):
    _, _, call, run = experiment
    call('kickoff', '--workers', 2, '--per-run')
    run(JOBS[0], 1, seconds=900)
    capsys.readouterr()
    call('start', *JOBS[0], 2, '--gpu', 0)
    assert '예상 15분' in capsys.readouterr().out
    call('start', *JOBS[1], 2, '--gpu', 0)
    text = capsys.readouterr().out
    # No run of this setting has finished: its pilot estimate, scaled by how the finished runs compared.
    assert '예상 15분' in text and '실측 반영' in text


def test_a_failed_run_is_reported_at_once_with_the_end_of_its_log(experiment, capsys):
    root, _, _, run = experiment
    run(JOBS[2], 1, seconds=300, code=1)
    text = capsys.readouterr().out
    assert text.count('---') == 1 and '실패' in text and '종료 코드 1' in text and 'RuntimeError: out of memory' in text
    assert '실패 1' in text and '대기열은 계속됩니다' in text
    assert json.loads((root / 'reports/runs/er-evit-r16-seed1.json').read_text())['status'] == 'failed'


def test_the_digest_comes_every_two_hours_with_each_run_beside_its_unreduced_one(experiment, capsys):
    root, now, call, run = experiment
    call('kickoff', '--workers', 2)
    run(JOBS[0], 1, base=60.)
    run(JOBS[1], 1, base=50.)
    run(JOBS[4], 1, base=60.)
    call('start', *JOBS[2], 1, '--gpu', 0)
    capsys.readouterr()
    assert not call.watch() and capsys.readouterr().out == ''   # one hour in
    now[0] += 3600
    # The running run is slow but alive: its last epoch was recorded a minute ago.
    (root / 'er-evit-r16-seed1').mkdir()
    (root / 'er-evit-r16-seed1/progress.jsonl').write_text(json.dumps({'task': 0, 'epoch': 2}) + '\n')
    os.utime(root / 'er-evit-r16-seed1/progress.jsonl', (now[0] - 60, now[0] - 60))
    assert not call.watch()
    text = capsys.readouterr().out
    assert text.count('---') == 1 and '진행* · 2시간 00분 경과' in text
    assert '진행: 완료 3 · 실행 중 1 · 대기 6 · 실패 0 · 전체 예상 종료' in text
    assert '지난 2시간 00분 동안 끝난 실행 3개 · 최종 class-IL 평균(%)' in text
    rows = text.split('```')[1].strip().splitlines()
    # The unreduced run of REP with this seed has not run yet.
    assert rows == ['method   train   seed  acc none', f'{"ER":8s} {"none":7s} {1:4d} 55.0 55.0',
                    f'{"ER":8s} {"ToMe 16":7s} {1:4d} 45.0 55.0', f'{"REP":8s} {"own 16":7s} {1:4d} 55.0    -']
    assert all(len(row) <= 40 for row in rows)
    assert '• GPU 0 · ER · EViT 16 · seed 1 · 1시간 00분 경과(예상 20분) · 태스크 1 에포크 3 · ⚠️ 예상 초과' in text
    assert not call.watch() and capsys.readouterr().out == ''   # not due again
    now[0] += 7200
    call('finish', *JOBS[2], 1, '--exit-code', 1)
    capsys.readouterr()
    assert not call.watch()
    text = capsys.readouterr().out
    assert '4시간 00분 경과' in text and '끝난 실행은 없습니다' in text
    assert '그동안 실패 1개(이미 알림): er-evit-r16-seed1' in text and '• GPU' not in text


def test_a_run_whose_records_stop_growing_is_reported_once(experiment, capsys):
    root, now, call, _ = experiment
    call('start', *JOBS[0], 1, '--gpu', 1)
    folder = root / 'er-tome-r0-seed1'
    folder.mkdir()
    (folder / 'progress.jsonl').write_text(json.dumps({'task': 3, 'epoch': 7}) + '\n')
    (folder / 'metrics.json').write_text(json.dumps([{}] * 3))
    os.utime(folder / 'progress.jsonl', (now[0] + 600, now[0] + 600))
    os.utime(folder / 'metrics.json', (now[0] + 300, now[0] + 300))
    now[0] += 600 + report.STALL
    assert not call.watch() and capsys.readouterr().out == ''
    now[0] += 60
    call.watch()
    text = capsys.readouterr().out
    assert text.count('멈춤') == 1 and 'ER · 압축 없이 학습 · seed 1 · GPU 1' in text
    assert '기록이 31분째 늘지 않습니다 · 시작한 지 41분 (예상 30분)' in text
    assert '평가한 태스크 3개, 마지막 기록 태스크 4 에포크 8' in text
    now[0] += 600
    call.watch()
    assert capsys.readouterr().out == ''
    # It moves again and stops again: a new stall.
    os.utime(folder / 'progress.jsonl', (now[0], now[0]))
    now[0] += report.STALL + 1
    call.watch()
    assert capsys.readouterr().out.count('멈춤') == 1


def test_a_queue_left_without_workers_is_reported_and_ends_the_watch(experiment, capsys):
    _, now, call, run = experiment
    run(JOBS[0], 1)
    assert not call.watch()   # no run is running, but workers that are only starting look the same
    now[0] += report.IDLE
    assert call.watch()
    text = capsys.readouterr().out
    assert '대기열 멈춤' in text and '9개가 남았습니다' in text and '진행* ·' not in text
    for seed in (1, 2):
        for job in JOBS:
            if (job, seed) != (JOBS[0], 1):
                run(job, seed)
    capsys.readouterr()
    # Everything has finished: nothing to watch and nothing to say.
    assert call.watch() and capsys.readouterr().out == ''


def test_tables_come_at_the_end_and_per_seed_on_request(experiment, capsys):
    root, _, call, run = experiment
    call('kickoff', '--workers', 2, '--seed-tables')
    for job in JOBS:
        run(job, 1, base=60.)
    text = capsys.readouterr().out
    assert text.count('seed 1 끝남') == 1 and '5개 중 완료 5, 실패 0' in text and '끝남* · 완료' not in text
    table = text[text.index('seed 1 끝남'):]
    assert 'ER       55.0 56.0>57.0 56.0>57.0' in table
    assert 'ER       55.0 56.0>57.0 55.0 56.0>57.0' in table and 'REP      55.0 56.0>57.0 55.0 56.0>57.0' in table
    assert all(len(line) <= 40 for block in table.split('```')[1::2] for line in block.splitlines())
    for job in JOBS[:-1]:
        run(job, 2, base=70.)
    run(JOBS[-1], 2, code=1)
    text = capsys.readouterr().out
    assert text.count('seed 2 끝남') == 1 and text.count('tocom-test 끝남') == 1
    final = text[text.index('tocom-test 끝남'):]
    assert '완료 9 · 실패 1' in final and '실패한 실행: rep-prompt-own-m16-seed2' in final
    # Mean of the two seeds where both finished, the one finished seed otherwise.
    assert 'ER       60.0 61.0>62.0 61.0>62.0' in final and 'REP      55.0 56.0>57.0 55.0 56.0>57.0' in final
    rows = (root / 'reports/final_grid.csv').read_text().splitlines()
    assert rows[0].startswith('method,family,kind,train_operator,train_degree,seed,infer_operator,infer_degree,bank')
    assert len(rows) > 9 * 6


def test_without_options_only_the_final_tables_are_posted(experiment, capsys):
    _, _, call, run = experiment
    call('kickoff', '--workers', 2)
    capsys.readouterr()
    for seed in (1, 2):
        for job in JOBS:
            run(job, seed)
    text = capsys.readouterr().out
    assert text.count('---') == 1 and 'tocom-test 끝남* · 완료 10 · 실패 0' in text and 'seed 1 끝남' not in text


def test_the_webhook_stays_out_of_every_record(experiment, monkeypatch, capsys):
    root, _, call, _ = experiment
    secret = report.WEBHOOK + 'services/T000/B000/secret'
    credentials = root / 'private.conf'
    credentials.write_text('{\n    wandb_entity: "someone",\n    slack_webhook_url: "%s",\n}\n' % secret)
    posted = []
    monkeypatch.setattr(report, 'post', lambda webhook, text: posted.append((webhook, text)) or True)
    call('--credentials', credentials, 'kickoff', '--workers', 2, dry=False)
    call('--credentials', credentials, 'say', '사전 점검 통과', dry=False)
    assert [webhook for webhook, _ in posted] == [secret, secret] and '사전 점검 통과' in posted[1][1]
    kept = ''.join(path.read_text() for path in (root / 'reports').rglob('*') if path.is_file())
    assert 'secret' not in kept and 'secret' not in capsys.readouterr().out
    events = [json.loads(line) for line in (root / 'reports/events.jsonl').read_text().splitlines()]
    assert [event['delivered'] for event in events] == [True, True]

    monkeypatch.setattr(report, 'post', lambda webhook, text: False)
    with pytest.raises(SystemExit):
        call('--credentials', credentials, 'kickoff', '--workers', 2, dry=False)
    # Later messages are recorded as undelivered and stop nothing; so is a missing credentials file.
    call('--credentials', credentials, 'say', 'y', dry=False)
    call('--credentials', root / 'missing.conf', 'say', 'x', dry=False)
    events = [json.loads(line) for line in (root / 'reports/events.jsonl').read_text().splitlines()]
    assert [event['delivered'] for event in events[2:]] == [False, False, False]


def test_status_prints_the_queue_without_posting(experiment, capsys, monkeypatch):
    _, now, call, run = experiment
    monkeypatch.setattr(report, 'post', lambda *_: pytest.fail('status must not post'))
    run(JOBS[0], 1)
    call('start', *JOBS[1], 1, '--gpu', 1)
    now[0] += 600
    capsys.readouterr()
    call('status', dry=False)
    text = capsys.readouterr().out
    assert '완료 1 · 실행 중 1 · 대기 8' in text and 'GPU 1  er-tome-r16-seed1: 10분 경과' in text


def test_watch_checks_once_from_the_command_line(experiment, capsys):
    _, now, call, run = experiment
    call('kickoff', '--workers', 2)
    run(JOBS[0], 1)
    call('start', *JOBS[1], 1, '--gpu', 0)
    now[0] += 7200
    capsys.readouterr()
    call('watch', '--once')
    assert '진행* · 2시간 20분 경과' in capsys.readouterr().out


def test_a_queue_may_train_with_any_registered_operator(tmp_path, monkeypatch, capsys):
    from backbone.token_compression import OPERATORS
    assert set(report.OPERATORS) == set(report.CODES) == set(OPERATORS)
    jobs = [(method, condition, value) for method in ('coda-prompt', 'inf-lora')
            for condition, values in (('tome', (0, 24)), ('evit', (24,)), ('cls-prune', (16, 24)), ('pitome', (24,)))
            for value in values]
    root = tmp_path / 'tocom-wide'
    (root / 'logs').mkdir(parents=True)
    (root / 'jobs.txt').write_text(''.join(f'{method} {condition} {value} 1993\n' for method, condition, value in jobs))
    (root / 'estimates.json').write_text(json.dumps({f'{m} {c} {v}': 1800 for m, c, v in jobs}))
    now = [1_800_000_000.]
    monkeypatch.setattr(report.time, 'time', lambda: now[0])

    def call(*arguments):
        report.main(['--experiment', str(root), '--dry-run', *map(str, arguments)])

    call('kickoff', '--workers', 6)
    assert '실행 12개' in capsys.readouterr().out
    targets = (0, 8, 16, 24)
    for method, condition, value in jobs:
        job = report.Job(method, condition, value, 1993)
        assert len(job.train) <= 7 and (job.id.split('-r')[0].endswith(condition))
        call('start', method, condition, value, 1993, '--gpu', 0)
        config = dict(config_of(method, condition, value, seed=1993), targets=list(targets),
                      eval_operators=list(report.OPERATORS) if not value else [condition])
        # A bank for every operator the model is inferred with; the last method has none, as before its banks exist.
        entry = {'off': {str(target): [60. - abs(target - value) / 4] * 2 for target in targets}}
        record = {'task': 1, 'class_il': [60., 50.], 'task_il': [90., 80.], 'seconds': 120., 'grid_seconds': 30.,
                  'operators': {name: dict(entry, on={} if method == 'inf-lora' else {
                      name: {target: [value + 1 for value in values] for target, values in entry['off'].items()}})
                      for name in config['eval_operators']}}
        (root / job.id).mkdir()
        (root / job.id / 'config.json').write_text(json.dumps(config))
        (root / job.id / 'metrics.json').write_text(json.dumps([record]))
        (root / job.id / 'summary.json').write_text(json.dumps({'execution_status': 'completed'}))
        now[0] += 600
        call('finish', method, condition, value, 1993, '--exit-code', 0)
    text = capsys.readouterr().out
    final = text[text.index('tocom-wide 끝남'):]
    blocks = final.split('```')[1::2]
    # Two tables per pair of operators, each under 40 characters; a run without banks leaves its cells out.
    assert len(blocks) == 4 and all(len(line) <= 40 for block in blocks for line in block.splitlines())
    assert 'method   base ToMe 24   EViT 24' in blocks[0] and 'method   base Prun 24   PiTM 24' in blocks[2]
    assert 'CODA     55.0 54.0>55.0 54.0>55.0' in blocks[0] and 'InfLoRA' not in final.split('```')[1]
    assert 'CODA     55.0 54.0>55.0 55.0 54.0>55.0' in blocks[1] and 'CODA     55.0 54.0>55.0 55.0 54.0>55.0' in blocks[3]
    with report.Queue(root).locked() as queue:
        rows = report.results(queue, [report.Job('inf-lora', 'cls-prune', 24, 1993).id])
    assert rows[1] == 'InfLoRA  Prun 24 1993 55.0 55.0'
    metrics = task_metrics(json.loads((root / 'inf-lora-pitome-r24-seed1993/metrics.json').read_text())[0])
    assert metrics['pitome-off/r24'] == 60 and not any('pitome-on' in key for key in metrics)


def test_runs_on_budget_paths_are_named_by_their_path(tmp_path, monkeypatch, capsys):
    jobs = [(method, operator, path) for method in ('coda-prompt', 'inf-lora') for operator in ('tome', 'evit')
            for path in ('P1', 'P5')]
    root = tmp_path / 'tocom-switch'
    (root / 'logs').mkdir(parents=True)
    (root / 'jobs.txt').write_text(''.join(f'{m} {o} {p} 1993\n' for m, o, p in jobs))
    (root / 'estimates.json').write_text(json.dumps({f'{m} {o} {p}': 1800 for m, o, p in jobs}))
    now = [1_800_000_000.]
    monkeypatch.setattr(report.time, 'time', lambda: now[0])

    def call(*arguments):
        report.main(['--experiment', str(root), '--dry-run', *map(str, arguments)])

    job = report.Job.parse(['coda-prompt', 'tome', 'P5', '1993'])
    assert (job.kind, job.id, job.train, job.inference) == ('switch', 'coda-prompt-tome-P5-seed1993', 'ToMe P5', 'ToMe')
    assert '예산 경로 P5' in job.title and report.Job.parse(['er', 'tome', '8', '1']).kind == 'plugin'
    for broken in (['coda-prompt', 'own', 'P5', '1'], ['coda-prompt', 'tome', 'Q5', '1']):
        with pytest.raises(ValueError):
            report.Job.parse(broken)
    call('kickoff', '--workers', 6)
    assert '예산 경로로 학습 8(경로 2개)' in capsys.readouterr().out
    schedule = [0] * 5 + [24] * 5
    for method, operator, path in jobs:
        job = report.Job(method, operator, path, 1993)
        call('start', method, operator, path, 1993, '--gpu', 0)
        config = dict(config_of(method, operator, 0, seed=1993), degree=None, schedule=schedule, path=path,
                      sources=[0, 8, 16, 24], eval_operators=[operator],
                      compensations={f'src{d}': [operator, [d], None] for d in (0, 8, 16, 24)})
        off = {str(t): [60. - t / 4] * 2 for t in TARGETS}
        record = {'task': 1, 'degree': 0, 'class_il': [60., 50.], 'task_il': [90., 80.], 'seconds': 60., 'grid_seconds': 30.,
                  'operators': {operator: {'off': off, 'on': {f'src{d}': {t: [v[0] + 1] * 2 for t, v in off.items()}
                                                              for d in (0, 8, 16, 24)}}}}
        (root / job.id).mkdir()
        (root / job.id / 'config.json').write_text(json.dumps(config))
        (root / job.id / 'metrics.json').write_text(json.dumps([record]))
        (root / job.id / 'summary.json').write_text(json.dumps({'execution_status': 'completed'}))
        now[0] += 600
        call('finish', method, operator, path, 1993, '--exit-code', 0)
    text = capsys.readouterr().out
    assert 'tocom-switch 끝남' in text and '```' not in text[text.index('tocom-switch 끝남'):]
    with report.Queue(root).locked() as queue:
        assert report.results(queue, [job.id])[1] == 'InfLoRA  EViT P5 1993 55.0    -'
    fields = identity(config)
    assert (fields['kind'], fields['train_operator'], fields['train_degree']) == ('switch', 'evit', 'P5')
    metrics = task_metrics(record)
    assert metrics['evit-src8/r16'] == 57 and metrics['evit-off/r16'] == 56 and not any('cross' in key for key in metrics)


def test_joint_and_mixed_runs_are_named_by_their_kind(tmp_path, capsys):
    joint, mix = report.Job.parse(['inf-lora', 'evit', 'J8', '1993']), report.Job.parse(['coda-prompt', 'tome', 'MIX', '1993'])
    assert (joint.kind, joint.id, joint.train, joint.inference) == ('joint', 'inf-lora-evit-J8-seed1993', 'EViT J8', 'EViT')
    assert (mix.kind, mix.id, mix.train, mix.inference) == ('mix', 'coda-prompt-tome-MIX-seed1993', 'ToMe MIX', 'ToMe')
    assert 'joint' in joint.title and report.Job.parse(['coda-prompt', 'tome', 'J0', '1']).inference == 'ToMe·EViT'
    within = report.Job.parse(['inf-lora', 'evit', 'P3M', '1993'])
    assert (within.kind, within.id, within.train, within.inference) == ('switch-mix', 'inf-lora-evit-P3M-seed1993', 'EViT P3M', 'EViT')
    assert '예산 경로 P3 안에서' in within.title
    variant = report.Job.parse(['inf-lora', 'evit', 'P3MR', '1993'])
    assert (variant.kind, variant.id) == ('switch-mix', 'inf-lora-evit-P3MR-seed1993') and '돌아오면 섞지 않음' in variant.title
    assert report.Job.parse(['inf-lora', 'evit', 'P5MW', '1993']).kind == 'switch-mix'
    for broken in (['coda-prompt', 'own', 'J8', '1'], ['coda-prompt', 'tome', 'MIXED', '1'], ['coda-prompt', 'tome', 'J', '1']):
        with pytest.raises(ValueError):
            report.Job.parse(broken)
    root = tmp_path / 'tocom-bounds'
    (root / 'logs').mkdir(parents=True)
    lines = ['inf-lora evit J8 1993', 'coda-prompt tome J0 1993', 'coda-prompt tome MIX 1993', 'coda-prompt tome P1 1993',
             'coda-prompt tome P3M 1993', 'inf-lora evit P3M 1993', 'inf-lora evit P5M 1993']
    (root / 'jobs.txt').write_text(''.join(f'{line}\n' for line in lines))
    (root / 'estimates.json').write_text(json.dumps({line.rsplit(' ', 1)[0]: 1800 for line in lines}))
    report.main(['--experiment', str(root), '--dry-run', 'kickoff', '--workers', '2'])
    assert '예산 경로로 학습 1(경로 1개) · joint 학습 2 · 제거량 섞어 학습 1 · 예산 안에서 섞어 학습 3(경로 2개)' in capsys.readouterr().out
    base = dict(method='coda-prompt', operator='tome', seed=1, schedule=None, path=None, matched_degree=None)
    fields = identity(dict(base, degree=0, mix=[0, 8, 16, 24], joint=False))
    assert (fields['kind'], fields['train_operator'], fields['train_degree']) == ('mix', 'tome', 'MIX')
    fields = identity(dict(base, degree=8, mix=None, joint=True))
    assert (fields['kind'], fields['train_operator'], fields['train_degree']) == ('joint', 'tome', 8)
    assert identity(dict(base, degree=8))['kind'] == 'plugin'
    fields = identity(dict(base, degree=None, schedule=[0] * 5 + [24] * 5, path='P5', mix=[0, 8, 16, 24], joint=False))
    assert (fields['kind'], fields['train_operator'], fields['train_degree']) == ('switch-mix', 'tome', 'P5M')
    fields = identity(dict(base, degree=None, schedule=[0] * 5 + [24] * 5, path='P5', mix=[0, 8, 16, 24], mix_own=.5, joint=False))
    assert fields['train_degree'] == 'P5MW'


def test_a_summary_table_takes_named_and_numeric_degrees_and_can_be_uploaded_again(tmp_path, monkeypatch, capsys):
    rows = [{'kind': 'joint', 'train_degree': 8, 'accuracy': 60.}, {'kind': 'mix', 'train_degree': 'MIX', 'accuracy': 61.5},
            {'kind': 'none', 'train_degree': 0, 'accuracy': None}]
    columns, data = report.table_data(rows)
    assert columns == ['kind', 'train_degree', 'accuracy'] and [row[1] for row in data] == ['8', 'MIX', '0']
    assert [row[2] for row in data] == [60., 61.5, None]
    assert report.table_data(rows[:1])[1] == [['joint', 8, 60.]]

    root = tmp_path / 'tocom-bounds'
    (root / 'logs').mkdir(parents=True)
    (root / 'jobs.txt').write_text('inf-lora evit J8 1993\ncoda-prompt tome MIX 1993\n')
    (root / 'estimates.json').write_text(json.dumps({'inf-lora evit J8': 1800, 'coda-prompt tome MIX': 1800}))

    def call(*arguments, dry=True):
        report.main(['--experiment', str(root), *(['--dry-run'] if dry else []), *map(str, arguments)])

    call('kickoff', '--workers', 2, '--wandb', 'someone/some-project')
    with pytest.raises(SystemExit):
        call('summary', dry=False)                                          # not finished
    for method, operator, value, config in (
            ('inf-lora', 'evit', 'J8', dict(config_of('inf-lora', 'evit', 8, seed=1993), joint=True)),
            ('coda-prompt', 'tome', 'MIX', dict(config_of('coda-prompt', 'tome', 0, seed=1993), mix=[0, 8, 16, 24]))):
        job = report.Job(method, operator, value, 1993)
        call('start', method, operator, value, 1993, '--gpu', 0)
        (root / job.id).mkdir()
        (root / job.id / 'config.json').write_text(json.dumps(config))
        (root / job.id / 'metrics.json').write_text(json.dumps([record_of(config)]))
        (root / job.id / 'summary.json').write_text(json.dumps({'execution_status': 'completed'}))
        call('finish', method, operator, value, 1993, '--exit-code', 0)
    assert 'MIX' in (root / 'reports' / 'final_grid.csv').read_text()

    wandb = pytest.importorskip('wandb')
    seen = {}

    class Run:
        url = 'https://wandb.ai/someone/some-project/runs/abc123'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def log(self, values):
            seen['table'] = values['final_grid']

        def save(self, *arguments, **keywords):
            seen['saved'] = arguments[0]

    monkeypatch.setattr(wandb, 'init', lambda **keywords: seen.update(keywords) or Run())
    monkeypatch.setattr(wandb, 'Table', lambda columns, data: (columns, data))
    earlier = root / 'reports' / 'wandb' / 'run-20261015_222050-abc123'
    earlier.mkdir(parents=True)
    (root / 'reports' / 'wandb' / 'latest-run').symlink_to(earlier.name)
    capsys.readouterr()
    call('summary', dry=False)
    assert (seen['id'], seen['resume'], seen['name']) == ('abc123', 'allow', 'tocom-bounds-summary')
    columns, data = seen['table']
    degree = columns.index('train_degree')
    assert {row[degree] for row in data} == {'8', 'MIX'} and seen['saved'].endswith('final_grid.csv')
    assert 'uploaded to https://wandb.ai/someone/some-project/runs/abc123 (run abc123 continued)' in capsys.readouterr().out


def test_runs_at_another_resolution_are_named_by_their_shape(tmp_path, capsys):
    shape, plain = report.Job.parse(['coda-prompt', 'tome', '8px176', '1993']), report.Job.parse(['inf-lora', 'tome', '0px128', '1'])
    assert (shape.kind, shape.id, shape.train, shape.brief, shape.inference) == \
        ('shape', 'coda-prompt-tome-r8-px176-seed1993', '176/8', '176 px + 8', 'ToMe')
    assert 'ToMe 학습 압축도 8, 입력 176 px' in shape.title and '압축 없이, 입력 128 px' in plain.title
    assert plain.id == 'inf-lora-tome-r0-px128-seed1' and report.Job.parse(['er', 'tome', '0', '1']).kind == 'none'
    for broken in (['coda-prompt', 'own', '8px176', '1'], ['coda-prompt', 'tome', 'px176', '1'], ['coda-prompt', 'tome', '8px', '1']):
        with pytest.raises(ValueError):
            report.Job.parse(broken)
    root = tmp_path / 'idea-res-shapes-0b'
    (root / 'logs').mkdir(parents=True)
    lines = [f'{method} tome {value} 1993' for method in ('coda-prompt', 'inf-lora') for value in ('0px128', '0px144', '8px176')]
    (root / 'jobs.txt').write_text(''.join(f'{line}\n' for line in lines))
    (root / 'estimates.json').write_text(json.dumps({line.rsplit(' ', 1)[0]: 1800 for line in lines}))

    def call(*arguments):
        report.main(['--experiment', str(root), '--dry-run', *map(str, arguments)])

    call('kickoff', '--workers', 6)
    assert '해상도를 바꿔 학습 6(모양 3개) (방법' in capsys.readouterr().out
    for line in lines:
        job = report.Job.parse(line.split())
        degree, resolution = map(int, job.value.split('px'))
        call('start', *line.split(), '--gpu', 0)
        config = dict(config_of(job.method, 'tome', degree, seed=1993), targets=[degree], eval_operators=['tome'],
                      compensations=None, resolution=resolution, eval_shapes=['224:0', '176:8'], cost=.35)
        record = {'task': 1, 'class_il': [62., 52.], 'task_il': [90., 80.], 'seconds': 60., 'grid_seconds': 1.,
                  'operators': {'tome': {'off': {str(degree): [62., 52.]}, 'on': {}}},
                  'shapes': {'224:0': [70., 60.], '176:8': [64., 56.]}}
        (root / job.id).mkdir()
        for name, content in (('config.json', config), ('metrics.json', [record]), ('summary.json', {'execution_status': 'completed'})):
            (root / job.id / name).write_text(json.dumps(content))
        call('finish', *line.split(), '--exit-code', 0)
    text = capsys.readouterr().out
    assert 'idea-res-shapes-0b 끝남' in text and '```\nmethod   base' not in text
    with report.Queue(root).locked() as queue:
        assert report.results(queue, [report.Job.parse(lines[2].split()).id])[1] == 'CODA     176/8   1993 57.0    -'
    import csv
    with (root / 'reports' / 'final_grid.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row['kind'] for row in rows} == {'shape'} and {row['train_resolution'] for row in rows} == {'128', '144', '176'}
    native = [row for row in rows if row['infer_resolution'] == '224']
    assert len(rows) == 18 and len(native) == 6 and {(row['infer_degree'], row['accuracy']) for row in native} == {('0', '65.0')}
    fields = identity(config)
    assert (fields['kind'], fields['train_operator'], fields['train_degree'], fields['train_resolution']) == ('shape', 'tome', 8, 176)
    assert identity(dict(config, degree=0))['train_operator'] == 'none' and 'train_resolution' not in identity(dict(config, resolution=None))
    metrics = task_metrics(record)
    assert metrics['shape/224px-r00'] == 65 and metrics['shape/176px-r08'] == 60 and metrics['tome-off/r08'] == 57
    # Summary tables of a queue whose runs name more fields than others keep every field.
    columns, data = report.table_data([{'kind': 'none', 'accuracy': 60.}, {'kind': 'shape', 'accuracy': 61., 'train_resolution': 176}])
    assert columns == ['kind', 'accuracy', 'train_resolution'] and data == [['none', 60., None], ['shape', 61., 176]]


def test_runs_on_another_dataset_name_it_and_group_apart(tmp_path, monkeypatch):
    wandb = pytest.importorskip('wandb')
    assert 'dataset' not in identity(config_of('er')) and 'dataset' not in identity(dict(config_of('er'), dataset='seq-cifar100-224'))
    config = dict(config_of('coda-prompt', 'tome', 8), dataset='seq-imagenet-r', resolution=176)
    fields = identity(config)
    assert (fields['dataset'], fields['kind'], fields['train_resolution']) == ('seq-imagenet-r', 'shape', 176)
    seen = {}

    class Run:
        id, name, url = 'abc', 'x', None

        def define_metric(self, *arguments, **keywords):
            pass

    monkeypatch.setattr(wandb, 'init', lambda **keywords: seen.update(keywords) or Run())
    run = tmp_path / 'coda-prompt-tome-r8-px176-seed1993'
    run.mkdir()
    Mirror(run, 'someone/some-project', 'idea-res-shapes-imr', config, {})
    assert seen['group'] == 'coda-prompt-tome-r8-px176@seq-imagenet-r' and 'seq-imagenet-r' in seen['tags']
    assert seen['config']['dataset'] == 'seq-imagenet-r'


def test_a_kickoff_counts_the_references_beside_the_named_kinds(tmp_path, capsys):
    root = tmp_path / 'idea-res-shapes-imr'
    (root / 'logs').mkdir(parents=True)
    lines = [f'{method} tome {value} 1993' for method in ('coda-prompt', 'inf-lora') for value in ('24', '8px176', '0px144', '0')]
    (root / 'jobs.txt').write_text(''.join(f'{line}\n' for line in lines))
    (root / 'estimates.json').write_text(json.dumps({line.rsplit(' ', 1)[0]: 1800 for line in lines}))
    report.main(['--experiment', str(root), '--dry-run', 'kickoff', '--workers', '8'])
    assert '실행 8개\n압축 없이 학습 2 · plug-in으로 줄여 학습 2 · 해상도를 바꿔 학습 4(모양 2개) (방법 2개' in capsys.readouterr().out
