---
name: track-experiment
description: ToCom plug-in 실험 대기열을 W&B에 기록하고 Slack으로 진행 상황(진행률, 예상 종료, 중간 결과)을 보고하게 하는 절차. 이 프로젝트에서 실험 대기열을 새로 만들거나 시작·재시작·중지할 때, seed나 방법을 더 돌릴 때, "wandb에 올려줘", "슬랙으로 알려줘", "실험 어디까지 됐어", "언제 끝나", "보고가 안 온다" 같은 말이 나올 때는 보고 이야기가 없더라도 이 스킬을 먼저 읽을 것. 보고 코드를 새로 쓰지 않고 저장소의 scripts/tocom/report.py와 tracking.py를 쓴다. 대기열 없이 한두 번 돌리는 짧은 점검 실행, 끝난 결과의 분석이나 그림 작업에는 쓰지 않는다.
---

# 실험 기록과 진행 보고

하루 넘게 도는 실험 대기열이 스스로 W&B에 기록하고 Slack으로 보고하게 만드는 절차다. 보고는 Claude가 하지 않는다. 대기열 옆에서 도는 프로세스가 한다. 이 스킬이 맡는 일은 그 프로세스를 새 대기열에 붙이고, 시작 전에 확인하고, 물으면 현황을 읽어 주는 것이다.

## 이미 있는 것

코드는 `tocom/plugin` 브랜치에 있다(커밋 `5726e44` 이후). 지금 체크아웃에 `scripts/tocom/report.py`가 없으면 그 브랜치의 워크트리에서 일한다. 워크트리 위치는 서버마다 다르므로 `git worktree list`로 찾는다(cal-07은 `.claude/worktrees/tocom-plugin`, ai-atom은 `.worktrees/tocom-plugin`). 명령은 모두 워크트리 루트에서 `uv run --no-sync`로 실행한다.

| 파일 | 하는 일 |
|---|---|
| `scripts/tocom/tracking.py` | 실행 하나를 W&B 실행 하나로 옮긴다. 에포크마다 `progress.jsonl`도 여기서 쓴다 |
| `scripts/tocom/report.py` | Slack 보고, 예상 시간, 대기열 현황. 명령 설명은 모듈 docstring에 있다 |
| `scripts/tocom/run_cl.py --wandb ENTITY/PROJECT` | 학습 실행. 이 옵션을 주면 W&B로 옮긴다 |
| `results/<실험>/settings.sh` | 실험별 설정: 결과 폴더, 보정기, W&B 프로젝트, 비공개 설정 파일 경로, `report` 함수 |
| `results/<실험>/start.sh` | kickoff 메시지를 보내고, 작업자와 watch를 띄운다 |
| `results/<실험>/worker.sh`, `run_job.sh` | 작업자와 실행 하나. 실행 앞뒤로 `report start`, `report finish`를 부른다 |
| `results/<실험>/watch.sh` | 5분마다 확인하는 프로세스. 멈춤 알림과 2시간 요약을 보낸다 |
| `results/<실험>/estimates.json` | 설정별 예상 시간(초). 키는 `method condition value` |

보고 코드를 새로 쓰지 않는다. 이 저장소에는 실험마다 `notify` 함수를 다시 만든 흔적이 15군데 있고, 그 결과가 실험마다 다른 보고였다. 필요한 동작이 없으면 `report.py`에 더한다.

## 지킬 것과 그 이유

- **실행 폴더가 원본이다.** W&B와 Slack에 가는 값은 실행 폴더의 파일(`progress.jsonl`, `metrics.json`, `summary.json`)에 먼저 적힌 것이다. 그래서 W&B나 Slack이 끊겨도 학습은 계속되고, 분석은 항상 파일로 한다.
- **webhook은 보고 프로세스만 읽는다.** 값은 메인 작업 폴더의 `wandb.conf`에 있고 `report.py`가 `--credentials`로 읽는다. 출력하지 않고, 워크트리로 복사하지 않고, 학습 명령의 인자나 환경변수로 넘기지 않는다. 파일을 훑을 때는 `hooks.slack.com` 주소를 가려서 출력한다. 이 파일이 없는 서버에서는 Slack 보고가 나가지 않으므로, 시작을 묻기 전에 그 사실을 알린다. 다른 서버로 옮기는 것은 사용자가 한다.
- **밖으로 나가는 동작은 승인을 받고 한다.** Slack 전송, W&B 업로드, GPU 실행이 그렇다. "시작해"라는 말은 그 실험 하나의 승인이다. 중지 뒤의 재시작은 따로 묻는다.
- **시작을 묻기 전에 실제로 적용되는 설정을 적어 보인다.** 배치 크기, 에포크, 학습률, seed, 동시 실행 수, 그리고 아무도 고르지 않은 기본값이 무엇인지다. 2026-10-12에 본 실험이 기본 배치 64로 시작됐다가 27분 만에 중지됐다. 이 연구의 학습 배치는 16, 평가 배치는 256이고 `run_job.sh`에 명시한다.
- **돌고 있는 대기열에는 `status`만 쓴다.** `kickoff`는 `--dry-run`이어도 `reports/kickoff.json`을 다시 쓰고 `watch.json`을 지운다. `watch --once`도 요약 시각을 기록한다. 둘 다 돌고 있는 대기열에서 실행하면 다음 요약이 사라지거나 경과 시간이 초기화된다.

## 새 대기열에 보고 붙이기

1. **틀을 만든다.** 가장 최근 실험에서 대기열 스크립트를 가져온다.
   ```bash
   uv run --no-sync python ${CLAUDE_SKILL_DIR}/scripts/new_queue.py --source results/<최근 실험> --target results/<새 실험>
   ```
   스크립트 다섯 개와 `jobs.txt`를 복사하면서 실험 이름을 바꾸고, 최근 실험의 실측 시간으로 `estimates.json`을 만든다. 결과나 보고 상태는 복사하지 않고, 아무것도 시작하지 않는다.
2. **`jobs.txt`와 `run_job.sh`를 이번 실험에 맞춘다.** `jobs.txt`는 한 줄에 `method condition value seed`다. `run_job.sh`의 학습 인자를 하나씩 확인한다. `estimates.json`에 없는 설정은 예상 시간이 "미정"으로 나오고 전체 예상 종료도 빠진다.
3. **보낼 메시지를 미리 본다.** 아직 시작하지 않은 새 폴더에서만 한다.
   ```bash
   uv run --no-sync python -m scripts.tocom.report --experiment results/<새 실험> --dry-run kickoff --workers 6 --wandb laymond1/tocom-plugin-cl --digest-hours 2 --seed-tables
   ```
4. **설정을 적어 보이고 시작을 묻는다.** 위의 kickoff 미리보기, 실제 적용되는 설정, 예상 종료 시각을 함께 보인다.
5. **사전 점검을 한다.** 코드가 지난 실험과 달라졌으면 짧은 실행 두 개를 2태스크만 돌려 가중치 지문(`metrics.json`의 `state_sha256`)과 평가값이 기준 기록과 같은지, W&B에 올라간 값이 파일과 같은지 본다. 점검 실행은 `tocom-plugin-cl-checks` 프로젝트로 보낸다. 결과를 `report say`로 한 줄 보내면 webhook도 함께 확인된다. 예시는 `results/tocom-cl-main-v2/preflight.sh`다.
6. **시작한다.** `bash results/<새 실험>/start.sh`. kickoff 메시지가 전달되지 않으면 아무것도 시작하지 않는다. 시작 뒤 `queue.log`와 `status`로 작업자 6개와 watch가 떴는지 확인한다.

보고 빈도의 기본은 2시간마다 진행 요약, seed가 끝날 때 요약표, 실패와 멈춤과 전체 끝은 즉시다. 10개 안팎의 짧은 대기열이면 `kickoff --per-run`으로 실행마다 시작과 완료를 보내도 된다. 메시지 종류와 실제 예시는 `references/messages.md`, W&B 프로젝트와 값의 이름은 `references/wandb.md`에 있다.

## 현황을 물으면

```bash
uv run --no-sync python -m scripts.tocom.report --experiment results/<실험> status
```

완료, 실행 중, 대기, 실패 수와 전체 예상 종료, 실행 중인 것들의 경과 시간과 위치를 출력한다. 아무것도 보내지 않는다. 답할 때는 진행률과 예상 종료를 먼저 말하고, 중간 결과는 끝난 실행의 최종 class-IL 평균을 같은 방법의 압축 없는 실행과 나란히 든다. 더 자세한 값은 다음에서 읽는다.

- `results/<실험>/reports/events.jsonl`: 보낸 메시지 전부와 전달 여부
- `results/<실험>/reports/runs/<실행>.json`: 실행별 상태, 시작 시각, 걸린 시간
- `results/<실험>/<실행>/metrics.json`, `summary.json`: 결과

## 보고가 이상할 때

- **메시지가 오지 않는다.** `reports/events.jsonl`의 `delivered`와 `logs/report.log`를 본다. 2시간 요약이 없으면 `watch.sh`가 살아 있는지 본다. watch는 실패하면 20번까지 다시 뜨고 그 기록이 `queue.log`에 남는다.
- **"멈춤"이 왔다.** 그 실행의 기록 세 개(`progress.jsonl`, `metrics.json`, 로그)가 30분 넘게 그대로라는 뜻이다. 예상 시간이 100분을 넘는 실행은 예상 시간의 30%가 기준이다. 프로세스와 GPU를 확인하고 사용자에게 알린다. 스스로 죽이거나 다시 돌리지 않는다.
- **"대기열 멈춤"이 왔다.** 남은 실행이 있는데 도는 것이 없다. 작업자가 연속 2회 실패로 모두 멈춘 경우다. 원인을 찾아 알리고, 재시작은 묻는다.
- **W&B 실행이 offline으로 남았다.** 실행 폴더의 `tracking.json`에 `offline: true`가 있다. 학습은 정상이다. 나중에 그 실행 폴더의 `wandb` 디렉터리를 `wandb sync`로 올리되, 올리기 전에 묻는다.
- **예상 시간이 크게 틀린다.** 첫 seed에서는 흔하다. 같은 설정의 실행이 끝나면 그 실측으로 바뀐다. 실측이 `estimates.json`보다 계속 길면 동시 실행 수나 설정이 추정 때와 다른 것이다.

## 중지

중지하라는 말이 있을 때만 한다. 그 실험의 `worker.sh`와 `watch.sh`를 먼저 끝내고, 실행 중인 `run_cl` 프로세스를 끝낸다. 결과 폴더에 `STOPPED.txt`로 언제, 왜 멈췄고 무엇이 결과가 아닌지 적는다(예: `results/tocom-cl-main-v2/STOPPED.txt`). `report say`로 한 줄 알린다. 중단된 실행 폴더는 지우지 않고, 다시 돌릴 때는 새 실험 폴더로 간다.
