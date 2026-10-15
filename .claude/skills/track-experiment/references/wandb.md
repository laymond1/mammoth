# W&B에 올리는 방식

`scripts/tocom/tracking.py`의 `Mirror`가 한다. `run_cl.py --wandb ENTITY/PROJECT`를 줄 때만 동작한다.

## 어디에 올리나

| 항목 | 값 |
|---|---|
| 프로젝트 | 본 실험은 `laymond1/tocom-plugin-cl`, 사전 점검 실행은 `laymond1/tocom-plugin-cl-checks`. `settings.sh`의 `W`, `W_CHECK` |
| 실행 이름 | 실행 폴더 이름과 같다. 예: `coda-prompt-tome-r8-seed1993` |
| group | 실행 이름에서 seed를 뺀 것. 예: `coda-prompt-tome-r8`. W&B에서 group으로 묶으면 seed 평균과 범위가 그려진다 |
| tags | 실험 이름(결과 폴더 이름), 학습 방식(`none`, `plugin`, `own`), 방법 계열 |
| job_type | 학습 실행은 `train`, 전체가 끝날 때 올리는 집계 실행은 `summary` |

이전 DeiT-Tiny 연구의 프로젝트 `compression-gap-cl`에는 올리지 않는다. 값의 이름이 달라서 섞이면 화면이 어지럽다. 인증은 서버에 이미 있는 로그인(`~/.netrc`)을 쓴다.

## 무엇을 올리나

| 언제 | 값 |
|---|---|
| 시작할 때 (config) | `experiment`, `method`, `family`, `kind`, `train_operator`, `train_degree`, `seed`, `targets`, `eval_operators`, `epochs`, `batch_size`, `commit`, `dirty`, 가중치와 보정기의 지문 |
| 에포크마다 | `epoch`, `train/task`, `train/loss`, `train/lr`, `train/epoch_seconds` |
| 태스크마다 | `task`, `acc/class_il`, `acc/task_il`, `task_acc/t01`…, `time/task_minutes`, `time/grid_minutes`, 그리고 추론 축소 방법과 추론 압축도마다 `tome-off/r16`, `tome-on/r16`, `tome-gain/r16` 같은 값 |
| 끝날 때 | `status`, `tasks_done`, `hours`, `final/curve-<축소 방법>`, 파일 `config.json` `provenance.json` `metrics.json` `summary.json` `progress.jsonl` |

`off`는 보정 끔, `on`은 그 축소 방법의 보정기 켬, `gain`은 켬 빼기 끔이다. `on0`과 `gain0`은 학습 압축도를 0으로 본 보정으로 CPS-Prompt와 REP의 자체 압축 실행에만 있다. `cross-<다른 방법>`은 다른 축소 방법의 보정기를 끼운 값이다.

가중치, 코드, git diff, 콘솔 로그는 올리지 않는다. Slack 주소는 W&B 어디에도 들어가지 않는다.

## 지켜지는 성질

- **학습 결과를 바꾸지 않는다.** `wandb.init` 앞뒤로 난수 상태를 그대로 되돌린다. 기록을 켠 실행과 끈 실행의 가중치 지문(`metrics.json`의 `state_sha256`)이 같아야 하고, 코드를 고쳤으면 사전 점검에서 다시 확인한다.
- **Mammoth의 자체 W&B 기록은 끈다.** 그 기록은 학습 스텝마다 프로파일러를 걸어 학습 경로를 바꾼다. 그래서 `run_cl.py`가 `WANDB_ENTITY`, `WANDB_PROJECT` 환경변수를 지우고 값을 직접 올린다. 이 대기열에서 Mammoth의 `--wandb_entity`, `--wandb_project`를 주지 않는다.
- **W&B가 끊겨도 학습은 계속된다.** 시작할 때 연결되지 않으면 offline으로 기록하고 실행 폴더의 `tracking.json`에 `offline: true`를 남긴다. 기록 도중 오류가 나면 그 실행의 기록만 멈춘다. 끝날 때 연결이 끊겨 있으면 15분까지 기다린 뒤 넘어간다.

## 새 값을 더할 때

실행 폴더의 파일에 먼저 적고, `tracking.py`의 `task_metrics`가 그 기록을 읽어 올리게 한다. W&B에만 있고 파일에 없는 값을 만들지 않는다. 이름은 `<묶음>/<항목>` 꼴로 지어 W&B 화면에서 묶음별로 모이게 한다.
