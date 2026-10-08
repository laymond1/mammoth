# PROL checkpoint 출력/RNG 진단

2026-10-08 최신 사용자 결정: **PROL 추가 작업을 보류하고 이번 비교에서 제외합니다.** 구현 코드는 보존하되 GPU 재현성 검증 미통과 상태를 유지합니다. 이미 완료한 실패 실행·추론 진단·원격 게시 증거를 보존하며 제안한 연산 격리 진단·결정론 교정·추가 checkpoint 검증·학습 재실행은 진행하지 않습니다. 아래는 완료된 진단 단계의 역사적 실행 기록입니다.

2026-10-08 승인한 후속 단계는 **재학습 없는 추론 진단 GPU process1개**입니다. branch `codex/prol-diagnostics`에서 진단 도구와 계측을 보완합니다. 원본 실패 실행·checkpoint·예측은 변경하지 않으며, 당시 실행 코드 commit `697036abe847c0beb269675d336eb8b7251f102e`의 Git bytes로 역사적 source42개를 검증합니다. [승인 설정](run_plan.json)을 고정합니다.

## 범위

저자 첫task 정확도는 99.3%였지만 checkpoint probe의 출력/RNG 결합 검사에서 실패했습니다. 그 전에 디스크 learner hash·복원 learner hash·복원 직후 RNG 검사는 통과한 경로입니다. 당시 before/after logits와 별도 RNG 결과가 저장되지 않아 실패 원인은 미확정입니다. [이전 기록](../prol_prompt/RUN_VALIDATION.md)과 [실패 실행](https://wandb.ai/laymond1/cl-efficiency/runs/2qvxzkht)을 보존합니다.

기존 checkpoint와 공통 local NPZ로 **재구성한 저자 모델 A**와 기존 compact clone 경로로 만든 **모델 B**를 비교합니다. 원래 process는 종료됐으므로 A는 당시 live 모델 자체가 아닙니다. 원래 task1 평가의 `evaluation_global_ids` 마지막40개를 같은 순서·전처리로 다시 구성합니다. 원본 logits는 복원할 수 없습니다.

| 조건 | 반복 | 입력 |
|---|---:|---|
| Author restored A, audit OFF | 10 | 같은 마지막40개 |
| Author restored A, audit ON | 10 | 같은 마지막40개 |
| Compact clone B, audit OFF | 10 | 같은 마지막40개 |
| Compact clone B, audit ON | 10 | 같은 마지막40개 |

총 **40 model evaluations/80 encoder forwards/20 A↔B paired comparisons/1,600 diagnostic image visits/unique40images**입니다. 각 evaluation은 query와 prompted ViT를 각각1회 실행합니다. 학습·backward·optimizer update·warmup·추가 평가·자동 재시도는 없습니다. GPU1/ViT-B/16/FP32/기존 공통 가중치·캐시 경로를 유지합니다. matrix 시작에 phase-derived diagnostic seed2993을 한 번 적용합니다. 이는 공통 base seed1993에서 파생한 이번 진단용 seed이며 원래 probe RNG bytes의 복원을 의미하지 않습니다.

## 증거와 해석

매 평가에서 logits·예측·query features·CPU/CUDA RNG pre/post·learner/frozen state hash·입력 hash를 저장합니다. ON은 당시 commit에서 추출한 실제 `ProlAudit.prompt_output` callback을 호출하며, 원전 `compute_att_over_prompt`의 실제 similarity/index/generated prefix와 affine pre/post 참조를 추가 기록합니다. OFF에는 내부 hook/wrapper를 설치하지 않습니다. ON의 추가 참조 유지·Python scheduling·후속 CPU materialization 자체가 관측에 영향을 줄 수 있으므로 ON/OFF 차이만으로 특정 CUDA 연산을 원인으로 단정하지 않습니다.

반복 내부 차이, A/B 차이, hook 조건 차이를 raw logits에서 계산합니다. 입력·상태·RNG가 같아도 출력이 변하는지, RNG 변화인지, 복원 모델과 clone의 차이인지 분리합니다. RNG/state 변화도 관측 증거로 보존합니다. 기존 정확도 하한·1.5/3%p·bitexact 판정 기준은 변경하지 않으며 이번 완료를 구현 검증 통과로 취급하지 않습니다. 불일치가 재현되지 않아도 원인 해결로 표시하지 않습니다. 원래40개 예측과의 일치는 진단이며 새 CL 정확도가 아닙니다.

기존 계측은 출력/RNG 검사 결과를 분리하고 각각의 flag·최대 logit 차이·예측 변화와 probe 원자료, 선행 sub-check 결과를 guard 전에 저장하도록 보완합니다. 오류 문맥 파일도 worker artifact에 포함합니다. 보완된 training worker는 이번에 학습 실행하지 않습니다.

## 실행·보고

기본 CLI는 dry이며 GPU·credentials·W&B·Slack을 호출하지 않습니다. 승인 실행은 아래 명령으로 한 번만 수행합니다. 실제 CPU 검사·GPU 결과·게시 증거는 이 문서의 실행 기록에 추가합니다.

```bash
UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/run_prol_diagnostic.py
UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/run_prol_diagnostic.py --execute
```

로컬 `results/prol_diagnostic_20261008`와 `laymond1/cl-efficiency`에 작은 원자료·설정·진단 결과를 저장하고 Slack으로 시작·중단·완료를 알립니다. checkpoint·전체 가중치와 raw image tensor는 local에 유지합니다. 원래 failed run과 새 diagnostic run의 terminal 상태·COMMITTED artifact·SHA/내용을 검증한 뒤 report-only W&B run을 게시합니다. 에너지·메모리·시간은 진단 도구 실행의 관측값이며 학습 효율성 평균에 넣지 않습니다.

진단 결과 보고 후 종료합니다. 저자/native 첫2태스크 재실행, 효율성3회와 FLOPs, LR/seed/판정 기준 변경, 논문식 수정과 비교 확대는 이번 범위에 포함하지 않습니다.

## CPU 검증 기록

실행 범위·original Git provenance·실패 재시도 금지 검사:

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv \
  uv run --no-sync --offline pytest -q tests/test_prol_diagnostic_protocol.py
```

**26 passed, 4.15s.** 진단 worker·원형 계측 focused 검사도 **36 passed, 9.32s**이며 synthetic 실제 저자 source로 4조건×10회/80encoderforwards/1,600visits, OFF hook 부재·ON 원형 callback·no backward/optimizer, checkpoint 출력/RNG 분리 증거와 clamp storage, SIGTERM의 실패 증거 저장을 확인했습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv \
  uv run --no-sync --offline pytest -q tests/test_prol_checkpoint_diagnostic.py tests/test_prol_worker.py
```

CLI dry 명령과 Git 원형42파일 검증도 exit0입니다. 별도 CPU preflight로 실제 공유 CIFAR100 test에서40개 입력을 재구성했습니다. shape `[40,3,224,224]`, unique IDs40, labels0–9, 전처리 repr/hash는 이전 평가와 일치하며 새로 재구성한 입력 hash는 `20aaa6b6e8e5f2145dcb50d85596ec7b5f26c7b98c30250ffa518a93ee1131fb`입니다. 이 확인은 model 생성·forward·GPU·logging 없이 수행했습니다.

clone에 ON callback을 등록하기 위한 hook container는 PyTorch `OrderedDict`로 유지하도록 보완했습니다. 이전 probe는 clone hook을 등록하지 않았으므로 이 인터페이스 수정은 원래 실패 원인으로 해석하지 않습니다.

통합 focused CPU 검증은 **286 passed, 71.32s**입니다. 실제 저자 source 기반 synthetic smoke와 Mammoth `meta_*` 경로 검사, 저장 원자료 기반 보고서 검증, W&B/Slack fake API를 통한 manifest·terminal status·내용 readback·알림 중복 방지 검사를 포함합니다. 실제 네트워크·GPU는 이 테스트에서 호출하지 않았습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv \
  uv run --no-sync --offline pytest -q \
  tests/test_prol_components.py tests/test_prol_prompt.py tests/test_prepare_prol_prompt.py \
  tests/test_prol_worker.py tests/test_prol_protocol.py tests/test_prol_validation_report.py \
  tests/test_prol_validation_publication.py tests/test_prol_diagnostic_protocol.py \
  tests/test_prol_checkpoint_diagnostic.py tests/test_prol_diagnostic_report.py \
  tests/test_prol_diagnostic_publication.py
```

`git diff --check`도 통과했습니다. 이번 변경은 진단·계측·보고 도구와 테스트이며 알고리즘·학습 설정·기존 판정 기준을 수정하지 않았습니다.

## 실제 실행 결과

진단 코드를 commit `cb3f7f5e1e791011397275117f171966987eb728`로 고정한 뒤 GPU1에서 승인한 process를 **1개** 실행했습니다. launcher와 worker 모두 completed이며 **40 evaluations/80 encoder forwards/1,600 image visits/unique40/학습·backward·optimizer0회**를 저장했습니다. 입력 hash는 CPU preflight와 일치했습니다. 재실행·후속 학습은 없습니다.

| 비교 | 저장 원자료에서 계산한 쌍 | bitexact 불일치 | 최대 logit 절대 차이 | 예측 변경 |
|---|---:|---:|---:|---:|
| A/B 동일 hook 조건 | 20 | 20 | 0.193215609 | 0 |
| 동일 모델·hook 조건의 반복 전체 쌍 | 180 | 180 | 0.193215609 | 0 |
| 같은 모델의 audit ON/OFF | 20 | 20 | 0.175273180 | 0 |

모든40회에서 CPU/CUDA RNG bytes와 learner/frozen state hash가 전후 동일했습니다. 시작 learner digest도 한 종류이며 저장 checkpoint와 일치했습니다. 모든 비교의 query features는 bitexact였고,40개 진단 이미지의 예측은 매회 동일하며 원래 저장된 마지막40개 예측과도 일치했습니다. 이는40개 이미지에 대한 진단 관측이며 새 CL 정확도를 계산한 결과가 아닙니다.

ON 내부 trace28비교(A/B10쌍 및 각 ON조건의 repeat2–10과 repeat1 비교18쌍)에서 첫 관측 차이는 모두 **layer0**이었습니다. routing indices와 affine pre/post tensor는 같지만 `similarity_new`와 생성 prefix가 달랐습니다. 동일 모델 반복에서도 출력 변동이 관측됐으므로 복원 A/B의 차이를 단독 원인으로 판단할 수 없습니다. 공통 입력·query features·index·RNG·learner가 유지된 상태에서 similarity 생성 이후 경로를 후속 조사 지점으로 좁혔습니다. 추가 trace가 관측에 영향을 주며 원래 live probe logits는 없으므로, 특정 CUDA 쓰기를 원인으로 확정하거나 해결·구현 통과를 선언하지 않습니다.

진단 worker Recorder 범위의 관측 비용은 **46.82s**, CUDA peak allocated **1,039.34MiB**, reserved **1,272MiB**, process RSS peak **1,718.86MiB**, GPU board energy **4,345.96J (1.20721Wh)**였습니다. 에너지 표본 coverage는99.9313%이며 저장 power 자료를 적분해 검증했습니다. 초기화·복원·state hashing·ON trace 보존·보고 자료 준비를 포함한 진단 비용이고 학습 효율성 pilot이나 알고리즘 비교 평균에 포함하지 않습니다.

실제 실행·게시 명령은 아래와 같습니다. reporter와 publisher는 저장 자료만 처리하며 새 GPU forward를 호출하지 않습니다.

```bash
UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/run_prol_diagnostic.py --execute
UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/report_prol_diagnostic.py --write
UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/publish_prol_diagnostic.py
UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/publish_prol_diagnostic.py --execute
```

- [진단 요약](RESULTS.md), [원자료 검증·비교 JSON](results.json), [비교 CSV](results.csv).
- [GPU 추론 진단 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/v80kh0np): finished, 작은 원자료 artifact COMMITTED.
- [보고서 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/zpbx6iq2): finished,52개 증거 파일의 SHA·크기·내용 및 report artifact COMMITTED readback 확인.
- [원격 source 검증](source_readback.json), [보고서 readback](publication_readback.json), [게시·Slack 영수증](publication.json).

원본 W&B `2qvxzkht`는 remote/local 모두 **failed**로 유지되었습니다. 신규 진단·report 각각의 저장 내용·계산 횟수·artifact manifest를 검증했고 Slack 시작·진단 완료·검증 후 보고서 완료 알림은 모두 `sent`입니다. 진단 결과 보고 단계에서 종료했으며 기존 bitexact guard와 정확도 허용 범위, 후속 실험 보류 상태를 유지합니다.
