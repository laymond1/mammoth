# PROL GPU 검증·효율성 실행 기록

2026-10-08 사용자 승인 범위: [run_plan.json](run_plan.json). 별도 로컬 branch `codex/prol-validation`에서 runner·보고 도구를 검증하고 실행합니다. 원형 구현은 `df5df65`, 독립 저자 source는 `bfff8418a4f603a24ae578f1e108bfac89af1e18`에 고정했습니다.

## 실행 정책

- 물리 GPU1, ViT-B/16 AugReg IN21K→IN1K, seq-cifar100-224/ease, class0–99, batch64, workers4, seed1993, FP32.
- nominal LR0.01, initial effective LR0.0025. source의 모든 optimizer group·sticky threshold·cosine 경로를 유지합니다.
- 저자/native 첫2tasks×1epoch 각1회 이후 구현 검증과 old/new 정확도·loss 추세 점검을 통과한 경우에만 native 첫task×1epoch 비용3회와 별도 FLOPs1회를 진행합니다. 최대6회이며 자동 재시도는 없습니다.
- task1<90% 또는 task2 seen<20%면 해당 worker에서 중단합니다. task2 20%를 넘는 것만으로 정상성을 통과시키지 않습니다. native 정확도 감소≤1.5%p 통과, >1.5–3%p 점검, >3%p 실패입니다.
- 로컬·laymond1/cl-efficiency W&B에 작은 원자료와 검증 evidence를 저장하고 Slack으로 실행·중단·완료를 알립니다. 초기·학습 가중치는 업로드하지 않습니다.

## 도구

`scripts/run_prol_worker.py`는 native에서 실제 Mammoth `train()`·`meta_*`를 호출하며, 저자 runner에서는 고정 source의 AST numerical engine을 실행합니다. 학습하는 LayerNorm·query의 사용하지 않는 classifier까지 초기 상태 catalog에 포함합니다. 실제 첫 batch 입력·labels·CPU/CUDA RNG·gradient, 전체 LR·sample stream, 동결 상태와 disk checkpoint 복원 출력을 확인합니다.

`scripts/run_prol_suite.py`는 6개 작업과 source/init pin을 고정합니다. 대조 후 생성하는 `validation_acceptance.json`과 별도 `normality_review.json` 없이는 pilot을 실행하지 않습니다. 보고·게시 도구는 raw 예측·정수 counts·power CSV와 remote W&B artifact를 다시 읽어 확인합니다. 중단한 경우에는 부분 관측만 보고하며 평균 비용·방법 간 비교를 생성하지 않습니다.

## 검증 명령과 결과

GPU 실행 전 전체 PROL focused 검사와 synthetic actual-trainer smoke를 마쳤습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv \
  uv run --no-sync --offline pytest -q tests/test_prol_components.py tests/test_prol_prompt.py \
  tests/test_prepare_prol_prompt.py tests/test_prol_worker.py tests/test_prol_protocol.py \
  tests/test_prol_validation_report.py tests/test_prol_validation_publication.py
```

**203 passed, 13.10s.** 실제 Mammoth `train()`과 독립 저자 epoch engine을 사용하는 synthetic 첫2tasks smoke를 포함합니다. 각각 train10,000/eval3,000/158updates와 batch tail을 확인했으며 입력·loss·gradient·RNG·전체 LR trace·state·예측을 대조했습니다. source/native disk checkpoint의 학습 norm을 바꾼 복원 검사, 정확도 중단·실패 resume 금지, raw energy·remote artifact 변조 검사도 포함합니다. 실제 CIFAR 또는 GPU 학습은 이 CPU 검사에 포함하지 않습니다.

별도 runner dry 계획 명령 `UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/run_prol_suite.py`도 exit0이며 실행6개·normality gate를 확인했습니다.

독립 CPU 초기 상태: `results/prol_initial_20261008/initial_state.pt`, SHA256 `1b7259b700fd29b721df8b53ccc94a620b1dcdb6b2678cf2b51e5db0db5cecd5`. 전체 learner catalog hash는 `48a898743f35a5ba1d121734f63864d22c52e0de608433358a0ce228de754a4b`이며 query/prompt의 canonical pretrained150 tensors는 모두 공통 hash `cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300`와 일치했습니다.

## 실행 명령

초기 상태는 독립 저자 branch의 CPU 초기화로 준비하며 실제 학습을 시작하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv \
  uv run --no-sync --offline python scripts/run_prol_worker.py \
  --prepare-initial-state results/prol_initial_20261008/initial_state.pt

UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline \
  python scripts/run_prol_suite.py --through validation_native --execute
```

성능 정상성 receipt가 통과했을 때만 같은 frozen source로 나머지 작업을 실행합니다.

```bash
UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline \
  python scripts/run_prol_suite.py --resume results/prol_validation_20261008 --through flops --execute

UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline \
  python scripts/report_prol_validation.py

UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline \
  python scripts/publish_prol_validation.py --execute
```

## 해석 범위

시간·에너지는 initialize·학습·평가·live audit·checkpoint40example probe를 포함하고 최종 W&B 업로드는 제외합니다. training phase는 전체 process에 포함됩니다. GPU board 에너지는 CPU/RAM/전체 시스템 에너지가 아닙니다. process-tree RSS는 공유 page를 중복 합산할 수 있고, CUDA allocated/reserved는 외부 할당을 포함하지 않습니다. 별도 FLOPs는 지원 연산의 부분 추정이며 timing3회 평균에 섞지 않습니다. 같은 seed3회는 비용 변동 측정이며 정확도의 seed 불확실성을 추정하지 않습니다.

source의 `GM - GM @ I`, 배치 간 similarity assignment, task-local scaler index와 평가 중 affine clamp를 유지합니다. 공식 활성 코드의 이식 검증이며 논문 수학식·발표 성능 재현을 의미하지 않습니다. 전체10tasks, Small/Tiny, 새 seed, 예산 비교와 기존9방법 비교 확대는 이번 범위에 포함하지 않습니다.
