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
  python scripts/report_prol_validation.py --write

UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline \
  python scripts/publish_prol_validation.py --output results/prol_validation_publication_20261008_verified --execute
```

## 실제 GPU 결과와 중단

실행 코드 commit **697036a**. suite는 `--through validation_native --execute`로 시작했으나 저자 worker가 task1 checkpoint probe guard에서 실패하여 exit1로 중단했습니다. 실제 GPU 실행은 **저자 기준선1회 시도**이며 firsttask의 train5,000/eval1,000/79updates까지 마쳤습니다. task2·native 검증·측정3회·FLOPs는 실행하지 않았고, 자동 재시도·LR 변경도 없습니다.

| 첫 태스크의 단일 관측 | 값 |
|---|---:|
| Seen 정확도 | 993/1000 = **99.3%** |
| Epoch loss mean | 0.448255577 |
| Inner cosine step 수 | 66 |
| 학습 phase | **23.024초** |
| Recorder 계측 구간 | **35.311초** |
| Launcher 전체 구간 | 45.299초 |
| CUDA peak allocated / reserved | **6,551.47 / 7,140.00 MiB** |
| Process-tree RSS peak | **10,197.65 MiB** |
| Process-tree CPU time | 46.205초 |
| GPU board energy | **9,966.060 J = 2.76835 Wh** |
| 원전력 coverage | **99.9055%** |

원전력 CSV의 재적분 값과 시간 범위는 기록과 정확히 일치했고 sensor error는 없었습니다. Recorder 구간에는 실제 학습·평가·상태 검사·40example checkpoint probe가 포함되며 launcher preflight와 최종 upload는 제외됩니다. 위 값은 **실패한 검증 실행의 개별 진단 관측**입니다. native 효율성 평균·FLOPs·다른 방법과의 비교 결과로 사용하지 않습니다.

실패 지점은 고정 `scripts/run_prol_worker.py:684`의 `before/after logits bitexact` 또는 `probe 후 CPU/CUDA RNG` 결합 검사입니다. 그 전에 디스크 학습 상태 digest, clone 학습 상태 digest와 복원 직후 RNG 검사는 통과한 경로입니다. CPU로 읽은 task1 checkpoint의 399 tensors/144,014 parameters는 모두 finite이고 digest·lifecycle·affine clamp 범위가 일치했습니다. 따라서 checkpoint 손상 또는 학습 parameter 복원 오류로 확정하지 않습니다.

일반 평가와 clone의 query/train·prompt/eval 모드는 같고 활성 dropout/drop-path는 0입니다. 마지막40example 배치에는 unique index8개/duplicate32개, generator column0 선택2개가 기록됐습니다. 공식 배치 간 advanced assignment는 중복 CUDA writes를 만들며 일반 평가의 audit hook은 clone probe보다 추가 동기화를 합니다. 이는 후보 원인이지만 logits·개별 RNG 판정을 저장하지 않았기 때문에 **원인은 미확정**입니다. `state_audits`는 probe 이후에 저장되어 실패 시 선행 sub-check 통과가 독립 receipt로 남지 않는 계측 한계도 확인했습니다.

후속 단계에서는 출력/RNG 판정을 분리하고 최대 logit 차이·예측 변경 수·routing/RNG 증거를 저장하며 live 반복 forward와 clone forward를 구분해 진단해야 합니다. 이번에는 추가 GPU forward·학습 또는 승인 기준 변경을 하지 않았습니다.

로컬 결과는 `results/prol_validation_20261008`, [부분 보고서](../prol_validation/RESULTS.md)에 보존했습니다. [저자 실행 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/2qvxzkht)는 `failed` 상태이며 정확도·비용·원자료를 보관합니다. 최초 W&B artifact v0에서 별도 `error_context.json`이 누락되어 게시 검증을 중단했고, 동일 failed run에 원본 result SHA `f077b107bbec8f1d26ec7d0b85680c00a499cc360cb4bcad8a233a03c2568630`를 유지한 small evidence **v1**을 보완했습니다. 이는 upload만 수행한 것이며 새 학습 run·GPU 계산이 아닙니다. 보완 receipt는 `results/prol_evidence_completion_20261008/completion.json`에 기록합니다. 실행 source pin과 원본 결과는 수정하지 않았습니다.

[최종 부분 보고서 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/gaabom8o)를 게시하고 remote summary·COMMITTED artifact의 파일 digest/내용을 재검증했습니다. 원본 실행은 failed, 보고서 run은 finished이며 stage는 stopped/publication_gate=blocked로 표시합니다. **게시 검증 완료는 구현 검증 통과를 의미하지 않습니다.** Slack 시작·실패·최종 완료 알림은 모두 `sent`입니다. 게시 증거는 [publication.json](../prol_validation/publication.json), [source_readback.json](../prol_validation/source_readback.json), [publication_readback.json](../prol_validation/publication_readback.json)에 보관합니다. 고정 source를 그대로 사용한 local report `--write`와 publisher dry 명령은 exit0, 보완 뒤 publisher `--output …_verified --execute`도 exit0입니다.

## 해석 범위

시간·에너지는 initialize·학습·평가·live audit·checkpoint40example probe를 포함하고 최종 W&B 업로드는 제외합니다. training phase는 전체 process에 포함됩니다. GPU board 에너지는 CPU/RAM/전체 시스템 에너지가 아닙니다. process-tree RSS는 공유 page를 중복 합산할 수 있고, CUDA allocated/reserved는 외부 할당을 포함하지 않습니다. 별도 FLOPs는 지원 연산의 부분 추정이며 timing3회 평균에 섞지 않습니다. 같은 seed3회는 비용 변동 측정이며 정확도의 seed 불확실성을 추정하지 않습니다.

source의 `GM - GM @ I`, 배치 간 similarity assignment, task-local scaler index와 평가 중 affine clamp를 유지합니다. 공식 활성 코드의 이식 검증이며 논문 수학식·발표 성능 재현을 의미하지 않습니다. 전체10tasks, Small/Tiny, 새 seed, 예산 비교와 기존9방법 비교 확대는 이번 범위에 포함하지 않습니다.
