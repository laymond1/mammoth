# HiDe-Prompt 구현 및 실행 준비

ViT-B/16 구현·CPU 검증을 마친 뒤 사용자가 **독립 기준선/native GPU 검증과 통과 후 효율성 측정**을 승인했습니다. 이후 양쪽 구현 검증의 batch도64로 맞췄습니다. 현재 승인 조건은 [execution_plan.json](execution_plan.json), 이전 CPU 준비 단계의 기록은 [validation_plan.json](validation_plan.json)에 보존합니다. 기존 공유 CIFAR100·NPZ만 사용하며 파일이 없거나 손상되면 다운로드 없이 실패합니다. 실행 결과는 로컬과 W&B `laymond1/cl-efficiency`에 저장하고 Slack으로 알립니다.

실행 순서는 독립 저자 초기값 추출 → 두 구현의 GPU smoke → 각각 첫2태스크 TII20 + prompt50/task 및 태스크2 두 CA30 → 네 정확도·구조 검사 → 통과 시 공통 전처리 첫태스크 TII1 + prompt1 효율성3회와 별도 FLOPs1회입니다. 모든 GPU 실행은 GPU1이며, batch64의 실제 LR은 TII0.000125, prompt0.0075, prompt head0.00075, TII CA0.00125, prompt CA0.005입니다. 초기값·학습 배치·augmentation worker·단계별 RNG·통계 배치·평가 배치 경계를 맞춥니다.

기준선은 고정 저자 ViT·prefix·학습·통계·CA 함수를 직접 사용합니다. Mammoth class-IL과 맞추기 위해 최종 prompt logits는 seen classes로 제한합니다. 저자 기본 평가 loop는100클래스 logits를 유지하므로 이 차이도 기록하며, 논문 수치 재현으로 부르지 않습니다. W&B 업로드용 immutable snapshot과 파일별 SHA를 보존하고 [readback 도구](../../../scripts/verify_hide_results.py)로 finished run·COMMITTED artifact·다운로드 해시를 확인합니다. 마지막 artifact 업로드/finish는 기존 EASE와 같이 Recorder 계측에서 제외하고 launcher wall time에 따로 남깁니다.

방법은 태스크 내부 예측(WTP), 태스크 식별(TII), 전체 관측 클래스 예측(TAP)을 나누어 최적화합니다. 고정 ViT의 uninstructed 표현으로 학습한 보조 분류기가 태스크를 추론하고, 그 태스크의 prefix prompt로 최종 클래스 예측을 수행합니다. 두 표현의 클래스 통계를 보관해 과거 이미지를 저장하지 않고 분류기를 정렬하며, 과거 prompt의 평균과 대조 정규화를 사용합니다. 이는 [원 논문](https://proceedings.neurips.cc/paper_files/paper/2023/file/d9f8b5abc8e0926539ecbb492af7b2f1-Paper-Conference.pdf)의 구성 요소에 따른 포트이며, 실제 수치 검증은 다음 단계입니다.

| 항목 | 준비된 조건 |
|---|---|
| 저자 코드 | [thu-ml/HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt/tree/0ed433334ff2cfb206c5e2899651c3044ec85f92), commit `0ed433334ff2cfb206c5e2899651c3044ec85f92` |
| 브랜치·CLI | `codex/hide-prompt` · `--model hide-prompt` |
| Backbone | ViT-B/16, 768차원·12 layers·12 heads; Tiny·Small은 후속 범위 |
| 가중치 | 기존 Base AugReg IN21K → IN1K NPZ 재사용; 파일·canonical tensor SHA-256 고정 |
| 데이터·태스크 | `seq-cifar100-224`, 10클래스/태스크, 0..99 고정 순서, seed1993 |
| GPU·공유 경로 | GPU1, `/mnt/sdab1/share/data`, `/mnt/sdab1/userHome/wslim/.cache` |
| 향후 원격 기록 | W&B `laymond1/cl-efficiency`, 기존 private Slack webhook; 이번 단계 연결 전송 없음 |

## 저자 설정과 실제 포트 범위

[training_scripts/train_cifar100_vit.sh](https://github.com/thu-ml/HiDe-Prompt/blob/0ed433334ff2cfb206c5e2899651c3044ec85f92/training_scripts/train_cifar100_vit.sh)를 기준으로 TII 20epoch와 prompt 50epoch를 준비합니다. 두 단계의 합은 Mammoth `n_epochs=70`이며, 추가 분류기 정렬(CA)의 30epoch는 별도 태스크 종료 작업입니다. 사용되는 인자는 `crct_epochs`이며, 저자 설정의 `ca_epochs`는 엔진에서 읽지 않습니다. CA는 첫 태스크에서 실행하지 않고 두 번째부터 실행합니다.

Prefix는 0..4번째 block, 길이5, pool10, task당1개입니다. 초기 태스크부터 prompt/head LR 비율10:1을 적용합니다. TII에만 residual MLP `[2]`가 있고, prompt 모델 MLP는 identity입니다. Prompt momentum0.01, 대조 정규화0.1, 온도0.8, 클래스당 centroid10을 기준으로 준비했습니다.

저자 parser에서 `--weight-decay`와 `--weight_decay`는 같은 destination을 가지며 **실제 기본값은 0.0**입니다. 파일의 나중 기본값5e-4를 보고 포트의 weight decay를 바꾸지 않았습니다. `unscale_lr=True`여도 LR에는 batch/256이 곱해집니다. TII CA LR만 동일하게 scale되고 prompt CA LR은0.005로 유지됩니다. 이러한 코드 동작과 버전·단일 GPU 적용 사항은 [source_review.json](source_review.json)에 기록합니다.

원 코드의 prefix reshape는 batch/dual 차원을 permute하지 않습니다. 기본 구현은 이 동작을 보존하며, `--hide_use_permute_fix` 교정은 별도 비교 대상으로 취급합니다. 단일 GPU에서는 분산 gather/barrier와 `model.module` 접근을 안전하게 바꾸되 prompt 수식과 CA sample/step 수는 보존합니다. 저자 코드는 모든 TII 태스크를 먼저 학습하고 그 checkpoint를 prompt 단계에서 불러오지만, Mammoth는 태스크마다 TII → prompt로 진행합니다. 독립 기준선 구현 시 phase RNG와 checkpoint 순서를 맞춰 이 차이가 성능 차이로 오인되지 않도록 해야 합니다.

통계 산출은 저자 코드처럼 **클래스를 먼저 분리한 후 ViT를 실행**합니다. 보존한 prefix reshape에서는 batch 위치가 prompted 표현에 영향을 주므로, 클래스가 섞인 batch에서 표현을 구한 뒤 나누면 동일한 동작이 아닙니다. Native는 현재 태스크의 클래스별 subset을 고정된 데이터 index 순서로 읽고 `shuffle=False`, `drop_last=False`를 사용합니다. 저자의 클래스별 RandomSampler를 따르는 기준선도 native와 같은 순서·batch 경계를 명시적으로 맞춰야 합니다.

Prompt StepLR은 저자 엔진의 `scheduler.step(epoch)`가 epoch 학습 뒤에 호출되는 순서를 반영합니다. 첫 LR 감소를 실제 학습에 사용하는 지점은 0기반 local epoch31, 즉 prompt 학습의32번째 epoch입니다. 30번째 index에서 미리 감소시키지 않습니다.

이 프로토콜은 **고정한 저자 코드와 native의 구현 비교**입니다. 논문 Appendix C의 prompt epoch·momentum·batch와 현재 저자 launcher가 다르고, 선택된 가중치는 사용자가 확정한 IN21K → IN1K 계열입니다. 논문 표 수치 재현으로 부르지 않습니다.

## 다음 GPU 단계의 후보 계획

| 목적 | 태스크·학습 | 전처리 | 판정·비교 범위 |
|---|---|---|---|
| 독립 기준선/native 구현 검증 | 첫2태스크, 각각 TII20 + prompt50, task2 두 CA30, batch64 | 새 `hide_prompt` config의 저자 CIFAR 전처리 | 첫태스크 및 task2 전체/과거/신규 정확도·routing·구조 검증 |
| 효율성 pilot 후보 | 첫1태스크, TII1 + prompt1, batch64 | 기존 `ease` 공통 pilot 전처리로 명시적 override | 양쪽 통계 산출·실제 routing·평가 비용 포함; task1 CA는 method상 해당 없음 |

두 전처리는 같은 원본 CIFAR100을 사용하지만 crop·jitter·정규화가 다릅니다. 검증용 `hide_prompt`는 저자 실제 동작인 crop scale(.08,1), bicubic, brightness jitter63/255, CIFAR mean/std를 사용합니다. 파일의 미사용 local `scale=(.05,1)`을 적용하지 않았습니다. 검증과 pilot 사이의 전처리 변경도 결과에 남겨야 합니다.

검증 정확도는 동일 기준선 대비 하락이 **1.5%p 이내 통과, 1.5~3.0%p 추가 점검, 3.0%p 초과 실패**입니다. 표시 반올림 값 대신 정답수/표본수로 판정합니다. 비정상적으로 낮은 성능, 비유한 값, frozen backbone 또는 old prompt 변경, 데이터·checkpoint 불일치, oracle task routing은 구현 문제로 조사합니다. 추가 점검·실패는 효율성 측정과 구현 수치 승인으로 넘어가지 않습니다.

효율성 pilot은 TII1 + prompt1, 즉 **두 학습 단계와 두 통계 pass**입니다. 기존 방법의1epoch와 동일 계산 예산을 뜻하지 않습니다. 첫태스크에서는 CA가 건너뛰어져 정렬의 비용과 망각을 측정할 수 없습니다. 필요하면 task2 비용을 후속으로 추가해야 합니다. 승인된 runner는 단계별 시간, CUDA/NVML peak, CPU time/RSS, GPU Wh와 sampling coverage, trainable/total params, throughput, 정확도/routing을 저장하고, 부분 연산 FLOPs는 별도 진단으로 분리해야 합니다.

## 학습 없는 준비 명령

```bash
python3 scripts/prepare_hide_prompt.py
python3 scripts/prepare_hide_prompt.py --check-local
```

기본 출력은 ignored `benchmarks/cl_efficiency/prepared/hide_prompt/`의 `plan.json`과 `candidate_commands.md`입니다. 두 번째 명령도 기존 파일의 SHA-256을 읽기만 하며, 파일이 없거나 다르면 실패합니다. ML 모듈·credential·센서·네트워크를 사용하지 않습니다. 생성된 native 명령은 설정 검토용이며 W&B·Slack·phase instrumentation을 갖춘 실행 runner가 아닙니다. 준비 도구에는 `--execute`가 없습니다.

다음 단계에서 독립 저자 기준선 runner와 matched phase RNG를 준비하고, 두 태스크의 실행 범위 및 효율성 pilot의 전처리·단계 예산을 확정해야 합니다. 전체 CL, 추가 seed, Tiny/Small, BudgetCL/aL-SAR와 원격 push는 이번 범위에 포함되지 않습니다.

## 완료한 CPU 검증

학습 없는 집중·회귀 검사229개가 통과했습니다. [검증 명령과 결과](checks.md), [실행 전 기록](environment/preflight.json)에 구체적인 명령·파일 SHA를 보존합니다. [독립 network 수식 대조](environment/network_source_audit.json)의9개 조건과 [통계·alignment 대조](environment/statistics_source_audit.json)의6개 조건에서 저자 함수와 출력·gradient·파라미터가 일치했습니다. 이 대조는 작은 합성 feature/tensor를 사용한 계산 검증이며 실제 CIFAR 학습 결과가 아닙니다.

[실제 Base CPU 구성](environment/native_cpu_initialization.json)에서는 기존 공유 NPZ를 읽어 두 ViT의 사전학습 텐서150개가 각각 고정 SHA와 일치함을 확인했습니다. 실제 이미지 forward나 Base optimizer update는 수행하지 않았습니다. Native CLI `--help`와 스킬의 모델 연결 검사도 통과했습니다. 독립 기준선 runner와 실제 정확도 acceptance는 후속 단계입니다.
