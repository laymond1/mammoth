# HiDe-Prompt 구현 검증 및 효율성 결과

Base 전체10태스크 후속 실험도 완료했습니다. 최종 정확도 **94.27%**, 평균 증분 정확도 **95.89%**, 평균 망각 **1.90%p**이며, [전체10태스크 보고서](full_cl/report.md)와 [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/shz1r5i1)에서 확인할 수 있습니다. 아래에는 첫2태스크 구현 검증과 첫태스크 효율성 pilot 기록을 보존합니다.

ViT-B/16의 독립 저자 기준선과 Mammoth native를 GPU1·batch64로 실제 학습해 **구현 acceptance PASS**를 확인했습니다. 첫 태스크와 태스크2 전체/과거/신규 정확도 네 항목이 각각 **99.40 / 97.95 / 98.90 / 97.00%**로 같고, 정확도 차이는 모두 **0.00%p**입니다. 13개 구조 검사가 통과했고 저장된 평가 예측 3,000개가 모두 일치했습니다. 세부 결과는 [보고서](report.md), [JSON](results.json), [CSV](results.csv), [그림 PNG](efficiency_comparison.png)·[PDF](efficiency_comparison.pdf)에 정리합니다.

현재 승인 조건은 [execution_plan.json](execution_plan.json), 이전 CPU 준비 단계의 기록은 [validation_plan.json](validation_plan.json)에 보존합니다. 기존 공유 CIFAR100·NPZ만 사용하며 파일이 없거나 손상되면 다운로드 없이 실패합니다. [독립 기준선 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/eiil7wjc)와 [native W&B](https://wandb.ai/laymond1/cl-efficiency/runs/iw6ohxji)는 finished/COMMITTED artifact의 다운로드 SHA readback까지 통과했습니다. Slack 진행 알림도 실행 기록에 남습니다.

실행 순서는 독립 저자 초기값 추출 → 두 구현의 GPU smoke → 각각 첫2태스크 TII20 + prompt50/task 및 태스크2 두 CA30 → 네 정확도·구조 검사 → 통과 후 공통 전처리 첫태스크 TII1 + prompt1 효율성3회와 별도 FLOPs1회입니다. 모든 GPU 실행은 GPU1이며, batch64의 실제 LR은 TII0.000125, prompt0.0075, prompt head0.00075, TII CA0.00125, prompt CA0.005입니다. 초기값·학습 배치·augmentation worker·단계별 RNG·통계 배치·평가 배치 경계를 맞췄습니다.

기준선은 고정 저자 ViT·prefix·학습·통계·CA 함수를 직접 사용합니다. Mammoth class-IL과 맞추기 위해 최종 prompt logits는 seen classes로 제한합니다. 저자 기본 평가 loop는100클래스 logits를 유지하므로 이 차이도 기록하며, 논문 수치 재현으로 부르지 않습니다. W&B 업로드용 immutable snapshot과 파일별 SHA를 보존하고 [readback 도구](../../../scripts/verify_hide_results.py)로 finished run·COMMITTED artifact·다운로드 해시를 확인합니다. 마지막 artifact 업로드/finish는 기존 EASE와 같이 Recorder 계측에서 제외하고 launcher wall time에 따로 남깁니다.

방법은 태스크 내부 예측(WTP), 태스크 식별(TII), 전체 관측 클래스 예측(TAP)을 나누어 최적화합니다. 고정 ViT의 uninstructed 표현으로 학습한 보조 분류기가 태스크를 추론하고, 그 태스크의 prefix prompt로 최종 클래스 예측을 수행합니다. 두 표현의 클래스 통계를 보관해 과거 이미지를 저장하지 않고 분류기를 정렬하며, 과거 prompt의 평균과 대조 정규화를 사용합니다. 이는 [원 논문](https://proceedings.neurips.cc/paper_files/paper/2023/file/d9f8b5abc8e0926539ecbb492af7b2f1-Paper-Conference.pdf)의 구성 요소에 따른 포트이며, 이번 결과는 명시한 프로토콜에서의 두 태스크 구현 검증입니다.

| 항목 | 실행한 조건 |
|---|---|
| 저자 코드 | [thu-ml/HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt/tree/0ed433334ff2cfb206c5e2899651c3044ec85f92), commit `0ed433334ff2cfb206c5e2899651c3044ec85f92` |
| 브랜치·CLI | `codex/hide-prompt` · `--model hide-prompt` |
| Backbone | ViT-B/16, 768차원·12 layers·12 heads; Tiny·Small은 후속 범위 |
| 가중치 | 기존 Base AugReg IN21K → IN1K NPZ 재사용; 파일·canonical tensor SHA-256 고정 |
| 데이터·태스크 | `seq-cifar100-224`, 10클래스/태스크, 0..99 고정 순서, seed1993 |
| GPU·공유 경로 | GPU1, `/mnt/sdab1/share/data`, `/mnt/sdab1/userHome/wslim/.cache` |
| 원격 기록 | W&B `laymond1/cl-efficiency`, 기존 private Slack webhook; GPU 실행별 기록·진행 알림 |

## 저자 설정과 실제 포트 범위

[training_scripts/train_cifar100_vit.sh](https://github.com/thu-ml/HiDe-Prompt/blob/0ed433334ff2cfb206c5e2899651c3044ec85f92/training_scripts/train_cifar100_vit.sh)를 기준으로 TII 20epoch와 prompt 50epoch를 실행했습니다. 두 단계의 합은 Mammoth `n_epochs=70`이며, 추가 분류기 정렬(CA)의 30epoch는 별도 태스크 종료 작업입니다. 사용되는 인자는 `crct_epochs`이며, 저자 설정의 `ca_epochs`는 엔진에서 읽지 않습니다. CA는 첫 태스크에서 실행하지 않고 두 번째부터 실행합니다. 이번 태스크2에서는 TII·prompt 각각30epoch/300 optimizer step을 확인했습니다.

Prefix는 0..4번째 block, 길이5, pool10, task당1개입니다. 초기 태스크부터 prompt/head LR 비율10:1을 적용합니다. TII에만 residual MLP `[2]`가 있고, prompt 모델 MLP는 identity입니다. Prompt momentum0.01, 대조 정규화0.1, 온도0.8, 클래스당 centroid10을 적용했습니다.

저자 parser에서 `--weight-decay`와 `--weight_decay`는 같은 destination을 가지며 **실제 기본값은 0.0**입니다. 파일의 나중 기본값5e-4를 보고 포트의 weight decay를 바꾸지 않았습니다. `unscale_lr=True`여도 LR에는 batch/256이 곱해집니다. TII CA LR만 동일하게 scale되고 prompt CA LR은0.005로 유지됩니다. 이러한 코드 동작과 버전·단일 GPU 적용 사항은 [source_review.json](source_review.json)에 기록합니다.

원 코드의 prefix reshape는 batch/dual 차원을 permute하지 않습니다. 기본 구현은 이 동작을 보존하며, `--hide_use_permute_fix` 교정은 별도 비교 대상으로 취급합니다. 단일 GPU에서는 분산 gather/barrier와 `model.module` 접근을 바꾸되 prompt 수식과 CA sample/step 수는 보존했습니다. 저자 코드는 모든 TII 태스크를 먼저 학습하고 그 checkpoint를 prompt 단계에서 불러오지만, Mammoth는 태스크마다 TII → prompt로 진행합니다. 독립 기준선도 같은 태스크별 순서와 phase RNG를 사용했고, 동일하게 export한 learnable 초기값을 두 구현에 주입했습니다.

통계 산출은 저자 코드처럼 **클래스를 먼저 분리한 후 ViT를 실행**합니다. 보존한 prefix reshape에서는 batch 위치가 prompted 표현에 영향을 주므로, 클래스가 섞인 batch에서 표현을 구한 뒤 나누면 동일한 동작이 아닙니다. Native와 독립 기준선 모두 현재 태스크의 클래스별 subset을 고정된 데이터 index 순서로 읽고 `shuffle=False`, `drop_last=False`를 사용해 순서·batch 경계를 맞췄습니다. 이는 저자의 클래스별 RandomSampler를 명시적으로 대체한 비교용 적용 사항입니다.

고정 저자 AST를 직접 컴파일해 사용하며 timm global 등록 충돌을 피했습니다. 현 환경에 맞춰 KMeans를 `init=k-means++`, Lloyd, `n_init=10`, seed+class로 고정하고, deprecated inclusive `torch.range`는 같은 범위의 `torch.arange`로 바꿨습니다. PyTorch2.13.0+cu126·torchvision0.28.0+cu126·timm0.9.8, float32·autocast off·TF32 matmul off를 양쪽에 적용했습니다. 모든 적용 사항은 [저자 실행 기록](runs/hide-author-validation-20261007T045302Z/runtime_adaptations.json)과 [최종 독립 감사](environment/full_validation_audit.json)에 남습니다.

Prompt StepLR은 저자 엔진의 `scheduler.step(epoch)`가 epoch 학습 뒤에 호출되는 순서를 반영합니다. 첫 LR 감소를 실제 학습에 사용하는 지점은 0기반 local epoch31, 즉 prompt 학습의32번째 epoch입니다. 30번째 index에서 미리 감소시키지 않습니다.

이 프로토콜은 **고정한 저자 코드와 native의 구현 비교**입니다. 논문 Appendix C의 prompt epoch·momentum·batch와 현재 저자 launcher가 다르고, 선택된 가중치는 사용자가 확정한 IN21K → IN1K 계열입니다. 논문 표 수치 재현으로 부르지 않습니다.

## 실제 실행 프로토콜

| 목적 | 태스크·학습 | 전처리 | 판정·비교 범위 |
|---|---|---|---|
| 독립 기준선/native 구현 검증 | 첫2태스크, 각각 TII20 + prompt50, task2 두 CA30, batch64 | 새 `hide_prompt` config의 저자 CIFAR 전처리 | 첫태스크 및 task2 전체/과거/신규 정확도·routing·구조 검증 |
| 효율성 pilot | 첫1태스크, TII1 + prompt1, batch64, 계측3회·FLOPs1회 | 기존 `ease` 공통 pilot 전처리로 명시적 override | 양쪽 통계 산출·실제 routing·평가 비용 포함; task1 CA는 method상 해당 없음 |

두 전처리는 같은 원본 CIFAR100을 사용하지만 crop·jitter·정규화가 다릅니다. 검증용 `hide_prompt`는 저자 실제 동작인 crop scale(.08,1), bicubic, brightness jitter63/255, CIFAR mean/std를 사용합니다. 파일의 미사용 local `scale=(.05,1)`을 적용하지 않았습니다. 공통 `ease` pilot은 crop scale(.05,1)·bilinear, mean0/std1이며 검증과 다른 전처리를 결과에 기록했습니다.

검증 정확도는 동일 기준선 대비 하락이 **1.5%p 이내 통과, 1.5~3.0%p 추가 점검, 3.0%p 초과 실패**입니다. 표시 반올림 값 대신 정답수/표본수로 판정합니다. 비정상적으로 낮은 성능, 비유한 값, frozen backbone 또는 old prompt 변경, 데이터·checkpoint 불일치, oracle task routing은 구현 문제로 조사합니다. 추가 점검·실패는 효율성 측정과 구현 수치 승인으로 넘어가지 않습니다.

효율성 pilot은 TII1 + prompt1, 즉 **두 학습 단계와 두 통계 pass**입니다. 각 계측은 TII79 + prompt79 optimizer step, 학습10,000표본·통계 각5,000표본·평가1,000표본입니다. Prompt 학습 중 실제 TII forward5,000표본과 평가 routing1,000표본 비용도 포함합니다. 기존 방법의1epoch와 동일 계산 예산을 뜻하지 않습니다. 첫태스크 CA는 **N/A**이며 hook overhead만 전체 실행 비용에 포함합니다. 정렬의 비용·망각·추가 seed·Tiny/Small 및 BudgetCL/aL-SAR 예산 비교는 후속 범위입니다. 전체10태스크 결과는 [별도 보고서](full_cl/report.md)에 추가했습니다.

계측 시간·에너지는 데이터/모델 준비, 학습, 두 통계 pass, routing 평가, checkpoint와 audit 비용을 포함하고 마지막 W&B artifact upload/finish는 제외합니다. 업로드까지 포함한 launcher wall time은 별도 기록합니다. 에너지는 선택된 GPU 보드의 sampled power 적분이며 CPU/RAM/전체 머신 전력을 뜻하지 않습니다. CUDA peak는 PyTorch allocator 최대값, RSS는 프로세스와 자식의 샘플 합계입니다. 중첩 phase 시간은 inclusive라서 서로 더해 전체 시간으로 해석하지 않습니다. 세 반복의 표준편차는 동일 seed 실행의 시스템 변동이며 seed 간 성능 불확실성이 아닙니다. FLOPs는 supported operator의 부분 합을 별도 profiler 실행에서 측정하며, 그 시간·메모리·에너지는 3회 평균에서 제외합니다.

## 완료된 첫태스크 효율성 결과

[효율성 실행 원본](runs/efficiency-pilot-b64-20261007/execution.json)은 계측3회·별도 FLOPs1회를 모두 완료해 summary PASS를 기록했습니다. 다음 수치는 세 계측의 평균±표본 표준편차(`ddof=1`)입니다.

| 지표 | 평균 ± 표준편차 |
|---|---:|
| Recorder 실행 시간 | 57.093 ± 0.152s |
| 업로드/완료 알림 포함 launcher 시간 | 67.050 ± 1.491s |
| GPU 보드 에너지 | 4.9918 ± 0.0147Wh |
| CUDA allocated peak | 6486.24 ± 0.00MiB |
| CUDA reserved peak | 7014.00 ± 0.00MiB |
| 프로세스·자식 RSS peak | 15899.01 ± 45.05MiB |
| 프로세스·자식 CPU 시간 | 91.496 ± 0.224s |
| GPU 에너지 sampling coverage | 99.9367% ± 0.0083%p |
| 첫태스크 정확도 | 96.60 ± 0.00% |

Coverage 행의 평균은 비율을 백분율로 표시하고 표준편차는 %p 단위입니다. GPU device memory peak는 세 번 모두7505MiB이며, 별도 진단의 부분 supported-operator FLOPs는 **1126.989TFLOPs**입니다. 이는 unsupported 연산·CPU sklearn clustering·데이터 decoding 등을 제외한 부분 연산량입니다.

기존 측정 중 **EASE·L2P·DualPrompt**를 표·그림에 재사용했습니다. 동일 원본 데이터·전처리 fingerprint·checkpoint·batch64·seed1993·workers와 실제 precision 정책7개를 엄격히 대조했습니다. CPU thread 수는 저장된 top-level 또는 precision 필드를 정규화하며 L2P·DualPrompt와 HiDe는2개가 일치합니다. **EASE의 CPU thread 수는 기존 결과에 미기록**이므로 일치 여부를 확인하지 못했으며 값을 추정하지 않습니다. **RanPAC은 학습 전처리의 `transform_repr_sha256`가 달라 동일 입력 비교에서 제외**했습니다. 모든 후보의 기존 측정을 재사용한 것으로 해석하지 않으며, 제외 이유와 메타데이터 가용성을 [보고서](report.md)에 표시합니다.

평균 phase 시간은 TII 학습7.071s, prompt 학습19.108s, TII 통계8.445s, prompt 통계8.367s, 실제 routing을 포함한 평가2.622s입니다. 파라미터는 두 ViT를 포함해 총174,498,248개이며, TII 학습 phase의 trainable2,440,036개·prompt 학습 phase의 trainable460,900개를 기록했습니다. 짧은1+1epoch 정확도는 구현 acceptance용20+50epoch 정확도와 구분합니다. 각 worker의 W&B URL은 실행 원본에 보존합니다. 효율성 W&B 네 run도 finished/COMMITTED와 다운로드 SHA를 확인했으며, [계측1](prepared/readbacks/measurement1/readback.json)·[계측2](prepared/readbacks/measurement2/readback.json)·[계측3](prepared/readbacks/measurement3/readback.json) 각9개와 [FLOPs](prepared/readbacks/flops/readback.json) 10개 파일의 readback이 PASS입니다. 최종 집계 보고서 발행은 별도 receipt에 기록합니다.

## 학습 없는 준비 명령

```bash
python3 scripts/prepare_hide_prompt.py
python3 scripts/prepare_hide_prompt.py --check-local
```

기본 출력은 ignored `benchmarks/cl_efficiency/prepared/hide_prompt/`의 `plan.json`과 `candidate_commands.md`입니다. 두 번째 명령도 기존 파일의 SHA-256을 읽기만 하며, 파일이 없거나 다르면 실패합니다. ML 모듈·credential·센서·네트워크를 사용하지 않습니다. 생성된 native 명령은 설정 검토용이며 W&B·Slack·phase instrumentation을 갖춘 실행 runner가 아닙니다. 준비 도구에는 `--execute`가 없습니다.

실제 실행에는 [독립 저자 runner](../../../scripts/run_hide_reference.py), [native 검증 runner](../../../scripts/run_hide_validation.py), [효율성 coordinator](../../../scripts/run_hide_efficiency.py)를 사용했습니다. 위 준비 명령은 계속 실행 계획을 읽는 용도이며 실제 학습 runner와 구분합니다.

## CPU 준비 이력과 최종 감사

초기 CPU 준비에서는 집중·회귀 검사229개가 통과했고, GPU 실행 직전 runner 포함230개와 최종 publisher19개도 통과했습니다. [검증 명령과 결과](checks.md), [실행 전 기록](environment/preflight.json)에 구체적인 명령·파일 SHA를 보존합니다. [독립 network 수식 대조](environment/network_source_audit.json)의9개 조건과 [통계·alignment 대조](environment/statistics_source_audit.json)의6개 조건에서 저자 함수와 출력·gradient·파라미터가 일치했습니다. 초기 합성 계산 검사와 실제 CIFAR 학습 검증은 별개의 증거입니다.

[초기 Base CPU 구성](environment/native_cpu_initialization.json)은 기존 공유 NPZ를 읽어 두 ViT의 사전학습 텐서150개가 각각 고정 SHA와 일치함을 확인한 이력입니다. 이 CPU 단계에는 이미지 forward나 Base optimizer update가 없었습니다. 이후 실제 GPU 검증의 [최종 독립 감사](environment/full_validation_audit.json)는 source SHA, 두 branch 초기값, 데이터/변환 fingerprint, 네 단계의 첫 batch input/label/seed/clipped gradient, 140epoch의 LR/손실, task2 두 CA 및 3,000개 예측을 대조해 PASS를 기록했습니다. Task2 prompt의 epoch 평균 scalar loss 최대 차이2.42e-8은 저자 inplace FP32 regularizer 합산과 native out-of-place float64 합산의 반올림 차이로 기록하며, 결과 정확도는 모두 같았습니다.

## 최종 결과 저장 확인

[W&B 비교 보고서](https://wandb.ai/laymond1/cl-efficiency/runs/o3qej29b)에 JSON·CSV·Markdown·PNG·PDF를 저장했습니다. Run finished·artifact COMMITTED와 업로드 파일6개의 로컬/다운로드 SHA 일치를 [원격 보고서 검증](environment/wandb_report_readback.json)에서 확인했습니다. [Slack 완료 알림](environment/slack_completion.json)은 sent이며, [최종 완료 기록](environment/finalization.json)에 실행 범위와 집계를 보존합니다. GPU 실행 source17개는 실행 중 commit `92d34e6`의 hash를 유지했습니다. 기준 브랜치 병합과 원격 push는 수행하지 않았습니다.
