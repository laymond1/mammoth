# RainbowPrompt의 공식 활성 경로 구현·검증

Mammoth 구현과 CPU 검증을 완료했고, 다음 **GPU 8회 실행과 로컬·W&B 저장·Slack 알림**을 승인받았습니다. [run_plan.json](run_plan.json)이 현재 승인된 실행 조건입니다. 원형 저자/native 첫 2태스크 검증 2회, 단축 저자/native 첫 태스크 검증 2회가 각각 통과하면 native 효율성 측정 3회와 별도 FLOPs 측정 1회를 진행합니다. 실제 완료 여부와 결과는 실행 기록 및 검증 보고서로 확인합니다.

**8회 실행을 모두 완료했습니다.** 원형·단축 저자/native 정확도 차이는 모두0.0%p이며 예측·gradient·학습 상태도 동일했습니다. [최종 검증·효율성 보고서](../rainbow_validation/RESULTS.md), [측정 범위·W&B 및 원격 내용 검증](../rainbow_validation/README.md), [실행 도구 CPU 검사](RUN_VALIDATION.md)를 기록했습니다. 실행 도구·수치 경로는 commit `08f07ae`로 고정했고, 기존 공통 데이터와 가중치를 재사용했습니다.

구현 기준은 [공식 저장소](https://github.com/Kiseong0753/RainbowPrompt)의 commit `0b84d3e3738669fe4bb516ccc1c526a535576ad3`입니다. 로컬 원본은 `/tmp/cl-rainbow-official-20261008`에 있으며, [source_review.json](source_review.json)에 전체 tracked 파일의 SHA-256과 활성 경로를 기록합니다. 사용자가 선택한 **공식 코드의 활성 경로 보존**을 따릅니다. 논문의 학습 가능한 gate를 새로 연결하는 작업은 후속 범위입니다.

[논문](https://arxiv.org/html/2507.22553v1#S3.SS3)은 task별 Gumbel-Softmax gate와 sparsity loss를 포함합니다. 고정한 공개 코드의 실행 경로에는 그 두 기능이 연결되어 있지 않습니다. 따라서 이번 구현의 의미는 `official-active-path`이며 논문의 전체 알고리즘이나 발표 성능을 재현했다는 뜻이 아닙니다. [논문 최적화 정의](https://arxiv.org/html/2507.22553v1#S3.SS4)

## 실제 활성 경로

Frozen ViT의 CLS 특징을 얻는 별도 query forward와 prefix prompt를 삽입하는 ViT forward를 사용합니다. 학습 중 현재 태스크의 base prompt와 task key를 선택하고, 이전 base prompt는 detach한 뒤 현재/이전 prompt를 변환·정렬하여 평균을 냅니다. prefix의 앞 절반은 key, 뒤 절반은 value입니다. 평가에서는 학습 중 저장한 prompt와 학습된 task key를 이용합니다. [공식 prompt 구현](https://github.com/Kiseong0753/RainbowPrompt/blob/0b84d3e3738669fe4bb516ccc1c526a535576ad3/prompt.py)

| 확인한 동작 | 보존할 의미와 한계 |
|---|---|
| `task_conditioning_step`의 attention | 동일한 task key를 prompt 길이 전체에 반복한 뒤 `base @ repeated_key.T`를 계산합니다. 각 row의 logit이 같으므로 수학적으로 uniform softmax가 되어 base prompt의 길이 방향 평균을 반복합니다. task key에 따른 비균등 attention으로 수정하지 않습니다. |
| 변환·정렬 | Q/K/V를 D2로 줄이고 token/feature 방향 attention 후 원래 차원으로 복원합니다. D1 폭의 ReLU FFN과 residual layer normalization을 적용합니다. self-attention layer는 현재 prompt만 KI_iter회 변환하며 다른 layer는 현재/이전 태스크별 결과를 평균합니다. |
| 저장 prompt | Rainbow 학습 forward에서 `stored_rainbow_prompts`를 갱신합니다. optimizer step 이전 값이므로 마지막 step 뒤 다시 계산한 prompt가 아닙니다. 태스크 종료 시 임의 재계산하지 않습니다. |
| dropout | 변환 내부 `F.dropout(..., 0.1, training=True)`를 보존합니다. 평가 경로는 저장 prompt를 읽어 변환을 실행하지 않으므로 이 dropout을 다시 적용하지 않습니다. |
| task routing | query CLS와 task key의 cosine similarity를 **minibatch 전체에서 평균**하여 하나의 태스크를 고릅니다. 평가 task label로 선택하거나 샘플별 routing으로 변경하지 않습니다. 따라서 평가 batch 구성·순서를 함께 고정해야 합니다. |
| loss | 현재 태스크 클래스의 CE에서 balancing×`sim_loss`를 뺍니다. 반환되는 sim_loss는 마지막 prompt layer의 값입니다. paper gate의 sparsity loss는 활성 경로에 없습니다. |
| gate·동결 helper | `adaptive_prompting.py`의 soft/hard gate 함수와 `freeze_components`는 활성 trainer에서 호출하지 않습니다. 이를 임의 연결하지 않습니다. 공유 prompt 변환기·정렬기는 학습 가능한 상태로 유지합니다. |
| 텐서 보관·backward | 원본 prompted attention의 마지막 query 보관과 `backward(retain_graph=True)`를 유지합니다. 별도 unprompted query encoder는 원본처럼 attention query를 보관하지 않습니다. 자원 측정에서 이 동작을 기록합니다. |
| 평가 class 범위 | 공식 class-IL evaluator는 기본적으로 미래 classifier logit도 남깁니다. Mammoth의 seen-class 평가와 같게 대조하려면 독립 기준선에도 같은 seen-class mask를 명시적으로 적용해야 합니다. 이 적응은 결과에 기록합니다. |

[공식 ViT 경로](https://github.com/Kiseong0753/RainbowPrompt/blob/0b84d3e3738669fe4bb516ccc1c526a535576ad3/vision_transformer.py#L505), [공식 학습·평가](https://github.com/Kiseong0753/RainbowPrompt/blob/0b84d3e3738669fe4bb516ccc1c526a535576ad3/engine.py), [미연결 gate helper](https://github.com/Kiseong0753/RainbowPrompt/blob/0b84d3e3738669fe4bb516ccc1c526a535576ad3/adaptive_prompting.py)에 근거한 정적 코드 검토입니다. CPU oracle 결과는 별도 checks 기록으로 확인하며 이 문서 자체를 실제 데이터 학습 검증으로 해석하지 않습니다.

## 기본 설정과 공통 조건

10태스크용 공개 shell 명령은 주석으로 보존되어 있으며, 실제 활성 shell 명령은 20태스크 설정입니다. 이번에는 공개 10태스크 명령의 method hyperparameter를 사용하고 데이터 capacity는 10태스크로 유지합니다. config 파일의 축소된 기본값과 shell 명령의 값을 섞지 않습니다. [CIFAR 공개 명령](https://github.com/Kiseong0753/RainbowPrompt/blob/0b84d3e3738669fe4bb516ccc1c526a535576ad3/run_cifar100.sh)

| 항목 | 준비 조건 |
|---|---|
| 데이터 | seq-cifar100-224/ease, 원래 클래스 0–99, 10태스크×10클래스, 첫 태스크 train 5000/test 1000 |
| backbone | ViT-B/16, AugReg IN21K→IN1K, 기존 로컬 NPZ와 frozen encoder SHA 유지 |
| method | prompt length 20(K10/V10), pool 10, top-k 1, D1=56, D2=96, KI_iter=10, 모든 12개 layer에 prefix, self-attention layer 0–5 |
| 원형 검증 | 태스크당 20epochs, 앞 5epochs Unique warm-up 후 Rainbow evolution |
| optimizer | Adam, weight decay 0, eps 1e-8, betas 0.9/0.999, clip norm 1, constant LR, 태스크마다 optimizer 재생성 |
| LR | 공개 nominal LR 0.03을 batch/256으로 조정. 공통 batch 64에서는 실제 LR 0.0075. 공개 shell의 batch 32에서는 0.00375 |
| 공통 실행 | batch 64, workers 4, seed 1993, FP32, Torch CPU threads 2, 물리 GPU 1 |
| 데이터·캐시 | /mnt/sdab1/share/data, /mnt/sdab1/userHome/wslim/.cache 및 기존 torch/HF/CLIP 하위 경로 재사용 |
| 저장·알림 | 로컬 기록과 laymond1/cl-efficiency의 W&B, 기존 Slack webhook 설정 재사용 |

공식 실제 train transform은 RRC224(scale 0.05–1.0, bilinear)·HFlip0.5·ToTensor이며 test는 Resize256 bicubic·CenterCrop224·ToTensor입니다. `pretrained_cfg`의 mean/std 메타데이터와 달리 loader는 Normalize를 적용하지 않습니다. 공통 EASE의 identity Normalize는 수치적으로 같지만 transform repr이 다르므로 향후 기준선에도 같은 공통 transform을 적용하여 실제 fingerprint를 확인합니다. [공식 데이터 전처리](https://github.com/Kiseong0753/RainbowPrompt/blob/0b84d3e3738669fe4bb516ccc1c526a535576ad3/datasets.py#L202)

Base 10태스크 설정에서 저장 prompt buffer는 구조상 7,372,800bytes(7.031MiB)이며 base prompt pool도 같은 크기입니다. Q/K/V/dense와 FFN의 학습 파라미터는 12layer 합계 4,593,696개입니다. 공개 `main.py`의 일부 parameter 출력은 이 변환기들을 제외합니다. 실제 parameter/optimizer/query backbone 저장량을 함께 계측할 계획이며 위 숫자는 구조 계산으로 실제 메모리 측정값이 아닙니다.

## 승인된 GPU 검증·측정

[validation_plan.json](validation_plan.json)은 이전 준비안을 보존한 문서입니다. 현재 [run_plan.json](run_plan.json)에 따라 저자 활성 코드와 native의 첫 2태스크를 같은 조건으로 대조합니다. 전체 10태스크·Small/Tiny·추가 seed·계산 예산 비교는 이번 계획에서 제외합니다. 기존 기준인 하락≤1.5%p pass, 1.5–3.0%p review, 3.0%p 초과 fail 및 구조 검사를 유지합니다. 비정상적으로 낮은 정확도나 구조 실패는 상대 정확도 차이와 별도로 후속 실행을 중단합니다.

**1epoch와 warm-up 5를 함께 두면 저장 RainbowPrompt가 갱신되지 않습니다.** 효율성 pilot은 사용자가 선택한 **1epoch/warm-up 0**을 사용합니다. 원형 20epoch/warm-up 5 검증과 구분하며, 단축 조건의 독립 저자/native 대조가 통과해야 측정을 시작합니다.

모든 실행은 GPU 1에서 새 모델과 같은 초기 상태로 시작하며, 데이터 순서·전처리·phase RNG·평가 batch를 공유합니다. 실제 Mammoth trainer와 `meta_*` 경로를 사용하고, 독립 기준선은 고정한 저자 코드의 수치 학습 경로를 사용합니다. 현재 클래스 CE와 seen-class 평가 범위 적응은 결과에 기록합니다. 학습 중 저장 cache와 optimizer update의 순서, warm-up cache, 이전 태스크 상태, 배치 평균 routing, frozen backbone, checkpoint 복원을 검사합니다.

[execution_plan.json](execution_plan.json)의 `training_authorized=false`는 이전 준비 단계의 기록으로 유지합니다. 이번 승인·실행 계획은 별도 [run_plan.json](run_plan.json)에 기록하며, [run_rainbow_suite.py](../../../scripts/run_rainbow_suite.py)가 두 검증을 통과한 경우에만 측정을 진행합니다. 효율성 평균에는 native 측정 3회만 포함하며 검증 실행과 FLOPs 실행은 제외합니다. GPU board energy는 전체 시스템 에너지가 아니며, FLOPs는 지원되는 연산만의 부분 추정입니다. 이번 단계의 결과는 별도 보고서로 저장하고 기존 9방법 비교 확대는 진행하지 않습니다.
