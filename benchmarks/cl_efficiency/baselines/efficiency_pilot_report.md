# EASE·RanPAC·L2P·DualPrompt 첫 태스크 효율성 비교

2026-10-07에 신규 세 방법의 측정9회와 별도 부분 FLOPs3회를 완료했습니다. EASE의 기존 측정3회·부분 FLOPs1회를 재사용하여 네 방법을 비교합니다. 이는 첫 태스크·1 epoch의 비용과 정확도 비교이며 전체 CL 성능이나 논문 성능 재현의 판정이 아닙니다.

## 측정 결과

각 효율성 칸은 측정3회의 중앙값 (최솟값~최댓값)입니다. FLOPs는 별도 진단1회의 값입니다.

| 방법 | 정확도 (%) | 전체 관측 시간 (s) | 학습 epoch (s) | task-end (s) | CPU 시간 (s) | GPU 에너지 (Wh) |
|---|---:|---:|---:|---:|---:|---:|
| ease | 96.50 | 29.763 (29.592~30.068) | 13.343 (13.338~13.343) | 6.541 (6.536~6.548) | 41.857 (41.775~41.943) | 2.443184 (2.437814~2.472424) |
| ranpac | 98.10 | 79.696 (79.336~82.410) | 13.734 (13.731~13.749) | 55.197 (55.096~55.277) | 140.551 (140.355~140.741) | 2.963684 (2.923512~3.001808) |
| l2p | 96.00 | 32.302 (32.203~32.623) | 19.565 (19.520~19.568) | 0.000 (0.000~0.000) | 41.667 (41.578~41.758) | 2.600428 (2.577454~2.605882) |
| dualprompt | 95.40 | 31.594 (31.539~31.687) | 18.644 (18.636~18.658) | 0.000 (0.000~0.000) | 40.913 (40.842~40.932) | 2.516095 (2.487178~2.519296) |

| 방법 | CUDA allocated peak (GiB) | CUDA reserved peak (GiB) | RSS 합계 peak (GiB) | GPU device peak (GiB) | 에너지 coverage (%) | 별도 부분 FLOPs (TFLOP) |
|---|---:|---:|---:|---:|---:|---:|
| ease | 5.625 (5.625~5.625) | 5.895 (5.895~5.895) | 13.042 (13.005~13.048) | 6.374 (6.374~6.374) | 99.8261 (99.8239~99.8274) | 562.696 |
| ranpac | 5.625 (5.625~5.625) | 6.025 (6.025~6.025) | 16.369 (16.361~16.487) | 6.505 (6.505~6.505) | 99.9454 (99.9297~99.9464) | 564.711 |
| l2p | 5.956 (5.956~5.956) | 6.371 (6.371~6.371) | 17.658 (17.657~17.766) | 6.851 (6.851~6.851) | 99.8585 (99.8488~99.8587) | 618.443 |
| dualprompt | 6.248 (6.248~6.248) | 6.852 (6.852~6.852) | 17.655 (17.306~17.751) | 7.331 (7.331~7.331) | 99.8537 (99.8065~99.9008) | 591.147 |

## Phase와 모델 상태

아래 phase 시간도 각 측정3회의 중앙값 (최솟값~최댓값)입니다. 중첩 phase와 서로 다른 실행에서 선택된 중앙값을 더해 전체 시간으로 대체하지 않습니다.

| 방법 | 초기화 (s) | 데이터 loader (s) | begin-task (s) | 학습 (s) | end-task (s) | 평가 (s) |
|---|---:|---:|---:|---:|---:|---:|
| ease | 0.632 (0.622~0.648) | 0.808 (0.803~0.808) | 0.013 (0.012~0.013) | 13.343 (13.338~13.343) | 6.541 (6.536~6.548) | 1.464 (1.463~1.464) |
| ranpac | 0.854 (0.840~0.858) | 0.731 (0.724~0.790) | 0.001 (0.001~0.001) | 13.734 (13.731~13.749) | 55.197 (55.096~55.277) | 1.512 (1.509~1.514) |
| l2p | 1.922 (1.900~1.922) | 0.689 (0.683~0.781) | 0.001 (0.001~0.001) | 19.565 (19.520~19.568) | 0.000 (0.000~0.000) | 2.489 (2.477~2.490) |
| dualprompt | 1.947 (1.946~1.978) | 0.727 (0.686~0.798) | 0.001 (0.001~0.001) | 18.644 (18.636~18.658) | 0.000 (0.000~0.000) | 2.486 (2.481~2.489) |

등록 상태의 bytes와 파라미터 수는 첫 측정의 스냅샷입니다. 학습 시작은 `after_begin_task`, 종료는 `after_end_task`입니다. 등록·auxiliary·RSS·CUDA 범주는 중복될 수 있으므로 합산하지 않습니다.

| 방법 | 학습 시작 전체 params | 학습 params | 시작 parameter bytes | 종료 registered buffer bytes | 종료 optimizer bytes | 종료 auxiliary bytes |
|---|---:|---:|---:|---:|---:|---:|
| ease | 86,995,969 | 1,197,313 | 347,983,876 | 30,741 | 4,789,252 | 0 |
| ranpac | 86,995,969 | 1,197,313 | 347,983,876 | 30,720,000 | 4,789,252 | 434,720,000 |
| l2p | 171,816,392 | 122,980 | 687,265,568 | 0 | 983,856 | 0 |
| dualprompt | 172,004,552 | 330,340 | 688,018,208 | 0 | 2,642,740 | 0 |

모든 단계의 추가 상태와 초기화 스냅샷은 JSON의 `models[method].measurements[*].model_snapshots`에 보존합니다. RanPAC의 G/Q CPU 상태와 W_rand projection 크기·공유 여부는 `method_state_storage`를 참고합니다.

## 공통 조건과 고유 설정

- 물리 GPU1, 공유 데이터/cache 재사용. `seq-cifar100-224`, 첫10클래스0~9, train5000/test1000, 클래스 재배열·noise·validation·부분 라벨 없음.
- 1 epoch·batch64·workers4·seed1993. 공통 `ease` preprocessing과 동일 AugReg IN21K→IN1K ViT-B/16 로컬 NPZ를 사용합니다.
- 가중치 SHA256: `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`. 신규 방법의 query/adaptation backbone canonical150tensor는 EASE 가중치와 일치합니다.
- FP32 파라미터·autocast 없음, CUDA matmul TF32 off, cuDNN TF32 허용·deterministic=True·benchmark=False. 전체 연산을 strict FP32로 표기하지 않습니다.
- 알고리즘 고유 LR·optimizer·prompt/RP 설정을 유지합니다. L2P·DualPrompt 입력 LR0.03은 native batch scaling 후0.0075이며 batchwise_prompt=0입니다. RanPAC LR0.01·rp_size10000·17 ridge 후보/최종 solve를 유지합니다.
- 회당 native outer79updates·5000학습·1000평가 presentations를 처리했습니다. RanPAC의 추가5000 training-image feature 수집·CPU ridge 작업, EASE의5000 prototype adapter-image pairs는 task-end 비용에 포함합니다.
- 이 비교의 명시 checkpoint와 preprocessing override는 기존 generic manifest의 기본 설정과 다릅니다. 특히 RanPAC 원래 기본 IN21K와 L2P/DualPrompt의 legacy checkpoint 성능을 검증한 결과로 표시하지 않습니다.

원본 첫 태스크 train/test에는 정확히 같은 이미지의 교차 split 쌍4개가 존재하며, 공통 비교의 raw samples를 유지하기 위해 그대로 보존했습니다. train/test overlap이0인 데이터로 표현하지 않습니다.

## 측정 해석

시간·GPU 에너지는 Recorder 관측 구간으로 import·초기화·native W&B 시작·학습·task-end·평가·감사·로컬 기록을 포함하고 마지막 artifact 업로드/finish는 제외합니다. frozen tensor CPU hash 및 finite 검사의 감사 비용도 포함합니다. 학습 phase 시간은 순수 kernel 시간과 다를 수 있습니다.

부분 FLOPs는 별도 profiler 실행의 지원 연산 합계입니다. CPU ridge solve·지원하지 않는 linear algebra와 fused/optimizer 연산은 누락될 수 있습니다. 학습만의 FLOPs나 phase별 FLOPs, 완전한 계산 예산으로 해석하지 않으며 backward에 임의 배수를 곱하지 않습니다. 진단 실행의 시간·메모리·CPU 시간·에너지는 측정 통계에 포함하지 않았습니다.

진단 pass에서는 profiler 실행 중 긴 power-sampling 간격이 발생했습니다. GPU 에너지 coverage는 아래와 같으며 진단 에너지·시간·메모리를 효율성 비교에 사용하지 않습니다. 측정 pass의 coverage와 구분합니다.

| 별도 진단 방법 | GPU 에너지 coverage (%) | 효율성 통계 포함 |
|---|---:|---|
| ease | 61.1845 | 제외 |
| ranpac | 81.1759 | 제외 |
| l2p | 83.1167 | 제외 |
| dualprompt | 66.3054 | 제외 |

GPU 에너지는 선택한 GPU 보드 power sample의 적분이며 CPU/RAM/전체 머신 에너지는 측정하지 않았습니다. RSS 합계의 공유 페이지는 중복 집계될 수 있고 device memory/power에는 다른 GPU 프로세스가 영향을 줄 수 있습니다. CPU 시간은 worker와 종료한 children의 user+system 누적값으로 살아 있는 child는 누락될 수 있으며 여러 thread/process 때문에 벽시간보다 클 수 있습니다.

seed1993의 반복3회는 실행 비용의 범위를 보여줍니다. seed 간 정확도 변동을 추정하지 않습니다. 첫 태스크만으로 forgetting/BWT, 여러 태스크의 prompt routing, EASE adapter 누적 비용이나 RanPAC 후속 태스크의 backward 제거 효과를 판정하지 않습니다. 신규 방법의 정확도20% 초과는 심한 붕괴 탐지용 하한이며 EASE의20-epoch 구현 검증 허용폭을 적용한 것이 아닙니다.

## 기록과 제외한 시도

[JSON 결과](efficiency_pilot_results.json)·[측정12행 CSV](efficiency_pilot_results.csv)와 [신규 원시 로그](/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.worktrees/cl-baselines/benchmarks/cl_efficiency/baselines/runs/baseline-efficiency-pilot-20261007T012659Z)를 보존합니다. Raw run 폴더는 ignored이며 Git 병합으로 복사되지 않습니다.

EASE는 source `49897a3`의 기존 측정을 재사용합니다. [변경하지 않은 EASE 결과](../ease/efficiency_pilot_results.json)의 SHA256은 `dc84f34f585ebd93e2add569acecd379170cd4a48c9c5779808201c401f41aa9`입니다. 신규 세 방법은 source `20ec3237699ccfd12a3fb07e74f70a53e1fba79f`를 고정하여 실행했습니다. 현재 worker에 후속 감사/실행 기능이 추가되어 과거 EASE worker source hash와 다를 수 있으므로 과거 snapshot을 현재 전체 source 일치의 증거로 해석하지 않습니다. EASE를 다시 학습하거나 새로 측정하지 않았습니다. EASE와 신규 세 방법은 서로 다른 시각에 순차 측정했으며 동시·무작위 순서 비교가 아닙니다. 같은 데이터·가중치·GPU·precision 조건을 맞췄지만 시각과 계측 source 차이까지 제거한 비교로 해석하지 않습니다.

[제외한 시도](environment/excluded_attempts.json)는 CLI `--pretrained 0`이 True로 해석되어 추가 HF checkpoint를 먼저 로드한 첫 시도입니다. RanPAC measurement3개가 시작됐고 계산은3개 완료했으나 logging까지 정상 완료한 것은2개이며3번째는 logging 중 중단됐습니다. 세 실행의 최종 canonical 가중치는 올바르지만 초기화 protocol이 달라 정확도·비용 등 모든 관측값을 이 비교에서 제외했습니다. Raw logs와 W&B URL은 삭제하지 않았습니다.

| 방법 | 측정1 | 측정2 | 측정3 | 별도 FLOPs |
|---|---|---|---|---|
| ease | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/zrctmk8h) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/fzn84ciy) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/qitfg8hb) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/tuk4smi4) |
| ranpac | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/1mgxgcav) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/qwtxt9zi) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/lc7ypbcl) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/7jm9mkdt) |
| l2p | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/6mrhex87) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/e3zw7v1l) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/e0l8cncz) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/1bmon0eh) |
| dualprompt | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/cejvk8hh) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/ftkmmfzv) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/6xdwgmqh) | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/5ft4jzyn) |

[독립 감사](/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.worktrees/cl-baselines/benchmarks/cl_efficiency/baselines/environment/independent_pilot_audit.json)에 별도 검증 결과를 보존합니다.

[W&B 서버 확인](/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.worktrees/cl-baselines/benchmarks/cl_efficiency/baselines/environment/wandb_pilot_readback.json)에 별도 검증 결과를 보존합니다.

Tiny·Small, 추가 seeds, 전체 CL 재현, HiDe-Prompt 구현, BudgetCL/aL-SAR 예산 비교와 원격 push는 이번 실행 범위에 포함하지 않았습니다. 로컬 병합과 원격 저장 최종 확인은 [검증 기록](checks.md)에 기록합니다.

검증된 실행 코드 `20ec323`을 `codex/cl-foundation`에 로컬 fast-forward 병합하고 동일 focused suite **490개**를 다시 통과했습니다. 공개 보고서 및 저장 확인 기록도 후속 커밋으로 같은 브랜치에 병합합니다. 원격 push는 수행하지 않습니다.

[W&B 4방법 집계 보고서](https://wandb.ai/laymond1/cl-efficiency/runs/pfg5ytbp)에 중앙값·범위와 부분 FLOPs 및 결과 artifact를 저장했습니다. 기존 결과를 업로드한 report-only 기록으로 추가 학습은 없습니다.
