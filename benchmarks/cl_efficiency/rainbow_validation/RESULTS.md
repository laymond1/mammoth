# RainbowPrompt 구현 검증과 첫 태스크 효율성 pilot

고정한 공식 코드의 활성 경로를 검증했습니다. 학습 가능한 gate와 sparsity loss는 연결되어 있지 않으며 논문 전체 성능 재현을 주장하지 않습니다.

CIFAR100-224/ease, ViT-B/16 AugReg IN21K→IN1K, GPU 1, batch64, seed1993, FP32입니다. 첫 2태스크 검증은 공식 20epochs·warm-up 5 일정을 유지하면서 batch·seed·전처리·seen-class 평가를 공통 조건으로 맞췄습니다. 단축 첫 태스크와 pilot은 **1epoch·warm-up 0**인 별도 recipe입니다.

| 검증 | 지표 | 저자 % | native % | 하락 %p | 판정 |
|---|---|---:|---:|---:|---|
| full | task1_seen_accuracy | 99.20 | 99.20 | 0.00 | pass |
| full | task2_seen_accuracy | 97.30 | 97.30 | 0.00 | pass |
| full | task2_old_accuracy | 98.90 | 98.90 | 0.00 | pass |
| full | task2_new_accuracy | 95.70 | 95.70 | 0.00 | pass |
| short | task1_seen_accuracy | 98.40 | 98.40 | 0.00 | pass |

정확도는 저장된 정답 수와 샘플 수의 정확한 분수로 계산했습니다. 하락≤1.5%p pass, 1.5–3.0%p review, 3.0%p 초과 fail이며 17구조 검사·초기 상태·실제 배치·평가 batch 순서도 대조했습니다. 첫 태스크의 저자/native 정확도 90% 하한과 태스크2의 20% 붕괴 탐지 하한을 적용했습니다. 태스크2 하한은 충분한 CL 성능을 보증하는 기준이 아닙니다.

| 첫 태스크 측정 | 평균 | 표본 SD |
|---|---:|---:|
| process_wall_s | 34.4969 | 0.0733 |
| process_tree_cpu_s | 45.0946 | 0.1559 |
| gpu_board_energy_wh | 2.7852 | 0.0272 |
| cuda_peak_allocated_mib | 12383.0400 | 0.0000 |
| peak_process_tree_rss_mib | 15768.1862 | 180.5720 |
| accuracy_pct | 98.40 | 0.00 |

별도 지원 연산 부분 FLOPs: **603.753 TFLOPs**.

측정 3회의 평균·표본 SD에 저자/native 검증 4회와 FLOPs 실행의 비용을 포함하지 않았습니다. 동일 seed의 비용 변동이며 seed 간 성능 불확실성이 아닙니다. GPU 보드 에너지는 CPU/RAM/전체 머신 에너지를 포함하지 않습니다. 부분 FLOPs는 미지원 연산과 CPU 작업을 누락할 수 있습니다.

첫 태스크 pilot은 전체 CL 성능·망각이나 동일 계산 예산 비교를 입증하지 않습니다. 신규 방법 구현 이후로 미룬 비교 확대와 기존 9방법 보고서는 이번 단계에 포함하지 않습니다.

| 실행 | W&B |
|---|---|
| full_author | [run](https://wandb.ai/laymond1/cl-efficiency/runs/re2bv0bf) |
| full_native | [run](https://wandb.ai/laymond1/cl-efficiency/runs/l4jra261) |
| short_author | [run](https://wandb.ai/laymond1/cl-efficiency/runs/4ahxenl5) |
| short_native | [run](https://wandb.ai/laymond1/cl-efficiency/runs/kuyl4nxq) |
| measurement1 | [run](https://wandb.ai/laymond1/cl-efficiency/runs/hfwbbdy0) |
| measurement2 | [run](https://wandb.ai/laymond1/cl-efficiency/runs/7o1yjgdh) |
| measurement3 | [run](https://wandb.ai/laymond1/cl-efficiency/runs/dzh6il94) |
| flops | [run](https://wandb.ai/laymond1/cl-efficiency/runs/hiic8qut) |
