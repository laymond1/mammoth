# PROL 구현 검증과 첫 태스크 효율성 pilot

고정한 공식 활성 코드를 공통 조건으로 검증합니다. LayerNorm 학습, 배치 간 indexing, 평가 중 clamp와 `GM - GM @ I` 비대각 항을 보존하며 논문 전체 성능 재현을 주장하지 않습니다.

CIFAR100-224/ease, ViT-B/16 AugReg IN21K→IN1K, GPU1, batch64, seed1993, FP32입니다. 검증과 pilot 모두 1epoch이며 nominal LR 0.01을 batch scaling하여 시작 LR 0.0025를 사용합니다.

**실행을 중단했습니다. 효율성 평균과 통과 판정을 생성하지 않습니다.**

| 실행 | 상태 | 관측 정확도 |
|---|---|---|
| validation_author | failed | {"task1_seen_accuracy": 99.3, "task1_new_accuracy": 99.3} |

완료 또는 실패한 관측을 진단 자료로 보존했습니다. 추가 GPU 실행·재학습·LR 변경은 포함하지 않습니다.

GPU 보드 에너지는 idle 전력을 포함하며 CPU/RAM/전체 머신 에너지를 제외합니다. RSS는 부모·자식의 합으로 공유 페이지를 중복 집계할 수 있고 CUDA peak는 PyTorch allocator 범위입니다. 부분 FLOPs는 미지원 연산·CPU 작업을 누락할 수 있습니다. 학습/평가/상태 점검 phase는 중첩될 수 있습니다.

첫 태스크 pilot은 전체 CL 성능·망각이나 동일 계산 예산 비교를 입증하지 않습니다. 기존 9방법 보고서와 비교 확대는 변경하지 않았습니다.

| 실행 | W&B |
|---|---|
| validation_author | [run](https://wandb.ai/laymond1/cl-efficiency/runs/2qvxzkht) |
