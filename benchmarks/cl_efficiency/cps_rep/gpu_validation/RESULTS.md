# CPS-Prompt / REP Tiny 구현 검증

CIFAR100-224 unified · Tiny AugReg IN21K→IN1K · batch64 · seed1993 · FP32 · 첫2태스크 × 3epochs.

| 실행 | Task1 정확도 | Task2 seen | Task2 old | Task2 new |
| --- | ---: | ---: | ---: | ---: |
| [cps-author](https://wandb.ai/laymond1/cl-efficiency/runs/7274vm2r) | 94.40% | 81.15% | 82.70% | 79.60% |
| [cps-native](https://wandb.ai/laymond1/cl-efficiency/runs/ls9yzb8s) | 94.40% | 81.15% | 82.70% | 79.60% |
| [coda-frozen](https://wandb.ai/laymond1/cl-efficiency/runs/t7w089u5) | 94.30% | 80.35% | 80.60% | 80.10% |
| [rep-off](https://wandb.ai/laymond1/cl-efficiency/runs/e2z7gk0d) | 94.30% | 80.35% | 80.60% | 80.10% |
| [rep-on](https://wandb.ai/laymond1/cl-efficiency/runs/pcqjet72) | 91.00% | 74.80% | 74.90% | 74.70% |

CPS native↔저자 및 REP OFF↔frozen-CODA는 1.5%p 이내 통과, 1.5~3%p 점검, 3%p 초과 실패. REP ON은 seen·old·new 정확도에서 frozen-CODA보다 3%p 초과 하락 시 후속 진행을 보류한다.

| 판정 | 상태 | 최대 정확도 차이/하락 |
| --- | --- | ---: |
| cps-native | pass | 0.00%p |
| rep-off | pass | 0.00%p |
| rep-on | hold | 5.70%p |

REP ON의 차이는 task1 seen -3.30%p, task2 seen -5.55%p, task2 old -5.70%p, task2 new -5.40%p이며 판정은 hold다. 공통 입력·초기화·고정 encoder·유한 상태·과거 prompt 보존 검증은 통과했다.

REP ON의 평균 실행 블록 수는 10.32~11.16/12이며, 평균 병합 수는 75.51~82.70 tokens다. 세 구성요소를 함께 변경했고 3 epochs만 검증했으므로 하락 원인은 분리되지 않았다. 논문 해석/연결을 재점검하고 Swift·AToM·ALD별 검증을 먼저 하는 것이 다음 추천 단계다. 추가 학습은 실행하지 않았다.

다음 값은 이번 검증 실행의 관측치이며 효율성 pilot 결과가 아니다. 프로세스 시간은 모델 초기화 이후부터 검증·기록을 포함한다.

| 실행 | 검증 시간(s) | 학습 epoch 시간(s) | CUDA peak allocated(MiB) | GPU 보드 에너지(kJ) |
| --- | ---: | ---: | ---: | ---: |
| cps-author | 118.2 | 73.8 | 869.7 | 11.73 |
| cps-native | 117.8 | 73.9 | 873.0 | 11.54 |
| coda-frozen | 120.9 | 76.2 | 1389.6 | 14.49 |
| rep-off | 119.4 | 75.9 | 1390.5 | 14.78 |
| rep-on | 113.8 | 71.3 | 1350.7 | 13.10 |

## 범위와 제한

- One seed and three epochs per task; implementation verification, not paper accuracy reproduction.
- Resource costs include loader worker startup, complete input hashing and validation audits; not an efficiency pilot.
- Runs on GPU0 and GPU1 overlap; CPU contention and GPU-board idle power are included.
- GPU board energy excludes CPU/RAM/wall power. No FLOPs profiler or whole-machine energy measurement was performed.
- REP is a documented paper-v6 reconstruction, without an independent full author runner.
- Swift uses a separate Tiny query even for a Tiny host; there is no smaller-query FLOP saving at this size.
- 20epoch pilots, Small/Base and batch16/64 expansion, and the held11method comparison were not launched.

전체 입력·예측·설정·전력 샘플·태스크별 체크포인트는 실행별 로컬 폴더와 W&B artifact에 저장한다.
