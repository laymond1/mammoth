# Base LoRA·RanPAC 공통 효율성 pilot

2026-10-08 사용자 승인: **“RanPAC 재측정까지 포함하여 추천안대로 진행.”**
승인한 신규 범위는 EWC-LoRA·SD-LoRA·InfLoRA·CL-LoRA·RanPAC 각 측정3회와 별도 부분 FLOPs1회, 합계20실행입니다. 기존 EASE·L2P·DualPrompt·HiDe-Prompt의 기록을 대조해9방법 표로 연결합니다.

**실제 상태는 부분 완료입니다.** EWC·Inf·CL·RanPAC16개는 통과했습니다. SD 첫 측정은10.0%(100/1000)로 격리했고 측정2회·FLOPs1회는 보류했습니다. 원본20개 계획은 유지하며 전체 완료로 표시하지 않습니다. [8방법 부분 보고서](../peft_pilot_partial/RESULTS.md), [JSON](../peft_pilot_partial/results.json), [CSV](../peft_pilot_partial/results.csv), [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/8hyym6hc)에 저장했습니다.

SD 원형 코드의 별도 공통조건1epoch 진단도10.0%였으며 첫 배치·첫 그래디언트·최종 모델의 모든 텐서·1000개 예측이 native와 완전히 같았습니다. 진단1회는20개 계획 및 효율성 평균에 포함하지 않습니다. 공통 전처리/LR0.008의 학습 안정성 원인은 미확정입니다. 임의로 학습률·epoch·통과 기준을 바꾸거나 SD를 재실행하지 않았습니다. [저자 코드 진단](https://wandb.ai/laymond1/cl-efficiency/runs/nm808wp6)과 [후속 결정 사항](../peft_pilot_partial/README.md)을 참고합니다.

[고정 계획](plan.json)은 CIFAR100 첫10클래스(train5000/test1000), 첫태스크1epoch, 공통 `ease` 전처리, ViT-B/16 AugReg IN21K→IN1K의 동일 로컬 NPZ, batch64, workers4, seed1993, GPU1, FP32와 LoRA rank10을 사용합니다. HiDe의 기존 pilot은 TII1+prompt1입니다. 각 방법 고유 학습률·optimizer와 전체 task-begin/end 작업을 유지하므로 동일 계산 예산 비교가 아닙니다.

LoRA의 검증용 원본 전처리와 이번 비교용 전처리를 분리합니다. 실제 loader의 원본 이미지·라벨·global index 순서와 train/test 전처리 hash를 기존 EASE와 대조합니다. RanPAC의 기존 학습 transform hash는 task-end 특징 추출 transform으로 덮어써졌으므로 새 worker는 학습 입력과 태스크 종료 입력을 각각 보존합니다.

방법별79 outer observe·5000 학습·1000 평가뿐 아니라 EWC의5000 Fisher 샘플, Inf의학습전5000/학습후5000 feature 샘플·SVD/GPM, CL의5000 prototype 샘플, RanPAC의 RP10000 및17후보+최종1회 CPU ridge solve를 기록합니다. SD는 별도 통계 pass가 없습니다. checkpoint·frozen weights·finite gradients·실제 예측·task state를 검사하며,20% 이하의 정확도나 비정상 저하는 조사 대상으로 처리합니다. 기존 저자 대조1.5/3.0%p 기준은 동일 설정 구현 검증의 기준으로 유지합니다.

자원 요약에는 별도 profiler 실행 비용을 넣지 않습니다. 같은 seed의3회는 실행 비용 변동을 나타내며 seed 간 성능 변동을 추정하지 않습니다. GPU 에너지는 보드 전력의 적분이고 CPU/RAM/전체 머신 에너지가 아닙니다. FLOPs는 지원 연산의 부분 추정이며 CPU 선형대수·미지원 연산 등 전체 비용을 보장하지 않습니다. generic 메모리 snapshot의 중첩 항목은 합산하지 않습니다. EASE의 과거 CPU thread 미기록, 기존 실행 시점/소스 차이, 방법별 RNG에 따른 crop 차이는 보고서에 표시합니다.

첫태스크에는 EWC의 과거 penalty, CL의 KD/이전 branch, Inf의 과거 subspace 상호작용이 없습니다. 둘째태스크 비용 진단, Small/Tiny 학습, 전체10태스크, rank/seed sweep, BudgetCL/aL-SAR 예산 controller와 원격 Git push는 후속 범위입니다.

아래 기본 명령은 모델·GPU·credential·원격 서비스를 사용하지 않고 계획만 출력합니다.

```bash
uv run --no-sync --offline python scripts/run_peft_efficiency.py
```

승인된 순차 실행과 완료 기록 재개는 다음과 같습니다. 각 worker는 별도 프로세스이며 GPU1 점유를 시작 전에 확인합니다. 완료 기록을 검증한 뒤 재개하므로 정상 완료 worker를 다시 학습하지 않습니다.

```bash
uv run --no-sync python scripts/run_peft_efficiency.py --execute
uv run --no-sync python scripts/run_peft_efficiency.py --execute --resume results/peft_pilot/base-20261008
```

로컬 원본은 `results/peft_pilot/`에 보관합니다. 완료 후 readonly report 도구로 JSON/CSV/Markdown·PNG/PDF를 작성하고, publisher로 W&B의20개 finished run·COMMITTED artifact·예측과 설정 해시·원격 기록을 확인합니다. 최종 보고서도 별도 W&B run에 저장하고 Slack에 완료를 알립니다.

위 원래 전체 보고서 명령은20개 모두 통과해야 동작합니다. 이번 부분 보고서는 원래 검사 기준을 유지하는 별도 도구로 작성·게시했습니다.

```bash
uv run --no-sync --offline python scripts/report_peft_partial.py --write
uv run --no-sync --offline python scripts/publish_peft_partial.py
```

첫 명령은 로컬 보고서를 작성하고 두 번째는 로컬 증거만 검사합니다. 추가 학습과 원격 재게시를 실행하지 않습니다. 실제 게시에는 명시적인 `--execute`가 필요하며 완료 receipt가 있는 출력 디렉터리를 덮어쓰지 않습니다.

```bash
uv run --no-sync --offline python scripts/report_peft_efficiency.py --write
uv run --no-sync python scripts/publish_peft_efficiency.py --execute
```

공유 데이터·가중치·캐시와 W&B/Slack credential 로딩은 기존 `scripts/cl_runtime.py`를 재사용합니다. 모델·backbone·과거 결과 파일은 수정하지 않습니다.
