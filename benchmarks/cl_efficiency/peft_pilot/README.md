# Base LoRA·RanPAC 공통 효율성 pilot

2026-10-08 사용자 승인: **“RanPAC 재측정까지 포함하여 추천안대로 진행.”**
승인한 신규 범위는 EWC-LoRA·SD-LoRA·InfLoRA·CL-LoRA·RanPAC 각 측정3회와 별도 부분 FLOPs1회, 합계20실행입니다. 기존 EASE·L2P·DualPrompt·HiDe-Prompt의 기록을 대조해9방법 표로 연결합니다.

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

```bash
uv run --no-sync --offline python scripts/report_peft_efficiency.py --write
uv run --no-sync python scripts/publish_peft_efficiency.py --execute
```

공유 데이터·가중치·캐시와 W&B/Slack credential 로딩은 기존 `scripts/cl_runtime.py`를 재사용합니다. 모델·backbone·과거 결과 파일은 수정하지 않습니다.
