# EASE Base 전체 10태스크 — HiDe와 공통 입력 조건

사용자 승인으로 EASE Base·adapter64를 기존 HiDe 전체 실험과 같은 CIFAR100-224 전처리에서 검증하고 실행합니다. GPU1·batch64·workers4·seed1993, 동일 AugReg IN21K → IN1K NPZ, FP32와 공유 데이터·캐시를 사용합니다. HiDe의 crop·augmentation·normalization만 공통 입력 기준으로 적용하며 EASE는 SGD lr0.025·momentum0.9·weight decay0.0005·cosine·20epoch/task를 유지합니다.

독립 저자 코드 기준선 첫2태스크 → native 첫2태스크 → 모든 정확도·구조 검사 통과 후 native 전체10태스크 순서입니다. 기준선과 native는 같은 초기 adapter·proxy 파라미터 및 태스크/epoch RNG를 사용하고 실제 첫 배치 입력을 대조합니다. 기준선 대비 하락≤1.5%p는 통과, 1.5~3.0%p는 추가 점검, >3.0%p는 실패이며 구조 오류도 진행을 차단합니다. 전체 실행의 첫2태스크 예측은 통과한 native 검증과 정확히 일치해야 합니다.

첫2태스크는 40epoch·3,160update·학습 입력200,000개·누적 평가3,000개, 전체10태스크는 200epoch·15,800update·학습 입력1,000,000개·누적 평가55,000개를 완료해야 합니다. Prototype 추출·과거 feature 공간 보완·분류기 평가의 추가 연산과 메모리도 별도로 기록합니다. 알고리즘·전처리·학습 source와 가중치/초기값을 실행 전에 hash로 고정합니다.

기존 batch48·`ease` 전처리의 EASE 결과와 HiDe 결과를 보존하고, 새 조건의 JSON·CSV·보고서·과학 그래프와 W&B artifact를 별도로 저장합니다. 원격 파일을 다시 읽어 SHA 일치를 확인한 뒤 기존 Slack webhook으로 완료 알림을 보냅니다. 자세한 승인 범위와 진행 상태는 [실행 계획](execution_plan.json)에 기록합니다.

최종 정확도·평균 증분 정확도·과거 태스크 망각과 전체/태스크/phase 시간, CPU 시간·RSS, CUDA allocated/reserved 및 NVML device memory, GPU 보드 에너지와 sampling coverage를 보고합니다. GPU 에너지는 CPU/RAM을 포함한 전체 머신 에너지가 아니며, 중첩 phase 시간은 단순 합산하지 않습니다. 마지막 artifact upload/finish는 Recorder 구간에서 제외합니다.

EASE20epoch/task와 HiDe TII20+prompt50 및 CA30은 서로 다른 학습량입니다. 같은 입력 조건의 실제 실행 비용과 성능을 비교하며 동일 계산 예산으로 해석하지 않습니다. 한 seed의 관측이고, 공통 전처리에 적응한 저자 기준선과의 구현 검증입니다. 논문 표 재현·추가 seed·Tiny/Small·BudgetCL/aL-SAR 예산 비교는 후속 범위입니다.
