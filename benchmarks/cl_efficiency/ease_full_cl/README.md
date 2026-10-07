# EASE Base 전체 10태스크 — HiDe와 공통 입력 조건

EASE Base·adapter64·seed1993의 독립 저자/native 첫2태스크 검증과 native 전체10태스크 실행을 완료했습니다. 최종 class-IL 정확도 **85.42%**, 평균 증분 정확도 **91.245476%**, 과거9태스크 평균 망각 **8.00%p**입니다. [전체 보고서](report/report.md), [JSON](report/results.json), [태스크·단계별 CSV](report/results.csv), [정확도 그래프](report/accuracy_matrix.png), [자원 그래프](report/task_resources.png)에 기록합니다. PDF도 같은 report 디렉터리에 보존합니다.

전체 Recorder 시간은 **3,493.678616초(58분14초)**, GPU 보드 에너지는 **404.253873Wh**, power sampling coverage는 **99.998708%**입니다. CUDA allocated/reserved peak는 **5,804.67/6,218.00MiB**, NVML device peak는 **6,709.00MiB**, 프로세스·자식 RSS peak는 **22,360.66MiB**, CPU 시간은 **4,467.72초**입니다. 초기화·학습·prototype·평가·checkpoint audit을 포함하며 마지막 artifact upload/finish를 제외합니다. 태스크 interval은 태스크 종료 후 Slack/W&B logging을 제외하지만 epoch logging은 포함합니다. 중첩 phase 시간을 합산하지 않습니다. GPU 에너지는 선택 보드의 측정값이며 CPU/RAM을 포함한 전체 머신 에너지는 측정하지 않았습니다.

독립 저자 기준선과 native 첫2태스크의 정확도 네 지표는 **99.20%·96.40%·95.70%·97.10%**로 일치했습니다. 정확도 하락≤1.5%p 통과, 1.5~3.0%p 추가 점검, >3.0%p 실패 기준을 모두 통과했고, native 구조 검사12개도 통과했습니다. 전체 실행의 첫2태스크40epoch 기록과 실제 입력·gradient·예측도 accepted native와 정확히 일치합니다. [검증 자료](environment/matched_validation.json), [기준선 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/de64hlyq), [native 검증 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/a53s6tl8)에 근거를 보존합니다. 이 두 실행의 자원 수치는 같은 GPU의 별도 normalization ablation과 겹쳐 효율성 비교에서 제외합니다.

GPU1·batch64·workers4·seed1993, ViT-B/16 AugReg IN21K → IN1K NPZ와 FP32를 사용했습니다. CIFAR100 고정 순서10클래스/태스크에서 기존 HiDe 전체 실험의 train/test crop·augmentation·normalization을 그대로 적용합니다. EASE 학습은 SGD lr0.025·momentum0.9·weight decay0.0005·cosine·20epoch/task를 유지했습니다. 공유 데이터 `/mnt/sdab1/share/data`와 `/mnt/sdab1/userHome/wslim/.cache`의 가중치를 재사용했습니다. 실행25파일의 SHA를 사전에 고정한 source는 `0b2b788d80c8cbce0cd38b6c312907552e38abdf`입니다.

실제 **200epoch·15,800optimizer update·학습 입력1,000,000개·누적 평가55,000개**를 완료했습니다. Prototype 입력50,000개와 adapter-image pair275,000개, 평가 adapter-image pair385,000개를 기록합니다. Pair는 한 이미지를 하나의 adapter 공간에 통과시킨 관측 횟수이며 고유 이미지 수나 FLOPs가 아닙니다. 분류기 prototype100×7680과10adapter, frozen backbone/old adapter 보존·초기값·checkpoint round-trip 등12구조 검사도 통과했습니다.

다중 태스크의 prototype 보완과 reweighting은 저자 원문 함수와 CPU에서 태스크3~10 조건을 대조해 통과했습니다. [감사 명령·해석](environment/semantic_completion_audit.md), [재현 코드](environment/audit_semantic_completion.py), [수치·SHA 증거](environment/semantic_completion_audit.json)를 보존합니다. 독립 전체10태스크 이미지 정확도 기준선이나 논문 표 재현으로 판정하지 않습니다.

전체 효율성 실행은 별도 GPU1 ablation9개 종료와 빈 GPU preflight를 확인한 뒤 시작했습니다. [GPU 점유 증거](environment/gpu_exclusivity_full_cl.json)의5초 간격678개 관측에서 foreign process/조회 오류는0개였으며 worker PID·spec/result·console·preflight SHA를 연결했습니다. 모니터 시작 전 공백과 sampling 사이 구간은 포함하지 않아 연속 독점이나 device lock을 뜻하지 않습니다. Power 계측 coverage와 GPU 점유 관측 범위는 별개입니다.

HiDe와 실제10태스크 raw 데이터·전처리·첫 입력·가중치·batch/seed/workers/GPU/precision 대조를 통과했습니다. 같은 입력에서 EASE는 최종85.42%·58분14초·404.25Wh, HiDe는94.27%·3시간15분35초·1,331.81Wh였습니다. EASE20epoch/task와 HiDe TII20+prompt50 및 CA30은 학습 작업량이 다르므로 동일 계산 예산 비교가 아닙니다. 이전 batch48·`ease` 전처리 결과는 보존하며 새 조건의 수치와 섞지 않습니다.

[전체 학습 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/ru1nmce0)의21개 고정 파일과 [보고서 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/xfrn5if3)의9개 고정 파일을 원격에서 다시 읽어 finished/COMMITTED·SHA 일치를 확인했습니다. [학습 readback](environment/full_cl_readback.json), [보고서 readback](environment/report_readback.json), [Slack 완료 수신 요청](environment/completion_notification.json)을 보존합니다. 태스크별 진행 알림10개와 최종 완료 알림은 모두 `sent`입니다. `sent`는 webhook 요청 성공 기록입니다.

검증 명령과 결과는 [validation_checks.json](environment/validation_checks.json)에, 승인 범위·실행 상태는 [execution_plan.json](execution_plan.json)에 기록합니다. 관련 CPU 테스트268개와 GPU proof 보고서 테스트43개, 실제 CPU 초기화/forward smoke 및 저장 예측 재계산을 통과했습니다. 두 테스트 묶음은 중복을 포함하므로 합산하지 않습니다.

한 seed의 한 실행으로 seed 변동이나 신뢰구간은 추정하지 않습니다. 논문 성능 재현·추가 seed·HiDe Tiny/Small·BudgetCL/aL-SAR 예산 비교·전체 FLOPs profiling은 후속 범위입니다.
