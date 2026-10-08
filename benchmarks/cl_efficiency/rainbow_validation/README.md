# RainbowPrompt 검증·측정 완료

2026-10-08, GPU 1의 승인된 8회 실행을 모두 완료했습니다. [전체 결과](RESULTS.md), [JSON](results.json), [CSV](results.csv), [실행 조건](../rainbow_prompt/run_plan.json), [CPU172검사·초기 상태](../rainbow_prompt/RUN_VALIDATION.md)를 기록합니다. 모든 실행은 ViT-B/16 AugReg IN21K→IN1K·CIFAR100-224/ease·batch64·seed1993·workers4·FP32를 사용합니다.

첫2태스크의 20epochs/warm-up5 원형 대조에서 저자/native는 첫태스크99.2%, 태스크2 전체97.3%·기존98.9%·신규95.7%로 동일했습니다. 별도1epoch/warm-up0 대조도98.4%로 동일했습니다. 17구조 검사와 실제 데이터·배치·RNG 검사를 통과했으며 모든40epoch의 첫배치 gradient, 두태스크의 최종 학습 상태와 예측 배열까지 일치했습니다. 학습 가능한 gate·sparsity loss가 없는 공식 활성 코드 기준이며 논문 전체 재현을 주장하지 않습니다.

효율성은 독립된 native 측정3회의 평균±표본 SD입니다. 같은 seed의 실행 비용 변동으로 해석합니다.

| 지표 | 평균±표본 SD |
|---|---:|
| 정확도 | 98.40±0.00% |
| 학습 구간 | 22.7150±0.0307초 |
| 전체 관측 구간 | 34.4969±0.0733초 |
| 프로세스 트리 CPU 시간 | 45.0946±0.1559초 |
| CUDA peak allocated | 12.0928±0.0000GiB |
| 프로세스 트리 peak RSS | 15.3986±0.1763GiB |
| GPU 보드 에너지 | 2.7852±0.0272Wh |
| 별도 지원 연산 부분 FLOPs | 603.753TFLOPs |

전체 관측 구간은 모델·데이터 준비, 학습·평가, 실제 경로 검사, 작은 checkpoint 저장·복원과40샘플 출력 probe를 포함하고 최종 W&B artifact 업로드는 제외합니다. probe는 일반 평가1000샘플 수에서 제외하지만 시간·에너지·FLOPs에는 포함합니다. 학습 구간도 실제 gradient·cache 검사 비용을 포함합니다. 포함 관계가 있는 phase 비용을 더해 전체 비용으로 해석하지 않습니다.

CUDA 메모리는 PyTorch allocator의 초기화 포함 최고치이며 비 PyTorch GPU 할당은 제외합니다. RSS는 부모와 자식의 합으로 공유 페이지가 중복 집계될 수 있어 호스트의 고유 물리 메모리 사용량이 아닙니다. GPU 에너지는 물리 GPU 보드의 유휴 전력까지 포함하며 CPU/RAM/전체 시스템 에너지와 구분합니다. 센서 오류는 없고 전력 coverage는99.86% 이상이었습니다. FLOPs는 초기화·학습·평가·검사·probe의 지원 연산 부분 합이며 미지원 연산·PIL 처리·optimizer 등이 누락될 수 있습니다. 검증4회와 FLOPs 실행의 비용은 측정3회 평균에서 제외합니다.

학습 가능 파라미터는6,521,476개, frozen 파라미터는171,674,212개입니다. 저장 prompt cache7,372,800bytes는 최대10태스크 용량을 선할당하며 현재 학습한 슬롯만 채워집니다. 원본 replay는 없습니다. 첫태스크 비용만으로 전체 CL 성능·망각·동일 계산 예산 비교를 판단하지 않습니다. 신규 방법 구현 이후의 비교 확대·전체10태스크·Small/Tiny·추가seed는 이번 실행 범위에 포함하지 않습니다.

[최종 W&B 보고서](https://wandb.ai/laymond1/cl-efficiency/runs/zlxinfgf)에 작은 결과 artifact를 저장했습니다. 8개 원본 run의 `finished`, artifact `COMMITTED`, 요약·파일 hash·레코드 내용과 최종 보고서 artifact를 원격으로 재검증했고 Slack 완료 알림을 전송했습니다. [게시 영수증](publication.json), [원본 원격 검증](source_readback.json), [보고서 원격 검증](publication_readback.json)을 로컬에 보존합니다. 원본 가중치·checkpoint는 업로드하지 않았습니다.

실행 수치 경로와 도구는 `codex/rainbow-validation`의 `08f07ae`를 사용했습니다. 로컬 실행 폴더는 `results/rainbow_validation_20261008/`, 초기 상태는 `results/rainbow_initial_20261008/initial_state.pt`, 게시 증거는 `results/rainbow_validation_publication_20261008/`에 있습니다. 기존9방법 보고서는 유지하고 PROL을 다음 구현 대상으로 준비합니다.
