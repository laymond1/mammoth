# 공통 첫 태스크 효율성 pilot 부분 결과

[보고서](RESULTS.md)·[JSON](results.json)·[CSV](results.csv)·[PNG](efficiency_comparison.png)·[PDF](efficiency_comparison.pdf)를 보관합니다. [W&B 보고서](https://wandb.ai/laymond1/cl-efficiency/runs/8hyym6hc)에 같은 결과를 저장했습니다. 실행 원본은 `results/peft_pilot/base-20261008/`, SD 원형 대조는 `results/peft_pilot/sd_author_common_ease_20261008/`에 있습니다. [고정 계획](../peft_pilot/plan.json)은 계획20개이며 실제 승인 기록16개·격리1개·보류3개와 구별합니다.

신규 EWC-LoRA·InfLoRA·CL-LoRA·RanPAC의 측정3회와 별도 FLOPs1회가 모두 통과했습니다. EASE·L2P·DualPrompt·HiDe-Prompt의 기존3회 기록을 실제 원본 데이터·전처리·가중치·정밀도 기준으로 대조해 재사용했습니다. SD10% 관측과 저자 진단은 비교 평균에 넣지 않습니다.

SD 저자 코드도 동일 초기값·공통 전처리·LR0.008·batch64·1epoch에서10%를 기록했습니다. 학습 첫 입력/라벨/그래디언트,79회 업데이트, 최종 전체 텐서와1000개 예측이 native와 일치했습니다. 기존 원형 전처리·20epoch 검증이 통과했다는 사실과 이번 공통1epoch 레시피의 실패를 각 조건에 맞게 보존합니다. 전처리·학습률·학습 길이 중 원인을 분리하는 검증은 아직 수행하지 않았습니다.

다음 결정은 SD의 안정성 진단 범위입니다. **공통 데이터·전처리·1epoch를 유지하고 학습률만 낮추는 소규모 진단을 먼저 진행**하는 것을 추천합니다. 후보와 실행 횟수를 확정한 뒤 독립 저자 코드와 native가 안정적인 정확도 및 일치를 함께 보이는지 확인하고, 통과 설정에서 측정3회·FLOPs1회를 새로 수행합니다. 실패한 측정은 새 평균에서 제외하되 원본은 보존합니다. 사용자 승인 전에는 새 학습률 진단·SD 재측정·둘째 태스크 비용 측정·Small/Tiny 학습·BudgetCL/aL-SAR controller를 실행하지 않습니다.

벽시간은 worker 프로세스에서 관측한 초기화·학습·태스크별 통계/평가·감사 및 해당 worker의 checkpoint 비용을 포함하며, 마지막 W&B artifact 업로드는 제외합니다. 학습 시간은 별도로 표시합니다. 메모리는 CUDA allocated/reserved, NVML 장치 사용량 및 프로세스 트리 RSS를 원본 JSON에 기록합니다. 중첩 메모리/phase 값은 합산하지 않습니다. GPU 보드 에너지는 전체 머신 에너지가 아니며 FLOPs는 지원 연산의 부분 추정입니다. 첫 태스크·동일 seed3회의 결과는 전체 CL 성능이나 seed 불확실성·동일 계산 예산 우열을 추정하지 않습니다. 기존 기록의 측정 날짜·소스 차이와 EASE CPU thread 미기록도 유지합니다.

측정 source는 commit `04559dbdc8c7e68adb96da7871d9d31c2fc5365e` 및54개 파일 SHA로 고정했습니다. 부분 재개와 SD 원형 대조의 별도 source SHA는 보고서 JSON에 있습니다. 전체 보고서 gate를 완화하지 않고 별도 부분 보고서/게시 도구를 사용했습니다. [검증 명령과 결과](checks.json), [원격 source 검증](source_readback.json), [최종 게시 receipt](publication.json)를 보관합니다. 전체 체크포인트·비밀 설정은 W&B artifact와 Git에 넣지 않았습니다.
