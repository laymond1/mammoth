# HiDe-Prompt Small → Tiny: 검증 및 첫 태스크 효율성

완료: Small → Tiny 순서로 각각 독립 저자 코드 첫 2태스크, native 첫 2태스크, 첫 태스크 계측 3회와 별도 부분 FLOPs 1회를 실행했습니다. **EASE 저자 코드 전체 10태스크 실행은 생략**했습니다. 이 단계의 Small/Tiny 전체 10태스크 및 추가 seed 실험은 포함하지 않습니다.

[상세 비교 보고서](report/report.md) · [JSON](report/results.json) · [CSV](report/results.csv) · [PNG](report/efficiency_comparison.png) · [PDF](report/efficiency_comparison.pdf) · [W&B 최종 보고서](https://wandb.ai/laymond1/cl-efficiency/runs/jdjy4oll)

| 크기 | 첫 2태스크 누적 정확도 | 저자−native 최대 차이 | 구현 검증 |
|---|---:|---:|---|
| Small | 96.55% | 0.00%p | PASS |
| Tiny | 93.80% | 0.00%p | PASS |

네 정확도 지표, 전체 top1/TII 예측 배열, 첫 배치 입력·라벨·seed·gradient hash, 학습 파라미터와 클러스터 통계가 동일합니다. 각 검증은 140epoch·11,060 outer step·700,000 학습 이미지·3,000 평가 이미지이며, 분류기 정렬은 첫 태스크 0step, 둘째 태스크 두 branch 각각 300step입니다. 판정은 동일 크기의 저자 기준선에 대해 1.5%p 이내 통과, 1.5~3.0%p 추가 점검, 3.0%p 초과 실패이며 구조 오류도 통과를 차단합니다.

| 첫 태스크 pilot, 3회 평균 | Base (기존 측정) | Small | Tiny |
|---|---:|---:|---:|
| 정확도 | 96.60% | 95.40% | 89.40% |
| 벽시간 | 57.09s | 29.78s | 21.32s |
| CUDA allocated peak | 6.334GiB | 3.032GiB | 1.507GiB |
| 샘플링된 NVML device peak | 7.329GiB | 3.759GiB | 2.095GiB |
| process-tree RSS peak | 15.526GiB | 12.627GiB | 12.042GiB |
| CPU 누적 시간 | 91.50s | 47.71s | 37.84s |
| GPU 보드 에너지 | 4.992Wh | 1.879Wh | 0.854Wh |
| 별도 부분 FLOPs | 1126.989TF | 296.624TF | 81.595TF |

Pilot 조건: CIFAR100 첫 태스크, 공통 `ease` 전처리, GPU1 RTX4090, FP32, batch64, workers4, CPU threads2, seed1993, TII1+prompt1 및 각 5,000개 전체 통계 추출·1,000개 실제 routing 평가. 158 outer step·10,000 학습 이미지·두 통계 pass를 포함합니다. CA는 첫 태스크에 해당 없음입니다. 검증은 `hide_prompt` 전처리와 TII20+prompt50을 사용합니다. 각 목적의 결과를 따로 비교합니다.

Tiny는 이번 단기 pilot에서 비용과 함께 정확도도 크게 낮아졌습니다. 동일 크기 저자 코드 대조를 통과했으며, 단기 pilot 결과로 전체 CL 성능·망각이나 여러 seed의 불확실성을 판단하지 않습니다. Base는 기존 시점의 측정을 재사용했습니다. 세 크기는 AugReg IN21K→IN1K 계열을 사용하지만 사전학습 recipe가 다릅니다. GPU 에너지는 CPU/RAM을 제외한 보드 전력 적분이고 FLOPs는 지원 연산의 부분 추정입니다.

## 재현 및 저장 기록

- [불변 설정](validation_plan.json): prefix 길이5·layer0~4·TII MLP 폭 비율2, Small D384/H6, Tiny D192/H3. 모든 worktree가 공유 데이터 `/mnt/sdab1/share/data` 및 캐시 `/mnt/sdab1/userHome/wslim/.cache`를 재사용합니다.
- [실행 상태](execution_plan.json) · [실제 명령](environment/command_log.json) · [CPU/테스트 검증](environment/preflight.json) · [독립 감사](environment/independent_audit.json).
- 모델/실행 코드 동결 커밋: `e060ccf5c6320cbd3cf13ef8a43ca8cdac7a8dc7`. 독립 [저자 코드](https://github.com/thu-ml/HiDe-Prompt/tree/0ed433334ff2cfb206c5e2899651c3044ec85f92)는 수정 없이 원본 함수/모델에 크기별 geometry를 전달했습니다.
- [GPU 점유 감사](environment/gpu_occupancy_summary.json): 3,190개 표본, 외부 연산/센서 오류 0. 종료된 자체 PID의 driver metadata 잔류 표본 1개를 따로 기록합니다. 점유 표본은 GPU 독점 잠금이 아니며 Small 기준선 시작 직후부터 수집했습니다. 두 pilot은 모두 관측 구간 안에 있습니다.
- 최종 `uv run --no-sync --offline pytest` 집중 16파일: **440 PASS, 11.31초**. 위 검증 기록에 실제 전체 명령을 저장했습니다. 기존 generic import/config/backbone smoke는 기본 로컬 데이터·가중치 다운로드가 네트워크 제한으로 실패했으며, 공유 CIFAR100/NPZ의 실제 CPU 순전파와 승인된 native GPU 검증을 완료했습니다.
- 신규 실험 12run의 원격 파일 **132개** 및 최종 보고서 **6개**를 finished/COMMITTED/SHA readback으로 확인했습니다. [최종 저장 증거](remote_readbacks/report.json) · [완료 알림](remote_readbacks/completion_notification.json). Slack 진행·완료 알림 전송 완료.

원본 실행은 보존한 `.worktrees/hide-sizes/benchmarks/cl_efficiency/hide_sizes/runs/`에 있습니다. 초기화와 다운로드 readback은 같은 worktree의 `prepared/`에 보존하며 대용량 원본은 Git에서 제외했습니다. 알고리즘 브랜치 `codex/hide-size-expansion`의 결과를 `codex/cl-foundation`에 로컬 통합하며 remote push는 포함하지 않습니다.
