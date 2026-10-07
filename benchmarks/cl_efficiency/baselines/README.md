# RanPAC·L2P·DualPrompt 첫 태스크 효율성 비교

2026-10-07 사용자 승인으로 기존 세 방법의 측정9회·별도 FLOPs3회를 완료했습니다. EASE Base64의 기존 측정3회·진단1회를 재사용한 [4방법 비교 보고서](efficiency_pilot_report.md), [JSON](efficiency_pilot_results.json), [CSV](efficiency_pilot_results.csv)에 결과를 보존합니다. 첫 태스크·1epoch 비교이며 전체 CL이나 논문 성능 재현을 뜻하지 않습니다.

- `seq-cifar100-224`, 같은 첫 10클래스·train5000/test1000·class order·EASE preprocessing.
- 동일 ViT-B/16 AugReg IN21K→IN1K NPZ 파일·SHA256. query와 adaptation backbone도 실제 CPU tensor 150개로 일치를 확인했습니다.
- GPU1·1 epoch·batch64·workers4·seed1993. 각 방법 측정3회와 별도 부분 FLOPs1회; 기존 EASE 측정은 재사용합니다.
- 고유 learning rate·prompt/RP 설정 유지. L2P·DualPrompt input lr0.03은 native batch scaling 후0.0075입니다. RanPAC rp_size10000·17 ridge 후보 및 최종 CPU solve를 유지합니다.
- FP32 parameters, no autocast, CUDA matmul TF32 off, cuDNN TF32 allowed, deterministic=True, benchmark=False.
- 완전한79updates·5000train·1000eval와 frozen/finite/data/prediction/final-state checks9개가 필수입니다. 정확도20% 초과는 심한 붕괴 탐지용 하한이며 논문 성능 재현 판정이 아닙니다. 낮은 정확도는 구조 검사 통과만으로 설명하지 않고 조사합니다.
- 실행 동안 source를 고정하고 local JSON/CSV·예측 NPZ·telemetry·phase·W&B·Slack을 기록합니다.

[승인 계획](validation_plan.json), [실제 checkpoint 로딩 감사](environment/checkpoint_loading_audit.json), [검증 명령](checks.md)을 참고합니다. 기본 명령은 scope를 출력하며 실행하지 않습니다.

[독립 감사](environment/independent_pilot_audit.json)와 [W&B 서버 확인](environment/wandb_pilot_readback.json)에서12worker의 예측·집계·artifact 일치를 확인했고, 모든 suite의 Slack 요청도 `sent`였습니다. 실행 source는 `20ec323`이며 계획 파일의 준비 상태는 승인 시점의 snapshot입니다. 초기 parser 오류의 [제외 기록](environment/excluded_attempts.json)과 [W&B 제외 표시 확인](environment/wandb_exclusions_readback.json)은 별도로 보존했습니다. 직접 실행할 때도 `--pretrained 0`과 명시적인 로컬 checkpoint를 함께 사용해야 공급 backbone의 기본 가중치 로딩을 피합니다.

```bash
uv run --no-sync --offline python scripts/run_baseline_efficiency.py
```

승인된 실행 명령은 `uv run --no-sync python scripts/run_baseline_efficiency.py --execute`입니다. Tiny/Small·HiDe-Prompt·추가 seeds·전체 CL·BudgetCL/aL-SAR·원격 push는 이번 범위에 포함하지 않습니다.

시간과 에너지는 마지막 W&B artifact 업로드/finish를 제외한 Recorder 관측 구간이며 import·초기화·학습·task-end·평가·감사를 포함합니다. FLOPs 진단 비용은 측정 통계에서 제외합니다. GPU 에너지는 CPU/RAM/전체 머신 에너지가 아닙니다. RanPAC의 CPU linalg solve는 부분 FLOPs에서 빠질 수 있으므로 task-end 시간과 CPU G/Q 크기도 봅니다. 메모리 등록/auxiliary 항목은 중복될 수 있어 합산하지 않습니다. CPU 시간은 resource self+종료된 child 합계이며 살아 있는 worker는 빠질 수 있습니다. 같은 seed의3회로 seed간 성능 변동이나 망각을 평가하지 않습니다.
