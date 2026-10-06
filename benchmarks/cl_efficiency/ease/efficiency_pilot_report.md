# EASE 첫 태스크 효율성 pilot — 2026-10-07

EASE ViT-B/16·adapter64의 측정3회와 별도 FLOPs1회를 완료했습니다. 각 실행은 동일 사전학습 가중치·seed1993에서 새 프로세스로 시작했고, 여덟 구조 검사·원격 결과 저장·Slack 요청 상태를 모두 확인했습니다. 기존 첫2태스크 구현 검증은 [Native 보고서](native_validation_report.md)에 별도로 보존합니다.

## 측정 결과

| 지표 | 중앙값 | 최솟값~최댓값 |
|---|---:|---:|
| 전체 관측 시간 (s) | 29.7629 | 29.5921~30.0680 |
| 학습 epoch 시간 (s) | 13.3425 | 13.3380~13.3428 |
| Prototype·태스크 종료 (s) | 6.5408 | 6.5356~6.5482 |
| 평가 시간 (s) | 1.4635 | 1.4631~1.4642 |
| CUDA peak allocated (MiB) | 5760.5054 | 5760.5054~5760.5054 |
| CUDA peak reserved (MiB) | 6036.0000 | 6036.0000~6036.0000 |
| GPU device memory peak (MiB) | 6527.0000 | 6527.0000~6527.0000 |
| 프로세스 트리 RSS 합계 peak (MiB) | 13355.5000 | 13317.0234~13361.3633 |
| GPU 보드 에너지 (Wh) | 2.4432 | 2.4378~2.4724 |
| 학습 처리량 (samples/s) | 374.7414 | 374.7333~374.8695 |

| 실행 | Class-IL 정확도 | 전체 관측 시간(s) | GPU 에너지(Wh) | 원격 결과 |
|---|---:|---:|---:|---|
| 측정1 | 96.50% (965/1000) | 30.0680 | 2.443184 | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/zrctmk8h) |
| 측정2 | 96.50% (965/1000) | 29.5921 | 2.437814 | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/fzn84ciy) |
| 측정3 | 96.50% (965/1000) | 29.7629 | 2.472424 | [W&B](https://wandb.ai/laymond1/cl-efficiency/runs/qitfg8hb) |

세 측정 모두 정확도96.50%입니다. 1epoch 결과이므로20epoch 공식/native 기준선과의1.5~3.0%p 판정을 적용하지 않습니다. 비정상 붕괴·NaN·데이터/구조 오류를 점검했고, 반복 결과도 일치했습니다. 같은 seed의3회로 seed간 성능 변동을 추정하지 않습니다.

## 실행 조건과 연산 범위

- 물리 GPU1 RTX4090; 공유 데이터/cache 재사용; 신규 데이터나 가중치 다운로드 없음.
- `seq-cifar100-224`, 첫10클래스0~9, train5000/test1000, validation·noise·부분 라벨 없음.
- 1epoch, batch64, workers4, seed1993, SGD lr0.025/momentum0.9/weight decay0.0005/cosine minlr0.
- 가중치 `vit_base_patch16_224.augreg_in21k_ft_in1k`, SHA256 `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`.
- FP32 파라미터·autocast 없음, CUDA matmul TF32 off, cuDNN TF32 허용·deterministic=True·benchmark=False. 전체 연산을 strict FP32라고 표기하지 않습니다.
- 회당79updates·5000학습 presentations·5000prototype adapter-image pairs·1000평가 presentations. Adapter1개, prototype `[10,768]`.

별도 [FLOPs 진단](https://wandb.ai/laymond1/cl-efficiency/runs/tuk4smi4)의 지원 연산 합계는 **562,696,180,049,887 FLOPs ≈ 562.70 TFLOP**입니다. 저장된 operator는 `mul`, `conv2d`, `add`, `addmm`, `bmm`, `mm`입니다. 초기화·학습 forward/backward/optimizer·prototype·평가·감사 경로의 부분 집계이며 phase별 FLOPs나 학습만의 FLOPs로 해석하지 않습니다. 지원하지 않는 연산은 빠질 수 있고 backward에 임의의 배수를 곱하지 않았습니다. Profiler 실행의 시간·메모리·에너지는 위3회 집계에 포함하지 않았습니다.

## 측정 해석

전체 시간/에너지는 Recorder 구간으로 import·초기화·native W&B 시작·학습·prototype·평가·감사·로컬 기록을 포함하고, 마지막 artifact 업로드/finish는 제외합니다. 학습13.34초 외에 prototype6.54초와 평가1.46초가 필요하므로 학습 시간만으로 EASE 비용을 설명하지 않습니다.

GPU 보드 에너지의 coverage는 **99.8239~99.8274%**이며 센서 오류는 없습니다. 전체 머신·CPU/RAM 에너지를 측정한 것은 아닙니다. RSS는 프로세스별 합계로 공유 페이지가 중복 집계될 수 있습니다. 짧은 phase의1초 전력 샘플 적분은 세밀한 단독 전력 측정으로 해석하지 않습니다. 전체값과 phase의 중앙값은 서로 다른 실행에서 선택될 수 있으므로 phase 중앙값을 합산해 전체 중앙값으로 대체하지 않습니다.

모델 파라미터86995969개/347983876bytes, 학습 파라미터1197313개, prototype등 등록 buffer30741bytes, momentum state4789252bytes입니다. Adapter의 적은 학습 파라미터와 frozen ViT의 activation/backward 비용은 별개입니다. 원본 이미지 replay buffer는 없습니다. 이번batch64 peak는 앞선batch48 검증의 peak와 직접 동등 조건으로 비교하지 않습니다.

## 저장과 검증

[JSON 결과](efficiency_pilot_results.json)·[측정3회 CSV](efficiency_pilot_results.csv), [실제 원시 로그](/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.worktrees/ease/benchmarks/cl_efficiency/ease/runs/ease-efficiency-pilot-20261006T182523Z)를 보존합니다. 원시 로그는 EASE worktree의 ignored run폴더에 남으며 Git 병합으로 복사되지 않습니다. 모든 worktree는 공유 데이터/cache를 재사용합니다.

[W&B 서버 확인](environment/wandb_efficiency_readback.json)에서 네 run `finished`, artifacts `COMMITTED`와 metrics/spec/NPZ/telemetry/phase파일의 로컬 일치를 확인했습니다. Slack도 모든 suite 알림이 `sent`입니다. [독립 감사](environment/efficiency_independent_audit.json)에서 예측 정답 수·raw 데이터·source/가중치 hash·완료 카운터·에너지 적분·측정만의 집계를 확인했습니다. Full checkpoints를 새로 저장하지 않았으므로 이번 pilot의 tensor값 근거는 실행 중 frozen state hash·finite 검사를 따릅니다.

실행 source commit은 `49897a3`입니다. 실행 전 focused suite295개와 실제 네 실행이 통과했습니다. 구체적인 명령과 로컬 병합 기록은 [검증 기록](checks.md)에 남깁니다. Tiny·Small·다른 알고리즘·추가 seeds·전체 CL 재현·BudgetCL/aL-SAR 비교·원격 push는 이번 범위에 포함하지 않았습니다.

[W&B 집계 보고서](https://wandb.ai/laymond1/cl-efficiency/runs/q61iouic)에는 세 측정의 중앙값·범위와 별도 FLOPs 값을 저장합니다. 이는 기존 결과를 업로드하는 report-only 기록이며 추가 실험이 아닙니다. 검증된 코드 `49897a3`은 `codex/cl-foundation`에 로컬 fast-forward 병합했습니다.
