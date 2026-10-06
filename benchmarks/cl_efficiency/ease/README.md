# EASE 구현과 실행 준비

2026-10-07, 독립 브랜치 `codex/ease`의 구현 준비 단계입니다. **실제 EASE 학습·ViT 이미지 추론·효율성 측정은 수행하지 않았습니다.** 공통 기반 `codex/cl-foundation`에 병합하지 않았으며, 첫 2개 태스크의 구현 검증이 통과해야 병합합니다.

## 확정한 조건

- 기준 backbone: ViT-B/16. Tiny·Small도 크기별 feature 차원 192·384·768을 따라가도록 지원합니다.
- 세 크기 모두 AugReg **IN21K → IN1K fine-tuned** 가중치 계열을 사용합니다. 정확한 파일과 SHA256은 다운로드/실행 전에 해결할 항목으로 남겼습니다.
- 기본 adapter bottleneck은 모두 64. Tiny16·Small32·Base64 비교는 별도 설정으로 준비했습니다.
- 데이터셋은 모두 `seq-cifar100-224`, 10 classes/task, 순서 0..99, seed1993입니다.
- 구현 검증: 첫 2 tasks, 20 epochs/task, batch48, SGD lr0.025/momentum0.9/weight decay0.0005/cosine.
- 효율성 pilot: 첫 1 task, 1 epoch, batch64, workers4, measurement와 FLOPs diagnostic pass 분리.
- 물리 GPU1, entity/project `laymond1/cl-efficiency`, 기존 공유 데이터·캐시와 비공개 Slack 설정을 재사용합니다.

[validation_plan.json](validation_plan.json)에 승인 상태, 다섯 설정, 동일 조건 reference 정책을 기록했습니다. `training_authorized=false`이며 실행 전에 별도 학습 승인이 필요합니다. [Tiny·Small 검토](tiny_small_review.md)에 크기별 예상 adapter 저장량과 비교의 한계를 정리했습니다.

## 구현 범위

[모델](../../../models/ease.py)은 Mammoth의 `meta_begin_task`/`meta_observe`/`meta_end_task`를 그대로 사용합니다. 매 `begin_task`에서 새 adapter와 현재 클래스용 cosine proxy head만 학습하고, 이전 adapter와 pretrained backbone은 고정합니다. `end_task`에서는 현재 태스크의 모든 training sample을 deterministic test transform으로 방문하여 raw feature mean을 구한 뒤, 과거 클래스의 새 feature space를 semantic similarity로 보완합니다. 다음 adapter는 다음 `begin_task`에 생성하여 평가 시 불필요한 미래 adapter를 포함하지 않습니다.

[Backbone](../../../backbone/ease_vit.py)의 adapter는 `ModuleList`, [분류기](../../../models/ease_utils/classifier.py)의 prototype은 buffer로 등록합니다. 따라서 device 이동·상태 저장·공통 효율성 도구의 파라미터/buffer 집계에 포함됩니다. Frozen FFN도 adapter로 전달할 gradient 경로는 유지합니다. 가중치를 선택하지 않은 random backbone, 부분 라벨, task-end 이전 prototype 평가와 맞지 않는 per-epoch evaluation은 거부합니다.

초기 지원 데이터 통합은 `seq-cifar100-224`의 [ease 설정](../../../datasets/configs/seq-cifar100-224/ease.yaml)입니다. 공식 CIFAR preprocessing을 따릅니다: train RandomResizedCrop224/flip, test Resize256 bicubic/CenterCrop224, normalization은 identity. timm checkpoint의 기본 preprocessing을 자동 적용하지 않으므로 공식 코드와 포팅 코드가 같은 preprocessing을 사용하도록 검증해야 합니다. `use_old_data`, `use_init_ptm`, 기타 공식 ablation은 이번 표준 replay-free 경로에 포함하지 않았습니다.

참조는 [저자 코드](https://github.com/sun-hailong/CVPR24-Ease/tree/6d67508ed8b7c4d23350ed4eb28c123914b40ae3), commit `6d67508ed8b7c4d23350ed4eb28c123914b40ae3`입니다. 공식 CIFAR config의 5+5 split과 class shuffle을 그대로 실행한 성능은 이번 10-class split의 acceptance 기준으로 사용하지 않습니다. reference도 동일 checkpoint·10-class split·class order·샘플·transform으로 맞춰야 합니다. 이번 검증은 논문 성능 재현이 아닙니다.

## 실행 없이 준비하는 도구

EASE worktree에서 다음 명령은 **계획 생성만 수행**합니다. 모델 import, 학습, dataset/weight 다운로드, W&B 업로드, Slack 알림을 하지 않습니다. 공통 runner의 준비 과정은 GPU 센서를 읽습니다.

```bash
uv run --no-sync python scripts/benchmark_cl.py
uv run --no-sync python scripts/prepare_ease.py
```

첫 명령의 현재 inventory는 71개이며 실행 준비 66개·보류 5개입니다. 두 번째 명령은 `benchmarks/cl_efficiency/prepared/ease/`에 다섯 variant의 manifest·plan·readiness와 `runner_commands.json`을 생성합니다. 이 도구에는 실행 옵션이 없습니다. 실제 효율성 측정은 검토 후 공통 `benchmark_cl.py`에 명시적으로 `--execute`를 추가할 때만 시작됩니다. 이 단계에서는 추가하지 않았습니다.

[commands.json](commands.json)의 native 명령은 모델 설정 확인용이며, 직접 실행하면 학습을 시작합니다. W&B 대상과 local Mammoth 로그 경로를 명시했지만, native 명령 자체에는 Slack 알림이 없습니다. telemetry·Slack·W&B artifact가 필요한 효율성 측정에는 위에서 생성한 **공통 runner 명령**을 사용해야 합니다. 공통 runner는 기존 local JSON/CSV, phase 시간, CPU/GPU memory, GPU board power/energy와 coverage, 등록 parameter/buffer bytes, 정확도 및 별도 부분 FLOPs를 기록하고 W&B·Slack 연결을 사용합니다. 향후 이 측정 과정에는 task-end prototype pass도 포함됩니다. GPU board energy는 전체 머신 에너지가 아니며, 첫 task에서 adapter 누적 비용이나 forgetting을 검증할 수 없습니다.

## 학습 없는 검증과 제한

검증은 parser/config 우선순위, 세 크기의 가중치 tag, adapter 등록·동결·상태 reload, synthetic feature의 raw mean/semantic completion/reweight, native wrapper 보존 및 dry run의 실행/알림 차단을 대상으로 합니다. Neural image forward/backward나 optimizer step은 실행하지 않았습니다. 기존 `test_import`, `test_module_register`, `test_model_config`, `test_dynamic_params`의 상당수도 내부에서 실제 학습을 수행하므로 사용자의 보류 요청에 따라 전체 playbook을 실행하지 않았습니다. 같은 계약의 학습 없는 focused tests와 native `--help` smoke를 수행했습니다.

현재 환경은 lock 기반 Python3.12.15, timm0.9.8입니다. 보통의 EASE pytest/helper import는 native dependency `pyarrow.lib` 로딩 중 SIGSEGV(139)를 보였습니다. `pyarrow` 선행 import를 사용한 검증은 통과했습니다. 기존 dataset discovery가 import 중 CLIP 가중치를 로딩하므로 최종 학습 없는 검사는 `open_clip.create_model_and_transforms`를 mock 처리했습니다. 최초 확인 중 기존 cache의 CLIP 생성/로딩 side effect는 있었지만, forward나 학습은 없었습니다. 실험 시작 전에 기본 worker import 경로의 충돌을 해결하고 재검증해야 합니다. 이 bootstrap/mock을 실제 실험의 통과 근거로 사용하지 않습니다.

검증 명령과 결과는 [checks.md](checks.md)에 기록합니다. W&B/Slack 연결은 10월 6일 공통 기반에서 합성 메시지로 확인한 상태를 재사용했습니다. 이번 EASE 준비에서는 새 외부 run이나 알림을 보내지 않았습니다.

## 다음 단계

1. 학습 시작 승인 후 Base checkpoint의 파일/hash를 확정하고 official/native가 같은 파일을 로딩하도록 준비합니다. official 환경 버전·샘플 mapping도 고정합니다.
2. 실제 image forward/backward 경로를 확인하고 official reference의 변동을 조사합니다. **포팅 결과를 acceptance 평가하기 전에 허용 accuracy 차이를 확정**합니다. 지금 임의의 pp threshold를 만들지 않았습니다.
3. Base의 첫 2개 태스크를 동일 조건으로 검증합니다. 전체/old/new class 정확도와 task별 prototype shape·adapter count를 확인하며, 비정상적으로 낮은 정확도는 구현 실패로 취급하여 병합을 보류합니다. 필요하면 3 seeds로 확대합니다.
4. 통과한 뒤 첫 태스크 효율성 pilot을 실행하고, Tiny·Small 및 비율 정렬 비교를 순차 확대합니다. 논문 재현과 BudgetCL/aL-SAR 예산 비교는 후속 단계입니다.
