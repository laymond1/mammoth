# EASE 구현·검증과 효율성 측정

2026-10-07, 독립 브랜치 `codex/ease`에서 EASE Base64·seed1993의 첫 2개 태스크 구현 검증과 첫 태스크 효율성 pilot을 완료했습니다. Native 구현 검증 정확도는 Task 1 99.40%, Task 2 전체 97.30%·과거 96.60%·신규 98.00%이며, 네 정확도 지표와 일곱 구조 검사가 모두 통과했습니다. [Native 검증 보고서](native_validation_report.md)와 [JSON 결과](native_validation_results.json)에 기록합니다. [공식 기준선·smoke 보고서](execution_report.md)는 당시 측정 이력으로 보존합니다. 이후 측정 3회·별도 부분 FLOPs 진단 1회의 W&B 저장·Slack 알림·독립 감사가 통과했고, 실행 source `49897a3`을 로컬 `codex/cl-foundation`에 fast-forward 병합했습니다. Tiny·Small 학습과 원격 push는 수행하지 않았습니다.

1-epoch 효율성 측정 3회의 정확도는 모두 **96.50%**입니다. 전체 관측 시간 중앙값 **29.7629초**, CUDA peak allocated **5760.5054 MiB**, GPU 보드 에너지 중앙값 **2.4432 Wh**, 별도 지원 연산의 부분 FLOPs 합계 **562.70 TFLOP**를 얻었습니다. [효율성 보고서](efficiency_pilot_report.md), [JSON 결과](efficiency_pilot_results.json), [측정 CSV](efficiency_pilot_results.csv)에 측정 범위와 제한을 함께 기록합니다. Profiler pass의 시간·메모리·에너지는 측정 3회 집계에 포함하지 않습니다.

## 확정한 조건

- 기준 backbone: ViT-B/16. Tiny·Small도 크기별 feature 차원 192·384·768을 따라가도록 지원합니다.
- 세 크기 모두 AugReg **IN21K → IN1K fine-tuned** 가중치 계열을 사용합니다. Base 파일과 SHA256은 [실행 계획](validation_plan.json)에 고정했습니다. Tiny·Small의 파일/hash는 각 실행 전에 확정합니다.
- 기본 adapter bottleneck은 모두 64. Tiny16·Small32·Base64 비교는 별도 설정으로 준비했습니다.
- 데이터셋은 모두 `seq-cifar100-224`, 10 classes/task, 순서 0..99, seed1993입니다.
- 구현 검증: 첫 2 tasks, 20 epochs/task, batch48, SGD lr0.025/momentum0.9/weight decay0.0005/cosine.
- 효율성 pilot: 첫 1 task, 1 epoch, batch64, workers4, measurement와 FLOPs diagnostic pass 분리.
- 물리 GPU1, entity/project `laymond1/cl-efficiency`, 기존 공유 데이터·캐시와 비공개 Slack 설정을 재사용합니다.

[validation_plan.json](validation_plan.json)에 단계별 승인 상태, 다섯 설정, 동일 조건 reference 정책을 기록합니다. 승인한 Base64의 두 태스크 acceptance 평가와 효율성 pilot은 통과했습니다. 다섯 variant 중 Base64만 실행했고 네 Tiny·Small variant는 미실행입니다. [Tiny·Small 검토](tiny_small_review.md)에 크기별 예상 adapter 저장량과 비교의 한계를 정리했습니다.

## 확정한 정확도 판정 기준 (2026-10-07)

사용자 결정으로 네 지표 각각의 공식 기준선 대비 하락을 판정합니다. **1.5%p 이내는 통과, 1.5%p 초과~3.0%p 이내는 추가 점검, 3.0%p 초과는 실패**입니다. 개선된 정확도는 통과이며, 반올림한 표시값 대신 정답 수/전체 샘플 수로 비교합니다.

| 지표 | 공식 기준선 | 통과 하한 (포함) | 추가 점검 하한 (포함) |
|---|---:|---:|---:|
| Task 1 전체 | 99.60% | 98.10% | 96.60% |
| Task 2 전체 | 97.20% | 95.70% | 94.20% |
| Task 2 과거 클래스 | 96.60% | 95.10% | 93.60% |
| Task 2 신규 클래스 | 97.80% | 96.30% | 94.80% |

추가 점검 구간은 해당 추가 점검 하한 이상, 통과 하한 미만입니다. 어느 지표든 실패하면 전체 실패, 실패 없이 추가 점검 지표가 있으면 전체 추가 점검입니다. 네 지표와 구조 검사가 모두 통과해야 전체 통과입니다. 추가 점검 결과는 원인을 조사하고 결론을 기록하기 전까지 acceptance 통과나 병합·효율성 pilot 진행 조건을 충족하지 않습니다.

NaN/비정상 gradient, 데이터 불일치, frozen backbone 또는 이전 adapter 변경, prototype shape·adapter count 오류는 정확도와 별개로 실패입니다. 이 폭은 사용자와 정한 운영 기준이며 단일 reference seed에서 통계적으로 추정한 허용오차가 아닙니다. 기준 확정 자체로 새 학습을 시작하지 않습니다.

위 정확도 판정은 동일 조건의 **20-epoch·2-task 구현 검증**에 적용합니다. **1-epoch·1-task 효율성 pilot에는 적용하지 않습니다.** Pilot은 정확도를 기록하면서 비정상 붕괴·NaN·데이터/구조 오류와 완전한 실행 여부를 점검했고, 여덟 구조 검사가 통과했습니다. 같은 seed의 측정 3회는 seed 간 성능 변동의 추정치가 아닙니다.

## 구현 범위

[모델](../../../models/ease.py)은 Mammoth의 `meta_begin_task`/`meta_observe`/`meta_end_task`를 그대로 사용합니다. 매 `begin_task`에서 새 adapter와 현재 클래스용 cosine proxy head만 학습하고, 이전 adapter와 pretrained backbone은 고정합니다. `end_task`에서는 현재 태스크의 모든 training sample을 deterministic test transform으로 방문하여 raw feature mean을 구한 뒤, 과거 클래스의 새 feature space를 semantic similarity로 보완합니다. 다음 adapter는 다음 `begin_task`에 생성하여 평가 시 불필요한 미래 adapter를 포함하지 않습니다.

[Backbone](../../../backbone/ease_vit.py)의 adapter는 `ModuleList`, [분류기](../../../models/ease_utils/classifier.py)의 prototype은 buffer로 등록합니다. 따라서 device 이동·상태 저장·공통 효율성 도구의 파라미터/buffer 집계에 포함됩니다. Frozen FFN도 adapter로 전달할 gradient 경로는 유지합니다. 가중치를 선택하지 않은 random backbone, 부분 라벨, task-end 이전 prototype 평가와 맞지 않는 per-epoch evaluation은 거부합니다.

초기 지원 데이터 통합은 `seq-cifar100-224`의 [ease 설정](../../../datasets/configs/seq-cifar100-224/ease.yaml)입니다. 공식 CIFAR preprocessing을 따릅니다: train RandomResizedCrop224/flip, test Resize256 bicubic/CenterCrop224, normalization은 identity. timm checkpoint의 기본 preprocessing을 자동 적용하지 않으므로 공식 코드와 포팅 코드가 같은 preprocessing을 사용하도록 검증해야 합니다. `use_old_data`, `use_init_ptm`, 기타 공식 ablation은 이번 표준 replay-free 경로에 포함하지 않았습니다.

참조는 [저자 코드](https://github.com/sun-hailong/CVPR24-Ease/tree/6d67508ed8b7c4d23350ed4eb28c123914b40ae3), commit `6d67508ed8b7c4d23350ed4eb28c123914b40ae3`입니다. 공식 CIFAR config의 5+5 split과 class shuffle을 그대로 실행한 성능은 이번 10-class split의 acceptance 기준으로 사용하지 않습니다. 이번 reference는 동일 checkpoint·10-class split·class order·샘플·transform으로 맞춰 실행했습니다. 이번 기준선은 논문 성능 재현이 아닙니다.

## 실행 없이 준비하는 도구

EASE를 반영한 worktree에서 다음 명령은 **계획 생성만 수행**합니다. 모델 import, 학습, dataset/weight 다운로드, W&B 업로드, Slack 알림을 하지 않습니다. 공통 runner의 준비 과정은 GPU 센서를 읽습니다.

```bash
uv run --no-sync python scripts/benchmark_cl.py
uv run --no-sync python scripts/prepare_ease.py
```

첫 명령의 현재 inventory는 71개이며 실행 준비 66개·보류 5개입니다. 두 번째 명령은 `benchmarks/cl_efficiency/prepared/ease/`에 다섯 variant의 manifest·plan·readiness와 `runner_commands.json`을 생성합니다. 이 도구에는 실행 옵션이 없습니다. 기존 측정 결과가 있어도 새로 생성한 계획의 `experiment_executed=false`는 계획 생성에서 실험을 시작하지 않았다는 뜻입니다. 실제 완료 이력은 [validation_plan.json](validation_plan.json)과 결과 파일에 따로 기록합니다. 승인한 Base64 pilot은 [검증 기록](checks.md)의 실행 명령으로 완료했고, 새 실험은 실행 범위를 별도로 정한 뒤에만 시작합니다.

[commands.json](commands.json)의 native 명령은 모델 설정 확인용이며, 직접 실행하면 학습을 시작합니다. W&B 대상과 local Mammoth 로그 경로를 명시했지만, native 명령 자체에는 Slack 알림이 없습니다. telemetry·Slack·W&B artifact가 필요한 효율성 측정에는 **공통 runner**를 사용합니다. 승인한 Base64 반복 측정은 `run_ease_efficiency.py`가 공통 runner를 호출했습니다. Local JSON/CSV, phase 시간, CPU/GPU memory, GPU board power/energy와 coverage, 등록 parameter/buffer bytes, 정확도와 별도 부분 FLOPs를 기록했고 task-end prototype pass도 포함했습니다. GPU board energy는 전체 머신 에너지가 아니며, 첫 task에서 adapter 누적 비용이나 forgetting을 검증할 수 없습니다.

## 학습 없는 검증과 제한

학습 없는 검증은 parser/config 우선순위, 세 크기의 가중치 tag, adapter 등록·동결·상태 reload, synthetic feature의 raw mean/semantic completion/reweight, native wrapper 보존 및 dry run의 실행/알림 차단을 대상으로 합니다. 별도로 두 미니배치의 실제 image forward/backward·optimizer step을 점검한 뒤 Base64의 첫 2개 태스크를 20 epochs/task로 학습·평가했고 acceptance를 통과했습니다. 기존 `test_import`, `test_module_register`, `test_model_config`, `test_dynamic_params`의 상당수도 내부에서 실제 학습을 수행하므로 준비 단계의 전체 playbook은 사용자의 보류 요청에 따라 실행하지 않았습니다. 같은 계약의 학습 없는 focused tests와 native `--help` smoke를 수행했습니다.

현재 환경은 lock 기반 Python3.12.15, timm0.9.8입니다. 초기 import 충돌은 PyArrow의 vendored jemalloc background thread에서 발생했습니다. `JE_ARROW_MALLOC_CONF=background_thread:false`를 native import 전에 설정해 해결했고, 환경 안정화 단계의 197개 focused tests와 native help는 선행 PyArrow import나 OpenCLIP mock 없이 통과했습니다. 진단은 [환경 기록](environment/native_import_diagnosis.json)에 보존합니다. 기존 dataset discovery의 cached CLIP 로딩은 남아 있지만 새 가중치를 다운로드하지 않았습니다.

검증 명령과 결과는 [checks.md](checks.md)에 기록합니다. 구현 검증과 효율성 pilot 모두 W&B artifact 저장과 Slack 알림을 사용했고, 서버에서 run 종료와 artifact의 COMMITTED 상태를 다시 조회해 확인했습니다. 20-epoch·2-task 구현 검증의 자원 관측은 공통 1-epoch 효율성 결과와 구분합니다. Pilot의 GPU board energy는 Recorder 관측 구간의 GPU 에너지이며 전체 머신 에너지는 아닙니다. 원시 로그는 EASE worktree의 ignored `runs/`에 남아 Git 병합으로 복사되지 않으며 [효율성 보고서](efficiency_pilot_report.md)에서 실제 위치로 연결합니다.

## 다음 단계

1. Tiny·Small 비교를 진행할지, 기존·추가 CL 방법의 공통 효율성 pilot을 우선할지 후속 실행 범위를 정합니다. 이번에는 Base64만 완료했습니다.
2. Tiny·Small을 실행한다면 checkpoint 파일/hash를 고정하고 구현 검증과 공통64·비율 정렬 비교의 순서를 정합니다. 현재 계획 생성은 자동으로 새 학습을 시작하지 않습니다.
3. 추가 seeds·전체 CL 논문 성능 재현·BudgetCL/aL-SAR 예산 비교는 별도 후속 단계입니다. 단일 seed1993의 반복 측정으로 seed 간 변동이나 전체 CL 성능을 추정하지 않습니다. 원격 push는 승인하거나 수행하지 않았습니다.
