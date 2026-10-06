# EASE 실제 실행 보고서 — 2026-10-07

이 보고서는 공식 기준선과 두 미니배치 smoke를 측정한 당시의 이력입니다. 이후 완료한 Base64·seed1993 native 두 태스크 검증은 [최신 검증 보고서](native_validation_report.md)와 [JSON 결과](native_validation_results.json)를 참고합니다. 아래 공식 수치와 [기존 JSON](execution_results.json)은 당시 기록으로 보존합니다.

당시 승인한 단계인 환경 안정화, Base checkpoint 고정, 포팅 코드의 최소 forward/backward 점검, 동일 조건 공식 코드의 첫 2개 태스크 기준선을 완료했습니다. 브랜치는 `codex/ease`입니다. 이 보고서의 측정 시점에는 포팅 코드의 두 태스크 정확도 acceptance 평가를 아직 실행하지 않았습니다.

## 결과

| 공식 코드 평가 시점 | 관측 클래스 | 전체 top-1 | 과거 클래스 | 신규 클래스 | Prototype shape | Adapter 수 |
|---|---:|---:|---:|---:|---|---:|
| Task 1 완료 | 10 | 99.60% (996/1000) | — | 99.60% | 10 × 768 | 1 |
| Task 2 완료 | 20 | 97.20% (1944/2000) | 96.60% (966/1000) | 97.80% (978/1000) | 20 × 1536 | 2 |

포팅 코드에서는 실제 CIFAR100 이미지로 batch2의 두 optimizer step만 실행했습니다. Frozen ViT는 optimizer에서 제외되고 gradient가 없었으며, adapter up projection이 두 step 모두 변경되고 두 번째 step에는 down projection gradient도 발생했습니다. Loss는 2.299607 → 2.309933이며, 두 step의 loss 감소를 통과 조건으로 사용하지 않았습니다. 이 점검에서는 task-end prototype 생성과 테스트 정확도를 평가하지 않았습니다.

| 공식 기준선의 측정 항목 | 결과 |
|---|---:|
| Recorder 전체 관측 시간 | 572.10초 (약 9분 32초) |
| 학습 epoch 시간 합계 (40 epochs) | 536.72초 |
| Prototype 계산·보완 | 21.02초 |
| 평가 (1000 + 2000 samples) | 6.82초 |
| PyTorch CUDA peak allocated | 4462.12 MiB (4.36 GiB) |
| PyTorch CUDA peak reserved | 4748.00 MiB |
| 센서의 GPU1 device memory peak | 5237 MiB |
| 프로세스 트리 RSS 합계 peak | 8363.17 MiB |
| GPU1 board energy | 235,990.51 J = 65.55 Wh |
| 에너지 관측 coverage | 99.9931% |
| 학습 sample presentations / update steps | 200,000 / 4,200 |
| 평가 sample presentations | 3,000 |
| FLOPs | 이번 단계에서 측정하지 않음 |

Task 1/2의 학습 시간은 각각 268.41/268.31초, prototype pass는 7.01/14.01초, 평가는 1.49/5.33초입니다. Task 2의 prototype 공간과 평가 클래스·샘플 수도 증가하므로 이 두 시간만으로 비용이 adapter 수에 정비례한다고 결론 내리지 않습니다. Prototype pass의 총 15,000 adapter-image pairs는 위 학습/평가 샘플 카운터와 구분됩니다.

전체 시간·에너지는 import·초기화·로컬 결과 저장을 포함하는 Recorder 관측 구간이며, 최종 W&B artifact 업로드/finish는 제외합니다. Parent 전체 시간은 579.55초입니다. `incremental_train` phase는 `train_epoch`와 `prototype_completion`을 포함하므로 중복 합산하지 않습니다. GPU energy는 물리 GPU1 보드의 센서 적분값이며 CPU/RAM 또는 전체 머신 에너지가 아닙니다. RSS는 프로세스별 합계여서 공유 메모리 페이지가 중복 집계될 수 있습니다. 이번 20-epoch/2-task 실행은 추후 공통 1-epoch/1-task 효율성 pilot과 별도입니다.

공식 코드의 과거 adapter는 등록되지 않은 Python list에 저장됩니다. Task 2에서 등록 파라미터 집계 외에 과거 adapter 1,189,632개 파라미터를 별도로 기록하고, checkpoint에도 저장했습니다. 이를 포함한 공식 모델은 88,216,322 parameters / 352,865,288 parameter bytes입니다. Task 1 current adapter와 Task 2 old adapter의 48개 저장 텐서도 CPU에서 비교하여 값이 모두 동일함을 확인했습니다. 포팅 코드는 `ModuleList`로 등록해 누적 저장량 집계에 포함합니다. 결과 JSON의 공식 파라미터 집계를 포팅 코드의 값으로 대체하지 않습니다.

## 동일 조건과 실행 기록

- Official source: [저자 코드](https://github.com/sun-hailong/CVPR24-Ease/tree/6d67508ed8b7c4d23350ed4eb28c123914b40ae3), commit `6d67508ed8b7c4d23350ed4eb28c123914b40ae3`. 깨끗한 source를 실행 폴더로 복사하고 별도 worker에서 import했습니다.
- Dataset: 공유 `/mnt/sdab1/share/data/CIFAR100`, 10 classes/task, class order0..99, seed1993, validation split 없음. Task 1 train/test는 5000/1000 samples, Task 2 train/test는 5000/2000 samples입니다.
- Backbone: `vit_base_patch16_224.augreg_in21k_ft_in1k`, adapter64, FP32, autocast 없음, TF32 off. 세 크기의 확정 family는 동일하지만 Tiny·Small은 아직 실행하지 않았습니다.
- Checkpoint: 공유 Torch cache의 `B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz`, SHA256 `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`. 실제 local npz를 양쪽에서 로딩했고 새 dataset/weight 다운로드는 하지 않았습니다.
- Training: GPU1 RTX4090, batch48, workers4, 20epochs/task, SGD lr0.025/momentum0.9/weight decay0.0005, cosine minlr0. 학습은 공식 `Learner`를 사용하며 split·cache·sample order·loader seed·checkpoint만 맞췄습니다.
- Transform: 공식 CIFAR recipe의 train crop/flip 및 test bicubic resize/center crop. Native identity Normalize와 공식 Normalize 생략은 같은 값 변환입니다. 두 코드의 Task 1 raw 이미지·라벨·원본 sample ID fingerprint와 공유 train/test 파일 hash가 모두 일치했습니다. Prototype loader의 reference shuffle=True/native False 차이는 기록했습니다.
기존 CIFAR100 원본에는 Task 1 train/test 사이에 raw pixel이 같은 이미지 4쌍이 있습니다. Prototype은 train split만 사용했고 이 중복은 데이터 원본에 존재합니다. 저장된 예측에서 해당 test 이미지 4개를 제외한 참고 값은 992/996=99.5984%이며, 공식 정확도 996/1000=99.60%는 그대로 기록합니다. 공통 비교 데이터셋을 별도로 변경하지 않았습니다.

- Environment: Python3.12.15, torch2.13.0+cu126, torchvision0.28.0+cu126, timm0.9.8, numpy2.5.2, PyArrow25.0.1, easydict1.13, wandb0.29.0. 상세 provenance와 실행 코드 SHA256은 [JSON 결과](execution_results.json)에 있습니다.

초기 import SIGSEGV는 PyArrow vendored jemalloc background thread에서 발생했고, native import 전에 `JE_ARROW_MALLOC_CONF=background_thread:false`를 설정해 해결했습니다. 기존 명시적인 allocator 설정은 보존합니다. [환경 진단](environment/native_import_diagnosis.json)에 재현과 gdb 결과를 남겼습니다. 첫 smoke의 checkpoint argparse `NoneType` 오류와 첫 reference의 공식 namespace/native package 충돌은 학습 전에 실패했으며 원인과 로그를 보존하고 수정 후 재실행했습니다.

## 로컬·W&B·Slack

- [Native smoke W&B](https://wandb.ai/laymond1/cl-efficiency/runs/s12v4g1o), 로컬 `runs/ease-base-smoke-20261006T160323Z/`.
- [Official reference W&B](https://wandb.ai/laymond1/cl-efficiency/runs/ldobe8bo), 로컬 `runs/ease-official-reference-20261006T161154Z/`.
- [JSON 결과](execution_results.json), [태스크별 CSV](execution_results.csv), run 폴더의 metrics/telemetry/phases/epochs/predictions. Full checkpoints는 로컬에 저장했고 결과 파일은 W&B artifact로 저장했습니다.
- W&B Public API에서 두 run의 `finished`, artifact `COMMITTED`를 확인했습니다. Official artifact의 정확도·epoch 결과·자원 지표·데이터 fingerprint·runtime patch를 다시 내려받아 로컬과 모두 일치함을 확인했습니다. [서버 조회 기록](environment/wandb_reference_readback.json).
- Slack은 smoke 시작/완료, reference 시작/Task1완료/Task2완료/종료가 모두 `sent`입니다. 이는 webhook 요청 성공 상태이며 사용자의 채널 열람 여부를 뜻하지 않습니다. 비밀값은 결과에 저장하지 않았습니다.

## 검증과 다음 결정

당시 Plain imports로 focused tests **197 passed**, 실제 Base smoke와 공식 기준선 모두 exit0이었습니다. 실행 명령과 상세 검증은 [checks.md](checks.md)에 기록합니다. 이 보고서의 측정 범위에는 Native 두 태스크 학습·효율성 pilot·Tiny/Small 학습·추가 reference seeds·병합·push가 포함되지 않았습니다.

측정 당시에는 허용 accuracy 차이가 미확정이었으므로 위 reference 결과 JSON에는 margin null과 port acceptance 미평가 상태를 기록했습니다. 이후 2026-10-07 사용자 결정으로 각 지표의 하락 1.5%p 이내는 통과, 1.5%p 초과~3.0%p 이내는 추가 점검, 3.0%p 초과는 실패로 확정했습니다. 현재 기준과 지표별 하한은 [EASE 검증 계획](README.md#확정한-정확도-판정-기준-2026-10-07)과 [validation_plan.json](validation_plan.json)에 기록합니다.

단일 seed1993 결과로 seed 간 변동을 추정할 수 없으므로 이 폭은 운영 기준입니다. 당시에는 이 기준으로 포팅 코드의 두 태스크 평가를 다음 단계로 남겼습니다. 이후 검증은 [최신 Native 보고서](native_validation_report.md)에 기록하며, 추가 점검과 실패 상태에서는 병합·효율성 pilot을 진행하지 않는 정책을 유지합니다. 이 기준선은 원 논문의 5+5 split/class shuffle과 다른 조건이며 논문 성능 재현이 아닙니다.
