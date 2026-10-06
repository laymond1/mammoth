# EASE 포팅 구현 검증 결과 — 2026-10-07

ViT-B/16·adapter64의 Mammoth 포팅 구현이 동일 조건 공식 코드 대비 **첫 2개 태스크 검증을 통과**했습니다. 네 정확도 지표와 일곱 구조 검사가 모두 통과했으며, W&B 서버의 종료·artifact 저장과 Slack 요청 성공을 확인했습니다. 브랜치는 `codex/ease`이며 공통 기반에 병합하지 않았습니다.

## 정확도와 판정

| Class-IL 지표 | 공식 코드 | 포팅 코드 | 공식 대비 하락 | 판정 |
|---|---:|---:|---:|---|
| Task 1 전체 | 99.60% (996/1000) | 99.40% (994/1000) | 0.20%p | 통과 |
| Task 2 전체 | 97.20% (1944/2000) | 97.30% (1946/2000) | −0.10%p | 통과 |
| Task 2 과거 클래스 | 96.60% (966/1000) | 96.60% (966/1000) | 0.00%p | 통과 |
| Task 2 신규 클래스 | 97.80% (978/1000) | 98.00% (980/1000) | −0.20%p | 통과 |

사용자와 확정한 기준은 지표별 하락 **1.5%p 이내 통과, 1.5%p 초과~3.0%p 이내 추가 점검, 3.0%p 초과 실패**입니다. 음수 하락은 정확도 개선입니다. 정답 수와 샘플 수를 유리수로 비교하여 표시값 반올림으로 경계를 판정하지 않았습니다. 전체 판정은 가장 나쁜 지표·구조 검사 결과를 따릅니다.

단일 seed1993의 구현 검증 결과이며 seed 간 변동을 추정하지 않습니다. 모든 지표가 경계에서 충분히 떨어져 있어 추가 seed를 실행하지 않았습니다. 공식 논문의 5+5 split·class shuffle과 다른 10-class 조건이므로 논문 성능 재현으로 분류하지 않습니다.

## 동일 조건과 구현 구조

- 데이터: `seq-cifar100-224`, 순서0..99, 10 classes/task, 첫 2 tasks, seed1993. 태스크별 학습5000개, 평가1000→2000개, validation 분할·라벨 노이즈·부분 라벨 없음.
- 가중치: `vit_base_patch16_224.augreg_in21k_ft_in1k`, 공유 cache의 실제 NPZ, SHA256 `ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124`.
- 학습: 물리 GPU1 RTX4090, 20 epochs/task, batch48, workers4, SGD lr0.025/momentum0.9/weight decay0.0005, cosine minlr0.
- 수치 설정: FP32, autocast 없음, CUDA matmul TF32 off, cuDNN deterministic=True·benchmark=False. cuDNN TF32 허용 상태도 공식 코드와 같이 기록합니다.
- 데이터 변환: 공식 CIFAR train crop/flip, test bicubic resize/center crop. 각 태스크의 train/test 이미지·라벨·원본 sample ID fingerprint가 공식 기준선과 일치합니다.
- 학습 경로: 기존 `utils.training.train`과 `meta_begin_task`/`meta_observe`/`meta_end_task`를 사용했습니다. 별도의 대체 학습 루프를 만들지 않았습니다. 진단 도구가 W&B run을 소유하고 native autologging은 비활성화했습니다.

| 구조 검사 | 결과 |
|---|---|
| 유한 loss·학습 파라미터·gradient | 통과 |
| 공식 기준선과 raw 데이터 일치 | 통과 |
| Frozen backbone 값 보존 | 초기·Task1·Task2 hash 일치 |
| Frozen backbone gradient 미생성 | 40 epochs 검사 통과 |
| 과거 adapter 값 보존 | Task1→Task2 hash 일치 |
| Adapter/prototype lifecycle | adapter1→2, prototype `[10,768]`→`[20,1536]`, task finalized 확인 |
| 승인한 전체 학습 protocol | 40 epochs, 4200 updates, 200000 train presentations 완료 |

저장된 NPZ 예측을 다시 집계하고, CPU에서 두 체크포인트의 텐서 유한성·동결 텐서 값 보존을 독립 확인했습니다. [독립 감사 기록](environment/native_independent_audit.json)을 함께 보존합니다. Prototype은 원본 training split으로 만들었습니다. CIFAR 원본 Task1의 train/test 동일 이미지 4쌍은 기존 기준선과 같으며 데이터셋을 변경하지 않았습니다.

초기화가 소비하는 RNG 순서, prototype loader의 native sequential/official shuffle=True, native 평가의 태스크별 loader 방문 순서는 구현 간 차이로 기록했습니다. 저장한 예측은 원본 sample ID 순으로 정렬하여 비교합니다. 공식 코드의 기준선과 실행 당시 결과는 [기존 보고서](execution_report.md)·[기존 JSON](execution_results.json)에 그대로 보존합니다.

## 자원 관측

| 포팅 코드의 측정 항목 | 결과 |
|---|---:|
| Recorder 전체 관측 시간 | 559.16초 (약 9분 19초) |
| 학습 epoch 시간 합계 | 522.16초 |
| Prototype 계산·보완 | 18.80초 |
| 평가 | 6.68초 |
| Parent 전체 시간 | 571.54초 |
| PyTorch CUDA peak allocated | 4421.40 MiB (4.32 GiB) |
| PyTorch CUDA peak reserved | 4792.00 MiB |
| GPU1 device memory peak | 5283 MiB |
| 프로세스 트리 RSS 합계 peak | 15181.51 MiB |
| GPU1 board energy | 228616.05 J = 63.50 Wh |
| 에너지 관측 coverage | 99.9941%, 센서 오류 없음 |
| 학습 / 평가 presentations | 200000 / 3000 |
| Prototype adapter-image pairs | 15000 |
| FLOPs | 이번 단계에서 측정하지 않음 |

Recorder 구간에는 import·초기화·학습·검증 감사와 로컬 결과/체크포인트 저장이 포함되고, 마지막 W&B artifact 업로드·종료는 제외됩니다. Parent 시간에는 마지막 외부 저장·알림 작업이 포함됩니다. `end_task`에 `prototype_completion`이 포함되므로 phase 시간을 중복 합산하지 않습니다. GPU board energy는 물리 GPU1의 센서 적분값이며 CPU/RAM·전체 머신 에너지는 아닙니다. RSS는 프로세스별 합계로 공유 페이지를 중복 셀 수 있으며 CUDA allocated/reserved 및 전체 device memory와 다른 지표입니다.

공식 코드와 native의 계측·framework overhead가 다르므로 두 관측의 차이로 알고리즘 효율성 순위를 매기지 않습니다. 동일 측정 조건의 첫 1-task/1-epoch pilot과 별도 FLOPs pass는 후속 단계입니다.

Task2 모델의 등록 파라미터는 **88185601개 / 352742404 bytes**, 등록 buffer는122917 bytes입니다. 현재 태스크 학습 파라미터1197313개는 adapter1189632개와10-class proxy head7681개입니다. Task 완료 후 trainable 수는0이며 optimizer state4789252 bytes는 남아 있습니다. Adapter가 `ModuleList`, prototype이 buffer로 등록되어 과거 adapter와 classifier 저장량도 집계됩니다. 원본 이미지 replay buffer는 없습니다.

## 로컬·W&B·Slack과 검증

- [Native validation W&B](https://wandb.ai/laymond1/cl-efficiency/runs/n3nlg2ah): `finished`, artifact `ease-native-validation-20261006T174313Z:v0`는 `COMMITTED`.
- [JSON 결과](native_validation_results.json), [네 지표 CSV](native_validation_results.csv), 로컬 [run 폴더](/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.worktrees/ease/benchmarks/cl_efficiency/ease/runs/ease-native-validation-20261006T174313Z/). Full checkpoints는 EASE worktree의 ignored run 폴더에 저장하고 metrics·예측·telemetry·phase·acceptance는 W&B artifact에 저장했습니다. Git 병합은 이 원시 폴더를 복사하지 않습니다.
- [W&B 서버 조회 기록](environment/wandb_native_readback.json): 원격 artifact의 정확도·epoch·자원 지표·데이터 fingerprint·동결 audit·판정·정밀도와 실행 spec가 로컬과 일치합니다.
- Slack 시작·Task1·Task2·종료 알림의 요청 상태가 모두 `sent`입니다. Webhook 요청 성공을 확인한 것이며 채널 열람 여부는 측정하지 않습니다. 비밀값은 보고서에 저장하지 않았습니다.
- Focused tests **239 passed**와 실제 native 실행 exit0. 실제 명령과 후속 확인은 [checks.md](checks.md)에 기록합니다.

실행 당시 Git HEAD는 `4f89aed`이고 새 runner/helper는 아직 커밋 전이었습니다. 실행한8개 source/lock 파일의 SHA256을 spec와 JSON에 보존하고 완료 후에도 동일함을 확인했습니다. 실행한 source bytes를 로컬 커밋 `4976fe5`에 보존했습니다. 공식 기준선 JSON도 실행 spec에 고정한 hash 그대로입니다.

이 단계는 완료됐습니다. 다음 검토 대상은 첫 태스크 효율성 pilot의 실행 범위와 병합 시점입니다. 이번에는 효율성 pilot·FLOPs·Tiny/Small 학습·추가 seeds·병합·push를 수행하지 않았습니다.
