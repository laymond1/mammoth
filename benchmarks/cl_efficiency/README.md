# CL 알고리즘 검토와 효율성 실험 준비

EASE Base64·seed1993의 첫 2개 태스크 구현 검증이 통과했습니다. Native 정확도는 Task 1 99.40%, Task 2 전체 97.30%·과거 96.60%·신규 98.00%이며, 네 정확도 지표와 일곱 구조 검사가 모두 통과했습니다. [Native 검증 보고서](ease/native_validation_report.md)와 [JSON 결과](ease/native_validation_results.json)에 결과를 기록합니다. 알고리즘별 특징·장단점, 추가 구현 후보와 첫 태스크 효율성 비교 도구도 준비했습니다. 전체 알고리즘의 공통 효율성 비교는 아직 실행하지 않았으며, 기본 실행은 준비 파일 생성만 수행합니다.

EASE Base64의 공통 조건 첫 태스크 효율성 pilot도 **측정 3회와 별도 부분 FLOPs 진단 1회**를 완료했습니다. 세 측정의 정확도는 모두 96.50%이며 전체 관측 시간 중앙값 29.7629초, CUDA peak allocated 5760.5054 MiB, GPU 보드 에너지 중앙값 2.4432 Wh입니다. [효율성 보고서](ease/efficiency_pilot_report.md), [JSON 결과](ease/efficiency_pilot_results.json), [측정 CSV](ease/efficiency_pilot_results.csv)에 연산 범위와 측정 한계를 함께 기록합니다. W&B 저장·Slack 알림·독립 감사가 통과했고 실행 source `49897a3`을 로컬 공통 기반에 병합했습니다. 이 EASE pilot은 Base64만 대상으로 했으며 Tiny·Small과 원격 push는 수행하지 않았습니다.

RanPAC·L2P·DualPrompt의 첫 태스크 효율성 pilot도 측정9회·별도 부분 FLOPs3회를 완료했습니다. 신규 실행 source는 `20ec323`이며 동일 로컬 NPZ·EASE preprocessing·GPU1·seed1993에서 측정했습니다. EASE의 기존3회 측정을 재사용한 [네 방법 비교 보고서](baselines/efficiency_pilot_report.md), [JSON/CSV](baselines/efficiency_pilot_results.json)에 결과와 초기화 protocol 오류로 제외한 첫 시도를 기록합니다. 첫 태스크 결과만으로 전체 CL 성능·망각·논문 재현을 판정하지 않습니다.

## 검토 자료

- [현재 71개 모델의 특징과 장단점](algorithms.md)
- [추가 검토할 14개 항목과 우선순위: 방법 13개·예산 프로토콜 1개](recommendations.md)
- [BudgetCL·aL-SAR의 추가 가능성 및 이식 범위](budgeted_cl_review.md)
- [공통 환경·worktree 사용법과 10월 6일 연결 검증](foundation.md)
- [EASE 구현 범위·실행 결과·다음 단계](ease/README.md)
- [EASE Base64 native 검증 결과](ease/native_validation_report.md), [공식 기준선·smoke 측정 이력](ease/execution_report.md)
- [EASE·RanPAC·L2P·DualPrompt 공통 첫 태스크 비교](baselines/efficiency_pilot_report.md)
- [EASE 첫 태스크 효율성 pilot 결과](ease/efficiency_pilot_report.md), [측정 3회 JSON/부분 FLOPs](ease/efficiency_pilot_results.json)
- [ViT Base·Small·Tiny 및 adapter 비교 설정](ease/tiny_small_review.md)
- [표 형태의 전체 목록](inventory.csv)
- [모델별 데이터셋과 인자 및 외부 요구사항](manifest.json)
- [생성된 실행 계획](prepared/plan.json), [환경 점검](prepared/readiness.json)

71개는 EASE를 추가한 소스의 모델 엔트리 수입니다. 정적 추론 기준선 3개와 Joint 기준선 2개를 포함하며, 학습 방법과 그 변형은 66개입니다. CaSpeR·CSCCT·LiDER·STAR 변형도 독립 실행 엔트리로 기록합니다. STAR 정규화 변형과 STARPrompt는 서로 다른 방법입니다.

## 현재 준비 상태 (10월 7일)

독립 worktree `.worktrees/ease`, 브랜치 `codex/ease`에 EASE의 기본 replay-free 경로를 구현했습니다. 첫 2개 태스크의 구현 검증과 첫 1개 태스크의 효율성 측정을 분리했으며, 세 크기 모두 **AugReg IN21K → IN1K fine-tuned**, 공통 adapter 64 및 Tiny16·Small32·Base64의 비율 정렬 비교를 준비했습니다. Base64 native 검증은 동일 조건 공식 기준선 대비 사용자 판정 기준을 통과했습니다. 이후 승인한 Base64 효율성 측정 3회·부분 FLOPs 진단 1회도 통과했고 W&B run 종료와 artifact COMMITTED 상태를 서버에서 확인했습니다. 실행 source `49897a3`은 `codex/cl-foundation`에 fast-forward 병합했습니다. Tiny·Small 학습과 원격 push는 수행하지 않았습니다. 이후 승인한 세 방법의 비교는 아래 후속 결과에 기록합니다.

현재 manifest는 71개, 실행 대상으로 준비한 엔트리66개·보류5개입니다. 계획 수를 완료한 실험 수로 해석하지 않습니다. 공통 첫 태스크 효율성 설정은 EASE Base64·RanPAC·L2P·DualPrompt의4개이며, 신규 세 방법은 명시 checkpoint·공통 preprocessing을 적용한 pilot입니다. 나머지 방법과 네 Tiny·Small 설정은 미실행입니다. [네 방법 비교 보고서](baselines/efficiency_pilot_report.md), [EASE 구현 검증](ease/native_validation_report.md)을 참고합니다. 아래10월1일·6일 기록은 당시의 상태입니다.

## 공통 기반 준비 당시 상태 (10월 6일)

공통 기반 `codex/cl-foundation`에 Python 3.12/uv lock, 물리 GPU **1**, W&B **laymond1/cl-efficiency**, 공유 데이터·캐시 경로 및 개인 설정 로더를 준비했습니다. CUDA 초기화·의존성 import·공유 경로 접근을 확인했고, W&B 연결 run의 서버 summary와 artifact 업로드, Slack 테스트 메시지 전달을 검증했습니다. 환경 상세는 로컬에만 저장하며 외부 연결 테스트에는 합성 표식만 전송했습니다.

EASE부터 구현하되 구현 검증과 논문 성능 재현을 분리합니다. 동일 조건의 검증에서 비정상적으로 낮은 성능은 구현 검증 실패로 취급합니다. BudgetCL·aL-SAR의 계산 예산 비교는 후속 단계입니다. **이번에는 EASE 및 다른 후보 구현, CL 학습·추론을 실행하지 않았습니다.**

현재 실행 계획은 GPU 1, 공유 데이터 루트, `num_workers=4`, `permute_classes=0`으로 갱신했습니다. 65개 실행 준비·5개 보류 상태를 유지합니다. 상세 명령과 검증 이력은 [공통 환경 문서](foundation.md)를 참고합니다.

## 10월 6일 후보 검토 반영

EASE와 HiDe-Prompt를 우선 검토하는 8개 방법에 포함했습니다. EASE는 태스크별 adapter 증가와 추론 비용, HiDe-Prompt는 태스크별 TII 학습·prompt 학습·routing 비용까지 함께 검토합니다.

BudgetCL은 계산 예산을 제한하는 평가 프로토콜과 uniform-history 기준선으로, aL-SAR은 적응적 layer 동결·빈도/유사도 기반 replay 방법으로 추가 가능하다고 판단했습니다. aL-SAR은 우선 ResNet 구현을 권장합니다. **현재 runner는 자원 측정 도구이며 계산 예산을 강제하는 controller는 구현하지 않았습니다.** 첫 태스크에서 측정한 부분 FLOPs나 동일 epoch 수로 BudgetCL·aL-SAR의 예산 조건을 충족했다고 해석하지 않습니다.

후보 검토 시점에는 문서만 변경했습니다. 후보 구현과 manifest 엔트리는 추가하지 않았으며, 기존 70개 목록과 65개 실행 준비 계획은 유지합니다. 이후 환경 준비 단계에서 합성 W&B·Slack 연결 테스트만 별도로 수행했습니다.

## 실행 조건

| 그룹 | 공통 데이터 | 모델 조건 |
|---|---|---|
| `classic_resnet224` | seq-cifar100-224, 첫 태스크 10 classes, clean labels | 일반 CL은 ResNet18, 224 pixel 입력 |
| `pretrained_224` | 위와 동일한 샘플·태스크·라벨 | Prompt·PEFT·사전학습 방법; CLIP와 ViT/사전학습은 별도 확인 |
| `noisy_clean_224` | 위와 동일한 샘플·태스크·라벨 | 노이즈 CL의 clean-label 조건. true_labels=labels 메타데이터만 제공 |
| `unsupported_biased` | 동일 데이터셋을 지정하되 실행 보류 | LwS는 biased-class-il 필드가 필요해 이 비교에 사용할 수 없음 |

사용자의 동일 데이터셋 요청에 맞춰 **모든 manifest 엔트리에 seq-cifar100-224를 지정**했습니다. 공통 pilot의 확정 조건은 outer loop **1 epoch, batch size 64, seed 1993, num_workers=4, permute_classes=0(클래스 재배열 미적용)**이며, 원본 rehearsal buffer가 있는 경우 500개 샘플입니다. 첫 태스크는 원래 클래스 라벨 0~9로 구성합니다. noise_rate=0, 모든 라벨 제공, validation split 없음으로 샘플과 라벨 조건을 통일합니다. 모델별 LR·optimizer·필수 transform은 manifest와 기존 parser defaults를 사용합니다. `--debug_mode=0`, `--stop_after=1`, `--eval_future=0`, `--enable_other_metrics=0`으로 전체 첫 태스크를 처리합니다. CLASS/TASK-IL 마스킹에 따라 정확도의 의미가 다르므로 모델의 COMPATIBILITY도 확인합니다.

위 조건은 10월 6일 확정한 설정이며 실행 도구와 `prepared/` 실행 계획에도 반영했습니다. 실행 계획은 측정 결과가 아닙니다.

native `MammothDatasetWrapper`가 만든 샘플 index/label의 SHA256을 train/test별로 기록해 실제 실행 후 같은 샘플인지 확인할 수 있습니다. SPR와 ER-ACE-AER-ABS에 필요한 true_labels는 wrapper의 `add_extra_return_field`로 clean labels의 복사본을 제공합니다. 노이즈를 주입하지 않으며 meta wrapper를 그대로 사용합니다. 이 조건으로 노이즈 강건성이나 CSSL 효과를 판단할 수는 없습니다.

모든 알고리즘의 내부 학습을 1 epoch로 바꾸지는 않습니다. GDumb의 buffer fitting, STARPrompt의 첫 단계와 생성 replay, `second_order`의 pre-tuning/Fisher/alignment, SPR의 expert 학습은 고유 알고리즘 비용으로 그대로 포함합니다. **1개 태스크가 짧은 실행을 보장하지 않습니다.** 기본 timeout은 pass당 2시간이고, 초과/실패/누락을 결과에 표시합니다. 논문 재현용 최적 hyperparameter나 공정한 최종 성능 순위로 해석할 조건은 아닙니다.

**71개 중 66개를 첫 태스크 실행 대상으로 준비하고 5개는 보류**합니다. Joint와 Joint-GCL은 다른 protocol이 필요합니다. LwS는 편향 데이터 필드가 필요하고, PuriDivER는 constructor에서 seq-cifar100-224를 허용하지 않습니다. CNLL은 hard augmentation이 32 pixel crop으로 고정되어 공통 224 pixel 입력과의 혼합 학습 경로를 이식해야 합니다. 보류한 방법의 알고리즘이나 protocol을 이번 비교에 맞추어 수정하지는 않았습니다.

ZSCL은 학습용 기준 데이터는 동일하게 사용하지만 distillation reference로 `data/conceptual_captions`를 추가로 필요로 합니다. CLIP·LLaVA·IDEFICS는 학습 없이 첫 태스크의 추론 비용과 정확도를 측정합니다. TwF pilot은 별도 논문 pretrained teacher checkpoint를 지정하지 않습니다. AttriCLIP에는 원논문 성능 미재현 경고가 있으므로 추천/순위에 그 한계를 표시합니다.

데이터셋은 동일하지만 backbone/사전학습 checkpoint, 모델의 필수 transform, optimizer, effective batch/virtual batch, precision은 다를 수 있습니다. 그룹은 다른 데이터셋을 뜻하지 않고 모델 조건의 차이를 나타냅니다. 실행별 `effective_args.json`, 모델 스냅샷, task hook 이후 transform 기록을 함께 확인해야 합니다. 전체 시스템의 실용 비용 비교와 같은 backbone을 사용한 방법 자체의 비교를 구분해 해석합니다.

## 수집하는 지표

| 항목 | 로컬 결과 | 의미와 제한 |
|---|---|---|
| 첫 태스크 정확도 | native Mammoth logs, metrics.json, results.csv | class-il와 task-il을 구분. PNN의 native class-il placeholder 0은 추가 결과에서 null/unsupported로 표시; 첫 태스크에서 forgetting/BWT는 미측정 |
| 샘플 조건 | data_snapshots, train/test_data_sha256 | 같은 index/label 조합인지 비교. 초기 loader 및 task hook 이후 실제 transform을 별도 기록 |
| 실행 시간 | process_wall_s, phases.json | 초기화·data_loading·begin_task·train_epoch·end_task·evaluation을 구분. CUDA를 phase 경계에서 동기화; 중첩 phase 시간은 합산하지 않음 |
| CPU/GPU 메모리 | telemetry.csv, allocator peak | process+live children RSS 샘플링, PyTorch allocated/reserved peak, 전체 GPU board 메모리. RSS shared page와 다른 GPU 프로세스의 영향이 있음 |
| 파라미터와 상태 저장량 | model_snapshots | initialize/begin_task/end_task 후 등록된 전체·trainable·optimizer 파라미터 수, parameter/buffer/optimizer/replay/aux tensor bytes. unregistered module은 개수에 빠질 수 있으며 RSS/allocator 수치와 함께 확인. 서로 겹치는 범주를 합쳐 총메모리로 쓰지 않음 |
| 처리량 | outer_training_examples_per_s, evaluation_examples_per_s | 일반 epoch loop의 입력 수와 평가 forward 입력 수. replay 및 task hook 내부 학습 샘플을 전체 데이터 접근량으로 가장하지 않음 |
| GPU 전력과 에너지 | power samples, gpu_board_energy_j/wh, coverage | nvidia-smi power의 사다리꼴 적분. 선택한 물리 GPU의 idle 및 다른 프로세스 전력 포함; CPU/RAM/벽전력 제외. missing/gap은 보간·0 대체하지 않음 |
| CPU 시간 | process_tree_cpu_s | worker와 종료한 children의 누적 user+system CPU 시간; telemetry용 nvidia-smi 실행 비용 포함 |
| FLOPs | flops_ops.json, flops_partial | 별도 diagnostic pass의 PyTorch 지원 operator 추정값. 주로 matmul/conv; fused attention·linear solve 등은 누락될 수 있음. 전체 모델의 정확한 FLOPs로 해석하지 않음 |

기본 `--flops paired`는 **같은 첫 태스크를 두 번 실행**합니다. 첫 번째 pass는 시간·메모리·전력 비교, 두 번째 pass는 profiler로 연산량을 조사합니다. profiler pass의 시간·메모리·에너지는 비교 CSV의 효율성 열에 넣지 않습니다. 두 pass는 같은 seed/config를 사용하지만 cache가 warm해질 수 있으므로 profiler pass의 속도를 첫 pass의 속도와 비교하지 않습니다. `--flops off`를 선택하면 한 번만 실행하며 연산량은 미측정입니다.

FLOPs 수치의 범위는 [PyTorch profiler 문서](https://docs.pytorch.org/docs/stable/profiler.html)의 `with_flops` 지원 범위에 따릅니다. GPU 센서가 없으면 에너지는 `null`로 저장합니다. 짧은 phase는 power sample coverage가 낮거나 없을 수 있습니다. energy coverage fraction을 반드시 함께 봐야 하며, task의 partial energy를 정확도와 나눠 전체 효율처럼 비교하면 안 됩니다. 전체 머신 에너지가 필요하면 외부 전력계나 검증된 CPU/RAM 계측을 추가해야 합니다.

## 준비만 하는 명령

저장소 표준 환경에서는 아래 명령이 **실험을 실행하지 않습니다**. 모델을 import하거나 checkpoint/dataset을 다운로드하지 않고 AST 목록 및 명령 파일을 생성합니다. 읽기 전용 GPU 센서 조회는 수행합니다.

```bash
uv run --no-sync python scripts/benchmark_cl.py
uv run --no-sync python scripts/benchmark_cl.py --models l2p,dualprompt,coda-prompt,ranpac,second-order,tak
```

준비 결과는 `benchmarks/cl_efficiency/prepared/`의 `plan.json`, `inventory.json`, `commands.txt`, `readiness.json`입니다. `commands.txt`는 native Mammoth 명령 참조용이며, telemetry·Slack·artifact 저장은 아래 runner를 통해 실행할 때 적용됩니다.

## W&B와 Slack 설정

공통 [runtime.json](runtime.json)의 기본 entity/project는 `laymond1/cl-efficiency`입니다. `--wandb-entity`/`--wandb-project` 또는 `WANDB_ENTITY`/`WANDB_PROJECT`로 덮어쓸 수 있습니다. 인증은 기존 `wandb login`, `WANDB_API_KEY` 또는 비공개 `wandb.conf` 로더로 제공하며 비밀값을 manifest나 문서에 적지 않습니다. worktree 간 비공개 설정 참조는 [공통 환경 문서](foundation.md)를 따릅니다.

Slack은 **Incoming Webhook이 연결된 채널**로 알립니다. `SLACK_WEBHOOK_URL` 환경변수에 실제 URL을 설정해야 합니다. URL은 결과나 W&B config에 저장하지 않습니다. 임의 채널을 선택하는 Slack connector 연결은 이 도구에서 수행하지 않습니다. [Slack Incoming Webhooks 문서](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/)

실험이 시작되면 suite 시작, 모델별 시작/완료/실패, 진행 중 10분 간격, suite 완료/중단을 전송합니다. 알림의 성공/실패는 `notifications.json`에 남고, 측정 실패와 알림 실패를 구분합니다. 10월 6일 환경 준비의 연결 확인에 이어 EASE 검증·효율성 실행의 진행 알림을 사용했고, 이번 효율성 pilot의 모든 suite 알림은 `sent`였습니다.

W&B에는 기존 Mammoth의 학습 loss/accuracy와 추가 효율성 scalar summary, 실행 config, 측정 파일 artifact가 같은 run에 저장됩니다. measurement와 FLOPs pass는 별도 run이며 suite group으로 연결합니다. 로컬에는 run별 native 로그·effective args·metrics·telemetry·phase 기록·console 로그와 suite 통합 JSON/CSV가 남습니다. artifact 업로드는 [W&B SDK 방식](https://docs.wandb.ai/models/artifacts/construct-an-artifact)을 사용합니다.

`--wandb-mode offline`은 로컬 W&B 기록만 저장하므로 나중에 `wandb sync`가 필요합니다. `--wandb-mode disabled --slack off`는 명시적으로 외부 기록/알림을 끕니다. 기본 online/on 실행에서 필요한 설정이 없으면 학습 전에 실패하며 외부 저장 없이 성공한 것처럼 진행하지 않습니다.

## 추후 실행할 명령

아래 범용 예시 명령은 `--execute`가 있어 실제 학습을 시작합니다. 이 전체 모델 묶음은 실행하지 않았습니다. 완료된 RanPAC·L2P·DualPrompt pilot은 [별도 승인 runner와 결과](baselines/README.md)를 따릅니다. 범용 manifest는 기존 고유 기본 설정을 유지하므로 해당 명령을 완료된 공통 checkpoint/preprocessing 비교와 동일하게 취급하지 않습니다. 추가 방법의 실행은 후속 범위입니다.

```bash
uv run --no-sync python scripts/benchmark_cl.py \
  --models l2p,dualprompt,coda-prompt,dap,ranpac,second-order,tak \
  --wandb-entity laymond1 --wandb-project cl-efficiency \
  --gpu 1 --output benchmarks/cl_efficiency/runs --execute
```

`--models all`이 기본값이며 위 5개는 보류 상태로 기록합니다. 실험은 GPU 간섭을 줄이기 위해 순차 실행하고, 실패한 모델의 로그를 남긴 뒤 나머지 모델을 계속 실행합니다. 일부 모델은 optional dependency, 대형 checkpoint 또는 외부 데이터 때문에 실행 시 실패할 수 있습니다. cache/사전학습 weight 준비 시간은 초기화 phase에 포함되므로 반복 비교에서는 같은 cache 상태를 유지해야 합니다.

## 도구 준비 당시 환경 확인과 검증 (10월 1일)

현재 셸의 기본 Python은 3.8이며 `uv`가 PATH에 없습니다. Conda base에 Python 3.12.2와 pytest가 있어 도구 테스트 및 dry run에 사용했습니다. GPU 조회는 NVIDIA driver 통신 실패입니다. Conda base의 Mammoth `--help`는 기존 `open_clip` 누락으로 실패했고, `wslim` Conda 환경에서는 기존 W&B SDK가 `wandb.proto.wandb_telemetry_pb2.Deprecated` import 오류를 냅니다. W&B entity/project/API key는 현재 실행 도구의 환경변수에 없습니다. SDK cache에 저장된 인증 여부는 조회하지 않았습니다.

사용자는 SLACK_WEBHOOK_URL을 등록했다고 알려주셨지만, 현재 Codex 실행 도구의 자식 프로세스에서는 환경변수가 보이지 않습니다. 비밀값은 조회·출력하지 않았습니다. 실제 실행 프로세스가 등록한 환경변수를 상속해야 알림이 전송되며, runner는 실행 직전에 이를 다시 확인합니다. 현재 Slack 전달을 검증한 상태는 아닙니다.

따라서 준비 도구의 로컬 검증과 실제 모델 실험 검증은 구분됩니다. 실험을 시작하려면 저장소의 `uv sync --group dev` 환경을 준비하거나 기존 Conda 의존성을 정리하고, GPU driver 및 W&B/Slack 대상을 설정해야 합니다. 환경 설치·학습·추론·실제 업로드·Slack 전달은 이번에 검증하지 않았습니다.

실제로 실행한 검증 명령과 결과는 다음과 같습니다. 알고리즘 학습을 시작하는 smoke run은 사용자의 실행 보류 요청에 따라 수행하지 않았습니다.

```bash
/mnt/sdab1/userHome/wslim/miniconda/bin/python -m py_compile scripts/benchmark_cl.py scripts/cl_efficiency.py tests/test_cl_efficiency.py
/mnt/sdab1/userHome/wslim/miniconda/bin/python scripts/benchmark_cl.py --help
/mnt/sdab1/userHome/wslim/miniconda/bin/python scripts/benchmark_cl.py --output benchmarks/cl_efficiency/prepared
```

위 세 명령은 통과했고, dry run 결과는 **70개 목록, 실행 계획 65개, 보류 5개, experiment_executed=false**입니다.

새 도구만 검증하는 테스트는 repository package의 선행 import와 기존 pytest plugin을 피하도록 임시 폴더에서 실행했습니다. framework와 W&B의 통합 경로는 mock으로 확인하며 실제 모델·데이터·네트워크를 사용하지 않습니다.

```bash
mkdir -p /tmp/cl-efficiency-validation
cp tests/test_cl_efficiency.py /tmp/cl-efficiency-validation/test_cl_efficiency.py
cd /tmp/cl-efficiency-validation
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl \
/mnt/sdab1/userHome/wslim/miniconda/bin/python -m pytest --noconftest \
  --rootdir=/tmp/cl-efficiency-validation /tmp/cl-efficiency-validation/test_cl_efficiency.py -q --tb=short
```

**27 passed**. 준비 시 실행/알림 차단, 단일 데이터셋 강제, clean-label 메타데이터, task-only 모델의 미지원 정확도 표시, meta hook 호출/복구, 실패 후 결과 보존, power 적분/coverage, profiler pass 분리, W&B 업로드/finish 순서와 Slack 비밀값 비노출을 검증합니다.

기존 환경 때문에 실패한 확인도 기록합니다. Conda base Python으로 `-m pytest tests/test_cl_efficiency.py -q`를 실행하면 기존 pytest-rerunfailures plugin의 socket 생성이 제한 환경에서 차단됩니다. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`로 재시도하면 repository `__init__.py`의 선행 import에서 기존 `kornia` 누락으로 실패합니다. 같은 Python으로 `main.py --model sgd --dataset seq-cifar10 --help`를 실행하면 기존 `open_clip` 누락으로 실패합니다. 기존 모델 테스트 suite와 실제 GPU 실행은 통과한 것으로 표시하지 않았습니다.

## 후보 문서 작성 당시 검증 (10월 6일, 환경 설치 전)

이번 문서 변경은 Conda base Python의 표준 라이브러리 검사로 확인했습니다. 3개 문서의 표 7개·로컬 링크 8개·공백 형식, 후보 수 8+4+2=14, 기존 inventory/manifest/plan 70개, 실행 준비 65개·보류 5개, 공통 데이터셋과 `experiment_executed=false`가 일치했습니다. 모델 import·실험·네트워크 전송을 수행하는 검사는 실행하지 않았습니다. 위의 도구 테스트 27개는 10월 1일 검증 결과이며 이번에 다시 실행하지 않았습니다.

문서 빌드 playbook의 첫 명령인 `uv run ../utils/args.py`를 `docs/`에서 실행했으나 `uv: command not found`로 실패했습니다. 따라서 Sphinx 문서 빌드는 수행하지 못했으며, 문서 정적 검사와 논문·공식 소스 검토까지만 완료했습니다.
