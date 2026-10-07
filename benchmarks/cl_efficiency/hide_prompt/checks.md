# HiDe-Prompt Base 구현·GPU 검증 기록 — 2026-10-07

`codex/hide-prompt`에서 실제 CIFAR100 첫2태스크 독립 저자/native 검증을 완료했고, 정확도4항목·구조13검사와 독립 감사가 **PASS**입니다. 아래 CPU 준비 기록은 실행 전 이력이며, 실제 GPU 학습·효율성 계측 기록은 별도 단락에 적습니다. 고정한 저자 commit은 `0ed433334ff2cfb206c5e2899651c3044ec85f92`이며 원본 저장소는 변경하지 않았습니다. 이번 결과는 명시한 적용 사항 아래의 구현 검증이며 논문 표 수치 재현이 아닙니다.

## 초기 CPU 준비 이력: 실제 실행한 명령과 결과

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline pytest tests/test_hide_prompt.py tests/test_hide_network.py tests/test_hide_statistics.py tests/test_prepare_hide_prompt.py tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_cl_runtime.py tests/test_baseline_pretrained_args.py -q --tb=short
```

229 passed, 0 failed (8.13s). 실제 데이터셋이나 사전학습 ViT를 학습하지 않고 작은 합성 feature·토큰의 수식, gradient, fake 분류기 optimizer 계약, task lifecycle, checkpoint, CLI 우선순위를 검사했습니다. 기존 EASE 및 공통 환경·가중치 인자 회귀 검사를 포함합니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline pytest tests/test_module_register.py::test_args_with_choices tests/test_module_register.py::test_register_backbone tests/test_module_register.py::test_register_dataset tests/test_prepare_hide_prompt.py -q --tb=short
```

9 passed, 0 failed (3.88s). 기존 `test_module_register.py` 중 실제 학습을 시작하지 않는 `--help` 등록·동적 인자3개와 최종 준비 도구6개입니다. 두 명령의 고유 테스트는232개입니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 JE_ARROW_MALLOC_CONF=background_thread:false UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline python /mnt/sdab1/userHome/wslim/.codex/skills/implement-pcl-method/scripts/check_pcl_method.py hide-prompt
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 JE_ARROW_MALLOC_CONF=background_thread:false UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline python main.py --model hide-prompt --dataset seq-cifar100-224 --model_config default --help > /tmp/hide-prompt-help-20261007.txt
python3 -S scripts/prepare_hide_prompt.py --check-local
```

모두 exit0입니다. Native `--help`를 smoke로 사용했고, 스킬의 debug 학습 명령과 실제 학습을 포함한 일반 import/config/basic suite는 사용자가 정한 이번 범위에 따라 후속으로 두었습니다. 모델·backbone discovery와 config/CLI 병합은 별도의 CPU 검사로 확인했습니다.

Ruff도 시도했지만 공유 환경에 실행 파일이 없어 exit2였습니다. 의존성을 추가하거나 환경을 sync하지 않았습니다. Python AST 파싱과 `git diff --check`를 수행했습니다.

## 독립 대조와 기존 가중치

[Network 대조](environment/network_source_audit.json)는 고정한 저자 AST 함수와 native의9개 조건에서 출력과 모든 parameter gradient의 차이가0임을 확인합니다. [통계·alignment 대조](environment/statistics_source_audit.json)는 두 branch × 세 storage mode의6개 조건에서 통계, 실제 합성 분류기 alignment 파라미터, prompt regularizer가 일치합니다. 이는 작은 합성 계산의 검증이며 실제 CL 성능을 증명하지 않습니다.

[초기 Base CPU 구성](environment/native_cpu_initialization.json)에서는 두 branch가 모두 기존 NPZ의 canonical pretrained tensor150개 SHA와 일치했습니다. 당시 이미지 forward와 Base optimizer step은0입니다. KMeans 초기화/seed, 클래스별 통계 batching/순서, retained prefix layout은 [소스 검토](source_review.json)와 [초기 계획](validation_plan.json)에 명시합니다. 이후 실제 독립 기준선/native GPU 검증에서는 learnable MLP/head/prompt 초기값도 동일 export 파일과 SHA로 맞췄습니다.

[Preflight JSON](environment/preflight.json)에 당시 native source SHA·각 검증 명령·공유 환경·기존 checkpoint hash를 보존합니다. 초기 준비 도구는 학습/ML import/네트워크/credential/센서를 사용하지 않으며, 당시 생성한 `executed`와 학습 권한은 모두false입니다. 이후 GPU 실행 승인은 [execution_plan.json](execution_plan.json)에 별도로 기록합니다.

## 초기 준비 당시의 후속 범위 — 이력

초기 CPU 준비 단계에서는 독립 저자 runner·동일 초기화/phase RNG 준비와 첫2태스크 TII20+prompt50 및 태스크2의 두 CA30 실행이 후속 범위였습니다. Native 이미지 smoke, W&B/Slack 실험 기록, 실제 정확도 판정과 효율성 계측은 이후 승인된 GPU 단계에서 진행했습니다. 기준 브랜치 병합은 이 검증 기록과 구분합니다.

## 승인된 GPU 실행 단계 준비 — batch64

독립 기준선·native·효율성 runner와 기존 모델/환경 검사를 함께 실행했습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/hide-final-uvcache UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline pytest tests/test_hide_reference.py tests/test_hide_execution.py tests/test_hide_efficiency.py tests/test_hide_prompt.py tests/test_hide_network.py tests/test_hide_statistics.py tests/test_cl_runtime.py -q --tb=short
```

230 passed, 0 failed (8.98s). 공유 경로 및 다운로드 차단 회귀, 실제 학습 배치/RNG 비교, CA gradient와 평가 finite 검사를 포함합니다.

GPU1 batch64에서 독립 기준선 smoke(단계당2배치)와 native smoke(단계당79배치)를 완료했습니다. 두 smoke는 학습량이 달라 정확도 acceptance에는 사용하지 않습니다. [첫 배치 대조](environment/matched_smoke_audit.json)에서 네 단계의 input/label/seed가 일치하며, 태스크1 두 단계의 loss 및 clipped gradient도 일치합니다.

W&B finished/COMMITTED artifact readback에서 smoke64 파일4개, 독립 smoke13개, native smoke10개의 로컬 SHA가 일치했습니다. 실패한 준비 시도는 [실행 시도 기록](environment/execution_attempts.json)에 보존합니다.

## 최종 보고서 publisher CPU 검사

작업 디렉터리는 모든 명령에서 `/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.worktrees/hide-prompt`입니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/cl-hide-efficiency-uvcache UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline pytest tests/test_publish_hide_report.py -q --tb=short
```

19 passed, 0 failed (4.41s). 실제 입력 provenance·보고서 재계산 일치·immutable snapshot·W&B finish 순서·비밀값 출력 방지·readback 선행 조건·Slack 중복 알림 방지를 검사했습니다. CPU 검사에서는 W&B/Slack을 stub으로 대체해 네트워크나 credential을 읽지 않았습니다. 완료 알림의 태스크2 정확도·최대 기준선 하락·3회 평균 시간/보드Wh/CUDA peak 표시도 검사했습니다.

## 실제 첫2태스크 GPU 구현 검증

양쪽 모두 GPU1·ViT-B/16·동일 IN21K→IN1K NPZ·seed1993·batch64·workers4로 실행했습니다. `seq-cifar100-224`의 전체10태스크 분할 중 첫2태스크만 학습했으며, 태스크당 TII20 + prompt50과 태스크2 TII CA30 + prompt CA30을 사용했습니다. 학습11,060 optimizer step/700,000표본, 두 태스크 통계 각10,000표본, 평가3,000표본입니다. 태스크1 CA는 N/A이며, 태스크2 두 CA는 각각300 optimizer step입니다.

Native wrapper를 실제 실행한 명령은 다음과 같습니다.

```bash
env -u MAMMOTH_TEST -u MAMMOTH_DEVICE UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 uv run --no-sync --offline python scripts/run_hide_validation.py --execute --mode validation --initial-state benchmarks/cl_efficiency/prepared/hide_prompt/author_initial_state.pt --reference-summary benchmarks/cl_efficiency/hide_prompt/runs/hide-author-validation-20261007T045302Z/result.json --result-dir benchmarks/cl_efficiency/hide_prompt/runs/native-validation-b64-20261007 > /tmp/hide-native-validation-20261007.log 2>&1
```

한 번 실행해 exit0으로 완료했습니다. [Native 결과](runs/native-validation-b64-20261007/result.json)와 [독립 저자 결과](runs/hide-author-validation-20261007T045302Z/result.json)를 exact 정답수/표본수로 대조했습니다.

| 검증 항목 | 저자·native 정답수/표본수 | 양쪽 정확도 | 저자−native 차이 | 판정 |
|---|---|---:|---:|---|
| 첫태스크 전체 | 994/1000 | 99.40% | 0.00%p | PASS |
| 태스크2 전체 관측 클래스 | 1959/2000 | 97.95% | 0.00%p | PASS |
| 태스크2 과거 클래스 | 989/1000 | 98.90% | 0.00%p | PASS |
| 태스크2 신규 클래스 | 970/1000 | 97.00% | 0.00%p | PASS |

판정 기준은 하락≤1.5%p PASS, 1.5~3.0%p 추가 점검, >3.0%p FAIL입니다. 표시 반올림 값으로 판정하지 않았습니다. 13개 구조 검사와 [최종 독립 감사](environment/full_validation_audit.json)가 모두 PASS이며, 저장된 예측에서 첫태스크1,000개+태스크2 2,000개의 top1 불일치는0개입니다.

| 계측 | 독립 저자 | Native |
|---|---:|---:|
| Recorder 실행 시간 | 2268.779s | 2260.442s |
| CUDA allocated peak | 6437.04MiB | 6486.24MiB |
| GPU 보드 에너지 | 261.670Wh | 261.660Wh |
| 에너지 sampling coverage | 99.9978% | 99.9986% |

이는 실제 검증 학습의 부가 계측이며 3회 효율성 pilot 평균과 구분합니다. 마지막 W&B artifact upload/finish는 Recorder 범위에서 제외했습니다. 에너지는 GPU 보드이며 전체 머신 에너지가 아닙니다.

저자 전체 실행의 최초 shell 호출은 복원한 기록이 없습니다. 아래는 완료된 `spec.json`에서 읽은 조건을 사용하는 **재실행 가능한 명령**이며, 최초 실행 명령의 원문으로 주장하지 않습니다.

```bash
UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline python scripts/run_hide_reference.py --execute --mode validation --reference-source /tmp/cl-hide-official-20261007 --initial-state benchmarks/cl_efficiency/prepared/hide_prompt/author_initial_state.pt --output benchmarks/cl_efficiency/hide_prompt/runs
```

저자 runner는 `--output` 아래 새로운 timestamp 디렉터리를 생성합니다. 기존 완료 디렉터리에 덮어쓰는 명령이 아닙니다.

Source/reference SHA4개와 native 실행 SHA17개가 감사 시점에도 일치했습니다. 두 branch의 pretrained canonical tensor150개 및 learnable 초기값 SHA, 데이터4분할 fingerprint, 네 학습 단계의 첫 batch input/label/seed/clipped gradient, 140epoch의 LR과 손실을 대조했습니다. 태스크2 prompt epoch 평균 scalar loss 최대 차이2.42e-8은 저자 inplace FP32 regularizer 합산과 native out-of-place float64 합산의 반올림 차이로 기록합니다. 정확도와 전체 예측은 같았습니다.

저자 source 적용 사항은 AST 컴파일로 timm global 등록을 격리하고, 단일 GPU gather/barrier, inclusive `torch.range`, KMeans 초기화, 태스크별 TII→prompt 순서, phase RNG, 클래스별 순차 통계, 별도 test-task batch 경계, seen-class class-IL logits를 명시적으로 맞춘 것입니다. Prefix reshape를 permute하지 않는 원 동작, weight decay0.0, batch/256 LR scaling과 prompt32번째 epoch의 첫 LR 감소를 보존했습니다. [Source 검토](source_review.json)와 [실제 runtime 적용 사항](runs/hide-author-validation-20261007T045302Z/runtime_adaptations.json)에 상세 기록이 있습니다.

[저자 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/eiil7wjc)와 [native W&B](https://wandb.ai/laymond1/cl-efficiency/runs/iw6ohxji)는 finished/COMMITTED artifact를 확인하고 각각13개·11개 파일의 로컬/다운로드 SHA가 일치했습니다. Readback 증거는 [저자](prepared/readbacks/author-validation/readback.json)와 [native](prepared/readbacks/native-validation/readback.json)에 있습니다. 이 검증은 실제 실행 후의 확인이며 초기 CPU 준비 이력의 ‘미실행’ 상태와 구분합니다.

## 실제 첫태스크 효율성 coordinator 실행

다음은 실제 실행한 coordinator 명령입니다. GPU/공유 경로와 W&B 설정은 approved plan·기존 런타임 환경에서 읽습니다.

```bash
UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline python scripts/run_hide_efficiency.py --acceptance-summary benchmarks/cl_efficiency/hide_prompt/runs/native-validation-b64-20261007/result.json --initial-state benchmarks/cl_efficiency/prepared/hide_prompt/author_initial_state.pt --result-dir benchmarks/cl_efficiency/hide_prompt/runs/efficiency-pilot-b64-20261007 --execute > /tmp/hide-efficiency-pilot-20261007.log 2>&1
```

검증 PASS와 source/초기값/data SHA 일치를 확인한 뒤, 공통 `ease` 전처리·batch64에서 첫태스크 TII1 + prompt1을 계측3회와 부분 FLOPs 진단1회로 분리했습니다. 각 계측은 TII79+prompt79 step, 학습10,000표본·통계 각5,000표본·평가1,000표본이며 prompt 학습 중 TII5,000표본·평가 routing1,000표본 비용을 포함합니다. Task1 CA는 N/A이고 hook overhead만 포함합니다. 기존 EASE/RanPAC/L2P/DualPrompt의1epoch와 동일 계산 예산을 뜻하지 않습니다.

실행 원본은 [execution.json](runs/efficiency-pilot-b64-20261007/execution.json), 최종 공개 집계는 [report.md](report.md)·[results.json](results.json)·[results.csv](results.csv) 및 [PNG](efficiency_comparison.png)·[PDF](efficiency_comparison.pdf)에 기록합니다. 부분 FLOPs profiler의 시간·메모리·에너지는 3회 계측 분포에서 제외합니다. Sampling coverage가 없는 구간은 외삽하지 않으며, 에너지는 GPU 보드 범위입니다.

Coordinator는 계측3회와 FLOPs1회를 모두 exit0/completed로 완료했고 summary는 PASS입니다. 실행 원본의 `values`에서 계산한 **평균±표본 표준편차(ddof=1)**는 다음과 같습니다. 동일 seed 반복이며 seed 간 성능 변동을 추정하지 않습니다.

| 지표 | 평균 ± 표준편차 |
|---|---:|
| Recorder 실행 시간 | 57.093 ± 0.152s |
| 업로드/완료 알림 포함 launcher 시간 | 67.050 ± 1.491s |
| GPU 보드 에너지 | 4.9918 ± 0.0147Wh |
| GPU 보드 에너지 | 17970.497 ± 52.783J |
| CUDA allocated peak | 6486.24 ± 0.00MiB |
| CUDA reserved peak | 7014.00 ± 0.00MiB |
| GPU device memory peak | 7505.00 ± 0.00MiB |
| 프로세스·자식 RSS peak | 15899.01 ± 45.05MiB |
| 프로세스·자식 CPU 시간 | 91.496 ± 0.224s |
| 에너지 coverage | 99.9367% ± 0.0083%p |
| 학습 throughput | 382.000 ± 0.537표본/s |
| 평가 throughput | 381.420 ± 0.351표본/s |
| 첫태스크 정확도 | 96.60 ± 0.00% |

학습 throughput의 분모는 `train_epoch` phase이며 전체 Recorder wall과 구분합니다. Native 구조13검사·TII/prompt 단계별 count·통계 및 실제 routing count·태스크1 CA의 N/A 상태·sensor 오류 없음·99% 이상 coverage를 세 번 모두 확인했습니다.

| Phase | 시간 평균 ± 표준편차 | GPU 보드 에너지 평균 |
|---|---:|---:|
| TII 학습 | 7.071 ± 0.012s | 0.6868Wh |
| Prompt 학습 | 19.108 ± 0.030s | 2.1994Wh |
| TII 통계 | 8.445 ± 0.012s | 0.8306Wh |
| Prompt 통계 | 8.367 ± 0.050s | 0.8356Wh |
| Routing 포함 평가 | 2.622 ± 0.002s | 0.2423Wh |
| Checkpoint | 1.526 ± 0.035s | 0.0760Wh |

두 ViT를 포함한 총 parameter174,498,248개, TII 학습 phase trainable2,440,036개, prompt 학습 phase trainable460,900개를 기록했습니다. 별도 profiler의 supported operator 부분 합은 **1,126,989,082,297,071 FLOPs(1126.989TFLOPs)**이며 unsupported 연산·CPU clustering·데이터 decoding·checkpoint/logging 등을 포함한 전체 방법 연산량으로 해석하지 않습니다. Phase는 inclusive라 중첩 비용을 모두 합하지 않습니다. Launcher 시간 구간에는 대응하는 별도 에너지 추정값이 없습니다.

각 worker의 W&B URL·COMMITTED artifact는 실행 원본에 보존합니다. 원격 readback에서 네 run이 finished/COMMITTED이고 계측3회 각9개·FLOPs10개 파일의 다운로드/로컬 SHA가 모두 일치했습니다. [계측1](prepared/readbacks/measurement1/readback.json)·[계측2](prepared/readbacks/measurement2/readback.json)·[계측3](prepared/readbacks/measurement3/readback.json)·[FLOPs](prepared/readbacks/flops/readback.json) 증거가 각각 PASS입니다. 최종 집계 보고서의 발행과 원격 검증은 별도 receipt를 기준으로 기록합니다.

## 최종 기존 결과 비교·보고서 검증

기존 precision 정책7개 필드는 non-null·타입·값이 모두 일치해야 비교에 포함합니다. CPU thread 메타데이터는 top-level 또는 `precision`의 실제 저장값을 읽어 정규화합니다. 양쪽에 기록된 thread 수가 다르거나 한 record 내 두 필드가 충돌하면 제외하고, 누락은 추정하지 않고 명시합니다. 따라서 EASE·L2P·DualPrompt를 재사용했고 EASE thread 수 미기록을 JSON/Markdown/그림에 표시했습니다. RanPAC은 학습 `transform_repr_sha256`가 실제로 달라 동일 입력 비교에서 제외했으며, 이유를 보고서에 표시했습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/cl-hide-efficiency-uvcache UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline pytest tests/test_report_hide_results.py tests/test_publish_hide_report.py -q --tb=short
```

36 passed, 0 failed (4.10s): report17개와 publisher19개입니다. 메타데이터 위치 차이와 실제 precision 차이를 구분하고, 필수 precision 누락·thread 불일치/충돌·EASE 누락 유지·RanPAC 전처리 차이·보고서의 재사용/제외 설명을 검사했습니다. 실제 GPU나 네트워크를 호출하지 않았습니다.

다음 명령으로 실제 저장된 완료 결과에서 공개 결과5개를 다시 생성했습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/cl-hide-report-mpl UV_CACHE_DIR=/tmp/cl-hide-efficiency-uvcache UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline python scripts/report_hide_results.py --reference benchmarks/cl_efficiency/hide_prompt/runs/hide-author-validation-20261007T045302Z/result.json --native benchmarks/cl_efficiency/hide_prompt/runs/native-validation-b64-20261007/result.json --efficiency benchmarks/cl_efficiency/hide_prompt/runs/efficiency-pilot-b64-20261007/execution.json
```

Exit0, accuracy acceptance PASS, efficiency completed입니다. PNG/PDF는 과학용 Matplotlib standalone 그림이며 EASE·L2P·DualPrompt·HiDe 네 방법을 포함하고 EASE CPU thread 미기록·RanPAC 제외를 footnote에 표시합니다. PNG를 직접 열어 레이블·단위·평균/표준편차·예산 차이·footnote를 확인했습니다. `git diff --check`도 통과했습니다.

발행 준비의 실제 입력 재계산 일치는 같은 uv Python에서 stdlib-only dry-run으로 확인했습니다.

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/cl-hide-efficiency-uvcache UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline python -S scripts/publish_hide_report.py --report-dir benchmarks/cl_efficiency/hide_prompt --output /tmp/hide-report-publication-preflight-20261007
```

Exit0, `execution=False`입니다. 이 준비 명령은 credential·네트워크·업로드·Slack 전송을 사용하지 않고 receipt 디렉터리를 생성하지 않았습니다. 이후 실제 발행의 immutable snapshot과 readback은 별도 실행 receipt로 확인합니다.

## 최종 결과 저장 확인

[W&B 비교 보고서](https://wandb.ai/laymond1/cl-efficiency/runs/o3qej29b)에 JSON·CSV·Markdown·PNG·PDF를 저장했습니다. Run finished·artifact COMMITTED와 업로드 파일6개의 로컬/다운로드 SHA 일치를 [원격 보고서 검증](environment/wandb_report_readback.json)에서 확인했습니다. [Slack 완료 알림](environment/slack_completion.json)은 sent이며, [최종 완료 기록](environment/finalization.json)에 실행 범위와 집계를 보존합니다. GPU 실행 source17개는 실행 중 commit `92d34e6`의 hash를 유지했습니다. 기준 브랜치 병합과 원격 push는 수행하지 않았습니다.

CSV는 Python CSV writer의 CRLF 행 종결자를 유지해 업로드 byte SHA와 일치시켰습니다. 기본 staged whitespace 검사는 이 CRLF를 trailing whitespace로 분류했으므로, 해당 형식만 허용한 `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check`를 실제 실행했고 exit0을 확인했습니다.
