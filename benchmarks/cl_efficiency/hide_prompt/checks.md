# HiDe-Prompt Base 구현 준비 검증 — 2026-10-07

이번 완료 범위는 `codex/hide-prompt`의 native 구현, CPU 합성 검증, 실행 계획 작성입니다. 실제 CIFAR 학습·성능 acceptance·효율성 측정은 다음 단계입니다. 고정한 저자 commit은 `0ed433334ff2cfb206c5e2899651c3044ec85f92`이며 원본 저장소는 변경하지 않았습니다.

## 실행한 명령과 결과

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

[실제 Base CPU 구성](environment/native_cpu_initialization.json)에서는 두 branch가 모두 기존 NPZ의 canonical pretrained tensor150개 SHA와 일치합니다. 이미지 forward와 Base optimizer step은0입니다. KMeans 초기화/seed, 클래스별 통계 batching/순서, retained prefix layout은 [소스 검토](source_review.json)와 [계획](validation_plan.json)에 명시합니다. 미래 독립 기준선은 learnable MLP/head/prompt 초기화도 맞춰야 합니다.

[Preflight JSON](environment/preflight.json)에 native source SHA·각 검증 명령·공유 환경·기존 checkpoint hash를 보존합니다. 준비 도구는 학습/ML import/네트워크/credential/센서를 사용하지 않으며, 생성한 `executed`와 학습 권한은 모두false입니다.

## 다음 단계

독립 저자 기준선 runner와 같은 초기화·phase RNG를 준비한 뒤, 첫2태스크 TII20+prompt50 및 태스크2의 두 CA30 실행을 확정해야 합니다. 구현 acceptance를 통과하면 TII1+prompt1 첫태스크 효율성 pilot 후보를 검토합니다. Native 이미지 학습 smoke, W&B/Slack 실험 기록, 실제 정확도 판정, 계측, 기준 브랜치 병합은 이번에 수행한 것으로 표기하지 않습니다.
