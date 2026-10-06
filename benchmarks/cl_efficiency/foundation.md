# 공통 CL 환경 준비 — 2026-10-06

이번 단계는 알고리즘 구현 전에 환경·계측·저장·알림의 공통 기반을 준비하는 단계입니다. **EASE 구현과 CL 학습·추론은 수행하지 않았습니다.** 연결 검사의 synthetic 값은 알고리즘 성능이나 효율성 측정 결과가 아닙니다.

## 확정한 조건

- 첫 구현은 EASE. HiDe-Prompt 등은 독립적인 후속 방법 브랜치에서 검토합니다.
- 구현 검증과 논문 성능 재현을 분리합니다. 공식 코드와 데이터·클래스 순서·전처리·backbone·checkpoint·학습 조건을 맞춘 검증에서 비정상적으로 낮은 성능이 나오면 구현 실패로 판단하고 병합을 보류합니다. 정상/비정상의 수치 기준은 EASE 구현 명세 단계에서 정합니다. 1 epoch pilot을 논문의 최종 정확도와 직접 비교하지 않습니다.
- 모든 방법의 공통 효율성 pilot은 `seq-cifar100-224`, 첫 태스크 10 classes, 1 outer epoch, batch 64, seed 1993, workers 4, 클래스 재배열 없음입니다. 자체 task hook의 내부 학습 비용은 유지합니다.
- BudgetCL/aL-SAR 계산 예산 비교와 controller 구현은 후속 단계입니다.
- 물리 GPU 1, entity `laymond1`, project `cl-efficiency`를 사용합니다. 프로세스 내부에서는 `CUDA_VISIBLE_DEVICES=1`이므로 장치 이름이 `cuda:0`입니다.

## 모든 worktree에서 재사용할 경로

공개 프로필 [runtime.json](runtime.json)을 공통 기반 커밋에 포함했습니다. `scripts/cl_runtime.py`와 benchmark runner가 읽으며, 비밀은 프로필에 넣지 않습니다.

| 용도 | 공유 위치 / 적용 방법 |
|---|---|
| 데이터 | `/mnt/sdab1/share/data`, Mammoth `--base_path` |
| 공통 캐시 | `/mnt/sdab1/userHome/wslim/.cache`, `CL_CACHE_ROOT`와 `XDG_CACHE_HOME` |
| PyTorch/timm URL 가중치 | `/mnt/sdab1/userHome/wslim/.cache/torch/hub/checkpoints`, `TORCH_HOME`은 상위 `.../.cache/torch` |
| Hugging Face Hub | `/mnt/sdab1/userHome/wslim/.cache/huggingface/hub`, `HF_HOME`·`HF_HUB_CACHE` |
| HF datasets 캐시 | `/mnt/sdab1/userHome/wslim/.cache/huggingface/datasets`, `HF_DATASETS_CACHE` |
| OpenAI CLIP | `/mnt/sdab1/userHome/wslim/.cache/clip`, 라이브러리의 기존 `~/.cache/clip` 기본값 |
| 패키지 다운로드 | 기존 사용자 uv 캐시 재사용 |

5개 데이터셋의 OpenCLIP 호출에서 worktree 내부 `checkpoints/ViT-B-16/cachedir/open_clip` 지정만 제거했습니다. 라이브러리가 HF 다운로드는 `HF_HUB_CACHE`, URL 다운로드는 `~/.cache/clip`을 사용합니다. 데이터 분할·전처리·CL 알고리즘 동작은 바꾸지 않았습니다. SUN397처럼 데이터 위치를 따로 지정하는 다운로드는 공유 데이터 루트 아래에 유지됩니다.

`.venv`는 worktree별로 생성합니다. 서로 다른 checkout의 editable package와 의존성이 충돌하지 않도록 가상환경 자체는 공유하지 않습니다. 실행 로그·측정 결과는 각 run의 출력 경로에 저장합니다. native checkpoint 경로는 worktree의 기본 `./checkpoints`이며, native `main.py`에서는 `--checkpoint_path`로 별도 지정할 수 있습니다.

새 알고리즘 worktree는 `codex/cl-foundation`의 커밋을 기반으로 만듭니다. 기존 `master` worktree에 이번 변경을 강제로 적용하지 않았습니다. 기반을 반영한 worktree에서는 다음 명령을 사용합니다.

```bash
uv sync --locked --group dev --extra extra
uv run --no-sync python scripts/cl_runtime.py -- python main.py --model sgd --dataset seq-cifar100-224 --help
uv run --no-sync python scripts/benchmark_cl.py
```

마지막 명령은 준비 파일만 만들며 실험·W&B·Slack 전송을 수행하지 않습니다. `--execute`는 실험 실행을 별도로 결정한 뒤에만 사용합니다. `uv run --no-sync`는 설치된 extra 의존성을 실행 중 제거하지 않습니다.

기존 worktree에서도 공통 wrapper의 절대 경로로 환경변수와 Mammoth 데이터 경로를 적용할 수 있습니다. 실행 대상 worktree의 Python 환경은 별도로 준비해야 합니다. 기존 checkout의 명시적 OpenCLIP 캐시 경로를 제거하는 수정은 공통 기반을 반영한 뒤 적용됩니다. 모든 실험 worktree를 이 기반에서 생성하거나 기반 커밋을 반영해 사용합니다.

```bash
uv run --no-sync python /mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/scripts/cl_runtime.py -- python main.py --model sgd --dataset seq-cifar100-224 --help
```

## 비공개 W&B·Slack 설정

`wandb.conf`와 W&B SDK의 로컬 디렉터리는 Git에서 제외합니다. loader는 JSON 또는 단순 `key: value`/`key=value` 설정을 읽고, shell/eval로 실행하지 않습니다. 환경변수에 이미 있는 비밀값이 우선합니다. 결과 파일에는 설정값 대신 연결 상태만 기록합니다.

같은 Git 저장소의 worktree가 원본 개인 설정을 참조하도록 공통 local Git 설정에 **파일 경로만** 등록했습니다.

```bash
git config --local cl.credentialsFile /mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/wandb.conf
```

비공개 파일 선택 순서는 `--credentials`, `CL_CREDENTIALS_FILE`, 위 Git 설정, 현재 worktree의 `wandb.conf`입니다. 원본 aa40 worktree를 삭제하기 전에는 개인 설정을 다른 비공개 위치로 옮겨 이 경로를 갱신해야 합니다. 설정 파일을 Git에 넣거나 worktree마다 복사하지 않습니다.

## 설치 및 검증 결과

Python 3.12.2, uv 0.12.23, PyTorch **2.13.0+cu126**, torchvision **0.28.0+cu126**, timm 0.9.8, W&B SDK 0.29.0과 extra/dev 의존성을 설치하고 `uv.lock`을 보관했습니다. 호스트 드라이버 570.133.07에 맞춰 CUDA 12.6 wheel을 선택했습니다. CUDA 13은 드라이버 580 이상을 요구합니다. [NVIDIA 호환성 표](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [PyTorch 공식 wheel 조합](https://pytorch.org/get-started/previous-versions/), [uv PyTorch index 설정](https://docs.astral.sh/uv/guides/integration/pytorch/).

GPU 1의 NVIDIA 센서와 CUDA 초기화/장치 속성 조회가 통과했습니다. RTX 4090, 24 GiB, compute capability 8.9이며 이 검사에서는 tensor 연산을 실행하지 않았습니다. 공통 경로 5개 모두 존재·읽기/쓰기 접근을 확인했습니다. 이 결과는 모델 backward·학습 수치 안정성까지 검증한 결과가 아닙니다.

W&B SDK 0.29에서 제거된 `run.get_url()`을 active native training 경로와 계측 worker에서 `run.url`로 교체했습니다. [W&B Run API](https://docs.wandb.ai/ref/python/experiments/run/). 실제 모델 학습은 보류하고 disabled SDK 초기화 테스트로 호환성을 확인했습니다.

검사 환경에서 torch/torchvision/timm을 먼저 import한 뒤 open_clip을 import하면 SIGSEGV가 발생했습니다. open_clip을 먼저 import하면 전체 의존성 검사와 CUDA 초기화가 통과했습니다. 환경 검사에 이 순서를 적용했습니다. native Mammoth help도 통과했으나 임의 import 순서의 호환성까지 보장하지 않습니다.

W&B 연결은 [온라인 테스트 run](https://wandb.ai/laymond1/cl-efficiency/runs/025b4482e284)을 종료한 후 서버 summary 재조회까지 통과했습니다. config와 artifact에는 `connectivity/test=1`, 학습·추론·실험 미실행 표식, 검사 종류와 schema만 포함합니다. 환경 상세·로컬 경로·GPU·패키지 정보는 업로드하지 않았습니다. Slack은 연결 확인 메시지 **1회 전송 성공**했고 W&B 수정 후 중복 메시지를 보내지 않았습니다.

- [환경 검사 및 Slack 전달 결과](environment_checks/20261006-foundation/report.json): 환경 passed, Slack sent, W&B 초기 설정 오류 ValidationError.
- [수정 후 W&B 검사 결과](environment_checks/20261006-wandb/report.json): 환경 passed, W&B passed, artifact_uploaded와 server_summary_verified 모두 true.
- [W&B에 보낸 최소 artifact 내용](environment_checks/20261006-wandb/connectivity_report.json).

처음의 상세 환경 artifact 업로드와 링크 포함 Slack 메시지는 자동 승인 검토가 거절했습니다. 외부 payload를 위의 합성 표식으로 축소해 다시 검토받고 실행했습니다. 인증 정보·개인 설정·SDK 로그를 업로드하지 않았습니다.

실제로 실행한 주요 검증 명령은 아래와 같습니다.

```bash
uv sync --group dev --extra extra --python /mnt/sdab1/userHome/wslim/miniconda/bin/python
uv run --no-sync python scripts/cl_runtime.py -- python -m pytest tests/test_cl_efficiency.py tests/test_cl_runtime.py -q --tb=short
uv run --no-sync python scripts/cl_runtime.py -- python -m pytest tests/test_cl_environment.py -q --tb=short
uv run --no-sync python scripts/cl_runtime.py -- python main.py --model sgd --dataset seq-cifar100-224 --help
uv run --no-sync python scripts/benchmark_cl.py
uv run --no-sync python scripts/check_cl_environment.py --check-wandb --check-slack --output benchmarks/cl_efficiency/environment_checks/20261006-foundation
uv run --no-sync python scripts/check_cl_environment.py --check-wandb --output benchmarks/cl_efficiency/environment_checks/20261006-wandb
```

도구/runtime 테스트 **61 passed**, 환경/연결 mock 테스트 **12 passed**. native help·dry run이 통과했고 dry run은 목록 70개, 준비 65개, 보류 5개, `experiment_executed=false`입니다. 일부 sandbox 실행은 공유 캐시·GPU 장치 접근 제한으로 실패해 허용된 호스트 실행으로 검증했습니다. Mammoth의 import 과정에서 기존 CLIP model 생성/weight 로딩이 발생하므로 help를 순수 AST 검사로 해석하지 않습니다. forward/학습 step은 수행하지 않았습니다.

변경 Python 파일의 `py_compile`, 문서 링크 및 계획 설정 정적 검사도 통과했습니다. 개인 설정의 실제 비밀값을 메모리 안에서 대조하여 저장 후보 26개 파일에 포함되지 않았음을 확인했습니다. 비밀값은 출력하지 않았습니다.

초기 등록 테스트의 `-k` 이름 필터는 유사 이름의 실행 테스트까지 선택한 실수였습니다. help 3개는 통과했으나 추가된 4개는 `NoopRun.get_url` 오류로 data loader·학습·평가 전에 종료했습니다. 정확한 ID로 재실행하려던 명령도 자동 승인 검토가 거절해 제외했습니다. 대신 위의 학습 없는 도구 검사와 CLI help로 검증했습니다. 기존 모델·dataset 학습 테스트 전체를 통과했다고 표시하지 않습니다.

다음 단계는 EASE 구현 명세 확정입니다. backbone/checkpoint, 태스크 분할, 공식 코드 기준 버전, 정상 성능 판정 기준을 정한 뒤 독립적인 `codex/ease` worktree에서 구현합니다.
