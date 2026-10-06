# EASE 검증 기록

최신 실제 실행과 검증은 [실행 보고서](execution_report.md)를 참고합니다. 아래 준비 단계 기록은 실행 승인 전의 이력입니다.

## 2026-10-07 실제 실행 승인 후 검증

최종 focused tests는 PyArrow 선행 import나 OpenCLIP mock 없이 **197 passed in 5.58s**입니다. CUDA를 비활성화한 테스트이며 여기서 실제 학습을 하는 기존 full suite를 실행하지는 않았습니다.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_prepare_ease.py tests/test_ease_reference_utils.py tests/test_ease_execution.py tests/test_cl_efficiency.py tests/test_cl_runtime.py -q --tb=short
```

실제 GPU1 실행 명령은 아래 두 개이며 모두 exit0입니다. 첫 명령은 batch2의 두 optimizer step만 수행하고 정확도를 평가하지 않습니다. 두 번째는 공식 source의 matched 첫 2개 태스크를 실행합니다. 포팅 코드의 full-task 학습 명령과 공통 효율성 runner에는 `--execute`를 추가하지 않았습니다.

```bash
uv run --no-sync python scripts/check_ease_runtime.py --execute
uv run --no-sync python scripts/run_ease_reference.py --execute
```

설정/manifest 갱신 후 `uv run --no-sync python -m pytest tests/test_prepare_ease.py tests/test_cl_efficiency.py -q --tb=short`도 **31 passed in 1.16s**였습니다. 이는 위 197개 중 해당 subset을 다시 확인한 값이며 총 228개로 합산하지 않습니다. `uv run --no-sync python scripts/prepare_ease.py`로 다섯 variant의 실행 없는 계획을 재생성했습니다.

W&B Public API의 finished/COMMITTED 상태와 원격 artifact metrics를 확인했고, 저장된 예측의 정확도와 task별 epoch/update/sample 수를 다시 계산했습니다. 공유 CIFAR 파일 hash·native/reference Task1 fingerprint가 같고, CPU-only `torch.load(map_location='cpu', weights_only=True)` 및 `torch.equal`로 과거 adapter의 48개 저장 텐서가 보존됨을 확인했습니다. 이 추가 확인에서는 학습이나 model image inference를 수행하지 않았습니다. 결과·관측 범위·초기 실패 시도는 [실행 보고서](execution_report.md)와 [JSON 결과](execution_results.json)에 기록합니다.

## 2026-10-07 실행 승인 전 준비 검증

모든 명령은 EASE worktree에서 실행했습니다. **학습, ViT image forward/backward, optimizer step, dataset/weight 다운로드와 새 W&B/Slack 전송은 수행하지 않았습니다.** 기존 framework import의 cached CLIP 생성/로딩 side effect는 최초 조사 중 발생했으며, 최종 검사는 factory mock으로 차단했습니다.

## 통과한 검증

| 검사 | 결과 |
|---|---|
| EASE lifecycle/parser/config, classifier, backbone/checkpoint, variant preparation, 기존 efficiency/runtime tests | **162 passed in 2.09s** |
| Native `main.py --model ease --dataset seq-cifar100-224 --help` | bootstrap/mock 조건으로 통과, initializer/trainer 호출 없음 |
| `implement-pcl-method` skill의 `check_pcl_method.py ease` | bootstrap/mock 조건으로 파일·import·ContinualModel subclass·NAME 검증 통과 |
| `python -m py_compile` 추가 모델/backbone/preparer/tests | 통과 |
| 공통 benchmark dry run | 71개 source entries, 실행 준비 66개·보류 5개, execution=false |
| EASE variants dry run | Base64·Small64·Tiny64·Small32·Tiny16의 5개, 각각 measurement/FLOPs 계획, execution=false |
| timm0.9.8 pretrained registry 조회 | 세 크기의 `augreg_in21k_ft_in1k` tag 등록 확인; weight 파일 미해결/hash null |

아래는 실제 실행한 최종 focused test 명령입니다. OpenCLIP factory patch는 기존 `seq_8vision` discovery의 가중치 로딩을 막습니다. 테스트는 metadata 모듈/meta tensors, mock 및 synthetic feature arithmetic으로 제한했습니다.

```bash
uv run --no-sync python scripts/cl_runtime.py -- python -u -c 'import pyarrow; import open_clip; from unittest.mock import patch; from torchvision.transforms import Compose; import sys; import pytest; guard=patch.object(open_clip,"create_model_and_transforms",return_value=(None,None,Compose([]))); guard.start(); sys.exit(pytest.main(["tests/test_ease.py","tests/test_ease_classifier.py","tests/test_ease_backbone.py","tests/test_prepare_ease.py","tests/test_cl_efficiency.py","tests/test_cl_runtime.py","-q","--tb=short"]))'
```

준비/문법 검증의 실제 명령은 다음과 같습니다.

```bash
uv run --no-sync python scripts/benchmark_cl.py
uv run --no-sync python scripts/prepare_ease.py
uv run --no-sync python -m py_compile models/ease.py models/ease_utils/network.py models/ease_utils/classifier.py models/ease_utils/vit.py backbone/ease_vit.py scripts/prepare_ease.py tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_prepare_ease.py
git diff --check
```

Help와 skill wiring은 위와 같은 선행 import/mock 뒤 `main()`의 `--help` SystemExit(0)를 확인하고, `runpy.run_path`로 `/mnt/sdab1/userHome/wslim/.codex/skills/implement-pcl-method/scripts/check_pcl_method.py`를 실행했습니다. Weight registry 확인은 `timm.list_models(pretrained=True)`와 `timm.models.get_pretrained_cfg(tag)`만 호출했으며, `create_model`을 호출하지 않았습니다. 등록 정보는 local `prepared/backbone_registry.json`에 남겼습니다.

## 실패와 해석의 한계

보통의 import 순서로 실행한 focused pytest와 skill wiring helper는 SIGSEGV, exit139로 종료했습니다. 조사에서는 `pyarrow.lib`가 마지막 로딩 native extension으로 관찰되었으며, `pyarrow` 선행 import를 사용하면 framework import가 완료됐습니다. 따라서 기본 native/benchmark worker import 경로가 안정화됐다고 판단하지 않습니다. 이 문제는 실제 실험 전 해결 대상입니다.

최초 bootstrap test는 129 passed/1 failed였습니다. 실패는 기존 `get_backbone_class`가 기본 호출에서 클래스를 반환하지 않는 API를 테스트가 잘못 사용한 데서 발생했습니다. 프레임워크는 수정하지 않고 `return_args=True`로 검사하도록 고쳤으며, 이후 새 검증을 포함한 최종 162개가 통과했습니다.

Full suite 및 기존 model/config/register/dynamic playbook의 학습을 수행하는 테스트, 실제 GPU smoke training은 사용자의 실험 보류 요청에 따라 실행하지 않았습니다. 현재 통과는 구현의 구조/연결 검증이며 정확도·성능 재현·효율성의 검증이 아닙니다. 허용 accuracy 차이와 checkpoint SHA256은 아직 null이며, 실제 두 태스크 검증 통과 전 병합하지 않습니다.
