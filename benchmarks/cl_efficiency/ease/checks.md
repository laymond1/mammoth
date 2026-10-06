# EASE 검증 기록

최신 효율성 실행은 [Pilot 보고서](efficiency_pilot_report.md), 구현 검증은 [Native 검증 보고서](native_validation_report.md)를 참고합니다. [기존 실행 보고서](execution_report.md)는 smoke와 공식 기준선의 이력입니다. 아래 이전 단계 기록은 각 단계 당시의 승인 범위와 결과를 보존합니다.

## 2026-10-07 효율성 pilot와 로컬 병합

사용자가 승인한 EASE ViT-B/16·adapter64만 GPU1에서 실행했습니다. `seq-cifar100-224`의 첫 태스크, 1 epoch, batch64, workers4, seed1993으로 새 프로세스 측정 3회와 별도 부분 FLOPs 진단 1회를 완료했습니다. 네 실행 모두 exit0·completed, 정확도965/1000=96.50%, 여덟 구조 검사 통과입니다. 전체 관측 시간 중앙값29.7629초·CUDA peak allocated5760.5054MiB·GPU 보드 에너지 중앙값2.4432Wh입니다. 진단의 지원 연산 합계562696180049887 FLOPs이며, 이 실행의 시간·메모리·에너지는 측정3회의 통계에 포함하지 않습니다.

실행 전 EASE worktree에서 아래 focused suite는 **295 passed in5.67s**였습니다. 실제 네 실행은 추가 smoke 역할도 수행했습니다.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest tests/test_ease_efficiency.py tests/test_ease_acceptance.py tests/test_ease_execution.py tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_prepare_ease.py tests/test_cl_efficiency.py tests/test_cl_runtime.py -q --tb=short
uv run --no-sync python scripts/run_ease_efficiency.py --execute
python3 -m py_compile scripts/benchmark_cl.py scripts/run_ease_efficiency.py tests/test_cl_efficiency.py tests/test_ease_efficiency.py
git diff --check
```

[독립 감사](environment/efficiency_independent_audit.json)에서 raw 데이터/source/사전학습 가중치 hash, frozen PTM hash·finite 근거,79updates·5000train·5000prototype·1000eval, NPZ 정확도와 모든 phase/전체 telemetry 에너지 적분을 확인했습니다. 세 측정만으로 재계산한 median/min/max도 동일합니다. 이번 pilot은 full .pt를 저장하지 않았으므로 저장 tensor 직접 재검사 대신 실행 중 검사와 hash 증거를 따릅니다. 같은 seed의 반복 비용 측정이며20-epoch 정확도 허용폭, forgetting이나 seed간 변동을 평가하지 않습니다.

[W&B 네 실행 확인](environment/wandb_efficiency_readback.json)은 finished/COMMITTED 상태, 원격 metrics 일치와 spec·NPZ·telemetry·phase파일의 정확한 SHA256 일치를 기록합니다. 모든 suite의 Slack 요청 상태는 sent입니다. [집계 W&B](https://wandb.ai/laymond1/cl-efficiency/runs/q61iouic)는 기존 결과만 저장한 report-only 기록이며 추가 실험이 아닙니다. [집계 서버 확인](environment/wandb_efficiency_summary_readback.json)에서 artifact5개파일의 정확한 hash 일치를 확인했습니다. 첫 dashboard 검사는 float에 완전 일치를 요구해 실패했으나, 실제 차이는 최대5.684e-14였습니다. API 재조회에서 float에만 relative/absolute1e-12를 적용해 모두 통과했고 artifact검증은 정확한 bytes 비교를 유지했습니다.

모델/worker/orchestrator source를 측정 도중 변경하지 않았습니다. 실행 commit49897a3과 source hashes는 원본 로그와 [집계 JSON](efficiency_pilot_results.json)에 보존했습니다. 완료·저장·독립 감사 확인 후 깨끗한 `codex/cl-foundation`의607703b를49897a3으로 fast-forward 병합했습니다. 집계 JSON의 report-commit pending 값은 업로드 시점의 이력이며, 결과 문서는 이후 별도 커밋으로 반영합니다. 원격 push는 하지 않습니다.

Foundation worktree의 이전 Miniconda Python3.12.2 환경을 아래 명령으로 검증한 managed Python3.12.15·lock 의존성에 맞췄습니다. 공유 캐시의 기존 파일로 오프라인 동기화했으며 데이터/가중치 다운로드나 추가 학습은 하지 않았습니다.

```bash
git merge --ff-only codex/ease
XDG_CACHE_HOME=/mnt/sdab1/userHome/wslim/.cache uv sync --locked --offline --python /mnt/sdab1/userHome/wslim/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 --group dev --extra extra
```

## 2026-10-07 Native 첫 2개 태스크 검증 완료

사용자가 추천안 진행을 승인하여 seed1993의 Base64 첫 2개 태스크만 실행했습니다. 정확도 네 지표와 일곱 구조 검사가 모두 통과했습니다. 추가 seeds, 공통 효율성 pilot, FLOPs pass, Tiny/Small 학습, 병합과 push는 수행하지 않았습니다.

실행 전 focused tests는 **239 passed in 4.89s**입니다. 정확도 경계 판정, 필수 구조 검사, 실행 없는 기본 동작과 scope guard, 기존 EASE·효율성·환경 계약을 확인했습니다. `test_ease_acceptance.py`의 60개는 이 239개에 포함되므로 합산하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest tests/test_ease_acceptance.py tests/test_ease_execution.py tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_cl_efficiency.py tests/test_cl_runtime.py -q --tb=short
uv run --no-sync python scripts/run_ease_validation.py --execute
```

실제 native 실행은 exit0, `status=completed`, `acceptance_status=pass`, `wandb_logging_status=uploaded`입니다. 기존 Mammoth trainer와 meta wrappers를 사용하여 40 epochs·4200 updates·200000 학습 presentations·3000 평가 presentations·15000 prototype adapter-image pairs를 완료했습니다. Task1은99.40%, Task2는 전체97.30%·과거96.60%·신규98.00%입니다.

학습 종료 후 예측 NPZ를 재집계하고 네 raw data fingerprints를 공식 결과와 비교했습니다. CPU-only `torch.load(map_location='cpu', weights_only=True)`로 모든 저장 tensor의 유한성, frozen backbone150개 tensor와 이전 adapter48개 tensor의 `torch.equal`, prototype/lifecycle metadata를 확인했습니다. Telemetry를 독립 적분한 GPU 에너지도 Recorder 값과 일치합니다. [독립 감사 기록](environment/native_independent_audit.json).

W&B Public API로 `laymond1/cl-efficiency/n3nlg2ah`의 `finished`, artifact `COMMITTED`를 조회했습니다. `artifact.get_entry(...).download(...)`로 metrics와 실행 spec를 내려받아 로컬과 비교했고 정확도·epoch·자원·데이터·동결 audit·판정·정밀도 및 spec가 모두 일치했습니다. [서버 조회 기록](environment/wandb_native_readback.json). Slack 시작·두 태스크 완료·종료 상태도 모두 `sent`입니다. 이 확인은 추가 학습이나 image inference를 수행하지 않았습니다.

실행 source8개 파일과 공식 기준선 JSON의 SHA256을 다시 확인했습니다. 실행한 runner/helper의 bytes는 변경하지 않았으며 당시 Git HEAD와 별도 SHA256을 [결과 JSON](native_validation_results.json)에 보존했습니다. 아래 이전 단계에 있는 native 미실행·승인 false·정확도 미검증은 당시 상태입니다.

최종 manifest·상태 갱신 후 아래 subset을 다시 실행하여 **44 passed in 1.09s**를 확인했습니다. 기존 239개와 중복되므로 합산하지 않습니다. 준비 명령은 공통71개·EASE5개 variant를 `execution=False`로 재생성했습니다.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest tests/test_ease_execution.py tests/test_prepare_ease.py tests/test_cl_efficiency.py -q --tb=short
uv run --no-sync python scripts/benchmark_cl.py
uv run --no-sync python scripts/prepare_ease.py
python3 -m py_compile scripts/run_ease_validation.py scripts/ease_acceptance.py tests/test_ease_acceptance.py tests/test_ease_execution.py
git diff --check
```

새 JSON·CSV·계획과 manifest의 일관성, Markdown 로컬 링크, 생성한 효율성 계획의 실행 false·1-task/1-epoch 설정도 정적으로 확인했습니다.

## 2026-10-07 정확도 기준 확정

사용자가 1.5%p 이내 통과, 1.5%p 초과~3.0%p 이내 추가 점검, 3.0%p 초과 실패로 확정했습니다. `validation_plan.json`에 네 지표별 공식 정답 수·샘플 수와 통과/추가 점검 하한, 판정 우선순위, 구조 실패 조건을 기록했습니다. 기존 실행 결과 파일과 W&B 기록은 당시의 reference 측정 이력이며 변경하지 않았습니다. 새 학습·추론·외부 run·Slack 알림은 수행하지 않았습니다.

```bash
python3 -m json.tool benchmarks/cl_efficiency/ease/validation_plan.json > /dev/null
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest tests/test_prepare_ease.py -q --tb=short
git diff --check
```

JSON syntax와 Decimal 기반 네 지표의 1.5/3.0%p 하한 계산, reference 정답 수 일치, 학습 승인 false 보존을 확인했습니다. 기존 dry-run tests는 **2 passed**입니다. 새로운 테스트 파일은 추가하지 않았습니다.

저장소의 Sphinx 빌드 명령 `uv run --no-sync sphinx-build -j auto . ../_build`를 `docs/`에서 실행했으나 기존 환경에 `sphinx-build`가 없어 exit2로 실패했습니다. 변경한 benchmark Markdown의 로컬 링크·표·계획 파일은 정적으로 확인했습니다.

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
