# PROL CPU 구현 검증 — 2026-10-08

최종 통합 검사 **90 passed in 6.71s**를 확인했습니다. `codex/prol`에서 공식 활성 코드의 Base 이식·독립 CPU 수치 대조·GPU6회 제안을 준비했습니다. 이 시간은 pytest 전체 소요 시간이며 학습 효율성 측정값이 아닙니다. GPU 계산, 실제 CIFAR training/inference, 데이터·가중치 다운로드, credentials 읽기, W&B·Slack 쓰기는 수행하지 않았습니다.

## 확인 범위

| 검사 | 결과 |
|---|---|
| 독립 source oracle 28개 | 고정한 source AST의 실제 LPrompt·ViT·classifier·train_one_epoch_with_aux·task prelude·freeze·LR scaling 실행. native math를 oracle로 재사용하지 않음 |
| Native wrapper 37개 | discovery·config·CLI precedence·protocol reject·meta_*·동결·task lifecycle·checkpoint·parameter accounting·실제 trainer 검증 |
| 준비 도구 21개 | 6개 job, 측정3회/FLOPs1회 분리, 공통 설정·source pins 검증. --execute 없음, import 시 ML·서비스 side effect 없음, 실행/조건 변경 거절 |
| 기존 안전한 등록·import 검사 4개 | synthetic dummy method/dataset의 import 후 actual train 및 backbone/dataset help·동적 등록 계약 검증 |
| Base NPZ CPU preflight | native 및 source의 독립 NPZ loader 모두 canonical encoder150개×2 동일. **전체 초기 tensor state·trainability·CPU RNG bitexact** |
| Skill wiring helper·CLI help | Python3.12 uv 환경에서 prol-prompt 연결·발견·parser 확인 |

독립 대조에서는 두 태스크·각3개의 합성 CPU 업데이트를 사용해 nominal LR0.01/0.05와 threshold0.3/100 조합 및 첫 batch에서만 trigger되는 sticky 전환을 확인했습니다. 각 업데이트의 loss, clipped gradient, Adam state, learned tensor, 각 optimizer group의 LR, inner/outer cosine, RNG가 source/native에서 bitexact입니다. 교정한 CE value를 사용하는 sticky probe는 test 전용이며 기본 recipe·source engine은 변경하지 않습니다.

직접 prompt 검사에는 asymmetric K/V, old key gradient, 현재 태스크 local scaler index, batch 간 similarity assignment, column0 미선택 시 zero generator, 평가 중 future affine row까지 clamp, raw adjacent-layer prompt sum, 수학적으로 0이 되는 NonD 항을 포함합니다. 첫 태스크 이후 generator와 이전 head의 동결을 확인했습니다. 전체 query 및 non-norm encoder는 동결하고, source가 학습하는 block/final LayerNorm은 유지합니다.

실제 `utils.training.train` smoke는 **8×8 합성 입력**, 20classes·2tasks×10classes, task당 train5000/test1000, batch64·1epoch로 실행했습니다. `store_masked_loaders`, `MammothDatasetWrapper` 및 실제 `meta_*` 호출을 유지하며 **158 optimizer updates·train10000·eval3000**을 처리했습니다. 마지막 training batch8 두 번과 evaluation batch40 세 번을 확인했습니다. 합성 smoke의 정확도는 실제 CL 성능 근거로 사용하지 않습니다.

Checkpoint 검사에서 unused query head의 동결 해제·승인하지 않은 query head 증가·초기 checkpoint 복원 시 과거 counter 잔존을 발견해 수정했습니다. Dynamic head/task/freeze 복원은 RNG를 진행시키지 않고, 조건·counter·future state가 불일치하면 거절합니다. Epoch 동안 유지하는 source의 unused key deepcopy는 메모리 비용을 보존하되 **추가 model parameter로 등록하지 않아** parameter telemetry·checkpoint에 가짜 key가 생기지 않도록 확인했습니다. Framework stream resume/loadcheck·중간 epoch 재개는 미지원입니다.

## 실제 실행한 명령

최종 통합 검사:

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline pytest -q tests/test_prol_components.py tests/test_prol_prompt.py tests/test_prepare_prol_prompt.py tests/test_import.py::test_import_after_register tests/test_module_register.py::test_args_with_choices tests/test_module_register.py::test_register_backbone tests/test_module_register.py::test_register_dataset
```

공통 Base 가중치의 native/source 독립 CPU 검증과 로컬 준비:

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/prol_cpu_preflight.py > /tmp/prol-cpu-preflight.log 2>&1
python3 scripts/prepare_prol_prompt.py --check-local
```

Skill wiring helper:

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu JE_ARROW_MALLOC_CONF=background_thread:false UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python /mnt/sdab1/userHome/wslim/.codex/skills/implement-pcl-method/scripts/check_pcl_method.py prol-prompt
```

PROL CLI help는 native allocator 설정 후 `sys.argv=['main.py','--model','prol-prompt','--dataset','seq-cifar100-224','--help']`로 `main.main()`을 실행하여 exit0을 확인했습니다. 새 Python15파일은 `ast.parse`로 검사하고 `git diff --check`를 통과했습니다. 시스템 Python3.8로 helper를 실행한 초기 시도와 allocator 초기 설정을 빠뜨린 uv helper는 실패했으며, 위 Python3.12·allocator 설정으로 바로잡았습니다. 처음 Base preflight에서 model discovery 전에 seed를 설정한 import-order 문제도 해결하여, 모든 ML bootstrap을 마친 뒤 동일 seed로 독립 초기 상태를 대조했습니다.

기존 test_import/model_config/basic_functionality 전체 파일에는 실제 dataset training이나 다운로드가 포함됩니다. 이번 미실험 범위를 지키기 위해 source-specific parser/config/discovery 검사와 위 synthetic smoke·안전한 네 등록 검사를 사용했습니다. 해당 전체 파일이나 전체 pytest가 통과했다고 주장하지 않습니다.

## 준비 이후 남은 작업

GPU recipe와 비정상 성능 하한을 검토한 뒤 instrumented independent author/native GPU worker·suite·report·publication을 구현하고 CPU 검사해야 합니다. 첫2task 대조가 통과한 후에만 효율성3회와 별도FLOPs1회를 진행하도록 준비했습니다. 공통 FP32·phase RNG를 유지해도 source의 duplicate advanced-index CUDA assignment는 GPU 재현성을 실제 점검해야 합니다. 이번 CPU bitexact 결과가 GPU bitexact 결과를 보장하지 않습니다.

첫 task≥90%, task2 seen≥20% collapse guard는 기존 Base 조건을 참고한 **미확정 제안**입니다. Source도 online1epoch·공통 batch64/가중치로는 아직 실행하지 않아 source/native 일치만으로 낮은 성능을 승인하지 않습니다. 학습 시간·메모리 peak·FLOPs·GPU board energy의 실측값은 아직 없습니다. [고정 조건](validation_plan.json), [실행 제안](execution_plan.json), [공식 동작의 특징과 한계](README.md)를 함께 검토합니다.
