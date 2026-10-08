# RainbowPrompt 구현·CPU 검증

공식 commit `0b84d3e3738669fe4bb516ccc1c526a535576ad3`의 활성 경로를 Mammoth에 연결했습니다. **학습 가능한 gate·sparsity loss는 미포함**이며 사용자가 선택한 범위입니다. 태스크 종료 시 prompt를 다시 계산하지 않고 원본의 optimizer 이전 cache를 보존합니다. 실제 CIFAR 학습·GPU 연산·W&B 업로드·Slack 알림은 수행하지 않았습니다.

CPU 집중 검사는 **78개 통과**입니다. 공식 checkout을 SHA로 검증한 14개 oracle 검사는 초기 파라미터 이름·값, Unique/Rainbow 출력과 모든 gradient, ViT 출력, 독립 K/V prefix와 cached prompt가 원본과 bitexact로 같음을 확인했습니다. 나머지 검사는 배치 평균 task routing·미래 태스크 차단·warmup cache·frozen encoder·현재 클래스 CE와 similarity loss·부분 라벨 wrapper·태스크별 Adam 재생성·backward 설정·상태 저장/복원·parser/config/등록·준비안 범위를 확인합니다.

Base·Small·Tiny는 CPU/meta 구조와 factory 인자를 확인했습니다. Base의 기존 공유 NPZ를 실제 CPU 메모리로 읽어 두 encoder의 canonical 150개 텐서가 공통 pin과 같은 것을 확인했습니다. [가중치 검사 JSON](cpu_weights.json)에 SHA와 구조 수치를 기록했습니다. 학습 가능한 파라미터는 6,521,476개, frozen 파라미터는 171,674,212개, prompt cache는 7,372,800bytes입니다. 이 수치는 **구조·가중치 검사 결과**이며 GPU peak 메모리·연산량·시간·에너지 측정값은 아닙니다.

실행한 검증 명령은 다음과 같습니다. CPU 명령에는 `CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv`를 적용했습니다.

```bash
uv run --no-sync --offline pytest -q tests/test_rainbow_components.py tests/test_rainbow_prompt.py tests/test_prepare_rainbow_prompt.py tests/test_module_register.py::test_args_with_choices tests/test_module_register.py::test_register_backbone tests/test_module_register.py::test_register_dataset
uv run --no-sync --offline python scripts/rainbow_cpu_preflight.py
uv run --no-sync --offline python main.py --model rainbow-prompt --dataset seq-cifar100-224 --help
uv run --no-sync --offline python main.py --model sgd --dataset seq-cifar10 --help
uv run --no-sync --offline python -S scripts/prepare_rainbow_prompt.py --check-local
JE_ARROW_MALLOC_CONF=background_thread:false uv run --no-sync --offline python /mnt/sdab1/userHome/wslim/.codex/skills/implement-pcl-method/scripts/check_pcl_method.py rainbow-prompt
```

새 preparer는 표준 라이브러리만 사용하며 실행 옵션이 없습니다. 로컬 [준비 manifest](../prepared/rainbow_prompt/plan.json)는 `executed=false`·`training_authorized=false`와 미정인 pilot 선택을 기록합니다. 독립 저자 GPU runner와 계측·W&B·Slack 연결은 후속 실행 단계에서 구현해야 합니다.

일반 test_import/model_config/basic_functionality 파일에는 실제 데이터 학습이 들어 있어 이번 승인 범위에서 실행하지 않았습니다. 대신 새 모델의 import·등록·CLI precedence·synthetic wrapper smoke를 실행했습니다. 공식 checkout이 없는 환경에서는 oracle 14개만 skip되고 나머지 core 검사는 실행됩니다. `ruff`는 현재 uv 환경에 없어 실행하지 못했습니다. 직접 Python import 두 번은 기존 PyArrow allocator 문제로 종료됐으며, 기존 `configure_native_environment()` bootstrap을 적용한 뒤 위의 가중치·skill 검사가 통과했습니다.

네트워크와 전체 모델의 CPU state round-trip을 확인했지만 generic `main.py --start_from/--loadcheck/--inference_only`로 데이터 스트림을 재개하는 경로는 이번 단계에서 지원하지 않습니다. 미학습 태스크를 skip하여 zero cache를 승인하는 상황을 피하도록 명시적으로 거부합니다.

향후 실제 데이터 검증은 seed·NPZ뿐 아니라 prompt/head의 **전체 초기 상태**, 데이터·transform·batch 순서·단계별 RNG를 같은 조건으로 맞춰야 합니다. CPU 수치 일치를 CIFAR 정확도 검증이나 논문 성능 재현으로 해석하지 않습니다. 기본 20epochs/warmup5는 유지했으며 `n_epochs <= warmup` 실행은 생성 시 거부합니다. GPU pilot의 1epoch/warmup0·6epochs/warmup5·보류 선택은 [검증·실행 계획](README.md)에 남겨두었습니다.
