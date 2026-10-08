# RainbowPrompt 실행 도구 검증

2026-10-08, 승인된 8회 GPU 실행에 사용할 도구의 실행 전 검증입니다. GPU 정확도나 효율성 측정 결과와 구분합니다. 현재 조건은 [run_plan.json](run_plan.json)에 기록합니다.

실행한 명령:

```bash
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline pytest -q tests/test_rainbow_components.py tests/test_rainbow_prompt.py tests/test_prepare_rainbow_prompt.py tests/test_rainbow_worker.py tests/test_rainbow_protocol.py tests/test_rainbow_validation_report.py tests/test_rainbow_validation_publication.py
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python main.py --model rainbow-prompt --dataset seq-cifar100-224 --help
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/run_rainbow_suite.py
CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu UV_CACHE_DIR=/tmp/cl-rainbow-uv uv run --no-sync --offline python scripts/run_rainbow_worker.py --prepare-initial-state results/rainbow_initial_20261008/initial_state.pt
```

통합 CPU 검사 **172개 통과**, 실제 CLI 도움말과 기본 dry-run 통과, 공통 초기 상태 준비 완료입니다. 작은 합성 데이터에서 저자 원본 `train_one_epoch`의 Unique/Rainbow 손실과 갱신된 상태가 native `meta_observe`와 동일했습니다. 실제 Mammoth `train()`을 사용하는 합성 CPU smoke는 79회 optimizer update, train 5000개, test 1000개, routing tail 40개와 checkpoint 출력 복원을 확인했습니다. Ruff는 현재 환경에 없어 실행하지 않았습니다.

공통 초기 파일은 로컬 `results/rainbow_initial_20261008/initial_state.pt`이며 SHA-256은 `b018351dde9c04416eb6987dfa7c662ac6f137c34ba33c3c398ffdfbe64916d5`입니다. 모든 학습 tensor·두 classifier head·초기 저장 cache의 catalogue SHA-256은 `47878b09051ac16fc9ecbc105fd6cb16b0b2b63bddd9dde19c6b27f2b2286baf`입니다. query/prompt 두 frozen encoder의 canonical SHA-256은 모두 기존 기준 `cd8acbba3d33807125c2e30d2d0159f90825208e63a95034ab57ef0af837a300`과 일치했습니다. 가중치 파일은 W&B artifact에 업로드하지 않습니다.

독립 코드 검토로 parser·실제 `meta_*` 호출·task-local Adam·cache의 마지막 update 전후 시점·이전 상태 보존·배치 평균 routing·실패 위치 기록을 확인했습니다. 각 epoch의 실제 입력·label·RNG·LR·샘플 순서 일치와 첫 학습 batch의 gradient 일치를 요구합니다. 이후 gradient·최종 상태·예측의 완전 동일성은 진단으로 기록하고, 정확도는 승인된 1.5/3%p 범위와 구조 검사를 함께 적용합니다.
