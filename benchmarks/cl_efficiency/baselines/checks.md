# Baseline pilot 검증 기록

알고리즘별 독립 worktree에서 checkpoint 옵션과 관련 테스트를 구현하고 `codex/cl-baseline-pilot`에 cherry-pick했습니다. 기본 경로는 유지합니다. RanPAC은 module 등록과 device 동기화를 수정하여 파라미터·buffer 계측이 가능하도록 했고 G/Q CPU solve는 그대로 유지합니다.

CPU checkpoint focused tests는 L2P11개·DualPrompt14개·RanPAC14개를 각각 통과했습니다. 실제 동일 NPZ를 로드하여 각 backbone의 canonical150tensor가 EASE 참조와 정확히 같음을 확인했습니다. 이미지 forward/backward·GPU·학습·다운로드·외부 로깅을 실행하지 않은 준비 검사입니다. checkpoint 감사 파일에 근거를 보존합니다.

통합 source의 checkpoint/반복측정/기존효율성/환경/EASE 관련 subset은239passed in9.65s, 합성 데이터 import/registry와 choices help는2passed in4.23s였습니다. 두 값은 이후 전체 focused suite와 중복되므로 합산하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync --offline python -m pytest tests/test_l2p_checkpoint.py tests/test_dualprompt_checkpoint.py tests/test_ranpac_checkpoint.py tests/test_baseline_efficiency.py tests/test_cl_efficiency.py tests/test_cl_runtime.py tests/test_ease_efficiency.py tests/test_ease_execution.py tests/test_prepare_ease.py -q --tb=short
CUDA_VISIBLE_DEVICES='' uv run --no-sync --offline python -m pytest tests/test_import.py::test_import_after_register tests/test_module_register.py::test_args_with_choices -q --tb=short
uv run --no-sync --offline python scripts/run_baseline_efficiency.py
python3 -m py_compile scripts/baseline_audit.py scripts/run_baseline_efficiency.py scripts/benchmark_cl.py tests/test_baseline_audit.py tests/test_baseline_efficiency.py
git diff --check
```

별도 source 감사 테스트76개와 반복 측정 runner 테스트67개도 통과했습니다. native meta wrappers와 restore, NaN/frozen/raw-data corruption, strict scope pin, complete counters와NPZ·집계경계를 확인했습니다. Runner mock은9invocation/12worker 및 실패/중단 시 후속실행 중지·부분로그 보존을 확인합니다.

실행 전 최종 focused suite는 **476 passed in11.01s**였습니다. 위 부분 검사와 중복되므로 개수를 합산하지 않습니다. 실제 실행 source와 계획을 커밋한 뒤 변경 없이12worker를 순차 실행합니다.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync --offline python -m pytest tests/test_l2p_checkpoint.py tests/test_dualprompt_checkpoint.py tests/test_ranpac_checkpoint.py tests/test_baseline_audit.py tests/test_baseline_efficiency.py tests/test_ease_efficiency.py tests/test_ease_acceptance.py tests/test_ease_execution.py tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_prepare_ease.py tests/test_cl_efficiency.py tests/test_cl_runtime.py tests/test_import.py::test_import_after_register tests/test_module_register.py::test_args_with_choices -q --tb=short
```
