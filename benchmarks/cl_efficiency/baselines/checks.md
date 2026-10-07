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

첫 실행의 독립 감사에서 dynamic backbone parser가 `--pretrained 0`을 `bool('0') == True`로 해석하는 문제를 발견해 중단했습니다. 최종 PTM은 동일하지만 초기화에서 기본 HF 가중치를 추가로 로드했으므로 비교에서 모두 제외했습니다. [제외 기록](environment/excluded_attempts.json)에 부분 로그와 W&B URL을 보존합니다. RanPAC/L2P에 명시적인 binary parser를 추가하고 모델 생성 전 실제 옵션 검사를 넣었습니다.

수정 후 실제 CLI parsing 회귀 검사 9개와 초기화 전 차단 검사를 포함한 focused suite는 **490 passed in11.15s**였습니다. 새 성공 측정에는 수정된 커밋을 사용합니다.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync --offline python -m pytest tests/test_baseline_pretrained_args.py tests/test_l2p_checkpoint.py tests/test_dualprompt_checkpoint.py tests/test_ranpac_checkpoint.py tests/test_baseline_audit.py tests/test_baseline_efficiency.py tests/test_ease_efficiency.py tests/test_ease_acceptance.py tests/test_ease_execution.py tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_prepare_ease.py tests/test_cl_efficiency.py tests/test_cl_runtime.py tests/test_import.py::test_import_after_register tests/test_module_register.py::test_args_with_choices -q --tb=short
```

수정 source `20ec323`을 고정하여 승인한 native 실행12개를 완료했습니다. 실제 첫 태스크의 observe/begin/end/evaluation flow도 모두 통과했고, 추가 debug 학습 없이 이 실행을 runtime smoke로 사용했습니다. [독립 감사](environment/independent_pilot_audit.json)에서 source·가중치·raw 데이터·NPZ·완료 카운터·전력 적분·측정만의 집계를 재계산했습니다. [원격 확인](environment/wandb_pilot_readback.json)에서12run이 `finished`, artifact가 `COMMITTED`, 핵심 metrics와 spec/NPZ/telemetry/phase 파일이 로컬과 일치함을 확인했습니다. 각 suite Slack 요청은 모두 `sent`입니다.

```bash
uv run --no-sync python scripts/run_baseline_efficiency.py --execute
python3 /tmp/audit-baseline-pilot-20261007.py benchmarks/cl_efficiency/baselines/runs/baseline-efficiency-pilot-20261007T012659Z --source-commit 20ec3237699ccfd12a3fb07e74f70a53e1fba79f --output benchmarks/cl_efficiency/baselines/environment/independent_pilot_audit.json
uv run --no-sync python /tmp/verify-baseline-pilot-20261007.py benchmarks/cl_efficiency/baselines/runs/baseline-efficiency-pilot-20261007T012659Z
uv run --no-sync python /tmp/mark-excluded-baselines-20261007.py
```

독립 감사는 framework/numpy/torch를 import하지 않는 별도 stdlib NPZ parser·전력 적분기입니다. CPU ridge18solve 표기는 수정하지 않은 source의17후보+최종1회 protocol이며 호출별 runtime 추적이라고 주장하지 않습니다. 측정9개의 최소 전력 coverage는99.806%이고 별도 FLOPs 실행은66.31~83.12%입니다. 진단의 시간·메모리·CPU 시간·에너지는 측정 집계에서 제외했습니다. 임시 검증 script와 공개 JSON에 검증 근거를 남겼으며, raw 실행은 ignored run폴더에 보존됩니다.

검증된 코드 `20ec323`을 `b8fba75`에서 `codex/cl-foundation`에 fast-forward 병합한 뒤 동일 focused suite를 Foundation에서 다시 실행하여 **490 passed in11.21s**를 확인했습니다. `main.py --model sgd --dataset seq-cifar10 --help`도 정상 종료했습니다. 후속 결과 문서는 별도 커밋으로 같은 브랜치에 fast-forward 병합합니다. 원격 push/PR은 수행하지 않습니다.

[W&B 4방법 집계](https://wandb.ai/laymond1/cl-efficiency/runs/pfg5ytbp)는 추가 학습이 없는 report-only run입니다. [서버 확인](environment/wandb_summary_readback.json)에서 `finished`·`COMMITTED`, 모든 집계값의 일치와7개 artifact 파일의 SHA256 일치를 확인했습니다. JSON↔서버 float 직렬화 차이에는 상대/절대1e-12 허용값을 사용했습니다. 결과 JSON의 `local_merge`는 이 집계 업로드 직전의 코드 병합 snapshot이며 `additional_report_commit_pending`도 해당 시점의 상태입니다. 최종 문서 병합 여부는 현재 Git 기록을 따릅니다.

```bash
git merge --ff-only 20ec3237699ccfd12a3fb07e74f70a53e1fba79f
uv run --no-sync --offline python main.py --model sgd --dataset seq-cifar10 --help
uv run --no-sync python /tmp/upload-baseline-summary-20261007.py
```
