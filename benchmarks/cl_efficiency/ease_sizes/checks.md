# EASE Small/Tiny 완료 확인 — 2026-10-07

Small64·Tiny64·Small32·Tiny16의 검증과 효율성 측정을 완료했습니다. 성공한 실험 worker는 24개(독립 기준선4, native 검증4, 비용 측정12, 별도 FLOPs 진단4)입니다. 최초 Small64 기준선 실패1회는 수정 후 재실행했고 집계에서 제외했습니다. 기존 Base64 결과는 재사용했습니다.

## 실제 실행한 검증

```bash
CUDA_VISIBLE_DEVICES='' UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline pytest tests/test_ease.py tests/test_ease_backbone.py tests/test_ease_classifier.py tests/test_ease_acceptance.py tests/test_ease_execution.py tests/test_ease_reference_utils.py tests/test_ease_efficiency.py tests/test_prepare_ease.py tests/test_cl_runtime.py tests/test_cl_efficiency.py tests/test_baseline_pretrained_args.py tests/test_baseline_efficiency.py tests/test_baseline_audit.py tests/test_ease_size_protocol.py tests/test_ease_sizes_runner.py tests/test_ease_size_reference.py tests/test_ease_size_validation.py tests/test_ease_size_efficiency.py -q --tb=short
```

결과: 750 passed, 0 failed (11.5s). classifier 폭 기본값 수정 이후 실행했습니다.

```bash
CUDA_VISIBLE_DEVICES='' UV_PROJECT_ENVIRONMENT=/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.venv uv run --no-sync --offline python main.py --model ease --dataset seq-cifar100-224 --model_config default --help
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check
```

두 명령 모두 exit0입니다. CSV의 표준 CRLF 줄끝을 허용하여 staged diff를 검사했습니다. 실제 학습 smoke는 승인된 native 검증 worker4회로 수행했습니다. 별도 실험을 추가하지 않았습니다. Native/독립 기준선 실제 CPU 생성8개와 synthetic classifier 검사8개도 통과했습니다.

[정확도·state 감사](environment/accuracy_state_audit.json)는 예측 배열로 정확도를 재계산하고16개 체크포인트의 finite 상태, canonical 사전학습 텐서150개, 이전 adapter와 관측 prototype 보존을 확인했습니다. 네 설정 모두 기준선 대비 최대 하락0.5%p로 통과했습니다.

[자원·원격 감사](environment/independent_pilot_audit.json)는 측정12개의 CPU 시간·전력 적분·자원 집계를 다시 계산하고24개 W&B run finished,24개 artifact COMMITTED,216개 파일을 대조했습니다. 태스크/실험 알림80개 요청이 모두 sent입니다. [소스·센서 감사](environment/pin_sensor_audit.json)도 통과했습니다.

[집계 보고서 readback](environment/report_readback.json)은 report-only run과16개 보고서 파일의 서버 저장본을 확인했습니다. 기존 결과 업로드이며 추가 학습을 하지 않았습니다. CSV15행 중3행은 재사용한 Base64입니다.

## 완료한 로컬 병합

`codex/cl-foundation`에서 `git merge --ff-only codex/ease-size-pilot`을 실행했습니다. 이전 HEAD는 `dc588b772d00c01e54985fc21b7f7236de298b0b`, 검증한 코드·결과 병합 HEAD는 `a01e7d3a706e7cb1a105e76fd9b00a69d2c91a50`입니다. 실행 소스는 `4d60f96adf8c0e18c83ff988880b8699147aeab8`입니다. [병합 기록](environment/local_merge.json)에 보존합니다. 이 문서와 원격 저장·Slack 완료 확인은 이후 문서 커밋으로 양쪽 브랜치에 반영합니다.

Base64 원본 JSON SHA256은 `dc84f34f585ebd93e2add569acecd379170cd4a48c9c5779808201c401f41aa9`로 유지됐습니다. 원시 worker 로그는 [실험 worktree](/mnt/sdab1/userHome/wslim/.codex/worktrees/aa40/compression-gap-cl/.worktrees/ease-sizes/benchmarks/cl_efficiency/ease_sizes/runs/ease-size-pilot-20261007T024856Z)에 남아 있으며 [공개 보고서](report.md), [결과 JSON](results.json), [측정 CSV](measurements.csv), [검증 CSV](validation.csv), 그래프는 기준 브랜치에도 저장했습니다.

[완료 메타데이터 업로드](environment/final_validation_upload.json), [서버 readback](environment/final_validation_readback.json), [Slack 완료 전송 확인](environment/completion_notification.json)에 마지막 저장 상태를 기록합니다. 추가 seed나 전체 CL 실행, 원격 push는 하지 않았습니다.
