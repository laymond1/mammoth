# HiDe-Prompt Base 전체 10태스크

승인된 다음 단계는 기존 독립 저자 기준선/native 첫2태스크 구현 검증을 통과한 HiDe-Prompt의 **Base 전체10태스크, seed1993 한 실행**입니다. GPU1·batch64, CIFAR100 고정 순서 10클래스/태스크, ViT-B/16 AugReg IN21K → IN1K 공유 NPZ를 사용합니다. 각 태스크에서 TII20 + prompt50epoch를 실행하며, 태스크2부터 두 branch의 분류기에 CA30epoch를 적용합니다. 상세 조건과 실행 상태는 [실행 계획](execution_plan.json)에 기록합니다.

기존 공유 데이터·캐시·저자 learnable 초기값을 재사용합니다. 첫2태스크의 예측 배열은 이전에 저자 기준선과 일치한 native 기록과 정확히 대조하며, 관련 모델·backbone·데이터·학습 source hash가 바뀌면 실행을 차단합니다. 전체10태스크에 대한 독립 기준선은 이번 기본 실행에 포함하지 않습니다. 비정상적으로 낮은 성능은 구현 문제로 조사하며, 서로 다른 recipe의 논문 수치에 1.5/3.0%p 기준을 직접 적용하지 않습니다.

완료 결과는 이 디렉터리의 JSON·CSV·Markdown, 정확도 행렬/추이 및 자원 사용량 PNG·PDF에 저장합니다. 최종 class-IL 정확도, 각 태스크 종료 시 관측 클래스 정확도의 평균, 과거9태스크 망각과 태스크별 시간·CUDA/NVML/RSS peak·GPU 보드 에너지를 보고합니다. 망각은 각 과거 태스크의 최종 태스크 이전 최고 정확도에서 최종 정확도를 빼고 평균하며, 개선되면 음수를 유지합니다. 미래 태스크의 행렬 항목은 null입니다. 한 seed의 한 실행이므로 seed 변동이나 신뢰구간은 추정하지 않습니다.

자원 계측은 데이터 로딩부터 태스크 종료 평가·예측·checkpoint 검사까지입니다. 태스크별 Slack/logging 및 마지막 artifact upload는 제외합니다. 중첩 phase 시간은 inclusive이며 단순 합산하지 않습니다. 에너지는 선택 GPU 보드의 sampled power 적분으로, 전체 머신 전력이 아닙니다. CUDA peak는 각 태스크에서 reset하고, RSS는 자식 프로세스의 공유 페이지를 중복 합산할 수 있습니다.

보고 도구는 저장된 결과만 읽으며 학습·GPU 작업을 시작하지 않습니다.

```bash
uv run --no-sync --offline python scripts/report_hide_fullcl.py \
  --native <full10-result.json> \
  --reference <accepted-author-two-task-result.json> \
  --accepted-native <accepted-native-two-task-result.json> \
  --output benchmarks/cl_efficiency/hide_prompt/full_cl

uv run --no-sync --offline python scripts/publish_hide_fullcl.py \
  --report-dir benchmarks/cl_efficiency/hide_prompt/full_cl \
  --output <fresh-publication-directory>
```

Publisher의 기본 실행은 로컬 검증만 합니다. `--execute`를 지정하면 `laymond1/cl-efficiency`에 report-only run·태스크/행렬/phase Table과 immutable artifact를 저장합니다. [기존 readback 도구](../../../../scripts/verify_hide_results.py)로 finished·COMMITTED·파일별 SHA 일치를 확인한 후 `--execute --notify-after-readback <receipt.json>`으로 Slack 완료 알림을 보냅니다. 이전 [첫2태스크 검증·효율성 pilot](../README.md)은 기존 기록을 유지합니다. Tiny·Small 및 BudgetCL/aL-SAR 예산 비교는 후속 단계입니다.
