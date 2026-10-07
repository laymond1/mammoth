# Continual LoRA implementation validation

The accepted source is [low-rank-cl](https://github.com/yaoyz96/low-rank-cl)
at `d007ef5c13a6d8a7d7db3eedf93f5e796e2ce429`. These are ports of that
repository's basic variants, rather than paper-result reproduction claims.

The common foundation contains source-compatible networks, canonical local NPZ
loading, native Mammoth lifecycle hooks, and independent validation tools.
Method entrypoints and their YAML defaults are added on separate local branches
in the order EWC-LoRA, SD-LoRA, InfLoRA, CL-LoRA. Each method is validated before
merging into `codex/cl-foundation`; no remote push is part of this stage.

| Method | Training per task | Preserved method behavior |
| --- | --- | --- |
| EWC-LoRA | 20 epochs, Adam, LoRA LR0.0005/head LR0.005 | Full empirical Fisher in weight space, gamma1, lambda1e7; separate accumulated A/B and reset |
| SD-LoRA | 20 epochs, Adam LR0.008 | Task-specific low-rank updates and source scaling parameters |
| InfLoRA | 20 epochs, Adam LR0.0005 | Full current-task feature collection before training, SVD initialization, post-training DualGPM update |
| CL-LoRA | 30 epochs, SGD LR0.03, momentum0.9, decay0.0001 | Shared blocks0–5/specific blocks6–11, Q/V LoRA, KD5/temperature2, separate KD then CE+orthogonality updates, prototypes |

## 방법별 검토 관점

| 방법 | 특징과 장점 | 비용과 한계 |
| --- | --- | --- |
| EWC-LoRA | weight-space Fisher로 새 저랭크 갱신을 제약하고 누적 지식을 보존합니다. 원본 이미지 replay를 저장하지 않습니다. | Base에서 24개 768×768 Fisher 행렬만 FP32 약54MiB이며, 태스크마다 전체 데이터의 추가 backward가 필요합니다. Fisher 강도·누적 A/B 방식의 영향을 검토해야 합니다. |
| SD-LoRA | 태스크별 LoRA와 학습 가능한 공유 scale로 과거 갱신의 영향을 조절합니다. 별도 Fisher·SVD 단계가 없어 학습 흐름이 단순합니다. | 과거 LoRA를 저장하고 매 forward에 누적해 사용하므로 태스크가 늘면 저장량·연산도 증가합니다. scale 조절에 따른 과거 클래스 변화가 검토 대상입니다. |
| InfLoRA | 과거 feature 서브스페이스를 고려해 A를 정하고 B를 학습하여 태스크 간 간섭을 줄이려는 방법입니다. 과거 이미지를 보관하지 않습니다. | 학습 전후 전체 데이터 feature 수집, SVD와 DualGPM 상태가 필요합니다. 학습 파라미터 수만으로 실제 CPU·시간·메모리 비용을 판단하기 어렵습니다. |
| CL-LoRA | shared/specific LoRA를 나누고 증류·직교 손실·prototype을 함께 사용합니다. 지식 공유와 태스크 특화를 각각 제어할 수 있습니다. | 두 번째 태스크부터 배치마다 KD와 CE의 두 optimizer step이 필요합니다. 이 통합 저장소의 추론은 학습한 각 태스크에 대해 전체 transformer를 다시 실행하므로, 논문의 효율성 주장을 그대로 적용하지 않습니다. |

위 장단점은 고정한 [통합 코드](https://github.com/yaoyz96/low-rank-cl/tree/d007ef5c13a6d8a7d7db3eedf93f5e796e2ce429)의
구조에 근거합니다. 실제 성능과 자원 비용은 검증 결과 및 후속 공통 효율성
실험으로 확인합니다. 다른 데이터셋에 연결할 때는
`--lora_classes_per_task`를 실제 태스크 크기와 맞춰야 합니다.

All methods use rank10, batch64, seed1993, workers4, FP32 and physical GPU1.
The complete `seq-cifar100-224` class order is 0–99, ten classes per task;
validation stops after the first two tasks without changing the task capacity.
Train preprocessing is the pinned source's bilinear RandomResizedCrop224,
horizontal flip, ToTensor and identity Normalize(0,1). Test preprocessing is
bilinear Resize224, ToTensor and identity Normalize(0,1).

The selected ViT-B/16 AugReg weights are IN21K pretrained and IN1K fine-tuned.
The complete 150 canonical encoder tensors must match the pinned digest.
The independent source constructors export all remaining initialization tensors
for exact transfer to the native network. Small and Tiny use the matching
AugReg family and are checked on CPU through creation and forward only.
Shared data/cache paths and credentials are loaded through `scripts/cl_runtime.py`.

The independent worker imports the immutable external learner and numerical
network bodies directly. Its adaptations are local checkpoint loading, a
1000-way unused head consistent with the selected checkpoint family (CL has no
head), stable raw sample order, explicit phase seeds, and read-only telemetry.
It never imports native lifecycle implementations. For CL-LoRA, modern PyTorch's
`allow_mutation_on_saved_tensors` preserves the original CE graph across the
separate KD parameter update in both workers.

For InfLoRA, CPU covariance tensors are converted to NumPy arrays before calling
the original `update_DualGPM`. In the installed NumPy/Torch combination,
`np.linalg.svd(torch_tensor)` returns Torch tensors, while the source expects
NumPy's zero-argument `transpose()`. The boundary conversion preserves the
covariance dtype/values and original SVD/threshold/complement arithmetic. The
first failed reference attempt is preserved separately and excluded from the
completed-run report.

The pinned EWC source evaluates before merging A/B. Native Mammoth evaluates
after `end_task`, so EWC computes Fisher there and defers only merge/reset to
`end_eval`. Checkpoints contain the finalized algorithm state; a saved pending
merge is restored and completed before the next task. Other task phases remain
in their native `begin_task`/`end_task` hooks. Training always uses `meta_*`.

Acceptance evaluates task1 seen accuracy, task2 seen accuracy, task2 old-class
accuracy and task2 new-class accuracy using exact correct/sample counts.
A drop ≤1.5 percentage points passes; >1.5–3 requires review; >3 fails.
Missing data, nonfinite losses/gradients, changed pretrained tensors, incomplete
updates, incorrect initialization, future/replay samples, or invalid method
state fail independently of accuracy. First-batch inputs, labels and gradients,
complete predictions with raw IDs, optimizer counts and checkpoint round-trips
are recorded for diagnosis.

Both workers save local records and online W&B runs in
`laymond1/cl-efficiency`, with Slack start/task/progress/completion notifications.
Resource observations describe these validation runs. They are not the later
first-task efficiency comparison. No full ten-task run, additional seeds,
Small/Tiny training, budget-method implementation or efficiency ranking is
authorized as part of this stage.

```bash
# Describe without training or external writes.
uv run python scripts/run_low_rank_reference.py --method ewc-lora
uv run python scripts/run_low_rank_validation.py --method ewc-lora

# Export actual independent source CPU initialization (no training).
uv run python scripts/run_low_rank_reference.py --method ewc-lora \
  --prepare-initial-state results/low_rank/initial/ewc-lora.pt

# Explicit execution, after CPU checks and source freeze.
uv run python scripts/run_low_rank_reference.py --method ewc-lora \
  --initial-state results/low_rank/initial/ewc-lora.pt --execute
uv run python scripts/run_low_rank_validation.py --method ewc-lora \
  --initial-state results/low_rank/initial/ewc-lora.pt \
  --reference-summary /absolute/reference/result/summary.json --execute
```
