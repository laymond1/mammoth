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
