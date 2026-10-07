# EASE Small/Tiny matched validation and efficiency pilot

The user authorized all four variants on 2026-10-07: Small/adapter64,
Tiny/adapter64, Small/adapter32 and Tiny/adapter16. Existing accepted
Base/adapter64 measurements are reused without another Base run.

All variants use CIFAR100-224 with ten classes per task, natural class order,
seed1993 and physical GPU1. Each has an independent reference from pinned
author code and a native Mammoth validation of two tasks, twenty epochs per
task, batch48 and four workers. Small/Tiny references specialize only the
author's architecture constructors and load the corresponding local AugReg
IN21K→IN1K checkpoint. These are size-variant references, not published
Small/Tiny paper reproductions. Different size-specific pretraining recipes
are recorded in `environment/checkpoints.json`.

Native accuracy is compared with its matching reference using exact saved
prediction counts: at most 1.5 percentage points lower passes; 1.5–3.0 points
requires inspection; more than 3.0 fails. All structural checks must pass.
Inspection or failure blocks efficiency measurements until resolved.

Each accepted variant then runs three independent first-task/one-epoch
measurements and one separate FLOPs diagnostic, batch64 and four workers.
Only measurement runs contribute time, memory and energy statistics. FLOPs
coverage is partial; diagnostic overhead is excluded from resource medians.
Repeats share one seed and measure run-to-run resource variation.

The complete authorized stage contains 24 new successful workers: four
references, four native validations, twelve measurements and four diagnostics.
Local raw artifacts are preserved under ignored `runs/`; each experiment
uploads to `laymond1/cl-efficiency` and sends existing configured Slack
notifications. Public summaries record source/checkpoint hashes and remote
verification. No full CL sequence or additional seed is part of this stage.

Preparation only:

```bash
uv run --no-sync python scripts/run_ease_sizes.py
```

Authorized execution:

```bash
uv run --no-sync python scripts/run_ease_sizes.py --execute
```

All worktrees share the paths pinned in `validation_plan.json`; private
credentials remain outside tracked experiment metadata.
