# Executed validation and provenance

The approved GPU stage completed five full-stream runs: CPS author/native on
physical GPU1, then frozen-CODA / REP OFF / REP ON on GPU0. Each run retained
10 original task partitions and trained only the first two, with three epochs
per task, 79 updates per epoch, batch64, seed1993, Tiny and FP32 without AMP/TF32.
GPU1 and GPU0 groups ran concurrently. No20epoch pilot or broader comparison ran.

Commands actually executed, using the existing local uv environment:

```bash
uv run --no-sync --offline pytest -q tests/test_cps_rep_validation.py tests/test_cps_prompt.py tests/test_rep_prompt.py tests/test_cpu_device.py
uv run --no-sync --offline python scripts/run_cps_rep_validation.py --prepare
uv run --no-sync --offline python scripts/run_cps_rep_validation.py --cpu-smoke cps-author --output results/cps_rep_reference_smoke_v4_20261009
uv run --no-sync --offline python scripts/run_cps_rep_validation.py --cpu-smoke cps-native --output results/cps_rep_reference_smoke_v4_20261009
uv run --no-sync --offline python scripts/run_cps_rep_validation.py --cpu-smoke coda-frozen --output results/cps_rep_reference_smoke_v4_20261009
uv run --no-sync --offline python scripts/run_cps_rep_validation.py --cpu-smoke rep-off --output results/cps_rep_reference_smoke_v4_20261009
uv run --no-sync --offline python scripts/run_cps_rep_validation.py --cpu-smoke rep-on --output results/cps_rep_reference_smoke_v4_20261009
uv run --no-sync --offline python scripts/run_cps_rep_validation.py --execute --output results/cps_rep_gpu_validation_20261009
uv run --no-sync --offline python scripts/report_cps_rep_validation.py
uv run --no-sync --offline python scripts/report_cps_rep_validation.py --publish
```

The focused test group passed47 tests in11.81s. All five CPU smoke workers
completed their two debug tasks; paired complete epoch input hashes and
Class-IL accuracies matched. GPU workers each completed474 updates,
30000 current training-image visits and3000 evaluation-image visits.

CPU checks used `CUDA_VISIBLE_DEVICES='' MAMMOTH_DEVICE=cpu WANDB_MODE=disabled`
and two OMP/MKL threads. Every ML process used
`JE_ARROW_MALLOC_CONF=background_thread:false`. UV used
`UV_CACHE_DIR=/tmp/cl-cps-rep-uv`. GPU controller workers independently set
physical `CUDA_VISIBLE_DEVICES` and loaded shared paths/private credentials
through `scripts/cl_runtime.py`.

Native Ruff checks passed with the repository configuration and `N999` excluded
for the hyphenated ancestor worktree path. Python compileall passed. Pinned
unmodified author fixtures are excluded from native formatting/whitespace checks.
The published CSV retains standard CRLF line endings to preserve its verified
W&B byte digest; it is excluded from Git's trailing-whitespace check.
The final post-publication focused runtime contract rerun passed17 tests in4.50s.

Preparation corrected a stale `--nowand` CLI assumption: this Mammoth version
derives it from W&B settings; the worker takes explicit ownership after
`initialize`. CPU smoke also exposed aliasing of saved CPU logits with the
evaluator's in-place Task-IL mask; capture now copies logits and checks saved
Class-IL predictions against Mammoth's results. These fixes preceded GPU runs.

GPU execution sources are pinned in every `spec.json` and `result.json` to
commit`c9adb84`. Model behavior was unchanged during the runs. A later metadata
fix changes the raw-power upload name from `samples.csv` to the actual
`telemetry.csv`. The original12-file worker artifacts remain intact; all five
sensor CSVs and Slack receipts were added to the final report artifact and their
digests verified. No GPU training was repeated for this publication correction.

W&B readback verified five finished workers, committed artifacts, exact stream
counts/accuracy summaries, file digests/sizes including the task checkpoints,
and immutable computation record fields. Final Slack completion was appended
after each worker upload, so record-semantic checks intentionally exclude that
last notification receipt. All20 worker notification requests, the controller
completion and verified-report completion returned `sent`.

Automatic approval review rejected an additional report upload of
`initial_learned.pt`, citing insufficient explicit authorization for that
parameter payload. The publisher was changed to upload only metrics, hashes,
sensor observations and notification receipts. Initial tensors remain local,
with their digest and source initialization recipe in the report. This safer
publication completed without asking the user for additional approval.

See `worker_readback.json` and `publication.json` for verified remote evidence.
The resource values in `RESULTS.md` include verification overhead and are not
accepted efficiency-pilot results. REP ON remains held; no cause was isolated
by this combined-component run.
