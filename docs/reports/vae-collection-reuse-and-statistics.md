# Collection reference reuse and deferred statistics

Date: 2026-10-08 (+08).
Implementation baseline: `957ac2b0e9fbd86a2fe1274f9d3f4eef98236c35` plus existing local VAE budget/default changes.
Status: implemented and validated. Remote paired validation is complete.
The retained implementation is recorded in Git history; run snapshots keep their original source identities.

## Implementation

The multi-motion command retains one validated flat reference index.
Reuse requires the same motion tensor, frame tensor, library and tensor version counters.
Subset writes, writes through views, frame advance, reset, wrap and restore copies change the counters.
Tensor or library replacement also triggers validation.
Inference tensors have no version counters and remain uncached.
Public motion-library queries still validate every call.
Reference gathers return fresh values; the command caches indices, not observations or policy actions.
Lookahead validates one index for both joint position and velocity.

The collector retains owned pre-step metadata for per-motion statistics.
It transfers per-tick scalar statistics in batches at report construction.
Frame coverage and initialization/early-window counts retain their prior meaning.
Per-tick float32 disagreement reductions convert to Python in the same order as before.
Teacher/motion mapping is still checked before each simulator step.
Global teacher-step/disagreement scalar transfers also occur after collection.
Pending buffers are cleared after building the report; repeated builds do not double count.

No reset/delay randomization, teacher label, executed action, normalizer source or VAE loss was changed.
Reset-path optimization remains deferred because it can change RNG/state semantics.

Files:
- `src/mjlab/tasks/tracking/distillation/motion_library.py`
- `src/mjlab/tasks/tracking/distillation/multi_motion.py`
- `src/mjlab/tasks/tracking/distillation/collector.py`
- `tests/test_tracking_distillation_multi_motion.py`
- `tests/test_tracking_distillation_collection_statistics.py`
- `docs/source/x2_tennis_distillation.rst`
- `docs/source/changelog.rst`

## Software validation

Final expanded command includes collection statistics, motion command, routing, sharded merge,
CLI/cohort CLI, motion library, standing trials, provenance and X2 coverage-adapter tests.
Result: **209 passed**,12 existing ONNX/model-timestep warnings,62.49 seconds.
[Captured output](/tmp/vae-next-collection-final-tests-20261008.log).
The first narrower run reports190 passes and is preserved separately.

Tests cover subset reset, frame wrap, timer resample, explicit restore-like copies,
replacement tensors/library, mutation through views and invalid frames/IDs after cache population.
Tests cover inference-tensor fallback and local CUDA cache invalidation.
Statistics tests cover owned metadata, exact counters and disagreement values,
missing metadata, unknown kinds, repeated report builds and mapping-change rejection.

Ruff and Ty pass for the changed modules/tests.
Pyright passes motion library and motion command.
The collector has one existing coverage-capture Iterable diagnostic.
The unchanged pre-task collector has the same diagnostic.
No full-repository suite was run.
Some routing tests construct and inspect a local X2 environment; they do not train or run policy rollouts.

One new test initially expected enabled lookahead from a fixture with lookahead disabled.
The corrected assertion checks that the disabled lookahead stays zero-width.
A test import initially used the wrong owner for `TeacherRoutingError`; it was corrected to `collector.py`.
No production behavior was changed to satisfy either fixture error.

## Local CUDA component evidence

Device: NVIDIA RTX5080 Laptop GPU.
4096 rows, eight synthetic motions, alternating baseline/optimized measurement order.

| Component | Baseline median | Optimized median | Speedup |
| --- | ---: | ---: | ---: |
| Ten reference queries per mutable tick |586.365µs |145.033µs |**4.043×** |
|32 ticks of per-motion stats including report flush |86.421ms |55.942ms |**1.545×** |

Ten reference tensor results match exactly.
Per-field statistics match the original path exactly in the checked3-tick comparison.
Each reference benchmark tick changes frames before querying.
The cache therefore revalidates each tick and only reuses indices inside that tick.
[Raw component results](../../logs/diagnostics/vae-next-collection-local-20261008/microbenchmark.json).
These are synthetic component measurements, not full collection throughput or quality evidence.

## Precommit validation

The combined VAE regression run reports **293 passed**,13 existing warnings,71.03seconds.
This includes budget/default, collection/routing, checkpoint, provenance and local CUDA checks.
Nine targeted tests pass after annotation-only precommit changes.

Ruff, Ty and production-module Pyright pass for all changed VAE Python files.
Precommit cleanup adds explicit optional-collector checks, an optional base-task annotation,
a coverage-capture iterable cast and accurate test-double annotations.
It does not change the measured optimization algorithms or sample budgets.

Repository-wide `make check` runs in a disposable snapshot to preserve concurrent work.
Formatting and Ruff pass. Ty stops with74 diagnostics outside the changed scope.
The untouched baseline reports91 diagnostics. All remaining diagnostic identities are present in the baseline.
No broad-check pass or full-repository-suite pass is claimed.
Logs: `/tmp/vae-session-commit-20261008/`.

The colleague-facing HTML and issue-list links, SVG structure and budget arithmetic also pass validation.
Generated checkpoints, source archives and raw profile data remain outside Git.

## Remote paired validation

User authorized remote testing and reported switching the shared checkout to `devel`.
Fresh inspection confirms `devel` at957ac2b0e, with only an untracked queue lock.
The comparison leaves that checkout untouched.
Both source archives import through explicit isolated `/tmp` snapshots.
The407-file inventories differ only in collector, motion library and multi-motion command.

Both runs use the same managed GPUs0–1, eight teachers,8192 environments,
microbatch16384, accumulation2, eight updates,32 collection steps, replay524288,
128-step bootstrap, numerical controls, standing mixture and seed0.
Each run lasts10 lifecycle iterations and runs sequentially on the same GPU pair.

Remote root: `/home/fushan/mjlab/runs/vae-collection-paired-20261008T145342Z`.
[Launch plan](../../logs/diagnostics/vae-collection-paired-20261008T145342Z/launch-plan.json).
Monitor:#15. Deadline:20 minutes per run. Exact owned-process cleanup is installed.

The first staging/preflight shell exceeded60 seconds.
Read-only inspection found both completed preflight records and no remaining preflight process.
It was not repeated or given a longer timeout.

Outcome: **both10/10-iteration runs complete; exit0**.
Normal iteration mean: **3.868→3.334seconds**, a **1.160× throughput improvement**.
Collection/transport mean:3.271→2.740seconds. Optimizer mean is approximately unchanged.
GPU sampled peaks:9293/4259MiB→9305/4271MiB, an increase of12MiB per device.
Sample/draw accounting matches; all report values are finite.
GPU training/rollout metrics are not bitwise identical; no policy-quality claim is made.
Both407-file inventories and38fetched artifact hashes pass verification.
All nine owned processes exit. GPUs return to1MiB and the shared `devel` HEAD/status is unchanged.
[Full remote evidence](../../logs/diagnostics/vae-collection-paired-20261008T145342Z/REPORT.md).

The local component benchmarks and the smaller remote end-to-end improvement are different measurement scopes.
No retry, production training, deployment or hardware operation occurred.
Next action: retain this measured collection change and validate any further reset/boundary optimization separately.
Further runs require user authorization.
