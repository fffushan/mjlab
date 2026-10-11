# VAE throughput defaults and combined reference validation

Date: 2026-10-08 (+08).
Implementation baseline: `58ea6edee3d93f67cf4374d782a3673323aee3b4`.
Status: implemented and validated. The retained implementation is recorded in Git history.

## Implementation

The public `distill train` command and `TrainingConfig` now share these constants:

- `DEFAULT_MINIBATCH_SIZE = 16384`
- `DEFAULT_ACCUMULATION_STEPS = 2`
- Effective batch:32768 rows.

Both values remain configurable. Existing checkpoint resumes must repeat their stored values.
CPU smoke tests need small explicit batch values. One CLI test helper now supplies those values.
No collection, draw-ratio or replay default was changed in this task.
The evidence supporting the batch choice is the [remote comparison](../../logs/diagnostics/vae-throughput-inc000-007-20261008T134320Z/REPORT.md).

`MotionLibrary.flat_index` now combines four valid-path GPU Boolean checks into one.
It uses a safe, clamped ID only for the speculative clip-length metadata gather.
It still rejects invalid input before any reference-storage address is computed.
Accepted IDs and frames are not clamped.
Detailed rejection remains ordered as invalid motion ID, negative frame, then over-long frame.
No persistent cache, reset behavior, RNG sequence, teacher label or loss change is introduced.

Files:
- `src/mjlab/scripts/distill.py`
- `src/mjlab/tasks/tracking/distillation/training_config.py`
- `src/mjlab/tasks/tracking/distillation/motion_library.py`
- `tests/test_tracking_distillation_cli.py`
- `tests/test_tracking_distillation_trainer.py`
- `tests/test_tracking_distillation_motion_library.py`
- `tests/test_tracking_distillation_sample_budget.py`
- `docs/source/x2_tennis_distillation.rst`
- `docs/source/changelog.rst`

## Validation evidence

The final six-suite CPU/fake-simulator regression run reports **212 passed**, eight existing ONNX deprecation warnings.
Duration:93.05 seconds. Output was written to `/tmp/vae-default-collection-final-tests-20261008.log`, which no longer exists.
No live simulator is built by these suites.

An earlier run was stopped after faulthandler reported a long backward call.
The CLI test helper had inherited the new16384-row GPU batch.
The helper now supplies microbatch4 and accumulation3.
The three affected cadence/resume tests pass in5.60 seconds.
The original diagnostic was written to `/tmp/vae-default-collection-tests-20261008.log`, which no longer exists.
The stop was not reported as a passing run.

A later local CUDA regression **passes** on the RTX5080 Laptop GPU.
It checks valid output, one `aten::item` decision, extreme int64 motion IDs and invalid frames.
It also checks that malformed IDs do not poison CUDA and that input mutations are revalidated.
The valid-path profiler count changes from four host-scalar decisions to one.

Ruff: pass for changed Python files.
Ty: pass for CLI, motion library, training config and sample-budget module.
Pyright: motion library, training config and sample-budget module pass.
The CLI retains its three previously verified optional-collector/base-task diagnostics.
`git diff --check`: pass.
Full repository suite: not tested.

## Local CUDA microbenchmark

User explicitly authorized the local GPU.
Device: NVIDIA RTX5080 Laptop GPU,16303MiB, PyTorch2.9.0+cu128.
The benchmark compares the unchanged baseline method with the combined guard.
Each call handles4096 valid rows across eight synthetic clip metadata entries.
There are eight paired blocks of300 calls, with alternating old/new order and warmup.
Each block has explicit CUDA synchronization before and after timing.
No model, teacher, training or simulator is constructed.

| Method | Median wall time per call |
| --- | ---: |
| Original four-decision guard | **69.800 µs** |
| Combined one-decision guard | **54.492 µs** |

Measured method speedup: **1.281×**; per-call time reduction21.93%.
[Raw paired results](../../logs/diagnostics/vae-flat-index-local-cuda-20261008/microbenchmark.json).

The first benchmark wrapper failed before GPU work with an undefined annotation namespace.
The wrapper namespace was repaired. The measured baseline algorithm was not changed.
The baseline method SHA-256 is stored with the result.

This result applies only to valid `flat_index` calls on this local GPU.
Full collection and remote4090 speedup are **not measured** after this implementation.
The local GPU returns to15MiB baseline with no compute process.

## Next collection directions

1. Reuse reference flat indices for one immutable tick snapshot.
2. Aggregate per-motion diagnostic reductions before transferring scalars to the CPU.
3. Inspect reset/delay-buffer work without changing its RNG or state history.

These are follow-up directions, not implemented optimizations.
Reference caching needs reset, wrap, phase-change, in-place mutation and restore invalidation tests.
The combined guard avoids those cache-lifecycle risks and is the bounded change implemented now.
Do not remove non-finite checks or domain randomization to improve a speed score.

## Operational status

No new server training, live simulation, deployment or hardware operation occurred.
The local CUDA regression and method microbenchmark are complete.
No background test process remains from this task.
Next action: run a matched full-collection benchmark if the user authorizes it.
Do not claim a collection speedup from the local method benchmark alone.
