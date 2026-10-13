# X2 x0 flow matching implementation plan

Date: 2026-10-13 (+08).
Baseline: `d94792f35d2971c9faf0f76b499b4bbc778d0f60`.
Status: implemented, reviewed, and parent-validated in the primary checkout. Changes are uncommitted. No production experiment or deployment authorized.

## Scope and fixed decisions

- Keep the implementation minimal. Add only small unit tests for key functions.
- Avoid generic frameworks, speculative options, exhaustive test matrices, and large fixture scaffolds.
- Reuse existing regression tests. The validation list below identifies key risks, not a requirement for separate tests per item.
- Reuse D1 stores, split assignments, train-only statistics, token caches, and the frozen VAE.
- Keep 41 rows and 167 channels (135 state + 32 latent).
- Reuse `StateLatentTransformer` without changing its structure or parameter count.
- Keep the two 1001-entry integer embedding tables. Sample independent levels per row and per state/latent slice.
- Predict clean `x0`. Use plain channel-mean x0 MSE, as in the diffusion baseline.
- History dropout is supported and matches the diffusion meaning. A nonzero value raises a whole window's history state and latent levels to pure Gaussian. The default is 0.0.
- Preserve existing diffusion defaults, checkpoints, CLI behavior, and numerical outputs.
- Add a separate opt-in flow family. Do not label flow artifacts as diffusion artifacts.
- Deliver bounded training, resumable checkpoints, and offline generated-token evaluation.
- Defer D3/controller integration, ONNX export, guidance, hardware, new data collection, and bulk training.

## Mathematical contract

Use noise fraction `s=k/1000`, where k is an integer in [0,1000].
For each row and slice, draw independent Gaussian noise e and form:

`x_s = (1-s)*x0 + s*e`.

The network receives the existing integer level IDs and predicts `x0_hat`.
Training samples independent state and latent levels for all rows, including history.
There is no extra clean-history mask in training. This preserves the baseline conditioning strategy.
Level 0 is exactly clean. Level 1000 is exactly Gaussian noise.

With a data-to-noise coordinate, `dx/ds = (x_s-x0)/s`.
Plain x0 MSE is the s-squared-weighted flow-velocity MSE for s>0.
It is not the unweighted velocity-regression objective. Level 0 uses reconstruction loss only.
Record `prediction_type=x0`, `path=linear_gaussian`, `loss=x0_mse`, and `time_encoding=integer_noise_fraction` in artifacts.

Sampling descends from k=1000 to k=0. For a source s>0 and destination r:

`x_r = (r/s)*x_s + (1-r/s)*x0_hat`.

This is an Euler step of the x0-parameterized flow ODE. It avoids explicit division of a predicted velocity near zero.
Never call the network for a source level of 0. The terminal update returns x0_hat on unknown entries.
Support per-row/per-slice source and destination levels in the primitive; reject invalid or ascending updates.
The default sampler uses the same source level on unknown slices at each call, as the current diffusion sampler does.
Known slices use level 0 and are restored exactly after initialization and each update.
Keep history state/latent and current state fixed. Generate current latent and future state/latent.
Use a configurable strictly descending integer grid with no duplicate levels (default 20 calls).

## Ownership and sequence

This is multi-seam work. Components have independently testable contracts.
All writers run serially in one isolated checkout. No concurrent writes occur.
Each component supplies a durable managed handoff before the next stage starts.

| Stage | Exclusive ownership | Gate | Handoff |
| --- | --- | --- | --- |
| 1. Flow primitives | New `flow_matching/__init__.py`, `process.py`, `sampler.py`; primitive/sampler tests | Endpoints, independent levels/noise, oracle updates, masks, invalid inputs | API names, changed files, exact test results |
| 2. Training and artifacts | New flow configuration, trainer, checkpoint modules; their tests | Tiny synthetic training, deterministic resume, identity checks, dropout rejection | Training/loading API and fixture recipe |
| 3. CLI and offline evaluation | New flow script, evaluation module, example config; CLI/evaluation tests | Synthetic end-to-end smoke, data identity guards, no future leakage | Runnable commands and metric semantics |
| 4. Integration and documentation | Minimal interface fixes; docs/index/changelog/runbook; cross-seam tests | Combined focused suite, relevant diffusion regression, scoped lint/type checks | Acceptance report and remaining limits |
| 5. Independent review | Read-only review of integrated diff and tests | Evidence-backed correctness/regression findings | Findings with severity and file references |

Stages 2 and 3 may import diffusion infrastructure. Avoid copying the entire diffusion trainer or CLI.
If sharing requires a small neutral extension hook, document it, preserve baseline behavior, and add regression tests.
Do not create fake cosine schedules to make flow artifacts pass diffusion checks.
Do not load a diffusion checkpoint as a flow checkpoint or allow cross-family resume.
Existing data contracts describe frozen data/VAE provenance; new model-family metadata describes the flow process.

## Validation

Use `uv run` with the isolated source on PYTHONPATH.
Use synthetic fixtures for bounded CPU tests. No remote hosts or physical robots.
Use 30 s inspection, 60 s single-test, and 120 s focused-suite limits.
After an unexpected timeout, diagnose before another attempt. Do not increase the same timeout blindly.
Long validation requires a parent-managed monitored command.

Required evidence:
- Exact endpoints and independent state/latent/row levels.
- Correct x0 loss, including level-zero behavior and finite gradients.
- Oracle x0 solver reaches the expected clean endpoint.
- Known entries remain bit-identical after every sampler call.
- Noise and level generators are reproducible; resume retains their states.
- Checkpoint model family, path, objective, data, statistics, and model settings are checked.
- Flow configuration accepts history dropout and raises the history pair to pure Gaussian.
Use 0.0 for the baseline arm.
- Offline metrics score unknown generated state and latent values, not velocity loss.
- CLI consumes existing data/statistics/caches without modifying them.
- Existing diffusion behavior remains unchanged.

## Authority and stop conditions

Children focus only on their assigned stage. No nested subagents, commits, pushes, GPU jobs, downloads, deployment, or shared artifact writes.
A source/API design conflict must be reported to the parent. Do not silently change the mathematical contract or broaden scope.
Stop the workflow on a child infrastructure failure. Preserve its partial diff and handoff.
The parent inspects evidence and disposition of reviewer findings before acceptance.

## Status and evidence

### Implementation

Completed on 2026-10-13 at 15:55 (+08). The reviewed files are installed in `/home/agiuser/projects/mjlab`.
Source: `src/mjlab/tasks/tracking/flow_matching/` and `src/mjlab/scripts/flow_matching.py`.
Configuration: `configs/flow_matching/x2_50hz_train.yaml`.
Runbook: `docs/source/x2_flow_matching_training.rst`.
The source, config, runbook, plan, and tests are additions. Existing edits affect only `docs/index.rst` and `docs/source/changelog.rst`.
No existing diffusion source or data file changed. No commit or push occurred.
The isolated checkout remains at `/home/agiuser/projects/.worktrees/mjlab-flow-x0-20261013`.

### Review disposition

Independent review and targeted re-check use `ds-oc/deepseek-flash:high`.
Verdict: **OK with notes**. No P0 or P1 finding remains.
Fixed: non-default `max_level` round-trip, mixed-schema rejection, small-level evaluation grid, and documentation scope statements.
Default evaluation uses `min(20, max_level)` calls. At `max_level=1000`, it still uses 20 calls.
Single-slice baseline metric names remain unchanged. The runbook now states their scope.
Three redundant tests were removed; one config round-trip test was added. Two edge checks reuse existing tests.
The reviewer accepts the remaining tests as focused function checks and one CLI smoke harness. One CUDA-gated sampler regression test was added after the local GPU smoke (see below), so the focused suite now holds 54 tests.
Checkpoint/trainer duplication remains a non-blocking limitation. Consolidation is deferred to avoid changing diffusion infrastructure.
The reviewers performed source inspection only. They did not execute tests.

### Parent validation

Parent monitor **2** completes with exit **0** in the primary checkout. The tests use CPU synthetic fixtures.
Environment: `PYTHONPATH=/home/agiuser/projects/mjlab/src`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `uv run --no-sync`.
Evidence: `/tmp/mjlab-flow-parent-acceptance-wCIecIa6/`.

| Check | Result |
| --- | --- |
| Seven `tests/test_tracking_flow_matching_*.py` files, `pytest -x -vv -o faulthandler_timeout=30` | **53 passed**, 5.17 s |
| Diffusion `{cli,train_cli,noising,sampler,model,trainer,checkpoint,schedule}` regressions, same pytest options | **62 passed**, 9.32 s |
| Ruff check, new source and tests | PASS |
| Ruff format check | PASS, 15 files |
| Ty check, new source and tests | PASS |
| Pyright, same scope | 0 errors, 0 warnings |
| Flow CLI help | PASS |
| Reviewed file comparison and SHA-256 verification | **19/19 files match** |
| `git diff --check` | PASS |

### History dropout arm (2026-10-13 17:15 +08)

Operator request: support history dropout in the flow family exactly as the diffusion family does.
A nonzero ``history_state_dropout_probability`` now raises the whole window's history state and
latent levels to ``process.max_level``, which is pure Gaussian on the linear path. The trainer
imports ``_history_dropout_mask`` from the diffusion trainer, so the two families cannot drift.
One draw per window comes from a dedicated generator whose state and counters are stored in the
flow checkpoint; a nonzero-dropout resume fails closed when that state is missing. Per-update,
per-epoch and summary reports carry ``history_state_dropout_windows``,
``history_state_total_windows`` and ``history_state_dropout_fraction``. Evaluation never applies
the mask, so held-out losses stay comparable across dropout settings, as in the diffusion family.

Bounded local GPU check, two arms on 32 real VGK36 training windows, 4 updates each
(one epoch at a batch budget of 128):
`logs/flow/flow-dropout-check2-20261013T172500+0800/dropout-check.json`.

| Arm | Probability | First-batch levels | Windows | Fraction | Checkpoint dropout RNG |
| --- | --- | --- | --- | --- | --- |
| `dropout0` | 0.0 | no mask; history levels sampled as before | 0 / 128 | 0.0 | present |
| `dropout100` | 1.0 | every history state and latent row at `max_level` | 128 / 128 | 1.0 | present |

The check is a software check of the mask and the reporting, not a quality result. It ran against
the source revision that is committed here. The earlier attempt in
`logs/flow/flow-dropout-check-20261013T172000+0800` produced the same arm values; its harness
expected 8 updates where one epoch yields 4, so its own final assertion failed after both arms
had already run.

### Dropout 20 % arm (2026-10-13 17:42 +08)

Operator request: launch a 20 % history-dropout arm against the baseline.

| Item | Value |
| --- | --- |
| Run | `logs/flow/flow-vgk36-dropout20-400k-seed0-20261013T094220Z` |
| tmux session | `flow_vgk36_dropout20`, container GPU 1 |
| Source | `49f7cd6b4` (includes the dropout change) |
| Dropout probability | 0.20 |
| Matched baseline | `flow-vgk36-400k-seed0-20261013T085536Z` on GPU 0 |
| First update | about 2026-10-13 17:47 (+08), after the same 11.8 GiB cache read |

The run config differs from the baseline config in exactly two lines: the contract path and the dropout
probability. Both configs live in their run directories and are hashed in the launch manifest.

Verified dropout behaviour on the running job:

| Observation | Value |
| --- | --- |
| Configured probability | 0.20 |
| Per-update fractions (first four updates) | 0.2227, 0.2012, 0.1719, 0.1992 |
| Epoch 1 | 92,436 of 464,039 windows, fraction **0.19919** |
| Epoch 2 | 92,873 of 464,039 windows, fraction **0.20013** |
| Epoch length | 907 optimizer updates, 3626 batches of 128, 464,039 windows |

Both trainers ran at about 7.2 updates/s while sharing the container, and each saturated one CPU core.
That is inside the rate spread measured for the baseline before the second arm existed, so there is no
evidence that the two runs slow each other.

Two reporting notes found while checking this run:

1. **The epoch row is built by spreading `**_average_metrics(epoch_updates)` last**, so any key that also
exists in the per-update metrics is replaced by its epoch mean instead of the explicit value written
just above it. Verified on the live dropout run: the epoch-1 row reports `global_step` 454 while the
epoch's last update was step 907, and 454 is exactly the mean of that epoch's update steps; the same
holds for epoch 2 (1361 = mean of 908..1814). The epoch-level `history_state_dropout_fraction` is also
the mean of the per-update fractions (0.19919064549585844) rather than the epoch ratio
(0.1991987742409582); the two agree to six decimals. `epoch`, `train_loss`, the window counters and
the `eval_model`/`eval_ema` blocks are unaffected.
   The diffusion trainer builds its epoch row the same way, and the finished D2 runs show the identical
   pattern (epoch 1 → 454, epoch 2 → 1361, epoch 3 → 2268), so this predates the flow family. Training,
   checkpoint selection and the authoritative records are unaffected: `train-report.json` and the
   checkpoint payloads carry the true step (the finished D2 runs record 400,000 updates over 441
   epochs). The only consumer that reads this field is `tools/diffusion_metrics_to_tensorboard.py`,
   which uses `row["global_step"]` as the x-coordinate for update **and** epoch scalars. Epoch curves
   therefore appear about half an epoch too early on the x-axis, with the offset growing per epoch;
   trends and relative comparisons are preserved. Tools that read `row["epoch"]`, such as
   `scripts/experiments/diffusion_wave_summary/report.py::read_validation_curve`, are correct.
   Fixed on 2026-10-13 in the local commit recorded below: the averaged block is spread first in both
the diffusion and the flow trainer, so an epoch row now keeps the epoch-boundary `global_step` and the
epoch-wide dropout ratio. Regression checks were added to `tests/test_tracking_diffusion_trainer.py`
and `tests/test_tracking_flow_matching_trainer.py`. This affects rows written by future runs: the two
live runs imported the earlier code, so their files keep the older mean-valued epoch rows, which is
internally consistent for each run. A resume of either run would load the fixed code.
2. The epoch window counters are correct and reconcile with 907 updates of 512 windows each.

### NAS baseline training run (in progress)

Authorized by the operator on 2026-10-13 (+08) as the flow arm of a matched comparison against the
diffusion baseline. Branch `fxy/test/flow-matching-baseline` was pushed to the `nas` remote
(`ssh://fushan@10.14.64.48:32221/home/fushan/mjlab`) and checked out there at `f7d6aee7f`.
Two untracked server copies of the VGK36 contract and collection config were byte-identical to the
branch versions; they were moved to `/home/fushan/remote-checkout-backups/flow-baseline-checkout-20261013T085504Z/`
before the checkout.

| Item | Value |
| --- | --- |
| Run | `logs/flow/flow-vgk36-400k-seed0-20261013T085536Z` |
| tmux session | `flow_vgk36_400k`, container GPU 0 (RTX 4090) |
| Dataset | `logs/diffusion/d1-vgk36-double-20261012` (published caches reused read-only) |
| Budget | 400,000 optimizer updates, seed 0 |
| Batch | 512 effective = 128 microbatch x 4 accumulation, BF16 |
| Optimizer | AdamW, lr 1e-4, warmup 10,000, cosine, weight decay 1e-3 |
| Matched baseline | `d2-vgk36-baseline-400k-seed0-retry1-20261012T092934Z` |
| Launch | 2026-10-13 16:55:36 (+08) |
| First update | about 2026-10-13 17:00 (+08), after cache validation read 11.8 GiB |
| Measured rate | 741 to 1998 updates in 180 s = **6.98 updates/s** (includes epoch validation) |
| Projected completion | about **16 h**, near 2026-10-14 09:00 (+08) |

Early progress at update 2,081 (2 epochs): train loss 0.4622, evaluation `unknown_mse` 0.4542 for the
model and 0.5192 for the EMA. These are early values, not acceptance evidence.

The flow and diffusion validation losses are **not** on a comparable scale, because one scores
x0 reconstruction under independent linear noise and the other scores DDPM clean-token error.
Any flow-versus-diffusion claim must compare generated-token metrics on the same windows.

**Source provenance of this run:** the process was launched at `f7d6aee7f` and that commit is
recorded in the run's `launch-manifest.json`. At 2026-10-13 18:05 (+08) the branch was pushed to
`d0172a947` (the history-dropout change), which updated the NAS working tree under the live job.
The running process keeps the modules it imported at startup, so its behaviour did not change; this
was confirmed by watching the update counter advance across an epoch boundary during and after the
push. A later resume of this run would load `d0172a947`, which for `probability == 0.0` draws no
extra randomness and is otherwise only additive in the report fields.

### Local GPU smoke on the real VGK36 dataset

Operator-authorized on 2026-10-13 (+08) with the local RTX 5080 Laptop GPU and the local dataset
`logs/diffusion/d1-vgk36-double-20261012` under the matched VGK36 contract.
Harness: `logs/flow/smoke-vgk36-local-20261013T164500+0800/smoke.py`. Passing run: monitor **5**, exit **0**.
The harness loads the persisted index, projects 64 real training and 16 real validation windows, and calls the
`FlowTrainer` and `evaluate_generation` APIs directly. It does not build a training token cache. The published
dataset files keep their size and modification time, and their checksums verify afterwards.

| Measurement | Result |
| --- | --- |
| Optimizer updates | **8/8**, BF16, microbatch 32, accumulation 2 |
| Sample draws | 512 |
| Windows | 64 train, 16 validation, motion `sh001` only |
| Training wall time, including validation and checkpoint writes | 2.988 s |
| Final training x0 loss | 1.0384657979011536 |
| Best validation x0 loss | 1.0729833841323853 |
| Peak CUDA allocated / reserved | 1.06 GB / 1.07 GB |
| Checkpoint reload | PASS, `checkpoint-last.pt` at step 8 |
| GPU generation | 4 validation windows, 20 sampler calls |
| Generated unknown token MSE | 2.2731 |
| `prior_only` baseline | 1.1999 |
| `zero_latent` baseline | 0.1905 |

The training path, checkpoint save/reload, and the 20-call GPU sampler all run on real data.
After only 8 updates the generated-token score is **worse** than both trivial baselines.
That is the expected result for an untrained network and it is **not** a quality or convergence finding.
The smoke is not a full cached training run, not a policy-quality result, and not closed-loop evidence.
Only one motion appears in the first windows, so this subset does not cover the 36-teacher cohort.

Two defects were found and fixed during this smoke:

1. API-level CUDA runs need `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Only the CLI sets it. The smoke launcher now sets it.
2. `euler_x0_update` built its level ratio on the CPU level device. GPU generation failed until the ratio follows
   the sample device and dtype. One CUDA-gated regression test covers this. CPU-only tests could not detect it.

A follow-up read of `add_linear_noise`, `draw_gaussian`, and `evaluate_generation` found no further device defect.
The formatting pass that followed the passing smoke changed whitespace and parentheses only. The smoke therefore
ran on equivalent logic, not on byte-identical text.

The first parent monitor fails before tests because `/bin/sh` rejects `pipefail`. The explicit Bash launcher fixes this setup issue.
The earlier integration startup failure stops after Pi cannot load `@earendil-works/chord`. The operator restarts Pi; a same-protocol retry completes integration.
One orchestration text check mistakes `No STAGE_BLOCKED condition` for a blocker. Parent reads the complete handoff and launches only the pending review.
Completed component implementations never rerun.

### Limits and next action

No production training, full cached dataset run, multi-motion evaluation, docs build, simulation, controller integration, ONNX export, or hardware test occurred.
The GPU smoke uses the direct trainer and evaluator APIs on 64 training windows of one motion.
Generator-state restoration is tested. Bit-identical uninterrupted versus mid-epoch resumed training is not established; resume restarts the epoch sampling order.
The package imports private diffusion helpers. Future helper changes can affect it.
The next experiment needs a chosen frozen dataset/VAE contract and an explicitly authorized update budget.
The example config names the original no-mixture data contract. Match its `contract` field and `--contract` to the selected dataset before use.

