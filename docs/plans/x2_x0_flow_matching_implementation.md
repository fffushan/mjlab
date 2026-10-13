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
- History dropout is forbidden for this family. Reject nonzero settings, rather than ignoring them.
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
- Flow configuration rejects any nonzero history dropout.
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

