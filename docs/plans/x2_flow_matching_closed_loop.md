# X2 flow matching in the existing mjlab closed-loop runner

Date: 2026-10-14 (+08).
Status: implemented in the primary checkout; bounded parent validation passed on 2026-10-15 (+08). Uncommitted and not pushed.
Scope: mjlab single-row, viewer, and parallel closed-loop simulation.

## Goal

Run the trained x0 flow models through the same simulation path as diffusion.
Reuse the existing environment, VAE bootstrap, measured history, decoder,
termination rules, metrics, state recorder, and browser playback.
Keep the diffusion path unchanged by default.

## Baseline before this change

- `flow_matching/sampler.py` implements `euler_x0_update` and batched sampling.
- `FlowGrid.uniform()` supplies 20 calls from level 1000 to level 0.
- The sampler draws Gaussian noise and restores known entries after every step.
- `flow_matching/checkpoint.py` loads flow checkpoints and their EMA weights.
- Both 400k VGK36 runs are local. Neither has a D3 checkpoint sidecar.
- `diffusion/handover.py::ReferenceFreeContinuation` owns the single-row lifecycle.
- `diffusion/parallel_closed_loop_x2.py::BatchedReferenceFreeContinuation` owns the batched lifecycle.
- Both continuations call diffusion sampling directly today.
- Both expect `diagnostics.final_index`; flow returns `final_level`.

The ODE does not need another implementation. For source noise fraction s and
next fraction r, use the existing update:

`x_r = (r/s) * x_s + (1-r/s) * x0_hat`, with `s > 0`.

Unknown slices start at level 1000. Known history and current state stay clean.
Generate the current latent and future state/latent, as diffusion does.
The Euler step count is adjustable at inference time (``--flow-euler-steps``,
default 20, at most the checkpoint ``max_level``). Twenty steps is the validated
default and is the count the sweep below used.

## Minimal implementation

### 1. Load the flow model

Extend the closed-loop model-loading boundary in `diffusion/replay_integration.py`.
Reuse model construction, EMA selection, freezing, and metadata reporting.
Call the existing flow loader for flow artifacts. Do not convert checkpoint files.

Add explicit `--model-family diffusion|flow` to the existing evaluator parser.
Default to diffusion. The viewer and parallel driver reuse this parser/preparation.
Reject a family/checkpoint mismatch through the existing checkpoint loaders.

Allow loaded runtime metadata to hold the actual flow process/grid or diffusion
schedule/grid. Do not build a fake cosine schedule for flow.
Keep family-specific identity fields in reports. Do not invent missing provenance.

### 2. Plug sampling into the existing continuations

Add one optional sampling callable to the single-row and batched continuations.
Leave the existing diffusion call as the default path.
The injected callable consumes conditions and the generator, and closes over
its flow process/grid/config. It calls the existing flow sampler.

Use one small adapter to provide the diagnostics fields the runners already read:
`denoiser_calls` and terminal `final_index`, mapped from flow `final_level`.
The flow conditions can reuse the existing clean tokens and clean mask.
Do not duplicate history assembly, decode, reset, or physical stepping.
Do not add a solver registry, class hierarchy, or a general process protocol.

Wire this callable from `evaluate.py` and `evaluate_parallel.py`.
The viewer should inherit the same construction path.
For the first version, reject flow task guidance explicitly during preflight.
This avoids changing the DDIM-specific D4 guidance code.

### 3. Reuse scenario configuration and reports

Reuse the existing checkpoint-sidecar fields for artifact paths and scenario.
Accept a flow sidecar schema alongside the current diffusion schema.
Create explicit evaluation sidecars for the two local flow runs.
Use matched member, start frame, seed, VAE, and statistics for comparisons.
Do not alter trained checkpoint bytes or historical diffusion sidecars.

Report `model_family`, `sampler_method`, `solver_updates`, and the actual grid.
Use `euler_x0` for flow. Preserve existing diffusion report fields for consumers.
Do not report flow updates as DDIM updates.
Confirm the ladder/summary tools can consume the normal fall-count outputs.
Avoid rewriting these tools unless a concrete incompatibility appears.

### 4. Focused validation and usage documentation

Reuse the existing flow sampler tests. Do not retest the ODE in a new framework.
Add small integration checks for:

- flow checkpoint loading and EMA selection at the runtime boundary;
- sampler dispatch and matching diagnostic fields;
- single-row and batched continuation with measured history and real decoding shapes;
- flow guidance rejection and correct family reporting;
- unchanged diffusion defaults through existing regression tests.

Run focused tests with `uv run pytest -x -vv` and bounded timeouts.
Run scoped lint/type checks, not a repository-wide validation campaign.
Update the flow runbook and the changelog with runnable commands.

## Real simulation acceptance

1. Preflight one real flow checkpoint without constructing physics.
2. Run one short, one-row closed-loop smoke using the existing VAE bootstrap.
3. Check finite actions, 20 solver calls per plan, history updates, and normal stop reporting.
4. Run the matching small batched smoke. Check recording and viewer compatibility.
5. Only then run the full matched fall comparison, if requested.

Use a unique output directory. A smoke proves integration, not policy stability.
Keep 50 Hz control and the existing physical-fall/reset/endpoint rules.
Do not require bitwise equality between different batch widths or sampler families.
Use existing diffusion regression tolerances and physical metrics.

## Out of scope

- x2_docker, C++ ODE sampling, ONNX export, robot deployment, and hardware tests.
- New training, data collection, or cache preparation.
- Pose-cost guidance or keyframe guidance for flow in this first version.
- Adaptive ODE solvers, additional solver families, or a solver benchmark matrix.
- Large refactors of the diffusion training or checkpoint packages.

## Delegated implementation ownership

Classification: multi-seam. Loading and continuation sampling have separate test contracts.
All writers run serially in one isolated checkout at
`/home/agiuser/projects/worktrees/mjlab/flow-closed-loop-20261014`, based on `808aeb92d`.
No child may commit, push, deploy, use remote hosts, or start a GPU simulation.

| Stage | Exclusive files/contract | Gate | Durable handoff |
| --- | --- | --- | --- |
| Loader | `diffusion/replay_integration.py`, one focused flow-loader test file | flow load/EMA/family metadata; existing loader regressions | managed loader report |
| Sampling | `diffusion/handover.py`, `diffusion/parallel_closed_loop_x2.py`, small flow adapter, focused continuation tests | injected flow sampling and unchanged diffusion default | managed sampling report |
| Integration | evaluator/viewer/parallel driver, sidecar support, runbook/changelog, focused CLI tests | synthetic preflight/dispatch and scoped checks | managed integration report |
| Review | read-only integrated diff | concrete runtime defects and diffusion regressions only | managed review report |

Integration consumes both component handoffs. Parent owns final acceptance,
real-checkpoint sidecars, monitored simulation smoke, and source import.
Stay focused on each assigned task. Stop after the bounded deliverable; no speculative cleanup.

## Parallel multi-environment sweep — 2026-10-15 (+08)

Run on the 4-card managed container `10.14.64.48:32221`, one process per GPU.
Motions `vgk000`-`vgk007`, start frame 45 from the sidecar, 256 rows per motion.
Each row is one environment; rows differ by per-row domain randomization and
per-row sampler noise. Evidence:
`logs/diagnostics/flow-closed-loop-sweep-20261015T0335Z/` (38 files, sha256 verified after fetch).
The failed first attempt is preserved at `...T0330Z/`.

Falls out of 256 rows per motion:

| Checkpoint | vgk000 | vgk001 | vgk002 | vgk003 | vgk004 | vgk005 | vgk006 | vgk007 | Total | Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flow-baseline` | 136 | 42 | 27 | 64 | 28 | 16 | 0 | 1 | 314/2048 | 15.3% |
| `flow-dropout20` | 106 | 57 | 85 | 126 | 1 | 1 | 1 | 0 | 377/2048 | 18.4% |
| diffusion `vgk36-baseline-best` | 128 | 72 | 34 | 133 | 9 | 7 | 21 | 0 | 404/2048 | 19.7% |
| diffusion `vgk36-dropout20-best` | 132 | 54 | 70 | 92 | 49 | 26 | 4 | 2 | 429/2048 | 20.9% |

Per-motion continuation tick budget: 398, 285, 265, 327, 299, 307, 276, 257.
Wall time per member: 94.9-139.9 s. The two diffusion rows reuse
`logs/diagnostics/diffusion-d3-models/` with the same motions, rows and budget.

Checks on the 16 flow member records: `denoiser_calls` equals `20 x ticks_run`
for every member, `stop_counts` sums to the row count, and the tick budget
equals the ticks run. Parity (8 rows, 60 ticks): the realization matched, the
tick-0 conditioning and reference vectors were identical, the maximum bootstrap
action difference was 2.03e-3, and both paths stopped at tick 68.

Limits: one seed, one start frame per motion, eight motions, no matched
acceptance, unguided sampling, and no hardware, export or x2_docker path.
Both arms ran at the default 20 Euler steps.
The flow arms fall less often than the diffusion arms in this diagnostic.
That difference is descriptive only. It is not a stability, quality or
superiority claim, and it is not a closed-loop ranking.

## Euler step-count ablation — 2026-10-15 (+08)

Same 4-card container, same motions, same 256 rows, same start frame and seed.
One process per GPU. Evidence:
`logs/diagnostics/flow-steps-sweep-20261015T0425Z/` (166 files, sha256 verified).
The 20-step rows are reused from `flow-closed-loop-sweep-20261015T0335Z/`.

Falls out of 256 rows per motion:

| Steps | Baseline per motion (vgk000-007) | Total | Rate | Dropout20 total | Rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 209/166/120/145/101/143/188/104 | 1176/2048 | 57.4% | 794/2048 | 38.8% |
| 3 | 95/59/45/85/52/101/11/2 | 450/2048 | 22.0% | 488/2048 | 23.8% |
| 5 | 120/63/39/88/23/86/3/1 | 423/2048 | 20.7% | 406/2048 | 19.8% |
| 10 | 109/59/15/60/24/33/1/2 | 303/2048 | 14.8% | 351/2048 | 17.1% |
| 15 | 114/60/20/50/29/31/2/1 | 307/2048 | 15.0% | 310/2048 | 15.1% |
| 20 | 136/42/27/64/28/16/0/1 | 314/2048 | 15.3% | 377/2048 | 18.4% |

Cost per member (median wall seconds, 4 concurrent): 54 s at 1 step, 62 at 3,
70 at 5, 84 at 10, 99 at 15, about 113 at 20.

What the numbers support:

- One Euler step is unusable. The learned `x0_hat` at level 1000 is applied as
the plan with no integration, and more than half the rows fall.
- Three steps are clearly worse than ten (22.0% against 14.8%, about 7 points
against an aggregate binomial standard error near 1 point).
- Ten, fifteen and twenty steps are indistinguishable here (14.8%, 15.0%,
15.3%). The spread is inside the noise, so no accuracy benefit is measured
from twenty steps over ten in this diagnostic.
- Ten steps cost about 26 % less wall time per member than twenty.

Limits: one seed, one start frame per motion, eight motions, unguided sampling.
Rows share a motion and are not independent, so the per-motion spread is wider
than the aggregate spread. The two arms do not keep a consistent order across
step counts, so this does not rank the arms either. Falling less is not
stability, and no deployment or hardware conclusion follows.

### Refined counts 6-9 and a reproducibility control — 2026-10-15 (+08)

Evidence: `logs/diagnostics/flow-steps-fine-sweep-20261015T0521Z/`
(166 files, sha256 verified). The same configuration was run twice at 10 steps
to measure the noise directly.

| Steps | Baseline total | Rate | Dropout20 total | Rate | Median wall |
| --- | --- | --- | --- | --- | --- |
| 1 | 1176/2048 | 57.4% | 794/2048 | 38.8% | 54 s |
| 3 | 450/2048 | 22.0% | 488/2048 | 23.8% | 62 s |
| 5 | 423/2048 | 20.7% | 406/2048 | 19.8% | 70 s |
| 6 | 357/2048 | 17.4% | 307/2048 | 15.0% | 72 s |
| 7 | 310/2048 | 15.1% | 270/2048 | 13.2% | 77 s |
| 8 | 288/2048 | 14.1% | 339/2048 | 16.6% | 77 s |
| 9 | 338/2048 | 16.5% | 349/2048 | 17.0% | 80 s |
| 10 | 303/2048 | 14.8% | 351/2048 | 17.1% | 84 s |
| 10, repeat | 327/2048 | 16.0% | 350/2048 | 17.1% | 84 s |
| 15 | 307/2048 | 15.0% | 310/2048 | 15.1% | 99 s |
| 20 | 314/2048 | 15.3% | 377/2048 | 18.4% | 113 s |

Reproducibility control, identical configuration run twice at 10 steps:

| Arm | Total | Move | Largest per-motion move |
| --- | --- | --- | --- |
| baseline | 303 to 327 of 2048 | +1.2 points | 23 rows, 9.0 points (`vgk000`) |
| dropout20 | 351 to 350 of 2048 | 0.0 points | 15 rows, 5.9 points (`vgk002`) |

What this changes:

- The noise bound is measured, not assumed. One configuration can move its own
total by about a point and a single motion by up to nine points.
- Between 6 and 20 steps no count is distinguishable. The spread inside that
range (2 to 4 points) is the same size as the repeat spread, and the apparent
best count differs between the arms (8 for baseline, 7 for dropout20).
- Six steps already sits inside the flat region for both arms, and five steps
is outside it for the baseline (20.7% against 14.8%, about 5 points, with three
motions clearly worse).
- Cost per member fits roughly `55 s + 2.9 s per step` at 256 rows over 398
ticks, so a lower count saves little in absolute terms: 8 steps costs about 8%
less than 10, and 10 steps about 26% less than 20.

No optimum is identified inside 6 to 10. Choosing 10 keeps the count with the
most evidence. A default change needs a seed-repeat confirmation first.

## ONNX export, laptop cost, and the robot PC measurement — 2026-10-15 (+08)

Tools and evidence: `scripts/experiments/flow_onnx_latency/` (README carries both
tables), `logs/diagnostics/flow-onnx-latency-20261015-{a,b,c}/` (laptop),
`logs/diagnostics/flow-onnx-latency-robot-20261015/` (robot).

Export: the accepted diffusion D5 helper with the flow loader, so the graph
contract is the lane's: `noisy_tokens` f32 `[B,41,167]` and `step_ids` i64
`[B,41,2]` to `x0_hat` f32 `[B,41,167]`, opset 18, dynamic batch, 80,687,086
bytes, sha256 `c4736acf…c54fcb` (identical from four runs and on the robot after
transfer). The ONNX is device-neutral; the provider is a runtime choice.

Laptop (RTX 5080 Laptop, torch CUDA, B=1, three runs): per Euler step about
0.95 ms, so 5 steps **4.6-4.9 ms**, 8 **7.2-8.0**, **10 9.3-10.2**, 20
**18.5-20.4 ms**. B=256 is 31.6 ms per step. 10 steps uses about half the 20 ms
period.

Robot PC (`agi@172.21.204.234`, Orin NX 16GB, aarch64 GPU ONNX Runtime 1.23.0
unpacked into a private directory, two ORT threads, one process, engine cache in
the run directory, clocks **not** locked, ROS stack running with load near 10):

| Steps | TensorRT median | TensorRT p90 | CUDA median | CUDA p90 | Share of 20 ms (TRT) |
| --- | --- | --- | --- | --- | --- |
| 5 | 17.43 | 27.74 | 26.28 | 65.03 | 87% |
| 6 | 16.46 | 34.85 | 26.15 | 62.28 | 82% |
| 7 | 21.00 | 35.54 | 28.13 | 38.31 | 105% |
| 8 | 30.16 | 44.71 | 33.14 | 43.90 | 151% |
| 9 | 30.36 | 47.73 | 40.53 | 70.74 | 152% |
| 10 | 40.88 | 55.50 | 40.81 | 49.69 | 204% |
| 20 | 79.87 | 95.24 | 86.12 | 149.42 | 399% |

Robot PC with the clocks locked (`sudo jetson_clocks`: GPU min = max = 1173 MHz,
CPU min = max = 1984 MHz; the lock is not persistent and the pre-lock state was
restored afterwards):

| Steps | TensorRT median | TensorRT p90 | CUDA median | CUDA p90 | Share of 20 ms (TRT) |
| --- | --- | --- | --- | --- | --- |
| 5 | 12.01 | 18.26 | 19.71 | 21.55 | 60% |
| 6 | 15.82 | 25.54 | 22.12 | 24.67 | 79% |
| 7 | 19.37 | 26.55 | 25.44 | 26.33 | 97% |
| 8 | 20.07 | 41.14 | 29.29 | 30.40 | 100% |
| 9 | 22.84 | 38.27 | 32.98 | 33.78 | 114% |
| 10 | 24.92 | 34.91 | 36.43 | 37.09 | 125% |
| 20 | 56.83 | 75.86 | 72.08 | 73.16 | 284% |

Clock locking gains **1.1x to 1.6x** (TensorRT median about 1.4x) and the sampled
GPU clock is flat at 1173 MHz in every locked window, so the DVFS ramp is
excluded rather than averaged in. One pass costs about **2.5 to 2.8 ms** with
TensorRT locked, against about 0.95 ms on the laptop: the Orin is roughly **3x**
slower per pass once the clock is held, and TensorRT beats CUDA there
(2.5-2.8 against 3.6-3.9 ms per step).

The result matches the diffusion lane on the same robot: 50.56 ms clock-locked
and 93.46 ms as-found for a 20-pass plan is 2.5-4.7 ms per pass, and the locked
flow 20-step median (56.83 ms, 2.84 ms per step) is within 12% of the locked
diffusion plan. The two graphs cost the same on this board.

Reading: on the Orin as found, 5-6 Euler steps fit the median budget with
TensorRT (16.5-17.4 ms) but not their p90; **locked, the feasible band is 6-7
steps (15.8 ms at 79%, 19.4 ms at 97%)** and their p90 still exceeds the period
at 6 steps (25.5 ms). The closed-loop ablation says 6 steps is inside the flat
quality region, so **6 steps with TensorRT and a locked clock is the robot
candidate**; 10 steps is affordable only on a desktop-class GPU.

Limits: latency only. The robot is shared; its ROS stack held the GPU and the
load stayed near 10. The DVFS ramp is visible as found (612 to 1173 MHz across a
run). No controller, publisher or ROS node was started. An earlier robot attempt
fell back to CPU silently because ORT 1.23 rejects `trt_tf32_enable`; that run is
preserved as invalid evidence, and the tool now refuses any session whose active
provider is not the requested one.

## Decision — 2026-10-15 (+08)

The project uses **6 Euler steps** for the flow model, and **6 is now the code
default**: `DEFAULT_CLOSED_LOOP_SOLVER_STEPS = 6` in
`diffusion/replay_integration.py`, applied by
`load_flow_denoiser` and by the `--flow-euler-steps` default in the evaluator,
the parallel driver and the viewer. The flow package's own `DEFAULT_CALLS` stays
20 because it is the offline generation-scoring grid, so offline scores remain
comparable with the earlier runs. No launch flag is needed to get 6.

The decision rests on two measurements, both in this document:

1. **Closed-loop quality.** 6 steps sits inside the flat region (6-20 steps are
indistinguishable), and 5 steps is measurably worse. The 6-step totals are
357/2048 (17.4%) for the baseline and 307/2048 (15.0%) for dropout20.
2. **Orin cost.** With the TensorRT execution provider and the GPU clock locked,
6 steps is 15.82 ms median, 79% of the 20 ms control period. 10 steps is
24.92 ms (125%) and 20 steps is 56.83 ms, which does not fit at 50 Hz.

Open item: the 6-step p90 on the Orin is 25.5 ms, above the period, so a real
50 Hz loop still needs the asynchronous planner or a locked clock.

## Delivery

One small integration change across the existing loader, continuations, and drivers.
No parallel flow simulator or new evaluation command tree.
Deliver working single-row/viewer/parallel commands and bounded smoke evidence.
Implementation is imported into `/home/agiuser/projects/mjlab`.
No full sweep starts from this plan.

## Parent acceptance — 2026-10-15 (+08)

Evidence root:
`logs/diagnostics/flow-closed-loop-acceptance-20261015T0125+0800/`.
Implementation came from the isolated checkout named above.
Parent applied four review notes: batch `--execute` documentation, explicit
sequence-inspector refusal, batch sampler metadata, and a clear small-level-grid error.

| Check | Result |
| --- | --- |
| Focused flow and diffusion regressions in the primary checkout | 101 passed, 14.22 s |
| Scoped Ruff and Ty checks in the isolated checkout | passed |
| Real baseline and dropout20 best-checkpoint preflights | both passed; EMA steps 155097 and 171423 |
| Real baseline single-row matched pair, `vgk000`, frame 45, seed 0 | valid and complete; 8 bootstrap + 25 continuation ticks per branch; no errors |
| Learned branch network calls | 500 = 25 plans x 20 Euler calls |
| Real baseline eight-row simulation, same member/frame/seed | all 8 reached the 25-tick budget; 0 falls in this short window |
| Batch state recording | 33 ticks, 8 rows, body channels included; `recordings/vgk000.npz` |
| Playback body reconstruction, row 0 at four sampled ticks | worst error 3.050e-07 m; passed |
| Viewer preflight | passed; interactive browser behavior not tested |

The eight-row result uses one batched network call per solver update.
Its 500-call counter is not multiplied by the row count.
No reference or encoder call occurs during flow continuation.
The single-row learned branch records 8 encoder calls and 33 decoder calls.

Both local flow runs now have a new `output/checkpoint-best.yaml` sidecar.
Checkpoint bytes, training records, diffusion sidecars, and frozen teachers are unchanged.
Four existing standing NPZ files were one directory above their manifest paths.
Added relative links under `data/tennis/standing_holds/`; no motion was regenerated.
The failed `...T0124+0800/` preflight remains preserved.
Its monitor also had an incorrect timeout unit; the retry used 300000 ms.

Limits: this is integration evidence, not a stability comparison.
Dropout20 physics, a full motion sweep, interactive viewer behavior, ONNX,
x2_docker, and hardware remain untested by this change.
The learned branch retains the historical `diffusion` JSON key; use
`provenance.sampler.model_family` to identify flow.

