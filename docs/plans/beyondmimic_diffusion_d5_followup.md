# D5 follow-up note: warmup, sustained timing and deployment gates

Date: 2026-10-07 (+08)

## Purpose

This note records the deferred D5 runtime investigations.
It does not authorize a controller launch, simulator run, export, deployment,
remote job or hardware operation.

The accepted D5a CPU runtime and the statistics-only model-backed loader remain
unchanged. The local GPU runs below are diagnostic evidence only.

## Evidence

- [Local GPU smoke report](../../logs/diagnostics/diffusion-d5-gpu-local-20261007/REPORT.md)
- [Cold traced run](../../logs/diagnostics/diffusion-d5-gpu-local-20261007/latency-investigation.json)
- [Warm decoder run](../../logs/diagnostics/diffusion-d5-gpu-local-20261007/warm-decoder-investigation.json)
- [D5 runtime plan](beyondmimic_diffusion_d5_runtime_adapter_executor.md)
- [Milestone gates](beyondmimic_diffusion_reproduction.md#6-milestones-and-acceptance-gates)
- [Active no-mixture contract](beyondmimic_diffusion_d1_no_mixtures_contract.yaml)

## Observed behavior

The cold traced run used the actual 167-channel denoiser, the statistics-only
normalizer, the frozen VAE and a spawned GPU worker on the local RTX 5080.
It created 96 requests, submitted 61, completed 60 and replaced 35 pending
requests before submission.

The scheduler has one active request and one pending request.
A newer pending request replaces an older pending request.
This is bounded latest-request-wins behavior, not an unbounded queue.

Planner result availability latency was:

- p50: 20.26 ms
- p95: 40.23 ms
- p99: 164.71 ms
- maximum: 172.62 ms

The first two results were 159.21 ms and 172.62 ms.
After those results, the maximum was 40.44 ms.
These are submit-to-owner-availability values.
They include control-tick polling alignment.
They are not isolated denoiser kernel measurements.

The cold run returned 85 actions from 96 control ticks.
It returned 10 waiting-for-plan statuses and one plan-exhausted status.
The initial waiting interval was ticks 8 through 15.
The plan-exhausted tick was tick 22.
Under the pre-change hard-limit implementation, the accepted plan had exceeded
`max_fallback_ticks=4`.
Ticks 23 and 24 then waited for the next plan.

The warm decoder run called the owner-side decoder three times before
measurement. The call times were 158.90 ms, 0.48 ms and 0.30 ms.
After this warmup, action-step latency was:

- p50: 3.06 ms
- p95: 3.42 ms
- p99: 3.52 ms
- maximum: 3.58 ms

The cold traced run had action-step p50 3.17 ms, p95 4.34 ms, p99 31.79 ms,
and maximum 172.26 ms.

The warm decoder run still had one planner first-result latency of 138.90 ms.
After that first result, planner latency had p50 19.94 ms, p95 23.23 ms,
p99 39.88 ms and maximum 40.03 ms.

These results identify two startup costs:

1. The owner-side VAE decoder has a first-call CUDA cost.
2. The spawned denoiser worker has a first-inference cost.

Warming the decoder removes the large owner-side action spike.
It does not remove the worker's first inference cost.

## Fallback policy change

The hard `max_fallback_ticks` action limit was removed after this investigation.
The parameter remains as a warning threshold for compatibility.
The scheduler emits one `FALLBACK_WARNING` event per accepted plan when the
threshold is reached.
It continues selecting later rows from the accepted plan until a newer plan
replaces it or the plan reaches `valid_until_tick`.
Trajectory-horizon exhaustion is the fallback-specific no-action condition.
No-plan, future, stale, invalid, reset, policy-change and shutdown cases remain
fail-closed no-action cases.

## Deferred controller warmup design

Add an explicit controller state before tracking becomes active:

```text
LOADING -> WARMING -> READY -> ACTIVE
```

The warmup phase must:

1. Load the owner-side VAE and the spawned denoiser worker.
2. Run several valid owner-side VAE decoder calls.
3. Submit and receive at least one valid planner request in the worker.
4. Synchronize the CUDA work that the measured loop will use.
5. Validate tensor shapes, finite values, identities, source frame and cleanup state.
6. Start tracking only after a valid current plan is available, or rebuild a plan
   from the latest state after the warmup period.

The warmup result must not become a robot command.
The controller must use its separately validated safe hold or disengaged state
while it is warming.
The runtime must not use a synthetic zero-action, standing or fall fallback.

The `mjlab` runtime now provides `AsyncLatentRuntime.warmup`.
It runs valid decoder warmup calls, waits for one valid spawned-worker plan,
validates and discards that plan, then resets scheduler state.
The `x2_docker` controller has not been changed.

Warmup alone is not a 50 Hz qualification.
The worker steady-state availability interval remains near 20–40 ms.

## Remaining D5 work

D5 is a research-reproduction runtime gate.
It is not a production reliability program.
Keep one essential test or measurement for each failure mode.
Do not add exhaustive state-machine matrices, transport fuzzing, or broad
regression suites as D5 completion conditions.

### Completed D5 software slice

The CPU controller lifecycle and execution bridge are accepted.
The essential CPU evidence is:

- one lifecycle smoke path: load, bootstrap, warmup, activate, step,
  acknowledge and release;
- one real spawned `LatentPlanner.process` path;
- warmup suppression;
- failed acknowledgement does not update history;
- stale/task-invalid work is rejected;
- reset/release cleanup is confirmed.

The larger existing test suite is retained as optional hardening evidence.
It is not a required D5 gate.

### 1. Sustained actual-model timing

Run one longer warmed workload with the actual denoiser, normalizer and VAE.
Report only the measurements needed to decide whether the research runtime can
meet its target:

- scheduled ticks and actions produced;
- end-to-end tick duration, including planner, decode and scheduling;
- planner first-result latency and replacement count;
- plan exhaustion or warning count;
- one reset/stale-work check;
- exact worker cleanup.

The target is a 50 Hz owner main loop with a 20 ms owner action-step budget.
A new planner result does not need to arrive every 20 ms when a valid accepted
trajectory remains available for reuse.
Keep the 41-step window, 32-step horizon, backbone and 20 DDIM updates.
Do not silently reduce the model or owner rate to obtain a pass.
Planner latency and plan replacement remain required measurements and residual
risks, not separate hard-real-time replanning gates.
The 2026-10-07 local RTX 5080 run met this owner-loop criterion: 240 paced ticks
at 49.9984 Hz and owner-step p99 5.3948 ms.
It recorded 239/240 actions because the first production tick after warmup had
no accepted production plan; this startup handover limitation is preserved but
does not block the asynchronous owner-loop acceptance.
This is a timing qualification, not a stability or quality claim.
Guided and unguided timing do not both need qualification; qualify the selected
paper-reproduction path and label the other path unmeasured.

### 2. Numerical and export parity

Validate one deployment representation for the active 167-channel model.
The essential checks are:

- tensor shapes and clean-mask semantics;
- statistics-only normalization;
- 20-update DDIM output parity;
- frozen VAE decoder action parity;
- exported artifact load and contract identity.

FP32 versus BF16/FP16, compiled inference, CUDA graphs and TensorRT are optional
engineering experiments. They are not D5 completion gates unless the selected
deployment representation uses them.
The historical 231-channel export notes cannot be reused without a new 167-channel
contract check.

### 3. Minimal vendor-MuJoCo sim2sim check

Run one nominal actual-controller handover with the selected D5 artifact.
The essential checks are:

- 167/99/31 dimension and frame contract;
- action publication and current-state decoding;
- one stale-plan or plan-exhaustion case;
- explicit release and exact process cleanup;
- a short residual-risk record.

Delay sweeps, dropout matrices, guided-path comparison and repeated motion grids
are optional diagnostics. They are not D5 completion gates.
No simulator stability or hardware claim follows from this check.

### 4. Versioned runtime bundle

Package the accepted runtime, selected model artifact, statistics-only normalizer,
VAE interface/artifact, active contract and hashes as one versioned bundle.
Record the timing result, sim2sim result and unresolved risks separately.
Do not call the bundle deployment-ready.

## Outside D5

The following work is D6 or later and has a separate plan:

- hardware state-estimator and communication validation;
- actual target topology and command mapping;
- operator-authorized physical handover and fallback;
- a small bounded robot trial.

D5 success does not authorize robot operation.

## Status boundary

Completed D5-related evidence:

- D5a latent planner, bounded executor and CPU runtime: accepted.
- Direct model-backed loader and runtime wiring: implemented and scoped checks pass.
- Controller lifecycle and execution bridge: accepted for the CPU software slice
  and integrated into primary, uncommitted. Parent validation passed 27 new
  controller cases and 132 adjacent cases. Independent recheck: OK with notes,
  no P0/P1. The integration test uses a real spawned worker and the actual
  `LatentPlanner.process` callback. [Report](../../logs/diagnostics/diffusion-d5-controller-cpu-20261007/REPORT.md).
  This does not establish physical safety or timing readiness.
- Actual checkpoint, statistics-only normalizer, frozen VAE and spawned GPU
  worker: loaded successfully on the local GPU.
- Cold-start and request-replacement behavior: measured.
- Owner decoder warmup effect: measured.
- `AsyncLatentRuntime.warmup`: implemented and CPU-tested. It warms valid
  owner-decoder calls and one spawned planner result, then discards the plan.
- Sustained actual-model owner-loop timing: **accepted under the asynchronous
  trajectory-reuse criterion**. The local RTX 5080 run completed 240 paced
  ticks at 49.9984 Hz; owner-step p99 was 5.3948 ms. Planner p95/p99 was
  37.8172/39.7540 ms, with 13 requests superseded before submission. One first
  post-warmup tick waited for a production plan. This remains a residual
  startup and planner-latency risk, not a 50 Hz owner-loop blocker under the
  accepted criterion. [Report](../../logs/diagnostics/diffusion-d5-gpu-sustained-20261007/REPORT.md).

Not completed:

- Essential numerical/export parity;
- minimal vendor-MuJoCo sim2sim check;
- versioned runtime bundle and residual-risk record.

The broader controller test matrix is optional hardening.
The deferred controller cleanup notes are not D5 completion gates.

No source files were changed by the historical GPU investigation runs.
The controller source and tests are now integrated and accepted for CPU software.
The observation and execution boundaries remain caller-supplied interfaces.
No native simulator/controller loop, deployment or hardware was activated.
