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

D5 is not complete. The remaining gates are:

### 1. Controller warmup and lifecycle

The `mjlab` runtime warmup method is implemented and CPU-tested.
The remaining controller-side work is:

- Call warmup before tracking becomes active.
- Define behavior for warmup failure, worker exit, stale plans and shutdown.
- Add tests for no command publication during warmup.
- Add tests for current-state revalidation after warmup.
- Preserve the eight measured bootstrap pairs and the no synthetic
  standing/zero-action fallback contract.

### 2. Sustained runtime qualification

Run a longer warmed actual-model workload.
The workload must report:

- end-to-end tick duration;
- action coverage for every scheduled tick;
- planner completion and first-use latency;
- pending replacement count;
- waiting and plan-exhaustion reasons;
- stale-plan and reset behavior;
- unguided and opt-in guided paths;
- exact worker cleanup.

The active contract requires a 20 ms end-to-end deadline.
The test must not silently reduce the 50 Hz rate, 41-step window, 32-step horizon,
backbone or 20 DDIM updates.
The current evidence does not meet this gate.

### 3. Numerical precision and export parity

Implement and validate the selected deployment representation for the active
167-channel model.
The remaining checks include:

- FP32 versus BF16/FP16 numerical parity;
- denoiser input/output shape and mask semantics;
- 20-update DDIM parity;
- statistics-only normalization parity;
- frozen VAE decoder parity;
- exported artifact identity and load-time contract checks.

No denoiser deployment artifact has been accepted yet.
The historical 231-channel export notes must not be applied to the active
167-channel no-mixture model without a new contract check.

### 4. Vendor simulator integration

Run the independent vendor-MuJoCo sim2sim gate after the runtime and export
contracts pass.
The gate must include:

- the actual controller handover;
- stale-plan handling;
- injected delay and dropout cases;
- safe release and cleanup;
- action and state dimension checks;
- unguided and, if enabled, guided paths;
- a residual-risk report.

No vendor simulator or physics run has been performed for the D5 model-backed
runtime.

### 5. Versioned runtime bundle and residual-risk record

Package the accepted runtime, model artifacts, normalizer, VAE interface and
provenance as one versioned bundle.
Record the exact source, artifact, contract and statistics identities.
Keep software timing, simulator evidence and hardware readiness as separate
claims.

## Outside D5

The following work is D6 or later, not a D5 completion condition:

- hardware state-estimator validation;
- communication and target-topology validation;
- physical safe handover and fallback validation;
- bounded robot trials;
- hardware stability or deployment acceptance.

## Status boundary

Completed D5-related evidence:

- D5a latent planner, bounded executor and CPU runtime: accepted.
- Direct model-backed loader and runtime wiring: implemented and scoped checks pass.
- Actual checkpoint, statistics-only normalizer, frozen VAE and spawned GPU
  worker: loaded successfully on the local GPU.
- Cold-start and request-replacement behavior: measured.
- Owner decoder warmup effect: measured.
- `AsyncLatentRuntime.warmup`: implemented and CPU-tested. It warms valid
  owner-decoder calls and one spawned planner result, then discards the plan.

Not completed:

- Controller-side warmup integration remains unimplemented.
- Sustained 50 Hz action-coverage qualification;
- numerical/export parity gate;
- vendor-MuJoCo sim2sim gate;
- versioned deployment runtime bundle.

No source files were changed by the investigation runs.
