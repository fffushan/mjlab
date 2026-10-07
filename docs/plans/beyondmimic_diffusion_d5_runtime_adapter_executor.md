# D5a — latent planner, executor and CPU integration

Authorization: 2026-10-06. The parent is implementing the concrete backend and
runtime wiring directly after the delegated reconnaissance wave timed out.

The D5 model-backed reproduction path does not require source-file or artifact
SHA values supplied by the caller. Hash fields remain optional provenance. The
loader still checks the active schema, dimensions, tensor shapes, finite values,
and model/VAE interfaces; missing or malformed assets fail at load time.

## Scope

Implement the latent-plan adapter, a concrete bounded executor, a concrete
model-backed reproduction loader and their CPU integration. Preserve the
accepted D3/D4 paths and learned artifacts. Do not run GPU, physics, remote,
controller, export, vendor-simulator or hardware jobs. Do not retrain. Do not
commit, push, stage or install packages.

It remains isolated. Its action-vector selector is not a complete latent runtime.
[Scheduler report](../../../worktrees/mjlab/d5-scheduler-20261006/logs/diagnostics/diffusion-d5-scheduler-direct-fixes-20261006/REPORT.md).

## Source and isolation

Current source base: `3d6a8e5b72a81de9f3332e83ac1b3c469d51c98d`.
This ID records the base. It does not authorize or gate execution.
Primary: `/home/agiuser/projects/mjlab`.
Preserve its dirty D4 documents and unrelated untracked files.

Parent-created worktrees:

- `/home/agiuser/projects/worktrees/mjlab/d5-runtime-20261006/latent`
- `/home/agiuser/projects/worktrees/mjlab/d5-runtime-20261006/executor`
- `/home/agiuser/projects/worktrees/mjlab/d5-runtime-20261006/integration`

Each worktree contains an independent copy of the directly fixed scheduler and
its tests. The original scheduler worktree remains unchanged.
The lanes share installed dependencies, not writable project source.
Use `uv run --no-sync`. Set `PYTHONPATH` to the lane's `src` and `tests`.
Set `CUDA_VISIBLE_DEVICES=''` and `PYTHONDONTWRITEBYTECODE=1` in each test shell.
Do not sync or modify the shared environment.

## Implementation topology

This is multi-seam work. Model sampling/normalization and worker resource
ownership have independent correctness contracts.

| Lane | Exclusive ownership | Gate and handoff |
| --- | --- | --- |
| Latent | `latent_planning.py`; narrow request-context additions to `scheduler.py`; `test_tracking_diffusion_latent_planning.py`; additive scheduler regressions | CPU sampling parity, owned context, raw 32-D sequences, unchanged scheduler tests; durable source and report |
| Executor | `planner_executor.py`; `test_tracking_diffusion_planner_executor.py` | Concrete one-worker submit/poll/shutdown, bounded work and exact cleanup; durable source and report |
| Integration | Copy explicit component files after both gates pass; `async_latent_runtime.py`; `test_tracking_diffusion_async_latent_runtime.py`; D5 README; scoped changelog | Tick-aligned selection, current 99-D decode, executed history, source-age gate, fake-clock and real CPU-worker checks |
| Review | Read-only integrated source and tests | Cited P0/P1/P2 findings; standalone and runtime verdicts kept separate |
| Parent | Plan, routing, status, final validation and acceptance | No edits inside active writer lanes; no automatic primary publication |

Every lane has one writer. The integration owner starts after both component
handoffs are complete. Only the integration owner copies component files.
Children do not delegate. They ask the parent before changing a shared seam.

## Shared worker interface

Preserve the scheduler protocol: `submit(PlannerRequest)`, nonblocking `poll()`
returning `PlannerResult`, and exact/idempotent `shutdown()`.

The latent owner may add an optional `context` field to `PlannerRequest` and an
optional context argument to `RuntimeScheduler.request`. Existing callers keep
identical defaults. The context must be an owned, serializable snapshot. It may
contain clean masks, source frame, task targets/revision and request-local RNG
state. It must not contain live simulator, decoder or mutable model objects.
No unbounded request-to-context registry is permitted.

The executor transports the full request, including optional context. It must
not depend on the latent module. Its callable/factory returns a base `Plan`.
Use a narrow explicit wire representation for `Plan` and `PlannerResult` if
needed. `Plan.schema_metadata` is a MappingProxyType and is not directly
pickleable. Do not add global pickle monkey patches.

The generic selector may internally transport latent vectors in `Plan.actions`,
but the latent adapter must label and validate this representation explicitly.
It must expose typed raw-latent outputs. It must never publish these vectors as
robot commands. No predecoded future action is permitted in the D5 facade.

## Latent planning contract

- Active contract: `mjlab-x2-state-latent-d1-no-mixtures-v1`.
- Window: 8 measured executed pairs, current measured state, 32 future rows.
- Dimensions: 41 x 167; physical state 135; normalized latent 32.
- Reuse `build_reference_free_plan`, existing sampler, schedule, projection and
  D4 inpainting/guidance code. Do not duplicate their mathematics.
- Preserve all 20 default DDIM jumps, eta=0 and exact fixed masks.
- Planning must not call the encoder or decoder after bootstrap.
- Return raw, denormalized latent rows for source tick through source tick + 32.
- Source tick, timestamp, period, root frame, contract, VAE/projection/model
  identities and task revision travel with the plan. Reject incompatible data.
- Use request-local random state. Preserve caller/model/normalizer/RNG state.
- Task snapshots retain absolute target ticks. A pending request cannot relabel
  the active request's goals or source frame. Do not send a mutable scoring
  ledger to the worker. Record realized target errors only in the owner thread.
- Support unguided, hard-keyframe, pose-guided and combined callable paths.

CPU acceptance uses stub denoisers and frozen synthetic statistics. Fixed-noise
samples must match the synchronous D3/D4 mathematics. No real assets are needed.

## Executor ownership contract

Use one persistent, explicitly owned worker. Prefer spawn-process isolation so a
stalled planner can be stopped within a declared shutdown bound. A thread must
not claim hard cancellation or bounded cleanup for an arbitrary hung callback.
Model/factory initialization occurs inside the worker. Do not fork a live CUDA
context. Tests use CPU callbacks only.

- Bound requests and completion storage. Reject hidden queued work.
- Preserve original request/generation identity in every completion.
- Serialize an owned request before an asynchronous transport can retain caller
  references. Handle serialization and initialization failures explicitly.
- `poll()` must not wait for inference. Measure no performance claims here.
- Convert planner and transport failures to explicit failure reports.
- Reset invalidates results, not resources. Old work remains active until its
  completion or confirmed shutdown. Do not infer cancellation from an exception.
- Shutdown is idempotent and bounded. Terminate/kill only the exact owned worker
  when graceful join fails. Confirm exit and close IPC resources.
- Test successful work, overlapping-submission rejection, planner/factory failure,
  payload round-trip, malformed output, blocked work, shutdown and resource exit.
- Use bounded Event/Condition waits in tests. Do not use sleep polling loops.

## Integration contract

Build a callable library facade. A runnable CPU-only smoke harness is sufficient;
this slice does not activate the mjlab or vendor simulator.

1. Snapshot measured history, current state and immutable task settings.
2. Submit planning without waiting for inference.
3. Select the raw latent for the execution tick.
4. Enforce finite source-age and configured validity limits using one monotonic
   clock domain. Reject future timestamps and incompatible plan identities.
5. Decode the selected latent with the current finite 99-D proprioception.
6. Return a finite 31-D action and the exact raw latent plus provenance.
7. Record the pre-action state/latent pair only after execution acknowledgement.
8. On reset, teleport, estimator discontinuity, policy switch or release,
   invalidate the relevant work and history. Rebuild eight measured pairs when
   the physical/history boundary requires it.
9. Task revision changes invalidate old task work but must not fabricate history.
10. On trajectory-horizon exhaustion or invalid data, return an explicit
    no-action status. Do not select a standing policy or physical fallback
    without separate validation.

The `max_fallback_ticks` setting is a warning threshold only. It does not
invalidate an accepted plan before its valid trajectory horizon. Emit one
structured fallback warning per accepted plan when the threshold is reached.

Fake-clock tests must cover on-time and late completions, current-state decoding,
execution-row selection, prefix history, task snapshots, source-age expiry,
reset/release, malformed results and exact cleanup. Add a small real CPU-worker
round-trip so the integration is not proved only with a fake executor.

## Model-backed reproduction seam

`src/mjlab/tasks/tracking/diffusion/model_backend.py` provides
`ModelBackendSpec`, `ModelPlannerFactory`, `load_model_backend` and
`build_model_backed_runtime`. The spec contains only explicit paths, the VAE
member, device strings and optional identities. The owner loads the VAE decoder;
the spawned planner worker loads the active contract, projection bundle and
EMA-selected diffusion denoiser, then returns `LatentPlanner.process`.

The reproduction path does not require contract-file, checkpoint, VAE or
projection SHA values from the caller. Contract-file SHA values remain optional
metadata and are excluded from semantic identity. Structural gates remain:
active schema, 135 physical channels, 32 latent channels, 167 token width,
99-D conditioning, 31-D action output, finite tensors and compatible saved model
schemas. Missing or malformed files still fail at the explicit loader boundary.

This is library/software wiring. No actual frozen-model run, GPU timing,
50 Hz/p99 result, export, simulator, controller, deployment or hardware claim
is implied by the CPU tests.


Focused commands: pytest with `-x -vv -o faulthandler_timeout=30`, Ruff lint and
format check, scoped Ty/Pyright, and untracked-aware whitespace checks.
Use 30 seconds for inspection, 60 seconds for one test/reproducer, and 120 seconds
for a focused suite. After a timeout, inspect and isolate; do not increase the
same command's timeout. Preserve pipeline exit status. Long work needs the parent.

Report exact commands, statuses, changed files, hashes and residual risks through
managed output bindings. Keep source in the owning worktree and do not stage it.
An infrastructure failure stops this protocol; no implicit CLI/foreground fallback.
A shared API conflict needs parent approval before changes.

## Acceptance boundary

Component gates and integrated CPU tests are software evidence only.
Independent review is not authority to publish or deploy.
The parent inspects and validates the candidate before copying source to primary.
Source publication, GPU timing, export, physics, vendor sim2sim and hardware remain
separate work. D5 is not complete when this CPU slice finishes.

## Accepted CPU slice — 2026-10-06

The parent accepts the isolated CPU implementation and validation slice.
Final independent review returned **OK with notes**, with no remaining blocker.
[Report, exact source, review and validation](../../../worktrees/mjlab/d5-runtime-20261006/integration/logs/diagnostics/diffusion-d5-runtime-cpu-20261006/REPORT.md).

Parent evidence: **39 executor + 163 runtime/tensor tests = 202 unique CPU cases**.
The affected scheduler file passed **71 tests** after a test-only type correction.
The primary revalidation independently passed the same **39 + 163** groups.
All eight source/test files pass scoped Ruff format/lint, Ty, Pyright and
untracked-aware whitespace checks. Final source hashes match the reviewed source.

The historical workflow failure, native timeout, wrong test copy and intermediate
monitor Ty failure remain separate failed records. Do not relabel them as passes.

The accepted CPU source, tests, README, changelog and evidence are now integrated
into the primary checkout by the local commit requested after acceptance.
The original isolated worktree and source archive remain preserved. No push was
performed. GPU timing, export, physics, controller, vendor sim2sim, deployment
and hardware readiness remain untested.
The next handoff is a separate actual-model backend and timing plan. No new
execution is authorized by this status entry.
