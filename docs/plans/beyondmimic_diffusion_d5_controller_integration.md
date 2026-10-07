# D5 controller integration in mjlab

Date: 2026-10-07 (+08)
Source base: `074fe430eaefefdc89dbb56b678509288ebef4e9`.
Status: accepted for the CPU software slice; integrated into primary, uncommitted.

## CPU acceptance — 2026-10-07 (+08)

The parent accepts the opt-in `mjlab` controller lifecycle and execution bridge.
Final independent recheck: OK with notes; no P0 or P1 finding.
The four original review findings are fixed.
Two bounded test notes and optional cleanup remain deferred.

Parent validation passed 27 new controller cases and 132 adjacent cases in primary.
Ruff format/lint, Ty and Pyright pass on all five new source/test files.
Reviewed source/test hashes match the primary copies.
Unrelated dirty/untracked file hashes are unchanged, and the index is empty.
No commit or push was made.

[Acceptance report, logs and final review](../../logs/diagnostics/diffusion-d5-controller-cpu-20261007/REPORT.md).

This is CPU software acceptance only.
The observation and execution boundaries remain caller-supplied interfaces.
Sustained GPU timing, export, simulator integration and hardware remain untested.
`x2_docker` is unchanged. D5 as a whole remains incomplete.

## Delegation status — 2026-10-07 (+08)

The original workflow is `ab2aebb7-e1e9-4f60-a7be-8347acb88fa1`.
Lifecycle and dispatch handoffs completed in separate worktrees.
The integration child `4644a928-32cb-4048-9d59-3c654f14b7c7` timed out after 1800000 ms.
The original workflow failed. It did not launch review.

The parent inspected the partial source, documentation and final tool logs.
Component copy hashes match both completed handoffs.
The child log reports 24 new tests passed in 16.49 s, plus scoped static checks.
The next adjacent-test command was aborted at the child deadline.
These are child-reported checks, not parent acceptance.

Partial changes are preserved under the lane root at
`orchestration/recovery-20261007T212847+0800/`.
The snapshot includes tracked diffs, untracked files, status and the exact base.
At that recovery boundary, primary had no controller source publication.

Recovery workflow `a76aeed8-0f71-45d6-8ec3-782baf437a48` resumed integration.
The child completed 71 scheduler, 40 executor, 16 runtime and five backend tests.
It also corrected the documentation and verified final hashes and whitespace.
Its upstream stream then failed before the final handoff was produced.
That failed child is `ee78d145-a21a-4131-a203-a26abd7034e4`.
Saved child evidence totals 24 new cases and 132 adjacent cases.
Parent acceptance remained pending at this recovery boundary.
Updated partial changes are preserved in
`orchestration/recovery-stream-20261007T213708+0800/`.

The user changed the subagent model because GPT was unstable.
Use `ds-oc/deepseek-flash:high` for all remaining children.
A retained resume cannot change its stored GPT model.
Workflow `3f9a1c88-20ed-464a-8c41-b146fbc083dd` completed handoff recovery and fresh review.
The reviewer returned OK with notes, with no P0 or P1 finding.
The parent accepted fixes for non-vacuous tests, deterministic warning coverage,
full task-snapshot comparison, and lifecycle/acknowledgement coverage.
Disposition: `orchestration/REVIEW-DISPOSITION.md` under the lane root.
Workflow `27cbb7f0-1835-4bd5-b98f-28b38e8abed5` completed those focused fixes.
The retained DeepSeek reviewer returned OK with notes on their exact blast radius.
Parent validation and scoped primary integration then completed.
No completed component writer was rerun, and no commit or push was made.



## Goal and authority

Add an opt-in controller lifecycle and execution bridge inside `mjlab`.
Use the existing model-backed asynchronous latent runtime.
Give all remaining children the model `ds-oc/deepseek-flash:high`.
The initial completed component writers used `lz-fast/gpt-5.6-luna:max`.

This is a CPU software integration task.
Do not change `x2_docker`, accepted D3/D4 evaluators, or learned artifacts.
Do not launch GPU jobs, physics, vendor simulators, remote jobs or hardware.
Do not export, deploy, install packages, stage, commit or push.
The parent can copy the validated, scoped candidate into the primary checkout.
A later commit requires user authorization.

A caller supplies measured observations and the execution boundary.
The bridge does not invent a safe hold, standing action or zero action.
A successful software state is not physical safety or 50 Hz readiness.

## Existing seams

- `diffusion/model_backend.py`: explicit asset loading and runtime construction.
- `diffusion/async_latent_runtime.py`: warmup, tick selection and acknowledgement.
- `diffusion/planner_executor.py`: one persistent spawned worker and bounded cleanup.
- `diffusion/scheduler.py`: one active request and one pending latest request.
- [D5 follow-up](beyondmimic_diffusion_d5_followup.md): outstanding qualification gates.
- [Active contract](beyondmimic_diffusion_d1_no_mixtures_contract.yaml).

The historical controller handoff uses 231 projected channels.
It is not authority for the active 167-channel interface.

## Frozen invariants

- Physical state: 135 channels. Raw latent: 32 channels. Tokens: 167 channels.
- Decoder conditioning: 99 channels. Decoder action: 31 channels.
- Window: eight measured past pairs, current state, and 32 future rows.
- Sampler: 20 DDIM updates; no silent rate, horizon or model reduction.
- Active normalization uses statistics only. It applies no matrix or pseudoinverse.
- Structural, shape, finite-value and interface checks remain mandatory.
- Caller-provided artifact digests remain optional reproduction metadata.
- Model initialization remains inside the spawned planner worker.
- Only the current execution-tick latent is decoded with current conditioning.
- History changes only after a matching, successful execution acknowledgement.
- Warmup output is discarded. It cannot reach the execution sink or history.
- `max_fallback_ticks` remains a structured warning threshold, not an action limit.
- Source-age and trajectory-horizon guards remain independent hard guards.
- Cleanup is confirmed only with exact `closed=True` and `worker_alive=False`.

## Component A: lifecycle and measured-input contract

Add `src/mjlab/tasks/tracking/diffusion/controller.py`.
Add `tests/test_tracking_diffusion_controller.py`.

Public symbols:

- `ControllerState`, `ControllerError`.
- `ControllerObservation`: owned current `WorldState`, finite 99-D conditioning,
  nonnegative integer `tick`, `sample_id`, `source_frame`, nonempty `segment_id`,
  and finite nonnegative monotonic `timestamp`.
- `D5Controller(runtime_factory, ...)`: sole owner of one `AsyncLatentRuntime`.

States:

```text
LOADING -> BOOTSTRAPPING -> WARMING -> READY -> ACTIVE
                                         |          |
                                         +-> FAULT <-+
RELEASED and SHUTDOWN are terminal.
```

`BOOTSTRAPPING` is explicit because warmup requires eight actual executed pairs.
Do not fill the prefix from standing history or synthetic values.

Required interface:

- `load()`: call the injected factory once; validate and enter `BOOTSTRAPPING`.
- `acknowledge_bootstrap(state, latent, *, success)`: delegate actual execution ack.
- `warmup(observation, *, task=None, task_revision=None, decoder_calls=3,
  timeout_seconds=10.0)`: enter `WARMING`, call real runtime warmup, then `READY`.
- `request_activation()`: arm activation only after successful warmup.
- `step(observation, *, task=None, task_revision=None) -> RuntimeStep`:
  use a fresh post-warmup observation; enter `ACTIVE` only on a valid fresh action.
- `acknowledge(step, *, success) -> bool`: delegate the exact pending action ack.
- `fault(detail)`: suppress commands and clean the exact owned runtime.
- `reset()` and `switch_policy(policy_id)`: invalidate old work/history/readiness.
  Close the previous runtime before another explicit `load()` can create one.
- `release()` and `shutdown()`: idempotent terminal cleanup.
- Read-only `state`, `runtime`, `cleanup_confirmed` and failure-detail access.

Use one injected monotonic clock domain for observations and runtime.
The factory must construct the runtime with that same clock.

After warmup, require a new `sample_id` and a capture timestamp at or after its end.
Do not demand a physical step merely to make a new observation.
Reject regressed samples, duplicate execution ticks and segment/frame changes.
Allow tick jumps; select the actual execution-tick row, not a consume-call row.
Validate current observation age before decode and dispatch.

Waiting, future, stale or exhausted plans produce no command.
Record explicit runtime status and remain inactive until a fresh valid action exists.
Planner, decoder, malformed-input/output and execution failures enter `FAULT`.
A task revision change must invalidate old task work without fabricating history.
No automatic fault recovery, repeated load, hidden worker restart or warmup in `step`.

Keep this state machine synchronous in the owner thread.
Only startup warmup may wait for planning.
Do not add a background controller thread or pacing loop.

## Component B: execution bridge

Add `src/mjlab/tasks/tracking/diffusion/controller_driver.py`.
Add `tests/test_tracking_diffusion_controller_driver.py`.

Public symbols:

- `ActionCommand`: owned finite 31-D action, execution ID, tick and provenance.
  Do not expose a raw latent as a command payload.
- `ControllerDispatch`: controller state, runtime step, dispatch/acceptance outcome,
  explicit detail and cleanup status where relevant.
- `D5ControllerDriver(controller, read_observation, execute_action)`.

Required behavior:

- `warmup(...)` reads the observation callback and calls controller warmup.
  It never calls the action sink.
- `request_activation()` delegates explicit activation intent.
- `tick(...)` reads a new observation, calls `controller.step`, and forwards
  only an `ACTION` result while the controller is `ACTIVE`.
- `execute_action(ActionCommand) -> bool` must return a literal boolean.
  `True` means execution acceptance sufficient for measured history, not queueing.
  Mere transport delivery is not this acknowledgement contract.
- A sink rejection, exception or nonboolean result must not enter history.
  Fail closed, fault the controller and do not retry that action.
- A successful sink calls `controller.acknowledge` exactly once with the original
  `RuntimeStep`. Preserve the pre-action measured state/raw latent binding.
- Duplicate calls, reentrant callbacks, resets or release during a callback must
  not publish or acknowledge a later stale action.
- Forward scheduler warning events without using Python global warnings.
- Normal ticks perform no sleep or planner wait. Caller owns the 50 Hz schedule.
- Provide a small assembly helper that uses `build_model_backed_runtime` from an
  explicit `ModelBackendSpec`; retain the spawned-worker/owner-decoder split.

Component B uses the Component A API above.
It starts after the lifecycle handoff passes and copies those two files for imports.
Treat these copied files as read-only dependencies.
It must not modify Component A or duplicate its lifecycle.
Use a protocol/fake controller for focused driver tests.
Ask the parent before changing the shared API.

## Integration and acceptance

After both component handoffs pass, one integration-only writer copies their
explicit source/test files into the integration worktree.
That writer adds a real CPU spawned-worker lifecycle integration test.
Use the existing `LatentPlanner.process` with synthetic denoiser/statistics.
Do not construct a simulator or require ignored frozen-model assets.

The test covers eight acknowledged bootstrap pairs, real runtime warmup,
zero warmup sink calls, a new measured input, active dispatch, exact ack,
reset/rebootstrap boundaries, warning telemetry and exact process cleanup.
Use bounded event/completion waits. Do not add sleep polling loops.

Update the D5 README, follow-up note and scoped changelog.
Document a complete load/bootstrap/warmup/activate/tick/reset/shutdown example.
State that the execution sink is injected and the caller owns the safe hold.
Keep physical, GPU timing, export and `x2_docker` integration claims untested.

Focused acceptance:

1. No action sink call before activation or during warmup.
2. The first active command uses fresh measured state and conditioning.
3. No failed execution fabricates history or reports cleanup as confirmed.
4. Reset, policy switch, segment changes and release cannot resurrect old work.
5. Existing scheduler/executor/runtime/backend tests remain unchanged and pass.
6. New source/tests pass scoped Ruff format/lint, Ty, Pyright and whitespace checks.
7. Fresh read-only review checks the actual integrated candidate.
8. The parent validates and copies only accepted scoped files to primary.

## Implementation topology and lane board

This is multi-seam work.
Lifecycle/input ownership and action dispatch have independent contracts.
The driver consumes a fixed API and can use a fake controller in isolation.
Run writers in dependency order: lifecycle, dispatch, then integration.
Do not manufacture a parallel writer before its import contract exists.

Lane root: `/home/agiuser/projects/worktrees/mjlab/d5-controller-20261007T201500+0800`.
All lanes start from the exact source base above.
The parent creates detached worktrees because primary has unrelated dirty files.
No stash or broad cleanup is permitted.

| Lane | Exclusive files | Cwd under lane root | Gate and handoff |
| --- | --- | --- | --- |
| Lifecycle | `controller.py`, lifecycle test | `lifecycle` | CPU lifecycle checks; file hashes and managed report |
| Dispatch | `controller_driver.py`, driver test | `dispatch` | CPU callback/ack checks; file hashes and managed report |
| Integration | Copy four component files; new integration test; D5 README, follow-up, changelog | `integration` | Assembled spawn test, adjacent checks; managed report and hashes |
| Review | Read-only actual integrated candidate | `integration` | Cited P0/P1/P2 findings; explicit verdict |
| Parent | Plan, decisions, primary publication and NOW index | primary | Exact scope check and parent validation; no commit/push |

Every cwd has at most one writer.
Children do not delegate or alter parent orchestration records.
Each component report records files, hashes, commands, failures and residual risks.
A source/tool/model infrastructure failure stops the workflow for parent triage.
Do not switch models or use CLI/foreground fallback.

## Validation environment

Share installed dependencies only. Do not sync the environment.
Use `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`,
`UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv`,
`PYTHONPATH=<lane>/src:<lane>/tests` and `uv run --no-sync`.
Do not link `.venv`: the directory-only ignore rule does not ignore that symlink.
Inspection timeout: 30 s. Small test: 60 s. Focused suite: 120 s.
Investigate unexpected timeouts before a retry. Preserve pipeline exit status.
Use pytest `-x -vv -o faulthandler_timeout=30` and write long logs to managed output.
A long command requires a parent-managed monitor, not an untracked process.
