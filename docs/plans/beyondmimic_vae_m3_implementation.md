# BeyondMimic M3: single-teacher closed-loop DAgger

Status: **authorized for implementation on 2026-09-26**. Implementation and
bounded validation only; no production training, M4, or hardware work.
Repository: `/home/agiuser/projects/mjlab`, branch `fxy/test/tracking-exp`.
Baseline: M1 `cdd9a8201213206d620a4f32b9125f7b0a9a2dc5`; accepted M2 committed
as `76ca932bb43cf8a9418a28a0695138cd72418d02` at the user's request.
All delegated workers/reviewers: **`lingzhi/gpt-5.6-luna`**.
Parent owns scope, decisions, 15-minute supervision, runtime resource gates, and
final independent acceptance. No automatic commit or push of M3.

Related: [architecture](beyondmimic_vae_distillation.md),
[milestones](beyondmimic_vae_implementation.md),
[M2 contract and acceptance](beyondmimic_vae_m2_implementation.md).
This document supersedes earlier statements that M3 is not yet authorized; those
statements describe the M2 acceptance boundary, not the current authority.

## 1. Deliverable and boundaries

Implement an end-to-end **single-teacher** collector, optimizer, checkpoint/resume
lifecycle, and usable train/evaluate CLI. Validate against `tennis_000` and its
existing 50 Hz single-motion environment before adding any multi-motion commands.

Use `configs/distillation/x2_tennis.yaml` to resolve source artifacts and select
one teacher. The manifest may contain both teachers; this does not authorize
multi-motion collection. Preserve all original checkpoint/ONNX/NPZ/saved YAML.
The known environment is
`Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations`.
Use the original teacher's action definition, observation transforms, control
period, and relevant environment behavior; do not silently substitute a newer
experiment's defaults or the paper's 25 Hz diffusion rate.

Keep the M2 gravity default: 68 encoder features, 99 conditioning features,
32 latent coordinates, 31 actions. Bind `cohort.actions.joint_names`, not the
synthetic placeholder names. Do not change the schema, add reference bypasses,
train anchor ablations, retrain teachers, implement diffusion, symmetry/export,
distributed collection, or general-teacher PPO.

## 2. Live environment and observation boundary

Build an opt-in distillation adapter/factory, leaving existing tracking/PPO paths
unchanged. Reuse registered task factories and known trusted classes, validate
the selected live contract against saved teacher data, and record every explicit
override. Do not execute arbitrary constructors/functions from saved YAML.
Resource overrides (environment count, output paths, device, viewer) and evaluation
sampling choices must be distinguished from semantic changes.

Resolve the robot/anchor/gyro sites and local rotations from the actual asset or
compiled CPU model. Verify joint/action order and reference body-subset indexing,
including comparison against original embedded ONNX body references where
available. NPZs have no body-name metadata: dimensions alone do not prove mapping.
If correspondence cannot be established, report the exact gap and ask the parent;
do not invent names. Asset-frame evidence is simulator evidence, not hardware
sensor calibration or hardware-transfer proof. Keep the M2 declaration schema
stable and store detailed audit evidence separately.

Produce teacher and student features from the **same captured observation**:

- Preserve teacher ordering, noise, delay groups, history, scale, and clipping.
- Reuse already transformed teacher measurements for overlapping student fields.
- Capture projected gravity from the declared root frame once per observation;
  explicitly record the additional gravity term's noise/delay policy.
- Do not independently recompute noisy shared terms or advance history twice.
- Previous action is the actually executed normalized command after any configured
  runner clipping, before action scaling/offset into PD targets. Reject unsupported
  action paths rather than silently changing them.
- Own/copy snapshots before simulator buffers can mutate.

Relevant source evidence: `ManagerBasedRlEnv.step` returns post-reset observations
under auto-reset, calls command updates before its final observation computation,
and `get_observations()` uses the observation manager's cache. Direct repeated
`compute_group()` is not an equivalent cache read and can advance delay/noise.
`MotionCommand._update_command` may resample/teleport at wraparound without an
environment termination. Explicit reset also computes commands with `dt=0`; do
not assume that means the reference frame remains numerically unchanged.

Track every reset, timer resample, and motion-end teleport as a segment boundary.
Do not rely only on a frame-index discontinuity: resampling can coincidentally
land at an apparently consecutive frame. Prefer an opt-in MotionCommand subclass
with explicit segment/generation bookkeeping over changing the shared command.
Escalate if a shared manager/command change is genuinely required.

## 3. Collection and evaluation

Use synchronous in-process TeacherBank labeling and one simulator. At each tick:

1. Capture reference, conditioning, exact teacher observation, and integer
   teacher/motion/frame/segment/collector-iteration metadata.
2. Compute the frozen deterministic teacher target and selected student action.
3. Store valid raw inputs and fixed teacher labels before stepping.
4. Execute the selected behavior action through the unchanged action path.
5. Account for termination, timeout, explicit reset, and motion resample; continue
   from the returned next snapshot, never a stale pre-reset observation.

Support a bounded teacher bootstrap and a configurable teacher-execution schedule
that reaches zero; choose **whole action vectors**, never their average. Keep
teacher targets, predictions, and executed actions distinct. Rollout latent
choice is explicit (mean baseline; sampled mode configurable), not inferred from
`.train()`. Optimization uses sampled latents as specified by M2.

Keep valid pre-failure labels. Reject nonfinite inputs/actions with diagnostics;
never step NaN actions or silently sanitize them into success. Handle invalid
samples/reset behavior explicitly. No successful-rollout-only DAgger filtering.
Keep raw FIFO replay; motion balancing belongs to M4. No cached latent targets.

Evaluation must be student-only or explicitly teacher-only, without optimizer,
replay, or normalization updates. Record fixed seeds and environment/sampling
settings. Report failures, survival/completion per segment, tracking errors
(including root-relative pose versus heading), and action magnitude/rate plus
available saturation diagnostics. Do not count reset/teleport transitions as
continuous tracking success or ordinary action jitter. Keep bounded step/episode
limits so short failures cannot trigger unbounded evaluation loops.

## 4. Optimizer core and lifecycle

Use the accepted reconstruction-plus-KL loss, beta `0.01`, Adam LR `5e-4`, and
15 microbatches per accumulated optimizer update with correct division. Validate
positive finite budgets/configuration. Expose small model settings for CPU tests,
but retain paper-size production defaults. No hidden clipping, KL warmup, free
bits, new loss terms, or PPO critic/advantages machinery.

Update student normalizers only at documented training boundaries from raw
samples, not during rollout/evaluation/labeling. Explicitly define whether new
samples are counted once; avoid repeatedly counting the whole replay as new data.
Freeze stats during collection/evaluation. Preserve teacher eval/frozen state.
Reject nonfinite losses/gradients before stepping; report numerical failures
rather than hiding them through altered objectives.

Separate pure minibatch optimization from the runner. The runner coordinates
bootstrap, collect/update cycles, evaluation, logging, and save/resume with bounded
configuration. Prefer local machine-readable logs; no external tracking uploads.
Default CLI behavior must not silently launch a large production run.

Checkpoint model settings, schema, normalizers, optimizer, counters/schedules,
RNG state (including explicit generators), resolved configuration, teacher source
hashes, and replay or a precisely declared replay reconstruction policy. Prefer
persisting the bounded raw replay for reproducible CPU resume tests. Narrow additive
replay state APIs are authorized; existing insertion/sampling semantics stay intact.
Use versioned validated tensor/plain-data state, safe loading where supported, and
atomic file replacement. Reject incompatible schema, joint order, teacher/artifact,
and control contracts before proceeding. Do not serialize live simulator objects.

Resuming training restarts the simulator unless an existing full-state mechanism
is explicitly validated. Log this boundary, reset observation/action history
correctly, and avoid reusing segment IDs across resume. CPU next-update/RNG parity
is a valid claim; bitwise simulator continuation is not.

## 5. Exclusive ownership and serial handoffs

This is **multi-seam**: live observation contracts, collection, optimization,
training-state persistence, and CLI integration have distinct testable boundaries.
The checkout has unrelated user work and discussion documentation; do not stash,
reset, clean, or commit them for worktree allocation. Use **exclusive serial
ownership in the existing checkout**, never concurrent writers. Parent reads only
while a writer is active and does not run competing heavy tests.

All rows use the repository/branch/baseline above and `worktree:false`. Each writer
returns a durable runtime-managed report with exact APIs, changed files, commands,
results, and risks. Downstream stages read actual source plus prior reports.

| Stage | Exclusive source ownership | Focused gate | Why separate |
| --- | --- | --- | --- |
| `m3-environment` | New `distillation/environment.py`, optional `adapter.py`; new environment/adapter tests | Saved/live contract checks, CPU asset audit, shared-snapshot/noise/history tests, segment counters | Simulator-facing semantics independent of optimizer |
| `m3-collector` | New `collector.py`, optional `evaluation.py`; new collector/evaluation tests | Fake vector env proves action/label alignment, auto-reset, wrap/timer segments, mixing endpoints, bounded pure evaluation | Tick-level DAgger correctness independent of learning |
| `m3-trainer` | New `training_config.py`, `trainer.py`; new trainer tests | Tiny-batch learning, accumulation/RNG equivalence, normalization lifecycle, frozen teachers, finite failure paths | Pure supervised update contract |
| `m3-lifecycle` | New `checkpoint.py`, `runner.py`; narrowly additive `storage.py` state APIs; new checkpoint/runner tests | Versioned roundtrip, deterministic CPU next update, replay ownership, mismatch rejection, bounded fake-env runner | Training orchestration and saved state depend on prior components |
| `m3-integration` | Additive CLI/public exports/docs/changelog; new integration tests and narrowly extended CLI tests; small M3 wiring fixes | CPU full regression suite and usable help/train/evaluate surface; exact commands for parent-run live gates | Integration only; missing major components go back to their owners |
| `m3-review` | Fresh-context **read-only** reviewer; no source edits | Concrete P0/P1/P2 findings tied to current source/tests/contracts | Independent review before parent acceptance |
| Parent | Plan documents, decisions, resource-gated real runs, independent acceptance | Inspect actual code, rerun checks/counterexamples, verify preservation and live results | Child reports alone cannot accept M3 |

Names may be refined within a row's boundary and recorded in its handoff. An owner
must ask before editing another row's files; integration's small wiring permission
is not permission to reimplement a missing component. Existing M1/M2 tests must
continue to pass. Do not weaken tests or parity tolerances to obtain acceptance.

Protected: original teacher artifacts and YAML/NPZ, robot assets, dependencies and
lockfiles, existing environment/manager/tracking/PPO behavior, all `tools/` work,
the separate general-tracker discussion, and unrelated changes. Public exports,
CLI, usage/changelog, and the narrowly additive replay state API are the specified
exceptions. Shared framework changes require a scoped parent decision and tests.
No commits/staging/push/upload, remote jobs, hardware access, or child fanout.

## 6. Validation and resource budget

All Python/tool commands use `uv run`. CPU tests use `CUDA_VISIBLE_DEVICES=''`
and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Inspection 30 seconds, probes 60 seconds,
focused suites 120 seconds; pytest uses `-x -vv -o faulthandler_timeout=30`.
Diagnose unexpected timeouts instead of retrying identically or inflating limits.

**Workers run CPU checks only by default.** CPU MuJoCo asset compilation is allowed;
GPU environment construction, rollout, or real training requires the parent to
inspect the proposed command and execute it under a monitor. Workers request this
through `contact_supervisor` and never spawn untracked background jobs.

Parent's initial M3 live-validation envelope on this laptop:

- One local GPU/process at a time; at most **16 environments**; no remote resources.
- Teacher baseline and each student evaluation: at most **512 control steps**,
  at most **8 environments**, fixed seeds; include starts and phase-start cases.
- Initial smoke: at most **20 collect/update iterations**, **32 collection steps
  per iteration**, **20 accumulated optimizer updates**, minibatch at most **256**,
  replay capacity at most **16,384**; teacher bootstrap at most **256 steps**.
- One bounded resume check, at most **2 additional iterations**.
- Total initial live-validation wall-clock budget **30 minutes**, including first
  compilation. A monitored first compilation may have a **10-minute hard bound**
  because MuJoCo/Warp kernel compilation is a known one-time cost; ordinary small
  tests retain the shorter bounds above. Record commands, progress, logs, exit
  codes, and exact owned cleanup. Do not keep retrying after a timeout.
- No hyperparameter sweep, convergence run, asset simplification, or dependence
  upgrades to fit the budget. Escalate resource exhaustion/environment failures.

These are validation ceilings, not instructions to spend the full budget. A tiny
smoke proves pipeline behavior, not an adequate trained policy. Full-quality
training and performance acceptance need a separately agreed budget and
baseline-relative thresholds. Student-only smoke metrics may be poor; report
that honestly instead of claiming successful reproduction.

Required regression checks: all distillation and tracking-command tests, CPU
runner export subset used for M2, relevant observation/reset/command tests if
touched, scoped Ruff/Pyright, `uv run ty check src tests`, and original real-teacher
CPU parity (64 samples, `atol=rtol=1e-5`). Whole-repository Pyright has known unrelated
failures; M2 precommit evidence is `/tmp/mjlab-m2-precommit.y837a_n0/`.

## 7. Supervision, recovery, and acceptance

Use one async subagent workflow with serial awaited stages and one independent
reviewer. Supervise every **15 minutes**; inspect actual activity/logs and answer
supervisor questions. Steer only concrete blockers/drift, not healthy work. Keep
run IDs, mapped output references, and terminal evidence. JSON-sanitize optional
result fields before emit/state/return; a previous workflow failed on `undefined`
`outputPathMapping` after a successful child.

On infrastructure/provider/tool failure, stop downstream launch, preserve the
partial diff and exact run/cwd/ref, and recover only through the same authorized
protocol/model. Do not rerun completed stages or silently switch to a CLI/model.
On implementation defects, parent dispatches a bounded retained-owner correction
and reruns affected gates. Periodic supervision ends only at terminal handoff to
parent review/recovery, cancellation, or the explicit bounded lifetime.

M3 implementation is accepted only after actual source inspection, CPU regression
and adversarial alignment/resume checks, real teacher baseline and bounded
collector/train/resume/student-evaluation evidence, and preservation checks.
If a real gate cannot run, report partial implementation and the exact blocker;
do not silently count fake-env tests as simulator validation. Final documentation
must separate implementation acceptance, smoke evidence, policy quality, and
hardware readiness. Stop before M4 and any production training.

## 8. Recovery and early review disposition — 2026-09-26

The initial async workflow `058f9d2a-322e-474d-a71a-41d17bf5f10d` delivered the
environment and collector candidates. Parent read-only inspection recorded
mandatory corrections in
`/tmp/mjlab-m3-baseline.2q3gay1y/environment-parent-review.md` and
`collector-parent-review.md`; those are findings, not acceptance records.

The trainer child `d0842401-4b99-470c-a4f8-afe0a1aa15cf` completed its initial
`training_config.py` write, then had no completed response/tool for over 24 minutes.
A queued follow-up remained unconsumed. Parent preserved the partial source/diff
under `/tmp/mjlab-m3-baseline.2q3gay1y/trainer-stall-20260926T1049`, then explicitly
soft-interrupted the exact child. Its process is terminal/paused and resumable;
the enclosing workflow terminated with “Paused after interrupt. Waiting for
explicit next action.” No failed test is being retried with a longer timeout.

Recovery stays on the same native protocol, mission, and Luna model:

1. Resume the retained trainer to finish only its missing component.
2. Resume the retained environment owner for the exact parent findings: compiled
   sensor/site/local-frame evidence, indexed numeric body-reference parity,
   saved/live semantic validation, real-joint schema binding, and route tests.
   This owner may additionally expose owned per-environment physical tracking
   metrics at a documented aligned snapshot time for the collector correction;
   these metrics must not draw noise or advance observation history.
3. Resume the retained collector owner for its exact findings: per-environment
   pose/yaw metrics, explicit completion/censoring outcomes, no zero-step segment
   pollution, cache safety after numerical failure, and a bounded fresh-raw-data
   handoff for once-per-data normalization. Consume the corrected adapter API and
   completed trainer API; do not average heterogeneous env metrics first.
4. Continue the not-yet-started lifecycle, integration, and fresh review stages.

These are exclusive scoped correction passes, not reruns of completed components.
The original environment owner's small additive public exports are left for the
integration review; subsequent component owners must not expand export ownership.
No other edit or resource authority changes. Replace the old-root monitor only
after terminal proof, and supervise the continuation every 15 minutes.

## 9. Parent candidate review — corrections required

The continuation delivered all components and a fresh read-only review. M3 is
**not accepted**. Parent CPU regression/static/teacher-parity gates passed, but
the reviewer returned BLOCK and direct parent counterexamples confirmed leaked
microbatch gradients, partial RNG restore, misleading post-step failure state,
and an additional CLI blocker: student evaluation cannot load the checkpoint
shape emitted by training, even after matching training settings. CLI seed
application also occurs after randomized environment construction and must move
to the private pre-construction boundary.

Correction authority is recorded in
`/tmp/mjlab-m3-baseline.2q3gay1y/parent-corrections.md`, with diagnostic evidence in
`parent-candidate-checks/` and a preserved `candidate-before-parent-fixes.tar.gz`.
Use exclusive serial retained Luna owners for trainer failure semantics, live
boundary/seed producer, collector recovery/consumption, checkpoint inference
loading, and final CLI integration, followed by retained independent review.
These are specific current-contract corrections, not a redesign or new milestone.
No GPU/live budget has been spent, and no live gate may be counted as passed until
executed under the original resource envelope.

Repository HEAD independently advanced to
`927f3b39be87c8cee4d218c8d8ff83b7d5aeb35b` (`remote fetch tool`) after the M2
commit; that commit changes only `tools/compare_remote_tasks.sh` and is preserved.
M3 source remains uncommitted and the index is empty.
