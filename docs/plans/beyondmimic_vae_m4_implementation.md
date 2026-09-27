# BeyondMimic M4: shared multiple-teacher distillation

Status: **implementation and bounded integration accepted**, 2026-09-27, with
explicit inherited repository-check exceptions recorded in section 11. This is
not a trained-policy quality pass, a clean full-repository gate, or a commit.
M3 is complete; the user
has accepted moving on after the single-teacher training/evaluation work,
including the completed 4096-environment 10,000-iteration run. This is the current
authority document; older M1–M3 statements deferring M4 are historical.

Repository: `/home/agiuser/projects/mjlab`.
Branch: `fxy/test/tracking-exp`.
Planning baseline: `5cb756eed94b5a2353a1b8b4aa93b65f0f160b7a`.
Implementation model: **`ds-oc/deepseek-flash`**, native async subagents.
Parent owns planning, supervision every **20 minutes**, resource approval,
review/finding disposition, and final acceptance. No automatic commit or push.

Related: [architecture](beyondmimic_vae_distillation.md),
[milestones](beyondmimic_vae_implementation.md),
[M3 contracts](beyondmimic_vae_m3_implementation.md), and
[usage](../source/x2_tennis_distillation.rst).

## 1. Deliverable and scope

Train **one shared conditional VAE** on the two existing motion-specific teachers,
in **one vectorized simulator**. Every environment's current clip determines both
its reference and its frozen deterministic labeling teacher. Optimize with
motion-balanced raw replay; evaluate each motion separately using the same saved
student. No ensemble, separate student per clip, or PPO training is substituted.

Use the existing `configs/distillation/x2_tennis.yaml`:

| Stable teacher ID | Reference | Frames | Rate | Initial weight |
| --- | --- | ---: | ---: | ---: |
| `tennis_000` | `data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz` | 453 | 50 Hz | 1.0 |
| `tennis_001` | `data/tennis/single_001_zhanghongyu_agibot_x2_tracking.npz` | 340 | 50 Hz | 1.0 |

The original model_29999 checkpoints, ONNX exports, and saved YAML remain the
source of truth. Resolve and audit **both** teachers; do not assume the second
motion's body reference is correct because the first was audited.

Preserve the accepted gravity model: reference 68, conditioning 99, latent 32,
actions 31; existing named schemas and model settings remain valid. Reference
q/dq and teacher/motion IDs never enter decoder conditioning. Keep Adam, KL beta
0.01, LR 5e-4, accumulation 15, loss reductions, mean rollout baseline, sampled
training latents, once-per-fresh-data normalization, and frozen teacher statistics.
No new objective, curriculum, or stabilization heuristic is part of M4.

Out of scope: diffusion, symmetry augmentation, teacher retraining, general-tracker
PPO, arbitrary robot/control cohorts, distributed collection, hardware, new viewer
UI, production convergence runs, remote jobs/installations, and checkpoint
migration presented as strict resume. The existing export feature is preserved;
M4 does not expand deployment/hardware claims.

M3 artifacts remain untouched. The 8192-env run stopped before 10,000 (latest saved
9,000); the 4096-env run completed 10,000. Neither fact is evidence of two-motion
policy quality. Do not silently warm-start M4 from an M3 checkpoint: train-from-
scratch is the baseline, and a future explicit weights-only warm start is separate
from resume.

## 2. Current code and concrete gaps

- `TeacherBank.label(codes, observations)` already groups rows, keeps each teacher's
  normalizer frozen, and restores row order. Reuse it.
- `SegmentMotionCommand` records resets/timer resamples/wraps as explicit generation
  changes. It inherits scalar clip length/indexing from `MotionCommand`.
- `DistillationEnvironmentAdapter` currently sets every `motion_id` and scalar
  `teacher_code` to zero; `DistillationSnapshot` has scalar teacher identity.
- `collector._teacher_action` calls one teacher; `_replay_batch` repeats its scalar
  code. Both must use the **captured pre-step row IDs**, not live post-reset IDs.
- `LabeledReplayBuffer` is a single FIFO, uniformly sampled; a shorter/easier motion
  can otherwise dominate storage and updates.
- `checkpoint.py` uses version-1 single-teacher hashes/control metadata. `runner.py`
  directly inspects FIFO internals to rebase resumed segment IDs.
- CLI train/evaluate/play/export and playback checkpoint swaps select one teacher.
  M4 training needs an explicit multi-teacher selection; inference needs validated
  selection of a member of the saved cohort, not a weakened global hash check.
- Shared tracking now includes standing-start fields. The original saved tennis
  cohort did not authorize the newer standing-start experiment: preserve the
  original default of zero and explicitly report phase-sampling overrides.

## 3. Reference library and multi-motion command

### 3.1 Immutable library

Add a distillation-owned motion library (suggested `motion_library.py`) with:

- Explicit ordered clip entries: stable motion ID, teacher ID/code, source hash,
  frame count, FPS, body mapping, and optional flat-storage offset.
- Read-only owned float32 reference tensors for q/dq, body position/orientation,
  and body linear/angular velocity. Resolve body subsets using compiled robot
  order and each original ONNX; never infer physical names from tensor dimensions.
- Vectorized gather by `(motion_ids[B], local_frames[B])`. Validate IDs and local
  bounds **before** offsetting/indexing. Negative values must not exploit Python
  indexing; endpoint lookahead clamps inside its own clip, never the next clip.
- No padded-frame sampling, silent FPS conversion, concatenation-based wrap, or
  writes to source arrays. Reject malformed/nonfinite/empty/incompatible motions.
- Stable mapping persisted in provenance; manifest reordering must not silently
  reinterpret numeric IDs. Strict resume rejects a changed ordered mapping.

Tests use unequal tiny clips with unmistakable sentinel values, including first/
last frames, mixed/permuted IDs, partial selections, invalid indices, and mutated
returned buffers. Add explicit unsupported lookahead rejection if the implementation
cannot preserve it; the selected cohort has zero-width lookahead.

### 3.2 One command, mixed environment rows

Add an opt-in `MultiMotionCommand`/config inside the distillation package, reusing
`SegmentMotionCommand` contracts. Do not change existing registered PPO commands.
Public reference/robot queries consumed by observations, rewards, terminations,
and metrics retain their meanings and shapes. Keep `time_steps` **clip-local**;
expose per-row clip length, motion IDs, and teacher codes explicitly.

All q/dq/body/anchor queries, reference-state resets, phase selection, relative-body
pose refreshes, and wrap detection must use each row's selected clip. A scalar
`motion.time_step_total` is not a valid mixed-clip bound. Do not swap a global
`command.motion` object while iterating environment rows.

**Collection policy for the first baseline: stratified environment slots.**
Allocate environment rows according to the manifest's positive weights (equal for
this cohort), using deterministic largest-remainder allocation and a seeded row
permutation. Require at least one row per selected motion, and reject a budget or
weight combination that cannot represent all selected motions; report requested
weights and actual row counts/fractions. Keep each slot's motion fixed through
ordinary per-row reset/timer/wrap. This deliberately avoids the clip-duration bias
of equal-probability motion sampling at each episode boundary and makes equal
fresh-data normalization coverage possible. It is an engineering choice, not a
paper claim or a promise of seamless transitions between clips.

Provide an explicit reset/selection seam for tests and pinned per-motion evaluation;
selection changes only together with a full documented reset of the selected rows.
No mid-segment ID mutation. Dynamic clip reassignment/curriculum is not needed for
M4. If a worker needs a different policy, obtain a parent decision before changing
this contract.

Use **uniform phase sampling for M4 training** and `start`/`uniform` for evaluation,
recorded as explicit private overrides from the teacher's adaptive sampling.
Do not share adaptive failure bins across clips. Adaptive/weighted multi-motion
sampling may be refused clearly rather than implemented speculatively.

Maintain exact boundary semantics: each reset/timer/wrap increments generation
once; completion belongs to the pre-boundary clip/generation; simultaneous
termination takes failure precedence. Test subset reset, unequal-length wrap,
timer+wrap, wrap+termination, zero-step reset compute, consecutive-looking frame
resampling, and unaffected rows. Preserve simulator forward/sense and delay/cache
semantics at teleports; no stale reference/robot pose on the next action.

The factory builds one trusted registered environment with a private config. Audit
common controls/observations/sensors and **each clip's** reference correspondence.
Saved YAML remains data, never executable constructors. Resource overrides and
uniform-phase/slot-allocation semantics are recorded separately. Missing evidence
is an error, not an `audited` placeholder.

## 4. Aligned snapshots, routing, and attribution

Extend snapshots additively with owned per-row integer teacher codes, aligned with
motion/frame/segment/generation tensors. Preserve the single-teacher constructor
and scalar fields for existing callers, but never use a scalar fallback for a
mixed batch. Fail clearly if row metadata is malformed or mapping disagrees.

Teacher/student features still come from one cached actor observation. Do not
recompute independent noisy shared fields or advance delay/history twice. Teacher
normalizers remain independent and immutable. The student has one shared pair
of normalizers, updated only from new raw collection rows once each.

At every collection tick:

1. Capture owned observation + motion/frame/teacher/segment metadata.
2. Route those rows through `TeacherBank`; validate shape, IDs, device, and finiteness.
3. Store the raw packed inputs, fixed label, and captured routing metadata.
4. Select whole teacher or student action vectors with the existing schedule.
5. Step once and consume explicit boundary events before the next snapshot.

Previous action remains the actually executed normalized command. Valid pre-failure
labels remain in replay. Teacher-only bootstrap and final student-only collection
work identically for mixed rows. Reject nonfinite inputs/actions without stepping
NaNs or silently filtering failures into successful metrics.

Extend segment/evaluation attribution with the motion/teacher that **started the
segment**, not the post-reset command. Mixed collection reports include per-motion
samples, disagreement, boundaries, and reference-frame coverage, as well as total
counts. Do not retain unbounded per-env arrays in ordinary progress logs.

Tests: use two synthetic teachers with intentionally distinct outputs/normalizers;
mix and permute rows, reset only a subset, and wrap one clip. Assert every stored
label and code belongs to the pre-step observation and original row, and verify
all teacher parameters/stats are unchanged. Include single-teacher regressions,
unknown teacher codes, wrong motion-to-teacher pairs, and cache mutation probes.

## 5. Balanced replay and normalization

Preserve `LabeledReplayBuffer` and its version-1 FIFO state/behavior for M3. Add an
opt-in balanced buffer (suggested `balanced_storage.py`) using per-motion FIFO
partitions under one total capacity, rather than resampling a global buffer after
one motion has already evicted the others.

- Allocate integer capacity quotas from the selected weights; capacities sum to
  the requested total and every selected motion receives at least one slot.
- Partition insertions by motion; validate the **whole incoming batch** (including
  teacher mapping and frame ranges) before mutating any partition. Preserve FIFO
  order within each motion. Oversized insertion cannot evict another motion.
- Draw requested minibatches with configured motion proportions, using unbiased
  randomized residual allocation (or an equivalently tested scheme), then shuffle
  rows; within each partition sample uniformly with replacement by default.
  Batch size one must not always choose the first motion.
- No silent omission of a missing motion: bootstrap/runner reports not-ready and
  collects until all selected partitions are populated within its bounded budget,
  or fails explicitly. The initial slot policy should populate both in one tick.
- Specify/test no-replacement behavior; reject impossible quotas rather than
  silently replacing or changing weights.
- Sampling returns owned tensors and never updates normalizers. The trainer uses
  raw inputs to recompute current latents; no latent caching.
- Stats update once from **all fresh raw rows**, not replay samples or retained
  quota subsets. Fixed collection slots make the intended mixture transparent;
  report rounding when env counts are not exactly weight-proportional.
- Expose small public validation and maximum-valid-segment-ID seams so checkpoint/
  runner do not reach into a particular replay implementation's storage layout.
  Preserve existing public APIs; trainer type changes must be behavior-neutral.

Report capacity/occupancy, inserted/retained/drawn samples, and coverage per motion.
Persist partition cursors/contents, ordered routing, quotas/weights, and any
sampler state. Explicit generator state belongs to the existing checkpoint RNG
contract. Require deterministic CPU next-sample/next-update equivalence.

## 6. Lifecycle, cohort identity, and backward compatibility

Add a versioned cohort contract (suggested `cohort_contract.py`) containing:

- Ordered stable teacher/motion identities and every teacher checkpoint/ONNX/YAML/
  motion digest; per-clip lengths/FPS and audited body mapping.
- Common action/joint/control/observation/schema settings and semantic overrides.
- Actual collection slot policy/counts, phase policy, replay weights/quotas, and
  all trainer/runner seeds/budgets/settings needed for strict resume.

Emit a new **version-2 training checkpoint** for multi-teacher state if extending
version 1 would change its meaning. Keep version-1 train resume and inference
readable; do not reinterpret existing 500–10,000-iteration artifacts. Validate a
whole candidate before mutating model/optimizer/replay/RNG, with rollback tests
for invalid later partitions, changed mapping, corrupted generator state, and
incompatible schema/controls. Keep safe tensor/plain-data loading and atomic saves.

Persist model/normalizers, optimizer, schedules/counters, bounded replay, global
and explicit RNGs, sampling/allocation configuration, and cohort identity. Resume
restarts simulator/action/delay history and creates segment namespaces above all
valid retained records in every partition. Do not claim bitwise simulator resume.

Strict multi-teacher resume requires the same ordered cohort, all hashes, weights,
resource/semantic settings, and replay partition policy, subject only to already
allowed runtime exceptions (total max iterations and reporting/checkpoint cadence).
No single-to-multi conversion via `--resume`.

For inference, add a **checked selection** of one teacher/motion from a saved M4
cohort. The requested member must exist; its artifacts and the common contract
must match. Relocated inference assets are accepted only by the existing content-
hash principle, without relaxing strict resume or stripping stored provenance.
The selected inference report retains the full trained cohort identity. Loading a
shared checkpoint for both motions must produce identical model parameters.

Retain legacy `load_inference_checkpoint` callers. Wire the checked member helper
through evaluation and playback/hot-swap. For export, preserve M3 behavior and
reuse the same selection validation if exporting an M4 member; do not fabricate a
single-teacher training history or claim multi-motion deployment qualification.
If safe M4 export cannot fit the existing seam, refuse it explicitly and report
that limitation to the parent rather than disabling checks.

## 7. CLI, evaluation, and documentation

Preserve existing commands/default single-teacher behavior. Add an explicit
selection such as `distill train --teacher-ids tennis_000 tennis_001`; make
singular/plural selection unambiguous and reject conflicting selections before
constructing a simulator. Do not silently turn today's `train` command into a
large cohort job. The exact implemented Tyro syntax must be documented/tested.

Suggested usage contract (confirm actual help before publishing commands):

```sh
uv run distill train --manifest configs/distillation/x2_tennis.yaml --repo-root . \
  --teacher-ids tennis_000 tennis_001 --num-envs 4 --device cpu \
  --max-iterations 4 --collection-steps 8 --bootstrap-steps 8 \
  --minibatch-size 32 --replay-capacity 256 \
  --seed 7 --output-dir logs/distillation/m4-smoke

# Same M4 checkpoint, separately pinned reference/teacher baseline comparison.
uv run distill evaluate --manifest configs/distillation/x2_tennis.yaml --repo-root . \
  --teacher-id tennis_001 --checkpoint <shared-m4-checkpoint> \
  --mode student --num-envs 4 --steps 512 --sampling-mode start --seed 7
```

Evaluation can use sequential pinned single-motion environments; it need not build
multiple simulators concurrently. Provide a bounded all-selected-motion evaluator
(e.g. additive `evaluate-cohort` command) that emits per-motion reports and aggregate
summaries. Load one student identity, evaluate each teacher baseline and student
with matching seeds/phase mode/resources/perturbations. All evaluation freezes
normalizers and constructs no optimizer or replay.

Always report per-motion failure/completion/survival, reference coverage, timeout,
step-cap and timer censoring, global/root-relative/anchor/heading errors, action
magnitude/rate and teacher disagreement, plus available saturation diagnostics.
Keep metric frame definitions and censoring denominators explicit. Do not claim
`completion_rate=1` over completed-or-failed segments means all initiated segments
completed; retain censored counts and full denominators. Execution errors are
separate from policy failures, and missing reports prevent an overall quality pass.

Add equal-motion macro averages and explicitly **clip-duration-weighted** summaries
(weights proportional to frames/FPS), not mislabeled pooled episode means. Include
raw counts and aggregation weights. A good aggregate must never hide one failing
motion. Every run retains seeds, code/checkpoint/cohort identity, actual environment
counts, phase policy, and mode. Train defaults remain periodic checkpoints every
500 and progress every 10, flushed stderr progress and JSON stdout, named outputs
under `logs/distillation/`.

Extend `docs/source/x2_tennis_distillation.rst`, CLI help/tests, public exports, and
Upcoming changelog. Update the existing export contract document only if a real
cohort-selection behavior is added. Do not add new viewer services or controls;
`distill play --teacher-id <member>` uses existing viewers on a pinned member.

## 8. Exclusive ownership and serial implementation board

Classification: **multi-seam**. Reference/reset mechanics, row routing, replay
sampling, checkpoint identity, and CLI integration have independently testable
contracts. They will be implemented serially in the existing checkout so ignored
real artifacts and the installed uv environment remain available. All stages use
`worktree:false`, the repo/branch/baseline above, and `ds-oc/deepseek-flash`.

| Stage/key | Exclusive source ownership | Gate/handoff | Why separate |
| --- | --- | --- | --- |
| `m4-reference` | New `motion_library.py`, `multi_motion.py`; additive opt-in factory in `environment.py`; corresponding new reference/command tests | Unequal-clip CPU gather/reset/wrap tests; documented row-ID/length/query API | Simulator reference semantics precede routing |
| `m4-routing` | `adapter.py`, `collector.py`; narrow TeacherBank access only if needed; new routing tests and necessary adapter/collector regression tests | Owned snapshot and mixed TeacherBank alignment, per-motion attribution, frozen teachers, fake-env reset probes | Tick/label contract independent of storage/lifecycle |
| `m4-replay` | New `balanced_storage.py`; additive protocol/helper APIs in `storage.py`; narrow trainer typing/validation calls; new balanced-replay tests | Quotas, atomic invalid insert/restore, starvation, seeded draws, once-only fresh normalization | Pure tensor sampling is independent of simulator |
| `m4-lifecycle` | `checkpoint.py`, `runner.py`, new `cohort_contract.py`; new multi-checkpoint/runner tests | Version-1 regressions, version-2 strict identity, transactional restore, deterministic CPU next update | Persistence consumes completed routing/replay contracts |
| `m4-integration` | `scripts/distill.py`, package exports, optional playback/export selection wiring, CLI/integration tests, usage/changelog | Usable train/evaluate-cohort/play surface, end-to-end tiny cohort, static/regression evidence | Integration only after durable component handoffs |
| `m4-review` | Fresh-context read-only reviewer; no source edits | Evidence-backed P0/P1/P2 findings against plan + actual diff | Independent candidate review, not acceptance |
| Parent | Plan/status docs, supervision, resource-gated real validation, finding disposition | Direct source inspection, adversarial probes, live evidence and preservation checks | Final acceptance is not delegated |

Every child reads actual upstream source and managed handoff reports. Return exact
APIs, changed files, test commands/results, remaining limitations, and next action.
Component gates are provisional; only the parent can accept the integrated M4.
Ask before editing another row's owned source. Integration may fix small wiring
mistakes, not silently reimplement a missing component. Shared fixture changes
must be additive and not weaken existing checks.

Protected: robot assets, teacher artifacts/YAML/NPZ/ONNX, `logs/`,
`distillation-runs/`, `tools/` (including the newly committed remote task queue),
dependencies/lockfiles, shared managers/PPO/tracking/viewer behavior, unrelated
changes, and parent-owned planning documents. No staging/commits/pushes, remote
jobs/uploads/installations, hardware access, or child fanout. Test outputs belong
in tmp paths or managed artifacts, never overwrite experiment outputs.

## 9. Validation gates and resource limits

All Python/tools use `uv run` (prefer `--no-sync` for the installed environment).
Headless CPU prefix:

```sh
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable DISPLAY='' WAYLAND_DISPLAY='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Inspection commands: 30 s; individual probes: 60 s; focused suites: 120 s. Use
`pytest -x -vv -o faulthandler_timeout=30`. After an unexpected timeout, inspect and
isolate; do not rerun identically or merely lengthen the timeout. Preserve command
exit codes. Long tests/builds use a parent-owned monitor with logs and exact owned
cleanup, never detached shell loops. Workers ask the supervisor for such runs.

### Gate A: pure CPU and compatibility

1. All new reference/routing/replay/lifecycle/CLI tests pass with distinct two-clip
   fixtures and adversarial reset/mapping cases, not only shape checks.
2. Existing `tests/test_tracking_distillation_*.py` pass, including export/playback
   and saved-schema inference/resume. Include tracking-command/reset tests affected
   by the new command and the CPU runner export subset used for earlier milestones.
3. Changed-file Ruff format/check, targeted Pyright and ty, `git diff --check`.
   Run/report whole-repository checks at integration; identify baseline failures
   from actual evidence (known earlier ty diagnostics in standing-start tests are
   not a blanket exemption). Run required `make check` before any later commit;
   do not auto-fix unrelated code just to obtain a green whole-repo check.
4. Real **both-teacher** CPU inference parity, 64 samples each, existing
   `atol=rtol=1e-5`; unchanged source hashes. No tolerance relaxation.

### Gate B: parent-monitored real simulator smoke

Workers may run pure tensor/fake-env tests and CPU asset compilation. Request
parent approval for live simulation/training (even CPU), GPU construction, or
heavy regression runs; they may prepare exact commands but not run them untracked.
No native/GUI rendering on the laptop.

Initial live envelope, one process at a time:

- CPU/headless preferred; at most 4 environments, with both motion slots present.
- Compile/audit both source motions against one real environment; fixed seeds and
  explicit sampling override/standing-start=0 provenance.
- At most 8 collect/update iterations, 16 collection steps/iteration, 64 bootstrap
  steps, 8 optimizer updates, minibatch at most 64, replay capacity at most 2048.
- One save/resume check of at most 2 extra iterations; no convergence claim.
- Teacher/student evaluations per motion, `start` and `uniform`, seed 7, at most
  4 envs × 512 steps each, using one shared student checkpoint for both motions.
- Total initial live validation ceiling 30 minutes. A monitored first compilation
  may have a 10-minute bound for known compilation cost; normal short checks keep
  the smaller bounds. Stop on resource/physics failures and report evidence.
- GPU use needs a separate concrete parent resource check. Remote production jobs
  remain unauthorized; if later approved, GPU 5 on 10.14.64.37 is defective and
  excluded, and no remote package install may be attempted.

Gate B verifies routing, finite training, resume, and runnable evaluation. A tiny
student can perform poorly; report those metrics honestly. Lack of a live gate
means implementation candidate only, not accepted real-environment integration.

### Gate C: later quality run (planned, not launched by this request)

After implementation acceptance, propose a separately budgeted two-teacher run,
initially matching the existing 10,000-iteration single-teacher scale, with 500-
iteration checkpoints. Fix seed/phase budgets before selecting checkpoints; use
three seeds (7, 11, 17), both start/uniform, both teacher baselines, and all expected
reports. Provisional quality targets to agree before that run: no extra failures
in the bounded matched sweep; per-motion global/root-relative/anchor errors no
more than 10% above the corresponding teacher means (report seed dispersion and
absolute gaps); no hidden degradation in heading/action-rate metrics. These are
engineering targets, not paper thresholds or sufficient rare-fall evidence.
Do not infer a quality pass from training loss or a macro average. Longer robustness,
latent-use diagnostics, sim2sim, and hardware qualification remain separate gates.

## 10. Supervision, rework, and definition of done

Use one top-level async subagent workflow with awaited serial component stages,
then a fresh read-only DeepSeek Flash review. Bind handoff outputs through the
runtime `output` field and return actual output references/run IDs; JSON-normalize
optional metadata rather than emitting `undefined`. Do not relaunch successful
components if a later stage fails.

A persistent 20-minute supervision loop is bounded to 12 hours / 36 checks for
this first implementation window. At each wake, inspect exact workflow/child
status, last activity and transcript, supervisor requests, and read-only diff
progress. Healthy ongoing work needs no interruption. For a real blocker, send
specific instructions/answer the supervisor; verify subsequent activity rather
than treating a queued receipt as compliance. Escalate scope decisions, preserve
partial diffs on infrastructure/provider failures, and resume the same retained
owner/model/protocol where eligible. Never silently switch models/CLIs, start a
second writer, or kill broad process patterns.

The loop stops only on completion/cancellation or its declared bound; unchanged
checks are not completion. If a terminal child needs bounded corrective work,
parent disposition selects the owning component and resumes it; serious findings
receive a follow-up review. Parent-controlled workflow state records handoffs,
corrections, remaining gates, and authority blockers across turns. No source edits
by parent while a worker owns the checkout.

M4 implementation is done only when the source contracts above, legacy regressions,
parent source/adversarial review, real both-teacher audit/parity, and bounded live
train/resume/per-motion evaluation pass, with residual risks explicitly recorded.
The final handoff distinguishes implementation correctness, smoke metrics, later
trained-policy quality, and deployment readiness. It does not claim M5 or diffusion
is complete and does not commit or start production training automatically.

## 11. Parent acceptance record — 2026-09-27

**Disposition:** M4 code paths and bounded real-environment integration are accepted
for proceeding to a separately budgeted two-teacher quality run. The student from
this smoke is NOT a usable trained controller. No commit, remote job, production
training, GUI, GPU run, or hardware validation is included in this acceptance.
Repository HEAD remains `5cb756eed94b5a2353a1b8b4aa93b65f0f160b7a`; M4 changes are
uncommitted. This is not a claim that all repository-wide checks are green.

### Implemented and reviewed

All five serial DeepSeek Flash components delivered candidates; a fresh read-only
review inspected the integrated source. Parent-directed integration corrections
established fresh identically seeded adapters per evaluation mode, consistent
cohort report identities, and exception-safe adapter cleanup.

Parent then reproduced four defects outside the worker's test suite and fixed
them directly: repeated target clip IDs were incorrectly rejected; requested
device aliases did not match allocated tensor devices; report settings still
exposed local clip IDs; and zero-step cohort evaluation was rejected too late.
Parent also added bounded replay telemetry to real train artifacts. The unchanged
four external probes changed from four failures to four passes; repository tests
were strengthened. Retained read-only review found no remaining P0/P1 issue in
these corrections. Its two non-blocking notes are deliberately deferred: legacy
M3 `evaluate --steps 0` behavior and an unused private library-constructor device
argument (the actual allocation now determines device identity).

### Final parent gates

| Gate | Result |
| --- | --- |
| Distillation components plus affected tracking/manager tests | **300 passed, 1 skipped** |
| M4 cohort CLI | **26 passed** |
| Existing single-teacher CLI | **24 passed** |
| Trainer file in isolation | **10 passed** |
| Unchanged external parent counterexamples | **4 passed** |
| CPU runner export subset (earlier parent gate, unchanged source) | **12 passed, 6 deselected** |
| Scoped Pyright, Ruff format/check, `git diff --check` | Pass |
| Real teacher parity, 64 samples each, `atol=rtol=1e-5` | Both pass |

Maximum real-teacher action differences: `tennis_000`
`1.1995434761047363e-6`; `tennis_001` `1.430511474609375e-6`.
Original checkpoint/ONNX association and embedded reference checks also pass.
Protected tracked robot assets, dependency files, and `tools/` have no changes.

**Inherited gate exceptions (not hidden or waived as passes):**

- `make check` was executed and exited **2** at `ty check`: the untouched
  `tests/test_x2_tracking_standing_start.py` has the existing sampling-mode type
  error at line 103 and unused-ignore diagnostic at line 150. Ruff left all 403
  files unchanged and passed; the targeted Pyright command separately passed.
  A pristine baseline reproduced the same two diagnostics. No introduced M4
  diagnostics remain.
- Running storage/model/trainer tests together reproduces the pre-existing
  order-dependent `test_accumulation_matches_one_equivalent_batch_with_controlled_rng`
  tolerance failure. Parent observed the same result, and the pristine baseline
  evidence is retained. The trainer file alone passes. No tolerance was weakened.
- Full `make test` was not run: this acceptance uses the bounded relevant CPU
  suites, not a whole-repository/GPU suite or PR acceptance claim.

### Real headless CPU smoke

Parent Monitor #12 completed with exit **0**, using one process at a time,
4 environments, seed 7, two slots per motion, replay capacity 512, minibatch 32,
accumulation 2 (a smoke resource override, not the production baseline), and
16 bootstrap/collection steps per iteration:

1. Three train iterations; version-2 periodic and final checkpoints saved.
2. Strict resume into a separate directory to lifetime iteration five, exactly
   two new iterations. Replay initially restored 192 records and the event
   explicitly recorded simulator restart and a new segment namespace.
3. Final replay contains 160 inserted/retained/drawn samples for EACH motion;
   both partitions remain ready and all update losses are finite.
4. One shared checkpoint evaluated independently for each motion and teacher/
   student mode, start/uniform sampling, 4 environments × 512 steps per case.
   All **eight** case reports exist, with no execution errors/missing reports.

Raw completed/failure segment counts (timeouts and step caps remain separately
recorded in JSON; these are not rates over every initiated segment):

| Sampling | Motion | Teacher complete / failure | Student complete / failure |
| --- | --- | ---: | ---: |
| start | tennis_000 | 4 / 0 | 0 / 150 |
| start | tennis_001 | 4 / 1 | 0 / 158 |
| uniform | tennis_000 | 7 / 0 | 5 / 119 |
| uniform | tennis_001 | 17 / 0 | 5 / 124 |

The teacher's one start-mode failure for `tennis_001` is explicitly retained.
The five-iteration student fails most segments and covers far less of each clip;
its low-looking average pose errors must NOT be interpreted as teacher parity.
This is successful execution/integration evidence only. Quality acceptance still
requires the separately budgeted training and matched evaluation in Gate C.

### Evidence and follow-up

Parent evidence root:
`/home/agiuser/.pi/agent/mjlab-m4-20260927/`:

- `parent-gate-a/`: initial regression, baseline failures, and real teacher parity.
- `parent-counterexamples-red.log`, `parent-corrections-focused.log`, and
  `test_parent_counterexamples.py`: failing/passing independent probes.
- `parent-final/`: final command logs and exact statuses (`make-check` remains
  nonzero; the focused test/static gates pass).
- `parent-live/`: train/resume checkpoint/report directories, start/uniform
  evaluation JSON, `status.tsv`, and `summary.json`.
- `parent-decisions.md`, `parent-corrections.md`, and canonical structured review
  copies: parent decisions, corrections, and review evidence. Managed text output
  was incomplete for two children; complete structured output was recovered
  without rerunning implementation. Follow-up review reported `OK with notes`.

M4 member playback uses the existing viewers, but no live GUI was exercised.
Version-2 cohort export remains an explicit unsupported operation; version-1
export is preserved. CPU tests and alias fixes are not GPU validation. The next
step is to agree the two-teacher training resource budget and then evaluate saved
checkpoints per motion; do not start that job automatically.
