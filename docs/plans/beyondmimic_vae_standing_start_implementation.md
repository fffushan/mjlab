# VAE standing-start transition learning — implementation plan

Status: **proposed; planning only**. No implementation, training, or teacher-quality
check is authorized or performed by this document. This is an additive M4
extension, not the diffusion milestone.

Repository: `/home/agiuser/projects/mjlab`.
Baseline: the accepted, uncommitted M4 working tree on `fxy/test/tracking-exp`,
HEAD `5cb756eed94b5a2353a1b8b4aa93b65f0f160b7a`.
Related: [M4 implementation and acceptance](beyondmimic_vae_m4_implementation.md),
[architecture](beyondmimic_vae_distillation.md), and
[usage](../source/x2_tennis_distillation.rst). This plan extends the M4 cohort
path; the M4 acceptance record above it does not mention or depend on it.

Notes:

- Predecessor milestone (M4) was committed separately; this document and its
  implementation are not part of that commit.

## 1. Goal, authority, and non-goals

Teach the shared conditional VAE to enter each selected tennis motion from the
robot's standing pose, while retaining ordinary full-motion tracking coverage.
Real deployment starts from standing; reference-frame initialization alone does
not train or validate this transition.

**User decision: teacher qualification is external.** The user manually selects
qualified teachers for real training. Do not add a teacher transition rollout,
success threshold, teacher-quality certificate, or teacher retraining prerequisite
to distillation. Standalone teacher-quality checks are deferred. Existing artifact,
reference, observation/control compatibility, and numerical parity checks remain
unchanged; these are not closed-loop teacher-quality checks.

In scope:

- Opt-in standing-start collection on the M4 cohort path, including a singleton
  cohort selected with `--teacher-ids "('tennis_000',)"`.
- Per-clip reset sampling and faithful standing-state initialization.
- Reset provenance, coverage reporting, and strict checkpoint/resume semantics.
- Explicit **student** standing-start evaluation, distinct from reference starts.
- CPU tests and a separately approved, bounded headless integration smoke.

Out of scope:

- Changing registered PPO tasks, legacy M3 training, VAE architecture/features,
  objective, normalizers, teacher/student mixing schedule, or control timing.
- Teacher quality assessment/retraining, reference blending or prepended standing
  frames, a transition controller, diffusion, clip-to-clip transitions, automatic
  replay reweighting, or a new training curriculum.
- Cohort export, deployment controller changes, GUI, GPU/remote production runs,
  sim2sim, hardware, commits, or pushes.

No subagent workflow, recurring loop, or training job is created by this plan.
Implementation and resource approval are subsequent actions.

## 2. Existing behavior and implementation seams

Verified source behavior:

- `mdp/commands.py:MotionCommand._resample_command` samples a reference frame,
  obtains root/joint pose and velocity, optionally substitutes a standing state,
  applies configured perturbations, clips joint positions, and writes simulator
  state. `_select_standing_start_envs` implements the PPO eligibility test.
- `config/agibot_x2/env_cfgs.py` sets the standing-start PPO task's conditional
  probability to 0.3 and inherits a 25-frame window. PPO caps that window at its
  first sampling bin and exempts `sampling_mode="start"`.
- `distillation/multi_motion.py:MultiMotionCommand` already supplies per-row clip
  reference queries, lengths, routing, and uniform phase sampling. It currently
  returns an all-false standing mask and rejects nonzero standing probability.
- `distillation/environment.py:build_multi_motion_environment` also refuses the
  standing-start PPO task. `SegmentMotionCommand` distinguishes full environment
  resets, timer resamples, wraps, and explicit frame teleports.
- `adapter.py` and `collector.py` capture owned PRE-step row metadata, label actual
  visited states, and attribute outcomes to the segment that just ended.
- `balanced_storage.py` balances motions, not initialization categories.
- `cohort_contract.py` and `checkpoint.py` persist strict cohort identities and
  version-2 checkpoints. Version-1 single-teacher behavior must remain intact.
- `scripts/distill.py:evaluate-cohort` already supports `--mode student`; existing
  `start` means **reference frame 0**, not standing. Its pinned evaluation uses a
  single-motion adapter, so merely editing the training command is insufficient.

The current tennis manifest uses the ordinary Reduced-Perturbations task, not the
Standing-Start task. Its reset perturbations are already halved from the generic
tracking defaults, including joint-position noise of +/-0.05 rad. Preserve those
resolved settings; do not accidentally restore generic defaults.

## 3. Proposed reset contract

### 3.1 Explicit opt-in; preserve legacy defaults

Add a validated, plain-data, distillation-owned reset configuration, suggested
module `distillation/reset_policy.py`.

Proposed training interface (names to be verified through Tyro help/tests):

```text
--reset-policy reference | standing-mixture       # default: reference
--standing-start-fraction 0.25                    # when enabled
--standing-start-window-frames 25
--standing-start-frame-zero-fraction 0.5
```

The 25% mixture and 50% frame-zero branch are proposed starting hyperparameters,
not measured optimal values. They must be configurable and reported. Do not reuse
PPO's `standing_start_prob` name for this different, unconditional mixture.

Reject invalid/nonfinite probabilities, nonpositive/noninteger windows, and
conflicting settings before resolving artifacts or constructing a simulator.
Reject new training options on the legacy singular M3 path with an actionable
message pointing to singleton cohort syntax. With no new options, preserve the
old reset behavior, checkpoint format, and RNG consumption.

Do not silently enable this extension because a teacher's saved YAML contains
standing-start fields. Build from the trusted base task with an explicit private
reset policy; record intentional reset overrides separately from immutable
teacher provenance. Supporting arbitrary nonzero-standing registered base tasks
is unnecessary for this extension and may remain an explicit refusal.

### 3.2 Collection mixture, clip-local frames

Apply the mixture independently to rows undergoing a **full environment reset**:
initial reset, termination/timeout reset, resumed simulator reset, or explicit
motion selection that performs a full row reset. Keep the motion slot and teacher
routing unchanged unless the caller explicitly selects a new motion.

For row i, with assigned clip length F_i:

```text
W_i = min(configured_window_frames, F_i)
standing = Bernoulli(standing_start_fraction)

if not standing:
    frame = UniformInteger(0, F_i - 1)
    physical initialization = reference state at frame
else:
    if Bernoulli(standing_start_frame_zero_fraction):
        frame = 0
    else:
        frame = UniformInteger(0, W_i - 1)
    physical initialization = standing state
```

The uniform early-window branch includes frame 0. Therefore the actual frame-zero
probability conditional on standing is `a + (1-a)/W_i`, not exactly `a`, where a is
the configured frame-zero branch fraction. Report the realized histogram.

This intentionally differs from first sampling the entire clip and then applying
PPO's 0.3 eligibility probability. The latter yields only `0.3 * W_i / F_i`
standing starts under uniform sampling. Use a clip-local window, not adaptive bins:
`min(25, F_i)` is the new contract. It agrees with the existing 25-frame window for
the current tennis clips but remains explicit for short or future clips.

Timer resamples, natural reference wraps, and `reset_to_frame` are **not deployment
starts**: retain their existing reference-state teleport semantics. Do not reset
all environment managers on a wrap to manufacture standing starts. Report these
boundaries separately; the configured fraction is over eligible full resets, not
all command generations or collected transitions.

Sampling one reset branch must not trigger a second probability test in the
inherited standing-mask hook. Sample and apply one decision per affected row;
non-reset rows must retain their pose, frame, metadata, and history.

### 3.3 Physical initialization

Reuse the existing PPO standing-state semantics without changing shared PPO code:

- Joint positions: entity default standing pose, in audited joint order.
- Root xy: reference root xy at the selected frame plus environment origin.
- Root z: entity default standing height plus environment origin z.
- Orientation: upright with yaw from the selected reference **anchor** quaternion,
  not an assumed root/anchor equivalence.
- Joint, root-linear, and root-angular velocities: zero before perturbations.
- Apply the existing task's pose, velocity, and joint-position perturbations once;
  preserve joint soft-limit clipping and root/joint write semantics.
- Leave reference arrays, target frame, lookahead, and normal motion advancement
  unchanged. No standing prefix, phase hold, pose interpolation, or assistance.

Prefer a private command hook/helper over copying the entire shared reset routine.
Use the pending standing mask in the inherited state-writing path where feasible.
If factoring shared code becomes unavoidable, require behavior-neutral PPO and M3
regressions; do not change their selection probability or `start` exemption.

Full resets must preserve existing simulator forward/sense, action-buffer,
observation-delay/history, and cached-feature reset ordering. The first teacher
label and student action must see the post-standing-reset state and intended
reference frame. Do not seed previous-action features with a teacher action or
recompute noisy observations independently for teacher and student.

Audit the resolved standing pose against the existing policy default-offset
contract. Do not claim that this alone verifies the deployed controller's history,
state estimator, or physical hardware readiness.

### 3.4 RNG and lifecycle

Choose one owned reset RNG stream for the enabled path, sampled on CPU then moved
to the target device if necessary; this avoids promising CUDA RNG persistence that
the current CPU-global checkpoint field does not provide. Seed it before the first
reset and persist/restore its state transactionally with the collector/lifecycle.
Do not consume extra RNG draws when the extension is disabled.

Resume restores the saved stream, then performs the existing fresh simulator reset
and new segment namespace. That reset consumes new samples; it is not continuation
of a saved physical standing transition. Do not promise bitwise simulator resume.

## 4. DAgger, attribution, replay, and coverage

Frozen teachers label the actual standing and student-visited states through the
existing observation snapshot and routing path. No extra teacher rollout, quality
gate, teacher update, or reward objective is introduced.

Add owned per-row provenance sufficient to identify:

- initialization kind: reference or standing (and legacy/unknown where needed);
- segment initial reference frame;
- number of executed steps since that segment began;
- existing motion, teacher, segment, and generation IDs.

Carry the provenance from command to snapshot, collection boundaries, labeled raw
replay, and evaluation segment. It must describe the PRE-step state; an autoreset
must not relabel the preceding failure sample as a new standing sample. A frame
index below 25 alone is not evidence of standing initialization.

For reports, define an early-transition sample as a sample from a standing-start
segment with segment age `< configured_window_frames`. This is a reporting window,
not a claim that physical convergence occurs in 25 steps. Keep whole standing-origin
segment counts distinct from these early samples.

Extend the balanced replay's enabled layout with bounded aligned provenance, and
preserve legacy batch constructors/layouts through optional fields or explicit
layout dispatch. Missing enabled-path metadata is an error; never infer standing
from a legacy frame index. Validate all new tensors before insertion or restore.

Do **not** add standing/reference replay quotas in the initial extension. Preserve
motion quotas, per-motion FIFO retention, within-motion sampling, and once-per-fresh-
data normalizer updates. Measure whether standing transitions survive into updates:

- Eligible resets and realized standing/reference counts per motion.
- Standing frame-zero and early-window start counts/histograms per motion.
- Collected, retained, and drawn samples by motion and initialization kind;
  report early-transition counts separately.
- Failures, timeouts, wraps, and censored segments by initialization kind.

Use bounded counters/histograms, not unbounded per-step logs. Make clear that a 25%
reset fraction does not imply 25% of replay or gradient updates. Any later replay
stratification or fraction tuning is an explicit follow-up, not an automatic
response to poor student metrics.

## 5. Persistence and compatibility

Enabled runs need a new, versioned reset/provenance contract. Proposed format:
**version-3 cohort checkpoints** for the extension, with a versioned cohort identity
and balanced-replay layout. Keep disabled M4 runs on the existing version-2 path,
and keep M3 version-1 semantics intact.

Persist and compare on strict resume:

- Reset-policy kind/version, mixture fractions, configured and per-clip effective
  windows, frame-zero branch, and full-reset versus wrap/timer behavior.
- Resolved standing joint pose, height, yaw-alignment rule, perturbation ranges,
  and their control/asset provenance, or an equivalently complete canonical record.
- Reset RNG state, replay provenance layout/data/counters, and existing model,
  normalizers, optimizer, schedules, source hashes, slots, and resource settings.

Validate the entire candidate before mutating any owner. Extend rollback tests to
new metadata, counters, and reset RNG; preserve safe tensor/plain-data loading and
atomic saves. Equal reset policies must canonicalize and hash identically.

Compatibility matrix:

| Input | Required behavior |
| --- | --- |
| v1 legacy training/inference/export | Unchanged |
| v2 reference-only M4 resume/inference | Unchanged; no new standing semantics inferred |
| v2 resumed with standing enabled | Reject strict resume with a clear policy/version error |
| v3 same-policy resume | Restore all durable state; restart simulator explicitly |
| v3 changed policy/window/fractions/pose/perturbations | Reject before partial restore |
| v3 checked member inference | Keep trained reset provenance; allow explicit evaluation-only reset profiles |
| v2/v3 cohort export | Continue explicit unsupported refusal |

Weights-only warm starts or v2-to-v3 training migration are not part of this work.
Update every checkpoint-version dispatcher, including CLI evaluation/playback and
hot-swap loading, so a v3 cohort cannot fall into a legacy single-teacher path.
Playback need not gain standing-start controls in this extension, but v3 model
loading and existing reference playback must remain well-defined and tested.

## 6. Student evaluation, separate from teacher qualification

Add explicit reset profiles for checked cohort-student evaluation:

| Profile | Physical initialization | Reference start |
| --- | --- | --- |
| Existing reference-start | Reference pose | 0 |
| Existing reference-uniform | Reference pose | Uniform over whole clip |
| New standing-start | Standing pose | 0 |
| New standing-window | Standing pose | Uniform over first min(25, F_i) frames |

Proposed option: `--reset-profile standing-start|standing-window`, with absence
preserving existing `--sampling-mode` behavior. Reject conflicting explicit phase
options rather than silently treating whole-clip `uniform` as early-window
sampling. Expose the window and a clean/configured reset-perturbation choice in
reports; clean means zero **reset** perturbations, not disabling all sensor noise
or domain randomization.

Wire new profiles through `evaluate-cohort --mode student` and checked cohort-member
`evaluate --mode student`. Use a singleton multi-motion adapter or a shared private
reset helper so training and evaluation cannot implement different standing states.
Keep existing reference evaluation paths and mode defaults backward compatible.
Do not schedule `teacher` or `both` transition evaluations or require their results.
The existing instantaneous teacher-disagreement diagnostic, if retained, is not a
closed-loop teacher test and must not become a qualification criterion.

For standing profiles, a trial begins with a full environment reset. Score each
row's initial segment until its first failure, timeout, reference completion,
external interruption, or step cap. Do not count later reference-wrap teleports as
additional standing trials. Repeated trials require explicit full resets and a
bounded, recorded seed/trial schedule. One simulator at a time; normalizers frozen,
no replay insertion or optimizer, guaranteed cleanup on errors.

Report per motion/profile/seed:

- Trial count, actual initial frame/pose provenance, and perturbation settings.
- Failure and survival counts through the first 25 executed steps; raw denominators,
  short-clip completions, and censored trials reported separately.
- Tracking error at initialization and over the early transition, followed by
  subsequent tracking/completion. Initially large reference error is expected.
- Action magnitude/rate, including the first command's change from the known
  standing hold where the action contract establishes it. Report available
  saturation diagnostics without inventing unavailable telemetry.
- All original failure/completion/timeout/step-cap distinctions and cohort IDs.

Do not auto-disable terminations, increase failure thresholds, or introduce a
standing grace period. State-only tests should check reset alignment and immediate
termination margins; an actual immediate termination is evidence to report, not a
reason to hide failures. Any termination-policy change needs separate approval.

Training's existing periodic evaluation may continue to report its configured
training reset distribution. Do not label that mixed evaluation as a deterministic
standing-start quality pass. Dedicated student profiles are explicit evaluations.

## 7. Ordered implementation work and gates

This is a serial dependency plan, not a running task backlog. Keep one source writer
and preserve the existing uncommitted M4 implementation and unrelated artifacts.

| Phase | Main files | Required gate before moving on |
| --- | --- | --- |
| 1. Reset contract | New `reset_policy.py`; config/CLI validation tests | Pure CPU policy validation, effective windows, seeded sampling, legacy-off RNG parity |
| 2. Command/factory | `multi_motion.py`, `environment.py`, `adapter.py` | Exact standing state, unchanged reference, per-row subset reset, reset/wrap separation, no duplicate boundary |
| 3. Provenance/coverage | `adapter.py`, `collector.py`, `storage.py`, `balanced_storage.py` | PRE-step labels/metadata, FIFO/sample alignment, per-motion counters, frozen teacher/normalizer regressions |
| 4. Lifecycle | `cohort_contract.py`, `checkpoint.py`, `runner.py` | v1/v2 compatibility, v3 round trip, strict mismatch and transactional rollback, seeded next-draw/update checks |
| 5. CLI/student evaluation | `scripts/distill.py`, private evaluator seams, inference/playback dispatch | Tyro contracts, standing student profiles, trial attribution/censoring, v3 checked-member loading |
| 6. Documentation/integration | Usage, Upcoming changelog, this plan's acceptance record | Static/regression evidence and separately approved headless smoke |

Add focused tests, suggested files:

- `tests/test_tracking_distillation_reset_policy.py`
- `tests/test_tracking_distillation_standing_start.py`
- Existing multi-motion/routing/balanced-replay/cohort-checkpoint/cohort-CLI and
  legacy adapter/collector/lifecycle/playback/export suites.

Mandatory adversarial coverage:

1. Distinct clips of unequal lengths, including fewer than 25 frames and one frame;
   first/last eligible frames, invalid probabilities/windows, singleton and 3+ clips.
2. Fractions 0/1, frame-zero branch 0/1, controlled random draws, and seeded
   distribution checks over reset calls (not physical training rollouts).
3. Upright yaw alignment to the anchor rather than root; nonzero scene origins;
   zero velocities before noise; exact configured perturbations applied once;
   joint clipping and immutable reference arrays/returned-state ownership.
4. Partial resets and repeated target motion IDs; unaffected rows unchanged;
   wrap/timer/full-reset collisions classified once with correct generations.
5. First action observes standing and the selected frame, with no off-by-one
   advancement, stale cached features, or double observation-history advance.
6. Two distinct synthetic teachers prove row routing and PRE-step labels through
   standing resets/autoresets; student-visited states remain the supervision input.
7. Provenance survives replay truncation, sample shuffling, copy ownership, save/
   resume, and segment rebasing. Invalid late partitions cannot partly mutate data.
8. Legacy disabled behavior and checkpoint readers remain unchanged; reset-policy,
   default-pose, RNG, and malformed provenance mismatches fail transactionally.
9. Standing student evaluations do not launch teacher rollouts; a reference reset
   cannot be reported as standing; post-wrap segments cannot inflate trial counts.
10. Existing PPO Standing-Start task semantics, old `start` evaluation, and M3
    observations/actions/export behavior remain unchanged.

## 8. Validation envelope and definition of done

All Python tooling uses `uv run` (prefer `--no-sync` for the installed environment).
Use the headless prefix:

```sh
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=disable DISPLAY='' WAYLAND_DISPLAY='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Inspection: 30 seconds; individual tests: 60 seconds; focused suites: 120 seconds.
Use `pytest -x -vv -o faulthandler_timeout=30` when diagnosing failures. Unexpected
timeouts require inspection and isolation before retry. Preserve exit statuses.
Long/real-simulator jobs need independently auditable monitors, bounded budgets,
logs, and cleanup. No GUI or uncontrolled GPU initialization on this machine.

Pure-code gates: new tests, affected legacy tests, scoped Ruff/Pyright/ty, and
`git diff --check`. Report whole-repository gates honestly. M4's recorded baseline
exceptions are the two `ty` diagnostics in `test_x2_tracking_standing_start.py` and
the order-dependent trainer accumulation test; do not weaken tests or claim those
checks pass. Recheck applicable evidence rather than treating the old record as a
blanket exemption. No commit while required commit gates fail.

Proposed later integration smoke, **not launched by this planning request**:

- One headless CPU process, at most 4 environments, both existing motion slots.
- State-only forced-standing/forced-reference probes, plus short mixed-reset
  collection proving both branches per motion. No teacher quality sweep.
- At most 3 initial train iterations plus 2 resumed iterations; 16 bootstrap and
  collection steps, one update/iteration, minibatch <=32, accumulation <=2, replay
  <=512. These are smoke resource settings, not production defaults.
- Save v3, strict resume into a separate output directory, verify namespace/reset
  provenance and finite losses. Exercise p=0/off legacy regressions separately.
- Student-only standing-start and standing-window evaluations of the same saved
  model, sequentially per motion, <=2 environments x 512 steps/profile, seed 7.
- At most 30 minutes total; a first compilation may have a separately justified
  10-minute monitored bound. Stop on infrastructure/physics failures, preserve
  evidence, and do not lengthen budgets automatically.

Teacher actions used as DAgger labels are ordinary pipeline execution, not a
teacher transition test. A tiny/untrained student may fail every standing trial;
that does not invalidate execution tests and must not be reported as quality
success. No training job waits for a teacher transition-quality result.

**Implementation done:** the reset contract, provenance, compatibility, and student
evaluation tests pass; the approved smoke demonstrates correct execution/resume;
residual failures and limitations are documented. The accepted old M4 evidence is
preserved and the new extension has its own acceptance record.

**Not implied:** trained-policy transition competence, convergence, sim2sim or
hardware readiness. A later user-authorized quality run uses manually qualified
teachers and evaluates each student's standing entry and full-motion tracking;
resource budgets and quality thresholds are agreed for that run, not invented as
a mandatory teacher gate here.

## 9. Parent acceptance record — 2026-09-27

**Disposition:** the implementation is accepted for the code paths it covers, with
the bounded real-simulator smoke below. The student from this smoke is NOT a usable
controller: it fails most standing trials. No commit, production training, GPU,
GUI, remote job, sim2sim, or hardware validation is included.

Repository: `/home/agiuser/projects/mjlab`, branch `fxy/test/tracking-exp`,
baseline `451d432a685792ea740a977086eb0bc656a26d6d` (M4 committed as `433a53919`).
Evidence root: `/home/agiuser/.pi/agent/mjlab-standing-20260927/`.

### Implementation and workflow

Seven serial `worker` stages plus one fresh read-only `reviewer` ran on
`lingzhi/gpt-5.6-luna` under mission `8698b3e3-6c01-4fe0-9ad4-aad3f4c09aab`. The
first workflow (`f494e10e`) failed at `ss-command` for an orchestration reason
only: the child produced its structured handoff but never wrote the bound
file-only output. Its code was complete and the parent verified it directly before
continuing. The second workflow (`526cfae0`) resumed that exact retained child to
close the coverage gaps and bind the handoff, then ran only the unlaunched stages.
The orchestration defect was fixed for the continuation (inline output plus a
verdict resolver unit-checked against ten shapes, including the observed failure).

Delivered: `reset_policy.py`, opt-in `standing-mixture` resets in
`MultiMotionCommand` with an owned CPU reset RNG and `reset_rng_state` /
`set_reset_rng_state`, `InitializationKind` provenance through snapshot,
collector, both replay layouts and evaluation segments, version-3 cohort
checkpoints with strict-compatible `ResetProvenance`, CLI options
(`--reset-policy`, `--standing-start-fraction`, `--standing-start-window-frames`,
`--standing-start-frame-zero-fraction`, `--reset-profile`,
`--reset-perturbations`), and documentation. Reference-only behavior is the
default and is byte-identical: a reference run still writes a version-2
checkpoint and gains no trial keys.

No teacher-quality gate, teacher rollout, competence threshold, retraining, or
termination weakening was added. Teacher qualification stays manual and external.

### Independent review and disposition

The read-only reviewer returned `blocked` with three verified findings, all fixed
by the parent: wrap/timer teleports were counted as fresh standing trials; the
standing evaluation lacked the required early-window buckets, real initial
provenance, and clean/configured perturbation reporting, and `segment_age` was a
constant `0`; and a top-level `--help` assertion had been deleted. See
`parent-decisions.md` for the full table.

The parent additionally found what the children missed: this milestone introduced
two `ty` diagnostics (an unused `type: ignore` and an unsupported dynamic class
base), the review's rekey correction had a second instance in the new
`trial_provenance` map (keyed by pre-rekey local ids, misattributing one member's
trials to another), and one diagnostic of the parent's own. `ty` is now back to
exactly the two inherited baseline diagnostics in the untouched
`tests/test_x2_tracking_standing_start.py`; pyright is clean.

### Gate A — pure CPU and compatibility

| Gate | Result |
| --- | --- |
| Distillation and affected tracking/manager suites | **343 passed, 1 skipped** |
| Cohort CLI | **37 passed** |
| Legacy single-teacher CLI | **24 passed** |
| Trainer in isolation | **10 passed** |
| Unchanged PPO standing-start task tests | **10 passed** |
| Unchanged M4 counterexample probes | **4 passed** |
| Ruff format/check, scoped Pyright, `git diff --check` | Pass |
| `ty check` | Exit 1 — only the two inherited baseline diagnostics |

### Gate B — bounded headless CPU smoke (all steps exit 0)

**The first Gate B run failed its own validation and is retained as evidence.** Its
seven execution steps exited 0 and its validator step exited 1 with three
attributions: (a) `trial_provenance` was keyed by pre-rekey local motion ids, so
one member's trials were published under another member's identity - a real
defect in this milestone, fixed in `_rekey_mode_report` with a regression test;
(b) the validator then required every motion to collect standing samples at 4
environments and a 0.25 fraction, which a short smoke cannot guarantee and which
the sampler check below shows is chance; and (c) the validator read
`per_motion` provenance under the pre-rekey key. The validator was corrected to
assert per-motion coverage only in a deliberately sized leg, and the whole gate
was re-run on the fixed code; only that re-run is reported below.

One process at a time, CPU and headless, `seed 7` unless noted, 4 environments,
replay 512, minibatch 32, accumulation 2, 16 bootstrap/collection steps.

1. Enabled standing-mixture training, 3 iterations; every update loss finite; the
   resolved policy records kind/fraction 0.25/window 25.
2. Strict resume of that version-3 checkpoint into a separate directory, exactly
   two further iterations to lifetime iteration 5, with `resumed_reset` recorded
   on the first resumed iteration.
3. Enabled checkpoint is version **3**; the reference-only regression checkpoint
   is version **2**, written by a run that never selected a standing option.
4. Deliberate coverage leg (8 environments, fraction 0.5, `seed 11`, 4
   iterations): both motions collected standing samples (0: 128, 1: 128).
5. Reset-sampler unbiasedness against the real clip lengths: per-row standing rate
   over 4000 reset rounds `[0.2412, 0.2488, 0.2510, 0.2548]`.
6. Student standing-start evaluation of the same shared checkpoint, configured
   perturbations, 2 environments x 512 steps: 24 trials (motion 0) and 22 trials
   (motion 1), buckets summing to the trial count, provenance recording standing
   initialization and real initial frames.
7. Student standing-window evaluation with `--reset-perturbations clean`:
   15 and 12 trials, `clean` reported true from the live command.
8. The same version-3 student with no standing profile: no trial metrics, no
   trial provenance, and no perturbation record — the reference path is untouched.

**Collection-coverage finding (reported, not hidden):** at the 4-environment
0.25-fraction leg, one motion received **zero** standing samples while the other
received 48, because a short smoke performs very few full resets. This is chance,
not sampler bias: the sampler is position-unbiased (item 5) and the effect
disappears with more rows (item 4). A real training budget should size the
standing fraction against the number of *resets*, not against collected samples,
and should verify per-motion standing coverage early instead of assuming it.

**Smoke metrics are execution evidence, not policy quality.** The five-iteration
student fails most trials: motion 0 - 6 within-window failures, 16 after it, 2
survivals; motion 1 - 3 within-window failures, 17 after it, 0 survivals. Trial
counts are over full resets only; later wrap/timer teleports are continuations and
were excluded from those denominators.

### Residual risk and unrun work

- No trained-quality run, convergence claim, or baseline-relative comparison.
- No GPU, remote, sim2sim, hardware, or GUI validation; version-3 cohort export
  remains an explicit unsupported refusal.
- `reset_provenance_from_adapter` is now exercised against a real enabled command
  in the smoke's save/resume path, but only at 4-8 environments and 5 iterations.
- No committed gate was run; the working tree is uncommitted by design.

## 10. Addendum — local 8192-environment run, 2026-09-27

A user-authorized bounded smoke ran **locally** on the RTX 5080 Laptop GPU
(16303 MiB), not on a remote box, so no remote job or fleet allocation is
involved. It is the first real validation of the enabled path at scale.

Command shape: cohort training with `tennis_000` and `tennis_001`,
`--reset-policy standing-mixture --standing-start-fraction 0.25
--standing-start-window-frames 25 --standing-start-frame-zero-fraction 0.5`,
`--device cuda:0 --num-envs 8192 --seed 7`, `bootstrap 128`, `collection 32`,
`4` updates of `minibatch 4096` with `accumulation 8`, `replay 262144`,
`checkpoint-every 50`, `max-iterations 200`. Code state: HEAD
`451d432a685792ea740a977086eb0bc656a26d6d` with the uncommitted milestone diff;
sha256 hashes of the six changed source files are recorded in the run log.

| Metric | Result |
| --- | --- |
| Iterations | 200/200, exit 0, all losses finite |
| Wall time | 626 s (3.1 s/iteration), plus a 19 s one-iteration pre-flight |
| Peak VRAM | 9247 MiB of 16303 (57 %), 59 C after, GPU idle afterwards |
| Checkpoints | iterations 50/100/150/200 plus final, ~293 MB each |
| Loss / disagreement | 83.76 -> 1.478 / 1.17 -> 0.146 |
| Standing samples, `tennis_000` | 8,386,779 of 26,607,616 (31.5 %) |
| Standing samples, `tennis_001` | 7,569,642 of 26,607,616 (28.4 %) |

At this scale the sparse per-motion standing coverage reported in section 9
disappears: both motions receive dense standing supervision, which is the
concrete reason to size the fraction against reset counts at the intended
environment count rather than at a small smoke size. The realized standing
**sample** share exceeds the configured 25 % per reset, and the run's own report
states the limitation that unchanged: the fraction applies to eligible resets,
not to replay or gradient updates.

Student standing evaluation of that checkpoint, student-only as the contract
requires (teacher/both transition evaluation is refused), 4 environments x 512
steps, seed 7, matched between profiles:

| Profile | Motion | Trials | Failed < 25 | Failed > 25 | Survived 25 | Censored | Continuations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standing-start | `tennis_000` | 8 | 0 | 2 | 4 | 2 | 2 |
| standing-start | `tennis_001` | 9 | 0 | 3 | 5 | 1 | 3 |
| standing-window, clean | `tennis_000` | 9 | 0 | 3 | 4 | 2 | 2 |
| standing-window, clean | `tennis_001` | 10 | 0 | 4 | 5 | 1 | 2 |

Reading, with the limits stated rather than implied: **no trial failed inside the
25-step entry window in any case**, so the standing-to-motion entry does not
destabilize this student immediately; surviving 25 steps is 0.5 s at 50 Hz and is
an entry check, not evidence of tracking. Failures do occur after the window
(2-4 per motion), and `failure_rate` over completed-or-failed segments is
0.50-0.71, so most trials that reached a known outcome still failed. Four
step-cap and one to two timeout censored segments per motion show the 512-step
evaluation budget is too short to observe long trials; those are excluded from
the rate denominators and reported separately. Provenance confirms the profiles
work as designed: standing-start trials all report initial frame 0, and
standing-window trials report frames 4-22 with `clean` true. Continuations are
excluded from trial counts (for example motion 0 holds 10 segments but 8 trials).

No teacher baseline exists by design, so no student-versus-teacher standing claim
is made. This remains execution and entry-behavior evidence, not policy quality,
and not a sim2sim or hardware claim.
