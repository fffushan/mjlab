# X2 VAE difficult-state coverage — implementation plan

Status: **full P1–P6 implementation is not complete; the bounded MVP path is implemented and validated in the primary checkout**. The MVP captures physical terminal states before reset, labels them once with the frozen teacher, inserts them into ordinary replay, and keeps the existing VAE objective. It requires a passing restore-parity report. Multi-step teacher branches, runtime coverage perturbations and protected replay remain deferred.

MVP evidence: ``logs/diagnostics/vae-coverage-mvp-20261008/`` contains the
restore-parity report and bounded training/resume smoke artifacts. The smoke is
simulator and lifecycle evidence only. It is not policy-quality, closed-loop,
hardware or deployment evidence.

Repository: `/home/agiuser/projects/mjlab`, branch `fxy/test/tracking-exp`.
Observed planning baseline: `8dcd46dde7ca5dddd246245d4c3b15b19fdafc92`, clean tracked
worktree. Recheck before implementation; other workstreams may advance HEAD.

**P0 addendum (2026-10-01):** [source inventory and proposed resolved contracts](beyondmimic_vae_difficult_state_coverage_p0.md),
[declarative design YAML](beyondmimic_vae_difficult_state_coverage_p0_contract.yaml).
P0 documentation/settings are confirmed; live capture/restore/dynamics support
remains a P2 gate. The directional profile,
no-new-disturbance teacher branches and explicit budget/queue rules in that
addendum refine the tentative ranges below. YAML is design data, not a runnable
configuration. No source implementation or simulation/training was performed.

## 1. Goal and fixed decisions

Expand the states on which the student faithfully reproduces its frozen teacher,
especially balance-relevant deviations poorly represented by ordinary tracking.
Keep teacher capability a separate question.

Implement an opt-in cohort collection path with:

1. Physically coherent root and leg/waist perturbations, sampled at multiple phases.
2. Preservation of valid reached terminal states before auto-reset destroys them.
3. Bounded teacher-controlled branches restored from student-visited difficult
   snapshots, including valid samples from branches in which the teacher fails.
4. Protected difficult-state replay with explicit storage and sampling budgets.
5. Fixed held-out fidelity evaluation and coverage reporting.

**Do not change:** teacher weights/normalizers; student architecture or observation
schema; the existing reconstruction-plus-KL objective, its reduction or beta;
per-joint loss weights; success/failure loss weights; ordinary task control timing,
gains, torque limits, collision assets, or deployment controller behavior.

Outcome metadata is diagnostic only. A valid teacher-failure label has the same
per-sample reconstruction weight as a teacher-success label. Sampling changes the
training distribution intentionally, but never selects or weights by teacher
success. Do not filter valid failed trajectories out of the dataset.

Frozen manifests/models, raw hardware data, and existing replay evidence stay
immutable. Hardware-derived states remain diagnostic holdouts, not training seeds.
Old checkpoint strict resume must not silently acquire new collection semantics.

## 2. Evidence and current implementation seams

Relevant evidence:

- [Historical retention audit](../../logs/diagnostics/vae_historical_state_retention_20261001.md).
- [Fixed hardware-state checkpoint replay](../../logs/diagnostics/vae_teacher_student_checkpoint_replay_20261001/REPORT.md).
- [Retained-replay cross-evaluation](../../logs/diagnostics/vae_checkpoint_replay_cross_eval_20261001/REPORT.md).

The later students improved on earlier checkpoints' retained simulator replay,
while all tested checkpoints disagreed substantially on the hardware-failure
trajectory. This supports inadequate coverage/learning of that region; it does
**not** establish that those states were never encountered anywhere in training.

Verified source behavior:

| Component | Existing behavior / implication |
| --- | --- |
| `distillation/collector.py:DAggerCollector.collect` | Inserts teacher-labelled PRE-step snapshots before executing student actions. Near-failure rows and the row whose next step terminates are already retained. |
| `envs/manager_based_rl_env.py:step` | Terminations/rewards and events run before reset; resetting rows never receive a post-action terminal observation. The subsequent forward/command/sense/observation pass sees reset states. |
| `managers/recorder_manager.py:record_pre_reset` | Has intact executed action, but cached `obs_buf` is the PRE-step observation, not a newly reached terminal observation. |
| `distillation/adapter.py` / `observations.py` | Own aligned policy snapshots. An observation snapshot is not a restorable simulator snapshot. |
| `distillation/multi_motion.py` | Per-row phase/routing, reset bookkeeping, and standing/reference reset logic. Current reset noise is scalar across joints. Wrap/timer resamples can teleport physical state. |
| `distillation/balanced_storage.py` | Per-motion FIFO; optional phase sampling does not protect difficult states. |
| `distillation/trainer.py` | Normalizers update once from unique fresh rows; replay resampling must not update them. |
| `distillation/cohort_setup.py`, `worker.py`, `sharded_collection.py` | Shared construction recipe, parent-owned replay, deterministic worker identity, and atomic reply validation. |
| `distillation/checkpoint.py`, `cohort_contract.py`, `runner.py` | Versioned strict resume; physical simulator is restarted on resume, not checkpointed as a live rollout. |
| `sim/sim.py`, `entity/entity.py`, `actuator/`, `utils/buffers/` | No verified complete distillation branch snapshot API. Dynamics, per-world model randomization and history/delay buffers must be audited before restore is trusted. |

Paths in this table are relative to `src/mjlab/`.

## 3. Proposed architecture

```text
ordinary student collector (existing auto-reset semantics)
    |
    +-- coherent perturbation events / reset perturbations
    +-- existing PRE-step teacher-labelled rows ----------------> recent replay
    +-- difficult PRE-step branch seeds -------------------------+
    +-- reached terminal raw state, captured before reset -------+--> bounded seed queue
    |                                                            |
    |                        isolated teacher-branch environment <-+
    |                            restore audited full snapshot
    |                            construct one aligned observation
    |                            frozen teacher acts; collect all valid labels
    |                            bounded recovery / bounded failed continuation
    |                                                            |
    +---------------- unique fresh labelled rows -----------------+--> protected replay
                                                                     + recent replay
                                                                     |
                                                   unchanged VAE trainer / loss
```

Each collection worker owns its main environment and, when enabled, one small
isolated branch environment with bounded slots. It never owns optimizer/replay.
The parent validates all replies and is the sole replay/normalizer owner.
Single-process mode uses the same construction and collection recipe.

Branch sampling must not modify the main environment, its observations, phase,
RNG, termination rules, or next executed action. No hidden teacher assistance is
introduced into ordinary student trajectories.

### 3.1 Distillation-owned opt-in policy

Add a validated plain-data `CoveragePolicy` (suggested new `coverage_policy.py`),
passed through `CohortSetup` and CLI. Default `off` preserves current source,
checkpoint layout and random draws. All new settings validate before environment
construction. Unsupported non-cohort paths fail explicitly.

Persist the complete resolved policy, not just a profile name:

- Ordered physical joint groups and per-axis position/velocity ranges.
- Perturbation timing, frame conventions, tier mixture and RNG scheme.
- Trigger thresholds, crossing/cooldown behavior and branch budgets.
- Restore layout version, supported actuator/observation configurations.
- Terminal reference advancement, soft-failure/hard-stop and endpoint semantics.
- Archive capacity, admission strata, replacement rule and draw fraction.
- Collection source/row identity layout and held-out split rules.

Suggested interface: `--coverage-policy off|difficult-state-coverage` plus an explicit
coverage config file. Final Tyro names are to be checked during implementation.
Do not overload `--reset-policy`, which already describes standing/reference
initialization. Do not automatically enable the new path from a saved PPO config.

## 4. Perturbation design

### Physical variables, not independent observation noise

- Root: roll/pitch/yaw, translation and world linear/angular velocity.
- Legs: named hip, knee and ankle joints, position at reset and joint velocity.
- Waist: named joints that affect pelvis-to-torso/anchor kinematics.
- Keep arms at the existing perturbation scale initially; no assumption that arms
  are irrelevant to balance. Resolve names against the audited 31-action order;
  reject missing/duplicated names rather than hard-code undocumented indices.
- Pelvis and torso anchor poses are derived through FK. Never independently
  perturb gravity, pelvis pose and anchor orientation after packing observations.
- Reference q/dq and authored reference bodies remain unchanged. Only measured
  physical state is perturbed; reference-relative transforms follow existing rules.

At full resets, combine the existing standing/reference initialization with one
resolved perturbation draw. New ranges **replace**, rather than accidentally add
to, the baseline noise for selected dimensions. Preserve/report existing reset
joint clipping; report requested versus realized samples and clipping counts.
For runtime perturbations, prefer root/joint velocity impulses at a specified
control boundary. Apply them before the one observation snapshot used to choose
the next action, with a tested forward/sense/history order; an event applied after
that snapshot must instead be attributed to the following transition. Never inject
a disturbance and then label a stale cached observation as the disturbed state.
Do not teleport joint positions while claiming natural dynamics.
Retain actual previous action and delay buffers across runtime disturbances.

Use supported entity state writers and verified frame conversions. The root-state
API uses world-frame velocities; direct MuJoCo free-joint qvel uses a different
angular-velocity convention. Validate non-identity orientation round trips.

### Proposed starting profile, not a measured optimum

| Setting | Starting proposal |
| --- | --- |
| Eligible perturbation tiers | 60% baseline, 30% moderate, 10% strong |
| Moderate / strong envelope | 2x / 4x resolved baseline root and selected leg/waist ranges |
| In-motion opportunities | Clip-local phase windows, with bounded per-segment event count and randomized event timing |
| Pose changes | Reset only in the first version |
| Runtime disturbances | Bounded root/selected-joint velocity impulses, no changes to actuator gains or action scale |

The tier fraction applies to perturbation opportunities, **not collected rows**.
Exact SI bounds, interval/event cap and trigger thresholds must be frozen in the
first policy config after state/geometry probes; do not infer them from late-fall
joint errors or claim 2x/4x is sufficient hardware coverage. A larger rare tilt
band may be evaluated later without changing the default profile silently.

Do not reject a state solely because the teacher will fail. Reject nonfinite or
malformed physical states. Record collision clearances/penetration severity,
joint-limit handling and reset impulses; do not silently lift the root or erase
velocities to turn an invalid seed into a nominally valid one. Establish numerical
hard-stop limits separately from difficulty thresholds.

## 5. Capture, restore and observation alignment — first critical gate

### 5.1 Two different objects

- `LabeledReplayBatch`: coherent raw policy inputs plus teacher action, sufficient
  for supervised replay but insufficient to run physics from that point.
- `BranchSnapshot`: restorable dynamics/reference/history state and provenance.
  Stored in a bounded seed pool, never passed to the neural decoder as extra input.

Suggested new `branch_state.py` defines an explicit versioned layout and audited
capture/restore handlers. Do not implement an arbitrary `__dict__` tensor copier.

Audit and support, for the actual frozen cohort configuration:

- Simulation qpos/qvel, activation/control, time and solver warm-start state.
- Mocap, external forces or other active dynamic fields.
- Per-world randomized model/actuator parameters, variant mapping and asset hashes.
- Processed/normalized action and previous-action state; actuator command delay,
  lag schedules, shared fused-buffer aliases and any active actuator history.
- Observation noise/bias, delay/history buffers, counters, cached actor view and
  sensor state needed by the schema.
- Command frame, timer, reference alignment, motion/teacher mapping, generation/
  segment/init metadata, episode counters and pending reference-boundary state.
- Future event/random streams, with explicit branch-local continuation semantics.

Branch-global RNG must not rewind the main process. Prefer independent derived
streams and record that future disturbances are independently sampled while the
restored instantaneous state/history match. If matched future disturbances are
required for a paired experiment, store their schedule explicitly. No bitwise
cross-device or cross-world physics guarantee is implied.

Unsupported active stateful components fail before collection with actionable
errors; never quietly clear history or revert randomized model parameters.

### 5.2 Difficult PRE-step seeds

Capture from the same instant used by the ordinary collector. Restore into a
branch slot and require equality of reference/conditioning, first teacher action,
previous action, phase, teacher routing and active delay state before stepping.
This is the simplest falsifiable restore test. Keep the student trajectory's
first seed label only once when it is shared with the branch; archive admission
is not another unique fresh sample.

### 5.3 Reached terminal state

Capture raw state before `_reset_idx` in an opt-in distillation recorder/hook.
The current recorder's cached actor observation must **not** be re-labelled as
the terminal observation. The hook copies raw dynamics, executed action and
pre-reset reference/history metadata; it does not forward/sense or advance noisy
observation history in the main environment.

Materialize the reached state in the isolated environment, then apply a defined
next-control-boundary command/forward/sense/observation sequence **once**:

- Advance reference normally once when there is a next frame.
- Stop at clip completion rather than teleport through a wrap into a fabricated
  recovery trajectory. Preserve timer/timeout boundaries as censored/interrupted.
- Keep the just-executed student action as the terminal observation's previous
  action. Append one reached-state observation to restored history/delay buffers.
- Recompute FK, anchor-relative transform, gravity and sensor values consistently.
- Mark this as a reconstructed reached-terminal observation, not an observation
  that the original auto-reset controller actually consumed.

Source terminations may use one-substep-stale derived geometry; record the original
termination cause separately from recomputed branch-state metrics.

Prefer the existing private recorder seam. If it cannot copy all required state
at a well-defined instant, add the smallest opt-in observer seam with off-path
regressions; do not duplicate the entire environment step or add an unconditional
second forward/observation pass. Hooks must also tolerate initial/explicit resets.

## 6. Teacher-controlled branches and failure retention

Trigger candidates from tilt/gravity direction, angular speed, anchor orientation
error, leg/waist q/dq error, and disagreement diagnostics. Use unions of interpretable
bands; q RMS alone is not a balance detector. Compute q errors with audited defaults
and use actual clip lengths for phase bins. Reference/standing transition mismatch
is an expected category, not automatically failure.

Trigger on threshold crossing, a scheduled perturbation, or source termination.
Bound candidates per motion/phase, cooldown and total queue capacity. Capture near
failure before collapse, not only after termination. Use diverse seed families and
bounded reservoir admission rather than thousands of adjacent frames from one fall.

Suggested initial budgets, to remain explicit configurable proposals:

- Teacher branch horizon: 100 control steps (2 s at 50 Hz).
- After first finite physical/tracking failure: retain at most 10 additional steps
  (0.2 s), still inside the 100-step bound.
- Teacher branch dynamics budget per collection call: at most 10% of main executed
  control transitions; expose unused budget and truncation counters.
- Bounded branch slots and seed queue sized from a measured memory budget; reject
  overload with recorded sampling/drop decisions, no untracked pending work.

The isolated branch environment uses a **collection-only** failure lifecycle:
normal tracking failure is recorded, not a reason to discard labels or immediately
auto-reset. Numerical invalidity, unsafe solver state, finite-floor-dwell cap,
clip completion, timeout and horizon remain explicit stop conditions. Merely using
`auto_reset=False` is insufficient: current manual-reset pending logic blocks the
next step. Implement a scoped lifecycle that separates diagnostic soft failures
from hard stops; do not clear private pending flags ad hoc or change the main task.

At each branch control instant, preserve aligned raw inputs and frozen teacher
labels; execute that teacher action. The first history comes from the student,
subsequent history from actual teacher execution. Retain valid finite terminal
labels even when no further step is possible. Numerical-invalid labels are rejected
with reasons; previously valid rows remain valid. Teacher/worker infrastructure
errors fail the collection transaction rather than masquerade as physical failure.

Record first failure and eventual outcome separately: recovered, failed, completed,
interrupted, horizon-censored or invalid. Recovery thresholds/dwell are diagnostic
config values, never loss weights or teacher qualification gates. No teacher
retraining, teacher ranking or alternative-target averaging.

## 7. Protected replay and equal-weight optimization

Suggested new `coverage_storage.py`: a protocol-compatible composite of recent
per-motion FIFO and a protected difficult-state archive. Preserve `BalancedReplayBuffer`
and its disabled behavior rather than rewriting all legacy rings.

Starting proposal:

- Preserve the baseline **total** replay capacity; reserve 20% for the archive and
  80% for recent rows, within each motion's existing capacity quota.
- Draw 80% recent / 20% archive per motion with the existing motion sampling weights.
- Storage fraction and draw fraction are separate recorded settings, not one flag.
- Archive admission covers student difficult states, teacher recovery states and
  reached-terminal/short post-failure states. Use bounded per-source/seed-family
  shares so terminal/floor states cannot dominate.
- Stratified reservoir replacement across motion, actual reference phase, deviation
  and history/source categories; preserve diversity across collection iterations.
  No success/failure outcome preference and no global re-enabling of `phase_bins`.
- Avoid an enormous Cartesian allocation with empty cells. Define finite strata,
  capacity rounding, sparse-cell behavior and deterministic replacement explicitly.
- Draws inside nonempty archive strata are bounded and reproducible. Until archive
  readiness, backfill from recent rows and report actual fractions. Never call an
  empty severe band covered, and never silently oversample one tiny seed family.

Carry optional enabled-path metadata through owned batches: source kind, unique row
ID, seed/family/branch IDs, trigger and perturbation tier, severity metrics/bins,
branch age, boundary/first-failure indicators and diagnostic outcome linkage.
Keep outcome tables separate if not yet resolved at insertion; no mutable borrowed
batch tensors. Persist raw tuples/labels, not encoder latents.

A row copied into both recent and archive is **one fresh observation**. Branch seed
labels shared with main collection are likewise not counted twice. Normalizers
update once from the union of unique new rows; archive draws never update them.
Student workers/evaluators keep frozen normalizers; each student checkpoint uses
its own saved normalizers during evaluation. Teacher normalizers remain frozen.

The existing summed-per-action reconstruction reduction and KL remain untouched.
Changing draw fractions changes exposure, not per-sample weights. Report that
explicitly rather than claiming the sampled distribution is unchanged.

## 8. Worker, checkpoint, CLI and compatibility integration

Add the enabled policy to `CohortSetup`; construct branch environments only when
needed. Worker replies include variable-size branch rows plus main tick-major rows.
Extend merge validation deliberately: existing assumptions of `ticks * num_envs`
cannot describe branch output. Maintain legacy main insertion order, define branch
insertion order, and namespace main/branch/family IDs across workers and resume.
Validate all replies before any parent replay/normalizer mutation.

Drain a bounded branch work unit before saving; checkpoints occur at a quiescent
collection barrier. There are no in-flight physics branches serialized in the
first version. A seed queue may be explicitly cleared at that barrier with counts,
or persisted only once its full restore layout is supported; this choice must be
fixed in the contract, not silently changed on resume.

Enabled runs use a new explicitly dispatched cohort checkpoint format (proposed
v4) and enabled replay layout. Persist policy, archives/counters, row/family
namespace, RNG states, stream seeds, normalizer fresh-ID state, diagnostic outcomes
and existing model/optimizer/trainer fields. Strict resume rejects changed ranges,
thresholds, branch lifecycle, archive policy or teacher identity transactionally.
Do not change the digest algorithm for existing `CohortIdentity` records.

| Case | Required behavior |
| --- | --- |
| Existing v1/v2/v3 and coverage off | Existing loading, execution, RNG, export and model inference behavior unchanged |
| Old checkpoint + new coverage strict resume | Reject with clear new-run guidance |
| Enabled checkpoint + identical policy resume | Restore durable state; fresh main/branch simulator reset with new namespace; no claim of live physics continuation |
| Enabled checkpoint + changed teacher/policy/layout | Reject before partial model/replay/RNG restoration |
| Enabled model-only inference/export | Reuse unchanged neural schema/control contract and test all version dispatchers; no collection environment needed |

Start the first controlled quality ablation from matched initial student weights,
not an implicit strict resume of the 20k run. A model-only warm-start migration,
including normalizer treatment, would be a separate explicit feature/decision.

Expose configuration, collection-only inspection and fixed-state evaluation via
`src/mjlab/scripts/distill.py` / experiment helpers. Do not mutate frozen teacher
manifests to store the new collector config. Update usage and Upcoming changelog
when implementation exists, not as if it were already delivered.

## 9. Evaluation and acceptance

### A. Correctness gates (no policy-quality claim)

1. Off-path golden regressions: reference/standing collection, action/observation
   cadence, RNG, original v1/v2/v3 inference/resume and registered PPO tasks.
2. Perturbations: named-joint routing, per-row subset isolation, quaternion/FK/gravity
   consistency, non-identity velocity frames, realized ranges/clipping and no double
   reset perturbation/history advancement.
3. Restore: first policy snapshot/teacher action parity; active delay/history/model
   randomization parity; unsupported components rejected. Perturb one copied field
   deliberately and require parity failure. Check unaffected main rows and queues.
4. Terminal capture: unique PRE-step and reached-state labels; original executed
   previous action preserved; no stale `obs_buf` reuse; correct reference increment;
   clip-end/timeout/reset collisions; initial reset emits no fabricated terminal row.
5. Branch lifecycle: both teacher recovery and teacher failure collect valid labels;
   bounded post-failure steps, floor dwell, numerical hard stop and exact horizon;
   ordinary evaluation retains normal failure termination. Manual-reset pending
   behavior handled through a tested scoped interface, not hidden flag mutation.
6. Archive: quota/rounding, severe-state survival under nominal flooding, deterministic
   reservoir, sparse strata/fallback, seed-family caps, no outcome weighting, row
   ownership/dedup and exactly-once normalizer updates.
7. Sharding/resume: unequal branch counts, empty branches, interrupted worker reply,
   namespace collisions, malformed late partition, next-draw RNG parity, rollback,
   atomic save and all checkpoint inference/export dispatchers.

Suggested focused new tests: `test_tracking_distillation_coverage_policy.py`,
`*_branch_state.py`, `*_terminal_coverage.py`, `*_teacher_branches.py`,
`*_coverage_replay.py`, plus affected existing collector/adapter/worker/storage/
cohort-checkpoint/CLI/auto-reset/standing-start suites.

Use `uv run`; run focused tests, scoped Ruff/Ty, `git diff --check`, and `make check`
where possible. Report existing unrelated failures separately; do not weaken tests
or call an exit-0 process sufficient semantic validation. Inspection 30 s,
individual tests 60 s, focused suites 120 s; unexpected timeouts require diagnosis
before retry. Longer physics/build work uses monitored execution and exact-owned
cleanup. Preserve the `AGENTS.md -> CLAUDE.md` symlink.

### B. Separately authorized headless integration smoke

Only after capture/restore CPU tests pass, run one small headless process at a time,
initially `tennis_001`, then an unequal-length two-motion case. Proposed bounds:
main <=4 rows, branch <=2 slots, <=128 main control ticks, <=16 steps per smoke
branch, replay <=1024, <=3 updates plus one save/resume check. Force deterministic
trigger/failure cases rather than rely on random falls. Test active delays and
model randomization. First establish backend viability; no automatic GPU fallback.

Verify teacher-action parity, finite retained terminal rows, bounded failures,
archive retention/draw counts, normalizer uniqueness and cleanup. This is execution
validation, not a useful trained policy. Resource choice and a justified monitored
runtime budget require subsequent authorization.

### C. Fixed held-out difficult-state fidelity

Generate a simulator-state benchmark with independent seeds, motion phases,
perturbation directions and seed families. Split **whole seed/trajectory families**,
including all teacher branches, before training admission; related rows cannot
appear in train and validation. Prefer a frozen benchmark collected with an
explicit frozen collector policy, so model comparison uses identical states.
No optimizer or normalizer update during evaluation.

Use the unchanged frozen teacher to label held-out tuples. Report raw normalized
RMS/p95 and target-position RMS/p95 using audited policy defaults/scales, plus
per-joint/per-motion and signed attitude/gyro/leg/anchor/history strata. Distinguish
per-component absolute p95 from p95 of per-row RMS. Report state distributions,
unique families and confidence uncertainty rather than just row count.

Keep the 1103-row hardware fixture separate and held out; it is a repeated
engineering diagnostic, not an untouched final qualification set. Do not train
on it or select perturbation/loss settings by repeatedly minimizing its error.

### D. Later controlled training ablation (not launched by this plan)

| Arm | Change from matched baseline |
| --- | --- |
| A | Existing collection + recent FIFO |
| B | Targeted coherent perturbations + reached-terminal labels, recent FIFO |
| C | B + bounded teacher branches, recent FIFO |
| D | C + protected difficult-state archive |

Match frozen teachers, architecture/loss, initial weights, optimizer draws/updates,
minibatches, seeds and total replay capacity. Use at least two paired seeds for an
initial quality comparison. Teacher branches incur extra physics; record total
main/branch transitions, unique labels, wall time, memory and per-motion gradient
exposure. Add a dynamics-budget-matched comparison if making an efficiency claim;
equal iterations alone is not equal data/compute.

Predeclare per-stratum fidelity improvement and ordinary-state regression margins
before training; exact margins follow benchmark variability, not arbitrary numbers
in this plan. Check difficult teacher-failure states as well as recovery states.
Closed-loop student testing under normal terminations is a separate quality gate;
teacher recovery does not become an acceptance prerequisite. Vendor sim2sim and
hardware stay separately authorized and are not implied by offline fidelity.

## 10. Ordered implementation phases

This is a document-level dependency plan, not an active task queue/workflow.

| Phase | Deliverable / likely source owners | Gate |
| --- | --- | --- |
| P0 | Exact supported-state inventory; resolved policy and lifecycle specification | Decide snapshot layout, timing, branch hard stops and budgets before coding physics restore |
| P1 | `coverage_policy.py`, enabled metadata, seeded perturbation helpers; `multi_motion.py` / `environment.py` integration | CPU validation, coherent FK/frame/partial-reset tests and disabled-path parity |
| P2 | `branch_state.py`, private terminal recorder seam, branch construction | Restore parity and reached-terminal alignment including active histories/delays |
| P3 | `teacher_branches.py`, triggers/queue/bounded lifecycle; adapter/collector integration | Teacher-failure retention and hard-budget tests; main trajectory unchanged |
| P4 | `coverage_storage.py`, replay metadata, admission/sampling/dedup reports | Difficult rows survive nominal flooding; exactly-once fresh normalization |
| P5 | Shared setup/worker merge/runner/checkpoint/CLI integration | Transactional multi-worker validation, new-format save/resume and old-format dispatch regressions |
| P6 | Frozen held-out evaluator, documentation/changelog and implementation acceptance record | Static/CPU gates; approved small integration smoke; quality run remains separate |

Single writer per checkout. No delegated review, workflow controller or jobs are
created by this request. Preserve unrelated work and re-read shared status before
editing it. If capture/restore cannot faithfully support an active component, stop
at P0/P2 and report the precise gap rather than downgrade to a position-only reset.

**Implementation done** means the enabled collector, replay, compatibility,
reporting and tests satisfy these gates, with any approved smoke documented.
**Research success** requires a later controlled fidelity/closed-loop comparison.
Neither is a claim that the teacher can recover from every state or that the
student is qualified for hardware.

## 11. Open engineering choices to close at P0

1. Concrete moderate/strong SI ranges, event timing and seed/trigger thresholds.
2. Full supported state inventory, copy cost, per-world model/variant transfer and
   shared delay-buffer alias handling for the frozen cohort.
3. Private observer/lifecycle seam sufficient for terminal reconstruction and
   bounded post-failure branch continuation without changing main task behavior.
4. Branch slot/queue memory caps, floor-dwell numerical bounds and quiescent queue
   handling at checkpoint barriers.
5. Archive strata, capacity rounding, family caps and readiness/fallback rules.
6. Benchmark split, fidelity/regression margins and separately approved smoke/
   production resource budgets.

These are engineering gates, not authority to improve teachers or reintroduce
success-based weighting. Initial implementation can use a singleton `tennis_001`
cohort for tests but must not bake its clip length, indices or scalar teacher code
into the multi-motion path.
