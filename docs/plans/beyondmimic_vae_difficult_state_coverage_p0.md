# P0 — X2 VAE difficult-state coverage contracts

Date: 2026-10-01. Status: **full P0 branch contract remains unvalidated; an MVP
one-step capture/label/replay path is now implemented and bounded simulator-tested.**
The MVP has passing X2 policy-input restore-parity evidence for the selected
``inc000``–``inc007`` cohort. It does not validate branch continuation, runtime
coverage perturbations, protected replay, closed-loop quality or hardware.

- Parent plan: [implementation phases](beyondmimic_vae_difficult_state_coverage_implementation.md).
- Declarative proposal: [P0 contract YAML](beyondmimic_vae_difficult_state_coverage_p0_contract.yaml).
  This lives under `docs/plans/`, deliberately not runnable training configs.
- Observed HEAD: `8dcd46dde7ca5dddd246245d4c3b15b19fdafc92`.
- Support target: frozen v3 cohort, initially singleton `tennis_001`, fixed-topology
  X2, builtin position actuators and existing correlated delays/randomization.

The YAML closes ambiguous engineering defaults for P1. It does not certify that
snapshot restoration, larger reset envelopes or a particular resource layout
works. User confirmation freezes these settings for the first implementation;
subsequent tuning is a versioned experiment, not an unreported automatic response.

## 1. Main recommendations

1. **Directional excitation, not independent large noise everywhere.** Keep 60%
   baseline opportunities; moderate/strong resets enhance one root direction,
   one leg/waist chain, or a coupled pair. Leave vertical reset noise small.
2. **Use reference-relative triggers.** Large authored tennis tilt is not a fall.
   Capture pelvis/anchor gravity-direction mismatch and angular/joint errors,
   not absolute tilt or q RMS alone.
3. **Restore existing delays and randomized dynamics.** A nominal-model or
   zero-history branch is a different state and invalidates the desired labels.
4. **No additional pushes in short teacher branches.** Preserve the initial
   disturbance and applied forces, but introduce no new scheduled perturbations.
   This isolates teacher responses and simplifies branch RNG/state semantics.
5. **Keep all valid admitted labels, regardless of outcome.** Bound the amount of
   post-failure physics and archive exposure rather than filter teacher failures.
6. **Bound work and diversity.** Use an explicit actual-physics budget and family
   limits; do not let one fall contribute thousands of near-duplicate updates.

These refine the parent's tentative uniform 2x/4x scale rule; no previous production
configuration is changed. Equal-weight reconstruction+KL remains fixed.

## 2. Source/config audit and artifact identity

All 15 v3 members' saved environment configs were read as YAML **data only** with
`BaseLoader` (Python tags never executed). They share 5-ms physics, decimation 4,
50-Hz policy, torso anchor and the same named event/delay families. The current
registered base-task factory was also instantiated as a configuration object;
**no `ManagerBasedRlEnv` or MuJoCo simulation was created**.

The manifest SHA is
`4b73aca9450d8186bdf2c12ab375899ec4d1420e7efe9e705aa160fa260b057a`.
Initial teacher/motion identity, resolved relative to repository root:

| Role | SHA-256 |
| --- | --- |
| `tennis_001/model_29999.pt` | `6fd5d68a67630c5ab726bd12a718734a4135e4620932cc9a4eca43c452a61ae2` |
| `tennis_001/params/env.yaml` | `1c22923f92856e5ef6018d716ab1dd1eb6ae94cc028b556d05b4697dc72a2dd7` |
| `tennis_001/params/agent.yaml` | `1ece1e17b7cdb1160ed37f3749c38fe07007c8e2ad46144fb050c51e67e61331` |
| `tennis_001/2026-09-23_19-28-08.onnx` | `552ded7bbc08a110e1a90c70a532b4aad267aa2b47c3632f0036b3dd4381782a` |
| `data/tennis/single_001_zhanghongyu_agibot_x2_tracking.npz` | `9860d19cf3cd9cd9bac5dfe0614ea1568ff9084b5ddb4b34de4d94473605ef7f` |

The teacher paths above are under
`logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/`.
This motion is the frozen teacher's source NPZ, not the differently packaged
motion file in the previous hardware replay bundle; do not conflate file hashes.
The ordinary cohort resolver remains authoritative for all selected members.

Verified baseline:

- Root body `pelvis`, policy anchor `torso_link`; 31 named hinge joints plus free
  root in the source XML. Neural layout stays reference 68 / conditioning 99 /
  latent 32 / actions 31. Compiled dimensions must be checked in P2.
- Root reset x/y +/-0.025 m, z +/-0.005 m, roll/pitch +/-0.05 rad, yaw +/-0.10 rad.
  Root reset velocities x/y +/-0.25 m/s, z +/-0.10 m/s, roll/pitch +/-0.26 rad/s,
  yaw +/-0.39 rad/s; all-joint position +/-0.05 rad.
- Actor gyro and joint q/dq delays: 0–1 **control step**, 0.9 hold probability,
  update period 1; q/dq share `encoder_packet`. Gyro noise +/-0.2 rad/s, joint
  q noise +/-0.01 rad, dq noise +/-0.5 rad/s and startup encoder bias +/-0.01 rad.
- Actuator command delays: 0–2 **physics steps**, hold probability 0.9, update
  period 1; builtin groups may fuse and share the same delay-buffer object.
- Startup randomized inertia/COM, friction, armature, effort, damping and foot
  size; reset PD gain scaling. These existing settings are not broadened here.
- Main push interval 4–8 s remains unchanged; episode timeout 10 s.
- Main physical terminations remain anchor-height error >0.25 m, end-effector
  height error >0.25 m and `bad_anchor_ori` >0.8. The last threshold is a
  difference in projected-gravity **z components**, not 0.8 rad of tilt.
- Installed `mujoco-warp` metadata reports 3.11.0. Its data layout has `history`,
  `eq_active`, `userdata` and optional sleep state, in addition to qpos/qvel.

## 3. Perturbation contract

### 3.1 Reset ranges

Values below are symmetric maximum magnitudes. Moderate/strong envelopes apply
only to the selected enhanced axes/chain; unselected dimensions keep baseline.
All draws are uniform within the stated intervals.

| Root variable | Baseline | Moderate | Strong |
| --- | ---: | ---: | ---: |
| x or y position (m) | 0.025 | 0.050 | 0.075 |
| z position (m) | 0.005 | 0.005 | 0.005 |
| roll or pitch (rad) | 0.050 | 0.150 | 0.300 |
| yaw (rad) | 0.100 | 0.100 | 0.200 |
| x or y velocity (m/s) | 0.250 | 0.500 | 1.000 |
| z velocity (m/s) | 0.100 | 0.100 | 0.100 |
| roll or pitch angular velocity (rad/s) | 0.260 | 0.750 | 1.500 |
| yaw angular velocity (rad/s) | 0.390 | 0.390 | 0.780 |

0.15/0.30 rad corresponds to roughly 8.6/17.2 degrees of single-axis attitude
noise. This is a proposed coverage envelope, not a recovery guarantee.

| Selected joint group | Baseline q (rad) | Moderate q | Strong q |
| --- | ---: | ---: | ---: |
| Hip pitch / knee | 0.050 | 0.100 | 0.200 |
| Ankle pitch | 0.050 | 0.080 | 0.150 |
| Hip roll / ankle roll | 0.050 | 0.080 | 0.150 |
| Waist pitch / roll | 0.050 | 0.080 | 0.150 |
| Waist yaw | 0.050 | 0.050 | 0.100 |

Joint velocity deltas: baseline zero; moderate +/-0.5 rad/s; strong +/-1.0 rad/s,
except knees +/-1.5 rad/s. Baseline reference velocities are not erased.

Reset tier mix: 60/30/10 baseline/moderate/strong. Baseline delegates to the
existing reset path. For enhanced tiers, choose root-only/chain-only/coupled with
probabilities 40/40/20. Root directions sagittal/lateral/yaw/horizontal-translation
have probabilities 40/40/10/10. A rotation direction enhances that orientation
axis and matching angular velocity; translation enhances one uniformly chosen
x/y axis and corresponding linear velocity. Coupled chooses root direction and
joint chain independently, not an undocumented fixed recovery synergy.

Joint chains: left/right hip-pitch–knee–ankle-pitch (25% each), left/right
hip-roll–ankle-roll (15% each), waist yaw/pitch/roll (20%). Draw per joint inside
one selected chain; retain clipping counters. Hip yaw, arms and head keep their
baseline noise. All names must resolve against the frozen joint-order contract.

World-frame translation/velocities; orientation is a world XYZ delta quaternion
left-multiplied onto initial root orientation, matching existing reset convention.
Root/pelvis/anchor FK and policy transforms are recomputed, not independently
noised. Never enlarge reset z to compensate for collisions or silently lift seeds.

### 3.2 Runtime timing

At each full segment start, with probability 0.5 schedule **at most one** runtime
opportunity. Uniformly select a nonempty eligible one of five phase bins, then an
integer event frame within it. Eligibility: at least 10 executed segment steps and
at least five reference steps remain. Frames behind the reset phase are ineligible;
short/late segments with no eligible frames have no event, with counts reported.
A reference teleport starts a new segment schedule; it is not counted as a new
standing-start reset. Sampling is once per segment, not Bernoulli every tick.

Draw the same tier and root-only/chain-only/coupled mix. Runtime root directions
are sagittal/lateral/yaw, 45/45/10: additive x/pitch, y/roll or yaw velocity deltas
only. Use the tier's reset velocity maxima; inactive root delta axes and vertical
delta are zero. Joint velocity deltas use the selected-chain envelope. A baseline
chain-only event has zero added dq and is explicitly reported as a no-op.
No root/joint position teleport and no manufactured action-history reset.

Place the new event after inherited interval events, before raw pre-reset capture
and the normal single forward/command/sense/observation pass. It affects the
**next** policy observation/action, not the already-selected action. Match event
frame attribution to that next command frame and never recompute observations in
the collector. The scheduled event frame denotes prospective `f+1`; placement
must test that frame, not fire a step late from the cached `f`. For a terminating
row, the perturbation is part of its reconstructed reached state. Existing main
pushes keep their settings; branches add none.

## 4. Trigger definitions

Use current unbiased physics at the same cached policy instant; do not use noisy
or delayed q/dq for physical-severity gating. Policy inputs/labels still use the
original cached observation. Persist both metrics and the observation tuple.

- Root/anchor gravity mismatch: angle between world gravity expressed in the
  measured body frame and in that same reference body frame; unit vectors,
  clamped dot product. Includes signed x/y projections in reports.
- Anchor full rotation mismatch: existing quaternion error magnitude, including
  yaw; do not call it pelvis tilt.
- Root angular-velocity error: norm of measured minus corresponding reference
  root angular velocity, both in world frame.
- Joint q/dq errors: RMS on the twelve leg joints plus three waist joints;
  unbiased absolute measured joints versus reference q/dq. Independent from
  action offsets and excludes head/arms from this trigger RMS.

Thresholds: root gravity 10 degrees; anchor gravity 10 degrees; full anchor
rotation 15 degrees; root angular-velocity error 1 rad/s; leg/waist q RMS 0.20 rad;
dq RMS 1 rad/s. Define `difficulty_score = max(metric / corresponding_threshold)`.
Nominal <1, moderate [1,2), strong >=2. This is a sampling metric, not a failure
classifier. Absolute authored tilt and teacher/student action disagreement are
reported only; no trigger or teacher preference based on them in v1.

Emit on first high state after reset, upward crossing, scheduled event or original
physical termination. Rearm after all trigger metrics remain <75% threshold for
five steps. Threshold cooldown 25 steps, at most two threshold candidates per
segment. If a crossing occurs during cooldown, retain one pending flag and emit
when cooldown expires only if still high. Event/terminal candidates bypass this
threshold cooldown, but not seed/byte/physics budgets. Timeouts alone do not
request a recovery branch. Standing-to-reference mismatch has explicit provenance
and may trigger once; it is not automatically physical failure.

## 5. Restorable state inventory and support boundaries

### 5.1 Required owned state

| Owner | Required fields / operation | Evidence |
| --- | --- | --- |
| Warp dynamics | `time`, `qpos`, `qvel`, `act`, `history`, `qacc_warmstart`, `ctrl`, `qfrc_applied`, `xfrc_applied`, `eq_active`, mocap pose/quaternion, `userdata`; include empty active fields explicitly | Installed Warp data declarations |
| Sensor-stage snapshot | Cached `sensordata`, qacc/qacc_smooth/act_dot where used by sensor phase; original teacher actor view and packed student inputs | `Simulation.sense`, adapter cached-view semantics; P2 must validate actual stage dependencies |
| Randomized model | Copy the 22-field expanded union below and required dependent constants; no redraw of source PD gains/friction/inertia/foot size | Current event `model_fields` declarations |
| Entity | Encoder bias, default pose/limits, all active actuator target tensors, correct global/local joint/control mappings | `entity/data.py` |
| Action manager / terms | Current/previous/previous-previous normalized action, raw and processed term targets | Action manager and joint-position term |
| Actuator delay | Each unique fused buffer's chronological history and push counts, current lag, schedule counter, phase offset; preserve aliases | `actuator/builtin_group.py` |
| Observation manager | Term delay histories, shared encoder-packet schedule, history windows if active, cache and any active noise bias | `managers/observation_manager.py` |
| Command / adapter | Motion/teacher ID and frame, reference alignment, timers/counters, segment/generation, initialized/reset frame flags, boundary events, standing provenance and age | `multi_motion.py`, `adapter.py`, command manager |
| Events / counters | Episode age, control/physics counters, copied timer state and new event schedule; branch-local failure bookkeeping | Environment/event manager |
| Stateful sensors | Self-collision force history; any cache invalidation and state required for branch diagnostic/reward paths | `sensor/contact_sensor.py` |
| RNG | Python/NumPy/CPU/device RNG isolation, owned policy/admission/branch RNG, no main stream advancement from branch construction or stepping | Current noise/delay routines default to global RNG |

Expanded model union:

```text
actuator_acc0 actuator_biasprm actuator_forcerange actuator_gainprm
body_inertia body_invweight0 body_ipos body_iquat body_mass body_subtreemass
dof_armature dof_damping dof_frictionloss dof_invweight0
geom_aabb geom_friction geom_rbound geom_size jnt_actfrcrange
tendon_actfrcrange tendon_invweight0 tendon_length0
```

Restore all copied source model values before recomputing derived constants.
Dependent constant parity is checked after recomputation. Shared defaults must
match; write into existing arrays in place, never replace arrays referenced by
CUDA graphs. Fail if an active expansion is outside the supported layout.

Source and destination scene origins may differ. Translate root/mocap and
world-position reference alignment by `destination_origin - source_origin`;
orientations, velocities and applied force/torque are unchanged. Require identical
flat terrain/scene topology; do not claim transfer to different obstacles.

The circular pointer is **global across rows**. Do not assign a source global
pointer into a branch batch: serialize history in chronological order and repack
the selected row at the destination pointer, preserving per-row valid push count
and lag scheduling. The fused delay aliases must still point to one buffer.
Partial branch-slot replacement must not change other slots' histories.

### 5.2 Initial unsupported configurations

Mesh variants, learned/recurrent/custom actuators, active sleeping, arbitrary
stateful noise/event components or additional scene dynamics require explicit
handlers and tests. Reject them initially rather than restore nominal defaults.
The current source-config target uses a plain `EntityCfg` with builtin position
actuators. Runtime supported-state preflight must still verify this after compile;
P0 config inspection is not that proof.

Do not copy contact/constraint packed workspace rows by assumed world index: their
allocation is not necessarily row-independent. Rebuild workspace through the
backend while preserving dynamics/warm starts; P2 validates the resulting next
transition. If a backend needs additional persistence state, amend the explicit
layout and tests before claiming support.

### 5.3 Randomness

Main and branch construction/stepping cannot share global draw side effects.
Preferred first same-process approach: a scoped complete RNG save/restore and
branch-owned stream state, serialized execution only. A separate worker process
is an alternative implementation, not an excuse to omit dynamic state. Never
reseed all devices on every branch or rewind the main stream. GPU initialization
itself is not reproducible mid-run; construct the bounded branch owner before
main rollout and establish the intended main RNG state afterward.

Pre-step seed: first teacher/student inputs are the **exact cached tensors**;
no new noise draw or history append. Future branch observations use independent
branch randomness, with original lag/history preserved. Teacher branches schedule
no new external perturbations or reset DR; source applied forces are still intact.
For paired next-transition tests, explicitly supply matched RNG/noise schedules.

## 6. Terminal and teacher-branch lifecycle

### Terminal reconstruction

The existing pre-reset recorder has intact executed action but PRE-step `obs_buf`.
Copy raw state there after events, before reset; ignore initial/explicit reset
hooks for terminal purposes. Copy selected rows only after bounded admission.
Do not call `forward`, command compute or noisy observation compute on the main
environment inside this hook.

In the branch environment, restore, forward, advance the source reference once
if `f+1 < F` and timer/timeout permits, sense and append **one** observation frame
to restored delay/history. Derive forward/sense quantities before constructing the
view; preserve copied solver warm-start state for the next actual physics step
if `forward()` modifies it. The just-executed student action supplies previous
action. Label the resulting tuple with the same frozen teacher. If no next
reference frame exists, retain the raw endpoint/cause; do not invent a next
reference label or replay through a wrap. Physical-termination-only rows may
continue; timeout/timer/endpoints are bounded/censored without teleports.

Original termination cause may have been computed from one-substep-stale geometry;
retain it separately from freshly recomputed terminal metrics. The reached-state
view is explicitly reconstructed, not an input the original auto-reset student
actually consumed.

### Branch controls and endpoints

Restore source phase, history, source dynamics and episode age. Execute only its
frozen routed teacher. Advance reference once per control transition, never wrap
or teleport. Preserve source remaining 10-s timeout. Original anchor/EE/anchor-ori
terminations become **soft diagnostics only in the branch collection environment**;
main/evaluation termination behavior is unchanged. Implement an explicit scoped
lifecycle, not ad-hoc clearing of `_manual_reset_pending`.

- Maximum 100 executed control transitions per branch (2 s).
- At most ten transitions after first original tracking failure (0.2 s), within
  that horizon. If the seed is already failed, counting starts at branch step 0.
- At each valid input, retain the finite label before checking continuation stop.
  The endpoint can add one valid label after the last executed transition.
- Stop earlier on floor dwell: root absolute tilt >=75 degrees **and** root
  height <=0.30 m relative to env origin for five consecutive steps. This is a
  finite-data collection cap, not contact detection, numerical invalidity or
  proof of irrecoverability. Existing pelvis collision is disabled in this model.
- Hard stop on raw nonfinite state/input/output, malformed quaternion
  (norm error >0.001), restore mismatch or solver/contact-capacity error.
  Do not sanitize invalid observations into apparently valid teacher labels.
- Prior valid labels are retained after physical/numerical row failure; shared
  backend corruption or inference/worker infrastructure failure aborts the
  transaction and is not reported as a successful physical-data collection.
- Budget exhaustion ends a branch as budget-censored, preserving valid collected
  rows. Completion, timeout, reference timer interruption, floor cap, failure
  continuation cap and horizon have separate reason codes.

Recovery diagnostic: score <0.75 and no original tracking failure for ten steps.
Continue to the bounded horizon after recovery. Report `first_failure` separately
from eventual recovery/completion; neither changes label weight or admission.

## 7. Work and memory budgets

Per worker, proposed starting caps:

- 32 branch slots; <=64 admitted branch starts per collection call.
- <=256 candidate snapshots and <=64 MiB total owned snapshot bytes, including
  copied model values, all histories and target tensors.
- <=1 GiB branch-owner resident bytes, **provisional startup measured cap**, not a
  claim a 32-world Warp environment fits. If exceeded, fail with an explicit
  proposed smaller layout; do not silently change configuration.
- Branch simulated row-steps <=floor(0.10 * actual main row-steps in this call).
  Count idle/padded worlds when a vector step advances them, not just labelled
  active branches. Cap labels separately through starts/horizon.
- Disable branches during teacher-only bootstrap, since there is no student
  recovery-distribution question there. Terminal capture stays bounded.
- No physics work credit across calls; no live pending branches at return/save.
  Unused/unstarted candidates are discarded with explicit budget/queue reason.

A 32-step, 8192-row main call has 262144 main row-steps and a 26214 branch work
ceiling, but the independent 64-start x 100-step bound may make actual work much
lower. Thus 10% is a ceiling, not a promised sample fraction or cost estimate.

Candidate reservoirs reserve 51/256 slots for terminal candidates and 205 for
pre-step/event seeds. Respect byte cap too. Distribute capacity across selected
motions with at least one slot each when possible, then deterministic proportional
rounding. Select by motion/phase/severity without teacher outcome preference.
Perform bounded candidate-ID admission before copying full snapshots. Overflow
means some rows are uncaptured; report counts instead of promising unlimited
terminal retention. Every **admitted valid** terminal label is retained regardless
of teacher outcome. P2 must measure copy cost before scaling the slot/queue caps.
The 64-start call budget reserves 13 terminal and 51 pre-step/event starts; unused reservations may
be borrowed after that quota round. Neither category gets priority from teacher
outcomes.

## 8. Protected replay contract

Preserve total replay capacity and existing motion weights. Reserve floor(20% of
motion quota) for archive; recent gets the remainder. Allocate archive category
quotas by deterministic largest remainder: student difficult 50%, teacher
continuation 40%, finite post-failure 10%, then five phase bins, then moderate/
strong evenly. Preserve unused capacity as reserved rather than silently letting
nominal traffic evict difficult rows. For very small smoke quotas, permit zero
cells and report uncovered categories.

Categories have precedence: currently failed/source terminal -> finite post-failure;
otherwise teacher branch -> continuation; otherwise score>=1 student -> difficult.
Teacher continuation that has recovered remains eligible; nominal-score branch
rows occupy the moderate cell but retain their true score. Source terminal labels
likewise retain actual severity/cause even if normalized score is below one.
No success-based admission or weighting; post-failure cap is a **state category**
cap rather than a gate excluding entire teacher trajectories that later fail.

Within each cell, protect diversity with a uniform reservoir over distinct root
trajectory families and up to 32 rows/family/cell; maximum 256 rows/family/motion
archive. Every student root segment and all its descendant branches share one
family for dedup/holdout purposes. Family/row namespace includes run, resume
namespace, worker and local segment identifiers. Evict complete family buckets
when the family reservoir replaces one, not random adjacent frames globally.
For a nonempty cell of capacity C, reserve max(1, floor(C/32)) family buckets,
each storing min(32,C) rows by reservoir; small unused remainders stay reserved and
reported. Refuse additional rows once a retained family reaches its 256-row
motion cap, with a counter. Tiny-cell rounding and family replacement are part
of P1/P4 pure tests.

Draw 80% recent / 20% archive per motion: archive count is floor(0.2 times that
motion's microbatch draw count), recent gets the remainder. Within archive use
largest-remainder 50/40/10 category target counts and nonempty phase/severity cells
evenly; draws without replacement per microbatch and <=32
rows/family/microbatch. Archive sampling starts only after eight families/motion.
Shortfall goes to recent and actual fractions are reported, not to aggressive
repetition of a tiny failure pool. Avoid duplicate row IDs across recent/archive
in a microbatch: exclude already-drawn IDs, then backfill from unique recent rows.
A remaining recent shortage prevents that update with a `ReplayNotReady` reason
rather than repeating samples. Existing `phase_bins` stays 0; these are new
archive-local strata, not the old phase-sampling feature.

Normalizers update once from **unique fresh** rows, including new branch rows.
Copying a row to archive is not fresh data; the branch's identical seed label is
not a second observation. Saved latents are forbidden as replay input. Outcome
tables are diagnostic side data and need no retroactive tensor mutation.

## 9. Acceptance and compatibility contract

- Proposed enabled cohort checkpoint v4, explicitly dispatched. Old v1/v2/v3
  coverage-off execution/RNG/inference/export stays unchanged.
- Strict old-checkpoint resume with coverage enabled or enabled-policy mismatch
  rejects transactionally. No implicit 20k warm start. Existing cohort digest
  algorithm is unchanged.
- Save only at a collection barrier with no active/pending branch physics.
  Snapshot queue is not persisted in v1; archives, counters, family/row namespace
  and RNG are. Simulator restarts on resume with explicit new namespace.
- All workers use the same policy/support inventory; parent validates all
  variable-size replies before replay or normalizer mutation.
- P1 can implement pure contracts and perturbation helpers; P2 must establish
  live capture/restore support before teacher branches run.

P2 minimum equality gates:

1. Copied dynamics/model/history/discrete state exact after row remapping.
2. PRE-step teacher and packed student input tensors exact (cached view, no noise
   redraw); deterministic frozen teacher action atol 2e-6 / rtol 1e-6.
3. Freshly recomputed geometry atol/rtol 1e-5, with a deliberately corrupted
   reference/frame/delay/model-field negative test.
4. Same-state **next-transition** parity under matched teacher action/noise,
   including active delays/model randomization and normal-contact cases. Record
   backend numerical tolerances from reproducibility probes before certifying
   support. First-input parity alone is not dynamics parity.
5. Main trajectory and global RNG unaffected by branch construction, copying,
   restore, step, teardown and partial branch-slot replacement.

Missing fields or unverified next-transition semantics are P2 blockers, not reasons
to fall back to a qpos/qvel-only reset. Physics/backend fields, copy time/resident
memory and geometry/reset impulses cannot be certified by this P0 source audit.

Fixed benchmark: root family SHA-256 split 80/10/10 train/validation/test before
admission, with all descendants inheriting the split. Require >=32 distinct
families and >=256 unique rows per reported motion/severity cell for an initial
coverage claim; smaller cells are sparse, not passing. Keep hardware fixture held
out as repeated engineering diagnostics. Fidelity/regression thresholds will be
predeclared after independent benchmark variability measurement, before training.

## 10. P0 disposition / next action

Completed here: source-only inventory, artifact binding, concrete proposed SI
ranges, trigger definitions, timing/lifecycle/stop rules, worker work/byte caps,
archive/sampling/dedup rules and checkpoint/validation contracts.

Not done: implementation of these interfaces, compiled-state inventory, copy-cost
measurement, snapshot/next-transition parity, physics smoke or quality training.
No resource authorization is implied.

**Recommended next action:** confirm the directional profile and the bounded
no-new-push teacher branches, then implement P1 pure validation/metadata and
perturbation helpers. Preserve P2's explicit support gate rather than presenting
this P0 document as capture/restore validation.
