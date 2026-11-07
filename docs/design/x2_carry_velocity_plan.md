# Carry-pose velocity policy: implementation plan

Date: 2026-11-07 (+08).
Status: **implementation plan; nothing here authorizes a training run.**

## Goal

Train an AgiBot X2 policy that stands and walks while holding the tennis carry pose, so the
robot needs no harness before the juggle policy engages.

On hardware the robot cannot hold the carry pose by itself, so it hangs in a harness. The
harness then restricts movement after the juggle policy engages. A velocity-tracking policy
that keeps the carry pose removes the harness from the engagement sequence.

## Non-goals

No ball, no racket strike, no tennis episode machinery, no controller change, no training
run, no hardware activity.

## 1. Task identity

| Item | Value |
|---|---|
| Task id | `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-CarryVelocity` |
| Factory | `agibot_x2_flat_velocity_carry_env_cfg(play=False)` in `src/mjlab/tasks/velocity/config/agibot_x2/env_cfgs.py` |
| Base | `agibot_x2_flat_velocity_env_cfg(play)` |
| Experiment name | `agibot_x2_velocity_carry_pose` |

Everything not listed below is inherited unchanged: observations, actions, terminations,
randomisation events, episode length (20 s), and every reward weight except the pose group.

Starting point for training is **from scratch**. No checkpoint is resumed, because changing
`default_joint_pos` changes the meaning of a zero action.

## 2. Entity: racket plus carry pose

- Robot: `get_x2_tennis_robot_cfg()`. It supplies the welded racket and
  `init_state.joint_pos = BAT_CARRY_POSE` at `BAT_CARRY_BASE_HEIGHT = 0.6698 m`.
- `spec_fn` must add the velocity foot sites on top of the tennis spec:
  `lambda: add_velocity_foot_sites(get_tennis_spec())`. `add_velocity_foot_sites` documents
  this composition itself.
- The racket carries no joints, so the joint count stays 31 and the actor width is
  unchanged. That keeps the existing controller family valid.
- `reset_base` height offset is reduced to `(0.0, 0.02)`. The carry stance has 2 mm of
  ground clearance; a 1-5 cm drop is a needless landing impulse with a racket in hand.

`default_joint_pos` becomes `BAT_CARRY_POSE`. With `use_default_offset=True` the action
target is `default + scale * action`, so zero action now means "hold the carry pose".
The deployed controller config must use the same default pose and root height.

PD gains are the **vendor reference-policy gains** (shoulder 50/3, elbow 50/3, wrist 20/2,
ankle 40/2, hip pitch 120/5), applied with `apply_vendor_rl_gains`, not the velocity task's
gains. The juggle policy this task hands over to runs the vendor set, and the robot can only
run one gain set at a time, so training under different gains would leave a discontinuity at
the handover. The config probe asserts equality with the tennis juggle task on all 29 vendor
joints.

## 3. Pose reward: keep `pose` over all joints, add a quadratic arm term

The shipped `pose` term is `exp(-mean((q - q*)^2 / std^2))` with a speed-dependent
per-joint tolerance. It saturates: at a 0.52 rad deviation with std 0.15 the exponent is
about -12 per joint, so the value and the gradient both vanish. A stance may use a
tolerance term; a held arm may not.

The fix is **additive**, not a split:

| Term | Joints | Form | Weight | Tolerance |
|---|---|---|---|---|
| `pose` | **all 31**, inherited unchanged | `variable_posture` | 1.0 | the base task's three tables, byte-identical |
| `pose_quadratic` | arms and waist, 17 joints | `mean((q - q*)^2 / std^2)` | -0.4 | std 0.10 rad |

- `pose` keeps its mean over 31 joints, so the same leg error costs what it costs in the
  base task. There is no hidden retune, and reward can be compared against the base run.
- `pose_quadratic` is a **subset** of `pose`. The arms and waist are scored twice, which is
  the point: `pose` supplies the inherited tolerance behaviour, and the quadratic term
  supplies a gradient at any distance so the racket never depends on a saturated term.
- The overlap is deliberate and asserted. An earlier draft restricted `pose` to legs and
  head; that divides its mean by 14 instead of 31 and makes an equal leg error cost
  **2.214 times** more (0.368 against 0.636 for a uniform 0.05 rad error), which would have
  silently retuned the baseline the run has to be compared against. Rejected.
- Starting weight -0.4 with std 0.10 rad: a 5.7 degree arm error on every joint costs -0.4
  per step, against a tracking maximum of 4.0. `pose_quadratic` is a mean over 17 joints, so
  one wrist 20 degrees off costs -0.287 while all seven racket-arm joints at 10 degrees cost
  -0.502. If the racket must be locked by the wrist, the reduction has to change.

## 4. Command range

| Quantity | Value | Reason |
|---|---|---|
| `lin_vel_x` | (-0.6, 0.6) | walking with a racket at chest height |
| `lin_vel_y` | (-0.3, 0.3) | lateral positioning |
| `ang_vel_z` | (-0.5, 0.5) | unchanged from the base task |
| `heading` | (-pi, pi) | inherited |
| `rel_standing_envs` | 0.25 | standing is a first-class skill here, not a 10 % accident |
| `curriculum` | `{}` | the base ramp reaches 3 m/s, which is not this task |

## 5. Probes (measurement, not tests)

Both are cheap: `num_envs <= 8`, `steps <= 100`.

1. `scripts/experiments/x2_carry_velocity/check_carry_velocity_cfg.py`
   - `default_joint_pos` equals `BAT_CARRY_POSE` for every joint.
   - The racket bodies exist in the compiled spec, and both foot sites exist.
   - Actor observation width equals the existing no-state-estimation velocity task's.
   - Action count is 31. A pattern-list length is not an action count: the 16 actuator
     groups expand to 31 joints.
   - `pose` covers all 31 joints and matches the base task's joint set and tolerance tables;
     `pose_quadratic` covers arms and waist and is a subset of `pose`.
   - Command ranges and the empty curriculum as in section 4.
   - `pose_quadratic` cost at 0, 5, 10 and 20 degrees of arm offset.
2. `scripts/experiments/x2_carry_velocity/probe_carry_stand.py`
   - Step 0, before any physics step, is the acceptance measurement:
     `default_joint_pos` equals `BAT_CARRY_POSE`; the zero-action target read from the
     action term equals `BAT_CARRY_POSE`; post-reset `joint_pos` equals `BAT_CARRY_POSE`
     clamped to the soft joint limits; the entity default root height equals
     `BAT_CARRY_BASE_HEIGHT`; the instantaneous self-collision force is 0; and
     `pose_quadratic` is 0 on the arm and waist group at the exact default pose.
   - 8 environments, 100 steps, zero actions. Root height, upright error, per-group pose
     error and fall count are a **diagnostic**, not a pass/fail criterion: with
     `use_default_offset=True` a zero action is only a PD hold at the default pose, not a
     balancing policy. The same recipe runs on the existing no-state-estimation velocity
     task for comparison.

## 5a. Inherited behaviour: soft-limit clamp at reset

The base task's `reset_joints_by_offset` event clamps the joint positions to the soft
joint limits. `BAT_CARRY_POSE` has two joints outside those limits, so the post-reset
state is not the exact pose:

| Joint | Carry pose | Clamped | Delta | Soft limits |
|---|---|---|---|---|
| `left_shoulder_roll_joint` | 0.0 | 0.0917 | +0.0917 rad (+5.254 deg) | (0.0917, 2.8403) |
| `head_yaw_joint` | 0.275995 | 0.270000 | -0.005995 rad (-0.344 deg) | (-0.27, 0.27) |

Consequences, measured and not fixed here:

- `pose_quadratic` is 0 at the exact `BAT_CARRY_POSE`, and 0.049464 raw (-0.019786
  weighted) at the post-reset clamped state.
- The `dof_pos_limits` term is 0 at the post-reset clamped state, and 0.097695 raw per
  step (weight -1.0) at the exact carry pose: `left_shoulder_roll_joint` 0.0917 plus
  `head_yaw_joint` 0.005995. PD control pulls the shoulder back to 0.0, so the carry pose
  is charged that penalty while it is held.
- The existing tennis task uses the same `BAT_CARRY_POSE` and the same reset event, so it
  has the same clamp and the same cost today. A divergence here would be worse than the
  documented clamp.
- The entity default root height is stored as float32: 0.669841051 m measured against
  0.669841065 m in the constants, a 1.4e-8 m difference.

## 6. Verification

- `pytest tests/test_task_configs.py -q` passes. The new task id is in `list_tasks()` by
  registration; `tests/test_task_configs.py` itself is unchanged because its only list
  names tasks whose corruption is deliberately disabled.
- `ruff format --check` and `ruff check` pass on every touched file.
- Both probes run and print their evidence. The stand probe's step-0 acceptance must pass;
  its 100-step drift and fall count are diagnostic only.
- No existing task's resolved config changes. The tennis stages and
  `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation` must be byte-identical before and
  after.

## Open decisions (not blocking the implementation)

Four findings from review that need an operator decision before a long run. The code is
correct without them; none is a defect in this task alone.

1. ~~PD gains do not match the juggle family.~~ **Resolved.** The task now applies the
   vendor reference-policy gains, the same set the juggle policy runs. The config probe
   asserts equality with the tennis juggle task on all 29 vendor joints.
2. ~~Physics step is 5 ms with a 1 ms-fitted bed contact.~~ **Closed as no risk.** There is
   no ball in this task and the racket is rigid, so the 1 ms-fitted string bed is never
   loaded. The 5 ms step is inherited unchanged.
3. ~~The ``pose`` mean now covers 14 joints, not 31.~~ **Rejected.** `pose` keeps all 31
   joints, so the base task's per-joint sensitivity is preserved. The arithmetic that made
   the restricted variant unattractive is kept in section 3.
4. **``pose_quadratic`` uses a mean over 17 joints.** A single wrist off by 20 degrees costs
   only -0.287; the 7 racket-arm joints off by 10 degrees costs -0.502. That is a global
   posture pull, not a per-joint racket lock. If the racket must be held tightly by the
   wrist, the reduction has to change.

## 7. Out of scope

Controller or deployment configuration, the ONNX export, the training launch, mixture
design, and any hardware activity. Those follow a separate, explicit request.

## What was built

Factory `agibot_x2_flat_velocity_carry_env_cfg(play=False)` in
`src/mjlab/tasks/velocity/config/agibot_x2/env_cfgs.py` and the task
`Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-CarryVelocity` with experiment name
`agibot_x2_velocity_carry_pose`. Probes live in `scripts/experiments/x2_carry_velocity/`.

Resolved values:

| Item | Value |
|---|---|
| Robot | `get_x2_tennis_robot_cfg()`, `spec_fn = add_velocity_foot_sites(get_tennis_spec())` |
| `pose` | 14 joints: 12 leg + 2 head; std tables 1 / 7 / 7 entries |
| `pose_quadratic` | 17 joints: 14 arm + 3 waist; std 0.10 rad; weight -0.4 |
| Actor / critic width | 102 / 117 (existing velocity task: 102 / 117) |
| Actions | 31 (existing velocity task: 31) |
| Command ranges | x (-0.6, 0.6), y (-0.3, 0.3), yaw (-0.5, 0.5), heading (-pi, pi) |
| `rel_standing_envs` | 0.25 |
| `curriculum` | `{}` |
| `reset_base` z offset | (0.0, 0.02) |
| `default_joint_pos` vs `BAT_CARRY_POSE` | max 5.508e-08 rad |

`check_carry_velocity_cfg.py` (RESULT: PASS) also reports: racket bodies `racket_clamp`
and `tennis_racket`, site `face_center`, foot sites `left_foot` and `right_foot` in the
compiled spec; `pose` and `pose_quadratic` disjoint and covering all 31 joints; and the
quadratic arm/waist cost below.

| Arm offset | Raw `pose_quadratic` | Weighted (-0.4) |
|---|---|---|
| 0 deg | 0.000000 | 0.000000 |
| 5 deg | 0.761543 | -0.304617 |
| 10 deg | 3.046174 | -1.218470 |
| 20 deg | 12.184696 | -4.873878 |

`probe_carry_stand.py` (STEP 0: PASS; drift diagnostic), 8 environments, seed 11:

- Step 0: `default_joint_pos` and the zero-action target equal `BAT_CARRY_POSE` within
  5.508e-08 rad; post-reset joints equal the soft-limit-clamped pose exactly; entity
  default root z 0.669841051 m; measured reset root z in [0.670659, 0.686748] m, i.e. the
  configured 0-2 cm offset; instantaneous self-collision force 0 N; `pose_quadratic` 0 at
  the exact pose and 0.049464 raw at the clamped reset state.
- Clamp: `left_shoulder_roll_joint` 0.0 -> 0.0917 rad (+5.254 deg), `head_yaw_joint`
  0.275995 -> 0.270000 rad (-0.344 deg).
- `dof_pos_limits`: 0 raw at the clamped reset state; 0.097695 raw at the exact carry
  pose, from `left_shoulder_roll_joint` 0.0917 and `head_yaw_joint` 0.005995.
- 100-step zero-action drift, carry vs base: root height mean 0.6227 vs 0.5926 m; torso
  tilt mean 29.42 vs 15.81 deg; leg+head pose error mean 2.641 vs 3.664 deg; arm+waist
  pose error mean 2.783 vs 2.084 deg; falls 8/8 in both, first fall at steps
  [64, 64, 66, 60, 65, 70, 70, 67] carry and [69, 70, 69, 63, 70, 78, 75, 69] base;
  self-collision force 0 N in both. The rollout uses a fixed seed; GPU reduction order
  gives small run-to-run variation in these diagnostic means.

Verification run: `pytest tests/test_task_configs.py -q` 9 passed; `ruff format --check`
and `ruff check` pass on every touched file; resolved-config fingerprints of
`Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation` and the S10 stage are identical before
and after the change. No training run, no hardware activity, no commit.
