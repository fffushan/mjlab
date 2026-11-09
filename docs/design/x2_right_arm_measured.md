# Measured right-arm servo: the second robot variant

Date: 2026-11-09 (+08).
Status: **implemented and registered. Nothing trained.**

A second robot variant for the tennis tasks, built from the right-arm joint step
response measured on the robot on 2026-11-09. It changes four damping values and
nothing else, and it exists for both the fixed-height ladder (S1 to S11) and the
target-height family (S1, S1a, S2, S3).

## 1. What was measured

`x2_docker/tools/joint_step_test.py` stepped each of the seven right-arm joints on
robot `.234`, 0.5/1/3/10 deg both signs, 3 s holds, at the vendor RL gains these
tasks train with, **with the racket held**, in the carry pose, controller stopped.
Report: `x2_docker/docs/reports/REPORT-joint-step-response.md`. Peer audit:
`AUDIT-joint-step-model-20261109.md`.

The racket matters more than it looks. It is a large part of the shoulder inertia:
the carry-pose mass-matrix diagonal at the right shoulder pitch is **0.2357 kg m²
with the racket welded on** against **0.1851 kg m² for the bare robot**. The first
version of this document used the bare value and produced `kd 4.25 / 4.46`; that was
wrong for this task family and is withdrawn.

## 2. What changed

Damping only, on the four proximal right-arm joints. Stiffness is untouched, but the
**closed-loop response changes**: raising kd raises the damping ratio, which slows the
rise and removes the overshoot. "Same kp, same speed" would be wrong.

How the values were chosen: not by dividing a damping ratio by a frequency. The measured
steps of each joint were matched by **the same shape residual the two-parameter servo fit
uses**, holding the simulated frequency fixed (kd cannot move it) and sweeping kd alone.
One parameter instead of two, same objective, so the result is the fit's own answer
rather than a formula. Residual over the 3 and 10 deg steps:

| joint | shipped kd | residual | **new kd** | residual | removed |
|---|---:|---:|---:|---:|---:|
| `right_shoulder_pitch` | 3.0 | 9.5 % | **4.74** | 2.1 % | 78 % |
| `right_shoulder_roll` | 3.0 | 10.1 % | **4.78** | 2.8 % | 72 % |
| `right_shoulder_yaw` | 3.0 | 6.2 % | **4.38** | 2.3 % | 64 % |
| `right_elbow` | 3.0 | 4.7 % | **3.90** | 2.9 % | 38 % |
| `right_wrist_yaw` | 2.0 | 3.3 % | 2.06 (not applied) | 3.3 % | 1 % |
| `right_wrist_pitch` | 2.0 | 4.9 % | 1.96 (not applied) | 4.9 % | 0 % |

**Provenance of the sweep, stated as a limitation.** The optimisation itself was
computed inline by the controller session and **no retained script or output exists**:
`x2_docker/tools/joint_step_fit.py` holds the two-parameter fit and the shape residual,
not a constrained sweep, and `~/data/robot234_step/fit.json` holds no swept optimum. The
independent audit covers the fitter (2-4 % shape agreement), not the sweep. The numbers
below are therefore a recorded computation, not a reproducible one, until the sweep is
committed with its output.

At those values the implied damping ratio in the tennis scene is **0.69** (pitch),
**0.70** (roll), **1.05** (shoulder yaw) and **1.10** (elbow) — the two distal ones above
critically damped, which is what the robot's 0-1.8 % overshoot there says.

Two earlier versions of this section are withdrawn:

* `kd = 2 zeta sqrt(kp J)` at the racket-held diagonal gave 4.81 / 4.78 on the shoulders
  and left the yaw and elbow alone. The shoulders **agree** (4.74 and 4.78); the yaw and
  elbow **do not** — the residual shows that leaving their damping error in place costs
  more than the over-damping does, so they move.
* A `kd / (2 kp)` ratio match gave 4.60 and 4.26 on those two joints. The residual
  optimum sits between the two rules (4.38 and 3.90) because the simulated frequency is
  19-29 % high there and the measured shape penalises the resulting time compression.

The racket-held inertia still matters: the shoulder pitch diagonal is **0.2357 kg m² with
the racket welded on** against **0.1851 kg m² for the bare robot**, and the bare value is
what a first reading of the report used. With the racket included the simulated shoulder
frequency already matches the measurement — **2.32 Hz against 2.34 Hz measured** on the
pitch axis — so on that axis the inertia is corroborated and damping was the only defect.

What the shipped gains cost at that inertia: `zeta` 0.437 and 0.439, and **21.7 % and
21.5 % overshoot** at 10 deg, against **0.8-0.9 % measured on the robot**. The report
first quoted 16.7 % and 18.6 % from the bare robot's inertia; it is corrected in its
section 0.

## 3. This is a training gain, not a deployment gain

**Do not send 4.74 / 4.78 / 4.38 / 3.90 to the robot.** The robot already produces
`zeta` 0.70 (shoulders) and 1.05-1.10 (yaw, elbow) at the commanded kd 3.0; that is what
the measurement is. The extra damping in the simulator imitates damping the hardware
already has, so commanding these values to the hardware would plausibly add it a second
time. **The size of that effect is not established**: it assumes the hardware damping
scales with the commanded kd, which is exactly what nobody has measured. What is
established is that the hardware already shows `zeta` 0.70 at kd 3.0, so kd 3.0 is the
value consistent with the measured response, and the command should stay there.

This is a live trap rather than a note, because the controller takes the RL gains from
the exported policy metadata times `policy.gain_multiplier`, and
`x2_docker/.../config/hardware/x2_tennis_hardware.yaml` states it: "The RL path is
unaffected either way: it always uses policy metadata x policy.gain_multiplier." A
policy trained on a measured twin therefore carries all four values into the robot
automatically, with no config edit and no warning.

Three routes, none of them taken yet:

1. **Train on these values, deploy at the vendor 3.0 / 2.0** through an explicit
   per-joint hardware gain override,
   so the exported metadata stays the training truth and the hardware path is a named
   decision. Preferred by the controller session.
2. **Measure the servo's gain response first.** A sweep (kp {25, 50, 75} x kd {3, 6})
   answers whether the firmware honours the commanded kd at all. If it ignores it, the
   metadata route is inert on this robot and the double count cannot happen; if it
   honours it, route 1 is required.
3. **Train at 3.0 and accept the shoulder ringing.** Honest and less accurate.

Nothing from this document has been exported, deployed or sent to a robot.

## 4. What was deliberately not changed

| joint | J (racket) | shipped ζ | residual at shipped | kd at the optimum | why untouched |
|---|---:|---:|---:|---:|---|
| `right_wrist_yaw` | 0.0440 | 1.07 | 3.3 % | 2.06 | no residual left to buy |
| `right_wrist_pitch` | 0.0396 | 1.12 | 4.9 % | 1.96 | same |
| `right_wrist_roll` | 0.0052 | 3.11 | — | — | ridge spans 3-32 Hz, nothing identified |

The wrists are the case where **both** of the rules on offer were wrong. The shipped
`2.0` lies within 3 % of the residual optimum, and the residual does not improve when kd
moves (3.3 % and 4.9 % at the optimum against 3.3 % and 4.9 % shipped). Whether that is a
noise floor is **not proved**: what is shown is that the residual stops responding, not
that it has reached an irreducible level. The `zeta` route would have softened them 15 %, the ratio route
stiffened them 10 %, and both would have moved away from the data. The reason to leave
them is not that the inertia is uncertain — it is that there is nothing for kd to buy,
which is a better reason because it also says when to stop.

Also unchanged: dry friction. The measured overshoot is amplitude dependent (0.9 % at
3 deg, 4.3-7.4 % at 10 deg), which no linear servo reproduces. The controller session
offers to fit a deadband; it is not fitted and not added.

The four values inherit the simulated frequency error, which is the degenerate
`(J, alpha)` axis of the report's section 4a. If that axis is ever resolved, recompute
them from the steps rather than carrying these forward.

## 5. How a right-arm-only value is expressed

A `BuiltinPositionActuatorCfg` holds one `damping` for all of its targets, and the
shipped articulation covers both arms with one regex per joint. A right-only damping
therefore cannot be written into that layout, so
`apply_right_arm_measured_servo` splits each affected group -- shoulder pitch, roll, yaw
and elbow -- into a left group and a right group, left first.

That order is what each shared group already resolved to, so the actuator list is
unchanged apart from four damping values. This is load-bearing: the action vector is
ordered by actuator, so a reordering would silently remap every checkpoint and the
controller's 29-wide action contract. Verified against the built model in both
families: 31 joints, identical order, identical `dof_armature`.

## 6. The tasks

`agibot_x2_tennis_juggle_stage_env_cfg` and
`agibot_x2_tennis_juggle_target_height_env_cfg` take `right_arm_measured: bool`.
The registrations are generated from `STAGE_DELTAS` and
`STAGE_TARGET_HEIGHT_RANGE_M`, so a stage cannot exist in one ladder and be missing
from the other. 23 tasks: 19 ladder stages, 4 target-height stages.

```
Mjlab-...-Ball3-Pelvis-Nominal-RightArmMeasured-<Stage>
```

Each family has its own experiment name, so a run and its measured twin can never land
in the same log directory:

```
agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_right_arm_measured
agibot_x2_velocity_tennis_ball_juggle_29dof_ball3_pelvis_nominal_s1_target_height_right_arm_measured
```

Widths are unchanged: 112 actor / 121 critic / 29 actions for the ladder, 113 / 122 / 29
for target height. Every shipped checkpoint resumes into its measured twin with
`strict=True`.

## 7. Verification

`scripts/experiments/x2_right_arm_measured/probe_right_arm_measured.py` — **RESULT:
PASS**. For all 23 pairs:

1. Every stage of both families has a measured twin, 23 registered.
2. Exactly four servo entries move, all damping, all to the values above.
3. The **action-to-joint mapping is identical**, read from the compiled models'
   `actuator_trnid` actuator by actuator rather than from config objects, so a silent
   reorder cannot pass. `dof_armature` is identical, every compiled stiffness is
   identical, every effort limit is identical, and the set of actuators whose damping
   moved is exactly the two the servo table names.
4. Eleven config sections are identical between a stage and its twin: observations,
   rewards, terminations, metrics, events, commands, actions, curriculum, sim,
   `episode_length_s`, decimation. That covers the action scale map, which is keyed by
   shared joint patterns and is therefore untouched by the split.
5. The compiled MuJoCo actuators carry kp 50 with kd 4.74 / 4.78 / 4.38 / 3.90, and
   the implied damping ratios 0.69 / 0.70 / 1.05 / 1.10.
6. The values are the recorded residual optima, not formula outputs, and the simulated
   frequency stays within 35 % of the measured one on every changed joint (1-8 % on the
   shoulders, 19-29 % on the yaw and elbow, which is the known gap).

`pytest tests/test_task_configs.py tests/test_x2_tennis_stage_task.py
tests/test_x2_tennis_juggle_task.py` — **89 passed**.

## 8. Independent review

`docs/reports/x2_right_arm_measured_review_20261109.md` (IdentificationDiag, same day,
reviewed commit `cfdd27014`). Verdict: a reasonable **calibration candidate**, not a
validated robot-response model. It confirms the architecture (loaded arm and racket
physics retained, no cascade onto the measured response, no direct joint overwrite) and
the four-kd change, reran the probe (23 pairs PASS) and the focused tests (89 passed).

Its three gates, all **open**:

1. Replay the measured commands through the actual simulator and compare responses.
   The residuals quoted here come from the scalar ODE, not from an `mj_step`.
2. Preserve and independently reproduce the constrained damping sweep (see the
   provenance limitation in section 2).
3. Enforce the simulation-versus-hardware gain distinction before deployment. Section 3
   describes the requirement; nothing enforces it yet.

## 9. What this does not establish

* The simulated frequency at 19-32 % above the measurement on the yaw, elbow and
  wrists is unresolved. It needs an independent torque calibration or a known added
  inertia to separate `J` from the gain scale.
* The amplitude dependence of the measured overshoot (0.9 % at 3 deg, 4.3-7.4 % at
  10 deg) is not reproduced by any linear servo. Dry friction is the candidate; it is
  not fitted and not added.
* The transport delay of the measurement is **not resolved**. The two logs share a
  clock but their origins differ by a constant near 7.25 s, estimated to scatter over
  7.30-7.43 s. That is a clock-origin offset, not a delay. Anything below about 0.1 s is
  inside the scatter, so the fit cannot separate delay from shape at that scale.
* Four arms are **training** on this variant at the time of writing (from scratch, 6000
  iterations, two seeds for the fixed 0.40 m family and two for the target-height family).
  They are a **damping ablation**: they can show that a policy learns under these gains,
  not that the simulator matches the measured robot. Whether the measured damping helps
  the juggle is **not measured** until those runs are analysed against their shipped
  baselines, and no analysis from them establishes calibration. The falsifiable check, from the controller session:
  `~/data/sim_vs_robot/arm_compare.py` compares right-arm position and velocity spectra,
  simulation against a `.234` run at the same pose and gains. Before the change the
  simulated `shoulder_pitch` had 8.43 deg p-p against 22.80 on the robot, with 75 % of
  its energy above 3 Hz against 2 % on the robot. If damping is the missing element, the
  simulated high-frequency content should fall and its p-p should rise.
