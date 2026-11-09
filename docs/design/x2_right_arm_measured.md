# Measured right-arm servo: the second robot variant

Date: 2026-11-09 (+08).
Status: **implemented and registered. Nothing trained.**

A second robot variant for the tennis tasks, built from the right-arm joint step
response measured on the robot on 2026-11-09. It changes two servo values and
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

Only damping, only the two right shoulder axes. `kd = 2 ζ √(kp J)` with `ζ = 0.70`
from the fit and `J` the racket-held diagonal.

| joint | J (racket held) | kp | shipped kd | shipped ζ | **measured ζ** | **new kd** |
|---|---:|---:|---:|---:|---:|---:|
| `right_shoulder_pitch` | 0.2357 | 50 | 3.0 | 0.44 | 0.70 | **4.81** |
| `right_shoulder_roll` | 0.2333 | 50 | 3.0 | 0.44 | 0.70 | **4.78** |

`stiffness` is untouched, so the commanded response speed is untouched. With the
racket included the simulated shoulder frequency already matches the measurement:
**2.32 Hz simulated against 2.34 Hz measured** on the pitch axis. That is the useful
part of this measurement: on that axis the inertia is corroborated and damping was the
only defect.

What the shipped gains cost, at the racket-held inertia: `zeta` 0.437 and 0.439, and
**21.7 % and 21.5 % overshoot** at 10 deg, against **0.8-0.9 % measured on the robot**.
The peer report first quoted 16.7 % and 18.6 %; those were computed on the bare robot's
inertia and are corrected in its section 0. The argument here is 1.3x stronger than the
first version of this document said.

## 3. This is a training gain, not a deployment gain

**Do not send 4.81 / 4.78 to the robot.** The robot already produces `zeta` 0.70 at the
commanded kd 3.0; that is what the measurement is. The extra damping in the simulator is
imitating damping the hardware already has, so commanding 4.81 to the hardware would add
it a second time and land near `zeta` 1.0, with the mismatch reappearing in the opposite
direction.

This is a live trap rather than a note, because the controller takes the RL gains from
the exported policy metadata times `policy.gain_multiplier`, and
`x2_docker/.../config/hardware/x2_tennis_hardware.yaml` states it: "The RL path is
unaffected either way: it always uses policy metadata x policy.gain_multiplier." A
policy trained on a measured twin therefore carries 4.81 into the robot automatically,
with no config edit and no warning.

Three routes, none of them taken yet:

1. **Train at 4.81, deploy at 3.0** through an explicit per-joint hardware gain override,
   so the exported metadata stays the training truth and the hardware path is a named
   decision. Preferred by the controller session.
2. **Measure the servo's gain response first.** A sweep (kp {25, 50, 75} x kd {3, 6})
   answers whether the firmware honours the commanded kd at all. If it ignores it, the
   metadata route is inert on this robot and the double count cannot happen; if it
   honours it, route 1 is required.
3. **Train at 3.0 and accept the shoulder ringing.** Honest and less accurate.

Nothing from this document has been exported, deployed or sent to a robot.

## 4. What was deliberately not changed

| joint | J (racket) | measured ζ | shipped ζ | sim ωn / measured ωn | ζ-match kd at sim J | ratio-route kd |
|---|---:|---:|---:|---:|---:|---:|
| `right_shoulder_yaw` | 0.0870 | 0.93 | 0.72 | 3.82 / 3.21 Hz | 3.88 | 4.60 |
| `right_elbow` | 0.0630 | 0.93 | 0.85 | 4.48 / 3.47 Hz | 3.30 | 4.26 |
| `right_wrist_yaw` | 0.0440 | 0.93 | 1.07 | 3.39 / 2.65 Hz | 1.74 | 2.23 |
| `right_wrist_pitch` | 0.0396 | 0.93 | 1.12 | 3.58 / 2.72 Hz | 1.65 | 2.17 |
| `right_wrist_roll` | 0.0052 | not identified | 3.11 | 9.89 Hz | — | — |

Two statements, and the peer audit is right that they are not the same statement.

1. **The damping ratio is matchable without knowing the inertia.** For
   `q'' = a u - a q - b q'` both ω and ζ scale with √α, so `ζ/ωn = kd/(2kp)` contains
   neither J nor α. An earlier version of this document said the residual gap "is the
   `(J, α)` axis, which no damping value fixes". That holds for the **frequency** gap and
   is wrong for ζ. The two ζ-match columns above exist because of that correction.
2. **It is still not done, for two reasons.** The mismatch at the shipped gains is small
   on the yaw and elbow (0.72 and 0.85 against 0.93), and at the wrists the motion is the
   other way: the shipped gains are already *over*-damped (1.07, 1.12) because the
   simulated wrist frequency is 25-28 % high. The two prescriptions disagree by 20-35 %
   on exactly the joints where that frequency gap is largest — the ratio route would
   raise the wrist kd and make the over-damping worse (1.19 against 0.93), while the
   ζ route would lower it by 15 %. Two rules that disagree is the signal that this data
   does not choose between them.

Also unchanged: dry friction. The measured overshoot depends on amplitude (0.9 % at
3 deg, 4.3-7.4 % at 10 deg), which no linear servo reproduces. The controller session
offers to fit a deadband; it is not fitted and not added.

## 5. How a right-arm-only value is expressed

A `BuiltinPositionActuatorCfg` holds one `damping` for all of its targets, and the
shipped articulation covers both arms with one regex per joint. A right-only damping
therefore cannot be written into that layout, so
`apply_right_arm_measured_servo` splits the two affected groups into a left group and
a right group, left first.

That order is what the shared group already resolved to, so the actuator list is
unchanged apart from two damping values. This is load-bearing: the action vector is
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
2. Exactly two servo entries move, both damping, both to the values above.
3. The **action-to-joint mapping is identical**, read from the compiled models'
   `actuator_trnid` actuator by actuator rather than from config objects, so a silent
   reorder cannot pass. `dof_armature` is identical, every compiled stiffness is
   identical, every effort limit is identical, and the set of actuators whose damping
   moved is exactly the two the servo table names.
4. Eleven config sections are identical between a stage and its twin: observations,
   rewards, terminations, metrics, events, commands, actions, curriculum, sim,
   `episode_length_s`, decimation. That covers the action scale map, which is keyed by
   shared joint patterns and is therefore untouched by the split.
5. The compiled MuJoCo actuators carry kp 50 with kd 4.81 / 4.78.
6. The values reproduce the fit: `2 ζ √(kp J)` at the racket-held diagonal, and the
   simulated frequency agrees with the measured one within 10 %.

`pytest tests/test_task_configs.py tests/test_x2_tennis_stage_task.py
tests/test_x2_tennis_juggle_task.py` — **89 passed**.

## 8. What this does not establish

* The simulated frequency at 19-32 % above the measurement on the yaw, elbow and
  wrists is unresolved. It needs an independent torque calibration or a known added
  inertia to separate `J` from the gain scale.
* The amplitude dependence of the measured overshoot (0.9 % at 3 deg, 4.3-7.4 % at
  10 deg) is not reproduced by any linear servo. Dry friction is the candidate; it is
  not fitted and not added.
* The transport delay of the measurement is 7.25 s plus a few milliseconds of scatter,
  so nothing below about 0.1 s is resolved by it.
* No policy has been trained on this variant. Whether the measured damping helps the
  juggle is **not measured**. The falsifiable check, from the controller session:
  `~/data/sim_vs_robot/arm_compare.py` compares right-arm position and velocity spectra,
  simulation against a `.234` run at the same pose and gains. Before the change the
  simulated `shoulder_pitch` had 8.43 deg p-p against 22.80 on the robot, with 75 % of
  its energy above 3 Hz against 2 % on the robot. If damping is the missing element, the
  simulated high-frequency content should fall and its p-p should rise.
