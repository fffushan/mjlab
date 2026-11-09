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

## 3. What was deliberately not changed

| joint | measured ζ | shipped ζ at racket J | sim ωn / measured ωn | why untouched |
|---|---:|---:|---:|---|
| `right_shoulder_yaw` | 0.93 | 0.72 | 3.82 / 3.21 Hz | a ζ match means kd 3.88, on a joint 19 % fast |
| `right_elbow` | 0.93 | 0.85 | 4.48 / 3.47 Hz | kd 3.30, on a joint 29 % fast |
| `right_wrist_yaw` | 0.93 | 1.07 | 3.39 / 2.65 Hz | kd 1.74 would *soften* a joint that already trails |
| `right_wrist_pitch` | 0.93 | 1.12 | 3.58 / 2.72 Hz | kd 1.65, same |
| `right_wrist_roll` | not identified | 3.11 | 9.89 Hz | no measurement to match |

The frequency gaps are the degenerate `(J, alpha)` axis of the report's section 4a:
any `(J, alpha)` pair on the same ratio reproduces the response, so a lower simulated
ωn cannot be attributed to inertia. No damping value fixes it. Moving kd on those
joints would trade a value with hardware trials behind it for a small change that does
not address the actual gap.

## 4. How a right-arm-only value is expressed

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

## 5. The tasks

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

## 6. Verification

`scripts/experiments/x2_right_arm_measured/probe_right_arm_measured.py` — **RESULT:
PASS**. For all 23 pairs:

1. Every stage of both families has a measured twin, 23 registered.
2. Exactly two servo entries move, both damping, both to the values above.
3. The resolved actuator order is identical, 31 joints, and `dof_armature` is
   identical.
4. Eleven config sections are identical between a stage and its twin: observations,
   rewards, terminations, metrics, events, commands, actions, curriculum, sim,
   `episode_length_s`, decimation.
5. The compiled MuJoCo actuators carry kp 50 with kd 4.81 / 4.78.
6. The values reproduce the fit: `2 ζ √(kp J)` at the racket-held diagonal, and the
   simulated frequency agrees with the measured one within 10 %.

`pytest tests/test_task_configs.py tests/test_x2_tennis_stage_task.py
tests/test_x2_tennis_juggle_task.py` — **89 passed**.

## 7. What this does not establish

* The simulated frequency at 19-32 % above the measurement on the yaw, elbow and
  wrists is unresolved. It needs an independent torque calibration or a known added
  inertia to separate `J` from the gain scale.
* The amplitude dependence of the measured overshoot (0.9 % at 3 deg, 4.3-7.4 % at
  10 deg) is not reproduced by any linear servo. Dry friction is the candidate; it is
  not fitted and not added.
* The transport delay of the measurement is 7.25 s plus a few milliseconds of scatter,
  so nothing below about 0.1 s is resolved by it.
* No policy has been trained on this variant. Whether the measured damping helps the
  juggle is **not measured**.
