# Review — using the loaded right-arm fit in tennis training

Date: 2026-11-09 (+08). Reviewed mjlab commit: `cfdd27014`.
Scope: source review, compiled-configuration checks, and focused software tests.
No new training, deployment, or robot operation was started.

## Verdict

**The implementation is a reasonable calibration candidate, not a validated robot-response model.**
It keeps the arm and racket physics and changes four simulated derivative gains.
It does not cascade the fitted loaded response with the existing joint response.
It does not overwrite joint positions.

Three gates remain:

1. Replay the measured commands through the actual simulator and compare the resulting responses.
2. Preserve and independently reproduce the constrained damping sweep.
3. Enforce the distinction between simulation calibration and hardware command gains before deployment.

The existing runs are damping-ablation experiments.
Their learning curves cannot establish that the simulator matches the measured robot.

## 1. What is implemented

Source: [`right_arm_measured.py`](../../src/mjlab/asset_zoo/robots/agibot_x2/right_arm_measured.py).

The actuator law remains the MuJoCo position servo, subject to the existing torque limits:

```text
tau = kp * (q_target - q) - kd_sim * q_velocity
```

| Joint | kp, unchanged | Shipped kd | Candidate kd |
| --- | ---: | ---: | ---: |
| right_shoulder_pitch | 50 | 3.00 | 4.74 |
| right_shoulder_roll | 50 | 3.00 | 4.78 |
| right_shoulder_yaw | 50 | 3.00 | 4.38 |
| right_elbow | 50 | 3.00 | 3.90 |

The wrists remain at `20/2`. Other joints retain their existing gains.
The implementation splits four shared left/right actuator groups into side-specific groups.
The left-side groups remain first, preserving actuator order.

The variant preserves armature, friction, target mapping, action scale, racket geometry, and racket attachment.
It adds no command filter and no prescribed joint trajectory.
The registration creates 23 tasks: 19 curriculum stages and four target-height stages.

Source paths:

- [`tennis_juggle_env_cfg.py`](../../src/mjlab/tasks/velocity/config/agibot_x2/tennis_juggle_env_cfg.py), `_tennis_robot_cfg()`.
- [`__init__.py`](../../src/mjlab/tasks/velocity/config/agibot_x2/__init__.py), generated `RightArmMeasured` registration.
- [`builtin_actuator.py`](../../src/mjlab/actuator/builtin_actuator.py), `BuiltinPositionActuator`.
- [Design document](../design/x2_right_arm_measured.md).

## 2. Why this is the appropriate model boundary

The measured fit includes the servo, loaded arm, racket, and measurement path near the carry pose.
It is not a separately identified motor transfer function.

The recommended use is to calibrate the simulator's complete target-to-position response against the measurement.
The simulator should still compute gravity, acceleration, coupling, and ball contact from its rigid-body model.
The loaded fit supplies a response target, not another physical load or an extra response filter.

For the present candidate, the four changes can be written as:

```text
kd_sim = kd_hardware_command + delta_kd_sim

delta_kd_sim = 1.74, 1.78, 1.38, 0.90
```

This is a useful software distinction, not proof that the missing mechanism is viscous friction.
The extra term could compensate for several errors in the simplified model.
Its effect must be checked in the full simulator.

Do not simply filter the desired position through the measured second-order response and retain the existing servo.
That would apply the measured loaded dynamics before another loaded dynamic system.
The resulting response would generally be slower and have more phase lag than the measurement.

Do not directly write fitted positions into `qpos` during normal training.
That would bypass torque, reaction, and contact behavior needed for ball juggling.

## 3. Findings

### F1 — Deployment gains are not protected by an executable guard

Severity: **high before any hardware deployment**.

[`exporter_utils.py`](../../src/mjlab/rl/exporter_utils.py):87 reads damping from the compiled simulator.
Line 148 writes those values as `joint_damping` metadata.
The controller's `BuildAdapterTarget()` multiplies that metadata by the configured scalar gain multiplier.
Source: `x2_docker/x2_controller/x2_rl_deploy_controller/src/policy_adapter.cc:531`.

Thus, exporting these tasks through the normal path would export the four candidate damping values.
The normal controller path would use them as hardware commands.
The measurement used hardware `kd = 3`, not these candidate values.
A global multiplier cannot restore the four unequal damping corrections without also changing other gains.

The commit named "block its deployment" adds warnings, not an executable block.
Both sessions confirmed this during review. No export or robot use was reported.

Required before deployment:

- Separate simulation-calibration parameters from hardware command gains explicitly.
- Reject hardware use when that distinction is absent or unresolved.
- Test the emitted hardware command values by joint name.
- Preserve both parameter tables and their provenance; do not silently relabel simulation gains as measured hardware gains.

The exact response to commanding larger gains on hardware is **not measured**.
Warnings should not claim a specific damping ratio or critical-damping crossing without a gain-response test.

### F2 — The reported error is a scalar-model error, not a simulated step-response error

Severity: **high for a calibration claim**.

The fitted damping values minimize a normalized shape error in an isolated second-order ordinary differential equation.
The mass input is the compiled robot entity's diagonal inertia at the carry pose, with racket included.

The [probe](../../scripts/experiments/x2_right_arm_measured/probe_right_arm_measured.py) compiles models and calls `mj_forward()` and `mj_fullM()`.
It does not run a commanded step through `mj_step()`.
StablizingBall confirmed that no dynamic step-response replay exists.

The actual simulator also includes:

- Existing dry friction: 0.12 N m on shoulder pitch/roll and 0.08 N m on shoulder yaw/elbow.
- Gravity that changes with pose, rather than only a constant fitted offset.
- Coupling through the other joints, which have finite holding stiffness.
- Base motion, contact constraints, and torque limits.
- The actual integration step and command-update path.

`M[j,j]` is the single-joint inertia when the other generalized accelerations are constrained to zero.
It does not by itself describe a coupled arm with other joints held by finite-gain servos.
Keeping the swing near the measured pose helps, but does not remove these terms.

Consequently, the reported 2.1–2.9% residual does not establish the same accuracy in the training simulator.
The damping changes may help. Their quantitative effect remains **not validated**.

### F3 — The constrained damping sweep has no retained runnable artifact

Severity: **medium; parameter provenance is incomplete**.

JugglingController confirmed that the sweep ran as an inline shell heredoc.
No script or numerical sweep output was retained.
The report and session messages contain the optimum table.
Neither `joint_step_fit.py --sim` nor the current `fit.json` reproduces that table.

The reported recipe pins `omega = sqrt(kp / J_racket)` and sweeps damping using `_shape_residual()`.
The raw steps are normalized by measured travel and aligned by their measured 50% crossings.
This is an in-sample shape objective, not a command-predictive validation.

The training probe compares the configured gains with copied literals.
It verifies installation of the table, not derivation of the table.

Required: preserve the sweep code, exact input identities, inertia source, grids, objective, and full output.
Then reproduce it independently before describing the four values as verified optima.
This review did not reconstruct or re-run that missing sweep.

### F4 — Documentation contains stale or unsupported statements

Severity: **medium for interpretation; low for task wiring**.

Examples in the design document and `right_arm_measured.py`:

- "Stiffness is untouched, so response speed is untouched." Damping changes bandwidth and rise time.
- The 0.8–0.9% shoulder overshoot is compared with a 10-degree simulator response despite amplitude dependence.
- The 7.25 s logging-origin offset is called a transport delay. It is not a measured transport delay.
- Simulated yaw/elbow damping ratios are described as measured hardware values.
- The module contains an obsolete block saying yaw and elbow remain unchanged, after the four-joint change.
- The document says nothing is trained, while the owner reports four training jobs already running.
- The residual is called a measurement floor. No repeated-run noise estimate establishes that floor.

Correct these statements before using the document as a validation record.
Do not infer a specific residual mechanism from the remaining error alone.

## 4. Independent checks performed

The review reran:

```bash
cd /home/agiuser/projects/mjlab
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  scripts/experiments/x2_right_arm_measured/probe_right_arm_measured.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest \
  tests/test_task_configs.py \
  tests/test_x2_tennis_stage_task.py \
  tests/test_x2_tennis_juggle_task.py \
  -x -vv -o faulthandler_timeout=60
```

Results:

- Probe: **PASS**, 23 task pairs.
- Exactly four compiled damping values change.
- Action-to-joint mapping, stiffness, effort limits, and armature remain equal in the checked arrays.
- The 11 compared configuration sections remain equal.
- Focused suite: **89 passed**, 18 warnings, 18.61 s.

The warnings concern entity timestep settings during attachment.
The simulation constructor applies `cfg.mujoco` after compilation; these warnings alone do not establish a timestep defect.
A dynamic replay should still record the final runtime timestep.

These checks validate software configuration and existing task behavior.
They do not validate the four candidate gains against hardware dynamics.

## 5. Next validation, before treating this as calibrated

1. Preserve the constrained sweep and its input versions.
2. Build the actual tennis scene for both baseline and candidate tasks.
3. Match racket, carry pose, gains, gravity, and the measurement's support condition.
4. If the support condition is unknown, state that limitation and test bounded support alternatives separately.
5. Hold the non-tested joints through their real simulated servos. Do not silently lock them to justify `M[j,j]`.
6. Apply the recorded target steps, including returns, in both directions.
7. Record actual positions, velocities, actuator forces, saturation, base motion, and final simulation settings.
8. Compare static sag, travel, rise, overshoot, and full trajectories using identical scoring rules.
9. Separate output-aligned shape scores from command-relative prediction. Do not report an unmeasured transport delay.
10. Select calibration values on one subset and validate on held-out transitions without fitting their output endpoints.
11. Check nearby poses and the actual policy's excursion and speed ranges before wider use.
12. Add the hardware-gain gate before any candidate export is used on the robot.

Policy evaluation is a separate gate.
Learning success under one simulator does not identify that simulator's physical accuracy.
A change in closed-loop spectra can arise from changed policy commands, not only changed actuator response.
Matched-command replay should precede a causal interpretation of those spectra.

## 6. Operational status and ownership

StablizingBall reports four previously authorized runs from scratch in `/mnt/fushan/mjlab-ra`.
Tasks: `RightArmMeasured-S1-15hz` and `RightArmMeasured-S1a-TargetHeight`, seeds 7 and 17.
Each run targets 6000 iterations with 4096 environments.
Its supplied progress snapshot is from 16:02 (+08); this review did not inspect the remote hosts.
Their current remote status is therefore **not independently verified**.

This review did not stop, restart, or modify those runs.
No production source, simulator parameter, export path, or hardware path was changed.
Only this review and the relevant shared-status entry were added or updated.

Implementation of the replay, provenance tooling, or deployment guard requires a separate request.
