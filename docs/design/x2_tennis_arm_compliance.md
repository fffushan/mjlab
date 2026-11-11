# X2 juggling: arm and racket compliance design

Date: 2026-11-11 (+08).
Status: **design only**. No implementation, simulation run, training, or robot operation.
Scope: the X2 right arm, its racket assembly, and the tennis juggling task.

## 1. Decision

Keep the string-bed contact unchanged. Identify the **coupled arm–racket response** before fitting new parameters.

The training task already has finite joint stiffness. It does not have structural flexibility in the wrist-to-racket assembly.
Start with the existing joint model. Add a small passive compliance model only if measured residual motion requires it.
Do not set contact restitution to the measured whole-system ratio or infer a spring constant from that number.

## 2. Evidence and limits

The [measurement report](../../../x2_docker/docs/reports/REPORT-racket-contact-restitution.md)
reports these results from 2026-11-11 (+08):

| Support condition | Median height-based ratio `e_h` | IQR | Bounces |
| --- | ---: | ---: | ---: |
| Racket supported on a rigid box | 0.87 | 0.86–0.88 | 25 |
| Robot-held racket | 0.39 | 0.28–0.46 | 34 |

The measurement owner corrected an earlier result of 0.43 from 18 bounces.
Automatic bed detection used the last eight seconds, when the robot-session ball was not resting.
It selected the wrong impacts. The corrected analysis uses explicit bed heights:

```sh
# Run from x2_docker. These are the measurement owner's reproduction commands.
python3 tools/mocap_restitution.py --session ~/data/mocap_probe/20261111T124202Z --bed 1.0245 --band 0.06
python3 tools/mocap_restitution.py --session ~/data/mocap_probe/20261111T125848Z --bed 0.8652 --band 0.06
```

These commands were supplied by the measurement owner; this design review did not rerun them.
Neither session tracked the racket. The tracked bodies were `robot-234` and `Tennis_10MK`.
The robot bed height comes from resting-ball height, not the unresolved forward-kinematics estimate.

Here, IQR means interquartile range. The reported metric is:

`e_h = sqrt((z_apex_after - z_impact) / (z_apex_before - z_impact))`.

For a stationary horizontal surface and valid ballistic arcs, `e_h²` measures vertical ball-energy retention.
The medians imply approximately 0.76 and 0.15. These are **not measurements of energy dissipated in the arm**.
Energy can enter arm motion, elastic deformation, ball spin, or lateral ball motion. The servo can also exchange energy.
Stiffness stores energy; damping dissipates energy. Recoil alone can reduce the ball's rebound.

The support-condition difference is substantial. It does not identify one physical cause or one parameter set.
The box-supported measurement supports retaining the existing contact fit. It does not validate every impact speed or support condition.

The matched simulator drop result is **unknown**. The moving-racket replay ratio is not a replacement for it.
The reported 96-episode simulation gate does not establish physical transfer.

## 3. Current implementation

Source inspection, 2026-11-11 (+08):

- [`x2_constants.py`](../../src/mjlab/asset_zoo/robots/agibot_x2/x2_constants.py) uses `BuiltinPositionActuatorCfg`.
  It includes proportional–derivative (PD) gains, torque limits, reflected rotor inertia, and joint friction.
- [`tennis_juggle_env_cfg.py`](../../src/mjlab/tasks/velocity/config/agibot_x2/tennis_juggle_env_cfg.py)
  applies the vendor RL gains through `_tennis_robot_cfg()`.
- [`tennis_constants.py`](../../src/mjlab/asset_zoo/robots/agibot_x2/tennis_constants.py)
  fixes `racket_clamp` and `tennis_racket` to `right_wrist_roll_link`.
  The assembly already contributes mass and inertia. It has no passive deformation coordinates.
- The task uses a 1 ms physics step and a 20 ms policy step.
  The contact constants were fitted at 1 ms. A 5 ms step can inject energy.
- [`test_x2_tennis_ball_task.py`](../../tests/test_x2_tennis_ball_task.py)
  has an isolated, stationary-bed drop fixture. It does **not** identify coupled arm compliance.
- `RightArmMeasured` changes simulated damping. Its previous review classifies it as a calibration candidate,
  not a validated robot-response model. See the [review](../reports/x2_right_arm_measured_review_20261109.md).

Thus, the missing feature is not simply “PD stiffness.” The possible gap includes actuator response,
structural flexibility, support motion, and the loaded racket geometry.

## 4. First gate: matched drops, with no new physics

Use the actual task factory and save the compiled model, full config, and source revision.
Preserve `BED_SOLREF`, `BED_SOLIMP`, contact priority, ball properties, and torque limits.
Record the exact policy/config used in each comparison. Do not assume a mutable ONNX path still names the measured policy.

Run two distinct fixtures:

1. **Stationary bed:** reuse the contact regression fixture. It checks the contact and numerical method.
2. **Robot-held racket:** use the full coupled robot, actual support condition, measured pose, and measured gains.
   Start with shoulder/elbow `kp=50, kd=3` and wrist `kp=20, kd=2`, as reported for these drops.
   Confirm the complete per-joint table from the retained command evidence.

Let PD dynamics move the arm during contact. Do not freeze arm positions or overwrite `qpos` each step.
A fixed-base or locked-arm run is a diagnostic ablation, not a reproduction of standing hardware.
The measurement used the standing policy. A constant-target hold is a separate protocol unless its motion is shown equivalent.
Prefer retained target replay when available; otherwise label the target history as missing.

Match the observed release-height range, lateral impact locations, and racket tilt.
Include the juggling speed near 2.5 m/s. A zero-speed drop near 0.32 m produces that speed under gravity.
Do not fit only a 0.9 m drop; that condition is faster than typical juggling.
Measure bed position in the loaded pose rather than assuming the CAD chain is correct.

Record at each physics step:

- Ball position, velocity, spin, contact state, and contact impulse.
- Racket face pose, normal, and velocity at the contact point.
- Joint targets, positions, velocities, actuator forces, and torque saturation.
- Base motion and any passive deformation coordinates.
- Ball, robot, and elastic energy; actuator work and numerical energy residual where available.

Keep two result streams: full-rate simulation truth and simulation sampled through the measurement pipeline.
The current analyser interpolates to 100 Hz and smooths three samples before locating extrema.
That smoothing can raise the impact minimum. Test sample phase, missing frames, and this filter's bias.
The owner reports a synthetic bias test and proposes corrected ratios near 0.875 and 0.41.
Keep those as sensitivity estimates, not calibrated measurement targets or proven bounds.
The synthetic contact and noise assumptions need validation. Marker-fit error is not automatically position-noise standard deviation.
Use the raw 0.87 and 0.39 results for pipeline-matched comparisons until that validation exists.
Use accepted ballistic arcs and the first rebound apex. Exclude hand placement, catches, and subsequent lifts.
Do not fit velocity through the unresolved contact. A later ballistic-arc fit is valid only after segment checks.
Use one ball-centre convention for both apex and impact heights. Record the radius assumption explicitly.

**Branch:** if matched simulation already reproduces the robot, do not add compliance to lower the rebound again.
Investigate contact-point velocity, pose, observation timing, and controller differences instead.

## 5. Model choice if the mismatch remains

### A. Calibrate the existing coupled joint model first

Use the same model for command response and impact response:

`M(q) q_ddot + h(q, q_dot) = tau_servo + tau_passive + J_contact^T F_contact`.

`tau_servo = clip(Kp (q_target - q) - Kd q_dot, torque_limits)` is the nominal PD law.
Here, `M` includes the assembly and reflected rotor inertia. `h` includes gravity and velocity-dependent terms.
Use the actual target delay and actuator update method when known.

Check inertia, friction, torque saturation, torque tracking, and delay against independent evidence.
Do not treat all of these as unconstrained knobs to match one rebound median.
Do not increase armature or friction on the assumption that either must lower rebound.
The direction depends on the coupled system and its contact timescale.

An acceptable joint model must reproduce both held-out command responses and impact responses.
Previous scalar step-shape fits do not identify absolute inertia or torque scale on their own.

### B. Add passive structural compliance only for an unresolved deformation mode

If joint motion cannot explain measured face motion, add compliance between the relevant rigid bodies.
A minimal **candidate** is one translation along the bed normal. Add two tipping rotations only if data require them.
Place the mode at the measured flexible component when known: transmission, connector, clamp, or racket shaft.
If its location is unknown, label it a reduced-order endpoint model, not a measured mount property.

For a normal displacement `x`, the passive restoring force is:

`F_passive = -k_n (x - x_0) - c_n x_dot`.

For a tipping angle `theta`, use the corresponding restoring torque:

`tau_passive = -k_theta (theta - theta_0) - c_theta theta_dot`.

Use positive stiffness and nonnegative damping. Apply equal and opposite forces through the physical connection.
MuJoCo must integrate these coordinates with the arm and ball at each physics step.
Retain the string-bed contact on the moving racket body.
Do not add a scripted ball-velocity reduction, one-sided damping force, or contact-triggered energy correction.

Reuse the existing assembly mass and inertia. Do not add a duplicate racket mass or arbitrary large modal mass.
Any inertia split must preserve total mass, centre of mass, and inertia and must have a stated physical basis.
Identify the rest position under gravity. Use only measured travel limits; a numerical stop must not become a hidden energy source.
An elastic transmission model with separate motor and link motion is a later option, not the first implementation.
It needs evidence for transmission flex and the hardware encoder location.

## 6. What identifies stiffness and damping

The existing ball traces can constrain rebound and gross post-impact motion. They cannot uniquely identify `k` and `c`.
Use the joint traces already recorded before requesting more robot time.
Forward kinematics can expose joint recoil, but cannot prove unmeasured connector or shaft deformation.

If existing data do not separate the candidates, request a supervised measurement plan:

- Track the racket face independently of joint encoders.
- Measure displacement under known small loads: the load–displacement slope constrains stiffness.
- Record recovery after load removal: oscillation and decay constrain dynamic response and damping.
- Repeat centred drops at several heights and arm poses. Record impact point and bed motion.
- Consider gain variation only under separate operator approval. Keep deployment gains unchanged for the first comparison.

Use repeats to estimate uncertainty. Reserve whole drops or conditions for validation before fitting.
Do not split adjacent samples from one bounce between fitting and validation.
Fit rebound, face displacement/tilt, recovery, and joint response together.
A model that matches `e_h=0.39` but predicts the wrong recoil is rejected.
Do not turn the observed IQR directly into a uniform stiffness range; release, spin, and tilt also affect it.

## 7. Training integration

Add an opt-in compliance configuration. Keep existing task IDs and nominal behaviour unchanged.
Proposed owners:

- Physics assembly: `src/mjlab/asset_zoo/robots/agibot_x2/tennis_constants.py`, or a separate compliance helper.
- Task variant and reset wiring: `src/mjlab/tasks/velocity/config/agibot_x2/`.
- Offline calibration fixtures: `scripts/experiments/x2_tennis_compliance/`.
- Regression and contract tests: `tests/test_x2_tennis_compliance.py`.

These are proposed paths, not implemented artifacts.

If passive joints are added, explicitly select the **original hardware joint names** everywhere.
Audit actions, proprioception, resets, penalties, randomization, metadata, and export order.
Keep the existing pelvis-history actor at 112 inputs and 29 actions.
Do not expose passive deformation or exact contact impulse to the actor unless hardware supplies those observations.
Use the actual deformed face for contact, rewards, and evaluation.
Keep actor-side kinematics consistent with the hardware measurement path.

Keep measured hardware command gains separate from simulation-only corrections.
Export must not silently turn fitted damping or passive stiffness into new robot gains.
Require a metadata/preflight check against the approved hardware gain table.

First evaluate the frozen baseline under the calibrated model. Then fine-tune a verified juggling checkpoint.
Keep the observation contract, reward, target height, and torque limits unchanged in the first training comparison.
Add compliance progressively if needed, but require the final policy to pass at full calibrated compliance.
Randomize only supported parameter ranges. Sample a coherent parameter set per episode, not a new restitution per bounce.
Preserve parameter correlations when the data constrain combinations rather than individual values.

## 8. Acceptance gates

1. **Numerics:** compare 1, 0.5, and 0.25 ms offline with the same physical parameters and 20 ms policy period.
   Bound rebound and recoil differences relative to measurement uncertainty. Check CPU MuJoCo and training MuJoCo Warp.
   Do not retune contact at each timestep or accept a model that relies on numerical energy loss.
2. **Measurement:** reproduce the stationary-bed control and held-out robot drops through the same analysis pipeline.
   Set numerical acceptance bands from repeatability before fitting. Current bands are **not established**.
3. **Dynamics:** preserve independently measured command response. Match recoil and recovery, not only ball apex.
4. **Software:** prove no cross-environment state leakage, correct reset equilibrium, unchanged actor/export contract,
   unchanged default task, unchanged contact parameters, and verified hardware gains.
5. **Policy:** score qualified free flights, apex relative to takeoff height, streak length, drops, landing error,
   torso motion, joint motion, and torque saturation. Contact count alone cannot distinguish juggling from dribbling.
6. **Physical transfer:** use a separately authorized supervised comparison with the original policy and matched releases.
   Require sustained juggling and safe loss behaviour. Simulation success is screening evidence only.

## 9. Next action and authority

The next implementation step is the matched-drop fixture and measurement-pipeline check, not a training run.
Implementation and local simulation require approval of this design's scope.
New hardware measurements, gain changes, training launches, and deployment need separate authorization.
