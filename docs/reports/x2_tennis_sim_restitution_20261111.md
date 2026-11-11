# Report — simulated racket restitution against the measured robot

Date: 2026-11-11 (+08).
Repository: `/home/agiuser/projects/mjlab`, commit `28f422d4e` plus the uncommitted fixtures
named in section 8.
Scope: software measurement only. No gain, contact parameter, mesh or task was changed.
No training run, no export, no robot operation.
Status: **measured, reviewed twice, revised twice.** On 2026-11-11 the reviewer approved
two decisions only: section 4.2 (no compliance element on the current evidence) and section
6 (the order of the next three steps). They did **not** certify the numerical tables, which
remain unmatched to the hardware conventions. The verdict is that the model must not change
yet. The numbers are not a validated contact model.

Review: [`x2_tennis_sim_restitution_review_20261111.md`](x2_tennis_sim_restitution_review_20261111.md),
by the LumpDesign session. Section 10 lists every defect and how this revision answers it.

## 1. Question

The robot juggles worse than the policy in simulation. Two proposals followed:

1. change the string-bed contact so that the whole-system restitution matches the robot;
2. add a passive compliance element (a "lumped mount") between the wrist and the racket.

The JugglingSim2Real session measured the real system with motion capture:

| support | median `e_h` | IQR | n |
| --- | ---: | --- | ---: |
| racket clamped to a rigid box | 0.87 | 0.86-0.88 | 25 |
| racket held by the robot, held releases 0.84-0.97 m | **0.45** | 0.41-0.49 | 18 |

`e_h = sqrt(h_reb / h_drop)`. This report measures the same quantity in the simulation and
states where the two sides are not yet protocol-matched.

## 2. Method

Model:

- Task `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal`
  (vendor gains, not the measured-right-arm variant).
- The free base is welded to the world, because the real robot stood on the floor.
- The model takes the **task's own MuJoCo options** through `cfg.sim.mujoco.apply(model)`:
  `implicitfast`, 10 solver iterations, 20 line-search iterations, pyramidal cone, 1 ms step.
- Ball: shipped ball, 0.056 kg, radius 0.0327 m, `condim 6`, friction 0.44.
- `BED_SOLREF = (-100000.0, -22.0)` unchanged.

Pose and gains:

- The right arm and the waist take the medians the standing lane held on the robot.
- Gains: shoulder and elbow `kp` 50, `kd` 3.0; wrists `kp` 20, `kd` 2.0.
- The controller targets are gravity-compensated by inverse dynamics so that the settled
  pose equals the measured pose. **The residual is 4.8e-10 rad immediately after that
  settle.** The 20 mm bed-surface probe described below then perturbs the arm, and the
  residual measured afterwards is 0.012 rad. The 0.012 rad is a probe artefact, not a
  solver-induced equilibrium error.
- This validates the equilibrium position only. It does not reproduce the standing policy's
  dynamic impedance, its 50 Hz target stream, or the work that loop does during an impact.

Estimators:

- **Simulation truth (`raw`)**: minimum and apex from the 1 ms trace.
- **Pipeline (`pipeline`)**: the same quantities after resampling to 100 Hz and a 3-sample
  moving average. This is the only estimate directly comparable with a 100 Hz
  motion-capture analysis, so it is the number to compare with the robot.
- First bounce: the release is a held ball in both experiments, so its height is known.
  `h_drop = z_release - z_min`, `h_reb = z_apex - z_min`. There is no pre-release arc, so
  the extremum rule cannot find a maximum before the first impact.
- Later bounces: the extremum rule, excluding the first and last three samples.
- Acceptance gates, applied to the pipeline estimate: the minimum is within 0.06 m of the
  bed reference, the drop is at least 0.08 m, and a rebound exists.

Bed reference, and its caveat:

- The reference is `min(z) - radius` from a 20 mm drop, which gives 1.10621 m.
- The unloaded plate face is at 1.12109 m, so the reference sits **14.9 mm below** the face.
  It is a transient minimum of a small drop, not a resting surface.
- Apex heights below are quoted **above this proxy** and that convention is named in every
  table. They are not byte-comparable with a hardware apex that uses another convention.

Scripts: `scripts/experiments/x2_tennis_drop/` (`model.py`, `drop.py`, `tangential.py`).

## 3. Results

### 3.1 First bounce

| release above proxy | `h_drop` | `e_h` raw | `e_h` pipeline | apex above proxy |
| --- | ---: | ---: | ---: | ---: |
| 0.90 m | 0.899 | 0.5572 | **0.5448** | 0.313 m |
| 0.70 m | 0.697 | 0.5608 | **0.5469** | 0.255 m |
| 0.50 m | 0.495 | 0.5558 | **0.5389** | 0.190 m |
| 0.30 m | 0.293 | 0.5501 | **0.5274** | 0.128 m |

The simulation is **flat at 0.53-0.55** on the pipeline estimate. The robot gives 0.45 at
the only height it measured (0.84-0.97 m, IQR 0.41-0.49).

The gap is about 0.09-0.10 of `e_h`. Its ratio of apparent retention,
`0.5448² / 0.45² = 1.47`, is **not** a certified energy ratio: the two sides use different
apex conventions and the hardware apex of 0.24 m still lacks pinned paired rows. Do not
quote that ratio as a physical result.

Solver options are a small part of the earlier difference. Under this revised protocol, the
only change from the compiled defaults to the task options moves `e_h` from 0.547 to 0.557,
about 0.010. The earlier figure of 0.519 came from a different bundle: another estimator,
another bed reference and another settling sequence. The 0.038 difference cannot be
attributed to the solver.

### 3.2 Decay, per release

Rows are identified by release and bounce index. Only the shipped-gain condition is listed.

| release | bounce | `h_drop` | apex | `e_h` |
| --- | ---: | ---: | ---: | ---: |
| 0.90 m | 2 | 0.264 | 0.064 | 0.494 |
| 0.70 m | 2 | 0.207 | 0.059 | 0.534 |
| 0.50 m | 2 | 0.147 | 0.044 | 0.550 |
| 0.30 m | 2 | 0.089 | 0.032 | 0.601 |

The 0.90 m release produces one accepted later bounce only. The robot's 16 secondary
bounces have a median `e_h` of 0.28, but they are not paired with these rows: they carry
spin and arrive from different directions.

### 3.3 Wrist-gain intervention

| release | `e_h` raw, shipped | `e_h` raw, intervention | `e_h` pipeline, shipped | `e_h` pipeline, intervention |
| --- | ---: | ---: | ---: | ---: |
| 0.90 m | 0.5572 | 0.6405 | 0.5448 | 0.6305 |
| 0.70 m | 0.5608 | 0.6475 | 0.5469 | 0.6364 |
| 0.50 m | 0.5558 | 0.6470 | 0.5389 | 0.6337 |
| 0.30 m | 0.5501 | 0.6446 | 0.5274 | 0.6259 |

The intervention is `kp` 5000 / `kd` 50 on both wrists. It raises `e_h` by 0.086-0.099.

This is a **sensitivity to one gain change in this fixture**. It is not an upper bound on
a separate compliance mode. The intervention keeps the wrist torque limits (±4.8 N m for
pitch and roll, ±24 N m for yaw), so the wrist is not rigid, and a new mode would change the
frequency-dependent impedance, the contact duration and the elastic return timing, none of
which this ablation samples.

The measured statement is narrow: **this gain change raises the vertical rebound by
0.086-0.099 of `e_h`.** No energy partition between arm and contact is identified, and no
mount mechanism is implied.

### 3.4 Contact geometry

- First contact duration: 9-10 ms at impacts between 2.4 and 4.2 m/s.
- Racket rotation during the first contiguous contact episode: **0.54-0.81 deg**.
- Plate tilt at the measured pose: **1.45 deg**.

### 3.5 Horizontal velocity change

Incoming tangential speed is set at release; the release point is offset so the ball lands
near the plate centre. The arm is re-settled before every trial. 25 releases per band,
random heading.

| incoming `v_xy` | `|dv_xy|` median | IQR | outgoing spin | `e_h` |
| --- | ---: | --- | ---: | ---: |
| 0.00 m/s, vertical | 0.161 | - | 0.84 rad/s | 0.554 |
| 0.02-0.15 m/s | 0.159 | 0.156-0.163 | 0.90 | 0.554 |
| 0.02-0.36 m/s | 0.162 | 0.155-0.175 | 1.01 | 0.554 |
| robot, held releases | **0.355** | 0.261-0.387 | not measured | 0.45 |

`dv_xy` is the norm of the velocity change, not the outgoing speed. A constant vector change
leaves the outgoing speed input-sensitive, so the table shows only that **these two sampled
bands have similar median change magnitudes**. A constant impulse magnitude would also arise
from sliding friction. The robot number is a median over hand releases that arrive with
0.02-0.36 m/s of tangential motion and scatter in heading, so this is not yet a like-for-like
comparison either.

## 4. What the measurement supports

### 4.1 Keep the contact as the control

The isolated bed fixture reproduces the measured free-racket restitution: 0.88-0.92 against
0.87 measured. That match is the reason to keep `BED_SOLREF` as the control.

It does not prove that every held-contact condition is correct. The held case is measured
only in the simulation so far.

**Decision: do not change `BED_SOLREF`.**

### 4.2 No compliance element on this evidence

The held comparison differs by about 0.09-0.10 of `e_h`, and the wrist-gain intervention
moves 0.086-0.099. The arm's impedance is therefore the same order as the difference. But
the comparison is not protocol-matched, and the standing-policy loop is unmeasured.

**Decision: do not add a compliance element.** The numerical gap is not an identified
physical target.

### 4.3 The plate tilt from the hardware chain is unreliable

The chain forward kinematics gave 2.5-5.6 deg. The simulation gives 1.45 deg at the same
joint angles. The measurement owner reports that the resting ball drifts 1-9 mm over
0.3-1.0 s intervals, with 1.3 mm over the longest still run of 1.02 s.

A freely rolling ball on a 4.5 deg slope would accelerate at about 0.5 m/s2, which is 0.16 m
in 0.8 s. That argues against a freely rolling ideal slope. It does not calibrate the plate
to 1 deg: string-bed indentation, rolling resistance, rim restraint, initial velocity and
support motion are not separated, and no independent face measurement exists.

**Decision: do not change the racket mount geometry.** The chain error stays open and is
probably shared with the 50-65 mm plate-height offset.

## 5. What remains unexplained

1. The ~0.09-0.10 difference in `e_h` at a 0.9 m drop. Candidates: the standing policy's
   loop, the string bed's real dissipation, the servo response, spin, and the arm's passive
   structure. None is separated.
2. The decay. The simulation keeps 0.49-0.60 at 0.09-0.26 m drops; the robot's secondary
   bounces median 0.28, but they are confounded by spin and approach direction.
3. The horizontal velocity change: 0.159-0.162 against 0.355 m/s. Untested: bed friction
   and spin transfer, string-bed compliance, the servo loop.
4. Spin. The simulation produces 0.84-1.01 rad/s. The robot value is unknown, because the
   measurement did not track spin.

## 6. Next steps

**Short verdict:** the revised fixture predicts lower rebound than the isolated bed, but the
hardware comparison is not yet protocol-matched. Keep the contact and arm models unchanged.
Validate the paired measurement definitions, then replay the recorded commands. Use residual
trajectory and recoil errors to select the next identification test. A sole contact cause is
not established.

In order:

1. **Pair the measurement definitions.** Pin the hardware rows per bounce with the same
   apex convention, the same smoothing and the same acceptance gates. Until then no gap can
   be quoted as physical.
2. **Replay the recorded 50 Hz controller targets** instead of the compensated hold. This
   matches commands; it does not reproduce the closed-loop response to changed states.
   Support and base motion, actuator response, geometry, release and spin differences remain.
   A joint-trace cross-correlation can absorb physical response lag, so it is not an
   independent clock calibration.
3. **Select the next identification test from the residual trajectory and recoil errors.**
   These data do not establish joint friction ahead of stiffness, or the reverse.

## 7. Limitations

- Constant compensated targets, not the standing policy. The residual validates position,
  not dynamic impedance.
- One pose, one racket, one ball, one day of robot data.
- Spin is measured in the simulation and not on the robot.
- The racket mount geometry and the bed mesh are assumed correct.
- The bed reference is a transient 20 mm-drop minimum, 14.9 mm below the unloaded face.
- The plate tilt of 1.45 deg is a model value; the hardware value is disputed and has no
  independent face measurement.
- The first-bounce estimator uses the known release height. It is declared because the
  measurement pipeline cannot detect a maximum before a release that has no arc.
- An unwelded fixture was attempted first, with the base free and the robot standing on the
  floor as it does on hardware. A constant-target PD hold is not a balancing controller and
  the robot falls during the first second (base 0.6698 -> 0.0327 m). That fixture needs the
  carry-velocity policy checkpoint; the welded hold is used here instead, and its cost is the
  first bullet above.
- No training result, no export, no deployment claim.

## 8. Reproduction

```bash
cd /home/agiuser/projects/mjlab

# first bounce at four releases, raw and pipeline estimates, with and without the
# wrist-gain intervention
.venv/bin/python scripts/experiments/x2_tennis_drop/drop.py 0.9 0.7 0.5 0.3

# horizontal velocity change, with and without incoming tangential motion
.venv/bin/python scripts/experiments/x2_tennis_drop/tangential.py
```

`model.py` applies `cfg.sim.mujoco` to the compiled model. Keep that call: without it the
fixture measures a different integrator.

## 9. Related records

- `x2_docker/docs/reports/REPORT-racket-contact-restitution.md` — the robot measurement.
- `docs/design/x2_tennis_arm_compliance.md` — the competing design. Its own ordering puts
  matched drops first and passive modes behind evidence of unresolved face deformation. This
  report supplies the first matched-drop attempt and does not change that ordering.
- `tests/test_x2_tennis_ball_task.py` — the isolated bed regression (0.88-0.92).

## 10. Defects from both reviews, and the answer

| defect | answer |
| --- | --- |
| F1 — compensated hold is not a substitute for the policy | wording removed; the residual is a position check; the policy confound is step 2 of section 6 |
| F2 — the fixture did not apply the task's options | `cfg.sim.mujoco.apply(model)` added; the solver's own effect is now measured at 0.010, and the old 0.038 attribution is withdrawn |
| F3 — analysis differed from the owner's; boundary maximum | first bounce uses the known release height; later bounces exclude three samples at each end; gates applied; **raw and pipeline estimates are now reported separately** |
| F4 — no per-release settle | the arm is re-settled before every release in `drop.py` and before every trial in `tangential.py` |
| F5 — robot rows did not support the claimed ratios | the "5 mm" and "7 cm" claims are withdrawn; apex conventions are named per table; the ratio of apparent retention is labelled unmatched |
| F6 — the "upper bound" | deleted; the effect is reported as a gain sensitivity of 0.086-0.099 on the vertical rebound, with no energy partition implied |
| F7 — `tangential.py` could not run | imports fixed; `ruff --select F821,F401` passes |
| F8 — `|dv_xy|` is a vector-norm change | the claim is narrowed to the sampled bands; the constant-impulse alternative is stated |
| F9 — contact motion spanned several bounces | measured over the first contiguous episode: 0.54-0.81 deg |
| F10 — a resting ball does not calibrate tilt | narrowed to an argument against a freely rolling ideal slope; no independent face measurement exists |
| F11 — the prediction overstated causality | section 6 states the order of work without causal claims |
| F12 — the design's ranking | left unchanged; this report feeds its first gate |
| R2-1 — the 0.038 solver attribution | withdrawn; measured as 0.010 under a fixed protocol |
| R2-2 — raw and pipeline minima differ | both reported; the pipeline estimate is the comparable one |
| R2-3 — the third decay row belonged to another release | decay table rebuilt with release and bounce identifiers |
| R2-4 — the residual was measured after a probe | corrected: 4.8e-10 rad after the settle, 0.012 rad after the probe |
| R2-5 — "only difference left" overclaim | deleted from section 6 |
| R2-6 — friction ranked ahead of stiffness | removed; the test is selected from residual and recoil evidence |
