# S11: a tilted start

Date: 2026-11-07 (+08).
Status: **implemented, not trained.** No run is authorized by this document.

S11 is S10 plus one condition: the episode starts with the robot **already leaning**.
Everything else is S10's — the visibility gate, the 70/7.5/7.5/15 mixture, the 30 s
budget, `pose_quadratic`, the loss-cost change and the observation contract.

## 1. Why

The operator ran S10 on the real robot: the dangerous swing is gone, but the robot cannot
come back from a tilt. The cause is in the configuration, not in the policy. Every rung
before S11 resets the robot upright in the carry pose with zero velocity
(`reset_base` has no `roll` or `pitch` range at all, and `reset_robot_juggle_joints` is a
zero offset), so a tilt could only appear transiently from a push, and `fell_over` ends the
episode at 70 degrees (`limit_angle = 1.2217`). A leaning posture was therefore never a
state the policy could act from for more than a moment.

## 2. The condition

Two reset terms are inserted **after** `reset_robot_juggle_joints` and **before**
`reset_ball`, so the ball is still placed relative to the resulting pose:

| term | joints | offset | distribution |
|---|---|---|---|
| `reset_tilt_body` | waist yaw/roll/pitch, both hip pitch/roll/yaw, both knees | **±15°** | uniform |
| `reset_tilt_ankle` | both ankle pitch and roll | **±30°** | uniform |

`reset_joints_by_offset` clamps each sample to the joint's soft limits, so the realised
distribution is truncated by the machine. Measured over 1280 resets:

| group | p50 | p95 | max |
|---|---|---|---|
| waist / hip / knee | 6.77° | 14.07° | **15.00°** |
| ankle | 13.51° | 27.02° | **29.99°** |

Two truncations matter and are expected, not defects:

- **Knee** cannot take the full negative offset: its soft minimum is +6.9°, so a −15°
  draw lands at −10.3° from the carry pose and stops there.
- **Ankle roll** saturates at **±13.5°**, its soft limit. Only ankle **pitch** reaches the
  full ±30°. If 30° of lateral ankle tilt is the intent, the joint range is the
  constraint, not the constant.

## 3. What is preserved

Rewards, terminations, metrics, the case mixture, the observation parameters, the
30 s budget and the actor width are byte-identical to S10. The stage test asserts it.

## 4. Acceptance to define before a run

- The robot starts leaning in **every** episode; there is no upright-start subgroup.
- Episode length does not collapse: most starts fall before the effect is learned, and the
  curve must recover.
- Juggling is measured against S10 (impacts 28.9 / 40.1 / 36.3, alignment 35.3 / 43.6 /
  36.2 mm) and posture against S10 (0.66 / 0.71 / 6.29).
- A hardware check remains separate and unauthorised.
