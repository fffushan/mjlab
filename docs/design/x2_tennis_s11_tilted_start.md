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
| `reset_tilt_body` | waist yaw/roll/pitch, both hip pitch/roll/yaw, both knees, both ankle rolls | **±5°** | uniform |
| `reset_tilt_ankle_pitch` | both ankle pitches | **±15°** | uniform |

Ankle pitch keeps three times the body offset because it is the joint that turns a joint
offset into a lean: rotating the ankle moves the whole body over the foot, whereas a hip or
knee offset mostly re-shapes the leg.

`reset_joints_by_offset` clamps each sample to the joint's soft limits. Measured over 1280
resets:

| group | p50 | p95 | max |
|---|---|---|---|
| waist, hip, knee, ankle roll (13 joints) | 2.51° | 4.72° | **5.00°** |
| ankle pitch (2 joints) | 3.69° | 13.45° | **14.99°** |

**One truncation remains, and it is small.** `right_hip_roll_joint` has a soft upper limit of
+4.5° against a 0° carry pose, so the positive half of its draw clamps there: 5 % of resets
land on +4.47° instead of up to +5°. Measured 64 of 1280 resets touch a limit, all from this
joint. At the previous 15°/30° setting two joints truncated badly — the knee could only take
−10.3° because its soft minimum is +6.9°, and ankle roll saturated at ±13.5° — so this
setting is both smaller and cleaner.

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
