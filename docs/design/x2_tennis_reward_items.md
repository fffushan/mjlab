# X2 tennis juggle: reward items, with S9c as the baseline

Status: working list. Written 2026-11-07 (+08).

This document covers **rewards only**. The baseline is stage S9c
(`-Nominal-S9c-PoseQuad`), the last rung trained on a live ball. Observation realism,
controller wiring, estimator work and robot preconditions are separate documents and
are not repeated here.

Scope: training and reward design. No hardware action is authorised by this document.

---

## 1. The baseline: all 21 reward terms of S9c

`raw·weight` is the mean per-step contribution. **Live** is an S8 juggle episode with
the S8 policy (16 envs, 300 steps). **Dead** is the S9c task with the ball placed below
the plate and the S9b2 policy (16 envs, 300 steps). Both probes read the reward
manager's own buffer, so the two columns are directly comparable. The table lists every
term the ladder has considered: the 21 in S9c, three proposed, two rejected.

**Status is one of three values.** `S9c` means present in the baseline and not
scheduled for change. `todo` means required work that is not done: a new term, a
re-weighting, a shape change or a pending decision. `removed` means deleted from the
ladder, either as dead weight or as a rejected idea. "(off in S9d)" marks a term that
S9c has and the dead-ball rung S9d drops, which is a per-rung removal, not a decision.

| # | term | weight | function / shape | live | dead | status |
|---|---|---|---|---|---|---|
| 1 | `ball_energy` | +2.0 | post-release only, apex 0.40 m, σ 1.5 | +0.970 | 0.000 | S9c (off in S9d) |
| 2 | `ball_centred` | +0.25 | plate frame, plateau 20 mm, σ 40 mm, sink 5 mm | +0.200 | 0.000 | S9c (off in S9d) |
| 3 | `ball_in_fov` | +0.2 | frustum step, plateau 0.6, σ 0.25, half-angles 1.49 × 1.12 rad | +0.199 | +0.116 | **todo** RW4 |
| 4 | `racket_under_ball` | +0.4 | wait phase only, plateau 10 mm, σ 30 mm, level_power 2 | +0.040 | +0.001 | S9c (off in S9d) |
| 5 | `ball_lost_penalty` | −4.0 | per step while ball z ≤ 0.5 m, 25-step grace | 0.000 | −0.298 | **todo** RW3 |
| 6 | `racket_lift` | −0.0 | deadband 30 mm above 1.1194 m, weight zero | 0.000 | 0.000 | **removed** RW5 |
| 7 | `pose` | +1.0 | `exp(−x)`, `x = mean((q−q*)/0.05)²` | +0.767 | +0.029 | **todo** RW1 |
| 8 | `pose_quadratic` | −0.04 | `x`, same reference and tolerance, **S9c addition** | 0.000 | −2.522 | S9c |
| 9 | `upright` | +1.0 | σ 0.447 (63° tilt) | +0.995 | +0.805 | S9c |
| 10 | `body_ang_vel` | −0.5 | torso rate, raised from −0.05 in S9a | −0.002 | −0.722 | S9c |
| 11 | `angular_momentum` | −0.02 | root angular momentum | −0.006 | −0.081 | S9c |
| 12 | `track_linear_velocity` | +2.0 | σ 0.5 m/s, zero command | +1.987 | +1.629 | S9c (RW7 candidate) |
| 13 | `track_angular_velocity` | +2.0 | σ 0.707 rad/s, zero command | +1.963 | +1.069 | S9c (RW7 candidate) |
| 14 | `dof_pos_limits` | −1.0 | joint limit proximity | 0.000 | −0.073 | S9c |
| 15 | `action_rate_l2` | −0.1 | action smoothness | 0.000 | −0.294 | S9c |
| 16 | `self_collisions` | −1.0 | force > 10 N | 0.000 | 0.000 | S9c (inert) |
| 17 | `foot_clearance` | −2.0 | target 0.1 m, on at zero command | 0.000 | 0.000 | S9c (inert, RW6) |
| 18 | `foot_swing_height` | −0.25 | target 0.1 m | 0.000 | 0.000 | S9c (inert, RW6) |
| 19 | `foot_slip` | −0.1 | | 0.000 | 0.000 | S9c (inert, RW6) |
| 20 | `air_time` | 0.0 | feet air time, weight zero | 0.000 | 0.000 | **removed** RW5 |
| 21 | `soft_landing` | −1e−05 | impact speed, weight below any effect | 0.000 | 0.000 | **removed** RW5 |
| | **total, S9c** | | | **+7.018** | **−0.342** | |
| | **logged per-second** | | `÷20`, logging only | **+0.351** | **−0.017** | |

### Proposed terms, not yet built

| # | term | weight | function / shape | live | dead | status |
|---|---|---|---|---|---|---|
| 22 | `held_ball_contact_penalty` | ~0.1–0.3 | price the held contact, or delete the termination instead | — | — | **todo** RW2 |
| 23 | `upright_after_loss` | unset | pays for staying up once the ball is lost, cannot be ended by a fall | — | — | **todo** RW8 |
| 24 | four ball weights ×2 | experiment | `ball_energy`, `ball_centred`, `ball_in_fov`, `racket_under_ball`, nothing else | — | — | **todo** RW7 |

### Rejected, do not re-propose

| # | term | weight | why it was rejected | live | dead | status |
|---|---|---|---|---|---|---|
| 25 | `far_field_racket_pull` | — | S8 closed a mean 145–156 mm placement to 25–30 mm alignment without it | — | — | **removed** |
| 26 | `joint_yaw_rate_penalty` | — | the S9a answer was one weight (`body_ang_vel`), not new machinery | — | — | **removed** |

Two facts from this table drive most of the items below.

1. **`pose` and `pose_quadratic` are two functions of one number.** `pose` is `exp(−x)`
   and the quadratic is `x`, with the same reference and the same 0.05 rad tolerance.
   At the carry pose the pair pays +1.0/step and charges 0. At the twisted posture
   `x ≈ 63`, so `pose` pays +0.029 and the quadratic charges 2.52. The pair is a bonus
   near the pose plus a pull far from it.
2. **The tracking pair is 38 % of the positive intake, the four ball terms 20 %.**
   Most of what the policy earns, it earns for standing still.

### Reward arithmetic, and the `÷20` trap

Two conventions are in play and they must not be mixed.

- **What PPO receives.** `RewardManager.compute` returns `Σ(raw · weight · dt)` with
  `dt = 0.02` and **no** division by the episode length. At 50 Hz the per-second value
  is therefore `Σ(raw · weight)`, and a weight of −4.0 at a raw value of 1.0 is
  **−4.0 per second of return**.
- **What the log prints.** `RewardManager.reset` writes
  `Episode_Reward/<term> = Σ(raw · weight · dt) / max_episode_length_s`, which for a
  full-length episode is the per-second average in raw-weight units. The `÷20` is a
  **logging normalisation only**. It does not scale the policy's reward.

Ratios between terms are identical in both conventions, so a share of intake can be
read from either. Absolute values cannot. The two measured columns in the table are
`raw · weight` per step, which is the PPO scale divided by `dt`, so the live-juggle
intake of **+7.018 per step is +7.018 per second of PPO return**, not 0.351.


---

## 2. Items

Each item states the observation, the evidence, the change and the acceptance test.

### RW1. `pose` and `pose_quadratic` overlap

- **Observation:** both terms read `x = mean_j((q_j − q*_j)/0.05)²` against the same
  `default_joint_pos`. One is `exp(−x)` with weight +1.0, the other `x` with −0.04.
- **Evidence:** measured 88× apart at the twisted posture (+0.029 against −2.522) and
  equal and opposite at the carry pose (+1.0 against 0).
- **Effect:** near the pose the pair pays twice for the same property; away from it only
  the quadratic works. That may be intended, but it is currently undocumented and
  nobody chose it.
- **Change:** pick one of three. **A** keep both and document the intent ("bonus near,
  pull far"). **B** drop `pose` from any rung that carries the quadratic, so the shape
  is one function of one quantity. **C** give the quadratic a tighter `std` so the two
  do not overlap.
- **Acceptance:** the sum of the two terms is monotone decreasing in `x` over
  `x ∈ [0, 100]`, with no flat or rising region. This is a numeric check, not a
  simulation.

### RW2. `held_ball_contact` has no cost

- **Observation:** it is a termination with no paired reward.
- **Evidence:** fires on 18–35 % of iterations in S7b, 30–44 % in S8, 0 % in S7a
  (exactly zero across 1304/1298/1324 logged iterations).
- **Effect:** the termination is **not free**. Ending the episode removes the remaining
  continuation return, measured at **+7.018 per step** in a live juggle, so a violation
  near the start of an episode costs most of the episode. The defect is the **shape** of
  that price, not its absence:
  - it is all-or-nothing, so the marginal cost of one more held contact is either zero
    or the whole rest of the episode, with nothing in between;
  - it discards the juggle it was learning from;
  - it inverts sign whenever the continuation return is negative (see RW3), where a
    termination becomes a reward rather than a cost.
- **Change:** price it marginally, or remove the termination and accept the behaviour.
- **Acceptance:** held contact below 5 % of iterations with no loss of flight time, or
  the termination removed and the rate reported as accepted.

### RW3. `ball_lost_penalty` makes falling cheaper than standing

- **Observation:** −4.0 per step while the ball is lost, after a 25-step grace.
- **Evidence:** measured −0.298/step with a dead ball, 3.9 % of the total magnitude.
- **Effect, first on the sign of the continuation.** Both signs are measured, and the
  negative one has a cause:
  - **Live juggle: +7.018 per step.** Standing with a lost ball is worth more than
    ending the episode, so the penalty does not reward falling.
  - **Dead ball, S9b2 policy: −0.342 per step.** This is **not** evidence that standing
    is intrinsically unprofitable. `pose_quadratic` alone contributes −2.522 on that
    policy's twisted posture. Remove the twist (`pose_quadratic → 0`, `pose → +1.0`) and
    the same reward set pays **+3.151 per step**. The negative total is a property of the
    twist, not of the dead ball.
- **Decision rule:** falling is preferred **iff the net continuation rate is negative**.
  In PPO units the standing reward is `dt · r_net` per step, so its discounted sum over
  the remaining `N` steps is
  `dt · r_net · (1 − γ^N) / (1 − γ)`. At `γ = 0.99`, `dt = 0.02` that factor is
  **2.0**, and a fall yields 0 for the same future. The discount multiplies both
  branches identically, so **it cancels and cannot flip the sign**. An earlier version
  of this document claimed the discount made falling win even when standing was
  positive. That was wrong and is retracted: the missing `dt` made the discounted sum
  100·`r` instead of 2·`r`, and the comparison never depended on the discount at all.

**Scope of the rule and the table: both are a toy model.** They assume a constant
per-step rate for the rest of the episode, a termination that returns exactly zero, no
terminal-step cost, and no dependence of future rewards on which action is taken. Real
settling and falling actions change the reward rate, the termination timing and the
value at the final step, and the counterfactual rows still carry the other terms of a
**twisted, transient** rollout. The last column therefore illustrates the **sign of a
net rate**. It is not a policy prediction and it is not causal evidence. The acceptance
test is the full discounted trajectory comparison, not this table.

| case | `r_net` per step | discounted future over 800 steps | toy-model preference |
|---|---|---|---|
| live juggle | +7.018 | +14.03 | stay |
| dead ball, S9b2 policy | −0.342 | −0.68 | **fall** |
| dead ball, untwisted **arithmetic** (counterfactual) | +3.151 | +6.30 | stay |
| dead ball, untwisted, **full −4.0 charge** | −0.551 | −1.10 | **fall** |

- **On the third row:** +3.151 is not a measured quiet-stand return. It is the probe's
  total with two terms replaced (`pose_quadratic → 0`, `pose → +1.0`), and it still
  contains the probe's **own measured** loss charge of −0.298 per step. Subtracting a
  further −4.0 double-counts. Replacing the measured −0.298 with the full per-step
  charge of −4.0 gives −0.551, which is the relevant comparison and is still negative.
- **Consequence:** the defect is that a per-step charge can exceed the gross standing
  reward for the whole tail, so the tail's **net** rate is negative and a fall ends it.
  The sign of the net rate decides, and the charge is what makes it negative.
- **Change:** charge the loss **once per episode**, so ending an episode cannot avoid
  it, or pay a survival bonus that dominates. Never combine a per-step charge with a
  terminal condition that ends it.
- **Acceptance:** `fell_over` after a loss does not exceed `fell_over` before it, tested
  in a rung whose tail net rate is negative and in one whose tail net rate is positive,
  comparing the **full discounted settling trajectory** against the **full discounted
  fall trajectory** rather than a single step. The rate sign is a hypothesis about the
  direction; only the trajectory comparison is evidence.

### RW4. `ball_in_fov` pays for watching

- **Observation:** +0.199/step in a live juggle, **+0.116/step with a dead ball**.
- **Effect:** the term pays the robot to keep the ball inside the camera frustum. With
  an unreachable ball that is a payment for watching it fall. The camera is rigid on
  `head_pitch_link`, so the term also acts as a weak head and torso orientation reward.
- **Change:** gate it on the ball being above the plate, or lower the weight, or hold it
  out of dead-ball rungs.
- **Acceptance:** below 1 % of the positive intake whenever the ball is below the plate.

### RW5. Three terms are dead weight

- **Observation:** `racket_lift` weight **−0.0**, `air_time` weight **0.0**,
  `soft_landing` weight **−1e−05**. All three measure 0.0000 in both probes.
- **Effect:** a reward table with inert rows invites wrong inferences. An earlier
  reading of `ball_no_bounce` was wrong for exactly this reason: the term existed,
  logged 0.0000, and told us nothing.
- **Change:** remove them, or give them a weight with a purpose.
- **Acceptance:** every term has `|weight| ≥ 1e−3` and a non-zero contribution in at
  least one probe.

### RW6. The gait terms are active in a standing task — measured inert

- **Observation:** `foot_clearance` (−2.0), `foot_swing_height` (−0.25) and `foot_slip`
  (−0.1) carry `command_threshold` gates of 0.05–0.5 m/s, so with the zero `twist`
  command they are **on**.
- **Evidence:** all three measure 0.0000 in both probes, because the feet do not leave
  the ground and do not slip.
- **Change:** none. Record the measurement so nobody "fixes" the gating and changes the
  task by accident.
- **Acceptance:** a probe still reads 0.0000 after any change to the command generator.

### RW7. The term balance may be the binding constraint under full DR

- **Observation:** `track_linear_velocity` + `track_angular_velocity` = 56 % of the
  positive sum; the four ball terms together are 20 %.
- **Evidence:** S9b2, trained with the full randomisation set, reaches 29.2–39.0
  impacts and 46.9–50.5 mm alignment, against S9a's 35–36 impacts and 20–27 mm with no
  randomisation. The gap persisted across three 5000-iteration arms.
- **Effect:** if the ball side is under-incentivised, more iterations cannot close that
  gap; only the weights can.
- **Change:** not recommended yet. If tried, one rung with the four ball weights
  doubled, nothing else.
- **Acceptance:** if impacts move by less than 10 %, the balance is not the constraint
  and the rung is reverted without further tuning.

### RW8. There is no term that expresses surrender

- **Observation:** the only term that reacts to an unrecoverable ball is
  `ball_lost_penalty`, and it is a per-step charge (RW3).
- **Effect:** the only way to train "give up and stay safe" is to delete every ball
  term, which is what S9d does. That also deletes the gradient that keeps the racket
  where a ball would arrive.
- **Change:** if mixed rungs are ever wanted, add a term that pays for staying upright
  once the ball is lost, and make sure a fall cannot end it (RW3).
- **Acceptance:** with a dead ball the total reward is positive and increasing in
  uprightness.

### RW9. `ball_energy` is a tight band in energy, and the units are easy to misread

- **Observation:** the term is `exp(−((E − E*)/σ)²)` with `E` in **J/kg** and
  `σ = 1.5 J/kg`. A height error of 1 m is 9.81 J/kg, which is 6.5σ, so the term is
  **saturated**, not linear, outside roughly ±0.2 m.
- **Consequence:** `σ = 1.5 J/kg` corresponds to a height tolerance of
  `1.5 / 9.81 = **0.153 m**`, not 1.5 m. The term is a **tight** band of about ±0.15 m
  of apex height around 0.40 m. Reading the raw σ as metres overstates the tolerance by
  a factor of 9.81.
- **Evidence:** +0.970/step in a live juggle (raw value about 0.49 of a maximum 2.0),
  0.000 with a dead ball.
- **Effect on the overhead problem:** the term constrains the **ball's apex**, not the
  racket's workspace. A racket held high with a small hop satisfies it. It does not close
  the short-hop exploit.
- **Change:** record the height-equivalent tolerance in the stage document, and stop
  citing `σ` as if it were metres.
- **Acceptance:** the term's value at 0.2, 0.4 and 0.6 m of apex is recorded, and its
  height tolerance is quoted as 0.153 m.

---

## 3. Considered and rejected

Listed so they are not re-proposed. Each was rejected on a measurement, not on taste.

- **A linear far-field pull on the racket.** S8 closed a mean 145–156 mm placement to
  25–30 mm alignment without it. Arriving is a precondition for every other reward, so
  the gradient already exists.
- **A torso-contact termination.** The velocity task answers a fall at body level:
  `fell_over` at 70° tilt plus `upright`, `body_ang_vel` and `angular_momentum`.
- **A joint yaw-rate penalty.** The same reason. The missing piece in S9a was one
  weight, not new machinery.
- **Re-weighting `upright`.** It is already the third-largest positive term at
  +0.995/step. The posture problem was saturation in `pose`, not a weak `upright`.

---

## 4. Order of work

1. **RW1** — a numeric check, no training. Decide the `pose` / `pose_quadratic` pair
   before S9e, because S9e is where the pair meets a live ball again.
2. **RW2 and RW3** — both are defects in the current set. RW3 already forced a change
   in S9d; RW2 is still open.
3. **RW4, RW5, RW6** — small, and they make the table readable.
4. **RW7** — only if the DR gap survives S9e.
5. **RW8** — only if mixed dead/live rungs are wanted.
6. **RW9** — documentation only.

---

## 5. Acceptance for the list

- No two terms measure the same quantity without a stated reason.
- No term has weight zero without being marked inert by measurement.
- A termination is either priced or absent.
- No term can be improved by ending the episode early.
- Every claim in this document cites a probe, a log or an artifact that can be re-run.
