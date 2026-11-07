# S10 second half: post-loss exposure, loss cost and the mixture

Date: 2026-11-07 (+08).
Status: **approved for implementation.** No training run is authorized by this document.

This document adds the §4.2, §4.3 and §5 halves of the
[mixed safety curriculum](x2_tennis_mixed_safety_curriculum.md) to stage **S10**, which
already carries the §4.1 measurement gate ([first half](x2_tennis_visibility_plan.md)).

S10 therefore stops being a one-delta rung. It becomes: **S9c plus** the visibility gate,
**plus** the mixture, **plus** the loss-cost and termination changes. A pure gate effect is
no longer measurable from S10 alone. That is the operator's decision and is recorded here
so nobody reads an S10 result as a gate-only result.

**Parent: S9c, `model_34498.pt`** (operator decision, 2026-11-07 +08, after evaluating both
candidates). S9c beats S9b arm 1 on the axes that matter here: impacts 25.9 / 28.6 / 36.3
against 21.9 / 22.9 / 28.6, alignment 38.8-44.0 mm against 46.9-50.5 mm, and a normalised
pose error of 2.20 / 4.40 / 6.69 against 57.1 / 79.7 / 94.2. S10 keeps
`pose_quadratic`: that term is what untwisted the posture, and resuming an S9c checkpoint
without it would hand the twist back.

## 1. Decisions

| item | decision | reason |
|---|---|---|
| episode budget | **30 s** for S10 only | with a loss induced at 5–20 s, at least 10 s of standing exposure remains |
| time-out bootstrapping | unchanged: `is_finite_horizon = False`, so a time-out bootstraps with `V(s_T)` and a real termination does not | a truncated episode must not be priced as a fall |
| case shares | **0.70 live, 0.075 ground, 0.075 never-seen, 0.15 live + induced loss** | §5, with the 15 % waiting group split evenly and logged separately |
| induced loss | horizontal velocity impulse of **2.0 m/s** in a random direction, once per episode, at `t ~ U[5, 20) s` | bounded, physical, no teleport; the ball leaves racket reach in 0.2–0.3 s |
| ground ball | resting on the floor **0.55 m in front** of the pelvis origin, lateral ±0.25 m | reachable only by a deep bend, which is the "do not chase" case |
| never-seen ball | resting on the floor **1.2 m behind** the pelvis origin | outside the 66° cone and beyond the 1.0 m range, so the gate admits nothing |
| per-step lost cost | **removed** | §4.3; a per-step charge plus a termination that ends it is the defect measured earlier |
| one-shot loss charge | implemented as `ball_lost_event`, weight **0.0** by default | §4.3 allows it, does not require it; a live ball that becomes lost is charged once, ground and never-seen starts never are |
| ball rewards for dead cases | **masked to zero** | §4.3: do not reward pursuit of an unreachable ball through FOV or alignment terms |
| ball-caused terminations | `ball_no_bounce` and `ball_not_lifted` **removed**; `held_ball_contact` **kept, masked to live** | §4.2: they must not discard the standing interval; the operator-held restriction still applies where a hold phase exists |
| parent | **S9c `model_34498.pt`**, keeping `pose_quadratic` | best verified juggling checkpoint, and the term that fixed the posture |
| unchanged | every other reward weight, the physics, the randomisation family, the actor width 112 | one change at a time after the contract |

The `+8.3 %` range bias and the 4.8–5.2 % `ambiguous` dropout stay out, as decided for the
first half.

## 2. Case model

One per-episode integer is drawn at reset and stored on the `reset_ball` event instance:

| case | meaning | ball placement | hold |
|---|---|---|---|
| `LIVE = 0` | normal juggle | S9b: 0.10–0.40 m above the plate, lateral ±0.02, offset 0.05 m/s | yes, 0–6 s delay |
| `GROUND = 1` | ball already on the floor | floor, 0.55 m ahead, lateral ±0.25 | no |
| `NEVER_SEEN = 2` | no valid measurement since engagement | floor, 1.2 m behind | no |
| `INDUCED = 3` | live juggle, then a forced loss | as `LIVE` | yes, then the impulse |

A single helper reads the mask so that the hold, the rewards and the terminations cannot
disagree about which case an environment is in. The mask is read from the event instance,
not recomputed.

## 3. Work items

### D1. Case state and placement (`mdp/events.py`, new `mdp/ball_cases.py`)

- `mdp/ball_cases.py`: the four case constants, and `case_of(env, event_name)` returning the
  per-environment tensor from the event instance.
- `reset_ball` gains `case_shares` and `loss_time_range_s`, stores `self.case` and
  `self.loss_at_s`, and places the ball per the table above. The live branch keeps the
  existing sampling order, so a live-only run draws the same numbers as before.
- `hold_ball` skips the dead cases: the ball is released immediately and `held` is false.

### D2. Induced loss (`mdp/events.py`)

New step event `induce_loss`, class-based, with per-environment `applied` state: when
`case == INDUCED` and the episode time reaches `loss_at_s` and no impulse has been applied,
write the ball root state with a horizontal velocity of `speed_mps` in a random direction
and mark it applied. Reset clears the flag. It never touches a non-induced environment.

### D3. Terminations

Remove `ball_no_bounce` and `ball_not_lifted` from the S10 config. `held_ball_contact` stays
and returns false for `GROUND` and `NEVER_SEEN`.

### D4. Rewards

- `ball_lost_penalty` removed; `ball_lost_event` added with weight
  `JUGGLE_BALL_LOST_EVENT_WEIGHT = 0.0`. It fires **once** per episode, only for `LIVE` or
  `INDUCED`, on the first step the ball is below 0.5 m, and never again while it stays lost.
- `ball_energy_after_release`, `ball_centred`, `ball_in_fov` and `racket_under_ball` are
  multiplied by the live mask, so a dead case earns nothing from the ball.
- Everything else is unchanged, including `body_ang_vel` at −0.5.

### D5. Budget and metrics

- `episode_length_s = 30.0` for the S10 stage.
- Metrics `ball_case_live`, `ball_case_ground`, `ball_case_never_seen` and
  `ball_case_induced`, each `reduce="mean"`, so the realized episode-start shares are
  logged next to the declared ones.

### D6. Stage

`s10-visible-ball` keeps `visibility_gate: True` and gains `safety_mix: True`, which applies
D3–D5. D1 and D2 are wired by the stage, not by the nominal factory, so every earlier stage
is untouched.

## 4. Acceptance, as a probe script

`scripts/experiments/x2_tennis_visibility/check_s10_mix.py`, same budget rule: at most 8
environments and a bounded number of control steps, runtime under two minutes.

| case | expected |
|---|---|
| realized shares | within ±5 percentage points of 0.70 / 0.075 / 0.075 / 0.15 over enough resets |
| episode budget | the ceiling is 1500 control steps (30 s); a live episode reaches it when it does not fall |
| induced loss | exactly one impulse per induced episode, at 5–20 s, with at least 10 s of episode left |
| no ball terminations | `ball_no_bounce` and `ball_not_lifted` are absent from the config |
| held-contact mask | never true for `GROUND` or `NEVER_SEEN` |
| dead-case rewards | zero ball reward over a dead episode's lifetime; non-zero for a live one |
| never-seen stream | the actor's 12 numbers stay zero for the whole episode |
| loss charge | `ball_lost_event` fires at most once per live episode and never for a dead start |

## 5. Out of scope

- Numeric acceptance limits for a training run (§8): still to be frozen by the operator.
- A racket-workspace metric or reward (§7).
- Any run: this document authorises code and probes only.
