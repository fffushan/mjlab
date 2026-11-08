# The juggle stage ladder, S1 to S11

Date: 2026-11-08 (+08).
Status: **reference document.** It records what each rung changed, what it measured, and
which rungs were adopted. It authorizes no run.

This is the register of the training ladder behind the tennis juggle task
(`/home/agiuser/projects/mjlab`). Each rung answers one question about a condition the
deployed chain imposes on the policy. Read a rung's section before you resume it, change
it, or quote a number from it.

## 1. What a rung is

A rung is **one condition added to a parent rung**. The policy is fine-tuned from the
parent's checkpoint. The rung exists to answer a yes/no question: can the policy hold the
juggle under this condition, and at what cost.

Three properties of the implementation are load-bearing:

* **Each row is self-contained.** A row of `STAGE_DELTAS` restates the whole condition,
  not only its own increment. The factory starts from the nominal pilot (S0) and applies
  the row. A later rung therefore repeats every earlier key and adds one. Read a row as
  "the condition this policy trains in", never as "the change since the previous row".
* **A stage never inherits a sibling's condition by accident.** `s3b-80-140ms` and
  `s3c-0-140ms` are siblings of `s3-40-140ms`; all three resume S2. `s4b-stalls` is a
  sibling of `s4-stalls`. `s11-tilted-start` resumes S10.
* **Training is resumable, not sequential in one process.** Each rung is its own run and
  its own log. A rung's checkpoint lives under
  `<log-root>/<experiment>/<run-name>/model_<iteration>.pt`.

## 2. How to run one

Stage keys are the keys of `STAGE_DELTAS` in
`src/mjlab/tasks/velocity/config/agibot_x2/tennis_juggle_stage_env_cfg.py`. Each key has a
registered task id with the suffix `-<STAGE NAME>`, built by
`agibot_x2_tennis_juggle_stage_env_cfg(stage=<key>, play=False)`.

Resume mechanics, which have cost time to rediscover:

* `--agent.max-iterations` is an **increment** when resuming, not an absolute target.
  Resuming `model_36497.pt` with `1000` stops at 37497.
* `--agent.resume` needs the explicit value `True`. A bare flag is not enough.
* The parent checkpoint must sit at
  `<log-root>/<experiment>/<load-run>/<load-checkpoint>`. Copy it into the child run's own
  log root; do not point at the parent's tree.
* The final checkpoint lands **one iteration short of the budget**. Always resolve it from
  the filesystem: `ls -1 <run-dir>/model_*.pt | sort -V | tail -1`.
* `play` has no `--env.*` flags. Environment overrides go through `MJLAB_JUGGLE_*`
  variables. `play=True` also removes `push_robot` and the observation corruption, so an
  acceptance cell measured in play mode reports a push-free, corruption-free robot.

## 3. The stage table

| stage key | parent | condition added | verdict |
|---|---|---|---|
| `s1-15hz` | nominal pilot | latency 0–40 ms, 15 Hz stream | adopted as the latency baseline |
| `s2-80-100ms` | S1 | latency band 80–100 ms | **superseded by S3**; the band is too narrow |
| `s3-40-140ms` | S2 | band 40–140 ms, redrawn per acquisition | **adopted**: flat 0–140 ms |
| `s3b-80-140ms` | S2 | same width, centre 110 ms | sibling; band centre is not the operative property |
| `s3c-0-140ms` | S2 | whole band, centre 70 ms | sibling; indistinguishable from S3 |
| `s4-stalls` | S3 | stalls: 2 %, 0.1–0.6 s | **rejected**: seed-unstable |
| `s4b-stalls` | S3 | stalls from a recorded clip: 10 %, 0.1–0.2 s | **adopted**; sets estimator requirement R9 |
| `s5-noise` | S4b | per-acquisition noise ±10/10/15 mm, bias 0 | **adopted**: costs nothing inside the band |
| `s6-release` | S5 | release anywhere 0–0.40 m, ±20 mm lateral | **adopted**; passes on three seeds |
| `s7a-delayed-release` | S6 | fixed 0.40 m hold, 0–2 s delay | hold metric was broken; superseded by S7b |
| `s7b-variable-hold` | S6 | hold 0.10–0.40 m, 0–2 s delay | **adopted**; fixes the deployment band at 0–160 ms |
| `s7c-long-hold` | S6 | hold with 0–6 s delay (mean 3.05 s) | adopted as the long-wait rung |
| `s8-offset-hold` | S7c | one-time per-episode ball offset | **adopted**: alignment 25–30 mm |
| `s9a-drop-tail` | S8 | drop the ball-loss termination, price the aftermath | **adopted**: falls 0.1773 → 0.0431 |
| `s9a2-drop-tail` (variant) | S9a | flatten the loss cost 4× → 1.2× | not adopted: the tail stops paying |
| `s9b-sim-to-real` | S9a | the tracking family's full domain randomisation | **adopted** (second arm) |
| `s9c-pose-quad` | S9b2 | `pose_quadratic`, std 0.05, weight −0.04 | **adopted**: posture 57 → 2–7 |
| `s9d-dead-ball` | S9c | ball removed, ball rewards and endings removed | **aborted** before launch |
| `s10-visible-ball` | S9c | visibility gate + 4-way ball-case mixture + 30 s budget | **adopted**: fall share 0.52–0.70 % |
| `s11-tilted-start` | S10 | start 5° body / 15° ankle pitch off the carry pose | **adopted**: adaptation, not recovery |

S0 is the nominal pilot: the ball3 task with exact measurements, no domain randomisation
beyond the pilot's own resets, a 20 s episode and the juggle reward set.

## 4. Stage detail

### S1 — latency 0–40 ms

15 Hz stream, transport latency uniform 0–40 ms. Closed-loop grid: perfect at 40–60 ms
(6754 and 6784 flights, 0 resets), 6004 flights at 80 ms, then a cliff — 61 flights and
1085 resets at 100 ms, 21 flights at 140 ms. Not measured: whether the failure is a
perception limit or a control limit; the later rungs show it is a control limit.

### S2 — latency 80–100 ms

Same rate, band moved to 80–100 ms, the improved-estimator target. Grid: perfect at 80 ms
(6688) and 100 ms (6720), graceful at 120 ms (5283 flights, 32 resets), collapse at 140 ms
(563 flights, 303 resets).

**The zero-latency control is the important result.** At 0 ms the policy collapses: 161
flights and 679 resets across 32 environments, apex sd 23.8 mm, radial p95 0.988. At 20 ms
it is degraded (2759 flights, 64 resets). At 40 ms it is perfect again (6673 flights,
0 resets, 32/32 censored). A policy that read the per-sample age would find 0 ms the
easiest condition it could face. This one found it the hardest, so it had learned a
**fixed lead of about 60–70 ms with ±35 ms tolerance**, not a latency reading.

### S3 — band 40–140 ms

The band is redrawn per acquisition, so the delay changes inside an episode; a fixed lead
cannot cover it and the policy must read the age the observation already carries. Budget
10996 iterations. Grid: **flat from 0 to 140 ms**, 6624–6718 flights, 0 resets, no drops.

### S3b and S3c — the band-centre siblings

S3b (80–140 ms, centre 110) and S3c (0–140 ms, centre 70) resume S2 and run the same
budget as S3 (centre 90). S3c is indistinguishable from S3, and includes 0 ms where the S2
policy died. **The conclusion is that band width, not the band's mean, is the property to
train.** That retired the idea of tuning the estimator to a single latency.

### S4 — stalls, assumed

S3's band plus stalls: probability 0.02, duration 0.1–0.6 s. This models the ball leaving
the field of view for up to 0.74 s of age. **Rejected**: at a clean 0 ms cell the three
seeds returned 97, 6703 and 77 flights. A rung whose seeds disagree by that much does not
support a conclusion about the condition.

### S4b — stalls, measured

The stall model comes from a recorded hardware clip: the ball leaves the camera field when
it is raised high, for about 0.1 s, on about 10 % of frames. Probability 0.10, duration
0.1–0.2 s. Worst age seen by the actor therefore falls from 0.74 s to 0.34 s while the
fraction of time the newest sample is stale rises from about 10 % to about 20 %.

All three seeds passed and closed the loop cleanly: episode length about 1000 steps, about
35 impacts, flight 0.553–0.557 s. This produced the estimator requirement **R9: latency
plus the longest in-view gap must stay at or below 0.34 s.**

### S5 — measurement noise

S4b's whole condition plus per-axis, per-acquisition noise uniform in ±(10, 10, 15) mm,
which is half a ball radius vertically, with **bias fixed at zero**. The three history
samples therefore carry independent error, as a real estimator gives. Ages stay exact;
only positions are wrong.

Result: noise costs nothing inside the trained band. The S5 seed-7 grid gives 6649 / 6615
/ 6622 flights with 0 resets at 0 / 80 / 140 ms, against 6592 / 6623 for the no-noise
controls. At 200 ms it breaks: 4475 flights, 30 resets, 14 of 32 environments censored.
**The age budget is the wall, not the noise.** The zero bias is a stated limit: a bias
direction cannot be measured with the current rig, and a wrongly directed bias would teach
the actor to compensate a calibration error that does not exist.

### S6 — release generalisation

The estimator condition is held fixed and the operator's release is varied: the ball
leaves the hand anywhere from rest on the strings to 0.40 m above them, uniformly, and
within ±20 mm of the plate centre. The target apex stays 0.40 m, so a high release must be
caught and re-bounced and a zero release has no fall energy at all.

All three seeds pass at 80 ms: 6599 / 6621 / 6597 flights, 0 resets, 32/32 censored.

### S7a — fixed hold

Ball held still 0.40 m above the plate and released after a uniform 0–2 s delay. Measured
mean hold 0.979 s. `held_ball_contact` read 0.0000 across every cell, which was a metric
bug rather than a policy result; the fix landed with S7b. S7a also removed S6's height
range, so its difference from S6 mixed the hold with the loss of the range — which is why
S7b exists.

### S7b — variable hold

The height range returns (0.10–0.40 m) and the hold is sampled once per episode by
`reset_ball`; `hold_ball` follows that sample. `held_ball_contact` is intermittent at
18.6–34.5 %, which is expected: only the parts of the range the operator can actually hold
produce contact.

This rung produced the **deployment band**: 0 ms 191.9, 80 ms 206.7, 140 ms 207.8,
**160 ms 207.2, all 32/32**, 180 ms 170.1, 200 ms 103.5. The band is 0–160 ms.

### S7c — long hold

Hold delay 0–6 s, mean measured 3.05 s. Flight 0.610–0.648 s. This is the rung where the
policy has to keep a ball at rest on the strings for seconds and then still lift it.

### S8 — one-time offset

One per-episode ball offset, applied during the hold at 0.05 m/s, so an episode's ball is
already long or short of the plate centre before it is released. Alignment 25–30 mm, which
ruled out a far-field linear reward term: the policy generalises across the offset.

### S9a — the drop tail

The ball-loss termination is removed and the aftermath is priced. Two changes only:
`ball_lost_penalty` at −4.0, and `body_ang_vel` on the torso from −0.05 to −0.5. Falls fall
from 0.1773 to 0.0431. The reasoning that supports this rung is recorded in
`x2_tennis_reward_items.md` (RW1–RW9), including the retraction of the "discounted fall"
argument: a per-step charge plus a terminal condition is a design defect, because falling
ends the charge.

### S9a2 — flattened loss cost

The loss cost is flattened fourfold to 1.2×. The tail stops paying and the behaviour
regresses toward the S9a parent. Not adopted; it is the counter-example that sizes the loss
term.

### S9b — full domain randomisation

The twelve randomisation events the nominal pilot removes are restored from
`make_tracking_env_cfg()`, the set with validated dynamic tennis motions behind it:
`base_com`, `encoder_bias`, `foot_friction`, `push_robot`, `randomize_pd_gains`,
`randomize_inertia`, `randomize_armature`, `randomize_effort_limits`,
`randomize_joint_friction`, `randomize_joint_damping`, `randomize_foot_size`, plus the
wrist effort-limit term.

`push_robot` is part of the set: a root velocity step every 1–3 s with x/y ±0.5 m/s,
z ±0.2 m/s, roll/pitch ±0.52 rad/s, yaw ±0.78 rad/s, per environment.

The first arm was truncated by the wall clock (`exit=124`) at `model_27500.pt`. **S9b2** is
the completed rung at `model_32499.pt`.

The cost is measured, and it is larger than the impact count suggests:

| metric, last 300 iterations | S9a (no DR) | S9b2 (full DR) |
|---|---:|---:|
| `ball_lost_events` per episode | 0.018 / 0.022 / 0.022 | **0.419 / 0.367 / 0.283** |
| `ball_impact_count` | 35.2–35.9 | 34.1 / 31.2 / 42.1 |
| `ball_flight_time` | 0.61–0.66 s | 0.675 / 0.703 / 0.601 s |

Ball losses rise **13 to 23 times**. Impacts do not rise systematically — they spread
across seeds. Alignment is worse by about 10 mm, and falls are 20–50 % lower.

An earlier report of "losses up about fourfold" in this ladder was wrong. The value came
from a table entry, not from the archived logs, and the logs above replace it.

### S9c — the quadratic posture term

`pose_quadratic` is added beside the shipped `pose`, with the same reference, the same
0.05 rad tolerance and weight −0.04. `pose` is `exp(-mean((q-q*)/std)^2)`, which is
numerically zero — value and gradient — at the twists S9b2 reached. The new term is the
negative log of the same quantity, so it pulls from any distance.

Measured posture probe (`mean_j((q-q*)/0.05)^2`, 8 envs × 200 steps): **57.1 / 94.2 / 79.7
→ 4.40 / 2.20 / 6.69**. The twist collapsed 12–43×, at the cost of about one impact in ten.

### S9d — dead ball

Built and aborted before launch: the launcher was killed and no job directory was created.
The condition is a ball placed below the plate, with every ball-caused termination and
reward removed, so the only way an episode ends early is a fall. Not measured.

### S10 — visible ball, mixed safety

A bundle of four changes on top of S9c, because they only make sense together:

* **Visibility gate.** The ball history admits a sample only if the ball was visible at the
  capture time, and emits twelve zeros until the first admitted frame. Before this, the
  history modelled lateness but never absence.
* **Four-case mixture**, sampled per episode: 70 % normal play, 7.5 % hold, 7.5 % ball
  lost and then returned, 15 % induced loss. The loss time is drawn from 5–20 s.
* **30 s episodes**, up from 20 s.
* **Loss cost** retained from S9a; `ball_lost_event` sits at weight 0.0.

Fall share 0.64 / 0.70 / 0.52 %. Posture probe 0.66 / 0.71 / 6.29.

### S11 — tilted start

Two reset terms inserted after `reset_robot_juggle_joints` and before `reset_ball`:
`reset_tilt_body` on waist yaw/roll/pitch, both hips, both knees and both ankle rolls at
**±5°**, and `reset_tilt_ankle_pitch` on both ankle pitches at **±15°**. The pelvis itself
is not tilted — it has no joint, it is the root body, and it reads 0.00° roll and pitch at
reset. The lean emerges from physics settling onto displaced feet.

Final metrics, last 300 iterations: flight 0.611 / 0.614 / 0.610 s, impacts 32.8 / 32.7 /
34.8, alignment 39.8 / 42.1 / 43.2 mm, fall share **0.60 / 0.43 / 0.85 %** against the S10
parents' 0.64 / 0.70 / 0.52 %. Posture probe, measured with upright resets in the S10 task:
**0.62 / 0.65 / 5.28**.

**Verdict: the tilted start became a state the policy can act from. It does not show
recovery.** The metric is a fall count, and no test starts the robot at a known tilt and
asks whether it returns.

Seed 27 is the exception. Its `pose` reads 0.005 against 0.393 and 0.390, and its
`pose_quadratic` −0.2775 against −0.0519 and −0.0518. The difference is the same quantity,
not a separate effect: `pose = exp(-Σ(Δq/0.05)²/31)`, so 0.005 is `exp(-5.0)` against
`exp(-0.94)`. Its deviation is spread over the whole body — waist 43.6 (of which
`waist_pitch` 41.5), both hip pitches 49.1, both arms 48.3 — a self-consistent forward
lean. Every other reward term ties seed 7 within 2 %, so the two posture terms account for
0.614 of the 0.68 per-step reward gap. Seed 27 sits in a worse solution, not a trade, and
the −0.04 weight is 3.6 % of intake, which is too small to pull it out.

## 5. What the ladder established

1. **A narrow latency band teaches a fixed lead; a wide one teaches age reading.** S2 dies
   at 0 ms and S3 is flat across 0–140 ms. Train the band, not the mean.
2. **A stall model must come from a recorded clip.** S4's assumed gaps are not what the
   camera produces and the rung was seed-unstable. S4b's measured gaps pass and set R9
   (latency + longest in-view gap ≤ 0.34 s).
3. **Measurement noise is nearly free inside the band; stale samples are not.** S5 changes
   nothing at 0–140 ms and dies at 200 ms.
4. **The exponential posture term saturates.** Only the quadratic form pulls from the
   twists S9b2 reached.
5. **Fall counts are not probabilities.** `Episode_Termination/fell_over` is
   `count_nonzero` of the termination mask, so its denominator is the reset rate, which
   differs per rung. Compare **fall share**: fell over ÷ all terminations in the same
   window. S9c 56.5 / 55.1 / 28.2 % → S10 0.64 / 0.70 / 0.52 % → S11 0.60 / 0.43 /
   0.85 %.
6. **`ball_impact_count` counts racket–ball impacts, not floor bounces.** No floor-bounce
   metric exists in the task.
7. **`ball_lost_events` does not exist after S10.** The term that logged it was removed and
   its replacement sits at weight 0.0, so its log is a constant zero. The safety half of
   the S10 mixture is currently unmeasured.

## 6. Artifacts

Local checkpoints, fetched from the managed container:
`~/data/tennis_s9b2_20261106/` (`model_32499.pt`), `~/data/tennis_s9c_20261106/`
(`model_34498.pt`), `~/data/tennis_s10_20261106/{s7,s17,s27}/` (`model_36497.pt`),
`~/data/tennis_s11_20261107/{s7,s17,s27}/` (`model_37496.pt`).

Server trees on `10.14.64.48`: `mjlab-s7b`, `mjlab-s7c`, `mjlab-s8`, `mjlab-s9a`,
`mjlab-s9a2`, `mjlab-s9b`, `mjlab-s9b2`, `mjlab-s9c`, `mjlab-s10`, `mjlab-s11`, each with
`src/`, `jobs/` and its launcher. `mjlab-s9d` exists with no job directory.

Closed-loop sweep data: `~/data/cl_sweep_s1s7`, `~/data/cl_sweep_s2s7`,
`~/data/cl_s2_zerolat`, `~/data/cl_sweep_s7b` (one `result.json` per cell).

## 7. Open items

* **Nothing measures recovery.** Build a test that starts the robot at a known tilt and
  reports whether it returns.
* **Seed 27's posture needs its own rung** with a `pose_quadratic` weight around −0.25 to
  −0.35, warm-started from `model_37496.pt`, accepted on the per-joint profile.
* **`ball_lost_event` needs a logging-only rising-edge counter** so the safety half of the
  S10 mixture becomes measurable.
* **`ball_in_camera_fov` still uses the raw fisheye cone**, 5.46× too wide horizontally and
  1.52× vertically against the delivered 66.4° × 53.5° image. The largest known reward
  defect, deliberately unfixed.
* **`ball_not_lifted` threshold** `JUGGLE_LIFT_ENERGY_THRESHOLD_J_PER_KG` is 1.0 J/kg,
  which is 102 mm, leaving a 2 mm silent hole above the 100 mm minimum. The proposed value
  is 0.5. It cannot affect S10 or S11, whose termination set no longer contains it.
* **A push rung is proposed but not built.** The current push is a root velocity step
  every 1–3 s; a *sustained* push would build a tilt and is the condition that continues
  the S11 question. It needs one new event term.

## 8. Appendix: why the ladder is not one scheduled task

A rung's condition could be switched by iteration count inside a single run, removing the
manual resumes. The cost is higher than it looks, and the exercise would lose the
attribute that makes the ladder useful.

* Every manager deepcopies its config at construction
  (`reward_manager.py:65`, `event_manager.py:155`, `termination_manager.py:41`,
  `observation_manager.py:171`), so a stage cannot be switched by editing `env.cfg` after
  the environment is built. A silent no-op is indistinguishable from a stage that ran.
* The safe design is to create every term at iteration 0 and gate it at runtime: rewards
  at weight 0.0, terminations behind a stage flag, one reset dispatcher covering drop,
  release range, hold, offset, mixture and tilt, and stream parameters read from the stage
  object. Startup-mode randomisation must be applied by hand and followed by a global
  reset; `geom_size` is runtime-safe because it recomputes `geom_rbound` and `geom_aabb`.
* Estimate: about 950 lines, of which 250 are a boundary probe that asserts the live
  managers equal the staged factory's at every boundary. Two days of implementation, about
  one more day chasing boundary mismatches. A launcher that chains the existing rungs and
  resumes automatically is about 80 lines and gives the same operation.
* The reason not to schedule the ladder is that many rungs are **corrections**, not
  difficulty steps: S7a's broken hold metric, S9a's reward-set change, S9c's new term, S4b's
  corrected stall model. A schedule would freeze those as hyperparameters. The merged form
  should hold the final reward and termination set from iteration 0 and schedule only the
  condition axis: latency band → stalls → noise → release range → hold and offset →
  mixture and tilt. That is seven steps, not eleven rungs.
