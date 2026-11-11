# Target-height juggle: S1, S1a, S2 and S3

Date: 2026-11-08 (+08), results added 2026-11-09 (+08).
Status: **plan and results for four rungs.** Section 8 reports what was measured.

A second juggle task family that takes the target apex height as an **input** instead of a
constant. S1 is the fixed 0.40 m condition, unchanged in behaviour. S1a draws the target
uniformly from 0.05 to 0.40 m. S2 and S3 then carry that family through the shipped
ladder's two perception bands, 80-100 ms and 40-140 ms.

## 1. Why a new family and not a stage

The target is a new actor and critic input, so the observation widens and no checkpoint
resumes with `strict=True`. The existing ladder keeps its 112-wide actor and 121-wide critic;
this family is separate from S1 of that ladder in name only — the conditions are the same.

| | existing ladder, 29-DOF | this family, 29-DOF |
|---|---:|---:|
| actor | 112 | **113** |
| critic | 121 | **122** |
| actions | 29 | 29 |

The two widths are measured by the probe, not assumed. The controller's 112-wide path does
not apply to this policy: it needs a 113-wide/29-action path plus a fresh ONNX export before
it can run on hardware. That work is out of scope here.

## 2. The target is a command, not a measurement

A `target_height` command term, modelled on `UniformVelocityCommand`:

* one scalar per environment, in **metres**, shape `(num_envs, 1)`;
* resampled on every episode reset, through the existing `CommandTerm.reset(env_ids)` path,
  so the target a reward scores against is the target the episode was given;
* range from the stage: `(0.40, 0.40)` for S1, `(0.05, 0.40)` for S1a;
* overridable for an evaluation cell with `MJLAB_JUGGLE_TARGET_HEIGHT_RANGE_M` (`H` or
  `LO,HI`), in the style of the existing `MJLAB_JUGGLE_*` overrides.

The command is **not corrupted**. It is a goal, not a sensor reading, and corrupting it
would teach the policy to hedge against a target error the operator does not have.

## 3. The observation

One new term, `target_height`, appended to both the actor and the critic group:

* normalised as `h / 0.40`, giving 0.125 to 1.0 across the range, so the input sits in the
  same magnitude band as the rest of the vector;
* appended **last** in both groups, so every existing column keeps its index;
* the critic gets the same value, uncorrupted, like its other privileged terms.

## 4. The four consumers

A conditioning input that no reward reads is ignored, and the failure is silent: the reward
curve stays normal and the input carries no information. Every consumer below must read the
command, and every rule below is a **no-op at 0.40 m**, so S1 reproduces today's condition.

| consumer | rule | at 0.40 m |
|---|---|---|
| `ball_energy` target | `target_height` from the command | 0.40 |
| `ball_energy` sigma | `sigma = JUGGLE_ENERGY_SIGMA_J_PER_M * h`, with the constant `1.5 / 0.40 = 3.75` J/kg per m | 1.5 J/kg |
| `ball_not_lifted` threshold | `threshold = JUGGLE_LIFT_ENERGY_J_PER_M * h`, with `1.0 / 0.40 = 2.5` J/kg per m | 1.0 J/kg |
| `ball_first_lift_s` threshold | same value as `ball_not_lifted` | 1.0 J/kg |

The sigma rule matters more than it looks. A constant 1.5 J/kg is 153 mm of height at any
target, so it would be **48 % of a 0.05 m target and 6 % of a 0.40 m target** — the same
policy facing an eightfold different relative tolerance. Scaling it holds the tolerance at a
constant 38 % of the target (153/400), which is what the shipped rung actually trains.

The threshold rule removes a defect by construction. The absolute 1.0 J/kg is 102 mm, so a
0.05 m target would end every perfectly juggled episode as "not lifted". Expressed as 25.5 %
of the target energy, the 100–102 mm silent hole cannot exist at any target.

`ball_energy_value` takes the same command, so a metric and the reward can never disagree
about which target is being scored.

## 5. The starting condition

Today the ball spawns at a fixed drop height above the string bed, and the rung's job is to
turn that first bounce into a juggle. A 0.05 m target with a 0.20 m spawn is not a smaller
juggle: the string bed's restitution is about 0.88 and a 0.20 m drop rebounds to about
0.205 m, so the policy would have to **remove energy at every contact** — a soft catch, not
a tap. That is a different skill, and it would confound the reference-height rung.

The spawn therefore scales with the target:

```
drop_height = JUGGLE_SPAWN_DROP_HEIGHT * (h_target / 0.40)      # 0.20 m at 0.40
```

so the whole episode is the shipped one scaled by `h/0.40`: impact speed falls as
`sqrt(h/0.40)` and flight time as `sqrt(h/0.40)`, which is what a scaled juggle looks like.
At 0.05 m the spawn is 25 mm above the bed.

**Lateral jitter does not scale.** `JUGGLE_HOLD_LATERAL_RANGE_M` is an absolute operator
error, so a ±20 mm offset on a 50 mm apex is proportionally four times harder than on a
400 mm apex. That is a real property of the hardware and is left unscaled on purpose.

Later rungs inherit this rule. The hold heights (S7a/S7b: 0.10–0.40 m) and the mixture's
release heights must become **release ≤ target** when those rungs are ported to this
family, or they will command an apex the ball cannot reach. Not implemented here.

## 6. Stages

| stage key | condition | target range | parent | budget |
|---|---|---|---|---|
| `s1` | the shipped `s1-15hz`: 15 Hz stream, 0–40 ms latency | fixed 0.40 m | none, from scratch | 6000 iterations |
| `s1a` | S1 | uniform 0.05–0.40 m | S1 | 6000 iterations |
| `s2` | the shipped `s2-80-100ms`: 15 Hz, **80–100 ms** latency | uniform 0.05–0.40 m | S1a | 6000 iterations |
| `s3` | the shipped `s3-40-140ms`: 15 Hz, **40–140 ms**, redrawn per acquisition | uniform 0.05–0.40 m | S2 | 6000 iterations |
| `s1-fixed15` | `s1-15hz` | fixed 0.15 m | none, from scratch | 6000 iterations |
| `s2-fixed15` | `s2-80-100ms` | fixed 0.15 m | `s1-fixed15` | 6000 iterations |
| `s3-fixed15` | `s3-40-140ms` | fixed 0.15 m | `s2-fixed15` | 6000 iterations |
| `s1-fixed30` | `s1-15hz` | fixed 0.30 m | none, from scratch | 6000 iterations |
| `s2-fixed30` | `s2-80-100ms` | fixed 0.30 m | `s1-fixed30` | 6000 iterations |
| `s3-fixed30` | `s3-40-140ms` | fixed 0.30 m | `s2-fixed30` | 6000 iterations |

Only `s1-fixed15` and `s1-fixed30` have trained. Section 8.1 reports both gates, which
failed. The S2 and S3 fixed rungs are defined in `STAGE_TARGET_HEIGHT_RANGE_M` and
`STAGE_BASE_STAGE` but have not run, so their perception transfer to a fixed low target is
unknown.

The perception condition of each stage comes from the ladder's own stage table:
`STAGE_BASE_STAGE` maps `s1` and `s1a` to `s1-15hz` and `s2` to `s2-80-100ms`, so the
latency, stall and noise ranges have one definition in the repository. S2 carries S1a's
target range unchanged, because a rung carries the complete condition and adds one
thing. The probe asserts that S1a and S2 differ in exactly one parameter,
`actor/ball_hist_pelvis/stream_latency_s`.

The condition is the existing stage factory's `s1-15hz`, so this family is a replay of the
ladder with the height axis inserted. Later rungs port one at a time into this family; they
are not part of this change.

Task ids:

```
Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF
  -Ball3-Pelvis-Nominal-S1-TargetHeight
Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF
  -Ball3-Pelvis-Nominal-S1a-TargetHeight
```

29-DOF only. `head_actuated=False`, as in the deployed ladder head.

## 7. What the rung has to show

S1 must be **behaviourally identical** to the shipped S1 condition at a 0.40 m target. The
probe proves that by comparing values, not by reading code:

1. **Widths.** Actor 113, critic 122, actions 29.
2. **Command is per environment.** Two environments given 0.05 and 0.40 report those values
   in the observation and in the reward.
3. **The reward equals the shipped one at 0.40.** Same ball state, target 0.40: `ball_energy`
   equals the value computed from the constant-0.40 form, within float tolerance.
4. **The reward peaks at the commanded apex.** For a ball at rest at height `h`, the reward
   is 1.0 at every `h` in the range, so the same state scored against a 0.05 target and a
   0.40 target gives two different values.
5. **Sigma and threshold scale.** The half-width is `3.75 · h` J/kg and the lift threshold is
   `2.5 · h` J/kg, checked at 0.05, 0.20 and 0.40.
6. **Spawn scales.** Mean reset spawn height over 256 resets is `0.5 · h` within sampling
   error, at 0.05 and at 0.40.
7. **The target survives a reset.** The command is resampled when the episode resets and does
   not drift inside an episode.

The measurement that decides whether S1a works is not in the probe. It is the **commanded
against achieved apex table**: bin the target (0.05–0.10, …, 0.35–0.40) and report the
achieved apex mean and sd per bin. A slope near 1.0 means the policy reads the input. A flat
line near 0.22 m means it ignores it, which is the failure mode this design exists to make
visible. The apex per ball bounce must be logged for that, which is a metrics addition, not a
reward change.

## 8. Results so far

Four rungs trained, two seeds each, 4096 envs, 6000 iterations per rung. All arms ended
`exit=0` with zero errors. Values are the mean over the last 300 iterations.

| term | S1 s7 / s17 | S1a s7 / s17 | S2 s7 / s17 | S3 s7 / s17 |
|---|---|---|---|---|
| `ball_energy` | 1.9518 / 1.9522 | 1.9374 / 1.9364 | 1.9367 / 1.9364 | 1.9361 / 1.9379 |
| `ball_centred` | 0.2464 / 0.2465 | 0.2470 / 0.2469 | 0.2468 / 0.2467 | 0.2470 / 0.2469 |
| `ball_in_fov` | 0.1996 / 0.1997 | 0.2000 / 0.2000 | 0.2000 / 0.2000 | 0.2000 / 0.2000 |
| `pose` | 0.9370 / 0.9411 | 0.9444 / 0.9446 | 0.9440 / 0.9445 | 0.9441 / 0.9454 |
| flight time | 0.5557 / 0.5559 s | 0.4063 / 0.4052 s | 0.4051 / 0.4041 s | 0.4053 / 0.4069 s |
| impacts | 35.00 / 35.00 | 51.07 / 51.18 | 51.25 / 51.27 | 51.35 / 50.98 |
| `time_out` | 4.1655 / 4.1647 | 4.1754 / 4.1725 | 4.1981 / 4.1828 | 4.1587 / 4.1668 |
| `fell_over` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| fall share | 0.00 / 0.00 % | 0.00 / 0.00 % | 0.00 / 0.00 % | 0.00 / 0.00 % |
| logged target | 0.4000 / 0.4000 | 0.2261 / 0.2248 | 0.2252 / 0.2241 | 0.2241 / 0.2251 |
| checkpoint | `model_5999.pt` | `model_11998.pt` | `model_17997.pt` | `model_23996.pt` |

Three readings, in order of how much they settle.

1. **The height command is used.** Flight time falls from 0.556 s to 0.406 s when the target
   becomes uniform 0.05-0.40 m (mean 0.225 m). A ballistic arc with a 0.225 m apex lasts
   `2 * sqrt(2h/g)` = 0.428 s, so the measured 0.406 s is that arc, minus restitution. The
   shorter flight raises contacts from 35.0 to 51.2, which is the same apex at a higher rate.
   A policy that ignored the input would keep the 0.40 m arc and 35 contacts.
2. **Perception cost nothing.** S2 (80-100 ms band) is within 0.04 % of S1a on energy and
   0.3 % on flight time, on both seeds. The 80-100 ms band is the shipped ladder's own
   condition, and it transfers to this family without retraining damage.
3. **The wide band held too.** S3 (40-140 ms, redrawn per acquisition) is within 0.07 % of
   S2 on energy, 0.04 % and 0.69 % on flight time, with `fell_over` zero on both seeds. The
   seed spread inside S3 (0.09 % on energy) is larger than the S2-to-S3 change, so the
   correct verdict is "held, and cannot be distinguished from S2", not "improved".

What this does **not** establish: the **commanded against achieved apex slope** is still
unmeasured. Flight time and impact count show that the policy changes its timing with the
commanded height; they do not show that the apex tracks the command bin by bin, and a
policy that only reads the mean would look the same in these aggregate terms. The per-bounce
apex recorder described in section 7 is still required before the slope can be reported.

### 8.1 Fixed targets: two causes separated, and the low heights fail

S1a draws its target, and two of its three seeds collapsed. Either the draw is the
difficulty, or some heights are hard alone. The `s1-fixed15` and `s1-fixed30` stages pin one
height and leave the draw behind, so they separate those two causes. The S1 rung of this
family is already a fixed 0.40 m target, so it serves as the control at the top of the draw.

All three ran 6000 iterations from scratch, 4096 envs, seed 7, the same perception condition
(`s1-15hz`). The closed-loop gate ran 60 s in 16 envs. Apex, resets, bounces and heights are
gate measurements; flight time and impacts are the last training window.

| condition | gate | apex p50 | apex p90 | resets / min | bounces / min | flight time | impacts |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed 0.40 (control), s7 / s17 | PASS | 0.3986 / 0.3949 | 0.4145 / 0.4111 | 16 | 1497 / 1456 | 0.5507 s | 35.0 |
| fixed 0.30 | **FAIL** | 0.3008 | 0.3012 | **288** | 467 | 0.0192 s | 153.9 |
| fixed 0.15 | **FAIL** | 0.1488 | 0.1504 | **320** | 315 | 0.0223 s | 183.6 |

1. **At 0.40 m the family is exonerated.** The control matches the shipped ladder S1 to four
digits: 1497 bounces per minute, 35.0 impacts, 0.55 s flight. Appending a `target_height`
observation and reading it costs nothing at the top of the draw.
2. **Both low targets start juggling and then collapse.** The early training windows show the
control's own flight band, 0.18-0.23 s. By the last window it is 0.019 s. The policy settles
into a cheaper solution: it pats the ball at the commanded height instead of lifting it. The
ball then sits at the target, so every height metric reports success.
3. **A dribble is not a low bounce rate.** The two collapsed arms bounce 315 and 467 times a
minute. The control bounces 1497. A bounce-rate floor near 200 would accept a dribble, and
`share_time_above_0p20_m` reaches 0.93 at the 0.30 m target. The separating measurement is
the **flight time**, 0.019-0.022 s against 0.55 s, and the **impact count**, 154-184 against
35.0. Apex p90 also separates them at fixed target: it is 0.0016 m above p50 when the policy
dribbles, 0.016 m when the ball flies.
4. **The draw is not the cause.** The failure reproduces at two fixed heights, and at 0.40 m
the same code trains normally. The operative property is the height.
5. **Both failures pass the apex rule.** `apex_p50` is at the commanded height in both cases,
0.3008 and 0.1488. The gate rejected them on resets only, 288 and 320.

What this does **not** establish: whether a low fixed target can be trained at all with a
longer budget, a different seed, or a reward that rejects the dribble. Only seed 7 was run at
each fixed height. The collapse onset is not measured per iteration, so the transition from
juggling to dribble is bracketed by the training windows, not timed.

## 9. Out of scope

* The 31-DOF variant and the head-actuated path.
* The controller's 103-wide/29-action path and the ONNX export.
* Every rung after S1a: stalls, noise, release range, hold, offset, drop tail, domain
  randomisation, the posture term, the mixture, the tilted start.
* Changing any existing reward weight.
