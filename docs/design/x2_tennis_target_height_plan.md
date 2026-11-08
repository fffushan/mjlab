# Target-height juggle: S1 and S1a

Date: 2026-11-08 (+08).
Status: **plan, approved for implementation.** No run is authorized by this document.

A second juggle task family that takes the target apex height as an **input** instead of a
constant. S1 is the fixed 0.40 m condition, unchanged in behaviour. S1a draws the target
uniformly from 0.05 to 0.40 m.

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

## 8. Out of scope

* The 31-DOF variant and the head-actuated path.
* The controller's 103-wide/29-action path and the ONNX export.
* Every rung after S1a: stalls, noise, release range, hold, offset, drop tail, domain
  randomisation, the posture term, the mixture, the tilted start.
* Changing any existing reward weight.
