# X2 tennis juggling: three-run audit and curriculum recommendation

Observed: 2026-11-04 20:43 (+08). Scope: local artifacts and CPU checks only.

## Decision

Use a curriculum, but first make task failure independent of racket orientation.
The current evidence does not show that the physical task is too difficult.
It shows weak task success, a loss-detector loophole, and a demanding initial training distribution.
Do not extend these policies to another long training run without these checks.

No task source, training settings, remote process, or robot was changed.
No policy rollout was run in this audit.

## Inputs and method

Saved policies and configurations:

- `/home/agiuser/data/tennis_runs_20261104/baseline/`
- `/home/agiuser/data/tennis_runs_20261104/clean/`
- `/home/agiuser/data/tennis_runs_20261104/nohist/`

Matching TensorBoard files:

- `/home/agiuser/data/tb-final/6_b3_baseline/`
- `/home/agiuser/data/tb-final/8_clean/`
- `/home/agiuser/data/tb-final/7_nohist/`

Each file has 30,000 finite points for each selected scalar, at iterations 0–29,999.
Each checkpoint records iteration 29,999 and 720,000 environment control steps.
All three use seed 7 and 4,096 environments in the saved configuration.
These files do not establish worker count or process exit status.

The ONNX actor widths are 112, 112, and 103, respectively.
All use 29 actions. These are variants of the named task, not its 31-action base configuration.

Analysis files: [`logs/diagnostics/tennis_curriculum_review_20261104/`](../../logs/diagnostics/tennis_curriculum_review_20261104/).

- `analyze.py`: scalar extraction, input hashes, window means, and plots.
- `inputs.json`: absolute input paths, sizes, and SHA-256 hashes.
- `windows.csv`: means and standard deviations across logged points.
- `scalars.npz`: extracted scalar arrays.
- `curves.png`: non-overlapping 500-iteration bin means.
- `probe_termination.py`: CPU-only loss-detector counterexample.

Standard deviations are not confidence intervals across seeds.
Training statistics are not deterministic evaluation results.
Source inspection used local HEAD `a89c847d9` with pre-existing unrelated edits.
The saved `git/` directories are empty. Historical source identity is not established.

## Saved configuration differences

| Item | baseline | clean | nohist |
|---|---|---|---|
| Actor ball input | Three positions with ages | Three positions with ages | One delayed position |
| Stream rate | 5–20 Hz | 25–30 Hz | Fixed delay update period: 5 control steps |
| Delay | 80–250 ms | 30–80 ms | 5–9 control steps: 100–180 ms |
| No-bounce termination | Absent | 2.5 s | Absent |
| Racket-lift penalty | Zero | Zero | Zero |
| Energy target | 0.40 m | 0.40 m | 0.40 m |
| Curriculum | None | None | None |

`clean` changes both timing and termination. It is not a single-factor comparison with `baseline`.
`nohist` also changes the corruption model, not only history length.
The stream models and the hold model use different stall mechanisms.

All three retain:

- Root-velocity pushes every 1–3 s: x/y ±0.5 m/s; z ±0.4 m/s.
- Angular-velocity pushes: roll/pitch ±0.52 rad/s; yaw ±0.78 rad/s.
- Proportional and derivative gain scales of 0.7–1.3 at reset.
- Other physics and observation randomization.
- A 20 s episode cap and a 0.02 s control step.
- Velocity tracking weights of 2.0 each and ball-energy weight 2.0.
- Raw-action change penalty weight −0.1, with no action clipping in the saved settings.
- A feed-forward actor. Only the critic receives clean ball velocity.

The current task guide describes a nonzero lift penalty.
The saved configurations and current constants instead set it to zero.
The constants explain that earlier runs improved when this penalty was removed.
This audit does not recommend restoring the old penalty without a separate test.

## What the training logs show

Final window: iterations 28,000–29,999 inclusive.

| Run | Mean episode duration | Contact edges/episode | Mean ball energy | Mean absolute energy error |
|---|---:|---:|---:|---:|
| baseline | 19.48 s | 3.72 | −9.75 J/kg | 13.74 J/kg |
| clean | 8.72 s | 18.41 | 3.28 J/kg | 1.36 J/kg |
| nohist | 7.34 s | 18.77 | 3.35 J/kg | 1.38 J/kg |

The target energy is 3.924 J/kg.
Contact edges include the initial passive impact and low bounces.
They are not counts of consecutive bounces at the target height.

The curves do not show that all learning stops at 2,000 iterations:

- `clean`: duration increases from 6.70 s at 1,500–1,999 to 8.72 s in the final window.
- `clean`: mean completed airborne interval increases from 0.269 s to 0.391 s.
- `nohist`: duration increases from 4.15 s to 7.34 s.
- `nohist`: contact count increases from 9.46 to 18.77.

These gains remain insufficient evidence of reliable juggling.
In `clean`, logged ball-drop counts greatly exceed robot-fall counts.
Those values are per-reset-call counts, not per-episode failure probabilities.
They suggest that ball loss, not robot collapse, dominates this training distribution.
The user's playback observations can differ from stochastic training behavior.

### Baseline survival is not evidence of holding the ball

The energy function is:

`E = 0.5 * vz**2 + 9.81 * (z_ball - 1.1530848028)`.

Kinetic energy is nonnegative. Therefore the mean energy bounds mean ball height from above.
At −9.75 J/kg, the corresponding mean height is at most about 0.159 m.
This is far below the normal racket surface at about 1.119 m.
The longest continuous racket contact averages only 0.00996 s.
These logs do not support the earlier description of this baseline as simply parking on the racket.
They are consistent with the ball spending much of an episode near the floor.
A rollout is still needed to identify the actual policy behavior.

### Loss-detector counterexample

[`ball_lost`](../../src/mjlab/tasks/velocity/mdp/terminations.py) tests the ball's signed height in the rotating racket frame.
It does not independently test world height, floor contact, or a reachable workspace.
Changing the racket normal can change the sign without moving the ball.

The CPU probe calls the actual detector with the same world positions for 25 steps:

- Ball centre z = 0.0327 m; racket z = 1.1194 m.
- Upward racket normal: terminates after 0.5 s.
- Downward racket normal: does not terminate.

The counterexample passes. It proves a detector loophole, not that this policy uses that exact orientation.
`clean` also has the independent 2.5 s no-impact timeout, which bounds this particular no-contact survival path.

Recommended correction: retain a recovery grace where appropriate, but add an independent floor/loss condition.
Use world/terrain height or ball-ground contact, plus a generous reachable-workspace bound.
Validate against tilted-racket cases. Do not define every failure relative to an action-controlled frame.

### Separate nohist anomaly

The raw-action penalty has extreme outliers:

- Iteration 22,019: `Episode_Reward/action_rate_l2 = -534053312`.
- Iteration 29,296: `Episode_Reward/action_rate_l2 = -2590719.5`.
- Final-window median: approximately −0.867.

This can strongly distort mean reward curves.
The cause is unknown. Inspect action, observation, and simulator-state extremes before interpreting this as a learning plateau.
Do not hide these samples through clipping or smoothing without diagnosis.

## Recommended curriculum

These settings and gates are proposals, not tested values.
Keep the measured contact physics, actuator limits, and robot geometry unchanged.

### Stage 1: learn the repeated bounce

- Use the full standing robot with zero movement command.
- Remove external pushes initially. Use nominal gains and narrow reset variation.
- Use fresh ball history, low noise, and no artificial stalls initially.
- Start with centred drop states, not an extended ball-holding phase.
- Use a 0.20 m target first. Require repeated active returns, not the first passive rebound.
- Preserve balance, joint-limit, and collision constraints.

Keep actor input dimensions compatible when possible.
The lower target reduces required launch speed, but also shortens flight time.
It is easier for actuation, not automatically easier with delayed perception.
Thus lower-height training should begin with the fresh stream.

A proposed advancement gate is at least 80% of fixed evaluation episodes with 10 consecutive qualified bounces.
A qualified bounce requires free flight, an apex inside the stage band, and a return to the racket.
Do not count the initial drop as a successful cycle.
Reject floor contact, robot falls, and sustained holding.
Validate the gate over several evaluation batches, not one training scalar.

### Stage 2: reach the final height and broaden starts

Increase target height from 0.20 to 0.30 to 0.40 m after the bounce gate passes.
Then broaden lateral offset, incoming velocity, and physically consistent release phase.
Introduce plate starts so the policy also learns the first lift.
Retain some easy starts to reduce forgetting.

Do not advance on total reward or survival alone.
A lower Gaussian energy target also increases reward for a stationary ball.
At sigma 1.5 J/kg, a resting ball scores 0.181 for a 0.20 m target, versus 0.00107 for 0.40 m.
Anti-holding checks must remain active.

Use qualified-bounce events as an additional success signal, not the sole sparse reward.
Keep dense flight/energy and return-position shaping with explicit validity checks.
Audit the return-position signal for moving-racket and release-only loopholes.

### Stage 3: add realistic uncertainty

Increase one difficulty axis at a time:

1. Reset variation and modest gain variation.
2. Measured latency, rate variation, noise, and stalls.
3. Stronger body disturbances.
4. Walking commands only after standing juggling succeeds.

Keep a fixed nominal evaluation and a separate deployment-distribution evaluation.
Track skill retention when each axis changes.
Curriculum state must survive save/resume. Ignore artificial initial episode-length staggering in mastery gates.

## Observation and optimization priorities

Three old positions are not a complete state estimator.
The history stores each position in the racket frame at its own capture time.
The frames differ when the racket moves. Impacts also interrupt ballistic motion.
Use synchronized proprioception/action history, or a recurrent estimator, if the fresh-stream policy succeeds but delayed training fails.
Clean critic velocity does not give the actor missing information.
A privileged-state actor is useful as a learnability diagnostic, not as a deployable policy without a matching estimator.

At a stationary strike height, a 0.40 m bounce has about 0.571 s of flight.
The baseline's 80–250 ms delay consumes about 14–44% of that interval before stalls.
No curriculum can recover unpredictable events that the observations do not reveal.
Hardware-rate improvement may still be needed.

The final `clean` action-change penalty is −1.010, versus +0.609 for ball energy in the same logging units.
This warrants an arm-versus-body penalty breakdown, not immediate removal of the penalty.
Check deterministic evaluation and contact metrics before tuning entropy or training duration.

## Smallest next authorized work

1. Run short deterministic rollouts of all three policies under saved and nominal disturbance settings.
2. Record world ball height, racket normal, ball-ground contact, impact times, action extremes, and termination reasons.
3. Correct and regression-test the loss detector before another training comparison.
4. Train one bounded simplified-condition pilot, approximately 3,000–5,000 iterations, with evaluation every 500 iterations.
5. If qualified bounce sequences do not improve, inspect impulse feasibility or test a privileged-state actor.
6. If the pilot succeeds, implement the mastery-gated curriculum and confirm with additional seeds.

Policy evaluation, task changes, and training launches still require operator approval.
Other remote sweep arms were not queried. Their current operational status is unknown to this audit.
