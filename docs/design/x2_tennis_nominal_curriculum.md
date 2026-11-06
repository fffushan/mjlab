# X2 juggling curriculum: implementation and real-robot acceptance plan

Date: 2026-11-05 (+08).
Status: **planning only**. No implementation, training, deployment or robot operation is authorized by this document.
The completed E0 evaluation is accepted only within its nominal simulation scope.

## 1. Decision and scope

**Real-robot trials are the decisive acceptance gate. Simulation is a screening and diagnosis tool.**
The next priority is a hardware-compatible observation path and a measured robot baseline.
The earlier proposal to widen simulated reset jitter first is no longer the default next action.

Keep the 0.40 m target and use the completed 0.40 m policies as initialization.
Do not start a 0.20 → 0.40 m height ladder.
Do not add a recurrent network, state estimator, full proprioception history or automatic curriculum scheduler.
Do not change rewards, gains or limits to hide an uncorrected deployment mismatch.

The development cycle is:

**measure the robot → correct contract/calibration faults → change one training condition → screen in simulation → evaluate on the robot → retain or reject.**

Passing a simulation gate alone never promotes a curriculum stage.
A failure in the known-imperfect vendor simulator alone does not establish poor hardware policy quality.
Use that simulator primarily to check controller integration and the operator procedure.
Observation/action parity and safety checks remain mandatory regardless of simulator behavior.

## 2. Current evidence and blockers

| Item | Evidence / status | Consequence |
|---|---|---|
| Nominal 0.40 m juggling | Three policies, 96/96 episodes, 34 consecutive qualified flights each | Suitable simulation starting point, not hardware readiness |
| Measured simulated apex | 0.3866–0.4261 m above the rest datum | The policy can produce the intended flights in this model |
| Nominal 0.20 m result | Raised-racket carry; zero qualified flights | Do not use it as an easier curriculum stage |
| Hardware behavior of these six checkpoints | **Not tested** | No hardware success or reliability claim |
| Controller observation support | Local `policy_adapter.cc` accepts `ball_pos`, not `ball_hist_pelvis` | A model-file swap is insufficient |
| Current ball conversion | `hardware_observation.cc` converts the latest camera point into the current plate frame | Does not implement capture-time pelvis history |
| Trained perception | Clean 50 Hz, zero latency/noise/stalls | Real measurements must not be assumed equivalent |
| Actual robot, head state, release procedure and sensor timing | **Not freshly measured for this plan** | Select and record the intended platform before fixing training ranges |

Sources: [E0 report](../reports/x2_tennis_nominal_evaluation_20261105.md),
[curve comparison](../reports/x2_tennis_nominal_comparison_20261105.md),
[pelvis-history contract](x2_tennis_pelvis_ball_history.md).
The [perception interface](../../../ESS/x2/docs/operator/TENNIS-PERCEPTION-INTERFACE.md)
contains historical rate, latency and calibration evidence. Its older implementation-status statements are not current authority.
Do not copy those historical values into a new training preset without checking the selected robot.

## 3. Implementation work packages

These packages describe required work. Each executable phase needs its own approval.
Keep one owner for each repository's edits and preserve existing dirty files.

### P0 — Define the robot baseline and collect non-commanding evidence

Select one robot, one racket/ball setup and one intended perception pipeline.
Record robot identity, head/waist configuration, camera calibration, controller version and existing safety configuration.
Use existing recordings first. If new recordings are needed, obtain approval for a supervised sensor-only session.
Do not engage the new policy to collect this baseline.

Measure:

- Point capture time, arrival time, accepted sample rate, delay distribution and longest gaps.
- Camera frame, camera-to-pelvis calibration, ball-centre convention and depth error over the intended workspace.
- Actual head/waist joint values and their measurement timestamps; do not substitute assumed healthy poses.
- Hand-release position, velocity and timing relative to the real controller handoff.
- Joint/IMU timing and existing controller command latency from available safe recordings.

**Artifacts:** a short robot-baseline report, selected bag/video paths and a plain parameter table with units and evidence sources.
Separate repeatable bias from noise, and sensor capture age from transport arrival age.
Record clock mapping and synchronization uncertainty before comparing different clock domains.

**Exit condition:** the real input path, reset/handoff and current limits are described well enough to reproduce them offline.
If a deterministic calibration or timestamp fault is found, fix it before using randomization as compensation.

### P1 — Implement the minimal 112-input controller contract

Owning repository: `x2_docker/`. Proposed edit surfaces:

- `x2_controller/x2_rl_deploy_controller/src/policy_adapter.cc` and `velocity_adapter.cc`.
- `include/x2_rl_deploy_controller/policy_adapter.h` and the ball-provider interface.
- `src/hardware_observation.cc`, `src/policy_family_tennis.cc`, and existing tennis frame helpers.
- Existing evidence hooks and narrow hardware-logic tests.

Required behavior:

1. Recognize `ball_hist_pelvis` as a separate metadata-declared contract. Preserve legacy `ball_pos` behavior.
2. Preserve the exact observation layout: gyro 3 + gravity 3 + joint position 31 + joint velocity 31 + previous action 29 + command 3 + ball history 12 = **112**.
3. Keep head measurements in the observation even though the policy has only 29 action outputs.
   Do not derive every observation width from the action-joint count.
4. Store three accepted samples, newest capture first, as `[x, y, z, age]` in seconds and metres.
   Match mjlab duplicate rejection, ordering, padding and age clipping. Do not invent samples on every controller tick.
5. Convert each sample once into its capture-time pelvis frame. Old samples retain their stored values while their ages increase.
6. Keep receipt time and capture time separately. A clock restart clears/reinitializes the new history under an explicit rule.
7. Refuse new-contract engagement when required measurements or frame/time conversion are unavailable.
8. Preserve the current safety state machine, command ownership, limits and legacy provider's hold-last behavior.
   Do not silently introduce a global ball-age shutdown policy.

**Frame-conversion decision:** prefer an existing capture-time camera-to-pelvis transform.
If the chain moves and no synchronized transform exists, current-joint conversion is an approximation, not capture-time truth.
Measure its error first. Then choose either a verified fixed-chain setup or the smallest bounded head/waist timestamp buffer needed for conversion.
Such a buffer is internal to coordinate conversion; it must not add actor features or full-robot history.
If exact conversion is unavailable, document and simulate the approved approximation rather than hiding it.

**Checks:** recorded-input Python/C++ observation parity; nonzero head/waist poses; stale/reordered samples; clock restart;
112-input/29-output ONNX versus PyTorch numerical parity; previous-action order and gains/defaults/scales.
Normalize exactly once. Use plain model paths, iterations, metadata and numerical checks; no hash machinery.

**Exit condition:** the real controller constructs the same declared observation and command contract on recorded inputs.
A loader's `--check-config` success alone is insufficient.

### P2 — Add a small real-measurement training preset and simulation screen

Owning repository: `mjlab/`.

- Keep `tennis_juggle_nominal_env_cfg.py` and the E0 baseline unchanged.
- Add one small opt-in curriculum/preset factory beside it, reusing the existing pelvis-history task.
- Use existing reset, ball-stream and randomization terms. Do not copy the task or build a scheduler.
- Record the selected stage and explicit changed values in each run's saved configuration.
- Extend `scripts/experiments/x2_tennis_nominal_eval/` only for declared stage deltas and needed diagnostic channels.
- Add raw actions and commanded/measured joint response when diagnosing actuation. Do not infer torque saturation from targets alone.

The first preset models the selected robot's known operating conditions, not arbitrary broad randomization.
A nominal policy trained only on perfect observations is not automatically a candidate for a delayed-camera hardware trial.
If P0 shows a large known mismatch, screen or fine-tune that condition before requesting active robot evaluation.

**Critical tests only:** exact preset delta, unchanged nominal defaults, history/frame timing, terminal capture and scoring denominators.
Run the ONNX/controller observation path offline against recorded data where possible.
Native simulation checks qualified flights under both nominal and proposed conditions.
Vendor simulation checks control plumbing; unexplained vendor-simulator falls are not a substitute for real measurements.

**Exit condition:** no contract error; no known safety regression; simulation result and remaining mismatch are explicitly reported.
This authorizes no hardware action by itself.

### P3 — Establish the first real-robot policy baseline

Use one preselected 0.40 m checkpoint initially, rather than sending all three seeds through hardware trials.
Default tie-break: seed 7 final checkpoint, because all three passed E0. Record selection before trials.
The other two remain independent confirmation candidates, not silent substitutes after a failed trial.

Use the current [robot commissioning procedure](../../../x2_docker/REAL_ROBOT_DEPLOYMENT.md#13-command-producing-commissioning-sequence)
with the owner physically present. Confirm the actual installed version and applicability before following historical instructions.

Required safeguards:

- Approved fall protection, clear workspace, healthy hardware and tested physical stop/deadman control.
- Exactly one command owner. No competing vendor or custom controller.
- Current joint/IMU/inference watchdogs and gain/limit settings remain active.
- A local human can stop the trial; an SSH keyboard is not the sole emergency path.
- Inspect observation-only output before command-producing commissioning.
- Start with the existing short commissioning exposures. Increase duration only after the operator reviews each result.
- Define ball-loss, perception-invalid and repeated-carry abort conditions for the trial. Do not improvise a recovery action.
- Record manual support, fall-arrest load and operator intervention; assisted behavior is not an unassisted success.

Do not run the simulation zero-action controls on the real robot as an automatic baseline.
Their default-pose movement is not a hardware-safe no-policy condition.
Use an already accepted controller only for its known safe commissioning role.

**Exit condition:** a documented physical baseline or an explicitly classified failure, not necessarily successful juggling.
A failed first trial feeds diagnosis; it does not trigger automatic retraining or another attempt.

### P4 — Diagnose one physical failure and train one bounded correction

Choose the next training change from the failure table below.
Correct wiring, calibration and command-contract bugs outside the curriculum.
Keep target 0.40 m and the existing anti-loss behavior.
Do not re-enable a lift penalty just because carrying was observed in the separate 0.20 m experiment.

**Proposed initial training budget:** one seed, +2000 iterations, 4096 environments, one managed GPU; save every 500 iterations.
Use a new run directory and the correct additional-iteration resume semantics.
Run through a monitor with a wall limit based on measured iteration cost, progress records and exact owned cleanup.
An extension to +5000 total requires an explicit decision based on evidence.
Confirm a useful change on the other seeds before claiming a general training effect.
Do not run large sweeps by default.

**Exit condition:** candidate passes contract checks and the native simulation screen, with the changed condition and retention result reported.
Then request the bounded robot comparison in P5.

### P5 — Accept or reject on the real robot

Compare the candidate with the last physically accepted policy under the same robot setup and release protocol.
If there is no accepted juggling policy yet, report absolute progress against the initial physical baseline.
Keep trial order, release envelope and scoring fixed before testing; alternate candidate/baseline only when both are safe to run.

Simulation screening documents regressions and uncertainty; **physical acceptance decides promotion**.
If validated robot behavior contradicts the simulator, investigate model fidelity instead of optimizing only for the simulator.
A candidate that improves simulation but regresses the robot is rejected.
Keep the previous working checkpoint, controller configuration and calibration available for rollback.

## 4. Which curriculum condition comes next?

There is no fixed order beyond establishing a valid input/command path.
Use the earliest measured limitation, and change one condition at a time.

| Measured physical problem | First action | Training change only if still needed |
|---|---|---|
| Wrong or displaced ball point | Check frame, rectification, extrinsics, timestamp association and centre definition | Residual measured bias/noise, not the known calibration error |
| Late reaction or loss during observation gaps | Measure actual capture-to-inference age and gap bursts | Measured update rate, then latency, then stalls as separate substeps |
| Failure only during engagement or hand release | Compare real handoff pose, blending and release timing | Realistic start pose/release distribution; do not assume the ideal E0 reset |
| Racket lags or commands saturate | Compare intended and published targets, gains, measured q/qdot and effort semantics | Measured delay/gain/actuator uncertainty within the approved physical limits |
| Rebound differs despite comparable racket motion | Check ball/racket construction, inflation/wear and contact evidence | Measured contact-parameter uncertainty; no arbitrary global physics weakening |
| Nominal real juggling works, release position is fragile | Quantify the actual release spread | Lateral/height reset variation based on that spread |
| Carrying or tiny hops replace juggling | Inspect direct video/trajectory and reward incentives | Isolated anti-carry task change only after evidence and approval |
| External disturbance is the remaining requirement | Define a separate risk-reviewed test | Push training last; no unapproved pushes on hardware |

The old ±5 → ±10 mm simulation reset test remains a useful small diagnostic.
It is not the default priority before measuring the real robot's release and sensor conditions.

### 4.1 Proposed delayed-release training condition

**Operator requirement:** the human releases the ball only after policy engagement.
This requirement is confirmed. The first rung of this design is **implemented as
`S7a`**: the held ball, the gated energy reward, the waiting-alignment term, the
held-contact termination and the paused no-bounce deadline. The measured hold
height randomisation, the measured hand release and the no-release evaluation
cases are not implemented.

Use one policy and one continuous episode: **engage while held → align and wait → release → juggle**.
Sample a bounded release delay independently of policy behavior. Do not wait for policy-defined readiness in the training sampler.
Use varied delays, including immediate-release retention cases. A fixed countdown can teach a timed strike instead of observation-based control.
Keep the ball fixed in world coordinates during the initial hold, not attached to the current racket or pelvis.
Release without a position jump or artificial impulse. Validate the hold at physics-substep resolution.
Do not reset the robot, previous action or ball history at release.
Do not supply the actor with a simulator-only held flag or countdown.
The environment can use its hold state for reward/termination accounting, not as an extra deployment input.

At rest at target height, the existing energy formula already gives full reward.
That term is action-independent during an ideal hold; it does not teach alignment or safe waiting.
Keep its formula unchanged and report held-phase energy separately from post-release performance.
The existing `ball_centred` uses plate-frame x/y and has a 20 mm full-reward plateau.
The nominal ±5 mm lateral jitter stays inside that plateau when the racket is aligned.
Thus, those resets alone provide no reward difference for finer alignment.
Tilting the plate toward an elevated ball can also improve centering without placing the racket vertically below the ball.
For waiting, consider a gravity-horizontal alignment term, with level-racket and low-motion constraints.
Keep the racket below the ball with approved clearance. Reject attempts to strike or contact the held ball.
These constraints are not proof of hand safety; hardware release remains a separate supervised acceptance gate.

Pause the no-bounce clock during the hold and start it at release, without inventing an impact.
Keep robot safety and ball-loss fixes. Keep post-release anti-carry behavior unchanged.
Use a fixed post-release evaluation duration so longer waits do not shorten the juggling test.
Keep the same sensor latency/history model through release. The policy cannot respond to a release before the evidence arrives.

The nominal reset drops from 0.20 m; the energy target is 0.40 m.
To isolate delayed release, first compare immediate and delayed starts at the same height.
A 0.40 m held start is reasonable, but it also changes release energy; assess that change separately.
Then widen release offsets and add measured hand motion or occlusion one condition at a time.
Score waiting alignment, plate tilt/speed, held-ball contact, first-return success and subsequent qualified-flight streaks separately.
Include short, long and no-release waiting checks. Do not treat high held-phase reward as juggling success.

### 4.1.1 Implementation status, 2026-11-06

`S7a` is stage key `s7a-delayed-release`, task suffix `-Nominal-S7a-DelayedRelease`.
It holds the ball at a **fixed 0.40 m** and releases it after a uniform 0-2 s wait,
because 0.40 m is the energy target: the release energy is then `E*` and any change
is attributable to waiting rather than to a different drop energy. S5's whole
perception condition is carried, and the actor is never told that the ball is held.

The hold is a per-environment write of the ball's `qpos`/`qvel` on every control step,
not a MuJoCo weld: the scene runs thousands of environments with independent release
times and a weld is model-level. Measured on four environments with zero action:

- the pinned ball's observed position error is **0.0000 mm** and its speed
  **0.000000 m/s** (the pin is applied after the substeps and before the observation);
- the energy reward is **exactly 0** while held and **1.0000** at the release step;
- the release has a **0.0000 mm** position jump and **0.000000 m/s** velocity, so no
  impulse is imparted;
- the alignment reward is **0.93-1.00** while held (the plate starts aligned in the
  carry pose) and exactly 0 after release;
- at a forced 3.0 s hold, `ball_no_bounce` never fires, while the same term without
  the hold pause fires at **2.48 s** with the ball still held. That control is what
  shows the pause works; a 2 s hold alone is below the 2.5 s deadline and would pass
  either way.

The release also zeroes `episode_length_buf` for that environment, so every episode
gets the same post-release window regardless of how long the operator waited; without
it a 2 s wait both shortens the juggling opportunity and stays in the reward
denominator.

`S7b` (`s7b-variable-hold`, suffix `-Nominal-S7b-VariableHold`) then restores the height
range S6 had and keeps the hold: uniform 0.10-0.40 m above the resting state, everything
else S7a's, resuming an **S6** checkpoint so that S7b - S6 is the hold alone and
S7b - S7a is the height range alone. The floor is 0.10 m because a hold at 0.0 rests the
ball on the strings, which is a plate start rather than a hold an operator can perform.
The height is sampled once by the reset and the pin follows it, measured to 0.0 m
disagreement over six environments, and `ball_release_alignment_m` logs the plate-frame
offset at release so the waiting alignment is measured rather than watched.

`S7c` (`s7c-long-hold`, suffix `-Nominal-S7c-LongHold`) keeps S7b's held ball and
height range and lengthens the wait to a uniform 0-6 s, so S7c - S7b is the wait
length alone. The upper bound is a reward argument, not an episode argument: an
episode is 20 s and episode reward is normalised by that maximum, so a mean 3 s
wait still leaves 14 s of juggling and 70 % of the reachable return, while 10 s
would halve the ceiling and turn the rung into an endurance test.

`S8` (`s8-offset-hold`, suffix `-Nominal-S8-OffsetHold`) moves the placement. The ball is
put down `0.05 m/s x hold duration` from the nominal hold point along a random world
direction, held there, and released from there, so the plate must travel to it. The
distance is tied to the duration to keep the reach speed constant at 5 cm/s, which makes
the task equally hard at every wait length. Displacement is uniform 0-0.30 m, mean 0.15 m,
against a plate half-extent of 0.157 x 0.117 m. No reward term changes: the exponential
waiting kernel is deliberately left to reveal whether long-offset episodes collect
anything, and a bounded linear term is the fix if they do not.

`S9a` (`s9a-drop-tail`, suffix `-Nominal-S9a-DropTail`) changes what a lost ball costs.
The drop termination is removed and replaced by a continuous lost-ball penalty, so the episode
survives the loss and the robot is scored on the aftermath until `ball_no_bounce` ends it 2.5 s
after the last strike. The torso angular-velocity weight goes from -0.05 to -0.5 because a
measured probe showed it was worth 1/550th of the upright reward, and the motion that broke a
connector was a rate, not a tilt.

`S9b` (`s9b-sim-to-real`, suffix `-Nominal-S9b-SimToReal`) restores the full domain
randomisation the nominal pilot removed, using the definitions and ranges of the validated
`Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion` task: base COM, encoder bias, foot friction,
push, PD gains, inertia, armature, effort limits, joint friction, joint damping, foot size
and the wrist effort re-derate. `push_robot` supplies the disturbance. Play mode skips the
set. The stock encoder bias (+-0.01 rad) is smaller than the measured hardware error
(3.4 deg mean, 14.5 deg max), which remains an open item.

Open: no measured hand release exists yet, and the "no release" evaluation case is
not built.

## 5. Real-robot measurement and acceptance

### Evidence independent of the policy's ball estimate

Use synchronized external video as the primary physical outcome record, with a visible length/height reference.
A suitable high-frame-rate view is preferred; record actual frame rate and uncertainty.
Use a second view when a single projection cannot resolve plate return or ball height.
Do not grade the policy only with the same delayed or biased detector that drives it.

Record, using existing tools where possible:

- Trial start, release, end, all stops and invalid-trial reasons.
- Raw ball messages with capture/arrival stamps, accepted history samples and the observation consumed by inference.
- Policy identity, action sequence, published targets, gains, joint/IMU state, available effort signals and their semantics.
- External video, camera calibration, real initial ball-rest height and measured release pose.

A script should produce a small trial CSV and an annotated event/video record. Manual review is acceptable initially.
Do not build a new online estimator or automated video stack before it is needed.

### Physical outcome definition

A qualified sequence contains visible departures, apexes and returns to the racket without carrying, floor contact or operator assistance.
Measure the 0.40 m apex relative to the **measured physical rest datum**, not the simulator's numeric world height.
Also report rise above the actual release point to distinguish lifting from flight.

Use the E0 0.35–0.45 m apex band only when measurement uncertainty supports that test.
If height or return is obscured, mark qualification **unverified**, not passed.
Do not claim 1 ms contact-duration checks from ordinary video.
Report observed consecutive flights separately from height-qualified flights when needed.

### Proposed physical gate, to agree before trials

After short commissioning is accepted:

- Two separate blocks of five releases, at most 20 s per release.
- At least **4/5 releases in each block** achieve **10 consecutive qualified flights**.
- No robot fall, safety intervention, persistent carry or manual assistance in an accepted block.
- Report every attempt, ball loss and abort. An operator stop is not a successful timeout.
- Predefine invalid-release/recording criteria. Report invalid attempts separately; do not discard unfavorable valid trials.
- Require no regression in the prior accepted physical condition.

These are proposed engineering gates, not demonstrated reliability limits.
The operator must approve exposure durations, release envelope, abort criteria and physical measurement quality before command-producing tests.

## 6. Software and artifact boundaries

| Owner | Minimal deliverable | Done condition |
|---|---|---|
| `ESS/x2` / existing capture tools | Real sensor/calibration baseline; timestamp/frame evidence | Measured parameters and uncertainty, no command publication |
| `x2_docker` | Metadata-gated pelvis Ball3 adapter and evidence fields | Recorded-input parity, legacy behavior retained, correct 31-observed/29-actuated mapping |
| `mjlab` | Opt-in measured-condition preset and bounded stage runs | Exact saved deltas, unchanged nominal baseline, simulation retention evidence |
| Evaluation scripts + operator | Physical trial CSV/video score and candidate comparison | Declared trial denominator and independent outcome evidence |
| Parent/research owner | Acceptance decision and next single change | Report and NOW entry updated; no automatic promotion |

Critical tests cover frame/time history, contract widths, normalizer use, command mapping, stage delta and score boundaries.
No full repository test sweep, new manifest system, checkpoint hashes or automatic training/deployment loop is required.
Do not overwrite shared remote checkouts or previous run artifacts.
Do not change the controller during a policy comparison; report each implementation change separately.

## 7. Preserved E0 baseline

[Accepted E0](../reports/x2_tennis_nominal_evaluation_20261105.md):
three 0.40 m policies × 32 episodes, 34 consecutive qualified flights per episode.
The 0.20 m policies carried the ball. The two zero-action controls lost it.
All 256 episodes completed; all exit statuses were zero.
Execution: 12:30:12–12:34:27 (+08), 2026-11-05. The GPU was released.

The frozen simulation rule remains unchanged:

1. Two complete bounding racket contacts in the same episode; exclude the initial fall and unfinished final flight.
2. At least 0.15 s airborne, without flagged ground/non-racket contact in the interval.
3. Upward then downward vertical velocity.
4. Rise at least half the target height above the sampled takeoff reference.
5. Apex within target ±0.05 m above the fixed simulation rest datum.
6. Each bounding continuous contact at most 0.05 s.
7. Return inside the modeled plate surface in plate coordinates.

The simulation gate remains 26/32 episodes with at least 10 consecutive qualified flights and no loss, hold or fall.
It is a **simulation screen**, not the physical gate.
History/recording approximations and the simultaneous-contact limitation stay documented in the E0 report.
No threshold is changed retrospectively to improve a result.

## 8. Next action and authorization

**First implementation milestone:** P0/P1 — define the intended robot's measured input path and implement the compatible pelvis Ball3 adapter.
Plan the hardware measurement and scoring setup at the same time.
Do not launch R1 fine-tuning simply because the nominal simulator passed.

Open operator decisions: intended robot and head configuration; available external recording; physical release setup; approved stop/commissioning procedure.
Existing logs may answer some of these without a new robot session.

This plan authorizes no code change, recording session, training, package installation, robot command or physical trial.
Implementation approval and command-producing hardware approval remain separate.
