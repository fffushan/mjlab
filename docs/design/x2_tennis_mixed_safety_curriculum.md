# X2 tennis: mixed safety and visibility curriculum

Date: 2026-11-07 (+08).
Status: **plan only; this document does not authorize implementation or training**.
Scope: the next simulation fine-tune after the existing juggling stages.

This plan extends the [nominal curriculum](x2_tennis_nominal_curriculum.md).
It records the next step from the operator discussion.
It does not authorize changes to running jobs, queued jobs, the controller, or hardware.

## 1. Decision

**Start from a verified juggling checkpoint. Do not restart from random initialization.**

Preserve the existing skill. Add stable waiting and safe behaviour after a ball loss.
Keep these behaviours in the final training distribution.
Do not use dead-ball-only training followed by live-ball-only training as the main sequence.
The second stage can erase the first behaviour.

The required priority is:

1. Remain balanced, whether juggling is possible or not.
2. Use only ball information that the real sensor can provide.
3. Juggle when possible. Rescue is optional, not a requirement.

More qualified flights must not compensate for more robot falls in acceptance.
A simulation pass does not establish hardware safety.

## 2. Define the cases separately

| Case | Required behaviour |
|---|---|
| Playable ball with recent measurements | Attempt the juggle inside a safe workspace |
| Ball briefly outside the field of view (FOV) | Use recent history if useful; do not receive fresh hidden positions |
| Ball already on the ground | Remain balanced; do not chase an unreachable ball |
| No valid ball measurement since engagement | Stand and wait; do not invent a ball location |
| Long measurement loss | Settle safely when the available history no longer supports a strike |
| Ball loss during a juggle | Settle from the moving posture, then remain balanced |
| Ball reappears or a new ball is released | Resume the attempt when the observations support it |

A missing measurement does not prove that the ball is physically lost.
A ground-start test does not test the transition from juggling to standing.
Both cases need separate evaluation.

## 3. Select and freeze the starting checkpoint

Select the checkpoint by evaluation, not by its stage number or training reward.

- Evaluate S9c when its checkpoint is available.
- Use S9c if it preserves qualified juggling and improves posture.
- Otherwise, retain the best verified earlier juggling checkpoint.
- Keep S9d as a standing diagnostic branch.
- Use an S9d checkpoint as initialization only if it also passes the live-ball retention test.

Freeze the selected checkpoint and its resolved configuration before fine-tuning.
Record the checkpoint path, stage, training seed, observation contract, and evaluation cases.
Preserve the original checkpoint for matched comparisons.

Evaluate normal juggling, ground-start standing, and induced-loss aftermaths before training.
Use the same evaluation cases for the baseline and candidate.
Do not select a checkpoint from training reward alone.

## 4. Required changes before the mixed stage

### 4.1 Truthful ball measurements

The current `ball_history_samples` term has no visibility gate.
It forces the first acquisition, even when a real camera would have no measurement.

The new observation contract must follow these rules:

1. Check visibility at sample capture time, with the camera pose at that time.
2. Do not admit a fresh measurement when the ball is outside the usable view.
3. Permit a delayed visible frame to arrive after the ball leaves the view.
4. Retain accepted history and increase its capture-time ages during a measurement gap.
5. Represent never-seen startup explicitly. Zero position with zero age is not an invalid sample.
6. Keep stored positions in their capture-time pelvis frame.
7. Match age clipping, padding, validity, and reset rules between training and deployment.
8. Do not expose simulator-only visibility or loss labels as actor inputs.

The actor is the network that produces actions.
The training critic estimates value and may retain privileged simulator state.
Reward calculations may also use simulator state.

Match the usable stereo region, not only the approximate raw-camera frustum.
Document range limits, rectification, and any omitted occlusion effects.
The current policy has 29 actions and cannot command the two head joints.
The chosen live-ball region must fit the held camera pose.

The current actor input width is 112.
Choose either an explicit validity channel or a documented invalid-history encoding.
Do not silently change the input meanings of an existing checkpoint.
A wider input requires deliberate weight and normalization migration.
Do not assume that the old optimizer state remains compatible.
Verify unchanged-input behaviour after migration, before further training.
This plan does not require a recurrent network or a new estimator.

### 4.2 Continue after a loss

The current S9c configuration retains `ball_no_bounce` at 2.5 s after the last strike.
That limit does not provide a long standing interval after a drop.

For safety cases, continue after a loss. Do not reset on the ball event.
This rule applies to losses from live starts as well as ground starts.
Keep true robot-fall termination. Do not keep a fallen robot artificially active.

**Initial proposal:** provide at least 10 s of post-loss exposure when the robot remains upright.
Place induced losses early enough, or extend the episode budget explicitly.
A late drop followed by an immediate time-out does not satisfy this exposure requirement.

Audit `ball_no_bounce`, `ball_not_lifted`, and `held_ball_contact` for each case.
They must not discard the intended safety interval.
Keep the restriction on contact with an operator-held ball where that phase applies.
Specify the episode budget and time-out bootstrapping before launch.
Do not mislabel a task failure as a time-out to change its return.

### 4.3 Minimal reward changes

- Keep balance, body-rate, joint-limit, and action-smoothness terms active throughout.
- Remove the per-step lost-ball cost after an unrecoverable loss.
- If needed, charge one event when a playable ball becomes physically lost.
- Do not charge that event for a ground start or a never-seen startup.
- Do not charge repeated loss events while the same ball remains lost.
- Do not treat a brief FOV exit as a physical loss event.
- Preserve juggling rewards for playable-ball cases.
- Do not reward pursuit of an unreachable ground ball through FOV or alignment terms.
- Do not add a large bonus merely for entry into the lost-ball state.

Keep unrelated reward weights and physics ranges fixed for the first mixed trial.
Do not double all ball rewards or strengthen every pose term at the same time.

Proximal Policy Optimization (PPO) receives `sum(raw * weight * dt)` at each step.
The episode-length division belongs to logging, not the PPO reward.
A per-step negative term does not, by itself, prove that falling is preferable.
Compare complete discounted settling and falling trajectories, with actual terminal semantics.
Arithmetic changes to a transient reward table do not measure a quiet-standing return.

## 5. Initial mixed distribution

These shares are a **starting proposal, not validated proportions**.
They refer to episode starts, not training steps.

| Episode type | Proposed share | Purpose |
|---|---:|---|
| Normal, initially visible juggling | 70% | Preserve the existing skill |
| Ground ball or no valid measurement at engagement | 15% | Learn stable waiting |
| Juggling followed by an induced loss | 15% | Learn the transition into stable standing |

Sample both ground-ball and never-seen cases within the 15% waiting group.
Log their results separately. Specify their internal split before launch.
Normal juggling episodes also need the post-loss interval when an unplanned drop occurs.

Initially, place playable balls comfortably inside the usable view.
Induce losses with a bounded ball perturbation beyond the safe racket reach.
Do not require a successful rescue as acceptance for this group.
Randomize the loss time within the available episode budget.

Log actual training-step shares as well as episode-start shares.
Long standing episodes can dominate the batch despite a small episode share.
Adjust the mixture from separate safety and juggling measurements, not total reward.

## 6. Increase difficulty without making the sensor dishonest

Keep the truthful measurement contract from the first mixed trial.
Make the physical situations easier initially; do not restore hidden measurements.

Increase difficulty in this order:

1. Easy visible juggling, ground starts, never-seen starts, and simple induced losses.
2. Brief measurement interruptions where recent history can still support a strike.
3. Longer interruptions where stable waiting is acceptable.
4. Reappearance or new release after a waiting interval.

Retain the earlier cases at every stage.
Do not train the final stage only on live balls.
A live-only retention check is evaluation, with no PPO updates.

Change one difficulty axis at a time after the initial contract and safety changes.
Keep observation timing and physics ranges fixed unless the stage explicitly changes them.
Log every change to the resolved configuration.

## 7. Measure the raised-racket shortcut separately

The reported overhead behaviour is a visual observation, not a measured racket-height result.
Add racket-height and free-flight-rise metrics before another long fine-tune.

Keep the existing absolute ball-energy target as guidance.
It does not constrain the racket workspace or guarantee useful free flight.
The energy sigma is 1.5 J/kg, equivalent to about 0.153 m of height.

Evaluate three properties separately:

- Absolute ball apex relative to the fixed rest datum.
- Free-flight rise above the ball separation height at a valid racket strike.
- Racket position inside an approved workspace, with head clearance.

Freeze the strike reference at separation.
Do not let later racket motion manufacture apparent free-flight rise.
Use qualified flights, not raw contact counts, for skill acceptance.

A later controlled change can add a workspace limit or qualified-flight reward.
Do not combine that reward study with broad changes to the first safety trial.
Do not claim that visibility masking alone removes the raised-racket shortcut.

## 8. Validation and acceptance

### Software checks

- An unseen reset produces no fabricated fresh ball sample.
- Hidden ball motion cannot alter the actor measurement history without a valid acquisition.
  Hold proprioception and prior measurements fixed for this test; exclude physical contact effects.
- Delayed frames obey capture-time visibility and the existing capture-time order.
- History ages increase during gaps, and reset clears prior-episode history.
- Ground starts and induced drops receive the configured standing interval unless the robot falls.
- An optional loss charge fires once for a real loss, not for reset or repeated lost steps.
- Observation migration preserves the declared layout and unchanged-input behaviour.

### Behaviour checks

| Evaluation case | Report separately |
|---|---|
| Ground ball at engagement | Falls, torso tilt/rate, settling time, arm motion |
| Never-seen or long-unseen ball | Falls, bounded actions, stable waiting duration |
| Induced loss during a juggle | Falls after loss, settling time, completed standing interval |
| Brief measurement interruption | Qualified flights, falls, successful reacquisition |
| Normal juggling | Qualified flights, streak length, apex, free-flight rise, visibility fraction |
| Reappearance or new release | Stable waiting before detection, successful return to juggling |
| Every applicable case | Racket height and workspace violations |

Freeze numeric acceptance limits before training.
Specify the test count, seeds, duration, allowed juggling regression, and safety thresholds.
Use separate gates for safety and skill; do not average them into one score.
Report falls as counts with denominators. Report elapsed exposure and time-out censoring.
A finite test with zero falls is not a safety guarantee.

Reject or revise a candidate that loses juggling while it improves standing.
Also reject a candidate that improves juggling while it increases falls or workspace violations.
Do not change acceptance thresholds after candidate evaluation.

## 9. Next authorized decision

The operator authorizes documentation only at this point.
No new implementation, simulation run, training job, deployment, or robot operation follows automatically.
This plan does not stop or alter an existing authorized S9c/S9d job.

Before an implementation or run request, decide:

1. The baseline checkpoint and matched evaluation configuration.
2. The validity encoding, age rules, and migration method.
3. The usable camera region and known geometry limitations.
4. The episode budget, post-loss interval, and optional loss-event cost.
5. The mixture, numeric acceptance limits, and bounded pilot budget.

Then request implementation and focused software checks.
After those checks, request a bounded mixed fine-tune and separate evaluation.
Hardware acceptance remains a later, separately authorized activity.

## References

- [S9c reward items and audit](x2_tennis_reward_items.md).
- [Nominal curriculum and hardware acceptance](x2_tennis_nominal_curriculum.md).
- [Pelvis-relative ball history](x2_tennis_pelvis_ball_history.md).
- [Juggling task guide](../source/x2_tennis_ball_juggle.rst).
- [Stage configuration](../../src/mjlab/tasks/velocity/config/agibot_x2/tennis_juggle_stage_env_cfg.py).
- [Ball observation implementation](../../src/mjlab/tasks/velocity/mdp/observations.py).
- [Reward implementation](../../src/mjlab/tasks/velocity/mdp/rewards.py).
- [Reward manager and reward units](../../src/mjlab/managers/reward_manager.py).
- [Existing evaluation runner](../../scripts/experiments/x2_tennis_closed_loop/run_sweep.py).

This plan follows the controlled-English rules of this repository.
This review does not verify official ASD-STE100 word-level compliance.
