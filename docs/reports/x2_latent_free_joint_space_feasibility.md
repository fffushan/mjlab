# Latent-free joint-space flow: change-scope review

Observation date: 2026-10-21 (+08).
Source: `fxy/test/flow-matching-baseline`, collector commit `1035b4faf146abab79636e8cf1a24876851271d8`, plus the existing working-tree changes.
Status: raw collection implementation and a bounded remote simulator smoke are complete. The flow model and rolling-reference adapter are not implemented.
The operator authorized this narrow research collector, its tests, and the bounded smoke. No training or deployment was run.

## Selected direction: generated pose reference plus frozen tracker

Operator clarification on 2026-10-21 (+08): predict future achieved root orientation, joint positions `q`, and joint velocities `dq`.
Use the existing frozen multi-motion tracker to follow that generated reference.
Exclude the existing 135-D root/body physical-state representation because real-robot estimation is inadequate.
Do not predict servo targets. Do not replace the tracker with a non-learned PD controller.
Defer DiT and conditioning-mechanism experiments.
The VAE encoder and decoder are absent from this hierarchy:

```text
flow -> future root orientation + q + dq
     -> reference adapter (root-to-torso FK, align time/frame, supply preview)
     -> frozen multi-motion PPO tracker + live robot observations
     -> existing position commands and servo
```

The tracker retains learned feedback from fresh robot observations.
Thus the direct-command concern about removing all learned correction does not apply to the selected architecture.
This supersedes the recommendation in [the DiT review](x2_dit_flow_matching_feasibility.md) and the direct-command alternatives below.

### Verified tracker inputs

Inspected local artifact: `logs/diagnostics/x2-multi-motion-deploy-20261015/policy.onnx`.
Metadata inspection only: `obs[1,226] -> actions[1,31]`, `anchor_body_name=torso_link`, `motion_lookahead_steps=10`.
No inference or simulation was run.

| Actor term | Width | Required data |
| --- | --- | --- |
| `command` | 62 | Reference q and dq at the selected reference time |
| `motion_lookahead` | 62 | Reference q and dq 10 frames later |
| `motion_anchor_ori_b` | 6 | Reference torso orientation relative to the live robot torso |
| `base_ang_vel` | 3 | Live pelvis-IMU angular velocity |
| `joint_pos` | 31 | Live policy joint-position observation |
| `joint_vel` | 31 | Live policy joint-velocity observation |
| `actions` | 31 | Previous tracker action |

At 50 Hz, the preview offset is 0.2 seconds.
The actor does not consume reference anchor position or base linear velocity.
This does not remove body/position data requirements from training rewards, terminations, viewers, or future goal-control tasks.

Sources: [reference observations](../../src/mjlab/tasks/tracking/mdp/observations.py),
[motion command](../../src/mjlab/tasks/tracking/mdp/commands.py),
[X2 configuration](../../src/mjlab/tasks/tracking/config/agibot_x2/env_cfgs.py),
[reference bundle](../../src/mjlab/tasks/tracking/rl/reference_bundle.py).

### Orientation requirement

Joint angles do not specify the floating-base orientation.
For example, the same joints can belong to an upright, tilted, or turned robot.
The tracker needs full reference torso orientation, including roll, pitch, and yaw.

The operator selects pelvis/root orientation prediction, not independent torso orientation prediction.
Compute reference torso orientation from the predicted root orientation and joint q using forward kinematics from the same robot model.
Do not independently generate a second orientation for the torso.
For root-to-torso kinematics `R_root_to_torso(q)`, use:

```text
R_ref_torso = R_ref_root * R_root_to_torso(q_ref)
R_relative = transpose(R_live_torso) * R_ref_torso
```

`motion_anchor_ori_b` packs the first two columns of `R_relative` in the source's reshape order.
Use full orientation, not heading alone.
If the flow predicts orientation in a plan-local frame, retain that frame with the plan and convert both sides consistently.
Do not substitute the current live orientation for the planned reference each tick; that would erase the intended orientation error.

### Selected compact representation

Use a 6D rotation representation for the predicted root orientation:

| Channels | Quantity | Source of measured history |
| --- | --- | --- |
| 0:6 | Root orientation, 6D rotation | Pelvis IMU orientation, with sensor-to-root calibration |
| 6:37 | Joint q, 31 values in radians | Joint encoders in the tracker joint order |
| 37:68 | Joint dq, 31 values in radians/second | Joint velocity measurements |

The proposed model window is 41 by 68: 8 history rows, the current row, and 32 future rows.
All 68 current-state channels are measurements, not unknown commands.
Future pose rows are generated. Reference execution time and current/future alignment remain explicit design work.
The old 167-channel token and the proposed intermediate 166-channel physical-state-plus-q token are not used.
There are no model channels for root position, root linear/angular velocity, body positions, or body velocities.
The tracker still reads its own live IMU gyro and previous actions; these are not added prediction targets.
The Transformer block structure can stay unchanged; input/output projections and representation-specific packing must change.
Define the new noise-level grouping explicitly rather than preserving an imaginary latent block.

Recommended frame convention: use a gravity-aligned local heading frame fixed at each plan's creation.
This retains roll/pitch and relative yaw changes without requiring absolute world position or globally accurate heading.
Store that frame with the plan and use it consistently during consumption.
IMU attitude, calibration, yaw drift, timestamp alignment, and encoder quality still require validation.
This removes translational/body-state estimation, not all sensor uncertainty or all attitude estimation.

The operator chooses to predict dq directly. No numerical differentiation is needed to provide reference dq.
Check generated q/dq temporal consistency separately; accurate per-channel prediction does not enforce it automatically.
Keep enough plan horizon for the 10-frame preview and the consumed prefix.
Use one plan revision for current q/dq and preview q/dq.
Predicted reference dq remains an input to the tracker, not a velocity command to the actuator.

### Next engineering checks

- Record achieved root orientation, q, and dq from the teacher. Keep policy-visible observations distinct from true simulator targets.
- Build the generated-reference adapter while leaving the teacher weights, observation normalizer, action conversion, gains, and servo unchanged.
- Preserve reference time continuity during replanning. Define current/future indexing and stale-plan behavior before choosing sampler masks.
- The exported actor has no embedded reference table, so external references can enter through the observation builder. The current runner still needs a rolling-reference path; the static bundle writer is not that path.
- First replay recorded achieved root/q/dq trajectories as references into the frozen tracker. Validate root-to-torso orientation and reference timing before using generated trajectories.
- Tracking achieved trajectories is a distribution change from tracking the original motion references. Feasibility and tracking quality are not established by artifact loading.
- Raw collection can run with a separate explicit rollout budget. Validate the reference adapter before claiming that the frozen tracker can follow generated trajectories. Training also needs an explicit budget.

## Raw collection implementation — 2026-10-21 (+08)

Implemented [collector](../../scripts/experiments/joint_pose_collection/collect.py),
[usage guide](../../scripts/experiments/joint_pose_collection/README.md), and
[focused tests](../../tests/test_joint_pose_collection.py).
One deepseek-v4.1-flash worker with max thinking implemented the collector in an isolated worktree.
The parent reviewed and integrated it, adding a finite-pose write check and three regression cases.

- One batched first-episode rollout per explicit clip; num_envs defaults to64.
- Full-clip step budget by default; `--max-steps` can shorten it. Existing output directories are refused.
- Reuses the frozen multi-motion teacher loader and saved preview. No VAE is loaded.
- Writes world-frame root quaternion (wxyz), actual joint q/dq, ticks, times, and trajectory boundaries/outcomes.
- Tick0 is the reset state before the first action; tick k is the achieved state after k control steps.
- Terminal states are recorded before manual reset. Later episodes on a finished row are not appended.
- Failed finite trajectories remain labeled failure; step caps are labeled budget, not success.
- One NPZ per clip and one small manifest. No heading alignment, 6D conversion, windows, statistics, cache, or training framework.
- Actor corruption remains enabled; `push_robot` is removed, matching the reused evaluation setup.

Validation in the primary checkout: **8 CPU tests pass in 2.19 s**. Scoped Ruff format/lint and Ty pass.
The CLI help renders without constructing a simulator. Parent monitors 1 and 2 exit 0.
Evidence: [primary log](../../logs/diagnostics/joint-pose-collector-implementation/primary-validation.log)
and [worktree log](../../logs/diagnostics/joint-pose-collector-implementation/parent-validation.log).
Commit `1035b4faf` contains the four collector milestone paths. Original tracked dirty changes were compared byte-for-byte and preserved.

### Original collection distribution

This section describes the original 144-motion run at commit `1035b4faf`, before the coverage extension below.
The collector loads the training task (`play=False`), then fixes the initial reference frame to0.
`sampling_mode="start"` does not remove reset perturbations.
Each environment starts from the clip's first root pose, root velocity, q, and dq.
Uniform reset offsets apply to root position, root orientation, root velocity, and q:

| Quantity | Additive range |
|---|---|
| Root x/y; z | ±0.05 m; ±0.01 m |
| Root roll/pitch; yaw | ±0.1 rad; ±0.2 rad |
| Root linear velocity x/y; z | ±0.5 m/s; ±0.2 m/s |
| Root angular velocity roll/pitch; yaw | ±0.52 rad/s; ±0.78 rad/s |
| Joint q | ±0.1 rad, then clipped to soft limits |
| Joint dq | Reference value; no separate reset noise |

Standing starts and random initial phases are disabled.
Seed0 is reapplied per clip, so this is not a multi-seed collection.
The64 environment rows receive different random draws within each clip.

The inherited task retains model randomization, encoder bias, sensor noise, sensor delay, and actuator delay.
Model randomization includes inertia, center of mass, friction, armature, effort limits, damping, foot size, and servo gains.
The remote GPU5 log confirms the startup and reset event lists for this run.
The collector removes periodic `push_robot` events and adds no action noise or difficult-state replay.
Saved q/dq and root orientation are achieved simulator values, not noisy policy observations.

Source: [collector configuration](../../scripts/experiments/joint_pose_collection/collect.py),
[X2 task](../../src/mjlab/tasks/tracking/config/agibot_x2/env_cfgs.py),
[reset implementation](../../src/mjlab/tasks/tracking/mdp/commands.py), and
[base randomization](../../src/mjlab/tasks/tracking/tracking_env_cfg.py).
This distribution does not establish broad recovery coverage.

### Remote simulator smoke

The smoke ran on the 8-card host `10.14.64.37:31505` at **2026-10-21 (+08)**.
It used side branch `x2-raw-joint-pose-smoke-20261021` and worktree
`/home/fushan/mjlab-joint-pose-smoke-20261021`.
The remote main checkout stayed on `9efad827d` with its prior dirty paths.

- GPUs 5, 6, and 7 ran clips `single_000`, `single_001`, and `single_002`.
- Each process used 8 environments and a 32-step budget.
- The checkpoint was `model_19999.pt` from the frozen multi-motion tracker.
- All three processes exited with status 0.
- Each manifest is complete. Each NPZ has 264 rows and 31 joint positions and velocities.
- Each NPZ has finite root quaternion, joint position, and joint velocity values.
- Each run has 8 `budget` trajectories. No `complete` outcome is claimed because the smoke stopped at 32 steps.
- Each manifest preserves the saved `lookahead_s=0.2` value.
- GPUs 5, 6, and 7 returned to 1 MiB and 0 percent utilization.

This validates checkpoint loading, task construction, saved preview wiring, achieved-state extraction, NPZ writing, manifest completion, and cleanup.
It does not validate full-clip tracking quality, failure rates, throughput, data coverage, or generated-reference tracking.
Evidence is under `/home/fushan/mjlab/logs/diagnostics/x2-joint-pose-collection-smoke-20261021` on the shared NAS.

The operator subsequently authorized all144 `single_*` clips on GPUs5–7.
The launched budget uses64 environments per clip, one first episode each, with seed0 and full-clip step caps.
This is9216 trajectories, not a budget matched to the historical latent pipeline.
Output: `/home/fushan/mjlab/logs/diagnostics/x2-joint-pose-collection-full-20261021`.
Monitor4 was stopped at the operator's request; this did not stop collection.
Full-run completion and data quality have not been audited here.
Preprocessing, training, and broader disturbance collection remain separate work.

## Raw collection coverage extension — 2026-10-21 (+08)

**Implemented and integrated locally; not committed.** Workflow5 extends the same four collector paths above.
The frozen teacher, action conversion, servos, preview, and production task modules are unchanged.

| Control | Default | Behavior |
|---|---|---|
| `--num-envs` | `64` | Trajectories per clip and seed |
| `--seed`, `--num-seeds` | `0`, `1` | Consecutive seeds from `seed` to `seed + num_seeds - 1` |
| `--sampling-mode` | `start` | Select `start` or independent uniform reference-frame sampling |
| `--push-interval-s` | `None` | No pushes, or an explicit uniform interval range in seconds |
| `--push-linear-velocity-range` | `-0.5 0.5` | Per-push delta range for each world x/y/z velocity, in m/s |
| `--push-angular-velocity-range` | `-0.52 0.52` | Per-push delta range for each world angular velocity axis, in rad/s |

Planned trajectories equal `motion_count × num_envs × num_seeds`, per scenario.
Each seed is reused across clips within its seed run.
Uniform sampling can select the same initial frame for multiple rows.
Reset perturbations remain enabled, and standing starts remain disabled.
A random-phase trajectory stops at its own clip endpoint or first failure/budget boundary.

Schema2 retains the original raw arrays and adds run, seed, scenario, initial-frame, and push count/tick provenance.
A baseline run has no periodic pushes; model and sensor randomization remain enabled.
A disturbed run enables the existing simulated root-velocity push, not an external force or random action command.
Each output directory has one scenario, and the manifest identifies every file.
Existing output directories are still refused.

`push_tick=k` means the event fired after tick k's physics integration.
Its response in saved root orientation/q/dq can first appear at tick k+1.
Terminal-tick and budget-tick pushes are counted, but no later recorded sample contains their response.
Push magnitudes are not stored; configured ranges, realized counts, and event ticks are stored.

### Review and validation evidence

- The implementation worker used `lz-fast/deepseek-v4.1-flash:max` in an isolated worktree.
- The first reviewer failed at provider prompt admission and produced no review.
- The operator selected `ds-oc/deepseek-flash:max` for the replacement review.
- The replacement review found no P0/P1 defect and returned `OK with notes`.
- The parent fixed all three P2 findings: a tautological start-mode assertion, push-timing documentation, and a sampling-uniqueness overstatement.
- The parent also corrected the interval help text: it describes a sampled range, not a mean.
- Candidate validation: **14 CPU tests passed in 4.71 s**.
- Primary validation: **14 CPU tests passed in 2.35 s**; Ruff format/lint, Ty, Pyright, CLI help, and whitespace checks passed.
- These tests use fake environments. The reviewer inspected source but had no execution tool; the parent ran the checks independently.
- Unrelated primary tracked changes matched the pre-task diff byte-for-byte after integration. No files were staged or committed.

Evidence: [primary validation](../../logs/diagnostics/joint-pose-coverage-implementation/primary-validation.log),
[candidate validation](../../logs/diagnostics/joint-pose-coverage-implementation/candidate-validation.log),
[independent review](../../logs/diagnostics/joint-pose-coverage-implementation/independent-review.md), and
[accepted source hashes](../../logs/diagnostics/joint-pose-coverage-implementation/accepted-source.sha256).
The accepted four-file patch SHA256 is `905581b729c03c7984fc3258a1d04dc6eb63e025488f2c5b02a1bf797566e593`.

### Limits and next action

At local acceptance, real random-phase sampling, interval-event integration, and push effects were **not tested**.
The earlier smoke validates the original collector only, not this extension.
No remote job, data collection, preprocessing, training, deployment, or monitor restart occurred during that integration.

### Authorized coverage smoke — PASS

The smoke ran on `10.14.64.37:31505`, GPUs 5–7, at **2026-10-21 16:33:43–16:34:48 (+08)**.
All three collectors exited 0. Driver wall time was **64.893 s**, within the 600 s budget.
[Acceptance and limits](../../logs/diagnostics/x2-joint-pose-coverage-smoke-20261021T1630/ACCEPTANCE.md),
[artifact audit](../../logs/diagnostics/x2-joint-pose-coverage-smoke-20261021T1630/artifact-audit.json), Workflow 6, Monitor 6.
Each lane used 2 clips, 2 seeds, 8 environments, and a 64-step cap.

| GPU | Phase / scenario | Rows | Complete | Failure | Budget | Pushes |
|---|---|---:|---:|---:|---:|---:|
| 5 | start / baseline | 2080 | 0 | 0 | 32 | 0 |
| 6 | uniform / baseline | 1797 | 6 | 2 | 24 | 0 |
| 7 | uniform / disturbed | 2009 | 3 | 1 | 28 | 342 |

Total: **96 trajectories, 5886 rows, 12 NPZ files**. All manifests are complete.
All raw states are finite. The 3 physical failures retain terminal states and failure labels; they are not collector errors.
The read-only artifact audit passed **1144 checks**, including offsets, ticks, seed/run identity, endpoint alignment, and preview `0.2 s`.
Uniform starts span frames 1–443 on GPU 6 and 0–409 on GPU 7. Start-mode frames are all 0.
All 32 disturbed trajectories have push events. Of 342 events, 339 have a later sample; 3 occur at the final tick.
Push intervals are 0.08–0.12 s, with each linear/angular velocity delta axis at −0.2 to +0.2 m/s or rad/s.
These settings test the event path; they are not a selected full-collection distribution.

Source ran from an isolated worktree at `1035b4faf` plus the accepted patch above.
Source, checkpoint, parameter, motion, and fetched artifact hashes passed.
Remote main branch/HEAD/patch/status and the local tracked patch remain unchanged.
At **16:36:01 (+08)**, no owned process remained; GPUs 5–7 each had 1 MiB and 0% utilization.
GPUs 0–4 retained their unrelated compute-process IDs. Old collection artifacts were not changed.

**Engineering smoke pass only.** Push magnitudes and recovery quality are not measured.
The uniform lanes have different initial frames; their failure counts do not measure the effect of pushes.
No larger collection, preprocessing, training, commit, push, or old-monitor restart occurred.
At smoke acceptance, larger collection remained unauthorized. The first production batch was subsequently authorized below.

### First 12-motion collection batch — accepted with limits

The operator approved the first 12-motion batch, not the remaining 132 motions.
[Plan, selection, and launch receipt](../../logs/diagnostics/x2-joint-pose-batch12-20261021T1700/PLAN.md).
Workflow 7 and Monitor 7 launched it at **2026-10-21 17:01:43 (+08)** on GPUs 5–7.

- Selected IDs: `000, 007, 021, 025, 034, 038, 050, 054, 091, 099, 114, 149`.
- Selection spans three length groups and four joint-velocity activity ranks in each group.
- Each of three groups uses 12 clips, 128 environments, and seeds 100–103.
- GPU 5: start/baseline; GPU 6: uniform/baseline; GPU 7: uniform/mild disturbance.
- Disturbed intervals: **1–2 s**. Per-axis linear/angular velocity delta: **±0.2 m/s or rad/s**.
- Full remaining clips, with no shortened step cap: **18,432 trajectories and 144 NPZ files** planned.
- The 1,800 s wall deadline and 300 s no-output-progress guard bound execution. They are not runtime estimates.
- Frozen teacher, source hashes, local/remote motion hashes, and import provenance passed before launch.
- Collection finished at **17:11:14 (+08)**, with driver and all three collectors exiting 0 in **571.048 s**.
- All 144 NPZ files contain **18,432 trajectories and 3,937,938 finite rows**, confirmed by both full scans.
- Source/input hashes and remote main snapshots passed after execution.
- At **17:13:09 (+08)**, owned processes were gone and GPUs 5–7 each had 1 MiB and 0% utilization.
- Final sampled GPU memory peaks were **3,851 MiB each**. These are sampled values, not exact allocation peaks.
- Monitor 8 passed **132,924 artifact checks** in **52.584 s**, with all raw states finite.
- NPZ size: **1,089,694,136 bytes**. Length-based 41-frame eligibility: **3,232,450 candidate windows**, not independent or necessarily successful samples.
- Physical failures by group: **3 / 71 / 83** of 6,144 trajectories each. No budget outcomes remain.
- Disturbed group: **9,754 push events**, with **7,721** having the full 8-history/current/32-future context.
- Independent audit **PASS_WITH_LIMITS**, with no P0/P1 finding and no required data correction.

The requested fresh **`ds-oc/deepseek-flash:max`** auditor worked at **17:18–17:27 (+08)**.
It independently scanned **all 144 NPZ files**, passed **7,912 vectorized assertions**, and reproduced the parent aggregates.
It checked source semantics, push spacing, phase evidence, hashes, GPU memory samples, and cleanup.
[Independent report](../../logs/diagnostics/x2-joint-pose-batch12-20261021T1700/independent-audit.md),
[parent acceptance and finding disposition](../../logs/diagnostics/x2-joint-pose-batch12-20261021T1700/ACCEPTANCE.md).

Important limits:
- Failures concentrate in motions 021, 050, and 054. The worst observed cell is disturbed motion 050: 24/512 (4.69%).
- Early-frame sparsity applies to the uniform groups. The combined batch already contains 6,144 frame-zero trajectories.
- 1,649 disturbed trajectories have no push. Of 9,754 events, 2,033 lack 32 future samples, including 72 final-tick events.
- Candidate windows include 1,864 from failed trajectories. Shape eligibility is not successful tracking or recovery.
- Realized push deltas and source-group split identities are absent. Source joint-order mapping remains an input precondition.
- Sampled memory rises and falls; the audit does not establish a leak or prove a longer single process will fit.

**Accepted as a raw collection pilot only.** No preprocessing, normalization fitting, window materialization, or training occurred.
The remaining 132 motions require a new decision. Prefer the same bounded 12-motion process size if expansion is authorized.
No source change, full 144-motion continuation, commit, push, or old-monitor restart occurred.

### Non-destructive eligibility gate — accepted; batch12 index built

The operator approved an eligibility gate and requested one implementation subagent.
[Acceptance, application, and limits](../../logs/diagnostics/joint-pose-eligibility-gate-implementation/ACCEPTANCE.md),
[brief](../../logs/diagnostics/joint-pose-eligibility-gate-implementation/brief.md), Workflow 8.
A single `lz-fast/deepseek-v4.1-flash:max` worker wrote the candidate in an isolated worktree; the parent reviewed it, fixed concrete gaps, integrated it, and applied it.

The gate accepts only complete trajectories with finite raw states and at least 41 rows.
It rejects whole failed/budget trajectories and whole trajectories containing non-finite required states.
It keeps complete disturbed trajectories, including zero-push cases.
Output is a compact, hash-bound eligibility index with retained/rejected counts and half-open window-start ranges. Raw NPZs are unchanged.

Parent review added checks the worker omitted, mainly joint-order and control-timing agreement across input manifests.
On a second pass nine of the eleven added checks were removed: they guarded conditions the collector cannot emit (`fps` versus `control_dt_s`, `file_name` versus motion path, motion identity across seeds, budget-source consistency, start-phase frame 0, endpoint overrun, strict push-tick ordering/length). Only cross-manifest compatibility remains, because it matters when independent batches are gated together.
Final: **48 focused CPU tests pass (34 gate + 14 collector)** in 2.54 s, plus scoped Ruff, Ty, Pyright, and whitespace checks.
Re-running the trimmed gate on batch12 produced an identical index value-set and per-file hash list.

Batch12 application (CPU only, **4.15 s**):

| Group | Trajectories | Failed | Too short | Retained windows |
|---|---:|---:|---:|---:|
| start/baseline | 6,144 | 3 | 0 | 1,731,260 |
| uniform/baseline | 6,144 | 71 | 766 | 771,258 |
| uniform/disturbed | 6,144 | 83 | 906 | 728,068 |
| **Total** | **18,432** | **157** | **1,672** | **3,230,586** |

Exact match to the audited prediction: 3,232,450 candidate, 1,864 rejected, **3,230,586 retained**, 157 `outcome_non_complete`, 0 `nonfinite_state`.
All 147 raw artifact hashes passed before and after, so raw data is byte-unchanged.
Index: `eligibility.json`, 5,051,610 bytes, SHA256 `7517c7b9486e046e621aede8f9949eb3fc4dd854a57003ecb766a9dd60bd0a82`, under `/home/fushan/mjlab/logs/diagnostics/x2-joint-pose-batch12-eligibility-20261021/`.

Limits:
- The gate is not wired into any trainer; a future preprocessing stage must honor `status` and `window_start_range`.
- The existing `failure` label aggregates task terminations (reference-relative torso height, a torso tilt proxy, ankle/wrist height). It is not a ground-contact fall detector, and the gate adds no new threshold.
- No window materialization, normalization, statistics, training, commit, or push occurred.

The gate is integrated locally but **uncommitted**. Gating later collections, building windows from the index, and training each require a new decision.

## Historical alternatives: state or direct command

The remainder records the earlier alternatives. Direct servo-command changes are not required for the selected frozen-tracker hierarchy.

`q_measured, dq_measured` describe achieved robot motion.
`q_target, dq_target` describe commands applied to a joint servo.
These pairs are not interchangeable.
A teacher deliberately sets a position target away from the measured position to produce torque.
Predicting the next measured pose does not reproduce that teacher's command.

The current path is:

```text
flow -> generated latent z_t
     -> frozen VAE decoder(z_t, observed conditioning_t)
     -> normalized action a_t
     -> position target
     -> position servo
```

A direct command policy would use:

```text
flow -> q_target_t, dq_target_t -> joint servo
```

A measured-trajectory model would instead predict future `q_measured, dq_measured`.
It needs an explicit execution rule that converts the prediction into servo commands.
A non-learned proportional-derivative (PD) tracker is one option, but its behavior requires simulation validation.

## Existing control interface

The X2 configuration uses `JointPositionActionCfg` and `BuiltinPositionActuatorCfg`.
The action conversion first applies scale and offset, then optional clipping.
The joint action also compensates encoder bias before setting the simulated target.
`ActionContract.to_physical` records only the affine scale/offset conversion.
Do not confuse that intermediate value with the final target under every randomization setting.

Sources:

- [actions.py](../../src/mjlab/envs/mdp/actions/actions.py), `BaseAction.process_actions` and `JointPositionAction.apply_actions`.
- [X2 constants](../../src/mjlab/asset_zoo/robots/agibot_x2/x2_constants.py), `_actuator`.
- [vendor gains](../../src/mjlab/tasks/tracking/config/agibot_x2/vendor_gains.py), position-actuator requirements.
- [action conversion](../../src/mjlab/tasks/tracking/diffusion/adapter.py), `ActionContract`.

The position actuator consumes the position target only.
Its damping acts against measured velocity; there is no independently commanded velocity input.
Calling `set_joint_velocity_target` alone does not add that input to this actuator.

A true position-plus-velocity command requires a compatible actuator and action adapter.
The repository has `BuiltinPdActuatorCfg` and explicit `IdealPdActuatorCfg` implementations.
The native paired PD implementation is a candidate, not an automatically validated replacement.
It needs checks for torque summation/clamping, delays, gain randomization, and joint order.
Hardware command support is not audited here.

Source: [builtin_actuator.py](../../src/mjlab/actuator/builtin_actuator.py), `BuiltinPositionActuator` and `BuiltinPdActuator`.

For a position-plus-velocity servo, the ideal law is:

```text
tau = Kp * (q_target - q_measured) + Kd * (dq_target - dq_measured)
```

Actual execution must retain the selected actuator's limits and delay semantics.
The existing teacher supplies position commands, not nonzero velocity-target labels.
Measured `dq` or the finite difference of position targets is not automatically the missing velocity command.

## Required changes

| Area | Required work |
| --- | --- |
| Data contract | Add a new named representation without a required VAE or latent identity. Specify units, joint order, target semantics, and timing. Preserve legacy readers. |
| Collection | Record per-step true joint `q,dq`, policy-visible joint observations separately when biased/noisy/delayed, and applied commands. Retain physical state, actions, teacher identity, boundaries, and perturbation evidence. |
| Collection policy | For a fully VAE-free pipeline, drive collection with the frozen PPO teacher directly. For a more controlled runtime-representation test, the VAE can remain only as the data generator. Declare which comparison is intended. |
| Preprocessing | Build new target windows and train-only statistics. Normalize q and dq separately. Rebuild caches under the new representation identity. |
| Model and process | Keep Transformer blocks, additive embedding mechanism, x0 loss form, and Euler mathematics. Change input/output packing and replace latent-specific slice meanings. Specify noise-level grouping for the new fields. |
| Sampler mask | Distinguish observed state from unknown commands. For command prediction, the current command is unknown. For measured-trajectory prediction, current q/dq are known and generated motion begins in the future. |
| Runtime | Replace latent history/decode with measured-state and executed-command history, or measured-trajectory history. Remove required VAE bootstrap and decoder loads. Use a declared teacher/standing startup to obtain real history. |
| Controller | Add command conversion/tracking, stale-plan checks, finite checks, and limits. Do not replay a long predicted trajectory without new observation feedback by default. |
| Evaluation/export | Add joint position/velocity and command errors, closed-loop metrics, new checkpoint identities, compatible export wrappers, and old-artifact refusal. |

Relevant pipeline sources:

- [contract.py](../../src/mjlab/tasks/tracking/diffusion/contract.py): required VAE identities and state/latent width relation.
- [storage.py](../../src/mjlab/tasks/tracking/diffusion/storage.py), `shard_array_shapes`: no per-step joint q/dq arrays.
- [runtime_x2.py](../../src/mjlab/tasks/tracking/diffusion/runtime_x2.py): live joint state is accessible during collection.
- [collector.py](../../src/mjlab/tasks/tracking/diffusion/collector.py): current primary rollout is VAE-controlled; PPO is the recovery path.
- [handover.py](../../src/mjlab/tasks/tracking/diffusion/handover.py): current-state/latent timing, bootstrap, history, and decode.
- [parallel_closed_loop_x2.py](../../src/mjlab/tasks/tracking/diffusion/parallel_closed_loop_x2.py), `true_joint_state` and `_capture`: true state differs from policy conditioning.

## Representation and feedback

X2 has 31 controlled joints. Joint q/dq therefore contains 62 values.
Do not implement this by merely setting `latent_dimension=62`.
A measured state and a command have different masks, timing, and runtime meanings.

For example, retaining the 135 physical-state fields and appending measured q/dq gives 197 state channels.
All current measured channels would be known, unlike the current unknown-latent block.
A command-prediction formulation also needs a separate command block.
Its total width depends on the selected observed-state fields and command fields.

Removing the latent does not require removing root/body-state observations.
Keep the current root/body information for the first representation experiment unless separately changed.
A q/dq-only observation cannot uniquely identify base orientation, translation, or contact state.

The decoder currently receives 99 conditioning values, including measured joint state and previous action.
Removing it also removes its learned fresh-observation correction.
The generator must receive the selected feedback inputs through the existing conditioning mechanism.
This is required input plumbing, not a new adaLN or goal-conditioning experiment.
PD feedback remains available at the servo, but it is not the removed learned decoder.
Asynchronous direct commands need explicit timestamps and plan-age behavior.

## What can be reused

Reuse the Transformer blocks, optimizer, batching approach, linear path, Euler formula, and simulation framework.
Reuse teacher artifacts and motion references when their control and observation contracts match.
Do not reuse latent statistics, latent caches, or latent checkpoints as compatible joint-space artifacts.

Old raw stores may support a direct position-command baseline from their saved actions and verified conversion metadata.
They do not provide the requested measured q/dq trajectories.
Additional saved recordings might support reconstruction, but no such complete source was verified here.
Plan fresh collection for the joint-state route.

Collect both state and command fields now so later comparisons do not require another schema extension.
Split whole trajectory/initial-state groups before windowing and statistics.
Never pair an action at t with the state at t+1 as if it were the pre-action state.

## Recommended comparison and validation

The cleanest removal-of-VAE baseline predicts the teacher's position command directly, retaining the existing position servo.
It can predict measured dq as an auxiliary future-state target without commanding it.
A full q-target/dq-target controller remains feasible, but also changes the control law and needs velocity-command semantics.

Before training, validate a recorded-command replay against the original command path.
For a measured-trajectory tracker, replay perfect recorded q/dq references first.
If that tracker fails, a better flow model alone cannot establish the missing control behavior.

Use software tests for field ordering, units, normalization, mask timing, reset isolation, and incompatible artifacts.
Measure q and dq prediction errors separately, with timestamp alignment.
Check position/velocity consistency without assuming finite-difference equality under coarse sampling.
Then measure matched closed-loop falls, tracking, actuator saturation, and end-to-end latency.
Lower prediction error does not prove stable control.

**Operator choice resolved:** predict achieved root orientation + q + dq, exclude the old 135-D physical state, and use the frozen multi-motion tracker.
**Remaining design choices:** precise heading-frame convention, new noise-level grouping, and rolling-reference time alignment.
**Implementation:** raw collector, focused tests, usage guide, and bounded remote simulator smoke complete. Flow model and rolling-reference adapter not implemented.
**Evidence:** 8 CPU tests, Ruff, and Ty pass. Three GPU smoke processes exit 0. All three manifests and NPZ files pass shape and finite-value checks. No training or deployment run.
