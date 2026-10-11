# Recording native tracking policy commands

`scripts/record_tracking_policy_commands.py` runs a motion-tracking checkpoint in its
training simulator (mjlab / MuJoCo-Warp) and records, for every actuated joint, the
reference motion state, the policy's input observation, the network action, the
processed position target and the simulator torque. It exists to answer one question:
when the deployment bridge commands a knee position target of about 47 deg while the
reference trajectory is at 65-72 deg, is that learned behavior or an
observation/export/environment difference?

The script records *native* motor commands from the same checkpoint, so the native
targets can be compared with the deployment targets joint by joint. It records all 29
joints of the G1 29DOF mode-15 tracking task, not just the knees.

## Running it

A bounded smoke run (a few policy steps, useful to validate the environment and the
checkpoint load):

```sh
uv run --no-sync python scripts/record_tracking_policy_commands.py \
  --checkpoint-file logs/rsl_rl/g1_29dof_mode_15_tracking/2026-09-11_20-36-01/model_15998.pt \
  --output-dir /tmp/mjlab_policy_commands_smoke \
  --max-steps 5
```

The full 15 s diagnostic (750 policy steps at 50 Hz) with the optional deploy-ONNX
parity check:

```sh
uv run --no-sync python scripts/record_tracking_policy_commands.py \
  --checkpoint-file logs/rsl_rl/g1_29dof_mode_15_tracking/2026-09-11_20-36-01/model_15998.pt \
  --output-dir /tmp/mjlab_policy_commands_15s \
  --onnx-file /home/agiuser/projects/deploy/robots/g1_29dof/config/policy/mimic/qianghuo_mode_15/exported/policy.onnx
```

Both commands above were run on 2026-09-12 with `model_15998.pt` (GPU: RTX 5080
Laptop, no other job running). The 15 s recording took about 11 s of wall clock.
The script refuses to overwrite an existing recording, so pick a fresh `--output-dir`
(or delete the old one) for every run; the exact output paths are printed at the end.

## What it does

* Loads the task with `load_env_cfg(task, play=False)`, the saved `lookahead_s` from
  `params/env.yaml`, and the checkpoint through the normal runner path
  (`RslRlVecEnvWrapper` + `runner.load` + `runner.get_inference_policy`), so the actor
  uses the observation normalization saved in the checkpoint, applied exactly once.
* Makes the run a *nominal* diagnostic and records every deviation: observation
  corruption off, all startup/interval domain randomization removed
  (`encoder_bias`, `base_com`, `foot_friction`, `push_robot`), reference-state
  randomization off, one environment, `auto_reset=False`, and only the episode horizon
  widened (15 s of recording + 1 s margin). Physical termination terms stay active.
* Starts at exact reference frame 0: `MotionCommand.reset_to_frame(env_ids, 0)`,
  then `sim.forward()`, `update_relative_body_poses()`, `sim.sense()` and a fresh
  observation computation, in that order.
* Snapshots the reference frame, the reference q/dq and the actor observation *before*
  inference and before `env.step`; `env.step` advances the motion frame and refreshes
  the observation afterwards, so the post-step observation is never reported as the
  input that produced the action.
* Stops after the requested number of steps, or at the first termination (recording
  the reason, the firing term and the achieved reference coverage). It never resets
  or stitches a new episode. The reference frame is checked before each step and again
  after it: the native command increments the frame by one per step and resamples
  (teleporting the robot and clearing its targets) once the frame reaches the end of
  the motion, so any other post-step frame aborts the recording instead of logging an
  invalid row. A request whose final step would land on the end of the motion is
  rejected before the environment is even constructed (see the limitations below).

Dynamics are not touched: physics timestep (5 ms), decimation (4), action scale and
offset, actuator gains, effort limits and command-delay settings come from the loaded
task config. The recorder compares the timestep, decimation, action scale, gains,
effort limits and minimum/maximum delay against saved `params/env.yaml` values.
It compares the `use_default_offset` flag, **not the actual offset values or saved
default joint positions**; effective offsets are recorded but their equality to the
saved defaults is unverified. These checks are not a blanket assertion of training
fidelity. Each comparison in `metadata.json -> saved_config.fidelity_checks`
carries a `status`:

* `match` / `mismatch` - both sides (saved YAML and loaded config) were resolved to
  one value per joint (or one scalar) and actually compared. Actuator `stiffness`,
  `damping`, `effort_limit`, `delay_min_lag` and `delay_max_lag` are compared per
  joint with the same regex-to-joint mapping used to build the actuators.
* `unverifiable` - the saved side is absent, incomplete (a joint uncovered), ambiguous
  (a joint matched twice) or of an unsupported shape; nothing is assumed equal.

For the reference checkpoint (`model_15998.pt`) all of these compared equal:
`decimation=4`, `physics_dt=0.005`, `lookahead_s=0`, the per-joint action scale, the
per-joint gains, effort limits and delay settings (max abs difference `0.0` over all
29 joints), `actions.joint_pos.clip` (present as `null` on both sides) and
`use_default_offset`. `sampling_mode`, `init_weight_s` and `seed` are recorded for
context only (`saved_values_not_compared`) because they do not affect the recorded
trajectory, and `agent.yaml` is recorded by path, hash and size - no agent setting is
compared.

The task config is the *working tree* config, so commit or record any local edits to
task code before comparing recordings: the override list covers configuration the
recorder changes, not unrelated source edits (for example reward weights, which do not
affect the recorded dynamics, observations or actors).

**Results must be labeled as a nominal, noise-free evaluation, not a sample from the
training distribution.**

## Reading the stages

The mode-15 G1 uses MuJoCo `<position>` actuators (`BuiltinPositionActuator`), so the
native control signal is a *position target*, not a torque:

| quantity | meaning | units |
| --- | --- | --- |
| `raw_action` | policy output | unitless |
| `clipped_action` | action after the wrapper's `clip_actions` (identical when `clip_actions` is null) | unitless |
| `processed_target` | `clipped_action * action_scale + action_offset` | rad |
| `sim_target` | `processed_target - encoder_bias`, written to the actuators | rad |
| `ctrl` | `data.ctrl` for those actuators (equals `sim_target` for a `<position>` actuator) | rad |
| `qfrc_actuator_post` | actuator torque projected into joint space, read after the step | N*m |
| `actuator_force_post` | scalar actuator output in actuation space | N*m |
| `pd_torque_requested_pre` | reconstruction `kp * (sim_target - q_pre) - kd * dq_pre`, before limits | N*m |
| `pd_torque_applied_pre` | the requested torque clamped to the configured effort limit | N*m |

`processed_target` is derived from the action that actually reached the action term,
i.e. the wrapper-clipped action, which is why both `raw_action` and `clipped_action` are
logged. An optional term-level `ActionTermCfg.clip` (applied *after* scale and offset)
is not modeled by the reconstruction: the recorded `processed_target` is the action
term's own value, and the cross-check against the scale/offset reconstruction fails
loudly rather than silently mislabeling the stage if such a term-level clip is set.

The deployment bridge computes an external PD *torque* into `ctrl`. Native `ctrl` is a
position target, so native and deployed `ctrl` are not comparable; compare the
processed/simulator position target and the joint-space actuator torque instead.

Two honest caveats, both visible in the artifacts:

* The PD-torque columns are reconstructions evaluated at the **pre-step** state, which
  is the state the action was computed from. They are not the torque that was
  integrated. `qfrc_actuator_post` is read after `env.step`, which calls
  `sim.forward()` after the decimation loop, so it is the actuator torque *recomputed
  at the post-step state*. Over the reference 15 s run the two differ by up to about
  31 N*m on a single joint, which is why both are recorded with distinct labels. A
  per-physics-substep torque trace would require instrumenting
  `ManagerBasedRlEnv.step` and is intentionally not done here.
* `metadata.json` records `consistency_checks`. For the reference run the reconstructed
  torque law `clamp(kp*(sim_target - q_post) - kd*dq_post, +/-effort_limit)` reproduced
  the simulator's `qfrc_actuator` to 7.6e-06 N*m, confirming the logged gains, joint
  mapping and the `<position>` actuator model. A large value there means the actuator
  adds terms this reconstruction ignores (for example gravity compensation) or that the
  gains/mapping differ.

With `events = {}` the encoder bias is exactly zero, as the recorded `encoder_bias`
column shows; `processed_target == sim_target` in that case. The columns exist so a run
with a nonzero bias is distinguishable.

## Output artifacts

* `policy_commands.npz` - the full recording with named axes. Per-joint arrays are
  `(T, 29)` in `joint_names` order; `actor_observation` is `(T, 154)` with the layout
  documented in `metadata.json` under `actor.obs_terms`; root arrays are `(T, 7)` pose
  and `(T, 6)` velocity; `sim_time_pre`/`sim_time_post`, `policy_step`, `ref_frame`,
  `ref_time_s`, `episode_length_pre/post`, `terminated`, `truncated`, `done` and
  `termination_flags` carry the step identity.
* `policy_commands.csv` - one row per (step, joint) with the same per-joint quantities,
  for quick grepping and plotting.
* `actor_observations.csv` - the actor input (154 values) per step.
* `metadata.json` - provenance (checkpoint/motion/params hashes), every diagnostic
  override with its reason, the effective gains/action scale/joint order, the
  saved-vs-effective fidelity checks, the consistency checks, termination terms, units
  and a per-joint summary.
* `onnx_parity.json` - only with `--onnx-file` (see below).

For the reference 15 s run the per-joint summary reports a mean
`sim_target - ref_joint_pos` of -9.3 deg (range -28.6 .. +15.4) for the left knee and
-5.2 deg (-40.7 .. +32.3) for the right knee, with references spanning 25-73 deg and
7.5-79 deg respectively, and a maximum torque utilization of 33 % (knee) and 61 %
(left ankle pitch, the highest joint overall against its effort limit). No joint
saturated its effort limit anywhere in the run, and the episode did not terminate.

## ONNX parity

`--onnx-file` feeds the recorded actor observations to the deploy export and to the
checkpoint actor; both apply their own baked-in normalization exactly once, and the
raw 29 actions are compared. For the reference run the deploy export
(`obs [1,154] -> actions [1,29]`) matched the checkpoint to 7.6e-06 over 200 sampled
steps, and the checkpoint actor reproduced the recorded actions exactly when rerun on
the recorded observations. This is a *numerical export check*: it shows the exporter
and normalizer are consistent with the training actor, and says nothing about whether
the native and deployed closed-loop trajectories match.

Do not substitute the checkpoint directory's auto-exported `<run>.onnx` (which wraps
the policy with a `time_step` input); the script rejects models that do not have
exactly one input.

## Assumptions and limitations

* The motion npz joint order is assumed to be the robot's joint order; the training
  task relies on this too. The recorder flags the start frame when the reference
  positions are clipped by the command's soft limits, which is the main observable
  symptom of a mismatched convention.
* One environment, one continuous episode, no resets; a failure before the requested
  bound ends the recording early and is reported explicitly (`end_reason`,
  `end_detail`, `achieved_reference_s`).
* Recording uses the actor joint-position term and its private `_processed_actions`
  (cross-checked against the public scale/offset reconstruction) and the public
  simulator accessors for state and torque. No shared mjlab source file is modified
  for logging.
* Command delay is verified, not assumed: the saved and effective `delay_min_lag` /
  `delay_max_lag` are compared per joint under
  `saved_config.fidelity_checks.actuator_settings`, and
  `fidelity_checks.effective_command_delay` describes the *effective* config only
  (`delay_max_lag = 0` for all 29 joints in the mode-15 task). Per-joint effective
  values are also in the NPZ.
* The motion-end window is enforced before the environment is built. Because the native
  command advances the reference frame after every step and resamples at the end of the
  motion, a request is rejected unless `start_frame + steps < motion_frames` for the
  motion's frame count (the supplied clip has 2286 frames, so frame 0 allows at most
  2285 steps). Equality with the frame count is already unsafe, and the recorder aborts
  if a step ever produces an unexpected post-step frame.

## Tests

`tests/test_record_tracking_policy_commands.py` covers the joint/ctrl mapping, the
raw/scaled/bias-adjusted target distinctions, observation/reference pairing before the
step, termination without reset contamination, the step bound, action clipping,
reference desynchronization, the motion-end window (including the last-frame boundary
and a native-style wraparound that still reports a coincident termination), the output
schema and overwrite refusal, saved-versus-effective provenance (matching, differing,
absent, partial, ambiguous and unsupported saved settings), the saved-scale comparison
and the ONNX parity helper. The rollout tests use a fake environment that mirrors the
real step ordering and motion wraparound, so the suite needs no GPU:

```sh
uv run --no-sync pytest tests/test_record_tracking_policy_commands.py
```
