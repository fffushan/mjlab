# BeyondMimic VAE distillation: proposed architecture

Status: architecture design; the teacher-loading foundation (M1) is implemented and parent-validated. VAE/DAgger training and later milestones remain planned. See [implementation plan and acceptance record](beyondmimic_vae_implementation.md).
Scope: reproduce the conditional-VAE/DAgger stage using existing mjlab tracking teachers; retain a clean interface for later state–latent diffusion.
Source: `/home/agiuser/Documents/beyondmimic.pdf`, main pp. 18–21 and Fig. 7, supplementary S3/S4 and Table S6.

## Initial teacher cohort selected by the user

Use AgiBot X2 teachers from `logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/` (all paths below are relative to the mjlab repository).

| Teacher | Checkpoint under that directory | Reference motion under `data/tennis/` | Frames / FPS |
| --- | --- | --- | --- |
| `tennis_000` | `tennis_000/model_29999.pt` | `single_000_zhanghongyu_agibot_x2_tracking.npz` | 453 / 50 |
| `tennis_001` | `tennis_001/model_29999.pt` | `single_001_zhanghongyu_agibot_x2_tracking.npz` | 340 / 50 |

Both selected run directories contain only one checkpoint, at iteration 29,999. Artifact inspection confirms:

- Actor architecture `164 -> 512 -> 256 -> 128 -> 31`, with independent saved observation normalizers.
- All exported ONNX metadata entries agree between teachers, including joint order, exported gains/default positions/action scales, observation ordering/scales/history, anchor (`torso_link`), and tracked body names.
- The reference joint-position and joint-velocity arrays embedded in each ONNX are exactly equal to the user's corresponding NPZ arrays. This check does not claim to have compared every body array.
- The teacher observation layout is `command` (62), `motion_anchor_ori_b` (6), `base_ang_vel` (3), `joint_pos` (31), `joint_vel` (31), and `actions` (31): 164 total. The metadata also names `motion_lookahead`, but the saved configuration explicitly sets `lookahead_s: 0.0`, so that term is zero-width.
- No global anchor-position or base-linear-velocity input is present in these teachers.

The repository task matching this experiment name is `Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations`. The recovered saved configuration confirms a 50 Hz control rate (0.005 s physics timestep, decimation 4), consistent with both motion files.

**Original configurations recovered:** the user copied `params/env.yaml` and `params/agent.yaml` alongside each checkpoint from the remote training servers. The two agent configurations are byte-for-byte identical; the only textual difference between the environment configurations is `commands.motion.motion_file`. The remote paths use `/home/fushan/mjlab/data/tennis/`; resolve these to the selected local NPZ files in the manifest without modifying the original provenance files.

The saved configurations confirm:

- Actor MLP `[512, 256, 128]`, ELU, observation normalization enabled, no recurrence, and no runner/action-term action clipping.
- Anchor `torso_link`; angular velocity reads builtin sensor `robot/imu_ang_vel`, with no task-level torso-IMU override in the saved scene sensors. Resolve its physical site/frame through the robot asset during teacher parity validation.
- Joint position and velocity share the `encoder_packet` delay group: lag 0–1 control steps, hold probability 0.9. Angular velocity has its own lag 0–1 delay with the same hold probability. Actuators use lag 0–2 physics steps with hold probability 0.9.
- PD stiffness/damping randomization uses a shared gain scale in `[0.7, 1.3]`; reduced pushes occur every 4–8 s. Preserve the saved randomization configuration rather than reconstructing it from experiment names.
- Reference lookahead is disabled, phase sampling is adaptive, episode limit is 10 s, and training used 8,192 parallel environments (the student environment count remains a separate resource choice).

This resolves the missing-configuration blocker and establishes a compatible saved configuration for the two teachers. Numerical teacher inference/rollout parity and robot-asset/frame verification are still implementation acceptance checks, not claimed as completed by this file comparison.

Proposed student schema for this cohort:

```text
encoder input: reference q(31) + reference dq(31) + anchor orientation error(6)
               = 68 dimensions
encoder:       68 -> [2048, 1024, 512] -> mu(32), logvar(32)
proprio:       root projected gravity(3) + teacher-compatible gyro(3)
               + relative joint position(31) + joint velocity(31)
               + previous executed action(31) = 99 dimensions
decoder:       [latent(32), proprio(99)] -> [2048, 1024, 512] -> action(31)
```

Projected gravity is a new student proprioceptive input, not a modification to the frozen teachers' 164-dimensional inputs. Verify its deployment frame/availability and the gyro frame before freezing the schema. Retain no-state-estimation operation: omit anchor position and linear velocity. Start with equal motion weights; prove one-teacher behavior using `tennis_000`, then include `tennis_001` in shared training. The first student is intended to cover both clips, not to train two separate VAEs.

## 1. Goal and boundaries

Train one shared reference-conditioned student for a collection of motion-specific PPO teachers on **one robot/control configuration**. The student must execute those motions in closed loop, not merely fit offline action labels.

The product is an encoder plus a separately deployable decoder:

```text
reference joint states + anchor error --> Encoder --> z (32)
                                                     |
current deployable proprioception ----------------> Decoder --> action
```

Later, diffusion replaces the encoder as the source of `z`. No reference, motion ID, teacher ID, or teacher-only state may bypass the latent bottleneck into the decoder.

Out of scope initially: diffusion training/guidance, new PPO teachers, arbitrary cross-robot distillation, recurrent teachers, asynchronous/distributed collection, and hardware execution. Rewards remain useful evaluation signals but do not define a PPO/value loss in this stage.

## 2. Existing code to reuse and constraints discovered

- `src/mjlab/tasks/tracking/tracking_env_cfg.py`: tracking observations, actions, rewards, events, and terminations.
- `src/mjlab/tasks/tracking/mdp/commands.py`: motion loading, reference state queries, initialization, phase sampling, and tracking metrics. Currently takes one `motion_file`; reference progression is one frame per environment step, even when NPZ FPS differs.
- `src/mjlab/tasks/tracking/mdp/observations.py`: existing reference/anchor observations and frame conventions.
- `src/mjlab/rl/runner.py`: current and legacy RSL-RL checkpoint conversion, including normalizers. Reuse/factor its conversion logic rather than invent another checkpoint format interpretation.
- `src/mjlab/rl/exporter_utils.py`: action and observation metadata, including joint names and sensor frames. Extend the metadata conventions for encoder/decoder schemas; the current exporter assumes an `actor` observation group.
- `src/mjlab/envs/manager_based_rl_env.py`: vectorized simulation, reset, and observation lifecycle. Automatic resets return post-reset observations; motion-end resampling can also change the reference/robot state and needs an explicit boundary signal for recording.
- Installed RSL-RL has ordinary student–teacher distillation, but its inspected implementation has one teacher and MSE/Huber supervision, not the desired multi-teacher conditional VAE. Do not modify installed dependencies or force the new loss through PPO.

Existing task variants include G1, G1 mode 15, and X2, with different sensor/anchor/observation choices. Select a compatible teacher cohort; do not assume current task defaults reproduce an old checkpoint.

## 3. Components and ownership

```text
Teacher manifest --> validation --> frozen TeacherBank
                       |
                       v
MotionLibrary --> multi-motion tracking environment
                       |
                 observation snapshot
                  /              \
          teacher input       reference + proprioception
               |                       |
          TeacherBank             VAE student
               |                       |
         target action           student action
               |                       |
               +--> DAgger collector <--+
                           |
                 bounded aggregate buffer
                           |
                    supervised trainer
                           |
               checkpoint / evaluation / export
```

### 3.1 Teacher manifest and compatibility validator

Explicitly pair every checkpoint with its reference motion and saved training configuration. Suggested manifest fields:

```yaml
# Illustrative schema, not an executable config yet.
base_task: <compatible tracking task>
teachers:
  - id: motion_a
    checkpoint: <run_a>/model_N.pt
    motion: <motion_a>.npz
    env_config: <run_a>/params/env.yaml
    agent_config: <run_a>/params/agent.yaml
    sampling_weight: 1.0
```

Resolve and save a contract containing robot asset/version, joint/action order, nominal gains and action scale/offset/clipping, simulation timestep and control decimation, reference FPS, observation names/order/transforms, sensor frames, anchor, lookahead/history settings, and checkpoint/motion/config hashes.

Initial implementation rejects incompatible robot/control/observation contracts. Different learned teacher normalizer statistics are expected and retained separately. Missing metadata requires explicit resolution rather than guessing. A later adapter can support heterogeneous observation schemas, but is not necessary for the first reproduction.

### 3.2 Frozen TeacherBank

Load actors only for inference, with their own frozen observation normalizers. Teachers use evaluation mode, no gradients, and deterministic inference actions rather than PPO exploration samples.

API concept:

```text
label(teacher_ids[B], teacher_observations[B, Dt]) -> actions[B, J]
```

Group rows by teacher ID, batch each teacher's inference, and scatter the outputs back. Teacher IDs are routing metadata, not student features. Prefer native PyTorch checkpoints for batched GPU labeling; ONNX is an export/equivalence target rather than a second required training backend.

Do not instantiate a PPO optimizer, critic, or extra simulated robot for each teacher. Validate the loader against the existing deterministic inference path before training.

### 3.3 MotionLibrary and multi-motion command

Use one vectorized simulator for a compatible cohort. A library holds reference tensors plus clip offsets, lengths, FPS, and teacher mapping. Each environment owns a motion ID, frame index, and episode/segment ID.

At reset: choose motion, select a valid phase, initialize from that motion using the existing reset semantics, and route labels to its teacher. Keep the motion/teacher assignment stable until an explicit reset or segment boundary. Do not concatenate clips and let indexing cross clip boundaries.

Add a multi-motion command backend that preserves the existing tracking-command query interface for observations, rewards, terminations, and metrics. Share reference indexing/reset logic where practical; preserve the existing single-motion task as a regression baseline.

Start with configurable equal-per-motion collection and balanced replay, with uniform phase coverage. Optional adaptive phase sampling must be tracked separately per motion. Report both motion-balanced and duration-weighted evaluation, since clips can have different lengths. Any motion-end resampling/teleport is an explicit new segment, including when the environment does not otherwise signal termination.

### 3.4 Observation adapter and student contract

Build named features, then pack them according to a versioned schema; avoid hard-coded slices inferred only from tensor size.

Three views come from the same observation time:

- `teacher_obs`: the exact saved actor input contract.
- `reference_obs`: reference joint positions/velocities and configured anchor error.
- `proprio_obs`: projected gravity, configured IMU velocities, joint states, and previous executed action.

The paper's full input formulation is the reference design. For already hardware-validated teachers without linear velocity/global position inputs, choose a deployable schema explicitly rather than introducing unavailable measurements. Lookahead-dependent teachers require an explicit design choice about encoder inputs; do not silently discard their conditioning. These are adaptations to document, not claims about unspecified paper settings.

Root, anchor, and IMU frames are distinct. Joint offsets, velocity conventions, Rot6D packing, measurement noise, delay, and scaling are part of the schema. Reuse one corrupted/delayed snapshot for overlapping teacher/student measurements, rather than independently resampling noise or advancing observation history twice.

Store input features before learned neural normalization. Teachers retain their individual statistics; the student has its own shared normalizers. Update student statistics at controlled training boundaries, not unpredictably during label generation, and freeze them for evaluation/export/data collection.

### 3.5 Conditional VAE

- Encoder: MLP `[2048, 1024, 512]`, ELU, separate 32-dimensional mean and log-variance heads.
- Training latent: `z = mu + exp(0.5 * logvar) * epsilon`.
- Decoder: MLP `[2048, 1024, 512]`, ELU, receives `[z, normalized_proprioception]`, outputs `J` normalized actions.
- No added tanh or action clipping unless required by the existing action contract.
- Deterministic evaluation/export uses `z = mu`; also evaluate posterior-sampled execution. The choice of rollout latent convention is an explicit implementation decision because the paper does not fully specify it.

Training objective: action reconstruction plus Gaussian posterior KL to `N(0, I)`. Start with the reported learning rate `5e-4`, KL coefficient `0.01`, and gradient accumulation of 15 minibatches. Use Adam as an explicit implementation choice, not a reported Table S6 field.

Fix and log the loss reduction convention: proposed reconstruction sums squared error over joints per sample, KL sums over latent dimensions, then both average over samples. Divide accumulated loss by the accumulation count. The paper does not specify all reduction details; the effective relative weight depends on them.

No extra motion reconstruction, temporal loss, or adversarial objective in the initial baseline. Stabilization modifications such as KL warmup remain separately named experiments.

### 3.6 DAgger collector and bounded aggregate buffer

First implementation uses synchronous in-process labeling for simplicity. A future collector can label saved complete teacher observations in batches afterward without changing the algorithm.

For each step:

1. Snapshot current reference, proprioception, teacher inputs, motion/frame/episode IDs.
2. Compute frozen teacher target and student action at that snapshot.
3. Record labeled student inputs before stepping; detach/copy reused environment buffers.
4. Execute the selected behavior action through the existing action-processing path.
5. Capture termination/reset/segment boundaries and continue from the returned observation.

Use an optional teacher-controlled bootstrap to establish a usable student, followed by student-controlled DAgger. A configurable teacher-execution probability can decay to zero if needed; when mixing, select whole action vectors instead of averaging teacher/student commands. This schedule is a proposed engineering choice, not a schedule specified by the paper. Final evaluation is always student-only.

The previous-action observation must be the action actually passed to the environment after configured clipping, not a counterfactual teacher action. Keep teacher targets, student predictions, and executed commands distinct.

The aggregate buffer is capacity-bounded, retaining a configurable recent/history mixture and motion-balanced samples. It stores:

```text
reference_obs, proprio_obs, teacher_action,
motion_id, teacher_id, reference_frame, episode_id, collector_iteration
```

Optional diagnostic shards also store teacher inputs, executed actions, and boundary flags. Ordinary VAE updates do not require full trajectories or next states. Recompute latents from stored reference inputs with the current encoder during optimization; do not train against stale cached latents from earlier student versions.

Keep valid pre-failure DAgger samples, not just successful rollouts: these are the corrections the method needs. Exclude invalid/non-finite samples and end/reset failed episodes. The paper's successful-rollout filter belongs to later diffusion-data collection, not to this supervision buffer.

### 3.7 Trainer, artifacts, and exports

Use a dedicated `VaeDistillationRunner` coordinating collection, replay, optimization, evaluation, logging, and checkpointing. It can reuse mjlab environment/CLI/logging utilities without depending on a PPO rollout/value/advantage loop.

Checkpoint encoder/decoder, student normalizers, optimizer, update counters, mixing schedule, RNG state, resolved configuration, schemas, and source hashes. Persist the bounded replay buffer or an explicit reconstruction policy for it. Resuming optimizer/training state does not imply a bitwise continuation of simulator state: restart environments unless a full simulator snapshot facility is deliberately implemented.

Export two separate inference interfaces:

- `encoder.onnx`: reference inputs -> posterior mean.
- `decoder.onnx`: latent + deployable proprioception -> normalized action.

Bake neural normalization into exports and ship schemas, frame/ordering/action metadata, and a shared model-version identifier. Optionally export a combined reference tracker for initial validation. Do not assume current single-motion ONNX consumers can already run these new interfaces; deployment integration is a later explicit step.

## 4. Evaluation and acceptance gates

1. **Teacher parity:** loaded teacher actions match the existing inference path; its rollout reproduces the known baseline under the selected configuration.
2. **One-teacher student:** overfit/check a small batch, then execute full clips and phase-start rollouts using only the VAE. Test both mean and sampled latents.
3. **Multi-teacher student:** evaluate every motion separately under matched seeds/perturbations; no strong motion may hide a failing one in an aggregate score.
4. **Export parity:** PyTorch/ONNX agreement, normalizer/schema/frame correctness, and target CPU inference budget.
5. **Deployment validation:** sim2sim, then separately authorized hardware validation, before accepting the controller for real use.

Primary metrics: completion/survival, per-body/anchor tracking error, action-rate/jitter, joint-limit/torque saturation, and failures per reference segment. Secondary metrics: per-motion/joint action MSE, KL, posterior statistics/active latent dimensions, reconstruction under swapped/ablated latents, and buffer coverage.

A small MSE alone is not success. Confirm the decoder genuinely uses the latent; a proprioception-only shortcut can look acceptable on repetitive motions while failing to provide a useful command space. Set quantitative tolerances relative to measured teacher baselines before the first acceptance run rather than inventing paper-reported thresholds.

Symmetry augmentation is part of the paper's target recipe. Add it after the unaugmented pipeline passes basic correctness gates, with robot-specific left/right permutations and sign/frame transforms for the complete reference/proprioception/action tuple, and verify the mapping is physically appropriate. Report its absence as a reproduction deviation until enabled.

## 5. Boundary to future diffusion

Keep VAE training replay separate from the later sequential dataset. Freeze encoder, decoder, normalizers, schema, and latent convention before collecting diffusion data. A changed encoder generally requires recollection/re-encoding and changes the diffusion model's latent interface.

Later collect time-aligned physical state, latent actually used, decoded/executed action, timestamps, and episode/segment boundaries from VAE-driven rollouts. Apply OU perturbations and the paper's stability filtering there; never train sequence windows across resets or reference teleports.

Define the decoder as `decode(z, current_proprioception)` from the start. Do not yet implement diffusion features, state emphasis projection, cost guidance, or asynchronous inference.

Preserve the teachers' validated control period for the first VAE reproduction. The paper uses 25 Hz for its diffusion pipeline, but changing a trained 50 Hz tracker to 25 Hz is not a harmless scheduling change. Matching that part of the paper later requires deliberately validated 25 Hz teachers/controllers and reference resampling, or a separately evaluated multi-rate design.

## 6. Proposed placement and implementation order

```text
src/mjlab/tasks/tracking/distillation/
  config.py          # manifest, schema, student/collector/training config
  teachers.py        # frozen teacher bank and compatibility validation
  observations.py    # named teacher/reference/proprio input views
  model.py           # conditional VAE and inference wrappers
  storage.py         # bounded labeled aggregate buffer
  runner.py          # DAgger collection and supervised training
  export.py          # encoder/decoder export and schema metadata
src/mjlab/tasks/tracking/mdp/
  motion_library.py  # multi-clip storage and routing
  ...                # multi-motion command integration
src/mjlab/scripts/
  distill.py         # first-class `uv run distill ...` entrypoint (proposed)
```

Implementation gates, not autonomous work instructions:

1. Select one robot/cohort; implement manifest validation and teacher parity.
2. Implement VAE plus one-teacher DAgger against the existing single-motion environment.
3. Add the motion library, multi-motion command, routing, and balanced replay; retain one-teacher tests.
4. Add symmetry, complete evaluation, and standalone encoder/decoder export.
5. Only after acceptance, design the diffusion collector/trainer.

Resolved user inputs: AgiBot X2, `tennis_000` and `tennis_001`, their corresponding `data/tennis/` motions, and original saved environment/agent configurations; see the initial-cohort section. No missing-config blocker remains. Teacher inference/rollout parity and physical sensor-frame checks belong to implementation validation. Confirm availability/frame of the additional projected-gravity student input during deployment.
