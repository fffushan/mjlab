# X2 general PPO teacher and goal-conditioned latent diffusion

Date: 2026-10-14 (+08).
Status: architecture agreed in discussion; implementation and training are not authorized by this record.
Scope: a new X2 pipeline with one shared teacher. Existing specialist-based experiments remain unchanged.

PPO teacher update, 2026-10-14 (+08): [implementation and software validation](../reports/x2_multi_motion_ppo_teacher_implementation.md).
The shared PPO task is integrated locally; the [bounded GPU smoke](../reports/x2_multi_motion_ppo_teacher_gpu_smoke.md) passes.
Tracking quality remains not tested.
VAE and diffusion changes remain separate future work.

## 1. Goal and agreed decisions

Train one X2 controller across many reference motions with proximal policy optimization (PPO).
Use that shared controller as the teacher for a variational autoencoder (VAE).
Then train diffusion to generate physical-state and latent sequences from actual history and task goals.
Use the VAE decoder to produce robot actions from generated latents and current robot observations.

The operator reports about 16 hours to train each specialist PPO teacher.
The new design avoids a separate PPO teacher for each reference motion.
It does not remove the VAE stage.
A shared teacher can still require substantial training time.
Its X2 training cost is not measured.

Agreed architecture:

```text
Many reference motions
        |
        v
One multi-motion PPO teacher
  inputs: robot observations + current/future reference features
        |
        v
VAE action distillation
  encoder: reference window + current reference error -> latent z
  decoder: latent z + actual robot observations -> action
        |
        v
Freeze the VAE; collect aligned state-latent sequences
        |
        v
Goal-conditioned diffusion
  inputs: actual history + goal pose/velocity
  outputs: predicted state sequence + latent sequence
        |
        v
VAE decoder at control time
  inputs: selected generated latent + fresh actual robot observations
  output: robot action
```

**Future reference information enters the VAE encoder, not the decoder.**
**Diffusion deployment does not require an externally supplied reference motion.**

These decisions define a design direction, not a validated controller.
The operator now selects the SCDP six-layer network and future-reference input for the PPO teacher.
See the [lean PPO implementation plan](../plans/x2_multi_motion_ppo_teacher.md).
That plan proposes one future sample at `0.2 s`; later VAE and diffusion interfaces remain open.

## 2. PPO teacher: one policy, multiple references

Each environment follows its own reference clip and local phase.
All environments update the same policy.
Rewards compare each robot with its own reference, not an average of several references.

Reference features communicate motion intent.
Preview features communicate the next motion changes.
A clip ID alone is not a suitable main condition for unseen-reference use.

The first implementation can reuse existing robot actions, rewards, and termination terms.
It must retain balanced motion coverage and clip-local reset behavior.
Evaluation must report each motion separately.
Mean reward alone can hide failures on difficult clips.

The current registered tracking tasks accept one `MotionCommandCfg.motion_file`.
The distillation path already provides reusable reference components:

- [`MotionLibrary`](../../src/mjlab/tasks/tracking/distillation/motion_library.py): loads several clips and checks clip-local queries.
- [`MultiMotionCommand`](../../src/mjlab/tasks/tracking/distillation/multi_motion.py): maintains clip and phase for each environment.
- [`environment.py`](../../src/mjlab/tasks/tracking/distillation/environment.py): builds private environments with the multi-motion command.

These components do not make multi-motion PPO available through the current training command.
The training input check, task configuration, playback, and export need adaptation.
The current multi-motion command refuses adaptive failure-based phase sampling.

A new task variant must preserve the single-motion baseline.
The shared teacher also needs a new distillation integration path.
The existing teacher-bank manifest is not automatically a shared-teacher interface.

## 3. Why the encoder must receive preview

Define these quantities:

| Symbol | Meaning |
| --- | --- |
| `o_t` | Actual robot observations available to the decoder at time `t` |
| `u_t` | Teacher observations; these can include privileged simulator features |
| `R_t:t+H` | Current and future reference features over a preview horizon |
| `e_t` | Current reference error, computed from an aligned robot/reference snapshot |
| `z_t` | VAE latent command |
| `a_t` | Robot action |
| `s_t` | Physical state used by the diffusion representation |
| `g` | Goal pose or velocity, with its frame and timing defined |

Teacher action:

```text
a_teacher_t = T(u_t, R_t:t+H)
```

Proposed VAE interface:

```text
z_t = E(R_t:t+H, e_t)
a_student_t = D(z_t, o_t)
```

The encoder compresses reference intent and preview into the latent.
The decoder combines that intent with actual robot observations.
The VAE training target is the aligned teacher action, not reference-pose reconstruction.

Two references can share a current pose and velocity but require different next actions.
An encoder with identical current inputs cannot distinguish those future intentions.
More network capacity does not fix absent information.
Therefore, a preview-dependent teacher requires sufficient student inputs to preserve its intent.

During diffusion deployment, the encoder no longer supplies the selected latent:

```text
(predicted_states, generated_latents) = Diffusion(actual_history, goal)
a_t = D(selected_generated_latent, fresh_actual_observations)
```

Teacher training uses reference motions.
Diffusion deployment does not require those motions.
The generated latent replaces the reference-conditioned encoder output.

The decoder must use actual observations for feedback.
Predicted states must not replace those observations.
If sensor-only observations cannot reproduce the teacher, observation history or a different observation design may be necessary.
This information limit requires validation; latent capacity alone does not resolve it.

## 4. Alternative considered: preview in the decoder

The discussion considered this alternative:

```text
z_t = E(current_reference_features)
a_t = D(z_t, actual_observations, generated_future_reference)
```

This is a possible architecture, but it is not the selected first design.
Preview would communicate intent through a path outside the latent.
The decoder might then use little latent information.
That outcome is a risk, not a proven failure.

This alternative also needs compatible preview features during training and deployment.
A clean training reference and a predicted physical trajectory are not interchangeable.
It would require separate tests for latent use, prediction errors, and action/preview consistency.

The selected design avoids that extra decoder input.
Generated states remain useful for goal constraints, guidance, and prediction checks.
They do not need to become decoder reference inputs.

## 5. State-latent pairs are not complete reference frames

The existing 135-dimensional hybrid state contains:

- Root position and orientation.
- Root linear and angular velocity.
- Tracked-body positions and linear velocities.

It does not contain joint positions, joint velocities, or every body orientation.
See [`diffusion/state.py`](../../src/mjlab/tasks/tracking/diffusion/state.py).
A generated state therefore cannot directly supply every joint-based preview feature.
Body positions alone do not establish a unique joint configuration.

Reference and realized trajectories also have different meanings:

| Quantity | Meaning |
| --- | --- |
| Reference trajectory | Motion requested from the tracker |
| Realized trajectory | Motion the robot actually executes |
| Generated state trajectory | Diffusion prediction of physical states |

Tracking errors separate the reference from the realized trajectory.
The sequential diffusion data contains realized states and the latents used during those rollouts.
It does not automatically contain clean future reference frames as state tokens.

The 129-dimensional local-kinematic variant omits root position and root linear velocity.
Absolute position or root-velocity goals therefore need another declared representation or an explicit additional interface.
Do not infer that the existing local-kinematic contract already supports those goals.

## 6. Goal-conditioned generation

The operator wants goal pose/velocity inputs, not full reference motions, at diffusion deployment.
The goal must define the controlled body, coordinate frame, units, and timing where applicable.
For example, a hand-pose target needs a target time or another completion rule.

Two methods remain candidates:

1. **Direct conditioning:** train diffusion with suitable goal labels and explicit goal inputs.
2. **Guidance or inpainting:** constrain selected generated-state features toward a goal at specified future times.

The discussion does not select one method.
Existing task-steering code can inform the design, but compatibility needs a source and contract audit.

Goal conditioning does not guarantee physical feasibility.
The collected motion and recovery coverage limits the learned behaviors.
Training-library coverage and held-out-motion performance require separate reports.
Disconnected clips do not establish reliable transitions between arbitrary motions.

The diffusion history must contain actual state-latent pairs from a declared control path.
Record actual latent commands, clean decoded actions, executed actions, and segment boundaries separately.
Do not form sequence windows across resets or reference teleports.
Freeze the encoder, decoder, normalizers, and latent convention before diffusion collection.

A reference-free claim starts after a declared initialization or handover boundary.
The startup method remains open.
Goal-frame alignment and physical-state estimation also remain open.
Goal conditioning does not remove their measurement requirements.

## 7. SCDP paper: useful evidence and limits

Source: *SCDP: Learning Humanoid Locomotion from Partial Observations via Mixed-Observation Distillation*.
Version: [arXiv:2603.09574v1](https://arxiv.org/abs/2603.09574v1).
Local PDF: `~/Downloads/scdp.pdf`.

### Reported PPO expert configuration

Page 3, Section II-D reports:

| Parameter | Paper value |
| --- | --- |
| Multi-motion policy hidden dimensions | `[1024, 1024, 1024, 1024, 512, 128]` |
| Parallel environments | `4096` |
| PPO iterations | `100000` |
| Learning rate | `3e-4` |
| Training time | About four days on one NVIDIA RTX 5090 |
| Simulator and robot | IsaacLab and Unitree G1 |

Page 2, Section II-A describes privileged body-state observations and multi-frame body-reference preview.
The paper does not specify a separate critic size or the PPO rollout length.
It does not provide a network-size comparison.
Its runtime estimate does not establish X2 or RTX 4090 training cost.

Page 6, Table IV reports MMP success of `98.94%` on training data and `99.03%` on test data.
These are simulation results on the AMASS-walk dataset.
They do not validate X2 tennis, jumps, or unrestricted whole-body motion.
The later SCDP diffusion model has `93.22%` test success in that table.
The `65` clips mirrored to `130` describe a diffusion collection subset, not the stated PPO library size.
The diffusion model uses a `16`-step prediction horizon, not a specified PPO preview horizon.

### Earlier X2 capacity proposals

The current X2 actor and critic both use `(512, 256, 128)`.
See [`agibot_x2/rl_cfg.py`](../../src/mjlab/tasks/tracking/config/agibot_x2/rl_cfg.py).
X2 still needs 31 action outputs; more motions do not increase the action dimension.

| Candidate | Actor hidden dimensions | Evidence |
| --- | --- | --- |
| Moderate baseline | `(512, 512, 256)` | Proposed; not tested |
| Larger compromise | `(1024, 1024, 512, 256)` | Proposed; not tested |
| Paper architecture reference | `(1024, 1024, 1024, 1024, 512, 128)` | Paper-reported for its MMP |

Later operator selection, 2026-10-14 (+08): use the paper architecture for the PPO actor.
The [PPO plan](../plans/x2_multi_motion_ppo_teacher.md) also uses that shape for the critic as an engineering choice.
The earlier size comparison is not a prerequisite for this research task.
Training throughput and inference latency still need measurement.

The earlier `0.1-0.2 s` preview suggestion is also a candidate, not a selected horizon.
The current `lookahead_s` returns one future sample and defaults to zero.
A multi-frame encoder window needs a new input schema.

## 8. Existing contracts and implementation boundaries

The existing default VAE encoder receives 68 current-reference features.
These include reference joint position, reference joint velocity, and anchor orientation error.
The decoder receives a 32-dimensional latent and 99 conditioning features, then outputs 31 actions.

Source interfaces:

- [`distillation/observations.py`](../../src/mjlab/tasks/tracking/distillation/observations.py)
- [`distillation/model.py`](../../src/mjlab/tasks/tracking/distillation/model.py)
- [`diffusion/async_latent_runtime.py`](../../src/mjlab/tasks/tracking/diffusion/async_latent_runtime.py)

The proposed encoder adds reference preview.
It requires a new schema, normalizers, checkpoints, exports, and diffusion data identities.
Existing bundles are not automatically compatible.
Preserve their contracts and artifacts.

## 9. Open items and next action

Before implementation, define:

1. The motion manifest and first diverse training subset.
2. Deployable actor/decoder observations and privileged critic observations.
3. Preview features, sample offsets, horizon, and endpoint rules.
4. Model-size comparisons and a bounded training budget.
5. Shared-teacher labels, VAE schema, and latent-use tests.
6. Diffusion state representation, goal interface, and startup behavior.
7. Per-motion, held-out, transition, and recovery evaluation criteria.
8. State-estimation, frame-alignment, and controller timing requirements.

Next action: approve the [multi-motion PPO implementation plan](../plans/x2_multi_motion_ppo_teacher.md).
The current plan covers the teacher only; VAE and diffusion implementation remain separate.
Implementation and execution need separate operator authorization.
This discussion does not authorize remote jobs, deployment, or robot operation.

Validation status: source and paper inspection only.
The proposed architecture, training quality, inference latency, and hardware behavior are not tested.

## 10. Related records

- [Earlier general-tracker and diffusion discussion](beyondmimic_general_tracker_and_diffusion.md)
- [Existing VAE architecture](beyondmimic_vae_distillation.md)
- [Existing state-latent diffusion design](beyondmimic_diffusion_reproduction.md)
- [Existing X2 distillation guide](../source/x2_tennis_distillation.rst)

The earlier discussion retains its historical specialist-teacher decision.
This record defines a new shared-teacher direction.
Existing experiment status stays unchanged.

## 11. Preview-aware flow-matching next step

Planning review: 2026-10-20 (+08).
Original status: proposed scope; no implementation or new run authorized by that review.
Implementation update: the [local port and CPU acceptance](../plans/x2_preview_vae_flow_matching_implementation.md#final-acceptance-2026-10-20-08) are complete.
The following text retains the planning rationale and pre-port findings.
This section updates the diffusion direction above to the existing x0 flow-matching family.

The preview-aware VAE reference is now 130-dimensional, including one sample at `t + 0.2 s`.
Conditioning stays 99-dimensional; the latent stays 32-dimensional; actions stay 31-dimensional.
The encoder can retain action-relevant preview information through its latent.
Action distillation does not guarantee reconstruction of every preview feature.

### Keep the flow core unchanged

For the existing hybrid representation, each token remains `[state(135), latent(32)]`, or 167 channels.
The window remains eight past rows, one current row, and 32 future rows at 50 Hz.
The Transformer does not consume the raw 68- or 130-dimensional encoder reference.
The existing linear noising process, x0 loss, and Euler sampler need no preview-specific mathematical change.
See [`flow_matching/trainer.py`](../../src/mjlab/tasks/tracking/flow_matching/trainer.py)
and [`flow_matching/sampler.py`](../../src/mjlab/tasks/tracking/flow_matching/sampler.py).
Changing to latent-only tokens or changing the physical-state representation is outside this minimal port.

Keep the training algorithm, but train a new model with new data and statistics.
A new VAE changes latent coordinates even when their width stays 32.
Do not reuse the old flow weights, latent statistics, or token caches as compatible artifacts.

### Port collection and VAE integration

Pre-implementation source inspection found these legacy assumptions outside the flow core:

- [`diffusion/adapter.py`](../../src/mjlab/tasks/tracking/diffusion/adapter.py)
  pins observation reference widths and `FrozenVaePolicy` to 68.
- [`diffusion/runtime_x2.py`](../../src/mjlab/tasks/tracking/diffusion/runtime_x2.py)
  constructs the batched schema with `make_schema(...)`, without cohort preview.
  Its scalar observation path also pins the reference width to 68.
- [`diffusion/replay_integration.py`](../../src/mjlab/tasks/tracking/diffusion/replay_integration.py)
  rejects a loaded VAE unless its reference width is 68.
- The single and batched closed-loop paths pin 68-wide startup references.
  [`closed_loop_runtime_x2.py`](../../src/mjlab/tasks/tracking/diffusion/closed_loop_runtime_x2.py)
  also permits only absent or zero-width `motion_lookahead` in the actor layout.

Use the loaded VAE schema and the cohort preview declaration consistently.
Preserve the legacy schema; do not replace every 68 with 130.
Update the new collection contract and cohort selection for all 144 `single_*` clips.
Bind one frozen checkpoint, its normalizers, the 0.2-second preview, and the action contract.

Collect VAE-controlled rollouts, not PPO rollouts with latents attached afterward:

```text
pre-action actual state s_t
aligned reference [q_t, dq_t, orientation_error_t, q_(t+10), dq_(t+10)]
    -> frozen encoder mean z_t
    -> frozen decoder(z_t, actual conditioning_t)
    -> clean action; apply declared collection perturbation separately
    -> simulator step
```

Store the actual decoded latent with its pre-action state.
Keep clean and executed actions separate when collection adds action noise.
Clamp preview within the same clip.
Never join training windows across resets, reference teleports, or controller changes.
Split rollout groups before fitting train-only statistics and building token caches.

### Separate preview support from goal control

Preview in the training latent does not tell deployment which future motion the operator wants.
A history-conditioned flow can learn possible continuations without an external reference stream.
Specific pose or velocity goals still need an explicit conditioning or guidance path.

The current flow evaluation runner rejects `--task-config`.
Its existing sparse-goal integration is specific to the diffusion DDIM sampler.
See [`evaluate.py`](../../scripts/experiments/diffusion_d3_closed_loop/evaluate.py)
and the [flow runbook](../source/x2_flow_matching_training.rst).
Porting inference-time guidance may preserve the training architecture and objective.
Adding a learned goal input requires suitable labels and training/model integration.
Choose this separately; neither follows automatically from the preview-aware VAE.

### Lean validation before a full run

1. Freeze the selected VAE checkpoint and make the collection/startup paths schema-aware.
2. Check one normal frame, one clip-end frame, and a subset reset for aligned preview and latent/action parity.
3. Collect a small sample and verify state-latent timing, boundaries, and unchanged 167-channel tokens.
4. Run a bounded train-save-load check and a short reference-free continuation after VAE startup.
5. Only then authorize full collection and a fresh flow training budget.

Validation here is source inspection only.
Preview-aware flow collection, training, closed-loop behavior, and goal control are not tested.
The current VAE training job is unchanged.
