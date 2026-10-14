# Plan: X2 multi-motion PPO teacher with reference preview

Date: 2026-10-14 (+08).
Status: implementation and bounded tests complete; teacher tracking quality is not qualified.
Scope: one X2 teacher trained with proximal policy optimization (PPO).
Implementation record: [software validation and usage](../reports/x2_multi_motion_ppo_teacher_implementation.md).
Design source: [agreed pipeline](../design/x2_general_teacher_goal_conditioned_latent_diffusion.md).

## 1. Target and research limits

Train one policy across an explicit list of X2 reference motions.
Give the actor both current reference features and one future reference frame.
Use the six hidden layers reported for the SCDP multi-motion policy.
Keep the existing X2 control and reward definitions.

Deliver a trainable task, checkpoint playback, and per-motion evaluation.
Do not implement VAE training, diffusion, or a robot controller in this work.

Keep the implementation small:

- Reuse the existing motion library, multi-motion queries, and PPO runner.
- Add a thin PPO-specific command and a file-based configuration.
- Use equal environment allocation and uniform phase sampling.
- Add tests only for data isolation, preview, episode boundaries, and task compatibility.
- Do not add a scheduler, plugin system, artifact registry, or new viewer.
- Do not refactor the existing distillation modules merely to change their names.

## 2. Proposed task and settings

Task ID: `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion`.
This is a new registration, not a change to existing single-motion tasks.

Use `agibot_x2_flat_tracking_increated_perturbations_vendor_gains_env_cfg` as the X2 base.
Retain its robot asset, vendor gains, action scale, torso anchor, pelvis IMU, rewards, and randomization.
Retain its no-state-estimation actor inputs.
The critic can retain the existing privileged observations.
This is an X2 adaptation, not a complete reproduction of SCDP observations.

Disable the inherited standing-start insert for the first version.
Start from perturbed reference states instead.
Standing-start and cross-motion transition training remain later research work.

| Setting | Plan value | Basis |
| --- | --- | --- |
| Actor hidden dimensions | `(1024, 1024, 1024, 1024, 512, 128)` | Operator choice; SCDP Section II-D |
| Critic hidden dimensions | `(1024, 1024, 1024, 1024, 512, 128)` | Engineering choice; paper does not specify a separate critic |
| Activation | `elu` | Existing X2 setting |
| Observation normalization | Enabled for actor and critic | Existing X2 setting |
| Action outputs | `31` | Existing X2 joint actions |
| Initial learning rate | `3e-4`, with existing adaptive schedule | Paper rate; mjlab schedule |
| Parallel environments | `4096`, configurable | Paper reference setting |
| Maximum iterations | `100000`, configurable | Paper reference cap; not launch authorization |
| Steps per environment per update | `24` | Existing X2 PPO setting |
| Other PPO settings | Unchanged | Avoid unrelated optimization changes |
| Control rate | Existing `50 Hz` | Existing X2 control contract |
| Future sample offset | `lookahead_s = 0.2` | Proposed research default; not specified by the paper |
| Training phase sampler | `uniform` | Existing multi-motion support |
| Motion allocation | Equal pinned environment counts, within rounding | Simple, balanced transition coverage |

The actor and critic receive the same future-reference sample.
The current no-state-estimation actor has 164 features with preview disabled.
The proposed sample adds 62 features, for an expected actor width of 226.
Derive the actual widths from observation terms; do not hard-code them in the network.

The paper reports about four days on an RTX 5090 in IsaacLab with G1.
X2 training time and memory cost are not measured.
The network shape is fixed for this first study; a network-size sweep is out of scope.

## 3. Motion input and command reuse

Add `motion_manifest: str` to a small PPO motion-command configuration.
Keep loaded arrays and the motion library out of the serializable environment configuration.
Build the library once, on the environment device, when the command starts.

Use one simple YAML list:

```yaml
motions:
  - data/tennis/example_a_agibot_x2_tracking.npz
  - data/tennis/example_b_agibot_x2_tracking.npz
```

These filenames are examples, not selected training motions.
Resolve relative paths from the mjlab repository root.
Require execution from that root and save the resolved ordered list with the run.
Reject an empty list, duplicate paths, nonexistent files, or incompatible arrays.
Require 31 joints and a common X2 joint/body order.
Array widths alone do not prove order; use motions from the same declared X2 conversion pipeline.

Require all clips to match the `50 Hz` control rate for this version.
Reject other rates rather than add interpolation or implicit speed changes.
Reuse `MotionLibrary.from_clips` for array, extent, and finite-value checks.
Pass the clip names through its existing identity fields; do not create or load teachers.

Reuse `stratified_slot_allocation` with equal weights.
Require at least one environment for each selected motion.
Each environment keeps its motion throughout training and samples a local phase on reset.
This balances transitions rather than episode counts.
Short clips can complete more episodes.
Their episode count does not increase their fraction of policy-update samples.

Do not call `build_multi_motion_environment` directly.
That factory requires a teacher cohort and saved teacher configurations.
Reuse the lower-level library, allocation, and command instead.

Save paths, frame counts, frame rates, allocation counts, and preview settings in the existing run directory.
Use the existing configuration files plus a small resolved-motion record if needed.
Do not build a new provenance framework.
Resume must not silently change the training motion list or preview offset.

## 4. Future-reference input

Reuse the existing `motion_lookahead` observation and clip-local query.
For each row, read:

```text
future_frame = min(current_frame + 10, clip_length - 1)
future_features = [reference_joint_q(future_frame), reference_joint_dq(future_frame)]
```

Ten reference steps equal `0.2 s` at `50 Hz`.
The future input is one frame with 31 positions and 31 velocities.
It is not a 16-frame horizon or a new body-state representation.
Keep the offset configurable through the existing `lookahead_s` field.

Clamp preview at the endpoint of the selected clip.
Never read the next clip in flat storage.
Do not delay or add sensor noise to reference features.
Keep the existing sensor delay and noise behavior unchanged.

The critic needs preview because the future return also depends on the requested next motion.
Do not add clip ID to actor or critic observations.
Later VAE work must preserve these preview features in its encoder.
That work is outside this plan.

## 5. Episode boundaries: the critical PPO adaptation

The existing multi-motion command can teleport a robot when a reference wraps.
That behavior must not create a continuous PPO transition across a teleport.

Add a small PPO-specific subclass with these rules:

1. Advance each row only inside its own clip.
2. Hold the final frame until an environment reset; do not perform an implicit wrap teleport.
3. After execution of the final-frame action, emit `clip_complete` through the termination manager.
4. Treat `clip_complete` as a terminal reference-task boundary for PPO, with `time_out=False`.
5. Count clean completion as success in evaluation, not as a physical failure.
6. Give physical failure precedence when failure and completion occur on the same step.
7. Reset only the rows whose episodes end.
8. Preserve the sampled reset frame through the first command update; do not skip frame zero.

Keep the existing episode time limit and timeout behavior separate.
A time-limit exit before the reference endpoint is not full-clip success.
This terminal-completion choice avoids new terminal-state bootstrap machinery.
It is an engineering choice, not a reported SCDP setting.

Use the existing environment step order and a task-local termination term.
The environment computes terminations before it advances command frames.
An endpoint test therefore uses the current clip-local frame after the physical step.
Do not rewrite `ManagerBasedRlEnv` or PPO return computation.
Keep the distillation command behavior unchanged.

## 6. Training, playback, and evaluation

### Training

Add an explicit multi-motion branch before the single-motion input check in `scripts/train.py`.
That branch validates the manifest.
It does not require a single file or W&B motion artifact.
Use the standard `MjlabOnPolicyRunner` for this task.
Do not pass the single-motion `registry_name` keyword to that runner.

Keep PyTorch checkpoints as the research teacher artifact.
Do not use the single-motion exporter that embeds one reference clip.
Automatic ONNX export and robot deployment are out of scope.
The existing actor-only exporter can support later distillation after the teacher qualifies.

### Playback

Support `play` with the new task, a local checkpoint, and one explicit `--motion-file`.
Build a singleton library for that selected clip.
Restore the training preview offset and network configuration before model load.
Use the existing robot viewer; omit the single-motion timeline scrubber if it cannot handle this command.
Do not add a multi-clip menu or new GUI.

### Per-motion evaluation

Adapt the local `scripts/evaluate_tracking_policy.py` path for the new task.
Evaluate one motion at a time with the same shared checkpoint.
Use start-phase initialization and a full-clip budget.
Keep the evaluation noise and randomization settings explicit.

Record only essential results:

- Full-clip completion fraction and physical failure count.
- Failure time and reference phase.
- Root-relative body-position error and joint-velocity error.
- Checkpoint, motion, seed, and evaluation settings.

Read terminal measurements before any reset.
A manual-reset evaluation loop can preserve those measurements.
The environment core remains unchanged.
Snapshot each reference before the physical step, as the existing reward/evaluator alignment requires.
Do not pair a reset robot pose with the terminal reference.

A simple loop over a motion list is sufficient.
Do not add an evaluation scheduler or dashboard.
Held-out clips use the same selected teacher task and network, not a specialist task.
Report training-library and held-out results separately.

## 7. Implementation sequence

| Step | Deliverable | Completion check |
| --- | --- | --- |
| 1. Command and input | File-based configuration, library reuse, equal allocation, task-local endpoint behavior | Critical clip/preview/reset tests pass |
| 2. X2 task and PPO configuration | New task with paper network and actor/critic preview | Correct dimensions; existing tasks unchanged |
| 3. CLI and evaluation | Manifest training, singleton playback, local per-motion evaluation | Save/reload and one real PPO update path work |
| 4. Research pilot | Small diverse motion set and per-motion results | Evidence supports expansion, or identifies the next issue |

Steps 1-3 are implemented after operator approval.
Their software checks and the [bounded real-runner smoke](../reports/x2_multi_motion_ppo_teacher_gpu_smoke.md) pass.
Step 4 needs separate execution approval and a selected motion list.
Do not implement the next VAE/diffusion stages as prerequisites for this teacher.

Likely code locations:

| Location | Small planned change |
| --- | --- |
| `src/mjlab/tasks/tracking/mdp/multi_motion_ppo.py` | New file-based config and thin PPO command subclass |
| `src/mjlab/tasks/tracking/mdp/terminations.py` | Clip-completion term |
| `src/mjlab/tasks/tracking/config/agibot_x2/env_cfgs.py` | New task factory; actor/critic preview |
| `src/mjlab/tasks/tracking/config/agibot_x2/rl_cfg.py` | Separate shared-teacher runner configuration |
| `src/mjlab/tasks/tracking/config/agibot_x2/__init__.py` | New task registration with standard runner |
| `src/mjlab/scripts/train.py`, `src/mjlab/scripts/play.py` | Narrow input branches for the new task |
| `scripts/evaluate_tracking_policy.py` | New-task singleton input and correct completion/terminal metrics |
| `tests/test_tracking_multi_motion_ppo.py` | Compact critical-feature tests |
| `configs/tracking/x2_multi_motion_pilot.yaml` | Selected pilot list after operator confirmation |

Do not move the existing library or rewrite unrelated observation and reward code.

## 8. Tests only for critical features

Keep new tests limited to critical features.
The implementation has 10 tests across three component files.
One test covers the evaluator-loop regression from review.
Reuse existing fixtures and checks rather than duplicate the full library suite.

| Critical feature | Required test |
| --- | --- |
| Motion isolation | Two synthetic clips with unequal lengths and distinct sentinel values; current references remain clip-local |
| Preview correctness | Exact offset, 62 features, per-clip endpoint clamp, actor and critic both receive preview |
| PPO boundaries | Endpoint produces done without wrap teleport; subset reset leaves other rows unchanged; reset frame is not skipped |
| Success attribution | Completion is distinct from physical failure; simultaneous failure wins; early timeout is not full-clip success |
| Task compatibility | New network has six specified hidden layers and 31 actions; config remains serializable; old task settings stay unchanged |

Include wrong-frame-rate rejection in the input test.
Check all other array validation through the reused library tests.
Do not add exhaustive tests for every CLI flag, log message, reward, or randomization setting.

Run relevant existing command and multi-motion tests after shared-path edits.
Use `uv run pytest ... -x -vv` with a focused-suite limit of 120 seconds.
Do not require the full repository suite for this research change.

One bounded real-GPU smoke must exercise the actual runner, not only fake environments.
Proposed smoke: two compatible clips, 8 environments, 10 PPO updates, and a checkpoint save/reload.
Force an endpoint and a subset reset within that smoke.
Check finite observations, actions, loss values, and at least one nonzero parameter update.

Run that command through monitored execution after authorization.
Record its exit status and exact cleanup.
A successful smoke confirms the training path, not tracking quality.

## 9. Research pilot and stop rules

Operator update, 2026-10-14 (+08): the [first all-tennis test](../reports/x2_multi_motion_ppo_all_tennis_100_iterations.md) uses all 148 available clips.
It completes 100 iterations at 4096 environments, not the earlier proposed 1000-iteration budget.
Return rises while episode length falls; inspect this behavior before any extension.
The smaller diverse-subset procedure below remains an option for diagnosis.

Select approximately 4-8 diverse feasible X2 clips for the first pilot.
The operator must confirm the exact list.
Include different motion patterns rather than many near-duplicate clips.
Retain separate held-out motions when the dataset permits.

Proposed initial pilot budget: `4096` environments and `1000` PPO updates on one managed GPU.
This is `98,304,000` environment transitions with `24` steps per update.
Measure startup memory and throughput before a longer run.
The pilot does not need a trained specialist for every clip.

Evaluate the pilot checkpoint per motion with fixed, recorded seeds.
Compare with existing qualified specialists only where compatible artifacts already exist.
Do not train new specialists just to build a baseline for this study.

Stop or revise the pilot if:

- A clip has invalid data or unsupported timing.
- The network produces non-finite values.
- Episode transitions cross reset or clip boundaries incorrectly.
- Resource use exceeds the authorized GPU budget.

Weak tracking quality calls for diagnosis, not another infrastructure layer.
Inspect reference feasibility, motion coverage, preview, and optimization first.
Add adaptive sampling or a more complex model only if evidence justifies it.
Only a later approved run can use the `100000`-iteration cap.
The paper result does not guarantee X2 performance.

## 10. Next action and evidence status

Next action: inspect reward balance and initialization/termination behavior, then authorize a scoped comparison if needed.
Use GenieStudio-managed resources by default; fleet use needs explicit authorization.

The [implementation record](../reports/x2_multi_motion_ppo_teacher_implementation.md) separates software checks from runtime evidence.
Parent validation passes 93 focused tests, with 5 unrelated generated-data skips.
The primary checkout passes all 10 new critical tests.
The [local GPU smoke](../reports/x2_multi_motion_ppo_teacher_gpu_smoke.md) completes 10 real PPO updates with reset/endpoint and checkpoint gates.
The [all-tennis test](../reports/x2_multi_motion_ppo_all_tennis_100_iterations.md) completes 100 iterations at 4096 environments.
No extension beyond that budget or hardware change occurs.
Teacher quality remains unqualified; full training cost remains not measured.

## Multi-GPU scaling: measured once, 2026-10-14 (+08)

A two-GPU test now exists: [two-GPU 8192-environment report](../reports/x2_multi_motion_ppo_2gpu_8192env_100iterations.md).
One management container, GPUs 3 and 4, 8192 environments per rank, 148 clips, 100 iterations, exit 0.
Both ranks trained, and rank 1 used `seed 8`.
The two-GPU run is better than the single-GPU 4096-environment reference on every logged scalar.
Mean return becomes positive (`0.23648756444454194`), and mean episode length ends at `0.3375800018310547 s` instead of falling to `0.16377999687194822 s`.
The test changed environment count, GPU count, and hardware at the same time.
It cannot separate those effects and is not a scaling-efficiency measurement.
It runs no per-motion or held-out evaluation; tracking quality remains unqualified.

### Source support, from inspection only

The existing training CLI supports one shared teacher across GPUs on one server.
It launches one `torchrunx` worker per GPU; each worker receives the full manifest and environment count.
Each worker uses `seed + global_rank` for independent reference phases and perturbations.
RSL-RL averages gradients and combines observation-normalizer statistics across workers.
This task has no multi-GPU runtime test; this capability check authorizes no remote launch.

With 4096 environments and 24 rollout steps **per GPU**, global batches are:

- 2 GPUs: 8192 environments and 196608 transitions per update.
- 4 GPUs: 16384 environments and 393216 transitions per update.
- 8 GPUs: 32768 environments and 786432 transitions per update.

All 148 clips already appear on each GPU; scaling adds independent phases and actual robot states.
Record the unique reference-frame coverage per clip. Do not choose a larger batch from coverage assumptions alone.
Short failed episodes can cover reference phases. That is not proof of continuous tracking.
Default TensorBoard episode curves report rank 0, not a combined global quality measure.
Use a bounded smoke before a larger pilot; resource and iteration budgets need authorization.
The `--gpu-ids` launch path uses one host only.
The team queue script `run_task.sh` pins one GPU per job, so multi-GPU needs a direct launch inside a container.
Fixed iterations multiply transitions with GPU count.
Iteration time and convergence remain unmeasured for a controlled comparison.

Source: [training launcher](../../src/mjlab/scripts/train.py), [task allocation](../../src/mjlab/tasks/tracking/mdp/multi_motion_ppo.py), and installed RSL-RL PPO/normalization code.

References:

- [Existing pipeline discussion](../design/x2_general_teacher_goal_conditioned_latent_diffusion.md)
- [Motion library](../../src/mjlab/tasks/tracking/distillation/motion_library.py)
- [Multi-motion command](../../src/mjlab/tasks/tracking/distillation/multi_motion.py)
- [Environment step order](../../src/mjlab/envs/manager_based_rl_env.py)
- [X2 PPO settings](../../src/mjlab/tasks/tracking/config/agibot_x2/rl_cfg.py)
- [SCDP paper](https://arxiv.org/abs/2603.09574v1), Section II-D; local file `~/Downloads/scdp.pdf`
