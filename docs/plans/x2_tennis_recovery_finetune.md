# X2 tennis-end recovery fine-tuning — implementation plan

## Goal and scope

Fine-tune the existing X2 no-state-estimation velocity policy to recover to standing from tennis-ending states, while retaining ordinary velocity-control capability.

First experiment:

| Environment group | Episode reset | Commands for the episode |
|---|---|---|
| 80% recovery | Uniformly choose a training trajectory, then one of its last 10 frames; restore its full state | Exactly zero planar velocity and yaw rate, including later command resamples |
| 20% retention | The original task's vanilla reset | The original velocity-command distribution and curriculum |

This is an opt-in task variant. Do not change the behavior of the existing velocity or tracking tasks. Do not introduce a new observation, action limiter, recovery reward, physics change or hardware-controller change in this first experiment.

Implementation and bounded local validation are authorized. Full training, remote GPU allocation, uploads, deployment and robot access are not part of implementation; supply a verified launch recipe instead.

## Checked inputs

- Repository: `/home/agiuser/projects/mjlab`.
- Planning-time checkout: clean `fxy/test/tracking-exp`, HEAD `4de481a9468b769dcb1fe8456a09b6b852fab56e`. This differs from the earlier vendor-play working tree. Recheck before editing; do not switch branches or restore old changes implicitly.
- Base task: `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation`.
- Source run: `logs/rsl_rl/agibot_x2_velocity/2026-09-16_01-53-44_x2-velocity-measured-gains/`.
- Source checkpoint: `model_19999.pt`; SHA256 `a8450c34d6fbea0a61ec8086707890bc8f4444aef7a79c6e456b1ee48e01f874`.
- Source run's `params/env.yaml` and `params/agent.yaml` are the configuration baseline; do not assume today's defaults reproduce every original setting.
- Dataset: `data/tennis/*.npz`, currently 144 files at 50 Hz. Last 10 frames provide 1,440 candidate states before train/validation separation.
- NPZ fields: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `fps`. Files do not contain joint/body name lists.
- Original resets use default joint configuration and zero root/joint velocities. Original commands have 10% standing probability, 3–8 s resampling and a velocity-range curriculum. Original episodes last 20 s.

## Design decisions

### 1. Immutable endpoint dataset and explicit ordering

Build a small reusable endpoint-pool loader under the velocity task, not a dependency on vendor playback or the tracking command's runtime.

- Load lazily at environment construction, not module import/task registration. Ordinary task listing must work without the tennis dataset.
- Validate file list, frame counts, common lengths, finite values, 31-joint/32-body shapes, positive compatible FPS and nondegenerate quaternions. Define informative failure behavior for empty directories and files shorter than the requested window; default to fail rather than silently taking an unexpected sample count.
- Establish and test the NPZ ordering against the X2 conversion/asset contract. Verify pelvis/root indexing; do not infer names from dimensions alone. Verify reference joint/body FK against the current model on representative frames. A material ordering/geometry mismatch is a blocker, not permission to guess.
- Preserve source path, original frame index, FPS and file hash for every candidate. Record model/order fingerprint, seed and split settings in a run manifest. Original NPZs remain unchanged.
- Form a deterministic trajectory-level train/validation split (default validation fraction 0.2, seed 42). This is separate from the 80/20 environment mix. Never put neighboring frames of one trajectory on both sides. With 144 files this is approximately 115 training and 29 validation trajectories; document the exact rounding algorithm.
- Allow explicit train/validation/all selection; training defaults to train, evaluation to validation. Sample trajectory uniformly and then frame uniformly.
- Diagnose invalid joint limits, severe ground penetration or self-intersection before training. Do not silently clip/drop difficult states or clamp velocities; report exclusions or contract failures explicitly. Keep feasible moving/airborne states.

### 2. Paired reset and command behavior

Use a task-local reset event plus a specialized velocity command/config (or an equally narrow equivalent), sharing one explicit environment-group mask.

- Group assignment is fixed for the environment lifetime, seeded and deterministic. Frame selection is repeated on each episode reset. Document rounding and tiny-num-env behavior; support forced recovery/retention modes for tests and evaluation.
- Preserve the original reset sequence for retention envs. Apply full reference state only to recovery envs being reset, without overwriting non-reset envs or domain-randomized physical parameters.
- Restore joint positions/velocities, root orientation, root linear/angular velocity and root height. Recenter horizontal location to the environment origin. If applying yaw augmentation, rotate world velocities consistently; preserve authored roll/pitch. Keep ground-relative geometry rather than imposing the default pelvis height.
- Confirm root-link versus COM and world versus body velocity conventions using the entity write APIs. Write state before the existing reset forward pass; do not read stale derived kinematics during reset.
- Recovery commands remain exactly zero after reset, timer expiry, heading updates and curriculum changes. Zero command must never zero physical qvel. Retention retains the original command generation and curriculum semantics.
- Keep the existing 20 s episode horizon and rewards for the first comparison. Do not reset at PPO buffer boundaries or at command resampling.
- Retain the ordinary action/history reset semantics for this MVP and document them. Do not fabricate previous actions from a reference pose. This does not reproduce the deployment controller's warm shadow-policy state; realistic handover-history sampling is a later extension, not a claim of this implementation.

Current lifecycle to respect: `_reset_idx` applies reset events, then resets observations/actions/rewards/metrics/curriculum/commands/events/terminations. `reset()` and auto-reset later forward physics and compute commands. Tests must cover both paths and partial resets.

### 3. Opt-in task and verified checkpoint continuation

Suggested task ID: `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery`.

- Build from the original X2 velocity config; keep actor/critic dimensions and ordering, sensor frames, default positions, action scale, PD gains, observation noise, domain randomization, reward weights and termination logic unchanged unless a baseline comparison identifies drift requiring explicit resolution.
- Expose dataset directory, last-N-frames (10), recovery fraction (0.8), dataset split and seed through ordinary typed config/CLI, not machine-specific hardcoding in library code.
- Preserve actor, critic, normalizers and checkpoint iteration/curriculum state through the existing migration/load path. Test with the actual source checkpoint, not only a synthetic checkpoint.
- Proposed initial fine-tuning learning rate: `1e-4`, with the original adaptive-KL schedule initially retained. This is an initial rate, not a guaranteed upper bound. Log effective rate and KL.
- Important: installed rsl-rl restores the optimizer's saved LR on load. Merely changing YAML `learning_rate` does not set the post-load rate. Implement a narrow explicit fine-tuning override applied after loading to both algorithm LR and optimizer groups; keep original task/resume behavior unchanged. Preserve optimizer moments if load-compatible. Escalate incompatibility rather than quietly resetting training state.
- Distinguish the first fine-tune initialization override from an ordinary continuation of a recovery checkpoint; normal recovery-run resume should retain its saved optimizer/LR state unless explicitly overridden again.
- Simplest launch compatibility: retain experiment root `agibot_x2_velocity` and use a new unique run name such as `x2-tennis-end-recovery`, selecting the source run/checkpoint with exact anchored patterns. The existing resolver searches inside the destination experiment root and does not accept an arbitrary source path as `load_run`. Do not document a nonfunctional cross-experiment resume recipe.
- All new logs/checkpoints go to a new timestamped run. Never overwrite the source checkpoint/run. Check how `max_iterations` is interpreted after resume and state the additional update count precisely.
- Produce and actually validate the CLI recipe. No auto-start of a production training run.

### 4. Evaluation and observability

Provide a bounded offline/simulation evaluation entry point or extend an existing appropriate one without changing ordinary playback defaults.

- Compare source and candidate on identical held-out states/seeds with zero commands. Include an original-task retention evaluation with original nonzero commands.
- Report per-group and per-trajectory results, not only an aggregate reward: fall/termination rate, time to settle, signed backward displacement relative to initial heading, total planar displacement, peak tilt and command/torque diagnostics with requested versus applied semantics identified.
- Proposed settling definition: planar speed <0.1 m/s, absolute yaw rate <0.2 rad/s and torso tilt <10 degrees, sustained for 0.5 s. Make thresholds configurable and report them; count failure to settle as censored/failure, not zero seconds.
- Report the first 3 s after reset separately from the full 20 s episode. Do not let long settled periods conceal poor initial recovery.
- Log episode/step counts, reset-source IDs and coverage for each group. Preserve the split manifest for reproducibility. No leakage of file IDs/group IDs into actor observations.
- Do not claim improvement from successful smoke tests. A future trained candidate must beat the frozen source baseline on held-out recovery without unacceptable retention degradation. Backward travel alone must not reward falling or prohibit necessary stabilizing steps.
- Further work, explicitly deferred: pool actual tracking-rollout handover states, match warm action histories, add perturbation curricula, shorten recovery episodes or reshape recovery rewards.

## Implementation sequence and ownership

Keep one writer per checkout. Before delegation, recheck branch/status and establish an exclusive edit boundary. This is a paired reset/command feature with configuration, evaluation and resume integration seams; if split across workers, use sequential exclusive owners with explicit interfaces and one integration owner rather than concurrent edits to shared task files.

1. **Pool contract:** loader, deterministic split/manifest, ordering/geometry checks and synthetic unit tests.
2. **Recovery behavior:** reset event, command subclass, opt-in config/registration, lifecycle and unchanged-base-task tests.
3. **Continuation and evaluation:** verified source checkpoint load/LR override, bounded evaluation/metrics, documentation/CLI and integration tests.
4. **Parent acceptance:** inspect complete diff, run focused regression/type checks, verify evidence and report remaining limitations. Request rework through the same delegated protocol if needed.

Likely touch points (prefer new focused files; exact names can follow repo conventions):

- `src/mjlab/tasks/velocity/mdp/` endpoint-pool/reset/command support and exports.
- `src/mjlab/tasks/velocity/config/agibot_x2/` opt-in env/runner config and registration.
- `src/mjlab/tasks/velocity/rl/` narrow recovery runner if required for explicit load behavior.
- `src/mjlab/tasks/velocity/scripts/` bounded evaluation if no existing suitable entry point.
- Focused `tests/test_x2_tennis_recovery_*.py` and existing velocity/reset regressions.
- User-facing documentation and `docs/source/changelog.rst` Upcoming entry.

Do not edit `x2_docker`, shared robot XML/physics, installed dependencies, unrelated vendor work or dataset inputs. Do not commit, reset, stash or switch branches without authorization.

## Acceptance tests

### Data and configuration

- Exactly the intended final frames, deterministic trajectory splits, no split leakage, reproducible sampling and source IDs/hashes.
- Reject malformed dimensions, mismatched ordering, NaNs, degenerate quaternions, empty datasets and incompatible FPS with actionable errors.
- Task registry import/list and base task construction do not read the dataset.
- Original task configuration/observation/action contracts remain unchanged; source checkpoint loads strictly and normalizers are preserved.

### Reset/command correctness

- Known nonzero reference qvel/root velocities survive a recovery reset and first command update.
- Recenter/yaw transforms are consistent and root height is retained; FK/geometry agrees with source conventions.
- Recovery command is zero through multiple resamples/heading updates; retention can issue nonzero commands.
- Partial reset leaves other states, masks, commands, timers and history buffers unchanged.
- Explicit reset and auto-reset behave consistently. No reset or physical state write occurs on command timer expiry/PPO buffer boundaries.
- Original reset behavior is retained for the retention group. Test against baseline with controlled RNG inputs/distributions rather than assuming identical global RNG consumption.

### Continuation and bounded integration

- Actual `model_19999.pt` load smoke: finite deterministic inference before any update, compatible actor/critic, normalizer preservation, restored counters and correct effective post-load LR.
- Ordinary resume versus explicit first-fine-tune LR override are tested separately.
- Run small local reset/step smoke and, if resources permit, only 1–2 PPO updates with <=64 envs. Bound wall time and logs; do not allocate remote machines or launch long training. If GPU validation is unavailable, report that gap explicitly.
- Check ONNX export/metadata and inference parity with the existing standing-policy observation/action contract; distinguish export success from hardware validation.
- Run scoped `uv run pytest ...`, `uv run ty check`, `uv run pyright`, and scoped Ruff checks. Follow `AGENTS.md`; always use `uv run` for Python.
- Provide changed files, exact validation commands/results, output paths, CLI recipe, known limitations and any skipped gates. Do not equate test passage with a better trained policy.

## Delegation preflight and ownership board

**Scheduling update:** the operator subsequently approved parallel execution. The active ownership/interface contract is [x2_tennis_recovery_parallel_contract.md](x2_tennis_recovery_parallel_contract.md), which supersedes the sequential scheduling below while preserving the experiment and acceptance requirements.

After the operator reloaded the session, `subagent action=models` confirms `lingzhi/glm-5.2`. Use that exact model for every implementation stage. The operator prefers GLM-5.2 to GLM-5.3 for detailed implementation briefs.

Classification: **multi-seam**, executed as one strictly sequential lane. The existing checkout has only this untracked plan; ignored dataset/checkpoint/environment assets are needed for validation. Use the current checkout deliberately, with exclusive non-overlapping component ownership and no overlapping worker lifetimes, rather than staging/committing the plan or silently changing branches to obtain managed worktrees. The parent does not edit source while a worker owns it.

All stages use `/home/agiuser/projects/mjlab`, branch `fxy/test/tracking-exp`, base HEAD `4de481a9468b769dcb1fe8456a09b6b852fab56e`. Isolation is sequential execution plus exclusive file ownership, not parallel worktrees. Before each stage recheck that HEAD/branch match and that preceding changes are retained. Each component must pass its focused tests and publish a runtime-bound handoff before the next starts. On setup failure or an unresolved contract, stop the lane rather than bypassing it.

| Stage | Exclusive component ownership | Focused gate / durable handoff |
|---|---|---|
| `pool` | New endpoint-pool module and its data/geometry tests | Pool schema, ordering, split and sampling tests; exported API and actual-data diagnostics |
| `runtime` | New paired recovery reset/command module, opt-in environment config factory and lifecycle tests; no shared registration yet | Full-state, zero-command, group/partial-reset and baseline-contract tests; factory/API handoff |
| `resume` | New recovery runner/config support and checkpoint/LR tests | Actual source checkpoint compatibility, normalizers/counters and explicit LR override tests; runner factory handoff |
| `evaluation` | New bounded recovery/retention evaluation entry point, metric helpers and their tests | Held-out sampling, signed displacement, settling/censoring and bounded CLI tests |
| `integration` | Shared task registration/exports, user guide/changelog and integration tests; wiring only | Focused combined suite, scoped type/lint checks, verified launch recipe, bounded local smoke and export checks |

The integration owner may correct wiring inconsistencies, not redesign earlier components or silently weaken their tests. Escalate substantive component defects to the parent. Parent acceptance/review follows the lane's completion; implementation is not proof of trained-policy improvement.
