# X2 multi-motion PPO teacher: implementation and validation

Date: 2026-10-14 (+08).
Scope: software implementation for one shared X2 teacher.
Status: integrated locally; bounded tests complete; teacher tracking quality is not qualified.

## 1. Implementation

New task: `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion`.
The task reuses the X2 vendor-gain configuration, rewards, actions, and randomization.
It does not change existing specialist tasks or VAE/diffusion contracts.

| Item | Implemented behavior |
| --- | --- |
| Actor and critic | Hidden dimensions `(1024, 1024, 1024, 1024, 512, 128)` |
| Reference preview | One future joint-position/velocity sample at `0.2 s`; configurable with `lookahead_s` |
| Preview inputs | Actor and critic both receive 62 future-reference features |
| Actor / critic contract | 226 / 412 features at the default preview setting; measured in the real GPU smoke |
| Actions | 31 existing X2 joint actions |
| PPO defaults | Initial rate `3e-4`, adaptive schedule, 24 rollout steps, 4096 environments, 100000-iteration cap |
| Motion input | YAML `motions:` list, or one explicit clip for playback/evaluation |
| Motion allocation | Equal pinned row counts, within rounding; uniform phase resets |
| Clip completion | Terminal reference-task boundary; no implicit wrap teleport |
| Initialization | Perturbed reference states; standing-start insert disabled |
| Checkpoints | Standard PyTorch runner; no embedded single-clip ONNX export |

The paper supplies the actor network shape.
The same critic shape is an engineering choice.
The preview offset is also an engineering choice, not a reported paper value.

Main interfaces:

- [`MultiMotionPpoCommandCfg` and command](../../src/mjlab/tasks/tracking/mdp/multi_motion_ppo.py)
- [X2 task factory](../../src/mjlab/tasks/tracking/config/agibot_x2/env_cfgs.py)
- [X2 PPO configuration](../../src/mjlab/tasks/tracking/config/agibot_x2/rl_cfg.py)
- [Training entry point](../../src/mjlab/scripts/train.py)
- [Playback entry point](../../src/mjlab/scripts/play.py)
- [Local per-clip evaluator](../../scripts/evaluate_tracking_policy.py)

## 2. Review fixes

The first Luna review blocks on two evaluator defects.
The original CLI worker fixes them and adds one evaluator-loop regression.
The retained reviewer then returns **OK with notes**, with no further findings in the reviewed fixes.

| Finding | Final behavior |
| --- | --- |
| Failure coincides with completion | OR physical-failure terms separately from `clip_complete`; failure wins |
| Reset changes the observation buffer | Pass the returned reset observations to the next policy call |
| Inactive row terminates again | Reset every done row; count only each row's first episode |
| Failure timestamp excludes the final step | Record elapsed post-step time, `(step + 1) * step_dt` |
| Loader cannot detect column permutations | Declare trusted X2 joint/body order as an input precondition |

The loop regression uses a real `TerminationManager` and a scripted environment.
It checks terminal measurements before reset, fresh observations, repeated inactive-row failures, and simultaneous failure/completion.
It does not establish live simulation behavior.

Three scoped writers use `ds-oc/deepseek-flash:high`.
Both reviews use `lz-fast/gpt-5.6-luna:max`.
The parent inspects the candidate and performs independent checks.

## 3. Validation evidence

Later runtime evidence, 2026-10-14 (+08): [local GPU smoke PASS](x2_multi_motion_ppo_teacher_gpu_smoke.md).
It verifies 10 real PPO updates, endpoint/reset gates, and in-process checkpoint reload.
The software results below remain separate from that runtime record.
The [all-tennis 100-iteration test](x2_multi_motion_ppo_all_tennis_100_iterations.md) also completes at 4096 environments.
Its aggregate return rises, but episode length falls; teacher quality remains unqualified.

Evidence directory: [`logs/diagnostics/x2-multi-motion-ppo-20261014/`](../../logs/diagnostics/x2-multi-motion-ppo-20261014/).

| Check | Result | Scope |
| --- | --- | --- |
| Parent focused suite in isolated worktree | **93 passed, 5 skipped** | 10 critical tests plus relevant command, runner, task, and X2 regressions |
| Parent critical tests in primary checkout | **10 passed** | Command, configuration, CLI, and evaluator-loop behavior |
| Ruff format/lint | Pass | 10 changed code/test files |
| Ty | Pass | 10 changed files, with `--extra-search-path tests` |
| Pyright | **0 errors, 0 warnings** | 10 changed files; explicit shared virtual-environment path |
| Train/play/evaluator help | Pass | Actual entry points and public flags; no environment launch |
| Integration hashes | **10/10 match** | Isolated candidate files equal primary files |
| Unrelated D5 preservation | **4/4 unchanged** | Source, test, and documentation hashes before/after integration |
| G1 action-scale test before integration | Existing failure reproduced | Unrelated baseline failure; no G1 source changes |

Five skipped standing-start checks require generated motion data absent from the isolated worktree.
No new critical test is skipped.
The full repository test suite is not run.

The fixture import initially fails in Ty because its test directory is not a search path.
The explicit test search path resolves it without a source or dependency change.
Pyright finds one fake-environment argument mismatch in a test.
An explicit test-only cast fixes it; the four CLI tests pass afterward.

The primary checkout advances from `a955208d1` to `7e5068f54` during delegated work.
Those newer D5 commits remain intact.
The parent applies only the six changed source files and four new code/test files.
No commit or push occurs.

Raw records:

- [Parent focused tests](../../logs/diagnostics/x2-multi-motion-ppo-20261014/parent-focused-tests.log)
- [Primary critical tests](../../logs/diagnostics/x2-multi-motion-ppo-20261014/primary-critical-tests.log)
- [Final review](../../logs/diagnostics/x2-multi-motion-ppo-20261014/review-final.md)
- [Initial review](../../logs/diagnostics/x2-multi-motion-ppo-20261014/review-initial.md)
- [Integration receipt](../../logs/diagnostics/x2-multi-motion-ppo-20261014/integration-receipt.json)
- [Baseline G1 failure](../../logs/diagnostics/x2-multi-motion-ppo-20261014/baseline-g1-action-scale.log)

## 4. Use

Run commands from the mjlab repository root.
Create a manifest with actual compatible reference paths:

```yaml
motions:
  - data/tennis/your_first_x2_tracking.npz
  - data/tennis/your_second_x2_tracking.npz
```

These paths are examples.
The separate all-tennis test uses this [148-file manifest](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/motions.yaml).
The clips must use the same compiled X2 joint/body order and the `50 Hz` reference rate.
The loader checks rate, dimensions, finite values, and clip-local bounds.
It cannot infer semantic column order from unnamed NPZ arrays.

### Train

This command starts training if executed.
The manifest and GPU budget require operator selection first.
The small iteration override prevents accidental use of the default 100000-iteration cap.
It is an example, not a completed or approved run.

```bash
uv run train Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion \
  --env.commands.motion.motion-manifest configs/tracking/my_x2_motions.yaml \
  --env.scene.num-envs 4096 \
  --env.commands.motion.lookahead-s 0.2 \
  --agent.max-iterations 1000
```

The run writes `params/env.yaml`, `params/agent.yaml`, and `params/resolved_motions.yaml`.
The resolved record stores clip identities, extents, rates, allocation, and preview settings.
A local resume rejects a changed saved clip mapping or preview offset.
A checkpoint without its saved record emits a warning; resume compatibility is then unverified.
Retain the `params/` directory from each run with its checkpoints.

### Play one clip

Use the same task and one explicit reference clip.
The entry point restores saved preview settings and clears the training manifest.
The fixed network shape comes from the task configuration.
The viewer omits only the unsupported single-motion command GUI.

```bash
uv run play Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion \
  --checkpoint-file logs/rsl_rl/agibot_x2_multi_motion_tracking/RUN/model_500.pt \
  --motion-file data/tennis/your_first_x2_tracking.npz
```

### Evaluate one clip

Run the same checkpoint on each clip separately.
The evaluator distinguishes clean completion, physical failure, and early timeout.
It saves source settings, completion counts, failure time/phase summaries, and tracking errors.
The body-position metric uses root-relative coordinates.

```bash
uv run python scripts/evaluate_tracking_policy.py \
  --task Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion \
  --checkpoint-file logs/rsl_rl/agibot_x2_multi_motion_tracking/RUN/model_500.pt \
  --motion-file data/tennis/your_first_x2_tracking.npz \
  --output-json logs/diagnostics/your_first_x2_eval.json
```

All checkpoint and motion paths above are examples.
No qualified shared-teacher checkpoint exists from this work.
The [smoke checkpoint](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/run-retry1/model_9.pt) is diagnostic only.

## 5. Operational status and limits

- The [local GPU smoke](x2_multi_motion_ppo_teacher_gpu_smoke.md) passes with 8 environments and 10 real PPO updates.
- The [all-tennis test](x2_multi_motion_ppo_all_tennis_100_iterations.md) completes 100 iterations with all 148 files and 4096 environments.
- Aggregate reward does not establish tracking quality; episode length decreases and height-error terminations increase.
- No extended production run, deployment, or robot operation occurs.
- In-process checkpoint save/reload is tested; fresh-process resume and interactive playback remain not tested.
- Production throughput, full GPU-memory peak, inference latency, and tracking quality remain not measured.
- Standing-start capability and arbitrary clip transitions remain not tested.
- Existing specialist tasks and later VAE/diffusion stages remain unchanged.

Next action: inspect reward balance and initialization/termination behavior before a longer run.
A new experiment or reward change requires operator authorization.
Use managed GPU resources by default; fleet use needs explicit authorization.

Related design: [shared teacher and goal-conditioned latent diffusion](../design/x2_general_teacher_goal_conditioned_latent_diffusion.md).
