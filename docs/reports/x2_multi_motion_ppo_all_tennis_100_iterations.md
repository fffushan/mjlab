# X2 shared PPO teacher: all-tennis 100-iteration test

Date: 2026-10-14 (+08).
Execution status: complete, exit code `0`.
Tracking-quality status: not qualified.
Scope: one local-GPU test from scratch, with all available tennis-directory NPZ files.

## 1. Configuration and integrity

The operator requests all reference files, 4096 environments, and 100 PPO iterations.
PPO means proximal policy optimization.
The test preserves the selected six-layer network and `0.2 s` reference preview.
It does not change reward weights, physics limits, motion sampling, or robot gains.

| Item | Value |
| --- | --- |
| Task | `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion` |
| GPU | Local NVIDIA GeForce RTX 5080 Laptop GPU, index 0 |
| Files | `148`: 144 `single_*_zhanghongyu_agibot_x2_tracking.npz` files and 4 `standing_*.npz` files |
| Data extent | `47653` frames; each clip has 213-500 frames |
| Input contract | All files have 31 joints, 32 bodies, and a `50 Hz` reference rate |
| Allocation | 100 clips receive 28 environments; 48 clips receive 27 environments |
| Family allocation | 3988 single-motion environments; 108 standing-reference environments |
| Actor / critic | Hidden layers `(1024, 1024, 1024, 1024, 512, 128)`; inputs 226 / 412 |
| Training | From scratch, seed `7`, 100 iterations, 24 steps per iteration |
| Optimizer | 5 epochs × 4 minibatches; initial rate `3e-4`, adaptive schedule |
| Completed transitions | `9830400` |
| Completed optimizer steps | `2000` for each populated optimizer state |
| Logging | Local TensorBoard, model upload disabled |
| Bounds | 1200-second service limit, 24 GiB host-memory cap, zero host-memory swap allowance |

All 148 files pass the shape, frame-rate, and finite-value inventory checks.
These checks do not prove physical feasibility or semantic joint/body order.
The test uses the existing X2 conversion-order input precondition.

The checkpoint preserves the requested six hidden layers for both networks.
Actor and critic parameter states remain finite.
All 148 input hashes and 7 selected source hashes match after the run.
The resolved clip order matches the requested manifest exactly.
No reference file is excluded or changed.

## 2. Reward curve

![Reward and episode-length curves](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/reward_curve.png)

The blue curve shows `Train/mean_reward`.
It is a rolling mean of the last 100 completed-episode summed returns.
The orange curve applies a trailing 10-iteration mean to those logged values.
The lower panel shows episode length from the same completed-episode buffer in seconds.
The iteration axis is one-based; raw TensorBoard steps are 0-99.

| Measurement | First 10 iterations | Last 10 iterations |
| --- | --- | --- |
| Mean logged episode return | `-1.8664690971374511` | `-0.12035190686583519` |
| Mean episode length, seconds | `0.3820200004577637` | `0.16377999687194822` |
| Foot/wrist height-error termination count | `185.40833740234376` | `482.71668395996096` |
| Clip-completion term count | `19.86250057220459` | `12.275000286102294` |
| Mean exploration standard deviation | `0.9808232545852661` | `0.8078201234340667` |

Termination entries are means of the logged per-batch-step row counts.
They are not disjoint outcome rates; several terms can fire on the same row.
Clip-completion terms therefore do not establish clean completion success.

The first logged return is `-1.8745206594467163`.
The last logged return is `-0.10720405727624893`.
The highest logged return is `-0.10370924323797226`, at iteration 93.
The final value loss is `0.0037575860042124987`.
The final adaptive learning rate is `3.950617247028276e-05`.
All exported reward, loss, and performance scalars remain finite.

## 3. Interpretation

The episode-return curve becomes less negative across this test.
However, episodes become shorter, and the foot/wrist error term fires more often.
The return increase is not evidence of successful motion tracking.
No per-motion, held-out-reference, or full-clip quality evaluation occurs.

The height-error term uses a `0.25 m` reference-error threshold for ankles and wrists.
It is not a direct physical-fall detector.
Do not describe its count as a measured fall rate.

The recorded action-rate penalty becomes smaller:

| Logged component | First 10 iterations | Last 10 iterations |
| --- | --- | --- |
| `Episode_Reward/action_rate_l2` | `-0.21321823298931122` | `-0.04551772847771644` |
| `Episode_Reward/joint_limit` | `-0.028615053184330462` | `-0.003969522938132286` |
| `Episode_Reward/self_collisions` | `-0.011751626618206501` | `-0.000264498709293548` |
| `Episode_Reward/motion_body_pos` | `0.02851585205644369` | `0.014416929706931115` |

The positive accumulated tracking components do not rise in this window comparison.
These components divide completed-episode sums by the configured maximum episode duration, not the actual episode duration.
Shorter episodes can therefore reduce these values.
This does not prove a larger tracking error.
The diagnostic ratio of return-window mean to duration-window mean also becomes less negative: `-4.885788950580897` to `-0.7348388641131347`.
This ratio is not a per-motion tracking score.

### Source-derived reward risk

The current action-rate term applies to raw actions before joint-target scaling:

```text
r_action_rate_raw = -0.1 * sum((a_t - a_previous)^2)
```

The six positive tracking weights sum to `5.0`.
Each tracking exponential is at most one.
At initialization, the Gaussian exploration standard deviation is `1.0` for each of 31 actions.

A simple constant-mean, independent-noise scale check gives:

```text
E[sum((noise_t - noise_previous)^2)] = 2 * 31 * 1^2 = 62
E[r_action_rate_raw] = -6.2
maximum_positive_tracking_reward_raw = +5.0
```

This comparison excludes reset steps and changes in action means.
It is not a measurement of the actual policy action-rate cost.
Both contributions receive the same control-period scaling in the reward manager.

The task has no separate survival bonus or physical-failure penalty.
Net-negative future return can make early termination an attractive local strategy.
The observed curves warrant this check, but they do not prove reward hacking or a specific cause.
A learned policy can also change action means to reduce the action-rate cost.

Source references:

- [Raw action-rate function](../../src/mjlab/envs/mdp/rewards.py)
- [Reward weights and termination thresholds](../../src/mjlab/tasks/tracking/tracking_env_cfg.py)
- [Reward scaling and episode-component normalization](../../src/mjlab/managers/reward_manager.py)

## 4. Runtime and cleanup

The journal records start at `2026-10-14T17:02:17+08:00` — `17:02:17 (+08)`.
It records completion at `2026-10-14T17:05:28+08:00` — `17:05:28 (+08)`.
Systemd service runtime is `191.048 s`.
Logged collection time totals `96.576975107193 s`.
Logged learning time totals `83.75536012649536 s`.
These internal timings exclude some setup, logging, and checkpoint work.

The owned unit is inactive with `MainPID=0`.
No local GPU compute process appears in the cleanup observation.
Total GPU-memory peak is not measured.
Unrelated work remains unchanged.

## 5. Artifacts

Root: [`logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/`](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/).

- [Plot](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/reward_curve.png)
- [Raw scalar CSV](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/metrics.csv)
- [Curve summary](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/curve-summary.json)
- [Reward-component summary](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/component-summary.json)
- [Reward diagnostic](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/reward-diagnostic.json)
- [Artifact audit](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/audit.json)
- [Inventory](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/inventory.json)
- [Manifest](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/motions.yaml)
- [Raw training log](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/train.log)
- [Cleanup observation](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/cleanup.txt)

Checkpoints: `model_0.pt`, `model_25.pt`, `model_50.pt`, `model_75.pt`, and `model_99.pt`.
Final checkpoint: [`model_99.pt`](../../logs/diagnostics/x2-multi-motion-ppo-all-tennis-4096env-100iter-20261014/training/agibot_x2_multi_motion_tracking/2026-10-14_17-02-21_all-tennis-4096-100/model_99.pt).
Final checkpoint SHA-256: `38d3d0c699cbc8a3ca024bc56ccfdb69c9b39e684187cbaa1f83c3a3cf93c267`.
The checkpoint records iteration 99 and common step counter 2400, which correspond to 100 completed iterations.

## 6. Next action

Do not qualify the general teacher from aggregate return alone.
Before a longer run, inspect reward balance and initialization/termination behavior.
A small per-motion check or one-factor exploration-noise comparison can distinguish these risks.
Any new run or reward change needs operator authorization.
No extension beyond the requested 100 iterations occurs.
No remote GPU job, VAE/diffusion change, deployment, or robot operation occurs.

Related records: [implementation](x2_multi_motion_ppo_teacher_implementation.md), [local smoke](x2_multi_motion_ppo_teacher_gpu_smoke.md), and [plan](../plans/x2_multi_motion_ppo_teacher.md).
