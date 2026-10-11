# Memo: Footstep Tracking Experiments for G1 29DOF Mode-15 (qianghuo motion)

Date: 2026-09-11. Task: `Mjlab-Tracking-Flat-Unitree-G1-29DOF-Mode-15-No-State-Estimation`,
motion: `data/qianghuo_smplx_unitree_g1_29dof_mode_15_tracking.npz` (2286 frames @ 50 Hz = 45.72 s).

## Problem

The policy tracked the upper body well but took visibly **smaller steps than the reference**,
especially during the 3 big forward steps at the beginning of the clip. Feet are not tracked
below the ankle: in this model `left_foot`/`right_foot` are only visual *sites* — the foot geoms
hang off `left/right_ankle_roll_link`, and the tracked-body list (14 bodies) stops at the ankles.

## What we changed

1. **Foot reward terms** — `src/mjlab/tasks/tracking/config/g1_29dof_mode_15/env_cfgs.py`
   - `motion_feet_pos`: `motion_relative_body_position_error_exp` on
     `("left_ankle_roll_link", "right_ankle_roll_link")`, weight **0.3**, std 0.3
   - `motion_feet_lin_vel`: `motion_global_body_linear_velocity_error_exp` on the ankles,
     weight **1.0**, std 1.0
   - Reward functions already supported `body_names` filtering; zero new reward code.
2. **Weighted init-frame sampling** — `src/mjlab/tasks/tracking/mdp/commands.py`
   - New `sampling_mode="weighted"` + `MotionCommandCfg.init_weight_s` (seconds).
   - Start frames sampled from a truncated exponential `p(t) ∝ exp(-t/τ)`.
   - τ=5: 65% of starts in first 3 s, 86% in first 10 s, ~0.1% beyond 35 s.
   - τ=15: 49% in first 10 s, ~10% beyond 35 s.

Both changes have changelog entries (`docs/source/changelog.rst`).

## Experiment log (all runs resume from a checkpoint, 1000 iters each, 4096 envs)

| Run dir (logs/rsl_rl/g1_29dof_mode_15_tracking/) | From | Recipe | Final ckpt |
|---|---|---|---|
| `2026-09-11_16-34-55` | fresh | baseline (adaptive sampling) | `model_14000.pt` |
| `2026-09-11_19-36-18` | 14000 | + foot rewards | `model_14999.pt` |
| `2026-09-11_20-08-29` | 14000 | + foot rewards + weighted τ=5 | `model_14999.pt` |
| `2026-09-11_20-36-01` | 20-08-29/14999 | + foot rewards + weighted τ=15 | `model_15998.pt` ← **best** |

## Full-clip evaluation (1024 envs, sampling start, corruption on, no push, 2286 steps)

| metric | baseline | +foot | +foot+τ=5 | **+foot+τ=15 (final)** |
|---|---|---|---|---|
| mpkpe (global) | 0.4036 | 0.4129 | 0.4047 | **0.3039** |
| r_mpkpe (pose) | 0.0430 | 0.0438 | 0.0491 | **0.0417** |
| joint vel err | 1.023 | 1.032 | 1.119 | **1.008** |
| ee_pos (ankles+wrists) | 0.0661 | 0.0665 | 0.0721 | **0.0621** |
| ee_ori | 0.180 | 0.189 | 0.221 | **0.178** |
| ankles pos err (L/R) | 0.087/0.087 | 0.086/0.079 | 0.090/0.083 | **0.080/0.074** |
| success_rate (whole clip) | 1.000 | 1.000 | 0.914 | **0.998** |

Failure-time histogram of the τ=5 run: 87 of 88 failures in the final 40–46 s bin
(median ~41.5 s); 0 failures in the first 40 s. Final run: 2 failures, both ~41 s.

## Lessons learned

1. **Exp-reward saturation limits foot-reward effect.** `exp(-e²/std²)` with std=0.3 is
   essentially flat at the errors the feet actually have (~0.08 m → value ≈ 0.93), so weight
   0.3 moves almost nothing. The +foot-only run changed ankle errors by only ~±10% with mixed
   sign. For a stronger pull: smaller std (≈0.15) or a linear penalty instead of exp.
2. **Init-frame focus works but starves the tail.** Exponential sampling with small τ fixes the
   focused region but gives ~0.1% of starts beyond ~35 s — the clip's ending becomes
   untrained and the whole-clip success rate drops (0.914) with failures exactly in the
   uncovered tail. First-10 s metrics looked great *because the eval window coincided with the
   focus window* — evaluate the full clip before trusting focused-training gains.
3. **Two-stage focus (τ=5 then τ=15) is the winning recipe.** The sharp focus first imprints
   the hard beginning; the relaxed pass re-covers the tail. Result: best on every metric,
   success rate restored to 0.998. (Remaining 0.2% tail failures could get one more pass at
   τ≈20–25 if ever needed.)
4. **No wandb on this machine.** Train with `--agent.logger tensorboard` (wandb.init fails with
   "api_key not configured"). The repo's `evaluate.py` is wandb-bound; use
   `scripts/evaluate_tracking_policy.py` instead (wandb-free, local checkpoint + motion file,
   same metrics + per-body root-relative errors + ankle-only EE error + failure-time histogram).

## Commands

Train (resume pattern; replace `--load-run`/`--load-checkpoint`/`--init-weight-s`):

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1-29DOF-Mode-15-No-State-Estimation \
  --env.commands.motion.motion-file data/qianghuo_smplx_unitree_g1_29dof_mode_15_tracking.npz \
  --env.commands.motion.sampling-mode weighted \
  --env.commands.motion.init-weight-s 15.0 \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard \
  --agent.resume True \
  --agent.load-run <run-dir> \
  --agent.load-checkpoint model_XXXXX.pt \
  --agent.max-iterations 1000
```

Full-clip eval:

```bash
uv run python scripts/evaluate_tracking_policy.py \
  --checkpoint-file logs/rsl_rl/g1_29dof_mode_15_tracking/<run>/model_XXXXX.pt \
  --motion-file data/qianghuo_smplx_unitree_g1_29dof_mode_15_tracking.npz \
  --num-envs 1024
```

Watch: same pattern as play.py with `--checkpoint-file` + `--motion-file`.

## Numbers to remember

- Clip: 2286 frames @ 50 Hz = 45.72 s; episode cap in training = 500 steps (10 s);
  for full-clip eval set `env_cfg.episode_length_s = 45.72`.
- Baseline short-step behaviour: pelvis avg speed 0.28 m/s over clip, net travel 2.59 m,
  90 steps, median swing excursion 0.16 m (p90 0.53 m) — the reference steps are modest but
  the policy tracked them even shorter.
