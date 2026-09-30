# Diffusion policy — controller integration handoff

Audience: a fresh session that will build a **controller** (sim and later real robot)
that runs the D2 state–latent diffusion generator for the AgiBot X2.

Read this first, then read the code it cites. Every claim below is from source; where
something is unmeasured or unimplemented it says so explicitly.

---

## 0. Status: read this before planning

**What exists and is verified.** A trained denoiser and a complete, tested inference
path exist: frozen DDIM sampler, frozen noise schedule, token layout, state
projection, evaluation harness. The diffusion model is a *replacement for the latent
source* feeding an already-deployed VAE decoder.

**What is NOT ready for a controller:**

| gap | detail |
|---|---|
| **No exported artifact** | No ONNX/TorchScript for the denoiser exists yet. Export is a separate task (was scoped as D5). |
| **No real-time timing evidence** | The D0 contract requires **50 Hz / 20 ms**. The only measurement on record is the *whole* VAE+controller loop at **~31–41 ms/step unplugged** (`NOW.md`), already over budget. 20 DDIM jumps per control step is a hard latency problem, not a detail. |
| **Model quality gate NOT met** | §7.8 of the D2 plan fails on pilot-scale data; bulk training is in progress. Do **not** judge this model's tracking quality from any existing number. |
| **Guidance not implemented** | `guidance_strength != 0` raises. Any cost/obstacle-guided behaviour is D4, unavailable. |
| **Hardware state estimate unvalidated** | D0 uses simulator ground truth. The hardware state estimator is a separate gate (D6). |
| **Fall detection caveat** | Physical-fall criterion is tilt-only (>70° root tilt). A pelvis-"sinking" level collapse can pass it. |

So: build and unit-test the *interface* now; do not trust it on hardware.

---

## 1. The pipeline in one picture

```
 measured history (41 steps @ 50 Hz)                ← controller supplies
        │
        ├─ state: 135-D hybrid  → normalize → project P → 199-D
        └─ latent: 32-D         → normalize            → 32-D
        │
        ▼
 token tensor (41, 231)  +  clean mask (41, 231)    ← 199 state | 32 latent
        │
        ▼  DDIM, 20 jumps, eta=0, x0-prediction      (denoiser: 6-layer transformer)
        │
        ▼
 predicted clean tokens (41, 231)
        │
        ▼  take token[current_index=8], latent slice [199:231]
        │
        ▼  denormalize latent (undo training normalization)
        │
        ▼  frozen VAE decoder:  decode(latent, conditioning[99]) → 31 actions
        │
        ▼  HAL joint commands @ 50 Hz
```

Token width: **231 = 199 projected state + 32 latent**. Sequence: **41 = 8 past + current(8) + 32 future**.

---

## 2. Inputs, exactly

### 2.1 The window

Frozen in `docs/plans/beyondmimic_diffusion_d0_contract.yaml` (`window:` block):

| field | value |
|---|---|
| `past_steps` | 8 |
| `current_index` | **8** |
| `future_steps` | 32 |
| `total_steps` | 41 |
| padding | **forbidden** |

The window must be a **constant episode segment and controller**: no reset, no
reference teleport, and no policy switch may cross a window
(`window.permit_reset_reference_teleport_or_policy_switch_crossing: false`).

### 2.2 The state vector (135-D) — the hard part

20 bodies in a **frozen order** (`state.body_order`):

```
pelvis, left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link, torso_link,
left_shoulder_roll_link, left_elbow_link, left_wrist_yaw_link,
left_wrist_pitch_link, left_wrist_roll_link, right_shoulder_roll_link,
right_elbow_link, right_wrist_yaw_link, right_wrist_pitch_link,
right_wrist_roll_link, head_yaw_link, head_pitch_link
```

Slices (`state.slices`):

| slice | indices | content |
|---|---|---|
| `root_position` | 0:3 | pelvis position |
| `root_rot6d` | 3:9 | rotation, **first matrix column then second** |
| `root_linear_velocity` | 9:12 | |
| `root_angular_velocity` | 12:15 | |
| `body_positions` | 15:75 | 20 × 3 |
| `body_linear_velocities` | 75:135 | 20 × 3 |

Frame conventions (get these wrong and nothing else matters):

- `raw_quaternion_convention: wxyz`
- root frame = **current-window-center yaw**; root **position and linear velocity** are
  *subtracted-then-rotated*; root **angular velocity** is *rotated without subtraction*
- body frame = **each timestep's root yaw**; body position/linear velocity are
  *subtract same-timestep root, then rotate*

**Do not reimplement this by hand.** `WorldState` → `world_to_hybrid(states,
center_index=...)` in `src/mjlab/tasks/tracking/diffusion/state.py:250` does exactly
this, and `hybrid_to_world` (`:292`) inverts it. `WorldFrameContext` (`:213`) retains
the world context needed for the inverse.

### 2.3 Normalization and the 199-D emphasis projection

`preprocessing` block of the contract. Order is **normalize → project**;
`normalize_after_projection: false`.

- statistics are **train-only**, accumulated in float64, ddof 0, `std_floor 1e-6`
- `P = vstack(A @ B, I)`; `y = P @ normalized_state`
- 64 Gaussian rows, root diagonal emphasis **6.0**, others 1.0, `rcond 1e-12`
- `projected_dimension 199`, `token_dimension 231`

The matrices and statistics live in the D1 dataset as `offline/projection.npz`, loaded
into a `ProjectionBundle` (`src/mjlab/tasks/tracking/diffusion/projection.py:116`):

```python
bundle.project_state(state_135)      # normalize + P        -> 199
bundle.inverse_state(projected_199)  # pinv + denormalize    -> 135
bundle.normalize_latent(latent_32)
bundle.denormalize_latent(latent_32)
```

`FeatureStats.normalize/denormalize` at `projection.py:47` / `:53`.

**A controller must reuse the identical `projection.npz`.** It is identity-hashed
(`matrix_sha256`, `statistics_sha256`, `pseudoinverse_sha256`) and enforced on load —
a mismatched or re-fitted bundle is a hard error, by design.

### 2.4 The conditioning (what is "known" vs "generated")

Built by `clean_mask(*, current_index, state_dimension, token_dimension, device)` at
`src/mjlab/tasks/tracking/diffusion/noising.py:166`:

| tokens | state (0:199) | latent (199:231) |
|---|---|---|
| 0–7 (past) | **fixed** | **fixed** |
| 8 (current) | **fixed** | **generated** |
| 9–40 (future) | **generated** | **generated** |

i.e. the past is conditioning, the current state is conditioning (feedback), and the
**current latent is the thing being generated**.

So the sampler needs two tensors:
- `clean_tokens` — shape `[41, 231]` (or `[B, 41, 231]`); fixed entries must be real values
- `clean_mask` — same shape, bool; unbatched `[41, 231]` is accepted

`apply_clean_mask` (`noising.py:187`) restores fixed entries bit-identically, and the
sampler reapplies it **after every jump**.

### 2.5 The step-id tensor

The denoiser takes `(noisy_tokens, step_ids)` where `step_ids` is `[B, 41, 2]`
`torch.long` — channel 0 = state noise level, channel 1 = latent noise level. The
sampler sets both to the source level, then **overrides fixed entries to 0**
(`sampler.py:180–190`): past state+latent rows fully, and the current row's *state*
channel only.

State and latent levels are drawn **independently** per token during training
(`state_and_latent_levels: independent_uniform_integers_0_through_K_per_token`), so the
two channels genuinely differ; keep them separate.

### 2.6 Startup constraint (easy to miss)

`startup` block:

- mode: `actual_vae_history_then_reference_free_handover`
- `minimum_actual_pairs: 8` — the 8 past steps must come from **real executed pairs**
  (actual measured state + the latent actually sent to the decoder)
- **`history_from_standing_policy: forbidden`**
- `reference_or_encoder_after_diffusion_handover: forbidden_except_requested_keyframes`

So the controller must buffer the last 8 *(state, latent actually decoded)* pairs and
may only start diffusing once it has them. Cold-starting the window from the standing
policy's history is explicitly not allowed.

---

## 3. The sampler, exactly

`sample_trajectory(...)` — `src/mjlab/tasks/tracking/diffusion/sampler.py:145`.

**Frozen configuration** (`noise:` block):

- `training_K: 1000`, `clean_index: 0`, `embedding_count: 1001`
- `sampler: ddim`, `eta: 0.0`, `updates: 20`
- endpoints: `[1000, 950, 900, ..., 100, 50, 0]` → **20 jumps, 21 endpoints**
- `denoiser_at_clean_zero: false` — the final destination 0 is reached by assignment,
  **not** by a denoiser call
- `reapply_clean_values_after_every_jump: true`

`SamplerConfig` **rejects** `eta != 0` and `guidance_strength != 0`, and validates the
grid against `build_inference_grid(schedule)` (`sampler.py:29–40`, `:158–163`).

### Algorithm (reimplementable from this)

```
a = alpha_bars                      # length 1001, a[0] == 1.0
srcs = [1000, 950, ..., 50]
dsts = [ 950, 900, ...,  0]

noise  = randn(41, 231) * initial_noise_scale
sample = apply_clean_mask(noise, clean_tokens, clean_mask)

for s, d in zip(srcs, dsts):
    step_ids        = full((41, 2), s)
    step_ids[fixed_state_rows, 0] = 0
    step_ids[fixed_latent_rows, 1] = 0
    x0_hat = denoiser(sample, step_ids)
    if d == 0:
        jumped = x0_hat
    else:
        eps_hat = (sample - sqrt(a[s]) * x0_hat) / sqrt(1 - a[s])
        jumped  = sqrt(a[d]) * x0_hat + sqrt(1 - a[d]) * eps_hat
    sample = apply_clean_mask(jumped, clean_tokens, clean_mask)

return sample                      # the 41x231 clean trajectory estimate
```

`x0_hat` is the network's **direct clean-token prediction** — the objective is
x0-prediction, not epsilon regression (`model.prediction: x0`). Epsilon is only ever
*derived* inside the jump equation above. Do not "simplify" this into an
epsilon-predicting network.

**Noise schedule** (`noise` block + `schedule.py`). Recompute, do not retype:

```
x   = linspace(0, 1000, 1001)
raw = cos(((x / 1000) + 0.008) / 1.008 * pi / 2) ** 2
raw = raw / raw[0]
beta_min, beta_max = 1e-5, 0.999
betas      = clip(1 - raw[1:] / raw[:-1], beta_min, beta_max)
alpha_bars = concat([1.0], cumprod(1 - betas))      # float64, len 1001
```

Pinned values to check against: `alpha_bars[0] == 1.0` exactly,
`[1] = 0.999958715775178`, `[50] = 0.9920072786842186`,
`[500] = 0.49384359044063775`, `[1000] = 2.4287669070348542e-09`.

**Diagnostics** (`SamplerDiagnostics`, `sampler.py:42`) report `denoiser_calls`,
`jumps`, `source_ids_returned`, `step_ids_seen`, `final_index`, `initial_noise_norm`.
Assert `denoiser_calls == 20` and `final_index == 0` in your integration tests — these
are cheap and catch a mis-wired loop immediately.

**Determinism:** the initial noise comes from a caller-supplied
`torch.Generator`. For reproducible replays, own that generator's seed.

**Guidance (D4, not available):** the frozen contract defines input-VJP guidance with
`x0_guided = x0_hat - lambda * (1 - a[k]) * clipped_grad_G`, per-trajectory L2 clip 1.0,
zeroed fixed-entry gradients. `guidance_strength != 0` raises today. If you find
yourself needing it, that is D4 work, not a controller tweak.

---

## 4. Output and how to turn it into actions

`sample_trajectory` returns the full **(41, 231)** predicted clean token trajectory.
Only one slice of it drives the robot:

```
latent_current = sample[current_index=8, 199:231]     # 32-D, still NORMALIZED
latent_phys    = bundle.denormalize_latent(latent_current)
actions_31     = vae.decode(latent_phys, conditioning_99)
```

`decode(latent, conditioning)` — `src/mjlab/tasks/tracking/distillation/model.py:331`.
**Undo latent normalization before decoding** (`undo_latent_normalization_before_decode:
true`). The decoder is **gravity-conditioned** (`decoder_mode: gravity`), and the
conditioning is **99-D with a fixed field order** (`vae_config.py:22`, `:110–145`):

| field | dim |
|---|---|
| `projected_gravity` | 3 |
| `gyro` | 3 |
| `relative_joint_q` | 31 |
| `joint_dq` | 31 |
| `previous_action` | 31 |

The decoder is **not reference-conditioned** in gravity mode — the reference influences
it *only* through the latent. That is exactly why generating the latent is sufficient to
steer behaviour.

The predicted **state** slice (0:199) is also returned. A controller should not use it
for feedback (use measured state), but it is what a future guidance cost would act on.

---

## 5. Reuse the decoder; you are replacing the latent source

The VAE decoder is already frozen, real-robot-validated and deployed:

- checkpoint `logs/distillation/mixed-10k/checkpoint-final.pt`,
  sha256 `69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117` (version-3 cohort)
- manifest `configs/distillation/x2_tennis_mixed.yaml`,
  sha256 `587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d`
- cohort ids `tennis_000`, `tennis_001`, `tennis_002_ss`
- dims `encoder_command 68 / conditioning 99 / latent 32 / action 31`

In the existing X2 controller a 32-D latent is decoded at 50 Hz to produce the 31
joint actions. **Verify the exact seam in `x2_docker` before assuming it** — the claim
here is functional (something supplies a 32-D latent to the frozen decoder), not a
citation of that repository's internals. **The diffusion model does not change the
decoder interface — it changes where the latent comes from.** Solve the problem at
that seam: replace "latent from reference" with "latent from diffusion", keep the
decoder, the action post-processing, joint
order, gains, offsets and the observation contract untouched
(`preserve_embedded_joint_order_action_scale_offset_gains_and_observation_contract:
true`).

**Deploy the EMA weights.** The offline eval path loads the checkpoint and calls
`ema.copy_to(model)` *before* generating, i.e. reported metrics are EMA-parameterized.
EMA lags the raw model early in training; that is expected. Export from the EMA.

---

## 6. Endpoint: hand off to standing

`recovery` block of the contract:

- ONNX `logs/rsl_rl/agibot_x2_velocity/2026-09-26_02-40-45_x2-tennis-recovery-n25/2026-09-26_02-40-45_x2-tennis-recovery-n25.onnx`
  sha256 `caf17c38f23ad180829a230a9a3259f14eedd3860dd90a046b8be92d3d047681`
- observation **102** (order `base_ang_vel(3) + projected_gravity(3) + joint_pos(31) + joint_vel(31) + actions(31) + command(3)`), action **31**, `observation_history: false`
- command `[0, 0, 0]` (zero-command standing)
- trigger `deployed_last_reference_frame_boundary`; `switch_resets_physics: false`
- preserve the previous-action handoff semantics

The endpoint switch happens **at the last reference frame**. PPO recovery rows are
**not** diffusion training data, and no training window crosses the switch.

---

## 7. Reference implementation — copy this, don't reinvent

The canonical end-to-end offline example is
`evaluate_generation(...)` in `src/mjlab/tasks/tracking/diffusion/evaluation.py`, driven
by `_command_evaluate_offline` in `src/mjlab/scripts/diffusion.py`. Read them in this
order; they show every step of §2–§4 including the projection, mask construction, the
sampler call, latent extraction and the physical-space reduction.

Worked artifacts to inspect:

```
logs/diffusion/d2-parent-gate-20260930/          # earlier gate runs + eval JSONs
logs/diffusion/d2-bulk-gate-20260930/primary/    # current bulk run
    checkpoint-best.pt / checkpoint-last.pt      # EMA + model + optimizer + identity
    resolved-config.json, metrics.jsonl
    tokens-{train,validation,test}-all.npy       # v2 token caches (2.0/0.26/0.28 GB)
logs/diffusion/d1-bulk-par/offline/projection.npz  # the normalization + P you must reuse
```

### Code index

| need | file | symbol |
|---|---|---|
| frozen constants | `docs/plans/beyondmimic_diffusion_d0_contract.yaml` | — |
| 135-D state, frames | `tasks/tracking/diffusion/state.py` | `WorldState:134`, `world_to_hybrid:250`, `hybrid_to_world:292`, `WorldFrameContext:213` |
| normalize/project/P | `tasks/tracking/diffusion/projection.py` | `FeatureStats:32`, `ProjectionBundle:116`, `project_state:199`, `inverse_state:203`, `normalize_latent:210`, `denormalize_latent:213` |
| mask + noising | `tasks/tracking/diffusion/noising.py` | `clean_mask:166`, `apply_clean_mask:187`, `x0_target:159`, `add_independent_noise:101` |
| schedule | `tasks/tracking/diffusion/schedule.py` | `DiffusionSchedule`, `build_inference_grid` |
| denoiser net | `tasks/tracking/diffusion/model.py` | `DenoiserSettings:15`, `StateLatentTransformer:71`, `forward:111` |
| sampler | `tasks/tracking/diffusion/sampler.py` | `SamplerConditions:20`, `SamplerConfig:29`, `SamplerDiagnostics:42`, `sample_trajectory:145` |
| generation eval | `tasks/tracking/diffusion/evaluation.py` | `evaluate_generation` |
| checkpoint IO | `tasks/tracking/diffusion/checkpoint.py` | `load_checkpoint`, `save_checkpoint`, `ExponentialMovingAverage` |
| trainer config | `tasks/tracking/diffusion/training_config.py` | `TrainingConfig` |
| CLI | `scripts/diffusion.py` | `train`, `evaluate-offline`, `audit`, `preflight` |
| VAE decode | `tasks/tracking/distillation/model.py` | `decode:331` |
| VAE layout/conditioning | `tasks/tracking/distillation/vae_config.py` | `CONDITIONING_DIMS:22`, field order `:110–145` |
| VAE export | `tasks/tracking/distillation/export.py` | version-3 bundle export |
| live sim observation adapter | `tasks/tracking/distillation/adapter.py` | `reset/step/snapshot` |
| tests to imitate | `tests/test_tracking_diffusion_{sampler,schedule,noising,model,eval,train_cli}.py` | — |

### Denoiser dimensions for export

`DenoiserSettings`: `token_dimension 231`, `sequence_length 41`,
`embedding_count 1001`, `width 512`, `layers 6`, `attention_heads 8`,
`ffn_width 2048`, `dropout 0.0`, `activation gelu`, `norm_first=True`.

Parameter count is exactly **20,197,607**. Inputs: `noisy_tokens float32 [B,41,231]`
and `step_ids int64 [B,41,2]`. Output `[B,41,231]`.

---

## 8. Timing: the thing most likely to sink this

- Required: **50 Hz → 20 ms per control step**; physics 0.005 s, decimation 4.
- Each control step needs **20 denoiser forward passes** (one per DDIM jump) →
  **1,000 forward passes per second**.
- Rough compute per pass at B=1: ~1.7 GFLOP → ~33 GFLOP per control step, ~1.7 TFLOP/s
  at 50 Hz. *Arithmetic estimate only* — the problem is **launch/latency-bound at B=1**,
  not FLOP-bound.
- Previously measured: the existing VAE+controller loop ran **~31–41 ms/step** with
  **no** diffusion in it, i.e. already over budget.

The contract explicitly **forbids** silently reducing the rate, the horizon, the
backbone, or the number of steps (`runtime_risk.forbid_silent_rate_horizon_backbone_or_step_reduction`).
If 20 jumps cannot fit, that is a *pre-registered, measured* engineering decision with a
documented trade-off — not a quiet optimization. Options worth measuring: batching the
20 jumps across the sequence, bf16/fp16 with a parity check
(`numerical_acceptance.bf16_fp16_deployment_parity` is a *separate* gate, not implied),
ONNX Runtime / TensorRT, and CUDA-graph capture.

Simulation may run slower than wall clock initially
(`initial_simulation_may_run_slower_than_wall_clock: true`); do not present that as
satisfying 50 Hz.

---

## 9. Suggested first steps for the new session

1. Read §2–§4 and `evaluate_generation`; then write a **standalone replay test** that
   loads a checkpoint + `projection.npz` and reproduces the offline latent for one
   window, asserting `denoiser_calls == 20`, `final_index == 0`, and a fixed-seed replay
   matches bit-for-bit.
2. Build the **state → token** path against `world_to_hybrid`, and assert a
   `project_state → inverse_state` roundtrip tolerance
   (`numerical_acceptance.fp64_projection_roundtrip_max_abs: 1e-10`).
3. Only then wire it to the VAE decoder seam (§5) and compare, for a fixed window, the
   decoder output against the offline harness.
4. Measure per-step latency **before** claiming anything about control rate.
5. Keep sim and hardware claims separate; hardware execution needs its own authorization.

Do not re-derive the contract constants, do not re-fit normalization, and do not
replace the decoder.
