# BeyondMimic state–latent diffusion: DDPM training, DDIM inference at 50 Hz

Status: **D0 specification/audit complete, with explicit runtime-risk acceptance
(2026-09-29).** The user approved the recommended defaults, endpoint qualification
and progression toward a small prototype before 50 Hz replanning is demonstrated.
The authoritative design is the [frozen D0 contract](beyondmimic_diffusion_d0_contract.yaml);
see [closeout, evidence and deferred gates](beyondmimic_diffusion_d0_validation.md#11-d0-closeout--2026-09-29).
This is not implementation or runtime acceptance. **D1 has not started**;
collection, training, remote jobs and hardware execution still require separate
authorization. Preserve DDPM-style training, 20-update DDIM inference and the
**50 Hz X2 control contract**; flow matching is deferred. No simulator or GPU job
was launched for this closeout.

Related: [VAE architecture](beyondmimic_vae_distillation.md),
[VAE milestones](beyondmimic_vae_implementation.md),
[earlier reproduction-repo review](beyondmimic_reproduction_review.md), and
[tracking versus planning](beyondmimic_general_tracker_and_diffusion.md).

## 1. Goal, boundaries, and evidence

Build a state–latent trajectory generator above the **frozen mjlab VAE decoder**:

```text
actual state history + executed latent history + current actual state
                              |
             DDIM state–latent trajectory sampler
                  ^           |
       keyframes / costs      +--> predicted physical states (planning/diagnostics)
                              |
                       time-aligned latent
                              |
latest deployable proprioception --> frozen VAE decoder --> 50 Hz action
```

First reproduce the mechanism on qualified X2 motions in simulation: valid
sequence collection, conditioned generation, closed-loop continuation, sparse
keyframe control, then task-cost guidance. This is **not** an immediate claim to
reproduce the paper's G1 cartwheels, locomotion repertoire, hardware success rates,
or navigation demonstrations. Tennis-only data cannot establish those results.
The user confirms the available VAE has **three teachers of approximately ten
seconds each**: use this for a bounded engineering prototype, not broad tennis
generation. The user froze `logs/distillation/mixed-10k/checkpoint-final.pt` as
this prototype's VAE baseline and confirmed that it has been validated on the
real robot (2026-09-29). Its SHA256 is
`69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117`;
retain its weights, normalizers and three-teacher contract unchanged. Actual clip
durations are **9.06 / 6.80 / 6.40 s** (22.26 s total). This user-reported VAE
hardware validation does not validate the future diffusion layer or authorize
new collection/training/hardware runs. Perturbed repeats expand local recovery
coverage, not the repertoire;
substantial behavioral expansion requires qualifying an expanded VAE before
large-scale collection. An eight-teacher manifest is not such qualification.

Evidence consulted:

- `/home/agiuser/Documents/beyondmimic.pdf`: main pp. 19–21, Fig. 7,
  supplement S3/S4 and Table S7; limitations on p. 12.
- Independent reference repo: `/home/agiuser/projects/BeyondMimic-Reproduction`,
  revision `55b37260ce02573e70648111c534b735972a2af4`. It is not the official
  implementation. Its README explicitly excludes official checkpoints and
  benchmark claims; inspected `outputs/` contains placeholders, not result logs.
- mjlab base revision: `46f90a244f08b8b47fa9a8d0f3a1655f10a09f53`, with existing
  uncommitted VAE/cohort work. Preserve that work; this plan changes no runtime.
- Local read-only device query confirms **RTX 5080 Laptop GPU**, 16,303 MiB,
  driver 595.91.07. This establishes device identity, **not measured latency**.

VAE M1–M4 are accepted; the broader M5 record is in progress. The selected
`mixed-10k` checkpoint is now user-confirmed as real-robot validated; this audit
did not reproduce those trials. CPU design/model work can proceed without
hardware. Substantial collection/training still requires authorization and must
retain the frozen candidate/schema/normalizers. Hardware readiness of the new
diffusion layer remains separate. A later changed VAE means a new latent/data
contract and normally new rollouts, not silent checkpoint reuse.

## 2. Fixed choices and explicit adaptations

| Item | Paper | Frozen X2 baseline v1 |
| --- | --- | --- |
| Control period | 40 ms on diffusion path | **20 ms**, unchanged physics/PD/action semantics |
| History | 4 past steps, approximately 0.16 s | **8 past steps**, 0.16 s |
| Future | 16 future steps, 0.64 s | **32 future steps**, 0.64 s |
| Window indexing | history + current + future | `[-8, ..., 0, ..., 32]`, **41 timesteps**, current index 8 |
| Latent | 32 | Existing frozen 32-D latent |
| Decoder | Current proprioception + latent | Existing gravity schema: 99 conditioning + 32 latent -> 31 actions |
| Denoiser | 6 layers, width 512, 8 heads, approximately 19.8M | Same reported backbone settings; count actual parameters |
| Prediction | Clean state–latent trajectory | `x0` prediction, independent state/latent noise levels |
| Sampling | DDPM formulation; 20 inference steps | **DDIM inference**, 20 reverse updates, `eta=0`; DDPM-style training unchanged |
| Training noise grid | Not fully specified | **`K_train=1000`**, separate from `num_inference_steps=20`; exact bounded schedule in the frozen contract |
| Symmetry | Sagittal augmentation | **Off**: X2 symmetry is already aborted for this project |
| Deployment | Async GPU generator + synchronous CPU decoder | Same separation; first target local RTX 5080 Laptop, not unmeasured Orin performance |

Do not downsample the accepted VAE to 25 Hz. Do not keep 16 future steps while
claiming a 0.64 s horizon at 50 Hz. The 41-timestep network has almost unchanged
parameter count but more compute than a 21-timestep network; GPU model names
alone cannot establish that 20 guided updates fit a 20 ms budget.

The paper does not fully specify the training noise discretization, reverse
variance, Transformer FFN/activation/token packing, projection width, input
normalization, guidance discretization, or startup initialization. Choices below
are labeled reproduction decisions, not newly discovered paper facts.

## 3. Reference-repo audit: useful pieces and things not to copy blindly

Paths in this section are relative to `BeyondMimic-Reproduction/`.

| Source | Useful reference | Qualification / required correction |
| --- | --- | --- |
| `configs/stage3/diffusion_full_50hz_h8_f32.yaml` | Already specifies 50 Hz, history 8, future 32, 41 timesteps | This config disables normalization; its existence is not a successful experiment |
| `stage3/models/state_latent_transformer.py` under `src/beyondmimic_repro/` | One `[state, latent]` token per timestep, separate state/latent step embeddings, bidirectional Transformer | FFN=2048, GELU, pre-norm, learned positions are reference choices, not all paper-specified; input IDs are silently clamped there—ours should reject invalid IDs |
| `stage3/diffusion/noising.py` | Independent state/latent noising and clean-target loss | Preserve an exact clean index for conditioning; test this with the real schedule, not only a hand-written fixture |
| `stage3/diffusion/schedule.py` | Cosine schedule | Returned index 0 is already noisy (`alpha_bar ~= 0.992007` for 20 steps), whereas inference clamps clean observations with step ID 0 |
| `stage3/diffusion/sampler.py` | Multistep conditioned sampling and guidance prototype | `_guided_ddim_sample` uses a deterministic DDIM-style update, matching our selected sampler family, **not ancestral DDPM**. Its fewer-step option uses a schedule prefix, not a full-range respaced grid; do not copy that behavior. The no-schedule fallback is just a single denoiser call |
| `stage3/inference/diffusion_vae_policy.py` | Decoder/front-end interface sketch | Calls the single-call fallback; do not treat this class as the complete 20-step controller |
| `adapters/isaac/live_rollout.py` | Actual full sampler path, collection timing, history rebasing, physical velocity offsets | Initial missing history is zero-padded and clamped as observed; reset handling also zeros history. This is their implementation choice, not a paper-documented startup method |
| `stage3/datasets/state_latent_builder.py` | VAE rollout provenance, hybrid frames, projected states | Emits complete windows; its main window loop checks accepted/done flags but does not independently prove constant segment IDs and consecutive timestamps. Do not inherit permissive defaults for missing acceptance metadata |
| `stage3/diffusion/training_runtime.py` | AdamW, accumulation, cosine LR, EMA, separate state/latent metrics | Randomly splits overlapping windows; normalization can be fitted before splitting. Use rollout-group splits and train-only statistics instead |
| `state.py`, `stage3/representation/` | Hybrid-state equations and persisted projection/pseudoinverse | Hard-coded defaults are G1: 14 bodies, 99 raw state dimensions, 163 projected. The main projection helper adds random **root-only** features; the paper describes a diagonal emphasis matrix over state features. Choose and version the interpretation explicitly |
| `stage3/guidance/physical_velocity.py` | Unprojection and adding current root velocity back for physical costs | Valuable frame/units reference. Its diagnostic sampler also runs an unguided comparison; do not include that extra pass in a production loop unnoticed |
| `stage3/guidance/obstacle.py` | Relaxed barrier formula | Uses planar root-to-circle distance, not the paper's per-body SDF sum; not equivalent whole-body avoidance |
| `adapters/isaac/live_rollout.py` OU setup | Action perturbation / stability filtering | Uses `dt=1/frequency_hz`; paper specifies OU `dt=1.0`. These produce different disturbances |

A CPU-only inspection instantiated the reference model: with a 195-D G1 token,
its parameter counts are 19,145,923 at length 21 and 19,156,163 at length 41—not
exactly 19.8M. This is structural evidence only, not a speed or policy test.

Use this repo as readable reference and a source of adversarial test cases, not
as a new runtime dependency or a drop-in implementation. Preserve its MIT notice
if substantial source is adapted. Its legacy README smoke pipeline is not the
stage-3 production path.

## 4. State, data, and frozen-interface contracts

### 4.1 Frozen VAE and cohort

Pin checkpoint bytes/hash, encoder/decoder and normalization state, gravity
schema, joint/action/body order, compiled asset, control period, gains, action
scale/offset, measurement semantics, and source cohort/motion hashes.

Use `z = mu` for initial collection and execution, matching deterministic VAE
export; sampled-latent collection is a separately identified dataset. Record the
**actual latent passed to decode**, not a later re-encoding. Reject
reference-conditioned decoder variants for the initial reference-free planner.

Do not assume the oldest two-/three-teacher docs identify today's selected model.
The checkout also contains `configs/distillation/x2_tennis_mixed_v2.yaml` with
eight teachers. D0 must bind the user's chosen VAE checkpoint to its actual
cohort; an eight-teacher manifest alone does not prove an accepted eight-teacher
student. D0 binds `mixed-10k` to its actual three-member cohort and source hashes;
an eight-teacher manifest does not replace that binding. No new teacher training
or cohort expansion is authorized here.

### 4.2 Physical state and frames

Store raw root pose/twist and named body positions/velocities in world coordinates,
then construct windows relative to each window's current character-yaw frame.
Use the same vectorized transformation offline and online; recompute history in
the **new current frame** at each plan. Do not cache already-rebased state tokens
and concatenate tokens expressed in different frames.

Follow S1–S3 explicitly:

- Root position: `R_yaw(t)^T * (p(t+n) - p(t))`.
- Root orientation: `R_yaw(t)^T * R(t+n)`, continuous Rot6D with declared packing.
- Root linear velocity: `R_yaw(t)^T * (v(t+n) - v(t))`, including the subtraction
  printed in S2; current relative linear velocity is therefore zero.
- Root angular velocity: `R_yaw(t)^T * omega(t+n)`.
- Body position/velocity: subtract the root position/linear velocity at that
  timestep and rotate by that timestep's root yaw, per S3. These velocity features
  are not to be silently replaced by derivatives of rotating-frame positions.

The selected root is the physical pelvis/root, **not** the torso tracking anchor
or an assumed IMU frame. The frozen contract records the compiled-asset identity
and the cohort's exact ordered **20 bodies**, so `D_state = 15 + 6*20 = 135`. The
list, not this number, is authoritative. Store raw quaternions as `wxyz`; Rot6D
concatenates the first matrix column then the second. Keep world root pose and
current velocity as planning context for reconstructing physical costs, not
hidden reference information.

Simulation ground truth is permitted for initial planning experiments and must
be labeled as such. Hardware needs a separately validated estimator for these
states; our no-linear-velocity VAE decoder does **not** remove that requirement.
Record estimator timestamps and frames when an estimated-state backend is added.

### 4.3 Normalization and emphasis

Frozen v1 preprocessing (engineering choices where the paper leaves details open):

1. Fit unprojected state and latent statistics on **training groups only**.
2. Standardize raw hybrid states; separately standardize latents for diffusion.
3. Apply persisted emphasis projection to standardized states.
4. Concatenate projected state and normalized latent.
5. Inverse-project and undo normalization before physical cost evaluation; undo
   latent normalization before passing a latent to the unchanged VAE decoder.

Use `P = vstack(A @ B, I)`, with `A` generated by NumPy PCG64 seed 42,
standard-normal float64 entries, and `B` a full state-dimensional diagonal matrix
(root entries 6, remaining entries 1). Use 64 added rows as a
**reference-inspired engineering setting**, not a paper-reported dimension.
Fit population statistics over complete training-window tokens in float64, with
standard deviation floored at `1e-6`. For 20 bodies this gives state width 199 and
combined token width **231**. Persist `P`, its pseudoinverse (`rcond=1e-12`), all
statistics, feature slices and hashes. Do not standardize projected channels
afterward and cancel the intended emphasis without a named ablation.

Tests must prove inverse roundtrip, root emphasis, frame invariance, and
physical cost recovery. For predicted points outside the projection's column
space, measure the projection residual. Partial body constraints must act in
recovered physical space, not arbitrary slices of a mixed projected vector.

### 4.4 Collector and dataset lifecycle

Use a dedicated frozen-VAE collector—not the shuffled DAgger replay buffer.
At each 20 ms tick, atomically record:

```text
run/shard/env/episode/segment IDs; integer tick and physical timestamp;
motion/frame and initialization provenance (metadata, not model inputs);
raw physical root/body state; optional measured state and estimator metadata;
raw decoder proprioception and previous executed action;
actual latent z; clean decoder action; OU noise; action sent to environment;
termination/truncation/reference-boundary flags; post-step terminal evidence;
VAE/cohort/asset/control hashes; acceptance reason and verification duration.
```

The row means `(s_t, z_t, a_t)` **before** executing `a_t`; the next continuous
row gives `s_(t+1)`. Capture terminal evidence before automatic reset replaces it
where available; otherwise mark unavailable rather than treating post-reset
state as a successful next state. Existing segment/generation events must split
reference teleports even if the environment does not terminate.

Initial perturbation recipe: add paper OU noise to **normalized action units**,
`theta=0.8, mu=0, sigma=0.1, dt_OU=1`, resetting OU state per episode. Do not
substitute physical `dt=0.02`. At 50 Hz this preserves the paper's per-step
recurrence but changes its correlation duration in seconds; log this adaptation
and measure variance/autocorrelation. A wall-time-matched OU variant is a named
future comparison, not a silent change. No latent perturbation is substituted.

Initial collection interpretation away from clip boundaries: retain a 2.5 s
(125-step) data interval, perturbed throughout it, then continue the same VAE
unperturbed to 5 s (250 steps) for stability verification. Preserve the verification
trace for audit, but do not silently count it as perturbation data. This is an
explicit interpretation of the paper's terse collection description. Reject the
retained interval on physical failure; distinguish physical failure,
reference-error termination, timeout, clip end, and administrative interruption.
The endpoint case below is a separately labeled adaptation, not evidence of
five-second VAE-only recovery.

**Known endpoint behavior (user-confirmed 2026-09-29):** on reaching the final
tennis reference frame, the real-robot controller switches to a separate PPO
velocity policy with zero twist command for standing. That policy was trained
with tennis-ending initial poses. No new endpoint network or indefinite VAE
final-pose hold is needed. Pin the user-designated export:

```text
logs/rsl_rl/agibot_x2_velocity/2026-09-26_02-40-45_x2-tennis-recovery-n25/
  2026-09-26_02-40-45_x2-tennis-recovery-n25.onnx
sha256: caf17c38f23ad180829a230a9a3259f14eedd3860dd90a046b8be92d3d047681
```

Read-only inspection found one ONNX in that directory, `obs [1,102] -> actions
[1,31]`, twist command metadata and no observation history. Saved environment
metadata has `dt=0.005`, decimation 4, and `TennisRecoveryResetEvent`. The user
identifies this as the latest-checkpoint export; the ONNX metadata does not bind
an exact checkpoint iteration. These are static checks, not a new rollout.

**Settled endpoint qualification role (user, 2026-09-29):** use the existing
standing policy for post-clip recovery qualification, not as state–latent
training data. A later recovery trace can qualify an earlier VAE segment without
entering its training windows. The chosen collector contract is:

- Switch controllers without resetting/teleporting the physical state. End VAE
  action perturbations at the switch; apply zero command to the recovery policy.
  Reproduce the deployed handoff's previous-action handling and policy-specific
  observation frames, action scaling/offset and gains rather than assuming that
  equal action widths imply identical contracts.
- Mark the exact control-ownership boundary. Only VAE-owned rows with an actual
  executed latent enter the state–latent dataset; PPO recovery has no such latent.
  Keep recovery states/actions in a separate verification trace, not fabricated
  zero/last/retrospectively encoded latents. No training window crosses the switch.
- Report `vae_only` and `vae_then_standing` verification separately. A five-second
  continuous hybrid episode can assess the deployed controller sequence, not
  five seconds of recovery by the VAE alone. This separately labeled hybrid
  qualification is the settled endpoint approach, not a claim to reproduce the
  original same-VAE filter. Numeric thresholds and outcome rules are now frozen
  in the D0 contract and report section 11; collector/handoff tests remain D1
  requirements before accepting data.
- If the clip ends before the 2.5 s collection interval is complete, record the
  actual shorter VAE duration/phase coverage. Retain only complete 41-step VAE
  windows; do not pad across the switch or claim full late-phase coverage. A reset
  or reference wrap never counts as survival.

Collection progression: tiny deterministic clean sample -> small OU pilot ->
approximately 100 **accepted** occurrences per feasible motion-phase sample, with
attempted/accepted/rejected coverage reported separately. Store failed-attempt
summaries for bias analysis, but do not train the accepted-sequence prior on them.
No symmetry augmentation. No teacher action mixing during this collection.

Use append-only bounded shards, globally unique identities, and an indexed/lazy
window dataset rather than materializing every overlapping 41-step window three
times as states/latents/tokens. Require constant episode/segment/motion identity,
consecutive ticks and consistent FPS throughout every window; exclude windows
without complete history/future. Split by whole collection episode (with shard,
run and env identity) **before** windowing/statistics; initially 80/10/10
train/validation/test, stratified by motion where data permits. Same-clip held-out
rollouts are robustness tests, not unseen-motion generalization.

Data estimates must use the selected cohort, not the paper's 2.5 hours of motion:
for total reference duration `L` seconds and 100-fold coverage, nominal retained
rows are about `L * 50 * 100`, before feasibility/endpoint effects. Report actual
rows, windows, unique motions, simulated seconds, shard bytes, acceptance rate,
and projected optimizer updates. Storage/RAM budgets are approved after the pilot.

## 5. Denoising, conditioning, guidance, and runtime

### 5.1 DDPM-style training with DDIM inference

Use the reference's readable Transformer layout as the first engineering choice:
one combined token per timestep, separate state/latent diffusion-step embeddings,
learned temporal positions, six width-512 pre-norm blocks, eight heads,
FFN 2048/GELU, dropout 0. No causal attention mask: generated future tokens may
interact, but **ground-truth** future values must not leak through conditioning.

**Training and sampling are separate contracts.** DDPM and DDIM can use the same
trained denoiser and the same forward noising marginals. Switching samplers does
not require changing the training noise schedule or retraining compatible weights.
Retain the paper's x0 objective rather than silently changing its noise-level
weighting by replacing it with unweighted epsilon regression.

Represent clean data by `k=0`, with `alpha_bar[0]=1`; noisy indices are
`1..K_train`. Use frozen **K_train=1000** with a bounded cosine-beta schedule and a
near-Gaussian terminal marginal; the exact grid is pinned in the D0 contract. This is an explicit
engineering choice, **not** a paper-reported number. A training batch samples
noise levels directly; it does not run a 1,000-step forward or reverse chain.
The previous draft's native 20-level stochastic DDPM choice is superseded.
Persist the full training schedule independently of the inference grid. The D0
CPU probe validated one candidate bounded cosine construction (`beta_min=1e-5`,
`beta_max=0.999`), not trained-model quality. Its 20-update grid has **21 endpoints**
`[1000, 950, ..., 50, 0]`, with 20 nonclean source IDs and no denoiser call at 0;
a list of only 20 endpoints describes 19 jumps.

Train x0 reconstruction with independently sampled state/latent levels in
`0..K_train`, matching the paper's factorized-noise idea. At inference, past
states and actually executed past latents are fixed at clean level 0; current
state is fixed, current latent/future are unknown. Unknown entries start from
Gaussian noise. Use **20 DDIM updates with eta=0**, on a strictly decreasing
subsequence spanning the full trained noise range and ending at clean index 0.
Pass the original training timestep IDs to the model, not renumbered sampler
loop counters. Proposed initial spacing is uniform in training index; save the
exact integer grid, including rounding and endpoints, in the inference config.

For an unconditioned entry, a jump from noise index `k` to `j < k` uses:

```text
eps_hat = (x_k - sqrt(alpha_bar[k]) * x0_hat) / sqrt(1 - alpha_bar[k])
x_j = sqrt(alpha_bar[j]) * x0_hat + sqrt(1 - alpha_bar[j]) * eps_hat
```

This is deterministic conditional on the initial noise and observations; it
still produces diverse samples across initial noise seeds. The final `j=0`
update returns `x0_hat`. Keep schedule arithmetic in adequate precision and
validate the near-zero/near-one endpoints. Reapply fixed clean constraints after
every update. No implicit action/latent clipping or schedule-prefix truncation.
Supporting positive eta or an ancestral DDPM comparison is optional future work,
not required for this baseline.

Test forward marginals and skipped-step DDIM equations against analytic fixtures,
including exact clean conditioning, terminal SNR, independent state/latent noise,
same-seed determinism, different initial-noise paths, final clean output,
full-range 20-update timestep selection and invalid-index rejection. Changing
inference step count can reuse the checkpoint but requires its own quality/latency
evaluation; changing the training schedule is a separate training contract.

Use Table S7's effective batch 512, LR 1e-4, weight decay 0.001, cosine LR,
10,000 optimizer-update warmup, EMA power 0.75/max 0.9999, and a proposed full-run
cap of 1,000 epochs. AdamW, BF16 after FP32 checks, gradient clipping 1.0 and
microbatch/accumulation are declared implementation choices. Pilot duration and
warmup must be proportionate and labeled; do not run 1,000 epochs automatically.
Define an epoch's sample budget under balanced sampling. Handle partial final
accumulation batches correctly; persist RNG/sampler/scaler/optimizer/EMA/LR state.

Start with mean squared error over modeled dimensions/timesteps/batch, and report
state, latent, root/body, physical-unit and per-noise-level errors separately.
Measure loss on unknown entries as well as all entries so copying clean context
cannot masquerade as generation. Evaluate EMA weights and sampler behavior, not
only the raw model's reconstruction loss. Record checkpoints/data/config hashes.

### 5.2 Beginning of a trajectory and resets

The **paper does not specify startup padding**. The reference zero-pads it; our
initial baseline will instead use real history:

- Start from a simulation-qualified VAE initialization/reference segment.
- Execute that VAE and collect at least eight contiguous actual state/latent
  pairs; if it fails or resets, discard the history and restart the warm-up.
- At the next tick, condition on that real history and current state and hand
  control to diffusion. This does not claim eight ticks are enough to stabilize
  every initial pose: qualification of the initialization is separate.
- After handover, no reference or encoder is available to the planner/decoder,
  except explicit user-requested sparse keyframes. Keep references only in an
  isolated evaluation recorder where appropriate.

Log that this is a **VAE-bootstrapped handover**, not autonomous cold start.
Repeated-state/zero-latent startup or variable-length-history conditioning is a
separate trained-and-tested extension. A standing policy outside the VAE cannot
supply fictitious historical latents. Clear histories independently per env on
reset, teleport, estimator discontinuity, or incompatible model switch.

### 5.3 Guided control

Progress from unguided continuation to full-state sparse keyframes drawn from
qualified motions, then differentiable costs. At 50 Hz, the paper's 0.2 s
keyframe interval is 10 ticks. Do not condition on future latents or densely
reconstruct the reference and call that task-agnostic generation.

For full-state inpainting, transform each target consistently into the planning
frame and projected representation and fix its state noise level at zero;
leave its latent unknown. Partial pose constraints require a correct physical
constraint/projection operator or soft cost, not falsely marking an entire
partially known state as clean. Explicit mask-conditioning extensions require
matching training and are not silently added to the independent-noise baseline.

Cost implementation order:

1. Physical-space pose/keyframe costs compatible with the initial tennis cohort.
2. S5 planar velocity; restore current root velocity before comparing to the goal.
3. S6 waypoint position/near-goal velocity blend with consistent goal frames.
4. S7/S8 per-body SDF/radii/barrier and composition, with body/world poses
   reconstructed correctly from root trajectory plus local body features.

Velocity/navigation/avoidance behavioral claims depend on a qualified locomotion
cohort. Their tensor/gradient tests do not require new teachers, but tennis-only
closed-loop results must not be marketed as those demonstrations.

The paper does not fully specify the guided reverse update. Frozen engineering
convention: compute `G` on recovered physical clean-state predictions, using
`dt * sum(cost_t)` per trajectory, and differentiate through the frozen denoiser
to the current noisy trajectory. Zero gradients on fixed conditions, then clip
the per-trajectory L2 norm to 1 (do not normalize every gradient to unit length).
With `a = alpha_bar[k]` and this clipped gradient `g`, use:

```text
score_guided = score - lambda * sqrt(a) * g
x0_guided    = x0_hat - lambda * (1-a) * g
```

These are equivalent score/x0 conventions for the selected forward process;
recompute epsilon from `x0_guided` for the DDIM jump. The noise weighting is an
explicit engineering choice, not a claim about the paper's undisclosed code.
`eta=0` does not disable guidance. Initial continuation uses `lambda=0`; positive
task strengths, weights and goals are preregistered in D4 before guided trials.
The D0 CPU conversion oracle is not a denoiser-VJP or physical-cost test. D2/D4
must check input gradients, finite differences, clean masks and same-noise
cost effects, including time/weight scaling at 50 Hz.

This differentiable baseline is not proof of the paper's exact CppAD code path.
It intentionally tests whether costs change the **executed latent and actual
motion**, not only predicted states. Forward-only TensorRT does not supply a
Transformer input VJP: guided deployment must preserve the selected derivative
path or validate a separately named approximation. Include backward/extra
forward evaluations in latency accounting.

### 5.4 Real-time policy semantics

Separate numerical/simulation validation from wall-clock validation:

- First use a synchronous deterministic simulator harness to isolate algorithmic
  correctness; slower-than-real-time simulation is acceptable at this stage.
- Production candidate: asynchronous GPU planner, synchronous 50 Hz decoder,
  bounded latest-request queue, immutable input snapshot and plan sequence IDs.
- Each plan carries source timestamp, root frame, VAE/schema hash and a latent
  sequence with physical target times. Select the latent for the **execution
  tick**, not always the stale source tick's `z_t`.
- A latest-wins planner must not let late results from old requests overwrite a
  newer plan. Record the latent actually decoded, not every predicted latent.
- Timestamp-aligned future-latent use addresses scheduling; it does not eliminate
  errors caused by unexpected state changes during planning. Evaluate delayed
  execution and disturbances explicitly.
- On missed updates, use only a bounded, still-valid suffix of the previous plan
  with current proprioception; no indefinite latent hold. On plan exhaustion,
  non-finite output or estimator invalidity, invoke a prevalidated fallback or
  terminate the isolated simulation. Physical fallback is a deployment gate.

Target a new usable plan every 20 ms. Measure batch-1 **end-to-end** latency,
including history/state preprocessing, transfers, every denoising/guidance
operation, unprojection, latent transfer, decoder, and scheduling. Report
p50/p95/p99/max, plan age, deadline misses, GPU/CPU load, memory, power mode and
thermal steady state. Proposed engineering target: p99 compute path <=15 ms to
reserve 5 ms for integration/jitter, plus measured 50 Hz deadline compliance.
This is a proposed budget, not a measured RTX 5080 result or a hard-real-time proof.

D0's corrected eager/BF16, untrained 41x231 tensor probes measured wall medians
approximately **21.44 ms** for 20 forward/DDIM jumps and **80.97 ms** for a
one-forward/input-VJP proxy. These exclude full runtime costs and do not validate
the planned conditioning/guidance semantics; see the D0 report's parent review.
They are a reason to profile/optimize, not proof that the 5080 cannot meet 50 Hz.
Neither the original flawed measurements nor these bounded samples certify p99.

Benchmark realistic guided and unguided modes separately. Start with FP32 parity,
then evaluate BF16/FP16, compiled inference/CUDA graphs and optional TensorRT.
Do not silently shrink the horizon/backbone/step count or reduce planner rate
when a deadline fails; report the trade-off for approval. If an asynchronous
slower planner still drives a 50 Hz decoder, label it **multi-rate**, not 50 Hz
replanning. Do not transfer laptop timing claims to Orin or a tethered robot.

## 6. Milestones and acceptance gates

These are ordered proposal phases, not launched jobs or standalone queued tasks.
Every phase hands over a versioned artifact and a validation report. Failed
numerical/closed-loop gates return to the relevant phase rather than being hidden
by larger runs. D0 freezes collection qualification and numerical tolerances.
Later task-specific closed-loop tolerances must be preregistered against matched
VAE baselines before D3/D4 comparisons; they are not invented paper targets.

**Current milestone status (2026-09-29): D0 specification/audit complete, with
runtime risk explicitly accepted.** Artifact selection, numerical contract,
qualification thresholds and a bounded pilot design are frozen. **D1 is not
started:** no sequence collector, qualification pilot or diffusion dataset has
been produced by this work. D2 and later phases are likewise not started. Design
approval is not test evidence or authority to launch runs.

| Phase | Work and artifact | Done condition / next handoff |
| --- | --- | --- |
| **D0 — freeze specification and feasibility (closed)** | Pin VAE/cohort and prior accepted evidence; freeze root/body schema, preprocessing, noise/DDIM grids, guidance convention, qualification/numerical tolerances and bounded pilot manifest; audit tensor timing | Complete as specification/audit with explicit 50 Hz runtime-risk acceptance. New adapter/parity/qualification tests are D1/D2 gates; sustained runtime is D5. No new simulator baseline claimed; see report section 11 for gate reconciliation |
| **D1 — sequence collector and dataset** | Clean pilot, OU pilot, acceptance/endpoint audit, indexed shards, grouped split, train-only statistics and projection | Replay reproduces stored clean actions; no reset/teleport/time leakage; acceptance and phase coverage auditable; bounded disk/RAM estimate. Only then approve approximately 100-fold accepted coverage |
| **D2 — diffusion tensor/training core** | Pure CPU tests, forward schedule/DDIM jumps/masks, Transformer, trainer/checkpoint/EMA; tiny fixed-batch overfit then bounded single-motion pilot | Analytic noising/sampling tests and resume pass; true inference conditioning works; normalized-latent inverse and decoder parity pass; fixed-seed held-out generation beats trivial copy/zero diagnostics. Hand candidate to D3, not directly to hardware |
| **D3 — unguided closed-loop continuation** | VAE-history handover, reference-free generation, current-proprio decoder, one-motion then qualified multi-motion runs | Per-motion survival/physical errors/actions assessed against frozen VAE baselines on matched initializations; predicted versus realized state consequences measured; no hidden encoder/reference fallback; history/reset/plan-age tests pass |
| **D4 — inpainting and cost guidance** | Sparse in-distribution keyframes first; physical cost gradients; conditional transitions; locomotion/navigation only with a qualified relevant cohort | Same-noise comparison changes executed latents and improves realized task metrics without unacceptable stability regression; fixed context remains fixed; transition failures/strong-guidance failures reported separately |
| **D5 — 50 Hz runtime, export and sim2sim** | Async scheduler, stale-plan handling, numerical export parity, guided backend, RTX 5080 sustained timing, independent vendor-simulator execution with injected delay/dropouts | End-to-end 20 ms cadence and declared p99 budget demonstrated in specified workload; no tensor-only timing claim; sim2sim policy/safety checks pass. Deliver versioned runtime bundle and residual-risk report |
| **D6 — hardware qualification (blocked on availability/authorization)** | State-estimator and communication validation, actual target topology, safe handover/fallback, bounded robot trials | Separately authorized physical evidence. D0–D5 success alone is not hardware acceptance |

D1 and the pure tensor portion of D2 can be engineered independently after D0,
but trained-model acceptance depends on validated D1 data. D3 should catch policy
failure **before** spending the full multi-motion training budget. Early synthetic
latency profiling belongs in D0; realistic sustained timing is repeated in D5.

Evaluation matrix:

- VAE reference tracking baseline, stored-latent replay diagnostic, diffusion
  continuation, keyframe-conditioned execution, guided execution.
- Clean and perturbed starts; reference-phase and separately qualified standing
  starts; handover, within-behavior operation, transition and endpoint buckets.
- At least three fixed evaluation seeds for meaningful candidates; per-motion
  counts and confidence intervals, not only an aggregate score. Budget the trial
  count in D0 rather than silently launching a large evaluation.
- Survival/falls, task error, root/body pose and velocity error where a target is
  defined, action derivative **per second**, joint/torque saturation, contact
  quality, latent norms/OOD diagnostics, predicted-realized consistency.
- Unguided samples may choose different valid continuations: do not interpret
  deviation from one arbitrary reference as failure by itself. Inpainting tests
  provide an unambiguous target for reference-based tracking comparisons.
- Check held-out noise masks and clean-context inputs; monitor motion modes and
  history-induced repetition, not only denoising MSE. No learned-distribution or
  feasibility guarantee follows from a small offline loss.

## 7. Proposed placement and integration seams

```text
src/mjlab/tasks/tracking/diffusion/
  config.py, contract.py              # VAE/cohort/state/control/data identity
  state.py, projection.py             # shared offline/online feature transforms
  collector.py, dataset.py            # frozen-VAE rollouts, shards, windows/splits
  model.py, schedule.py, sampler.py    # x0 training, DDIM inference, noise/masks
  trainer.py, checkpoint.py           # supervised offline training, EMA/resume
  guidance.py                        # physical costs and declared reverse guidance
  policy.py, runtime.py               # history, timestamped plan/decoder bridge
  evaluation.py, export.py            # behavior, parity and deployment artifacts
src/mjlab/scripts/diffusion.py         # proposed first-class CLI, not yet present
configs/diffusion/x2_50hz.yaml         # proposed config, not yet present
tests/test_tracking_diffusion_*.py
```

Reuse the existing `distillation/model.py` encode/decode interfaces,
`adapter.py` snapshot and boundary conventions, `motion_library.py`, cohort
contracts and `export.py` decoder artifacts. Capture additional physical state
without advancing the observation manager twice. The current
`DistillationSnapshot` is not already the complete diffusion state. Extend via a
narrow dedicated adapter instead of altering accepted DAgger storage semantics.

A new reference-free evaluation environment must disable reference-driven
resampling/teleports and inappropriate reference-error termination **in its own
config**. Keep physical failure criteria; preserve existing VAE/PPO task defaults.
No tracking reference, motion ID or teacher ID is a hidden denoiser/decoder input.

Proposed CLI capabilities: validate contract, collect, audit dataset, train,
evaluate offline, evaluate closed loop, benchmark, export. Document runnable
commands only when implemented. No dependency on the reference repo's installation
or absolute home paths in library code. Real assets stay out of Git.

Training/collection should default to the GenieStudio-managed resources once an
explicit run budget is approved; do not use shared `g1`–`g4` without permission.
Reserve the RTX 5080 laptop for target inference measurements and bounded local
experiments. Use existing remote execution practices and monitored jobs rather
than untracked long processes. No remote action is authorized by this document.

## 8. D0 decision record and next handoff

The user approved the recommended defaults on 2026-09-29. The versioned
[design contract](beyondmimic_diffusion_d0_contract.yaml) is authoritative for v1;
changes require a new revision rather than silently reusing incompatible data.

- **VAE:** frozen `mixed-10k` and its three-teacher cohort; real-robot validation
  confirmed by the user. This does not validate the future diffusion controller.
- **Representation/sampler:** ground-truth pelvis/20-body state, 41-step windows,
  train-only normalization, 64-row projection, 1,000-level bounded cosine schedule,
  and exactly 20 DDIM jumps with `eta=0`. Real VAE history supplies startup.
- **Qualification:** five seconds total, optionally switching to zero-command
  standing without a physical reset. PPO recovery is separate evidence, not latent
  data. Existing physical and reference-error thresholds are kept distinct; see
  report section 11 for exact sources, units and failure precedence.
- **Feasibility:** proceed toward a small prototype despite unproven real-time
  performance. Keep 50 Hz as the deployment target; no silent rate, horizon,
  backbone or step-count reduction. Full runtime acceptance remains D5.
- **Bounded D1 pilot design:** 27 clean then 27 matched OU trials over three motions,
  three starting-phase fractions and three seeds; five seconds maximum per trial.
  Cap total attempts at 54 (failures count), simulated time at 270 s, one GPU/eight
  environments, wall time at 15 minutes including compilation, and output at
  512 MiB. Use a monitored, separately allocated managed resource; no automatic
  retries beyond the budget, bulk collection, training, video or hardware runs.
  No pilot has been launched or authorized by this closeout.

**Next, only after explicit D1 authorization:** implement the collector, replay
and handoff checks, qualification accounting, grouped split and train-only
statistics. Pass CPU/integrity gates before the clean pilot and pass clean
coverage/handoff gates before the OU pilot. Budget exhaustion or missing coverage
produces a partial report, not an automatic larger run. Detailed numerical gates
and pilot stop rules are in the contract. A 1,000-epoch training run is not the
next action.
