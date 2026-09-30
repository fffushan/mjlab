# BeyondMimic-Reproduction review and comparison with our mjlab distillation

Date: 2026-09-27.
Status: review and comparison notes, not authorization for implementation,
training, or commits. No source changes result from this document.

Repository reviewed: `/home/agiuser/projects/BeyondMimic-Reproduction`
(independent public reproduction of BeyondMimic, arXiv 2508.08241, for the
Unitree G1 with LAFAN1 motion data). Paper source used for cross-checks:
`/home/agiuser/Documents/beyondmimic.txt` (main text Methods plus
supplementary S3 and Tables S6/S7).

Related documents:

- [VAE distillation architecture](beyondmimic_vae_distillation.md)
- [Implementation milestones](beyondmimic_vae_implementation.md)
- [M4 implementation and acceptance](beyondmimic_vae_m4_implementation.md)
- [General tracker and latent diffusion discussion](beyondmimic_general_tracker_and_diffusion.md)
- [Later 50 Hz diffusion reproduction plan and stage-3 source audit](beyondmimic_diffusion_reproduction.md): selects DDPM-style training with DDIM inference, distinguishes sampler semantics, audits startup/history and dataset splitting, and defines the proposed X2 adaptation. The historical stage-2 comparison below is not evidence of stage-3 closed-loop performance.

## 1. The reviewed repository's structure

The repro is contract-driven: every artifact crossing a stage boundary is a
versioned NPZ schema with validators in `src/beyondmimic_repro/contracts/`
(`teacher_rollout`, `dagger_dataset`, `vae_rollout`, `state_latent`). Isaac
Sim/Lab dependencies are isolated behind runtime import boundaries
(`adapters/isaac/`), so the package imports and tests without a simulator.
Pipeline stages are numbered directories under `scripts/` (00 setup through
10 mujoco/isaac entrypoints).

Stage 2 (conditional action VAE distillation, `src/beyondmimic_repro/stage2/`):

- `models/conditional_action_vae.py`: `PaperConditionalActionVAE`, an
  asymmetric conditional VAE matching paper Table S6 exactly — encoder input
  67 (ref q 29 + ref dq 29 + anchor position error 3 + anchor rotation error
  Rot6D 6), decoder conditioning 96 (projected gravity 3 + root linear
  velocity 3 + gyro 3 + joint pos 29 + joint vel 29 + previous action 29),
  latent 32, MLPs [2048, 1024, 512] ELU, lr 5e-4, KL coefficient 0.01,
  gradient accumulation 15, logvar clamped to ±10.
- Three training runtimes: D0 offline BC warm start on teacher rollouts
  (`training_runtime.py`), simple DAgger rounds
  (`train_vae_dagger_runtime`), and a structured/overnight trainer
  (`vae_dagger_structured_training.py`) with motion-environment rollout
  validation splits, per-motion and worst-motion action MSE, early stopping
  on `val_action_mse`, phase-balanced (motion, phase-bin) weighted sampling
  with a custom NumPy sampler beyond 2^24 samples, sagittal-symmetry
  augmentation (`sagittal_symmetry.py`), mixed D0-union-D1 training, and
  resumable history JSONL with milestone checkpoints.
- DAgger semantics in `dagger/collector_core.py` are correct: the student
  action is executed; the teacher action is a label queried at the visited
  state. Protocols (`RobotBackend`, `StudentPolicy`, `TeacherPolicy`) decouple
  the loop from any simulator; the real collector is
  `scripts/09_isaac/collect_dagger_round.py` (AppLauncher-first import order,
  8192 envs headless).
- Bridge to stage 3: `adapters/isaac/live_rollout.py` rolls the trained VAE
  with Ornstein-Uhlenbeck (OU) action noise and an acceptance gate (2.5 s
  perturbation window, accepted only if the robot survives 5 s closed loop),
  emitting the `vae_rollout` contract consumed by the stage-3 state-latent
  dataset builder (4 history + current + 16 future tokens at Table S7
  settings; hybrid 99-D yaw-centric character-frame states or 163-D emphasis
  projections per paper S3).

Stage 3 (`src/beyondmimic_repro/stage3/`) covers the paper's diffusion half:
`StateLatentTransformer` (embedding 512, 8 heads, 6 layers, 20 denoising
steps, independent per-token state/latent noise levels), DDPM-style forward noising
with x0 prediction and DDIM-style reverse sampling, guidance costs matching paper S5–S8 (joystick, waypoint
with position-to-velocity blending, SDF obstacle barrier), and the
receding-horizon `DiffusionVAEPolicyFrontend` that denoises, extracts the
current latent, and decodes with up-to-date proprioception.

Two VAE families coexist: the paper-faithful `stage2/` package and a legacy
state-action smoke baseline (`vae/torch_model.py`) that the README quick-start
still trains — a documentation trap.

## 2. Fidelity observations on the repro

1. Model and training hyperparameters match Table S6 exactly; the encoder/
   decoder asymmetry (reference intent in, proprioception out) is implemented
   as the paper describes.
2. OU noise `dt` differs from the paper: paper S4 sets Δt = 1.0 per control
   step; the repro passes `dt = 1.0 / frequency_hz` (0.02 s at 50 Hz). With
   θ = 0.8 and σ = 0.1 unchanged, the resulting error band is far weaker
   than intended (about 50× smaller per-step drift, ~7× weaker diffusion).
   The acceptance gate itself (5 s survival) matches the paper.
3. Input normalization is declared in contracts ("dataset mean/std") and a
   strict checkpoint format requiring a `normalization` key exists in
   `dagger/checkpointing.py`, but no stage-2 trainer fits or applies mean/std
   normalization; the strict checkpoint builder has no callers. Stage-2 inputs
   are effectively raw units.
4. Honest seams are visible rather than hidden: teacher-rollout loading picks
   the anchor body by name preference (torso_link, Torso, pelvis, body 0) and
   `joint_position_semantics` carries an explicit verify-before-claims caveat.
5. Tests are contract/shape-oriented (OU reproducibility, DAgger schema, VAE
   and diffusion shapes, guidance costs); the test suite is small but focused.

## 3. Comparison with our mjlab distillation

Scope difference: the repro targets the G1 (29-DOF) with LAFAN1 motions and
covers the entire paper pipeline breadth-first, with no real training
artifacts in-tree. Our mjlab distillation (`src/mjlab/tasks/tracking/`)
targets the X2 (31-DOF) with two frozen real tennis teachers, covers only the
VAE stage depth-first (M1–M4 accepted; M5 export/symmetry pending; diffusion
deliberately deferred), and has run real training (4096-env, 10,000-iteration
single-teacher run; multi-teacher policy quality remains to be established).

### 3.1 Contract side-by-side

| Dimension | Repro stage 2 | mjlab distillation |
| --- | --- | --- |
| Encoder input | 67: ref q(29)+ref dq(29)+anchor pos err(3)+anchor rot err Rot6D(6) | 68: ref q(31)+ref dq(31)+anchor rot err Rot6D(6) |
| Decoder conditioning | 96: gravity(3)+root lin vel(3)+gyro(3)+q(29)+dq(29)+prev action(29) | 99: gravity(3)+gyro(3)+q(31)+dq(31)+prev action(31) |
| Architecture | latent 32, [2048,1024,512] ELU | latent 32, [2048,1024,512] ELU |
| Optimizer/hyperparams | Adam, lr 5e-4, β=0.01, accum 15 | Adam, lr 5e-4, β=0.01, accum 15 (minibatch 256) |
| Rollout latent convention | z = mu default, sampled optional | z = mu eval/export, sampled training |
| Reference bypass into decoder | Decoder takes (latent, proprio) only | Structurally enforced + schema identity in `state_dict` |
| Input normalization | Declared, not implemented (raw units) | `StudentNormalizer`: Welford, atomic, frozen at eval |
| Loss reduction | Element-wise `torch.mean` | Sum over dims, mean over batch (documented) |

Both omit anchor-position error or root linear velocity relative to the paper
for deployability in our case only: the repro keeps the paper's full input
set; we deliberately dropped anchor position error and root linear velocity
because the X2 teachers are no-state-estimation. These are documented
adaptations in our architecture doc, not silent deviations.

### 3.2 Where both agree with the paper (and each other)

- Encoder sees only reference intent; decoder sees only proprioception; the
  latent is the only command interface. Neither has reference bypass.
- DAgger semantics: student executes, teacher labels at the visited state;
  executed (not counterfactual) previous action in conditioning.
- Deterministic latent at deployment; sampled during training.
- KL β = 0.01, gradient accumulation 15, no adversarial or temporal losses.
- The OU-perturbed VAE rollout + survival acceptance belongs to diffusion-data
  collection, not VAE supervision — the repro places it there; our plan
  defers it to the later diffusion phase the same way.

The independent convergence of both codebases on the same Table S6 reading
strengthens our confidence in our interpretation of the paper's VAE stage.

### 3.3 Methodological divergence: training topology

The repro is artifact-oriented and round-based: Isaac collectors write DAgger
NPZ shards; separate offline trainers run up to 1000 epochs with early
stopping on `val_action_mse`; D0 warm start, then D1 rounds, then mixed
D0-union-D1 training. Ours is process-oriented and online: one runner
interleaves closed-loop collection and bounded-replay optimization in-process
with health/poison guards, RNG-state checkpointing, and fresh-data-gated
normalizer updates; `teacher_probability` bootstrap/mixing (whole action
vectors) replaces their D0 warm start. Both are legitimate DAgger cadences:
theirs audits well as durable files; ours trains efficiently and couples
normalizers to data but produces fewer durable intermediate artifacts.

### 3.4 Notable differences

1. Loss-reduction conventions differ but nearly cancel numerically: the repro
   means over B×29 recon elements and B×32 KL elements (effective per-sample
   recon:KL weight ratio ≈ 110); ours sums per sample and means over batch
   (ratio 100). Ours documents the convention; the repro's is incidental.
2. Sampling/balancing: the repro phase-balances inside the trainer over
   (motion, phase-bin) cells; we control phase coverage at collection
   (uniform phase sampling, stratified per-motion environment slots) with
   motion-balanced replay quotas and adaptive phase sampling refused for the
   M4 baseline. Their trainer-side phase balancing is the more complete
   answer to phase coverage we currently lack.
3. Symmetry augmentation: implemented in the repro (`sagittal_symmetry.py`);
   deferred to our M5 and reported as a deviation until then. Their
   joint-name-driven mirror index/sign mapping, polar-vs-axial vector
   handling, and Rot6D sign flips are a concrete reference for our M5 work,
   but the X2 joint set and frames must be re-derived physically.
4. Validation: the repro holds out whole (motion, environment) rollouts for
   offline `val_action_mse`; we hold no offline split and rely on closed-loop
   acceptance gates with mandated per-motion evaluation. Theirs fits their
   offline trainer; ours fits the actual question (does the student track).
5. Teacher rigor: ours is far ahead (manifest/contract machine with file
   hashes, saved-config recovery, 1e-5 ONNX parity, frozen per-teacher
   normalizers, order-preserving routing); the repro's teacher side is
   Protocols plus synthetic fixtures with external Isaac adapters.
6. Stage-3 coverage: the repro has the full diffusion half (hybrid state,
   emphasis projection, per-token noise levels, guidance costs S5–S8,
   receding-horizon frontend); we have none yet, deliberately, until the VAE
   and latent convention freeze.
7. Export: ours is ahead (encoder/decoder ONNX, schema metadata, parity tests); the repro's stage 2 has no ONNX export path.
8. The repro's OU `dt` bug (§2.2) is directly relevant to our future
   diffusion-data collection: do not copy `dt = 1/frequency_hz`; the paper's
   Δt = 1.0 is per control step.

### 3.5 Borrowing directions

For our continuation (M5 and the diffusion phase):

- Phase-balanced (motion, phase-bin) sampling in the trainer, alongside our
  collection-side phase control — a small addition to `balanced_storage.py`.
- X2 sagittal-symmetry augmentation was considered for M5 but is **aborted for this project**. The observed sim-to-real gap makes synthetic mirrored teacher labels an unvalidated assumption. Future cohorts will use manually selected balanced reference motions and user-qualified teachers instead.
- Persisting validated DAgger rounds as schema'd artifacts, like the repro's
  NPZ contracts, for auditability across collection sessions.
- Their NumPy weighted sampler for very large replays (>2^24 samples), if our
  replay ever grows that large.

For the repro (if we ever patch/upstream): our `StudentNormalizer`,
schema-in-`state_dict` mismatch rejection, teacher manifest/provenance
rigor, and (motion, environment) leakage-safe splits are things it claims in
contracts but does not enforce.

## 4. Bottom line

The two efforts implement nearly identical learned components — same
architecture, hyperparameters, latent conventions, and no-bypass decoder
contract — validating a common reading of the paper's VAE stage. They differ
in what surrounds the model: the repro is breadth-first for the G1 with the
paper's exact input schemas and the full pipeline down to guidance; ours is
depth-first for the X2, trading two paper input features for
no-state-estimation deployability and stage-3 breadth for teacher
provenance, normalizer hygiene, in-process DAgger, and export infrastructure.
Neither set of choices is wrong; they sit at different points on the
fidelity-vs-deployability axis, and each side has concrete pieces the other
lacks.
