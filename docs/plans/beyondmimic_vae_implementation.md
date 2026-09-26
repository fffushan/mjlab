# BeyondMimic VAE: implementation plan

Architecture: [beyondmimic_vae_distillation.md](beyondmimic_vae_distillation.md).
Repository: `/home/agiuser/projects/mjlab`.
Current status: **M1 and M2 are parent-accepted and committed** as `cdd9a8201` and `76ca932bb`. The [M2 acceptance record](beyondmimic_vae_m2_implementation.md) documents the gravity-first core and explicit future ablation schemas. **M3 implementation and bounded validation are now authorized**, using `lingzhi/gpt-5.6-luna` serial component workers and 15-minute parent supervision under the [M3 plan](beyondmimic_vae_m3_implementation.md). Production training, M4+, diffusion, and hardware work remain separate.

## M1 acceptance record — 2026-09-26

M1 is implemented and parent-validated after one retained-worker correction pass. Parent review caught and verified fixes for relative manifest resolution under an explicit repository root, stale bank device bookkeeping after `Module.to`, saved/exported anchor and body-order consistency plus runner action clipping, and non-finite action parity. The parent also added rejection/tests for empty or whitespace-only teacher IDs.

Final parent-run evidence:

- Four `tests/test_tracking_distillation_*.py` files plus `tests/test_tracking_commands.py`: **70 passed** (`-x -vv -o faulthandler_timeout=30`).
- `tests/test_runner.py -k 'onnx or export_paths' --deselect tests/test_runner.py::test_export_policy_to_onnx`: **12 passed, 6 deselected** (the parent deliberately excluded environment construction).
- Changed-file Ruff formatting/lint, `uv run ty check src tests`, and targeted Pyright: pass. Full-repository Pyright is not an acceptance claim; the worker reported unrelated existing errors.
- Real-cohort validation from `/tmp`, using `uv run --project /home/agiuser/projects/mjlab distill validate-teachers --manifest configs/distillation/x2_tennis.yaml --repo-root /home/agiuser/projects/mjlab`: both teachers pass, 64 samples each, unchanged `atol=rtol=1e-5`.
- Maximum action differences: `tennis_000` **9.5367431640625e-7**, `tennis_001` **1.5497207641601562e-6**. Exported actor weights/mean and embedded reference q/dq match their selected source artifacts exactly. Source artifact hashes remain unchanged across parent review and the correction pass.

Usage is documented in `docs/source/x2_tennis_distillation.rst`; the cohort manifest is `configs/distillation/x2_tennis.yaml`. Physical sensor-frame resolution, body-reference subset matching, and closed-loop rollout quality are **not** established by M1 numerical parity. No VAE, DAgger runner, training run, or hardware deployment is included. At M1 acceptance, the broader project remained at the M2 dependency boundary. M2 has since been separately implemented and accepted under the plan linked above; M3 and training remain unauthorized. No changes were committed, and unrelated `tools/` work was preserved.

## Goal and fixed inputs

Distill the user's two 50 Hz AgiBot X2 tennis trackers into one shared reference-conditioned VAE with a separately deployable decoder. Do not retrain teachers or change their behavior.

Teacher root: `logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/`.

| Teacher | Checkpoint relative to teacher root | Local reference |
| --- | --- | --- |
| `tennis_000` | `tennis_000/model_29999.pt` | `data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz` |
| `tennis_001` | `tennis_001/model_29999.pt` | `data/tennis/single_001_zhanghongyu_agibot_x2_tracking.npz` |

Each run now has original `params/env.yaml` and `params/agent.yaml`. The corresponding ONNX files are `tennis_000/2026-09-23_21-24-56.onnx` and `tennis_001/2026-09-23_19-28-08.onnx`. Their actor networks are 164 -> [512, 256, 128] -> 31, ELU, with individual learned normalizers. Saved configs agree except for the motion file; remote `/home/fushan/mjlab/...` paths must be overridden by the explicitly selected local references, not edited in the source YAML.

## Milestone map

| Milestone | Deliverable | Depends on | Acceptance gate |
| --- | --- | --- | --- |
| **M1: teacher foundation** | Manifest/contract validation, frozen TeacherBank, CPU inference-parity command and tests | Existing artifacts | Both real teachers load; PyTorch actions match original ONNX; routing/normalizers/contracts are tested |
| **M2: latent policy core** | Named reference/decoder-conditioning views, conditional VAE, normalizers, loss, bounded labeled replay | M1 | Gravity default: 68 reference, 99 conditioning, 32 latent, 31 actions, no reference bypass; explicit opt-in anchor/combined schemas; CPU gradient/loss/storage tests |
| **M3: single-teacher DAgger** | Collector/runner, checkpoint/resume, first-class train/evaluate CLI for `tennis_000` | M2 | Correct observation/action/reset alignment; deterministic teacher rollout baseline; bounded smoke training; student-only evaluation |
| **M4: shared two-motion controller** | Motion library, multi-motion command, teacher routing and balanced replay/collection | M3 | No clip-boundary leakage; both teachers correctly routed; per-motion evaluation of one shared student |
| **M5: reproduction and deployment package** | Verified symmetry augmentation, complete evaluation, separate ONNX encoder/decoder and metadata | M4 | PyTorch/ONNX parity, latent-use tests, CPU inference budget, sim2sim validation; hardware requires separate authorization |

Diffusion collection/training/guidance is a later project phase. Freeze the accepted VAE, normalization, schema, and latent convention before collecting its sequence dataset.

## M1: accepted implementation contract (historical)

### Why this is one bounded implementation seam

M1 is the actor-loading boundary, not a miniature end-to-end VAE project. Its manifest, validator, loader, routing, and parity tests share the same artifact-to-inference contract; splitting them between concurrent writers would create overlapping ownership. One worker owns this seam; the parent owns the plan, review, and any resource/architecture decisions. No concurrent source writer.

### 1. Manifest and resolved teacher contract

Add small typed configuration objects and a versioned manifest for the selected cohort. Suggested durable example: `configs/distillation/x2_tennis.yaml`.

Each teacher entry includes a unique ID, checkpoint, local motion, saved environment/agent YAML, original ONNX for parity, and a positive finite sampling weight. Resolve relative paths against an explicit repository root (CLI may default it to the current working directory), and document/test those semantics. Do not hard-code this user's home directory in library code.

Read the original YAML as data, safely handling the dump's Python tuple/name/enum tags without executing arbitrary constructors or imports. Prefer reusing or minimally factoring the safe loader already used by `load_saved_lookahead_s` in `tracking/mdp/commands.py`. Do not blindly instantiate whatever `class_name` or callable a manifest supplies. M1 supports the selected feedforward MLP actor configuration; reject unsupported recurrent/CNN policies clearly.

Produce a resolved contract including paths and file hashes, actor architecture, observation schema/order and saved transforms/noise/delay, action definition, control period, anchor/sensor declarations, reference shapes/FPS, joint-order evidence from the original ONNX, and provenance. Distinguish declared sensor names from physical site/frame resolution, which requires the robot asset.

Validate required files, nonempty/unique IDs, positive weights, finite positive timing/FPS, motion array shapes, checkpoint input/output dimensions, and compatibility across teachers. Validate control period against reference FPS: the current tracker advances one reference frame per control step. Reject action/observation/control incompatibility instead of only comparing vector dimensions.

For this first cohort, conservative equality of saved environment configurations excluding the explicit motion-path difference is acceptable; document every excluded nonsemantic field if broadening that policy. Keep each teacher's learned normalizer statistics separate, not part of cross-teacher equality. Local path overrides must agree with the intended reference, not silently pick any file with a compatible shape.

### 2. Frozen teacher inference

Implement an actor-only `FrozenTeacher`/`TeacherBank` interface, approximately:

```text
TeacherBank.label(teacher_ids[B], teacher_observations[B, D]) -> actions[B, J]
```

Reuse the installed RSL-RL `MLPModel` behavior and checkpoint normalizers. A dummy observation TensorDict can establish dimensions without creating a simulator. Load the actor strictly, including the appropriate saved distribution state if required by the model, but always request deterministic output. Never instantiate a PPO optimizer/critic or update teacher normalizers.

Use `eval()`, freeze parameters, and disable gradients during labeling. Guard against a surrounding student `.train()` call accidentally enabling normalization updates. Group rows by teacher ID, evaluate each needed teacher once per group, and restore original batch order. Define empty-batch behavior, and reject bad IDs/shapes/devices clearly. Do not sample PPO exploration noise or add action clipping.

Reuse existing checkpoint migration rules in `src/mjlab/rl/runner.py` where necessary. A minimal shared helper plus regression coverage is allowed; avoid a broad checkpoint-loader rewrite. Support the actual selected checkpoint format first and reject unsupported formats explicitly if legacy factoring would expand the slice.

### 3. CPU validation entrypoint

Provide a first-class CLI, proposed:

```sh
CUDA_VISIBLE_DEVICES='' uv run distill validate-teachers \
  --manifest configs/distillation/x2_tennis.yaml \
  --repo-root .
```

Names may follow existing Tyro conventions; the worker must document the exact implemented command. No placeholder train command that pretends to work.

The command should validate the cohort and run both real teachers on reproducible finite input batches, including inputs near each teacher's own normalization statistics. Compare deterministic native outputs against the original ONNX `actions` output via CPU ONNX Runtime. Original ONNX inputs are `obs` and `time_step`, with batch dimension 1; handle that explicitly rather than changing the source export. Use representative valid motion frame indices for the time-step input.

Default numerical gate: float32 `atol=1e-5`, `rtol=1e-5`. Record maximum absolute errors and inputs/seed/sample count. If the gate fails, investigate checkpoint/export association and preprocessing rather than loosening tolerances silently. Also verify original ONNX embedded reference joint-position/velocity data against the selected NPZ (tensors may be Constant-node attributes rather than named initializers).

Return/report machine-readable provenance, dimensions, control rate, per-teacher parity results, and explicit unsupported/unverified fields. Nonzero exit on a failed check. Do not present random-input numerical parity as proof of closed-loop motion performance or physical sensor-frame compatibility.

ONNX Runtime is already a development dependency on supported Python versions. Missing validation dependencies should produce actionable errors, not an automatic environment upgrade or a false passing result.

### 4. Tests and documentation

Add focused CPU tests, using small generated checkpoints/configs/motions rather than checking private binary artifacts into Git:

- Correct deterministic actor/normalizer restoration.
- Frozen normalizers and parameters across repeated calls and surrounding train-mode changes.
- Mixed/permuted teacher IDs route to the correct actor and preserve row order.
- Malformed/missing artifacts, unsupported model configuration, bad IDs, mismatched dimensions/FPS, and incompatible action/observation contracts fail clearly.
- Safe YAML tag handling and documented path resolution.
- Generated ONNX roundtrip/parity where the optional test dependency is present.

Keep local real-artifact validation explicit and separate: a skipped fixture on another machine is not evidence that the selected tennis teachers passed. Add usage documentation and an Upcoming changelog entry. Do not fabricate an issue reference.

### Ownership / allowed edits

Primary new paths:

- `src/mjlab/tasks/tracking/distillation/__init__.py`
- `src/mjlab/tasks/tracking/distillation/config.py`
- `src/mjlab/tasks/tracking/distillation/teachers.py`
- A narrowly scoped parity/validation module in that package if useful.
- `src/mjlab/scripts/distill.py` and its `pyproject.toml` entrypoint.
- `configs/distillation/x2_tennis.yaml`
- Focused `tests/test_tracking_distillation_*.py` files.
- `docs/source/` usage/changelog entries and necessary toctree link.

Minimal shared safe-YAML/checkpoint helpers and associated regression tests are allowed only when they reduce duplication without changing existing PPO behavior. Escalate broader changes. The parent-owned architecture and this implementation plan are read-only to the worker; propose amendments in the handoff.

Do not modify `tools/`, existing logs/checkpoints/ONNX/NPZ/config artifacts, robot assets, simulator stepping, tracking rewards, existing experiment defaults, lockfiles, installed dependencies, or unrelated work. No stage/commit/push, external upload, remote server job, hardware command, or subagent fanout.

### Validation and resource budget

- Always use `uv run` for Python/tool commands.
- CPU-only for M1. No environment rollout or GPU compilation is necessary for this seam.
- Inspect/simple commands: 30 s; one test/probe: 60 s; focused suite: 120 s. Use pytest `-x -vv -o faulthandler_timeout=30` while debugging.
- After unexpected timeout, stop and diagnose; do not rerun identically or simply increase the timeout. Preserve exit status when piping output.
- Run targeted tests, changed-file Ruff formatting/lint, `uv run ty check`, and `uv run pyright` (or targeted checks with any repository baseline issues explicitly reported). Parent decides whether broader checks are needed after reviewing the diff.
- Long jobs require a parent-managed monitor; the child should ask instead of spawning untracked processes. Do not execute the full training suite or full GPU-heavy test suite in this milestone.

### M1 definition of done / handoff

A working, documented validation command successfully loads and numerically verifies **both** real tennis teachers without a simulator. Focused tests cover routing, frozen normalization, and rejection paths. Source diff is restricted to the seam; untouched local artifacts remain untouched. Worker returns changed files, exact commands and exit results, real-teacher maximum parity errors, any skipped checks, residual risks, and the API handed to M2.

Parent then inspects the actual diff and runs/consumes focused evidence. Only after parent acceptance may the milestone be called complete. On infrastructure/provider/tool failure, report the exact run failure and partial edits; do not switch models or execution protocols silently.

## M2 and later milestone contracts

### M2: latent policy core — accepted

Implemented and parent-validated under the [gravity-first M2 plan](beyondmimic_vae_m2_implementation.md). Final evidence: 98 focused tests plus 12 export regressions, targeted lint/type checks, original real-teacher parity, and additional normalizer/replay/state probes. The default is the named 68-dimensional encoder, 99-dimensional gravity-conditioned decoder observation, and 32-dimensional latent. Explicit schema modes also prepare anchor-only and gravity-plus-anchor conditioning, without training ablations or changing the default. Controlled normalizers, explicit loss reductions, sampled/mean inference, and bounded raw labeled replay are present. No collection/training runner is included.

### M3: one-teacher closed-loop DAgger — authorized for implementation

Follow the [bounded M3 plan](beyondmimic_vae_m3_implementation.md), including exclusive component ownership, runtime resource gates, and independent parent acceptance.

Integrate with the existing single-motion environment before adding multi-motion commands. Establish the deterministic teacher rollout baseline and physical sensor frames. Snapshot before stepping, use executed previous actions, retain valid pre-failure labels, handle auto-reset and motion-end segments, and implement optional teacher bootstrap/mixing followed by student-only collection. Add resumable checkpoints, logging, evaluation, and bounded resource-configurable smoke runs. Large training runs need a separately specified device/environment/iteration budget.

### M4: multi-motion training

Add motion-indexed reference storage and command/reset handling, no cross-clip indexing or accidental motion-ID leaks into the decoder, equal initial motion weights, and motion-balanced bounded replay. One shared VAE trains on both teachers. Require per-motion baseline-relative evaluation, not only aggregate reward/MSE.

### M5: export and reproduction validation

Add physically verified X2 left/right symmetry for all reference/proprio/action fields, latent-use diagnostics, deterministic/sample rollout comparisons, separate encoder/decoder exports with frozen normalizers and full metadata, and sim2sim integration. CPU latency and hardware deployment are measured separately; numerical export parity is not hardware safety evidence.
