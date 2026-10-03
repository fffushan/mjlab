# Bounded offline real-model D3 seam replay

Compares the D3 handover seam (`ReferenceFreeContinuation`) against a direct
existing-sampler + denormalize + frozen-decoder path, from identical generator
state and input, using:

- the frozen D2 diffusion checkpoint (`checkpoint-best.pt`, EMA-selected —
  evaluation convention: the stored EMA shadow is copied into the model before
  replay; reported as `ema_selected: true` with the stored EMA `updates`),
- the active no-mixture contract YAML (hash-checked),
- the active no-mixture projection bundle (file digest + checkpoint
  projection-hash cross-check),
- the frozen X2 VAE cohort member decoder loaded through
  `distillation/checkpoint.load_cohort_member_inference` against the live
  cohort manifest, and
- bounded recorded D1 pilot store rows as the measured executed-pair history.

## What this is NOT

- **Not a closed-loop evaluation**: `closed_loop: false` is hard-coded.
- The generated current-index latent is **supplied to the frozen decoder as
  data** and is never executed; no stability, training-quality, real-world
  conditioning or 50 Hz claim is made.
- The 99-D decoder conditioning is a **deterministic labeled synthetic
  fixture** (`synthetic-fixture`): D1 recorded rows carry no measured gravity
  conditioning and reconstructing one from body poses is forbidden. The
  seam-vs-direct comparison does not depend on the conditioning values.

## Exact bounded real-model command

Run from the worktree root (`PYTHONPATH="$PWD/src"`, shared `.venv` read-only):

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/diffusion_d3_replay/evaluate.py \
  --checkpoint /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/checkpoint-best.pt \
  --checkpoint-sha256 252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
  --projection /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/projection.npz \
  --projection-sha256 e258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe \
  --vae-checkpoint /home/agiuser/projects/mjlab/logs/distillation/mixed-10k/checkpoint-final.pt \
  --vae-checkpoint-sha256 69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117 \
  --vae-manifest /home/agiuser/projects/mjlab/configs/distillation/x2_tennis_mixed.yaml \
  --vae-member tennis_000 \
  --vae-repo-root /home/agiuser/projects/mjlab \
  --history-shard /home/agiuser/projects/mjlab/logs/diffusion/d1-pilot-20260930b/store/shard-000000.npz \
  --history-row-offset 0 \
  --samples 2 \
  --seed-0 0 --seed-1 1 \
  --cpu-threads 1 \
  --output logs/diagnostics/diffusion_d3_replay/report.json
```

Bounded to TWO independent windows (seeds 0 and 1) plus one same-seed replay
for determinism, at most 4 continuation steps per window, frozen default
20-step DDIM grid with eta=0.

## Gates (fail closed, no substitution)

- checkpoint / projection / VAE file digests must match the passed SHA256s;
- checkpoint contract identity must equal the active YAML contract's identity
  (legacy projected checkpoints are rejected by the loader);
- checkpoint-recorded projection hashes (matrix / pseudoinverse / statistics)
  must equal the supplied bundle's;
- frozen VAE digest must equal the active contract's pinned VAE identity;
- the cohort member must exist in the live manifest with matching artifacts;
- recorded history rows must be clean measured rows: controller `vae`,
  `latent_present == 1`, no reset/teleport/reference boundary, qualified
  segment, strictly increasing ticks.

## Cheap tests

`tests/test_tracking_diffusion_d3_replay.py` covers loader gates, projection
cross-check, recorded-layout history gates, seam-vs-direct parity with stub
models, EMA selection semantics and CLI bound/hash gates using cheap fixtures.
These tests never require the real artifacts and are not real-model evidence.
