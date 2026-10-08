# D5 numerical and export parity

This directory contains the bounded parity gate for the active no-mixture
representation.

The gate loads one explicit artifact set:

- the `167`-channel diffusion checkpoint;
- the statistics-only state/latent normalizer;
- the active `135 + 32` contract;
- one frozen `68/99/32/31` VAE cohort member.

It then:

1. checks dimensions, finite values, statistics-only keys and normalization
   round trips;
2. checks the `8`-history plus current-state clean mask;
3. runs the frozen `20`-update DDIM sampler and an independent DDIM oracle;
4. exports the denoiser and VAE decoder to ONNX;
5. reloads both graphs with CPU ONNX Runtime;
6. compares denoiser trajectory and decoder action outputs; and
7. stamps and verifies contract and artifact identities in the ONNX metadata.

The active loader uses only `state_mean`, `state_std`, `latent_mean` and
`latent_std`. The persisted `projection.npz` may also retain identity
`matrix` and `pseudoinverse` arrays for historical `ProjectionBundle`
compatibility. The gate checks that these arrays are identity and confirms that
the active `StateLatentNormalizer` does not apply them.

## Run the active local artifacts

Run from the `mjlab` repository root:

```sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv
export PYTHONPATH="$PWD/src:$PWD/tests"

OUT=logs/diagnostics/diffusion-d5-parity-$(date +%Y%m%dT%H%M%S)
uv run --no-sync python scripts/experiments/diffusion_d5_parity/validate.py \
  --output-dir "$OUT" \
  --expected-diffusion-sha256 \
  252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae \
  --expected-vae-manifest-sha256 \
  587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d
```

The command writes `REPORT.json`, `denoiser.onnx` and `vae_decoder.onnx` to
the output directory. The `logs/` directory is ignored by Git. The report
contains the source and generated-artifact SHA-256 values.

The frozen diffusion checkpoint predates the D5 semantic-identity loader change.
The gate accepts its compatible legacy source-bound identity and records both
identities in the report and ONNX metadata. It does not accept an unrelated
checkpoint identity.

The default comparison tolerance is FP32 `atol=1e-5` and `rtol=1e-5`. The
script uses CPU PyTorch and CPU ONNX Runtime. It does not claim that an ONNX
runtime is selected for `x2_docker`; that is a later integration decision.
