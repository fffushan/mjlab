# D5 numerical and export parity

This directory contains the bounded parity gate for the active no-mixture
representation.

The gate loads one explicit artifact set:

- the `167`-channel diffusion checkpoint;
- the statistics-only state/latent normalizer;
- the active `135 + 32` contract;
- one frozen VAE cohort member whose widths come from its own saved schema.

It then:

1. checks dimensions, finite values, statistics-only keys and normalization
   round trips;
2. checks the `8`-history plus current-state clean mask;
3. runs the frozen `20`-update DDIM sampler and an independent DDIM oracle;
4. exports the denoiser and the VAE encoder and decoder to ONNX;
5. reloads all three graphs with CPU ONNX Runtime;
6. compares the denoiser trajectory and the encoder latent and decoder action
   outputs; and
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

The command writes `REPORT.json`, `denoiser.onnx`, `encoder.onnx` and
`vae_decoder.onnx` to the output directory.  The decoder output is named
`action` and the encoder output `latent`, which are the names the controller
lane requires. The `logs/` directory is ignored by Git. The report
contains the source and generated-artifact SHA-256 values.

The frozen diffusion checkpoint predates the D5 semantic-identity loader change.
The gate accepts its compatible legacy source-bound identity and records both
identities in the report and ONNX metadata. It does not accept an unrelated
checkpoint identity.

The default comparison tolerance is FP32 `atol=1e-5` and `rtol=1e-5`. An exceeded
tolerance is recorded as a warning: `status_detail` becomes
`pass_with_tolerance_warnings`, every exceeded comparison appears in the report's
`tolerance_warnings` list with its measured `max_abs` and `max_rel`, and one
`WARNING:` line is printed. The gate was once blocking. It is not blocking now
because the comparison is between two FP32 implementations of the same graph: a
checkpoint whose forward pass is not FP32-reproducible at the frozen tolerance
cannot pass it through any faithful export. Structural checks stay hard
failures - wrong shape, non-finite values, an unrelated checkpoint identity, a
missing artifact, a wrong provider, or an unsupported graph.

Before treating a warning as harmless, measure the floor: compare one forward
pass of the checkpoint in PyTorch float64 against PyTorch float32. If the
exported graph matches PyTorch about as well as PyTorch matches its own float64
result, the export is faithful and the deviation is the checkpoint's arithmetic.
Measured example (2026-10-13): the VGK36 arms differ from PyTorch by 9.05e-5
while their own float64-to-float32 difference is 8.83e-5, and the accepted
reference differs by 1.01e-5 against a 5.54e-6 floor.

The script uses CPU PyTorch and CPU ONNX Runtime. It does not claim that an ONNX
runtime is selected for `x2_docker`; that is a later integration decision. A GPU
execution provider is a different arithmetic: ONNX Runtime's CUDA provider
enables TF32 by default, and the D5 TensorRT lane is deliberately TF32.
