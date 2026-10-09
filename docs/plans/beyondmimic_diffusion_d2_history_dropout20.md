# VGK8 diffusion history dropout ablation

Status: implemented, not launched.

## Purpose

Measure whether training with incomplete history improves diffusion generation when
history is unavailable. This is a training ablation. It does not change the D1
VGK8 contract, the collected dataset, the token cache, or the deployment model
shape.

## Training augmentation

The trainer samples one Bernoulli decision per training-window exposure. The
configured probability is `0.20`.

For a selected window, the eight history rows (`0:8`) have both slices masked:

- state slice: `0:135`
- latent slice: `135:167`

The trainer sets both per-token diffusion levels to `K=1000` before it adds the
normal independent state and latent Gaussian noise. The clean state and latent
target remains unchanged. The current row and future rows retain their sampled
levels. The model architecture and runtime input contract remain unchanged.

The decision is re-sampled for every training exposure. A separate seeded random
generator records its state in the checkpoint. The default probability is `0.0`.

## Configuration

```yaml
ablation:
  history_state_dropout_probability: 0.20
```

The resolved training configuration hash changes for the ablation. The D1
contract identity and dataset identity remain unchanged. The ablation starts
from a new model initialization. It must not resume the complete-history
baseline.

## Validation

Focused tests cover configuration validation, deterministic sampling, probability
extremes, both state and latent history slices, clean-target preservation,
checkpoint counters, and offline generation masks.

The offline evaluator accepts:

```text
--history-condition-dropout-probability 0.20
```

This hides both state and latent history rows in the same deterministic test
window subset for stress comparison. The option does not modify the training
checkpoint or the normal runtime path.

## Limits

A normal-history evaluation is required for the regression check. A missing-history
evaluation is an offline stress test. Neither result establishes closed-loop
stability, hardware readiness, or deployment readiness.
