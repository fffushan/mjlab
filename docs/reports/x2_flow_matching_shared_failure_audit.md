# X2 direct and latent flow: shared failure audit

Checked: 2026-10-23 11:18 (+08).
Scope: the two supplied reports and local source at `32f6979ff`.
Validation: source inspection and a small CPU probe with PyTorch `2.9.0+cu128`.
Operational status: remote jobs were not checked, started, stopped, or changed.
No training code, data, or checkpoint changed. No causal fix is validated.

## 1. Evidence and working hypothesis

Inputs:

- [Direct-flow report](x2_joint_pose_flow_training_runs.md).
- [Latent-flow report](x2_flow_matching_divergence_ablation_20261023.md).

Both families learn first, then show abrupt loss growth and incomplete recovery.
The direct report also records very large raw gradient norms.
The latent report does not supply comparable gradient traces.
Thus, a shared numerical mechanism is plausible, but not established.

Both families use `StateLatentTransformer` in
`src/mjlab/tasks/tracking/diffusion/model.py`.
The direct model imports it through `joint_pose/model.py`.
Both use linear Gaussian interpolation, clean-token x0 mean squared error, AdamW,
and gradient clipping before the optimizer step. Both support bf16 training.
The default architecture uses six pre-normalized blocks, width 512, eight heads,
and feed-forward width 2048. These common components deserve attention before
VAE-specific explanations.

The direct lane has no VAE latent. A VAE preview defect cannot explain both
failures by itself. Different mechanisms remain possible.

## 2. Corrections to the report interpretations

### Gradient clipping and AdamW

Both update paths clip before `optimizer.step()`:

- `joint_pose/runner.py`, `FlowMatchingRunner.learn`.
- `flow_matching/trainer.py`, `FlowTrainer._optimizer_update`.

`clip_grad_norm_` returns the norm before clipping. The direct log records that
return value. A logged norm of one million is not the gradient AdamW receives.
With a finite post-clipping total norm at most one, each gradient coordinate is
also bounded by one. The reported trillion-scale second-moment injection does
not follow from these update paths.

The CPU probe assigned gradient `[1e6, 0, 0, 0]` to one parameter tensor. Clipping
returned `1000000`, left norm `1`, and AdamW stored maximum `exp_avg_sq = 0.001`
after its first update. This checks API semantics, not any remote optimizer state.
Gradient clipping also does not directly bound the AdamW parameter-update norm.

A moderate prediction error can coexist with a very large parameter gradient:
`gradient = model_Jacobian^T * output_error_gradient`.
Large hidden derivatives do not require large output values.
Log activations, parameter gradients, and actual parameter updates separately.

### Ablation limits

- Failure at batch sizes 128 through 2048 shows that batch reduction alone was
  insufficient. It does not rule out a batch interaction.
- Failure at learning rates 1e-4 through 2e-4 does not rule out learning-rate
  sensitivity. Lower rates and other architecture/data combinations remain untested.
- Another paper's stable weight decay does not rule out weight-decay interactions here.
- The latent table has best epochs 16, 38, and 39 in smaller-batch arms. It does
  not support a universal failure threshold at epochs 85 through 98.
- The short and long runs have different learning-rate schedules. At update
  index 85000, the shared helper gives `1.33957e-5` for a 100k budget and
  `1.82298e-4` for a 400k budget, with peak 2e-4 and warmup 10k.
  Short-run stability cannot isolate early stopping from earlier decay.
- Cache gigabytes do not measure independent data diversity. Count unique
  trajectories, motion duration, overlap, and split groups instead.
- Finite values rule out NaN/Inf, not semantic data errors. Joint order, reset
  boundaries, time alignment, velocity derivation, and normalization still matter.
- The direct report's 53.9% versus 4.0% jump rates differ by about 13.5 times.
  They are not, by themselves, a four-times loss or difficulty measurement.
- Direct metrics occur every 100 updates. Those samples do not locate the first
  bad update or establish that one batch caused the change.

The cited latent loss/fall comparison has the same ordering: 0.0244 and 41.7%
are both better than 0.164 and 53.4%. It does not demonstrate reversed ranking.
A small checkpoint set still cannot establish loss as a reliable control metric.

### Initialization and preview

`joint_pose/runner.py` seeds dedicated sampling generators after model creation.
Neither that runner nor `scripts/train_flow.py` seeds model initialization.
The direct arms are therefore not guaranteed to share initial weights, despite
matching `seed` values. An external launch seed was not audited.
The latent CLI does seed before construction.

The latent flow trainer reads cached 135-state + 32-latent tokens. It does not
read the 130-wide VAE encoder reference during training. Changing `lookahead_s`
cannot remove preview information from an existing latent cache. A preview
ablation needs a compatible encoder and regenerated latent tokens and statistics.
Using the old 36-teacher bundle is a useful reproduction control, not a pure
preview ablation.

## 3. Shared model investigation

`diffusion/model.py` sets `norm_first=True` and explicitly omits the final
encoder normalization (`norm=None`). The output projection reads the residual
stream directly. Internal pre-normalization does not bound that residual stream.
This is a candidate for measurement, not a proven cause.

The CPU construction probe also confirms that `TransformerEncoder` clones
identically initialized blocks. They are separate parameters, not shared weights.
Treat independent block initialization as a separate experiment if needed.

Inspect healthy, pre-failure, and degraded checkpoints on the same fixed inputs:

1. Measure loss by noise level, time row, channel group, motion, and data source.
2. Record output and residual-stream root mean square, maxima, and non-finite counts.
3. Record attention logits, attention entropy, and normalization-input variance.
4. Record raw and clipped gradient norms per block, clipping frequency, AdamW
   moments, and actual update-to-parameter norm ratios.
5. Compare FP32 and bf16 on identical tensors and weights. Training/evaluation
   mode and attention backend must be explicit. Check fast-path differences too.
6. Preserve offending window IDs, levels, masks, Gaussian noise, model, optimizer,
   schedule position, and random generator states before applying the bad update.

The latent trainer currently computes the gradient norm but does not put it in
its update metrics. Add that diagnostic before another long run.

## 4. Separate numerical stability from generation quality

The direct lane keeps eight clean history rows and uses one level for all
generated rows. Its loss covers rows 8 through 40: 33 rows, including current.
The latent lane independently noises every row and both slices, including history.
Its sampler instead uses clean known entries and synchronized unknown levels.
The latent denoising validation task is therefore not the deployed generation task.
Bidirectional attention can use relatively clean future rows during validation.
That is not a label-leak implementation bug, but it can make the metric optimistic.

Both losses are plain x0 losses, not unweighted velocity-regression losses.
For noise fraction `s > 0`, squared x0 error equals `s^2` times squared velocity error.
There is no training-time division by `s` in these loss paths. An endpoint
singularity is therefore not established as the training-failure cause.

Latent root-mask runs exclude named channels from the training loss on every
window. Validation still scores all channels without that mask. Also, best
checkpoint selection uses EMA validation, while the report table says
`eval_model`. Match objective, channels, weights, and checkpoint identity before
comparing numbers.

Keep two evaluation groups:

- Stability: a fixed denoising panel, raw and EMA weights, component errors.
- Utility: generation with deployment masks/grid, decoded action error, and
  matched closed-loop simulation. Joint-pose utility also needs reference
  continuity and physical plausibility checks. Do not infer utility from x0 loss.

## 5. Recommended order

1. Add diagnostics and fixed-panel validation. Seed direct model construction.
   Preserve pre-failure snapshots. Do not overwrite existing checkpoints.
2. Replay from a healthy checkpoint with the original schedule and exact draws.
   Compare bf16 and FP32 as one change. Do not replan cosine through a shorter budget.
3. Test a final LayerNorm as a separate model variant. Keep the same data,
   objective, and optimizer recipe. Do not stack this with horizon and scale changes.
   Preserve the old architecture and checkpoint loader.
4. If data tails dominate the first bad gradient, test physically justified
   normalization floors or a robust loss separately. A 29-sigma value can be
   valid motion; blanket target clipping changes the learned motion distribution.
5. Test a shorter effective horizon or near-term weighting for control utility.
   A mean over 32 rows is not automatically twice the gradient of 16 rows.
   Horizon is a worthwhile ablation, not a demonstrated primary cause.
6. Use a raw-gradient skip/abort threshold as a diagnostic guard. Save the batch,
   report its rate, and stop after repeated events. It does not fix the mechanism.
7. Reproduce the old 36-teacher bundle under current code and original settings.
   Reuse its compatible VAE and statistics. This separates current-code behavior
   from the newer data/representation bundle, but not preview alone.

All training, remote evaluation, and architecture changes remain proposals.
They require authorization. No new long-run matrix or data purge is recommended
before a bounded reproduction identifies the first instability.
