# X2 flow divergence: literature review and recommended diagnosis

Checked: 2026-10-25 (+08).
Scope: local source, saved training metrics, and published literature.
Source revision inspected: `4e4a8e5d1`; the working tree has unrelated uncommitted export changes.
No training, simulator, remote-server, or hardware job was started or checked.
No training source, checkpoint, or dataset was changed.

## 1. Recommendation

Treat the loss explosion as an unresolved optimization or numerical problem.
Do not treat it as an inevitable property of flow matching.
Investigate internal Transformer conditioning before another broad training matrix.
Keep generation quality as a separate acceptance question.

The strongest literature match is attention-logit growth and mixed-precision instability.
This is a hypothesis for X2, not a diagnosed cause.
A failed final-LayerNorm ablation does not test query/key normalization inside attention.

Inputs:

- [State-latent divergence report](x2_flow_matching_divergence_20261024.md).
- [Earlier shared audit](x2_flow_matching_shared_failure_audit.md).
- [Joint-pose findings](x2_joint_pose_flow_divergence_findings.md).
- [Flow runbook](../source/x2_flow_matching_training.rst).

## 2. What the existing evidence establishes

Training loss and validation loss both rise.
This is not the usual signature of overfitting alone.
It does not prove that regularization, representation, or data interactions are irrelevant.

The report's causal exclusions are too strong:

- Failure at `1e-4` and `2e-4` does not rule out an excessive learning rate.
- Failure across tested batch sizes shows that batch reduction alone was insufficient.
- Failure across three model sizes does not rule out a shared architecture defect.
- Reaching zero learning rate freezes model parameters. Flat raw-model loss is then expected.
  This supports early stopping as containment, not stable continued learning.
- Accumulated learning rate is a useful exposure measure, not the actual AdamW update distance.
  Neither a universal update threshold nor a universal learning-rate-area threshold is established.
- Two training seeds cannot establish a population standard deviation of `±7` percentage points.
  Report the observed `13.03`-point gap instead.
- The tested checkpoints do not have the same loss and fall-rate ordering.
  This does not make validation loss useless for detecting catastrophic deterioration.

The reported best-update range is `74k–145k`, almost twofold.
Do not describe it as a precise update clock.

## 3. Additional read-only evidence from saved metrics

Source directory: `logs/diagnostics/remote-metrics-20261024/`.
Method: stream each JSONL file and select `kind == "update"`.
Read `global_step`, `loss`, `gradient_norm`, and `learning_rate` directly.
Sum the recorded learning rates in update order.
Exclude the first 10,000 updates when finding gradient threshold crossings.
All five inspected update streams start at step 1 and have no internal step gaps.
Several files are partial copies; their final row is not the run's final update.

`gradient_norm` is the total norm **before clipping**.
The crossing below is a diagnostic definition, not a measured onset of divergence.

| Saved metric file | Last recorded update | First raw gradient > 10 after warmup | Loss at crossing | LR sum at crossing |
|---|---:|---:|---:|---:|
| `fn-2048-400k.jsonl` | 187,043 | 84,939 | 0.07364 | 15.54097 |
| `cap-6x384-w384.jsonl` | 226,298 | 100,929 | 0.07672 | 18.39435 |
| `cap-4x512-l4.jsonl` | 247,300 | 121,192 | 0.06722 | 21.81024 |
| `lrh-160k-b512.jsonl` | 200,000 | 146,023 | 0.03253 | 15.98013 |
| `lrh-120k-b512.jsonl` | 160,000 | none | — | total 12.00000 |

The 160k-horizon file contains 200,000 updates, despite the report's 160k budget row.
The horizon and the executed update count must remain separate.
The final learning rate is zero in both horizon files.

A particularly informative row exists in `cap-6x384-w384.jsonl`:

- Update: `221582`.
- Loss: `3.8356528282165527`.
- Raw gradient norm: `783058840059904.0`, about `7.83e14`.

A moderate output error with an enormous parameter gradient points toward a severely
ill-conditioned model derivative. It does not identify the responsible layer.
It is much more informative than a loss spike alone.
The late row does not identify the initiating event.

The current trainer already logs gradient norms and has optional batch-capture diagnostics.
The earlier audit's request to add the gradient scalar predates this implementation.
Reuse the existing recorder before adding another logging system.

## 4. Relevant mechanisms from the literature

### 4.1 Attention-logit growth and mixed precision: high diagnostic priority

Zhai et al. [1] identify attention entropy collapse across Transformer tasks.
They associate it with oscillating loss and divergence.
Their proposed remedy, sigma reparameterization, controls linear-layer spectral norms.

Esser et al. [2], section 5.3.2, report a closer flow-matching example.
Their rectified-flow Transformer becomes unstable during high-resolution mixed-precision training.
Full precision remedies that instability at a speed cost.
Query/key normalization controls attention-logit growth and permits efficient mixed-precision training.
Their exact recipe is not a universal prescription for X2.

X2 uses standard `nn.TransformerEncoderLayer` blocks in `diffusion/model.py`.
The blocks use pre-normalization but no explicit query/key normalization.
Pre-normalization constrains the input to the projections, not their learned output magnitudes:

```text
q = Wq * LayerNorm(h)
k = Wk * LayerNorm(h)
attention_logits = q @ k.T / sqrt(head_width)
```

The final output LayerNorm is downstream of these calculations.
Its ablation cannot exclude an internal attention problem.

Measure per block and head:

- Query and key root mean square (RMS).
- Maximum absolute attention logit and its distribution.
- Attention entropy, preferably divided by `log(sequence_length)`.
- Residual-stream RMS and maxima.
- LayerNorm input variance and learned gain.
- Raw gradients by parameter tensor, including input/output projections and embeddings.

Low attention entropy alone is not a defect.
Look for a change before the first gradient surge.
Check LayerNorm derivatives as well as attention; do not assume every large gradient comes from softmax.

### 4.2 Weight, activation, and update magnitude drift

Karras et al. [3] document uncontrolled magnitude drift in diffusion networks.
Their EDM2 design controls activation, weight, and update magnitudes together.
This supports internal scale measurements, not a claim that EDM2 directly fixes this Transformer.

Wortsman et al. [4] reproduce Transformer instabilities with small models.
They study attention-logit growth and AdamW epsilon interactions.
Therefore, reducing the X2 model to 11.6 million parameters does not exclude these mechanisms.

X2 clips gradients before AdamW. Clipping does **not** bound the final AdamW parameter update.
Also, a huge global gradient can suppress otherwise useful gradients in other tensors.
Measure the actual update, rather than infer it from the clip threshold:

```text
relative_update = norm(parameter_after - parameter_before)
                  / max(norm(parameter_before), small_positive_floor)
```

Log both absolute and relative updates. Relative values are unreliable for near-zero tensors.
Measure AdamW's first and second moments and the fraction of denominators dominated by epsilon.
Do not blindly reduce epsilon to `1e-15` because another model used it.

The configured decay is small: `weight_decay=0.001` on all parameters.
For a 400k cosine schedule, decay-only shrinkage is approximately `exp(-0.001 * sum(lr))`.
It is about `0.980` at peak `1e-4` and `0.961` at peak `2e-4`.
This weakens a simple large-shrinkage explanation.
It does not rule out nonlinear interactions or prove that actual parameter norms shrink by those amounts.

### 4.3 Prediction target and timestep weighting: mainly a quality question here

The X2 path and loss are:

```text
x_s = (1 - s) * x0 + s * noise
loss = mean((x0_hat - x0)^2)
```

For `s > 0`, define velocity toward increasing noise:

```text
u = noise - x0
u_hat = (x_s - x0_hat) / s
||x0_hat - x0||^2 = s^2 * ||u_hat - u||^2
```

Thus, plain `x0` MSE is not uniform velocity MSE.
Relative to velocity error, it gives less weight near the clean endpoint.
There is no training-time division by `s` in the current code.
Do not diagnose a `1/s` training singularity that the trainer does not contain.
The implemented Euler update also uses a bounded ratio `r/s` for `0 <= r <= s`.

The original flow-matching framework [5] does not require eventual divergence.
Min-SNR [6] demonstrates conflicting timestep objectives in diffusion training.
Esser et al. [2] study timestep sampling for rectified flow.
A recent numerical study [8] examines interactions between parameterization, weighting, and architecture.
None establishes a universally best parameterization for robot motion tokens.

After numerical diagnosis, test direct velocity prediction as a separate baseline.
Use the finite target `noise - x0` and a matching sampler.
Do not emulate it by attaching an unbounded `1/s^2` weight to the existing loss.
A bounded reweighting experiment is another option, but it changes the objective.
Never compare absolute loss values across different targets as a quality ranking.

### 4.4 Training noise patterns versus deployment

The current trainer independently samples noise levels for every row and both token slices.
The deployment sampler instead keeps known history clean and synchronizes unknown noise levels.
Independent per-token noising is a legitimate design; Diffusion Forcing [7] uses it.
However, the original paper's causal architecture differs from this bidirectional X2 model.
Do not transfer its guarantees automatically.

Bidirectional attention can use less-noisy future rows in the denoising validation task.
Those rows are unavailable at the first generation step.
The full deployment pattern has negligible frequency under independent level draws.
This can make the denoising metric optimistic, but it does not prove the cause of falls.

First evaluate fixed windows using the exact deployment mask and synchronized levels.
Then consider a controlled mixture of independent-noise and deployment-pattern training examples.
For the new conditional examples, score unknown targets explicitly.
Keep independent-noise validation and deployment-pattern validation separate.

### 4.5 Normalization and data tails

`diffusion/projection.py` uses per-feature standardization with a default `std_floor=1e-6`.
The joint-pose report records valid or potentially valid velocities near `29` normalized standard deviations.
Standardization does not remove temporal correlation or heavy tails.

Measure scale, quantiles, and gradient contribution by channel group and trajectory.
Check reset boundaries, time alignment, and finite differences on triggering windows.
A finite array is not proof of valid semantics.

Use physically justified scales or floors only after those measurements.
Do not delete a latent dimension merely because its raw variance is small.
Do not clip valid fast motion merely to improve a plotted loss.
Any normalization change needs new statistics, cache identity, and compatible inference assets.

## 5. Recommended experiment order

All experiments below remain proposals and require operator authorization.

### First: capture and reproduce the first failure

Use a known failing seed, not another wide architecture/data matrix.
Keep the original schedule and the original raw model weights.
Save a rolling checkpoint before the rise, including optimizer, EMA, RNG state, and data position.
Capture exact windows, noise tensors, levels, and masks.
A checkpoint captured after gradients already exploded cannot locate the first cause.

The trainer has batch-capture support, but its saved RNG state alone does not prove
an exact mid-epoch continuation. Validate the replay sequence explicitly.
Record resolved environment overrides, especially `MJLAB_FLOW_LR_HORIZON_UPDATES`.
The current configuration hash does not capture that override.

### Second: bounded, paired branches from the same healthy state

| Branch | Change | Question |
|---|---|---|
| Control | none | Does the captured continuation reproduce the failure? |
| Precision | bf16 to FP32 | Does precision change the first abnormal forward/backward/update? |
| Step size | learning rate multiplied by 0.25 | Does a smaller nonzero update avoid the local failure? |
| Attention, conditional | add query/key normalization | If logits grow, does controlling them remove the precursor? |

Do not compare a freshly constructed QK-normalized model with an unrelated initialized control.
For a retrofit, record the intentional function change and use a matched healthy checkpoint.
For a fresh architecture test, match all shared initial tensors and data/noise draws.

Use full FP32 as a diagnostic reference.
Pin training mode and the attention backend when comparing forward/backward paths.
A short successful branch is local evidence, not proof of long-run stability.
If the failure persists, localize its gradient before changing optimizer epsilon or moments.
An optimizer-reset branch can then test state dependence, but is not a production fix.

### Third: improve useful generation

Once a stable learning branch exists, test these separately:

1. Deployment-pattern denoising and six-step sampling evaluation.
2. Timestep weighting or direct velocity prediction.
3. Physically justified scale changes if channel tails dominate.
4. Near-term and decoded-action metrics for checkpoint selection.

Compare raw and EMA models on fixed inputs.
Use a small common-seed closed-loop panel before an expensive full evaluation.
Repeat promising recipes across training seeds and across more than eight motions.
Report per-motion outcomes; rows sharing a trained model do not measure training-seed uncertainty.

## 6. Do not use these as first-line remedies

- Another million-update run without pre-failure capture.
- A larger cache solely because its byte count is larger.
- Blanket target clipping or deleting low-variance latent dimensions.
- Tightening global clipping indefinitely while ignoring the actual update.
- Switching every architecture, loss, and schedule feature together.
- Treating frozen zero-learning-rate loss as stable continued optimization.
- Assuming a lower denoising loss guarantees a lower fall rate.

## 7. Literature

[1] Zhai et al. (ICML 2023), **Stabilizing Transformer Training by Preventing Attention Entropy Collapse**.
[Paper and abstract](https://proceedings.mlr.press/v202/zhai23a.html).
Evidence: attention entropy collapse, oscillating loss, divergence, and sigma reparameterization.

[2] Esser et al. (ICML 2024), **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**.
[Paper](https://arxiv.org/html/2403.03206v1), section 5.3.2 and Figure 5.
Evidence: rectified-flow mixed-precision divergence, full-precision control, and QK normalization.
Section 3.1 and Appendix B discuss timestep sampling.

[3] Karras et al. (CVPR 2024), **Analyzing and Improving the Training Dynamics of Diffusion Models**.
[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.html).
Evidence: activation, weight, and effective-update magnitude drift; magnitude-preserving design and EMA.
Scope: image diffusion, not a robot-motion Transformer validation.

[4] Wortsman et al. (2023), **Small-scale proxies for large-scale Transformer training instabilities**.
[Paper](https://arxiv.org/abs/2309.14322).
Evidence: small-model attention-logit instabilities and optimizer-epsilon interactions.
Scope: language-model experiments; output-classification-logit remedies do not directly apply to X2 regression.

[5] Lipman et al. (ICLR 2023), **Flow Matching for Generative Modeling**.
[Paper](https://arxiv.org/abs/2210.02747).
Evidence: flow matching as simulation-free vector-field regression.

[6] Hang et al. (ICCV 2023), **Efficient Diffusion Training via Min-SNR Weighting Strategy**.
[Paper](https://arxiv.org/abs/2303.09556).
Evidence: timestep gradient conflicts and loss reweighting for faster diffusion convergence.
Not evidence that Min-SNR fixes the observed X2 explosion.

[7] Chen et al. (NeurIPS 2024), **Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion**.
[Paper](https://arxiv.org/abs/2407.01392).
Evidence: independent per-token noise levels and conditional sequence generation.
Scope caveat: its causal implementation differs from the current bidirectional X2 backbone.

[8] Martin, Gribonval, and Massias (2026 preprint), **Training Flow Matching: The Role of Weighting and Parameterization**.
[Paper](https://arxiv.org/html/2603.06454v2).
Evidence: numerical comparisons of target, weighting, architecture, and data geometry.
Treat this as supplementary, not a universal robot-training prescription.
