# X2 flow matching: FP32 recovery and the mixed-precision literature

Reviewed: 2026-10-27 (+08).
Scope: supplied reports, selected local source, published papers, and recent preprints.
No training, simulator, remote-server check, or hardware test was performed.
No training source, checkpoint, or dataset was changed.

## 1. Answer

Yes. Mixed-precision divergence that disappears in full precision is a known failure class.
The Stable Diffusion 3 paper reports a particularly close rectified-flow example [1].
The X2 signature also matches the established attention-entropy-collapse literature [2].
This is not evidence that diffusion or flow matching requires FP32 in general.

The supplied report compares **BF16 mixed precision with FP32**, not FP16 with FP32.
BF16 and FP16 have different numerical limits.
The distinction changes the likely diagnosis and the useful tests.

## 2. What the X2 evidence supports

Inputs:

- [Production measurements](x2_flow_matching_measurements_20261027.md).
- [Paired precision branch](x2_flow_matching_precision_branch_20261025.md).
- [Attention precursor analysis](x2_flow_matching_divergence_precursor_20261025.md).
- [Earlier literature review](x2_flow_matching_literature_recommendations_20261025.md).

The paired branch restores the same checkpoint, optimizer, data position, and schedule.
Only `mixed_precision` changes.
Across 15,025 shared updates, FP32 loss improves from 0.0286 to 0.0261.
BF16 loss rises from 0.0289 to 0.2414.
The reported BF16 control reproduces the original continuation.
This is strong evidence that the precision configuration changes the failure outcome.

The production note adds longer-duration evidence at a different model shape.
At its observation point, FP32 reaches 905,646 updates without the earlier runaway.
That observation is not a completed million-update result or a multi-seed guarantee.
It does not establish generated-motion quality or closed-loop policy quality.

The failing checkpoint has one extreme attention layer:

| Quantity | FP32 branch | BF16 branch |
|---|---:|---:|
| Layer 3 mean absolute logit | 497.6 | 25,410,428 |
| Layer 3 mean attention entropy | 2.288 | 0.000 |
| Layer 3 normalization input variance | 2.3 | 4,568.6 |

These statistics support attention collapse as the failure signature.
They do not identify the first defective numerical operation.
The local analyzer reconstructs eager attention from saved weights and captured layer inputs.
It does not directly measure the fused training kernel's internal forward or backward arithmetic.
See `src/mjlab/scripts/analyze_flow_capture.py:_attention_stats`.

## 3. Closest literature

### 3.1 Esser et al., Stable Diffusion 3, ICML 2024 [1]

Section 5.3.2 reports instability during high-resolution rectified-flow training:

> “mixed precision training can become unstable and the loss diverges.”

It then states:

> “This can be remedied by switching to full precision training”

Figure 5 shows attention-logit growth and attention-entropy collapse.
The authors use query/key normalization to control the logit growth.
Their implementation uses RMSNorm with learnable scales on queries and keys.
Their resulting BF16 recipe also uses AdamW epsilon `1e-15`.
The paper explicitly warns that the recipe is not universal.

This is the closest precedent for X2: flow training, mixed-precision failure, and full-precision recovery.
The paper's approximately twofold full-precision slowdown is not an X2 throughput measurement.
Do not copy its optimizer epsilon without a separate numerical test.

### 3.2 Zhai et al., ICML 2023 [2]

This paper studies attention entropy across vision, language, translation, and speech tasks.
It associates pathologically concentrated attention with oscillating loss and divergence.
The authors call this pattern **attention entropy collapse**.
They propose sigma reparameterization: spectral normalization of linear layers with a learned scale.

This supports the X2 logit and entropy diagnostics.
It does not establish that every low-entropy head is defective.
It also does not identify BF16 rounding as the cause in X2.

### 3.3 Qiu and Yao, ICLR 2026 [3]

This work analyzes BF16 FlashAttention failure in GPT-2 experiments.
It finds an anomalous spectral-norm increase in one attention layer and selected heads.
Its mechanism combines structured low-rank representations with biased rounding errors.
The errors compound and corrupt weight updates.

The localization result matters for X2.
In the studied failure, recomputing attention output `O = P V` in FP32 stabilizes training.
That output enters the backward quantity `delta = rowsum(dO * O)`.
Thus, a Q/K projection runaway can originate from errors outside the `QK^T` matmul.
The paper's causal claim applies to its reproduced failure, not every production FlashAttention kernel.

### 3.4 Recent preprints: additional hypotheses, not settled X2 diagnoses

**Chen et al., 2026 [4]:** a 450M-parameter model trains normally before late gradient growth.
The reported gradient norm grows a thousandfold without any NaN.
Recomputing two attention backward passes in FP32 removes almost all excess gradient.
The paper studies a forward numerical defect and BF16 rounding of the softmax score gradient.
Rounding violates the exact zero-row-sum property of that gradient.
The resulting error couples to the mean key and can grow with key magnitude.
These findings motivate backward comparisons, not only forward-output comparisons.
The tested architecture and backend differ from X2; transfer remains untested.

**Xie et al., 2026 [5]:** controlled errors at different locations produce the same Q/K runaway.
Correcting the Q/K gradient route can block the runaway while the injected source remains active.
The authors distinguish the error source from the path through which failure grows.
This is a useful warning: a successful Q/K intervention does not uniquely identify the source.

### 3.5 General mixed-precision training [6]

Micikevicius et al. describe FP32 master weights, loss scaling, and FP32 accumulation.
These methods allow reduced-precision training to match full-precision results in many networks.
They are engineering methods, not a guarantee for every architecture and kernel.

## 4. Why BF16 can fail despite its wide range

| Format | Exponent bits | Fraction bits | Significant bits for normal values |
|---|---:|---:|---:|
| FP16 | 5 | 10 | 11 |
| BF16 | 8 | 7 | 8 |
| FP32 | 8 | 23 | 24 |

BF16 has approximately FP32's exponent range, but much coarser spacing between values.
FP16 has finer spacing than BF16 at the same magnitude, but a much narrower range.
Loss scaling primarily addresses small-gradient underflow.
It does not restore lost BF16 significand bits or repair a biased attention derivative.

For normal BF16 values near 30, adjacent representable values differ by `0.125`.
A single round-to-nearest conversion can have an absolute error up to `0.0625` there.
The number 30 itself is exactly representable.
These are representation limits, not measurements of the X2 matmul's total error.

A simplified attention calculation is:

```text
q = Wq * LayerNorm(h)
k = Wk * LayerNorm(h)
scores = q @ k.T / sqrt(head_width)
p = softmax(scores)
output = p @ v
```

Pre-normalizing `h` does not constrain the learned scales of `Wq` and `Wk`.
A final output LayerNorm is also not query/key normalization.
As internal scales grow, rounding can increasingly affect small differences and derivative cancellations.
The forward result can remain finite while its training gradient becomes inaccurate.

The current X2 source uses `nn.TransformerEncoderLayer` without explicit Q/K normalization.
Its ordinary BF16 mode retains FP32 parameters and casts eligible forward operations with autocast.
The loss is already formed from `prediction.float()`.
Making only the scalar loss FP32 is therefore not a new remedy.
See `diffusion/model.py` and `flow_matching/trainer.py` under `src/mjlab/tasks/tracking/`.

## 5. Qualifications to the measurement report's proposed mechanism

1. **Autocast does not prove all attention internals use FP32.**
   Standalone CUDA `softmax` and `layer_norm` are eligible for FP32 autocasting [7].
   Fused scaled-dot-product attention has its own implementation and intermediate dtypes [8].
   The active training backend and its backward arithmetic still need inspection.

2. **Sharper softmax does not inherently produce an exploding derivative.**
   Its exact Jacobian is `diag(p) - p p^T`.
   This Jacobian tends toward zero as `p` becomes exactly one-hot.
   Projection scales, numerical cancellation, optimizer state, and structured errors require separate analysis.
   A small perturbation also cannot generally flip a clear winner with a large score margin.

3. **Absolute logit scale is not the whole attention distribution.**
   Adding the same constant to every logit in one row leaves exact softmax unchanged.
   Measure row-centered spread, top-two score gaps, and per-head entropy as well as absolute scale.
   Large common offsets can still matter numerically.

4. **Different failing layers do not exclude a shared implementation defect.**
   The same kernel executes in many layers.
   Initialization, learned scales, and data can determine where a defect becomes visible first.

5. **A successful narrow FP32 branch would localize sensitivity, not prove the complete feedback mechanism.**
   Preserve attention outputs while replacing backward calculations to test derivative errors separately.
   Also inspect `P V`, reductions, and casts; do not restrict all tests to `QK^T`.

6. **The pure-BF16 plateau does not prove mixed precision itself is conceptually wrong.**
   BF16 parameter storage changes the update quantization as well as the forward calculation.
   Small updates can disappear when written back to BF16 parameters.
   Whether that explains this plateau is not measured.

## 6. Recommended next tests

These are proposals. They require authorization and must not disturb the working FP32 run.

| Order | Change or measurement | Question |
|---|---|---|
| 1 | Fixed tensors and checkpoint; inspect selected attention backend and compare gradients with FP32 | Where does numerical disagreement first appear? |
| 2 | BF16 elsewhere; FP32 attention core, including `QK^T`, softmax, `P V`, and backward | Does attention arithmetic explain the local failure? |
| 3 | Separate FP32 Q/K projection branch; then narrow the successful precision boundary | Are projections or attention internals dominant? |
| 4 | BF16 with per-head Q/K normalization | Can controlling attention scale recover efficient training? |
| 5 | Longer matched runs and multiple seeds; then generation and closed-loop evaluation | Does the remedy generalize and preserve useful policy quality? |

A math-backend control is useful, but record its actual precision settings.
Current PyTorch documentation says its math SDPA backend upcasts BF16 inputs by default [8].
Backend changes can therefore change precision too.
Disable TF32 for a strict FP32 reference and record the installed PyTorch/CUDA versions.
Do not infer strict IEEE FP32 from tensor dtype or from unchanged throughput alone.

Use per-head Q/K norms, centered logits, entropy, gradient error, and actual parameter updates.
Keep the original schedule and identical windows/noise in each paired comparison.
Do not combine Q/K normalization, optimizer epsilon changes, and learning-rate changes in one causal test.
Keep FP32 as the current reference until an alternative passes those checks.

## 7. Sources

[1] Esser et al. **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**.
ICML 2024. Section 5.3.2, QK-Normalization; Figure 5.
https://arxiv.org/html/2403.03206v1#S5.SS3.SSS2

[2] Zhai et al. **Stabilizing Transformer Training by Preventing Attention Entropy Collapse**.
ICML 2023.
https://proceedings.mlr.press/v202/zhai23a.html

[3] Qiu and Yao. **Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention**.
ICLR 2026; arXiv first posted in 2025. Sections 3.1–3.3 and limitations.
https://arxiv.org/html/2510.04212v4

[4] Chen et al. **Broken Symmetry in BF16 Attention: Why FlashAttention Gradients Blow Up Late in Training**.
2026 preprint. Abstract, backward-error analysis, and Appendix K's plain-Transformer control.
https://arxiv.org/html/2609.34272v1

[5] Xie et al. **One QK Channel, Many Sources: Tracing Low-Precision Attention Collapse**.
2026 preprint. Abstract and source/channel intervention experiments.
https://arxiv.org/html/2608.02091v2

[6] Micikevicius et al. **Mixed Precision Training**.
ICLR 2018; arXiv first posted in 2017.
https://arxiv.org/abs/1710.03740

[7] PyTorch. **Automatic Mixed Precision package**, CUDA autocast operation reference.
https://docs.pytorch.org/docs/stable/amp.html
Readable documentation extraction failed; the upstream Markdown was retrieved instead:
https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/amp.md

[8] PyTorch. **Numerical accuracy**, TF32, reduced-precision GEMMs, and SDPA sections.
https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
Readable documentation extraction failed; the upstream Markdown was retrieved instead:
https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/notes/numerical_accuracy.md
The upstream `main` documentation is not proof of the installed runtime's behavior.
