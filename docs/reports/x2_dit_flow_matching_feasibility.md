# DiT-style flow-matching backbone: feasibility review

Observation date: 2026-10-21 (+08).
Source: branch `fxy/test/flow-matching-baseline`, HEAD `295402f7374e772a62e74f6cef012cd614eb163f`, plus the existing working-tree changes.
Paper: [SCDP](file:///home/agiuser/Documents/scdp.pdf), arXiv:2603.09574v1, Figure 2 and Sections II-B/II-D.

## Initial recommendation — deferred

Operator follow-up on 2026-10-21 (+08): defer conditioning changes and investigate direct joint q/dq prediction without a latent.
See [the latent-free change-scope review](x2_latent_free_joint_space_feasibility.md). The findings below remain reference material.

**An opt-in DiT-style backbone is feasible.** DiT means diffusion Transformer.
The backbone does not require a change from flow matching to a diffusion noise schedule.
Keep the current clean-token prediction (`x0`), linear noise path, and Euler solver.
Treat SCDP's sensor-only formulation as a separate experiment.

This is a recommendation, not an approved implementation plan or training budget.
The operator requested a feasibility review only.

## Current model

All inspected flow training and inference paths construct `StateLatentTransformer`.
Diffusion training uses the same class.

| Property | Current full-state model |
| --- | --- |
| Input/output | `[batch, 41, 167]` |
| Token | 135 state channels plus 32 latent channels |
| Window | 8 history rows, current row, 32 future rows |
| Hidden width / depth | 512 / 6 blocks |
| Attention heads / feed-forward width | 8 / 2048 |
| Parameters | 20,132,007 |
| Noise input | Two integer levels per row: `[batch, 41, 2]` |
| Conditioning | Learned position, state-level, and latent-level embeddings, added at the input |
| Attention | Fully bidirectional; no attention mask |
| Output meaning | Predicted clean state–latent tokens |

The reduced local-kinematic contract uses 161 channels instead of 167.
It uses the same backbone class.
These are architecture facts, not fresh checks of any running task.

Sources: [model.py](../../src/mjlab/tasks/tracking/diffusion/model.py), lines 73–211;
[process.py](../../src/mjlab/tasks/tracking/flow_matching/process.py), lines 157–189.

## Paper versus proposed change

SCDP uses separate encoders for observations, actions, privileged states, and commands.
It interleaves these streams and changes attention rules between context and prediction.
Its reported model has 6 layers, width 512, 4 heads, and feed-forward width 1024.
It predicts 16 future steps. Action loss is zero beyond 8 steps.

The paper does not specify canonical DiT adaptive LayerNorm with zero-initialized gates (`adaLN-Zero`).
Figure 2 labels a decoder layer; Section II-D calls the backbone an encoder.
Thus an adaLN-Zero implementation is a proposed adaptation, not an exact SCDP reproduction.

SCDP's main contributions also change the learning problem:

- Sensor observations condition the model; privileged future states remain prediction targets.
- Pelvis linear velocity is omitted from denoising inputs but retained in supervision.
- Context pairs use the action actually executed to produce the next state.
- Context attention is bidirectional; prediction attention has causal constraints.

A backbone change alone does not establish sensor-only operation or reproduce these results.
The paper's real-robot inference uses a remote RTX 5090 workstation.
Its throughput does not establish latency on the X2 onboard computer.

## Recommended first experiment

Add a separate, explicitly selected DiT-style model. Preserve the baseline class and defaults.

1. Keep the selected data contract and the two-input model interface unchanged.
2. Initially keep width 512, 6 blocks, 8 heads, and feed-forward width 2048.
3. Keep the current position encoding and bidirectional attention for the first comparison.
4. Replace input-only noise conditioning with per-row adaptive normalization and residual gates.
5. Preserve both independent noise levels in each row's conditioning vector.
6. Keep the selected loss, dropout, state-mask policy, and solver settings unchanged between arms.
7. Train fresh flow weights. Do not resume a baseline checkpoint as the new architecture.

For row `i`, a possible conditioning vector is:

```text
c_i = state_level_embedding(k_state_i) + latent_level_embedding(k_latent_i)
```

Each block maps `c_i` to normalization scale, shift, and residual gates.
A single global timestep cannot replace this information without changing the current training contract.
Per-row modulation is a DiT-style adaptation, rather than a literal image-DiT port.

**Equal width and depth do not imply equal cost.**
A separate `512 -> 3072` modulation projection in each of 6 blocks adds 9,455,616 parameters.
This count includes biases and excludes any other proposed changes.
That is about 47% of the current model's parameter count.
Measure parameter count, memory, throughput, and inference latency.
A later parameter-matched comparison can separate conditioning gains from added capacity.

An internal state/latent token split is another architecture option.
It can split and recombine existing tokens without changing the external data contract.
It does not require the complete SCDP sensor-history redesign.
Test that option separately from adaptive normalization.

## Artifact reuse

For an architecture-only experiment, reuse the existing data, split assignments, statistics, and token caches read-only.
This requires the same frozen variational autoencoder (VAE), token representation, normalization, and window definition.
The VAE need not be retrained. The flow backbone does need fresh training.
A different VAE invalidates this reuse even if latent width remains 32.
Local cache validity and baseline checkpoint presence were not checked in this review.

Full SCDP-style mixed observations need additional data.
The current shard schema stores physical state, actions, latents, and metadata.
It does not store joint positions, joint velocities, or the complete live 99-channel decoder-conditioning vector.
Recovering that exact sensor history from the current cache is not established.
Plan new collection/storage fields unless an independently validated reconstruction source exists.
New observation fields need normalization; unchanged physical fields need not automatically be refitted.
Any changed cache representation needs explicit versioning and validation.

Source: [storage.py](../../src/mjlab/tasks/tracking/diffusion/storage.py), lines 566–585.

The working-tree state-field mask is not SCDP restricted denoising.
It removes input information and excludes the selected channels from the loss.
SCDP retains those channels as prediction targets.
A restricted-denoising experiment must also prevent the solver from feeding those predictions back into restricted inputs.
Changing the loss alone does not establish matching training and inference behavior.

Source: [trainer.py](../../src/mjlab/tasks/tracking/flow_matching/trainer.py), `_forward_loss`.

## Integration scope

This is more than replacing one model class.

- Add an explicit architecture configuration. Reject unknown or misplaced keys.
- Use the selected architecture in flow training, offline evaluation, and closed-loop loading.
- Reconstruct models from validated checkpoint architecture metadata, with legacy fallback for old checkpoints.
- Record architecture settings in readable run metadata and strict resume identity.
- Preserve the flow/diffusion checkpoint-family distinction.
- Update export support, or explicitly refuse the new architecture until export is supported.

The current export wrapper copies baseline submodules and reimplements the baseline forward calculation.
A new model can fail construction or export incorrect mathematics if that wrapper is reused without changes.
Matching input/output shapes alone is insufficient.

Key sources:

- [Flow CLI](../../src/mjlab/scripts/flow_matching.py), training and offline model construction.
- [Flow configuration](../../src/mjlab/tasks/tracking/flow_matching/training_config.py), `from_mapping`.
- [Checkpoint identity](../../src/mjlab/tasks/tracking/flow_matching/checkpoint.py), `model_identity`.
- [Closed-loop loader](../../src/mjlab/tasks/tracking/diffusion/replay_integration.py), `load_flow_denoiser`.
- [Export wrapper](../../scripts/experiments/diffusion_d5_parity/validate.py), `DenoiserExport`.

Several configuration and trainer files already have uncommitted changes.
An implementation must preserve those changes and establish edit ownership first.

## Validation and limits

Recommended software checks:

- Shape, dtype, level-range, per-row conditioning, and deterministic evaluation checks.
- Zero-gated block initialization equals the identity map, not an ordinary active residual block.
- Finite learning steps and eventual gate/attention/feed-forward updates.
- Do not require nonzero gradients in every parameter on the first backward pass.
  Zero residual gates, and possibly a zero output head, intentionally delay some gradients.
- Test attention and level sensitivity after opening gates or brief training, not only at zero initialization.
- New and legacy checkpoint loading, architecture-mismatch refusal, and configuration round trips.
- Exact preservation of known entries in the unchanged Euler sampler.
- PyTorch/ONNX numerical parity before any exported use.

Compare freshly trained baseline and DiT arms at the same update budget and data exposure.
Use matched warmup and learning-rate schedules; do not compare a short DiT run with a mature baseline as a fair ranking.
Use multiple seeds when estimating small effects.
A short run can screen candidates, but no fixed update count guarantees the final ranking.

Report held-out state and latent errors separately.
Then measure closed-loop falls and tracking errors under a matched protocol.
Measure latency at the intended solver count and target hardware.
Offline error is not a substitute for closed-loop or hardware validation.

**Implementation:** not started. Only this report and the shared research index were updated.
**Validation evidence:** parent paper/source inspection plus a read-only `deepseek-v4.1-flash` scout with high thinking.
**Not tested:** new backbone, learning quality, exported parity, latency, simulator behavior, or hardware behavior.
**Operational status:** no tests, training, remote inspection, GPU job, or robot operation started by this review.
**Next action:** clarify the joint-state versus servo-target semantics in the linked latent-free review. DiT implementation is deferred.
