# Plan: reproduce one flow divergence with pre-failure evidence

Date: 2026-10-25 (+08).
Status: proposal. No job was started. All commands below need operator authorization.
Scope: the state-latent flow trainer and its diagnostics sidecar. Training and simulation only.
Input: `docs/reports/x2_flow_matching_literature_recommendations_20261025.md`.

## 1. Short answer

The trainer already records most of what the literature report asks for. The per-update log
holds the raw gradient norm, the per-block gradient norms, the loss decomposition by level,
by window row and by channel group, and the learning rate. The offline analyzer already
computes attention logits, attention entropy, LayerNorm input variance, block activation
norms and AdamW moment scales from a saved capture.

Four things are genuinely missing, and one of them blocks the whole investigation:

| Gap | Why it blocks |
| --- | --- |
| No capture exists **before** the first gradient rise | Every saved capture starts after the crossing, so no precursor can be seen |
| The loss trigger fires at `8 ×` the median loss | By then the event is minutes old and the weights have already moved |
| Resume has never been shown to be **exact** | Without that proof, a control branch and a treatment branch are not comparable |
| No pre-failure checkpoint ladder | Only `checkpoint-best` at the loss minimum and five consecutive trigger checkpoints at `107 069` to `107 073` exist |

The plan below fixes these in one instrumented run of about `2.5` hours, then runs four
paired branches from a pre-failure checkpoint.

## 2. What already exists

Read this list before you add anything. A duplicate of these would waste the run.

| Capability | Where | Notes |
| --- | --- | --- |
| Per-update `gradient_norm`, pre-clip | `flow_matching/trainer.py`, `_optimizer_update` | `clip_grad_norm_` returns the norm before rescaling |
| Per-block gradient norms, 12 blocks | `diagnostics.gradient_norms_by_block` | Six encoder layers plus projections, embeddings and `root` |
| Loss by noise level, window row, channel group | `diagnostics.observe_batch` | 22 level bins, 41 rows, channel groups |
| Loss by state and latent slice, unknown and clean subsets | `trainer._x0_metrics` | Per update row in `metrics.jsonl` |
| Saved capture of a batch | `diagnostics.dump` | Window indices, both level tensors, both masks, per-window loss, generator state |
| Offline replay of a capture | `scripts/analyze_flow_capture.py` | Rebuilds the exact noisy batch and reports a replay error |
| Attention logit magnitude and entropy | `analyze_flow_capture._attention_stats` | Mean and max only, per layer |
| LayerNorm input variance and activation norms | `analyze_flow_capture._hook_pass` | Per layer |
| AdamW moment scales per block | `analyze_flow_capture._adamw_moments` | From the checkpoint optimizer state |
| Checkpoint with optimizer, EMA, RNG, step, epoch | `checkpoint.save_flow_checkpoint` | Everything a resume needs |
| Model shape in the checkpoint | `model_identity.type` and `.settings` | The closed-loop loader already reads it |

Existing diagnostic coverage is therefore wide. Eight of the nine arms in the capacity and
final-norm groups ran with the sidecar enabled.

## 3. What the existing evidence cannot answer

Measured from the saved files on 2026-10-25.

| Arm | Best epoch | First gradient > 10 | First capture | Captures | Logged updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cap-6x384-w384` | 25 | `100 929` | `107 069` | 4 455 | 400 000 |
| `cap-4x512-l4` | 29 | `121 192` | `146 202` | 470 | 400 000 |
| `fn-b512-400k` | 29 | not crossed in copy | `149 030` | 7 | 400 000 |
| `fn-2048-400k` | 82 | `84 939` | `76 529` | 15 | 304 515 |

Three consequences.

1. In `cap-6x384-w384` the first capture is `6 140` updates **after** the crossing. Its ring
   holds 16 micro-batches, so it reaches back only to about `107 053`. The onset region
   `100 929` to `107 053` has no batch capture at all.
2. `cap-6x384-w384` has five trigger checkpoints at `107 069` to `107 073`. Five consecutive
   updates do not form a ladder, and all five sit after the crossing.
3. The only capture from before a crossing belongs to `fn-2048-400k` at `76 529`. That arm
   crosses at `84 939`. One window is not a trend.

The stored log still answers one class of question without any new run: it holds the raw
gradient norm and the 12 block norms at every update from step 1, so the first abnormal
**gradient** movement can be located offline. What it cannot do is show which internal
quantity moved first, because activations were never stored.

## 4. Reproduction target and design

### 4.1 Target arm

`cap-6x384-w384` has `11 560 103` parameters. It uses batch `512`, peak rate `2e-4` and a
horizon of `400 000` updates.

| Reason | Detail |
| --- | --- |
| Fail-fast | Crossing at `100 929` updates, then a `7.83e14` gradient at `221 582` |
| Cheap | `185 MB` per checkpoint against `322 MB` for the production shape |
| Fast | The full `400 000`-update run took `11.25` hours, so about `2.7` updates per second |
| Diagnostic history | Sidecar enabled for the whole run |
| Pre-failure state exists | `checkpoint-best.pt` at epoch 25, about `90 825` updates, save time `2026-10-24 17:29 (+08)` |

Limit: this is the smallest model, so a mechanism found here must be re-checked on the
production shape before any claim about the `20.1 M` model.

### 4.2 Why a resumed run can be compared with the original

The checkpoint stores the model, the EMA, the optimizer state, the global step, the epoch
and both generator states (`torch`, `torch_cuda`, `sampler_generator`, `noise_generator`,
`state_mask_generator`, `history_dropout_generator`). The sampler draws from the restored
`sampler_generator`, so a resume from `checkpoint-best.pt` reproduces the batch order that
the original run used after that epoch.

That claim has no proof today. Section 6 makes it a gate, not an assumption.

### 4.3 Trigger ladder, not one trigger

Replace the single loss trigger with a ratio trigger on the gradient norm, and allow several
events. Each event writes a capture and a pre-step checkpoint.

```text
trigger when  gradient_norm > 3.0 * median(gradient_norm over the last 200 updates)
```

A ratio self-calibrates as the norm drifts and fires on the **first sustained rise**, which
is the event that matters. Keep at most 8 events, and keep the ring at 32 micro-batches so
each event reaches 32 updates back.

### 4.4 Stop rule

Run from `90 825` to `112 000` updates, which is `21 175` updates, or stop early at the
third gradient event, whichever comes first. Do not run to `221 582`: the blow-up is already
recorded in the existing log.

### 4.5 Paired branches

All four branches start from the same pre-failure checkpoint, with the same restored RNG,
and run `5 000` updates each. They differ in one field only.

| Branch | Change | Question |
| --- | --- | --- |
| Control | none | Does the event reproduce in the instrumented path? |
| Precision | `mixed_precision: fp32` | Does the first abnormal forward or backward move? |
| Step size | peak rate `2e-4` to `5e-5` from the resume point | Does a smaller nonzero step avoid it? |
| QK normalization | new `qk_norm` option, retrofitted weights | If logits grow, does containing them remove the precursor? |

The QK branch needs care. A retrofit changes the function of the model, so it is a separate
artifact, and its loss is not comparable with the control loss. Compare the **trajectory
shape** and the first gradient event, not the absolute loss.

## 5. Instrumentation to add

Two layers: per-update scalars in the trainer, richer per-batch quantities in the analyzer.

### 5.1 Trainer, per update, gated by the existing sidecar directory

| Field | Definition | Cost |
| --- | --- | --- |
| `gradient_norm_tensor/<name>` | L2 norm of each parameter gradient, full name | Only every `50` updates; 12 keys today, about 80 after |
| `update_relative/<block>` | `‖θ_after − θ_before‖ / ‖θ_before‖` per block and global | One `20 M`-element pass, about `0.3 ms` |
| `adamw_eps_fraction` | Share of elements where `sqrt(v̂) < 10 × eps` | Read from optimizer state, no allocation |
| `lr_area` | Running sum of the learning rate in update order | One float |
| `gradient_median_200` | The trigger baseline, so a reader can recompute the trigger | One deque |

The existing sidecar already writes 1.5 GB per `400 000` updates. Per-tensor gradients every
50 updates add about 1.5 GB for the same budget, which is acceptable on NAS and is bounded by
`MJLAB_FLOW_DIAGNOSTICS_MAX_CHECKPOINTS`-style limits.

### 5.2 Trainer, checkpoint policy

| Feature | Setting | Effect |
| --- | --- | --- |
| Rolling pre-failure checkpoints | every `2 000` updates, keep last `3` | A branch point exists before any event |
| Gradient-ratio trigger | factor `3.0`, max `8` events | Captures the first rise, not the blow-up |
| Capture ring | `32` micro-batches | Reaches `32` updates back from the event |
| Forced capture | one dump at a fixed update, default `91 000` | Gives a healthy batch at known weights |
| Stop on event count | `3` events | Bounds the run |

Storage: `3 × 185 MB` rolling plus `8 × 185 MB` trigger plus captures of about `8 × 30 MB`,
so about `2.3 GB` for the run. Report the exact list, do not delete.

### 5.3 Analyzer, per captured batch

Add to `scripts/analyze_flow_capture.py`. All of it is offline work on a saved capture.

| Quantity | Why |
| --- | --- |
| Query and key RMS per layer and per head | The direct test of the attention-logit-growth hypothesis |
| Attention logit quantiles: median, p99, max | A distribution shows growth that a mean hides |
| Attention entropy minimum, p10, median | Collapse is a low-tail event |
| Per-sublayer residual RMS and maximum | Karras-style magnitude drift |
| LayerNorm learned gain: mean, absolute maximum | A growing gain moves everything downstream |
| `adamw_eps_fraction` from the checkpoint state | Tests the Wortsman epsilon interaction |
| `--diff OTHER.json` | Prints healthy-versus-precursor deltas instead of two separate reports |

### 5.4 Run record

`resolved-config.json` does not contain the environment overrides. `MJLAB_FLOW_WIDTH`,
`MJLAB_FLOW_LAYERS`, `MJLAB_FLOW_FFN` and `MJLAB_FLOW_LR_HORIZON_UPDATES` exist only in the
launch environment. Add a `run-env.json` beside it with those four values, the device name,
the torch and CUDA versions, and the sidecar settings. Shape and horizon are recoverable
from the checkpoint identity and the update rows, but a reader should not have to do that.

## 6. Phases, gates and cost

| Phase | Work | Gate | Cost |
| --- | --- | --- | --- |
| 0a | Mine the existing `updates.jsonl` for the first abnormal gradient movement in four arms | A step index and a block are named, with the loss still near baseline | 1 to 2 h, CPU only |
| 0b | Run the analyzer now on `capture-0107069.npz` against `checkpoint-trigger-0107069.pt`, and on the pre-crossing `fn-2048` capture at `76 529` | Either a q/k growth signal already appears, or the attention hypothesis loses its cheapest test | 1 h, CPU only |
| 1 | Implement section 5.1, 5.2 and 5.4, with tests | Unit tests pass; a `200`-update smoke run writes every new field | 3 to 4 h, local |
| 2 | Determinism gate: resume `cap-6x384-w384` from `checkpoint-best.pt` for `200` updates, then diff loss and gradient norm against the stored rows. Dump one healthy capture at a fixed update | Relative difference below `1e-5` on every update, or the branch design is invalid. The dump gives the healthy baseline that no current capture provides | 10 min GPU |
| 3 | Reproduction run, `90 825` to `112 000`, ratio triggers on | A gradient event is captured, and its capture replays with a replay error near zero | 2.5 h GPU |
| 4 | Analyzer extensions and the healthy-versus-precursor diff | One table that names the first internal quantity that moved, or states that none did | 2 to 3 h, CPU |
| 5 | Four paired branches, `5 000` updates each, four GPUs in parallel | Control reproduces the event; each treatment either moves it or does not | 1 h wall clock |

Phase 0 is free and should run first, because it may already name the block and the update,
which decides where phase 3 sets its ring.

## 7. Decisions that need the operator

| Decision | Default in this plan | Alternative |
| --- | --- | --- |
| Reproduce on the small model or the production shape | Small model, then confirm | Production shape costs `1.7 ×` per update and `1.7 ×` storage |
| Change the trainer trigger | Yes, ratio trigger with 8 events | Keep the loss trigger and accept post-onset captures |
| QK normalization branch | Retrofit from the pre-failure checkpoint | Fresh architecture test, which needs its own control |
| Where the run writes | A new group under the existing preview-bulk root | A fresh root, which needs its own contract copy |

## 8. Out of scope

- Deployment-mask validation loss and the sampler-mask mismatch. That is a generation-quality
  question and belongs in the acceptance phase, after a stable branch exists.
- Timestep weighting, velocity parameterization and channel rescaling. These change the
  objective, so they cannot be mixed into a mechanism diagnosis.
- Any closed-loop claim. Falls are policy evidence, and a mechanism fix must still be scored
  on the closed-loop harness.
- A second million-update run.
