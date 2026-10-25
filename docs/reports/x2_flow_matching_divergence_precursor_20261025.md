# X2 flow divergence: what the saved evidence shows before any new run

Date: 2026-10-25 (+08).
Execution status: two offline analyses completed. No training job started.
Validation status: measured on saved logs and saved checkpoints; no new training.
Scope: simulation only. CPU work on `10.14.64.37`, GPU analysis on `10.14.64.48` (4-card).
Precedes: the instrumented reproduction in `../plans/x2_flow_divergence_prefailure_capture.md`.

## 1. Result summary

The saved evidence names the signal that leads the loss, and it is not the gradient.

| Quantity | Behaviour across five failing arms | Behaviour across two survivor arms |
| --- | --- | --- |
| Gradient norm | Rises at the same time as the loss, within `864` updates | Never exceeds `3 ×` its own median |
| Attention logit scale in one layer | `700` to `36 000` at the loss minimum, then grows by `4` to `6` orders | `15` to `116`, except one arm that grew and was then frozen |
| Attention entropy in that layer | Falls to `0.004` and then to `0.000` | Stays above `0.15` |

One layer per run runs away. The layer index differs between runs (`2`, `3`, `4`).
The logit scale of that layer is already `100` to `1 000` times the survivor level at the
**loss minimum**, which means the checkpoint with the best validation loss is already on the
runaway path.

## 2. What was run

| Step | Work | Where | Result |
| --- | --- | --- | --- |
| 0a | Mined eight `updates.jsonl` files for the first abnormal gradient movement | CPU, `10.14.64.37` | Table 1 |
| 0b | Analyzed four saved captures against their own pre-step checkpoints | GPU 0, `10.14.64.48` | Tables 2 and 3 |
| 0b control | Ran the two survivor arms on one fixed batch | Same | Tables 2 and 3 |

## 3. Gradient and loss move together

`grad_5x` is the first update where the gradient norm exceeds `5 ×` its own 200-update
rolling median. `area` is the sum of the learning rate up to that update.

| Arm | Model | Updates | `grad_5x` | `area` there | Loss `1.5 ×` | Gap | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fn-2048-400k` | 20.1 M, batch 2048 | 304 515 | 76 527 | 13.99 | 76 527 | **0** | diverged |
| `fn-b512-400k` | 20.1 M, batch 512 | 400 000 | 106 158 | 19.30 | 106 157 | **−1** | diverged |
| `cap-6x384` | 11.6 M | 400 000 | 102 907 | 18.74 | 102 043 | +864 | diverged |
| `cap-4x512` | 13.8 M | 400 000 | 121 251 | 21.82 | 121 535 | −284 | diverged |
| `fresh-b512` | 20.1 M | 164 145 | 152 250 | 26.54 | 152 251 | −1 | diverged |
| `lrh-120k-b512` | 20.1 M, anneal 120 k | 160 000 | **none** | 12.00 | none | — | flat at `0.02265` |
| `lrh-160k-b512` | 20.1 M, anneal 160 k | 200 000 | **none** | 16.00 | none | — | frozen at `0.03633` |

Two consequences.

1. The gradient is a **co-indicator**, not a precursor. In two arms the two events land on
   the same update. A trigger on the gradient therefore fires at the event, not before it.
2. The accumulated learning rate does **not** separate the arms. The lowest failing value is
   `13.99`; one survivor stopped at `16.00`. The ordering is not monotone across the five
   failures, so no single threshold holds.

## 4. Attention logits carry the signal

Mean absolute attention logit per layer, from `_attention_stats` on 8 windows of one batch.
The 0b control evaluates all checkpoints on the same `fn-2048` batch at update `76 529`.

| Checkpoint | Step | Loss | L0 | L1 | L2 | L3 | L4 | L5 | Worst entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lrh-120k` best | 156 262 | `0.0214` | 2.8 | 5.1 | 14.5 | 43.5 | 96.7 | 115.8 | `0.224` |
| `lrh-160k` best | 119 922 | `0.0208` | 4.3 | 7.2 | 21.9 | 113.1 | **35 269** | 364.2 | `0.159` |
| `lrh-160k` last | 200 000 | `0.0355` | 4.7 | 9.5 | 32.7 | 239.1 | **2 265 832** | 617.4 | `0.030` |
| `fn-2048` best | 71 811 | `0.0230` | 2.2 | 255.1 | **737.0** | 99.2 | 55.6 | 2.4 | `0.067` |
| `fn-2048` trigger | 76 529 | `0.2260` | 2.6 | 1 100.0 | **16 098.8** | 127.8 | 94.7 | 2.6 | `0.051` |
| `fn-2048` trigger | 91 478 | `0.5749` | 4.2 | 379.5 | **40 888 288** | 93.7 | 84.2 | 2.1 | `0.000` |
| `cap-6x384` best | 90 850 | `0.0281` | 3.3 | 7.7 | 69.6 | **1 308.0** | 42.9 | 733.3 | `0.609` |
| `cap-6x384` trigger | 107 069 | `6.2836` | 4.6 | 14.4 | 66.2 | **714 634.3** | 18.6 | 359.0 | `0.004` |
| `cap-6x384` last | 400 000 | `39.7460` | 13.3 | 148.2 | 167.3 | **81 678 520.0** | 16.1 | 207.5 | `0.000` |

Read the bold column.

1. In `fn-2048` the runaway layer is `2`. It is `737` at the loss minimum, `16 098` at the
   loss spike, and `40.9` million at update `91 478` while the loss is still only `0.575`.
2. In `cap-6x384` the runaway layer is `3`. It is `1 308` at the loss minimum, `714 634` at
   the first capture of the ramp, and `81.7` million at the end.
3. The survivor `lrh-120k` never exceeds `116` in any layer and keeps every entropy above
   `0.22`.
4. `lrh-160k` grew its layer `4` to `35 269` and then to `2.27` million, but its loss rose
   only from `0.021` to `0.036`. Logit growth alone does not immediately destroy the loss.
   The anneal then froze the arm, so the growth stopped.

The scale also grows with depth in the healthy arm: `2.8, 5.1, 14.5, 43.5, 96.7, 115.8`.
Later layers start closer to the runaway, which is what an architecture without query/key
normalization permits.

## 5. Interpretation, and what it does not establish

| Claim | Status |
| --- | --- |
| Attention logit growth precedes the loss rise by `5 000` to `16 000` updates | measured in two arms |
| Attention entropy collapses in the same layer | measured, `0.004` and `0.000` |
| The gradient rise is simultaneous with the loss rise, not earlier | measured in five arms |
| A single accumulated-learning-rate threshold exists | contradicted by one survivor at `16.00` |
| The loss minimum is already abnormal in these arms | measured: `737` and `1 308` against survivor `15` to `116` |
| Attention logits cause the divergence | **not established** |
| Query/key normalization would prevent it | **not tested** |
| The same layer is responsible in every run | contradicted: layers `2`, `3` and `4` |

The causal step has no measurement yet. The logit scale and the loss both depend on the same
weights, so a third quantity could drive both. The reproduction plan adds three measurements
that can separate them: query and key RMS per head, per-tensor gradients on the attention
projections of the runaway layer, and the relative parameter update of those tensors.

## 6. Practical consequences

1. **Model selection by validation loss cannot catch this.** The best-loss checkpoint of both
   failing arms already carries a runaway layer. A guardrail must read a weight statistic,
   not a loss.
2. **The gradient trigger is nearly useless as a precursor.** It fires within a few updates
   of the loss. A capture cadence on the logit scale is the instrument that sees the ramp.
3. **A quiet arm is not a healthy arm.** `lrh-160k` never crossed the gradient threshold and
   still grew a layer to `2.27` million logits. Its loss stayed low because the anneal froze
   it early.

## 7. Reproducibility notes

| Item | Value |
| --- | --- |
| 0a script | `logs/diagnostics/flow-divergence-phase0/phase0a.py` |
| 0a outputs | `logs/diagnostics/flow-divergence-phase0/phase0a/*.json` |
| 0b scripts | `phase0b_prep.py`, `phase0b_finalnorm.sh`, `phase0b_controls.sh` in the same directory |
| 0b outputs | `logs/diagnostics/flow-divergence-phase0/phase0b/*.json` |
| Portable bundles | `/home/fushan/data/mjlab-144/phase0b/` on the shared NAS |
| Code fix required | `e0b1fa2d5`, the analyzer must read the model type from the checkpoint |
| Replay check | The same-update rows reproduce within `5%`; the other rows in a 16-update ring belong to earlier weights |

Two limits of the measurement. The activation pass sees 8 windows of one batch, so the
statistics are per-batch, not per-run. The logit scale appears as a mean and a maximum. The
report has no quantiles, so a heavy tail cannot be separated from a broad shift.
