# X2 joint-pose flow trainer: four-arm training status

Kind: experiment record and diagnosis.
Checked: 2026-10-23 11:11 (+08).
Scope: the latent-free 68-channel flow trainer, four 400,000-update arms on one 4-card
host, the training failure that three arms showed, and the comparison with two reference
papers.

This record contains evidence only. It does not authorize a data purge, a diverged-run
restart, DDP work, or a commit to the main checkout.

## 1. Implementation status

| Item | State |
|---|---|
| Trainer `src/mjlab/tasks/tracking/joint_pose/{config,dataset,model,runner,evaluate}.py` | implemented, committed, 21 tests pass |
| CLI entry point `train-flow` with a mandatory `--max-steps` | implemented, committed |
| Reference window source: 39,893 windows at sampling weight 30 | built; train share 0.030312 |
| Dataset preparation tools | committed, with README |
| Trackability survey, failure ledger, mode analysis | committed, with reports |
| DDP | not implemented |
| Validation loop and best-checkpoint selection | not implemented |
| Horizon weighting, input clip at ±10σ, extreme-gradient skip | not implemented |

Commits on branch `fxy/test/flow-matching-baseline`: `bde2746e4` (trainer), `cdb1349f6`
(preparation tools), `e8d636e4e` (ascending microbatch gather). Commits
`32f6979ff`, `74f8682ba`, `916c66ae3`, `151c1d55f` above them come from the separate D1
work and are not part of this record.

## 2. Data status

| Quantity | Value |
|---|---:|
| Train windows | 32,038,066 (32,004,718 achieved plus 33,348 reference) |
| Validation windows | 3,021,163 |
| Test windows | 3,258,983 |
| Cache on local NVMe | 426.9 GB, content verified identical to the NAS copy |
| Data removed | none; the failure ledger holds pass 1 only |

Two measured properties of the cache matter for training stability.

| Property | Result |
|---|---|
| Largest normalised magnitude, channel 57 | 25.8σ in the reference source, 29.2σ in the achieved source |
| Channel 57 identity | left wrist pitch joint velocity; standard deviation 0.411 rad/s |
| Windows whose future jumps more than 8σ from the last history row | 53.9 % of reference windows, 4.0 % of achieved windows |

The tail in channel 57 comes from normalisation, not from corrupt data. A 12 rad/s wrist
movement becomes 29σ because that channel has a small standard deviation. All values are
finite. No NaN and no infinity occurs in either source.

## 3. Arms and results

All arms use the same dataset, warmup of 10,000 updates, cosine schedule, EMA power 0.75
and EMA maximum 0.9999, width 512, 8 heads and 6 layers.

| Arm | Host GPU | Batch | Peak LR | History dropout | Result | Final loss | Final gradient norm |
|---|---|---:|---:|---:|---|---:|---:|
| `400k-20261022` | 0 | 512 | 1e-4 | 0.0 | diverged | 1.1560 | 5,851,786 |
| `400k-b1024-lr2e4-20261022` | 1 | 1024 | 2e-4 | 0.0 | diverged, running at step 381,400 | 0.6645 | 179,732 |
| `400k-b512-mask02-20261022` | 2 | 512 | 1e-4 | 0.2 | stable to budget | 0.0643 | 0.08 |
| `400k-b1024-mask02-20261022` | 3 | 1024 | 2e-4 | 0.2 | diverged, running at step 328,500 | 0.6206 | 6,429 |

Run directories are under `/home/fushan/runs/joint-pose-flow/`. Each holds
`metrics.jsonl`, `params/{trainer.json,dataset.json,pointer.txt}` and periodic
`model_XXXXXXX.pt` checkpoints. Checkpoint counts are 40, 36, 40 and 30. The four runs use
60 GB on the shared NAS.

Two arms finished in about 23,600 s each, which is 6 h 34 min. The stable arm produced 40
checkpoints of 320,622,814 B.

## 4. Failure signature

The healthy arm had a loss minimum of 0.0315 at step 153,200. A single batch then changed
the gradient norm by two orders of magnitude.

```text
step 227900  loss 0.0414  gradient_norm      0.38
step 228000  loss 0.2394  gradient_norm     12.82
step 228100  loss 0.1200  gradient_norm     18.09
step 260000  loss 1.6265  gradient_norm 452974.28
step 400000  loss 1.1560  gradient_norm 5851786.50
```

The arm recovered partly near step 280,000, then degraded again. The gradient norm grew by
seven orders of magnitude across training.

Gradient clipping at 1.0 bounds each update, but it cannot protect the AdamW second moment.
One gradient of magnitude 10⁶ inflates that moment by about 10¹² in the affected
coordinates. Later steps in those coordinates become saturated steps at the full learning
rate. This mechanism explains why the loss never returned to 0.03.

The magnitude of the failure does not come from the data. A target of 29σ against a
prediction of 0 produces a loss of 0.37 after averaging and a per-element gradient near
0.03. A gradient of 10⁶ therefore requires the model output itself to grow, not the target.

The stable arm never exceeded a gradient norm of 2.79 across 400,000 updates. It has the
same batch, learning rate, warmup and budget as the diverged arm at GPU 0. The only
difference is history dropout at probability 0.2. One run cannot separate protection by the
mask from luck.

## 5. Comparison with the reference papers

### 5.1 BeyondMimic

| Parameter | BeyondMimic | This trainer |
|---|---|---|
| Batch size | 512 | 512 and 1024 |
| Peak learning rate | 1e-4 | 1e-4 and 2e-4 |
| Weight decay | 0.001 | 0.001 |
| Schedule | cosine, 10,000 warmup steps | cosine, 10,000 warmup updates |
| EMA power and maximum | 0.75 and 0.9999 | 0.75 and 0.9999 |
| Embedding dimension | 512 | 512 |
| Attention heads | 8 | 8 |
| Transformer layers | 6 | 6 |
| Horizon | 16 | 32 generated rows |
| Observation history | 4 | 8 |

The optimiser and architecture parameters agree. The horizon and the history length do not.

### 5.2 SCDP

| Parameter | SCDP | This trainer |
|---|---|---|
| Batch size | 128 | 512 and 1024 |
| Gradient updates | 1,000,000 | 400,000 |
| Peak learning rate | 1e-4, decay to 1e-6 | 1e-4, decay to about 0 |
| Warmup | 2,000 | 10,000 |
| Weight decay | 1e-6 | 1e-3 |
| Horizon | 16 steps, action loss zeroed beyond 8 steps | 32 rows, all equally weighted |
| History length ablated | 4, 8, 16 | 8 |

Weight decay is not the cause of the failure. SCDP uses a decay one thousand times smaller
than this trainer and trains stably. Both reference papers limit the effective horizon.
SCDP states the reason: action losses are zeroed beyond 8 steps for near-term accuracy.
This trainer weights all 32 generated rows equally.

Neither paper states its noise-level sampling distribution. This trainer samples uniformly
over 1,000 levels, including levels near pure noise, where the target is not predictable
from the history.

## 6. Findings

| Finding | Evidence |
|---|---|
| The optimiser and architecture parameters match BeyondMimic | Section 5.1 |
| Weight decay is not the discriminating parameter | Section 5.2 |
| The effective horizon is the largest structural difference | 32 equally weighted rows against 16, or 16 with rows beyond 8 zeroed |
| The data is not corrupt | no non-finite values; the largest magnitude is 29σ |
| The data is badly scaled in one channel | section 2; standard deviation 0.411 rad/s |
| The reference source is four times harder than the achieved source | 53.9 % against 4.0 % of windows above 8σ jump |
| The failure is an optimiser instability, not a data defect | section 4 |
| The masked arm is the only arm stable to the full budget | section 3 |
| The failure is invisible in summary statistics | only the per-100-update rows show it |

## 7. Unknowns

| Unknown | Consequence |
|---|---|
| Which checkpoint is best in any arm | no validation loss exists for any checkpoint |
| Whether history dropout protects, or the arm was fortunate | one run per setting |
| The noise-level sampling of both papers | the difference cannot be adopted without the source |
| Whether channel 57 triggers the divergence | correlation only; not proven |
| Whether the 1024 arms would recover | both are still running and still degraded |

## 8. Test directions

| # | Direction | Change required | Cost | Expected value |
|---|---|---|---:|---|
| 1 | Horizon weighting: zero or discount rows beyond 16 | loss in the runner, one config flag | hours | highest; both papers do a version of this |
| 2 | Clip normalised inputs and targets at ±10σ, with a counter | dataset loader | hours | removes the 26–29σ entries without deleting data |
| 3 | Skip an update when the raw gradient norm exceeds 100, and log it | runner | hours | stops one batch from corrupting AdamW state |
| 4 | Validation pass every 5,000 updates, keep the best checkpoint | runner | about half a day | required for every later comparison |
| 5 | Re-run batch 512, LR 1e-4, mask 0 with directions 1 to 3 applied | configuration only | 6 h 34 min | decides whether the fix is sufficient |
| 6 | Arm at LR 5e-5 with direction 4 applied | configuration | 6 h 34 min | safety margin for non-mocap data |
| 7 | Arms at history dropout 0.2 and 0.4 | configuration | 13 h | separates the mask effect from luck |
| 8 | DDP across four cards | about 300 lines | one day plus | 3.5 times throughput; only after stability is understood |

Directions 1 to 4 come before any new long run. Direction 4 has priority inside that set,
because no arm can be compared without a held-out loss.

## 9. Operational status

Checked 2026-10-23 11:11 (+08).

| Item | State |
|---|---|
| Finished arms | 2 of 4, diverged at GPU 0, stable at GPU 2 |
| Running arms | 2, both diverged, GPU 1 at 81 % and GPU 3 at 77 % |
| Idle GPUs | 0 and 2 |
| Shared NAS free space | 977 TB |
| Watcher | Monitor 12, reconnecting loop, 15 min cadence |

## 10. Next action

1. Decide whether to stop the two diverged 1024-update arms. They hold GPU 1 and GPU 3.
2. Authorize directions 1 to 4.
3. Re-run after direction 4 exists.

Authorization is required for items 1 and 2. A purge of the failure ledger stays
unauthorized.
