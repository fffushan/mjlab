# X2 joint-pose flow: divergence, schedule and seed findings

Checked: 2026-10-24 16:00 (+08).
Scope: the latent-free 68-channel joint-pose flow trainer, twenty training arms run
between 2026-10-22 and 2026-10-24, and what the results establish about
divergence, the learning-rate schedule and the initial draw.
Validation: training metrics from each arm, source inspection, and arithmetic on
the recorded learning-rate schedule. No held-out evaluation and no closed-loop
number is included, because none exists yet.
Related: `x2_joint_pose_flow_training_runs.md` (the first four arms, as a dated
status record), `../plans/x2_joint_pose_seed_sweep.md` (the sweep that was run),
`../plans/x2_joint_pose_flow_closed_loop_handover.md` (the controller loop that is
not built).

## 1. What is established

| # | Finding | Evidence | Confidence |
|---|---|---|---|
| 1 | The plain configuration diverges only for some initial draws | identical batches, noise and schedule; 2 of 4 seeds fail at ~181 k, 2 survive 400 k | measured, n = 4 |
| 2 | The failure point tracks accumulated learning-rate-weighted update, not the step index | breaks at 15.1, 16.3, 17.0 and 32.9 units; totals are 20 (400 k) and 50 (1 M) | measured, n = 5 |
| 3 | A longer budget does not stabilise a seed | the two seeds that failed at 400 k also fail at 1 M, earlier in steps | measured |
| 4 | A longer budget does improve the model that survives it | best 0.0253 against 0.0271 at 400 k, same seed, 6.6 % lower | measured, training loss only |
| 5 | The added final LayerNorm is not protective | both mask cells destabilised with it; both without it stayed clean | measured, n = 1 per cell |
| 6 | A mask of 0.2 does not stabilise either | the 400 k mask arm survived, but the seeded mask cell D was clean without any mask effect and the normed mask cell C diverged | measured |
| 7 | Training loss cannot select a checkpoint | the best value in the 1 M arm is 395,000 updates before the end, and the final loss is 35 % worse | measured |
| 8 | The mechanism of the divergence is not identified | no activation, weight or attention statistic was captured before the first bad update | not established |

## 2. The data and the trainer

| Quantity | Value |
|---|---:|
| Training windows, achieved | 32,004,718 |
| Training windows, reference source | 33,348 |
| Reference sampling weight | 30, giving share 0.030312 |
| Weighted population | 33,005,158 |
| Windows per update | 512 |
| Updates per epoch | 64,463 |
| 400,000 updates | 6.20 epochs |
| 1,000,000 updates | 15.5 epochs |
| Window | 41 rows × 68 channels: root 6-D orientation, 31 `q`, 31 `dq` |
| Model | 6 layers, width 512, 8 heads, 20,030,532 parameters |
| Optimiser | AdamW, weight decay 1e-3, clip 1.0, bf16, EMA 0.75 / 0.9999 |

Two data properties matter and are measured, not assumed:

| Property | Value |
|---|---|
| Largest normalised value | channel 57, left wrist pitch velocity, 25.8σ in the reference source and 29.2σ in the achieved source, because its standard deviation is 0.411 rad/s |
| Windows whose future jumps more than 8σ from the last history row | 53.9 % of reference windows, 4.0 % of achieved windows |

The trainer has **no validation loop** and no best-checkpoint selection. Every
number below is a training loss.

## 3. The arms

Twenty arms were launched. Configurations other than the noted field are batch 512,
peak lr 1e-4, cosine, 10,000-update warmup, mask 0, no final norm, seed 20261023
when seeded.

| Run | Varying fields | Budget | Outcome | Best loss | Peak gradient after warmup |
|---|---|---:|---|---:|---:|
| `400k-20261022` | unseeded | 400 k | **diverged** | 0.0315 @ 153,200 | 2.9e7 |
| `400k-b1024-lr2e4-20261022` | batch 1024, lr 2e-4, unseeded | 400 k | **diverged**, stopped at 388 k | 0.0337 @ 101 k | 1.6e6 |
| `400k-b1024-mask02-20261022` | batch 1024, lr 2e-4, mask 0.2, unseeded | 400 k | **diverged**, stopped at 335 k | 0.0340 @ 91 k | 1.2e6 |
| `400k-b512-mask02-20261022` | mask 0.2, unseeded | 400 k | clean | 0.0386 @ 366 k | 0.49 |
| `400k-base-seeded-b512-20261023` | seed 20261023 | 400 k | clean | **0.0271 @ 375,900** | 0.536 |
| `400k-fn-b512-20261023` | final norm | 400 k | excursion | 0.0302 @ 374,400 | 95.7 @ 195,400 |
| `400k-base-b512-mask02-20261023` | mask 0.2 | 400 k | clean | 0.0384 @ 366,100 | 0.429 |
| `400k-fn-b512-mask02-20261023` | final norm, mask 0.2 | 400 k | **diverged** at 306,000 | 0.0424 @ 249,000 | 9,172 |
| `400k-seed1-b512-20261023` | seed 1 | 400 k, stopped at 332 k | clean | 0.0282 @ 319,900 | 0.53 |
| `400k-seed2-b512-20261023` | seed 2 | 400 k, stopped at 335 k | **diverged** at 180,900 | 0.0330 @ 130,500 | 1.7e8 |
| `400k-seed3-b512-20261023` | seed 3 | 400 k, stopped at 336 k | **diverged** at 181,100 | 0.0319 @ 153,200 | 3.0e9 |
| `400k-seed4-b512-20261023` | seed 4 | stopped at 42 k | clean to that point | 0.0546 @ 36,000 | 0.43 |
| `1m-b512-seed20261023-20261023` | seed 20261023 | 1 M, **completed** | clean | **0.0253 @ 604,500** | 2.221 |
| `1m-b512-seed1-20261024` | seed 1 | 1 M, stopped at 772 k | **diverged** at 371,000 | 0.0283 @ 336,400 | 6.5e6 |
| `1m-b512-seed2-20261024` | seed 2 | 1 M, stopped at 779 k | **diverged** at 178,900 | 0.0328 @ 153,200 | 1.3e11 |
| `1m-b512-seed3-20261024` | seed 3 | 1 M, stopped at 778 k | **diverged** at 171,900 | 0.0324 @ 153,200 | 3.0e9 |

## 4. What the divergence looks like

The first failing arm recorded this sequence:

```text
step 227900  loss 0.0414  gradient_norm       0.38
step 228000  loss 0.2394  gradient_norm      12.82
step 228100  loss 0.1200  gradient_norm      18.09
step 260000  loss 1.6265  gradient_norm  452974.28
step 400000  loss 1.1560  gradient_norm 5851786.50
```

Properties that hold in every failing arm:

1. A single update starts it, then the loss partly recovers, then the gradient
   norm escalates over tens of thousands of updates.
2. The pre-clip gradient norm grows by five to eleven orders of magnitude.
3. No failing arm recovered, at any learning rate, including the near-zero tail of
   the cosine schedule.

**Correction, recorded because an earlier draft of this work claimed otherwise.**
`clip_grad_norm_` returns the norm measured *before* clipping, so a logged value of
1e6 is not the gradient that AdamW receives; every coordinate of the update is
bounded by the clip value of 1.0. A CPU probe in the independent audit clipped a
1e6 gradient and found the optimiser's second moment at 0.001 after the first step.
The mechanism that turns one update into permanent damage is therefore **not**
established. What is established is that the raw gradient becomes enormous, so the
clipped update direction is dominated by a few coordinates.

## 5. The final-norm ablation

Design: four arms, one 2×2 of the two switches, all four built from the same
initial draw, `init_seed=20261023`, so they also share the data order and the noise
draws. The variant adds one `nn.LayerNorm` between the encoder and the output
projection, 2 × width parameters, and converts base checkpoints through
`load_base_state_dict`.

| Cell | Final norm | Mask | Final loss | Best loss | Peak gradient after warmup | Verdict |
|---|---|---:|---:|---:|---:|---|
| B | no | 0.0 | **0.0309** | **0.0271** | 0.536 | clean |
| A | yes | 0.0 | 0.0346 | 0.0302 | 95.7 | excursion at 195,400 |
| D | no | 0.2 | 0.0643 | 0.0384 | 0.429 | clean |
| C | yes | 0.2 | 0.3509 | 0.0424 | 9,172 | **diverged at 306,000** |

Verdict: the hypothesis that a missing final normalisation causes the divergence is
**refuted in this configuration**, and the added norm is associated with worse
stability in both cells. The mask raises training loss by construction, roughly
doubling it, so the two masked arms are comparable only with each other.

## 6. The seed sweep at 400 k

Design: four arms that differ only in `--trainer.init_seed`, at the same 400,000
budget so the learning rate at every step matches the control.

| Seed | Outcome | Onset |
|---|---|---|
| 1 | clean to 332,700 | — |
| 2 | **diverged** | step 180,900 |
| 3 | **diverged** | step 181,100 |
| 4 | stopped at 42,300 for the 1 M batch | — |
| 20261023 (control) | clean to 400,000 | — |

Two of four seeds failed, and the two failures began within 200 updates of each
other in the same batch stream. With n = 4 the 95 % interval on a 50 % rate is
roughly 7 % to 93 %, so the rate is an order of magnitude, not a measurement.

## 7. The 1 M schedule batch

Design: the same three seeds as the sweep, at 1,000,000 updates, so the cosine
learning rate stays high far longer. Seed 20261023 was already running at 1 M.

| Seed | 400 k outcome | 1 M outcome | Break step at 1 M |
|---|---|---|---|
| 2 | diverged at 180,900 | **diverged** | 178,900 |
| 3 | diverged at 181,100 | **diverged** | 171,900 |
| 1 | clean to 332,700 | **diverged** | 371,000 |
| 20261023 | clean to 400,000 | **completed 1,000,000 clean** | — |

So a longer budget did not protect the seeds that failed early, and it also broke
the seed that survived the short budget. It did produce the best model:

| Arm | Best loss | Step of best | Final loss | Elapsed |
|---|---:|---:|---:|---:|
| 400 k control, seed 20261023 | 0.0271 | 375,900 | 0.0309 | 6.75 h |
| 1 M, seed 20261023 | **0.0253** | 604,500 | 0.0342 | 16.82 h |

The 1 M arm spent its last 395,000 updates giving back 35 % of its best value, at
learning rates below 1e-5.

## 8. The accumulated-learning-rate threshold

Compute the integral of the scheduled learning rate from step 0 to the break step.
Units are 1e-4 × 1,000 updates.

| Run | Break step | LR at break | Accumulated to break |
|---|---:|---:|---:|
| 400 k, seed 2 | 180,900 | 5.97e-05 | 15.1 |
| 400 k, seed 3 | 181,100 | 5.96e-05 | 15.1 |
| 1 M, seed 3 | 171,900 | 9.36e-05 | 16.3 |
| 1 M, seed 2 | 178,900 | 9.30e-05 | 17.0 |
| 1 M, seed 1 | 371,000 | 7.06e-05 | 32.9 |
| 1 M, seed 20261023, still clean at step 676,400 | — | 2.41e-05 | 47.3 |
| Whole-budget totals | 400 k: 20.0, 1 M: 50.0 | | |

Reading: each initial draw tolerates some amount of accumulated update before it
destabilises, and the amount differs by a factor of at least three across four
draws. A higher learning rate reaches a given amount sooner, which is why the 1 M
schedule broke the same seeds *earlier in steps*. Lengthening the budget does not
move the threshold; it only spends more of it.

Limits of this reading: n = 5 breaks, the integral is derived from the recorded
schedule rather than measured per update, and nothing here identifies what the
threshold physically is.

## 9. Comparison with the reference papers

| Parameter | BeyondMimic | SCDP | This trainer |
|---|---:|---:|---:|
| Batch | 512 | 128 | 512 and 1024 |
| Peak LR | 1e-4 | 1e-4, decay to 1e-6 | 1e-4 and 2e-4 |
| Weight decay | 1e-3 | 1e-6 | 1e-3 |
| Warmup | 10,000 | 2,000 | 10,000 |
| EMA | 0.75 / 0.9999 | — | 0.75 / 0.9999 |
| Width, heads, layers | 512, 8, 6 | 512, 4, 6 | 512, 8, 6 |
| Horizon | 16 steps | 16 steps, action loss zeroed beyond 8 | **32 rows, all equally weighted** |
| History | 4 | 4, 8, 16 ablated | 8 |

The optimiser settings match BeyondMimic almost exactly, and SCDP trains stably with
a decay a thousand times smaller, so weight decay is not the discriminating
parameter. Our effective horizon is two to four times longer, which the SCDP paper
explicitly discounts, and horizon weighting has not been tested here.

## 10. Checkpoint selection, and the local copies

Every minimum falls between two saved checkpoints, because checkpoints are written
every 10,000 updates. Combined with the late drift of the 1 M arm, the "best
checkpoint" is a proxy for training loss, not a measurement.

| Arm | Best loss | Step of best | Nearest saved checkpoint | Loss at that checkpoint |
|---|---:|---:|---|---:|
| 400 k control, seed 20261023 | 0.0271 | 375,900 | `model_0380000.pt` | — |
| 1 M, seed 20261023 | 0.0253 | 604,500 | `model_0600000.pt` | 0.0265 |
| 1 M, seed 1 | 0.0283 | 336,400 | `model_0340000.pt` | — |

Fetched to this machine, byte-verified against the server:

| Local directory | Contents |
|---|---|
| `data/joint-pose-flow/400k-base-seeded-b512-20261023/` | `model_0370000.pt`, `model_0380000.pt`, metrics, params |
| `data/joint-pose-flow/1m-b512-seed20261023-20261023/` | `model_0600000.pt`, `model_0610000.pt`, metrics, params |

`data/*` is git-ignored, so those copies cannot enter a commit.

## 11. What is not established

| Question | State |
|---|---|
| The mechanism that turns one bad update into permanent damage | unknown; no pre-failure activation, weight or attention statistics were captured |
| Whether any checkpoint is good on held-out data | unmeasured; the trainer has no validation loop and no arm has been evaluated |
| Whether any checkpoint is good in closed loop | one operator test of a 400 k checkpoint was reported as "not great"; no numbers were recorded |
| Whether a lower peak rate prevents the failure | untested; it is the lever the accumulated-update reading supports |
| Whether horizon weighting helps | untested |
| The per-seed failure rate | estimated at 2 of 4, interval roughly 7 % to 93 % |
| Why the mask roughly doubles training loss | expected from the objective, not measured per channel |

## 12. Next experiments, in order

| # | Experiment | Cost | What it decides |
|---|---|---|---|
| 1 | Evaluate the 400 k control, the 1 M reference and 1 M seed 1 on the validation split with `evaluate.py` | one GPU, minutes | whether training loss ranks checkpoints at all, and whether the 1 M gain is real |
| 2 | Add a validation pass and best-checkpoint selection to the trainer | small code change | makes every later arm selectable |
| 3 | One arm at peak lr 5e-5, 1 M updates, seed 20261023 | 16.8 h, one card | whether halving the rate per update pushes every observed threshold past one budget |
| 4 | Horizon weighting, later rows discounted or zeroed beyond 16 | small code change plus one arm | the largest structural difference from both papers |
| 5 | A seed sweep with eight seeds at 400 k | four cards, 6.8 h | bounds the failure rate below 31 % rather than 93 % |

## 13. Artefacts

| Item | Location |
|---|---|
| Runs | `/home/fushan/runs/joint-pose-flow/`, 16 directories, 298 GB after the stops |
| Dataset | `/home/fushan/data/mjlab-latent-free-144/data`, and a byte-identical copy at `/tmp/joint-pose-144/data` |
| Final-norm variant code | `src/mjlab/tasks/tracking/diffusion/model_final_norm.py`, committed |
| Seed-sweep plan | `docs/plans/x2_joint_pose_seed_sweep.md` |
| Controller-loop handover | `docs/plans/x2_joint_pose_flow_closed_loop_handover.md` |

## 14. Limits of this record

- Every loss quoted is a training loss on the window distribution described in
  section 2. Nothing here measures generation quality.
- The comparison with the two papers is a parameter comparison, not a reproduction.
- The accumulated-update threshold is an interpretation of five break points. It is
  consistent with all of them and is not proven.
- The two mask cells and the two norm cells each hold a single seed, so the verdict
  in section 5 is about those draws.
- No hardware, real-time or deployment claim is made anywhere in this lane.
