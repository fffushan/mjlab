# D2 history-dropout series: 0, 20, 50 and 100 percent

Date: 2026-10-13 (+08).
Container: `d-20261004164752-xrjr5`, SSH `10.14.64.48:32221` (four RTX 4090).
Status: training complete for all arms listed. **Held-out loss only.**
No closed-loop, D3, hardware or deployment claim follows from this record.

## Question

Measure the effect of **history state-and-latent dropout** on the D2 state-latent
diffusion student, including the untested **100 percent** extreme.

Dropout replaces both the state and the latent history rows with pure noise
(`k = training_k = 1000`). The current token and all 32 future tokens keep their
own noise levels, so only the history condition changes.

## Arms and artifacts

| Lane | Dropout | Run directory (server) | Updates | Epochs |
| --- | ---: | --- | ---: | ---: |
| 8-teacher | 0% | `logs/diffusion/d2-vgk8-double-20261009/seed0` | 199,000 | 1000 |
| 8-teacher | 20% | `logs/diffusion/d2-vgk8-history-dropout20-20261009/seed0` | 199,000 | 1000 |
| 8-teacher | 50% | `logs/diffusion/d2-vgk8-history-dropout50-20261009/seed0` | 199,000 | 1000 |
| 8-teacher | **100%** | `logs/diffusion/d2-vgk8-history-dropout100-1000ep-seed0-retry1-20261012T185214Z/output` | 199,000 | 1000 |
| 36-teacher | 0% | `logs/diffusion/d2-vgk36-baseline-400k-seed0-retry1-20261012T092934Z/output` | 400,000 | 441 |
| 36-teacher | 20% | `logs/diffusion/d2-vgk36-history-dropout20-400k-seed0-retry1-20261012T092934Z/output` | 400,000 | 441 |
| 36-teacher | **100%** | `logs/diffusion/d2-vgk36-history-dropout100-400k-seed0-20261012T184130Z/output` | 400,000 | 441 |

Every arm uses seed 0, the same model, optimizer, noise schedule, effective batch
512, BF16 and cosine schedule. The 8-teacher arms share dataset
`d1-vgk8-double-20261009` and its contract; the 36-teacher arms share dataset
`d1-vgk36-double-20261012` and its contract.

## Result: best validation loss

The evaluation split is `validation`. Evaluation does **not** apply the dropout
mask, so every arm is scored with full history available and the numbers are
directly comparable across dropout probabilities.

| Lane | 0% | 20% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: |
| 8-teacher, 199k updates | 0.031233 | 0.031739 | 0.032828 | **0.358583** |
| 36-teacher, 400k updates | 0.025154 | 0.025212 | not run | **0.276873** |

Relative to each lane's matched baseline:

| Lane | 20% | 50% | 100% |
| --- | ---: | ---: | ---: |
| 8-teacher | +1.6% | +5.1% | **11.5x** |
| 36-teacher | +0.2% | not run | **11.0x** |

Final training loss (not comparable across dropout probabilities, because the
masked history tokens change the loss composition): 8-teacher 0.011840 / 0.013479
/ 0.014239 / 0.015192; 36-teacher 0.450066 / 0.503182 / 0.603623 for 0 / 20 /
100%.

## Reading

- 20% and 50% history dropout are close to free: the student tolerates hiding
  part of its history.
- **Fully hiding the history costs about 11x the baseline held-out loss in both
  lanes.** The two independent lanes agree closely on the size of the penalty.
- The penalty pattern is consistent with history being a genuinely useful
  condition. It does not show that a 100% arm is unusable for a task where
  history is unavailable; that question needs its own evaluation protocol.

## Verification gates (100% arms)

| Gate | 8-teacher | 36-teacher |
| --- | --- | --- |
| `phase` | `completed` | `completed` |
| `exit.status` | 0 | 0 |
| `global_step` | 199000 | 400000 |
| `planned_optimizer_updates` | 199000 | 400000 |
| `epochs_completed` | 1000 | 441 |
| `history_state_dropout_fraction` | 1.0 | 1.0 |
| Checkpoints plus sidecars | `checkpoint-best/last.pt` + `.yaml` | same |
| Other outputs | `metrics.jsonl`, `train-report.json`, `run-manifest.yaml`, `resolved-config.json`, `README.md` | same |
| Wall time | 22,674.9 s | 42,500.9 s |
| Finished | 2026-10-13 09:10:45 (+08) | 2026-10-13 14:38:29 (+08) |

Smoke gate: 8/8 planned updates on both arms before the full run.

## Source identity

- 36-teacher lane: all three arms used isolated `e71a89ab2` plus permission fix
  `d1cf9c81…`; the 100% arm's snapshot is byte-identical to the running arms.
- 8-teacher lane: the 0/20/50 arms ran from `1ce9da41` (2026-10-09); the 100% arm
  used `e71a89ab2` plus the same permission fix. Between those revisions the only
  change under the diffusion task that could affect training numerics is
  `trainer.py` **+27 lines** for checkpoint sidecars — verified by diff. The
  training step, dropout mask, loss and optimizer code are unchanged.
- Every arm keeps the frozen contract, dataset, statistics and token caches. No
  dataset, cache or statistics file was rewritten.

## Limits

- Held-out offline loss only. No closed-loop rollout, D3 evaluation, hardware
  test or deployment was performed.
- One seed per arm. No confidence interval for the dropout penalty is reported.
- The 36-teacher lane has no 50% arm, so only 0/20/100 are matched there.
- The 100% arm's evaluation sees full history, which is the intended protocol but
  is not the condition the model trained under.
- A first 8-teacher 100% attempt
  (`d2-vgk8-history-dropout100-1000ep-seed0-20261012T184130Z`) stopped at a setup
  input gate because that dataset predates the `exit.status` convention. It
  never trained. The attempt and its logs are preserved.
