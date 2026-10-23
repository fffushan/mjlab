# X2 joint-pose flow: initialisation seed sweep

Date: 2026-10-23 (+08).
Status: **prepared, not launched.** The launcher and the reporting script exist and
are syntax-checked. No arm has started.
Scope: whether the divergence of the 2026-10-22 four-arm set was a property of the
configuration or of the initial parameter draw.

## 1. Why this sweep

The 2026-10-22 set was built without a recorded initial draw. Three of four arms
diverged. Repeating one of them with a recorded seed produced a clean run:

| Run | Init | Post-warmup rows above gradient 1.0 | Peak gradient | Best loss |
|---|---|---:|---:|---:|
| `400k-20261022` | unseeded | 1,495 | 2.9e7 | 0.0315 |
| `400k-base-seeded-b512-20261023` (B) | 20261023 | **0** | 0.536 | **0.0275** |

`params/trainer.json` differs between those two runs in 2 of 22 fields:
`init_seed` and `final_norm`. B is the same batch, learning rate, warmup,
schedule, data, mask and generator seeds. B also passed step 228,000, where the
old arm first spiked, by 108,000 steps at the time of writing.

Two arms that share B's initial weights but add a final LayerNorm, A and C, both
show gradient excursions, and C is formally diverged. So the added norm is not
the explanation, and the surviving explanation for the old set is the initial
draw.

One sample cannot measure a rate. This sweep measures it.

## 2. Design

| Item | Value |
|---|---|
| Arms | 4, one per GPU |
| Varying field | `--trainer.init-seed`, values 1, 2, 3, 4 |
| Budget | 400,000 updates, so the learning-rate curve at every step matches B exactly |
| Batch | 512, single microbatch |
| Learning rate | 1e-4, cosine, 10,000-update warmup |
| Weight decay, clip, precision | 1e-3, 1.0, bf16 |
| History dropout | 0.0 |
| Final norm | False, the shared model |
| Data | `/tmp/joint-pose-144/data`, 32,038,066 train windows |
| Checkpoints | every 10,000 updates |
| Expected duration | about 6.6 hours per arm, 4 arms in parallel |

The budget must stay 400,000. A shorter budget re-plans the cosine schedule, so
the learning rate at step 228,000 would differ from B's and the comparison would
be about two things at once. Running a 400,000-update configuration and stopping
after 250,000 is allowed, because each step keeps B's rate.

## 3. Prepared scripts

| File | Purpose |
|---|---|
| `/tmp/jp_launch_seed.sh` on the server | launch one arm: `bash /tmp/jp_launch_seed.sh <gpu> <init_seed> [budget]` |
| `/tmp/jp_seed_report.py` on the server | one row per arm: seed, step, loss, gradient, best loss with its step, and the divergence verdict |

The launcher refuses an existing run directory and uses the same deployed
worktree as the current arms.

Planned tags: `400k-seed1-b512-20261023` through `400k-seed4-b512-20261023`.

## 4. Decision rule

Measured per arm, after the 10,000-update warmup:

1. First step with gradient norm above 100. That is divergence.
2. First step with loss above 5 times the running minimum up to that row.
3. Best loss and the step it occurred.
4. Rows above gradient 1.0, as a graded measure of instability.

The outcome is a count: how many of the seeds diverge. A seed that diverges is
the observation we want, not a failure of the sweep.

## 5. Statistical limit, stated before the run

If all four new seeds plus B are clean, that is 5 samples with zero divergence.
The 95 percent upper bound on the per-seed divergence rate is then about 45
percent, from `1 - 0.05^(1/5)`. That does not support a claim that divergence is
rare. It only shows that the configuration is not deterministically unstable.

| Clean samples | 95 percent upper bound on the failure rate |
|---:|---:|
| 5 | 45 % |
| 8 | 31 % |
| 12 | 22 % |

A second batch of 4 seeds is the cheapest way to reach 8 samples if the first
batch is clean.

## 6. Deliberately excluded

| Excluded | Reason |
|---|---|
| Final-norm cells | one sample each, both with excursions; adding seeds there answers a question this sweep does not ask |
| Batch 1024 at learning rate 2e-4 | batch and learning rate change together, so the old early failures there stay confounded |
| Mask cells | the plain configuration is the one with a matching clean sample |
| Divergence at fixed step with matched learning rate | a different design; not needed for the rate estimate |

## 7. Launch order

The sweep starts when the four current arms release the GPUs, expected around
19:45 (+08). Card assignment: seed 1 to GPU 0, seed 2 to GPU 1, seed 3 to GPU 2,
seed 4 to GPU 3.

A reporting loop must be created at launch. The existing watcher, loop 35, covers
the ablation pair from 2026-10-23 and its glob does not match the seed tags.

## 8. Prohibitions

- Do not change the frozen teacher, the action conversion, the servo path or the
  observation terms.
- Do not compare this sweep with the 1024-wide arms as if learning rate were fixed.
- Do not report a rate from fewer than the stated samples, and always state the
  number of samples next to a bound.
