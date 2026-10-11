# VAE distillation optimizer batching — measured effect of accumulation

Status: **measured; one configuration pair, two hosts, one seed cohort.** This is
an operations note. It does not change the VAE objective, the teacher contract,
the replay policy, or the sample budget.

## 1. Finding

For the X2 VAE distillation trainer, set the **minibatch size equal to the
intended effective batch** and set `accumulation_steps = 1`.

Measured on the 36-teacher cohort with an otherwise identical budget:

| Setting | Optimizer phase per iteration | Whole iteration |
| --- | --- | --- |
| `minibatch 4096`, `accumulation 8` | 8.40 s | 16.67–17.23 s |
| `minibatch 32768`, `accumulation 1` | 1.72 s | 7.43–7.75 s |

The optimizer phase fell by about **4.9x**. The whole iteration fell by about
**2.3x**.

Both arms kept the same effective batch, the same number of optimizer updates,
and the same number of gradient draws. Only the number of sequential micro-batch
forward/backward passes per optimizer step changed.

## 2. What was measured

### Cohort and budget (both arms)

| Field | Value |
| --- | --- |
| Teachers | 36 (`vgk000`–`vgk034` minus `vgk008`, `vgk012`, `vgk032`, plus `sh001`–`sh004`) |
| Manifest | `configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v3.yaml` |
| `num_envs` | 36864 (36 × 1024) |
| Collection workers | 3, plus one dedicated trainer card |
| `collection_steps` | 32 |
| `draw_to_collection_ratio` | 0.5 |
| Fresh rows per iteration | 1179648 |
| Gradient draws per iteration | 589824 |
| **Effective batch size** | **32768 in both arms** |
| Optimizer updates per iteration | 18 in both arms |
| `replay_retention_iterations` | 2 → `replay_capacity` 2359296 |
| Iterations | 10000 |

The accumulation-1 arm was checked against these values in
`/tmp/vae-36t-acc1-smoke/train-report.json` on the container:
`minibatch_size 32768`, `accumulation_steps 1`, `effective_batch_size 32768`,
`updates_per_iteration 18`, `training_draws_per_iteration 589824`.

### Runs

| Arm | Host | Run | Evidence |
| --- | --- | --- | --- |
| accumulation 8 | bare-metal 8-card host `10.14.64.37:31505` | `runs/vae-vgk36-1024-env-per-teacher-4gpu-dedicated-trainer-20261011T081238Z` | `run.log` preserved at `logs/diagnostics/vae-stop-20261011/` |
| accumulation 1 | 4-GPU container `10.14.64.48:32221` | `runs/vae-vgk36-1024-env-per-teacher-4gpu-container-acc1-20261011T085632Z` | running at the time of writing |

Comparison run, accumulation 8, different cohort and environment count:
`runs/vae-vgk32-1024-env-per-teacher-4gpu-dedicated-trainer-20261011T045435Z`,
14.05–14.11 s/iter at `num_envs 32766`.

### Iteration rate

Accumulation 8, 36 teachers:

```text
iter 50  16.67 s/iter
iter 100 17.23 s/iter
iter 150 16.68 s/iter
iter 200 16.92 s/iter
```

Accumulation 1, 36 teachers:

```text
iter 50  7.75 s/iter
iter 200 7.48 s/iter
iter 300 7.44 s/iter
iter 350 7.43 s/iter
```

### Phase durations

Method: sample `nvidia-smi --query-gpu=utilization.gpu` every 0.5 s for 100–120
samples. Classify each sample as worker-busy (any of the three collection cards
above 5%) or trainer-busy (the trainer card above 5%). Report the mean length of
consecutive same-phase samples.

| Arm | Worker burst | Trainer burst | Worker runs / trainer runs per window |
| --- | --- | --- | --- |
| accumulation 8 | 3.10 s | 8.40 s | 5 / 5 |
| accumulation 1 | 3.90 s | 1.72 s | 10 / 9 |

The phases never overlap. Both arms show 0% of samples with worker and trainer
busy at the same time.

The bursts do not account for the whole iteration. A host-only portion remains
between phases, for example replay insertion, normalizer updates, and reporting.
The 0.5 s sampling is coarse, so treat these burst lengths as approximate.

### Memory

The trainer card held about `18981 MiB` in **both** arms. The replay ring
dominates that allocation, so this change did not increase trainer memory. The
collection cards held 11.5–12.3 GiB.

### Loss check

The recipe should be unchanged, and the observed optimisation values agree at
the first comparable points.

| Iteration | Accumulation 8 | Accumulation 1 |
| --- | --- | --- |
| 50 | loss 1.8967, disagreement 0.1599 | loss 1.8756, disagreement 0.1588 |
| 100 | loss 1.4484, disagreement 0.1392 | loss 1.4079, disagreement 0.1363 |
| 200 | loss 1.1188, disagreement 0.1220 | loss 1.1485, disagreement 0.1220 |

This is a consistency check, not a quality comparison. One seed pair cannot
separate two training arms.

## 3. Why it happens

An optimizer step with `accumulation_steps = A` runs `A` sequential
micro-batch forward/backward passes, then one optimizer update. Each pass is a
separate set of kernel launches and host round trips.

The VAE is small: MLPs of `[2048, 1024, 512]` with a 32-dimensional latent. The
arithmetic per pass is therefore cheap, and per-pass overhead dominates. A
larger minibatch performs the same total arithmetic in one pass instead of `A`
passes.

Evidence for the overhead explanation: the trainer card is only about 20% busy
while the learning phase runs. The phase is launch-bound and host-bound, not
compute-bound.

## 4. How to apply this

1. Choose the effective batch size from the training recipe.
2. Set `--minibatch-size` to that value and `--accumulation-steps 1`.
3. Keep the derived budget consistent. `--draw-to-collection-ratio` derives
   `updates = ceil(r * num_envs * collection_steps / effective_batch)` and
   `effective_batch = minibatch_size * accumulation_steps`. Holding the
   effective batch constant therefore holds draws and updates constant.
4. Check VRAM before a long run. A larger minibatch raises activation memory
   only; the replay ring is unaffected.
5. Run a bounded smoke first, and read the resolved
   `sample_budget.resolved.effective_batch_size`, `updates_per_iteration`, and
   `training_draws_per_iteration` from the report.

Relevant note for reversed cases: if VRAM does not allow the full effective
batch, accumulation remains the correct tool. This note says only that
accumulation should not be used when the memory is available.

## 5. Limits

- The two arms ran on **different hosts**. The container has a 56-CPU cgroup
  quota; the bare-metal host has 128 CPUs. Its collection phase is 26% slower
  (3.90 s versus 3.10 s), which is expected for a CPU-bound collector. On equal
  hardware the whole-iteration gain is therefore larger than the measured 2.3x.
- The optimizer-phase comparison, 8.40 s to 1.72 s, is the direct measurement
  of the change and does not depend on that host difference.
- One cohort, one seed, one effective batch size. No other effective batch size
  was tested.
- `bf16` mixed precision, `torch.compile`, fused optimizers, and pinned-memory
  accumulation were not tested.
- Collection and learning still run strictly in sequence. Overlapping them is a
  separate change with its own policy-lag and resume-identity consequences.
- These are wall-clock measurements. They say nothing about model quality.

## 6. Consequences for checkpoints and resume

The minibatch and accumulation split is part of the stored training settings.
Equal effective batch alone does not permit a different split on resume. See the
sample-budget section of `docs/source/x2_tennis_distillation.rst`. A run that
changes the split is a new run, not a continuation.

## 7. Open questions

- Does the same effect hold at other effective batch sizes, for example 8192 or
  65536?
- How much of the remaining per-iteration cost is host-side work between phases,
  and can it overlap with the collection?
- Would a larger minibatch change the optimisation trajectory at a fixed number
  of updates? The loss check above is too coarse to answer this.

## 8. Sources

- Runbook: `docs/source/x2_tennis_distillation.rst` (sample-budget section).
- Multi-GPU plan and its measurement records:
  `docs/plans/beyondmimic_vae_multi_gpu_training_implementation.md`.
- Preserved logs of the stopped accumulation-8 runs:
  `logs/diagnostics/vae-stop-20261011/`.
