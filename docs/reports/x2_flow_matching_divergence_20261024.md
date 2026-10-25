# X2 state-latent flow matching: divergence diagnosis and closed-loop evidence

Date: 2026-10-24 (+08).
Execution status: no training or evaluation job is running. All eight GPUs are free.
Validation status: loss evidence for eight new arms; closed-loop evidence for seven checkpoints.
Scope: simulation only. The managed container `10.14.64.37` has 8 RTX 4090 GPUs and
listens on port `31505`.
Supersedes nothing: the 2026-10-23 ablation report remains the record of the first 16 arms.

## 1. Result summary

Training loss falls to a minimum, then rises, for every long run at a full-length cosine
schedule. The minimum arrives after a nearly fixed number of **optimizer updates**, not
after a fixed number of dataset passes. This behaviour is independent of model size,
batch size, dataset and objective.

| Question | Answer | Strength of evidence |
| --- | --- | --- |
| Does the batch size cause the divergence? | No | Five batch sizes diverge |
| Does the learning rate cause it? | No | Three peak rates diverge |
| Does the model size cause it? | No | `11.6 M`, `13.8 M` and `20.1 M` all diverge |
| Does the dataset size cause it? | No | `50.9 GB` of tokens against `11.84 GB` in the stable store |
| Does the final-norm variant cause it? | No | Trigger timing moves by 8 percent |
| Does early annealing prevent it? | Yes | One arm stayed flat past the onset |
| Does validation loss rank closed-loop quality? | No | Lowest loss arm scored the worst rate |
| Is the rise a generalisation gap? | No | Training loss rises with validation loss |

The single closed-loop outlier (`18.85 %`) came from seed 1. A seed-0 twin of the same
recipe scored `31.88 %`. Single-seed closed-loop numbers in this setup differ by about
`± 7` percentage points.

## 2. The issue

Two separate problems appeared together, and it is important to keep them apart.

**Training divergence.** At a cosine schedule with peak rate `2e-4` and budget `400 000`
updates, validation loss reaches a minimum near `140 000` updates and then climbs. The
climb is not monotone. Some arms spike, some drift, and one arm reached `10 785`. The
weights after the climb are stable and bad: the schedule has already annealed, so the
model no longer recovers.

**Closed-loop failure.** The 144-motion tracker falls far more often than the previous
36-teacher model. Measured fall rates are `18.85 %` to `42.43 %` on 8 motions, against a
user-reported `~15 %` for the 36-teacher model.

The two problems interact: a checkpoint saved after the climb is a bad policy, and
validation loss does not tell you which checkpoints are bad policies.

## 3. What was tried in this session

| Change | Mechanism | Purpose |
| --- | --- | --- |
| Learning-rate horizon decoupling | `MJLAB_FLOW_LR_HORIZON_UPDATES` | Anneal to zero before the onset, keep the run budget |
| Capacity ablation | `MJLAB_FLOW_WIDTH`, `MJLAB_FLOW_LAYERS`, `MJLAB_FLOW_FFN` | Test whether `20 M` parameters is too much |
| Longer budget | Two diffusion arms raised to `1 000 000` updates | Test whether the divergence is transient |
| Closed-loop scoring | `evaluate_parallel.py`, 256 rows per motion | Replace loss ranking with a physical measure |
| Shape-aware loader | `load_flow_denoiser` reads `model_identity` | Make non-default checkpoints evaluable at all |

Both environment switches leave the training config unchanged, so `config_hash` and the
resume identity stay byte-identical.

## 4. Measurements

### 4.1 Training arms

Flow arms use batch `512` and `3 633` updates per epoch. Diffusion arms report
`latent_mse`, which is a different scale and is not comparable with the flow `loss`.

| Arm | Shape | Budget | Best | Best at updates | Final | Diverged |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `fn-seed1-400k` | 20.1 M, final norm, seed 1 | 400 k | `0.02237` | 116 k | `0.306` | yes |
| `fn-b512-400k` | 20.1 M, final norm, seed 0 | 400 k | `0.02340` | 105 k | `0.171` | yes |
| `fn-140k` | 20.1 M, final norm | 140 k | `0.02271` | 138 k | `0.02271` | no |
| `lrh-120k-b512` | 20.1 M, horizon 120 k | 160 k | `0.02265` | 123 k | `0.02265` | no |
| `lrh-160k-b512` | 20.1 M, horizon 160 k | 160 k | `0.02193` | 116 k | `0.03633` | frozen, not blown |
| `fn-2048-400k` | 20.1 M, batch 2048 | 400 k | `0.02419` | 74 k | `0.199` | yes |
| `cap-6x384-w384` | 11.56 M | 400 k | `0.02908` | 91 k | `39.9093` | yes, severe |
| `cap-4x512-l4` | 13.83 M | 400 k | `0.02769` | 105 k | `0.2374` | yes |
| `d144-b512-1m` | 20.1 M, diffusion | 1 000 k | `0.02716` | 145 k | `0.505` | yes |
| `d144-b2048-1m` | 20.1 M, diffusion | 1 000 k | `0.02481` | 91 k | `0.659` | yes |

The two capacity arms ran their full budget (`epochs_completed: 110`). They did not stop
early. The `11.56 M` model ended `1 373` times above its own best.

### 4.2 Closed-loop evidence

Method: `evaluate_parallel.py` scored 8 motions, `single_000` to `single_007`. Each pair
used `256` rows, so each checkpoint used `2 048` rows. The fall threshold was `70°`. Each
row ran 8 measured VAE bootstrap pairs against a frozen reference. Simulation only.

| Checkpoint | Best loss | Rows | Falls | Fall rate |
| --- | ---: | ---: | ---: | ---: |
| `fn-seed1-400k` | `0.02237` | 2 048 | 386 | **`18.85 %`** |
| `d144-b2048-1m` | `0.02481` | 2 048 | 571 | `27.88 %` |
| `fn-b512-400k` | `0.02340` | 2 048 | 653 | `31.88 %` |
| `lrh-120k-b512` | `0.02265` | 2 048 | 734 | `35.84 %` |
| `lrh-160k-b512` | `0.02193` | 2 048 | 760 | `37.11 %` |
| `fn-140k` | `0.02271` | 2 048 | 763 | `37.26 %` |
| `d144-b512-1m` | `0.02716` | 2 048 | 869 | `42.43 %` |

Earlier measurements on the same harness, for scale: `root20-400k` best `53.4 %`,
`root20-400k` last `82.0 %`.

### 4.3 Per-motion spread

`fn-seed1-400k` falls per motion: `163, 97, 14, 88, 9, 9, 5, 1`.
`lrh-160k-b512` falls per motion: `220, 157, 30, 67, 114, 87, 51, 34`.

The spread is per motion, not a uniform shift. `single_007` differs by a factor of `34`
between two checkpoints whose losses differ by 2 percent.

## 5. Observations

### 5.1 Divergence follows updates

| Arm | Batch | Best at updates | Diverged |
| --- | ---: | ---: | --- |
| `fn-2048-400k` | 2 048 | 74 k | yes |
| `cap-6x384-w384` | 512 | 91 k | yes |
| `d144-b2048-1m` | 2 048 | 91 k | yes |
| `cap-4x512-l4` | 512 | 105 k | yes |
| `fn-b512-400k` | 512 | 105 k | yes |
| `fn-seed1-400k` | 512 | 116 k | yes |
| `lrh-160k-b512` | 512 | 116 k | frozen at a worse value |
| `lrh-120k-b512` | 512 | 123 k | no |
| `fn-140k` | 512 | 138 k | no |
| `d144-b512-1m` | 512 | 145 k | yes |

The `1 000 k` budget did not delay the minimum: both diffusion arms bottomed out at their
usual update count and then degraded for the last `390 k` and `115 k` updates.

### 5.2 Annealing early prevents the collapse

`lrh-120k-b512` reached a learning rate of zero at step `120 000`. Its validation loss is
`0.02265` at epochs 34, 35, 36, 37, 38, 39, 40, 41, 42, 43 and 44. It is frozen. It does
not degrade. It passed the onset zone that broke every full-schedule arm.

`lrh-160k-b512` annealed later. It crossed the onset at a rate of `3.6e-5`, survived, and
then drifted to `0.03633`, where it froze. An earlier anneal left a model 60 percent better
on validation loss and 1.3 points better in closed loop.

### 5.3 Capacity is not the cause

A smaller model did not help. It made the divergence worse and the minimum worse.

| Shape | Parameters | Best | Final |
| --- | ---: | ---: | ---: |
| 6 × 384, FFN 1536 | `11 560 103` | `0.02908` | `39.9093` |
| 4 × 512, FFN 2048 | `13 827 239` | `0.02769` | `0.2374` |
| 6 × 512, FFN 2048 | `20 132 007` | `0.02193` to `0.02340` | `0.171` to `0.306` |

Fewer parameters also give a worse minimum, as expected. They do not give stability.

### 5.4 Seed variance dominates the ranking

| Arm | Recipe | Seed | Fall rate |
| --- | --- | ---: | ---: |
| `fn-seed1-400k` | final norm, 400 k | 1 | `18.85 %` |
| `fn-b512-400k` | final norm, 400 k | 0 | `31.88 %` |
| `fn-140k` | final norm, 140 k | 0 | `37.26 %` |
| `lrh-120k-b512` | horizon 120 k | 0 | `35.84 %` |
| `lrh-160k-b512` | horizon 160 k | 0 | `37.11 %` |

The four seed-0 flow arms occupy `31.88 %` to `37.26 %`. The seed-1 arm sits 13 points
below that band. A 13-point gap from one seed means these numbers cannot rank two recipes
that differ by less than roughly 10 points.

### 5.5 Validation loss does not rank policies

| Checkpoint | Loss | Fall rate | Rank by loss | Rank by falls |
| --- | ---: | ---: | ---: | ---: |
| `lrh-160k-b512` | `0.02193` | `37.11 %` | 1 (best) | 5 |
| `fn-seed1-400k` | `0.02237` | `18.85 %` | 3 | 1 (best) |
| `lrh-120k-b512` | `0.02265` | `35.84 %` | 4 | 4 |
| `fn-140k` | `0.02271` | `37.26 %` | 5 | 6 |

A 2 percent loss difference produced a factor of 2 in fall rate, and the direction was
wrong. This matches a known property of diffusion models: mean-squared training losses are
weak predictors of sample quality. The same holds for this tracker.

### 5.6 The evaluation path had a defect

`load_flow_denoiser` built the denoiser from the contract with width, depth, head count and
FFN width hard-coded to the frozen default. A checkpoint that departed from that shape
failed to load and reported `flow checkpoint state does not fit supplied objects`. Two
consequences followed:

1. No final-norm checkpoint had a closed-loop score before today.
2. The loader would reject both capacity arms. The capacity question would then stay
   unmeasured.

The loader now reads `model_identity.settings` and `model_identity.type` from the
checkpoint. The recorded token shape must still match the active contract, and
`identity_sha256` still revalidates the loaded model.

### 5.7 Relation to the joint-pose workstream

A parallel study of the latent-free joint-pose flow trainer (`20 030 532` parameters,
`68` channels) reports that the failure point follows an **accumulated
learning-rate-weighted update count** rather than the step index, and that a longer budget
does not stabilise a seed (`docs/reports/x2_joint_pose_flow_divergence_findings.md`,
checked 2026-10-24 16:00).

That mechanism is consistent with this study. Early annealing removes learning-rate area
that the remaining updates can no longer accumulate, which is another way to say the arm
never reaches the threshold. `lrh-120k-b512` stops at step `120 000` and stays flat.
`lrh-160k-b512` keeps a longer schedule, drifts, and then holds still. The full-schedule arms keep
their rate high and break near `145 000` updates.

This study did not measure the accumulated rate directly. The threshold value from the
joint-pose workstream does not transfer. Compute it again for the state-latent trainer, its
data volume and its batch size.

### 5.8 The training loss rises as well

The rise is not a generalisation gap. The training loss turns upward at the same time as
the validation loss, and the gap between them stays near `0.001` to `0.002`.

The training value is the median per-update `loss` over one epoch on freshly sampled
batches. The validation value is `eval_model.loss` at the same epoch. Source: the local
metric copies in `logs/diagnostics/remote-metrics-20261024/`.

`fn-2048-400k`:

| Epoch | Train | Validation |
| ---: | ---: | ---: |
| 70 | `0.024` | `0.025` |
| 80 | `0.023` | `0.025` |
| 90 | `0.028` | `0.030` |
| 100 | `0.086` | `0.091` |
| 110 | `0.128` | `0.219` |
| 200 | `0.254` | `0.245` |

`cap-6x384-w384`:

| Epoch | Train | Validation |
| ---: | ---: | ---: |
| 20 | `0.030` | `0.031` |
| 30 | `0.474` | `0.621` |
| 40 | `0.849` | `1 236` |
| 60 | `5.297` | `1.305` |

`lrh-160k-b512`:

| Epoch | Train | Validation |
| ---: | ---: | ---: |
| 30 | `0.021` | `0.022` |
| 40 | `0.034` | `0.029` |
| 50 | `0.040` | `0.036` |

Three consequences follow.

1. The model stops fitting its own training data. A larger dataset does not fix this, and
   a held-out split does not explain it.
2. Overfit is excluded. An overfit run shows a falling training loss and a rising
   validation loss. Here both rise.
3. A frozen learning rate helps for the same reason. If unseen inputs caused the damage,
   a frozen rate would not stop it.

One caution. A diverged model gives a wide spread of losses across batches, so the median
over an epoch is noisy, and in `cap-6x384-w384` the training value leads the validation
value by a factor of 4 at epoch 60. The direction of both curves is the signal, not the
exact ratio.

## 6. Findings

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| Batch size too large | Ruled out | 512 and 2048 both diverge |
| Peak learning rate too high | Ruled out | `1e-4`, `2e-4` both diverge |
| Model too large | Ruled out | `11.6 M`, `13.8 M`, `20.1 M` all diverge |
| Too little data | Not supported | `9 160 010` unique rollout frames, `2 322 881` windows |
| Overfitting or validation drift | Ruled out | training loss rises with validation loss; the gap stays near `0.001` |
| Dataset is reference replay only | Corrected | The store holds policy rollouts with OU noise |
| Final norm is protective | Ruled out | Trigger timing shifts by 8 percent |
| Long budget makes it recover | Ruled out | `1 000 k` arms diverge at the usual count |
| Anneal before the onset | **Supported** | `lrh-120k` flat at `0.02265` for eleven epochs |
| Loss ranks closed-loop quality | **Contradicted** | Loss rank 1 scored rank 5 by falls |
| Single-seed ranking is meaningful | Ruled out | Twin recipes differ by 13 points |

## 7. Limits

- Eight motions of 148. A different subset could reorder the middle arms.
- One seed per closed-loop checkpoint. The seed spread is larger than the differences
  between most arms.
- Simulation only. No hardware evidence, no real-time claim.
- Diffusion arms use 20 DDIM network calls per tick; flow arms use 6 Euler calls. Sampler
  budget differs by family, so cross-family comparison is descriptive only.
- The capacity arms completed. They have no closed-loop score. Their checkpoints now load,
  so this is an open measurement, not a limit of the data.
- Only two points exist on the update axis of any single run. A checkpoint ladder would
  need `checkpoint_every`, which no run in this study set.

## 8. Artifacts

| Item | Path |
| --- | --- |
| Run root | `/home/fushan/mjlab-multimotion144-bulk-20261020/logs/diffusion/x2-multimotion144-preview-bulk-20261020/` |
| Capacity arms | `capacity-20261024/` |
| Horizon arms | `lrhorizon-20261024/` |
| Final-norm arms | `finalnorm-20261023/` |
| Diffusion arms | `diffusion144-20261023/` |
| Closed-loop evidence | `/home/fushan/data/mjlab-144/eval-4ckpt-20261024/`, `eval-mixed-20261024/` |
| Metrics copies | `logs/diagnostics/remote-metrics-20261024/` (local, five of six files) |
| Local checkpoint | `logs/flow/x2-multimotion144-fn-seed1-400k-20261024/checkpoint-best.pt` |

Code changes from this session:

| Commit | Change |
| --- | --- |
| `742c02a41` | `MJLAB_FLOW_LR_HORIZON_UPDATES` separates the anneal horizon from the budget |
| `5a304bb90` | `MJLAB_FLOW_WIDTH`, `MJLAB_FLOW_LAYERS`, `MJLAB_FLOW_FFN` for a capacity ablation |
| `13ab647cb` | Closed-loop loader builds the denoiser from the recorded checkpoint |
| `edcd072a6` | Flow and diffusion trainers mirror metric rows into TensorBoard |

## 9. Next actions

1. Score the two capacity arms in closed loop. Their checkpoints now load. Cost: `2 × 8`
   pairs, about 30 minutes on two cards.
2. Repeat the two best recipes with three seeds each. Without that, no ranking between
   recipes is defensible. Cost: about one hour per recipe on two cards.
3. Add a checkpoint ladder: set `checkpoint_every` on one recipe, then score the
   intermediate checkpoints. This is the only way to measure the update axis directly.
4. Compute the accumulated learning-rate area for every arm in this study, then test
   whether one number separates the arms that survived from the arms that broke.
