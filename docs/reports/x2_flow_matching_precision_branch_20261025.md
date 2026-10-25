# X2 flow divergence: pure fp32 prevents the runaway

Date: 2026-10-25 (+08).
Execution status: both branches stopped at the planned target; no job is running.
Validation status: 15 000 updates per arm, one seed, one model shape, training metrics and one checkpoint analysis.
Scope: training only. No simulator, no closed-loop score, no hardware claim.
Container: `10.14.64.48` (4-card), GPUs 0 and 1.

## 1. Result

Pure fp32 arithmetic stopped the divergence. The bf16 control from the same checkpoint
reproduced it exactly.

| Arm | Precision | Updates | Loss first | Loss last | Loss max | Gradient max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fp32` | fp32 | 15 055 | `0.0286` | **`0.0261`** | `0.0289` | `0.188` |
| `bf16-control` | bf16 | 15 025 | `0.0289` | `0.2414` | `1.1879` | `1.27e3` |
| `original` | bf16 | 309 151 | `0.0286` | `40.0429` | `399 049` | `1.17e19` |

The fp32 arm finished `9.4` times lower than the control at the last shared step, and never
exceeded it once in `15 025` comparable steps.

## 2. Setup

Both arms resume `cap-6x384-w384/run/checkpoint-best.pt` at update `90 850` with the same
config, the same dataset, the same sampler state and the same schedule.

| Field | Value |
| --- | --- |
| Model | 6 layers, width `384`, 8 heads, FFN `1536`, `11 560 103` parameters |
| Batch | `512`, one microbatch, no accumulation |
| Peak rate | `2e-4`, warmup `10 000`, cosine horizon `400 000` |
| Stop rule | watcher stops the arm when `global_step` reaches `105 850` |
| Difference | `training.mixed_precision`: `fp32` against `bf16` |

The schedule horizon stays at the original `400 000` for both arms. A shorter budget would
have shortened the horizon and changed the learning rate at every step.

## 3. Three-way comparison

Loss, gradient norm and learning rate. The original column and the control column are
identical because the control reproduced the recorded run bitwise.

| Step | fp32 loss / grad | bf16 control | original run |
| ---: | --- | --- | --- |
| 90 900 | `0.0277` / `0.080` | `0.0281` / `0.0919` | `0.0281` / `0.0919` |
| 92 000 | `0.0277` / `0.117` | `0.0282` / `0.118` | `0.0282` / `0.118` |
| 94 000 | `0.0270` / `0.106` | `0.0290` / `0.169` | `0.0290` / `0.169` |
| 96 000 | `0.0269` / `0.0965` | `0.0306` / `0.253` | `0.0306` / `0.253` |
| 98 000 | `0.0270` / `0.113` | `0.0340` / `0.448` | `0.0340` / `0.448` |
| 100 000 | `0.0261` / `0.076` | `0.0535` / `1.65` | `0.0535` / `1.65` |
| 102 000 | `0.0261` / `0.0827` | `0.0807` / `5.64` | `0.0807` / `5.64` |
| 104 000 | `0.0257` / `0.071` | `0.1198` / `16` | `0.1198` / `16` |
| 106 000 | not reached | stopped at `105 875` | `0.2908` / `146` |

Learning rate is identical in all three columns at every step, from `1.8e-4` to `1.7e-4`.

## 4. Crossing analysis

Over the `15 025` steps that both new arms recorded:

| Quantity | Count |
| --- | ---: |
| Steps where fp32 loss exceeded the control loss | **0** |
| Steps where fp32 gradient norm exceeded the control gradient norm | 312 |
| Last step where the fp32 gradient was higher | `93 905` |
| Control-to-fp32 loss ratio at the last shared step | `9.36` |
| Largest control-to-fp32 loss ratio | `46.6` |

The 312 gradient rows all sit in the early phase, where both norms are below `0.2`. They are
a precision transient, not a trend: after `93 905` the fp32 gradient stays below the
control's at every step.

## 5. Internal corroboration

One batch of `8 192` windows, scored by three checkpoints at update `105 386` on the same
container. Mean absolute attention logit per layer, then attention entropy per layer.

| Checkpoint | Loss mean | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp32` best | `0.0263` | 3.3 | 8.2 | 36.4 | **497.6** | 32.2 | 357.5 |
| `bf16` last | `0.1708` | 3.7 | 6.1 | 284.5 | **25 410 428** | 53.9 | 1 574.1 |

| Checkpoint | L3 entropy | L3 activation norm | L3 LayerNorm input variance |
| --- | ---: | ---: | ---: |
| `fp32` | `2.288` | `28.9` | `2.3` |
| `bf16` | `0.000` | `41.5` | `4 568.6` |

At the same step and on the same batch, the bf16 arm carries a layer-3 logit scale five
orders of magnitude above the fp32 arm, with its entropy collapsed to zero and its LayerNorm
input variance grown by a factor of `2 000`. This matches the pattern measured in the phase 0
report: one layer runs away, its entropy collapses, then the loss rises.

The fp32 checkpoint keeps its attention statistics inside the range that the surviving arms
showed in phase 0.

## 6. What this establishes

| Claim | Status |
| --- | --- |
| bf16 arithmetic is sufficient to trigger this divergence | measured, twice: control equals original, both fail |
| Pure fp32 avoids it for `15 000` updates | measured, one seed, one shape |
| The loss rise in bf16 accompanies a layer-3 logit explosion | measured on one batch, same step |
| The fp32 run keeps that layer bounded | measured on one batch |
| Precision is the only difference between the two arms | by construction, one config field |
| fp32 would survive to `400 000` updates | **not tested** |
| fp32 helps the `20.1 M` production shape | **not tested** |
| fp32 changes the closed-loop fall rate | **not tested** |
| The cause is bf16 rounding rather than bf16 accumulation order | **not established** |

## 7. Limits

- One seed, one model shape (`11.6 M`), one batch size, `15 000` updates. The failure appears
  in the `20.1 M` shape too, but not in this same paired design.
- The stop rule ends the arms at `105 850`; the control was stopped at `105 875`. Longer runs
  could still diverge in fp32, only later.
- The attention numbers come from `8` windows of one batch, so they describe a batch, not a
  population.
- Two differences remain between the arms that the design does not separate: the arithmetic
  width, and the reduced tensor-core usage that comes with it. A `bf16` run with
  `float32_matmul_precision` forced per-operation was not part of this test.
- Training evidence only. No claim about policy quality, survival, or hardware.

## 8. Reproduce

```bash
# one arm, resumed from the same checkpoint, one field changed
CUDA_VISIBLE_DEVICES=0 \
MJLAB_FLOW_WIDTH=384 MJLAB_FLOW_LAYERS=6 MJLAB_FLOW_FFN=1536 \
MJLAB_FLOW_LR_HORIZON_UPDATES=400000 \
python -m mjlab.scripts.flow_matching train \
  --dataset-dir /tmp/mjlab-144/data \
  --config <run>/fp32/config.yaml \
  --contract <root>/control/contract.yaml \
  --out <run>/fp32/run --device cuda:0 \
  --max-updates 400000 --allow-long-run \
  --resume <root>/capacity-20261024/cap-6x384-w384/run/checkpoint-best.pt
```

The `mixed_precision` field is the only difference between the two configs. Resume accepts it
because the configuration hash gate was removed in `066e6d98a`.

## 9. Artifacts

| Item | Path |
| --- | --- |
| Run root | `.../precision-20261025/` on the 4-card container |
| fp32 run | `fp32/run/{metrics.jsonl,checkpoint-best.pt,checkpoint-last.pt}` |
| Control run | `bf16-control/run/{metrics.jsonl,checkpoint-last.pt}` |
| Comparison script | `logs/diagnostics/flow-divergence-phase0/compare_precision.py` |
| Watch scripts | `launch_precision.sh`, `precision_watch.sh`, `watch_fp32.sh` in the same directory |
| Checkpoint analysis | `/home/fushan/data/mjlab-144/phase0b/analysis/precision-105k.json` |
