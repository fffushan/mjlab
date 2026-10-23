# X2 flow matching: closed-loop failure and the training divergence ablation

Date: 2026-10-23 (+08).
Execution status: all listed arms completed or were stopped by design; no arm is running.
Validation status: closed-loop evidence for three checkpoints; loss evidence for thirteen arms.
Scope: simulation only. Managed containers `10.14.64.37` (8-card) and `10.14.64.48` (4-card).

## 1. Result summary

The current flow-matching tracker fails in closed loop much more often than the previous
36-teacher model. Validation loss does not predict the closed-loop order.

| Checkpoint | Closed-loop fall rate | Motions | Rows | Status |
| --- | --- | --- | --- | --- |
| `root20-400k` best | `53.4%` | 8 | 2048 | measured |
| `root20-400k` last | `82.0%` | 8 | 2048 | measured |
| `base-100k` best | `41.7%` | 8 | 2048 | measured 2026-10-21 |
| `root20-100k` best | `59.2%` | 8 | 2048 | measured 2026-10-21 |
| 36-teacher model | `~15%` | unknown | unknown | user-reported; not re-measured |

The loss order contradicts the closed-loop order. `base-100k` best has the lowest
validation loss (`0.0244`) and scores `41.7%`. `root20-400k` best has a worse recorded
validation loss (`0.164`) and scores `53.4%`.

## 2. Issue

Three questions were open at the start of the ablation.

1. Does the large batch size (`2048`) cause the divergence?
2. Is the model too large for the dataset?
3. Does validation loss rank closed-loop quality?

## 3. Method

### 3.1 Configuration under test

| Field | Value |
| --- | --- |
| Model | 6-layer transformer, 8 heads, width `512`, FFN `2048` |
| Parameters | `20,132,007` (measured from the checkpoint) |
| Effective batch | `2048` (microbatch `2048`, accumulation `1`) |
| Learning rate | `2e-4`, linear warmup `10000`, cosine to zero |
| Update horizon | `400000` |
| Precision | `bf16`, gradient clip `1.0`, weight decay `1e-3` |
| Data | 144-clip offline store, `148` clips, `47653` frames, `15.9 min` at `50 Hz` |
| Training windows | `1,860,237` (each frame appears in `39.0` windows) |
| Token cache | `50.9 GB` train, `6.0 GB` validation |
| Reference width | `130` (`encoder_command`, includes the preview window) |

### 3.2 Arms

Two waves. All arms use the same data, model, warmup and cosine shape. Only the listed
fields change. Records are under
`logs/diffusion/x2-multimotion144-preview-bulk-20261020/matrix-8arm`,
`.../matrix-batch-ablation` and `.../matrix-abl2`.

| Arm | Batch | lr | Seed | `level_shift_gamma` | Horizon | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `base-100k` | 2048 | 2e-4 | 0 | 3.0 | 100 k | short reference |
| `base-400k` | 2048 | 2e-4 | 0 | 3.0 | 400 k | control |
| `hist20-400k` | 2048 | 2e-4 | 0 | 3.0 | 400 k | history dropout 20% |
| `root20-400k` | 2048 | 2e-4 | 0 | 3.0 | 400 k | masked state 20% |
| `root50-400k` | 2048 | 2e-4 | 0 | 3.0 | 400 k | masked state 50% |
| `root100-400k` | 2048 | 2e-4 | 0 | 3.0 | 400 k | masked state 100% |
| `a128-1p6m` | 128 | 2e-4 | 0 | 3.0 | 1.6 M | paper batch, equal samples to 36-teacher |
| `b512-400k` | 512 | 2e-4 | 0 | 3.0 | 400 k | batch axis |
| `c2048-lr1e-4` | 2048 | 1e-4 | 0 | 3.0 | 400 k | lr axis |
| `d1024-lr2e-4` | 1024 | 2e-4 | 0 | 3.0 | 400 k, stopped 120 k | dose response |
| `e2048-lr1p5e-4` | 2048 | 1.5e-4 | 0 | 3.0 | 400 k, stopped 120 k | lr threshold |
| `f2048-seed1` | 2048 | 2e-4 | **1** | 3.0 | 400 k, stopped 120 k | reproducibility |
| `g2048-gamma1` | 2048 | 2e-4 | 0 | **1.0** | 400 k, stopped 120 k | level-noise hypothesis |
| `i512-lr1e-4` | 512 | 1e-4 | 0 | 3.0 | 400 k | 36-teacher recipe on this data |

Arms marked "stopped 120 k" keep the `400000` update horizon in the config, so their learning
rate curve matches the control, and a watcher sends `SIGTERM` when `global_step` passes
`120000`. They produce curves, not final weights.

### 3.3 Comparison configuration

| Property | SCDP paper | 36-teacher run | this run |
| --- | ---: | ---: | ---: |
| Effective batch | `128` | `512` | `2048` |
| Peak learning rate | `1e-4` | `1e-4` | `2e-4` |
| Updates | `1,000,000` | `400,000` | `400,000` |
| FFN width | `1024` | not recorded | `2048` |
| Attention heads | `4` | not recorded | `8` |
| `vae.dimensions.encoder_command` | not applicable | **`68`** | **`130`** |
| Data | 130 clips, 5,200 rollouts, `~750 k` timesteps | 564 rollout shards, `2.5 GB`, tokens `11.84 GB` | 148 clips, `47,653` frames, tokens `50.9 GB` |

## 4. Results

### 4.1 Training divergence

`eval_model` loss, from `metrics.jsonl`. Milestone columns are epochs.

| Arm | Step | Epochs | Best @epoch | Best @updates | Last | e80 | e100 | e120 | e200 | e300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base-100k` | 100000 | 110 | 0.0244 @94 | 85 k | 0.025 | 0.025 | 0.024 | — | — | — |
| `base-400k` | 400000 | 440 | 0.0235 @85 | 77 k | 0.360 | 0.024 | **3.190** | 0.076 | 0.357 | 0.404 |
| `hist20-400k` | 400000 | 440 | 0.0232 @95 | 86 k | 0.418 | 0.024 | 0.026 | 0.506 | 0.769 | 0.645 |
| `root20-100k` | 100000 | 110 | 0.0834 @109 | 99 k | 0.083 | 0.088 | 0.084 | — | — | — |
| `root20-400k` | 400000 | 440 | 0.0930 @97 | 88 k | 1.077 | 0.097 | 0.101 | **6.972** | 4.550 | 1.432 |
| `root50-100k` | 100000 | 110 | 0.0830 @110 | 100 k | 0.083 | 0.087 | 0.083 | — | — | — |
| `root50-400k` | 400000 | 440 | 0.0921 @92 | 84 k | 2.092 | 0.094 | 0.298 | 1.127 | 1.444 | 1.348 |
| `root100-400k` | 400000 | 440 | 0.0897 @98 | 89 k | 73.115 | 0.094 | 0.090 | 0.834 | 3.804 | **10785.2** |
| `a128-1p6m` | 1126116 | 77 | **0.0209** @16 | 232 k | 0.333 | — | — | — | — | — |
| `b512-400k` | 400000 | 110 | **0.0208** @38 | 138 k | 0.375 | 0.485 | 0.402 | — | — | — |
| `c2048-lr1e-4` | 153621 | 168 | 0.0313 @97 | 88 k | 0.145 | 0.034 | 0.032 | 0.100 | — | — |
| `d1024-lr2e-4` | 120223 | 66 | 0.0223 @50 | 91 k | 2.882 | — | — | — | — | — |
| `e2048-lr1p5e-4` | 120153 | 132 | 0.0256 @94 | 85 k | 1.417 | 0.027 | 0.031 | 0.791 | — | — |
| `f2048-seed1` | 120122 | 132 | 0.0232 @90 | 82 k | 0.562 | 0.024 | 0.026 | 0.586 | — | — |
| `g2048-gamma1` | 120012 | 132 | 0.0235 @85 | 77 k | 0.792 | 0.024 | **3.190** | 0.076 | — | — |
| `i512-lr1e-4` | 400000 | 110 | 0.0260 @39 | 142 k | 0.485 | 0.523 | 0.479 | — | — | — |

Update counts in the "Best @updates" column use `1,860,237 / batch` updates per epoch.

### 4.2 Observations from the table

1. Eleven of eleven arms that ran past their peak degraded.
2. The peak is at epoch 77 to 98 for arms that reach it; the two 100 k arms end near their peak.
3. Smaller batch gives a **lower** best loss: `0.0208` at batch `512`, `0.0209` at `128`, against `0.0235` at `2048`.
4. Smaller batch delays the degradation in update terms: `77 k` at batch `2048`, `91 k` at `1024`, `138 k` at `512`, `232 k` at `128`.
5. `root100-400k` does not plateau. It reaches `10785.2` at epoch 300, then `20.3` at epoch 400. It is divergence, not convergence to a worse solution.
6. Cosine reaches zero, so the final epochs train at a learning rate near `3.8e-7`. The loss does not recover. The degraded weights are stable.

### 4.3 Invalid arms

| Arm | Why it is invalid |
| --- | --- |
| `g2048-gamma1` | The config uses `level_sampling: uniform`, where `level_shift_gamma` has no effect. The curve is bit-identical to `base-400k`, including the `3.190` spike. The arm cost three GPU-hours and tested nothing. |
| `root100-400k` last | Loss `73` to `10785`. Not sent to closed-loop evaluation. |

## 5. Findings

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| Batch too large | **Ruled out as the cause** | 128, 512, 1024 and 2048 all degrade |
| Learning rate too high | **Ruled out as the cause** | 1e-4, 1.5e-4 and 2e-4 all degrade |
| Seed luck | **Ruled out** | Seed 1 degrades like seed 0 |
| Model too large for the data | **Not supported** | This store has `50.9 GB` of tokens against `11.84 GB` for the stable 36-teacher store |
| Training runs too long | **Supported** | Every arm peaks at epoch 85 to 98; the two 100 k arms end at their peak |
| Loss ranks closed-loop quality | **Contradicted** | `base-100k` (0.0244) scores `41.7%`; `root20-400k` (0.164) scores `53.4%` |
| Loss detects degradation inside one arm | **Supported** | `root20-400k` best 0.0930 → `53.4%`; last 1.077 → `82.0%` |

## 6. Remaining candidate

The strongest untested difference between the stable 36-teacher run and this unstable run is
the **reference preview window**. The two contracts share the same schema and the same
conditioning (`99`), latent (`32`), action (`31`), state (`135`) and control period (`0.02 s`).

| Contract field | 36-teacher run (stable) | this run (unstable) |
| --- | ---: | ---: |
| `vae.dimensions.encoder_command` | `68` | `130` |
| all other dimensions | identical | identical |

The `62`-value difference is the lookahead preview added to the objective.

## 7. Limits

- Closed-loop numbers exist for three checkpoints only. Ten arms have loss curves and no closed-loop score. I measured the loss-to-failure link for one arm pair.
- You reported the `~15%` value, and a different configuration produced it. I did not re-measure it here.
- A watcher stopped four wave-2 arms at `120000` updates. They carry no final weights.
- No arm used the 36-teacher dataset for training. That dataset is intact; see Section 8.
- The `b512`, `a128` and `c2048-lr1e-4` arms use `bf16` and one seed each. I measured seed variance for batch `2048` only.
- Simulation only. No physical validation.

## 8. Artifacts and reproduction

### 8.1 This dataset

| Item | Path |
| --- | --- |
| Dataset (local NVMe, used by training) | `/tmp/mjlab-144/data` |
| Dataset (NAS copy) | `/home/fushan/data/mjlab-144/data` |
| Contract | `.../x2-multimotion144-preview-bulk-20261020/control/contract.yaml` |
| Base config | `.../x2-multimotion144-preview-bulk-20261020/control/flow-train144.yaml` |
| Arms | `.../matrix-8arm`, `.../matrix-batch-ablation`, `.../matrix-abl2` |

The two dataset paths are different files with the same content size. They are not the same
inode. Training used the local copy: `--dataset-dir /tmp/mjlab-144/data`.

### 8.2 The 36-teacher dataset

| Item | Path or size |
| --- | --- |
| Dataset root | `/home/fushan/mjlab/logs/diffusion/d1-vgk36-double-20261012` |
| Same directory, second path | `/home/fushan/mjlab-diffvae/logs/diffusion/d1-vgk36-double-20261012` |
| Inode evidence | both paths report `225:4537052146`; one directory, not two copies |
| Rollout store | `store/`, 564 shards, `2.5 GB` |
| Token caches | `offline/tokens-train` `11.84 GB`, validation and test caches present |
| Contract of that era | `d2-vgk36-baseline-400k-seed0-retry1-20261012T092934Z/contract.yaml` |
| Token caches built? | yes; no preparation step is needed |

### 8.3 Launch pattern used

Every arm ran in a `tmux` session on the 8-card host with these environment settings and
command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<worktree>/src \
UV_PROJECT_ENVIRONMENT=/home/fushan/mjlab/.venv OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv run --no-sync --project /home/fushan/mjlab \
  python -m mjlab.scripts.flow_matching train \
  --dataset-dir /tmp/mjlab-144/data --config <arm>/config.yaml \
  --contract <control>/contract.yaml --out <arm>/run --device cuda:0 \
  --max-updates <horizon> --allow-long-run --json <arm>/run/train.json
```

Three operational notes from this session:

1. `setsid nohup` children did not survive the SSH session. `tmux` sessions did.
2. Startup validates the dataset identity over the token caches. It reads at about `26 MB/s`
   and saturates one CPU core, so first metrics appear after roughly 35 minutes per arm.
3. Several arm directories hold a small stale `run/metrics.jsonl` from a 16-update smoke.
   Read the larger `run-local/metrics.jsonl` or the file with the newest timestamp.

## 9. Next actions

Two tests remain, in order of cost.

| Option | Change | Cost | What it decides |
| --- | --- | --- | --- |
| A | Train this 144 store with `lookahead_s = 0`, batch `512`, lr `2e-4` | one 400 k arm | Is the preview term the trigger? |
| B | Train this config on the 36-teacher store with its `68`-wide contract | one 400 k arm, plus that era's VAE checkpoint | Reproduces the stable run with current code |

Option A isolates one variable on existing data. If it stays stable past epoch 180, the
preview term is the cause, not the batch, the learning rate or the data volume.

Both options require authorization before launch. Six GPUs were idle on the 8-card host at the
end of this session.
