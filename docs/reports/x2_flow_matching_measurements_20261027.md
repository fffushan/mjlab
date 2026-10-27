# X2 flow matching: the 1 M-update runs and the measured cost structure

Status: measurement note, written 2026-10-27 11:30 (+08).
Owner: training workstream. Scope: three production flow-matching runs, the
precision result, and the cost of an update, a restart and a byte of storage.

This note records numbers, not plans. Each section states what was measured, on
which host, and what remains unknown.

## 1 Purpose

Two questions are answered here.

1. Does a pure fp32 run survive a full 1 000 000-update cosine schedule at the
   production shape, on the same data that broke every bf16 arm?
2. Where do the wall-clock seconds go, so that runs can be sized and scheduled?

## 2 Runs covered

State at 2026-10-27 11:24 (+08).

| Run | Host | Precision | Masking | Step | Exposure | Best loss | Validation |
|---|---|---|---:|---:|---:|---:|---:|
| `fp32-prod-1m` | 4-card | fp32 | none | 905 646 | 50.09 | 0.01428 | 0.01713 |
| `hist20-fp32-1m` | 8-card | fp32 | 20 % history dropout | 550 995 | 43.46 | 0.01692 | 0.01851 |
| `rootmask20-fp32-1m` | 8-card | fp32 | root position + root linear velocity, 20 % | 549 490 | 43.18 | 0.08277 | 0.09494 |

Exposure is the cumulative sum of the learning rate. It is the only axis
comparable across recipes, because it integrates rate and step count.

The root-mask arm drops its six masked channels from the loss mean, so its
absolute loss averages 129 of 135 channels and is not comparable with the other
two rows.

## 3 Divergence: what the measurements support

### 3.1 Precision is sufficient to trigger the failure

Two arms started from the same `cap-6x384` checkpoint and changed one field,
`mixed_precision`.

| Arm | Loss first → last | Loss maximum | Gradient maximum | 15 025-step outcome |
|---|---:|---:|---:|---|
| fp32 | 0.0286 → **0.0261** | 0.0289 | 0.188 | no failure |
| bf16 control | 0.0289 → **0.2414** | 1.1879 | 1 270 | diverged |

Source: `x2_flow_matching_precision_branch_20261025.md`.

### 3.2 The failure signature is an attention runaway in one layer

Same batch, same optimizer step, step 105 386.

| Quantity | fp32 | bf16 |
|---|---:|---:|
| Layer 3 logit scale | 497.6 | 25 410 428 |
| Layer 3 attention entropy | 2.288 | 0.000 |
| Layer 3 normalisation input variance | 2.3 | 4 568.6 |

Across nine failing arms, exactly one layer runs away in each. The layer differs:
L2 for `fn-2048`, L3 for `cap-6x384`, L4 for `lrh-160k`. At the loss minimum the
affected layer is already 100 to 1 000 times the survivor level.

### 3.3 The weight scale confirms the absence of a runaway

The phase-0 instrument is the per-layer attention logit scale. The RMS of each
layer's attention input projection tracks it and needs only a checkpoint read.
Measured on the 4-card container at 2026-10-27 11:26 (+08).

| Checkpoint | Step | L0 | L1 | L2 | L3 | L4 | L5 | max abs weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32-prod-1m` | 904 866 | 0.051 | 0.056 | 0.056 | 0.053 | 0.062 | 0.069 | 0.578 |
| `cap-6x384` (diverged) | 400 000 | 0.083 | 0.071 | 0.084 | **1.038** | 0.073 | 0.115 | 2.511 |

The fp32 run has all six layers within a factor of 1.36 of each other after
905 000 updates. The diverged arm has one layer at 1.038, about 14 times its own
siblings. This is the first direct instrument reading on a long fp32 run, and it
agrees with the loss curve.

### 3.4 Growth precedes the loss rise

In two arms the runaway began 5 000 to 16 000 updates before the loss rose. The
gradient is a co-indicator, not a precursor: the first 5x gradient event led the
first 1.5x loss event by 0, −1, +864, −284 and −1 updates across five arms, and
two survivors never produced a gradient event at all.

Annealing can freeze a runaway without repairing it: `lrh-160k` grew layer 4 to
2.27 M while its loss moved only from 0.021 to 0.036.

### 3.5 The fp32 run has passed every earlier failure exposure

Highest exposure at which a bf16 arm first showed an abnormal gradient: 26.54.
The fp32 run is at 50.09, 1.9 times that value, with loss still falling,
gradient norm 0.033, and flat per-layer weights.

This is evidence for one recipe, one seed and one shape. It is not a proof that
fp32 cannot diverge at larger exposure.

### 3.6 The mechanism remains a hypothesis

The supported reading is a positive feedback loop inside one attention layer,
seeded by bf16 matmul rounding:

1. Logits grow, softmax sharpens, entropy falls.
2. A sharper softmax concentrates gradient on fewer positions.
3. The projections that produce Q and K grow, so logits grow again.

Rounding to 8 mantissa bits perturbs a logit of magnitude 30 by about 0.03. Near
saturation that is enough to change which position wins, so a head can flip its
attention mass between steps instead of settling. The measured 4 568 normalisation
variance is consistent with the resulting collapse.

Three facts narrow the candidate:

- `torch.autocast` already keeps softmax and LayerNorm in fp32, so the rounding
  that matters is in the matmuls that create the logits.
- `bf16-pure`, which stores bf16 parameters but keeps fp32 moments and fp32 EMA,
  did not diverge. It plateaued at 0.40, about 27 times worse than fp32.
- The affected layer differs between arms, which a fixed bug would not produce.

Not established: whether the loop needs rounding at all, or whether rounding only
amplifies a sharpening that exact arithmetic also shows at long horizons.

The test that would settle it: run bf16 with the Q/K projections and the `QKᵀ`
matmul in fp32, everything else bf16. If the runaway disappears, the mechanism is
confirmed and bf16 becomes usable with one narrow exception. Not run.

## 4 Throughput: where the seconds go

### 4.1 One update, split by phase

Measured on an idle RTX 4090 on the 8-card container, production shape,
batch 512, fp32, via `bench_update_phases.py`.

| Phase | Time | Share of an observed step |
|---|---:|---:|
| Forward and backward | 76.0 ms | 56 % |
| TensorBoard mirror on NFS | 15 to 44 ms | 11 to 33 % |
| Window gather from the token memmap | 2.6 ms | 2 % |
| EMA update, fp32, 20.1 M parameters | 1.5 ms | 1 % |
| AdamW step | 0.06 ms | below 1 % |
| Metrics, gradient clip, batch IPC, Python | about 40 ms | 30 % |

At 2.4 TFLOPs per update, 76 ms is about 31 TFLOPS effective. That is close to
the practical fp32 limit of the card, so the forward and backward pass is near
hardware speed.

### 4.2 TF32 gives no gain

Forward and backward measured 76.05 ms with TF32 enabled against 76.04 ms with it
disabled. Matmul precision therefore does not limit the step. The bottleneck is
the normalisation, softmax and layout work around the matrices.

### 4.3 TensorBoard written to NFS was the only large avoidable cost

| Host | Write target | Cost per mirrored row | Effect on rate |
|---|---|---:|---|
| 4-card | NFS run directory | 48.3 ms | 0.95 updates/s |
| 4-card | local NVMe | 0.1 ms | **7.5 updates/s** |
| 8-card | NFS, event file 85 MB | 14.8 ms | 7.1 updates/s |
| 8-card | NFS, event file 100 MB | 43.9 ms, p95 146 ms | 5.0 updates/s |
| 8-card | local NVMe | 0.88 ms | **7.7 updates/s** |

Each change moved one environment variable, `MJLAB_TENSORBOARD_DIR`, and nothing
else. The cost grows as the event file grows, so the same arm degrades during a
run unless the writer is local.

### 4.4 `metrics.jsonl` is not a cost

Measured on the 8-card container.

| Operation | Median | p95 |
|---|---:|---:|
| NFS append with `flush()` | 0.001 ms | 0.003 ms |
| NFS append with `fsync()` | 0.69 ms, later 1.3 ms | 0.87 ms, later 30.8 ms |
| Local append with `flush()` | 0.001 ms | 0.002 ms |
| JSON encode and decode of one row | 0.009 ms | not measured |

The trainer calls `flush()` and never `fsync()`, so the first row applies. One row
is about 250 bytes; at 7.5 updates/s the write rate is **2 KB/s**, and the file is
56 MB after 222 000 updates. The cost of a mirrored row is a round trip, not
bandwidth.

## 5 Storage behaviour

### 5.1 An NFS page fault can stall a training process indefinitely

One arm stopped producing updates on the 4-card container. Its state was:

| Check | Value |
|---|---|
| Main thread | sleeping on a futex, waiting for a worker |
| One worker thread | uninterruptible wait in a page fault |
| `read_bytes` | frozen at 6 082 281 472 B across four samples |
| CPU | 5 % of one core |
| GPU | idle |

The dataset directory on that host was a symbolic link to the NAS, so the 51 GB
token cache was memory-mapped over NFS. Moving the data to local NVMe removed the
exposure.

### 5.2 Six streams copy the store about 25 times faster than one

| Method | Rate | 73 GB projection |
|---|---:|---:|
| One rsync stream, source on NFS | 4 MB/s | about 5 h |
| Six parallel streams | about 100 MB/s | about 12 min |

Verification after the copy: source and destination both 74 874 786 108 bytes,
both 2 241 store shards, all three token caches equal in size, and the 51 GB train
cache byte-identical by `cmp`.

### 5.3 Local disk use is bounded

| Host | Total | Used | Our share | Growth to 1 M updates |
|---|---:|---:|---:|---:|
| 4-card `/tmp` | 894 GB | 312 GB | 70 GB dataset copy | below 100 MB |
| 8-card `/tmp` | 894 GB | 490 GB | 70 GB dataset, existing | about 0.7 GB |

Run artifacts stay on the NAS: about 1 GB per arm at present, about 1.5 GB at
1 M updates. Checkpoints are bounded because `checkpoint_every` is unset, so only
`checkpoint-best.pt` and `checkpoint-last.pt` are written.

## 6 Restart and resume costs

### 6.1 Startup costs 18 to 25 minutes and is not I/O

Measured on both hosts, with the token cache on local NVMe.

| Observation | Value |
|---|---|
| Wall time to first update | 18 to 25 min |
| CPU | 95 to 100 % of one core |
| Disk reads during the phase | 0 to 122 KB |
| Resident memory | grows to about 33 GB |
| Change from moving data to local NVMe | none measured |

The phase is single-threaded CPU work over 1.86 million windows. It runs before
the model reaches the GPU, so the GPU shows 1 MiB and the task looks absent.

### 6.2 A byte-identical copy was rejected by a resume gate

The first restart onto local storage failed after a full startup with
`train: resume dataset identity mismatch`. The checkpoint's dataset identity
records the `directory` field, so the same bytes at a new path were refused.

The gate is removed in commit `34033bbbd`. Both trainers now print the fields that
differ, for example:

```
resume note: dataset identity differs in 1 field(s): directory
```

The family, process, statistics, model, contract, schedule and projection identity
checks remain.

### 6.3 Each restart rolls back about 2 900 updates

Checkpoints are written once per epoch, which is 3 633 updates at this dataset
size. A restart therefore repeats part of an epoch. Setting `checkpoint_every`
would bound that loss; it was left unset for these runs.

## 7 Measurement cautions

- **Rate probes must not parse the metrics file inside the rate window.** Two
  earlier probes re-read a 200 MB file while dividing by a fixed sleep, which
  under-counted elapsed time and produced rates about four times too low. Rates in
  this note come from `grep -c` counters over at least 60 seconds of wall time
  taken from `date +%s`.
- **A 60-second window can land on an epoch boundary.** A validation pass plus a
  checkpoint write makes the writer silent. The affected arm differs each hour, so
  the artifact is identifiable by comparing with the hour-averaged step count.
- **A resumed run replays step numbers from its checkpoint.** Step comparison
  cannot separate old rows from new ones; file row count can.
- **The VPN drops silent SSH sessions.** A command that sleeps 60 seconds without
  output was closed twice. `ServerAliveInterval=10` plus periodic output fixed it.

## 8 Working rules that follow

1. Keep `MJLAB_TENSORBOARD_DIR` on local disk whenever the run directory is on
   NFS. On the 4-card host this is worth 8 times the throughput.
2. Keep the memory-mapped token cache off NFS. A single page fault can stop a run.
3. Use a counter-only rate probe over at least 60 seconds.
4. Read the per-layer attention weight RMS to judge a checkpoint, not the loss.
   The best-loss checkpoint of every failing arm was already inside the runaway.
5. Delete a gate that blocks a legitimate experiment instead of adding an
   override. The dataset-identity gate blocked a byte-identical copy.

## 9 Evidence paths

| Item | Path |
|---|---|
| Phase split benchmark | `logs/diagnostics/flow-divergence-phase0/bench_update_phases.py` |
| I/O and TensorBoard latency probe | `logs/diagnostics/flow-divergence-phase0/probe_io_latency.py` |
| Attention weight probe | `logs/diagnostics/flow-divergence-phase0/probe_attention_weights.py` |
| Restart scripts | `logs/diagnostics/flow-divergence-phase0/restart_fp32_local.sh`, `restart_8card_local_tb.sh` |
| Copy scripts | `logs/diagnostics/flow-divergence-phase0/copy_local_parallel.sh` |
| Precision branch report | `x2_flow_matching_precision_branch_20261025.md` |
| Precursor report | `x2_flow_matching_divergence_precursor_20261025.md` |
| Runs | `/home/fushan/mjlab-multimotion144-bulk-20261020/logs/diffusion/x2-multimotion144-preview-bulk-20261020/{fp32,hist20-fp32,rootmask20-fp32}-1m-20261025/` |
| Local dataset copy | `/tmp/mjlab-local/data` on the 4-card container |
