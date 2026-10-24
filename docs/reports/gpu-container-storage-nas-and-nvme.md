# Managed GPU container storage: large NAS, fast local NVMe

Kind: operational note and measurement record.
Checked: 2026-10-24 16:30 (+08).
Scope: the storage available inside the managed GPU containers, the throughput that
matters for training, and the rules that follow from it.
Measured on: `ssh -p 32221 fushan@10.14.64.48`, a 4-card container on an 8-card host.
The 8-card container `ssh -p 31505 fushan@10.14.64.37` shares the same NAS.

Every number below is labelled **measured** with its date, or **not measured**.

## 1. The two tiers

| Tier | Path | Media | Capacity | Shared with |
|---|---|---|---:|---|
| NAS | `/home/fushan/` | GPFS appliance `10.14.80.16:/gpfs/geniestudio/ml-qbotrd/home/fushan` | 1.5 PB, 964 TB free, 34 % used (2026-10-24) | **every** managed container and every GenieStudio task |
| Local NVMe | `/tmp/` | Kubernetes `emptyDir` on `/dev/nvme0n1` | 894 GB, 670 GB free (2026-10-24 after cleanup) | other pods on the same node, about 198 GB in use |

Three properties follow, and they drive every operational rule in this note:

1. The NAS is the only durable tier. `/tmp` belongs to the pod and disappears when
   the container restarts.
2. `/tmp` is not private. Another pod on the same node consumes the same 894 GB.
3. Both tiers are attached to the **container**, not to the physical host, so IP
   addresses and ports change when a container is recreated.

## 2. NAS throughput, measured

| Measurement | Value | Date | Method |
|---|---:|---|---|
| Random read, small blocks | **0.54 MiB/s** | 2026-10-20 | D1 latent dataset, random window reads |
| Sequential read, one file | 74 MB/s | 2026-10-22 | `dd`/pread over a 356 GB memmap |
| Random 11 KB reads, 512 per update | 2.7 MB/s | 2026-10-22 | training loop on the NAS copy, `D` state, `wait_on_page_bit_common` |
| Single-stream `rsync` over SSH, 642 MB | **3.57 and 3.68 MB/s** | 2026-10-23, 2026-10-24 | two independent checkpoint fetches |
| Three parallel `rsync` streams, dataset copy | **77 to 88 MB/s per stream** | 2026-10-22 | 426.9 GB dataset, 15 files |

The two `rsync` rows disagree by a factor of twenty, and that is the useful
observation: a single stream is throttled to a few MB per second, while three
streams each ran at 77 to 88 MB/s. Parallel streams are the only way to use the NAS
link. This has not been re-tested since, so the exact per-stream ceiling is
**not measured**.

## 3. Local NVMe throughput, measured

| Measurement | Value | Date | Method |
|---|---:|---|---|
| Random read, small blocks | **3,433 MiB/s** | 2026-10-20 | same access pattern as the NAS row above |
| Training throughput on the local copy | **2.95 updates/s** at batch 512, 1,512 windows/s | 2026-10-22 | 100-update smoke, `/tmp/joint-pose-144/data` |
| Training throughput on the NAS copy | **0.04 updates/s** | 2026-10-22 | same configuration, same data, NAS path |
| `rsync` NAS to NVMe, three streams | 77 to 88 MB/s per stream | 2026-10-22 | 398 GiB, verified byte-identical |

The training comparison is the number that matters: **74 times slower** on the NAS
for the identical configuration. The step was idle in `D` state with the GPU at 0 %.

## 4. What the slowness actually was

Not bandwidth. The loader issued 512 independent 11 KB reads per update, one per
window in the batch, scattered across a 356 GB memmap. Each read missed the page
cache, so each became a synchronous round trip.

| Change | Effect |
|---|---|
| Same file, sequential read | 74 MB/s, so the file itself was not slow |
| Sort the microbatch indices before gathering | turns ~512 random reads into a few ascending runs; committed as `e8d636e4e` |
| Move the dataset to local NVMe | 74x step-time improvement |

The lesson is that the NAS punishes random small reads far more than sequential
ones, and that a sorted gather is worth more than any transport change.

## 5. Sizes, so that budgets are not guesses

| Item | Size | Note |
|---|---:|---|
| S1 joint-pose dataset, apparent | 426.9 GB | 38,278,319 windows, fp32 |
| Same dataset, allocated on NAS | 139 GB | `du` after compression/dedup on GPFS |
| Same dataset, NVMe copy | 398 GiB | 15 files, byte-identical |
| One checkpoint, 68-channel model | 320,622,878 B | width 512, 6 layers; raw weights plus EMA plus optimizer |
| A 400,000-update arm | 16 GB | 40 checkpoints |
| A 1,000,000-update arm | 31 to 32 GB | 100 checkpoints |
| Sixteen finished arms | 298 GB | the 2026-10-22 to 2026-10-24 batch |

Checkpoint size does not depend on the training budget, only on the model and the
cadence: `size = 320 MB x (budget / checkpoint_every)`.

## 6. Rules that follow

| Rule | Reason |
|---|---|
| Copy a dataset to `/tmp` before training on it | 74x step-time penalty otherwise |
| Keep checkpoints and metrics on the NAS | `/tmp` is ephemeral and shared |
| Never `rsync` between two hosts | `/home/fushan` is a single mount visible from all of them; a copy would duplicate hundreds of GB for nothing |
| Use parallel streams for a large transfer | one stream is throttled to a few MB/s |
| Verify a transfer with SHA256 on both ends | transfers of this size have been cut mid-stream elsewhere on this network |
| Expect `/tmp` to vanish | a pod restart deletes it; anything needed later must be on the NAS or fetched to the laptop |
| Check free space before a batch | four 1 M arms are 128 GB of checkpoints alone |
| Record paths in the run identity with care | the dataset identity includes `data_dir` as a string, so deleting that path makes old checkpoints refuse to load |

The last rule was learned the hard way on 2026-10-24: deleting `/tmp/joint-pose-144`
freed 397 GB and made every 2026-10-23 and 2026-10-24 checkpoint unloadable until the
recorded path exists again, either as a restored copy or a symlink to the NAS copy.

## 7. Observed failure modes

| Symptom | Cause | Handling |
|---|---|---|
| Step time of 25 s with the GPU idle | random small reads against the NAS | move to NVMe; sort the gather |
| Training process in `D` state, `wait_on_page_bit_common` | synchronous page faults on a memmap over NAS | as above |
| Transfer stalls at a few MB/s | single stream | parallel streams, resumable `rsync --partial` |
| `/tmp` full although only one dataset was copied | other pods share the 894 GB | check `df /tmp` and `du` of own trees before blaming the node |
| SSH `Connection closed by <host> port` mid-session | transient; the host stayed up 102 days in the observed case | retry once after verifying the host and the port respond |

## 8. How to re-measure

| Goal | Command shape |
|---|---|
| Random-read rate of a cache | 512 random windows of one row from the memmap, timed, with the page cache dropped if possible |
| Step time effect | 100-update smoke with `--trainer.max-steps 100` against each data path |
| Transfer rate | `rsync -a --partial --info=progress2` with one stream, then with three |
| Free space | `df -h /home/fushan /tmp` and `du -sh` of the owned trees |

## 9. Not measured

| Question | State |
|---|---|
| Per-stream ceiling of the NAS link | unknown; 3.6 MB/s single and 77 to 88 MB/s in three streams were observed on different days |
| Whether the NAS has a per-client quota | unknown |
| `/tmp` eviction policy under pressure from other pods | unknown; the 894 GB figure is current, not guaranteed |
| NVMe endurance or throughput under parallel readers | not measured; the 3,433 MiB/s figure is one pattern |
| Behaviour of the 8-card container | assumed identical because it mounts the same NAS; the NVMe tier is per node and was not measured there |
