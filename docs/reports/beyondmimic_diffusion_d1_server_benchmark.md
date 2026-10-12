# D1 server throughput benchmark

Authorization: operator requests a server test to measure optimization differences, 2026-10-12 (+08).
Run: `/home/fushan/mjlab/logs/diagnostics/d1-throughput-benchmark-20261012T103034Z/`.
Host: `d-20261004164752-xrjr5`, SSH `10.14.64.48:32221`.
GPU: physical index 3 only. GPUs 0/1 keep the isolated VGK36 diffusion training runs.
Controller: Workflow 19; completion monitor 10.
Tmux: `d1_throughput_bench_20261012_103034`.

## Design

- Baseline source: `e71a89ab2ece99f84872fd562f8374267ffc8624`.
- Optimized source: baseline plus 14 hash-verified accepted local source files.
- Both snapshots use the same frozen VGK36 manifest, checkpoint, recovery policy and motion inputs.
- Shared checkout, frozen stores, statistics and caches are not modified.
- Physics: CUDA, 50 Hz control, 5 ms physics, four substeps.
- Schedule: 36 references × two seeds × three starts × clean/OU = 432 scheduled trials per arm.
- Collection arms, in order: original CPU64, optimized CPU64, optimized GPU64, GPU128, GPU256, repeat GPU64, repeat original CPU64.
- Runtime-factory instrumentation measures construction, collection and store append intervals. It does not change the schedule.
- CUDA parity uses eight actual reset observations and the frozen VAE. Tolerance: rtol/atol 1e-5.
- Two original collected stores supply identical inputs for original and optimized merges.
- All preparation arms consume the same original merged store. Compare original jobs1, optimized jobs1, optimized jobs8.
- Measure inspection, replay, grouped window build, persisted index, statistics, and new token caches separately.
- Check window coverage, split assignments, statistics counts and moments, and token-cache counts.

## Bounds and ownership

- Collection subprocess: 900 s. Internal optimized collection budget: 12 min.
- Offline phases: 120–900 s, based on the phase cost and small dataset size.
- Inactivity: 300 s. Driver total wall limit: 5400 s (90 min).
- Each arm has a new store and report. Output budget: 1024 MiB per collection.
- Driver uses new subprocess sessions and stops only its exact process group.
- First failing arm ends the driver. No identical automatic retry.
- Driver writes `stages.json`, `collection-summary.json`, per-stage logs and exit files.
- Completion uses native tmux signaling and monitor wake; no shell sleep loop.

## Result — setup failure, no throughput measurement

The driver stops at `2026-10-12T18:34:29.501819+08:00` (18:34:29 +08), exit 1.
This is a driver failure, not an SSH disconnect or a timeout.

| Stage | Wall time | Result |
| --- | ---: | --- |
| Original artifact preflight | 17.6184 s | exit 0 |
| Optimized artifact preflight | 20.3178 s | exit 0 |
| CUDA parity setup | 32.2647 s | exit 1, before simulation |
| Seven collection arms | not measured | not started |
| Merge and preparation arms | not measured | not started |

Exact error: `RuntimeX2Error: a shared environment needs at least one row per motion`.
The probe selected all 36 references but requested eight environments.
The eight-row comparison size was incorrectly used as the runtime environment count.
The preflight timing difference is not a throughput result.
CUDA observation/action parity is **not tested** in this run.
No data, fall count, qualification result, or speed ratio was measured.

The corrected local probe requests 64 environments and compares eight reset observations.
It passes syntax and minimum-row checks. The corrected server probe is **not tested**.
The failed remote driver and probe remain unchanged. No automatic relaunch occurs.

## Operational status

At 18:36 (+08), GPU 3 is back to 1 MiB and no active benchmark subprocess remains.
The exited driver remains as a defunct process until its parent reaps it; it executes no work.
The same training PIDs `441945` and `441954` remain active on GPUs 1 and 0.
All captured frozen control-input hashes match. Shared HEAD remains `e71a89ab2`.

Local retained evidence: `logs/diagnostics/d1-throughput-benchmark-20261012T103034Z/`.
This folder holds `driver.log`, `stages.json`, `failure.json`, `final.json`, and `cuda-parity.log`.
Remote failed outputs remain in the original run directory.

Next: operator authorizes diagnosed retries until measured numbers exist.
Fresh retry1: `/home/fushan/mjlab/logs/diagnostics/d1-throughput-benchmark-retry1-20261012T105328Z/`.
Tmux: `d1_bench_retry1_20261012_105328`; Workflow 20; completion monitor 11.
The corrected probe uses 64 environments and stores CPU/GPU arrays before parity assertions.
The driver checks source imports with a bound and records CPU activity for inactivity detection.
Baseline and 14 accepted source hashes are unchanged. Failed first-attempt evidence is retained.
Retry1 stops at 18:56:01 (+08) on CUDA latent parity. Real runtime starts; CPU tensor observation/action parity is exact.
Saved-input diagnosis passes under monitor12: TF32 latent error `0.000775933`, action error `0.001261532`; IEEE FP32 latent error `6.25849e-7`, action error `1.04308e-6`.
IEEE FP32 passes the unchanged rtol/atol `1e-5`; no tolerance widening occurs.
Benchmark-only `configure_torch_backends(allow_tf32=False)` runs after runtime construction in all arms. A local mocked regression confirms it.
Accepted source and frozen model bytes remain unchanged. Unconfigured production CUDA parity remains unqualified.
Fresh retry2: `/home/fushan/mjlab/logs/diagnostics/d1-throughput-benchmark-retry2-20261012T105928Z/`.
Tmux: `d1_bench_retry2_20261012_105928`; Workflow20, monitor13.
Same14 source hashes and GPU3 idle verified.

## Retry2 interim measurements — 19:25:31 (+08)

All seven collection arms finish with exit0, `partial:false`, and `stop_reason:null`.
Both merge paths finish. Current phase is **original-stats**, not optimized statistics.
The benchmark deliberately runs both original and optimized stages, plus repeated collection arms.
The total benchmark duration is not the optimized pipeline duration.

| Collection arm | Environments | Total wall s | Stored rows/s | Versus pooled original |
| --- | ---: | ---: | ---: | ---: |
| Original CPU, two repeats | 64 | 179.36 / 183.39 | 589.45 pooled | 1.00x |
| Optimized CPU | 64 | 178.31 | 594.25 | 1.01x |
| Optimized GPU, two repeats | 64 | 160.26 / 160.39 | 668.89 pooled | 1.13x |
| Optimized GPU | 128 | 133.04 | 806.07 | 1.37x |
| Optimized GPU | 256 | 116.44 | 899.02 | 1.53x |

Totals include source imports, frozen-artifact preflight, runtime creation, collection and writing.
Runtime creation is 16.5–17.4s. Store appends remain about25–26s per arm.
Optimized CPU collection shows **no material throughput gain** in this small run.
GPU inference and larger batches improve measured throughput. CPU recovery/evidence/writes remain.
Accepted rows and skipped OU partners differ across arms; matched schedules are not identical trajectories.
Do not infer policy quality or exact data parity from timing.

Original merge: **88.6004s**. Optimized merge: **28.2117s**, **3.14x faster**.
Both consume identical stores and report **213818 input/output rows**, 54 shards.
Optimized merge uses `mode:shard-copy`.
Same-reset IEEE CUDA parity passes: latent6.26e-7/action1.04e-6 maximum error, rtol/atol1e-5.

Original inspection36.43s, replay42.27s, grouped build57.11s, persisted index43.14s complete.
## Retry2 final result — completed at 19:42:14 (+08)

Exact driver finish field: `2026-10-12T19:42:14.779602+08:00` (19:42:14 +08).
Driver and all33 subprocess stages exit0. Total benchmark wall time: **2538.7s (42min18.7s)**.
This duration includes original/optimized pipelines, repeated arms, parity and integrity checks.
It is not an optimized production pipeline duration.

### Offline preparation on the same 213818 rows

| Stage | Original jobs1, s | Optimized jobs1, s | Optimized jobs8, s | Original / optimized jobs8 |
| --- | ---: | ---: | ---: | ---: |
| Inspection | 36.43 | 29.61 | 22.61 | 1.61x |
| Replay | 42.27 | 26.90 | 19.79 | 2.14x |
| Grouped window build | 57.11 | 58.15 | 42.01 | 1.36x |
| Persisted index | 43.14 | 42.01 | 24.48 | 1.76x |
| Statistics | 111.84 | 46.69 | 23.05 | **4.85x** |
| Token caches, all splits | 93.64 | 88.29 | 365.48 | **0.26x; 3.90x slower** |
| Stage sum | **384.44** | **291.64** | **497.42** | **0.77x; 29.4% longer** |

One-worker optimized preparation takes24.1% less time than the original (1.32x speed ratio).
Using eight workers for every stage makes the pipeline slower in this small dataset.
Token-cache split times with eight workers are112.6s/train,113.4s/validation,121.7s/test.
With one optimized worker they are57.9s/train,5.2s/validation,4.5s/test.
The source uses spawned workers per split. The timing does not isolate startup from storage contention.
Exact cache slowdown cause is **not measured**. Do not call this a validated token-cache speedup.

### Data and integrity gates

- Seven collection arms: exit0, no partial report, no stop reason.
- Fast merge: **54 shards** are hash-verified byte-identical to the original source shards.
- Same-row preparation: exact split assignments and window coverage match across three arms.
- Windows: **43680 train /5552 validation /4796 test =54028**.
- State/latent statistics counts match. Maximum moment difference: **5.17e-15**.
- Token caches build all three splits. Counts match; byte identity across changed statistics is not claimed.
- Statistics SHA differs as expected after the floating-point reduction-order change.
- Saved-reset IEEE CUDA parity passes the unchanged1e-5 tolerance.
- These tests do not establish closed-loop trajectory equivalence or hardware quality.

Collection physical-fall counts: original64 repeats4/3; optimized CPU64=7; GPU64 repeats3/3; GPU128=5; GPU256=9.
Skipped OU partners: original64=3/3; optimized CPU64=6; GPU64=2/2; GPU128=2; GPU256=9.
All arms schedule432 trials, but accepted/stored totals vary. Compare rates without inferring quality.

### Recommended settings for this measured scope

- GPU-native VAE with IEEE FP32 improves measured collection throughput.
- **256 environments is fastest tested for this small, one-GPU, two-seed experiment.**
- This does not establish a production optimum with four collectors or larger datasets.
- Use fast shard-copy merge.
- Use **jobs8** for validation/index/statistics on this213818-row dataset.
- Use **cache-jobs1**, not8, for this small dataset.
- A mixed-setting stage sum is **220.23s** (1.75x faster than384.44s).
  This is a calculated sum of completed stages, **not a measured single end-to-end run**.
- No production configuration, shared source, historical dataset/statistics/cache, or training source is changed.

### Validation scope clarification and follow-up

The timing benchmark prepares datasets from the **original CPU-collected merged store**.
Its per-optimized-collection validation gap is now closed by a separate read-only audit.
Monitor14 exits0 after122.605s, with `ok:true`.
Exact observation: `2026-10-12T21:17:44.527375+08:00` (21:17:44 +08).
Evidence: `result-validation-v1/audit.json` in the retry2 directory and local evidence folder.

- All7 original/optimized collection stores and both213818-row merged stores pass full row replay.
- All shard hashes, array shapes/dtypes, finite values and controller/latent consistency pass.
- All stores have unique identities, zero arithmetic/semantic errors, and50Hz tick/timestamp continuity.
- Each collection store includes all36 motions and clean/OU phases.
- Accepted-row counts match reports; completed+skipped trials equal432 for every arm.
- Three prepared datasets × three splits pass full `.npy` content-hash, shape, float32 dtype, finite-value and provenance checks.
- Cache shapes are43680×41×167/train,5552×41×167/validation,4796×41×167/test.
- Five raw windows per split are regenerated for each prepared dataset (**45 comparisons**).
  Every sampled cache window matches its regenerated float32 tokens exactly (maximum error0).
- All input manifests and captured frozen control inputs remain unchanged.

**Result: raw benchmark stores pass integrity/replay validation; prepared datasets pass source/cache validation.**
No collection, cache rebuild, training, or GPU work occurs during this audit.
Full numerical regeneration of every cached window is **not tested**.
The optimized collection stores are validated raw stores, not separately prepared trainable bundles.
Preparation correctness is tested on the same original merged rows. No hardware/policy-quality claim follows.

### Final operational checks and evidence

At19:43 (+08), GPU3 is1MiB. The last benchmark subprocess is gone.
The same training PIDs441945/441954 remain active on GPUs1/0.
All captured frozen control-input hashes and14 accepted source hashes match.
Shared HEAD remains `e71a89ab2ece99f84872fd562f8374267ffc8624`.
`final-readonly-check.json` passes source hashes, source-shard copies, stage statuses and cleanup.

Evidence: `logs/diagnostics/d1-throughput-benchmark-retry2-20261012T105928Z/`.
This folder holds stage logs, arm reports, offline reports, parity, equivalence and ownership checks.
Run `uv run --no-sync python logs/diagnostics/d1-throughput-benchmark-retry2-20261012T105928Z/aggregate.py` to rebuild `measurements.json`.
Frozen raw stores and generated token arrays stay on the server; only small final evidence is fetched.
Concurrent training and shared storage limit generalization. No statistical confidence interval or full-scale extrapolation is claimed.
Benchmark work is complete. No additional retry is required to obtain numbers.
No production optimization rollout, new training or robot operation is included.
