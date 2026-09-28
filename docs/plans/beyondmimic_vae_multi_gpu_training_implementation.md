# Multi-GPU distillation training — implementation plan

Status: **approved; in progress.** Step 1 (identity and configuration) is in the
working tree, uncommitted. A multi-worker *run* is refused with an explicit
diagnostic until step 2 lands, so no invocation can silently train on one
device while claiming several. Implementation does not change the VAE
objective, the replay/draw contracts, or the single-device path's behaviour.

Repository: `/home/agiuser/projects/mjlab`.
Branch: `fxy/test/tracking-exp`. Planning baseline: `8de96dc10`.

Related documents:

- [VAE distillation architecture](beyondmimic_vae_distillation.md)
- [M4 implementation and acceptance](beyondmimic_vae_m4_implementation.md)
- [Phase-balanced sampling plan](beyondmimic_vae_phase_balanced_sampling_implementation.md)
- [Reproduction review](beyondmimic_reproduction_review.md)

## 1. Goal, authority, and non-goals

The next cohort has **eight teachers**. At 8192 envs the current run measures
~9.5 GB VRAM and ~3.8 s/iter, and collection — not the VAE update — dominates
that time. Doubling to 16384 envs on one GPU fits in VRAM (≈19 GB) but
serializes an already-saturated simulation.

**Goal:** one training run using N GPUs, so collection wall-time falls with N
while every existing contract keeps its meaning.

- Target run: 16384 envs on the bare-metal host as
  `--device cuda:0 --worker-devices "('cuda:1','cuda:2')"`, i.e. a dedicated
  trainer GPU plus two 8192-env shards.
- Single-device behaviour is bit-for-bit unchanged when no worker list is
  given.

Non-goals:

- No gradient-parallel DDP, no multi-node, no NCCL dependency.
- No changes to the VAE model, loss, teacher bank, manifest, or replay policy.
- No sharding of the offline `evaluate-cohort` / `play` CLIs.
- No change to the sample:update ratio; that stays a launch-recipe knob (§9).

## 2. Verified current architecture (seams this must preserve)

- `DistillationRunner.run_iteration` (`runner.py:171`) drives
  bootstrap/collection/update/evaluation. `_collect` (`runner.py:137-151`)
  builds `CollectionConfig(steps, teacher_probability, rollout_latent,
  seed=self.config.seed + self.iteration, collector_iteration=self.iteration,
  rng_mode="persistent")` and calls `collector.collect(...)`.
- `DAggerCollector` (`collector.py:919`) owns `(adapter, bank, model, replay)`;
  `collect(config, reset)` (`collector.py:977`) returns `CollectionResult`
  (`collector.py:200`) with ticks/samples/teacher/student counters,
  `boundaries`, `disagreement_mean`, `fresh_data` (`FreshTrainingData`) and
  per-motion `motion_stats`.
- **Insertion happens per tick inside the collector**:
  `self.replay.insert(fresh_batch)` (`collector.py:1088`); the appended batches
  become `FreshTrainingData` (`collector.py:1161`). There is exactly one
  logical replay.
- **No early exit on the success path** (`collector.py:1050-1180`): a call
  produces exactly `config.steps` ticks of `env.num_envs` rows each. Ragged
  output only occurs on the exception path, which returns a prefix.
- Normalizers are updated **once per fresh row**, not per draw
  (`trainer.update_normalizers_from_new_data`), an invariant with a dedicated
  test.
- The env is one vectorized `MultiMotionDistillationAdapter` built with
  `num_envs` and `device` (`adapter.py:1483-1546`); rows carry their own
  `motion_id`/`teacher_id` routing; `build_cohort_teacher_bank(cohort,
  device=device)` is per device.
- `rebase_segment_ids(iteration, namespace)` (`runner.py:153-157`) rewrites
  segment ids after a resume so a restarted simulator cannot collide with
  retained rows.
- Evaluation runs **on the collector's adapter**
  (`evaluate_distillation(self.collector.adapter, ...)`, `runner.py:206-227`),
  then invalidates the snapshot and sets `_reset_required = True`.
  `EvaluationResult` (`collector.py:392`) retains per-segment records and
  derives `metrics`, `per_motion`, `completion_rate`, `failure_rate` from them
  — so it is composable across shards without information loss.
- Resume restarts the simulator by design (`resume_restarts_simulator`), and
  strict resume compares the stored cohort record and resolved configuration.
- CLI: `distill train --device STR --num-envs INT`; no `--gpu-ids` and no
  distributed code anywhere in this package (grep-verified).

## 3. Resolved design decisions

| # | Decision | Choice |
|---|---|---|
| 1 | GPU layout | **3 GPUs**: dedicated trainer on `--device` (no env) + workers on `--worker-devices`. Sharing a GPU with a worker stays permitted but is documented as contended. |
| 2 | Row transport | **`spawn` + pinned-host staging through `multiprocessing` queues.** No new dependency, no process-group lifecycle, one transport code path for CPU tests and CUDA. |
| 3 | Resume with a changed worker list | **Refused** (worker count, devices, split and seed scheme join the strict-compared resolved configuration). |
| 4 | Env split | **Even**: `num_envs % N == 0`, refused before any environment exists. |
| 5 | Scale | Design for **N ≤ 8** workers; validate N=2 on GPU and N=8 with CPU fakes. |

### 3.1 Shape: sharded collection, single learner

N worker processes, one per worker device, each hosting its own env shard and
its own copy of the (small) teacher bank. The launched process is the parent
and owns the trainer: the single replay, model, optimizer, generators,
checkpoint writer and report. Workers collect and return rows; the parent
merges, inserts, updates normalizers and runs the VAE updates.

Rejected: **gradient-parallel DDP** (per-rank env, replay and model copy). It
fragments the replay, so motion- and phase-balanced draws would be computed
per rank over data that differs from the union; the normalizer "once per fresh
row" invariant would become per-rank; and strict resume would have to
reconcile N divergent buffers and RNG streams. It also buys nothing: the VAE
is tiny, the simulation is the cost, and sharding parallelises that either
way.

Invariants workers must never violate: no checkpoint writing, no normalizer
updates, no replay insertion, no report emission.

### 3.2 Configuration surface

- `--device` stays the trainer device (unchanged default).
- New opt-in `--worker-devices` (Tyro collection literal, e.g.
  `"('cuda:1','cuda:2')"`), length N.
- `--num-envs` is the TOTAL, split evenly across N.
- Refusals, all before any environment is constructed: an empty list; a
  repeated device (compared after canonicalization, so `cuda` and `cuda:0`
  collide); a device string `torch.device` cannot parse (the same requirement
  the trainer device already carries, so `CUDA:01` is refused);
  `num_envs % N != 0`; N > 8; and any explicit list at all until the pool
  exists.
- Worker entries are canonicalized with `torch.device`, which needs no CUDA
  runtime. **Deferred to step 2:** refusing a device that parses but does not
  *exist* (`cuda:9` on an 8-GPU host). That check needs the pool that binds the
  device, and until then no explicit list can reach a run, so it is unreachable
  rather than unimplemented; it lands with the pool preflight.
- Repeated `cpu` entries are allowed deliberately: a CPU shard contends for no
  physical device, and repeating `'cpu'` is how the N-worker merge path is
  exercised without GPUs. Repeated GPU entries stay refused.
- Resolved configuration records one compared `runtime.workers` object
  (`mode` `single`/`sharded`, `devices`, `count`, `envs_per_worker`,
  `seed_scheme`) plus an audit-only `execution.transport`.
- Absent `--worker-devices` ⇒ today's single-process path, byte-identical apart
  from that one new compared entry and the audit key.

### 3.3 Deterministic seeding

- worker env seed: `base + worker_index * 1_000_003`
- worker collection generator: `base + iteration + worker_index * 7_919`
- worker evaluation: `base + iteration + worker_index * 7_919` (same domain as
  collection; they are distinct calls, never concurrent)
- `worker_seed_scheme_version = 1` is recorded, so any future change is a
  resume-identity break rather than silent distribution drift.

Workers must get **distinct** env seeds: identical seeds would simulate N
identical instances and quietly reduce effective coverage.

A worker re-seeds its collection generator **per call** from the derived value
(`rng_mode="per_call"`) instead of carrying a persistent stream. This is what
makes a sharded resume reproduce collection without adding per-worker RNG state
to the checkpoint: iteration `k` draws the same randomness for worker `w`
whether or not the run was interrupted before it. The persistent per-worker
stream a single-process collector uses would have to be saved and restored for
each worker, which the version-2 checkpoint has no place for.

### 3.4 Process model and protocol

- `spawn` (required: a forked CUDA context is invalid).
- Per worker: one command queue and one result queue, with monotonic message
  ids; a mismatched or stale id is a fail-closed error, never a warning.
- Commands: `collect(iteration, steps, probability, seed, weights, reset)`,
  `evaluate(iteration, steps, mode, seed, weights)`, `shutdown`.
- Weight broadcast: the student `state_dict` is flattened and sent before
  **each** collection call, so collection always runs exactly the policy the
  trainer holds — no stale-policy collection, preserving DAgger semantics.
- Every wait is bounded. A worker exception, crash or timeout aborts the
  iteration, kills all workers, invalidates the collector snapshot
  (`_invalidate_on_failure` semantics) and raises; no hang, no orphan.
- The child watches the parent: no message for a bounded interval ⇒ exit, so a
  `SIGKILL`ed parent cannot leave GPU processes behind. `SIGINT`/`SIGTERM` on
  the parent triggers a bounded grace-period teardown of the whole group.
- The parent validates the cohort identity (manifest digest, teacher digests)
  once and does not build an env; each worker builds its own bank from that
  validated cohort record.

### 3.5 Transport

Each worker stages its completed rows into **pinned host memory** and sends
them per tick. The parent copies them to `cuda:0` and inserts. Costs at N=2,
8192 rows/worker/tick, ≈0.9 kB/row: ≈7.4 MB per tick per worker, ≈236 MB per
worker per iteration ⇒ ~50–60 ms at a realistic 10 GB/s effective PCIe rate,
against a ~1.5–2 s iteration — a few percent. Simple, dependency-free, and it
exercises the same code path the CPU tests use.

**Measured trigger for an upgrade:** if the smoke shows transport above 10% of
iteration wall-time, switch to CUDA-IPC or NCCL `send/recv` (an internal
change behind the same protocol, not a redesign).

The recorded transport name is **audit-only**: it sits with `checkpoint_every`
on the non-compared side of the resolved configuration, because the transport
does not change which rows are sampled and switching to a faster one must not
refuse an existing run's resume.

### 3.6 Merge, insert, and segment identity

- **Insert order: per tick, workers ascending.** Tick k from every worker is
  inserted before tick k+1, preserving the "row block per tick" FIFO semantics
  of today's insert-per-tick loop.
- **Atomic per iteration:** the parent inserts only after every worker
  returned successfully. A failed collection call inserts nothing, so an
  aborted iteration leaves the replay and normalizers untouched. (Deliberate
  improvement over the single-device prefix-on-exception path, which stays
  unchanged.)
- Segment ids become `[namespace | worker | local]` so ids stay globally
  unique in the merged replay; `rebase_segment_ids` rewrites only the
  namespace field and therefore keeps its resume meaning per worker. The parent
  applies the worker offset during the merge (`WORKER_SEGMENT_STRIDE = 1 << 40`)
  because segment identity is a parent-side concern, and a shard whose local ids
  would reach into its neighbour's block is refused rather than silently
  overlapping.
- Every merged insert still goes through `validate_replay_batch`, so a worker
  cannot inject a malformed or mis-routed row.

### 3.7 Telemetry and reporting

- Each worker reports the distinct reference frames it observed per motion, so
  merged frame coverage is the **exact union** (its count and min/max); the
  largest-shard fallback survives only for a caller that cannot supply frame
  identity, and is documented as a lower bound.
- `ticks` is the **union** (equal to `steps` on success), not a sum;
  `samples`, `teacher_steps` and `student_steps` are **sums**;
  `disagreement_mean` is **sample-weighted**; `motion_stats` are summed per
  motion.
- Boundaries are concatenated workers-ascending, each worker's records in tick
  order, each gaining an additive `worker_index` field — env indices are
  per-worker and must never be read as one global index. This is deliberately
  *not* the tick-major nesting the row stream uses: attributing one boundary to
  one tick needs a collector contract change (the collector appends boundaries
  per tick but returns them flat), and the ordering that actually matters for
  correctness is the row order, which decides what a full replay forgets first.
  A consumer correlating boundaries with rows therefore correlates per shard.
- Diagnostics are prefixed with the worker index; seed provenance is reported
  per worker; per-worker collection/evaluation wall-time joins the iteration
  event so the speedup can be read from the run itself.
- Replay telemetry is unchanged: there is still exactly one replay.

### 3.8 Evaluation

Evaluation is **sharded across the same workers** with the common reset
discipline: each worker resets and rolls out its shard, the parent concatenates
the per-segment records from all workers and re-derives
`metrics`/`per_motion`/`completion_rate`/`failure_rate` from the merged
segments. This keeps the evaluated env count equal to the training env count —
the same invariant as today — so metrics stay comparable across the
single-device and multi-GPU configurations, and it costs no extra VRAM.

Because `EvaluationResult` derives everything from retained segments, the merge
is a concatenation plus re-derivation with no loss of information. After
evaluation every worker's collector is invalidated, so the next collection
passes `reset=True` to **all** workers — the per-worker mapping of today's
`_reset_required` flag.

### 3.9 Checkpoint, resume, identity

Mechanism unchanged: the parent writes the versioned cohort checkpoint with
model, optimizer, normalizers, replay, RNG and counters. Resume restarts
simulators by design, so workers are recreated rather than restored. A
multi-worker checkpoint cannot be resumed by a single-worker configuration (or
vice versa): the trainer device and the whole `runtime.workers` object — mode,
devices, count, envs per worker, seed scheme — are strict-compared. Keeping the
identity in one object is what makes the legacy default safe: it is applied
whole, so a partially populated worker record can never be read as the
single-process case. `execution.transport` is deliberately not compared.

### 3.10 Parent-side artifacts without a local environment

In the sharded layout the parent owns no environment, so three things the
single-process path reads from its adapter have to come from elsewhere:

- **Environment description.** Each worker answers a `describe` request with the
  reset-policy enabled flag, the reset provenance and the sampling mode read
  from *its* live command. The parent records those, so the run still describes
  what was built rather than what was requested.
- **Evaluation.** There is no adapter to hand to `evaluate_distillation`, so
  every worker evaluates its own shard with the broadcast weights and the parent
  merges the retained per-segment records (§3.8).
- **Collector RNG in the checkpoint.** The parent has no collector, so sharded
  checkpoints are written with `collector=None`, which the existing format
  already permits; the optional version-3 reset-RNG record therefore stays
  single-device-only. Per-worker generator state is unnecessary because of the
  per-call derivation in §3.3.

**Shared construction recipe.** Workers build their environment through
`cohort_setup.CohortSetup`, the same object the single-process path uses, so the
CLI and every worker construct one environment by construction rather than by
convention; worker index 0 with a base seed reproduces the single-process
environment exactly. A worker's collector is built with `replay=None`: its rows
are returned to the parent, and the parent stays the replay's only writer.

**Row framing.** A worker returns its rows *concatenated tick by tick* together
with `rows_per_tick`, and the parent slices the ticks back out. That is one
transfer per iteration instead of one per tick, while the tick structure stays
exact because every tick contributes the same fixed number of rows.

**Failure protocol.** Every request carries a monotonic `call_id`; a reply that
does not match the outstanding id, a worker that exits, and a worker that stops
answering within the request bound all abort the pool, close it, and raise —
never warn and continue. A child also exits on its own once it is reparented,
which is what keeps a `SIGKILL`ed parent from leaving GPU-holding orphans.

## 4. Implementation steps

1. **Identity and config (no behaviour change):** segment-id composition
   `[namespace | worker | local]`; `--worker-devices` plus all §3.2 refusals;
   resolved-config recording; `rebase_segment_ids` namespace-field aware.
2. **Worker and pool:** `distillation/worker.py` (child entry: adapter/bank
   construction, weight apply, collect/evaluate, teardown, parent watchdog)
   and the pool inside the runner (spawn, broadcast, bounded waits, recovery).
3. **Merge and telemetry:** extract the per-tick insert path so one worker
   (today) and N workers (new) share it; atomic per-iteration insertion;
   aggregation rules of §3.7.
4. **Sharded evaluation** with merged-segment recomputation, and per-worker
   invalidation after eval.
5. **CLI, resume identity, docs:** parent-side construction without an
   environment (cohort identity and replay weights from a worker's `describe`
   reply, extended to carry the library and the live audit), report/progress
   output, changelog, usage docs, this plan's status. Also here: remove the
   staged `--worker-devices` refusal, install the parent-side `SIGINT`/`SIGTERM`
   teardown the plan requires, and add the real sharded save/resume round trip
   (there is no worker-produced checkpoint to test against yet).
6. **Tests and the authorized smoke** (§5).

Ownership: one seam, one writer (collection execution, pool, CLI wiring). The
parent reviews the diff and runs the focused suites; the GPU smoke needs
separate resource authorization.

## 5. Tests and acceptance

CPU, with the existing fake-adapter harness (no GPU):

1. Single-device suites pass unmodified (49 cohort CLI tests and 427
   distillation tests as landed), with the default path's compared runtime
   entry pinned by an exact key-set assertion. A full golden fixture in the
   phase-balanced style is deferred to step 2: nothing here changes draw or
   collection behaviour, and the resolved configuration changed by design (one
   new compared entry plus one audit key).
2. Two CPU workers: divisibility and device-list refusals; tick-major
   worker-ascending insert order; counters/motion-stats/weighted-disagreement
   aggregation; union of rows equals the concatenation of worker rows with no
   duplication or loss.
3. Determinism: two runs with the same seed produce identical worker rows,
   identical insert order and identical checkpoints.
4. Failure atomicity: a worker that raises aborts the iteration with the
   replay, normalizers and counters untouched (nothing from that iteration
   inserted); no child process survives.
5. Segment-id namespaces are disjoint across workers and the resumed-rebase
   path still works.
6. Resume identity: a changed worker list, changed split, changed trainer
   device or changed seed-scheme version is refused; the unchanged list
   resumes.
7. Normalizers still count every fresh row exactly once, on the parent only.
8. Sharded evaluation: merged metrics equal the merged-segment recomputation;
   a two-shard eval of identical synthetic shards reproduces the single-shard
   rates; every worker is invalidated afterwards.
9. N=8 CPU workers (2048 envs each) run a bounded iteration.

The independent review of step 1 (§9.1) named the failures ordinary tests would
miss in the later steps. These are required coverage, not aspirations:

- **Segment identity and merge:** maximum ids, a resumed namespace rebase, an
  out-of-order worker completion, and a worker failing after several ticks must
  not duplicate, reorder, or half-insert segments while leaving the replay and
  boundaries superficially plausible.
- **Failure protocol and transport:** a crash, queue timeout, or `SIGTERM`
  during and after a device→host transfer must leave no partial insertion, no
  blocked queue, and no surviving child or GPU context; the atomic boundary is
  an explicit parent-side transaction under backpressure.
- **Evaluation and normalizers:** merged per-segment recomputation is compared
  against a single synthetic union with empty and failing shards included, and
  every fresh row is still counted exactly once by the parent's normalizers.

On the 8-card host (authorized separately):

10. Bounded smoke: 2 workers × 4096 envs, 2 iterations — per-GPU VRAM, s/iter,
    per-worker collection time, transport share of the iteration, aggregated
    report, and a save→resume round trip.
11. Target run: 3-GPU layout, 2 × 8192 = 16384 envs on the eight-teacher v2
    manifest, reporting s/iter against the current 3.78 s/iter single-GPU
    baseline and ≈9.5 GB per worker GPU.

## 6. Definition of done

The default path is unchanged; a 3-GPU run completes the smoke with aggregated
telemetry, a checkpoint and a resume; the tests above pass; the CLI records and
enforces the worker configuration. No production training run is part of this
milestone until the smoke passes and the user authorizes the launch.

## 7. Target run recipe (decided)

The 8-teacher run keeps the live run's calibration: 0.5 gradient draws per
fresh row, 10 phase cells per motion, and the same total collected rows and
gradient steps as the 3-teacher phase-balanced run.

```
--manifest configs/distillation/x2_tennis_mixed_v2.yaml
--teacher-ids "('tennis_000','tennis_001','tennis_002','tennis_003_ss','tennis_004_ss','tennis_005_ss','tennis_006_ss','tennis_007_ss')"
--device cuda:0 --worker-devices "('cuda:1','cuda:2')"   # dedicated trainer + two shards
--num-envs 16384 --collection-steps 32                    # 524288 fresh rows per iteration
--replay-capacity 1048576                                 # 2.7x: 13107 rows per phase cell, as today
--updates-per-iteration 8 --accumulation-steps 8 --minibatch-size 4096   # holds draws/fresh = 0.5
--max-iterations 5000                                     # same totals as 10000 x 8192
--bootstrap-steps 128 --phase-bins 10 --seed 7
--reset-policy standing-mixture --standing-start-fraction 0.25
--standing-start-window-frames 25 --standing-start-frame-zero-fraction 0.5
--checkpoint-every 500 --progress-every 50
```

Why these numbers, measured from `checkpoint-iter-002500` of the live run
(replay 336.6 MB / 393216 rows = **856 bytes per row**; model 22.8 MB;
optimizer 45.7 MB):

| | live 3-teacher run | target 8-teacher run |
|---|---|---|
| envs / collection steps | 8192 / 32 | 16384 / 32 |
| fresh rows per iteration | 262,144 | 524,288 |
| rows per teacher per iteration | 87,381 (87,381.3) | 65,536 |
| gradient draws per iteration | 131,072 (0.50 of fresh) | 262,144 (0.50 of fresh) |
| replay capacity | 393,216 rows = 337 MB | 1,048,576 rows = 898 MB |
| rows per motion per phase cell | 13,107 (13,107.2) | 13,107 (13,107.2) |
| iterations | 10,000 | 5,000 |
| total fresh rows / draws | 2.62e9 / 1.31e9 | 2.62e9 / 1.31e9 |
| expected wall-clock | 10.5 h (3.78 s/iter measured) | ≈7 h (projection) |

Capacity has two independent drivers: `motions x phase cells` sets coverage,
and `motions x steps x envs` sets the replay's horizon in iterations. Doubling
the envs alone doubles the capacity needed for the same horizon; doubling the
collection steps as well requires **4x**, not 2x. At 856 bytes per row even 4x
(1.35 GB) is trivial on the dedicated trainer GPU, so capacity is not the
constraint — the trainer's gradient work and the resulting wall-clock are.

A 3-GPU layout buys **throughput, not shorter iterations**: each worker runs
8192 envs x 32 steps, which is exactly the live run's per-iteration collection
in the same wall-clock while holding twice the rows. Doubling the update budget
is what spends part of that gain to keep every fresh row trained on as heavily
as before.

Two things the recipe deliberately does **not** preserve, both unavoidable once
total rows are held fixed:

- rows per teacher per iteration fall from 87,381 to 65,536, because eight
teachers share the same total instead of three;
- the replay's horizon rises from 1.5 to 2.0 iterations of fresh rows, because
capacity grows 2.7x while fresh rows per iteration grow 2x.

Cell counts are nominal: 13,107.2 rows per motion-phase cell means integer
quotas with a distributed remainder, which is what the replay already does. The
wall-clock projection assumes a particular learner and transport overhead, and
replacing it with a measurement is exactly what the smoke is for.

## 8. Adjacent, not part of this plan

The bare-metal host's queue still sets `EXCLUDE_GPUS=5` and its dispatcher is
not running after the GPU-5 repair; the interrupted standing-teacher jobs are
marked pending/failed. Lifting the exclusion is a small separate change plus a
dispatcher restart (`queue/dispatch.sh` is tracked in the repo).

## 9. Progress

| Step | State | Evidence |
|---|---|---|
| 1. Identity and configuration | in the working tree (uncommitted), independently reviewed | `--worker-devices` refuses every explicit list before an environment exists; one compared `runtime.workers` object plus audit-only `execution.transport`; whole-object legacy defaulting; 49 cohort CLI tests and 427 distillation tests pass |
| 2. Worker and pool | in the working tree (uncommitted), independently reviewed | `worker.py`: request/reply protocol with monotonic call ids, pinned-host staging, bounded waits that also watch the child, fail-closed pool abort, teardown leaving no process, and a child watchdog on reparenting; `cohort_setup.py` holds the construction recipe both paths share; the collector accepts `replay=None` so a worker cannot write the parent's replay |
| 3. Merge and telemetry | implemented, independently reviewed, committed | tick-major merge, atomic insert, exact counter/frame aggregation, segment namespacing, reset telemetry, fail-closed invalidation after a post-reply failure |
| 3. Merge and telemetry | not started | |
| 4. Sharded evaluation | implemented | each shard evaluates its own environments with the broadcast weights and the parent recomputes every rate, mean and trial bucket from the union of retained segments through the same aggregator the single-process path uses, so denominators add up across shards instead of averaging averages |
| 5. CLI reporting and docs | implemented, user docs outstanding | the CLI accepts `--worker-devices` and builds the parent side from a worker's `describe` reply (schema, reference library, live audit, motion-teacher codes); `SIGINT`/`SIGTERM` close the pool; the staged refusal is gone. User-facing docs in `docs/source/x2_tennis_distillation.rst` are deliberately untouched while that file carries unrelated uncommitted edits |
| 6. Smoke and acceptance | not started; needs GPU authorization | |

Backward compatibility is verified against a real artifact, not only fixtures:
the live run's `checkpoint-iter-002500.pt` (recorded before the option existed)
still compares as compatible under the new single-device configuration, is
refused under a sharded configuration, and the single-teacher runtime entry is
still compared verbatim.

### 9.1 Independent review of step 1

A separate reviewer process (`gpt-5.6-luna`, max thinking, tools restricted to
`read,grep,find,ls,bash`) reviewed the uncommitted diff against this plan and
re-ran both suites itself. Verdicts: C1 verified; C2/C3/C5 partial; C4 verified
for the implemented paths; C6 verified; C7 not provable from that host.

It found a real blocker the author's tests had missed: a **one-entry** list
passed the `len(...) > 1` guard, so `--worker-devices "('cuda:1',)"` trained on
the trainer device while the checkpoint recorded
`worker_devices: ["cuda:1"]`. Reproduced by the author (exit 0, checkpoint
written, `device: cpu` next to `worker_devices: ["cuda:1"]`) and fixed by
refusing every explicit list until the pool exists; the reproduction is kept as
a test.

Accepted and fixed: unparseable device strings are refused and entries
canonicalized; a repeated device is detected after canonicalization; the worker
identity is one compared object with an explicit `mode`/`count`/`seed_scheme`
instead of loose fields, so legacy defaulting applies whole and a partially
populated worker record can no longer be read as single-process; the run report
exposes the same identity plus the recorded transport; the single-teacher
refusal test now proves no adapter was constructed.

Rejected, with reasons: an explicit `worker_count` decoupled from the device
list (it is derived in one place, and a second field could disagree with it);
transport in the compared identity (it does not change sampled rows, so
comparing it would refuse a legitimate resume after a transport optimization —
it is recorded as audit, the `checkpoint_every` precedent); a device-existence
check in step 1 (unreachable while every list is refused; lands with the pool).

Acknowledged limits: no real multi-worker checkpoint can exist yet, so the
changed-worker resume test injects a stored identity rather than producing one,
and per-teacher rows cannot match the three-teacher run by construction.

### 9.2 Independent review of steps 2 and 3

Two reviewers ran in parallel against the frozen tree: a `reviewer` child for
adversarial code review and an `oracle` child for independent verification
(both `gpt-5.6-luna`, max thinking, read-only). The verifier reproduced the
suites itself (448 passing at review time) and independently confirmed the merge
order at shapes the author's tests did not use, the weighted means, the
zero-row/`None` handling, the frame-count behaviour, and that a pool failure
leaves the replay untouched. The reviewer's verdict was **BLOCK**, and it was
right on every point the author re-verified:

- **Pool collection was serialized** — worker 1 was asked only after worker 0's
  reply was consumed, so collection wall time was the sum of the shards, not the
  maximum. Every counter and report would have looked correct; only the lost
  parallelism would have shown, in the smoke. Fixed by dispatching every request
  before awaiting any reply, with an interval-overlap test that does not depend
  on process startup cost.
- **Segment ids were not worker-namespaced**, so two simulators' segment 7
  collided in one replay and the resume rebase could never separate them. Fixed
  as in §3.6, with a refusal when a shard's local ids would overflow its block.
- **Reset telemetry was dropped**: `eligible_resets`/`initialization_resets`
  never crossed the worker boundary and silently defaulted to empty. Now carried
  and summed.
- **Frame coverage was reported as a lower bound under an exact-sounding name.**
  Workers now report their observed frame sets and the merge takes the exact
  union (§3.7).
- **Post-reply failures did not fail closed**: a merge or insert failure left
  the shards' advanced environments and the source claiming no reset was
  required. Now any post-reply failure marks the source invalidated, closes the
  pool, and raises.
- **Worker freeze could be undone by the weight load** (`_frozen` is a
  registered buffer) and **a reparented child could wait 15 minutes** before
  noticing. The freeze is re-applied after every load, and the child now watches
  its parent on a 5-second interval.

The author's own new tests caught two further defects while fixing these: the
segment offset was applied worker-major instead of tick-major, and the weight
broadcast called `.detach()` on every `state_dict` value — but a model state
dict also carries plain-data `_extra_state`, so the sharded path would have
crashed on its first iteration.

Rejected, with reasons: renaming `reference_frames_observed` (the exact union is
now transported, so the name is honest); a replay-insert rollback transaction
(`insert` already validates the whole batch before mutating any partition, and
documents that a failure cannot leave part of an insertion committed); making
boundaries tick-major (§3.7 records why that needs a collector contract change).

Reviewer recommendations still open and folded into step 5: the parent-side
environmentless construction, the parent-side signal teardown, and a real
sharded save/resume round trip. Repeated `cpu` worker devices are still allowed
on purpose so the merge path stays testable without GPUs.

### 9.3 Re-review of the fix pass

The same two reviewers re-ran against the frozen tree. The verifier reproduced
both suites itself (29 focused, 457 full) and independently falsified each fix:
segment ids disjoint and tick-major at shapes the author's tests did not use
(including after a namespace rebase), exact frame unions for overlapping and
disjoint sets, reset telemetry summed per motion and kind, fail-closed
invalidation, a real `_extra_state` state-dict round trip, and the normalizer
re-freeze after a load. Six of the eight items were VERIFIED outright;

two further lifecycle gaps were found and are now fixed:

- **The per-worker deadline was receive-anchored.** Because requests are
  dispatched together and awaited in worker order, worker 1's timeout clock
  started only once worker 0 had answered, so an exchange could take N timeouts.
  Deadlines are now anchored at dispatch, with a test that warms the shards,
  keeps worker 0 inside the budget, and proves the pool fails at about one
  timeout rather than one plus a worker.
- **The child's parent watch did not run during a request.** The in-loop check
  only ran between requests, so a parent killed mid-collection left the child
  busy for the rest of that rollout. A watcher thread now exits the child on
  reparenting regardless of what it is executing, and `WorkerPool.close` gained
  a configurable teardown grace so a caller that must stop promptly (an
  operator's signal handler in step 5) is not held by a hung worker.

Remaining before the CLI is runnable, unchanged from §9.2 and confirmed by both
reviewers: parent-side environmentless construction, parent-side
`SIGINT`/`SIGTERM` teardown, sharded evaluation, and a real worker-produced
save/resume round trip. "Steps 2 and 3 are implemented" is not a claim that the
multi-worker CLI runs yet; it still refuses every explicit worker list.

### 9.4 Steps 4 and 5 landed

Everything above is now implemented in the same working tree:

- **Sharded evaluation.** `evaluate_distillation`'s reporting is one pure
  function of its segment list (`aggregate_evaluation_metrics`), extracted so the
  sharded merge calls it on the concatenation of every shard's segments rather
  than averaging shard results. Rates therefore keep their true denominators,
  trial buckets add up, and `reference_frames_observed`-style distinct counts are
  not double-counted. Segments carry the shard that produced them.
- **Parent-side construction.** A worker's `describe` reply now carries the
  reference library, the live multi-motion audit and the motion-teacher codes, so
  the parent can build the replay, the cohort identity and the trainer without
  owning an environment. `cohort_identity_from_parts` makes the identity
  buildable from those parts; worker 0's environment seed is the single-process
  derivation, so a sharded run records the identity a single-process run of the
  same recipe would.
- **Teardown.** `SIGINT`/`SIGTERM` release the pool before the process exits, in
  addition to the child-side watcher that covers a killed parent.
- The staged `--worker-devices` refusal is removed: the flag now selects the
  sharded builder, and the single-teacher path still refuses it.

What the CPU suites can and cannot prove: the pool, merge, aggregation, refusal
and dispatch behaviour are covered by 466 distillation tests, but the sharded
end-to-end path needs real environments. The first real exercise is the smoke
below, on the 8-card host, which is also what validates env construction through
the shared recipe, the transport's real cost, and a worker-produced checkpoint.

Smoke command (2 workers, bounded):

```
.venv/bin/distill train \
  --manifest configs/distillation/x2_tennis_mixed_v2.yaml \
  --teacher-ids "('tennis_000','tennis_001','tennis_002','tennis_003_ss','tennis_004_ss','tennis_005_ss','tennis_006_ss','tennis_007_ss')" \
  --device cuda:0 --worker-devices "('cuda:1','cuda:2')" \
  --num-envs 8192 --bootstrap-steps 8 --collection-steps 4 \
  --updates-per-iteration 1 --minibatch-size 1024 --accumulation-steps 2 \
  --replay-capacity 16384 --phase-bins 10 --max-iterations 2 \
  --reset-policy standing-mixture --standing-start-fraction 0.25 \
  --standing-start-window-frames 25 --standing-start-frame-zero-fraction 0.5 \
  --checkpoint-every 1 --progress-every 1 --seed 7 --output-dir <run dir>
```

Deliberately small: its job is to prove that two worker processes build their
environments through the shared recipe, that rows arrive and merge into one
replay, that a checkpoint is written with the worker identity, and that a resume
of it is accepted — not to produce a policy.
