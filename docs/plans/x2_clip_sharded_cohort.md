# Clip-sharded cohorts: one teacher subset per worker card

Status: implemented 2026-11-06 (+08), CPU-validated only. Reviewed the same day
(`logs/diagnostics/clip-shard-20261106/review-correctness-performance.md`) and
corrected in `logs/diagnostics/clip-shard-20261106/fix-report.md`. This document
does not authorize a training run, and the server verification below has not run.

## Problem

A multi-clip distillation cohort does not fit one GPU once the clip count is
large. Every worker builds the whole cohort, so the per-member cost is paid
seven times over and on the parent as well.

Measured on `.37` with the shared-export manifest (`rsl_rl`-style X2 tracking
teacher, one 3 261-clip cohort, single worker unless stated):

| Members | Envs | Worker card | Trainer card | Result |
| --- | --- | --- | --- | --- |
| 144 | 288 | 3 123 MiB | 4 506 MiB | ok |
| 144 | 2 880 | 5 057 MiB | 5 020 MiB | ok |
| 1 000 | 1 000 | 17 171 MiB | 17 634 MiB | ok |
| 2 000 | 2 000 | 22 571 MiB | 393 MiB | OOM |
| 3 261 | 3 261 | 22 571 MiB | 393 MiB | OOM |
| 3 261 | 6 522 | 22 579 MiB | 393 MiB | OOM |

Fit:

```text
worker VRAM  ≈ 0.62 GiB + 15.8 MiB × members + 0.75 MiB × envs
```

Arithmetic only, **not measured**. The fixed term comes from the two uncensored
144-member rows: the per-env slope is `(5 057 - 3 123) / (2 880 - 288) = 0.746
MiB/env`, and `3 123 - 288 × 0.746 - 144 × 15.8 = 632.9 MiB = 0.618 GiB`. That
fit models the 1 000-member row at 17 179 MiB against 17 171 MiB measured. The
2 000- and 3 261-member rows are censored at the card limit (22 571 MiB), so
they cannot refine the fit.

At 3 261 members the per-member term alone is 50.3 GiB per process, so **no k
fits**. The per-member term is consistent with one copy of the teacher actor per
member (4.0 M parameters at fp32 ≈ 16 MB). A single-process parent grows the
same way, because it holds one bank for the cohort it labels rows with.

## Goal

Run one cohort of all 3 261 clips on 7 worker cards plus one trainer card, by
giving each worker a disjoint subset of the clips, and by keeping the parent's
aggregation metadata-only.

Target: 466 members per worker → 7.19 GiB of bank + 0.62 GiB base → 7.81 GiB,
leaving 12.19 GiB of a 20 GiB card → about 16 600 envs per worker → `k ≈ 35`
envs per clip, with the full cohort covered. `k = 36` models at 20.10 GiB, right
at the 20 GiB bound, so the bound is `k ≤ 35`. Arithmetic only, not measured.

## Design

### 1. Global clip identity

`MotionClipSpec.motion_id: int | None = None`, honoured by
`MotionLibrary.from_clips` (`motion_library.py:294` currently forces
`motion_id=position`). Validate non-negative and unique within a library.
`teacher_code` already comes from the cohort-wide index
(`multi_motion.py:263`), so only `motion_id` needs the override.

`plan_multi_motion` (the module's build plan) gains `motion_ids:
Mapping[str, int] | None = None`; absent means the current positional behaviour.
The library translates a clip id to its own storage slot on every per-row
lookup, so a slice labels its rows with cohort-wide ids while reading only its
own tensors.

### 2. Shard plan

New pure function beside `worker_specs` (`worker.py:1000`):

```python
def clip_shard_plan(
  cohort, teacher_ids, worker_devices, *, envs_per_motion
) -> tuple[ClipShard, ...]
```

`ClipShard` carries `worker_index`, `device`, `teacher_ids` (its subset, in
cohort order), `motion_ids` (global ids), `num_envs = envs_per_motion ×
len(subset)`. Partition is contiguous and balanced: sizes differ by at most one,
in cohort order, so it is deterministic and reproducible.

Refusals before any environment exists: an empty cohort, more shards than
clips, `envs_per_motion < 1`, a repeated device, or a clip assigned twice.

### 3. Worker setup

`CohortSetup` gains `motion_ids: tuple[tuple[str, int], ...] = ()`, carried by
value with `teacher_ids` as today, so a worker builds only its subset and labels
rows with global ids.

The per-worker frozen teacher bank is subset the same way, because the measured
per-member term (15.8 MiB x members = 50.3 GiB at 3 261 clips) is one actor per
member in that bank. `build_cohort_teacher_bank` takes the slice, registers only
its modules, and keeps the cohort-wide code table: `teacher_code`, `motion_id`,
routing, and the replay partition are unchanged, and a code the shard does not
hold raises `TeacherBankError` instead of labelling with another teacher. A
build with no clip-id mapping keeps the whole cohort's bank.

### 4. Parent aggregation, metadata only

Replace `_require_one_shard_contract` (`distill.py:1670`) with a union check:

* everything per-shard by design stays excluded (seed provenance, row slots);
* schema, body selection, anchor, phase policy, reset contract and the audited
  contract must still agree across shards;
* the shards' `motion_id` sets must be **disjoint**, and their union must equal
  the requested `teacher_ids` with the requested `motion_id → teacher_id`
  mapping;
* the returned object carries merged clip metadata only: `motion_id`,
  `teacher_id`, `frames`, `fps`, `sampling_weight`.

The parent then builds replay `weights`, `frame_counts`, `teacher_codes` and the
identity's clip mapping from that metadata, without resolving teachers or
building a `MotionLibrary` with tensors. `cohort_identity_from_parts` needs a
`library` argument today; give it a metadata-carrying stand-in whose
`clips`/`mapping_digest()` are the merged view, and keep the digest identical to
a single-process run of the same recipe.

### 5. CLI and resume

`--shard-clips` (boolean, default off). It requires `--worker-devices` and
`--envs-per-motion`, and refuses `--num-envs`. `_workers_identity` records the
plan: per worker its device, its teacher ids and its env count. A resume must
reproduce the plan exactly; a different partition is a refusal, not a silent
re-shard.

Default behaviour is unchanged when the flag is absent.

## Tests

| Test | Asserts |
| --- | --- |
| plan covers the cohort | every requested clip appears exactly once, sizes differ by ≤ 1, order matches cohort order |
| env allocation | each shard's `num_envs == envs_per_motion × len(subset)`; unequal totals across shards are accepted in this mode only |
| refusals | empty cohort, more shards than clips, repeated device, duplicate clip, `--num-envs` together with the flag |
| library ids | explicit `motion_id` is honoured; duplicate or negative ids are refused |
| union contract | overlapping shards are refused; a shard that omits a clip is refused; a shard with a different schema is refused |
| parent weights | replay weights and frame counts cover all 3 261 members from metadata alone |
| unequal shards | three replies of 6, 6 and 4 rows per tick over 2 ticks merge tick-major; each per-tick block is the sum of the replies' that tick; a reply whose `batch_size` disagrees with `ticks × its rows_per_tick` is refused |
| parent teacher bank | the clip-sharded builder hands its trainer `teacher=None` and builds no cohort bank |
| resume | the recorded plan round-trips; a changed partition is refused |
| regression | with the flag absent, every existing cohort test is unchanged |

## Verification on the server

1. Two workers, `k=4`, 200-clip subset: each worker logs its own clip count, the
   run completes one iteration, and the parent's VRAM stays flat.
2. Seven workers, full 3 261 clips, `k=32` (14 912 envs, 18.73 GiB modelled):
   peak ≤ 20 GiB per card, all clips covered, two iterations complete, iteration
   time recorded. The target is `k=32` because `k=36` models at 20.10 GiB, right
   at the bound.
3. Only after that: the real VAE run.

## Out of scope

* Multi-trainer support. The trainer stays single-device.
* Reducing the per-member cost itself (fp16 actors, lazy teacher construction).
  Each slice now pays that cost once for its own members instead of once for the
  whole cohort, but the cost per member is unchanged. Worth a separate change
  later. The clip-sharded parent builds no teacher bank at all: it labels no
  row, so its trainer gets `teacher=None`.
* Any deployment, bundle export, or controller change.
