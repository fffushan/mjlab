# Phase-balanced replay sampling — implementation plan

Status: **proposed; planning only**. No implementation, training, or collection
is authorized or performed by this document. This is an additive training-side
sampling change for the accepted M4 cohort path; it is not the diffusion
milestone and not a new objective.

Repository: `/home/agiuser/projects/mjlab`.
Branch: `fxy/test/tracking-exp`.
Planning baseline: `451d432a6` (HEAD at planning time). **The working tree
carries uncommitted standing-start work** (modified `distillation/*.py`,
untracked `reset_policy.py` and its tests). This plan's implementation touches
several of the same files; the user decides whether to land standing-start
first or implement on top of it.

Related documents:

- [VAE distillation architecture](beyondmimic_vae_distillation.md)
- [M4 implementation and acceptance](beyondmimic_vae_m4_implementation.md)
- [Reproduction review motivating this change](beyondmimic_reproduction_review.md)
- [Usage](../source/x2_tennis_distillation.rst)

Motivation (from the reproduction review): the reviewed BeyondMimic
reproduction equalizes training exposure over (motion × reference-phase) cells
inside its offline trainer. Our pipeline balances the *motion* mixture in
`BalancedReplayBuffer.sample` and controls phase coverage at collection
(uniform phase sampling at reset), but within a motion the replay draws rows
uniformly by density, so retained-row density skew translates directly into
gradient-mass skew: long-surviving segments of a clip accumulate more rows
than hard-to-reach phases. A phase-balanced draw equalizes expected gradient
mass per phase cell of each motion. This is static coverage balancing only —
it is difficulty-blind by design and must not become adaptive phase sampling
(`multi_motion.py` refuses adaptive sampling for multi-motion, and that
refusal stands).

## 1. Goal, authority, and non-goals

Change what a minibatch draw from `BalancedReplayBuffer` looks like *within*
each motion partition: from density-proportional to phase-cell-uniform, as an
opt-in constructor policy. The motion-level mixture (configured weights,
quota allocation, readiness contract) is unchanged. Default configuration
must reproduce today's draw distribution bit-for-bit.

In scope:

- An opt-in `phase_bins` policy on `BalancedReplayBuffer` (0 = disabled =
  current behavior).
- Two-stage per-motion draws: allocate counts across non-empty phase cells,
  then uniform draws within each cell.
- CLI exposure on the cohort train/eval paths and recording in the resolved
  configuration with strict-resume compatibility.
- Per-phase observability in the replay report.
- CPU tests and a bounded headless integration smoke (already-existing
  pattern).

Out of scope:

- The single-teacher `LabeledReplayBuffer` path (no partition structure, no
  `frame_counts` today; a follow-up if ever needed).
- Adaptive, loss-aware, or failure-weighted sampling of any kind.
- Changes to collection, phase-sampling-at-reset policy, model, objective,
  normalizers, teacher contracts, `ReplayPolicy` identity (§3.4), or control
  timing.
- Symmetry augmentation, diffusion work, GPU production runs, commits.

## 2. Verified existing behavior (implementation seams)

- `VaeDistillationTrainer.train_update` (`trainer.py:435`) is the only
  training-path caller of `replay.sample(minibatch_size,
  generator=self.replay_generator)`; the trainer holds no other sampling
  logic. The replay generator's state is checkpointed under the RNG bundle.
- `BalancedReplayBuffer.sample` (`balanced_storage.py:747`) draws in three
  documented stages: motion counts via `_draw_counts` (weights → floors +
  unbiased residual `multinomial`, `_draw_counts` at line 685), then one
  uniform `torch.randint`/`randperm` per motion in ascending motion order
  (`_partition_indices`), then a global row shuffle. Determinism contract:
  "deterministic for a given generator state" with this exact draw order; the
  buffer owns no RNG.
- Every partition row already stores `reference_frame`, validated at insert
  by `_validate_routing` (`balanced_storage.py:598`) to lie in
  `[0, frame_count)` for that row's motion — local clip frames, so
  phase = `reference_frame / frame_count` needs no new data and no wraparound
  handling.
- `frame_counts` is already a first-class, motion-aligned constructor argument
  of `BalancedReplayBuffer` and is used only for insertion validation today.
- `_active_indices(partition, device)` is the single ring→logical mapping
  shared by insertion, sampling, and persistence.
- Readiness fails closed: draws raise `ReplayNotReadyError` while any motion
  partition is empty; `replacement=False` requests that exceed a partition's
  occupancy are refused, never silently reweighted.
- Normalizer updates count every fresh raw row once on insertion
  (`test_tracking_distillation_balanced_replay.py:704`); draws never update
  statistics. A draw-distribution change must not alter this.
- The cohort CLI constructs the buffer in `scripts/distill.py` (`train`
  path ~line 842) with weights, `teacher_codes`, `frame_counts`; the resolved
  configuration already records `"replay": {"capacity": ...}` and trainer
  settings, and strict resume compares recorded policy against the live
  components (`require_same_replay_policy` for the partition policy).
- `ReplayPolicy` (`cohort_contract.py:689`) is part of `CohortIdentity` and
  its digest (`cohort_contract.py:905`). Adding fields there changes digests
  and breaks old checkpoint resume — see §3.4.

## 3. Design

### 3.1 Policy location: buffer-owned, trainer-transparent

`phase_bins` is a constructor argument of `BalancedReplayBuffer`
(int, default 0; validated positive when nonzero). The trainer, the
`ReplayBufferProtocol` signature, and `LabeledReplayBuffer` are untouched:
the trainer keeps calling `sample(batch_size, generator=...)` and the policy
lives where the partitions, `frame_counts`, and generators already live.

Rejected alternatives:

1. *Trainer-side weighted sampler* (the reproduction's approach): would pull
   sampling out of the buffer, break the buffer-owns-draws determinism
   contract, and require protocol changes for partition internals.
2. *Per-row inverse-cell-count weights with a single weighted draw*: awkward
   without-replacement semantics and asymmetric with the existing two-stage
   motion logic.

### 3.2 Draw algorithm (two-stage, mirroring `_draw_counts`)

With `phase_bins = B > 0`, for each motion's `c` rows in a draw:

1. Compute the cell of each retained row:
   `b_i = min(B - 1, floor(reference_frame_i / frame_count_m * B))`.
   Only *non-empty* cells participate; a cell with no retained rows cannot be
   drawn (balancing equalizes exposure among present data; it cannot create
   absent coverage — collection-side coverage remains a prerequisite).
2. Allocate `c` across the non-empty cells: equal floors
   `c // |C|`, then the residual rows by an unbiased random allocation among
   the non-empty cells — exactly the floor-plus-residual structure of
   `_draw_counts`, so odd and one-row batches keep cell-uniformity in
   expectation instead of always preferring the first cell.
3. Within each cell, draw uniformly: `torch.randint` with replacement, or
   `randperm`-sliced without replacement.
4. Concatenate cells in ascending cell order, then the existing global row
   shuffle.

Documented draw order becomes: motion counts → per motion in ascending order
(cell allocation, then per-cell index draws) → global shuffle. All randomness
flows through the caller's generator on the same device as today.

With `replacement=False`, the per-cell refusal rule mirrors the existing
per-partition rule: if any cell's allocated count exceeds that cell's retained
row count, refuse the whole draw before mutating any counter — never silently
reweight. The whole-batch refusal precedes any partition mutation, preserving
the current transactional contract.

With `phase_bins = 0`, `sample` executes today's code path exactly (identical
generator consumption), so every existing test passes unmodified.

Phase-bin lookup may be recomputed from `reference_frame` at draw time
(O(partition size) per draw, acceptable at default capacities) or cached as a
parallel per-slot tensor maintained at insert; implementation may choose,
stating which, with a preference for the insert-time cache only if draw-time
cost measurably matters.

### 3.3 Statistics and reporting

Extend `MotionReplayStats` with optional `retained_by_phase_bin` and
`drawn_by_phase_bin` (present only when `phase_bins > 0`), additive in
`as_dict` for the JSON report. This is observability, not identity: it
exposes the coverage/density skew that motivates the feature and lets a
future run decide whether a phase is collection-limited. The report remains
non-checkpointed state.

### 3.4 Checkpoint and resume compatibility (the one delicate decision)

`ReplayPolicy`/`CohortIdentity` are **not** extended: adding a field changes
`cohort_digest` and would invalidate every existing checkpoint. Instead,
`phase_bins` is a replay-construction setting recorded in the resolved
configuration next to replay capacity (`"replay": {"capacity": ...,
"phase_bins": ...}`).

Strict resume must then treat a missing `phase_bins` in an old checkpoint as
`0`. During implementation, read the resolved-config comparison seam in
`checkpoint.py`/`resume_cohort` and make the comparison
default-compatible for this one key (old checkpoint, new code, default 0 →
valid resume with unchanged behavior; new checkpoint, mismatched value →
refuse). If the comparison turns out to be whole-dict equality with no
per-key defaulting seam, extend it minimally for this key rather than
touching digests. Verify with a resume test against a checkpoint saved by
current HEAD code.

### 3.5 CLI

Add `--phase-bins` (default 0) to the cohort train command alongside
`--replay-capacity`, threaded into the `BalancedReplayBuffer` constructor and
the resolved-config record. The single-teacher command does not gain the flag
this milestone (its buffer has no partitions); refusing it there is a
documentation note, not code.

## 4. Implementation steps

1. `balanced_storage.py`: constructor argument + validation; two-stage draw in
   `sample`; per-cell refusal for `replacement=False`; per-phase stats;
   state_dict unchanged (policy is constructor state, not ring state).
2. `scripts/distill.py`: CLI flag, constructor threading, resolved-config
   recording, default-compatible resume comparison (with `checkpoint.py` seam
   adjustment if needed).
3. Tests (new file `tests/test_tracking_distillation_phase_balanced.py`,
   plus targeted additions to the balanced-replay and cohort-checkpoint
   suites where behavior borders meet).
4. Docs: usage section in `docs/source/x2_tennis_distillation.rst`, changelog
   entry, and a status update on this plan.

Ownership: small enough for direct implementation or a single bounded worker
seam; the user selects the execution mode and, for delegation, the worker
model. No concurrent writers with standing-start work on the same files (§0
caveat).

## 5. Tests and acceptance

- **Distribution property:** skewed fixture (e.g. one motion, bin 0 holding
  10× the rows of bin 9); many seeded draws with `phase_bins=10` → per-cell
  drawn counts equal in expectation within a stated tolerance; with
  `phase_bins=0` → density-proportional counts (today's behavior).
- **Motion-mixture invariance:** with balancing on, the motion-level mixture
  tests (`test_batch_size_one_and_odd_batches_keep_the_configured_mixture`)
  still hold; motion mixture is orthogonal to within-motion balancing.
- **Determinism:** identical generator state → identical batches; documented
  draw order (cell allocation before per-cell draws, ascending motion and
  cell order, final shuffle); `phase_bins=0` consumes the generator exactly as
  current code (bitwise-identical draws).
- **Fail-closed semantics:** `replacement=False` per-cell refusal leaves all
  counters and partitions untouched; readiness contract unchanged; invalid
  `phase_bins` values rejected at construction.
- **State/round trip:** state_dict keys and round-trip behavior unchanged;
  stats/report fields additive.
- **Trainer integration:** `train_update` draws through the balanced path;
  normalizer counting still happens on fresh rows only; a tiny CPU
  cohort-fixture training step with `phase_bins > 0` runs and reports
  per-phase stats.
- **Resume compatibility:** a checkpoint saved by current-HEAD code resumes
  under new code with default `phase_bins=0` (behavior identical); a
  mismatched nonzero value is refused.
- Existing suites pass unmodified (the default-off guarantee).
- Changed-file Ruff formatting/lint, `uv run ty check`, targeted Pyright per
  repository conventions.

## 6. Validation and resource budget

- CPU-only; small synthetic buffers; no GPU compilation, no environment
  rollout beyond the existing CPU cohort fixture pattern if needed.
- Timeouts per repository convention (30 s inspection / 60 s single test /
  120 s focused suite); pytest `-x -vv -o faulthandler_timeout=30` while
  debugging; preserve exit status when piping.
- The bounded headless integration smoke (if exercised) follows the existing
  cohort smoke pattern and needs separate user authorization, as in M4.

## 7. Definition of done

`--phase-bins` exists on the cohort train path, default-off preserves current
draws bitwise, phase-balanced draws equalize expected per-cell exposure for
non-empty cells with documented deterministic ordering, per-phase statistics
are reported, old checkpoints resume, and the tests/lint/type gates above
pass. No production training run, commit, or push is part of this milestone.
