# BeyondMimic diffusion D1 — implementation and review

Status: implementation and independent review authorized by the user on
2026-09-29. All delegated stages use **`lingzhi/gpt-5.6-luna:max`**.
The initial workflow `40306ef1-0d8f-4a5b-ac26-1ccc9bc0a688` was **stopped on
session reload** after data-core completed. Its collector child
`48ee46f7-ec61-4276-9931-8608b6a30fc9` was stopped mid-edit, not resumable;
`adapter.py` is partial. The user requested continuation. Parent inspected the
stopped-child status/transcript, worktree, pre-existing file hashes and empty
index; resume only the remaining stages with a new same-protocol workflow
**`8dd7ca26-9935-4579-8a55-6606b52ef239`** (collector recovery → integration → two
fresh reviews; resumed via `recovery-workflow.js`).
Completed data-core handoff:
`/home/agiuser/.pi/agent/sessions/--home-agiuser-projects--/subagent-artifacts/outputs/40306ef1-0d8f-4a5b-ac26-1ccc9bc0a688/mjlab-d1/data-core.md`.
This is **not completed D1**: real collection/qualification evidence and the
full dataset remain separate from code, mock tests and review.

## Scope and references

- [Frozen D0 contract](beyondmimic_diffusion_d0_contract.yaml), SHA256
  `42694cdd3acd72afd2d234a85ca29e9c7a646d64df831021c8dd43e55f3f73e7`.
- [D0 closeout](beyondmimic_diffusion_d0_validation.md#11-d0-closeout--2026-09-29)
  and [overall plan](beyondmimic_diffusion_reproduction.md).
- Implement collection, qualification, shard storage, window construction,
  grouped splits, train-only statistics/projection, replay and CLI validation.
  Do not implement the D2 denoiser/trainer, change the frozen VAE or PPO policies,
  or retune accepted production tracking/velocity/distillation configurations.
- This delegated wave may run bounded CPU unit tests and read-only CPU artifact
  inspection. GPU/simulator execution, remote access and the 54-trial pilot are
  held for parent-managed resource/validation approval after code review.
  No bulk collection, training, installs, commit, staging, reset, checkout or push.
- `execution_authorized: false` in the D0 YAML is the immutable design-closeout
  record, not a request to rewrite the frozen file when later work is authorized.
  Runtime CLI opt-in/budget/preflight must be separate and default to no launch.

## Implementation topology and ownership

**Multi-seam:** offline trajectory/data integrity and online controller/episode
lifecycle have independent interfaces and tests. Integration/CLI is a third seam.
The existing checkout is dirty and contains uncommitted D0 artifacts; do not
stash/commit it to obtain a managed worktree. Use **sequential, exclusive component
owners in the same checkout**, never concurrent writers. Each component emits a
durable API/validation handoff before the next starts. The integration owner
starts only after both component handoffs. Read-only reviewers run after writing
has stopped; the parent owns final acceptance and any follow-up disposition.

Repo/cwd: `/home/agiuser/projects/mjlab`; branch `fxy/test/tracking-exp`;
initial HEAD `107fe65bff995186ae7d61c9b1632a1f1947caf0`. A later unrelated HEAD
change must not be attributed to this session. Baseline dirty-tree evidence is
under ignored `logs/diffusion/d1-implementation-20260929/`.

| Stage | Exclusive ownership / decision | Gate and handoff |
|---|---|---|
| Data core writer | New `tracking/diffusion/{__init__,contract,state,projection,storage,dataset}.py` and matching core tests; typed raw-row/shard API, frames, boundaries, splits and statistics | CPU adversarial tests; durable public API/shape/ownership handoff; do not implement a simulator driver or CLI |
| Collector writer | New `tracking/diffusion/{adapter,policies,qualification,collector}.py` and matching lifecycle tests; actual frozen-policy loading, control ownership, post-step evidence and trial budgets | Consume core API; fake-environment/replay/terminal tests plus bounded CPU artifact checks; provide actual-runtime adapter with no rollout execution |
| Integration writer | New `scripts/diffusion.py`, collection config, CLI tests, this D1 implementation record and scoped changelog entry; minimal integration fixes in new D1 modules only | Connect components to executable preflight/collect/build/inspect paths, run bounded focused tests, document reviewed-vs-unrun gates |
| Fresh review: data | Read-only core, shards, windows, frame math, split/statistics/provenance review | Concrete P0/P1/P2 findings with source/test proof; verdict and missing evidence |
| Fresh review: lifecycle | Read-only runtime adapter, VAE/PPO handoff, terminal evidence, budget and CLI safety review | Concrete P0/P1/P2 findings with source/test proof; verdict and missing evidence |

The filename partition is a coordination boundary, not a requirement to create
empty abstractions. Ask the parent before extending it or changing shared APIs.
Review/test artifacts use tool-bound output paths; no scratch reports at repo root.

## Acceptance checklist for implementation (not dataset acceptance)

- Typed, fail-closed contract loader and exact artifact/body/action identity.
- Raw pre-action state/latent/clean action/OU/executed action and post-step evidence
  are aligned; PPO rows carry no fictitious latent; final-step failure precedes
  success; clip boundary causes a continuous deployed-rule handoff, not a reset.
- Every retained training window has 41 contiguous, same-controller VAE rows and
  correct current index 8. No reset, teleport, policy-switch, missing terminal,
  NaN, padding or unqualified tail becomes accepted data.
- Physical tilt >70 degrees and the frozen reference-error guards keep their
  distinct meanings; reference-only guards stop at PPO ownership. Runtime flags
  and matching terminal evidence are recorded rather than inferred from survival.
- Bounded append-only shards, safe schema/hash checks, failed-attempt summaries,
  deterministic paired clean/OU initializations and full phase-coverage accounting.
- Group before windowing/statistics; keep clean/OU families and duplicate initial
  states together. No validation/test data influences fitting. Empty coverage is
  explicit rather than silently resampled. Persist statistics/projection hashes.
- A usable real adapter and CLI, not only protocol stubs or fake-data demos.
  Reads of accepted distillation APIs and X2 controller endpoint semantics are
  allowed; changes to those existing production implementations require escalation.
- Scoped pytest with visible test names and fail-fast, numerical tolerances from
  D0, formatting/lint/type checks where available. Use `uv run --no-sync`.
  Inspection ~30 s, small tests ~60 s, focused suites <=120 s; diagnose a timeout
  instead of repeating/increasing it. No unmanaged long processes.
- Reviewer findings must distinguish current defects from unperformed simulator
  validation. CPU/mock success must not be represented as an actual pilot.

- Added a fail-closed integration CLI at `mjlab.scripts.diffusion` with
  `preflight`, `collect`, `build`, `stats`, `inspect`, and offline `replay`
  commands. Help and all offline commands avoid environment construction;
  collection requires `--execute`, a concrete `MODULE:FUNCTION` runtime
  factory, exact frozen artifact preflight, and bounded trial/transition
  budgets. The runtime factory returns a `RuntimeBundle` containing a real
  `DiffusionCollector` and trial specifications, while the existing callback
  adapter remains the simulator binding seam.
- Added `configs/diffusion/x2_50hz_collection.yaml` as a non-authorizing pilot
  input record and added CPU CLI tests. This implementation wave does not
  launch the simulator, collect a pilot, train a model, or claim D1 dataset
  completion; the recovery ONNX and 50 Hz runtime parity remain unverified.

## Integration-wave validation record — 2026-09-29

The integration seam connects the data-core and collector APIs without changing
frozen D0 files or production policies. `preflight` loads and hash-checks the
frozen contract and both selected artifact bytes. `build` assigns groups before
indexing windows, `stats` fits and persists train-only projection/statistics
bundles, `inspect` validates bounded shard loading, and `replay` checks stored
action arithmetic and VAE/PPO latent ownership without starting a simulator.
`collect` is deliberately fail-closed and requires explicit operational opt-in,
a runtime factory, exact artifacts, and bounded budgets.

No simulator, ONNX execution, pilot, GPU, remote job, training, deployment, or
hardware validation was run. Passing CPU tests is integration evidence only and
is not D1 dataset completion or reviewer acceptance. Parent-managed next gates
are concrete runtime-factory binding, artifact parity/replay checks, and the
resource-approved 27-clean / 27-OU pilot. The later monitored command shape is
published here but was not run:

    uv run --no-sync python -m mjlab.scripts.diffusion collect \
      --config configs/diffusion/x2_50hz_collection.yaml \
      --execute --runtime-factory <approved.module:factory> \
      --output <pilot-shards> --report <pilot-report.json> \
      --max-trials 54 --max-control-transitions 13500

## Lifecycle fix pass — 2026-09-29

The lifecycle review blockers were fixed in the owned adapter, qualification,
collector, policy, and CLI seams. Post-step bundles now carry the runtime
physical timestamp and post-action observation; qualification derives physical
fall/reference rejection from that post-step state and the row stores the final
parsed evidence. VAE reference metrics are required while VAE owns control,
and PPO recovery always builds the 102-D model input with the converted
endpoint previous action. Action metadata verification covers joint order,
scale, offset, and gains, with gains retained as an audited contract field.
Clean/OU trials carry pair and initial-state identities; the collector stores
and restores the clean snapshot, and the CLI rejects missing, mismatched, or
out-of-order pairs. The collector accepts a remaining transition budget and
stops before an action beyond the cap; the CLI checks wall time and output size
between trials.

There is still no repository-local concrete X2 simulator factory in this wave.
The CLI therefore fails closed unless a parent-managed factory explicitly
returns a verified `RuntimeBundle` with `environment: agibot_x2`, a bounded
collector interface, and concrete runtime provenance. The callback adapter is
only a binding seam and is not pilot evidence. Wall/output checks are
in-process checks between completed collector calls; they do not hard-preempt
a hung simulator step. A parent-managed monitor is required for the D0 wall,
inactivity, and process/output interruption guarantees. No simulator, ONNX
execution, pilot, GPU, remote, training, deployment, or hardware validation
was run; those remain parent-managed pilot gates.


After parent acceptance and resource approval: execute the frozen 27-clean /
27-OU pilot with its stop rules. Inspect replay parity, handoff, qualification,
coverage and throughput before requesting a bulk collection budget. D1 ultimately
includes the actual qualified dataset and exported split/statistics artifacts;
finishing this implementation/review wave alone is not D1 completion.

## Fix waves 2 and final — 2026-09-29

Independent review kept returning BLOCK across three rounds plus two re-check
rounds, so fixes were applied in waves. Every finding was verified against
source by the parent before acceptance; child "no blocker" self-reports were not
treated as acceptance.

**Wave 2 (`088313d4-b208-4376-ac83-f809ef62f565`), 62 CPU tests.** The frozen
VAE is now loaded through the correct existing loader for the `version=3`
`mjlab-m4-cohort-distillation` artifact (`load_cohort_member_inference` against
the pinned 3-teacher manifest), instead of the version-1-only loader that cannot
open it at all. Recovery ONNX metadata is decoded from the real exporter schema
(`joint_names`, `action_scale`, `default_joint_pos`, `joint_stiffness`,
`joint_damping`, CSV-encoded) with stiffness/damping kept separate. VAE
reference metrics are required while VAE owns control; snapshots are keyed by
`(pair_id, initial_state_id)` with full initialization fingerprints; the whole
clean phase must precede OU and OU is blocked when its clean partner did not
qualify/handoff; `max_envs != 1` fails closed; duck-typed/self-attested runtime
bundles are rejected; `TerminalEvidence` booleans are strict and malformed NPZ
containers raise `StorageError`; provenance aliases link paired/duplicate
families.

**Final wave (`62d9286a-78ee-46da-8931-ec783840b7c8`), 65 CPU tests**, fixed the
last four verified defects: PPO previous-action now patches the actions field at
`68:99` and preserves the zero `command` at `99:102` (it previously patched
`71:102`, corrupting `command`); a missing post-step observation fails
physical-fall qualification in **both** phases instead of skipping the tilt
check; collector-generated group keys still receive pair/initial-state provenance
aliases while genuinely explicit keys stay isolated; and env setup/snapshot
failures still invoke `reset_after_trial()`.

**Parent follow-up, 66 CPU tests.** The final re-check found that
`CallbackEnvironmentAdapter` declared `trial_reset_fn` but never exposed
`reset_after_trial()`, so any collector driven through the advertised callback
bridge would raise `AttributeError` in cleanup. The parent added the one-line
delegation and a focused adapter test, then re-ran the suite (66 passed, scoped
ruff/format clean).

### Honest status

- Code seam: implementation plus CPU/fake-environment tests. The focused
  diffusion suite passes 66 tests; scoped ruff/format/pyright are clean; no files
  are staged and the frozen D0 contract is byte-identical.
- Not demonstrated anywhere: simulator handoff, real collection, ONNX execution
  or decoder parity, sustained 50 Hz, the 27-clean/27-OU pilot, any qualified
  dataset, training, deployment or hardware.
- Explicit parent-managed prerequisites (fail-closed, not faked): the concrete
  repository-local X2 runtime factory, and hard process-level interruption of a
  hung simulator step.
- A concurrent workspace session committed `x2_tennis_mixed_v3.yaml` at HEAD
  `376a94943`; it is unrelated to D1 and does not affect the frozen 3-teacher
  cohort contract.
- **D1 is not complete.** D1's deliverable is the qualified dataset, which still
  depends on the parent-managed simulator pilot run under its stop rules.

## Runtime integration and the pilot run — 2026-09-30

The concrete runtime the offline code was waiting for now exists:
`src/mjlab/tasks/tracking/diffusion/runtime_x2.py` implements the
`--runtime-factory` entry point (`build_x2_runtime`) over one single-motion
tracking environment per clip, the frozen version-3 cohort VAE, and the frozen
zero-command recovery policy.  Evidence and full detail:
`logs/diffusion/d1-runtime-20260929/README.md`.

Defects found only by running the simulator, all fixed:

1. `reset_to_frame` leaves the anchor-relative reference cache stale
   (`command.update_relative_body_poses()` is required), which made the tracking
   task's `ee_body_pos` termination fire spuriously.
2. The D1 end-effector guard compared `body_pos_w` instead of the task's own
   `body_pos_relative_w`.
3. The recovery observation omitted the training environment's **biased**
   relative joint position and its IMU sensor sources; parity against
   `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation` is now within every
   declared corruption bound.
4. The tracking task's `anchor_pos`/`anchor_ori`/`ee_body_pos` terminations
   stayed active after the handoff, so the standing policy lost the reference
   and the episode ended.  The contract marks them
   `active_after_ppo_handoff: false`; they are now suspended at the handoff and
   restored per trial.
5. `WindowIndex._rows` re-scanned and re-decompressed every shard per window, so
   `fit_training_statistics` was quadratic (`stats` exceeded 900 s on 1830
   windows, 16.8 s after one-pass caching).
6. `CollectionRequest`/`RuntimeBundle` moved to
   `mjlab/tasks/tracking/diffusion/runtime.py`: under `python -m` the CLI script
   is imported twice, so a factory's `isinstance` check failed.

The frozen 54-trial pilot was then run (`--execute`, local RTX 5080, laptop
unplugged, 4 min 16 s).  All 27 clean trials qualified (18 `vae_then_standing`,
9 `vae_only`) and the OU phase stopped at its first pair with
`stop_reason="ou_blocked_clean_partner"`, because the `frac=0.1` start cannot
reach the clip end within `control_steps: 250` (407/306/288 steps remain) and
`DiffusionCollector` blocks an OU trial whose clean partner has no handoff.

So the run produced a qualified **clean VAE dataset** (6750 rows, 1830 windows,
train-only statistics) but **no OU trials**.  The contract's own stop rule is
per *motion*, and all three clips hand off at `frac=0.5`/`0.8`, so the contract
supports proceeding to OU; the CLI's per-pair break and
`_validate_trial_pairs`' complete-matrix requirement are stricter.  Resolving
that (run as-is and accept a clean-only pilot, amend the start fractions, or
relax the per-pair OU gate) needs an explicit user decision.

Still not demonstrated: the OU perturbation axis, sustained real-time 50 Hz, and
any hardware result.  **D1 remains incomplete**: the qualified dataset exists for
the clean phase only.

## OU gate amendment and the complete pilot — 2026-09-30

The frozen OU precondition (a *clip-end handoff* from the clean partner) was
wrong, and it was the reason the first pilot run produced no OU data.  The
paper's supplementary "Diffusion Dataset Collection" defines a successful
rollout as one that does not fail before 5 s, and names no clip-end
requirement; `vae_only` is one of the contract's two valid outcome labels.  The
perturbation is applied only over the retained 2.5 s interval, so a rollout that
never reaches the clip end still supplies its full perturbed interval.

Two changes, authorized by the user on 2026-09-30:

1. `collector.DiffusionCollector.collect` requires only that the clean partner
   **qualified**; `handoff_step is None` no longer blocks its OU counterpart.
2. `scripts/diffusion._command_collect` **skips and reports** a pair whose clean
   partner failed (`ou_skipped_clean_partner`) instead of stopping the run, and
   stops only if no pair can run OU (`ou_phase_unavailable`).

The frozen D0 YAML is unchanged (`42694cdd…`): it is the contract identity
recorded in every shard manifest, so amending it would invalidate the stored
evidence.  This section is the amendment record.

The complete pilot then ran clean: **54/54 trials qualified** — 27 clean
(9 `vae_only`, 18 `vae_then_standing`) and 27 OU with the same distribution —
13500 transitions, `partial: false`, 10 min 14 s wall.  Offline: 13500 rows, 54
shards, `replay` ok, 3660 windows (train 2640 / test 1020 / validation 0),
train-only statistics from 2640 windows.  Evidence:
`logs/diffusion/d1-pilot-20260930b/README.md`.

Known limitation: `grouped_split` groups by the collector's generated
`{motion_id}:{start_frame}` key, which omits the seed, so all three seeds of a
start frame (clean + OU) form one group.  That gives 9 groups and an empty
validation partition.  This grouping is the *conservative* choice (same-frame
rollouts are near-duplicates and stay in one split); a validation split would
need pair-family grouping (`motion:seed:start_frame` → 27 groups).  Deferred to
the bulk collection, where the group count supports both.

Status at that point: **D1's pilot dataset existed** — a qualified 54-trial
clean+OU dataset with indexed windows and train-only statistics.  The ~100-fold
bulk coverage was still outstanding; it is closed out in the next section.

## Bulk collection and D1 closeout — 2026-09-30

The bulk run executed on a managed-container GPU (RTX 4090) as eight parallel
single-env worker processes, then a merge, then an offline pass, unattended.

- **Collection:** 1026 trials attempted (513 pair families = 3 clips x 3 start
  fractions x 57 seeds); 1003 collected and 23 OU partners skipped because their
  clean partner failed.  248420 transitions, every worker `partial=False`,
  `stop_reason=None`, and zero errors in any `collect-w*.log`.  Qualified clean
  trials: 319 `vae_then_standing`, 171 `vae_only`, 23 failed.  Parallelism gave
  ~3.6 s/trial against 9-12 s single-process and moved the GPU from 26% to 99%
  utilisation, so the run became GPU-bound rather than host-bound.
- **Merged store:** 248420 rows (171818 VAE / 76602 PPO) in 64 shards, 376 MB,
  unique identities, `replay` `ok` with no errors.
- **Offline pass:** `build` produced **67133 windows — train 53087 / validation
  6764 / test 7282** across 410/51/52 groups, `stats` fit train-only statistics
  from the 53087 train windows into `projection.npz`, and `inspect`/`replay`
  confirmed the store.  All four steps returned rc=0.
- **Coverage:** accepted rows are 106567 (bulk) + 5838 (pilot) = **112405
  against the 111300 sizing target (101.0%)**.  The sizing rule is `total
  reference duration x 50 x 100`; see the open interpretation below.
- **Grouping limitation resolved:** `build_bulk_trials` sets an explicit
  pair-family `group_key` (`motion:seed:start_frame`) and a seed-bearing
  `initial_state_id`, so clean and OU share a split while seeds separate.  The
  validation partition is populated (51 groups) instead of floored to zero.
- **Evidence:** `logs/diffusion/d1-bulk-par/{RESULTS.md,README.md,driver.log,`
  `merge.log,offline/offline.log}` plus per-worker `report-w*.json`.

Three real defects surfaced during this milestone, all fixed and each covered
by a regression test or a re-run:

1. `AppendOnlyShardStore.append` called `identities()`, which re-read,
   re-hashed and decompressed every existing shard.  That made appending
   quadratic and re-read the whole store over the network filesystem on every
   trial.  The identity set is now seeded once and maintained incrementally
   (the manifest is still read from disk, so on-disk-rewrite integrity checks
   still observe the file).
2. `_read_shard` indexed `data[name]` inside the per-row loop, and
   `NpzFile.__getitem__` re-decompresses the whole array on every access: about
   4.5 MB per row, i.e. **18.6 GB for a 4096-row shard**.  The pilot's ~250-row
   shards hid this; the merged store's shards exposed it and it was what
   restarted the container (the cgroup showed `oom_kill 0` with freshly reset
   counters and the tmux socket directory was gone).  Arrays are now
   materialised once and reused: 50000 rows cost 0.99 GB, measured.
3. A stale first-generation `offline.sh` survived the kill of its child process
   (it had no `set -e` on its steps) and re-ran `inspect`, writing its own log
   lines into `offline/inspect.json`.  The offline pass was re-run clean and all
   artifacts re-hashed.

**Open interpretation (not a defect):** whether the 100-fold requirement means
all accepted rows (met, 101.0%) or the perturbation band alone.  The band is
about half the accepted rows (~48% of the target), and since clean and OU rows
are stored together with exact per-row `ou_noise` provenance, the clean/band
sampling mix is a trainer-side weight rather than a collection decision.  A
second batch would only be needed under the band-only reading.

Status: **D1 is complete.**  A qualified 1026-trial clean+OU dataset exists with
indexed windows, a populated validation split and train-only statistics; the
sizing target is met under the all-accepted-rows reading.  Still not achieved,
and deliberately out of D1's scope: sustained real-time 50 Hz control (the
unplugged loop runs ~31-41 ms/step against a 20 ms period), diffusion training,
hardware execution, and any closed-loop evaluation of the generator.
