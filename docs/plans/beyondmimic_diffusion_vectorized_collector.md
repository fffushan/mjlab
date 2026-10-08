# X2 Diffusion Vectorized Collector

## Status

The batched collector path is implemented.

The single-row collector path remains available.
It remains the reference path.

The batched path uses one shared environment.
It uses independent row slots.
It batches policy calls, OU state updates, qualification state, evidence, and row data.

The implementation does not change the row schema.
It does not change qualifier thresholds or accepted-prefix rules.
It does not change the frozen D0 or D1 contracts.

## Implemented

Changed files:

- `src/mjlab/tasks/tracking/diffusion/adapter.py`
- `src/mjlab/tasks/tracking/diffusion/collector.py`
- `src/mjlab/tasks/tracking/diffusion/policies.py`
- `src/mjlab/tasks/tracking/diffusion/qualification.py`
- `src/mjlab/tasks/tracking/diffusion/runtime_x2.py`
- `src/mjlab/tasks/tracking/diffusion/runtime.py`
- `src/mjlab/tasks/tracking/diffusion/state.py`
- `src/mjlab/tasks/tracking/diffusion/__init__.py`
- `src/mjlab/scripts/diffusion.py`
- `tests/test_tracking_diffusion_vectorized_collector.py`
- `tests/test_tracking_diffusion_collector.py`
- `tests/test_tracking_diffusion_cli.py`

The new batch observations preserve aligned `WorldState` rows.
The VAE and PPO wrappers accept selected row indexes.
The collector calls only the policy that owns each selected row.

`BatchedOUProcess` keeps one NumPy random stream and one AR(1) state per slot.
The state update is batched across selected rows.

`BatchedQualificationTracker` stores counters and verdict state in `[N]` arrays.
The tracker keeps the scalar qualifier rules.

`TrialSlotScheduler` fills empty slots after completion, failure, or termination.
It admits an OU trial only after its own clean result is available and qualified.
It skips only the OU partner of a failed clean trial.

The runtime can build `X2MultiMotionEnvironment`.
It uses `MultiMotionCommand` row pinning.
It keeps per-row clip lengths and reference frames.

The CLI uses one batched store append for a batched collection.
The existing append-only shard format remains unchanged.
The existing dataset code still groups by `group_key` and row identity.
The row identity remains `(run_id, env_id, episode_id, segment_id, tick)`.
The batched tests verify that rows from multiple trials share the configured pair group.
The wider dataset tests pass.

## Pair-local ordering

The old global rule is removed.
A clean trial for pair B may run after the OU trial for pair A.

The pair-local checks remain in `DiffusionCollector.collect`.
An OU trial must have its own pair id, initialization id, matching fingerprint,
and qualified clean result in `_initial_snapshots`.

`_validate_trial_pairs` now allows interleaved complete pairs.
It still requires each OU item to follow its own clean item in the schedule.
`build_trials` emits each clean item followed by its own OU item.

The old global rule protected the process-wide snapshot map.
It also prevented an OU trial from finding an absent clean snapshot.
The fingerprint and qualification checks protect this actual dependency.
The global rule was not required for unrelated pairs.

## Profile of the current single-row collector

Command:

```sh
cd /home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006
PYTHONPATH=/home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006/src \
  /home/agiuser/projects/mjlab/.venv/bin/python /tmp/profile_current_collector.py
```

Configuration:

- Local device: `cuda:0`.
- GPU: RTX 5080 Laptop GPU.
- Environment rows: `1`.
- Motion: `tennis_000`.
- Seed: `0`.
- Start frame: `45`.
- The clean trial ran first to capture the OU snapshot.
- The reported trial was the OU trial.
- Control steps: `250`.
- The current VAE loader kept the VAE model on the CPU.
- Environment stepping used `cuda:0`.

Observed output:

| stage | calls | total ms | ms per call | ms per control step |
| --- | ---: | ---: | ---: | ---: |
| observation build | 250 | 380.143719 | 1.520575 | 1.520575 |
| VAE forward | 250 | 1283.589955 | 5.134360 | 5.134360 |
| OU sample | 125 | 3.303003 | 0.026424 | 0.013212 |
| `env.step` | 250 | 2890.430386 | 11.561722 | 11.561722 |
| evidence build and parse | 500 | 204.356448 | 0.408713 | 0.817426 |
| qualification tracker | 250 | 9.369286 | 0.037477 | 0.037477 |
| row assembly | 250 | 5.532506 | 0.022130 | 0.022130 |
| store write | 1 | 33.885239 | 33.885239 | 0.135541 |

The measured trial elapsed time was `4.864097124002001` seconds.
This is `19.456388496008` ms per control step.

Evidence timing includes the runtime post-step evidence conversion and the
scalar evidence parser.
Store timing is one NPZ append divided by 250 steps.
These numbers are measured values.
They are not estimates.

## Batched throughput and memory

Command:

```sh
cd /home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006
PYTHONPATH=/home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006/src \
  /home/agiuser/projects/mjlab/.venv/bin/python /tmp/batched_throughput.py
```

The run used this configuration:

- Local device: `cuda:0`.
- GPU: RTX 5080 Laptop GPU.
- One shared `X2MultiMotionEnvironment`.
- Three reference clips: `tennis_000`, `tennis_001`, `tennis_002_ss`.
- `32` parallel rows.
- `64` trials: `32` clean trials and `32` OU trials.
- `250` control steps per trial.
- `16,000` row transitions.
- No store write.
- The batched VAE remained on the CPU.
  This preserves the current single-row model device convention.

Observed output:

```text
rows=32
motions=3
trials=64
transitions=16000
elapsed_s=21.85116556399953
transitions_per_second=732.2263864203425
ms_per_row_step=1.3656978477499706
rss_mib=4495.95703125
gpu_allocated_mib=10.669921875
gpu_reserved_mib=26.0
```

The concurrent `nvidia-smi` sampler observed local GPU 0 memory from `15 MiB`
to a peak of `409 MiB` during this run.
The process peak resident set size was `4495.95703125 MiB`, or about `4.39 GiB`.
The throughput was `732.2263864203425` row-steps per second.

This is collector throughput.
It is not a closed-loop stability measurement.
It is not hardware or deployment evidence.

## Tests and validation

New focused tests:

- `test_slot_scheduler_replaces_rows_and_admits_pair_locally`
- `test_batched_rows_are_bit_identical_to_single_row_rows`
- `test_batched_store_receives_one_write_for_multiple_trials`
- `test_batch_policy_methods_accept_selected_subsets`

The existing pair-order tests were updated for pair-local ordering.

Focused command:

```sh
PYTHONPATH=/home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006/src \
  /home/agiuser/projects/mjlab/.venv/bin/python -m pytest \
  tests/test_tracking_diffusion_vectorized_collector.py \
  tests/test_tracking_diffusion_collector.py \
  tests/test_tracking_diffusion_cli.py -q
```

Result: **31 passed**.

Wider focused command:

```sh
PYTHONPATH=/home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006/src \
  /home/agiuser/projects/mjlab/.venv/bin/python -m pytest \
  tests/test_tracking_diffusion_*.py tests/test_diffusion_*.py -q
```

Result: **492 passed** and **3 warnings**.
The warnings are existing zero-variance divisions in
`test_diffusion_future_predictability.py`.

Ruff command:

```sh
/home/agiuser/projects/mjlab/.venv/bin/python -m ruff check .
```

Result: **passed**.

Changed-file format command:

```sh
/home/agiuser/projects/mjlab/.venv/bin/python -m ruff format --check \
  src/mjlab/scripts/diffusion.py \
  src/mjlab/tasks/tracking/diffusion/__init__.py \
  src/mjlab/tasks/tracking/diffusion/adapter.py \
  src/mjlab/tasks/tracking/diffusion/collector.py \
  src/mjlab/tasks/tracking/diffusion/policies.py \
  src/mjlab/tasks/tracking/diffusion/qualification.py \
  src/mjlab/tasks/tracking/diffusion/runtime.py \
  src/mjlab/tasks/tracking/diffusion/runtime_x2.py \
  src/mjlab/tasks/tracking/diffusion/state.py \
  tests/test_tracking_diffusion_cli.py \
  tests/test_tracking_diffusion_collector.py \
  tests/test_tracking_diffusion_vectorized_collector.py
```

Result: **12 files already formatted**.

The full-repository `ruff format --check .` command does not pass.
It reports four pre-existing files outside this change:

- `docs/plans/beyondmimic_diffusion_controller_handoff.md`
- `docs/plans/beyondmimic_diffusion_d2_implementation.md`
- `src/mjlab/scripts/distill.py`
- `src/mjlab/tasks/tracking/distillation/vae_config.py`

Those files were not reformatted.

The type check used for the changed diffusion source passed:

```sh
/home/agiuser/projects/mjlab/.venv/bin/python -m ty check \
  src/mjlab/tasks/tracking/diffusion src/mjlab/scripts/diffusion.py
```

## Bit-identity result

The test `test_batched_rows_are_bit_identical_to_single_row_rows` passes.
It compares all tested scalar and batched row fields.
It uses fake scalar and batch environments and a deterministic fake model.
It proves the collector arithmetic, row indexing, OU stream for slot zero,
qualification state, handoff, and row assembly against that test double.

It does **not** prove end-to-end bit identity on the real X2 runtime.
A real-runtime check was run with one shared row, motion `tennis_000`, seed `7`,
start frame `45`, four control steps, and the local frozen VAE/PPO assets.

Command:

```sh
cd /home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006
PYTHONPATH=/home/agiuser/projects/worktrees/mjlab/diffusion-vectorized-collector-20261006/src \
  /home/agiuser/projects/mjlab/.venv/bin/python /tmp/actual_eq.py
```

Observed result:

```text
MISMATCH 0 clean_action 0.22905927896499634
[-0.63825113  0.80064875  0.73926336]
[-0.59667838  0.62846529  0.72523445]
```

The first differing field is `clean_action`.
The check stops at that field.
It does not claim identity for later fields.

The scalar run uses `X2MotionEnvironment` and scalar `FrozenVaePolicy.action`.
The batched run uses `X2MultiMotionEnvironment` and `FrozenVaePolicy.batch_action`.
The real-runtime check therefore found a difference between the scalar and
batched action results.
The source of the difference was not isolated between complete observation
parity and batched model evaluation within this scoped verification.
No tolerance was added.
The real-runtime bit-identity deliverable is incomplete.

## Not established

- Real-runtime bit identity is not established.
- The source of the real-runtime `clean_action` mismatch is not established.
- A full production CLI collection with the new multi-row factory was not run.
- No closed-loop quality, simulator stability, hardware, deployment, or release
  evidence is established by this change.
- Full-repository format cleanliness is not established because four unrelated
  pre-existing files fail `ruff format --check .`.

## Commit

The change is committed locally on branch
`fxy/diffusion-vectorized-collector`.
It is not pushed.
