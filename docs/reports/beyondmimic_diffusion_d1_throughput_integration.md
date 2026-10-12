# D1 throughput integration

Date: 2026-10-12 (+08).
Baseline: `e71a89ab2ece99f84872fd562f8374267ffc8624`.
Lane: `integration`. Worktree:
`/home/agiuser/projects/worktrees/mjlab/d1-throughput-20261012/integration`.

This report records component integration, not new component implementation.
It does not claim a measured speedup. CPU and scalar behaviour remain the
default, and every speed option is explicit.

## Scope

The integration applies four component patches onto the clean baseline, wires
the CLI, adds the reusable preparation and benchmark tools, and records the
runbook and changelog entries. Component writers own their source:

| Component | Patch | Integrated files |
| --- | --- | --- |
| statistics | `statistics.patch` | `diffusion/dataset.py`, `projection.py`, `state.py` |
| storage | `storage.patch` | `diffusion/storage.py`, `offline_validation.py`, `scripts/diffusion_merge.py` |
| tensor runtime | `tensor-runtime.patch` | `diffusion/adapter.py`, `runtime_x2.py` |
| streaming | `streaming.patch` | `diffusion/collector.py`, `collection_controls.py` |

All four patches applied with `git apply --binary` after `git apply --check`.
The component files are disjoint. No component source was rewritten.

## Integration changes

* `scripts/diffusion.py`:
  * `inspect` and `replay` take `--jobs` and use the bounded
    `offline_validation` shard-level paths. The JSON payload keys and the replay
    exit code are unchanged.
  * `build` takes `--jobs` and builds the index with `build_window_index`.
  * `stats` takes `--jobs`, `--batch-windows` and `--index`. Passing `--index`
    reuses the persisted, source-bound index instead of rebuilding it, and a
    refused foreign index exits 1 with a message instead of a traceback.
  * `collect` builds streaming controls for the batched path. Wall-clock and
    inactivity limits always apply, and the report writes `partial`,
    `stop_reason` and `unstarted_trials` truthfully. A partial batched run warns
    on standard error and exits 1, so an exit-status-only supervisor cannot
    mistake a truncated run for a complete one. Streaming publication, compact
    results, bounded appends and tensor dispatch are opt-in. `--policy-device`
    sets `MJLAB_D1_POLICY_DEVICE` for the runtime factory call only, and the
    report records the resolved `policy_device` and the requested
    `policy_device_setting`. `--progress PATH` appends one JSON progress record
    per event. The scalar path is unchanged except for the copied policy-device
    fields and the refusal of batched-only options.
  * `collect` refuses `--stream-rows`, `--compact-results`,
    `--max-rows-per-append`, `--tensor-dispatch` and `--progress` with exit 2
    when the request is scalar (`--max-envs 1`), before any environment or
    checkpoint work. Previously these options were accepted and silently
    ignored on the scalar path. Because the resolved device is now a top-level
    report field, the `streaming.policy_device` key, which held the raw flag,
    is gone from the batched payload.
* `scripts/experiments/diffusion_d1_throughput/prepare_dataset.py`: scalable
  preparation entry point. Per-input report gates, inspect and replay, a
  byte-preserving merge, the index build once with reload validation, the
  train-only statistics fit and the token-cache preparation. It stages new
  outputs and never starts training.
* `scripts/experiments/diffusion_d1_throughput/benchmark_matrix.py`: matched
  CPU/GPU VAE and 64/128/256-environment command generator. It does not run a
  benchmark.
* `docs/source/x2_diffusion_training.rst`: runbook sections for streaming
  controls, shard-level inspect/replay, index reuse, the preparation tool and
  the benchmark matrix.
* `docs/source/changelog.rst`: scoped `Added` and `Changed` entries.

## Narrow seam fix

The streaming lane test `test_tensor_dispatch_requires_the_full_tensor_api`
used the base `FrozenVaePolicy` as a policy without a device tensor API. The
tensor-runtime lane adds `batch_action_tensor` to that base class, so the test
assertion no longer held. The test now uses an explicit stand-in policy that
omits `batch_action_tensor`. The refinement adds
`test_tensor_dispatch_runs_the_base_frozen_vae_tensor_method`, so the base policy
method is now covered directly. The dispatch gate itself is unchanged.

## Validation

The rows below are measurements from the `refine-cli` worktree on 2026-10-12,
with `UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv`,
`PYTHONPATH=src` and `uv run --no-sync`. The tests are CPU-only. They are
focused groups, not the full suite.

| Command | Result |
| --- | --- |
| `pytest tests/test_tracking_diffusion_throughput_tooling.py -x -vv` | 23 passed (40 s) |
| `pytest -q` on statistics_throughput, fast_merge, parallel_validation, tensor_runtime, streaming_controls, throughput_tooling | 100 passed, 2 skipped (46 s) |
| `pytest -q` on the same six files plus dataset, storage, collector, cli, window_index_persisted, window_dataset, migration, token_cache_builder | 12 failed, 150 passed, 2 skipped (50 s) |

All 12 failures are in `tests/test_tracking_diffusion_window_dataset.py`. They
are missing-artifact failures, not code failures and not passing tests: the
fixture dataset `logs/diffusion/d1-pilot-20260930b` and its
`offline/assignments.json` do not exist in this isolated worktree. No failing
test exercises a changed seam.

The earlier candidate and component reports also list a full-suite row
(`pytest tests/ -k diffusion -q`, 848 passed, 27 failed, 3 skipped) and a
13-file row of 152 passed. Those counts are prior-candidate evidence. They were
not re-run during the refinement, and the 13-file count is not reproducible from
this worktree: the same file list measured 150 passed and 12 failed here. The
full-suite claim must be re-measured before it is quoted.

`ty check` and `ruff check` for the integrated tree are measured in the
refinement section below; the rows above are focused test groups only.

## Unmeasured speed

No speedup is measured. Startup, stepping, serialization, merge, validation,
indexing, statistics and cache work are not benchmarked here. The component
lanes report descriptive timings only. A speedup claim needs the matched
protocol that `benchmark_matrix.py` generates, on one machine with a warm page
cache. CUDA numerical parity, CUDA speed and network-attached storage (NAS)
throughput are not measured; the local NVMe timings are descriptive only.

## Unresolved semantic and parity questions

1. `statistics_sha256` changes for any fit through the batched accumulator,
   including `--jobs 1`: the accumulator updates once per `--batch-windows`
   windows instead of once per window, so the float64 reduction order differs
   from the historical per-window fit. The values equal the sequential fit to
   floating-point tolerance, not bit for bit. Existing frozen `projection.npz`
   files and the token caches keyed by the old statistics hash must be kept, not
   refitted or overwritten.
2. `compact_results` and `max_rows_per_append` require `stream_rows`, and the
   stored-output limit needs streaming to be measurable. These are collector
   contracts, not integration choices.
3. The batched collect path enforces the wall-clock and inactivity budgets that
   the request always carries. A run that previously ignored the budget can now
   stop, report `partial: true` and exit 1. This is the intended truthful report,
   but it is a behaviour change for batched runs. The scalar path still exits 0
   on a partial run, as it did before.
4. `jobs > 1` needs the `fork` start method. The token-cache, index, statistics
   and validation workers all use it.
5. The tensor dispatch requires all three device methods. The base
   `FrozenVaePolicy.batch_action_tensor` exists, but a CPU fake environment
   without `observe_vae_tensor_batch`/`step_tensor_batch` is still refused.
6. `--tensor-dispatch` with a CPU policy device is a reference and parity path,
   not a speedup: the device encode/decode runs on the same host as the NumPy
   path. CUDA numerical parity and CUDA speed are not measured. The CPU arm of
   the benchmark matrix is a matched control.

## Component limitations carried forward

* Statistics: the full production store is not tested (no access).
* Statistics: the projection bundle publish is now atomic (staged temporary plus
  rename), so the earlier "written in place" limitation is resolved. See the
  refinement section.
* Storage: the merge copies validated payload bytes; it does not re-verify
  payload values. Duplicate detection uses 128-bit digests; a collision can only
  cause a refusal.
* Tensor runtime: CUDA parity and speed are not measured. The finiteness checks
  synchronize the device.
* Streaming: CPU tensor parity is proven; CUDA parity and speed are not. Tensor
  mode still requests the CPU batch observation for row identity and evidence.

## Intended defaults

* CPU and scalar policy inference stay selected. `MJLAB_D1_POLICY_DEVICE` is
  unset by default.
* Batched publication keeps full rows and one append after the schedule unless
  `--stream-rows` is given.
* `--jobs 1` and `--cache-jobs 1` stay sequential.
* Historical scalar collection and the legacy row merge stay available.

## Refinement (second pass, 2026-10-12)

Four delta patches refined the candidate. Order matters: the statistics patch
depends on the storage `validate_shards` option, so storage lands first.

| Order | Patch | Files |
| --- | --- | --- |
| 1 | `refinement-storage.patch` | `storage.py`, `offline_validation.py`, storage/validation tests |
| 2 | `refinement-statistics.patch` | `dataset.py`, `projection.py`, `state.py`, `token_cache_builder.py`, statistics tests |
| 3 | `refinement-runtime.patch` | `adapter.py`, `runtime_x2.py`, `collector.py`, `collection_controls.py`, tensor/streaming tests |
| 4 | `refinement-cli.patch` | `scripts/diffusion.py`, preparation tooling, tooling tests, runbook, changelog, this report |

Candidate integrity: before applying, all 25 candidate files matched
`/tmp/mjlab-d1-throughput-20261012/candidate-file-hashes.json` byte for byte.
Each patch passed `git apply --check` and then applied with `git apply`. No
finished component was rewritten outside the accepted seams.

### Reconciliations applied for integration

* `validate_shards`: the storage patch adds
  `AppendOnlyShardStore(validate_shards=False)`. The statistics and token-cache
  patches pass it in the partition workers. Every consumed shard still runs the
  read-time hash, shape, dtype, finite and row checks. No caller uses a second
  mechanism.
* Tensor API: `VaeObservationTensorBatch` keeps the `batch_size` and `device`
  properties and adds optional row evidence, so the collector and tests that read
  the batch size or device keep working.
* Progress/chunk API: `BoundedRowPublisher.add` buffers and `flush` publishes the
  tail. `BatchedCollectionResult.stored_bytes` and `output_limit_exceeded` carry
  the final-flush verdict. The CLI fake-bundle tests construct the result with
  keyword arguments that keep defaults for the new fields.
* The refinement reintroduced three `ty` diagnostics in
  `tests/test_tracking_diffusion_streaming_controls.py` (string stand-in rows and
  an untyped fake store). Integration narrowed them with `cast(Any, ...)`. The
  project `ty` count returns to the candidate's 80, with no new diagnostic.

### Coverage added by the refinement

| Item | Tests |
| --- | --- |
| base tensor VAE and collector dispatch | `test_tensor_vae_batch_matches_numpy_batch_and_returns_posterior_mean`, `test_tensor_dispatch_runs_the_base_frozen_vae_tensor_method` |
| CLI batched bundle and base policy | `test_collect_batched_clean_run_reports_truthful_counts`, `test_collect_batched_stopped_run_returns_nonzero_and_warns`, `test_collect_batched_streaming_controls_write_progress` |
| whole-store hash avoidance | `test_fit_does_not_hash_the_whole_store_per_worker`, `test_token_worker_does_not_validate_the_whole_store`, `test_statistics_read_every_shard_once` |
| batch transform parity | `test_high_window_count_fit_matches_the_scalar_reference` |
| last-flush output limits | `test_final_flush_output_overage_keeps_the_stop_verdict`, `test_output_size_limit_stops_after_a_bounded_publication` |
| bounded shard counts | `test_streaming_coalesces_finished_trials_into_one_bounded_shard`, `test_publication_files_scale_with_row_chunks_not_trials`, `test_bounded_row_publisher_coalesces_and_flushes` |
| scalar flag refusal | `test_collect_refuses_batched_only_options_on_a_scalar_request`, `test_collect_refuses_a_batched_only_option_when_the_factory_is_scalar` |
| resolved device metadata | `test_collect_scopes_the_policy_device_override` |
| preparation jobs > 1 | `test_prepare_parallel_jobs_preserve_sources_and_output` |
| atomic projection bundle | `ProjectionBundle.save` staging; `test_stats_reuses_the_persisted_index` |

### Refinement re-run evidence (integration worktree)

Commands use `UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv`,
`PYTHONPATH=src`, `uv run --no-sync` and `-o faulthandler_timeout=120`.

| Command | Result |
| --- | --- |
| runtime group: streaming_controls, tensor_runtime, collector, throughput_tooling | 74 passed, 2 skipped (47 s) |
| statistics/storage group: statistics_throughput, dataset, token_cache_builder, storage, parallel_validation, fast_merge, window_index_persisted | 94 passed (51 s) |
| broad group: the 16 diffusion files (statistics, fast_merge, parallel_validation, tensor_runtime, streaming_controls, throughput_tooling, cli, dataset, projection, state, storage, collector, vectorized_collector, window_index_persisted, migration, contract) | 188 passed, 2 skipped (50 s) |
| `ty check` (whole project) | 80 diagnostics, same as the candidate; 2 are the pre-existing `dataset.py` lines |
| `ruff check` / `ruff format --check` on the changed files | clean |

The 2 skipped tests are the CUDA tensor cases; GPU execution was not
authorized. `tests/test_tracking_diffusion_window_dataset.py` still fails on the
absent `logs/diffusion/d1-pilot-20260930b` artifact; it is not in the groups
above. No new test failure was observed in any group.

### Corrected earlier counts and limits

* `ProjectionBundle.save` now stages `projection.npz` in a temporary beside the
target and renames it. The earlier "written in place" limitation is resolved.
The window index, JSON reports and merged store remain atomic.
* `ty check` and `ruff check` are measured for the integrated tree (table
above), not unverified.
* Whole-store hashing is removed from partition workers: `jobs=1` no longer
  validates the whole store, and `jobs=J` no longer does it `J` times. Each
  consumed shard is still hashed once at read time.
* `statistics_sha256` can change for any batched fit, including `--jobs 1`; the
  guidance to keep existing frozen projection bundles and their token caches is
  now documented in the runbook.

## Next step

An independent review of the integrated candidate. Then a parent-authorized,
matched benchmark on one machine to measure the CPU/GPU VAE and environment-count
arms. Production runs and GPU jobs need separate authorization.
