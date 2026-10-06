# D5 model-backed reproduction backend — CPU software slice

Date: 2026-10-06.

## Scope

This slice replaces the blocked contract-file identity policy and adds the
concrete model-backed assembly directly in the primary working source.

Implemented:

- `src/mjlab/tasks/tracking/diffusion/model_backend.py`
  - `ModelBackendSpec` for explicit contract/checkpoint/projection/VAE paths.
  - `ModelPlannerFactory`, which is pickleable and creates the denoiser and
    planner inside the bounded spawned worker.
  - `load_model_backend`, which loads the owner-process VAE decoder.
  - `build_model_backed_runtime`, which composes the worker executor and the
    accepted `AsyncLatentRuntime` facade.
- `replay_integration.py` optional checkpoint, projection and VAE digest gates
  for reproduction callers. Model schema, dimensional and finite-value checks
  remain active.
- `contract.py` no longer requires a contract YAML file digest to be registered.
  The file digest remains descriptive metadata and is excluded from semantic
  `identity_hash()`.
- Projection, storage and persisted-index loading no longer reject compatible
  reproduction data only because a recorded contract-file identity differs.
  Schema compatibility and shard/assignment/content checks remain.
- Public package exports, D5 plan, runtime README, changelog and focused tests.

No standing, zero-action, synthetic-model or robot fallback was added.

## Validation

All commands used `uv run --no-sync`, `CUDA_VISIBLE_DEVICES=''` and
`PYTHONDONTWRITEBYTECODE=1` for test execution.

- `49 passed in 15.80s`: model backend, D3 replay, contract, VAE revision,
  projection and checkpoint tests.
- `68 passed in 59.72s`: latent planner, async runtime and bounded executor
  tests.
- `38 passed, 1 skipped in 47.03s`: window dataset, persisted index,
  selection-provenance, device and training CLI tests.
- `91 passed in 3.09s`: ablation surface, D4 integration, evaluation and
  migration tests.
- `38 passed in 1.96s`: closed-loop runtime, CLI and handover tests.
- Ruff format and lint passed for the new backend and all directly changed
  loader/contract/projection/storage/test files.
- Ty passed for the new backend and directly changed loader/contract/projection/
  storage files.
- Pyright passed with 0 errors and 0 warnings for those same files.

The repository's unrelated dirty `dataset.py`/window-dataset work still has a
pre-existing Ty diagnostic in its `np.savez` call and unrelated test-dictionary
type diagnostics. Those are not claimed as fixed by this slice.

Source hashes are recorded in
`logs/diagnostics/diffusion-d5-model-backend-cpu-20261006/source.sha256`.


- The active no-mixture diffusion checkpoint and projection bundle are not
  present in this laptop checkout, so no real frozen-model replay ran.
- The VAE loader still requires an explicit compatible checkpoint, manifest,
  member and model schema at runtime; missing assets fail instead of falling
  back.
- No GPU, sustained 50 Hz/p99 timing, simulator, export, controller,
  deployment, vendor sim2sim, physics or hardware evidence was collected.
- This is research reproduction software evidence, not production safety or
  deployment readiness.
