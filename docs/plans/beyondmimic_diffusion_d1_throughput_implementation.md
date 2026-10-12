# D1 collection and dataset throughput implementation

Date: 2026-10-12 (+08).
Baseline: `e71a89ab2ece99f84872fd562f8374267ffc8624` on `devel`.
Status: implementation authorized; focused subagents assigned; performance not measured.
Controller, VAE-training, robot, and production-server changes are outside scope.

## Goal and constraints

Reduce D1 collection and preprocessing costs without changing the frozen dataset definition.
The operator requests `ds-oc/deepseek-flash` with high thinking for focused implementation subagents.
No commit, push, deployment, remote job, hardware operation, or GPU benchmark is authorized.
Keep checkpoints, manifests, source motions, historical stores, and token caches unchanged.
Keep 50 Hz control, 5 ms physics, four substeps, clean/OU pairing and seeded noise semantics.
Keep qualification, fallback, manual resets, exact row identities and terminal evidence.
Statistics must use training windows only, with original overlap multiplicity and window-relative coordinates.
Deterministic reduction changes can alter final floating-point bits. Do not claim hash compatibility without testing.
Keep scalar/CPU paths and historical stores readable. New outputs must not overwrite originals.

## Ownership board: multi-seam implementation

All isolated worktrees start from the exact baseline under:
`/home/agiuser/projects/worktrees/mjlab/d1-throughput-20261012/`.

| Lane | Directory | Exclusive source ownership | Deliverable | Why independent |
| --- | --- | --- | --- | --- |
| Statistics | `statistics` | `diffusion/dataset.py`, `state.py`, `projection.py`; unique statistics tests | One-pass batched train-only moments, segment/index reuse, deterministic bounded CPU parallelism | Statistical and frame-transform contract |
| Storage | `storage` | `diffusion/storage.py`, new offline-validation module, `scripts/diffusion_merge.py`; unique storage/validation tests | Immutable validated shard merge; bounded parallel inspect/replay APIs | Storage and integrity contract |
| Tensor runtime | `tensor-runtime` | `diffusion/adapter.py`, `runtime_x2.py`; unique tensor tests | Opt-in GPU-native VAE observation/action APIs, CPU-compatible tensor tests | Simulator/policy interface contract |
| Streaming | `streaming` | `diffusion/collector.py`, new collection-controls module; unique streaming tests | Bounded row flush, snapshot release, progress and enforced resource limits; optional tensor dispatch | Trial lifecycle and publication contract |
| Integration | `integration` | Only after all component handoffs; CLI wiring, runbook/changelog, benchmark tooling, seam fixes | Integrated candidate, CPU regression evidence, implementation report | Consumes all contracts, no overlapping writer |

Component handoffs: `/tmp/mjlab-d1-throughput-20261012/handoffs/<lane>.patch`.
Each component leaves its worktree intact. No commits or resets.
Use `git add -N` for new intended files before creating `git diff --binary HEAD` patches.
Reports use the subagent tool's managed output path, not repository-root scratch files.
Parent imports an accepted candidate into the primary checkout only after review and validation.

## Tensor interface agreement

Tensor-runtime lane exposes these opt-in methods:
- `env.observe_vae_tensor_batch(env_ids)` returns a tensor observation with `reference`, `conditioning`, and `batch_size`. Collection also requires owned `state`, `motion_id`, `reference_frame`, and `reference_phase` evidence from the same pre-step capture.
- `vae.batch_action_tensor(observation)` returns `(latent_tensor, clean_action_tensor)` on the model device.
- `env.step_tensor_batch(action_tensor)` executes the full batch without converting actions through NumPy.
- Runtime enablement is explicit, CPU inference remains selectable, and the chosen policy device is recorded.

Streaming lane may dispatch through these methods when all are available.
Existing OU noise remains generated with the seeded NumPy process; moving its sampled values to CUDA must not change the stream.
Record CPU copies of state/action/latent evidence after inference; avoid the GPU-to-CPU-to-GPU observation round trip.
CPU/fake tensor parity proves API semantics only, not CUDA numerical parity or speed.
If the contracts need revision, ask the parent before diverging.

## Acceptance gates

- Statistics: original overlap weighting and center frame; scalar reference parity; no validation/test influence; jobs=1 versus jobs>1; nonfinite and malformed inputs rejected.
- Storage: source bytes unchanged, output staged and atomically published, duplicates and mixed identities refused, complete row accounting, shard validation, boundary cases, deterministic parallel results.
- Tensor runtime: no gradients, posterior mean unchanged, action conversion unchanged, observation/latent/action parity on CPU tensors, explicit CPU fallback, no lost qualification/physics evidence.
- Streaming: clean noise exactly zero, matching OU stream, exact pair admission and resets, bounded flush, accepted/verification labels retained, limit/fault partial reports and progress, failed writer never silently accepted.
- Integration: CLI backward compatibility, per-input inspect/replay before merge, strict completeness and row checks, index/cache reuse, bounded workers and atomic output, consistent progress/resource controls.
- Review: fresh read-only correctness/statistics/storage and runtime/lifecycle reviews. Parent decides findings.

## Validation and limits

Use `uv run --no-sync` with the existing environment and the lane's `src` on `PYTHONPATH`.
Focused tests use CPU/fakes/synthetic temporary stores. No external artifacts are required.
Inspection commands: 30 s; small tests: 60 s; focused suite: 120 s with `-x -vv` and faulthandler diagnostics.
Do not rerun an unexpected timeout unchanged or increase its limit. Isolate it first.
Long validation requires parent-managed monitored execution.
Provide benchmark tooling and a matched baseline/optimized protocol, but do not run production or GPU experiments.
Measure startup, stepping, serialization, merge, validation, indexing, statistics and cache work separately later.
Do not remove integrity checks, noise, physics, or qualification to improve a speed score.

## Next action

Component writers produce patches and focused evidence. One integration-only writer consumes them.
Independent reviewers inspect the integrated candidate. Parent validates and records acceptance.
Performance benchmarks, server deployment, and full-scale runs require separate authorization.
