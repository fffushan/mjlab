# Parallel implementation contract: X2 tennis recovery

This refines `x2_tennis_recovery_finetune.md` for the operator-approved parallel implementation. It does not change the scientific experiment. GLM model: `lingzhi/glm-5.2`.

Allocated worktrees (both verified at the pinned base HEAD; imports resolve to their own `src/`):

- Runtime: `/home/agiuser/projects/worktrees/mjlab/tennis-recovery-parallel-ATFKNq/runtime`
- Resume/evaluation: `/home/agiuser/projects/worktrees/mjlab/tennis-recovery-parallel-ATFKNq/resume-eval`

The old workflow is `e9f2ee15-7fc5-40b1-b186-e0d54c99be23`; retained pool child is `09733d12-0677-4aa6-a112-4c08efb5893f`. Mission: `05ccab6d-2d73-4409-8e56-9525b34319ab`. Worktree creation was deliberate under the operator's parallelization approval; no managed-worktree preflight failed and no isolation was dropped.

## Ownership and scheduling

- **Pool lane (existing child):** current `/home/agiuser/projects/mjlab` checkout; owns only `mdp/tennis_endpoint_pool.py` and `tests/test_x2_tennis_endpoint_pool.py`. Keep its existing progress. At its final handoff stop the OLD sequential scheduler: its structured `verdict=blocked` means an intentional scheduling barrier, while its summary separately states actual pool completion/tests. It must not launch successors.
- **Runtime lane:** a new isolated detached worktree of base HEAD `4de481a9468b769dcb1fe8456a09b6b852fab56e`; owns `mdp/tennis_recovery.py`, `config/agibot_x2/tennis_recovery_env_cfg.py`, and runtime tests.
- **Resume/evaluation lane:** another isolated worktree at the same base; one resume stage owns `rl/tennis_recovery_runner.py`, `config/agibot_x2/tennis_recovery_rl_cfg.py`, and resume tests. A subsequent metrics stage owns `mdp/tennis_recovery_metrics.py`, bounded evaluation support/tests, but no runtime/pool implementation.
- **Final integration:** only after all actual component results are inspected. Parent integrates approved file-scoped changes, then a single integration owner wires shared registrations/exports, full evaluation CLI, guide/changelog and end-to-end tests. No concurrent writes to the source checkout or common worktree.
- No component writes shared `__init__.py`, generic train/runner code, base X2 configs, XML, datasets, source checkpoint or installed dependencies. Do not commit/stage/revert/clean/reset. Worktrees and uncommitted changes remain available for parent diff-based integration.
- If a real contract problem appears, contact the parent. A queued steering message is not proof of compliance; parent must verify the old scheduler stops before any duplicate stage starts.

## Stable pool/runtime API

Module: `mjlab.tasks.velocity.mdp.tennis_endpoint_pool`.

Export `EndpointPool` with this construction contract (keyword names are fixed):

```python
EndpointPool.from_directory(
    directory,
    *,
    last_n_frames=10,
    split="train",           # train | validation | all
    validation_fraction=0.2,
    seed=42,
)
```

`directory` accepts `str | pathlib.Path`. Construction validates and loads; import does no dataset I/O. Return a pool with read-only NumPy arrays:

| Attribute | Shape / meaning |
|---|---|
| `joint_pos` | `[N,31]`, radians, exact X2 MuJoCo hinge-joint order excluding free joint |
| `joint_vel` | `[N,31]`, rad/s, same order |
| `root_pos_w` | `[N,3]`, pelvis/root-link world position |
| `root_quat_w` | `[N,4]`, unit wxyz quaternion |
| `root_lin_vel_w` | `[N,3]`, root-link origin world linear velocity, m/s |
| `root_ang_vel_w` | `[N,3]`, world angular velocity, rad/s |
| `trajectory_ids` | `[N]` integer IDs stable across train/validation/all modes |
| `frame_indices` | `[N]` original trajectory frame indices |
| `joint_names` | tuple of 31 verified model-order names |
| `body_names` | tuple of 32 verified model-order names, index zero pelvis |
| `manifest` | JSON-serializable mapping containing full source inventory/hashes, split, seed, FPS and ordering proof/fingerprint |

`len(pool)` returns N. Concatenate rows in stable trajectory order, chronological order within each final window. Fail on a short trajectory rather than silently changing rows per trajectory. Since every included trajectory has exactly `last_n_frames` rows, uniform row sampling is exactly equivalent to uniform trajectory followed by uniform frame. Runtime can cache arrays on device once and use seeded torch indices; do not read files on reset.

Pool may also expose `sample_indices(count, rng)` for pure NumPy consumers (`rng` a `numpy.random.Generator`), returning integer row indices. Tests and evaluation may index arrays directly; no other pool method is required by runtime.

Real-data ordering evidence already established by pool worker: FK positions agree around 1e-7 m, quaternion dot errors around 1e-8. NPZ body rows exclude world and begin pelvis; hinge joint order excludes root free joint. Notably waist order is **yaw, pitch, roll** in these model/NPZ arrays. This is distinct from controller mappings in other projects; do not re-order by guesswork.

Any deviation needed from this interface requires a parent decision and a shared contract update. Do not create a second competing pool implementation in a parallel worktree.

## Runtime without waiting for loader implementation

- A small structural protocol/test fake with the attributes above is allowed; production type is EndpointPool. No duplicate loader or stub production module.
- Inject a supplied in-memory pool into a narrow state-reset helper in tests. Production construction lazily resolves/loads EndpointPool when the recovery environment is actually instantiated. A local import at this explicit runtime boundary is acceptable to avoid a premature module import dependency; don't add a broad plugin framework.
- Keep factory name `agibot_x2_tennis_recovery_env_cfg(play=False)`; expose configurable pool/group fields through its command or event config. Factory construction/task registration must not touch files. Runtime final report names command/config types and exact parameter paths for integration.
- Production physics can be tested using a small synthetic pool matching the verified array contract; real-data integration waits for the pool handoff.
- Full state at reset, velocities not zeroed, XY recentered, reference height preserved, optional yaw applied to orientation and world velocities consistently. Group ownership fixed; exactly zero recovery twist forever; retention remains original command/reset distribution.
- Do not depend on resume runner or evaluator to pass component tests.

## Resume and metrics independence

- Runner/config implementation depends only on existing velocity runner and checkpoint contract, not the new pool/runtime.
- Factory name `agibot_x2_tennis_recovery_ppo_runner_cfg()`; inherited base actor/critic dimensions and distributions preserved. Explicit optional initial-fine-tune LR override handled after load; ordinary resume unset preserves saved optimizer LR. Do not silently reset optimizer, normalizers or counters.
- For actual checkpoint loading tests, use original velocity environment (small local test) or an isolated faithful runner fixture. Do not wait for recovery registration or change base task behavior.
- Metrics helper receives state/command/termination traces and explicit timestamps/entry heading. No dependency on the pool module or runtime factory is needed to test displacement, sustained settling, censoring and early-window statistics.
- Evaluation sampling/config support can rely on the API above and validated bounds, but do not claim an end-to-end evaluator passed until final integration supplies real runtime/loader. Avoid a fake production evaluator that merely returns synthetic success.

## Worktree environment and evidence

Use the original project's installed environment read-only while importing code from the owned worktree:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --no-sync --project /home/agiuser/projects/mjlab python ...
```

Use the same prefix for pytest/type/lint tools. Verify `mjlab.__file__` resolves to the owned worktree and `sys.executable` to the intended original virtualenv. Do not run `uv sync` or mutate the shared environment from a worktree. Read data/checkpoints through their original absolute paths; write outputs only to private temp or runtime-managed artifact locations.

Use 30 s inspection, 60 s small tests, 120 s focused suites. Ask parent to manage genuinely long compilation/GPU validation. Run CPU-only/pure tests concurrently where practical; parent serializes resource-heavy GPU validation. No full training, remote GPU access, uploads, deployments or robot contact.

Each runtime-bound component handoff includes API, file list, exact test/lint commands and results, limits/skipped gates, and worktree path. Parent reviews actual files and applies only owned paths; do not delete worktrees during worker cleanup.
