# Collection-linked VAE sample budgets

Date: 2026-10-08, 20:24 (+08).
Source baseline: `f237f30c19704d9b7799cdf0752cfdb1a119ffe7`.
Implementation status: implemented and validated; retained changes are recorded in Git history.

## Implementation

The public `distill train` entry point accepts three optional budget inputs.

| Input | Derived total | Conflicting explicit input |
| --- | --- | --- |
| `--envs-per-motion E` | selected motion count × E | `--num-envs` |
| `--draw-to-collection-ratio r` | ceil(r × main rows / effective batch) updates | `--updates-per-iteration` |
| `--replay-retention-iterations h` | ceil(h × main rows) replay capacity | `--replay-capacity` |

Main rows = `num_envs * collection_steps`.
Effective batch = `minibatch_size * accumulation_steps`.
Minibatch size and accumulation remain independent throughput settings.
Unset pairs preserve 1 environment, 1 update and 16384 replay rows.
Conflicts fail before lifecycle construction, including explicit legacy-default values.
Decimal-rational arithmetic prevents binary-rounding errors before `ceil`.

The single-teacher and cohort paths receive the same resolved totals.
Worker divisibility is checked after environment count derivation.
Reports and checkpoints add `resolved_config.sample_budget`.
This record includes requested inputs, achieved totals and live per-motion allocations.
Formula inputs are audit metadata. Resolved runtime, schedule and replay remain resume invariants.

Files:
- [`sample_budget.py`](../../src/mjlab/tasks/tracking/distillation/sample_budget.py)
- [`distill.py`](../../src/mjlab/scripts/distill.py)
- [`usage`](../../docs/source/x2_tennis_distillation.rst)
- [`changelog`](../../docs/source/changelog.rst)
- [`budget tests`](../../tests/test_tracking_distillation_sample_budget.py)
- [`single-teacher CLI tests`](../../tests/test_tracking_distillation_cli.py)
- [`cohort CLI tests`](../../tests/test_tracking_distillation_cohort_cli.py)

## Validation evidence

Final command:

```bash
uv run --no-sync pytest \
  tests/test_tracking_distillation_sample_budget.py \
  tests/test_tracking_distillation_cli.py \
  tests/test_tracking_distillation_cohort_cli.py \
  tests/test_tracking_distillation_coverage_execution_path.py \
  -x -vv -o faulthandler_timeout=30
```

Result: **161 passed**, one existing Tyro parser deprecation warning, 47.14 seconds.
Captured output was written to `/tmp/vae-sample-budget-final-tests-20261008.log`, which no longer exists.
Monitor `#6` reports `exit=0`.

Tests replace the simulator, not replay, trainer, Adam or checkpoint code.
Tests cover public Tyro parsing, 1/8/15/30-motion arithmetic, rounding and conflicts.
Tests cover real optimizer draw counts, metadata persistence and sharded-builder dispatch.
Tests cover derived-to-explicit resume and legacy metadata absence.
A changed resolved update count fails resume.
The existing fake cohort adapter now seeds slot allocation independently.
Production slot allocation is unchanged.

Ruff format/check: pass for all changed Python files.
Ty: pass for `distill.py` and `sample_budget.py`.
Pyright: new budget module passes.
Pyright reports three existing errors in `distill.py`.
The unchanged baseline produces the same three diagnostics:
- optional `collector.adapter` access in single-teacher seed reporting;
- optional `collector.adapter` access in single-teacher cleanup;
- optional manifest `base_task` passed to a required string parameter.

`git diff --check`: pass.
Full repository suite: not tested.
The earlier accumulation-equivalence failure is outside this change and remains unresolved.

## Operational status and limits

No training, live simulator, GPU job, deployment or hardware operation was launched.
Existing models, manifests and unrelated untracked files are unchanged.

Ratios count total replay draws, not distinct trained rows.
The denominator excludes bootstrap, evaluation and extra terminal-predecessor rows.
Protected draws must fit inside the total if protected sampling is enabled later.
Protected replay is not implemented by this change.
Unequal manifest weights preserve the existing allocation policy.
`envs-per-motion` is then a mean, not a minimum allocation.
Shard-local rounding can change per-motion ratios; the report exposes this effect.
A replay horizon is nominal; it does not prove full state or motion coverage.

For 30 equal-weight motions, E=1024, collection_steps=32, effective_batch=32768,
r=1 and h=2 produce 30720 environments, 30 updates and 1966080 replay rows.
These are arithmetic results, not quality or GPU-memory acceptance.

Next action: select the collection, draw and retention budgets for a controlled run.
GPU memory, throughput and per-motion holdout fidelity require separate validation.
Launching that run requires user authorization.
