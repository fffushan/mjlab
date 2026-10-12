# D1 throughput parent acceptance

Date: 2026-10-12 (+08).
Baseline: `e71a89ab2ece99f84872fd562f8374267ffc8624`.
Candidate: `/home/agiuser/projects/worktrees/mjlab/d1-throughput-20261012/integration`.
Status: implementation accepted for local integration; GPU/NAS performance not measured.

## Implementation

- Train-only state and latent statistics use one traversal and batched array transforms.
- Window overlap multiplicity, center-relative coordinates and qualification rules remain unchanged.
- Parallel readers no longer hash the entire store once per worker. Each consumed shard still validates its hash and contents.
- Fast merge copies compatible validated shards without constructing and re-encoding every row.
- Inspection/replay and index/cache preparation support bounded worker counts.
- GPU-native frozen-VAE inference and action dispatch are explicit options. CPU inference remains available.
- Tensor collection uses one pre-step snapshot and retains required row evidence.
- Streaming coalesces completed trials into bounded chunks; compact reports preserve counts.
- Collection resource stops and final flushes report truthful partial status.
- Preparation tooling applies report and integrity gates before merging; it never starts training.
- Benchmark tooling generates matched commands with explicit seeds/cohort/worker settings. It records each arm status and continues after a stopped arm.

## Independent review

Four isolated component writers and four scoped refinement writers use `ds-oc/deepseek-flash:high`.
The retained integration worker combines patches. Two retained read-only reviewers recheck them.
Original data/runtime findings are resolved. Both reviewers report no remaining P0/P1 findings.
Reviewers could not execute tests; parent execution supplies validation evidence.

Parent closes remaining findings:
- preparation catches storage validation errors and writes a failure report;
- tensor collection refuses missing state/frame identity evidence before physics;
- generated benchmark shell preserves status and runs remaining arms;
- coalesced flush, tensor evidence, byte figures and limit wording are accurate;
- hand-built intra-shard A/B/A segment interleaving is an explicit limitation of the parallel index reader;
- tests narrow full versus compact collection result types, removing two new warnings.

## Parent validation

All tests use CPU/fake environments and synthetic temporary stores.
Commands use `UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv`, `PYTHONPATH=src`, `uv run --no-sync`.

| Gate | Result |
| --- | --- |
| Tooling, streaming, tensor suites | 65 passed, 2 CUDA tests skipped |
| Statistics/storage/index/cache/trainer/permission suites | 111 passed |
| Final 19-suite integration gate | **222 passed, 2 CUDA tests skipped**, 65.30 s |
| Final benchmark/tooling/vectorized recheck | **30 passed**, 28.98 s |
| Ruff on changed source/tests/tooling | PASS |
| `git diff --check` | PASS |
| Whole-project Ty | **78 diagnostics**, exactly the untouched baseline identities; no new diagnostics |

The broad integration gate includes statistics, dataset, token-cache builder, storage, parallel validation, fast merge, persisted index, tensor runtime, streaming controls, tooling, CLI, projection, state, collector, vectorized collector, migration, contract, trainer, and long-run permission tests.
A monitor initially used `/bin/sh` with Bash-only `pipefail`; pytest did not start. The corrected explicit Bash runner passed.
Prior subagent full-suite counts are not parent full-suite evidence.
Full repository suite, full frozen artifact fixtures and Sphinx build: not tested.

## Separate authorized training repair

The local candidate also preserves the parent-approved long-run permission repair.
CLI passes explicit `allow_long_run` to the trainer. Constructor and resolved budgets enforce it.
Config loading/serialization do not grant or revoke execution authority.
The 400,000-update CPU constructor, serialization round-trip, and refusal without permission pass.
The repaired baseline/dropout20 server runs use isolated earlier sources plus this fix only.
Local optimization integration cannot change those live sources.

## Limits

- CUDA tensor branches are not executed. CPU tensor parity is not CUDA numerical parity.
- No measured collection, merge, statistics, cache, or NAS speedup is claimed.
- Batched statistics can change floating-point bits and statistics SHA even at jobs=1. Keep historical statistics/caches unchanged.
- Parallel workers require Linux `fork`; tests emit the Python multithreaded-fork deprecation warning. Worker-count tuning remains unmeasured.
- Scalar partial collection retains its historical exit-0 behavior; the report must still be checked.
- The parallel index reader assumes contiguous trial segments from supported producers. Arbitrary hand-built intra-shard interleaving requires the sequential index path.
- Post-step CPU qualification evidence and CPU recovery-policy inference remain.
- No commit, push, deployment, new collection, optimization benchmark, or robot operation occurs during acceptance.

## Next action

With separate authorization, stage isolated source on the managed server and run a bounded CUDA parity check.
Then compare CPU/GPU VAE and 64/128/256 environments. Measure all preparation stages separately.
Use new outputs and read-only frozen inputs; keep existing training on GPUs 0/1 unchanged.
