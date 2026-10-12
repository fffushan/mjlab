# D1 optimization review disposition and refinement

Date: 2026-10-12 (+08).
Candidate: isolated `integration` worktree, baseline e71a89ab2.
The two read-only reviewers supplied source review but could not run tests.
Their verdicts are OK with notes; the parent does not accept the current candidate yet.

## Accepted fixes

- Data P1-1: remove per-worker full-store hashing; retain assigned-shard hash and semantic validation.
- Data P2-1: document batched statistics hash changes even with jobs=1.
- Data P2-2: publish projection/statistics bundle atomically.
- Data P2-3: add CLI refusal test for foreign persisted index.
- Data P2-4: keep inspect fail-closed on malformed controller/latent combinations.
- Data P2-5: reject interleaved/noncontiguous segment runs rather than drop windows.
- Data P2-6: reconcile report validation counts.
- Runtime P2-1: reject unsupported scalar streaming flags and record resolved policy device.
- Runtime P2-2: remove duplicate full snapshot/CPU reference-conditioning work in tensor dispatch; retain all row evidence.
- Runtime P2-3: coalesce completed trials into bounded full-sized shards, cache byte accounting, avoid metadata walking per progress event.
- Runtime P2-4: check output-size verdict after final flush, including last-trial overflow.
- Runtime P2-5: nonzero CLI status for partial collection and accurate budget-enforcement documentation.
- Runtime P2-6: CLI clean/partial fake bundle and base tensor policy collection coverage.
- Runtime P2-7: explicit float32 tensor-observation conversion with unchanged finite checks.
- Parent review: `_qualified_row_partition` still performs quadratic searches; replace with a linear identical-order partition.
- Parent review: statistics still transforms each window in a Python loop; vectorize the batched center-relative transform while retaining window weighting.

## Refined ownership

Each lane starts from the completed integrated candidate, in an isolated worktree.
The existing `integration` directory remains read-only until all refinement patches are ready.

| Lane | Worktree suffix | Exclusive files | Handoff |
| --- | --- | --- | --- |
| Statistics | `refine-statistics` | dataset.py, state.py, projection.py, token_cache_builder.py; statistics tests | refinement-statistics.patch |
| Storage | `refine-storage` | storage.py, offline_validation.py; storage/validation tests | refinement-storage.patch |
| Runtime | `refine-runtime` | adapter.py, runtime_x2.py, collector.py, collection_controls.py; tensor/streaming tests | refinement-runtime.patch |
| CLI/integration tooling | `refine-cli` | scripts/diffusion.py; throughput scripts/tests; runbook/changelog/reports | refinement-cli.patch |

Storage adds `validate_shards=True` as a backward-compatible constructor option. Trusted partition readers can set False because each consumed shard still validates its content/hash. Statistics owns caller changes.
All writers export a text-only delta against the original candidate, not against HEAD. New files must use /dev/null headers.
The parent supplies a local delta-export script; patches remain under `/tmp/mjlab-d1-throughput-20261012/handoffs/`.
No commit/reset/push/GPU/SSH/live rollout/frozen-artifact change.
No long-run training permission fix or remote relaunch is included: that repair still requires separate approval.

## Gates

Run CPU synthetic tests with existing venv and lane src. Keep 120-second focused groups and diagnose timeout before retry.
Confirm no whole-store hash per worker; scalar/batched/parallel stats parity, windows starting above tick zero and all shard boundary cases.
Confirm source/output integrity, deterministic parallel errors and controller/latent fail-closed checks.
Confirm tensor evidence equivalence, unchanged noise and handoff, coalesced writes, bounded pending rows, last-flush limits and partial status.
Integration then re-runs affected CLI/tooling tests and fresh scoped reviews. Parent acceptance follows real tests, not just reviewer verdict.
