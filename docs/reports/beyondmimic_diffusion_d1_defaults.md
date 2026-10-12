# D1 throughput default enablement

Date: 2026-10-12 (+08).
Operator authorizes the scoped commit and requests default optimizations.
Server benchmark and dataset audit: [measured report](beyondmimic_diffusion_d1_server_benchmark.md).

## Defaults

- CLI collection uses256 environments; X2 batched VAE follows simulation device.
- CUDA policy uses IEEE FP32 after runtime construction. TF32 is not parity-qualified.
- Batched CLI streaming and compact results are on. Compatible runtimes use tensor dispatch.
- CLI legacy opt-outs: `--no-stream-rows`, `--no-compact-results`, `--no-tensor-dispatch`, `--policy-device cpu`.
- Scalar callers use `--max-envs 1`; automatic defaults do not refuse scalar factories.
- Library collectors retain historical defaults and result types unless controls are supplied.
- Inspection/replay/index/statistics CLI uses8 bounded workers. Statistics batch size is512.
- Statistics automatically reuse `window_index.npz` beside assignments if present, with source validation.
- Preparation tool usesjobs8/batch512 and independent **cache-jobs1**.
- Token-cache CLI/trainer use1 worker. Cache computation is CPU-based, not one GPU per cache worker.
- Runbook preserves one collector per GPU and best-tested one cache worker for the small213818-row dataset.
- Full-scale/four-collector optimum remains not measured. Frozen artifacts and active server jobs are unchanged.

## Local verification

- Final20-suite CPU gate: **227 passed /2 CUDA skipped**,63.56s.
- Last CLI/defaults/tensor/trainer/permission recheck: **74 passed /2 skipped**,32.66s.
- Ruff format/check and scoped Ty pass for commit source/tools/defaults tests.
- Scoped Pyright passes for changed source excluding unchanged `checkpoint.py`.
- Full `make check` is run with `UV_NO_SYNC=1 UV_OFFLINE=1`.
  Formatting/lint pass. Ty fails with **76 pre-existing repository diagnostics**.
  Untouched baseline has78. Two errors in touched dataset annotations are fixed.
- Pyright's broad diffusion-package check has remaining unchanged checkpoint errors/export warnings.
  No new commit-source errors remain. Whole-repository type-check success is not claimed.
- Full repository tests and full-default remote rerun: not tested.
  Defaults select the same numerical paths/options already benchmarked. CPU tests prove default wiring.

## Scope

Commit includes the accepted optimization, defaults, focused tests, VGK36 collection contract/config,
benchmark/acceptance reports, plans, runbook, and the separately validated long-run permission repair.
Unrelated concurrent decoder/evaluation changes stay outside the commit.
No push, server checkout update, new experiment, robot operation, or frozen cache/statistics rewrite occurs.
