# D1 throughput tooling

Opt-in helper tools for the D1 collection and dataset throughput work. Neither
tool constructs a simulator, uses a GPU or starts training.

## `prepare_dataset.py`

Scalable preparation entry point. It runs four phases in order and reports the
wall time of each:

1. `merge` (only with `--input`): gate every worker store with its collection
   report, `inspect` and `replay`, then merge the stores byte-preservingly.
2. `dataset`: build the grouped split assignments and persist the window index
   once, then reload it to prove it describes the store and assignments.
3. `statistics`: fit train-only projection statistics from the persisted index.
4. `cache`: prepare the token caches with bounded worker counts.

A worker needs its completed collection report (`<run>/report.json`). The tool
refuses a partial report, a stop reason, a contract mismatch, a row accounting
mismatch, a duplicate identity, an unreadable store and an existing dataset
store. It prints the training command with `--require-prepared-cache` and does
not run it.

```sh
uv run --no-sync python scripts/experiments/diffusion_d1_throughput/prepare_dataset.py \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml \
  --dataset-dir logs/diffusion/d1-vgk8-expanded-<stamp> \
  --input logs/diffusion/d1-run-<stamp>/store-worker-0 \
  --input logs/diffusion/d1-run-<stamp>/store-worker-1 \
  --jobs 4 --cache-jobs 4 --max-cache-gib 8
```

Bounded validation/index/statistics default to eight workers with 512-window
statistics batches. Token caches independently default to one CPU worker.
The managed-server small dataset test found one cache worker faster than eight.
Do not map cache workers to GPUs; use one collection worker per GPU instead.

## `benchmark_matrix.py`

Generates a matched D1 collection benchmark matrix. It varies only the VAE
policy device (`cpu`, `cuda:0`) and the environment count (64, 128, 256). Every
arm shares the contract, artifact root, config, runtime factory, trial budget
and streaming flags. The tool writes `plan.json` and `commands.sh` to a new plan
directory and refuses an existing output. It does not run the commands.

```sh
uv run --no-sync python scripts/experiments/diffusion_d1_throughput/benchmark_matrix.py \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml \
  --artifact-root . \
  --config configs/diffusion/x2_50hz_collection_vgk8.yaml \
  --stamp <stamp> \
  --plan-dir logs/diffusion/d1-vgk8-benchmark-<stamp>
```

The generator pins one seed, one worker, and the eight VGK8 references by
default. For VGK36, supply its contract, manifest, teacher selection, and a
216-trial budget for one seed. The generated shell records each arm's exit
status and continues after a budget stop or failure. Such an arm is not a
successful collection; inspect its report before using any output.

The inputs are read-only references and each arm writes to a new output
directory. A measured speedup needs the same machine, a warm page cache and a
matched baseline. This tool only produces the commands.
