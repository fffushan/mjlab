X2 VGK8 Diffusion Training Pipeline
====================================

This page is the runbook for the VGK8 state--latent diffusion pipeline. It
covers data collection, dataset preparation, D2 training, and offline
checkpoint evaluation. It does not authorize a run. Every command below starts
work only when an operator executes it.

The pipeline uses the frozen X2 VAE and the no-mixture VGK8 contract. It does
not change the VAE objective, teacher routing, action contract, control timing,
or numerical controls.

Pipeline stages
---------------

The stages are ordered. Do not start D2 training from a collection directory
before the D1 gates pass.

.. code-block:: text

   frozen VAE + frozen recovery policy
                 |
                 v
   D1 collection: clean and OU trial rows
                 |
                 v
   inspect + replay + merge
                 |
                 v
   grouped windows + train-only statistics
                 |
                 v
   train/validation/test token caches
                 |
                 v
   D2 state--latent denoiser training
                 |
                 v
   offline generation evaluation
                 |
                 v
   optional D3 closed-loop simulation

D3 closed-loop simulation is a separate gate. D2 offline evaluation does not
prove closed-loop stability, real-time performance, hardware quality, or
deployment readiness.

Frozen VGK8 identity
--------------------

Use one contract, one VAE checkpoint, one teacher manifest, and one dataset
identity for a run. Record their hashes in the run report.

The active identities and shapes are:

* Contract: ``docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml``.
* Contract semantic ID: ``00c7915339a1312b126cc6ef46e8a67c72796b041bac7f756753439ec65c410d``.
* Contract YAML SHA-256: ``9594e3dac29c7cd4fe168d349a3748a813d8c8895bebb381948d44d5f66c3a7d``.
* VAE shape: reference 68 / conditioning 99 / latent 32 / action 31.
* Diffusion token: 167 channels: 135 state + 32 latent.
* Window: 41 rows, current index 8, 8 history + current + 32 future.
* Training noise: ``K=1000`` cosine schedule.
* Inference sampler: 20-step DDIM, ``eta=0``.

The active eight-teacher manifest is usually:

.. code-block:: text

   configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v1.yaml

It contains ``vgk000`` through ``vgk007``. The collection runtime reads this
manifest through ``MJLAB_D1_COHORT_MANIFEST``.

Server best practice
--------------------

The current collection default uses 256 environments, GPU-native VAE inference,
bounded streaming, and compact counts. Use one collector process per physical
GPU. Scalar collection remains available with ``--max-envs 1``.

The latest managed-server test used physical GPU 3 on
``10.14.64.48:32221``. The process sees that GPU as logical ``cuda:0`` because
of ``CUDA_VISIBLE_DEVICES``.

The 2026-10-12 (+08) VGK36 test used 36 references and two seeds per arm.
Stored-row throughput was 589 rows/s for original CPU64, 669 for GPU64,
806 for GPU128, and 899 for GPU256. GPU256 was **1.53x** the original rate.
This is the fastest tested small-run setting, not a full-scale four-GPU optimum.
The earlier VGK8 CPU-inference sweep favored 64 environments. That result does
not select the optimum for the new GPU-inference path.

Defaults for offline validation, indexing, and statistics are ``--jobs 8``.
Statistics use batches of 512 windows and reuse a source-bound persisted index
beside the assignments when one exists. Merge copies validated shards by default.

**Token caches use one worker by default.** On the measured 213,818-row dataset,
optimized cache preparation took 88.29 s with one worker and 365.48 s with eight.
One worker was best among the tested counts (1 and 8). Do not call it a
universal optimum. Token-cache workers run on the CPU; they do not own GPUs.
Keep one collector per GPU, then prepare token caches with ``--jobs 1`` or
``--cache-jobs 1``. Do not derive the cache worker count from the GPU count.

Use ``nvidia-smi`` before launch. Use a monitored process with wall and inactivity
limits. Keep concurrent training and shared storage effects in timing reports.
See ``docs/reports/beyondmimic_diffusion_d1_server_benchmark.md`` for full evidence.

Prepare the shell
-----------------

Run from a clean isolated worktree. Do not use a dirty shared checkout for a
bulk run. Set ``ROOT`` to the worktree. Link the read-only artifact directories
when the worktree does not contain them:

.. code-block:: bash

   export ROOT=/path/to/mjlab-worktree
   cd "$ROOT"
   # When artifacts are stored in a separate read-only checkout, create these
   # links once inside the worktree:
   # ln -s /home/fushan/mjlab/data/tennis "$ROOT/data/tennis"
   # ln -s /home/fushan/mjlab/logs/distillation "$ROOT/logs/distillation"
   # ln -s /home/fushan/mjlab/logs/rsl_rl "$ROOT/logs/rsl_rl"

   export CONTRACT=$ROOT/docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml
   export COLLECTION_CONFIG=$ROOT/configs/diffusion/x2_50hz_collection_vgk8.yaml
   export TRAIN_CONFIG=$ROOT/configs/diffusion/x2_50hz_train_vgk8.yaml
   export MANIFEST=$ROOT/configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v1.yaml

   export CUDA_VISIBLE_DEVICES=3
   export MJLAB_D1_DEVICE=cuda:0
   export MJLAB_D1_COHORT_MANIFEST=$MANIFEST
   export MJLAB_D1_ROLLOUT_TEACHERS=vgk000,vgk001,vgk002,vgk003,vgk004,vgk005,vgk006,vgk007
   export MJLAB_D1_SCHEDULE=bulk
   export MJLAB_D1_WORKERS=1
   export MJLAB_D1_WORKER=0
   export MJLAB_D1_WORKER_OFFSET=0
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1

Check the physical GPU before the run:

.. code-block:: bash

   nvidia-smi -i 3

Check the frozen inputs before collection:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion preflight \
     --contract "$CONTRACT" --artifact-root "$ROOT"

D1 collection
-------------

The collection input is non-authorizing. The explicit ``--execute`` flag is
required. Use a unique output directory for every attempt. The following is a
bounded 48-trial smoke that uses the best tested setting:

.. code-block:: bash

   export STAMP=$(date -u +%Y%m%dT%H%M%SZ)
   export MJLAB_D1_BULK_SEEDS=1
   export MJLAB_D1_RUN_ID=vgk8-d1-throughput-smoke-$STAMP
   export STORE=$ROOT/logs/diffusion/d1-vgk8-throughput-smoke-$STAMP/store
   export REPORT=${STORE%/store}/report.json

   mkdir -p "${STORE%/store}"
   timeout --signal=TERM --kill-after=10s 900s \
     env PYTHONPATH="$ROOT/src" \
     uv run --no-sync python -m mjlab.scripts.diffusion collect \
       --contract "$CONTRACT" \
       --artifact-root "$ROOT" \
       --config "$COLLECTION_CONFIG" \
       --execute \
       --runtime-factory mjlab.tasks.tracking.diffusion.runtime_x2:build_x2_runtime \
       --output "$STORE" \
       --report "$REPORT" \
       --max-trials 48 \
       --max-envs 256 \
       --max-gpus 1 \
       --max-wall-minutes 12 \
       --monitor-inactivity-seconds 180 \
       --max-output-mib 1024

For a bulk run, set ``MJLAB_D1_BULK_SEEDS`` and the trial budget from a
reviewed collection plan. Do not reuse the smoke output. Record the seed range,
worker offset, run ID, output path, and source commit before launch.

For several workers, each worker must have a unique output directory and a
disjoint trial offset. A pair's clean and OU trials must remain in one worker.
After every worker finishes, run inspect and replay before merging its store.
Do not merge a failed or partial store.

The collection command writes raw rows and a machine-readable report. A valid
report must have ``partial: false`` and ``stop_reason: null``. A physical fall
is a trial result. It is not automatically a software failure. An OU pair is
skipped when its clean partner does not qualify.

Collection publication and limits
---------------------------------

The batched CLI enables bounded streaming and compact counts by default.
Compatible runtimes use device-native tensor dispatch. The X2 batched runtime
loads the VAE on the simulation device and pins IEEE FP32 after construction.
TF32 failed the tight CPU/CUDA parity test; IEEE FP32 passed rtol/atol ``1e-5``.
Scalar collection stays CPU-based unless the device is explicitly selected.
Library ``collect_batched`` callers retain their existing behavior unless they
supply controls. CLI defaults do not silently change library result types.

Use these overrides when needed:

* ``--no-stream-rows`` restores end-of-schedule publication. Compact counts are
  then disabled unless explicitly requested, which is refused;
* ``--no-compact-results`` retains full per-trial rows;
* ``--max-rows-per-append N`` changes the default 4096-row publication bound;
* ``--no-tensor-dispatch`` selects NumPy observation/action dispatch; and
* ``--policy-device cpu|simulation|cuda[:N]`` selects the VAE inference device.
  An existing ``MJLAB_D1_POLICY_DEVICE`` setting remains an explicit override.

Positive streaming, compact, tensor, append-bound and progress options need a
batched runtime. Scalar requests with these explicit options are refused.
Unspecified defaults do not cause scalar refusal. A factory without tensor
methods keeps NumPy dispatch; explicit tensor requests still fail if unsupported.

``--tensor-dispatch`` with the CPU policy device is a reference and parity
option, not a speedup: the device encode/decode then runs on the same host the
NumPy path uses. Tensor dispatch reuses one pre-step snapshot and transfers
only the required row evidence to the CPU. Post-step CPU observations remain
required for qualification. Collection tensor batches must include ``state``,
``motion_id``, ``reference_frame`` and ``reference_phase`` as well as policy
inputs. Missing row evidence is refused before a physics step.
Saved-reset IEEE CUDA parity passes. This is not full trajectory equivalence.

The wall-clock, inactivity and stored-output budgets are reported truthfully.
A batched run that hits a budget writes ``partial: true`` and a
``stop_reason``, prints the stop reason to standard error, and exits non-zero
(1). Do not merge such a store. The stored-output limit is measured only with
``--stream-rows`` and is checked after each published chunk and the final flush.
A smaller ``--max-rows-per-append`` reduces the publication granularity.
``--progress PATH`` appends one JSON progress record per
event, and the final record carries the same stop reason. The report records
``policy_device`` (the resolved device the runtime used) and
``policy_device_setting`` (the requested setting) in both the batched and the
scalar report.

Inspect and replay
------------------

Run both checks on every store. These commands do not start a simulator: they
validate one shard per worker and return the same JSON payload keys as before.

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion inspect \
     --contract "$CONTRACT" --artifact-root "$ROOT" --store "$STORE" --jobs 4 \
     > "${STORE%/store}/offline-inspect.json"

   uv run --no-sync python -m mjlab.scripts.diffusion replay \
     --contract "$CONTRACT" --artifact-root "$ROOT" --store "$STORE" --jobs 4 \
     > "${STORE%/store}/offline-replay.json"

Require these results before preprocessing:

* ``unique_identities: true``;
* the inspected row count equals the stored report count;
* the contract SHA matches the active contract;
* replay has ``ok: true`` and ``errors: []``; and
* no shard hash or finite-value error exists.

Merge worker stores
-------------------

Use ``diffusion_merge`` only after each input store passes inspect and replay.
The command accepts ``--input`` more than once. It refuses an existing output,
stages the merged store in a sibling directory and publishes it with one atomic
rename. The default copies validated source shards byte for byte;
``--legacy-row-merge`` keeps the historical row-level rewrite. The default
output shard bound follows the largest input shard, so a byte-preserving merge
never splits a shard.

.. code-block:: bash

   export STAMP=$(date -u +%Y%m%dT%H%M%SZ)
   export DATASET=$ROOT/logs/diffusion/d1-vgk8-expanded-$STAMP
   mkdir -p "$DATASET/offline"

   uv run --no-sync python -m mjlab.scripts.diffusion_merge \
     --contract "$CONTRACT" \
     --input /path/to/store-worker-0 \
     --input /path/to/store-worker-1 \
     --output "$DATASET/store"

Inspect and replay the merged store again. Keep the original worker stores and
all failed attempts. Do not overwrite the original VGK8 dataset.

The ``prepare_dataset.py`` tool below performs the per-input inspect and replay
and the report gates automatically before it merges, and refuses a partial or
stopped worker.

Build windows and statistics
----------------------------

Build grouped train, validation, and test windows from the merged store. The
``--jobs`` count parallels the index build:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion build \
     --contract "$CONTRACT" \
     --artifact-root "$ROOT" \
     --store "$DATASET/store" \
     --assignments "$DATASET/offline/assignments.json" \
     --report "$DATASET/offline/dataset.json" \
     --jobs 4

Persist the window index once. The command writes the index, reads it back and
refuses a success exit unless the reloaded index reproduces the built one:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion build-index \
     --dataset-dir "$DATASET" \
     --contract "$CONTRACT" \
     --jobs 4

Fit statistics from the training split only. Pass ``--index`` to reuse the
persisted index instead of rebuilding it; the loader refuses an index that does
not describe the given store and assignments. The ``--jobs`` count uses
disjoint shard ranges, and ``--batch-windows`` bounds one accumulator update
(the module default applies when it is omitted):

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion stats \
     --contract "$CONTRACT" \
     --artifact-root "$ROOT" \
     --store "$DATASET/store" \
     --assignments "$DATASET/offline/assignments.json" \
     --bundle "$DATASET/offline/projection.npz" \
     --report "$DATASET/offline/stats.json" \
     --index "$DATASET/offline/window_index.npz" \
     --jobs 4

The split assignment is made before windowing. Clean and OU pair families must
stay in one split. Validation and test data must not fit the projection or
normalization statistics. The batched fit accumulates ``--batch-windows``
windows per accumulator update, so its floating-point reduction order differs
from the historical per-window fit even at ``--jobs 1``. The statistics are
therefore equal to the sequential fit to floating-point tolerance, not bit for
bit, and ``statistics_sha256`` can change for any fit through this path. Treat
that hash as a new artifact identity, not as a reproduction of the old one:

* keep every historical frozen ``projection.npz`` and its recorded hash; do not
  overwrite a frozen statistics file with a refit;
* a checkpoint or evaluation that pins the old statistics hash must use its
  existing ``projection.npz``, not a fresh fit;
* a token cache keyed by the old statistics hash is stale. Rebuild it from the
  projection file the run actually uses, or keep the cache that matches it; and
* ``--jobs`` above 1 adds a partition on top of the batched reduction, so it
  changes bits for the same reason.

Prepare a dataset in one pass
-----------------------------

The ``scripts/experiments/diffusion_d1_throughput/prepare_dataset.py`` entry
point runs the merge, dataset, statistics and cache phases in order. It uses
bounded worker counts, prepares the token caches with the existing lifecycle
controls and never starts training:

.. code-block:: bash

   uv run --no-sync python \
     "$ROOT/scripts/experiments/diffusion_d1_throughput/prepare_dataset.py" \
     --contract "$CONTRACT" \
     --dataset-dir "$DATASET" \
     --input /path/to/store-worker-0 \
     --input /path/to/store-worker-1 \
     --jobs 8 --cache-jobs 1 --max-cache-gib 8 \
     --report "$DATASET/offline/prepare.json"

Each worker needs its completed collection report (``<run>/report.json``). The
tool refuses a partial report, a stop reason, a contract mismatch, a row
accounting mismatch, a duplicate identity, an unreadable store and an existing
dataset store. The merged store, the window index and the JSON reports are
staged and published atomically; the projection bundle is written by
``ProjectionBundle.save``. It prints the training command with
``--require-prepared-cache`` when it finishes, but it does not run it.

Build and validate token caches
--------------------------------

Build all caches before training. Use a cache ceiling that is larger than the
largest expected split. Eight GiB was sufficient for the expanded VGK8 dataset:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion build-token-cache \
     --dataset-dir "$DATASET" \
     --contract "$CONTRACT" \
     --splits train,validation,test \
     --jobs 1 \
     --max-cache-gib 8 \
     --json "$DATASET/offline/token-cache.json"

Validate without writing:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion build-token-cache \
     --dataset-dir "$DATASET" \
     --contract "$CONTRACT" \
     --splits train,validation,test \
     --validate-only \
     > "$DATASET/offline/token-cache-validate.log"

Training must use ``--require-prepared-cache``. This prevents a training
process from silently starting a cache builder.

D2 training
-----------

The baseline VGK8 training configuration is:

.. code-block:: text

   configs/diffusion/x2_50hz_train_vgk8.yaml

It uses 1000 epochs, effective batch size 512, microbatch size 128,
accumulation 4, learning rate ``1e-4``, BF16 mixed precision, and EMA. The
training command requires an explicit budget. ``--allow-long-run`` is required
for the full budget:

.. code-block:: bash

   export STAMP=$(date -u +%Y%m%dT%H%M%SZ)
   export RUN=$ROOT/logs/diffusion/d2-vgk8-baseline-$STAMP
   export CUDA_VISIBLE_DEVICES=3

   uv run --no-sync python -m mjlab.scripts.diffusion train \
     --dataset-dir "$DATASET" \
     --config "$TRAIN_CONFIG" \
     --contract "$CONTRACT" \
     --out "$RUN" \
     --device cuda:0 \
     --epochs 1000 \
     --allow-long-run \
     --require-prepared-cache \
     --max-cache-gib 8 \
     --json "$RUN/train.json"

Use a new output directory for every seed and training arm. Do not resume a
complete-history baseline from a dropout arm. The history-dropout arms use
fresh initialization and these separate configurations:

.. code-block:: text

   configs/diffusion/x2_50hz_train_vgk8_history_dropout20.yaml
   configs/diffusion/x2_50hz_train_vgk8_history_dropout50.yaml

History dropout hides both state and latent history in rows ``0:8``. It does
not change the current row, future rows, clean targets, or the D1 contract.
Compare dropout arms only with matched offline protocols.

For a bounded software or GPU smoke, replace ``--epochs 1000`` with an explicit
small ``--max-updates`` value and use a separate output directory. Do not infer
full-run behavior from a smoke.

Offline checkpoint evaluation
-----------------------------

``evaluate-offline`` scores a checkpoint without constructing a simulator. Run
validation and test evaluation separately:

.. code-block:: bash

   export CHECKPOINT=$RUN/checkpoint-best.pt

   CUDA_VISIBLE_DEVICES=3 uv run --no-sync python -m mjlab.scripts.diffusion \
     evaluate-offline \
     --run "$RUN" \
     --checkpoint "$CHECKPOINT" \
     --contract "$CONTRACT" \
     --split validation \
     --device cuda:0 \
     --sampler ddim \
     --start-level 1000 \
     --seed 0 \
     --json "$RUN/evaluate-validation.json"

   CUDA_VISIBLE_DEVICES=3 uv run --no-sync python -m mjlab.scripts.diffusion \
     evaluate-offline \
     --run "$RUN" \
     --checkpoint "$CHECKPOINT" \
     --contract "$CONTRACT" \
     --split test \
     --device cuda:0 \
     --sampler ddim \
     --start-level 1000 \
     --seed 0 \
     --json "$RUN/evaluate-test.json"

Use ``--max-windows`` only for a bounded smoke. A full evaluation must record
the split name, checkpoint hash, sampler, start level, seed, and window count.
Report projected-state, identity-state, latent, and token metrics separately.
A scalar MSE is not enough because the projection emphasizes selected state
rows.

The optional ``--history-condition-dropout-probability`` flag is an offline
stress condition. It is not a replacement for a checkpoint trained with the
same dropout policy and is not a closed-loop evaluation.

D3 closed-loop simulation
--------------------------

D3 simulation is the first closed-loop stage. It constructs one X2 MuJoCo
simulation and executes the frozen VAE bootstrap followed by the diffusion
continuation. It is simulation evidence only. It is not hardware evidence and
does not qualify real-time control, stability, deployment, or physical quality.

Use ``view.py`` for the browser-based diagnostic. It runs one selected branch.
The default branch is ``diffusion``. The viewer starts paused and provides
``Play``, ``Pause``, ``Step``, and ``Reset Environment`` controls. A reset clears
policy history and performs a new eight-step VAE bootstrap. It does not mutate
the reference motion or permit a reference after handover.

The training command writes a portable sidecar beside every checkpoint. For the
best checkpoint, the file is:

.. code-block:: text

   checkpoint-best.pt
   checkpoint-best.yaml

The sidecar contains repository-relative references to the contract, projection
bundle, VAE checkpoint, VAE cohort manifest, cohort members, seed, and default
start frame. The training run also writes ``run-manifest.yaml`` before training.
The sidecar contains no server-local absolute path. The fetch step must copy the
sidecar and all referenced artifacts to the local repository layout.

The runner refuses a missing or malformed sidecar. SHA-256 command arguments are
not required. The runner still performs contract, dimension, schema, VAE, and
runtime compatibility checks. A sidecar does not replace those checks.

Run a simulator-free preflight first:

.. code-block:: bash

   export CHECKPOINT=$RUN/checkpoint-best.pt
   PYTHONPATH="$ROOT/src" uv run --no-sync python \
     "$ROOT/scripts/experiments/diffusion_d3_closed_loop/view.py" \
     --checkpoint "$CHECKPOINT" \
     --device cpu

Launch the local browser viewer after the preflight passes:

.. code-block:: bash

   PYTHONPATH="$ROOT/src" uv run --no-sync python \
     "$ROOT/scripts/experiments/diffusion_d3_closed_loop/view.py" \
     --checkpoint "$CHECKPOINT" \
     --execute

The default viewer address is ``http://127.0.0.1:8080``. The default session
uses 5000 continuation steps and a 1200-second wall budget. The output directory
is derived from the checkpoint run name. An existing output directory is allowed;
the evaluator and viewer overwrite their output files. Pass
``--output-dir`` to select a different directory.

The three VGK8 best checkpoints use the same D3 pattern:

.. list-table:: VGK8 D2 best checkpoints
   :header-rows: 1

   * - Training arm
     - Checkpoint
     - Default member
   * - No history dropout
     - ``logs/diffusion/d2-vgk8-double-20261009/seed0/checkpoint-best.pt``
     - ``vgk000``
   * - 20 percent history dropout
     - ``logs/diffusion/d2-vgk8-history-dropout20-20261009/seed0/checkpoint-best.pt``
     - ``vgk000``
   * - 50 percent history dropout
     - ``logs/diffusion/d2-vgk8-history-dropout50-20261009/seed0/checkpoint-best.pt``
     - ``vgk000``

A GUI run is one branch and one selected cohort member. It does not perform the
matched baseline-versus-diffusion acceptance comparison. Use ``evaluate.py``
when a bounded matched pair is required. Keep the GUI output, preflight output,
and any stop reason with the checkpoint sidecar.

Throughput benchmark matrix
---------------------------

The ``scripts/experiments/diffusion_d1_throughput/benchmark_matrix.py`` tool
generates a matched D1 collection matrix. It varies only the VAE policy device
(``cpu`` and ``cuda:0``) and the environment count (64, 128 and 256). Every arm
shares the contract, artifact root, collection config, runtime factory, trial
budget and streaming flags. The tool writes ``plan.json`` and ``commands.sh`` to
a new plan directory and refuses to overwrite an output. It never constructs a
runtime and never runs a benchmark:

.. code-block:: bash

   export STAMP=$(date -u +%Y%m%dT%H%M%SZ)
   uv run --no-sync python \
     "$ROOT/scripts/experiments/diffusion_d1_throughput/benchmark_matrix.py" \
     --contract "$CONTRACT" \
     --artifact-root "$ROOT" \
     --config "$COLLECTION_CONFIG" \
     --stamp "$STAMP" \
     --plan-dir "$ROOT/logs/diffusion/d1-vgk8-benchmark-$STAMP"

The inputs are read-only references. Each arm writes to its own new output
directory. The CPU device arm is the matched control, not a speed arm:
``--tensor-dispatch`` with a CPU policy device is a reference and parity path,
so it is not expected to be faster than the NumPy path. The 2026-10-12
benchmark measured CUDA throughput and same-reset IEEE parity; see the report
above. Re-test after a source, GPU, simulator or cohort change. The generator
still produces commands only and does not authorize execution.

Gates and evidence boundaries
-----------------------------

Before accepting a training run, keep these claims separate:

Before accepting a training run, keep these claims separate:

.. list-table:: Evidence boundaries
   :header-rows: 1

   * - Evidence
     - What it supports
   * - Collection report
     - Trial rows, qualification, and partial status.
   * - Inspect and replay
     - Stored-row identity and arithmetic integrity.
   * - Dataset report
     - Grouped split and window counts.
   * - Statistics report
     - Train-only projection and normalization provenance.
   * - Token-cache report
     - Prepared cache completeness and provenance.
   * - Training report
     - Optimizer lifecycle, losses, and checkpoints.
   * - Offline evaluation
     - Generation error on recorded windows.
   * - D3 simulation
     - Bounded closed-loop simulator behavior.
   * - Hardware/deployment
     - Physical robot and release behavior.

Passing an earlier row does not imply passing a later row. In particular,
collection integrity is not diffusion quality, offline quality is not
closed-loop stability, and closed-loop simulation is not hardware qualification.

Useful source documents
-----------------------

* D1 contract: ``docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml``
* D1 implementation: ``docs/plans/beyondmimic_diffusion_d1_implementation.md``
* D1 throughput plan: ``docs/plans/beyondmimic_diffusion_d1_throughput_implementation.md``
* Dataset preparation tool: ``scripts/experiments/diffusion_d1_throughput/prepare_dataset.py``
* Benchmark matrix tool: ``scripts/experiments/diffusion_d1_throughput/benchmark_matrix.py``
* D2 implementation: ``docs/plans/beyondmimic_diffusion_d2_implementation.md``
* D3 plan: ``docs/plans/beyondmimic_diffusion_d3_closed_loop_evaluation.md``
* CLI source: ``src/mjlab/scripts/diffusion.py``
* X2 runtime factory: ``src/mjlab/tasks/tracking/diffusion/runtime_x2.py``
* VAE and teacher usage: ``docs/source/x2_tennis_distillation.rst``
