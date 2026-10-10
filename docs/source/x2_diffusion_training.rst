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

The managed server test used physical GPU 3 on
``10.14.64.48:32221``. The process sees that GPU as logical ``cuda:0`` because
of ``CUDA_VISIBLE_DEVICES``.

The current best tested collection setting is:

.. code-block:: text

   physical GPU:       3
   CUDA_VISIBLE_DEVICES=3
   runtime device:     cuda:0
   max-envs:           64
   OMP_NUM_THREADS:    1
   MKL_NUM_THREADS:    1
   workers:            1
   teachers:           vgk000--vgk007

The setting was selected from a bounded eight-teacher sweep on 2026-10-10
(+08). It achieved 101.457 accepted rows per second. The 128- and
256-environment tests did not improve the result. The 256-environment arm
achieved 99.736 accepted rows per second.

This is a server-specific engineering default. Re-test it after a source,
VAE, manifest, GPU, or simulator change. The result is a throughput result,
not a policy-quality result. Use ``nvidia-smi`` before launch and use a process
supervisor with an inactivity and wall-time limit for long collection jobs.

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
       --max-envs 64 \
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

Inspect and replay
------------------

Run both checks on every store. These commands do not start a simulator:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion inspect \
     --contract "$CONTRACT" --artifact-root "$ROOT" --store "$STORE" \
     > "${STORE%/store}/offline-inspect.json"

   uv run --no-sync python -m mjlab.scripts.diffusion replay \
     --contract "$CONTRACT" --artifact-root "$ROOT" --store "$STORE" \
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
The command accepts ``--input`` more than once:

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

Build windows and statistics
----------------------------

Build grouped train, validation, and test windows from the merged store:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion build \
     --contract "$CONTRACT" \
     --artifact-root "$ROOT" \
     --store "$DATASET/store" \
     --assignments "$DATASET/offline/assignments.json" \
     --report "$DATASET/offline/dataset.json"

Fit statistics from the training split only:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion stats \
     --contract "$CONTRACT" \
     --artifact-root "$ROOT" \
     --store "$DATASET/store" \
     --assignments "$DATASET/offline/assignments.json" \
     --bundle "$DATASET/offline/projection.npz" \
     --report "$DATASET/offline/stats.json"

The split assignment is made before windowing. Clean and OU pair families must
stay in one split. Validation and test data must not fit the projection or
normalization statistics.

Build and validate token caches
--------------------------------

Build all caches before training. Use a cache ceiling that is larger than the
largest expected split. Eight GiB was sufficient for the expanded VGK8 dataset:

.. code-block:: bash

   uv run --no-sync python -m mjlab.scripts.diffusion build-token-cache \
     --dataset-dir "$DATASET" \
     --contract "$CONTRACT" \
     --splits train,validation,test \
     --jobs 4 \
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
* D2 implementation: ``docs/plans/beyondmimic_diffusion_d2_implementation.md``
* D3 plan: ``docs/plans/beyondmimic_diffusion_d3_closed_loop_evaluation.md``
* CLI source: ``src/mjlab/scripts/diffusion.py``
* X2 runtime factory: ``src/mjlab/tasks/tracking/diffusion/runtime_x2.py``
* VAE and teacher usage: ``docs/source/x2_tennis_distillation.rst``
