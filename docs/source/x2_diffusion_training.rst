X2 VGK8 Diffusion Training Pipeline
====================================

This page is the runbook for the VGK8 state--latent diffusion pipeline. It
covers data collection, dataset preparation, D2 training, offline checkpoint
evaluation, and the D5 ONNX export for the controller lane. It does not
authorize a run. Every command below starts work only when an operator executes
it.

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
                 |
                 v
   D5 ONNX export for the controller lane

D3 closed-loop simulation is a separate gate. D2 offline evaluation does not
prove closed-loop stability, real-time performance, hardware quality, or
deployment readiness. The D5 export is a separate artifact step. It does not
authorize a controller run, a simulator engagement, or hardware use.

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

Parallel headless D3 sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~

``view.py`` runs one environment. Use ``evaluate_parallel.py`` to run the same
``--branch diffusion`` arm headless on many environment rows at once and to
count falls. The command reproduces the viewer arm: it resets every row to the
sidecar member and start frame, drives exactly eight measured VAE bootstrap
pairs per row, freezes the reference for the whole batch, and then runs the
frozen 20-step DDIM continuation per tick with one batched sampler call.

Run the sweep for every cohort member at 256 rows:

.. code-block:: bash

   export CHECKPOINT=logs/diffusion/d2-vgk8-history-dropout20-20261009/seed0/checkpoint-best.pt
   uv run python scripts/experiments/diffusion_d3_closed_loop/evaluate_parallel.py \
     --checkpoint "$CHECKPOINT" \
     --members all \
     --num-envs 256 \
     --execute

Without ``--execute`` the command prints a simulator-free preflight report.
The default run length is the remaining clip: ``clip_frames - 2 - start_frame -
8`` continuation ticks, so the episode covers the configured motion from the
configured start frame to its end. ``--steps`` overrides that length.
``--members`` accepts a comma list, ``all``, or ``configured`` (the sidecar
member). ``--fall-degrees`` defaults to 70.0, the viewer value.

Evidence is written per member under
``logs/diagnostics/diffusion-d3-parallel/parallel-<stamp>/``. Each member file
holds the row count, the step budget, the fall count and rate, the first-fall
tick per fallen row, the stop-reason counts, the per-row maximum tilt, and the
runtime provenance.

Stop reasons per row
^^^^^^^^^^^^^^^^^^^^

A row stops on the first of these events, in the viewer's order:

- ``physical_fall``: the root tilt exceeds ``--fall-degrees``;
- ``reset`` or ``teleport``: the runtime observed a reset or a reference
  discontinuity;
- ``terminated`` or ``timeout``: the task termination or the time-out flag;
- ``tick_discontinuity``, ``segment_change``, or ``period_discontinuity``: a
  batch-wide continuity break;
- ``reference_leakage``: a reference appeared after handover;
- ``step_budget``: the row completed its own step budget.

A row that stops keeps simulating, because the private environment sets
``auto_reset = False``. The driver sends a zero action to a stopped row and
counts only its first fall.

Fidelity check against the single-row path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--parity`` runs the real single-row runtime and ``VisualDiagnosticPlayback``
for the same member, start frame, and seed, and it also runs a small batched
batch that records row zero. It reports the maximum applied-action difference
per tick, the tick-zero conditioning and reference differences, and the stop
reason of both paths.

The parity batch is small on purpose. ``--parity-rows`` defaults to 8 and is
limited to 12. The task draws a per-row reset perturbation (RSI root pose,
velocity, and joint positions) and stratifies that draw once the batch is
large: measurements here show row zero of a 1-row, 2-row, 4-row, 6-row, 8-row,
or 12-row environment receives one realization, while row zero of a 16-row,
64-row, or 256-row environment receives a different one. Above the boundary a
row-zero comparison would compare two different realizations, not two code
paths, so the command refuses that request.

Two properties make the comparison meaningful below the boundary. First,
``sample_trajectory`` draws its noise in one ``torch.randn`` call of the whole
batch shape, so row zero of a wider batch consumes the first slice of the
stream. Second, on CUDA that slice equals a one-row draw at the same seed; on
CPU it does not, because the CPU normal generator does not preserve the prefix.
Run the parity check on the default device.

The raw world-position signature is not the test. It includes each row's
environment origin, which differs by a constant in a wider batch. The
origin-independent conditioning (99 channels) and reference (68 channels)
vectors are the test, and the record states whether they matched at tick zero.

What a row is, and what it is not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every row shares one member and one start frame. The rows differ through the
task's own per-row reset perturbation, through its per-row push events, and
through their own sampler noise. The command adds no initial-state perturbation
of its own. Do not present the row count as a number of independent initial
conditions, and do not present the result as a stability, deployment, or
hardware claim.

Comparing several checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``run_model_ladder.sh`` to run the same sweep for many checkpoints and
``summarize_models.py`` to print one comparison table. One wave is one
checkpoint. Inside a wave every motion gets its own process on its own GPU, so
a wave on the eight-card server takes about the time of one motion, 90 to
115 seconds. With a single GPU the script runs the motions in sequence.

.. code-block:: bash

   bash scripts/experiments/diffusion_d3_closed_loop/run_model_ladder.sh \
     --python /home/fushan/mjlab/.venv/bin/python \
     --gpus 0,1,2,3,4,5,6,7 \
     --output-root logs/diagnostics/diffusion-d3-models \
     vgk8-baseline-best:logs/diffusion/d2-vgk8-double-20261009/seed0/checkpoint-best.pt \
     vgk8-baseline-last:logs/diffusion/d2-vgk8-double-20261009/seed0/checkpoint-last.pt

Relevant options are ``--motions`` (default ``vgk000`` to ``vgk007``),
``--num-envs`` (default 256), ``--steps`` to override the tick budget,
``--extra`` for extra driver arguments, and ``--dry-run``. The script exits
non-zero if any motion fails and prints the one-line member summary of every
motion.

Then print the table:

.. code-block:: bash

   .venv/bin/python scripts/experiments/diffusion_d3_closed_loop/summarize_models.py \
     --extra-label vgk8-dropout20-best=logs/diagnostics/diffusion-d3-parallel-server

The tool reads every label directory under the evidence root and reports the
fall count per motion, the total, the rate, and the per-motion tick budget. It
marks an incomplete wave, so a partial sweep cannot pass for a finished one. Use
``--extra-label name=path`` for evidence outside the root.

Checkpoint sidecars
^^^^^^^^^^^^^^^^^^^

The evaluator requires a sidecar YAML beside every checkpoint. Some older runs
never wrote one. Backfill it from the run's own artifacts, and copy the minimal
schema:

.. code-block:: yaml

   schema: mjlab-d3-diffusion-checkpoint-v1
   repo_root: ../../../..
   contract: docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk8_contract.yaml
   projection: logs/diffusion/d1-vgk8-double-20261009/offline/projection.npz
   vae_checkpoint: logs/distillation/vae-cov-10k-20261008T154401Z/vae-vgk8-cov-on/output/checkpoint-final.pt
   manifest: configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v1.yaml
   member: vgk000
   seed: 0
   start_frame: 45

``repo_root`` is relative to the sidecar's own directory. Do not invent
``global_step``: write it only where the run states it. Mark the file as
backfilled in a comment, and require the preflight to pass for every checkpoint
before you spend GPU time. The preflight cross-checks the contract identity, the
projection statistics, and the VAE checkpoint, so a wrong path fails loudly.

Reduced contracts on the server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shared server checkout's D3 gate predates the contract-driven change: it
calls ``_require_frozen_contract`` and refuses the 129-D local-kinematic
contract. Do not overwrite the shared tree. Use the ``lk129`` worktree instead,
where the 129-D library support exists, copy in the gate commit's files, and put
that worktree on ``PYTHONPATH`` while the artifacts stay in the shared checkout:

.. code-block:: bash

   export PYTHONPATH=/home/fushan/mjlab-lk129/src
   cd /home/fushan/mjlab
   bash /home/fushan/mjlab-lk129/scripts/experiments/diffusion_d3_closed_loop/run_model_ladder.sh \
     --python /home/fushan/mjlab/.venv/bin/python \
     --output-root logs/diagnostics/diffusion-d3-models \
     vgk8-localkin129-best:logs/diffusion/d2-vgk8-localkin-1000ep-seed0-20261013T183840Z/checkpoint-best.pt

Running the driver from the worktree also selects that worktree's
``evaluate.py``, because the driver imports it as a sibling.

Reading the result
^^^^^^^^^^^^^^^^^^

Report the tick budget beside every rate. The budget differs by motion, from 257
to 398 continuation ticks, so a longer motion gives a policy more time to fall.
Read the per-motion cells, not only the totals.

A checkpoint comparison is descriptive unless the arms match. The measured
families differ in training data, in contract width, and in update budget, so a
rate difference between families does not attribute cause. State the arms in the
same table as the numbers.

A measured result on this tool (2026-10-14): every ``checkpoint-last.pt`` fell
far more often than its ``checkpoint-best.pt``, by a factor of 1.7 to 5.0, while
the validation loss moved only 1.37x to 1.42x. The two 36-teacher ``last``
checkpoints fell on 98.7 % and 99.5 % of 2048 rows each. Select the best
checkpoint for any downstream use, and treat the end of training as unusable for
these recipes.

One sweep is not reproducible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sweep is a diagnostic, not a measurement with a fixed answer. Six runs of the
same motion (``vgk000``, 256 rows, the same checkpoint, the same host) fell in
this range:

.. list-table:: ``vgk000``, 256 rows, six runs of one configuration
   :header-rows: 1

   * - Recording
     - Fall count of 256
   * - Off
     - 122, 132, 137
   * - On
     - 134, 135, 139

The spread is **17 rows, or 6.6 %**, for one motion. Two runs of the same
8-motion arm gave 404 and 401 falls of 2048 (19.7 % and 19.6 %). Recording does
not change the outcome: the recorded runs fall inside the range of the
unrecorded runs.

Consequences for a comparison:

- A single run gives each motion a rate that is uncertain by about 4 %. Do not
  read a small per-motion difference between two models.
- Two whole arms that differ by about 2 percentage points, or 40 rows of 2048,
  are not separated by one run each. Repeat a close arm, or say that the two arms
  are tied.
- A difference of 7 percentage points or more, and every best-to-last gap, is far
  larger than this spread and stands as reported.

The cause is not measured. Small floating-point differences in the sampler or in
the contact solver can change which episodes cross the fall threshold. Enabling
deterministic kernels and repeating every arm would test that.

Recording robot states
^^^^^^^^^^^^^^^^^^^^^^

The sweep counts falls and keeps no trajectory. Pass ``--record-states DIR`` to
write one ``<member>.npz`` per motion with the per-tick robot state, so the
motion can be rendered and inspected. Recording is off by default, and the tool
refuses to overwrite an existing file.

.. code-block:: bash

   bash scripts/experiments/diffusion_d3_closed_loop/run_model_ladder.sh \
     --python /home/fushan/mjlab/.venv/bin/python \
     --motions vgk000 --num-envs 256 \
     --extra "--record-states logs/diagnostics/d3-states" \
     vgk8-baseline-best:logs/diffusion/d2-vgk8-double-20261009/seed0/checkpoint-best.pt

Every array has the shape ``[ticks, rows, width]``, so row ``r`` is one episode
and tick ``t`` is one control step. The eight channel groups hold 107 values per
row per tick:

.. list-table:: Recorded channels
   :header-rows: 1

   * - Group
     - Width
   * - ``root_position``
     - 3
   * - ``root_quaternion_wxyz``
     - 4
   * - ``root_linear_velocity``
     - 3
   * - ``root_angular_velocity``
     - 3
   * - ``joint_position``
     - 31
   * - ``joint_velocity``
     - 31
   * - ``action``
     - 31
   * - ``tilt_degrees``
     - 1

The file also carries ``tick``, ``phase`` (0 bootstrap, 1 continuation),
``active``, ``joint_names``, ``body_names``, ``member``, ``start_frame``,
``clip_frames``, ``bootstrap_ticks``, ``period_seconds``, ``fall_first_tick``,
``stop_code`` with ``stop_labels``, and ``complete``.

``--record-rows N`` keeps only the first N rows and ``--record-stride K`` keeps
every K-th tick, for a smaller file. ``--record-bodies`` adds
``body_positions`` and ``body_linear_velocities`` (60 values each). The default
107-value set at 256 rows and about 400 ticks is about 44 MB per motion; the
body set doubles it. ``logs/`` is git-ignored. A measured run at 256 rows and 60
ticks recorded 430 bytes per row-tick, which is 44.7 MB for a full 406-tick
motion, and cost 43.0 s against 42.2 s without recording.

Use your own tool to render the file, or check first that the channels do
describe the robot:

.. code-block:: bash

   uv run python scripts/experiments/diffusion_d3_closed_loop/verify_recording.py \
     logs/diagnostics/d3-states/vgk000.npz

That tool writes the recorded root pose and joints back into a fresh
environment, forwards the kinematics, and compares the rebuilt body positions
against the recorded ones. It needs a recording made with ``--record-bodies``.
A measured run reported **4.8e-07 m** at a float32 rounding level, against a
repeat-read noise floor of 0.0 m.

**Do not build the joint channels from the actor observation.** The actor
``joint_pos`` term of this task family is a *relative* joint position with added
encoder bias and noise, and it can be one control step old. A recording built
from it rebuilt the body pose only to **0.14 m**, which is enough to make a
render misleading. The recorder reads the simulator's own joint position and
velocity for this reason.

Playing a recorded state file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``play_recording.py`` shows one recorded row in a browser. It loads the
standalone X2 MJCF, writes the recorded root pose and joint positions into the
model, and computes the kinematics. No physics runs and no policy, checkpoint,
VAE or dataset is needed, so the picture is exactly the recorded motion.

.. code-block:: bash

   uv run python scripts/experiments/diffusion_d3_closed_loop/play_recording.py \
     logs/diagnostics/d3-states-vgk000.npz

The default row is the row that falls first. Serve on another port with
``--port``, choose a row with ``--row``, start slower or faster with ``--rate``
(``1.0`` is real time at 50 control steps per second), and start looping with
``--loop``. The page has play, pause, single-step, seek, speed, loop and row
controls, plus a readout of the tick, the time, the tilt, the lowest body and
the row's fall tick.

``--row`` takes a row index or one of three names: ``first-fall`` (the
default), ``last-fall`` (the row that runs longest before it falls) and
``survivor`` (a row that never falls).

Playback stops 25 ticks after a fall by default, because the sweep stops
stepping a row once it falls and the rest of that row is a frozen state.
``--margin-ticks`` changes that. ``--no-recenter`` keeps the recorded world
position instead of moving the row's start to the origin.

Two checks run without a browser:

.. code-block:: bash

   # rebuild sampled ticks and compare with the recorded body positions
   uv run python scripts/experiments/diffusion_d3_closed_loop/play_recording.py \
     logs/diagnostics/d3-states-vgk000.npz --check

   # list the rows, falls first, then exit
   uv run python scripts/experiments/diffusion_d3_closed_loop/play_recording.py \
     logs/diagnostics/d3-states-vgk000.npz --list-rows

``--frames DIR`` writes one PNG per interesting tick instead of opening a
browser, which is useful for a report:

.. code-block:: bash

   uv run python scripts/experiments/diffusion_d3_closed_loop/play_recording.py \
     logs/diagnostics/d3-states-vgk000.npz --frames /tmp/fall-frames

``--gif PATH`` writes the whole played row as an animated GIF, which is the
handy way to check how a policy runs and then falls:

.. code-block:: bash

   for m in vgk000 vgk001 vgk002 vgk003 vgk004 vgk005 vgk006 vgk007; do
     uv run python scripts/experiments/diffusion_d3_closed_loop/play_recording.py \
       logs/diagnostics/d3-states/$m.npz --row last-fall --track \
       --gif logs/diagnostics/d3-states/gifs/$m-lastfall.gif
   done

Each frame carries a caption: member, row, time, tilt and the fall tick. Two
camera modes are available. Without ``--track`` the camera holds still and the
whole trajectory stays in view, which shows the travel and costs about
**1.9 MB** for 406 frames. With ``--track`` the camera follows the robot, so the
pose is large and clear, at about **9.5 MB** at 640x480 and **6.1 MB** at
480x360. A measured 406-frame GIF takes about **9 s** to render.

``--gif-stride K`` renders every K-th tick, which halves the file and sets the
frame delay to ``20 * K`` milliseconds. Use ``--frame-height`` and
``--frame-width`` to change the size. Nothing that exists is overwritten.

A measured run on ``vgk36-baseline-best``, motion ``vgk000``, row 176: the
reconstruction matched the recording to **4.8e-06 m**; the row stands at pelvis
0.628 m and tilt 5.1 deg, the fall threshold is passed at tick 150 (73.1 deg),
and by tick 160 the robot lies at 89.2 deg with the pelvis at 0.276 m. The
robot travels about 4 m while it falls, so both camera paths frame the whole
trajectory; a camera that only frames the robot start loses it.

D5 ONNX export and the controller lane
--------------------------------------

The D5 lane is the deployment engine. The controller loads ONNX graphs, not a
PyTorch checkpoint. Export one trained checkpoint with the parity gate, then
assemble the bundle that the controller loader accepts.

The gate reads the physical-state and token widths from the active contract. The
window geometry, the latent width, the conditioning width, the action width, and
the 20 DDIM updates stay frozen.

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=''
   export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
   export PYTHONPATH="$ROOT/src:$ROOT/tests"

   export CHECKPOINT=$RUN/checkpoint-best.pt
   export NORMALIZATION=logs/diffusion/d1-vgk8-localkin-double-20261013T183840Z/offline/projection.npz
   export OUT=logs/diagnostics/diffusion-d5-lane-$(date +%Y%m%dT%H%M%S)

   uv run --no-sync python "$ROOT/scripts/experiments/diffusion_d5_parity/validate.py" \
     --output-dir "$OUT" \
     --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_local_kinematic_vgk8_contract.yaml \
     --diffusion-checkpoint "$CHECKPOINT" \
     --normalization-bundle "$NORMALIZATION" \
     --vae-checkpoint logs/distillation/vae-cov-10k-20261008T154401Z/vae-vgk8-cov-on/output/checkpoint-final.pt \
     --vae-manifest configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v1.yaml \
     --vae-member vgk001

The gate writes three graphs and one report:

.. list-table:: Gate outputs
   :header-rows: 1

   * - File
     - Contents
   * - ``denoiser.onnx``
     - the lane graph; this file goes into the bundle
   * - ``denoiser_native.onnx``
     - the model at its own width; evidence only
   * - ``vae_decoder.onnx``
     - the decoder of the selected VAE member
   * - ``REPORT.json``
     - every check, every width, and every hash

Require ``status = pass``, no tolerance warning, and ``lane_adapter.probe``
equal to zero on both channel sets. Run the gate twice. A repeat run must
reproduce all three graph files byte for byte.

The controller lane fixes its widths at compile time
----------------------------------------------------

The X2 controller lane compiles one token layout (``diffusion_types.h``:
``kStateDim = 135``, ``kLatentDim = 32``, ``kTokenDim = 167``). The loader
refuses a descriptor that declares any other width, including a narrower
statistics array. A model trained on a reduced state representation therefore
cannot load directly.

Every reduced representation in this package is a channel selection of the
135-channel hybrid state. ``hybrid_to_local_kinematic`` keeps every retained
channel unchanged and removes only ``root_position`` and
``root_linear_velocity``. The 129-D ablation is 129 state channels plus the
32-channel latent, so 161 token channels.

Six filler channels take 129 to 135
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 129-D channel map into the 135-D lane state
   :header-rows: 1

   * - Model channel
     - Lane channel
     - Field
   * - 0:6
     - 3:9
     - ``root_rot6d``
   * - 6:9
     - 12:15
     - ``root_angular_velocity``
   * - 9:69
     - 15:75
     - ``body_positions``
   * - 69:129
     - 75:135
     - ``body_linear_velocities``
   * - 129:161
     - 135:167
     - the 32-channel latent

The lane state has six channels that the model does not own: lane channels 0, 1,
2 (``root_position``) and 9, 10, 11 (``root_linear_velocity``). Fill them with
dummy channels to reach the lane width of 135:

* In the graph, the six channels pass through unchanged and the model never
  reads them.
* In the statistics artifact, they carry mean ``0.0`` and standard deviation
  ``1.0``.

The boundary is exact for every channel the model reads, because the graph
gathers those channels and writes the model result back. It adds no arithmetic
to them. Nothing is fabricated: do not reconstruct root translation, do not pad
a reduced state back to 135, and do not upsample any signal.

Statistics artifact for a reduced layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``emit_stats.py`` places the reduced statistics at their lane positions and
fills the rest. Pass the layout explicitly, or read it from the gate report so
that no value is transcribed by hand:

.. code-block:: bash

   python3 "$X2/tools/diffusion_d5/emit_stats.py" \
     --projection "$NORMALIZATION" \
     --out "$BUILD/stats" \
     --source-report "$OUT/REPORT.json"

The tool refuses a layout that does not tile the source width, and it refuses a
group that the lane does not have. Its report lists the source layout, the
dropped lane groups, and the filled channels.

Assemble and install the bundle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse every asset that does not depend on the model from an accepted bundle of
the same VAE cohort, and prove the reuse. Export the bootstrap encoder from the
same VAE first:

.. code-block:: bash

   uv run --no-sync python "$X2/tools/diffusion_d5/export_bootstrap_encoder.py" \
     --repo-root "$ROOT" \
     --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_local_kinematic_vgk8_contract.yaml \
     --vae-checkpoint="$VAE" --vae-manifest="$MANIFEST" --vae-member vgk001 \
     --model-id "$VAE_MODEL_ID" --contract-id "$MEMBER_CONTRACT_ID" \
     --out "$BUILD/encoder"

A re-exported encoder must be byte-identical to the accepted encoder. A decoder
may differ in metadata only, because the export records the checkpoint identity
of its own arm. Compare decoder graphs with the metadata block removed.

.. code-block:: bash

   python3 "$X2/tools/diffusion_d5/build_bundle.py" --out "$BUILD/bundle" \
     --denoiser-onnx "$OUT/denoiser.onnx" \
     --encoder-onnx "$BUILD/encoder/encoder.onnx" \
     --decoder-onnx "$OUT/vae_decoder.onnx" \
     --stats-json "$BUILD/stats/stats.json" \
     --motion-json "$STORE/$COHORT/vgk001/motion.json" \
     --vae-contract-json "$STORE/$COHORT/vgk001/contract.json" \
     --vae-descriptor-json "$STORE/$ACCEPTED_BUNDLE/vae_descriptor.json" \
     --contract-identity "$MEMBER_CONTRACT_ID" \
     --vae-identity "$VAE_MODEL_ID" \
     --projection-identity "$STATISTICS_IDENTITY" \
     --model-identity "$CHECKPOINT_SHA256" \
     --source-revision-x2 "$X2_REVISION" \
     --source-revision-mjlab "$MJLAB_REVISION" \
     --bootstrap-member vgk001 \
     --frozen-root "$X2" --frozen-root "$ROOT"

Use the checkpoint file digest as ``--model-identity``, the source statistics
identity as ``--projection-identity``, the VAE member ``contract_id`` as
``--contract-identity``, and the VAE descriptor ``model_id`` as
``--vae-identity``. Put the output directory outside both repositories.

Install the bundle and check it with the read-only controller modes:

.. code-block:: bash

   cd "$X2"
   ./install_d5_bundle.sh "$BUILD/bundle" --name POLICY_NAME
   docker exec x2-humble bash -lc 'cd /work/x2_docker \
     && ./start_controller.sh --config x2_sim_diffusion_gpu.yaml --list-policies'
   docker exec x2-humble bash -lc 'cd /work/x2_docker \
     && ./start_controller.sh --config x2_sim_diffusion_gpu.yaml \
        --tracking-policy POLICY_NAME --check-config'

Add one flipped-byte negative control before you trust the preflight. The check
output does not name the diffusion policy, so a rejection is the proof that the
check reads the bundle. Restore the file and re-run the check afterwards. The
C++ loader can also be exercised on the host, without ROS and without ONNX
Runtime, by building ``x2_diffusion_bundle_tests`` from its CMake source list
and passing the bundle directory as the optional third argument.

GPU inference is a configuration choice, not a property of the artifact. The
bundle must declare ``backend.name: cpu_ort``. The lane selects the provider in
the configuration:

.. code-block:: yaml

   sim:
     diffusion:
       execution_provider:
         name: cuda
         device_id: 0

Reference motions for the guided (``a+8``) arm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A bundle can declare reference motions. The guided arm then constrains the
future state rows ``f+16`` and ``f+30`` from the selected reference during
denoising. Attach one table per cohort member and repeat the flag for each:

.. code-block:: bash

   python3 "$X2/tools/diffusion_d5/build_bundle.py" ... \
     --guidance-reference vgk000="$ACCEPTED/references/vgk000.json" \
     --guidance-reference vgk001="$ACCEPTED/references/vgk001.json"

The tool validates the cohort-member binding, copies each table to
``references/<ID>.json``, and writes the descriptor entry. A bundle without the
list keeps the unguided ``a+7`` path only.

A table is a world-state clip, not a state-layout artifact, so one table serves
both a full and a reduced contract. Export one from the original tracking NPZ:

.. code-block:: bash

   python3 "$X2/tools/diffusion_d5/export_guidance_reference.py" \
     --motion "$ORIGINAL_TRACKING_NPZ" \
     --member-contract "$STORE/$COHORT/<member>/contract.json" \
     --out "$BUILD/references/<id>.json"

A table must hold at least 17 frames at 50 Hz, the frozen body order, finite
values, and non-degenerate root quaternions. A non-bootstrap member must embed
its exact ``member_contract_json``. The bootstrap member needs no extra field,
and any embedded contract for it must be byte-identical to the loaded member
contract. Do not resample a clip.

For a reduced contract the model reads the 129 channels that it owns in those
fixed rows, so the reference pose, root orientation, and body velocities do
reach the model. The reference root translation does not, because the model has
no channel for it.

Verify the declarations before you rely on them. The C++ closure suite must
accept the real bundle, the guidance unit suite must pass, and a one-byte change
to a ``references/<id>.json`` file must be refused with
``guidance_references[i] declared sha256 does not match``. Restore the file and
re-run afterwards.

Limits of the lane adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~

* The bundle declares 135 state channels while the model reads 129. The six
  filler channels are unread in unguided continuation.
* The adapter is exact only for the channels that the model owns. It does not
  reconstruct the dropped channels, so a reduced model is not equivalent to a
  135-D model.
* Generated future ``root_position`` and ``root_linear_velocity`` channels are
  meaningless. A guided reference constrains future state rows, but for a
  reduced contract the model reads only its own channels of those rows, so the
  reference root translation never reaches the model. Never use the dropped
  channels in a pose cost, and treat guided tracking quality as unmeasured.
* Neither the full nor the reduced model is trained with keyframe inpainting.
  The trainer never fixes a future state row, so a guided engagement is an
  out-of-distribution overlay whose quality is unmeasured. This limit is the
  same for the accepted full-contract bundles.
* The CUDA execution provider enables TF32, so its arithmetic differs from the
  CPU parity numbers.
* A passing export and a passing preflight are software results. They do not
  measure closed-loop stability, real-time performance, or hardware quality.

A worked example, with hashes and a container preflight, is
``x2_docker/orchestration/LK129-D5-BUNDLE-20261014.md``.

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
