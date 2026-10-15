X2 x0 Flow-Matching Training
============================

This page is the runbook for the opt-in X2 x0 flow-matching family. It covers
bounded training, offline generated-token evaluation and the optional closed-loop
simulation path in the existing D3 runner. It does not authorize a
run. Every command below starts work only when an operator executes it.

The flow family is separate from the diffusion family. It writes flow-format
checkpoints and never reads or writes a diffusion schedule, sidecar or
checkpoint. The frozen X2 VAE, the D1 token caches and the D1 statistics are
shared read-only.

Mathematical contract
---------------------

Use the integer level ``k`` in ``[0, max_level]``. The noise fraction is
``s = k / max_level``. For every state and latent slice, draw an independent
Gaussian ``e`` and form the linear path:

``x_s = (1 - s) * x0 + s * e``

Level 0 is exactly clean. Level ``max_level`` is exactly Gaussian. The network
predicts ``x0_hat``. Training draws independent levels for every row and for the
state and latent slices, including the history rows.

``history_state_dropout_probability`` is optional and matches the diffusion
family. When it is non-zero, a whole window's history pair (the first ``8`` rows
of the state and latent slices) is raised to level ``max_level``, which is pure
Gaussian, so that window cannot read its true history. The decision is one draw
per window. The default is ``0.0``, and the example config sets it explicitly.
Evaluation never applies the mask, so held-out losses stay comparable across
dropout settings. The run reports ``history_state_dropout_fraction`` per epoch
and in ``train-report.json``.

Sampling descends from level ``max_level`` to level 0 with the Euler update:

``x_r = (r / s) * x_s + (1 - r / s) * x0_hat``

The network is never called at source level 0. Known conditioning entries stay
bit-identical at every call. The default grid uses 20 calls with a level step of
50 for the default ``max_level=1000``; for a smaller process it uses
``min(20, max_level)`` calls so level 0 stays reachable.

Flow artifacts record ``prediction_type=x0``, ``path=linear_gaussian``,
``loss=x0_mse`` and ``time_encoding=integer_noise_fraction``.

Inputs
------

* Flow training config: ``configs/flow_matching/x2_50hz_train.yaml``.
* Frozen data contract:
  ``docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml``.
* A D1 dataset directory with split assignments and train-only statistics.

The model dimensions come from the frozen contract through
``DenoiserSettings.from_contract``. The flow config holds only the linear process
field ``max_level`` and the shared training fields. It holds no ``noise``
section; ``scheduler: cosine`` is the learning-rate schedule only.

Run the commands with ``uv run``.

Step 1: prepare the token caches
--------------------------------

The flow CLI validates caches read-only. Prepare a missing cache with the shared
diffusion command:

.. code-block:: bash

   uv run python -m mjlab.scripts.diffusion build-token-cache \
     --dataset-dir DATASET --contract CONTRACT --splits train,validation

Step 2: bounded training
------------------------

Training requires an explicit ``--max-updates`` or ``--epochs`` budget. A budget
above 200,000 updates also requires ``--allow-long-run``.

.. code-block:: bash

   uv run python -m mjlab.scripts.flow_matching train \
     --dataset-dir DATASET \
     --config configs/flow_matching/x2_50hz_train.yaml \
     --contract CONTRACT \
     --out logs/flow/RUN --device cuda:0 --max-updates <N> \
     --json logs/flow/RUN/train.json

Training writes ``flow-run.json`` (schema ``mjlab-flow-matching-run-v1``),
``resolved-config.json``, ``train-report.json``, ``metrics.jsonl``,
``checkpoint-last.pt``, ``checkpoint-best.pt`` and optional periodic checkpoints.
The flow checkpoint records the ``max_level`` process identity, the data
contract identity, the dataset and statistics identities, the model identity and
the noise and sampler generator states. A resume rejects any identity mismatch.

Step 3: offline generation evaluation
-------------------------------------

.. code-block:: bash

   uv run python -m mjlab.scripts.flow_matching evaluate-offline \
     --run logs/flow/RUN --split validation --device cuda:0

The evaluator runs the sampler on the fixed D1 entries and scores the generated
unknown entries only. It reports ``model_unknown_state_mse``,
``model_unknown_latent_mse`` and ``model_unknown_token_mse``, plus the
``copy_current_state``, ``copy_current_latent``, ``zero_latent``,
``prior_only_state`` and ``prior_only_latent`` baselines and their gate booleans.
It writes ``<run>/flow-generation-<split>.json``. It never reports a training
loss or a velocity error.

A baseline that predicts one slice only is scored on that slice alone, so its
``<name>_unknown_token_mse`` entry is the mean over that slice, not over all
tokens. For example ``copy_current_state_unknown_token_mse`` covers unknown
state entries only, and ``copy_current_latent_unknown_token_mse`` covers unknown
latent entries only. The gates compare the slice-specific
``model_unknown_state_mse`` and ``model_unknown_latent_mse`` values, so the
restricted token means never affect a gate.

``--split`` defaults to ``validation``. Use ``--split test`` with
``--checkpoint <run>/checkpoint-last.pt`` for a final report.

Closed-loop simulation in the diffusion runner
----------------------------------------------

``scripts/experiments/diffusion_d3_closed_loop/evaluate.py`` and its siblings
``evaluate_parallel.py`` and ``view.py`` accept
``--model-family diffusion|flow``. The default is ``diffusion``, so every
existing command keeps its behavior. With ``flow`` the CLI loads a flow
checkpoint, injects the flow Euler sampler into the existing continuations, and
reports ``sampler_method: euler_x0`` instead of DDIM. The environment, VAE
bootstrap, measured history, decoder, termination rules, metrics, recorder and
browser playback are shared with the diffusion path.

A flow run takes no diffusion schedule and no diffusion inference grid. It also
rejects ``--task-config``: the D4 sparse task guidance is implemented for the
DDIM sampler only. The rejection happens in preflight, before any model or
environment is constructed.

The command needs a YAML sidecar beside the checkpoint. The default path is the
checkpoint path with the ``.yaml`` suffix (so ``checkpoint-best.pt`` reads
``checkpoint-best.yaml``); ``--checkpoint-config`` selects another file. A flow
sidecar uses the same fields as the diffusion sidecar and the schema
``mjlab-d3-flow-matching-checkpoint-v1``:

.. code-block:: yaml

   schema: mjlab-d3-flow-matching-checkpoint-v1
   repo_root: .                     # relative to the sidecar file when not absolute
   contract: docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml
   projection: <path>/projection.npz
   vae_checkpoint: logs/distillation/mixed-10k/checkpoint-final.pt
   manifest: configs/distillation/x2_tennis_mixed.yaml
   member: vgk005
   seed: 0
   start_frame: 45

The sidecar schema and ``--model-family`` must agree. A flow sidecar with the
default diffusion family fails closed, and a diffusion sidecar with
``--model-family flow`` fails closed. The checkpoint format is the second gate:
the diffusion loader rejects a flow checkpoint and the flow loader rejects a
diffusion checkpoint. Sidecar paths are explicit. The parent creates these local
ignored sidecars; this page only defines their schema.

Simulator-free preflight. This loads the sidecar and the frozen flow model and
constructs no environment:

.. code-block:: bash

   PYTHONPATH="$PWD/src" uv run --no-sync python \
     scripts/experiments/diffusion_d3_closed_loop/evaluate.py \
     --model-family flow \
     --checkpoint logs/flow/RUN/seed0/checkpoint-best.pt \
     --device cpu

One bounded matched pair. The learned branch is still recorded under the
existing ``diffusion`` key of ``summary.json``, so read
``provenance.sampler.model_family`` to identify the family:

.. code-block:: bash

   PYTHONPATH="$PWD/src" uv run --no-sync python \
     scripts/experiments/diffusion_d3_closed_loop/evaluate.py \
     --model-family flow \
     --checkpoint logs/flow/RUN/seed0/checkpoint-best.pt \
     --continuation-steps 25 \
     --max-wall-seconds 1200 \
     --max-output-bytes 134217728 \
     --device cuda:0 \
     --output-dir logs/diagnostics/flow-closed-loop/smoke-vgk005-seed0 \
     --execute

The same flag is accepted by the batched driver and by the viewer. The viewer
keeps ``--branch diffusion`` for the learned branch, because the branch choice is
baseline versus learned and the family selects the learned model:

.. code-block:: bash

   PYTHONPATH="$PWD/src" uv run --no-sync python \
     scripts/experiments/diffusion_d3_closed_loop/evaluate_parallel.py \
     --model-family flow \
     --checkpoint logs/flow/RUN/seed0/checkpoint-best.pt \
     --members vgk005 --num-envs 8 --steps 25 \
     --device cuda:0 \
     --output-dir logs/diagnostics/flow-closed-loop/parallel-smoke \
     --execute

``run_model_ladder.sh`` and ``summarize_models.py`` keep working unchanged. A
flow wave passes the family through the driver's extra-argument channel:

.. code-block:: bash

   EXTRA="--model-family flow" scripts/experiments/diffusion_d3_closed_loop/run_model_ladder.sh \
     --summarize flow-400k:logs/flow/RUN/seed0/checkpoint-best.pt

Multi-environment parallel sweep
--------------------------------

The batched driver is the parallel multi-environment path. ``--num-envs``
selects the row count, and one process per GPU is enough because every row
batches into the same solver call. The row budget does not change the number of
network calls: one Euler update is one batched call for all rows.

.. code-block:: bash

   MOTIONS=vgk000,vgk001,vgk002,vgk003 GPUS=0,1,2,3 NUM_ENVS=256 \
     EXTRA="--model-family flow" PYTHON=.venv/bin/python \
     bash scripts/experiments/diffusion_d3_closed_loop/run_model_ladder.sh \
       flow-baseline:$PWD/logs/flow/RUN/output/checkpoint-best.pt

Run one wave per four motions when four cards host the sweep, so each card
holds one process. Repeat with the other motion group and the other checkpoint,
keep one ``--output-root`` for the whole comparison, and print the table with
``summarize_models.py --root <output-root>``. 256 rows per motion is the
convention used for the diffusion comparison, so a flow table is comparable to
it. Rows differ only by per-row domain randomization and per-row sampler noise.

A server checkout is a separate tree. It must contain every module the driver
imports, including ``mjlab.tasks.tracking.diffusion.state_recording`` and
``mjlab.tasks.tracking.flow_matching.closed_loop``. Verify with an explicit
import check before spending GPU time, because a missing module fails every
wave at start-up rather than during preflight.

Euler step count
----------------

The flow sampler resolves the level scale from ``max_level`` down to 0 in a
fixed number of Euler steps. ``--flow-euler-steps N`` sets that count for the
evaluator, the parallel driver and the viewer. **The default is 6.**

**Six steps is the chosen count for this project.** That decision rests on two
measurements from 2026-10-15 (+08):

- Closed loop, 256 environments per motion over eight motions: 6, 7, 8, 9, 10,
  15 and 20 steps are statistically indistinguishable (14.1 % to 17.4 % falls),
  while 5 steps is measurably worse for the baseline (20.7 %) and 1-3 steps are
  bad (57.4 %, 22.0 %). The same configuration run twice at 10 steps moved its own
  total by 1.2 points, which bounds the noise.
- Orin latency, TensorRT execution provider, batch 1, GPU clock locked to
  1173 MHz: 6 steps **15.82 ms** (79 % of the 20 ms control period), 7 steps
  19.37 ms, 10 steps 24.92 ms, 20 steps 56.83 ms. Unlocked, 6 steps is 16.46 ms.

Pass ``--flow-euler-steps N`` to override, or `solver_steps=N` to
`load_flow_denoiser`. The closed-loop default is
`DEFAULT_CLOSED_LOOP_SOLVER_STEPS = 6`
in ``diffusion/replay_integration.py``. The flow package's own ``DEFAULT_CALLS``
stays 20 because it is the offline generation-scoring grid, so offline scores
stay comparable with the earlier runs.
An open question for a real 50 Hz loop: the 6-step p90 on the Orin is 25.5 ms,
above the period, so the loop still needs the asynchronous planner or a locked
clock.

.. code-block:: bash

   PYTHONPATH="$PWD/src" uv run --no-sync python \
     scripts/experiments/diffusion_d3_closed_loop/evaluate.py \
     --model-family flow \
     --checkpoint logs/flow/RUN/output/checkpoint-best.pt \
     --flow-euler-steps 6 \
     --device cpu

.. code-block:: bash

   EXTRA="--model-family flow --flow-euler-steps 6" \
     scripts/experiments/diffusion_d3_closed_loop/run_model_ladder.sh \
       --num-envs 256 flow-baseline:logs/flow/RUN/output/checkpoint-best.pt

The count is an inference-time choice, not a checkpoint property. Training
samples the whole level scale, so every count descends the same trained path.
A smaller count is cheaper and coarser; a larger count is finer and slower.
The run report always carries the grid actually built, so a run cannot claim the
default after another count was requested. The diffusion family keeps its frozen
20 DDIM updates and refuses the flag.

The count must be a positive integer and must not exceed the checkpoint process
``max_level``. A larger request cannot form a strictly descending integer grid
and fails closed with that reason.

``inspect_sequence.py`` shares the evaluator parser but rejects
``--model-family flow`` before loading artifacts. Its sequence sampling remains
DDIM-only. Use ``evaluate.py`` or ``view.py`` for a flow run.

Reported sampler facts
----------------------

The preflight report and the run provenance both carry a ``sampler`` mapping.
For both families it names ``model_family``, ``sampler_method``,
``inference_grid`` and ``solver_updates``. A diffusion run reports
``sampler_method: ddim`` with ``ddim_updates: 20`` and start level ``1000``. A
flow run reports ``sampler_method: euler_x0``, the checkpoint's real
``process_identity``, the actual ``FlowGrid`` levels, ``solver_updates`` equal to
the grid call count, and start level equal to the process ``max_level``. A flow
report never contains ``ddim_updates``, and flow updates are never called DDIM.
A flow checkpoint has no diffusion schedule, no diffusion grid and no cosine
schedule, so none is fabricated.

Scope of the claims
-------------------

The parent validates the closed-loop integration on 2026-10-15 (+08).
The focused flow and diffusion regression suite passes 101 tests.
Both local VGK36 best checkpoints pass real-artifact preflight with EMA weights.
The baseline checkpoint completes a real single-row matched pair and an eight-row
simulation. Each uses eight VAE bootstrap ticks and 25 flow continuation ticks.
Each plan uses 20 Euler updates, because that run predates the six-step default.
All eight rows reach the short step budget.
This is integration evidence, not a full-motion stability result.

State recording works. Playback reconstructs the sampled body poses with a worst
error of 3.050e-07 m. Viewer preflight passes; interactive browser behavior is not
tested. The acceptance record is ``docs/plans/x2_flow_matching_closed_loop.md``.
Raw evidence is under
``logs/diagnostics/flow-closed-loop-acceptance-20261015T0125+0800/``.

Dropout20 physics, a full motion sweep, controller integration, ONNX export, and
hardware behavior are not tested by this change. No x2_docker, vendor simulator,
hardware, export or remote job is used. The flow-model module files are not part
of the run's module fingerprint list, which currently covers the diffusion and
distillation modules; the executed CLI file itself is fingerprinted.
