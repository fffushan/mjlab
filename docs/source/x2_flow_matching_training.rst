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

State-field mask
~~~~~~~~~~~~~~~~

``masked_state_fields`` removes named information from the model input. It is a
list of hybrid-state field names, not channel numbers:

``root_position``, ``root_rot6d``, ``root_linear_velocity``,
``root_angular_velocity``, ``body_positions``, ``body_linear_velocities``.

Each masked field is replaced by one constant on **every** row of the window,
and those channels are excluded from the loss mean. The token shape stays
``41 x 167``. The linear path samples one level per row, so a per-channel noise
level is not available; a constant is the substitute.

``masked_state_fill_value`` selects that constant in normalized units. Zero is
the normalized mean. Any constant carries no sample information, so the choice
does not change what the model can learn. A larger magnitude, for example
``6.0``, is easier to audit: stored tokens then show the mark explicitly, while a
real zero velocity cannot be distinguished from a masked zero.

The model projects the token with a linear layer and uses pre-norm Transformer
blocks, so a constant fill becomes a fixed offset in the hidden state. A value
near six standard deviations is inside the range that pre-norm absorbs; do not
use an extreme sentinel such as ``1e6``.

The reported channels are ``masked_state_channels`` and
``masked_state_fraction_of_state`` in ``train-report.json``. Root translation is
6 channels: ``root_position`` 0-2 and ``root_linear_velocity`` 9-11, which is
4.4 percent of the state and 3.6 percent of the token.

A masked run is only consistent with a matching inference call. Because the mask
is a constant, the caller should keep those channels **known** in the sampler
``clean_mask`` and supply the same ``masked_state_fill_value``. Marking them
unknown would make the sampler generate different values and would break the
input distribution the model was trained on.

The 129-D local-kinematic layout is a different option: it removes the two root
translation fields entirely instead of masking them. Use the mask when the token
width must stay 167; use the local-kinematic contract when the fields should not
exist at all.

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
* A preview run also needs the 144-member cohort manifest
  ``configs/distillation/x2_tennis_multimotion_v1.yaml`` and a contract
  revision that selects the frozen preview VAE.

The model dimensions come from the frozen contract through
``DenoiserSettings.from_contract``. The flow config holds only the linear process
field ``max_level`` and the shared training fields. It holds no ``noise``
section; ``scheduler: cosine`` is the learning-rate schedule only.

Run the commands with ``uv run``.

Preview-aware VAE (version 2)
-----------------------------

The flow family accepts both frozen gravity VAE layouts. Version 1 keeps
reference 68 / conditioning 99 / latent 32 / action 31. Version 2 uses reference
130 / conditioning 99 / latent 32 / action 31. The extra 62 reference values are
one q/dq sample at ``t + 0.2 s`` for the same clip. The decoder still receives
only the latent and the same 99 gravity conditioning values, so it has no direct
reference input.

The cohort declares the live layout. Every member of one manifest shares one
``lookahead_s`` in its saved teacher export. Collection and closed-loop startup
derive the live layout from that declaration and check it against the loaded
VAE schema. A 68-wide observation with a preview model fails, and
a 130-wide observation with a no-preview model fails. The code accepts no other
reference width.

The preview stays clip-local. It is sampled from the same command frame as the
current reference, clamps to the clip's last frame at the clip end, and does not
cross a subset reset into another clip. After handover, the continuation reads
no reference or preview values. It decodes from the generated latent and fresh
99-D gravity conditioning.

The token layout does not change: 41 rows by 167 channels, where 135 channels
are the physical state and 32 are the latent. The current row stays at index 8,
with 8 history rows and 32 future rows. The flow process, x0 loss, linear path
and Euler sampler are also unchanged.

Preview experiment preparation (not authorized)
-----------------------------------------------

This page does not authorize collection, training or evaluation. A later
operator authorization must supply the missing values and budgets. Use the
existing inputs and selection mechanism; do not add a second cohort registry.

.. list-table::
   :header-rows: 1

   * - Input
     - Status for a preview experiment
   * - VAE checkpoint
     - Unset. Select the frozen preview-aware artifact at authorization time.
   * - VAE checkpoint SHA-256
     - Unset. Compute it from the selected file and record it in the new
       contract revision. Do not reuse an old no-preview digest or write a
       digest from memory.
   * - Cohort manifest
     - ``configs/distillation/x2_tennis_multimotion_v1.yaml``, 144 members.
       Its saved teacher exports declare a 0.2 s preview.
   * - Manifest SHA-256
     - Compute it from the selected manifest file and record it with the
       contract's cohort manifest. The collection runtime records its own
       manifest digest.
   * - Collection selection
     - Set ``MJLAB_D1_COHORT_MANIFEST`` to the manifest. Set
       ``MJLAB_D1_ROLLOUT_TEACHERS`` to every member, or to a pilot subset of
       those 144 IDs. The existing comma-separated selection accepts
       non-contiguous IDs.
   * - Dataset, statistics, caches, run directory
     - New paths. The 0.2 s preview changes the encoder input, so old VAE
       latent statistics, normalizers, projections, token caches and
       checkpoints are not valid for this experiment.
   * - Collection and training budgets
     - Unset. Set the trial count, environment count, update budget and
       wall-clock limit only in the authorized run plan.

The existing collection mechanics stay in
``docs/source/x2_diffusion_training.rst``. Use its D1 collection, window,
statistics and token-cache steps with these preview inputs.

The new contract revision must record the 130-wide encoder layout
(``dimensions.encoder_command: 130``) for audit. The runtime still derives the
live width from the cohort and the loaded model schema. Also record the selected
checkpoint and digest, the cohort manifest and its digest, and
``decoder_mode: gravity``. Keep the existing 135-D physical state, 32-D latent,
167-channel token and 41-row window. Keep ``execution_authorized: false`` in the
preparation revision.

Assign groups to train, validation and test before windowing. Fit state and
latent statistics on the training split only. Do not fit on validation or test
windows, and do not reuse learned artifacts from an earlier run.

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

The shared D3 closed-loop CLI accepts both supported VAE reference widths
(68 or 130), with conditioning 99 / latent 32 / action 31 unchanged.
The ``evaluate.py`` preflight report and run provenance record the loaded
reference width. ``evaluate_parallel.py`` and ``view.py`` share its admission check.
Real preview runs still need a selected VAE, compatible flow artifacts, and
separate execution authorization.

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

Preview validation status
-------------------------

Software validation at the source level is separated from artifact and run
validation.

Tested on CPU with synthetic tensors only:

* Version 1 and version 2 reference layouts reach the encoder, and the
  collection wrapper returns the deterministic 32-D encoder mean and the 31-D
  action decoded from that mean.
* Scalar, batched and device-tensor collection paths accept the declared layout
  and reject a width that disagrees with the loaded model schema.
* Scalar and batched startup build 68-D or 130-D references in saved schema
  order. The clip-end sample clamps to the last frame of the same clip.
* After handover, the runtime reads no command, anchor or preview reference
  slices. The continuation still accepts a 32-D latent and 99-D conditioning.
* The assembled reference joins the unchanged 41x167 state-latent window, and
  the existing flow sampler consumes that window.
* The shared CLI accepts the preview width and reports it in preflight and
  run provenance, using stub artifact loaders.
* The existing flow closed-loop continuation tests still pass.

Not tested:

* Loading a real preview VAE checkpoint. No preview artifact was selected.
* Real preview collection, simulator rollout, training or controller deployment.
* The D3 closed-loop CLI with real preview VAE and flow artifacts.
* Tracking quality, policy stability and physical readiness.

No collection, training, simulator, GPU or remote job was executed for this
change.

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
