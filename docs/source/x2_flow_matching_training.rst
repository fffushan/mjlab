X2 x0 Flow-Matching Training
============================

This page is the runbook for the opt-in X2 x0 flow-matching family. It covers
bounded training and offline generated-token evaluation. It does not authorize a
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

Scope of the claims
-------------------

The flow family is software-validated on CPU with synthetic fixtures, plus one bounded GPU smoke on the
local VGK36 dataset that exercises 8 optimizer updates, a checkpoint reload, and a 20-call GPU sampler
on 64 real training windows of one motion. Full-dataset cached training, multi-motion evaluation, GPU
throughput, closed-loop simulation, controller integration, ONNX export and hardware behavior are not
tested. An offline generation score is not a closed-loop stability or deployment claim.
