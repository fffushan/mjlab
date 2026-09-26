X2 Tennis-End Recovery Fine-Tuning
==================================

Overview
--------

The ``Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery`` task
is an opt-in variant of the X2 no-state-estimation velocity task that
fine-tunes the standing-velocity policy to recover to standing from
tennis-ending motion states.

**Experiment design:**

.. list-table:: Reset and command mixture
   :header-rows: 1

   * - Group
     - Episode reset
     - Commands
   * - 80% recovery
     - Last-10-frame full state
     - Zero planar velocity and yaw rate
   * - 20% retention
     - Original vanilla reset
     - Original velocity-command distribution

The source velocity checkpoint (``model_19999.pt`` from
``2026-09-16_01-53-44_x2-velocity-measured-gains``) loads strictly into the
recovery runner. Physics, rewards, observations, actions, PD gains, domain
randomization, episode horizon (20 s), and source normalizers are preserved.


Components
----------

Endpoint Pool
~~~~~~~~~~~~~

``mjlab.tasks.velocity.mdp.tennis_endpoint_pool.EndpointPool`` loads tennis
motion-capture NPZ files and exposes validated, read-only NumPy arrays of the
last *N* frames of each trajectory. The pool supports a deterministic
trajectory-level train/validation split (default 20% validation, seed 42).

.. code-block:: python

   from mjlab.tasks.velocity.mdp.tennis_endpoint_pool import EndpointPool

   pool = EndpointPool.from_directory(
       "data/tennis",
       last_n_frames=10,
       split="train",
       validation_fraction=0.2,
       seed=42,
   )

Recovery Reset Event
~~~~~~~~~~~~~~~~~~~~

``TennisRecoveryResetEvent`` restores full reference state (joint pos/vel,
pelvis world pose, world-frame linear/angular velocity) for recovery
environments. XY is recentered to the environment origin; reference height is
preserved. NPZ world angular velocity is converted to local qvel via the
entity setter (``write_root_link_velocity_to_sim``).

Retention environments are untouched — the original reset events handle them.

Recovery Velocity Command
~~~~~~~~~~~~~~~~~~~~~~~~~

``TennisRecoveryVelocityCommand`` keeps recovery environments at exactly zero
twist across resets, resamples, heading updates, and curriculum changes. The
zero command never writes physical qvel. Retention environments use the
original ``UniformVelocityCommand`` distribution.

Recovery Runner
~~~~~~~~~~~~~~~

``TennisRecoveryOnPolicyRunner`` inherits the velocity runner and adds an
optional ``initial_finetune_lr`` override applied after checkpoint load. When
set, it overrides both ``alg.learning_rate`` and all optimizer param-group
``lr`` values. When ``None`` (ordinary resume), the saved optimizer LR is
preserved.

The runner resolves ``map_location=None`` to ``self.device`` so CUDA-saved
checkpoints load correctly when CUDA is hidden.


Launch Recipe
-------------

Initial fine-tune from the source checkpoint (run from the mjlab checkout):

.. code-block:: bash

   uv run train \
     Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery \
     --agent.resume True \
     --agent.initial-finetune-lr 1e-4 \
     --agent.load-run '^2026-09-16_01-53-44_x2-velocity-measured-gains$' \
     --agent.load-checkpoint '^model_19999\.pt$' \
     --agent.seed 42 \
     --env.scene.num-envs 4096 \
     --env.events.tennis-recovery-reset.params.pool-directory data/tennis \
     --env.events.tennis-recovery-reset.params.recovery-fraction 0.8 \
     --env.events.tennis-recovery-reset.params.last-n-frames 10 \
     --agent.max-iterations 2000

Key points:

- ``--agent.resume True`` enables checkpoint resume.
- ``--agent.initial-finetune-lr 1e-4`` overrides the saved optimizer LR after
  load (first fine-tune only; subsequent resumes should omit this flag).
- ``--agent.load-run`` and ``--agent.load-checkpoint`` select the source run
  and checkpoint via regex patterns searched inside the experiment root
  ``agibot_x2_velocity``.
- ``--agent.max-iterations 2000`` requests **2000 additional PPO updates**.
  Checkpoint iteration labels follow rsl-rl's resumed-loop indexing.
- ``experiment_name`` stays ``agibot_x2_velocity`` so the resolver can find
  the source run; ``run_name`` is ``x2-tennis-end-recovery`` so new
  logs/checkpoints go to a new timestamped directory.

Subsequent resume of the recovery checkpoint:

.. code-block:: bash

   uv run train \
     Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Recovery \
     --agent.resume True \
     --agent.load-run '^.*_x2-tennis-end-recovery$' \
     --env.scene.num-envs 4096 \
     --env.events.tennis-recovery-reset.params.pool-directory data/tennis \
     --agent.max-iterations 2000

Omit ``--agent.initial-finetune-lr`` to preserve the saved optimizer LR.


Evaluation
----------

The bounded evaluation CLI runs deterministic held-out episodes with identical
row/seed schedules for source and candidate:

.. code-block:: bash

   uv run python -m mjlab.tasks.velocity.scripts.tennis_recovery_eval \
     --checkpoint /path/to/model_19999.pt \
     --pool-directory data/tennis \
     --num-episodes 10 \
     --episode-length 1000 \
     --duration-s 20.0 \
     --fps 50.0 \
     --seed 42 \
     --split validation \
     --recovery-fraction 0.8 \
     --last-n-frames 10 \
     --device cpu \
     --output tennis_recovery_eval.json

The ``--last-n-frames`` option selects the endpoint window (number of trailing
reference frames) used for *both* pool row selection and the recovery reset, so
the environment and the metrics score the same states. Pass the window the
evaluated checkpoint was trained on: a policy trained on a wider window must not
be scored silently on a narrower one. To compare two checkpoints, evaluate each
on its own window *and* both on a common window (descriptors are identical for a
given schedule seed, so the same ``--seed``/``--split``/``--num-episodes`` gives
an apples-to-apples comparison). The window is recorded in the output JSON under
``pool.manifest.last_n_frames``.

The evaluation uses sequential one-env-per-episode design with
``auto_reset=False``. Recovery episodes force the recovery group and assign a
validated explicit eval row before reset. Retention episodes use the original
velocity task config with its native command manager. Terminal states are
captured before any reset. The actual ``step_dt`` is used for the metric
sample-gap bound. All recorded arrays are deep-copied snapshots. Torso tilt is
computed from the ``torso_link`` body quaternion. Real termination cause is
preserved (``orientation`` vs ``timeout``), including when a failure coincides
with the time limit. Requested ``fps`` must agree with the task's real control
rate. ``episode-length`` caps control steps; a shorter cap also shortens the
finite evaluation horizon. Outputs include a hash of the copied state/command
trace for source-versus-itself reproducibility checks.

To compare source vs candidate:

.. code-block:: bash

   uv run python -m mjlab.tasks.velocity.scripts.tennis_recovery_eval \
     --checkpoint source.pt \
     --candidate-checkpoint candidate.pt \
     --output comparison.json


Limitations
-----------

- **Zero action-history reset:** the recovery reset does not fabricate
  previous actions from a reference pose. This is an MVP limitation, not a
  warm-shadow parity claim.
- **Shallow foot overlap:** up to 5.756 mm of shallow foot overlap remains
  documented, not silently fixed.
- **Orientation termination:** orientation termination is not independently
  measured ground-contact fall.
- **Retention scoring:** walking retention is scored with command error and
  survival, not zero-velocity settling.
- Passing implementation tests or a bounded PPO smoke run does not establish
  trained-policy improvement or hardware safety. Evaluate a trained candidate
  on held-out recovery states and original velocity commands before deployment.
