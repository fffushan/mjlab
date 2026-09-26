X2 Tennis Distillation Teacher Foundation (M1)
==============================================

Overview
--------

This page documents the first milestone of the BeyondMimic-style conditional
VAE distillation effort: freezing and validating the two selected 50 Hz AgiBot
X2 tennis tracking teachers before any student network is trained. The design
proposal lives in ``docs/plans/beyondmimic_vae_distillation.md`` and the
milestone contract in ``docs/plans/beyondmimic_vae_implementation.md``.

**M1 implements teacher loading and validation only.** There is no train
command, no VAE, no DAgger collection, and no rollout in this milestone; the
``distill`` CLI deliberately exposes a single ``validate-teachers`` subcommand.

Selected cohort
---------------

Both teachers come from
``logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/`` and are
paired with the local tennis reference clips in the example manifest
``configs/distillation/x2_tennis.yaml``:

.. list-table:: Teacher cohort
   :header-rows: 1

   * - ID
     - Checkpoint
     - Reference
     - Frames / FPS
   * - ``tennis_000``
     - ``tennis_000/model_29999.pt``
     - ``data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz``
     - 453 / 50
   * - ``tennis_001``
     - ``tennis_001/model_29999.pt``
     - ``data/tennis/single_001_zhanghongyu_agibot_x2_tracking.npz``
     - 340 / 50

Both actors are ``164 -> [512, 256, 128] -> 31`` ELU MLPs with individual
learned observation normalizers. The two saved ``params/agent.yaml`` files are
identical and the two saved ``params/env.yaml`` files differ only in
``commands.motion.motion_file``, which still records the original remote
``/home/fushan/mjlab/...`` path.

Manifest
--------

A manifest is plain YAML, is versioned, and never names a Python callable:

.. code-block:: yaml

   version: 1
   name: x2-tennis-teachers
   robot: agibot_x2
   base_task: Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations
   teachers:
     - id: tennis_000
       checkpoint: logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/tennis_000/model_29999.pt
       motion: data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz
       env_config: logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/tennis_000/params/env.yaml
       agent_config: logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/tennis_000/params/agent.yaml
       onnx: logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/tennis_000/2026-09-23_21-24-56.onnx
       sampling_weight: 1.0

The manifest path and the relative artifact paths inside it resolve against the
same explicit repository root, which the CLI takes from ``--repo-root`` and
defaults to the current working directory, so ``--manifest
configs/distillation/x2_tennis.yaml --repo-root /path/to/mjlab`` works from any
working directory. Absolute paths are used as given, so no machine layout is
hard-coded. The local ``motion`` entry must name the same clip as the saved
``commands.motion.motion_file``; the original provenance files are never
modified.

The saved ``params/env.yaml`` and ``params/agent.yaml`` artifacts are read as
data, including their ``!!python/tuple`` and ``!!python/name`` tags. Callables
are decoded to their qualified-name string and are never imported or called.

What validation checks
----------------------

``distill validate-teachers`` resolves the manifest and then, per teacher:

* loads the actor **only**: ``actor_state_dict`` with ``mlp.*``,
  ``obs_normalizer.*`` and ``distribution.*`` entries, in ``weights_only``
  mode. Recurrent, CNN, legacy ``model_state_dict``, and per-actuator action
  offsets outside M1 scope are rejected with explicit messages;
* restores the actor's **own** observation normalizer statistics;
* confirms the actor MLP weight chain matches the saved ``hidden_dims`` and that
  the ONNX ``obs``/``time_step`` inputs and ``actions`` output match the
  checkpoint dimensions;
* rebuilds the ordered actor observation schema from the saved term
  configuration (``command`` 62, ``motion_lookahead`` 0 because
  ``lookahead_s: 0.0``, ``motion_anchor_ori_b`` 6, ``base_ang_vel`` 3,
  ``joint_pos`` 31, ``joint_vel`` 31, ``actions`` 31; 164 total) and requires it
  to agree with the exported observation names, scales, history lengths and
  clips;
* resolves the saved action scale map onto the exported joint order and compares
  it with the exported ``action_scale``, which is the joint-order evidence for
  the original export;
* compares the saved ``commands.motion.anchor_body_name`` and tracked
  ``body_names`` list (including order) with the export metadata for **every**
  teacher, so an edit applied to all saved configurations cannot pass unnoticed
  through cohort equality;
* rejects a non-``null`` runner ``clip_actions``, because M1 labels raw actor
  outputs while the runner clip would replace them before execution;
* requires the saved control period (``0.005 s`` x ``decimation 4`` = 50 Hz) to
  match the reference FPS, because the tracker advances one reference frame per
  control step;
* then runs both teachers on a reproducible finite batch on CPU, including
  inputs at the teacher's learned normalizer mean and +/- 1 and 2 standard
  deviations, and compares the deterministic native actions with the original
  ONNX ``actions`` output through CPU ONNX Runtime with
  ``atol=1e-5, rtol=1e-5``. Non-finite actions on either side fail the gate
  explicitly instead of comparing as equal, because float comparisons accept
  same-sign infinities;
* checks that the export was produced from the selected checkpoint (exported
  ``mlp.*`` weights and normalizer mean must be bit-identical, and the folded
  normalizer divisor must equal the checkpoint standard deviation plus one
  scalar epsilon);
* compares the embedded ONNX reference ``joint_pos``/``joint_vel`` arrays with
  the selected NPZ, requiring exact equality.

Cross-teacher compatibility is enforced on the actor and observation/action
control contract, on the saved environment configuration (identical apart from
``commands.motion.motion_file``), and on the actor-relevant agent configuration
(``obs_groups`` and ``actor``).

Usage
-----

.. code-block:: bash

   CUDA_VISIBLE_DEVICES='' uv run distill validate-teachers \
     --manifest configs/distillation/x2_tennis.yaml \
     --repo-root .

The JSON report goes to stdout (and to ``--report PATH`` when given), the human
summary goes to stderr, and any failed check makes the command exit nonzero.
``--samples``, ``--seed``, ``--atol`` and ``--rtol`` control the parity batch
and gate; the defaults are 64 samples, seed 0, and ``1e-5``/``1e-5``.

Representative output for the selected cohort:

.. code-block:: text

   cohort x2-tennis-teachers: 2/2 teachers passed the parity gate (obs 164, actions 31, 50.000 Hz)
     tennis_000: parity max|delta|=9.54e-07 (atol=1e-05, rtol=1e-05, n=64), association max|delta|=0, reference exact=True -> PASS
     tennis_001: parity max|delta|=1.55e-06 (atol=1e-05, rtol=1e-05, n=64), association max|delta|=0, reference exact=True -> PASS
     unverified: ONNX embedded body references are not compared with the NPZ: ...
     unverified: Physical sensor sites/frames are declared only; resolving them needs the robot asset, which M1 does not load.
   [OK] 2/2 teachers passed

Python API
----------

``mjlab.tasks.tracking.distillation`` exposes the manifest/contract resolution
and the frozen inference bank used by M2 and later:

.. code-block:: python

   from mjlab.tasks.tracking.distillation import TeacherBank, load_manifest, resolve_cohort
   from mjlab.tasks.tracking.distillation.teachers import build_frozen_teacher

   cohort = resolve_cohort(load_manifest("configs/distillation/x2_tennis.yaml", "."))
   bank = TeacherBank(
       [
           build_frozen_teacher(t.id, t.actor_state_dict, t.actor, "cpu")
           for t in cohort.teachers
       ],
       device="cpu",
   )
   actions = bank.label(bank.code("tennis_000") * torch.ones(4, dtype=torch.int64), obs)

``TeacherBank.label(teacher_ids, teacher_observations)`` groups rows by integer
teacher code (``bank.code(id)``), evaluates each needed teacher once per batch,
restores the original row order, returns an empty ``[0, J]`` tensor for an empty
batch, and rejects unknown codes, non-integer codes, shape mismatches, and
device mismatches. Teachers are always in ``eval`` mode with frozen parameters:
a surrounding ``.train()`` call cannot re-enable normalizer updates, labeling
raises if the actor is forced back into train mode, and labeling never samples
PPO exploration noise or applies action clipping.

Scope limits of M1
------------------

* Numerical parity is measured on random/statistically placed observations
  through the actor. It verifies the export and the checkpoint/normalizer
  association; it is **not** evidence of closed-loop motion quality or physical
  sensor-frame compatibility.
* Sensor sites and frames (for example ``robot/imu_ang_vel``) are recorded as
  declared names only. Resolving them physically needs the robot asset, which
  M1 does not load.
* The ONNX body reference arrays are not compared with the NPZ: the NPZ carries
  no body names, so the exported subset of tracked bodies cannot be identified.
* Parity inputs require the saved observation normalizer statistics; a
  non-normalized actor is rejected explicitly instead of being probed with an
  arbitrary input scale.
* Training, collection, evaluation rollouts, and export of student networks are
  later milestones and are not implemented here.

Tests
-----

.. code-block:: bash

   uv run pytest tests/test_tracking_distillation_manifest.py \
                 tests/test_tracking_distillation_teachers.py \
                 tests/test_tracking_distillation_parity.py \
                 tests/test_tracking_distillation_cli.py

The tests generate miniature checkpoints, saved configurations, motions, and
ONNX exports through ``tests/tracking_distillation_fixtures.py``, so they run on
CPU without private binary artifacts. The real tennis cohort is validated
explicitly with the CLI command above; a skipped generated fixture on another
machine is not evidence about the selected teachers.
