X2 Tennis Distillation Teacher Foundation and Latent Core (M1/M2/M3)
=======================================================================

Overview
--------

This page documents the first two milestones of the BeyondMimic-style
conditional VAE distillation effort: freezing and validating the two selected
50 Hz AgiBot X2 tennis tracking teachers, then adding a pure tensor core for
schema packing, conditional VAE inference/loss, and bounded raw replay. The
design proposal lives in ``docs/plans/beyondmimic_vae_distillation.md`` and the
milestone contracts live in ``docs/plans/beyondmimic_vae_implementation.md`` and
``docs/plans/beyondmimic_vae_m2_implementation.md``.

**M1 validates teachers, M2 provides the pure tensor core, and M3 adds a
bounded native single-teacher collector, trainer, checkpoint lifecycle, and
``distill train``/``distill evaluate`` commands.** M3 remains an implementation
and smoke-validation surface: it does not claim policy quality, production
training, hardware readiness, student export, diffusion, or multi-motion
collection. The selected live environment is the existing 50 Hz X2 tracking
task, with ``tennis_000`` as the only collection teacher.

M3 bounded lifecycle usage
---------------------------

The new commands use the trusted native Luna adapter and preserve the saved
teacher control/observation contract. Defaults are bounded (one iteration and
32 collection steps on CPU), so invoking ``train`` does not launch a production
run. ``--max-iterations`` is a total lifetime budget, including after
``--resume``; a resumed simulator is intentionally reset and bootstrap is not
repeated when replay is present. Reports record the selected teacher, motion,
50 Hz control cadence, model defaults (learning rate ``5e-4`` and KL beta
``0.01``), resource overrides, and whether evidence is implementation/smoke
or policy-quality evidence.

.. code-block:: bash

   # Teacher-only baseline; parent-owned live command (bounded to 512 steps).
   uv run distill evaluate --manifest configs/distillation/x2_tennis.yaml \\
     --repo-root . --teacher-id tennis_000 --device cuda:0 --num-envs 4 \\
     --mode teacher --steps 512 --seed 7 --sampling-mode start \\
     --report /tmp/mjlab-m3-live-validation/teacher-start.json

   # Bounded collection/training smoke; parent owns GPU execution.
   uv run distill train --manifest configs/distillation/x2_tennis.yaml \\
     --repo-root . --teacher-id tennis_000 --device cuda:0 --num-envs 4 \\
     --max-iterations 8 --checkpoint-every 4 --bootstrap-steps 128 --collection-steps 32 \\
     --updates-per-iteration 1 --teacher-probability 0 \\
     --minibatch-size 128 --accumulation-steps 15 --replay-capacity 8192 \\
     --seed 7 --rollout-latent mean \\
     --output-dir /tmp/mjlab-m3-live-validation/smoke

   # Student-only evaluation of a train-produced checkpoint; model-only, so no
   # trainer setting has to be repeated on the command line.
   uv run distill evaluate --manifest configs/distillation/x2_tennis.yaml \\
     --repo-root . --teacher-id tennis_000 --device cuda:0 --num-envs 4 \\
     --mode student --checkpoint /tmp/mjlab-m3-live-validation/smoke/checkpoint-final.pt \\
     --steps 512 --seed 7 --sampling-mode start \\
     --report /tmp/mjlab-m3-live-validation/student-start.json

For a resumed run, repeat the semantic, trainer, and resource settings (teacher
id, device, ``--num-envs``, seed, bootstrap/collection/update counts, teacher
probability, evaluation settings, learning rate, beta, accumulation, minibatch,
and replay capacity) and pass the resulting ``checkpoint-final.pt`` as
``--resume``. ``--max-iterations`` is the total lifetime budget, so extending it
(for example ``10`` for two iterations after an eight-iteration smoke) is the one
recorded setting a resume may change; every other entry of the checkpoint's
``resolved_config`` provenance is compared against the requested settings and a
difference is refused instead of silently changing the stored schedule or
semantics. Use a separate output directory for the resumed run.

Periodic saving is opt-in. ``--checkpoint-every N`` writes
``checkpoint-iter-<lifetime iteration>.pt`` into ``--output-dir`` after every
``N`` completed iterations, using the same atomic save and the same provenance,
schedule, teacher-hash, and control-contract metadata as the final checkpoint,
so any of them is a valid ``--resume`` input. The default ``0`` keeps the
single ``checkpoint-final.pt`` behavior, and ``checkpoint-final.pt`` is always
written at the end. Iterations are the total lifetime counter, so a resumed run
continues the same filename sequence rather than overwriting an earlier
period's files, and the report lists every checkpoint written by the invocation
(``checkpoints``, final one included) next to the final one (``checkpoint``).
The cadence is recorded in ``resolved_config`` but is not a resume invariant:
changing ``--checkpoint-every`` between a run and its resume is accepted, while
every other stored semantic setting is still compared and a difference is
refused. A negative cadence is rejected before any environment is constructed.

Every ``train``/``evaluate`` invocation forwards ``--seed`` into environment
construction, so the private environment configuration is seeded *before* MuJoCo
startup randomization instead of relying on a post-construction global seed.
The machine-readable report records the requested seed, the resolved seed that
the environment factory reported applying, its provenance, and (for ``train``)
the complete resolved runner/trainer/runtime configuration that is also stored
in the checkpoint: device, ``num_envs``, seed, bootstrap/collection/update
counts, evaluation settings/mode/sampling/latent, teacher probability, learning
rate, beta, accumulation, minibatch, replay capacity, schema and model settings,
teacher id and artifact hashes, and the control contract.

Boundary evidence in a ``train`` report is summarized by default. Each
iteration's ``collection.boundaries`` is an object with ``mode: summary``, the
iteration's ``ticks``, the number of boundary ``records`` and environment
``env_mentions``, a ``reasons`` mapping from boundary reason (for example
``explicit_reset``, ``terminated``, ``generation``, or a ``+``-joined
combination) to its own record and environment-mention counts, and the observed
before/after ``*_range`` segment and generation ids at the mentioned
environments. The raw alternative stores one record per boundary tick, and each
record keeps four full-batch arrays (before/after segment and generation) plus
``env_indices``; at 4096 environments that is roughly 2 MB per iteration before
indentation, so the raw form is opt-in with ``--report-boundaries full`` (any
other value is refused before environment construction) and produces a large
report. The summary is computed as each iteration finishes, so a long summary
run never holds the whole run's boundary arrays in memory; the collector's own
per-tick boundary representation is unchanged.

Student-only evaluation is model-only: ``--mode student --checkpoint PATH``
infers the saved schema and model settings from the checkpoint, so it accepts no
trainer-only flags (learning rate, beta, accumulation, minibatch, replay
capacity) and does not require the checkpoint's optimizer, replay buffer, or
collector RNG. The report echoes the inferred model identity and the
checkpoint's stored provenance. ``InferenceModel`` and
``load_inference_checkpoint`` are exported from
``mjlab.tasks.tracking.distillation``.

Evaluation metric frames are explicit. ``tracking_root_relative_pose_error``
subtracts only the root translation and keeps world axes, so a root yaw
difference between the reference and the robot contributes to it: it is
root-centered WORLD-axis pose error, not heading-invariant articulation error,
and it must not be read as joint-space tracking fidelity. Heading is reported
separately as ``tracking_heading_error``/``tracking_heading_yaw_error``, the
wrapped root-relative yaw delta, while ``tracking_global_body_pose_error`` keeps
the world-frame translation.

Live commands are resource-gated parent validation, not evidence produced by
CPU tests or this documentation change. Evaluation outcomes distinguish
reference completion, failure, timeout, teleport/timer boundaries, and step
caps; a missing live completion flag is reported conservatively rather than as
zero completion.

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

M2 pure latent core
--------------------

The M2 APIs are simulator-independent and consume one already captured,
named snapshot. The caller is responsible for aligning all fields to the same
observation time, supplying the validated cohort joint order (the default
``joint_00`` through ``joint_30`` names are explicit placeholders), and using
the actually executed normalized previous action. M2 does not query the
simulator, add noise, advance delays/history, or mutate snapshots.

``ObservationSnapshot`` and ``pack_observations`` in
``mjlab.tasks.tracking.distillation.observations`` produce a
``PackedObservationBatch``. The default ``DecoderMode.GRAVITY`` schema is
reference/conditioning/latent/action width ``68/99/32/31``. Its decoder
conditioning is gyro (3), relative joint position (31), joint velocity (31),
and previous action (31), plus projected root gravity (3); reference q/dq and
teacher/motion IDs are not decoder inputs. ``DecoderMode.ANCHOR`` and
``DecoderMode.GRAVITY_ANCHOR`` are explicit opt-in schemas with conditioning
widths 102 and 105. They are distinct schema identities and are not trained by
this milestone. Every schema records ordered fields, dimensions, joint order,
frames, and ``declared_unverified`` physical-frame status.

``ConditionalVAE`` (aliases ``DistillationVAE``/``VAE``) exposes
``encode``, ``decode``, deterministic ``mean_inference``, and explicit
``sampled_inference``/``forward(sample=..., noise=...)``. It owns separate
``StudentNormalizer`` instances for reference and conditioning features;
statistics change only through explicit ``update`` and can be ``freeze``d for
evaluation/export. Before the first update, normalization is a documented
zero-mean/unit-scale identity; it does not increment count or moments. After an
update, population variance plus the persisted epsilon is used. ``vae_loss``
returns total, summed-per-joint
reconstruction, and summed-over-latent Gaussian KL terms with default
``beta=0.01``. ``LabeledReplayBuffer`` stores detached raw packed tensors,
fixed teacher actions, and integer routing metadata in a bounded FIFO ring;
its sampling is uniform and reproducible with a supplied ``torch.Generator``.
No cached latent, optimizer, trajectory, or simulator state is stored.

The M2 assembler therefore expects the M3 caller to provide overlapping
teacher/student measurements from the same snapshot and to obtain fixed labels
from a frozen M1 ``TeacherBank``. M2 does not implement collection, DAgger,
Adam training, checkpoint/optimizer management, or policy export. M1 teacher
parity remains the source of truth for the preserved 164-dimensional teacher
input and is independent of the student's 68-dimensional reference encoder.

To bind the pure core to the already validated teacher joint order, reuse the
``cohort`` loaded above. Given an aligned ``ObservationSnapshot`` named
``snapshot``:

.. code-block:: python

   import torch
   from mjlab.tasks.tracking.distillation import (
       ConditionalVAE, make_schema, pack_observations,
   )

   schema = make_schema("gravity", joint_order=cohort.actions.joint_names)
   packed = pack_observations(snapshot, schema)
   student = ConditionalVAE(schema=schema)

   # Update statistics only at an explicit training-data boundary.
   student.reference_normalizer.update(packed.reference)
   student.conditioning_normalizer.update(packed.conditioning)
   student.reference_normalizer.freeze()
   student.conditioning_normalizer.freeze()
   student.eval()
   with torch.no_grad():
       actions = student.mean_inference(packed.reference, packed.conditioning)

This constructs an **untrained** network and demonstrates the API only; these
outputs are not a policy ready to execute. The M1 cohort already supplies the
validated joint order. A future live adapter must bind its tensors to that
order rather than relying on the pure core's placeholder names.

Physical sensor/site mapping is intentionally unresolved: root gravity, gyro,
and anchor frames are declared contract identities, not verified hardware
frames. Synthetic integration tests use placeholder feature values only and
must not be read as closed-loop or deployment evidence. A future caller must
verify the physical frame mapping and observation timing.

Scope limits of M1 and M2
--------------------------


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
* M2 schema metadata preserves declared frame names but does not verify
  physical IMU/site frames; this requires later asset and hardware evidence.
* The default joint order in the pure core is an explicit placeholder until a
  caller supplies the validated X2 cohort order. Synthetic tests do not claim
  physical frame or closed-loop fidelity.
* M2 has no collection, training runner, rollout, checkpoint/resume, or student
  ONNX export. Training and deployment are later milestones.

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
