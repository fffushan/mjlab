X2 Tennis Distillation: Teacher Foundation, Latent Core, and Shared Cohorts (M1-M4)
====================================================================================

Overview
--------

This page documents the first four milestones of the BeyondMimic-style
conditional VAE distillation effort: freezing and validating the two selected
50 Hz AgiBot X2 tennis tracking teachers, then adding a pure tensor core for
schema packing, conditional VAE inference/loss, and bounded raw replay, then a
bounded native single-teacher collector, trainer, checkpoint lifecycle, and
``distill train``/``distill evaluate`` commands, and finally **one shared
student trained over both teachers's clips in one simulator**. The design
proposal lives in ``docs/plans/beyondmimic_vae_distillation.md`` and the
milestone contracts live in ``docs/plans/beyondmimic_vae_implementation.md`` and
``docs/plans/beyondmimic_vae_m2_implementation.md``; the current authority
document for M4 is ``docs/plans/beyondmimic_vae_m4_implementation.md``.

**M1 validates teachers, M2 provides the pure tensor core, M3 adds a bounded
native single-teacher collector, trainer, checkpoint lifecycle, and ``distill
train``/``distill evaluate`` commands, and M4 adds one shared student over the
multi-motion ``--teacher-ids`` cohort with per-motion balanced replay,
version-2 cohort checkpoints, and a bounded ``distill evaluate-cohort``
command.** M3 and M4 remain implementation and smoke-validation surfaces: they
do not claim policy quality, production training, hardware readiness,
diffusion, or multi-motion deployment qualification. Single-teacher defaults and
version-1 artifacts are preserved unchanged.

A standing-start extension is documented below. It is opt-in on the cohort path;
ordinary M3/M4 reference-start behavior remains the default and is not inferred from
saved teacher configuration.

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

Periodic saving defaults to every 500 completed iterations. ``--checkpoint-every
N`` writes ``checkpoint-iter-<lifetime iteration>.pt`` into ``--output-dir``
after every ``N`` completed iterations, using the same atomic save and the same
provenance, schedule, teacher-hash, and control-contract metadata as the final
checkpoint, so any of them is a valid ``--resume`` input. ``--checkpoint-every
0`` disables periodic saves, while ``checkpoint-final.pt`` is always written at
the end. Iterations are the total lifetime counter, so a resumed run continues
the same filename sequence rather than overwriting an earlier period's files,
and the report lists every checkpoint written by the invocation
(``checkpoints``, final one included) next to the final one (``checkpoint``).
The cadence is recorded in ``resolved_config`` but is not a resume invariant:
changing ``--checkpoint-every`` between a run and its resume is accepted, while
every other stored semantic setting is still compared and a difference is
refused. A negative cadence is rejected before any environment is constructed.

When ``--output-dir`` is omitted, training writes to
``logs/distillation/<task-id>/`` when ``--task-id`` is supplied. Otherwise it
uses a UTC timestamp directory under ``logs/distillation/``. This prevents the
default output from colliding with the repository's test artifacts; use an
explicit directory when intentionally resuming or grouping multiple runs.
Progress reporting defaults to every 10 completed iterations. ``--progress-every
N`` writes flushed ``[progress]`` lines to stderr; ``--progress-every 0``
disables them. The progress stream is separate from the JSON report and should
be redirected to ``train.log`` by a launcher when a persistent text log is
desired.

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
``mjlab.tasks.tracking.distillation``. A **version-2 M4 cohort checkpoint** is
also accepted here through checked member selection: ``--teacher-id`` must name
a member of the saved cohort, that member's artifact digests, clip extent, and
common action/control/observation contract are validated against the live
manifest, the environment is still one pinned single-motion simulator, and the
report keeps the full stored cohort identity (and any artifact accepted only by
content digest) next to the pinned member. Resume stays path-strict; only the
model-only member load reports relocations.

Because ``control_contract['motion']`` stores an absolute path, an
inference-only load also accepts a relocated, byte-identical motion artifact:
the move is permitted only when the supplied teacher hashes (which include the
motion digest) match the checkpoint and every other contract field agrees
exactly, and the stored provenance is returned unchanged. A missing or
different hash, or any other contract difference, is still rejected, and the
strict training-resume path (``load_checkpoint``) never uses this relaxation.

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

M4 shared multi-teacher cohort
-------------------------------

M4 trains **one shared conditional VAE** on both clips in **one vectorized
simulator**. Each environment row keeps its own reference clip and is labeled by
that clip's frozen teacher. There is no ensemble and no per-clip student, and no
separate student is created for the second motion.

.. code-block:: bash

   # One shared student over both clips; explicit plural selection.
   uv run distill train --manifest configs/distillation/x2_tennis.yaml \
     --repo-root . --teacher-ids "('tennis_000','tennis_001')" \
     --num-envs 4 --device cpu --max-iterations 4 --collection-steps 8 \
     --bootstrap-steps 8 --minibatch-size 32 --replay-capacity 256 \
     --seed 7 --output-dir logs/distillation/m4-smoke

   # The same M4 checkpoint, one member pinned as a single-motion environment.
   uv run distill evaluate --manifest configs/distillation/x2_tennis.yaml \
     --repo-root . --teacher-id tennis_001 \
     --checkpoint logs/distillation/m4-smoke/checkpoint-final.pt \
     --mode student --num-envs 4 --steps 512 --sampling-mode start --seed 7

   # Bounded all-selected-motion evaluation: one pinned environment per motion,
   # teacher and student measured with matching seeds/phase/resources.
   uv run distill evaluate-cohort --manifest configs/distillation/x2_tennis.yaml \
     --repo-root . --teacher-ids "('tennis_000','tennis_001')" \
     --checkpoint logs/distillation/m4-smoke/checkpoint-final.pt \
     --mode both --num-envs 4 --steps 512 --sampling-mode start --seed 7 \
     --report /tmp/mjlab-m4-live-validation/cohort-start.json
   # Playback pins one member of the saved cohort (existing viewers).
   uv run distill play --manifest configs/distillation/x2_tennis.yaml \
     --repo-root . --teacher-id tennis_001 \
     --checkpoint logs/distillation/m4-smoke/checkpoint-final.pt --seed 7

Selection syntax
^^^^^^^^^^^^^^^^

``--teacher-id`` (singular) keeps the M3 single-teacher path and
``--teacher-ids`` (plural) selects the cohort path. The two are mutually
exclusive and supplying both is refused **before any environment is
constructed**. With neither flag the single-teacher path keeps the historical
``tennis_000`` default, so every existing invocation behaves exactly as before.

This repository's shared Tyro configuration
(``mjlab.TYRO_FLAGS``) sets ``UsePythonSyntaxForLiteralCollections``, so a
collection flag takes a **Python literal** rather than a space-separated list:
``--teacher-ids "('tennis_000','tennis_001')"``. A single element keeps the
trailing comma (``--teacher-ids "('tennis_000',)"``), because without it the
literal is a string and not a tuple. The same spelling applies to
``evaluate-cohort --teacher-ids``; omitting it there evaluates every manifest
teacher in manifest order.

What M4 constructs
^^^^^^^^^^^^^^^^^^

* One mixed-slot environment: the registered tracking task is copied privately
  and only its motion command is replaced, so observations, rewards,
  terminations, and control timing keep their meanings.
* Stratified fixed environment slots: rows are allocated by the manifest's
  positive sampling weights with deterministic largest-remainder allocation and
  a seeded row permutation. Every selected motion gets at least one row, and an
  impossible budget/weight combination is refused. Each slot keeps its clip
  through ordinary per-row reset/timer/wrap, which deliberately avoids the
  clip-duration bias of equal-probability motion sampling at episode boundaries
  and makes equal fresh-data normalization coverage possible. This is an
  engineering choice, not a paper claim and not a promise of seamless
  transitions between clips.
* Uniform phase sampling for training, ``start``/``uniform`` for pinned
  evaluation, both recorded as explicit private overrides of the teacher's
  adaptive sampling. Adaptive/weighted multi-motion sampling is refused rather
  than sharing failure bins across clips.
* One frozen teacher bank over the whole manifest, so a row's teacher code
  always selects the teacher that trained on that row's clip. Motion and teacher
  ids are routing metadata only; they never enter the decoder conditioning, and
  the gravity schema stays reference 68 / conditioning 99 / latent 32 /
  actions 31.
* One per-motion balanced replay: the capacity is split into integer quotas by
  the configured weights (summing exactly to the capacity, at least one slot per
  motion), each motion has its own FIFO partition so a shorter or easier clip
  cannot evict another, the whole incoming batch is validated before any
  partition is mutated, draws use the configured motion proportions with
  unbiased residual allocation, and a missing motion is reported as not ready
  instead of being silently omitted. Fresh raw rows update the normalizers once,
  balanced replay draws never do. The selection and the cheap budgets (unknown
  or repeated teacher ids, and a replay capacity that cannot give each selected
  motion a slot) are validated before any environment is constructed, and a
  failure after the environment exists closes it exactly once.

Reports and cohort identity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A cohort ``train`` report adds ``cohort`` (ordered teacher ids, cohort and
mapping digests, slot counts/weights and row count, replay partitions/quotas,
clip frame counts, phase policy) next to the per-motion collection attribution
(``collection.motion_stats``: per-motion samples, teacher/student steps,
boundaries, disagreement, and reference-frame coverage). Each iteration and the
final report also include ``replay`` telemetry: capacity, retained/inserted/drawn
counts, occupancy, readiness, and per-motion quotas/counters. Replay ``coverage``
means retained rows divided by that motion's capacity quota, not clip-frame
coverage. ``evaluate-cohort`` requires a positive step budget and
normalizes ``--teacher-ids`` into **manifest order** (a repeated id is refused),
records the request next to the normalized selection, and pins each motion into
its own single-motion environment.  The teacher baseline and the student never
share a live environment: each mode builds a **fresh identically seeded**
adaptor and closes it before the next one is built, because a reset is not proof
that startup randomization, event timers, and adapter state were restored; one
student model is loaded once and reused.

Every reported level is attributed to the motion's **cohort identity**, not to
the pinned environment's local clip 0 (a ``tennis_001`` pin reports local 0 while
its cohort motion id is 1).  For a saved cohort the ``motion_id`` and
``teacher_code`` come from the stored cohort record, whose membership and
contracts are re-checked against the live manifest; a teacher-only run states
the explicit manifest position instead.  Each motion block records which source
it used (``motion_id_source``), and the per-motion entry, its censoring entry,
the aggregate per-motion maps, and the aggregate outcome counts all use the same
identity.  Each motion block therefore holds separate ``teacher`` and
``student`` reports with their own segment counts, outcome counts,
completion/failure denominators, censored-segment counts, and metrics, plus two
aggregate views: ``macro`` (equal motion weight) and ``clip_duration_weighted``
(weights proportional to the clip duration in seconds, ``frames/fps``).  Both
keep the per-motion values, the aggregation weights, the contributing-motion
counts, and any metric that only some motions reported, because a good aggregate
must never hide one failing motion.  A high ``completion_rate`` is defined over
completed-or-failed segments only and must not be read as every initiated
segment completing; execution errors are reported separately in
``evaluation_errors`` (per motion and mode), a missing report makes ``complete``
false, and the ``quality`` block explicitly states that the aggregate is not a
quality pass.

A cohort run saves **version-2** checkpoints that record the ordered member
identities with every artifact path and digest, per-clip extent/source
digest/audited body mapping, the ordered mapping digest, the common
action/control/observation contract, the slot and phase policy with realized
counts, the replay partition policy, resource settings, and all RNG state.
Strict resume reproduces the whole record and refuses reordered, missing, or
changed members; changed artifacts, clip extents, body mapping, common contract,
slot/phase policy, replay policy, or resource settings; a corrupted later replay
partition; a corrupted generator state; and a FIFO replay buffer. Resume
restarts the simulator, creates a segment namespace above every retained
partition, and never claims bitwise simulator continuation. The only runtime
exceptions are the total ``--max-iterations`` budget and the
``--checkpoint-every``/reporting cadence. Version-1 and version-2 artifacts are
never converted into each other, and each loader names the loader that fits.
Periodic checkpointing (default every 500 completed iterations) and flushed
``[progress]`` stderr lines (default every 10 iterations) behave exactly as in
M3, and stdout remains the machine-readable JSON report.

Standing-start distillation (opt-in)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real deployment engages the robot from its standing pose. Reference-frame-only
initialization never presents the student with the transition from that pose into
a selected tennis motion, so it cannot train or validate that transition. The
standing-start extension adds that collection and student-evaluation surface
without changing the VAE architecture, observation features, objective,
normalizers, teacher/student mixing schedule, or 50 Hz control timing.

Training reset policy
~~~~~~~~~~~~~~~~~~~~~

The extension is selected explicitly on the shared cohort path. The following is
the exact singleton-cohort spelling; the comma is required because the shared
Tyro configuration parses ``--teacher-ids`` as a Python literal tuple:

.. code-block:: bash

   uv run distill train --manifest configs/distillation/x2_tennis.yaml \\
     --repo-root . --teacher-ids "('tennis_000',)" \\
     --reset-policy standing-mixture \\
     --standing-start-fraction 0.25 \\
     --standing-start-window-frames 25 \\
     --standing-start-frame-zero-fraction 0.5 \\
     --num-envs 4 --device cpu --max-iterations 4 \\
     --collection-steps 8 --bootstrap-steps 8 --minibatch-size 32 \\
     --replay-capacity 256 --seed 7 --output-dir logs/distillation/standing-smoke

The training options and their CLI defaults are:

* ``--reset-policy reference`` (the default), which preserves the reference-only
  reset path and its RNG consumption exactly;
* ``--standing-start-fraction 0.25``;
* ``--standing-start-window-frames 25``; and
* ``--standing-start-frame-zero-fraction 0.5``.

The latter three are **proposed starting hyperparameters, not tuned values**.
Their defaults have no standing effect while ``--reset-policy reference`` is in
use. Enabling standing resets on the singular M3 path is refused; use the
singleton tuple above (and omit ``--teacher-id``). With no new options, the
legacy reset sampling, checkpoint format, and evaluation behavior remain
unchanged.

For each row undergoing an eligible full environment reset, the enabled policy
first chooses standing with probability ``standing_start_fraction``. A reference
row samples uniformly over its whole assigned clip and uses the reference pose.
A standing row uses the robot's default standing joints and height, reference
root ``x/y`` at the selected frame, upright yaw from that frame's reference
anchor quaternion, and zero velocities before the existing reset perturbations
and joint clipping are applied. It does not prepend standing frames, interpolate
poses, hold phase, or add assistance. Initial resets, termination/timeout resets,
resumed simulator resets, and explicit full row resets are eligible; timer
resamples, natural reference wraps, and ``reset_to_frame`` remain reference-state
teleports and do not manufacture standing starts.

For a clip with ``F_i`` frames, the effective standing window is
``W_i = min(standing_start_window_frames, F_i)``. The non-frame-zero standing
branch samples uniformly from local frames ``0`` through ``W_i - 1``; frame zero
is therefore included in that branch. If ``a`` is
``standing_start_frame_zero_fraction``, the realized frame-zero probability
conditional on a standing reset is
``a + (1 - a) / W_i``, not ``a`` alone. With the defaults this is
``0.5 + 0.5 / W_i`` (and it is 1.0 for a one-frame clip). Reports retain the
realized per-clip reset/frame-zero counts. The configured standing fraction is
a fraction of eligible full resets, not a fraction of replay rows or gradient
updates.

Teacher qualification remains a manual, standalone user activity. This pipeline
uses the selected frozen teachers to label the actual student-visited states,
but it does not run a teacher transition rollout, impose a success threshold,
create a teacher-quality certificate, retrain a teacher, or gate training on
teacher competence.

Standing student evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standing profiles are student-only checks. They do not qualify teachers, and
``--reset-profile`` is rejected for ``--mode teacher`` and ``--mode both``.
The profile option is available on both ``evaluate`` and ``evaluate-cohort``;
when it is present, omit ``--sampling-mode``. The exact command forms are:

.. code-block:: bash

   uv run distill evaluate --manifest configs/distillation/x2_tennis.yaml \\
     --repo-root . --teacher-id tennis_000 \\
     --checkpoint <version-3-standing-checkpoint> --mode student \\
     --reset-profile standing-start --steps 512 --seed 7

   uv run distill evaluate-cohort \\
     --manifest configs/distillation/x2_tennis.yaml --repo-root . \\
     --teacher-ids "('tennis_000','tennis_001')" \\
     --checkpoint <version-3-standing-checkpoint> --mode student \\
     --reset-profile standing-window --steps 512 --seed 7 \\
     --report /tmp/standing-window.json

The four checked student profiles are:

.. list-table:: Student reset profiles
   :header-rows: 1

   * - Profile
     - Physical initialization
     - Reference frame selection
     - CLI spelling
   * - Reference-start
     - Reference pose
     - Frame 0
     - no profile; ``--sampling-mode start``
   * - Reference-uniform
     - Reference pose
     - Uniform over the whole clip
     - no profile; ``--sampling-mode uniform``
   * - Standing-start
     - Standing pose
     - Frame 0
     - ``--reset-profile standing-start``
   * - Standing-window
     - Standing pose
     - Uniform over frames ``0 .. min(25, F_i)-1``
     - ``--reset-profile standing-window``

The standing window is a reset-selection window, not a convergence guarantee,
grace period, or step cap. Each trial starts with a full environment reset and is
attributed to the segment initialized there. A later reference wrap or timer
teleport is not another standing trial. Reports retain initial frame/pose
provenance, reset perturbations, raw segment and outcome counts, and tracking and
action metrics. ``reference_complete`` and ``failure`` are the known-outcome
denominator for completion/failure rates. Timeouts, timer resamples, reference
teleports, explicit resets, and step caps are censored outcomes and remain in
separate counts; they are not silently treated as completion or failure. Thus a
completion rate over completed-or-failed segments is not a rate over every
initiated trial. Normalizers are frozen and evaluation creates no replay or
optimizer state.

Standing checkpoint compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference-only M4 checkpoints stay version 2; enabled standing runs write
version-3 cohort checkpoints containing the reset policy, effective per-clip
windows, standing pose and perturbation provenance, reset RNG, and provenance-v1
replay metadata. The compatibility matrix is:

.. list-table:: Standing-start checkpoint compatibility
   :header-rows: 1

   * - Input
     - Required behavior
   * - Version-1 legacy training, inference, or export
     - Unchanged.
   * - Version-2 reference-only M4 resume or inference
     - Unchanged; no standing semantics are inferred.
   * - Version-2 resume with standing enabled
     - Reject strict resume with a policy/version error; start a new
       version-3 run. There is no v2-to-v3 resume migration.
   * - Version-3 same-policy resume
     - Restore the durable reset/replay/model state and reset RNG, then
       explicitly restart the simulator with a new segment namespace.
   * - Version-3 changed policy, window, fractions, pose, or perturbations
     - Reject before partial restore.
   * - Version-3 checked member inference
     - Preserve trained reset provenance and allow an explicit evaluation-only profile.
   * - Version-2 or version-3 cohort export
     - Explicitly unsupported; version-1 single-teacher export remains unchanged.

A standing checkpoint is not evidence of transition competence. This is
simulation-only implementation and evaluation work: it makes no policy-quality,
convergence, sim2sim, hardware-readiness, or deployment claim.

Limitations
^^^^^^^^^^^

* ``distill export`` refuses a version-2 M4 cohort checkpoint explicitly. The
exported bundle records one teacher's artifacts, motion, and physical sensor
anchor audit, and the asset-audit producer validates one audited single-motion
environment, so a cohort member cannot be certified through that seam. This is a
reported limitation, not a disabled check; version-1 export is unchanged. See
``docs/source/x2_vae_export_contract.rst``.
* A cohort checkpoint cannot be resumed from a relocated checkout: the stored
identity keeps absolute manifest and artifact paths. Model-only inference of one
member does accept a byte-identical relocated artifact by content digest and
reports the relocation.
* M4 does not add random clip reassignment, curriculum, curriculum-scale
  transitions, or multi-motion deployment qualification.

Interactive playback (``distill play``)
---------------------------------------

``distill play`` reuses the same audited environment and the existing Viser or
native viewers to inspect a checkpointed student without any training
machinery: it constructs no trainer, optimizer, replay buffer, collector, or PPO
runner, and the student is reconstructed model-only with
``load_inference_checkpoint`` (a version-2 cohort artifact through checked
member selection) and evaluated with deterministic mean-latent inference (no
latent sampling, no normalizer update). One environment and
``--sampling-mode start`` are the defaults. For a version-2 cohort checkpoint,
``--teacher-id`` must name a member of the saved cohort and that member's clip
is what the pinned environment plays; the Viser Checkpoints tab dispatches each
discovered artifact to the loader its version requires, so a directory holding
both a version-1 and a version-2 checkpoint can be inspected in one session.

The checkpoint is loaded *before* the simulator is built, so a missing or
incompatible artifact never constructs an environment, and the saved schema is
what the live packing is built from. That means a checkpoint trained with the
``anchor`` or ``gravity_anchor`` decoder packs correctly instead of being
rejected for disagreeing with a default gravity schema; the adapter still
validates the live joint order, teacher sensors/action, and control cadence
against the saved contract. After the sampling override and before the first
viewer action, play performs one audited seeded reset (the environment
constructor does not reset, and unlike the PPO path there is no vector-env
wrapper doing it).

.. code-block:: bash

   uv run distill play --manifest configs/distillation/x2_tennis.yaml \\
     --repo-root . --teacher-id tennis_000 --device cuda:0 --num-envs 1 \\
     --checkpoint /tmp/mjlab-m3-live-validation/smoke/checkpoint-final.pt \\
     --sampling-mode start --viewer viser --seed 7

``--viewer viser`` (the default) serves the browser viewer; ``--viewer native``
uses MuJoCo's passive viewer, and ``--viewer auto`` picks native when
``DISPLAY``/``WAYLAND_DISPLAY`` is set and Viser otherwise. The Viser viewer's
Checkpoints tab discovers this run's ``checkpoint-iter-<lifetime>.pt`` and
``checkpoint-final.pt`` artifacts directly (the PPO ``model_<N>.pt`` sort key
does not apply) and every hot swap goes through the same model-only validation
as the initial load, including the live schema, so an incompatible artifact is
refused instead of being installed or mis-packed. A rejected swap leaves the
running policy and the viewer intact.

Playback environment semantics
------------------------------

The play surface deliberately does **not** load the registered task's
``play=True`` configuration. That PPO play preset disables actor corruption,
removes the push event, clears the reset pose/velocity ranges, sets ``start``
sampling, and makes episodes effectively infinite. Those overrides contradict
the saved teacher contract (for example the live actor corruption setting is
asserted against the saved one), so playback keeps the training-time contract —
actor corruption, reset perturbations, the push event, and finite episodes —
and the only sampling control is the explicit ``--sampling-mode
{start,uniform}`` override on the private command copy, exactly as bounded
evaluation applies it. Playback therefore reproduces the noisy, reset-prone
observations the student was normalized against, and the reference ghost shown
in the viewer is the NPZ reference, not a teacher-policy rollout.

The viewer's motion scrubber (``Start Here`` and the paused frame slider) edits
the live command/reference state after ``env.reset`` has cached observations.
Viser runs GUI callbacks on a worker thread, so the frame write and the
observation refresh are queued to the viewer's main loop rather than mutating
live tensors from the callback.  ``Start Here`` captures the environment, the
frame, and the ``all envs`` selection at click time, so a later environment
switch cannot retarget the queued reset; ``all envs`` resets every environment
and ignores the captured one.

On the main loop the cache is refreshed through the public scoped
``ObservationManager.refresh(env_ids, baseline=...)`` hook: the edited
environments are re-read at the current instant and their history/delay buffers
are backfilled, while every other environment keeps the observation row it had
*before* the GUI action.  A partial ``Start Here`` resets only the captured
environment, so the viewer snapshots the pre-reset cache with
``ObservationManager.cached_observations()`` first and passes it as the
``baseline``; otherwise ``env.reset``'s whole-batch recompute (and its
resampled raw noise) would leak into the untouched environments.  The baseline
is an owned clone, so an intervening reset or buffer reuse cannot change it;
``all envs`` refreshes every environment normally.  The first action after
scrubbing therefore sees the scrubbed state, no extra history tick or lag draw
is added anywhere, and ``env.obs_buf`` stays in agreement with the manager
cache.

The global noise RNG may still advance when the edited rows are recomputed on an
explicit GUI action; the untouched observable rows and their history/delay
timelines do not change.  This recompute runs only on an explicit user action,
never on a normal render.

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
