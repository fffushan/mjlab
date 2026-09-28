X2 VAE export artifact contract
================================

The offline ``distill export`` command writes a version-2 ``vae-<teacher-id>``
bundle.  It never constructs a simulator or trainer.  A physical asset audit is
required through ``--asset-audit``; a tensor schema's ``declared_unverified``
frame names are not accepted as sensor or anchor provenance.

Bundle layout
-------------

::

  vae-tennis_000.yaml         # v2 descriptor in the policy store
  vae-tennis_000/
    bundle.json                # copy of the descriptor for the contained bundle
    contract.json     # semantic contract and provenance
    motion.json                 # frame-major float32 q/dq/anchor table
    parity.json                # deterministic ORT/PyTorch inputs and expected outputs
    encoder.onnx      # reference [1,68] -> latent [1,32]
    decoder.onnx      # latent [1,32], conditioning [1,99] -> actions [1,31]

``vae-tennis_000.yaml`` has ``format: mjlab-vae-tracking``, ``version: 2``,
``family: vae_tracking``, matching ``model_id`` and ``contract_id``, and a
``files`` object.  Every path is a contained relative path and every referenced
byte stream is SHA-256 hashed.  Component graphs carry the same model and
contract IDs as the descriptor and contract; a consumer must reject mixed pairs.
``bundle.json`` is a contained copy of the descriptor and also includes
``parity.json`` in its hash map.  ``parity.json`` records the deterministic
float32 reference/conditioning inputs, expected PyTorch mean latent/action,
model ID, contract ID, and measured max errors.  C++ consumers can retain it as
an offline numeric fixture; it is not a runtime dependency.


Tensor interfaces
-----------------

The encoder input is the exact ordered reference vector ``reference_q`` (31),
``reference_dq`` (31), and ``anchor_orientation_error`` (6).  It emits the
posterior mean ``latent`` (32), not a sampled latent.  The decoder accepts that
latent and the gravity conditioning vector ordered as ``projected_gravity`` (3),
``gyro`` (3), ``relative_joint_q`` (31), ``joint_dq`` (31), and
``previous_action`` (31), then emits the raw normalized joint-position action
(31).  The learned reference and conditioning normalizers are inside their
respective graphs exactly once.  Their float32 epsilon, count, mean, M2,
population-variance convention, cold-start identity behavior, and frozen state
are recorded in ``contract.json``.

Contract and provenance
-----------------------

``contract.json`` records the saved checkpoint schema/settings, dimensions,
50-Hz control/reference cadence, final-reference hold behavior, exact joint
order/scales/offset, and full-precision audited default positions, Kp, and Kd.
The supported audit producer is ``make_export_audit(env, cohort, teacher_id,\
asset_path=...)``.  It runs the existing compiled-environment
``validate_live_contract``/``audit_live_asset`` checks, records a hash of the
verified asset release, full robot body names and tracked-to-full body indices,
compiled gyro site/body/local quaternion, and reads defaults/Kp/Kd from the
compiled robot actuators.  Its training anchor source is truthfully recorded as
``compiled_body_orientation``; it does not claim to audit the C++ waist FK.  The
contract separately records the v1 deployment requirement
for the downstream adapter to resolve and validate.  The resulting JSON is the
accepted ``--asset-audit`` input; handwritten objects with only a verification
flag are refused.

The source checkpoint, teacher artifacts, and motion hashes are included.  The
``sensor_anchor`` provenance must carry ``verification: audited`` and explicit
root frame, gravity source/frame, gyro source/frame, and anchor source/frame.
The anchor source describes the compiled training/reference body orientation;
the deployment measurement requirement is recorded separately and is not proof
that the C++ FK path has been audited here.
  The exporter refuses absent or
``declared_unverified`` evidence and refuses ONNX's rounded text metadata as a
replacement for full-precision control values.

``motion.json`` stores all frames in source order and only ``q``, ``dq``, and the
selected anchor ``anchor_quat_wxyz``.  Arrays must be source float32, finite,
shape-matched, and unit quaternions in MuJoCo ``wxyz`` order.  The exporter
selects the anchor using the audited teacher body ordering; it does not infer a
physical body index from a raw tensor schema.

Real CPU audit recipe
----------------------

A bounded validation-only smoke uses the compiled environment, not an arbitrary
XML/hash supplied by a caller.  The parent-monitored recipe is
``/home/agiuser/.pi/agent/vae-x2-20260927-01a0e10d/smoke-vQtTXc/audit_export_smoke.py``.
It resolves the original ``configs/distillation/x2_tennis.yaml`` cohort,
constructs one CPU environment with ``build_distillation_environment``, checks
the X2 ``get_spec`` factory, saves the actual compiled model to
``compiled-environment.mjb``, hashes both the verified X2 XML source
``src/mjlab/asset_zoo/robots/agibot_x2/xmls/x2_ultra.xml`` and the compiled MJB,
then calls ``make_export_audit`` before exporting the offline candidate
``logs/distillation/m3-distill-8192-10k-20260927-005210/checkpoint-iter-005000.pt``.
The successful example produced
``bundle/vae-tennis_000.yaml`` and
``asset-audit.json`` under the smoke directory, with root
``robot/pelvis``, gyro ``robot/imu_ang_vel`` on site ``robot/imu_0``, anchor
``torso_link``, and full-robot anchor index 15.  This is export validation only;
it is not hardware evidence or a deployment authorization.


Version-3 cohort bundles
------------------------

``distill export --asset-audits <index.json>`` exports a saved version-2/3 M4
cohort checkpoint (one shared student trained over several teachers) through
``export_cohort_bundle(checkpoint, manifest, output_dir, repo_root=...,\
asset_audits=...)``.  The index is a JSON object mapping every manifest
member's teacher id to that member's audit file; each audit is produced by
``make_export_audit`` against a pinned single-teacher environment of that
member, exactly as the v2 seam requires.  A missing or extra member is
refused, and audits that describe different compiled robots (asset identity,
sensor/anchor provenance, or audited gains) are refused: the shared action and
sensor blocks must be true for every member.

The layout keeps one shared student and per-member triples::

  vae-x2-tennis-teachers-mixed.yaml    # version-3 descriptor
  vae-x2-tennis-teachers-mixed/
    bundle.json                         # contained copy of the descriptor
    shared/encoder.onnx                 # reference [1,68] -> latent [1,32]
    shared/decoder.onnx                 # latent [1,32], conditioning [1,99] -> actions [1,31]
    <teacher-id>/contract.json          # byte-shape v2 member contract
    <teacher-id>/motion.json            # that teacher's frame-major table
    <teacher-id>/parity.json            # parity bound to the member contract id

The descriptor's top-level ``model_id``/``contract_id`` are the cohort
identity stamped into both graphs' metadata; each member entry carries its own
``contract_id`` (matching that member's ``contract.json``), and the member
order is the manifest order, which is the order the deployed controller plays
the members in.  Member contracts remain version-2 shaped so the C++ deployment
parser is unchanged; they share the student ``model_id`` and differ in teacher
provenance, motion digest, and contract id.  Parity fixtures are per member
with the member contract id; the numbers are shared because the graphs are.

``export_bundle(checkpoint, manifest, teacher_id, output_dir, repo_root=...,\
asset_audit=...)`` returns an ``ExportResult`` with all paths and IDs.
``validate_export_parity(result, inference_model.model, reference, conditioning)``
loads the emitted bytes with CPU ONNX Runtime and compares latent and action
outputs to PyTorch at ``atol=rtol=1e-5``.  Focused tests use self-contained tiny
checkpoints and motions and cover parity, all-frame ordering/float32
round-tripping, malformed schema/source refusal, and normalizer immutability.

``vae-tennis_000.yaml`` and temporary test directories containing
``vae-tennis_000/{bundle.json,contract.json,motion.json,encoder.onnx,decoder.onnx}``;
no generated fixture is checked into the repository.  The candidate 5000-step
checkpoint remains an offline export candidate only and is not selected or
hardware-qualified by this seam.
