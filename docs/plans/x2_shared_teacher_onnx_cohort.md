# Shared teacher ONNX with per-clip references from the NPZ

Status: proposed, 2026-11-06 (+08). No implementation yet. This document plans a
change to the distillation cohort artifact contract. It does not authorize a run.

## Problem

The distillation cohort stores the same teacher weights once per clip.

A cohort member is one teacher export plus the reference tables of one clip.
The export script states the rule: "one export serves exactly one clip. The
weights are the same shared teacher in every export; only the embedded tables
differ."

Measured on `single_150_zhanghongyu_agibot_x2_tracking.onnx`:

| Part | Tensors | Bytes |
| --- | --- | --- |
| Policy weights | 16 | 15 904 652 |
| Embedded reference tables | 6 | 441 784 |
| File total | 22 | 16 353 845 |

So 97 % of each file is a duplicate of the other 3 260 files.

For the 3 261-clip union cohort this costs, measured on `.37`:

| Quantity | Value | Evidence |
| --- | --- | --- |
| Export wall time | 0.47 s per clip, 25 min total | 40 clips in 19 s, then 3 261 clips |
| Disk | 50 GB, 3 261 files | `du -sh teachers/` |
| Resolve time | 280 ms per member, 913 s for 3 261 | resolve timed at 10, 40, 160 members |
| Resident memory | 52.9 GB | `RSS` of the trainer process during resolve |

The resolve time breaks the collector handshake: `DEFAULT_REQUEST_TIMEOUT_S =
900.0` (`worker.py:65`), and 913 s exceeds it, so the pool aborts with "worker 0
did not answer DescribeWorker within 900s of dispatch". No environment is built
and no measurement is produced.

The memory and time both come from `load_onnx_artifact` (`config.py:1351`),
which converts every initializer of the file to a NumPy array, including the
15.90 MB policy block that is identical in every member and that the training
path does not use.

## What actually needs the per-clip tables

Only two checks read the embedded reference tables:

1. `adapter._compare_body_reference_arrays` (`adapter.py:507`, called at 605 and
   840) compares the motion library against the embedded arrays.
2. `export._verify_motion_body_binding` (`export.py:355`) compares the clip NPZ,
   indexed by the audited tracked-body indices, against the embedded arrays.

Both checks exist to catch one class of defect: the library or the export using
the wrong body set or the wrong body order. The reference data itself comes from
the clip NPZ, which every member already carries. The ONNX copy is an equality
oracle, not a data source.

Metadata that the training path does need is small and identical across members:
`anchor_body_name`, `body_names`, `joint_names`, `action_scale`,
`default_joint_pos`, `joint_stiffness`, `joint_damping`, `motion_lookahead_steps`,
`observation_names`, `observation_terms_clip`, plus the observation normaliser
initializers.

## Target design

One shared teacher artifact per cohort, plus one clip NPZ per member.

```text
manifest.yaml
  teacher_onnx: <shared teacher/model.onnx>     # 15.9 MB, no reference tables
  teachers:
    - id: m0000
      motion: <clip 0>.npz                      # reference data lives here
      env_config: ..., agent_config: ..., checkpoint: ...
    - id: m0001
      motion: <clip 1>.npz
      ...
```

Rules:

* `teacher_onnx` is optional and cohort-wide. A per-entry `onnx` stays supported
  and wins when both are present.
* Resolution parses the shared artifact once per `resolve_cohort` call. Memory
  becomes the shared artifact plus one NPZ per member, not one ONNX per member.
* Reference data is read from the member NPZ everywhere. No consumer reads
  reference tables from ONNX.
* The ONNX still provides metadata and the graph interface: inputs
  `obs (1, N)`, `time_step (1, 1)`, output `actions (1, A)`.
* The body-order check keeps its purpose with a new anchor: the metadata
  `body_names` list must map the NPZ body axis onto the audited tracked indices.
  A swapped or truncated body order still fails.

## Implementation steps

1. `scripts/export_x2_teacher_reference_onnx.py`: add `--no-reference`. It writes
   one artifact with the same graph and metadata but no embedded reference
   tensors, about 15.9 MB. Keep the default behaviour unchanged.

2. `config.py`, manifest schema (lines 100-145, 214-228):
   * Add `teacher_onnx: Path | None` to `Manifest`.
   * Make `TeacherEntry.onnx` `Path | None`.
   * Validation: refuse a cohort with neither a per-entry `onnx` nor a
     `teacher_onnx`; resolve per-entry first, shared second; check the file
     exists.
   * `TeacherEntry.paths()` keeps a digest entry for the ONNX role, using the
     effective path, so the recorded provenance stays complete.

3. `config.py`, resolution:
   * `_resolve_teacher` takes the effective ONNX path and gets the artifact from
     the cache.
   * `_ResolveCache` gains an `onnx(path)` memo beside `saved_yaml` and
     `actor_state`. A shared path is parsed once.
   * `_require_onnx_interface`: keep `obs`, `time_step`, `actions`. Make the six
     reference outputs optional.
   * `_require_same_contract`: unchanged; its metadata comparisons become
     trivial when the artifact is shared, which is correct.

4. `adapter.py`, reference checks:
   * `_compare_body_reference_arrays`: compare the library tensors against the
     member NPZ subset directly, and drop the ONNX side.
   * `_compare_clip_body_references` (line 698): compare the library clip against
     the NPZ, keyed by the metadata body names.
   * Both keep the existing tolerances and the same error text shape.

5. `export.py`, member writing and body audit:
   * `_verify_motion_body_binding`: build the expected arrays from the NPZ using
     the metadata `body_names` order, then compare against the NPZ indexed by the
     audited tracked indices. The check still detects a wrong index set or order.
   * Member reference tables written into a bundle come from the NPZ.
   * The bundle descriptor records the shared ONNX path for the teacher metadata.

6. `parity.py`, `validate-teachers`: action parity always runs. Reference-output
   parity runs only when the artifact embeds the tables; otherwise it reports
   "reference outputs not embedded; checked against the clip NPZ" as an explicit
   note rather than a skip that looks like a pass.

7. `worker.py` / `distill.py`: no change. The 900 s deadline stays; resolve drops
   far below it.

## Tests

| Test | Asserts |
| --- | --- |
| `test_tracking_distillation_manifest.py` | shared `teacher_onnx` accepted; per-entry `onnx` optional; both present resolves the per-entry file; neither present is refused |
| new `test_tracking_distillation_shared_onnx.py` | `onnx.load` is called once for a 64-member shared cohort; resolve finishes under a fixed time budget; RSS proxy not asserted |
| `test_tracking_distillation_adapter.py` | a tampered NPZ body array still raises; a correct NPZ passes; a swapped body order raises |
| `test_tracking_distillation_export.py` | a member with no per-clip ONNX still writes its reference tables, and the bytes match the per-clip export to `1e-5` |
| `test_tracking_distillation_parity.py` | action parity runs without embedded references; the reference note is present |
| `test_tracking_distillation_cohort_checkpoint.py` | the 144-member per-clip manifest resolves byte-identically to today |

## Verification on the server

1. Resolve the 3 261-member manifest with the shared artifact. Expect under 10 s
   and under 100 MB resident, against 913 s and 52.9 GB today.
2. Start a collector handshake on one card. Expect `DescribeWorker` answered in
   seconds, so the 900 s deadline is irrelevant.
3. Run the two bounded probes that the current design blocked: point A (k=1) and
   point B (k=42) for trainer-card VRAM.
4. Re-run the 144-member cohort regression and confirm its contract digest is
   unchanged.
5. Delete the 50 GB of per-clip exports only after step 4 passes. Keep
   `export-report.json` as provenance.

## Risks and limits

* The equality oracle moves from "ONNX copy" to "metadata body order plus NPZ".
  A defect that corrupts both the NPZ and the metadata identically is no longer
  detected. That defect is not one the old check caught either, because both
  files were produced by the same exporter run.
* The export path for the deployed bundle changes source of truth for member
  tables. A byte-compare test on one member is the guard.
* Cohort contract digests will change for manifests that adopt `teacher_onnx`;
  they stay recorded and gate nothing, per the 2026-11-05 decision.
* This does not change the 256-member bundle cap, the training recipe, or the
  controller.

## Out of scope

* Raising `MAX_COHORT_MEMBERS`.
* Deployment, D1 collection, and the VAE run itself.
* Any change to the diffusion or controller code.

## Effort

About one day: half a day for steps 1-6 with tests, half a day for the server
verification and the 3 261-member measurement. Net change is roughly 250 lines
across five files plus tests.
