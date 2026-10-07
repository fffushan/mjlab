# Merge manifest — difficult-state coverage integration

Created 2026-10-07 20:2x (+08). Base: the previous integration lane copy.
This tree has no git history, so this file is its provenance record.

## Modules, with the lane each came from

| File | Source lane | Lines | sha1-8 |
| --- | --- | --- | --- |
| `coverage_policy.py` | integration | 733 | b856a591 |
| `coverage_metadata.py` | integration | 296 | 92c71986 |
| `coverage_perturbations.py` | integration | 567 | abce9741 |
| `coverage_evaluator.py` | integration | 359 | 1e42e2d5 |
| `branch_state.py` | runtime | 413 | e1c6de05 |
| `teacher_branches.py` | runtime | 472 | e62a63ae |
| `coverage_environment.py` | runtime | 379 | ee7e1552 |
| `coverage_collector.py` | runtime | 89 | b189402e |
| `coverage_storage.py` | replay | 2005 | aa8e5a38 |
| `x2_state_adapter.py` | runtime | 1503 | 1b97c34b |
| `../../utils/buffers/delay_buffer.py` | runtime | 434 | 38eeda4d |
| `../../utils/buffers/circular_buffer.py` | integration | 383 | c6e2a028 |
| `checkpoint.py` | integration | 1610 | ddd4e3b0 |
| `../../../scripts/distill.py` | integration | 3585 | 564418f2 |

## Test files chosen

The coverage runtime, replay and x2-adapter test files were taken from the lane
that owns the module under test, because each lane wrote its tests against its own
revision. The integration-lane versions are kept in `merge-backup/`.

## Patch applied to this tree

`checkpoint.py` constructed `CohortMemberInference` with `coverage_policy_identity`,
and the dataclass in the same file did not accept that keyword. Twelve cohort tests
failed with `TypeError`. Fix: the class gained the optional field
`coverage_policy_identity: str | None = None`. The value is `None` for every
non-coverage checkpoint. The edit was checked with `ast.parse` before the write.

## Evidence

| Check | Result |
| --- | --- |
| coverage tests, 8 files | 83 passed, exit 0 |
| base feature recovery tests, 5 files | 174 passed, exit 0, same count as before the merge |
| cohort checkpoint and cohort CLI tests | 107 passed, exit 0 |
| cohort export tests | 12 passed, exit 0 |
| distillation and buffer regression, 645 tests | 633 passed and 12 failed before the patch; **645 passed, 0 failed, exit 0** after the patch (2026-10-07 20:52 +08) |

The regression logs are container-local files under `/tmp`. The durable record of
this integration is
`/home/agiuser/projects/mjlab/logs/diagnostics/vae-coverage-integration-20261007/REPORT.md`.

## Not in this tree

No collection flag exists. `runner.py`, `worker.py` and `storage.py` are unchanged.
The library is integrated. A collection is not drivable by configuration.
