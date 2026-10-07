# Acceptance record — coverage integration

This record replaces the `still open` status of `PARENT-ACCEPTANCE.md`. That file
stays in place as history. Date: 2026-10-07, 21:10 (+08).

## Accepted

| Item | Evidence |
| --- | --- |
| Module integration | nine modules plus `x2_state_adapter.py` in the newest revision of every lane |
| Replay defects | four independent specification tests in `tests/test_tracking_distillation_coverage_replay.py` |
| Coverage suite | 83 passed, exit 0 |
| Base feature suite | 174 passed, exit 0, same count as before the merge |
| Cohort checkpoint and CLI | 107 passed, exit 0 |
| Cohort export | 12 passed, exit 0 |
| Distillation and buffer regression | 645 passed, exit 0 |
| Execution path slice 1 | configuration surface, `CohortSetup.coverage_policy`, fail-closed gate; 74 passed with the cohort suites |
| Merge provenance | `MERGE-MANIFEST.md` |

## The four replay defects, verified closed

1. A non-finite recent conditioning tensor is rejected. Verified by
   `test_replay_defect_nan_conditioning_is_rejected` and
   `test_replay_defect_infinite_conditioning_is_rejected`.
2. A valid provenance-v1 recent checkpoint loads. Verified by
   `test_replay_defect_valid_provenance_state_round_trips`.
3. The nested ring restore no longer leaves device, dtype or provenance as `None`.
   Verified by the same test.
4. The nested ring flags and layout survive a round trip. Verified by the same test.
5. A rejected load leaves the buffer unchanged. Verified NaN-aware, because the
   free slots of the ring hold NaN as an empty-slot sentinel.

## Accepted since the first draft of this record

| Slice | Content | Evidence |
| --- | --- | --- |
| 2a | `CollectReply.branch_batch` for variable-size branch rows; `namespace_coverage_rows` and `namespace_coverage_batch` | 12 tests, then 669 passed in the shared suite |
| 2b | `_require_mergeable` validates branch rows through `validate_collect_reply`; `merge_branch_rows` defines the branch order and namespaces each reply | 9 tests, then 678 passed in the shared suite |
| 2c-1 | `build_coverage_collector` builds nothing when the policy is off, so the disabled path stays unchanged | 4 tests, then 682 passed in the shared suite |
| 3 (barrier) | The P0 contract fixes `barrier_seed_queue_policy: drain_and_clear_with_counts` (user decision 2026-10-07). `DistillationRunner.drain_branch_work_unit` drains, clears and counts; a run without a component writes no new counters, and an enabled run without the backend refuses to checkpoint | 7 tests, plus a guard that the contract keeps the decision; the shared suite then reported 689 passed, 0 failed |
| Replay defects | four independent specification tests verify all recorded defects closed | 30 passed in the replay file |
| Records | `MERGE-MANIFEST.md`, the changelog entry, this record | review and integration reports, 0 STE violations |

## Not accepted

1. **The production environment backend.** The capture path calls three handlers on
   the environment, and **no production adapter implements any of them**:
   `coverage_cached_policy_view(index)`, `coverage_executed_action(index)` and
   `coverage_reconstruct_next_boundary(action)`. They exist only in test fakes
   (`tests/test_tracking_distillation_coverage_runtime.py` lines 54 and 57, and the
   fake bridge in `tests/test_tracking_distillation_coverage_x2_adapter.py`). The
   targets are `MultiMotionDistillationAdapter` and `DistillationEnvironmentAdapter`.
   An enabled run fails closed with `CoverageRuntimeError`, so nothing mis-collects.
   The test rig for this work exists: `FakeEnv`, `FakeSim`, `FakeObservationManager`
   and `FakeScene` in the adapter test file. The P0 contract fixes the semantics:
   cached policy instant for the view, no noisy or delayed `q/dq` for severity
   severity decisions, and one control-decimation boundary per branch step.
2. **The collection barrier is decided and hooked, not yet exercised.** The policy is
   fixed in the contract and the runner drains, clears and counts. The drain itself
   cannot run until the production environment backend exists, because the branch
   unit needs it. When a component is enabled without that backend, the runner
   refuses to checkpoint rather than drop work.
3. **The strict-resume matrix is partly proven.** Three rows are tested and pass:
   the coverage-off formats keep their behaviour, a coverage-off checkpoint cannot
   enter the enabled v4 loader, and an enabled save fails closed for a legacy
   replay. The enabled replay seam `coverage_replay_policy()` is now implemented on
   `ProtectedCoverageReplayBuffer`, so that save path is no longer blocked by
   code that does not exist. The three v4 loader rejections stay **unverified**,
   because a real v4 file needs a trainer and a cohort identity built over the
   enabled replay. That is one fixture function of work. The continuation row needs
   an enabled run and stays unclaimed.
4. P2 and P3 physics evidence: restore parity, reached-terminal alignment,
   teacher-failure retention, hard budgets.
4b. **The three environment handlers are implemented.** `MultiMotionDistillationAdapter`
    now exposes `coverage_cached_policy_view`, `coverage_executed_action` and
    `coverage_reconstruct_next_boundary`. The view reads the cached actor tensor
    through `observation_manager.cached_observations()` with no recompute and no new
    noise draw, and returns an owned row. The action handler returns the action
    `step` last applied, cloned on read so a disabled run does no extra work. The
    boundary handler is one `env.step` and does not consume the boundary events,
    which belong to the main trajectory. An enabled run fails closed with
    `CoverageRuntimeError` when the seam is absent. Two tests cover this and skip
    the full cohort and bank construction. The shared suite then reported 694
    passed, 0 failed, which is 692 plus those two tests.
    Still open before any physics evidence: the collector call sites in
    `worker.py` are not wired to these handlers, and the branch backend factory does
    not exist, so an enabled run still refuses to start and the barrier still
    refuses to checkpoint.

5. No simulator run, no training run, no hardware run. The contract records
   `execution_authorized: false`.

## Standing rule

Live collection stays blocked until P2 next-transition parity closes. Nothing in
this record authorizes a run.
