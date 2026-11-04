# Plan: simple pelvis-relative ball history

Date: 2026-11-04 (+08).
Status: implemented and software-validated, 2026-11-04 22:48 (+08).

## Goal

Store three capture-time pelvis-relative ball positions and their ages.
Keep stored positions unchanged. Do not align old samples to the current pelvis.
No proprioception history, velocity estimator, recurrent network, or extra actor features.
**Do not over-engineer. Add tests for critical functions only. No hash checks.**

## Small implementation

1. Add a pelvis-relative position helper in `src/mjlab/tasks/velocity/mdp/observations.py`.
   Use the pelvis body origin and axes, not its centre of mass:
   `p_local = R_pelvis.T @ (p_ball_world - p_pelvis_world)`.
2. Reuse the existing `ball_history_samples` stream with a small reference-frame option.
   Keep legacy plate behavior as the default. Do not copy the history implementation.
   Preserve the existing latency, noise, sample ordering, ages, padding, and reset logic.
   The simulator's existing ball-only delay ring may remain. Add no joint/IMU history.
3. Add one opt-in task:
   `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis`.
   Use actor term `ball_hist_pelvis` and a distinct experiment name.
   This avoids silently changing the meaning of old 112-wide checkpoints.
   Actor/actions/critic widths remain 112/29/121. Keep the critic unchanged.
   Use existing exported observation names; add no new metadata framework.
4. Update the juggling guide and add one Upcoming changelog entry.

## Critical validation only

- Test the coordinate transform with translated/rotated pelvis poses.
  Racket motion alone must not affect the result.
- Test that an old captured sample stays unchanged when the pelvis later moves,
  while its age advances and a new capture uses the new pelvis pose.
- Run the existing relevant history tests and a small config/export contract check.
  Confirm the new term name and unchanged legacy behavior. Avoid redundant new tests.
- Run focused lint/type checks where practical.

Use `uv run`. Use bounded tests: 60 s for one test, 120 s for a focused suite.
Investigate unexpected timeouts before retrying. Do not merely extend the timeout.
No hash checks, artifact manifests, broad validation framework, or full-repository suite.

## Boundaries

The operator already fixed ball loss. Preserve commits `7f5714759` and `e061ba5eb`.
Do not change rewards, terminations, curriculum, physics, resets, or perception magnitudes.
Preserve pre-existing changes, including the untracked audit report.

This step changes mjlab only. The current controller does not support three-sample input.
Document the simple future controller contract, but do not modify deployment code.
Exact camera-to-pelvis conversion needs a fixed transform or capture-time chain state.
Using arrival-time joint state without history is an approximation; state that limit.

One implementation subagent, then one independent read-only reviewer.
Use `ds-oc/deepseek-flash:high` for both. Fix only concrete review findings.
The operator authorized a commit after acceptance. No pushes, remote jobs, policy
playback, training launches, or hardware operations are authorized.
Report software validation separately from learning performance, which is not tested.

## Acceptance

- Implementer and independent read-only reviewer used `ds-oc/deepseek-flash:high`.
- Parent fixed the ignored-frame guard and documentation findings.
- Parent ran the five critical tests and 39 existing juggling tests: **44 passed**.
  Command: `uv run pytest tests/test_x2_tennis_pelvis_ball_history.py tests/test_x2_tennis_juggle_task.py -x -vv -o faulthandler_timeout=30`.
- The real-environment test confirms actor/actions/critic widths of 112/29/121
  and exported observation name `ball_hist_pelvis`.
- Ruff format/lint passed on the four changed Python files.
  Type checking passed for the new test file.
- The implementer also reported 91 existing task-config/ball-task tests passed.
  Its broader type checks still reported pre-existing diagnostics; they were not clean passes.
- No hashes were checked. Controller code and the operator's ball-loss fixes are unchanged.
- Learning improvement, controller integration, and hardware behavior are **not tested**.
