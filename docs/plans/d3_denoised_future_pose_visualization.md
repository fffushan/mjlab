# D3 denoised future-pose visualization plan

Status: live integration deferred. The user authorized a simpler standalone tool instead.
Scope below: the original local D3/D4 browser diagnostic proposal. No policy, training, controller, or hardware change.

## Implemented first stage: standalone inspection

`inspect_sequence.py --execute` executes eight VAE bootstrap transitions, then samples once.
It fits 32 future poses and opens the looped playback viewer in that same command.
`--replay` is optional and only reopens a result from an earlier run.
The viewer starts paused at pose 1. Play loops through the fitted poses at 0.25x by default.
Step pauses and advances one pose. Reset pauses and returns to pose 1.
All-ghosts mode retains the original overlay; single-pose playback uses a solid robot.
Saved poses can be reopened without neural inference or physics.
There is no live replanning or sequence history.
See [launch and replay instructions](../../scripts/experiments/diffusion_d3_closed_loop/README.md#one-shot-future-pose-inspection).

CPU validation: 27 tool tests and 82 combined regressions pass.
Scoped Ruff/format/Ty/Pyright and the real artifact preflight pass.
The follow-up source review returned `OK with notes`; blocking findings are resolved.
The tests cover prefixed scene body names, static controls, opacity, replay errors, and server cleanup.
Operator reported off-center rendering. The static camera now uses the scene's world coordinates
and fitted body bounds. It has a `Center view` button.
Camera validation: the saved bundle's CPU framing check passes.
The operator accepted the camera correction and requested temporal playback.
Current playback validation: 29 tool tests and scoped Ruff/format/Ty/Pyright pass.
The 32 saved poses match full-forward body poses exactly when rendered with kinematics only.
The tests cover loop order, pause, step, reset, seek, speed, GUI synchronization, and cleanup.
Looped playback remains visually not tested in a browser.
The original live-integration plan below is not the implementation scope.

## Goal and interpretation

Show all 32 future poses from each completed denoising sequence.
Use the transparent robot mesh style of the reference-motion display.
Keep the measured robot visible in the same scene.

A sequence means the final clean sampler result for one control tick.
It does not mean every intermediate reverse-diffusion level.
The window has 41 rows: history `0:8`, current `8`, future `9:41`.
Future row `r` represents absolute tick `plan_tick + r - 8`.
At 50 Hz, the offsets are `+0.02` through `+0.64` seconds.

## Representation limitation

The 135 state channels contain root pose, root velocities, 20 body positions,
and 20 body velocities. They contain no joint angles or body orientations.
The reference-motion ghost renderer instead needs generalized coordinates (`qpos`).
Thus, the predicted states do not uniquely define full robot meshes.

Use inverse kinematics (IK) to fit display-only joint angles to predicted body positions.
Keep the predicted root pose fixed during each fit.
Render raw body points as an optional overlay beside the fitted mesh.
Report body-position fit errors. Never label fitted joints as direct model predictions.
Do not use future reference joint angles or roll out the VAE decoder to obtain poses.
Those alternatives show a different trajectory, not the denoised state sequence.

## Data flow

`final sampler result -> owned snapshot -> predicted world poses -> display-only IK -> reference-style ghosts`

### 1. Capture the completed sample

- Add an opt-in snapshot seam to `ReferenceFreeContinuation.step()` in `handover.py`.
- Capture the actual final sample after inpainting/guidance, not a separate inference.
- Copy the 32 future state rows to owned CPU storage.
- Bind the measured planning-center state and same-tick measured joint seed.
- Record generation, plan tick, future ticks, body order, and active keyframe rows.
- Preserve sampling draws, generated latent, decoded action, and history behavior.
- Keep the default headless evaluator path unchanged when snapshots are disabled.

### 2. Restore predicted world poses

- Reuse `pose_costs.world_poses_from_normalized_state()` with frozen state statistics.
- Restore each root using the measured planning-center position and yaw.
- Restore body positions using the predicted root yaw of each future row.
- Do not use recorded future yaws through `state.hybrid_to_world()`.
- Validate dimensions, finite values, body order, and six-dimensional root rotations.
- Mark an invalid row as invalid; do not fabricate its pose or alter control execution.

### 3. Fit robot meshes

- Use an isolated CPU MuJoCo model and data for display kinematics.
- Map the contract body names and robot joint names to the model explicitly.
- Start the first fit from the same-tick measured joints.
- Start each subsequent fit from the previous fitted future pose.
- Fit the 20 predicted body positions with joint limits and bounded iterations.
- Use a weak joint-seed term for unobservable joints and ambiguous solutions.
- Keep predicted root position and orientation fixed.
- Do not modify predictions to enforce contact, balance, or ground clearance.
- Preserve raw points, fitted joints, convergence flags, and per-body errors.
- If a fit fails, show its raw points and failure label instead of an invented mesh.

### 4. Render the sequence

- Add a D3-local overlay through `visual_playback.py` and `view.py`.
- Reuse the Viser reference-style ghost mesh path and visual-only model appearance.
- Show all 32 fitted meshes by default, at their predicted world poses.
- Use one stable ghost model and batched handles. Do not copy models every tick.
- Replace the displayed sequence when a new snapshot arrives.
- Do not accumulate ghosts from previous plans in the scene.
- Add controls: visibility, opacity, all-poses/selected-pose mode, and future index `1..32`.
- Show generation, plan tick, future tick, time offset, fit error, and fit status.
- Mark hard-keyframe rows when they are inside the future window.
- Freeze the latest sequence while simulation is paused or stopped.
- Clear snapshots and overlay handles on reset; never mix generations.

### 5. Preserve every sequence for inspection

- Capture every completed sample, even when browser rendering is slower than inference.
- Keep one latest snapshot for the live scene.
- Retain a bounded session history, with a plan selector available while paused.
- The history bound comes from the configured continuation-step limit.
- Older-plan selection changes only the overlay, not physics, policy history, or task state.
- Cache each IK result. Re-rendering must not resample, refit, or step physics.
- Optional export uses bounded chunked files, not an ever-growing JSON status record.
- Record omitted/invalid display results explicitly; do not silently drop plans.
- Measure IK and rendering cost before making a real-time claim.

## Verification and acceptance

1. Synthetic tests verify exactly 32 future rows and correct absolute tick mapping.
2. World-pose tests cover translation, center yaw, predicted yaw, and degenerate rotations.
3. Forward-kinematics fixtures verify IK on known reachable X2 poses.
4. Underdetermined and unreachable fixtures verify fit-error reporting and fallback display.
5. Fixed-seed tests compare actions, latents, RNG state, and sampler counts with visualization off/on.
6. Fake viewer tests verify 32 ghosts, sequence replacement, pause, plan selection, and reset cleanup.
7. Same-tick snapshot tests prevent mutable simulator data from changing an older plan.
8. Viewer tests verify every sample is retained even if rendering skips an update.
9. CPU preflight validates artifacts and task config without constructing a simulator.
10. After implementation approval, run a separately authorized bounded browser simulation.

The browser check uses `vgk005`, start frame `45`, and the frame-75 target at tick `38`.
The first plan starts at tick `8`; the target is future index `30` in that plan.
All 32 future slots must be accounted for. Invalid slots must have explicit labels.
Mesh fit quality must remain separate from diffusion prediction quality.
No validation has been performed for this proposed feature.

## Main files

- `src/mjlab/tasks/tracking/diffusion/handover.py`: opt-in final-sample snapshot.
- `src/mjlab/tasks/tracking/diffusion/pose_costs.py`: existing predicted-world-pose conversion.
- `src/mjlab/tasks/tracking/diffusion/future_pose_visualization.py`: proposed snapshot, IK, and display helpers.
- `src/mjlab/tasks/tracking/diffusion/visual_playback.py`: snapshot ownership and D3-local overlay.
- `scripts/experiments/diffusion_d3_closed_loop/view.py`: opt-in CLI and lifecycle.
- `src/mjlab/viewer/viser/scene.py`: existing batched reference-style ghosts; avoid global behavior changes.
- `tests/test_tracking_diffusion_visual_playback.py`: lifecycle regression coverage.
- `tests/test_tracking_diffusion_future_pose_visualization.py`: proposed extraction, geometry, and IK tests.
- `scripts/experiments/diffusion_d3_closed_loop/README.md`: launch and interpretation limits.
- `docs/source/changelog.rst`: user-facing feature record after implementation.
