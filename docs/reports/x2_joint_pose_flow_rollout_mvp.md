# X2 joint-pose flow: rollout evaluation MVP

Kind: implementation record and first measurement.
Date: 2026-10-23 (+08).
Scope: the offline rollout tool, its CPU tests, and one closed-loop A/B on one clip
with one checkpoint. No hardware, no deployment, no training change.
Plan: `docs/plans/x2_joint_pose_flow_rollout_eval_plan.md`.
Model source: stable arm `400k-b512-mask02-20261022`, final checkpoint
`model_0400000.pt` (320,622,878 B, `global_step` 400000).

## 1. Implementation status

| Item | State |
|---|---|
| `src/mjlab/tasks/tracking/joint_pose/rollout.py` (886 lines) | implemented: window codec, `WindowSampler` protocol, `joint_pose_flow` backend, chunk loop, diagnostics |
| `src/mjlab/scripts/joint_pose_rollout.py` (351 lines) | implemented: CLI, generated/control pair, `rollout.json` |
| `tests/test_joint_pose_flow_rollout.py` (462 lines) | implemented: 16 tests, CPU only |
| `scripts/play_reference.py` | implemented: reference-only kinematic playback, added after the first measurement (3.6) |
| `--keyframes current,last` inpainting | implemented after the first measurement (3.7): pins the anchor row and the window's last row to the reference |
| Changelog bullet | added |
| Dataset directory read | none: schema comes from the checkpoint payload |
| Hardcoded window geometry in the loop | none: `past_steps`, `window_steps`, `channels` come from the model |
| Digests or input inventory | none, by operator decision |

The tool reads `past_steps`, `window_steps`, `channels`, `channel_layout`, the
denoiser sizes, `final_norm` and `max_level` from the checkpoint. Stride is
derived as `window_steps - past_steps - 1`. A model with a different window or
history length uses the same code path.

## 2. Commands that produced this record

```bash
uv run python -m mjlab.scripts.joint_pose_rollout \
  --checkpoint-file ~/data/joint-pose-flow-20261023/400k-b512-mask02-20261022/model_0400000.pt \
  --projection ~/data/joint-pose-flow-20261023/400k-b512-mask02-20261022/projection.npz \
  --motion-file ~/projects/mjlab/data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \
  --start-frame 0 --target-frames 453 --steps 20 --seed 0 --device cuda:0 \
  --output-dir logs/diagnostics/joint-pose-rollout-20261023/single_000
```

Artifacts are in `logs/diagnostics/joint-pose-rollout-20261023/single_000/`
(git-ignored): `generated.npz` and `control.npz` (867,926 B each, 453 rows),
`rollout.json`, two evaluator JSONs, and two `play` logs.

`--projection` is required for these arms. The checkpoint records
`data_dir: /tmp/joint-pose-144/data`, which does not exist.

The reference alone is inspected with the kinematic player, which runs no
policy and no episode:

```bash
uv run python scripts/play_reference.py \
  --motion-file logs/diagnostics/joint-pose-rollout-20261023/single_000/generated.npz
```

The keyframe arm adds one flag to the generator command:

```bash
uv run python -m mjlab.scripts.joint_pose_rollout ... --keyframes current,last \
  --output-dir logs/diagnostics/joint-pose-rollout-20261023/single_000-keyframes
```

## 3. Validation evidence

### 3.1 Tool correctness (CPU)

| Check | Result |
|---|---|
| `pytest tests/test_joint_pose_flow_rollout.py -q` | **16 passed in 1.36 s** |
| `ruff format --check`, `ruff check` | clean |
| `ty check` on both source files | clean |
| Encoder parity with `prepare.segment_windows` | ≤ 1e-6 |
| Stub-reconstruction (41/8 and toy 20/4) | clip reproduced ≤ 1e-5 rad, overlap 0, history rows returned unchanged |
| Export substitution | body arrays identical to the clip rows, control identical to the clip |
| Real-checkpoint determinism, seed 0 twice | byte-identical `joint_pos` |
| Real-checkpoint seed 1 | differs, max 1.50 rad |

The stub test is the harness proof. A sampler that returns the recorded future
must reproduce the clip. It does, on both geometries. A frame-accounting error
fails this test. The first version of the code had such an error: the chunk-0
anchor ignored `start_frame`. Two tests caught it, and it is fixed.

### 3.2 Window convention on the real model

For each chunk, the generated centre row should have zero yaw in its own window
frame, because every training window is centred there. Measured deviation:

| Chunk | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Deviation (deg) | 0.27 | 0.51 | 0.12 | −0.53 | 0.33 | −0.54 | 3.11 | 1.23 | 0.65 | 1.07 | −0.12 | −0.57 | 0.84 | −1.48 |

Maximum 3.1 deg, typical under 1.5 deg. The model does respect the window
convention, so the rollover feeds the frames the model was trained on.

### 3.3 Rollout behaviour, one clip, one seed

453 rows, 14 chunks, stride 32, `stop_reason = target`, wall time 5.8 s
(about 4 s is process start). Generation is not the bottleneck.

| k | start | anchor yaw (deg) | overlap (rad) | joint drift (rad) | yaw drift vs clip (deg) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | −4.08 | – | 0.067 | 7.4 |
| 1 | 32 | −6.26 | 0.0139 | 0.207 | 1.6 |
| 2 | 64 | −12.92 | 0.0182 | 0.243 | 0.4 |
| 3 | 96 | −47.54 | 0.0194 | 0.244 | −5.2 |
| 4 | 128 | −70.09 | 0.0258 | 0.415 | 33.4 |
| 5 | 160 | −34.91 | 0.0198 | 0.556 | 48.9 |
| 6 | 192 | 68.58 | 0.0266 | 0.607 | 44.7 |
| 7 | 224 | 105.85 | 0.0299 | 0.505 | 75.5 |
| 8 | 256 | 115.48 | 0.0258 | 0.381 | 102.8 |
| 9 | 288 | 100.60 | 0.0234 | 0.395 | −117.6 |
| 10 | 320 | 3.72 | 0.0175 | 0.376 | 172.8 |
| 11 | 352 | −14.42 | 0.0115 | 0.372 | −84.1 |
| 12 | 384 | −20.20 | 0.0149 | 0.314 | −5.9 |
| 13 | 416 | 8.28 | 0.0270 | 0.361 | −40.9 |

Overlap is the regenerated centre row against its previous value: 0.011–0.030 rad,
which is small. Joint drift grows from 0.067 rad at the first chunk to 0.3–0.6 rad
after chunk 4. The anchor heading leaves the clip's heading from chunk 3 and then
wanders over more than 180 deg. Whole-clip joint RMS against the clip is
**0.383 rad**. The maximum absolute joint angle is 2.12 rad and every value is
finite.

### 3.4 Closed loop, frozen teacher, 128 episodes, seed 0

Teacher: `model_19999.pt` of the 20k multi-motion run, `lookahead_s 0.2`,
`episode_length_s 9.06`, actor corruption on, `push_robot` off.

| Arm | Complete | Failed | Timeout | `r_mpkpe` (m) | `jvel` | Steps |
|---|---:|---:|---:|---:|---:|---:|
| control (recorded clip) | 128 | 0 | 0 | 0.0460 | 1.0377 | 453 |
| generated | 0 | 128 | 0 | 0.0774 | 1.0022 | 151 |

Failure time: min 1.46 s, median 1.52 s, max 3.02 s. Failure phase fraction:
0.159 to 0.332 of the clip. The evaluator counts outcomes and does not record
which termination term fired.

### 3.5 Visual arm

`play` was started on both files with the teacher and the viser viewer. Both log
`Using local motion file`, `Loading checkpoint`, and
`Restored lookahead_s=0.2 from params/env.yaml`, and both show the 62-wide
`motion_lookahead` term. The viewer loop has no exit condition, so each run ended
by the harness timeout (exit 124) after stepping. No exception occurred.
`play` is a visual check only and produced no numbers.

### 3.6 Reference-only playback

`scripts/play_reference.py` plays the reference alone, with no physics, no policy
and no episode reset, so inspection never stops when the reference would make
the robot fall. Playback loops until the window closes and the camera tracks the
pelvis.

| File | forward kinematics vs stored bodies |
|---|---:|
| `control.npz` | 0.00 mm worst, 0.00 mm best |
| `generated.npz` | **481.57 mm worst, 0.00 mm best** |
| control with the joint columns reversed (a deliberate fault) | 639.27 mm worst, 291.47 mm best, refused with exit 2 |

The zero best value is the clip-prefix rows, which are real joints. The 0.48 m
worst value is the substitution defect, measured: at those frames the stored
body channels and the generated joints describe different poses. The viewer
shows the model's joints under the clip's root pose, so feet leave the floor
where that mismatch is large. Exit codes verified: 0 when playable, 2 when no
sampled frame reproduces its bodies, 1 when no viewer is available.

### 3.7 Keyframe inpainting, `current,last`

`WindowSampler.sample` now accepts `pinned` rows. `JointPoseFlowSampler` marks
them clean and restores them at level 0 after every Euler step. `rollout` builds
each pinned row from the clip in the window's anchor frame, passes it, and fails
if the sampler does not return it unchanged. The names resolve from the model
geometry: `current` is row `past_steps` (8 here), `last` is row
`window_steps - 1` (40 here). Both are pinned because with `stride = 32` the last
row of a chunk is the anchor row of the next.

Like-for-like pair: `start_frame 0`, `target_frames 453`, `steps 20`, `seed 0`,
`model_0400000.pt`, 14 chunks each.

| Metric | `none` | `current,last` |
|---|---:|---:|
| chunk interior joint drift, mean of 14 chunks (pinned rows excluded) | 0.361 rad | **0.299 rad (−17 %)** |
| whole-file joint RMS against control | 0.383 rad | 0.316 rad |
| snap at the seam, per chunk | – | 0.251–0.375 rad |
| forward kinematics vs stored bodies, worst | 481.57 mm | 467.59 mm |
| teacher, 128 envs, seed 0, complete | 0 / 128 | 0 / 128 |
| failure time, median | 1.52 s | **2.40 s** (phase 0.265, max 3.1 s) |
| `r_mpkpe` | 0.0774 | 0.0809 |
| `jvel` | 1.0022 | 1.2625 |

Read the heading with care. `yaw_drift_deg` reports 0.0 on every pinned chunk,
but that is tautological: the metric samples the chunk's last row, which is the
pinned one. The real heading evidence is that the anchor yaw now equals the
clip's at every chunk, and that survival moved from 1.5 s to 2.4 s.

So the pin works mechanically and is not free. It removes heading drift, cuts
the interior joint error by 17 %, and buys about 0.9 s of teacher survival. It
also adds a 0.25–0.38 rad snap at every chunk and sharpens the velocity
reference, which raises `jvel` from 1.00 to 1.26. It does not fix interior
accuracy: no episode completes, and the worst frame is still 0.47 m away from
the stored bodies.

## 4. Interpretation

- The harness is validated. The control arm completes 128 of 128 episodes on the
  recorded clip, the stub sampler reproduces the clip, and the model's own
  centre-row convention holds to 3 deg. A failure on the generated file is not a
  frame-accounting artefact.
- The model at step 400000 of the stable arm does not hold a usable heading. The
  first chunk is already 7.4 deg off in yaw and 0.067 rad in joints. From chunk 3
  the anchor heading leaves the clip and wanders. The teacher fails at about
  1.5 s, which is inside chunk 2, where joint drift is 0.24 rad.
- The stable arm is the only one of four that did not diverge. This result is a
  property of that checkpoint, not of the tool.

Scope labels: the A/B is **one clip, one seed, 128 environments, one checkpoint**.
It is **not** a noise floor and **not** a quality score. The generated reference
takes its root path, heading and body channels from the recorded clip, so only
joint-space fidelity is under test. Generation is open loop: the model never sees
the robot.

## 5. Limits

- Body channels are the recorded clip rows. Root translation and world orientation
  are the clip's, so a generated turn or travel is not represented.
- Generated `dq` is a model output, not the finite difference of generated `q`.
  The mismatch is not measured here.
- Once the rollout drifts, the reference body arrays and the generated joints
  disagree. The A/B difference is then partly a reference inconsistency.
- One seed at 128 environments. The recorded teacher noise floor at 1024
  environments was 0.1–0.4 percent and plus or minus 4 failures. At 128
  environments the floor is wider and not measured.
- No hardware, deployment or real-time claim.

## 6. Next actions

1. Repeat the A/B on two or three clips, including one turning clip and one
   in-place clip, to separate clip difficulty from model quality. Needs a decision.
2. Gate the rollout on the first-window error before spending simulator time:
   abort when chunk 0 exceeds a joint or yaw threshold.
3. Use the replay-survey adapter when aggregate numbers are needed; `play` is
   visual only.
4. The training side still needs the fixes listed in
   `docs/reports/x2_joint_pose_flow_training_runs.md`: horizon weighting, a
   bounded input clip, and a validation loop for checkpoint selection. This
   evidence is one input to that decision, not the decision.
