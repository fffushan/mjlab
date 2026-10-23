# X2 joint-pose flow: offline rollout to closed-loop playback (MVP)

Date: 2026-10-23 (+08).
Status: implemented and validated on 2026-10-23. Code, tests and the first
measurement are recorded in `docs/reports/x2_joint_pose_flow_rollout_mvp.md`;
this document keeps the decisions, the contract and the runbook.
Scope: one tool that turns a trained window-generating flow model and one reference
clip into one long reference motion file, then runs the frozen X2 multi-motion
teacher on that file in closed loop, with a recorded-clip control arm.
Related: `docs/plans/x2_joint_pose_flow_closed_loop_handover.md` (sections 8, 9, 11),
`docs/reports/x2_joint_pose_flow_training_runs.md`.

## 1. Decision record

| Id | Decision | Reason |
|---|---|---|
| D1 | The tool hardcodes no window geometry. It reads `past_steps`, `window_steps`, `channels`, `channel_layout`, model width/layers/heads, `final_norm` and `max_level` from the checkpoint | the same tool must roll any trained arm or any later model in this lane without a code change |
| D2 | Chunk stride is `window_steps - past_steps - 1` | that is the largest stride whose next centre row was already produced by the previous chunk, so the heading frame matches the training convention exactly. Current schema: 41 − 8 − 1 = 32 |
| D3 | Body channels come from the source clip at the same absolute frames | cheap substitution, the pattern already used at 199,879 reference segments. Joint channels come from the model |
| D4 | The `past_steps` history rows are written as a real prefix, and the file is truncated to `target_frames` rows | both files start at the requested frame and hold the same row count, so the A/B is row-aligned and the body substitution never runs past the clip end |
| D5 | The numeric arm is `scripts/evaluate_tracking_policy.py`; the control arm is the recorded clip for the same span | no new metric code, existing teacher-matched metrics |
| D6 | No digests and no input inventory. Provenance is the input paths, the checkpoint `global_step`, the loaded schema and the sampler settings, written to one JSON | the tool is run by one operator against one known checkpoint; hashing buys nothing here |
| D7 | No change to the frozen teacher, the command, the observation terms, the dataset builder, the controller, or the export bundle | evaluation tool only |

D3 and D4 are operator decisions (2026-10-23). D1 and D2 are what make the tool
model-agnostic. The rollout loop and the CLI depend on a small sampler protocol,
not on a checkpoint format (section 3).

## 2. What the tool reads from a model

Nothing below is a constant in the code. Each value comes from the checkpoint, and
the tool refuses only what cannot work.

| Value | Source key | Fallback |
|---|---|---|
| `past_steps`, `window_steps`, `channels` | `payload["dataset_cfg"]` | refuse; the tool never guesses |
| `channel_layout` | `payload["dataset_cfg"]["channel_layout"]` | the lane constant `joint_pose.config.CHANNEL_LAYOUT` |
| width, layers, heads, `final_norm` | `payload["model_settings"]` | refuse |
| `max_level` | `payload["trainer"]["max_level"]` | `model_settings.embedding_count - 1` |
| parameter set | `payload["ema"]` present or absent | `ema` when present, else `model`, and record which |
| `global_step` | `payload["global_step"]` | `null` |
| normalizer | `<dataset_cfg["data_dir"]>/projection.npz` | `--projection` override |

Two limits of the current lane, stated so the seam is honest. The channel layout is
fixed to X2: `root_rot6d 0:6`, `joint_positions 6:37`, `joint_velocities 37:68`,
because `JointPoseDatasetCfg` validates that sum. A different robot needs a layout
entry, not a tool change. A model with different channel *semantics*, such as the
D3 latent lane, needs a new backend and a new exporter, not a change to the loop.

## 3. Backend seam

`joint_pose/rollout.py` defines the only interface the loop uses:

```python
class WindowSampler(Protocol):
  past_steps: int
  window_steps: int
  channels: int

  def sample(
    self, history: Tensor, *, steps: int, generator: torch.Generator
  ) -> Tensor: ...  # [window_steps, channels], history rows returned unchanged
```

`joint_pose_flow` is the default backend. It loads the payload, builds the denoiser
with `build_denoiser`, loads `model`, applies `apply_ema_selection`, and samples
with `sample_trajectory` under a mask that fixes the first `past_steps` rows. It is
about 60 lines and needs no dataset directory, because it does not call
`FlowMatchingRunner.load` (that path refuses a checkpoint without the 427 GB cache,
`joint_pose/runner.py:277`).

A second backend is one class of about 40 lines. A stub backend that returns a
recorded future is the M2 correctness proof, so the seam is tested from the first
commit.

The **export** stage is joint-pose specific, because the body substitution needs
`q`, `dq` and a clip. That is a property of the lane, not of the tool.

## 4. Deliverables

| Path | Role |
|---|---|
| `src/mjlab/tasks/tracking/joint_pose/rollout.py` | window codec, `WindowSampler` protocol, `joint_pose_flow` backend, chunk loop, diagnostics |
| `src/mjlab/scripts/joint_pose_rollout.py` | tyro CLI, npz writer, control file, JSON record |
| `tests/test_joint_pose_flow_rollout.py` | CPU tests, tiny checkpoint fixture, toy-schema model |
| `scripts/play_reference.py` | reference-only kinematic playback; added after the first measurement, see 7.4 |
| `docs/reports/x2_joint_pose_flow_rollout_mvp.md` | evidence and limits |

Two untracked files in this worktree, `scripts/export_x2_joint_pose_bundle.py` and
`tests/test_joint_pose_bundle_export.py`, belong to the bundle workstream. This
plan does not edit them.

## 5. Milestones

### M1 — Window codec and backend (3–4 h)

`rollout.py`:

- `encode_window(root_quat_w, joint_pos, joint_vel, start, *, past_steps,
  window_steps)` returns `[window_steps, channels]` float32 in the dataset's
  heading-local convention. Generic over the window size.
- `reanchor_window(rows, centre_quat, *, past_steps)` re-expresses a denormalized
  window in the frame of a new centre row: rotate the 6D root channel only, leave
  `q` and `dq` untouched.
- `normalize` / `denormalize` over `evaluate.ChannelNormalization`.
- `joint_pose_flow(checkpoint, *, parameters="ema")` returns a `WindowSampler` plus
  the loaded schema. No dataset argument.

Gate: encoder parity with `prepare.segment_windows` is at most `1e-6` absolute for
the current schema. The centre row of every encoded window has zero yaw; the
residual is tilt only. A toy schema (20 rows, 4 history) loads and encodes, which
proves the geometry is not hardcoded. One real-checkpoint GPU smoke on one window
is finite and repeats at one seed.

### M2 — Chunk rollover (2–3 h)

`rollout.py`:

- `rollout(clip, sampler, normalization, *, start_frame, target_frames, steps,
  generator, drift_limit=None)` returns generated rows in physical units plus
  per-chunk diagnostics.
- Frame accounting: chunk 0 writes rows `s+past .. s+window-1`; chunk k starts at
  `s + stride*k`. Each chunk adds `stride` frames and regenerates one overlap row.
  Chunk count is `ceil(max(0, target_frames - window) / stride) + 1`, because
  `target_frames` counts file rows and the prefix supplies the first `past` of
  them. A 453-frame target on the current schema needs 14 chunks: they cover 449
  generated frames, the prefix makes 457 rows, and the export truncates to 453.
- Diagnostics per chunk: `anchor_yaw_deg`, `overlap_rms_rad` (regenerated centre
  row against its previous value), `joint_drift_rms_rad` and `yaw_drift_deg`
  against the source clip.
- Optional stop rule: abort and label the result `drift_limit` when the joint RMS
  drift passes `drift_limit`.

Gate: a stub sampler that returns the recorded target window reconstructs the
source clip. Over 200 frames the maximum joint difference is at most `1e-5` rad,
and the overlap residual is zero. The same test passes for the current schema and
for the toy schema, so the loop is not tuned to 41/8.

### M3 — Export, playback, and the closed-loop A/B (2 h)

`joint_pose_rollout.py` writes three files into `--output-dir`:

- `generated.npz`: `past_steps` history rows plus generated frames. `joint_pos` and
  `joint_vel` come from the rollout. `body_pos_w`, `body_quat_w`,
  `body_lin_vel_w`, `body_ang_vel_w` are the source clip rows for the same
  absolute frames. `fps` is the clip rate.
- `control.npz`: the same span of the source clip, unchanged, same row count.
- `rollout.json`: checkpoint path, `global_step`, parameter set, loaded schema,
  projection path, clip path, start frame, `target_frames`, file rows, chunk count,
  stride, sampler steps, seed, per-chunk diagnostics, and the D3 statement.

Both files are truncated to `target_frames` rows, where `target_frames` defaults to
`clip frames - start_frame`. The rollout may produce up to `stride - 1` frames more
than it exports; the extra tail is not written.

Then run the visual arm and the numeric arm on both files.

Gate: both files load in the command loader. `play` exits 0 for both.
`evaluate_tracking_policy.py` writes JSON for both. No NaN in any array. The first
`past_steps` joint rows of `generated.npz` equal the clip rows to float32
tolerance.

### M4 — Evidence and handoff (1–2 h)

`docs/reports/x2_joint_pose_flow_rollout_mvp.md` holds the A/B table, the drift
table, the exact commands, wall times, and the section 8 limits. Add one compact
entry to `research/NOW.md`. Commit code, tests, plan and report as one scoped
commit on `fxy/final-norm-ablation`; do not push without authorization.

Gate: numbers are labelled single-run with no repeats. No quality claim beyond the
pair.

## 6. Test plan

| Test | Proves |
|---|---|
| encoder parity with `prepare.segment_windows` | the runtime codec cannot drift from the dataset codec |
| centre-row yaw is zero | the heading convention is the training one |
| re-anchor round trip | yaw composition is invertible and does not touch joints |
| toy schema loads, encodes and rolls | no geometry is hardcoded |
| stub-sampler reconstruction, both schemas | frame accounting and re-anchoring are correct without a trained model |
| stride and chunk-count formula | file rows are predictable for any `window`/`past` |
| backend refusals | a missing `model_settings`, a layout mismatch, and a wrong channel count fail loudly |
| sampler determinism | one seed gives one file |
| export substitution | body arrays are the clip rows; joint arrays are the generated rows |
| history prefix | the file starts at the requested frame and the prefix is the real clip |
| JSON completeness | every field in section 5 M3 is present |

All tests run on CPU with a tiny model. No dataset directory is needed.

## 7. Runbook

Prerequisites: one joint-pose flow checkpoint plus its `projection.npz`, one
tracking clip, and the frozen teacher checkpoint with its `params/env.yaml`.
The arms used below are staged at `~/data/joint-pose-flow-20261023/<arm>/`.

### 7.1 Generate the pair

```bash
C=~/data/joint-pose-flow-20261023/400k-b512-mask02-20261022
uv run python -m mjlab.scripts.joint_pose_rollout \
  --checkpoint-file $C/model_0400000.pt \
  --projection $C/projection.npz \
  --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \
  --start-frame 0 --target-frames 453 --steps 20 --seed 0 --device cuda:0 \
  --output-dir logs/diagnostics/joint-pose-rollout-<date>/single_000
```

Writes `generated.npz`, `control.npz` and `rollout.json`. 453 frames take about
6 s, most of it process start.

| Flag | Meaning | Default |
|---|---|---|
| `--checkpoint-file` | joint-pose flow checkpoint | required |
| `--motion-file` | tracking clip | required |
| `--output-dir` | output directory | required |
| `--projection` | `projection.npz` with `mean` and `std` | `<checkpoint data_dir>/projection.npz` |
| `--start-frame` | first clip frame written | `0` |
| `--target-frames` | rows wanted in both files | clip length − start frame |
| `--steps` | denoiser calls per chunk | `20` |
| `--keyframes` | `none`, or `current`, `last`, `current,last`; pins reference rows inside every window | `none` |
| `--parameters` | `auto`, `ema`, `model` | `auto` |
| `--drift-limit` | stop early above this joint drift, radians | off |
| `--device` | `cuda:0` or `cpu` | `cpu` |

`--projection` is required for these arms: the checkpoint records
`data_dir: /tmp/joint-pose-144/data`, which does not exist. Any clip under
`data/tennis/single_*.npz` works; keep `start_frame + target_frames` inside the
clip.

`--keyframes` pins recorded clip rows inside every generated window. `current`
is row `past_steps` of the window, the anchor row the tracker calls now; `last`
is row `window_steps - 1`. Both names resolve from the model geometry, so they
stay correct for a different window size. Pin both: with `stride = window -
past - 1` the last row of a chunk is the anchor row of the next, and the loop
rewrites it, so pinning only `last` loses the value one chunk later. A pinned
row is the clip frame at the same absolute frame, expressed in the window's
anchor frame, so each chunk becomes a bridge between two real poses and the
heading is re-anchored on the reference every `stride` frames. The model was
trained with clean rows only in the history, so this is an inference-time
request, not a trained behaviour. Each chunk reports `pinned` and
`seam_rms_rad`, the snap between the generated history and the pinned pose.

### 7.2 Read the record first
`rollout.json` holds one entry per chunk with `joint_drift_rms_rad` and
`yaw_drift_deg` against the clip. A large chunk-0 value means the checkpoint is
not usable and a longer rollout will not help. On `single_000` at step 400000
chunk 0 is 0.067 rad and the heading error passes 45 degrees at frame 183.

### 7.3 Closed-loop arms

Visual, with the frozen teacher:

```bash
T=logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/training/agibot_x2_multi_motion_tracking/2026-10-14_21-06-44_all-tennis-8192x8-resume-20k/model_19999.pt
uv run play Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion \
  --checkpoint-file $T \
  --motion-file logs/diagnostics/joint-pose-rollout-<date>/single_000/generated.npz
```

`play` needs `params/env.yaml` beside the checkpoint to restore `lookahead_s 0.2`;
without it the actor width is wrong. The viewer has no exit condition, so close
the window.

Numeric, both arms, same seed:

```bash
for arm in control generated; do
  uv run python scripts/evaluate_tracking_policy.py \
    --checkpoint-file $T \
    --motion-file logs/diagnostics/joint-pose-rollout-<date>/single_000/$arm.npz \
    --task Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion \
    --num-envs 512 --seed 0 \
    --output-json logs/diagnostics/joint-pose-rollout-<date>/single_000/$arm-eval.json
done
```

The evaluator's default task is the G1 task, so `--task` is mandatory. Without
the control arm the generated number means nothing.

### 7.4 Reference-only playback

`scripts/play_reference.py` shows the reference alone: it writes the pose into
`mj_forward` every frame and runs no physics, no policy and no episode, so the
walkthrough does not stop when a reference would make the robot fall. Playback
repeats until the window is closed, and the camera tracks the pelvis.

```bash
uv run python scripts/play_reference.py --motion-file <out>/generated.npz
uv run python scripts/play_reference.py --motion-file <out>/control.npz
```

| Flag | Effect |
|---|---|
| `--speed` | playback rate, `0.5` is half speed |
| `--passes N` | fixed number of passes, `0` repeats forever |
| `--once` | one pass, then hold the last pose |
| `--start-frame`, `--end-frame` | frame range |
| `--free-camera` | disable pelvis tracking |
| `--check-only` | verify and print, open no window |
| `--allow-mismatch` | play despite a failed layout check |

The tool checks itself before playing: it sets `qpos` from the file, runs
forward kinematics and compares with the stored `body_pos_w`. Exit 2 means no
sampled frame reproduces its stored bodies, so the joint order or the root
layout is wrong. A generated file matches at its clip-prefix rows and disagrees
later, which is reported and allowed: on `single_000` the worst mismatch is
0.48 m while the best is 0.00 mm, because the body channels are the clip's and
the joints are the model's. The viewer therefore shows the model's joints under
the clip's root pose, and feet leave the floor where that mismatch is large.

## 8. Limits the report must state

- This is open-loop generation. The model never sees the robot. The tool measures
  the frozen teacher on a generated reference.
- The cheap substitution means root translation and root orientation come from the
  recorded clip. A generated turn or travel is not represented. Only joint-space
  fidelity is under test.
- Generated `dq` is a model output, not the finite difference of generated `q`.
  Report the mismatch; do not correct it.
- The reference body arrays and the generated joints can disagree once the rollout
  drifts. The A/B difference is then partly a reference inconsistency, not only a
  tracking failure.
- 3 of 4 trained arms diverged. Use the stable arm and name it in every report.
- The stable arm trained with history dropout `0.2`, so it was trained to work
  with masked history.
- `play` gives no numbers. The numbers come from the evaluator, and one seed is
  not a noise floor. The recorded teacher noise floor at 1024 environments was
  0.1–0.4 percent and plus or minus 4 failures; at 512 environments it is wider.
- No hardware, deployment, or real-time claim.

## 9. Non-goals

- Mid-episode reference installation in the live command (handover doc section 5).
- FK body reconstruction and a root-height rule (handover doc section 9).
- The replay-survey adapter, multi-clip sweeps, and aggregate metrics.
- A backend for the latent D3 lane.
- Training, checkpoint selection, dataset changes, controller changes, ONNX
  export, and any `x2_docker` change.

## 10. Risks and stop rules

| Risk | Mitigation |
|---|---|
| Heading-frame slip compounds across chunks | stride from D2, the stub-reconstruction test, and `anchor_yaw_deg` in every record |
| Model drift makes long rollouts meaningless | per-chunk drift table, optional `--drift-limit`, and `--target-frames` |
| Wrong checkpoint arm | name the arm and `global_step` in the record; the stable arm is `400k-b512-mask02-20261022` |
| Actor width mismatch in `play` | check `params/env.yaml` and `lookahead_s 0.2` before the run |
| Laptop memory for 512 environments | start at 128 and record the count; the server is available if needed |
| Tool passes while the model is wrong | the A/B pair plus the stub test separate harness faults from model quality |

## 11. MVP definition of done

One command produces `generated.npz`, `control.npz` and `rollout.json` from one
checkpoint and one clip, with the schema read from the model. `play` runs both
files with the frozen teacher, exits 0, and reaches the clip endpoint or records a
termination. The evaluator writes one JSON per file. The report states the A/B
numbers, the drift table, and every limit in section 8. A model with a different
window or history length rolls through the same code path. No source file outside
this plan's deliverables changes.
