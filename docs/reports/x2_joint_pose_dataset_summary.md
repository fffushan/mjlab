# X2 joint-pose dataset: format, storage, location

Status: prepared dataset exists. Not yet used by any training run.
Observed: 2026-10-22 (+08).
Related: `x2_joint_pose_datasets.md` (comparison with the D1 dataset),
`x2_latent_free_joint_space_feasibility.md` (design rationale).

## Summary

The dataset holds achieved joint-space motion of the AgiBot X2. A frozen PPO
tracker produced every row. The dataset is the training input for a latent-free
flow model that predicts future root orientation, joint positions `q`, and joint
velocities `dq`.

The dataset does not contain a VAE latent, an action, or a reference command.

## Location

All paths are on the shared GPFS NAS (`10.14.80.16`). Every GPU host sees them
under `/home/fushan`.

| Path | Content |
|---|---|
| `/home/fushan/data/mjlab-latent-free-144/data/` | prepared dataset: store, split, index, statistics, cache |
| `/home/fushan/mjlab/logs/diagnostics/x2-joint-pose-batch12-20261021T1700/` | raw NPZ of the 12-motion batch |
| `/home/fushan/mjlab/logs/diagnostics/x2-joint-pose-remaining132-20261021/batch00…batch10/` | raw NPZ of the 11 remaining batches |
| `/home/fushan/mjlab/logs/diagnostics/x2-joint-pose-all144-eligibility-20261022/index/eligibility.json` | eligibility index |
| `/home/fushan/mjlab-joint-pose-coverage-smoke-20261021T1630/` | frozen collector worktree |

## Provenance

The dataset uses 144 tennis motions. A frozen multi-motion PPO tracker rolled
them out in 12 collections. Each collection used three lanes of 128
environments and four seeds (100 to 103). The lanes were start/baseline,
uniform/baseline, and uniform/disturbed.

| Stage | Count |
|---|---:|
| Trajectories collected | 221,184 |
| Raw rows | 46,771,215 |
| Raw NPZ files | 1,728 |
| Raw size | 16.4 GB |
| Accepted trajectories | 199,879 |
| Rejected trajectories | 21,305 |
| Store rows | 46,273,479 |
| Retained windows | **38,278,319** |

The gate rejected 1,262 non-complete trajectories and 20,043 trajectories with
fewer than 41 rows. It rejected no trajectory for a non-finite value.

## Format 1: raw row

One row is one control step at 50 Hz. A row holds `root_quat_wxyz` (4 values),
`joint_pos` (31), and `joint_vel` (31). A trajectory also holds its outcome,
seed, scenario, group, and push events.

The raw format has no root translation and no body kinematics.

## Format 2: store

Three dense float32 memmaps in trajectory order, plus one manifest.

```
store/root_quat_wxyz.npy   [46273479, 4]
store/joint_pos.npy        [46273479, 31]
store/joint_vel.npy        [46273479, 31]
store/manifest.json        per-segment table: source, motion, group, seed, offsets
```

The store holds 199,879 segments. It uses 14 GB on disk and 12.2 GB of payload.

## Format 3: window

A window is 41 rows by 68 channels. It holds 8 history rows, the current row, and
32 future rows. The current row is index 8.

| Channels | Content |
|---|---|
| 0:6 | root orientation 6D: `R00,R10,R20,R01,R11,R21`, heading-local to the center row |
| 6:37 | joint positions, radians |
| 37:68 | joint velocities, radians/second |

Heading localization applies `Rz(-yaw(center))` to every row of the window.
One window uses 11,152 bytes in float32.

## Format 4: cache

The cache is one `.npy` memmap per split. The dtype is float32. The values are
normalized with train-only statistics.

| Split | Windows | File size |
|---|---:|---:|
| train | 32,004,718 | 356.9 GB |
| validation | 3,018,009 | 33.7 GB |
| test | 3,255,592 | 36.3 GB |
| total | **38,278,319** | **426.9 GB** |

The split is by whole motion: 120 train, 12 validation, 12 test. The assignment
is a deterministic stride over the sorted motion names.

The storage reports 139 GB allocated for 426.9 GB apparent. This is
storage-side compression. We did not configure it and did not verify it. Do not
plan capacity from the allocated value.

## Statistics

`projection.npz` holds the per-channel mean and standard deviation for the 68
channels. The fit used the train split only: 166,648 segments and 1,312,193,438
samples at stride 1.

Normalization of the six root-6D channels break exact orthonormality.
Re-orthonormalize at inference.

## Group composition

| Group | Accepted trajectories | Windows |
|---|---:|---:|
| start/baseline | 73,700 | 20,491,328 (53.5 %) |
| uniform/baseline | 64,012 | 9,154,758 |
| uniform/disturbed | 62,167 | 8,632,233 |

The window ratio of baseline to disturbed is 3.4 to 1. Training weights must
correct this ratio.

## Identity

| Artifact | Identity |
|---|---|
| Eligibility index, 60,601,557 bytes | SHA-256 `996c5a2ed788863cac5f5fcaf87f564043c433e45c49c1b2830b25ca9ab08d41` |
| Collector source | commit `1035b4faf` plus the coverage patch |
| Gate source | commit `185ea1e4d` |
| Tracker checkpoint | SHA-256 `82f0dc32…`, `model_19999.pt` |

## Measurement

The dataset has been measured for trackability and mode coverage. Results and the
tooling that produced them are in `x2_joint_pose_trackability_and_modes.md`.
Briefly: 10.4 % of segments fail when replayed as references, the dominant
termination is end-effector height, and the achieved motion is more upright, more
crouched and 14 % slower than the reference. No data has been removed.

## Limitations

1. The raw format has no root translation and no body kinematics. A replay test
   cannot use the full reference record without a new collection run.
2. The outcome `failure` is a tracker termination label. It covers torso height,
   torso tilt, and ankle or wrist height. It is not a fall detector.
3. The dataset records shape eligibility only. It has no per-window measure of
   tracking quality.
4. The two uniform lanes lost about 13 % of their trajectories to the 41-row
   minimum. Their trajectory counts therefore overstate their usable coverage.
5. The preparation script `prepare.py` is a one-time tool. It is not committed
   and has no unit tests.
6. The split is not stratified by clip length or activity.
