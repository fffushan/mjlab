# Two X2 datasets: D1 windows and raw joint pose

Status: introduction only. This note does not authorize training or deployment.
Observation date: 2026-10-21 (+08).

## Purpose

The flow-matching work uses two different datasets. They answer different
questions. This note lists what each one stores.

## D1 dataset (used by the current diffusion and flow models)

The D1 dataset records rollouts of the frozen VAE student.

- One stored row is one control step.
- One row holds the 135-D physical state, the clean action, the OU noise, the
  executed action, and the 32-D VAE latent.
- The 135-D state is `root_position` (3) + `root_rot6d` (6) +
  `root_linear_velocity` (3) + `root_angular_velocity` (3) +
  `body_positions` (60) + `body_linear_velocities` (60), for 20 bodies.
- One window is 41 rows by 167 channels (135 state + 32 latent). Row 8 is the
  current row: 8 history rows, the current row, and 32 future rows.

The latent is the reason for the 167 width. The VAE encoder reads a 68-D input:
reference `q` (31) + reference `dq` (31) + anchor orientation error (6). It writes
`z` (32). The decoder maps `[z, proprio 99]` to 31 actions.

Measured dataset `d1-vgk36-double-20261012`:

| Item | Value |
|---|---|
| References and seeds | 36 references, 2 seeds |
| Trials | 9,216 |
| Stored rows | 2,294,642 |
| Windows | 579,762 (train 464,039 / validation 57,815 / test 57,908) |
| Shards | 562 |
| Train token cache | 12.47 GiB, float32 |
| Bytes per window | 26.7 KiB (41 x 167 x 4) |

## Raw joint-pose dataset (new, for the latent-free flow model)

The raw dataset records rollouts of the frozen multi-motion PPO tracker.

- One NPZ file holds one clip and one seed run. A JSON manifest describes it.
- One row stores `root_quat_wxyz` (4 values), `joint_pos` (31), and `joint_vel`
  (31). That is 66 stored values per row.
- The root quaternion is the achieved world-frame orientation. The collector
  does no heading alignment.
- One trajectory is the first episode of one environment row, including the
  terminal state. Each trajectory carries its outcome, seed, scenario, and push
  events.
- No latent, no action, and no normalization are stored.

The planned window is 41 rows by 68 channels: root 6D rotation (6), `q` (31), and
`dq` (31). The 6D packing happens later, at window preparation time. The widths
match the D1 encoder input, but the 6 rotation channels have a different meaning.
D1 stores an anchor orientation error. This dataset stores an absolute achieved
root rotation.

Measured batch of 12 motions:

| Item | Value |
|---|---|
| Configurations | 3 lane groups, 128 environments, 4 seeds |
| Files | 144 NPZ |
| Trajectories | 18,432 |
| Stored rows | 3,937,938 |
| Raw bytes | 1.09 GB |
| Candidate windows | 3,232,450 |
| Windows retained by the outcome gate | 3,230,586 |
| Bytes per window | 10.9 KiB (41 x 68 x 4) |

The remaining 132 motions are collecting in 11 batches. The full set is
221,184 planned trajectories. Row and window totals for the full set are
`not measured` until the collection and the audit finish.

## Differences that matter

| Item | D1 dataset | Raw joint-pose dataset |
|---|---|---|
| Controller that made the rows | VAE student | frozen PPO tracker |
| State per row | 135-D physical state | 66 values: root quaternion, `q`, `dq` |
| Latent | yes, 32-D | no |
| Window width | 167 | 68 |
| Storage | sharded row store | one NPZ per clip and seed, plus manifest |
| Eligibility | window index plus split groups | outcome gate: complete, finite, at least 41 rows |
| Root rotation | 6D in the state | quaternion `wxyz`, 6D later |
| Train cache | built, 12.47 GiB | planned with the same cache builder |

Both datasets use 41-row windows with the same 8 / 1 / 32 split. Both keep the
rows as the source of truth and keep the window as an index.

## Two warnings

1. Do not reuse the D1 projection or normalizer. The channel meanings differ.
2. Do not treat the two window counts as comparable samples. D1 windows come
   from VAE rollouts. Joint-pose windows come from tracker rollouts with pushes,
   phase randomization, and early terminations.
