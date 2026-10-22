# Joint-pose dataset preparation

Two one-time tools that turn the collected raw rollouts into the window dataset
the flow trainer consumes. Neither is a library and neither runs during training.

| Tool | Output |
|---|---|
| `prepare.py` | `store/` (three float32 memmaps plus a per-segment manifest), the motion split, `window_index.npz`, `preprocess_spec.json`, train-only statistics in `projection.npz` and `stats.json`, and the window cache `offline/tokens-<split>.npy` |
| `add_reference_source.py` | `reference/<split>.npy`, the same 68-channel windows built from the reference clips, annotated with per-source sampling weights |

Window contract: 41 rows by 68 channels, 8 history rows, the current row at index
8, 32 later rows; channels are 6 root-6D, 31 joint positions and 31 joint
velocities, all heading-local to the window centre row and normalized by the
train-only mean and standard deviation.

`prepare.py` stages are resumable and each writes `stage/<name>.json`. Segment
caches are read once per source file, not once per trajectory, because the first
version reopened a file per trajectory and amplified reads 128 times.

`add_reference_source.py` imports the window packing functions from
`prepare.py`, so the two sources cannot drift apart. It refuses a clip whose rate
is not the 50 Hz control rate. The split follows each clip's motion assignment, so
a motion never appears as reference in one split and achieved in another.

Both tools were run once against:

```
/home/fushan/data/mjlab-latent-free-144/data
```

Results and known limitations are recorded in
`docs/reports/x2_joint_pose_dataset_summary.md` and
`docs/reports/x2_joint_pose_trackability_and_modes.md`.
