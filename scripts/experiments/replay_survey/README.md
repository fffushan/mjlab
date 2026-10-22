# Trackability survey and mode analysis

Tools that measure the prepared joint-pose dataset. They answer two questions:

1. Which recorded segments cannot be used as reference motions? (`survey.py`)
2. Which motion modes does the dataset cover, and how far is it from the
   reference it was collected from? (`analyze_modes.py`)

Nothing here modifies the dataset. The purge tools only read.

## Environment

These scripts were developed against the prepared dataset on the shared NAS:

```
/home/fushan/data/mjlab-latent-free-144/data
```

Paths to the dataset, the tennis source clips and the teacher checkpoint are
constants near the top of each file. Change them for another host, or set
`MJLAB_WORKTREE` to point at a different checkout for the collector helpers and
the robot XML.

Run them with the server interpreter directly. Do not use `uv run` there, because
`uv` would sync an environment other jobs share:

```bash
PYTHONPATH=$W/src /home/fushan/mjlab/.venv/bin/python -u survey.py --help
```

`survey.py` imports `collect.py` for the validated outcome classification, so the
collector must be importable from `$MJLAB_WORKTREE/scripts/experiments/joint_pose_collection`.

## survey.py — one trackability pass

Replays every prepared segment as a reference motion and records whether the
frozen teacher can follow it, plus the termination term that fired.

Cost note: building a fresh environment per reference costs about four seconds
regardless of step count, so a naive one-clip-per-run design needs about 55 GPU
hours for 199,879 segments. The driver instead builds one environment per batch
and pins environment *i* to clip *i* through `select_motion_ids`, which brings the
same work to about 66 minutes on four cards.

```bash
python -u survey.py \
  --checkpoint-file <teacher>/model_19999.pt \
  --output-dir /tmp/survey-pass1/lane0 \
  --batch 128 --lanes 4 --lane 0 --seed 500
```

Each batch writes `batchNNNNN.json`; a completed batch is skipped on restart, so a
pass resumes after a crash. Batches are dealt from a length-sorted list, which
keeps each lane's length mix similar and limits idle steps inside a batch.

## failure_ledger.py — accumulate passes

Merges pass results into one per-segment ledger:

```bash
python failure_ledger.py --pass-dir /tmp/survey-pass1 --pass-id pass1 --batch 128
```

`--pass-id` makes the merge idempotent: re-merging the same id replaces that pass
instead of adding it twice, and a second replicate is added as `pass2`.
`/home/fushan/data/mjlab-latent-free-144/failure_ledger.json` holds, for each
segment, `runs`, `failures`, the firing terms, and the observed step counts.

## purge_plan.py — evaluate a purge rule, read only

```bash
python purge_plan.py --min-runs 1 --min-failures 1 --out exclude.json
```

Reports the segments and windows a rule would remove, per group and per split,
which motions would be emptied, and the failure histogram. It writes an exclusion
list only when `--out` is given, and it never touches the dataset.

## analyze_modes.py — mode distribution

Samples segments and rows, computes per-segment behaviour summaries, and reports
three distributions per axis: the reference the teacher was asked to follow, the
achieved state that was recorded, and the failed subset from the ledger.

Body positions come from MuJoCo forward kinematics with the free joint pinned at
the origin, so every number is in the robot's own root frame and independent of
where the robot was standing or facing. The joint-order check prints the mean
absolute difference between achieved and reference joint values at a segment's
first row; a wrong column order shows up as a value near 1 rad instead of 0.05.

```bash
python -u analyze_modes.py --stride-segments 40 --stride-rows 8
```

Axes: root yaw rate, net heading change, heading path length, left and right
shoulder pitch, highest wrist above the pelvis, mean knee flexion, lowest ankle
below the pelvis, foot height difference, torso tilt from vertical, mean joint
speed, acceleration, direction reversals, arm activity asymmetry. Artifacts:
segment length, start phase, group, split.

Four caveats are built into the implementation and must be kept:

- Torso tilt must come from the torso body's rotation matrix. The torso link
  origin sits on the waist axis, so a position difference reports zero for every
  segment.
- Quaternions need double-cover canonicalisation before differencing. `q` and
  `-q` are the same rotation, and a yaw series across a sign flip invents a 2π
  jump.
- The yaw-rate axis measures orientation change per sampled interval, including
  jitter. It is comparable between reference and achieved, but it is not a
  physical turning speed.
- Raw left and right shoulder pitch cannot be compared directly without checking
  the model's mirror sign convention. The overhead-arm claim rests on wrist
  height, which is mirror-free.

## Measured results

Pass 1, recorded 2026-10-22:

| | Count | Share |
|---|---:|---:|
| Segments replayed | 199,879 | 100 % |
| Complete | 179,127 | 89.6 % |
| Failed | 20,752 | 10.4 % |
| Windows affected | 4,136,828 | 10.8 % |

Termination attribution: `ee_body_pos` 17,941, `anchor_pos` 2,809, `anchor_ori` 2.
Failure rises with segment length (2.0 % below 50 rows to 31 % above 400) and with
achieved joint speed (31 % at 1.62 rad/s, 60 % at 1.86, 100 % at 2.0).

Mode analysis, 4,997 sampled segments:

| Axis | Reference p50 | Achieved p50 | Bias |
|---|---:|---:|---:|
| torso tilt from vertical | 21.9° | 13.1° | −8.8° |
| mean knee flexion | 45.1° | 51.3° | +6.2° |
| mean joint speed | 1.49 rad/s | 1.27 rad/s | −14 % |

Absent modes: overhead reach (wrist never above 0.54 m over the pelvis at p95),
straight legs (knee flexion p5 = 42.8°, bins 20–32° empty), very slow motion
(below 0.66 rad/s near zero), late clip phase (start phase p95 = 0.80).

## Limits

- One replicate per segment. A single failed replay cannot separate a defect from
  an unlucky draw, so a purge rule should use at least three passes.
- The survey replays joint-only references. Body kinematics come from the source
  clip at the matching frames, not from the achieved trajectory, so this is not a
  complete-reference replay.
- The mode report measures sampled rows, not every window. Histograms are stable
  at these sample sizes, but per-bin failure rates are not per-window rates.
