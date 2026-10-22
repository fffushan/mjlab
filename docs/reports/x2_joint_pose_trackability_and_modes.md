# X2 joint-pose trackability and mode measurement

Status: pass 1 measured; one policy decision open. No data has been removed.
Observed: 2026-10-22 (+08).
Tooling: `scripts/experiments/replay_survey/README.md`.
Dataset: `x2_joint_pose_dataset_summary.md`.

## Purpose

The prepared joint-pose dataset holds achieved motion. This report records two
measurements of it:

1. **Trackability.** Which recorded segments can act as reference motions for the
   frozen teacher, and which termination term fires when they cannot.
2. **Mode coverage.** Which motion modes the dataset contains, how the achieved
   motion differs from the reference it came from, and where failures concentrate.

Both are needed before a purge rule or a training run is chosen.

## Method

Each recorded segment is written out as a joint-overlay reference: its achieved
`q` and `dq`, with the source clip's body kinematics at the frames the episode
visited. Body rows are clamped at the clip's last frame, because the command
clamps there too.

The teacher then replays each reference once from frame 0 in the collection
environment: same task, same observation corruption, no pushes, 128 references
resident per batch with one environment pinned to each.

Environment build dominates a naive design: one fresh environment per reference
costs about 4 s regardless of the step count, so 199,879 separate runs need about
55 GPU hours. Pinning one environment per reference in a 128-clip library brings
the same coverage to 66 minutes on four cards. That is the only reason full
coverage is affordable.

## Result: pass 1 trackability

| | Count | Share |
|---|---:|---:|
| Segments replayed | 199,879 | 100 % |
| Complete | 179,127 | 89.6 % |
| Failed | 20,752 | 10.4 % |
| Windows affected | 4,136,828 | 10.8 % of 38,278,319 |

| Termination term | Count | Share of failures |
|---|---:|---:|
| `ee_body_pos` | 17,941 | 86.5 % |
| `anchor_pos` | 2,809 | 13.5 % |
| `anchor_ori` | 2 | 0.01 % |

The dominant term is end-effector height: a wrist or ankle link deviating 0.25 m
in z. Torso tilt is effectively absent, so these are not falls.

| Group | Segments | Failed | Rate |
|---|---:|---:|---:|
| start/baseline | 73,700 | 5,803 | 7.9 % |
| uniform/baseline | 64,012 | 7,436 | 11.6 % |
| uniform/disturbed | 62,167 | 7,513 | 12.1 % |

Failure rises with segment length and with achieved joint speed:

| Bin | Failure rate |
|---|---:|
| under 50 rows | 2.0 % |
| 100–199 rows | 12.3 % |
| over 400 rows | 31.4 % |
| joint speed 1.62–1.74 rad/s | 31 % |
| 1.86–1.98 rad/s | 60 % |
| knee flexion 63–67° | 53 % |
| foot height difference above 0.15 m | 46 % |

The failing corner is fast, deep-kneed, single-support motion. That is consistent
with an ankle-height termination and inconsistent with balance collapse.

All 144 motions are affected. The heaviest are `single_106` (710 of 1,401
segments), `single_049` (634 of 1,371) and `single_059` (632 of 1,394). No motion
would lose every segment under the aggressive rule.

## Result: mode coverage

Sample: 4,997 segments, every 46th control row, 230k forward-kinematics states.

### Recorded motion is biased away from the reference

| Axis | Reference p50 | Achieved p50 | Bias |
|---|---:|---:|---:|
| torso tilt from vertical | 21.9° | 13.1° | −8.8° |
| mean knee flexion | 45.1° | 51.3° | +6.2° |
| mean joint speed | 1.49 rad/s | 1.27 rad/s | −14 % |

All three biases point the same way: the achieved motion is more upright, more
crouched and slower than what the teacher was asked to follow. The dataset is
therefore not a neutral sample of the reference distribution. A generative model
trained on it reproduces the teacher's posture, not the demonstrated motion.

### Four modes are nearly absent

| Missing mode | Evidence |
|---|---|
| Overhead arm reach | highest wrist 0.54 m above the pelvis at p95, never higher |
| Straight legs | knee flexion p5 = 42.8°; the 20–32° bins are empty |
| Very slow motion | joint speed below 0.66 rad/s is near zero share |
| Late clip phase | start phase p95 = 0.80 |

Turning is covered: net heading change spans −116° to +96° at p5/p95, so left and
right turns are both present. Arm use is balanced: activity asymmetry median
+0.05, p5/p95 −0.08/+0.24.

The overhead gap matters most for this dataset's purpose. Tennis contains serves,
smashes and high volleys, and this dataset cannot generate any of them.

## Consequences

| Decision | Evidence |
|---|---|
| Do not purge on one replicate | one failed draw cannot separate a defect from an unlucky draw |
| Do not purge only the failing corner without replacement | that corner is the only fast, deep, single-support motion in the data |
| Record more reference data for the missing modes | overhead reach, straight legs, slow motion |
| Carry the three biases into any training decision | they are properties of the teacher and will be inherited |

## Limits

1. One replicate per segment. Only `0/1` and `1/1` histograms exist so far.
2. The replay uses joint-only references. Body kinematics come from the source
   clip at matching frames, not from the achieved trajectory.
3. Body positions are reported in the robot root frame, so absolute ground height
   is not measured; the achieved dataset has no root translation.
4. The mode sample covers 2.5 % of segments. Histograms are stable at that size;
   per-bin failure rates are indicative, not exact.
5. `failure` remains a termination label, not a fall detector.
6. The passing rate of 89.6 % is a property of this teacher, this environment and
   this reference construction. It is not a general statement about the dataset.
