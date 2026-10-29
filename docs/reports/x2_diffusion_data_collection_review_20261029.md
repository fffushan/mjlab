# X2 diffusion data collection: what the 144-clip dataset contains

Status: review note, written 2026-10-29 12:10 (+08).
Owner: diffusion data workstream. Scope: the D1 collection procedure and the
built 144-clip dataset that trained the flow-matching arms.

This note describes the procedure and the stored data. It records numbers, not
plans. Section 8 corrects two statements that this workstream made earlier.

## 1 Purpose

An operator asked whether the dataset holds data that starts at the first
reference frame of a clip. This note answers that question and records the rest
of the procedure, so that the operator can plan a schedule change against facts.

## 2 What produces the data

Three policy artifacts appear in this pipeline. They have different roles.

| Role | Artifact | When it acts |
|---|---|---|
| Tracking **teacher** | multi-motion PPO checkpoint `model_19999.pt` | Upstream only, during distillation |
| Tracking **student** | frozen VAE, `contract.vae_checkpoint` | During D1 collection, every clean step |
| **Recovery** policy | `x2-tennis-recovery-n25.onnx` | During D1 collection, after a physical fall |

The D1 rollout uses the VAE, and not the PPO teacher. `build_x2_runtime`
constructs `FrozenVaePolicy.from_checkpoint` for tracking and
`FrozenPpoRecoveryPolicy.from_onnx` for recovery, at
`runtime_x2.py:1320-1348`. The tracking teacher supplies one input only: its
ONNX metadata gives the action contract, and the code reads no policy weights
from it, at `runtime_x2.py:1230-1231`.

`FrozenVaePolicy.latent_and_action` computes one step as
`mu = encode(reference[68])` and `action = decode(mu, conditioning[99])`
(`adapter.py:845-863`). The 32-value `mu` becomes the latent channel of the
stored row.

After a physical fall the recovery policy takes control, so the trial reaches
its full length. Those rows record `controller: "ppo"`. They never enter the
window index, because `_row_can_be_indexed` requires `controller == "vae"`
(`dataset.py:136-152`).

## 3 Trial schedule

| Item | Value |
|---|---|
| Clips | 144, from 213 to 453 frames, mean 317 |
| Start frames per clip | 3, at fractions 0.1, 0.5, 0.8 |
| Start frame formula | `floor(fraction x (N - 1))`, `runtime_x2.py:90-97` |
| Start frame values | 153 distinct values, from 21 to 361 |
| Seeds per start | 43, numbered 0 to 42 |
| Trial types | clean, and OU with `theta 0.8`, `sigma 0.1` |
| Planned groups | 18 576 |
| Indexed groups | 18 280 |
| Planned trials | 37 152 |
| Indexed segments | 36 287 (97.7 %) |
| Trials without a window | 865 (2.33 %) |
| Control steps per trial | 250 |
| Retained steps | 125 |
| Workers, environments | 8 processes, 64 environments each |
| Members per worker | 18 |

The OU noise applies to the clean action for the first 125 steps of an OU trial
(`collector.py:516-521`). A trial stops early when the reference clip ends.
The collection config is `configs/diffusion/x2_50hz_collection_vgk36.yaml` with
`retained_steps: 125` and `control_steps: 250`.

## 4 Stored volume and splits

The index holds 2 322 881 windows.

| Split | Windows | Groups | Segments |
|---|---:|---:|---:|
| Train | 1 860 237 | 14 620 | 29 032 |
| Validation | 231 470 | 1 833 | 3 641 |
| Test | 231 174 | 1 827 | 3 614 |
| All | 2 322 881 | 18 280 | 36 287 |

The store holds 9 160 010 rows in 2 241 shards. A trial writes 250 rows, so
the store holds about 36 640 trials. That value is an estimate, because the
division is not exact. Only 3 774 361 rows are indexable, which is the window
count plus 40 rows for each segment. The other rows belong to the recovery
policy, to a reset, or to a reference boundary. The split key is the group, and
a group is `motion:seed:start_frame`. The fractions are 0.8, 0.1 and 0.1 with
seed 42.

The train token cache is 50 948 171 084 bytes with shape
`(1860237, 41, 167)` in float32. One token holds 135 state values and 32 latent
values.

## 5 Window layout

One window holds 41 rows. Rows 0 to 7 are history, row 8 is the current row,
and rows 9 to 40 are the predicted future (`contract.py:134-140`). The stride
is 1. A segment of length 125 frames yields 85 windows, and the index confirms
this count.

`ref_start` spans 0 to 84. Every segment contributes exactly one window at
`ref_start` 0, so 36 287 windows begin at the first recorded frame of a
segment.

## 6 Reference-frame coverage

Coverage is the number of distinct clip frames that any window covers. I
computed it from `ref_start`, the segment length, and the clip length.

| Quantity | Value |
|---|---:|
| Reference frames in all clips | 45 653 |
| Frames covered | 39 924 (87.45 %) |
| Frames in the clip opening | 4 490 uncovered (9.84 %) |
| Frames in interior gaps | 1 239 uncovered (2.71 %) |
| Coverage per clip | 75.1 % to 89.7 %, median 88.5 % |
| First covered frame | 21 for the shortest clip, 45 for the longest |

Each clip has three bands. Band 3 starts at fraction 0.8, so it keeps about
0.2 x N frames. The median band-3 length is 64 frames. Band 1 and band 2 hold
125 frames when the trial survives.

Two interior gaps remain in a long clip. Clip `single_000` misses frames 169 to
225 and frames 350 to 360.

The opening is 0.42 s to 0.90 s of each clip at 50 Hz. No window holds these
frames, as history or as a target.

## 7 Findings

1. The rollout policy is the frozen VAE. The PPO tracking teacher does not act
   during collection.
2. The dataset has no clip opening. The earliest reference frame in any window
   is 21, and the earliest predicted frame is 30.
3. The phase schedule causes the gap at the opening. The 8-frame history does
   not cause it, because a segment records its own history from step 0.
4. History dropout and the root mask change no coverage property, because both
   act on the loss and not on the index.
5. The OU perturbation does not reduce survival. 65.3 % of clean segments and
   65.1 % of OU segments reach 125 steps. Both values include the band-3
   clip-end truncation.
6. Of the 12 623 segments shorter than 125 frames, 12 132 are band-3 segments
   that stop at the clip end. The other 491 are band-2 segments with a median
   length of 117.
7. 865 of the 37 152 planned trials (2.33 %) produce no window. This count
   mixes two cases: the store has no rows for the trial, or the trial has rows
   but no qualified window. I did not separate them. A fall inside the first 41
   steps is a candidate for the second case.
8. Validation is not motion-disjoint. A group separates a start frame and a
   seed, not a clip. The same clip appears in training, validation and test at
   different start frames. This is one reason why validation loss does not
   rank closed-loop survival.

## 8 Corrections to earlier statements

This workstream made two statements in conversation on 2026-10-29. Both were
wrong. This section records them, so that the record is not silent about them.

1. "One frozen PPO tracking policy plays each clip." The rollout policy is the
   frozen VAE. The PPO checkpoint acted only during distillation.
2. "A third of the trials stop early, because of a fall." The 34.8 % short
   figure is mostly clip-end truncation in band 3, which holds 12 132 of the
   12 623 short segments. The measured early-stop rate is 491 segments, plus
   865 trials with no window.

## 9 Limits

- I ran no new collection for this note. Every number comes from one store,
  built on 2026-10-20, and from the source at commit
  `fxy/test/flow-matching-baseline`.
- The coverage number assumes a band ends at its last indexed window. The store
  keeps the rows after a fall, and the index keeps them out by design.
- I did not measure how the VAE survival rate depends on the start frame beyond
  the band counts in item 6.
- The cause of the 865 unindexed trials is unknown.
- No approved ASD-STE100 word list is available in this environment, so I
  applied the sentence and voice rules but cannot claim word-level compliance.

## 10 Next actions

The operator asked for data at the opening. Three schedule changes give it.
Each start frame costs 86 trials per clip, which is 43 seeds times two
phases. The present data hold 37 152 trials and 3 774 361 indexable rows.

| Option | Change | Extra trials | Extra indexable rows | Share of present rows |
|---|---|---:|---:|---:|
| 1 | Add fraction 0.0 | 12 384 | 1 548 000 | 41 % |
| 2 | Add fractions 0.0, 0.05 and 0.95 | 37 152 | 3 310 484 | 88 % |
| 3 | Replace the fractions with a stride of 8 frames | 397 664 | 44 886 496 | 1 189 % |

I computed these values per clip from its length. A band adds `min(125, N -
start)` indexable rows. Option 3 has 35.1 starts per clip instead of 3.

The model gives 3 892 446 rows for the present schedule, which is 3.1 % above
the measured 3 774 361. Treat the extra rows as estimates against the measured
baseline.

Option 1 closes the opening and keeps the present bands. Option 3 closes every
gap, at about 12 times the present row count.

A new collection needs the full chain again: inspect, replay, merge, window
index, statistics, token cache. The present store cannot take extra trials
without a rebuild. Do not merge a partial store.

The start frame 0 also needs a decision on one point. At frame 0 the reference
speed is near zero for a motion that begins at rest, so the first frames test
the model at a low speed that the present data rarely show. That is a data
change, and it needs a measurement, not an assumption.

## 11 Evidence

Index and store:

```text
/home/fushan/data/mjlab-144/data/offline/window_index.npz
/home/fushan/data/mjlab-144/data/store/            # 2 241 shards
```

Collection driver, remote copy and local copy:

```text
/home/fushan/mjlab-multimotion144-bulk-20261020/logs/diffusion/
  x2-multimotion144-preview-bulk-20261020/control/run-bulk.sh
logs/diagnostics/preview-flow-port-20261020/multimotion144-bulk/run-bulk.sh
```

Source read for this note:

```text
src/mjlab/tasks/tracking/diffusion/runtime_x2.py    lines 55, 90-97, 1242-1400
src/mjlab/tasks/tracking/diffusion/adapter.py       lines 683-863
src/mjlab/tasks/tracking/diffusion/collector.py     lines 505-620
src/mjlab/tasks/tracking/diffusion/dataset.py       lines 136-152, 976-999
src/mjlab/tasks/tracking/diffusion/contract.py      lines 134-140
configs/diffusion/x2_50hz_collection_vgk36.yaml
```

Count commands used these index fields: `ref_start`, `ref_segment`,
`ref_length`, `ref_split`, `ref_group`, `segment_ids`, `group_keys`. The
segment id carries the clip, seed, start frame and phase, for example
`x2-multimotion144-preview-bulk-20261020:single_000:13:226:clean:vae`. The store
metadata row carries `tick`, `reference_frame`, `controller` and
`terminal_evidence`.
