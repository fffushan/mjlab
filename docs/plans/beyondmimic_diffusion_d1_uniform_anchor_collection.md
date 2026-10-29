# D1 uniform-anchor collection protocol

Status: plan, written 2026-10-29 17:05 (+08). Not executed.
Owner: diffusion data workstream.
Scope: the D1 offline collection that builds the training store for the X2
flow-matching arms. It replaces the three-start-frame schedule described in
`docs/reports/x2_diffusion_data_collection_review_20261029.md` revision 2.

This plan records the design we settled in discussion. It states what to build,
what to measure, and what we rejected. It does not claim measured results: no
run exists yet.

## 1. Goal

Give every reference frame an equal chance to be the current row of a training
window. The present schedule starts three rollouts per clip, at 10 %, 50 % and
80 % of the clip. That leaves the clip opening, and part of the interior, out of
the dataset. It also caps a trial at 125 kept rows, which no longer matches how
the data will be used.

## 2. Decisions

| # | Decision | It replaces |
|---|---|---|
| D1 | The **anchor** is the current row of a window. Anchors run `0 … N−33`. | Three fixed phase fractions 0.1 / 0.5 / 0.8 |
| D2 | One rollout starts at frame 0 and runs to frame `N−1`. No step cap. | 250 control steps, 125 retained steps |
| D3 | Two initial poses per rollout slot: the reference at frame 0, and `STAND_KEYFRAME`. | One initial pose |
| D4 | 22 seeds, shared by both initial poses. | 43 seeds |
| D5 | The OU perturbation runs on every step of an OU rollout. | First 125 steps only |
| D6 | Eight `DEFAULT_ROW` rows pad every rollout. | No padding |
| D7 | The stability test runs once, at the last row of the rollout. | A test on every step |
| D8 | No standing handoff. The VAE drives the whole rollout. | VAE hands to the recovery ONNX at the clip end |
| D9 | If a rollout fails, retry once without the init perturbation, then drop it. | No retry |

Decision D7 follows the paper: the collection is a stability test, not an
accuracy test. A rollout that deviates in the middle and is correct at the last
row is kept. A rollout that falls is kept only if it recovers.

## 3. The protocol

### 3.1 Schedule

For one clip with `N` frames:

- Anchors are `a = 0 … N−33`. The bound exists because a window needs 32 future
  rows.
- A clip needs `N ≥ 33`. The cohort minimum is 213, so every clip passes.
- Every anchor appears in `2 initial poses × 22 seeds × 2 phases = 88` windows.
- Windows are cut at stride 1 from each rollout. The last 32 rows of a rollout
  are future targets only.

Anchor counts for this cohort: 181 per clip at `N = 213`, 421 at `N = 453`, 285
on average, 41 045 in total.

### 3.2 Rollouts

| Variant | Initial pose |
|---|---|
| A | the reference state at frame 0 |
| B | `STAND_KEYFRAME`: root z at the default, root yaw from the reference, roll and pitch zero, joint positions from `STAND_KEYFRAME`, all velocities zero |

Variant B copies the standing insert that already exists at
`src/mjlab/tasks/tracking/mdp/commands.py:436-441`. It is implemented in the D1
runtime, not by enabling `standing_start_prob`, so the reset stays deterministic.

Both variants then apply the seed perturbation:

| Term | Range | Source |
|---|---|---|
| root x, y | `U(−0.05, 0.05)` m | paper, RSI section |
| root z | `U(−0.01, 0.01)` m | paper, RSI section |
| root roll, pitch | `U(−0.1, 0.1)` rad | paper, RSI section |
| root yaw | `U(−0.2, 0.2)` rad | paper, RSI section |
| joint offsets | `U(−0.01, 0.01)` rad; ankles `U(−0.1, 0.1)` rad | paper, Table S2 |
| velocities | zero | our choice for the first build |

The same seed gives the same perturbation for variant A and variant B, and for
the clean and OU phase of a pair.

A rollout runs from frame 0 to frame `N−1`. All terminations are suspended for
the whole rollout. The env's `time_out` is at 500 steps and the longest clip is
453 frames, so it never fires.

### 3.3 The default row

`DEFAULT_ROW` is computed once, before any rollout.

| Field | Value | Width |
|---|---|---|
| `reference_q` | `STAND_KEYFRAME` joint positions | 31 |
| `reference_dq` | zero | 31 |
| `anchor_orientation_error` | zero | 6 |
| latent | `encode(reference)`, the frozen VAE encoder mean | 32 |
| `root_position` | the reference root position at frame 0 of the same clip | 3 |
| `root_rot6d` | identity | 6 |
| `root_linear_velocity` | zero | 3 |
| `root_angular_velocity` | zero | 3 |
| `body_positions` | forward kinematics of `STAND_KEYFRAME` from an identity root | 60 |
| `body_linear_velocities` | zero | 60 |

`root_position` is copied from the clip so that the fake history does not
introduce a step discontinuity in a world-absolute channel.

Every rollout writes eight of these rows at ticks 0–7 with
`reference_frame = −8 … −1`. Real rows follow at tick `8 … N+7` with
`reference_frame = 0 … N−1`. One layout for every rollout:

```
reference_frame = tick − 8
```

Negative reference frames are allowed: `TrajectoryRow.reference_frame` has a type
check only (`storage.py:236-246`).

The padded rows are history. They are never a current row and never a target.
The index must skip any window whose current row has `reference_frame < 0`.

A segment therefore holds `N+8` rows and yields `N−32` windows: the anchor tick
runs from 8 to `N−25`, because a window needs eight rows behind the anchor and
32 ahead.

### 3.4 The OU perturbation

Unchanged from the current implementation:

```
x_{t+1} = (1 − θ)·x_t + θ·μ + σ·ε_t       θ = 0.8, μ = 0, σ = 0.1, dt = 1.0
executed = clean_action + x_t
```

One change: the noise runs on every step of an OU rollout, not only the first
125. It is applied in normalized action space, after the decoder
(`policies.py:43-87`, `collector.py:516-522`).

Keep `dt = 1.0` for the first build. At `dt = 1.0` the process is nearly white
(lag-1 autocorrelation 0.2). The paper describes a persistent error band and
argues that i.i.d. noise is suppressed by the PD gains. Section 7 gives the
measurement that decides whether `dt` must change.

### 3.5 The test

Two classes of check. Only the first class runs on every step.

**Integrity, every step.** Any hit fails the rollout.

| Check | Reason |
|---|---|
| `nonfinite_post_step` | a NaN poisons every window that touches it |
| `reference_teleport` | the reference frame must advance by one |
| `unexpected_reset` | rows after a reset belong to another episode |
| `missing_terminal_evidence` | bookkeeping |

**Stability, last row only** (`reference_frame == N−1`).

| Check | Limit |
|---|---|
| tilt | ≤ 70° |
| `|anchor_z_error|` | ≤ 0.25 m |
| `|gravity_z_error|` | ≤ 0.8 |
| `|end_effector_z_error|` | ≤ 0.25 m |

The thresholds do not change. Only their evaluation point moves. They are the
pinned values in `QualificationConfig` (`qualification.py:241-247`).

A rollout that passes sets `segment_qualified = True` on every row. A rollout
that fails sets it false on every row, and the index keeps all of them out.

### 3.6 The fallback

A pair is the clean rollout and its OU partner. They share one initial state, so
the fallback is atomic over the pair:

```
attempt 1: perturbed initial pose, clean + OU rollouts
  if both pass the endpoint test: keep both
attempt 2: exact initial pose, clean + OU rollouts
  if both pass: keep both, tag provenance
  else: drop the pair
```

Provenance field on every row: `initial_state_mode = "perturbed" | "reference"`,
and `initial_pose = "reference" | "standing"`.

The OU rollout restores the clean rollout's simulator snapshot, so it cannot run
without its clean partner. This is the current behaviour and it stays.

## 4. Data volume

Computed from the real clip lengths: 144 clips, 45 653 frames, 213–453 frames,
mean 317.

| Quantity | Present | This plan |
|---|---:|---:|
| rollouts | 37 152 | 12 672 |
| store rows | 9 160 010 | 4 118 840 |
| — of which padding | 0 | 101 376 |
| windows | 2 322 881 | 3 611 960 |
| rows per window | 3.9 | 1.14 |
| anchors covered | ~117 per clip, with gaps | all `0 … N−33` |
| windows per anchor | 86 | 88 |
| train token cache | 50.9 GB | 73.7 GiB |

The store shrinks by 55 %. The window count rises by 55 %. Each window now costs
1.14 rows instead of 3.9, because three overlapping 125-row trials no longer
spend 375 rows to cover 117 anchors.

## 5. Code changes

| File | Change |
|---|---|
| `diffusion/contract.py` | Add `padding_steps: int = 8`. Keep `past_steps = 8`, `current_index = 8`, `future_steps = 32`. |
| `diffusion/qualification.py` | Replace `control_steps` / `retained_steps` with `padding_steps` and the endpoint rule. Keep the four thresholds. `observe()` keeps the integrity checks only. `finish()` evaluates the last row. Delete `vae_then_standing`, `invalid_ppo_reference_boundary`, and the `controller == "ppo"` branches. |
| `diffusion/runtime_x2.py` | Replace `PHASE_FRACTIONS` and `start_frames_for` with `anchor_frames_for(N) -> range(0, N−32)`. Replace `reset(seed, start_frame)` with `reset(seed, init_pose, perturb)`. Suspend the reference terminations for the whole rollout. Remove the handoff and the recovery policy from `build_x2_runtime`. Add the standing-init write. Emit the eight `DEFAULT_ROW` rows. |
| `diffusion/policies.py` | Remove the OU step horizon. Keep the recurrence and its parameters. |
| `diffusion/collector.py` | New trial shape: two initial poses, no step cap, endpoint verdict, per-rollout summary, provenance. Keep the clean-snapshot restore path, which the OU rollout still needs. |
| `diffusion/dataset.py` | Skip windows whose current row has `reference_frame < 0`. Allow `ref_start` below zero. Keep the same-segment and monotonic-tick guards. |
| `scripts/diffusion.py` | `collect` gains `--seeds`, `--initial-poses {reference,standing,both}`; drops `--control-steps`. |
| `configs/diffusion/x2_50hz_collection_vgk36.yaml` | Remove `retained_steps` and `control_steps`. Add `seeds: 22`, `initial_poses: [reference, standing]`, `padding_steps: 8`, `perturbation` block, `endpoint_guards` block. |
| `run-bulk.sh` | `FULL_SEEDS=22`. Drop the 150-minute per-worker cap or raise it, because rollouts are longer. |

Nothing changes in `storage.py`: rows already carry `provenance`, and
`reference_frame` accepts negative values.

## 6. Rebuild chain

The present store cannot accept these rows. The chain runs again from the start:

```
collect (12 672 rollouts) → merge → inspect → replay
  → build (window index + split) → stats → build-token-cache → validate
```

The split is still keyed on the group `motion:seed:anchor`. Both initial poses
share that group, so they never straddle a split. The clean/OU pair identity
carries the initial pose, `motion:seed:anchor:pose`, because the collector keys
its snapshot map on `(pair_id, initial_state_id)` (`collector.py:450`) and its
"clean partner finished" map on `pair_id` (`collector.py:371, 392`, used by
`_admissible`). Sharing the pair identity across poses would let the OU rollout
of one pose be admitted against the clean rollout of the other.

Because the assignment is a permutation of the sorted group roots under
`PCG64(42)` (`dataset.py:212-214`), any change to the group set reshuffles every
group. The index load fails the `assignments_sha256` check otherwise.

## 7. Measurements to take in the pilot

Run the 4-seed pilot first. These three numbers decide three open knobs.

| Measurement | Decides |
|---|---|
| State spread of the OU rollout against the clean rollout, per joint and per pelvis height, over the first 100 steps | whether `ou_dt = 1.0` produces a real error band, or must become `0.02` s |
| Fraction of variant-B rollouts that pass the endpoint test | whether the standing initial pose is worth its cost |
| Fraction of pairs that need the fallback | whether the retry is worth keeping |

Also record per rollout: `min_tilt`, `max|anchor_z_error|`, `max|ee_z_error|`,
`fall_frames` (count of rows with tilt above 70°), and the pass or fail verdict.
The dataset can then be filtered later without a re-simulation.

## 8. Acceptance checks

1. Every clip contributes anchors `0 … N−33`, with no gap.
2. No indexed window has a padded current row.
3. Window count is 3 611 960, minus any rollouts dropped by the endpoint test.
4. Each indexed segment reports exactly `N−32` windows, its full anchor range.
5. `rows − windows == 32 × rollouts + padding_rows`, that is
   `4 118 840 − 3 611 960 == 405 504 + 101 376`.
6. The split holds out groups, not rows, and both initial poses of a clip land
   in the same split.
7. The token-cache validation stage passes.

## 9. Risks

| Risk | Status |
|---|---|
| The VAE cannot stand up from a fall, so "fall and recover" will be rare | Accepted. The owner expects it cannot. Keep the rows if it happens. |
| Rows recorded while the robot is prone condition the model on a state the deployment rarely sees | Accepted and now measurable through the per-rollout summary. |
| A prone robot under a position-tracking policy stresses the contact solver | Accepted. `nonfinite` catches NaN only. Log peak velocity. |
| The OU process may be nearly white at `dt = 1.0`, which is the regime the paper rejects | Open. Section 7 gives the measurement. |
| The train cache grows to 73.7 GiB | Within the 128 GiB cap the driver passes. |
| The standing rollout starts 0.42–1.42 rad from the reference (median 0.71), and 0 of 144 clips start near standing | Accepted. The endpoint test does not look at the start. |

## 10. Rejected options

| Option | Reason |
|---|---|
| One rollout per anchor | 625 853 906 rows and 67× the present simulation. It adds histories, not anchors. |
| Keep 43 seeds with two initial poses | 7 852 316 rows and a 144 GiB cache, above the cap. |
| Keep `retained_steps` and add a `qualified_until` cut index | The owner accepts the failure cost. A boolean is simpler. |
| Perturb the latent instead of the action | The stored latent is `encode(reference)` and must stay exact. Noise belongs after the decoder. |
| A guard-free warm-up of k frames for the standing start | The endpoint test makes it unnecessary. |
| Keep the standing handoff | No data above the clip end is wanted. |

## 11. Effect on existing documents

`docs/reports/x2_diffusion_data_collection_review_20261029.md` revision 2
describes the three-fraction protocol and its coverage numbers. Sections 3 to 6
and the ledger stop being true after this change. The note needs a revision 3
after the rebuild, with the new counts. Do not edit it before the run, because
its present numbers are the measured baseline for the comparison.

## 12. Evidence

Source read for this plan:

```text
src/mjlab/tasks/tracking/diffusion/runtime_x2.py     55, 90-97, 339, 394-441, 469, 478-514, 518-534, 700
src/mjlab/tasks/tracking/diffusion/policies.py       43-87, 90-150, 153-164
src/mjlab/tasks/tracking/diffusion/collector.py      66-75, 316-345, 440-530, 595-601
src/mjlab/tasks/tracking/diffusion/qualification.py  238-247, 300-386, 602-635
src/mjlab/tasks/tracking/diffusion/dataset.py        139-153, 200-228, 314-339, 878, 916
src/mjlab/tasks/tracking/diffusion/contract.py       126-142
src/mjlab/tasks/tracking/mdp/commands.py             416-441, 761-776
src/mjlab/tasks/tracking/tracking_env_cfg.py         430-451, 486
src/mjlab/asset_zoo/robots/agibot_x2/x2_constants.py 231-300
configs/diffusion/x2_50hz_collection_vgk36.yaml
docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk36_contract.yaml  184-187, 233-236
logs/diagnostics/preview-flow-port-20261020/multimotion144-bulk/run-bulk.sh  11-14
```

Paper: `~/Documents/papers/beyondmimic.txt`, "Diffusion Dataset Collection"
(supplementary S6) and the domain-randomization and RSI-passages that give the
perturbation ranges.

Clip lengths: `data/tennis/*.npz` for the 144 ids in
`configs/distillation/x2_tennis_multimotion_v1.yaml`. Sum 45 653 frames, range
213–453.
