# X2 shared teacher versus the eight vendor-gain specialists

Date: 2026-10-15 (+08).
Status: measurement complete.
Scope: simulation only, in-distribution, root-relative tracking.

## 1. Question

Does one shared multi-motion PPO teacher match the eight per-clip specialists `vgk000` to `vgk007`?

Answer: **yes, on root-relative tracking.**
The teacher is level with the specialists on root-relative keypoint error and joint velocity error.
It also has fewer physical failures.

## 2. What was compared

The eight specialists come from `configs/distillation/x2_tennis_vendor_gains_dr_waist_kd4_v1.yaml`.
Each was trained for 30,000 iterations on one clip.

| Member | Clip |
| --- | --- |
| `vgk000` to `vgk007` | `single_000` to `single_007` |

The teacher is `model_19999.pt` from the [20,000-iteration run](x2_multi_motion_ppo_8gpu_20k_resume_result.md).

### Matched protocol

Both arms used the same evaluation settings:

| Setting | Value |
| --- | --- |
| Clip start | first frame (`sampling_mode="start"`) |
| Episode budget | the full clip duration |
| Episodes | `256` |
| Seed | `0` |
| Actor corruption | enabled |
| `push_robot` event | removed |
| Lookahead | restored from each checkpoint's own saved parameters |

The two arms could not share one environment, because the tasks fix different observation widths.
`r_mpkpe` and `jvel` come from the same helper functions in both paths, so those two are directly comparable.

Both arms are no-state-estimation policies, and their actor observation terms are identical apart from one added future-reference term. The teacher also has a wider network. Section 6 gives the exact difference.

## 3. Results

Values are centimetres for `r_mpkpe` and radians per second for `jvel`.
`d_r` and `d_j` are teacher minus specialist, so a negative value favours the teacher.

| Clip | Frames | Teacher complete | Teacher `r_mpkpe` | Specialist `r_mpkpe` | `d_r` | Teacher `jvel` | Specialist `jvel` | `d_j` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_000` | 453 | 1.000 | 0.0460 | 0.0442 | +0.0018 | 1.0360 | 0.9782 | +0.0578 |
| `single_001` | 340 | 1.000 | 0.0573 | 0.0606 | -0.0033 | 1.3646 | 1.3849 | -0.0203 |
| `single_002` | 320 | 1.000 | 0.0547 | 0.0597 | -0.0050 | 1.4108 | 1.4547 | -0.0439 |
| `single_003` | 382 | 1.000 | 0.0564 | 0.0563 | +0.0001 | 1.3106 | 1.3526 | -0.0420 |
| `single_004` | 354 | 1.000 | 0.0499 | 0.0534 | -0.0035 | 1.1559 | 1.1994 | -0.0435 |
| `single_005` | 362 | 1.000 | 0.0444 | 0.0443 | +0.0001 | 0.9770 | 0.9643 | +0.0127 |
| `single_006` | 331 | 1.000 | 0.0454 | 0.0448 | +0.0006 | 0.9664 | 1.0167 | -0.0503 |
| `single_007` | 312 | 1.000 | 0.0454 | 0.0435 | +0.0019 | 0.9596 | 0.9595 | +0.0001 |

### Summary

| Measure | Teacher | Specialists | Favours |
| --- | ---: | ---: | --- |
| Clips with zero failures | `8 of 8` | `7 of 8` | teacher |
| Physical failures, 2048 episodes | `0` | `5` (all on `single_002`) | teacher |
| Mean `r_mpkpe`, centimetres | `0.04993` | `0.05085` | teacher, by `1.8 %` |
| Mean `jvel`, radians per second | `1.14760` | `1.16379` | teacher, by `1.4 %` |
| Clips with lower `r_mpkpe` | `3` | `5` | specialists |
| Clips with lower `jvel` | `5` | `3` | teacher |

Per-clip differences are small: `r_mpkpe` differs by at most `5 mm`, and `jvel` by at most `0.058 rad/s`.
The teacher is not uniformly better on `r_mpkpe`; it wins on 3 clips and loses on 5.

## 4. Integrity checks

| Check | Result |
| --- | --- |
| Clips with both arms present | `8 of 8` |
| Same `steps_taken` in both arms per clip | yes: `453, 340, 320, 382, 354, 362, 331, 312` |
| Teacher episodes | `256` per clip |
| Specialist episodes | `256` per clip |
| Process exit status | `0` for all 16 runs |
| Teacher completion fraction | `1.000` on all 8 clips |

The equal `steps_taken` values show that both arms ran the whole clip budget, so an arm that stops early cannot bias the comparison.

## 5. Sample budget

The specialists and the teacher differ greatly in how much data each clip received.

| Arm | Environment steps | Per clip |
| --- | ---: | ---: |
| Specialist | `8192 x 24 x 30000` = `5.90e9` | `5.90e9` (one clip) |
| Teacher | `65536 x 24 x 19101` = `3.00e10` | about `2.03e8` (148 clips) |

The teacher therefore trained on about **`3.4 %`** of the per-clip sample budget of one specialist and reached the same root-relative tracking level.

This is an arithmetic comparison of budgets. It is not a sample-efficiency measurement, because the two arms differ in task, network inputs, and objective.

## 6. Limits

- **Nearly the same observation contract.** The actor terms are identical in both arms: `command`, `motion_lookahead`, `motion_anchor_ori_b`, `base_ang_vel`, `joint_pos`, `joint_vel`, `actions`. Neither actor has a root linear velocity or a root position term, so both are no-state-estimation policies. The one observation difference is the future reference: the teacher uses `lookahead_s = 0.2`, which adds a 62-wide future joint position and velocity sample to the actor **and** the critic, while the specialists use `lookahead_s = 0.0`. The teacher actor is 226-wide against 164, and its critic 412 against 350.
- **Wider network.** The first layer of the teacher is 1024 wide; the first layer of the specialist is 512 wide.
- **The teacher therefore competes with an advantage.** It has strictly more information and more capacity than the specialists, and it still only ties on root-relative tracking. Parity here does **not** show that a preview-free general teacher would also match a specialist.
- **Different episode rule.** The specialist task wraps the reference and ends on the time limit. The multi-motion task ends at the clip endpoint. The comparable event is the clip end without a physical failure, and the boundary differs by one step.
- **Both arms are in distribution.** Each specialist is evaluated on its own training clip, and the teacher has all 148 clips in its training set. This is **not** a held-out generalisation test.
- **`r_mpkpe` is root-relative.** Global root translation is excluded. The weakest aggregate term of the teacher is global root position at `0.465` of its maximum, and **this comparison does not measure that term**. The teacher path also does not report absolute MPKPE, so only the specialist column has it.
- **No noise floor.** No arm was repeated, so a `5 mm` difference cannot be separated from run-to-run variation. The failure-count difference, `5 of 256` on one clip, is more likely to be real than the mean differences.
- The `5` specialist failures all occur on one clip, `single_002`. That may be one difficult episode regime rather than a general defect.
- Simulation only. No physical or hardware claim.

## 7. What this changes

The shared teacher is a usable replacement for a bank of per-clip specialists for root-relative tracking.
The earlier concern that a shared teacher would be clearly worse per clip is not supported for these eight clips.

It does not close the root-position gap. That remains the weakest term and needs its own measurement.

## 8. Next actions

1. Add absolute MPKPE, and end-effector error, to the multi-motion evaluation path, so the two arms can be compared on global root position. This needs a small change to `scripts/evaluate_tracking_policy.py` and a focused test.
2. Repeat one clip per arm to obtain a noise floor, then re-read the per-clip differences.
3. Keep the root-position weakness as an open item; see the reward-ceiling table in the [20k report](x2_multi_motion_ppo_8gpu_20k_resume_result.md).

No new training run is authorized by this report.

## 9. Artifacts

- [Comparison CSV](../../logs/diagnostics/x2-permotion-vgk8-comparison.csv)
- [Comparison summary](../../logs/diagnostics/x2-permotion-vgk8-summary.json)
- [Comparison script](../../logs/diagnostics/x2-permotion-vgk8/compare.py)
- Teacher reports: [`logs/diagnostics/x2-permotion-vgk8/teacher/`](../../logs/diagnostics/x2-permotion-vgk8/teacher/)
- Specialist reports: [`logs/diagnostics/x2-permotion-vgk8/specialist/`](../../logs/diagnostics/x2-permotion-vgk8/specialist/)
- Driver script on the container: `logs/diagnostics/mm-permotion-vgk8-20261015-driver.sh`

The container copy of the raw results is at `/home/fushan/mjlab/logs/diagnostics/mm-permotion-vgk8-20261015/`.

Related records: [20k teacher result](x2_multi_motion_ppo_8gpu_20k_resume_result.md), [NCCL hang diagnosis](x2_multi_motion_ppo_8gpu_nccl_hang_diagnosis.md), and [plan](../plans/x2_multi_motion_ppo_teacher.md).
