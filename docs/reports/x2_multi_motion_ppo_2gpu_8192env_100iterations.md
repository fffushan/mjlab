# X2 shared PPO teacher: two-GPU 8192-environment test

Date: 2026-10-14 (+08).
Execution status: complete, exit code `0`.
Tracking-quality status: not qualified.
Scope: one authorized two-GPU smoke test on a managed container.

## 1. Configuration

The operator requests two GPUs, 8192 environments per GPU, and 100 iterations.
The test uses all 148 tennis references, the selected six-layer network, and the existing `0.2 s` preview.
It does not change reward weights, physics limits, motion sampling, or robot gains.

| Item | Value |
| --- | --- |
| Host | `10.14.64.37:31505`, container `d-20260923051532-ntk4g` |
| GPUs | Physical 3 and 4, NVIDIA RTX 4090, 23028 MiB each |
| GPU choice | GPUs 0, 1 and 2 carried a concurrent D3 closed-loop evaluation and were not used |
| Task | `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion` |
| References | `148` clips: 144 single-motion and 4 standing |
| Environments | `8192` per rank; `16384` global |
| Allocation | 55-56 environments per clip on each rank |
| Network | Actor and critic layers `(1024, 1024, 1024, 1024, 512, 128)`; inputs 226 and 412 |
| Training | From scratch, seed `7` on rank 0 and `8` on rank 1, 100 iterations, 24 steps per iteration |
| Global transitions | `393216` per update; `39321600` total |
| Optimizer steps per rank | `2000` |
| Logging | TensorBoard on rank 0 only; model upload disabled |
| Checkpoint interval | 25 iterations |

Source deployment was necessary because the remote checkout did not contain the multi-motion code.
Five source files, four `standing_*.npz` files, and the 148-clip manifest were transferred.
Every transferred file hash matches the local originals.
The remote had no local edits in the deployed paths.
Original files are copied to `source-backup/` under the artifact directory.

## 2. Execution verification

| Check | Result |
| --- | --- |
| Exit status | `0` |
| Torchrunx message | `All workers completed successfully.` |
| Rank 0 | `device=cuda:0`, `seed=7`, `rank=0` |
| Rank 1 | `device=cuda:1`, `seed=8`, `rank=1` |
| Reference count per rank | 148 on both ranks |
| Final checkpoint | `model_99.pt` |
| Checkpoint iteration | `99` |
| Common step counter | `2400` |
| Optimizer step values | `{2000}` |
| Actor and critic parameters | Finite |
| Logged scalars | Finite |
| Recorded environments per rank | `8192` |
| Recorded preview | `0.2 s` |
| Owned processes after the run | None |
| Owned tmux session after the run | Absent |
| GPU state after the run | All eight GPUs idle |

The pre-existing team tmux session `queue-dispatch` was not created or changed by this test.

## 3. Curve comparison

![Comparison curves](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/comparison-curves.png)

The local reference run is the [single-GPU 4096-environment test](x2_multi_motion_ppo_all_tennis_100_iterations.md).
Both runs use 148 clips, 100 iterations, seed 7, and the same network and preview.
The local run used one RTX 5080 Laptop GPU; this run used two RTX 4090 GPUs.

Values below are means of the first and last 10 iterations.

| Logged scalar | Local 1 x 4096, first 10 | Local 1 x 4096, last 10 | 2 x 8192, first 10 | 2 x 8192, last 10 |
| --- | ---: | ---: | ---: | ---: |
| Mean episode return | `-1.8664690971374511` | `-0.12035190686583519` | `-1.763521659374237` | `0.23648756444454194` |
| Mean episode length, seconds | `0.3820200004577637` | `0.16377999687194822` | `0.3784600067138672` | `0.3375800018310547` |
| `ee_body_pos` terminations per step | `185.40833740234376` | `482.71668395996096` | `368.8333465576172` | `432.58750915527344` |
| `clip_complete` per step | `19.86250057220459` | `12.275000286102294` | `39.900001525878906` | `48.49583435058594` |
| Joint position error, radians | `1.9619418621063232` | `1.9246860027313233` | `1.9683610320091247` | `1.7371085047721864` |
| Joint velocity error, radians per second | `21.992645263671875` | `23.795628547668457` | `21.896096801757814` | `18.641747856140135` |

The two runs agree closely for the first 20 iterations.
They then diverge in every logged scalar.

The episode-length curves explain the difference:

- Both runs fall to about `0.20 s` by iteration 40.
- The local run continues to fall and ends at `0.16 s`.
- The two-GPU run reverses direction after about iteration 65 and ends at `0.34 s`.

The `ee_body_pos` term fires on 780 of 8192 rows per step at iteration 55 in the two-GPU run, and then falls to 432.
The local run rises to about 490 and stays there.
The two-GPU run therefore ends lower on this term, and it completes about four times as many clips per step.

### Interpretation and limits

The two-GPU run is better on every logged measure.
This is a **descriptive** result about the recorded curves, not a tracking-quality result.

The test changed more than one variable. Environment count changed from 4096 to 16384 in total, GPU count changed from one to two, and the hardware changed.
The measurement cannot separate the effect of the larger batch from the effect of the second GPU.
It is not a scaling-efficiency measurement.

Further limits:

- The curves come from rank 0 only.
  They describe the 8192 environments of one rank, not the combined 16384.
- The mean episode length is about `0.34 s`, or 17 control steps.
  The episode budget is 10 s, or 500 control steps.
  Most episodes still end early, so the result does not demonstrate sustained tracking.
- Short episodes weight the initial transient.
  The joint-error metrics therefore do not measure steady-state tracking.
- A mean joint position error near `1.74 rad` remains large.
- The test uses one seed and no per-motion or held-out evaluation.
- The local reference differs in hardware, so its timing is not comparable.

No cause for the episode-length reversal is established.
Larger batches give less noisy advantage estimates, which is one possible cause.
This is a hypothesis, and the test does not test it.

## 4. Runtime

| Measurement | Value |
| --- | --- |
| Start | `2026-10-14 18:07:18 (+08)` |
| End | `2026-10-14 18:12:59 (+08)` |
| Wall-clock service time | about `341 s` |
| Rank 0 reported training-loop time | `0:04:14`, or `254 s` |
| First reported iteration time | `4.40 s` |
| Last reported iteration time | `2.53 s` |
| Local reference, 100 iterations at 4096 environments | `191 s`, about `1.9 s` per iteration |

The two-GPU run processed about four times the transitions per update in about 1.3 times the wall-clock time.
The hardware differs, so this ratio is not a multi-GPU speedup measurement.
Total GPU-memory peak was not measured.

## 5. Artifacts

Root: [`logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/`](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/).

- [Comparison plot](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/comparison-curves.png)
- [Comparison CSV](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/comparison-metrics.csv)
- [Comparison summary](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/comparison-summary.json)
- [Comparison script](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/compare_runs.py)
- [Artifact audit](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/audit.json)
- [Launch script](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/run.sh)
- [Remote training log](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/train.log)
- [Cleanup record](../../logs/diagnostics/x2-multi-motion-ppo-2gpu-8192env-100iter-20261014/cleanup.txt)

Final checkpoint: `model_99.pt` in the run directory.
Final checkpoint SHA-256: `785a7ed9bbb286b86a2d00ab0cbcacbc4d09b6e7fd4f18290727f83efe1bb76b`.

## 6. Next action and open items

Two decisions need operator authority:

1. The remote checkout now holds the multi-motion source as uncommitted files.
   Confirm whether to keep the deployment or restore `source-backup/`.
2. Do not extend this test or start a larger run without an approved budget.

Useful next measurements, in order of value:

1. Evaluate both checkpoints per motion on the same fixed references.
   Report clean full-clip completion, failure time, and phase.
2. Run a matched comparison that changes **one** variable.
   Compare 1 GPU at 8192 environments with 2 GPUs at 8192 environments on the same host.
   That separates the batch effect from the second-GPU effect.
3. Inspect why most episodes still end within about 17 control steps.

Related records: [implementation](x2_multi_motion_ppo_teacher_implementation.md), [single-GPU 100-iteration test](x2_multi_motion_ppo_all_tennis_100_iterations.md), [local smoke](x2_multi_motion_ppo_teacher_gpu_smoke.md), and [plan](../plans/x2_multi_motion_ppo_teacher.md).
