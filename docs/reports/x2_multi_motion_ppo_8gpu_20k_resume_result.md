# X2 shared PPO teacher: 8-GPU resumed run to 20,000 iterations

Date: 2026-10-15 (+08).
Execution status: complete, exit code `0`.
Tracking-quality status: aggregate evidence only; not qualified per motion.
Scope: one authorized 8-GPU run on a managed container, resumed from the 900-iteration diagnostic checkpoint.

## 1. Result summary

The run trained the shared X2 teacher for 20,000 iterations over all 148 tennis references.
The policy now runs each reference to its clip endpoint.

| Measurement | Value |
| --- | --- |
| Iterations completed | `19999/20000` |
| Exit status | `0` |
| Wall-clock service time | `15 h 05 m 35 s` |
| Reported training time | `13:22:49` |
| Iterations in the resumed segment | `19101` |
| Transitions in the resumed segment | `30043275264` |
| Cumulative transitions over 20,000 iterations | `31457280000` |
| Final checkpoint | `model_19999.pt` |
| Checkpoint SHA-256 | `82f0dc32fb3b79406242d6a83e91d7e1a761b4de099d47a072f41c039431e51d` |
| NCCL collective timeouts | `0` |
| `SIGABRT` events | `0` |
| Flight-recorder dump file | none; no timeout occurred |

## 2. Provenance and configuration

The run resumes the checkpoint produced by the [900-iteration diagnostic run](x2_multi_motion_ppo_8gpu_nccl_hang_diagnosis.md).

| Item | Value |
| --- | --- |
| Resume source | `model_899.pt`, SHA-256 `65f0f1d9bbd177fca06738bbd94989d66672d8f560aff8f34ab1433a00ebbf38` |
| Resume check | `Loading model checkpoint from: .../model_899.pt` |
| Resume mechanism | `--agent.resume True --agent.load-run ... --agent.load-checkpoint model_899.pt` |
| Iteration offset evidence | Log header reads `Learning iteration 899/20000` |
| Additional iterations requested | `19101`, because `total = start_it + num_learning_iterations` |
| Task | `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion` |
| Source commit | `9efad827d` |
| GPUs | 8 x RTX 4090 on container `d-20260923051532-ntk4g` |
| Environments | `8192` per rank, `65536` global |
| References | `148` clips, `55` to `56` environments per clip |
| Network | actor and critic layers `(1024, 1024, 1024, 1024, 512, 128)`; inputs 226 and 412 |
| Preview | `0.2 s` |
| Seed | `7` |
| Save interval | `100` |

The copied resume source matched the recorded SHA-256 before launch.
The final checkpoint records iteration `19999`, `400020` optimizer steps, and common step counter `480024`.
Every saved actor and critic parameter is finite.
The final adaptive learning rate is `3.950617247028276e-05`.

The counters include iteration `899` twice, because the resumed loop begins at the checkpoint iteration index.
This is expected and is not double training of a different step.

## 3. Behaviour across the run

![Combined curves](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/combined-curves.png)

The plot joins the diagnostic run (iterations 0 to 898) and the resumed run (899 to 19999).
The dashed line marks the resume seam.

| Iteration | Mean return | Episode length (s) | Joint position error (rad) | Joint velocity error (rad/s) | `ee_body_pos` per step | `clip_complete` per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 899 | `1.0823` | `0.4468` | `1.0236` | `11.0161` | `7.7917` | `27.0833` |
| 3000 | `9.2612` | `3.5302` | `0.9712` | `7.1177` | `2.3333` | `49.4167` |
| 5000 | `8.9407` | `3.1236` | `0.9360` | `6.7367` | `1.6250` | `50.4583` |
| 10000 | `8.1982` | `2.8420` | `0.9091` | `6.5182` | `1.1667` | `51.3750` |
| 15000 | `8.8176` | `2.9194` | `0.8719` | `6.2560` | `0.6667` | `48.6667` |
| 19999 | `9.1600` | `3.0336` | `0.8791` | `6.2462` | `0.8750` | `52.1250` |

All logged scalars are finite.

### The episode-length collapse is gone

The earlier single-GPU run fell to `0.16 s` episodes and kept falling.
This run rises to about `3.0 s` and stays there.

The termination split at the final iteration, per step over `8192` rows:

| Termination term | Count per step |
| --- | ---: |
| `clip_complete` | `53.2083` |
| `ee_body_pos` | `0.8750` |
| `anchor_pos` | `0.1250` |
| `anchor_ori` | `0.0000` |
| `time_out` | `0.0000` |

`clip_complete` is `98.2 %` of all terminations.
Physical failures are `1.8 %`.

### Independent cross-check on episode length

A reset samples a uniform random phase inside the clip of that row.
A row that then runs to the clip endpoint therefore has an expected length of half its clip.
The 148 clips average `6.439594594594595 s`, so the expected mean episode is `3.2197972972972977 s`.
The measured mean episode is `3.0336 s`, which is `94 %` of that expectation.

This agreement supports the conclusion that episodes end at the clip endpoint, and that the residual `6 %` is early termination.
The check does not by itself prove per-clip completion.

## 4. Where the policy is still limited

Reward components at the final iteration, as a fraction of the maximum reachable sum for the measured episode length:

| Component | Achieved fraction of maximum |
| --- | ---: |
| `motion_body_pos` | `1.0051` |
| `motion_global_root_ori` | `0.9546` |
| `motion_body_ori` | `0.8041` |
| `motion_body_lin_vel` | `0.7503` |
| `motion_body_ang_vel` | `0.6411` |
| `motion_global_root_pos` | `0.4646` |

Body position tracking is saturated; the value slightly above `1.0` shows the ceiling estimate is not exact.
Body orientation, and both body velocity terms, have room left.
**Root position is the weakest term at `0.4646`.**
This agrees with `Metrics/motion/error_anchor_pos` at `0.4906 m`.

Root translation drift is the largest error source that remains in this teacher.

## 5. Optimisation plateau

Aggregate return is flat from about iteration `2000`:

- `9.26` at iteration 3000
- `9.16` at iteration 19999

Joint errors continue to improve slowly over the whole run, but most of the gain arrives in the first few thousand iterations.
The last 15,000 iterations produce little change in these rank-0 aggregates.

Consequences:

- A 20,000-iteration budget is more than this objective needs for aggregate metrics.
- Per-motion detail could still improve, and that gain may not appear in these aggregates.
  Only a per-motion evaluation can decide that.
- Do not treat the plateau as proof of convergence for every clip.

## 6. The resume seam

The seam is at iteration `899`.
The resumed process restores the actor, critic, optimizer state, iteration, and common step counter.
It also restores the observation normalizer, because that normalizer lives inside the actor and critic state dictionaries.

The seam is **not** bit-continuous:

- No random number generator state is saved in a checkpoint.
- `learn()` runs with `init_at_random_ep_len=True`, which re-randomises episode lengths at the start.
- Iteration `899` is executed once more.

The aggregate curves show no visible step or spike at the seam in the plot.
A small transient would be difficult to separate from the per-iteration noise band.
This is a descriptive observation about the rank-0 aggregates, not proof that the trajectory is continuous.

## 7. The intermittent NCCL hang

The [earlier 8-GPU run](x2_multi_motion_ppo_8gpu_nccl_hang_diagnosis.md) died at iteration 297 in a 600-second NCCL collective timeout.

This run performed `19101` further iterations with:

- `0` collective timeouts,
- `0` aborts,
- no flight-recorder dump, because no timeout occurred,
- iteration times of minimum `2.49 s`, median `2.52 s`, 99th percentile `2.56 s`, mean `2.522 s`, and a maximum of `4.55 s` on the first iteration only.

The configuration carried the same NCCL observability variables as the diagnostic run.
The cause of the original hang therefore remains **unknown**.
The evidence now supports an intermittent defect that did not recur in `19101` iterations, but it does not show that the defect is fixed.

## 8. Limits

- Every curve is **rank 0 only**. The other seven ranks produce gradients but no logged curve.
- The result is an **aggregate over 148 mixed clips**. No per-motion and no held-out evaluation exists.
- Several clips could be tracked poorly while the aggregate looks good, and the reverse.
- Short episodes could flatter average errors; this run avoids that failure mode because episodes reach a stable `3.0 s`, but the average still mixes the reset transient with steady tracking.
- The run is a software and simulation result. It makes no physical, stability, or hardware claim.
- One seed.
- The reward-component ceiling table depends on the assumed `maximum_episode_length_s` of `10.0` used to convert the logged normalized values back into episode sums.
- The `0.4906 m` anchor position error is a mean over mixed clips and reference phases.

## 9. Resource observations

| Item | Value |
| --- | --- |
| Run directory size | `24 GB` |
| Checkpoints written | `192` |
| GPU memory per rank | about `11.75 GiB`; rank 0 held `15.22 GiB` |
| Host memory used | about `46 GB` of `1007 GB` |
| Throughput | about `623000` `Perf/total_fps` |

The `save_interval` of `100` was chosen to bound loss if the intermittent hang returned.
It costs about `98 MB` per checkpoint.

The wall-clock service time exceeds the reported training time by about `1.7 h`, or `11 %`.
The measured iteration time covers only the rollout and the update.
It excludes logging and checkpoint writes, which 192 checkpoints dominate.
A larger `save_interval` would reduce this overhead at the cost of a longer replay after a failure.
The exact split between logging and checkpoint writing is not measured.

## 10. Cleanup

| Check | Result |
| --- | --- |
| Owned tmux session `x2resume` | gone |
| Owned training processes | none |
| All 8 GPUs | idle |
| `exit.status` | `0` |
| torchrunx | `All workers completed successfully` |

The pre-existing team tmux session `queue-dispatch` was not created or changed.

## 11. Artifacts

Root: [`logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/`](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/).

- [Combined curves](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/combined-curves.png)
- [Combined metrics CSV](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/combined-metrics.csv)
- [Combined summary](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/combined-summary.json)
- [Analysis script](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/analyze_combined.py)
- [Launch script](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/run.sh)
- [Training log](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/train.log)
- [Preflight record](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/preflight.txt)
- [Exit status](../../logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/exit.status)

The final checkpoint remains on the container at `24 GB` for the whole run directory.
A copy of the checkpoint is required before any downstream use.

## 12. Next action

The teacher is now good enough in aggregate to attempt the next stage, but only after a per-motion evaluation.

Recommended order:

1. Evaluate the final checkpoint **per motion** with fixed settings.
   Report clean full-clip completion, failure time and phase, and root-relative pose error.
2. Attack root position error, which is the weakest term at `0.4646` of maximum.
3. Decide whether further teacher iterations help.
   Base that decision on the per-motion result, not on the aggregate plateau.

No new training run is authorized by this report.

Related records: [NCCL hang diagnosis](x2_multi_motion_ppo_8gpu_nccl_hang_diagnosis.md), [single-GPU 100-iteration test](x2_multi_motion_ppo_all_tennis_100_iterations.md), [two-GPU test](x2_multi_motion_ppo_2gpu_8192env_100iterations.md), and [plan](../plans/x2_multi_motion_ppo_teacher.md).
