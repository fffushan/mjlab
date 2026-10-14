# X2 8-rank PPO: NCCL hang diagnosis

Date: 2026-10-14 (+08).
Status of the defect: **not reproduced**; cause remains unknown.
Status of the 8-rank configuration: it runs 900 iterations cleanly.

## 1. The failure being investigated

The 10,000-iteration 8-GPU run started at `19:34:44 (+08)` and died at `20:01:01`.
It made its last normal progress at `19:49:22`, around iteration 297.

Every rank logged, at `19:59:26`:

```
Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=19605, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
         Timeout(ms)=600000) ran for 600000 milliseconds before timing out.
```

Then the watchdog aborted the process group, one local rank received `SIGABRT`, the elastic agent terminated the other ranks, and `torchrunx` raised `WorkerFailedError`.
`exit.status` was `1`. No container restart occurred.

This was a crash, not an operator stop.
A `SIGINT` stop gives `exit.status 130` and no collective timeout.

## 2. What the crash logs establish

Per-rank NCCL work counters at the timeout:

| Rank | Last enqueued | Last completed |
| --- | --- | --- |
| **rank 0** | **19604** | 19604 |
| rank 1 to rank 7 | 19605 | 19604 |

Seven ranks reached the collective and waited.
**Rank 0 never enqueued it.**
Rank 0 is therefore the straggler, not a victim.

The collective count fixes the location.
`learn()` performs one initial parameter broadcast.
Each update performs 20 minibatches.
Each minibatch performs three collectives: the KL all-reduce, the learning-rate broadcast, and the gradient all-reduce.
Each update also performs 6 observer-normalizer reductions.
The total is 66 collectives per iteration.

```text
1 + 66 x 297 = 19603
```

The counters reached 19604, so the stuck operation is the second collective of the next update: the **learning-rate broadcast**, where rank 0 is the source.

Rank 0 therefore stalled inside this block, between a scalar all-reduce and a scalar broadcast:

```python
if self.gpu_global_rank == 0:
    if kl_mean > self.desired_kl * 2.0:
        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
if self.is_multi_gpu:
    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
    torch.distributed.broadcast(lr_tensor, src=0)
```

That block contains host-device synchronizations: the `kl_mean` comparisons read a device tensor, and `torch.tensor(..., device=...)` copies to the device.

### Two eliminations

- The observer normalizer has an early return before its collectives, which could desynchronize ranks:

  ```python
  if self.until is not None and self.count >= self.until:
      return
  ```

  It cannot fire here. `mlp_model.py:62` builds the normalizer as `EmpiricalNormalization(self.obs_dim)` with no `until`, so `until` is `None`.

- The dump produced no stack trace for a mechanical reason: the failed run left the flight recorder disabled, so there was nothing to dump. The dump then reported `Future for Flight recorder dump in heartbeatMonitor timed out after 15000 ms`.

## 3. The diagnostic run

The diagnostic kept the training configuration identical and added observability only.

| Item | Value |
| --- | --- |
| Artifacts | `logs/diagnostics/x2-multi-motion-ppo-8gpu-nccl-diag-20261014` |
| Start | `2026-10-14 20:11:51 (+08)` |
| End | `2026-10-14 20:54:30 (+08)` |
| Source | `9efad827d` |
| GPUs | 8, RTX 4090 |
| Environments | 8192 per rank, 65536 global |
| Clips | 148 |
| Iterations | 900 |
| Result | `exit.status 0`, 900 iterations, no timeout |

Observability variables, all forwarded by `torchrunx` because they match its `TORCH*` and `NCCL*` patterns:

```text
TORCH_NCCL_TRACE_BUFFER_SIZE=20000
TORCH_NCCL_ENABLE_TIMING=1
TORCH_NCCL_DUMP_ON_TIMEOUT=1
TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=120000
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60
TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/nccl-debug-info.txt
NCCL_DEBUG=WARN
PYTHONFAULTHANDLER=1
```

The dump target lives on the local NVMe of the container, not on the NAS.
No dump file appeared, because no timeout occurred.

`TORCH_NCCL_TRACE_BUFFER_SIZE` is deprecated in this PyTorch version; the message reads `use TORCH_FR_BUFFER_SIZE instead`.

### Results

| Measurement | Value |
| --- | --- |
| Iterations completed | `900` of `900` |
| Iteration time, minimum | `2.50 s` |
| Iteration time, median | `2.54 s` |
| Iteration time, 90th percentile | `2.56 s` |
| Iteration time, maximum | `4.45 s` (first iteration) |
| Iteration time, mean | `2.543 s` |
| Rank NCCL timeouts | none on any rank |
| Checkpoints written to the NAS | 9, from `model_0.pt` to `model_899.pt` |
| GPU state after the run | all eight idle |

The iteration times show no slow outlier after startup.
There was no near-miss.

## 4. Interpretation

The diagnostic ran 3 times past the failure point with no timeout and no slow iteration.

This does **not** establish that the defect is fixed, and it does **not** identify the cause.
The correct conclusion is that the failure is **intermittent**.

The diagnostic differed from the failed run in four ways, so the two runs are not a controlled comparison:

1. Observability variables were set. The flight recorder and the heartbeat settings change watchdog behaviour.
2. `save_interval` was `100` instead of `500`. This wrote **more** to the NAS, not less, so NAS write volume is not a supported trigger.
3. The iteration bound was `900` instead of `10000`.
4. No process tailed the training log. The failed run carried an attached log monitor until 48 seconds before its hang. A reader of the log cannot block a collective, but this difference is not eliminated.

One clean 900-iteration run gives no failure-rate estimate.
Two observed outcomes, one failure at iteration 297 and one success to iteration 900, are consistent with a configuration that is unreliable but sometimes succeeds.

## 5. Next single-variable tests

| Test | Purpose | Cost |
| --- | --- | --- |
| A. Relaunch the 10k run with the flight recorder enabled, no other change | Serves the training goal and captures a real dump if it hangs | about 7 h |
| B. Repeat 3 bounded 900-iteration runs, unchanged configuration | Estimates the failure rate | about 2 h |
| C. Same as B but `--log-root` on container-local NVMe | Tests the shared-NAS I/O hypothesis | about 2 h |
| D. Move the learning-rate sync from every minibatch to once per update, or use a fixed schedule | Removes 40 scalar collectives per iteration, including the one that stalled | code change plus review |

Test A is the only one that both advances training and improves future diagnosis.
Test B is the only one that measures reliability.
No test is authorized yet.

## 6. Artifacts

- Failed run: `logs/diagnostics/x2-multi-motion-ppo-8gpu-8192env-10kiter-20261014/`
- Diagnostic run: `logs/diagnostics/x2-multi-motion-ppo-8gpu-nccl-diag-20261014/`
- Diagnostic launch script: `run.sh` in the diagnostic directory
- Run parameters: `params/agent.yaml`, `params/env.yaml`, `params/resolved_motions.yaml` in each run directory
- Per-rank torchrunx logs: `training/*/*/torchrunx/<timestamp>/localhost[0-7].log` in each run directory

Related records: [8-GPU 10k run](../reports/x2_multi_motion_ppo_2gpu_8192env_100iterations.md), [plan](../plans/x2_multi_motion_ppo_teacher.md).
