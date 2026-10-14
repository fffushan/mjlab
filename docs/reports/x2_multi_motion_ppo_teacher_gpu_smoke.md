# X2 multi-motion PPO teacher: local GPU smoke

Date: 2026-10-14 (+08).
Status: **PASS** for the bounded training-path smoke.
Scope: real MuJoCo environment and proximal policy optimization (PPO) on the local GPU.
This is not a tracking-quality or hardware qualification result.

## 1. Configuration

| Item | Value |
| --- | --- |
| Task | `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion` |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, index 0, 16303 MiB total memory |
| References | `single_000_zhanghongyu_agibot_x2_tracking.npz`: 453 frames; `single_001_zhanghongyu_agibot_x2_tracking.npz`: 340 frames |
| Reference rate and joint width | Both `50 Hz`, both `31` joints |
| Environments | `8`, allocated `4/4` across the clips |
| Actor / critic observations | `226 / 412` features, measured in the real environment |
| Actor and critic hidden layers | `(1024, 1024, 1024, 1024, 512, 128)` |
| Future reference | One joint-position/velocity frame at `0.2 s` |
| PPO updates | `10` runner iterations, `24` rollout steps per iteration |
| Optimizer steps | `200` for each populated optimizer state: 5 epochs × 4 minibatches × 10 iterations |
| Training transitions | `1920`: 8 environments × 240 batch steps |
| Seed | `7` |
| Logging | Local TensorBoard; model upload disabled |
| Bounds | 300-second service limit; 12 GiB host-memory cap; no host-memory swap allowance |

The smoke calls the actual `train.run_train()` path with instrumentation around the standard runner.
It uses the real environment, policy, and optimizer.
Instrumentation wraps the environment step and PPO update, then calls their original implementations.
The instrumentation checks outputs and captures evidence.
Runtime probes occur before collection; all environments reset before PPO training starts.
Probe transitions do not enter PPO storage.

Driver: [`scripts/experiments/x2_multi_motion_ppo_smoke/run.py`](../../scripts/experiments/x2_multi_motion_ppo_smoke/run.py).

## 2. Passed gates

| Gate | Evidence |
| --- | --- |
| Real GPU environment and runner | Both construct successfully; actor input 226, critic input 412 |
| Exact network shape | Both saved models contain the six requested hidden-layer sizes |
| Clip allocation | Four rows follow each reference |
| Future-reference indexing | Runtime q/dq matches the source NPZ at the exact 10-step offset, with endpoint clamp |
| Forced endpoint | Final-frame execution sets `clip_complete` and the environment done flag |
| Subset reset | Reset row keeps frame zero; other rows keep their frames, joint states, and generation IDs |
| Finite collection | Observations, sampled actions, and rewards remain finite across 240 batch steps |
| Finite PPO updates | All three reported loss fields remain finite for all 10 updates |
| Actual learning | Maximum actor parameter change is `0.006405848544090986` |
| Checkpoint save | `model_0.pt`, `model_5.pt`, and `model_9.pt` exist |
| Checkpoint reload | After deliberate actor-weight perturbation, load restores the saved state exactly |
| Inference parity | Action difference after reload is exactly `0.0` on one frozen observation batch |
| Artifact audit | 7 source hashes and 2 input hashes match; checkpoint reports iteration 9 and optimizer step 200 |
| Cleanup | Both smoke PIDs are absent; both owned systemd units have `MainPID=0` and are inactive |

Checkpoint reload uses the same live runner instance.
A fresh-process resume and interactive playback are not tested in this smoke.

The rollout reports 93 done-row events and 6 `clip_complete` events.
These are reset/termination counts, not tracking success rates.
Completion terms can coincide with physical failure.
Uniform start phases also allow short reference tails.
This smoke does not establish full-clip tracking quality.

## 3. Time and memory

The journal records service start at `2026-10-14T16:18:48+08:00` — `16:18:48 (+08)`.
Its final service record is `2026-10-14T16:19:00+08:00` — `16:19:00 (+08)`.

| Measurement | Value | Scope |
| --- | --- | --- |
| Internal elapsed time | `7.278487308998592 s` | Probes, training, saves, and reload checks; excludes Python import time |
| Systemd service runtime | `11.456 s` | Includes process startup |
| Peak Torch allocation | `282951680 B` = `269.84375 MiB` | Torch allocator only |
| Peak Torch reservation | `293601280 B` = `280 MiB` | Torch allocator only |
| Peak process RSS | `4804116480 B` = `4.47418212890625 GiB` | Resident set size; includes shared resident pages |

Torch memory does not include all MuJoCo Warp or driver allocations.
Total GPU-memory peak is not measured.
Another local D3 validation uses the GPU during the retry.
That process is not stopped or modified.
The timings are therefore not an isolated throughput benchmark.
They do not predict production training time or capacity at 4096 environments.

## 4. Failed attempts and corrections

All failed evidence remains in the artifact directory.
No PPO updates run in the failed attempts.

1. **Monitor 1: launcher error.** The monitor shell rejects Bash `pipefail` under `sh`.
   No training process starts.
   The corrected command explicitly uses `/usr/bin/bash -lc`.
2. **Monitor 2: smoke-driver assertion error.** The environment and runner build successfully.
   The driver incorrectly expects 512 critic inputs; the actual configured width is 412.
   The driver now derives widths from joint/body feature counts.
   The teacher implementation does not change.
3. **Monitor 3: corrected retry.** All gates pass; exit status is `0`.
   The budget stays at 8 environments and 10 updates, with the same time and memory limits.

## 5. Artifacts

Root: [`logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/`](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/).

- [Smoke result](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/run-retry1/smoke-result.json)
- [Independent artifact audit](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/audit.json)
- [Successful raw log](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/monitor-retry1.log)
- [First real attempt](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/run/smoke-result.json)
- [Launcher failure](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/launch-error.txt)
- [Reference manifest](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/motions.yaml)
- [Source snapshot](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/source-before-retry1.json)
- [Owned-process cleanup](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/cleanup.txt)
- [Diagnostic checkpoint](../../logs/diagnostics/x2-multi-motion-ppo-smoke-20261014/run-retry1/model_9.pt)

Checkpoint size: `97700935` bytes.
SHA-256: `335e38ce3050ec52fffc29edee0ef142acf2d97e11886ccb7cb428052fb00409`.
This checkpoint is a smoke artifact, not a qualified tracker or deployment policy.

## 6. Status and next action

The bounded local training path now passes its runtime gate.
Teacher source files and both reference clips remain unchanged.
The experimental smoke driver contains the corrected width assertion.
No production pilot, remote GPU job, VAE/diffusion change, or robot operation occurs.

Next action: select the diverse pilot motion list and authorize a training budget.
Per-motion tracking quality and held-out-reference behavior still need evaluation.
The [implementation report](x2_multi_motion_ppo_teacher_implementation.md) and [plan](../plans/x2_multi_motion_ppo_teacher.md) retain those limits.
