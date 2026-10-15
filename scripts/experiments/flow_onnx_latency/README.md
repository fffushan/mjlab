# Flow denoiser ONNX export and sampling-pass latency

Two tools.

`measure.py` (laptop, torch) exports the x0 flow denoiser to ONNX with the same
helper the diffusion D5 lane uses, and measures the wall time of a whole
sampling pass for a range of Euler step counts and batch sizes on this machine.

`orin_ort_latency.py` (robot, NumPy plus ONNX Runtime) runs the exported graph
and the same Euler loop on the X2 robot PC, with no torch, and reports the same
quantities plus the observed GPU clock.

## Export

The export calls the **accepted diffusion D5 export helper**
(`scripts/experiments/diffusion_d5_parity/validate.py`: `DenoiserExport`,
`_export_onnx`), so the graph contract is the one the controller lane requires:

| Item | Value |
| --- | --- |
| inputs | `noisy_tokens` float32 `[batch, 41, 167]`, `step_ids` int64 `[batch, 41, 2]` |
| output | `x0_hat` float32 `[batch, 41, 167]` |
| opset / nodes | 18 / 986 |
| dynamic axes | batch only |
| bytes | 80,687,086 |
| sha256 | `c4736acfb6bfc314b84d6efa4ee2eef7ea8e03313d5458a153a6165c82454fcb` |
| metadata | `model_family=x0_linear_flow`, `sampler_method=euler_x0`, `process_max_level=1000`, `checkpoint_step=155097` |

The same bytes came out of three separate runs. The only family-specific parts
are the loader (`load_flow_denoiser`) and the sampler; the graph itself is
shared with the diffusion family.

The ONNX bytes are device-neutral, exactly as in the diffusion lane. The
provider is a runtime choice, not an artifact.

## Measured sampling pass

`checkpoint-best.pt` of `flow-vgk36-400k-seed0-20261013T085536Z`, EMA weights,
RTX 5080 Laptop, torch CUDA, `torch.set_num_threads(1)`. Window `[41, 167]`,
eight known history rows plus the current row's state, rows 9-40 unknown, which
is the window the lane builds each tick.

Three independent runs, median milliseconds (run C measured B=1 only):

| Steps | B=1 run A | B=1 run B | B=1 run C | B=256 run A | B=256 run B | per step (B=1) |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 4.88 | 4.57 | 4.55 | 157.21 | 157.93 | ~0.92 |
| 6 | 5.61 | 5.61 | 5.44 | 185.15 | 186.39 | ~0.92 |
| 7 | 6.85 | 6.63 | 6.33 | 217.83 | 218.39 | ~0.94 |
| 8 | 7.40 | 7.97 | 7.23 | 251.27 | 250.99 | ~0.94 |
| 9 | 8.67 | 9.25 | 8.10 | 282.97 | 283.39 | ~0.96 |
| 10 | 9.31 | 10.19 | 9.98 | 316.78 | 317.46 | ~0.98 |
| 20 | 18.69 | 20.38 | 18.53 | 630.69 | 633.78 | ~0.96 |

Share of the 20 ms 50 Hz control period at B=1, mean of the three runs:

| Steps | 5 | 6 | 7 | 8 | 9 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Share | 23% | 28% | 33% | 38% | 43% | 49% | 96% |

Components at B=1: one denoiser forward is about 0.70 ms (0.67-0.74 across runs),
the sampler adds about 0.25 ms per step (Euler update, clean-mask restore, and
the per-call finiteness check, which synchronizes to the host).

## Cross-check

The `x2_docker` D5 lane measured the diffusion denoiser with the C++ ONNX
Runtime lane on an RTX 4090 at **16.5 ms for a 20-update plan**, which is
**0.82 ms per update**, against about 0.95 ms per update here for the flow model
through torch on a laptop 5080. The two agree in order of magnitude.

## How to run

```bash
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 uv run --no-sync python \
  scripts/experiments/flow_onnx_latency/measure.py \
  --output-dir logs/diagnostics/flow-onnx-latency-<stamp> \
  --steps 5 6 7 8 9 10 20 --batch-sizes 1 256 --warmup 5 --repeats 20 --skip-ort
```

## Robot PC measurement — 2026-10-15 (+08)

Tool: `orin_ort_latency.py`. The same probe window and the same Euler loop, but
NumPy plus ONNX Runtime and no torch, so it runs on the Orin. Robot
`agi@172.21.204.234` (Orin NX 16GB, L4T R36.4.3), aarch64 GPU ONNX Runtime 1.23.0
unpacked from the local wheel into a private directory (no pip install), engine
cache in the run directory, two ONNX Runtime threads, one process.
Evidence: `logs/diagnostics/flow-onnx-latency-robot-20261015/`.

Two arms were run: **as found** (GPU min frequency 306 MHz, governed by
`nvhost_podgov`) and **clock locked** (`sudo jetson_clocks`: GPU min = max =
1173 MHz, CPU min = max = 1984 MHz). The step order is reversed in the TensorRT
arm and ascending in the CUDA arm, so order sensitivity is visible. 8 warmup and
20 repeats per row. The lock was restored afterwards: GPU governor back to
`nvhost_podgov` with min 306 MHz, CPU min back to 729.6 MHz.

TensorRT arm, as found:

| Steps | Median ms | p90 ms | Per step | Share of 20 ms | GPU clock MHz |
| --- | --- | --- | --- | --- | --- |
| 5 | 17.43 | 27.74 | 3.49 | 87% | 1020 |
| 6 | 16.46 | 34.85 | 2.74 | 82% | 1020 |
| 7 | 21.00 | 35.54 | 3.00 | 105% | 918 |
| 8 | 30.16 | 44.71 | 3.77 | 151% | 918-1020 |
| 9 | 30.36 | 47.73 | 3.37 | 152% | 918-1173 |
| 10 | 40.88 | 55.50 | 4.09 | 204% | 918 |
| 20 | 79.87 | 95.24 | 3.99 | 399% | 612-1173 |

TensorRT arm, clock locked:

| Steps | Median ms | p90 ms | Per step | Share of 20 ms | GPU clock MHz |
| --- | --- | --- | --- | --- | --- |
| 5 | 12.01 | 18.26 | 2.40 | 60% | 1173 |
| 6 | 15.82 | 25.54 | 2.64 | 79% | 1173 |
| 7 | 19.37 | 26.55 | 2.77 | 97% | 1173 |
| 8 | 20.07 | 41.14 | 2.51 | 100% | 1173 |
| 9 | 22.84 | 38.27 | 2.54 | 114% | 1173 |
| 10 | 24.92 | 34.91 | 2.49 | 125% | 1173 |
| 20 | 56.83 | 75.86 | 2.84 | 284% | 1173 |

CUDA arm, median milliseconds, as found then locked:

| Steps | As found | Locked | Locked per step |
| --- | --- | --- | --- |
| 5 | 26.28 | 19.71 | 3.94 |
| 6 | 26.15 | 22.12 | 3.69 |
| 7 | 28.13 | 25.44 | 3.63 |
| 8 | 33.14 | 29.29 | 3.66 |
| 9 | 40.53 | 32.98 | 3.66 |
| 10 | 40.81 | 36.43 | 3.64 |
| 20 | 86.12 | 72.08 | 3.60 |

Session providers were `['TensorrtExecutionProvider', 'CUDAExecutionProvider',
'CPUExecutionProvider']` and `['CUDAExecutionProvider', 'CPUExecutionProvider']`,
never CPU alone. The first TensorRT call built the engine in 6.66 s as found.

What this says:

- Clock locking gains **1.1x to 1.6x** (TensorRT median about 1.4x), and the
sampled GPU clock is flat at **1173 MHz** in every locked window, so the DVFS
ramp is excluded rather than averaged in.
- One pass costs about **2.5 to 2.8 ms** with TensorRT locked, against about
0.95 ms on the laptop, so the Orin is roughly **3x** slower per pass once the
clock is held. TensorRT beats CUDA here: 2.5-2.8 against 3.6-3.9 ms per step.
- **Locked TensorRT: 6 steps is 15.82 ms (79% of the period), 7 steps 19.37 ms
(97%), 10 steps 24.92 ms (125%).** The feasible band is 6 to 7 steps, and the
TensorRT p90 at 6 steps (25.5 ms) still exceeds the period.
- The locked 20-step median (56.83 ms, 2.84 ms per step) is within 12% of the
diffusion lane's clock-locked 50.56 ms for a 20-pass plan on this robot, so the
two graphs cost the same here.

Limits: latency only. The robot is shared; its ROS stack (including a TensorRT
stereo engine) held the GPU and load stayed near 10. The locked state is not the
robot's resting state: the lock reverts on reboot, and the pre-lock values were
restored by explicit writes. No controller, publisher or ROS node was started.
An earlier attempt fell back to CPU silently because ORT 1.23 rejects
`trt_tf32_enable`; that run is preserved as invalid evidence, and the tool now
refuses any session whose active provider is not the requested one.

## Limits

- Latency only, and only on the two machines measured here. The Orin is the
  robot PC, and its absolute numbers do not transfer to any other board.
- The laptop arm used `--skip-ort`: the GPU ONNX Runtime provider is not
  installed there, so the graph itself was timed through torch on the same
  arithmetic. The robot arm did time the graph, through TensorRT and CUDA.
- Reported robot numbers came from a shared robot with the ROS stack running.
  The clock-locked arm is not the robot's resting state.
- Latency is not a policy-quality measurement. A `--skip-ort` session is created
  with `intra_op_num_threads=2`, because the ONNX Runtime default takes every
  core. The step-count choice also rests on the closed-loop ablation in
  `docs/plans/x2_flow_matching_closed_loop.md`.

## Decision: six Euler steps

The project uses **6 Euler steps** for the flow model, and 6 is the code default
(`DEFAULT_CLOSED_LOOP_SOLVER_STEPS`), so no flag is needed. The flow package's
offline reference grid stays 20.

| Evidence | At 6 steps | Comparison |
| --- | --- | --- |
| Closed-loop falls, baseline | 357/2048 = 17.4% | 10 steps 14.8%, 20 steps 15.3%, all inside the repeat noise |
| Closed-loop falls, dropout20 | 307/2048 = 15.0% | 10 steps 17.1%, 20 steps 18.4% |
| 5 steps, rejected | baseline 423/2048 = 20.7% | about 5 points worse, three motions clearly worse |
| Orin latency, TensorRT, clock locked | **15.82 ms** median | 79% of the 20 ms period; 10 steps 24.92 ms, 20 steps 56.83 ms |

Open item: the 6-step p90 on the Orin is 25.5 ms, above the period, so a real
50 Hz loop still needs the asynchronous planner or a locked clock.
