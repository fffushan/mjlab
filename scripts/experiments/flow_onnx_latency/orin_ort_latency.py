"""Time the exported flow denoiser on the X2 robot PC with ONNX Runtime.

This is the robot counterpart of ``measure.py``.  It needs no torch: the graph
is the exported flow denoiser, the Euler update and the clean-mask restore are
plain NumPy, and the sampler step count is an argument.  Run it on the robot
with a GPU ONNX Runtime on ``PYTHONPATH``:

    LD_LIBRARY_PATH=<ort>/onnxruntime/capi PYTHONPATH=<ort> \
    python3 orin_ort_latency.py --onnx denoiser.onnx --output-dir <dir> \
      --steps 5 6 7 8 9 10 20 --repeats 20

The providers are tried in order and the report names the one that carried the
work (`tensorrt` first, then `cuda`).  A TensorRT engine is built on the first
call, so the run reports the build separately and excludes it from the timing.
Timing uses the real bursty pattern of the deployed lane: one request, then N
sequential passes with host work between them, not a sustained loop.

Limits: latency only.  It is not a policy-quality, stability or accuracy
measurement, and it does not start or touch the controller.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_WINDOW = 41
_TOKEN_DIMENSION = 167
_STATE_DIMENSION = 135
_HISTORY_ROWS = 8
_MAX_LEVEL = 1000


def _gpu_clock_node() -> Path | None:
  """Return the GPU devfreq node, so the clock can be sampled without sudo."""
  for path in sorted(glob.glob("/sys/class/devfreq/*")):
    try:
      name = (Path(path) / "name").read_text().strip().lower()
    except OSError:
      continue
    if "ga10b" in name or "gpu" in name:
      return Path(path) / "cur_freq"
  return None


def _clock_sampler(node: Path | None, stop: threading.Event) -> dict[str, Any]:
  samples: list[int] = []
  if node is not None:
    while not stop.is_set():
      try:
        samples.append(int(node.read_text().strip()))
      except OSError:
        pass
      time.sleep(0.002)
  return {"samples": samples}


def _probe() -> tuple[np.ndarray, np.ndarray]:
  """Rows 0-7 fully known, row 8 state only, rows 9-40 unknown."""
  tokens = np.zeros((1, _WINDOW, _TOKEN_DIMENSION), dtype=np.float32)
  mask = np.zeros((1, _WINDOW, _TOKEN_DIMENSION), dtype=bool)
  mask[:, :_HISTORY_ROWS, :] = True
  mask[:, _HISTORY_ROWS, :_STATE_DIMENSION] = True
  return tokens, mask


def _levels(calls: int) -> list[int]:
  return [round(_MAX_LEVEL * (calls - index) / calls) for index in range(calls + 1)]


def _step_ids(mask: np.ndarray, source: int) -> np.ndarray:
  fixed_state = mask[..., :_STATE_DIMENSION].all(axis=-1)
  fixed_latent = mask[..., _STATE_DIMENSION:].all(axis=-1)
  ids = np.full((*mask.shape[:2], 2), source, dtype=np.int64)
  ids[..., 0] = np.where(fixed_state, 0, ids[..., 0])
  ids[..., 1] = np.where(fixed_latent, 0, ids[..., 1])
  return ids


def _session(onnx: Path, provider: str, cache: Path, threads: int) -> Any:
  import onnxruntime as ort

  options = ort.SessionOptions()
  options.intra_op_num_threads = int(threads)
  options.inter_op_num_threads = 1
  if provider == "tensorrt":
    # These are the keys the deployed lane sends.  ORT 1.23 and 1.24 reject
    # trt_tf32_enable, so it must not be added: with fp16 off, the TensorRT fp32
    # builder default is already the TF32 path the lane measured.
    trt = {
      "device_id": "0",
      "trt_fp16_enable": "False",
      "trt_int8_enable": "False",
      "trt_max_partition_iterations": "1000",
      "trt_min_subgraph_size": "1",
      "trt_engine_cache_enable": "True",
      "trt_engine_cache_path": str(cache),
    }
    providers: list[Any] = [("TensorrtExecutionProvider", trt), "CUDAExecutionProvider"]
  else:
    providers = [("CUDAExecutionProvider", {"device_id": "0"})]
  session = ort.InferenceSession(str(onnx), sess_options=options, providers=providers)
  active = session.get_providers()
  # ONNX Runtime retries on CPU when a provider is rejected, which would turn
  # this into a CPU measurement without saying so.  Refuse that silently wrong
  # result: the requested provider must be the first active one.
  if not active or provider.lower() not in active[0].lower():
    raise SystemExit(
      f"requested provider {provider!r} is not active; session providers {active}"
    )
  return session


def _collect_clock(holder: dict[str, Any], node: Path | None, stop: Any) -> None:
  holder.update(_clock_sampler(node, stop))


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--provider", choices=("tensorrt", "cuda"), default="tensorrt")
  parser.add_argument("--steps", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10, 20])
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--repeats", type=int, default=20)
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--threads",
    type=int,
    default=2,
    help="ONNX Runtime intra-op threads (default 2; the robot is shared)",
  )
  args = parser.parse_args(argv)

  output = Path(args.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  onnx_path = Path(args.onnx).resolve()
  report: dict[str, Any] = {
    "schema": "x2-flow-onnx-robot-latency-v1",
    "onnx": str(onnx_path),
    "onnx_bytes": onnx_path.stat().st_size,
    "provider_requested": args.provider,
    "batch": args.batch,
    "steps": args.steps,
    "warmup": args.warmup,
    "repeats": args.repeats,
    "probe": {
      "window": _WINDOW,
      "token_dimension": _TOKEN_DIMENSION,
      "known_history_rows": _HISTORY_ROWS,
      "current_row_state_only": _HISTORY_ROWS,
    },
    "timing": {},
    "notes": [],
  }

  import onnxruntime as ort

  report["onnxruntime_version"] = ort.__version__
  report["providers_available"] = ort.get_available_providers()

  cache = output / "trt-cache"
  cache.mkdir(exist_ok=True)
  session = _session(onnx_path, args.provider, cache, args.threads)
  report["providers_session"] = session.get_providers()
  names = [item.name for item in session.get_inputs()]
  report["graph_inputs"] = names

  clock_node = _gpu_clock_node()
  report["gpu_clock_node"] = None if clock_node is None else str(clock_node)

  batch = int(args.batch)
  clean_tokens, mask = _probe()
  if batch != 1:
    clean_tokens = np.repeat(clean_tokens, batch, axis=0)
    mask = np.repeat(mask, batch, axis=0)
  rng = np.random.default_rng(args.seed)
  noise = rng.standard_normal((batch, _WINDOW, _TOKEN_DIMENSION)).astype(np.float32)

  def pass_once(calls: int) -> float:
    levels = _levels(calls)
    sample = np.where(mask, clean_tokens, noise).astype(np.float32)
    for source, destination in zip(levels[:-1], levels[1:], strict=True):
      ids = _step_ids(mask, source)
      started = time.perf_counter()
      x0_hat = session.run(None, {names[0]: sample, names[1]: ids})[0]
      elapsed = time.perf_counter() - started
      fraction = destination / source
      sample = (fraction * sample + (1.0 - fraction) * x0_hat).astype(np.float32)
      sample = np.where(mask, clean_tokens, sample).astype(np.float32)
    return elapsed

  # One call at the widest step count settles the engine build and kernel load.
  print("first call: TensorRT engine build happens here", flush=True)
  build_started = time.perf_counter()
  single = session.run(
    None,
    {
      names[0]: np.where(mask, clean_tokens, noise).astype(np.float32),
      names[1]: _step_ids(mask, _MAX_LEVEL),
    },
  )[0]
  report["first_call_seconds"] = time.perf_counter() - build_started
  report["first_call_finite"] = bool(np.isfinite(single).all())

  for calls in args.steps:
    for _ in range(args.warmup):
      pass_once(calls)
    samples: list[float] = []
    stop = threading.Event()
    holder: dict[str, Any] = {}
    sampler = threading.Thread(
      target=_collect_clock, args=(holder, clock_node, stop), daemon=True
    )
    sampler.start()
    for _ in range(args.repeats):
      started = time.perf_counter()
      pass_once(calls)
      samples.append(time.perf_counter() - started)
    stop.set()
    sampler.join(timeout=1.0)
    samples.sort()
    clocks = holder.get("samples") or []
    report["timing"][f"steps_{calls}"] = {
      "median_ms": statistics.median(samples) * 1000.0,
      "p10_ms": samples[max(0, int(0.1 * len(samples)) - 1)] * 1000.0,
      "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))] * 1000.0,
      "min_ms": samples[0] * 1000.0,
      "max_ms": samples[-1] * 1000.0,
      "per_step_ms": statistics.median(samples) * 1000.0 / calls,
      "control_period_share_at_50hz": statistics.median(samples) * 1000.0 / 20.0,
      "gpu_clock_min_mhz": None if not clocks else min(clocks) / 1e6,
      "gpu_clock_max_mhz": None if not clocks else max(clocks) / 1e6,
      "gpu_clock_samples": len(clocks),
      "repeats": len(samples),
    }

  report_path = output / f"robot-latency-{args.provider}.json"
  report_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

  print(
    f"onnxruntime {report['onnxruntime_version']} session {report['providers_session']}"
  )
  print(
    f"gpu clock node {report['gpu_clock_node']} | first call {report['first_call_seconds']:.2f} s"
  )
  print(
    f"{'steps':>5} {'median ms':>10} {'p90 ms':>8} {'per-step':>9} {'50Hz share':>11} {'gpu MHz':>16}"
  )
  for calls in args.steps:
    item = report["timing"][f"steps_{calls}"]
    clock = (
      "n/a"
      if item["gpu_clock_max_mhz"] is None
      else f"{item['gpu_clock_min_mhz']:.0f}-{item['gpu_clock_max_mhz']:.0f}"
    )
    print(
      f"{calls:5d} {item['median_ms']:10.2f} {item['p90_ms']:8.2f} "
      f"{item['per_step_ms']:9.2f} {item['control_period_share_at_50hz']:10.1%} {clock:>16}"
    )
  print("report:", report_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
