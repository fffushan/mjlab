"""Export the flow x0 denoiser to ONNX and measure the sampling-pass latency.

The export reuses the accepted diffusion D5 export helper
(``scripts/experiments/diffusion_d5_parity/validate.py``: ``DenoiserExport``,
``_step_ids``, ``_export_onnx``) so the graph contract is identical:
inputs ``noisy_tokens`` float32 ``[B, 41, 167]`` and ``step_ids`` int64
``[B, 41, 2]``, output ``x0_hat`` float32 ``[B, 41, 167]``, dynamic batch axis,
opset 18.  Only the checkpoint loader and the sampler differ, because the flow
family has its own linear process and Euler solver.

The ONNX bytes are device-neutral, exactly as in the diffusion lane: the
provider is a runtime choice.  This tool therefore reports

1. the exported graph and its torch-versus-ONNX parity,
2. the wall time of the whole sampling pass on the requested device for a range
   of Euler step counts and batch sizes, and
3. the same pass through ONNX Runtime when a runtime provider is available.

Every measurement is a latency measurement on the named device.  It is not a
controller budget claim: the deployment target is the robot PC, not this GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[3]
_D5_PARITY = _ROOT / "scripts/experiments/diffusion_d5_parity/validate.py"

_STATE_DIMENSION = 135
_LATENT_DIMENSION = 32
_TOKEN_DIMENSION = 167
_WINDOW = 41
_HISTORY_ROWS = 8


def _load_d5_helper() -> Any:
  """Load the accepted diffusion export helper as an importable module."""
  spec = importlib.util.spec_from_file_location("d5_parity_export", _D5_PARITY)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load the export helper {_D5_PARITY}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


@dataclass(frozen=True, slots=True)
class Probe:
  """One fixed conditioning window used for export and parity."""

  tokens: torch.Tensor
  mask: torch.Tensor

  def to(self, device: torch.device, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = self.tokens.to(device=device).expand(batch, -1, -1).contiguous()
    mask = self.mask.to(device=device).expand(batch, -1, -1).contiguous()
    return tokens, mask


def _probe(batch: int) -> Probe:
  """Build the reference-free window shape: eight history rows, one current row.

  Rows 0-7 are fully known (measured history state and latent), row 8 knows its
  state (the measured current state) and not its latent, and rows 9-40 are
  unknown.  This mirrors the plan the D3/flow lane builds each tick, so the
  sampler work per step matches the real loop.
  """
  tokens = torch.zeros(batch, _WINDOW, _TOKEN_DIMENSION, dtype=torch.float32)
  mask = torch.zeros(batch, _WINDOW, _TOKEN_DIMENSION, dtype=torch.bool)
  mask[:, :_HISTORY_ROWS, :] = True
  mask[:, _HISTORY_ROWS, :_STATE_DIMENSION] = True
  return Probe(tokens=tokens, mask=mask)


def _step_ids(mask: torch.Tensor, source: int) -> torch.Tensor:
  """Integer level ids the sampler feeds the denoiser, on the mask's device.

  Same values as the accepted diffusion export helper builds: level 0 on the
  slices the window fixes (history state and latent, current state), and the
  source level everywhere else.
  """
  fixed_state = mask[..., :_STATE_DIMENSION].all(dim=-1)
  fixed_latent = mask[..., _STATE_DIMENSION:].all(dim=-1)
  result = torch.full(
    (*mask.shape[:2], 2), source, dtype=torch.long, device=mask.device
  )
  result[..., 0] = torch.where(
    fixed_state, torch.zeros_like(result[..., 0]), result[..., 0]
  )
  result[..., 1] = torch.where(
    fixed_latent, torch.zeros_like(result[..., 1]), result[..., 1]
  )
  return result


def _grid(calls: int, max_level: int) -> tuple[int, ...]:
  return tuple(round(max_level * (calls - index) / calls) for index in range(calls + 1))


def _time_callable(
  call: Any, *, device: torch.device, warmup: int, repeats: int
) -> dict[str, float]:
  """Time one callable on the device and return milliseconds."""
  for _ in range(warmup):
    call()
  if device.type == "cuda":
    torch.cuda.synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      call()
      end.record()
      torch.cuda.synchronize(device)
      samples.append(float(start.elapsed_time(end)))
  else:
    samples = []
    for _ in range(repeats):
      started = time.perf_counter()
      call()
      samples.append((time.perf_counter() - started) * 1000.0)
  samples.sort()
  return {
    "median_ms": statistics.median(samples),
    "p10_ms": samples[max(0, int(0.1 * len(samples)) - 1)],
    "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
    "min_ms": samples[0],
    "max_ms": samples[-1],
    "repeats": len(samples),
  }


def _sampler_pass(
  denoiser: Any,
  probe: Probe,
  *,
  device: torch.device,
  batch: int,
  calls: int,
  max_level: int,
  seed: int,
) -> Any:
  """Return a zero-argument callable that runs one full sampling pass."""
  from mjlab.tasks.tracking.flow_matching.process import LinearNoiseProcess
  from mjlab.tasks.tracking.flow_matching.sampler import (
    FlowConditions,
    FlowGrid,
    FlowSamplerConfig,
    sample_trajectory,
  )

  tokens, mask = probe.to(device, batch)
  conditions = FlowConditions(
    clean_tokens=tokens, clean_mask=mask, state_dimension=_STATE_DIMENSION
  )
  config = FlowSamplerConfig(
    grid=FlowGrid(levels=_grid(calls, max_level)),
    process=LinearNoiseProcess(max_level=max_level),
  )
  generator = torch.Generator(device=device).manual_seed(seed)

  def run() -> Any:
    return sample_trajectory(
      denoiser, conditions=conditions, config=config, generator=generator
    )

  return run


def _ort_sessions(path: Path) -> dict[str, Any]:
  """Return one ONNX Runtime session per available provider.

  Threads are capped: the default CPU session takes every core and makes the
  workstation unusable while a measurement runs.  The CPU provider is a
  reference number only; it is never the deployment path.
  """
  try:
    import onnxruntime as ort
  except ImportError:
    return {}
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  options.inter_op_num_threads = 1
  sessions: dict[str, Any] = {}
  for name in ort.get_available_providers():
    if name == "AzureExecutionProvider":
      continue
    try:
      sessions[name] = ort.InferenceSession(
        str(path), sess_options=options, providers=[name]
      )
    except Exception as exc:  # pragma: no cover - provider specific
      sessions[f"{name}:failed"] = str(exc)
  return sessions


def _ort_denoiser(session: Any, device: torch.device) -> Any:
  """Wrap an ONNX Runtime session as a torch-callable denoiser.

  Feed names are read from the session so the wrapper cannot silently swap the
  graph inputs.  The returned tensor is a CPU tensor converted back to the
  sampling device, which is what a runtime boundary would do.
  """
  feed_names = [item.name for item in session.get_inputs()]

  def denoiser(sample: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    values = {feed_names[0]: sample.detach().cpu().numpy()}
    values[feed_names[1]] = step_ids.detach().cpu().numpy()
    outputs = session.run(None, values)
    return torch.from_numpy(outputs[0]).to(device=device)

  return denoiser


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo-root", default=str(_ROOT))
  parser.add_argument(
    "--contract",
    default="docs/plans/beyondmimic_diffusion_d1_no_mixtures_vgk36_contract.yaml",
  )
  parser.add_argument(
    "--flow-checkpoint",
    default=(
      "logs/flow/flow-vgk36-400k-seed0-20261013T085536Z/output/checkpoint-best.pt"
    ),
  )
  parser.add_argument(
    "--normalization-bundle",
    default="logs/diffusion/d1-vgk36-double-20261012/offline/projection.npz",
  )
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--steps", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10, 20])
  parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 256])
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--repeats", type=int, default=20)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--skip-ort", action="store_true")
  parser.add_argument(
    "--torch-threads",
    type=int,
    default=1,
    help="host thread cap for the measurement process (default: 1)",
  )
  args = parser.parse_args(argv)

  torch.set_num_threads(int(args.torch_threads))

  root = Path(args.repo_root).resolve()
  output = Path(args.output_dir)
  if not output.is_absolute():
    output = root / output
  output.mkdir(parents=True, exist_ok=True)

  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but is unavailable")

  from mjlab.tasks.tracking.diffusion.replay_integration import (
    load_active_contract,
    load_flow_denoiser,
    load_projection_bundle,
  )

  contract = load_active_contract(root / args.contract)
  projection = load_projection_bundle(
    root / args.normalization_bundle, contract=contract
  )
  loaded = load_flow_denoiser(
    root / args.flow_checkpoint,
    contract=contract,
    projection_bundle=projection,
    verify_projection_hashes=False,
    solver_steps=max(args.steps),
  )
  max_level = loaded.process.max_level
  report: dict[str, Any] = {
    "schema": "mjlab-flow-onnx-latency-v1",
    "flow_checkpoint": str((root / args.flow_checkpoint).resolve()),
    "checkpoint_step": loaded.global_step,
    "ema_selected": loaded.ema_selected,
    "process_max_level": max_level,
    "process_identity": loaded.process_identity,
    "device": str(device),
    "device_name": (
      torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    ),
    "torch_version": torch.__version__,
    "declared_solver_steps": loaded.grid.calls,
    "probe": {
      "window": _WINDOW,
      "token_dimension": _TOKEN_DIMENSION,
      "known_history_rows": _HISTORY_ROWS,
      "current_row": _HISTORY_ROWS,
      "note": "rows 0-7 fully known, row 8 state only, rows 9-40 unknown",
    },
    "export": {},
    "parity": {},
    "torch_cuda": {},
    "onnxruntime": {},
  }

  # ---- export the denoiser with the accepted diffusion helper ----------------
  helper = _load_d5_helper()
  export_model = helper.DenoiserExport(loaded.model)
  probe = _probe(1)
  export_path = output / "denoiser.onnx"
  helper._export_onnx(
    export_model,
    (probe.tokens.clone(), _step_ids(probe.mask, max_level)),
    export_path,
    input_names=("noisy_tokens", "step_ids"),
    output_name="x0_hat",
    metadata={
      "artifact_kind": "flow_denoiser",
      "model_family": "x0_linear_flow",
      "sampler_method": "euler_x0",
      "process_max_level": str(max_level),
      "token_dimension": str(_TOKEN_DIMENSION),
      "sequence_length": str(_WINDOW),
      "checkpoint_step": str(loaded.global_step),
    },
  )
  report["export"] = {
    "path": str(export_path),
    "bytes": export_path.stat().st_size,
    "sha256": helper._canonical_digest(export_path),
    "inputs": ["noisy_tokens", "step_ids"],
    "outputs": ["x0_hat"],
    "dtype": "float32 inputs and output, int64 step ids",
    "opset": 18,
    "dynamic_batch": True,
  }

  # ---- parity: torch versus ONNX Runtime, and CPU versus CUDA ----------------
  model_cpu = loaded.model.to("cpu").eval()
  tokens, mask = probe.to(torch.device("cpu"), 2)
  step_ids = _step_ids(mask, max_level)
  with torch.no_grad():
    expected = model_cpu(tokens, step_ids).numpy()
  sessions = {} if args.skip_ort else _ort_sessions(export_path)
  for provider, session in sessions.items():
    if isinstance(session, str):
      report["parity"][provider] = {"session_error": session}
      continue
    names = [item.name for item in session.get_inputs()]
    got = session.run(None, {names[0]: tokens.numpy(), names[1]: step_ids.numpy()})[0]
    import numpy as np

    difference = np.max(np.abs(got - expected))
    scale = max(float(np.max(np.abs(expected))), 1e-12)
    report["parity"][provider] = {
      "max_abs_difference": float(difference),
      "max_rel_difference": float(difference / scale),
      "reference_max_abs": scale,
    }
  if device.type == "cuda":
    import numpy as np

    with torch.no_grad():
      on_gpu = (
        model_cpu.to(device)(tokens.to(device), step_ids.to(device))
        .detach()
        .cpu()
        .numpy()
      )
    report["parity"]["torch_cpu_vs_cuda"] = {
      "max_abs_difference": float(np.max(np.abs(on_gpu - expected)))
    }

  # ---- the whole sampling pass ---------------------------------------------
  model = loaded.model.to(device).eval()
  # One full pass per batch size before any measurement: the first call on a
  # device loads kernels and allocates workspace, which would otherwise inflate
  # whichever step count is measured first.
  for batch in args.batch_sizes:
    with torch.no_grad():
      _sampler_pass(
        model,
        probe,
        device=device,
        batch=batch,
        calls=max(args.steps),
        max_level=max_level,
        seed=args.seed,
      )()
  if device.type == "cuda":
    torch.cuda.synchronize(device)

  for steps in args.steps:
    key = f"steps_{steps}"
    report["torch_cuda"][key] = {}
    for batch in args.batch_sizes:
      pass_call = _sampler_pass(
        model,
        probe,
        device=device,
        batch=batch,
        calls=steps,
        max_level=max_level,
        seed=args.seed,
      )
      with torch.no_grad():
        timing = _time_callable(
          pass_call,
          device=device,
          warmup=args.warmup,
          repeats=args.repeats,
        )
        tokens_b, mask_b = probe.to(device, batch)
        step_ids_b = _step_ids(mask_b, max_level)

        def single_call(
          calls_tokens: torch.Tensor = tokens_b,
          calls_ids: torch.Tensor = step_ids_b,
        ) -> torch.Tensor:
          return model(calls_tokens, calls_ids)

        one_call = _time_callable(
          single_call,
          device=device,
          warmup=args.warmup,
          repeats=args.repeats,
        )
      timing["per_step_ms"] = timing["median_ms"] / steps
      timing["single_denoiser_call_ms"] = one_call["median_ms"]
      timing["control_period_share_at_50hz"] = timing["median_ms"] / 20.0
      report["torch_cuda"][key][f"batch_{batch}"] = timing

  # ---- ONNX Runtime pass, when a runtime provider exists -------------------
  for provider, session in sessions.items():
    if isinstance(session, str):
      continue
    runtime_denoiser = _ort_denoiser(session, device)
    report["onnxruntime"].setdefault(provider, {})
    for steps in args.steps:
      report["onnxruntime"][provider][f"steps_{steps}"] = {}
      for batch in args.batch_sizes:
        pass_call = _sampler_pass(
          runtime_denoiser,
          probe,
          device=device,
          batch=batch,
          calls=steps,
          max_level=max_level,
          seed=args.seed,
        )
        with torch.no_grad():
          timing = _time_callable(
            pass_call,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
          )
        timing["per_step_ms"] = timing["median_ms"] / steps
        report["onnxruntime"][provider][f"steps_{steps}"][f"batch_{batch}"] = timing

  report_path = output / "latency-report.json"
  report_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

  # ---- readable summary ----------------------------------------------------
  print(f"device {report['device_name']} | max_level {max_level} | ")
  print("torch CUDA sampling pass, median milliseconds")
  header = f"{'steps':>5s} " + " ".join(
    f"{'B=' + str(b):>9s}" for b in args.batch_sizes
  )
  print(header + f" {'per-step(B=1)':>13s}")
  for steps in args.steps:
    row = f"{steps:5d} "
    for batch in args.batch_sizes:
      item = report["torch_cuda"][f"steps_{steps}"][f"batch_{batch}"]
      row += f"{item['median_ms']:9.2f} "
    row += f"{report['torch_cuda'][f'steps_{steps}']['batch_1']['per_step_ms']:13.2f}"
    print(row)
  if report["onnxruntime"]:
    print("ONNX Runtime sampling pass, median milliseconds")
    for provider, payload in report["onnxruntime"].items():
      print(f"  provider {provider}")
      for steps in args.steps:
        row = f"  {steps:4d} "
        for batch in args.batch_sizes:
          item = payload[f"steps_{steps}"][f"batch_{batch}"]
          row += f"{item['median_ms']:9.2f} "
        print(row)
  print("parity:", json.dumps(report["parity"], sort_keys=True))
  print("report:", report_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
