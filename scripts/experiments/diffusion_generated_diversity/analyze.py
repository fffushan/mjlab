"""Generated-diversity (dispersion) measurement for the diffusion generator.

Purpose
-------
The D2 offline gates all pass while the recorded D3 closed loop fell 5/6 against
0/6 for the deterministic VAE baseline.  One candidate explanation is that an MSE
objective on a multi-modal future makes the sampler regress to the conditional
mean: an excellent normalised MSE, but a generator with no usable variety.

This script measures that directly.  For each window it draws ``K`` full DDIM
samples (the frozen 20-step grid, the same CPU-generator convention as
``evaluate_generation``) and decomposes the squared error against the clean target:

    mean_k ||x_k - x||^2  ==  spread  +  bias2
    (model error)            mean_k ||x_k - mean_k x_k||^2   ||mean_k x_k - x||^2

``spread`` is the generator's own variety in the same normalised token units as the
measured intrinsic residual spread of the data given the clean prefix
(``scripts/experiments/diffusion_future_predictability``), so the two are directly
comparable.  A model whose spread sits far below that residual is under-dispersed
and its low MSE is regression to the mean rather than skill.

Self-check
----------
With ``--cross-check`` the script also runs the frozen ``evaluate_generation`` on
seed 0 over the same records and compares its ``model_normalized_unknown_*_mse``
against this script's seed-0 error.  If the two disagree, this sampler path is not
the evaluator's path and no number here should be trusted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
from typing import Any

import numpy as np
import torch

from mjlab.tasks.tracking.diffusion.evaluation import evaluate_generation
from mjlab.tasks.tracking.diffusion.noising import clean_mask
from mjlab.tasks.tracking.diffusion.sampler import (
  SamplerConditions,
  SamplerConfig,
  sample_trajectory,
)
from mjlab.tasks.tracking.diffusion.schedule import build_inference_grid

PAIRED_EVAL = (
  pathlib.Path(__file__).resolve().parents[1]
  / "diffusion_paired_eval"
  / "paired_eval.py"
)


def _paired_eval_module():
  """Import the paired evaluator to reuse its validated arm loading."""
  spec = importlib.util.spec_from_file_location("paired_eval_reused", PAIRED_EVAL)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def block_slices(contract: Any) -> dict[str, slice]:
  """Channel slices of one token: the physical state block and the latent block."""
  state_stop = int(contract.projected_state_dimension)
  token_stop = int(contract.token_dimension)
  if not 0 < state_stop < token_stop:
    raise ValueError("contract token layout is not state-then-latent")
  return {"state": slice(0, state_stop), "latent": slice(state_stop, token_stop)}


def dispersion_decomposition(
  samples: np.ndarray, target: np.ndarray, unknown: np.ndarray
) -> dict[str, float]:
  """Split the squared error into generated spread and mean-sample bias.

  ``samples`` is ``(K, ...)``, ``target`` and ``unknown`` are ``(...)`` for one
  block.  ``spread`` is the within-window variance averaged over the unknown
  elements with the unbiased ``K/(K-1)`` correction, so an unpredictable future is
  not underestimated.  With that correction the identity
  ``error == spread * (K-1)/K + bias2`` holds; with the biased variance the two
  terms sum to the error exactly.
  """
  if samples.ndim < 1 or samples.shape[0] < 1:
    raise ValueError("samples must be (K, ...) with K >= 1")
  mask = np.broadcast_to(np.asarray(unknown, dtype=bool), target.shape)
  if mask.sum() == 0:
    raise ValueError("no unknown elements to measure")
  k = samples.shape[0]
  mean = samples.mean(axis=0)
  bias2 = float(((mean - target)[mask] ** 2).mean())
  if k < 2:
    # A single sample carries no spread information; its error is all bias.
    return {"elements": int(mask.sum()), "spread": 0.0, "bias2": bias2}
  # Restrict the variance to the unknown elements: known entries are clamped to the
  # target and identical across seeds, so counting them in the denominator would
  # under-report the spread by the ratio of unknown to total entries.
  masked = samples[:, mask]
  elements = int(masked.shape[1])
  residual = masked - mean[mask][None]
  spread = float((residual**2).sum() / ((k - 1) * elements))
  return {"elements": elements, "spread": spread, "bias2": bias2}


def sample_unknown_region(
  arm: dict[str, Any], records: list[Any], seeds: list[int], device: torch.device
) -> np.ndarray:
  """Return ``(K, windows, 41, tokens)`` DDIM samples, known entries clamped."""
  contract = arm["contract"]
  model = arm["model"]
  schedule = arm["schedule"]
  config = SamplerConfig(
    grid=build_inference_grid(schedule, contract=contract), start_level=1000
  )
  tokens = np.stack([np.asarray(record.tokens, dtype=np.float32) for record in records])
  mask = clean_mask(
    current_index=int(contract.current_index),
    state_dimension=int(contract.projected_state_dimension),
    token_dimension=int(contract.token_dimension),
    device="cpu",
  )
  out: list[np.ndarray] = []
  with torch.no_grad():
    for seed in seeds:
      generator = torch.Generator(device="cpu").manual_seed(int(seed))
      collected: list[np.ndarray] = []
      for start in range(0, len(tokens), 32):
        clean = torch.from_numpy(tokens[start : start + 32]).to(
          device=device, dtype=torch.float32
        )
        generated, _ = sample_trajectory(
          model,
          schedule=schedule,
          conditions=SamplerConditions(clean, mask.to(device)),
          config=config,
          generator=generator,
        )
        collected.append(generated.detach().cpu().numpy().astype(np.float64))
      out.append(np.concatenate(collected, axis=0))
  return np.stack(out, axis=0)


def measure_arm(
  arm: dict[str, Any],
  records: list[Any],
  seeds: list[int],
  device: torch.device,
  cross_check: bool,
) -> dict[str, Any]:
  contract = arm["contract"]
  slices = block_slices(contract)
  tokens = np.stack([np.asarray(record.tokens, dtype=np.float32) for record in records])
  mask = clean_mask(
    current_index=int(contract.current_index),
    state_dimension=int(contract.projected_state_dimension),
    token_dimension=int(contract.token_dimension),
    device="cpu",
  )
  unknown = np.broadcast_to(
    (~mask).detach().cpu().numpy().astype(bool), tokens.shape
  ).copy()
  samples = sample_unknown_region(arm, records, seeds, device)
  target64 = tokens.astype(np.float64)
  result: dict[str, Any] = {
    "name": arm["name"],
    "windows": len(records),
    "seeds": list(seeds),
    "checkpoint": arm["checkpoint"],
    "dataset": arm["dataset"],
    "blocks": {},
  }
  for name, channel_slice in slices.items():
    block_unknown = unknown[..., channel_slice]
    stats = dispersion_decomposition(
      samples[..., channel_slice], target64[..., channel_slice], block_unknown
    )
    per_seed_error = [
      float(
        (
          (samples[k][..., channel_slice] - target64[..., channel_slice])[block_unknown]
          ** 2
        ).mean()
      )
      for k in range(samples.shape[0])
    ]
    result["blocks"][name] = {
      **stats,
      "model_error_mean_over_seeds": float(np.mean(per_seed_error)),
      "model_error_seed0": per_seed_error[0],
      "spread_over_intrinsic": None,  # filled by the caller from the reference report
    }
  result["blocks"]["token"] = {
    "model_error_mean_over_seeds": float(
      np.mean(
        [
          float(((samples[k] - target64)[unknown] ** 2).mean())
          for k in range(samples.shape[0])
        ]
      )
    ),
    "model_error_seed0": float(((samples[0] - target64)[unknown] ** 2).mean()),
  }
  if cross_check:
    report = evaluate_generation(
      arm["model"],
      schedule=arm["schedule"],
      grid=build_inference_grid(arm["schedule"], contract=contract),
      records=records,
      projection=arm["loaded"].projection,
      contract=contract,
      seed=int(seeds[0]),
      device=device,
      split="validation",
    ).as_dict()
    metrics = report["metrics"]
    result["evaluator_cross_check"] = {
      "state": {
        "script_seed0": result["blocks"]["state"]["model_error_seed0"],
        "evaluator": metrics.get("model_normalized_unknown_state_mse"),
      },
      "latent": {
        "script_seed0": result["blocks"]["latent"]["model_error_seed0"],
        "evaluator": metrics.get("model_normalized_unknown_latent_mse"),
      },
      "token": {
        "script_seed0": result["blocks"]["token"]["model_error_seed0"],
        "evaluator": metrics.get("model_normalized_unknown_token_mse"),
      },
    }
  return result


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--arm",
    action="append",
    required=True,
    metavar="NAME=RUN_DIR:CONTRACT[:CHECKPOINT]",
    help="repeat once per arm",
  )
  parser.add_argument("--split", default="validation")
  parser.add_argument("--seeds", default="0,1,2,3")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--checkpoint", default="checkpoint-best.pt")
  parser.add_argument("--max-windows", type=int, default=None)
  parser.add_argument(
    "--intrinsic-residual",
    default=None,
    help="JSON from diffusion_future_predictability; adds spread/intrinsic ratios",
  )
  parser.add_argument("--cross-check", action="store_true")
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()

  device = torch.device(args.device)
  paired = _paired_eval_module()
  seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
  if not seeds:
    raise SystemExit("--seeds must list at least one seed")

  arms = []
  for raw in args.arm:
    name, rest = raw.split("=", 1)
    parts = rest.split(":")
    run_dir = pathlib.Path(parts[0])
    contract_path = pathlib.Path(parts[1])
    checkpoint = parts[2] if len(parts) > 2 else args.checkpoint
    arms.append(
      paired.load_arm(name, run_dir, contract_path, checkpoint, args.split, device)
    )

  datasets = {arm["dataset"] for arm in arms}
  if len(datasets) != 1:
    raise SystemExit(
      f"all arms must resolve to one dataset for comparability; got {sorted(datasets)}"
    )
  records = arms[0]["records"]
  if args.max_windows is not None:
    records = records[: args.max_windows]
  print(f"{len(records)} windows from {arms[0]['dataset']}, seeds {seeds}")

  intrinsic: dict[str, float] = {}
  if args.intrinsic_residual is not None:
    payload = json.loads(pathlib.Path(args.intrinsic_residual).read_text())
    for key, block in (
      ("ratio_state", "state"),
      ("ratio_latent", "latent"),
    ):
      if key in payload:
        intrinsic[block] = float(payload[key])
    if not intrinsic:
      for group, values in payload.items():
        if isinstance(values, dict) and "ratio_state" in values:
          intrinsic = {
            "state": float(values["ratio_state"]),
            "latent": float(values["ratio_latent"]),
          }
          print(f"intrinsic residual taken from group {group}: {intrinsic}")
          break

  payload_out: dict[str, Any] = {
    "kind": "diffusion_generated_diversity",
    "split": args.split,
    "seeds": seeds,
    "windows": len(records),
    "intrinsic_residual": intrinsic,
    "arms": [
      measure_arm(arm, records, seeds, device, args.cross_check) for arm in arms
    ],
  }
  for arm_result in payload_out["arms"]:
    for block in ("state", "latent"):
      value = arm_result["blocks"][block]["spread"]
      if block in intrinsic and intrinsic[block] > 0:
        arm_result["blocks"][block]["spread_over_intrinsic"] = value / intrinsic[block]
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload_out, indent=2, sort_keys=True) + "\n")
  print(
    json.dumps(
      {
        arm["name"]: {
          block: {
            "error_seed0": round(arm["blocks"][block]["model_error_seed0"], 6),
            "spread": round(arm["blocks"][block]["spread"], 6),
            "bias2": round(arm["blocks"][block]["bias2"], 6),
            "spread_over_intrinsic": (
              round(arm["blocks"][block]["spread_over_intrinsic"], 4)
              if arm["blocks"][block]["spread_over_intrinsic"] is not None
              else None
            ),
          }
          for block in ("state", "latent")
        }
        for arm in payload_out["arms"]
      },
      indent=2,
      sort_keys=True,
    )
  )
  print(f"report: {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
