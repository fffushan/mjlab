#!/usr/bin/env python
"""Analyze one flow-matching capture against one or more checkpoints.

The trainer records the *inputs* of a batch, never the activations, because
hooks on every step would cost more than the recording.  This script closes the
loop: it rebuilds the exact noisy batch from the recorded generator state, runs
each checkpoint over it, and reports what the training loop could not afford.

    python scripts/analyze_flow_capture.py \
      --capture diag/captures/capture-00091034.npz \
      --token-cache /tmp/mjlab-144/data/offline/tokens-train-<hash>.npy \
      --config run/resolved-config.json \
      --checkpoint best=run/checkpoint-best.pt \
      --checkpoint last=run/checkpoint-last.pt \
      --out diag/analysis-00091034.json

Reported per checkpoint:

* the per-window loss distribution, so one exceptional sample is distinguishable
  from a model that became broadly worse;
* attention logit magnitude and attention entropy per layer and head;
* the variance of what each LayerNorm actually receives;
* the activation norm of every block;
* the AdamW first and second moments per block, and the update scale they imply.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.state import state_field_channels
from mjlab.tasks.tracking.flow_matching.checkpoint import FLOW_CHECKPOINT_FORMAT
from mjlab.tasks.tracking.flow_matching.process import (
  LinearNoiseProcess,
  add_linear_noise,
)


@dataclass
class CheckpointResult:
  label: str
  path: Path
  global_step: int
  write: dict[str, Any]


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--capture", type=Path, required=True)
  parser.add_argument("--token-cache", type=Path, required=True)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument(
    "--checkpoint",
    type=_label_path,
    action="append",
    required=True,
    metavar="LABEL=PATH",
  )
  parser.add_argument("--out", type=Path)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--attention-windows",
    type=int,
    default=8,
    help="windows used for the attention pass; attention weights are 41x41 per head",
  )
  parser.add_argument(
    "--chunk",
    type=int,
    default=512,
    help="windows per forward pass; the per-layer hooks are not enabled here",
  )
  return parser


def _label_path(raw: str) -> tuple[str, Path]:
  if "=" not in raw:
    raise argparse.ArgumentTypeError("--checkpoint expects LABEL=PATH")
  label, _, path = raw.partition("=")
  if not label or not path:
    raise argparse.ArgumentTypeError("--checkpoint expects LABEL=PATH")
  return label, Path(path)


def _load_checkpoint(path: Path, device: torch.device):
  payload = torch.load(path, map_location="cpu", weights_only=False)
  if payload.get("format") != FLOW_CHECKPOINT_FORMAT:
    raise SystemExit(f"{path} is not a flow-matching checkpoint")
  settings = DenoiserSettings(**payload["model_identity"]["settings"])
  model = StateLatentTransformer(settings)
  model.load_state_dict(payload["model"], strict=True)
  model.to(device)
  model.eval()
  process = LinearNoiseProcess(max_level=int(payload["process"]["max_level"]))
  return model, process, payload


def _rebuild_noised(
  capture: Mapping[str, np.ndarray],
  clean: Tensor,
  *,
  process: LinearNoiseProcess,
  masked_channels: Sequence[int],
  fill: float,
  device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
  """Rebuild the exact noisy input the trainer fed to the model."""
  rows = np.asarray(capture["microbatch_rows"], dtype=np.int64)
  states = np.asarray(capture["generator_state"])
  if rows.sum() != int(capture["window_indices"].size):
    raise SystemExit("capture row counts disagree with the window index array")
  pieces: list[Tensor] = []
  offset = 0
  for index, count in enumerate(rows.tolist()):
    generator = _generator_for_state(states[index], device)
    piece_clean = clean[offset : offset + count]
    state_levels = torch.as_tensor(
      capture["state_levels"][offset : offset + count], dtype=torch.long
    )
    latent_levels = torch.as_tensor(
      capture["latent_levels"][offset : offset + count], dtype=torch.long
    )
    noised = add_linear_noise(
      piece_clean[..., : process_state_dimension(capture)].to(device),
      piece_clean[..., process_state_dimension(capture) :].to(device),
      state_levels=state_levels.to(device),
      latent_levels=latent_levels.to(device),
      process=process,
      generator=generator,
    )
    pieces.append(noised.tokens)
    offset += count
  tokens = torch.cat(pieces, dim=0)
  mask = torch.as_tensor(capture["state_mask"], dtype=torch.bool)
  if masked_channels and bool(mask.any()):
    channels = list(masked_channels)
    rows_selected = mask[:, None, None]
    replacement = torch.full_like(tokens[..., channels], float(fill))
    tokens[..., channels] = torch.where(
      rows_selected, replacement, tokens[..., channels]
    )
  info = {
    "windows": int(tokens.shape[0]),
    "microbatches": int(rows.size),
    "masked_windows": int(mask.sum()),
    "history_dropout_windows": int(np.asarray(capture["history_dropout"]).sum()),
  }
  return tokens, info


def _generator_for_state(state: np.ndarray, device: torch.device) -> torch.Generator:
  """Rebuild the generator that produced a recorded noise draw.

  A CUDA generator serializes a 16-byte Philox state, a CPU generator a 5 056
  byte MT19937 state, and each rejects the other's.  The device is therefore
  taken from the state size rather than assumed.
  """
  tensor = torch.from_numpy(np.ascontiguousarray(state))
  cpu_state_bytes = torch.Generator(device="cpu").get_state().numel()
  if int(tensor.numel()) == int(cpu_state_bytes):
    generator = torch.Generator(device="cpu")
    generator.set_state(tensor)
    return generator
  if device.type != "cuda":
    raise SystemExit(
      "the capture holds a CUDA generator state; rerun with --device cuda:N"
    )
  # The generator lives on the device, but set_state takes the 16-byte Philox
  # state as a CPU ByteTensor even then.
  generator = torch.Generator(device=device)
  generator.set_state(tensor)
  return generator


def process_state_dimension(capture: Mapping[str, np.ndarray]) -> int:
  return int(np.asarray(capture["state_dimension"]).reshape(-1)[0])


def _step_ids(capture: Mapping[str, np.ndarray], device: torch.device) -> Tensor:
  state = torch.as_tensor(capture["state_levels"], dtype=torch.long).to(device)
  latent = torch.as_tensor(capture["latent_levels"], dtype=torch.long).to(device)
  return torch.stack((state, latent), dim=-1)


def _attention_stats(
  x: Tensor, layer: torch.nn.TransformerEncoderLayer
) -> dict[str, float]:
  """Recompute eager attention from the layer's fused projection.

  ``nn.TransformerEncoder`` uses the fused SDPA path, so weights never
  materialize.  One 41-token pass is cheap enough to recompute here.
  """
  attention = layer.self_attn
  weight: Tensor = attention.in_proj_weight
  bias = attention.in_proj_bias
  batch, steps, width = x.shape
  heads = int(attention.num_heads)
  head_dim = width // heads
  with torch.no_grad():
    projected = F.linear(x, weight, bias)
  q, k, _ = projected.split(width, dim=-1)
  q = q.view(batch, steps, heads, head_dim).transpose(1, 2)
  k = k.view(batch, steps, heads, head_dim).transpose(1, 2)
  logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
  probabilities = torch.softmax(logits, dim=-1)
  entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(-1)
  return {
    "attention_logit_abs_mean": float(logits.abs().mean()),
    "attention_logit_max": float(logits.max()),
    "attention_entropy_mean": float(entropy.mean()),
    "attention_entropy_max": float(entropy.max()),
    "attention_log_ratio_mean": float(entropy.mean() / math.log(float(steps))),
  }


def _hook_pass(
  model: StateLatentTransformer,
  tokens: Tensor,
  step_ids: Tensor,
) -> dict[str, Any]:
  """Collect activation, normalization and attention evidence on a few windows.

  Retaining per-layer activations for the whole capture needs about 16 GiB, so
  the loss pass runs separately in chunks and this pass sees a small subset.
  """
  layers = list(model.encoder.layers)
  layer_inputs: dict[int, Tensor] = {}
  norm_inputs: dict[str, float] = {}
  activation_norm: dict[str, float] = {}
  handles = []

  for index, layer in enumerate(layers):

    def capture_attention_input(
      module: Any, inputs: Any, *, index: int = index
    ) -> None:
      del module
      layer_inputs[index] = inputs[0].detach()

    def capture_norm_input(
      module: Any, inputs: Any, *, name: str = f"layer{index}"
    ) -> None:
      del module
      value = inputs[0].detach().float()
      norm_inputs[f"{name}_input_var"] = float(value.var(dim=-1).mean())
      norm_inputs[f"{name}_input_abs_max"] = float(value.abs().max())

    handles.append(layer.self_attn.register_forward_pre_hook(capture_attention_input))
    handles.append(layer.norm1.register_forward_pre_hook(capture_norm_input))
    if getattr(layer, "norm2", None) is not None:
      handles.append(
        layer.norm2.register_forward_pre_hook(
          lambda module, inputs, index=index: capture_norm_input(
            module, inputs, name=f"layer{index}_ffn"
          )
        )
      )

  def capture_activation(name: str) -> Any:
    def hook(module: Any, inputs: Any, output: Any) -> None:
      del module, inputs
      value = output[0] if isinstance(output, tuple) else output
      activation_norm[f"{name}_activation_norm"] = float(
        value.detach().float().norm(dim=-1).mean()
      )

    return hook

  for name, module in (
    ("input_projection", model.input_projection),
    ("output_projection", model.output_projection),
  ):
    handles.append(module.register_forward_hook(capture_activation(name)))
  for index, layer in enumerate(layers):
    handles.append(layer.register_forward_hook(capture_activation(f"layer{index}")))

  with torch.no_grad():
    model(tokens, step_ids)
  for handle in handles:
    handle.remove()

  attention: dict[str, Any] = {}
  for index in sorted(layer_inputs):
    for key, value in _attention_stats(layer_inputs[index], layers[index]).items():
      attention[f"layer{index}_{key}"] = value
  return {
    "activation_norm": activation_norm,
    "normalization_input": norm_inputs,
    "attention": attention,
  }


def _adamw_moments(
  payload: Mapping[str, Any], model: StateLatentTransformer
) -> dict[str, Any]:
  """Summarize the saved AdamW state per parameter block.

  The state is keyed by the parameter index inside the optimizer group, which
  follows ``model.parameters()`` order, so the names line up without extra
  metadata.
  """
  optimizer_state = payload.get("optimizer")
  if not isinstance(optimizer_state, Mapping) or not optimizer_state.get("state"):
    return {}
  names = [name for name, _ in model.named_parameters()]
  blocks: dict[str, dict[str, float]] = {}
  for key, entry in optimizer_state["state"].items():
    index = int(key)
    if index >= len(names):
      continue
    parts = names[index].split(".")
    block = (
      ".".join(parts[:3]) if parts[0] == "encoder" and len(parts) > 2 else parts[0]
    )
    first = entry.get("exp_avg")
    second = entry.get("exp_avg_sq")
    if not isinstance(first, Tensor) or not isinstance(second, Tensor):
      continue
    record = blocks.setdefault(block, {"first_norm": 0.0, "second_norm": 0.0})
    record["first_norm"] += float(first.float().pow(2).sum())
    record["second_norm"] += float(second.float().pow(2).sum())
  betas = optimizer_state["param_groups"][0]
  beta2 = float(betas.get("betas", (0.9, 0.999))[1])
  bias_correction = 1.0 - beta2
  result: dict[str, Any] = {}
  for block, record in sorted(blocks.items()):
    first = math.sqrt(record["first_norm"])
    second = math.sqrt(record["second_norm"])
    # sqrt of the bias-corrected second moment, the scale that divides the
    # first moment in the step.
    result[f"{block}_adamw_first_norm"] = first
    result[f"{block}_adamw_second_norm"] = second
    result[f"{block}_adamw_scale"] = first / (
      second / math.sqrt(bias_correction) + 1e-12
    )
  return result


def _replay_error(
  capture: Mapping[str, np.ndarray], window_loss: np.ndarray, *, step: int
) -> tuple[int, float]:
  """Compare the replay with the record on rows from the same update.

  A capture ring spans several updates, and every update uses the weights it
  produced.  Comparing the whole ring against one checkpoint would therefore
  report a difference that is real training progress, not a replay defect.

  The offset depends on when the checkpoint was written.  A pre-step snapshot
  (the current behaviour) carries the step that consumed the batch; an older
  post-step snapshot carries that step plus one.  Both are tried, and the
  closer match is reported with the offset that produced it, so mixed
  checkpoints stay readable instead of silently mislabelled.
  """
  recorded = np.asarray(capture["window_loss"], dtype=np.float64)
  updates = np.asarray(capture["microbatch_update_index"], dtype=np.int64)
  rows = np.repeat(updates, np.asarray(capture["microbatch_rows"], dtype=np.int64))
  if rows.size != recorded.size:
    return 0, float("nan")
  best: tuple[int, float] = (0, float("nan"))
  for offset in (0, 1):
    matched = rows + offset == int(step)
    if not bool(matched.any()):
      continue
    delta = float(np.abs(window_loss[matched] - recorded[matched]).max())
    if best[0] == 0 or delta < best[1]:
      best = (int(matched.sum()), delta)
  return best


def _chunked_window_loss(
  model: StateLatentTransformer,
  tokens: Tensor,
  clean: Tensor,
  step_ids: Tensor,
  *,
  chunk: int,
) -> np.ndarray:
  """Mean squared error per window, evaluated in bounded chunks."""
  values: list[Tensor] = []
  with torch.no_grad():
    for start in range(0, int(tokens.shape[0]), chunk):
      stop = start + chunk
      prediction = model(tokens[start:stop], step_ids[start:stop])
      error = (prediction.float() - clean[start:stop].float()) ** 2
      values.append(error.mean(dim=(1, 2)).detach().to("cpu"))
  return torch.cat(values).numpy()


def _distribution(values: np.ndarray) -> dict[str, float]:
  return {
    "loss_mean": float(values.mean()),
    "loss_median": float(np.median(values)),
    "loss_p95": float(np.percentile(values, 95)),
    "loss_max": float(values.max()),
    "loss_min": float(values.min()),
  }


def _analyze(args: argparse.Namespace) -> dict[str, Any]:
  device = torch.device(args.device)
  capture = dict(np.load(args.capture))
  config = json.loads(args.config.read_text())
  training = config.get("training", config)
  masked_fields = tuple(training.get("masked_state_fields") or ())
  masked_channels = state_field_channels(masked_fields) if masked_fields else ()
  fill = float(training.get("masked_state_fill_value", 0.0))

  cache = np.load(args.token_cache, mmap_mode="r")
  indices = np.asarray(capture["window_indices"], dtype=np.int64)
  clean = torch.from_numpy(np.array(cache[indices], dtype=np.float32))

  results: list[CheckpointResult] = []
  per_window: dict[str, np.ndarray] = {}
  recorded = np.asarray(capture["window_loss"], dtype=np.float64)
  for label, path in args.checkpoint:
    model, process, payload = _load_checkpoint(path, device)
    tokens, info = _rebuild_noised(
      capture,
      clean,
      process=process,
      masked_channels=masked_channels,
      fill=fill,
      device=device,
    )
    step_ids = _step_ids(capture, device)
    window_loss = _chunked_window_loss(
      model, tokens, clean.to(device), step_ids, chunk=args.chunk
    )
    subset = min(int(args.attention_windows), int(tokens.shape[0]))
    forward = _hook_pass(model, tokens[:subset], step_ids[:subset])
    per_window[label] = window_loss
    matched, replay_error = _replay_error(
      capture, window_loss, step=int(payload.get("global_step", -1))
    )
    write = {
      "checkpoint": str(path),
      "global_step": int(payload.get("global_step", -1)),
      "epoch": int(payload.get("epoch", -1)),
      **_distribution(window_loss),
      "recorded_window_loss_mean": float(recorded.mean()),
      "replay_abs_error": replay_error,
      "replay_matched_rows": matched,
      **info,
      **forward["activation_norm"],
      **forward["normalization_input"],
      **forward["attention"],
      **_adamw_moments(payload, model),
    }
    results.append(CheckpointResult(label, path, write["global_step"], write))

  comparison: dict[str, Any] = {}
  labels = [item.label for item in results]
  if len(labels) > 1:
    base = per_window[labels[0]]
    for label in labels[1:]:
      other = per_window[label]
      delta = other - base
      comparison[f"{label}_vs_{labels[0]}"] = {
        "windows_worse": float((delta > 0).mean()),
        "windows_better": float((delta < 0).mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "max_delta": float(delta.max()),
        "worst_window_index": int(
          np.asarray(capture["window_indices"])[int(delta.argmax())]
        ),
        "correlation": float(np.corrcoef(base, other)[0, 1]),
      }
  return {
    "capture": str(args.capture),
    "token_cache": str(args.token_cache),
    "checkpoints": [item.write for item in results],
    "comparison": comparison,
    "recorded_reason": _sidecar_reason(args.capture),
  }


def _sidecar_reason(path: Path) -> str:
  sidecar = path.with_suffix(".json")
  if not sidecar.exists():
    return "unknown"
  payload = json.loads(sidecar.read_text())
  return str(payload.get("reason", "unknown"))


def _summary(report: Mapping[str, Any]) -> str:
  lines = ["# Capture analysis", "", f"Capture: `{report['capture']}`", ""]
  header = "| checkpoint | step | mean | p95 | max | replay error |"
  lines += [header, "| --- | ---: | ---: | ---: | ---: | ---: |"]
  for item in report["checkpoints"]:
    lines.append(
      f"| {item['checkpoint'].split('/')[-1]} | {item['global_step']} | "
      f"{item['loss_mean']:.6f} | {item['loss_p95']:.6f} | {item['loss_max']:.6f} | "
      f"{item['replay_abs_error']:.2e} |"
    )
  for name, stats in report["comparison"].items():
    lines += [
      "",
      f"## {name}",
      "",
      f"- windows worse: {stats['windows_worse']:.1%}",
      f"- windows better: {stats['windows_better']:.1%}",
      f"- mean delta: {stats['mean_delta']:.6f}",
      f"- correlation: {stats['correlation']:.4f}",
      f"- worst window index: {stats['worst_window_index']}",
    ]
  lines += ["", f"Trigger reason: {report['recorded_reason']}", ""]
  return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  report = _analyze(args)
  text = json.dumps(report, indent=2, sort_keys=True)
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(_summary(report), encoding="utf-8")
  print(_summary(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
