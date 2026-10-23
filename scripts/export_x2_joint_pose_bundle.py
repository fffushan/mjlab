"""Export a trained joint-pose flow checkpoint as an X2 controller bundle.

The bundle holds one latent-free denoiser graph and the reference statistics the
controller needs to build a normalized 68-channel window:

* ``denoiser.onnx`` -- ``noisy_tokens`` float32 ``[B, 41, 68]`` and
  ``step_ids`` int64 ``[B, 41, 2]`` in, ``x0_hat`` float32 ``[B, 41, 68]`` out,
  dynamic batch axis, opset 18.
* ``reference_stats.json`` -- the train-only per-channel mean and standard
  deviation copied from ``projection.npz``.  The file holds exactly ``mean`` and
  ``std``, because the controller's bundle closure parses that asset with a
  strict key set; the dataset provenance is written to ``export-report.json``.
* ``tracker.onnx`` -- the frozen reference-free multi-motion teacher copied from
  ``--tracker-onnx``.
* ``descriptor.json`` -- the controller bundle declaration.
* ``SHA256SUMS`` -- digests of the three assets above.
* ``export-report.json`` -- checkpoint, dataset, parameter and export evidence.

The denoiser is built with :func:`mjlab.tasks.tracking.joint_pose.model.build_denoiser`
and traced with the accepted D5 helper
``scripts/experiments/diffusion_d5_parity/validate.py::_export_onnx``.  The
module itself is traced, not the D5 ``DenoiserExport`` wrapper, because that
wrapper reimplements the base forward and omits the optional final LayerNorm.

Usage:
  uv run python scripts/export_x2_joint_pose_bundle.py \\
    --checkpoint logs/flow/.../model_0400000.pt \\
    --data-dir /path/to/prepared-dataset \\
    --tracker-onnx /path/to/policy.onnx \\
    --output-dir /path/to/bundle
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

from mjlab.tasks.tracking.diffusion.model import DenoiserSettings
from mjlab.tasks.tracking.diffusion.replay_integration import apply_ema_selection
from mjlab.tasks.tracking.joint_pose.config import CHANNEL_LAYOUT, JointPoseDatasetCfg
from mjlab.tasks.tracking.joint_pose.model import build_denoiser
from mjlab.tasks.tracking.joint_pose.runner import CHECKPOINT_FORMAT


def _optional_onnxruntime() -> ModuleType | None:
  """Return ONNX Runtime, or ``None`` when the package is not installed."""
  try:
    import onnxruntime
  except ImportError:  # The export still runs; only the parity check is skipped.
    return None
  return onnxruntime


_ort = _optional_onnxruntime()

_ROOT = Path(__file__).resolve().parents[1]
_D5_VALIDATE = _ROOT / "scripts/experiments/diffusion_d5_parity/validate.py"

CHANNELS = 68
WINDOW_STEPS = 41
PAST_STEPS = 8
ACTION_DIMENSION = 31
OBSERVATION_DIMENSION = 226
CONTROL_PERIOD_SECONDS = 0.02
LOOKAHEAD_STEPS = 10
REGENERATION_PERIOD_TICKS = 16
MOTION_LOOKAHEAD_STEPS_KEY = "motion_lookahead_steps"
REFERENCE_STATS_FORMAT = "mjlab-x2-joint-pose-stats-v1"
_REPORT_FORMAT = "mjlab-x2-joint-pose-bundle-export-report-v1"
_PARITY_SEED = 20261023
_PARITY_ATOL = 1.0e-4
_PARITY_RTOL = 1.0e-4


class BundleExportError(RuntimeError):
  """The checkpoint or one of its inputs cannot produce a valid bundle."""


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
  try:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
  except (TypeError, ValueError) as exc:
    raise BundleExportError(f"value is not JSON serializable: {exc}") from exc
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
  path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_d5_export_helper() -> ModuleType:
  """Load the accepted D5 export helper as an importable module.

  The helper imports ``onnxruntime`` at module scope.  When that package is
  absent, a temporary stub satisfies the import so ``_export_onnx`` and
  ``_add_metadata`` remain usable; the caller skips the ONNX parity check.
  """
  spec = importlib.util.spec_from_file_location("d5_parity_export", _D5_VALIDATE)
  if spec is None or spec.loader is None:
    raise BundleExportError(f"cannot load the export helper {_D5_VALIDATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  stub = None
  if _ort is None:
    stub = ModuleType("onnxruntime")
    sys.modules["onnxruntime"] = stub
  try:
    spec.loader.exec_module(module)
  finally:
    if stub is not None:
      sys.modules.pop("onnxruntime", None)
  return module


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
  try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
  except (OSError, RuntimeError, EOFError, ValueError) as exc:
    raise BundleExportError(f"cannot read checkpoint {path}: {exc}") from exc
  if not isinstance(payload, Mapping):
    raise BundleExportError(f"checkpoint {path} does not hold a mapping")
  if payload.get("format") != CHECKPOINT_FORMAT:
    raise BundleExportError(
      f"checkpoint {path} has format {payload.get('format')!r}, expected "
      f"{CHECKPOINT_FORMAT!r}"
    )
  return payload


def _model_settings(payload: Mapping[str, Any]) -> DenoiserSettings:
  recorded = payload.get("model_settings")
  if not isinstance(recorded, Mapping):
    raise BundleExportError("checkpoint does not record model_settings")
  try:
    return DenoiserSettings(**dict(recorded))
  except (TypeError, ValueError) as exc:
    raise BundleExportError(f"checkpoint model_settings are invalid: {exc}") from exc


def _validate_dimensions(settings: DenoiserSettings) -> None:
  if settings.token_dimension != CHANNELS:
    raise BundleExportError(
      f"checkpoint token_dimension is {settings.token_dimension}, expected {CHANNELS}"
    )
  if settings.sequence_length != WINDOW_STEPS:
    raise BundleExportError(
      f"checkpoint sequence_length is {settings.sequence_length}, expected "
      f"{WINDOW_STEPS}"
    )


def _validate_recorded_past_steps(payload: Mapping[str, Any]) -> None:
  dataset_cfg = payload.get("dataset_cfg")
  if not isinstance(dataset_cfg, Mapping):
    return
  value = dataset_cfg.get("past_steps")
  if value is None:
    return
  if isinstance(value, bool) or not isinstance(value, int):
    raise BundleExportError("checkpoint dataset_cfg has a non-integer past_steps")
  if value != PAST_STEPS:
    raise BundleExportError(f"checkpoint past_steps is {value}, expected {PAST_STEPS}")


def _resolve_max_level(
  payload: Mapping[str, Any],
  settings: DenoiserSettings,
  requested: int | None,
) -> int:
  trainer = payload.get("trainer")
  recorded = trainer.get("max_level") if isinstance(trainer, Mapping) else None
  if recorded is not None and (
    isinstance(recorded, bool) or not isinstance(recorded, int)
  ):
    raise BundleExportError("checkpoint trainer settings have a non-integer max_level")
  if requested is None:
    if recorded is None:
      raise BundleExportError(
        "checkpoint does not record max_level; pass --max-level explicitly"
      )
    resolved = recorded
  else:
    if recorded is not None and requested != recorded:
      raise BundleExportError(
        f"--max-level {requested} differs from the checkpoint max_level {recorded}"
      )
    resolved = requested
  if settings.embedding_count != resolved + 1:
    raise BundleExportError(
      f"checkpoint has {settings.embedding_count} level embeddings but "
      f"max_level {resolved} needs {resolved + 1}"
    )
  return resolved


def _reference_stats(
  data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path]:
  projection = data_dir / "projection.npz"
  if not projection.is_file():
    raise BundleExportError(f"dataset statistics not found: {projection}")
  try:
    with np.load(projection, allow_pickle=False) as stats:
      missing = [key for key in ("mean", "std") if key not in stats.files]
      if missing:
        raise BundleExportError(f"{projection} is missing {', '.join(missing)}")
      mean = np.asarray(stats["mean"], dtype=np.float64)
      std = np.asarray(stats["std"], dtype=np.float64)
  except (OSError, ValueError) as exc:
    raise BundleExportError(f"cannot read {projection}: {exc}") from exc
  if mean.ndim != 1 or std.ndim != 1:
    raise BundleExportError(
      f"{projection} mean/std must be 1-D, got {mean.shape} and {std.shape}"
    )
  if mean.size != CHANNELS or std.size != CHANNELS:
    raise BundleExportError(
      f"{projection} mean/std length is {mean.size}/{std.size}, expected "
      f"{CHANNELS}/{CHANNELS}"
    )
  if mean.shape != std.shape:
    raise BundleExportError(
      f"{projection} mean {mean.shape} and std {std.shape} differ"
    )
  if not np.isfinite(mean).all() or not np.isfinite(std).all():
    raise BundleExportError(f"{projection} holds a non-finite mean or std")
  if bool((std <= 0.0).any()):
    raise BundleExportError(f"{projection} holds a non-positive std")
  return mean, std, projection


def _tracker_metadata(path: Path) -> dict[str, str]:
  if not path.is_file():
    raise BundleExportError(f"tracker ONNX not found: {path}")
  try:
    model = onnx.load(str(path))
  except Exception as exc:
    raise BundleExportError(f"cannot read tracker ONNX {path}: {exc}") from exc
  return {item.key: item.value for item in model.metadata_props}


def _require_tracker_lookahead(metadata: Mapping[str, str], path: Path) -> None:
  value = metadata.get(MOTION_LOOKAHEAD_STEPS_KEY)
  if value is None:
    raise BundleExportError(
      f"tracker ONNX {path} does not declare "
      f"{MOTION_LOOKAHEAD_STEPS_KEY}={LOOKAHEAD_STEPS}"
    )
  try:
    observed = int(value)
  except (TypeError, ValueError) as exc:
    raise BundleExportError(
      f"tracker ONNX {path} declares {MOTION_LOOKAHEAD_STEPS_KEY}={value!r}, "
      f"expected {LOOKAHEAD_STEPS}"
    ) from exc
  if observed != LOOKAHEAD_STEPS:
    raise BundleExportError(
      f"tracker ONNX {path} declares {MOTION_LOOKAHEAD_STEPS_KEY}={observed}, "
      f"expected {LOOKAHEAD_STEPS}"
    )


def _select_state(
  payload: Mapping[str, Any], parameters: str
) -> tuple[Mapping[str, object], Mapping[str, object] | None, bool, str]:
  raw = payload.get("model")
  if not isinstance(raw, Mapping):
    raise BundleExportError("checkpoint does not record a model state dict")
  ema = payload.get("ema")
  ema_available = isinstance(ema, Mapping)
  if parameters == "model":
    return raw, ema if isinstance(ema, Mapping) else None, ema_available, "model"
  if not ema_available:
    raise BundleExportError(
      "checkpoint has no 'ema' state; re-export with --parameters model to "
      "use the raw model weights"
    )
  assert isinstance(ema, Mapping)
  shadow = ema.get("shadow")
  if not isinstance(shadow, Mapping):
    raise BundleExportError("checkpoint 'ema' state has no 'shadow' mapping")
  return shadow, ema, ema_available, "ema.shadow"


def _final_norm_from_state(
  state: Mapping[str, object], based_on: str
) -> tuple[bool, tuple[str, ...]]:
  keys = {str(key) for key in state}
  required = {"final_norm.weight", "final_norm.bias"}
  present = keys & required
  if present == required:
    return True, tuple(sorted(present))
  if not present:
    return False, ()
  missing = sorted(required - present)
  raise BundleExportError(
    f"checkpoint {based_on} state holds an incomplete final-norm parameter "
    f"set; missing {missing}"
  )


def _build_model(
  settings: DenoiserSettings,
  data_dir: Path,
  max_level: int,
  final_norm: bool,
) -> nn.Module:
  dataset = JointPoseDatasetCfg(
    data_dir=str(data_dir),
    window_steps=settings.sequence_length,
    past_steps=PAST_STEPS,
    channels=settings.token_dimension,
  )
  model = build_denoiser(
    dataset,
    max_level=max_level,
    width=settings.width,
    layers=settings.layers,
    heads=settings.attention_heads,
    final_norm=final_norm,
  )
  if model.settings != settings:
    differing = [
      field.name
      for field in fields(DenoiserSettings)
      if getattr(model.settings, field.name) != getattr(settings, field.name)
    ]
    raise BundleExportError(
      "checkpoint model_settings differ from the joint-pose denoiser build: "
      f"{', '.join(differing)}"
    )
  return model


def _load_weights(
  model: nn.Module,
  payload: Mapping[str, Any],
  ema_state: Mapping[str, object] | None,
  parameters: str,
) -> int | None:
  raw = payload.get("model")
  if not isinstance(raw, Mapping):
    raise BundleExportError("checkpoint does not record a model state dict")
  try:
    model.load_state_dict(raw, strict=True)
  except Exception as exc:
    raise BundleExportError(
      f"checkpoint 'model' state does not match the denoiser: {exc}"
    ) from exc
  if parameters == "model":
    return None
  if ema_state is None:
    raise BundleExportError("checkpoint has no 'ema' state")
  try:
    selected, updates = apply_ema_selection(model, ema_state)
  except Exception as exc:
    raise BundleExportError(f"cannot restore the EMA weights: {exc}") from exc
  if not selected:
    raise BundleExportError("the EMA state was not applied")
  return updates


def _parameters_sha256(model: nn.Module) -> str:
  """Fingerprint the exported weights like ``joint_pose.evaluate`` does."""
  digest = hashlib.sha256()
  for name, value in sorted(model.state_dict().items()):
    digest.update(name.encode("utf-8"))
    array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
    digest.update(array.tobytes())
  return digest.hexdigest()


def _source_revision() -> str:
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=_ROOT,
      capture_output=True,
      text=True,
      check=False,
    )
  except OSError:
    return "unknown"
  if result.returncode != 0:
    return "unknown"
  return result.stdout.strip() or "unknown"


def _export_denoiser(
  model: nn.Module,
  path: Path,
  *,
  max_level: int,
  parameters: str,
  final_norm: bool,
) -> None:
  helper = _load_d5_export_helper()
  tokens = torch.zeros((1, WINDOW_STEPS, CHANNELS), dtype=torch.float32)
  step_ids = torch.zeros((1, WINDOW_STEPS, 2), dtype=torch.long)
  helper._export_onnx(
    model,
    (tokens, step_ids),
    path,
    input_names=("noisy_tokens", "step_ids"),
    output_name="x0_hat",
    metadata={
      "artifact_kind": "joint_pose_flow_denoiser",
      "model_family": "joint_pose_x0_linear_flow",
      "token_dimension": str(CHANNELS),
      "sequence_length": str(WINDOW_STEPS),
      "sampler_method": "euler_x0",
      "process_max_level": str(max_level),
      "parameters": parameters,
      "final_norm": "true" if final_norm else "false",
    },
  )


def _require_graph_contract(session: Any) -> None:
  inputs = {item.name: item.shape for item in session.get_inputs()}
  expected_inputs = {
    "noisy_tokens": (None, WINDOW_STEPS, CHANNELS),
    "step_ids": (None, WINDOW_STEPS, 2),
  }
  if set(inputs) != set(expected_inputs):
    raise BundleExportError(
      f"exported ONNX inputs are {sorted(inputs)}, expected {sorted(expected_inputs)}"
    )
  for name, expected in expected_inputs.items():
    shape = inputs[name]
    if len(shape) != len(expected):
      raise BundleExportError(
        f"exported ONNX input {name} has rank {len(shape)}, expected {len(expected)}"
      )
    for observed, required in zip(shape[1:], expected[1:], strict=True):
      if observed != required:
        raise BundleExportError(
          f"exported ONNX input {name} has shape {shape}, expected {expected}"
        )
  outputs = session.get_outputs()
  if len(outputs) != 1 or outputs[0].name != "x0_hat":
    raise BundleExportError(
      "exported ONNX output is not exactly ['x0_hat']: "
      f"{[item.name for item in outputs]}"
    )
  shape = outputs[0].shape
  if len(shape) != 3 or tuple(shape[1:]) != (WINDOW_STEPS, CHANNELS):
    raise BundleExportError(
      f"exported ONNX output 'x0_hat' has shape {shape}, expected "
      f"[batch, {WINDOW_STEPS}, {CHANNELS}]"
    )


def _verify_export(path: Path, model: nn.Module, max_level: int) -> dict[str, object]:
  zero_tokens = torch.zeros((1, WINDOW_STEPS, CHANNELS), dtype=torch.float32)
  zero_ids = torch.zeros((1, WINDOW_STEPS, 2), dtype=torch.long)
  with torch.no_grad():
    zero_output = model(zero_tokens, zero_ids)
  if tuple(zero_output.shape) != (1, WINDOW_STEPS, CHANNELS):
    raise BundleExportError(
      f"torch zero probe has shape {tuple(zero_output.shape)}, expected "
      f"[1, {WINDOW_STEPS}, {CHANNELS}]"
    )
  if not bool(torch.isfinite(zero_output).all()):
    raise BundleExportError("torch zero probe produced a non-finite value")

  rng = np.random.default_rng(_PARITY_SEED)
  probe_tokens = rng.standard_normal((1, WINDOW_STEPS, CHANNELS)).astype(np.float32)
  probe_ids = rng.integers(0, max_level + 1, size=(1, WINDOW_STEPS, 2), dtype=np.int64)
  with torch.no_grad():
    torch_output = (
      model(torch.from_numpy(probe_tokens), torch.from_numpy(probe_ids))
      .detach()
      .cpu()
      .numpy()
    )
  if tuple(torch_output.shape) != (1, WINDOW_STEPS, CHANNELS):
    raise BundleExportError(
      f"torch parity probe has shape {tuple(torch_output.shape)}, expected "
      f"[1, {WINDOW_STEPS}, {CHANNELS}]"
    )
  if not np.isfinite(torch_output).all():
    raise BundleExportError("torch parity probe produced a non-finite value")

  if _ort is None:
    print(
      "WARNING: onnxruntime is not importable; the ONNX shape and parity "
      "checks were skipped. The exported graph is NOT verified against torch.",
      file=sys.stderr,
      flush=True,
    )
    return {
      "onnxruntime_available": False,
      "onnx_parity_verified": False,
      "torch_zero_probe_shape": list(zero_output.shape),
      "torch_parity_probe_finite": True,
    }

  try:
    session = _ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  except Exception as exc:
    raise BundleExportError(
      f"ONNX Runtime could not load the exported graph {path}: {exc}"
    ) from exc
  _require_graph_contract(session)
  feed = {
    "noisy_tokens": zero_tokens.numpy(),
    "step_ids": zero_ids.numpy(),
  }
  zero_onnx = np.asarray(session.run(["x0_hat"], feed)[0], dtype=np.float32)
  if zero_onnx.shape != (1, WINDOW_STEPS, CHANNELS):
    raise BundleExportError(
      f"ONNX zero probe has shape {tuple(zero_onnx.shape)}, expected "
      f"[1, {WINDOW_STEPS}, {CHANNELS}]"
    )
  if not np.isfinite(zero_onnx).all():
    raise BundleExportError("ONNX zero probe produced a non-finite value")

  parity_feed = {"noisy_tokens": probe_tokens, "step_ids": probe_ids}
  onnx_output = np.asarray(session.run(["x0_hat"], parity_feed)[0], dtype=np.float32)
  if onnx_output.shape != torch_output.shape:
    raise BundleExportError(
      f"ONNX parity output shape {onnx_output.shape} differs from torch "
      f"{torch_output.shape}"
    )
  if not np.isfinite(onnx_output).all():
    raise BundleExportError("ONNX parity probe produced a non-finite value")
  difference = np.abs(
    np.asarray(onnx_output, dtype=np.float64)
    - np.asarray(torch_output, dtype=np.float64)
  )
  maximum = float(difference.max(initial=0.0))
  if not np.allclose(onnx_output, torch_output, atol=_PARITY_ATOL, rtol=_PARITY_RTOL):
    raise BundleExportError(
      "ONNX parity gate failed: max_abs_difference "
      f"{maximum:.6e} exceeds atol={_PARITY_ATOL:g}, rtol={_PARITY_RTOL:g}"
    )
  return {
    "onnxruntime_available": True,
    "onnx_parity_verified": True,
    "zero_probe_shape": list(zero_onnx.shape),
    "parity_seed": _PARITY_SEED,
    "parity_atol": _PARITY_ATOL,
    "parity_rtol": _PARITY_RTOL,
    "parity_max_abs_difference": maximum,
  }


def export_bundle(args: argparse.Namespace) -> dict[str, object]:
  """Validate the inputs and write one controller bundle."""
  output = args.output_dir.expanduser()
  descriptor_path = output / "descriptor.json"
  if descriptor_path.exists():
    raise BundleExportError(
      f"output directory already holds {descriptor_path}; refusing to overwrite "
      "an existing bundle"
    )

  checkpoint = args.checkpoint.expanduser()
  payload = _load_checkpoint(checkpoint)
  settings = _model_settings(payload)
  _validate_dimensions(settings)
  _validate_recorded_past_steps(payload)
  max_level = _resolve_max_level(payload, settings, args.max_level)

  data_dir = args.data_dir.expanduser()
  mean, std, projection_path = _reference_stats(data_dir)

  tracker = args.tracker_onnx.expanduser()
  tracker_metadata = _tracker_metadata(tracker)
  _require_tracker_lookahead(tracker_metadata, tracker)

  try:
    from mjlab.tasks.tracking.flow_matching.sampler import FlowGrid

    grid = FlowGrid.uniform(calls=args.steps, max_level=max_level)
  except ValueError as exc:
    raise BundleExportError(f"invalid sampler grid: {exc}") from exc

  selected_state, ema_state, ema_available, based_on = _select_state(
    payload, args.parameters
  )
  final_norm, final_norm_keys = _final_norm_from_state(selected_state, based_on)
  model = _build_model(settings, data_dir, max_level, final_norm)
  ema_updates = _load_weights(model, payload, ema_state, args.parameters)
  model.eval()
  parameters_sha256 = _parameters_sha256(model)

  dataset_identity = payload.get("dataset_identity")
  if not isinstance(dataset_identity, Mapping):
    raise BundleExportError("checkpoint does not record a dataset identity")
  dataset_sha256 = _canonical_sha256(dataset_identity)
  revision = _source_revision()

  output.mkdir(parents=True, exist_ok=True)
  denoiser_path = output / "denoiser.onnx"
  _export_denoiser(
    model,
    denoiser_path,
    max_level=max_level,
    parameters=args.parameters,
    final_norm=final_norm,
  )
  verification = _verify_export(denoiser_path, model, max_level)

  tracker_path = output / "tracker.onnx"
  if tracker.resolve() != tracker_path.resolve():
    shutil.copyfile(tracker, tracker_path)

  # The controller's bundle closure parses this asset with a STRICT key set: exactly
  # ``mean`` and ``std``.  Provenance therefore travels in export-report.json, not here.
  # src/joint_pose_bundle.cc::ParseReferenceStats refuses any additional key.
  reference_stats = {
    "mean": mean.tolist(),
    "std": std.tolist(),
  }
  stats_path = output / "reference_stats.json"
  _write_json(stats_path, reference_stats)

  denoiser_sha256 = _sha256_file(denoiser_path)
  stats_sha256 = _sha256_file(stats_path)
  tracker_sha256 = _sha256_file(tracker_path)
  checksums = [
    f"{denoiser_sha256}  denoiser.onnx",
    f"{stats_sha256}  reference_stats.json",
    f"{tracker_sha256}  tracker.onnx",
  ]
  (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")

  channel_layout = {name: [start, stop] for name, start, stop in CHANNEL_LAYOUT}
  sampler = {
    "method": "euler_x0",
    "max_level": max_level,
    "steps": args.steps,
    "levels": list(grid.levels),
  }
  parameter_tag = args.parameters + ("@finalnorm" if final_norm else "")
  descriptor = {
    "family": "joint_pose_tracking",
    "schema_version": 1,
    "task": {
      "denoiser_identity": f"{checkpoint.stem}@{parameter_tag}",
      "tracker_identity": tracker_sha256,
      "dataset_identity": dataset_sha256,
      "source_revision_x2": args.source_revision_x2,
      "source_revision_mjlab": revision,
      "mode": "generated_reference",
    },
    "dimensions": {
      "channels": CHANNELS,
      "window": WINDOW_STEPS,
      "past": PAST_STEPS,
      "lookahead_steps": LOOKAHEAD_STEPS,
      "joint": ACTION_DIMENSION,
      "action": ACTION_DIMENSION,
      "observation": OBSERVATION_DIMENSION,
      "control_period_seconds": CONTROL_PERIOD_SECONDS,
    },
    "channel_layout": channel_layout,
    "sampler": sampler,
    "regeneration": {"period_ticks": REGENERATION_PERIOD_TICKS},
    "assets": {
      "denoiser_onnx": {
        "path": "denoiser.onnx",
        "sha256": denoiser_sha256,
      },
      "reference_stats_json": {
        "path": "reference_stats.json",
        "sha256": stats_sha256,
      },
      "tracker_onnx": {
        "path": "tracker.onnx",
        "sha256": tracker_sha256,
      },
    },
    "defaults": {
      "max_source_age_seconds": 0.5,
      "history_gap_ticks": 2,
      "exhaustion_fallback_ticks": 20,
      "yaw_jump_warn_deg": 15.0,
    },
  }
  report = {
    "format": _REPORT_FORMAT,
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": _sha256_file(checkpoint),
    "checkpoint_step": int(payload.get("global_step", 0)),
    "parameters": args.parameters,
    "ema_available": ema_available,
    "ema_updates": ema_updates,
    "parameters_sha256": parameters_sha256,
    "final_norm": final_norm,
    "final_norm_based_on": based_on,
    "final_norm_keys": list(final_norm_keys),
    "selection_order": "selected_state_then_final_norm_then_trace",
    "source_revision_x2": args.source_revision_x2,
    "source_revision_mjlab": revision,
    "dataset_identity": dict(dataset_identity),
    "dataset_identity_sha256": dataset_sha256,
    "channel_layout": channel_layout,
    "sampler": sampler,
    "onnx_bytes": denoiser_path.stat().st_size,
    "denoiser_onnx": {
      "path": "denoiser.onnx",
      "bytes": denoiser_path.stat().st_size,
      "sha256": denoiser_sha256,
    },
    "reference_stats_json": {
      "path": "reference_stats.json",
      "sha256": stats_sha256,
      "format": REFERENCE_STATS_FORMAT,
      "channels": CHANNELS,
      "source": {
        "dataset_dir": str(data_dir.resolve()),
        "projection_npz": "projection.npz",
        "projection_npz_sha256": _sha256_file(projection_path),
      },
    },
    "tracker_onnx": {
      "path": "tracker.onnx",
      "source": str(tracker.resolve()),
      "sha256": tracker_sha256,
    },
    "verification": verification,
  }
  _write_json(output / "export-report.json", report)
  # The descriptor is the completion marker the exporter refuses to overwrite.
  _write_json(descriptor_path, descriptor)
  print(f"exported joint-pose bundle to {output}", flush=True)
  return report


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--tracker-onnx", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--max-level",
    type=int,
    default=None,
    help="Sampler maximum level; defaults to the checkpoint trainer value.",
  )
  parser.add_argument("--steps", type=int, default=6)
  parser.add_argument(
    "--source-revision-x2",
    default="unspecified",
    help=(
      "Controller revision this bundle targets. The bundle closure requires the "
      "field, and this tool cannot read the x2_docker checkout, so the caller "
      "records it."
    ),
  )
  parser.add_argument(
    "--parameters",
    choices=("ema", "model"),
    default="ema",
    help="Weights to export: the EMA shadow (default) or the raw model state.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  try:
    export_bundle(args)
  except BundleExportError as exc:
    print(f"x2-joint-pose-bundle-export: {exc}", file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
