"""Atomic flow-matching checkpoints with strict flow-family identity.

A flow checkpoint uses its own on-disk ``format`` string, so a diffusion
checkpoint can never be loaded as a flow checkpoint and a flow checkpoint can
never be loaded as a diffusion checkpoint.  The payload records the linear
process identity (family, prediction type, path, objective and time encoding),
the data contract identity, the dataset and statistics identities, and the model
settings identity.  It records no diffusion noise schedule.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from .process import LinearNoiseProcess

FLOW_CHECKPOINT_FORMAT = "mjlab-x2-flow-matching-checkpoint-v1"

_SELECTION_SPLITS = frozenset(("train", "validation", "test"))
_REQUIRED_PROCESS_FIELDS = (
  "family",
  "prediction_type",
  "path",
  "loss",
  "time_encoding",
  "max_level",
  "embedding_count",
)


class FlowCheckpointError(ValueError):
  """A flow checkpoint is malformed or incompatible with the requested run."""


@dataclass(frozen=True, slots=True)
class FlowCheckpointState:
  """Restored flow checkpoint metadata and state dictionaries."""

  model_state: Mapping[str, Any]
  ema_state: Mapping[str, Any] | None
  optimizer_state: Mapping[str, Any] | None
  scaler_state: Mapping[str, Any] | None
  config: Mapping[str, Any]
  config_hash: str
  family: str
  process: Mapping[str, Any]
  process_identity: str
  data_identity: str
  dataset_identity: Mapping[str, Any]
  statistics_identity: Mapping[str, str]
  model_identity: Mapping[str, Any]
  model_identity_hash: str
  global_step: int
  epoch: int
  best_validation_loss: float | None
  best_metric_split: str | None
  selection_split: str | None
  rng_state: Any
  fp32_checked: bool
  torch_version: str
  cuda_version: str | None


def _state_dict(value: Any, name: str) -> Mapping[str, Any] | None:
  if value is None:
    return None
  method = getattr(value, "state_dict", None)
  if not callable(method):
    raise FlowCheckpointError(f"{name} must provide state_dict()")
  result = method()
  if not isinstance(result, Mapping):
    raise FlowCheckpointError(f"{name}.state_dict() must return a mapping")
  return result


def _identity(value: Any, name: str) -> str:
  method = getattr(value, "identity_hash", None)
  if not callable(method):
    method = getattr(value, "sha256", None)
  if not callable(method):
    raise FlowCheckpointError(f"{name} must provide identity_hash() or sha256()")
  digest = str(method())
  if len(digest) != 64 or any(
    character not in "0123456789abcdef" for character in digest
  ):
    raise FlowCheckpointError(f"{name} identity is not a lowercase SHA256 digest")
  return digest


def _config_payload(config: Any) -> tuple[dict[str, Any], str]:
  to_values = getattr(config, "as_dict", None)
  if callable(to_values):
    raw = to_values()
    if not isinstance(raw, Mapping):
      raise FlowCheckpointError("config.as_dict() must return a mapping")
    values: dict[str, Any] = {str(key): value for key, value in raw.items()}
  elif isinstance(config, Mapping):
    values = {str(key): value for key, value in config.items()}
  else:
    raise FlowCheckpointError("config must provide as_dict() or be a mapping")
  digest_method = getattr(config, "sha256", None)
  if callable(digest_method):
    digest = str(digest_method())
  else:
    digest = hashlib.sha256(
      json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
  return values, digest


def _jsonable(value: Any, name: str) -> dict[str, Any]:
  try:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    result = json.loads(encoded)
  except (TypeError, ValueError) as exc:
    raise FlowCheckpointError(f"{name} must be JSON serialisable") from exc
  if not isinstance(result, dict):  # pragma: no cover - json object guard
    raise FlowCheckpointError(f"{name} must be a mapping")
  return result


def identity_sha256(value: Mapping[str, Any]) -> str:
  """Return the canonical SHA256 of a JSON-serialisable identity mapping."""
  encoded = json.dumps(
    _jsonable(value, "identity"), sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _settings_payload(settings: Any) -> Any:
  if settings is None:
    return None
  if is_dataclass(settings) and not isinstance(settings, type):
    return asdict(cast("Any", settings))
  if isinstance(settings, Mapping):
    return {str(key): value for key, value in settings.items()}
  return repr(settings)


def model_identity(model: Any) -> dict[str, Any]:
  """Describe a model by type, settings and parameter shapes.

  The settings come from ``model.settings`` when present, so a Transformer
  records its exact dimensions.  Parameter shapes are recorded for every state
  entry, which fails closed on an architecture change before state loading.
  """
  settings_value = _settings_payload(getattr(model, "settings", None))
  state = getattr(model, "state_dict", None)
  shapes: dict[str, list[int]] = {}
  if callable(state):
    state_values = state()
    if isinstance(state_values, Mapping):
      for name, value in state_values.items():
        if isinstance(value, Tensor):
          shapes[str(name)] = [int(size) for size in value.shape]
  return {
    "type": type(model).__qualname__,
    "settings": settings_value,
    "parameter_shapes": shapes,
  }


def _validate_split_field(value: object, name: str) -> None:
  if value is None:
    return
  if not isinstance(value, str):
    raise FlowCheckpointError(f"{name} must be a string or None")
  if value not in _SELECTION_SPLITS:
    raise FlowCheckpointError(
      f"{name} must be one of train, validation or test, got {value!r}"
    )


def _float_or_none(value: Any, name: str) -> float | None:
  if value is None:
    return None
  try:
    result = float(value)
  except (TypeError, ValueError) as exc:
    raise FlowCheckpointError(f"{name} must be finite or None") from exc
  if not torch.isfinite(torch.tensor(result)):
    raise FlowCheckpointError(f"{name} must be finite or None")
  return result


def save_flow_checkpoint(
  path: str | Path,
  *,
  model: Any,
  ema: Any,
  optimizer: Any,
  scaler: Any,
  config: Any,
  process: LinearNoiseProcess,
  data_contract: Any,
  dataset_identity: Mapping[str, Any],
  model_description: Mapping[str, Any],
  global_step: int,
  epoch: int,
  best_validation_loss: float | None,
  rng_state: Any,
  best_metric_split: str | None = None,
  selection_split: str | None = None,
  fp32_checked: bool = False,
) -> None:
  """Atomically write model, optimizer, EMA, RNG and flow identity state."""
  if not isinstance(process, LinearNoiseProcess):
    raise FlowCheckpointError("process must be a LinearNoiseProcess")
  if global_step < 0 or epoch < 0:
    raise FlowCheckpointError("global_step and epoch must be non-negative")
  if not isinstance(fp32_checked, bool):
    raise FlowCheckpointError("fp32_checked must be a boolean")
  best_value = _float_or_none(best_validation_loss, "best_validation_loss")
  _validate_split_field(best_metric_split, "best_metric_split")
  if selection_split is None:
    selection_split = best_metric_split
  else:
    _validate_split_field(selection_split, "selection_split")
    if best_metric_split is not None and selection_split != best_metric_split:
      raise FlowCheckpointError(
        "selection_split contradicts best_metric_split; a checkpoint may not "
        "record two different selection splits"
      )

  config_values, config_hash = _config_payload(config)
  flow = process.as_dict()
  process_identity = process.identity_hash()
  data_identity = _identity(data_contract, "data contract")
  identity_values = _jsonable(dataset_identity, "dataset identity")
  statistics = identity_values.get("projection_hashes", {})
  if not isinstance(statistics, Mapping):
    statistics = {}
  statistics_identity = {str(key): str(value) for key, value in statistics.items()}
  model_values = _jsonable(model_description, "model identity")

  model_state = _state_dict(model, "model")
  if model_state is None:
    raise FlowCheckpointError("model cannot be None")
  payload: dict[str, Any] = {
    "format": FLOW_CHECKPOINT_FORMAT,
    "model": model_state,
    "ema": _state_dict(ema, "ema"),
    "optimizer": _state_dict(optimizer, "optimizer"),
    "scaler": _state_dict(scaler, "scaler"),
    "config": config_values,
    "config_hash": config_hash,
    "family": str(flow["family"]),
    "process": flow,
    "process_identity": process_identity,
    "data_identity": data_identity,
    "dataset_identity": identity_values,
    "statistics_identity": statistics_identity,
    "model_identity": model_values,
    "model_identity_hash": identity_sha256(model_values),
    "global_step": int(global_step),
    "epoch": int(epoch),
    "best_validation_loss": best_value,
    "best_metric_split": best_metric_split,
    "selection_split": selection_split,
    "rng_state": rng_state,
    "fp32_checked": fp32_checked,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
  }
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary: Path | None = None
  try:
    descriptor, temporary_name = tempfile.mkstemp(
      dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "wb") as handle:
      torch.save(payload, handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, destination)
    temporary = None
  except (OSError, RuntimeError) as exc:
    raise FlowCheckpointError(
      f"could not atomically write flow checkpoint {destination}"
    ) from exc
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def _require_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
  value = payload.get(name)
  if not isinstance(value, Mapping):
    raise FlowCheckpointError(f"checkpoint field {name!r} must be a mapping")
  return value


def _restore_process(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Return the recorded process dict after checking it is a linear process."""
  raw = _require_mapping(payload, "process")
  missing = [name for name in _REQUIRED_PROCESS_FIELDS if name not in raw]
  if missing:
    raise FlowCheckpointError(
      f"flow process is missing required fields: {', '.join(missing)}"
    )
  max_level = raw["max_level"]
  if isinstance(max_level, bool) or not isinstance(max_level, int):
    raise FlowCheckpointError("flow process max_level must be an integer")
  reconstructed = LinearNoiseProcess(max_level)
  expected = reconstructed.as_dict()
  actual = {str(key): value for key, value in raw.items()}
  if actual != expected:
    raise FlowCheckpointError(
      "flow process identity is inconsistent with the linear Gaussian contract"
    )
  if payload.get("process_identity") != reconstructed.identity_hash():
    raise FlowCheckpointError("flow process identity hash mismatch")
  if payload.get("family") != expected["family"]:
    raise FlowCheckpointError("flow family field disagrees with the process")
  return expected


def load_flow_checkpoint(
  path: str | Path,
  *,
  model: Any,
  ema: Any | None = None,
  optimizer: Any | None = None,
  scaler: Any | None = None,
  map_location: str | torch.device = "cpu",
  expected_process_identity: str | None = None,
  expected_data_identity: str | None = None,
) -> FlowCheckpointState:
  """Load a flow checkpoint and optionally restore supplied mutable components.

  A checkpoint written by the diffusion family is rejected because its on-disk
  format differs, so cross-family resume can never succeed silently.
  """
  source = Path(path)
  try:
    payload = torch.load(source, map_location=map_location, weights_only=False)
  except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
    raise FlowCheckpointError(f"could not read flow checkpoint {source}") from exc
  if (
    not isinstance(payload, Mapping) or payload.get("format") != FLOW_CHECKPOINT_FORMAT
  ):
    raise FlowCheckpointError("unsupported or malformed flow checkpoint")
  required = {
    "model",
    "config",
    "family",
    "process",
    "process_identity",
    "data_identity",
    "dataset_identity",
    "statistics_identity",
    "model_identity",
    "model_identity_hash",
    "global_step",
    "epoch",
    "best_validation_loss",
    "best_metric_split",
    "rng_state",
    "fp32_checked",
    "torch_version",
    "cuda_version",
  }
  if not required.issubset(payload):
    raise FlowCheckpointError("flow checkpoint is missing required identity fields")
  process = _restore_process(payload)
  process_identity = str(payload["process_identity"])
  data_identity = str(payload["data_identity"])
  if expected_process_identity is not None and process_identity != (
    expected_process_identity
  ):
    raise FlowCheckpointError(
      "flow checkpoint process identity differs from the active linear process"
    )
  if expected_data_identity is not None and data_identity != expected_data_identity:
    raise FlowCheckpointError(
      "flow checkpoint data identity differs from the active data contract"
    )

  model_state = _require_mapping(payload, "model")
  config = _require_mapping(payload, "config")
  dataset_identity = _require_mapping(payload, "dataset_identity")
  statistics_raw = _require_mapping(payload, "statistics_identity")
  model_values = _require_mapping(payload, "model_identity")
  # The configuration is recorded, not policed: a research run may resume one
  # checkpoint with one field changed, and a hash that rejects that forbids the
  # comparison.  The value stays in the payload for the closed-loop sidecars.
  config_hash = str(payload.get("config_hash", ""))
  model_identity_hash = str(payload["model_identity_hash"])
  if model_identity_hash != identity_sha256(model_values):
    raise FlowCheckpointError("flow checkpoint model identity hash mismatch")

  try:
    global_step = int(payload["global_step"])
    epoch = int(payload["epoch"])
  except (TypeError, ValueError) as exc:
    raise FlowCheckpointError("flow checkpoint step and epoch are invalid") from exc
  if global_step < 0 or epoch < 0:
    raise FlowCheckpointError("flow checkpoint step and epoch must be non-negative")
  best = _float_or_none(payload["best_validation_loss"], "best_validation_loss")

  best_metric_split = payload["best_metric_split"]
  _validate_split_field(best_metric_split, "checkpoint best_metric_split")
  selection_split = payload.get("selection_split")
  if selection_split is not None:
    _validate_split_field(selection_split, "checkpoint selection_split")
    if best_metric_split is not None and selection_split != best_metric_split:
      raise FlowCheckpointError(
        "checkpoint selection split contradicts its best metric split"
      )
  fp32_checked = payload["fp32_checked"]
  if not isinstance(fp32_checked, bool):
    raise FlowCheckpointError("flow checkpoint fp32_checked state is invalid")

  ema_state = payload.get("ema")
  if ema is not None and not isinstance(ema_state, Mapping):
    raise FlowCheckpointError(
      "flow checkpoint EMA state is required when an EMA object is supplied"
    )
  if ema_state is not None and not isinstance(ema_state, Mapping):
    raise FlowCheckpointError("flow checkpoint EMA state must be a mapping or None")
  optimizer_state = payload.get("optimizer")
  if optimizer is not None and not isinstance(optimizer_state, Mapping):
    raise FlowCheckpointError(
      "flow checkpoint optimizer state is required when an optimizer is supplied"
    )
  if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
    raise FlowCheckpointError(
      "flow checkpoint optimizer state must be a mapping or None"
    )
  scaler_state = payload.get("scaler")
  if scaler_state is not None and not isinstance(scaler_state, Mapping):
    raise FlowCheckpointError("flow checkpoint scaler state must be a mapping or None")

  load_model = getattr(model, "load_state_dict", None)
  if not callable(load_model):
    raise FlowCheckpointError("model must provide load_state_dict()")
  try:
    load_model(model_state)
    if ema is not None and ema_state is not None:
      ema.load_state_dict(ema_state)
    if optimizer is not None and optimizer_state is not None:
      optimizer.load_state_dict(optimizer_state)
    if scaler is not None and scaler_state is not None:
      scaler.load_state_dict(scaler_state)
  except (RuntimeError, ValueError, TypeError, KeyError) as exc:
    raise FlowCheckpointError(
      f"flow checkpoint state does not fit supplied objects: {source}"
    ) from exc

  return FlowCheckpointState(
    model_state=model_state,
    ema_state=ema_state,
    optimizer_state=optimizer_state,
    scaler_state=scaler_state,
    config=config,
    config_hash=config_hash,
    family=str(payload["family"]),
    process=process,
    process_identity=process_identity,
    data_identity=data_identity,
    dataset_identity=dict(dataset_identity),
    statistics_identity={str(k): str(v) for k, v in statistics_raw.items()},
    model_identity=dict(model_values),
    model_identity_hash=model_identity_hash,
    global_step=global_step,
    epoch=epoch,
    best_validation_loss=best,
    best_metric_split=best_metric_split,
    selection_split=selection_split,
    rng_state=payload["rng_state"],
    fp32_checked=fp32_checked,
    torch_version=str(payload["torch_version"]),
    cuda_version=None
    if payload["cuda_version"] is None
    else str(payload["cuda_version"]),
  )


__all__ = [
  "FLOW_CHECKPOINT_FORMAT",
  "FlowCheckpointError",
  "FlowCheckpointState",
  "identity_sha256",
  "load_flow_checkpoint",
  "model_identity",
  "save_flow_checkpoint",
]
