"""Versioned, atomic persistence for the bounded M3 training lifecycle.

Only validated tensor/plain-data state is serialized.  Live simulator objects,
teachers, and callables never cross this boundary.  Loading validates all
compatibility contracts before mutating the supplied trainer or replay.

A training checkpoint is a training artifact: it legitimately carries the
optimizer, raw replay, collector RNG, and trainer minibatch/replay settings.
:func:`load_inference_checkpoint` therefore restores only the saved model
schema, model settings, weights, and normalizers for model-only evaluation,
while :func:`load_checkpoint` stays the strict full training-resume path.
"""

from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab.tasks.tracking.distillation.collector import DAggerCollector
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.storage import (
  LabeledReplayBuffer,
  ReplayValidationError,
)
from mjlab.tasks.tracking.distillation.trainer import (
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.vae_config import ModelSettings, VaeSchema

CHECKPOINT_VERSION = 1
_CHECKPOINT_KIND = "mjlab-m3-distillation"


class CheckpointValidationError(ValueError):
  """A checkpoint is malformed or incompatible with the live components."""


@dataclass(frozen=True, slots=True)
class LifecycleState:
  """State returned after a validated checkpoint load."""

  counters: dict[str, int]
  schedule: dict[str, Any]
  resolved_config: dict[str, Any]
  teacher_hashes: dict[str, str]
  control_contract: dict[str, Any]
  resume_restarts_simulator: bool = True


@dataclass(frozen=True, slots=True)
class InferenceModel:
  """Model-only artifact reconstructed from a training checkpoint.

  The saved optimizer, replay, RNG, and trainer settings are deliberately not
  represented: model-only evaluation must not require them.  ``schema`` and
  ``settings`` are the *saved* identities used to rebuild ``model``.
  """

  model: ConditionalVAE
  schema: VaeSchema
  settings: ModelSettings
  counters: dict[str, int]
  schedule: dict[str, Any]
  resolved_config: dict[str, Any]
  teacher_hashes: dict[str, str]
  control_contract: dict[str, Any]


def _plain(value: Any, name: str) -> Any:
  """Reject executable/non-portable checkpoint metadata."""
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, (list, tuple)):
    return [_plain(item, name) for item in value]
  if isinstance(value, Mapping):
    if any(not isinstance(key, str) for key in value):
      raise CheckpointValidationError(f"{name} has a non-string key")
    return {key: _plain(item, name) for key, item in value.items()}
  raise CheckpointValidationError(f"{name} contains unsupported value {type(value)!r}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise CheckpointValidationError(f"{name} must be a mapping")
  result = _plain(value, name)
  assert isinstance(result, dict)
  return result


def _hashes(value: Any, name: str) -> dict[str, str]:
  result = _mapping(value, name)
  if any(
    not isinstance(key, str) or not isinstance(item, str)
    for key, item in result.items()
  ):
    raise CheckpointValidationError(f"{name} must map strings to strings")
  return result


def _schema_metadata(value: Any, name: str) -> dict[str, Any]:
  if isinstance(value, VaeSchema):
    return value.compatibility_metadata()
  return _mapping(value, name)


def _model_settings(metadata: Any) -> ModelSettings:
  """Rebuild the *saved* model settings; never a hard-coded architecture."""
  if not isinstance(metadata, Mapping) or set(metadata) != {
    "latent_dim",
    "hidden_dims",
    "activation",
    "beta",
  }:
    raise CheckpointValidationError("checkpoint model settings are malformed")
  hidden_dims = metadata["hidden_dims"]
  if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
    raise CheckpointValidationError(
      "checkpoint model settings hidden_dims must be a non-empty sequence"
    )
  if any(
    not isinstance(width, int) or isinstance(width, bool) or width <= 0
    for width in hidden_dims
  ):
    raise CheckpointValidationError(
      "checkpoint model settings hidden_dims must be positive integers"
    )
  try:
    return ModelSettings(
      latent_dim=metadata["latent_dim"],
      hidden_dims=tuple(hidden_dims),
      activation=metadata["activation"],
      beta=metadata["beta"],
    )
  except ValueError as exc:
    raise CheckpointValidationError(
      f"checkpoint model settings are invalid: {exc}"
    ) from exc


def _trainer_config(trainer: VaeDistillationTrainer) -> dict[str, Any]:
  config = trainer.config
  return {
    "learning_rate": config.learning_rate,
    "beta": config.beta,
    "accumulation_steps": config.accumulation_steps,
    "minibatch_size": config.minibatch_size,
    "latent_mode": config.latent_mode,
  }


def _global_rng_state() -> torch.Tensor:
  return torch.random.get_rng_state().detach().clone()


def _validate_rng_state(value: Any, name: str) -> torch.Tensor:
  """Return the state as the CPU uint8 vector PyTorch's RNG APIs demand.

  ``torch.load(map_location=...)`` may place a saved state on another device,
  and a checkpoint may store the state under a wider dtype holding the same
  bytes.  Both ``torch.random.set_rng_state`` and
  ``torch.Generator.set_state`` reject anything but a CPU ``uint8`` vector, so
  normalize here and return the normalized tensor for the caller to install
  instead of the raw payload tensor.  Reinterpreting another dtype as bytes is
  lossless; the probe below still rejects any payload that is not a real
  generator state.
  """
  if not isinstance(value, torch.Tensor) or value.ndim != 1:
    raise CheckpointValidationError(f"{name} must be a uint8 vector")
  try:
    state = value.detach().to("cpu").contiguous()
    if state.dtype != torch.uint8:
      state = state.view(torch.uint8)
    state = state.clone()
    probe = torch.Generator(device="cpu")
    probe.set_state(state)
  except (RuntimeError, TypeError, ValueError) as exc:
    raise CheckpointValidationError(f"{name} has an invalid generator state") from exc
  return state


def _payload(
  trainer: VaeDistillationTrainer,
  replay: LabeledReplayBuffer,
  *,
  counters: Mapping[str, int],
  schedule: Mapping[str, Any],
  resolved_config: Mapping[str, Any],
  teacher_hashes: Mapping[str, str],
  control_contract: Mapping[str, Any],
  collector: DAggerCollector | None,
) -> dict[str, Any]:
  if trainer.replay is not replay:
    raise CheckpointValidationError(
      "trainer and replay must be the same ownership instance"
    )
  clean_counters = _mapping(counters, "counters")
  if any(
    not isinstance(value, int) or isinstance(value, bool)
    for value in clean_counters.values()
  ):
    raise CheckpointValidationError("counters must contain integers")
  clean_schedule = _mapping(schedule, "schedule")
  clean_config = _mapping(resolved_config, "resolved_config")
  clean_hashes = _hashes(teacher_hashes, "teacher_hashes")
  clean_contract = _mapping(control_contract, "control_contract")
  rng: dict[str, Any] = {
    "global_cpu": _global_rng_state(),
    "trainer": trainer.generator_states(),
  }
  if collector is not None:
    rng["collector"] = collector.generator_state()
  return {
    "kind": _CHECKPOINT_KIND,
    "version": CHECKPOINT_VERSION,
    "schema": trainer.model.schema_metadata,
    "model_settings": trainer.model.settings.to_metadata(),
    "trainer_config": _trainer_config(trainer),
    "model": {
      key: value.detach().clone()
      if isinstance(value, torch.Tensor)
      else copy.deepcopy(value)
      for key, value in trainer.model.state_dict().items()
    },
    "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
    "trainer_counters": {
      "optimizer_steps": trainer.optimizer_steps,
      "microbatches_seen": trainer.microbatches_seen,
      "samples_seen": trainer.samples_seen,
      "normalization_update_ids": list(trainer.normalization_update_ids),
    },
    "counters": clean_counters,
    "schedule": clean_schedule,
    "resolved_config": clean_config,
    "teacher_hashes": clean_hashes,
    "control_contract": clean_contract,
    "replay": replay.state_dict(),
    "rng": rng,
    "resume_restarts_simulator": True,
  }


def _validate_payload(
  payload: Any,
  trainer: VaeDistillationTrainer,
  replay: LabeledReplayBuffer,
  *,
  expected_teacher_hashes: Mapping[str, str] | None,
  expected_control_contract: Mapping[str, Any] | None,
  collector: DAggerCollector | None,
) -> dict[str, Any]:
  if not isinstance(payload, Mapping):
    raise CheckpointValidationError("checkpoint root must be a mapping")
  required = {
    "kind",
    "version",
    "schema",
    "model_settings",
    "trainer_config",
    "model",
    "optimizer",
    "trainer_counters",
    "counters",
    "schedule",
    "resolved_config",
    "teacher_hashes",
    "control_contract",
    "replay",
    "rng",
    "resume_restarts_simulator",
  }
  if (
    set(payload) != required
    or payload["kind"] != _CHECKPOINT_KIND
    or payload["version"] != CHECKPOINT_VERSION
  ):
    raise CheckpointValidationError("unsupported or malformed checkpoint version")
  if (
    payload["schema"] != trainer.model.schema_metadata
    or payload["schema"] != replay.schema.compatibility_metadata()
  ):
    raise CheckpointValidationError(
      "checkpoint schema does not match live model/replay"
    )
  if payload["model_settings"] != trainer.model.settings.to_metadata():
    raise CheckpointValidationError("checkpoint model settings do not match live model")
  if payload["trainer_config"] != _trainer_config(trainer):
    raise CheckpointValidationError(
      "checkpoint trainer configuration does not match live trainer"
    )
  hashes = _hashes(payload["teacher_hashes"], "checkpoint teacher_hashes")
  if expected_teacher_hashes is not None and hashes != _hashes(
    expected_teacher_hashes, "expected_teacher_hashes"
  ):
    raise CheckpointValidationError("teacher artifact hashes do not match checkpoint")
  contract = _mapping(payload["control_contract"], "checkpoint control_contract")
  if expected_control_contract is not None and contract != _mapping(
    expected_control_contract, "expected_control_contract"
  ):
    raise CheckpointValidationError("control contract does not match checkpoint")
  counters = _mapping(payload["counters"], "checkpoint counters")
  if any(
    not isinstance(value, int) or isinstance(value, bool) for value in counters.values()
  ):
    raise CheckpointValidationError("checkpoint counters must contain integers")
  _mapping(payload["schedule"], "checkpoint schedule")
  _mapping(payload["resolved_config"], "checkpoint resolved_config")
  model_state = payload["model"]
  if not isinstance(model_state, Mapping):
    raise CheckpointValidationError("checkpoint model state is invalid")
  expected_model_keys = set(trainer.model.state_dict())
  if set(model_state) != expected_model_keys:
    raise CheckpointValidationError(
      "checkpoint model parameters do not match live model"
    )
  for key, value in model_state.items():
    current = trainer.model.state_dict()[key]
    if isinstance(current, torch.Tensor):
      if not isinstance(value, torch.Tensor):
        raise CheckpointValidationError(f"checkpoint model tensor {key!r} is missing")
      if value.shape != current.shape or value.dtype != current.dtype:
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is incompatible"
        )
      if not torch.isfinite(value).all().item():
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is non-finite"
        )
    elif value != current:
      raise CheckpointValidationError(
        f"checkpoint model metadata {key!r} is incompatible"
      )
  tc = _mapping(payload["trainer_counters"], "trainer_counters")
  required_tc = {
    "optimizer_steps",
    "microbatches_seen",
    "samples_seen",
    "normalization_update_ids",
  }
  if set(tc) != required_tc or any(
    not isinstance(tc[key], int) or isinstance(tc[key], bool)
    for key in required_tc - {"normalization_update_ids"}
  ):
    raise CheckpointValidationError("checkpoint trainer counters are invalid")
  if not isinstance(tc["normalization_update_ids"], list):
    raise CheckpointValidationError("checkpoint normalization identities are invalid")
  if len(set(map(repr, tc["normalization_update_ids"]))) != len(
    tc["normalization_update_ids"]
  ):
    raise CheckpointValidationError(
      "checkpoint normalization identities are duplicated"
    )
  rng = payload["rng"]
  if not isinstance(rng, Mapping) or set(rng) != (
    {"global_cpu", "trainer"} | ({"collector"} if collector is not None else set())
  ):
    raise CheckpointValidationError(
      "checkpoint RNG streams do not match live components"
    )
  # Restore must install these normalized states, not the raw payload ones:
  # a checkpoint read with a CUDA ``map_location`` carries CUDA tensors here.
  clean_rng: dict[str, Any] = {
    "global_cpu": _validate_rng_state(rng["global_cpu"], "global RNG state")
  }
  trainer_states = rng["trainer"]
  if not isinstance(trainer_states, Mapping) or set(trainer_states) != {
    "replay",
    "latent",
  }:
    raise CheckpointValidationError("checkpoint trainer RNG state is invalid")
  clean_rng["trainer"] = {
    "replay": _validate_rng_state(trainer_states["replay"], "replay RNG state"),
    "latent": _validate_rng_state(trainer_states["latent"], "latent RNG state"),
  }
  if collector is not None:
    clean_rng["collector"] = _validate_rng_state(
      rng["collector"], "collector RNG state"
    )
  try:
    replay._validated_state_storage(payload["replay"])  # noqa: SLF001
  except ReplayValidationError as exc:
    raise CheckpointValidationError(str(exc)) from exc
  if not isinstance(payload["optimizer"], Mapping):
    raise CheckpointValidationError("checkpoint optimizer state is invalid")
  clean = dict(payload)
  clean["rng"] = clean_rng
  return clean


def save_checkpoint(
  path: str | os.PathLike[str],
  trainer: VaeDistillationTrainer,
  replay: LabeledReplayBuffer,
  *,
  counters: Mapping[str, int] | None = None,
  schedule: Mapping[str, Any] | None = None,
  resolved_config: Mapping[str, Any] | None = None,
  teacher_hashes: Mapping[str, str] | None = None,
  control_contract: Mapping[str, Any] | None = None,
  collector: DAggerCollector | None = None,
) -> Path:
  """Atomically save a validated CPU-safe lifecycle checkpoint.

  A poisoned trainer (post-step failure without a validated restore) is
  refused through :meth:`VaeDistillationTrainer.assert_healthy` before any file
  is created, so a corrupt in-memory state can never be persisted.
  """
  trainer.assert_healthy()
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  payload = _payload(
    trainer,
    replay,
    counters=counters or {},
    schedule=schedule or {},
    resolved_config=resolved_config or {},
    teacher_hashes=teacher_hashes or {},
    control_contract=control_contract or {},
    collector=collector,
  )
  fd, temporary = tempfile.mkstemp(
    prefix=f".{destination.name}.", dir=destination.parent
  )
  try:
    with os.fdopen(fd, "wb") as handle:
      torch.save(payload, handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination
  except Exception:
    try:
      os.unlink(temporary)
    except FileNotFoundError:
      pass
    raise


def load_checkpoint(
  path: str | os.PathLike[str],
  trainer: VaeDistillationTrainer,
  replay: LabeledReplayBuffer,
  *,
  expected_teacher_hashes: Mapping[str, str] | None = None,
  expected_control_contract: Mapping[str, Any] | None = None,
  collector: DAggerCollector | None = None,
  map_location: str | torch.device = "cpu",
) -> LifecycleState:
  """Validate then restore state; failures roll all owned state back."""
  try:
    payload = torch.load(path, map_location=map_location, weights_only=True)
  except Exception as exc:
    raise CheckpointValidationError(f"could not load checkpoint: {exc}") from exc
  clean = _validate_payload(
    payload,
    trainer,
    replay,
    expected_teacher_hashes=expected_teacher_hashes,
    expected_control_contract=expected_control_contract,
    collector=collector,
  )
  old_model = {
    key: value.detach().clone()
    if isinstance(value, torch.Tensor)
    else copy.deepcopy(value)
    for key, value in trainer.model.state_dict().items()
  }
  old_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
  old_replay = replay.state_dict()
  old_counters = (
    trainer.optimizer_steps,
    trainer.microbatches_seen,
    trainer.samples_seen,
    set(trainer._normalization_update_ids),
  )  # noqa: SLF001
  old_rng = trainer.generator_states()
  old_global = _global_rng_state()
  old_collector = None if collector is None else collector.generator_state()
  try:
    trainer.model.load_state_dict(clean["model"], strict=True)
    trainer.optimizer.load_state_dict(clean["optimizer"])
    replay.load_state_dict(clean["replay"])
    tc = clean["trainer_counters"]
    trainer.optimizer_steps = tc["optimizer_steps"]
    trainer.microbatches_seen = tc["microbatches_seen"]
    trainer.samples_seen = tc["samples_seen"]
    trainer._normalization_update_ids = set(tc["normalization_update_ids"])  # noqa: SLF001
    trainer.set_generator_states(clean["rng"]["trainer"])
    torch.random.set_rng_state(clean["rng"]["global_cpu"])
    if collector is not None:
      collector.set_generator_state(clean["rng"]["collector"])
    # Clear post-step poison only as the last step of a fully validated
    # restore.  A failure here rolls the model/optimizer/replay/RNG back and
    # therefore leaves both the previous state and the poison intact.
    trainer.clear_poison_after_validated_restore()
  except Exception as exc:
    trainer.model.load_state_dict(old_model, strict=True)
    trainer.optimizer.load_state_dict(old_optimizer)
    replay.load_state_dict(old_replay)
    trainer.optimizer_steps, trainer.microbatches_seen, trainer.samples_seen, ids = (
      old_counters
    )
    trainer._normalization_update_ids = ids  # noqa: SLF001
    trainer.set_generator_states(old_rng)
    torch.random.set_rng_state(old_global)
    if collector is not None and old_collector is not None:
      collector.set_generator_state(old_collector)
    raise CheckpointValidationError(
      f"checkpoint restore failed before completion: {exc}"
    ) from exc
  return LifecycleState(
    counters=dict(clean["counters"]),
    schedule=dict(clean["schedule"]),
    resolved_config=dict(clean["resolved_config"]),
    teacher_hashes=dict(clean["teacher_hashes"]),
    control_contract=dict(clean["control_contract"]),
    resume_restarts_simulator=True,
  )


def load_inference_checkpoint(
  path: str | os.PathLike[str],
  *,
  device: str | torch.device = "cpu",
  expected_schema: VaeSchema | Mapping[str, Any] | None = None,
  expected_teacher_hashes: Mapping[str, str] | None = None,
  expected_control_contract: Mapping[str, Any] | None = None,
) -> InferenceModel:
  """Rebuild a model-only artifact from a training checkpoint.

  The saved schema and model settings are inferred, so the actual trained
  architecture is restored instead of a caller-supplied default.  Schema joint
  order, weight keys, shapes, dtypes, and finiteness, plus the teacher hashes
  and control contract, are validated before the model is returned.  Raw
  tensors are always read on CPU and only the model/normalizers are allocated
  on ``device``; no optimizer or replay buffer is constructed at all.
  """
  try:
    payload = torch.load(path, map_location="cpu", weights_only=True)
  except Exception as exc:
    raise CheckpointValidationError(f"could not load checkpoint: {exc}") from exc
  if not isinstance(payload, Mapping):
    raise CheckpointValidationError("checkpoint root must be a mapping")
  if (
    payload.get("kind") != _CHECKPOINT_KIND
    or payload.get("version") != CHECKPOINT_VERSION
  ):
    raise CheckpointValidationError("unsupported or malformed checkpoint version")
  required = {
    "kind",
    "version",
    "schema",
    "model_settings",
    "model",
    "teacher_hashes",
    "control_contract",
  }
  missing = required - set(payload)
  if missing:
    raise CheckpointValidationError(
      f"checkpoint is missing inference fields {sorted(missing)}"
    )
  try:
    schema = VaeSchema.from_metadata(payload["schema"])
  except ValueError as exc:
    raise CheckpointValidationError(f"checkpoint schema is invalid: {exc}") from exc
  settings = _model_settings(payload["model_settings"])
  if (
    expected_schema is not None
    and _schema_metadata(expected_schema, "expected_schema")
    != schema.compatibility_metadata()
  ):
    raise CheckpointValidationError(
      "checkpoint schema does not match the expected schema"
    )
  hashes = _hashes(payload["teacher_hashes"], "checkpoint teacher_hashes")
  if expected_teacher_hashes is not None and hashes != _hashes(
    expected_teacher_hashes, "expected_teacher_hashes"
  ):
    raise CheckpointValidationError("teacher artifact hashes do not match checkpoint")
  contract = _mapping(payload["control_contract"], "checkpoint control_contract")
  if expected_control_contract is not None and contract != _mapping(
    expected_control_contract, "expected_control_contract"
  ):
    raise CheckpointValidationError("control contract does not match checkpoint")
  model_state = payload["model"]
  if not isinstance(model_state, Mapping):
    raise CheckpointValidationError("checkpoint model state is invalid")
  model = ConditionalVAE(schema, settings)
  current_state = model.state_dict()
  if set(model_state) != set(current_state):
    raise CheckpointValidationError(
      "checkpoint weights do not match the saved model schema/settings"
    )
  for key, value in model_state.items():
    current = current_state[key]
    if isinstance(current, torch.Tensor):
      if not isinstance(value, torch.Tensor):
        raise CheckpointValidationError(f"checkpoint model tensor {key!r} is missing")
      if value.shape != current.shape or value.dtype != current.dtype:
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is incompatible"
        )
      if not torch.isfinite(value).all().item():
        raise CheckpointValidationError(
          f"checkpoint model tensor {key!r} is non-finite"
        )
    elif value != current:
      raise CheckpointValidationError(
        f"checkpoint model metadata {key!r} is incompatible"
      )
  counters = _mapping(payload.get("counters", {}), "checkpoint counters")
  if any(
    not isinstance(value, int) or isinstance(value, bool) for value in counters.values()
  ):
    raise CheckpointValidationError("checkpoint counters must contain integers")
  schedule = _mapping(payload.get("schedule", {}), "checkpoint schedule")
  resolved_config = _mapping(
    payload.get("resolved_config", {}), "checkpoint resolved_config"
  )
  try:
    model.load_state_dict(dict(model_state), strict=True)
  except Exception as exc:
    raise CheckpointValidationError(
      f"checkpoint weights could not be restored: {exc}"
    ) from exc
  # Drop the raw file contents so no training optimizer/replay tensors outlive
  # this call, then move only the requested model/normalizers.
  del payload, model_state
  model.to(device)
  model.eval()
  return InferenceModel(
    model=model,
    schema=schema,
    settings=settings,
    counters=counters,
    schedule=schedule,
    resolved_config=resolved_config,
    teacher_hashes=hashes,
    control_contract=contract,
  )


__all__ = [
  "CHECKPOINT_VERSION",
  "CheckpointValidationError",
  "InferenceModel",
  "LifecycleState",
  "load_checkpoint",
  "load_inference_checkpoint",
  "save_checkpoint",
]
