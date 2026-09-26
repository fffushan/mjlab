"""Frozen actor-only teacher inference.

Teachers are loaded for labeling only: parameters are frozen, observation
normalizers are never updated, and actions are the deterministic distribution
mean. No PPO algorithm, critic, optimizer, or simulator is constructed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

from mjlab.tasks.tracking.distillation.config import (
  ActorArchitecture,
  DistillationError,
  UnsupportedTeacherError,
  load_actor_state_dict,
)

_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def build_actor_model(arch: ActorArchitecture) -> MLPModel:
  """Construct the empty actor described by ``arch``.

  A zero-filled observation TensorDict establishes the observation dimensions,
  so no simulator or environment is needed to build the model.
  """
  if len(arch.obs_groups) != 1:
    raise UnsupportedTeacherError(
      f"Actor observation groups {arch.obs_groups} cannot be labeled from a flat "
      "[B, D] observation vector"
    )
  obs = TensorDict({arch.obs_groups[0]: torch.zeros(1, arch.obs_dim)})
  return MLPModel(
    obs=obs,
    obs_groups={"actor": list(arch.obs_groups)},
    obs_set="actor",
    output_dim=arch.action_dim,
    hidden_dims=list(arch.hidden_dims),
    activation=arch.activation,
    obs_normalization=arch.obs_normalization,
    distribution_cfg=None
    if arch.distribution_cfg is None
    else dict(arch.distribution_cfg),
  )


class FrozenTeacher(nn.Module):
  """A teacher actor frozen for deterministic labeling.

  The actor keeps its own learned observation statistics; different teachers
  are expected to have different normalizers. Training-mode requests are
  ignored, so a surrounding student's ``.train()`` cannot re-enable normalizer
  updates or restore gradients.
  """

  def __init__(
    self,
    teacher_id: str,
    model: MLPModel,
    arch: ActorArchitecture,
    device: str | torch.device = "cpu",
  ) -> None:
    super().__init__()
    self.teacher_id = teacher_id
    self.obs_group = arch.obs_groups[0]
    self.obs_dim = arch.obs_dim
    self.action_dim = arch.action_dim
    self.model = model.to(torch.device(device))
    self.model.eval()
    self.model.requires_grad_(False)

  def train(self, mode: bool = True) -> FrozenTeacher:
    """Keep the frozen teacher in eval mode regardless of the requested mode."""
    del mode
    super().train(False)
    return self

  @property
  def device(self) -> torch.device:
    return next(self.parameters()).device

  def label(self, observations: torch.Tensor) -> torch.Tensor:
    """Return deterministic teacher actions for a batch of observations."""
    _require_observation_batch(observations, self.obs_dim, self.teacher_id)
    if observations.device != self.device:
      raise ValueError(
        f"observations are on {observations.device} but teacher "
        f"{self.teacher_id!r} is on {self.device}"
      )
    if self.model.training:
      raise RuntimeError(
        f"Frozen teacher {self.teacher_id!r} is in train mode; labeling would allow "
        "observation-normalizer updates to corrupt the saved statistics"
      )
    with torch.no_grad():
      # stochastic_output=False requests the deterministic (mean) action. PPO
      # exploration noise and action clipping are not part of teacher labels.
      return self.model(
        TensorDict({self.obs_group: observations}), stochastic_output=False
      )


class TeacherBank(nn.Module):
  """Deterministic batched routing over frozen teachers.

  Rows are labeled by the teacher selected for them; each needed teacher is
  evaluated once per batch and the original row order is preserved. Teacher
  codes are routing metadata only and never become model features.
  """

  def __init__(
    self,
    teachers: Sequence[FrozenTeacher],
    device: str | torch.device = "cpu",
  ) -> None:
    super().__init__()
    if not teachers:
      raise ValueError("TeacherBank requires at least one teacher")
    ids = [teacher.teacher_id for teacher in teachers]
    if len(set(ids)) != len(ids):
      raise ValueError(f"Teacher ids must be unique, got {ids}")
    dims = {(teacher.obs_dim, teacher.action_dim) for teacher in teachers}
    if len(dims) != 1:
      raise DistillationError(
        f"All teachers in a bank must share (obs_dim, action_dim), got {sorted(dims)}"
      )
    if any("." in teacher_id for teacher_id in ids):
      raise ValueError(f"Teacher ids must not contain '.': {ids}")

    self._ids: tuple[str, ...] = tuple(ids)
    self._codes = {teacher_id: code for code, teacher_id in enumerate(self._ids)}
    self.obs_dim, self.action_dim = next(iter(dims))
    for code, teacher in enumerate(teachers):
      self.add_module(f"teacher_{code}", teacher)
    self.to(torch.device(device))

  @property
  def device(self) -> torch.device:
    """Device of the registered teachers, read from their actual parameters.

    A device cached at construction goes stale after ``.to(...)`` and would then
    reject observations and codes that are on the teachers' real device.
    """
    return next(self.parameters()).device

  def code(self, teacher_id: str) -> int:
    """Return the integer routing code of ``teacher_id``."""
    try:
      return self._codes[teacher_id]
    except KeyError as exc:
      raise ValueError(
        f"Unknown teacher id {teacher_id!r}; bank has {list(self._ids)}"
      ) from exc

  def teacher(self, code: int) -> FrozenTeacher:
    """Return the frozen teacher registered under ``code``."""
    self.teacher_id(code)
    module = getattr(self, f"teacher_{code}")
    assert isinstance(module, FrozenTeacher)
    return module

  def teacher_id(self, code: int) -> str:
    if not 0 <= code < len(self._ids):
      raise ValueError(
        f"Teacher code {code} is out of range for {len(self._ids)} teachers"
      )
    return self._ids[code]

  def train(self, mode: bool = True) -> TeacherBank:
    """Keep every frozen teacher in eval mode regardless of the requested mode."""
    del mode
    super().train(False)
    return self

  def label(
    self, teacher_ids: torch.Tensor, observations: torch.Tensor
  ) -> torch.Tensor:
    """Label ``observations`` with each row's teacher, preserving row order.

    Args:
      teacher_ids: Integer routing codes of shape ``[B]``, from :meth:`code`.
      observations: Teacher observation vectors of shape ``[B, obs_dim]``.

    Returns:
      Actions of shape ``[B, action_dim]``. An empty batch returns an empty
      action tensor.
    """
    if not isinstance(teacher_ids, torch.Tensor):
      raise TypeError(f"teacher_ids must be a Tensor, got {type(teacher_ids).__name__}")
    if teacher_ids.ndim != 1:
      raise ValueError(f"teacher_ids must be 1-D, got shape {tuple(teacher_ids.shape)}")
    if teacher_ids.dtype not in _INTEGER_DTYPES:
      raise ValueError(
        f"teacher_ids must be an integer tensor, got dtype {teacher_ids.dtype}"
      )
    if teacher_ids.device != self.device:
      raise ValueError(
        f"teacher_ids are on {teacher_ids.device} but the bank is on {self.device}"
      )
    _require_observation_batch(observations, self.obs_dim, "bank")
    if observations.device != self.device:
      raise ValueError(
        f"observations are on {observations.device} but the bank is on {self.device}"
      )
    if teacher_ids.shape[0] != observations.shape[0]:
      raise ValueError(
        f"teacher_ids has {teacher_ids.shape[0]} rows but observations has "
        f"{observations.shape[0]}"
      )

    actions = observations.new_zeros((observations.shape[0], self.action_dim))
    for code in torch.unique(teacher_ids).tolist():
      if not 0 <= code < len(self._ids):
        raise ValueError(
          f"Teacher code {code} is out of range for {len(self._ids)} teachers "
          f"{list(self._ids)}"
        )
      rows = teacher_ids == code
      actions[rows] = self.teacher(int(code)).label(observations[rows])
    return actions


def build_frozen_teacher(
  teacher_id: str,
  actor_state_dict: Mapping[str, torch.Tensor],
  arch: ActorArchitecture,
  device: str | torch.device = "cpu",
) -> FrozenTeacher:
  """Build a frozen teacher from an already-loaded actor state dictionary."""
  model = build_actor_model(arch)
  try:
    model.load_state_dict(dict(actor_state_dict), strict=True)
  except RuntimeError as exc:
    raise DistillationError(
      f"Teacher {teacher_id!r} checkpoint does not match the resolved actor "
      f"architecture: {exc}"
    ) from exc
  return FrozenTeacher(teacher_id, model, arch, device)


def load_frozen_teacher(
  teacher_id: str,
  checkpoint_path: Path,
  arch: ActorArchitecture,
  device: str | torch.device = "cpu",
) -> FrozenTeacher:
  """Load a frozen teacher from its checkpoint path."""
  return build_frozen_teacher(
    teacher_id, load_actor_state_dict(checkpoint_path), arch, device
  )


def _require_observation_batch(
  observations: torch.Tensor,
  obs_dim: int,
  owner: str,
) -> None:
  if not isinstance(observations, torch.Tensor):
    raise TypeError(f"observations must be a Tensor, got {type(observations).__name__}")
  if observations.ndim != 2 or observations.shape[1] != obs_dim:
    raise ValueError(
      f"{owner} expects observations of shape [B, {obs_dim}], got "
      f"{tuple(observations.shape)}"
    )
