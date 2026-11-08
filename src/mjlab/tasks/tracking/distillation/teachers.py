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
  CohortContract,
  DistillationError,
  UnsupportedTeacherError,
  load_actor_state_dict,
)

_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


class TeacherBankError(ValueError):
  """A routing code names a teacher the bank does not hold."""


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

  A bank may hold only part of a cohort's teachers: one collection shard of a
  large cohort registers its own slice, because one actor per member is what
  makes a whole-cohort bank too large for one card.  The code table stays
  cohort-wide in that case (``codes``), so a shard labels rows with the same
  codes every other shard and the parent use, and a code it does not hold is
  refused instead of being routed to a neighbour or dropped silently.
  """

  def __init__(
    self,
    teachers: Sequence[FrozenTeacher],
    device: str | torch.device = "cpu",
    *,
    codes: Sequence[int] | None = None,
    teacher_ids: Sequence[str] | None = None,
    groups: Sequence[int] | None = None,
  ) -> None:
    """Register ``teachers`` under their routing codes.

    ``codes[i]`` is the cohort-wide routing code of ``teachers[i]`` and
    ``teacher_ids`` is the full code-order id tuple of the cohort the bank
    belongs to.  Both default to the held teachers' own positions and ids, so a
    bank over the whole cohort is unchanged.  The full id tuple is kept even
    when only some of its codes are held: a report that names a code needs the
    name, not the module.
    """
    super().__init__()
    if not teachers:
      raise ValueError("TeacherBank requires at least one teacher")
    held = list(range(len(teachers))) if codes is None else list(codes)
    # Codes that resolve to equal weights may share one labeling call.  The
    # grouping is decided by the builder, which compares the actor states, so
    # the bank never guesses: an absent ``groups`` gives every code its own
    # group, which is the one-call-per-code behaviour.
    if groups is None:
      held_groups = list(range(len(held)))
    else:
      held_groups = list(groups)
      if len(held_groups) != len(held):
        raise ValueError(
          f"got {len(teachers)} teachers but {len(held_groups)} grouping labels"
        )
    if len(held) != len(teachers):
      raise ValueError(f"got {len(teachers)} teachers but {len(held)} routing codes")
    for code in held:
      if isinstance(code, bool) or not isinstance(code, int) or code < 0:
        raise ValueError(f"Teacher codes must be non-negative integers, got {code!r}")
    if len(set(held)) != len(held):
      raise ValueError(f"Teacher codes must be unique, got {sorted(held)}")
    ids = (
      [teacher.teacher_id for teacher in teachers]
      if teacher_ids is None
      else list(teacher_ids)
    )
    if len(set(ids)) != len(ids):
      raise ValueError(f"Teacher ids must be unique, got {ids}")
    if any(not teacher_id for teacher_id in ids):
      raise ValueError(f"Teacher ids must be non-empty, got {ids}")
    dims = {(teacher.obs_dim, teacher.action_dim) for teacher in teachers}
    if len(dims) != 1:
      raise DistillationError(
        f"All teachers in a bank must share (obs_dim, action_dim), got {sorted(dims)}"
      )
    if any("." in teacher_id for teacher_id in ids):
      raise ValueError(f"Teacher ids must not contain '.': {ids}")
    for code, teacher in zip(held, teachers, strict=True):
      if code >= len(ids) or ids[code] != teacher.teacher_id:
        raise ValueError(
          f"teacher {teacher.teacher_id!r} cannot carry routing code {code}: the "
          f"code order {ids} assigns that code to a different teacher"
        )

    self._ids: tuple[str, ...] = tuple(ids)
    self._codes = {teacher_id: code for code, teacher_id in enumerate(self._ids)}
    self._held_codes = frozenset(held)
    self.obs_dim, self.action_dim = next(iter(dims))
    for code, teacher in zip(held, teachers, strict=True):
      self.add_module(f"teacher_{code}", teacher)
    # Row groups are resolved by the module a code lands on, not by the code
    # itself.  A clip-sharded cohort of one shared export gives thousands of
    # codes to the same frozen weights, so grouping by code would run an
    # identical actor once per member per tick (measured: 8.5 s of a 22.5 s
    # worker iteration at 466 members x 32 ticks).
    group_of_held = dict(zip(held, held_groups, strict=True))
    self._group_of_code = tuple(group_of_held.get(code, -1) for code in range(len(ids)))
    self.to(torch.device(device))

  @property
  def device(self) -> torch.device:
    """Device of the registered teachers, read from their actual parameters.

    A device cached at construction goes stale after ``.to(...)`` and would then
    reject observations and codes that are on the teachers' real device.
    """
    return next(self.parameters()).device

  @property
  def teacher_ids(self) -> tuple[str, ...]:
    """Teacher ids in code order: ``teacher_ids[code]`` names that teacher."""
    return self._ids

  def code(self, teacher_id: str) -> int:
    """Return the integer routing code of ``teacher_id``."""
    try:
      return self._codes[teacher_id]
    except KeyError as exc:
      raise ValueError(
        f"Unknown teacher id {teacher_id!r}; bank has {list(self._ids)}"
      ) from exc

  def teacher(self, code: int) -> FrozenTeacher:
    """Return the frozen teacher registered under ``code``.

    Raises:
      TeacherBankError: if this bank holds only part of the cohort and the code
        belongs to a teacher it does not hold.
    """
    self.teacher_id(code)
    if code not in self._held_codes:
      raise TeacherBankError(
        f"this teacher bank does not hold code {code} ({self._ids[code]!r}); it "
        f"holds {sorted(self._held_codes)}. A shard labels rows with cohort-wide "
        "codes but can only resolve the clips it owns"
      )
    module = getattr(self, f"teacher_{code}")
    assert isinstance(module, FrozenTeacher)
    return module

  @property
  def held_codes(self) -> frozenset[int]:
    """Codes this bank holds a teacher for, cohort-wide codes otherwise."""
    return self._held_codes

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
    out_of_range = (teacher_ids < 0) | (teacher_ids >= len(self._ids))
    if bool(out_of_range.any().item()):
      raise ValueError(
        f"Teacher code {int(teacher_ids[out_of_range][0].item())} is out of range "
        f"for {len(self._ids)} teachers {list(self._ids)}"
      )
    groups = torch.tensor(
      self._group_of_code, dtype=torch.int64, device=observations.device
    )[teacher_ids]
    for group in torch.unique(groups).tolist():
      rows = groups == group
      codes = teacher_ids[rows]
      code = int(codes.min().item())
      if not 0 <= code < len(self._ids):
        raise ValueError(
          f"Teacher code {code} is out of range for {len(self._ids)} teachers "
          f"{list(self._ids)}"
        )
      if code not in self._held_codes:
        raise TeacherBankError(
          f"cannot label routing code {code} ({self._ids[code]!r}): this teacher "
          f"bank holds {sorted(self._held_codes)}. A shard labels rows with "
          "cohort-wide codes but can only resolve the clips it owns"
        )
      actions[rows] = self.teacher(code).label(observations[rows])
    return actions


def build_cohort_teacher_bank(
  cohort: CohortContract,
  device: str | torch.device = "cpu",
  *,
  teacher_ids: Sequence[str] | None = None,
) -> TeacherBank:
  """Build one frozen bank over the cohort's teachers, in manifest order.

  Bank codes are positions in ``cohort.teachers``, which is exactly the routing
  code the multi-motion library assigns to a clip of that teacher.  Building the
  bank over the whole cohort (rather than only the selected teachers) keeps that
  identity stable when a caller selects a subset, so a snapshot's per-row teacher
  code can never select a teacher other than the one that trained on the row's
  clip.

  ``teacher_ids`` restricts which teachers are *registered*, not how they are
  numbered: this is how one collection shard of a large clip-sharded cohort
  holds only its own slice of the frozen actors, which is the measured
  per-member cost.  The codes stay cohort-wide, so both kinds of shard label
  rows identically and a code outside the slice raises
  :class:`TeacherBankError` instead of labelling with another teacher.
  """
  # The code of a teacher is its position in the cohort, so the mapping is
  # built once: a subset of a large cohort is thousands of members.
  codes = {teacher.id: position for position, teacher in enumerate(cohort.teachers)}
  if teacher_ids is None:
    held = tuple(cohort.teachers)
  else:
    requested = tuple(teacher_ids)
    if not requested:
      raise ValueError("a teacher subset needs at least one teacher id")
    if len(set(requested)) != len(requested):
      raise ValueError(f"teacher subset repeats a teacher id: {requested}")
    unknown = [teacher_id for teacher_id in requested if teacher_id not in codes]
    if unknown:
      raise ValueError(f"cohort has no teacher(s) {unknown}")
    wanted = set(requested)
    # Keep manifest order: the code of a teacher is its cohort position.
    held = tuple(teacher for teacher in cohort.teachers if teacher.id in wanted)
  return TeacherBank(
    [
      build_frozen_teacher(
        teacher.id, teacher.actor_state_dict, teacher.actor, device=device
      )
      for teacher in held
    ],
    device=device,
    codes=[codes[teacher.id] for teacher in held],
    teacher_ids=list(cohort.teacher_ids),
    groups=_equivalent_actor_groups(held),
  )


def _equivalent_actor_groups(held: Sequence[Any]) -> list[int]:
  """Group cohort members whose actor weights are equal.

  A cohort that names one shared export gives every member the same weights, so
  labeling by code ran the identical actor once per member per tick (measured:
  8.5 s of a 22.5 s worker iteration at 466 members x 32 ticks).  Members keep
  their own module and their own ``teacher_id``; only the labeling call is
  shared, which cannot change a label because the weights are compared equal
  here.  An unequal pair simply keeps two groups, so a cohort of distinct actors
  is unaffected, and this is not a provenance gate: nothing is recorded.
  """
  groups: list[int] = []
  references: list[Mapping[str, torch.Tensor]] = []
  for teacher in held:
    group = None
    for index, reference in enumerate(references):
      if _same_actor_state(reference, teacher.actor_state_dict):
        group = index
        break
    if group is None:
      group = len(references)
      references.append(teacher.actor_state_dict)
    groups.append(group)
  return groups


def _same_actor_state(
  left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
  """Return whether two actor state dictionaries hold equal tensors.

  The first tensor decides most mismatches, so a cohort of distinct actors pays
  one small comparison per member rather than a full dictionary sweep.
  """
  if set(left) != set(right):
    return False
  names = sorted(left)
  for position, name in enumerate(names):
    first, second = left[name], right[name]
    if first.shape != second.shape or first.dtype != second.dtype:
      return False
    if not torch.equal(first.detach().to("cpu"), second.detach().to("cpu")):
      return False
    if position == 0 and len(names) == 1:
      return True
  return True


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
