"""Explicit, bounded capture/restore state for difficult-state branches.

This module deliberately does not copy an environment ``__dict__``.  Backends may
opt in with ``coverage_capture_fields``/``coverage_restore_fields``; the fallback
only handles the audited MuJoCo data fields.  Unknown active components are
rejected rather than silently reset.
"""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from mjlab.sim.sim_data import TorchArray

SNAPSHOT_LAYOUT = "x2-distillation-branch-state-v1"


class BranchStateError(RuntimeError):
  """The explicit state contract cannot be captured or restored."""


class BranchStateMismatch(BranchStateError):
  """Source and destination state layouts or values are incompatible."""


DYNAMIC_FIELDS = (
  "time",
  "qpos",
  "qvel",
  "act",
  "history",
  "qacc_warmstart",
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
  "eq_active",
  "mocap_pos",
  "mocap_quat",
  "userdata",
)
MODEL_FIELDS = (
  "actuator_acc0",
  "actuator_biasprm",
  "actuator_forcerange",
  "actuator_gainprm",
  "body_inertia",
  "body_invweight0",
  "body_ipos",
  "body_iquat",
  "body_mass",
  "body_subtreemass",
  "dof_armature",
  "dof_damping",
  "dof_frictionloss",
  "dof_invweight0",
  "geom_aabb",
  "geom_friction",
  "geom_rbound",
  "geom_size",
  "jnt_actfrcrange",
  "tendon_actfrcrange",
  "tendon_invweight0",
  "tendon_length0",
)


@dataclass(frozen=True, slots=True)
class BranchState:
  """Owned state for one source row; tensors are detached copies."""

  fields: dict[str, Any]
  manager_state: dict[str, Any] = field(default_factory=dict)
  buffer_state: dict[str, Any] = field(default_factory=dict)
  rng_state: dict[str, Any] = field(default_factory=dict)
  metadata: dict[str, Any] = field(default_factory=dict)
  layout: str = SNAPSHOT_LAYOUT
  capture_instant: str = "cached_pre_step"

  def __post_init__(self) -> None:
    if self.layout != SNAPSHOT_LAYOUT:
      raise BranchStateMismatch(f"unsupported snapshot layout {self.layout!r}")
    if self.capture_instant not in ("cached_pre_step", "raw_reached_pre_reset"):
      raise BranchStateMismatch(f"unsupported capture instant {self.capture_instant!r}")
    object.__setattr__(self, "fields", _owned(self.fields))
    object.__setattr__(self, "manager_state", _owned(self.manager_state))
    object.__setattr__(self, "buffer_state", _owned(self.buffer_state))
    object.__setattr__(self, "rng_state", _owned(self.rng_state))
    object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
    _check_finite(self.fields)

  @classmethod
  def capture(
    cls,
    owner: Any,
    *,
    env_index: int | None = None,
    capture_instant: str = "cached_pre_step",
    metadata: Mapping[str, Any] | None = None,
  ) -> "BranchState":
    if _is_manager_env(owner) and not callable(
      getattr(owner, "coverage_capture_fields", None)
    ):
      from .x2_state_adapter import X2FixedTopologyStateAdapter

      fields, manager, buffers = X2FixedTopologyStateAdapter(
        owner, source_row=env_index
      ).capture()
    else:
      fields, manager, buffers = _capture_explicit(owner, env_index)
    return cls(
      fields,
      manager,
      buffers,
      _capture_rng(owner),
      dict(metadata or {}),
      capture_instant=capture_instant,
    )

  def restore(
    self,
    destination: Any,
    *,
    env_index: int | None = None,
    preserve_destination_rng: bool = True,
  ) -> None:
    if _is_manager_env(destination) and not callable(
      getattr(destination, "coverage_restore_fields", None)
    ):
      from .x2_state_adapter import X2FixedTopologyStateAdapter

      if env_index is None:
        raise BranchStateError("X2 restore requires destination row")
      X2FixedTopologyStateAdapter(destination, source_row=env_index).restore(
        self.fields,
        self.manager_state,
        self.buffer_state,
        destination_row=env_index,
      )
    else:
      _restore_explicit(
        destination, self.fields, self.manager_state, self.buffer_state, env_index
      )
    if not preserve_destination_rng:
      _restore_rng(destination, self.rng_state)

  def assert_policy_view(self, other: "BranchState", *, atol: float = 2e-6) -> None:
    """Compare cached policy inputs when both snapshots expose them."""
    if set(self.fields) != set(other.fields):
      raise BranchStateMismatch("cached policy view inventory differs")
    for name in (
      "teacher_actor_view",
      "student_conditioning",
      "student_reference",
      "previous_action",
    ):
      if name not in self.fields:
        raise BranchStateMismatch(f"cached policy view field is missing: {name}")
      a, b = self.fields[name], other.fields[name]
      if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if not torch.equal(a, b):
          raise BranchStateMismatch(f"cached policy view mismatch in {name}")
      elif a != b:
        raise BranchStateMismatch(f"cached policy view mismatch in {name}")


def _is_manager_env(owner: Any) -> bool:
  return all(
    hasattr(owner, name)
    for name in ("sim", "scene", "action_manager", "observation_manager")
  )


def _tensor_value(value: Any) -> torch.Tensor | None:
  if isinstance(value, torch.Tensor):
    return value
  if isinstance(value, TorchArray):
    return value._tensor
  return None


def _owned(value: Any) -> Any:
  tensor = _tensor_value(value)
  if tensor is not None:
    return tensor.detach().clone()
  if isinstance(value, np.ndarray):
    return value.copy()
  if isinstance(value, dict):
    return {key: _owned(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_owned(item) for item in value]
  if isinstance(value, tuple):
    return tuple(_owned(item) for item in value)
  return copy.deepcopy(value)


def _check_finite(value: Any, path: str = "state") -> None:
  tensor = _tensor_value(value)
  if tensor is not None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
      raise BranchStateError(f"non-finite {path}")
  elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
    if not bool(np.isfinite(value).all()):
      raise BranchStateError(f"non-finite {path}")
  elif isinstance(value, dict):
    for key, item in value.items():
      _check_finite(item, f"{path}.{key}")
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      _check_finite(item, f"{path}[{index}]")
  elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
    raise BranchStateError(f"non-finite {path}")
  elif isinstance(value, (int, np.integer, str, bool, type(None))):
    return


def _call(owner: Any, name: str, *args: Any) -> Any:
  method = getattr(owner, name, None)
  if method is None:
    return None
  if not callable(method):
    raise BranchStateError(f"{name} must be callable")
  return method(*args)


def _select(value: Any, index: int | None) -> Any:
  if index is None or not isinstance(value, (torch.Tensor, np.ndarray)):
    return value
  if value.ndim and value.shape[0] > index:
    return value[index]
  return value


def _capture_explicit(
  owner: Any, index: int | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  custom = _call(owner, "coverage_capture_fields", index)
  if custom is not None:
    if not isinstance(custom, tuple) or len(custom) != 3:
      raise BranchStateError(
        "coverage_capture_fields must return fields, managers, buffers"
      )
    fields, manager, buffers = custom
    if not all(isinstance(item, dict) for item in custom):
      raise BranchStateError("coverage capture groups must be dictionaries")
    return _owned(fields), _owned(manager), _owned(buffers)
  data = getattr(getattr(owner, "sim", owner), "data", None)
  if data is None:
    data = getattr(owner, "data", None)
  if data is None:
    raise BranchStateError("backend has no explicit coverage state handler or sim.data")
  raise BranchStateError(
    "generic fallback is disabled: installed backend requires explicit field inventory"
  )


def _capture_named_components(owner: Any, index: int | None) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for name in (
    "action_manager",
    "command_manager",
    "observation_manager",
    "event_manager",
    "scene",
  ):
    component = getattr(owner, name, None)
    if component is None:
      continue
    captured = _call(component, "coverage_capture_state", index)
    if captured is not None:
      result[name] = captured
  return result


def _capture_buffers(owner: Any) -> dict[str, Any]:
  result: dict[str, Any] = {}
  custom = _call(owner, "coverage_capture_buffers")
  if isinstance(custom, dict):
    return _owned(custom)
  for name in ("action_manager", "observation_manager", "scene"):
    component = getattr(owner, name, None)
    if component is None:
      continue
    captured = _call(component, "coverage_capture_buffers")
    if captured is not None:
      result[name] = captured
  return result


def _restore_explicit(
  owner: Any,
  fields: dict[str, Any],
  managers: dict[str, Any],
  buffers: dict[str, Any],
  index: int | None,
) -> None:
  custom = getattr(owner, "coverage_restore_fields", None)
  if callable(custom):
    custom(_owned(fields), _owned(managers), _owned(buffers), index)
    return
  data = getattr(getattr(owner, "sim", owner), "data", None)
  if data is None:
    data = getattr(owner, "data", None)
  if data is None:
    raise BranchStateError("destination has no explicit restore handler")
  for name, value in fields.items():
    target = (
      getattr(getattr(owner, "sim", owner), "model", None)
      if name in MODEL_FIELDS
      else data
    )
    if not hasattr(target, name):
      raise BranchStateMismatch(f"destination is missing field {name}")
    destination = getattr(target, name)
    source = value
    if (
      index is not None
      and isinstance(destination, (torch.Tensor, np.ndarray))
      and destination.ndim
      and destination.shape[0] > index
    ):
      destination[index] = source
    elif isinstance(destination, torch.Tensor) and isinstance(source, torch.Tensor):
      destination.copy_(source)
    elif isinstance(destination, np.ndarray):
      destination[...] = source
    else:
      setattr(target, name, _owned(source))
  for name, value in managers.items():
    component = getattr(owner, name, None)
    if component is None:
      raise BranchStateMismatch(f"destination missing manager {name}")
    restore = getattr(component, "coverage_restore_state", None)
    if not callable(restore):
      raise BranchStateError(f"active manager {name} has no restore handler")
    restore(_owned(value), index)
  for name, value in buffers.items():
    component = getattr(owner, name, None)
    restore = (
      getattr(component, "coverage_restore_buffers", None)
      if component is not None
      else None
    )
    if not callable(restore):
      raise BranchStateError(f"active buffer owner {name} has no restore handler")
    restore(_owned(value))


def _capture_rng(owner: Any) -> dict[str, Any]:
  state: dict[str, Any] = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "owner": _owned(_call(owner, "coverage_capture_rng")),
  }
  if torch.cuda.is_initialized():
    state["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
  return state


def _restore_rng(owner: Any, state: Mapping[str, Any]) -> None:
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"])
  if "cuda" in state and torch.cuda.is_initialized():
    torch.cuda.set_rng_state_all(state["cuda"])
  if state.get("owner") is not None:
    _call(owner, "coverage_restore_rng", _owned(state["owner"]))


@contextmanager
def isolated_branch_rng(seed: int) -> Iterator[np.random.Generator]:
  """Run branch construction/stepping on independent initialized streams."""
  saved = _capture_rng(object())
  try:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
      torch.cuda.manual_seed_all(seed)
    yield np.random.default_rng(seed)
  finally:
    _restore_rng(object(), saved)


def restore_field_parity(
  source: BranchState,
  destination: BranchState,
  *,
  atol: float = 1e-5,
  rtol: float = 1e-5,
) -> None:
  if set(source.fields) != set(destination.fields):
    raise BranchStateMismatch("captured field union differs")
  for name in source.fields:
    a, b = source.fields[name], destination.fields[name]
    a_tensor, b_tensor = _tensor_value(a), _tensor_value(b)
    if a_tensor is not None and b_tensor is not None:
      if not torch.allclose(a_tensor, b_tensor, atol=atol, rtol=rtol):
        raise BranchStateMismatch(f"field parity failed: {name}")
    elif a_tensor is not None or b_tensor is not None:
      raise BranchStateMismatch(f"field parity failed: {name}")
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
      if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise BranchStateMismatch(f"field parity failed: {name}")
    elif a != b:
      raise BranchStateMismatch(f"field parity failed: {name}")
