"""D4a sparse hard-keyframe inpainting for the frozen 41x167 continuation window.

This module owns one seam: turning sparse, absolute-tick
:class:`TimedStateKeyframe` targets into an independent
:class:`~.handover.ContinuationPlan` whose target rows have their complete 135
normalized physical-state slice fixed at clean timestep id 0 while their 32
latent entries stay unknown (noisy at the reverse source level).

Deliberately absent, because they are separate authorities:

* no dense reference tracking, no future encoder latent, no learned mask
  embedding, and no partial-state hard mask (partial pose objectives belong to
  the soft pose cost in ``pose_costs.py``);
* no simulation, controller or learned-model code path, and no mutation of the
  caller's plan, history or measured state;
* keyframes are task inputs.  They are rebased into the *current measured*
  window-center frame with the frozen :func:`~.state.world_to_hybrid`; they are
  never pushed into :class:`~.handover.HandoverHistory` and never replace the
  measured past/current conditioning rows.

An absolute world target keeps its tick and world pose across replanning: it is
re-derived against whatever the current measured root position, yaw and velocity
are, so it is neither shifted forward nor independently centered on itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .handover import ContinuationPlan
from .projection import ProjectionBundle
from .state import StateValidationError, WorldState, world_to_hybrid

_ACTIVE_SCHEMA = "mjlab-x2-state-latent-d1-no-mixtures-v1"

# Minimum scheduling spacing between two hard keyframes, in control ticks.
# The frozen control period is 0.02 s, so this is 0.2 s of motion.
_MINIMUM_KEYFRAME_SPACING = 10

_STATE_ARRAY_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)

_PAYLOAD_FIELDS = frozenset(
  {
    "tick",
    "provenance",
    "contract_schema",
    "contract_identity",
    "body_names",
    "state",
  }
)


class InpaintingError(ValueError):
  """A keyframe, plan or contract at the inpainting seam is invalid."""


def _require_active_contract(contract: DiffusionContract) -> None:
  """Reject legacy projected/legacy-schema contracts at the seam."""
  if contract.projection_rows != 0:
    raise InpaintingError(
      "hard-keyframe inpainting requires the active no-mixture 167-channel "
      "contract; legacy 231-channel projected contracts are not accepted"
    )
  if contract.schema_version != _ACTIVE_SCHEMA:
    raise InpaintingError(
      f"legacy contract schema {contract.schema_version!r} is not accepted by "
      "hard-keyframe inpainting"
    )


def _require_matching_projection(
  projection: ProjectionBundle, contract: DiffusionContract
) -> None:
  projection_contract = getattr(projection, "contract", None)
  if projection_contract is None or not hasattr(projection_contract, "identity_hash"):
    raise InpaintingError("projection bundle does not carry its diffusion contract")
  if projection_contract.identity_hash() != contract.identity_hash():
    raise InpaintingError(
      "projection bundle contract identity differs from the inpainting contract"
    )


def _require_tick(value: object, name: str) -> int:
  """Return ``value`` as a non-negative integer control tick."""
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise InpaintingError(f"{name} must be a non-negative integer control tick")
  tick = int(value)
  if tick < 0:
    raise InpaintingError(f"{name} must be non-negative, got {tick}")
  return tick


@dataclass(frozen=True, slots=True)
class TimedStateKeyframe:
  """One sparse hard keyframe: absolute control tick, full world state, origin.

  The state is a complete unbatched six-array :class:`WorldState` in the frozen
  20-body order, so a partial-state or dense-reference keyframe cannot be
  expressed here.  The keyframe owns copies of all six arrays; callers may
  reuse their buffers afterwards.  This is an explicit task input, never a
  fabricated executed-history row.
  """

  tick: int
  state: WorldState
  provenance: str

  def __post_init__(self) -> None:
    object.__setattr__(self, "tick", _require_tick(self.tick, "keyframe tick"))
    if not isinstance(self.state, WorldState):
      raise InpaintingError(
        "keyframe state must be a complete WorldState with the six physical arrays"
      )
    if self.state.root_position.ndim != 1:
      raise InpaintingError("keyframe state must be one unbatched measurement")
    if not isinstance(self.provenance, str) or not self.provenance.strip():
      raise InpaintingError(
        "keyframe provenance must be a non-empty source label (recording, "
        "selection or fixture it came from)"
      )
    # Own snapshots of all six arrays: a caller may keep mutating the sensor or
    # fixture buffers it built the state from.
    object.__setattr__(
      self,
      "state",
      WorldState(
        **{
          name: np.array(getattr(self.state, name), dtype=np.float64, copy=True)
          for name in _STATE_ARRAY_FIELDS
        }
      ),
    )

  def as_dict(
    self, *, contract: DiffusionContract = DEFAULT_CONTRACT
  ) -> dict[str, Any]:
    """Return a JSON-compatible payload bound to a contract and body order.

    The payload carries the contract identity and ordered body names so a
    deserializer can reject a keyframe authored against a different contract
    instead of silently reinterpreting the channels.
    """
    _require_active_contract(contract)
    return {
      "tick": self.tick,
      "provenance": self.provenance,
      "contract_schema": contract.schema_version,
      "contract_identity": contract.identity_hash(),
      "body_names": list(contract.body_names),
      "state": {
        name: np.asarray(getattr(self.state, name), dtype=np.float64).tolist()
        for name in _STATE_ARRAY_FIELDS
      },
    }

  @classmethod
  def from_dict(
    cls, payload: Mapping[str, Any], *, contract: DiffusionContract = DEFAULT_CONTRACT
  ) -> "TimedStateKeyframe":
    """Rebuild one keyframe, rejecting unknown fields and identity mismatches."""
    _require_active_contract(contract)
    if not isinstance(payload, Mapping):
      raise InpaintingError("keyframe payload must be a mapping")
    unknown = sorted(set(payload) - _PAYLOAD_FIELDS)
    if unknown:
      raise InpaintingError(f"keyframe payload has unknown fields {unknown}")
    missing = sorted(_PAYLOAD_FIELDS - set(payload))
    if missing:
      raise InpaintingError(f"keyframe payload is missing fields {missing}")
    if payload["contract_identity"] != contract.identity_hash():
      raise InpaintingError(
        "keyframe payload contract identity differs from the active contract"
      )
    if payload["contract_schema"] != contract.schema_version:
      raise InpaintingError(
        "keyframe payload contract schema differs from the contract"
      )
    body_names = payload["body_names"]
    if (
      not isinstance(body_names, (list, tuple))
      or tuple(body_names) != contract.body_names
    ):
      raise InpaintingError(
        "keyframe payload body order differs from the active contract body order"
      )
    state = payload["state"]
    if not isinstance(state, Mapping) or set(state) != set(_STATE_ARRAY_FIELDS):
      raise InpaintingError(
        "keyframe payload state must carry exactly the six physical arrays"
      )
    try:
      world = WorldState(
        **{
          name: np.asarray(state[name], dtype=np.float64)
          for name in _STATE_ARRAY_FIELDS
        }
      )
    except StateValidationError as exc:
      raise InpaintingError(f"malformed keyframe state: {exc}") from exc
    if world.root_position.ndim != 1:
      raise InpaintingError("keyframe state must be one unbatched measurement")
    return cls(tick=payload["tick"], state=world, provenance=payload["provenance"])


@dataclass(frozen=True, slots=True)
class ActiveKeyframeTarget:
  """One keyframe that lands inside the plan's ``1..future_steps`` window."""

  tick: int
  offset: int
  row: int


def _validated_keyframes(
  keyframes: Iterable[TimedStateKeyframe],
) -> tuple[TimedStateKeyframe, ...]:
  """Return the keyframes sorted by tick after duplicate/spacing validation.

  Duplicate ticks and pairs closer than :data:`_MINIMUM_KEYFRAME_SPACING` ticks
  are input errors, not silently deduplicated or merged: the scheduled
  hard-keyframe list is a task-level contract.
  """
  if keyframes is None or isinstance(keyframes, (str, bytes, Mapping)):
    raise InpaintingError(
      "keyframes must be an iterable of TimedStateKeyframe, not a scalar or mapping"
    )
  try:
    items = tuple(keyframes)
  except TypeError as exc:
    raise InpaintingError(
      "keyframes must be an iterable of TimedStateKeyframe"
    ) from exc
  for item in items:
    if not isinstance(item, TimedStateKeyframe):
      raise InpaintingError("keyframes must contain only TimedStateKeyframe values")
  ticks = [item.tick for item in items]
  if len(set(ticks)) != len(ticks):
    raise InpaintingError("duplicate keyframe target ticks are not accepted")
  ordered = tuple(sorted(items, key=lambda item: item.tick))
  for previous, following in zip(ordered, ordered[1:], strict=False):
    if following.tick - previous.tick < _MINIMUM_KEYFRAME_SPACING:
      raise InpaintingError(
        f"keyframe targets {previous.tick} and {following.tick} are closer than the "
        f"{_MINIMUM_KEYFRAME_SPACING}-tick minimum spacing"
      )
  return ordered


def _validated_plan_tick(plan: ContinuationPlan, contract: DiffusionContract) -> int:
  """Validate the base continuation window and return its absolute tick."""
  if not isinstance(plan, ContinuationPlan):
    raise InpaintingError("plan must be a ContinuationPlan built by the D3 seam")
  tick = _require_tick(plan.tick, "plan tick")
  tokens, mask = plan.clean_tokens, plan.clean_mask
  shape = (contract.window_steps, contract.token_dimension)
  if not isinstance(tokens, Tensor) or not tokens.is_floating_point():
    raise InpaintingError("plan clean_tokens must be a floating-point torch tensor")
  if tuple(tokens.shape) != shape:
    raise InpaintingError(
      f"plan clean_tokens must have shape {shape}, got {tuple(tokens.shape)}"
    )
  if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
    raise InpaintingError("plan clean_mask must be a bool torch tensor")
  if tuple(mask.shape) != shape:
    raise InpaintingError(
      f"plan clean_mask must have shape {shape}, got {tuple(mask.shape)}"
    )
  state_dimension = contract.projected_state_dimension
  if not bool(mask[: contract.current_index].all()):
    raise InpaintingError(
      "plan must fix every past history row (state and executed latent); "
      "inpainting never replaces measured history"
    )
  current_row = mask[contract.current_index]
  if not bool(current_row[:state_dimension].all()) or bool(
    current_row[state_dimension:].any()
  ):
    raise InpaintingError(
      "plan must fix exactly the current physical-state slice and leave the "
      "current latent unknown"
    )
  return tick


def active_keyframe_targets(
  plan: ContinuationPlan,
  keyframes: Iterable[TimedStateKeyframe],
  *,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> tuple[ActiveKeyframeTarget, ...]:
  """Return the bounded active targets of ``keyframes``, sorted by tick.

  Active means ``1 <= tick - plan.tick <= future_steps``; reached, expired and
  beyond-horizon targets are inactive and are reported by omission.  The
  returned rows are ``current_index + offset`` in the plan window, so callers
  can record bounded active goal IDs without reading any tensor.
  """
  _require_active_contract(contract)
  plan_tick = _validated_plan_tick(plan, contract)
  ordered = _validated_keyframes(keyframes)
  active: list[ActiveKeyframeTarget] = []
  for keyframe in ordered:
    offset = keyframe.tick - plan_tick
    if 1 <= offset <= contract.future_steps:
      active.append(
        ActiveKeyframeTarget(
          tick=keyframe.tick,
          offset=offset,
          row=contract.current_index + offset,
        )
      )
  return tuple(active)


def _normalized_target_row(
  current_state: WorldState,
  keyframe: TimedStateKeyframe,
  projection: ProjectionBundle,
  contract: DiffusionContract,
) -> np.ndarray:
  """Rebase one world keyframe into the current frame and normalize it.

  ``world_to_hybrid`` with ``center_index=0`` expresses the keyframe relative to
  the current measured root position, yaw and velocity, exactly as
  :func:`~.handover.build_reference_free_plan` expresses the measured rows
  relative to the current state.  Centering on the keyframe itself (its own
  frame) would discard the absolute world goal and is deliberately not done.
  """
  hybrid, _context = world_to_hybrid(
    WorldState.stack([current_state, keyframe.state]), center_index=0
  )
  # Row 0 is the centered current measurement; row 1 is the keyframe rebased
  # into that same measured frame.
  normalized = np.asarray(projection.project_state(hybrid[1:2]), dtype=np.float64)
  width = contract.projected_state_dimension
  if normalized.shape != (1, width) or not np.isfinite(normalized).all():
    raise InpaintingError(
      f"normalized keyframe state must be finite with shape [{width}]"
    )
  return np.ascontiguousarray(normalized[0], dtype=np.float32)


def inpaint_keyframes(
  plan: ContinuationPlan,
  current_state: WorldState,
  keyframes: Iterable[TimedStateKeyframe],
  *,
  projection: ProjectionBundle,
  contract: DiffusionContract = DEFAULT_CONTRACT,
) -> ContinuationPlan:
  """Return an independent plan with the active hard keyframes fixed.

  Every existing fixed entry is preserved bit-for-bit.  An active target fixes
  only its 135 normalized physical-state entries: the 32 latent entries of that
  row stay unknown, so the sampler still generates the latent at the reverse
  source level while the state is hard conditioning.  Fixed data that already
  conflicts with a requested keyframe is rejected, never overwritten; repeating
  the same request on an already-inpainted plan is a no-op.
  """
  _require_active_contract(contract)
  _require_matching_projection(projection, contract)
  plan_tick = _validated_plan_tick(plan, contract)
  if not isinstance(current_state, WorldState):
    raise InpaintingError("current state must be a complete WorldState")
  if current_state.root_position.ndim != 1:
    raise InpaintingError("current state must be one unbatched measurement")
  targets = active_keyframe_targets(plan, keyframes, contract=contract)
  by_tick = {keyframe.tick: keyframe for keyframe in _validated_keyframes(keyframes)}

  state_dimension = contract.projected_state_dimension
  tokens = plan.clean_tokens.clone()
  mask = plan.clean_mask.clone()
  for target in targets:
    requested = torch.from_numpy(
      _normalized_target_row(current_state, by_tick[target.tick], projection, contract)
    ).to(device=tokens.device, dtype=tokens.dtype)
    row = target.row
    if bool(mask[row, state_dimension:].any()):
      raise InpaintingError(
        f"row {row} already fixes latent entries; a hard keyframe fixes only the "
        "physical-state slice, so an already-conditioned future latent is "
        "refused rather than kept or overwritten"
      )
    fixed_state = mask[row, :state_dimension]
    if bool(fixed_state.any()) and not torch.equal(
      tokens[row, :state_dimension][fixed_state], requested[fixed_state]
    ):
      raise InpaintingError(
        f"row {row} already holds fixed conditioning conflicting with the "
        f"keyframe at tick {target.tick}; refusing to overwrite it"
      )
    tokens[row, :state_dimension] = requested
    mask[row, :state_dimension] = True
  return ContinuationPlan(tick=plan_tick, clean_tokens=tokens, clean_mask=mask)


__all__ = [
  "ActiveKeyframeTarget",
  "InpaintingError",
  "TimedStateKeyframe",
  "active_keyframe_targets",
  "inpaint_keyframes",
]
