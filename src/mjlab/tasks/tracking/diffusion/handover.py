"""D3 handover seam: executed-pair history and reference-free continuation.

This module implements the bounded offline/runtime seam between the frozen D2
denoiser and the frozen X2 VAE decoder:

* control is handed over only after ``past_steps`` (8) state / latent pairs
  exist (contract ``startup.minimum_actual_pairs``); whether those pairs are
  genuinely measured/executed, contiguous in tick, and never sourced from a
  standing policy is a caller obligation — ``push`` accepts whatever pair the
  caller supplies, and the documented lifecycle requires pairing each recorded
  state ``s_t`` with the latent ``z_t`` that the decoder decoded *on* ``s_t``
  (applied at ``s_t``), not with the next tick's state ``s_(t+1)``;
* the conditioning window is rebuilt from raw history states in the current
  window-center frame at every tick (history is never cached across frames);
* past states/latents and the current physical state are hard conditioning at
  clean level 0; the current latent and all 32 future steps are generated;
* the seam has no reference/encoder call path of its own:
  :class:`ReferenceFreeContinuation` accepts no reference input and calls only
  the supplied frozen ``decode(latent, conditioning)`` callable with the 99-D
  gravity conditioning.  Structural provenance beyond that — measured vs.
  executed authenticity, standing-origin exclusion, and tick contiguity —
  cannot be authenticated by generic ``push``/``step`` inputs and remains a
  caller/adapter obligation.

Token layout stays the active no-mixture contract: 41 steps (8 past + current
at index 8 + 32 future) by the declared token width.  The frozen contract uses
167 channels (135 normalized physical-state + 32 normalized VAE latent), while
the opt-in local-kinematic contract uses 161 channels (129 + 32).

This module is deliberately simulator-free and claims no closed-loop, simulator,
export or hardware evidence; a controller or evaluation harness supplies
measured states and the frozen decoder callable.

Two optional D4 hooks let a task lane steer the frozen continuation without
duplicating this seam's history, tick-age guard, decoding or prediction logic:
``plan_transform`` rewrites the freshly built plan (for example hard-keyframe
inpainting) and ``sampler_factory`` supplies a per-tick replacement for
:func:`~.sampler.sample_trajectory` (for example pose-cost guidance).  Both
default to ``None``, which reproduces the D3 call sequence and sample exactly.
A sampler may return diagnostics carrying a bounded ``task`` mapping; it is
attached to :class:`ContinuationStep` for the evaluation report.

One further optional hook replaces the sampler for a different model family:
when ``sampler`` is supplied, :meth:`ReferenceFreeContinuation.step` calls it
with the built conditions and the shared generator in place of the frozen
diffusion ``sample_trajectory``, so a family adapter can close over its own
model, solver and level grid.  A continuation built with one then needs no
diffusion ``schedule`` or ``grid``; the default ``None`` keeps the D3 diffusion
path unchanged.

Two more optional lifecycle hooks stay reporting-only: ``on_reset`` runs right
after :meth:`ReferenceFreeContinuation.reset` clears history, so a task lane can
drop its own ledger before the mandatory fresh bootstrap, and ``measured_post``
receives one trusted measured post-step observation through
:meth:`ReferenceFreeContinuation.observe_measured_post` without sampling,
decoding, recording history or advancing the planning count.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from .contract import DEFAULT_CONTRACT, DiffusionContract
from .noising import clean_mask
from .projection import StateLatentTransformLike
from .sampler import (
  SamplerConditions,
  SamplerConfig,
  sample_trajectory,
)
from .schedule import DiffusionSchedule, InferenceGrid
from .state import (
  HYBRID_STATE,
  LOCAL_KINEMATIC_STATE,
  WorldState,
  hybrid_to_local_kinematic,
  world_to_hybrid,
)

# Frozen gravity-mode decoder conditioning width
# (tasks/tracking/distillation/vae_config.py CONDITIONING_DIMS["gravity"]).
_CONDITIONING_DIM = 99

# A task lane must not smuggle unbounded state or gradient tensors into the
# evaluation report through the per-step task record.
_MAX_TASK_RECORD_BYTES = 8192


class HandoverError(ValueError):
  """A handover history, decoder input or seam contract is invalid."""


class PlanAgeError(HandoverError):
  """A control tick was reused or regressed without an explicit reset."""


def _require_no_mixture(contract: DiffusionContract) -> None:
  """Reject any legacy projected (231-channel) contract at the seam."""
  if contract.projection_rows != 0:
    raise HandoverError(
      "the D3 seam requires a no-mixture contract; legacy projected "
      f"{contract.token_dimension}-channel contracts are not accepted"
    )


def _require_matching_projection(
  projection: StateLatentTransformLike, contract: DiffusionContract
) -> None:
  projection_contract = getattr(projection, "contract", None)
  if projection_contract is None or not hasattr(projection_contract, "identity_hash"):
    raise HandoverError("projection bundle does not carry its diffusion contract")
  if projection_contract.identity_hash() != contract.identity_hash():
    raise HandoverError(
      "projection bundle contract identity differs from the seam contract"
    )


_STATE_ARRAY_FIELDS = (
  "root_position",
  "root_quaternion_wxyz",
  "root_linear_velocity",
  "root_angular_velocity",
  "body_positions",
  "body_linear_velocities",
)


@dataclass(frozen=True, slots=True)
class ExecutedPair:
  """One state snapshot and the latent the decoder decoded on that state.

  Pairing semantics (``s_t`` with ``z_t`` applied on ``s_t``, not ``s_(t+1)``)
  and provenance are caller obligations; this dataclass only owns independent
  snapshots of both arrays so later caller buffer reuse cannot rewrite them.
  """

  state: WorldState
  latent: np.ndarray

  def __post_init__(self) -> None:
    if self.state.root_position.ndim != 1:
      raise HandoverError("executed-pair state must be one unbatched measurement")
    latent = np.asarray(self.latent, dtype=np.float64)
    if (
      latent.shape != (DEFAULT_CONTRACT.latent_dimension,)
      or not np.isfinite(latent).all()
    ):
      raise HandoverError("executed latent must be finite with shape [32]")
    # Own snapshots of all six arrays: callers may reuse their sensor buffers
    # between ticks, so stored history rows must not alias caller memory.
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
    object.__setattr__(self, "latent", latent.copy())


class HandoverHistory:
  """The bounded buffer of state/executed-latent pairs.

  ``push`` accepts the pair the caller supplies; there is no API that could
  insert a motion reference, an encoder output computed after the fact, or a
  standing-policy latent.  However, generic inputs cannot structurally prove
  that recorded pairs are measured rather than synthesized, or that ticks are
  contiguous: origin and contiguity are caller obligations, and the buffer
  must be cleared on every reset, teleport or policy switch.
  """

  def __init__(self, *, contract: DiffusionContract = DEFAULT_CONTRACT) -> None:
    self.contract = contract
    self._pairs: deque[ExecutedPair] = deque(maxlen=contract.past_steps)

  def push(self, state: WorldState, latent: Sequence[float] | np.ndarray) -> None:
    """Record one ``(state, latent)`` pair, keeping the last 8.

    The caller must pair each recorded state ``s_t`` with the latent that the
    decoder decoded on ``s_t`` (applied at ``s_t``), and must retain ``s_t``
    safely until this call so reused sensor buffers cannot rewrite it.
    """
    self._pairs.append(ExecutedPair(state, np.asarray(latent, dtype=np.float64)))

  def reset(self) -> None:
    """Clear all history; handover must be re-established from scratch."""
    self._pairs.clear()

  @property
  def ready(self) -> bool:
    return len(self._pairs) == self.contract.past_steps

  @property
  def pairs(self) -> tuple[ExecutedPair, ...]:
    return tuple(self._pairs)


@dataclass(frozen=True, slots=True)
class ContinuationPlan:
  """One reference-free conditioning snapshot at a control tick."""

  tick: int
  clean_tokens: Tensor
  clean_mask: Tensor


def build_reference_free_plan(
  history: HandoverHistory,
  current_state: WorldState,
  *,
  projection: StateLatentTransformLike,
  tick: int,
  contract: DiffusionContract = DEFAULT_CONTRACT,
  device: str | torch.device = "cpu",
) -> ContinuationPlan:
  """Build the 41x167 clean-conditioning window from history and current state.

  The nine stored states (8 history + current) are re-hybridized in the
  current window-center frame, so history is always expressed in the new
  current frame.  Future rows and the current latent stay zero in
  ``clean_tokens``; the clean mask marks them unknown so the sampler never
  reads those placeholder values.  State origin and tick contiguity are not
  verifiable here: the caller supplies all measured values.
  """
  _require_no_mixture(contract)
  _require_matching_projection(projection, contract)
  if not history.ready:
    raise HandoverError(
      "handover requires "
      f"{contract.past_steps} measured state/executed-latent pairs, "
      f"got {len(history.pairs)}"
    )
  if current_state.root_position.ndim != 1:
    raise HandoverError("current state must be one unbatched measurement")
  try:
    resolved = torch.device(device)
    torch.empty(0, device=resolved)
  except (RuntimeError, ValueError) as exc:
    raise HandoverError(f"invalid runtime device {device!r}") from exc
  states = WorldState.stack([*(pair.state for pair in history.pairs), current_state])
  hybrid, _context = world_to_hybrid(states, center_index=contract.current_index)
  fields = tuple(contract.state_fields)
  if fields == tuple(HYBRID_STATE.slices):
    packed_state = hybrid
  elif fields == tuple(LOCAL_KINEMATIC_STATE.slices):
    packed_state = hybrid_to_local_kinematic(hybrid)
  else:
    raise HandoverError(
      f"unsupported simulator state layout for contract {contract.schema_version!r}"
    )
  normalized_state = projection.project_state(packed_state)
  normalized_latent = projection.normalize_latent(
    np.stack([pair.latent for pair in history.pairs])
  )
  tokens = np.zeros((contract.window_steps, contract.token_dimension), dtype=np.float32)
  state_slice = slice(0, contract.projected_state_dimension)
  latent_slice = slice(contract.projected_state_dimension, contract.token_dimension)
  tokens[: contract.past_steps, state_slice] = normalized_state[: contract.past_steps]
  tokens[contract.current_index, state_slice] = normalized_state[contract.current_index]
  tokens[: contract.past_steps, latent_slice] = normalized_latent
  clean = torch.from_numpy(np.ascontiguousarray(tokens)).to(
    device=resolved, dtype=torch.float32
  )
  mask = clean_mask(
    current_index=contract.current_index,
    state_dimension=contract.projected_state_dimension,
    token_dimension=contract.token_dimension,
    device=resolved,
  )
  return ContinuationPlan(tick=int(tick), clean_tokens=clean, clean_mask=mask)


DecodeFn = Callable[[Tensor, Tensor], Tensor]
# A task lane rewrites the built plan with the measured current state, for
# example to fix sparse hard keyframes.
PlanTransform = Callable[["ContinuationPlan", WorldState], "ContinuationPlan"]
# A task lane supplies the per-tick sampler callable; it is invoked exactly like
# ``sample_trajectory`` and must return ``(sample, diagnostics)``.
SamplerCallable = Callable[..., tuple[Tensor, Any]]
SamplerFactory = Callable[["ContinuationPlan", WorldState], SamplerCallable]
# A non-diffusion family sampler injected into the seam.  It receives the built
# conditions and the shared generator and returns ``(sample, diagnostics)``,
# where the diagnostics expose ``denoiser_calls`` and a terminal
# ``final_index``.  The family model, solver and level grid are closed over by
# the callable, so a continuation built with one needs no diffusion schedule,
# inference grid or ``SamplerConfig``.
FamilySampler = Callable[[SamplerConditions, torch.Generator], tuple[Tensor, Any]]
# Called after ``reset`` clears history, before the mandatory fresh bootstrap.
ResetHook = Callable[[], None]
# Called with one trusted measured post-step (tick, state) observation.
MeasuredPostHook = Callable[[int, WorldState], None]


def _bounded_task_record(value: object) -> Mapping[str, Any] | None:
  """Return a bounded, JSON-serializable per-step task record or ``None``."""
  if value is None:
    return None
  if not isinstance(value, Mapping):
    raise HandoverError("a sampler task record must be a mapping")
  record = dict(value)
  try:
    encoded = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
  except (TypeError, ValueError) as exc:
    raise HandoverError(f"sampler task record is not serializable: {exc}") from exc
  if len(encoded.encode()) > _MAX_TASK_RECORD_BYTES:
    raise HandoverError(
      f"sampler task record exceeds the {_MAX_TASK_RECORD_BYTES}-byte bound"
    )
  return record


@dataclass(frozen=True, slots=True)
class ContinuationStep:
  """One decoded diffusion control step and its recorded provenance.

  ``task`` is the optional bounded task-lane record for this tick (active goal
  and keyframe identifiers, denoiser/VJP/cost counts and bounded cost/norm
  scalars).  It is ``None`` for the default unguided path.
  """

  tick: int
  action: np.ndarray
  executed_latent: np.ndarray
  denoiser_calls: int
  final_index: int
  predicted_next_hybrid_state: np.ndarray | None = None
  task: Mapping[str, Any] | None = None


class ReferenceFreeContinuation:
  """VAE-history handover plus reference-free DDIM continuation and decode.

  The lifecycle a controller must follow (PRE-ACTION pairing: each history
  row is the state ``s_t`` at execution time paired with the latent ``z_t``
  that was decoded on ``s_t`` and applied there; the caller keeps ``s_t``
  safe until recording after the execution is acknowledged):

  1. run the frozen VAE and :meth:`record_executed` for at least 8 such pairs;
  2. call :meth:`step` each tick with the measured current state and the
     measured 99-D gravity conditioning; the step plans, samples, denormalizes
     the current latent and calls the frozen decoder;
  3. after executing the returned action and measuring the next state, pair
     them with :meth:`record_executed` using ``step.executed_latent``;
  4. call :meth:`reset` on any reset, teleport, estimator discontinuity or
     policy switch; the next :meth:`step` then fails closed until history is
     rebuilt.

  ``decode`` receives ``(latent_raw_32, conditioning_99)`` float32 batch
  tensors and must return ``[1, 31]``.  Pass the frozen gravity-mode VAE's
  ``decode``; the seam never encodes and never substitutes a teacher or
  standing-policy latent.  The seam's own guarantee is limited to having no
  reference/encoder call path; the caller remains responsible for the origin,
  contiguity and authenticity of what it pushes.
  """

  def __init__(
    self,
    *,
    denoiser: Callable[[Tensor, Tensor], Tensor],
    decode: DecodeFn,
    schedule: DiffusionSchedule | None = None,
    grid: InferenceGrid | None = None,
    projection: StateLatentTransformLike,
    contract: DiffusionContract = DEFAULT_CONTRACT,
    device: str | torch.device = "cpu",
    generator: torch.Generator,
    sampler_config: SamplerConfig | None = None,
    sampler: FamilySampler | None = None,
    plan_transform: PlanTransform | None = None,
    sampler_factory: SamplerFactory | None = None,
    on_reset: ResetHook | None = None,
    measured_post: MeasuredPostHook | None = None,
  ) -> None:
    _require_no_mixture(contract)
    _require_matching_projection(projection, contract)
    try:
      self._device = torch.device(device)
      torch.empty(0, device=self._device)
    except (RuntimeError, ValueError) as exc:
      raise HandoverError(f"invalid runtime device {device!r}") from exc
    diffusion_config: SamplerConfig | None = None
    if sampler is not None:
      if not callable(sampler):
        raise HandoverError("sampler must be callable or None")
      # A task ``sampler_factory`` is allowed alongside an injected family
      # sampler: its result replaces that sampler for the tick and therefore
      # takes the family calling convention ``(conditions, generator)``.
      if schedule is not None or grid is not None or sampler_config is not None:
        raise HandoverError(
          "an injected family sampler owns its own solver and level grid; do "
          "not pass schedule, grid or sampler_config as well"
        )
    else:
      if schedule is None or grid is None:
        raise HandoverError(
          "the default diffusion sampler requires both schedule and grid; "
          "inject a family sampler to supply its own"
        )
      diffusion_config = sampler_config or SamplerConfig(grid)
    self.contract = contract
    self.projection = projection
    self.schedule = schedule
    self.history = HandoverHistory(contract=contract)
    self._denoiser = denoiser
    self._decode = decode
    self._generator = generator
    self._sampler = sampler
    self._sampler_config = diffusion_config
    if plan_transform is not None and not callable(plan_transform):
      raise HandoverError("plan_transform must be callable or None")
    if sampler_factory is not None and not callable(sampler_factory):
      raise HandoverError("sampler_factory must be callable or None")
    self._plan_transform = plan_transform
    self._sampler_factory = sampler_factory
    for name, hook in (("on_reset", on_reset), ("measured_post", measured_post)):
      if hook is not None and not callable(hook):
        raise HandoverError(f"{name} hook must be callable or None")
    self._on_reset = on_reset
    self._measured_post = measured_post
    self._last_tick: int | None = None

  def record_executed(
    self, state: WorldState, latent: Sequence[float] | np.ndarray
  ) -> None:
    """Record one state with the latent the decoder decoded on it.

    The caller must retain the state it executed against until this call;
    this method snapshots (copies) the pair it is given.
    """
    self.history.push(state, latent)

  def reset(self) -> None:
    """Drop history and tick age; a fresh VAE bootstrap is now required.

    An installed ``on_reset`` hook runs after the seam has cleared its own
    state, so a task lane drops its ledger before the next measured pair is
    recorded.  Without a hook this is exactly the D3 reset.
    """
    self.history.reset()
    self._last_tick = None
    if self._on_reset is not None:
      self._on_reset()

  def observe_measured_post(self, tick: int, state: WorldState) -> bool:
    """Report a trusted post, returning whether an observer actually ran.

    The evaluation runner calls this only for a post-step observation that is
    finite and reset/teleport/segment/tick/period/reference-safe.  It is a
    reporting-only seam: it never samples, decodes, changes history or advances
    the planning count.  Without a hook it returns False and adds no task count.
    """
    if self._measured_post is None:
      return False
    self._measured_post(int(tick), state)
    return True

  def plan(
    self,
    current_state: WorldState,
    *,
    tick: int,
    device: str | torch.device | None = None,
  ) -> ContinuationPlan:
    """Build the conditioning window without sampling (evaluation seam).

    When a ``plan_transform`` is installed it is applied to the built plan with
    the same measured current state, so a task lane can fix sparse targets in
    the window without rebuilding history or conditioning.
    """
    plan = build_reference_free_plan(
      self.history,
      current_state,
      projection=self.projection,
      tick=tick,
      contract=self.contract,
      device=device or self._device,
    )
    if self._plan_transform is None:
      return plan
    return self._plan_transform(plan, current_state)

  def step(
    self,
    current_state: WorldState,
    conditioning: Sequence[float] | np.ndarray,
    *,
    tick: int,
  ) -> ContinuationStep:
    """Plan from measured state, sample, denormalize, and decode."""
    if not self.history.ready:
      raise HandoverError(
        "diffusion handover requires "
        f"{self.contract.past_steps} measured state/executed-latent pairs, "
        f"got {len(self.history.pairs)}"
      )
    stale = self._last_tick
    if stale is not None and int(tick) <= stale:
      raise PlanAgeError(
        f"control tick {tick} is not newer than the last executed tick {stale}; "
        "call reset() after any reset or plan discontinuity"
      )
    plan = self.plan(current_state, tick=tick)
    conditions = SamplerConditions(plan.clean_tokens, plan.clean_mask)
    if self._sampler is not None:
      # The injected family sampler owns its model, solver and grid; the seam
      # supplies only the built conditions and the shared generator.  A task
      # factory result replaces it for that tick and takes the same convention.
      step_sampler = (
        self._sampler_factory(plan, current_state)
        if self._sampler_factory is not None
        else self._sampler
      )
      sample, diagnostics = step_sampler(conditions, self._generator)
    else:
      # __init__ requires both of these whenever no family sampler is injected.
      assert self.schedule is not None and self._sampler_config is not None
      if self._sampler_factory is None:
        sample, diagnostics = sample_trajectory(
          self._denoiser,
          schedule=self.schedule,
          conditions=conditions,
          config=self._sampler_config,
          generator=self._generator,
        )
      else:
        sampler = self._sampler_factory(plan, current_state)
        sample, diagnostics = sampler(
          self._denoiser,
          schedule=self.schedule,
          conditions=conditions,
          config=self._sampler_config,
          generator=self._generator,
        )
    task_record = _bounded_task_record(getattr(diagnostics, "task", None))
    latent_slice = slice(
      self.contract.projected_state_dimension, self.contract.token_dimension
    )
    normalized = (
      sample[self.contract.current_index, latent_slice]
      .detach()
      .to(device="cpu", dtype=torch.float32)
      .numpy()
      .astype(np.float64)
    )
    executed_latent = self.projection.denormalize_latent(normalized)
    action = self._decode_action(executed_latent, conditioning)
    predicted_next_hybrid_state: np.ndarray | None = None
    next_index = self.contract.current_index + 1
    if next_index < sample.shape[0]:
      predicted = self.projection.inverse_state(
        sample[next_index, : self.contract.projected_state_dimension]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .astype(np.float64)[None, :]
      )
      if (
        predicted.shape != (1, self.contract.state_dimension)
        or not np.isfinite(predicted).all()
      ):
        raise HandoverError(
          "sampler predicted next state must be finite with shape "
          f"[{self.contract.state_dimension}]"
        )
      predicted_next_hybrid_state = np.array(predicted[0], copy=True)
    self._last_tick = int(tick)
    return ContinuationStep(
      tick=int(tick),
      action=action,
      executed_latent=executed_latent,
      denoiser_calls=diagnostics.denoiser_calls,
      final_index=diagnostics.final_index,
      predicted_next_hybrid_state=predicted_next_hybrid_state,
      task=task_record,
    )

  def _decode_action(
    self, executed_latent: np.ndarray, conditioning: Sequence[float] | np.ndarray
  ) -> np.ndarray:
    values = np.asarray(conditioning, dtype=np.float64)
    if values.shape != (_CONDITIONING_DIM,) or not np.isfinite(values).all():
      raise HandoverError("decoder conditioning must be finite with shape [99]")
    latent_tensor = torch.from_numpy(
      np.ascontiguousarray(executed_latent, dtype=np.float32)
    ).to(device=self._device)[None]
    conditioning_tensor = torch.from_numpy(
      np.ascontiguousarray(values, dtype=np.float32)
    ).to(device=self._device)[None]
    with torch.no_grad():
      action = self._decode(latent_tensor, conditioning_tensor)
    if not isinstance(action, Tensor):
      raise HandoverError("decoder must return a torch tensor")
    decoded = action.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
    if (
      decoded.shape != (self.contract.action_dimension,)
      or not np.isfinite(decoded).all()
    ):
      raise HandoverError("decoder action must be finite with shape [31]")
    return decoded


__all__ = [
  "ContinuationPlan",
  "ContinuationStep",
  "DecodeFn",
  "ExecutedPair",
  "FamilySampler",
  "HandoverError",
  "HandoverHistory",
  "MeasuredPostHook",
  "PlanAgeError",
  "PlanTransform",
  "ReferenceFreeContinuation",
  "ResetHook",
  "SamplerCallable",
  "SamplerFactory",
  "build_reference_free_plan",
]
