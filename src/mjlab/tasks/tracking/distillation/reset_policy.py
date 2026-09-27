"""Validated reset sampling for standing-start distillation collection.

This module intentionally has no simulator or command dependencies.  The
``reference`` policy is a legacy no-op: it does not consume the supplied
random generator and returns a reference mask with frame zero as a sentinel.
The caller must retain its existing reference-frame sampler in that case.

For ``standing-mixture``, sampling is performed independently for each row
that the caller has identified as undergoing a full environment reset.  A
standing row samples its frame from the clip-local early window, while a
reference row samples uniformly from its whole clip.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import torch

ResetPolicyKind = Literal["reference", "standing-mixture"]
"""Supported distillation reset-policy kinds."""

RESET_POLICY_VERSION = 1
"""Persisted schema version for :class:`ResetPolicy`."""

_RESET_POLICY_KINDS: tuple[ResetPolicyKind, ...] = ("reference", "standing-mixture")
_RESET_POLICY_FIELDS = (
  "version",
  "kind",
  "standing_start_fraction",
  "standing_start_window_frames",
  "standing_start_frame_zero_fraction",
)


class ResetPolicyError(ValueError):
  """A reset policy or reset-sampling input is invalid."""


def _validate_probability(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ResetPolicyError(
      f"{name} must be a finite probability in [0, 1], got {value!r}"
    )
  probability = float(value)
  if not math.isfinite(probability):
    raise ResetPolicyError(
      f"{name} must be finite; received {value!r}. Use a value in [0, 1]."
    )
  if not 0.0 <= probability <= 1.0:
    raise ResetPolicyError(f"{name} must be in [0, 1], got {probability!r}")
  return probability


def _validate_window(value: object) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ResetPolicyError(
      "standing_start_window_frames must be a positive integer number of frames; "
      f"got {value!r}"
    )
  if value <= 0:
    raise ResetPolicyError(
      f"standing_start_window_frames must be positive; got {value!r}"
    )
  return value


def _require_mapping(payload: object) -> Mapping[str, object]:
  if not isinstance(payload, Mapping):
    raise ResetPolicyError("reset policy serialization must be a mapping")
  if any(not isinstance(key, str) for key in payload):
    raise ResetPolicyError("reset policy serialization keys must be strings")
  missing = [key for key in _RESET_POLICY_FIELDS if key not in payload]
  unknown = [key for key in payload if key not in _RESET_POLICY_FIELDS]
  if missing or unknown:
    raise ResetPolicyError(
      "reset policy serialization fields do not match the schema: "
      f"missing={missing}, unknown={unknown}"
    )
  return cast(Mapping[str, object], payload)


@dataclass(frozen=True, slots=True)
class ResetPolicy:
  """Plain-data, opt-in reset policy for distillation collection.

  ``standing_start_fraction`` is an unconditional per-row Bernoulli
  probability over eligible full resets.  It is not PPO's conditional
  standing-start probability.  ``standing_start_window_frames`` is clipped
  independently for every assigned clip.  Within a standing reset, frame zero
  is selected with ``standing_start_frame_zero_fraction``; otherwise the
  uniform early-window sampler also includes frame zero.  Thus the realized
  frame-zero probability conditional on standing is
  ``a + (1 - a) / W_i``.

  The default ``reference`` policy preserves the legacy reset path.  Its other
  fields are still validated and persisted so malformed newly supplied options
  fail before artifacts are resolved.  The three numeric values are proposed
  starting hyperparameters, not measured optima.
  """

  kind: ResetPolicyKind = "reference"
  standing_start_fraction: float = 0.25
  standing_start_window_frames: int = 25
  standing_start_frame_zero_fraction: float = 0.5

  def __post_init__(self) -> None:
    if self.kind not in _RESET_POLICY_KINDS:
      raise ResetPolicyError(
        "reset_policy kind must be 'reference' or 'standing-mixture', "
        f"got {self.kind!r}"
      )
    fraction = _validate_probability(
      self.standing_start_fraction, "standing_start_fraction"
    )
    window = _validate_window(self.standing_start_window_frames)
    frame_zero_fraction = _validate_probability(
      self.standing_start_frame_zero_fraction,
      "standing_start_frame_zero_fraction",
    )
    object.__setattr__(self, "standing_start_fraction", fraction)
    object.__setattr__(self, "standing_start_window_frames", window)
    object.__setattr__(self, "standing_start_frame_zero_fraction", frame_zero_fraction)

  @property
  def enabled(self) -> bool:
    """Whether this policy owns a standing-mixture sampling decision."""
    return self.kind == "standing-mixture"

  def as_dict(self) -> dict[str, Any]:
    """Return the canonical plain-data representation for persistence."""
    return {
      "version": RESET_POLICY_VERSION,
      "kind": self.kind,
      "standing_start_fraction": self.standing_start_fraction,
      "standing_start_window_frames": self.standing_start_window_frames,
      "standing_start_frame_zero_fraction": self.standing_start_frame_zero_fraction,
    }

  def canonical_json(self) -> str:
    """Return hash-stable JSON with no non-finite values."""
    return canonical_json(self)

  def digest(self) -> str:
    """Return the SHA-256 digest of the canonical policy representation."""
    return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

  @classmethod
  def from_dict(cls, payload: object) -> ResetPolicy:
    """Rebuild and validate a policy from its persisted plain-data form."""
    values = _require_mapping(payload)
    version = values["version"]
    if isinstance(version, bool) or not isinstance(version, int):
      raise ResetPolicyError("reset policy version must be an integer")
    if version != RESET_POLICY_VERSION:
      raise ResetPolicyError(
        f"unsupported reset policy version {version}; expected {RESET_POLICY_VERSION}"
      )
    kind = values["kind"]
    if not isinstance(kind, str):
      raise ResetPolicyError("reset policy kind must be a string")
    return cls(
      kind=cast(ResetPolicyKind, kind),
      standing_start_fraction=cast(float, values["standing_start_fraction"]),
      standing_start_window_frames=cast(int, values["standing_start_window_frames"]),
      standing_start_frame_zero_fraction=cast(
        float, values["standing_start_frame_zero_fraction"]
      ),
    )

  @classmethod
  def from_json(cls, payload: str) -> ResetPolicy:
    """Rebuild and validate a policy from canonical JSON."""
    try:
      decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
      raise ResetPolicyError("reset policy JSON is invalid") from exc
    return cls.from_dict(decoded)


def make_reset_policy(
  *,
  kind: ResetPolicyKind = "reference",
  standing_start_fraction: float = 0.25,
  standing_start_window_frames: int = 25,
  standing_start_frame_zero_fraction: float = 0.5,
) -> ResetPolicy:
  """Validate the reset options before any artifact or simulator resolution."""
  return ResetPolicy(
    kind=kind,
    standing_start_fraction=standing_start_fraction,
    standing_start_window_frames=standing_start_window_frames,
    standing_start_frame_zero_fraction=standing_start_frame_zero_fraction,
  )


def _clip_frames_tensor(clip_frames: Sequence[int] | torch.Tensor) -> torch.Tensor:
  if isinstance(clip_frames, torch.Tensor):
    if clip_frames.ndim != 1:
      raise ResetPolicyError(
        f"clip_frames must be one-dimensional, got shape {tuple(clip_frames.shape)}"
      )
    if clip_frames.dtype not in (
      torch.int8,
      torch.int16,
      torch.int32,
      torch.int64,
      torch.uint8,
    ):
      raise ResetPolicyError("clip_frames must contain integer frame counts")
    values = clip_frames.detach().to(device="cpu", dtype=torch.int64)
  else:
    try:
      values_tuple = tuple(clip_frames)
    except TypeError as exc:
      raise ResetPolicyError(
        "clip_frames must be a one-dimensional integer sequence"
      ) from exc
    if any(
      isinstance(value, bool) or not isinstance(value, int) for value in values_tuple
    ):
      raise ResetPolicyError("clip_frames must contain integer frame counts")
    values = torch.tensor(values_tuple, dtype=torch.int64)
  if bool((values <= 0).any()):
    raise ResetPolicyError(
      f"clip_frames must be positive for every clip, got {values.tolist()}"
    )
  return values


def effective_window_frames(
  clip_frames: Sequence[int] | torch.Tensor,
  configured_window_frames: int,
) -> torch.Tensor:
  """Return each clip's ``min(configured_window_frames, F_i)`` window."""
  window = _validate_window(configured_window_frames)
  values = _clip_frames_tensor(clip_frames)
  return torch.minimum(values, torch.full_like(values, window))


def effective_windows(
  policy: ResetPolicy, clip_frames: Sequence[int] | torch.Tensor
) -> torch.Tensor:
  """Return effective clip-local windows for a validated policy."""
  if not isinstance(policy, ResetPolicy):
    raise ResetPolicyError(f"policy must be a ResetPolicy, got {type(policy).__name__}")
  return effective_window_frames(clip_frames, policy.standing_start_window_frames)


class ResetSample(NamedTuple):
  """Per-row reset decision returned by :func:`sample_reset_policy`.

  ``kind`` is a boolean tensor: ``False`` means reference initialization and
  ``True`` means standing initialization.  ``frame`` is a CPU int64 tensor.
  For the disabled reference policy, frame zero is a sentinel and the caller
  retains the legacy reference-frame decision instead.
  """

  kind: torch.Tensor
  frame: torch.Tensor


def _uniform_frames(
  upper_bounds: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
  """Sample exact clip-local integers with one shared rejection loop."""
  if upper_bounds.numel() == 0:
    return torch.empty(0, dtype=torch.int64)
  bounds = upper_bounds.to(dtype=torch.int64)
  result = torch.empty(bounds.numel(), dtype=torch.int64)
  for upper in torch.unique(bounds).tolist():
    rows = torch.nonzero(bounds == upper, as_tuple=False).flatten()
    result[rows] = torch.randint(
      int(upper), (rows.numel(),), generator=generator, device="cpu"
    )
  return result


def sample_reset_policy(
  policy: ResetPolicy,
  clip_frames: Sequence[int] | torch.Tensor,
  generator: torch.Generator,
) -> ResetSample:
  """Sample per-row initialization kind and clip-local frame.

  The helper is simulator-independent and samples on CPU.  For an enabled
  policy, standing selection is an unconditional Bernoulli per row, followed
  by the specified whole-clip or clip-local-window frame sampler.  The
  disabled reference policy performs no random draw at all, including no draw
  from a global RNG; its frame output is a zero sentinel for the caller's
  existing reference sampler.
  """
  if not isinstance(policy, ResetPolicy):
    raise ResetPolicyError(f"policy must be a ResetPolicy, got {type(policy).__name__}")
  if not isinstance(generator, torch.Generator):
    raise ResetPolicyError("generator must be a torch.Generator")
  if generator.device.type != "cpu":
    raise ResetPolicyError(
      "reset-policy sampling requires a CPU torch.Generator; sample on CPU "
      "and move the returned decisions to the target device"
    )
  values = _clip_frames_tensor(clip_frames)
  size = values.numel()
  kind = torch.zeros(size, dtype=torch.bool)
  frames = torch.zeros(size, dtype=torch.int64)
  if not policy.enabled or size == 0:
    return ResetSample(kind, frames)

  if policy.standing_start_fraction == 0.0:
    standing = kind
  elif policy.standing_start_fraction == 1.0:
    standing = torch.ones(size, dtype=torch.bool)
  else:
    standing = (
      torch.rand(size, generator=generator, device="cpu")
      < policy.standing_start_fraction
    )
  reference = ~standing
  if bool(reference.any()):
    frames[reference] = _uniform_frames(values[reference], generator)

  standing_indices = torch.nonzero(standing, as_tuple=False).flatten()
  if standing_indices.numel() == 0:
    return ResetSample(kind, frames)
  windows = effective_windows(policy, values)[standing_indices]
  if policy.standing_start_frame_zero_fraction == 1.0:
    frame_zero = torch.ones(standing_indices.numel(), dtype=torch.bool)
  elif policy.standing_start_frame_zero_fraction == 0.0:
    frame_zero = torch.zeros(standing_indices.numel(), dtype=torch.bool)
  else:
    frame_zero = (
      torch.rand(standing_indices.numel(), generator=generator, device="cpu")
      < policy.standing_start_frame_zero_fraction
    )
  nonzero_branch = ~frame_zero
  sampled = torch.zeros(standing_indices.numel(), dtype=torch.int64)
  if bool(nonzero_branch.any()):
    sampled[nonzero_branch] = _uniform_frames(windows[nonzero_branch], generator)
  frames[standing_indices] = sampled
  return ResetSample(standing, frames)


def canonical_json(policy: ResetPolicy | Mapping[str, object]) -> str:
  """Serialize a policy using sorted, compact, NaN-free JSON."""
  payload = policy.as_dict() if isinstance(policy, ResetPolicy) else dict(policy)
  try:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
  except (TypeError, ValueError) as exc:
    raise ResetPolicyError("reset policy cannot be canonically serialized") from exc


def reset_policy_digest(policy: ResetPolicy) -> str:
  """Return the canonical SHA-256 digest used by lifecycle persistence."""
  if not isinstance(policy, ResetPolicy):
    raise ResetPolicyError(f"policy must be a ResetPolicy, got {type(policy).__name__}")
  return hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()


__all__ = [
  "RESET_POLICY_VERSION",
  "ResetPolicy",
  "ResetPolicyError",
  "ResetPolicyKind",
  "ResetSample",
  "canonical_json",
  "effective_window_frames",
  "effective_windows",
  "make_reset_policy",
  "reset_policy_digest",
  "sample_reset_policy",
]
