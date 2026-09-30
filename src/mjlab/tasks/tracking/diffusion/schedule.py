"""Frozen cosine diffusion schedule and the 20-step inference grid.

The schedule is deliberately implemented with NumPy ``float64`` arithmetic.  In
particular, index zero is an exact clean level rather than a near-one value
produced by a beta clamp.  The sampler converts this array to a torch tensor at
its execution dtype only after the schedule has been constructed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FROZEN_TRAINING_K = 1000
_FROZEN_UPDATES = 20
_FROZEN_COSINE_OFFSET = 0.008
_FROZEN_BETA_MIN = 1.0e-5
_FROZEN_BETA_MAX = 0.999
_FROZEN_SOURCE_IDS = tuple(range(1000, 0, -50))
_FROZEN_DESTINATION_IDS = tuple(range(950, -1, -50))


class ScheduleError(ValueError):
  """A diffusion schedule or inference grid is malformed."""


def _float64_hash(value: np.ndarray) -> str:
  """Hash the contiguous little-endian representation used by NumPy here."""
  return hashlib.sha256(
    np.ascontiguousarray(value, dtype=np.float64).tobytes()
  ).hexdigest()


@dataclass(frozen=True, slots=True)
class DiffusionSchedule:
  """The bounded-cosine training schedule, including the clean level."""

  training_k: int
  cosine_offset: float
  beta_min: float
  beta_max: float
  betas: np.ndarray
  alpha_bars: np.ndarray

  def __post_init__(self) -> None:
    if self.training_k <= 0:
      raise ScheduleError("training_k must be positive")
    if not 0.0 <= self.beta_min <= self.beta_max < 1.0:
      raise ScheduleError("beta bounds must satisfy 0 <= min <= max < 1")
    if self.cosine_offset < 0.0:
      raise ScheduleError("cosine_offset must be non-negative")

    betas = np.asarray(self.betas, dtype=np.float64)
    alpha_bars = np.asarray(self.alpha_bars, dtype=np.float64)
    object.__setattr__(self, "betas", betas)
    object.__setattr__(self, "alpha_bars", alpha_bars)

    if betas.shape != (self.training_k,):
      raise ScheduleError(
        f"betas must have shape {(self.training_k,)}, got {betas.shape}"
      )
    if alpha_bars.shape != (self.training_k + 1,):
      raise ScheduleError(
        f"alpha_bars must have shape {(self.training_k + 1,)}, got {alpha_bars.shape}"
      )
    if not np.isfinite(betas).all() or not np.isfinite(alpha_bars).all():
      raise ScheduleError("schedule contains non-finite values")
    if np.any(betas < self.beta_min) or np.any(betas > self.beta_max):
      raise ScheduleError("betas are outside the configured bounds")
    if alpha_bars[0] != 1.0:
      raise ScheduleError("alpha_bars[0] must be exactly 1.0")
    if np.any(np.diff(alpha_bars) >= 0.0):
      raise ScheduleError("alpha_bars must be strictly decreasing")

    expected = np.concatenate(
      (
        np.asarray([1.0], dtype=np.float64),
        np.cumprod(1.0 - betas, dtype=np.float64),
      )
    )
    if not np.array_equal(alpha_bars, expected):
      raise ScheduleError("alpha_bars are not the cumulative product of betas")

  @classmethod
  def build(
    cls,
    *,
    training_k: int = _FROZEN_TRAINING_K,
    cosine_offset: float = _FROZEN_COSINE_OFFSET,
    beta_min: float = _FROZEN_BETA_MIN,
    beta_max: float = _FROZEN_BETA_MAX,
  ) -> "DiffusionSchedule":
    """Build the frozen bounded-cosine schedule in NumPy ``float64``."""
    if training_k <= 0:
      raise ScheduleError("training_k must be positive")
    x = np.linspace(0.0, float(training_k), training_k + 1, dtype=np.float64)
    raw = (
      np.cos(
        ((x / float(training_k)) + cosine_offset)
        / (1.0 + cosine_offset)
        * math.pi
        / 2.0
      )
      ** 2
    )
    raw = raw / raw[0]
    betas = np.clip(1.0 - raw[1:] / raw[:-1], beta_min, beta_max).astype(
      np.float64, copy=False
    )
    alpha_bars = np.concatenate(
      (
        np.asarray([1.0], dtype=np.float64),
        np.cumprod(1.0 - betas, dtype=np.float64),
      )
    )
    return cls(
      int(training_k),
      float(cosine_offset),
      float(beta_min),
      float(beta_max),
      betas,
      alpha_bars,
    )

  @classmethod
  def from_contract(cls, contract: Any) -> "DiffusionSchedule":
    """Build the schedule using the D0 values represented by ``contract``.

    The current D1 contract stores the model/noise dimensions as part of the
    frozen YAML rather than as fields on :class:`DiffusionContract`, so the
    defaults below are intentional.  If a richer contract exposes these fields,
    a disagreement is rejected instead of silently changing the schedule.
    """
    training_k = _FROZEN_TRAINING_K
    cosine_offset = _FROZEN_COSINE_OFFSET
    beta_min = _FROZEN_BETA_MIN
    beta_max = _FROZEN_BETA_MAX
    values: tuple[tuple[str, int | float, int | float], ...] = (
      ("training_k", training_k, training_k),
      ("cosine_offset", cosine_offset, cosine_offset),
      ("beta_min", beta_min, beta_min),
      ("beta_max", beta_max, beta_max),
    )
    for name, expected, comparison in values:
      if hasattr(contract, name):
        value = getattr(contract, name)
        if isinstance(expected, int):
          if int(value) != int(comparison):
            raise ScheduleError(f"contract {name}={value!r} disagrees with D0")
        elif float(value) != float(comparison):
          raise ScheduleError(f"contract {name}={value!r} disagrees with D0")
    return cls.build(
      training_k=training_k,
      cosine_offset=cosine_offset,
      beta_min=beta_min,
      beta_max=beta_max,
    )

  @property
  def embedding_count(self) -> int:
    """Number of learned step embeddings, including the clean level."""
    return self.training_k + 1

  @property
  def clean_index(self) -> int:
    """The exact clean level index."""
    return 0

  @property
  def terminal_snr(self) -> float:
    """Signal-to-noise ratio at the terminal training level."""
    terminal = float(self.alpha_bars[-1])
    return terminal / (1.0 - terminal)

  def alpha_bar_at(self, k: int) -> float:
    """Return the alpha-bar at integer training level ``k``."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
      raise ScheduleError("diffusion level must be an integer")
    if k < 0 or k > self.training_k:
      raise ScheduleError(f"diffusion level {k} is outside [0, {self.training_k}]")
    return float(self.alpha_bars[int(k)])

  def alpha_bars_torch(self, *, device: Any = None, dtype: Any = None) -> Any:
    """Convert the schedule to a torch tensor without rebuilding it."""
    # Keep torch out of module import and schedule construction.  The local
    # conversion is the only bridge needed by the model and sampler.
    import torch

    kwargs: dict[str, Any] = {"device": device}
    if dtype is not None:
      kwargs["dtype"] = dtype
    return torch.as_tensor(self.alpha_bars, **kwargs)

  def sha256(self) -> str:
    """Return the self-consistency hash of the float64 alpha-bar bytes."""
    return _float64_hash(self.alpha_bars)

  def identity_hash(self) -> str:
    """Alias used by checkpoint identities."""
    return self.sha256()

  def as_dict(self) -> dict[str, object]:
    """Return serialisable construction parameters and integrity metadata."""
    digest = self.sha256()
    return {
      "training_k": self.training_k,
      "cosine_offset": self.cosine_offset,
      "beta_min": self.beta_min,
      "beta_max": self.beta_max,
      "embedding_count": self.embedding_count,
      "clean_index": self.clean_index,
      "sha256": digest,
      "alpha_bars_sha256": digest,
      "terminal_snr": self.terminal_snr,
    }

  def save(self, path: str | Path) -> None:
    """Save schedule arrays and metadata to an exact path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(self.as_dict(), sort_keys=True)
    with destination.open("wb") as handle:
      np.savez_compressed(
        handle,
        betas=self.betas,
        alpha_bars=self.alpha_bars,
        metadata=np.asarray(metadata),
      )

  @classmethod
  def load(cls, path: str | Path) -> "DiffusionSchedule":
    """Load a schedule and reject altered arrays or construction metadata."""
    source = Path(path)
    try:
      with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        betas = np.array(data["betas"], dtype=np.float64, copy=True)
        alpha_bars = np.array(data["alpha_bars"], dtype=np.float64, copy=True)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ScheduleError(f"invalid schedule file {source}") from exc

    required = {
      "training_k",
      "cosine_offset",
      "beta_min",
      "beta_max",
      "sha256",
      "terminal_snr",
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
      raise ScheduleError("schedule metadata is incomplete")
    try:
      schedule = cls(
        int(metadata["training_k"]),
        float(metadata["cosine_offset"]),
        float(metadata["beta_min"]),
        float(metadata["beta_max"]),
        betas,
        alpha_bars,
      )
      canonical = cls.build(
        training_k=schedule.training_k,
        cosine_offset=schedule.cosine_offset,
        beta_min=schedule.beta_min,
        beta_max=schedule.beta_max,
      )
    except (KeyError, TypeError, ValueError, ScheduleError) as exc:
      raise ScheduleError(f"invalid schedule construction in {source}") from exc
    if not np.array_equal(schedule.betas, canonical.betas) or not np.array_equal(
      schedule.alpha_bars, canonical.alpha_bars
    ):
      raise ScheduleError("schedule arrays differ from their canonical construction")
    if metadata["sha256"] != schedule.sha256():
      raise ScheduleError("schedule alpha-bar hash mismatch")
    if metadata.get("alpha_bars_sha256", metadata["sha256"]) != schedule.sha256():
      raise ScheduleError("schedule alpha-bar hash mismatch")
    if not math.isclose(
      float(metadata["terminal_snr"]), schedule.terminal_snr, rel_tol=0.0, abs_tol=0.0
    ):
      raise ScheduleError("schedule terminal SNR metadata mismatch")
    return schedule


@dataclass(frozen=True, slots=True)
class InferenceGrid:
  """The frozen 20-jump DDIM source and destination indices."""

  source_ids: tuple[int, ...]
  destination_ids: tuple[int, ...]

  def __post_init__(self) -> None:
    sources = tuple(int(value) for value in self.source_ids)
    destinations = tuple(int(value) for value in self.destination_ids)
    object.__setattr__(self, "source_ids", sources)
    object.__setattr__(self, "destination_ids", destinations)
    if len(sources) != _FROZEN_UPDATES or len(destinations) != _FROZEN_UPDATES:
      raise ScheduleError("the frozen inference grid must contain 20 jumps")
    if sources[0] != _FROZEN_TRAINING_K:
      raise ScheduleError("the inference grid must start at training level 1000")
    if destinations[-1] != 0:
      raise ScheduleError("the inference grid must end at clean level 0")
    if any(value < 0 or value > _FROZEN_TRAINING_K for value in sources + destinations):
      raise ScheduleError("inference grid ids are outside the training schedule")
    if len(set(sources)) != len(sources) or len(set(destinations)) != len(destinations):
      raise ScheduleError("inference grid ids must not repeat within a direction")
    if any(sources[index] <= sources[index + 1] for index in range(len(sources) - 1)):
      raise ScheduleError("source ids must be strictly decreasing")
    if any(
      destinations[index] <= destinations[index + 1]
      for index in range(len(destinations) - 1)
    ):
      raise ScheduleError("destination ids must be strictly decreasing")
    if any(
      source <= destination
      for source, destination in zip(sources, destinations, strict=True)
    ):
      raise ScheduleError("each source id must be greater than its destination id")

  @property
  def updates(self) -> int:
    """Number of reverse jumps."""
    return len(self.source_ids)

  def as_dict(self) -> dict[str, object]:
    """Return the grid in a JSON-compatible representation."""
    return {
      "source_ids": list(self.source_ids),
      "destination_ids": list(self.destination_ids),
      "updates": self.updates,
    }

  @classmethod
  def uniform(
    cls, schedule: DiffusionSchedule, *, updates: int = _FROZEN_UPDATES
  ) -> "InferenceGrid":
    """Construct evenly spaced rounded endpoints from a schedule."""
    if updates != _FROZEN_UPDATES:
      raise ScheduleError("D0 freezes the inference grid at 20 updates")
    grid = (
      np.linspace(schedule.training_k, 0, updates + 1, dtype=np.float64)
      .round()
      .astype(np.int64)
    )
    return cls(
      tuple(int(value) for value in grid[:-1]), tuple(int(value) for value in grid[1:])
    )


def build_schedule(**kwargs: Any) -> DiffusionSchedule:
  """Build a :class:`DiffusionSchedule` using its frozen defaults."""
  return DiffusionSchedule.build(**kwargs)


def build_inference_grid(
  schedule: DiffusionSchedule, *, updates: int = _FROZEN_UPDATES, contract: Any = None
) -> InferenceGrid:
  """Build the frozen grid and optionally verify its D0 contract endpoints."""
  grid = InferenceGrid.uniform(schedule, updates=updates)
  if contract is not None:
    if (
      grid.source_ids != _FROZEN_SOURCE_IDS
      or grid.destination_ids != _FROZEN_DESTINATION_IDS
    ):
      raise ScheduleError("inference grid differs from the frozen D0 endpoints")
  return grid


__all__ = [
  "DiffusionSchedule",
  "InferenceGrid",
  "ScheduleError",
  "build_inference_grid",
  "build_schedule",
]
