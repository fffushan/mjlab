"""Train-only normalization for the active no-mixture 135-D state schema.

``ProjectionBundle`` remains as a compatibility name for persisted callers, but
its active transform is only per-feature normalization.  The legacy D0 bundle
with random/emphasis rows is intentionally not accepted by the active contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from .contract import DEFAULT_CONTRACT, DiffusionContract


class ProjectionError(ValueError):
  """Projection/statistics input or persisted bundle is invalid."""


def _finite(
  name: str, value: np.ndarray, shape: tuple[int, ...] | None = None
) -> np.ndarray:
  value = np.asarray(value, dtype=np.float64)
  if shape is not None and value.shape != shape:
    raise ProjectionError(f"{name} must have shape {shape}, got {value.shape}")
  if not np.isfinite(value).all():
    raise ProjectionError(f"{name} contains non-finite values")
  return value


@dataclass(frozen=True, slots=True)
class FeatureStats:
  """Population moments fitted by a streaming float64 accumulator."""

  count: int
  mean: np.ndarray
  std: np.ndarray

  def __post_init__(self) -> None:
    mean = _finite("mean", self.mean)
    std = _finite("std", self.std)
    if self.count <= 0 or mean.shape != std.shape or np.any(std <= 0):
      raise ProjectionError("invalid feature statistics")
    object.__setattr__(self, "mean", mean)
    object.__setattr__(self, "std", std)

  def normalize(self, values: np.ndarray) -> np.ndarray:
    values = _finite("values", values)
    if values.shape[-1] != self.mean.size:
      raise ProjectionError("feature width does not match statistics")
    return (values - self.mean) / self.std

  def denormalize(self, values: np.ndarray) -> np.ndarray:
    values = _finite("values", values)
    if values.shape[-1] != self.mean.size:
      raise ProjectionError("feature width does not match statistics")
    return values * self.std + self.mean

  def as_dict(self) -> dict[str, object]:
    return {"count": self.count, "mean": self.mean.tolist(), "std": self.std.tolist()}


class _Moments:
  def __init__(self, width: int) -> None:
    self.count = 0
    self.mean = np.zeros(width, dtype=np.float64)
    self.m2 = np.zeros(width, dtype=np.float64)

  def update(self, values: np.ndarray) -> None:
    values = _finite("statistics values", values)
    if values.ndim != 2 or values.shape[1] != self.mean.size:
      raise ProjectionError("statistics values must be [N, width]")
    if not len(values):
      return
    count = values.shape[0]
    batch_mean = values.mean(axis=0, dtype=np.float64)
    centered = values - batch_mean
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
    if self.count == 0:
      self.count, self.mean, self.m2 = count, batch_mean, batch_m2
      return
    total = self.count + count
    delta = batch_mean - self.mean
    self.m2 += batch_m2 + delta * delta * (self.count * count / total)
    self.mean += delta * (count / total)
    self.count = total

  def finish(self, floor: float) -> FeatureStats:
    if self.count == 0:
      raise ProjectionError("cannot fit statistics from empty training windows")
    std = np.sqrt(self.m2 / self.count)
    std = np.maximum(std, floor)
    return FeatureStats(self.count, self.mean.copy(), std)


def fit_feature_stats(
  chunks: Iterable[np.ndarray], *, std_floor: float = 1e-6
) -> FeatureStats:
  """Fit population moments without concatenating chunks in RAM."""
  chunks = iter(chunks)
  try:
    first = _finite("statistics chunk", next(chunks))
  except StopIteration as exc:
    raise ProjectionError("cannot fit statistics from empty chunks") from exc

  if first.ndim != 2:
    raise ProjectionError("statistics chunks must be two-dimensional")
  moments = _Moments(first.shape[1])
  moments.update(first)
  for chunk in chunks:
    moments.update(chunk)
  return moments.finish(std_floor)


def _statistics_hash(state_stats: FeatureStats, latent_stats: FeatureStats) -> str:
  payload = json.dumps(
    {"state": state_stats.as_dict(), "latent": latent_stats.as_dict()},
    sort_keys=True,
    separators=(",", ":"),
  ).encode()
  return hashlib.sha256(payload).hexdigest()


@runtime_checkable
class StateLatentTransform(Protocol):
  """Normalization operations shared by active and legacy bundles."""

  @property
  def contract(self) -> DiffusionContract: ...

  @property
  def statistics_sha256(self) -> str: ...

  def project_state(self, state: np.ndarray) -> np.ndarray: ...

  def inverse_state(self, state: np.ndarray, /) -> np.ndarray: ...

  def normalize_latent(self, latent: np.ndarray) -> np.ndarray: ...

  def denormalize_latent(self, latent: np.ndarray) -> np.ndarray: ...

  def make_tokens(self, state: np.ndarray, latent: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class StateLatentNormalizer:
  """Statistics-only normalization for the active 135+32 representation.

  The normalizer does not store or apply a state projection.  ``project_state``
  and ``inverse_state`` are compatibility names for callers that still use the
  historical ``ProjectionBundle`` interface.
  """

  contract: DiffusionContract
  state_stats: FeatureStats
  latent_stats: FeatureStats
  statistics_sha256: str

  def __post_init__(self) -> None:
    if self.contract.token_dimension != (
      self.contract.state_dimension + self.contract.latent_dimension
    ):
      raise ProjectionError("normalizer contract must have a 135+32 token layout")
    if self.state_stats.mean.size != self.contract.state_dimension:
      raise ProjectionError("normalizer state statistics width does not match contract")
    if self.latent_stats.mean.size != self.contract.latent_dimension:
      raise ProjectionError(
        "normalizer latent statistics width does not match contract"
      )
    expected = _statistics_hash(self.state_stats, self.latent_stats)
    if self.statistics_sha256 != expected:
      raise ProjectionError("normalizer statistics hash mismatch")

  @classmethod
  def create(
    cls,
    state_stats: FeatureStats,
    latent_stats: FeatureStats,
    *,
    contract: DiffusionContract = DEFAULT_CONTRACT,
  ) -> "StateLatentNormalizer":
    return cls(
      contract=contract,
      state_stats=state_stats,
      latent_stats=latent_stats,
      statistics_sha256=_statistics_hash(state_stats, latent_stats),
    )

  @classmethod
  def load(
    cls, path: str | Path, *, contract: DiffusionContract = DEFAULT_CONTRACT
  ) -> "StateLatentNormalizer":
    """Load means and standard deviations without requiring matrix fields."""
    try:
      with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        bundle_contract = metadata.get("contract", {})
        if bundle_contract.get("schema_version") != contract.schema_version:
          raise ProjectionError("normalizer contract schema mismatch")
        state_meta = metadata["state_stats"]
        latent_meta = metadata["latent_stats"]
        state = FeatureStats(
          int(state_meta["count"]), data["state_mean"], data["state_std"]
        )
        latent = FeatureStats(
          int(latent_meta["count"]), data["latent_mean"], data["latent_std"]
        )
        recorded = str(metadata.get("statistics_sha256", ""))
    except ProjectionError:
      raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ProjectionError(f"invalid statistics-only normalizer {path}") from exc
    expected = _statistics_hash(state, latent)
    if recorded and recorded != expected:
      raise ProjectionError("normalizer statistics hash mismatch")
    return cls(contract, state, latent, expected)

  def normalize_state(self, state: np.ndarray) -> np.ndarray:
    return self.state_stats.normalize(state)

  def denormalize_state(self, state: np.ndarray) -> np.ndarray:
    return self.state_stats.denormalize(state)

  def project_state(self, state: np.ndarray) -> np.ndarray:
    return self.normalize_state(state)

  def inverse_state(self, state: np.ndarray) -> np.ndarray:
    return self.denormalize_state(state)

  def normalize_latent(self, latent: np.ndarray) -> np.ndarray:
    return self.latent_stats.normalize(latent)

  def denormalize_latent(self, latent: np.ndarray) -> np.ndarray:
    return self.latent_stats.denormalize(latent)

  def make_tokens(self, state: np.ndarray, latent: np.ndarray) -> np.ndarray:
    normalized_state = self.normalize_state(state)
    normalized_latent = self.normalize_latent(latent)
    if normalized_state.shape[:-1] != normalized_latent.shape[:-1]:
      raise ProjectionError("state and latent leading dimensions differ")
    return np.concatenate((normalized_state, normalized_latent), axis=-1)

  def as_dict(self) -> dict[str, object]:
    return {
      "contract": self.contract.as_dict(),
      "state_stats": self.state_stats.as_dict(),
      "latent_stats": self.latent_stats.as_dict(),
      "statistics_sha256": self.statistics_sha256,
    }

  def save(self, path: str | Path) -> None:
    """Persist only statistics and metadata for the active representation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      path,
      state_mean=self.state_stats.mean,
      state_std=self.state_stats.std,
      latent_mean=self.latent_stats.mean,
      latent_std=self.latent_stats.std,
      metadata=np.asarray(json.dumps(self.as_dict(), sort_keys=True)),
    )


@dataclass(frozen=True, slots=True)
class ProjectionBundle:
  """Normalized state/latent statistics plus P and its Moore-Penrose inverse."""

  contract: DiffusionContract
  state_stats: FeatureStats
  latent_stats: FeatureStats
  matrix: np.ndarray
  pseudoinverse: np.ndarray
  matrix_sha256: str
  pseudoinverse_sha256: str
  statistics_sha256: str

  def __post_init__(self) -> None:
    c = self.contract
    _finite(
      "projection matrix", self.matrix, (c.projected_state_dimension, c.state_dimension)
    )
    _finite(
      "projection pseudoinverse",
      self.pseudoinverse,
      (c.state_dimension, c.projected_state_dimension),
    )
    if c.projection_rows == 0:
      identity = np.eye(c.state_dimension, dtype=np.float64)
      if not np.array_equal(self.matrix, identity) or not np.array_equal(
        self.pseudoinverse, identity
      ):
        raise ProjectionError("no-mixture bundles require identity state transforms")
    if (
      self.state_stats.mean.size != c.state_dimension
      or self.latent_stats.mean.size != c.latent_dimension
    ):
      raise ProjectionError("statistics widths do not match contract")
    if (
      self._hash(self.matrix) != self.matrix_sha256
      or self._hash(self.pseudoinverse) != self.pseudoinverse_sha256
    ):
      raise ProjectionError("projection matrix hash mismatch")
    if self.statistics_hash() != self.statistics_sha256:
      raise ProjectionError("statistics hash mismatch")

  @staticmethod
  def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(
      np.ascontiguousarray(value, dtype=np.float64).tobytes()
    ).hexdigest()

  def statistics_hash(self) -> str:
    return _statistics_hash(self.state_stats, self.latent_stats)

  @classmethod
  def create(
    cls,
    state_stats: FeatureStats,
    latent_stats: FeatureStats,
    *,
    contract: DiffusionContract = DEFAULT_CONTRACT,
  ) -> "ProjectionBundle":
    if (
      state_stats.mean.size != contract.state_dimension
      or latent_stats.mean.size != contract.latent_dimension
    ):
      raise ProjectionError("statistics dimensions do not match contract")
    if contract.schema_version == "mjlab-x2-state-latent-d0-v1":
      rng = np.random.Generator(np.random.PCG64(contract.projection_seed))
      a = rng.normal(
        0.0, 1.0, size=(contract.projection_rows, contract.state_dimension)
      ).astype(np.float64)
      emphasis = np.ones(contract.state_dimension, dtype=np.float64)
      emphasis[:15] = 6.0
      matrix = np.vstack(
        (a * emphasis[None, :], np.eye(contract.state_dimension, dtype=np.float64))
      )
      pseudoinverse = np.linalg.pinv(matrix, rcond=contract.projection_rcond)
    else:
      if contract.projection_rows != 0:
        raise ProjectionError(
          "active no-mixture bundles cannot contain added projection rows; "
          "use the explicit migration command"
        )
      identity = np.eye(contract.state_dimension, dtype=np.float64)
      matrix = identity.copy()
      pseudoinverse = identity.copy()
    temporary = object.__new__(cls)
    object.__setattr__(temporary, "contract", contract)
    object.__setattr__(temporary, "state_stats", state_stats)
    object.__setattr__(temporary, "latent_stats", latent_stats)
    object.__setattr__(temporary, "matrix", matrix)
    object.__setattr__(temporary, "pseudoinverse", pseudoinverse)
    object.__setattr__(temporary, "matrix_sha256", cls._hash(matrix))
    object.__setattr__(temporary, "pseudoinverse_sha256", cls._hash(pseudoinverse))
    object.__setattr__(temporary, "statistics_sha256", temporary.statistics_hash())
    return temporary

  def project_state(self, state: np.ndarray) -> np.ndarray:
    normalized = self.state_stats.normalize(state)
    if self.contract.projection_rows == 0:
      return normalized
    return normalized @ self.matrix.T

  def inverse_state(self, projected: np.ndarray) -> np.ndarray:
    projected = _finite("projected state", projected)
    if projected.shape[-1] != self.contract.projected_state_dimension:
      raise ProjectionError("projected state width does not match contract")
    normalized = (
      projected
      if self.contract.projection_rows == 0
      else projected @ self.pseudoinverse.T
    )
    return self.state_stats.denormalize(normalized)

  def normalize_latent(self, latent: np.ndarray) -> np.ndarray:
    return self.latent_stats.normalize(latent)

  def denormalize_latent(self, latent: np.ndarray) -> np.ndarray:
    return self.latent_stats.denormalize(latent)

  def make_tokens(self, state: np.ndarray, latent: np.ndarray) -> np.ndarray:
    projected = self.project_state(state)
    normalized_latent = self.normalize_latent(latent)
    if projected.shape[:-1] != normalized_latent.shape[:-1]:
      raise ProjectionError("state and latent leading dimensions differ")
    return np.concatenate((projected, normalized_latent), axis=-1)

  def as_dict(self) -> dict[str, object]:
    return {
      "contract": self.contract.as_dict(),
      "state_stats": self.state_stats.as_dict(),
      "latent_stats": self.latent_stats.as_dict(),
      "matrix_sha256": self.matrix_sha256,
      "pseudoinverse_sha256": self.pseudoinverse_sha256,
      "statistics_sha256": self.statistics_sha256,
    }

  def save(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      path,
      matrix=self.matrix,
      pseudoinverse=self.pseudoinverse,
      state_mean=self.state_stats.mean,
      state_std=self.state_stats.std,
      latent_mean=self.latent_stats.mean,
      latent_std=self.latent_stats.std,
      metadata=np.asarray(json.dumps(self.as_dict(), sort_keys=True)),
    )

  @classmethod
  def load(
    cls, path: str | Path, *, contract: DiffusionContract = DEFAULT_CONTRACT
  ) -> "ProjectionBundle":
    try:
      with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        bundle_schema = metadata["contract"]["schema_version"]
        if bundle_schema != contract.schema_version:
          if bundle_schema == "mjlab-x2-state-latent-d0-v1":
            raise ProjectionError(
              "legacy projected D0 bundle rejected by the active no-mixture contract; "
              "run the explicit raw-store migration"
            )
          raise ProjectionError("projection contract schema mismatch")
        state = FeatureStats(
          int(metadata["state_stats"]["count"]), data["state_mean"], data["state_std"]
        )
        latent = FeatureStats(
          int(metadata["latent_stats"]["count"]),
          data["latent_mean"],
          data["latent_std"],
        )
        matrix, pinv = data["matrix"], data["pseudoinverse"]
    except ProjectionError:
      raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ProjectionError(f"invalid projection bundle {path}") from exc
    bundle = cls(
      contract=contract,
      state_stats=state,
      latent_stats=latent,
      matrix=np.asarray(matrix, dtype=np.float64),
      pseudoinverse=np.asarray(pinv, dtype=np.float64),
      matrix_sha256=str(metadata["matrix_sha256"]),
      pseudoinverse_sha256=str(metadata["pseudoinverse_sha256"]),
      statistics_sha256=str(metadata["statistics_sha256"]),
    )
    if metadata.get("contract", {}).get("schema_version") != contract.schema_version:
      raise ProjectionError("projection contract schema mismatch")
    # The source YAML digest is optional reproduction provenance.  Projection
    # compatibility is defined by the active schema and the numerical bundle,
    # not by byte identity of the manifest file.
    return bundle


StateLatentTransformLike = ProjectionBundle | StateLatentTransform


__all__ = [
  "FeatureStats",
  "ProjectionBundle",
  "ProjectionError",
  "StateLatentNormalizer",
  "StateLatentTransform",
  "StateLatentTransformLike",
  "fit_feature_stats",
]
