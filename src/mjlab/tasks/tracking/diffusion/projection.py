"""Train-only normalization and the persisted 199-D state projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    payload = json.dumps(
      {"state": self.state_stats.as_dict(), "latent": self.latent_stats.as_dict()},
      sort_keys=True,
      separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()

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
    return normalized @ self.matrix.T

  def inverse_state(self, projected: np.ndarray) -> np.ndarray:
    projected = _finite("projected state", projected)
    if projected.shape[-1] != self.contract.projected_state_dimension:
      raise ProjectionError("projected state width does not match contract")
    normalized = projected @ self.pseudoinverse.T
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
        if metadata["contract"]["schema_version"] != contract.schema_version:
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
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise ProjectionError(f"invalid projection bundle {path}") from exc
    bundle = cls.create(state, latent, contract=contract)
    if metadata.get("contract", {}).get("schema_version") != contract.schema_version:
      raise ProjectionError("projection contract schema mismatch")
    if metadata.get("contract", {}).get("contract_sha256") != contract.contract_sha256:
      raise ProjectionError("projection contract identity mismatch")
    if (
      metadata.get("matrix_sha256") != bundle.matrix_sha256
      or metadata.get("pseudoinverse_sha256") != bundle.pseudoinverse_sha256
      or metadata.get("statistics_sha256") != bundle.statistics_sha256
    ):
      raise ProjectionError("projection metadata hash mismatch")
    if not np.array_equal(bundle.matrix, matrix) or not np.array_equal(
      bundle.pseudoinverse, pinv
    ):
      raise ProjectionError(
        "projection matrices differ from deterministic contract matrices"
      )
    return bundle


__all__ = ["FeatureStats", "ProjectionBundle", "ProjectionError", "fit_feature_stats"]
