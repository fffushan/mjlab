"""Policy-side action semantics for the frozen D1 collection path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .adapter import ActionContract, FrozenPpoRecoveryPolicy, FrozenVaePolicy


class PolicyError(ValueError):
  """A policy action or perturbation is invalid."""


def _action(value: Sequence[float] | np.ndarray, name: str = "action") -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (31,) or not np.isfinite(result).all():
    raise PolicyError(f"{name} must be finite with shape [31]")
  return result.copy()


@dataclass(frozen=True, slots=True)
class PolicyAction:
  """One normalized decoder action and its exact perturbation components."""

  clean: np.ndarray
  ou_noise: np.ndarray
  executed: np.ndarray

  def __post_init__(self) -> None:
    clean = _action(self.clean, "clean action")
    noise = _action(self.ou_noise, "OU noise")
    executed = _action(self.executed, "executed action")
    if not np.array_equal(clean + noise, executed):
      raise PolicyError("executed action must equal clean action plus OU noise")
    object.__setattr__(self, "clean", clean)
    object.__setattr__(self, "ou_noise", noise)
    object.__setattr__(self, "executed", executed)


class OUProcess:
  """The D0 normalized-action OU recurrence (with ``dt=1`` by contract)."""

  def __init__(
    self,
    *,
    theta: float = 0.8,
    mu: float = 0.0,
    sigma: float = 0.1,
    dt: float = 1.0,
    seed: int = 0,
  ) -> None:
    values = (theta, mu, sigma, dt)
    if (
      not all(np.isfinite(value) for value in values)
      or theta < 0
      or sigma < 0
      or dt <= 0
    ):
      raise PolicyError(
        "OU parameters must be finite, theta/sigma non-negative, dt positive"
      )
    self.theta, self.mu, self.sigma, self.dt = theta, mu, sigma, dt
    self._seed = int(seed)
    self._rng = np.random.default_rng(self._seed)
    self._state = np.full(31, mu, dtype=np.float64)

  @property
  def state(self) -> np.ndarray:
    return self._state.copy()

  def reset(self, seed: int | None = None) -> None:
    if seed is not None:
      self._seed = int(seed)
    self._rng = np.random.default_rng(self._seed)
    self._state.fill(self.mu)

  def sample(self) -> np.ndarray:
    noise = self._rng.normal(size=31)
    self._state += self.theta * (self.mu - self._state) * self.dt
    self._state += self.sigma * np.sqrt(self.dt) * noise
    return self._state.copy()

  def zero(self) -> np.ndarray:
    return np.zeros(31, dtype=np.float64)


class BatchedOUProcess:
  """Independent OU streams with vectorized state updates for trial slots."""

  def __init__(
    self,
    seeds: Sequence[int],
    *,
    theta: float = 0.8,
    mu: float = 0.0,
    sigma: float = 0.1,
    dt: float = 1.0,
  ) -> None:
    if not seeds:
      raise PolicyError("batched OU needs at least one seed")
    values = (theta, mu, sigma, dt)
    if (
      not all(np.isfinite(value) for value in values)
      or theta < 0
      or sigma < 0
      or dt <= 0
    ):
      raise PolicyError(
        "OU parameters must be finite, theta/sigma non-negative, dt positive"
      )
    self.theta, self.mu, self.sigma, self.dt = theta, mu, sigma, dt
    self._rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    self._state = np.full((len(self._rngs), 31), mu, dtype=np.float64)

  @property
  def batch_size(self) -> int:
    return self._state.shape[0]

  @property
  def state(self) -> np.ndarray:
    return self._state.copy()

  def reset(self, env_ids: Any, seeds: Sequence[int]) -> None:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    values = tuple(int(seed) for seed in seeds)
    if ids.size != len(values) or ids.size == 0:
      raise PolicyError("OU reset rows and seeds must be non-empty and aligned")
    if np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise PolicyError("OU reset row is outside the batch")
    for row, seed in zip(ids.tolist(), values, strict=True):
      self._rngs[row] = np.random.default_rng(seed)
      self._state[row].fill(self.mu)

  def sample(self, env_ids: Sequence[int]) -> np.ndarray:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
      return np.empty((0, 31), dtype=np.float64)
    if np.any(ids < 0) or np.any(ids >= self.batch_size):
      raise PolicyError("OU sample row is outside the batch")
    noise = np.stack([self._rngs[row].normal(size=31) for row in ids.tolist()])
    self._state[ids] += self.theta * (self.mu - self._state[ids]) * self.dt
    self._state[ids] += self.sigma * np.sqrt(self.dt) * noise
    return self._state[ids].copy()

  def zero(self, env_ids: Sequence[int]) -> np.ndarray:
    ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
    return np.zeros((ids.size, 31), dtype=np.float64)


def make_policy_action(
  clean: Sequence[float] | np.ndarray,
  ou_noise: Sequence[float] | np.ndarray | None = None,
) -> PolicyAction:
  """Build an action while enforcing the stored pre/post arithmetic."""
  clean_value = _action(clean, "clean action")
  noise_value = (
    np.zeros(31, dtype=np.float64)
    if ou_noise is None
    else _action(ou_noise, "OU noise")
  )
  return PolicyAction(clean_value, noise_value, clean_value + noise_value)


__all__ = [
  "ActionContract",
  "FrozenPpoRecoveryPolicy",
  "FrozenVaePolicy",
  "BatchedOUProcess",
  "OUProcess",
  "PolicyAction",
  "PolicyError",
  "make_policy_action",
]
