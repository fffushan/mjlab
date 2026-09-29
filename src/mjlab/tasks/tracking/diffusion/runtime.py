"""Runtime-factory request/bundle types shared by the CLI and X2 factory.

These two types live in a library module rather than in the CLI script so that a
runtime factory and the CLI agree on one class object.  When the CLI is run as
``python -m mjlab.scripts.diffusion``, the script is imported twice (as
``__main__`` and as ``mjlab.scripts.diffusion``); defining the bundle in the
script makes ``isinstance(bundle, RuntimeBundle)`` fail for a factory that
imports the library copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CollectionRequest:
  """Inputs supplied to a concrete runtime factory."""

  contract: Any
  store: Any
  output: Path
  trials: tuple[Any, ...]
  max_trials: int
  max_control_transitions: int
  max_envs: int
  max_gpus: int
  max_wall_minutes: int
  monitor_inactivity_seconds: int
  max_output_mib: int


@dataclass(frozen=True)
class RuntimeBundle:
  """Verified concrete runtime and trial list returned by a factory."""

  collector: Any
  trials: tuple[Any, ...]
  runtime_verified: bool = False
  runtime_provenance: dict[str, str | int] | None = None


__all__ = ["CollectionRequest", "RuntimeBundle"]
