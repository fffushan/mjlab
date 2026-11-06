"""Compare a runtime environment config against the saved config of a training run.

The six E0 policies were trained by configs that the pilot wrote to
``params/env.yaml``. An evaluation that quietly resolves a different config would
report on a task nobody trained, so the evaluator compares its own resolved config
against the arm's saved file and refuses to run on an unexpected difference.

Only these may differ, and each has a stated reason:

* the number of environments (32 scored episodes, not 4096 training envs),
* the environment seed (one shared episode seed, not the training seed),
* the capture additions this evaluator installs: one read-only contact sensor and
  one recorder term, when the comparison is made after they are installed,
* two lambda qualnames that ``yaml`` truncates when it writes a saved config.

This module is pure Python: it takes two plain dicts so the comparison can be
tested, and so the caller decides where the saved dict comes from.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from typing import Any

RECORDER_TERM_NAME = "nominal_trace"
"""Name of the recorder term the evaluator adds."""

ANY_CONTACT_SENSOR_NAME = "ball_any_contact"
"""Name of the read-only contact sensor the evaluator appends."""

ALLOWED_RUNTIME_DIFFERENCES: dict[str, str] = {
  "scene.num_envs": "the evaluation runs 32 environments, not the training 4096",
  "scene.terrain.num_envs": "mirrors scene.num_envs",
  "seed": "one fixed episode seed is shared by all eight arms",
  "scene.entities.robot.spec_fn": "yaml truncates a lambda qualname; same function",
  "scene.terrain.spec_fn": "yaml truncates a lambda qualname; same function",
}
"""Config paths that may differ from the saved training config, with the reason."""


def normalize(value: Any) -> Any:
  """Convert a config value into something that compares by text.

  ``asdict`` leaves callables, enums and slices in place, while a saved
  ``env.yaml`` reloaded by :func:`mjlab.utils.os.load_saved_yaml` holds strings,
  one-element lists and lists of slice fields. The two encodings are reconciled
  here so a real difference is not lost in representation noise.

  Args:
    value: Any value from a configuration dump.

  Returns:
    The same structure with callables as ``module.qualname`` strings, enums as a
    one-element list of their value, and slices as ``[start, stop, step]``.
  """
  if isinstance(value, enum.Enum):
    return [value.value]
  if callable(value):
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", repr(value))
    return f"{module}.{qualname}"
  if isinstance(value, slice):
    return [value.start, value.stop, value.step]
  if isinstance(value, Mapping):
    return {str(key): normalize(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [normalize(item) for item in value]
  return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
  """Flatten a nested config into ``dotted.path[index]`` leaves.

  Args:
    value: A normalized config value.
    prefix: Path prefix used by the recursion.

  Returns:
    Path to leaf value. Empty containers contribute no path.
  """
  flat: dict[str, Any] = {}
  if isinstance(value, Mapping):
    for key, item in value.items():
      child = f"{prefix}.{key}" if prefix else str(key)
      flat.update(flatten(item, child))
  elif isinstance(value, list):
    for index, item in enumerate(value):
      flat.update(flatten(item, f"{prefix}[{index}]"))
  else:
    flat[prefix] = value
  return flat


def strip_capture_additions(
  runtime: dict[str, Any], saved: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Remove this evaluator's capture additions from a runtime config dump.

  The additions are one recorder term and one appended scene sensor. Both are
  removed from the runtime copy after checking they are exactly what this module
  declares, and only when the saved config does not already contain them.

  Args:
    runtime: Runtime ``asdict(env_cfg)`` dump. Not modified.
    saved: Saved training config dump.

  Returns:
    ``(runtime_without_additions, report)``. The report lists what was removed.

  Raises:
    ValueError: If the saved config already carries the capture additions, or the
      runtime additions are not the declared ones.
  """
  stripped = {key: value for key, value in runtime.items()}
  scene = dict(stripped.get("scene") or {})
  report: dict[str, Any] = {}

  saved_sensors = list((saved.get("scene") or {}).get("sensors") or [])
  runtime_sensors = list(scene.get("sensors") or [])
  if len(runtime_sensors) == len(saved_sensors):
    # Compared before the capture additions were installed. Nothing to strip.
    pass
  elif len(runtime_sensors) == len(saved_sensors) + 1:
    extra = runtime_sensors[-1]
    if extra.get("name") != ANY_CONTACT_SENSOR_NAME:
      raise ValueError(
        f"the appended sensor is {extra.get('name')!r}, not {ANY_CONTACT_SENSOR_NAME!r}"
      )
    report["appended_sensor"] = extra.get("name")
    scene["sensors"] = runtime_sensors[:-1]
  else:
    raise ValueError(
      "expected the saved sensors plus at most one appended capture sensor, "
      f"runtime has {len(runtime_sensors)} and the saved config has "
      f"{len(saved_sensors)}"
    )
  stripped["scene"] = scene

  saved_recorders = saved.get("recorders") or {}
  runtime_recorders = stripped.get("recorders") or {}
  if saved_recorders:
    raise ValueError(
      "the saved config already has recorder terms: "
      f"{sorted(saved_recorders)}; the capture addition cannot be separated"
    )
  if sorted(runtime_recorders) not in ([], [RECORDER_TERM_NAME]):
    raise ValueError(
      f"unexpected recorder terms: {sorted(runtime_recorders)}; expected only "
      f"{RECORDER_TERM_NAME!r}"
    )
  if runtime_recorders:
    report["recorder_terms"] = sorted(runtime_recorders)
  stripped["recorders"] = {}
  return stripped, report


def compare_env_config(
  runtime: dict[str, Any],
  saved: dict[str, Any],
  *,
  allowed: Mapping[str, str] = ALLOWED_RUNTIME_DIFFERENCES,
) -> dict[str, Any]:
  """Compare a runtime config dump against a saved training config dump.

  Args:
    runtime: Runtime ``asdict(env_cfg)`` dump.
    saved: Saved ``params/env.yaml`` dump, as loaded by
      :func:`mjlab.utils.os.load_saved_yaml`.
    allowed: Path to reason for the differences that are expected.

  Returns:
    A report with the compared path count, every difference found (each marked
    allowed or not), the capture additions that were stripped, and ``ok``.
  """
  stripped, capture = strip_capture_additions(runtime, saved)
  runtime_flat = flatten(normalize(stripped))
  saved_flat = flatten(normalize(saved))

  differences = []
  for path in sorted(set(runtime_flat) | set(saved_flat)):
    in_runtime = path in runtime_flat
    in_saved = path in saved_flat
    runtime_value = runtime_flat.get(path)
    saved_value = saved_flat.get(path)
    equal = in_runtime and in_saved and repr(runtime_value) == repr(saved_value)
    if equal:
      continue
    differences.append(
      {
        "path": path,
        "runtime": repr(runtime_value) if in_runtime else None,
        "saved": repr(saved_value) if in_saved else None,
        "allowed_reason": allowed.get(path),
      }
    )

  unexpected = [row for row in differences if row["allowed_reason"] is None]
  return {
    "compared_paths": len(set(runtime_flat) & set(saved_flat)),
    "capture_additions": capture,
    "differences": differences,
    "unexpected_differences": unexpected,
    "ok": not unexpected,
  }


__all__ = [
  "ALLOWED_RUNTIME_DIFFERENCES",
  "ANY_CONTACT_SENSOR_NAME",
  "RECORDER_TERM_NAME",
  "compare_env_config",
  "flatten",
  "normalize",
  "strip_capture_additions",
]
