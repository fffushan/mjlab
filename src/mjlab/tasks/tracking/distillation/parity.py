"""CPU validation of frozen teachers against their original ONNX exports.

This module answers one question per teacher: does the checkpoint the manifest
selected, through the resolved observation contract, reproduce the actions of
the original ONNX export? It also ties the export back to the artifacts by
comparing exported parameters with the checkpoint and the embedded reference
arrays with the selected NPZ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  DistillationError,
  ResolvedTeacher,
  find_normalizer_divisor,
  require_onnxruntime,
)
from mjlab.tasks.tracking.distillation.teachers import TeacherBank, build_frozen_teacher

DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-5
DEFAULT_SAMPLES = 64
DEFAULT_SEED = 0

_NORMALIZER_EPS_TOLERANCE = 1e-5
"""The exported divisor is folded in float32, so recovering ``eps`` from
``divisor - std`` leaves rounding residue of order float32 epsilon."""

_ONNX_NUMPY_DTYPES = {
  "tensor(float)": np.float32,
  "tensor(double)": np.float64,
  "tensor(int64)": np.int64,
  "tensor(int32)": np.int32,
}


@dataclass(frozen=True, slots=True)
class ParityResult:
  """Numerical agreement between native teacher inference and the ONNX export."""

  samples: int
  seed: int
  atol: float
  rtol: float
  max_abs_error: float
  max_abs_error_over_rms: float
  passed: bool


@dataclass(frozen=True, slots=True)
class ReferenceResult:
  """Comparison of one embedded ONNX reference array with the selected motion."""

  array: str
  found: bool
  shape: tuple[int, ...] | None
  max_abs_diff: float | None
  exact: bool
  passed: bool


@dataclass(frozen=True, slots=True)
class AssociationResult:
  """Evidence that the ONNX export was produced from the selected checkpoint."""

  compared: tuple[str, ...]
  missing: tuple[str, ...]
  max_abs_diff: float
  normalizer_eps_spread: float | None
  passed: bool


def build_parity_inputs(
  teacher: ResolvedTeacher, samples: int, seed: int
) -> torch.Tensor:
  """Build a finite observation batch near this teacher's normalization statistics.

  The batch starts with deterministic probes (zeros, the learned mean, and
  mean +/- 1 and 2 standard deviations) so the gate also covers the region the
  teacher actually operates in, then continues with seeded Gaussian samples
  around the same statistics.
  """
  if samples < 1:
    raise ValueError(f"samples must be positive, got {samples}")
  state = teacher.actor_state_dict
  mean = state.get("obs_normalizer._mean")
  std = state.get("obs_normalizer._std")
  if mean is None or std is None:
    raise DistillationError(
      f"Teacher {teacher.id!r} has no observation normalizer statistics; the M1 gate "
      "requires the saved normalizer to place parity inputs"
    )
  mean = mean.reshape(-1).to(torch.float32)
  std = std.reshape(-1).to(torch.float32)

  probes = [
    torch.zeros_like(mean),
    mean.clone(),
    mean + std,
    mean - std,
    mean + 2.0 * std,
    mean - 2.0 * std,
  ]
  generator = torch.Generator(device="cpu").manual_seed(seed)
  rows = [probe.unsqueeze(0) for probe in probes[: min(samples, len(probes))]]
  remaining = samples - len(rows)
  if remaining > 0:
    noise = torch.randn((remaining, mean.numel()), generator=generator)
    rows.append(mean.unsqueeze(0) + std.unsqueeze(0) * noise)
  batch = torch.cat(rows, dim=0).contiguous()
  if not torch.isfinite(batch).all():
    raise DistillationError(f"Teacher {teacher.id!r} parity inputs are not finite")
  return batch


def build_time_steps(teacher: ResolvedTeacher, samples: int) -> np.ndarray:
  """Return valid reference frame indices, spread over the whole clip."""
  frames = teacher.reference.frames
  indices = np.linspace(0, frames - 1, samples).round().astype(np.int64)
  return indices.reshape(-1, 1)


def make_onnx_session(teacher: ResolvedTeacher) -> Any:
  """Create a CPU-only ONNX Runtime session for a teacher export."""
  ort = require_onnxruntime()
  return ort.InferenceSession(
    str(teacher.onnx.path), providers=["CPUExecutionProvider"]
  )


def run_onnx_actions(
  session: Any,
  teacher: ResolvedTeacher,
  observations: np.ndarray,
  time_steps: np.ndarray,
) -> np.ndarray:
  """Run the original export one row at a time, matching its batch-1 contract."""
  input_types = {value.name: value.type for value in session.get_inputs()}
  flat_type = np.dtype(_ONNX_NUMPY_DTYPES.get(input_types["obs"], np.float32))
  time_type = np.dtype(_ONNX_NUMPY_DTYPES.get(input_types["time_step"], np.float32))
  output_names = [value.name for value in session.get_outputs()]
  if "actions" not in output_names:
    raise DistillationError(
      f"ONNX export {teacher.onnx.path} has no 'actions' output (found {output_names})"
    )
  actions = np.empty(
    (observations.shape[0], teacher.actor.action_dim), dtype=np.float32
  )
  for row in range(observations.shape[0]):
    outputs = session.run(
      ["actions"],
      {
        "obs": observations[row : row + 1].astype(flat_type, copy=False),
        "time_step": time_steps[row : row + 1].astype(time_type, copy=False),
      },
    )
    actions[row] = outputs[0].reshape(-1)
  return actions


def compare_actions(
  native: torch.Tensor,
  exported: np.ndarray,
  atol: float,
  rtol: float,
  samples: int,
  seed: int,
) -> ParityResult:
  """Compare native teacher actions with the original ONNX actions.

  Non-finite actions on either side are rejected instead of compared: floating
  comparisons accept same-sign infinities, which would report a pair of broken
  outputs as a numerical pass.
  """
  native_array = native.detach().to(torch.float32).numpy()
  if native_array.shape != exported.shape:
    raise DistillationError(
      f"Native teacher actions {native_array.shape} do not match the ONNX actions "
      f"{exported.shape}"
    )
  if not np.isfinite(exported).all():
    raise DistillationError(
      f"Original ONNX actions contain {int((~np.isfinite(exported)).sum())} non-finite "
      "entries; the parity gate cannot pass"
    )
  if not np.isfinite(native_array).all():
    raise DistillationError(
      f"Native teacher actions contain {int((~np.isfinite(native_array)).sum())} "
      "non-finite entries; refusing to report a parity pass"
    )
  max_abs_error = _max_abs_difference(native_array, exported)
  scale = float(np.sqrt(np.mean(np.square(exported.astype(np.float64)))))
  return ParityResult(
    samples=samples,
    seed=seed,
    atol=atol,
    rtol=rtol,
    max_abs_error=max_abs_error,
    max_abs_error_over_rms=max_abs_error / scale if scale else 0.0,
    passed=bool(np.allclose(native_array, exported, atol=atol, rtol=rtol)),
  )


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
  """Return the maximum absolute element difference, or 0 for empty arrays."""
  if left.size == 0:
    return 0.0
  return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max())


def check_checkpoint_onnx_association(teacher: ResolvedTeacher) -> AssociationResult:
  """Check that the export was produced from this checkpoint.

  Exported parameters are compared with the saved actor tensors, and the folded
  normalizer divisor must equal the checkpoint standard deviation plus a single
  scalar epsilon.
  """
  state = teacher.actor_state_dict
  compared: list[str] = []
  missing: list[str] = []
  max_abs_diff = 0.0
  keys = sorted(key for key in state if key.startswith("mlp."))
  if "obs_normalizer._mean" in state:
    keys.append("obs_normalizer._mean")
  for key in keys:
    saved = state[key]
    exported = teacher.onnx.tensor(f"policy.{key}")
    if exported is None:
      exported = teacher.onnx.tensor(key)
    if exported is None:
      missing.append(key)
      continue
    if tuple(exported.shape) != tuple(saved.shape):
      raise DistillationError(
        f"Teacher {teacher.id!r} ONNX tensor {key!r} has shape {exported.shape} but the "
        f"checkpoint has {tuple(saved.shape)}"
      )
    max_abs_diff = max(max_abs_diff, _max_abs_difference(exported, saved.numpy()))
    compared.append(key)

  eps_spread: float | None = None
  divisor = find_normalizer_divisor(teacher.onnx, teacher.actor.obs_dim)
  std = state.get("obs_normalizer._std")
  if divisor is not None and std is not None:
    epsilon = divisor.reshape(-1) - std.numpy().reshape(-1)
    eps_spread = float(epsilon.max() - epsilon.min())
  return AssociationResult(
    compared=tuple(compared),
    missing=tuple(missing),
    max_abs_diff=max_abs_diff,
    normalizer_eps_spread=eps_spread,
    passed=not missing
    and max_abs_diff == 0.0
    and (eps_spread is None or eps_spread <= _NORMALIZER_EPS_TOLERANCE),
  )


def compare_reference_arrays(teacher: ResolvedTeacher) -> tuple[ReferenceResult, ...]:
  """Compare embedded ONNX reference joint arrays with the selected motion."""
  with np.load(teacher.reference.motion_path, allow_pickle=False) as data:
    results: list[ReferenceResult] = []
    for kind in ("joint_pos", "joint_vel"):
      selected = np.asarray(data[kind])
      found = teacher.onnx.reference_tensor(kind)
      if found is None:
        results.append(
          ReferenceResult(
            array=kind,
            found=False,
            shape=None,
            max_abs_diff=None,
            exact=False,
            passed=False,
          )
        )
        continue
      _, exported = found
      if exported.shape != selected.shape:
        raise DistillationError(
          f"Teacher {teacher.id!r} ONNX reference {kind!r} has shape {exported.shape} "
          f"but the selected motion has {selected.shape}"
        )
      difference = _max_abs_difference(exported, selected)
      results.append(
        ReferenceResult(
          array=kind,
          found=True,
          shape=tuple(int(d) for d in exported.shape),
          max_abs_diff=difference,
          exact=bool(np.array_equal(exported, selected)),
          passed=difference == 0.0,
        )
      )
  return tuple(results)


def validate_teachers(
  cohort: CohortContract,
  samples: int = DEFAULT_SAMPLES,
  seed: int = DEFAULT_SEED,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
  """Validate every teacher of a cohort and return a machine-readable report.

  The report is JSON-serializable and records the resolved contract, artifact
  hashes, per-teacher parity, association, and reference-array evidence, plus
  the fields M1 explicitly does not verify.
  """
  if samples < 1:
    raise ValueError(f"samples must be positive, got {samples}")
  for name, value in (("atol", atol), ("rtol", rtol)):
    if not np.isfinite(value) or value < 0.0:
      raise ValueError(f"{name} must be finite and non-negative, got {value}")

  bank = TeacherBank(
    [
      build_frozen_teacher(teacher.id, teacher.actor_state_dict, teacher.actor, "cpu")
      for teacher in cohort.teachers
    ],
    device="cpu",
  )
  teacher_reports: list[dict[str, Any]] = []
  for teacher in cohort.teachers:
    association = check_checkpoint_onnx_association(teacher)
    references = compare_reference_arrays(teacher)
    observations = build_parity_inputs(teacher, samples, seed)
    codes = torch.full((samples,), bank.code(teacher.id), dtype=torch.int64)
    native = bank.label(codes, observations)
    session = make_onnx_session(teacher)
    exported = run_onnx_actions(
      session, teacher, observations.numpy(), build_time_steps(teacher, samples)
    )
    parity = compare_actions(native, exported, atol, rtol, samples, seed)
    passed = (
      parity.passed
      and association.passed
      and all(result.passed for result in references)
    )
    teacher_reports.append(
      {
        "id": teacher.id,
        "sampling_weight": teacher.entry.sampling_weight,
        "hashes": dict(teacher.hashes),
        "reference": {
          "path": str(teacher.reference.motion_path),
          "declared_motion_file": teacher.reference.declared_motion_file,
          "frames": teacher.reference.frames,
          "fps": teacher.reference.fps,
          "duration_s": teacher.reference.duration_s,
          "array_shapes": {
            name: list(shape) for name, shape in teacher.reference.array_shapes.items()
          },
        },
        "unverified": list(teacher.unverified),
        "parity": {
          "samples": parity.samples,
          "seed": parity.seed,
          "atol": parity.atol,
          "rtol": parity.rtol,
          "max_abs_error": parity.max_abs_error,
          "max_abs_error_over_rms": parity.max_abs_error_over_rms,
          "passed": parity.passed,
        },
        "checkpoint_onnx_association": {
          "compared": list(association.compared),
          "missing": list(association.missing),
          "max_abs_diff": association.max_abs_diff,
          "normalizer_eps_spread": association.normalizer_eps_spread,
          "passed": association.passed,
        },
        "embedded_reference": [
          {
            "array": result.array,
            "found": result.found,
            "shape": None if result.shape is None else list(result.shape),
            "max_abs_diff": result.max_abs_diff,
            "exact": result.exact,
            "passed": result.passed,
          }
          for result in references
        ],
        "passed": passed,
      }
    )

  return {
    "manifest": {
      "path": str(cohort.manifest.path),
      "sha256": cohort.manifest.sha256,
      "version": cohort.manifest.version,
      "name": cohort.manifest.name,
      "robot": cohort.manifest.robot,
      "base_task": cohort.manifest.base_task,
      "repo_root": str(cohort.manifest.repo_root),
    },
    "actor": {
      "class_name": cohort.actor.class_name,
      "obs_dim": cohort.actor.obs_dim,
      "action_dim": cohort.actor.action_dim,
      "hidden_dims": list(cohort.actor.hidden_dims),
      "activation": cohort.actor.activation,
      "obs_normalization": cohort.actor.obs_normalization,
      "distribution_class_name": cohort.actor.distribution_class_name,
      "obs_groups": list(cohort.actor.obs_groups),
    },
    "observations": {
      "group": cohort.observations.group,
      "names": list(cohort.observations.names),
      "widths": list(cohort.observations.widths),
      "total_dim": cohort.observations.total_dim,
      "enable_corruption": cohort.observations.enable_corruption,
      "terms": [
        {
          "name": term.name,
          "func": term.func,
          "width": term.width,
          "scale": None if term.scale is None else list(term.scale),
          "clip": None if term.clip is None else list(term.clip),
          "history_length": term.history_length,
          "flatten_history_dim": term.flatten_history_dim,
          "noise": (
            None
            if term.noise is None
            else {"operation": term.noise.operation, "params": term.noise.params}
          ),
          "delay_min_lag": term.delay_min_lag,
          "delay_max_lag": term.delay_max_lag,
          "delay_hold_prob": term.delay_hold_prob,
          "delay_group": term.delay_group,
          "params": term.params,
        }
        for term in cohort.observations.terms
      ],
    },
    "actions": {
      "term": cohort.actions.term,
      "dim": cohort.actions.dim,
      "joint_names": list(cohort.actions.joint_names),
      "joint_scales": list(cohort.actions.joint_scales),
      "offset": cohort.actions.offset,
      "uses_default_offset": cohort.actions.uses_default_offset,
    },
    "control": {
      "sim_timestep": cohort.control.sim_timestep,
      "decimation": cohort.control.decimation,
      "control_period_s": cohort.control.control_period_s,
      "control_hz": cohort.control.control_hz,
    },
    "reference_fps": cohort.fps,
    "lookahead_s": cohort.lookahead_s,
    "anchor_body_name": cohort.anchor_body_name,
    "body_names": list(cohort.body_names),
    "sensors": _unique_sensor_declarations(cohort),
    "excluded_env_fields": list(cohort.excluded_env_fields),
    "unverified": list(cohort.unverified),
    "teachers": teacher_reports,
    "passed": all(report["passed"] for report in teacher_reports),
  }


def _unique_sensor_declarations(cohort: CohortContract) -> list[dict[str, str]]:
  declarations: list[dict[str, str]] = []
  for teacher in cohort.teachers:
    for sensor in teacher.sensors:
      item = {"term": sensor.term, "sensor_name": sensor.sensor_name}
      if item not in declarations:
        declarations.append(item)
  return declarations
