"""Native-vs-ONNX parity, artifact association, and reference comparisons."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tracking_distillation_fixtures import (
  DEFAULT_TEACHER_IDS,
  build_tiny_cohort,
  write_manifest,
)

from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  load_manifest,
  load_onnx_artifact,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.parity import (
  build_parity_inputs,
  build_time_steps,
  compare_actions,
  validate_teachers,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _require_validation_dependencies():
  pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")


def resolve_tiny_cohort(root: Path):
  cohort = build_tiny_cohort(root)
  return cohort, resolve_cohort(load_manifest(cohort.manifest, root))


def test_validate_teachers_reports_parity_for_generated_cohort(tmp_path: Path) -> None:
  cohort, contract = resolve_tiny_cohort(tmp_path)

  report = validate_teachers(contract, samples=16, seed=3)

  assert report["passed"] is True
  assert report["manifest"]["path"] == str(cohort.manifest.resolve())
  assert report["actor"]["obs_dim"] == cohort.obs_dim
  assert report["control"]["control_hz"] == 50.0
  assert report["lookahead_s"] == 0.0
  assert [teacher["id"] for teacher in report["teachers"]] == list(DEFAULT_TEACHER_IDS)
  for teacher in report["teachers"]:
    assert teacher["passed"] is True
    assert teacher["parity"]["samples"] == 16
    assert teacher["parity"]["max_abs_error"] <= teacher["parity"]["atol"]
    assert teacher["checkpoint_onnx_association"]["passed"] is True
    assert teacher["checkpoint_onnx_association"]["missing"] == []
    assert all(item["exact"] for item in teacher["embedded_reference"])


def test_validate_teachers_is_deterministic(tmp_path: Path) -> None:
  _, contract = resolve_tiny_cohort(tmp_path)

  first = validate_teachers(contract, samples=8, seed=1)
  second = validate_teachers(contract, samples=8, seed=1)

  assert first["teachers"] == second["teachers"]


def test_validate_teachers_flags_an_onnx_export_from_another_checkpoint(
  tmp_path: Path,
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  swapped = [
    replace(cohort.teachers[0], onnx=cohort.teachers[1].onnx),
    cohort.teachers[1],
  ]
  manifest = write_manifest(
    tmp_path / "configs" / "swapped.yaml", swapped, root=tmp_path
  )

  report = validate_teachers(
    resolve_cohort(load_manifest(manifest, tmp_path)), samples=8
  )

  assert report["passed"] is False
  mismatched = report["teachers"][0]
  assert mismatched["passed"] is False
  assert mismatched["checkpoint_onnx_association"]["max_abs_diff"] > 0.0
  assert any(item["exact"] is False for item in mismatched["embedded_reference"])


def test_validate_teachers_flags_a_modified_reference_motion(tmp_path: Path) -> None:
  cohort, contract = resolve_tiny_cohort(tmp_path)
  motion = cohort.teachers[0].motion
  with np.load(motion, allow_pickle=False) as data:
    arrays = {name: data[name] for name in data.files}
  arrays["joint_vel"] = arrays["joint_vel"] + np.float32(0.5)
  np.savez(motion, **arrays)

  report = validate_teachers(resolve_cohort(load_manifest(cohort.manifest, tmp_path)))

  teacher = report["teachers"][0]
  assert report["passed"] is False
  reference = {item["array"]: item for item in teacher["embedded_reference"]}
  assert reference["joint_vel"]["passed"] is False
  assert reference["joint_vel"]["max_abs_diff"] == pytest.approx(0.5)
  assert reference["joint_pos"]["exact"] is True


def test_compare_actions_reports_tolerance_failures() -> None:
  native = torch.zeros(3, 2)
  exported = np.zeros((3, 2), dtype=np.float32)

  passing = compare_actions(native, exported, 1e-5, 1e-5, 3, 0)
  assert passing.passed is True
  assert passing.max_abs_error == 0.0

  perturbed = exported + np.float32(1e-3)
  failing = compare_actions(native, perturbed, 1e-5, 1e-5, 3, 0)
  assert failing.passed is False
  assert failing.max_abs_error == pytest.approx(1e-3, rel=1e-3)
  assert failing.max_abs_error_over_rms > 0.0

  with pytest.raises(ValueError, match="do not match"):
    compare_actions(torch.zeros(3, 3), exported, 1e-5, 1e-5, 3, 0)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_compare_actions_rejects_non_finite_actions(value: float) -> None:
  exported = np.array([[value]], dtype=np.float32)
  with pytest.raises(DistillationError, match="ONNX actions contain 1 non-finite"):
    compare_actions(torch.tensor([[value]]), exported, 1e-5, 1e-5, 1, 0)
  with pytest.raises(
    DistillationError, match="Native teacher actions contain 1 non-finite"
  ):
    compare_actions(
      torch.tensor([[value]]), np.zeros((1, 1), dtype=np.float32), 1e-5, 1e-5, 1, 0
    )


def test_compare_actions_still_rejects_finite_mismatch_at_default_tolerance() -> None:
  result = compare_actions(
    torch.zeros(1, 1),
    np.array([[1e-3]], dtype=np.float32),
    1e-5,
    1e-5,
    1,
    0,
  )

  assert result.passed is False
  assert np.isfinite(result.max_abs_error)
  assert result.max_abs_error == pytest.approx(1e-3, rel=1e-3)


def test_parity_inputs_cover_the_saved_normalization_statistics(
  tmp_path: Path,
) -> None:
  _, contract = resolve_tiny_cohort(tmp_path)
  teacher = contract.teacher("tiny_000")

  batch = build_parity_inputs(teacher, 10, seed=0)

  assert batch.shape == (10, contract.actor.obs_dim)
  assert torch.isfinite(batch).all()
  mean = teacher.actor_state_dict["obs_normalizer._mean"].reshape(-1)
  std = teacher.actor_state_dict["obs_normalizer._std"].reshape(-1)
  assert torch.equal(batch[0], torch.zeros_like(mean))
  assert torch.allclose(batch[1], mean)
  assert torch.allclose(batch[2], mean + std)
  assert torch.allclose(batch[5], mean - 2.0 * std)
  assert build_parity_inputs(teacher, 4, seed=0).shape == (4, batch.shape[1])
  with pytest.raises(ValueError, match="samples must be positive"):
    build_parity_inputs(teacher, 0, seed=0)


def test_time_steps_stay_inside_the_reference_clip(tmp_path: Path) -> None:
  _, contract = resolve_tiny_cohort(tmp_path)
  teacher = contract.teacher("tiny_001")
  frames = teacher.reference.frames

  steps = build_time_steps(teacher, 9)

  assert steps.shape == (9, 1)
  assert steps.min() == 0
  assert steps.max() == frames - 1
  assert steps.max() < frames


def test_reference_tensor_lookup_accepts_constant_nodes() -> None:
  """Exports may embed reference data as Constant attributes, not initializers."""
  onnx = pytest.importorskip("onnx")

  joint_pos = np.arange(12, dtype=np.float32).reshape(4, 3)
  constant = onnx.helper.make_node(
    "Constant",
    inputs=[],
    outputs=["joint_pos"],
    value=onnx.numpy_helper.from_array(joint_pos, name="joint_pos_value"),
  )
  initializer = onnx.numpy_helper.from_array(
    np.zeros((4, 3), dtype=np.float32), name="joint_vel.1"
  )
  graph = onnx.helper.make_graph(
    [constant],
    "reference_tensors",
    [],
    [],
    initializer=[initializer],
  )
  model = onnx.helper.make_model(graph)

  import tempfile

  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "reference.onnx"
    onnx.save(model, str(path))
    artifact = load_onnx_artifact(path, "generated reference export")

  found = artifact.reference_tensor("joint_pos")
  assert found is not None
  assert np.array_equal(found[1], joint_pos)
  assert artifact.reference_tensor("joint_vel") is not None
  assert artifact.reference_tensor("body_pos_w") is None


def test_parity_inputs_require_saved_normalizer_statistics(tmp_path: Path) -> None:
  _, contract = resolve_tiny_cohort(tmp_path)
  teacher = contract.teacher("tiny_000")
  without_normalizer = replace(
    teacher,
    actor_state_dict={
      key: value
      for key, value in teacher.actor_state_dict.items()
      if not key.startswith("obs_normalizer.")
    },
  )

  with pytest.raises(DistillationError, match="normalizer statistics"):
    build_parity_inputs(without_normalizer, 4, seed=0)
