"""One shared teacher export serving a whole cohort.

A cohort member is a teacher plus one clip.  The weights and metadata are the
same for every clip, so the only per-member artifact that matters is the clip
file.  These tests lock the shared-export shape: the manifest key that names one
export, the per-entry override, the refusal when neither is given, the single
parse per resolution, and the degradation of the reference-array comparisons
that used to read a second copy of the clip from the export.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tracking_distillation_fixtures import build_tiny_cohort, write_manifest

from mjlab.tasks.tracking.distillation import config as config_module
from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.export import strip_teacher_reference_tables
from mjlab.tasks.tracking.distillation.parity import compare_reference_arrays


def _shared_manifest_body(root: Path, teachers, *, shared: Path, own: bool) -> str:
  lines = [
    "version: 1",
    "name: shared-cohort",
    "robot: agibot_x2",
    f"teacher_onnx: {shared.relative_to(root)}",
    "teachers:",
  ]
  for teacher in teachers:
    lines.append(f"  - id: {teacher.id}")
    for field, value in (
      ("checkpoint", teacher.checkpoint),
      ("motion", teacher.motion),
      ("env_config", teacher.env_config),
      ("agent_config", teacher.agent_config),
    ):
      lines.append(f"    {field}: {Path(value).relative_to(root)}")
    if own:
      lines.append(f"    onnx: {Path(teacher.onnx).relative_to(root)}")
  return "\n".join(lines) + "\n"


def test_shared_teacher_onnx_serves_every_member(tmp_path: Path) -> None:
  root = tmp_path
  teachers = build_tiny_cohort(root).teachers
  shared = teachers[0].onnx
  path = write_manifest(
    root / "configs" / "shared.yaml",
    teachers,
    root=root,
    body=_shared_manifest_body(root, teachers, shared=shared, own=False),
  )

  manifest = load_manifest(path, root)

  assert manifest.teacher_onnx == shared.resolve()
  assert {teacher.onnx for teacher in manifest.teachers} == {shared.resolve()}
  cohort = resolve_cohort(manifest)
  assert len(cohort.teachers) == 2


def test_per_entry_onnx_wins_over_the_shared_export(tmp_path: Path) -> None:
  root = tmp_path
  teachers = build_tiny_cohort(root).teachers
  path = write_manifest(
    root / "configs" / "own.yaml",
    teachers,
    root=root,
    body=_shared_manifest_body(root, teachers, shared=teachers[0].onnx, own=True),
  )

  manifest = load_manifest(path, root)

  for entry, teacher in zip(manifest.teachers, teachers, strict=True):
    assert entry.onnx == Path(teacher.onnx).resolve()


def test_a_member_without_any_teacher_onnx_is_refused(tmp_path: Path) -> None:
  root = tmp_path
  teachers = build_tiny_cohort(root).teachers
  body = _shared_manifest_body(root, teachers, shared=teachers[0].onnx, own=False)
  body = "\n".join(
    line for line in body.splitlines() if not line.startswith("teacher_onnx:")
  )
  path = write_manifest(
    root / "configs" / "absent.yaml", teachers, root=root, body=body + "\n"
  )

  with pytest.raises(DistillationError, match="teacher_onnx"):
    load_manifest(path, root)


def test_the_shared_export_is_parsed_once_per_resolution(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """At 3 261 members a parse per member cost 913 s and 52.9 GB of arrays."""
  root = tmp_path
  teachers = build_tiny_cohort(root).teachers
  shared = teachers[0].onnx
  path = write_manifest(
    root / "configs" / "once.yaml",
    teachers,
    root=root,
    body=_shared_manifest_body(root, teachers, shared=shared, own=False),
  )

  calls: list[Path] = []
  original = config_module.load_onnx_artifact

  def counting(onnx_path: Path, where: str):
    calls.append(Path(onnx_path))
    return original(onnx_path, where)

  monkeypatch.setattr(config_module, "load_onnx_artifact", counting)
  resolve_cohort(load_manifest(path, root))

  assert len(calls) == 1, f"parsed the shared export {len(calls)} times"


def test_reference_array_parity_reports_nothing_without_embedded_tables(
  tmp_path: Path,
) -> None:
  """A shared export embeds no tables, so there is nothing to compare."""
  root = tmp_path
  teachers = build_tiny_cohort(root).teachers
  shared = Path(teachers[0].onnx)
  strip_teacher_reference_tables(shared)
  path = write_manifest(
    root / "configs" / "stripped.yaml",
    teachers,
    root=root,
    body=_shared_manifest_body(root, teachers, shared=shared, own=False),
  )

  cohort = resolve_cohort(load_manifest(path, root))

  assert compare_reference_arrays(cohort.teachers[0]) == ()


def test_a_per_clip_export_keeps_its_reference_parity(tmp_path: Path) -> None:
  root = tmp_path
  teachers = build_tiny_cohort(root).teachers
  path = write_manifest(
    root / "configs" / "own_parity.yaml",
    teachers,
    root=root,
    body=_shared_manifest_body(root, teachers, shared=Path(teachers[0].onnx), own=True),
  )

  cohort = resolve_cohort(load_manifest(path, root))

  results = compare_reference_arrays(cohort.teachers[0])
  assert results, "a per-clip export must still be checked against its own motion"
  assert all(result.passed for result in results)
