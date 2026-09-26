"""Manifest, path-resolution, and compatibility validation for teacher cohorts."""

from __future__ import annotations

from pathlib import Path

import pytest
from tracking_distillation_fixtures import (
  DEFAULT_TERMS,
  build_tiny_cohort,
  sha256_file,
  write_manifest,
  write_teacher,
)

from mjlab.tasks.tracking.distillation.config import (
  DistillationError,
  UnsupportedTeacherError,
  load_manifest,
  resolve_cohort,
)
from mjlab.utils.os import load_saved_yaml

ARTIFACT_FIELDS = ("checkpoint", "motion", "env_config", "agent_config", "onnx")


def single_teacher_cohort(root: Path, teacher_id: str = "solo"):
  """Write one teacher plus a manifest, so no cohort equality can mask a fault."""
  teacher = write_teacher(root, teacher_id)
  manifest = write_manifest(root / "configs" / "solo.yaml", [teacher], root=root)
  return teacher, manifest


def test_load_manifest_resolves_relative_paths_and_hashes(tmp_path: Path) -> None:
  cohort = build_tiny_cohort(tmp_path)

  manifest = load_manifest(cohort.manifest, tmp_path)

  assert manifest.name == "tiny-cohort"
  assert manifest.version == 1
  assert manifest.robot == "agibot_x2"
  assert manifest.repo_root == tmp_path.resolve()
  assert [entry.id for entry in manifest.teachers] == ["tiny_000", "tiny_001"]
  for entry, teacher in zip(manifest.teachers, cohort.teachers, strict=True):
    for field in ARTIFACT_FIELDS:
      assert getattr(entry, field) == getattr(teacher, field)
    assert entry.sampling_weight == 1.0


def test_load_manifest_defaults_repo_root_to_working_directory(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  build_tiny_cohort(tmp_path)
  monkeypatch.chdir(tmp_path)

  manifest = load_manifest(Path("configs/tiny_teachers.yaml"))

  assert manifest.repo_root == tmp_path.resolve()
  assert manifest.teachers[0].checkpoint.is_file()


def test_load_manifest_reports_missing_artifacts(tmp_path: Path) -> None:
  cohort = build_tiny_cohort(tmp_path)
  text = cohort.manifest.read_text().replace(
    "runs/tiny_001/model_5.pt", "runs/tiny_001/model_absent.pt"
  )
  cohort.manifest.write_text(text)

  with pytest.raises(DistillationError, match="Missing teacher artifacts"):
    load_manifest(cohort.manifest, tmp_path)


def test_load_manifest_rejects_python_tags_without_executing(tmp_path: Path) -> None:
  cohort = build_tiny_cohort(tmp_path)
  marker = tmp_path / "executed"
  cohort.manifest.write_text(
    cohort.manifest.read_text().replace(
      "  - id: tiny_000",
      f'  - id: !!python/object/apply:os.system ["touch {marker}"]',
    )
  )

  with pytest.raises(DistillationError, match="plain YAML"):
    load_manifest(cohort.manifest, tmp_path)
  assert not marker.exists()


@pytest.mark.parametrize("teacher_id", ["", "   "])
def test_load_manifest_rejects_empty_teacher_ids(
  tmp_path: Path, teacher_id: str
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  cohort.manifest.write_text(
    cohort.manifest.read_text().replace("  - id: tiny_000", f'  - id: "{teacher_id}"')
  )

  with pytest.raises(DistillationError, match="non-empty string"):
    load_manifest(cohort.manifest, tmp_path)


def test_load_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
  cohort = build_tiny_cohort(tmp_path)
  cohort.manifest.write_text(
    cohort.manifest.read_text().replace("  - id: tiny_001", "  - id: tiny_000")
  )

  with pytest.raises(DistillationError, match="not unique"):
    load_manifest(cohort.manifest, tmp_path)


def test_load_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
  cohort = build_tiny_cohort(tmp_path)
  cohort.manifest.write_text(
    cohort.manifest.read_text().replace(
      "    sampling_weight: 1.0", "    sampling_weight: 1.0\n    class_name: evil"
    )
  )

  with pytest.raises(DistillationError, match="unsupported keys"):
    load_manifest(cohort.manifest, tmp_path)


@pytest.mark.parametrize("weight", ["0.0", "-1.0", ".nan"])
def test_load_manifest_rejects_invalid_sampling_weight(
  tmp_path: Path, weight: str
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  cohort.manifest.write_text(
    cohort.manifest.read_text().replace(
      "sampling_weight: 1.0", f"sampling_weight: {weight}"
    )
  )

  with pytest.raises(DistillationError, match="sampling_weight"):
    load_manifest(cohort.manifest, tmp_path)


def test_saved_yaml_decodes_python_tags_without_executing(tmp_path: Path) -> None:
  marker = tmp_path / "executed"
  path = tmp_path / "env.yaml"
  path.write_text(
    f"""names: !!python/tuple [a, b]
func: !!python/name:mjlab.envs.mdp.observations.generated_commands ''
nested:
  fn: !!python/name:mjlab.entity.entity.%3Clambda%3E ''
  unknown: !!python/object/apply:os.system ["touch {marker}"]
"""
  )

  loaded = load_saved_yaml(path)

  assert loaded["names"] == ("a", "b")
  assert loaded["func"] == "mjlab.envs.mdp.observations.generated_commands"
  assert loaded["nested"]["fn"] == "mjlab.entity.entity.<lambda>"
  assert loaded["nested"]["unknown"] == (f"touch {marker}",)
  assert not marker.exists()


def test_resolve_cohort_matches_selected_teacher_contract(tmp_path: Path) -> None:
  cohort = build_tiny_cohort(tmp_path, terms=DEFAULT_TERMS)

  contract = resolve_cohort(load_manifest(cohort.manifest, tmp_path))

  assert contract.teacher_ids == ("tiny_000", "tiny_001")
  assert contract.actor.obs_dim == 24
  assert contract.actor.action_dim == 3
  assert contract.actor.hidden_dims == (8, 4)
  assert contract.actor.activation == "elu"
  assert contract.actor.obs_normalization is True
  assert contract.actor.obs_groups == ("actor",)
  assert contract.actor.distribution_class_name == "GaussianDistribution"
  assert contract.observations.names == DEFAULT_TERMS
  assert contract.observations.widths == (6, 0, 6, 3, 3, 3, 3)
  assert contract.observations.total_dim == 24
  assert contract.lookahead_s == 0.0
  assert contract.actions.joint_names == cohort.joint_names
  assert contract.actions.joint_scales == (0.3, 0.3, 0.5)
  assert contract.actions.clip is None
  assert contract.actions.offset == 0.0
  assert contract.actions.uses_default_offset is True
  assert contract.control.control_hz == 50.0
  assert contract.control.sim_timestep == 0.005
  assert contract.control.decimation == 4
  assert contract.fps == 50.0
  assert contract.anchor_body_name == "torso_link"
  assert contract.body_names == ("pelvis", "torso_link")
  assert contract.excluded_env_fields == ("commands.motion.motion_file",)

  teacher = contract.teacher("tiny_000")
  assert teacher.reference.frames == 6
  assert teacher.reference.joint_dim == 3
  assert teacher.reference.declared_motion_file.endswith("tiny_000_tracking.npz")
  assert teacher.hashes["onnx"] == sha256_file(cohort.teachers[0].onnx)
  assert [(sensor.term, sensor.sensor_name) for sensor in teacher.sensors] == [
    ("base_ang_vel", "robot/imu_ang_vel")
  ]
  assert any("robot asset" in note for note in contract.unverified)


def test_resolve_cohort_accepts_a_single_teacher(tmp_path: Path) -> None:
  teacher = write_teacher(tmp_path, "solo")
  manifest = write_manifest(
    tmp_path / "configs" / "solo.yaml", [teacher], root=tmp_path
  )

  contract = resolve_cohort(load_manifest(manifest, tmp_path))

  assert contract.teacher_ids == ("solo",)


def test_resolve_cohort_rejects_motion_override_with_a_different_clip(
  tmp_path: Path,
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  other = tmp_path / "data" / "tennis" / "unrelated.npz"
  other.write_bytes(cohort.teachers[0].motion.read_bytes())
  cohort.manifest.write_text(
    cohort.manifest.read_text().replace(
      "data/tennis/tiny_000_tracking.npz", "data/tennis/unrelated.npz"
    )
  )

  with pytest.raises(DistillationError, match="must match the saved reference"):
    resolve_cohort(load_manifest(cohort.manifest, tmp_path))


def test_resolve_cohort_rejects_reference_fps_mismatch(tmp_path: Path) -> None:
  teacher = write_teacher(tmp_path, "solo", fps=25.0)
  manifest = write_manifest(
    tmp_path / "configs" / "solo.yaml", [teacher], root=tmp_path
  )

  with pytest.raises(DistillationError, match="must agree"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_resolve_cohort_rejects_observation_width_mismatch(tmp_path: Path) -> None:
  teacher = write_teacher(tmp_path, "solo")
  manifest = write_manifest(
    tmp_path / "configs" / "solo.yaml", [teacher], root=tmp_path
  )
  # Swap a width-3 term for a width-6 term without re-exporting the ONNX.
  teacher.env_config.write_text(
    teacher.env_config.read_text().replace(
      "mjlab.envs.mdp.observations.joint_vel_rel",
      "mjlab.tasks.tracking.mdp.observations.motion_anchor_ori_b",
    )
  )

  with pytest.raises(DistillationError, match="sum to 27"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_resolve_cohort_rejects_observation_metadata_order_mismatch(
  tmp_path: Path,
) -> None:
  teacher = write_teacher(
    tmp_path,
    "solo",
    observation_names=("actions", *DEFAULT_TERMS[:-1]),
  )
  manifest = write_manifest(
    tmp_path / "configs" / "solo.yaml", [teacher], root=tmp_path
  )

  with pytest.raises(DistillationError, match="observation_names"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_resolve_cohort_rejects_differing_saved_environments(tmp_path: Path) -> None:
  teachers = [
    write_teacher(tmp_path, "tiny_000", seed=0),
    write_teacher(tmp_path, "tiny_001", seed=1, joint_position_noise_min=-0.02),
  ]
  manifest = write_manifest(tmp_path / "configs" / "tiny.yaml", teachers, root=tmp_path)

  with pytest.raises(DistillationError, match="different saved environment"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_resolve_cohort_rejects_recurrent_teacher(tmp_path: Path) -> None:
  teacher = write_teacher(tmp_path, "solo", rnn_type="lstm")
  manifest = write_manifest(
    tmp_path / "configs" / "solo.yaml", [teacher], root=tmp_path
  )

  with pytest.raises(UnsupportedTeacherError, match="rnn_type"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_resolve_cohort_rejects_legacy_checkpoint_format(tmp_path: Path) -> None:
  import torch

  teacher = write_teacher(tmp_path, "solo")
  torch.save(
    {
      "model_state_dict": {"actor.mlp.0.weight": torch.zeros(8, 24)},
      "iter": 0,
      "infos": {},
    },
    teacher.checkpoint,
  )
  manifest = write_manifest(
    tmp_path / "configs" / "solo.yaml", [teacher], root=tmp_path
  )

  with pytest.raises(UnsupportedTeacherError, match="model_state_dict"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_load_manifest_resolves_relative_manifest_against_repo_root(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  elsewhere = tmp_path / "elsewhere"
  elsewhere.mkdir()
  monkeypatch.chdir(elsewhere)

  manifest = load_manifest(Path("configs/tiny_teachers.yaml"), tmp_path)

  assert manifest.path == (tmp_path / "configs" / "tiny_teachers.yaml").resolve()
  assert manifest.teachers[0].checkpoint.is_file()
  # The current working directory must not be used for a relative manifest.
  with pytest.raises(DistillationError, match="Manifest file not found"):
    load_manifest(Path("configs/tiny_teachers.yaml"), elsewhere)
  assert cohort.manifest.is_file()


def test_load_manifest_keeps_absolute_manifest_and_artifact_paths(
  tmp_path: Path,
) -> None:
  cohort = build_tiny_cohort(tmp_path)
  teacher = cohort.teachers[0]
  manifest_path = tmp_path / "configs" / "absolute.yaml"
  manifest_path.write_text(
    "version: 1\nname: absolute\nrobot: agibot_x2\nteachers:\n"
    f"  - id: solo\n    checkpoint: {teacher.checkpoint}\n    motion: {teacher.motion}\n"
    f"    env_config: {teacher.env_config}\n    agent_config: {teacher.agent_config}\n"
    f"    onnx: {teacher.onnx}\n    sampling_weight: 1.0\n"
  )
  unused_root = tmp_path / "unused"
  unused_root.mkdir()

  manifest = load_manifest(manifest_path, unused_root)

  assert manifest.path == manifest_path.resolve()
  assert manifest.repo_root == unused_root.resolve()
  assert manifest.teachers[0].checkpoint == teacher.checkpoint
  assert manifest.teachers[0].onnx == teacher.onnx


def test_resolve_cohort_rejects_saved_anchor_that_disagrees_with_export(
  tmp_path: Path,
) -> None:
  teacher, manifest = single_teacher_cohort(tmp_path)
  teacher.env_config.write_text(
    teacher.env_config.read_text().replace(
      "anchor_body_name: torso_link", "anchor_body_name: pelvis"
    )
  )

  with pytest.raises(DistillationError, match="anchor_body_name 'pelvis'"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_resolve_cohort_rejects_saved_body_order_that_disagrees_with_export(
  tmp_path: Path,
) -> None:
  teacher, manifest = single_teacher_cohort(tmp_path)
  teacher.env_config.write_text(
    teacher.env_config.read_text().replace(
      "    - pelvis\n    - torso_link", "    - torso_link\n    - pelvis"
    )
  )

  with pytest.raises(DistillationError, match="saved tracked bodies"):
    resolve_cohort(load_manifest(manifest, tmp_path))


def test_resolve_cohort_rejects_runner_action_clipping(tmp_path: Path) -> None:
  teacher, manifest = single_teacher_cohort(tmp_path)
  teacher.agent_config.write_text(
    teacher.agent_config.read_text().replace("clip_actions: null", "clip_actions: 0.1")
  )

  with pytest.raises(UnsupportedTeacherError, match="clip_actions"):
    resolve_cohort(load_manifest(manifest, tmp_path))
