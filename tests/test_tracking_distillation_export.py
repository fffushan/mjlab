from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper
from tracking_distillation_fixtures import build_tiny_cohort

from mjlab.tasks.tracking.distillation import export as export_module
from mjlab.tasks.tracking.distillation.checkpoint import CHECKPOINT_VERSION
from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort
from mjlab.tasks.tracking.distillation.export import (
  ExportValidationError,
  export_bundle,
  make_export_audit,
  validate_export_parity,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.vae_config import make_schema


def _audit(joint_names: tuple[str, ...]) -> dict:
  return {
    "producer": "mjlab.tasks.tracking.distillation.export.make_export_audit",
    "verification": "audited",
    "asset_identity": {"path": "tiny.xml", "sha256": "0" * 64},
    "robot_bodies": ["pelvis", "torso_link", "extra", "extra2"],
    "tracked_bodies": ["pelvis", "torso_link"],
    "tracked_body_indices": [0, 1],
    "root_frame": "pelvis",
    "anchor_body": "torso_link",
    "gravity": {"source": "pelvis_imu", "frame": "pelvis"},
    "gyro": {
      "source": "pelvis_gyro",
      "frame": "pelvis",
      "sensor_name": "pelvis_gyro",
      "site_name": "pelvis_imu_site",
      "body_name": "pelvis",
      "local_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    },
    "anchor": {"source": "waist_fk", "frame": "torso_link"},
    "action_metadata": {
      "joint_names": list(joint_names),
      "default_joint_position": [0.000000123] * len(joint_names),
      "kp": [1.0000001] * len(joint_names),
      "kd": [0.9999999] * len(joint_names),
    },
  }


def _checkpoint(
  path: Path,
  *,
  schema,
  model: ConditionalVAE,
  teacher_hashes: dict[str, str],
  control_contract: dict,
) -> None:
  torch.save(
    {
      "kind": "mjlab-m3-distillation",
      "version": CHECKPOINT_VERSION,
      "schema": schema.compatibility_metadata(),
      "model_settings": model.settings.to_metadata(),
      "model": model.state_dict(),
      "teacher_hashes": teacher_hashes,
      "control_contract": control_contract,
      "counters": {},
      "schedule": {},
      "resolved_config": {},
    },
    path,
  )


def _fixture(tmp_path: Path):
  names = (
    *(f"hip_{i:02d}_hip_joint" for i in range(30)),
    "waist_joint",
  )
  cohort = build_tiny_cohort(
    tmp_path, teacher_ids=("tiny_000",), joint_names=names, frames=4
  )
  teacher = cohort.teachers[0]
  frames = 4
  q = np.arange(frames * 31, dtype=np.float32).reshape(frames, 31)
  dq = (q + np.float32(100.25)).astype(np.float32)
  anchor = np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (frames, 1))
  body_quat = np.zeros((frames, 2, 4), dtype=np.float32)
  body_quat[:, 1] = anchor
  with np.load(teacher.motion, allow_pickle=False) as old:
    body_arrays = {
      name: old[name].astype(np.float32)
      for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    }
    body_arrays["body_quat_w"][:, 1] = anchor
    full_shapes = {
      "body_pos_w": (3,),
      "body_quat_w": (4,),
      "body_lin_vel_w": (3,),
      "body_ang_vel_w": (3,),
    }
    full_arrays = {}
    for name, array in body_arrays.items():
      full = np.zeros((frames, 4, *full_shapes[name]), dtype=np.float32)
      full[:, :2] = array
      full[:, 2] = np.float32(7.0)
      full[:, 3] = np.float32(9.0)
      full_arrays[name] = full
    np.savez(
      teacher.motion,
      joint_pos=q,
      joint_vel=dq,
      **full_arrays,
      fps=np.array([50.0], dtype=np.float32),
    )
  graph = onnx.load(teacher.onnx)
  for name, array in full_arrays.items():
    graph.graph.initializer.append(numpy_helper.from_array(array[:, [0, 1]], name=name))
  onnx.save(graph, teacher.onnx)
  resolved = resolve_cohort(load_manifest(cohort.manifest, tmp_path))
  resolved_teacher = resolved.teacher("tiny_000")
  schema = make_schema(joint_order=names)
  model = ConditionalVAE(schema)
  model.reference_normalizer.update(torch.randn(8, 68))
  model.conditioning_normalizer.update(torch.randn(8, 99))
  model.reference_normalizer.freeze()
  model.conditioning_normalizer.freeze()
  checkpoint = tmp_path / "checkpoint.pt"
  control = {
    "control_period_s": resolved_teacher.control.control_period_s,
    "control_hz": resolved_teacher.control.control_hz,
    "sim_timestep": resolved_teacher.control.sim_timestep,
    "decimation": resolved_teacher.control.decimation,
    "action_joint_names": list(resolved_teacher.actions.joint_names),
    "action_dim": resolved_teacher.actions.dim,
    "action_scales": list(resolved_teacher.actions.joint_scales),
    "action_offset": resolved_teacher.actions.offset,
    "teacher_id": "tiny_000",
    "motion": str(resolved_teacher.entry.motion),
    "task": resolved.manifest.base_task,
  }
  _checkpoint(
    checkpoint,
    schema=schema,
    model=model,
    teacher_hashes=resolved_teacher.hashes,
    control_contract=control,
  )
  return cohort, schema, model, checkpoint


def test_audit_rejects_root_frame_or_sensor_identity_mismatch() -> None:
  teacher = SimpleNamespace(
    body_names=("pelvis", "torso_link"), anchor_body_name="torso_link"
  )
  audit = _audit(tuple(f"joint_{i}" for i in range(31)))
  audit["gravity"]["frame"] = "torso_link"
  with pytest.raises(ExportValidationError, match="gravity frame"):
    export_module._require_audited_provenance(audit, teacher)
  audit = _audit(tuple(f"joint_{i}" for i in range(31)))
  audit["gyro"]["sensor_name"] = None
  audit["gyro"]["site_name"] = ""
  with pytest.raises(ExportValidationError, match="gyro"):
    export_module._require_audited_provenance(audit, teacher)


def test_audit_producer_binds_compiled_asset_and_control_values(
  tmp_path: Path, monkeypatch
) -> None:
  cohort, _, _, _ = _fixture(tmp_path)
  resolved = resolve_cohort(load_manifest(cohort.manifest, tmp_path))
  teacher = resolved.teacher("tiny_000")
  asset = SimpleNamespace(
    robot_bodies=("pelvis", "torso_link"),
    tracked_bodies=teacher.body_names,
    tracked_body_indices=(0, 1),
    root_frame="pelvis",
    anchor_body="torso_link",
    anchor_body_id=1,
    sensor_evidence=(
      SimpleNamespace(
        expected_type="gyro",
        name="pelvis_gyro",
        site_name="pelvis_site",
        body_name="pelvis",
        local_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
      ),
    ),
  )
  monkeypatch.setattr(
    export_module,
    "validate_live_contract",
    lambda *args, **kwargs: SimpleNamespace(asset=asset),
  )
  action_term = SimpleNamespace(
    cfg=SimpleNamespace(entity_name="robot"),
    target_ids=torch.arange(31),
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(default_joint_pos=torch.zeros(1, 31)),
  )
  model = SimpleNamespace(
    actuator=lambda name: SimpleNamespace(id=0),
    actuator_gainprm=np.ones((1, 1), dtype=np.float32),
    actuator_biasprm=np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
  )
  env = SimpleNamespace(
    action_manager=SimpleNamespace(get_term=lambda name: action_term),
    scene={"robot": robot},
    sim=SimpleNamespace(mj_model=model),
  )
  asset_path = tmp_path / "robot.xml"
  asset_path.write_text("compiled-asset")
  audit = make_export_audit(env, resolved, "tiny_000", asset_path=asset_path)
  assert audit["producer"].endswith("make_export_audit")
  assert audit["asset_identity"]["sha256"]
  assert audit["action_metadata"]["kp"] == [1.0] * 31


def test_export_emits_hash_bound_graphs_motion_and_ort_parity(tmp_path: Path) -> None:
  cohort, schema, model, checkpoint = _fixture(tmp_path)
  result = export_bundle(
    checkpoint,
    cohort.manifest,
    "tiny_000",
    tmp_path / "rl_model",
    repo_root=tmp_path,
    asset_audit=_audit(schema.joint_order),
  )
  descriptor = json.loads(result.descriptor.read_text())
  contract = json.loads(result.contract.read_text())
  motion = json.loads(result.motion.read_text())
  assert descriptor["version"] == 2
  assert descriptor["model_id"] == contract["model_id"]
  assert descriptor["contract_id"] == contract["contract_id"]
  assert set(descriptor["files"]) == {
    "encoder.onnx",
    "decoder.onnx",
    "contract.json",
    "motion.json",
    "parity.json",
  }
  for _name, item in descriptor["files"].items():
    path = result.descriptor.parent / item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
  assert np.asarray(motion["q"], dtype=np.float32).shape == (4, 31)
  assert np.array_equal(
    np.asarray(motion["q"], dtype=np.float32)[:, 0],
    np.arange(0, 124, 31, dtype=np.float32),
  )
  parity_fixture = json.loads(
    (result.descriptor.parent / "vae-tiny_000" / "parity.json").read_text()
  )
  assert parity_fixture["fixture"]["model_id"] == result.model_id
  assert len(parity_fixture["fixture"]["reference"][0]) == 68
  assert contract["action"]["kp"][0] == 1.0000001
  assert result.report["parity"]["latent_max_abs_error"] <= 1e-5
  assert result.report["parity"]["action_max_abs_error"] <= 1e-5
  assert result.report["parity"]["fixture"]["model_id"] == result.model_id
  assert result.report["parity"]["fixture"]["contract_id"] == result.contract_id
  parity = validate_export_parity(result, model, torch.randn(1, 68), torch.randn(1, 99))
  assert parity["latent_max_abs_error"] <= 1e-5
  assert parity["action_max_abs_error"] <= 1e-5


def test_export_refuses_unverified_provenance_wrong_binding_and_non_gravity_schema(
  tmp_path: Path,
) -> None:
  cohort, schema, _, checkpoint = _fixture(tmp_path)
  bad_audit = _audit(schema.joint_order)
  bad_audit["verification"] = "declared_unverified"
  with pytest.raises(ExportValidationError, match="verification"):
    export_bundle(
      checkpoint,
      cohort.manifest,
      "tiny_000",
      tmp_path / "out",
      repo_root=tmp_path,
      asset_audit=bad_audit,
    )
  wrong_index = _audit(schema.joint_order)
  wrong_index["tracked_body_indices"] = [1, 0]
  with pytest.raises(ExportValidationError, match="anchor index|body binding"):
    export_bundle(
      checkpoint,
      cohort.manifest,
      "tiny_000",
      tmp_path / "out-wrong-index",
      repo_root=tmp_path,
      asset_audit=wrong_index,
    )

  resolved = resolve_cohort(load_manifest(cohort.manifest, tmp_path))
  teacher = resolved.teacher("tiny_000")
  anchor_schema = make_schema("anchor", schema.joint_order)
  anchor_model = ConditionalVAE(anchor_schema)
  anchor_checkpoint = tmp_path / "anchor.pt"
  control = {
    "control_period_s": teacher.control.control_period_s,
    "control_hz": teacher.control.control_hz,
    "sim_timestep": teacher.control.sim_timestep,
    "decimation": teacher.control.decimation,
    "action_joint_names": list(teacher.actions.joint_names),
    "action_dim": teacher.actions.dim,
    "action_scales": list(teacher.actions.joint_scales),
    "action_offset": teacher.actions.offset,
    "teacher_id": "tiny_000",
    "motion": str(teacher.entry.motion),
    "task": resolved.manifest.base_task,
  }
  _checkpoint(
    anchor_checkpoint,
    schema=anchor_schema,
    model=anchor_model,
    teacher_hashes=teacher.hashes,
    control_contract=control,
  )
  with pytest.raises(ExportValidationError, match="gravity"):
    export_bundle(
      anchor_checkpoint,
      cohort.manifest,
      "tiny_000",
      tmp_path / "out2",
      repo_root=tmp_path,
      asset_audit=_audit(schema.joint_order),
    )


def test_export_refuses_wrong_checkpoint_identity_and_output_collision(
  tmp_path: Path,
) -> None:
  cohort, schema, _, checkpoint = _fixture(tmp_path)
  resolved = resolve_cohort(load_manifest(cohort.manifest, tmp_path))
  teacher = resolved.teacher("tiny_000")
  payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
  payload["teacher_hashes"] = {**teacher.hashes, "motion": "0" * 64}
  wrong = tmp_path / "wrong.pt"
  torch.save(payload, wrong)
  with pytest.raises(ExportValidationError, match="hashes"):
    export_bundle(
      wrong,
      cohort.manifest,
      "tiny_000",
      tmp_path / "wrong-out",
      repo_root=tmp_path,
      asset_audit=_audit(schema.joint_order),
    )
  payload["teacher_hashes"] = teacher.hashes
  payload["control_contract"]["control_hz"] = 49.0
  wrong_control = tmp_path / "wrong-control.pt"
  torch.save(payload, wrong_control)
  with pytest.raises(ExportValidationError, match="control contract"):
    export_bundle(
      wrong_control,
      cohort.manifest,
      "tiny_000",
      tmp_path / "wrong-control-out",
      repo_root=tmp_path,
      asset_audit=_audit(schema.joint_order),
    )
  output = tmp_path / "collision"
  output.mkdir()
  (output / "vae-tiny_000.yaml").write_text("unrelated")
  with pytest.raises(ExportValidationError, match="already exists"):
    export_bundle(
      checkpoint,
      cohort.manifest,
      "tiny_000",
      output,
      repo_root=tmp_path,
      asset_audit=_audit(schema.joint_order),
    )


def test_export_does_not_mutate_frozen_normalizers(tmp_path: Path) -> None:
  cohort, schema, model, checkpoint = _fixture(tmp_path)
  before = {
    name: value.detach().clone()
    for name, value in model.state_dict().items()
    if "normalizer" in name
  }
  result = export_bundle(
    checkpoint,
    cohort.manifest,
    "tiny_000",
    tmp_path / "out",
    repo_root=tmp_path,
    asset_audit=_audit(schema.joint_order),
  )
  validate_export_parity(result, model, torch.randn(1, 68), torch.randn(1, 99))
  after = {
    name: value.detach().clone()
    for name, value in model.state_dict().items()
    if "normalizer" in name
  }
  assert before.keys() == after.keys()
  for name in before:
    assert torch.equal(before[name], after[name]), name


def test_parity_failure_does_not_publish_bundle(tmp_path: Path, monkeypatch) -> None:
  cohort, schema, _, checkpoint = _fixture(tmp_path)
  output = tmp_path / "failed"

  def fail_parity(*args, **kwargs):
    raise ExportValidationError("injected numeric parity failure")

  monkeypatch.setattr(export_module, "validate_export_parity", fail_parity)
  with pytest.raises(ExportValidationError, match="injected numeric parity failure"):
    export_module.export_bundle(
      checkpoint,
      cohort.manifest,
      "tiny_000",
      output,
      repo_root=tmp_path,
      asset_audit=_audit(schema.joint_order),
    )
  assert not (output / "vae-tiny_000.yaml").exists()
  assert not (output / "vae-tiny_000").exists()
