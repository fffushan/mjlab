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
from mjlab.tasks.tracking.distillation.balanced_storage import BalancedReplayBuffer
from mjlab.tasks.tracking.distillation.checkpoint import (
  CHECKPOINT_VERSION,
  COHORT_CHECKPOINT_VERSION,
  save_cohort_checkpoint,
)
from mjlab.tasks.tracking.distillation.cohort_contract import (
  CohortIdentity,
  build_cohort_identity,
)
from mjlab.tasks.tracking.distillation.config import load_manifest, resolve_cohort
from mjlab.tasks.tracking.distillation.export import (
  ExportValidationError,
  export_bundle,
  export_cohort_bundle,
  make_export_audit,
  validate_export_parity,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.motion_library import BodySelection
from mjlab.tasks.tracking.distillation.multi_motion import plan_multi_motion
from mjlab.tasks.tracking.distillation.observations import (
  ObservationSnapshot,
  pack_observations,
)
from mjlab.tasks.tracking.distillation.storage import LabeledReplayBatch
from mjlab.tasks.tracking.distillation.trainer import (
  FreshTrainingData,
  VaeDistillationTrainer,
)
from mjlab.tasks.tracking.distillation.training_config import TrainingConfig
from mjlab.tasks.tracking.distillation.vae_config import ModelSettings, make_schema


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


# --- version-3 cohort export -------------------------------------------------

COHORT_TEACHER_IDS = ("tiny_000", "tiny_001")
COHORT_FRAMES = 4


def _cohort_audit(joint_names: tuple[str, ...]) -> dict:
  return _audit(joint_names)


def _cohort_fixture(tmp_path: Path):
  """Two tiny teachers, a saved M4 cohort checkpoint, and per-member audits."""
  names = (
    *(f"hip_{i:02d}_hip_joint" for i in range(30)),
    "waist_joint",
  )
  cohort = build_tiny_cohort(
    tmp_path,
    teacher_ids=COHORT_TEACHER_IDS,
    joint_names=names,
    frames=COHORT_FRAMES,
  )
  preliminary = resolve_cohort(load_manifest(cohort.manifest, tmp_path))
  assert preliminary.teacher_ids == COHORT_TEACHER_IDS

  # Every member gets a valid npz (finite, unit anchor quaternions) and an ONNX
  # whose embedded body references agree with the audited tracked subset.
  for teacher_id in COHORT_TEACHER_IDS:
    teacher = preliminary.teacher(teacher_id)
    frames = COHORT_FRAMES
    member_index = COHORT_TEACHER_IDS.index(teacher_id)
    q = (
      (np.arange(frames * 31, dtype=np.float64) / 97.0 + 10.0 * member_index)
      .astype(np.float32)
      .reshape(frames, 31)
    )
    dq = (q + np.float32(0.25)).astype(np.float32)
    anchor = np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (frames, 1))
    with np.load(teacher.entry.motion, allow_pickle=False) as old:
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
      teacher.entry.motion,
      joint_pos=q,
      joint_vel=dq,
      **full_arrays,
      fps=np.array([50.0], dtype=np.float32),
    )
    graph = onnx.load(str(teacher.entry.onnx))
    for name, array in full_arrays.items():
      for initializer in graph.graph.initializer:
        if initializer.name == name:
          graph.graph.initializer.remove(initializer)
      graph.graph.initializer.append(
        numpy_helper.from_array(array[:, [0, 1]], name=name)
      )
    onnx.save(graph, str(teacher.entry.onnx))

  # Resolve again after the rewrite so recorded digests match the final bytes.
  resolved = resolve_cohort(load_manifest(cohort.manifest, tmp_path))
  schema = make_schema(joint_order=names)
  torch.manual_seed(11)
  model = ConditionalVAE(schema, ModelSettings(hidden_dims=(8, 8)))
  model.reference_normalizer.update(torch.randn(8, 68))
  model.conditioning_normalizer.update(torch.randn(8, 99))
  model.reference_normalizer.freeze()
  model.conditioning_normalizer.freeze()

  plan = plan_multi_motion(
    resolved,
    COHORT_TEACHER_IDS,
    2,
    phase_policy="uniform",
    slot_generator=torch.Generator().manual_seed(5),
  )
  replay = BalancedReplayBuffer(
    8,
    schema,
    {clip.motion_id: 1.0 for clip in plan.library.clips},
    teacher_codes={clip.motion_id: clip.teacher_code for clip in plan.library.clips},
    frame_counts={clip.motion_id: clip.frames for clip in plan.library.clips},
  )
  generator = torch.Generator().manual_seed(2)
  size = 2
  motion = torch.arange(size, dtype=torch.int64) % len(plan.library.clips)
  batch = LabeledReplayBatch(
    pack_observations(
      ObservationSnapshot(
        reference_q=torch.randn(size, 31, generator=generator),
        reference_dq=torch.randn(size, 31, generator=generator),
        anchor_orientation_error=torch.normal(0, 1, (size, 3, 3), generator=generator),
        projected_gravity=torch.randn(size, 3, generator=generator),
        gyro=torch.randn(size, 3, generator=generator),
        relative_joint_q=torch.randn(size, 31, generator=generator),
        joint_dq=torch.randn(size, 31, generator=generator),
        previous_action=torch.randn(size, 31, generator=generator),
      ),
      schema,
    ),
    torch.randn(size, schema.action_dim, generator=generator),
    motion,
    torch.tensor(
      [plan.library.clip(int(item)).teacher_code for item in motion.tolist()]
    ),
    torch.zeros(size, dtype=torch.int64),
    torch.zeros(size, dtype=torch.int64),
    torch.zeros(size, dtype=torch.int64),
  )
  trainer = VaeDistillationTrainer(
    model,
    replay,
    TrainingConfig(accumulation_steps=1, minibatch_size=2),
    seed=13,
  )
  replay.insert(batch)
  trainer.begin_training(FreshTrainingData(batch, "initial"))
  trainer.train_update()
  trainer.freeze_normalizers()
  body_selection = BodySelection(
    indices=tuple(range(len(resolved.body_names))),
    source_body_count=plan.library.source_body_count,
    names=tuple(resolved.body_names),
  )
  identity: CohortIdentity = build_cohort_identity(
    resolved,
    plan.library,
    plan.slots,
    phase_policy="uniform",
    replay=replay,
    device="cpu",
    requested_seed=5,
    effective_seed=5,
    body_selection=body_selection,
  )
  checkpoint = tmp_path / "cohort.pt"
  save_cohort_checkpoint(
    checkpoint,
    trainer,
    replay,
    cohort=identity,
    counters={"iteration": 1},
    schedule={"max_iterations": 10},
    resolved_config={"command": "train", "teacher_ids": list(COHORT_TEACHER_IDS)},
  )
  audits = {teacher_id: _cohort_audit(names) for teacher_id in COHORT_TEACHER_IDS}
  return resolved, schema, model, checkpoint, audits


def test_cohort_export_emits_v3_bundle_with_shared_graphs_and_member_triples(
  tmp_path: Path,
) -> None:
  resolved, schema, model, checkpoint, audits = _cohort_fixture(tmp_path)
  out = tmp_path / "out"
  result = export_cohort_bundle(
    checkpoint,
    resolved,
    out,
    repo_root=tmp_path,
    asset_audits=audits,
  )
  descriptor = json.loads(result.descriptor.read_text())
  assert descriptor["version"] == 3
  assert descriptor["format"] == "mjlab-vae-tracking"
  assert descriptor["family"] == "vae_tracking"
  assert descriptor["policy_id"] == "vae-tiny-cohort"
  assert set(descriptor["files"]) == {"encoder.onnx", "decoder.onnx"}
  assert [m["teacher_id"] for m in descriptor["members"]] == list(COHORT_TEACHER_IDS)
  assert len({m["contract_id"] for m in descriptor["members"]}) == 2

  bundle_dir = result.descriptor.parent / "vae-tiny-cohort"
  for _name, item in descriptor["files"].items():
    path = result.descriptor.parent / item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
  for member in descriptor["members"]:
    for _key, item in member["files"].items():
      path = result.descriptor.parent / item["path"]
      assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

  # Shared graphs are stamped with the cohort identity only.
  for graph_name in ("encoder.onnx", "decoder.onnx"):
    proto = onnx.load(bundle_dir / "shared" / graph_name)
    metadata = {item.key: item.value for item in proto.metadata_props}
    assert metadata["mjlab_model_id"] == descriptor["model_id"]
    assert metadata["mjlab_contract_id"] == descriptor["contract_id"]
    assert metadata["mjlab_bundle_version"] == "3"
  assert descriptor["model_id"] == result.model_id
  assert descriptor["contract_id"] == result.contract_id

  # Member contracts: v2 shape, shared student, own contract id and teacher.
  member_contracts = []
  for teacher_id in COHORT_TEACHER_IDS:
    contract = json.loads((bundle_dir / teacher_id / "contract.json").read_text())
    assert contract["version"] == 2
    assert contract["model_id"] == descriptor["model_id"]
    assert contract["teacher_id"] == teacher_id
    member_contracts.append(contract)
    parity = json.loads((bundle_dir / teacher_id / "parity.json").read_text())
    assert parity["fixture"]["model_id"] == descriptor["model_id"]
    assert parity["fixture"]["contract_id"] == contract["contract_id"]
    assert parity["latent_max_abs_error"] <= 1e-5
  declared = {m["teacher_id"]: m["contract_id"] for m in descriptor["members"]}
  for contract in member_contracts:
    assert declared[contract["teacher_id"]] == contract["contract_id"]

  # Motions are distinct and match the source npz frames.
  motions = {}
  for teacher_id in COHORT_TEACHER_IDS:
    motion = json.loads((bundle_dir / teacher_id / "motion.json").read_text())
    motions[teacher_id] = motion
    assert motion["frames"] == COHORT_FRAMES
    assert motion["fps"] == 50.0
    teacher = resolved.teacher(teacher_id)
    with np.load(teacher.entry.motion, allow_pickle=False) as source:
      assert np.array_equal(
        np.asarray(motion["q"], dtype=np.float32), source["joint_pos"]
      )
      assert np.array_equal(
        np.asarray(motion["dq"], dtype=np.float32), source["joint_vel"]
      )
  assert motions[COHORT_TEACHER_IDS[0]]["q"] != motions[COHORT_TEACHER_IDS[1]]["q"]

  # The two members' contract ids must differ (different provenance/motion).
  assert member_contracts[0]["contract_id"] != member_contracts[1]["contract_id"]
  assert result.report["teacher_ids"] == list(COHORT_TEACHER_IDS)
  assert result.report["member_contract_ids"] == [
    declared[teacher_id] for teacher_id in COHORT_TEACHER_IDS
  ]


def test_cohort_export_refuses_missing_or_extra_audits(tmp_path: Path) -> None:
  resolved, _, _, checkpoint, audits = _cohort_fixture(tmp_path)
  missing = {k: v for k, v in audits.items() if k != COHORT_TEACHER_IDS[1]}
  with pytest.raises(ExportValidationError, match="missing"):
    export_cohort_bundle(
      checkpoint,
      resolved,
      tmp_path / "out-missing",
      repo_root=tmp_path,
      asset_audits=missing,
    )
  extra = {**audits, "tiny_999": audits[COHORT_TEACHER_IDS[0]]}
  with pytest.raises(ExportValidationError, match="unexpected"):
    export_cohort_bundle(
      checkpoint,
      resolved,
      tmp_path / "out-extra",
      repo_root=tmp_path,
      asset_audits=extra,
    )
  # Two audits that describe different compiled robots (self-consistent but
  # different asset identity) are refused: the shared action/sensor block would
  # be a lie for at least one member.
  disagree = dict(audits)
  other_identity = {
    **audits[COHORT_TEACHER_IDS[1]],
    "asset_identity": {
      "path": "other.xml",
      "sha256": "1" * 64,
    },
  }
  disagree[COHORT_TEACHER_IDS[1]] = other_identity
  with pytest.raises(ExportValidationError, match="provenance disagrees"):
    export_cohort_bundle(
      checkpoint,
      resolved,
      tmp_path / "out-disagree",
      repo_root=tmp_path,
      asset_audits=disagree,
    )


def test_cohort_export_refuses_v1_checkpoint_and_single_audit_misuse(
  tmp_path: Path,
) -> None:
  resolved, schema, _, checkpoint, audits = _cohort_fixture(tmp_path)
  # A version-1 single-teacher checkpoint through the cohort seam.
  v1 = tmp_path / "v1.pt"
  torch.save(
    {
      "kind": "mjlab-m3-distillation",
      "version": CHECKPOINT_VERSION,
      "schema": schema.compatibility_metadata(),
      "model_settings": {
        "latent_dim": 32,
        "hidden_dims": [8, 8],
        "activation": "ELU",
        "beta": 0.01,
      },
      "model": {},
      "teacher_hashes": {},
      "control_contract": {},
      "counters": {},
      "schedule": {},
      "resolved_config": {},
    },
    v1,
  )
  with pytest.raises(ExportValidationError, match="cohort checkpoint refused"):
    export_cohort_bundle(
      v1,
      resolved,
      tmp_path / "out-v1",
      repo_root=tmp_path,
      asset_audits=audits,
    )
  assert _module_cohort_version(checkpoint) == COHORT_CHECKPOINT_VERSION


def _module_cohort_version(path: Path) -> int:
  payload = torch.load(path, map_location="cpu", weights_only=True)
  return int(payload["version"])


def test_cli_export_dispatches_cohort_checkpoint_to_the_cohort_seam(
  tmp_path: Path, capsys, monkeypatch
) -> None:
  from mjlab.scripts import distill as distill_module

  resolved, _, _, checkpoint, audits = _cohort_fixture(tmp_path)
  index = tmp_path / "audit-index.json"
  index.write_text(
    json.dumps({k: str(tmp_path / f"audit-{k}.json") for k in audits}) + "\n"
  )
  for teacher_id, audit in audits.items():
    (tmp_path / f"audit-{teacher_id}.json").write_text(json.dumps(audit) + "\n")

  calls = {}

  def fake_export(checkpoint_arg, manifest_arg, output_dir, **kwargs):
    calls["checkpoint"] = checkpoint_arg
    calls["audits"] = kwargs.get("asset_audits")
    return SimpleNamespace(
      report={"ok": True},
      descriptor=tmp_path / "out" / "vae-tiny-cohort.yaml",
    )

  monkeypatch.setattr(distill_module, "export_cohort_bundle", fake_export)
  rc = distill_module._export(
    checkpoint,
    resolved.manifest.path,
    output_dir=tmp_path / "out",
    repo_root=tmp_path,
    asset_audits=index,
  )
  assert rc == 0
  assert calls["checkpoint"] == checkpoint
  assert calls["audits"] == index
  out = capsys.readouterr().out
  assert json.loads(out.strip()) == {"ok": True}

  # A cohort checkpoint with the singular audit is refused, not reduced.
  rc = distill_module._export(
    checkpoint,
    resolved.manifest.path,
    output_dir=tmp_path / "out2",
    repo_root=tmp_path,
    asset_audit=tmp_path / "audit-tiny_000.json",
  )
  assert rc == 1
  assert "--asset-audits" in capsys.readouterr().err

  # Neither audit argument is refused.
  rc = distill_module._export(
    checkpoint, resolved.manifest.path, output_dir=tmp_path / "out3"
  )
  assert rc == 1
  assert "--asset-audit" in capsys.readouterr().err
