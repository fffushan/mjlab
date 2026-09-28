"""Portable v2/v3 export for the gravity-conditioned distillation VAE.

The exporter is intentionally offline and strict.  It restores the saved model
and schema from a checkpoint, requires an independently audited frame/control
record, and writes two batch-one float32 ONNX graphs plus an immutable motion
JSON.  No simulator, trainer, optimizer, or replay buffer is constructed.

A version-2 bundle is the single-teacher artifact (:func:`export_bundle`).  A
version-3 bundle (:func:`export_cohort_bundle`) exports one saved M4 cohort
checkpoint whose shared student was trained over several teachers: the two
graphs are written once, stamped with the cohort identity, and each member gets
its own ``{contract, motion, parity}`` triple.  Every member is audited with the
same single-teacher audit producer, so a cohort is never certified by one
selected motion standing in for the rest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from mjlab.tasks.tracking.distillation.adapter import validate_live_contract
from mjlab.tasks.tracking.distillation.checkpoint import (
  CheckpointValidationError,
  load_cohort_member_inference,
  load_inference_checkpoint,
)
from mjlab.tasks.tracking.distillation.cohort_contract import (
  require_member_matches,
)
from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  load_manifest,
  resolve_cohort,
  sha256_file,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE
from mjlab.tasks.tracking.distillation.vae_config import DecoderMode

BUNDLE_VERSION = 2
COHORT_BUNDLE_VERSION = 3
BUNDLE_FORMAT = "mjlab-vae-tracking"
EXPORT_OPSET = 18
MAX_COHORT_MEMBERS = 16  # the C++ descriptor parser admits at most 16 members


class ExportValidationError(ValueError):
  """A checkpoint, source contract, or export artifact is unsafe to emit."""


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode()


def _digest(value: Any) -> str:
  return hashlib.sha256(_canonical(value)).hexdigest()


def _finite_float(value: Any, where: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ExportValidationError(f"{where} must be a number")
  result = float(value)
  if not math.isfinite(result):
    raise ExportValidationError(f"{where} must be finite")
  return result


def _finite_vector(value: Any, width: int, where: str) -> list[float]:
  if not isinstance(value, (list, tuple)) or len(value) != width:
    raise ExportValidationError(f"{where} must contain exactly {width} values")
  return [_finite_float(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _load_audit(value: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
  if isinstance(value, (str, os.PathLike)):
    try:
      raw = json.loads(Path(cast(str | os.PathLike[str], value)).read_text())
    except (OSError, json.JSONDecodeError) as exc:
      raise ExportValidationError(f"could not read asset audit {value}: {exc}") from exc
  else:
    raw = dict(value)
  if not isinstance(raw, dict):
    raise ExportValidationError("asset audit must be a JSON object")
  return raw


def _require_audited_provenance(audit: Mapping[str, Any], teacher) -> dict[str, Any]:
  """Validate the serialized form produced by :func:`make_export_audit`."""
  required = {
    "producer",
    "verification",
    "asset_identity",
    "robot_bodies",
    "tracked_bodies",
    "tracked_body_indices",
    "root_frame",
    "anchor_body",
    "gravity",
    "gyro",
    "anchor",
  }
  missing = required - set(audit)
  if missing:
    raise ExportValidationError(
      "asset audit is missing bound physical evidence " + ", ".join(sorted(missing))
    )
  if audit["producer"] != "mjlab.tasks.tracking.distillation.export.make_export_audit":
    raise ExportValidationError("asset audit has no supported producer identity")
  if audit["verification"] != "audited":
    raise ExportValidationError("asset audit verification must be exactly 'audited'")
  identity = audit["asset_identity"]
  if not isinstance(identity, Mapping) or identity.get("sha256") is None:
    raise ExportValidationError("asset audit needs an asset_identity.sha256")
  asset_hash = str(identity["sha256"])
  if not re.fullmatch(r"[0-9a-f]{64}", asset_hash):
    raise ExportValidationError("asset_identity.sha256 must be a SHA-256 digest")
  robot_bodies = audit["robot_bodies"]
  tracked_bodies = audit["tracked_bodies"]
  indices = audit["tracked_body_indices"]
  if not (
    isinstance(robot_bodies, list)
    and isinstance(tracked_bodies, list)
    and isinstance(indices, list)
    and tuple(tracked_bodies) == teacher.body_names
    and len(robot_bodies) == len(set(robot_bodies))
    and len(indices) == len(tracked_bodies)
    and all(
      isinstance(index, int) and 0 <= index < len(robot_bodies) for index in indices
    )
  ):
    raise ExportValidationError("asset audit body mapping is malformed")
  if audit["anchor_body"] != teacher.anchor_body_name:
    raise ExportValidationError(
      f"audited anchor {audit['anchor_body']!r} disagrees with saved "
      f"{teacher.anchor_body_name!r}"
    )
  anchor_position = teacher.body_names.index(teacher.anchor_body_name)
  anchor_index = int(indices[anchor_position])
  if robot_bodies[anchor_index] != teacher.anchor_body_name:
    raise ExportValidationError(
      "asset audit anchor index is not bound to the anchor body"
    )
  root_frame = audit["root_frame"]
  if not isinstance(root_frame, str) or not root_frame:
    raise ExportValidationError("asset audit root_frame must be non-empty")
  if audit["gravity"].get("frame") != root_frame:
    raise ExportValidationError("gravity frame must agree with the audited root frame")
  provenance: dict[str, Any] = {
    "producer": audit["producer"],
    "verification": "audited",
    "asset_identity": dict(identity),
    "robot_bodies": list(robot_bodies),
    "tracked_bodies": list(tracked_bodies),
    "tracked_body_indices": list(indices),
    "anchor_body_index": anchor_index,
    "root_frame": audit["root_frame"],
    "anchor_body": audit["anchor_body"],
  }
  if not isinstance(provenance["root_frame"], str) or not provenance["root_frame"]:
    raise ExportValidationError("asset audit root_frame must be non-empty")
  for name in ("gravity", "gyro", "anchor"):
    item = audit[name]
    if not isinstance(item, Mapping):
      raise ExportValidationError(f"asset audit {name} evidence must be an object")
    if not isinstance(item.get("source"), str) or not isinstance(
      item.get("frame"), str
    ):
      raise ExportValidationError(f"asset audit {name} source/frame must be strings")
    if not item["source"] or not item["frame"]:
      raise ExportValidationError(f"asset audit {name} source/frame must be non-empty")
    provenance[name] = dict(item)
  gyro = audit["gyro"]
  for key in ("sensor_name", "site_name", "body_name"):
    if not isinstance(gyro.get(key), str) or not gyro[key]:
      raise ExportValidationError(f"asset audit gyro {key} must be non-empty")
  if not all(
    key in gyro for key in ("sensor_name", "site_name", "body_name", "local_quat_wxyz")
  ):
    raise ExportValidationError(
      "asset audit gyro lacks compiled sensor transform evidence"
    )
  local_quat = _finite_vector(gyro["local_quat_wxyz"], 4, "gyro.local_quat_wxyz")
  if not np.allclose(local_quat, (1.0, 0.0, 0.0, 0.0), atol=1e-6, rtol=0.0):
    raise ExportValidationError(
      "v1 gravity export requires an identity gyro local frame"
    )
  if gyro["body_name"] != audit["root_frame"]:
    raise ExportValidationError("gyro body is not the audited root frame")
  provenance["gyro"]["local_quat_wxyz"] = local_quat
  return provenance


def _expected_control_contract(cohort, teacher) -> dict[str, Any]:
  return {
    "control_period_s": teacher.control.control_period_s,
    "control_hz": teacher.control.control_hz,
    "sim_timestep": teacher.control.sim_timestep,
    "decimation": teacher.control.decimation,
    "action_joint_names": list(teacher.actions.joint_names),
    "action_dim": teacher.actions.dim,
    "action_scales": list(teacher.actions.joint_scales),
    "action_offset": teacher.actions.offset,
    "teacher_id": teacher.id,
    "motion": str(teacher.entry.motion),
    "task": cohort.manifest.base_task,
  }


def _action_metadata(audit: Mapping[str, Any], teacher) -> dict[str, Any]:
  raw = audit.get("action_metadata")
  if not isinstance(raw, Mapping):
    raise ExportValidationError(
      "asset audit must provide full-precision action_metadata; ONNX text metadata "
      "is not accepted as exact gain/default evidence"
    )
  names = raw.get("joint_names")
  if tuple(names or ()) != teacher.actions.joint_names:
    raise ExportValidationError(
      "audited action joint order disagrees with saved teacher"
    )
  result = {
    "joint_names": list(teacher.actions.joint_names),
    "default_joint_position": _finite_vector(
      raw.get("default_joint_position"),
      teacher.actions.dim,
      "action_metadata.default_joint_position",
    ),
    "kp": _finite_vector(raw.get("kp"), teacher.actions.dim, "action_metadata.kp"),
    "kd": _finite_vector(raw.get("kd"), teacher.actions.dim, "action_metadata.kd"),
  }
  for field, metadata_key in (
    ("default_joint_position", "default_joint_pos"),
    ("kp", "joint_stiffness"),
    ("kd", "joint_damping"),
  ):
    text = teacher.onnx.metadata.get(metadata_key)
    if text is None:
      raise ExportValidationError(
        f"teacher export has no rounded cross-check {metadata_key!r}"
      )
    try:
      rounded = [float(item) for item in text.split(",")]
    except ValueError as exc:
      raise ExportValidationError(
        f"teacher export metadata {metadata_key!r} is malformed"
      ) from exc
    if len(rounded) != teacher.actions.dim or not np.allclose(
      rounded, result[field], atol=1e-3, rtol=0.0
    ):
      raise ExportValidationError(
        f"audited action metadata {field!r} disagrees with teacher export cross-check"
      )
  return result


def make_export_audit(
  env: Any,
  cohort: CohortContract,
  teacher_id: str,
  *,
  asset_path: str | os.PathLike[str],
  task_id: str | None = None,
) -> dict[str, Any]:
  """Produce the only accepted audit JSON from the compiled tracking asset.

  ``validate_live_contract`` and ``audit_live_asset`` perform the existing
  saved/live checks first.  The caller supplies the hash-matched asset release
  file; the exporter never treats a handwritten hash or frame name as evidence.
  """
  asset_file = Path(asset_path)
  if not asset_file.is_file() or asset_file.is_symlink():
    raise ExportValidationError(f"asset_path is not a regular file: {asset_file}")
  live = validate_live_contract(env, cohort, teacher_id, task_id=task_id)
  teacher = cohort.teacher(teacher_id)
  action_term = env.action_manager.get_term(teacher.actions.term)
  robot = env.scene[action_term.cfg.entity_name]
  model = env.sim.mj_model
  asset = live.asset
  try:
    gyro = next(item for item in asset.sensor_evidence if item.expected_type == "gyro")
  except StopIteration as exc:
    raise ExportValidationError("compiled audit has no verified gyro sensor") from exc
  kp: list[float] = []
  kd: list[float] = []
  for joint_name in teacher.actions.joint_names:
    actuator_id = int(model.actuator(f"robot/{joint_name}").id)
    kp.append(float(model.actuator_gainprm[actuator_id, 0]))
    kd.append(float(-model.actuator_biasprm[actuator_id, 2]))
  defaults = robot.data.default_joint_pos[0, action_term.target_ids].detach().cpu()
  robot_bodies = list(asset.robot_bodies)
  evidence_identity = {
    "asset_sha256": sha256_file(asset_file),
    "robot_bodies": robot_bodies,
    "tracked_bodies": list(asset.tracked_bodies),
    "tracked_body_indices": list(asset.tracked_body_indices),
    "anchor_body_id": asset.anchor_body_id,
    "gyro": {
      "name": gyro.name,
      "site_name": gyro.site_name,
      "body_name": gyro.body_name,
      "local_quat_wxyz": list(gyro.local_quat_wxyz),
    },
    "kp": kp,
    "kd": kd,
    "default_joint_position": [float(item) for item in defaults],
  }
  return {
    "producer": "mjlab.tasks.tracking.distillation.export.make_export_audit",
    "verification": "audited",
    "asset_identity": {
      "path": asset_file.name,
      "sha256": sha256_file(asset_file),
      "compiled_evidence_sha256": _digest(evidence_identity),
    },
    "robot_bodies": robot_bodies,
    "tracked_bodies": list(asset.tracked_bodies),
    "tracked_body_indices": list(asset.tracked_body_indices),
    "root_frame": asset.root_frame,
    "anchor_body": asset.anchor_body,
    "gravity": {"source": "gravity_vec_w", "frame": asset.root_frame},
    "gyro": {
      "source": gyro.name,
      "frame": gyro.body_name,
      "sensor_name": gyro.name,
      "site_name": gyro.site_name,
      "body_name": gyro.body_name,
      "local_quat_wxyz": list(gyro.local_quat_wxyz),
    },
    "anchor": {"source": "compiled_body_orientation", "frame": asset.anchor_body},
    "action_metadata": {
      "joint_names": list(teacher.actions.joint_names),
      "default_joint_position": evidence_identity["default_joint_position"],
      "kp": kp,
      "kd": kd,
    },
  }


def _verify_motion_body_binding(teacher, audit: Mapping[str, Any]) -> None:
  indices = audit["tracked_body_indices"]
  body_count = len(audit["robot_bodies"])
  try:
    with np.load(teacher.entry.motion, allow_pickle=False) as data:
      for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if name not in data:
          raise ExportValidationError(f"motion is missing {name!r} for body binding")
        raw = np.asarray(data[name])
        if raw.ndim < 2 or raw.shape[1] != body_count:
          raise ExportValidationError(
            f"motion body binding for {name!r} does not match the full audited asset body count"
          )
        embedded = teacher.onnx.reference_tensor(name)
        if embedded is None:
          raise ExportValidationError(f"teacher export has no embedded {name!r}")
        selected = raw[:, list(indices)]
        expected = np.asarray(embedded[1])
        if selected.shape != expected.shape or not np.allclose(
          selected, expected, atol=1e-5, rtol=1e-5
        ):
          raise ExportValidationError(
            f"motion body binding for {name!r} disagrees with the audited tracked indices"
          )
  except OSError as exc:
    raise ExportValidationError(f"could not verify motion body binding: {exc}") from exc


def _extract_motion(
  teacher, *, joint_order: tuple[str, ...], anchor_index: int
) -> tuple[dict[str, Any], str]:
  try:
    with np.load(teacher.entry.motion, allow_pickle=False) as data:
      names = set(data.files)
      required = {"joint_pos", "joint_vel", "body_quat_w", "fps"}
      if not required <= names:
        raise ExportValidationError(
          f"motion is missing arrays {sorted(required - names)}"
        )
      q = np.asarray(data["joint_pos"])
      dq = np.asarray(data["joint_vel"])
      body_quat = np.asarray(data["body_quat_w"])
      fps = np.asarray(data["fps"]).reshape(-1)
      if (
        q.dtype != np.float32 or dq.dtype != np.float32 or body_quat.dtype != np.float32
      ):
        raise ExportValidationError("motion q/dq/body_quat_w must be float32")
      if (
        q.ndim != 2
        or q.shape[0] <= 0
        or dq.shape != q.shape
        or q.shape[1] != len(joint_order)
      ):
        raise ExportValidationError(
          "motion joint_pos/joint_vel have an invalid or empty shape"
        )
      if (
        body_quat.ndim != 3
        or body_quat.shape[0] != q.shape[0]
        or body_quat.shape[2] != 4
      ):
        raise ExportValidationError("motion body_quat_w must be [frames,bodies,4]")
      if (
        fps.size != 1
        or not math.isfinite(float(fps[0]))
        or float(fps[0]) <= 0
        or not math.isclose(float(fps[0]), 50.0, rel_tol=0.0, abs_tol=1e-6)
      ):
        raise ExportValidationError("motion fps must be exactly 50 Hz")
      if anchor_index < 0 or anchor_index >= body_quat.shape[1]:
        raise ExportValidationError("audited anchor index exceeds motion body rows")
      anchor_quat = body_quat[:, anchor_index, :]
      if (
        not np.isfinite(q).all()
        or not np.isfinite(dq).all()
        or not np.isfinite(anchor_quat).all()
      ):
        raise ExportValidationError("motion arrays contain non-finite values")
      norms = np.linalg.norm(anchor_quat.astype(np.float64), axis=1)
      if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise ExportValidationError("motion anchor quaternions must be unit wxyz")
      payload = {
        "format": "mjlab-vae-motion",
        "version": 2,
        "frames": int(q.shape[0]),
        "fps": float(np.float32(fps[0])),
        "joint_order": list(joint_order),
        "q": q.tolist(),
        "dq": dq.tolist(),
        "anchor_quat_wxyz": anchor_quat.tolist(),
      }
  except OSError as exc:
    raise ExportValidationError(
      f"could not read motion {teacher.entry.motion}: {exc}"
    ) from exc
  return payload, sha256_file(teacher.entry.motion)


def _frozen_normalize(value: torch.Tensor, normalizer) -> torch.Tensor:
  """Tensor-only equivalent of StudentNormalizer.normalize for ONNX tracing."""
  variance = normalizer.m2 / normalizer.count.clamp_min(1.0)
  scale = torch.where(
    normalizer.count > 0,
    torch.sqrt(variance + normalizer._eps),
    torch.ones_like(variance),
  )
  return (value - normalizer.mean) / scale


class _EncoderGraph(nn.Module):
  def __init__(self, model: ConditionalVAE) -> None:
    super().__init__()
    self.reference_normalizer = model.reference_normalizer
    self.encoder = model.encoder
    self.mu_head = model.mu_head

  def forward(self, reference: torch.Tensor) -> torch.Tensor:
    normalized = _frozen_normalize(reference, self.reference_normalizer)
    return self.mu_head(self.encoder(normalized))


class _DecoderGraph(nn.Module):
  def __init__(self, model: ConditionalVAE) -> None:
    super().__init__()
    self.conditioning_normalizer = model.conditioning_normalizer
    self.decoder = model.decoder
    self.action_head = model.action_head

  def forward(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
    normalized = _frozen_normalize(conditioning, self.conditioning_normalizer)
    return self.action_head(self.decoder(torch.cat((latent, normalized), dim=1)))


def _normalizer_metadata(normalizer) -> dict[str, Any]:
  if not normalizer.frozen:
    normalizer.freeze()
  count = normalizer.count.detach().cpu().to(torch.float32)
  mean = normalizer.mean.detach().cpu().to(torch.float32)
  m2 = normalizer.m2.detach().cpu().to(torch.float32)
  return {
    "dimension": int(normalizer.dimension),
    "eps": _finite_float(normalizer.eps, "normalizer.eps"),
    "count": _finite_float(count.item(), "normalizer.count"),
    "mean": mean.tolist(),
    "m2": m2.tolist(),
    "variance_convention": "population_m2_div_count",
    "cold_start_identity": True,
    "frozen": True,
  }


def _write_graphs(model: ConditionalVAE, directory: Path) -> tuple[Path, Path]:
  model.eval()
  encoder_path = directory / "encoder.onnx"
  decoder_path = directory / "decoder.onnx"
  with torch.no_grad():
    torch.onnx.export(
      _EncoderGraph(model).eval(),
      (torch.zeros(1, model.schema.reference_dim),),
      encoder_path,
      input_names=["reference"],
      output_names=["latent"],
      opset_version=EXPORT_OPSET,
      dynamo=False,
      external_data=False,
    )
    torch.onnx.export(
      _DecoderGraph(model).eval(),
      (
        torch.zeros(1, model.schema.latent_dim),
        torch.zeros(1, model.schema.conditioning_dim),
      ),
      decoder_path,
      input_names=["latent", "conditioning"],
      output_names=["actions"],
      opset_version=EXPORT_OPSET,
      dynamo=False,
      external_data=False,
    )
  try:
    import onnx

    for path, graph, outputs in (
      (encoder_path, "encoder", ["latent"]),
      (decoder_path, "decoder", ["actions"]),
    ):
      proto = onnx.load(path)
      onnx.checker.check_model(proto)
      if proto.ByteSize() != path.stat().st_size:
        raise ExportValidationError(f"{graph} ONNX has unexpected external data")
      if [item.name for item in proto.graph.output] != outputs:
        raise ExportValidationError(f"{graph} ONNX output names are malformed")
      onnx.save(proto, path)
  except ImportError as exc:
    raise ExportValidationError(
      "ONNX export validation requires the onnx package"
    ) from exc
  return encoder_path, decoder_path


def _graph_metadata(
  path: Path,
  *,
  model_id: str,
  contract_id: str,
  graph: str,
  bundle_version: int = BUNDLE_VERSION,
) -> None:
  import onnx

  proto = onnx.load(path)
  for item in proto.metadata_props:
    if item.key in {
      "mjlab_model_id",
      "mjlab_contract_id",
      "mjlab_graph",
      "mjlab_bundle_version",
    }:
      proto.metadata_props.remove(item)
  for key, value in (
    ("mjlab_bundle_version", str(bundle_version)),
    ("mjlab_model_id", model_id),
    ("mjlab_contract_id", contract_id),
    ("mjlab_graph", graph),
  ):
    entry = proto.metadata_props.add()
    entry.key, entry.value = key, value
  onnx.save(proto, path)


@dataclass(frozen=True, slots=True)
class ExportResult:
  """Paths and identities emitted by :func:`export_bundle`."""

  descriptor: Path
  contract: Path
  motion: Path
  encoder: Path
  decoder: Path
  model_id: str
  contract_id: str
  report: dict[str, Any]


def export_bundle(
  checkpoint: str | os.PathLike[str],
  manifest: str | os.PathLike[str] | CohortContract,
  teacher_id: str,
  output_dir: str | os.PathLike[str],
  *,
  repo_root: str | os.PathLike[str] | None = None,
  asset_audit: Mapping[str, Any] | str | os.PathLike[str],
) -> ExportResult:
  """Export one audited gravity VAE into a self-contained v2 bundle."""
  if isinstance(manifest, CohortContract):
    cohort = manifest
  else:
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    cohort = resolve_cohort(load_manifest(Path(manifest), root))
  teacher = cohort.teacher(teacher_id)
  if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", teacher_id):
    raise ExportValidationError("teacher_id is not a safe policy short name")
  if (
    not math.isclose(teacher.control.control_hz, 50.0, rel_tol=0.0, abs_tol=1e-6)
    or not math.isclose(teacher.reference.fps, 50.0, rel_tol=0.0, abs_tol=1e-6)
    or not math.isclose(
      teacher.control.control_period_s, 0.02, rel_tol=0.0, abs_tol=1e-9
    )
  ):
    raise ExportValidationError(
      "v1 export requires exactly 50 Hz control and reference cadence"
    )
  audit = _load_audit(asset_audit)
  provenance = _require_audited_provenance(audit, teacher)
  action_metadata = _action_metadata(audit, teacher)
  expected_hashes = dict(teacher.hashes)
  expected_contract = _expected_control_contract(cohort, teacher)
  try:
    inference = load_inference_checkpoint(
      checkpoint,
      device="cpu",
      expected_teacher_hashes=expected_hashes,
      expected_control_contract=expected_contract,
    )
  except CheckpointValidationError as exc:
    raise ExportValidationError(f"checkpoint refused: {exc}") from exc
  if inference.schema.mode is not DecoderMode.GRAVITY:
    raise ExportValidationError(
      "v1 export supports only the saved gravity decoder schema"
    )
  if tuple(inference.schema.joint_order) != teacher.actions.joint_names:
    raise ExportValidationError("checkpoint schema joint order disagrees with teacher")
  model = inference.model
  _verify_motion_body_binding(teacher, audit)
  motion_payload, motion_source_hash = _extract_motion(
    teacher,
    joint_order=tuple(inference.schema.joint_order),
    anchor_index=int(provenance["anchor_body_index"]),
  )
  model_id = _digest(
    {
      "checkpoint_sha256": sha256_file(Path(checkpoint)),
      "teacher_id": teacher_id,
      "schema": inference.schema.compatibility_metadata(),
      "settings": inference.settings.to_metadata(),
      "normalizers": {
        "reference": _normalizer_metadata(model.reference_normalizer),
        "conditioning": _normalizer_metadata(model.conditioning_normalizer),
      },
    }
  )
  directory = Path(output_dir)
  directory.mkdir(parents=True, exist_ok=True)
  final = directory / f"vae-{teacher_id}"
  descriptor_root = directory / f"vae-{teacher_id}.yaml"
  if (
    final.exists()
    or final.is_symlink()
    or descriptor_root.exists()
    or descriptor_root.is_symlink()
  ):
    raise ExportValidationError(
      f"output bundle or descriptor already exists: {final} / {descriptor_root}"
    )
  with tempfile.TemporaryDirectory(prefix="vae-export-", dir=directory) as temp:
    staging = Path(temp)
    encoder, decoder = _write_graphs(model, staging)
    contract_semantics: dict[str, Any] = {
      "format": "mjlab-vae-contract",
      "version": 2,
      "family": "vae_tracking",
      "teacher_id": teacher_id,
      "model_id": model_id,
      "schema": inference.schema.compatibility_metadata(),
      "model_settings": inference.settings.to_metadata(),
      "normalizers": {
        "reference": _normalizer_metadata(model.reference_normalizer),
        "conditioning": _normalizer_metadata(model.conditioning_normalizer),
      },
      "interfaces": {
        "encoder": {"inputs": {"reference": [1, 68]}, "outputs": {"latent": [1, 32]}},
        "decoder": {
          "inputs": {"latent": [1, 32], "conditioning": [1, 99]},
          "outputs": {"actions": [1, 31]},
        },
      },
      "control": {
        "control_period_s": float(teacher.control.control_period_s),
        "control_hz": float(teacher.control.control_hz),
        "reference_fps": float(teacher.reference.fps),
        "endpoint": "hold_final_reference_continue_inference",
      },
      "action": {
        "joint_names": list(teacher.actions.joint_names),
        "scales": [float(x) for x in teacher.actions.joint_scales],
        "offset": float(teacher.actions.offset),
        "normalized_semantics": "raw_normalized_joint_position_action",
        **action_metadata,
      },
      "provenance": {
        "checkpoint_sha256": sha256_file(Path(checkpoint)),
        "teacher_artifacts": dict(teacher.hashes),
        "motion_sha256": motion_source_hash,
        "sensor_anchor": provenance,
        "deployment_measurement_requirement": "pelvis_imu_plus_waist_fk",
        "source_schema": inference.schema.compatibility_metadata(),
      },
    }
    contract_id = _digest(contract_semantics)
    contract = {**contract_semantics, "contract_id": contract_id}
    _graph_metadata(
      encoder, model_id=model_id, contract_id=contract_id, graph="encoder"
    )
    _graph_metadata(
      decoder, model_id=model_id, contract_id=contract_id, graph="decoder"
    )
    (staging / "contract.json").write_text(
      json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (staging / "motion.json").write_text(
      json.dumps(motion_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    file_hashes = {
      name: sha256_file(staging / name)
      for name in ("encoder.onnx", "decoder.onnx", "contract.json", "motion.json")
    }
    descriptor = {
      "format": BUNDLE_FORMAT,
      "version": BUNDLE_VERSION,
      "policy_id": f"vae-{teacher_id}",
      "family": "vae_tracking",
      "model_id": model_id,
      "contract_id": contract_id,
      "files": {
        name: {"path": name, "sha256": digest} for name, digest in file_hashes.items()
      },
    }
    descriptor_path = staging / "bundle.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
    report = {
      "bundle": str(final),
      "policy_id": f"vae-{teacher_id}",
      "model_id": model_id,
      "contract_id": contract_id,
      "files": descriptor["files"],
      "parity": "pending",
    }
    staging_result = ExportResult(
      descriptor=descriptor_path,
      contract=staging / "contract.json",
      motion=staging / "motion.json",
      encoder=encoder,
      decoder=decoder,
      model_id=model_id,
      contract_id=contract_id,
      report=report,
    )
    parity = validate_export_parity(
      staging_result,
      model,
      torch.linspace(-1.0, 1.0, 68, dtype=torch.float32).reshape(1, 68),
      torch.linspace(-1.0, 1.0, 99, dtype=torch.float32).reshape(1, 99),
    )
    report["parity"] = parity
    parity_path = staging / "parity.json"
    parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
    descriptor_files = cast(dict[str, dict[str, str]], descriptor["files"])
    descriptor_files["parity.json"] = {
      "path": "parity.json",
      "sha256": sha256_file(parity_path),
    }
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
    report["files"] = descriptor_files
    os.replace(staging, final)
  descriptor_root = directory / f"vae-{teacher_id}.yaml"
  files: dict[str, dict[str, str]] = cast(
    dict[str, dict[str, str]], descriptor["files"]
  )
  root_descriptor = {
    **descriptor,
    "files": {
      name: {"path": f"vae-{teacher_id}/{item['path']}", "sha256": item["sha256"]}
      for name, item in files.items()
    },
  }
  descriptor_root.write_text(
    json.dumps(root_descriptor, indent=2, sort_keys=True) + "\n"
  )
  result = ExportResult(
    descriptor=descriptor_root,
    contract=final / "contract.json",
    motion=final / "motion.json",
    encoder=final / "encoder.onnx",
    decoder=final / "decoder.onnx",
    model_id=model_id,
    contract_id=contract_id,
    report=report,
  )
  return result


@dataclass(frozen=True, slots=True)
class CohortExportResult:
  """Paths and identities emitted by :func:`export_cohort_bundle`."""

  descriptor: Path
  encoder: Path
  decoder: Path
  members: tuple[Path, ...]
  model_id: str
  contract_id: str
  report: dict[str, Any]


def _require_cohort_audit_mapping(
  asset_audits: Mapping[str, Any] | str | os.PathLike[str],
  teacher_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
  """Accept exactly one audit per member, keyed by teacher id.

  The accepted input is either an inline mapping (teacher id -> audit object) or
  a JSON file holding that mapping, optionally with audit objects or paths to
  audit JSON files.  A missing or extra teacher is refused: a cohort bundle is
  certified by one audited environment per member, never by a selected motion
  standing in for the rest.
  """
  if isinstance(asset_audits, (str, os.PathLike)):
    path = Path(cast(str | os.PathLike[str], asset_audits))
    try:
      raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
      raise ExportValidationError(
        f"could not read the cohort asset audit index {path}: {exc}"
      ) from exc
  else:
    raw = dict(asset_audits)
  if not isinstance(raw, Mapping):
    raise ExportValidationError("the cohort asset audit index must be a mapping")
  expected = set(teacher_ids)
  actual = set(raw)
  if actual != expected:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise ExportValidationError(
      "the cohort asset audit index must name every member exactly; missing "
      f"{missing}, unexpected {extra}"
    )
  audits: dict[str, dict[str, Any]] = {}
  for teacher_id in teacher_ids:
    audits[teacher_id] = _load_audit(raw[teacher_id])
  return audits


def export_cohort_bundle(
  checkpoint: str | os.PathLike[str],
  manifest: str | os.PathLike[str] | CohortContract,
  output_dir: str | os.PathLike[str],
  *,
  repo_root: str | os.PathLike[str] | None = None,
  asset_audits: Mapping[str, Any] | str | os.PathLike[str],
) -> CohortExportResult:
  """Export one saved M4 cohort checkpoint into a version-3 bundle.

  The shared student's graphs are written once and stamped with the cohort
  identity (model id + cohort contract id); every manifest member contributes
  its own ``{contract, motion, parity}`` triple in manifest order, which is the
  order the deployed controller plays them in.  Every member must be covered
  by exactly one audit produced by :func:`make_export_audit` against a pinned
  single-teacher environment of that member.
  """
  if isinstance(manifest, CohortContract):
    cohort = manifest
  else:
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    cohort = resolve_cohort(load_manifest(Path(manifest), root))
  teacher_ids = cohort.teacher_ids
  if not teacher_ids:
    raise ExportValidationError("the manifest selects no teacher")
  if len(teacher_ids) > MAX_COHORT_MEMBERS:
    raise ExportValidationError(
      f"a version-3 bundle admits at most {MAX_COHORT_MEMBERS} members; the "
      f"manifest has {len(teacher_ids)}"
    )
  for teacher_id in teacher_ids:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", teacher_id):
      raise ExportValidationError(
        f"teacher_id {teacher_id!r} is not a safe bundle member name"
      )
  audits = _require_cohort_audit_mapping(asset_audits, teacher_ids)
  try:
    member_zero = load_cohort_member_inference(Path(checkpoint), cohort, teacher_ids[0])
  except CheckpointValidationError as exc:
    raise ExportValidationError(f"cohort checkpoint refused: {exc}") from exc
  stored = member_zero.cohort
  for teacher_id in teacher_ids[1:]:
    try:
      require_member_matches(stored, cohort, teacher_id)
    except Exception as exc:  # CohortContractError; re-raised as export refusal
      raise ExportValidationError(
        f"cohort member {teacher_id!r} refused: {exc}"
      ) from exc
  if tuple(stored.teacher_ids) != teacher_ids:
    raise ExportValidationError(
      "the stored cohort member order disagrees with the manifest order"
    )
  if member_zero.schema.mode is not DecoderMode.GRAVITY:
    raise ExportValidationError(
      "version-3 export supports only the saved gravity decoder schema"
    )
  model = member_zero.model
  schema = member_zero.schema
  if tuple(schema.joint_order) != cohort.actions.joint_names:
    raise ExportValidationError(
      "checkpoint schema joint order disagrees with the cohort contract"
    )
  checkpoint_sha = sha256_file(Path(checkpoint))

  # Per-member validation, identical in strength to the v2 single-teacher export.
  members: list[dict[str, Any]] = []
  for teacher_id in teacher_ids:
    teacher = cohort.teacher(teacher_id)
    if (
      not math.isclose(teacher.control.control_hz, 50.0, rel_tol=0.0, abs_tol=1e-6)
      or not math.isclose(teacher.reference.fps, 50.0, rel_tol=0.0, abs_tol=1e-6)
      or not math.isclose(
        teacher.control.control_period_s, 0.02, rel_tol=0.0, abs_tol=1e-9
      )
    ):
      raise ExportValidationError(
        f"version-3 export requires exactly 50 Hz control and reference cadence "
        f"(teacher {teacher_id!r})"
      )
    provenance = _require_audited_provenance(audits[teacher_id], teacher)
    action_metadata = _action_metadata(audits[teacher_id], teacher)
    _verify_motion_body_binding(teacher, audits[teacher_id])
    motion_payload, motion_source_hash = _extract_motion(
      teacher,
      joint_order=tuple(schema.joint_order),
      anchor_index=int(provenance["anchor_body_index"]),
    )
    members.append(
      {
        "teacher_id": teacher_id,
        "provenance": provenance,
        "action_metadata": action_metadata,
        "motion_payload": motion_payload,
        "motion_source_hash": motion_source_hash,
        "teacher": teacher,
      }
    )

  # The gains are cohort-common in the training contract, and every member's
  # audit read them from the same compiled robot; a disagreement means the
  # audits describe different physical assets and the shared action block
  # would be a lie for at least one member.  ``asset_identity.path`` is only
  # the audit snapshot's own file name and is excluded; the content digests
  # (asset sha256 and compiled evidence digest) must agree.
  def _physical_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    identity = {
      key: value for key, value in provenance["asset_identity"].items() if key != "path"
    }
    return {**provenance, "asset_identity": identity}

  first_action = members[0]["action_metadata"]
  first_provenance = _physical_provenance(members[0]["provenance"])
  for member in members[1:]:
    if member["action_metadata"] != first_action:
      raise ExportValidationError(
        f"audited action metadata disagrees between teachers "
        f"{members[0]['teacher_id']!r} and {member['teacher_id']!r}"
      )
    if _physical_provenance(member["provenance"]) != first_provenance:
      raise ExportValidationError(
        f"audited sensor/anchor provenance disagrees between teachers "
        f"{members[0]['teacher_id']!r} and {member['teacher_id']!r}"
      )

  model_id = _digest(
    {
      "checkpoint_sha256": checkpoint_sha,
      "cohort": {
        "manifest_name": cohort.manifest.name,
        "manifest_sha256": cohort.manifest.sha256,
        "teacher_ids": list(teacher_ids),
      },
      "schema": schema.compatibility_metadata(),
      "settings": member_zero.settings.to_metadata(),
      "normalizers": {
        "reference": _normalizer_metadata(model.reference_normalizer),
        "conditioning": _normalizer_metadata(model.conditioning_normalizer),
      },
    }
  )

  directory = Path(output_dir)
  directory.mkdir(parents=True, exist_ok=True)
  bundle_name = f"vae-{cohort.manifest.name}"
  final = directory / bundle_name
  descriptor_root = directory / f"{bundle_name}.yaml"
  if (
    final.exists()
    or final.is_symlink()
    or descriptor_root.exists()
    or descriptor_root.is_symlink()
  ):
    raise ExportValidationError(
      f"output bundle or descriptor already exists: {final} / {descriptor_root}"
    )
  with tempfile.TemporaryDirectory(prefix="vae-cohort-export-", dir=directory) as temp:
    staging = Path(temp)
    shared = staging / "shared"
    shared.mkdir()
    encoder, decoder = _write_graphs(model, shared)

    common_semantics: dict[str, Any] = {
      "schema": schema.compatibility_metadata(),
      "model_settings": member_zero.settings.to_metadata(),
      "normalizers": {
        "reference": _normalizer_metadata(model.reference_normalizer),
        "conditioning": _normalizer_metadata(model.conditioning_normalizer),
      },
      "interfaces": {
        "encoder": {"inputs": {"reference": [1, 68]}, "outputs": {"latent": [1, 32]}},
        "decoder": {
          "inputs": {"latent": [1, 32], "conditioning": [1, 99]},
          "outputs": {"actions": [1, 31]},
        },
      },
      "control": {
        "control_period_s": float(cohort.control.control_period_s),
        "control_hz": float(cohort.control.control_hz),
        "reference_fps": float(cohort.fps),
        "endpoint": "hold_final_reference_continue_inference",
      },
      "action": {
        "joint_names": list(cohort.actions.joint_names),
        "scales": [float(x) for x in cohort.actions.joint_scales],
        "offset": float(cohort.actions.offset),
        "normalized_semantics": "raw_normalized_joint_position_action",
        **members[0]["action_metadata"],
      },
    }
    if tuple(schema.joint_order) != tuple(cohort.actions.joint_names):
      raise ExportValidationError(
        "checkpoint schema joint order disagrees with the cohort contract"
      )

    # Per-member contracts: byte-shape v2 (the C++ deployment parser is
    # unchanged), each carrying the shared student model id and its own
    # contract id.
    member_contract_ids: list[str] = []
    for member in members:
      teacher = member["teacher"]
      teacher_id = member["teacher_id"]
      semantics = {
        "format": "mjlab-vae-contract",
        "version": 2,
        "family": "vae_tracking",
        "teacher_id": teacher_id,
        "model_id": model_id,
        **common_semantics,
        "provenance": {
          "checkpoint_sha256": checkpoint_sha,
          "teacher_artifacts": dict(teacher.hashes),
          "motion_sha256": member["motion_source_hash"],
          "sensor_anchor": member["provenance"],
          "deployment_measurement_requirement": "pelvis_imu_plus_waist_fk",
          "source_schema": schema.compatibility_metadata(),
        },
      }
      contract_id = _digest(semantics)
      member_contract_ids.append(contract_id)
      member_dir = staging / teacher_id
      member_dir.mkdir()
      contract = {**semantics, "contract_id": contract_id}
      (member_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
      )
      (member_dir / "motion.json").write_text(
        json.dumps(member["motion_payload"], indent=2, sort_keys=True, allow_nan=False)
        + "\n"
      )

    # The cohort identity binds the shared graphs to the exact ordered member
    # list; the graphs carry this contract id (and only this one).
    cohort_contract_id = _digest(
      {
        "format": "mjlab-vae-contract",
        "version": 3,
        "family": "vae_tracking",
        "model_id": model_id,
        "cohort": {
          "name": cohort.manifest.name,
          "manifest_sha256": cohort.manifest.sha256,
          "teacher_ids": list(teacher_ids),
        },
        "members": [
          {"teacher_id": member["teacher_id"], "contract_id": contract_id}
          for member, contract_id in zip(members, member_contract_ids, strict=True)
        ],
        **common_semantics,
      }
    )
    _graph_metadata(
      encoder,
      model_id=model_id,
      contract_id=cohort_contract_id,
      graph="encoder",
      bundle_version=COHORT_BUNDLE_VERSION,
    )
    _graph_metadata(
      decoder,
      model_id=model_id,
      contract_id=cohort_contract_id,
      graph="decoder",
      bundle_version=COHORT_BUNDLE_VERSION,
    )

    # Bundle-relative paths: the contained bundle.json copy keeps the same
    # convention as the v2 bundle (paths relative to the bundle directory).
    descriptor_files = {
      "encoder.onnx": {
        "path": "shared/encoder.onnx",
        "sha256": sha256_file(encoder),
      },
      "decoder.onnx": {
        "path": "shared/decoder.onnx",
        "sha256": sha256_file(decoder),
      },
    }
    descriptor_members: list[dict[str, Any]] = []
    member_parity: list[dict[str, Any]] = []
    for member, contract_id in zip(members, member_contract_ids, strict=True):
      teacher_id = member["teacher_id"]
      parity = validate_export_parity(
        ExportResult(
          descriptor=staging / "bundle.json",
          contract=staging / teacher_id / "contract.json",
          motion=staging / teacher_id / "motion.json",
          encoder=encoder,
          decoder=decoder,
          model_id=model_id,
          contract_id=contract_id,
          report={},
        ),
        model,
        torch.linspace(-1.0, 1.0, 68, dtype=torch.float32).reshape(1, 68),
        torch.linspace(-1.0, 1.0, 99, dtype=torch.float32).reshape(1, 99),
      )
      parity_path = staging / teacher_id / "parity.json"
      parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
      member_parity.append(
        {
          "teacher_id": teacher_id,
          "contract_id": contract_id,
          "latent_max_abs_error": parity["latent_max_abs_error"],
          "action_max_abs_error": parity["action_max_abs_error"],
          "atol": 1e-5,
          "rtol": 1e-5,
        }
      )
      descriptor_members.append(
        {
          "teacher_id": teacher_id,
          "contract_id": contract_id,
          "files": {
            "contract.json": {
              "path": f"{teacher_id}/contract.json",
              "sha256": sha256_file(staging / teacher_id / "contract.json"),
            },
            "motion.json": {
              "path": f"{teacher_id}/motion.json",
              "sha256": sha256_file(staging / teacher_id / "motion.json"),
            },
            "parity.json": {
              "path": f"{teacher_id}/parity.json",
              "sha256": sha256_file(parity_path),
            },
          },
        }
      )
    descriptor = {
      "format": BUNDLE_FORMAT,
      "version": COHORT_BUNDLE_VERSION,
      "policy_id": bundle_name,
      "family": "vae_tracking",
      "model_id": model_id,
      "contract_id": cohort_contract_id,
      "files": descriptor_files,
      "members": descriptor_members,
    }
    descriptor_path = staging / "bundle.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")

    def _prefixed(entry: dict[str, str]) -> dict[str, str]:
      return {**entry, "path": f"{bundle_name}/{entry['path']}"}

    report = {
      "bundle": str(final),
      "policy_id": bundle_name,
      "model_id": model_id,
      "contract_id": cohort_contract_id,
      "teacher_ids": list(teacher_ids),
      "member_contract_ids": list(member_contract_ids),
      "files": {name: _prefixed(item) for name, item in descriptor_files.items()},
      "members": [
        {
          "teacher_id": member["teacher_id"],
          **{name: _prefixed(item) for name, item in entry["files"].items()},
        }
        for member, entry in zip(members, descriptor_members, strict=True)
      ],
      "parity": {
        "verified": True,
        "atol": 1e-5,
        "rtol": 1e-5,
        "members": member_parity,
      },
    }
    os.replace(staging, final)

  # Publish the store-level descriptor: written to a temporary sibling and
  # atomically renamed, so no partial descriptor is ever visible at the store
  # level. If that publication fails, the just-published bundle directory is
  # removed again: a retry can start from a clean output directory instead of
  # being refused by half-published remains.
  root_descriptor = {
    **descriptor,
    "files": {name: _prefixed(item) for name, item in descriptor_files.items()},
    "members": [
      {
        **member,
        "files": {name: _prefixed(item) for name, item in member["files"].items()},
      }
      for member in descriptor_members
    ],
  }
  descriptor_temp = directory / f".{bundle_name}.descriptor.tmp"
  try:
    descriptor_temp.write_text(
      json.dumps(root_descriptor, indent=2, sort_keys=True) + "\n"
    )
    os.replace(descriptor_temp, descriptor_root)
  except BaseException:
    descriptor_temp.unlink(missing_ok=True)
    shutil.rmtree(final, ignore_errors=True)
    raise
  return CohortExportResult(
    descriptor=descriptor_root,
    encoder=final / "shared" / "encoder.onnx",
    decoder=final / "shared" / "decoder.onnx",
    members=tuple(final / member["teacher_id"] for member in members),
    model_id=model_id,
    contract_id=cohort_contract_id,
    report=report,
  )


def validate_export_parity(
  result: ExportResult,
  model: ConditionalVAE,
  reference: torch.Tensor,
  conditioning: torch.Tensor,
  *,
  atol: float = 1e-5,
  rtol: float = 1e-5,
) -> dict[str, Any]:
  """Compare saved PyTorch mean inference with both emitted ORT graphs."""
  try:
    import onnxruntime as ort
  except ImportError as exc:
    raise ExportValidationError("ORT parity requires onnxruntime") from exc
  if reference.dtype is not torch.float32 or conditioning.dtype is not torch.float32:
    raise ExportValidationError("parity inputs must be float32")
  if reference.ndim != 2 or reference.shape[1] != model.schema.reference_dim:
    raise ExportValidationError("parity reference has the wrong shape")
  if conditioning.ndim != 2 or conditioning.shape[1] != model.schema.conditioning_dim:
    raise ExportValidationError("parity conditioning has the wrong shape")
  with torch.no_grad():
    expected_mu, _ = model.encode(reference)
    expected_action = model.decode(expected_mu, conditioning)
  encoder_session = ort.InferenceSession(
    result.encoder.read_bytes(), providers=["CPUExecutionProvider"]
  )
  decoder_session = ort.InferenceSession(
    result.decoder.read_bytes(), providers=["CPUExecutionProvider"]
  )
  actual_mu = encoder_session.run(None, {"reference": reference.numpy()})[0]
  actual_action = decoder_session.run(
    None, {"latent": actual_mu, "conditioning": conditioning.numpy()}
  )[0]
  expected_mu_np = expected_mu.numpy()
  expected_action_np = expected_action.numpy()
  mu_error = float(np.max(np.abs(actual_mu - expected_mu_np)))
  action_error = float(np.max(np.abs(actual_action - expected_action_np)))
  if not np.allclose(actual_mu, expected_mu_np, atol=atol, rtol=rtol):
    raise ExportValidationError(f"encoder parity failed: max_abs_error={mu_error}")
  if not np.allclose(actual_action, expected_action_np, atol=atol, rtol=rtol):
    raise ExportValidationError(f"decoder parity failed: max_abs_error={action_error}")
  return {
    "latent_max_abs_error": mu_error,
    "action_max_abs_error": action_error,
    "fixture": {
      "model_id": result.model_id,
      "contract_id": result.contract_id,
      "dtype": "float32",
      "reference": reference.detach().cpu().tolist(),
      "conditioning": conditioning.detach().cpu().tolist(),
      "expected_latent": expected_mu_np.tolist(),
      "expected_action": expected_action_np.tolist(),
    },
  }


__all__ = [
  "BUNDLE_FORMAT",
  "BUNDLE_VERSION",
  "COHORT_BUNDLE_VERSION",
  "MAX_COHORT_MEMBERS",
  "CohortExportResult",
  "ExportResult",
  "ExportValidationError",
  "export_bundle",
  "export_cohort_bundle",
  "make_export_audit",
  "validate_export_parity",
]
