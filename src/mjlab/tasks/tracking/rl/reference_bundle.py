"""Write X2 reference clips as the ``mjlab-vae-motion`` v2 files the controller reads.

A reference-free tracking policy needs its reference data from somewhere other than the
ONNX graph. The X2 controller already reads one reference format: the ``mjlab-vae-motion``
v2 JSON that the VAE bundles carry. Reusing that format means the reference of a
tracking policy needs no new reader, and it means a clip the policy never trained on is
just another file.

The controller parses the file strictly, so this writer mirrors that reader exactly: a
fixed key set, ``version 2``, ``format mjlab-vae-motion``, exactly ``50.0`` frames per
second, one joint row per frame in the policy's joint order, and one unit anchor
quaternion per frame.

Only reference tables live here, never weights. That is the whole point of the split.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

MOTION_FORMAT = "mjlab-vae-motion"
MOTION_FORMAT_VERSION = 2
MANIFEST_FORMAT = "x2-reference-clip-bundle"
MANIFEST_VERSION = 1

REQUIRED_NPZ_KEYS = ("fps", "joint_pos", "joint_vel", "body_quat_w")


class ReferenceBundleError(ValueError):
  """A clip cannot be written in the format the controller accepts."""


@dataclass(frozen=True)
class ReferenceContract:
  """The binding between a clip file and the policy that will replay it.

  ``body_names`` must be the full robot body order, because the anchor column is selected
  from that order. The motion npz stores every body, while the policy observes a subset.
  """

  joint_names: Sequence[str]
  body_names: Sequence[str]
  anchor_body_name: str
  fps: float = 50.0

  def __post_init__(self) -> None:
    if not self.joint_names:
      raise ReferenceBundleError("the joint order is empty")
    if not self.body_names:
      raise ReferenceBundleError("the robot body list is empty")
    if self.anchor_body_name not in self.body_names:
      raise ReferenceBundleError(
        f"anchor body {self.anchor_body_name!r} is not in the robot body list"
      )
    if not math.isclose(self.fps, 50.0, rel_tol=0.0, abs_tol=1e-9):
      raise ReferenceBundleError("the X2 reference rate must be exactly 50 Hz")

  @property
  def anchor_body_index(self) -> int:
    """Column of the anchor quaternion inside the motion frame body table."""
    return list(self.body_names).index(self.anchor_body_name)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def build_clip_payload(npz_path: Path, contract: ReferenceContract) -> dict:
  """Build one ``mjlab-vae-motion`` v2 payload from a motion npz."""
  try:
    with np.load(npz_path, allow_pickle=False) as data:
      missing = [name for name in REQUIRED_NPZ_KEYS if name not in data]
      if missing:
        raise ReferenceBundleError(f"motion is missing {missing}")
      q = np.asarray(data["joint_pos"]).astype(np.float32)
      dq = np.asarray(data["joint_vel"]).astype(np.float32)
      body_quat = np.asarray(data["body_quat_w"]).astype(np.float32)
      fps = np.asarray(data["fps"]).reshape(-1)
  except OSError as exc:
    raise ReferenceBundleError(f"could not read motion {npz_path}: {exc}") from exc

  joint_count = len(contract.joint_names)
  if q.ndim != 2 or q.shape[0] <= 0 or dq.shape != q.shape or q.shape[1] != joint_count:
    raise ReferenceBundleError(
      f"{npz_path.name}: joint_pos/joint_vel must be [frames, {joint_count}]"
    )
  if (
    body_quat.ndim != 3
    or body_quat.shape[0] != q.shape[0]
    or body_quat.shape[2] != 4
    or body_quat.shape[1] != len(contract.body_names)
  ):
    raise ReferenceBundleError(
      f"{npz_path.name}: body_quat_w must be [frames, {len(contract.body_names)}, 4]"
    )
  if fps.size != 1 or not math.isfinite(float(fps[0])):
    raise ReferenceBundleError(f"{npz_path.name}: fps must be one finite value")
  if float(fps[0]) != 50.0:
    raise ReferenceBundleError(
      f"{npz_path.name}: fps must be exactly 50.0, got {float(fps[0])!r}"
    )
  if not np.isfinite(q).all() or not np.isfinite(dq).all():
    raise ReferenceBundleError(f"{npz_path.name}: reference joint rows are not finite")

  anchor_quat = body_quat[:, contract.anchor_body_index, :]
  if not np.isfinite(anchor_quat).all():
    raise ReferenceBundleError(f"{npz_path.name}: anchor quaternions are not finite")
  norms = np.linalg.norm(anchor_quat.astype(np.float64), axis=1)
  if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
    raise ReferenceBundleError(
      f"{npz_path.name}: anchor quaternions must be unit length in wxyz order"
    )

  return {
    "format": MOTION_FORMAT,
    "version": MOTION_FORMAT_VERSION,
    "frames": int(q.shape[0]),
    "fps": float(np.float32(fps[0])),
    "joint_order": list(contract.joint_names),
    "q": q.tolist(),
    "dq": dq.tolist(),
    "anchor_quat_wxyz": anchor_quat.tolist(),
  }


def write_reference_bundle(
  out_dir: Path,
  clips: Mapping[str, Path],
  contract: ReferenceContract,
) -> dict:
  """Write ``motions/<name>.json`` for each clip plus a manifest.

  Returns the manifest. Clip names must be safe file stems so the controller can resolve
  them from a menu token.
  """
  if not clips:
    raise ReferenceBundleError("no clips were given")
  motions = out_dir / "motions"
  motions.mkdir(parents=True, exist_ok=True)

  entries: list[dict] = []
  for name in sorted(clips):
    if not name or Path(name).name != name or name.startswith("."):
      raise ReferenceBundleError(f"clip name {name!r} is not a safe file stem")
    source = Path(clips[name])
    payload = build_clip_payload(source, contract)
    target = motions / f"{name}.json"
    target.write_text(json.dumps(payload) + "\n")
    entries.append(
      {
        "name": name,
        "motion_file": f"motions/{name}.json",
        "source_npz": str(source),
        "source_npz_sha256": sha256_file(source),
        "motion_sha256": sha256_file(target),
        "frames": payload["frames"],
        "fps": payload["fps"],
        "duration_s": round(payload["frames"] / payload["fps"], 6),
      }
    )

  manifest = {
    "format": MANIFEST_FORMAT,
    "version": MANIFEST_VERSION,
    "reference_contract": {
      "fps": 50.0,
      "joint_order": list(contract.joint_names),
      "anchor_body_name": contract.anchor_body_name,
      "anchor_body_index": contract.anchor_body_index,
      "body_names": list(contract.body_names),
    },
    "clips": entries,
  }
  (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def contract_from_motion_command(
  motion_cfg, joint_names: Sequence[str], body_names_full: Sequence[str]
) -> ReferenceContract:
  """Build the contract from a live motion command cfg and the robot body order.

  ``joint_names`` must be the policy's joint order, and ``body_names_full`` the robot body
  order of the motion npz, so the anchor column resolves the way the reference tables do.
  """
  return ReferenceContract(
    joint_names=tuple(joint_names),
    body_names=tuple(body_names_full),
    anchor_body_name=str(motion_cfg.anchor_body_name),
  )


def clip_paths(
  paths: Iterable[Path], *, name_pattern: str | None = None
) -> dict[str, Path]:
  """Map clip names to paths, refusing a duplicate name.

  ``name_pattern`` extracts the clip name from the file name with its first capture
  group, so a bundle can carry short operator-facing ids while the source files keep
  their long training names.
  """
  pattern = re.compile(name_pattern) if name_pattern else None
  out: dict[str, Path] = {}
  for path in paths:
    if pattern is None:
      name = path.stem
    else:
      match = pattern.search(path.name)
      if match is None or match.groups() == ():
        raise ReferenceBundleError(
          f"clip name pattern did not capture a name in {path.name!r}"
        )
      name = match.group(1)
    if name in out:
      raise ReferenceBundleError(f"duplicate clip name {name!r}")
    out[name] = path
  return out
