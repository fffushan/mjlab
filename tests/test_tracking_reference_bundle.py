"""Tests for the X2 reference clip bundle writer.

The writer must produce exactly the file the deployed controller already parses, so these
tests pin the key set, the types and the numeric content, and compare against a real
bundle that the controller accepted.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.tracking.rl.reference_bundle import (
  ReferenceBundleError,
  ReferenceContract,
  build_clip_payload,
  clip_paths,
  write_reference_bundle,
)

# The strict key set of the X2 controller's motion reader
# (vae_tracking_adapter.cc, StrictMap over the motion document).
CONTROLLER_MOTION_KEYS = {
  "anchor_quat_wxyz",
  "dq",
  "format",
  "fps",
  "frames",
  "joint_order",
  "q",
  "version",
}

JOINT_NAMES = ("j0", "j1", "j2")
BODY_NAMES = ("pelvis", "waist_yaw_link", "torso_link")
ANCHOR = "torso_link"

# A real bundle the controller accepted, and the clip it was built from.
X2_TREE = Path("/home/agiuser/projects/x2_docker")
REAL_MOTION_JSON = (
  X2_TREE
  / "x2_controller/x2_rl_deploy_controller/config/rl_model/vgk8-baseline/motions/vgk000.json"
)
REAL_SOURCE_NPZ = Path("data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz")
# The anchor column of the real npz, from that bundle's audited robot body order.
REAL_FULL_BODY_ORDER = tuple(
  "torso_link" if index == 15 else f"body_{index}" for index in range(32)
)


def _write_npz(
  path: Path,
  *,
  frames: int = 4,
  joints: int = 3,
  bodies: int = 3,
  fps: float = 50.0,
  anchor_scale: float = 1.0,
) -> Path:
  rng = np.random.default_rng(0)
  anchor = rng.standard_normal((frames, 4)).astype(np.float32)
  anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
  np.savez(
    path,
    fps=np.float32(fps),
    joint_pos=rng.standard_normal((frames, joints)).astype(np.float32),
    joint_vel=rng.standard_normal((frames, joints)).astype(np.float32),
    body_quat_w=(anchor[:, None, :] * anchor_scale)
    .astype(np.float32)
    .repeat(bodies, axis=1),
  )
  return path


def _contract(**overrides):
  values = {
    "joint_names": JOINT_NAMES,
    "body_names": BODY_NAMES,
    "anchor_body_name": ANCHOR,
  }
  values.update(overrides)
  return ReferenceContract(**values)


def test_payload_uses_exactly_the_controller_key_set(tmp_path):
  payload = build_clip_payload(_write_npz(tmp_path / "clip.npz"), _contract())
  assert set(payload) == CONTROLLER_MOTION_KEYS
  assert payload["format"] == "mjlab-vae-motion"
  assert payload["version"] == 2
  # The reader compares fps to 50.0 exactly, so the writer must not emit 50 or 50.000001.
  assert payload["fps"] == 50.0
  assert isinstance(payload["fps"], float)
  assert payload["frames"] == 4
  assert payload["joint_order"] == list(JOINT_NAMES)
  assert payload["q"] == np.asarray(payload["q"]).tolist()


def test_payload_row_widths_follow_the_joint_order(tmp_path):
  payload = build_clip_payload(_write_npz(tmp_path / "clip.npz", frames=6), _contract())
  assert len(payload["q"]) == 6
  assert all(len(row) == len(JOINT_NAMES) for row in payload["q"])
  assert len(payload["dq"]) == 6
  assert all(len(row) == len(JOINT_NAMES) for row in payload["dq"])
  assert len(payload["anchor_quat_wxyz"]) == 6
  assert all(len(row) == 4 for row in payload["anchor_quat_wxyz"])


def test_anchor_column_comes_from_the_full_body_order(tmp_path):
  """The npz stores every body, so the anchor is selected by the full body order."""
  path = _write_npz(tmp_path / "clip.npz", frames=3, bodies=5)
  contract = _contract(body_names=("a", "b", "torso_link", "d", "e"))
  payload = build_clip_payload(path, contract)
  with np.load(path) as data:
    assert np.allclose(payload["anchor_quat_wxyz"], data["body_quat_w"][:, 2, :])


def test_a_wrong_frame_rate_is_refused(tmp_path):
  path = _write_npz(tmp_path / "clip.npz", fps=60.0)
  with pytest.raises(ReferenceBundleError, match="50.0"):
    build_clip_payload(path, _contract())


def test_a_body_count_mismatch_is_refused(tmp_path):
  path = _write_npz(tmp_path / "clip.npz", bodies=4)
  with pytest.raises(ReferenceBundleError, match="body_quat_w"):
    build_clip_payload(path, _contract())


def test_a_joint_count_mismatch_is_refused(tmp_path):
  path = _write_npz(tmp_path / "clip.npz", joints=4)
  with pytest.raises(ReferenceBundleError, match="joint_pos/joint_vel"):
    build_clip_payload(path, _contract())


def test_a_non_unit_anchor_is_refused(tmp_path):
  path = _write_npz(tmp_path / "clip.npz", anchor_scale=2.0)
  with pytest.raises(ReferenceBundleError, match="unit length"):
    build_clip_payload(path, _contract())


def test_a_missing_required_array_is_refused(tmp_path):
  path = tmp_path / "clip.npz"
  np.savez(path, fps=np.float32(50.0), joint_pos=np.zeros((2, 3), np.float32))
  with pytest.raises(ReferenceBundleError, match="missing"):
    build_clip_payload(path, _contract())


def test_directories_and_duplicate_stems_are_refused(tmp_path):
  with pytest.raises(ReferenceBundleError, match="safe file stem"):
    write_reference_bundle(tmp_path / "out", {"a/b": tmp_path / "x.npz"}, _contract())
  with pytest.raises(ReferenceBundleError, match="no clips"):
    write_reference_bundle(tmp_path / "out", {}, _contract())
  first = tmp_path / "one" / "clip.npz"
  second = tmp_path / "two" / "clip.npz"
  first.parent.mkdir()
  second.parent.mkdir()
  with pytest.raises(ReferenceBundleError, match="duplicate clip name"):
    clip_paths([first, second])


def test_a_name_pattern_shortens_long_source_names(tmp_path):
  paths = [
    tmp_path / "single_000_zhanghongyu_agibot_x2_tracking.npz",
    tmp_path / "single_007_zhanghongyu_agibot_x2_tracking.npz",
  ]
  clips = clip_paths(paths, name_pattern=r"^(single_\d+)")
  assert list(clips) == ["single_000", "single_007"]
  # Two sources that shorten to one name are refused, not silently overwritten.
  with pytest.raises(ReferenceBundleError, match="duplicate clip name"):
    clip_paths(
      [
        tmp_path / "single_000_a.npz",
        tmp_path / "single_000_b.npz",
      ],
      name_pattern=r"^(single_\d+)",
    )
  with pytest.raises(ReferenceBundleError, match="did not capture"):
    clip_paths([tmp_path / "other.npz"], name_pattern=r"^(single_\d+)")


def test_bundle_writes_one_file_per_clip_plus_a_manifest(tmp_path):
  clip_a = _write_npz(tmp_path / "single_000.npz", frames=5)
  clip_b = _write_npz(tmp_path / "single_001.npz", frames=7)
  out = tmp_path / "bundle"
  manifest = write_reference_bundle(out, clip_paths([clip_a, clip_b]), _contract())

  written = sorted(path.name for path in (out / "motions").glob("*.json"))
  assert written == ["single_000.json", "single_001.json"]
  assert (out / "manifest.json").is_file()

  assert manifest["format"] == "x2-reference-clip-bundle"
  assert [entry["name"] for entry in manifest["clips"]] == ["single_000", "single_001"]
  assert [entry["frames"] for entry in manifest["clips"]] == [5, 7]
  assert manifest["reference_contract"]["anchor_body_name"] == ANCHOR
  assert manifest["reference_contract"]["anchor_body_index"] == 2
  assert manifest["reference_contract"]["joint_order"] == list(JOINT_NAMES)
  for entry in manifest["clips"]:
    assert len(entry["source_npz_sha256"]) == 64
    assert len(entry["motion_sha256"]) == 64
    # The written file must round-trip back to the controller's key set.
    payload = json.loads((out / entry["motion_file"]).read_text())
    assert set(payload) == CONTROLLER_MOTION_KEYS


def test_contract_refuses_an_anchor_outside_the_body_list():
  with pytest.raises(ReferenceBundleError, match="not in the robot body list"):
    ReferenceContract(
      joint_names=JOINT_NAMES, body_names=BODY_NAMES, anchor_body_name="missing"
    )


@pytest.mark.skipif(
  not (REAL_MOTION_JSON.is_file() and REAL_SOURCE_NPZ.is_file()),
  reason="the deployed X2 bundle and its source clip are not present",
)
def test_writer_reproduces_the_format_the_controller_already_accepts(tmp_path):
  """The same npz must produce the same tables as the bundle in the field."""
  deployed = json.loads(REAL_MOTION_JSON.read_text())
  payload = build_clip_payload(
    REAL_SOURCE_NPZ,
    ReferenceContract(
      joint_names=tuple(deployed["joint_order"]),
      body_names=REAL_FULL_BODY_ORDER,
      anchor_body_name=ANCHOR,
    ),
  )

  assert set(payload) == set(deployed) == CONTROLLER_MOTION_KEYS
  assert payload["format"] == deployed["format"]
  assert payload["version"] == deployed["version"]
  assert payload["fps"] == deployed["fps"]
  assert payload["frames"] == deployed["frames"]
  assert payload["joint_order"] == deployed["joint_order"]
  assert np.allclose(payload["q"], deployed["q"], atol=0.0)
  assert np.allclose(payload["dq"], deployed["dq"], atol=0.0)
  assert np.allclose(
    payload["anchor_quat_wxyz"], deployed["anchor_quat_wxyz"], atol=0.0
  )
