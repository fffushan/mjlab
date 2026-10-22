"""Window dataset for latent-free joint-pose flow matching.

One flat index spans two sources: the achieved windows the tracker produced and
the reference windows it was asked to follow.  Windows are already normalized by
the dataset's train-only statistics, so this class never rescales them.

Sampling is drawn per source, not per row.  A full weighted draw over 32 M rows
would make every sample O(N); drawing the source first and then a uniform row
inside it is the same distribution at constant cost.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import JointPoseDatasetCfg

SPLITS = ("train", "validation", "test")


class JointPoseDatasetError(ValueError):
  """The prepared dataset on disk does not match the configuration."""


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _window_sample_sha256(path: Path, rows: int = 64) -> str:
  """Digest evenly spaced windows without reading a multi-hundred-GB file.

  Counting windows and comparing file sizes cannot detect a cache that was
  rewritten in place with the same shape. Sampling fixed window positions does:
  any change to the stored values moves at least one sampled window.
  """
  array = np.load(path, mmap_mode="r")
  count = int(array.shape[0])
  if count == 0:
    return hashlib.sha256(b"empty").hexdigest()
  take = min(rows, count)
  positions = sorted({int(round(i * (count - 1) / max(take - 1, 1))) for i in range(take)})
  digest = hashlib.sha256()
  for position in positions:
    digest.update(np.ascontiguousarray(array[position]).tobytes())
  return digest.hexdigest()


class JointPoseWindowDataset(Dataset):
  """Union of the achieved and reference window sources for one split.

  Rows ``[0, n_achieved)`` are achieved windows; rows ``[n_achieved, len)`` are
  reference windows.  Both are ``[window_steps, channels]`` float32 slices of a
  memory-mapped ``.npy`` file, so opening the dataset does not read the data.
  """

  def __init__(self, cfg: JointPoseDatasetCfg, split: str | None = None) -> None:
    self.cfg = cfg
    self.split = split or cfg.split
    if self.split not in SPLITS:
      raise JointPoseDatasetError(f"unknown split {self.split!r}")
    root = Path(cfg.data_dir).expanduser()
    if not root.is_dir():
      raise JointPoseDatasetError(f"dataset directory not found: {root}")
    self.root = root

    achieved_path = root / cfg.achieved_dir / f"tokens-{self.split}.npy"
    if not achieved_path.is_file():
      raise JointPoseDatasetError(f"achieved window cache not found: {achieved_path}")
    self._achieved = np.load(achieved_path, mmap_mode="r")
    self._check_shape(self._achieved, achieved_path)
    self.n_achieved = int(self._achieved.shape[0])

    self._reference: np.ndarray | None = None
    self.n_reference = 0
    reference_path = root / cfg.reference_dir / f"{self.split}.npy"
    if cfg.include_reference:
      if not reference_path.is_file():
        raise JointPoseDatasetError(f"reference window cache not found: {reference_path}")
      self._reference = np.load(reference_path, mmap_mode="r")
      self._check_shape(self._reference, reference_path)
      self.n_reference = int(self._reference.shape[0])
      self._check_split_agreement()
      self._warn_on_reference_weight()

    if self.n_achieved + self.n_reference == 0:
      raise JointPoseDatasetError(f"split {self.split} has no windows")

  def _check_shape(self, array: np.ndarray, path: Path) -> None:
    expected = (self.cfg.window_steps, self.cfg.channels)
    if tuple(array.shape[1:]) != expected:
      raise JointPoseDatasetError(
        f"{path} holds {tuple(array.shape[1:])} per window, expected {expected}"
      )
    if array.dtype != np.float32:
      raise JointPoseDatasetError(f"{path} is {array.dtype}, expected float32")

  def _check_split_agreement(self) -> None:
    """Refuse a reference source that crosses the motion split.

    A motion must not appear as reference in one split and as achieved in
    another, or the same motion leaks across train and validation.
    """
    assignments_path = self.root / "assignments.json"
    reference_manifest = self.root / self.cfg.reference_dir / "manifest.json"
    if not assignments_path.is_file() or not reference_manifest.is_file():
      return
    assignments = json.loads(assignments_path.read_text())
    expected = {
        motion: (value if isinstance(value, str) else value["split"])
        for motion, value in assignments["motions"].items()
    }
    manifest = json.loads(reference_manifest.read_text())
    crossed = [
        source["motion"]
        for source in manifest.get("sources", [])
        if expected.get(source["motion"], source["split"]) != source["split"]
    ]
    if crossed:
      raise JointPoseDatasetError(
        f"reference source crosses the split for {len(crossed)} motions, "
        f"for example {crossed[0]}"
      )

  def _warn_on_reference_weight(self) -> None:
    """Report a config weight that disagrees with the builder's solved weight."""
    manifest_path = self.root / self.cfg.reference_dir / "manifest.json"
    if not manifest_path.is_file():
      return
    sampling = json.loads(manifest_path.read_text()).get("sampling", {})
    solved = sampling.get("weight_reference")
    if solved is None:
      return
    solved = float(solved)
    if solved <= 0.0:
      return
    if abs(self.cfg.reference_weight - solved) > 0.01 * solved:
      warnings.warn(
          f"reference_weight {self.cfg.reference_weight:g} differs from the "
          f"weight {solved:g} recorded in {manifest_path.name}; the sampled "
          f"mixture is not the one the reference source was built for",
          stacklevel=2,
      )

  # ------------------------------------------------------------------ sizes

  def __len__(self) -> int:
    return self.n_achieved + self.n_reference

  @property
  def channels(self) -> int:
    return self.cfg.channels

  @property
  def current_index(self) -> int:
    return self.cfg.past_steps

  @property
  def reference_weight(self) -> float:
    return self.cfg.reference_weight if self.n_reference else 0.0

  @property
  def reference_share(self) -> float:
    """Expected share of sampled windows that come from the reference source."""
    if not self.n_reference:
      return 0.0
    reference_mass = self.reference_weight * self.n_reference
    achieved_mass = float(self.n_achieved)
    return reference_mass / (reference_mass + achieved_mass)

  # ------------------------------------------------------------------ access

  def source_of(self, index: int) -> str:
    if index < 0 or index >= len(self):
      raise IndexError(f"window index {index} is out of range")
    return "achieved" if index < self.n_achieved else "reference"

  def __getitem__(self, index: int) -> torch.Tensor:
    if index < 0:
      index += len(self)
    if index < 0 or index >= len(self):
      raise IndexError(f"window index {index} is out of range")
    if index < self.n_achieved:
      window = self._achieved[index]
    else:
      assert self._reference is not None
      window = self._reference[index - self.n_achieved]
    return torch.from_numpy(np.array(window, dtype=np.float32, copy=True))

  def draw_indices(
    self, count: int, generator: torch.Generator | None = None
  ) -> torch.Tensor:
    """Draw ``count`` window indices from the weighted two-source mixture."""
    if count <= 0:
      raise JointPoseDatasetError("count must be positive")
    if not self.n_reference:
      return torch.randint(0, self.n_achieved, (count,), generator=generator)
    share = self.reference_share
    take_reference = torch.rand(count, generator=generator) < share
    n_reference = int(take_reference.sum().item())
    n_achieved = count - n_reference
    indices = torch.empty(count, dtype=torch.long)
    if n_reference:
      indices[take_reference] = self.n_achieved + torch.randint(
          0, self.n_reference, (n_reference,), generator=generator
      )
    if n_achieved:
      indices[~take_reference] = torch.randint(
          0, self.n_achieved, (n_achieved,), generator=generator
      )
    return indices

  def __iter__(self) -> Iterator[torch.Tensor]:
    for index in range(len(self)):
      yield self[index]

  # -------------------------------------------------------------- identity

  @property
  def metadata(self) -> dict[str, object]:
    return {
        "split": self.split,
        "achieved_windows": self.n_achieved,
        "reference_windows": self.n_reference,
        "reference_weight": self.reference_weight,
        "reference_share": round(self.reference_share, 8),
        "channels": self.cfg.channels,
        "window_steps": self.cfg.window_steps,
        "past_steps": self.cfg.past_steps,
    }

  @property
  def dataset_identity(self) -> dict[str, object]:
    """Record written next to a checkpoint and compared on resume."""
    identity = dict(self.metadata)
    identity["dataset_type"] = type(self).__qualname__
    identity["data_dir"] = str(self.root)
    identity["window_count"] = len(self)
    achieved_path = self.root / self.cfg.achieved_dir / f"tokens-{self.split}.json"
    if achieved_path.is_file():
      payload = json.loads(achieved_path.read_text())
      identity["achieved_cache_bytes"] = int(payload.get("bytes", 0))
      identity["achieved_segments"] = int(payload.get("segments", 0))
    achieved_array = self.root / self.cfg.achieved_dir / f"tokens-{self.split}.npy"
    if achieved_array.is_file():
      identity["achieved_content_sha256"] = _window_sample_sha256(achieved_array)
    if self._reference is not None:
      reference_array = self.root / self.cfg.reference_dir / f"{self.split}.npy"
      identity["reference_content_sha256"] = _window_sample_sha256(reference_array)
    reference_manifest = self.root / self.cfg.reference_dir / "manifest.json"
    if reference_manifest.is_file() and self.n_reference:
      payload = json.loads(reference_manifest.read_text())
      identity["reference_format"] = str(payload.get("format", ""))
      identity["reference_source_count"] = len(payload.get("sources", []))
    assignments = self.root / "assignments.json"
    if assignments.is_file():
      identity["split_policy"] = str(
          json.loads(assignments.read_text()).get("policy", "")
      )
    index_path = self.root / "window_index.npz"
    if index_path.is_file():
      identity["window_index_bytes"] = index_path.stat().st_size
      identity["window_index_sha256"] = _file_sha256(index_path)
    projection_path = self.root / "projection.npz"
    if projection_path.is_file():
      # The normalizer decides what every training value means.
      identity["projection_sha256"] = _file_sha256(projection_path)
    if assignments.is_file():
      identity["assignments_sha256"] = _file_sha256(assignments)
    if self.n_reference:
      reference_manifest_path = self.root / self.cfg.reference_dir / "manifest.json"
      if reference_manifest_path.is_file():
        identity["reference_manifest_sha256"] = _file_sha256(
            reference_manifest_path
        )
    return identity

  def identity_sha256(self) -> str:
    payload = json.dumps(self.dataset_identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
