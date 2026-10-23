"""Chunked open-loop rollout for the latent-free joint-pose flow model.

A training window is ``[window_steps, channels]``: rows ``[0, past_steps)`` are
clean conditioning history and rows ``[past_steps, window_steps)`` are generated.
Every row is heading-local to the window's centre row ``past_steps``, so one
window says nothing about where the next window starts.  This module turns a
long reference clip into one long generated series by re-expressing each new
window in the heading frame of a row that is already known.

Frame accounting
----------------
``stride = window_steps - past_steps - 1`` is the largest step whose next centre
row was produced by the previous chunk.  With ``P = past_steps`` and
``W = window_steps``, the file row ``i`` is clip frame ``start_frame + i``:

* rows ``[0, P)`` are the clip prefix, written from the clip, not generated;
* chunk 0 conditions on clip frames ``s .. s+P-1``, takes its anchor from clip
  frame ``s+P`` and writes absolute frames ``s+P .. s+W-1``;
* chunk ``k > 0`` starts at ``s + stride*k``, conditions on the ``P``
  accumulated rows before the last one and takes its anchor from the last
  accumulated row, which is absolute frame ``s + stride*k + P``;
* the anchor row itself is regenerated, so each chunk adds exactly ``stride``
  new frames, and the next centre row is the previous chunk's last row.

Chunk count is ``ceil(max(0, target_frames - W) / stride) + 1`` and the series is
truncated to ``target_frames`` rows.  A 453-frame target on the ``[41, 68]``
schema needs 14 chunks: they cover 449 generated frames, the prefix makes 457
rows, and the tail beyond 453 is not written.

Every window is generated in the frame of its own anchor: the ``P`` history rows
are rotated by ``Rz(-yaw(anchor))``, the window is generated there, and the
generated rows are rotated back by ``+yaw(anchor)``.  Root translation and world
root orientation never enter the model; only the joint channels are generated.

What this module does not do
----------------------------
It never opens the prepared dataset directory and never runs
:class:`~mjlab.tasks.tracking.joint_pose.runner.FlowMatchingRunner`.  The schema,
the denoiser sizes and the noise scale come from the checkpoint payload, and a
payload that does not describe one model is refused.  The loop depends on
:class:`WindowSampler`, not on a checkpoint format, so a stub sampler that
returns recorded futures is a correctness test of the rollover itself.  This is
open-loop generation: no simulator and no robot is involved.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from ..diffusion.model_final_norm import FINAL_NORM_PARAMETER_NAMES
from ..diffusion.replay_integration import apply_ema_selection
from ..flow_matching.process import LinearNoiseProcess
from ..flow_matching.sampler import (
  FlowConditions,
  FlowGrid,
  FlowSamplerConfig,
  sample_trajectory,
)
from .config import CHANNEL_LAYOUT, JointPoseDatasetCfg, JointPoseFlowConfigError
from .evaluate import ChannelNormalization
from .model import build_denoiser
from .runner import CHECKPOINT_FORMAT

_LAYOUT_SLICES = {name: slice(start, stop) for name, start, stop in CHANNEL_LAYOUT}
ROOT_ROT6D = _LAYOUT_SLICES["root_rot6d"]
JOINT_POSITIONS = _LAYOUT_SLICES["joint_positions"]
JOINT_VELOCITIES = _LAYOUT_SLICES["joint_velocities"]
_LAYOUT_CHANNELS = CHANNEL_LAYOUT[-1][2]
"""Channels the joint-pose contract declares: root 6D, joints, joint velocities."""

_ROOT_CHANNELS = ROOT_ROT6D.stop - ROOT_ROT6D.start
_QUATERNION_WIDTH = 4
"""Values in one wxyz quaternion."""

HISTORY_TOLERANCE = 1.0e-4
"""Largest allowed change of the conditioning rows a sampler returns."""

KEYFRAME_CURRENT = "current"
"""The anchor row: the first row a window generates, and the row the tracker calls now."""

KEYFRAME_LAST = "last"
"""The last row of the window: the chunk's final frame."""

KEYFRAME_NAMES = (KEYFRAME_CURRENT, KEYFRAME_LAST)
"""Row positions a keyframe pin can name, resolved from the model geometry."""


class RolloutError(ValueError):
  """A rollout input, checkpoint or sampler does not match its contract."""


# ---------------------------------------------------------------------- maths


def quat_from_yaw(yaw: np.ndarray | float) -> np.ndarray:
  """Return wxyz quaternions for a rotation about +z by ``yaw``."""
  values = np.asarray(yaw, dtype=np.float64)
  half = values * 0.5
  quat = np.zeros(values.shape + (_QUATERNION_WIDTH,), dtype=np.float64)
  quat[..., 0] = np.cos(half)
  quat[..., 3] = np.sin(half)
  return quat


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product for broadcastable wxyz quaternions."""
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  out = np.empty(
    np.broadcast(a[..., 0], b[..., 0]).shape + (_QUATERNION_WIDTH,),
    dtype=np.float64,
  )
  out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
  out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
  out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
  out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
  return out


def yaw_of(quat: np.ndarray) -> np.ndarray:
  """Return the heading angle of wxyz quaternions."""
  values = np.asarray(quat, dtype=np.float64)
  w, x, y, z = values[..., 0], values[..., 1], values[..., 2], values[..., 3]
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
  """Return ``[..., 3, 3]`` rotation matrices from wxyz quaternions."""
  values = np.asarray(quat, dtype=np.float64)
  if values.shape[-1] != _QUATERNION_WIDTH:
    raise RolloutError(
      f"a quaternion needs {_QUATERNION_WIDTH} values, got {values.shape[-1]}"
    )
  w, x, y, z = (values[..., index] for index in range(_QUATERNION_WIDTH))
  xx, yy, zz = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = w * x, w * y, w * z
  row_zero = np.stack(
    (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), axis=-1
  )
  row_one = np.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), axis=-1)
  row_two = np.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), axis=-1)
  return np.stack((row_zero, row_one, row_two), axis=-2)


def quat_to_6d(quat: np.ndarray) -> np.ndarray:
  """Return the first two rotation-matrix columns, as ``prepare.py`` packs them.

  The order is ``R00, R10, R20, R01, R11, R21``: column zero, then column one.
  """
  matrix = matrix_from_quat(quat)
  columns = np.swapaxes(matrix[..., :, :2], -1, -2)
  return np.reshape(columns, matrix.shape[:-2] + (6,))


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
  """Return unit wxyz quaternions from ``[..., 3, 3]`` proper rotations.

  The largest of the four diagonal combinations names the dominant quaternion
  component, so the division never amplifies rounding noise near a half turn.
  """
  m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
  m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
  m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
  candidates = np.stack(
    (
      1.0 + m00 + m11 + m22,
      1.0 + m00 - m11 - m22,
      1.0 - m00 + m11 - m22,
      1.0 - m00 - m11 + m22,
    ),
    axis=-1,
  )
  index = np.argmax(candidates, axis=-1)
  largest = np.maximum(
    np.take_along_axis(candidates, index[..., None], axis=-1)[..., 0], 0.0
  )
  root = np.sqrt(largest)
  half = 0.5 * root
  safe = np.where(root > 0.0, root, 1.0)
  scale = np.where(root > 0.0, 0.5 / safe, 0.0)

  def pick(values: tuple[np.ndarray, ...]) -> np.ndarray:
    stacked = np.stack(values, axis=-1)
    return np.take_along_axis(stacked, index[..., None], axis=-1)[..., 0]

  quat = np.stack(
    (
      pick((half, (m21 - m12) * scale, (m02 - m20) * scale, (m10 - m01) * scale)),
      pick(((m21 - m12) * scale, half, (m01 + m10) * scale, (m02 + m20) * scale)),
      pick(((m02 - m20) * scale, (m01 + m10) * scale, half, (m12 + m21) * scale)),
      pick(((m10 - m01) * scale, (m02 + m20) * scale, (m12 + m21) * scale, half)),
    ),
    axis=-1,
  )
  norm = np.linalg.norm(quat, axis=-1, keepdims=True)
  return quat / np.where(norm < 1.0e-12, 1.0, norm)


def quat_from_6d(six: np.ndarray) -> np.ndarray:
  """Return unit wxyz quaternions from two packed rotation columns.

  The two packed columns are neither orthogonal nor unit length after a
  denormalization, so Gram-Schmidt restores an orthonormal basis first.  This is
  the re-orthonormalization the preprocessing contract asks for at inference.
  """
  values = np.asarray(six, dtype=np.float64)
  if values.shape[-1] != 6:
    raise RolloutError(f"a 6D rotation needs 6 values, got {values.shape[-1]}")
  first = values[..., 0:3]
  second = values[..., 3:6]
  first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
  axis_one = first / np.where(first_norm < 1.0e-8, 1.0, first_norm)
  projection = np.sum(second * axis_one, axis=-1, keepdims=True)
  orthogonal = second - projection * axis_one
  second_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
  axis_two = orthogonal / np.where(second_norm < 1.0e-8, 1.0, second_norm)
  axis_three = np.cross(axis_one, axis_two)
  return _quat_from_matrix(np.stack((axis_one, axis_two, axis_three), axis=-1))


def _index(value: Any, name: str) -> int:
  """Return ``value`` as an ``int``, refusing anything that is not integral."""
  try:
    return int(operator.index(value))
  except TypeError as exc:
    raise RolloutError(f"{name} must be an integer, got {value!r}") from exc


# ------------------------------------------------------------------- codec


@dataclass(frozen=True)
class ClipArrays:
  """One reference clip in the joint-space channels the model was trained on.

  ``root_quat_wxyz`` is the root body's world-frame wxyz quaternion, which a
  tracking file carries as the first body of ``body_quat_w``.  ``joint_pos`` is
  in radians and ``joint_vel`` in radians/second, both ``[rows, joints]``.
  """

  root_quat_wxyz: np.ndarray
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  fps: float

  def __post_init__(self) -> None:
    quat = np.asarray(self.root_quat_wxyz, dtype=np.float32)
    pos = np.asarray(self.joint_pos, dtype=np.float32)
    vel = np.asarray(self.joint_vel, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != _QUATERNION_WIDTH:
      raise RolloutError(
        f"root_quat_wxyz must be [rows, {_QUATERNION_WIDTH}], got {quat.shape}"
      )
    if pos.ndim != 2 or vel.shape != pos.shape:
      raise RolloutError(
        f"joint_pos {pos.shape} and joint_vel {vel.shape} must be equal [rows, J]"
      )
    if quat.shape[0] != pos.shape[0]:
      raise RolloutError(
        f"{quat.shape[0]} root quaternions but {pos.shape[0]} joint rows"
      )
    if not (
      np.isfinite(quat).all() and np.isfinite(pos).all() and np.isfinite(vel).all()
    ):
      raise RolloutError("clip holds a non-finite value")
    if not np.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
      raise RolloutError("fps must be finite and positive")
    object.__setattr__(self, "root_quat_wxyz", quat)
    object.__setattr__(self, "joint_pos", pos)
    object.__setattr__(self, "joint_vel", vel)
    object.__setattr__(self, "fps", float(self.fps))

  @property
  def rows(self) -> int:
    """Clip frames available."""
    return int(self.root_quat_wxyz.shape[0])

  @classmethod
  def from_arrays(cls, arrays: Mapping[str, Any]) -> "ClipArrays":
    """Build a clip from the arrays of one tracking motion file."""
    try:
      quat = np.asarray(arrays["body_quat_w"], dtype=np.float32)
      pos = np.asarray(arrays["joint_pos"], dtype=np.float32)
      vel = np.asarray(arrays["joint_vel"], dtype=np.float32)
      fps = float(np.asarray(arrays["fps"], dtype=np.float64).ravel()[0])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
      raise RolloutError(f"clip arrays are incomplete: {exc}") from exc
    if quat.ndim != 3 or quat.shape[2] != _QUATERNION_WIDTH:
      raise RolloutError(
        f"body_quat_w must be [rows, bodies, {_QUATERNION_WIDTH}], got {quat.shape}"
      )
    return cls(root_quat_wxyz=quat[:, 0], joint_pos=pos, joint_vel=vel, fps=fps)

  @classmethod
  def from_npz(cls, path: Path | str) -> "ClipArrays":
    """Read a tracking motion file, taking the root body of ``body_quat_w``."""
    source = Path(path).expanduser()
    try:
      with np.load(source, allow_pickle=False) as data:
        arrays = {
          name: np.asarray(data[name])
          for name in ("body_quat_w", "joint_pos", "joint_vel", "fps")
        }
    except (OSError, KeyError, ValueError) as exc:
      raise RolloutError(f"could not read clip {source}: {exc}") from exc
    try:
      return cls.from_arrays(arrays)
    except RolloutError as exc:
      raise RolloutError(f"{source}: {exc}") from exc


def encode_window(
  root_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  start: int,
  *,
  past_steps: int,
  window_steps: int,
) -> np.ndarray:
  """Return one heading-local window of ``[window_steps, channels]``.

  Reproduces ``prepare.segment_windows``: every row of the window is rotated by
  ``Rz(-yaw(quat[start + past_steps]))``, channels ``0:6`` are the 6D packing of
  the rotated quaternion, then the joint positions and the joint velocities.
  """
  quat = np.asarray(root_quat_wxyz, dtype=np.float64)
  pos = np.asarray(joint_pos, dtype=np.float64)
  vel = np.asarray(joint_vel, dtype=np.float64)
  past = _index(past_steps, "past_steps")
  window = _index(window_steps, "window_steps")
  first = _index(start, "start")
  if quat.ndim != 2 or quat.shape[1] != _QUATERNION_WIDTH:
    raise RolloutError(
      f"root_quat_wxyz must be [rows, {_QUATERNION_WIDTH}], got {quat.shape}"
    )
  if pos.ndim != 2 or vel.shape != pos.shape or quat.shape[0] != pos.shape[0]:
    raise RolloutError(
      f"joint_pos {pos.shape}, joint_vel {vel.shape} and {quat.shape[0]} root "
      "quaternions do not describe one clip"
    )
  if not 0 <= past < window:
    raise RolloutError(f"past_steps {past} must lie inside a {window}-row window")
  if first < 0 or first + window > quat.shape[0]:
    raise RolloutError(
      f"window start {first} plus {window} rows exceeds {quat.shape[0]} clip rows"
    )
  rows = slice(first, first + window)
  rotate = quat_from_yaw(-yaw_of(quat[first + past]))
  local = quat_mul(rotate[None, :], quat[rows])
  joints = pos.shape[1]
  out = np.empty((window, _ROOT_CHANNELS + 2 * joints), dtype=np.float32)
  out[:, ROOT_ROT6D] = quat_to_6d(local)
  out[:, _ROOT_CHANNELS : _ROOT_CHANNELS + joints] = pos[rows]
  out[:, _ROOT_CHANNELS + joints :] = vel[rows]
  return out


def rotate_rows(rows: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
  """Rotate the root channel of physical rows by ``Rz(yaw(target_quat_wxyz))``.

  The result is ``Rz(yaw(target)) * quat(row)``.  Callers therefore pass
  ``quat_from_yaw(-yaw(anchor))`` to express rows in the anchor frame, and the
  anchor itself to return generated rows to the world frame.  The joint position
  and joint velocity channels are copied unchanged, because a heading change does
  not move a joint.  Rows stay in their own units; nothing is normalized here.
  """
  values = np.asarray(rows, dtype=np.float64)
  target = np.asarray(target_quat_wxyz, dtype=np.float64)
  if values.ndim != 2 or values.shape[1] <= _ROOT_CHANNELS:
    raise RolloutError(
      f"rows must be [count, channels] with more than {_ROOT_CHANNELS} channels, "
      f"got {values.shape}"
    )
  if target.shape != (_QUATERNION_WIDTH,):
    raise RolloutError(
      f"target_quat_wxyz must be [{_QUATERNION_WIDTH}], got {target.shape}"
    )
  rotate = quat_from_yaw(yaw_of(target))
  rotated = quat_mul(rotate[None, :], quat_from_6d(values[:, ROOT_ROT6D]))
  out = np.array(values, dtype=np.float64, copy=True)
  out[:, ROOT_ROT6D] = quat_to_6d(rotated)
  return out


# ------------------------------------------------------------------ sampler


@runtime_checkable
class WindowSampler(Protocol):
  """One window generator, with the geometry the rollover needs.

  ``sample`` receives ``[past_steps, channels]`` normalized history rows and
  returns ``[window_steps, channels]`` normalized rows.  Rows ``0:past_steps``
  of the result must equal the input rows: they are conditioning, not output.
  """

  past_steps: int
  window_steps: int
  channels: int

  def sample(
    self,
    history: Tensor,
    *,
    steps: int,
    generator: torch.Generator,
    pinned: Mapping[int, Tensor] | None = None,
  ) -> Tensor:
    """Return one window; ``pinned`` rows must also come back unchanged."""
    ...


class JointPoseFlowSampler:
  """Window generator backed by one joint-pose flow checkpoint.

  Every value a run must report comes from the checkpoint payload and is kept in
  :attr:`scheme`, so a caller writes the schema it used instead of a code
  default.  ``parameters`` names the weight set: ``"ema"`` is the shadow the
  trainer maintains for evaluation, ``"model"`` is the live set.
  """

  def __init__(
    self,
    model: torch.nn.Module,
    *,
    past_steps: int,
    window_steps: int,
    channels: int,
    data_dir: str,
    width: int,
    layers: int,
    heads: int,
    final_norm: bool,
    max_level: int,
    global_step: int | None,
    parameters: str,
    device: torch.device,
  ) -> None:
    self.past_steps = past_steps
    self.window_steps = window_steps
    self.channels = channels
    self.max_level = max_level
    self.global_step = global_step
    self.parameters = parameters
    self.scheme: dict[str, object] = {
      "past_steps": past_steps,
      "window_steps": window_steps,
      "channels": channels,
      "data_dir": data_dir,
      "width": width,
      "layers": layers,
      "heads": heads,
      "final_norm": final_norm,
      "max_level": max_level,
    }
    self._model = model
    self._process = LinearNoiseProcess(max_level)
    self._device = device

  def sample(
    self,
    history: Tensor,
    *,
    steps: int,
    generator: torch.Generator,
    pinned: Mapping[int, Tensor] | None = None,
  ) -> Tensor:
    """Generate one window from clean history, then return it on the CPU.

    Rows ``[0, past_steps)`` start at the clean level and the generator rows
    start as pure Gaussian at ``max_level``; the uniform integer grid descends
    in ``steps`` denoiser calls.  The mask marks whole rows known or unknown, so
    the state/latent split the sampler infers cannot change any step id.

    ``pinned`` holds extra clean rows by row index, in normalized channels.  They
    are the inpainting surface: the sampler restores them at level 0 after every
    step, so the generated rows have to reach them.  The model was trained with
    clean rows only in the history, so a pinned future row is an inference-time
    request, not a trained behaviour.
    """
    if not isinstance(history, Tensor):
      raise RolloutError("history must be a torch tensor")
    if tuple(history.shape) != (self.past_steps, self.channels):
      raise RolloutError(
        f"history must be [{self.past_steps}, {self.channels}], "
        f"got {tuple(history.shape)}"
      )
    calls = _index(steps, "steps")
    if not 1 <= calls <= self.max_level:
      raise RolloutError(
        f"steps {calls} must lie in 1..{self.max_level}, the trained noise scale"
      )
    clean = torch.zeros(
      (1, self.window_steps, self.channels),
      dtype=torch.float32,
      device=self._device,
    )
    clean[0, : self.past_steps] = history.to(device=self._device, dtype=torch.float32)
    if pinned:
      for row, values in pinned.items():
        index = _index(row, "pinned row")
        if not self.past_steps <= index < self.window_steps:
          raise RolloutError(
            f"pinned row {index} must lie in [{self.past_steps}, {self.window_steps})"
          )
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self._device)
        if tuple(tensor.shape) != (self.channels,):
          raise RolloutError(
            f"pinned row {index} must be [{self.channels}], got {tuple(tensor.shape)}"
          )
        clean[0, index] = tensor
    if not bool(torch.isfinite(clean).all()):
      raise RolloutError("history contains NaN or Inf")
    mask = torch.zeros(
      (1, self.window_steps, self.channels),
      dtype=torch.bool,
      device=self._device,
    )
    mask[:, : self.past_steps] = True
    if pinned:
      mask[0, list(pinned)] = True
    grid = FlowGrid.uniform(calls=calls, max_level=self.max_level)
    with torch.no_grad():
      sample, _ = sample_trajectory(
        self._model,
        conditions=FlowConditions(clean_tokens=clean, clean_mask=mask),
        config=FlowSamplerConfig(grid=grid, process=self._process),
        generator=generator,
      )
    return sample[0].detach().to(device="cpu", dtype=torch.float32)


def _mapping(payload: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
  """Return one required mapping of a checkpoint payload."""
  value = payload.get(key)
  if not isinstance(value, Mapping):
    raise RolloutError(f"{path} records no {key!r} mapping")
  return value


def _integer(payload: Mapping[str, Any], key: str, path: Path) -> int:
  """Return one required integer field of a checkpoint payload."""
  if key not in payload:
    raise RolloutError(f"{path} records no {key!r} value")
  return _index(payload[key], f"{path}: {key}")


def joint_pose_flow(
  checkpoint: Path | str, *, parameters: str = "auto", device: str = "cpu"
) -> JointPoseFlowSampler:
  """Load one joint-pose flow checkpoint as a window generator.

  The schema, the denoiser sizes and the noise scale are read from the payload,
  and a payload that does not describe one model is refused rather than
  repaired.  No dataset directory is opened: the recorded ``data_dir`` is
  reported, not read.  ``parameters`` is ``auto`` (the EMA shadow when the
  payload carries one, else the live weights), or an explicit ``ema`` or
  ``model`` that is refused when it cannot be served.  The resolved set is
  recorded on the sampler.
  """
  path = Path(checkpoint).expanduser()
  try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
  except (OSError, RuntimeError, EOFError, ValueError) as exc:
    raise RolloutError(f"could not read joint-pose flow checkpoint {path}") from exc
  if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
    raise RolloutError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
  if parameters not in ("auto", "ema", "model"):
    raise RolloutError("parameters must be 'auto', 'ema' or 'model'")
  shadow = payload.get("ema")
  if parameters == "auto":
    parameters = "model" if shadow is None else "ema"
  if parameters == "ema" and shadow is None:
    raise RolloutError(
      f"{path} carries no EMA shadow, so parameters='ema' cannot be served"
    )
  dataset = _mapping(payload, "dataset_cfg", path)
  settings = _mapping(payload, "model_settings", path)
  trainer = _mapping(payload, "trainer", path)
  state = _mapping(payload, "model", path)
  past_steps = _integer(dataset, "past_steps", path)
  window_steps = _integer(dataset, "window_steps", path)
  channels = _integer(dataset, "channels", path)
  recorded_layout = dataset.get("channel_layout")
  if recorded_layout is not None:
    expected_layout = [[name, start, stop] for name, start, stop in CHANNEL_LAYOUT]
    if [list(item) for item in recorded_layout] != expected_layout:
      raise RolloutError(
        f"{path} records channel_layout {recorded_layout}, expected {expected_layout}"
      )
  data_dir = dataset.get("data_dir")
  if not isinstance(data_dir, str):
    raise RolloutError(f"{path} records no dataset 'data_dir' string")
  width = _integer(settings, "width", path)
  layers = _integer(settings, "layers", path)
  heads = _integer(settings, "attention_heads", path)
  embeddings = _integer(settings, "embedding_count", path)
  max_level = (
    _integer(trainer, "max_level", path) if "max_level" in trainer else embeddings - 1
  )
  if embeddings != max_level + 1:
    raise RolloutError(
      f"{path} records max_level {max_level} but {embeddings} level embeddings"
    )
  recorded_tokens = _integer(settings, "token_dimension", path)
  recorded_rows = _integer(settings, "sequence_length", path)
  if (recorded_tokens, recorded_rows) != (channels, window_steps):
    raise RolloutError(
      f"{path} model settings describe [{recorded_rows}, {recorded_tokens}] but "
      f"the dataset config describes [{window_steps}, {channels}]"
    )
  final_norm = settings.get("final_norm")
  if final_norm is None:
    final_norm = any(name in state for name in FINAL_NORM_PARAMETER_NAMES)
  if not isinstance(final_norm, bool):
    raise RolloutError(f"{path} records a non-boolean final_norm {final_norm!r}")
  try:
    dataset_cfg = JointPoseDatasetCfg(
      data_dir="",
      window_steps=window_steps,
      past_steps=past_steps,
      channels=channels,
    )
  except (JointPoseFlowConfigError, TypeError) as exc:
    raise RolloutError(f"{path} records an unusable window schema: {exc}") from exc
  try:
    model = build_denoiser(
      dataset_cfg,
      max_level=max_level,
      width=width,
      layers=layers,
      heads=heads,
      final_norm=final_norm,
    )
  except (ValueError, TypeError) as exc:
    raise RolloutError(f"{path} model settings cannot be built: {exc}") from exc
  try:
    model.load_state_dict(dict(state), strict=True)
  except (RuntimeError, TypeError) as exc:
    raise RolloutError(f"{path} state does not match the built model: {exc}") from exc
  if parameters == "ema":
    ema_state = shadow
    try:
      selected, _ = apply_ema_selection(model, ema_state)
    except (ValueError, RuntimeError, TypeError) as exc:
      raise RolloutError(f"{path} EMA shadow cannot be restored: {exc}") from exc
    if not selected:
      raise RolloutError(f"{path} EMA shadow could not be applied to the model")
  try:
    target = torch.device(device)
  except (RuntimeError, TypeError) as exc:
    raise RolloutError(f"device {device!r} is not a torch device") from exc
  if target.type == "cuda" and not torch.cuda.is_available():
    raise RolloutError("CUDA device requested but CUDA is unavailable")
  model = model.to(target)
  model.eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  global_step = payload.get("global_step")
  return JointPoseFlowSampler(
    model,
    past_steps=past_steps,
    window_steps=window_steps,
    channels=channels,
    data_dir=data_dir,
    width=width,
    layers=layers,
    heads=heads,
    final_norm=final_norm,
    max_level=max_level,
    global_step=None if global_step is None else _index(global_step, "global_step"),
    parameters=parameters,
    device=target,
  )


# ------------------------------------------------------------------ rollout


@dataclass(frozen=True)
class ChunkRecord:
  """What one chunk did: where it started, and how far it left the record.

  ``overlap_rms_rad`` is the joint RMS change of the regenerated centre row
  against the value the previous chunk already wrote there, and is ``None`` for
  the first chunk.  ``joint_drift_rms_rad`` and ``yaw_drift_deg`` compare the
  generated rows with the clip at the same absolute frames, and are ``None``
  when no generated frame of the chunk has a clip row to compare with.
  """

  index: int
  start_frame: int
  anchor_yaw_deg: float
  overlap_rms_rad: float | None
  joint_drift_rms_rad: float | None
  yaw_drift_deg: float | None
  pinned: tuple[int, ...] = ()
  seam_rms_rad: float | None = None


@dataclass(frozen=True)
class RolloutResult:
  """Generated rows in world frame, plus the per-chunk diagnostics.

  Row ``i`` is clip frame ``start_frame + i``.  Rows ``[0, past_steps)`` are the
  clip prefix, not generated rows.
  """

  root_quat_wxyz: np.ndarray
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  chunks: tuple[ChunkRecord, ...]
  stop_reason: str
  steps: int

  def __post_init__(self) -> None:
    quat = np.asarray(self.root_quat_wxyz, dtype=np.float32)
    pos = np.asarray(self.joint_pos, dtype=np.float32)
    vel = np.asarray(self.joint_vel, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != _QUATERNION_WIDTH:
      raise RolloutError(
        f"root_quat_wxyz must be [rows, {_QUATERNION_WIDTH}], got {quat.shape}"
      )
    if pos.ndim != 2 or vel.shape != pos.shape or quat.shape[0] != pos.shape[0]:
      raise RolloutError(
        f"root_quat_wxyz {quat.shape}, joint_pos {pos.shape} and joint_vel "
        f"{vel.shape} do not describe one series"
      )
    if self.stop_reason not in ("target", "drift_limit"):
      raise RolloutError(f"unknown stop_reason {self.stop_reason!r}")
    object.__setattr__(self, "root_quat_wxyz", quat)
    object.__setattr__(self, "joint_pos", pos)
    object.__setattr__(self, "joint_vel", vel)


def _normalize(normalization: ChannelNormalization, rows: np.ndarray) -> np.ndarray:
  """Map physical rows into the dataset's normalized channel space.

  ``ChannelNormalization`` only exposes the inverse direction, so the forward
  direction is spelled out here with the same per-channel statistics.
  """
  values = np.asarray(rows, dtype=np.float64)
  if values.shape[-1] != normalization.channels:
    raise RolloutError(
      f"rows have {values.shape[-1]} channels, the normalization holds "
      f"{normalization.channels}"
    )
  return (values - normalization.mean) / normalization.std


def _world_rows(clip: ClipArrays, start: int, count: int, channels: int) -> np.ndarray:
  """Return ``count`` clip frames as world-frame physical rows."""
  rows = slice(start, start + count)
  out = np.empty((count, channels), dtype=np.float64)
  out[:, ROOT_ROT6D] = quat_to_6d(np.asarray(clip.root_quat_wxyz[rows], np.float64))
  out[:, JOINT_POSITIONS] = clip.joint_pos[rows]
  out[:, JOINT_VELOCITIES] = clip.joint_vel[rows]
  return out


def _sampled_window(sampled: Any, window_steps: int, channels: int) -> np.ndarray:
  """Return one sampler result as finite float64 numpy rows."""
  if not isinstance(sampled, Tensor):
    raise RolloutError("sampler.sample must return a torch tensor")
  if tuple(sampled.shape) != (window_steps, channels):
    raise RolloutError(
      f"sampler.sample returned {tuple(sampled.shape)}, expected "
      f"[{window_steps}, {channels}]"
    )
  values = sampled.detach().to(device="cpu", dtype=torch.float64).numpy()
  if not np.isfinite(values).all():
    raise RolloutError("sampler.sample returned NaN or Inf")
  return values


def _rms(values: np.ndarray) -> float:
  return float(np.sqrt(np.mean(np.square(values))))


def _wrapped_degrees(delta_rad: float) -> float:
  return float(np.rad2deg(np.arctan2(np.sin(delta_rad), np.cos(delta_rad))))


def parse_keyframes(spec: str) -> tuple[str, ...]:
  """Resolve one keyframe request into the row names to pin.

  ``"none"`` or an empty string pins nothing.  Otherwise a comma-separated list
  of :data:`KEYFRAME_NAMES`, in any order and without duplicates.  The names are
  row positions, resolved from the model geometry by :func:`rollout`, so they do
  not depend on the window size.
  """
  if not isinstance(spec, str):
    raise RolloutError("keyframes must be a string")
  text = spec.strip().lower()
  if text in ("", "none"):
    return ()
  names = [item.strip() for item in text.split(",") if item.strip()]
  unknown = [name for name in names if name not in KEYFRAME_NAMES]
  if unknown:
    raise RolloutError(
      f"unknown keyframes {unknown}; choose none or {', '.join(KEYFRAME_NAMES)}"
    )
  if len(set(names)) != len(names):
    raise RolloutError(f"keyframes {spec!r} repeats a name")
  return tuple(name for name in KEYFRAME_NAMES if name in names)


def _reference_row(
  clip: ClipArrays,
  frame: int,
  anchor_yaw: float,
  normalization: ChannelNormalization,
  channels: int,
) -> np.ndarray:
  """Return one normalized clip row expressed in the window's anchor frame."""
  row = np.empty((1, channels), dtype=np.float64)
  quat = np.asarray(clip.root_quat_wxyz[frame], dtype=np.float64)
  row[0, ROOT_ROT6D] = quat_to_6d(quat[None, :])[0]
  row[0, JOINT_POSITIONS] = np.asarray(clip.joint_pos[frame], dtype=np.float64)
  row[0, JOINT_VELOCITIES] = np.asarray(clip.joint_vel[frame], dtype=np.float64)
  return _normalize(normalization, rotate_rows(row, quat_from_yaw(-anchor_yaw)))[0]


def rollout(
  *,
  clip: ClipArrays,
  sampler: WindowSampler,
  normalization: ChannelNormalization,
  start_frame: int,
  target_frames: int,
  steps: int,
  generator: torch.Generator,
  drift_limit: float | None = None,
  keyframes: str = "none",
) -> RolloutResult:
  """Generate one reference series from one clip, chunk by chunk.

  ``sampler`` supplies the window geometry; ``normalization`` maps between the
  model's channel space and radians.  ``start_frame`` is the first clip frame
  written, ``target_frames`` the number of rows wanted, and ``steps`` the
  denoiser calls per chunk.  ``generator`` owns every noise draw.

  The series stops at ``target_frames`` rows, or after the first chunk whose
  joint drift against the clip exceeds ``drift_limit`` when that limit is given.

  ``keyframes`` pins reference rows inside every window: ``"current"`` fixes the
  anchor row, ``"last"`` fixes the chunk's final row, and both are given as a
  comma-separated list.  A pinned row is the recorded clip frame for the same
  absolute frame, so the chunk becomes a bridge between two real poses.  Pin
  both to keep the value: with ``stride = window - past - 1`` the last row of a
  chunk is the anchor row of the next, and the loop rewrites it.
  """
  past = _index(sampler.past_steps, "sampler.past_steps")
  window = _index(sampler.window_steps, "sampler.window_steps")
  channels = _index(sampler.channels, "sampler.channels")
  first = _index(start_frame, "start_frame")
  frames = _index(target_frames, "target_frames")
  calls = _index(steps, "steps")
  if channels != _LAYOUT_CHANNELS:
    raise RolloutError(
      f"sampler channels {channels} do not match the {_LAYOUT_CHANNELS}-channel "
      "joint-pose layout"
    )
  if window - past - 1 <= 0:
    raise RolloutError(
      f"window_steps {window} leaves no stride after past_steps {past}"
    )
  if normalization.channels != channels:
    raise RolloutError(
      f"normalization holds {normalization.channels} channels, sampler {channels}"
    )
  if _ROOT_CHANNELS + 2 * clip.joint_pos.shape[1] != channels:
    raise RolloutError(
      f"clip joint width {clip.joint_pos.shape[1]} does not fill {channels} "
      "channels with a 6D root"
    )
  if first < 0:
    raise RolloutError(f"start_frame {first} must be non-negative")
  if frames < past + 1:
    raise RolloutError(
      f"target_frames {frames} must exceed past_steps {past}, because the prefix "
      "is not generated"
    )
  if frames > clip.rows - first:
    raise RolloutError(
      f"target_frames {frames} does not fit the {clip.rows} clip rows that "
      f"remain after start_frame {first}"
    )
  if calls < 1:
    raise RolloutError("steps must be positive")
  limit: float | None = None
  if drift_limit is not None:
    limit = float(drift_limit)
    if not np.isfinite(limit) or limit < 0.0:
      raise RolloutError("drift_limit must be finite and non-negative")

  keyframe_names = parse_keyframes(keyframes)
  keyframe_rows: dict[int, str] = {}
  if KEYFRAME_CURRENT in keyframe_names:
    keyframe_rows[past] = KEYFRAME_CURRENT
  if KEYFRAME_LAST in keyframe_names:
    keyframe_rows[window - 1] = KEYFRAME_LAST

  stride = window - past - 1
  span = max(0, frames - window)
  chunk_count = (span + stride - 1) // stride + 1
  clip_pos = np.asarray(clip.joint_pos, dtype=np.float64)
  clip_quat = np.asarray(clip.root_quat_wxyz, dtype=np.float64)
  series = _world_rows(clip, first, past, channels)
  chunks: list[ChunkRecord] = []
  stop_reason = "target"

  for index in range(chunk_count):
    start = first + stride * index
    centre = index * stride + past
    if index == 0:
      # ``centre`` indexes ``series``, whose row 0 is clip frame ``first``; the
      # clip frame of the anchor row is therefore ``first + centre``.
      anchor = clip_quat[first + centre]
    else:
      anchor = quat_from_6d(series[centre, ROOT_ROT6D])
    anchor_yaw = float(yaw_of(anchor))
    previous = None if index == 0 else np.array(series[centre], dtype=np.float64)
    # Rows [0, past) end just before the anchor row, which is already known.
    history = series[centre - past : centre]
    local = rotate_rows(history, quat_from_yaw(-anchor_yaw))
    normalized = _normalize(normalization, local)
    pinned: dict[int, np.ndarray] = {}
    for row in keyframe_rows:
      frame = start + row
      if frame >= clip.rows:
        # The clip has run out, so that keyframe does not exist.
        continue
      pinned[row] = _reference_row(clip, frame, anchor_yaw, normalization, channels)
    seam = None
    if past in pinned and previous is not None:
      # How far the pinned pose sits from the value the previous chunk wrote at
      # the same frame: the snap the pin introduces at the seam.
      local_previous = rotate_rows(previous[None, :], quat_from_yaw(-anchor_yaw))
      seam = _rms(
        normalization.denormalize(local_previous)[0, JOINT_POSITIONS]
        - normalization.denormalize(pinned[past][None, :])[0, JOINT_POSITIONS]
      )
    sampled = sampler.sample(
      torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)),
      steps=calls,
      generator=generator,
      pinned={
        row: torch.from_numpy(np.ascontiguousarray(row_values, dtype=np.float32))
        for row, row_values in pinned.items()
      },
    )
    values = _sampled_window(sampled, window, channels)
    checks: list[tuple[np.ndarray, np.ndarray]] = []
    if past:
      checks.append((np.arange(past), normalized))
    for row, row_values in pinned.items():
      checks.append((np.array([row]), row_values[None, :]))
    deviation = 0.0
    for rows, expected in checks:
      deviation = max(deviation, float(np.max(np.abs(values[rows] - expected))))
    if deviation > HISTORY_TOLERANCE:
      raise RolloutError(
        f"chunk {index} changed its conditioning rows by {deviation:.3g}, above "
        f"{HISTORY_TOLERANCE:g}"
      )
    generated = rotate_rows(normalization.denormalize(values[past:]), anchor)
    # The regenerated centre row replaces the row the previous chunk wrote, so
    # exactly ``stride`` new frames are appended.
    series = np.concatenate((series[:centre], generated), axis=0)

    overlap = None
    if previous is not None:
      shadow = previous[JOINT_POSITIONS] - generated[0, JOINT_POSITIONS]
      overlap = _rms(shadow)
    comparable = min(start + window, first + frames, clip.rows) - (start + past)
    drift = None
    yaw_drift = None
    if comparable > 0:
      difference = (
        generated[:comparable, JOINT_POSITIONS]
        - clip_pos[start + past : start + past + comparable]
      )
      drift = _rms(difference)
      last = start + past + comparable - 1
      generated_yaw = float(yaw_of(quat_from_6d(generated[comparable - 1, ROOT_ROT6D])))
      yaw_drift = _wrapped_degrees(generated_yaw - float(yaw_of(clip_quat[last])))
    chunks.append(
      ChunkRecord(
        index=index,
        start_frame=start,
        anchor_yaw_deg=float(np.rad2deg(anchor_yaw)),
        overlap_rms_rad=overlap,
        joint_drift_rms_rad=drift,
        yaw_drift_deg=yaw_drift,
        pinned=tuple(sorted(pinned)),
        seam_rms_rad=seam,
      )
    )
    if limit is not None and drift is not None and drift > limit:
      stop_reason = "drift_limit"
      break

  rows = min(frames, int(series.shape[0]))
  if stop_reason == "target" and rows < frames:
    raise RolloutError(
      f"the chunk loop produced {series.shape[0]} rows, fewer than {frames}"
    )
  return RolloutResult(
    root_quat_wxyz=quat_from_6d(series[:rows, ROOT_ROT6D]),
    joint_pos=series[:rows, JOINT_POSITIONS],
    joint_vel=series[:rows, JOINT_VELOCITIES],
    chunks=tuple(chunks),
    stop_reason=stop_reason,
    steps=calls,
  )


__all__ = [
  "KEYFRAME_CURRENT",
  "KEYFRAME_LAST",
  "KEYFRAME_NAMES",
  "ChunkRecord",
  "ClipArrays",
  "HISTORY_TOLERANCE",
  "JointPoseFlowSampler",
  "RolloutError",
  "RolloutResult",
  "WindowSampler",
  "encode_window",
  "joint_pose_flow",
  "matrix_from_quat",
  "parse_keyframes",
  "quat_from_6d",
  "quat_from_yaw",
  "quat_mul",
  "quat_to_6d",
  "rotate_rows",
  "rollout",
  "yaw_of",
]
