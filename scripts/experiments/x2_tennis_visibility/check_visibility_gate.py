"""Check the ball-stream visibility gate, as a script rather than a test.

The eight cases of ``docs/design/x2_tennis_visibility_plan.md`` section W5, plus the
gate-on admitted-path case. Each case prints its own numbers and the script exits
non-zero if any case fails. It is a **probe**: at most 8 environments and 40 control
steps per case, no training, no checkpoint, no hardware.

The gate-off case compares against a baseline captured from the tree **before** the gate
existed. Reproduce it with the pre-change source, then re-run this script::

    .venv/bin/python scripts/experiments/x2_tennis_visibility/capture_baseline.py /tmp/vis_baseline_before_cpu.pt cpu
    .venv/bin/python scripts/experiments/x2_tennis_visibility/capture_baseline.py /tmp/vis_baseline_before.pt cuda:0
    .venv/bin/python scripts/experiments/x2_tennis_visibility/check_visibility_gate.py

If those two files are lost the case fails loudly instead of passing quietly.

**Device note, and it applies to every future probe in this repository.** ``cuda:0``
physics is **not** bit-reproducible across processes: two runs of the unmodified S9c
probe (8 envs, seed 11, 41 recorded frames of 40 zero-action steps) differ in 1081 of
3936 ball-history values by up to 2.48e-4, starting at step 10. Exact comparisons
therefore run on ``cpu``, where the same probe is bit-exact across processes. The
``cuda:0`` comparison below is a physics-noise control only and is **not** an invariance
measure.

Two baselines must have been captured before the gate existed:

* ``/tmp/vis_baseline_before.pt`` - ``cuda:0``, actor ``obs[:, 100:112]``, ``(41, 8,
  12)``: the reset frame and 40 zero-action steps of the S9c task at 8 envs, seed 11.
* ``/tmp/vis_baseline_before_cpu.pt`` - the same probe on ``cpu``.

Usage::

    .venv/bin/python \\
        scripts/experiments/x2_tennis_visibility/check_visibility_gate.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_VISIBILITY_DROPOUT_PROB,
  BALL_VISIBLE_HALF_ANGLES_RAD,
  BALL_VISIBLE_MAX_RANGE_M,
  BALL_VISIBLE_MIN_RANGE_M,
  CAMERA_MOUNT_QUAT,
  CAMERA_MOUNT_XYZ,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_env_cfg import BALL_ENTITY
from mjlab.tasks.velocity.config.agibot_x2.tennis_juggle_stage_env_cfg import (
  agibot_x2_tennis_juggle_stage_env_cfg,
)
from mjlab.tasks.velocity.mdp.ball_visibility import ball_in_camera_region
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)
from mjlab.utils.torch import configure_torch_backends

STAGE = "s9c-pose-quad"
"""The S9c rung the baselines were captured from."""

BALL_SLICE = slice(100, 112)
"""The actor's twelve ball-history numbers: the ``ball_hist_pelvis`` term is last."""

PROBE_ENVS = 8
PROBE_SEED = 11
PROBE_STEPS = 40

BASELINE_CUDA = Path("/tmp/vis_baseline_before.pt")
BASELINE_CPU = Path("/tmp/vis_baseline_before_cpu.pt")

PRE_CHANGE_REPEAT_RUN = (1081, 3936, 2.48e-4)
"""Two runs of the **unmodified** probe on cuda:0: differing values, values, max |diff|.

Measured 2026-11-07 before any edit. It is the floor any cuda:0 comparison is judged
against: a physics-noise control, not an invariance measure.
"""

STEP_DT = 0.02
"""The tennis control step (20 substeps of 1 ms), which the scripted term reads."""

PELVIS_BODY = 0
HEAD_BODY = 1

PASS = "PASS"
FAIL = "FAIL"


##
# Helpers.
##


def _diff(got: torch.Tensor, want: torch.Tensor) -> tuple[int, float]:
  """Number of differing values and the largest absolute difference."""
  if got.shape != want.shape:
    return (max(got.numel(), want.numel()), float("inf"))
  return (int((got != want).sum()), float((got - want).abs().max()))


def _all_zero(tensor: torch.Tensor) -> bool:
  return bool((tensor == 0.0).all())


def _f(values: torch.Tensor) -> str:
  return "[" + ", ".join(f"{v:.6g}" for v in values.flatten().tolist()) + "]"


def _head_pose_for_camera(
  cam_pos_w: torch.Tensor, cam_quat_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Head pose that puts the camera at the requested world pose.

  The mount transform is inverted, not bypassed, so the constants the term receives are
  the ones exercised here.
  """
  mount_xyz = torch.tensor(CAMERA_MOUNT_XYZ).expand_as(cam_pos_w)
  mount_quat = torch.tensor(CAMERA_MOUNT_QUAT).expand_as(cam_quat_w)
  head_quat_w = quat_mul(cam_quat_w, quat_conjugate(mount_quat))
  head_pos_w = cam_pos_w - quat_apply(head_quat_w, mount_xyz)
  return head_pos_w, head_quat_w


def _ball_in_optical_frame(
  cam_pos_w: torch.Tensor, theta_x: float, theta_y: float, range_m: float
) -> torch.Tensor:
  """Ball position at a chosen horizontal angle, vertical angle and range, metres.

  The optical axes are +x right, +y down, +z forward, so ``(tan(theta_x),
  tan(theta_y), 1)`` has exactly those two angles to the axis at every range.
  """
  direction = torch.tensor([math.tan(theta_x), math.tan(theta_y), 1.0])
  direction = direction / torch.linalg.vector_norm(direction)
  return cam_pos_w + direction.expand_as(cam_pos_w) * range_m


def _geometry(
  cam_pos_w: torch.Tensor, cam_quat_w: torch.Tensor, ball_pos_w: torch.Tensor
) -> tuple[float, float, float]:
  """Optical-frame horizontal angle, vertical angle and straight-line range."""
  delta = quat_apply_inverse(cam_quat_w, ball_pos_w - cam_pos_w)[0]
  return (
    math.atan2(float(delta[0]), float(delta[2])),
    math.atan2(float(delta[1]), float(delta[2])),
    float(torch.linalg.vector_norm(delta)),
  )


class _ScriptedEnv:
  """A scripted pelvis, head and ball, with no physics.

  The gate's cases are geometry and schedule claims, so a scripted pose is enough and it
  keeps them off the physics path. Body 0 is the pelvis (the term's reference frame) and
  body 1 is the head that carries the camera.
  """

  def __init__(self, num_envs: int) -> None:
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = STEP_DT
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self._pelvis_pos = torch.zeros(num_envs, 3)
    self._pelvis_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_envs, 4).clone()
    self._ball_pos = torch.zeros(num_envs, 3)
    self._head_pos = torch.zeros(num_envs, 3)
    self._head_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_envs, 4).clone()
    self.scene = {
      "robot": SimpleNamespace(
        data=SimpleNamespace(body_link_pos_w=None, body_link_quat_w=None)
      ),
      BALL_ENTITY: SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=self._ball_pos)
      ),
    }
    self.set_camera(torch.zeros(num_envs, 3), self._head_quat.clone())
    self.set_ball(self._ball_pos.clone())

  def set_camera(self, cam_pos_w: torch.Tensor, cam_quat_w: torch.Tensor) -> None:
    self._head_pos, self._head_quat = _head_pose_for_camera(cam_pos_w, cam_quat_w)
    self._refresh()

  def set_ball(self, ball_pos_w: torch.Tensor) -> None:
    self._ball_pos = ball_pos_w.clone()
    self._refresh()

  def _refresh(self) -> None:
    self.scene["robot"].data.body_link_pos_w = torch.stack(
      [self._pelvis_pos, self._head_pos], dim=1
    )
    self.scene["robot"].data.body_link_quat_w = torch.stack(
      [self._pelvis_quat, self._head_quat], dim=1
    )
    self.scene[BALL_ENTITY].data.root_link_pos_w = self._ball_pos


def _scripted_params(
  *,
  gate: bool,
  latency_s: tuple[float, float],
  rate_hz: tuple[float, float],
) -> dict:
  """Term parameters for the scripted cases: exact samples, no noise and no stalls."""
  params = {
    "ball_cfg": SceneEntityCfg(BALL_ENTITY),
    "pelvis_cfg": SceneEntityCfg("robot", body_ids=[PELVIS_BODY]),
    "reference_frame": "pelvis",
    "bias_half_width_m": 0.0,
    "position_noise_m": (0.0, 0.0, 0.0),
    "stream_rate_hz": rate_hz,
    "stream_latency_s": latency_s,
    "stall_prob": 0.0,
    "stall_range_s": (0.0, 0.0),
  }
  if gate:
    params.update(
      visibility_head_cfg=SceneEntityCfg("robot", body_ids=[HEAD_BODY]),
      visibility_mount_xyz=CAMERA_MOUNT_XYZ,
      visibility_mount_quat=CAMERA_MOUNT_QUAT,
      visibility_half_angles=BALL_VISIBLE_HALF_ANGLES_RAD,
      visibility_min_range_m=BALL_VISIBLE_MIN_RANGE_M,
      visibility_max_range_m=BALL_VISIBLE_MAX_RANGE_M,
      visibility_dropout_prob=BALL_VISIBILITY_DROPOUT_PROB,
    )
  return params


def _scripted_run(
  ball_positions: list[torch.Tensor],
  *,
  gate: bool = True,
  latency_s: tuple[float, float] = (0.0, 0.0),
  rate_hz: tuple[float, float] = (50.0, 50.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Drive the term over a scripted ball path.

  Args:
    ball_positions: One ``(num_envs, 3)`` world position per control step, step 0
      included.

  Returns:
    ``(outputs, counts, visible)``: the 12 numbers per step, the history count per step,
    and the visibility ring entry written at each step (``None`` when the gate is off).
  """
  env = _ScriptedEnv(int(ball_positions[0].shape[0]))
  params = _scripted_params(gate=gate, latency_s=latency_s, rate_hz=rate_hz)
  term = mdp.ball_history_samples(
    ObservationTermCfg(func=mdp.ball_history_samples, params=params), env
  )
  outs, counts, visible = [], [], []
  for position in ball_positions:
    env.set_ball(position)
    env.episode_length_buf += 1
    outs.append(term(env, **params).clone())
    counts.append(term._hist_count.clone())
    if term._vis_ring is not None:
      slot = (int(term._step[0].item()) - 1) % term._ring_len
      visible.append(term._vis_ring[:, slot].clone())
  return (
    torch.stack(outs),
    torch.stack(counts),
    torch.stack(visible) if visible else None,
  )


def _task_probe(gate: bool, device: str) -> dict:
  """The S9c baseline probe: 8 envs, seed 11, reset, then 40 zero-action steps.

  Returns the 12 ball-history numbers (reset frame plus every step), the visibility ring
  entry written at each frame, and the number of history slots filled per frame.
  """
  if gate:
    os.environ["MJLAB_JUGGLE_VISIBILITY_GATE"] = "1"
  else:
    os.environ.pop("MJLAB_JUGGLE_VISIBILITY_GATE", None)
  cfg = agibot_x2_tennis_juggle_stage_env_cfg(stage=STAGE)
  cfg.scene.num_envs = PROBE_ENVS
  cfg.seed = PROBE_SEED
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  term = env.observation_manager.get_term_cfg("actor", "ball_hist_pelvis").func
  visible: list[torch.Tensor] = []
  counts: list[torch.Tensor] = []

  def record_frame(steps_taken: int) -> None:
    counts.append(term._hist_count.clone().cpu())
    if term._vis_ring is not None:
      slot = steps_taken % term._ring_len
      visible.append(term._vis_ring[:, slot].clone().cpu())

  try:
    obs, _ = env.reset()
    records = [obs["actor"][:, BALL_SLICE].cpu().clone()]
    record_frame(0)
    action = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim),
      device=env.device,
      dtype=torch.float32,
    )
    for step in range(1, PROBE_STEPS + 1):
      obs, _, _, _, _ = env.step(action)
      records.append(obs["actor"][:, BALL_SLICE].cpu().clone())
      record_frame(step)
  finally:
    env.close()
  return {
    "obs": torch.stack(records),
    "visible": torch.stack(visible) if visible else None,
    "counts": torch.stack(counts),
    "device": device,
  }


##
# Cases.
##


def case_gate_off_identity() -> bool:
  """Case 1: with the gate off, the stream is the pre-change stream."""
  print("[1] gate off: identical to the pre-change probe")
  ok = True
  for path, label in ((BASELINE_CPU, "cpu"), (BASELINE_CUDA, "cuda:0")):
    if not path.is_file():
      print(f"  {label}: MISSING baseline {path}")
      ok = False
      continue
    want = torch.load(path)
    got = _task_probe(gate=False, device=label)["obs"]
    differing, worst = _diff(got, want)
    if label == "cpu":
      verdict = "exact match" if differing == 0 else "NOT a match"
      print(
        f"  cpu    invariance measure: differing {differing} of {want.numel()}, "
        f"max |diff| {worst:.3e} -> {verdict}"
      )
      ok = ok and differing == 0
    else:
      floor_n, floor_total, floor_max = PRE_CHANGE_REPEAT_RUN
      print(
        f"  cuda:0 physics-noise control, not an invariance measure: differing "
        f"{differing} of {want.numel()}, max |diff| {worst:.3e}"
      )
      print(
        f"         pre-change repeat-run floor: {floor_n} of {floor_total}, "
        f"max |diff| {floor_max:.3e} (two runs of the unmodified probe)"
      )
  return ok


def case_unseen_reset() -> bool:
  """Case 2: nothing published yet reads as all zeros, not a pelvis-origin sample."""
  print("[2] unseen reset: all twelve zeros, no fabricated pelvis-origin sample")
  behind = torch.zeros(1, 3) + torch.tensor([[0.0, 0.0, -0.5]])
  positions = [behind.clone() for _ in range(5)]
  outs, counts, visible = _scripted_run(positions)
  ungated, _, _ = _scripted_run(positions, gate=False)
  print(
    f"  ball 0.5 m behind the camera, visible per step: {visible.flatten().tolist()}"
  )
  for step in range(5):
    print(f"  step {step}: gate on {_f(outs[step])}  count {int(counts[step][0])}")
  print("  gate off, same script (a position the camera could not measure):")
  print(f"    step 0 {_f(ungated[0])}")
  print(f"    step 4 {_f(ungated[4])}")
  ok = _all_zero(outs) and int(counts.max()) == 0
  print(f"  -> {PASS if ok else FAIL}: all outputs zero and nothing admitted")
  return ok


def case_off_axis_70() -> bool:
  """Case 3: 70 degrees off-axis is outside the delivered image."""
  print("[3] 70 deg off-axis: never admitted")
  camera, cam_quat = torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  ball = _ball_in_optical_frame(camera, math.radians(70.0), 0.0, 0.6)
  theta_x, theta_y, range_m = _geometry(camera, cam_quat, ball)
  outs, counts, visible = _scripted_run([ball.clone() for _ in range(10)])
  print(
    f"  theta_x {math.degrees(theta_x):.1f} deg (right limit "
    f"{math.degrees(BALL_VISIBLE_HALF_ANGLES_RAD[1]):.1f}), theta_y "
    f"{math.degrees(theta_y):.1f} deg, range {range_m:.3f} m"
  )
  print(f"  visible every step: {visible.flatten().tolist()}")
  print(f"  last output {_f(outs[-1])}, admitted slots {int(counts.max())}")
  ok = (not bool(visible.any())) and _all_zero(outs) and int(counts.max()) == 0
  print(f"  -> {PASS if ok else FAIL}: invisible, nothing admitted, output stays zero")
  return ok


def case_off_axis_40() -> bool:
  """Case 4: 40 degrees off-axis at 0.6 m is inside the region."""
  print("[4] 40 deg off-axis at 0.6 m: admitted")
  camera, cam_quat = torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  ball = _ball_in_optical_frame(camera, math.radians(40.0), 0.0, 0.6)
  theta_x, theta_y, range_m = _geometry(camera, cam_quat, ball)
  outs, counts, visible = _scripted_run([ball.clone() for _ in range(5)])
  print(
    f"  theta_x {math.degrees(theta_x):.1f} deg, theta_y {math.degrees(theta_y):.1f} deg, "
    f"range {range_m:.3f} m"
  )
  print(f"  visible every step: {visible.flatten().tolist()}")
  print(f"  step 0 output {_f(outs[0])}")
  print(f"  admitted slots after 5 steps: {int(counts[-1][0])}")
  position_ok = bool(torch.allclose(outs[0][0, 0:3], ball[0], atol=1e-6))
  age_ok = float(outs[0][0, 3]) == 0.0
  ok = bool(visible.all()) and position_ok and age_ok and int(counts[-1][0]) == 3
  print(
    f"  -> {PASS if ok else FAIL}: admitted with its capture position and a zero age"
  )
  return ok


def case_max_range_sphere() -> bool:
  """Case 5: 0.7 m forward but 1.2 m from the camera centre is outside the sphere."""
  print("[5] 0.7 m forward, 1.2 m from the camera centre: rejected by the sphere")
  camera, cam_quat = torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  lateral = math.sqrt(1.2**2 - 0.7**2)
  ball = camera + torch.tensor([[lateral, 0.0, 0.7]])
  theta_x, theta_y, range_m = _geometry(camera, cam_quat, ball)
  env = _ScriptedEnv(1)
  env.set_camera(camera, cam_quat)
  env.set_ball(ball)
  head_cfg = SceneEntityCfg("robot", body_ids=[HEAD_BODY])
  ball_cfg = SceneEntityCfg(BALL_ENTITY)
  gate = ball_in_camera_region(
    env,
    ball_cfg,
    head_cfg,
    CAMERA_MOUNT_XYZ,
    CAMERA_MOUNT_QUAT,
    BALL_VISIBLE_HALF_ANGLES_RAD,
    BALL_VISIBLE_MIN_RANGE_M,
    BALL_VISIBLE_MAX_RANGE_M,
  )
  frame_only = ball_in_camera_region(
    env,
    ball_cfg,
    head_cfg,
    CAMERA_MOUNT_XYZ,
    CAMERA_MOUNT_QUAT,
    BALL_VISIBLE_HALF_ANGLES_RAD,
    BALL_VISIBLE_MIN_RANGE_M,
    float("inf"),
  )
  outs, counts, visible = _scripted_run([ball.clone() for _ in range(10)])
  print(
    f"  theta_x {math.degrees(theta_x):.1f} deg (inside the "
    f"{math.degrees(BALL_VISIBLE_HALF_ANGLES_RAD[1]):.1f} deg right limit), "
    f"range {range_m:.4f} m (limit {BALL_VISIBLE_MAX_RANGE_M:.1f} m)"
  )
  print(
    f"  frame test alone: {bool(frame_only[0])}   frame test plus sphere: "
    f"{bool(gate[0])}"
  )
  print(f"  last output {_f(outs[-1])}, admitted slots {int(counts.max())}")
  ok = (
    bool(frame_only[0])
    and not bool(gate[0])
    and (not bool(visible.any()))
    and _all_zero(outs)
    and int(counts.max()) == 0
  )
  print(f"  -> {PASS if ok else FAIL}: the range gate, not the angles, rejects it")
  return ok


def case_capture_visible_then_left_view() -> bool:
  """Case 6: a frame visible at capture is admitted after the ball has left the view."""
  print("[6] captured visible, ball leaves the view before the sample arrives")
  camera = torch.zeros(1, 3)
  visible_ball = camera + torch.tensor([[0.0, 0.0, 0.6]])
  hidden_ball = camera + torch.tensor([[0.0, 0.0, -0.5]])
  # 5-step latency, one acquisition attempt per step, visible only at step 5: the frame
  # captured at step 5 can first be served at step 10, and the ball is out of view from
  # step 6 on.
  positions = [hidden_ball.clone() for _ in range(12)]
  positions[5] = visible_ball.clone()
  outs, counts, visible = _scripted_run(
    positions, latency_s=(0.1, 0.1), rate_hz=(50.0, 50.0)
  )
  admitted_at = [
    step
    for step in range(1, len(counts))
    if int(counts[step][0]) > int(counts[step - 1][0])
  ]
  print(f"  visible per step:      {[int(v) for v in visible.flatten().tolist()]}")
  print(f"  admitted slots:        {[int(c) for c in counts.flatten().tolist()]}")
  if admitted_at:
    print(
      f"  first admission at step {admitted_at[0]}, capture age "
      f"{float(outs[admitted_at[0]][0, 3]):.3f} s"
    )
    print(f"  output at that step {_f(outs[admitted_at[0]])}")
  else:
    print("  no admission")
  capture_age_ok = (
    bool(admitted_at) and abs(float(outs[admitted_at[0]][0, 3]) - 0.1) < 1e-6
  )
  capture_pos_ok = bool(admitted_at) and bool(
    torch.allclose(outs[admitted_at[0]][0, 0:3], visible_ball[0], atol=1e-6)
  )
  one_sample = bool(admitted_at) and int(counts[-1][0]) == 1
  ok = capture_age_ok and capture_pos_ok and one_sample
  print(f"  -> {PASS if ok else FAIL}: one sample, served with its 5-step capture age")
  return ok


def case_long_gap() -> bool:
  """Case 7: a refused stream is a growing age, not a frozen or fresh sample."""
  print("[7] long gap: ages increase monotonically, slot 0 exceeds one period")
  camera = torch.zeros(1, 3)
  visible_ball = camera + torch.tensor([[0.0, 0.0, 0.6]])
  hidden_ball = camera + torch.tensor([[0.0, 0.0, -0.5]])
  positions = [hidden_ball.clone() for _ in range(40)]
  positions[0] = visible_ball.clone()
  # 10 Hz against one attempt per step with 0.1 s latency: the frames the later
  # acquisitions need are all invisible, so the stream falls silent after step 5.
  outs, counts, visible = _scripted_run(
    positions, latency_s=(0.1, 0.1), rate_hz=(10.0, 10.0)
  )
  ages = outs[:, 0, 3]
  monotonic = bool((ages[1:] >= ages[:-1]).all())
  period_s = 1.0 / 10.0
  print(
    f"  slot 0 age at steps 0/5/10/20/39: "
    f"{[round(float(ages[s]), 3) for s in (0, 5, 10, 20, 39)]}"
  )
  print(
    f"  monotonic: {monotonic}, max age {float(ages.max()):.3f} s, one period "
    f"{period_s:.3f} s"
  )
  print(
    f"  admitted slots at step 0/5/10/39: {[int(counts[s][0]) for s in (0, 5, 10, 39)]}"
  )
  ok = monotonic and float(ages.max()) >= period_s - 1e-6 and int(counts[10][0]) == 2
  print(f"  -> {PASS if ok else FAIL}: the sample held, and its age reached one period")
  return ok


def case_two_envs_independent() -> bool:
  """Case 8: one environment's visibility says nothing about another's."""
  print("[8] two envs, one visible and one not: independent streams")
  positions = [torch.tensor([[0.0, 0.0, 0.6], [0.0, 0.0, -0.5]]) for _ in range(10)]
  outs, counts, visible = _scripted_run(positions)
  print(f"  visible per env: {visible[0].tolist()}")
  print(f"  admitted slots at step 9: {counts[-1].tolist()}")
  print(f"  env 0 output at step 9: {_f(outs[-1][0])}")
  print(f"  env 1 output at step 9: {_f(outs[-1][1])}")
  print(f"  env 1 slot 0 age at step 9: {float(outs[-1][1, 3]):.3f} s")
  ok = (
    int(counts[-1][0]) == 3
    and int(counts[-1][1]) == 0
    and _all_zero(outs[:, 1])
    and float(outs[-1][0, 3]) == 0.0
  )
  print(f"  -> {PASS if ok else FAIL}: env 0 keeps acquiring, env 1 holds at zero")
  return ok


def case_gate_on_admitted_path() -> bool:
  """Case 9: with every frame visible, the gate admits exactly what the gate-off path
  admits, so the new branch does not perturb an admitted sample."""
  print("[9] gate on, every frame visible: identical to the gate-off cpu baseline")
  if not BASELINE_CPU.is_file():
    print(f"  MISSING baseline {BASELINE_CPU}")
    return False
  want = torch.load(BASELINE_CPU)
  probe = _task_probe(gate=True, device="cpu")
  got, visible = probe["obs"], probe["visible"]
  if visible is None:
    print("  FAIL: the gate is not installed; no visibility ring exists")
    return False
  # The stream can only diverge from the gate-off baseline once a captured frame is
  # refused, so the window is every frame up to the first invisible one.
  invisible = (~visible).nonzero()
  last_visible = int(invisible[0][0]) - 1 if len(invisible) else PROBE_STEPS
  window = last_visible + 1
  if window <= 0:
    # Nothing was visible, so there is no admitted path to compare. Report it as a
    # failure rather than diffing empty tensors, which raises.
    print("  FAIL: no frame was visible at step 0; the case cannot be evaluated")
    return False
  differing, worst = _diff(got[:window], want[:window])
  print(
    f"  window: frames 0..{last_visible} ({window} frames x {PROBE_ENVS} envs), "
    f"visible frames {int(visible[:window].sum())} of {visible[:window].numel()}"
  )
  print(
    f"  window differing {differing} of {want[:window].numel()}, max |diff| "
    f"{worst:.3e} -> {'exact match' if differing == 0 else 'NOT a match'}"
  )
  total, total_worst = _diff(got, want)
  print(
    f"  whole probe differing {total} of {want.numel()}, max |diff| {total_worst:.3e}; "
    f"first invisible (step, env) {invisible[0].tolist() if len(invisible) else 'none'}"
  )
  ok = differing == 0 and window > 1
  print(
    f"  -> {PASS if ok else FAIL}: the admitted path changes nothing, and the gate "
    f"starts refusing only where a frame leaves the region"
  )
  return ok


CASES = (
  case_gate_off_identity,
  case_unseen_reset,
  case_off_axis_70,
  case_off_axis_40,
  case_max_range_sphere,
  case_capture_visible_then_left_view,
  case_long_gap,
  case_two_envs_independent,
  case_gate_on_admitted_path,
)


def main() -> int:
  configure_torch_backends()
  torch.set_grad_enabled(False)
  print(
    f"ball visibility gate checks; stage {STAGE}, probe {PROBE_ENVS} envs, "
    f"seed {PROBE_SEED}, {PROBE_STEPS} zero-action steps"
  )
  print(
    f"half-angles {BALL_VISIBLE_HALF_ANGLES_RAD} rad, range "
    f"[{BALL_VISIBLE_MIN_RANGE_M}, {BALL_VISIBLE_MAX_RANGE_M}] m, dropout "
    f"{BALL_VISIBILITY_DROPOUT_PROB}"
  )
  print()
  results = [(case.__name__, case()) for case in CASES]
  print()
  print("summary")
  for name, ok in results:
    print(f"  {PASS if ok else FAIL}  {name}")
  failed = [name for name, ok in results if not ok]
  print()
  print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}")
  return 0 if not failed else 1


if __name__ == "__main__":
  raise SystemExit(main())
