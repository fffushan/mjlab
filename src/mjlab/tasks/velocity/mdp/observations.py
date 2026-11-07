from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_HISTORY_AGE_CLIP_S,
  BALL_HISTORY_DEPTH,
  BALL_POSITION_NOISE_M,
  BALL_STREAM_LATENCY_RANGE_S,
  BALL_STREAM_RATE_RANGE_HZ,
  BALL_STREAM_STALL_PROB,
  BALL_STREAM_STALL_RANGE_S,
  BALL_TRUE_HISTORY_S,
  BALL_VISIBILITY_DROPOUT_PROB,
  BALL_VISIBLE_HALF_ANGLES_RAD,
  BALL_VISIBLE_MAX_RANGE_M,
  BALL_VISIBLE_MIN_RANGE_M,
  CAMERA_MOUNT_QUAT,
  CAMERA_MOUNT_XYZ,
  FACE_SITE_POS,
)
from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.ball_visibility import ball_in_camera_region
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


##
# Tennis ball, expressed in a body frame of the robot.
#
# These are X2-tennis specific and live here only because the observation is the
# primitive: the reward and termination terms import the same helper, so there is
# exactly one place where the frame convention is defined.
##


def _single_body_id(cfg: SceneEntityCfg) -> int:
  """The one body index an observation frame config must resolve to."""
  ids = cfg.body_ids
  if isinstance(ids, slice) or len(ids) != 1:
    raise ValueError(f"{cfg.name} must resolve to exactly one body, got {ids}")
  return ids[0]


def _face_site_pos(
  position_dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
  """``FACE_SITE_POS`` as a tensor on the caller's device, ready to broadcast."""
  return torch.tensor(FACE_SITE_POS, dtype=position_dtype, device=device)


def _plate_origin_and_quat(
  env: ManagerBasedRlEnv, plate_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """World position and orientation of the plate frame.

  The plate frame shares its orientation with the racket body, so this is just
  that body's pose.
  """
  plate: Entity = env.scene[plate_cfg.name]
  body_id = _single_body_id(plate_cfg)
  return (
    plate.data.body_link_pos_w[:, body_id],
    plate.data.body_link_quat_w[:, body_id],
  )


def plate_relative_ball_state(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Ball position and velocity relative to the plate, in the plate frame.

  Args:
    plate_cfg: The racket body inside the robot entity. Must resolve to one body.
    ball_cfg: The ball entity.

  Returns:
    ``(position, velocity)``, each shape ``(num_envs, 3)``. Position is measured
    from the centre of the plate's **top surface** to the ball's **centre**, so a
    ball resting on the strings reads ``z = BALL_RADIUS``. Callers that want the
    gap above the strings subtract the radius. Velocity is the ball's velocity
    relative to the plate, expressed in the plate frame.

  The plate is an axis-aligned ellipse in the racket body frame, so the plate
  frame is the racket body frame translated to ``FACE_SITE_POS``.
  """
  plate: Entity = env.scene[plate_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  plate_pos_w, plate_quat_w = _plate_origin_and_quat(env, plate_cfg)
  body_id = _single_body_id(plate_cfg)

  # Position: from the plate's top-surface centre to the ball's centre.
  arm_w = ball.data.root_link_pos_w - plate_pos_w
  position = quat_apply_inverse(plate_quat_w, arm_w) - _face_site_pos(
    position_dtype=arm_w.dtype, device=arm_w.device
  )

  # Velocity: relative to the *material point* of the plate at the ball, not to
  # the racket body origin. The face centre is 0.515 m from that origin, so a
  # 2 rad/s wrist turn moves it about 1 m/s faster than the origin; ignoring the
  # rotation would report a still ball as moving whenever the wrist turns.
  plate_point_vel_w = plate.data.body_link_lin_vel_w[:, body_id] + torch.cross(
    plate.data.body_link_ang_vel_w[:, body_id],
    arm_w,
    dim=1,
  )
  velocity = quat_apply_inverse(
    plate_quat_w, ball.data.root_link_lin_vel_w - plate_point_vel_w
  )
  return position, velocity


def ball_position_in_plate(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Ball position relative to the plate centre, in the plate frame."""
  position, _ = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  return position


def ball_velocity_in_plate(
  env: ManagerBasedRlEnv,
  plate_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Ball velocity relative to the plate, in the plate frame.

  A ball moving with the bat reads zero, which is what "held" means.
  """
  _, velocity = plate_relative_ball_state(env, plate_cfg, ball_cfg)
  return velocity


def ball_position_in_pelvis(
  env: ManagerBasedRlEnv,
  pelvis_cfg: SceneEntityCfg,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Ball position relative to the pelvis, in the pelvis frame.

  Measured from the pelvis body **origin**, rotated by the pelvis body axes:
  ``p_local = R_pelvis.T @ (p_ball_world - p_pelvis_world)``. A controller must
  use the same body origin, not the centre of mass.

  Args:
    pelvis_cfg: The pelvis body inside the robot entity. Must resolve to one body.
    ball_cfg: The ball entity.

  Returns:
    Tensor of shape ``(num_envs, 3)``.
  """
  robot: Entity = env.scene[pelvis_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  body_id = _single_body_id(pelvis_cfg)
  offset_w = ball.data.root_link_pos_w - robot.data.body_link_pos_w[:, body_id]
  return quat_apply_inverse(robot.data.body_link_quat_w[:, body_id], offset_w)


class ball_position_hold(ManagerTermBase):
  """Ball position in the plate frame, with rejected frames served from a hold.

  The real perception chain does not deliver a measurement on every control step:
  it publishes nothing when the colour detector finds no ball or more than one
  candidate, when it cannot fit a consistent depth, and nothing at all when the
  stream breaks. A deployed controller therefore repeats the last valid position
  until a new one arrives, and the policy must train against that behaviour or it
  will be fitted to a guaranteed-fresh measurement it never has.

  On each step the term either passes the fresh position through or, with
  probability ``rejection_prob``, starts a hold that repeats the last accepted
  value for a number of steps drawn from ``hold_steps``. The held value and the
  hold counter are per environment, so one rejected frame does not freeze another
  environment's measurement, and both are cleared on reset so a new episode cannot
  repeat the previous episode's position.

  The first measurement after a reset is **always** accepted: there is no older
  value to hold yet, and inventing one would feed the policy a fabricated
  position. For the same reason the step after a hold always delivers a fresh
  value. Without that refractory step a rejection probability of 1 would hold one
  value forever, so the refresh the hold model exists to bound could not happen.
  The cost is that an unlimited stream break is not representable: the longest
  stale run is ``hold_steps`` long.

  ``bias_half_width_m`` adds a **constant** offset per environment per episode,
  drawn once at reset and applied to every measurement of that episode. It models a
  biased measurement chain rather than a noisy one: the robot's camera-to-pelvis
  disagreement is a fixed 24.69 mm / 1.24 deg, which zero-mean noise cannot
  represent. The default of 0.0 reproduces the unbiased chain exactly, and the
  offset is drawn per episode so a policy cannot average it out within one.

  Args:
    plate_cfg: The racket body inside the robot entity. Must resolve to one body.
    ball_cfg: The ball entity.
    rejection_prob: Probability per control step that a frame is rejected.
    hold_steps: Inclusive ``(min, max)`` hold length in control steps.
    bias_half_width_m: Half-width of the constant per-episode offset, in metres on
      each plate axis. 0.0 disables it.

  Returns:
    Tensor of shape ``(num_envs, 3)``, the same shape and frame as
    :func:`ball_position_in_plate`.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    half = float(cfg.params.get("bias_half_width_m", 0.0) or 0.0)
    self._bias_half_width_m = half
    self._bias = self._draw_bias(env.num_envs, env.device, half)
    self._held = torch.zeros(env.num_envs, 3, device=env.device)
    self._steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._has_value = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._just_held = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    plate_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    rejection_prob: float,
    hold_steps: tuple[int, int],
    bias_half_width_m: float = 0.0,
  ) -> torch.Tensor:
    del bias_half_width_m  # Drawn once per episode in ``reset``/``__init__``.
    fresh = ball_position_in_plate(env, plate_cfg, ball_cfg) + self._bias
    # One draw per environment per step, so a rejection in one environment says
    # nothing about any other. An environment that is already holding consumes a
    # draw and ignores it.
    rejected = torch.rand(self.num_envs, device=fresh.device) < rejection_prob
    start = rejected & (self._steps_left == 0) & self._has_value & ~self._just_held
    low, high = hold_steps
    drawn = torch.randint(int(low), int(high) + 1, (self.num_envs,), device=env.device)
    self._steps_left = torch.where(start, drawn, self._steps_left)

    holding = self._steps_left > 0
    self._held = torch.where(holding[:, None], self._held, fresh)
    self._has_value = self._has_value | ~holding
    # The step that serves the last held value arms the refractory step.
    self._just_held = holding & (self._steps_left == 1)
    self._steps_left = (self._steps_left - 1).clamp(min=0)
    return torch.where(holding[:, None], self._held, fresh)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._held[env_ids] = 0.0
    self._steps_left[env_ids] = 0
    self._has_value[env_ids] = False
    self._just_held[env_ids] = False
    if self._bias_half_width_m > 0.0:
      n = self._bias[env_ids].shape[0]
      self._bias[env_ids] = self._draw_bias(
        n, self._bias.device, self._bias_half_width_m
      )

  @staticmethod
  def _draw_bias(n: int, device, half: float):
    if half <= 0.0:
      return torch.zeros(n, 3, device=device)
    return (torch.rand(n, 3, device=device) * 2.0 - 1.0) * half


class ball_history_samples(ManagerTermBase):
  """Three timestamped ball samples in a body frame, newest first.

  A measurement chain cannot deliver the ball's current position: it delivers a
  stream of older positions at an irregular rate. This term models that stream
  directly instead of applying a fixed delay to a fresh position. Each environment
  keeps a ring of its **true** ball positions, written every control step in the
  chosen reference frame, and samples that ring on its own acquisition schedule. An
  acquisition draws a latency, reads the true position ``latency`` seconds back, adds
  the per-episode constant bias and per-sample noise, and pushes ``(position, step)``
  into a three-slot history. The output is the history flattened as ``x, y, z, age``
  per slot, newest first, where ``age`` is the time between the sample's capture step
  and the current control step, clipped at :data:`BALL_HISTORY_AGE_CLIP_S`.

  **Newest is by capture time, not by arrival time.** Latency jitter lets a frame
  arrive after one that was captured later than itself, so pushing on arrival alone
  would put that older frame in slot 0 and make the ages non-monotonic. The history
  instead keeps the three largest capture steps, so ``t0 <= t1 <= t2`` always holds.

  **Latency becomes an observable input.** The age of the newest sample is the
  stream's latency plus the time since it arrived, so the policy reads the latency
  from an input instead of being trained against one fixed delay model. That is what
  lets one policy cover the two robots whose estimators run at 18.44 Hz and 5.40 Hz
  with different latencies.

  **Three samples are the minimum that constrains curvature.** Three
  ``(position, age)`` pairs determine a parabola, so the policy can fit the ball's
  trajectory, extrapolate it ballistically to the current step, and reject a single
  outlier - a 2-3 m ghost from the real estimator - by its inconsistency with the
  other two samples.

  **A stream stall is just a growing age**, which is why the old hold hack
  (:class:`ball_position_hold`) is not used here. A stalled stream needs no special
  state: the same sample stays in the history and its age grows, which is exactly the
  information the policy needs. Fewer than ``BALL_HISTORY_DEPTH`` samples are
  available only at the start of an episode; the oldest available slot is repeated
  **with its own age**, so the duplicate is self-evident rather than a fabricated
  fresh measurement.

  **The visibility gate is off unless a head is given.** The real estimator publishes
  nothing for a frame in which the ball was not detected, so a simulator stream that
  always delivers a position lets the actor rely on a measurement the robot could not
  have made. When ``visibility_head_cfg`` is an entity config, the term:

  1. writes a boolean **visibility ring** entry per control step, from the camera pose
     at that step, using
     :func:`~mjlab.tasks.velocity.mdp.ball_visibility.ball_in_camera_region`;
  2. admits a due acquisition **only if** the frame at ``capture_step`` was visible,
     after the latency draw, so the draws of a seeded replay stay in the same order;
  3. inserts nothing when the frame was refused and retries on the next control step,
     so the history ages grow: that is the silence the estimator shows;
  4. emits **all twelve numbers zero** while nothing has been admitted, because there
     is no measurement to report, instead of a reset-origin position at a growing age;
  5. draws a dropout random number **only** when ``visibility_dropout_prob`` is
     positive, so a zero-dropout run consumes the same randomness as before.

  The gate is pure geometry plus that optional dropout. Range bias, occlusion and
  ghost rejection are not modelled.

  The term applies its own delay and its own noise, so its
  :class:`~mjlab.managers.observation_manager.ObservationTermCfg` must set neither
  ``noise`` nor any ``delay_*`` field. The noise is uniform with the per-axis
  half-widths ``position_noise_m``, which defaults to
  :data:`BALL_POSITION_NOISE_M`: the same placeholders and the same distribution
  the rest of the velocity task uses. A zero triple makes the samples exact.

  **The reference frame is chosen once, and a sample keeps the frame it was
  captured in.** ``reference_frame="plate"`` is the shipped behaviour and
  ``reference_frame="pelvis"`` stores the stream in the pelvis frame instead. No
  stored sample is ever re-expressed in a later frame: an old position is served
  exactly as it was measured, and only its age advances. This history does not
  compensate pelvis motion between samples.

  Args:
    ball_cfg: The ball entity.
    plate_cfg: The racket body inside the robot entity, used when
      ``reference_frame="plate"``. Must resolve to one body.
    pelvis_cfg: The pelvis body inside the robot entity, used when
      ``reference_frame="pelvis"``. Must resolve to one body.
    reference_frame: ``"plate"`` (default) or ``"pelvis"``.
    bias_half_width_m: Half-width of the constant per-episode offset, in metres on
      each frame axis. 0.0 disables it.
    position_noise_m: Per-axis half-widths of the per-sample position noise, in
      metres. Defaults to :data:`BALL_POSITION_NOISE_M`; ``(0.0, 0.0, 0.0)``
      disables it.
    visibility_head_cfg: The camera-carrying head body. ``None`` (default) leaves the
      gate off and the stream identical to the ungated one. Must resolve to one body
      when given.
    visibility_mount_xyz: Camera position in the head body frame, metres.
    visibility_mount_quat: Camera orientation in the head body frame, wxyz.
    visibility_half_angles: ``(left, right, up, down)`` half-angles of the delivered
      image, radians. Defaults to :data:`BALL_VISIBLE_HALF_ANGLES_RAD`.
    visibility_min_range_m: Near limit of the region, straight-line from the camera
      centre.
    visibility_max_range_m: Far limit of the region, also straight-line, so the
      region it closes is a sphere.
    visibility_dropout_prob: Probability that an otherwise-visible frame is marked
      invisible at write time. 0.0 (default) draws no random number.

  Returns:
    Tensor of shape ``(num_envs, 12)``: ``[x0, y0, z0, t0, x1, y1, z1, t1, x2, y2,
    z2, t2]`` in the reference frame, newest sample first.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._bias_half_width_m = float(cfg.params.get("bias_half_width_m", 0.0) or 0.0)
    noise_half_m = tuple(
      float(v) for v in cfg.params.get("position_noise_m", BALL_POSITION_NOISE_M)
    )
    self._step_dt = float(env.step_dt)
    # The ring must cover the longest latency plus the three-sample span at the
    # slowest stream rate, with margin. The epsilon keeps float dust in
    # 1.5 / 0.02 from adding a slot to the intended 75.
    self._ring_len = max(1, math.ceil(BALL_TRUE_HISTORY_S / self._step_dt - 1e-9))
    n = env.num_envs
    device = env.device
    self._bias = self._draw_bias(n, device, self._bias_half_width_m)
    self._ring = torch.zeros(n, self._ring_len, 3, device=device)
    self._hist_pos = torch.zeros(n, BALL_HISTORY_DEPTH, 3, device=device)
    self._hist_step = torch.zeros(
      n, BALL_HISTORY_DEPTH, dtype=torch.long, device=device
    )
    self._hist_count = torch.zeros(n, dtype=torch.long, device=device)
    # 0 means the first control step of the episode acquires immediately. That
    # guarantees a sample exists before any latency is drawn, and reset() restores
    # it for every new episode.
    self._next_acq = torch.zeros(n, dtype=torch.long, device=device)
    # The term counts its **own** control steps instead of reading
    # ``episode_length_buf``. rsl_rl seeds that buffer with randint(0,
    # max_episode_length) at the start of every learn(), so the first rollout would
    # otherwise age the samples by a random amount and read ring slots that this
    # episode never wrote - a fabricated ball at the plate origin with a plausible
    # age. The same trap produced phantom ball_not_lifted counts in the stage-1
    # terms; the rule there, applied here, is that elapsed time is counted per term.
    self._step = torch.zeros(n, dtype=torch.long, device=device)
    self._last_buf: torch.Tensor | None = None
    self._last_out: torch.Tensor | None = None
    self._noise_half = torch.tensor(noise_half_m, dtype=torch.float32, device=device)
    # Visibility gate. ``None`` means off, and then no ring exists, no geometry is
    # evaluated and no random number is drawn beyond the ungated stream's own draws.
    self._visibility_head_cfg: SceneEntityCfg | None = cfg.params.get(
      "visibility_head_cfg"
    )
    self._visibility_mount_xyz = tuple(
      cfg.params.get("visibility_mount_xyz", CAMERA_MOUNT_XYZ)
    )
    self._visibility_mount_quat = tuple(
      cfg.params.get("visibility_mount_quat", CAMERA_MOUNT_QUAT)
    )
    self._visibility_half_angles = tuple(
      cfg.params.get("visibility_half_angles", BALL_VISIBLE_HALF_ANGLES_RAD)
    )
    self._visibility_min_range_m = float(
      cfg.params.get("visibility_min_range_m", BALL_VISIBLE_MIN_RANGE_M)
    )
    self._visibility_max_range_m = float(
      cfg.params.get("visibility_max_range_m", BALL_VISIBLE_MAX_RANGE_M)
    )
    self._visibility_dropout_prob = float(
      cfg.params.get("visibility_dropout_prob", BALL_VISIBILITY_DROPOUT_PROB)
    )
    self._vis_ring: torch.Tensor | None = (
      torch.zeros(n, self._ring_len, dtype=torch.bool, device=device)
      if self._visibility_head_cfg is not None
      else None
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg | None = None,
    pelvis_cfg: SceneEntityCfg | None = None,
    reference_frame: str = "plate",
    bias_half_width_m: float = 0.0,
    position_noise_m: tuple[float, float, float] = BALL_POSITION_NOISE_M,
    stream_rate_hz: tuple[float, float] = BALL_STREAM_RATE_RANGE_HZ,
    stream_latency_s: tuple[float, float] = BALL_STREAM_LATENCY_RANGE_S,
    stall_prob: float = BALL_STREAM_STALL_PROB,
    stall_range_s: tuple[float, float] = BALL_STREAM_STALL_RANGE_S,
    visibility_head_cfg: SceneEntityCfg | None = None,
    visibility_mount_xyz: tuple[float, float, float] = CAMERA_MOUNT_XYZ,
    visibility_mount_quat: tuple[float, float, float, float] = CAMERA_MOUNT_QUAT,
    visibility_half_angles: tuple[float, float, float, float] = (
      BALL_VISIBLE_HALF_ANGLES_RAD
    ),
    visibility_min_range_m: float = BALL_VISIBLE_MIN_RANGE_M,
    visibility_max_range_m: float = BALL_VISIBLE_MAX_RANGE_M,
    visibility_dropout_prob: float = BALL_VISIBILITY_DROPOUT_PROB,
  ) -> torch.Tensor:
    del bias_half_width_m  # Drawn once per episode in ``reset``/``__init__``.
    del position_noise_m  # Resolved once per term in ``__init__``.
    del visibility_head_cfg  # Resolved once per term in ``__init__``.
    del visibility_mount_xyz
    del visibility_mount_quat
    del visibility_half_angles
    del visibility_min_range_m
    del visibility_max_range_m
    del visibility_dropout_prob
    # The stream magnitudes are parameters, not module globals, so an ablation arm can
    # vary them from the configuration and the value a run used is recorded in the
    # run's own params/env.yaml. They default to the measured constants.
    if not (0.0 < stream_rate_hz[0] <= stream_rate_hz[1]):
      raise ValueError(
        f"stream_rate_hz must be positive and ordered, got {stream_rate_hz}"
      )
    if not (0.0 <= stream_latency_s[0] <= stream_latency_s[1]):
      raise ValueError(
        f"stream_latency_s must be non-negative and ordered, got {stream_latency_s}"
      )
    device = self._ring.device

    # A call at the same control step is a re-entry: a subset reset computes the
    # observation for the whole batch, and the viewer refreshes the same way. Those
    # paths must not push a second sample, redraw the schedule, or advance the
    # clock, so return the previous output unchanged. ``episode_length_buf`` is used
    # only to recognise that case - never as a clock.
    # ``clone`` matters: on a CPU env ``.to(device).long()`` returns the buffer
    # itself, so caching it un-cloned would alias a tensor that the env increments in
    # place and the guard below would then match on every step and freeze the term at
    # its reset-time output.
    buf = env.episode_length_buf.to(device=device).long().clone()
    if (
      self._last_out is not None
      and self._last_buf is not None
      and torch.equal(buf, self._last_buf)
    ):
      return self._last_out

    now_step = self._step
    rows = torch.arange(self.num_envs, device=device)
    if reference_frame == "plate":
      if plate_cfg is None:
        raise ValueError("reference_frame='plate' needs plate_cfg")
      true_pos = ball_position_in_plate(env, plate_cfg, ball_cfg)
    elif reference_frame == "pelvis":
      if pelvis_cfg is None:
        raise ValueError("reference_frame='pelvis' needs pelvis_cfg")
      true_pos = ball_position_in_pelvis(env, pelvis_cfg, ball_cfg)
    else:
      raise ValueError(
        f"reference_frame must be 'plate' or 'pelvis', got {reference_frame!r}"
      )

    # Write the true position at this control step. Indexing by the step itself
    # makes a repeated call at the same step idempotent, and makes the ring read
    # below a direct lookup of a known step.
    self._ring[rows, now_step % self._ring_len] = true_pos
    if self._vis_ring is not None:
      # Whether the camera could see the ball at this step, from the camera pose at
      # this step. Written before the admission test, so the frame a sample is captured
      # from is always already in the ring - including the reset step's, which
      # ``reset`` clears for a new episode.
      head_cfg = self._visibility_head_cfg
      assert head_cfg is not None  # guarded by ``self._vis_ring`` above
      visible = ball_in_camera_region(
        env,
        ball_cfg,
        head_cfg,
        self._visibility_mount_xyz,
        self._visibility_mount_quat,
        self._visibility_half_angles,
        self._visibility_min_range_m,
        self._visibility_max_range_m,
      )
      if self._visibility_dropout_prob > 0.0:
        # Drawn only when the probability is positive: a zero-dropout run consumes
        # exactly the randomness it did before the gate existed.
        dropped = (
          torch.rand(self.num_envs, device=device) < self._visibility_dropout_prob
        )
        visible = visible & ~dropped
      self._vis_ring[rows, now_step % self._ring_len] = visible

    due = now_step >= self._next_acq

    # One draw per environment per step, so an acquisition in one environment says
    # nothing about any other. The values on a step with no acquisition are
    # consumed and ignored.
    latency = stream_latency_s[0] + torch.rand(self.num_envs, device=device) * (
      stream_latency_s[1] - stream_latency_s[0]
    )
    lag_steps = torch.round(latency / self._step_dt).long().clamp(min=0)
    # Before the episode has enough true history the latency cannot be honoured.
    # Clamping to the reset step keeps the sample a real state of this episode.
    capture_step = (now_step - lag_steps).clamp(min=0)
    # Admit the frame only if the camera could see the ball at its capture step. The
    # latency draw above happens whether or not the frame is admitted, so a seeded
    # replay consumes the same draws in the same order.
    admitted = (
      due
      if self._vis_ring is None
      else due & self._vis_ring[rows, capture_step % self._ring_len]
    )
    measured = self._ring[rows, capture_step % self._ring_len] + self._bias
    measured = measured + (torch.rand(self.num_envs, 3, device=device) * 2.0 - 1.0) * (
      self._noise_half
    )

    # Keep the three samples with the largest capture step. Sorting by capture time
    # rather than dropping the oldest arrival keeps ``t0 <= t1 <= t2`` when a late
    # frame is older than the one already in slot 0. The tie-break puts the new
    # sample first among equal steps: on the first step of an episode every step is
    # still the reset zero and the new sample must not be displaced by a zero slot.
    # ``torch.where`` keeps every environment not due this step frozen; a new tensor
    # is built rather than shifting with overlapping slices.
    all_steps = torch.cat([capture_step.unsqueeze(1), self._hist_step], dim=1)
    all_pos = torch.cat([measured.unsqueeze(1), self._hist_pos], dim=1)
    tie_break = torch.arange(
      BALL_HISTORY_DEPTH + 1, 0, -1, device=device, dtype=all_steps.dtype
    )
    keep = (all_steps * (BALL_HISTORY_DEPTH + 1) + tie_break).argsort(
      dim=1, descending=True
    )[:, :BALL_HISTORY_DEPTH]
    self._hist_pos = torch.where(
      admitted[:, None, None],
      all_pos.gather(1, keep[:, :, None].expand(-1, -1, 3)),
      self._hist_pos,
    )
    self._hist_step = torch.where(
      admitted[:, None], all_steps.gather(1, keep), self._hist_step
    )
    self._hist_count = torch.where(
      admitted,
      (self._hist_count + 1).clamp(max=BALL_HISTORY_DEPTH),
      self._hist_count,
    )

    # Schedule the next acquisition: one interval at the drawn stream rate, plus a
    # stall with the measured probability. The stall is a longer gap, not a separate
    # state; its only visible effect is the growing age of the held sample.
    rate = stream_rate_hz[0] + torch.rand(self.num_envs, device=device) * (
      stream_rate_hz[1] - stream_rate_hz[0]
    )
    interval_steps = torch.round((1.0 / rate) / self._step_dt).long().clamp(min=1)
    stall = torch.rand(self.num_envs, device=device) < stall_prob
    stall_s = stall_range_s[0] + torch.rand(self.num_envs, device=device) * (
      stall_range_s[1] - stall_range_s[0]
    )
    stall_steps = torch.round(stall_s / self._step_dt).long()
    next_acq = (
      now_step
      + interval_steps
      + torch.where(stall, stall_steps, torch.zeros_like(stall_steps))
    )
    self._next_acq = torch.where(due, next_acq, self._next_acq)
    if self._vis_ring is not None:
      # A refused frame inserts nothing and the next control step is due again, so the
      # gaps that follow are the silent stretches the estimator produces.
      self._next_acq = torch.where(due & ~admitted, now_step + 1, self._next_acq)

    # Repeat the oldest available slot in the unfilled slots, with that slot's own
    # age, so a duplicate cannot be mistaken for a fresh sample.
    slots = torch.arange(BALL_HISTORY_DEPTH, device=device)
    pick = torch.minimum(slots[None, :], (self._hist_count - 1).clamp(min=0)[:, None])
    positions = self._hist_pos.gather(1, pick[:, :, None].expand(-1, -1, 3))
    steps = self._hist_step.gather(1, pick)
    ages = (now_step[:, None] - steps).to(torch.float32) * self._step_dt
    ages = ages.clamp(min=0.0, max=BALL_HISTORY_AGE_CLIP_S)
    out = torch.cat([positions, ages.unsqueeze(-1)], dim=-1).reshape(self.num_envs, -1)
    if self._vis_ring is not None:
      # Nothing admitted yet: no measurement exists, so every number is zero. Without
      # this the unfilled slots would serve the reset zero position at a growing age,
      # which reads as a fresh sample at the pelvis origin.
      out = torch.where((self._hist_count == 0)[:, None], torch.zeros_like(out), out)
    self._step += 1
    self._last_buf = buf
    self._last_out = out
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._ring[env_ids] = 0.0
    if self._vis_ring is not None:
      # A cleared ring is what stops a previous episode's visibility from admitting a
      # sample in this one.
      self._vis_ring[env_ids] = False
    self._hist_pos[env_ids] = 0.0
    self._hist_step[env_ids] = 0
    self._hist_count[env_ids] = 0
    self._next_acq[env_ids] = 0
    self._step[env_ids] = 0
    # The cached output belongs to the step being left, so it cannot be served to a
    # new episode even if two resets happen at the same episode_length_buf value.
    self._last_out = None
    if self._bias_half_width_m > 0.0:
      n = self._bias[env_ids].shape[0]
      self._bias[env_ids] = self._draw_bias(
        n, self._bias.device, self._bias_half_width_m
      )

  @staticmethod
  def _draw_bias(n: int, device, half: float) -> torch.Tensor:
    if half <= 0.0:
      return torch.zeros(n, 3, device=device)
    return (torch.rand(n, 3, device=device) * 2.0 - 1.0) * half
