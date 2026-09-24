"""Shared utilities for ONNX policy export across RL tasks."""

import mujoco
import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction


def list_to_csv_str(
  arr, *, decimals: int = 3, delimiter: str = ",", sub_delimiter: str = ";"
) -> str:
  """Convert list to CSV string with specified decimal precision.

  Elements that are themselves sequences (e.g. a per-dimension scale or a
  [min, max] clip range) are joined with `sub_delimiter` instead of being
  `str()`-formatted, which would otherwise embed a second, ambiguous set of
  commas inside the top-level comma-delimited string.
  """
  fmt = f"{{:.{decimals}f}}"

  def format_scalar(x) -> str:
    return fmt.format(x) if isinstance(x, (int, float)) else str(x)

  def format_entry(x) -> str:
    if isinstance(x, (list, tuple)):
      return sub_delimiter.join(format_scalar(v) for v in x)
    return format_scalar(x)

  return delimiter.join(format_entry(x) for x in arr)


def resolve_site_sensor_frame(
  mj_model: mujoco.MjModel, sensor_name: str
) -> tuple[str, str]:
  """Resolve a site-backed builtin sensor to its ``(site, parent body)`` names.

  A gyro/velocimeter/accelerometer reads the frame of a site rigidly attached to
  a body, so the site names the physical location and the parent body names the
  frame its reading is expressed in. Names are returned exactly as they appear
  in the compiled model. Returns ``("", "")`` for a missing sensor or one that
  is not a site sensor (e.g. a joint or subtree sensor), so a caller never
  claims a frame it did not resolve.
  """
  sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
  if sensor_id < 0:
    return "", ""
  if mj_model.sensor_objtype[sensor_id] != mujoco.mjtObj.mjOBJ_SITE:
    return "", ""
  site_id = int(mj_model.sensor_objid[sensor_id])
  site_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
  body_id = int(mj_model.site_bodyid[site_id])
  body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
  return site_name, body_name


def get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Get base metadata common to all RL policy exports.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.

  Returns:
    Dictionary of metadata fields that are common across all tasks.
  """
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  # Build mapping from joint name to actuator ID for natural joint order.
  # Each spec actuator controls exactly one joint (via its target field).
  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  # Get actuator IDs in natural joint order (same order as robot.joint_names).
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names  # global joint order
    if jname in joint_name_to_ctrl_id  # skip non-actuated joints
  ]
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  observation_term_scale: list = []
  observation_term_flatten_history_dim: list = []
  observation_term_history_length: list = []
  observation_term_clip: list = []
  # Sensor-backed actor terms (e.g. ``base_ang_vel``) additionally record which
  # builtin sensor they read and, for site sensors, the site and its parent
  # body. That makes the IMU body/site and the angular-velocity frame explicit
  # in the export instead of inferring it from the term name.
  observation_term_sensor_name: list = []
  observation_term_sensor_site: list = []
  observation_term_sensor_body: list = []
  observation_names = env.observation_manager.active_terms["actor"]

  # Compiled body/site names carry the entity prefix (``robot/torso_link``); the
  # rest of the metadata (``anchor_body_name``, ``body_names``) uses the bare
  # names, so strip it here too.
  entity_prefixes = tuple(f"{name}/" for name in env.scene.entities)

  def unqualified(mj_name: str) -> str:
    for prefix in entity_prefixes:
      if mj_name.startswith(prefix):
        return mj_name[len(prefix) :]
    return mj_name

  for active_term in observation_names:
    cfg = env.observation_manager.get_term_cfg("actor", active_term)

    sensor_name = cfg.params.get("sensor_name")
    if isinstance(sensor_name, str):
      site, body = resolve_site_sensor_frame(env.sim.mj_model, sensor_name)
      observation_term_sensor_name.append(sensor_name)
      observation_term_sensor_site.append(unqualified(site))
      observation_term_sensor_body.append(unqualified(body))
    else:
      observation_term_sensor_name.append("")
      observation_term_sensor_site.append("")
      observation_term_sensor_body.append("")

    if cfg.scale is None:
      observation_term_scale.append(1.0)
    else:
      raw_scale = cfg.scale
      scale = (
        raw_scale.cpu().tolist() if isinstance(raw_scale, torch.Tensor) else raw_scale
      )
      observation_term_scale.append(scale)

    raw_clip = cfg.clip
    if raw_clip is None:
      observation_term_clip.append([float("-inf"), float("inf")])
    else:
      observation_term_clip.append(list(raw_clip))

    observation_term_flatten_history_dim.append(cfg.flatten_history_dim)
    observation_term_history_length.append(cfg.history_length)

  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": observation_names,
    "observation_terms_scale": observation_term_scale,
    "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
    "observation_terms_history_length": observation_term_history_length,
    "observation_terms_clip": observation_term_clip,
    "observation_terms_sensor_name": observation_term_sensor_name,
    "observation_terms_sensor_site": observation_term_sensor_site,
    "observation_terms_sensor_body": observation_term_sensor_body,
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
  }


def attach_metadata_to_onnx(
  onnx_path: str, metadata: dict[str, list | str | float]
) -> None:
  """Attach metadata to an ONNX model file.

  Args:
    onnx_path: Path to the ONNX model file.
    metadata: Dictionary of metadata key-value pairs to attach.
  """
  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
