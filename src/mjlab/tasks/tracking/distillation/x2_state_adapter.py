"""Concrete fixed-topology X2/Warp branch-state adapter.

The adapter is intentionally source-named rather than reflective.  It inventories
one ManagerBasedRlEnv whose motion command is MultiMotionCommand, whose actions
are builtin joint-position actions, whose sensors are builtin/contact sensors,
and whose simulation exposes the Warp bridge fields listed by the P0 contract.
Unknown active state is rejected before any destination mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import isfunction
from typing import Any, Literal

import torch

from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sensor.builtin_sensor import BuiltinSensor
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.buffers import CircularBuffer, DelayBuffer

from .branch_state import (
  DYNAMIC_FIELDS,
  MODEL_FIELDS,
  BranchStateError,
  BranchStateMismatch,
  _check_finite,
  _owned,
  _tensor_value,
)
from .multi_motion import MultiMotionCommand
from .vae_config import ACTION_DIM, DEFAULT_SCHEMA, JOINT_DIM

Scope = Literal["world", "global", "cached"]


@dataclass(frozen=True, slots=True)
class FieldDescriptor:
  name: str
  path: tuple[str, ...]
  scope: Scope


DYNAMIC_TABLE = tuple(
  FieldDescriptor(name, ("sim", "data", name), "world") for name in DYNAMIC_FIELDS
)
MODEL_TABLE = tuple(
  FieldDescriptor(name, ("sim", "model", name), "global") for name in MODEL_FIELDS
)
CACHED_TABLE = (
  FieldDescriptor("sensordata", ("sim", "data", "sensordata"), "world"),
  FieldDescriptor("obs_buf", ("obs_buf",), "cached"),
)

_ENTITY_FIELDS = (
  "encoder_bias",
  "default_root_state",
  "default_joint_pos",
  "default_joint_vel",
  "default_joint_pos_limits",
  "joint_pos_limits",
  "soft_joint_pos_limits",
  "gravity_vec_w",
  "forward_vec_b",
  "joint_pos_target",
  "joint_vel_target",
  "joint_effort_target",
  "tendon_len_target",
  "tendon_vel_target",
  "tendon_effort_target",
  "site_effort_target",
)
_ACTION_MANAGER_FIELDS = ("_action", "_prev_action", "_prev_prev_action")
_ACTION_TERM_FIELDS = (
  "_raw_actions",
  "_processed_actions",
  "_target_ids",
  "_target_names",
  "_scale",
  "_offset",
)
_COMMAND_ROW_FIELDS = (
  "motion_ids",
  "time_steps",
  "_row_lengths",
  "_teacher_codes",
  "_row_lookahead_steps",
  "_row_bin_counts",
  "generation_ids",
  "segment_ids",
  "time_left",
  "command_counter",
  "body_pos_relative_w",
  "body_quat_relative_w",
)
_COMMAND_MEMBERSHIP_FIELDS = (
  "_pending_reset_frame_ids",
  "_reset_resample_ids",
  "_timer_resample_ids",
  "_wrap_resample_ids",
)
_ENV_ROW_FIELDS = ("episode_length_buf", "_manual_reset_pending", "_command_dt")


class X2FixedTopologyStateAdapter:
  """Capture and restore explicit X2 state at one source/destination row."""

  def __init__(self, env: Any, *, source_row: int | None) -> None:
    self.env, self.source_row = env, source_row
    self._validate_static_layout()

  @property
  def _num_envs(self) -> int:
    value = getattr(self.env, "num_envs", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
      raise BranchStateError("X2 adapter requires a positive integer env.num_envs")
    return value

  def _validate_static_layout(self) -> None:
    sim = getattr(self.env, "sim", None)
    if (
      sim is None
      or getattr(sim, "data", None) is None
      or getattr(sim, "model", None) is None
    ):
      raise BranchStateError("X2 adapter requires env.sim.data and env.sim.model")
    expanded = getattr(sim, "expanded_fields", None)
    if not isinstance(expanded, set):
      raise BranchStateError("X2 adapter requires explicit sim.expanded_fields")
    unknown = expanded - set(MODEL_FIELDS)
    if unknown:
      raise BranchStateError(f"unsupported expanded model fields: {sorted(unknown)}")
    scene = getattr(self.env, "scene", None)
    if scene is None or not callable(getattr(scene, "collect_variant_info", None)):
      raise BranchStateError("X2 adapter requires a concrete Scene variant inventory")
    if scene.collect_variant_info():
      raise BranchStateError("mesh/model variants are unsupported by P0 restore")
    sensors = getattr(scene, "sensors", None)
    if not isinstance(sensors, Mapping):
      raise BranchStateError("scene sensor inventory is missing")
    for sensor_name, sensor in sensors.items():
      if not isinstance(sensor, (BuiltinSensor, ContactSensor)):
        raise BranchStateError(
          f"unsupported active sensor {sensor_name!r}: {type(sensor).__name__}"
        )
    origins = _tensor_required(getattr(scene, "env_origins", None), "scene.env_origins")
    if origins.ndim != 2 or origins.shape != (self._num_envs, 3):
      raise BranchStateError("scene.env_origins must have shape [num_envs, 3]")
    if self.source_row is not None and not 0 <= self.source_row < self._num_envs:
      raise BranchStateError("source row is outside the environment batch")
    entities = getattr(scene, "entities", None)
    if not isinstance(entities, Mapping) or "robot" not in entities:
      raise BranchStateError("X2 adapter requires the fixed-topology robot entity")
    actuators = getattr(entities["robot"], "actuators", None)
    if not isinstance(actuators, Sequence) or not actuators:
      raise BranchStateError("X2 adapter requires builtin position actuators")
    if any(type(actuator) is not BuiltinPositionActuator for actuator in actuators):
      raise BranchStateError(
        "learned, recurrent, custom, or non-position actuators are unsupported"
      )
    command_manager = getattr(self.env, "command_manager", None)
    get_term = getattr(command_manager, "get_term", None)
    if not callable(get_term):
      raise BranchStateError("X2 adapter requires command_manager.get_term")
    command = get_term("motion")
    if type(command) is not MultiMotionCommand:
      raise BranchStateError(
        "X2 adapter requires the source MultiMotionCommand, not a scalar or custom command"
      )
    self._validate_event_inventory()
    observation = getattr(self.env, "observation_manager", None)
    if observation is None:
      raise BranchStateError("X2 adapter requires an ObservationManager")
    if getattr(observation, "_obs_buffer", None) is not getattr(
      self.env, "obs_buf", None
    ):
      raise BranchStateError(
        "environment and ObservationManager cached observation aliases differ"
      )

  def _validate_event_inventory(self) -> None:
    manager = getattr(self.env, "event_manager", None)
    cfgs = getattr(manager, "_mode_term_cfgs", None)
    if not isinstance(cfgs, Mapping):
      raise BranchStateError("event manager active-term inventory is missing")
    allowed_modules = {
      "mjlab.envs.mdp.events",
      "mjlab.envs.mdp.dr._core",
      "mjlab.envs.mdp.dr.body",
      "mjlab.envs.mdp.dr.actuator",
      "mjlab.envs.mdp.dr.geom",
      "mjlab.envs.mdp.dr.joint",
      "mjlab.envs.mdp.dr.site",
      "mjlab.envs.mdp.dr.tendon",
    }
    for mode_cfgs in cfgs.values():
      if not isinstance(mode_cfgs, list):
        raise BranchStateError("event mode inventory is malformed")
      for term_cfg in mode_cfgs:
        func = getattr(term_cfg, "func", None)
        if not isfunction(func) or func.__module__ not in allowed_modules:
          raise BranchStateError(
            f"stateful or unknown event term {getattr(func, '__qualname__', func)!r} "
            "has no explicit X2 restore handler"
          )

  def capture(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if self.source_row is None:
      raise BranchStateError("X2 capture requires a source row")
    fields: dict[str, Any] = {}
    for descriptor in DYNAMIC_TABLE:
      fields[descriptor.name] = self._capture_world(descriptor)
    for descriptor in MODEL_TABLE:
      target = self._path(descriptor.path)
      if target is None:
        raise BranchStateError(f"required model field missing: {descriptor.name}")
      fields[descriptor.name] = self._capture_model(descriptor.name, target)
    for descriptor in CACHED_TABLE:
      target = self._path(descriptor.path)
      if target is None:
        raise BranchStateError(f"required cached field missing: {descriptor.name}")
      if descriptor.name == "obs_buf":
        fields[descriptor.name] = _capture_row_nested(target, self.source_row)
      else:
        fields[descriptor.name] = _owned(self._capture_world_value(target))
    fields.update(self._capture_policy_inputs())
    managers = self._capture_managers()
    buffers = self._capture_buffers()
    _check_finite(fields)
    _check_finite(managers)
    _check_finite(buffers)
    self._validate_quaternions(fields)
    return fields, managers, buffers

  def restore(
    self,
    fields: dict[str, Any],
    managers: dict[str, Any],
    buffers: dict[str, Any],
    *,
    destination_row: int,
  ) -> None:
    self._validate_payload(fields, managers, buffers, destination_row)

    # Everything above this point is validation only.  Model values are written
    # before the backend recomputes derived constants; no pointer is replaced.
    for descriptor in MODEL_TABLE:
      target = self._path(descriptor.path)
      assert target is not None
      self._restore_model(
        descriptor.name, target, fields[descriptor.name], destination_row
      )
    for descriptor in DYNAMIC_TABLE:
      target = self._path(descriptor.path)
      assert target is not None
      self._restore_world(
        descriptor.name, target, fields[descriptor.name], destination_row, managers
      )
    sensordata = self._path(CACHED_TABLE[0].path)
    assert sensordata is not None
    self._copy_world_target(sensordata, fields["sensordata"], destination_row)

    self._restore_environment(managers["environment"], destination_row)
    self._restore_action_and_entity(managers, destination_row)
    self._restore_command(managers["command"], destination_row)
    self._restore_events(managers["events"], destination_row)
    self._restore_sensors(managers["sensors"], destination_row)

    sim = self.env.sim
    recompute = getattr(sim, "recompute_constants", None)
    if not callable(recompute):
      raise BranchStateError("X2 simulation has no recompute_constants handler")
    recompute(RecomputeLevel.set_const)
    forward = getattr(sim, "forward", None)
    if not callable(forward):
      raise BranchStateError("X2 simulation has no forward handler")
    forward()
    # mj_forward may update solver warm starts.  The copied value is the state
    # for the next branch transition, so restore it after forward exactly.
    warmstart = self._path(("sim", "data", "qacc_warmstart"))
    assert warmstart is not None
    self._copy_world_target(warmstart, fields["qacc_warmstart"], destination_row)

    obs_buf = self._path(("obs_buf",))
    assert obs_buf is not None
    _restore_row_nested(obs_buf, fields["obs_buf"], destination_row)
    self._restore_policy_inputs(fields, destination_row)
    self._restore_buffers(buffers, destination_row)

  def _path(self, path: tuple[str, ...]) -> Any:
    value: Any = self.env
    for name in path:
      value = getattr(value, name, None)
      if value is None:
        return None
    return value

  def _capture_model(self, name: str, target: Any) -> Any:
    tensor = _tensor_required(target, f"sim.model.{name}")
    if name in self.env.sim.expanded_fields:
      return _owned(self._capture_world_value(target))
    return _owned(tensor)

  def _capture_world_value(self, value: Any) -> Any:
    tensor = _tensor_required(value, "world field")
    assert self.source_row is not None
    if tensor.ndim == 0 or tensor.shape[0] <= self.source_row:
      raise BranchStateError("world field row mapping is invalid")
    return tensor[self.source_row]

  def _capture_world(self, descriptor: FieldDescriptor) -> Any:
    value = self._path(descriptor.path)
    if value is None:
      raise BranchStateError(f"required dynamic field missing: {descriptor.name}")
    return _owned(self._capture_world_value(value))

  def _capture_policy_inputs(self) -> dict[str, Any]:
    assert self.source_row is not None
    explicit = getattr(self.env, "coverage_cached_policy_inputs", None)
    if callable(explicit):
      result = explicit(self.source_row)
      if not isinstance(result, Mapping):
        raise BranchStateError("cached policy input handler must return a mapping")
      values = dict(result)
    else:
      values = self._derive_policy_inputs()
    required = {
      "teacher_actor_view",
      "student_reference",
      "student_conditioning",
      "previous_action",
    }
    if set(values) != required:
      raise BranchStateError(
        f"cached policy input inventory mismatch: expected {sorted(required)}"
      )
    copied = {name: _owned(values[name]) for name in required}
    for name, value in copied.items():
      if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise BranchStateError(f"cached policy field {name} must be one row tensor")
    if copied["student_reference"].shape != (DEFAULT_SCHEMA.reference_dim,):
      raise BranchStateError("cached student reference has the wrong X2 width")
    if copied["student_conditioning"].shape != (DEFAULT_SCHEMA.conditioning_dim,):
      raise BranchStateError("cached student conditioning has the wrong X2 width")
    if copied["previous_action"].shape != (ACTION_DIM,):
      raise BranchStateError("cached previous action has the wrong X2 width")
    return copied

  def _derive_policy_inputs(self) -> dict[str, torch.Tensor]:
    """Read the existing observation cache without another compute/noise draw."""
    assert self.source_row is not None
    observation = self.env.observation_manager
    cached_method = getattr(observation, "cached_observations", None)
    if not callable(cached_method):
      raise BranchStateError("ObservationManager lacks its cached_observations seam")
    cached = cached_method()
    if not isinstance(cached, Mapping) or not isinstance(
      cached.get("actor"), torch.Tensor
    ):
      raise BranchStateError("cached actor observation is not a flat tensor")
    actor = cached["actor"]
    if actor.ndim != 2 or actor.shape[0] != self._num_envs:
      raise BranchStateError("cached actor observation must be [num_envs, width]")
    names = getattr(observation, "_group_obs_term_names", None)
    dims = getattr(observation, "_group_obs_term_dim", None)
    concatenate = getattr(observation, "_group_obs_concatenate", None)
    if (
      not isinstance(names, Mapping)
      or not isinstance(dims, Mapping)
      or not isinstance(concatenate, Mapping)
    ):
      raise BranchStateError("observation term layout inventory is missing")
    if concatenate.get("actor") is not True:
      raise BranchStateError("X2 actor observation must use concatenated terms")
    term_names = names.get("actor")
    term_dims = dims.get("actor")
    if not isinstance(term_names, list) or not isinstance(term_dims, list):
      raise BranchStateError("actor observation term layout is malformed")
    if len(term_names) != len(term_dims):
      raise BranchStateError("actor observation names and dimensions disagree")
    spans: dict[str, slice] = {}
    offset = 0
    for name, dim in zip(term_names, term_dims, strict=True):
      if not isinstance(name, str) or not isinstance(dim, tuple) or len(dim) != 1:
        raise BranchStateError("actor observation term dimension is not explicit")
      width = int(dim[0])
      if width == 0:
        # A disabled observation term is legitimate.  motion_lookahead returns an
        # explicit empty tensor when preview is off (lookahead_s <= 0, the
        # documented default).  A zero-width term contributes an empty span and
        # shifts nothing, so skip it and keep the checks on terms that are present.
        continue
      if width < 0:
        raise BranchStateError("actor observation term width must not be negative")
      spans[name] = slice(offset, offset + width)
      offset += width
    if offset != actor.shape[1]:
      raise BranchStateError("actor observation layout does not cover cached width")
    required = {
      "command",
      "motion_anchor_ori_b",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    }
    if not required.issubset(spans):
      raise BranchStateError("X2 actor observation omits a required named term")
    row = actor[self.source_row]
    command = _slice_term(row, spans["command"], 2 * JOINT_DIM, "command")
    anchor = _slice_term(row, spans["motion_anchor_ori_b"], 6, "motion_anchor_ori_b")
    gyro = _slice_term(row, spans["base_ang_vel"], 3, "base_ang_vel")
    joint_pos = _slice_term(row, spans["joint_pos"], JOINT_DIM, "joint_pos")
    joint_vel = _slice_term(row, spans["joint_vel"], JOINT_DIM, "joint_vel")
    actions = _slice_term(row, spans["actions"], ACTION_DIM, "actions")
    robot = self.env.scene.entities["robot"]
    gravity = _tensor_required(
      robot.data.projected_gravity_b, "robot projected gravity"
    )
    if gravity.shape[0] != self._num_envs:
      raise BranchStateError("robot projected gravity has the wrong world axis")
    return {
      "teacher_actor_view": row.detach().clone(),
      "student_reference": torch.cat((command, anchor)),
      "student_conditioning": torch.cat(
        (gravity[self.source_row], gyro, joint_pos, joint_vel, actions)
      ),
      "previous_action": actions,
    }

  def _capture_managers(self) -> dict[str, Any]:
    assert self.source_row is not None
    robot = self.env.scene.entities["robot"]
    data = robot.data
    action = self.env.action_manager
    action_state: dict[str, Any] = {}
    for name in _ACTION_MANAGER_FIELDS:
      action_state[name] = _owned(
        _row_required(getattr(action, name), self.source_row, name)
      )
    terms = getattr(action, "_terms", None)
    if not isinstance(terms, Mapping) or not terms:
      raise BranchStateError("X2 action manager has no explicit active terms")
    term_state: dict[str, Any] = {}
    for term_name, term in terms.items():
      if type(term) is not JointPositionAction:
        raise BranchStateError(
          f"unsupported X2 action term {term_name!r}: {type(term).__name__}"
        )
      values: dict[str, Any] = {}
      for field_name in _ACTION_TERM_FIELDS:
        if not hasattr(term, field_name):
          raise BranchStateError(f"action term {term_name!r} lacks {field_name}")
        value = getattr(term, field_name)
        if field_name in {"_raw_actions", "_processed_actions"}:
          values[field_name] = _owned(_row_required(value, self.source_row, field_name))
        elif field_name == "_target_names":
          if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
          ):
            raise BranchStateError(
              f"action term {term_name!r} target names are malformed"
            )
          values[field_name] = tuple(value)
        elif field_name == "_target_ids":
          values[field_name] = _owned(value)
        else:
          values[field_name] = _owned(value)
      entity = getattr(term, "_entity", None)
      if entity is not robot:
        raise BranchStateError("X2 action term controls an unexpected entity")
      term_state[str(term_name)] = values

    entity_state: dict[str, Any] = {}
    for field_name in _ENTITY_FIELDS:
      if not hasattr(data, field_name):
        raise BranchStateError(f"X2 entity data lacks required field {field_name}")
      entity_state[field_name] = _owned(
        _row_required(getattr(data, field_name), self.source_row, field_name)
      )

    command = self.env.command_manager.get_term("motion")
    command_state = self._capture_command(command)
    return {
      "environment": self._capture_environment(),
      "scene_origin": _owned(
        _row_required(self.env.scene.env_origins, self.source_row, "scene_origin")
      ),
      "action": action_state,
      "action_terms": term_state,
      "entity": entity_state,
      "command": command_state,
      "events": self._capture_events(),
      "observation": self._capture_observation_manager(),
      "sensors": self._capture_sensors(),
    }

  def _capture_environment(self) -> dict[str, Any]:
    assert self.source_row is not None
    state: dict[str, Any] = {}
    for name in ("_sim_step_counter", "common_step_counter"):
      value = getattr(self.env, name, None)
      if isinstance(value, bool) or not isinstance(value, int):
        raise BranchStateError(f"environment counter {name} is missing or malformed")
      state[name] = value
    for name in _ENV_ROW_FIELDS:
      state[name] = _owned(
        _row_required(getattr(self.env, name), self.source_row, name)
      )
    return state

  def _capture_command(self, command: Any) -> dict[str, Any]:
    assert self.source_row is not None
    if type(command).__name__ != "MultiMotionCommand":
      raise BranchStateError("X2 command changed type during capture")
    row: dict[str, Any] = {}
    for name in _COMMAND_ROW_FIELDS:
      if not hasattr(command, name):
        raise BranchStateError(f"X2 motion command lacks {name}")
      row[name] = _owned(_row_required(getattr(command, name), self.source_row, name))
    metrics = getattr(command, "metrics", None)
    if not isinstance(metrics, Mapping):
      raise BranchStateError("X2 motion command metrics inventory is missing")
    row_metrics = {
      str(name): _owned(_row_required(value, self.source_row, f"metric:{name}"))
      for name, value in metrics.items()
    }
    membership = {
      name: _row_membership(getattr(command, name), self.source_row, name)
      for name in _COMMAND_MEMBERSHIP_FIELDS
    }
    event_fields = (
      "_event_available",
      "_event_completed",
      "_event_interrupted",
      "_event_completed_generation",
      "_event_interrupted_generation",
    )
    events = {
      name: _owned(_row_required(getattr(command, name), self.source_row, name))
      for name in event_fields
    }
    reasons = getattr(command, "_event_reasons", None)
    if (
      not isinstance(reasons, list)
      or len(reasons) != self._num_envs
      or not isinstance(reasons[self.source_row], str)
    ):
      raise BranchStateError("X2 command event reasons are malformed")
    reset_getter = getattr(command, "reset_rng_state", None)
    if not callable(reset_getter):
      raise BranchStateError("X2 motion command has no reset RNG state handler")
    return {
      "type": type(command).__name__,
      "row": row,
      "metrics": row_metrics,
      "membership": membership,
      "events": events,
      "event_reason": reasons[self.source_row],
      "pending_forward": bool(getattr(command, "_pending_forward", False)),
      "reset_rng_state": _owned(reset_getter()),
    }

  def _capture_events(self) -> dict[str, Any]:
    assert self.source_row is not None
    manager = self.env.event_manager
    state: dict[str, Any] = {}
    for name in (
      "_interval_term_time_left",
      "_reset_term_last_triggered_step_id",
      "_reset_term_last_triggered_once",
    ):
      values = getattr(manager, name, None)
      if not isinstance(values, list):
        raise BranchStateError(f"event manager field {name} is missing")
      state[name] = tuple(
        _capture_event_value(value, self.source_row, self._num_envs) for value in values
      )
    return state

  def _capture_observation_manager(self) -> dict[str, Any]:
    observation = self.env.observation_manager
    instances = getattr(observation, "_group_obs_class_instances", None)
    if not isinstance(instances, Mapping):
      raise BranchStateError("observation manager noise inventory is missing")
    if any(group for group in instances.values()):
      raise BranchStateError(
        "stateful observation noise models are unsupported without an explicit handler"
      )
    return {}

  def _capture_sensors(self) -> dict[str, Any]:
    assert self.source_row is not None
    sensors = getattr(self.env.scene, "sensors", None)
    if not isinstance(sensors, Mapping):
      raise BranchStateError("scene sensor inventory is missing")
    result: dict[str, Any] = {}
    for name, sensor in sensors.items():
      if isinstance(sensor, BuiltinSensor):
        result[str(name)] = {"kind": "builtin"}
      elif isinstance(sensor, ContactSensor):
        air = getattr(sensor, "_air_time_state", None)
        history = getattr(sensor, "_history_state", None)
        result[str(name)] = {
          "kind": "contact",
          "air": _capture_contact_air(air, self.source_row),
          "history": _capture_contact_history(history, self.source_row),
        }
      else:
        raise BranchStateError(
          f"unsupported active sensor {name!r}: {type(sensor).__name__}"
        )
    return result

  def _capture_buffers(self) -> dict[str, Any]:
    assert self.source_row is not None
    robot = self.env.scene.entities["robot"]
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Sequence):
      raise BranchStateError("robot actuator inventory is missing")
    aliases: dict[str, Any] = {}
    alias_members: dict[str, tuple[int, ...]] = {}
    seen: dict[int, str] = {}
    for index, actuator in enumerate(actuators):
      delay = getattr(actuator, "_delay_buffer", None)
      if delay is None:
        continue
      if not isinstance(delay, DelayBuffer):
        raise BranchStateError("custom actuator delay is unsupported")
      key = seen.get(id(delay))
      if key is None:
        key = _actuator_alias_key(index, actuator, delay)
        seen[id(delay)] = key
        aliases[key] = delay.capture_state([self.source_row])
        alias_members[key] = (index,)
      else:
        alias_members[key] = (*alias_members[key], index)
    observation = self.env.observation_manager
    observation_state: dict[str, Any] = {}
    for group_name, group in observation._group_obs_term_delay_buffer.items():
      for term_name, delay in group.items():
        if not isinstance(delay, DelayBuffer):
          raise BranchStateError("custom observation term delay is unsupported")
        observation_state[f"term:{group_name}:{term_name}"] = delay.capture_state(
          [self.source_row]
        )
    for group_name, group in observation._group_obs_delay_group_buffer.items():
      for delay_name, delay in group.items():
        if not isinstance(delay, DelayBuffer):
          raise BranchStateError("custom observation group delay is unsupported")
        observation_state[f"group:{group_name}:{delay_name}"] = delay.capture_state(
          [self.source_row]
        )
    for group_name, group in observation._group_obs_term_history_buffer.items():
      for term_name, history in group.items():
        if not isinstance(history, CircularBuffer):
          raise BranchStateError("custom observation history is unsupported")
        observation_state[f"history:{group_name}:{term_name}"] = history.capture_state(
          [self.source_row]
        )
    return {
      "actuator_aliases": aliases,
      "actuator_alias_members": alias_members,
      "observation": observation_state,
    }

  def _validate_payload(
    self,
    fields: dict[str, Any],
    managers: dict[str, Any],
    buffers: dict[str, Any],
    destination_row: int,
  ) -> None:
    if (
      not isinstance(destination_row, int)
      or isinstance(destination_row, bool)
      or not 0 <= destination_row < self._num_envs
    ):
      raise BranchStateError("destination row is outside the environment batch")
    expected_fields = {
      descriptor.name for descriptor in DYNAMIC_TABLE + MODEL_TABLE + CACHED_TABLE
    } | {
      "teacher_actor_view",
      "student_reference",
      "student_conditioning",
      "previous_action",
    }
    if set(fields) != expected_fields:
      raise BranchStateMismatch(
        "X2 field table is incomplete or contains unknown fields"
      )
    expected_managers = {
      "environment",
      "scene_origin",
      "action",
      "action_terms",
      "entity",
      "command",
      "events",
      "observation",
      "sensors",
    }
    if set(managers) != expected_managers:
      raise BranchStateMismatch("X2 manager state inventory is incomplete")
    if set(buffers) != {"actuator_aliases", "actuator_alias_members", "observation"}:
      raise BranchStateMismatch("X2 buffer state inventory is incomplete")
    _check_finite(fields)
    _check_finite(managers)
    _check_finite(buffers)
    self._validate_quaternions(fields)
    for descriptor in DYNAMIC_TABLE:
      target = self._path(descriptor.path)
      assert target is not None
      _validate_world_payload(
        target, fields[descriptor.name], destination_row, descriptor.name
      )
    for descriptor in MODEL_TABLE:
      target = self._path(descriptor.path)
      assert target is not None
      _validate_model_payload(
        target,
        fields[descriptor.name],
        descriptor.name,
        descriptor.name in self.env.sim.expanded_fields,
        destination_row,
      )
    sensordata = self._path(("sim", "data", "sensordata"))
    assert sensordata is not None
    _validate_world_payload(
      sensordata, fields["sensordata"], destination_row, "sensordata"
    )
    obs_buf = self._path(("obs_buf",))
    assert obs_buf is not None
    _validate_row_nested_target(obs_buf, fields["obs_buf"], destination_row)
    self._validate_policy_fields(fields)
    self._validate_manager_payload(managers, destination_row)
    self._validate_buffer_payload(buffers, destination_row)

  def _validate_policy_fields(self, fields: dict[str, Any]) -> None:
    expected = {
      "teacher_actor_view": fields["teacher_actor_view"],
      "student_reference": fields["student_reference"],
      "student_conditioning": fields["student_conditioning"],
      "previous_action": fields["previous_action"],
    }
    for name, value in expected.items():
      if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise BranchStateMismatch(f"cached policy field {name} is malformed")
    if expected["student_reference"].shape != (DEFAULT_SCHEMA.reference_dim,):
      raise BranchStateMismatch("cached reference width differs from X2 schema")
    if expected["student_conditioning"].shape != (DEFAULT_SCHEMA.conditioning_dim,):
      raise BranchStateMismatch("cached conditioning width differs from X2 schema")
    if expected["previous_action"].shape != (ACTION_DIM,):
      raise BranchStateMismatch("cached previous action width differs from X2 schema")

  def _validate_manager_payload(self, managers: dict[str, Any], row: int) -> None:
    environment = managers["environment"]
    for name in ("_sim_step_counter", "common_step_counter"):
      if isinstance(environment.get(name), bool) or not isinstance(
        environment.get(name), int
      ):
        raise BranchStateMismatch(f"environment counter {name} is malformed")
    for name in _ENV_ROW_FIELDS:
      target = getattr(self.env, name, None)
      if target is None:
        raise BranchStateMismatch(f"destination environment field {name} is missing")
      _validate_world_payload(target, environment[name], row, name)
    _validate_world_payload(
      self.env.scene.env_origins, managers["scene_origin"], row, "scene_origin"
    )
    action = self.env.action_manager
    for name in _ACTION_MANAGER_FIELDS:
      _validate_world_payload(
        getattr(action, name), managers["action"][name], row, name
      )
    terms = getattr(action, "_terms", None)
    if not isinstance(terms, Mapping) or set(terms) != set(managers["action_terms"]):
      raise BranchStateMismatch("destination action term inventory differs")
    for term_name, state in managers["action_terms"].items():
      term = terms[term_name]
      if type(term) is not JointPositionAction:
        raise BranchStateMismatch(
          f"destination action term {term_name!r} is unsupported"
        )
      for field_name in ("_raw_actions", "_processed_actions"):
        _validate_world_payload(
          getattr(term, field_name), state[field_name], row, field_name
        )
      ids = _tensor_required(
        getattr(term, "_target_ids", None), f"term {term_name} target ids"
      )
      if not torch.equal(ids, state["_target_ids"]):
        raise BranchStateMismatch(f"action target IDs differ for {term_name!r}")
      if tuple(getattr(term, "_target_names", ())) != tuple(state["_target_names"]):
        raise BranchStateMismatch(f"action target names differ for {term_name!r}")
    entity = self.env.scene.entities["robot"].data
    for name in _ENTITY_FIELDS:
      _validate_world_payload(
        getattr(entity, name), managers["entity"][name], row, name
      )
    command = self.env.command_manager.get_term("motion")
    if self._num_envs != 1 and bool(managers["command"].get("pending_forward")) != bool(
      getattr(command, "_pending_forward", False)
    ):
      raise BranchStateMismatch(
        "global pending-forward state cannot map one row without changing another slot"
      )
    self._validate_command_payload(command, managers["command"], row)
    self._validate_events_payload(managers["events"], row)
    self._validate_sensors_payload(managers["sensors"], row)

  def _validate_command_payload(
    self, command: Any, state: dict[str, Any], row: int
  ) -> None:
    if (
      state.get("type") != "MultiMotionCommand"
      or type(command) is not MultiMotionCommand
    ):
      raise BranchStateMismatch("motion command type differs")
    row_state = state.get("row")
    if not isinstance(row_state, Mapping) or set(row_state) != set(_COMMAND_ROW_FIELDS):
      raise BranchStateMismatch("motion command row inventory differs")
    for name in _COMMAND_ROW_FIELDS:
      _validate_world_payload(getattr(command, name), row_state[name], row, name)
    metrics = getattr(command, "metrics", None)
    if not isinstance(metrics, Mapping) or set(metrics) != set(state["metrics"]):
      raise BranchStateMismatch("motion command metric inventory differs")
    for name, value in state["metrics"].items():
      _validate_world_payload(metrics[name], value, row, f"metric:{name}")
    membership = state.get("membership")
    if not isinstance(membership, Mapping) or set(membership) != set(
      _COMMAND_MEMBERSHIP_FIELDS
    ):
      raise BranchStateMismatch("motion command membership inventory differs")
    for name in _COMMAND_MEMBERSHIP_FIELDS:
      if not isinstance(membership[name], bool):
        raise BranchStateMismatch(f"motion command membership {name} is malformed")
    if not isinstance(state.get("pending_forward"), bool):
      raise BranchStateMismatch("motion command pending-forward flag is malformed")
    reset_state = state.get("reset_rng_state")
    if (
      not isinstance(reset_state, torch.Tensor)
      or reset_state.dtype != torch.uint8
      or reset_state.ndim != 1
    ):
      raise BranchStateMismatch("motion command reset RNG state is malformed")
    events = state.get("events")
    if not isinstance(events, Mapping):
      raise BranchStateMismatch("motion command boundary events are missing")
    for name in (
      "_event_available",
      "_event_completed",
      "_event_interrupted",
      "_event_completed_generation",
      "_event_interrupted_generation",
    ):
      _validate_world_payload(getattr(command, name), events[name], row, name)
    if not isinstance(state.get("event_reason"), str):
      raise BranchStateMismatch("motion command event reason is malformed")

  def _validate_events_payload(self, state: dict[str, Any], row: int) -> None:
    manager = self.env.event_manager
    for name in (
      "_interval_term_time_left",
      "_reset_term_last_triggered_step_id",
      "_reset_term_last_triggered_once",
    ):
      live = getattr(manager, name, None)
      saved = state.get(name)
      if (
        not isinstance(live, list)
        or not isinstance(saved, tuple)
        or len(live) != len(saved)
      ):
        raise BranchStateMismatch(f"event field {name} layout differs")
      for live_value, saved_value in zip(live, saved, strict=True):
        _validate_event_value(live_value, saved_value, row, self._num_envs, name)

  def _validate_sensors_payload(self, state: dict[str, Any], row: int) -> None:
    sensors = self.env.scene.sensors
    if set(sensors) != set(state):
      raise BranchStateMismatch("active sensor inventory differs")
    for name, sensor_state in state.items():
      sensor = sensors[name]
      if isinstance(sensor, BuiltinSensor):
        if sensor_state != {"kind": "builtin"}:
          raise BranchStateMismatch(f"builtin sensor {name!r} state is malformed")
      elif isinstance(sensor, ContactSensor):
        if sensor_state.get("kind") != "contact":
          raise BranchStateMismatch(f"contact sensor {name!r} state is malformed")
        _validate_contact_state(sensor, sensor_state, row)
      else:
        raise BranchStateMismatch(f"unsupported destination sensor {name!r}")

  def _validate_buffer_payload(self, state: dict[str, Any], row: int) -> None:
    live_aliases, live_members = _destination_actuator_alias_layout(
      self.env.scene.entities["robot"].actuators
    )
    saved_aliases = state["actuator_aliases"]
    if set(live_aliases) != set(saved_aliases) or set(saved_aliases) != set(
      state["actuator_alias_members"]
    ):
      raise BranchStateMismatch("actuator delay alias layout differs")
    for key, delay in live_aliases.items():
      members = state["actuator_alias_members"][key]
      if tuple(members) != live_members[key]:
        raise BranchStateMismatch("actuator delay alias members differ")
      if not isinstance(delay, DelayBuffer):
        raise BranchStateMismatch("destination actuator delay is unsupported")
      _validate_delay_state(delay, saved_aliases[key], row)
    observation_state = state["observation"]
    live_observation = _destination_observation_buffers(self.env.observation_manager)
    if set(live_observation) != set(observation_state):
      raise BranchStateMismatch("observation delay/history layout differs")
    for key, (kind, buffer) in live_observation.items():
      if kind == "delay":
        assert isinstance(buffer, DelayBuffer)
        _validate_delay_state(buffer, observation_state[key], row)
      else:
        assert isinstance(buffer, CircularBuffer)
        _validate_circular_state(buffer, observation_state[key], row)

  def _restore_model(self, name: str, target: Any, source: Any, row: int) -> None:
    if name in self.env.sim.expanded_fields:
      self._copy_world_target(target, source, row)
    else:
      _copy_tensor_target(target, source, name)

  def _restore_world(
    self,
    name: str,
    target: Any,
    source: Any,
    row: int,
    managers: dict[str, Any],
  ) -> None:
    value = source
    if name in {"qpos", "mocap_pos"}:
      value = _translate_origin(
        name,
        source,
        managers["scene_origin"],
        self.env.scene.env_origins[row],
        self.env.scene.entities["robot"],
      )
    self._copy_world_target(target, value, row)

  def _copy_world_target(self, target: Any, source: Any, row: int) -> None:
    tensor = _tensor_required(target, "world destination")
    source_tensor = _tensor_required(source, "world source")
    tensor[row].copy_(source_tensor)

  def _restore_environment(self, state: dict[str, Any], row: int) -> None:
    self.env._sim_step_counter = state["_sim_step_counter"]
    self.env.common_step_counter = state["common_step_counter"]
    for name in _ENV_ROW_FIELDS:
      self._copy_world_target(getattr(self.env, name), state[name], row)

  def _restore_action_and_entity(self, managers: dict[str, Any], row: int) -> None:
    action = self.env.action_manager
    for name in _ACTION_MANAGER_FIELDS:
      self._copy_world_target(getattr(action, name), managers["action"][name], row)
    for term_name, state in managers["action_terms"].items():
      term = action._terms[term_name]
      for name in ("_raw_actions", "_processed_actions"):
        self._copy_world_target(getattr(term, name), state[name], row)
    entity = self.env.scene.entities["robot"].data
    for name in _ENTITY_FIELDS:
      self._copy_world_target(getattr(entity, name), managers["entity"][name], row)

  def _restore_command(self, state: dict[str, Any], row: int) -> None:
    command = self.env.command_manager.get_term("motion")
    for name, value in state["row"].items():
      self._copy_world_target(getattr(command, name), value, row)
    for name, value in state["metrics"].items():
      self._copy_world_target(command.metrics[name], value, row)
    for name, is_member in state["membership"].items():
      current = _tensor_required(getattr(command, name), name)
      setattr(command, name, _membership_tensor(current, is_member, row))
    if state["pending_forward"] and self._num_envs != 1:
      raise BranchStateMismatch(
        "global pending-forward state cannot map one row safely"
      )
    if self._num_envs == 1:
      command._pending_forward = state["pending_forward"]
    setter = getattr(command, "set_reset_rng_state", None)
    if not callable(setter):
      raise BranchStateError("motion command reset RNG restore handler is missing")
    setter(state["reset_rng_state"])
    for name, value in state["events"].items():
      self._copy_world_target(getattr(command, name), value, row)
    command._event_reasons[row] = state["event_reason"]

  def _restore_events(self, state: dict[str, Any], row: int) -> None:
    manager = self.env.event_manager
    for name in (
      "_interval_term_time_left",
      "_reset_term_last_triggered_step_id",
      "_reset_term_last_triggered_once",
    ):
      for live, saved in zip(getattr(manager, name), state[name], strict=True):
        _restore_event_value(live, saved, row, self._num_envs)

  def _restore_sensors(self, state: dict[str, Any], row: int) -> None:
    for name, sensor_state in state.items():
      sensor = self.env.scene.sensors[name]
      if isinstance(sensor, ContactSensor):
        _restore_contact_state(sensor, sensor_state, row)

  def _restore_buffers(self, state: dict[str, Any], row: int) -> None:
    live_aliases, _ = _destination_actuator_alias_layout(
      self.env.scene.entities["robot"].actuators
    )
    for key, payload in state["actuator_aliases"].items():
      delay = live_aliases[key]
      delay.restore_state(
        payload, torch.tensor([row], dtype=torch.long, device=delay.device)
      )
    live_observation = _destination_observation_buffers(self.env.observation_manager)
    for key, payload in state["observation"].items():
      kind, buffer = live_observation[key]
      if kind == "delay":
        assert isinstance(buffer, DelayBuffer)
        buffer.restore_state(
          payload, torch.tensor([row], dtype=torch.long, device=buffer.device)
        )
      else:
        assert isinstance(buffer, CircularBuffer)
        buffer.restore_state(
          payload, torch.tensor([row], dtype=torch.long, device=buffer.device)
        )

  def _restore_policy_inputs(self, fields: dict[str, Any], row: int) -> None:
    explicit = getattr(self.env, "coverage_restore_policy_inputs", None)
    if callable(explicit):
      explicit(
        {
          name: _owned(fields[name])
          for name in (
            "teacher_actor_view",
            "student_reference",
            "student_conditioning",
            "previous_action",
          )
        },
        row,
      )
    derived = self._capture_policy_inputs()
    for name in (
      "teacher_actor_view",
      "student_reference",
      "student_conditioning",
      "previous_action",
    ):
      if not torch.equal(derived[name], fields[name]):
        raise BranchStateMismatch(f"cached policy input parity failed: {name}")

  @staticmethod
  def _validate_quaternions(fields: dict[str, Any]) -> None:
    qpos = fields.get("qpos")
    if isinstance(qpos, torch.Tensor) and qpos.numel() >= 7:
      norm = torch.linalg.vector_norm(qpos.reshape(-1)[3:7])
      if abs(float(norm) - 1.0) > 0.001:
        raise BranchStateError("qpos quaternion norm invalid")
    quat = fields.get("mocap_quat")
    if isinstance(quat, torch.Tensor) and quat.numel():
      if quat.numel() % 4:
        raise BranchStateError("mocap quaternion layout is malformed")
      if bool(
        (
          torch.abs(torch.linalg.vector_norm(quat.reshape(-1, 4), dim=1) - 1.0) > 0.001
        ).any()
      ):
        raise BranchStateError("mocap quaternion norm invalid")

  def coverage_cached_policy_view(self, index: int) -> torch.Tensor:
    """One cached actor observation row, cloned.  No recompute, no noise draw."""
    from mjlab.tasks.tracking.distillation.adapter import cached_policy_view_row

    return cached_policy_view_row(self.env, index)

  def coverage_executed_action(self, index: int) -> torch.Tensor:
    """One row of the action the environment last applied.

    The action manager keeps it, so no bookkeeping is needed on this side.
    """
    from mjlab.tasks.tracking.distillation.adapter import executed_action_row

    manager = getattr(self.env, "action_manager", None)
    action = getattr(manager, "action", None)
    return executed_action_row(action, index)


def _tensor_required(value: Any, name: str) -> torch.Tensor:
  tensor = _tensor_value(value)
  if tensor is None:
    raise BranchStateError(f"{name} must be a TorchArray or torch.Tensor")
  return tensor


def _row_required(value: Any, row: int, name: str) -> torch.Tensor:
  tensor = _tensor_required(value, name)
  if tensor.ndim == 0 or tensor.shape[0] <= row:
    raise BranchStateError(f"{name} has no source row")
  return tensor[row]


def _slice_term(row: torch.Tensor, span: slice, width: int, name: str) -> torch.Tensor:
  value = row[span]
  if value.shape != (width,):
    raise BranchStateError(
      f"actor term {name} has shape {tuple(value.shape)}, expected {(width,)}"
    )
  return value.detach().clone()


def _capture_row_nested(value: Any, row: int) -> Any:
  if isinstance(value, Mapping):
    return {key: _capture_row_nested(item, row) for key, item in value.items()}
  tensor = _tensor_value(value)
  if tensor is not None:
    return _owned(_row_required(value, row, "cached row"))
  raise BranchStateError("cached observation contains an unsupported value")


def _restore_row_nested(target: Any, source: Any, row: int) -> None:
  if isinstance(target, Mapping) and isinstance(source, Mapping):
    if set(target) != set(source):
      raise BranchStateMismatch("cached observation keys differ")
    for key in target:
      _restore_row_nested(target[key], source[key], row)
    return
  target_tensor = _tensor_required(target, "cached row destination")
  source_tensor = _tensor_required(source, "cached row source")
  if target_tensor.ndim == 0 or target_tensor.shape[0] <= row:
    raise BranchStateMismatch("cached row destination has no destination row")
  if (
    target_tensor[row].shape != source_tensor.shape
    or target_tensor.dtype != source_tensor.dtype
    or target_tensor.device != source_tensor.device
  ):
    raise BranchStateMismatch("cached row dtype/device/shape differs")
  target_tensor[row].copy_(source_tensor)


def _validate_row_nested_target(target: Any, source: Any, row: int) -> None:
  if isinstance(target, Mapping) and isinstance(source, Mapping):
    if set(target) != set(source):
      raise BranchStateMismatch("cached observation keys differ")
    for key in target:
      _validate_row_nested_target(target[key], source[key], row)
    return
  target_tensor = _tensor_required(target, "cached row destination")
  source_tensor = _tensor_required(source, "cached row source")
  if (
    target_tensor.ndim == 0
    or target_tensor.shape[0] <= row
    or target_tensor[row].shape != source_tensor.shape
  ):
    raise BranchStateMismatch("cached row shape differs")
  if (
    target_tensor.dtype != source_tensor.dtype
    or target_tensor.device != source_tensor.device
  ):
    raise BranchStateMismatch("cached row dtype/device differs")


def _copy_tensor_target(target: Any, source: Any, name: str) -> None:
  target_tensor = _tensor_required(target, name)
  source_tensor = _tensor_required(source, name)
  if (
    target_tensor.shape != source_tensor.shape
    or target_tensor.dtype != source_tensor.dtype
    or target_tensor.device != source_tensor.device
  ):
    raise BranchStateMismatch(f"global field {name} shape/dtype/device differs")
  target_tensor.copy_(source_tensor)


def _copy_shape_matches(target: Any, source: Any, row: int, name: str) -> None:
  target_tensor = _tensor_required(target, name)
  source_tensor = _tensor_required(source, name)
  if (
    target_tensor.ndim == 0
    or target_tensor.shape[0] <= row
    or target_tensor[row].shape != source_tensor.shape
  ):
    raise BranchStateMismatch(f"world field {name} shape differs")
  if (
    target_tensor.dtype != source_tensor.dtype
    or target_tensor.device != source_tensor.device
  ):
    raise BranchStateMismatch(f"world field {name} dtype/device differs")


def _validate_world_payload(target: Any, source: Any, row: int, name: str) -> None:
  _copy_shape_matches(target, source, row, name)


def _validate_model_payload(
  target: Any, source: Any, name: str, expanded: bool, row: int
) -> None:
  if expanded:
    _copy_shape_matches(target, source, row, name)
  else:
    _copy_tensor_target_shape(target, source, name)


def _copy_tensor_target_shape(target: Any, source: Any, name: str) -> None:
  target_tensor = _tensor_required(target, name)
  source_tensor = _tensor_required(source, name)
  if (
    target_tensor.shape != source_tensor.shape
    or target_tensor.dtype != source_tensor.dtype
    or target_tensor.device != source_tensor.device
  ):
    raise BranchStateMismatch(f"global field {name} shape/dtype/device differs")


def _row_membership(value: Any, row: int, name: str) -> bool:
  tensor = _tensor_required(value, name)
  if tensor.ndim != 1:
    raise BranchStateError(f"{name} must be a row-index tensor")
  return bool((tensor == row).any())


def _membership_tensor(value: torch.Tensor, member: bool, row: int) -> torch.Tensor:
  if value.ndim != 1:
    raise BranchStateMismatch("membership destination is malformed")
  keep = value[value != row]
  if member:
    keep = torch.cat(
      (keep, torch.tensor([row], dtype=value.dtype, device=value.device))
    )
  return keep


def _capture_event_value(value: Any, row: int, num_envs: int) -> Any:
  tensor = _tensor_required(value, "event state")
  if tensor.ndim and tensor.shape[0] == num_envs:
    return _owned(tensor[row])
  if tensor.ndim and tensor.shape[0] == 1:
    return _owned(tensor)
  raise BranchStateError("event state has no explicit global/row layout")


def _validate_event_value(
  live: Any, saved: Any, row: int, num_envs: int, name: str
) -> None:
  live_tensor = _tensor_required(live, name)
  saved_tensor = _tensor_required(saved, name)
  if live_tensor.ndim and live_tensor.shape[0] == num_envs:
    _validate_world_payload(live, saved_tensor, row, name)
  elif live_tensor.ndim and live_tensor.shape[0] == 1:
    if (
      live_tensor.shape != saved_tensor.shape
      or live_tensor.dtype != saved_tensor.dtype
      or live_tensor.device != saved_tensor.device
    ):
      raise BranchStateMismatch(f"global event field {name} differs")
  else:
    raise BranchStateMismatch(f"event field {name} has no supported layout")


def _restore_event_value(live: Any, saved: Any, row: int, num_envs: int) -> None:
  live_tensor = _tensor_required(live, "event destination")
  saved_tensor = _tensor_required(saved, "event source")
  if live_tensor.shape[0] == num_envs:
    live_tensor[row].copy_(saved_tensor)
  else:
    live_tensor.copy_(saved_tensor)


def _capture_contact_air(air: Any, row: int) -> Any:
  if air is None:
    return None
  names = (
    "current_air_time",
    "last_air_time",
    "current_contact_time",
    "last_contact_time",
  )
  if not all(hasattr(air, name) for name in names):
    raise BranchStateError("contact air-time state inventory is incomplete")
  return {name: _owned(_row_required(getattr(air, name), row, name)) for name in names}


def _capture_contact_history(history: Any, row: int) -> Any:
  if history is None:
    return None
  if not isinstance(history, Mapping):
    raise BranchStateError("contact history state is malformed")
  return {
    name: _owned(_row_required(value, row, f"contact history:{name}"))
    for name, value in history.items()
  }


def _validate_contact_state(
  sensor: ContactSensor, state: dict[str, Any], row: int
) -> None:
  air = getattr(sensor, "_air_time_state", None)
  saved_air = state.get("air")
  if (air is None) != (saved_air is None):
    raise BranchStateMismatch("contact air-time enabled state differs")
  if air is not None:
    if not isinstance(saved_air, Mapping):
      raise BranchStateMismatch("contact air-time state is malformed")
    for name in (
      "current_air_time",
      "last_air_time",
      "current_contact_time",
      "last_contact_time",
    ):
      _validate_world_payload(getattr(air, name), saved_air[name], row, name)
  history = getattr(sensor, "_history_state", None)
  saved_history = state.get("history")
  if (history is None) != (saved_history is None):
    raise BranchStateMismatch("contact history enabled state differs")
  if history is not None:
    if not isinstance(saved_history, Mapping):
      raise BranchStateMismatch("contact history state is malformed")
    if set(history) != set(saved_history):
      raise BranchStateMismatch("contact history fields differ")
    for name, value in history.items():
      _validate_world_payload(
        value, saved_history[name], row, f"contact history:{name}"
      )


def _restore_contact_state(
  sensor: ContactSensor, state: dict[str, Any], row: int
) -> None:
  air = getattr(sensor, "_air_time_state", None)
  if air is not None:
    for name, value in state["air"].items():
      _tensor_required(getattr(air, name), name)[row].copy_(
        _tensor_required(value, name)
      )
  history = getattr(sensor, "_history_state", None)
  if history is not None:
    for name, value in state["history"].items():
      _tensor_required(history[name], name)[row].copy_(_tensor_required(value, name))


def _actuator_alias_key(index: int, actuator: Any, delay: DelayBuffer) -> str:
  absorbed = getattr(actuator, "absorbed_actuators", (actuator,))
  names = tuple(tuple(getattr(item, "target_names", ())) for item in absorbed)
  ctrl_ids = tuple(
    int(item)
    for item in _tensor_required(actuator.ctrl_ids, "actuator ctrl_ids")
    .reshape(-1)
    .tolist()
  )
  return repr(
    (
      "actuator",
      index,
      type(actuator).__name__,
      names,
      ctrl_ids,
      delay.min_lag,
      delay.max_lag,
      delay.update_period,
      delay.per_env_phase,
    )
  )


def _destination_actuator_alias_layout(
  actuators: Sequence[Any],
) -> tuple[dict[str, DelayBuffer], dict[str, tuple[int, ...]]]:
  result: dict[str, DelayBuffer] = {}
  members: dict[str, tuple[int, ...]] = {}
  seen: dict[int, str] = {}
  for index, actuator in enumerate(actuators):
    delay = getattr(actuator, "_delay_buffer", None)
    if delay is None:
      continue
    if not isinstance(delay, DelayBuffer):
      raise BranchStateMismatch("destination actuator delay is unsupported")
    key = seen.get(id(delay))
    if key is None:
      key = _actuator_alias_key(index, actuator, delay)
      seen[id(delay)] = key
      result[key] = delay
      members[key] = (index,)
    else:
      members[key] = (*members[key], index)
  return result, members


def _destination_actuator_aliases(actuators: Sequence[Any]) -> dict[str, DelayBuffer]:
  return _destination_actuator_alias_layout(actuators)[0]


def _destination_observation_buffers(observation: Any) -> dict[str, tuple[str, Any]]:
  result: dict[str, tuple[str, Any]] = {}
  for group_name, group in observation._group_obs_term_delay_buffer.items():
    for term_name, delay in group.items():
      result[f"term:{group_name}:{term_name}"] = ("delay", delay)
  for group_name, group in observation._group_obs_delay_group_buffer.items():
    for delay_name, delay in group.items():
      result[f"group:{group_name}:{delay_name}"] = ("delay", delay)
  for group_name, group in observation._group_obs_term_history_buffer.items():
    for term_name, history in group.items():
      result[f"history:{group_name}:{term_name}"] = ("history", history)
  return result


def _validate_delay_state(
  delay: DelayBuffer, state: dict[str, object], row: int
) -> None:
  for name in (
    "min_lag",
    "max_lag",
    "per_env",
    "hold_prob",
    "update_period",
    "per_env_phase",
  ):
    if state.get(name) != getattr(delay, name):
      raise BranchStateMismatch(f"delay layout differs for {name}")
  _validate_circular_state(delay._buffer, state, row)
  current_lags = state.get("current_lags")
  step_count = state.get("step_count")
  phase_offsets = state.get("phase_offsets")
  for name, value in (
    ("current_lags", current_lags),
    ("step_count", step_count),
    ("phase_offsets", phase_offsets),
  ):
    if (
      not isinstance(value, torch.Tensor)
      or value.shape != (1,)
      or value.dtype != torch.long
    ):
      raise BranchStateMismatch(f"delay state {name} is malformed")
  assert isinstance(current_lags, torch.Tensor)
  assert isinstance(phase_offsets, torch.Tensor)
  if bool((current_lags < delay.min_lag).any()) or bool(
    (current_lags > delay.max_lag).any()
  ):
    raise BranchStateMismatch("delay current lag is outside configured bounds")
  if delay.update_period > 0 and bool((phase_offsets < 0).any()):
    raise BranchStateMismatch("delay phase offset is invalid")
  generator_state = state.get("generator_state")
  generator_device = state.get("generator_device")
  if delay.generator is None:
    if generator_state is not None or generator_device is not None:
      raise BranchStateMismatch("destination delay has no generator for saved state")
  elif (
    not isinstance(generator_state, torch.Tensor)
    or generator_state.dtype != torch.uint8
  ):
    raise BranchStateMismatch("saved delay generator state is missing or malformed")
  elif generator_device != str(getattr(delay.generator, "device", "cpu")):
    raise BranchStateMismatch("delay generator device differs")


def _validate_circular_state(
  buffer: CircularBuffer, state: dict[str, object], row: int
) -> None:
  if state.get("max_len") != buffer.max_length:
    raise BranchStateMismatch("circular history length differs")
  source_size = state.get("source_batch_size")
  rows = state.get("source_row_ids")
  if (
    not isinstance(source_size, int)
    or not isinstance(rows, torch.Tensor)
    or rows.shape != (1,)
  ):
    raise BranchStateMismatch("circular history source row layout is malformed")
  if rows.dtype != torch.long or int(rows[0]) < 0 or int(rows[0]) >= source_size:
    raise BranchStateMismatch("circular history source row is invalid")
  chronology = state.get("chronological")
  lengths = state.get("current_length")
  if (
    not isinstance(lengths, torch.Tensor)
    or lengths.shape != (1,)
    or lengths.dtype != torch.long
  ):
    raise BranchStateMismatch("circular history lengths are malformed")
  if bool((lengths < 0).any()) or bool((lengths > buffer.max_length).any()):
    raise BranchStateMismatch("circular history length value is invalid")
  if chronology is None:
    return
  if not isinstance(chronology, torch.Tensor) or chronology.ndim < 2:
    raise BranchStateMismatch("circular history row mapping is malformed")
  if chronology.shape[:2] != (1, buffer.max_length):
    raise BranchStateMismatch("circular history chronology shape differs")
  if (
    str(chronology.dtype) != state.get("dtype") or state.get("device") != buffer.device
  ):
    raise BranchStateMismatch("circular history chronology dtype/device differs")
  if buffer._buffer is not None and buffer._buffer.shape[2:] != chronology.shape[2:]:
    raise BranchStateMismatch("circular history trailing shape differs")


def _translate_origin(
  name: str, source: Any, source_origin: Any, destination_origin: Any, robot: Any
) -> torch.Tensor:
  value = _tensor_required(source, name).clone()
  delta = _tensor_required(destination_origin, "destination origin") - _tensor_required(
    source_origin, "source origin"
  )
  if name == "mocap_pos":
    if value.ndim != 2 or value.shape[-1] != 3:
      raise BranchStateMismatch("mocap position layout is malformed")
    value += delta
    return value
  indexing = getattr(robot.data, "indexing", None)
  addresses = getattr(indexing, "free_joint_q_adr", None)
  if addresses is None:
    raise BranchStateMismatch("robot free-joint qpos addresses are missing")
  addresses_tensor = _tensor_required(addresses, "free joint qpos addresses").reshape(
    -1
  )
  if addresses_tensor.numel() < 3:
    raise BranchStateMismatch("robot free-joint qpos addresses are malformed")
  value[addresses_tensor[:3]] += delta
  return value
