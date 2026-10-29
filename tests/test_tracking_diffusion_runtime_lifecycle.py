"""CPU regressions for the concrete batched X2 termination lifecycle."""

from types import SimpleNamespace

import numpy as np
import torch

from mjlab.tasks.tracking.diffusion.adapter import AdapterError
from mjlab.tasks.tracking.diffusion.runtime_x2 import (
  InitialStateSnapshot,
  X2MultiMotionEnvironment,
)


def test_reference_guards_are_suspended_and_time_outs_survive() -> None:
  calls = []
  terms = {}
  for name in ("anchor_pos", "anchor_ori", "ee_body_pos", "time_out"):

    def original(env, *, marker):
      calls.append(marker)
      return torch.ones(3, dtype=torch.bool)

    terms[name] = SimpleNamespace(func=original, params={"marker": name})
  manager = SimpleNamespace(active_terms=list(terms), get_term_cfg=terms.__getitem__)
  runtime = X2MultiMotionEnvironment.__new__(X2MultiMotionEnvironment)
  runtime.num_envs = 3
  runtime.env = SimpleNamespace(termination_manager=manager, device="cpu")
  runtime._install_termination_masks()
  runtime.set_active_env_ids(np.array([0, 2]))
  for name, cfg in terms.items():
    actual = cfg.func(runtime.env, **cfg.params)
    # Every reference guard stays suspended for the whole rollout; only the
    # task time-out may end an active row.
    expected = [name == "time_out", False, name == "time_out"]
    torch.testing.assert_close(actual, torch.tensor(expected))
  assert calls == list(terms)
  runtime.set_active_env_ids(np.array([1]))
  actual = terms["time_out"].func(runtime.env, **terms["time_out"].params)
  torch.testing.assert_close(actual, torch.tensor([False, True, False]))


def test_batched_snapshot_signature_is_scene_origin_independent() -> None:
  runtime = X2MultiMotionEnvironment.__new__(X2MultiMotionEnvironment)
  runtime.device = "cpu"
  runtime.env = SimpleNamespace(
    scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]]))
  )
  runtime.robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.0, 2.0, 0.5], [3.0, 6.0, 0.5]]),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
      joint_pos=torch.zeros((2, 2)),
      joint_vel=torch.zeros((2, 2)),
    )
  )
  runtime.command = SimpleNamespace(time_steps=torch.tensor([12, 12]))
  first = runtime.capture_initial_state_batch(np.array([0]))[0]
  second = runtime.capture_initial_state_batch(np.array([1]))[0]
  np.testing.assert_array_equal(first.signature, second.signature)


def test_restore_parity_allows_origin_rounding_but_rejects_state_change() -> None:
  runtime = X2MultiMotionEnvironment.__new__(X2MultiMotionEnvironment)
  runtime._RESTORE_PARITY_ATOL = 1e-6
  runtime.capture_initial_state_batch = lambda _ids: (
    InitialStateSnapshot(0, 0, (1.0000002,)),
  )
  runtime.restore_initial_state_batch(
    np.array([0]), (InitialStateSnapshot(0, 0, (1.0,)),)
  )
  runtime.capture_initial_state_batch = lambda _ids: (
    InitialStateSnapshot(0, 0, (1.00001,)),
  )
  try:
    runtime.restore_initial_state_batch(
      np.array([0]), (InitialStateSnapshot(0, 0, (1.0,)),)
    )
  except AdapterError:
    pass
  else:
    raise AssertionError("restore parity accepted a state change above tolerance")


def test_idle_rows_hold_and_active_rows_stop_at_their_last_frame() -> None:
  runtime = X2MultiMotionEnvironment.__new__(X2MultiMotionEnvironment)
  runtime.num_envs = 3
  runtime.device = "cpu"
  runtime._row_ids = torch.arange(3)
  runtime._active_rows = np.array([True, False, True])
  frames = torch.tensor([0, 0, 1])

  def update(ids):
    frames[ids] += 1

  runtime.command = SimpleNamespace(
    _update_command=update,
    time_steps=frames,
    row_lengths=torch.tensor([2, 2, 2]),
  )
  runtime._install_reference_advance_mask()
  # Row 0 advances, row 1 is idle, row 2 is already on its last frame.
  runtime.command._update_command()
  torch.testing.assert_close(frames, torch.tensor([1, 0, 1]))
  # The reference never wraps past the last frame, so no row is resampled.
  runtime.command._update_command()
  torch.testing.assert_close(frames, torch.tensor([1, 0, 1]))
  runtime.command._update_command(torch.tensor([1]))
  torch.testing.assert_close(frames, torch.tensor([1, 0, 1]))
