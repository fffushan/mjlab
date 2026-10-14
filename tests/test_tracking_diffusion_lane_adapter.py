"""Focused tests for the D5 controller-lane token adapter.

The controller lane is compiled for one token layout: 135 hybrid state channels
followed by the 32 raw latent channels.  A reduced representation is a channel
selection of that layout, so an adapter exists and must be exact on every channel
the model reads.  These tests pin the channel map, the statistics placement, the
pass-through rule and the fail-closed cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from mjlab.tasks.tracking.diffusion.contract import DEFAULT_CONTRACT
from mjlab.tasks.tracking.diffusion.lane_adapter import (
  LANE_LATENT_DIMENSION,
  LANE_STATE_DIMENSION,
  LANE_TOKEN_DIMENSION,
  LaneAdapterError,
  LaneTokenAdapter,
  describe_lane_adapter,
  lane_dropped_fields,
  lane_gather_index,
  lane_state_statistics,
)
from mjlab.tasks.tracking.diffusion.state import (
  HYBRID_STATE,
  LOCAL_KINEMATIC_STATE,
  hybrid_to_local_kinematic,
)


class _DoubleCore(nn.Module):
  """A stand-in model boundary: it doubles the channels it is given."""

  def forward(self, tokens: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    del step_ids
    return tokens * 2.0


def _frozen_lane_contract() -> dict[str, tuple[int, int]]:
  return {name: (item.start, item.stop) for name, item in HYBRID_STATE.slices.items()}


def test_frozen_layout_needs_no_adapter() -> None:
  report = describe_lane_adapter(DEFAULT_CONTRACT.state_slices)
  assert report["required"] is False
  assert report["lane_token_dimension"] == LANE_TOKEN_DIMENSION == 167
  assert report["model_token_dimension"] == 167
  assert report["dropped_lane_state_fields"] == []
  assert report["dropped_lane_state_channels"] == []
  assert report["gather_index"] == list(range(167))


def test_reduced_layout_gather_index_is_the_hybrid_channel_selection() -> None:
  report = describe_lane_adapter(LOCAL_KINEMATIC_STATE.slices)
  assert report["required"] is True
  assert report["lane_state_dimension"] == 135
  assert report["model_state_dimension"] == 129
  assert report["model_token_dimension"] == 161
  assert report["dropped_lane_state_fields"] == [
    "root_position",
    "root_linear_velocity",
  ]
  assert report["dropped_lane_state_channels"] == [0, 1, 2, 9, 10, 11]
  index = np.asarray(report["gather_index"], dtype=np.int64)
  assert index.size == 161
  assert index.tolist() == list(range(3, 9)) + list(range(12, 135)) + list(
    range(135, 167)
  )
  assert lane_dropped_fields(LOCAL_KINEMATIC_STATE.slices) == (
    "root_position",
    "root_linear_velocity",
  )


def test_gather_index_reproduces_the_state_transform() -> None:
  """The channel map must agree with the real hybrid-to-reduced transform."""
  rng = np.random.default_rng(20261014)
  hybrid = rng.normal(size=(9, LANE_STATE_DIMENSION))
  index = lane_gather_index(LOCAL_KINEMATIC_STATE.slices)
  state_channels = index[: LOCAL_KINEMATIC_STATE.dimension]
  selected = hybrid[:, state_channels]
  expected = hybrid_to_local_kinematic(hybrid)
  assert selected.shape == expected.shape == (9, 129)
  assert np.array_equal(selected, expected)
  # The latent block keeps its position: the last 32 lane channels.
  assert index[LOCAL_KINEMATIC_STATE.dimension :].tolist() == list(range(135, 167))


def test_lane_state_statistics_place_the_reduced_values_and_fill_the_rest() -> None:
  mean = np.linspace(0.1, 0.2, num=LOCAL_KINEMATIC_STATE.dimension)
  std = np.linspace(0.01, 0.02, num=LOCAL_KINEMATIC_STATE.dimension)
  lane_mean, lane_std, dropped = lane_state_statistics(
    mean, std, LOCAL_KINEMATIC_STATE.slices
  )
  assert lane_mean.shape == lane_std.shape == (LANE_STATE_DIMENSION,)
  assert dropped == ("root_position", "root_linear_velocity")
  index = lane_gather_index(LOCAL_KINEMATIC_STATE.slices)
  state_channels = index[: LOCAL_KINEMATIC_STATE.dimension]
  assert np.array_equal(lane_mean[state_channels], mean)
  assert np.array_equal(lane_std[state_channels], std)
  assert lane_mean[[0, 1, 2, 9, 10, 11]].tolist() == [0.0] * 6
  assert lane_std[[0, 1, 2, 9, 10, 11]].tolist() == [1.0] * 6


def test_lane_token_adapter_moves_only_the_model_channels() -> None:
  adapter = LaneTokenAdapter(_DoubleCore(), LOCAL_KINEMATIC_STATE.slices)
  lane_tokens = torch.linspace(
    -1.5, 1.5, 2 * 41 * LANE_TOKEN_DIMENSION, dtype=torch.float32
  ).reshape(2, 41, LANE_TOKEN_DIMENSION)
  step_ids = torch.zeros((2, 41, 2), dtype=torch.long)
  output = adapter(lane_tokens, step_ids)
  assert output.shape == lane_tokens.shape
  index = lane_gather_index(LOCAL_KINEMATIC_STATE.slices)
  model_channels = torch.from_numpy(index)
  doubled = lane_tokens * 2.0
  assert torch.equal(
    torch.index_select(output, -1, model_channels),
    torch.index_select(doubled, -1, model_channels),
  )
  dropped = torch.tensor([0, 1, 2, 9, 10, 11], dtype=torch.long)
  assert torch.equal(
    torch.index_select(output, -1, dropped),
    torch.index_select(lane_tokens, -1, dropped),
  )


def test_lane_token_adapter_keeps_the_latent_block_from_the_model() -> None:
  adapter = LaneTokenAdapter(_DoubleCore(), LOCAL_KINEMATIC_STATE.slices)
  lane_tokens = torch.zeros((1, 41, LANE_TOKEN_DIMENSION), dtype=torch.float32)
  lane_tokens[..., LANE_STATE_DIMENSION:] = 3.0
  step_ids = torch.zeros((1, 41, 2), dtype=torch.long)
  output = adapter(lane_tokens, step_ids)
  latent_block = output[..., LANE_STATE_DIMENSION:]
  assert latent_block.shape == (1, 41, LANE_LATENT_DIMENSION)
  assert torch.equal(latent_block, torch.full_like(latent_block, 6.0))


def test_lane_token_adapter_refuses_the_lane_width() -> None:
  adapter = LaneTokenAdapter(_DoubleCore(), LOCAL_KINEMATIC_STATE.slices)
  with pytest.raises(LaneAdapterError):
    adapter(
      torch.zeros((1, 41, LOCAL_KINEMATIC_STATE.dimension + 32)),
      torch.zeros((1, 41, 2), dtype=torch.long),
    )


def test_lane_token_adapter_requires_a_reduced_layout() -> None:
  with pytest.raises(LaneAdapterError):
    LaneTokenAdapter(_DoubleCore(), _frozen_lane_contract())


def test_adapter_refuses_a_field_the_lane_does_not_have() -> None:
  layout = {
    "root_rot6d": (0, 6),
    "root_angular_velocity": (6, 9),
    "body_positions": (9, 69),
    "body_linear_velocities": (69, 129),
    "unknown_field": (129, 135),
  }
  with pytest.raises(LaneAdapterError, match="unknown_field"):
    describe_lane_adapter(layout)


def test_adapter_refuses_a_block_of_the_wrong_width() -> None:
  layout = {
    "root_rot6d": (0, 6),
    "root_angular_velocity": (6, 10),
    "body_positions": (10, 70),
    "body_linear_velocities": (70, 130),
  }
  with pytest.raises(LaneAdapterError, match="width"):
    lane_gather_index(layout)


def test_adapter_refuses_statistics_of_the_wrong_width() -> None:
  with pytest.raises(LaneAdapterError, match="shape"):
    lane_state_statistics(np.zeros(128), np.ones(128), LOCAL_KINEMATIC_STATE.slices)
