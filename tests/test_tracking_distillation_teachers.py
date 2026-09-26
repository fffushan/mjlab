"""Frozen teacher inference: determinism, routing, freezing, and rejection paths."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from rsl_rl.modules import EmpiricalNormalization
from tracking_distillation_fixtures import make_actor, write_checkpoint

from mjlab.tasks.tracking.distillation.config import (
  ActorArchitecture,
  DistillationError,
  load_actor_state_dict,
)
from mjlab.tasks.tracking.distillation.teachers import (
  FrozenTeacher,
  TeacherBank,
  build_actor_model,
  build_frozen_teacher,
  load_frozen_teacher,
)

OBS_DIM = 24
ACTION_DIM = 3


def make_architecture(
  obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM
) -> ActorArchitecture:
  return ActorArchitecture(
    class_name="MLPModel",
    hidden_dims=(8, 4),
    activation="elu",
    obs_normalization=True,
    obs_groups=("actor",),
    distribution_class_name="GaussianDistribution",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "std_type": "scalar",
      "init_std": 1.0,
    },
    obs_dim=obs_dim,
    action_dim=action_dim,
  )


def make_bank(
  *, seeds: tuple[int, ...] = (0, 1), ids: tuple[str, ...] | None = None
) -> TeacherBank:
  arch = make_architecture()
  if ids is None:
    ids = tuple(f"teacher_{index}" for index in range(len(seeds)))
  teachers = []
  for teacher_id, seed in zip(ids, seeds, strict=True):
    model = make_actor(OBS_DIM, ACTION_DIM, seed=seed)
    teachers.append(build_frozen_teacher(teacher_id, model.state_dict(), arch))
  return TeacherBank(teachers, device="cpu")


def sample_observations(rows: int = 4, seed: int = 7) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  return torch.randn(rows, OBS_DIM, generator=generator) * 2.0 + 0.5


def test_frozen_teacher_matches_normalized_manual_forward() -> None:
  arch = make_architecture()
  model = make_actor(OBS_DIM, ACTION_DIM, seed=3)
  teacher = build_frozen_teacher("solo", model.state_dict(), arch)
  observations = sample_observations(6)

  expected = model.mlp(model.obs_normalizer(observations))
  actual = teacher.label(observations)

  torch.testing.assert_close(actual, expected)
  assert teacher.obs_dim == OBS_DIM
  assert teacher.action_dim == ACTION_DIM


def test_labels_are_deterministic_and_do_not_sample_noise() -> None:
  teacher = build_frozen_teacher(
    "solo", make_actor(OBS_DIM, ACTION_DIM, seed=5).state_dict(), make_architecture()
  )
  observations = sample_observations(8)

  first = teacher.label(observations)
  second = teacher.label(observations.clone())

  assert torch.equal(first, second)
  assert not first.requires_grad


def test_train_mode_requests_are_ignored_and_statistics_stay_frozen() -> None:
  bank = make_bank()
  teacher = bank.teacher(0)
  observations = sample_observations()
  before = {key: value.clone() for key, value in teacher.state_dict().items()}
  expected = teacher.label(observations)

  bank.train()
  assert bank.training is False
  assert teacher.training is False
  assert teacher.model.training is False
  assert teacher.model.obs_normalizer.training is False

  for _ in range(3):
    assert torch.equal(
      bank.label(torch.zeros(4, dtype=torch.int64), observations), expected
    )

  # Even a direct normalization update cannot move the frozen statistics while
  # the teacher is in eval mode.
  normalizer = cast(EmpiricalNormalization, teacher.model.obs_normalizer)
  normalizer.update(observations * 100.0)
  for key, value in teacher.state_dict().items():
    assert torch.equal(value, before[key]), key


def test_labeling_refuses_a_model_switched_back_to_train_mode() -> None:
  bank = make_bank()
  teacher = bank.teacher(0)
  teacher.model.train()

  with pytest.raises(RuntimeError, match="is in train mode"):
    teacher.label(sample_observations(2))
  with pytest.raises(RuntimeError, match="is in train mode"):
    bank.label(torch.zeros(2, dtype=torch.int64), sample_observations(2))


@pytest.mark.parametrize("permutation", [(0, 1, 0, 1), (1, 1, 0, 0), (1, 0, 1, 0)])
def test_mixed_codes_route_to_the_right_teacher_and_preserve_order(
  permutation: tuple[int, ...],
) -> None:
  bank = make_bank()
  observations = sample_observations(len(permutation), seed=11)
  codes = torch.tensor(permutation, dtype=torch.int64)

  actual = bank.label(codes, observations)

  for row, code in enumerate(permutation):
    expected = bank.teacher(code).label(observations[row : row + 1])
    torch.testing.assert_close(actual[row : row + 1], expected)
  assert not torch.allclose(
    bank.teacher(0).label(observations), bank.teacher(1).label(observations)
  )


def test_each_needed_teacher_is_evaluated_once_per_batch(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bank = make_bank()
  calls: list[int] = []
  for code in (0, 1):
    teacher = bank.teacher(code)
    original = teacher.label

    def counting(observations, original=original, code=code):
      calls.append(code)
      return original(observations)

    monkeypatch.setattr(teacher, "label", counting)

  bank.label(torch.tensor([1, 1, 0, 1], dtype=torch.int64), sample_observations(4))

  assert sorted(calls) == [0, 1]


def test_bank_device_follows_registered_teacher_parameters() -> None:
  bank = make_bank()
  observations = sample_observations(2)

  assert bank.device == torch.device("cpu")
  assert bank.device == bank.teacher(0).device
  actions = bank.label(torch.zeros(2, dtype=torch.int64), observations)
  assert actions.device == torch.device("cpu")

  # A transfer must update the reported device, not leave a stale constructor
  # string that rejects correctly placed codes/observations.
  bank.to("meta")
  assert bank.device == torch.device("meta")
  assert bank.device == bank.teacher(0).device
  with pytest.raises(
    ValueError, match="teacher_ids are on cpu but the bank is on meta"
  ):
    bank.label(torch.zeros(2, dtype=torch.int64), observations)


def test_empty_batch_returns_empty_actions() -> None:
  bank = make_bank()

  actions = bank.label(torch.zeros(0, dtype=torch.int64), torch.zeros(0, OBS_DIM))

  assert actions.shape == (0, ACTION_DIM)
  assert actions.device == bank.device


def test_bank_rejects_unknown_codes_and_bad_observations() -> None:
  bank = make_bank()

  with pytest.raises(ValueError, match="out of range"):
    bank.label(torch.tensor([2], dtype=torch.int64), sample_observations(1))
  with pytest.raises(ValueError, match="out of range"):
    bank.label(torch.tensor([-1], dtype=torch.int64), sample_observations(1))
  with pytest.raises(ValueError, match="integer tensor"):
    bank.label(torch.tensor([0.0]), sample_observations(1))
  with pytest.raises(ValueError, match="1-D"):
    bank.label(torch.zeros((1, 1), dtype=torch.int64), sample_observations(1))
  with pytest.raises(ValueError, match="rows"):
    bank.label(torch.zeros(2, dtype=torch.int64), sample_observations(1))
  with pytest.raises(ValueError, match="expects observations of shape"):
    bank.label(torch.zeros(1, dtype=torch.int64), torch.zeros(1, OBS_DIM + 1))
  with pytest.raises(ValueError, match="the bank is on"):
    bank.label(
      torch.zeros(1, dtype=torch.int64), torch.zeros(1, OBS_DIM, device="meta")
    )
  with pytest.raises(ValueError, match="Unknown teacher id"):
    bank.code("absent")


def test_bank_rejects_duplicate_or_incompatible_teachers() -> None:
  arch = make_architecture()
  first = build_frozen_teacher(
    "dup", make_actor(OBS_DIM, ACTION_DIM, seed=0).state_dict(), arch
  )
  second = build_frozen_teacher(
    "dup", make_actor(OBS_DIM, ACTION_DIM, seed=1).state_dict(), arch
  )
  with pytest.raises(ValueError, match="unique"):
    TeacherBank([first, second])

  other_arch = make_architecture(obs_dim=OBS_DIM, action_dim=2)
  third = build_frozen_teacher(
    "small", make_actor(OBS_DIM, 2, seed=2).state_dict(), other_arch
  )
  with pytest.raises(DistillationError, match="share"):
    TeacherBank([first, third])
  with pytest.raises(ValueError, match="at least one teacher"):
    TeacherBank([])


def test_teacher_rejects_wrong_observation_shape() -> None:
  teacher = build_frozen_teacher(
    "solo", make_actor(OBS_DIM, ACTION_DIM, seed=0).state_dict(), make_architecture()
  )

  with pytest.raises(ValueError, match="expects observations of shape"):
    teacher.label(torch.zeros(1, OBS_DIM + 1))
  with pytest.raises(ValueError, match="expects observations of shape"):
    teacher.label(torch.zeros(OBS_DIM))
  with pytest.raises(TypeError, match="must be a Tensor"):
    teacher.label([0.0] * OBS_DIM)  # type: ignore[arg-type]


def test_build_frozen_teacher_rejects_mismatched_state_dict() -> None:
  arch = make_architecture()
  model = make_actor(OBS_DIM + 1, ACTION_DIM, seed=0)

  with pytest.raises(DistillationError, match="does not match the resolved actor"):
    build_frozen_teacher("solo", model.state_dict(), arch)


def test_multi_group_actor_is_rejected() -> None:
  arch = make_architecture()
  multi = ActorArchitecture(
    class_name=arch.class_name,
    hidden_dims=arch.hidden_dims,
    activation=arch.activation,
    obs_normalization=arch.obs_normalization,
    obs_groups=("actor", "critic"),
    distribution_class_name=arch.distribution_class_name,
    distribution_cfg=arch.distribution_cfg,
    obs_dim=arch.obs_dim,
    action_dim=arch.action_dim,
  )

  with pytest.raises(DistillationError, match="flat"):
    build_actor_model(multi)


def test_load_frozen_teacher_reads_a_checkpoint(tmp_path: Path) -> None:
  model = make_actor(OBS_DIM, ACTION_DIM, seed=9)
  checkpoint = tmp_path / "model_5.pt"
  write_checkpoint(checkpoint, model)
  arch = make_architecture()

  teacher = load_frozen_teacher("tiny", checkpoint, arch)
  observations = sample_observations(2)

  assert isinstance(teacher, FrozenTeacher)
  expected = make_actor(OBS_DIM, ACTION_DIM, seed=9)
  torch.testing.assert_close(
    teacher.label(observations), expected.mlp(expected.obs_normalizer(observations))
  )
  assert set(load_actor_state_dict(checkpoint)) == set(model.state_dict())
