"""Seeding and construction recipe shared by the parent and every worker.

Sharded collection depends on two properties that are cheap to state and easy
to get silently wrong: two workers must never share an environment or generator
seed, and a worker must build its environment through the same recipe the
single-process path uses.  These tests pin both, without a simulator.
"""

from __future__ import annotations

import pytest

from mjlab.tasks.tracking.distillation import cohort_setup
from mjlab.tasks.tracking.distillation.cohort_setup import (
  ENV_SEED_STRIDE,
  GENERATOR_SEED_STRIDE,
  CohortSetup,
  worker_env_seed,
  worker_evaluation_seed,
  worker_generator_seed,
)
from mjlab.tasks.tracking.distillation.reset_policy import make_reset_policy


def _setup(tmp_path, *, base_seed: int = 11) -> CohortSetup:
  return CohortSetup(
    manifest=tmp_path / "manifest.yaml",
    repo_root=tmp_path,
    teacher_ids=("tennis_000", "tennis_001"),
    task_id=None,
    phase_policy="uniform",
    reset_policy=make_reset_policy(),
    base_seed=base_seed,
  )


def test_worker_seeds_are_disjoint_deterministic_and_collision_free() -> None:
  """No two workers may share an environment or a generator seed.

  Identical environment seeds would simulate the same instances N times, and
  identical generator seeds would draw the same latents and the same
  teacher/student selections; either one reduces sharded collection to one
  shard's worth of data while every counter still looks correct.  The
  derivation must also be positional, because iteration ``k`` has to reproduce
  the same randomness after an interruption without per-worker RNG state in the
  checkpoint.
  """
  base = 7
  workers = range(8)
  env_seeds = [worker_env_seed(base, index) for index in workers]
  generator_seeds = [
    worker_generator_seed(base, iteration=3, worker_index=index) for index in workers
  ]

  assert len(set(env_seeds)) == 8
  assert len(set(generator_seeds)) == 8
  # Worker 0 is the single-process derivation: the requested seed is unchanged.
  assert env_seeds[0] == base
  assert generator_seeds[0] == base + 3
  assert env_seeds[2] - env_seeds[1] == ENV_SEED_STRIDE
  assert generator_seeds[2] - generator_seeds[1] == GENERATOR_SEED_STRIDE
  # Distinctness is required *within* each domain, and it holds for every
  # iteration (not just the one sampled above).  The two domains deliberately
  # use different strides, but they seed separate RNGs -- the environment's own
  # seeding versus this shard's torch generator -- so an integer coincidence
  # across domains is harmless and is not claimed here.
  assert ENV_SEED_STRIDE != GENERATOR_SEED_STRIDE
  for iteration in (0, 1, 7, 10_000):
    values = [worker_generator_seed(base, iteration, index) for index in workers]
    assert len(set(values)) == 8, iteration
  assert [worker_generator_seed(base, 3, index) for index in workers] == generator_seeds


def test_build_adapter_derives_the_environment_seed_from_the_worker_index(
  monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
  """The worker index, not the caller, decides the environment seed.

  A caller that could pass its own seed would let two shards collide, and the
  collision would be invisible in the run: both would collect valid rows from
  the same states.
  """
  recorded: list[dict] = []

  def fake_factory(cohort, teacher_ids, **kwargs):
    recorded.append({"cohort": cohort, "teacher_ids": tuple(teacher_ids), **kwargs})
    return object()

  monkeypatch.setattr(
    cohort_setup, "make_multi_teacher_distillation_adapter", fake_factory
  )
  monkeypatch.setattr(cohort_setup, "load_manifest", lambda path, root: "manifest")
  monkeypatch.setattr(cohort_setup, "resolve_cohort", lambda manifest: "cohort")
  setup = _setup(tmp_path, base_seed=11)

  setup.build_adapter(device="cpu", num_envs=2, worker_index=0)
  setup.build_adapter(device="cpu", num_envs=2, worker_index=1)

  assert [call["seed"] for call in recorded] == [
    worker_env_seed(11, 0),
    worker_env_seed(11, 1),
  ]
  assert recorded[0]["cohort"] == "cohort"
  assert recorded[0]["teacher_ids"] == ("tennis_000", "tennis_001")
  assert recorded[0]["num_envs"] == 2
  assert recorded[0]["device"] == "cpu"
  assert recorded[0]["phase_policy"] == "uniform"
  # The reset policy the parent resolved is the one every worker applies.
  assert recorded[0]["reset_policy"] == setup.reset_policy


def test_collection_and_evaluation_seeds_stay_disjoint() -> None:
  """The two derived streams must not meet inside any configured run.

  Evaluation draws its own standing/reference decisions, so a seed that also
  appears in the collection stream would let an evaluation replay the resets the
  collection of another iteration had already made.  Both formulas are additive,
  so a container can always be built that makes them meet; what matters is that
  no run this project configures reaches one, and that is what this scans rather
  than asserting.
  """
  from mjlab.scripts.distill import _MAX_WORKERS

  collection = {
    worker_generator_seed(7, iteration, worker)
    for iteration in range(10_001)  # the plan's iteration budget
    for worker in range(_MAX_WORKERS)
  }
  evaluation = {
    worker_evaluation_seed(7, iteration, worker)
    for iteration in range(10_001)
    for worker in range(_MAX_WORKERS)
  }

  assert not (collection & evaluation), (
    "collection and evaluation seeds overlap inside the configured domain"
  )
