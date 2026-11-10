"""CPU/gloo data-parallel invariants of the flow trainer.

Every test here runs on CPU with the gloo backend, because the invariants are
arithmetic and must hold without a card: the per-rank microbatch that keeps the
global batch fixed, the disjoint equal-length sampler shards, the reduction that
makes rank 0's metric row the group mean, and the world-size invariant that a
resume enforces.  ``WORLD_SIZE == 1`` is checked to be untouched: no process
group, no wrapper, and the historical sampler selection.
"""

import json
import os
import socket
import warnings
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed import (
  GradBucket,
  all_gather_object,
  destroy_process_group,
  is_initialized,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, Sampler

from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
from mjlab.tasks.tracking.diffusion.training_config import TrainingConfig
from mjlab.tasks.tracking.flow_matching import (
  FlowTrainer,
  FlowTrainerError,
  FlowTrainingConfig,
  epoch_shard,
  initialize_distributed,
  per_rank_microbatch,
)

MARKS = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
"""Environment variables a launcher sets for one data-parallel worker."""


class WindowSet(Dataset[Tensor]):
  """Deterministic clean token windows with a fixed dataset identity."""

  def __init__(self, count: int, *, weights: Sequence[float] | None = None) -> None:
    generator = torch.Generator().manual_seed(23)
    self.tokens = torch.randn((count, 41, 167), generator=generator)
    self.dataset_identity = {
      "directory": "flow-ddp-fixture",
      "assignments_hash": "a" * 64,
      "split_coverage": {"train": count, "validation": 0, "test": 0},
      "projection_hashes": {"matrix": "b" * 64, "statistics": "c" * 64},
    }
    self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)

  def __len__(self) -> int:
    return len(self.tokens)

  def __getitem__(self, index: int) -> Tensor:
    return self.tokens[index]


class ConstantDenoiser(nn.Module):
  """Predict one learnable token for every row.

  The output ignores the noised input, so a batch loss depends only on which
  windows the batch holds, not on the level draw.  That is what makes a two-rank
  update and a single-process update comparable without reproducing their RNG
  streams: the noise differs, the batch content does not.
  """

  def __init__(self) -> None:
    super().__init__()
    self.value = nn.Parameter(torch.zeros((1, 41, 167)))

  def forward(self, tokens: Tensor, step_ids: Tensor) -> Tensor:
    del step_ids
    return self.value.expand(tokens.shape[0], -1, -1)


class FixedSampler(Sampler[int]):
  """Yield one explicit window order, so two runs consume identical batches."""

  def __init__(self, indices: Sequence[int]) -> None:
    self.indices = [int(index) for index in indices]

  def __iter__(self) -> Iterator[int]:
    return iter(self.indices)

  def __len__(self) -> int:
    return len(self.indices)


def _equality_config() -> FlowTrainingConfig:
  return FlowTrainingConfig(
    training=TrainingConfig(
      epochs=1,
      max_updates=1,
      effective_batch_size=512,
      microbatch_size=128,
      gradient_accumulation_steps=4,
      mixed_precision="fp32",
      warmup_updates=0,
      eval_split="validation",
      seed=0,
    )
  )


def _short_config() -> FlowTrainingConfig:
  """A one-update recipe whose per-rank microbatch is two windows."""
  return FlowTrainingConfig(
    training=TrainingConfig(
      epochs=1,
      max_updates=1,
      effective_batch_size=8,
      microbatch_size=4,
      gradient_accumulation_steps=2,
      mixed_precision="fp32",
      warmup_updates=0,
      eval_split="validation",
      seed=0,
    )
  )


class RecordingTrainer(FlowTrainer):
  """Record the gradient each update applies.

  The step post-hook fires after ``optimizer.step()`` and before ``zero_grad``,
  so the recorded tensor is exactly the divided, clipped gradient the update
  consumed, without reimplementing the update in the test.
  """

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.update_gradients: list[dict[str, Tensor]] = []
    self.optimizer.register_step_post_hook(self._record_gradient)

  def _record_gradient(
    self, optimizer: torch.optim.Optimizer, args: Any, kwargs: Any
  ) -> None:
    del optimizer, args, kwargs
    self.update_gradients.append(
      {
        name: parameter.grad.detach().clone()
        for name, parameter in self.module.named_parameters()
        if parameter.grad is not None
      }
    )


class FixedOrderTrainer(RecordingTrainer):
  """Recording trainer whose batches follow one explicit window order."""

  def __init__(self, order: Sequence[int], **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._order = [int(index) for index in order]

  def _sampler(self, epoch: int) -> Sampler[int]:
    del epoch
    return FixedSampler(self._order)


def _trainer(
  output: Path,
  *,
  windows: int = 512,
  config: FlowTrainingConfig | None = None,
  weights: Sequence[float] | None = None,
  order: Sequence[int] | None = None,
  resume: Path | None = None,
) -> RecordingTrainer:
  torch.manual_seed(0)
  trainer_class = RecordingTrainer if order is None else FixedOrderTrainer
  extra = {} if order is None else {"order": order}
  return trainer_class(
    config=config if config is not None else _equality_config(),
    contract=DiffusionContract(),
    model=ConstantDenoiser(),
    train_dataset=WindowSet(windows, weights=weights),
    eval_datasets={},
    output_dir=output,
    device="cpu",
    resume=resume,
    **extra,
  )


def _count_gradient_reduces(trainer: RecordingTrainer) -> list[int]:
  """Count the DDP gradient all-reduces of one run.

  A comm hook is called once per bucket reduction, so the count is the number of
  times the group paid for a gradient all-reduce.  ``default_hooks.allreduce_hook``
  is the hook-free reduction, so the recorded numbers stay comparable.
  """
  model = trainer.model
  assert isinstance(model, DistributedDataParallel)
  calls: list[int] = []
  # Local import: torch's comm-hook package raises five FutureWarnings on
  # import, and nothing else in this suite uses it.
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

  def hook(state: Any, bucket: GradBucket) -> torch.futures.Future[Tensor]:
    calls.append(1)
    return default_hooks.allreduce_hook(state, bucket)

  model.register_comm_hook(state=None, hook=hook)
  return calls


def _sampler_indices(trainer: FlowTrainer, epoch: int) -> list[int]:
  """Return one epoch's sampled window indices.

  A data-parallel trainer always returns a shard sampler; only the
  single-process unweighted path returns ``None`` and lets the loader shuffle.
  """
  sampler = trainer._sampler(epoch)
  assert sampler is not None
  return [int(index) for index in sampler]


def _parameter(trainer: FlowTrainer) -> Tensor:
  """Return the learnable token of the constant denoiser."""
  return dict(trainer.module.named_parameters())["value"]


def _update_losses(path: Path) -> list[float]:
  """Return the per-update losses rank 0 wrote to ``metrics.jsonl``."""
  if not path.is_file():
    return []
  rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
  return [float(row["loss"]) for row in rows if row["kind"] == "update"]


def _free_port() -> int:
  with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    return int(probe.getsockname()[1])


def _activate_rank(rank: int, world_size: int, port: int) -> None:
  """Export the variables one data-parallel worker receives from its launcher."""
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["RANK"] = str(rank)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)


def _balanced_order(world_size: int, *, windows: int, microbatch: int) -> list[int]:
  """Interleave equal shards so each global micro-batch holds one piece per rank.

  ``world_size`` ranks consume their shards in order, so global micro-batch ``i``
  is the union of every rank's ``i``-th piece.  A single process that consumes
  this order in micro-batches of ``microbatch * world_size`` windows sees the
  same global batches the ranks produce together.
  """
  shards = [
    epoch_shard(
      dataset_size=windows,
      sample_count=windows,
      epoch=0,
      seed=0,
      rank=rank,
      world_size=world_size,
    )
    for rank in range(world_size)
  ]
  piece = microbatch
  blocks = windows // (microbatch * world_size)
  return [
    int(index)
    for block in range(blocks)
    for shard in shards
    for index in shard[block * piece : (block + 1) * piece]
  ]


@pytest.fixture(autouse=True)
def without_launcher_environment(monkeypatch: pytest.MonkeyPatch):
  """Remove any launcher variable so the parent process stays single-process."""
  for name in MARKS:
    monkeypatch.delenv(name, raising=False)


def test_per_rank_microbatch_holds_the_global_batch() -> None:
  for world_size, expected in ((1, 128), (2, 64), (4, 32), (8, 16)):
    assert per_rank_microbatch(512, 4, world_size) == expected
  with pytest.raises(FlowTrainerError, match="not divisible"):
    per_rank_microbatch(512, 3, 2)
  with pytest.raises(FlowTrainerError, match="below one"):
    per_rank_microbatch(0, 4, 2)
  with pytest.raises(FlowTrainerError, match="world size must be positive"):
    per_rank_microbatch(512, 4, 0)


def test_epoch_shard_is_disjoint_equal_length_and_reproducible() -> None:
  shards = [
    epoch_shard(
      dataset_size=64, sample_count=64, epoch=0, seed=0, rank=rank, world_size=2
    )
    for rank in range(2)
  ]
  first, second = (set(shard.tolist()) for shard in shards)
  assert len(shards[0]) == len(shards[1]) == 32
  assert first.isdisjoint(second)
  # 64 divides by 2, so the union is exactly the dataset.
  assert first | second == set(range(64))
  # Same epoch and seed reproduce the same shard on every rank.
  assert torch.equal(
    shards[0],
    epoch_shard(
      dataset_size=64, sample_count=64, epoch=0, seed=0, rank=0, world_size=2
    ),
  )
  # A different epoch permutes the shard, so the dropped tail rotates.
  later = [
    epoch_shard(
      dataset_size=65, sample_count=65, epoch=epoch, seed=0, rank=rank, world_size=2
    )
    for epoch in (0, 1)
    for rank in range(2)
  ]
  assert len(later[0]) == len(later[1]) == 32
  dropped = [
    (set(range(65)) - set(pair[0].tolist()) - set(pair[1].tolist())).pop()
    for pair in (later[0:2], later[2:4])
  ]
  assert dropped[0] != dropped[1]
  with pytest.raises(FlowTrainerError, match="smaller than the world size"):
    epoch_shard(dataset_size=64, sample_count=1, epoch=0, seed=0, rank=0, world_size=2)


def test_world_size_one_keeps_the_historical_sampler_and_no_group(
  tmp_path: Path,
) -> None:
  context = initialize_distributed("cpu")
  assert (context.world_size, context.rank, context.backend) == (1, 0, None)
  assert not context.is_distributed
  assert not is_initialized()

  uniform = _trainer(tmp_path / "uniform", windows=16)
  assert uniform.model is uniform.module
  assert per_rank_microbatch(512, 4, 1) == uniform.microbatch_size
  # The unweighted run keeps the shuffle path: no sampler at all.
  assert uniform._sampler(0) is None

  weighted = _trainer(tmp_path / "weighted", windows=8, weights=[1.0] * 4 + [2.0] * 4)
  sampler = weighted._sampler(0)
  assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
  weights = weighted._sampling_weights()
  assert weights is not None
  # The selection reproduces the historical one: same weights, same count, and a
  # generator seeded seed + 2 with no rank term.
  reference = torch.utils.data.WeightedRandomSampler(
    weights.tolist(),
    num_samples=8,
    replacement=True,
    generator=torch.Generator(device="cpu").manual_seed(weighted.training.seed + 2),
  )
  assert list(sampler) == list(reference)


def _worker_shards(
  rank: int, world_size: int, port: int, output_dir: str, windows: int
) -> None:
  _activate_rank(rank, world_size, port)
  try:
    output = Path(output_dir)
    plain = _trainer(
      output / f"sampler-rank{rank}", windows=windows, config=_short_config()
    )
    epoch0 = _sampler_indices(plain, 0)
    epoch1 = _sampler_indices(plain, 1)
    weighted = _trainer(
      output / f"weighted-rank{rank}",
      windows=windows,
      config=_short_config(),
      weights=[1.0] * (windows // 2) + [3.0] * (windows // 2),
    )
    weighted_draws = _sampler_indices(weighted, 0)
    record = {
      "epoch0": epoch0,
      "epoch1": epoch1,
      "weighted": weighted_draws,
    }
    gathered: list[dict[str, list[int]] | None] = [None] * world_size
    all_gather_object(gathered, record)
    trained = _trainer(
      output / f"run-rank{rank}", windows=windows, config=_short_config()
    )
    trained.train()
    # Every rank resumes the one file rank 0 wrote, and each takes its own RNG
    # entry: a stream one rank drew cannot be restored from another's entry.
    checkpoint = output / "run-rank0" / "checkpoint-last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed = _trainer(
      output / f"resume-rank{rank}",
      windows=windows,
      config=_short_config(),
      resume=checkpoint,
    )
    entry = payload["rng_state"]["rank_generators"][rank]
    assert resumed.global_step == payload["global_step"] == 1
    assert torch.equal(resumed.noise_generator.get_state(), entry["noise_generator"])
    assert torch.equal(
      resumed.sampler_generator.get_state(), entry["sampler_generator"]
    )
    if rank == 0:
      (output / "shards.json").write_text(
        json.dumps(gathered, sort_keys=True), encoding="utf-8"
      )
  finally:
    destroy_process_group()


def test_data_parallel_sampler_shards_and_world_size_invariant(
  tmp_path: Path,
) -> None:
  world_size = 2
  windows = 64
  port = _free_port()
  mp.spawn(
    _worker_shards,
    args=(world_size, port, str(tmp_path), windows),
    nprocs=world_size,
    join=True,
  )
  gathered = json.loads((tmp_path / "shards.json").read_text())
  lengths = {len(record["epoch0"]) for record in gathered}
  assert lengths == {windows // world_size}
  sets = [set(record["epoch0"]) for record in gathered]
  assert sets[0].isdisjoint(sets[1])
  assert sets[0] | sets[1] == set(range(windows))
  # Both ranks built the same permutation, so the parent reproduces each shard.
  for rank, observed in enumerate(gathered):
    expected = epoch_shard(
      dataset_size=windows,
      sample_count=windows,
      epoch=0,
      seed=0,
      rank=rank,
      world_size=world_size,
    )
    assert observed["epoch0"] == expected.tolist()
    assert observed["epoch1"] != observed["epoch0"]
  # A weighted draw never leaves the rank's shard, and the lengths still match.
  for rank, record in enumerate(gathered):
    assert len(record["weighted"]) == windows // world_size
    assert set(record["weighted"]) <= sets[rank]

  # The checkpoint is written by rank 0 and holds one RNG state per rank.
  checkpoint = tmp_path / "run-rank0" / "checkpoint-last.pt"
  payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
  assert payload["rng_state"]["world_size"] == world_size
  assert len(payload["rng_state"]["rank_generators"]) == world_size
  assert set(payload["model"]) == {"value"}
  # A single-process resume of a two-rank checkpoint is refused, and the error
  # names the stored world size.
  with pytest.raises(FlowTrainerError, match="world_size 2"):
    _trainer(
      tmp_path / "resume-one-rank",
      windows=windows,
      config=_short_config(),
      resume=checkpoint,
    )


def _worker_equal_batch(
  rank: int, world_size: int, port: int, output_dir: str, windows: int
) -> None:
  _activate_rank(rank, world_size, port)
  try:
    trainer = _trainer(Path(output_dir) / f"rank{rank}", windows=windows)
    reduces = _count_gradient_reduces(trainer)
    trainer.train()
    torch.save(
      {
        "gradients": trainer.update_gradients,
        "parameter": _parameter(trainer).detach().clone(),
        "ema": trainer.ema.shadow["value"].detach().clone(),
        "reduces": len(reduces),
      },
      Path(output_dir) / f"rank{rank}.pt",
    )
  finally:
    destroy_process_group()


def test_equal_global_batch_matches_single_process(tmp_path: Path) -> None:
  world_size = 2
  windows = 512
  # 2 ranks x microbatch 64 x accumulation 4 against 1 process x microbatch 128
  # x accumulation 4: the same window sets, so the loss and the accumulated
  # gradient must agree even though the noise draws differ.
  order = _balanced_order(world_size, windows=windows, microbatch=64)
  reference = _trainer(tmp_path / "single", windows=windows, order=order)
  reference.train()
  reference_losses = _update_losses(tmp_path / "single" / "metrics.jsonl")

  port = _free_port()
  mp.spawn(
    _worker_equal_batch,
    args=(world_size, port, str(tmp_path), windows),
    nprocs=world_size,
    join=True,
  )
  shards = [
    torch.load(tmp_path / f"rank{rank}.pt", map_location="cpu", weights_only=False)
    for rank in range(world_size)
  ]
  # Rank 0 alone writes rows, so its single update row is the group mean.
  group_losses = _update_losses(tmp_path / "rank0" / "metrics.jsonl")
  assert len(reference_losses) == 1
  assert len(group_losses) == 1
  assert reference_losses[0] == pytest.approx(group_losses[0], abs=1e-6)

  expected = reference.update_gradients[0]["value"]
  assert len(reference.update_gradients) == 1
  # Four micro-batches and one update: the group pays one all-reduce, not four.
  assert [shard["reduces"] for shard in shards] == [1, 1]
  for shard in shards:
    assert len(shard["gradients"]) == 1
    observed = shard["gradients"][0]["value"]
    # The gradient magnitude is O(1) here, so a missed or doubled all-reduce is
    # far outside this window while float ordering stays inside it.
    assert (observed - expected).abs().max().item() <= 1e-6
    # The update step is deterministic, so the parameters must agree too.
    difference = (shard["parameter"] - _parameter(reference)).abs().max().item()
    assert difference <= 1e-6
    # The EMA is built on the unwrapped module, so its shadow is the
    # single-process shadow rather than a copy of any wrapper state.
    ema_difference = (shard["ema"] - reference.ema.shadow["value"]).abs().max().item()
    assert ema_difference <= 1e-6
