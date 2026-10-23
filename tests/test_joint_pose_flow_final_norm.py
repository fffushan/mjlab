"""CPU tests for the final-LayerNorm ablation variant.

The variant adds one ``nn.LayerNorm`` between the shared Transformer's encoder
and its output projection.  These tests pin four properties:

1. The change is exactly the norm.  No other parameter differs, and the shared
   weights are identical when both models are built under the same init seed.
2. ``forward`` normalises before projecting.  The test recomputes the forward
   path from the submodules and compares tensors, so a reordering fails.
3. A base checkpoint converts.  ``load_base_state_dict`` accepts a base state
   dict and refuses a state dict with any other mismatch.
4. The run identity records the choice.  ``final_norm`` appears in the trainer
   record and in the resume-critical fields, so two arms cannot resume into
   each other.

The tests use a small model and synthetic windows.  They never touch a GPU or a
prepared dataset, and they make no claim about training stability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.tracking.diffusion.model import (
  DenoiserSettings,
  StateLatentTransformer,
)
from mjlab.tasks.tracking.diffusion.model_final_norm import (
  FINAL_NORM_PARAMETER_NAMES,
  StateLatentTransformerFinalNorm,
  load_base_state_dict,
)
from mjlab.tasks.tracking.joint_pose import (
  FlowMatchingRunner,
  FlowMatchingTrainerCfg,
  JointPoseDatasetCfg,
  JointPoseFlowConfigError,
  JointPoseFlowRunnerError,
  JointPoseWindowDataset,
  build_denoiser,
)
from mjlab.tasks.tracking.joint_pose.runner import RESUME_CRITICAL_FIELDS

WINDOW_STEPS = 41
PAST_STEPS = 8
CHANNELS = 68
SMALL = {"width": 32, "layers": 1, "heads": 2}


def _settings() -> DenoiserSettings:
  return DenoiserSettings(
    token_dimension=CHANNELS,
    sequence_length=WINDOW_STEPS,
    embedding_count=101,
    width=SMALL["width"],
    layers=SMALL["layers"],
    attention_heads=SMALL["heads"],
    ffn_width=64,
  )


def _batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
  generator = torch.Generator().manual_seed(7)
  tokens = torch.randn(batch, WINDOW_STEPS, CHANNELS, generator=generator)
  levels = torch.randint(0, 101, (batch, WINDOW_STEPS, 2), generator=generator)
  return tokens, levels


def _base() -> StateLatentTransformer:
  return StateLatentTransformer(_settings())


def _variant() -> StateLatentTransformerFinalNorm:
  return StateLatentTransformerFinalNorm(_settings())


def test_variant_adds_exactly_the_two_norm_tensors() -> None:
  """The variant differs from the base model by one LayerNorm, nothing else."""
  base, variant = _base(), _variant()
  base_keys = set(base.state_dict())
  variant_keys = set(variant.state_dict())
  assert variant_keys - base_keys == set(FINAL_NORM_PARAMETER_NAMES)
  assert base_keys - variant_keys == set()

  width = _settings().width
  assert variant.parameter_count() - base.parameter_count() == 2 * width
  assert variant.extra_parameter_count() == 2 * width
  # Every non-norm tensor keeps the base shape, so nothing else changed shape.
  for name, tensor in base.state_dict().items():
    assert variant.state_dict()[name].shape == tensor.shape, name


def test_forward_normalises_before_the_output_projection() -> None:
  """forward equals output_projection(final_norm(_encode(...))), in that order."""
  model = _variant().eval()
  tokens, levels = _batch()
  with torch.no_grad():
    encoded = model._encode(tokens, levels)
    expected = model.output_projection(model.final_norm(encoded))
    actual = model(tokens, levels)
  assert torch.equal(actual, expected)

  # The base path is the same residual stream with no norm in between, so the
  # extraction left the frozen forward unchanged.
  base = _base().eval()
  with torch.no_grad():
    assert torch.equal(
      base(tokens, levels), base.output_projection(base._encode(tokens, levels))
    )


def test_norm_bounds_the_residual_stream_that_the_projection_reads() -> None:
  """A large input leaves the normalised stream bounded; the raw one is not.

  This is a property of LayerNorm over the feature axis, not evidence about
  training.  It is the quantity the ablation is meant to change: the output
  projection of the variant never reads an unbounded residual value.
  """
  model = _variant().eval()
  tokens, levels = _batch()
  tokens = tokens * 29.0
  with torch.no_grad():
    raw = model._encode(tokens, levels)
    normalised = model.final_norm(raw)
  width = _settings().width
  assert float(raw.abs().max()) > 10.0
  # Per-token feature RMS becomes 1 after the norm, so the maximum of an
  # uninformative vector stays near sqrt(width); allow a wide factor for the
  # learned gain and bias of a freshly initialised LayerNorm.
  assert float(normalised.abs().max()) <= 4.0 * (width**0.5)
  assert float(normalised.pow(2).mean(dim=-1).max()) < 8.0
  with torch.no_grad():
    assert bool(torch.isfinite(model(tokens, levels)).all())


def test_shared_weights_match_when_both_models_use_one_init_seed() -> None:
  """A matched ablation pair starts from identical shared weights."""
  dataset = JointPoseDatasetCfg()
  base = build_denoiser(
    dataset, max_level=1000, init_seed=11, final_norm=False, **SMALL
  )
  variant = build_denoiser(
    dataset, max_level=1000, init_seed=11, final_norm=True, **SMALL
  )
  assert isinstance(variant, StateLatentTransformerFinalNorm)
  assert type(base) is StateLatentTransformer
  for name, tensor in base.state_dict().items():
    assert torch.equal(tensor, variant.state_dict()[name]), name

  other = build_denoiser(
    dataset, max_level=1000, init_seed=12, final_norm=True, **SMALL
  )
  assert not torch.equal(
    other.state_dict()["input_projection.weight"],
    variant.state_dict()["input_projection.weight"],
  )


def test_unseeded_build_still_returns_the_frozen_class() -> None:
  """final_norm=False keeps the shared class and its dtype/device behaviour."""
  model = build_denoiser(JointPoseDatasetCfg(), max_level=1000, **SMALL)
  assert type(model) is StateLatentTransformer
  assert not hasattr(model, "final_norm")


def test_load_base_state_dict_converts_and_rejects_mismatches() -> None:
  """A base checkpoint loads; any mismatch beyond the norm raises."""
  base, variant = _base(), _variant()
  load_base_state_dict(variant, base.state_dict())
  for name, tensor in base.state_dict().items():
    assert torch.equal(tensor, variant.state_dict()[name]), name
  # The norm keeps its LayerNorm initialisation: weight 1, bias 0.
  assert torch.equal(
    variant.final_norm.weight, torch.ones_like(variant.final_norm.weight)
  )
  assert torch.equal(variant.final_norm.bias, torch.zeros_like(variant.final_norm.bias))

  # A variant state dict round-trips with no missing keys.
  load_base_state_dict(_variant(), variant.state_dict())

  renamed = dict(base.state_dict())
  del renamed["input_projection.weight"]
  with pytest.raises(RuntimeError, match="does not match the final-norm variant"):
    load_base_state_dict(_variant(), renamed)

  with pytest.raises(RuntimeError, match="does not match the final-norm variant"):
    load_base_state_dict(
      _variant(),
      {k: v for k, v in base.state_dict().items() if k != "input_projection.bias"},
    )

  with pytest.raises(TypeError, match="StateLatentTransformerFinalNorm"):
    load_base_state_dict(_base(), base.state_dict())  # type: ignore[arg-type]


def test_final_norm_is_recorded_and_resume_critical() -> None:
  """The choice is part of the run identity and blocks a cross resume."""
  assert "final_norm" in RESUME_CRITICAL_FIELDS
  base_cfg = FlowMatchingTrainerCfg(max_steps=10, warmup_updates=1, final_norm=False)
  variant_cfg = FlowMatchingTrainerCfg(max_steps=10, warmup_updates=1, final_norm=True)
  assert base_cfg.as_dict()["final_norm"] is False
  assert variant_cfg.as_dict()["final_norm"] is True
  assert base_cfg.as_dict() != variant_cfg.as_dict()

  seeded = FlowMatchingTrainerCfg(max_steps=10, warmup_updates=1, init_seed=11)
  assert seeded.as_dict()["init_seed"] == 11
  assert (
    FlowMatchingTrainerCfg(max_steps=10, warmup_updates=1).as_dict()["init_seed"]
    is None
  )

  with pytest.raises(JointPoseFlowConfigError, match="final_norm must be bool"):
    FlowMatchingTrainerCfg(max_steps=10, warmup_updates=1, final_norm=1)  # type: ignore[arg-type]
  with pytest.raises(JointPoseFlowConfigError, match="init_seed"):
    FlowMatchingTrainerCfg(max_steps=10, warmup_updates=1, init_seed=1.5)  # type: ignore[arg-type]


def test_the_dimension_checks_still_apply_to_the_variant() -> None:
  """The variant inherits the shared input validation, not a copy of it."""
  model = _variant()
  tokens, levels = _batch()
  with pytest.raises(ValueError, match="does not match settings"):
    model(tokens[:, :-1], levels[:, :-1])
  with pytest.raises(ValueError, match="step_ids must have shape"):
    model(tokens, levels[:, :, :1])


def _windows(count: int, *, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.standard_normal((count, WINDOW_STEPS, CHANNELS)).astype(np.float32)


@pytest.fixture
def training_root(tmp_path: Path) -> Path:
  root = tmp_path / "data"
  for relative, count, seed in (
    ("offline/tokens-train.npy", 32, 0),
    ("reference/train.npy", 4, 1),
  ):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, _windows(count, seed=seed))
  return root


def _runner(root: Path, run_dir: Path, **overrides: object) -> FlowMatchingRunner:
  dataset_cfg = JointPoseDatasetCfg(
    data_dir=str(root),
    window_steps=WINDOW_STEPS,
    past_steps=PAST_STEPS,
    channels=CHANNELS,
  )
  dataset = JointPoseWindowDataset(dataset_cfg)
  settings: dict[str, object] = {
    "max_steps": 2,
    "micro_batch": 4,
    "gradient_accumulation_steps": 1,
    "warmup_updates": 1,
    "checkpoint_every": 1,
    "log_every": 1,
    "mixed_precision": "none",
  }
  settings.update(overrides)
  return FlowMatchingRunner(
    dataset,
    {},
    FlowMatchingTrainerCfg(**settings),  # type: ignore[arg-type]
    "cpu",
    run_dir=run_dir,
    **SMALL,
  )


def test_runner_trains_the_variant_and_refuses_the_other_architecture(
  training_root: Path, tmp_path: Path
) -> None:
  """A variant run writes records a base run cannot resume into."""
  variant_run = tmp_path / "variant"
  runner = _runner(training_root, variant_run, final_norm=True, init_seed=11)
  assert isinstance(runner.model, StateLatentTransformerFinalNorm)
  runner.learn()
  checkpoint = variant_run / "model_0000002.pt"
  assert checkpoint.is_file()

  # Reopening the same run directory with the other architecture refuses in the
  # constructor, before any weight is loaded.
  with pytest.raises(JointPoseFlowRunnerError, match="final_norm"):
    _runner(training_root, variant_run, final_norm=False, init_seed=11)

  base_run = tmp_path / "base"
  base_runner = _runner(training_root, base_run, final_norm=False, init_seed=11)
  assert type(base_runner.model) is StateLatentTransformer
  base_runner.learn()
  with pytest.raises(JointPoseFlowRunnerError, match="final_norm"):
    _runner(
      training_root, tmp_path / "variant-other", final_norm=True, init_seed=11
    ).load(base_run / "model_0000002.pt")
