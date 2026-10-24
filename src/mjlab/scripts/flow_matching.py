"""Opt-in command line for the X2 x0 flow-matching family.

``train`` runs one explicitly bounded flow training job.  The dataset-owned
token caches are read-only inputs: the run validates the published cache and
fails closed when it is missing, and a missing cache is prepared with the shared
``python -m mjlab.scripts.diffusion build-token-cache`` command.
``evaluate-offline`` loads a flow checkpoint, generates tokens with the Stage 1
x0 Euler sampler, and scores the generated unknown entries and the trivial
baselines without a simulator.

This module never constructs a diffusion noise schedule, never reads or writes a
diffusion checkpoint, and does not change diffusion CLI behavior.  It imports
the frozen diffusion contract, model, trainer and cache infrastructure read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
  from mjlab.tasks.tracking.diffusion.model import DenoiserSettings
  from mjlab.tasks.tracking.flow_matching.process import LinearNoiseProcess

DEFAULT_CONTRACT_PATH = Path(
  "docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"
)
FLOW_RUN_METADATA_NAME = "flow-run.json"
FLOW_RUN_SCHEMA = "mjlab-flow-matching-run-v1"


def _json_write(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_config_contract(config_path: Path, contract_path: Path) -> None:
  """Refuse a flow config that names a contract other than ``--contract``."""
  try:
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
  except (OSError, ImportError, TypeError, ValueError) as exc:
    raise ValueError(f"could not read flow config {config_path}: {exc}") from exc
  if not isinstance(raw, Mapping) or raw.get("contract") is None:
    return
  declared = Path(str(raw["contract"]))
  if not declared.is_absolute():
    declared = Path.cwd() / declared
  if declared.resolve() != contract_path.resolve():
    raise ValueError(
      f"flow config contract path disagrees with --contract: {raw['contract']!r}"
    )


def _split_dataset(
  source: Any, loaded: Any, split: str, *, source_hash: str
) -> tuple[Any, Path]:
  """Validate and open one split's published, read-only token cache."""
  from mjlab.tasks.tracking.diffusion.token_cache_lifecycle import (
    validate_token_cache_read_only,
  )
  from mjlab.tasks.tracking.diffusion.window_dataset import TokenWindowDataset

  prepared = validate_token_cache_read_only(
    source, loaded, split, source_hash=source_hash
  )
  dataset = TokenWindowDataset(
    prepared.path,
    list(prepared.provenance),
    source=source,
    split=split,
    validated_cache=prepared.cache,
  )
  return dataset, prepared.path


def _settings_from_environment(settings: "DenoiserSettings") -> "DenoiserSettings":
  """Apply optional width, depth and FFN overrides for a capacity ablation.

  The frozen default shape is left alone; only a run that sets one of these
  variables builds a different model, and the override is recorded in the model
  identity, so a checkpoint names the shape that produced it.
  """
  from dataclasses import replace

  raw = {
    "width": os.environ.get("MJLAB_FLOW_WIDTH", ""),
    "layers": os.environ.get("MJLAB_FLOW_LAYERS", ""),
    "ffn_width": os.environ.get("MJLAB_FLOW_FFN", ""),
  }
  if not any(raw.values()):
    return settings
  values: dict[str, int] = {}
  for name, value in raw.items():
    if value == "":
      continue
    try:
      parsed = int(value)
    except ValueError as exc:
      raise SystemExit(f"{name} override must be an integer, got {value!r}") from exc
    if parsed < 1:
      raise SystemExit(f"{name} override must be positive, got {parsed}")
    values[name] = parsed
  return replace(settings, **values)


def _build_denoiser(settings: "DenoiserSettings", *, final_norm: bool):
  """Build the shared denoiser, or the final-norm ablation variant.

  The variant adds one LayerNorm between the encoder and the output projection.
  It is opt-in because it changes the parameter shapes, so a checkpoint of one
  kind cannot be loaded into the other without conversion.
  """
  from mjlab.tasks.tracking.diffusion.model import StateLatentTransformer

  settings = _settings_from_environment(settings)
  if not final_norm:
    return StateLatentTransformer(settings)
  from mjlab.tasks.tracking.diffusion.model_final_norm import (
    StateLatentTransformerFinalNorm,
  )

  return StateLatentTransformerFinalNorm(settings)


def _checkpoint_process(path: Path) -> "LinearNoiseProcess":
  """Read the linear process a flow checkpoint records before model construction.

  ``load_flow_checkpoint`` restores state into a caller-built model, so the model
  shape must be known first.  The checkpoint is therefore read once for this
  identity; the load then revalidates the same process, so a tampered process
  mapping cannot reach training or sampling.
  """
  import torch

  from mjlab.tasks.tracking.flow_matching.checkpoint import (
    FLOW_CHECKPOINT_FORMAT,
    FlowCheckpointError,
    _restore_process,
  )
  from mjlab.tasks.tracking.flow_matching.process import LinearNoiseProcess

  try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
  except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
    raise FlowCheckpointError(f"could not read flow checkpoint {path}") from exc
  if (
    not isinstance(payload, Mapping) or payload.get("format") != FLOW_CHECKPOINT_FORMAT
  ):
    raise FlowCheckpointError("unsupported or malformed flow checkpoint")
  return LinearNoiseProcess(int(_restore_process(payload)["max_level"]))


def _recorded_eval_seed(config: Mapping[str, Any]) -> int:
  """Read the evaluation seed recorded inside a flow checkpoint config."""
  training = config.get("training")
  if not isinstance(training, Mapping):
    raise ValueError("flow checkpoint config has no training section")
  seed = training.get("eval_seed", 0)
  if isinstance(seed, bool) or not isinstance(seed, int):
    raise ValueError("flow checkpoint eval_seed is not an integer")
  return seed


def _command_train(args: argparse.Namespace) -> int:
  """Build read-only D1 window datasets and run one bounded flow training job."""
  try:
    import torch

    from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
    from mjlab.tasks.tracking.diffusion.model import (
      DenoiserSettings,
    )
    from mjlab.tasks.tracking.diffusion.trainer import _set_seed
    from mjlab.tasks.tracking.diffusion.window_dataset import (
      DatasetSource,
      _source_dataset_hash,
    )
    from mjlab.tasks.tracking.flow_matching import (
      FlowTrainer,
      FlowTrainingConfig,
      FlowTrainingConfigError,
    )

    if args.device == "cuda:0":
      os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = FlowTrainingConfig.from_yaml(args.config)
    _validate_config_contract(args.config, args.contract)
    if (
      args.max_updates is None
      and args.epochs is None
      and config.training.max_updates is None
    ):
      raise FlowTrainingConfigError(
        "train requires an explicit --max-updates, --epochs, or config max_updates"
      )
    if args.max_updates is not None:
      config = replace(
        config, training=replace(config.training, max_updates=args.max_updates)
      )
    if args.epochs is not None:
      config = replace(config, training=replace(config.training, epochs=args.epochs))
    config.validate(allow_long_run=args.allow_long_run)

    process = config.process
    contract = DiffusionContract.from_yaml(args.contract)
    source = DatasetSource.resolve(args.dataset_dir, contract_path=args.contract)
    loaded = source.load()
    source_hash = _source_dataset_hash(source, loaded)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, train_cache = _split_dataset(
      source, loaded, "train", source_hash=source_hash
    )
    if len(train_dataset) == 0:
      raise ValueError("training split is empty")
    eval_split = config.training.eval_split
    cache_paths = {"train": str(train_cache)}
    windows = {"train": len(train_dataset)}
    eval_datasets: dict[str, Any] = {}
    if eval_split == "train":
      eval_datasets["train"] = train_dataset
    else:
      eval_dataset, eval_cache = _split_dataset(
        source, loaded, eval_split, source_hash=source_hash
      )
      eval_datasets[eval_split] = eval_dataset
      cache_paths[eval_split] = str(eval_cache)
      windows[eval_split] = len(eval_dataset)

    settings = DenoiserSettings.from_contract(contract, training_k=process.max_level)
    _set_seed(config.training.seed)
    model = _build_denoiser(settings, final_norm=bool(args.final_norm))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
      raise RuntimeError("CUDA device requested but CUDA is unavailable")
    run_metadata = {
      "schema": FLOW_RUN_SCHEMA,
      "family": process.as_dict()["family"],
      "process": process.as_dict(),
      "process_identity": process.identity_hash(),
      "config_path": str(args.config),
      "config_sha256": config.sha256(),
      "history_state_dropout_probability": float(
        config.training.history_state_dropout_probability
      ),
      "contract_path": str(args.contract),
      "contract_identity": contract.identity_hash(),
      "dataset_directory": str(source.directory),
      "dataset_identity": dict(getattr(train_dataset, "dataset_identity", {})),
      "seed": config.training.seed,
      "device": str(device),
      "evaluation_split": eval_split,
      "resume": None if args.resume is None else str(args.resume),
      "allow_long_run": bool(args.allow_long_run),
      "cache_paths": cache_paths,
      "windows": windows,
      "note": "bounded engineering run metadata; no model-quality claim",
    }
    metadata_path = output_dir / FLOW_RUN_METADATA_NAME
    _json_write(metadata_path, run_metadata)
    preamble = {**run_metadata, "run_metadata": str(metadata_path)}
    print(json.dumps(preamble, indent=2, sort_keys=True))
    trainer = FlowTrainer(
      config=config,
      contract=contract,
      model=model,
      train_dataset=train_dataset,
      eval_datasets=eval_datasets,
      output_dir=output_dir,
      device=device,
      resume=args.resume,
      allow_long_run=args.allow_long_run,
    )
    result = trainer.train()
    payload = {
      **preamble,
      "global_step": result.global_step,
      "epochs_completed": result.epochs_completed,
      "best_validation_loss": result.best_validation_loss,
      "final_train_loss": result.final_train_loss,
      "metrics_path": str(result.metrics_path),
      "checkpoint_paths": {
        name: str(path) for name, path in result.checkpoint_paths.items()
      },
      "train_report": str(output_dir / "train-report.json"),
      "note": "bounded engineering training; no model-quality claim",
    }
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
  except (OSError, RuntimeError, TypeError, ValueError) as exc:
    print(f"train: {exc}", file=sys.stderr)
    return 1


def _command_evaluate_offline(args: argparse.Namespace) -> int:
  """Load a flow checkpoint and score generated tokens without a simulator."""
  try:
    import torch

    from mjlab.tasks.tracking.diffusion.contract import DiffusionContract
    from mjlab.tasks.tracking.diffusion.model import (
      DenoiserSettings,
    )
    from mjlab.tasks.tracking.diffusion.trainer import ExponentialMovingAverage
    from mjlab.tasks.tracking.diffusion.window_dataset import (
      DatasetSource,
      _source_dataset_hash,
      load_window_records,
    )
    from mjlab.tasks.tracking.flow_matching import load_flow_checkpoint
    from mjlab.tasks.tracking.flow_matching.evaluation import evaluate_generation

    run_dir = Path(args.run)
    checkpoint_path = (
      Path(args.checkpoint)
      if args.checkpoint is not None
      else run_dir / "checkpoint-best.pt"
    )
    if not checkpoint_path.is_file():
      fallback = run_dir / "checkpoint-last.pt"
      raise FileNotFoundError(
        f"checkpoint not found: {checkpoint_path}; "
        f"pass --checkpoint {fallback} explicitly"
      )
    if args.max_windows is not None and args.max_windows < 0:
      raise ValueError("--max-windows must be non-negative")
    contract = DiffusionContract.from_yaml(args.contract)
    process = _checkpoint_process(checkpoint_path)
    settings = DenoiserSettings.from_contract(contract, training_k=process.max_level)
    model = _build_denoiser(
      settings, final_norm=bool(getattr(args, "final_norm", False))
    )
    ema = ExponentialMovingAverage(model)
    device = torch.device(args.device)
    state = load_flow_checkpoint(
      checkpoint_path,
      model=model,
      ema=ema,
      map_location=device,
      expected_process_identity=process.identity_hash(),
      expected_data_identity=contract.identity_hash(),
    )
    dataset_path = state.dataset_identity.get("directory")
    if not isinstance(dataset_path, str) or not dataset_path:
      raise ValueError("flow checkpoint does not record a dataset directory")
    source = DatasetSource.resolve(dataset_path, contract_path=args.contract)
    loaded = source.load()
    if Path(dataset_path).resolve() != source.directory.resolve():
      raise ValueError("checkpoint dataset directory could not be resolved")
    expected_identity = state.dataset_identity
    if expected_identity.get("source_dataset_hash") != _source_dataset_hash(
      source, loaded
    ):
      raise ValueError("checkpoint dataset source identity mismatch")
    if expected_identity.get("assignments_hash") != loaded.assignments.sha256():
      raise ValueError("checkpoint assignments identity mismatch")
    if expected_identity.get("split_coverage") != loaded.index.coverage():
      raise ValueError("checkpoint split coverage mismatch")
    if expected_identity.get("contract_hash") != contract.identity_hash():
      raise ValueError("checkpoint dataset contract identity mismatch")
    if dict(expected_identity.get("projection_hashes", {})) != {
      "matrix": loaded.projection.matrix_sha256,
      "pseudoinverse": loaded.projection.pseudoinverse_sha256,
      "statistics": loaded.projection.statistics_sha256,
    }:
      raise ValueError("checkpoint projection identity mismatch")
    if expected_identity.get("motion_ids") is not None:
      raise ValueError(
        "checkpoint dataset identity has a motion filter; this run "
        "was trained on the dataset-wide cache"
      )
    if state.ema_state is not None:
      ema.copy_to(model)
    model.to(device)
    seed = _recorded_eval_seed(state.config) if args.seed is None else args.seed
    diagnostics_path = run_dir / f"flow-generation-{args.split}.json"
    report = evaluate_generation(
      model,
      process=process,
      records=load_window_records(loaded, args.split, limit=args.max_windows),
      contract=contract,
      seed=seed,
      device=device,
      max_windows=args.max_windows,
      diagnostics_path=diagnostics_path,
      split=args.split,
    )
    payload = report.as_dict()
    payload["checkpoint"] = str(checkpoint_path)
    payload["global_step"] = state.global_step
    payload["parameters"] = "ema" if state.ema_state is not None else "model"
    payload["dataset"] = str(source.directory)
    payload["device"] = str(device)
    if device.type == "cuda":
      payload["device_name"] = torch.cuda.get_device_name(device)
    if report.windows == 0 and args.max_windows != 0:
      print(
        f"evaluate-offline: split {args.split!r}: no windows were selected",
        file=sys.stderr,
      )
      return 1
    if args.json is not None:
      _json_write(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
  except (OSError, RuntimeError, TypeError, ValueError) as exc:
    print(f"evaluate-offline: {exc}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="flow-matching",
    description=(
      "Opt-in X2 x0 flow-matching training and offline generation evaluation.  "
      "Existing D1 contracts, statistics and token caches are read-only inputs."
    ),
  )
  commands = parser.add_subparsers(dest="command", required=False)

  train = commands.add_parser("train", help="run explicitly bounded flow training")
  train.add_argument("--dataset-dir", type=Path, required=True)
  train.add_argument("--config", type=Path, required=True)
  train.add_argument("--out", type=Path, required=True)
  train.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  train.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
  budget = train.add_mutually_exclusive_group()
  budget.add_argument("--max-updates", type=int)
  budget.add_argument("--epochs", type=int)
  train.add_argument("--resume", type=Path)
  train.add_argument("--allow-long-run", action="store_true")
  train.add_argument(
    "--final-norm",
    action="store_true",
    help="add a LayerNorm between the encoder and the output projection",
  )
  train.add_argument("--json", type=Path)
  train.set_defaults(handler=_command_train)

  evaluate = commands.add_parser(
    "evaluate-offline",
    help="score generated tokens of a flow checkpoint without a simulator",
  )
  evaluate.add_argument("--run", type=Path, required=True)
  evaluate.add_argument("--checkpoint", type=Path)
  evaluate.add_argument(
    "--split", choices=("train", "validation", "test"), default="validation"
  )
  evaluate.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
  evaluate.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
  evaluate.add_argument("--max-windows", type=int)
  evaluate.add_argument("--seed", type=int)
  evaluate.add_argument("--json", type=Path)
  evaluate.set_defaults(handler=_command_evaluate_offline)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Run the CLI and return a process exit code."""
  parser = build_parser()
  args = parser.parse_args(argv)
  if args.command is None:
    parser.print_help()
    return 0
  return int(args.handler(args))


if __name__ == "__main__":
  raise SystemExit(main())
