"""Command-line tests for safe cache preparation and read-only validation."""

from __future__ import annotations

import json
from pathlib import Path

from mjlab.scripts import diffusion

PILOT = Path("logs/diffusion/d1-pilot-20260930b")
CONTRACT = Path("docs/plans/beyondmimic_diffusion_d0_contract.yaml")


def test_build_token_cache_parser_exposes_explicit_modes() -> None:
  args = diffusion.build_parser().parse_args(
    [
      "build-token-cache",
      "--dataset-dir",
      str(PILOT),
      "--contract",
      str(CONTRACT),
      "--splits",
      "train,validation",
      "--validate-only",
      "--lock-timeout",
      "2.5",
    ]
  )
  assert args.validate_only is True
  assert args.lock_timeout == 2.5
  assert args.jobs == 1


def test_validate_only_missing_cache_writes_no_files_and_starts_no_workers(
  tmp_path: Path, capsys
) -> None:
  report = tmp_path / "report.json"
  before = sorted(PILOT.joinpath("offline").glob("tokens-*"))
  code = diffusion.main(
    [
      "build-token-cache",
      "--dataset-dir",
      str(PILOT),
      "--contract",
      str(CONTRACT),
      "--splits",
      "test",
      "--validate-only",
      "--json",
      str(report),
    ]
  )
  assert code == 1
  assert not report.exists()
  assert sorted(PILOT.joinpath("offline").glob("tokens-*")) == before
  output = capsys.readouterr()
  payload = json.loads(output.out)
  assert payload["complete"] is False
  assert payload["splits"]["test"]["status"] == "missing"
  assert "workers" not in output.out.lower()


def test_train_parser_requires_explicit_cache_preparation() -> None:
  args = diffusion.build_parser().parse_args(
    [
      "train",
      "--dataset-dir",
      str(PILOT),
      "--config",
      "configs/diffusion/x2_50hz_train.yaml",
      "--out",
      "run",
      "--max-updates",
      "1",
    ]
  )
  assert args.cache_tokens is True
  assert args.prepare_token_cache is False
  assert args.cache_jobs == 1


def test_train_cache_policy_defaults_to_prepare_with_strict_opt_in() -> None:
  """Legacy behaviour must stay the default: prepare a missing cache.

  The strict path exists for a dataset that was prepared in advance, and the
  lock wait must be long enough for losers of a concurrent preparation to reuse
  the published cache instead of failing.
  """
  from mjlab.scripts.diffusion import build_parser

  parser = build_parser()
  base = [
    "train",
    "--dataset-dir",
    "somewhere",
    "--config",
    "config.yaml",
    "--out",
    "out",
    "--contract",
    "contract.yaml",
  ]
  legacy = parser.parse_args(base)
  assert legacy.prepare_token_cache is False
  assert legacy.require_prepared_cache is False
  assert legacy.cache_lock_timeout >= 600.0, (
    "a short lock wait makes concurrent preparers fail instead of reusing"
  )

  strict = parser.parse_args(base + ["--require-prepared-cache"])
  assert strict.require_prepared_cache is True

  explicit = parser.parse_args(base + ["--prepare-token-cache"])
  assert explicit.prepare_token_cache is True
