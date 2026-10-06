"""CPU-only integration tests for the D1 diffusion command line."""

from pathlib import Path

from mjlab.scripts import diffusion
from mjlab.tasks.tracking.diffusion import TrialSpec


def test_bare_diffusion_cli_only_prints_help(capsys) -> None:
  assert diffusion.main([]) == 0
  output = capsys.readouterr().out
  assert "preflight" in output
  assert "collect" in output


def test_collect_requires_explicit_operational_opt_in(tmp_path: Path, capsys) -> None:
  code = diffusion.main(
    [
      "collect",
      "--output",
      str(tmp_path / "store"),
      "--report",
      str(tmp_path / "report.json"),
      "--runtime-factory",
      "example:factory",
    ]
  )
  assert code == 2
  assert "requires --execute" in capsys.readouterr().err


def test_preflight_reports_contract_failure_without_launch(monkeypatch, capsys) -> None:
  monkeypatch.setattr(
    diffusion,
    "_load_contract",
    lambda path: (_ for _ in ()).throw(ValueError("bad frozen hash")),
  )
  assert diffusion.main(["preflight", "--contract", "missing.yaml"]) == 1
  output = capsys.readouterr().out
  assert '"ok": false' in output
  assert "bad frozen hash" in output


def test_cli_pair_validation_requires_clean_before_matching_ou() -> None:
  clean = TrialSpec("run", "motion", 0, 0, pair_id="p", initial_state_id="s")
  ou = TrialSpec("run", "motion", 0, 0, "ou", pair_id="p", initial_state_id="s")
  assert diffusion._validate_trial_pairs((ou, clean)) is not None
  assert diffusion._validate_trial_pairs((clean, ou)) is None


def test_cli_rejects_mismatched_partner_identity() -> None:
  clean = TrialSpec("run", "motion", 0, 0, pair_id="p", initial_state_id="s")
  mismatched = TrialSpec(
    "run", "other-motion", 0, 0, "ou", pair_id="p", initial_state_id="s"
  )
  assert diffusion._validate_trial_pairs((clean, mismatched)) is not None


def test_cli_allows_interleaved_clean_and_ou_phases() -> None:
  clean_a = TrialSpec("run", "motion", 0, 0, pair_id="a", initial_state_id="s-a")
  clean_b = TrialSpec("run", "motion", 0, 1, pair_id="b", initial_state_id="s-b")
  ou_a = TrialSpec("run", "motion", 0, 0, "ou", pair_id="a", initial_state_id="s-a")
  ou_b = TrialSpec("run", "motion", 0, 1, "ou", pair_id="b", initial_state_id="s-b")
  assert diffusion._validate_trial_pairs((clean_a, ou_a, clean_b, ou_b)) is None


def test_cli_allows_duplicate_initial_state_ids_without_pair_aliasing() -> None:
  clean_a = TrialSpec("run", "motion", 0, 0, pair_id="a", initial_state_id="same")
  clean_b = TrialSpec("run", "motion", 1, 0, pair_id="b", initial_state_id="same")
  ou_a = TrialSpec("run", "motion", 0, 0, "ou", pair_id="a", initial_state_id="same")
  ou_b = TrialSpec("run", "motion", 1, 0, "ou", pair_id="b", initial_state_id="same")
  assert diffusion._validate_trial_pairs((clean_a, clean_b, ou_a, ou_b)) is None


def test_cli_rejects_unverified_runtime_bundle() -> None:
  bundle = diffusion.RuntimeBundle(object(), ())
  assert diffusion._validate_runtime_bundle(bundle) is not None


def test_cli_rejects_verified_duck_typed_runtime_bundle() -> None:
  class _DuckCollector:
    def collect(self, spec, *, remaining_transitions=None):
      del spec, remaining_transitions

  bundle = diffusion.RuntimeBundle(
    _DuckCollector(),
    (),
    True,
    {"environment": "agibot_x2", "max_envs": "1", "max_gpus": "1"},
  )
  duck_result = diffusion._validate_runtime_bundle(bundle)
  assert duck_result is not None
  assert "concrete DiffusionCollector" in duck_result


def test_cli_allows_a_multi_environment_budget_and_bounds_it() -> None:
  """The semantic max_envs == 1 rule was removed on 2026-10-06.

  A shared multi-motion environment held 30 reference clips and 64 parallel rows
  in 4.3 GiB, against 36.9 GiB for 30 single-motion environments, so a declared
  budget above 1 must validate.  An absurd value must still meet the ceiling.
  """
  parser = diffusion.build_parser()
  accepts = parser.parse_args(
    ["collect", "--output", "store", "--report", "report.json", "--max-envs", "64"]
  )
  assert diffusion._validate_resource_budgets(accepts) is None
  refuses = parser.parse_args(
    [
      "collect",
      "--output",
      "store",
      "--report",
      "report.json",
      "--max-envs",
      "100000",
    ]
  )
  result = diffusion._validate_resource_budgets(refuses)
  assert result is not None
  assert "max_envs" in result


def test_cli_allows_a_longer_monitor_tolerance_but_bounds_it() -> None:
  """A stall longer than the default tolerance stopped a real collection.

  The default stays at 120 s.  A longer explicit value must validate, because a
  shared-storage stall of about ten minutes stopped a 2565-trial collection
  (2026-10-06), and the ceiling must still refuse an effectively unbounded value.
  """
  parser = diffusion.build_parser()
  longer = parser.parse_args(
    [
      "collect",
      "--output",
      "store",
      "--report",
      "report.json",
      "--monitor-inactivity-seconds",
      "900",
    ]
  )
  assert diffusion._validate_resource_budgets(longer) is None
  too_long = parser.parse_args(
    [
      "collect",
      "--output",
      "store",
      "--report",
      "report.json",
      "--monitor-inactivity-seconds",
      "1801",
    ]
  )
  result = diffusion._validate_resource_budgets(too_long)
  assert result is not None
  assert "monitor_inactivity_seconds" in result
  default = parser.parse_args(
    ["collect", "--output", "store", "--report", "report.json"]
  )
  assert default.monitor_inactivity_seconds == 120


def test_cli_collect_rejects_a_standalone_transition_budget() -> None:
  """The transition count comes from the schedule, not a separate cap.

  A hand-set cap smaller than a worker's shard silently truncated a real
  collection (2026-10-06), so the CLI must not accept one and the collection
  path must not expose a constant for it.
  """
  parser = diffusion.build_parser()
  try:
    parser.parse_args(
      [
        "collect",
        "--output",
        "store",
        "--report",
        "report.json",
        "--max-control-transitions",
        "600000",
      ]
    )
  except SystemExit:
    pass
  else:  # pragma: no cover
    raise AssertionError("a standalone transition budget was accepted")
  args = parser.parse_args(["collect", "--output", "store", "--report", "report.json"])
  assert not hasattr(args, "max_control_transitions")
  assert not hasattr(diffusion, "MAX_COLLECTION_TRANSITIONS")
  assert diffusion._validate_resource_budgets(args) is None


def test_collection_config_accepts_a_multi_environment_budget(tmp_path: Path) -> None:
  config = tmp_path / "collection.yaml"
  config.write_text(
    "schema_version: mjlab-x2-diffusion-d1-collection-v1\n"
    "execution_authorized: false\n"
    "pilot:\n"
    "  max_envs: 64\n"
  )
  payload = diffusion._load_collection_config(config)
  assert payload["pilot"]["max_envs"] == 64
  bad = tmp_path / "bad.yaml"
  bad.write_text(
    "schema_version: mjlab-x2-diffusion-d1-collection-v1\n"
    "execution_authorized: false\n"
    "pilot:\n"
    "  max_envs: 0\n"
  )
  try:
    diffusion._load_collection_config(bad)
  except ValueError as exc:
    assert "max_envs" in str(exc)
  else:  # pragma: no cover
    raise AssertionError("a non-positive environment budget was accepted")


def test_runtime_factory_requires_module_function_syntax() -> None:
  try:
    diffusion._load_factory("not-qualified")
  except ValueError as exc:
    assert "MODULE:FUNCTION" in str(exc)
  else:  # pragma: no cover
    raise AssertionError("invalid factory syntax was accepted")


def test_ou_pair_block_reason_allows_vae_only_clean_partner() -> None:
  """A qualified ``vae_only`` partner must not block its OU trial."""

  class _Qualification:
    qualified: bool = True
    label: str = "vae_only"
    reason: str | None = None
    handoff_step: int | None = None

  class _Result:
    qualification = _Qualification()

  assert diffusion._ou_pair_block_reason(None) is not None
  assert diffusion._ou_pair_block_reason(_Result()) is None

  _Qualification.qualified = False
  _Qualification.reason = "physical_fall"
  blocked = diffusion._ou_pair_block_reason(_Result())
  assert blocked is not None
  assert "physical_fall" in blocked
