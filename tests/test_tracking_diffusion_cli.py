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


def test_cli_rejects_interleaved_clean_and_ou_phases() -> None:
  clean_a = TrialSpec("run", "motion", 0, 0, pair_id="a", initial_state_id="s-a")
  clean_b = TrialSpec("run", "motion", 0, 1, pair_id="b", initial_state_id="s-b")
  ou_a = TrialSpec("run", "motion", 0, 0, "ou", pair_id="a", initial_state_id="s-a")
  assert diffusion._validate_trial_pairs((clean_a, ou_a, clean_b)) is not None


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


def test_cli_rejects_multi_environment_budget_until_identity_is_carried() -> None:
  args = diffusion.build_parser().parse_args(
    [
      "collect",
      "--output",
      "store",
      "--report",
      "report.json",
      "--max-envs",
      "2",
    ]
  )
  budget_result = diffusion._validate_resource_budgets(args)
  assert budget_result is not None
  assert "max_envs" in budget_result


def test_collection_config_rejects_multi_environment_budget(tmp_path: Path) -> None:
  config = tmp_path / "collection.yaml"
  config.write_text(
    "schema_version: mjlab-x2-diffusion-d1-collection-v1\n"
    "execution_authorized: false\n"
    "pilot:\n"
    "  max_envs: 8\n"
  )
  try:
    diffusion._load_collection_config(config)
  except ValueError as exc:
    assert "max_envs" in str(exc)
  else:  # pragma: no cover
    raise AssertionError("multi-environment config was accepted")


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
