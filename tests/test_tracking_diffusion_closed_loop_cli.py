"""CPU-only tests for the D3 closed-loop integration CLI.

These tests exercise parser, preflight and dispatch gates only.  They do not
construct MuJoCo, allocate a GPU environment, or claim physical evidence.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

_CLI_PATH = (
  Path(__file__).parents[1] / "scripts/experiments/diffusion_d3_closed_loop/evaluate.py"
)
_CLI_SPEC = spec_from_file_location("d3_closed_loop_cli", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
evaluate = module_from_spec(_CLI_SPEC)
sys.modules[_CLI_SPEC.name] = evaluate
_CLI_SPEC.loader.exec_module(evaluate)


_CONTRACT_SHA = "a" * 64


def _request(tmp_path: Path, **updates: object) -> Namespace:
  values: dict[str, object] = {
    "seed": 0,
    "start_frame": 10,
    "continuation_steps": 25,
    "max_wall_seconds": 1200.0,
    "max_output_bytes": 128 * 1024 * 1024,
    "output_dir": tmp_path / "new-run",
    "repo_root": tmp_path,
    "device": "cpu",
  }
  values.update(updates)
  return Namespace(**values)


def _execute_args(tmp_path: Path) -> list[str]:
  return [
    "--contract",
    "contract.yaml",
    "--contract-sha256",
    _CONTRACT_SHA,
    "--checkpoint",
    "checkpoint.pt",
    "--checkpoint-sha256",
    _CONTRACT_SHA,
    "--projection",
    "projection.npz",
    "--projection-sha256",
    _CONTRACT_SHA,
    "--vae-checkpoint",
    "vae.pt",
    "--vae-checkpoint-sha256",
    _CONTRACT_SHA,
    "--manifest",
    "manifest.yaml",
    "--manifest-sha256",
    _CONTRACT_SHA,
    "--repo-root",
    str(tmp_path),
    "--member",
    "tennis_000",
    "--seed",
    "0",
    "--start-frame",
    "0",
    "--output-dir",
    str(tmp_path / "new-run"),
    "--execute",
  ]


class _StubReport:
  def __init__(
    self,
    *,
    status: str = "complete",
    paired: bool = True,
    errors: tuple[str, ...] = (),
    output: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
  ) -> None:
    self.status = status
    self.paired = paired
    self.valid = status == "complete" and paired
    self.closed_loop = True
    self.synthetic = False
    self.errors = errors
    self.baseline = None
    self.diffusion = None
    self.output = output or {"representation": "marker", "error": None}
    self.provenance = provenance or {
      "integrity": {"unchanged": True},
      "execution": {"within_wall_budget": True},
    }

  def as_dict(self) -> dict[str, object]:
    return {
      "status": self.status,
      "paired": self.paired,
      "valid": self.valid,
      "closed_loop": self.closed_loop,
      "synthetic": self.synthetic,
      "branches": {
        "baseline": {"ticks": [{"tick": index} for index in range(3)]},
        "diffusion": {"ticks": [{"tick": index} for index in range(3)]},
      },
      "provenance": self.provenance,
      "errors": list(self.errors),
      "output": self.output,
    }


def test_parser_requires_frozen_paths_and_has_explicit_execute_flag() -> None:
  parser = evaluate.build_parser()
  args = parser.parse_args(
    [
      "--contract",
      "contract.yaml",
      "--contract-sha256",
      _CONTRACT_SHA,
      "--checkpoint",
      "checkpoint.pt",
      "--checkpoint-sha256",
      _CONTRACT_SHA,
      "--projection",
      "projection.npz",
      "--projection-sha256",
      _CONTRACT_SHA,
      "--vae-checkpoint",
      "vae.pt",
      "--vae-checkpoint-sha256",
      _CONTRACT_SHA,
      "--manifest",
      "manifest.yaml",
      "--manifest-sha256",
      _CONTRACT_SHA,
      "--repo-root",
      ".",
      "--member",
      "tennis_000",
      "--seed",
      "0",
      "--start-frame",
      "42",
      "--output-dir",
      "new-run",
    ]
  )
  assert args.execute is False
  assert args.member == "tennis_000"
  assert args.continuation_steps == 5000
  assert args.max_output_bytes == 128 * 1024 * 1024


def test_parser_loads_checkpoint_sidecar_defaults(tmp_path: Path) -> None:
  parser = evaluate.build_parser()
  run_dir = tmp_path / "d2-vgk8-double-20261009" / "seed0"
  run_dir.mkdir(parents=True)
  checkpoint = run_dir / "checkpoint-best.pt"
  checkpoint.write_bytes(b"placeholder")
  (run_dir / "checkpoint-best.yaml").write_text(
    "\n".join(
      [
        "schema: mjlab-d3-diffusion-checkpoint-v1",
        f"repo_root: {tmp_path}",
        "contract: contract.yaml",
        "projection: projection.npz",
        "vae_checkpoint: vae.pt",
        "manifest: manifest.yaml",
        "member: tennis_000",
        "seed: 0",
        "start_frame: 45",
      ]
    )
    + "\n",
    encoding="utf-8",
  )
  args = parser.parse_args(["--checkpoint", str(checkpoint), "--device", "cpu"])
  evaluate._validate_request(args)
  assert args.member == "tennis_000"
  assert args.seed == 0
  assert args.start_frame == 45
  assert args.contract == tmp_path / "contract.yaml"
  assert args.projection == tmp_path / "projection.npz"
  assert args.vae_checkpoint == tmp_path / "vae.pt"
  assert args.manifest == tmp_path / "manifest.yaml"
  assert args.output_dir == (
    tmp_path / "logs/diagnostics/diffusion-d3-visual/d2-vgk8-double-20261009"
  )


def test_missing_checkpoint_sidecar_fails_closed(tmp_path: Path) -> None:
  parser = evaluate.build_parser()
  checkpoint = tmp_path / "checkpoint-best.pt"
  checkpoint.write_bytes(b"placeholder")
  args = parser.parse_args(["--checkpoint", str(checkpoint), "--device", "cpu"])
  with pytest.raises(evaluate.ClosedLoopCliError, match="sidecar YAML is required"):
    evaluate._validate_request(args)

  for field, value in (
    ("continuation_steps", 5001),
    ("max_wall_seconds", 1201.0),
    ("max_output_bytes", evaluate.MIN_OUTPUT_CAP_BYTES - 1),
    ("max_output_bytes", 128 * 1024 * 1024 + 1),
  ):
    args = _request(tmp_path, **{field: value})
    with pytest.raises(evaluate.ClosedLoopCliError, match="must be"):
      evaluate._validate_request(args)


def test_existing_output_directory_is_allowed_for_overwrite(
  tmp_path: Path,
) -> None:
  output_dir = tmp_path / "already-there"
  output_dir.mkdir()
  args = _request(tmp_path, output_dir=output_dir)
  repo_root, device = evaluate._validate_request(args)
  assert repo_root == tmp_path.resolve()
  assert device.type == "cpu"


def test_artifact_digest_metadata_is_optional_and_not_validated(
  tmp_path: Path,
) -> None:
  parser = evaluate.build_parser()
  args = parser.parse_args(
    [
      "--contract",
      "contract.yaml",
      "--checkpoint",
      "checkpoint.pt",
      "--projection",
      "projection.npz",
      "--vae-checkpoint",
      "vae.pt",
      "--manifest",
      "manifest.yaml",
      "--repo-root",
      str(tmp_path),
      "--member",
      "tennis_000",
      "--seed",
      "0",
      "--start-frame",
      "0",
      "--output-dir",
      str(tmp_path / "new-run"),
    ]
  )
  assert args.contract_sha256 is None
  assert args.checkpoint_sha256 is None
  assert args.projection_sha256 is None
  assert args.vae_checkpoint_sha256 is None
  assert args.manifest_sha256 is None


def test_main_without_execute_never_dispatches_execution(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  fake = object()
  monkeypatch.setattr(evaluate, "_prepare", lambda args: fake)
  monkeypatch.setattr(
    evaluate,
    "_preflight_report",
    lambda prepared: {"ok": prepared is fake, "execution": "not-run"},
  )
  monkeypatch.setattr(
    evaluate,
    "_execute",
    lambda prepared: (_ for _ in ()).throw(AssertionError("execution dispatched")),
  )
  code = evaluate.main(
    [
      "--contract",
      "contract.yaml",
      "--contract-sha256",
      _CONTRACT_SHA,
      "--checkpoint",
      "checkpoint.pt",
      "--checkpoint-sha256",
      _CONTRACT_SHA,
      "--projection",
      "projection.npz",
      "--projection-sha256",
      _CONTRACT_SHA,
      "--vae-checkpoint",
      "vae.pt",
      "--vae-checkpoint-sha256",
      _CONTRACT_SHA,
      "--manifest",
      "manifest.yaml",
      "--manifest-sha256",
      _CONTRACT_SHA,
      "--repo-root",
      str(tmp_path),
      "--member",
      "tennis_000",
      "--seed",
      "0",
      "--start-frame",
      "0",
      "--output-dir",
      str(tmp_path / "new-run"),
    ]
  )
  assert code == 0
  assert '"execution": "not-run"' in capsys.readouterr().out


@pytest.mark.parametrize(
  ("report", "expected"),
  [
    (_StubReport(status="partial", paired=True, errors=("inference_error",)), 1),
    (
      _StubReport(output={"representation": "marker", "error": "write failed"}),
      1,
    ),
    (
      _StubReport(provenance={"integrity": {"unchanged": False}}),
      0,
    ),
    (
      _StubReport(provenance={"execution": {"within_wall_budget": False}}),
      1,
    ),
    (
      _StubReport(
        provenance={
          "integrity": {"unchanged": True},
          "execution": {"within_wall_budget": True},
          "outcome": "physical_fall",
        }
      ),
      0,
    ),
  ],
)
def test_execute_dispatch_returns_nonzero_for_software_failures_but_not_valid_falls(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  report: _StubReport,
  expected: int,
) -> None:
  monkeypatch.setattr(evaluate, "_prepare", lambda args: object())
  monkeypatch.setattr(evaluate, "_execute", lambda prepared: report)
  assert evaluate.main(_execute_args(tmp_path)) == expected
  capsys.readouterr()


def test_stdout_reads_persisted_marker_without_dumping_omitted_ticks(
  tmp_path: Path,
) -> None:
  output = tmp_path / "marker"
  output.mkdir()
  (output / "summary.json").write_text(
    '{"output":{"representation":"marker"},"branches":{"baseline":{"ticks":[]}}}\n',
    encoding="utf-8",
  )
  report = _StubReport()
  payload = evaluate._stdout_payload(report, output)
  assert payload["output"]["representation"] == "marker"
  assert payload["branches"]["baseline"]["ticks"] == []


def test_execute_reuses_public_bounded_persistence_after_integrity_update(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  output = tmp_path / "run"
  args = Namespace(
    output_dir=output,
    seed=0,
    start_frame=0,
    continuation_steps=1,
    max_output_bytes=evaluate.MIN_OUTPUT_CAP_BYTES,
    max_wall_seconds=1200.0,
  )
  model = SimpleNamespace(
    to=lambda **kwargs: model,
    eval=lambda: model,
  )
  prepared = SimpleNamespace(
    args=args,
    denoiser=SimpleNamespace(model=model),
    vae=SimpleNamespace(model=model),
    projection=SimpleNamespace(normalize_latent=lambda values: values),
    device="cpu",
    integrity_before={"state": 1},
  )
  report = _StubReport()
  calls: list[str] = []
  monkeypatch.setattr(evaluate, "_runtime_factory", lambda _: lambda: None)
  # The factory now accepts the optional D4 task conditioning argument; this
  # stub still returns an unused continuation for the no-task path.
  monkeypatch.setattr(
    evaluate, "_continuation_factory", lambda _prepared, _task=None: lambda: None
  )
  monkeypatch.setattr(evaluate, "_provenance", lambda *_: {})

  def fake_runner(*unused_args, **unused_kwargs):
    calls.append("runner")
    return report

  def fake_persist(value, destination):
    assert value is report
    assert destination == output
    calls.append("persist")
    return value

  monkeypatch.setattr(evaluate, "run_matched_pair", fake_runner)
  monkeypatch.setattr(evaluate, "persist_evaluation_report", fake_persist)
  result = evaluate._execute(prepared)
  assert result is report
  assert calls == ["runner", "persist"]


def test_source_fingerprints_use_imported_files_not_artifact_root(
  tmp_path: Path,
) -> None:
  provenance = evaluate._source_fingerprints()
  assert evaluate.__file__ is not None
  assert provenance["executed_cli"]["path"] == str(Path(evaluate.__file__).resolve())
  assert (
    provenance["modules"]["mjlab.tasks.tracking.diffusion.closed_loop_types"]["sha256"]
    == "6d603152f0685b51f98e8550dfb8f1ea6a03b926c88964d3b62e92bb4e38f129"
  )
  assert provenance["modules"]["mjlab.tasks.tracking.diffusion.closed_loop"][
    "path"
  ].endswith("src/mjlab/tasks/tracking/diffusion/closed_loop.py")
  assert all(
    not entry["path"].startswith(str(tmp_path))
    for entry in provenance["modules"].values()
  )
  assert provenance["source_checkouts"]
