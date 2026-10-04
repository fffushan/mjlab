"""CPU-only tests for the preregistered D3 pilot driver.

These tests exercise the frozen matrix, the deterministic horizon, the batch
stop/continue policy, the bounded subprocess capture and the progress records.
They never construct a simulator, allocate a GPU environment or claim physical
evidence.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections.abc import Mapping
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

_PILOT_PATH = (
  Path(__file__).parents[1]
  / "scripts/experiments/diffusion_d3_closed_loop/run_pilot.py"
)
_PILOT_SPEC = spec_from_file_location("d3_closed_loop_pilot", _PILOT_PATH)
assert _PILOT_SPEC is not None and _PILOT_SPEC.loader is not None
pilot = module_from_spec(_PILOT_SPEC)
sys.modules[_PILOT_SPEC.name] = pilot
_PILOT_SPEC.loader.exec_module(pilot)


def _summary(
  *,
  status: str = "complete",
  paired: bool = True,
  closed_loop: bool = True,
  baseline_stop: str = "step_budget",
  diffusion_stop: str = "step_budget",
  integrity: bool = True,
  within_wall: bool = True,
  output_error: object = None,
) -> dict[str, object]:
  return {
    "status": status,
    "paired": paired,
    "closed_loop": closed_loop,
    "branches": {
      "baseline": {"stop_reason": baseline_stop},
      "diffusion": {"stop_reason": diffusion_stop},
    },
    "provenance": {
      "integrity": {"unchanged": integrity},
      "execution": {"within_wall_budget": within_wall},
    },
    "output": {"error": output_error},
  }


def _fake_runner(
  summaries: Mapping[int, dict[str, object] | None],
  *,
  elapsed_s: float = 1.0,
):
  def run(spec, condition_dir: Path, timeout_s: float, output_cap: int):
    condition_dir.mkdir(parents=True)
    summary = summaries.get(spec.index)
    if summary is not None:
      (condition_dir / "summary.json").write_text(json.dumps(summary))
    return pilot.ConditionExecution(
      returncode=0,
      timed_out=False,
      elapsed_s=elapsed_s,
      stderr_bytes=0,
      stderr_truncated=False,
    )

  return run


def _plan(tmp_path: Path) -> Any:
  assets = {
    role: {"path": str(tmp_path / role), "sha256": "a" * 64}
    for role in (
      "contract",
      "diffusion_checkpoint",
      "projection",
      "vae_checkpoint",
      "vae_manifest",
    )
  }
  return pilot.PilotPlan(
    repo_root=tmp_path,
    contract=tmp_path / "contract",
    checkpoint=tmp_path / "diffusion_checkpoint",
    projection=tmp_path / "projection",
    vae_checkpoint=tmp_path / "vae_checkpoint",
    manifest=tmp_path / "vae_manifest",
    assets=assets,
    clip_frames=453,
    clip_fps=50.0,
    conditions=pilot.build_conditions(453),
    device="cuda:0",
    max_wall_seconds=pilot.MAX_WALL_SECONDS,
    max_output_bytes=pilot.MAX_OUTPUT_BYTES,
  )


def test_conditions_are_the_preregistered_tennis_matrix() -> None:
  conditions = pilot.build_conditions(453)
  assert [spec.index for spec in conditions] == list(range(6))
  assert [
    (spec.seed, spec.start_frame, spec.continuation_steps) for spec in conditions
  ] == [
    (0, 45, 250),
    (0, 226, 217),
    (1, 45, 250),
    (1, 226, 217),
    (2, 45, 250),
    (2, 226, 217),
  ]


@pytest.mark.parametrize("clip_frames", [40, 100, 453])
def test_horizon_keeps_the_last_post_frame_below_the_boundary(
  clip_frames: int,
) -> None:
  for spec in pilot.build_conditions(clip_frames):
    assert spec.continuation_steps >= 1
    assert spec.continuation_steps <= pilot.MAX_CONTINUATION_STEPS
    assert (
      spec.start_frame + pilot.BOOTSTRAP_PAIRS + spec.continuation_steps
      <= clip_frames - 2
    )


def test_execution_bounds_are_fail_closed() -> None:
  def args(**updates: object) -> Namespace:
    values: dict[str, object] = {
      "max_wall_seconds": pilot.MAX_WALL_SECONDS,
      "max_output_bytes": pilot.MAX_OUTPUT_BYTES,
    }
    values.update(updates)
    return Namespace(**values)

  pilot._validate_bounds(args())
  for bad in (
    {"max_wall_seconds": pilot.MAX_WALL_SECONDS + 1.0},
    {"max_wall_seconds": 0.0},
    {"max_output_bytes": pilot.MAX_OUTPUT_BYTES + 1},
    {"max_output_bytes": pilot.MIN_OUTPUT_CAP_BYTES - 1},
  ):
    with pytest.raises(pilot.PilotError):
      pilot._validate_bounds(args(**bad))


def test_condition_command_is_the_frozen_single_pair_execution(
  tmp_path: Path,
) -> None:
  plan = _plan(tmp_path)
  spec = plan.conditions[0]
  command = plan.condition_command(spec, tmp_path / "out", 16 * 1024 * 1024)
  assert command[0] == sys.executable
  assert command[1].endswith("scripts/experiments/diffusion_d3_closed_loop/evaluate.py")
  assert command[command.index("--member") + 1] == "tennis_000"
  assert command[command.index("--continuation-steps") + 1] == "250"
  assert command[command.index("--max-output-bytes") + 1] == str(16 * 1024 * 1024)
  assert command[command.index("--device") + 1] == "cuda:0"
  assert command[-1] == "--execute"


def test_manifest_records_the_declared_matrix_and_frozen_bounds(
  tmp_path: Path,
) -> None:
  manifest = _plan(tmp_path).as_manifest()
  assert manifest["member"] == "tennis_000"
  assert manifest["clip_frames"] == 453
  assert manifest["bootstrap_pairs"] == 8
  assert "- 8" in manifest["horizon_formula"]
  assert len(manifest["conditions"]) == 6
  assert manifest["condition_output_bytes"] == 16 * 1024 * 1024
  assert manifest["max_output_bytes"] == 128 * 1024 * 1024
  assert manifest["frozen_preserved"]["token_channels"] == 167
  assert manifest["frozen_preserved"]["ddim_updates"] == 20


def test_batch_continues_after_bounded_physical_outcomes(tmp_path: Path) -> None:
  conditions = pilot.build_conditions(453)
  summaries = {
    0: _summary(),
    1: _summary(status="partial", paired=False, diffusion_stop="physical_fall"),
    2: _summary(
      status="partial",
      paired=False,
      baseline_stop="pair_mismatch",
      diffusion_stop="pair_mismatch",
    ),
    3: _summary(),
    4: _summary(),
    5: _summary(),
  }
  result = pilot.run_batch(
    conditions,
    output_root=tmp_path,
    run_condition=_fake_runner(summaries),
  )
  assert result["conditions_attempted"] == 6
  assert result["stop_reason"] is None
  assert [record["classification"] for record in result["records"]] == [
    "paired_complete",
    "physical_fall",
    "pair_mismatch",
    "paired_complete",
    "paired_complete",
    "paired_complete",
  ]


def test_first_tick_physical_fall_is_a_bounded_outcome_not_untrusted(
  tmp_path: Path,
) -> None:
  conditions = pilot.build_conditions(453)
  summaries = {
    index: _summary(
      status="partial",
      paired=False,
      closed_loop=False,
      baseline_stop="physical_fall",
      diffusion_stop="physical_fall",
    )
    for index in range(6)
  }
  result = pilot.run_batch(
    conditions,
    output_root=tmp_path,
    run_condition=_fake_runner(summaries),
  )
  assert result["conditions_attempted"] == 6
  assert result["stop_reason"] is None
  assert result["records"][0]["classification"] == "physical_fall"


def test_batch_stops_on_software_failure_and_keeps_earlier_records(
  tmp_path: Path,
) -> None:
  conditions = pilot.build_conditions(453)
  summaries = {
    0: _summary(),
    1: _summary(
      status="partial",
      paired=False,
      baseline_stop="runtime_error",
      diffusion_stop="runtime_error",
    ),
  }
  result = pilot.run_batch(
    conditions,
    output_root=tmp_path,
    run_condition=_fake_runner(summaries),
  )
  assert result["conditions_attempted"] == 2
  assert result["stop_reason"] == "software_failure:runtime_error"
  assert result["records"][0]["classification"] == "paired_complete"
  assert result["records"][1]["batch_action"] == "stop"


@pytest.mark.parametrize(
  ("summary", "reason"),
  (
    (_summary(integrity=False), "integrity_unverified_or_changed"),
    (_summary(within_wall=False), "per_condition_wall_budget_exceeded"),
    (_summary(closed_loop=False), "untrusted_runtime"),
    (_summary(output_error="write failed"), "output_error"),
  ),
)
def test_batch_stops_on_untrusted_evidence(
  tmp_path: Path, summary: dict[str, object], reason: str
) -> None:
  conditions = pilot.build_conditions(453)
  result = pilot.run_batch(
    conditions,
    output_root=tmp_path,
    run_condition=_fake_runner({0: summary}),
  )
  assert result["conditions_attempted"] == 1
  assert result["stop_reason"] == reason


def test_batch_stops_when_the_global_wall_budget_is_exhausted(
  tmp_path: Path,
) -> None:
  conditions = pilot.build_conditions(453)
  now = {"value": 0.0}
  summary = _summary()

  def clock() -> float:
    return now["value"]

  def runner(spec, condition_dir: Path, timeout_s: float, output_cap: int):
    condition_dir.mkdir(parents=True)
    (condition_dir / "summary.json").write_text(json.dumps(summary))
    now["value"] += 700.0
    return pilot.ConditionExecution(0, False, 700.0, 0, False)

  result = pilot.run_batch(
    conditions,
    output_root=tmp_path,
    run_condition=runner,
    max_wall_seconds=1200.0,
    clock=clock,
  )
  assert result["conditions_attempted"] == 2
  assert result["stop_reason"] == "wall_budget_exhausted"


def test_batch_stops_when_the_global_output_budget_is_spent(
  tmp_path: Path,
) -> None:
  conditions = pilot.build_conditions(453)
  state = {"used": 0}

  def measure(_root: Path) -> int:
    return state["used"]

  def runner(spec, condition_dir: Path, timeout_s: float, output_cap: int):
    condition_dir.mkdir(parents=True)
    (condition_dir / "summary.json").write_text(json.dumps(_summary()))
    state["used"] += 60 * 1024 * 1024
    return pilot.ConditionExecution(0, False, 1.0, 0, False)

  result = pilot.run_batch(
    conditions,
    output_root=tmp_path,
    run_condition=runner,
    max_output_bytes=128 * 1024 * 1024,
    measure_bytes=measure,
  )
  assert result["conditions_attempted"] == 3
  assert result["stop_reason"] == "output_budget_exhausted"


def test_progress_records_every_terminal_condition_result(tmp_path: Path) -> None:
  conditions = pilot.build_conditions(453)
  events: list[dict[str, object]] = []
  result = pilot.run_batch(
    conditions,
    output_root=tmp_path,
    run_condition=_fake_runner({index: _summary() for index in range(6)}),
    progress=events.append,
  )
  kinds = [event["event"] for event in events]
  assert kinds.count("condition_start") == 6
  assert kinds.count("condition_end") == 6
  assert kinds[-1] == "batch_end"
  assert len(result["records"]) == 6


def test_subprocess_capture_preserves_returncode_and_bounds_stderr(
  tmp_path: Path,
) -> None:
  stderr_path = tmp_path / "condition.stderr.txt"
  command = [
    sys.executable,
    "-c",
    "import sys; print('discarded'); sys.stderr.write('e' * 100); sys.exit(7)",
  ]
  execution = pilot.run_evaluator_subprocess(
    command, cwd=tmp_path, timeout_s=60.0, stderr_path=stderr_path
  )
  assert execution.returncode == 7
  assert execution.timed_out is False
  assert stderr_path.read_bytes() == b"e" * 100


def test_last_condition_failure_has_nonzero_batch_exit(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  args = Namespace(
    execute=True,
    output_root=tmp_path / "pilot",
    max_wall_seconds=1200.0,
    max_output_bytes=pilot.MAX_OUTPUT_BYTES,
  )
  parser = Namespace(parse_args=lambda argv: args)
  monkeypatch.setattr(pilot, "_parser", lambda: parser)
  monkeypatch.setattr(pilot, "build_plan", lambda args: _plan(tmp_path))
  monkeypatch.setattr(
    pilot,
    "run_batch",
    lambda *args, **kwargs: {
      "schema": "test-batch",
      "conditions_planned": 6,
      "conditions_attempted": 6,
      "stop_reason": "software_failure:runtime_error",
      "wall_seconds": 1.0,
      "output_bytes": 0,
      "records": [],
    },
  )
  assert pilot.main([]) == 1


def test_subprocess_timeout_is_killed_and_recorded(tmp_path: Path) -> None:
  command = [sys.executable, "-c", "import time; time.sleep(30)"]
  execution = pilot.run_evaluator_subprocess(
    command,
    cwd=tmp_path,
    timeout_s=0.5,
    stderr_path=tmp_path / "condition.stderr.txt",
  )
  assert execution.timed_out is True
  assert execution.returncode is None
