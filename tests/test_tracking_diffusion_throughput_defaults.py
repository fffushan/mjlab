"""D1 CLI speed defaults stay independent from scalar and cache behavior."""

from mjlab.scripts import diffusion
from mjlab.tasks.tracking.diffusion.runtime_x2 import _configure_policy_precision


def test_offline_defaults_and_single_worker_token_cache() -> None:
  parser = diffusion.build_parser()
  for command in ("inspect", "replay"):
    assert parser.parse_args([command, "--store", "s"]).jobs == 8
  assert (
    parser.parse_args(["build-index", "--dataset-dir", "d", "--contract", "c"]).jobs
    == 8
  )
  stats = parser.parse_args(
    ["stats", "--store", "s", "--assignments", "a", "--bundle", "b", "--report", "r"]
  )
  assert stats.jobs == 8 and stats.batch_windows == 512
  assert (
    parser.parse_args(
      ["build-token-cache", "--dataset-dir", "d", "--contract", "c"]
    ).jobs
    == 1
  )


def test_collection_automatic_defaults_do_not_refuse_scalar() -> None:
  parser = diffusion.build_parser()
  args = parser.parse_args(
    ["collect", "--output", "s", "--report", "r", "--max-envs", "1"]
  )
  assert args.stream_rows is None and args.compact_results is None
  assert args.tensor_dispatch is None
  assert diffusion._batched_only_option(args) is None
  legacy = parser.parse_args(
    [
      "collect",
      "--output",
      "s",
      "--report",
      "r",
      "--no-stream-rows",
      "--no-compact-results",
      "--no-tensor-dispatch",
      "--policy-device",
      "cpu",
    ]
  )
  assert legacy.stream_rows is False and legacy.compact_results is False
  assert legacy.tensor_dispatch is False and legacy.policy_device == "cpu"


def test_cuda_policy_defaults_pin_ieee_without_cpu_change(monkeypatch) -> None:
  import mjlab.utils.torch as backend

  calls = []
  monkeypatch.setattr(
    backend, "configure_torch_backends", lambda **kwargs: calls.append(kwargs)
  )
  _configure_policy_precision("cpu")
  assert calls == []
  _configure_policy_precision("cuda:0")
  assert calls == [{"allow_tf32": False}]


def test_collection_default_environment_count() -> None:
  args = diffusion.build_parser().parse_args(
    ["collect", "--output", "s", "--report", "r"]
  )
  assert args.max_envs == 256
