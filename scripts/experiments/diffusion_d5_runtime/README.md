# D5 controller and asynchronous latent runtime

This directory documents the CPU software path for the D5 controller bridge.
The path is opt-in. It does not construct MuJoCo physics, publish to a robot,
or provide a safe action.

## Model-backed assembly

Use an explicit `ModelBackendSpec` and
`build_model_backed_controller_driver` from
`mjlab.tasks.tracking.diffusion.controller_driver`:

```python
from pathlib import Path

import torch

from mjlab.tasks.tracking.diffusion.controller_driver import (
  build_model_backed_controller_driver,
)
from mjlab.tasks.tracking.diffusion.model_backend import ModelBackendSpec

spec = ModelBackendSpec(
  contract_path=Path("docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml"),
  diffusion_checkpoint=Path("/path/to/checkpoint-best.pt"),
  normalization_bundle=Path("/path/to/statistics.npz"),
  vae_checkpoint=Path("/path/to/vae.pt"),
  vae_manifest=Path("/path/to/vae-manifest.yaml"),
  repo_root=Path("."),
  vae_member="member-0",
  planner_device="cpu",
  decoder_device="cpu",
)

generator = torch.Generator(device="cpu").manual_seed(7)
driver = build_model_backed_controller_driver(
  spec,
  generator=generator,
  read_observation=read_observation,
  execute_action=execute_action,
)
controller = driver.controller
controller.load()
```

The active representation has 135 physical state channels and 32 raw latent
channels. The denoiser therefore receives 167-channel tokens. The normalizer
is statistics-only. It does not apply a projection matrix or pseudoinverse.
`ModelBackendSpec.normalization_bundle` points to the statistics bundle.
Schema, dimensions, finite values and model interfaces remain mandatory load
checks. Missing or malformed paths fail closed. No standing, zero-action or
other synthetic model is substituted.

The model-backed assembly keeps the ownership split explicit. The owner
process creates the VAE decoder. The `BoundedPlannerExecutor` starts one
spawned worker, and that worker creates the denoiser and the
`LatentPlanner.process` callback. Normal controller ticks do not wait for
inference. Startup warmup is the only controller operation that waits for a
bounded planner completion.

## Complete lifecycle example

The caller supplies measured observations and the execution boundary. The
caller must execute eight bootstrap pairs through the external VAE and
acknowledge each pair. A bootstrap acknowledgement is not a queue receipt. It
must mean that the external execution boundary accepted the measured command
and that the pair is valid for history.

```python
# Each item contains one measured pre-action state and the 32-D VAE latent
# decoded for that state. The list is supplied by the caller, not fabricated by
# the controller.
for pair in measured_bootstrap_pairs[:8]:
  accepted = execute_bootstrap(pair)  # external VAE execution, literal bool
  assert type(accepted) is bool
  assert controller.acknowledge_bootstrap(
    pair.state,
    pair.raw_latent,
    success=accepted,
  )

# warmup() reads one fresh current observation. It runs owner-side decoder
# calls, waits for one real spawned planner result, validates it, and discards
# the result. The execution sink is not called by warmup.
report = driver.warmup(decoder_calls=3, timeout_seconds=10.0)
assert report.decoder_calls == 3
assert controller.state.value == "ready"

# Activation is explicit. The caller remains in its safe disengaged state until
# this point. No standing or zero action is supplied by this bridge.
driver.request_activation()

# Each tick reads one new measured observation. The sink is called only for a
# finite 31-D ACTION result. The caller owns the schedule and its safe hold for
# all non-action statuses.
dispatch = driver.tick()
if dispatch.execution.value == "accepted":
  assert dispatch.command is not None
  assert dispatch.command.action.shape == (31,)
elif dispatch.dispatch.value == "not_dispatched":
  hold_safe_state()

# A reset closes the exact old worker and invalidates old actions. A new load,
# eight new acknowledged measured pairs, warmup and explicit activation are
# required before tracking can resume.
driver.reset()
controller.load()
for pair in next_measured_bootstrap_pairs[:8]:
  assert controller.acknowledge_bootstrap(
    pair.state,
    pair.raw_latent,
    success=execute_bootstrap(pair),
  )
driver.warmup()
driver.request_activation()

# Shutdown is terminal and idempotent. It does not publish a fallback action.
driver.shutdown()
assert controller.cleanup_confirmed
```

`read_observation` must return a fresh `ControllerObservation` with an owned
`WorldState`, finite 99-D current conditioning, sample identity, source frame,
segment identity and a timestamp in the controller clock domain. The first
post-warmup tick must use a new sample. The driver calls `execute_action` only
for an active finite 31-D decoded command. The callback must return the literal
`bool` `True` to acknowledge the exact runtime step. A rejection, exception or
non-boolean receipt faults the controller and does not enter history. The
bridge never publishes raw 32-D latents.

The sink owns the physical handoff. A non-action runtime status means the
bridge publishes nothing. The caller owns its safe hold. This software example
does not prove that the external execution boundary is safe, and it does not
claim a 50 Hz deadline.

## Runtime contract

`AsyncLatentRuntime` owns a `RuntimeScheduler` and one planner executor. Its
normal step path:

1. snapshots the eight acknowledged measured state/latent pairs and the
   current observation;
2. submits one bounded planner request without waiting;
3. selects only the raw 32-D row for the execution tick;
4. validates plan identity, source age, fixed 33-row horizon and finite values;
5. decodes that row with the current finite 99-D conditioning; and
6. returns one finite 31-D action and its execution provenance.

A returned action enters history only after the execution callback returns
literal `True` and the controller acknowledges the exact `RuntimeStep`.
Warmup output is discarded. Future rows are not predecoded. Planner, decoder,
malformed-input/output, stale-plan, future-plan and horizon failures return
explicit no-action statuses or fault the lifecycle owner. The
`max_fallback_ticks` value is a structured warning threshold. It is not an
action limit.

`reset()` and `switch_policy()` clear old work and require a new load. `release()`
and `shutdown()` stop the exact owned executor. Cleanup is confirmed only when
`closed is True` and `worker_alive is False`. A failed worker factory or worker
callback cannot report readiness. The caller must inspect the cleanup marker
before replacing an owned runtime.

## CPU validation

Run the focused integration test with the repository environment. Do not sync
dependencies:

```sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv
export PYTHONPATH="$PWD/src:$PWD/tests"
uv run --no-sync pytest -p no:cacheprovider -x -vv \
  -o faulthandler_timeout=30 \
  tests/test_tracking_diffusion_controller_integration.py
```

The integration test uses synthetic states, statistics, denoiser and decoder.
It uses a real `BoundedPlannerExecutor` spawn worker and the actual
`LatentPlanner.process` callback. It checks eight acknowledged bootstrap
pairs, warmup sink suppression, fresh current conditioning, active 31-D
execution, exact acknowledgement, fallback warning telemetry, reset and
rebootstrap boundaries, failed-worker behavior and exact cleanup.

These checks provide CPU software integration evidence only. They do not
provide sustained timing, 50 Hz readiness, numerical or export parity,
physics, vendor simulator, `x2_docker`, deployment or hardware evidence.
