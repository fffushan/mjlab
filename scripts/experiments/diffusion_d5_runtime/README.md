# D5 asynchronous latent runtime

This directory documents the CPU-only D5 runtime facade. The facade connects the
bounded planner scheduler, the raw 32-dimensional latent plan, and the current
99-dimensional X2 decoder input. It does not start a simulator, load a frozen
asset, use a GPU, or activate a controller.

## Runtime contract

`AsyncLatentRuntime` owns a `RuntimeScheduler` and receives a
`PlannerExecutor`. The recommended concrete executor is
`BoundedPlannerExecutor`. Its worker factory must be pickleable and must return
a callback that accepts `PlannerRequest` and returns a scheduler `Plan`.
`LatentPlanner.process` is the component callback for a real planner.

At each `step` call, the facade:

1. snapshots the eight measured state/latent pairs, current state, task
   settings, source frame and request-local random state;
2. queues one planner request without waiting for inference;
3. polls the scheduler and selects only the raw 32-D row for the execution
   tick;
4. validates the plan schema, identities, fixed 33-row horizon, originating
   request tick/timestamp, source age and clock domain;
5. decodes that one latent with the current finite 99-D conditioning; and
6. returns one finite 31-D action and `ExecutionProvenance`.

Raw latent rows are never published as robot commands. Future rows are not
predecoded. A returned action is not entered into history until the caller
confirms it with `runtime.acknowledge(step, success=True)`. A failed
acknowledgement, failed decode, planner failure, stale plan, future plan,
malformed plan, or exhausted plan returns an explicit no-action status. There
is no standing or robot fallback.

Call `acknowledge_bootstrap(..., success=True)` after each measured VAE
bootstrap execution. The facade requires eight successful pairs before it
submits diffusion planning. `reset()` and `switch_policy()` clear history and
old work. A task revision change clears old task work but retains measured
history. `release()` and `shutdown()` prevent later decoding and stop the exact
owned executor resources. Cleanup is confirmed only when the concrete executor
reports both `closed=True` and `worker_alive=False`.

## Clock and identity settings

`clock` is an injected monotonic callable. `source_timestamp` is captured from
that same callable. `max_source_age_seconds` and
`max_future_skew_seconds` are finite, non-negative limits. The optional
`expected_source_frame` rejects plans from another source frame. The runtime
also checks the active no-mixture contract, root frame, VAE identity, model
identity, projection identity, task revision and task mode.

The D4 task snapshot contains absolute keyframe and pose-goal ticks. It carries
no live scoring ledger. D4 hard masks and the soft-guidance input-vector
Jacobian path remain in the latent planner component.

## CPU checks

Use the repository environment without dependency synchronization:

```sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/tests"
uv run --no-sync pytest \
  tests/test_tracking_diffusion_scheduler.py \
  tests/test_tracking_diffusion_latent_planning.py \
  tests/test_tracking_diffusion_planner_executor.py \
  tests/test_tracking_diffusion_async_latent_runtime.py \
  -p no:cacheprovider -x -vv -o faulthandler_timeout=30
```

The tests use synthetic states, statistics, denoisers and decoder callbacks.
One test uses the real CPU spawn executor. The checks provide software
integration evidence only. They do not provide D5 timing, simulator, export,
deployment or hardware evidence.
