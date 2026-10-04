# D3 closed-loop evaluation

This is the integration-only entry point for one bounded, matched X2 pair:

1. eight deterministic frozen gravity-mode VAE bootstrap transitions;
2. a reference-tracking VAE baseline continuation; and
3. a continuation branch handed to the frozen 167-channel D2 denoiser without
   resetting or advancing the simulator.

The runner returns measurement evidence, not a policy-quality verdict. Fake
runtime tests are software evidence only; they are not physical closed-loop
evidence. No x2_docker, vendor simulator, hardware, export, training, or
remote job is used by this tool.

## Preflight and execution

Every artifact path, expected digest, selected cohort member, seed, start frame,
output directory, and resource budget is explicit. Omitting `--execute` performs
all frozen contract/checkpoint/projection/VAE/manifest loader checks and prints a
JSON preflight without constructing an environment. `--execute` is required to
construct mjlab simulation.

The active defaults are immutable: 167 token channels (135 state + 32 raw VAE
latent), 8 history pairs, 32 future steps, 20 DDIM updates, start level 1000,
`eta=0`, no guidance, FP32 model loading, and the original gravity VAE
convention (reference 68 / conditioning 99 / raw mean latent 32 / action 31).
Continuation uses measured runtime conditioning: projected gravity plus the
cached actor gyro, joint position, joint velocity, and previous applied action.

Hard CLI limits are continuation steps `1..250`, wall budget `(0,1200]` seconds,
and output cap `4096..128 MiB`; the output directory must not already exist.
The runner and CLI share the public `MIN_OUTPUT_CAP_BYTES` bound and the same
bounded persistence function. Each run is persisted as `full`, `compact`, or
`marker` representation with truthful omitted-row counts and actual byte
counts; after-run integrity and wall-budget booleans are retained in compact or
marker output. Stdout reads the persisted bounded summary rather than dumping
in-memory ticks when they were omitted. A report/inference/runtime failure,
output error, post-run integrity mismatch, or wall-budget failure exits nonzero.
A complete paired physical attempt, including a physical fall, remains exit 0.
The initial smoke budget is exactly 8 bootstrap + 25 evaluation ticks per
branch.

Source provenance fingerprints the executed CLI and imported module files from
their actual `__file__` locations, with source checkout heads kept separate
from the explicit artifact-resolution `--repo-root`. Passing a primary checkout
as the artifact root therefore does not make its source files current evidence.

Example simulator-free preflight (the command still loads the frozen models,
but never constructs the simulator):

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/diffusion_d3_closed_loop/evaluate.py \
  --repo-root "$PWD" \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
  --contract-sha256 f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed \
  --checkpoint /path/to/checkpoint-best.pt \
  --checkpoint-sha256 252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae \
  --projection /path/to/projection.npz \
  --projection-sha256 e258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe \
  --vae-checkpoint /path/to/mixed-10k/checkpoint-final.pt \
  --vae-checkpoint-sha256 69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117 \
  --manifest configs/distillation/x2_tennis_mixed.yaml \
  --manifest-sha256 587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d \
  --member tennis_000 \
  --seed 0 \
  --start-frame <floor-0.1-times-N-minus-1> \
  --output-dir logs/diagnostics/diffusion-d3-closed-loop/preflight-example
```

The public persistence API is shared by the runner and this CLI:

```python
from mjlab.tasks.tracking.diffusion.closed_loop import (
  MIN_OUTPUT_CAP_BYTES,
  persist_evaluation_report,
)

# The runner persists once; the CLI calls this again after final provenance.
persist_evaluation_report(report, output_dir)
```


After independent review, the parent may run one local headless GPU smoke using
`tennis_000`, seed `0`, and `START_FRAME=45=floor(0.1*(453-1))` from the
selected motion's preflight frame count. Use a fresh output directory and an
exact-owned
monitor; do not retry or enlarge the budget after a fall or timeout.

```sh
START_FRAME=45  # tennis_000 has N=453 frames
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/diffusion_d3_closed_loop/evaluate.py \
  --repo-root "$PWD" \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
  --contract-sha256 f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed \
  --checkpoint /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/checkpoint-best.pt \
  --checkpoint-sha256 252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae \
  --projection /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/projection.npz \
  --projection-sha256 e258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe \
  --vae-checkpoint "$PWD/logs/distillation/mixed-10k/checkpoint-final.pt" \
  --vae-checkpoint-sha256 69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117 \
  --manifest configs/distillation/x2_tennis_mixed.yaml \
  --manifest-sha256 587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d \
  --member tennis_000 \
  --seed 0 \
  --start-frame "$START_FRAME" \
  --continuation-steps 25 \
  --max-wall-seconds 1200 \
  --max-output-bytes 134217728 \
  --device cuda:0 \
  --output-dir logs/diagnostics/diffusion-d3-closed-loop/smoke-tennis000-seed0 \
  --execute
```

`summary.json` records both branches separately, including measured conditioning
source, handover/reference provenance, physical counters and partial outcomes,
checkpoint selected step versus the frozen 104000-update training budget, EMA
selection, sampler settings, actual imported source/environment fingerprints,
artifact root, and before/after model, normalizer, non-Tensor extra-state, and
frozen-file integrity. The persisted representation may omit tick arrays under
the shared cap; `ticks.jsonl` and `output.omitted_row_count` state that omission.
A low-tilt collapsed pose is not successful behavior, and height/contact/torque
fields remain descriptive or unavailable unless the runtime captures them with
known semantics.

## Preregistered serial pilot driver

`run_pilot.py` runs one frozen, preregistered headless batch through
`evaluate.py`: `tennis_000`, seeds `0/1/2` x start fractions `0.1/0.5` = six
matched pairs, eight actual VAE bootstrap transitions per branch, and at most
`250` post-bootstrap evaluation ticks per branch. Each condition's horizon is
`min(250, clip_frames - 2 - start_frame - 8)` with
`start_frame = floor(start_fraction * (clip_frames - 1))`; the frames and
horizons are declared before any physics runs, so neither branch attempts a
final-reference-frame action or clip wrap and the horizon is not tuned from
outcomes.

The driver never retries, retunes or expands the matrix. A physical fall,
reference boundary, bounded budget stop, pair mismatch or physical bootstrap
failure is an attempted outcome that is retained and does not stop later
preregistered conditions; a software, integrity, output or untrusted-runtime
failure stops the batch and preserves partial artifacts. Bounds are one serial
batch with a whole-batch wall cap of `1200` s and a whole-batch output cap of
`128` MiB (`16` MiB per condition), always one local GPU environment at a time.
Stdout is discarded, stderr is bounded, and a concise `progress.jsonl` plus
`pilot_manifest.json` and `batch_result.json` record every terminal result and
the exact frozen assets and source hashes.

Without `--execute` the driver is a simulator-free preflight: it verifies the
frozen hash gates, confirms the selected cohort member's clip length and rate,
and prints the preregistered plan without constructing an environment.

```sh
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" uv run --no-sync python \
  scripts/experiments/diffusion_d3_closed_loop/run_pilot.py \
  --repo-root "$PWD" \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
  --contract-sha256 f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed \
  --checkpoint /path/to/checkpoint-best.pt \
  --checkpoint-sha256 252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae \
  --projection /path/to/projection.npz \
  --projection-sha256 e258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe \
  --vae-checkpoint /path/to/mixed-10k/checkpoint-final.pt \
  --vae-checkpoint-sha256 69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117 \
  --manifest configs/distillation/x2_tennis_mixed.yaml \
  --manifest-sha256 587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d \
  --output-root logs/diagnostics/diffusion-d3-closed-loop/pilot-tennis000-6x \
  --device cuda:0
```

Add `--execute` to launch the six serial evaluator subprocesses under the
`1200` s / `128` MiB caps. The batch result is a recorded pilot comparison, not
an invented quality-pass verdict; matching, handover and predicted-versus-
realized evidence must be read from the per-condition `summary.json` artifacts.

## Browser visual diagnostic

`view.py` is a separate, opt-in authority next to `evaluate.py`. It runs
**one** operator-selected branch (`--branch baseline` or `--branch diffusion`)
so the operator can watch the robot move in a localhost Viser page. It reuses
the evaluator's explicit-path frozen loaders/preflight and the existing Viser
viewer, and it deliberately does **not** instantiate a second branch or a
matched comparison, so it makes no 1e-3 acceptance claim. Every status surface
is labelled diagnostic-only.

The viewer starts **paused**. Controls: `Play`/`Pause`, `Step` (advance exactly
one control tick while paused), `Reset Environment` (clears policy history and
performs a brand-new legitimate eight-step VAE bootstrap; generations are never
mixed), and `Slower`/`1x`/`Faster` playback. Camera navigation comes from the
shared Viser scene. The stock motion scrubber/seek controls are disabled and
rejected, so they can never mutate physical command state or reintroduce a
reference after handover. The status panel shows the selected branch/member,
`bootstrap` versus `reference_free` phase, tick/sim time, tilt and pelvis
height, bootstrap count, evaluation steps, and the stop/error reason. The final
stopped pose stays visible until the viewer resets it, the session wall
budget elapses, or the viewer closes. Playback is bounded by
`--continuation-steps` (`1..250`). `--max-wall-seconds` is a fixed **whole
session** budget: it starts before device/model load and also counts rendering,
paused/idle time, inference, stepping and explicit resets, so a user reset never
extends it. When it elapses the session ends and the launcher closes the
server, viewer and env; a fall, boundary, terminal flag or runtime error stops
the run with no automatic restart.

Simulator-free preflight (loads the frozen models, constructs no environment,
viewer or server):

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/diffusion_d3_closed_loop/view.py \
  --repo-root "$PWD" \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
  --contract-sha256 f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed \
  --checkpoint /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/checkpoint-best.pt \
  --checkpoint-sha256 252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae \
  --projection /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/projection.npz \
  --projection-sha256 e258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe \
  --vae-checkpoint "$PWD/logs/distillation/mixed-10k/checkpoint-final.pt" \
  --vae-checkpoint-sha256 69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117 \
  --manifest configs/distillation/x2_tennis_mixed.yaml \
  --manifest-sha256 587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d \
  --member tennis_000 --seed 0 --start-frame 45 \
  --branch diffusion \
  --output-dir logs/diagnostics/diffusion-d3-visual/preflight
```

Launching the local browser diagnostic (identical artifacts plus `--execute`):

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/diffusion_d3_closed_loop/view.py \
  --repo-root "$PWD" \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
  --contract-sha256 f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed \
  --checkpoint /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/checkpoint-best.pt \
  --checkpoint-sha256 252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae \
  --projection /home/agiuser/projects/worktrees/mjlab/d3-focused-20261003T185144Z/assets/projection.npz \
  --projection-sha256 e258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe \
  --vae-checkpoint "$PWD/logs/distillation/mixed-10k/checkpoint-final.pt" \
  --vae-checkpoint-sha256 69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117 \
  --manifest configs/distillation/x2_tennis_mixed.yaml \
  --manifest-sha256 587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d \
  --member tennis_000 --seed 0 --start-frame 45 \
  --branch diffusion \
  --host 127.0.0.1 --port 8080 \
  --continuation-steps 25 --max-wall-seconds 1200 \
  --output-dir logs/diagnostics/diffusion-d3-visual/tennis000-seed0 \
  --execute
# Open http://localhost:8080 (the server binds 127.0.0.1 only).
```

The bound server, viewer, runtime and env are closed on exit or on any failure;
a bounded `viewer-status.json` records the final diagnostic status. This is a
single-branch visual diagnostic only: it is not physical evidence and it does
not qualify the matched-pair gate.

## Opt-in D4 sparse task conditioning

The `--task-config PATH` flag is opt-in and additive. Without it the command
keeps the D3 unguided behaviour exactly: the same plan, sampler, Gaussian RNG
draws and actions, and no task provenance or per-tick task metadata in the
report. Report bytes are not literally identical to a pre-D4 run — the
fingerprint list truthfully includes the newly imported D4 modules — but no
sampling, decoding, guard or default metadata differs. With a task config, one
versioned, bounded JSON file adds sparse world-space targets to the diffusion
branch only:

- **hard keyframes** fix the complete 135-channel physical-state slice of a
  future row (its 32 latent entries stay unknown and are generated) through the
  frozen inpainting seam;
- **soft pose goals** add an explicit, unit-tagged physical cost whose input VJP
  steers the reverse loop.

A task config binds the frozen contract identity, the 20-body order, the
control period, `frame: world` and
`tick_origin: absolute_control_tick_including_bootstrap`, then carries at most
64 targets with at least 10 ticks (0.2 s) between two targets in the same list.
Unknown or duplicate fields, duplicate JSON keys, dense schedules, boolean or
non-integral ticks, non-finite or ill-shaped states, legacy schemas, target
latents/encoder outputs and fabricated "verified" provenance are all rejected
by the loader, which runs before any runtime or environment exists. A task file
and the recording files it binds are also added to the run's before/after
integrity expectations, so editing either during an attempt is detected.

Modes are `unguided`, `keyframe`, `poseguided` and `combined`, and each mode
owns its own target kind: `keyframe` schedules hard keyframes only,
`poseguided` soft pose goals only, and `combined` requires both. A hard keyframe
and a soft goal may not share a tick (the hard row would make the soft cost a
constant with zero gradient), while nearby targets on different ticks stay
allowed. `lambda = 0` and an unguided task both reproduce the D3 path bitwise.

The eight bootstrap pairs, the measured current conditioning, the stale/tick
guards, the sampler row-9 prediction metrics and the zero-encoder/zero-reference
after handover are unchanged. A trusted measured post-step observation —
including the final budget post and a trustworthy fall post — is reported to the
task lane so hard keyframes and soft goals are scored from real measurements;
planning ticks and measured observations are counted separately. `reset()`
clears the whole task ledger before the mandatory fresh bootstrap, so no result
leaks across an episode boundary. When a hard keyframe fixes the next row, that
tick's next-row metric is labelled `hard_keyframe_row` instead of
`denoiser_forecast` so learned prediction quality is not overstated.

Example configs live in `task_examples/`; both are hand-authored diagnostic
fixtures explicitly marked `synthetic_fixture` / `unverified`, and neither is a
physical measurement.

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/diffusion_d3_closed_loop/task_config.py \
  --task-config scripts/experiments/diffusion_d3_closed_loop/task_examples/synthetic_combined.example.json
```

### Preparing targets from a qualified recording

`prepare_task_config.py` copies chosen sparse target states out of an existing
qualified baseline recording. It never samples motion and never runs a
simulator, and it validates its own output with the strict loader before writing
it atomically to a new path. Two layouts are accepted, exclusively:

* `--recording-dir` reads an **actual D3 closed-loop run** (`summary.json` +
  `ticks.jsonl`), the layout of the archived `diffusion-d3-recorded-comparison-*`
  baselines. Only `baseline` rows are used; the summary must be a completed,
  paired, closed-loop, non-synthetic attempt whose baseline branch is the frozen
  VAE reference path (zero denoiser calls), rows must be contiguous in tick,
  segment and control period with matching post/pre measurements, and the
  recorded contract identity plus member/seed/start-frame must match the run.
  This is the only layout that can be `verified`.
* `--source-shard` reads a raw D1 NPZ shard (optionally store-bound). Its
  recorded provenance is weaker, so it stays `declared`.

Verification is performed by the **loader**, not trusted from the file: it
re-reads the declared recording and summary bytes, re-checks their digests, the
rows and the recorded scenario, and re-extracts every target state at its tick
before accepting the task. A hand-written `preparation_tool:` prefix therefore
cannot upgrade a source, and a missing, modified or scenario-mismatched
recording is refused. The digest and declared-record checks prove the recorded
bytes and their structural qualification; they are not a cryptographic or
independent authentication that the recording reflects physical execution.
Under `--execute` a non-verified source is refused, so a manual/synthetic or
`declared` task can only be used for CPU diagnostics and preflight.

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/diffusion_d3_closed_loop/prepare_task_config.py \
  --recording-dir /path/to/raw/seed0-start0045-frac0.1 \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml \
  --member tennis_000 --seed 0 --start-frame 45 \
  --mode combined --keyframe-ticks 18,38 --goal-ticks 28 \
  --strength 0.1 --root-position-weight 1.0 \
  --root-orientation-weight 1.0 --body-positions-weight 1.0 \
  --output logs/diagnostics/task.json
```

An extracted example from the archived `seed0-start0045-frac0.1` baseline is kept
at `orchestration/source-fix/d3-baseline-seed0-start0045-keyframes-poseguided.json`
(hard keyframes at ticks 18 and 38, one soft goal at 28, weights 1/1/1,
`strength 0.1`). That strength is illustrative, **not** a tuned or quality-
qualified value.

Realized target error is only ever measured at the target tick from the trusted
measured observation: targets that were never reached, fell in a gap, or expired
before the window are reported as `unreached` / `missing` / `expired`, never as
zero error and never as a censored success, and a post observation that is
untagged, non-contiguous in segment/tick/control period or reference-leaking
never enters the ledger. Predicted values and trivially-zero hard-inpainting
residuals are never reported as guidance improvement. A positive-strength run is
illustrative; this tool makes no model-quality, stability, real-time or
deployment claim.

## Software validation

From the integration worktree, the focused software gate is:

```sh
PYTHONPATH="$PWD/src" uv run --no-sync pytest -q \
  tests/test_tracking_diffusion_d4_integration.py \
  tests/test_tracking_diffusion_inpainting.py \
  tests/test_tracking_diffusion_pose_costs.py \
  tests/test_tracking_diffusion_guidance.py \
  tests/test_tracking_diffusion_closed_loop_cli.py \
  tests/test_tracking_diffusion_closed_loop_runtime.py \
  tests/test_tracking_diffusion_closed_loop.py \
  tests/test_tracking_diffusion_closed_loop_pilot.py \
  tests/test_tracking_diffusion_handover.py \
  tests/test_tracking_diffusion_visual_playback.py \
  tests/test_tracking_diffusion_d3_replay.py
```

The fake-runtime tests deliberately do not construct MuJoCo. The broader
isolated diffusion regression is a separate parent gate after hash-matched
fixture staging.
