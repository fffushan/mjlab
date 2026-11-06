# Nominal X2 juggling pilot

Date: 2026-11-04 (+08).
Status: all six pilots completed successfully; verified 2026-11-05 03:32 (+08).

## Question and bounded batch

Can the pelvis-history policy learn repeated juggling under nominal conditions?
Does the 0.20 m target learn when the 0.40 m target does not?

Run six independent fresh policies: targets 0.20/0.40 m × seeds 7/17/27.
Each run: one GPU, 4,096 environments, 5,000 iterations, save every 500, TensorBoard.
Use four idle GPUs on the managed four-card container and two idle GPUs on the
managed eight-card container. Recheck actual occupancy and health before launch.
Keep each seed's two heights on the same server: seeds 7/17 on four-card, seed 27
on eight-card. Use fewer GPUs if fewer are free; never evict a job.
Known endpoints to verify: `.48:32221` (four-card), `.37:31505` (eight-card).
No fleet g1–g4 use. No curriculum scheduler, controller changes, or hardware operations.

## Small configuration

Add one opt-in `...-Juggle-29DOF-Ball3-Pelvis-Nominal` task by wrapping the existing
pelvis-history config, not copying it. Preserve all existing task defaults.

- Zero linear/angular movement commands; no command curriculum.
- Remove pushes and physics/encoder randomization events. Use nominal physical values.
- No arm/waist reset perturbation; fixed head reset pose. Keep the robot free-standing.
- Disable actor observation corruption and any artificial observation delay.
- Ball stream: 50 Hz, zero latency/stalls/bias/noise; still three positions and ages.
- Use centred drop starts: height 0.20 m, lateral jitter ±0.005 m,
  zero lateral velocity and zero pre-release elapsed time. No plate starts yet.
- Preserve measured contact physics, nominal gains, actuator limits, geometry,
  reward weights, and all corrected loss/no-bounce terminations.
- Target changes between arms only. Use the existing energy-target override and
  keep energy reward/metrics targets consistent. Default nominal target may be 0.20 m.

Expose only the few currently hardcoded noise/reset arguments needed by this config,
with old values as defaults. No global monkeypatches or generic curriculum framework.
The ball-history term applies its own noise: disabling the observation manager's
corruption alone is not sufficient. Verify the term itself is noise-free.

## Validation

Critical tests only: nominal resolved settings and old defaults; zero-noise history;
nominal reset placement. Reuse existing tests. No hash checks or new manifests.
Local `uv run`; focused tests <=120 s. No repeated identical timeout retries.
Before six full runs, one bounded GPU smoke per server (32–64 envs, 2 iterations).
Validate dimensions 112/29/121, finite progress, checkpoint, and exit status.
Read each full run's saved env/agent settings; launch lines alone are not evidence.
The pilot is not a plate-versus-pelvis comparison or a hardware-readiness test.

## Deployment and launch boundaries

Use a new isolated source directory under `/mnt/fushan/`; shared storage is visible
from both containers. Never overwrite the main checkout or active sweep worktrees.
Stage once, including this uncommitted implementation; no commit or Git push requested.
Use the existing server interpreter directly, never remote `uv run`.
Set `PYTHONPATH` explicitly and verify `mjlab.__file__` points into the staged source.
Record plain source revision, changed paths, launch commands, and resolved settings.
No hashes, hash verification, or checksum manifests.

Each job needs its own output directory, log, exact owned PID/session identity,
exit-status file, and an 8-hour wall limit with bounded cleanup. Previous measured
4096-env iteration costs suggest roughly 4–6 hours for 5k; remeasure after startup.
Keep jobs independent and preserve each training command's exit code.

Use parent MonitorCreate for long commands if unavailable to the launcher subagent.
The launcher must request parent-managed monitors with exact prepared commands,
then verify startup. No untracked detached jobs or shell sleep polling loops.
Prefer tmux for remote connection survival, with a monitor following its log until
its exact pane process exits. Do not build a generic launch/monitor framework.
Report all GPU mappings, paths, monitor IDs, and startup evidence.

## Delegation

Use `ds-oc/deepseek-flash:high` throughout.
Implementation and read-only server preflight may run in parallel.
Then perform a narrow independent review and launch from the reviewed source.
Parent owns NOW.md and the durable workflow. Child agents must not edit them.

## Launch and startup evidence

Implementation and independent review used `ds-oc/deepseek-flash:high`.
The parent confirmed all seven critical nominal tests passed.
Both server GPU smokes completed with exit 0, finite observations, and saved checkpoints.
The nominal source changes remain uncommitted; base revision is `31211d9fa`.

Staged source and artifacts: `/mnt/fushan/mjlab-juggle-nominal-20261104/`.
Both hosts use its `src/` through explicit `PYTHONPATH` and the existing platform interpreter.
The main checkout and the three existing eight-card sweeps were not changed.
A plain detailed record is at `<stage>/LAUNCH-RECORD.md`.

| Arm | Server | Configured GPU | Target | Seed |
|---|---|---:|---:|---:|
| t020-s7 | `.48:32221` | 0 | 0.20 m | 7 |
| t040-s7 | `.48:32221` | 1 | 0.40 m | 7 |
| t020-s17 | `.48:32221` | 2 | 0.20 m | 17 |
| t040-s17 | `.48:32221` | 3 | 0.40 m | 17 |
| t020-s27 | `.37:31505` | 6 | 0.20 m | 27 |
| t040-s27 | `.37:31505` | 7 | 0.40 m | 27 |

All six saved configurations were checked: 4,096 environments, 5,000 iterations,
checkpoint interval 500, correct seed and all three energy targets, zero commands,
no physics/encoder randomization, noise-free 50 Hz ball history, and nominal resets.
At 23:39 (+08), each run was near iteration 70, at approximately 2.9–3.0 s/iteration.
Startup rate suggests completion around 03:45–04:00 on 2026-11-05 (+08), not a guarantee.
Each trainer has an eight-hour limit and its own exit-status file.

One process per authorized GPU is confirmed. The four-card container reports outer
namespace GPU PIDs, so its exact arm-to-index mapping is configured but not independently
matched by PID. Eight-card GPUs 6/7 are matched to their arm processes.

Per-arm paths are `<stage>/jobs/<arm>/logs/train.log`, `run/`, and `exit-status`.
Remote sessions are `nom-<arm>` and survive loss of the local SSH log follower.
The batch command is `bash /tmp/nominal-launch/launch_all_six.sh`; reconnecting it
follows existing sessions rather than starting duplicate training.
Monitor #3 started the batch. After startup handoff, only its local log followers
were stopped and reattached as Monitor #4. The remote tmux jobs were not restarted.

## Completion evidence — 2026-11-05 03:32 (+08)

All six arms have `exit-status` equal to `0`, final logged iteration `4999/5000`,
and a nonempty `model_4999.pt` file (5,458,037 bytes each).
All six pilot tmux sessions are gone; no pilot trainer process remains.
The four-card server's GPUs 0–3 and the eight-card server's GPUs 6–7 are idle
(1 MiB and 0% utilization). The existing sweeps still occupy eight-card GPUs 0–5.

| Arm | Finished on 2026-11-05 (+08) | Exit status | Final checkpoint |
|---|---|---:|---|
| t020-s7 | 03:30:35 | 0 | model_4999.pt |
| t040-s7 | 03:09:21 | 0 | model_4999.pt |
| t020-s17 | 03:31:53 | 0 | model_4999.pt |
| t040-s17 | 03:06:35 | 0 | model_4999.pt |
| t020-s27 | 03:31:37 | 0 | model_4999.pt |
| t040-s27 | 03:11:55 | 0 | model_4999.pt |

Finish times come from each arm's `logs/train.log` end marker.
Final checkpoints are under each arm's existing `run/<experiment>/<timestamp>_<arm>/`.
This check confirmed file presence and size, not checkpoint loading or policy behavior.

Monitors #4/#5 lost their SSH connections while the trainers continued.
Monitor #6 followed the existing jobs to completion and exited 0.
No job was restarted or extended. The completion wake is complete; no new monitor is needed.

## Curve analysis — 2026-11-05 (+08)

[Reviewed comparison and curves](../reports/x2_tennis_nominal_comparison_20261105.md)
cover all six arms and all 5000 iterations.
Final-window mean airborne intervals are **0.02321 s at 0.20 m** and **0.54919 s at 0.40 m**.
The curve analysis identified the 0.40 m family as the first evaluation candidate.
The 0.20 m curves suggested raised-racket short hops; the later evaluation resolved this as a carry with contact chatter.

## Deterministic evaluation — 2026-11-05 (+08)

[Accepted E0 report](../reports/x2_tennis_nominal_evaluation_20261105.md):
all **96 target-0.40 m episodes** completed **34 consecutive qualified flights** each.
The **96 target-0.20 m episodes** carried the ball, then ended through `ball_no_bounce` at 2.98–3.22 s.
Both controls lost the ball. All 256 episodes completed; independent raw reconstruction confirmed the scores.
This validates nominal simulation with fixed robot pose and ±5 mm ball jitter, not hardware or wider-reset robustness.

**Next:** [real-robot-gated implementation plan](x2_tennis_nominal_curriculum.md).
First establish the compatible pelvis-history controller path and measured real input/reset conditions.
Physical trials decide curriculum acceptance; the earlier ±10 mm simulator test is now an optional diagnostic.
No implementation, training, deployment or robot operation is authorized by the plan.
