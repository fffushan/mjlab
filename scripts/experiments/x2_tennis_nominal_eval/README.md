# X2 nominal juggling E0 evaluation

Deterministic evaluation of the six nominal juggling policies and two zero-action
controls. This directory is the whole evaluator; it does not modify the task,
the reward, the physics or the controller.

**Completed E0 (2026-11-05 +08):** 256 episodes; all three 0.40 m policies passed
32/32 episodes with 34 consecutive qualified flights each. See the
[accepted report](../../../docs/reports/x2_tennis_nominal_evaluation_20261105.md)
for the measured scope, sampling limits and next proposal.

Source of truth for the budget, the rule and the gate:
[`docs/design/x2_tennis_nominal_curriculum.md`](../../../docs/design/x2_tennis_nominal_curriculum.md),
section "Preserved E0 baseline". Later curriculum stages use the separate physical
acceptance process in that plan; this evaluator alone cannot promote them.

## Budget

| Item | Value |
|---|---|
| Arms | 6 policies (0.20/0.40 m x seeds 7/17/27) + 2 zero-action controls |
| Episodes per arm | 32, one per environment, first episode only |
| Episode cap | 20 s |
| Parallel environments | 32 per arm (32 total, arms run sequentially) |
| Total episodes | 256 |
| GPUs | one, rechecked for occupancy before the batch starts |
| Wall bound | 1800 s for the whole batch, including interpreter and CUDA startup |

Zero action is a zero position-policy action offset: the default joint pose held by
the PD controller. It is not zero torque.

## What each arm changes relative to the saved training config

The environment comes from the task registry and is compared field by field against
the run's own `params/env.yaml` before the environment is built. Exactly these
differences are allowed, and anything else aborts the arm:

| Path | Runtime | Reason |
|---|---|---|
| `scene.num_envs`, `scene.terrain.num_envs` | 32 | 32 scored episodes, not 4096 training envs |
| `seed` | `20261105` | one episode seed shared by all eight arms |
| `scene.sensors` | + `ball_any_contact` | read-only 1 ms contact sensor, see below |
| `recorders` | + `nominal_trace` | the trace recorder |
| `scene.entities.robot.spec_fn`, `scene.terrain.spec_fn` | unchanged | `yaml` truncates a lambda qualname |

Rewards, terminations, events, physics, actuators, observation terms and the action
space must match the saved file exactly, including the three energy-target fields
that carry the arm's target height.

## Contract checks at arm startup

* `mjlab.__file__` must resolve inside the staged source tree (`--expect-source-root`).
* `MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M` must be set to the arm's target before the
  process starts, no other `MJLAB_*` variable may be set, and the reward and both
  energy metrics must resolve to that same target.
* The checkpoint must be actor 112 in / 29 out, critic 121 in, and must carry an
  observation normalizer.
* The built environment must report actor 112, critic 121, actions 29.
* The policy must be in eval mode. Every observation-normalizer mean/variance
  value and its count must be identical before and after the rollout. Values are
  compared directly; no hashes or checksums are computed.
* Every action and every actor/critic observation must be finite at every step;
  every recorded float array must be finite.

## Determinism and matched resets

Each arm seeds the environment with `EPISODE_SEED = 20261105` **after** the policy is
loaded, then resets once. The wrapper's own construction reset is discarded. Policy
construction draws random initial weights, so without the reseed the two control arms
would start from different states than the trained arms. `summarize_batch.py`
re-reads all eight initial ball positions/velocities, plate positions/orientations,
and environment origins. It fails the batch (exit 21) if values differ by more than 1e-6.

Policies act with the deterministic mean output, not samples.

## Recording and temporal alignment

`recorder.py` adds one contact sensor and one recorder term:

* `ball_any_contact`: MuJoCo contact sensor over the ball body with no secondary
  filter, one control step of substeps in its history. Non-racket contact is
  `any_contact and not racket_contact` per 1 ms substep. There is no ground geofence
  and no height proxy: the floor and every robot body are seen through the same
  channel. **Limit:** a substep in which the ball touches the racket *and* something
  else is attributed to the racket.
* `nominal_trace`: copies ball world position and velocity, the plate
  (`face_center` site) world pose, both contact histories, the first-episode active
  mask and every registered termination term into device buffers, once per control
  step, and stops recording an environment after its first reset.

The terminal row is captured in `record_pre_reset`, i.e. after the metrics are
computed and before `_reset_idx`. No post-reset state is scored. The accessors used
there are derived quantities that `env.step()` refreshes only after the reset block,
so they lag the integrated state by one 1 ms substep - the same lag the termination
decision sees.

Contact flags are per 1 ms substep, concatenated in step order, so a contact that
spans a control-step boundary is one run: contact durations carry across steps and
are never capped at the 20 ms control step. Ball, plate and velocity samples are per
control step (20 ms); a ballistic apex between samples is under-read by at most
`g*dt^2/8 = 0.5 mm` during ballistic motion. The takeoff reference can be up to one
control step late. Treat sampled rise as an approximation, not a general bound on
exact takeoff height. E0's observed margin is recorded in the accepted report.

## Scoring rule

`scoring.py` applies the plan's seven conditions to the recorded trace. A flight from
racket contact `k` to contact `k+1` qualifies only if: the next contact exists in the
same episode; the airborne interval is at least 0.15 s with no non-racket contact;
vertical velocity goes from upward to downward; the ball rises at least half the
target above its actual takeoff position; the apex above the fixed rest datum is
within target +-0.05 m; both bounding contacts last at most 0.05 s continuously; and
the returning ball is inside the modeled bed ellipse in plate coordinates, using the
recorded plate orientation. An unfinished returning contact at the episode boundary
is rejected because its full duration is unknown.

Flights are counted from racket contact to racket contact, so the initial release
fall can never be a candidate. A rejected flight, a hold, a loss (`ball_dropped`), a
fall (`fell_over`) or the episode end breaks the consecutive count. Total contact
time is **not** used as a park rule.

`PLAN_THRESHOLDS` copies the plan's provisional numbers verbatim. Do not tune them
after reading a success rate.

### Gate

An episode passes the gate when it reaches 10 consecutive qualified flights and has
no hold, no loss and no fall. An arm passes when at least 26 of its 32 episodes pass.
The gate is reported per arm and per target; a different episode count fails the gate.
The two targets are never pooled, and
the 32 reset samples are not 32 independent policies.

## Files

| File | Role |
|---|---|
| `evaluate_arm.py` | one arm: contract checks, config validation, rollout, artifacts |
| `recorder.py` | the read-only contact sensor and the trace recorder |
| `scoring.py` | qualified-flight rule, arm summary, matched-state check |
| `config_check.py` | runtime-versus-saved config comparison |
| `run_batch.sh` | the eight-arm batch with one wall bound, lock and GPU check |
| `summarize_batch.py` | cross-arm table, release-state check, per-target view |

Arm artifacts, per arm: `trace_raw.npz`, `episodes.json`, `episodes.csv`,
`arm_summary.json`, `arm_metadata.json`, `trajectories.png`.

## Staging (one short command)

The trained source and the checkpoints stay where they are and are not copied or
modified: `SRC = /mnt/fushan/mjlab-juggle-nominal-20261104/src` and
`JOBS = /mnt/fushan/mjlab-juggle-nominal-20261104/jobs`. Only this directory is
copied, into a fresh stage directory:

```sh
tar -C /home/agiuser/projects/mjlab/scripts/experiments \
    --exclude=__pycache__ -cf - x2_tennis_nominal_eval \
  | ssh -p 32221 fushan@10.14.64.48 \
      'mkdir -p /mnt/fushan/mjlab-juggle-eval-20261105/code \
       && tar -C /mnt/fushan/mjlab-juggle-eval-20261105/code \
              --strip-components=1 -xf -'
```

Then confirm the copy landed and nothing else is in the stage:

```sh
ssh -p 32221 fushan@10.14.64.48 \
  'ls -l /mnt/fushan/mjlab-juggle-eval-20261105/code; \
   ls -l /mnt/fushan/mjlab-juggle-eval-20261105'
```

## Launch (parent, through a monitor)

Reverify GPU occupancy first; the script rechecks it immediately before the first
arm and fails closed with exit 10 if the GPU is busy. Recommended host and GPU:
`.48:32221` GPU 0. At 11:49 (+08), both managed containers had idle cards.
That snapshot is not permission to use a card without a fresh check.

```sh
ssh -p 32221 fushan@10.14.64.48 \
  'GPU=0 bash /mnt/fushan/mjlab-juggle-eval-20261105/code/run_batch.sh'
```

The batch applies its own 1800 s bound (`timeout --signal=TERM --kill-after=60`), so
a monitor bound of 2400 s is enough. Do not add a second timeout that kills the SSH
session earlier: the batch writes `<out>/exit-status` on the way out.

```sh
# Recheck occupancy without touching anything.
ssh -p 32221 fushan@10.14.64.48 \
  'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

## Reading the result

* `<out>/exit-status`: 0 ok, 10 preflight refused, 11 duplicate/stale lock, 12 would
  overwrite, 143 inner TERM (outer timeout exits 124 on the wall limit),
  20 an arm failed, 21 release states disagree, 22 summary failed.
  **Absent means unknown, not success.** Check `preflight-fail` for refusals.
  A refused duplicate does not change the existing batch's status or progress.
* `<out>/progress.jsonl`: one JSON line per arm start/end and per batch event.
* `<out>/arm-exit-status/<arm>`, `<out>/logs/<arm>.log`: per-arm status and output.
* `<out>/batch_summary.json`: per-arm table, per-target gate view, release-state
  comparison.

A rerun needs a new stage or output directory; the batch refuses to overwrite an
existing one and never retries an arm by itself.

## Not tested by this directory

* No GPU or policy rollout was executed while preparing it. First contact with a
  checkpoint, the CUDA-graph warm-up cost, and the real wall time of 256 episodes are
  unmeasured.
* The thresholds are the plan's provisional numbers; they are not calibrated against
  a measured qualified-flight rate.
* There is no hardware, deployment or sim-to-real claim.
