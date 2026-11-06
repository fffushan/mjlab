# X2 closed-loop latency sweep

One checkpoint, many latency cells, each run closed loop for a fixed horizon with
no `time_out` termination. The task is the pelvis ball-history juggling task, so
the latency cell *is* the actor's perception condition:
`observations["actor"].terms["ball_hist_pelvis"].params["stream_latency_s"]`.

This directory adds a runner and a thin loop. It modifies no existing file. It
imports the sibling E0 evaluator (`../x2_tennis_nominal_eval`) read-only for the
trace recorder, the qualified-flight scoring rule and the checkpoint loader, so
there is one recorder, one scoring rule and one policy loader in this repository.

## What one cell measures

The runner builds the task's environment, overrides the latency range in process,
removes `time_out`, and steps the policy with its deterministic mean (or a zero
action with `--zero-action`) for `horizon_s / step_dt` control steps. Every other
termination stays installed — `ball_dropped`, `ball_no_bounce`,
`ball_not_lifted`, `fell_over` — because inside this loop a termination **is** the
failure signal. The episode length is set to the horizon so `max_episode_length`,
which sizes the recorder buffers, matches the loop.

Two independent signals are reported, plus one new metric.

### Signal A — the recorded streak

`recorder.py`'s `nominal_trace` term records one episode per environment and stops
recording that environment after its first reset. One environment therefore gives
exactly one streak. Per environment the runner reports:

* the qualified flights in that streak and its longest qualified sequence,
* the apex of every qualified flight above the rest datum,
* the duration of every racket contact and of every qualified flight,
* the termination term that ended the streak.

### Signal B — resets

`env.reset_buf` is summed per environment over the whole horizon.
`resets_per_min` is `resets_total / (steps_run * step_dt / 60)`. It counts every
termination, including the auto-reset episodes the recorder does not score, so it
stays meaningful when signal A censors every environment. The trace recorder keeps
its own reset counter; the runner asserts the two agree once the pre-step resets
are added back, and writes the check as `reset_count_check`.

### Lateral landing error

For every racket contact segment in a streak, the ball position at that contact is
expressed in the plate (`face_center` site) frame, using the recorded plate
orientation. The sample is the control-step sample at or just after the contact's
first substep, which is the same sample rule the scoring module uses for its
takeoff and return-plate reads. The runner reports:

* `lateral_x_m`, `lateral_y_m`: the spread of the in-plane coordinates over the
  cell, with the sample standard deviation of each axis,
* `radial_fraction`: `sqrt((x/0.157)^2 + (y/0.117)^2)`, the radial fraction of the
  modeled plate half-extent, where `1.0` is the bed edge.

`radial_fraction` above 1.0 means the contact happened outside the modeled string
bed ellipse — the plate rim or the mesh beyond it — not that the measurement is
wrong. A contact is recorded wherever the ball touches the racket body.

## Censoring rule

An environment that does not terminate inside the horizon is **censored**, not a
failure. It is written as `censored: true` with `termination_term: null`, and its
streak is reported as the flights observed so far. Censoring is a lower bound, not
a success: it says the policy survived the remaining horizon.

`streak_qualified_flights_uncensored` and `streak_longest_sequence_uncensored` use
the uncensored environments only and are `null` when nothing terminated. The
`_all_envs` variants include censored environments and are the fallback view; there
the censored streak is a lower bound. Read the two together with `censored_envs`.

## One cell

Run from the repository root. The target apex height must match the task's reward
target: export `MJLAB_JUGGLE_ENERGY_TARGET_HEIGHT_M` when `--target` is not 0.40
(the shipped default). The runner refuses a mismatch instead of measuring one
target while the task rewards another.

```sh
uv run python scripts/experiments/x2_tennis_closed_loop/run_sweep.py \
  --task Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal-S1-15Hz \
  --checkpoint $HOME/data/tennis_s1_20261105/jobs/s1-15hz-s7/run/*/2026-11-05_*/model_6998.pt \
  --latency-min 0.08 --latency-max 0.10 --target 0.4 \
  --horizon-s 180 --num-envs 32 \
  --output-dir logs/diagnostics/x2_closed_loop/lat_0.08_0.10 --device cuda:0
```

The runner prints the resolved latency immediately after the override, then the
terminations left installed, then progress every tenth of the horizon, then a JSON
summary of the aggregates. It refuses a non-empty output directory.

## A whole sweep

`--cells` takes space-separated `MIN:MAX` pairs in seconds. One cell is one
directory under `--output-root`, holding `result.json`, `run.log` and
`exit-status`. A cell directory that already exists is refused (exit 11), and a
failing cell stops the sweep with that cell's status, so a partial sweep cannot
look complete.

```sh
bash scripts/experiments/x2_tennis_closed_loop/sweep.sh \
  --task Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle-29DOF-Ball3-Pelvis-Nominal-S1-15Hz \
  --checkpoint $HOME/data/tennis_s1_20261105/jobs/s1-15hz-s7/run/*/2026-11-05_*/model_6998.pt \
  --target 0.4 \
  --cells "0.00:0.00 0.04:0.04 0.08:0.08 0.10:0.10 0.12:0.12" \
  --output-root logs/diagnostics/x2_closed_loop/s1-15hz-s7 \
  --horizon-s 180 --num-envs 32 --device cuda:0
```

`sweep.sh` applies no wall bound, no GPU check and no retry: the caller owns those.
Bound the whole sweep externally and check GPU occupancy first.

## `result.json`

| Key | Meaning |
|---|---|
| `task`, `checkpoint`, `policy_kind`, `device` | what ran |
| `latency.requested_s`, `latency.resolved_s` | the requested range and the value the config held after the override |
| `latency.quantised_control_steps` | the same range in whole control steps, which is what the term can produce |
| `horizon_s`, `num_envs`, `steps_run`, `simulated_s`, `step_dt_s` | loop size |
| `removed_time_out`, `terminations` | the termination set the cell ran with |
| `target_m`, `resolved_reward_targets_m`, `rest_z_m` | the measurement datum and the task's reward datum |
| `episode_seed` | the shared release-state seed (`20261105`, the E0 seed) |
| `reset_count_check` | `env.reset_buf` total against the recorder's own reset total |
| `environments[]` | one record per recorded environment: `censored`, `steps`, `termination_term`, `resets`, `qualified_flights`, `longest_qualified_sequence`, `apex_above_rest_m`, `lateral_x_m`, `lateral_y_m`, `radial_fraction`, `racket_contact_durations_s`, `qualified_flight_durations_s`, `rejected_by_criterion`, … |
| `aggregates` | the cell-level numbers below |
| `limits` | the limit list repeated in file |

`aggregates` keys: `resets_total`, `resets_per_min`
(`resets_per_min_denominator_s`), `censored_envs` (`censored_env_indices`),
`uncensored_envs`, `termination_term_counts`,
`streak_qualified_flights_uncensored`, `streak_longest_sequence_uncensored`,
`streak_qualified_flights_all_envs`, `streak_longest_sequence_all_envs`,
`apex_above_rest_m`, `lateral_x_m`, `lateral_y_m`, `radial_fraction`,
`racket_contact_durations_s`, `qualified_flight_durations_s`,
`qualified_return_radial_fraction`.

Every distribution key holds `count`, `mean`, `sd`, `min`, `p50`, `p95`, `max`, or
`null` when there is no sample. `p50` is the median. **Apex sd** is `apex_above_rest_m.sd`; the
**lateral sd per axis** is `lateral_x_m.sd` and `lateral_y_m.sd`; the **radial
p50/p95** is `radial_fraction.p50` and `radial_fraction.p95`. `sd` is the sample
standard deviation (`ddof=1`) and is `null` for a single sample. Nothing is
reported as 0 when it was not measured.

## Smoke evidence (2026-11-05, GPU `cuda:0`)

Two bounded runs, 8 environments and 6 s each, against
`.../s1-15hz-s7/.../model_6998.pt` with `--latency-min 0.08 --latency-max 0.10`:

* Policy: exit 0, 300 steps, 8 resets (`resets_per_min` 80.0), 0 censored, streak
  2.5 qualified flights on average (all eight ended on `ball_dropped`), `apex sd`
  0.0300 m, `lateral sd` 0.0505 m / 0.0208 m, `radial p50/p95` 0.324 / 1.075, 69
  racket contacts. These are one run's numbers; see the reproducibility limit.
* `--zero-action`: exit 0, 40 resets (`resets_per_min` 400.0), 0 censored, streak
  0.0 qualified flights in all eight environments, `apex` and
  `qualified_flight_durations_s` `null` because no flight qualified. A zero-action
  policy cannot juggle, which is what the smoke had to show.

The zero-action run is a smoke only. It is not a control arm of any study.

## Known limits

* **One streak per environment.** The recorder stops at an environment's first
  reset, so a second streak in the same environment is never recorded, whatever
  the horizon. A long horizon buys more reset samples through signal B, not more
  recorded streaks through signal A.
* **Latency is quantised to 20 ms.** The observation term computes
  `round(latency / step_dt)` with `step_dt = 0.02 s`, so a requested 80-100 ms cell
  runs at 4-5 control steps. Two cells inside one rounding bin are the same
  condition, however they are named. The runner reports the quantised range next
  to the requested one.
* **Censoring is not success**, and a censored streak is a lower bound.
* **Resets per minute** counts every termination including unrecorded auto-reset
  episodes, so it and the streak counts answer different questions and are not
  expected to agree.
* **Not bit-reproducible.** `env.seed` fixes the release states, but `mujoco_warp`
  is not bit-reproducible (see `mjlab/utils/random.py`). Two identical 2 s smoke
  cells agreed in every count and differed by about `1e-7` m in the first contact
  position. Identical 6 s policy cells gave 66 to 69 racket contacts and a mean
  streak of 2.375 to 2.5 (the zero-action cells were repeatable, because the robot
  holds one pose). Small float differences grow in a chaotic contact system, so do
  not read a one-flight difference between two cells as an effect.
* **The thresholds are the plan's frozen values**, not a calibration against this
  sweep. The lateral spread is the spread of recorded contact samples, not of a
  controlled landing target.
* **Simulation only.** No hardware, deployment or sim-to-real claim.

## Files

| File | Role |
|---|---|
| `run_sweep.py` | one latency cell: config override, closed-loop rollout, both signals, `result.json` |
| `sweep.sh` | one `run_sweep.py` call per `MIN:MAX` cell, one directory per cell |
| `README.md` | this file |
