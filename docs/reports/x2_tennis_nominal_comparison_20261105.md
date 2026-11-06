# Nominal X2 juggling: six-run curve comparison

Date: 2026-11-05 (+08).
Status: curve analysis complete. The later [E0 evaluation](x2_tennis_nominal_evaluation_20261105.md)
accepted the 0.40 m policies under nominal simulation conditions; curve-only limits below remain applicable.

## Recommendation

**Evaluate the 0.40 m policies next. Do not start a 0.20 → 0.40 m height curriculum.**
The 0.40 m statistics are consistent with repeated flights on the expected time scale.
The 0.20 m statistics show very short flights despite reaching the energy target.
This suggests a raised-racket, short-hop behavior, but no rollout has confirmed it.

[Conditional curriculum plan](../design/x2_tennis_nominal_curriculum.md)
starts with deterministic evaluation, not new training.

## Comparison

Six policies: targets 0.20/0.40 m × seeds 7/17/27.
Every run used 4096 environments and completed 5000 iterations.
Compare iterations **4500–4999**, with 500 points per arm.
Values below are means across three seed-window means, with the seed range in parentheses.

| Training statistic | Target 0.20 m | Target 0.40 m |
|---|---:|---:|
| Episode return | 145.70 (145.65–145.77) | 164.17 (163.84–164.39) |
| Episode duration | 19.956 s (19.945–19.962) | 20.000 s (all seeds) |
| Racket-contact edges per episode | 177.54 (174.57–182.08) | 35.06 (35.03–35.11) |
| Mean airborne interval | 0.02321 s (0.02234–0.02468) | 0.54919 s (0.54820–0.54992) |
| Mean vertical ball energy | 1.9558 J/kg (1.9523–1.9611) | 3.8764 J/kg (3.8737–3.8779) |
| Target energy | 1.962 J/kg | 3.924 J/kg |
| Pose reward | 0.00360 (0.00199–0.00477) | 0.92621 (0.91388–0.93513) |

Each target's three seeds show the same large difference in flight duration.
A theoretical equal-height flight with a 0.40 m rise lasts **0.571 s**.
The measured training mean near 0.549 s makes the 0.40 m policies useful evaluation candidates.
It is not an apex measurement or proof of consecutive successful returns.

![Six-run curves](../../logs/diagnostics/tennis_nominal_comparison_20261105/fig1_behaviour_overview.png)

Thin lines are raw points. Thick lines use a centered 100-iteration mean.
Blue is target 0.20 m; red is target 0.40 m. Duration is converted from steps to seconds.

### Late progress

From iterations 2000–2499 to 4500–4999:

- At 0.40 m, airborne intervals increase **3.32–4.52%** in all seeds.
- At 0.40 m, step contact edges decrease **3.75–5.21%** in all seeds.
- At 0.20 m, both quantities change in different directions across seeds.
- Energy changes little in either family. Behavior convergence is not established.

There is no measured basis for an extra-training duration or a predicted future success rate.

## Why the 0.20 m target is not necessarily easier

Energy uses a fixed world-height datum:

`E = 0.5 vz² + g (z − z_rest)`.

A ball held 0.20 m above its rest datum has the full 0.20 m target energy without a flight.
The nominal task has no active racket-lift penalty: its weight is zero.
The 0.20 m drop also starts at the target energy.
Thus, high energy reward does not require learned repetition at that target.
The same holding possibility exists at 0.40 m if the robot can raise the plate that far.
The different flight statistics, not energy alone, motivate evaluation of the 0.40 m policies.

The total-return gap is almost matched by the pose-reward contribution.
It must not be reported as a measured difference in juggling success.
Low pose reward indicates departure from the default joint pose, not measured plate height.

## Metric limits and review corrections

- Contact edges use a 20 ms step flag. Flight durations use 1 ms substeps.
  Multiple short contacts can merge into one step edge.
  Their aggregates cannot reconstruct contact duty or the actual number of substep flights.
- `mean(E)/g` bounds the matched mean height above the datum. It does not measure apex height.
- `g × mean(flight_time)² / 8` assumes equal-height ballistic flights.
  Release flights and moving contact heights prevent direct apex reconstruction.
- `ball_sustained_contact` carries continuous contact across steps.
  It is **not capped at 20 ms**, contrary to the reviewer drafts.
  A CPU check of the staged helper returns 20 then 40 ms for two fully contacting windows.
- The final 0.40 m contact-duration value near 10 ms is not a software ceiling.
- Termination values are overlapping counts per reset call, not failure probabilities.
- No logged scalar measures consecutive qualified contact–flight–apex–return sequences.

The parent corrected two draft report errors: late-trend signs and mixed contact/flight denominators.
The parent also fixed a plot that labelled control steps as seconds without conversion.
Derived apex and duty claims were removed or relabelled.
All plots were regenerated after those corrections.

## Validation evidence

Three subagents used `ds-oc/deepseek-flash:high`: extraction, metric audit, and independent review.
The parent performed the final numerical and source checks.

- All 50 tags have 5000 finite points in all six arms. No gap or duplicate step was found.
- Full saved YAML text differs only in seed, run name and three energy-target fields.
- Eleven relevant staged source files match the local source by direct text comparison.
- Raw scalar, log and independent final-window calculations agree.
- Six logs confirm **112 actor inputs / 121 critic inputs / 29 actions** from the existing task contract.
- Every exit-status file is `0`. Final checkpoint presence was verified at 03:32 (+08).
- No checkpoint was loaded in this analysis.

Source base: `31211d9fa` plus the uncommitted nominal implementation.
Remote artifacts: `/mnt/fushan/mjlab-juggle-nominal-20261104/jobs/<arm>/`.
Selected saved configurations and logs are retained locally with the analysis.
The staged-source comparison checks the files observed now; it is not a tamper-proof execution record.
No hashes or checksum manifests were used.

## Artifacts

Local root: `logs/diagnostics/tennis_nominal_comparison_20261105/`.

- [Full reviewed analysis](../../logs/diagnostics/tennis_nominal_comparison_20261105/analysis.md)
- [Per-seed tables](../../logs/diagnostics/tennis_nominal_comparison_20261105/per_seed_windows.csv)
- [Paired differences](../../logs/diagnostics/tennis_nominal_comparison_20261105/paired_differences.csv)
- [Energy curves](../../logs/diagnostics/tennis_nominal_comparison_20261105/fig2_energy_vs_target.png)
- [Raw late-window curves](../../logs/diagnostics/tennis_nominal_comparison_20261105/fig6_late_window_raw.png)
- [Parent verification](../../logs/diagnostics/tennis_nominal_comparison_20261105/parent_verification.txt)

Reproduce from `mjlab/`:

```sh
uv run python logs/diagnostics/tennis_nominal_comparison_20261105/analyze.py
uv run python logs/diagnostics/tennis_nominal_comparison_20261105/parent_verify.py
```

Large raw artifacts remain under the local diagnostic directory, not Git.
The report and plan preserve the main findings separately.

## Next action and authorization

The proposed 256-episode batch has now completed. The [E0 report](x2_tennis_nominal_evaluation_20261105.md)
confirms 34 consecutive qualified flights in all 96 target-0.40 m episodes.
The 0.20 m policies carried the ball; controls lost it.

The [implementation plan](../design/x2_tennis_nominal_curriculum.md) now makes real-robot evaluation the decisive acceptance gate.
First resolve controller-contract compatibility and measure real sensor/reset conditions.
Select fine-tuning changes from physical evidence, with simulation used for screening.
No implementation, training, deployment or hardware action is authorized by this plan.
