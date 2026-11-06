# Nominal X2 juggling: deterministic evaluation

Date: 2026-11-05 (+08).
Status: **E0 accepted for the 0.40 m policies under the tested nominal conditions.**
Scope: simulation only; no training or hardware operation.

## Result

All three 0.40 m policies completed **34 consecutive qualified flights in every evaluated episode**.
Each policy passed 32/32 episodes. The predeclared requirement was at least 26/32 episodes with 10 consecutive flights.
There were **3264 qualified flights across 96 episodes from three trained policies**.
No policy episode ended through ball loss or robot fall.

The 0.20 m policies did not juggle repeatedly.
They raised the plate approximately 0.20 m and carried the ball with brief contact gaps.
All 96 episodes ended through `ball_no_bounce` after 2.98–3.22 s.

Both zero-action controls lost the ball after 1.18–1.20 s.
They had no qualified flights.

## Protocol and completion

The operator approved E0 before execution.
The evaluator used each policy's final `model_4999.pt` and deterministic mean actions.
It restored the trained observation normalizer and checked that its values did not change.
No scoring threshold was changed after viewing policy results.

| Item | Value |
|---|---|
| Policies | Targets 0.20/0.40 m × training seeds 7/17/27 |
| Controls | Zero policy-action offset at each target; position-control gains remain active |
| Episodes | 32 per arm; eight arms; **256 total** |
| Reset distribution | Same robot pose; common recorded ball jitter within ±5 mm |
| Episode cap | 20 s; only each environment's first episode is scored |
| Observation/action contract | Actor 112, critic 121, actions 29; pelvis Ball3 history |
| State recording | Ball/plate position, ball velocity, plate orientation every 20 ms |
| Contact recording | Racket and other-contact flags every 1 ms |
| Execution | `.48:32221`, GPU0; eight sequential arm processes |
| Batch interval | **12:30:12–12:34:27 (+08)**; 255 s |
| Exit evidence | Monitor #7, batch and all eight arms: **0** |
| Cleanup evidence | GPU0 idle; no evaluator process at 12:34:44 (+08) |

Initial ball position/velocity, plate position/orientation and environment origins match exactly across arms.
The 32 reset samples are distinct in ball lateral position, not robot pose.
The measured x/y jitter spans are 9.98/9.28 mm.

## Per-arm outcome

A qualified flight is a completed racket-contact → airborne interval → apex → racket-return sequence.
It must satisfy the frozen timing, rise, apex, contact-duration and plate-return checks.
The initial release fall and the final unfinished flight are not counted.

| Arm | Episodes | Duration range | Qualified flights per episode | Passing episodes | End cause |
|---|---:|---:|---:|---:|---|
| t020-s7 | 32 | 2.98–3.22 s | 0 | 0/32 | `ball_no_bounce` |
| t020-s17 | 32 | 2.98–3.16 s | 0 | 0/32 | `ball_no_bounce` |
| t020-s27 | 32 | 3.00–3.16 s | 0 | 0/32 | `ball_no_bounce` |
| t040-s7 | 32 | 20.00 s | 34 | **32/32** | `time_out` |
| t040-s17 | 32 | 20.00 s | 34 | **32/32** | `time_out` |
| t040-s27 | 32 | 20.00 s | 34 | **32/32** | `time_out` |
| zero-t020 | 32 | 1.18–1.20 s | 0 | 0/32 | `ball_dropped` |
| zero-t040 | 32 | 1.18–1.20 s | 0 | 0/32 | `ball_dropped` |

![Recorded ball and plate motion](../../logs/diagnostics/tennis_nominal_eval_20261105/analysis/figure_e0_behavior.png)

The upper row shows 0.20 m carrying. The second row shows repeated 0.40 m flights.
The controls lose the ball. Heights subtract each environment's fixed rest datum.

## Direct flight measurements at 0.40 m

All 3264 completed flights pass the frozen criteria.
These measurements come from raw trajectories and contacts, not from training rewards.

| Quantity | Required | Observed range |
|---|---|---|
| Airborne time | At least 0.15 s | **0.524–0.582 s** |
| Continuous bounding contact | At most 0.05 s | **0.009–0.010 s** |
| Upward then downward velocity | Required | Present in every qualified flight |
| Rise from sampled takeoff | At least 0.20 m | **0.2926–0.4060 m** |
| Apex above fixed rest datum | 0.35–0.45 m | **0.3866–0.4261 m** |
| Other contact during flight | None detected | Zero flagged substeps |
| Return within bed ellipse | Normalized radial distance ≤1 | **0.036–0.254** |

Median apex is **0.3964 m**. Mean apex per seed is approximately 0.3981, 0.3991 and 0.3996 m.
Median airborne time is 0.560 s.
The apex band has at least 23.9 mm of observed margin to its nearest limit.

Every episode has 35 racket strikes and 34 completed flights.
The last strike occurs at 19.464–19.726 s and opens another flight.
The 20 s cap ends that unfinished flight; it is neither accepted nor rejected as a completed candidate.
Thus, 34 is an observation within this time window, not a maximum policy capability.

## The 0.20 m failure mode is now observed

After the first second, the plate sits **0.1977–0.2015 m above its initial height**.
The mean ball-centre gap above the plate is **0.0326–0.0332 m**, close to the 0.0327 m ball radius.
The ball moves little vertically and rests on the raised plate.

Contact is intermittent at 1 ms resolution, with many 1–3 ms gaps.
After settling, every 20 ms control step still contains racket contact.
The step-level contact flag therefore stays high and generates no new rising edges.

`ball_no_bounce` restarts its timer only on a rising step-level edge.
In all 96 episodes, termination occurs exactly **2.500 s after the last rising edge**.
This explains why contact chatter does not prevent the no-bounce termination.
A zero continuous-hold count does not mean the ball was not carried.

Each episode contains one early airborne interval long enough to reach the 0.15 s test.
Its sampled rise is below the 0.10 m requirement. Later intervals are too short.
Even accepting that isolated near-miss could not create a sequence of 10 flights.

The energy reward permits this state: a stationary ball at the target world height has target energy.
The racket-lift penalty is disabled.
This shows how the 0.20 m policies satisfy the energy objective without repeated juggling.
It does not prove that a 0.40 m carry is outside the robot's workspace.

Training episodes at 0.20 m were much longer than these deterministic episodes.
The changed contact-edge pattern explains the no-bounce timing difference.
Exploration noise is a plausible cause of that pattern, but no sampled-action evaluation was run to test causality.

## What the control establishes

Zero action means zero position-policy offset, with the ordinary position controller active.
The robot moves toward its default joint targets, not necessarily the initial carry pose.
The plate drops before the first ball impact, then the ball leaves it and reaches the floor.

This is a specific zero-action baseline, not a zero-torque or perfectly fixed-racket experiment.
Its failure supports the distinction from the successful policies under this baseline.
It does not establish that every passive or fixed-pose controller must fail.

## Validation and review

- Subagents implemented and independently reviewed the evaluator before execution.
- Parent corrections passed **23 focused tests**, plus targeted lint and shell syntax checks.
- Parent launch guards preserve previous output on refused duplicates and bound total execution to 1800 s.
- Runtime checks confirm all configurations, checkpoint widths, finite observations/actions, and unchanged normalizer values.
- All 256 episodes have complete active masks and pre-reset terminal records.
- An independent subagent rebuilt contact segments and scored all flights without calling the supplied scorer.
- It matched all per-episode counts. The behavior-analysis subagent also reproduced the stored scores.
- The parent independently checked all **3264 qualified 0.40 m flights** from the raw arrays.
- No hash or checksum manifest was used. Thresholds and training source were unchanged.

## Limits

1. This is **nominal simulation validation**, not hardware or deployment validation.
2. There are three trained policies per target. The 96 episodes are not 96 independent policies.
3. Robot resets are fixed; only ±5 mm ball jitter varies. Wider resets and perception disturbances are not tested.
4. Episodes stop at 20 s. Longer-duration behavior is not measured.
5. Ball and plate state use a 20 ms grid. The terminal derived state can lag one physics substep.
6. Sampled takeoff rise is an approximation, not a general bound on exact takeoff height.
   Its observed margin is large. No claim depends on an exact takeoff reconstruction.
7. Other-contact flags cannot separate simultaneous racket/body contact during a strike.
   No flagged other contact occurs in the measured airborne intervals.
8. Return position uses a sampled plate frame and an ellipse around an inscribed collision polygon.
   All measured returns are far from the boundary.
9. Actions and torques are not in the raw trajectory. Torque saturation and control effort are not measured.

## Curriculum decision

**Accept the 0.40 m family as the simulation starting point. Do not use a lower-height ladder.**
The initial proposal was a ±5 → ±10 mm simulated release test before fine-tuning.
The operator then made real-robot evaluation the decisive acceptance gate.
The [implementation plan](../design/x2_tennis_nominal_curriculum.md) now prioritizes the missing controller contract and measured real sensor/reset conditions.
It uses supervised physical outcomes to select and accept each later training change.
The wider-reset simulation test remains an optional diagnostic, not the default next milestone.

No implementation, new training, deployment or robot trial is authorized by this planning update.

## Artifact locations

Remote: `/mnt/fushan/mjlab-juggle-eval-20261105/out/`.
Local: `logs/diagnostics/tennis_nominal_eval_20261105/`.

- [Batch summary](../../logs/diagnostics/tennis_nominal_eval_20261105/remote/batch_summary.json)
- [Trajectory figure](../../logs/diagnostics/tennis_nominal_eval_20261105/analysis/figure_e0_behavior.png)
- [Per-episode measurements](../../logs/diagnostics/tennis_nominal_eval_20261105/analysis/episode_metrics.csv)
- [Qualified-flight measurements](../../logs/diagnostics/tennis_nominal_eval_20261105/analysis/flights_qualified.csv)
- [Contact-gap histogram](../../logs/diagnostics/tennis_nominal_eval_20261105/analysis/contact_chatter_histogram.csv)
- [Evaluator runbook](../../scripts/experiments/x2_tennis_nominal_eval/README.md)

Offline reproduction, from `mjlab/`:

```sh
uv run python logs/diagnostics/tennis_nominal_eval_20261105/analysis/analyze_nominal_eval.py
```

The command reads saved arrays only. It does not start a simulator or policy.
