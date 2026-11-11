# Review: simulated tennis rebound report

Date: 2026-11-11 22:06 (+08).
Reviewer scope: source inspection, compile-time option checks, and bounded CPU diagnostics.
No report, experiment source, production model, training, or hardware changes.

Reviewed report: [simulated restitution](x2_tennis_sim_restitution_20261111.md).
Design: [arm compliance](../design/x2_tennis_arm_compliance.md).
Repository base: `28f422d4e`. The reviewed additions are uncommitted.

## Verdict

**“No model change yet” survives. The report is not ready to circulate as validation evidence.**

The fixture shows that the existing finite-gain arm can substantially reduce rebound.
It does not validate the training simulator against the robot.
It does not bound the benefit of a passive mount model.
Fix the fixture, analysis, and paired hardware evidence before selecting a model change.

## 1. Method validity — report sections 1, 2, 3.1, 3.2, 7, 8

### F1 — Constant targets are a useful diagnostic, not a demonstrated equivalent protocol

Gravity-compensated targets are reasonable for a local, stationary-pose arm-impedance probe.
The 0.0007 rad residual checks equilibrium position. It does not check dynamic impedance or actuator work.
The welded base removes physical support motion. The standing policy can update targets before, during, and after impact.
First-apex agreement cannot independently validate these assumptions; that is the outcome under test.

The fixture is therefore **valid as a bounded diagnostic**, but **not established as a matched hardware substitute**.
Bound the effect with recorded target/base motion or a matched controller/support comparison.
Do not reject the diagnostic entirely, and do not label it “the same protocol.”

### F2 — Blocking: the fixture does not apply the task's simulation options

`model.py:57-58` compiles an entity and sets only the timestep.
It never calls `cfg.sim.mujoco.apply(model)`, which the environment simulation path uses.
A compile-time comparison produced:

| Option | Report fixture | Task config |
| --- | ---: | ---: |
| timestep | 0.001 | 0.001 |
| integrator | Euler (`0`) | implicitfast (`3`) |
| iterations | 100 | 10 |
| line-search iterations | 50 | 20 |
| CCD iterations | 35 | 50 |

CCD means continuous collision detection.
Both use Newton and the pyramidal cone, but that does not erase these differences.
Actuator damping is numerically different under Euler and implicitfast.
Apply the resolved task options and retain the option dump before calling the result “current simulation.”
Then compare the CPU fixture with the training backend and check timestep convergence.

### F3 — Blocking: the advertised measurement-pipeline equivalence is false

`drop.py:129-150` rewrites the analyser rather than calling the owner's implementation.
Its extrema loop includes index 1. The owner's loop excludes the first and last three samples.
The simulation starts with a descending ball and no held-release segment.
Zero padding in `convolve(..., mode="same")` creates its initial maximum at index 1.
A diagnostic confirmed maxima beginning `[1, 67, ...]`; the owner's boundary rule begins at `[67, ...]`.
Thus, the first-bounce input maximum depends on an artificial boundary event that the owner rejects.

Prepend a valid held-release segment, or supply the known release state through an explicitly different first-drop estimator.
Verify one shared analyser against both data formats.
The band reference also differs: the script uses a minimum ball-centre height, while the owner passes a surface height.
The radius offset changes which minima pass the band.
`tangential.py` uses full-rate velocities and contact-window extrema, not the reported 100 Hz pipeline at all.

### F4 — Trials do not restore one complete initial state

`drop.py:192-194` settles once per gain condition, not once per release height.
`tangential.py:25-56` resets the ball but leaves the arm state from the previous trial.
Neither restores a complete saved state for every release.
A fixed release-position correction also assumes that the racket remains at the original face pose.

Restore a settled snapshot, including relevant solver/actuator state, for each independent trial.
Check pre-impact joint/face velocity and actual impact location.
Report exclusions and retain per-trial records. These effects are **not bounded** by the current scripts.

### F5 — Blocking: first-apex agreement lacks a consistent paired height definition

Section 3.2 gives robot rows `(h_drop, apex, e_h)` of `(0.84, 0.24, 0.47)` and `(~0.21, 0.04, 0.28)`.
If these are the two heights in the stated equation, they imply **0.535** and **0.436**, respectively.
They do not imply 0.47 and 0.28.

If “apex” uses another reference, define and convert it before comparing it with simulation `h_reb`.
If the columns summarize different bounces, label that and supply paired rows; the 5 mm comparison cannot use unpaired summaries.
The held-release `0.45`, 18-bounce cohort may legitimately differ from the owner's all-bounce `0.39`, 34-bounce cohort.
However, the referenced robot report does not contain the new cohort selection or paired rows.
Pin the exact bounce IDs, release classes, heights, and analysis command.

The duplicate radius in `z_ref + radius + d` is acknowledged in section 7, but must still be fixed.
Reporting the observed `h_drop` does not undo the changed physical release condition.

## 2. The 0.06 claim — report sections 3.3, 4.2, 9

### F6 — Blocking: the wrist-gain ablation supplies no upper bound on mount compliance

The exact supplied command reproduces `0.519` at shipped gains and `0.582` at high gains.
That establishes a **0.063 sensitivity to that gain intervention in this fixture**.
It does not establish a maximum possible effect of another physical model.

Reasons:

- A finite-gain wrist is not a rigid constraint. The torque limits remain active.
- The compiled wrist pitch/roll limit remains ±4.8 N m; wrist yaw remains ±24 N m.
- `set_wrist()` changes both wrists, not only the racket-holding wrist.
- A separate compliant mode changes the frequency-dependent impedance, contact duration, and energy-return timing.
- The stiffening intervention explores neither the parameter space nor the dynamic modes of a passive mount.

Delete “at most about 0.06” and the repeated bound in section 9.
Also replace “removes most of the ball's energy” with a scoped rebound observation.
The ablation does not partition dissipated energy from recoil, elastic storage, and actuator work.
The 0.9 m condition is near 4.2 m/s, not the approximately 2.5 m/s juggling condition.

## 3. Tangential experiment — report sections 2, 3.5, 5.2, 7, 8

### F7 — Blocking: the saved script cannot run

`tangential.py:3` uses `pathlib.Path` without importing `pathlib`.
`uv run ruff check --select F821 scripts/experiments/x2_tennis_drop/` fails on this exact line.
The reported tangential outputs therefore need a runnable script and retained per-trial provenance.
This does not prove the outputs were never generated; it prevents reproducing them from the submitted source.

### F8 — Similar velocity-change norms do not show insensitive outgoing velocity

`dv_xy` is `norm(v_out_xy - v_in_xy)`. It is not `norm(v_out_xy)`.
Even a constant vector change gives `v_out = v_in + constant`: outgoing velocity still depends on incoming velocity.
A nearly constant impulse magnitude can also occur in a sliding-friction regime.
It does not prove missing tangential sensitivity or correct tangential contact.

The two speed bands overlap. Random headings and median norms hide directional dependence.
World `xy` is not the tangent plane of a tilted, moving racket.
Use paired trials at fixed headings and a speed grid. Report signed input/output vectors.
Also report ball–racket relative contact-point slip, including spin, in the instantaneous contact frame.
Compare with hardware using the same measurement estimator and matched initial conditions.
The acceptable current wording is only: “The two sampled bands have similar median horizontal velocity-change magnitudes.”

The launch offset is a useful first estimate. It does not prove a centred impact on a moving face.
The event detector must require the ball–bed pair, not any contact involving the bed.
Reset-state and pipeline defects F3/F4 also apply here.
Section 7 incorrectly says spin is unmeasured on both sides; the script records simulated spin explicitly.

## 4. Overstatement — report sections 3.4, 4.1, 4.3, 5, 6, 7

### F9 — Blocking: the contact-motion metrics have the wrong interval and object

`drop.py:198` takes the first and last contact sample over the entire 1.6 s rollout.
Its 1.49-degree rotation therefore spans multiple bounces, not only the first contact.
The exact reproduction prints **29 total contact steps**. A separate segmentation check finds a **10 ms first contact**.
Do not mix these intervals.

`pos` records the ball centre at `drop.py:116`.
`drift` at line 206 is the ball's three-dimensional displacement, not racket translation or lateral displacement.
The exact command prints **127.2 mm**, not a measured 1 mm racket displacement.
No racket-translation trace is returned by this script.
Record the face transform at consistent pre/post-step times and segment one contiguous ball–bed contact.

### F10 — A resting ball does not independently calibrate the face angle

Section 4.3 argues against an unobstructed freely rolling ball on a particular ideal slope model.
That can be a useful consistency check. It does not establish a one-degree hardware angle bound.
String-bed shape, local indentation, rolling resistance, rim restraint, initial velocity, and support motion need exclusion.
The robot log does not track the racket independently.

Do not declare the chain geometry wrong, the real bed level, or the height offset explained from this observation alone.
Keep both mount geometry and chain accuracy unresolved pending independent geometry/face measurements.
“No geometry change yet” remains a defensible decision under this uncertainty.

### F11 — The proposed height test cannot identify spin as the cause by elimination

Section 6 says a flat first-bounce curve would prove that the decay gap comes from spin.
It would not. Secondary bounces also differ in impact point, arm recovery state, support motion, and approach direction.
A low-height mismatch also does not identify joint friction rather than other candidate mechanisms.
Use the proposed height series to test speed dependence, not to declare a unique cause.
Keep physical measurement authorization separate from this review.

## 5. Design section 5(B)

### F12 — The report treats a conditional candidate as the next authorized implementation

The existing design already puts matched drops first, joint-model calibration second, and passive modes behind an evidence gate.
It explicitly says that a constant-target/fixed-base fixture is diagnostic unless equivalence is established.
It does not authorize building section 5(B) next.

**Keep the ranking and the gate.** The one normal translation is a minimum-dimensional candidate, not a measured mechanism.
Do not lower or remove it on the basis of the invalid 0.06 bound.
The report supplies useful preliminary evidence about existing arm recoil, not an objection that reverses the design.
No design physics change is warranted by this review.

## Verification and provenance

Passed: exact CPU reproduction `uv run python scripts/experiments/x2_tennis_drop/drop.py 0.9`.
It reproduces first-bounce ratios 0.519 and 0.582 and the reported secondary-bounce values.
Passed: compiled-option comparison, arithmetic checks, and a two-case CPU event/boundary diagnostic.
Failed: `ruff` undefined-name check for `tangential.py`.
Not tested: corrected tangential experiment, full training-backend parity, hardware paired rows, or physical transfer.

Reviewed SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `model.py` | `d9e1af65ee4a62305b1dbc4852459ae722036241be5cb3bd23105daaf49fa78c` |
| `drop.py` | `0a5c3d170497222dc63ce6d10af36ccf553d8112e1dc2d5bb9fcc29e937e1869` |
| `tangential.py` | `4c717f183a3898629741702a195d146e744c1f6580f48b4f81b127d5a742fb6d` |
| reviewed report | `fecd993974f12267ab4bcd40ae46a95ac0cd63b67a80a670af9879a49c68b9ff` |

## Revision re-check — 2026-11-11 22:13 (+08)

Scope: F2, F3, F5, F6, F9, and the revised verdict in sections 4 and 6.
The report and scripts remain unchanged by this reviewer.

**Verdict: keep “no model change yet.” Do not approve the quantitative hardware comparison or the sole-cause claim.**

### F2 — option application fixed; causal attribution still unsupported

The revised builder applies `cfg.sim.mujoco`. The option-application defect is closed.
The exact four-height command reproduces the new table:

- Shipped wrist: **0.557 / 0.560 / 0.555 / 0.549**.
- High-gain intervention: **0.640 / 0.647 / 0.646 / 0.644**.

However, the old-to-new **0.038** difference also includes changes to release height,
settling, ball parking, surface reference, and the estimator.
It cannot be assigned to solver options alone.
A controlled diagnostic kept the revised protocol and changed only the option selection:

| Revised protocol at 0.9 m | Shipped wrist | High-gain wrist |
| --- | ---: | ---: |
| Task options | 0.557 | 0.640 |
| Prior options: Euler, 100/50/35 iterations | 0.547 | 0.622 |

This is about **0.010**, not 0.038, for the shipped wrist in that diagnostic.
The reference-height calculation follows the selected options in each case.
Use a fully frozen initial state for a strict one-factor physical sensitivity estimate.

### F3 — boundary defect fixed; hardware-estimator equivalence remains open

Using a known release height removes dependence on the artificial initial maximum.
That is a valid separately declared simulation estimator.
But `first_bounce()` takes `z_min` from the **raw 1 kHz trace** at `drop.py:160`.
Only its apex comes from the sampled, smoothed trace.
The hardware analyser uses sampled, smoothed minima as well.
The first-bounce routine also omits the advertised band and minimum-drop acceptance checks.

For the reproduced 0.9 m case, with the same release height and apex:

- Raw minimum: **1.139823392 m**; `e_h = 0.5568293`.
- Sampled/smoothed minimum: **1.156778652 m**; `e_h = 0.5447914`.

This is a **16.96 mm** minimum difference and **0.0120** in the ratio.
It is not a correction to the robot data or a full bias bound.
Retain separate truth and measurement-pipeline metrics. Compare hardware only with a matched estimator.

### F5 — physical height reference and paired rows remain unresolved

`z_surface = min(z_slow) - radius` uses a transient 20 mm drop, not a resting surface measurement.
The reproduced proxy is **1.106210020 m**; the unloaded face site is **1.121087007 m**.
The proxy sits **14.877 mm below** that face.
It includes impact penetration and arm motion, so it cannot silently be called the physical bed surface.

The same first apex has three distinct heights:

| Reference/convention | Height |
| --- | ---: |
| Ball centre above the slow-drop proxy | 0.312383 m |
| Ball centre above the pre-drop face | 0.297506 m |
| Ball bottom above the pre-drop face | 0.264806 m |

The hardware **0.24 m** still lacks pinned paired rows and an identical convention.
Do not certify a 7 cm hardware gap or 55% greater returned energy yet.
`(0.557/0.45)^2 = 1.5321` is an arithmetic ratio of apparent retention estimates,
not an established absolute energy comparison between matched trials.

Section 3.2 also still mixes rows. The exact four-height output puts **0.089 / 0.601**
after the **0.30 m** release. It does not print that pair as a third bounce of the 0.90 m release.
The 0.90 m case prints the accepted next bounce **0.264 / 0.494** only.
Pin per-release and per-bounce IDs before constructing the decay table.

Related correction: `residual` is calculated **after the 0.4 s diagnostic slow drop**, not after settling.
The reproduced maximum error is **4.83e-10 rad immediately after the initial settle**,
and **0.01201245 rad after the slow drop**.
The latter does not show that task solver options cause a 0.012 rad equilibrium error.

### F6 — upper-bound defect closed; narrow the remaining mechanism language

The invalid upper bound is removed. The reproduced gain sensitivities are correctly scoped to this fixture.
Keep that change. Prefer “the gain intervention changes vertical rebound” over a partition of the ball's energy loss.
A separate mount mode is not identified or bounded by this intervention.

### F9 — event-span defect closed

The script now isolates the first contiguous ball–bed contact.
The exact run reproduces **9–10 contact steps at 1 ms** and **0.54–0.81 degrees** for the shipped wrist.
The false racket-translation claim is gone.
Keep pre/post-step timing conventions explicit for future impulse or precise motion comparisons.

### Revised decisions — report sections 4.1, 4.2, 6

- **4.1:** retaining the contact fit as the control is supportable. The isolated fit does not prove every held-contact condition correct.
- **4.2:** no new compliance element on this evidence is supportable. The residual mismatch is not yet a calibrated physical parameter target.
- **6.1:** recorded-target replay is a useful next diagnostic. Delete “Then the contact model is the only difference left.”
  Replay matches the command sequence, not the closed-loop policy's response to changed simulated states.
  Support motion, actuator response, tooling geometry, release/spin, and measurement differences remain.
  Joint-trace cross-correlation can absorb physical response lag; it is not independent proof of clock alignment.
- **6.3:** the evidence does not rank joint friction ahead of stiffness. Choose the next mechanism from response measurements and sensitivity tests.

Suggested replacement verdict:

> The revised fixture produces a lower rebound than the isolated bed. The reported hardware comparison is not yet protocol-matched.
> Keep the contact and arm models unchanged. Validate the measurement definitions and paired rows, then replay recorded commands.
> Use the remaining trajectory and recoil errors to select the next identification test. A sole contact cause is not established.

The claim that all twelve findings are closed is also premature: the tangential runner still resets only the ball,
and section 7 still says spin is unmeasured on both sides despite recorded simulated spin.
Section 3.5 still says tilt does not explain the gap without an independent face-angle measurement.
These were observed during the targeted re-check; the tangential numerical table was not rerun.

Validation: four-height CPU command passes; `ruff --select F821,F401` passes;
controlled option comparison and reference/minimum/state-timing diagnostics complete.
No physical transfer or training-backend validation.

Revised SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `model.py` | `694426418e3121f36b3374b48714b78cc057545c420b2eb4600caeecbf4b1345` |
| `drop.py` | `567db9367fe93217dc5699afd19aa5f621e8e59d9830f6263c99dace48b85c92` |
| `tangential.py` | `75b781b934f762702c0e8ef762609ae12dc5bb824655ae8e24fdc29cb1da1681` |
| revised report | `24a58599f33e91a8e539dc2571e0b09bcbd9e07851f7005c4255531124633597` |

