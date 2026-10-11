# X2 actuator response and robustness across robots

Date: 2026-10-09 (+08).
Status: design note; proposed work, not an implementation or qualification record.

This note records the discussion after the operator compares four X2 variational autoencoder (VAE) policies on hardware.
The objective is one policy per motion or shared motion set, with low-cost adaptation across robots.
Full policy training for each physical robot is not the preferred cost model.

## 1. Method

This note combines the operator report, earlier waist-response measurements, and a read-only source inspection.
The source snapshot is `mjlab` commit `487189b2b`.
The inspected actuator files have no local changes at this observation.

Terms:

- **Proportional-derivative (PD) control:** joint control with position stiffness `Kp` and velocity damping `Kd`.
- **Domain randomization (DR):** simulation training across a distribution of dynamics or observations.
- **Actuator response:** the relation between an issued command and joint effort, position, and velocity over time.
- **Difficult-state coverage:** extra teacher labels from states near or after student failures.
- **Actuator coverage:** training and evaluation across plausible actuator responses.
- **Rapid Motor Adaptation (RMA):** a policy with a dynamics context that a history-based estimator supplies at deployment.

The proposed methods below describe design choices.
Their effectiveness for this X2 pipeline remains untested.

## 2. Integrity

The operator report provides a qualitative hardware comparison.
Trial counts, tested members, initial states, and matched operating conditions remain unknown.
The comparison changes both deployed gains and learned teacher/student policies.
It does not isolate gain causality or measure a coverage effect size.

Source configuration does not prove the settings of every historical run.
Use the saved configuration and provenance for each run to establish its actual settings.
`MJLAB_DR_AXES` can remove configured axes before task construction.

This document changes no actuator, policy, controller, or training configuration.
No new simulation, training, deployment, or hardware operation accompanies this note.

The prose follows controlled-English rules.
An approved-word dictionary is unavailable; this note does not claim ASD-STE100 certification.

## 3. Results available at this date

### 3.1 Operator hardware comparison

The operator reports this order:

| Policy | Reported hardware behavior |
| --- | --- |
| `vgk-8t-8192-cov-nc` | Best observed behavior |
| `vgk-8t-8192-nocov-nc` | Slightly worse; one reported fall |
| VAE policies from `incXXX` teachers | Worse; many reported falls |

The report does not establish a fall rate because trial counts remain unknown.
It supports a provisional hypothesis: tuned vendor gains provide much of the improvement, and difficult-state coverage adds robustness.
The relative contribution of each remains unmeasured.

The VGK bundles use vendor gains with a deliberate waist adjustment:

| Joint | Kp | Kd |
| --- | ---: | ---: |
| Waist yaw | 40.0 | 4.0 |
| Waist pitch | 200.0 | 4.0 |
| Waist roll | 200.0 | 4.0 |

These are selected nominal gains, not proven optimal gains for all robots.

### 3.2 Earlier waist-response evidence

The earlier waist-kd4 comparison reports these changes against the original `vendor-dr-001` trial:

| Metric | Change |
| --- | ---: |
| Measured waist-yaw position amplitude, 20–30 Hz | −64.8% |
| Measured waist-pitch position amplitude, 20–30 Hz | −45.6% |
| Measured waist-roll position amplitude, 20–30 Hz | −57.7% |
| Waist-group position root-mean-square error (RMSE) | +3.0% |

The trials have different initial states and timing.
The evidence supports reduced vibration in those trials.
It does not prove improved overall position tracking.
It shows why damping and response timing need separate metrics from average position error.

Source: [waist comparison](/home/agiuser/data/20261008T124731Z_tune-kd-001/analysis/comparison_tuned_vs_original.md).

### 3.3 Current actuator randomization

The vendor-gain task inherits the correlated-DR configuration.
These settings describe source defaults with the relevant axes enabled:

| Property | Configured variation | Sampling |
| --- | --- | --- |
| Servo Kp/Kd | 0.7–1.3 × nominal | Per reset; Kp and Kd share one scale per environment and target |
| Armature | 0.95–1.05 × nominal | Startup |
| Effort limits | 0.95–1.0 × nominal | Startup; X2 also has a wrist-specific term |
| Dry joint friction | 0.95–1.05 × nominal | Startup |
| Passive joint damping | Absolute ranges per X2 joint group | Startup |
| Command delay | 0–2 physics steps; 0–10 ms at 5 ms per step | Actuator delay buffer; hold probability 0.9 |
| Encoder observation delay | 0–1 control steps; 0–20 ms at 50 Hz | Joint position and velocity share an encoder packet delay |

`shared_gain_scale=True` couples Kp and Kd for each target.
It does not impose one common gain scale across the whole robot.
This distinction matters when robot-wide effects correlate responses across joints.

The no-PD-gain-DR task removes only `randomize_pd_gains`.
It retains passive damping and command delay.

Sources:

- [Tracking DR settings](../../src/mjlab/tasks/tracking/tracking_env_cfg.py).
- [X2 task inheritance and correlated DR](../../src/mjlab/tasks/tracking/config/agibot_x2/env_cfgs.py).
- [X2 actuator delays](../../src/mjlab/asset_zoo/robots/agibot_x2/x2_constants.py).
- [PD gain sampling](../../src/mjlab/envs/mdp/dr/actuator.py).
- [Tuned vendor gains](../../src/mjlab/tasks/tracking/config/agibot_x2/vendor_gains.py).

These ranges are configured bounds, not measured bounds across the robot fleet.

## 4. Findings and candidate methods

### Finding 1: a distribution of actuator models can preserve robustness

A model fit for one nominal robot can cause overfitting.
A model fit across robots and operating conditions can improve the relevance of DR.
Measurements can constrain that distribution and retain uncertainty.
Data from one robot cannot establish fleet-wide bounds.
Use provisional margins and validate on robots outside the fit data.

The actuator and learned policy form two feedback loops.
The joint loop converts a target into effort.
The policy loop uses robot state to choose the next target.
A policy learns the joint response in its training environments.
Response differences can change foot placement and support timing on hardware.

The hypothesis concerns response mismatch, not an established defect in the motor.
Delay, friction, compliance, effort limits, and load can all contribute.

### Finding 2: response-based DR covers more than gain scaling

Gain DR changes stiffness and damping.
It does not fully reproduce torque-response lag, backlash, or speed-dependent torque limits.
Existing command delay covers target transport, not every source of actuator bandwidth loss.

A simple local joint approximation illustrates the distinction:

```text
J * q_ddot + Kd * q_dot + Kp * q = Kp * q_target
omega_n = sqrt(Kp / J)
zeta = Kd / (2 * sqrt(Kp * J))
```

`J` is effective inertia, `omega_n` is natural frequency, and `zeta` is damping ratio.
This approximation excludes delay, saturation, coupled motion, contact, and nonlinear transmission effects.
It is not an identified X2 model.
Even equal scale factors for Kp and Kd do not keep damping ratio constant.

Candidate response properties include:

- Command delay and its variation over time.
- Torque-response time constant.
- Rise time, settling time, and overshoot under load.
- Friction and available effort across joint speeds.
- Mechanical compliance where measurements require it.

Fit simple models first.
Add complexity only when simpler models fail on held-out response data.
Sample correlated physical parameters that produce plausible responses.
Do not independently sample response metrics that the model cannot realize together.

Use three levels of variation:

| Level | Examples | Proposed persistence |
| --- | --- | --- |
| Robot-wide | Common drive or firmware response effects | Fixed within an episode |
| Joint-specific | Friction, effective gain, joint effort limit | Fixed within an episode |
| Time-dependent | Communication jitter, gradual response drift | Bounded changes during an episode |

The distribution must retain nominal cases as well as difficult cases.
Wider ranges alone do not guarantee better transfer.
Unrealistic combinations can reduce useful performance and increase training cost.

### Finding 3: action and state history can support implicit adaptation

A history encoder or recurrent policy can infer response differences from recent commands and motion.
This can support one policy across robots without explicit motor-parameter identification.

Candidate inputs include recent joint positions, velocities, issued targets, and sample intervals.
A single observation can confuse a recent target change with a slow actuator response.
History reduces this ambiguity but does not guarantee identification.

Training must include relevant variation and sufficient response information.
The policy must handle the start of an engagement before enough history exists.
History reset, missing samples, and stale samples need defined behavior.

The student must preserve the adaptation information of the teacher.
A teacher with history cannot always provide reproducible labels to a student with only the current state.
History support changes the current observation and export contracts.
This method requires coordinated training, distillation, and controller changes.

### Finding 4: an actuator-context estimator can avoid policy retraining per robot

An RMA-style design separates response estimation from motion control.
During simulation training, a policy receives a compact dynamics context.
An estimator learns to infer that context from action and observation history.
At deployment, the estimator supplies context without simulator parameters.

Keep actuator context separate from the VAE motion latent:

- Motion latent: the movement to execute.
- Actuator context: how this robot responds.

A candidate estimator can use training across many simulated robots.
Deployment can update its context without changes to the motion policy weights.
A short calibration can provide an initial context.

The method does not require exact recovery of every physical parameter.
It requires context that helps control on held-out dynamics.
Training cost, inference cost, and adaptation speed remain unmeasured for X2.
Reuse across motions is a design objective, not a validated result.

### Finding 5: joint-level calibration can reduce the variation seen by the policy

A short calibration can select gains or bounded compensation for a physical robot.
The target is a common acceptable joint response across robots.
Calibration adjusts a small controller parameter set rather than every motion teacher.

This method complements DR.
Residual response differences still need coverage during training.
Calibration cannot remove all delay or create unavailable effort.
A joint response measured without load does not establish the response during contact.

A change to deployed gains also changes the dynamics seen by existing policies.
Each proposed calibration range needs policy validation.
No universal optimal gain pair follows from the current evidence.

### Finding 6: a learned actuator model is an optional simulation method

A learned model can map command and state history to effort.
An ensemble or conditioned model can represent response variation instead of one fitted robot.
The repository already contains a [learned actuator implementation](../../src/mjlab/actuator/learned_actuator.py).
Its existence does not establish X2 integration or response accuracy.

This method needs reliable data and validation outside the training data.
Reported joint effort may differ from actual applied torque.
A learned model can also produce wrong outputs outside its data range.
Use this method only if simpler actuator models leave material response errors.

### Finding 7: difficult-state coverage and actuator coverage address different gaps

Difficult-state coverage adds examples where the student deviates from nominal motion.
Actuator coverage varies the dynamics under which those deviations and corrections occur.
Neither method replaces the other.

Audit the student collection distribution as well as teacher training.
A robust teacher does not automatically make the distilled student robust.
Collection must preserve randomized dynamics, delay history, and the information required for teacher labels.
See the [coverage contract](beyondmimic_vae_difficult_state_coverage_p0.md).

Evaluate fixed actuator models separately from aggregate randomized performance.
Average success can hide failures at the edges of the distribution.
A curriculum or objective that emphasizes difficult plausible models is a candidate after the first evaluation.

## 5. Established position

The preferred cost model is one broadly trained policy with inexpensive calibration or adaptation per robot.
The reported full teacher training cost is approximately 16 hours per motion.
This estimate is workload-specific; it is not a measured cost for the proposed methods.

The first recommendation is to improve the actuator distribution around the tuned nominal gains.
History and explicit adaptation remain follow-up options if a fixed policy cannot cover the measured variation.
A learned actuator model remains a later option when simple models fail.

This is a recommendation for future work.
The operator has not selected an implementation or authorized new experiments through this document.

## 6. Open items and proposed sequence

| Step | Work | Reviewable result |
| --- | --- | --- |
| 1 | Audit existing bags and saved configurations | Known response differences, timing limits, and actual training settings |
| 2 | Specify measurements across robots and loads | Response metrics, sample conditions, and data provenance |
| 3 | Fit a simple model distribution | Versioned parameters, correlations, uncertainty, and held-out prediction errors |
| 4 | Evaluate existing teachers and students on fixed model variants | Per-motion success, failure onset, and response metrics |
| 5 | Compare the current DR distribution with one proposed change | Matched seeds, motions, budgets, and nominal/difficult model results |
| 6 | Audit or modify distillation coverage | Evidence that actuator robustness survives student training |
| 7 | Add history or explicit context if needed | Benefit against the fixed-policy baseline and measured runtime cost |
| 8 | Validate transfer on other physical robots | Matched hardware trials outside the calibration data |

Response analysis must distinguish issued-command time from actuator receipt time.
The existing hardware interface does not echo the received target.
Sensor delay and external load can confound inferred actuator delay.
Recorded policy commands alone do not establish an open-loop causal response.

For every comparison, record:

- Policy, motion, member, controller, and configuration identities.
- Robot identity, firmware, load, and available operating-condition measurements.
- Trial count, motion phase, initial state, and engagement mode.
- Joint error, response lag, vibration, and effort-limit activity where observable.
- Fall count and exposure, with failures during motion separate from failures after the endpoint.
- Nominal performance and performance on fixed difficult models.

Split validation by robot or session when possible.
Random sample splits from one recorded session do not test transfer to another robot.
Set acceptance thresholds after measurement, before the comparison.
Do not invent fleet bounds or promise a reduction in policy training time before evidence exists.

Next action: specify the offline audit and measurement plan.
New training, controller changes, calibration, and hardware operations require a separate scoped request.

Related reference: [existing sim-to-real DR documentation](../source/sim2real_domain_randomization.md).
Some historical explanations in that reference are approximate; the inspected source and saved run configurations determine actual sampling behavior.
