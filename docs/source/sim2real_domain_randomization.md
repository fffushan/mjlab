# mjlab Randomizable Parameters and Sim-to-Real Robustness

Every parameter below is randomized through mjlab's domain-randomization (DR)
module (`src/mjlab/envs/mdp/dr/`) plus actuator inline delay fields. DR
functions are registered as event terms (`func=dr.<name>` in an
`EventTermCfg`) with mode `"startup"` (once per env) or `"reset"` (every
episode).

Common knobs for every `dr.*` function:

- `distribution`: `"uniform"`, `"log_uniform"`, `"gaussian"`
- `operation`: `"scale"` (multiply compile-time defaults, no accumulation),
  `"abs"` (set absolute values), `"add"`
- `ranges`: `(min, max)` per axis; entity-level functions like `dr.pd_gains`
  take named ranges (`kp_range`, `kd_range`, ...)

---

## Actuator / actuation channel

| Function / config | What it randomizes | Sim-to-real robustness effect |
|---|---|---|
| `dr.pd_gains` (`kp_range`, `kd_range`) | PD proportional gain **kp** (`stiffness`) and derivative gain **kd** (`damping`) of the joint servo | Policy doesn't overfit to one closed-loop stiffness/damping. Covers real actuators whose effective gain/bandwidth differs from nominal (firmware differences, voltage droop, per-unit variation, load-dependent bandwidth). **Main knob for "wider actuator quality".** |
| `dr.effort_limits` (`effort_limit_range`) | Actuator force/torque range (`actuator_forcerange`, `jnt_actfrcrange`) | Policy learns to stay within torque budgets; robust to weaker/stronger motors, current limits, thermal derating. Pairs with `pd_gains` to prevent over-reliance on raw torque. |
| Actuator config `delay_min_lag` / `delay_max_lag` (+ `delay_hold_prob`, `delay_update_period`) | Command-channel latency: policy targets arrive 0–N physics steps late at the control law (per-env, resampled over time) | Robust to bus/communication latency, firmware loop jitter, low-bandwidth motor controllers. Distinct from observation delay (sensor pipeline). |
| `dr.joint_damping` / `dof_damping` | Passive velocity-proportional joint damping | Robust to viscous friction variation (oil temperature, bearing condition) in the transmission. |
| `dr.joint_friction` / `dof_frictionloss` | Dry friction loss in the joint | Robust to stiction/breakaway torque differences, seal friction, wear. |
| `dr.joint_armature` / `dof_armature` | Added rotor inertia (models geared transmissions) | Robust to different gear ratios/rotor masses; changes how the actuator responds to torque commands. Triggers `set_const_0` recomputation. |
| `dr.joint_stiffness` / `jnt_stiffness` | Passive spring stiffness pulling toward reference position (mechanical compliance) | Robust to gearbox elasticity/backlash behavior. **Not** the PD kp — a separate physical compliance knob. |
| `dr.joint_limits` / `jnt_range` | Lower/upper joint position limits | Robust to mechanical hard-stops at slightly different angles across units. |
| `dr.joint_default_pos` / `qpos0` | Reference/zero-spring equilibrium position | Robust to assembly differences in rest pose and offset in position sensors. |
| `dr.tendon_damping`, `dr.tendon_stiffness`, `dr.tendon_friction`, `dr.tendon_armature` | Tendon counterparts of the joint fields (tendon-driven robots) | Same robustness effects for cable-driven/tensegrity systems. |

## Body / mass properties

| Function | What it randomizes | Sim-to-real robustness effect |
|---|---|---|
| `dr.body_mass` | Body mass | Robust to payload differences, machining tolerances, added equipment. Triggers `set_const` recomputation. |
| `dr.body_com_offset` / `body_ipos` | COM position in body frame | Robust to uneven component placement / center-of-gravity shifts; avoids policies that rely on exact balance points. |
| `dr.body_pos`, `dr.body_quat` | Body frame pose in parent frame | Robust to link geometry/assembly tolerances. |
| `dr.pseudo_inertia` | Physically consistent joint randomization of mass, COM, and inertia tensor (Rucker & Wensing 2022) | Like `body_mass`/`body_com_offset` but guaranteed to produce valid positive-definite inertia — safer for large perturbation ranges. |

## Contact / terrain interaction

| Function | What it randomizes | Sim-to-real robustness effect |
|---|---|---|
| `dr.geom_friction` | Sliding/torsional/rolling friction coefficients (`geom_friction`) | Robust to floor/shoe/rubber material variation, dust/wet surfaces. One of the most impactful knobs for legged locomotion sim2real. |
| `dr.pair_friction` | Per-pair friction override (`pair_friction`) | Same, but scoped to specific contact pairs; can independently tune feet-vs-floor vs body-vs-object. |
| `dr.geom_pos`, `dr.geom_quat` | Geom pose in parent body | Robust to foot placement/gear alignment tolerances on the real robot. |
| `dr.geom_size` | Geom size parameters (radius, half-lengths) | Robust to manufacturing tolerances of links/feet; recomputes bounds (`geom_rbound`, `geom_aabb`). |

## Sensing / perception (visual sim2real)

| Function | What it randomizes | Sim-to-real robustness effect |
|---|---|---|
| `dr.encoder_bias` | Fixed per-joint bias added to position readings (`entity.data.encoder_bias`) | Robust to encoder calibration errors and zero-offsets on real joints. |
| `dr.cam_fovy`, `dr.cam_pos`, `dr.cam_quat`, `dr.cam_intrinsic` | Camera field of view, pose, intrinsics `[fx, fy, cx, cy]` | Robust to camera mounting variation and calibration error; essential for vision policies. |
| `dr.light_pos`, `dr.light_dir`, `dr.light_diffuse`, `dr.light_specular`, `dr.light_ambient`, `dr.light_attenuation`, `dr.light_cutoff`, `dr.light_exponent` | Lighting environment | Robust to real-world illumination differences (indoor/outdoor, shadows, time of day) for RGB-based policies. |
| `dr.mat_rgba`, `dr.mat_emission`, `dr.mat_specular`, `dr.mat_shininess`, `dr.mat_texrepeat`, `dr.mat_texid` | Material color/appearance/texture | Robust to surface appearance variation; mostly for RGB rendering sim2real. |
| `dr.geom_rgba` | Geom color and transparency | Appearance variation; useful for domain-randomized visual identity. |

---

## Recommendations

- **Locomotion / manipulation core:** `dr.pd_gains` + `dr.effort_limits` +
  `dr.geom_friction` + `dr.joint_friction`/`dof_damping` cover the biggest
  sim2real gaps (gains, torque, friction).
- **Add command delay** (`delay_min_lag`/`delay_max_lag`) on actuators to model
  real bus latency — often as impactful as gain randomization.
- **Vision policies:** add `dr.cam_*`, `dr.light_*`, `dr.mat_*`, `dr.geom_rgba`.
- **Payload / hardware variants:** `dr.body_mass`, `dr.body_com_offset`,
  `dr.pseudo_inertia`.
- Use `"scale"` (default) so repeated per-episode randomization doesn't
  accumulate; ranges stay relative to compile-time defaults.

Source: `src/mjlab/envs/mdp/dr/`, `docs/source/randomization.rst`,
`src/mjlab/actuator/actuator.py` (delay fields).

---

## G1 motor Kp/Kd bandwidth analysis (current gains)

Computed from the G1 model at the knees-bent keyframe: mass-matrix diagonal
(link inertia) + reflected rotor armature, with the gains from
`src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py`:

- Design intent: natural frequency `NATURAL_FREQ = 10 Hz` (62.83 rad/s) at the
  rotor armature, damping ratio `DAMPING_RATIO = 2.0`.
- Loop model per joint: `J·q'' + kd·q' + kp·q = kp·q_cmd` with
  `J = J_link + armature`,
  `wn = sqrt(kp/J)`, `zeta = kd/(2·sqrt(kp·J))`, and -3 dB bandwidth
  `wbw = wn·sqrt(1-2ζ²+sqrt((1-2ζ²)²+1))`.

| Joint group | Joints | kp (N·m/rad) | kd (N·m·s/rad) | ωn (Hz) | ζ | −3 dB bw (Hz) |
|---|---|---|---|---|---|---|
| 7520-22 | knee | 99.1 | 6.31 | 4.3 | 0.86 | 3.4 |
| 7520-22 | hip roll | 99.1 | 6.31 | 1.9 | 0.39 | 2.7 |
| 7520-14 | hip pitch | 40.2 | 2.56 | 1.1 | 0.22 | 1.6 |
| 7520-14 | hip yaw / waist yaw | 40.2 | 2.56 | 2.7 / 1.9 | 0.54 / 0.37 | 3.3 / 2.6 |
| 5020 | shoulder pitch / roll / elbow | 14.3 | 0.91 | 1.4–3.1 | 0.28–0.68 | 2.1–3.5 |
| 4010 | wrist | 16.8 | 1.07 | 6.9–8.4 | 1.4–1.7 | 2.7–2.9 |
| 2×5020 | ankle / waist pitch | 28.5 | 1.81 | 8.5–9.7 / 1.1 | 1.7–2.0 / 0.23 | 2.7 / 1.7 |

**Result: natural frequency 0.96–9.7 Hz, closed-loop −3 dB bandwidth ≈
1.45–3.5 Hz (mean ≈ 2.7 Hz).** The heavy joints (hip pitch, waist pitch/roll,
shoulder pitch) sit at 1.5–2 Hz; only wrist/ankle approach the 10 Hz design
intent, because the design uses the rotor armature only while link inertia
dominates the big joints (e.g. hip pitch: J_link ≈ 0.85 vs armature ≈ 0.01
kg·m²).

### Headroom for Kp randomization

Context: physics at 200 Hz (dt = 0.005), policy/action rate 50 Hz (decimation
= 4).

- **Numerical stability: lots of headroom.** `BuiltinPositionActuator` uses
  MuJoCo implicit integration, stable far beyond explicit PD divergence. Even
  4× kp leaves the fastest joint (wrist, 9.7 Hz → ~19 Hz ωn) far below the
  200 Hz physics Nyquist (100 Hz).
- **Practical bound is damping, not bandwidth.** Scaling only kp by `s` drops
  ζ by `1/sqrt(s)` (heavy joints are already underdamped at ζ ≈ 0.2–0.4 and
  become oscillatory); scaling kp and kd together keeps ζ within `sqrt(s)`
  (≈ ±6% at s ∈ [0.7, 1.3]).
- **Effort coupling:** kp·error generates torque; the G1 action scale is
  `0.25·effort/stiffness`, so higher kp with fixed action magnitude means
  smaller effective commanded torque.

Recommended ranges for `dr.pd_gains` on G1:

| Range | Effect on servo bandwidth | Notes |
|---|---|---|
| `(0.7, 1.3)` matched kp=kd | ≈ 1.3–3.9 Hz | Safe default, mild sim2real gain mismatch |
| `(0.5, 1.5)` matched | ≈ 1.2–4.3 Hz | Wider actuator-quality spread; heavy joints dip to ζ ≈ 0.15 (oscillatory in sim — policy must learn to cope) |
| kp-only (kd fixed) | ζ drops by √s | Only for *softer* servos (s<1); avoid s>1 alone |

Bandwidth varies with configuration (mass matrix changes as the robot moves),
so values are indicative at the home keyframe.

### Wiring (tracking task)

Added to `src/mjlab/tasks/tracking/tracking_env_cfg.py` events:

```python
"randomize_pd_gains": EventTermCfg(
  mode="reset",
  func=dr.pd_gains,
  params={
    "asset_cfg": SceneEntityCfg("robot"),
    "kp_range": (0.7, 1.3),
    "kd_range": (0.7, 1.3),
    "operation": "scale",
  },
),
```

The wired default is the safe ``(0.7, 1.3)`` matched range (≈ 1.3–3.9 Hz on
G1): heavy joints stay damped enough to avoid excessive oscillation while the
policy still sees a meaningful servo-gain spread. ``mode="reset"`` re-draws
gains every episode; switch to ``"startup"`` for a fixed per-env draw across
the whole run. ``"scale"`` multiplies the compile-time default gains, so
repeated per-episode randomization does not accumulate.

## Wired into the tracking tasks

Every tracking task (G1, G1-29DOF-mode-15, X2, each with and without state
estimation) carries the axes below in
``src/mjlab/tasks/tracking/tracking_env_cfg.py``.

These axes exist **only** in the tracking tasks: the velocity tasks randomize
nothing but the actuator PD gains. The tracking task is the one that has to
reproduce a measured hardware behaviour, so its ranges are capped at 5 % of
nominal — wide enough that a policy cannot over-fit the exact model, narrow
enough that it still trains against roughly the machine it will be deployed on.
`foot_size` (0.97–1.03x) and `obs_delay` (0–1 policy step) are already at or
below that bar, and a discrete lag has no nominal to take a percentage of.

| Axis | Event | Range | Why |
|---|---|---|---|
| `inertia` | `dr.pseudo_inertia` | mass 0.95–1.05x, COM ±1 cm | Link masses and inertias differ between the vendor simulator (mesh-derived, no `<inertial>` on the pelvis: 43.54 kg) and this model (URDF v1.3.0: 41.97 kg), and hardware adds payload. `pseudo_inertia` keeps mass, inertia and COM physically consistent; `dr.body_mass` alone leaves the inertia tensor stale and warns. |
| `armature` | `dr.joint_armature` | 0.95–1.05x | Reflected rotor inertia from the PFP module table. The shipped PD gains used to be derived from it, which is what made randomizing it necessary; that link is gone for the X2, whose gains are measured hardware values, so the armature is now purely the physical reflected inertia. |
| `effort_limits` | `dr.effort_limits` | 0.95–1.0x, weaker only | The vendor simulator derates its motors below the URDF peaks (118 vs 120 N·m on the hip), and hardware derates thermally. Weaker-only is the failure mode that matters. Its larger wrist gap (2.2 vs 4.8 N·m) needs the dedicated term below, not this range. |
| `effort_limits` (wrists) | `dr.effort_limits_wrist` | 0.95–1.0x | A 5 % residual of the derating that used to probe the vendor's wrist clamp (2.2/4.8 = 0.46x). Modelling that clamp now means setting the wrist's *nominal* limit to 2.2 N·m and dropping the term (open item 5.6 of `x2_gain_provenance.md`). Must be applied after the whole-robot term; the X2 config asserts that the actuator index it targets is still the wrist group. |
| `joint_friction` / `joint_damping` | `dr.joint_friction` / `dr.joint_damping` | frictionloss 0.95–1.05x, damping 0–0.03 (absolute, and robot configs should scale it per group) | The shipped models have `damping=0.0`, i.e. no viscous transmission loss at all, and an MJCF-supplied stiction value that may be one blanket number for every joint. See *Absolute ranges are calibrated for a leg joint* below. |
| `foot_size` | `dr.geom_size` | 0.97–1.03x (already inside the 5 % cap) | Foot pad wear and mounting tolerance. Robot configs set the foot sphere pattern. |
| `pd_gains` | `dr.pd_gains` | kp/kd 0.7–1.3x matched, per episode | See the G1 section above. |
| `obs_delay` | actor `joint_pos`, `joint_vel`, `base_ang_vel` terms | 0–1 policy steps (0–20 ms) | Sensor pipeline latency, distinct from the command delay. |

Command delay is not an event: it lives on the actuator group
(``delay_min_lag``/``delay_max_lag``, 0–2 physics steps = 0–10 ms for X2), so a
30 Hz-to-1 kHz deployment bus is modelled where it belongs. The decision period
already holds each action for 0–20 ms, so this is the transport part on top.

### Selecting axes

``MJLAB_DR_AXES`` selects a subset at process start (the config is built at task
registration, so set it before launching):

```sh
# Sweep one axis on a trained checkpoint: does it survive this alone?
MJLAB_DR_AXES=armature uv run python scripts/evaluate_tracking_policy.py \
  --checkpoint-file logs/rsl_rl/agibot_x2_tracking/<run>/model_29999.pt \
  --motion-file data/qianghuo_smplx_agibot_x2_tracking.npz \
  --task Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation --num-envs 64

# The pre-axis behaviour (push, torso COM, encoder bias, foot friction only),
# i.e. the set the currently deployed X2 policy was trained with:
MJLAB_DR_AXES=none uv run python scripts/evaluate_tracking_policy.py ...
```

An unknown axis name raises at config build rather than silently doing nothing.
``tests/test_tracking_dr.py`` locks the wiring in.

## X2 no-state-estimation correlated-DR ablations

The two X2 ablations below are new tasks; the existing
``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation`` task and its trained
checkpoints remain unchanged. They retain the NSE actor/critic term order,
observation dimensions, action schema, rewards, nominal gains and geometry.
They use separate log directories so checkpoints cannot overwrite one another.

### Ablation 2 — correlated DR

``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR`` changes
only these enabled-axis knobs:

- ``pd_gains`` uses ``shared_gain_scale=True`` with the existing matched
  ``kp_range=kd_range=(0.7, 1.3)`` and ``operation="scale"``. Each joint and
  environment still draws independently, but Kp and Kd use the same multiplier.
- ``obs_delay`` makes actor ``joint_pos`` and ``joint_vel`` one
  ``delay_group="encoder_packet"`` (0–1 policy steps), with
  ``delay_hold_prob=0.9`` and ``delay_update_period=1``. ``base_ang_vel`` keeps
  its own delay buffer, also at 0–1 steps with the same hold/update settings.

Train from scratch with a local motion file:

```sh
uv run train Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR \
  --env.commands.motion.motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
```

Its PPO runs are written below ``logs/rsl_rl/agibot_x2_tracking_correlated_dr``.
Local resume flags (``--agent.resume``, ``--agent.load-run``, and
``--agent.load-checkpoint``) search the selected ``experiment_name`` directory,
not an arbitrary checkpoint path. To fine-tune a baseline checkpoint without
copying it, you can override ``--agent.experiment-name agibot_x2_tracking`` and
select its run/checkpoint with those flags. Use a distinct ``--agent.run-name``
for the ablation: this creates a new timestamped run, but deliberately gives up
the default separate experiment directory. Match the baseline motion and
``--env.commands.motion.lookahead-s`` setting when resuming; training does not
automatically restore lookahead from the checkpoint's saved environment config.

### Ablation 3 — reduced perturbations

``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations``
starts from Ablation 2 and differs *only* in reset/push perturbations:

- Every bound in reset ``pose_range``, reset ``velocity_range``, and
  ``joint_position_range`` is multiplied by 0.5.
- Every bound in push ``velocity_range`` is multiplied by 0.5 and its interval
  is ``(4.0, 8.0)`` seconds.
- No curriculum is enabled. The reset and push velocity dictionaries are copied
  independently, so constructing this task cannot change the baseline or the
  other ablation.

```sh
uv run train Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations \
  --env.commands.motion.motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
```

Its PPO runs are written below
``logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations``.

### Additive observation ablations from the reduced-perturbation baseline

The selected ABLATION3 baseline is
``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations``.
Its associated checkpoint is
``logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations/2026-09-21_18-37-07_ablation3``.
The three tasks below are additive fresh-training configurations; the historical
baseline, correlated-DR task, and reduced-perturbation task are unchanged.
All use the same 31-joint robot, motion/reference/action/gains/geometry, DR
(including correlated encoder delay), reset and push settings, rewards,
terminations, episode settings, critic observations, PPO settings, and
observation normalization as ABLATION3.

At ``lookahead_s=0``, the actor contracts are:

| Task suffix | Actor observation order | Dimensions |
|---|---|---:|
| ``-Projected-Gravity`` | ``command, motion_lookahead, projected_gravity, base_ang_vel, joint_pos, joint_vel, actions`` | 161 |
| ``-Projected-Gravity-And-Anchor`` | ``command, motion_lookahead, projected_gravity, motion_anchor_ori_b, base_ang_vel, joint_pos, joint_vel, actions`` | 167 |
| ``-Vendor-Velocity-Scaling`` | Existing 164-dim order; measured ``base_ang_vel`` scale ``0.25`` and ``joint_vel`` scale ``0.05`` | 164 |

``projected_gravity`` is the measured robot root/pelvis gravity vector in the
root body frame, independent of the motion reference and yaw. It uses the
existing ``mdp.projected_gravity`` term and copies ABLATION3's additive
``+/-0.05`` corruption without sharing mutable configuration. The retained
motion-anchor orientation remains unchanged in the anchor variant. The vendor
scales apply after noise to measured gyro and joint velocity only; command,
reference joint velocity, and actions are not scaled.

Train each task from scratch with the same motion artifact, seed, and training
budget as ABLATION3:

```sh
uv run train Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Projected-Gravity \
  --env.commands.motion.motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
uv run train Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Projected-Gravity-And-Anchor \
  --env.commands.motion.motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
uv run train Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Vendor-Velocity-Scaling \
  --env.commands.motion.motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
```

The separate PPO experiment directories are
``agibot_x2_tracking_correlated_dr_reduced_perturbations_projected_gravity``,
``agibot_x2_tracking_correlated_dr_reduced_perturbations_projected_gravity_anchor``,
and ``agibot_x2_tracking_correlated_dr_reduced_perturbations_vendor_velocity_scaling``.
Play uses each task's own checkpoint and the same motion file; for example:

```sh
uv run play Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Projected-Gravity \
  --checkpoint-file logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations_projected_gravity/<run>/model_XXXXX.pt \
  --motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
```

Replace the task and checkpoint directory with either of the other listed
variants as needed. Play disables actor
corruption and pushes, samples from the start, and retains the reduced joint
reset range. Set ``MJLAB_DR_AXES`` to the same ``none`` or subset value for
comparable runs; disabled axes are not recreated by these variants.

All three actor and critic models retain ``obs_normalization=True``. Therefore
Vendor-Velocity-Scaling does not test disabling normalization: normalization
may largely cancel a constant preprocessing scale, so its measured outcome may
be small. The existing X2 Docker adapter is strict; changed dimensions, term
names, or scales require future controller adaptation, and a matching 164
shape alone does not make the scaled export plug-compatible. No controller
support is included here. These are new-training configs, not checkpoint
conversions or automatically fine-tuned runs; do not blindly resume ABLATION3
because changed dimensions or normalizer statistics can invalidate that run.


```sh
uv run play Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR \
  --checkpoint-file logs/rsl_rl/agibot_x2_tracking_correlated_dr/<run>/model_XXXXX.pt \
  --motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
```

For a fair checkpoint comparison, evaluate *all* checkpoints with the same
``--task``, ``MJLAB_DR_AXES`` distribution, iteration count, motion file, and
seed/number of environments. For example, using the correlated task for every
checkpoint avoids the confound of evaluating the reduced-perturbation policy on
its own easier reset distribution. The local evaluator below starts at frame
zero and disables pushes for every task, while keeping observation corruption,
delays and the remaining task DR; it therefore does not test push recovery:

```sh
MJLAB_DR_AXES=all uv run python scripts/evaluate_tracking_policy.py \
  --task Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR \
  --checkpoint-file <checkpoint> \
  --motion-file data/qianghuo_smplx_agibot_x2_tracking.npz \
  --num-envs 64
```

Play mode removes actor noise and pushes, uses start-frame sampling, and clears
pose/velocity reset ranges; it is not an all-DR-off evaluation. Likewise,
``MJLAB_DR_AXES=none`` removes only the listed tracking axes, leaving the
pre-existing push, torso-COM, encoder-bias, and foot-friction terms enabled.

### Anchor ablation — pelvis anchor

``Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Pelvis-Anchor``
is the reduced-perturbation baseline with ``commands.motion.anchor-body-name``
changed from ``torso_link`` to ``pelvis``. Nothing else changes: same robot,
motion, actions, rewards, terminations, DR (including the correlated encoder
delay), reset and push bounds, episode settings, PPO settings, critic terms and
observation dimensions. It is a fresh-training config with its own PPO
directory, not a checkpoint conversion.

The anchor is the body frame the reference is tracked in, and it is *not* the
simulated root. The X2 root link is the pelvis: ``x2_ultra.xml`` declares the
``floating_base_joint`` free joint on the ``pelvis`` body, and the gyro and
velocimeter behind the ``base_ang_vel``/``base_lin_vel`` observations sit on
site ``imu_0``, also on the pelvis. ``motion_global_root_pos`` and
``motion_global_root_ori`` are named after the *anchor*, not the root: they are
implemented by ``mdp.motion_global_anchor_*``. So ``torso_link`` is the only
thing in the shipped task that is torso-based, and this ablation is the one
switch that moves it.

Moving the anchor changes exactly these quantities:

| Quantity | ``torso_link`` anchor | ``pelvis`` anchor |
|---|---|---|
| ``motion_anchor_ori_b`` (actor) | reference torso in robot torso | reference pelvis in robot pelvis |
| ``motion_anchor_pos_b`` (critic; absent in NSE) | torso offset | pelvis offset |
| ``motion_relative_body_*`` rewards | bodies re-anchored on the torso | bodies re-anchored on the pelvis |
| ``motion_global_root_*`` rewards | torso position/orientation error | base position/orientation error |
| ``anchor_pos`` / ``anchor_ori`` terminations | torso height and tilt gates | base height and tilt gates |
| ``error_anchor_*`` metrics, ghost visualization | torso | pelvis |

Two consequences are worth expecting before comparing curves:

- The reference's waist motion is no longer divided out of the relative-body
targets, because the pelvis is upstream of the waist joints. The robot has to
reproduce waist yaw/pitch/roll through the tracked joints instead of having the
anchor absorb it. The body terms can therefore score lower at equal competence,
and the ``anchor_ori`` termination no longer reacts to waist bend at all.
- The ``anchor_pos`` threshold is unchanged (0.25 m) but now gates the base
height (reference ≈0.65 m) rather than the torso height (≈0.80 m). Both are
per-frame world-z gates, so this changes what is measured, not the tolerance.

The trained anchor and the deployment contract are independent in one
direction: the controller is stricter than training. ``policy_loader.cc``
rejects any ``anchor_body_name`` other than ``torso_link``, and
``HardwareRuntime`` builds the live anchor from the pelvis IMU plus measured
waist joints (``ComputeX2TorsoOrientation``). A pelvis-anchor export keeps the
same 164 actor dimensions but puts different numbers in the six anchor slots,
so matching shapes are not plug-compatibility. Deploying this variant would
require relaxing that check and feeding the pelvis orientation through
directly; no controller support is included here.

Train from scratch with the same motion artifact and budget as ABLATION3:

```sh
uv run train Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Pelvis-Anchor \
  --env.commands.motion.motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
```

Its PPO runs are written below
``logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations_pelvis_anchor``.
Play needs the task's own checkpoint:

```sh
uv run play Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Correlated-DR-Reduced-Perturbations-Pelvis-Anchor \
  --checkpoint-file logs/rsl_rl/agibot_x2_tracking_correlated_dr_reduced_perturbations_pelvis_anchor/<run>/model_XXXXX.pt \
  --motion-file data/qianghuo_smplx_agibot_x2_tracking.npz
```

Compare it against the reduced-perturbation baseline at the same seed,
iteration budget, motion file and ``MJLAB_DR_AXES`` setting. Training reward is
not the deciding metric for an anchor change either: as with the observation
ablations above, the reward terms that the anchor weights most heavily
(``motion_global_root_ori`` and the body velocity terms) are not the ones that
track hardware survival.

### Ordering rules

Every ``dr.*`` function samples from the *compiled* defaults, never from the
current value, so ``operation="scale"`` cannot accumulate — but two events
writing the same field mean *the last writer wins*. Two pairs depend on it:

- ``randomize_inertia`` before ``base_com``, so the torso keeps its own wider
  payload COM offset and the limbs keep the ±2 cm one.
- ``randomize_effort_limits_wrist`` after ``randomize_effort_limits``.

### Absolute ranges are calibrated for a leg joint

Some ranges are absolute values in the model's units, not scales, and the
robot-agnostic default was chosen for a 120 N.m joint. A robot with a much
smaller joint group has to replace them, not inherit them:

- ``dr.joint_damping`` is an absolute ``(0.0, 0.03) N.m.s/rad``, a value chosen
  for a 120 N.m leg joint with no meaning for a 0.6 N.m head joint, so the X2
  tracking config replaces the ranges with ``effort_limit / 4000`` per group
  (``X2_DAMPING_RANGES``), which reproduces the generic 0.03 for the legs. Note
  what the 5 % cap did here: with a nominal passive damping of exactly zero there
  is no value to take a percentage of, so this axis is now all but inert (0.03
  against a derivative gain of 4 on an X2 hip) rather than merely narrower.
- ``dof_frictionloss`` comes from the MJCF, and a vendored model may carry one
  blanket value for every joint. The X2's does: 0.3 N.m for all 31 joints, which
  is 50% of ``head_pitch``'s effort limit. Because ``dr.joint_friction`` scales
  it while ``dr.effort_limits`` only weakens the budget, friction could reach the
  joint's entire torque range and lock it. The X2 sets friction per actuator
  group instead.

Both examples are X2-specific; the general lesson is to check every *absolute*
range against the smallest joint group in the model.

### Not covered

State-estimator error cannot be randomized: training can only feed the actor a
perfect quantity plus noise. The X2 deployment reconstructs ``base_lin_vel`` from
the pelvis pose, so the honest fix is to not depend on it — which is why the X2
work trains the ``-No-State-Estimation`` variant, whose actor drops
``motion_anchor_pos_b`` and ``base_lin_vel`` while the critic keeps them.

Where the *nominal* values themselves come from is a separate question from how
they are randomized. For the X2 the actuator gains and friction are pinned to
measured hardware values, not to a physical model, and
``docs/source/x2_gain_provenance.md`` records that evidence and its limits. A
policy trained with these values carries them in its ONNX metadata and the
deployment controller commands exactly them, so a gain change invalidates older
checkpoints rather than merely making them stale.
