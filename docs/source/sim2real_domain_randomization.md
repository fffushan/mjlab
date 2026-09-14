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
