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
