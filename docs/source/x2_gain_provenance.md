# AgiBot X2 actuator gains: where every number comes from

The X2 actuator configuration in `src/mjlab/asset_zoo/robots/agibot_x2/x2_constants.py`
carries nominal PD gains, dry friction per joint group and an action scale per group.
This page records the evidence behind those numbers, because the project has five
different "vendor" gain tables in circulation and they disagree by up to 12x.

Nothing here changes the model's physical parameters: body inertials, kinematics,
collision geometry and joint armature are untouched.

## 1. The five gain tables

| # | table | what it is | where it lives | hip / knee |
|---|---|---|---|---|
| 1 | mjlab training | the reflected-inertia 10 Hz template this repo used | this repo, `x2_constants.py` (before this change) | 170.8 / 10.87 |
| 2 | vendor controller defaults | fallback table compiled into the vendor RL node | `extra/x2_rl_deploy/.../include/.../robot_model.h` | 40/4, 80/8 |
| 3 | vendor hold table | `default_joints` the vendor controller interpolates to on JOINT_DEFAULT | `extra/x2_rl_deploy/.../config/motion_control.yaml` | 40/4, 80/8 |
| 4 | vendor sample-policy gains | the gains the vendor's own example RL policy deploys with (`rl_config.kps`/`kds`) | same `motion_control.yaml` | 120/5, 150/5 |
| 5 | **native runtime** | what the real robot's own motion controller commands while standing (STAND_DEFAULT) and while damping | **no file — measured from recorded ROS 2 messages** | 100/4, 150/5 |

Tables 2-4 are files in the vendor SDK tree (outside this repo). Table 5 exists only on
the robot.

## 2. The measurement chain (table 5)

```
logs/native_capture/<ts>/bag                  # rosbag2 sqlite3, recorded on the robot
  -> native_capture/analyze_bag.py            # decodes /aima/hal/joint/{leg,waist,arm,head}/command
     -> analysis/series.npz, report.md
        -> native_capture/emit_hardware_profile.py
           -> analysis/x2_native_profile.yaml  # damping / joint_default / standing tables
```

`JointCommandArray` carries `position, velocity, effort, stiffness, damping` per joint, and
the gains below are the recorded `stiffness`/`damping` fields — **not** a configuration file.
Two independent captures agree, on two firmware builds:

| capture | firmware | mode | hip_pitch | knee | ankle_pitch | ankle_roll | waist_pitch | shoulder_pitch | elbow | wrist_pitch | head_yaw | head_pitch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260915T130441Z_native_standing` | `dev-...v1.1.0.95` | STAND_DEFAULT | 100/4 | 150/5 | 40/3 | 30/2 | 40/5 | 30/1 | 50/1 | 20/1 | 3.4/0.114 | 10/0.8 |
| `20260915T141139Z_native_joint_stand` | `release-...v1.1.4` | STAND_DEFAULT | 100/4 | 150/5 | 40/3 | 30/2 | 40/5 | 30/1 | 50/1 | 20/1 | 3.4/0.114 | 10/0.8 |

Within a mode window the recording holds exactly **one** `(kp, kd)` per joint (checked
directly in `analysis/csv/command_*.csv`), so these are not medians over a ramp. The
`stiffness_std` that the analyzer reports next to them is ~24.7% *relative for every
joint*, which is a segment-boundary artifact, not a vendor gain schedule — do not read it
as one.

Two things make these numbers more than a curiosity:

- **They are the same law MuJoCo implements.** Regressing the *measured* joint torque
  against the *commanded* PD law over clean windows gives
  `tau_meas = Kp (q* - q) - Kd qdot + (0.03 .. 0.3) N.m` with a residual standard
  deviation of ~0.1 N.m on the legs. The real servo is a position-PD torque source, which
  is exactly `BuiltinPositionActuatorCfg`.
- **They are ordered by joint, not by reflected inertia.** The template's per-group gains
  scale with rotor inertia, which varies 1.1x-165x across these joints relative to the
  load inertia they actually move; the measured gains do not do that, which is why the
  template was 12x too soft on head pitch and 1.7x too stiff on the hip.

Caveats, kept deliberately in view:

- These are **commanded** gains, and the measurement says nothing about the loop the
  motor-side firmware actually closes. See §5.
- The standing table is a *balance controller's* gains and the `JOINT_DEFAULT` table
  (40/4, 80/8, waist 300/3 in table 3) is a *posture hold* table. Neither is an RL
  policy's gains; they bracket them. The vendor's own example RL policy (table 4) sits
  within ~25% of the standing table on the legs, which is why the standing table is the
  nominal here and the DR width below was left alone.
- On the real robot the *commanded* position deliberately differs from the measured one
  (up to 0.21 rad on the waist): that error is what generates the holding torque. Do not
  read the commanded profile as the robot's pose; the *measured* state is the pose, and
  that is what the keyframe was aligned to.

## 3. What the config now uses

| group | joints | Kp / Kd | source |
|---|---|---|---|
| `hip_pitch` | L/R | 100 / 4 | measured |
| `hip_roll` | L/R | 100 / 3 | measured |
| `hip_yaw` | L/R | 100 / 3 | measured |
| `knee` | L/R | 150 / 5 | measured |
| `ankle_pitch` | L/R | 40 / 3 | measured |
| `ankle_roll` | L/R | 30 / 2 | measured |
| `shoulder_pitch` | L/R | 30 / 1 | measured |
| `shoulder_roll` | L/R | 20 / 1 | measured |
| `shoulder_yaw` | L/R | 20 / 1 | measured |
| `elbow` | L/R | 50 / 1 | measured |
| `wrist_yaw` | L/R | 50 / 1 | measured |
| `wrist_pitch_roll` | L/R `wrist_pitch`, `wrist_roll` | 20 / 1 | measured |
| `head_yaw` | | 3.4 / 0.114 | measured |
| `waist_yaw` | | 40 / 8 | measured (see §7) |
| `waist_pitch_roll` | `waist_pitch`, `waist_roll` | 48.312 / 3.076 | template |
| `head_pitch` | | 0.798 / 0.051 | template |

Rejected alternatives, and why:

- **Apply table 3 or 4 instead.** Table 3 is a static hold table (`waist_pitch` at Kp 300
  against a 48 N.m limit saturates at 0.16 rad of error) and table 4's joint order is not
  pinned by the artifacts available here, so neither can be mapped onto 31 joints with
  confidence. They are used as an envelope, not as the nominal.
- **Waist yaw** used to keep the template value for the same reason, until a third and
  fourth source resolved it: the vendor's sample policy ships 40.1792/2.5579 for this
  joint, which is the 10 Hz template applied to the armature in AgiBot's own Isaac
  actuator config — i.e. the vendor's design point is ~40, and the measured 40/8 sits
  there. The measurement now wins and the PFP-96 armature inference is gone (§7).
- **Waist pitch/roll.** Still the vendor's two tables disagreeing 7.5x (40/5 vs 300/3),
  so the template value stays; it is 21% above the measured stand value anyway. This is
  now the weakest entry in the table.
- **`head_pitch`.** The measured gain (10/0.8) contradicts this joint's 0.6 N.m effort
  limit: it would saturate at 0.06 rad of error. The capture shows a 0.2 rad tracking
  error with only ~0.05 N.m of measured torque and 16 samples above the limit, so the
  limit, the torque estimate or the gain is wrong. The template value is kept until a
  hardware torque measurement settles it.

### Action scale

`X2_ACTION_SCALE` is now the larger of two floors instead of `0.25 * effort / Kp`:

- **torque floor** `0.25 * effort_limit / Kp` — a unit action stays inside 25% of the
  torque budget, so the joint does not saturate;
- **tracking floor** the reference motion's p95 deviation from the keyframe divided by 5,
  so the motion stays reachable inside a 5-unit action.

Decoupling matters because the old rule silently pinned every low-torque gain: keeping a
sane wrist action scale required keeping the wrist `Kp` 6x below what the robot runs, so
the measured wrist and head gains could never be adopted.

With the maximum rule, the ten groups where the torque floor still wins sit at exactly 25%
of the torque budget for a unit action, and the six where the tracking floor wins sit at:

| group | `Kp * scale / effort_limit` |
|---|---|
| `shoulder_pitch` | 27% |
| `shoulder_yaw` | 29% |
| `wrist_yaw` | 33% |
| `knee` | 33% |
| `elbow` | 50% |
| `wrist_pitch_roll` | 60% |

That is the honest place to be — those six joints really are torque-limited at their
reference amplitudes, and the real joint saturates there too. Two consequences worth
knowing:

- The wrist pitch/roll group is now the sharpest case: a unit action commands 2.9 N.m
  against a nominal 4.8 N.m limit, and against the 0.45x floor of the tracking task's
  wrist effort derating that is 2.16 N.m, so in the weakest randomized environment the
  wrist saturates at ~0.75 action units. The derating exists to match the vendor
  simulator's 2.2 N.m clamp (§5.6), so this is consistent with how the wrist is modelled,
  but it does mean the wrist's action range is torque-limited rather than gain-limited.
- The action table lives on the robot config, not on the task, so the velocity task shares
  it and shares the new keyframe. The tracking floor was measured on the tracking
  reference motion and is a *lower* bound for locomotion rather than a fitted value; a
  velocity-only policy would want at least as much action range, and a checkpoint carries
  its own action scale in the ONNX metadata regardless.

The tracking floor is measured on `data/qianghuo_smplx_agibot_x2_tracking.npz` and
`tests/test_x2_constants.py` re-derives it while that file is present, so a keyframe or
motion change that invalidates it fails the test rather than silently under-scaling the
action.

## 4. Friction and damping

`xmls/x2_ultra.xml` inherits the vendor `<default>` block
`<joint damping="0.0" armature="0.03" frictionloss="0.3"/>`, so all 31 joints used to get
`dof_frictionloss = 0.3 N.m` regardless of size. That is 50% of `head_pitch`'s effort
limit, and it interacted badly with randomization: `dr.joint_friction` scales 0.5-1.5x
while `dr.effort_limits` only weakens, so friction could reach 0.45 N.m against a 0.45 N.m
budget and lock the joint. The armature was always overridden per group; the friction was
not.

Friction is now set per group, scaled down with the torque class (legs keep the 0.3 the
measured residual supports; the low-torque groups get 0.01-0.12), and the per-group
viscous damping randomization `X2_DAMPING_RANGES` replaces the generic absolute
`(0.0, 0.3) N.m.s/rad` range with `effort_limit / 400`, which reproduces 0.3 for the legs
and 0.0015 for `head_pitch`. `tests/test_x2_constants.py` asserts friction stays at or
below 2% of each joint's rated torque, which the old blanket value violated by 25x.

These friction values are order-of-magnitude estimates from the measured torque residual
in the same captures, not identifications. They are a deliberate downgrade of a
known-bad blanket value, not a claim of accuracy.

## 5. Open evidence gaps

1. **Effective vs commanded gains.** No capture tests what the firmware does with
   `Kp = 170.785, Kd = 10.872`. The deployment currently sends exactly the gains in the
   policy's ONNX metadata (`joint_stiffness`/`joint_damping`), so the training loop is
   self-consistent; whether the hardware realizes them is unknown. A gain step-response
   sweep is the missing measurement, and until then the commissioning plan's 0.7x
   multiplier is the only guard.
2. **Firmware pinning.** Gains were reproduced on `v1.1.0.95` and `v1.1.4`; pin the
   firmware version with any profile copied from a capture.
3. **`head_pitch`** torque limit / gain contradiction (§3).
4. **Waist gains** (§3).
5. **Quantization, documented but not modelled.** The real firmware quantizes velocity
   feedback (`0.0122 rad/s` legs, `0.0073` arms, `0.0049` head) and torque output
   (`0.088 N.m` legs) where MuJoCo uses exact `qvel` and continuous torque. The actor's
   existing `joint_vel` observation noise is +-0.5 rad/s, ~40x the velocity quantum, so
   the quantization is not observable at the policy rate; the torque quantum is <0.1% of
   a leg joint's limit. Neither justifies a model change, but a harder task (torque-level
   control, or a policy that reacts within one control step) would need them.
6. **Wrist effort limit.** The tracking task still derates the wrist group to 0.45-1.0x of
   4.8 N.m because the *vendor simulator* clamps it to 2.2 N.m. That is an
   interface-fidelity argument, not a hardware measurement; the measured wrist torque in
   the captures peaks at 0.72 N.m and says nothing about the limit.

## 6. Reproducing the comparison

The vendor tables are outside this repo. To re-derive table 5 from a new capture:

```sh
# on the robot or in the ROS container
python3 native_capture/analyze_bag.py <capture_dir>
# then read analysis/x2_native_profile.yaml (native_standing_profile) and
# cross-check against the raw messages:
grep -l . analysis/csv/command_leg.csv   # one unique (kp, kd) per joint per mode window
```

To print the compiled model's gains for comparison against any table:

```sh
uv run python -c "
import re
from mjlab.asset_zoo.robots.agibot_x2 import x2_constants as X
from mjlab.entity import Entity
m = Entity(X.get_x2_robot_cfg()).spec.compile()
scale = lambda n: next(v for k, v in X.X2_ACTION_SCALE.items() if re.fullmatch(k, n))
for i in range(m.nu):
    a = m.actuator(i); dof = m.jnt_dofadr[a.trnid[0]]
    print(f'{a.name:28} {a.gainprm[0]:9.3f} {-a.biasprm[2]:8.4f} {a.forcerange[1]:6.2f} '
          f'{m.dof_frictionloss[dof]:6.3f} {scale(a.name):7.4f}')
"
```

`X2_ACTION_SCALE` is keyed by joint-name *pattern*, not by joint name, which is why the
snippet resolves the scale with `re.fullmatch`.

Any change to these numbers invalidates existing checkpoints: `joint_stiffness`,
`joint_damping` and `action_scale` are exported into the policy's ONNX metadata and read by
the deployment controller, so a policy trained before the change must not be deployed with
a config generated after it.

## 7. A third-party reference: the sonic-x2 bundle

`~/projects/x2_references/sonic-x2` is a standalone play/deploy bundle for running the
SONIC tracking policy on an X2, contributed by the GR00T/gear_sonic line of work. It is
the closest thing to a peer implementation we have, so it is worth recording what it
settles and what it only appears to settle.

### What it contains

A vendor MJCF (the *original* AgiBot `x2_ultra.xml`, model `x2t2.5`, not a URDF
conversion), one script holding all the constants (`armature → PD gains`, effort limits,
default pose), a player that runs the ONNX policy at 50 Hz with an explicit Python PD, and
two real-robot tuning presets. It carries **no training config and no randomization**; the
upstream training side is `gear_sonic/` in the GR00T tree, where the X2 robot file both
scripts cite (`envs/manager_env/robots/x2_ultra.py`) is not present, so its *trained* gains
cannot be checked directly — only the armature-derived ones the player claims are
training-equivalent.

### Their gains are the same template on a different armature table

Like our pre-2026-09 config, they derive `kp = armature * (20π)²` and
`kd = 4 * armature * (20π)`. Their armature table comes from AgiBot's Isaac
`ImplicitActuatorCfg`:

| group | sonic-x2 | ours (PFP modules) | ratio |
|---|---|---|---|
| hip, knee | 0.025101925 | 0.043260 | 1.72 |
| waist_yaw | 0.010177520 | 0.043260 | 4.25 |
| ankle, waist_p/r, shoulder, elbow, wrist_yaw | 0.003609725 | 0.004951 (0.012237 waist_p/r) | 1.37 (3.39) |
| wrist_p/r, head (one lumped "smallmotor") | 0.004250 | 0.000809 / 0.000387 / 0.000202 | 0.19 / 0.09 / 0.05 |

Two of those rows are informative. Their `waist_yaw` armature (0.010177520) is *the same
number* that reproduces the vendor sample policy's `waist_yaw` gain (40.1792/2.5579), so
two independent vendor-side sources agree — **and they agree with our measurement** (the
native stand profile commands 40/8 for that joint). Our PFP-96 assignment was the
inference, not the measurement: the waist yaw module is 120 N·m in both URDF revisions,
so that joint's armature is the same across revisions and the PFP-96 row was simply
applied to the wrong joint. That is what §3 now uses.

The rest of their table is *not* adopted. The hip/knee row disagrees 1.72x with a
derivation we can point at (rotor inertia × gear ratio² from the PFP datasheet) and the
difference is 1.9% vs 3.3% of the hip's load inertia, so it changes nothing measurable.
The smallmotor row is a single lumped value for four different joints (the wrist's own
load inertia is 72% of it, the head's 111%); our separate PFP-41/25/12 rows reproduce the
*measured* wrist and head stand gains (20/1 and 3.4/0.114) far better than the lumped
value would, so it stays.

### The trap: their effort limits are the newer revision

Their tuning presets and constants use `waist_pitch/roll = 36 N·m` and
`wrist_pitch/roll = 6`, and their notes describe a model trained at 32 N·m "saturating at
the hardware's 24". Adopting those would be a revision mix-up. Both vendor MJCFs are in
`x2_references/agibot_x2_urdf/`, and their `actuatorfrcrange` histograms differ:

| | v1.3.0 `x2_ultra.xml` (this model) | v1.4.0 `X2-Ultra.xml` |
|---|---|---|
| hip, knee, waist yaw | 9 × 120 | 9 × 120 |
| waist pitch/roll | 2 × 48 | (in the 36 group) |
| ankle pitch, shoulders | 6 × 36 | 6 × **60** |
| ankle roll, shoulder yaw, elbow, wrist yaw | 8 × 24 | (in the 36 group) — 10 × **36** |
| wrist pitch/roll | 4 × 4.8 | 4 × **6** |
| head | 2.6, 0.6 | 2 × 2.3 |

This model is v1.3.0 by explicit choice and its compiled limits match that column exactly
(`test_effort_limits_are_the_flagship_revisions`), so their 36/6 are the *new revision's*
numbers, not corrections to ours. The captures agree: waist pitch reaches 42.6 N·m in the
mode-transition window (2 samples above 36, 49 above 30, a smooth tail rather than a flat
cap), which a 36 N·m limit would forbid. Their own MJCF is inconsistent with their own
constants in the same way — its motor `ctrlrange` is the v1.3.0 set (48 waist, 4.8, and
2.2 at the wrist) while their constants use the v1.4.0 set.

### What is worth borrowing later

- **The actuator integration path.** They record that the same numerical KP behaves
  *stiffer* under Isaac's implicit PD than under an explicit `ctrl`-torque loop, and that
  the explicit path needs **ankle KP ×1.5** to recover the lost loop gain (their G16b
  sweep), with the effect being ankle-specific. This is a sim-to-sim result, and it
  confirms that our choice of native `<position>` actuators with `integrator=implicitfast`
  is the path these policies were trained under. It also means any comparison against a
  simulator that applies explicit PD (including the vendor's) carries a ~1.5x effective
  stiffness discrepancy on the ankle.
- **Their ablations agree with two of our choices.** Removing the blanket 0.3 N·m
  friction loss was a wash (their G14: +0.19 s at 2k, neutral at 4k, −0.21 s at 6k,
  `take_a_sip` −2.20 s) and was reverted, which supports keeping ~0.3 on the legs while we
  fix the low-torque groups (§4). Richer contact (condim 4, softer `solref`/`solimp`,
  torsional friction) regressed and was reverted, so our simpler hull setup is not
  obviously the wrong side of that trade.
- **Their deploy-side safety layers** are worth copying into the hardware controller
  rather than the simulator: per-group target-deviation clamps (0.30 rad default, 0.70
  legs, 0.45–0.70 waist, 2.00 arms, 0.50 head) around the default pose, per-group target
  low-pass filters (8 Hz global, 16 Hz legs, off for waist/arms), and an action clip of
  20. Their preset headers name the hardware failure signatures these guard against (a
  5–7 Hz ankle ring, waist wobble) and the test order that exposed them.
- **Their model has the vendor's pelvis defect** and ours does not: total mass 43.475 kg
  vs our 41.966 kg, the entire difference being a pelvis of 5.0318 kg derived from mesh
  density because the body carries no `<inertial>` (our URDF-sourced 3.5235 kg). Every
  other body matches to four decimals. It is a reminder that a vendor MJCF is not
  ground truth just because it is the vendor's.
