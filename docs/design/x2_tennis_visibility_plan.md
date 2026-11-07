# Visibility gate: implementation plan (measurement part of §4.1)

Date: 2026-11-07 (+08).
Status: **approved for implementation.** No training run is authorized by this document.

This plan implements only the **measurement** half of
[mixed safety curriculum](x2_tennis_mixed_safety_curriculum.md) §4.1. Rewards, the
mixture, induced losses and any new stage are out of scope.

## 1. Decisions taken by the operator

| item | decision |
|---|---|
| camera mount | **training value** `(0.05058, 0.03013, 0.032504)` and quat `(0.497273, -0.5026607, 0.5070336, -0.4929186)`, parented to `head_pitch_link` |
| dropout | a **config parameter**, default **0.0** for this stage; the gate is **pure geometry** |
| range bias | **not modelled**; the measurement stays unbiased |
| baseline checkpoint | **S9b arm 1**, `model_27500.pt`, impacts 21.9 / 22.9 / 28.6 |
| frame convention | keep **capture-time** pelvis-frame storage; the controller's read-time re-expression is a known limitation to fix there (C6) |
| invalid history | **all 12 zeros** until the first admitted frame, matching `HasValue() == false`; no width change, actor stays 112 |

## 2. Geometry, from the estimator session

The estimator sees a **480×288 rectified pinhole**, not the raw fisheye frame. Values
are for `.147` at 516×388; `.162` differs by under 1.5° per side.

| quantity | value |
|---|---|
| horizontal half-angle, left | 65.7° = 1.1469 rad |
| horizontal half-angle, right | 67.2° = 1.1728 rad |
| vertical half-angle, up | 53.9° = 0.9407 rad |
| vertical half-angle, down | 54.1° = 0.9443 rad |
| minimum range | 0.100 m (hard floor in the reach mask) |
| maximum range | 1.000 m **straight-line from the left camera centre** — a sphere, not forward depth |
| smallest range validated on hardware | 0.412 m; anything closer is untested, not forbidden |

The existing reward constants `CAMERA_HALF_ANGLE_X = 1.4907` (85.4°) and
`CAMERA_HALF_ANGLE_Y = 1.1168` (64.0°) are the **raw fisheye** half-angles. In tan space
that cone is 5.43× wider horizontally and 1.49× wider vertically than the delivered
image. **This plan does not change them.** The mismatch is recorded here; changing a
reward cone is a reward decision and is out of scope.

## 3. Work items

### W1. Constants (`tennis_constants.py`)

Add, next to the camera constants:

- `BALL_VISIBLE_HALF_ANGLES_RAD = (1.1469, 1.1728, 0.9407, 0.9443)` — left, right, up, down.
- `BALL_VISIBLE_MIN_RANGE_M = 0.1`
- `BALL_VISIBLE_MAX_RANGE_M = 1.0`
- `BALL_VISIBILITY_DROPOUT_PROB = 0.0`
- `BALL_MIN_VALIDATED_RANGE_M = 0.412` — documented only, not enforced.

Each gets a docstring stating the source, the units, and that the value is a
prior from the estimator's shipped code rather than a calibration.

### W2. Geometry helper (`mdp/ball_visibility.py`)

Already present: `camera_pose_w` and `ball_in_camera_region`. Change
`ball_in_camera_region` to take **four** half-angles `(left, right, up, down)` instead of
two symmetric ones, and to test:

```
theta_x in [-left, +right]  and  theta_y in [-up, +down]
|delta| in [min_range_m, max_range_m]
forward > 0
```

Keep it pure geometry: no dropout, no randomness, no state. Dropout belongs to the
stream, not to the geometry test.

### W3. Stream gate (`mdp/observations.py`, class `ball_history_samples`)

New parameters, all optional and defaulting to the gate being **off**:

- `visibility_head_cfg: SceneEntityCfg | None = None`
- `visibility_mount_xyz`, `visibility_mount_quat`
- `visibility_half_angles: tuple[float, float, float, float]`
- `visibility_min_range_m`, `visibility_max_range_m`
- `visibility_dropout_prob: float = 0.0`

Behaviour when `visibility_head_cfg` is not None:

1. Every control step, write a **boolean visibility ring** entry for the true ball at
   that step, computed from the camera pose **at that step**.
2. When an acquisition is due, admit it **only if** the frame at `capture_step` was
   visible. The latency draw still happens in the same order, so seeded replays stay
   reproducible.
3. If the sampled frame was not visible, **insert nothing** and retry on the **next**
   control step (`next_acq = now + 1`). The history ages grow, which is the silence.
4. `dropout_prob > 0` marks a fraction of otherwise-visible frames invisible at write
   time. **When the probability is 0.0 no random number is drawn**, so a gate-off or
   zero-dropout run consumes exactly the same randomness as today.
5. Never-seen: while no sample has been admitted, output **all 12 numbers zero**.
   Today the term emits zeros with an age that grows from 0.0, which reads as a fresh
   sample at the pelvis origin.
6. The reset-time acquisition is no longer forced: `_next_acq = 0` still makes step 0
   *due*, but the gate decides whether it is admitted.
7. Visibility ring entries are cleared on reset, so a prior episode cannot admit a
   sample.

Behaviour when `visibility_head_cfg` is None: **identical to today**, including the
number of random draws.

### W4. Configuration wiring (`tennis_juggle_env_cfg.py`)

Pass the W1 constants to the actor's `ball_hist_pelvis` term. Enable the gate from an
ablation variable, `MJLAB_JUGGLE_VISIBILITY_GATE`, default **off**, so every existing
stage keeps its current stream byte for byte. A later stage turns it on through its
stage delta; this plan adds no stage.

### W5. Checks, as scripts not tests

Under `scripts/experiments/x2_tennis_visibility/`, one runnable script with cases. It is
a **script**, not a unit test: the repository rule is no new test files.

| case | expected |
|---|---|
| gate off | the 12 numbers are **identical** to the same probe run captured before the change (same seed, same env count, same step count) |
| unseen reset | all 12 zeros on the first steps; no fabricated sample at the pelvis origin |
| 70° off-axis | never admitted |
| 40° off-axis, 0.6 m | admitted |
| 0.7 m forward but 1.2 m from the camera centre | rejected by the sphere |
| captured visible, ball leaves the view before the sample arrives | the sample is still admitted, with the capture-step age |
| long gap | ages increase monotonically, slot 0 ≥ one period |
| two envs, one visible and one not | independent: one keeps acquiring, the other holds |

Probe budget: at most **8 environments** and **40 control steps** per case.
Total runtime target under two minutes on one GPU.

## 4. Acceptance

- Gate off: bit-identical stream versus the pre-change baseline.
- Gate on: the eight cases above pass, with numbers printed.
- `tests/test_x2_tennis_stage_task.py` and `tests/test_task_configs.py` still pass.
  Existing assertions may be updated if the observation contract changed; **no new
  test file and no new test function**.
- `ruff format` and `ruff check` clean on every touched file.
- No training job, no checkpoint, no hardware action.

## 5. Explicit non-goals

- Reward cones, `ball_in_fov`, alignment or energy weights.
- A dropout rate above zero, estimator range bias, correlated noise.
- Occlusion, ghost rejection, replay rejection, epoch restart: there is no stamp domain
  in the simulator, so these are documented as not modelled.
- The mixed mixture, induced losses, episode-budget changes, a new stage.
- Any run.
