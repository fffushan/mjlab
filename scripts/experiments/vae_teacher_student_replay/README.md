# X2 VAE teacher-fidelity replay

This directory contains a reusable **offline one-step fidelity evaluator** for the X2 VAE. It feeds the same recorded state/reference/history to the frozen teacher and VAE student, then measures teacher-versus-student raw-action and target-position disagreement.

This is intentionally not a closed-loop stability evaluator. It does not apply either action to MuJoCo, an actuator model, a robot, contact state, or a next-state transition. The metric answers the narrower VAE objective:

> Does the student reproduce the teacher's action on the same recorded policy state?

## Recorded fixture

The compact fixture is tracked at:

```text
tests/fixtures/vae_teacher_student/recorded_states.npz
tests/fixtures/vae_teacher_student/manifest.json
```

It contains 1,103 policy-state snapshots:

- 110 frames from the first native-bag-backed failure;
- 993 frames from successful `r4/index=1/tennis_001` handovers (generations 2, 4 and 6).

The arrays preserve the deployed VAE inputs (`reference[68]`, `conditioning[99]`), recorded student raw actions/latents, reference frame, engagement generation, assembly timestamp and run kind. This is a compact policy-state fixture, not a replacement for the original ROS bags or actuator/contact telemetry. `manifest.json` records source trace and model SHA-256 values.

## Models

The evaluator deliberately takes model paths as an argument instead of silently loading mutable training or deployment files. The model directory must contain:

```text
baseline-tennis-001.onnx
encoder.onnx
decoder.onnx
contract.json
```

The frozen copies used for the initial evaluation are under:

```text
/home/agiuser/data/analysis_20260930/pc2_teacher_vae_trace_20260930T152353Z/early_failure/models/
```

Verify those files against `tests/fixtures/vae_teacher_student/manifest.json` before interpreting results.

## Run

From the repository root:

```bash
uv run python scripts/experiments/vae_teacher_student_replay/evaluate.py \
  --states tests/fixtures/vae_teacher_student/recorded_states.npz \
  --model-dir /path/to/frozen/models \
  --output-dir logs/diagnostics/vae_teacher_student_replay
```

The output directory contains `metrics.json`, `comparison.png`, and one compressed NPZ per run (`failure.npz`, `r4_g2.npz`, `r4_g4.npz`, `r4_g6.npz`). `logs/` is intentionally ignored by git.

The teacher observation is reconstructed as:

```text
command q31+dq31, motion_anchor_ori_b6, pelvis gyro3,
relative joint q31, joint dq31, previous accepted action31 = 164 values
```

The teacher `time_step` is the recorded reference frame. The evaluator validates teacher embedded q/dq parity and student encoder/decoder replay parity before reporting disagreement. Raw normalized action is the primary metric; target-position comparison uses each policy's own frozen default/scales because the teacher ONNX metadata rounds four scale values.

## Current captured result

The initial run on these frozen copies found successful-run target disagreement of approximately `0.0285–0.0289 rad RMS` over complete trajectories, versus `0.1316 rad RMS` on the first failure. This number is policy-fidelity evidence, not evidence that the teacher would have prevented the physical fall.
