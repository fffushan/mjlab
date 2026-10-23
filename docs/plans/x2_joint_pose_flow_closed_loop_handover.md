# X2 joint-pose flow: closed-loop controller handover

Date: 2026-10-23 (+08).
Status: design analysis only. No controller code written, no simulation run, no hardware.
Scope: how a trained 68-channel joint-pose flow model can drive the frozen X2
multi-motion tracker in closed loop, and what blocks that today.
Related: `docs/plans/x2_flow_matching_closed_loop.md` (the latent 167-token lane),
`docs/reports/x2_joint_pose_flow_training_runs.md` (the training arms).

## 1. The intended loop

The operator stated the target design:

1. Build one environment and one robot.
2. Read every measurement the model needs directly from the environment. Those
   measurements are the environment state.
3. The model denoises a trajectory from that state.
4. The controller feeds the needed frames to the multi-motion tracker.
5. The tracker produces the final action.
6. The action goes to the robot, in simulation through `env.step`, on hardware
   through the control topics.

That design is correct and it matches the pattern the D3 latent lane already
implements. This document records the differences for the joint-pose lane, the
missing pieces, and the open questions.

## 2. What exists today

| Component | Location | What it does |
|---|---|---|
| Handover plan builder | `diffusion/handover.py::build_reference_free_plan` | stacks 8 measured history states plus the current state, re-expresses them in the current window-centre frame, writes a `41 x 167` conditioning window, marks future rows unknown |
| Continuation | `diffusion/handover.py::ReferenceFreeContinuation` | denoises that window per tick, decodes it, packs `snapshot.packed.reference` |
| Runtime | `diffusion/runtime_x2.py` | builds the env, drives the frozen policy and the model, calls `env.step`, records post states |
| Batched variant | `diffusion/parallel_closed_loop_x2.py` | the same lifecycle for many rows |
| Outer harness | `scripts/experiments/replay_survey/survey.py` | builds the multi-motion env, pins one reference per row, replays from frame 0, records completion and termination terms |

The D3 lane never writes the tracker's reference clip. It replaces the **reference
preview observation** that the frozen policy reads, and the clip stops being
consulted after handover. That is why that lane needs a VAE: the latents are the
intermediate representation of the preview.

## 3. What the frozen tracker actually consumes

Verified in source, not assumed:

| Consumer | Reads | Absolute root x, y needed |
|---|---|---|
| `command` observation | reference `joint_pos` and `joint_vel` at the current index | no |
| `motion_lookahead` observation | reference `joint_pos` and `joint_vel` at `_MULTI_MOTION_PREVIEW_S = 0.2` s, that is 10 frames | no |
| `motion_anchor_ori_b` observation | orientation difference between reference anchor and robot anchor | no |
| `motion_anchor_pos_b` observation | position **difference** in the robot anchor frame; dropped entirely in the no-state-estimation variant | no |
| `anchor_pos` termination | `bad_anchor_pos_z_only`, threshold 0.25 m | no, z only |
| `ee_body_pos` termination | `bad_motion_body_pos_z_only`, threshold 0.25 m | no, z only |
| `anchor_ori` termination | `bad_anchor_ori`, threshold 0.8 | no |
| `motion_body_*` rewards | body offsets rotated into the anchor frame | no |
| `motion_global_root_pos` reward | world anchor position difference | needed for training, not for evaluation |

**Correction to an earlier statement in this workstream.** Root translation is
not required as an input. The only global quantity is the anchor difference,
which is what the operator said. The reference anchor's **height** matters
because both position guards are z-only. The absolute x and y placement is
arbitrary for evaluation.

This means a reference can be built from a generated window alone, with no clip
substitution and no matched recorded body trajectory.

## 4. Why the D3 lane cannot be reused unchanged

| D3 requirement | Joint-pose model |
|---|---|
| 135-D state, 167-channel token | 68 channels: root 6-D orientation, 31 `q`, 31 `dq` |
| frozen VAE encoder and decoder | no VAE, no latent |
| checkpoint sidecar naming contract, projection, VAE, cohort | no sidecar of that shape |
| replaces a policy preview observation | the frozen multi-motion tracker reads the command's joint arrays instead |

Using D3 would require inventing a contract and a fake VAE. That is the wrong
tool.

## 5. The loop that must be built

Per control tick, 20 ms at 50 Hz:

1. Read the measured state from the env: root pose and velocity, `q`, `dq`.
2. Encode the last 8 rows into the 68-channel window, using the dataset's
   heading-local frame convention.
3. Every K ticks, run the flow model on 9 known rows and receive 41 rows. Rows
   0 to 7 are history. Rows 8 to 40 are generated.
4. **Install** the generated rows as the tracker's reference for the next K
   ticks.
5. Let the frozen tracker build its observations and produce an action.
6. `env.step(action)`, record the post state, return to step 1.

Step 4 is the missing piece. No API replaces a multi-motion reference in the
middle of an episode; a source search found none.

| Option | Size | Risk |
|---|---|---|
| Write generated rows into `command.motion.joint_pos` and `joint_vel` ahead of the current index, then refresh the anchor cache | about 80 lines plus a cache refresh | mutates buffers the command also uses for metrics and time-step accounting |
| Subclass the command with a ring buffer that the generator fills | about 200 lines | duplicates time-step, metric and anchor logic; divergence from the frozen command is a correctness risk |

The first option is smaller. The second is safer against metric drift. Neither is
written.

## 6. Timing budget

| Quantity | Value | Source |
|---|---|---|
| Window length | 41 rows, 8 history plus 33 generated | dataset contract |
| Window duration | 0.82 s at 50 Hz | derived |
| Reference preview offset | 0.2 s, 10 frames | `_MULTI_MOTION_PREVIEW_S` |
| Denoising steps | 20 Euler calls | `sampler.DEFAULT_CALLS` |
| Control period | 20 ms | `env.step_dt` |
| Model cost per window, batch 1 | **not measured**; rough estimate 20 to 60 ms on one RTX 4090 | estimate only |
| Regeneration period K | candidate 16 ticks, 320 ms | design choice, not tested |

The model cannot run every tick at the estimate above. With K = 16 the compute
is 6 to 19 percent of one period budget. Smaller K gives fresher references and
costs more compute. Larger K gives fewer re-anchorings and staler references.

**A static replay is not useful.** Replaying one generated window gives an
episode shorter than one second. The loop must roll.

## 7. Frame continuity

Each training window is heading-local to its own centre row. This is convenient
for training and awkward at runtime. At every regeneration the generated future
must be rotated into the world frame with the measured current heading, or the
reference heading jumps. The D3 lane solves the equivalent problem by
re-expressing 8 measured states in the new centre frame. The joint-pose version
is the same idea with yaw only, and each regeneration adds a small re-anchoring
jump that must be logged and bounded.

## 8. Representation limits

| Limit | Consequence |
|---|---|
| No root translation in the 68 channels | the reference is in place: the pelvis stays at its anchor |
| Net travel not representable | a clip whose pelvis moves across the floor cannot be described; the anchor guards would then ask the robot to remain near the anchor |
| Root height not in the channels | must be recovered by forward kinematics and a ground-contact rule |
| Heading-local windows | continuity depends on the yaw composition in section 7 |

These limits are a property of the representation, not of the harness. Removing
them needs root-position channels and a rebuilt dataset, which is a separate
decision.

## 9. Building a reference from a generated window

1. Forward kinematics from the 31 generated joint angles, using `x2_ultra.xml`,
   the same asset the simulator compiles.
2. Root orientation from channels `0:6`, composed with a chosen world heading.
3. Root height: lower the pelvis until the lowest foot point touches `z = 0`.
4. Root x and y: the environment origin.
5. Body linear and angular velocities: finite difference of the forward-kinematic
   sequence at 50 Hz.
6. Write an NPZ with `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
   `body_lin_vel_w`, `body_ang_vel_w`, `fps`.

Validation before any simulation: the complete-reference re-recordings hold both
`q` and all 32 body poses. Forward kinematics of `q` must reproduce the recorded
body positions in the pelvis frame. A few millimetres is the pass mark. A joint
order error would show up immediately; an earlier audit measured a 0.049 rad
self-consistency check on the same ordering.

## 10. Open questions for the next session

1. Which K, and does the reference hold constant between regenerations or
   interpolate?
2. Does the install write only joints, or also forward-kinematic body poses?
   The z-only guards read body heights.
3. Does the harness reuse `survey.py` environment construction or need a new
   driver? `survey.py` currently builds references from the achieved store, so a
   generated-window path needs an input adapter either way.
4. Should evaluation run with the teacher's domain randomisation as configured,
   or with the play-mode overrides?
5. Do the mask arms need history masked at test time for a fair comparison? The
   deployed condition uses real measured history.
6. What is the pass criterion: completion rate over N windows, which error
   terms, and which control run.

## 11. Control arm is mandatory

The replay survey measured **89.6 percent completion** over 199,879 recorded
reference segments, so a raw completion number from generated references means
nothing on its own. The closed loop must run the same N windows twice: once with
generated joints and once with the recorded reference. The difference is the
model. Without that pair the result is not interpretable.

## 12. What can be built with CPU only

| Step | Testable without a GPU |
|---|---|
| Window encoding from measured history | yes, round-trip against a recorded window |
| Reference construction by forward kinematics | yes, against the complete-reference re-recordings |
| Export directory to survey manifest adapter | yes |
| Install hook | yes, with a tiny fake model |
| The loop itself | no, needs the simulator and one GPU |

## 13. Prohibitions

- Do not modify the frozen teacher, the action conversion, the servo path, the
  observation terms or the preview offset.
- Do not claim deployment, real-time performance or physical validity from a
  simulation result.
- Do not run on hardware from this workstream.
- Do not change the training arms to suit the controller.
