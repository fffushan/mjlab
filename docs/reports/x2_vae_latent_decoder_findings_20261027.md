# X2 vgk36 VAE: the latent and the decoder

Date: 2026-10-27 (+08). Kind: findings note. Scope: software and simulation
evidence only. No robot test ran, and no training or export changed.

This note collects what the measurements establish about the 32-channel latent and
about the VAE decoder that consumes it. It supersedes the decoder sections of
`vae-vgk36-latent-shape-20261024` and `vgk36-decoder-probe-261453` where they
disagree.

## 1 Method

I used four sources.

| Source | Content |
| --- | --- |
| Decoder probe | the trained `ConditionalVAE`, the deployed `decoder.onnx`, and the recorded action of one hardware window |
| Hardware window | `d5-window-261453-1792844071919039.bin`, 437 executed ACTION rows, 8.78 s |
| Rollout store | `d1-vgk36-double-20261012`, 1,431,189 student rows with a latent |
| Closed loop | five rollouts of 5 s on the local GPU, terminations cleared |

The hardware window shows the deployed regime. The rollouts are controlled
experiments.

## 2 Integrity

The decoder is faithful to its checkpoint.

| Check | Value |
| --- | ---: |
| `decoder.onnx` against the trained model | 5.07e-07 |
| Decoder against the recorded action, 437 rows | 2.98e-07 |
| Torch decoder against the ONNX graph, attribution study | 1.19e-06 |
| Level attribution completeness | 1.02e-04 against 0.1266 |
| Increment linearisation, midpoint | R² 0.9983 |

## 3 Results

### 3.1 The decoder input has no reference channel

The input has 99 channels.

| Block | Width |
| --- | ---: |
| `projected_gravity` | 3 |
| `gyro` | 3 |
| `joint_q_minus_default` | 31 |
| `joint_dq` | 31 |
| `previous_action` | 31 |

The 130-wide reference vector is an **encoder** input. The decoder never sees it.
All motion content must therefore travel through the 32-channel latent.

### 3.2 The action is a PD target, not a setpoint

```
target = default_joint_position + scales * action
tau    = kp * (target - q) - kd * dq
```

The decoder shapes a feedback law. The joints settle where that law balances
gravity and contact. That equilibrium is not the target.

### 3.3 The latent is low dimensional

| Quantity | Deployed window | Rollout store |
| --- | ---: | ---: |
| Effective channels | 3.0 of 32 | about 6 of 32 |
| Participation ratio | 3.16 | 5.31 |
| Effective rank | 3.93 | 5.71 |
| Active channels | {6, 10, 9, 17, 25, 18, 2} | same set |
| Share of the largest channel | channel 10 carries 51.2% | dim 25 largest |

In the deployed window the latent step norm has median 0.379, and its per-channel
standard deviation is 0.306 of the training marginal at the median.

### 3.4 The latent has little authority at a standing pose

I swapped latents while I held the conditioning fixed.

| Conditioning | Action with a fast latent | Action with a standing latent | Change |
| --- | ---: | ---: | ---: |
| standing pose | 0.182 | 0.180 | 0.0093 |
| pose in motion | 0.803 | 0.787 | 0.0479 |

The real change between the standing regime and the motion regime is **0.791**.
The latent supplies 1/85 of that change at a standing pose, and 1/17 at a pose in
motion.

### 3.5 The authority saturates

I scaled the latent along the standing-to-fast direction.

| Scale | Latent norm | Mean `|action|` | Change |
| ---: | ---: | ---: | ---: |
| 1 | 0.55 | 0.180 | 0.0023 |
| 5 | 2.76 | 0.179 | 0.0142 |
| 10 | 5.53 | 0.233 | 0.115 |

A latent 10 times outside the training range still gives 0.115 of the required
0.791.

### 3.6 The latent-to-action map is almost rank 2

Jacobian of the 32 latent channels against the 31 action channels, at the
standing point.

| Quantity | Value |
| --- | ---: |
| Largest singular value | 0.0935 |
| Smallest singular value | 1.28e-04 |
| Condition number | 733 |
| Effective rank | 1.96 |

### 3.7 Attribution, one hardware window

The increment shares need no baseline.

| Block | Increment share | Increment R² | Level share, neutral baseline | Level share, mean baseline |
| --- | ---: | ---: | ---: | ---: |
| latent | **0.462** | 0.167 | 2.33% | 4.8% |
| `previous_action` | 0.388 | **0.612** | **65.72%** | 22.1% |
| `joint_q_minus_default` | 0.050 | 0.123 | 29.82% | 5.9% |
| `joint_dq` | 0.044 | -0.002 | 1.07% | 0.2% |
| `gyro` | 0.036 | 0.039 | 0.47% | 0.2% |
| `projected_gravity` | 0.020 | 0.045 | 0.61% | **66.8%** |

The level shares depend on the reference point of the decomposition. The
increment shares do not.

### 3.8 Closed loop with a fixed latent

| Reference | Latent | Mean `|dq|` | Mean `|ref-q|` | Root height min | Root height final |
| --- | --- | ---: | ---: | ---: | ---: |
| standing | standing, fixed | 0.144 | 0.038 | 0.647 | 0.657 |
| standing | fast, fixed | 0.149 | 0.048 | 0.652 | 0.654 |
| fast | fast, fixed | 0.481 | 0.350 | 0.149 | 0.173 |
| fast | standing, fixed | 0.510 | 0.339 | 0.115 | 0.136 |
| fast | encoder, changes each tick | 1.886 | 0.155 | 0.477 | 0.563 |

The reference root height of the fast clip is 0.483 to 0.697 m.

## 4 Findings

1. The decoder is faithful to its checkpoint. The export is not a defect.

2. The decoder input has no reference channel. The latent is the only path for
   motion content.

3. The action is a PD target, so the decoder shapes a feedback law. A fixed latent
   fixes that law.

4. The latent is low dimensional. It uses 3 effective channels in the deployed
   window and about 6 in the store.

5. The latent has 1/85 of the authority it needs at a standing pose, and 1/17 at a
   pose in motion.

6. The decoder saturates. A latent 10 times outside the training range still
   delivers 0.115 of the required change.

7. The latent-to-action map is almost rank 2 at a standing pose, with a condition
   number of 733.

8. The latent drives the action increment more than any other block, at 0.462 of
   the share. It drives the level at 4.8% or less.

9. A fixed latent cannot reach a fast state. On a standing reference it changes
   nothing. On a fast reference it produces large actions and the robot falls.

10. A latent that changes on every tick tracks the same fast reference. The root
    stays inside the reference height band and the tracking error is 0.155 rad.

11. A fixed law cannot correct an offset that it did not create. The same frozen
    standing latent stands from the standing start with 0.080 rad of joint drift,
    and falls from the tennis start with 0.319 rad. The two start poses differ by
    0.276 rad on average.

12. The training store has no cold start. No episode starts within the first 9
    frames of a clip. The first action of an episode is 0.77 of that episode's
    median.

13. Only the student rows of the store carry a latent: 1,431,189 rows against
    863,453 teacher rows without one. The latent diffusion therefore trains on
    student-visited states only.

## 5 Established

- The decoder reads 99 channels and has no reference input.
- The latent is a low-dimensional posture-like code with about 2 effective
  directions of action authority at a standing pose.
- Motion requires the latent to change over time. A constant latent gives one
  feedback law and one equilibrium.
- The deployed plan window is piecewise constant, so the deployed loop receives a
  near-constant feedback law.

## 6 Open items

1. The latent has the largest increment share of 0.462 but a single-block R² of
   0.167. I do not know why. A partial-correlation check on the increment can
   answer this.

2. I did not measure whether the fixed point of the deployed loop is stable under
   small perturbation, or only nearby.

3. I did not test a latent that advances at a different rate than the reference.
   I did not measure the relation between latent phase and reference phase.

4. Nobody measured the data that trains the decoder. The DAgger set is a separate
   artifact, and the short-circuit question about pose and latent co-occurrence
   applies to it.

5. No robot test ran. Physical behaviour is unknown.

## 7 Limits

I applied the ASD-STE100 syntax rules. The approved-word dictionary is not
available in this environment, so I cannot guarantee word-level compliance.

The attribution rows are 437 autocorrelated samples of one trajectory of 8.78 s.
I ran no significance test. The closed-loop evidence is five runs of 5 s, on one
fast clip and one standing clip, with one seed and with terminations cleared.

## 8 Where the evidence lives

| Content | Path |
| --- | --- |
| Latent shape, store statistics | `logs/diagnostics/vae-vgk36-latent-shape-20261024/` |
| Decoder probe, cohort comparison | `logs/diagnostics/vgk36-decoder-probe-261453/` |
| Input attribution, two studies | `../../x2_docker/logs/diagnostics/vgk36-input-attribution-20261027/` |
| Latent authority, open loop | `logs/diagnostics/vgk36-latent-authority-20261027/` |
| Closed loop, five rollouts | `logs/diagnostics/vgk36-latent-authority-20261027/CLOSED-LOOP.md` |
| Episode start profile | `logs/diagnostics/vgk36-initial-pulse-20261027/` |
| 5 s playback with a live reference | `logs/diagnostics/vgk36-single000-5s-20261027/` |
