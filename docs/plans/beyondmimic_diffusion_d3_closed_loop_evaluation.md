# D3 mjlab closed-loop evaluation — bounded implementation plan

Observation / authorization: 2026-10-04. User requested a plan, delegated
implementation using `lz-fast/gpt-5.6-luna:max`, and 30-minute progress checks.

## Operator milestone disposition (2026-10-05 +08:00)

The user accepts **D3 as done for implementation** and wants to advance the
implementation pipeline, deferring model-quality polishing. The six-condition
pilot's five diffusion falls remain recorded evidence; this is not a claim of
robust behavior, sustained real-time performance or hardware qualification.
D4 sparse keyframe inpainting and physical-cost guidance is the next planned
phase. This status decision does not launch another job or publish source.

## Goal and boundary

Connect the committed D3 seam (`5b0c95854`) to **mjlab simulation**, using the
frozen no-mixture D2 checkpoint and frozen gravity-mode VAE. Establish a truthful
closed-loop measurement tool before judging autonomous behavior. This is not an
x2_docker change, model retraining, vendor sim2sim, export, deployment or hardware
The original implementation request did not authorize a commit or push. A later
user decision (2026-10-05 +08:00) explicitly authorizes the scoped D4a/D3
implementation commit; push and deployment remain unauthorized.

Implementation baseline: `5f17fc6f9b9a28e51baa44425b50106ba0561e86`.
Primary checkout's untracked VAE planning files and old orchestration are excluded.
All delegated writes stay in dedicated detached worktrees under
`/home/agiuser/projects/worktrees/mjlab/d3-closed-loop-20261004T071550Z/`.
Parent owns this plan, frozen seam contract, phase state and final acceptance.

## Frozen inputs and behavior

- Active YAML: `docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml`,
  SHA256 `f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed`.
- D2 checkpoint: staged `../d3-focused-20261003T185144Z/assets/checkpoint-best.pt`,
  SHA256 `252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae`.
- Projection: sibling `assets/projection.npz`, SHA256
  `e258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe`.
- VAE: primary `logs/distillation/mixed-10k/checkpoint-final.pt`, SHA256
  `69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117`.
- Cohort: primary `configs/distillation/x2_tennis_mixed.yaml`; verify manifest and
  all resolved artifacts through existing loaders. Start with `tennis_000`.
- 167-channel tokens, 8 past + current + 32 future; 20 DDIM jumps, eta=0,
  start level 1000, FP32, no guidance, no modified objective or action clipping.
- Simulation control period stays 0.02 s. Wall time may be slower than real time;
  do not claim sustained 50 Hz inference or deadline qualification.

## Execution lifecycle and matching

1. Explicit preflight validates budgets, hashes, schema, joint/body order and
   output path **before constructing an environment**. CLI help/preflight are
   non-operational. Physics needs explicit `--execute`.
2. Reset one selected-motion environment to a declared seed/reference frame.
   Use the existing registry and saved observation/action semantics; no saved
   YAML constructors. Auto reset is disabled in this private evaluation instance.
3. Execute exactly eight frozen-VAE mean-latent actions using live reference68
   and conditioning99 from the same snapshot. Record each **pre-action** state
   with its decoded latent only after successful, boundary-free step evidence.
   Failed bootstrap is reported, not repaired with synthetic history.
4. Diffusion branch switches to a reference-free runtime mode without resetting
   or changing physical state. Observation then has `reference=None` and
   `reference_frame=None`; the policy path never calls encoder/teacher/reference.
   Disable reference-driven timer/wrap resampling, teleports and reference-error
   terminations locally. Do not invoke distillation `adapter.step/snapshot` in
   continuation if they capture reference inputs. Keep physical safety evidence.
5. Each tick uses owned measured WorldState and actual conditioning99 (projected
   gravity, cached gyro/joint q/dq/previous applied action). Reuse caches without
   advancing noise/delay/history twice. Apply generated action through the same
   action manager and record `s_t,z_t`, not `s_(t+1),z_t`.
6. Frozen VAE reference-tracking baseline uses identical reset seed/frame and
   eight-tick bootstrap. Before comparison, verify complete physical initialization
   signatures and bootstrap pre/post state/conditioning/action/latent trajectories
   within 1e-3 (user-selected 2026-10-04; report max differences). This matching
   tolerance does not change frozen decoder parity, normalization floors or
   physical fall thresholds. A mismatch is invalid pairing, not a policy
   effect. Resetting from the same seed is NOT an assumption of matching.
7. Baseline may use reference during its continuation; stop at clip boundary
   rather than wrap/teleport and call it survival. Both branches stop on physical
   failure, nonfinite state/action, reset/teleport, unexpected termination,
   missing terminal evidence, time limit, or declared step budget. No PPO rescue.
8. Reset clears history/age/previous-action state. Ticks must be integral,
   contiguous, and in the same segment. Handover is the only allowed ownership
   change without clearing the eight VAE pairs. Never retry a failed branch to
   improve its score.

## Measurement and reporting

Report per branch/pair, not just an aggregate: attempted/completed ticks, simulation
duration, stop reason and evidence availability, tilt trajectory/peak, pelvis
height trajectory/minimum, action norms and derivatives **per second**, latent
raw and normalized norms, finite checks, encoder/decoder/denoiser counts, and
measured-state/conditioning provenance. Action/torque saturation and contact
metrics are reported only if captured with known semantics; otherwise unavailable.

Retain the established >70-degree physical-fall criterion. Height/contact are
**descriptive**, not new acceptance thresholds. A low-tilt collapsed pose is a
known limitation; do not market tilt-only survival as successful behavior.
Reference error is descriptive for the VAE baseline, not an unguided-diffusion
failure criterion.

Expose the generated row-9 state from the SAME sampler invocation (minimal
additive `ContinuationStep.predicted_next_hybrid_state` field allowed). Compare
it to the next measured state rebased to the row-8 pre-action center with
`world_to_hybrid` on `[s_t,s_(t+1)]`, center_index=0. Separate root position,
rot6d, linear/angular velocity and body-relative position/velocity errors; no
mixed-units 'physical MSE', no second sampler pass, no using a post-action frame
as planning context. This tests one-step prediction consistency, not all 32
future rows or distributional feasibility.

Write atomic summary JSON and bounded per-tick NPZ/JSONL under a new
`logs/diagnostics/diffusion-d3-closed-loop/<run-id>/`; refuse existing run outputs.
Record source HEAD/diff fingerprints, frozen hashes, selected checkpoint step
versus total training budget, environment/control/observation semantics, seeds,
frames, all attempts and partial outcomes. Label real `mjlab` evidence separately
from fake/stub tests; `closed_loop=true` only for physically stepped mjlab trials.
Model/normalizer/file integrity is checked before/after; preserve non-Tensor
`state_dict` extra metadata too. Never alter frozen artifacts.

## Resource and acceptance gates

**Software gate (workers/integrator):** CPU fake-runtime tests for 8-pair startup,
pre-action ownership under reused buffers, no encoder/reference after handover,
matching failure, reset/teleport/terminal precedence, frame-aligned predicted
state metric, partial-report behavior, non-operational CLI and budget/hash/schema
rejection. Existing D3 tests remain passing. Scoped Ruff format/check, Ty and
`git diff --check`; relevant full diffusion regression after fixture staging.
Do not launch physics from child workers.

**Initial physics smoke (parent only after software + independent review):**
one local GPU, one environment, headless, member `tennis_000`, reset seed 0,
start frame floor(0.1*(N-1)), exactly 8 bootstrap + at most 25 evaluation ticks
per branch (2 branches, at most 66 transitions / 1.32 simulated seconds).
Verify GPU availability immediately before launch. Use monitored execution with
visible progress and exact-owned cleanup; compilation may require up to 20 min
wall time, no automatic repeat/timeout extension, output cap 128 MiB. Failure/fall
is a valid recorded result, NOT permission to tune or enlarge the run.

**Subsequent behavior pilot (not automatically launched):** after smoke tool
acceptance, preregister one-motion matched trials for seeds 0/1/2, two starting
phases and up to 5 s per branch, with explicit total attempts/time/output budget.
Set baseline-derived behavior tolerances before seeing diffusion outcomes.
Report all failures and uncertainty. Qualified multi-motion evaluation of the
original 3-member cohort follows separately; no standing or out-of-cohort claim.
A smoke pair does not close D3. Survival/physical metrics and matched comparisons
remain required; D5 timing/export and D6 hardware remain separate.

## Implementation topology / lane board

This is **multi-seam**: simulator ownership and simulator-independent evaluation
are independently testable contracts. Avoid an issue-wide single writer.

| Lane | Exclusive scope | Isolation | Gate / durable handoff |
|---|---|---|---|
| Parent contract | `closed_loop_types.py` seed + this plan | seed in all 3 worktrees | fixed snapshots/protocol; workers ask before schema change |
| Runtime worker | `closed_loop_runtime_x2.py`, `tests/test_tracking_diffusion_closed_loop_runtime.py` | `runtime/` | fake env/cache/boundary tests; scoped source patch + report |
| Evaluation worker | `closed_loop.py`, additive row-9 output in `handover.py`, `tests/test_tracking_diffusion_closed_loop.py` | `evaluator/` | fake-runtime/model tests, unchanged sampler behavior; patch + report |
| Integration worker | accepted component copying + CLI/README `scripts/experiments/diffusion_d3_closed_loop/`, seam doc/changelog | `integration/` | CLI, full regression, hashes/report; no redesign/physics |
| Reviewer | exact integrated diff/contracts/evidence, read-only | `integration/` | concrete P0/P1/P2 findings and merge verdict; parent disposition |

All children: `lz-fast/gpt-5.6-luna:max`, fresh context, no nested delegation,
no commits/pushes, no primary edits, no model/physics/default-config changes.
This restriction applied to the original implementation request; the later
scoped commit authorization does not authorize any of those other operations.
Focus: inspect the cited seam, implement the smallest direct solution, run the
focused tests, and stop. Do not reopen sampler math, research reproduction,
optimize speed, build generic frameworks, or repeatedly polish low-risk details.
One concrete missing prerequisite triggers a supervisor question, not broad
exploration. Initial component deadlines 90 min; integration 60 min; review 45
min, with progress/checkpoint requests before a deadline. No hard tool budget on
writers. Commands: inspection 30 s; one test 60 s; focused suite 120 s. Unexpected
timeout => isolate/diagnose; truly long jobs belong to parent MonitorCreate.

Parent observes the exact workflow every **30 minutes**, maximum 16 fires/8 h.
Do not stop merely because a check is unchanged or one child is finished. Stop
when review and parent disposition are complete, explicit cancellation, or the
bounded observation window expires. Preserve partial work on blockers.

## Current status

**Current disposition (2026-10-04): D3 implementation and GUI prototype done;
formal D3 behavioral acceptance still open.** The operator manually inspected
the GUI and accepted it as "not perfect, but usable"; no further GUI polish.
See [GUI handoff and manual acceptance](../../logs/diagnostics/diffusion-d3-gui-implementation-20261004/REPORT.md).

**Recorded pilot COMPLETE (run2026-10-04UTC; disposition2026-10-05+08):** user selected
**1e-3 bootstrap/initialization matching**, citing usual teacher-reference error
and acceptable manual GUI behavior. Fresh DeepSeekFlashHIGH workflow `82bfe9d9`
completed the source/regression change, one parent-monitored serial batch and
analysis; candidate remains at `worktrees/mjlab/d3-baseline-20261004T155333Z/implementation/`.
[Bounded brief/current routing](../../../worktrees/mjlab/d3-baseline-20261004T155333Z/orchestration/launch.json).

Preregistered: `tennis_000`, seeds0/1/2, startfractions0.1/0.5 (frames45/226),
8bootstrap+up to250evaluationsteps/branch. Use deterministic boundary-safe
`min(250,N-2-start_frame-8)` horizons (250/217), both branches share horizon.
Sixpairedattempts maximum, one localGPU/environment sequentially, <=2898
transitions/57.96simulatedseconds, **global1200s wall /128MiB output**. No retries,
tuning, extra members/seeds, GUIserver/video or expanded pilot. Worker prepares
CPUgates/driver, asks parent for tracked Monitor execution, then analyzes those
recordings; no untracked long child commands. Source/normalizer/file integrity,
paired trajectories, noencoderafterhandover and same-sample predicted-realized
metrics remain mandatory. This is a descriptive recordedpilot, not an invented
formalqualitypass; broader qualified three-motion behavior acceptance remains
open. Source publication is separate; no commit/push/controller/hardware work.
The later user authorization permits only the scoped implementation/docs commit;
it does not authorize push, controller or hardware work.
The earlier failed1e-6 attempts below remain historical and are not relabeled.

**Final pilot disposition (Monitor#3):**
all6conditions completed and matched at1e-3, fullrawtickcoverage and model/input/
source/76fixture integrity verified. VAEbaseline reached every declared horizon;
diffusion crossed the existing >70deg fall criterion in5conditions, and only
seed1/frame226 reached its budget. Driver103.73swall (monitor~2minincludingstartup),
raw+analysis70,508,498B (<128MiB), no retry/tuning. Actual fall stamps including
0.16sbootstrap are3.50/2.14/3.66/3.56/2.42s; actualphysical2438transitions/48.76s.
Rawsummaryduration is safe-prefix time and excludes final20ms for falls. Parent
independently reproduced allaggregates,941prediction samples, encoder8only/no
posthandoverreference, post-step times and exactbefore/after snapshots.
[Durable final report, plots, raw data and parent checks](../../logs/diagnostics/diffusion-d3-recorded-comparison-20261004/REPORT.md).

**D3 implementation COMPLETE by user acceptance (2026-10-05 +08:00); model-quality
polishing deferred.** The recorded stability gap remains evidence rather than a
blocker to implementing D4. Next implementation phase: sparse keyframe inpainting
and physical-cost guidance, as specified in the reproduction plan §5.3. Root-cause
investigation and broader multi-motion quality comparisons are deferred with model
polishing; no claim is made that guidance will repair the falls. GUI remains done.
Progress26deleted and comparisonmission closed after finalpersistence; no activeGPUjobs
or extra experiment authority.

Runtime/evaluator components completed on `gpt-5.6-luna` / **max** thinking.
Their scoped archive/source hashes match; parent repeated **6 runtime tests** and
**40 evaluator + existing D3 tests**, all passing. Original workflow `dbc48c6f`
failed only on optional `undefined` metadata while persisting component results;
no integration/review had launched. Same-protocol continuation
`2e8f23ca-a1b6-4800-8874-27486eb85e07` completed integration (**51 focused
software tests**, integrator evidence) and independent review. Review verdict
**BLOCK**: 4 P1 reporting/terminal-boundary/provenance defects and 3 small P2
budget/provenance/fingerprint defects accepted after parent source inspection
and CPU repros (`orchestration/parent-review-repros.json`). Retained original
owners completed exclusive runtime/evaluator fixes and CLI reintegration in
`dfd67617`: **68 focused / 219 diffusion tests** (integrator evidence). Retained
review recheck resolved R1/R2/R4–R7, but R3 still failed when its minimum 4 KiB
marker copied full real-model integrity metadata. Parent made a small scoped
marker-helper/test fix with fail-before evidence; independently **70 focused /
220 diffusion tests passed, 1 GPU-device test skipped**, Ruff/Ty/Pyright/diff
checks pass. All **76** independent fixtures, sources and primary originals
remain hash-identical. CPU-loaded actual frozen D2/VAE integrity metadata was
**17548 bytes**; its final marker is now **1408 bytes**, retaining status and
before/after digests within the 4096-byte cap. This probe used a fake rollout,
`synthetic=true`, `closed_loop=false`, not physics. Final retained narrow review `92f5020a` passed **OK with notes**, R1–R7 resolved.
The first authorized Monitor **#1** smoke exited **1 before environment
construction** on the known zero-width `motion_lookahead` slot; this was not
physical evidence. Parent added only the explicit empty-slot compatibility fix;
**74 focused / 224 diffusion tests passed, 1 GPU skip**, scoped checks pass, and
compatibility review `0879e00e` returned **OK with notes**.

After direct user approval, exactly one fresh bounded smoke ran as Monitor **#2**:
`tennis_000`, seed0, frame45, one CUDA environment per branch, 8 bootstrap plus
25 evaluation budget, max66 transitions/1.32 simulated seconds, 1200-second wall
cap and 128-MiB output cap. The real `mjlab` factory constructed successfully
and both branches recorded real simulator measurements. The baseline completed
33 ticks and observed bounded survival (peak tilt 22.998 degrees, minimum pelvis
height 0.545 m). The diffusion branch completed only its 8 bootstrap ticks
(8 encoder/8 decoder/0 denoiser calls) and stopped at `pair_mismatch` before
reference-free handover. The pair gate failed at tolerance 1e-6 despite equal
initial signatures: bootstrap state max difference 2.38e-6, conditioning
4.53e-6, action/latent 2.15e-6, bootstrap-end state 1.55e-6 and conditioning
3.81e-6. Overall result is **partial / invalid / unpaired**; no diffusion
continuation or D3 behavioral acceptance was obtained. Output is complete at
1.09 MiB, within cap; frozen inputs and all 76 fixture/source/original hashes
remain unchanged. [Durable report, raw smoke output and review evidence](../../logs/diagnostics/diffusion-d3-closed-loop-implementation-20261004/REPORT.md)
are in primary diagnostics. The source candidate was initially isolated/uncommitted;
the later user-authorized scoped commit is the publication handoff, with no push.

Workflow **#22**, revision 3, is now **completed** after its one approved
retry attempt was honestly disposed as partial/invalid/unpaired. Progress
monitor **#23** was deleted after the declared completion condition was met;
no simulator/jobs are active. No x2_docker/remote/hardware/export/deployment or
primary source landing/commit/push occurred. The next technical question is
reproducible matched bootstrap initialization within the predeclared tolerance;
no automatic rerun is authorized.
