# D4a — keyframe inpainting and pose-cost guidance implementation

Authorization: 2026-10-05 +08:00 (2026-10-04 UTC). User accepts D3 implementation
as done, defers model polishing, and requests this plan plus delegated
implementation. D3's five recorded diffusion falls remain negative quality
evidence, not a reason to reopen D3 or an assertion that guidance will fix them.

## Deliverable and boundaries

Implement two opt-in steering mechanisms on the active frozen 41×167 trajectory:
1. **Full modeled-state keyframe inpainting** at sparse future ticks.
2. **Physical pose-cost guidance**, differentiating clean predictions through
   the frozen denoiser to the current noisy trajectory.

Wire both into the existing private mjlab headless evaluation seam, with a
strict task-config format, bounded diagnostics, and CPU/fake-runtime proof of
executed latent/action effects. Preserve the default unguided path. Supply
instructions/config preparation for future separately authorized physical trials.
GUI usability is done: no GUI polish or required GUI extension in this slice.

No training, new teachers/datasets, normalization fitting, new learned objective,
mixture/root-emphasis features, alternative sampler, inference optimization,
controller/x2_docker/export/deployment/hardware, remote jobs, new physics batch.
The user later authorized the scoped implementation/docs commit after acceptance;
push and physical execution remain unauthorized. CPU analytic tests, recorded-data
processing and frozen-asset static preflight are authorized implementation gates.
Slow frozen-model numerical checks must be parent-monitored, never detached by
children. Navigation/velocity/obstacle costs and transition-quality campaigns are
later D4 slices, not scope creep here.

## Source and isolation

Primary mjlab is dirty and HEAD has advanced to 5c62aeed; preserve unrelated VAE
work. Pin this candidate to the actually tested D3 base
`5f17fc6f9b9a28e51baa44425b50106ba0561e86` plus the **16 hash-pinned executed D3
comparison files**, including the user's1e-3 matching and parent pilot fixes.
Do not substitute the historical GUI's1e-6 headless candidate.

Root: `/home/agiuser/projects/worktrees/mjlab/d4-control-20261004T171154Z/`.
Three detached worktrees: `inpainting/`, `guidance/`, `integration/`.
Each has explicit .venv link and76 independently copied, hash-matched ignored
fixtures; bootstrap evidence under `orchestration/`. Models/projection/contract
and original fixtures remain immutable. Primary edits are this plan/status only.

Frozen assets (same D3 provenance):
- Active YAML `docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml`,
  SHA f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed.
- D2 `../d3-focused-20261003T185144Z/assets/checkpoint-best.pt`,
  SHA252fb3a8ddd8da98f8cd3c8890328bec38aa5107767b13a576e5db8a43b956ae.
- Projection sibling `projection.npz`,
  SHAe258cf27bdfd7e9e3d81762287f1afd992738e0c5f6c81222e8c1b2c065217fe.
- Frozen primary VAE `logs/distillation/mixed-10k/checkpoint-final.pt`,
  SHA69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117.
- Original cohort `configs/distillation/x2_tennis_mixed.yaml`,
  SHA587550f26345a278f14c350404be11843dca379d708c2a3c418f0aa8340f9b2d.

## A. Sparse inpainting contract

- `TimedStateKeyframe`: an **absolute control tick** (same clock as D3, including
  eight bootstrap ticks), a complete unbatched six-array WorldState, and explicit
  source provenance. Own copies, finite values, valid nonzero quaternion, exact
  body order and contract identity at serialization boundary. It is an explicit
  task input, not a fabricated executed history row.
- `inpaint_keyframes(plan, current_state, keyframes, *, projection, contract)`:
  preserve every existing fixed entry and tensor ownership. For active future
  targets with `1 <= target_tick-plan.tick <=32`, map row to
  `current_index + offset`. Rebase each target against **current measured root
  position/yaw/velocity**, not the target's own frame; normalize with frozen stats.
  Only the complete135-state slice becomes clean/fixed. Its state timestep ID is0;
  its32-latent slice remains unknown at the reverse source ID.
- Default and task-input minimum spacing is10ticks/0.2s between scheduled
  hardkeyframes (and between sparse posegoals). Absolute world targets and ticks persist
  across replanning; rebase every tick without shifting an absolute goal forward.
  Past/reached and beyond-window targets are inactive, not replacements for
  measured current/past states. Reject negative/non-integral/bool ticks, duplicate
  target ticks, inconsistent shapes/nonfinite/legacy schemas; report active IDs.
- No dense-reference mode, future encoder latent, partial-state hard mask or
  explicit learned mask embedding. Partial pose objectives belong in soft cost.

Focused proof: target/current asymmetric translations/yaws/velocities; owned
arrays/tensors; full-state clean IDs/unknown latent IDs at **every denoiser call**;
bitwise fixed values after everyjump; none/empty equivalence to D3; rolling
window boundaries, expired/far-future targets, duplicate/NaN/wrong-layout failures.

## B. Physical pose cost and reverse guidance

New modules `pose_costs.py` and `guidance.py`, no mutation of D2 sampler defaults
or rejection behavior. Public opt-in entry `sample_trajectory_guided`; reuse
existing DDIM arithmetic, grid, Gaussian RNG and conditioning helpers.

### Pose targets and costs

Use `TimedPoseGoal` with absolute tick/owned WorldState target, plus explicit
finite nonnegative `PoseCostWeights`: rootposition, rootorientation and
bodyposition costs (optional20 named/per-body coefficients for partial poses).
Root/body position costs are squared meters; orientation uses a documented
sign-invariant SO(3) chordal rotation-matrix cost, dimensionless, not radians.
Weights/scales must be explicit in task provenance; no invisible objective
changes. No velocity/contact/torque/smoothing cost added in this slice.

Torch-only physical recovery from x0's first135 **normalized** channels:
- denormalize with frozen state mean/std;
- restore world root position using currentcenter position/yaw;
- reconstruct proper root orientation from predicted rot6d, then world orientation;
- reconstruct each body world position from predicted root world position and
  **that predicted timestep's root yaw**, plus the local body position features.

Never use `hybrid_to_world`'s stored future root_yaws as an oracle for generated
states; that inverse helper assumes a known recorded trajectory. No NumPy or
`.detach()` on the differentiable physical recovery path. Context constants may
be copied from measured state before sampling. Reject nonfinite/degenerate
rotation/cost/gradient cases truthfully; document numerical geometry guards
separately from immutable training normalization floors.

`G(x0) = 0.02 * sum(active_future_pose_costs)` **pertrajectory**, including actual
weights. Hard-inpainted points are fixed; do not sell their trivially zero predicted
cost as a guidance improvement. Distinguish hardkeyframes and softposegoals.

### Frozen engineering update

For each of20 reversejumps from sourcek (no call atclean0):
1. Make detached current x_k a grad-enabled leaf; call frozen evaldenoiser once.
2. Reapply clean conditions to x0 before physicalcost evaluation.
3. Compute input VJP `g = dG/dx_k`, not `dG/dx0` alone.
4. Zero fixed-entry gradients **before** pertrajectory L2 clip to max1.0 (do not
   normalize all nonzero gradients to unitlength).
5. `x0_guided = x0_hat - lambda * (1-alpha_bar[k]) * clipped_g`.
6. Recompute epsilon from x0_guided using existing ddim_jump, then restore exact
   clean values. Detach before next jump to bound graph lifetime.

No parameter/normalizer/optimizer/RNG side effects, no decoder backward,
no extra denoiser pass or embedded unguided comparison. GuidanceConfig's strength
is finite nonnegative and clip is the declared1.0. Existing SamplerConfig still
rejects nonzero legacyguidance. λ0 must delegate exactly to original sampler,
without cost/VJP calls. Handle no active softgoals explicitly. Enable inputgrad
under torch.no_grad; reject inference_mode contexts that cannot support VJP.
Diagnostics expose actual forward/VJP/cost counts and bounded perstepcost/norms,
sourceIDs, activegoals/fixedmask, precision/backend and invalidity—not timing
qualification. Positive demo strengths are named illustrative configs, not
accepted quality-tuned production parameters.

Focused proof: FP64 finite differences with coupled state–latent oracle; sign,
(1-a), dt, weights and clip scaling; fixed-gradientzero-before-clip; same-noise λ0
bitwise equivalence; frozenparameters/grads/stats/RNG unchanged; exactly20 forwards;
same-noise λpositive alters currentlatent and decodedaction via a coupled oracle,
not only predictedfuturestate. Worldbodyyaw uses predicted orientation with
nonzero centertranslation/yaw; quaternion sign and rotated-frame invariance.

## C. Integration and opt-in task surface

Integration-only owner consumes the two durable component handoffs. Introduce
`conditioned_continuation.py` and the smallest hook/refactor in handover.py to
reuse history, current measuredconditioning, latent denormalization/decoder,
stale/reset guards and SAME sampled row9 metrics. Do not duplicate the full D3
runtime/evaluator or replace predictions with keyframe truth in measured history.

Extend the existing headless `evaluate.py` with opt-in `--task-config PATH` (or
an equally explicit task flag) and a small strict taskloader/preparation tool
under the same experiment directory. Default no taskconfig retains the original
unguided behavior/provenance. Mode labels: unguided / keyframe / poseguided /
combined. Add no auto-executing launcher or retrybatch.

Task JSON is a separately versioned, bounded input (max1MiB/64scheduledtargets),
not a revision of the frozen learned contract. Bind contractidentity, bodyorder,
controlperiod, frame=world, absolute tick origin, hardkeyframes vs softposegoals,
weights/strength and source hashes. No arbitrary dynamic imports/callables/pickle.
Provide preparation from existing qualified baseline recordings (explicit selected
sparse ticks), retaining inputhash/member/seed/physicalprovenance; don't silently
read a reference each controltick. Allow synthetic testfixtures clearly labeled,
never claim qualified physical provenance from fabricated data.

Preflight rejects malformed taskconfig beforeenvironment creation; help/preflight
remain non-operational. Goal expiry/horizon selection and reset lifecycle explicit.
Record activegoalIDs, guide diagnostics, predictedposecost and **realized** sparse
poseerrors from trusted measuredobservations at targetticks; missing/unreached
points reported as such, not zero/errorfree/censored-success. Keep physical fall,
terminal-priority, wall/output, pairing1e-3, source/hash/integrity and fake-vs-real
claims intact. Default VAEbaseline remains unchanged; positiveguidance matched
noise comparison is a future declaredtrial, not hidden extrabranches now.

Provenance includes actual new imported sourcefiles and full taskfileSHA. Compact/
minimum-cap markers remain bounded and truthful; do not serialize entire goals,
modelmetadata, trajectories or unbounded perstep VJP tensors into markers/stdout.

## Lane board / sequencing

Classification: **multi-seam**. Inpainting clean-mask/frame contract and physical
cost/input-VJP contract are independently testable. Integration is a dependent
third seam; independent final review is not an implementation owner.

| Lane | Exclusive files | Gate / handoff | Independence |
|---|---|---|---|
| inpainting | inpainting.py; test_tracking_diffusion_inpainting.py | Owned synthetic CPU tests + durable filehash/patch list | Full-state target framing/masks; no gradient sampler ownership |
| guidance | guidance.py, pose_costs.py; their two dedicated testfiles | Analytic FP64/VJP/physicalframe CPU tests + durable handoff | Cost/gradient mathematics; no inpainting/handover/CLI edits |
| integration | conditioned_continuation.py, narrow handover.py/closed_loop.py/evaluate.py hooks; taskloader/preparation/examples; dedicatedintegrationtests; README/changelog | Consume exact component files, sequential focused+fulldiffusion regressions, frozenCPUpreflight, scopedchecks | Wire finished cores, not rewrite algorithms |
| reviewer | Integrated candidate read-only | Evidence-backed P0/P1/P2 and OK/BLOCK verdict | Freshcontext, independent masks/gradient/execution/report audit |
| parent | Primaryplan/NOW/routing, sourcepin, test reruns/disposition/archive | Final acceptance | No concurrent edits to a live writerworktree |

One workflow, **fresh ds-oc/deepseek-flash:high** for allchildren, no nested
delegation. Serialize component stages and tests to preserve CPU-budget ownership;
integrator begins only after both successful handoffs. One writer perworktree.
Budget each component30min, integration45min, review20min; on nearing deadline
checkpoint files/tests/remainingwork, stop after active tool, not midmutation.
No automatic repeated review/polishloop; parent dispositions findings then scopes
retained owner fixes. Infrastructure/model/tool failure stops governed workflow;
never substitute CLI/foreground/model without operator approval.

Commands: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=$PWD/src uv run --no-sync ...`. Inspection30s, onecase60s,
focusedsuite120s, failfastverbose and faulthandler_timeout30. Fullsuite sequential,
120s first; onunexpectedtimeout diagnose smallercase, don't repeat/extend blindly.
Direct boundedlogs preserve commandexit; no pipe-to-tail success masking. Truly
long frozen-model work asks parent for tracked Monitor. No broad process kills.

## Acceptance and next handoff

Implementation accepted when analytic/synthetic/CLI/boundedreport regressions,
staticfrozenpreflight, independentreview and parentinspection pass; all original
frozen inputs and baseline source/history stayunchanged except explicitly owned
D4hook files. Preserve D3 zero-guidance trajectory/action parity and all existing
GUI/headless guards. A model unable to improve a task is truthful quality evidence,
not grounds to silently tune/retrain or mark guidance implemented without a VJP.

Final artifact: isolated integratedsource snapshot/hashmanifest, component reports,
review/disposition, testlogs, taskconfig example/preparation instructions and
limitations. No physical-success/realtime/deployment claim. The next decision is
an explicitly budgeted keyframe/poseguided mjlab trial; it is NOT launched here.

Operational status: **D4a implementation accepted (2026-10-05 +08:00)**.
[Durable final report/source/reviews/parent gates](../../logs/diagnostics/diffusion-d4-keyframes-pose-guidance-20261004/REPORT.md).
Parent222focused /402diffusiontests pass,1GPUskip; scopedRuff/Ty/Pyright/diff
clean,76fixturecopies/sources and mathcores unchanged. R1–R7 plus postobserver
edge findings fixed; final retainedreview **OK, no P0/P1**. Monitor5 realFP32
frozen-model numericalprobe passes (80forwards/20VJPs,0physics/encoder); hard
keyframes affect decodedactions, softλ0.1 effect small/notqualityproof.
Monitor8 exact29-file finalCLIpreflight passes, allfrozen/task/source hashes
match,0runtime/inference/physicscalls. Failedshell/legacy-metadata attempts
remain historical; no physicalretry or tuning.
The user later authorized a scoped implementation/docs commit after the parent
acceptance gates passed. Push, deployment and physical execution remain
unauthorized.
Task configuration is opt-in/headless; accepted GUI unchanged. Next boundary is
an explicitly authorized/budgeted guidedphysicalrehearsal, NOTlaunched here.
Velocity/navigation/obstacle/transition-quality and D5/D6 remain separate.
Mission`ca02f7e5-db92-4261-98b9-3aac9aaf2a8d` closed as completed after persistence;
progress28 reached configuredfirecap, no loops remain configured.
[Final routing](../../../worktrees/mjlab/d4-control-20261004T171154Z/orchestration/launch.json).
