# Preview-aware VAE to latent flow matching: implementation plan

Date: 2026-10-20 (+08).
Status: local port, final-VAE data-path smoke, fresh statistics/caches, and bounded flow-training smoke are accepted.
Original launcher/audit failures remain preserved; see sections 12 and 13.
Full collection and long training remain unauthorized.
Model for every delegated stage: `ds-oc/deepseek-flash:max`.
Source baseline: `948ae52c177bcb6800bf25581a9ca51f5065aafe` in `mjlab`.

## Research scope: operator clarification

This is a research attempt, not a production-hardening project.
Keep the implementation small. Reuse existing interfaces; add no generic framework, registry, or configuration generator.
Add tests only for critical changed behavior:

1. Correct v1/v2 schema routing and rejection of incompatible model/reference pairs.
2. Correct preview frame alignment, including clip-end and subset-reset boundaries where the port changes them.
3. Reference-free handover and unchanged state/latent token shape.

Reuse existing tests for unchanged behavior. Do not duplicate coverage across collection, startup, and integration stages.
One small integration check is sufficient if the component tests do not already cover the interface.
Skip exhaustive malformed-input matrices, broad hardening, full-suite runs, and cosmetic refactors.
The checks below describe critical behavior, not a requirement to add one test per bullet or per call path.
Prefer concise preparation instructions over new configuration templates.
Reviewers should report concrete defects in scope, not impose production-readiness requirements.

## 1. Goal

Make the existing state–latent flow pipeline accept the multi-motion preview-aware VAE.
Keep the flow model, training objective, and solver unchanged.
Support collection and reference-based startup, followed by reference-free continuation.
Preserve the old no-preview path.

The implementation ends at software validation and a reviewed local source patch.
Freezing the final VAE artifact and launching experiments remain separate operator actions.

## 2. Fixed interfaces

| Item | Existing no-preview VAE | Preview-aware VAE |
| --- | ---: | ---: |
| Encoder reference | 68 | 130 |
| Decoder conditioning | 99 | 99 |
| Latent | 32 | 32 |
| Action | 31 | 31 |
| Reference preview | none | q/dq at t+0.2 s, 62 values |

- Derive reference layout from the loaded VAE schema and the cohort's declared preview.
- Use `distillation.adapter.default_schema_for_cohort` where the cohort defines the live layout.
- Require agreement between the live reference and the selected VAE.
- Do not replace all constants `68` with `130` or accept arbitrary reference widths.
- Preserve field order, gravity decoder mode, joint order, normalizers, and action semantics.
- Preserve legacy constructors/default behavior unless an explicit schema is supplied or safely inferred and checked.
- Keep 135 physical-state channels plus 32 latent channels: 167 per token.
- Keep eight past rows, one current row, and 32 future rows at 50 Hz.
- The existing 129-state alternative must not regress; do not select it for this experiment.
- Keep the flow Transformer, x0 loss, linear process, and Euler formulas unchanged.
- Keep startup reference-based. After handover, generated latents replace the encoder; decode with fresh observations.
- Preview must remain clip-local, including end clamping and subset resets.

## 3. Authority and exclusions

This section records the original implementation authority. Section 11 adds the later bounded operational authorization.

Allowed: isolated local source edits, focused CPU tests, documentation, and parent integration.
Forbidden: remote access, artifact downloads, simulator or GPU jobs, full collection/training, deployment, and robot operations.
Do not touch `/home/fushan/mjlab-distill-20261015` or any running job, launcher, monitor, or queue.
Do not change x2_docker, goal guidance, recovery policy behavior, qualification thresholds, sampling budgets, or VAE training.
Do not select a final checkpoint, invent hashes, or reuse old VAE latent statistics/caches.
Do not commit or push. Staging is allowed only in owned isolated worktrees to make complete patches.
Children must focus only on their assigned scope. Stop and ask before crossing file ownership.
No nested delegation.

## 4. Topology and ownership

Classification: **multi-seam**.
Collection and closed-loop startup have independently testable source boundaries.
Run two component writers in separate worktrees. Then run one integration-only writer and one fresh read-only reviewer.
The parent owns final acceptance and transfer into the primary checkout.

Worktree prefix: `/home/agiuser/projects/.worktrees/mjlab-preview-flow-20261020-1800`.
All worktrees start at the source baseline. Existing primary documentation edits remain untouched.
The source `.venv` is shared read-only through `UV_PROJECT_ENVIRONMENT`; `PYTHONPATH` selects the candidate source.
Any shared ignored inputs are read-only. Test output and caches stay private.

| Stage | Decision / independent scope | Exclusive file ownership | Worktree suffix | Gate / handoff |
| --- | --- | --- | --- | --- |
| A: collection | Correct reference schema reaches frozen VAE in scalar, batched, and tensor collection | `diffusion/adapter.py`, `diffusion/runtime_x2.py`; collection adapter/runtime tests only | `-collection` | Focused v1/v2 tests; staged source patch and managed report |
| B: closed-loop | Load preview VAE, construct startup reference, then stop reference consumption after handover | `diffusion/replay_integration.py`, `closed_loop.py`, `closed_loop_types.py`, `closed_loop_runtime_x2.py`, `parallel_closed_loop_x2.py`, `visual_playback.py`; corresponding closed-loop/replay/visual tests only | `-startup` | Focused v1/v2 load/startup/handover tests; staged source patch and managed report |
| C: integration | Combine A/B interfaces; document artifact preparation without selecting a checkpoint | Apply A/B patches; seam-only corrections in those files, one focused integration test, runbooks, new non-executable config templates | `-candidate` | Combined CPU gates; unchanged flow core; managed integrated report |
| D: review | Check implemented diff for concrete defects and scope violations | Read-only candidate | `-candidate` | Evidence-backed P0/P1/P2 findings with paths; parent decides disposition |

Stages A and B must not edit the other's files or shared existing test fixtures.
Add lane-specific fixtures/tests if needed. Stage C may reconcile only interface incompatibilities or test integration.
It must not take over unfinished component features. Return those to the component owner.
Reports use subagent-managed paths, not ad-hoc files in the repository root.

## 5. Stage A: collection

Inspect `FrozenVaePolicy`, observation dataclasses, scalar observation extraction, batched snapshot construction, and tensor inference.
Accept the supported gravity v1/v2 layouts and check observation width against the actual model schema.
Preserve scalar/batch/tensor action parity and the existing independent decoder conditioning.
Use the cohort preview declaration in the live snapshot schema.
Do not change latent convention: collect deterministic encoder mean actually passed to the decoder.
Preserve pre-action state/latent alignment, clean versus executed actions, and existing segment eligibility.
Ensure rollout selection can use all manifest members without assuming contiguous `single_*` numbers.
Use the existing manifest override/selection mechanisms; do not invent a second cohort registry.

Required checks:
- Legacy 68/99 paths still work.
- Preview 130/99 paths reach the encoder intact and emit 32-dimensional latents and 31 actions.
- A 68-wide observation paired with a preview VAE is rejected, and vice versa.
- Normal, clip-end, and subset-reset reference packing retains the correct same-clip preview.
- Collection metadata exposes enough schema/preview information to audit the selected path without changing old identities unnecessarily.

## 6. Stage B: loading and startup

Remove the unconditional 68-width rejection in the shared inference loader, not the schema checks themselves.
Make startup observations, scalar/batched references, and viewer validation agree with the selected schema.
Allow the known 62-wide `motion_lookahead` actor term for a declared preview cohort.
Construct preview from the same reference frame and live orientation error used by the VAE path.
Keep zero-preview layout and startup results unchanged.
Keep the post-handover runtime reference-free; never use predicted states as decoder conditioning.
Preserve current solver-step defaults, state dimensions, flow/diffusion family separation, and rejection of flow task guidance.

Required checks:
- Valid gravity v1 and v2 VAEs load; malformed layouts or incompatible observations fail clearly.
- Scalar and batched startup produce the expected reference width and order.
- Clip-end preview clamps rather than wrapping into another clip.
- After handover, reference access is unnecessary; use a fake source that raises if read.
- Existing flow continuation tests still use the unchanged sampler.

## 7. Stage C: integration and experiment preparation

Read both component reports before applying their staged patches into the candidate worktree.
Confirm patches contain only their claimed files and no shared inputs or outputs.
Resolve only interface mismatches in the implemented seams.
Run a minimal CPU path from preview reference through encoder/decoder and unchanged state–latent token packing.
Use a tiny synthetic VAE where real artifacts are unavailable; label the evidence accurately.

Provide a clearly non-executable contract/config template, or concise preparation instructions if templates duplicate existing configuration.
The instructions must name the existing 144-member manifest `configs/distillation/x2_tennis_multimotion_v1.yaml`.
Use the full manifest for VAE identity and the existing selection mechanism for any pilot subset.
Explain the later required inputs: chosen checkpoint, actual digest, manifest digest, and fresh output/statistics/cache paths.
Do not pin an old checkpoint or insert a plausible-looking digest.
Explain train-only statistics, group split before windowing, and no reuse of old learned artifacts.
Keep collection budgets, full-run training budget, and checkpoint choice unset until authorized.
Update the flow runbook and the Upcoming changelog. Record tested and untested status separately.

## 8. Validation policy

Use `CUDA_VISIBLE_DEVICES='' UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv PYTHONPATH="$PWD/src" uv run --no-sync ...`.
Use `pytest -x -vv -o faulthandler_timeout=30` on focused tests.
Use 30-second inspection, 60-second single-test, and 120-second focused-suite limits.
After an unexpected timeout, inspect the smallest reproducer before retrying.
Do not extend routine test timeouts or run the whole repository suite.
Children without a monitor tool must ask the parent for genuinely long validation.
Run scoped Ruff and type checks where practical; preserve command exit status.
Do not alter shared dependencies or fetch missing test artifacts.
Report missing artifacts separately from code failures.

The final parent gate checks:
- Only planned files changed; no live training tree changed.
- Legacy and preview collection/startup paths pass their focused CPU tests.
- A test proves reference independence after handover.
- State/latent token shapes and flow core source remain unchanged.
- No blocker remains in fresh review; deferred items are explicit.
- Documentation distinguishes software readiness from real-checkpoint, simulator, collection, and policy-quality validation.

## 9. Completion and handoff

Deliver the reviewed local patch, test commands/results, remaining artifact-binding steps, and exact limits.
The selected VAE checkpoint and 144-motion experiment are not validated by synthetic tests.
Do not claim tracking quality or physical readiness.
The next operational step is an explicitly authorized real-checkpoint smoke after artifact selection.

## 10. Parent acceptance and scope correction

Subagent workflow: `6ebd0c1a-a1d2-4a74-9841-84236155850b`.
Mission: `07e351e0-11e1-4da9-8cbd-e0f20dc806e2`.
All four delegated stages completed. Their patches remain isolated until parent acceptance.
The initial review reports no P0/P1 defect and four P2 notes.

The component file boundaries omitted the shared command-line preflight.
`evaluate.py` still rejected reference130 after the runtime port.
This prevents the requested preview evaluation path from working.
The parent therefore adds this narrow integration correction under the existing implementation authorization:

- Update `scripts/experiments/diffusion_d3_closed_loop/evaluate.py` to accept supported 68/130 reference widths.
- Keep conditioning99/latent32/action31 checks and flow task-guidance refusal unchanged.
- Report the loaded width in preflight and runtime provenance.
- Extend the existing flow preflight test to cover both widths, without adding a new harness.

P2 disposition:

- Correct the runbook: startup derives the layout from cohort preview; it does not call `default_schema_for_cohort` directly.
- Replace the helper-only schema test with a mocked-environment test of both real collection constructors.
- Replace duplicate batched shape/helper checks with one real batched-runtime constructor/startup/handover test using a fake environment.
- Remove the fake clip-end test with the overstated name. Keep the real motion-command clamping test.
- Remove the extra malformed-layout matrix. Keep model/reference mismatch tests and existing legacy checks.

The first parent check passes: **33 tests**, including the corrected public preflight.
No source under `flow_matching/` changes. No real artifact or simulator validation is implied.

### Final acceptance, 2026-10-20 (+08)

- Parent gate: **181 passed, 2 skipped, 2 deselected**, 10.77 seconds; monitor #4 exits 0.
- The two skipped tests require CUDA. This gate deliberately hides CUDA devices.
- The two deselected tests require ignored historical teacher artifacts absent from the isolated worktree.
- Scoped Ruff and Ty checks pass. `git diff --check` passes.
- The first gate stopped on a literal old source digest after 175 tests passed.
  The test now verifies the actual imported file path and computes its expected hash from that file.
  The failure log remains preserved; the corrected full focused gate passes.
- Fresh review and focused recheck find no P0/P1 defects. All documentation notes are corrected.
- The patch is applied to the primary `mjlab` checkout without staging or committing.
- **16/16 transferred files** match the tested candidate SHA-256 values exactly.
- The flow package, model, loss, solver, state transform, projection, and noising source remain unchanged.
- Seventeen new preview test cases cover the critical interfaces. Existing tests supply the remaining regression coverage.
- No new framework, configuration generator, goal-control feature, or experiment is added.
- The primary design notes and canonical plan are preserved. The three isolated worktrees remain available for inspection.

Evidence under `logs/diagnostics/preview-flow-port-20261020/`:

- [Final test and static-check log](../../logs/diagnostics/preview-flow-port-20261020/parent-validation.log)
- [Preserved first-gate failure](../../logs/diagnostics/preview-flow-port-20261020/parent-validation-first.log)
- [Initial review](../../logs/diagnostics/preview-flow-port-20261020/initial-review.md)
- [Focused recheck](../../logs/diagnostics/preview-flow-port-20261020/recheck.md)
- [Primary transfer verification](../../logs/diagnostics/preview-flow-port-20261020/primary-transfer-verification.log)
- [Validated file hashes](../../logs/diagnostics/preview-flow-port-20261020/validated-source.sha256)
- [Reviewed patch](../../logs/diagnostics/preview-flow-port-20261020/reviewed-candidate.patch),
  SHA-256 `8b25fe7a8b3b5787da3a5bde7a4bec95cedc3f504f9ea925d53bfe1d8f1b9f1f`.

Limits: no real preview checkpoint load, simulator validation, collection, flow training, export, deployment, or policy-quality claim.
No remote access occurred. The live VAE training tree and queue were not touched.
Next: select and freeze the VAE artifact, then authorize a bounded real-checkpoint smoke.

## 11. Authorized bounded smoke, 2026-10-20 (+08)

The operator says: “yes, first commit then send a subagent for the smoke test”.
This accepts the preceding proposal. Commit the reviewed port before delegating the smoke.
Use one subagent. This is an operational validation task, not another implementation project.
The precommit formatter changes whitespace only in `parallel_closed_loop_x2.py`.
The critical preview, CLI, and parallel regression gate then passes **51 tests** on primary source.
Ruff formatting, lint, and Ty checks pass; log: `logs/diagnostics/preview-flow-port-20261020/precommit-validation.log`.

Preparation may run before VAE completion:

- Stage the committed source on a new NAS branch and isolated worktree.
- Keep `/home/fushan/mjlab-distill-20261015` and the dirty main NAS checkout unchanged.
- Reuse the existing server environment. Check import origin and committed source identity.
- Verify the 144-member manifest, referenced teacher assets, recovery ONNX and its saved environment metadata.
- Prepare a new non-authorizing data contract and unique smoke output paths.

Checkpoint gate:

- Source run: `/home/fushan/mjlab/logs/distillation/x2-multimotion-vae-10k-resume2-20261019/`.
- Require successful completion, final iteration 10000, and no remaining owned trainer/collector processes.
- Inspect the final artifact and training report; compute actual checkpoint and manifest hashes.
- Bind the full 144-member cohort. Do not substitute a historical or intermediate VAE checkpoint.
- Do not stop, restart, modify, or repair VAE training. If it fails, report the blocker.

Smoke budget:

- One idle GPU on the existing managed server, with eight environments.
- Eight manifest members: `single_000` through `single_007`; verify each ID exists.
- One seed, start fractions 0.1/0.5/0.8, clean/OU pairs: 48 scheduled trials.
- At most 12000 control transitions, a 15-minute execution cap, and 512 MiB of new output.
- Use the existing collector and unchanged qualification criteria.
- Keep the full cohort identity and select only rollout IDs for this pilot.
- Use the existing schedule with `MJLAB_D1_SCHEDULE=bulk` and `MJLAB_D1_BULK_SEEDS=1`; the name does not authorize bulk collection.
- Enforce the execution cap with monitored ownership and exact-process cleanup, including child processes.

Acceptance:

- Prove reference/conditioning/latent/action dimensions 130/99/32/31 and preview 0.2 seconds.
- Check finite policy outputs, actual pre-action state/decoded-mean alignment, and clean/executed action semantics.
- Inspect/reset-boundary evidence and valid 41x167 raw windows without crossing controller or episode boundaries.
- Report attempted, skipped, rejected, retained and stored counts separately, including per-motion coverage.
- Empty data or absent valid windows cannot pass the data-path smoke.
- Preserve logs, exit status, hashes and cleanup evidence. A fall is not automatically a software failure.
- No old statistics or token caches may be used. No flow training, goal guidance, robot deployment, or quality claim is authorized.

If the child lacks monitored execution tools, it must ask the parent to run the prepared command.
Use native workflow scheduling for a pending checkpoint. Never wait with an untracked sleep loop.
Stop and ask on a code defect, missing artifact, identity mismatch, new installation, or budget change.

## 12. Final-VAE smoke acceptance, 2026-10-20 (+08)

**Accepted scope:** real simulator collection and offline data validation for eight selected motions.
Application source stays at `295402f7374e772a62e74f6cef012cd614eb163f`.
No runtime source changed after that commit. No second GPU rollout ran.

The final VAE completed at iteration 10000 with `exit=0`.
Its trainer and seven collectors exited before the smoke started.
Final checkpoint:
`/home/fushan/mjlab/logs/distillation/x2-multimotion-vae-10k-resume2-20261019/train/checkpoint-final.pt`.

| Bound input | SHA-256 |
| --- | --- |
| Final VAE, 4,344,228,754 bytes | `9d069b4552ec2a25990160477a99de24c3025c39ee1221660ebaf25b54a5ee00` |
| 144-member cohort manifest | `cd597b1cf1f734562c37295233f2165aa62399544aee485dff5d47075dceaf21` |
| Smoke data contract | `558de16628fd67df12554c54dc0b6826113a8bf4fa0654b55960872c8b9e6b1a` |

### Validation evidence

- Original monitored driver: 20:53:03–20:59:43 (+08), within its 900-second cap.
- Collection: exit 0, 184.511 seconds, no timeout, sampled GPU memory peak 639 MiB.
- All 48 trials qualified under the existing five-second criteria. No trials were rejected or skipped.
- Six trials ran per motion: one seed, three start phases, and clean/perturbed pairs.
- Outcome labels: 16 VAE-only trials and 32 VAE-to-standing-recovery trials.
- Stored rows: 12000; VAE rows 7998; recovery rows 4002.
- Accepted rows: 5146; verification-only rows: 6854. Recovery rows are not latent-training data.
- Stored data: 14,080,051 bytes, or 13.4 MiB.
- Store inspection, action-arithmetic replay, and window building passed.
- Valid raw windows: **3210**, each **41x167**; train 2656, validation 224, test 330.
- Every selected motion has valid windows, ranging from 384 to 440.
- Corrected offline audit: **21/21 checks true**, no errors, exit 0 in 37.823 seconds.
- Parent verified the audit hashes, exact correction, unchanged row/window counts, and committed handoff code.

### Audit-only false failure

The original driver exits 1 at its final audit. That failed result remains preserved.
Its assertion incorrectly requires the last VAE frame to equal the first recovery frame.
The VAE row records the pre-action frame. Recovery starts at the following post-step frame.
For `single_000`, the last VAE row has frame 451 and the first recovery row has frame 452.

The corrected assertion requires an exact one-frame advance, the clip's final frame, and the boundary marker.
All 32 handoff trials meet these conditions. No tolerance or collection threshold changes.
The corrected audit uses the same saved store and writes a separate `audit-corrected.json`.
Original `audit.json`, `exit.status`, and `logs/audit.log` remain unchanged and failed.
The workflow accepts the collected data and corrected audit, not an invented successful original exit status.

### Artifacts, cleanup, and limits

Remote worktree: `/home/fushan/mjlab-preview-vae-smoke-20261020`.
Data: `logs/diffusion/preview-vae-smoke-20261020/` under that worktree.
Contract and driver logs: `logs/diagnostics/preview-vae-smoke-20261020/` under that worktree.
Local evidence:

- [Original monitor log](../../logs/diagnostics/preview-flow-port-20261020/smoke-monitor.log).
- [Final worker audit](../../logs/diagnostics/preview-flow-port-20261020/smoke-final-evidence/worker-final-audit.md).
- [Corrected audit](../../logs/diagnostics/preview-flow-port-20261020/smoke-final-evidence/audit-corrected.json).
- [Preserved original audit](../../logs/diagnostics/preview-flow-port-20261020/smoke-final-evidence/audit.json).
- [Preserved original exit status](../../logs/diagnostics/preview-flow-port-20261020/smoke-final-evidence/exit.status).

Parent cleanup check at 21:00:38 (+08), and worker recheck at 21:05 (+08), find no smoke processes.
All eight GPUs return to 1 MiB and 0% utilization.
Main NAS branch, HEAD, and workstream dirty-file status snapshots match before and after.
The transient queue lock is excluded from that status comparison. No queue operation was performed.

Limits: eight of 144 motions, one seed, bounded trials, no hardware or general tracking-quality claim.
The dynamic run proves model loading and collection with the 130/99/32/31 layout.
Preview clamping and deterministic mean-latent lineage also use committed source and earlier critical-test evidence.
This smoke does not independently re-decode saved rows or record a second live preview-value probe.
No new statistics, token caches, flow training, or trained-flow continuation were produced.
Next: separately authorize collection scope, train-only statistics, fresh caches, and a bounded flow-training smoke.

## 13. Fresh statistics/cache and flow-training smoke, 2026-10-20 (+08)

The operator says: “Ok, launch them” after the section 12 handoff.
Authorization covers fresh train-only statistics, new token caches, and a bounded flow-training smoke.
It does not authorize more collection, long training, simulator evaluation, or a model change.

Use the accepted eight-clip store and original split assignments as read-only inputs.
Copy the small store into a separate dataset directory. Keep original artifacts and failed audit files unchanged.
Use the same final VAE and data-contract hashes from section 12.

Execution plan, Workflow #21:

- Source: committed `295402f73` in `/home/fushan/mjlab-preview-vae-smoke-20261020`.
- New root: `logs/flow/preview-vae-flow-smoke-25updates-20261020/` under that worktree.
- Build a persisted window index; fit statistics from **2656 train windows only**.
- Build and validate fresh caches for train 2656, validation 224, and test 330 windows.
- Use four CPU workers for index/statistics and one cache writer. Cache ceiling: 1 GiB per split.
- Run **25 optimizer updates** on one currently idle GPU, with no resume or old flow weights.
- Keep `configs/flow_matching/x2_50hz_train.yaml` unchanged except for its artifact-contract path in a private copy.
- Preserve the 512 effective batch, 128 microbatch, accumulation 4, BF16, and 10000-update warmup.
- Reload `checkpoint-last.pt` and generate **8 validation windows** with the existing default Euler sampler.
- Limit the whole command to 900 seconds, plus brief exact-process cleanup.
- Require finite metrics, exact update/window counts, saved/reloaded checkpoint identity, and unchanged original input hashes.
- Preserve all stage logs and exit codes. Stop on failure; do not change code or increase the budget automatically.

Preparation result: index, train-only statistics, cache build, and cache validation each exit 0.
Monitor #6 then stops on a launcher-only report-path assumption, before GPU training.
`build-token-cache --validate-only` deliberately ignores `--json` and prints its report to standard output.
The continuation consumes the saved successful output. It does not rebuild any prepared artifact.
Monitor #7 runs only the unstarted training and generation stages.
The original overall deadline remains **21:38:45 (+08)**. The first failed log and exit status remain preserved.

The short warmup-only training run checks execution and checkpoint handling, not learning quality.
No training-quality or deployment claim follows from generation metrics or baseline comparisons.

### Accepted result

Workflow #21 is accepted after direct artifact and cleanup checks.
Monitor #6 preserves the preparation-launcher failure. Monitor #7 exits 0 for the remaining stages.
No successful preparation stage or training update was repeated.
The original command starts at **21:23:45 (+08)**. The continuation finishes at **21:29:17 (+08)**.
This is within the original **21:38:45 (+08)** deadline.

| Check | Result |
| --- | --- |
| Train-only statistics | 2656 training windows; no validation/test fitting |
| Fresh cache shapes | train `(2656,41,167)`, validation `(224,41,167)`, test `(330,41,167)` |
| Cache data | float32; all values finite; validation passes |
| Training | exactly 25 optimizer updates; exit 0; 51.752 seconds |
| Checkpoint reload/generation | 8 validation windows, 20 default Euler calls; exit 0; 17.996 seconds |
| Sampled training GPU memory peak | 2125 MiB on physical GPU 0 |
| Source and original inputs | source `295402f73` tracked-clean; input hashes unchanged |
| Cleanup | no owned flow processes; all eight GPUs 1 MiB / 0% at 21:30:34 (+08) |

Identities:

- Statistics: `1dd4d8166dc982370e0f6c1edb5a94d1393955070f57e9c3fc2258614e682707`.
- Train cache: `55e704cec2d0aaad6b6d3d80fa416ed0dc6b8244d4f55553c16eb459cac8b669`.
- Validation cache: `d6e54b0dd300899cfdaa9d124ad0d3e0a6011b9cca77d39ed7272536ace27723`.
- Test cache: `110b46e4bcc2b09a24234bef6eeac9376911ece5a5f1574a1372fcac0acf2e5d`.
- `checkpoint-last.pt`: `44de8d4b4760c6bbe8de29db2d489fc3389b327e10cc60358bcf417d1d73ef34`, 322260127 bytes.
- Generated sample: `53fb0234d563bb7e8fa0d0adb5d802309c714650fdfdb3c39d3f76a2247a9a68`.

Training loss is 2.295902; best validation loss is 2.523321.
All generation baseline gates are false on the eight sampled validation windows.
The 25-update run remains inside the unchanged 10000-update warmup; this is not a learned-policy quality pass.
It proves data preparation, optimization, checkpoint saving/loading, identity checks, and finite generation execute together.

Remote run: `/home/fushan/mjlab-preview-vae-smoke-20261020/logs/flow/preview-vae-flow-smoke-25updates-20261020/`.
The dataset is under `dataset/`, the model under `train25/`, and the launcher evidence under `control/`.
Local evidence:

- [Original preparation log](../../logs/diagnostics/preview-flow-port-20261020/flow-training-smoke/monitor.log).
- [Successful continuation log](../../logs/diagnostics/preview-flow-port-20261020/flow-training-smoke/monitor-continue.log).
- [Result summary](../../logs/diagnostics/preview-flow-port-20261020/flow-training-smoke/evidence/summary.json).
- [Training report](../../logs/diagnostics/preview-flow-port-20261020/flow-training-smoke/evidence/train.json).
- [Generation report](../../logs/diagnostics/preview-flow-port-20261020/flow-training-smoke/evidence/generation-smoke.json).

Next: choose and authorize the 144-motion collection and training budgets.
These smoke statistics cover eight clips only. Refit statistics and rebuild caches for the expanded dataset.
No full collection, long training, trained-flow simulator test, or deployment has started.
