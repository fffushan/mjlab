# Preview-aware VAE to latent flow matching: implementation plan

Date: 2026-10-20 (+08).
Status: local port accepted. The operator authorizes commit, isolated server staging, and the bounded smoke in section 11.
Full collection and flow training remain unauthorized.
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
