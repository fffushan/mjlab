# mjlab reports

Records of work that is already done: validation reports, reviews, measurements and
operational notes. Plans and their acceptance records live in `../plans/`. Rationale and
architecture of record live in `../design/`.

These files are not part of the published Sphinx build. The published manual is under
`../source/`.

## Contract exception

The eight frozen machine-readable contracts stay in `../plans/` on purpose. Their paths are
load-bearing: 25 `configs/**/*.yaml` files, 15 test files, four source files (including the
`mjlab-diffusion` CLI defaults), four experiment scripts and `run_task.sh` reference them by
path. Sixty-five tracked files in total, before the docs pages and the research landing page.
Moving them requires a separate, explicitly authorized change that updates all of them.

## Index

| File | Kind | Subject | Status |
|---|---|---|---|
| `beyondmimic_diffusion_d0_validation.md` | audit | D0 diffusion specification audit and closeout | complete; D1/D2 authorization comes from the contract, not this report |
| `beyondmimic_reproduction_review.md` | review | comparison with the external BeyondMimic-Reproduction repo | notes only; no source change |
| `beyondmimic_diffusion_d5_followup.md` | measurement note | D5 warmup, sustained timing and deferred gates | diagnostic evidence only |
| `beyondmimic_diffusion_no_mixtures_migration.md` | operational note | migrating to the 167-channel no-mixture representation | current procedure |
| `beyondmimic_vae_optimizer_batching.md` | measurement note | accumulation versus minibatch cost on the VAE trainer | measured; one cohort, one seed |
| `vae-collection-reuse-and-statistics.md` | validation | collection reference reuse and deferred statistics | implemented and validated |
| `vae-sample-budget-implementation.md` | validation | collection-linked VAE sample budgets | implemented and validated |
| `vae-throughput-defaults-and-collection.md` | validation | throughput defaults and combined reference validation | implemented and validated |
| `tracking_policy_command_recording.md` | tool note | recording native tracking policy commands and reading the stages | current tool |
| `tracking_footstep_experiments.md` | experiment log | G1 29DOF mode-15 footstep tracking experiments | 2026-09-11 record; not revalidated |
| `ACCEPTANCE-20261007.md` | acceptance | difficult-state coverage integration acceptance | supersedes the removed `PARENT-ACCEPTANCE.md` |
| `beyondmimic_diffusion_d1_throughput_integration.md` | validation | D1 collection and dataset throughput integration | implemented; speed not measured |
| `beyondmimic_diffusion_d1_throughput_acceptance.md` | validation | Parent D1 optimization acceptance and evidence limits | 222 CPU tests pass; CUDA/NAS unmeasured |
| `beyondmimic_diffusion_d1_server_benchmark.md` | experiment | Matched GPU3 D1 collection and preparation benchmark | merge3.14x/stats4.85x; cache-jobs8 slower |
| `beyondmimic_diffusion_d1_defaults.md` | validation | Enabled D1 defaults and local commit gates | 227 CPU passes; cache worker1 |
| `MERGE-MANIFEST.md` | provenance | lane provenance for the coverage integration tree | historical |
| `mvp-source-report.md` | note | fixed-state coverage evaluator scope | historical |

Two Simplified Chinese documents moved out of this directory to the shared research landing
page: `research/mjlab/vae-current-issues.zh-CN.md` and
`research/mjlab/vae-distillation-overview.zh-CN.html`.

## Reading rules

- Every status line is dated. A date shows when the author last checked the claim, not now.
- Separate implementation status, validation evidence and operational status.
- An accepted record is not fresh validation.
