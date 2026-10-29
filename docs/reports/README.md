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
| `x2_joint_pose_dataset_summary.md` | dataset record | latent-free X2 joint-pose dataset: format, storage, location, limitations | prepared; not yet used by training |
| `x2_joint_pose_datasets.md` | comparison | D1 window dataset versus the raw joint-pose dataset | introduction |
| `x2_latent_free_joint_space_feasibility.md` | feasibility | latent-free joint-space flow proposal and the S1/S2/S3 plan | plan; S1 done |
| `x2_joint_pose_trackability_and_modes.md` | measurement + tool note | trackability pass 1, mode coverage, and the survey tooling | pass 1 measured; no data removed |
| `x2_joint_pose_flow_training_runs.md` | experiment + diagnosis | four-arm 400k flow-training status, divergence signature, reference-paper comparison | checked 2026-10-23; 1 arm stable, 3 diverged; fixes only proposed |
| `x2_joint_pose_flow_divergence_findings.md` | experiment + diagnosis | 20 arms: seed dependence of the divergence, the accumulated-learning-rate threshold, the final-norm refutation, schedule and checkpoint-selection consequences | checked 2026-10-24; training loss only; mechanism and held-out quality unmeasured |
| `gpu-container-storage-nas-and-nvme.md` | operational note + measurement | managed-container storage: NAS versus local NVMe, measured throughput, sizes, rules and failure modes | checked 2026-10-24; throughput measured on two dates, per-stream ceiling unknown |
| `x2_flow_matching_divergence_ablation_20261023.md` | experiment | closed-loop failure against a 13-arm batch, lr, seed and level ablation; the remaining preview-term candidate | measured; divergence reproduced in 11 of 11 arms |
| `x2_flow_matching_divergence_20261024.md` | experiment | divergence diagnosis: capacity, anneal horizon, 1M-update budgets, and closed-loop scoring of seven checkpoints | measured; early annealing prevents the collapse in one arm; capacity and budget ruled out; seed spread 13 points |
| `x2_flow_matching_divergence_precursor_20261025.md` | experiment + diagnosis | offline phase 0 on saved logs and captures: gradient versus loss ordering, per-layer attention logit scale, survivor controls | measured; one layer runs away 100 to 1 000 times the survivor level at the loss minimum; cause not established |
| `x2_flow_matching_precision_branch_20261025.md` | experiment | pure fp32 against bf16 from one checkpoint, 15 000 updates each, plus a same-batch attention comparison | measured; fp32 stayed flat and never crossed the control; one seed, one shape |
| `x2_flow_matching_measurements_20261027.md` | measurement note | the 1 M-update fp32 runs, per-layer attention weight evidence, and the measured cost of an update, a restart and storage | measured; fp32 passed exposure 50.09 with flat per-layer weights; TensorBoard on NFS confirmed as the throughput cause; ~40 ms of each step unmeasured |
| `x2_diffusion_data_collection_review_20261029.md` | review | the D1 collection procedure and the built 144-clip dataset: rollout policy, schedule, splits, window layout and reference-frame coverage | measured on one store, 2026-10-29; 87.45 % of reference frames covered, no frame 0 to 20; three earlier statements retracted; the standing policy runs at clip end, not at a fall |
| `x2_vae_latent_decoder_findings_20261027.md` | findings note | what the latent and the VAE decoder do: input layout, effective rank, attribution, open-loop and closed-loop authority | current; software and simulation evidence only, no robot test |
| `x2_vae_decoder_literature_review_20261028.md` | literature review | motion-initiation failures, latent underuse, copycat policies, and state-conditioned decoder semantics | reviewed 2026-10-28; cause of the X2 failure not established; no new experiments |
| `tracking_footstep_experiments.md` | experiment log | G1 29DOF mode-15 footstep tracking experiments | 2026-09-11 record; not revalidated |
| `ACCEPTANCE-20261007.md` | acceptance | difficult-state coverage integration acceptance | supersedes the removed `PARENT-ACCEPTANCE.md` |
| `beyondmimic_diffusion_d1_throughput_integration.md` | validation | D1 collection and dataset throughput integration | implemented; speed not measured |
| `beyondmimic_diffusion_d1_throughput_acceptance.md` | validation | Parent D1 optimization acceptance and evidence limits | 222 CPU tests pass; CUDA/NAS unmeasured |
| `beyondmimic_diffusion_d1_server_benchmark.md` | experiment | Matched GPU3 D1 collection and preparation benchmark | merge3.14x/stats4.85x; cache-jobs8 slower |
| `beyondmimic_diffusion_d2_history_dropout_series.md` | experiment | D2 history state+latent dropout 0/20/50/100% | 100% costs about11x held-out loss; loss only |
| `beyondmimic_diffusion_d1_defaults.md` | validation | Enabled D1 defaults and local commit gates | 227 CPU passes; cache worker1 |
| `MERGE-MANIFEST.md` | provenance | lane provenance for the coverage integration tree | historical |
| `mvp-source-report.md` | note | fixed-state coverage evaluator scope | historical |
| `x2_multi_motion_ppo_8gpu_nccl_hang_diagnosis.md` | diagnosis | 8-rank multi-motion PPO NCCL hang | not reproduced; cause unknown; one 900-iteration run clean |
| `x2_multi_motion_ppo_8gpu_20k_resume_result.md` | result | 8-GPU 20,000-iteration shared teacher, resumed from iteration 899 | exit 0; episode collapse solved; root position the weakest term; one seed, no held-out evaluation |
| `x2_multi_motion_ppo_vs_vgk8_specialists.md` | comparison | shared teacher against the eight per-clip `vgk` specialists | teacher level on root-relative error and zero failures in 2048 episodes; in-distribution; root position not measured |

Two Simplified Chinese documents moved out of this directory to the shared research landing
page: `research/mjlab/vae-current-issues.zh-CN.md` and
`research/mjlab/vae-distillation-overview.zh-CN.html`.

## Reading rules

- Every status line is dated. A date shows when the author last checked the claim, not now.
- Separate implementation status, validation evidence and operational status.
- An accepted record is not fresh validation.
