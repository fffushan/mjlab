# X2 tracking training lineage: teachers, VAEs, datasets and flow models

Status: artefact register, written 2026-11-12 (+08).
Scope: the three training generations produced so far, with the recipe, dataset, VAE,
teacher and flow-model location for each. The register records *where things are*, not
whether they are good.

Every digest here was read on 2026-11-11 or 2026-11-12 (+08). A digest identifies an
artefact. It does not gate a run. A date shows when the value was checked, not that the
value is current.

## 0 Generations at a glance

| Gen | References | Teacher | VAE | Flow model | State |
| --- | --- | --- | --- | --- | --- |
| A | 144 clips | 20 000-iteration PPO, 8 GPU | 10 000 iterations, latent 32 | five arms, up to 1 M updates | complete; closed loop measured |
| B | 3 261 clips | 30 000-iteration PPO, 8 GPU | 20 000 iterations, latent 32 | four arms, 1 M updates each | complete; closed loop partly measured |
| C | 3 261 clips | 30 000-iteration PPO, 8 GPU, measured servo damping | not trained | not trained | teacher complete 2026-11-12; rest missing |

Generation C is a fork of generation B. It changes one thing: the actuator damping comes
from the measured servo model instead of the vendor table. It keeps the same references,
seed and budget.

## 1 Generation A: 144-clip multi-motion

### A.1 Teacher policy

| Item | Value |
| --- | --- |
| Task | `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion` |
| Iterations | 20 000 (resumed from `model_899.pt`) |
| Environments | 8 192 per rank, 8 ranks, 65 536 total |
| References | 148 clips, 55 to 56 envs per clip |
| Source commit | `9efad827d` |
| Wall clock | 15 h 05 m 35 s service, 13:22:49 reported training |
| Final checkpoint | `logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/training/agibot_x2_multi_motion_tracking/2026-10-14_21-06-44_all-tennis-8192x8-resume-20k/model_19999.pt` |
| SHA-256 | `82f0dc32fb3b79406242d6a83e91d7e1a761b4de099d47a072f41c039431e51d` |
| Evidence | `docs/reports/x2_multi_motion_ppo_8gpu_20k_resume_result.md` |

The checkpoint is present locally. The run exited `0` with no NCCL timeout.

### A.2 Teacher ONNX cohort

| Item | Value |
| --- | --- |
| Directory | `logs/diagnostics/x2-multi-motion-distill-20261015/teachers/` |
| Files | 144 `single_<id>.onnx`, one per clip |
| Export report | same directory, `export-report.json` |
| Observation width | 226 |
| Source checkpoint | the 20 000-iteration checkpoint in A.1 |
| Size per file | 16 494 845 B for `single_000.onnx` |
| SHA-256 (`single_000`) | `1262bcc764cfee4b783763ce147a3cdd00156b0197f2ecd062c63517bc0e736d` |

The export report names one checkpoint for all 144 clips. The files differ per clip
because the exporter bakes the clip reference into the graph. Failures recorded in the
report: none.

Note the count: the teacher was trained on 148 references, and the cohort holds 144.
Four clips are not in the cohort. A 148-versus-144 comparison across generations is
therefore not like for like.

### A.3 VAE

| Item | Value |
| --- | --- |
| Checkpoint | `logs/distillation/x2-multimotion-vae-10k-resume2-20261019/train/checkpoint-final.pt` |
| Size | 4 344 428 754 B |
| SHA-256 | `9d069b4552ec2a25990160477a99de24c3025c39ee1221660ebaf25b54a5ee00` |
| Iterations | 10 000 |
| Latent | 32 |
| Decoder mode | gravity |
| Dimensions | encoder command 68, conditioning 99, latent 32, action 31 |
| Cohort manifest | `configs/distillation/x2_tennis_multimotion_v1.yaml` |
| Cohort manifest SHA-256 | `cd597b1cf1f734562c37295233f2165aa62399544aee485dff5d47075dceaf21` |
| Evidence | `docs/plans/beyondmimic_diffusion_d1_no_mixtures_multimotion144_contract.yaml` |

Real-robot validation of this VAE: none (`user_confirmed_real_robot_validation: false` in
the contract).

### A.4 Dataset

| Item | Value |
| --- | --- |
| Durable location | `/mnt/fushan/mjlab-multimotion144-bulk-20261020/data` |
| Size | 140 GB |
| Contract | `docs/plans/beyondmimic_diffusion_d1_no_mixtures_multimotion144_contract.yaml` |
| Contract identity | `c045906e4b0c434c39a3dca01972c48e8e91ee7c6b8df22cc85fe842954d5385` |
| Collection date | 2026-10-20 (+08) |
| Coverage evidence | `docs/reports/x2_diffusion_data_collection_review_20261029.md` (87.45 % of reference frames covered; no frames 0 to 20) |

Two `/tmp` working copies of this dataset existed (`/tmp/mjlab-local/data`,
`/tmp/mjlab-uniform/data`). Both are gone. The NAS path above is the surviving copy.

### A.5 Flow models

| Directory under `logs/flow/` | Updates | Dropout | Train loss | Validation loss | Checkpoints kept |
| --- | --- | --- | --- | --- | --- |
| `x2-multimotion144-base-100k-best-20261021` | 100 000 | 0 | not recorded | not recorded | `checkpoint-best.pt` |
| `x2-multimotion144-fn-seed1-400k-20261024` | 400 000 | 0 | not recorded | not recorded | `checkpoint-best.pt` |
| `x2-multimotion144-fp32-prod-1m-20261025` | 1 000 000 | 0 | 0.01428 | 0.01713 | `checkpoint-best.pt`, `checkpoint-last.pt` |
| `uniform-hist0-1m-20261030` | 1 000 000 | 0 | 0.01909 | 0.02020 | `checkpoint-last.pt` |
| `uniform-hist20-1m-20261030` | 1 000 000 | 0.2 | 0.02555 | 0.02559 | `checkpoint-last.pt` |

All five are `x0_linear_flow`. The two `uniform-*` arms were trained with uniform anchor
sampling. The three earlier arms used the 2026-10-20 dataset with its original sampling,
so their loss values are not comparable with the `uniform-*` rows.

Evidence on training behaviour and closed-loop scoring:
`docs/reports/x2_flow_matching_divergence_20261024.md`,
`docs/reports/x2_flow_matching_divergence_precursor_20261025.md`,
`docs/reports/x2_flow_matching_measurements_20261027.md`.
Closed-loop evaluation artefacts: `logs/diagnostics/flow-144-eval-20261021/`.

## 2 Generation B: 3 261-clip union

### B.1 Teacher policy

| Item | Value |
| --- | --- |
| Task | `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion` |
| Manifest | `/tmp/mjlab-refs-20261103/motions-union.yaml`, 3 261 clips (3 117 per-variant + 144 originals) |
| Manifest SHA-256 | `247be896f36141ddaf9a5ca73f02da627965a256560e0b5ba2ee854620b289e3` |
| Iterations | 30 000 |
| Environments | 8 192 per rank, 8 ranks |
| Seed | 7 |
| Lookahead | 0.2 s |
| Source commit | `4f35093da9514ce226ab72e2c841f70355f3f999` |
| Wall clock | 21 h 48 m (13:51 on 2026-11-05 to 03:40Z on 2026-11-06) |
| Final checkpoint | `/mnt/fushan/teacher-union-30k-20261105/training/agibot_x2_multi_motion_tracking/2026-11-05_13-51-26_teacher-union-8192x8-30k/model_29999.pt` |
| SHA-256 | `c760252636f1b97cf487be5f471b8c4891647c7c916d4fc72d92bc94452311cc` |
| Log | `/mnt/fushan/teacher-union-30k-20261105.log` |

A continuation run exists and is **not** the cohort teacher:
`.../2026-11-06_12-56-15_teacher-union-8192x8-32k/model_31500.pt`. Its log ends after
1 772 iterations of a 2 000-iteration plan, with no exit record. The reason it stopped is
unknown. The cohort manifest names the 30 000 checkpoint instead. Do not confuse the two
when re-exporting.

### B.2 Teacher ONNX (shared)

| Item | Value |
| --- | --- |
| File | `/mnt/fushan/vae-multimotion3261-20261106/shared/single_150_zhanghongyu_agibot_x2_tracking.onnx` |
| Size | 15 910 854 B |
| SHA-256 | `77a6cbb54d066eebfb17aed579e0ad8a2ebe83138442377be840fa013c96b5b2` |
| Exported from | the model_29999 checkpoint in B.1 |

This generation uses **one** ONNX for all 3 261 clips, not one file per clip as in A.2.

### B.3 Cohort manifest

| Item | Value |
| --- | --- |
| File | `/mnt/fushan/vae-multimotion3261-20261106/x2_tennis_multimotion3261_shared_v1.yaml` |
| SHA-256 | `56becc5776be07600e612d92c3a6846b1a334e7017091f2fbad0062a34a825c0` |
| Entries | 3 261 |
| Base task | `Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations-Vendor-Gains` |

The base task name is load-bearing: the cohort carries vendor damping. Generation C
therefore needs a new cohort, not a new checkpoint path.

### B.4 VAE

| Item | Value |
| --- | --- |
| Run directory (server) | `/mnt/fushan/vae-multimotion3261-20261106/run-clipshard-k28-10k/` |
| Checkpoints | 31, `checkpoint-iter-001000.pt` to `checkpoint-iter-020000.pt` plus `checkpoint-final.pt` |
| Run size | 197 GB |
| Iterations | 20 000 (10 000 base + 10 000 extension on 2026-11-09) |
| Seed | 0 |
| Latent | 32 |
| Final printed loss | 0.4186 (last `total_loss` in `k28-extend20k.log`) |
| Lowest printed loss | 0.4009 in the extension segment |
| Local mirror (final) | `logs/distillation/vae-multimotion3261-20261106/checkpoint-final.pt`, 5 595 210 181 B, SHA-256 `d0c0ef71…` |
| Local mirror (17 000) | `logs/distillation/vae-multimotion3261-20261106/checkpoint-iter-017000.pt`, 5 595 198 789 B, SHA-256 `098d9af0…` |

Caveat on the loss values: the extension log holds 25 785 `total_loss` records for
iterations 10 000 to 20 000. The values are per iteration and noisy. The lowest record
(0.4009) is not at iteration 17 000, so the reason `checkpoint-iter-017000.pt` was kept as
the "lowest" arm is not recorded anywhere in the diagnostics directory. Treat the choice
as historical until someone re-derives it from the checkpoint records.

Warning: `run-clipshard-k28-10k/train-report.json` is 37.5 GB, because it keeps a record
per iteration. Do not `cat` it and do not load it in a notebook. Use
`checkpoint-*.pt` and the launch logs instead.

Closed-loop comparison of the VAE checkpoints against the teacher is in
`/mnt/fushan/vae-multimotion3261-20261106/eval-vae-arms/`: 16 members, 256 envs,
400 steps. Aggregated by `logs/diagnostics/vae-multimotion3261-20261106/compare_arms.py`,
the error was 0.3209 for the 17 000 arm, 0.3399 for the 10 000 arm, 0.3299 for the final
arm and 0.2847 for the teacher. The 17 000 arm was the best student on every error metric
in that comparison. Lower error is better.

### B.5 Dataset

| Item | Value |
| --- | --- |
| Durable location | `/mnt/fushan/d1-multimotion3261-20261109/run-2seeds-20261109` |
| Size | 187 GB: store 9.4 GB, token arrays the rest |
| Working copy (NVMe) | `/tmp/d1-3261-local`, 172 GB, **not durable**, train and validation arrays only |
| Contract SHA-256 | `493cc81f19bbb6e0b329b1ea673a1817200e7a8d9027b725ccc7b0b57dfa1cae` |
| Windows | train 5 222 396, validation 650 414, test 652 818, total 6 525 628 |
| Assignments SHA-256 | `6412f1f779bb91100d612d04a88fcccb90342e4187e9b42827b79cf2e87e321c` |
| Token arrays | `offline/tokens-train-b56816dfc404098a.npy` (143 030 981 776 B), `tokens-validation-2d344d14444d3aaf.npy` (17 813 538 760 B), `tokens-test-2fa1c67eaad54565.npy` (17 879 379 512 B) |

Both copies carry `dtype: <f4` and the same contract digest. The NAS copy also holds the
test array and the 1 886 store shards; the NVMe copy holds train and validation only,
which is enough for flow training.

Two derived indexes exist **only** in the `/tmp` working copy. Both are rebuildable:

| Index | Size | Value |
| --- | --- | --- |
| `offline/window_index.npz` | 1 335 885 550 B | SHA-256 `f9a46cea44a11bd2cc8c46f04e54390712fa4eda84b530f8c3c227bc8e2e99ea` |
| `offline/tokens-train-*.provenance.bin` | 124.51 MiB (plus 26.34 MiB JSON) | provenance index; 58.2 s to build |

Rebuild the window index with `python -m mjlab.scripts.diffusion build-index --jobs 8`.
Without it, every process rebuilds it serially and pays about 1 020 s per open.

### B.6 Flow models

Four arms under `/mnt/fushan/flow-3261-20261110/`. Each arm: 1 000 000 updates,
effective batch 512, 98 epochs. Each arm keeps 20 interval checkpoints (every 50 000),
plus `checkpoint-best.pt` and `checkpoint-last.pt`.

| Arm | Dropout | Train loss | Validation loss |
| --- | --- | --- | --- |
| `flow-3261-hist0-seed0` | 0 | 0.02647 | 0.02842 |
| `flow-3261-hist0-seed1` | 0 | 0.02654 | 0.02844 |
| `flow-3261-hist20-seed0` | 0.2 | 0.03898 | 0.03433 |
| `flow-3261-hist20-seed1` | 0.2 | 0.03911 | 0.03464 |

Seed spread is 0.26 % to 0.33 %. The dropout arms oscillate between epochs in a band of
0.038 to 0.042 and do not converge monotonically.

Known defect in all four: `train-report.json` has no `best_metric_split`, so
`checkpoint-best.pt` was written but never selected by a validation metric. For the
dropout arms the best validation loss is also *below* the final training loss, which
makes a best-versus-last comparison invalid. `perf_switches` is absent from these
reports because the runs started before that field was recorded.

Related non-primary artefacts: `flow-3261-sync-seed0` (sync-removal A/B, stopped at
153 000), `flow-3261-{w4,w16,w32,w64}-seed0` (dataloader worker sweep),
`flow-3261-hist0-seed0-ddp4` (4-card DDP, 1.20×, stopped).

## 3 Generation C: measured-servo teacher

| Item | Value |
| --- | --- |
| Task | `Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion-MeasuredServo` |
| Run root | `/mnt/fushan/measured-servo-teacher-30k-20261111/training/agibot_x2_multi_motion_tracking/2026-11-11_19-10-43_measured-servo-8192x8-30k/` |
| Iterations | 30 000 |
| Environments | 8 192 per rank, 8 ranks |
| Seed | 7 |
| Lookahead | 0.2 s |
| Manifest | the B.1 3 261-clip union manifest (same file, same digest) |
| Source commit | `28f422d4e` |
| Deployed tree | `/home/fushan/mjlab-measured-servo-20261111` |
| Wall clock | 22 h 09 m 34 s (19:10 on 2026-11-11 to 18:26 on 2026-11-12) |
| Iteration time | 2.75 to 2.84 s (reference 2.59 s) |
| Checkpoints | 61, `model_0.pt` to `model_29999.pt`, 5.8 GB total |
| Final metrics | mean reward 9.55, mean episode length 130.15 |
| Final checkpoint (server) | `model_29999.pt`, 97 702 031 B, SHA-256 `120cf2aa9525e7715fbb9f2b8c94b63203feda80e32eb2752dde515193b67ac7` |
| Local mirror | `logs/rsl_rl/agibot_x2_multi_motion_tracking/2026-11-11_19-10-43_measured-servo-8192x8-30k/` |

Local mirror contents: `model_29999.pt` and `params/{agent,env,resolved_motions}.yaml`
(SHA-256 `a308982b…`, `682bc896…`, `a9dfb233…`). The checkpoint carries no gains. Gains
come from the task config at export time.

What changed against B: 19 of 31 joints have different damping, and the articulation uses
25 actuator groups instead of 16. Waist pitch goes from 4.0 to 34.71, an 8.7× change.
Hip pitch goes from 5.0 to 14.48.

What this run measured: on reward and episode length the measured-servo teacher is
indistinguishable from the B.1 vendor-kd teacher at the same iteration (see below).
PPO reward does not separate the two damping models.

| Iteration | Measured servo | Vendor kd (B.1) |
| --- | --- | --- |
| 5 000 | 9.00 | 8.71 |
| 10 000 | 8.61 | 8.42 |
| 15 000 | 9.67 | 9.35 |
| 20 000 | 9.13 | 9.02 |
| 25 000 | 9.80 | 10.46 |
| 30 000 | 9.55 | 9.58 |

Missing for a full generation C: teacher ONNX export, cohort manifest, token collection,
VAE training, flow training. None of these has started.

## 4 Where the bytes live

| Class | Path | Durable | Note |
| --- | --- | --- | --- |
| NAS, training artefacts | `/mnt/fushan/` | yes | survives container restart |
| Container `/tmp` | `/tmp/d1-3261-local`, `/tmp/mjlab-refs-20261103` | **no** | lost on host restart |
| Local mirrors | `logs/flow/`, `logs/distillation/`, `logs/rsl_rl/` | yes, on this laptop | 1.1 TB free on `/` at check time |

Reference clips (`/tmp/mjlab-refs-20261103`: 144 originals plus 3 117 per-variant) are the
input to every teacher run. They are on `/tmp` only. If that path is lost, the union
manifest cannot be resolved and no teacher can be retrained without re-collecting the
references.

## 5 Reproduction recipes

Generation A teacher (recorded in `logs/diagnostics/x2-multi-motion-ppo-8gpu-resume-20k-20261014/`):

```bash
python -m mjlab.scripts.train Mjlab-Tracking-Flat-AgiBot-X2-Multi-Motion \
  --env.commands.motion.motion-manifest <motions.yaml> \
  --env.scene.num-envs 8192 --agent.max-iterations 20000 \
  --agent.seed 7 --agent.logger tensorboard \
  --gpu-ids "[0, 1, 2, 3, 4, 5, 6, 7]"
```

Generation B teacher (`/mnt/fushan/teacher-union-30k-20261105`): the same command with
`--env.commands.motion.motion-manifest /tmp/mjlab-refs-20261103/motions-union.yaml` and
`--agent.max-iterations 30000`.

Generation C teacher:
`logs/diagnostics/measured-servo-teacher-20261111/launch_measured_servo_teacher.sh`
(supports `--dry-run`; preflights the manifest, the task and the wrist fix).

Generation B flow arm (as launched by
`logs/diagnostics/flow-3261-20261110/launch_flow_3261_arms.sh`):

```bash
python -m mjlab.scripts.flow_matching train \
  --dataset-dir /tmp/d1-3261-local \
  --config <arm>/config.yaml \
  --contract <contract.yaml> \
  --out <arm>/run \
  --device cuda:0 \
  --max-updates 1000000 \
  --allow-long-run \
  --json <arm>/run/train.json
```

## 6 How to check an artefact without trusting this file

```bash
# newest checkpoint, resolved from the filesystem and not from a log line
ls -1 <run>/*.pt | sort -V | tail -1

# identity
sha256sum <file>

# a run's own record of what it trained
grep -E "num_envs|max_iterations|motion_manifest|lookahead_s" <run>/params/env.yaml
```

The last checkpoint of a run is not always named after the requested total. Generation C
ended at `model_29999.pt` for a 30 000-iteration plan, and the B.1 continuation ended at
`model_31500.pt`. Resolve the name from the directory, never from the plan.

## 7 Open defects that affect this register

1. `best_metric_split` is `None` in all four generation-B flow arms.
2. Generation C has no ONNX, cohort, dataset, VAE or flow model.
3. The reference clips under `/tmp/mjlab-refs-20261103` have no NAS copy.
4. `vae-multimotion3261-20261106/run-clipshard-k28-10k/train-report.json` is 37.5 GB.
5. Generation A datasets that lived in `/tmp` are gone; only the NAS copy survives.
