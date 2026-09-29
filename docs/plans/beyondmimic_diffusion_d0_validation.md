# BeyondMimic diffusion D0 validation report (corrected audit)

**Audit date:** 2026-09-29 (+08:00)  
**Repository:** `/home/agiuser/projects/mjlab`, branch `fxy/test/tracking-exp`  
**Disposition:** **D0 specification/audit complete, with explicit runtime-risk acceptance (2026-09-29). D1/D2 execution is not authorized by this report.** The current decision and gate reconciliation are in section 11; earlier open-gate statements are retained as audit history, not current status. The [frozen contract](beyondmimic_diffusion_d0_contract.yaml) supersedes the draft in section 7.

This is a bounded specification/evidence correction. It does not implement a
production diffusion package, collect data, train, run a simulator, use remote
resources, deploy, or operate hardware. The three-teacher scope remains a
**bounded engineering prototype**, not a broad tennis generator.

**Operator decision after audit (2026-09-29):** freeze
`logs/distillation/mixed-10k/checkpoint-final.pt` at the SHA256 in section 2.
The user confirms this VAE was validated on the real robot. This supersedes
candidate-freeze blockers in the historical audit/attestation below; hardware
validation is user-reported, not rerun by D0. It does not validate the future
diffusion layer or authorize new collection, training or hardware trials.

**Endpoint clarification (user, 2026-09-29):** the deployed controller switches
at the last reference frame to a separate zero-command PPO velocity policy,
trained from tennis-ending poses. The exact `x2-tennis-recovery-n25` ONNX and
SHA256 are pinned in the reproduction plan, section 4.4. Static ONNX inspection
found `obs [1,102] -> actions [1,31]`; no inference or rollout was run. This
resolves which controller follows the clip. The user subsequently settled its
role as post-clip data qualification: hybrid verification is labeled separately
from VAE-only recovery; no PPO rows or switch-crossing windows enter the VAE
state–latent dataset. The user then approved the recommended numerical defaults
and progression toward a bounded prototype despite unproven 50 Hz replanning.
Section 11 freezes qualification thresholds and reconciles the remaining tests.
**D1 has not started:** a D0 contract is not an implemented collector, a tested
qualification pipeline or a collected dataset.

## 1. Preservation and audit evidence

At the start of this follow-up, the checkout was:

```text
HEAD 107fe65bff995186ae7d61c9b1632a1f1947caf0
branch fxy/test/tracking-exp (ahead 12)
```

The earlier D0 run observed `ca3cc65b1874c68a16116f37a585081fe6f5ea55` initially,
then observed an external concurrent change to the current `107fe65...` commit.
The report previously inferred parent authorship; that attribution was incorrect.
This report makes **no authorship inference**: the commit was an externally
observed concurrent workspace change, owner unknown to this audit. No commit,
stage, reset, push, or checkout operation was performed here.

Existing modified/untracked work was preserved:

```text
 M docs/plans/beyondmimic_reproduction_review.md
 M docs/plans/beyondmimic_vae_distillation.md
 M docs/plans/beyondmimic_vae_implementation.md
 M docs/plans/beyondmimic_vae_m4_implementation.md
 M docs/source/x2_tennis_distillation.rst
?? docs/plans/beyondmimic_diffusion_reproduction.md
?? docs/plans/beyondmimic_diffusion_d0_validation.md
```

The requested ignored evidence bundle is:

```text
logs/diffusion/d0-validation-20260929T062344Z/
```

Its `SHA256SUMS` index covers corrected scripts, raw JSON/logs, rerun commands,
fresh source hashes, and preserved superseded probe artifacts. The first corrected
GPU invocation failed before model execution because the reference package import
path was absent; `initial_gpu_probe_failure.log` preserves that diagnosis. The
rerun added the explicit read-only reference `src` path and completed successfully.

The independent read-only reference repository remains verified at:

```text
/home/agiuser/projects/BeyondMimic-Reproduction
HEAD 55b37260ce02573e70648111c534b735972a2af4
working tree clean
```

## 2. Selected checkpoint and provenance distinction

The audited candidate, subsequently frozen by the user, is:

```text
logs/distillation/mixed-10k/checkpoint-final.pt
fresh sha256:
69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117
```

The checkpoint is a version-3 `mjlab-m4-cohort-distillation` artifact whose
**embedded metadata** reports schema `68/99/32/31`, gravity mode, a non-reference-
conditioned decoder, ELU `[2048,1024,512]` VAE MLPs, frozen VAE normalizers, 50 Hz
control (`0.02 s`, `0.005 s` simulation timestep, decimation 4), and three cohort
members. These are artifact metadata checks, not fresh action, decoder, frame, or
closed-loop tests.

The embedded cohort lists:

| member | frames/FPS | embedded checkpoint hash | provenance |
|---|---:|---|---|
| `tennis_000` | 453 / 50 | `6dec4ead960f38d516351f8e6b3ee6399254565662dcd2949d57ab748adcbbc4` | `model_29999.pt` |
| `tennis_001` | 340 / 50 | `6fd5d68a67630c5ab726bd12a718734a4135e4620932cc9a4eca43c452a61ae2` | `model_29999.pt` |
| `tennis_002_ss` | 320 / 50 | `c02b53bea52b7ad416821e38fc7b7e5324eecb75fc54e253b7fc4474932fffc0` | **mid-training `model_16500.pt`**, not final |

The third teacher's frozen provenance is
`logs/distillation/teachers/tennis_002_ss/PROVENANCE.md`; its 16,500-iteration
status is a known limitation, not silently upgraded here.

The fresh hashes generated during this correction are in
`fresh_source_hashes.txt`. They include the candidate checkpoint, manifest,
all three source teacher checkpoints, all three motion NPZs, all three ONNX files,
and saved environment/agent YAMLs. The fresh hashes match the relevant embedded
hash records, but this still verifies bytes/provenance only. It does **not** prove
fresh decoder action parity, ONNX parity, sensor-frame parity, or closed-loop
quality. Existing accepted M1/M4 evidence is cited as historical evidence rather
than re-labeled as fresh validation.

The eight-teacher `x2_tennis_mixed_v2.yaml` is explicitly not substituted and is
not an accepted eight-teacher VAE. Existing M4 reports establish a bounded shared
VAE baseline over three clips, with known evaluator nondeterminism around `1e-3`;
they do not establish general tennis generation, deployment, sim2sim, or hardware
qualification.

## 3. X2 contract checks and limits

Structural source/asset inspection and a bounded MuJoCo compile probe established:

- physical free-joint/root body: `pelvis`;
- tracking anchor in the selected VAE cohort: `torso_link`;
- `imu_0` site body: `pelvis`;
- `imu_1` site body: `torso_link`;
- 31 compiled joints in the candidate's named order;
- 20 tracked bodies in the candidate's ordered subset, with indices
  `[0,2,4,6,8,10,12,15,17,19,20,21,22,24,26,27,28,29,30,31]`;
- proposed raw state width `15 + 6*20 = 135`;
- VAE conditioning fields: gravity 3 + gyro 3 + joint position 31 + joint velocity
  31 + previous executed action 31 = 99;
- latent 32, action 31, and 50 Hz period.

The compiled asset has 32 non-world bodies, but the 20-body tracked subset—not the
full asset count—is authoritative for the proposed state. The XML fresh hash is
`ae0dcbceda3ef74e029e3bf4a9ea24ea8cc58ab44f1518e635653fc79f928125`.

These are **structural checks**, not runtime behavior checks. In particular, the
candidate metadata's gravity and gyro frame fields are `declared_unverified`; no
fresh decoder action test, estimator test, physical velocity reconstruction test,
or closed-loop frame test was run here. The planned representation still requires
root pose/twist in the current character-yaw frame, local body pose/velocity in
each instantaneous root-yaw frame, and restoration of current root velocity before
physical velocity costs.

The candidate's three clips are 9.06 s, 6.80 s, and 6.40 s. The user subsequently
identified the existing endpoint handoff to zero-command PPO standing (see the
update above and plan section 4.4). It can provide a physically continuous hybrid
verification trace, but not five-second VAE-only survival. Reference wraps,
teleports, timer resamples, or post-reset observations must not count as survival.
Thresholds and coverage rules are frozen in the v1 contract (section 11).
Real-history handover/qualification execution remains a D1/D3 test obligation;
hardware estimator topology remains a separate D6 gate.

## 4. Corrected CPU schedule/DDIM audit

### 4.1 Paper/reference distinction

The paper reports DDPM formulation, independent state/latent denoising indices, an
`x0` reconstruction objective, and 20 denoising steps in Table S7. Its reported
deployment is 25 Hz with history 4 and horizon 16 (0.64 s). The requested X2
50 Hz, history 8, future 32 adaptation is an engineering choice, not a paper
setting. The paper's approximate RTX 4060 Mobile 20 ms timing is not an X2 5080
closed-loop result.

The reference repo's `diffusion_full_50hz_h8_f32.yaml` has the desired 41-token
shape and 20 denoising steps, but normalization is disabled. Its old schedule
returns 20 noisy entries with first `alpha_bar=0.992007315158844`; it has no exact
clean index. Its sampler clamps IDs and its reduced-step path uses a schedule
prefix. Those behaviors are retained as adversarial reference findings, not copied.

### 4.2 Correct canonical schedule

For the proposed engineering choice `K_train=1000`, the corrected contract has
**1001 timestep embeddings**, with clean index `0` and noisy indices `1..1000`.
A bounded cosine-beta construction was explicitly tested:

```text
raw cosine alpha-bar: alpha_bar[0]=1
betas[k] = clamp(1 - raw_alpha_bar[k+1]/raw_alpha_bar[k], 1e-5, 0.999)
alpha_bar = [1, cumprod(1-beta)]
```

The resulting terminal `alpha_bar[1000]` is `2.4287669070348542e-09`, terminal
SNR `2.428766912933763e-09`, and therefore near-Gaussian without claiming the
unbounded raw cosine value (`~3.75e-33`) as the bounded schedule result.
`1e-5` and `0.999` are explicit proposed engineering bounds, pending parent
approval; they are not paper-reported settings.

Twenty reverse **jumps** require 21 grid points. The canonical full-range grid
used by the corrected probes is:

```text
points:       [1000,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50,0]
source IDs:   [1000,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50]
destination:  [950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50,0]
```

Thus there are exactly **20 denoiser calls**, **20 DDIM jumps**, 20 nonclean
source IDs, and final clean index 0. There is no denoiser call at index 0. This
corrects the prior 19-jump CPU probe and the inconsistent `999..0` manifest.

### 4.3 Corrected CPU evidence

Command:

```sh
uv run --no-sync python logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.py
```

Evidence:

```text
logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.py
logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.json
logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.log
```

Assertions passed for 1001-entry schedule, bounded beta range, exact canonical
IDs, strict source>destination ordering, 20 calls/jumps, final destination 0,
and finite skipped-step DDIM oracle. Maximum analytic oracle error was
`2.220446049250313e-16`.

The corrected probe also creates separate state/latent noise tensors and asserts
shape and non-aliasing storage. This is a real check, unlike the old string field.
It does **not** establish statistical independence; that criterion is explicitly
`NOT CHECKED`.

The prior raw CPU/GPU scripts and old measurements are preserved under
`superseded_original_*` in the evidence bundle. They are superseded and must not
be used as current DDIM evidence.

## 5. Corrected bounded RTX 5080 probe

At the start and after completion, local `nvidia-smi` showed the RTX 5080 Laptop
with 15 MiB / 16,303 MiB and approximately idle utilization. No compilation,
TensorRT, power change, persistent process, remote resource, or hardware command
was used.

The corrected probe uses the plan's proposed projected-state token width:
`state_dim=199` (135 raw plus proposed 64 projection rows), latent 32, combined
width 231, sequence length 41, six 512-wide transformer layers and eight heads.
It instantiates 1001 timestep embeddings and reports the actual parameter count,
`20,197,607`. The 64-row projection remains a proposed contract; it was not
silently treated as an accepted trained artifact.

Command:

```sh
timeout 120 uv run --no-sync python logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.py
```

The probe precomputes source/destination IDs, alpha coefficients, and step-ID
tensors before the timed loops. It performs no per-reverse-step `int()`/`.item()`
conversion of CUDA scalars and freezes model parameters in eval mode. A final
finite-output `.item()` check is inside each timed sample, so its synchronization
is included. It first
runs one FP32 finite-output check, then uses BF16 autocast with FP32 weights.
Recorded zeros are re-applied by masks after every jump: tokens 0–7 state+latent
and token 8 state are fixed; current latent and future entries remain unknown.
However, this probe supplies the source noise ID to **all** state/latent step
embeddings, including the clamped entries. Therefore it does not validate the
planned exact-clean conditioning semantics; that requires zero IDs on the known
blocks. The timings below are accepted only as untrained operation-cost evidence.

Evidence files:

```text
logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.py
logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.json
logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.log
```

Measured raw samples (20 unguided, 10 guided) are retained in JSON. No p95/p99 is
reported because the sample counts are small; median/min/max are reported instead:

```text
actual parameters: 20,197,607
shape: [1,41,231]
embedding count: 1001
source IDs: 20 values, 1000 down to 50
DDIM jumps/calls: 20 / 20
final clean index: 0

unguided forward-only:
  CUDA-event median 21.4317 ms; wall median 21.4394 ms
  wall min 19.8115 ms; wall max 24.2473 ms
  peak allocated 89.88 MiB; peak reserved 110.0 MiB

guided one-forward/input-VJP proxy:
  CUDA-event median 80.9590 ms; wall median 80.9748 ms
  wall min 79.8622 ms; wall max 82.4797 ms
  peak allocated 137.68 MiB; peak reserved 156.0 MiB
```

The guidance proxy is deliberately not the paper's guidance implementation. Per
update it performs exactly one denoiser forward, defines
`G=mean(x0_hat[future_state]^2)`, computes the input VJP
`dG/dx_t`, forms `x0_used = x0_hat - 0.05*normalize(dG/dx_t)`, then performs the
true DDIM jump. It performs no redundant second denoiser forward and does not
backpropagate into frozen parameters. It is a directional feasibility proxy only;
its cost, scale, and conversion to a score are not approved D0 semantics.

These are untrained tensor feasibility measurements only. They omit preprocessing,
transfers, actual projection/unprojection, VAE decode, scheduler/queue overhead,
thermal steady state, and trained-policy behavior. They do not certify 50 Hz
replanning, end-to-end latency, p99, or deployment. The corrected unguided result
is near/above the proposed 20 ms period; whether an optimized/exported runtime can
meet the requirement remains open rather than being declared impossible.

## 6. Corrected D0 checklist — audit snapshot before final decisions

This table distinguishes tests actually run during the original audit. Section 11
resolves the design decisions and assigns unperformed runtime tests to their
owning phase; it does not convert NOT CHECKED rows into tested passes.

| Gate | Status | Evidence / limitation |
|---|---|---|
| Paper/reference revision and relevant files inspected | PASS | Paper companion and reference HEAD `55b37260...` inspected |
| Provisional three-teacher candidate located without guessing | PASS | `mixed-10k/checkpoint-final.pt`; fresh candidate and source hashes in bundle |
| Embedded cohort/provenance reconciled with fresh source hashes | PASS (bytes/provenance only) | Fresh hashes match embedded records; no parity claim follows |
| Candidate VAE schema/normalizer metadata inspected | PASS (metadata only) | 68/99/32/31, gravity mode, frozen VAE normalizers |
| Candidate decoder action/parity test | NOT CHECKED | Existing accepted evidence is historical; no fresh decoder/parity rerun |
| Candidate frozen for downstream use | RESOLVED (operator decision) | User selected `mixed-10k` and confirmed real-robot validation after the audit; new runs remain separately authorized |
| Root/anchor/IMU structural identity | PASS (source/compile structure) | pelvis root, torso anchor, imu_0 pelvis, imu_1 torso |
| Runtime frame/estimator/action semantics | NOT CHECKED | No simulator/hardware/runtime parity test |
| Ordered 20-body subset and 31-joint metadata/compile order | PASS (structural) | Named order and compiled order inspected; no runtime state capture |
| 50 Hz / 41 / 99 / 32 / 31 dimensions | PASS (declared metadata/spec) | No diffusion runtime exists; action execution not tested |
| Physical velocity reconstruction | NOT CHECKED | Equations audited; no numerical runtime test |
| Reset/history/teleport behavior | NOT CHECKED | No collector or simulator test |
| Five-second late-phase endpoint policy | ROLE SETTLED; collector untested | User settled zero-command PPO standing as post-clip data qualification, not latent training data; label hybrid vs VAE-only outcomes, specify numeric thresholds and test handoff |
| Bounded cosine-beta schedule and clean index | PASS (CPU assertions) | 1001 entries, bounded beta, clean 0, terminal SNR recorded |
| Exact 20-jump DDIM grid | PASS (CPU/GPU probe assertions) | 20 calls, sources 1000..50, final destination 0 |
| Independent state/latent noise | NOT CHECKED (statistical) | Shape/non-alias assertions pass only |
| Guidance score/x0 semantics and acceptance tolerances | NOT CHECKED | One-forward proxy is not approved guidance |
| Corrected local GPU tensor timing | PASS (bounded evidence) | Raw samples, CUDA events, memory, finite checks retained; not end-to-end |
| Fresh simulator VAE baseline | NOT CHECKED | Existing historical baseline inspected only |
| Broad tennis/generalization | FAIL / out of scope | Three clips are an engineering prototype |
| Sim2sim/hardware qualification | NOT CHECKED | Outside D0 and unauthorized |

At this audit snapshot D0 remained open. The later user decisions and section 11
closeout supersede that disposition, not the unperformed-test statuses above.

## 7. Superseded draft manifest and owner decisions

This draft is retained for provenance only. Use the versioned
[beyondmimic_diffusion_d0_contract.yaml](beyondmimic_diffusion_d0_contract.yaml)
for the frozen contract; neither document is an executable runtime configuration.

```yaml
schema_version: d0-x2-state-latent-v2
vae:
  checkpoint: logs/distillation/mixed-10k/checkpoint-final.pt
  checkpoint_sha256: 69891dffb59af31539388e40041efe30242a2028f2876b746d1f7c5ef44ac117
  cohort_ids: [tennis_000, tennis_001, tennis_002_ss]
  latent_convention: mu_at_collection_and_decode
  decoder_schema: {conditioning: 99, latent: 32, action: 31, mode: gravity}
control: {frequency_hz: 50, period_s: 0.02, sim_timestep_s: 0.005, decimation: 4}
window: {past_steps: 8, current: true, future_steps: 32, sequence_length: 41}
state:
  raw_state_dim: 135
  projected_state_dim: 199  # 64 rows proposed, pending approval
  token_dim: 231
  root_body: pelvis
  anchor_body: torso_link
  tracked_body_count: 20
noise:
  training_grid_K: 1000
  embedding_count: 1001
  clean_index: 0
  beta_min: 1.0e-5       # proposed bound, pending approval
  beta_max: 0.999        # proposed bound, pending approval
  inference_updates: 20
  source_ids: [1000,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50]
  destination_ids: [950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50,0]
  sampler: ddim
  eta: 0.0
  target: x0
  state_latent_noise: independently sampled  # statistical test pending
startup:
  mode: vae_bootstrapped_real_history
  minimum_pairs: 8
endpoint:
  policy: final_frame_switch_to_zero_twist_ppo
  recovery_onnx: logs/rsl_rl/agibot_x2_velocity/2026-09-26_02-40-45_x2-tennis-recovery-n25/2026-09-26_02-40-45_x2-tennis-recovery-n25.onnx
  recovery_sha256: caf17c38f23ad180829a230a9a3259f14eedd3860dd90a046b8be92d3d047681
  five_second_verification: hybrid_qualification_not_vae_only
  acceptance_thresholds: pending
  recovery_rows_in_state_latent_dataset: false
```

The original pending decisions were verification thresholds/coverage, schedule,
projection, guidance conversion, VAE evidence and timing-risk treatment. They are
now reconciled in section 11. Independent noise tensor/level construction remains
an implementation test, not a statistical-certification or user-approval gate.

## 8. Recommended next action and residual risks

D0 design is now closed as described in section 11. After separate D1 execution
authorization, implement and test collector/replay/qualification contracts before
running the capped pilot. D2 owns the production noising/DDIM/mask/frame/projection
and guidance-gradient tests; mathematical audit fixtures are not those tests.
No collection, training, simulator baseline, deployment or hardware trial follows
automatically from the D0 closeout.

Residual risks remain: three clips cannot support general tennis claims; the
iteration-16,500 third teacher is not final; no fresh decoder/export/sim2sim/frame
parity exists from this audit; hybrid qualification and estimator execution are
untested; corrected tensor measurements omit end-to-end costs; the guided
production path remains unimplemented rather than equated to the paper's CppAD.

## 9. Exact commands and artifact index

Executed commands include:

```text
uv run --no-sync python logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.py                         # PASS
timeout 120 uv run --no-sync python logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.py           # PASS after import-path diagnosis
sha256sum <candidate and all source artifacts listed in fresh_source_hashes.txt>                                     # PASS
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits                # PASS, idle before/after
```

The failed initial GPU invocation and exact corrected rerun are preserved in
`initial_gpu_probe_failure.log`, `corrected_gpu_probe.log`, and `rerun_commands.sh`.
The old flawed probes/raw measurements are preserved as `superseded_original_*`
and are explicitly superseded. `SHA256SUMS` is the evidence index.

## 10. Parent evidence review — 2026-09-29

The parent read both corrected scripts and raw JSON, verified all 17 candidate/source
hashes and all 14 indexed evidence files, and independently reran the CPU probe in
an isolated temporary directory using the existing mjlab environment. The JSON
matched exactly; the parent additionally asserted 21 endpoints, 20 jumps and
analytic error below `1e-12`. This establishes schedule/jump arithmetic, not a
complete learned sampler. The GPU timings were source-reviewed, not rerun by
the parent.

Additional limitations retained explicitly rather than hidden by a PASS label:

- The GPU script clamps known data but does not supply clean noise IDs for those
  blocks (section 5). It is an operation-cost probe, not validated conditioning.
- Guided timings measure one forward plus an input VJP and an illustrative x0
  correction, **not** the proposed score-consistent guided sampler. Gradients on
  fixed entries are not zeroed before its proxy norm; real guidance tests must
  cover that distinction. No cost-descent, trained quality or parity claim follows.
- No seed is set in the corrected GPU script, and its FP32 check is finite-output
  only, not FP32/BF16 numerical parity. The final finite check synchronizes once
  per measured sample. The small sample count does not establish p99 or sustained
  deadline behavior, regardless of CUDA-event versus wall timing agreement.
- The CPU noise probe checks separate RNG draws/storage, not separate state/latent
  timestep IDs or the production noising code. Those are D2 test obligations;
  proving RNG statistical independence is not a new user approval requirement.
- Prior accepted evidence remains usable when the candidate/contract is unchanged.
  Fresh baseline trials should fill an identified coverage/tolerance gap, not
  introduce a new full teacher-quality qualification solely because reports are
  historical. Hardware and previously untested endpoints remain separate gates.

**Parent conclusion at initial review:** accept the artifact inventory,
reproducible CPU arithmetic and bounded tensor-cost evidence, not complete sampler
correctness, 50 Hz feasibility or execution authority. D0 was open then; the
subsequent user-approved design closeout is in section 11. The technical timing
limitations above remain valid. No further child is running.

## 11. D0 closeout — 2026-09-29

**Accepted scope:** specification and bounded audit, with the user explicitly
accepting progression toward a small prototype before real-time feasibility is
proved. This is not a claim that all original checklist rows passed. The selected
VAE's accepted prior evidence and user-confirmed real-robot validation are reused;
no fresh simulator baseline was required solely because earlier evidence was
historical. The new collector/adapter still needs its own tests.

Frozen design: `docs/plans/beyondmimic_diffusion_d0_contract.yaml`, SHA256
`42694cdd3acd72afd2d234a85ca29e9c7a646d64df831021c8dd43e55f3f73e7`.
This is a documentation manifest, not a CLI-ready config. Source paths resolve
from the mjlab root. It pins the VAE/recovery artifacts, 20-body order, frames,
normalization/projection recipe, 1,000-level bounded cosine schedule, 20 DDIM
jumps, clean masks/IDs, real-history startup and a score-consistent guidance
convention. Initial continuation is unguided; D4 preregisters positive cost
weights/strengths before guided tests. Seed 42 and float64 PCG64 Gaussian projection
are explicit engineering defaults, not paper facts.

### 11.1 Qualification criteria and their actual sources

Read-only saved-config and termination-source inspection found:

| Criterion | Frozen rule | Meaning |
|---|---|---|
| Physical fall | Root tilt **>70 degrees** (`1.2217304763960306` rad), existing standing task `fell_over` / `bad_orientation` | Reuse in both controller phases as an explicit collector adaptation; the saved tracking task instead used reference-error guards |
| Tracking anchor height | Absolute reference error **>0.25 m** | Reject during VAE tracking, label `tracking_error`, not `physical_fall` |
| Tracking orientation | Absolute difference of reference/actual anchor projected-gravity **z components >0.8** | Dimensionless, not 0.8 radians; active only during VAE tracking |
| Tracked end-effector height | Any selected ankle/wrist z-reference error **>0.25 m** | Same four bodies as all three saved teacher configs; active only during VAE tracking |
| Numerical/data integrity | No nonfinite physics, observations or executed actions; VAE latents finite when VAE owns control; no mid-trial resets/teleports/unexpected resamples | PPO rows have no VAE latent and cannot enter latent training data |
| Completion | **250 continuous 20 ms transitions**, including final post-step checks, from first trial action | Five seconds total; a failure on the final transition takes precedence over success |
| Clip end | Existing zero-command PPO handoff; no physical reset | Label `vae_then_standing`, distinct from `vae_only`; disable obsolete reference-error guards after handoff |
| Timeout/interruption before completion | Incomplete, not successful | Never use missing post-reset/terminal evidence as survival |

All three teacher configs agree on the reference-error thresholds. The standing
config and both termination source files are hash-bound in the manifest. The
70-degree rule is inherited from the existing standing task and deliberately
applied to both phases; this is not a claim that the tracking task already used
it. No new contact-force or root-height thresholds are invented. Passing this
filter means bounded survival/tracking qualification, **not** certified stable
standing, contact quality, recovery from all disturbances, or hardware safety.
Record endpoint velocities/tilt/contact/torque diagnostics without pretending
that a settling-quality threshold has been calibrated.

Only pre-switch VAE rows from the first 125 trial steps (or fewer at clip end)
are candidate training data. Keep the remaining qualification trace separately;
no PPO/verification padding or policy-switch-crossing window is allowed. An early
clip end can shorten available data without making the continuous hybrid episode
a failure. Publish the actual VAE duration, valid-window count, phase coverage,
and rejection reasons. Split rollout families before windows/statistics; paired
clean/OU variants and duplicate initial-state/phase groups stay together. If a
split lacks eligible data, report that gap rather than leaking groups or silently
collecting replacements.

### 11.2 Bounded D1 pilot design — not launched

- Three motions × three phase fractions (`0.1, 0.5, 0.8`, starting frame
  `floor(fraction*(N-1))`) × three seeds (`0,1,2`): **27 clean trials**, then
  **27 paired OU trials** from the same saved initial states.
- Five seconds maximum per trial: **54 attempts, 270 aggregate simulated seconds,
  13,500 control transitions**. Failed trials count against the cap.
- One GPU, at most eight environments, **15 minutes wall time including
  compilation/warm-up**, **512 MiB output**, no video. Use a separately allocated
  GenieStudio-managed resource and monitored execution with a 120 s inactivity
  bound; no fleet use or remote launch is authorized by this design.
- Stop immediately on schema, parity or data-integrity failures. Before OU,
  require each motion to have a clean qualified training window and a successful
  endpoint handoff. These are coverage/integration prerequisites, not a new
  broad VAE quality gate. Ordinary physical/tracking failures reject that trial
  and remain in the report; no silent retry-to-success or bulk 100-fold collection.
- Budget exhaustion produces a partial report. A new budget requires explicit
  approval; D0 closeout does not authorize this pilot or training.

### 11.3 Fresh CPU closeout checks

Evidence: `logs/diffusion/d0-finalization-20260929/` contains `check_contract.py`,
raw JSON/log, projection fixture, rerun command and SHA256 index. CPU only; no
simulator, ONNX inference or CUDA execution was started.

| Check | Result |
|---|---|
| YAML parsing, body ordering, 41/135/199/231 contract invariants and pilot budget arithmetic | PASS |
| Pinned artifact/source hashes | **23 verified** (includes source-hash index and referenced artifacts) |
| All three saved tracking thresholds and standing 70-degree threshold | PASS (source/config identity, not rollout behavior) |
| Frozen schedule agrees with corrected prior CPU evidence | PASS; terminal alpha `2.4287669070348542e-09` |
| Exactly 20 DDIM jumps, analytic skipped-step oracle | Max absolute error **`2.22e-16`** (gate `<1e-12`) |
| Score correction ↔ x0 correction at fixed illustrative gradient | Max absolute error **`1.87e-12`** (conversion check `<1e-9`); not an input-VJP/cost test |
| Seed-42 199×135 projection/pseudoinverse roundtrip | Max absolute error **`1.64e-14`** (gate `<1e-10`) |

Projection fixture SHA256:
`76d11ed848f6380deefa4a14b622f4b33a2333025d6bb6c71f14565a1e13c5b3`.
The fixture uses NumPy 2.5.1 and no fitted dataset statistics. D1 persists its
actual train-only statistics/matrices and hashes before data/model consumption.

### 11.4 Gate reconciliation — not relabeling untested work as PASS

| Item from the original D0 checklist | Disposition / owning phase |
|---|---|
| Candidate selection, endpoint role, numeric defaults, qualification thresholds, pilot budget | **Closed design decisions** in frozen v1 contract |
| New simulator baseline solely to requalify the unchanged selected VAE | Not required for D0 closeout; use accepted evidence and user-confirmed hardware validation |
| Collector action replay, actual last-frame handoff, finite/terminal/reset tests and qualification pilot | **D1 prerequisites**, not run; FP32 same-backend replay `atol=rtol=1e-5`, zero leakage/nonfinite accepted rows |
| Production state/frame/projection/noising/mask/DDIM and input-VJP tests | **D1/D2/D4**, not replaced by these audit formulas; FP64 frame roundtrip `<1e-10`, clean values/IDs exact |
| Closed-loop diffusion quality tolerances against matched VAE initializations | Preregister in **D3/D4 before comparisons**, rather than fabricating results or gates for an untrained model |
| FP32/BF16/export parity, sustained guided/unguided 20 ms timing, stale-plan safety | **D5 hard acceptance gates**, still unproved; no silent rate/horizon/backbone/step reduction |
| New diffusion-controller hardware qualification | **D6**, separately authorized; not inherited from the VAE's hardware validation |

The user-approved adaptation is therefore **D0 design/audit closed with runtime
risk accepted**, not "50 Hz validated". No remaining D0 owner decision is being
left implicit. **D1/D2 are not started and no execution authority is granted.**

## 12. Historical subagent attestation

The original attestation is retained below for provenance. Its open-decision
statements describe the pre-closeout audit and are superseded only by section 11;
it is not an independent acceptance gate.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Corrected the same D0 report and probe evidence: canonical 1001-entry schedule, exactly 20 DDIM jumps/calls, bounded cosine-beta assertions, real noise shape/storage assertions, corrected CUDA timing with frozen parameters and no timed CUDA scalar conversions, fresh source hashes, superseded raw evidence, and an explicit PASS/FAIL/NOT CHECKED checklist."
    }
  ],
  "changedFiles": [
    "docs/plans/beyondmimic_diffusion_d0_validation.md",
    "logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.py",
    "logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.py",
    "logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.json",
    "logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.json",
    "logs/diffusion/d0-validation-20260929T062344Z/SHA256SUMS"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "uv run --no-sync python logs/diffusion/d0-validation-20260929T062344Z/corrected_cpu_probe.py",
      "result": "passed",
      "summary": "1001-entry bounded cosine-beta schedule; 20 source IDs/calls and 20 jumps ending at clean 0; DDIM oracle error 2.22e-16."
    },
    {
      "command": "timeout 120 uv run --no-sync python logs/diffusion/d0-validation-20260929T062344Z/corrected_gpu_probe.py",
      "result": "passed",
      "summary": "41x231 untrained RTX 5080 probe; 20 corrected DDIM calls, frozen parameters, CUDA-event median 21.4317 ms unguided and 80.9590 ms one-forward VJP proxy."
    },
    {
      "command": "sha256sum candidate and source artifacts listed in fresh_source_hashes.txt",
      "result": "passed",
      "summary": "Fresh hashes recorded and reconciled with embedded cohort hashes."
    },
    {
      "command": "git status --short --branch; git rev-parse HEAD",
      "result": "passed",
      "summary": "Current HEAD 107fe65bff995186ae7d61c9b1632a1f1947caf0; shared changes preserved; no commit or staging performed."
    }
  ],
  "validationOutput": [
    "Canonical corrected grid: source 1000..50 (20 nonclean IDs), destination 950..0, exactly 20 jumps/calls, clean index 0, 1001 embeddings.",
    "Bounded cosine-beta CPU result: alpha_bar[1000]=2.4287669070348542e-09, terminal SNR=2.428766912933763e-09.",
    "Corrected GPU actual parameter count 20,197,607 at token width 231 and 1001 timestep embeddings.",
    "Corrected GPU raw timing: unguided wall median 21.4394 ms; one-forward input-VJP proxy wall median 80.9748 ms; no p95/p99 claimed.",
    "Fresh source hashes and superseded-probe hashes are indexed in logs/diffusion/d0-validation-20260929T062344Z/SHA256SUMS."
  ],
  "residualRisks": [
    "D0 remains open: candidate freeze, endpoint policy, schedule bounds, projection, guidance semantics, statistical independence, and tolerances require parent decisions.",
    "No fresh decoder action/parity, runtime frame/estimator, simulator baseline, sim2sim, hardware, or general-tennis qualification.",
    "Corrected timings are untrained tensor feasibility evidence, not end-to-end 50 Hz/p99 certification."
  ],
  "noStagedFiles": true,
  "diffSummary": "Corrected the existing D0 report and added only the requested ignored evidence bundle; no production implementation or training artifacts.",
  "reviewFindings": [
    "The prior 19-jump, inconsistent-grid and inefficient-DDIM evidence is explicitly superseded and preserved for traceability.",
    "No blockers to this corrected bounded audit; D0 itself remains not complete."
  ],
  "manualNotes": "Parent must review the corrected evidence and resolve the remaining owner decisions; this report does not self-approve D1/D2 or the sampler design."
}
```
