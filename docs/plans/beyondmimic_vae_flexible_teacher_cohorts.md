# VAE flexible teacher cohorts — implementation plan

Status: **proposed; planning only**. No implementation, refactor, or training is
authorized or performed by this document, and it changes no currently committed
behavior. The open decisions in section 10 are recorded, not resolved.

Repository: `/home/agiuser/projects/mjlab`.
Baseline: `fxy/test/tracking-exp`, HEAD `203cb6ff8dcef1c2b080e047e0128c78f7409918`
(the merge that carried the queue-untracking commit on top of the M4 and
standing-start milestones).
Related: [M4 implementation and acceptance](beyondmimic_vae_m4_implementation.md),
[standing-start extension](beyondmimic_vae_standing_start_implementation.md),
[architecture](beyondmimic_vae_distillation.md),
[usage](../source/x2_tennis_distillation.rst).

Notes:

- **A temporary hack is already in the tree**, taken deliberately before this plan
  was approved, to mix a standing-start variant teacher with plain-task teachers:
  `_EXCLUDED_ENV_FIELDS` in `distillation/config.py` now also excludes
  `commands.motion.standing_start_prob` and
  `commands.motion.standing_start_window_frames`, and
  `_without_excluded_env_fields` prunes arbitrary dotted paths. It is marked
  temporary in the source and is superseded by Stage 1, which replaces the fixed
  list with a selection-scoped, contract-scoped, recorded policy. The hack does
  **not** address `manifest_sha256` being part of the cohort identity, so a mixed
  cohort still needs its own manifest file
  (`configs/distillation/x2_tennis_mixed.yaml`) rather than being added to an
  existing one.
- All claims below were measured on this repository and its teacher artifacts on
  the GPU server (`ssh://fushan@10.14.64.37:31505/home/fushan/mjlab`); the
  supporting evidence, including the exact diffs and metadata dumps, is recorded
  in `~/.pi/agent/mjlab-flex-investigation/`.
- Two facts from current work set the context: a 10000-iteration two-teacher
  cohort run (`tennis_000`, `tennis_001`) is in flight, and a standing-start
  teacher for clip 002 is registered as `tennis_002_ss` from a frozen artifact
  directory, with the mixed cohort in
  `configs/distillation/x2_tennis_mixed.yaml`.

## 1. Goal, authority, and non-goals

Train one shared student VAE from **many** teachers that were not produced by a
single PPO task. Today the framework requires every teacher in a manifest to be
interchangeable with every other one, which makes teacher supply the binding
constraint on any general model: a viable teacher for a given clip frequently
requires its own task variant (different randomization, rewards, resets, or even
a different observation representation).

**User decision: this plan is saved for later. No stage is authorized yet.**
Teacher qualification stays external to the pipeline, exactly as recorded in the
standing-start plan: no rollout, success threshold, certificate, or retraining
prerequisite is added by any stage here.

In scope (documented for later approval):

- Separating semantic compatibility from training-provenance equality.
- Per-teacher observation views, so teachers with different actor observation
  layouts can label one student.
- Decoupling reference clips from teachers, so many clips no longer require many
  teachers, and one clip may have several.
- Teacher supply tooling: manifest generation, artifact freezing, per-teacher
  validation reporting.

Out of scope:

- Changing the student architecture, the canonical VAE schema
  (reference 68 / conditioning 99 / latent 32 / actions 31), or the diffusion
  interface.
- Relaxing action-space, control-timing, robot/body-mapping, or cadence
  compatibility (section 9).
- Retargeting across different robots or joint sets.
- Rewriting, converting, or invalidating existing checkpoint versions.

## 2. The problem, measured

Three complete teacher families exist on the server. They share the same robot,
the same 31-D action layout, the same joint names and scales, and differ only in
their actor observation composition:

| Family | Actor obs dim | Actor obs terms |
|---|---|---|
| `agibot_x2_tracking_correlated_dr_reduced_perturbations` (incl. `tennis_000/001`) | 164 | command, motion_lookahead, **motion_anchor_ori_b**, base_ang_vel, joint_pos, joint_vel, actions |
| `..._projected_gravity` | 161 | swaps that 6-D term for **projected_gravity** (3-D) |
| `..._projected_gravity_anchor` | 167 | 164 plus projected_gravity |

Diffing a gravity-anchor teacher against `tennis_000`, the semantic difference is
**one extra observation term**. The remaining differences are training
provenance: `seed` (env and agent), `experiment_name`, `run_name`, and
`commands.motion.motion_file` (already excluded). `agent.yaml` differs only in
those provenance fields.

The student's own input is already teacher-independent: `ObservationSnapshot`
(`distillation/observations.py`) packs a fixed canonical schema from raw named
features. Teacher heterogeneity therefore does not touch the student's input
space — it only decides which frozen network produces a row's action labels. The
rigidity is that the code additionally requires a single shared **teacher-facing**
observation vector (`obs["actor"]`, asserted as `[B, 164]`) and one
`(obs_dim, action_dim)` for the whole teacher bank.

## 3. Current enforcement surface

| # | Invariant | Enforced at | Assessment |
|---|---|---|---|
| 1 | Manifest entry = 5 artifacts + weight; unique ids | `config.py:86-207` | Keep |
| 2 | Every entry's artifacts exist | `config.py:205-216` | Keep (but see 4.3) |
| 3 | **All** manifest entries mutually compatible | `resolve_cohort` → `_require_equal_saved_configs(manifest.teachers)` `config.py:507-520` | Incidental: should scope to the selection |
| 4 | Saved env configs equal on the union of keys except `motion_file` | `config.py:38, 1323-1346` | Incidental: far stricter than the contract |
| 5 | Resolved contract equal (actor class/hidden dims/activation/normalization/groups/dims, observation names/widths/corruption, action term/joint names/scales/offset/default-offset, sim timestep/decimation, reference fps/joint dim, anchor body, body names, lookahead, sensors) | `_require_same_contract` `config.py:1263-1321` | **Fundamental**: make it the mandatory guard |
| 6 | Agent equality limited to `obs_groups` + `actor`; the rest is provenance | `config.py:1334-1341` | Precedent for provenance-only fields |
| 7 | Live env actor obs order/dims/term policy equals the cohort's single expected contract | `_validate_observation_contract` `adapter.py:355-...` | Becomes per-teacher (Stage 2) |
| 8 | One shared teacher-facing observation vector, literal `[B, 164]` | `adapter.py:1163-1166` | Incidental: remove the literal |
| 9 | All teachers share `(obs_dim, action_dim)` | `TeacherBank.__init__` `teachers.py` | Incidental for obs, keep for action |
| 10 | 1:1 clip ↔ teacher by position and code | `_require_clip_teacher_mapping` `adapter.py:652-676`; `motion_library.py:200-212` | Incidental: the clip/teacher conflation |
| 11 | Slots by weight over clips; replay quotas per clip | `multi_motion.py`, `balanced_storage.py` | Keep; reuse for clips |
| 12 | Cohort identity (member digests, common contract, slots, replay); strict resume | `cohort_contract.py` | Keep; extend, never mutate |
| 13 | Checkpoint v1/v2/v3 never converted; standing provenance | `checkpoint.py` | Keep; add v4 if needed |

## 4. Fundamental versus incidental

### 4.1 Fundamental for one shared student

- **Action space**: dimension, joint names and order, scales, offset,
  `uses_default_offset`, clipping semantics. The student's decoder emits one
  layout and labels must live in it.
- **Control semantics**: sim timestep, decimation, control period/Hz. Labels are
  per control step.
- **Robot and body mapping**: robot, anchor body, tracked bodies, joint names —
  these feed the student's own observations.
- **Reference cadence**: fps, lookahead, joint dimension.
- **Actor interface sanity**: ONNX ↔ checkpoint association, observation
  normalizer divisor consistency, deterministic (mean) labels.
- **Routing correctness**: a row's labels come from the teacher that owns its clip.
- **Strict resume** and per-member artifact digests.

### 4.2 Incidental rigidity

- Whole-saved-config equality (item 4): it also compares randomization, rewards,
  events, terminations, resets, and seeds, none of which change what a teacher
  computes as a function of observations.
- Manifest-wide validation (item 3): one incompatible entry refuses every run
  from that file, including selections that never mention it. This was observed
  directly: adding a standing-start teacher to `x2_tennis.yaml` made
  `--teacher-ids "('tennis_000','tennis_001')"` fail with
  `commands.motion.standing_start_prob is present in only one configuration`.
- Teacher-facing observation identity (items 7, 8, 9), including the literal 164.
- Clip/teacher conflation (item 10): N clips require N teachers; one teacher
  cannot serve many clips, and one clip cannot have several teachers.

### 4.3 Two hazards the plan must handle

- **Live artifact drift.** A training run re-exports its ONNX on every periodic
  save. Pointing a manifest at a live run directory changes the artifact digests
  that cohort identity records, which makes earlier students un-resumable and
  un-evaluable. Observed while freezing `tennis_002_ss`. Mitigation: freeze
  accepted teachers into stable directories with a provenance/ digest record
  (Stage 4).
- **Label cleanliness.** A teacher trained with observation corruption or noise
  must still be labeled with clean observations. Today this is protected by the
  `enable_corruption` equality check; per-teacher views (Stage 2) must preserve
  the property explicitly.

## 5. Stage 1 — selection-scoped validation and a recorded equality policy

Motivation: unlock same-contract teachers that differ in training-time
configuration, and remove the "one bad entry breaks the file" hazard.

Change surface:

- `config.py`: `resolve_cohort(manifest)` → `resolve_cohort(manifest, selected)`;
  `_require_equal_saved_configs` operates on the selection.
- `config.py`: replace `_EXCLUDED_ENV_FIELDS` (one hard-coded path) with a
  declared policy: a strictness flag plus an explicit ignore-path list, defaulting
  to contract-bearing paths only.
- Record the effective policy in the cohort identity so a run cannot silently
  change strictness between resume and evaluation.
- `_require_same_contract` (item 5) stays **mandatory and unchanged**.

Invariants preserved: everything in 4.1, plus the recorded policy.

Risks: a too-permissive default would admit teachers whose *inference* contract
differs. Mitigation: keep the resolved-contract check mandatory, compare the
policy digest on resume, and require the ignore list to be explicit per manifest.

Verification: extend `tests/test_tracking_distillation_manifest.py`
(`test_resolve_cohort_rejects_differing_saved_environments` becomes a
policy-parameterised test; add a case proving a provenance-only difference is
accepted and a contract difference is still refused), plus a cohort-CLI test that
an unselected incompatible entry no longer breaks a valid selection.

## 6. Stage 2 — per-teacher observation views

Motivation: mix teachers whose actor observation composition differs (the
projected-gravity and gravity-anchor families already on the server, plus the
mixed-task case generally).

Change surface:

- Build each teacher's observation from the observation manager's term system
  rather than one cached `obs["actor"]`: compute the union of required terms once,
  then materialise each teacher's vector with its own order, scale, history, clip,
  and flatten-history policy. The specification already exists per teacher in its
  own ONNX metadata (`observation_names`, `observation_terms_scale`,
  `observation_terms_history_length`, `observation_terms_flatten_history_dim`,
  `observation_terms_clip`).
- `teachers.py`: `TeacherBank` accepts heterogeneous `obs_dim` per code; labeling
  is already grouped per unique code, so cost scales with the teachers present in
  a batch, not with cohort size. Keep `action_dim` shared.
- `adapter.py`: remove the literal `[B, 164]` assertion and the single-snapshot
  teacher path.
- `cohort_contract.py`: record each member's observation specification and digest.
- `config.py`: items 5 and 7 become per-teacher capability checks (each teacher's
  view must be constructible from the live environment) instead of cross-teacher
  equality.
- `environment.py`: expose or construct the per-teacher actor observation groups.

Invariants preserved: shared action space, control timing, bodies, cadence,
routing, clean labeling.

Risks: view construction bugs silently shift a teacher's inputs. Mitigation: a
numeric parity test per teacher comparing the constructed view against the
teacher's own recorded metadata, and per-teacher reports that state the view used.

Verification: new tests for heterogeneous obs dims in the bank; a parity test that
a gravity/anchor teacher's view reproduces its ONNX-declared term layout; an
integration test that a three-family cohort labels rows with the correct teacher
and view.

## 7. Stage 3 — clips decoupled from teachers

Motivation: the structural enabler for a general model over many motions. Today
positional motion ids and `_require_clip_teacher_mapping` force N clips ⇒ N
teachers, so every new clip requires a matching teacher.

Change surface:

- Manifest schema: separate `clips:` (motion id, file, fps, frames) from
  `teachers:`, plus an explicit routing map (clip → teacher(s), with weights or
  tiers).
- `motion_library.py`: clip table independent of teacher codes; derive positional
  codes from the routing map so the "wrong teacher labels the row" guard survives.
- `multi_motion.py`: slot allocation per clip, teacher routing per row.
- `balanced_storage.py`: replay quotas per clip.
- `cohort_contract.py` + `checkpoint.py`: cohort identity **v4** recording clips,
  routing, and per-member views; v1–v3 remain loadable and unmodified.
- Reports and evaluation: per-clip attribution stating which teacher labeled it.

Invariants preserved: action/control/body/cadence compatibility, per-row routing
correctness, balanced coverage, strict resume within a version.

Risks: identity and resume semantics across versions. Mitigation: a new version
rather than mutation, and explicit refusal of cross-version resume.

Verification: manifest/schema tests; a plan test where two clips share one teacher
and one clip has two teachers; a resume test proving v4 refuses a v3 checkpoint
with a clear message.

## 8. Stage 4 — teacher supply pipeline

Motivation: 50+ teachers are unmaintainable by hand, and live artifact drift
breaks reproducibility (4.3).

Change surface (tooling, mostly outside the trainer):

- Manifest generator: scan `logs/rsl_rl/**` for a complete tuple (checkpoint +
  `params/env.yaml` + `params/agent.yaml` + ONNX export), refuse partial ones.
- Artifact freezing: copy accepted teachers into a stable directory with a
  provenance file recording source, iteration, mtimes, and sha256 digests.
- Per-teacher validation report: a command producing a machine-checkable record of
  observation specification, action layout, body mapping, cadence, ONNX ↔
  checkpoint association, and native-vs-ONNX parity — replacing the status quo
  where this lives only in tests and parent gates.
- Sizing guidance recorded with the run: slots need at least one row per clip;
  replay quotas need at least one slot per clip; bank memory is roughly 7 MB per
  teacher; the practical limit is samples per clip, not the framework.

Verification: generator refuses incomplete tuples; frozen digests reproduce across
machines (local mirror verified byte-identical for `tennis_002_ss`); the validation
report matches the existing parity test results for `tennis_000/001`.

## 9. What must not be relaxed

- Action space, control timing, robot/body mapping, reference cadence.
- ONNX ↔ checkpoint association and deterministic labeling.
- Strict resume and per-member artifact digests; new identity versions rather than
  mutation of existing ones.
- The canonical student schema and the diffusion interface.
- Teacher qualification remains user-owned and external.

## 10. Open decisions (recorded, not resolved)

1. Target shape: one student over many clips (Stage 3) or several students?
2. Which teacher-facing observation variants must be supported (baseline /
   projected gravity / gravity+anchor / others)?
3. Are all candidate teachers the same robot and action space? If not, is explicit
   action-space projection acceptable, or must the framework refuse it?
4. Compatibility strictness as a recorded per-run policy (recommended) or a global
   default?
5. Add a v4 cohort record for clips and routing, keeping v1–v3 loadable?
6. May we assume inference-time observation corruption is always off for labeling?

## 11. Evidence and reproduction

- Flexibility evidence directory: `~/.pi/agent/mjlab-flex-investigation/`
  (`FLEXIBILITY-ANALYSIS.md`, `inventory-families.sh`, `diff-families.sh`).
- Teacher family inventory and saved-config diffs: `inventory-families.sh` and
  `diff-families.sh` on the server.
- Standing-start teacher registration and freezing, including the refusal text for
  mixing families: `~/.pi/agent/mjlab-server-standing-10k/`
  (`inspect-ss-teacher.sh`, `freeze-ss-teacher.sh`, `verify-ss-registration.sh`),
  and the frozen artifacts with digests in
  `logs/distillation/teachers/tennis_002_ss/PROVENANCE.md`.
