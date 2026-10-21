# Keyframe inpainting for the x0 flow family — implementation plan

Status: proposed 2026-10-17 (+08). Repo `mjlab`, branch `fxy/test/flow-matching-baseline`.
Scope: **mjlab D3/D4 tooling only.** No controller (C++) work, no retraining, no quality claim.

## Goal

Make `--task-config` with **hard keyframes** runnable for `--model-family flow`. A hard keyframe is
a plan rewrite that fixes the 135 normalized physical-state channels of one future row and leaves
that row's 32 latent entries to the sampler. Nothing in that operation is diffusion-specific.

## What already exists (verified in the tree, not assumed)

| Piece | Where | Family-specific? |
|---|---|---|
| Hard-keyframe plan rewrite | `inpainting.inpaint_keyframes`, called from `conditioned_continuation.py:422` inside `TaskConditioning.plan_transform` | no: it takes a `ContinuationPlan` and edits state channels |
| Per-tick task hook on the seam | `handover.py:380` `plan_transform` and `sampler_factory` parameters on `ReferenceFreeContinuation` | no: the flow run uses the same continuation |
| Flow sampler injected into the seam | `flow_matching/closed_loop.py::FlowSamplingAdapter`, typed `FamilySampler` (`handover.py:298`) | this is the family sampler |
| Soft pose goals | `PoseCost` + `GuidanceConfig` + guided sampler at `conditioned_continuation.py:454-470`, which requires `schedule: DiffusionSchedule` | **yes**: DDIM only |
| Blanket CLI refusal | `evaluate.py:584` | this is policy, not numerics |

**The one real blocker.** `TaskConditioning._unguided_sampler` (`conditioned_continuation.py:490`)
does not delegate to the seam's sampler. It calls the DDIM `sample_trajectory(...)` directly with
`schedule` and `config: SamplerConfig`. So even a keyframe-only task config would replace the flow
sampler with DDIM, which is wrong. That is the work item; the CLI guard is the second, trivial one.

## Non-goals

- No retraining and no change to the model. The training mask stays prefix-only.
- No soft pose goals for flow. They stay refused, with a message that names the reason.
- No controller (D5 C++) change. The deployed `KEYFRAME_INPAINTING` request mode needs a source of
  fixed future rows (a reference clip), which a flow bundle does not carry; that is separate work.
- No claim that flow inpainting is *good*. It is unmeasured, and the model never saw a clean future
  state row during training.

## Tasks

### T1 — let the task lane use the family sampler

`conditioned_continuation.py`:

- Add an optional `delegate_sampler: FamilySampler | None = None` to `TaskConditioning`.
- In `_unguided_sampler`, when a delegate is present, call it with the built `conditions` and the
  generator instead of `sample_trajectory`. Both diagnostics carry `denoiser_calls`, so the task
  record logic stays as it is.
- Diffusion behaviour is unchanged when no delegate is passed, and `evaluate.py` passes none for
  diffusion.

Acceptance: a unit test proves the delegate is called exactly once per tick and that
`denoiser_calls` reaches the task record.

### T2 — narrow the CLI guard

`evaluate.py`:

- Parse the task spec **before** the family guard (the guard currently runs before any parse, so this
  is a deliberate ordering change and must be stated in the diff).
- Refuse `flow` only when the spec has soft goals or a positive strength:
  "soft pose goals need the DDIM guided sampler; a flow run supports hard keyframes only".
- Allow `flow` with keyframes only, and pass the flow adapter as `delegate_sampler`.

Acceptance: `--model-family flow --task-config <keyframes-only>` proceeds past the guard;
`<goals>` is refused with that message; diffusion is unchanged.

### T3 — prove it on a real bounded run

One closed-loop flow run with a keyframe-only task file, existing flow checkpoint, CPU or GPU:

- the returned plan row carries the requested state values bit-exactly;
- its latent entries are not fixed (`inpaint_keyframes` refuses a fixed latent row, so a violation
  would raise);
- provenance reports `sampler_method: euler_x0`, not `ddim`, which is the proof that the family
  sampler served the tick;
- the task record reports `denoiser_calls` from the flow sampler.

Acceptance: the run completes and those four facts are readable from its output.

### T4 — tests, critical only

1. Delegate sampler is used for a keyframe-only spec (T1).
2. Flow plus goals is refused; flow plus keyframes is allowed (T2).
3. A keyframe rewrite on a flow-produced plan fixes only state channels and preserves every existing
   fixed entry, reusing the shape of the existing diffusion test.
4. The existing diffusion task-guidance tests still pass unchanged.

### T5 — docs

- `docs/source/x2_flow_matching_training.rst`: replace the "flow rejects `--task-config`" paragraph
  with the hard-keyframes-only rule and the soft-goals refusal.
- `docs/source/changelog.rst`: one entry.
- State the unmeasured quality limit and the prefix-only training mask in both places.

## Risks

| Risk | Control |
|---|---|
| The task record or `_validate_tick_record` assumes DDIM diagnostics | Diff both samplers' diagnostic fields first; if a field is missing, report it instead of inventing one |
| Moving the guard changes when the task file is validated for diffusion | Keep the diffusion parse order identical; add the parse earlier only for the spec's shape |
| The flow plan's row layout differs from F4's | Assert `state_dimension == 135` and `latent == 32` on the flow plan before the rewrite |
| A keyframe is out of distribution for a prefix-only-trained model | Declared, not fixed, here. The run measures whether the plan respects the keyframe; it does not measure stability |

## Acceptance

Build clean, the four test items above, and one real keyframe-only flow run whose plan carries the
fixed row and whose provenance says `euler_x0`. Nothing is pushed.

## Estimate

Three code edits (one function, one guard, one wiring site) plus tests. The long part is the run and
reading its output, not the code.

## Status 2026-10-17 (implementation)

**T1, T2 and T4 are done. T3 is blocked by the task-preparation path, not by the sampler.**

Implemented:

- `handover.py`: `sampler` and `sampler_factory` are no longer mutually exclusive. When a family
  sampler is injected, a factory result replaces it for that tick and takes the family calling
  convention `(conditions, generator)`. The diffusion arguments stay refused.
- `conditioned_continuation.py`: `TaskConditioning(delegate_sampler=...)`. The unguided path calls
  the delegate instead of the DDIM `sample_trajectory` and still records `denoiser_calls`. A
  delegate combined with soft goals or a positive strength is refused, because those need
  `PoseCost` over a diffusion schedule.
- `evaluate.py`: the blanket flow refusal is replaced by `_require_task_family_support`, which
  refuses flow only for goals or a positive strength. The flow continuation now receives
  `conditioning=` and the adapter as `delegate_sampler`.
- Tests: a delegate serves the tick and the keyframe fixes only the state channels while the latent
  slice is untouched; a delegate refuses soft goals; the two tests that pinned the old blanket
  refusal now pin the narrowed rule.

Test state: **107 passed** across the handover, inpainting, flow continuation, flow integration and
D4 suites. The same **4 D4 CLI tests** fail with and without this change (`test_malformed_task_config_…`,
`test_valid_task_config_reaches_preflight_…`, `test_load_time_gates_…`,
`test_provenance_and_preflight_…`); all four fail at HEAD too, with the one cause being that
`_validate_request` now requires the checkpoint sidecar before any task validation. That is
pre-existing and out of scope here.

### T3 completed (2026-10-17 18:35 +08)

The end-to-end run works. Two producer fixes were needed, and neither weakens a claim:

1. The run's `vae` provenance did not record the VAE **file** digest, and the task checker compares
   the contract's `vae.sha256` against it. The block's `member_artifact_hashes.checkpoint` is the
   **cohort member's teacher** checkpoint, not the VAE, which is why the two never matched. The run
   now writes `vae.sha256 = sha256_file(vae checkpoint)`.
2. A flow sidecar records its contract as a **path**, not an identity hash, so the flow run's
   `checkpoint` provenance carried no `contract_identity`. The flow run now reports the identity of
   the contract it loaded from that sidecar and validated the checkpoint's data identity against.

Result, `logs/diagnostics/flow-keyframe-20261017/run3` (vgk000, seed 0, frame 45, 20 plans,
keyframe at tick 12, task file `qualification: verified`):

| Fact | Value |
|---|---|
| Sampler for every plan | `euler_x0`, 6 updates, **120 denoiser calls** (20 × 6) |
| Keyframe applied | **`keyframe_steps: 4`** plans |
| Soft-guidance calls | `vjp_calls: 0`, `cost_calls: 0`, `guided_steps: 0` |
| Realized root position error at tick 12 | **2.29 mm** |
| Realized root orientation (chordal) | 0.0202 |
| Realized body positions | **RMS 3.67 mm**, max 7.99 mm |
| Unreached / missing / expired targets | none |

So the flow family serves a hard keyframe through the task lane, keeps its own sampler, and the
measured state reaches the requested pose. The task file claims `verified`, with the loader's
recheck list in its `source.verification`.

### T5 pending

`docs/source/x2_flow_matching_training.rst` and `docs/source/changelog.rst` still say flow rejects
`--task-config`. Both files carry uncommitted edits from another session in this checkout, so this
change deliberately leaves them alone rather than sweeping them into this commit.
