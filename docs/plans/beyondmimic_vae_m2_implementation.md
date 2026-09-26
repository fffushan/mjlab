# BeyondMimic M2: gravity-first latent policy core

Status: **implemented and parent-accepted on 2026-09-26** after one scoped correction pass. M3 and training remain unauthorized; see the acceptance record below.
Repository: `/home/agiuser/projects/mjlab`.
Worker model: **`lingzhi/gpt-5.6-luna`**. Parent supervises every 15 minutes and owns acceptance.
Depends on the accepted [M1 teacher foundation](beyondmimic_vae_implementation.md).
Architecture context: [VAE distillation design](beyondmimic_vae_distillation.md).

## 1. Scope and user decision

Implement the pure tensor/model/storage core needed by M3, with **projected gravity as the default decoder orientation input**. Make anchor-only and gravity-plus-anchor conditioning explicit, cheap extensions; do not train any ablation yet. Preserve the existing frozen teachers and their 164-dimensional inputs.

In scope: named feature/schema packing, conditional VAE, controlled student normalization, reconstruction/KL loss, bounded raw labeled replay, CPU unit/integration tests, and documentation.

Not in scope: live simulator adapters, stepping or rollouts, DAgger collection, a training runner/CLI, optimizer checkpoint/resume infrastructure, motion-library changes, M3+, diffusion, symmetry, student ONNX exports, GPU jobs, remote jobs, or hardware execution. Small synthetic CPU gradient/optimizer-step tests are verification, not permission for policy training.

## 2. Input contract and extension point

The encoder stays `reference q(31) + reference dq(31) + motion_anchor_ori_b(6) = 68`. Reference q is not the relative joint-position proprioceptive field. Anchor orientation is the existing robot-anchor-to-reference-anchor relative rotation, packed exactly like `mat[..., :2].reshape(B, -1)` (first two matrix columns, interleaved by the existing reshape).

Common decoder conditioning is `gyro(3) + relative joint q(31) + joint dq(31) + previous executed action(31) = 96`.

| Explicit decoder mode | Additional orientation features | Conditioning width | Width including latent(32) | Reference-conditioned decoder? |
| --- | --- | --- | --- | --- |
| `gravity` (default) | Root projected gravity(3) | 99 | 131 | No |
| `anchor` (opt-in future experiment) | Anchor orientation error(6) | 102 | 134 | Yes |
| `gravity_anchor` (opt-in future experiment) | Gravity(3), then anchor orientation error(6) | 105 | 137 | Yes |

Implement the three small schema/packing choices and shape tests now; production/default configuration remains gravity. They are separate model/schema identities, not runtime toggles on a trained checkpoint. Do not put zero-filled placeholders for reference features in the default gravity decoder or infer a mode from tensor width.

Use ordered, named fields with a versioned serializable schema recording mode, joint order, dimensions, declared frames, and reference-conditioning status. Gravity refers to the root frame, gyro to the teacher-compatible declared IMU frame, anchor error to the anchor frame. Physical IMU/site verification remains unresolved until M3; metadata must say declared/unverified rather than claiming deployment fidelity. Do not replace the teacher gyro with a torso sensor silently.

The M2 assembler consumes an already captured named tensor snapshot. It does not query the simulator, draw noise, advance delays/history, or mutate inputs. Overlapping teacher/student measurements will come from the same snapshot in M3; document that caller obligation. Student inputs and replay remain before learned student normalization. Prior action means actually executed normalized action, not a counterfactual teacher output. Teacher/motion IDs are routing/storage metadata only.

Suggested shared API (final names may be refined by the first component owner and documented in its handoff):

- `vae_config.py`: `DecoderMode`, model settings, and immutable/versioned student schema/layout metadata.
- `observations.py`: named feature snapshot, packed observation batch (`reference`, `conditioning`), pure validated packing.
- `model.py`: VAE, explicit-update normalizers, and loss terms/function (a small extra normalizer/loss module is allowed only if clarity warrants it).
- `storage.py`: bounded labeled replay accepting packed raw observations and labels/metadata.

The model must obtain dimensions from the schema, not hard-coded slicing. The selected X2 defaults above are acceptance targets; there is no requirement for a generic cross-robot framework.

## 3. Conditional VAE and normalization

Default encoder and decoder hidden layers: `[2048, 1024, 512]`, ELU. Latent dimension: 32. Encoder returns `mu, logvar`; sampled latent is `mu + exp(0.5 * logvar) * epsilon`. Expose deterministic mean inference and explicit sampled inference, with a controlled RNG/noise injection path for tests. Make the sampling choice explicit, not an accidental consequence of `.train()`.

Decoder accepts only latent plus the selected packed conditioning. It outputs 31 raw normalized actions with no new tanh, clipping, teacher ID, motion ID, or hidden reference branch. In gravity mode the reference/anchor error can affect actions only through the latent. Anchor-containing modes are clearly marked reference-conditioned and will require an explicit orientation command/error source in a future diffusion pipeline.

Use separate student normalization statistics for encoder features and decoder conditioning; never normalize `z` as part of proprioception and never reuse/update frozen teacher normalizers. Statistics update only through an explicit API on raw training samples. Forward/encode/decode calls do not update statistics even in train mode. Provide explicit freezing for evaluation/export; register all moment/count state as module buffers and preserve it through `state_dict` and device/dtype moves. Reject nonfinite updates without partially corrupting statistics. Document variance and epsilon conventions; avoid automatic normalization/clipping behavior hidden in the forward path.

Model state roundtrip must preserve behavior, normalizer state and schema identity. At minimum provide serialization/compatibility checking of schema/version/mode/joint order so a future checkpoint reader cannot silently load a same-width but different schema. Full optimizer/RNG/training checkpoint management belongs to M3.

Use configurable small hidden sizes for tests; instantiate the paper-size default once for a small CPU forward/shape check. No broad architecture registry or recurrent models.

## 4. Loss contract

For each sample, sum squared action errors over joints and sum Gaussian KL over latent coordinates, then average each over the batch:

```text
reconstruction = mean_B(sum_J((predicted_action - teacher_action)^2))
kl = mean_B(-0.5 * sum_Z(1 + logvar - mu^2 - exp(logvar)))
total = reconstruction + beta * kl
beta default = 0.01
```

Return total and both components for logging. Teacher labels are fixed targets. Reject shape mismatch, invalid beta, empty reduction batches and nonfinite inputs/results clearly instead of producing misleading NaN success. No hidden KL warmup, free bits, temporal loss, or latent clipping; escalate a stability change rather than quietly changing the baseline objective.

M3 will own Adam, LR `5e-4`, and accumulation count 15 with correct loss division. M2 does not implement a training loop. Synthetic tests may use an optimizer to prove gradients, a decreasing tiny-batch loss, and accumulation equivalence under fixed latent noise.

## 5. Bounded raw replay

Use a simple capacity-bounded tensor ring buffer with explicit positive capacity and schema. Store detached owned copies of raw packed reference/conditioning, teacher actions, and integer motion ID, teacher ID, reference frame, episode/segment ID, collector iteration metadata. Never store cached latents or autograd graphs. No full trajectories or next states are required.

Provide batch insertion and reproducible uniform minibatch sampling. State sample-with/without-replacement and empty-buffer behavior explicitly. Handle empty inserts, partially filled buffers, wraparound and insert batches larger than capacity deterministically; retain the most recent capacity records for the initial FIFO policy. Sampled records must not alias mutable buffer storage. Validate the complete incoming batch before mutating storage.

Keep replay independent of a live model/optimizer/simulator and of private checkpoint paths. Device/dtype behavior must be explicit, not silently transfer mixed inputs. Motion IDs stay available for M4 balanced replay; balanced sampling and recent/history mixture policies are deferred, not falsely claimed by this FIFO baseline.

## 6. Implementation topology and ownership

This is **multi-seam**: schema/packing, probabilistic model/loss, and replay each have independently testable contracts. Do not assign all outcomes to one unconstrained writer. M1 was committed at the user's request as `cdd9a8201213206d620a4f32b9125f7b0a9a2dc5`. The checkout still contains the new parent-owned M2 plans and unrelated user changes, so it does not satisfy managed worktree clean-source preconditions; do not stash, reset, clean, or commit those unrelated files to enable isolation.

Use **serial component ownership with exclusive time windows in the existing checkout**, not concurrent writers. Each component produces a durable runtime-managed report and passes its focused gate before the next starts. All component workers use the requested Luna model. A separate integration-only worker runs after all three handoffs. The parent remains read-only during writer activity and reviews after terminal handoff.

| Lane | Exclusive edit ownership | Inputs / dependency | Gate / durable handoff |
| --- | --- | --- | --- |
| `m2-schema` | New `vae_config.py`, `observations.py`, `tests/test_tracking_distillation_observations.py`; an optional new core fixture helper | This plan and existing M1/observation conventions | Packing/order/default isolation/all three mode tests; final API and exact validation report |
| `m2-model` | New `model.py` (optional small normalization/loss module), `tests/test_tracking_distillation_model.py` | Completed schema report and source, read-only | Shapes, reparameterization, analytic loss, gradients, explicit/frozen moments, state/schema roundtrip, reference isolation |
| `m2-storage` | New `storage.py`, `tests/test_tracking_distillation_storage.py` | Completed schema report and source, read-only | Capacity/wraparound/ownership/metadata alignment/RNG/raw-data/schema tests |
| `m2-integration` | Package `__init__.py`, new `tests/test_tracking_distillation_core.py`, `docs/source/x2_tennis_distillation.rst`, Upcoming changelog entry; narrowly necessary cross-component wiring fixes | All three durable component handoffs | Synthetic raw-batch → replay → VAE → teacher-label loss → optimizer step; all M1/M2 regressions, type/lint and real M1 parity |
| Parent | This plan and architecture/milestone plans; final source review and acceptance | All handoffs and actual final code | Independent parent inspection/reruns; request scoped fixes when needed |

Every lane uses `/home/agiuser/projects/mjlab`, branch `fxy/test/tracking-exp`; dispatch HEAD is the accepted M1 commit `cdd9a8201213206d620a4f32b9125f7b0a9a2dc5`. The same physical checkout is temporally isolated: lane N must be terminal before N+1 can write. There is no concurrent mutation or separate worktree to merge. Integration may fix actual wiring defects but may not implement a missing major component or revise the selected architecture; escalate that instead. Component owners must ask before editing another owner's files.

Protect all existing M1 sources/tests/CLI/manifests (except integration's allowed additive public exports and docs), parent plans, `tools/compare_remote_tasks.sh`, other `tools/` work, original artifacts, robot assets, saved experiments, installed dependencies and lockfiles. No staging, commits, pushes, external uploads, child fanout, model fallback or execution-protocol fallback.

## 7. Validation, monitoring and stop conditions

All Python/tool commands use `uv run`; set `CUDA_VISIBLE_DEVICES=''` for tests/inference. CPU only. Keep synthetic fixtures/buffers small, avoid simulator fixtures, and limit numerical-library CPU threads if needed. Inspection 30 s, small probe 60 s, focused suite 120 s. Pytest debugging: `-x -vv -o faulthandler_timeout=30`. An unexpected timeout requires diagnosis, not an identical retry with a larger limit. Long work requires a parent-managed monitor rather than a shell background process.

Each component runs its focused tests plus changed-file Ruff/type checks. Integration runs:

- All `tests/test_tracking_distillation_*.py` and `tests/test_tracking_commands.py` (inspect collection/fixtures; no simulator construction).
- The known CPU export regressions in `tests/test_runner.py`, with `-k 'onnx or export_paths' --deselect tests/test_runner.py::test_export_policy_to_onnx`.
- Changed-file Ruff format/check, `uv run ty check src tests`, and targeted Pyright; separate existing repository-wide failures from new ones.
- Existing real `distill validate-teachers` for both tennis teachers, 64 samples and unchanged `atol=rtol=1e-5`, CPU. Do not modify binary artifacts to make this pass.
- Final diff/scope and original-file preservation checks against the parent baseline.

Parent supervision runs every **15 minutes**: inspect exact workflow/child status, latest bounded transcript and command evidence; answer supervisor questions or steer only when concretely stuck/drifting. Healthy work is not a reason to restart, launch a competing writer or remove the monitor. Native completion wakes the parent immediately. Stop the periodic implementation monitor only after terminal handoff to review/recovery, cancellation, or its explicit bounded lifetime.

On provider/runtime/tooling failure: stop the affected lane, preserve partial edits and exact error/run/ref information, notify the parent, and do not continue downstream or switch model/protocol silently. Acceptance requires actual code inspection and reproducible tests, not just child prose.

## 8. Definition of done

Default gravity schema is 68/99/32/31 with no decoder reference bypass; alternative conditioning schemas are explicit and correctly packed/tested without being trained. VAE gradients/loss/normalization and replay ownership/bounds have CPU evidence. M1 still passes unchanged. Documentation distinguishes implemented core from unimplemented collection/training and records deferred physical-frame/closed-loop checks. Parent accepts the final code and stops at the M3 boundary; no training run starts automatically.

## 9. Parent acceptance record — 2026-09-26

All four component stages were delivered using `lingzhi/gpt-5.6-luna`. A parent workflow progress-serialization error was recovered through the same protocol without rerunning the completed schema stage. Parent code review and direct CPU probes then identified six missed defects; the retained integration worker fixed them with regression coverage before acceptance:

- Normalizer updates no longer attach autograd graphs; forward remains input-differentiable.
- Normalization epsilon is persistent and restored by standalone/nested state dictionaries.
- Uninitialized normalization is zero-mean/unit-scale identity, not a 316x amplification; moments still update only explicitly.
- Valid empty replay inserts no longer establish an unspecified device/dtype policy.
- Requested device aliases are canonicalized, demonstrated with CPU `cpu:0`.
- Teacher actions are fixed/detached loss targets; prediction and latent gradients remain intact.

Final **parent-executed** checks:

- All `tests/test_tracking_distillation_*.py` plus `tests/test_tracking_commands.py`: **98 passed**, 82 ONNX deprecation warnings.
- CPU runner/export subset with the simulator-building test explicitly deselected: **12 passed, 6 deselected**.
- Changed-source/test Ruff formatting/lint, `uv run ty check src tests`, and targeted Pyright: pass. This is not a claim that repository-wide Pyright is clean; M1 pre-commit checks documented unrelated failures in ignored diagnostic/vendor logs and an existing test.
- Both real teachers validated from a different cwd with explicit repo root, 64 samples and unchanged `atol=rtol=1e-5`: action max differences **9.5367431640625e-7** / **1.5497207641601562e-6**. Original parameter/reference association still passes.
- Additional CPU assertions covered all six repaired behaviors, repeated backward, `weights_only=True` serialization with epsilon/frozen state, default decoder reference isolation, float64 module conversion, and finite sampled losses/gradients across **40 fixed cold-start seeds**.
- All three modes constructed and ran a tiny synthetic forward using the **real validated cohort joint names**. Thus the binding source is available; the pure-core defaults remain intentionally synthetic placeholders, and the live adapter is still deferred.
- Before parent acceptance-document edits, **29 protected baseline files** were byte/hash-identical. M1 HEAD stayed `cdd9a8201213206d620a4f32b9125f7b0a9a2dc5`, no files were staged, and unrelated `tools/` work was preserved.

Primary modules: `vae_config.py`, `observations.py`, `model.py`, `storage.py` under `src/mjlab/tasks/tracking/distillation/`. Usage, including binding `cohort.actions.joint_names`, is in `docs/source/x2_tennis_distillation.rst`.

M2 is accepted as a **pure tensor/model/replay core**, not a trained policy. No ablation training, simulator rollout, GPU/remote/hardware job, student export or M3 runner was executed. Physical sensor frames, live snapshot timing/action alignment and closed-loop performance remain M3/later gates. M2 changes remain uncommitted; only M1 was explicitly committed at the user's request.
