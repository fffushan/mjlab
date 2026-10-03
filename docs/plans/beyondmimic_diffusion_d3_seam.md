# BeyondMimic diffusion D3 — bounded handover seam (implemented slice)

Status: **bounded D3 seam implemented (2026-10-05); no closed-loop, simulator,
controller, export or hardware evidence exists.** This record documents only the
implemented offline/runtime seam and its tests. D3's measured acceptance gates
(survival, physical errors, and matched VAE-baseline comparisons from the
[reproduction plan §6](beyondmimic_diffusion_reproduction.md)) remain open and
require a controller/evaluation harness plus explicit authorization.

## Implemented seam

Module: `src/mjlab/tasks/tracking/diffusion/handover.py`
(re-exported from `mjlab.tasks.tracking.diffusion`).

- `HandoverHistory` — bounded 8-pair buffer of state snapshots plus the
  latent actually passed to the decoder (`startup.minimum_actual_pairs: 8`).
  Pairing is PRE-ACTION: each row pairs the state `s_t` on which the latent
  `z_t` was decoded and applied, never the next state `s_(t+1)`; the caller
  must retain `s_t` safely until recording after the execution is
  acknowledged.  `history_from_standing_policy: forbidden` and row origin /
  tick-contiguity are caller obligations: the API offers no way to insert a
  standing-policy or reference latent, but generic inputs cannot structurally
  authenticate that pairs are measured, contiguous or standing-free.
- `build_reference_free_plan` — re-hybridizes the 8 history states plus
  the current state with `world_to_hybrid(center_index=8)` on every
  plan, so history is always re-expressed in the current window-center frame;
  normalizes state/latent with the active no-mixture `ProjectionBundle` and
  builds the D0 `clean_mask` (past state+latent and current state fixed at
  clean level 0; current latent and all 32 future steps unknown).
- `ReferenceFreeContinuation.step` — plans from history plus the current
  state, samples the frozen 20-jump DDIM (`eta=0`), denormalizes the
  current-index latent and calls a caller-supplied frozen
  `decode(latent_32, conditioning_99)`; the step signature and the decoder
  contract admit **no reference, encoder, teacher, or hidden-reference
  fallback call path in this module**; the measured 99-D gravity conditioning
  is caller-supplied.  Whether the fed history/conditioning are genuinely
  measured, executed and contiguous is an adapter obligation this module does
  not authenticate.
- Plan-age guard: control ticks must strictly increase; after any reset,
  teleport, estimator discontinuity or policy switch, `reset()` clears history
  and the seam fails closed until a fresh 8-pair VAE bootstrap is recorded.
- Contract guards: legacy 231-channel projected contracts and projection-bundle
  identity mismatches are rejected before any sampling or decoding.

Token/window semantics are unchanged from the active no-mixture contract
(`mjlab-x2-state-latent-d1-no-mixtures-v1`): 41 steps, current index 8, 167
channels (135 normalized physical-state + 32 normalized VAE latent). The frozen
VAE decoder semantics are untouched: the seam only supplies the latent that the
decoder already accepts.

## Tests

`tests/test_tracking_diffusion_handover.py` (12 tests, CPU-only, stub
denoiser/decoder): executed-pair validation, owned snapshots against caller
buffer reuse, 8-pair handover requirement, plan alignment and mask semantics,
history rebasing into the current frame, clean-mask round trip, pre-handover
fail-closed, step mask/decode semantics with 20 denoiser calls and final index
0, plan-age/reset semantics, reference-free signature/decoder-input invariants,
legacy-contract rejection, and projection-identity mismatch rejection.

Pending (explicitly deferred, not completed): a hand-computed geometric
invariant test beyond the world_to_hybrid-as-own-oracle rebasing check, and
additional negative tests for nonintegral/bool tick inputs and decoder
shape/nonfinite rejection.

## Not demonstrated

Closed-loop survival or stability, realized-motion versus frozen-VAE baseline
comparisons, 50 Hz timing, latency, planning/async scheduling, export,
sim2sim and hardware. These remain D3/D5/D6 work requiring their own
authorization and evidence.
