# BeyondMimic diffusion D2 — implementation plan: the DDPM trainer

Status: **plan written 2026-09-30, implementation not started.** D0 is closed as a
specification and D1 is complete (a qualified 248420-row clean+OU dataset exists).
D2 is the supervised **training core**: the frozen schedule, per-token noising, the
Transformer denoiser, the 20-update DDIM sampler, the trainer with EMA/checkpoint/
resume, an offline generation diagnostic, and the CLI that drives them.

This document is a plan and an acceptance checklist, not evidence. Nothing here
authorizes a long training run, a bulk-dataset run, hardware work or a deployment
artifact. The D2 done condition is inherited from the reproduction plan §6:

> Pure CPU tests, forward schedule/DDIM jumps/masks, Transformer, trainer/
> checkpoint/EMA; tiny fixed-batch overfit then bounded single-motion pilot.
> Analytic noising/sampling tests and resume pass; true inference conditioning
> works; normalized-latent inverse and decoder parity pass; fixed-seed held-out
> generation beats trivial copy/zero diagnostics. Hand candidate to D3, not
> directly to hardware.

## 1. Scope and references

Authoritative inputs, in precedence order:

1. The [frozen D0 contract](beyondmimic_diffusion_d0_contract.yaml), SHA256
   `42694cdd3acd72afd2d234a85ca29e9c7a646d64df831021c8dd43e55f3f73e7`. D2 must
   not change it. Every numeric constant below is read from it or derived from its
   frozen construction; where this plan restates a derived number it records the
   exact value so tests can pin it.
2. The [reproduction plan](beyondmimic_diffusion_reproduction.md), especially §4.3
   (normalization/emphasis), §5.1 (DDPM training vs DDIM inference), §5.2 (startup),
   §5.3 (guidance, deferred), §6 (the D2 row) and §7 (placement).
3. The [D1 implementation record](beyondmimic_diffusion_d1_implementation.md) for
   the dataset API and the storage contracts D2 consumes.
4. The independent reference repo `BeyondMimic-Reproduction` at
   `55b37260ce02573e70648111c534b735972a2af4` as **readable reference and source
   of adversarial test cases, not a runtime dependency**. Its
   `stage3/diffusion/*` and `stage3/models/state_latent_transformer.py` informed
   this plan; the corrections the reproduction plan §3 already demanded are
   restated as requirements in §5. Do not import it from mjlab.

Out of D2 scope, deferred by phase: guidance and physical costs (`guidance.py`,
D4), the closed-loop reference-free environment, handover policy and planning
runtime (D3), export/TensorRT/sustained timing (D5), and hardware (D6).

## 2. D0 constants D2 must consume, not redefine

Read them from `DiffusionContract`; do not hard-code a second copy in D2 modules.
The values below are reproduced only so the tests can pin them.

| Item | Value |
| --- | --- |
| control | 50 Hz, 0.02 s period, physics 0.005 s, decimation 4 |
| window | 8 past + current + 32 future = **41** steps, `current_index` **8** |
| state / latent / action | 135 raw, **32** latent, 31 action |
| projected state / token | **199** = 135 + 64 added rows; token **231** = 199 + 32 |
| chain | `K_train=1000`, clean index **0**, **1001** step embeddings |
| schedule | cosine ᾱ → clamp each β to `[1e-5, 0.999]` → cumulative product |
| sampler | DDIM, `eta=0`, 20 updates, endpoints `[1000, 950, …, 50, 0]` |
| conditioning | fix past states+latents and current state with step id 0 |
| unknown entries | current latent and all future entries |
| model | 6 layers, width 512, 8 heads, FFN 2048, GELU, pre-norm, dropout 0, learned positions, no causal mask, separate state/latent step embeddings, x0 prediction |
| training | effective batch 512, LR 1e-4, weight decay 0.001, cosine, 10000-update warmup, EMA power 0.75 / max 0.9999, grad clip 1.0, AdamW, BF16 after FP32 checks |
| v1 precision gates | fp64 DDIM oracle ≤ 1e-12, fp64 projection roundtrip ≤ 1e-10, fp32 same-backend decoder replay atol/rtol 1e-5 |

The training schedule is **independent** of the inference grid. Switching sampler
or step count must not change the training schedule or require retraining for
schedule-arithmetic reasons; changing the training schedule is a new contract.

### 2.1 The frozen schedule, exactly

Recomputed from the contract construction (`corrected_cpu_probe.py` in
`logs/diffusion/d0-validation-20260929T062344Z/` is the D0 oracle):

```text
x           = linspace(0, 1000, 1001)                          # float64
raw         = cos(((x/1000) + 0.008)/1.008 * pi/2)**2
raw         = raw / raw[0]
betas       = clamp(1 - raw[1:]/raw[:-1], 1e-5, 0.999)         # length 1000
alpha_bars  = concat([1.0], cumprod(1 - betas))                # length 1001
grid        = linspace(1000, 0, 21).round().long()             # 21 points
```

Pinned expected values for tests (assert to ≤1e-15 relative in float64):

| Quantity | Value |
| --- | --- |
| `len(alpha_bars)` | 1001 |
| `alpha_bars[0]` | exactly `1.0` |
| `alpha_bars[1]` | 0.999958715775178 |
| `alpha_bars[50]` | 0.9920072786842186 |
| `alpha_bars[500]` | 0.49384359044063775 |
| `alpha_bars[1000]` | 2.4287669070348542e-09 |
| terminal SNR | 2.428766912933763e-09 |
| `betas.min()` / `max()` | 4.128422482196914e-05 / 0.999 |
| source ids | `[1000, 950, 900, …, 100, 50]` (20) |
| destination ids | `[950, 900, …, 50, 0]` (20) |

The table above is a **mathematical** contract, not a byte-level one. It was
recorded from the D0 oracle, which builds the array with `torch.float64`
(`torch.cos`, `torch.cumprod`) while a NumPy `float64` construction uses the C
library's `cos`. Measured 2026-09-30, those two constructions differ at **exactly
one index**: `alpha_bars[732]` differs by 1 ulp (absolute 2.78e-17, relative
1.69e-16). Every pinned value above is nonetheless bit-identical, the fp64 DDIM
oracle error is 2.220e-16 for both, and the resulting sampled trajectory differs
by **exactly 0.0**. Consequences, which are binding:

- The schedule **values** are computed with pure NumPy `float64`: `betas` and
  `alpha_bars` must never be produced by torch arithmetic. Torch is permitted
  only as an explicit, labelled conversion of the already-final array for the
  model and sampler, and `alpha_bars_torch(device, dtype)` is that bridge. Keep
  torch out of module scope in `schedule.py` and import it inside the bridge
  function, so that "torch does not construct the schedule" is structurally
  visible instead of being a claim about a function body. A test must parse the
  module source and assert there is no module-level torch import.
- Compare the pinned values with a relative tolerance of **<= 4e-16**, not exact
  equality, because those printed decimals are themselves libm-dependent.
- `sha256(alpha_bars)` is a **self-consistency identity only**: it hashes the
  implementation's own array, is persisted with the schedule, and is re-checked on
  load to detect corruption, tampering or a silently different construction. It
  must never be compared against a constant recorded from another
  implementation or platform. The two library families necessarily produce
  different bytes (`d4ea818a…` for the D0 torch oracle, `ff6a0e…` for NumPy) and
  neither is more correct.
- A **D0-oracle agreement test** must additionally reimplement the oracle's
  `torch.float64` construction inside the test and assert
  `max|alpha_bars - oracle| <= 1e-12` (four orders of magnitude inside the
  contract's own fp64 DDIM tolerance), recording the observed maximum delta and
  the index where it occurs (expected: 2.78e-17 at index 732).

`alpha_bars[0] == 1.0` exactly is load-bearing: index 0 is the clean level and the
DDIM final destination, so it must not be a clamped near-1 value. `sqrt(1-a)`
appears in the ε̂ denominator and is exactly 0 at index 0 — D2 must never divide
by it, because the denoiser is never called at 0.

### 2.2 The frozen skip-step update

For a jump from source index `k` to destination `j < k`, with `a = alpha_bars`:

```text
eps_hat = (x_k - sqrt(a[k]) * x0_hat) / sqrt(1 - a[k])
x_j     = sqrt(a[j]) * x0_hat + sqrt(1 - a[j]) * eps_hat
```

`eta=0` makes this deterministic conditional on the initial noise. The final
`j=0` destination returns `x0_hat` unchanged (since `a[0]=1`). The sampler must
reapply the fixed-clean values **after every jump**, must pass the original
training timestep ids (not a loop counter), and must support only `eta=0.0` in v1
(any other value raises). `guidance_strength` must be `0.0` in D2; the guided
formula is D4 and a non-zero value must raise rather than be silently ignored.

## 3. D1 interface D2 consumes, and measured facts that constrain it

Dataset directory layout (identical locally and on the NAS):

```text
<dataset-dir>/store/                 append-only shards + manifest.json
<dataset-dir>/offline/assignments.json
<dataset-dir>/offline/dataset.json
<dataset-dir>/offline/projection.npz
<dataset-dir>/offline/stats.json     (bulk only; informational)
```

Public API D2 must use unchanged: `AppendOnlyShardStore`, `SplitAssignments`,
`ProjectionBundle`, `WindowIndex.build`, `WindowDataset`, `WindowSample`,
`TrajectoryRow`. `WindowSample` already provides exactly the D2 input:
`tokens (41,231)`, `projected_state (41,199)`, `normalized_latent (41,32)`,
`clean_actions (41,31)`, `executed_actions (41,31)`, `current_index == 8`.

**Critical integration requirement — open the store with the YAML-derived
contract.** `AppendOnlyShardStore` validates the manifest against
`contract.identity_hash()`. `DEFAULT_CONTRACT = DiffusionContract()` carries
`contract_sha256=None` and therefore hashes to a *different* identity than the
contract the datasets were written with. The datasets were written with
`DiffusionContract.from_yaml(<frozen yaml>)`, whose identity hash is
`fc0bc53d0efb1a58f700f78716e5cf36dca24d527c2a7d9b46bffe49cc9a23de` (versus
`330f7fe0394f1c8b2e32b0383a54d2de8abaa51ed4f1de3f7675f3ba2c263f14` for the
default). D2 must thread the loaded contract through every call and a test must
pin both hashes, otherwise every existing dataset is unreadable by D2.

Measured on the local pilot dataset (2026-09-30, for test design only):

- pilot coverage `train 2640 / validation 0 / test 1020` windows — the pilot has
  **no validation split** (9 groups, integer flooring). Bulk coverage is
  `53087 / 6764 / 7282`. D2 must therefore take the evaluation split as an
  explicit argument and must accept `validation` being empty.
- clean and OU windows are **1:1** in both splits (`train 1320/1320`,
  `test 510/510`), and a pair family never straddles a split.
- token scale is strongly non-uniform: the 64 added projection rows have
  variance ≈ Σ B² ≈ 660 (root emphasis 6), the 135 identity rows ≈ 1, and the
  normalized latent ≈ 1. Observed over 50 pilot windows: state slice std ≈ 12.9,
  latent slice std ≈ 1.019. **Do not re-standardize after projection** (the
  contract forbids it). Consequently a scalar MSE is dominated by the 64
  emphasised rows, and D2 must report per-slice metrics (projected rows vs
  identity rows vs latent, and per noise level) rather than a single number.
- provenance carries `phase ∈ {clean, ou}`, `pair_id`, `initial_state_id`,
  `initial_seed`, `initial_start_frame`, `reference_phase`, `group_key_generated`.
  The balanced sampler reads `phase`. The clean/band mix is a **trainer sampling
  weight**, not a dataset property; with the natural 1:1 layout, upweighting
  `ou` by 2 is exactly the "perturbation-band-only" reading, so both readings are
  reachable without re-collecting.

## 4. Deliverable topology and writer seams

New modules in `src/mjlab/tasks/tracking/diffusion/` (existing D1 files are
`__init__, contract, state, projection, storage, dataset, adapter, policies,
qualification, collector, runtime, runtime_x2`; **do not modify them** except
`__init__.py` exports):

```text
schedule.py        frozen bounded-cosine schedule + the 20-jump DDIM grid
noising.py         per-token independent noising, x0 target, clean mask
model.py           StateLatentTransformer + DenoiserSettings
sampler.py         unguided DDIM reverse loop (eta=0), diagnostics
window_dataset.py  store→WindowDataset loader, provenance, token cache, torch Dataset
training_config.py TrainingConfig schema, YAML load/validate, resolved-config hash
trainer.py         training loop, EMA, LR, accumulation, loss/metrics, resume
checkpoint.py      checkpoint save/load with identity + RNG/optimizer/EMA state
evaluation.py      offline conditioned generation + trivial-baseline diagnostics
```

Plus `configs/diffusion/x2_50hz_train.yaml`, CLI subcommands in
`src/mjlab/scripts/diffusion.py`, tests in `tests/test_tracking_diffusion_*.py`,
and the append-only implementation record at the end of this document.

| Stage | Exclusive ownership | Gate and handoff |
| --- | --- | --- |
| **W1 tensor core** | `schedule.py`, `noising.py`, `model.py`, `sampler.py`, `tests/test_tracking_diffusion_{schedule,noising,model,sampler}.py`, `__init__.py` exports for those symbols | CPU analytic gates (§7.1–7.4); durable API handoff |
| **W2 trainer core** | `window_dataset.py`, `training_config.py`, `trainer.py`, `checkpoint.py`, `tests/test_tracking_diffusion_{window_dataset,trainer,checkpoint}.py`, `__init__.py` exports | CPU gates (§7.5–7.7); resume/accumulation/EMA/cache evidence |
| **W3 integration** | `evaluation.py`, `configs/diffusion/x2_50hz_train.yaml`, `src/mjlab/scripts/diffusion.py` (`train`, `evaluate-offline`, `audit`), `tests/test_tracking_diffusion_eval.py`, `tests/test_tracking_diffusion_train_cli.py`, the records appended to this document | end-to-end CLI on the pilot dataset; bounded overfit; §7.8 baselines |
| **R1 review** | read-only review of every D2 path above | P0/P1/P2 findings with source/test proof; verdict and missing evidence |
| **W4 fixes** | only files named by R1 findings | fixes plus a regression test each |
| **R2 re-review** | read-only re-review of W4 | verdict and residual risk |

Coordination rules, inherited from D1 because the checkout is shared:

- Sequential exclusive writers in `/home/agiuser/projects/mjlab`. Never two
  writers at once. Do not use git worktrees: the checkout is dirty with another
  workstream's edits.
- Children must not run any state-changing git command (no add/commit/stash/
  checkout/reset/push) and must not touch the 7 files already modified by another
  workstream — in particular **do not edit `docs/source/changelog.rst`**.
- `uv run --no-sync` for everything; `CUDA_VISIBLE_DEVICES=''` for CPU tests.
- W2/W3 may read W1/W2 APIs but must not redefine them; a needed shared change
  goes back to the parent.
- No scratch files at the repository root. Evidence under
  `logs/diffusion/d2-<stage>-<date>/` (gitignored).

## 5. Module specification

### 5.1 `schedule.py`

```python
@dataclass(frozen=True, slots=True)
class DiffusionSchedule:
  training_k: int            # 1000
  cosine_offset: float       # 0.008
  beta_min: float            # 1e-5
  beta_max: float            # 0.999
  betas: np.ndarray          # float64 (K,), clamped
  alpha_bars: np.ndarray     # float64 (K+1,), [0] == 1.0
  # .embedding_count -> K+1 ; .clean_index -> 0 ; .terminal_snr -> float
  # .alpha_bar_at(k) -> float ; .alpha_bars_torch(device=, dtype=) -> Tensor
  # .as_dict() -> includes sha256 over the float64 bytes and terminal_snr
  # .save(path) / .load(path) with hash verification
  @classmethod build(cls, *, training_k=1000, cosine_offset=0.008,
                     beta_min=1e-5, beta_max=0.999) -> DiffusionSchedule
  @classmethod from_contract(cls, contract) -> DiffusionSchedule

@dataclass(frozen=True, slots=True)
class InferenceGrid:
  source_ids: tuple[int, ...]       # 20, descending, excludes 0
  destination_ids: tuple[int, ...]  # 20, includes 0
  # validates: 20 pairs, strictly decreasing, dest[-1] == 0, src[0] == training_k,
  #            src[i] > dest[i], 0 <= ids <= training_k, no repeats
  def as_dict(self) -> dict[str, object]
  @classmethod uniform(cls, schedule, *, updates=20) -> InferenceGrid

def build_schedule(**kwargs) -> DiffusionSchedule
def build_inference_grid(schedule, *, updates=20, contract=None) -> InferenceGrid
```

`build_inference_grid` with a contract must additionally assert that the
produced tuple equals `contract`'s frozen endpoints, so the frozen grid cannot
drift from the contract.

Compute the values with **NumPy `float64`** and `math.pi`. This module is
importable without torch: construction and validation use only NumPy, and the
only torch usage is a function-local import inside the `alpha_bars_torch` bridge
that converts the finished array into a tensor (see §2.1). The reference repo's
float32 torch construction is not the frozen one, and a torch-based float64
construction is not more faithful than NumPy's.
`as_dict()` carries the self-consistency `sha256` of the array this
implementation computed plus `terminal_snr`; `load` recomputes and compares that
same-implementation hash.

### 5.2 `noising.py`

```python
def sample_levels(shape, *, training_k, generator) -> torch.Tensor    # long, 0..K inclusive
def add_independent_noise(state, latent, k_state, k_latent, alpha_bars, *, generator)
    -> NoisedPair(state, latent, noise_state, noise_latent, tokens)
def x0_target(clean_tokens) -> torch.Tensor
def clean_mask(*, current_index, state_dimension, token_dimension, device) -> Tensor  # bool
def apply_clean_mask(sample, clean, mask) -> torch.Tensor
```

Requirements:

- Levels are sampled as **independent uniform integers in `[0, K]` inclusive**
  for the state and latent slices of every token (`randint(0, K+1)`), matching
  `embedding_count` 1001. This includes 0, the clean level.
- Noise is drawn as **two separate independent tensors** for the state and latent
  slices; never reuse one draw. A test asserts different storage and that
  perturbing one slice's seed changes only that slice.
- `k=0` must reproduce the clean value exactly (assert equality, not tolerance):
  `sqrt(1)*x + sqrt(0)*noise == x`. Because `1-a[0]` is exactly 0, this is exact
  in float64 and must not be softened by an epsilon clamp.
- The mask must be exactly: tokens `0..current_index-1` in full (state+latent),
  and token `current_index` restricted to `[0, state_dimension)`. Everything
  else is unknown. This is the D0 `conditioning`/`unknown_entries` pair.
- An analysed marginal check: for a fixed clean value and large sample, the noisy
  mean/std must equal `sqrt(a[k])*x` and `sqrt(1-a[k])` within a stated
  statistical tolerance.

### 5.3 `model.py`

```python
@dataclass(frozen=True, slots=True)
class DenoiserSettings:
  token_dimension: int = 231
  sequence_length: int = 41
  embedding_count: int = 1001
  width: int = 512
  layers: int = 6
  attention_heads: int = 8
  ffn_width: int = 2048
  dropout: float = 0.0
  activation: str = "gelu"
  norm_first: bool = True
  @classmethod from_contract(cls, contract, *, training_k=None) -> DenoiserSettings

class StateLatentTransformer(nn.Module):
  def __init__(self, settings: DenoiserSettings) -> None
  def forward(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor
  def parameter_count(self) -> int
```

- Input `noisy_tokens [B,41,231]`, `step_ids [B,41,2]` long, `[...,0]` for the
  state slice and `[...,1]` for the latent slice.
- **Reject invalid step ids** (`< 0` or `>= embedding_count`) with a clear error.
  The reference silently clamps; clamping would hide an inference-grid bug, so it
  is forbidden here.
- One combined token per timestep: project the full 231-D token once, add the
  learned temporal position, then add the two independent step embeddings.
  Output projects back to 231.
- No causal mask. A test must show token 40 influences token 0 (bidirectionality).
- Parameter count for the frozen settings must equal **20,197,607** (the D0 GPU
  probe value). If W2's implementation produces a different count the writer must
  **stop and report** rather than adjust the number; a genuine architectural
  difference needs parent adjudication before it becomes the trained artifact.

### 5.4 `sampler.py`

```python
@dataclass(frozen=True, slots=True)
class SamplerConfig:
  grid: InferenceGrid
  eta: float = 0.0                 # only 0.0 accepted in v1
  guidance_strength: float = 0.0   # only 0.0 accepted in D2
  initial_noise_scale: float = 1.0

@dataclass(frozen=True, slots=True)
class SamplerDiagnostics:
  denoiser_calls: int
  jumps: int
  source_ids_returned: tuple[int, ...]
  step_ids_seen: tuple[int, ...]          # what the denoiser was actually given
  final_index: int
  initial_noise_norm: dict[str, float]

def sample_trajectory(
  denoiser, *, schedule, conditions, config, generator,
) -> tuple[Tensor, SamplerDiagnostics]
```

- `conditions` carries `clean_tokens` and the `clean_mask`; unknown entries start
  from `initial_noise_scale * randn` in token space, drawn from `generator` so
  same-seed runs are identical and different seeds differ.
- Per jump the denoiser receives step id `k` on unknown entries and **id 0 on
  fixed entries**, then the clean mask is reapplied to the result of every jump.
- Exactly 20 denoiser calls for the frozen grid, never a call with a step id of 0
  on unknown entries (the final `j=0` update reuses the `x0_hat` from `k=50`).
- Return the token tensor (all 41 steps; the caller takes `current_index`), plus
  diagnostics that a test can assert: call count, ids seen, and that the fixed
  entries are bit-identical to the conditioned values at the end.

### 5.5 `window_dataset.py`

```python
@dataclass(frozen=True, slots=True)
class DatasetSource:
  directory: Path
  contract_path: Path
  @classmethod resolve(cls, directory, *, contract_path) -> DatasetSource
  def load(self) -> LoadedDataset
  def token_cache_path(self, split: str) -> Path

@dataclass(frozen=True, slots=True)
class LoadedDataset:
  contract, projection, assignments, store, index
  def dataset(self, split: str) -> WindowDataset
  def refs(self, split: str) -> tuple[WindowRef, ...]

@dataclass(frozen=True, slots=True)
class WindowRecord:
  tokens: np.ndarray        # (41,231) float32
  split: str
  group_key: str
  motion_id: str
  phase: str                # "clean" | "ou" | ""
  ou_noise_norm: float
  pair_id: str
  start_tick: int

def load_window_records(source, split, *, motion_ids=None, limit=None) -> list[WindowRecord]
def build_token_cache(source, split, path, *, motion_ids=None, dtype=np.float32) -> tuple[Path, int]
class TokenWindowDataset(torch.utils.data.Dataset)   # reads the cache, returns (tokens, weight, record_index)
def sampling_weights(records, *, clean_weight, perturbed_weight) -> np.ndarray
```

Requirements:

- Load the contract with `DiffusionContract.from_yaml` and keep it for the store,
  the projection and the index. Assert the store's identity hash matches.
- `motion_ids` filtering is applied to the *records* and must never change which
  rows are eligible, only which windows are selected, so a filtered run is
  exactly a subset of the unfiltered run.
- The token cache is a `float32` `.npy` memmap of `(N,41,231)` plus a sibling
  `.json` recording the source dataset hash, split, contract hash, projection
  hash, window count, dtype and creation time. A cache whose metadata does not
  match the request must be rejected, not silently reused.
- **Memory rule:** the lazy `WindowIndex` caches every store row in RAM
  (`_segment_cache`). A DataLoader with `num_workers>0` therefore multiplies that
  cache per worker. Training must read from the token cache with
  `num_workers = 0` (or 2 at most), never from the store. Report the projected
  cache size before writing it and refuse above a `max_cache_gib` bound
  (default 8). Pilot: train 2640 × 41 × 231 × 4 B ≈ 100 MiB. Bulk: train 53087 ≈
  2.0 GiB, validation ≈ 256 MiB, test ≈ 276 MiB.
- `float32` tokens are a training convenience only. The D0 fp64 gates apply to
  the schedule/oracle/projection, not to cached tokens.

### 5.6 `training_config.py`

```python
@dataclass(frozen=True, slots=True)
class TrainingConfig:
  epochs: int = 1000                 # cap, never automatic
  max_updates: int | None = None
  effective_batch_size: int = 512
  microbatch_size: int = 128
  gradient_accumulation_steps: int = 4
  learning_rate: float = 1.0e-4
  weight_decay: float = 0.001
  scheduler: str = "cosine"
  warmup_updates: int = 10000
  max_grad_norm: float = 1.0
  mixed_precision: str = "bf16"      # after fp32 checks
  loss_reduction: str = "token_mean"
  seed: int = 0
  epoch_sample_budget: int | None = None
  clean_weight: float = 1.0
  perturbed_weight: float = 1.0
  eval_split: str = "test"
  eval_seed: int = 0
  # .validate() enforces: microbatch * accum == effective batch,
  #   max_updates is not None or epochs is set for this invocation,
  #   mixed_precision in {fp32, bf16}, non-negative weights with at least one > 0
  # .as_dict() / .sha256() ; from_mapping / from_yaml
```

`max_updates` and `epochs` are both budgets; when both are present the smaller
wins and the resolved value is recorded. The CLI must refuse to start without an
explicit budget, which is how "do not run 1000 epochs automatically" is enforced,
and must refuse an unbounded large budget (> 200000 updates) without an explicit
`--allow-long-run` flag.

### 5.7 `trainer.py` and `checkpoint.py`

```python
@dataclass(frozen=True, slots=True)
class TrainResult:
  global_step: int
  epochs_completed: int
  best_validation_loss: float | None
  final_train_loss: float
  metrics_path: Path
  checkpoint_paths: dict[str, Path]

class DiffusionTrainer:
  def __init__(self, *, config, contract, schedule, model, train_dataset,
               eval_datasets: Mapping[str, Dataset], output_dir: Path,
               device, resume: Path | None = None) -> None
  def train(self) -> TrainResult
  def evaluate(self, split: str, *, parameters="model") -> dict[str, float]

def save_checkpoint(path, *, model, ema, optimizer, scaler, config, contract,
                    schedule, dataset_identity, global_step, epoch,
                    best_validation_loss, rng_state) -> None
def load_checkpoint(path, *, model, ema=None, optimizer=None, scaler=None,
                    map_location="cpu") -> CheckpointState
```

Training requirements:

- Objective: mean squared error of `x0_hat` against the clean tokens, reduction
  `token_mean` (equal weight per token dimension), retaining the paper's x0
  objective. Do not switch to an unweighted epsilon regression.
- Per-update: sample state and latent levels independently per token, draw
  separate independent noise, build noisy tokens, predict, backprop the scaled
  loss, clip gradients to 1.0, step optimiser **only on the final micro-batch of
  each accumulation group** (a partial final group in an epoch must be handled by
  dividing by the *actual* accumulated micro-batches, not the nominal count), and
  update EMA once per optimizer update.
- AdamW; LR follows warmup then cosine to zero over the resolved total updates;
  the LR actually used is recorded per update.
- BF16 autocast is allowed only after an FP32 finite-output parity check on the
  first micro-batch. NaN/Inf loss must raise, and the run must stop rather than
  write a checkpoint from a diverged state.
- Metrics, per update and per epoch: total/state/latent MSE; MSE split into the
  64 emphasised projected rows, the 135 identity rows and the latent slice; MSE
  on **unknown entries only** (the D0 mask) as well as on all entries; MSE bucketed
  by noise-level decade. All written as JSONL plus a `train-report.json` summary.
- Evaluation uses EMA weights by default (report both raw-model and EMA numbers).
- Determinism: a fixed seed sets torch/numpy/python RNG and `torch.use_deterministic_algorithms(True)` where the backend allows it. Resume must restore the
  RNG state, optimizer, scaler, EMA, LR position, step and epoch, and a resume
  test must show continuation equals an uninterrupted run at the tolerance the
  backend actually supports (**exact** equality when deterministic algorithms are
  available; otherwise the measured residual must be recorded, not assumed).
- Checkpoints carry: model, EMA, optimizer, scaler, RNG, step, epoch, best
  validation, the resolved config and its hash, contract identity hash, schedule
  identity hash, projection matrix/statistics hashes, dataset directory +
  assignments hash + split coverage, and the torch/CUDA version. A checkpoint
  whose identities do not match the loaded dataset or contract must be rejected
  on resume.
- Writes are atomic (`tmp` + `os.replace`), and the run directory carries a
  `README.md` and the resolved config.

### 5.8 `evaluation.py`

```python
@dataclass(frozen=True, slots=True)
class GenerationReport:
  split: str
  windows: int
  seed: int
  metrics: dict[str, float]        # model vs baselines, per slice and per horizon bucket
  baselines: dict[str, float]
  beats_trivial_copy: bool
  beats_zero_latent: bool
  diagnostics_path: Path

def evaluate_generation(model, *, schedule, grid, records, projection, contract,
                        seed, device, max_windows=None) -> GenerationReport
```

Offline conditioned generation: take each held-out window, condition on the D0
fixed entries, sample with the frozen 20-update grid, and score the **unknown
entries** (current latent and all future) in three spaces:

1. normalized token space (model-native),
2. recovered physical space via `ProjectionBundle.inverse_state` + `denormalize`
   for the state slice, and `denormalize_latent` for the latent slice,
3. by horizon (future step buckets) and by phase (clean vs OU window), because
   the two have different reference dynamics.

Trivial baselines, all computed on the same windows and seeds:

- `copy_current_state` — repeat the current state for all future steps.
- `copy_current_latent` — repeat the current latent.
- `zero_latent` — the normalized latent mean (0) for the unknown latent entries.
- `prior_only` — score the initial Gaussian sample without any denoising, the
  floor that any trained model must beat.

The gate is `model < copy_current_latent` and `model < zero_latent` on the unknown
latent entries, and `model < copy_current_state` on the unknown state entries,
with the achieved margin recorded. Failing the gate is a reported D2 outcome, not
a number to adjust.

### 5.9 CLI

Extend `src/mjlab/scripts/diffusion.py` (keep the existing
`preflight/inspect/replay/build/stats/collect` verbs unchanged):

```text
train            --dataset-dir DIR --config PATH --out DIR [--device cpu|cuda:0]
                 [--max-updates N | --epochs N] [--motion-id ID]... [--resume PATH]
                 [--cache-tokens/--no-cache-tokens] [--allow-long-run] [--json OUT]
evaluate-offline --run DIR --split test [--max-windows N] [--json OUT]
audit            --dataset-dir DIR [--json OUT]
```

- All three must work without constructing an mjlab simulator environment.
- `train` prints the resolved config hash, dataset identity, cache size, planned
  optimizer updates and effective batch before starting, and refuses to start when
  any identity check fails.
- `audit` reports split coverage per motion and phase, cache sizes, token
  statistics per slice and noise-level sanity, and exits non-zero on an identity
  mismatch.
- `evaluate-offline` reads a run directory, loads `checkpoint-best.pt` (or
  `--checkpoint`), and writes the generation report.

### 5.10 Config file

`configs/diffusion/x2_50hz_train.yaml` — a non-authorizing input record whose
values match the contract and Table S7; the trainer must reject a config whose
model/noise fields disagree with the contract:

```yaml
schema_version: mjlab-x2-state-latent-train-v1
contract: docs/plans/beyondmimic_diffusion_d0_contract.yaml
model: {layers: 6, width: 512, attention_heads: 8, ffn_width: 2048, dropout: 0.0}
noise: {training_k: 1000, cosine_offset: 0.008, beta_min: 1.0e-5, beta_max: 0.999, updates: 20}
training:
  epochs: 1000            # cap; an explicit budget is required to start
  effective_batch_size: 512
  microbatch_size: 128
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  weight_decay: 0.001
  scheduler: cosine
  warmup_updates: 10000
  max_grad_norm: 1.0
  mixed_precision: bf16
  loss_reduction: token_mean
  seed: 0
ema: {power: 0.75, max_decay: 0.9999}
sampling: {clean_weight: 1.0, perturbed_weight: 1.0}
eval: {split: test, seed: 0}
```

`sampling` is the D2 lever for the clean/band question; the default is the natural
1:1 layout and any non-default value must be recorded in the resolved config and
in the run's report.

## 6. Acceptance checklist (implementation, not model acceptance)

- Frozen schedule, grid, conditioning mask and update equation implemented exactly
  as pinned, with the contract as the single source of truth.
- Independent per-token state/latent levels and independent noise draws; exact
  clean values at index 0; no clamp that could hide an invalid step id.
- The denoiser uses the frozen Transformer layout, rejects invalid ids, is
  bidirectional, and has the expected parameter count.
- The sampler performs exactly 20 jumps with 20 denoiser calls, never calls the
  denoiser at clean index 0, passes original timestep ids, reapplies clean
  conditioning after every jump, and rejects `eta != 0` and guidance in v1.
- The trainer honours accumulation including a partial final group, the warmup+
  cosine LR, EMA power/max, gradient clipping, mixed precision after an FP32 check,
  NaN/Inf abort, and the epoch sample budget under balanced sampling.
- Checkpoints are atomic and carry full identity plus RNG/optimizer/scaler/EMA
  state; resume reproduces continuation at the documented tolerance and rejects a
  mismatched dataset, contract, schedule or projection.
- The dataset loader opens the store with the YAML-derived contract, must accept
  an empty `validation` split, and must not silently reuse a stale token cache.
- Offline generation conditions the true unknown entries, reports per-slice and
  per-horizon metrics, and compares against copy/zero/prior-only baselines with
  the margin recorded.
- Bounded local pilot: tiny fixed-batch overfit plus a bounded run on the pilot
  dataset, both with fixed seeds and recorded budgets.
- Normalized-latent inverse roundtrip and frozen-decoder parity pass at the D0
  tolerances.
- Scoped `pytest -q -k diffusion`, `ruff format --check`, `ruff check`, and
  scoped `ty check` are clean for the new files, using `uv run --no-sync`.
- Nothing is launched beyond the bounded local gate; the bulk dataset, long runs
  and hardware remain separate authorizations.

## 7. Numerical gates

### 7.1 Schedule gates

`len(alpha_bars) == 1001`; `alpha_bars[0] == 1.0` exactly; `alpha_bars` strictly
decreasing; every beta within `[1e-5, 0.999]`; terminal SNR matches
2.428766912933763e-09 to ≤4e-16 relative; the pinned values in §2.1 match to
≤4e-16 relative; `betas`/`alpha_bars` are constructed by NumPy arithmetic with no
module-level torch import in `schedule.py` (the `alpha_bars_torch` bridge uses a
function-local import, and an AST test asserts the module scope stays torch-free);
the array agrees with an
in-test reimplementation of the D0 `torch.float64` oracle to ≤1e-12 max abs
(observed 2.78e-17 at index 732); the self-consistency `sha256` round-trips
through `save`/`load` and a tampered file is rejected. **Do not** assert equality
against a fixed cross-implementation byte hash: the same schedule legitimately
hashes to `d4ea818a…` under torch and `ff6a0e…` under NumPy.

### 7.2 Noising gates

Levels within `[0, 1000]` and, over a large sample, approximately uniform;
`k=0` reproduces the clean value exactly for both slices; the two noise draws are
independent storage; the combined noisy marginal matches
`sqrt(a)*x ± sqrt(1-a)` statistically; the clean mask is exactly the D0 mask.

### 7.3 Sampler gates

A closed-form fp64 oracle (the D0 probe's construction, reimplemented in the test)
reproduces every skipped-step destination to **≤1e-12 max abs** when `x0_hat` is
exact; exactly 20 denoiser calls; the step ids given to the denoiser are
`[1000,950,…,50]` on unknown entries and `0` on fixed entries; the final output
equals the `x0_hat` from the `k=50` call with the clean mask reapplied (bit
equality); same seed → identical output; different seeds → different output;
`eta=0.1` and `guidance_strength>0` raise.

### 7.4 Model gates

Forward shape `[B,41,231]→[B,41,231]`; invalid ids (`-1`, `1001`) raise; dropout 0
means two calls in `train()` mode are identical; perturbing token 40 changes
token 0's output (no causal mask); gradients reach every parameter; parameter
count equals 20,197,607 for the frozen settings; a 41-step forward in fp32 is
finite for a random batch.

### 7.5 Dataset/trainer-data gates

Both pinned contract identity hashes (§3) as a regression guard; the pilot store
reopens on the laptop; `train/validation/test` coverage matches `dataset.json`;
an empty `validation` split loads and reports zero rather than raising; the token
cache round-trips and a metadata mismatch is rejected; tokens are finite; per-slice
token statistics match the measured pilot ranges; `motion_ids` filtering produces
a strict subset.

### 7.6 Trainer gates

A synthetic well-formed dataset (fixed seed) drives: accumulation with a partial
final group; LR equals the analytic warmup/cosine value at chosen steps; EMA decay
equals `min(0.9999, 1-(1+n)^-0.75)`; the loss falls on a tiny fixed batch overfit
(target < 1e-3 on the fixed draw within a bounded update count); NaN injection
raises; the checkpoint round-trips and resume rejects a mismatched identity.

### 7.7 Parity gates

`ProjectionBundle` normalized-latent inverse roundtrip ≤ 1e-10 fp64; the projected
state roundtrip ≤ 1e-10 fp64; decoding a predicted latent through the frozen VAE
inference path reproduces the same-backend fp32 result at atol/rtol 1e-5. The
decoder parity must use the existing `load_cohort_member_inference` /
`InferenceModel` seam, not a new loader.

### 7.8 Generation gates

On held-out pilot windows: the model beats `copy_current_latent`, `zero_latent`
(on unknown latent entries) and `copy_current_state` (on unknown state entries),
and beats `prior_only`; same-seed repeatability; the report records every margin.

## 8. Bounded local pilot gate

Runs on the laptop's RTX 5080 using the **local** pilot dataset, which needs no
NAS access:

```text
dataset: logs/diffusion/d1-pilot-20260930b   (store 13500 rows/54 shards; train 2640, test 1020)
```

Budget, all three enforced by the invocation and recorded:

1. **Tiny fixed-batch overfit** — 8 windows, <= 2000 updates, target train loss
   < 1e-3 on a fixed noise draw, CPU or `cuda:0`, <= 5 min.
2. **Bounded pilot training** — `--max-updates 300`, effective batch 512
   (microbatch 128 × 4), seed 0, `--motion-id tennis_000` for the strict
   single-motion run plus one 3-motion run; <= 15 min each; `checkpoint-best.pt`
   selected on the pilot test split only because `validation` is empty (this must
   be labeled, not presented as a clean validation).
3. **Offline generation diagnostic** — on the 1020 held-out test windows and on
   the non-training windows of the chosen motion, reporting the §7.8 margins.

These are bounded engineering gates, not model-quality claims. No bulk-dataset
run, no 1000-epoch run, no export and no hardware are part of D2. A bulk-dataset
run is D2b and needs explicit authorization and a monitored managed-container job.

## 9. Non-goals in D2

- `guidance.py`, physical costs, keyframe inpainting and any non-zero guidance
  strength (D4).
- The reference-free closed-loop environment, handover, history warming and plan
  scheduling (D3).
- Export, ONNX/TensorRT, async planner, sustained timing (D5).
- Any change to the D0 contract, the D1 modules, the D1 datasets or the frozen VAE.
- Any claim about model quality, real-time performance, or hardware behaviour.
- Editing `docs/source/changelog.rst` (owned by an in-flight workstream).

## 10. Verification commands

```bash
cd /home/agiuser/projects/mjlab
uv run --no-sync ruff format --check <new paths>
uv run --no-sync ruff check <new paths>
uv run --no-sync ty check <new paths>
CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q -o faulthandler_timeout=60 \
  tests/ -k diffusion
uv run --no-sync python -m mjlab.scripts.diffusion audit \
  --dataset-dir logs/diffusion/d1-pilot-20260930b --json /tmp/d2-audit.json
```

Focused suites get <= 120 s; an unexpected timeout is diagnosed, not repeated.
GPU gates use `cuda:0` explicitly and must print the device name.

## 11. Handoff to D3

D2 hands over: the frozen schedule/grid/module APIs, a trained (pilot-scale)
checkpoint with EMA and full identity, the offline generation report with its
margins and residual risks, and the explicit list of what was not demonstrated.
D3 consumes the sampler and model to build reference-free closed-loop
continuation with the VAE-history handover; it must not re-derive the schedule or
the conditioning mask, and it must not import the reference repo.

## 12. Implementation and validation record

### 12.1 W3 integration record — 2026-09-30

Implemented the W3 slice of D2 without changing the D0 contract, D1 modules, or
W1/W2 tensor/trainer modules.  The integration adds offline conditioned
20-jump DDIM generation diagnostics and trivial/prior baselines, an identity and
provenance audit, the non-authorizing 50 Hz training YAML, and bounded `train`,
`evaluate-offline`, and `audit` CLI verbs.  The generation report scores the D0
unknown mask in normalized token space and recovered state/latent physical space,
with phase and future-horizon metrics; it records margins rather than changing a
failed gate.

Changed W3-owned files:

- `src/mjlab/tasks/tracking/diffusion/evaluation.py`
- `configs/diffusion/x2_50hz_train.yaml`
- `src/mjlab/scripts/diffusion.py`
- `tests/test_tracking_diffusion_eval.py`
- `tests/test_tracking_diffusion_train_cli.py`

The following evidence is bounded to the local pilot and is retained under
`logs/diffusion/d2-implementation-20260930/`:

1. **Pilot audit (measured, no training):**
   `env CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m
   mjlab.scripts.diffusion audit --dataset-dir logs/diffusion/d1-pilot-20260930b
   --json logs/diffusion/d2-implementation-20260930/audit.json` returned exit 0.
   The raw JSON/output is in `audit.json` and `audit.stdout.txt`; timing is in
   `audit.time.txt`.  It measured 13,500 rows / 54 shards, window coverage
   `train=2640, validation=0, test=1020`, and matching YAML-derived store and
   projection identity `fc0bc53d0efb1a58f700f78716e5cf36dca24d527c2a7d9b46bffe49cc9a23de`.
   The audit ran on CPU in 7.85 seconds wall time.

2. **Tiny fixed-draw overfit (measured engineering gate):** an inline
   `uv run --no-sync` CPU harness used exactly 8 fixed windows, seed 123, one
   fixed state/latent Gaussian draw (draw seed 456, both levels 500), and a
   2,000-update ceiling.  It reached train loss
   `2.4948931809376518e-12 < 1e-3` in 24.91 seconds inside the process (29.75
   seconds including command startup), exit 0.  Raw JSON is in
   `overfit.stdout.txt`, timing in `overfit.time.txt`, and the checkpoint is
   `overfit-run/checkpoint-last.pt`.  This is an objective/optimization gate,
   not evidence of diffusion model quality.

3. **Bounded end-to-end CLI smoke (measured plumbing gate):**
   `CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m
   mjlab.scripts.diffusion train --dataset-dir logs/diffusion/d1-pilot-20260930b
   --config configs/diffusion/x2_50hz_train.yaml
   --out logs/diffusion/d2-implementation-20260930/cli-smoke-all-run
   --device cuda:0 --max-updates 5
   --json logs/diffusion/d2-implementation-20260930/cli-smoke-all.json` returned
   exit 0 in 15.39 seconds.  It printed the device
   `NVIDIA GeForce RTX 5080 Laptop GPU`, effective batch 512, five planned and
   completed optimizer updates, and wrote `checkpoint-last.pt` plus the run
   metadata and token caches.  Raw output/JSON/timing are in
   `cli-smoke-all.stdout.txt`, `cli-smoke-all.json`, and
   `cli-smoke-all.time.txt`.  This was not the 300-update pilot and carries no
   quality claim.  An initial invocation without `CUBLAS_WORKSPACE_CONFIG`
   failed closed before the first update because PyTorch's deterministic CUDA
   cuBLAS requirement was not set; that raw failed attempt is retained as
   `cli-smoke-first-failed.*`, and the successful bounded command set the
   required environment variable explicitly.

Verified by the diffusion suite and static checks:

- `CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q -o faulthandler_timeout=60
  tests/ -k diffusion` — **111 passed, 2032 deselected**, 26.28 seconds; raw
  output/timing are in `diffusion-suite-final.stdout.txt` and
  `diffusion-suite-final.time.txt`.
- `uv run --no-sync ruff format --check` on the four W3 Python paths — passed.
- `uv run --no-sync ruff check` on the four W3 Python paths — passed.
- `uv run --no-sync ty check` on the four W3 Python paths — passed.
  Combined raw output is in `static-checks.txt`.
- Focused W3 tests: **7 passed** in 1.57 seconds.

What is verified by tests is the report schema, unknown-mask scoring, physical
projection seam, phase/horizon metric production, same-seed repeatability, CLI
budget refusal/parser behavior, frozen YAML loading, plus the whole inherited
D2 diffusion suite.  What is measured above is dataset identity/audit, a fixed
objective overfit, and checkpoint-writing CLI plumbing.  None of those
measurements demonstrates model quality.

Not demonstrated by this record: the 300-update pilot, held-out generation
quality or the §7.8 baseline-beating outcome, a decoder/VAE parity run, bulk/NAS
training, sustained 50 Hz timing, closed-loop behavior, export, or hardware.
No model-quality or deployment claim is made; the parent must complete any
remaining authorized generation-quality and handoff gates separately.

### 12.2 W4 review-fix record — 2026-09-30

W4 fixed every P0/P1 finding in `r1-review.md` and both cheap, in-scope P2
findings.  No finding was rejected or deferred, and no D0/W1 schedule, noising,
model, or sampler semantics were changed; the sampler only now rejects a grid
that is not the already-frozen D0 grid.

Finding disposition and regression coverage:

- **P0 physical-space generation mask — fixed.**  The projected-state mask is
  reduced to a per-timestep raw-state mask before indexing the 135-D inverse
  projection.  `test_generation_uses_raw_state_mask_for_real_projection_shape`
  exercises a real `ProjectionBundle`.
- **P1 non-frozen sampler grid — fixed.**  `sample_trajectory` requires the
  frozen 20-jump endpoints.  `test_non_frozen_grid_is_rejected_before_sampling`
  covers a structurally valid but wrong grid.
- **P1 checkpoint optimizer/EMA omission — fixed.**  Resume loading now
  rejects missing or null optimizer/EMA state when those components are supplied;
  `test_resume_rejects_missing_optimizer_or_ema_state` covers both fields.
- **P1 fresh-run seeding — fixed.**  CLI model construction goes through the
  seeded factory before weight initialization; `test_seeded_model_factory_repeats_initial_weights`
  verifies repeatability.
- **P1 BF16 FP32-check resume state — fixed.**  Checkpoints persist and restore
  `fp32_checked`; `test_resume_restores_fp32_check_state` and the checkpoint
  round-trip assertion cover it.
- **P1 motion-filter dataset identity — fixed.**  Normalized motion IDs and
  limits are part of cached and no-cache identities, and offline evaluation
  reapplies the checkpoint filter.  The window-dataset identity assertion and
  `test_checkpoint_motion_filter_is_normalized` cover the filter contract.
- **P1 no-cache checkpoint identity — fixed.**  The no-cache TensorDataset gets
  the complete source/contract/projection/split/filter identity before training;
  `test_no_cache_identity_keeps_source_and_motion_filter` covers the attachment.
- **P1 empty/mislabeled evaluation selection — fixed.**  Empty splits cannot
  produce a best checkpoint, while reports explicitly record the selected split
  and best-evaluation aliases.  `test_empty_evaluation_split_is_not_a_best_checkpoint`
  covers the false-zero case.
- **P1 ignored EMA YAML — fixed.**  Frozen EMA values are parsed, validated,
  included in the resolved config hash, and passed to the EMA; config tests cover
  both accepted values and drift rejection.
- **P1 resolved epoch long-run guard — fixed.**  The CLI checks the resolved
  epoch/update budget after dataset sizing; `validate_resolved_updates` tests the
  refusal and explicit opt-in.
- **P2 audit cache identity — fixed.**  Audit validates discovered cache arrays
  and metadata against the loaded source and makes failures set `ok=false`;
  `test_audit_rejects_stale_cache_metadata` covers a tampered cache.
- **P2 target-leaking latent baseline — fixed.**  `copy_current_latent` repeats
  the last known historical latent (token 7), not the unknown held-out current
  latent; `test_copy_current_latent_uses_last_known_history_not_target` covers it.

W4 validation after the fixes:

- `CUDA_VISIBLE_DEVICES='' uv run --no-sync pytest -q -o
  faulthandler_timeout=60 tests/ -k diffusion` — **121 passed, 2032 deselected**.
- Scoped `uv run --no-sync ruff format --check`, `ruff check`, and `ty check` on
  the changed D2 modules, CLI, and regression tests — all passed.

The required generation-quality, decoder-parity, sustained-timing, bulk-data,
hardware, and uninterrupted-versus-resumed numerical gates remain unperformed;
these are residual D2 evidence gaps, not silently claimed fixes.

## 13. Post-review fix round (independent review, 2026-09-30)

An independent reviewer (fresh context, read-only, running on the current tree)
returned **not ready for acceptance** with five findings. The parent checked
all five against source: **every one is valid**. Two corrections were made to
the report itself - the recorded JSON key is `evaluation_split`, not
`best_evaluation_split`; and F1 is broader than reported because the test split
is baked into both `TrainingConfig.eval_split` and the shipped YAML, so a plain
`evaluate-offline --run <dir>` silently evaluates a test-selected
`checkpoint-best.pt`. The reviewer's own counts (**122 passed, 1 skipped**) were
reproduced exactly, confirming the review ran against the current tree.

One invariant governs this round: **no artifact may be trusted beyond what is
independently re-derived.** Model selection may not read the test split; cached
tokens must be provable from a digest; audit counts must come from D1; the
bounded path must actually be bounded; and an empty result must not report
success.

Scope is the five findings and the regressions that pin them. The failed
generation-quality gate (§7.8) is **not** in scope: it is diagnosed separately
and must not be papered over here.

| id | severity | finding |
|----|----------|---------|
| F1 | P1 | the test split drives best-checkpoint selection, and the offline CLI defaults to that checkpoint for "held-out" evaluation |
| F2 | P1 | token-cache contents are never verified, so a tampered cache passes source-identity validation and is trained on |
| F3 | P2 | `audit` derives the expected window count from the cache's own metadata, so a self-consistent truncated cache reports `ok=true` |
| F4 | P2 | the `--cache-tokens` path retains every materialized token array (2.37 GiB on bulk) next to the memmap it just wrote |
| F5 | P2 | `evaluate-offline` exits 0 when filtering leaves zero windows |

### 13.1 F1 - the test split must not drive model selection

Design (frozen):

1. `training_config.py`: `eval_split` default becomes `"validation"`. In
   `validate()`, immediately after the existing membership check, raise
   `TrainingConfigError` when `eval_split == "test"`, with a message stating
   that the test split is reserved for the final report and must never drive
   model selection. The config guard, not review diligence, is what makes
   test-set selection impossible.
2. `configs/diffusion/x2_50hz_train.yaml`: `eval: {split: validation, seed: 0}`.
3. `checkpoint.py` + `trainer.py`: persist `selection_split` in checkpoint
   metadata - the split the best checkpoint was selected on, or `None` when no
   best checkpoint was written. The trainer already tracks `best_metric_split`.
   This is metadata, not identity: resume must tolerate its absence (legacy
   checkpoints) and must not treat it as a resume-identity mismatch.
4. `scripts/diffusion.py` `_command_evaluate_offline`: keep the default
   checkpoint `checkpoint-best.pt`, but when it is absent the error must name
   `--checkpoint <run>/checkpoint-last.pt` explicitly. After loading, refuse
   (`return 1`) any checkpoint whose recorded `selection_split == "test"`, with
   a message naming `checkpoint-last.pt` as the explicit alternative. Include
   `selection_split` in the printed payload so no report can present a
   test-selected checkpoint as clean held-out evidence. A `None`
   `selection_split` (legacy or `last`) is allowed and reported as `null`.

Regressions (each must be shown to fail before its fix):

- `TrainingConfig(eval_split="test").validate()` raises; the default is
  `"validation"`.
- The shipped YAML resolves to `eval_split == "validation"` (update the
  existing `test_training_yaml_matches_the_frozen_config` assertion if it pinned
  `test`).
- A trainer run that writes a best checkpoint records
  `selection_split == "validation"`.
- `evaluate-offline` refuses a checkpoint whose `selection_split` is `"test"`
  with a non-zero status and a message naming `checkpoint-last.pt`.

### 13.2 F2 - cache contents must be provable

Design (frozen):

1. Bump `_CACHE_VERSION` to a v2 string so every pre-digest cache is rejected
   with an actionable "rebuild the cache" message rather than silently trusted.
   The D1 stores and the D0 contract are untouched; only `.npy` token caches
   are invalidated.
2. `_cache_metadata(...)` gains a required `content_sha256` field: the SHA-256
   of the `.npy` file bytes.
3. Introduce a module constant of non-identity keys (`created_at`,
   `content_sha256`) and a single `_identity_metadata(metadata)` helper,
   replacing every current `_metadata_without_time` call site. The digest is
   excluded from identity comparison because a caller cannot know it in
   advance; that is exactly why it must be verified separately.
4. `build_token_cache`: compute the digest **while writing**, by streaming the
   temporary `.npy` in bounded chunks before `os.replace`, and store it in the
   metadata written after the replace. Never `read_bytes()` a multi-GiB file.
5. `_validate_cache`: after the existing metadata, shape, dtype and memmap
   checks, stream the file and compare against the stored digest, raising
   `WindowCacheError` on mismatch. A v2 cache missing the digest, or a v1
   cache, must raise rather than pass. This is what makes the existing
   `_validate_cache(self.path, metadata_path, metadata)` self-comparison in
   `TokenWindowDataset.__init__` a real content check.
6. Digest verification is O(file size) once per open by design; record that
   cost rather than hiding it.

Regressions:

- The reviewer's exact repro: build a cache, modify one cached token
  (e.g. `array[0,0,0] += 100`) directly in the `.npy`, then reopen through
  `TokenWindowDataset(..., source=...)` or revalidation - it must raise.
- Tampering with the metadata's `content_sha256` raises.
- Positive control: a freshly built cache opens cleanly and its digest equals
  an independently computed `sha256` of the file.

### 13.3 F3 - audit counts must come from D1

Design (frozen): `_validate_audit_cache` must derive the expected window count
and shape from the loaded dataset - using the cache metadata's `split`,
`motion_ids` and `limit` but counting through D1's own selection logic - and
pass those derived values into `_validate_cache` as `expected`. It must require
`metadata["window_count"] == derived_count` and `shape == (derived_count,
window_steps, token_dimension)`, reporting `ok=false` on any mismatch. Counting
must stream (bounded peak), not materialize every split.

Regression - the discriminating construction, chosen because it is the only
shape a digest cannot catch: build a source whose split holds 1 window and a
cache that is **perfectly self-consistent but wrong** - truncated to 0 rows with
metadata `window_count=0`, `shape=[0,41,231]`, and `content_sha256` recomputed
over the truncated file. `audit_dataset` must still report `ok=false`, proving
F3 is not redundant with F2.

### 13.4 F4 - the bounded path must be bounded

Design (frozen): `build_token_cache` already streams row-wise into a memmap, and
`_iter_window_records` already "yield[s] records one at a time so cache
construction stays bounded". The retention is purely the CLI calling
`load_window_records` (which is `list(_iter_window_records(...))`) and holding
every token array for the whole run.

1. Add a frozen `WindowProvenance` dataclass carrying
   `split, group_key, motion_id, phase, ou_noise_norm, pair_id, start_tick` and
   **no tokens**, with the same split/phase validation as `WindowRecord`.
2. Add a public streaming `iter_window_provenance(...)` built on the existing
   iterator, dropping each record's token array as it advances, and export it.
3. Relax `sampling_weights` and `TokenWindowDataset.records` typing to accept
   either type. `sampling_weights` reads only `record.phase`, and the dataset
   reads only `.split` and `.phase`, so behaviour must be identical - prove it
   by equivalence, not by assertion.
4. In `make_dataset`, the `--cache-tokens` path must not call
   `load_window_records`: build the cache, take provenance from the streaming
   helper, and construct `TokenWindowDataset(cache_path, provenance, ...)`.
   Use `len(provenance)` for the reported train window count. The non-cache
   path keeps materializing tensors, because it needs them.
5. Removing the redundant duplicate `path = output_dir / ...` assignment in the
   same block is allowed cleanup.

Regressions:

- Equivalence: `sampling_weights(provenance)` equals `sampling_weights(records)`
  for the same windows, and dataset `len`/`weights` agree.
- Structural discriminator: no object retained by the cache path carries a
  `(41, 231)` token array (assert the provenance objects have no `tokens`). A
  structural test is used deliberately because peak-memory assertions are
   environment-dependent; state that reason in the test.

### 13.5 F5 - an empty result must not report success

Design (frozen): `_command_evaluate_offline` returns non-zero when
`report.windows == 0` **unless** `args.max_windows == 0`, which is the
intentional plumbing smoke. That covers both the empty-split default and a
positive `--max-windows N` that yields nothing. The stderr message must name
the split and say that no windows were selected.

Regressions: `--max-windows 0` exits 0; an empty requested split exits non-zero
with a message naming the split (`tennis_000` on the pilot is the real case).

### 13.6 Constraints for this round

- Do not modify the D0 contract, the D1 datasets, `logs/` artifacts, the seven
  unrelated in-flight files, or `docs/source/changelog.rst`.
- Do not change diffusion mathematics: schedule, noising, sampler grid and
  `eta`, the x0 objective, model architecture, or contract constants.
- No training beyond CPU unit gates, no bulk/NAS access, no export, no
  hardware. Never stage, commit, reset, checkout, stash, push, or use
  worktrees/branches. Use `uv run --no-sync` and `CUDA_VISIBLE_DEVICES=''`.
- Existing caches become invalid by design (the v2 bump). Do not rebuild,
  delete, or migrate any cache under `logs/`; the fix must simply reject them
   with an actionable message.
- Every regression must be shown to fail before its fix, with the scratch
  revert restored to a byte-identical file (report `sha256sum` before and
  after), and the full suite must pass as the final action.
- A `checkpoint-last.pt` from before this round has `selection_split=None` and
  remains evaluable; this round's earlier gate artifacts are not invalidated.

### 13.7 Acceptance for the fix round

The round is acceptable when: the five regressions exist and each is shown to
fail without its fix; the full diffusion suite passes (122 passed + 1 skipped
before this round, strictly more after); scoped ruff/ty are clean; no file
outside the intended set changed; and an independent adversarial pass cannot
reproduce any of the five original findings. Model quality remains open and is
not claimed by this round.

### 13.8 Fix-round record

The five post-review findings were implemented and covered by CPU regressions;
fail-without-fix transcripts and byte-identical scratch restores are recorded in
`logs/diffusion/d2-fix-round-20260930/verification.txt`.

- **F1 — FIXED** (`training_config.py:57,146-152`, `checkpoint.py:42,169-170,289-327`, `trainer.py` checkpoint/report paths, `scripts/diffusion.py:891-913,943-951,978-980`, and the shipped YAML): model selection now defaults to validation, rejects test selection, persists non-identity `selection_split`, tolerates legacy absence, rejects test-selected offline checkpoints, and reports the selection split. Regressions: `test_config_defaults_to_validation_and_rejects_test_selection`, `test_training_yaml_matches_the_frozen_config`, `test_best_checkpoint_records_the_validation_selection_split`, `test_legacy_checkpoint_without_selection_split_is_tolerated`, and `test_evaluate_offline_refuses_test_selected_checkpoint`. Fail proof: verification transcript F1.
- **F2 — FIXED** (`window_dataset.py:37-38,380-480,582-588,636-639`): token caches are v2, carry a streamed `.npy` SHA-256, compare identity through one helper that excludes only creation time and content digest, and verify bytes on every open with an O(file-size) bounded streaming pass. Regressions: `test_token_cache_digest_rejects_array_tampering`, `test_token_cache_digest_rejects_metadata_tampering`, `test_token_cache_v1_requires_actionable_rebuild`, plus the positive digest assertion in `test_token_cache_round_trip_filter_and_stale_metadata_rejection`. Fail proof: verification transcript F2.
- **F3 — FIXED** (`evaluation.py:531-605`): audit cache validation streams D1-selected provenance using metadata filters, derives count and contract shape, and rejects self-consistent truncated caches before byte validation. Regression: `test_audit_rejects_self_consistent_truncated_cache_from_d1_count`. Fail proof: verification transcript F3.
- **F4 — FIXED** (`window_dataset.py:154-318,623-772`, package exports, `scripts/diffusion.py:723-774`): frozen token-free `WindowProvenance` and streaming iterator are exported; cache training retains provenance only and uses its length, while non-cache training remains materialized. Regression: `test_window_provenance_stream_is_token_free_and_weight_equivalent` (the test documents why a structural bound is used instead of environment-dependent peak-memory measurement). Fail proof: verification transcript F4.
- **F5 — FIXED** (`scripts/diffusion.py:985-990`): offline evaluation returns non-zero for an empty selected split unless `--max-windows 0`, and names the split plus “no windows were selected”. Regression: `test_evaluate_offline_empty_split_requires_nonzero_unless_zero_bound`. Fail proof: verification transcript F5.

Scoped format, lint and type checks passed. The provisional diffusion suite
passed **129 tests with 1 skip**; the final suite result is appended to the
verification transcript as the last action. No model-quality, training-scale,
hardware, export, or NAS claim is made by this fix round.

### 13.9 Parent hardening after two independent verification lanes

Two independent lanes ran after the writer, deliberately with different
capabilities. `r1-review` was a static `reviewer`-agent pass with no shell and
said so explicitly, treating the writer's transcript as corroboration rather
than proof. `r2-adversarial` executed everything, kept all scratch data under
`/tmp`, and could not reproduce any of F1-F5: the test-selection config raises,
a real `selection_split=test` checkpoint is refused, one-sided cache tampering
(and a missing or forged digest) is rejected, the self-consistent zero-row
cache still reports `ok=false` against a real D1 source, and the real
`--cache-tokens` CLI path retains zero token objects with identical ordering,
weights and counts. Its attempted defeats that **failed** are also load-
bearing: a truncated `.npy`, a metadata swap between two valid builds, and a
motion-filtered audit were all correctly rejected.

That pass left three residuals, all in the selection-provenance seam:

- **R-a** - an unknown or whitespace-variant `selection_split` (`"banana"`,
  `"test "`) was accepted, because only `isinstance(str)` was checked and the
  CLI compared `== "test"` exactly (r1 P2, reproduced by r2 as A3).
- **R-b** - `best_metric_split="test"` hidden behind `selection_split=None`
  was accepted (r2 A3b). This is not hypothetical: the legacy config fallback
  reaches it, and it applied to this round's own test-selected
  `all-motions-300/checkpoint-best.pt`. Both lanes and the parent converged on
  this independently.
- **R-c** - a coordinated rewrite of both the `.npy` bytes and the stored
  digest is accepted (r2 A5). Recorded as an **accepted limitation**, not a
  defect: a digest stored beside the data it describes cannot authenticate it,
  and no caller can know the expected digest without rebuilding the cache.
  Defeating that needs a trusted root outside the cache directory, or binding
  the digest into the training checkpoint. Section 13.2 asked for provable
  contents, not a signature, so no change is made for R-c.

F6 closes R-a and R-b (`checkpoint.py`, `scripts/diffusion.py`): one
`_validate_split_field` enum check on both save and load, rejection of a
contradictory `selection_split`/`best_metric_split` pair, and a CLI check on the
**effective** selection split (`selection_split`, else `best_metric_split`) that
refuses the test split unless the artifact is `checkpoint-last.pt`. `last` is
allowed because it is not a selection artifact - it was never chosen by any
split - and refusing it would break re-running this stage's own gate evidence;
its provenance is instead disclosed through `selection_split`,
`best_metric_split` and `test_selection_provenance` in the report payload.

Regressions (6, all in `tests/test_tracking_diffusion_selection_provenance.py`):
`test_save_rejects_unknown_selection_split`,
`test_save_rejects_contradictory_selection_splits`,
`test_load_rejects_unknown_selection_split`,
`test_load_rejects_contradictory_selection_splits`,
`test_evaluate_offline_refuses_legacy_best_selected_on_test`,
`test_evaluate_offline_allows_legacy_last_but_flags_test_provenance`.

Fail-proof: with the three guards neutralised **all six fail** (6 failed), and
the restore was byte-identical (`checkpoint.py`
`988e55ce508b0b12b113dc6157a48f793678cc32d7266a85d5199673ffd1ebd7`,
`scripts/diffusion.py`
`bcc9dc7888f2fa4d4fbbd62a52a9ed3589b761249c01d2582feafab7aba694e4`) against
pre-neutralisation hashes, after which all six pass. Real-artifact check on this
round's own evidence: default resolution of `all-motions-300` now refuses the
test-selected best and names the explicit `last` path, while the explicit `last`
checkpoint remains evaluable and reports
`test_selection_provenance: true`. Suite after F6: **135 passed, 1 skipped**.
