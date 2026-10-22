# X2 joint-pose flow matching: implementation plan

Status: dataset, config and dataset loader exist. Runner, CLI, tests and
evaluation are not written. No training run is authorized.
Observed: 2026-10-22 (+08).

## Already built

| Path | Content |
|---|---|
| `src/mjlab/tasks/tracking/joint_pose/config.py` | `JointPoseDatasetCfg`, `FlowMatchingTrainerCfg`, `JointPoseFlowRunCfg`, `joint_pose_flow_cfg()` |
| `src/mjlab/tasks/tracking/joint_pose/dataset.py` | `JointPoseWindowDataset`: two sources, per-source sampling, split-leak refusal, identity record |
| `src/mjlab/tasks/tracking/joint_pose/__init__.py` | exports |
| `scripts/experiments/replay_survey/` | trackability survey, ledger, purge planner, mode analysis |
| `scripts/experiments/joint_space_prepare/add_reference_source.py` | reference window source |

Model, noise process and EMA are imported from the diffusion and flow-matching
modules. Nothing in those modules is edited, and no contract is used.

## Decisions of record

| # | Decision |
|---|---|
| 1 | Condition on the 8 history rows. Keep the history mask option with the existing semantics: with probability `history_dropout_probability` the whole history of a window is replaced by pure Gaussian so the model cannot read it. Default 0.0 |
| 2 | Both sources, reference sampling weight 30.0 |
| 3 | Failure-ledger downweighting is not used. Every window inside a source has weight 1.0 |
| 4 | Epoch means one full pass over the training windows. A run is bounded by `max_steps`, never by a fraction of an epoch |
| 5 | No purge |
| 6 | fp32 cache |
| 7 | Four-card server |
| 11 | x0 prediction with channel-mean MSE |
| 12 | Keep EMA |

## Stage 1 — runner

File: `src/mjlab/tasks/tracking/joint_pose/runner.py`

```python
class FlowMatchingRunner:
  def __init__(self, dataset, eval_datasets: dict[str, JointPoseWindowDataset],
               cfg: FlowMatchingTrainerCfg, device: str) -> None
  def learn(self) -> dict            # returns the final metrics summary
  def save(self, path: Path) -> None
  def load(self, path: Path) -> None
  def sample(self, history: torch.Tensor, steps: int) -> torch.Tensor
```

Training definition, stated exactly because it is the part that is easy to get
wrong:

| Item | Value |
|---|---|
| Model | `StateLatentTransformer(DenoiserSettings(token_dimension=68, sequence_length=41, ...))` imported from `diffusion/model.py` |
| Noise | `LinearNoiseProcess(max_level)`, integer level per window, `s = k / max_level` |
| Model input | rows 0:8 clean history (or pure Gaussian when the history mask fires), rows 8:41 noised at the sampled level |
| Loss target | rows 8:41 only, x0, channel-mean MSE |
| History mask | whole-window decision, `history_dropout_probability`, RNG captured for resume |
| Optimizer | AdamW, `learning_rate`, `weight_decay` |
| Schedule | linear warmup then cosine, computed from `max_steps` |
| Precision | bf16 autocast on CUDA, fp32 loss |
| Accumulation | `gradient_accumulation_steps` microbatches per update |
| Clipping | global norm `max_grad_norm` |
| EMA | `ema_power`, `ema_max_decay` |
| Sampling | `dataset.draw_indices(count, generator)`; never a per-row `WeightedRandomSampler` over 32 M rows |
| Logs | `logs/flow_matching/<experiment_name>/<timestamp>/` |
| Written once at start | `params/trainer.json`, `params/dataset.json`, `params/pointer.txt` |
| Written during the run | `metrics.jsonl`, `model_<step>.pt` every `checkpoint_every` |
| Resume | refuse when the dataset identity or any optimisation field differs, including `max_steps`: the warmup-plus-cosine schedule is planned over the whole budget, so a different budget is a different learning-rate curve |
| Evaluation parameters | the exponential moving average is the default sample source; a report must state which set produced its numbers |

`sample` runs the flow from pure Gaussian with the given clean history and
returns `[B, 41, 68]`; it is the only entry point the evaluation tool uses.

## Stage 2 — CLI

File: `src/mjlab/scripts/train_flow.py`, entry point `train-flow` in
`pyproject.toml`.

| Element | Requirement |
|---|---|
| Argument surface | `JointPoseFlowRunCfg` fields, `--device`, `--output` |
| Budget | `--max-steps` required; a run without it must not start |
| Behaviour | builds both datasets, prints counts and the reference share, then `learn()` |
| Style | tyro like `scripts/train.py`; no YAML, no contract, no schema hash |

## Stage 3 — tests

File: `tests/test_joint_pose_flow_training.py` plus
`tests/test_joint_pose_flow_dataset.py`.

| Test | Assertion |
|---|---|
| channel and window arithmetic | config refuses a mismatch |
| budget required | `require_budget()` raises without `max_steps` |
| synthetic dataset fixture | tiny `.npy` sources in `tmp_path`, correct `__len__`, dtype and shape |
| source sampling | with a fixed generator, the reference share matches `reference_share` within 2 % over 20 000 draws |
| split-leak refusal | a reference manifest that crosses splits is rejected |
| history mask | probability 1.0 replaces rows 0:8 and leaves rows 8:41 untouched |
| loss masking | rows 0:8 do not contribute to the reported loss |
| runner smoke | 3 updates on a 64-window fixture: loss finite, parameters changed, checkpoint written and reloaded |
| resume identity | a changed dataset identity is refused |

All tests run on CPU with a small model (`width=32, layers=1, heads=2`) and must
finish in under 60 s in total.

## Stage 4 — evaluation

File: `src/mjlab/tasks/tracking/joint_pose/evaluate.py`.

| Output | Definition |
|---|---|
| MSE per split | channel-mean x0 error on held-out windows |
| Baselines | repeat-last-history-row, and constant velocity; the model must beat both |
| Mode distance | histogram distance over the mode axes of `analyze_modes.py` between generated and reference windows |
| Replay export | write `K=200` generated windows as tracking motion files for `scripts/experiments/replay_survey/survey.py` |

## Review outcome (2026-10-22)

Three findings were raised and closed in the code:

| Finding | Resolution |
|---|---|
| Resume identity compared counts and file sizes only | identity now carries a sampled content digest of each cache, plus SHA-256 of `projection.npz`, `window_index.npz`, `assignments.json` and the reference manifest |
| EMA was updated and saved but never used | `evaluate.generate_windows` defaults to `parameters="ema"` |
| No test proved resume continuation, and `sample` was untested | three tests added: continuation against an uninterrupted run, sample shape and history preservation, and identity sensitivity to rewritten content |

Two defects were found by the tests rather than by review:

| Defect | Fix |
|---|---|
| The window-arithmetic check was a tautology, so `past_steps == window_steps` passed validation | refuse a window with no generated rows |
| With the last update at learning rate zero, one update was spent drawing windows and moving the EMA for no step | plan the schedule over `budget + 1` |

## Stage 5 — review

Two independent reviewers, read only:

| Reviewer | Scope |
|---|---|
| correctness | history conditioning, loss masking, split leakage, resume identity, sampling weights |
| numerics | autocast, EMA, accumulation, schedule, checkpoint round trip |

## Non-goals

| Excluded | Reason |
|---|---|
| Contract, schema hashes, projection bundle | not needed by a 68-channel self-describing dataset |
| Latent, VAE, action channels | outside the latent-free design |
| Any training launch | requires an explicit budget decision (#8, #9, #10) |
| Purge or dataset rewrite | decision #5 is "do not purge" |
| Goal or command conditioning | later change; v1 is unconditional given history |
