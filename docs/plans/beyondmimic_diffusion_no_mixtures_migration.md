# X2 diffusion no-mixture representation

The active diffusion schema is
`mjlab-x2-state-latent-d1-no-mixtures-v1` (contract SHA256
`f5237d4327fd7cfd4c065604806555a43f8afa97c0ea17a9e59a3482a6d0caed`). At 50 Hz
each window has 8 past steps, current, and 32 future steps (`41` rows). Each
row is the concatenation of:

* `135` train-normalized physical hybrid-state values (`15 + 6 * 20`), and
* the unchanged train-normalized frozen VAE `32`-D latent.

Thus the token width is exactly `167`. The state transform is elementwise
`(physical - train_mean) / train_std`; there are no random rows, emphasis
multipliers, or projected/additional channels. `ProjectionBundle` remains a
compatibility class name for callers, but active bundles contain identity
matrices and zero added rows. DDPM training, DDIM/DDPM schedules, masks,
attention, and model dimensions otherwise retain their previous semantics.

## Legacy data and checkpoints

The historical `mjlab-x2-state-latent-d0-v1` YAML, stores, projected bundle, and
checkpoints remain readable only when explicitly opened with that historical
contract. Active loaders fail closed on a legacy projected bundle with an
actionable migration error; they do not truncate or reinterpret channels.

To rebuild an existing raw D0 store without recollecting simulator data:

```sh
UV_PROJECT_ENVIRONMENT=/home/agiuser/projects/mjlab/.venv \
PYTHONPATH=$PWD/src \
uv run --no-sync python -m mjlab.scripts.diffusion migrate-no-mixtures \
  --source-directory /path/to/d1-bulk-par \
  --destination-directory /path/to/d1-no-mixtures \
  --source-contract docs/plans/beyondmimic_diffusion_d0_contract.yaml \
  --contract docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml
```

The destination must be new and is a derived dataset: raw physical rows,
latent values, eligibility, split assignments, VAE/robot/frame/state semantics,
and source manifest provenance are copied; fresh statistics are fitted from
training windows only. A full rebuild is intentionally not launched by this
change. Train and checkpoint under a new output directory using the new
contract; never overwrite historical artifacts.
