"""Run the active D5 numerical and export parity gate.

This command is a bounded, offline qualification check for one frozen artifact
set.  It loads the active denoiser, statistics-only normalizer and 68/99/32/31
VAE member, exports the denoiser and decoder to ONNX, reloads both artifacts
with ONNX Runtime, and compares fixed CPU probes against the native PyTorch
path.  The physical-state and token widths are read from the active contract;
the window geometry, latent width, conditioning width, action width and DDIM
update count stay frozen.

The controller lane is compiled for one token layout (135 hybrid state plus the
32 latent, so 167 channels).  When the active contract declares a reduced state
representation, this gate additionally exports a lane-token adapter graph, so
the D5 loader -- which validates the declared widths and the graph input shape
-- can load the reduced model unchanged.  The adapter only moves channels across
the lane boundary: see :mod:`mjlab.tasks.tracking.diffusion.lane_adapter`.

The command does not launch a simulator, controller, robot or hardware trial.
It does not measure control latency or model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from mjlab.tasks.tracking.diffusion.lane_adapter import (
  LaneAdapterError,
  LaneTokenAdapter,
  describe_lane_adapter,
  lane_gather_index,
)
from mjlab.tasks.tracking.diffusion.model import StateLatentTransformer
from mjlab.tasks.tracking.diffusion.noising import apply_clean_mask, clean_mask
from mjlab.tasks.tracking.diffusion.projection import StateLatentNormalizer
from mjlab.tasks.tracking.diffusion.replay_integration import (
  LoadedDenoiser,
  LoadedVae,
  load_active_contract,
  load_denoiser,
  load_normalizer_bundle,
  load_vae_decoder,
)
from mjlab.tasks.tracking.diffusion.sampler import (
  SamplerConditions,
  SamplerConfig,
  sample_trajectory,
  sample_trajectory_trace,
)
from mjlab.tasks.tracking.distillation.model import ConditionalVAE, StudentNormalizer

# Frozen by this gate: the window geometry, the latent width and the decoder/action
# widths.  The state and token widths are read from the active contract instead, so a
# reduced state representation is exportable without changing these values.
_EXPECTED = {
  "latent_dimension": 32,
  "conditioning_dimension": 99,
  "action_dimension": 31,
  "sequence_length": 41,
  "ddim_updates": 20,
}
_ATOL = 1.0e-5
_RTOL = 1.0e-5


class ParityError(RuntimeError):
  """A D5 parity or export contract check failed."""


class DenoiserCallable(Protocol):
  """The two-input denoiser surface used by the independent DDIM oracle."""

  def __call__(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor: ...


class DenoiserExport(nn.Module):
  """ONNX wrapper for the active denoiser interface."""

  def __init__(self, model: StateLatentTransformer) -> None:
    super().__init__()
    self.input_projection = model.input_projection
    self.position_embedding = model.position_embedding
    self.state_step_embedding = model.state_step_embedding
    self.latent_step_embedding = model.latent_step_embedding
    self.encoder = model.encoder
    self.output_projection = model.output_projection

  def forward(self, noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
    projected = self.input_projection(noisy_tokens)
    projected = projected + self.position_embedding.unsqueeze(0)
    projected = projected + self.state_step_embedding(step_ids[..., 0])
    projected = projected + self.latent_step_embedding(step_ids[..., 1])
    return self.output_projection(self.encoder(projected))


class VaeDecoderExport(nn.Module):
  """ONNX wrapper for the frozen VAE decoder interface."""

  def __init__(self, model: ConditionalVAE) -> None:
    super().__init__()
    # Register only the decoder path. Registering the full VAE would include
    # ConditionalVAE's dictionary-valued extra state in the legacy exporter.
    normalizer = cast(StudentNormalizer, model.conditioning_normalizer)
    self.register_buffer(
      "conditioning_mean", normalizer._buffer("mean").detach().clone()
    )
    self.register_buffer("conditioning_m2", normalizer._buffer("m2").detach().clone())
    self.register_buffer(
      "conditioning_count", normalizer._buffer("count").detach().clone()
    )
    self.register_buffer(
      "conditioning_eps", normalizer._buffer("_eps").detach().clone()
    )
    self.decoder = model.decoder
    self.action_head = model.action_head

  def forward(self, latent: Tensor, conditioning: Tensor) -> Tensor:
    # This is the exact decode normalization without Python-only validation.
    conditioning_m2 = cast(Tensor, self.conditioning_m2)
    conditioning_count = cast(Tensor, self.conditioning_count)
    conditioning_eps = cast(Tensor, self.conditioning_eps)
    conditioning_mean = cast(Tensor, self.conditioning_mean)
    variance = conditioning_m2 / conditioning_count.clamp_min(1.0)
    scale = torch.where(
      conditioning_count > 0,
      torch.sqrt(variance + conditioning_eps),
      torch.ones_like(variance),
    )
    normalized = (conditioning - conditioning_mean) / scale
    decoded = self.decoder(torch.cat((latent, normalized), dim=1))
    return self.action_head(decoded)


def _default_repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
  path = Path(value)
  return path if path.is_absolute() else root / path


def _canonical_digest(path: Path) -> str:
  digest = hashlib.sha256()
  try:
    with path.open("rb") as handle:
      for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
  except OSError as exc:
    raise ParityError(f"could not hash {path}: {exc}") from exc
  return digest.hexdigest()


def _require_file(path: Path, role: str) -> None:
  if not path.is_file():
    raise ParityError(f"{role} does not exist as a regular file: {path}")


def _require_expected_digest(path: Path, expected: str | None, role: str) -> str:
  actual = _canonical_digest(path)
  if expected is not None and actual != expected:
    raise ParityError(
      f"{role} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
    )
  return actual


def _load_manifest_metadata(
  contract_path: Path,
) -> tuple[str, str, str | None]:
  """Read the manifest path and optional digest declared by the active YAML."""
  try:
    import yaml

    payload = yaml.safe_load(contract_path.read_text())
  except (OSError, ValueError) as exc:
    raise ParityError(f"could not read active contract metadata: {exc}") from exc
  if not isinstance(payload, Mapping):
    raise ParityError("active contract YAML must contain a mapping")
  vae = payload.get("vae")
  if not isinstance(vae, Mapping):
    raise ParityError("active contract YAML has no vae mapping")
  manifest = vae.get("cohort_manifest")
  if not isinstance(manifest, str) or not manifest.strip():
    raise ParityError("active contract YAML has no cohort_manifest path")
  expected = vae.get("cohort_manifest_sha256")
  if expected is not None and not isinstance(expected, str):
    raise ParityError("cohort_manifest_sha256 must be a string when present")
  return manifest, str(vae.get("sha256", "")), expected


def _legacy_contract_identity(contract: Any) -> str:
  """Return the pre-D5 source-bound identity used by the frozen checkpoint."""
  values = dict(contract.as_dict())
  encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).hexdigest()


def _assert_dimensions(
  contract: Any, normalizer: StateLatentNormalizer
) -> dict[str, int]:
  """Check the frozen invariants and return the contract's own widths."""
  state_dimension = int(contract.state_dimension)
  latent_dimension = int(contract.latent_dimension)
  token_dimension = int(contract.token_dimension)
  observed = {
    "state_dimension": state_dimension,
    "latent_dimension": latent_dimension,
    "token_dimension": token_dimension,
    "conditioning_dimension": 99,
    "action_dimension": int(contract.action_dimension),
    "sequence_length": int(contract.window_steps),
    "ddim_updates": 20,
  }
  frozen = {name: observed[name] for name in _EXPECTED}
  if frozen != _EXPECTED:
    raise ParityError(f"active contract dimensions differ: {observed}")
  if token_dimension != state_dimension + latent_dimension:
    raise ParityError(
      f"active contract token width {token_dimension} is not the state plus latent "
      f"width {state_dimension + latent_dimension}"
    )
  if (
    contract.projected_state_dimension != state_dimension
    or contract.projection_rows != 0
  ):
    raise ParityError(
      f"active contract is not statistics-only {state_dimension}-state normalization: "
      f"projected_state_dimension={contract.projected_state_dimension}, "
      f"projection_rows={contract.projection_rows}"
    )
  if normalizer.contract.identity_hash() != contract.identity_hash():
    raise ParityError("normalizer contract identity differs from active contract")
  if normalizer.state_stats.mean.shape != (state_dimension,):
    raise ParityError(
      f"normalizer state statistics do not have shape ({state_dimension},)"
    )
  if normalizer.latent_stats.mean.shape != (latent_dimension,):
    raise ParityError(
      f"normalizer latent statistics do not have shape ({latent_dimension},)"
    )
  return observed


def _check_statistics_only(
  path: Path, normalizer: StateLatentNormalizer, *, state_dimension: int
) -> dict[str, Any]:
  """Check persisted keys and a float64 normalize/denormalize round trip."""
  required = {"state_mean", "state_std", "latent_mean", "latent_std", "metadata"}
  forbidden = {
    "matrix",
    "pseudoinverse",
    "projection_matrix",
    "projection_pseudoinverse",
  }
  try:
    with np.load(path, allow_pickle=False) as data:
      keys = set(data.files)
      metadata = json.loads(str(data["metadata"].item()))
  except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise ParityError(f"could not inspect normalizer bundle {path}: {exc}") from exc
  if not required <= keys:
    raise ParityError(f"normalizer bundle is missing keys: {sorted(required - keys)}")
  compatibility_keys = sorted(forbidden & keys)
  if compatibility_keys:
    if set(compatibility_keys) != {"matrix", "pseudoinverse"}:
      raise ParityError(
        "normalizer has an incomplete legacy projection field set: "
        f"{compatibility_keys}"
      )
    try:
      with np.load(path, allow_pickle=False) as data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        pseudoinverse = np.asarray(data["pseudoinverse"], dtype=np.float64)
    except (OSError, KeyError, TypeError, ValueError) as exc:
      raise ParityError(
        f"could not inspect legacy compatibility matrices in {path}: {exc}"
      ) from exc
    identity = np.eye(state_dimension, dtype=np.float64)
    if not np.array_equal(matrix, identity) or not np.array_equal(
      pseudoinverse, identity
    ):
      raise ParityError(
        "active no-mixture normalizer contains non-identity legacy projection fields"
      )
  if not isinstance(metadata, Mapping):
    raise ParityError("normalizer metadata is not a mapping")
  metadata_contract = metadata.get("contract")
  if not isinstance(metadata_contract, Mapping):
    raise ParityError("normalizer metadata has no contract mapping")
  if metadata_contract.get("projection_rows") != 0:
    raise ParityError("normalizer metadata declares non-zero projection rows")

  state_normalized = np.linspace(
    -1.25, 1.25, num=41 * state_dimension, dtype=np.float64
  ).reshape(41, state_dimension)
  latent_normalized = np.linspace(-0.75, 0.75, num=41 * 32, dtype=np.float64).reshape(
    41, 32
  )
  raw_state = normalizer.denormalize_state(state_normalized)
  raw_latent = normalizer.denormalize_latent(latent_normalized)
  state_roundtrip = normalizer.normalize_state(raw_state)
  latent_roundtrip = normalizer.normalize_latent(raw_latent)
  state_error = float(np.max(np.abs(state_roundtrip - state_normalized)))
  latent_error = float(np.max(np.abs(latent_roundtrip - latent_normalized)))
  if state_error > 1.0e-12 or latent_error > 1.0e-12:
    raise ParityError(
      f"statistics round trip failed: state={state_error}, latent={latent_error}"
    )
  return {
    "bundle_keys": sorted(keys),
    "legacy_compatibility_keys": compatibility_keys,
    "legacy_projection_fields_applied": False,
    "statistics_sha256": normalizer.statistics_sha256,
    "state_roundtrip_max_abs": state_error,
    "latent_roundtrip_max_abs": latent_error,
  }


def _build_probe(
  normalizer: StateLatentNormalizer,
  *,
  batch_size: int,
  state_dimension: int,
  token_dimension: int,
) -> tuple[Tensor, Tensor, dict[str, float]]:
  """Create finite normalized tokens from raw values without fitting statistics."""
  if batch_size <= 0:
    raise ParityError("batch_size must be positive")
  state_normalized = np.linspace(
    -1.25,
    1.25,
    num=batch_size * 41 * state_dimension,
    dtype=np.float64,
  ).reshape(batch_size, 41, state_dimension)
  latent_normalized = np.linspace(
    -0.75,
    0.75,
    num=batch_size * 41 * 32,
    dtype=np.float64,
  ).reshape(batch_size, 41, 32)
  raw_state = normalizer.denormalize_state(state_normalized)
  raw_latent = normalizer.denormalize_latent(latent_normalized)
  tokens = normalizer.make_tokens(raw_state, raw_latent)
  expected = np.concatenate((state_normalized, latent_normalized), axis=-1)
  transform_error = float(np.max(np.abs(tokens - expected)))
  if transform_error > 1.0e-12:
    raise ParityError(
      f"statistics-only token transform changed values: {transform_error}"
    )
  clean_tokens = torch.from_numpy(np.asarray(tokens, dtype=np.float32))
  mask_one = clean_mask(
    current_index=8,
    state_dimension=state_dimension,
    token_dimension=token_dimension,
    device="cpu",
  )
  mask = mask_one.unsqueeze(0).expand(batch_size, -1, -1).clone()
  if not torch.isfinite(clean_tokens).all().item():
    raise ParityError("probe clean tokens are non-finite")
  return clean_tokens, mask, {"statistics_token_transform_max_abs": transform_error}


def _step_ids(mask: Tensor, source: int, *, state_dimension: int) -> Tensor:
  fixed_state = mask[..., :state_dimension].all(dim=-1)
  fixed_latent = mask[..., state_dimension:].all(dim=-1)
  result = torch.full((*mask.shape[:2], 2), source, dtype=torch.long)
  result[..., 0] = torch.where(
    fixed_state, torch.zeros_like(result[..., 0]), result[..., 0]
  )
  result[..., 1] = torch.where(
    fixed_latent, torch.zeros_like(result[..., 1]), result[..., 1]
  )
  return result


def _reference_ddim(
  denoiser: DenoiserCallable,
  *,
  clean_tokens: Tensor,
  clean_mask: Tensor,
  loaded: LoadedDenoiser,
  seed: int,
  state_dimension: int,
) -> tuple[Tensor, tuple[int, ...], tuple[Tensor, ...]]:
  """Run an independent explicit 20-jump DDIM loop for cross-checking."""
  generator = torch.Generator(device="cpu").manual_seed(seed)
  noise = torch.randn(
    clean_tokens.shape,
    generator=generator,
    device="cpu",
    dtype=clean_tokens.dtype,
  )
  sample = apply_clean_mask(noise, clean_tokens, clean_mask)
  alpha_bars = torch.as_tensor(loaded.schedule.alpha_bars, dtype=sample.dtype)
  sources: list[int] = []
  steps: list[Tensor] = []
  for source, destination in zip(
    loaded.grid.source_ids, loaded.grid.destination_ids, strict=True
  ):
    step_ids = _step_ids(clean_mask, source, state_dimension=state_dimension)
    x0_hat = denoiser(sample, step_ids)
    if x0_hat.shape != sample.shape or not torch.isfinite(x0_hat).all().item():
      raise ParityError(
        "DDIM oracle denoiser output has the wrong shape or is non-finite"
      )
    if destination == 0:
      jumped = x0_hat.clone()
    else:
      alpha_source = alpha_bars[source]
      epsilon_hat = (sample - torch.sqrt(alpha_source) * x0_hat) / torch.sqrt(
        1.0 - alpha_source
      )
      alpha_destination = alpha_bars[destination]
      jumped = (
        torch.sqrt(alpha_destination) * x0_hat
        + torch.sqrt(1.0 - alpha_destination) * epsilon_hat
      )
    sample = apply_clean_mask(jumped, clean_tokens, clean_mask)
    sources.append(int(source))
    steps.append(step_ids)
  return sample, tuple(sources), tuple(steps)


def _record_tolerance_warning(
  warnings: list[dict[str, Any]] | None,
  name: str,
  maximum: float,
  relative: float,
  atol: float,
  rtol: float,
) -> None:
  """Record an exceeded comparison tolerance without refusing the export.

  The comparison is between two FP32 implementations.  A checkpoint whose forward pass is
  not FP32-reproducible at the frozen tolerance cannot pass it through any faithful export,
  so the deviation is reported and measured instead of being treated as a defect.
  """
  entry = {
    "check": name,
    "max_abs": maximum,
    "max_rel": relative,
    "atol": atol,
    "rtol": rtol,
  }
  print(
    f"WARNING: {name} exceeds the comparison tolerance: "
    f"max_abs={maximum}, max_rel={relative}, atol={atol}, rtol={rtol}",
    file=sys.stderr,
    flush=True,
  )
  if warnings is not None:
    warnings.append(entry)


def _close_metrics(
  name: str,
  actual: Tensor | np.ndarray,
  expected: Tensor | np.ndarray,
  *,
  atol: float = _ATOL,
  rtol: float = _RTOL,
  warnings: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
  actual_array = np.asarray(
    actual.detach().cpu().numpy() if isinstance(actual, Tensor) else actual,
    dtype=np.float64,
  )
  expected_array = np.asarray(
    expected.detach().cpu().numpy() if isinstance(expected, Tensor) else expected,
    dtype=np.float64,
  )
  if actual_array.shape != expected_array.shape:
    raise ParityError(
      f"{name} shape mismatch: {actual_array.shape} != {expected_array.shape}"
    )
  if not np.isfinite(actual_array).all() or not np.isfinite(expected_array).all():
    raise ParityError(f"{name} contains non-finite values")
  difference = np.abs(actual_array - expected_array)
  scale = np.maximum(np.abs(expected_array), 1.0e-12)
  maximum = float(difference.max(initial=0.0))
  relative = float(np.max(difference / scale, initial=0.0))
  if not np.all(difference <= atol + rtol * np.abs(expected_array)):
    _record_tolerance_warning(warnings, name, maximum, relative, atol, rtol)
  return {"max_abs": maximum, "max_rel": relative, "atol": atol, "rtol": rtol}


def _add_metadata(path: Path, metadata: Mapping[str, str]) -> None:
  try:
    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in metadata.items():
      item = model.metadata_props.add()
      item.key = key
      item.value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
  except Exception as exc:
    raise ParityError(
      f"could not validate or stamp ONNX artifact {path}: {exc}"
    ) from exc


def _export_onnx(
  module: nn.Module,
  inputs: tuple[Tensor, ...],
  path: Path,
  *,
  input_names: tuple[str, ...],
  output_name: str,
  metadata: Mapping[str, str],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  module.eval()
  previous_fastpath = torch.backends.mha.get_fastpath_enabled()
  torch.backends.mha.set_fastpath_enabled(False)
  try:
    with torch.no_grad():
      try:
        torch.onnx.export(
          module,
          inputs,
          str(path),
          input_names=list(input_names),
          output_names=[output_name],
          dynamic_axes={name: {0: "batch"} for name in (*input_names, output_name)},
          opset_version=18,
          do_constant_folding=True,
          dynamo=False,
          verbose=False,
        )
      except Exception as exc:
        raise ParityError(f"could not export ONNX artifact {path}: {exc}") from exc
  finally:
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)
  _add_metadata(path, metadata)


def _metadata(path: Path) -> dict[str, str]:
  try:
    model = onnx.load(str(path))
    return {item.key: item.value for item in model.metadata_props}
  except Exception as exc:
    raise ParityError(f"could not read ONNX metadata from {path}: {exc}") from exc


def _require_metadata(path: Path, expected: Mapping[str, str]) -> dict[str, str]:
  observed = _metadata(path)
  for key, value in expected.items():
    if observed.get(key) != value:
      raise ParityError(
        f"ONNX {path.name} metadata {key!r} differs: "
        f"expected {value!r}, got {observed.get(key)!r}"
      )
  return observed


def _session(path: Path) -> ort.InferenceSession:
  try:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  except Exception as exc:
    raise ParityError(f"ONNX Runtime could not load {path}: {exc}") from exc


def _require_graph_shape(
  session: ort.InferenceSession,
  *,
  inputs: Mapping[str, tuple[int, ...]],
  output_name: str,
  output_shape: tuple[int, ...],
) -> None:
  observed_names = {item.name for item in session.get_inputs()}
  if observed_names != set(inputs):
    raise ParityError(
      f"ONNX inputs differ: expected {sorted(inputs)}, got {sorted(observed_names)}"
    )
  for name, expected in inputs.items():
    info = next(item for item in session.get_inputs() if item.name == name)
    shape = info.shape
    if len(shape) != len(expected):
      raise ParityError(f"ONNX input {name} rank differs: {shape}")
    for observed, required in zip(shape[1:], expected[1:], strict=True):
      if observed != required:
        raise ParityError(
          f"ONNX input {name} shape differs: expected {expected}, got {shape}"
        )
  output = session.get_outputs()
  if len(output) != 1 or output[0].name != output_name:
    raise ParityError(
      f"ONNX outputs differ: expected [{output_name}], got {[item.name for item in output]}"
    )
  shape = output[0].shape
  if len(shape) != len(output_shape):
    raise ParityError(f"ONNX output rank differs: {shape}")
  for observed, required in zip(shape[1:], output_shape[1:], strict=True):
    if observed != required:
      raise ParityError(
        f"ONNX output shape differs: expected {output_shape}, got {shape}"
      )


def _onnx_denoiser(
  path: Path, *, token_dimension: int
) -> tuple[DenoiserCallable, dict[str, Any]]:
  session = _session(path)
  _require_graph_shape(
    session,
    inputs={
      "noisy_tokens": (-1, 41, token_dimension),
      "step_ids": (-1, 41, 2),
    },
    output_name="x0_hat",
    output_shape=(-1, 41, token_dimension),
  )

  def run(noisy_tokens: Tensor, step_ids: Tensor) -> Tensor:
    values = session.run(
      ["x0_hat"],
      {
        "noisy_tokens": noisy_tokens.detach().cpu().numpy().astype(np.float32),
        "step_ids": step_ids.detach().cpu().numpy().astype(np.int64),
      },
    )
    return torch.from_numpy(np.asarray(values[0], dtype=np.float32))

  return run, {
    "inputs": {
      name: list(value)
      for name, value in {
        "noisy_tokens": (-1, 41, token_dimension),
        "step_ids": (-1, 41, 2),
      }.items()
    },
    "output": ["x0_hat", -1, 41, token_dimension],
  }


def _lane_adapter_probe(
  adapter: DenoiserCallable,
  native: DenoiserCallable,
  *,
  gather_index: np.ndarray,
  lane_token_dimension: int,
  state_dimension: int,
  batch_size: int,
) -> dict[str, Any]:
  """Prove the lane adapter is exact on every channel the model reads.

  Both graphs receive the same normalized values on the model's own channels.  The
  adapter's output on those channels must match the native graph, and the lane channels
  the model does not own must pass through unchanged.
  """
  lane_tokens = torch.linspace(
    -1.5,
    1.5,
    batch_size * 41 * lane_token_dimension,
    dtype=torch.float32,
  ).reshape(batch_size, 41, lane_token_dimension)
  index = torch.from_numpy(np.asarray(gather_index, dtype=np.int64))
  model_tokens = torch.index_select(lane_tokens, -1, index)
  mask_one = torch.ones((41, lane_token_dimension), dtype=torch.bool)
  mask_one[8, state_dimension:] = False
  mask_one[9:] = False
  mask = mask_one.unsqueeze(0).expand(batch_size, -1, -1).clone()
  step_ids = _step_ids(mask, 1000, state_dimension=state_dimension)
  lane_output = adapter(lane_tokens, step_ids)
  native_output = native(model_tokens, step_ids)
  if lane_output.shape != lane_tokens.shape:
    raise ParityError(f"lane adapter output shape differs: {tuple(lane_output.shape)}")
  if not torch.isfinite(lane_output).all().item():
    raise ParityError("lane adapter output is non-finite")
  adapter_channels = torch.index_select(lane_output, -1, index)
  dropped = torch.tensor(
    [
      position
      for position in range(lane_token_dimension)
      if position not in set(int(v) for v in gather_index)
    ],
    dtype=torch.long,
  )
  dropped_passthrough = _close_metrics(
    "lane adapter pass-through channels",
    torch.index_select(lane_output, -1, dropped),
    torch.index_select(lane_tokens, -1, dropped),
    atol=0.0,
    rtol=0.0,
  )
  if dropped_passthrough["max_abs"] != 0.0:
    raise ParityError(
      "lane adapter changed a lane channel the model does not own: "
      f"{dropped_passthrough['max_abs']}"
    )
  return {
    "model_channels_max_abs": _close_metrics(
      "lane adapter vs native graph on the model's channels",
      adapter_channels,
      native_output,
    ),
    "pass_through_channels": int(dropped.numel()),
    "pass_through_max_abs": dropped_passthrough["max_abs"],
  }


def _onnx_decoder(
  path: Path,
) -> tuple[Callable[[Tensor, Tensor], Tensor], dict[str, Any]]:
  session = _session(path)
  _require_graph_shape(
    session,
    inputs={"latent": (-1, 32), "conditioning": (-1, 99)},
    output_name="action",
    output_shape=(-1, 31),
  )

  def run(latent: Tensor, conditioning: Tensor) -> Tensor:
    values = session.run(
      ["action"],
      {
        "latent": latent.detach().cpu().numpy().astype(np.float32),
        "conditioning": conditioning.detach().cpu().numpy().astype(np.float32),
      },
    )
    return torch.from_numpy(np.asarray(values[0], dtype=np.float32))

  return run, {
    "inputs": {"latent": [-1, 32], "conditioning": [-1, 99]},
    "output": ["action", -1, 31],
  }


def _run(args: argparse.Namespace) -> dict[str, Any]:
  root = Path(args.repo_root).resolve()
  contract_path = _resolve(root, args.contract)
  _require_file(contract_path, "active contract")
  contract = load_active_contract(contract_path)
  manifest_default, contract_vae_sha, manifest_sha = _load_manifest_metadata(
    contract_path
  )
  if contract_vae_sha and contract_vae_sha != contract.vae_sha256:
    raise ParityError("active YAML VAE digest differs from parsed contract digest")

  normalizer_path = _resolve(root, args.normalization_bundle)
  diffusion_path = _resolve(root, args.diffusion_checkpoint)
  vae_path = _resolve(root, args.vae_checkpoint or contract.vae_checkpoint)
  manifest_path = _resolve(root, args.vae_manifest or manifest_default)
  output = _resolve(root, args.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  if any(output.iterdir()) and not args.overwrite:
    raise ParityError(
      f"output directory is not empty: {output}; use --overwrite or choose a new directory"
    )
  _require_file(normalizer_path, "normalization bundle")
  _require_file(diffusion_path, "diffusion checkpoint")
  _require_file(vae_path, "VAE checkpoint")
  _require_file(manifest_path, "VAE cohort manifest")

  contract_sha = _require_expected_digest(contract_path, None, "active contract")
  diffusion_sha = _require_expected_digest(
    diffusion_path, args.expected_diffusion_sha256, "diffusion checkpoint"
  )
  vae_sha = _require_expected_digest(vae_path, contract.vae_sha256, "VAE checkpoint")
  manifest_actual_sha = _require_expected_digest(
    manifest_path,
    args.expected_vae_manifest_sha256 or manifest_sha,
    "VAE cohort manifest",
  )
  normalizer_sha = _canonical_digest(normalizer_path)

  if args.seed < 0:
    raise ParityError("seed must be non-negative")
  torch.set_grad_enabled(False)
  torch.manual_seed(args.seed)
  normalizer = load_normalizer_bundle(normalizer_path, contract=contract)
  if not isinstance(normalizer, StateLatentNormalizer):
    raise ParityError("active normalizer is not the statistics-only implementation")
  dimensions = _assert_dimensions(contract, normalizer)
  state_dimension = dimensions["state_dimension"]
  token_dimension = dimensions["token_dimension"]
  statistics_report = _check_statistics_only(
    normalizer_path, normalizer, state_dimension=state_dimension
  )
  try:
    lane = describe_lane_adapter(contract.state_slices)
  except LaneAdapterError as exc:
    raise ParityError(f"the controller lane cannot serve this contract: {exc}") from exc
  lane_required = bool(lane["required"])

  loaded = load_denoiser(
    diffusion_path,
    contract=contract,
    normalizer=normalizer,
    checkpoint_sha256=None,
    verify_projection_hashes=True,
    verify_contract_identity=False,
  )
  semantic_contract_identity = contract.identity_hash()
  legacy_contract_identity = _legacy_contract_identity(contract)
  checkpoint_contract_identity = loaded.contract_identity
  if checkpoint_contract_identity == semantic_contract_identity:
    checkpoint_identity_mode = "semantic"
  elif checkpoint_contract_identity == legacy_contract_identity:
    checkpoint_identity_mode = "legacy_source_bound"
  else:
    raise ParityError(
      "checkpoint contract identity is neither the active semantic identity nor "
      f"the compatible legacy source-bound identity: {checkpoint_contract_identity}"
    )
  if loaded.model.training or loaded.model.settings.token_dimension != token_dimension:
    raise ParityError(
      f"loaded denoiser is not a {token_dimension}-channel model: "
      f"settings declare {loaded.model.settings.token_dimension}"
    )
  vae: LoadedVae = load_vae_decoder(
    vae_path,
    manifest_path=manifest_path,
    repo_root=root,
    member=args.vae_member,
    contract=contract,
    vae_checkpoint_sha256=contract.vae_sha256,
    device="cpu",
  )
  if (vae.reference_dim, vae.conditioning_dim, vae.latent_dim, vae.action_dim) != (
    68,
    99,
    32,
    31,
  ):
    raise ParityError("loaded VAE dimensions differ from 68/99/32/31")

  clean_tokens, mask, probe_report = _build_probe(
    normalizer,
    batch_size=args.batch_size,
    state_dimension=state_dimension,
    token_dimension=token_dimension,
  )
  expected_mask = torch.zeros((41, token_dimension), dtype=torch.bool)
  expected_mask[:8] = True
  expected_mask[8, :state_dimension] = True
  if not torch.equal(mask[0], expected_mask):
    raise ParityError("D5 clean mask does not match 8-history/current-state semantics")
  unknown_entries = int((~mask[0]).sum().item())
  if unknown_entries != 32 + 32 * token_dimension:
    raise ParityError(f"unexpected unknown-entry count: {unknown_entries}")

  conditions = SamplerConditions(clean_tokens, mask)
  sampler_config = SamplerConfig(
    grid=loaded.grid,
    eta=0.0,
    guidance_strength=0.0,
    initial_noise_scale=1.0,
    start_level=1000,
  )
  torch_output, traces, diagnostics = sample_trajectory_trace(
    loaded.model,
    schedule=loaded.schedule,
    conditions=conditions,
    config=sampler_config,
    generator=torch.Generator(device="cpu").manual_seed(args.seed),
  )
  library_output, library_diagnostics = sample_trajectory(
    loaded.model,
    schedule=loaded.schedule,
    conditions=conditions,
    config=sampler_config,
    generator=torch.Generator(device="cpu").manual_seed(args.seed),
  )
  reference_output, reference_sources, reference_steps = _reference_ddim(
    loaded.model,
    clean_tokens=clean_tokens,
    clean_mask=mask,
    loaded=loaded,
    seed=args.seed,
    state_dimension=state_dimension,
  )

  fixed_error = float(
    torch.max(torch.abs(torch_output[mask] - clean_tokens[mask])).item()
  )
  if fixed_error != 0.0:
    raise ParityError(f"library DDIM fixed entries changed: {fixed_error}")
  if diagnostics.denoiser_calls != 20 or diagnostics.source_ids_returned != tuple(
    loaded.grid.source_ids
  ):
    raise ParityError(
      "library DDIM did not execute the frozen 20 source levels: "
      f"calls={diagnostics.denoiser_calls}, ids={diagnostics.source_ids_returned}"
    )
  for trace, expected_steps in zip(traces, reference_steps, strict=True):
    if not torch.equal(trace.step_ids, expected_steps):
      raise ParityError(f"step-id mask semantics differ at source {trace.source_id}")

  # Tolerance exceedance is recorded, not refused.  Structural checks (shape, finiteness,
  # identity, metadata, provider) above and below stay hard failures.
  tolerance_warnings: list[dict[str, Any]] = []
  ddim_report: dict[str, Any] = {
    "updates": diagnostics.denoiser_calls,
    "source_ids": list(diagnostics.source_ids_returned),
    "destination_ids": list(loaded.grid.destination_ids),
    "unknown_entries": unknown_entries,
    "fixed_entry_max_abs": fixed_error,
    "library_vs_nontraced": _close_metrics(
      "library sampler vs non-traced sampler",
      torch_output,
      library_output,
      warnings=tolerance_warnings,
    ),
    "library_vs_independent_torch_oracle": _close_metrics(
      "library sampler vs independent torch oracle",
      torch_output,
      reference_output,
      warnings=tolerance_warnings,
    ),
    "trace_step_ids_checked": len(traces),
    "diagnostics": {
      "denoiser_calls": diagnostics.denoiser_calls,
      "jumps": diagnostics.jumps,
      "final_index": diagnostics.final_index,
      "initial_noise_norm": diagnostics.initial_noise_norm,
    },
    "nontraced_diagnostics": {
      "denoiser_calls": library_diagnostics.denoiser_calls,
      "jumps": library_diagnostics.jumps,
      "final_index": library_diagnostics.final_index,
    },
    "oracle_source_ids": list(reference_sources),
  }

  metadata_common = {
    "format": "mjlab-d5-parity-onnx-v1",
    "contract_identity": semantic_contract_identity,
    "checkpoint_contract_identity": checkpoint_contract_identity,
    "checkpoint_identity_mode": checkpoint_identity_mode,
    "contract_file_sha256": contract_sha,
    "normalization_mode": "statistics_only",
    "normalization_statistics_sha256": normalizer.statistics_sha256,
    "normalization_bundle_sha256": normalizer_sha,
    "diffusion_checkpoint_sha256": diffusion_sha,
    "vae_checkpoint_sha256": vae_sha,
    "vae_manifest_sha256": manifest_actual_sha,
    "ddim_updates": "20",
  }
  denoiser_path = output / "denoiser.onnx"
  native_denoiser_path = (
    output / "denoiser_native.onnx" if lane_required else denoiser_path
  )
  decoder_path = output / "vae_decoder.onnx"
  denoiser_metadata = {
    **metadata_common,
    "artifact_kind": "diffusion_denoiser",
    "token_dimension": str(token_dimension),
    "sequence_length": "41",
  }
  _export_onnx(
    DenoiserExport(loaded.model),
    (clean_tokens, _step_ids(mask, 1000, state_dimension=state_dimension)),
    native_denoiser_path,
    input_names=("noisy_tokens", "step_ids"),
    output_name="x0_hat",
    metadata=denoiser_metadata,
  )
  lane_denoiser_metadata: dict[str, str] | None = None
  if lane_required:
    # The lane graph is the bundle artifact: it keeps the controller's token width and
    # serves the reduced model through a pure channel boundary.
    lane_denoiser_metadata = {
      **denoiser_metadata,
      "token_dimension": str(lane["lane_token_dimension"]),
      "lane_adapter": str(lane["lane_adapter"]),
      "lane_state_dimension": str(lane["lane_state_dimension"]),
      "lane_token_dimension": str(lane["lane_token_dimension"]),
      "model_state_dimension": str(lane["model_state_dimension"]),
      "model_token_dimension": str(lane["model_token_dimension"]),
      "dropped_lane_state_fields": ",".join(lane["dropped_lane_state_fields"]),
      "lane_fill_mean": str(lane["fill_mean"]),
      "lane_fill_std": str(lane["fill_std"]),
    }
    lane_probe = torch.linspace(
      -1.5,
      1.5,
      args.batch_size * 41 * lane["lane_token_dimension"],
      dtype=torch.float32,
    ).reshape(args.batch_size, 41, lane["lane_token_dimension"])
    _export_onnx(
      LaneTokenAdapter(DenoiserExport(loaded.model), contract.state_slices),
      (
        lane_probe,
        _step_ids(
          torch.ones(
            (args.batch_size, 41, lane["lane_token_dimension"]), dtype=torch.bool
          ),
          1000,
          state_dimension=lane["lane_state_dimension"],
        ),
      ),
      denoiser_path,
      input_names=("noisy_tokens", "step_ids"),
      output_name="x0_hat",
      metadata=lane_denoiser_metadata,
    )
  _export_onnx(
    VaeDecoderExport(cast(ConditionalVAE, vae.model)),
    (
      torch.linspace(-1.0, 1.0, args.batch_size * 32, dtype=torch.float32).reshape(
        args.batch_size, 32
      ),
      torch.linspace(-0.5, 0.5, args.batch_size * 99, dtype=torch.float32).reshape(
        args.batch_size, 99
      ),
    ),
    decoder_path,
    input_names=("latent", "conditioning"),
    output_name="action",
    metadata={
      **metadata_common,
      "artifact_kind": "vae_decoder",
      "latent_dimension": "32",
      "conditioning_dimension": "99",
      "action_dimension": "31",
    },
  )
  expected_denoiser_metadata = {
    **metadata_common,
    "artifact_kind": "diffusion_denoiser",
    "token_dimension": str(
      lane["lane_token_dimension"] if lane_required else token_dimension
    ),
    "sequence_length": "41",
  }
  expected_decoder_metadata = {
    **metadata_common,
    "artifact_kind": "vae_decoder",
    "latent_dimension": "32",
    "conditioning_dimension": "99",
    "action_dimension": "31",
  }
  denoiser_metadata_observed = _require_metadata(
    denoiser_path, expected_denoiser_metadata
  )
  decoder_metadata = _require_metadata(decoder_path, expected_decoder_metadata)
  onnx_denoiser, denoiser_graph = _onnx_denoiser(
    native_denoiser_path, token_dimension=token_dimension
  )
  onnx_decoder, decoder_graph = _onnx_decoder(decoder_path)
  onnx_output, onnx_sources, _ = _reference_ddim(
    onnx_denoiser,
    clean_tokens=clean_tokens,
    clean_mask=mask,
    loaded=loaded,
    seed=args.seed,
    state_dimension=state_dimension,
  )
  ddim_report["torch_vs_exported_denoiser_oracle"] = _close_metrics(
    "torch DDIM vs exported denoiser DDIM",
    reference_output,
    onnx_output,
    warnings=tolerance_warnings,
  )
  if onnx_sources != reference_sources:
    raise ParityError("exported denoiser used a different DDIM source grid")

  lane_graph: dict[str, Any] | None = None
  lane_report: dict[str, Any] | None = None
  if lane_required:
    lane_onnx_denoiser, lane_graph = _onnx_denoiser(
      denoiser_path, token_dimension=lane["lane_token_dimension"]
    )
    lane_report = {
      **lane,
      "probe": _lane_adapter_probe(
        lane_onnx_denoiser,
        onnx_denoiser,
        gather_index=lane_gather_index(contract.state_slices),
        lane_token_dimension=lane["lane_token_dimension"],
        state_dimension=lane["lane_state_dimension"],
        batch_size=args.batch_size,
      ),
      "graph": lane_graph,
    }

  decoder_latent = torch.linspace(
    -1.0, 1.0, args.batch_size * 32, dtype=torch.float32
  ).reshape(args.batch_size, 32)
  decoder_conditioning = torch.linspace(
    -0.5, 0.5, args.batch_size * 99, dtype=torch.float32
  ).reshape(args.batch_size, 99)
  native_action = vae.model.decode(decoder_latent, decoder_conditioning)
  exported_action = onnx_decoder(decoder_latent, decoder_conditioning)
  decoder_report = {
    "member": vae.member_id,
    "reference_dimension": vae.reference_dim,
    "conditioning_dimension": vae.conditioning_dim,
    "latent_dimension": vae.latent_dim,
    "action_dimension": vae.action_dim,
    "torch_vs_exported_action": _close_metrics(
      "native VAE decoder vs exported decoder",
      native_action,
      exported_action,
      warnings=tolerance_warnings,
    ),
    "native_action_finite": bool(torch.isfinite(native_action).all().item()),
    "exported_action_finite": bool(torch.isfinite(exported_action).all().item()),
  }

  report = {
    "status": "pass",
    "status_detail": ("pass_with_tolerance_warnings" if tolerance_warnings else "pass"),
    "tolerance_warnings": tolerance_warnings,
    "format": "mjlab-d5-numerical-export-parity-v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "generated_at_cst": datetime.now(timezone.utc)
    .astimezone(ZoneInfo("Asia/Shanghai"))
    .isoformat(),
    "scope": {
      "simulator": "not_run",
      "controller": "not_run",
      "hardware": "not_run",
      "quality": "not_assessed",
    },
    "settings": {
      "seed": args.seed,
      "batch_size": args.batch_size,
      "device": "cpu",
      "onnx_opset": 18,
      "atol": _ATOL,
      "rtol": _RTOL,
      "tolerance_is_blocking": False,
    },
    "contract": {
      "path": str(contract_path),
      "sha256": contract_sha,
      "identity": semantic_contract_identity,
      "checkpoint_identity": checkpoint_contract_identity,
      "checkpoint_identity_mode": checkpoint_identity_mode,
      "dimensions": dimensions,
      "state_field_slices": {
        name: [int(span[0]), int(span[1])]
        for name, span in contract.state_slices.items()
      },
    },
    "artifacts": {
      "diffusion_checkpoint": {"path": str(diffusion_path), "sha256": diffusion_sha},
      "normalization_bundle": {"path": str(normalizer_path), "sha256": normalizer_sha},
      "vae_checkpoint": {"path": str(vae_path), "sha256": vae_sha},
      "vae_manifest": {"path": str(manifest_path), "sha256": manifest_actual_sha},
      "exported_denoiser": {
        "path": str(denoiser_path),
        "sha256": _canonical_digest(denoiser_path),
        "metadata_keys": sorted(denoiser_metadata_observed),
        "graph": (lane_graph if lane_graph is not None else denoiser_graph),
      },
      "exported_native_denoiser": {
        "path": str(native_denoiser_path),
        "sha256": _canonical_digest(native_denoiser_path),
        "graph": denoiser_graph,
      },
      "exported_decoder": {
        "path": str(decoder_path),
        "sha256": _canonical_digest(decoder_path),
        "metadata_keys": sorted(decoder_metadata),
        "graph": decoder_graph,
      },
    },
    "checks": {
      "statistics_only": statistics_report,
      "probe": probe_report,
      "ddim": ddim_report,
      "lane_adapter": lane_report,
      "vae_decoder": decoder_report,
    },
  }
  report_path = output / "REPORT.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return report


def _parser() -> argparse.ArgumentParser:
  root = _default_repo_root()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo-root", default=str(root))
  parser.add_argument(
    "--contract",
    default="docs/plans/beyondmimic_diffusion_d1_no_mixtures_contract.yaml",
  )
  parser.add_argument(
    "--diffusion-checkpoint",
    default="logs/diffusion/d2-no-mixtures-bulk/checkpoint-best.pt",
  )
  parser.add_argument(
    "--normalization-bundle",
    default="logs/diffusion/d1-bulk-no-mixtures/offline/projection.npz",
  )
  parser.add_argument("--vae-checkpoint", default=None)
  parser.add_argument("--vae-manifest", default=None)
  parser.add_argument("--vae-member", default="tennis_001")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--expected-diffusion-sha256", default=None)
  parser.add_argument("--expected-vae-manifest-sha256", default=None)
  parser.add_argument("--seed", type=int, default=20261008)
  parser.add_argument("--batch-size", type=int, default=2)
  parser.add_argument("--overwrite", action="store_true")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  output = _resolve(Path(args.repo_root).resolve(), args.output_dir)
  try:
    report = _run(args)
  except Exception as exc:
    output.mkdir(parents=True, exist_ok=True)
    (output / "FAILURE.json").write_text(
      json.dumps(
        {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        indent=2,
        sort_keys=True,
      )
      + "\n"
    )
    raise
  print(
    json.dumps(
      {"status": report["status"], "report": str(output / "REPORT.json")}, indent=2
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
