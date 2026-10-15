"""Deploy export for the X2 multi-motion tracking teacher.

The teacher covers many reference clips with one set of weights, so it has no single clip
to embed in the export. Its deploy graph is therefore the raw ``obs -> actions`` model,
and the reference travels as a separate file that the caller selects.

That split is what allows one policy to replay a clip the teacher never trained on: the
reference file carries tables, not weights. It also means the exported observation layout
must describe itself, because the reader can no longer recover the reference contract from
embedded tensors. See :mod:`mjlab.tasks.tracking.rl.export_metadata`.
"""

from __future__ import annotations

import os
from pathlib import Path

import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.utils import WandbLogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.rl.export_metadata import (
  command_lookahead_steps,
  motion_export_metadata,
)


class MultiMotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that also writes the reference-free deploy graph for the teacher."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def deploy_metadata(self, run_name: str = "local") -> dict[str, object]:
    """Metadata for one teacher export, including the resolved preview offset."""
    env = self.env.unwrapped
    command = env.command_manager.get_term("motion")
    metadata: dict[str, object] = dict(get_base_metadata(env, run_name))
    metadata.update(
      motion_export_metadata(command.cfg, command_lookahead_steps(command))
    )
    return metadata

  def export_deploy_onnx(
    self,
    path: str,
    filename: str = "policy.onnx",
    run_name: str = "local",
    verbose: bool = False,
  ) -> Path:
    """Write ``obs -> actions`` plus the metadata the X2 loader compiles.

    The graph is the raw policy from ``as_onnx``, so the observation normalizer is baked
    in and the output is the deterministic mean. No reference tensor is embedded.
    """
    os.makedirs(path, exist_ok=True)
    self.export_policy_to_onnx(path, filename, verbose)
    onnx_path = Path(path) / filename
    attach_metadata_to_onnx(str(onnx_path), self.deploy_metadata(run_name))
    return onnx_path

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      is_wandb = isinstance(self.logger.writer, WandbLogWriter)
      run_name: str = (
        wandb.run.name if is_wandb and wandb.run else "local"  # type: ignore[assignment]
      )
      self.export_deploy_onnx(str(policy_dir), filename, run_name)
      if is_wandb and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
