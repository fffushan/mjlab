"""X0 flow-matching family for X2 tracking (implementation Stages 1-4).

This package holds the opt-in flow family: the exact linear Gaussian noising
process, the x0 Euler sampler, bounded training, strict flow checkpoints and
offline generated-token evaluation.  The flow CLI lives in
:mod:`mjlab.scripts.flow_matching`.  The package never reads or writes diffusion
artifacts; the frozen diffusion package remains the only source of diffusion
behavior, noise schedules and diffusion checkpoints.

The mathematical contract is fixed: integer levels ``k`` in
``[0, process.max_level]``, noise fraction ``s = k / max_level``, noising
``x_s = (1 - s) * x0 + s * e`` with independent Gaussian ``e``, an ``x0``
network output, and the Euler update ``x_r = (r / s) * x_s + (1 - r / s) * x0_hat``
for ``r < s``.  There is no history dropout in this family.
"""

from .checkpoint import (
  FLOW_CHECKPOINT_FORMAT,
  FlowCheckpointError,
  FlowCheckpointState,
  identity_sha256,
  load_flow_checkpoint,
  model_identity,
  save_flow_checkpoint,
)
from .evaluation import (
  GenerationReport,
  evaluate_generation,
)
from .process import (
  DEFAULT_MAX_LEVEL,
  DEFAULT_PROCESS,
  FAMILY,
  FlowMatchingError,
  LevelPair,
  LinearNoisedPair,
  LinearNoiseProcess,
  add_linear_noise,
  draw_gaussian,
  sample_independent_levels,
)
from .sampler import (
  DEFAULT_CALLS,
  FlowConditions,
  FlowGrid,
  FlowSamplerConfig,
  FlowSamplerDiagnostics,
  euler_x0_update,
  sample_trajectory,
)
from .trainer import (
  FlowTrainer,
  FlowTrainerError,
  FlowTrainResult,
)
from .training_config import FlowTrainingConfig, FlowTrainingConfigError

__all__ = [
  "DEFAULT_CALLS",
  "DEFAULT_MAX_LEVEL",
  "DEFAULT_PROCESS",
  "FAMILY",
  "FLOW_CHECKPOINT_FORMAT",
  "FlowCheckpointError",
  "FlowCheckpointState",
  "FlowConditions",
  "FlowGrid",
  "FlowMatchingError",
  "FlowSamplerConfig",
  "FlowSamplerDiagnostics",
  "FlowTrainResult",
  "FlowTrainer",
  "FlowTrainerError",
  "FlowTrainingConfig",
  "FlowTrainingConfigError",
  "GenerationReport",
  "LevelPair",
  "LinearNoisedPair",
  "LinearNoiseProcess",
  "add_linear_noise",
  "draw_gaussian",
  "euler_x0_update",
  "evaluate_generation",
  "identity_sha256",
  "load_flow_checkpoint",
  "model_identity",
  "sample_independent_levels",
  "sample_trajectory",
  "save_flow_checkpoint",
]
