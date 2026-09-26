"""Teacher foundation for BeyondMimic-style VAE distillation.

M1 provides the manifest/contract validation and the frozen teacher inference
bank. ``parity`` is intentionally not re-exported: it imports ONNX Runtime
lazily, and this package is imported eagerly with ``mjlab.tasks``.
"""

from mjlab.tasks.tracking.distillation.config import (
  CohortContract,
  DistillationError,
  Manifest,
  MissingValidationDependencyError,
  ResolvedTeacher,
  TeacherEntry,
  UnsupportedTeacherError,
  load_manifest,
  resolve_cohort,
)
from mjlab.tasks.tracking.distillation.teachers import (
  FrozenTeacher,
  TeacherBank,
  build_frozen_teacher,
  load_frozen_teacher,
)

__all__ = [
  "CohortContract",
  "DistillationError",
  "FrozenTeacher",
  "Manifest",
  "MissingValidationDependencyError",
  "ResolvedTeacher",
  "TeacherBank",
  "TeacherEntry",
  "UnsupportedTeacherError",
  "build_frozen_teacher",
  "load_frozen_teacher",
  "load_manifest",
  "resolve_cohort",
]
