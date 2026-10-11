# MVP source report

The fixed-state coverage evaluator is CPU-only and evaluates a supplied student against frozen teacher action labels. It does not launch environments, call a teacher, update normalizers, or make hardware/deployment claims.

Family splits use a SHA-256 bucket keyed by `(seed, family_id)`, so descendants sharing a family stay together and existing family assignments are unchanged by row reordering or unrelated families. Supplied assignments carry row identity/family identity and are validated against current metadata before indexing.

Evaluation temporarily switches the model to inference mode and restores every module training flag, both normalizer state dicts, and the CPU RNG state. Hardware fixtures are rejected. Report provenance and quality claim are explicitly `teacher_fidelity_on_fixed_states_only` (CPU fixed states), not stability, recovery, or closed-loop quality.
