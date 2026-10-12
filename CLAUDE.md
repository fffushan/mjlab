# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

## Offline X2 VAE teacher-fidelity replay

The repository includes a compact recorded-state evaluation for the X2 VAE in
``scripts/experiments/vae_teacher_student_replay/``. It feeds the frozen teacher
and VAE student the same recorded reference, proprioception, anchor, gyro and
previous-action state, then reports teacher/student raw-action and target
position disagreement. This is a one-step **teacher-fidelity** evaluation: it
measures whether the VAE reproduces teacher actions, and does not claim that a
teacher would remain stable in closed loop or on hardware.

The policy-state fixture is tracked in
``tests/fixtures/vae_teacher_student/``. Frozen model files are supplied
explicitly rather than loaded from mutable training/deployment paths:

```sh
uv run python scripts/experiments/vae_teacher_student_replay/evaluate.py \
  --states tests/fixtures/vae_teacher_student/recorded_states.npz \
  --model-dir /path/to/frozen/models \
  --output-dir logs/diagnostics/vae_teacher_student_replay
```

See the evaluator README for the required model files, state layout and
provenance limits.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

## Remote training queue monitor

`tools/queue_status.sh` monitors and manages the remote GPU training queue. Pass
an SSH command as one quoted argument, for example:

```sh
REMOTE='ssh -p 31505 fushan@10.14.64.37'

# Show queue states, progress and GPU occupancy.
tools/queue_status.sh "$REMOTE"

# Show the queued command rows, dispatcher diagnostics and recent log entries.
tools/queue_status.sh --jobs --dispatch --tail 50 "$REMOTE"
```

The default operation is read-only. `--stop` stops only the queue dispatcher;
training jobs already running in their tmux sessions continue. `--remove JOB_ID`
removes a non-running job row from `jobs.tsv` and retains its state and batch
artifacts; it refuses to stop or remove an active job.

To enqueue a job, put the remote SSH specification before `--`; every argument
after `--` is forwarded to `/home/fushan/mjlab/run_task.sh` without
queue-specific task parsing:

```sh
tools/queue_status.sh \\
  --add inc004 "$REMOTE" -- \\
  Mjlab-Tracking-Flat-AgiBot-X2-No-State-Estimation-Increated-Perturbations \\
  30000 tennis/single_004_zhanghongyu_agibot_x2_tracking.npz
```

Extra or future `run_task.sh` arguments can be passed the same way. The wrapper
shell-quotes them when writing `jobs.tsv`, preserving argument boundaries when
the dispatcher executes the command. The queue ID is the only task metadata
validated by the wrapper; it must be a safe identifier because it is used in
state filenames and the automatically generated `train_<JOB_ID>` tmux session.
