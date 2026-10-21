# X2 raw joint-pose collection

Opt-in research collector for the latent-free X2 pipeline. It rolls out the
frozen multi-motion PPO teacher and writes the **achieved simulator state**:
root-link orientation, joint position, and joint velocity. It does not train a
model, unpack a 6D rotation, align heading, build windows, load the VAE, or run
any reference adapter. Those steps belong to later preprocessing.

The collector reuses the evaluator's playback setup:

- `auto_reset=False`, so terminal state is read before a row is reset;
- the checkpoint's saved `lookahead_s` is required and restored (the teacher
  uses `0.2 s`), so the preview is not silently changed;
- actor observation corruption is on, and the inherited reset pose, velocity,
  and joint perturbations stay on;
- `standing_start_prob` is forced to `0`, so no standing-pose insert is added;
- the policy is loaded once, kept frozen, and evaluated under `torch.no_grad`.

`--sampling-mode start` (the default) starts every trajectory at its clip's
first frame. `--sampling-mode uniform` uses the existing multi-motion command's
uniform phase sampler. Each row samples its start frame independently; repeated
frames are possible. The reset perturbations and preview are unchanged. A uniform-phase
trajectory can reach its clip endpoint before the full-clip budget and ends
there without wrapping into a new clip.

The baseline scenario removes `push_robot`. The opt-in disturbed scenario
reinstalls the task's existing interval `push_by_setting_velocity` term. Each
push adds a uniformly sampled **world-frame root-velocity delta**, not a force
or torque.

## Scenario labels

`scenario` describes the configured treatment:

- `baseline`: no pushes. Reset, model, and sensor randomization remain on, so
  baseline is not noise-free.
- `disturbed`: interval root-velocity pushes are configured.

`trajectory_outcome` is separate: `complete`, `failure`, or `budget`. A
disturbed trajectory can have `trajectory_push_count == 0` when it finishes
before the first interval fires. The label still says `disturbed`; the zero
count says no push happened.

## Usage

Baseline (legacy defaults: 64 environments, one seed, start phase, no pushes):

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/joint_pose_collection/collect.py \
  --checkpoint-file logs/rsl_rl/agibot_x2_multi_motion_tracking/<run>/model_19999.pt \
  --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \
  --output-dir logs/joint_pose_collection/run_000 \
  --num-envs 64 --seed 0
```

Randomized starts over four finite seed runs:

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/joint_pose_collection/collect.py \
  --checkpoint-file <teacher-run>/model_19999.pt \
  --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \
  --output-dir logs/joint_pose_collection/random_phase \
  --num-envs 64 --seed 0 --num-seeds 4 \
  --sampling-mode uniform
```

Disturbed collection, with an explicit interval and velocity-delta ranges:

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/joint_pose_collection/collect.py \
  --checkpoint-file <teacher-run>/model_19999.pt \
  --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \
  --output-dir logs/joint_pose_collection/disturbed \
  --num-envs 64 --seed 0 --num-seeds 4 \
  --sampling-mode uniform \
  --push-interval-s 1.0 3.0 \
  --push-linear-velocity-range -0.5 0.5 \
  --push-angular-velocity-range -0.52 0.52
```

Pass more clips after one `--motion-file` flag. Relative clip paths resolve
from the repository root, the same convention as the multi-motion command.
`--max-steps` caps each trajectory below the clip length; omitted, the full
clip budget is used. Each clip rollout is one batched environment and one NPZ,
so only the current rollout is held in memory.

`--num-seeds` is finite and must be positive. The seed schedule is `seed`,
`seed + 1`, ..., `seed + num-seeds - 1`; each seed run reuses that seed for
every clip in the run. For example, one clip, `--num-envs 64`, and
`--num-seeds 4` plans `1 * 4 = 4` clip rollouts and `1 * 4 * 64 = 256`
trajectories. The manifest records the seeds, planned rollouts, and total
trajectories.

The checkpoint must have its training `params/env.yaml` sibling; without it
the collector refuses to guess the preview. The output directory must not
exist. The collector never resumes and never overwrites an existing run.
Non-finite achieved pose values stop the run before that clip's NPZ is written;
the manifest remains incomplete and records the error.

## Output

```text
<output-dir>/
  manifest.json
  trajectories/
    000_<clip-stem>.npz                 # single-seed baseline only
    run000_seed000_000_<clip-stem>_disturbed.npz
    run001_seed001_000_<clip-stem>_baseline.npz
```

One run has one scenario. The two non-legacy names above illustrate the
multi-seed baseline and disturbed forms; they are not produced by one run.

The legacy `NNN_<clip-stem>.npz` name is kept for the default one-seed
baseline. Any multi-seed run, and every disturbed run, includes run, seed,
clip, and scenario in the file name. `manifest.json` entries include
`run_index`, `seed`, `scenario`, `output`, and `pushes_observed`; use the
manifest, not positional file order, to bind a file to its run.

Each NPZ stores flat rows plus explicit per-trajectory boundaries:

| Key | Shape | Meaning |
| --- | --- | --- |
| `root_quat_wxyz` | `[rows, 4]` | achieved root-link quaternion, world/simulator frame, `wxyz` |
| `joint_pos` | `[rows, 31]` | achieved joint positions in robot joint order, rad |
| `joint_vel` | `[rows, 31]` | achieved joint velocities in robot joint order, rad/s |
| `tick` | `[rows]` | control tick: `0` is after reset and before the first action; `k` is after `k` steps |
| `time_s` | `[rows]` | `tick * control_dt_s` |
| `env_index` | `[rows]` | environment row that produced the state |
| `trajectory_offsets` | `[trajectories + 1]` | row slice of one environment row's first episode |
| `trajectory_env_index` | `[trajectories]` | environment row of that episode |
| `trajectory_outcome` | `[trajectories]` | `complete`, `failure`, or `budget` |
| `trajectory_steps` | `[trajectories]` | control steps after tick `0` |
| `trajectory_initial_frame` | `[trajectories]` | actual reference frame written by the reset |
| `trajectory_seed` | `[trajectories]` | seed of the seed run |
| `trajectory_run_index` | `[trajectories]` | index in the finite seed schedule |
| `trajectory_scenario` | `[trajectories]` | `baseline` or `disturbed` |
| `trajectory_push_count` | `[trajectories]` | interval pushes observed during the first episode |
| `push_offsets` | `[trajectories + 1]` | push slice for one trajectory |
| `push_tick` | `[pushes]` | event tick, after that tick's physics integration |
| `push_env_index` | `[pushes]` | environment row of each observed push |

A push recorded at tick `k` changes root velocity after that tick's physics
integration. Its effect on the saved root orientation, q, and dq can first
appear at tick `k + 1`. A terminal-tick or budget-tick push is counted, but no
later sample records its response. Thus, a positive push count alone does not
prove that the recorded trajectory contains a response to a push.

`trajectory_initial_frame` is the reference frame at reset, not elapsed time.
`tick` and `time_s` remain the elapsed control timeline; they are distinct even
when uniform sampling starts at a late frame. A trajectory that starts at or
near the clip endpoint can finish after only a few ticks.

`manifest.json` uses schema version `2`. Compared with the original collector,
the change is additive: new collection fields and per-trajectory arrays supply
seed/run/scenario, initial-frame, and push provenance. Manifest collection
fields include the finite seed schedule, clip and seed-run counts, total
trajectories, scenario, configured phase, and the resolved push settings. The
disturbance document names the interval range and per-axis velocity-delta
ranges. Push magnitudes are not saved; only the configured ranges are
recorded, plus the observed count and tick of each push.

A trajectory is one environment row's **first** episode. Rows can finish on
different ticks. A finished row is reset only to keep the other rows stepping;
its post-reset state is never appended, and a later episode on that row is not
part of the recorded trajectory and cannot add push evidence. A physical
failure keeps its terminal row and is labelled `failure`; a clip endpoint is
`complete`; an explicit `--max-steps` stop is `budget`. No outcome is silently
rewritten as success.

The collector stores no root position, root linear velocity, body state, command
reference value, policy observation, or action. The `31` joint width above is the
X2 contract; the manifest's `joint_names` is authoritative for a different robot.

## Tests

```sh
PYTHONPATH="$PWD/src" uv run --no-sync pytest -x -vv \
  -o faulthandler_timeout=30 tests/test_joint_pose_collection.py
```

The tests use fake environments. They check achieved-versus-reference state,
initial/terminal row ownership, asynchronous endings, reset contamination,
frozen no-grad inference, NPZ fields and joint order, saved-preview wiring,
seed budgets, phase selection, push timing and inactive-row exclusion, and
output/budget refusal. They do not run the simulator or validate teacher
quality, push magnitudes, or collection outcomes in a real environment.
