# X2 raw joint-pose collection

Opt-in research collector for the latent-free X2 pipeline. It rolls out the
frozen multi-motion PPO teacher and writes the **achieved simulator state**:
root-link orientation, joint position, and joint velocity. It does not train a
model, unpack a 6D rotation, align heading, build windows, load the VAE, or run
any reference adapter. Those steps belong to later preprocessing.

The collector reuses the evaluator's playback setup:

- each clip is played from its first frame with `sampling_mode="start"`;
- `auto_reset=False`, so terminal state is read before a row is reset;
- the checkpoint's saved `lookahead_s` is required and restored (the teacher
  uses `0.2 s`), so the preview is not silently changed;
- actor observation corruption is on and `push_robot` is removed, matching the
  existing evaluation path;
- the policy is loaded once, kept frozen, and evaluated under `torch.no_grad`.

## Usage

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/joint_pose_collection/collect.py \
  --checkpoint-file logs/rsl_rl/agibot_x2_multi_motion_tracking/<run>/model_19999.pt \
  --motion-file data/tennis/single_000_zhanghongyu_agibot_x2_tracking.npz \
  --output-dir logs/joint_pose_collection/run_000 \
  --num-envs 64 --seed 0
```

Pass more clips after one `--motion-file` flag. Relative clip paths resolve
from the repository root, the same convention as the multi-motion command.
`--max-steps` caps each trajectory below the clip length; omitted, the full
clip is used. Each clip is one batched environment and one file, so a
multi-clip run keeps only the current clip's rollout in memory.

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
    000_<clip-stem>.npz
    001_<clip-stem>.npz
```

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

`manifest.json` records schema version `1`, checkpoint and clip paths, seed,
number of environments, effective step budget, control `dt`, joint names, units,
quaternion order and frame, sample timing, and per-clip outcome counts. It is
rewritten after each finished clip, with `"complete": false` until the run ends.

A trajectory is one environment row's **first** episode. Rows can finish on
different ticks. A finished row is reset only to keep the other rows stepping;
its post-reset state is never appended, and a later episode on that row is not
part of the recorded trajectory. A physical failure keeps its terminal row and is
labelled `failure`; a clip endpoint is `complete`; an explicit `--max-steps` stop
is `budget`. No outcome is silently rewritten as success.

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
frozen no-grad inference, NPZ fields and joint order, saved-preview wiring, and
output/budget refusal. They do not run the simulator or validate teacher quality.
