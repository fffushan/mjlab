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

## Eligibility gate

The gate at `scripts/experiments/joint_pose_collection/gate.py` reads complete
schema-version 2 collection manifests and their NPZ files. It writes one compact
eligibility index and never edits, deletes, or rewrites raw data. It does not
materialize windows, normalize values, fit statistics, or load a model. It is
not wired into any trainer.

```sh
PYTHONPATH="$PWD/src" uv run --no-sync python \
  scripts/experiments/joint_pose_collection/gate.py \
  --manifest logs/joint_pose_collection/start/manifest.json \
             logs/joint_pose_collection/uniform_baseline/manifest.json \
             logs/joint_pose_collection/uniform_disturbed/manifest.json \
  --output-dir logs/joint_pose_eligibility/batch12
```

Pass all manifests after one `--manifest` flag. Each explicit manifest is a
separate input, and results are grouped by `sampling_mode/scenario`, so
`start/baseline` and `uniform/baseline` stay distinct even when motion names,
seeds, and scenario labels overlap. The output directory must be new and must
be outside every input collection directory. An existing directory is refused;
the gate does not resume or overwrite.

### Policy and window contract

A trajectory is eligible only when all of these hold:

- its stored `trajectory_outcome` is `complete`;
- `root_quat_wxyz`, `joint_pos`, and `joint_vel` are finite throughout its
  recorded row range; and
- it has at least `41` raw rows.

The window is fixed: `8` history rows, the current row, and `32` future rows,
with no padding and no cross-trajectory rows. For `S` actions there are `S + 1`
rows and `max(S - 39, 0)` candidate windows. A `41`-row trajectory has one
window; a `42`-row trajectory has two. The report stores the half-open range of
window-start rows: each start `s` means raw rows `[s, s + 41)`.

The gate rejects the whole trajectory. It does not salvage a prefix of a
failed or budget-stopped trajectory, and it does not keep windows around a
corrupt sample. Rejection reasons are disjoint and use this precedence:

1. `outcome_non_complete`: stored outcome is `failure` or `budget`.
2. `nonfinite_state`: a required raw state value is non-finite.
3. `too_short`: a complete trajectory has fewer than `41` rows.

A complete trajectory shorter than one window is still a valid raw record, but
it has no eligible training window and is reported as `too_short`. Push
exposure is not a gate requirement: complete disturbed trajectories are kept
with pushes, without pushes, or with a zero push count. Stored outcome labels
are used as recorded; the gate does not infer success from clip-end length and
adds no height, tilt, contact, tracking-error, quaternion-normalization, or
push-severity threshold. A `failure` label means a task termination fired. It
is not proof of ground contact or an exact fall detector. A `complete` label is
not proof of perfect tracking.

### Output

The gate writes `<output-dir>/eligibility.json`. It is a compact JSON object
with `schema_version: 1` and `kind: x2_joint_pose_eligibility`. It contains:

- `policy` and `window_contract`, including the fixed dimensions and rejection
  precedence;
- `inputs`, with each manifest path, SHA256, schema, task, joint names, control
  timing, seeds, and group;
- `files`, with each NPZ path, SHA256, motion provenance, run/seed/scenario/
  sampling identity, and per-trajectory records;
- each trajectory's row range, outcome, accepted/rejected status, primary
  reason, candidate/retained/rejected window counts, and retained window-start
  range;
- `totals`, `by_group`, `by_motion`, `by_reason`, and `count_identities`.

The hashes cover the exact bytes that were parsed and validated. Each NPZ array
is read once per file, so memory is bounded by one NPZ plus the index. The
count identities are exact: candidate windows equal retained plus rejected
windows, and all trajectories equal accepted plus rejected trajectories.

A future consumer, including a preprocessing or training stage that does not
yet exist, must read this index and honor its accepted status and retained
window ranges. The gate does not prove training readiness, sample independence,
or recovery quality. It also does not reopen the source motion clip; the
manifest's motion path is copied as provenance, and the raw NPZ state is what
is validated.

The gate refuses an incomplete manifest, an unsupported schema version, an
absent NPZ, duplicate manifest or NPZ sources, inconsistent joint order or
control timing, malformed offsets/ticks, NPZ provenance or outcome-count
mismatches, an unknown outcome label, a clip output path that escapes its
collection directory, and an output directory inside an input collection.

## Tests

```sh
PYTHONPATH="$PWD/src" uv run --no-sync pytest -x -vv \
  -o faulthandler_timeout=30 tests/test_joint_pose_collection.py \
  tests/test_joint_pose_collection_gate.py
```

The collector tests use fake environments. They check achieved-versus-reference
state, initial/terminal row ownership, asynchronous endings, reset
contamination, frozen no-grad inference, NPZ fields and joint order,
saved-preview wiring, seed budgets, phase selection, push timing and
inactive-row exclusion, and output/budget refusal. They do not run the
simulator or validate teacher quality, push magnitudes, or collection outcomes
in a real environment.

The gate tests use small synthetic schema2 collections. They check the public
CLI, the `40`/`41`/`42`-row boundaries, whole-trajectory failure and budget
exclusion, non-finite root/quaternion, joint-position, and joint-velocity
rejection, push-independent disturbed retention, trajectory-boundary ranges,
multiple group identities, hashes and count round-trips, malformed input
refusal, unchanged input bytes, and the absence of simulator imports or window
materialization. They do not validate a real batch or training quality.
