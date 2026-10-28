# T40 tennis juggling ablation: resume handover

Updated: 2026-10-28 11:25 (+08). Written for an agent that has no history of this workstream.

## The job

Five PPO arms of the X2 tennis **juggling** task, 30 000 iterations each, one GPU per arm,
8 192 environments per arm, checkpoints every 500 iterations.

Task id: `Mjlab-Velocity-Flat-AgiBot-X2-No-State-Estimation-Tennis-Ball-Juggle`

The arms differ only in one reward weight, so the comparison isolates that weight. The arm
definitions live in `/home/fushan/mjlab-tennis40/launch_arm.sh`; do not re-derive them by hand.

| arm | override | GPU | tmux | log | purpose |
|---|---|---|---|---|---|
| A0 | none | 4 | `t40-A0` | `~/logs/t40-A0-resume.log` | baseline, as committed |
| A1 | `MJLAB_JUGGLE_ENERGY_WEIGHT=4.0` | 5 | `t40-A1` | `~/logs/t40-A1-resume.log` | energy weight doubled |
| A2 | `MJLAB_JUGGLE_TRACK_WEIGHT=1.0` | 6 | `t40-A2` | `~/logs/t40-A2-resume.log` | ball tracking weight raised |
| A3 | `MJLAB_JUGGLE_LIFT_WEIGHT=0.0` | 7 | `t40-A3` | `~/logs/t40-A3-resume.log` | lift term removed |
| A4 | `MJLAB_JUGGLE_TRACK_WEIGHT=0.0` | 1 | `t40-A4` | `~/logs/t40-A4-resume.log` | ball tracking removed |

A3 is the arm whose iteration-10 500 checkpoint was exported and deployed as
`x2_velocity_tennis.onnx`. Expect A3 to be the reference for anything already measured.

## Current state (verified 2026-10-28 11:24 +08)

All five arms are **running**, resumed from `model_9500.pt`, each reporting `Learning iteration
9500/30000`, each at 19 531 MiB on its GPU, GPU 4/5/6/7/1 busy.

`20500` more iterations at the measured 4.7 s per iteration is about **27 h**, so completion near
**2026-10-29 14:00 (+08)**. Checkpoints appear in the new run directory every 500 iterations.

Run directories, resumed 2026-10-28:

```
/home/fushan/runs/tennis40/<arm>/agibot_x2_velocity_tennis_ball_juggle/2026-10-28_<hh-mm-ss>/
```

The pre-shutdown directories `2026-10-26_22-49-*` (A4: `2026-10-26_23-04-56`) hold iterations
0–9 500 and its `model_9500.pt`. **Do not delete or move them**; they are the resume source.

Iteration 9 500 metrics, as a sanity reference for a resumed run: A3 `ball_energy` 0.5004,
`racket_lift` 0.0, episode length 402.7, reward 15.71.

## Access and layout

```bash
ssh -p 31505 fushan@10.14.64.37        # managed container, 8x RTX 4090
```

| what | path |
|---|---|
| code (frozen copy used by these runs) | `/home/fushan/mjlab-tennis40/src` |
| launcher | `/home/fushan/mjlab-tennis40/launch_arm.sh` |
| trainer wrapper | `/home/fushan/mjlab-tennis40/run_task_t40.sh` |
| venv (symlinked into the tree as `.venv`) | `/home/fushan/mjlab/.venv` |
| runs | `/home/fushan/runs/tennis40/<arm>/` |
| logs | `/home/fushan/logs/` |

`PYTHONPATH=/home/fushan/mjlab-tennis40/src` is what makes the venv import the frozen code rather
than an installed copy. The launcher sets it; do not drop it.

## Resume one arm

The launcher forwards extra arguments, and `ITERS` overrides the iteration count:

```bash
tmux new-session -d -s t40-A3 \
  "ITERS=20500 bash /home/fushan/mjlab-tennis40/launch_arm.sh A3 7 \
     --agent.resume True --agent.load-checkpoint model_9500.pt 2>&1 \
     | tee -a /home/fushan/logs/t40-A3-resume2.log"
```

The trainer chooses the source run itself: it takes the alphabetically last run directory under
`<arm>/agibot_x2_velocity_tennis_ball_juggle/` and the newest checkpoint matching
`--agent.load-checkpoint`. Pass that checkpoint explicitly, as above.

## Traps, each one measured today

1. **`--agent.max-iterations` is the number of *additional* iterations.** A resume from 9 500 with
   `30000` prints `9500/39500` and would train 30 000 more. The correct value is `20500`, which
   prints `9500/30000`. This is the trap most likely to silently double a run.
2. **A resume writes a new timestamped directory.** The old one remains the source. Two directories
   per arm are normal, not an error.
3. **`launch_arm.sh` must forward `${@:3}`, not `$@`.** With `$@` the arm name and GPU number are
   passed as the motion file and the run dies with `[error] motion file not found: A3`.
4. **`sed -i` fails silently in the root-owned tree.** `/home/fushan/mjlab-tennis40` is `root:root`,
   so `sed -i` cannot create its temporary file and exits without changing anything; use `sudo sed`.
   A failed patch looks exactly like a working one until the arm restarts from iteration 0.
5. **Default checkpoint selection is alphabetical.** `model_10000.pt` sorts *before*
   `model_9500.pt`, so the default regex would pick the older file once 10 000 exists. Always pass
   `--agent.load-checkpoint` explicitly.
6. **Do not `tee -a` onto an old log** while diagnosing: the stale iteration lines from the previous
   attempt appear first and look like the current run.

## Damage from the IT shutdown, already repaired

The maintenance pass left the container in a state where nothing could start. All of the following
was fixed; re-check rather than repeat.

| symptom | cause | fix applied |
|---|---|---|
| `train: No such file or directory` | `mjlab-tennis40/.venv` missing | `sudo ln -s /home/fushan/mjlab/.venv /home/fushan/mjlab-tennis40/.venv` |
| venv unusable | `.venv/bin/python*` symlinks deleted, shebangs broken | recreated `python`, `python3`, `python3.13` → `~/.local/share/uv/python/cpython-3.13.15-.../bin/python3.13` |
| `PermissionError` on run dirs | `/home/fushan`, `.config`, `.logs`, `.cache` and `runs/tennis40/A0` owned by root | `sudo chown` on the four paths, `sudo chown -R fushan:fushan /home/fushan/runs/tennis40` |
| launcher ignored extra args | `run_task_t40.sh ... 30000` with no pass-through | last line now `... "${ITERS:-30000}" "${@:3}"` |

Environment verified after repair: Python 3.13.15, torch 2.9.0+cu128 with CUDA available, mujoco
3.11.0, warp 1.14.0, `mjlab` importing from `/home/fushan/mjlab-tennis40/src`.

If more paths turn up root-owned, `sudo chown fushan:fushan <path>` is the fix. Do **not** run
`chown -R` on all of `/home/fushan`: it is a bind mount onto the 1.5 PB NAS.

## Verification

```bash
tmux ls                                            # expect t40-A0 .. t40-A4
pgrep -af "bin/train Mjlab"                        # one process per arm, args include --agent.resume
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for a in A0 A1 A2 A3 A4; do grep -m1 "Learning iteration" ~/logs/t40-$a-resume.log; done
```

Healthy arm: `Learning iteration <n>/30000` with `n` increasing, 19 531 MiB on its GPU, about
4.7 s per iteration, and `Iteration time` lines rather than a traceback.

## Do not touch

- **GPUs 0, 2 and 3** are not part of these arms. Other sessions use this host.
- A `rsync` of `/home/fushan/data/mjlab-144/...` belongs to another workstream. Leave it.
- Other people's run directories under `/home/fushan/runs/` and `/home/fushan/logs/`.
- The `2026-10-26_*` run directories: they are the resume source for every arm.

## Open questions for the operator

- Whether to extend any arm past 30 000 iterations. Not authorised; all five stop at 30 000.
- Nothing has been published from the ablation yet: no per-arm comparison, no evaluation, no export.
  A3's 13 500 checkpoint exists in the old directory but is not deployed.
