#!/usr/bin/env python3
"""One-time preparation of the latent-free X2 joint-pose dataset (S1 steps 1-7).

Reads the published eligibility index, and builds:

  1. store/      dense trajectory-ordered rows (root quat, q, dq)
  2. split       whole-motion train/validation/test assignment
  3. index       window index over the store
  4. spec        preprocessing contract (heading localization, 6D packing)
  5. stats       train-only per-channel mean/std
  6. cache       [windows, 41, 68] float32 memmap per split

No step overwrites. Every step writes stage/<name>.json so a partial run is
visible. This is a stopgap until the `mjlab.scripts.joint_space` CLI exists.

Representation
--------------
channels 0:6   root orientation 6D, heading-local to the window center row
channels 6:37  joint positions q, radians
channels 37:68 joint velocities dq, radians/second

Heading localization rotates every row of a window by Rz(-yaw(center)).
The center row is index 8 of the window: 8 history rows, the current row,
32 future rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

WINDOW = 41
PAST = 8
FUTURE = 32
CHANNELS = 68
JOINTS = 31
CACHE_FORMAT = "mjlab-x2-joint-pose-window-cache-v1"
SPLITS = ("train", "validation", "test")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def emit(value: dict) -> None:
    print(json.dumps(value), flush=True)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    """Return wxyz quaternions for a rotation about +z by ``yaw``."""
    half = yaw * 0.5
    quat = np.zeros(yaw.shape + (4,), dtype=np.float64)
    quat[..., 0] = np.cos(half)
    quat[..., 3] = np.sin(half)
    return quat


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for broadcastable wxyz quaternions."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast(a[..., 0], b[..., 0]).shape + (4,), dtype=np.float64)
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def yaw_of(quat: np.ndarray) -> np.ndarray:
    """Return the yaw angle of a wxyz quaternion."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_6d(quat: np.ndarray) -> np.ndarray:
    """Return the first two rotation-matrix columns as six values."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1.0 - 2.0 * (yy + zz)
    m10 = 2.0 * (xy + wz)
    m20 = 2.0 * (xz - wy)
    m01 = 2.0 * (xy - wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m21 = 2.0 * (yz + wx)
    return np.stack((m00, m10, m20, m01, m11, m21), axis=-1)


def segment_windows(
    quat: np.ndarray, q: np.ndarray, dq: np.ndarray, starts: np.ndarray
) -> np.ndarray:
    """Build ``[n, 41, 68]`` heading-local windows from one segment.

    ``starts`` are store-relative row starts inside the segment. Every row of a
    window is rotated by ``Rz(-yaw)`` of that window's center row.
    """
    # sliding_window_view puts the window axis last: (windows, channels, 41).
    view_quat = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(quat, WINDOW, axis=0), -1, 1
    )
    view_q = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(q, WINDOW, axis=0), -1, 1
    )
    view_dq = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(dq, WINDOW, axis=0), -1, 1
    )
    windows_quat = view_quat[starts]  # [n, 41, 4]
    center = quat[starts + PAST]  # [n, 4]
    rotate = quat_from_yaw(-yaw_of(center))  # [n, 4]
    local = quat_mul(rotate[:, None, :], windows_quat)  # [n, 41, 4]
    out = np.empty((starts.shape[0], WINDOW, CHANNELS), dtype=np.float32)
    out[..., 0:6] = quat_to_6d(local)
    out[..., 6 : 6 + JOINTS] = view_q[starts]
    out[..., 6 + JOINTS : CHANNELS] = view_dq[starts]
    return out


# --------------------------------------------------------------------------- #
# index loading
# --------------------------------------------------------------------------- #
def load_index(path: Path) -> dict:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("kind") != "x2_joint_pose_eligibility":
        raise SystemExit(f"unexpected index kind: {payload.get('kind')!r}")
    payload["_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def accepted_segments(
    index: dict, max_motions: int | None, max_segments_per_motion: int | None
) -> list[dict]:
    """Return one entry per accepted trajectory, in a deterministic order."""
    motions = sorted(index["by_motion"], key=lambda item: item["motion"])
    if max_motions is not None:
        motions = motions[:max_motions]
    wanted = {item["motion"] for item in motions}
    segments: list[dict] = []
    per_motion: dict[str, int] = {}
    for record in sorted(index["files"], key=lambda item: item["path"]):
        motion = record["motion"]
        if motion not in wanted:
            continue
        used = per_motion.get(motion, 0)
        if max_segments_per_motion is not None and used >= max_segments_per_motion:
            continue
        for trajectory in record["trajectory_records"]:
            if trajectory["status"] != "accepted":
                continue
            if max_segments_per_motion is not None and used >= max_segments_per_motion:
                break
            segments.append(
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "motion": motion,
                    "group": record["group"],
                    "seed": record["seed"],
                    "run_index": record["run_index"],
                    "env_index": trajectory["env_index"],
                    "trajectory_index": trajectory["trajectory_index"],
                    "row_start": trajectory["row_start"],
                    "row_stop": trajectory["row_stop"],
                    "windows": trajectory["retained_windows"],
                }
            )
            used += 1
        per_motion[motion] = used
    if not segments:
        raise SystemExit("no accepted trajectories selected")
    return segments


# --------------------------------------------------------------------------- #
# step 1: store
# --------------------------------------------------------------------------- #
def _group_segments_by_file(segments: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group segments by source NPZ, preserving the deterministic order."""
    grouped: dict[str, list[dict]] = {}
    for seg in segments:
        grouped.setdefault(seg["path"], []).append(seg)
    return list(grouped.items())


def step_store(index: dict, segments: list[dict], out: Path, args: argparse.Namespace) -> dict:
    store = out / "store"
    store.mkdir(parents=True, exist_ok=False)
    total_rows = sum(seg["row_stop"] - seg["row_start"] for seg in segments)
    quat = np.lib.format.open_memmap(
        store / "root_quat_wxyz.npy", mode="w+", dtype=np.float32, shape=(total_rows, 4)
    )
    pos = np.lib.format.open_memmap(
        store / "joint_pos.npy", mode="w+", dtype=np.float32, shape=(total_rows, JOINTS)
    )
    vel = np.lib.format.open_memmap(
        store / "joint_vel.npy", mode="w+", dtype=np.float32, shape=(total_rows, JOINTS)
    )
    cursor = 0
    table: list[dict] = []
    position = 0
    # Group by source file: one NPZ load per file, not per trajectory.  A file
    # holds 128 trajectories, so a per-segment load would decompress it 128 times.
    for path, group in _group_segments_by_file(segments):
        with np.load(path, allow_pickle=False) as data:
            quat_file, pos_file, vel_file = (
                data["root_quat_wxyz"],
                data["joint_pos"],
                data["joint_vel"],
            )
            for seg in group:
                lo, hi = seg["row_start"], seg["row_stop"]
                rows = hi - lo
                quat[cursor : cursor + rows] = quat_file[lo:hi]
                pos[cursor : cursor + rows] = pos_file[lo:hi]
                vel[cursor : cursor + rows] = vel_file[lo:hi]
                table.append(
                    {
                        "segment": position,
                        "motion": seg["motion"],
                        "group": seg["group"],
                        "seed": seg["seed"],
                        "env_index": seg["env_index"],
                        "trajectory_index": seg["trajectory_index"],
                        "row_start": cursor,
                        "rows": rows,
                        "windows": seg["windows"],
                        "source": path,
                        "source_sha256": seg["sha256"],
                    }
                )
                cursor += rows
                position += 1
                if position % 5000 == 0:
                    emit({"store": {"segment": position, "of": len(segments)}})
    for array in (quat, pos, vel):
        array.flush()
    del quat, pos, vel
    manifest = {
        "format": "mjlab-x2-joint-pose-store-v1",
        "created_utc": now(),
        "rows": total_rows,
        "segments": len(table),
        "arrays": {
            "root_quat_wxyz": ["float32", [total_rows, 4], "achieved world-frame wxyz"],
            "joint_pos": ["float32", [total_rows, JOINTS], "radians"],
            "joint_vel": ["float32", [total_rows, JOINTS], "radians/second"],
        },
        "source_index_sha256": index["_sha256"],
        "table": table,
    }
    (store / "manifest.json").write_text(json.dumps(manifest))
    return {"rows": total_rows, "segments": len(table), "bytes": total_rows * 4 * (4 + 62)}


# --------------------------------------------------------------------------- #
# step 2: split
# --------------------------------------------------------------------------- #
def step_split(index: dict, segments: list[dict], out: Path, args: argparse.Namespace) -> dict:
    motions = sorted({seg["motion"] for seg in segments})
    stride = max(4, 2 * len(motions) // max(1, args.val_motions + args.test_motions))
    assignment = {}
    for position, motion in enumerate(motions):
        if position % stride == 0:
            assignment[motion] = "test"
        elif position % stride == stride // 2:
            assignment[motion] = "validation"
        else:
            assignment[motion] = "train"
    counts = {name: 0 for name in SPLITS}
    for motion in motions:
        counts[assignment[motion]] += 1
    if counts["train"] == 0:
        raise SystemExit(f"degenerate split: {counts}")
    if min(counts.values()) == 0:
        emit({"split_warning": f"empty split in a bounded run: {counts}"})
    payload = {
        "format": "mjlab-x2-joint-pose-split-v1",
        "created_utc": now(),
        "policy": "whole-motion, deterministic stride over sorted motion names",
        "motions": assignment,
        "counts": counts,
    }
    (out / "assignments.json").write_text(json.dumps(payload, indent=1))
    return {"counts": counts}


# --------------------------------------------------------------------------- #
# steps 3-6
# --------------------------------------------------------------------------- #
def step_index(segments: list[dict], table: list[dict], out: Path, args: argparse.Namespace) -> dict:
    split_of = json.loads((out / "assignments.json").read_text())["motions"]
    motions = sorted(split_of)
    motion_index = {name: position for position, name in enumerate(motions)}
    split_index = {name: position for position, name in enumerate(SPLITS)}
    rows = []
    for entry in table:
        rows.append(
            (
                entry["row_start"],
                entry["rows"],
                entry["windows"],
                split_index[split_of[entry["motion"]]],
                motion_index[entry["motion"]],
            )
        )
    array = np.array(rows, dtype=np.int64)
    counts = np.zeros(3, dtype=np.int64)
    for split in range(3):
        counts[split] = int(array[array[:, 3] == split, 2].sum())
    np.savez_compressed(
        out / "window_index.npz",
        seg_row_start=array[:, 0],
        seg_rows=array[:, 1],
        seg_windows=array[:, 2],
        seg_split=array[:, 3].astype(np.int8),
        seg_motion=array[:, 4].astype(np.int32),
        motions=np.array(motions, dtype="U128"),
        split_windows=counts,
    )
    return {
        "segments": int(array.shape[0]),
        "windows": int(counts.sum()),
        "by_split": {name: int(counts[position]) for name, position in split_index.items()},
    }


def step_spec(out: Path, args: argparse.Namespace) -> dict:
    payload = {
        "format": "mjlab-x2-joint-pose-window-spec-v1",
        "created_utc": now(),
        "window": {"rows": WINDOW, "past": PAST, "current": 1, "future": FUTURE},
        "channels": {
            "0:6": "root orientation 6D, heading-local to the window center row",
            "6:37": "joint positions q, radians",
            "37:68": "joint velocities dq, radians/second",
        },
        "heading": {
            "mode": "per_window_center",
            "rotation": "Rz(-yaw(center_row)) applied to every row of the window",
            "center_row": PAST,
        },
        "packing": {
            "quaternion_order": "wxyz",
            "rot6d": "first two rotation-matrix columns: R00,R10,R20,R01,R11,R21",
        },
        "normalization": {
            "mode": "per_channel_zscore",
            "fit_on": "train split only",
            "rot6d_note": "normalized channels are not exactly orthonormal; re-orthonormalize at inference",
        },
        "windows_per_segment": "max(rows - 41 + 1, 0)",
    }
    (out / "preprocess_spec.json").write_text(json.dumps(payload, indent=1))
    return payload["window"]


def iter_split_segments(out: Path, split: str):
    index = np.load(out / "window_index.npz")
    position = SPLITS.index(split)
    mask = index["seg_split"] == position
    starts = index["seg_row_start"][mask]
    rows = index["seg_rows"][mask]
    windows = index["seg_windows"][mask]
    for start, count, count_windows in zip(starts, rows, windows, strict=True):
        if count_windows <= 0:
            continue
        yield int(start), int(count), int(count_windows)


_STORE_CACHE: dict[str, np.ndarray] = {}


def _store_array(out: Path, name: str) -> np.ndarray:
    """Open one store array once and reuse the read-only memmap."""
    key = f"{out}:{name}"
    array = _STORE_CACHE.get(key)
    if array is None:
        array = np.load(out / "store" / f"{name}.npy", mmap_mode="r")
        _STORE_CACHE[key] = array
    return array


def read_store(out: Path, start: int, count: int):
    sl = slice(start, start + count)
    quat = np.asarray(_store_array(out, "root_quat_wxyz")[sl], np.float64)
    pos = np.asarray(_store_array(out, "joint_pos")[sl], np.float64)
    vel = np.asarray(_store_array(out, "joint_vel")[sl], np.float64)
    return quat, pos, vel


def step_stats(out: Path, args: argparse.Namespace) -> dict:
    total = np.zeros(CHANNELS, dtype=np.float64)
    total_sq = np.zeros(CHANNELS, dtype=np.float64)
    samples = 0
    seen = 0
    for start, count, _ in iter_split_segments(out, "train"):
        if args.max_stats_segments is not None and seen >= args.max_stats_segments:
            break
        seen += 1
        quat, pos, vel = read_store(out, start, count)
        starts = np.arange(0, count - WINDOW + 1, max(1, args.stats_stride), dtype=np.int64)
        block = segment_windows(quat, pos, vel, starts).astype(np.float64)
        total += block.sum(axis=(0, 1))
        total_sq += (block * block).sum(axis=(0, 1))
        samples += block.shape[0] * WINDOW
        if seen % 5000 == 0:
            emit({"stats": {"segments": seen, "samples": samples}})
    if samples == 0:
        raise SystemExit("statistics pass saw no windows")
    mean = total / samples
    variance = np.maximum(total_sq / samples - mean * mean, 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-8, 1.0, std)
    np.savez(
        out / "projection.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        channels=np.arange(CHANNELS, dtype=np.int32),
    )
    payload = {
        "format": "mjlab-x2-joint-pose-stats-v1",
        "created_utc": now(),
        "split": "train",
        "segments": seen,
        "samples": int(samples),
        "stats_stride": args.stats_stride,
        "mean_abs_max": float(np.abs(mean).max()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }
    (out / "stats.json").write_text(json.dumps(payload, indent=1))
    return payload


def step_cache(out: Path, args: argparse.Namespace) -> dict:
    stats = np.load(out / "projection.npz")
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    index = np.load(out / "window_index.npz")
    summary = {}
    for split in SPLITS:
        count = int(index["split_windows"][SPLITS.index(split)])
        window_windows = index["seg_windows"][index["seg_split"] == SPLITS.index(split)]
        if count == 0:
            continue
        path = out / "offline" / f"tokens-{split}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        progress_path = out / "stage" / f"cache-{split}.json"
        if path.is_file() and progress_path.is_file():
            # Resume a killed build instead of rewriting completed windows.
            progress = json.loads(progress_path.read_text())
            if progress.get("shape") != [count, WINDOW, CHANNELS]:
                raise SystemExit(f"cache progress shape mismatch for {split}")
            cache = np.lib.format.open_memmap(path, mode="r+")
            start_segment = int(progress["segments"])
            # A valid resume point is always a segment boundary.  Recompute the
            # cursor from the index so a stale or hand-edited progress file
            # cannot shift every later window.
            expected = int(window_windows[:start_segment].sum())
            if int(progress["cursor"]) != expected:
                emit({"cache_resume_warning": {
                    "split": split, "stored": int(progress["cursor"]), "recomputed": expected}})
            cursor = expected
            emit({"cache_resume": {"split": split, "segments": start_segment, "windows": cursor}})
        else:
            cache = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(count, WINDOW, CHANNELS)
            )
            start_segment = 0
            cursor = 0
        written = 0
        skipped = 0
        for start, rows, windows in iter_split_segments(out, split):
            if skipped < start_segment:
                skipped += 1
                continue
            quat, pos, vel = read_store(out, start, rows)
            block = segment_windows(quat, pos, vel, np.arange(windows, dtype=np.int64))
            block = (block - mean) / std
            cache[cursor : cursor + windows] = block.astype(np.float32)
            cursor += windows
            written += 1
            if written % 2000 == 0:
                progress_path.write_text(
                    json.dumps(
                        {
                            "split": split,
                            "shape": [count, WINDOW, CHANNELS],
                            "segments": start_segment + written,
                            "cursor": cursor,
                        }
                    )
                )
                emit({"cache": {"split": split, "segments": start_segment + written,
                                "windows": cursor, "of": count}})
        cache.flush()
        del cache
        progress_path.write_text(
            json.dumps(
                {
                    "split": split,
                    "shape": [count, WINDOW, CHANNELS],
                    "segments": int((index["seg_split"] == SPLITS.index(split)).sum()),
                    "cursor": count,
                }
            )
        )
        metadata = {
            "format": CACHE_FORMAT,
            "split": split,
            "shape": [count, WINDOW, CHANNELS],
            "dtype": "float32",
            "bytes": count * WINDOW * CHANNELS * 4,
            "segments": int((index["seg_split"] == SPLITS.index(split)).sum()),
            "created_utc": now(),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=1))
        summary[split] = metadata
    return summary


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def rot6d_yaw(six: np.ndarray) -> float:
    """Return the heading angle encoded by a 6D rotation's first column."""
    return float(np.arctan2(six[1], six[0]))


def validate(out: Path) -> dict:
    index = np.load(out / "window_index.npz")
    stats = np.load(out / "projection.npz")
    mean, std = stats["mean"].astype(np.float64), stats["std"].astype(np.float64)
    checks: dict[str, bool] = {}
    checks["index_windows_match"] = int(index["split_windows"].sum()) == int(index["seg_windows"].sum())
    checks["stats_finite_positive"] = bool(np.isfinite(mean).all() and (std > 0).all())
    for split in SPLITS:
        path = out / "offline" / f"tokens-{split}.npy"
        if not path.is_file():
            continue
        cache = np.load(path, mmap_mode="r")
        head = np.asarray(cache[: min(256, cache.shape[0])], dtype=np.float64) * std + mean
        checks[f"{split}_finite"] = bool(np.isfinite(head).all())
        c0, c1 = head[:, PAST, 0], head[:, PAST, 1]
        checks[f"{split}_center_heading_zero"] = bool(np.all(np.abs(np.arange(head.shape[0]) * 0.0 + np.arctan2(c1, c0)) < 1e-3))
        col = head[:, PAST, 0:3]
        checks[f"{split}_center_column_unit"] = bool(np.all(np.abs(np.linalg.norm(col, axis=1) - 1.0) < 2e-2))
        checks[f"{split}_shape"] = list(cache.shape[1:]) == [WINDOW, CHANNELS]
    checks["all"] = all(checks.values())
    return checks


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-motions", type=int, default=12)
    parser.add_argument("--test-motions", type=int, default=12)
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--max-segments-per-motion", type=int, default=None)
    parser.add_argument("--max-stats-segments", type=int, default=None)
    parser.add_argument("--stats-stride", type=int, default=1)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run instead of refusing the output directory",
    )
    args = parser.parse_args(argv)

    out = args.output_dir.expanduser().resolve()
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    stages = out / "stage"
    stages.mkdir(exist_ok=True)

    index = load_index(args.index.expanduser().resolve())
    segments = accepted_segments(index, args.max_motions, args.max_segments_per_motion)
    emit({"selected": {"motions": len({seg['motion'] for seg in segments}), "segments": len(segments)}})

    started = time.monotonic()

    def done(*names: str) -> bool:
        return all((out / name).exists() for name in names)

    if done("store/manifest.json"):
        store = json.loads((stages / "store.json").read_text())
        emit({"stage": "store", "skipped": True, **store})
    else:
        store = step_store(index, segments, out, args)
        (stages / "store.json").write_text(json.dumps(store))
        emit({"stage": "store", **store, "elapsed_s": round(time.monotonic() - started, 2)})

    if done("assignments.json"):
        split = json.loads((out / "assignments.json").read_text())
        emit({"stage": "split", "skipped": True, "counts": split["counts"]})
    else:
        split = step_split(index, segments, out, args)
        (stages / "split.json").write_text(json.dumps(split))
        emit({"stage": "split", **split})

    if done("window_index.npz"):
        emit({"stage": "index", "skipped": True})
    else:
        table = json.loads((out / "store" / "manifest.json").read_text())["table"]
        window_index = step_index(segments, table, out, args)
        (stages / "index.json").write_text(json.dumps(window_index))
        emit({"stage": "index", **window_index})

    if done("preprocess_spec.json"):
        emit({"stage": "spec", "skipped": True})
    else:
        step_spec(out, args)
        emit({"stage": "spec", "ok": True})

    if done("projection.npz", "stats.json"):
        stats = json.loads((out / "stats.json").read_text())
        emit({"stage": "stats", "skipped": True, "samples": stats["samples"]})
    else:
        stats = step_stats(out, args)
        (stages / "stats.json").write_text(json.dumps(stats))
        emit({"stage": "stats", "segments": stats["segments"], "samples": stats["samples"]})

    if not args.skip_cache:
        summary = step_cache(out, args)
        (stages / "cache.json").write_text(json.dumps(summary))
        emit({"stage": "cache", "splits": {k: v["shape"] for k, v in summary.items()}})

    checks = validate(out)
    (out / "validation.json").write_text(json.dumps(checks, indent=1))
    emit({"validation": checks, "elapsed_s": round(time.monotonic() - started, 2)})
    if not checks["all"]:
        emit({"error": "validation failed"})
        return 1
    print(json.dumps({"output_dir": str(out), "index_sha256": index["_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
