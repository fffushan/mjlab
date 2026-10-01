#!/usr/bin/env python3
"""Evaluate VAE teacher fidelity on recorded X2 policy-state snapshots.

This evaluator compares a frozen teacher action with a frozen VAE student's
recorded/replayed action on the same observation. It is an offline, one-step
fidelity metric; it is not a closed-loop stability or hardware evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import onnxruntime as ort

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_pitch_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
]
WINDOWS = {
  "0.0_0.5": (0, 25),
  "0.5_1.0": (25, 50),
  "1.0_1.5": (50, 75),
  "1.5_2.0": (75, 100),
  "2.0_3.0": (100, 150),
  "3.0_4.0": (150, 200),
  "4.0_5.0": (200, 250),
  "5.0_6.6": (250, 331),
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def rms_stats(values: np.ndarray) -> dict[str, float]:
  absolute = np.abs(np.asarray(values, dtype=np.float64))
  return {
    "rms": float(np.sqrt(np.mean(absolute * absolute))),
    "p50_abs": float(np.percentile(absolute, 50)),
    "p95_abs": float(np.percentile(absolute, 95)),
    "max_abs": float(np.max(absolute)),
  }


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
  w, x, y, z = first
  v, i, j, k = second
  return np.array(
    [
      w * v - x * i - y * j - z * k,
      w * i + x * v + y * k - z * j,
      w * j - x * k + y * v + z * i,
      w * k + x * j - y * i + z * v,
    ]
  )


def model_arrays(
  session: ort.InferenceSession,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
  metadata = session.get_modelmeta().custom_metadata_map
  default = np.asarray(
    [float(value) for value in metadata["default_joint_pos"].split(",")],
    dtype=np.float32,
  )
  scale = np.asarray(
    [float(value) for value in metadata["action_scale"].split(",")],
    dtype=np.float32,
  )
  if metadata["joint_names"].split(",") != JOINT_NAMES:
    raise ValueError("teacher joint_names metadata does not match the X2 contract")
  expected = (
    "command,motion_lookahead,motion_anchor_ori_b,base_ang_vel,"
    "joint_pos,joint_vel,actions"
  )
  if metadata["observation_names"] != expected:
    raise ValueError(
      "teacher observation_names metadata does not match the 164-D contract"
    )
  if default.shape != (31,) or scale.shape != (31,):
    raise ValueError("teacher default/action scale metadata must have 31 elements")
  return default, scale, metadata


def teacher_observation(reference: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
  if reference.shape != (68,) or conditioning.shape != (99,):
    raise ValueError(
      "recorded VAE state has an unexpected reference/conditioning shape"
    )
  # Deployed tracking layout: command(62), anchor error(6), gyro(3),
  # relative q(31), dq(31), previous action(31).
  return np.concatenate(
    (
      reference[:62],
      reference[62:68],
      conditioning[3:6],
      conditioning[6:37],
      conditioning[37:68],
      conditioning[68:99],
    )
  ).astype(np.float32, copy=False)


def load_contract(path: Path) -> tuple[np.ndarray, np.ndarray]:
  contract = json.loads(path.read_text())
  action = contract["action"]
  default = np.asarray(action["default_joint_position"], dtype=np.float32)
  scale = np.asarray(action["scales"], dtype=np.float32)
  if (
    action["joint_names"] != JOINT_NAMES
    or default.shape != (31,)
    or scale.shape != (31,)
  ):
    raise ValueError("student contract does not match the X2 31-joint action contract")
  return default, scale


def load_states(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as archive:
    required = {
      "vae_reference",
      "vae_conditioning",
      "student_raw_action",
      "student_latent",
      "reference_frame",
      "engagement_generation",
      "assembly_wall_ns",
      "run_kind",
    }
    missing = required - set(archive.files)
    if missing:
      raise ValueError(f"recorded state fixture is missing arrays: {sorted(missing)}")
    states = {name: archive[name].copy() for name in required}
  n = states["vae_reference"].shape[0]
  expected_shapes = {
    "vae_reference": (n, 68),
    "vae_conditioning": (n, 99),
    "student_raw_action": (n, 31),
    "student_latent": (n, 32),
    "reference_frame": (n,),
    "engagement_generation": (n,),
    "assembly_wall_ns": (n,),
    "run_kind": (n,),
  }
  for name, shape in expected_shapes.items():
    if states[name].shape != shape:
      raise ValueError(f"{name} has shape {states[name].shape}, expected {shape}")
  if (
    not np.isfinite(states["vae_reference"]).all()
    or not np.isfinite(states["vae_conditioning"]).all()
  ):
    raise ValueError("recorded state fixture contains non-finite inputs")
  return states


def session_options() -> Any:
  ort_any = cast(Any, ort)
  options: Any = ort_any.SessionOptions()
  options.intra_op_num_threads = 1
  options.inter_op_num_threads = 1
  options.graph_optimization_level = ort_any.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
  return options


def run_output(
  session: ort.InferenceSession,
  inputs: dict[str, np.ndarray],
  output_index: int,
) -> np.ndarray:
  outputs: Any = session.run(None, inputs)
  return cast(np.ndarray, outputs[output_index])


def run_group(
  states: dict[str, np.ndarray],
  indexes: np.ndarray,
  teacher: ort.InferenceSession,
  encoder: ort.InferenceSession,
  decoder: ort.InferenceSession,
  teacher_default: np.ndarray,
  teacher_scale: np.ndarray,
  student_default: np.ndarray,
  student_scale: np.ndarray,
) -> dict[str, Any]:
  teacher_actions: list[np.ndarray] = []
  student_actions: list[np.ndarray] = []
  teacher_targets: list[np.ndarray] = []
  student_targets: list[np.ndarray] = []
  target_differences: list[np.ndarray] = []
  raw_differences: list[np.ndarray] = []
  q_errors: list[float] = []
  dq_errors: list[float] = []
  latent_errors: list[float] = []
  student_action_errors: list[float] = []
  for index in indexes:
    reference = states["vae_reference"][index]
    conditioning = states["vae_conditioning"][index]
    teacher_result = [
      run_output(
        teacher,
        {
          "obs": teacher_observation(reference, conditioning)[None],
          "time_step": np.asarray(
            [[states["reference_frame"][index]]], dtype=np.float32
          ),
        },
        output_index,
      )
      for output_index in range(3)
    ]
    teacher_action = np.asarray(teacher_result[0][0], dtype=np.float32)
    student_action = states["student_raw_action"][index]
    student_latent = run_output(encoder, {"reference": reference[None]}, 0)
    replayed_student = run_output(
      decoder,
      {"latent": student_latent, "conditioning": conditioning[None]},
      0,
    )[0]
    # The embedded teacher reference must be the same authored q/dq as the
    # recorded VAE state when the reference frame is passed as time_step.
    q_errors.append(float(np.max(np.abs(teacher_result[1][0] - reference[:31]))))
    dq_errors.append(float(np.max(np.abs(teacher_result[2][0] - reference[31:62]))))
    latent_errors.append(
      float(np.max(np.abs(student_latent[0] - states["student_latent"][index])))
    )
    student_action_errors.append(
      float(np.max(np.abs(replayed_student - student_action)))
    )
    teacher_actions.append(teacher_action)
    student_actions.append(student_action)
    teacher_target = teacher_default + teacher_scale * teacher_action
    student_target = student_default + student_scale * student_action
    teacher_targets.append(teacher_target)
    student_targets.append(student_target)
    raw_differences.append(teacher_action - student_action)
    target_differences.append(teacher_target - student_target)
  # Reference anchor parity is checked in the fixture-preparation audit;
  # this evaluator keeps the policy-state array compact and does not need
  # to duplicate native quaternion columns.
  teacher_actions_array = np.asarray(teacher_actions)
  student_actions_array = np.asarray(student_actions)
  teacher_targets_array = np.asarray(teacher_targets)
  student_targets_array = np.asarray(student_targets)
  raw_difference_array = np.asarray(raw_differences)
  target_difference_array = np.asarray(target_differences)
  frames = states["reference_frame"][indexes]
  result: dict[str, Any] = {
    "rows": int(len(indexes)),
    "reference_frames": [int(frames.min()), int(frames.max())],
    "reference_table_parity": {
      "teacher_q_max_abs": max(q_errors),
      "teacher_dq_max_abs": max(dq_errors),
    },
    "student_graph_replay_parity": {
      "latent_max_abs": max(latent_errors),
      "action_max_abs": max(student_action_errors),
    },
    "whole": {
      "teacher_vs_student_raw_action": rms_stats(raw_difference_array),
      "teacher_vs_student_target_rad": rms_stats(target_difference_array),
    },
    "per_window": {},
    "first_thresholds": {},
    "arrays": {
      "reference_frame": frames,
      "teacher_action": teacher_actions_array,
      "student_action": student_actions_array,
      "teacher_target": teacher_targets_array,
      "student_target": student_targets_array,
      "target_difference": target_difference_array,
    },
  }
  for name, (low, high) in WINDOWS.items():
    mask = (frames >= low) & (frames < high)
    if not np.any(mask):
      continue
    result["per_window"][name] = {
      "frames": [int(frames[mask].min()), int(frames[mask].max())],
      "teacher_vs_student_raw_action": rms_stats(raw_difference_array[mask]),
      "teacher_vs_student_target_rad": rms_stats(target_difference_array[mask]),
    }
  disagreement = np.sqrt(np.mean(target_difference_array**2, axis=1))
  for threshold in (0.02, 0.05, 0.10, 0.20):
    indices = np.flatnonzero(disagreement >= threshold)
    result["first_thresholds"][str(threshold)] = (
      None
      if not len(indices)
      else {
        "frame": int(frames[indices[0]]),
        "relative_s": float(frames[indices[0]] * 0.02),
        "rms_target_difference_rad": float(disagreement[indices[0]]),
      }
    )
  return result


def make_plot(results: dict[str, dict[str, Any]], output: Path) -> None:
  colors = {
    "failure": "tab:red",
    "r4_g2": "tab:blue",
    "r4_g4": "tab:green",
    "r4_g6": "tab:orange",
  }
  figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
  window_names = list(WINDOWS)
  for label, result in results.items():
    arrays = result["arrays"]
    time = arrays["reference_frame"] * 0.02
    disagreement = np.sqrt(np.mean(arrays["target_difference"] ** 2, axis=1))
    axes[0].plot(time, disagreement, label=label, color=colors[label])
  axes[0].axvline(
    1.369,
    color="black",
    linestyle="--",
    linewidth=0.8,
    label="failure native pelvis 30 deg",
  )
  axes[0].set_ylabel("teacher-student target RMS (rad)")
  axes[0].set_title("VAE teacher-fidelity replay on recorded X2 states")
  axes[0].grid(alpha=0.2)
  axes[0].legend(ncol=3, fontsize=8)

  x = np.arange(len(window_names))
  width = 0.18
  for offset, (label, result) in enumerate(results.items()):
    values = [
      result["per_window"]
      .get(name, {})
      .get("teacher_vs_student_target_rad", {})
      .get("rms", np.nan)
      for name in window_names
    ]
    axes[1].bar(
      x + (offset - 1.5) * width, values, width, label=label, color=colors[label]
    )
  axes[1].set_xticks(x, [name.replace("_", "-") for name in window_names])
  axes[1].set_ylabel("target RMS (rad)")
  axes[1].set_xlabel("fixed reference-frame window")
  axes[1].grid(axis="y", alpha=0.2)
  axes[1].legend(ncol=4, fontsize=8)
  figure.savefig(output, dpi=160)
  plt.close(figure)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--states", type=Path, required=True, help="recorded_states.npz")
  parser.add_argument(
    "--model-dir",
    type=Path,
    required=True,
    help="directory containing frozen model copies",
  )
  parser.add_argument(
    "--output-dir", type=Path, required=True, help="directory for metrics and plots"
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  model_dir = args.model_dir
  model_paths = {
    "teacher": model_dir / "baseline-tennis-001.onnx",
    "encoder": model_dir / "encoder.onnx",
    "decoder": model_dir / "decoder.onnx",
    "contract": model_dir / "contract.json",
  }
  for path in model_paths.values():
    if not path.is_file():
      raise FileNotFoundError(path)
  states = load_states(args.states)
  options = session_options()
  teacher = ort.InferenceSession(
    str(model_paths["teacher"]), options, providers=["CPUExecutionProvider"]
  )
  encoder = ort.InferenceSession(
    str(model_paths["encoder"]), options, providers=["CPUExecutionProvider"]
  )
  decoder = ort.InferenceSession(
    str(model_paths["decoder"]), options, providers=["CPUExecutionProvider"]
  )
  teacher_default, teacher_scale, teacher_metadata = model_arrays(teacher)
  student_default, student_scale = load_contract(model_paths["contract"])
  results: dict[str, dict[str, Any]] = {}
  failure = states["run_kind"] == "failure"
  successful = states["run_kind"] == "successful"
  for label, mask in (
    ("failure", failure),
    ("r4_g2", successful & (states["engagement_generation"] == 2)),
    ("r4_g4", successful & (states["engagement_generation"] == 4)),
    ("r4_g6", successful & (states["engagement_generation"] == 6)),
  ):
    indexes = np.flatnonzero(mask)
    if len(indexes):
      results[label] = run_group(
        states,
        indexes,
        teacher,
        encoder,
        decoder,
        teacher_default,
        teacher_scale,
        student_default,
        student_scale,
      )
  results_for_json = {}
  for label, result in results.items():
    results_for_json[label] = {
      key: value for key, value in result.items() if key != "arrays"
    }
    arrays = result["arrays"]
    np.savez_compressed(args.output_dir / f"{label}.npz", **arrays)
  metadata = {
    "scope": "one-step teacher/student fidelity replay on recorded X2 VAE states",
    "not_a_rollout": True,
    "states_sha256": sha256(args.states),
    "model_sha256": {name: sha256(path) for name, path in model_paths.items()},
    "teacher_observation_layout": "command62 + anchor6 + pelvis gyro3 + relative q31 + dq31 + previous action31 = 164",
    "student_input_layout": "reference68 + conditioning99",
    "teacher_time_step": "recorded reference_frame",
    "action_scale_metadata_max_abs_difference": float(
      np.max(np.abs(teacher_scale - student_scale))
    ),
    "teacher_metadata": {
      key: teacher_metadata[key] for key in ("observation_names", "anchor_body_name")
    },
    "results": results_for_json,
  }
  (args.output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
  make_plot(results, args.output_dir / "comparison.png")
  print(
    json.dumps(
      {label: result["whole"] for label, result in results_for_json.items()}, indent=2
    )
  )


if __name__ == "__main__":
  main()
