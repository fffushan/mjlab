"""Render the generated X2 guard040 tennis references as headless MuJoCo GIFs.

This is *kinematic reference playback*, with no policy, physics integration or
balance claim. The camera follows pelvis XY to keep long court translations in
view. Example from mjlab root:

  MUJOCO_GL=egl uv run python scripts/render_x2_tennis_guard_gifs.py
  MUJOCO_GL=egl uv run python scripts/render_x2_tennis_guard_gifs.py --clip 000 --output-dir /tmp/guard-preview
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Select the headless backend BEFORE MuJoCo imports its rendering implementation.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mjlab.asset_zoo.robots.agibot_x2.x2_constants import X2_XML  # noqa: E402

SOURCE = Path("data/tennis/guard040_linear05")
SIZE = 384
STRIDE = 2
TILE = 192
GRID_COLUMNS = 5


def model_with_floor(width: int) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(X2_XML))
  spec.worldbody.add_geom(
    name="guard_reference_render_floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[100.0, 100.0, 0.1],
    rgba=[0.78, 0.80, 0.83, 1.0],
    group=0,
  )
  model = spec.compile()
  model.vis.global_.offwidth = width
  model.vis.global_.offheight = width
  return model


def scene_options() -> mujoco.MjvOption:
  options = mujoco.MjvOption()
  options.geomgroup[:] = 0
  options.geomgroup[0] = 1  # floor
  options.geomgroup[1] = 1  # visual meshes; omit collision geometry in group 3
  return options


def camera() -> mujoco.MjvCamera:
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance = 2.4
  cam.azimuth = 125.0
  cam.elevation = -12.0
  return cam


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
  return (
    ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()
  )


def render_reference_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  renderer: mujoco.Renderer,
  cam: mujoco.MjvCamera,
  options: mujoco.MjvOption,
  positions: np.ndarray,
  orientations: np.ndarray,
  joints: np.ndarray,
  index: int,
) -> Image.Image:
  data.qpos[:3] = positions[index]
  data.qpos[3:7] = orientations[index]
  data.qpos[7:] = joints[index]
  mujoco.mj_forward(model, data)
  cam.lookat[:2] = positions[index, :2]
  cam.lookat[2] = 0.75
  renderer.update_scene(data, camera=cam, scene_option=options)
  return Image.fromarray(renderer.render())


def annotate(
  image: Image.Image, clip_id: str, index: int, total: int, original: int
) -> None:
  draw = ImageDraw.Draw(image)
  section = (
    "GUARD -> CLIP"
    if index < 25
    else "CLIP"
    if index < 25 + original
    else "CLIP -> GUARD"
  )
  accent = (
    "#68d47a" if index < 25 else "#7cc9ff" if index < 25 + original else "#ffb969"
  )
  draw.rectangle((0, 0, image.width, 49), fill="#171a22")
  draw.text((8, 5), f"TENNIS {clip_id}  |  {section}", fill=accent, font=font(16))
  draw.text(
    (8, 27),
    f"{index / 50:.2f}s / {(total - 1) / 50:.2f}s  |  reference only",
    fill="white",
    font=font(12),
  )
  draw.rectangle((0, image.height - 5, image.width, image.height), fill="#191c25")
  draw.rectangle(
    (0, image.height - 5, int(image.width * index / (total - 1)), image.height),
    fill=accent,
  )


def save_gif(path: Path, images: list[Image.Image], ms_per_frame: int) -> None:
  # FASTOCTREE quantization bounds CPU and memory for 20 multi-second clips.
  frames = [
    frame.quantize(colors=128, method=Image.Quantize.FASTOCTREE) for frame in images
  ]
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
    prefix=f".{path.stem}-", suffix=".gif", dir=path.parent, delete=False
  ) as tmp:
    staging = Path(tmp.name)
  try:
    frames[0].save(
      staging,
      save_all=True,
      append_images=frames[1:],
      duration=ms_per_frame,
      loop=0,
      optimize=False,
      disposal=2,
    )
    staging.replace(path)
  finally:
    staging.unlink(missing_ok=True)


def render_clip(
  model: mujoco.MjModel,
  renderer: mujoco.Renderer,
  options: mujoco.MjvOption,
  clip_path: Path,
  output: Path,
  stride: int,
) -> tuple[str, int]:
  clip_id = clip_path.name.split("_")[1]
  with np.load(clip_path, allow_pickle=False) as clip:
    positions = clip["body_pos_w"][:, 0, :]
    orientations = clip["body_quat_w"][:, 0, :]
    joints = clip["joint_pos"]
    total = joints.shape[0]
    if float(clip["fps"][0]) != 50.0 or total <= 50:
      raise ValueError(f"invalid 50 Hz guard motion: {clip_path}")
    original = total - 50
    indices = list(range(0, total, stride))
    if indices[-1] != total - 1:
      indices.append(total - 1)
    data = mujoco.MjData(model)
    cam = camera()
    images = []
    for index in indices:
      frame = render_reference_frame(
        model, data, renderer, cam, options, positions, orientations, joints, index
      )
      annotate(frame, clip_id, index, total, original)
      images.append(frame)
    save_gif(output / f"tennis_{clip_id}.gif", images, round(1000 * stride / 50))
    return clip_id, len(indices)


def render_overview(
  model: mujoco.MjModel,
  renderer: mujoco.Renderer,
  options: mujoco.MjvOption,
  clips: list[Path],
  output: Path,
) -> None:
  """20-way comparison of entry and exit only; explicit CUT skips each clip's core."""
  if len(clips) != 20:
    return
  # 13 frames for the 0.5-s prefix, a marked cut, 13 for the suffix.
  samples = [*range(0, 25, 2), *range(-25, 0, 2)]
  frames = []
  data = mujoco.MjData(model)
  cam = camera()
  sample_cache: list[list[Image.Image]] = []
  for path in clips:
    ident = path.name.split("_")[1]
    with np.load(path, allow_pickle=False) as clip:
      positions = clip["body_pos_w"][:, 0, :]
      orientations = clip["body_quat_w"][:, 0, :]
      joints = clip["joint_pos"]
      total = len(joints)
      indices = [
        i if i >= 0 else (total - 1 if i == -1 else total + i) for i in samples
      ]
      shots = []
      for t, index in enumerate(indices):
        shot = render_reference_frame(
          model, data, renderer, cam, options, positions, orientations, joints, index
        )
        shot = shot.resize((TILE, TILE), resample=Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(shot)
        draw.rectangle((0, 0, TILE, 23), fill="#171a22")
        draw.text(
          (5, 4),
          f"{ident}  {'ENTRY' if t < 13 else 'EXIT'}",
          fill="white",
          font=font(12),
        )
        shots.append(shot)
      sample_cache.append(shots)
  for t in range(len(samples)):
    canvas = Image.new("RGB", (GRID_COLUMNS * TILE, 4 * TILE + 34), "#171a22")
    for i, shots in enumerate(sample_cache):
      canvas.paste(
        shots[t], ((i % GRID_COLUMNS) * TILE, (i // GRID_COLUMNS) * TILE + 34)
      )
    draw = ImageDraw.Draw(canvas)
    label = (
      "GUARD -> CLIP  (first 0.5s)" if t < 13 else "CUT: CLIP -> GUARD  (last 0.5s)"
    )
    draw.text(
      (10, 7), f"{label}  |  KINEMATIC REFERENCE ONLY", fill="white", font=font(16)
    )
    frames.append(canvas)
  save_gif(output / "all_20_transitions.gif", frames, 80)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-dir", type=Path, default=SOURCE)
  parser.add_argument("--output-dir", type=Path, default=SOURCE / "renders")
  parser.add_argument("--clip", help="one numeric ID (e.g. 000); default: all 20")
  parser.add_argument(
    "--stride", type=int, default=STRIDE, help="reference frames per GIF frame"
  )
  parser.add_argument("--size", type=int, default=SIZE)
  args = parser.parse_args()
  if args.stride < 1 or args.size < 128:
    parser.error("stride must be >=1 and size must be >=128")
  manifest = json.loads((args.source_dir / "manifest.json").read_text())
  clips = [args.source_dir / entry["clip"] for entry in manifest["members"]]
  if args.clip is not None:
    clips = [clip for clip in clips if clip.name.split("_")[1] == args.clip]
    if len(clips) != 1:
      parser.error(f"unknown clip id {args.clip}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  model = model_with_floor(args.size)
  opts = scene_options()
  with mujoco.Renderer(model, height=args.size, width=args.size) as renderer:
    for path in clips:
      ident, count = render_clip(
        model, renderer, opts, path, args.output_dir, args.stride
      )
      print(f"rendered tennis_{ident}.gif ({count} frames)", flush=True)
    if args.clip is None and args.size >= TILE:
      render_overview(model, renderer, opts, clips, args.output_dir)
      print("rendered all_20_transitions.gif", flush=True)
  (args.output_dir / "README.md").write_text(
    "# Kinematic previews of guard040 reference motions\n\n"
    "Individual tennis_XXX.gif files show the full clip at 25 fps (sampled from 50 Hz). "
    "The camera follows pelvis XY. Green/blue/orange captions mark entry/original/exit. "
    "all_20_transitions.gif shows only the two half-second transitions, with an explicit cut "
    "between entry and exit, played at 2x slow motion. These are poses replayed with MuJoCo forward kinematics, NOT "
    "closed-loop tracking or balance simulations.\n\n"
    + "\n".join(
      f"- [tennis {clip.name.split('_')[1]}](tennis_{clip.name.split('_')[1]}.gif)"
      for clip in clips
    )
    + "\n"
  )
  print(f"GIFs in {args.output_dir}", flush=True)


if __name__ == "__main__":
  main()
