"""Render the X2 tennis bat-carry pose offscreen, as a PNG contact sheet.

A preview tool, not a test: it compiles the tennis robot spec, puts it in the
task's bat-carry keyframe, drops the ball at its spawn point, and renders four
fixed views. Use it to check a change to the carry pose, the racket transform or
the ball spawn without opening a viewer.

    MUJOCO_GL=egl uv run python scripts/experiments/x2_tennis_carry_pose_views.py \
        --out /tmp/carry.png

``--tint-bed`` paints the string-bed *collision* surface magenta and translucent,
which is the only way to see how much smaller it is than the painted hoop: the
plate is inset 18 mm per side from the painted head.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from mjlab.asset_zoo.robots.agibot_x2.tennis_constants import (
  BALL_SPAWN_LOCAL_POS,
  BAT_CARRY_BASE_HEIGHT,
  BAT_CARRY_POSE,
  BED_GEOM,
  FACE_SITE,
  get_tennis_ball_spec,
  get_tennis_spec,
)

WIDTH = 640
HEIGHT = 640


@dataclass(frozen=True)
class View:
  """One free-camera placement, in MuJoCo's azimuth/elevation convention."""

  lookat: tuple[float, float, float]
  distance: float
  azimuth: float
  elevation: float


VIEWS: dict[str, View] = {
  "carry pose, 3/4 front-right": View((0.25, 0.0, 0.78), 2.3, -118.0, -12.0),
  "right side, camera level (bat horizontality)": View(
    (0.30, 0.0, 0.80), 2.2, -90.0, 0.0
  ),
  "string bed from above": View((0.56, -0.024, 1.13), 0.70, -90.0, -88.0),
  "ball-on-bed profile (1 mm spawn clearance)": View(
    (0.56, -0.024, 1.15), 0.40, -150.0, -6.0
  ),
}


def build_model(*, tint_bed: bool) -> mujoco.MjModel:
  spec = get_tennis_spec()
  spec.attach(get_tennis_ball_spec(), prefix="ball/", frame=spec.worldbody.add_frame())
  spec.worldbody.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5.0, 5.0, 0.1],
    rgba=[0.35, 0.37, 0.40, 1.0],
    contype=1,
    conaffinity=1,
  )
  # Shadowless directional lights: this is a shape preview, and a shadow-mapped
  # directional light over a 5 m plane otherwise covers the scene in acne.
  spec.worldbody.add_light(
    name="key",
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    pos=[0.0, 0.0, 3.0],
    dir=[-0.35, 0.45, -0.82],
    diffuse=[0.9, 0.9, 0.9],
    castshadow=False,
  )
  spec.worldbody.add_light(
    name="fill",
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    pos=[0.0, 0.0, 2.0],
    dir=[0.5, -0.6, -0.62],
    diffuse=[0.4, 0.4, 0.42],
    castshadow=False,
  )
  if tint_bed:
    bed = spec.geom(BED_GEOM)
    bed.rgba = [1.0, 0.1, 0.9, 0.35]
    bed.group = 1
  spec.visual.global_.offwidth = WIDTH
  spec.visual.global_.offheight = HEIGHT
  return spec.compile()


def carry_data(model: mujoco.MjModel) -> mujoco.MjData:
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  free = int(np.ravel(model.joint("floating_base_joint").qposadr)[0])
  data.qpos[free : free + 3] = [0.0, 0.0, BAT_CARRY_BASE_HEIGHT]
  data.qpos[free + 3 : free + 7] = [1.0, 0.0, 0.0, 0.0]
  for name, value in BAT_CARRY_POSE.items():
    data.qpos[int(np.ravel(model.joint(name).qposadr)[0])] = value
  ball_q = int(np.ravel(model.joint("ball/ball_free").qposadr)[0])
  data.qpos[ball_q : ball_q + 3] = BALL_SPAWN_LOCAL_POS
  data.qpos[ball_q + 3 : ball_q + 7] = [1.0, 0.0, 0.0, 0.0]
  mujoco.mj_forward(model, data)
  return data


def _labelled(image: np.ndarray, text: str) -> np.ndarray:
  pil = Image.fromarray(image)
  draw = ImageDraw.Draw(pil)
  draw.rectangle([0, 0, pil.width, 26], fill=(0, 0, 0))
  draw.text((8, 7), text, fill=(255, 255, 255))
  return np.asarray(pil)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", default="/tmp/x2_tennis_carry_pose.png")
  parser.add_argument(
    "--tint-bed",
    action="store_true",
    help="paint the string-bed collision surface magenta and translucent",
  )
  args = parser.parse_args()

  model = build_model(tint_bed=args.tint_bed)
  data = carry_data(model)
  site = int(np.ravel(model.site(FACE_SITE).id)[0])
  print("pelvis z            :", round(float(data.qpos[2]), 6))
  print("face_center (world) :", np.round(data.site_xpos[site], 6).tolist())
  print("ball centre (world) :", np.round(BALL_SPAWN_LOCAL_POS, 6).tolist())

  renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  tiles = []
  for title, view in VIEWS.items():
    camera.lookat[:] = view.lookat
    camera.distance = view.distance
    camera.azimuth = view.azimuth
    camera.elevation = view.elevation
    renderer.update_scene(data, camera=camera)
    label = title + (" [collision bed tinted]" if args.tint_bed else "")
    tiles.append(_labelled(renderer.render(), label))
  renderer.close()

  top = np.concatenate(tiles[:2], axis=1)
  bottom = np.concatenate(tiles[2:], axis=1)
  Image.fromarray(np.concatenate([top, bottom], axis=0)).save(args.out)
  print("wrote", args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
