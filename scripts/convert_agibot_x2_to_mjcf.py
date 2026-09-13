"""Convert an AgiBot X2 URDF/MJCF release into an mjlab-convention MJCF.

The vendor releases (https://github.com/AgibotTech/agibot_x2_urdf) ship an
Isaac-Sim flavored MJCF plus a URDF. Neither is directly usable as an mjlab
asset:

* the MJCF carries ``<motor>`` actuators (mjlab builds its own position
  actuators), Isaac-style sensor names, and no body inertials -- mass comes
  from mesh density, which is ~3.6% off the URDF values,
* every geom is unnamed, so mjlab's name-pattern based collision and friction
  configuration (which needs to tell feet from the rest) cannot match anything.

This script rewrites the MJCF in place-free fashion: it injects the URDF
inertials per body, names the collision geoms, drops the actuators, replaces
the sensor block with mjlab's IMU sensor names, copies the referenced meshes,
and writes the result next to them as ``<model>.xml``.

Running it for the flagship (``X2_URDF-v1.3.0``)::

  uv run python scripts/convert_agibot_x2_to_mjcf.py \
    --src-dir ~/projects/agibot_x2_urdf/X2_URDF-v1.3.0 \
    --model x2_ultra \
    --dst-dir src/mjlab/asset_zoo/robots/agibot_x2/xmls
"""

from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import tyro

import mjlab

# Foot collision spheres live in the ankle-roll body; everything else becomes
# ``<body>_collision`` so mjlab can select feet by name.
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

# mjlab reads IMU observations through these sensor names (see g1.xml).
IMU_SITE = "imu_0"
SENSORS = (
  ("gyro", {"name": "imu_ang_vel", "site": IMU_SITE}),
  ("velocimeter", {"name": "imu_lin_vel", "site": IMU_SITE}),
  ("accelerometer", {"name": "imu_lin_acc", "site": IMU_SITE}),
  (
    "framezaxis",
    {
      "name": "imu_upvector",
      "objtype": "body",
      "objname": "world",
      "reftype": "site",
      "refname": IMU_SITE,
    },
  ),
  ("subtreeangmom", {"name": "root_angmom", "body": "pelvis"}),
)


def _rpy_to_quat(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
  """Convert URDF fixed-axis RPY to a (w, x, y, z) quaternion."""
  r, p, y = (a / 2.0 for a in rpy)
  cr, sr = math.cos(r), math.sin(r)
  cp, sp = math.cos(p), math.sin(p)
  cy, sy = math.cos(y), math.sin(y)
  return (
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  )


def parse_urdf_inertials(path: Path) -> dict[str, dict]:
  """Extract per-link inertial data (mass, CoM, inertia tensor)."""
  root = ET.parse(path).getroot()
  out: dict[str, dict] = {}
  for link in root.findall("link"):
    ine = link.find("inertial")
    link_name = link.get("name")
    assert link_name is not None
    if ine is None:
      continue
    mass_element = ine.find("mass")
    inertia_element = ine.find("inertia")
    assert mass_element is not None
    assert inertia_element is not None
    mass = float(mass_element.get("value", "0"))
    origin = ine.find("origin")
    xyz_str = origin.get("xyz") if origin is not None else None
    rpy_str = origin.get("rpy") if origin is not None else None
    xyz = [float(v) for v in (xyz_str or "0 0 0").split()]
    rpy = [float(v) for v in (rpy_str or "0 0 0").split()]
    inertia = {
      k: float(inertia_element.get(k, "0") or 0.0)
      for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    }
    out[link_name] = dict(mass=mass, xyz=xyz, rpy=rpy, inertia=inertia)
  return out


def make_inertial(data: dict) -> ET.Element:
  """Build a MuJoCo ``<inertial>`` element from URDF inertial data."""
  el = ET.Element("inertial")
  el.set("pos", " ".join(f"{v:g}" for v in data["xyz"]))
  quat = _rpy_to_quat(tuple(data["rpy"]))
  if quat != (1.0, 0.0, 0.0, 0.0):
    el.set("quat", " ".join(f"{v:g}" for v in quat))
  i = data["inertia"]
  el.set(
    "fullinertia",
    " ".join(f"{i[k]:g}" for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")),
  )
  el.set("mass", f"{data['mass']:g}")
  return el


def name_collision_geoms(root: ET.Element) -> tuple[int, int]:
  """Give every collision geom a unique name. Returns (n_named, n_feet)."""
  n_named = n_feet = 0
  for body in root.iter("body"):
    body_name = body.get("name")
    assert body_name is not None
    foot_index = 0
    used: set[str] = set()
    for geom in body.findall("geom"):
      if geom.get("class") != "collision":
        continue
      if body_name in FOOT_BODIES:
        foot_index += 1
        name = f"{body_name.split('_')[0]}_foot{foot_index}_collision"
        n_feet += 1
      else:
        name = f"{body_name}_collision"
        if name in used:
          raise ValueError(f"duplicate collision name {name!r}")
        used.add(name)
      geom.set("name", name)
      n_named += 1
  return n_named, n_feet


def main(
  src_dir: Path,
  model: str = "x2_ultra",
  dst_dir: Path = Path("src/mjlab/asset_zoo/robots/agibot_x2/xmls"),
) -> None:
  """Convert ``src_dir/<model>.xml`` into an mjlab asset under ``dst_dir``."""
  src_xml = src_dir / f"{model}.xml"
  src_urdf = src_dir / f"{model}.urdf"
  assert src_xml.exists(), src_xml
  assert src_urdf.exists(), src_urdf

  tree = ET.parse(src_xml)
  root = tree.getroot()

  # Meshes move next to the converted xml, under assets/.
  compiler = root.find("compiler")
  assert compiler is not None
  compiler.set("meshdir", "assets")

  # mjlab builds position actuators from EntityArticulationInfoCfg, and reads
  # joint state from the data arrays rather than from sensors.
  for tag in ("actuator", "sensor"):
    for el in root.findall(tag):
      root.remove(el)
  sensors = ET.SubElement(root, "sensor")
  for tag, attrs in SENSORS:
    el = ET.SubElement(sensors, tag)
    for k, v in attrs.items():
      el.set(k, v)
  # Move the sensor block after worldbody, matching the usual MJCF ordering.
  root.remove(sensors)
  root.append(sensors)

  inertials = parse_urdf_inertials(src_urdf)
  n_bodies = n_inertials = 0
  urdf_mass = sum(d["mass"] for d in inertials.values())
  for body in root.iter("body"):
    n_bodies += 1
    name = body.get("name")
    if name not in inertials:
      raise KeyError(f"body {name!r} has no matching URDF link inertial")
    # The vendor xml only states inertials for some bodies (rounded to
    # diagonal form), so replace whatever is there with the URDF values.
    for existing in body.findall("inertial"):
      body.remove(existing)
    body.insert(0, make_inertial(inertials[name]))
    n_inertials += 1

  n_geoms, n_feet = name_collision_geoms(root)

  dst_dir.mkdir(parents=True, exist_ok=True)
  assets = dst_dir / "assets"
  assets.mkdir(exist_ok=True)
  n_meshes = 0
  for mesh in root.findall("asset/mesh"):
    file = mesh.get("file")
    assert file is not None
    src_mesh = src_dir / "meshes" / file
    if not src_mesh.exists():
      raise FileNotFoundError(src_mesh)
    shutil.copy2(src_mesh, assets / file)
    n_meshes += 1

  ET.indent(tree, space="  ")
  header = (
    f"<!-- AgiBot X2 ({model}) MJCF for mjlab.\n\n"
    f"     Converted from the vendor release {src_dir.name}/{model}.xml by\n"
    f"     scripts/convert_agibot_x2_to_mjcf.py: body inertials taken from\n"
    f"     {model}.urdf, collision geoms named (feet as <side>_footN_collision),\n"
    f"     actuators removed, sensors renamed to mjlab's IMU conventions.\n-->\n"
  )
  out = dst_dir / f"{model}.xml"
  body = ET.tostring(root, encoding="unicode")
  out.write_text(header + body + "\n")

  print(f"bodies: {n_bodies} ({n_inertials} with URDF inertials, {urdf_mass:.3f} kg)")
  print(f"collision geoms named: {n_geoms} ({n_feet} foot spheres)")
  print(f"meshes copied: {n_meshes}")
  print(f"wrote {out}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
