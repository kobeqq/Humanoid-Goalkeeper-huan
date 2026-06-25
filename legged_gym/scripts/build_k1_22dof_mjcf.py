#!/usr/bin/env python3
"""Build MuJoCo MJCF from K1_22dof.urdf (Isaac Gym training asset).

Output: resources/robots/k1/urdf/K1_22dof.xml
- Kinematics / inertia / collisions from URDF (via MuJoCo compiler)
- Floating base on body ``Trunk`` (matches sim2sim / Isaac fix_base_link=False)
- Ground plane, IMU site, torque motors with URDF effort limits
- Foot box contacts (stable sim2sim; mesh foot collision kept visual-only)

Run from repo root:
  python legged_gym/scripts/build_k1_22dof_mjcf.py
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import mujoco

URDF_DIR = Path(__file__).resolve().parents[1] / "resources" / "robots" / "k1" / "urdf"
URDF_PATH = URDF_DIR / "K1_22dof.urdf"
OUT_PATH = URDF_DIR / "K1_22dof.xml"

# Isaac K122Cfg.asset.armature
JOINT_ARMATURE = {
    "AAHead_yaw": 0.01,
    "Head_pitch": 0.01,
    "ALeft_Shoulder_Pitch": 0.01,
    "Left_Shoulder_Roll": 0.01,
    "Left_Elbow_Pitch": 0.01,
    "Left_Elbow_Yaw": 0.01,
    "ARight_Shoulder_Pitch": 0.01,
    "Right_Shoulder_Roll": 0.01,
    "Right_Elbow_Pitch": 0.01,
    "Right_Elbow_Yaw": 0.01,
    # A slightly larger lower-body armature makes MuJoCo rollout less brittle for
    # the Isaac-trained loco policy without changing the policy or joint order.
    "Left_Hip_Pitch": 0.05,
    "Left_Hip_Roll": 0.05,
    "Left_Hip_Yaw": 0.05,
    "Left_Knee_Pitch": 0.05,
    "Left_Ankle_Pitch": 0.05,
    "Left_Ankle_Roll": 0.05,
    "Right_Hip_Pitch": 0.05,
    "Right_Hip_Roll": 0.05,
    "Right_Hip_Yaw": 0.05,
    "Right_Knee_Pitch": 0.05,
    "Right_Ankle_Pitch": 0.05,
    "Right_Ankle_Roll": 0.05,
}

ACTUATOR_ORDER = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def _prepare_urdf_text() -> str:
    text = URDF_PATH.read_text()
    text = text.replace("<!-- <mujoco>", "<mujoco>").replace("</mujoco> -->", "</mujoco>")
    if "meshdir" not in text:
        text = text.replace(
            "<mujoco>",
            '<mujoco>\n    <compiler meshdir="meshes/" balanceinertia="true" discardvisual="false"/>',
            1,
        )
    return text


def _compile_urdf() -> tuple[str, str]:
    """Return (asset_inner_xml, worldbody_inner_xml) from MuJoCo URDF import."""
    tmp_urdf = URDF_DIR / "_K1_22dof_build.urdf"
    tmp_xml = URDF_DIR / "_K1_22dof_compiled.xml"
    try:
        tmp_urdf.write_text(_prepare_urdf_text())
        model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
        mujoco.mj_saveLastXML(str(tmp_xml), model)
        compiled = tmp_xml.read_text()
    finally:
        tmp_urdf.unlink(missing_ok=True)
        tmp_xml.unlink(missing_ok=True)

    asset_m = re.search(r"<asset>(.*)</asset>", compiled, flags=re.DOTALL)
    world_m = re.search(r"<worldbody>(.*)</worldbody>", compiled, flags=re.DOTALL)
    if not asset_m or not world_m:
        raise RuntimeError("Failed to parse compiled asset/worldbody from URDF")
    return asset_m.group(1).strip(), world_m.group(1).strip()


def _patch_joint_lines(robot_inner: str) -> str:
    lines = []
    for line in robot_inner.splitlines():
        jm = re.search(r'<joint name="([^"]+)"', line)
        if jm and "/>" in line:
            jname = jm.group(1)
            arm = JOINT_ARMATURE.get(jname, 0.01)
            if "armature=" not in line:
                line = line.replace("/>", f' armature="{arm}" />')
        lines.append(line)
    return "\n".join(lines)


def _add_foot_box_geoms(robot_inner: str) -> str:
    """Add box foot contact (sim2sim); disable mesh foot collision."""
    foot_box = (
        '                <geom name="{name}_contact" type="box" '
        'pos="0.014 0 -0.008" size="0.08 0.035 0.016" '
        'rgba="0.75294 0.75294 0.75294 1"/>'
    )
    replacements = [
        (
            '<geom type="mesh" rgba="0.75294 0.75294 0.75294 1" mesh="Left_Foot"/>',
            '<geom type="mesh" contype="0" conaffinity="0" rgba="0.75294 0.75294 0.75294 1" mesh="Left_Foot"/>\n'
            + foot_box.format(name="left_foot_link"),
        ),
        (
            '<geom type="mesh" rgba="0.75294 0.75294 0.75294 1" mesh="Right_Foot"/>',
            '<geom type="mesh" contype="0" conaffinity="0" rgba="0.75294 0.75294 0.75294 1" mesh="Right_Foot"/>\n'
            + foot_box.format(name="right_foot_link"),
        ),
    ]
    out = robot_inner
    for old, new in replacements:
        if old not in out:
            raise RuntimeError(f"Foot mesh marker not found: {old[:50]}...")
        out = out.replace(old, new, 1)
    return out


def _parse_joint_torque_limits(robot_inner: str) -> dict[str, float]:
    limits = {}
    for jname in ACTUATOR_ORDER:
        m = re.search(
            rf'<joint name="{jname}"[^>]*actuatorfrcrange="-([0-9.]+) \1"',
            robot_inner,
        )
        if not m:
            m = re.search(rf'<joint name="{jname}"[^>]*actuatorfrcrange="-([0-9.]+)', robot_inner)
        if m:
            limits[jname] = float(m.group(1))
        else:
            limits[jname] = 20.0
    return limits


def _build_actuators(torque_limits: dict[str, float]) -> str:
    lines = []
    for jname in ACTUATOR_ORDER:
        t = torque_limits[jname]
        lines.append(f'    <motor name="{jname}" joint="{jname}" forcerange="-{t} {t}"/>')
    return "\n".join(lines)


def build_mjcf() -> str:
    asset_inner, robot_inner = _compile_urdf()
    robot_inner = _patch_joint_lines(robot_inner)
    robot_inner = _add_foot_box_geoms(robot_inner)
    torque_limits = _parse_joint_torque_limits(robot_inner)

    indented_asset = textwrap.indent(asset_inner, "    ")
    indented_robot = textwrap.indent(robot_inner, "      ")
    actuators = _build_actuators(torque_limits)

    return f"""<mujoco model="K1_22dof">
  <!-- Generated by legged_gym/scripts/build_k1_22dof_mjcf.py from K1_22dof.urdf -->
  <compiler angle="radian" meshdir="meshes/" autolimits="true"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
             width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
{indented_asset}
  </asset>

  <worldbody>
    <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
    <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
    <geom name="ground" type="plane" pos="0 0 0" size="0 0 1" material="matplane" condim="3"
          friction="1 1 0.005" solimp="0.9 0.95 0.001 0.5 2" solref="0.02 1"/>

    <body name="Trunk">
      <site name="imu" size="0.01" pos="0 0 0"/>
      <joint name="world_joint" type="free" limited="false" actuatorfrclimited="false"/>
{indented_robot}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
    <framequat name="orientation" objtype="site" objname="imu" noise="0.001"/>
    <gyro name="angular-velocity" site="imu" noise="0.005"/>
  </sensor>
</mujoco>
"""


def _verify(path: Path) -> None:
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.nq == 29, f"expected nq=29 (7 free + 22 joints), got {m.nq}"
    assert m.nu == 22, f"expected nu=22, got {m.nu}"
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
    lfoot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    rfoot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    print(f"[OK] {path.name}: nq={m.nq} nv={m.nv} nu={m.nu}")
    print(f"     bodies: Trunk={trunk} left_foot={lfoot} right_foot={rfoot}")


def main() -> None:
    xml = build_mjcf()
    OUT_PATH.write_text(xml)
    _verify(OUT_PATH)
    print(f"[OK] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
