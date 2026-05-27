"""Sim2Sim: run Isaac Gym policies in MuJoCo.

Aligned with K1 Move AMP training (`LeggedRobotMoveAmp`, `K1MoveAmpCfg`) and
`deploy_mujoco.py` + `configs/k1_move_amp.yaml`.
"""

import argparse
import glob
import os
import select
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR

CONFIG_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy", "deploy_mujoco", "configs")
DEFAULT_CONFIG = "k1_move_amp.yaml"


def quat_rotate_inverse_xyzw(q, v):
    """Rotate vector v by inverse of quaternion q (xyzw)."""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def mujoco_quat_to_xyzw(quat_wxyz):
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)


def compute_torques(joint_pos_target, dof_pos, dof_vel, p_gains, d_gains):
    return p_gains * (joint_pos_target - dof_pos) - d_gains * dof_vel


def get_foot_body_ids(model):
    return [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_foot_link", "right_foot_link")
    ]


def min_foot_height(model, data, foot_ids):
    return float(min(data.xpos[i, 2] for i in foot_ids))


def spawn_robot(model, data, joint_pos, cfg):
    """Place robot at reset pose. foot_contact avoids the ~0.35 m air gap at isaac z=0.8."""
    spawn_mode = cfg.get("spawn_mode", "isaac")
    if cfg.get("auto_ground_base", False):
        spawn_mode = "foot_contact"

    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = joint_pos

    foot_ids = get_foot_body_ids(model)
    if spawn_mode == "foot_contact":
        target_foot = float(cfg.get("foot_contact_z", cfg.get("foot_clearance", 0.0)))
        lo, hi = 0.4, 0.95
        for _ in range(50):
            mid = (lo + hi) / 2
            data.qpos[2] = mid
            mujoco.mj_forward(model, data)
            if min_foot_height(model, data, foot_ids) > target_foot:
                hi = mid
            else:
                lo = mid
        data.qpos[2] = lo
    elif "base_pos" in cfg:
        data.qpos[:3] = np.array(cfg["base_pos"], dtype=np.float64)

    mujoco.mj_forward(model, data)
    if spawn_mode == "foot_contact":
        trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
        print(
            "[WARN] spawn_mode=foot_contact lowers trunk to ~0.45 m; policy expects ~0.8 m "
            "and gait often looks like hopping. Prefer spawn_mode: isaac for locomotion."
        )
    return spawn_mode, foot_ids


def apply_ground_friction(model, friction, contact_soft=True):
    if friction is None and not contact_soft:
        return
    mu = float(friction[0]) if friction is not None else 1.0
    mu_t = float(friction[1]) if friction is not None and len(friction) > 1 else mu
    for i in range(model.ngeom):
        if model.geom(i).name == "ground":
            model.geom_condim[i] = 3
            model.geom_friction[i, 0] = mu
            model.geom_friction[i, 1] = mu_t
            if contact_soft:
                # Softer contacts reduce foot bounce (hop-like gait) in MuJoCo.
                model.geom_solimp[i] = np.array([0.9, 0.95, 0.001, 0.5, 2.0], dtype=np.float64)
                model.geom_solref[i] = np.array([0.05, 1.0], dtype=np.float64)
            break


def setup_viewer_camera(viewer, model, cam_cfg):
    if not cam_cfg:
        return
    mode = cam_cfg.get("mode", "tracking")
    if mode == "tracking":
        body = cam_cfg.get("body", "Trunk")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    elif mode == "fixed":
        name = cam_cfg["name"]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if "lookat" in cam_cfg:
            viewer.cam.lookat[:] = np.array(cam_cfg["lookat"], dtype=np.float64)
    if "distance" in cam_cfg:
        viewer.cam.distance = float(cam_cfg["distance"])
    if "azimuth" in cam_cfg:
        viewer.cam.azimuth = float(cam_cfg["azimuth"])
    if "elevation" in cam_cfg:
        viewer.cam.elevation = float(cam_cfg["elevation"])


def load_config(config_name):
    path = config_name if os.path.isabs(config_name) else os.path.join(CONFIG_DIR, config_name)
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["policy_path"] = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    cfg["xml_path"] = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    return cfg


def build_one_step_obs(d, trunk_id, default_angles, target_world, target_use_z,
                       ang_vel_scale, dof_pos_scale, dof_vel_scale, action):
    """Match LeggedRobotMoveAmp.compute_observations actor slice (75 dims)."""
    trunk_pos = d.xpos[trunk_id].astype(np.float32)
    # Free joint is on Trunk — use qpos/qvel like Isaac root + upper_body on same link.
    quat_xyzw = mujoco_quat_to_xyzw(d.qpos[3:7])
    base_angvel_world = d.qvel[3:6].astype(np.float32)
    omega = quat_rotate_inverse_xyzw(quat_xyzw, base_angvel_world) * ang_vel_scale

    qj = d.qpos[7:].astype(np.float32)
    dqj = d.qvel[6:].astype(np.float32)
    qj = (qj - default_angles) * dof_pos_scale
    dqj = dqj * dof_vel_scale
    gravity = quat_rotate_inverse_xyzw(quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    target = target_world.copy()
    if not target_use_z:
        target[2] = trunk_pos[2]
    target_local = quat_rotate_inverse_xyzw(quat_xyzw, target - trunk_pos)
    if not target_use_z:
        target_local[2] = 0.0

    return np.concatenate((target_local, omega, gravity, qj, dqj, action)).astype(np.float32)


def resolve_policy_path(cfg, override):
    if override:
        return override
    path = cfg.get("policy_path")
    if path and os.path.isfile(path):
        return path
    candidates = glob.glob(
        os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "**", "exported", "*.pt"),
        recursive=True,
    )
    if candidates:
        return max(candidates, key=os.path.getmtime)
    raise FileNotFoundError(
        "Policy not found. Set policy_path in yaml or pass --policy_path."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sim2Sim MuJoCo deployment (K1 Move AMP / goal-conditioned locomotion)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"YAML config under deploy/deploy_mujoco/configs/ (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--policy_path", type=str, default=None, help="Override policy JIT path")
    parser.add_argument("--sim_duration", type=float, default=None, help="Override simulation_duration")
    parser.add_argument("--dt", type=float, default=None, help="Override simulation_dt")
    parser.add_argument("--decimation", type=int, default=None, help="Override control_decimation")
    parser.add_argument("--action_scale", type=float, default=None, help="Override action_scale")
    parser.add_argument(
        "--smooth_factor",
        type=float,
        default=None,
        help="EMA on policy actions (overrides yaml action_smooth_factor)",
    )
    parser.add_argument(
        "--target_smooth",
        type=float,
        default=None,
        help="EMA on joint targets after action_scale (overrides yaml)",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="World-frame goal point (default from yaml target_init)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    policy_path = resolve_policy_path(cfg, args.policy_path)
    xml_path = cfg["xml_path"]
    sim_duration = args.sim_duration if args.sim_duration is not None else cfg["simulation_duration"]
    sim_dt = args.dt if args.dt is not None else cfg["simulation_dt"]
    decimation = args.decimation if args.decimation is not None else cfg["control_decimation"]
    action_scale = args.action_scale if args.action_scale is not None else cfg["action_scale"]

    kp_gain_scale = float(cfg.get("kp_gain_scale", cfg.get("pd_gain_scale", 1.0)))
    kd_gain_scale = float(cfg.get("kd_gain_scale", cfg.get("pd_gain_scale", 1.0)))
    kps = np.array(cfg["kps"], dtype=np.float32) * kp_gain_scale
    kds = np.array(cfg["kds"], dtype=np.float32) * kd_gain_scale
    warmup_steps = int(cfg.get("warmup_steps", 0))
    action_smooth = (
        args.smooth_factor
        if args.smooth_factor is not None
        else float(cfg.get("action_smooth_factor", 0.0))
    )
    target_smooth = (
        args.target_smooth
        if args.target_smooth is not None
        else float(cfg.get("target_smooth_factor", 0.0))
    )
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)

    ang_vel_scale = cfg["ang_vel_scale"]
    dof_pos_scale = cfg["dof_pos_scale"]
    dof_vel_scale = cfg["dof_vel_scale"]

    num_actions = cfg["num_actions"]
    num_obs = cfg["num_obs"]
    num_one_step_obs = cfg.get("num_one_step_obs")
    num_actor_history = cfg.get("num_actor_history", 10)

    target_world = np.array(
        args.target if args.target is not None else cfg.get("target_init", [5.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    target_use_z = cfg.get("target_use_z", False)

    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Model: {xml_path}")
    print(f"[INFO] Policy: {policy_path}")
    print(f"[INFO] Obs: {num_one_step_obs} x {num_actor_history} = {num_obs}")
    print(f"[INFO] Target (world): {target_world}, use_z={target_use_z}")
    print(
        f"[INFO] PD kp_scale={kp_gain_scale}, kd_scale={kd_gain_scale}, "
        f"action_smooth={action_smooth}, target_smooth={target_smooth}, warmup={warmup_steps}"
    )

    policy = torch.jit.load(policy_path, map_location="cpu")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt
    if "mujoco_iterations" in cfg:
        m.opt.iterations = int(cfg["mujoco_iterations"])
    apply_ground_friction(
        m, cfg.get("ground_friction"), contact_soft=cfg.get("soft_ground_contact", False)
    )

    trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
    if "torque_limits" in cfg:
        torque_limits = np.array(cfg["torque_limits"], dtype=np.float32)
    else:
        torque_limits = m.actuator_forcerange[:, 1].astype(np.float32)

    spawn_joints = (
        np.array(cfg["init_joint_pos"], dtype=np.float64)
        if "init_joint_pos" in cfg
        else default_angles.astype(np.float64)
    )
    spawn_mode, foot_ids = spawn_robot(m, d, spawn_joints, cfg)
    foot_z = min_foot_height(m, d, foot_ids)
    print(
        f"[INFO] spawn_mode={spawn_mode}: trunk z={d.xpos[trunk_id, 2]:.3f}, "
        f"foot z={foot_z:.3f} (isaac play uses z=0.8 with feet on ground in PhysX)"
    )

    action = np.zeros(num_actions, dtype=np.float32)
    smoothed_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    filtered_target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    use_policy = warmup_steps == 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        setup_viewer_camera(viewer, m, cfg.get("viewer_camera"))
        print("Set goal (x, y, z) in world frame: ", end="", flush=True)
        start = time.time()
        while viewer.is_running() and time.time() - start < sim_duration:
            loop_start = time.time()

            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        target_world[:] = map(float, parts)
                        print(
                            f"Updated target: {target_world}\nSet goal (x, y, z): ",
                            end="",
                            flush=True,
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numbers.\nSet goal (x, y, z): ", end="", flush=True)

            # Match Isaac legged_robot.step: policy @ 50Hz, then decimation physics substeps.
            if use_policy and counter % decimation == 0:
                one_step_obs = build_one_step_obs(
                    d, trunk_id, default_angles, target_world, target_use_z,
                    ang_vel_scale, dof_pos_scale, dof_vel_scale, action,
                )
                if num_one_step_obs is not None and num_obs == num_one_step_obs * num_actor_history:
                    obs[:-num_one_step_obs] = obs[num_one_step_obs:]
                    obs[-num_one_step_obs:] = one_step_obs
                else:
                    obs[: len(one_step_obs)] = one_step_obs

                raw_action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                raw_action = np.clip(raw_action, -100.0, 100.0)
                smoothed_action = (
                    smoothed_action * action_smooth
                    + raw_action * (1.0 - action_smooth)
                )
                action = smoothed_action
                desired_target = action * action_scale + default_angles
                filtered_target_dof_pos = (
                    filtered_target_dof_pos * target_smooth
                    + desired_target * (1.0 - target_smooth)
                )
                target_dof_pos = filtered_target_dof_pos

            for _ in range(decimation):
                tau = compute_torques(target_dof_pos, d.qpos[7:], d.qvel[6:], kps, kds)
                d.ctrl[:] = np.clip(tau, -torque_limits, torque_limits)
                mujoco.mj_step(m, d)
                counter += 1
                if counter >= warmup_steps:
                    use_policy = True

            viewer.sync()
            time_until_next = decimation * m.opt.timestep - (time.time() - loop_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()
