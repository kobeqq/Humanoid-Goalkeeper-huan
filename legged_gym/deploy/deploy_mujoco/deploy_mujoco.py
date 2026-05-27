import argparse
import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.deploy_mujoco.sim2sim import min_foot_height, spawn_robot
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def compute_torques(joint_pos_target, dof_pos, dof_vel, p_gains, d_gains):
    """Position PD, same as legged_robot_move_amp._compute_torques (control_type='P')."""
    return p_gains * (joint_pos_target - dof_pos) - d_gains * dof_vel


def apply_ground_friction(model, friction):
    if friction is None:
        return
    mu = float(friction[0])
    mu_t = float(friction[1]) if len(friction) > 1 else mu
    for i in range(model.ngeom):
        if model.geom(i).name == "ground":
            model.geom_condim[i] = 3
            model.geom_friction[i, 0] = mu
            model.geom_friction[i, 1] = mu_t
            break


def quat_rotate_inverse_xyzw(q, v):
    # q: [x, y, z, w]
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def mujoco_quat_to_xyzw(quat_wxyz):
    # MuJoCo quaternion is [w, x, y, z]
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)


def setup_viewer_camera(viewer, model, cam_cfg):
    """Set passive-viewer camera for watching the simulation."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]  #
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        pd_gain_scale = float(config.get("pd_gain_scale", 1.0))
        kps = np.array(config["kps"], dtype=np.float32) * pd_gain_scale
        kds = np.array(config["kds"], dtype=np.float32) * pd_gain_scale
        warmup_steps = int(config.get("warmup_steps", 0))

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_one_step_obs = config.get("num_one_step_obs")
        num_actor_history = config.get("num_actor_history", 10) #
        # For k1_move_amp fix_target, this is a world-frame target point.
        target_init = np.array(config.get("target_init", [0.0, 0.0, 0.8]), dtype=np.float32)
        target_use_z = config.get("target_use_z", False)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    use_policy = warmup_steps == 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    if "mujoco_iterations" in config:
        m.opt.iterations = int(config["mujoco_iterations"])
    apply_ground_friction(m, config.get("ground_friction"))
    trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
    if "torque_limits" in config:
        torque_limits = np.array(config["torque_limits"], dtype=np.float32)
    else:
        torque_limits = m.actuator_forcerange[:, 1].astype(np.float32)
    spawn_joints = (
        np.array(config["init_joint_pos"], dtype=np.float64)
        if "init_joint_pos" in config
        else default_angles.astype(np.float64)
    )
    spawn_mode, foot_ids = spawn_robot(m, d, spawn_joints, config)
    print(
        f"[INFO] spawn_mode={spawn_mode}: trunk z={d.xpos[trunk_id, 2]:.3f}, "
        f"foot z={min_foot_height(m, d, foot_ids):.3f}"
    )

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        setup_viewer_camera(viewer, m, config.get("viewer_camera"))
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            loop_start = time.time()

            if use_policy and counter % control_decimation == 0:
                trunk_pos = d.xpos[trunk_id].astype(np.float32)
                quat_xyzw = mujoco_quat_to_xyzw(d.qpos[3:7])
                omega = quat_rotate_inverse_xyzw(quat_xyzw, d.qvel[3:6].astype(np.float32)) * ang_vel_scale

                qj = (d.qpos[7:] - default_angles) * dof_pos_scale
                dqj = d.qvel[6:] * dof_vel_scale
                gravity_orientation = quat_rotate_inverse_xyzw(
                    quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32)
                )
                target_world = target_init.copy()
                if not target_use_z:
                    target_world[2] = trunk_pos[2]
                target_local = quat_rotate_inverse_xyzw(quat_xyzw, target_world - trunk_pos)
                if not target_use_z:
                    target_local[2] = 0.0

                one_step_obs = np.concatenate(
                    (target_local, omega, gravity_orientation, qj, dqj, action)
                ).astype(np.float32)

                if num_one_step_obs is not None and num_obs == num_one_step_obs * num_actor_history:
                    obs[:-num_one_step_obs] = obs[num_one_step_obs:]
                    obs[-num_one_step_obs:] = one_step_obs
                else:
                    obs[: len(one_step_obs)] = one_step_obs

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            for _ in range(control_decimation):
                tau = compute_torques(target_dof_pos, d.qpos[7:], d.qvel[6:], kps, kds)
                d.ctrl[:] = np.clip(tau, -torque_limits, torque_limits)
                mujoco.mj_step(m, d)
                counter += 1
                if counter >= warmup_steps:
                    use_policy = True

            viewer.sync()
            time_until_next_step = control_decimation * m.opt.timestep - (time.time() - loop_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
