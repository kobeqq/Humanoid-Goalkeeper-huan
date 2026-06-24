"""Sim2Sim: run Isaac Gym policies in MuJoCo.

Supports:
- K1 Move AMP (`LeggedRobotMoveAmp`, obs 75 x 10 = 750)
- K1 Loco AMP (`LeggedRobotK1LocoAmp`, obs 65 x 10 = 650)
- K1 Goalkeeper Foundation (`LeggedRobotGoalkeeperFoundation`, obs 86 x 10 = 860)
"""

import argparse
import glob
import math
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

PHASE_STABLE = 0
PHASE_EXECUTE = 1
PHASE_RECOVER = 2

DEFAULT_LEG_JOINT_INDICES = list(range(10, 22))


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


def is_foundation_task(cfg):
    return cfg.get("task") == "goalkeeper_foundation"


def is_loco_amp_task(cfg):
    return cfg.get("task") == "k1_loco_amp"


def get_trunk_body_state(d):
    """Trunk-frame lin/ang vel and projected gravity (matches K1 upper_body_link=Trunk)."""
    quat_xyzw = mujoco_quat_to_xyzw(d.qpos[3:7])
    lin_vel_world = d.qvel[0:3].astype(np.float32)
    ang_vel_world = d.qvel[3:6].astype(np.float32)
    lin_vel = quat_rotate_inverse_xyzw(quat_xyzw, lin_vel_world)
    ang_vel = quat_rotate_inverse_xyzw(quat_xyzw, ang_vel_world)
    gravity = quat_rotate_inverse_xyzw(
        quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )
    return quat_xyzw, lin_vel, ang_vel, gravity


def is_stable_standing(lin_vel, ang_vel, gravity, foot_ids, d, foot_ground_z=0.0):
    lin_ok = np.linalg.norm(lin_vel[:2]) < 0.15
    ang_ok = np.linalg.norm(ang_vel[:2]) < 0.5
    grav_ok = np.linalg.norm(gravity[:2]) < 0.3
    foot_heights = [d.xpos[i, 2] - foot_ground_z for i in foot_ids]
    both_feet = len(foot_heights) >= 2 and all(z < 0.06 for z in foot_heights)
    return lin_ok and ang_ok and grav_ok and both_feet


def mask_upper_body_action(action, leg_indices):
    masked = action.copy()
    upper = np.ones(len(action), dtype=bool)
    upper[leg_indices] = False
    masked[upper] = 0.0
    return masked


def apply_leg_only_targets(action, default_angles, action_scale, leg_indices):
    """Match LeggedRobotGoalkeeperFoundation._compute_torques (legs only)."""
    target = default_angles.copy()
    target[leg_indices] = action[leg_indices] * action_scale + default_angles[leg_indices]
    return target


def apply_12d_leg_targets(action, default_angles, action_scale, leg_indices):
    target = default_angles.copy()
    if len(action) != len(leg_indices):
        raise RuntimeError(
            f"Expected 12-d leg action, got action={len(action)} leg_indices={len(leg_indices)}"
        )
    target[leg_indices] = default_angles[leg_indices] + action * action_scale
    return target


def build_loco_amp_one_step_obs(
    d,
    default_angles,
    action,
    command,
    command_scale,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
):
    quat_xyzw = mujoco_quat_to_xyzw(d.qpos[3:7])
    base_angvel_world = d.qvel[3:6].astype(np.float32)
    omega = quat_rotate_inverse_xyzw(quat_xyzw, base_angvel_world) * ang_vel_scale
    gravity = quat_rotate_inverse_xyzw(
        quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )
    qj = (d.qpos[7:].astype(np.float32) - default_angles) * dof_pos_scale
    dqj = d.qvel[6:].astype(np.float32) * dof_vel_scale
    cmd_obs = np.asarray(command, dtype=np.float32) * np.asarray(command_scale, dtype=np.float32)
    return np.concatenate((cmd_obs, omega, gravity, qj, dqj, action)).astype(np.float32)


def build_foundation_one_step_obs(
    d,
    default_angles,
    action,
    lin_vel_scale,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
    goal_z_scale,
    cmd_vx,
    cmd_vy,
    goal_z_world,
    standing_goal_z,
    time_left,
    progress_phase,
    gait_phase,
    episode_phase,
):
    """Match LeggedRobotGoalkeeperFoundation._build_one_step_obs (86 dims)."""
    _, lin_vel, ang_vel, gravity = get_trunk_body_state(d)
    lin_vel = lin_vel * lin_vel_scale
    ang_vel = ang_vel * ang_vel_scale

    qj = (d.qpos[7:] - default_angles) * dof_pos_scale
    dqj = d.qvel[6:] * dof_vel_scale

    cmd_xy = (
        np.array([cmd_vx, cmd_vy], dtype=np.float32)
        if episode_phase == PHASE_EXECUTE
        else np.zeros(2, dtype=np.float32)
    )
    goal_z_obs = np.array([(goal_z_world - standing_goal_z) * goal_z_scale], dtype=np.float32)
    time_left_obs = np.array([time_left], dtype=np.float32)

    progress_angle = 2.0 * math.pi * progress_phase
    gait_angle = 2.0 * math.pi * gait_phase
    phase_obs = np.array(
        [
            math.sin(progress_angle),
            math.cos(progress_angle),
            math.sin(gait_angle),
            math.cos(gait_angle),
        ],
        dtype=np.float32,
    )
    phase_onehot = np.zeros(3, dtype=np.float32)
    phase_onehot[episode_phase] = 1.0

    return np.concatenate(
        (
            lin_vel,
            ang_vel,
            gravity,
            qj,
            dqj,
            action,
            cmd_xy,
            goal_z_obs,
            time_left_obs,
            phase_obs,
            phase_onehot,
        )
    ).astype(np.float32)


class FoundationEpisodeFSM:
    """Mirrors LeggedRobotGoalkeeperFoundation episode FSM for sim2sim."""

    def __init__(self, cfg, control_dt):
        self.control_dt = control_dt
        self.stable_hold_s = float(cfg.get("stable_hold_s", 0.5))
        self.T_recover_default = float(cfg.get("T_recover", 2.0))
        self.T_execute_min = float(cfg.get("T_execute_min", 1.0))
        self.gait_period = float(cfg.get("gait_period", 0.8))
        self.episode_length_s = float(cfg.get("episode_length_s", 10.0))
        self.min_cycle_time_s = float(cfg.get("min_cycle_time_s", 3.5))
        self.auto_cycle = bool(cfg.get("auto_cycle", True))
        cmd_init = cfg.get("command_init", {})
        self.default_vx = float(cmd_init.get("vx", 0.6))
        self.default_vy = float(cmd_init.get("vy", 0.0))
        self.default_goal_z = cmd_init.get("goal_z", None)
        self.default_t_execute = float(cmd_init.get("T_execute", 4.0))
        self.reset(standing_goal_z=0.8)

    def reset(self, standing_goal_z):
        self.phase = PHASE_STABLE
        self.standing_goal_z = float(standing_goal_z)
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.goal_z = self.standing_goal_z
        self.T_execute = 1.0
        self.T_recover_buf = 1.0
        self.command_time_left = 0.0
        self.stable_timer = 0.0
        self.recover_timer = 0.0
        self.progress_phase = 0.0
        self.gait_time = 0.0
        self.episode_time = 0.0

    def episode_time_remaining(self):
        return max(0.0, self.episode_length_s - self.episode_time)

    def _clamp_execute_duration(self, t_execute):
        remaining = self.episode_time_remaining()
        max_t = max(self.T_execute_min, remaining - self.T_recover_default)
        return float(np.clip(t_execute, self.T_execute_min, max_t))

    def _allocate_recover_duration(self):
        remaining = self.episode_time_remaining()
        return float(np.clip(min(remaining, self.T_recover_default), self.stable_hold_s, remaining))

    def _can_start_execute(self):
        return self.episode_time_remaining() >= self.min_cycle_time_s

    def begin_execute(self, vx, vy, goal_z, t_execute):
        self.phase = PHASE_EXECUTE
        self.cmd_vx = float(vx)
        self.cmd_vy = float(vy)
        self.goal_z = float(goal_z)
        self.T_execute = self._clamp_execute_duration(t_execute)
        self.command_time_left = self.T_execute
        self.progress_phase = 0.0
        self.stable_timer = 0.0
        self.recover_timer = 0.0

    def _enter_recover(self):
        self.phase = PHASE_RECOVER
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.goal_z = self.standing_goal_z
        self.T_recover_buf = self._allocate_recover_duration()
        self.command_time_left = self.T_recover_buf
        self.progress_phase = 0.0
        self.recover_timer = 0.0

    def _start_execute_from_defaults(self):
        goal_z = self.standing_goal_z if self.default_goal_z is None else float(self.default_goal_z)
        self.begin_execute(self.default_vx, self.default_vy, goal_z, self.default_t_execute)

    def time_left_norm(self):
        if self.phase == PHASE_STABLE:
            return float(np.clip(self.stable_timer / max(self.stable_hold_s, 1e-6), 0.0, 1.0))
        if self.phase == PHASE_EXECUTE:
            return float(
                np.clip(self.command_time_left / max(self.T_execute, 1e-6), 0.0, 1.0)
            )
        return float(
            np.clip(self.command_time_left / max(self.T_recover_buf, 1e-6), 0.0, 1.0)
        )

    def gait_phase(self):
        if self.gait_period <= 0.0:
            return 0.0
        return (self.gait_time % self.gait_period) / self.gait_period

    def update(self, stable):
        dt = self.control_dt
        self.episode_time += dt
        self.gait_time += dt

        if self.episode_time >= self.episode_length_s:
            self.reset(self.standing_goal_z)
            return

        if self.phase == PHASE_STABLE:
            self.stable_timer += dt
            if self.auto_cycle and stable and self.stable_timer >= self.stable_hold_s:
                if self._can_start_execute():
                    self._start_execute_from_defaults()

        elif self.phase == PHASE_EXECUTE:
            self.command_time_left -= dt
            self.progress_phase = 1.0 - float(
                np.clip(self.command_time_left / max(self.T_execute, 1e-6), 0.0, 1.0)
            )
            if self.command_time_left <= 0.0:
                self._enter_recover()

        elif self.phase == PHASE_RECOVER:
            self.command_time_left -= dt
            self.recover_timer += dt
            if stable and self.recover_timer >= self.stable_hold_s:
                self.phase = PHASE_STABLE
                self.stable_timer = 0.0
                self.command_time_left = 0.0
                self.progress_phase = 0.0


def push_obs_history(obs, one_step_obs, num_one_step_obs, num_actor_history, num_obs):
    if num_one_step_obs is not None and num_obs == num_one_step_obs * num_actor_history:
        obs[:-num_one_step_obs] = obs[num_one_step_obs:]
        obs[-num_one_step_obs:] = one_step_obs
    else:
        obs[: len(one_step_obs)] = one_step_obs
    return obs


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
        description="Sim2Sim MuJoCo deployment (K1 Move AMP / Goalkeeper Foundation)."
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
        help="Move AMP: world-frame goal point (default from yaml target_init)",
    )
    parser.add_argument(
        "--cmd",
        type=float,
        nargs=4,
        metavar=("VX", "VY", "GOAL_Z", "T_EXEC"),
        default=None,
        help="Foundation: inject execute command (vx, vy, goal_z, T_execute seconds)",
    )
    parser.add_argument(
        "--loco_cmd",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        default=None,
        help="K1 loco AMP command: body-frame vx vy wz",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    foundation = is_foundation_task(cfg)
    loco_amp = is_loco_amp_task(cfg)

    policy_path = resolve_policy_path(cfg, args.policy_path)
    xml_path = cfg["xml_path"]
    sim_duration = args.sim_duration if args.sim_duration is not None else cfg["simulation_duration"]
    sim_dt = args.dt if args.dt is not None else cfg["simulation_dt"]
    decimation = args.decimation if args.decimation is not None else cfg["control_decimation"]
    action_scale = args.action_scale if args.action_scale is not None else cfg["action_scale"]
    control_dt = sim_dt * decimation

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
    lin_vel_scale = cfg.get("lin_vel_scale", 2.0)
    goal_z_scale = cfg.get("goal_z_scale", 1.0)
    foot_ground_z = float(cfg.get("foot_ground_z", 0.0))

    num_actions = cfg["num_actions"]
    num_obs = cfg["num_obs"]
    num_one_step_obs = cfg.get("num_one_step_obs")
    num_actor_history = cfg.get("num_actor_history", 10)

    leg_indices = np.array(cfg.get("leg_joint_indices", DEFAULT_LEG_JOINT_INDICES), dtype=np.int64)

    target_world = np.array(
        args.target if args.target is not None else cfg.get("target_init", [5.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    target_use_z = cfg.get("target_use_z", False)
    loco_command = np.array(
        args.loco_cmd if args.loco_cmd is not None else cfg.get("command_init", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    command_scale = np.array(cfg.get("command_scale", [1.0, 1.0, 1.0]), dtype=np.float32)

    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Task: {'goalkeeper_foundation' if foundation else 'k1_loco_amp' if loco_amp else 'move_amp'}")
    print(f"[INFO] Model: {xml_path}")
    print(f"[INFO] Policy: {policy_path}")
    print(f"[INFO] Obs: {num_one_step_obs} x {num_actor_history} = {num_obs}")
    if foundation:
        print(f"[INFO] Leg-only control indices: {leg_indices.tolist()}")
        print(f"[INFO] FSM auto_cycle={cfg.get('auto_cycle', True)}, episode={cfg.get('episode_length_s', 10.0)}s")
    elif loco_amp:
        print(f"[INFO] Loco command body-frame: {loco_command}")
        print(f"[INFO] 12-d leg action indices: {leg_indices.tolist()}")
    else:
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
    standing_goal_z = float(d.xpos[trunk_id, 2])
    print(
        f"[INFO] spawn_mode={spawn_mode}: trunk z={standing_goal_z:.3f}, "
        f"foot z={foot_z:.3f} (isaac play uses z=0.8 with feet on ground in PhysX)"
    )

    fsm = FoundationEpisodeFSM(cfg, control_dt) if foundation else None
    if fsm is not None:
        fsm.reset(standing_goal_z)
        if args.cmd is not None:
            vx, vy, gz, t_exec = args.cmd
            goal_z = standing_goal_z if gz < 0 else float(gz)
            fsm.begin_execute(vx, vy, goal_z, t_exec)

    action = np.zeros(num_actions, dtype=np.float32)
    obs_action = np.zeros(num_actions, dtype=np.float32)
    smoothed_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    filtered_target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    use_policy = warmup_steps == 0

    stdin_prompt = (
        "Set command (vx vy goal_z T_execute; goal_z<0 = standing height): "
        if foundation
        else "Set loco command vx vy wz: "
        if loco_amp
        else "Set goal (x, y, z) in world frame: "
    )

    with mujoco.viewer.launch_passive(m, d) as viewer:
        setup_viewer_camera(viewer, m, cfg.get("viewer_camera"))
        print(stdin_prompt, end="", flush=True)
        start = time.time()
        while viewer.is_running() and time.time() - start < sim_duration:
            loop_start = time.time()

            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if foundation:
                        if len(parts) == 4:
                            vx, vy, gz, t_exec = map(float, parts)
                            goal_z = fsm.standing_goal_z if gz < 0 else gz
                            fsm.begin_execute(vx, vy, goal_z, t_exec)
                            print(
                                f"Execute: vx={vx}, vy={vy}, goal_z={goal_z:.3f}, T={t_exec}s\n"
                                f"{stdin_prompt}",
                                end="",
                                flush=True,
                            )
                        else:
                            raise ValueError
                    elif loco_amp and len(parts) == 3:
                        loco_command[:] = map(float, parts)
                        print(
                            f"Updated loco command: {loco_command}\n{stdin_prompt}",
                            end="",
                            flush=True,
                        )
                    elif len(parts) == 3:
                        target_world[:] = map(float, parts)
                        print(
                            f"Updated target: {target_world}\n{stdin_prompt}",
                            end="",
                            flush=True,
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print(f"Invalid input.\n{stdin_prompt}", end="", flush=True)

            if use_policy and counter % decimation == 0:
                if foundation:
                    obs_action = mask_upper_body_action(action, leg_indices)
                    one_step_obs = build_foundation_one_step_obs(
                        d,
                        default_angles,
                        obs_action,
                        lin_vel_scale,
                        ang_vel_scale,
                        dof_pos_scale,
                        dof_vel_scale,
                        goal_z_scale,
                        fsm.cmd_vx,
                        fsm.cmd_vy,
                        fsm.goal_z,
                        fsm.standing_goal_z,
                        fsm.time_left_norm(),
                        fsm.progress_phase,
                        fsm.gait_phase(),
                        fsm.phase,
                    )
                elif loco_amp:
                    one_step_obs = build_loco_amp_one_step_obs(
                        d,
                        default_angles,
                        action,
                        loco_command,
                        command_scale,
                        ang_vel_scale,
                        dof_pos_scale,
                        dof_vel_scale,
                    )
                else:
                    one_step_obs = build_one_step_obs(
                        d,
                        trunk_id,
                        default_angles,
                        target_world,
                        target_use_z,
                        ang_vel_scale,
                        dof_pos_scale,
                        dof_vel_scale,
                        action,
                    )

                obs = push_obs_history(
                    obs, one_step_obs, num_one_step_obs, num_actor_history, num_obs
                )

                raw_action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                raw_action = np.clip(raw_action, -100.0, 100.0)
                smoothed_action = (
                    smoothed_action * action_smooth + raw_action * (1.0 - action_smooth)
                )
                action = smoothed_action
                if foundation:
                    desired_target = apply_leg_only_targets(
                        action, default_angles, action_scale, leg_indices
                    )
                elif loco_amp:
                    desired_target = apply_12d_leg_targets(
                        action, default_angles, action_scale, leg_indices
                    )
                else:
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

            if foundation and use_policy and counter % decimation == 0:
                _, lin_vel, ang_vel, gravity = get_trunk_body_state(d)
                stable = is_stable_standing(
                    lin_vel, ang_vel, gravity, foot_ids, d, foot_ground_z
                )
                fsm.update(stable)

            viewer.sync()
            time_until_next = decimation * m.opt.timestep - (time.time() - loop_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()
