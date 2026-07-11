"""2D point-goal locomotion task with lower-body actions and AMP.

这个任务是从 k1_move_amp 拆出来的全新任务：
- 目标点在水平面内用极坐标采样；
- 策略只输出 12 维下肢动作，上肢保持默认姿态；
- yaw 只作为软约束和独立失败统计，不污染 fall_rate；
- AMP 先对齐当前 MotionLib 的两步 32 维 lower-body state expert obs。
"""

import math

import torch
from isaacgym import gymtorch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.g1.g1_utils import MotionLib, load_imitation_dataset, build_lower_body_amp_step_obs
from legged_gym.utils.math import quat_rotate_inverse


def euler_from_quaternion(quat_angle):
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


def wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class _StaticLowerBodyMotion:
    """当 K1 motion 目录为空时的最小 fallback，只用于保证 AMP shape 可跑通。"""

    def __init__(self, default_lower_pose, obs_dim_per_step, num_steps, device):
        self.device = device
        gravity = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 3)
        step_obs = build_lower_body_amp_step_obs(
            default_lower_pose.view(1, -1),
            torch.zeros(1, default_lower_pose.numel(), device=device),
            torch.zeros(1, 3, device=device),
            torch.zeros(1, 3, device=device),
            gravity,
        )
        self.obs = step_obs.repeat(1, num_steps).view(1, -1)
        if self.obs.shape[-1] != obs_dim_per_step * num_steps:
            raise RuntimeError(
                f"static AMP fallback dim mismatch: {self.obs.shape[-1]} vs {obs_dim_per_step * num_steps}"
            )

    def get_expert_obs(self, batch_size):
        return self.obs.repeat(batch_size, 1)


class LeggedRobotMoveAmp2D(LeggedRobot):
    """K1 lower-body 2D point-goal locomotion task."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.use_ball_actor = getattr(cfg.env, "use_ball_actor", False)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Init / AMP
    # ------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()
        self._init_lower_body_action_mapping()
        self._reinit_amp_motions_for_lower_body()
        self._init_path_task_buffers()
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_targets(all_env_ids)
        self._update_path_quantities()

    def _init_lower_body_action_mapping(self):
        # 只让策略控制左右腿 12 个关节；其余关节由 PD 拉回默认姿态。
        lower_names = list(self.cfg.control.left_leg_joints) + list(self.cfg.control.right_leg_joints)
        missing = [name for name in lower_names if name not in self.dof_names]
        if missing:
            raise ValueError(f"Lower-body joints are missing from asset DOFs: {missing}")

        lower_ids = [self.dof_names.index(name) for name in lower_names]
        self.lower_body_dof_names = lower_names
        self.lower_body_dof_indices = torch.tensor(lower_ids, dtype=torch.long, device=self.device)
        upper_ids = [idx for idx in range(self.num_dof) if idx not in set(lower_ids)]
        self.upper_body_dof_indices = torch.tensor(upper_ids, dtype=torch.long, device=self.device)

    def _reinit_amp_motions_for_lower_body(self):
        # lower_body_state：单步 32 维；runner 拼接连续两步 agent obs -> 64 维，与 MotionLib num_steps=2 对齐。
        self.amp_lower_dof_names = self.lower_body_dof_names
        self.amp_lower_dof_indices = self.lower_body_dof_indices
        self.amp_obs_per_step = int(getattr(self.cfg.amp, "num_obs_per_step", 32))
        multidataset, mapping = load_imitation_dataset(
            self.cfg.dataset.folder.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            self.cfg.dataset.joint_mapping.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
        )
        num_steps = int(getattr(self.cfg.amp, "num_steps", 2))
        amp_obs_type = getattr(self.cfg.amp, "obs_type", "lower_body_state")
        self.motions = {}
        if not multidataset:
            print(
                "[k1_move_amp_2d] WARNING: no .pt motions found in dataset folder; "
                "using static lower-body fallback AMP expert obs."
            )
            default_lower_pose = self.default_dof_poses[0, self.amp_lower_dof_indices].detach().clone()
            self.motions["static_stand"] = _StaticLowerBodyMotion(
                default_lower_pose, self.amp_obs_per_step, num_steps, self.device
            )
            self.ref_motion_buffers = []
            return
        for key, dataset in multidataset.items():
            self.motions[key] = MotionLib(
                dataset,
                mapping,
                self.amp_lower_dof_names,
                self.keyframe_names,
                fps=self.cfg.dataset.frame_rate,
                min_dt=self.cfg.dataset.min_time,
                device=self.device,
                amp_obs_type=amp_obs_type,
                num_steps=num_steps,
                include_dof_vel=True,
            )
        self._init_ref_motion_sampler()

    def _init_ref_motion_sampler(self):
        weights = getattr(self.cfg.dataset, "motion_weights", {}) or {}
        self.ref_motion_buffers = []
        self.ref_motion_probs = []
        for name, buffer in self.motions.items():
            weight = self._motion_weight(name, weights)
            if weight <= 0.0:
                continue
            self.ref_motion_buffers.append(buffer)
            self.ref_motion_probs.append(weight)
        if not self.ref_motion_buffers:
            self.ref_motion_buffers = list(self.motions.values())
            self.ref_motion_probs = [1.0 for _ in self.ref_motion_buffers]
        probs = torch.tensor(self.ref_motion_probs, dtype=torch.float, device=self.device)
        self.ref_motion_probs = probs / probs.sum().clamp(min=1e-6)

    @staticmethod
    def _motion_weight(name, weights):
        lowered = name.lower()
        if name in weights:
            return float(weights[name])
        if lowered in weights:
            return float(weights[lowered])
        return float(weights.get("__default__", weights.get("default", 1.0)))

    def _sample_ref_motion_state(self, count):
        if not getattr(self, "ref_motion_buffers", None):
            return None
        motion_ids = torch.multinomial(self.ref_motion_probs, count, replacement=True)
        dof_pos = torch.zeros(count, len(self.amp_lower_dof_indices), device=self.device)
        dof_vel = torch.zeros_like(dof_pos)
        base_z = torch.zeros(count, device=self.device)
        for idx, buffer in enumerate(self.ref_motion_buffers):
            mask = motion_ids == idx
            n = int(mask.sum().item())
            if n == 0:
                continue
            random_time = getattr(self.cfg.init_state, "state_init", "Default") == "Random"
            ref = buffer.sample_reference_state(n, random_time=random_time)
            dof_pos[mask] = ref["dof_pos"]
            dof_vel[mask] = ref["dof_vel"]
            base_z[mask] = ref["base_z"]
        return {"dof_pos": dof_pos, "dof_vel": dof_vel, "base_z": base_z}

    def _ref_init_mask(self, env_ids):
        mode = getattr(self.cfg.init_state, "state_init", "Default")
        count = len(env_ids)
        if count == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        if mode == "Default":
            return torch.zeros(count, dtype=torch.bool, device=self.device)
        if mode in ("Start", "Random"):
            return torch.ones(count, dtype=torch.bool, device=self.device)
        if mode == "Hybrid":
            prob = float(getattr(self.cfg.init_state, "ref_init_prob", 0.3))
            return torch.rand(count, device=self.device) < prob
        return torch.zeros(count, dtype=torch.bool, device=self.device)

    def _init_path_task_buffers(self):
        n = self.num_envs
        device = self.device
        self.target_pos = torch.zeros(n, 3, dtype=torch.float, device=device)
        self.path_start_xy = torch.zeros(n, 2, dtype=torch.float, device=device)
        self.path_dir_xy = torch.zeros(n, 2, dtype=torch.float, device=device)
        self.path_normal_xy = torch.zeros(n, 2, dtype=torch.float, device=device)
        self.path_length = torch.ones(n, dtype=torch.float, device=device)
        self.prev_target_dist = torch.zeros(n, dtype=torch.float, device=device)
        self.reset_yaw = torch.zeros(n, dtype=torch.float, device=device)
        self.yaw_error = torch.zeros(n, dtype=torch.float, device=device)
        self.cross_track_error = torch.zeros(n, dtype=torch.float, device=device)
        self.remain_dist = torch.zeros(n, dtype=torch.float, device=device)
        self.target_vec_body_xy = torch.zeros(n, 2, dtype=torch.float, device=device)
        self.v_ref_world_xy = torch.zeros(n, 2, dtype=torch.float, device=device)
        self.v_ref_body_xy = torch.zeros(n, 2, dtype=torch.float, device=device)
        self.target_base_z = self.torso_pos[:, 2].clone()

        self.episode_speed_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_success = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_fall = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_yaw_fail = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_cross_track_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_progress_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_velocity_error_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_path_steps = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_yaw_error_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_amp_reward_sum = torch.zeros(n, dtype=torch.float, device=device)

    # ------------------------------------------------------------------
    # Target sampling / path metrics
    # ------------------------------------------------------------------

    def _current_xy(self):
        return self.torso_pos[:, :2] if hasattr(self, "torso_pos") else self.root_states[:, :2]

    def _sample_range(self, value_range, shape):
        lo = float(value_range[0])
        hi = float(value_range[1])
        if abs(hi - lo) < 1e-6:
            return torch.full(shape, lo, dtype=torch.float, device=self.device)
        return torch.empty(*shape, dtype=torch.float, device=self.device).uniform_(lo, hi)

    def _resample_targets(self, env_ids):
        if len(env_ids) == 0:
            return

        # 极坐标采样目标点：以 reset 后 torso xy 为圆心，避免 x/y 独立均匀采样带来的分布偏置。
        # 路径由 path_start_xy -> target_xy 定义，后续 reward/obs 都围绕这条直线展开。
        cfg = self.cfg.commands
        count = len(env_ids)
        start_xy = self._current_xy()[env_ids]
        _, _, yaw0 = euler_from_quaternion(self.root_states[env_ids, 3:7])

        radius = self._sample_range(getattr(cfg, "target_radius", [0.8, 2.0]), (count,))
        angle_range = getattr(cfg, "target_angle", [-math.pi, math.pi])
        theta = self._sample_range(angle_range, (count,))
        offset = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)

        path_length = torch.norm(offset, dim=-1).clamp(min=1e-4)
        path_dir = offset / path_length.unsqueeze(-1)
        path_normal = torch.stack((-path_dir[:, 1], path_dir[:, 0]), dim=-1)

        self.path_start_xy[env_ids] = start_xy
        self.path_dir_xy[env_ids] = path_dir
        self.path_normal_xy[env_ids] = path_normal
        self.path_length[env_ids] = path_length
        self.reset_yaw[env_ids] = yaw0

        self.target_pos[env_ids, :2] = start_xy + offset
        if getattr(cfg, "target_use_z", False):
            self.target_pos[env_ids, 2] = self._sample_range(getattr(cfg, "target_z", [0.5, 1.1]), (count,))
        else:
            self.target_pos[env_ids, 2] = self.torso_pos[env_ids, 2]
        self.target_base_z[env_ids] = self.torso_pos[env_ids, 2]

        self.prev_target_dist[env_ids] = torch.norm(
            self.target_pos[env_ids, :2] - self.torso_pos[env_ids, :2],
            dim=-1,
        )

    def _rotate_world_xy_to_body(self, world_xy, yaw):
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        body_x = c * world_xy[:, 0] + s * world_xy[:, 1]
        body_y = -s * world_xy[:, 0] + c * world_xy[:, 1]
        return torch.stack((body_x, body_y), dim=-1)

    def _clip_xy_norm(self, xy, max_norm):
        norm = torch.norm(xy, dim=-1, keepdim=True)
        scale = torch.clamp(float(max_norm) / norm.clamp(min=1e-6), max=1.0)
        return xy * scale

    def _update_path_quantities(self):
        # 路径条件量：沿路径进度 s、剩余距离 remain、横向偏移 cross_track。
        # 不直接把 target_local_xy 给 actor，是因为机器人可通过大幅转 yaw 把任意目标变成“正前方”，产生多解。
        # v_ref 沿 path_dir 前进，并用 path_normal 拉回 cross_track；接近目标时 tanh 减速，避免冲过终点。
        xy = self.torso_pos[:, :2]
        rel = xy - self.path_start_xy
        progress_s = torch.sum(rel * self.path_dir_xy, dim=-1)
        cross = torch.sum(rel * self.path_normal_xy, dim=-1)
        remain = torch.clamp(self.path_length - progress_s, min=0.0)

        target_vec_world = self.target_pos[:, :2] - xy
        self.target_vec_body_xy = self._rotate_world_xy_to_body(target_vec_world, self.yaw)
        self.cross_track_error = cross
        self.remain_dist = remain
        self.yaw_error = wrap_to_pi(self.yaw - self.reset_yaw)

        cfg = self.cfg.commands
        v_max = float(getattr(cfg, "v_max", 0.65))
        v_min = float(getattr(cfg, "v_min", 0.0))
        slow_down_dist = max(float(getattr(cfg, "slow_down_dist", 0.7)), 1e-4)
        k_cross = float(getattr(cfg, "k_cross", 0.8))
        v_des = v_max * torch.tanh(remain / slow_down_dist)
        v_des = torch.where(remain > 0.05, torch.clamp(v_des, min=v_min), torch.zeros_like(v_des))
        v_ref = v_des.unsqueeze(-1) * self.path_dir_xy - k_cross * cross.unsqueeze(-1) * self.path_normal_xy
        self.v_ref_world_xy = self._clip_xy_norm(v_ref, v_max)
        self.v_ref_body_xy = self._rotate_world_xy_to_body(self.v_ref_world_xy, self.yaw)

    def _target_xy_distance(self):
        return torch.norm(self.target_pos[:, :2] - self.torso_pos[:, :2], dim=-1)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _path_actor_obs(self):
        cfg = self.cfg.commands
        max_radius = max(float(getattr(cfg, "target_radius", [0.8, 2.0])[1]), 1e-4)
        v_max = max(float(getattr(cfg, "v_max", 0.65)), 1e-4)
        cross_scale = max(float(getattr(cfg, "cross_track_obs_scale", 1.0)), 1e-4)
        target_xy = self.target_vec_body_xy / max_radius
        v_ref_body = self.v_ref_body_xy / v_max
        remain = (self.remain_dist / max_radius).unsqueeze(-1)
        cross = torch.clamp(self.cross_track_error / cross_scale, -2.0, 2.0).unsqueeze(-1)
        yaw = self.yaw_error.unsqueeze(-1)
        yaw_rate_ref = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        return torch.cat((target_xy, v_ref_body, remain, cross, yaw, yaw_rate_ref), dim=-1)

    def _build_actor_one_step_obs(self):
        return torch.cat(
            (
                self._path_actor_obs(),
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

    def _build_privileged_obs(self, actor_obs):
        max_radius = max(float(getattr(self.cfg.commands, "target_radius", [0.8, 2.0])[1]), 1e-4)
        v_max = max(float(getattr(self.cfg.commands, "v_max", 0.65)), 1e-4)
        target_world = (self.target_pos[:, :2] - self.torso_pos[:, :2]) / max_radius
        return torch.cat(
            (
                actor_obs,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.v_ref_world_xy / v_max,
                target_world,
                self.cross_track_error.unsqueeze(-1),
                self.yaw_error.unsqueeze(-1),
            ),
            dim=-1,
        )

    def compute_observations(self):
        actor_obs = self._build_actor_one_step_obs()
        if actor_obs.shape[1] != self.num_one_step_obs:
            raise RuntimeError(
                f"actor obs dim mismatch: got {actor_obs.shape[1]}, expected {self.num_one_step_obs}"
            )
        current_actor_obs = actor_obs
        if self.add_noise:
            current_actor_obs = current_actor_obs + (
                2.0 * torch.rand_like(current_actor_obs) - 1.0
            ) * self.noise_scale_vec[: self.num_one_step_obs]
        self.obs_buf = torch.cat(
            (self.obs_buf[:, self.num_one_step_obs : self.actor_obs_length], current_actor_obs),
            dim=-1,
        )
        self.privileged_obs_buf = self._build_privileged_obs(actor_obs)

    def compute_termination_observations(self, env_ids):
        actor_obs = self._build_actor_one_step_obs()
        return self._build_privileged_obs(actor_obs)[env_ids]

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        start = 0
        path_noise = getattr(noise_scales, "path", getattr(noise_scales, "target", 0.05)) * noise_level
        noise_vec[start : start + 2] = path_noise
        start += 2
        noise_vec[start : start + 2] = getattr(noise_scales, "vel_ref", 0.02) * noise_level
        start += 2
        noise_vec[start : start + 1] = path_noise
        start += 1
        noise_vec[start : start + 1] = path_noise
        start += 1
        noise_vec[start : start + 1] = getattr(noise_scales, "yaw", 0.01) * noise_level
        start += 1
        noise_vec[start : start + 1] = 0.0
        start += 1
        noise_vec[start : start + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        start += 3
        noise_vec[start : start + 3] = noise_scales.gravity * noise_level
        start += 3
        noise_vec[start : start + self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        start += self.num_dof
        noise_vec[start : start + self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        start += self.num_dof
        noise_vec[start : start + self.num_actions] = 0.0
        return noise_vec

    # ------------------------------------------------------------------
    # Control / reset
    # ------------------------------------------------------------------

    def get_amp_observations(self):
        # 单步 32 维 lower-body AMP 状态；runner 会把连续两步拼成 64 维送入判别器。
        q_leg = self.dof_pos[:, self.amp_lower_dof_indices]
        dq_leg = self.dof_vel[:, self.amp_lower_dof_indices]
        return build_lower_body_amp_step_obs(
            q_leg, dq_leg, self.base_lin_vel, self.base_ang_vel, self.projected_gravity
        )

    def record_amp_reward(self, amp_reward, active_mask=None):
        if active_mask is None:
            self.episode_amp_reward_sum += amp_reward.detach()
        else:
            self.episode_amp_reward_sum += amp_reward.detach() * active_mask.float()

    def _compute_torques(self, actions):
        # policy 只输出 12 维下肢动作；上肢/头部保持默认姿态，由 PD 维持。
        # 不能把 12 维 action 直接加到 22 维 default pose 上——DOF 顺序与维度都不匹配。
        # 最终仍输出 22 维 torque，因为 Isaac Gym 底层 actuation 覆盖全部 DOF。
        if actions.shape[1] != len(self.lower_body_dof_indices):
            raise RuntimeError(
                f"2D move AMP expects {len(self.lower_body_dof_indices)} actions, got {actions.shape[1]}"
            )

        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_poses.clone()
        self.joint_pos_target[:, self.lower_body_dof_indices] = (
            self.default_dof_poses[:, self.lower_body_dof_indices] + actions_scaled
        )
        if len(self.upper_body_dof_indices) > 0:
            self.joint_pos_target[:, self.upper_body_dof_indices] = self.default_dof_poses[:, self.upper_body_dof_indices]

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos)
                - self.d_gains * self.Kd_factors * self.dof_vel
            )
        else:
            raise NameError("LeggedRobotMoveAmp2D currently supports P control only.")

        torques = torques + self.actuation_offset + self.joint_injection
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        ref_mask = self._ref_init_mask(env_ids)
        ref_state = self._sample_ref_motion_state(int(ref_mask.sum().item())) if ref_mask.any() else None

        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dof_pos = self.standpos * torch.empty(len(env_ids), self.num_dof, device=self.device).uniform_(
                self.cfg.domain_rand.initial_joint_pos_scale[0],
                self.cfg.domain_rand.initial_joint_pos_scale[1],
            )
            init_dof_pos += torch.empty(len(env_ids), self.num_dof, device=self.device).uniform_(
                self.cfg.domain_rand.initial_joint_pos_offset[0],
                self.cfg.domain_rand.initial_joint_pos_offset[1],
            )
            self.dof_pos[env_ids] = torch.clip(init_dof_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.standpos * torch.ones(len(env_ids), self.num_dof, device=self.device)

        self.dof_vel[env_ids] = 0.0

        # 参考动作初始化：只覆盖下肢 q/dq；上肢保持默认姿态且速度为 0。
        # base 高度/yaw 仍用默认站姿，不直接使用 motion 中的大 yaw 或 z（与仿真坐标系标定有关）。
        if ref_state is not None:
            ref_ids = env_ids[ref_mask]
            ref_rows = ref_ids[:, None]
            lower_cols = self.lower_body_dof_indices[None, :]
            upper_cols = self.upper_body_dof_indices[None, :]
            self.dof_pos[ref_rows, lower_cols] = ref_state["dof_pos"]
            self.dof_vel[ref_rows, lower_cols] = ref_state["dof_vel"]
            self.dof_pos[ref_rows, upper_cols] = self.default_dof_poses[ref_rows, upper_cols]
            self.dof_vel[ref_rows, upper_cols] = 0.0

        self.init_dof_pos[env_ids] = self.dof_pos[env_ids].clone()
        if getattr(self, "use_ball_actor", False):
            actor_ids = (2 * env_ids).to(dtype=torch.int32)
        else:
            actor_ids = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        vel_range = float(getattr(self.cfg.init_state, "root_vel_init_range", 0.15))
        self.root_states[env_ids, 7:13] = torch.empty(len(env_ids), 6, device=self.device).uniform_(
            -vel_range,
            vel_range,
        )

        if getattr(self, "use_ball_actor", False):
            self.ball_states[env_ids] = self.base_init_state
            self.ball_states[env_ids, :3] = self.env_origins[env_ids]
            self.ball_states[env_ids, 2] = -10.0
            self.ball_states[env_ids, 7:13] = 0.0
            all_states = torch.cat((self.root_states.unsqueeze(1), self.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
            actor_ids = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        else:
            all_states = self.root_states.contiguous()
            actor_ids = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(all_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        if hasattr(self, "target_pos"):
            self._resample_targets(env_ids)

    def _post_physics_step_callback(self):
        # push_interval 在 _parse_cfg 中已从秒换算为 control step 数，不能直接用 push_interval_s 取模。
        interval = int(getattr(self.cfg.domain_rand, "push_interval", 0))
        if self.cfg.domain_rand.push_robots and interval > 0 and self.common_step_counter % interval == 0:
            self._push_robots()

    def _push_robots(self):
        max_vel = float(getattr(self.cfg.domain_rand, "max_push_vel_xy", 0.0))
        if max_vel <= 0.0:
            return
        self.root_states[:, 7:9] = torch.empty(self.num_envs, 2, device=self.device).uniform_(-max_vel, max_vel)
        if getattr(self, "use_ball_actor", False):
            all_states = torch.cat((self.root_states.unsqueeze(1), self.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
        else:
            all_states = self.root_states.contiguous()
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_states))

    # ------------------------------------------------------------------
    # Main loop / termination
    # ------------------------------------------------------------------

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        upper_quat = self.rigid_body_states[:, self.upper_body_index, 3:7]
        self.base_lin_vel = quat_rotate_inverse(upper_quat, self.rigid_body_states[:, self.upper_body_index, 7:10])
        self.base_ang_vel = quat_rotate_inverse(upper_quat, self.rigid_body_states[:, self.upper_body_index, 10:13])
        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        if not getattr(self.cfg.commands, "target_use_z", False):
            self.target_pos[:, 2] = self.torso_pos[:, 2]
        self.projected_gravity[:] = quat_rotate_inverse(upper_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        self._update_path_quantities()

        target_dist = self._target_xy_distance()
        lin_vel_world = self.rigid_body_states[:, self.torso_index, 7:10]
        progress = torch.clamp(self.prev_target_dist - target_dist, min=0.0)
        vel_err = torch.norm(lin_vel_world[:, :2] - self.v_ref_world_xy, dim=-1)
        self.episode_speed_sum += torch.norm(lin_vel_world[:, :2], dim=-1)
        self.episode_cross_track_sum += torch.abs(self.cross_track_error)
        self.episode_progress_sum += progress
        self.episode_velocity_error_sum += vel_err
        self.episode_yaw_error_sum += torch.abs(self.yaw_error)
        self.episode_path_steps += 1.0

        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)

        self._post_physics_step_callback()
        self.compute_reward()
        self.prev_target_dist[:] = target_dist
        self.check_termination()

        goal_ok = target_dist < float(getattr(self.cfg.rewards, "target_reach_threshold", 0.35))
        yaw_ok = torch.abs(self.yaw_error) < float(getattr(self.cfg.rewards, "yaw_success_limit", 0.35))
        step_success = goal_ok & yaw_ok & (~self.fall_buf)
        self.episode_success = torch.maximum(self.episode_success, step_success.float())
        if getattr(self.cfg.commands, "reset_on_success", True):
            self.reset_buf = self.reset_buf | step_success

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        cfg = self.cfg.rewards
        knee_height = torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim=-1).values
        knee_height_buf = knee_height < float(getattr(cfg, "termination_knee_height", 0.10))
        base_height_buf = self.torso_pos[:, 2] < float(getattr(cfg, "termination_base_height", 0.20))
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.gravity_termination_buf = torch.norm(self.projected_gravity[:, 0:2], dim=-1) > float(
            getattr(cfg, "termination_gravity_xy", 0.8)
        )
        sharpforce_buf = (
            torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim=-1)
            > float(getattr(cfg, "termination_contact_force_scale", 1.5)) * self.cfg.rewards.max_contact_force
        )
        # yaw 软约束在 reward 中处理；超过 hard limit 才 reset，避免早期探索被 20° 硬截断。
        self.yaw_limit_buf = torch.abs(self.yaw_error) > float(getattr(cfg, "yaw_limit", 0.61))

        self.fall_buf = knee_height_buf | base_height_buf | self.gravity_termination_buf | sharpforce_buf
        self.reset_buf = self.time_out_buf | self.fall_buf | self.yaw_limit_buf
        self.episode_fall = torch.maximum(self.episode_fall, self.fall_buf.float())
        self.episode_yaw_fail = torch.maximum(self.episode_yaw_fail, self.yaw_limit_buf.float())

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        episode_lengths = torch.clip(self.episode_length_buf[env_ids].float(), min=1.0)
        final_target_dist = self._target_xy_distance()[env_ids].clone()
        final_success = self.episode_success[env_ids].clone()
        final_fall = self.episode_fall[env_ids].clone()
        final_yaw_fail = self.episode_yaw_fail[env_ids].clone()
        final_speed_sum = self.episode_speed_sum[env_ids].clone()
        path_steps = torch.clip(self.episode_path_steps[env_ids].clone(), min=1.0)
        final_cross = self.episode_cross_track_sum[env_ids].clone() / path_steps
        final_progress = self.episode_progress_sum[env_ids].clone()
        final_vel_err = self.episode_velocity_error_sum[env_ids].clone() / path_steps
        final_yaw_err = self.episode_yaw_error_sum[env_ids].clone() / path_steps
        final_path_length = self.path_length[env_ids].clone()
        final_amp_reward = self.episode_amp_reward_sum[env_ids].clone() / episode_lengths

        super().reset_idx(env_ids)

        if "episode" not in self.extras:
            self.extras["episode"] = {}
        self.extras["episode"]["target_dist"] = torch.mean(final_target_dist)
        self.extras["episode"]["success_rate"] = torch.mean(final_success)
        self.extras["episode"]["fall_rate"] = torch.mean(final_fall)
        self.extras["episode"]["yaw_fail_rate"] = torch.mean(final_yaw_fail)
        self.extras["episode"]["mean_speed"] = torch.mean(final_speed_sum / episode_lengths)
        self.extras["episode"]["mean_cross_track"] = torch.mean(final_cross)
        self.extras["episode"]["mean_progress"] = torch.mean(final_progress)
        self.extras["episode"]["mean_velocity_error"] = torch.mean(final_vel_err)
        self.extras["episode"]["mean_yaw_error"] = torch.mean(final_yaw_err)
        self.extras["episode"]["path_length"] = torch.mean(final_path_length)
        self.extras["episode"]["amp_reward_mean"] = torch.mean(final_amp_reward)

        self.episode_speed_sum[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
        self.episode_fall[env_ids] = 0.0
        self.episode_yaw_fail[env_ids] = 0.0
        self.episode_cross_track_sum[env_ids] = 0.0
        self.episode_progress_sum[env_ids] = 0.0
        self.episode_velocity_error_sum[env_ids] = 0.0
        self.episode_path_steps[env_ids] = 0.0
        self.episode_yaw_error_sum[env_ids] = 0.0
        self.episode_amp_reward_sum[env_ids] = 0.0
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self._update_path_quantities()

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _reward_progress(self):
        target_dist = self._target_xy_distance()
        progress = torch.clamp(self.prev_target_dist - target_dist, min=0.0)
        denom = max(float(getattr(self.cfg.commands, "v_max", 0.65)) * self.dt, 1e-4)
        return torch.clamp(progress / denom, max=2.0)

    def _reward_velocity_track(self):
        lin_vel_world = self.rigid_body_states[:, self.torso_index, 7:10]
        error = torch.sum(torch.square(lin_vel_world[:, :2] - self.v_ref_world_xy), dim=-1)
        sigma = max(float(getattr(self.cfg.rewards, "velocity_sigma", 0.25)), 1e-6)
        return torch.exp(-error / sigma)

    def _reward_cross_track(self):
        sigma = max(float(getattr(self.cfg.rewards, "cross_track_sigma", 0.35)), 1e-6)
        return torch.exp(-torch.square(self.cross_track_error) / sigma)

    def _reward_goal(self):
        reach = self._target_xy_distance() < float(getattr(self.cfg.rewards, "target_reach_threshold", 0.35))
        yaw_ok = torch.abs(self.yaw_error) < float(getattr(self.cfg.rewards, "yaw_success_limit", 0.35))
        return (reach & yaw_ok).float()

    def _reward_yaw(self):
        sigma = max(float(getattr(self.cfg.rewards, "yaw_sigma", 0.25)), 1e-6)
        return torch.exp(-torch.square(self.yaw_error) / sigma)

    def _reward_yaw_violation(self):
        # 超过 soft limit 后二次惩罚；scale 在 config 中为负，鼓励保持 reset yaw 附近。
        soft_limit = float(getattr(self.cfg.rewards, "yaw_soft_limit", math.radians(20.0)))
        return torch.square(torch.clamp(torch.abs(self.yaw_error) - soft_limit, min=0.0))

    def _reward_upright(self):
        gravity_xy_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        return torch.exp(-3.0 * gravity_xy_error)

    def _reward_height(self):
        sigma = max(float(getattr(self.cfg.rewards, "height_sigma", 0.04)), 1e-6)
        return torch.exp(-torch.square(self.torso_pos[:, 2] - self.target_base_z) / sigma)

    def _reward_feet_slip(self):
        if len(self.contact_feet_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        foot_vel = self.rigid_body_states[:, self.contact_feet_indices, 7:10]
        contact = self.contact_forces[:, self.contact_feet_indices, 2] > 1.0
        return torch.sum(torch.square(torch.norm(foot_vel[:, :, :2], dim=-1)) * contact.float(), dim=1)
