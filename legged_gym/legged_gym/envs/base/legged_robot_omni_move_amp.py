import math

import torch
from isaacgym import gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import quat_rotate_inverse


DIR_FRONT = 0
DIR_LEFT = 1
DIR_BACK = 2
DIR_RIGHT = 3


def euler_from_quaternion(quat_angle):
    """Convert a quaternion into roll, pitch, yaw."""
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


class LeggedRobotOmniMoveAmp(LeggedRobot):
    """AMP locomotion task with balanced omnidirectional goals and yaw limits."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.use_ball_actor = getattr(cfg.env, "use_ball_actor", False)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ---------------------------------------------------------------------
    # Goal and curriculum helpers
    # ---------------------------------------------------------------------

    def _curriculum_stage_idx(self):
        if not hasattr(self, "curriculum_stage"):
            return 0
        return int(self.curriculum_stage.item())

    def _stage_value(self, name):
        values = getattr(self.cfg.commands, name)
        stage = min(self._curriculum_stage_idx(), len(values) - 1)
        return values[stage]

    def _current_reach_threshold(self):
        if hasattr(self.cfg.commands, "stage_reach_thresholds"):
            return float(self._stage_value("stage_reach_thresholds"))
        return float(getattr(self.cfg.rewards, "target_reach_threshold", 0.35))

    def _current_yaw_limit(self):
        if hasattr(self.cfg.commands, "stage_yaw_limits"):
            return float(self._stage_value("stage_yaw_limits"))
        return float(getattr(self.cfg.commands, "yaw_limit", 0.35))

    def _current_yaw_success_limit(self):
        if hasattr(self.cfg.commands, "stage_yaw_success_limits"):
            return float(self._stage_value("stage_yaw_success_limits"))
        return self._current_yaw_limit()

    def _current_radius_range(self):
        if hasattr(self.cfg.commands, "stage_radius_ranges"):
            return list(self._stage_value("stage_radius_ranges"))
        return list(getattr(self.cfg.commands, "target_radius", [0.8, 2.0]))

    def _current_allowed_bins(self):
        if hasattr(self.cfg.commands, "stage_allowed_bins"):
            return list(self._stage_value("stage_allowed_bins"))
        return [DIR_FRONT, DIR_LEFT, DIR_BACK, DIR_RIGHT]

    def _current_bin_width(self):
        if hasattr(self.cfg.commands, "stage_bin_widths"):
            return float(self._stage_value("stage_bin_widths"))
        return math.pi / 2.0

    def _current_yaw(self):
        _, _, yaw = euler_from_quaternion(self.root_states[:, 3:7])
        return yaw

    def _yaw_error(self):
        yaw = self.yaw if hasattr(self, "yaw") else self._current_yaw()
        yaw_ref = self.yaw_ref if hasattr(self, "yaw_ref") else torch.zeros_like(yaw)
        return wrap_to_pi(yaw - yaw_ref)

    def _yaw_obs(self):
        yaw_error = self._yaw_error()
        return torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)

    def _uses_target_z(self):
        return bool(getattr(self.cfg.commands, "target_use_z", True))

    def _target_xy_distance(self):
        return torch.norm((self.target_pos - self.torso_pos)[:, :2], dim=-1)

    def _target_distance(self):
        if self._uses_target_z():
            return torch.norm(self.target_pos - self.torso_pos, dim=-1)
        return self._target_xy_distance()

    def _target_base_height(self, env_ids):
        if hasattr(self, "torso_pos"):
            return self.torso_pos[env_ids, 2]
        return self.root_states[env_ids, 2]

    def _update_target_z(self):
        if not self._uses_target_z():
            self.target_pos[:, 2] = self.torso_pos[:, 2]

    def _target_local_obs(self):
        """Target offset in upper-body frame (matches velocity / reward frames)."""
        base_quat = self.rigid_body_states[:, self.upper_body_index, 3:7]
        target_local = quat_rotate_inverse(base_quat, self.target_pos - self.torso_pos)
        if not self._uses_target_z():
            target_local = target_local.clone()
            target_local[:, 2] = 0.0
        return target_local

    def _sample_balanced_headings(self, num_targets):
        allowed = torch.tensor(self._current_allowed_bins(), dtype=torch.long, device=self.device)
        choice = torch.randint(0, len(allowed), (num_targets,), device=self.device)
        dir_bins = allowed[choice]
        centers = torch.tensor(
            [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0],
            dtype=torch.float,
            device=self.device,
        )
        jitter = torch.empty(num_targets, dtype=torch.float, device=self.device).uniform_(
            -0.5 * self._current_bin_width(),
            0.5 * self._current_bin_width(),
        )
        headings = wrap_to_pi(centers[dir_bins] + jitter)
        return headings, dir_bins

    def _resample_targets(self, env_ids):
        if len(env_ids) == 0 or not hasattr(self, "target_pos"):
            return

        cfg = self.cfg.commands
        num_targets = len(env_ids)
        radius_range = self._current_radius_range()
        radius = torch.empty(num_targets, dtype=torch.float, device=self.device).uniform_(
            float(radius_range[0]),
            float(radius_range[1]),
        )
        heading_ref, dir_bins = self._sample_balanced_headings(num_targets)
        self.target_heading[env_ids] = heading_ref
        self.target_dir_bin[env_ids] = dir_bins

        if hasattr(self, "yaw_ref"):
            heading_world = wrap_to_pi(heading_ref + self.yaw_ref[env_ids])
        else:
            heading_world = heading_ref

        target_offset = torch.zeros(num_targets, 3, dtype=torch.float, device=self.device)
        target_offset[:, 0] = radius * torch.cos(heading_world)
        target_offset[:, 1] = radius * torch.sin(heading_world)
        self.target_pos[env_ids] = self.env_origins[env_ids] + target_offset

        if self._uses_target_z():
            target_z = torch.empty(num_targets, dtype=torch.float, device=self.device).uniform_(
                float(cfg.target_z[0]),
                float(cfg.target_z[1]),
            )
            self.target_pos[env_ids, 2] = target_z
        else:
            self.target_pos[env_ids, 2] = self._target_base_height(env_ids)

    # ---------------------------------------------------------------------
    # Observations and rewards
    # ---------------------------------------------------------------------

    def _build_current_obs(self):
        return torch.cat(
            (
                self._target_local_obs(),
                self._yaw_obs(),
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.base_lin_vel * self.obs_scales.lin_vel,
            ),
            dim=-1,
        )

    def compute_observations(self):
        current_obs = self._build_current_obs()
        current_actor_obs = current_obs[:, : self.num_one_step_obs]
        if self.add_noise:
            current_actor_obs = current_actor_obs + (
                2 * torch.rand_like(current_actor_obs) - 1
            ) * self.noise_scale_vec[: self.num_one_step_obs]
        self.obs_buf = torch.cat(
            (self.obs_buf[:, self.num_one_step_obs : self.actor_obs_length], current_actor_obs),
            dim=-1,
        )
        self.privileged_obs_buf = current_obs

    def compute_termination_observations(self, env_ids):
        return self._build_current_obs()[env_ids]

    def _reward_tracking_target(self):
        return torch.exp(-self._target_distance() / self.cfg.rewards.target_sigma)

    def _reward_move_to_target(self):
        target_vec = self.target_pos - self.torso_pos
        target_dist = torch.norm(target_vec[:, :2], dim=-1, keepdim=True).clamp(min=1e-6)
        target_dir = target_vec[:, :2] / target_dist
        lin_vel_world = self.rigid_body_states[:, self.torso_index, 7:10]
        velocity_to_target = torch.sum(lin_vel_world[:, :2] * target_dir, dim=-1)
        return torch.clamp(velocity_to_target, min=0.0, max=1.0)

    def _reward_body_velocity_tracking(self):
        target_local = self._target_local_obs()[:, :2]
        dist = torch.norm(target_local, dim=-1, keepdim=True)
        target_dir_body = target_local / dist.clamp(min=1e-6)
        desired_speed = float(getattr(self.cfg.commands, "desired_speed", 0.55))
        slow_radius = float(getattr(self.cfg.commands, "slow_radius", 0.7))
        speed_scale = torch.clamp(dist / max(slow_radius, 1e-6), 0.0, 1.0)
        desired_vel_body = target_dir_body * desired_speed * speed_scale
        vel_error = self.base_lin_vel[:, :2] - desired_vel_body
        sigma = float(getattr(self.cfg.rewards, "body_vel_sigma", 0.25))
        return torch.exp(-torch.sum(torch.square(vel_error), dim=-1) / max(sigma, 1e-6))

    def _reward_success_bonus(self):
        target_dist = self._target_distance()
        yaw_ok = torch.abs(self._yaw_error()) < self._current_yaw_success_limit()
        success = target_dist < self._current_reach_threshold()
        return (success & yaw_ok).float()

    def _reward_yaw_tracking(self):
        yaw_error = self._yaw_error()
        sigma = float(getattr(self.cfg.rewards, "yaw_sigma", 0.25))
        return torch.exp(-torch.square(yaw_error / max(sigma, 1e-6)))

    def _reward_yaw_rate_z(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_upright(self):
        gravity_xy_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        return torch.exp(-3.0 * gravity_xy_error)

    # ---------------------------------------------------------------------
    # Buffers, reset, and curriculum stats
    # ---------------------------------------------------------------------

    def _init_buffers(self):
        super()._init_buffers()
        self.dir_names = list(getattr(self.cfg.commands, "direction_names", ["front", "left", "back", "right"]))
        self.curriculum_stage = torch.zeros((), dtype=torch.long, device=self.device)
        self.yaw_ref = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.target_heading = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.target_dir_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.episode_speed_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_fall = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_yaw_fail = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.dir_success_ema = torch.zeros(4, dtype=torch.float, device=self.device)
        self.dir_yaw_fail_ema = torch.zeros(4, dtype=torch.float, device=self.device)
        self.dir_episode_count = torch.zeros(4, dtype=torch.float, device=self.device)
        self.dir_ema_ready = torch.zeros(4, dtype=torch.bool, device=self.device)
        self._resample_targets(torch.arange(self.num_envs, device=self.device))

    def _update_direction_ema(self, bins, success, yaw_fail):
        alpha = float(getattr(self.cfg.commands, "curriculum_ema_alpha", 0.05))
        for direction_id in range(4):
            mask = bins == direction_id
            if not torch.any(mask):
                continue
            batch_success = torch.mean(success[mask].float())
            batch_yaw_fail = torch.mean(yaw_fail[mask].float())
            batch_count = torch.sum(mask.float())
            if self.dir_ema_ready[direction_id]:
                self.dir_success_ema[direction_id] = (
                    (1.0 - alpha) * self.dir_success_ema[direction_id] + alpha * batch_success
                )
                self.dir_yaw_fail_ema[direction_id] = (
                    (1.0 - alpha) * self.dir_yaw_fail_ema[direction_id] + alpha * batch_yaw_fail
                )
            else:
                self.dir_success_ema[direction_id] = batch_success
                self.dir_yaw_fail_ema[direction_id] = batch_yaw_fail
                self.dir_ema_ready[direction_id] = True
            self.dir_episode_count[direction_id] += batch_count

    def _maybe_update_curriculum(self):
        cfg = self.cfg.commands
        if not bool(getattr(cfg, "curriculum_enable", False)):
            return
        stage = self._curriculum_stage_idx()
        gates = getattr(cfg, "curriculum_success_gates", [])
        yaw_gates = getattr(cfg, "curriculum_yaw_fail_gates", [])
        if stage >= len(gates) or stage >= len(yaw_gates):
            return

        allowed = torch.tensor(self._current_allowed_bins(), dtype=torch.long, device=self.device)
        min_count = torch.min(self.dir_episode_count[allowed])
        min_required = float(getattr(cfg, "curriculum_min_episodes_per_bin", 1000))
        if float(min_count.item()) < min_required:
            return

        min_success = torch.min(self.dir_success_ema[allowed])
        max_yaw_fail = torch.max(self.dir_yaw_fail_ema[allowed])
        if float(min_success.item()) > float(gates[stage]) and float(max_yaw_fail.item()) < float(yaw_gates[stage]):
            self.curriculum_stage += 1
            self.dir_success_ema.zero_()
            self.dir_yaw_fail_ema.zero_()
            self.dir_episode_count.zero_()
            self.dir_ema_ready.zero_()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        episode_lengths = torch.clip(self.episode_length_buf[env_ids].float(), min=1.0)
        final_target_dist = self._target_distance()[env_ids].clone()
        final_success = self.episode_success[env_ids].clone()
        final_fall = self.episode_fall[env_ids].clone()
        final_yaw_fail = self.episode_yaw_fail[env_ids].clone()
        final_bins = self.target_dir_bin[env_ids].clone()
        final_speed_sum = self.episode_speed_sum[env_ids].clone()

        super().reset_idx(env_ids)
        self._update_direction_ema(final_bins, final_success, final_yaw_fail)

        if "episode" not in self.extras:
            self.extras["episode"] = {}

        self.extras["episode"]["target_dist"] = torch.mean(final_target_dist)
        self.extras["episode"]["success_rate"] = torch.mean(final_success)
        self.extras["episode"]["mean_speed"] = torch.mean(final_speed_sum / episode_lengths)
        self.extras["episode"]["fall_rate"] = torch.mean(final_fall)
        self.extras["episode"]["yaw_fail_rate"] = torch.mean(final_yaw_fail)
        self.extras["episode"]["curriculum_stage"] = self.curriculum_stage.float()

        allowed = torch.tensor(self._current_allowed_bins(), dtype=torch.long, device=self.device)
        self.extras["episode"]["min_dir_success_ema"] = torch.min(self.dir_success_ema[allowed])
        self.extras["episode"]["max_dir_yaw_fail_ema"] = torch.max(self.dir_yaw_fail_ema[allowed])

        for direction_id, name in enumerate(self.dir_names[:4]):
            mask = final_bins == direction_id
            count = torch.sum(mask.float())
            if torch.any(mask):
                self.extras["episode"][f"success_{name}"] = torch.mean(final_success[mask])
                self.extras["episode"][f"yaw_fail_{name}"] = torch.mean(final_yaw_fail[mask])
            else:
                self.extras["episode"][f"success_{name}"] = torch.zeros((), dtype=torch.float, device=self.device)
                self.extras["episode"][f"yaw_fail_{name}"] = torch.zeros((), dtype=torch.float, device=self.device)
            self.extras["episode"][f"count_{name}"] = count
            self.extras["episode"][f"ema_success_{name}"] = self.dir_success_ema[direction_id]
            self.extras["episode"][f"ema_yaw_fail_{name}"] = self.dir_yaw_fail_ema[direction_id]
            self.extras["episode"][f"episode_count_{name}"] = self.dir_episode_count[direction_id]

        self._maybe_update_curriculum()

        self.episode_speed_sum[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
        self.episode_fall[env_ids] = 0.0
        self.episode_yaw_fail[env_ids] = 0.0

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        start = 0
        noise_vec[start : start + 3] = noise_scales.target * noise_level
        start += 3
        noise_vec[start : start + 2] = getattr(noise_scales, "yaw", 0.0) * noise_level
        start += 2
        noise_vec[start : start + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        start += 3
        noise_vec[start : start + 3] = noise_scales.gravity * noise_level
        start += 3
        noise_vec[start : start + self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        start += self.num_dof
        noise_vec[start : start + self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        start += self.num_dof
        noise_vec[start : start + self.num_actions] = 0.0
        start += self.num_actions
        if start != self.num_one_step_obs:
            raise ValueError(f"Noise scale length {start} != num_one_step_obs {self.num_one_step_obs}")
        return noise_vec

    # ---------------------------------------------------------------------
    # Per-step simulation update and termination
    # ---------------------------------------------------------------------

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        upper_quat = self.rigid_body_states[:, self.upper_body_index, 3:7]
        self.base_lin_vel = quat_rotate_inverse(
            upper_quat,
            self.rigid_body_states[:, self.upper_body_index, 7:10],
        )
        self.base_ang_vel = quat_rotate_inverse(
            upper_quat,
            self.rigid_body_states[:, self.upper_body_index, 10:13],
        )

        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        self._update_target_z()
        self.projected_gravity[:] = quat_rotate_inverse(upper_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        target_dist = self._target_distance()
        lin_vel_world = self.rigid_body_states[:, self.torso_index, 7:10]
        self.episode_speed_sum += torch.norm(lin_vel_world[:, :2], dim=-1)

        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)

        self._post_physics_step_callback()
        self.compute_reward()
        self.check_termination()

        yaw_ok = torch.abs(self._yaw_error()) < self._current_yaw_success_limit()
        reach_ok = target_dist < self._current_reach_threshold()
        step_success = reach_ok & yaw_ok & (~self.fall_buf)
        self.episode_success = torch.maximum(self.episode_success, step_success.float())
        if bool(getattr(self.cfg.commands, "reset_on_success", True)):
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

    def _post_physics_step_callback(self):
        push_interval = int(getattr(self.cfg.domain_rand, "push_interval", 0))
        if self.cfg.domain_rand.push_robots and (
            push_interval > 0 and self.common_step_counter % push_interval == 0
        ):
            self._push_robots()

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch.empty(
            self.num_envs, 2, device=self.device
        ).uniform_(-max_vel, max_vel)

        if getattr(self, "use_ball_actor", True):
            all_states = torch.cat(
                (self.root_states.unsqueeze(1), self.ball_states.unsqueeze(1)),
                dim=1,
            ).view(-1, 13)
        else:
            all_states = self.root_states.contiguous()

        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(all_states),
        )

    def check_termination(self):
        knee_height_buf = torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim=-1).values < 0.10
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.gravity_termination_buf = torch.norm(self.projected_gravity[:, 0:2], dim=-1) > 0.8
        sharpforce_buf = (
            torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim=-1)
            > 1.5 * self.cfg.rewards.max_contact_force
        )
        self.yaw_limit_buf = torch.abs(self._yaw_error()) > self._current_yaw_limit()

        self.fall_buf = knee_height_buf | self.gravity_termination_buf | sharpforce_buf
        self.reset_buf = self.time_out_buf | self.fall_buf | self.yaw_limit_buf
        self.episode_fall = torch.maximum(self.episode_fall, self.fall_buf.float())
        self.episode_yaw_fail = torch.maximum(self.episode_yaw_fail, self.yaw_limit_buf.float())

    # ---------------------------------------------------------------------
    # Control and reset hooks
    # ---------------------------------------------------------------------

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_poses + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos)
                - self.d_gains * self.Kd_factors * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques + self.actuation_offset + self.joint_injection
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_root_states(self, env_ids):
        """Reset robot root states and sample a new target for each reset env."""
        if len(env_ids) == 0:
            return

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if hasattr(self, "yaw_ref"):
            if getattr(self.cfg.commands, "yaw_ref_world", True):
                self.yaw_ref[env_ids] = float(getattr(self.cfg.commands, "yaw_ref", 0.0))
            else:
                _, _, yaw0 = euler_from_quaternion(self.root_states[env_ids, 3:7])
                self.yaw_ref[env_ids] = yaw0

        self.root_states[env_ids, 7:13] = torch.empty(
            len(env_ids), 6, device=self.device
        ).uniform_(-0.3, 0.3)

        if getattr(self, "use_ball_actor", True):
            self.ball_states[env_ids] = self.base_init_state
            self.ball_states[env_ids, :3] = self.env_origins[env_ids]
            self.ball_states[env_ids, 2] = -10.0
            self.ball_states[env_ids, 7:13] = 0.0

            all_states = torch.cat(
                (self.root_states.unsqueeze(1), self.ball_states.unsqueeze(1)),
                dim=1,
            ).view(-1, 13)
            env_ids_int32 = torch.cat((2 * env_ids, 2 * env_ids + 1)).to(dtype=torch.int32)
        else:
            all_states = self.root_states.contiguous()
            env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(all_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self._resample_targets(env_ids)

    def _reset_dofs(self, env_ids):
        """Reset DOF states with actor ids matching one-actor or two-actor env layouts."""
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)

        if self.cfg.domain_rand.continue_keep and torch.rand(1).item() > 0.2:
            sampled_ids = torch.randint(
                0,
                self.num_envs,
                (len(env_ids),),
                device=self.dof_pos.device,
            )
            self.dof_pos[env_ids] = self.dof_pos[sampled_ids]
        else:
            if self.cfg.domain_rand.randomize_initial_joint_pos:
                init_dof_pos = self.standpos * torch.empty(
                    len(env_ids), self.num_dof, device=self.device
                ).uniform_(
                    self.cfg.domain_rand.initial_joint_pos_scale[0],
                    self.cfg.domain_rand.initial_joint_pos_scale[1],
                )
                init_dof_pos += torch.empty(
                    len(env_ids), self.num_dof, device=self.device
                ).uniform_(
                    self.cfg.domain_rand.initial_joint_pos_offset[0],
                    self.cfg.domain_rand.initial_joint_pos_offset[1],
                )
                self.dof_pos[env_ids] = torch.clip(init_dof_pos, dof_lower, dof_upper)
            else:
                self.dof_pos[env_ids] = self.standpos * torch.ones(
                    len(env_ids), self.num_dof, device=self.device
                )

        self.init_dof_pos[env_ids] = self.dof_pos[env_ids].clone()
        self.dof_vel[env_ids] = 0.0

        if getattr(self, "use_ball_actor", True):
            actor_ids = 2 * env_ids.clone().to(dtype=torch.int32)
        else:
            actor_ids = env_ids.clone().to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )
