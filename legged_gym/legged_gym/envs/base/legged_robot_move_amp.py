from webbrowser import get
import torch
from isaacgym import gymtorch
from legged_gym.utils.math import quat_rotate_inverse
from legged_gym.envs.base.legged_robot import LeggedRobot


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


class LeggedRobotMoveAmp(LeggedRobot):
    """Clean AMP locomotion task.

    This class starts as a thin wrapper around the original goalkeeper env.
    We will remove ball-specific logic step by step.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.use_ball_actor = getattr(cfg.env, "use_ball_actor", True)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


    def _target_xy_distance(self):
        return torch.norm((self.target_pos - self.torso_pos)[:, :2], dim=-1)

    def _uses_target_z(self):
        return getattr(self.cfg.commands, "target_use_z", True)

    def _update_target_z(self):
        if self._uses_target_z():
            return
        self.target_pos[:, 2] = self.torso_pos[:, 2]

    def _target_local_obs(self):
        """Target offset in upper-body frame (matches velocity / reward frames)."""
        base_quat = self.rigid_body_states[:, self.upper_body_index, 3:7]
        target_local = quat_rotate_inverse(base_quat, self.target_pos - self.torso_pos)
        if not self._uses_target_z():
            target_local = target_local.clone()
            target_local[:, 2] = 0.0
        return target_local

    def compute_observations(self):
        target_local = self._target_local_obs()
        current_obs = torch.cat((
            target_local,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_lin_vel * self.obs_scales.lin_vel,
        ),dim=-1)
        current_actor_obs = current_obs[:, :self.num_one_step_obs]
        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[:(self.num_one_step_obs)]
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1)
        self.privileged_obs_buf = current_obs

    def compute_termination_observations(self, env_ids):#TODO: 这个函数是用于计算终止观测值
        target_local = self._target_local_obs()
        current_obs = torch.cat((
            target_local,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_lin_vel * self.obs_scales.lin_vel,
        ), dim=-1)
        return current_obs[env_ids]

    #TODO: 这些函数是用于计算跟踪目标奖励
    def _reward_tracking_target(self):
        target_error = self._target_xy_distance() if not self._uses_target_z() else torch.norm(
            self.target_pos - self.torso_pos, dim=-1
        )
        return torch.exp(-target_error / self.cfg.rewards.target_sigma)
    def _reward_move_to_target(self):
        target_vec = self.target_pos - self.torso_pos
        target_dist = torch.norm(target_vec[:, :2], dim=-1, keepdim=True).clamp(min=1e-6)
        target_dir = target_vec[:, :2] / target_dist
        # World-frame torso velocity dotted with world-frame direction to target.
        lin_vel_world = self.rigid_body_states[:, self.torso_index, 7:10]
        velocity_to_target = torch.sum(lin_vel_world[:, :2] * target_dir, dim=-1)
        return torch.clamp(velocity_to_target, min=0.0, max=1.0)
    def _reward_upright(self):
        gravity_xy_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        return torch.exp(-3.0 * gravity_xy_error)

    #TODO: 这个函数是用于初始化目标位置
    def _init_buffers(self):
        super()._init_buffers()
        self.target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self._resample_targets(torch.arange(self.num_envs, device=self.device))

        self.episode_speed_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_fall = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _resample_targets(self, env_ids):
        if len(env_ids) == 0:
            return

        cfg = self.cfg.commands
        num_targets = len(env_ids)

        target_local = torch.zeros(num_targets, 3, dtype=torch.float, device=self.device)
        use_y = getattr(cfg, "target_use_y", True)
        use_z = getattr(cfg, "target_use_z", True)

        if getattr(cfg, "fix_target", False):
            target_local[:, 0] = getattr(cfg, "fixed_target_x", 1.0)
        else:
            target_local[:, 0] = torch.empty(num_targets, device=self.device).uniform_(cfg.target_x[0], cfg.target_x[1])

        if use_y and not getattr(cfg, "fix_target", False):
            target_local[:, 1] = torch.empty(num_targets, device=self.device).uniform_(cfg.target_y[0], cfg.target_y[1])
        else:
            target_local[:, 1] = 0.0

        if use_z and not getattr(cfg, "fix_target", False):
            target_local[:, 2] = torch.empty(num_targets, device=self.device).uniform_(cfg.target_z[0], cfg.target_z[1])

        self.target_pos[env_ids] = self.env_origins[env_ids] + target_local
        if use_z:
            self.target_pos[env_ids, 2] = target_local[:, 2]
        else:
            self.target_pos[env_ids, 2] = self.torso_pos[env_ids, 2]

    #TODO: 这个函数是用于重置目标位置
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        episode_lengths = torch.clip(self.episode_length_buf[env_ids].float(), min=1.0)
        if self._uses_target_z():
            final_target_dist = torch.norm(self.target_pos[env_ids] - self.torso_pos[env_ids], dim=-1)
        else:
            final_target_dist = torch.norm(
                (self.target_pos[env_ids] - self.torso_pos[env_ids])[:, :2], dim=-1
            )

        super().reset_idx(env_ids)

        if "episode" not in self.extras:
            self.extras["episode"] = {}

        self.extras["episode"]["target_dist"] = torch.mean(final_target_dist)
        self.extras["episode"]["success_rate"] = torch.mean(self.episode_success[env_ids])
        self.extras["episode"]["mean_speed"] = torch.mean(self.episode_speed_sum[env_ids] / episode_lengths)
        self.extras["episode"]["fall_rate"] = torch.mean(self.episode_fall[env_ids])

        self.episode_speed_sum[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
        self.episode_fall[env_ids] = 0.0

        self._resample_targets(env_ids)

    #TODO: 这个函数是用于获取噪声尺度向量
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise

        noise_scales = self.cfg.noise.noise_scales
        # print("CFG CLASS:", self.cfg.__class__)
        # print("NOISE SCALES:", self.cfg.noise.noise_scales.__dict__)

        noise_level = self.cfg.noise.noise_level

        start = 0

        # target_local
        noise_vec[start:start + 3] = noise_scales.target * noise_level
        start += 3

        # base_ang_vel
        noise_vec[start:start + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        start += 3

        # projected_gravity
        noise_vec[start:start + 3] = noise_scales.gravity * noise_level
        start += 3

        # dof_pos
        noise_vec[start:start + self.num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        start += self.num_dof

        # dof_vel
        noise_vec[start:start + self.num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        start += self.num_dof

        # previous actions: no noise
        noise_vec[start:start + self.num_actions] = 0.0

        return noise_vec

#TODO: 这个函数是用于执行物理步骤，计算奖励，检查终止条件，重置环境
    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self.base_lin_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.upper_body_index, 3:7],
            self.rigid_body_states[:, self.upper_body_index, 7:10],
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.upper_body_index, 3:7],
            self.rigid_body_states[:, self.upper_body_index, 10:13],
        )

        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        self._update_target_z()
        self.projected_gravity[:] = quat_rotate_inverse(
            self.rigid_body_states[:, self.upper_body_index, 3:7],
            self.gravity_vec,
        )
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        target_dist = self._target_xy_distance() if not self._uses_target_z() else torch.norm(
            self.target_pos - self.torso_pos, dim=-1
        )
        lin_vel_world = self.rigid_body_states[:, self.torso_index, 7:10]
        self.episode_speed_sum += torch.norm(lin_vel_world[:, :2], dim=-1)
        self.episode_success = torch.maximum(
            self.episode_success,
            (target_dist < self.cfg.rewards.target_reach_threshold).float(),
        )

        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)

        self._post_physics_step_callback()

        self.compute_reward()
        self.check_termination()

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

#TODO: 随机扰动实验
    def _post_physics_step_callback(self):
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval_s == 0
        ):
            self._push_robots()

#TODO: 这个函数是用于推动机器人
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

#TODO: 这个函数是用于计算力矩，将策略输出转换为力矩输出
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


#TODO: 1. 重置机器人状态
# 2. 设置初始位置
# 3. 给随机初速度
# 4. 隐藏球
# 5. 更新 simulator
# 6. 重新采样任务目标
    def _reset_root_states(self, env_ids):
        """Reset robot root states and sample a new target for each reset env."""
        if len(env_ids) == 0:
            return

        # Reset robot base pose around each parallel env origin.
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # Small random initial velocity helps the policy learn recovery,
        # but keep it mild so early training is not dominated by falls.
        self.root_states[env_ids, 7:13] = torch.empty(
            len(env_ids), 6, device=self.device
        ).uniform_(-0.3, 0.3)

        # The ball actor still exists during the transition phase, so keep its
        # state tensor valid but move it out of the way and make it static.
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

        # In goal-conditioned RL, changing the goal on reset defines the task
        # distribution the policy learns to solve.
        self._resample_targets(env_ids)


#TODO: 这个函数是用于重置DOF状态
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


    def check_termination(self):
        knee_height_buf = torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim=-1).values < 0.10
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.gravity_termination_buf = torch.norm(self.projected_gravity[:, 0:2], dim=-1) > 0.8
        sharpforce_buf = (
            torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim=-1)
            > 1.5 * self.cfg.rewards.max_contact_force
        )

        self.fall_buf = knee_height_buf | self.gravity_termination_buf | sharpforce_buf
        self.reset_buf = self.time_out_buf | self.fall_buf
        self.episode_fall = torch.maximum(self.episode_fall, self.fall_buf.float())


