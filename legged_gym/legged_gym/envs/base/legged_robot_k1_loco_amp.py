"""K1 command-conditioned lower-body locomotion AMP task."""

import math

import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_move_amp_2d import LeggedRobotMoveAmp2D, euler_from_quaternion, wrap_to_pi
from legged_gym.envs.g1.g1_utils import MotionLib, build_locomotion_amp_step_obs, load_imitation_dataset
from legged_gym.utils.math import quat_rotate_inverse


class _MirroredLocomotionMotion:
    """Mirror a left-diagonal lower-body motion into a right-diagonal motion."""

    def __init__(self, source, obs_dim_per_step):
        self.source = source
        self.obs_dim_per_step = obs_dim_per_step
        if obs_dim_per_step != 46:
            raise ValueError(f"mirrored locomotion AMP expects 46 dims, got {obs_dim_per_step}")

    @staticmethod
    def _mirror_leg_state(values):
        # Per-leg order: hip yaw, hip roll, hip pitch, knee pitch,
        # ankle pitch, ankle roll. Sagittal reflection swaps legs and flips
        # the yaw/roll axes.
        sign = values.new_tensor([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
        return torch.cat((values[..., 6:12] * sign, values[..., 0:6] * sign), dim=-1)

    def _mirror_obs(self, obs):
        steps = obs.view(obs.shape[0], -1, self.obs_dim_per_step)
        mirrored = steps.clone()
        mirrored[..., 0:12] = self._mirror_leg_state(steps[..., 0:12])
        mirrored[..., 12:24] = self._mirror_leg_state(steps[..., 12:24])
        mirrored[..., 25] *= -1.0  # base linear velocity y
        mirrored[..., 27] *= -1.0  # base angular velocity x
        mirrored[..., 29] *= -1.0  # base angular velocity z
        mirrored[..., 31] *= -1.0  # projected gravity y
        left_pos = steps[..., 34:37].clone()
        right_pos = steps[..., 37:40].clone()
        left_vel = steps[..., 40:43].clone()
        right_vel = steps[..., 43:46].clone()
        mirrored[..., 34:37] = right_pos * right_pos.new_tensor([1.0, -1.0, 1.0])
        mirrored[..., 37:40] = left_pos * left_pos.new_tensor([1.0, -1.0, 1.0])
        mirrored[..., 40:43] = right_vel * right_vel.new_tensor([1.0, -1.0, 1.0])
        mirrored[..., 43:46] = left_vel * left_vel.new_tensor([1.0, -1.0, 1.0])
        return mirrored.reshape(obs.shape[0], -1)

    def get_expert_obs(self, batch_size):
        return self._mirror_obs(self.source.get_expert_obs(batch_size))

    def sample_reference_state(self, batch_size, random_time=True):
        state = self.source.sample_reference_state(batch_size, random_time=random_time)
        return {
            "dof_pos": self._mirror_leg_state(state["dof_pos"]),
            "dof_vel": self._mirror_leg_state(state["dof_vel"]),
            "base_z": state["base_z"],
        }


class LeggedRobotK1LocoAmp(LeggedRobotMoveAmp2D):
    """K1 first-stage locomotion task: body-frame velocity command + AMP imitation."""

    def _init_buffers(self):
        LeggedRobot._init_buffers(self)
        self._init_lower_body_action_mapping()
        self._reinit_amp_motions_for_lower_body()
        self._init_loco_command_buffers()
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_yaw[:] = self.yaw
        self._resample_commands(all_env_ids)
        self.gait_phase[all_env_ids] = torch.rand(
            len(all_env_ids), device=self.device
        ) * 2.0 * math.pi

    def _init_loco_command_buffers(self):
        n = self.num_envs
        device = self.device
        self.commands = torch.zeros(n, 3, dtype=torch.float, device=device)
        self.command_type_ids = torch.zeros(n, dtype=torch.long, device=device)
        self.command_time_left = torch.zeros(n, dtype=torch.float, device=device)
        self.gait_phase = torch.zeros(n, dtype=torch.float, device=device)
        self.reset_yaw = torch.zeros(n, dtype=torch.float, device=device)
        self.yaw_error = torch.zeros(n, dtype=torch.float, device=device)
        self.target_base_z = self.torso_pos[:, 2].clone()
        self.episode_speed_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_command_error_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_yaw_error_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_amp_reward_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_fall = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_steps = torch.zeros(n, dtype=torch.float, device=device)
        self.feet_air_time = torch.zeros(n, 2, dtype=torch.float, device=device)
        self.last_foot_contacts = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.flight_time = torch.zeros(n, dtype=torch.float, device=device)
        # Contact columns: left, right, both, neither.
        self.episode_contact_sums = torch.zeros(n, 4, dtype=torch.float, device=device)
        self.episode_double_flight_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_touchdown_air_time_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_touchdown_count = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_base_height_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_base_height_sq_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_base_vz_sq_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_vertical_acc_sq_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_torque_saturation_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_any_torque_saturation_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.episode_forward_speed_sum = torch.zeros(n, dtype=torch.float, device=device)
        self.left_foot_index, self.right_foot_index = self._resolve_loco_foot_indices()
        self.loco_foot_indices = (
            torch.tensor(
                [self.left_foot_index, self.right_foot_index],
                dtype=torch.long,
                device=device,
            )
            if self.left_foot_index is not None and self.right_foot_index is not None
            else None
        )

    def _resolve_loco_foot_indices(self):
        """Resolve left/right contact feet by body name, with an ordered fallback."""
        contact_indices = [int(index.item()) for index in self.contact_feet_indices]
        named_indices = [
            (str(self.body_names[index]).lower(), index) for index in contact_indices
        ]
        left = next((index for name, index in named_indices if "left" in name), None)
        right = next((index for name, index in named_indices if "right" in name), None)
        if left is not None and right is not None:
            return left, right
        if len(contact_indices) >= 2:
            # K1 currently enumerates left_foot_link before right_foot_link. Keep a
            # conservative fallback for assets whose body names omit side labels.
            print(
                "[k1_loco_amp] warning: could not resolve left/right foot by body name; "
                "falling back to the first two contact feet."
            )
            return contact_indices[0], contact_indices[1]
        return None, None

    def _reinit_amp_motions_for_lower_body(self):
        self.amp_lower_dof_names = self.lower_body_dof_names
        self.amp_lower_dof_indices = self.lower_body_dof_indices
        self.amp_obs_per_step = int(getattr(self.cfg.amp, "num_obs_per_step", 32))
        multidataset, mapping = load_imitation_dataset(
            self.cfg.dataset.folder.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            self.cfg.dataset.joint_mapping.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
        )
        num_steps = int(getattr(self.cfg.amp, "num_steps", 2))
        amp_obs_type = getattr(self.cfg.amp, "obs_type", "lower_body_state")
        weights = getattr(self.cfg.dataset, "motion_weights", {}) or {}
        filter_by_weight = bool(getattr(self.cfg.dataset, "filter_motion_by_weight", False))

        self.motions = {}
        for key, dataset in multidataset.items():
            weight = self._motion_weight(key, weights)
            if filter_by_weight and weight <= 0.0:
                print(f"[k1_loco_amp] skip motion {key}: weight={weight}")
                continue
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

        if "diagonal" in self.motions and "diagonal_right" not in self.motions:
            self.motions["diagonal_right"] = _MirroredLocomotionMotion(
                self.motions["diagonal"], self.amp_obs_per_step
            )
            print("[k1_loco_amp] generated mirrored expert motion: diagonal -> diagonal_right")

        self._init_ref_motion_sampler()
        self.amp_motion_buffers = []
        self.amp_motion_probs = []
        self.amp_motion_names = []
        for name, buffer in self.motions.items():
            weight = self._motion_weight(name, weights)
            if weight <= 0.0:
                continue
            self.amp_motion_buffers.append(buffer)
            self.amp_motion_probs.append(weight)
            self.amp_motion_names.append(name)
        if not self.amp_motion_buffers:
            raise RuntimeError("No positive-weight AMP motions for k1_loco_amp.")
        probs = torch.tensor(self.amp_motion_probs, dtype=torch.float)
        self.amp_motion_probs = probs / probs.sum().clamp(min=1e-6)
        self._init_amp_command_motion_map()

    def _init_amp_command_motion_map(self):
        specs = getattr(self.cfg.commands, "motion_commands", {})
        command_names = list(specs.keys())
        explicit_map = getattr(self.cfg.amp, "command_motion_map", {}) or {}
        motion_names = list(self.motions.keys())
        self.amp_command_names = command_names
        self.amp_command_motion_names = [
            self._resolve_command_motion_name(name, explicit_map, motion_names)
            for name in command_names
        ]

    @staticmethod
    def _resolve_command_motion_name(command_name, explicit_map, motion_names):
        if command_name in explicit_map:
            return explicit_map[command_name]
        lowered = command_name.lower()
        motion_lowers = {name.lower(): name for name in motion_names}
        if lowered in motion_lowers:
            return motion_lowers[lowered]
        if lowered.startswith("diagonal") and "diagonal" in motion_lowers:
            return motion_lowers["diagonal"]
        for name in motion_names:
            name_lower = name.lower()
            if lowered in name_lower or name_lower in lowered:
                return name
        return None

    @staticmethod
    def _motion_weight(name, weights):
        lowered = name.lower()
        motion_keys = [
            key for key in weights.keys()
            if str(key).lower() not in ("__default__", "default")
        ]
        for key in sorted(motion_keys, key=lambda item: len(str(item)), reverse=True):
            value = weights[key]
            key_lower = str(key).lower()
            if lowered == key_lower or key_lower in lowered:
                return float(value)
        return float(weights.get("__default__", weights.get("default", 1.0)))

    def _sample_uniform(self, value_range, shape):
        lo = float(value_range[0])
        hi = float(value_range[1])
        if abs(hi - lo) < 1e-6:
            return torch.full(shape, lo, dtype=torch.float, device=self.device)
        return torch.empty(*shape, dtype=torch.float, device=self.device).uniform_(lo, hi)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        specs = getattr(self.cfg.commands, "motion_commands")
        names = list(specs.keys())
        probs = torch.tensor([float(specs[name]["prob"]) for name in names], dtype=torch.float, device=self.device)
        probs = probs / probs.sum().clamp(min=1e-6)
        sampled = torch.multinomial(probs, len(env_ids), replacement=True)
        self.command_type_ids[env_ids] = sampled

        for i, name in enumerate(names):
            mask = sampled == i
            if not mask.any():
                continue
            ids = env_ids[mask]
            spec = specs[name]
            self.commands[ids, 0] = self._sample_uniform(spec["vx"], (len(ids),))
            self.commands[ids, 1] = self._sample_uniform(spec["vy"], (len(ids),))
            if getattr(self.cfg.commands, "train_yaw_command", False):
                self.commands[ids, 2] = self._sample_uniform(spec["wz"], (len(ids),))
            else:
                self.commands[ids, 2] = 0.0
        self.command_time_left[env_ids] = float(getattr(self.cfg.commands, "resampling_time", 2.0))

    def _command_actor_obs(self):
        scale = torch.tensor(self.cfg.commands.command_scale, dtype=torch.float, device=self.device).view(1, 3)
        command_obs = self.commands * scale
        phase_obs = torch.stack(
            (torch.sin(self.gait_phase), torch.cos(self.gait_phase)), dim=-1
        )
        return torch.cat((command_obs, phase_obs), dim=-1)

    def _build_actor_one_step_obs(self):
        return torch.cat(
            (
                self._command_actor_obs(),
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

    def _build_privileged_obs(self, actor_obs):
        return torch.cat((actor_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        start = 0
        noise_vec[start : start + 3] = getattr(noise_scales, "command", 0.02) * noise_level
        start += 3
        # The deterministic gait clock is not sensor data and receives no noise.
        noise_vec[start : start + 2] = 0.0
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
            raise ValueError(
                f"noise obs dim mismatch: filled {start}, expected {self.num_one_step_obs}"
            )
        return noise_vec

    def _update_gait_phase(self):
        speed = torch.norm(self.commands[:, :2], dim=-1)
        cfg = self.cfg.rewards
        base_freq = float(getattr(cfg, "gait_phase_base_frequency", 1.15))
        gain = float(getattr(cfg, "gait_phase_speed_frequency_gain", 1.0))
        min_freq = float(getattr(cfg, "gait_phase_min_frequency", 1.0))
        max_freq = float(getattr(cfg, "gait_phase_max_frequency", 2.1))
        min_speed = float(getattr(cfg, "gait_phase_min_speed", 0.06))
        freq = torch.clamp(base_freq + gain * speed, min=min_freq, max=max_freq)
        moving = (speed > min_speed).float()
        self.gait_phase = torch.remainder(
            self.gait_phase + moving * 2.0 * math.pi * freq * self.dt,
            2.0 * math.pi,
        )

    def compute_observations(self):
        actor_obs = self._build_actor_one_step_obs()
        if actor_obs.shape[1] != self.num_one_step_obs:
            raise RuntimeError(f"actor obs dim mismatch: got {actor_obs.shape[1]}, expected {self.num_one_step_obs}")
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

    def get_amp_observations(self):
        q_leg = self.dof_pos[:, self.amp_lower_dof_indices]
        dq_leg = self.dof_vel[:, self.amp_lower_dof_indices]
        foot_rel_pos, foot_rel_vel = self._get_current_foot_state()
        return build_locomotion_amp_step_obs(
            q_leg,
            dq_leg,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.torso_pos[:, 2:3],
            foot_rel_pos,
            foot_rel_vel,
        )

    def get_amp_motion_ids(self):
        return self.command_type_ids

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
        self.projected_gravity[:] = quat_rotate_inverse(upper_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        self.yaw_error = wrap_to_pi(self.yaw - self.reset_yaw)
        self.command_time_left -= self.dt
        resample_ids = (self.command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        self._resample_commands(resample_ids)
        self._update_gait_phase()

        command_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=-1)
        self.episode_speed_sum += torch.norm(self.base_lin_vel[:, :2], dim=-1)
        self.episode_forward_speed_sum += self.base_lin_vel[:, 0]
        self.episode_command_error_sum += command_error
        self.episode_yaw_error_sum += torch.abs(self.yaw_error)
        self.episode_steps += 1.0
        self._update_locomotion_diagnostics()

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

    def _foot_contacts(self):
        if self.left_foot_index is None or self.right_foot_index is None:
            return torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        threshold = float(getattr(self.cfg.rewards, "foot_contact_threshold", 1.0))
        return self.contact_forces[:, self.loco_foot_indices, 2] > threshold

    def _get_current_foot_state(self):
        if self.left_foot_index is None or self.right_foot_index is None:
            zeros = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
            return zeros, zeros

        feet_world_pos = self.rigid_body_states[:, self.loco_foot_indices, 0:3]
        feet_world_vel = self.rigid_body_states[:, self.loco_foot_indices, 7:10]
        torso_world_pos = self.torso_pos[:, None, :]
        torso_world_vel = self.rigid_body_states[:, self.upper_body_index, 7:10][:, None, :]

        torso_quat = self.rigid_body_states[:, self.upper_body_index, 3:7]
        flat_quat = torso_quat[:, None, :].repeat(1, 2, 1).reshape(-1, 4)
        rel_pos_world = (feet_world_pos - torso_world_pos).reshape(-1, 3)
        rel_vel_world = (feet_world_vel - torso_world_vel).reshape(-1, 3)

        foot_rel_pos = quat_rotate_inverse(flat_quat, rel_pos_world).view(self.num_envs, 2, 3)
        foot_rel_vel = quat_rotate_inverse(flat_quat, rel_vel_world).view(self.num_envs, 2, 3)
        return foot_rel_pos, foot_rel_vel

    def _update_locomotion_diagnostics(self):
        contacts = self._foot_contacts()
        left = contacts[:, 0]
        right = contacts[:, 1]
        no_contact = ~left & ~right
        self.flight_time = torch.where(
            no_contact,
            self.flight_time + self.dt,
            torch.zeros_like(self.flight_time),
        )
        self.episode_contact_sums[:, 0] += left.float()
        self.episode_contact_sums[:, 1] += right.float()
        self.episode_contact_sums[:, 2] += (left & right).float()
        self.episode_contact_sums[:, 3] += no_contact.float()
        grace = float(getattr(self.cfg.rewards, "flight_grace_s", 0.05))
        self.episode_double_flight_sum += (self.flight_time > grace).float()
        height = self.torso_pos[:, 2]
        self.episode_base_height_sum += height
        self.episode_base_height_sq_sum += torch.square(height)
        self.episode_base_vz_sq_sum += torch.square(self.base_lin_vel[:, 2])
        self.episode_vertical_acc_sq_sum += torch.square(self.base_lin_acc[:, 2])
        saturated = torch.abs(self.torques) >= 0.999 * self.torque_limits.unsqueeze(0)
        self.episode_torque_saturation_sum += saturated.float().mean(dim=1)
        self.episode_any_torque_saturation_sum += saturated.any(dim=1).float()

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
        if getattr(cfg, "enable_yaw_termination", True):
            yaw_limit_buf = torch.abs(self.yaw_error) > float(
                getattr(cfg, "yaw_limit", math.radians(45.0))
            )
        else:
            yaw_limit_buf = torch.zeros_like(self.time_out_buf, dtype=torch.bool)
        self.fall_buf = knee_height_buf | base_height_buf | self.gravity_termination_buf | sharpforce_buf
        self.reset_buf = self.time_out_buf | self.fall_buf | yaw_limit_buf
        self.episode_fall = torch.maximum(self.episode_fall, self.fall_buf.float())

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        episode_lengths = torch.clip(self.episode_length_buf[env_ids].float(), min=1.0)
        episode_steps = torch.clip(self.episode_steps[env_ids].clone(), min=1.0)
        final_speed_sum = self.episode_speed_sum[env_ids].clone()
        final_command_error = self.episode_command_error_sum[env_ids].clone() / episode_steps
        final_yaw_error = self.episode_yaw_error_sum[env_ids].clone() / episode_steps
        final_fall = self.episode_fall[env_ids].clone()
        final_amp_reward = self.episode_amp_reward_sum[env_ids].clone() / episode_lengths
        final_contact_ratios = self.episode_contact_sums[env_ids].clone() / episode_steps.unsqueeze(-1)
        final_double_flight_ratio = self.episode_double_flight_sum[env_ids].clone() / episode_steps
        final_mean_air_time = self.episode_touchdown_air_time_sum[env_ids].clone() / torch.clamp(
            self.episode_touchdown_count[env_ids].clone(), min=1.0
        )
        height_mean = self.episode_base_height_sum[env_ids].clone() / episode_steps
        height_sq_mean = self.episode_base_height_sq_sum[env_ids].clone() / episode_steps
        final_height_std = torch.sqrt(torch.clamp(height_sq_mean - torch.square(height_mean), min=0.0))
        final_base_vz_rms = torch.sqrt(
            self.episode_base_vz_sq_sum[env_ids].clone() / episode_steps
        )
        final_vertical_acc_rms = torch.sqrt(
            self.episode_vertical_acc_sq_sum[env_ids].clone() / episode_steps
        )
        final_torque_saturation = self.episode_torque_saturation_sum[env_ids].clone() / episode_steps
        final_any_torque_saturation = (
            self.episode_any_torque_saturation_sum[env_ids].clone() / episode_steps
        )

        if getattr(self.cfg.domain_rand, "randomize_rigid_props_on_reset", False):
            self.refresh_actor_rigid_shape_props(env_ids)
            if (
                getattr(self.cfg.domain_rand, "randomize_payload_mass", False)
                or getattr(self.cfg.domain_rand, "randomize_com_displacement", False)
                or getattr(self.cfg.domain_rand, "randomize_link_mass", False)
            ):
                self.refresh_actor_rigid_body_props(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self.joint_powers[env_ids] = 0.0
        self.reset_buf[env_ids] = 1

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0],
                self.cfg.domain_rand.kp_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0],
                self.cfg.domain_rand.kd_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection[env_ids] = torch_rand_float(
                self.cfg.domain_rand.joint_injection_range[0],
                self.cfg.domain_rand.joint_injection_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
            self.joint_injection[env_ids[:, None], self.curriculum_dof_indices[None, :]] = 0.0
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(
                self.cfg.domain_rand.actuation_offset_range[0],
                self.cfg.domain_rand.actuation_offset_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
            self.actuation_offset[env_ids[:, None], self.curriculum_dof_indices[None, :]] = 0.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids] / episode_lengths / self.dt
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.reset_yaw[env_ids] = self.yaw[env_ids]
        self.yaw_error[env_ids] = 0.0
        self.target_base_z[env_ids] = self.torso_pos[env_ids, 2]
        self._resample_commands(env_ids)
        self.gait_phase[env_ids] = torch.rand(
            len(env_ids), device=self.device
        ) * 2.0 * math.pi

        self.extras["episode"]["mean_speed"] = torch.mean(final_speed_sum / episode_lengths)
        self.extras["episode"]["forward_speed"] = torch.mean(
            self.episode_forward_speed_sum[env_ids].clone() / episode_steps
        )
        self.extras["episode"]["mean_command_error"] = torch.mean(final_command_error)
        self.extras["episode"]["mean_yaw_error"] = torch.mean(final_yaw_error)
        self.extras["episode"]["fall_rate"] = torch.mean(final_fall)
        self.extras["episode"]["amp_reward_mean"] = torch.mean(final_amp_reward)
        self.extras["episode"]["left_contact_ratio"] = torch.mean(final_contact_ratios[:, 0])
        self.extras["episode"]["right_contact_ratio"] = torch.mean(final_contact_ratios[:, 1])
        self.extras["episode"]["both_contact_ratio"] = torch.mean(final_contact_ratios[:, 2])
        self.extras["episode"]["flight_ratio"] = torch.mean(final_contact_ratios[:, 3])
        self.extras["episode"]["double_flight_ratio"] = torch.mean(final_double_flight_ratio)
        self.extras["episode"]["feet_air_time_mean"] = torch.mean(final_mean_air_time)
        self.extras["episode"]["base_height_std"] = torch.mean(final_height_std)
        self.extras["episode"]["base_vz_rms"] = torch.mean(final_base_vz_rms)
        self.extras["episode"]["vertical_acc_rms"] = torch.mean(final_vertical_acc_rms)
        self.extras["episode"]["torque_saturation_ratio"] = torch.mean(final_torque_saturation)
        self.extras["episode"]["steps_with_any_torque_saturation_ratio"] = torch.mean(
            final_any_torque_saturation
        )

        self.episode_speed_sum[env_ids] = 0.0
        self.episode_command_error_sum[env_ids] = 0.0
        self.episode_yaw_error_sum[env_ids] = 0.0
        self.episode_amp_reward_sum[env_ids] = 0.0
        self.episode_fall[env_ids] = 0.0
        self.episode_steps[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_foot_contacts[env_ids] = False
        self.flight_time[env_ids] = 0.0
        self.episode_contact_sums[env_ids] = 0.0
        self.episode_double_flight_sum[env_ids] = 0.0
        self.episode_touchdown_air_time_sum[env_ids] = 0.0
        self.episode_touchdown_count[env_ids] = 0.0
        self.episode_base_height_sum[env_ids] = 0.0
        self.episode_base_height_sq_sum[env_ids] = 0.0
        self.episode_base_vz_sq_sum[env_ids] = 0.0
        self.episode_vertical_acc_sq_sum[env_ids] = 0.0
        self.episode_torque_saturation_sum[env_ids] = 0.0
        self.episode_any_torque_saturation_sum[env_ids] = 0.0
        self.episode_forward_speed_sum[env_ids] = 0.0

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reward_tracking_lin_vel(self):
        error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=-1)
        sigma = max(float(getattr(self.cfg.rewards, "tracking_sigma", 0.25)), 1e-6)
        return torch.exp(-error / sigma)

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        sigma = max(float(getattr(self.cfg.rewards, "yaw_rate_sigma", 0.25)), 1e-6)
        return torch.exp(-error / sigma)

    def _reward_yaw_stability(self):
        sigma = max(float(getattr(self.cfg.rewards, "yaw_sigma", 0.35)), 1e-6)
        return torch.exp(-torch.square(self.yaw_error) / sigma)

    def _reward_stand_still(self):
        cmd_norm = torch.norm(self.commands[:, :2], dim=-1)
        standing = cmd_norm < 0.05
        leg_vel = torch.sum(torch.square(self.dof_vel[:, self.lower_body_dof_indices]), dim=-1)
        sigma = max(float(getattr(self.cfg.rewards, "stand_still_sigma", 0.1)), 1e-6)
        return standing.float() * torch.exp(-leg_vel * sigma)

    def _moving_gate(self):
        min_speed = float(getattr(self.cfg.rewards, "locomotion_min_speed", 0.06))
        return (torch.norm(self.commands[:, :2], dim=-1) > min_speed).float()

    def _reward_base_vz(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_vertical_acc(self):
        return torch.square(self.base_lin_acc[:, 2])

    def _reward_bilateral_flight(self):
        grace = float(getattr(self.cfg.rewards, "flight_grace_s", 0.05))
        return self._moving_gate() * (self.flight_time > grace).float()

    def _reward_feet_air_time(self):
        """Reward bounded single-foot swing durations only at touchdown.

        This is intentionally not an unbounded air-time reward: simultaneous
        flight is handled by a separate penalty and standing commands are gated
        out, so the term cannot improve by turning walking into hopping.
        """
        contacts = self._foot_contacts()
        self.feet_air_time += self.dt
        touchdown = contacts & (~self.last_foot_contacts)
        target = float(getattr(self.cfg.rewards, "feet_air_time_target", 0.22))
        sigma = max(float(getattr(self.cfg.rewards, "feet_air_time_sigma", 0.01)), 1e-6)
        minimum = float(getattr(self.cfg.rewards, "feet_air_time_min", 0.08))
        valid_touchdown = touchdown & (self.feet_air_time >= minimum)
        reward = torch.exp(-torch.square(self.feet_air_time - target) / sigma)
        reward = torch.sum(reward * valid_touchdown.float(), dim=1) * self._moving_gate()
        touchdown_air_time = torch.sum(self.feet_air_time * valid_touchdown.float(), dim=1)
        self.episode_touchdown_air_time_sum += touchdown_air_time
        self.episode_touchdown_count += valid_touchdown.float().sum(dim=1)
        self.feet_air_time[contacts] = 0.0
        self.last_foot_contacts[:] = contacts
        return reward

    def _reward_soft_heading(self):
        deadzone = float(getattr(self.cfg.rewards, "heading_deadzone", math.radians(12.0)))
        excess = torch.clamp(torch.abs(self.yaw_error) - deadzone, min=0.0)
        return self._moving_gate() * torch.square(excess)
