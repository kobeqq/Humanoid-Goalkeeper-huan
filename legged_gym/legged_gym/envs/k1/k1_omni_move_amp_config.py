import math

from legged_gym.envs.k1.k1_22_config import K122Cfg, K122CfgPPO


class K1OmniMoveAmpCfg(K122Cfg):
    class env(K122Cfg.env):
        num_ballobs = 0
        # target_local 3 + yaw_error sin/cos 2
        num_target_obs = 5
        # target/yaw 5 + base_ang_vel 3 + gravity 3 + dof_pos 22 + dof_vel 22 + actions 22
        num_one_step_observations = num_target_obs + 6 + K122Cfg.env.num_dofs * 2 + K122Cfg.env.num_actions
        num_privileged_obs = num_one_step_observations + 3
        num_observations = K122Cfg.env.num_actor_history * num_one_step_observations
        episode_length_s = 5
        play = False
        use_ball_actor = False

    class commands(K122Cfg.commands):
        fix_target = False
        target_use_y = True
        target_use_z = False

        direction_names = ["front", "left", "back", "right"]
        curriculum_enable = True

        # direction bin: 0 front, 1 left, 2 back, 3 right
        stage_allowed_bins = [
            [0],
            [0, 1, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
        stage_bin_widths = [
            1.0,
            math.pi / 2.0,
            math.pi / 2.0,
            math.pi / 2.0,
        ]
        stage_radius_ranges = [
            [0.6, 1.3],
            [0.7, 1.6],
            [0.8, 2.0],
            [0.8, 2.2],
        ]
        stage_yaw_limits = [
            0.80,
            0.65,
            0.50,
            0.35,
        ]
        stage_reach_thresholds = [
            0.45,
            0.40,
            0.35,
            0.30,
        ]
        stage_yaw_success_limits = [
            0.70,
            0.55,
            0.42,
            0.30,
        ]
        curriculum_success_gates = [
            0.85,
            0.88,
            0.90,
        ]
        curriculum_yaw_fail_gates = [
            0.10,
            0.06,
            0.03,
        ]
        curriculum_min_episodes_per_bin = 1000
        curriculum_ema_alpha = 0.05

        reset_on_success = True
        desired_speed = 0.55
        slow_radius = 0.7

        yaw_ref_world = True
        yaw_ref = 0.0

        # Compatibility fields; omnidirectional sampling uses radius + heading.
        target_x = [-2.0, 2.0]
        target_y = [-2.0, 2.0]
        target_z = [0.5, 1.1]

    class rewards(K122Cfg.rewards):
        class scales:
            move_to_target = 10
            tracking_target = 5
            body_velocity_tracking = 2.5
            success_bonus = 8
            upright = 3
            yaw_tracking = 2.0
            yaw_rate_z = -0.05

            ang_vel_xy = -0.03
            dof_acc = -2.5e-7
            smoothness = -0.004
            torques = -1e-5
            dof_vel = -5e-4
            dof_pos_limits = -1.5
            dof_vel_limits = -0.2
            torque_limits = -0.15

        target_sigma = 1.2
        target_reach_threshold = 0.35
        yaw_sigma = 0.25
        body_vel_sigma = 0.25
        only_positive_rewards = False
        soft_dof_pos_limit = K122Cfg.rewards.soft_dof_pos_limit
        soft_dof_vel_limit = K122Cfg.rewards.soft_dof_vel_limit
        soft_torque_limit = K122Cfg.rewards.soft_torque_limit
        max_contact_force = K122Cfg.rewards.max_contact_force

    class noise(K122Cfg.noise):
        class noise_scales(K122Cfg.noise.noise_scales):
            target = 0.05
            yaw = 0.0


class K1OmniMoveAmpCfgPPO(K122CfgPPO):
    class runner(K122CfgPPO.runner):
        run_name = "k1_omni_move_amp"
        experiment_name = "k1_omni_move_amp"
        wandb_project = "k1_omni_move_amp"

    class policy(K122CfgPPO.policy):
        estimate_ball_dim = 0

    amp = K1OmniMoveAmpCfg.amp
