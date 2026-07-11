import math

from legged_gym.envs.k1.k1_move_amp_config import K1MoveAmpCfg, K1MoveAmpCfgPPO
from legged_gym.envs.k1.k1_22_config import K122Cfg


class K1LocoAmpCfg(K1MoveAmpCfg):
    class env(K1MoveAmpCfg.env):
        num_ballobs = 0
        num_actions = 12
        # Command-related actor block: scaled (vx, vy, wz) + gait phase (sin, cos).
        # self.commands itself remains three-dimensional.
        num_command_obs = 5
        num_one_step_observations = num_command_obs + 3 + 3 + K122Cfg.env.num_dofs * 2 + num_actions
        num_privileged_obs = num_one_step_observations + 3
        num_observations = K122Cfg.env.num_actor_history * num_one_step_observations
        episode_length_s = 10
        use_ball_actor = False
        play = False

    class commands(K1MoveAmpCfg.commands):
        resampling_time = 5
        max_vx = 0.35
        max_vy = 0.35
        max_wz = 0.5
        command_scale = [1.0 / max_vx, 1.0 / max_vy, 1.0 / max_wz]
        train_yaw_command = False
        motion_commands = {
            "standing": {"prob": 0.10, "vx": [0.0, 0.0], "vy": [0.0, 0.0], "wz": [0.0, 0.0]},
            "forward": {"prob": 0.15, "vx": [0.20, 0.35], "vy": [-0.04, 0.04], "wz": [0.0, 0.0]},
            "backward": {"prob": 0.15, "vx": [-0.32, -0.18], "vy": [-0.04, 0.04], "wz": [0.0, 0.0]},
            "leftstep": {"prob": 0.20, "vx": [-0.04, 0.04], "vy": [0.18, 0.32], "wz": [0.0, 0.0]},
            "rightstep": {"prob": 0.20, "vx": [-0.04, 0.04], "vy": [-0.32, -0.18], "wz": [0.0, 0.0]},
            "diagonal_left": {"prob": 0.10, "vx": [0.18, 0.32], "vy": [0.12, 0.28], "wz": [0.0, 0.0]},
            "diagonal_right": {"prob": 0.10, "vx": [0.18, 0.32], "vy": [-0.28, -0.12], "wz": [0.0, 0.0]},
        }

    class init_state(K1MoveAmpCfg.init_state):
        root_vel_init_range = 0.10
        state_init = "Hybrid"
        ref_init_prob = 0.30

    class domain_rand(K1MoveAmpCfg.domain_rand):
        randomize_joint_injection = True
        joint_injection_range = [-0.005, 0.005]
        randomize_actuation_offset = True
        actuation_offset_range = [-0.005, 0.005]
        randomize_payload_mass = True
        payload_mass_range = [-0.5, 1.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.02, 0.02]
        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]
        randomize_rigid_props_on_reset = False
        randomize_friction = True
        friction_range = [0.6, 1.5]
        randomize_restitution = False
        restitution_range = [0.0, 0.03]
        randomize_kp = True
        kp_range = [0.9, 1.1]
        randomize_kd = True
        kd_range = [0.9, 1.1]
        randomize_initial_joint_pos = False
        continue_keep = False
        delay = True
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.2

    class rewards(K1MoveAmpCfg.rewards):
        class scales:
            # Task: command completion only.
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.1
            stand_still = 0.3

            # The gait clock remains an actor input, but no longer supplies a
            # style reward. AMP is responsible for the motion style.
            gait_phase = 0.0

            # Regularization: stability and physical plausibility only.
            upright = 0.7
            height = 0.25
            yaw_stability = 0.0
            feet_slip = -0.1
            feet_air_time = 0.25
            bilateral_flight = -0.5
            base_vz = -0.5
            vertical_acc = -0.002
            soft_heading = -0.5
            ang_vel_xy = -0.03
            dof_acc = -2.5e-7
            smoothness = -0.004
            torques = -2e-5
            dof_vel = -5e-4
            dof_pos_limits = -1.5
            dof_vel_limits = -0.1
            torque_limits = -0.2

        tracking_sigma = 0.14
        yaw_rate_sigma = 0.25
        yaw_sigma = 0.35
        yaw_limit = math.radians(45.0)
        enable_yaw_termination = False
        gait_phase_min_speed = 0.06
        gait_phase_full_speed = 0.30
        gait_phase_base_frequency = 1.15
        gait_phase_speed_frequency_gain = 1.0
        gait_phase_min_frequency = 1.0
        gait_phase_max_frequency = 2.1
        gait_phase_swing_height = 0.055
        gait_phase_sigma = 0.6
        locomotion_min_speed = 0.06
        foot_contact_threshold = 1.0
        flight_grace_s = 0.05
        feet_air_time_target = 0.22
        feet_air_time_sigma = 0.01
        feet_air_time_min = 0.08
        heading_deadzone = math.radians(15.0)
        height_sigma = 0.04
        stand_still_sigma = 0.1
        termination_knee_height = 0.10
        termination_base_height = 0.20
        termination_gravity_xy = 0.8
        termination_contact_force_scale = 1.5
        only_positive_rewards = False
        soft_dof_pos_limit = K122Cfg.rewards.soft_dof_pos_limit
        soft_dof_vel_limit = K122Cfg.rewards.soft_dof_vel_limit
        soft_torque_limit = K122Cfg.rewards.soft_torque_limit
        max_contact_force = K122Cfg.rewards.max_contact_force

    class noise(K1MoveAmpCfg.noise):
        class noise_scales(K1MoveAmpCfg.noise.noise_scales):
            command = 0.02

    class amp(K1MoveAmpCfg.amp):
        obs_type = "locomotion_style"
        use_all_dofs = False
        use_leg_dofs = True
        include_dof_vel = True
        num_steps = 2
        num_obs_per_step = 46
        num_obs = num_obs_per_step * num_steps
        enable_discriminator = True
        skip_base_motion_init = True
        amp_coef = 0.40
        reward_mode = "additive"
        amp_scale = 0.5
        command_motion_map = {
            "standing": "standing",
            "forward": "forward",
            "backward": "backward",
            "leftstep": "leftstep",
            "rightstep": "rightstep",
            "diagonal_left": "diagonal",
            "diagonal_right": "diagonal_right",
        }
        adaptive_amp_scale = True
        amp_target_fraction = 0.40
        amp_scale_min = 0.20
        amp_scale_max = 1.0
        amp_scale_ema_alpha = 0.02

    class dataset(K1MoveAmpCfg.dataset):
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper_from_pkl_k1_locomotion"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper_from_pkl_k1_locomotion/joint_id_k1.txt"
        filter_motion_by_weight = True
        motion_weights = {
            "__default__": 0.0,
            "standing": 0.15,
            "forward": 1.0,
            "backward": 1.0,
            "leftstep": 1.0,
            "rightstep": 1.0,
            "diagonal": 1.0,
            "diagonal_right": 1.0,
            "dive_jump": 0.0,
            "turn_or_yaw_heavy": 0.0,
        }


class K1LocoAmpCfgPPO(K1MoveAmpCfgPPO):
    class runner(K1MoveAmpCfgPPO.runner):
        run_name = "k1_loco_amp_v2"
        experiment_name = "k1_loco_amp_v2"
        wandb_project = "k1_loco_amp_v2"

    class policy(K1MoveAmpCfgPPO.policy):
        estimate_ball_dim = 0

    amp = K1LocoAmpCfg.amp
