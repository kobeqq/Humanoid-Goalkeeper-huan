import math

from legged_gym.envs.k1.k1_loco_amp_config import K1LocoAmpCfg, K1LocoAmpCfgPPO


class K1LocoAmpFullCfg(K1LocoAmpCfg):
    class commands(K1LocoAmpCfg.commands):
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

    class rewards(K1LocoAmpCfg.rewards):
        class scales(K1LocoAmpCfg.rewards.scales):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.2
            gait_phase = 0.0
            upright = 0.5
            height = 0.2
            yaw_stability = 0.2
            stand_still = 0.1

            feet_air_time = 0.0
            bilateral_flight = 0.0
            base_vz = 0.0
            vertical_acc = 0.0
            soft_heading = 0.0

            feet_slip = -0.03
            ang_vel_xy = -0.02
            dof_acc = -2.5e-7
            smoothness = -0.003
            torques = -1e-5
            dof_vel = -3e-4
            dof_pos_limits = -1.0
            dof_vel_limits = -0.05
            torque_limits = -0.1

        tracking_sigma = 0.14
        yaw_rate_sigma = 0.25
        yaw_sigma = 0.35
        yaw_limit = math.radians(45.0)
        height_sigma = 0.04
        stand_still_sigma = 0.1
        only_positive_rewards = False

    class amp(K1LocoAmpCfg.amp):
        obs_type = "lower_body_state"
        use_all_dofs = False
        use_leg_dofs = True
        include_dof_vel = True
        num_steps = 2
        num_obs_per_step = 32
        num_obs = num_obs_per_step * num_steps

        enable_discriminator = True
        skip_base_motion_init = True
        reward_mode = "mixture"
        amp_coef = 0.3
        amp_scale = 1.0
        adaptive_amp_scale = False
        condition_on_command = False
        command_motion_map = {}

    class dataset(K1LocoAmpCfg.dataset):
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper_from_pkl_k1"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper_from_pkl_k1/joint_id_k1.txt"
        frame_rate = 30
        min_time = 0.1
        include_files = ["goalkeeper_from_pkl_k1.pt"]
        exclude_files = []
        filter_motion_by_weight = False
        motion_weights = {"__default__": 1.0}


class K1LocoAmpFullCfgPPO(K1LocoAmpCfgPPO):
    class runner(K1LocoAmpCfgPPO.runner):
        run_name = "k1_loco_amp_full"
        experiment_name = "k1_loco_amp_full"
        wandb_project = "k1_loco_amp_full"

    class policy(K1LocoAmpCfgPPO.policy):
        estimate_ball_dim = 0

    amp = K1LocoAmpFullCfg.amp
