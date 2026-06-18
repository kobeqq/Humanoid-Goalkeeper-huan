import math

from legged_gym.envs.k1.k1_move_amp_config import K1MoveAmpCfg, K1MoveAmpCfgPPO
from legged_gym.envs.k1.k1_22_config import K122Cfg


class K1MoveAmp2DCfg(K1MoveAmpCfg):
    class env(K1MoveAmpCfg.env):
        num_ballobs = 0
        num_actions = 12
        num_path_obs = 8
        # path 8 + base_ang_vel 3 + gravity 3 + dof_pos 22 + dof_vel 22 + lower action 12
        num_one_step_observations = num_path_obs + 6 + K122Cfg.env.num_dofs * 2 + num_actions
        # privileged: actor obs + base_lin_vel 3 + v_ref_world 2 + target_world 2 + cross 1 + yaw 1
        num_privileged_obs = num_one_step_observations + 3 + 2 + 2 + 1 + 1
        num_observations = K122Cfg.env.num_actor_history * num_one_step_observations
        episode_length_s = 6
        use_ball_actor = False
        play = False

    class commands(K1MoveAmpCfg.commands):
        # 2D point-goal：极坐标采样目标，不再依赖旧的 fixed +x 目标。
        fix_target = False
        target_use_y = True
        target_use_z = False
        target_radius = [0.6, 1.8]
        target_angle = [-math.pi, math.pi]
        target_z = [0.5, 1.1]

        # path-conditioned 速度参考：沿路径前进，横向偏差用法向速度拉回。
        v_max = 0.65
        v_min = 0.05
        slow_down_dist = 0.65
        k_cross = 0.8
        cross_track_obs_scale = 0.8
        reset_on_success = True

    class init_state(K1MoveAmpCfg.init_state):
        root_vel_init_range = 0.15
        # Hybrid：以 ref_init_prob 概率从拆分 motion 采样下肢 q/dq，否则默认站姿。
        state_init = "Hybrid"
        ref_init_prob = 0.3

    class domain_rand(K1MoveAmpCfg.domain_rand):
        # 最小可跑通版本先关闭强随机化，避免把路径跟踪和物理鲁棒性混在一起调。
        randomize_joint_injection = False
        randomize_actuation_offset = False
        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_link_mass = False
        randomize_friction = False
        randomize_restitution = False
        randomize_kp = False
        randomize_kd = False
        randomize_initial_joint_pos = False
        continue_keep = False
        delay = False
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.5

    class rewards(K1MoveAmpCfg.rewards):
        class scales:
            progress = 3.0
            velocity_track = 2.0
            cross_track = 1.5
            goal = 5.0
            yaw = 1.5
            yaw_violation = -2.0
            upright = 1.0
            height = 0.5
            feet_slip = -0.05

            ang_vel_xy = -0.03
            dof_acc = -2.5e-7
            smoothness = -0.01
            torques = -1e-5
            dof_vel = -5e-4
            dof_pos_limits = -1.5
            dof_vel_limits = -0.2
            torque_limits = -0.5

        target_reach_threshold = 0.35
        velocity_sigma = 0.25
        cross_track_sigma = 0.35
        yaw_sigma = 0.25
        yaw_soft_limit = math.radians(20.0)
        yaw_success_limit = math.radians(20.0)
        yaw_limit = math.radians(35.0)
        height_sigma = 0.04
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
            path = 0.05
            vel_ref = 0.02
            yaw = 0.01

    class amp(K1MoveAmpCfg.amp):
        obs_type = "lower_body_state"
        use_all_dofs = True
        num_steps = 2
        num_obs_per_step = 30
        num_obs = num_obs_per_step * num_steps
        amp_coef = 0.15

    class dataset(K1MoveAmpCfg.dataset):
        # 使用 split_k1_motion_dataset.py 从 goalkeeper_from_pkl_k1.pt 拆分出的 locomotion 子集。
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper_from_pkl_k1_locomotion"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper_from_pkl_k1_locomotion/joint_id_k1.txt"
        # AMP expert 采样权重：侧向/前后/对角步态用于 2D 全向移动；standing 少量；dive_jump 排除。
        motion_weights = {
            "__default__": 0.0,
            "leftstep": 1.0,
            "rightstep": 1.0,
            "forward": 1.0,
            "backward": 0.8,
            "diagonal": 0.8,
            "standing": 0.3,
            "turn_or_yaw_heavy": 0.2,
            "dive_jump": 0.0,
        }


class K1MoveAmp2DCfgPPO(K1MoveAmpCfgPPO):
    class runner(K1MoveAmpCfgPPO.runner):
        run_name = "k1_move_amp_2d"
        experiment_name = "k1_move_amp_2d"
        wandb_project = "k1_move_amp_2d"

    class policy(K1MoveAmpCfgPPO.policy):
        estimate_ball_dim = 0

    amp = K1MoveAmp2DCfg.amp
