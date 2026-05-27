from legged_gym.envs.k1.k1_22_config import K122Cfg, K122CfgPPO



class K1MoveAmpCfg(K122Cfg):
    class env(K122Cfg.env):
        num_ballobs = 0
        num_target_obs = 3
         # target_local 3 + base_ang_vel 3 + gravity 3 + dof_pos 22 + dof_vel 22 + actions 22
        num_one_step_observations = num_target_obs + 6 + K122Cfg.env.num_dofs * 2 + K122Cfg.env.num_actions
        num_privileged_obs = num_one_step_observations + 3
        num_observations = K122Cfg.env.num_actor_history * num_one_step_observations
        episode_length_s = 5
        play = False

        use_ball_actor = False

    class commands(K122Cfg.commands):
        # Curriculum: fixed forward goal first (XY task only, Z ignored).
        fix_target = True
        fixed_target_x = 5.0  # meters ahead of each env origin (+x)
        target_use_y = False  # no lateral offset (y = env origin y)
        target_use_z = False  # goal height tracks torso; success/reward use XY only

        target_x = [0.5, 2.0]
        target_y = [-1.2, 1.2]
        target_z = [0.5, 1.1]

    class rewards(K122Cfg.rewards):
        class scales:
            # Task: move_to_target is the main locomotion signal; tracking gives far-field gradient.
            move_to_target = 15
            tracking_target = 10
            upright = 1.5

            # Regularization: keep penalties weaker during locomotion curriculum so they
            # do not dominate and encourage "stand still / tiny motion".
            ang_vel_xy = -0.03
            dof_acc = -2.5e-7
            smoothness = -0.01
            torques = -1e-5
            dof_vel = -5e-4
            dof_pos_limits = -1.5
            dof_vel_limits = -0.2
            torque_limits = -0.5

        # Wider Gaussian -> stronger reward gradient when still ~1 m from goal.
        target_sigma = 2.0
        # Slightly easier success signal while learning to walk (XY distance).
        target_reach_threshold = 0.35
        # False: penalties stay in the return so "violent / limit-hitting" steps are worse than calm ones.
        only_positive_rewards = False
        soft_dof_pos_limit = K122Cfg.rewards.soft_dof_pos_limit
        soft_dof_vel_limit = K122Cfg.rewards.soft_dof_vel_limit
        soft_torque_limit = K122Cfg.rewards.soft_torque_limit
        max_contact_force = K122Cfg.rewards.max_contact_force

    class noise(K122Cfg.noise):
        class noise_scales(K122Cfg.noise.noise_scales):
            target = 0.05

class K1MoveAmpCfgPPO(K122CfgPPO):
    class runner(K122CfgPPO.runner):
        run_name = "k1_move_amp"
        experiment_name = "k1_move_amp"
        wandb_project = "k1_move_amp"
        # logger = "wanb"
    

    class policy(K122CfgPPO.policy):
        estimate_ball_dim = 0

    amp = K1MoveAmpCfg.amp