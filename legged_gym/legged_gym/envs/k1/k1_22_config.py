from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class K122Cfg(LeggedRobotCfg):#TODO: 这个类是用于创建机器人环境
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        num_actor_history = 10 
        
        
        num_actions = 22 # 机器人关节数
        num_dofs = 22
        num_ballobs = 3
        num_one_step_observations = 6 + num_ballobs + num_dofs * 2 + num_actions  #基座角速度3+投影重力向量3+球的位置3+关节位置29+关节速度29+上一时刻动作29
        num_privileged_obs = 6 + num_ballobs + num_dofs * 2 + num_actions  + 3 + 1 + 6 + 6 + 1  ##增加的特权观测：基座线速度3+目标区域1+末端目标位置3+球的速度3+左手位置3+右手位置3+球的距离1

        num_observations = num_actor_history * num_one_step_observations

        env_spacing = 5.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 3 # episode length in seconds
        ball_gravity = True
        play = False

    class commands:
        
        class ranges_0:
            height = [0.4, 1.2] 
            width =  [0.2, 1.2]

            maxh = [0.3, 1.5]
            maxw = [0.0, 1.8]

            evalh = [0.3, 1.5]
            evalw = [0.0, 1.5]

        class ranges_1:
            height = [0.4, 1.2] 
            width = [-1.2, -0.2]

            maxh = [0.3, 1.5] 
            maxw = [-1.8, -0.0]

            evalh = [0.3, 1.5]
            evalw = [-1.5, 0.0]


        class ranges_2:
            height = [1.2, 1.6] 
            width = [0, 1.0]

            maxh = [1.2, 1.8] 
            maxw = [0, 1.5]
        
            evalh = [1.2, 1.8] 
            evalw = [0, 1.5]

        class ranges_3:
            height = [1.2, 1.6] 
            width = [-1.0, 0.0]

            maxh = [1.2, 1.8] 
            maxw = [-1.5, 0.0]

            evalh = [1.2, 1.8] 
            evalw = [-1.5, 0.0]

        class ranges_4:
            height = [0.1, 0.3] 
            width = [0.2, 1.2]

            maxh = [0.1, 0.3]
            maxw = [0.0, 1.8]

            evalh = [0.1, 0.3] 
            evalw = [0.0, 1.5]

        
        class ranges_5:
            height = [0.1, 0.3] 
            width = [-1.2, -0.2]

            maxh = [0.1, 0.3]
            maxw = [-1.8, -0.0]

            evalh = [0.1, 0.3] 
            evalw = [-1.5, -0.0]



    class init_state(LeggedRobotCfg.init_state):#TODO: 这个类是用于初始化机器人状态
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            
            ".*Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.25,
            "Right_Shoulder_Roll": 1.25,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.15,
            ".*_Knee_Pitch": 0.3,
            ".*_Ankle_Pitch": -0.15,

            }

        init_pos = [-0.34930936, -0.03763366, -0.22198406,  0.93093884, -0.50943524, -0.08583859,
            0.13749947, -0.44516975, -0.06791031,  0.11570476, -0.17351833,  0.34241587,
              0.00395479,  0.49003497, -0.00168978,
            1.2062242,  0.00319979, -0.4975251,
            -0.00450607,  1.20307243]
        



    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {".*Shoulder_Pitch": 20.0,
                ".*Shoulder_Roll": 20.0,
                ".*Elbow_Pitch": 20.0,
                ".*Elbow_Yaw": 20.0,
                ".*_Hip_Pitch": 100.0,
                ".*_Hip_Roll": 100.0,
                ".*_Hip_Yaw": 100.0,
                ".*_Knee_Pitch": 100.0,
                ".*_Ankle_Pitch": 50,
                ".*_Ankle_Roll": 50,

                     }  # [N*m/rad]
        damping = {".*Shoulder_Pitch": 2.0,
                ".*Shoulder_Roll": 2.0,
                ".*Elbow_Pitch": 2.0,
                ".*Elbow_Yaw": 2.0,
                ".*_Hip_Pitch": 2.0,
                ".*_Hip_Roll": 2.0,
                ".*_Hip_Yaw": 2.0,
                ".*_Knee_Pitch": 2.0,
                ".*_Ankle_Pitch": 1,
                ".*_Ankle_Roll": 1,

                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25 #TODO: 这个参数是用于控制动作的缩放比例
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  #TODO: 这个参数是用于控制动作的更新频率
        curriculum_joints = ['Left_Shoulder_Roll', 'Left_Elbow_Yaw', 'Right_Shoulder_Roll', 'Right_Elbow_Yaw']  #c
        left_leg_joints = ['Left_Hip_Yaw', 'Left_Hip_Roll', 'Left_Hip_Pitch', 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll']
        right_leg_joints = ['Right_Hip_Yaw', 'Right_Hip_Roll', 'Right_Hip_Pitch', 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll']
        knee_joints = ['Left_Knee_Pitch', 'Right_Knee_Pitch']
        left_arm_joints = ['ALeft_Shoulder_Pitch', 'Left_Shoulder_Roll', 'Left_Elbow_Pitch', 'Left_Elbow_Yaw']
        right_arm_joints = ['ARight_Shoulder_Pitch', 'Right_Shoulder_Roll', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw']

        elbow_joints = ['Left_Elbow_Pitch', 'Left_Elbow_Yaw', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw']




        upper_body_link = "pelvis"  # "torso_link"   #这个是上体链接
        torso_link = "torso_link"  #这是上体链接

        left_hip_joints = ['Left_Hip_Yaw', 'Left_Hip_Roll', 'Left_Hip_Pitch']
        right_hip_joints = ['Right_Hip_Yaw', 'Right_Hip_Roll', 'Right_Hip_Pitch']


    class terrain:#TODO: 这个类是用于定义地形
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            ball_vel = 0.2
            ball_pos = 0.3
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


    class noise: #TODO: 这个类是用于定义噪声
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            ball = 0.08
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class asset(LeggedRobotCfg.asset):#TODO: 这个类是用于定义机器人资产
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/k1/urdf/K1_22dof.urdf'
        ballfile = '{LEGGED_GYM_ROOT_DIR}/resources/gymassets/urdf/ball.urdf'
        name = "g1"

        foot_name = "Ankle_Pitch"
        contact_foot_names = "foot_link"

        hand_name = "hand"
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "hand", "head"]
        terminate_after_contacts_on = []

        #waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        ankle_joints = [ "Left_Ankle_Pitch", "Left_Ankle_Roll","Right_Ankle_Pitch","Right_Ankle_Roll"]
        #imu_link = "imu_link"
        knee_names = ["Left_Ankle_Cross", "Right_Ankle_Cross"]
        
        #keyframe_name = "keyframe"

        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

        density = 0.001   #密度：用于控制机器人的质量
        angular_damping = 0.01   #角阻尼：用于控制机器人的角速度
        linear_damping = 0.01   #线阻尼：用于控制机器人的线速度
        max_angular_velocity = 1000.   #最大角速度：用于控制机器人的最大角速度
        max_linear_velocity = 1000.   #最大线速度：用于控制机器人的最大线速度
        armature = 0.01   #阻尼：用于控制机器人的阻尼
        thickness = 0.01   #厚度：用于控制机器人的厚度
    class domain_rand(LeggedRobotCfg.domain_rand):#TODO: 这个类是用于定义域随机化
        
        randomize_joint_injection = True
        joint_injection_range = [-0.01, 0.01]   #关节注入范围：用于控制关节注入的随机性
        
        randomize_actuation_offset = True
        actuation_offset_range = [-0.01, 0.01]   #执行偏移范围：用于控制执行的随机性

        randomize_payload_mass = True
        payload_mass_range = [-5, 10]   #负载质量范围：用于控制负载的质量

        randomize_com_displacement = True
        com_displacement_range = [-0.1, 0.1]   #质心位移范围：用于控制质心的位移

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]   #连杆质量范围：用于控制连杆的质量
        
        randomize_friction = True
        friction_range = [0.1, 2.0]   #摩擦力范围：用于控制摩擦力
        
        randomize_restitution = True  
        restitution_range = [0.0, 1.0]   #恢复系数范围：用于控制恢复系数
        
        randomize_kp = True
        kp_range = [0.8, 1.2]   #比例增益范围：用于控制比例增益
        
        randomize_kd = True
        kd_range = [0.8, 1.2]   #微分增益范围：用于控制微分增益
        
        randomize_initial_joint_pos = True
        continue_keep = True
        initial_joint_pos_scale = [0.5, 1.5]   #初始关节位置范围：用于控制初始关节位置
        initial_joint_pos_offset = [-0.1, 0.1]   #初始关节位置偏移：用于控制初始关节位置的偏移
        
        push_robots = True
        push_interval_s = 15   #推机器人间隔：用于控制推机器人的间隔
        max_push_vel_xy = 1.5   #最大推速度：用于控制推机器人的最大速度

        ball_interval_s = 0.5   #球间隔：用于控制球的间隔
        max_ball_vel = 0.5   #最大球速度：用于控制球的最大速度


        delay = True   #延迟：用于控制延迟

        
    class rewards: #TODO: 这个类是用于定义奖励
        class scales:
            
            # task rewards
            eereach = 10.0   #到达奖励：用于控制到达奖励
            success = 5.0   #成功奖励：用于控制成功奖励
            stopball = 100.0   #停止球奖励：用于控制停止球奖励

            # move rewards
            stayonline = -2.0   #保持在线奖励：用于控制保持在线奖励
            noretreat = -2.0   #不撤退奖励：用于控制不撤退奖励

            # feet rewards
            successland = 4.0   #成功着陆奖励：用于控制成功着陆奖励
            feetorientaion = 3.0   #脚姿态奖励：用于控制脚姿态奖励
            penalize_sharpcontact = -100.   #尖锐接触惩罚：用于控制尖锐接触惩罚
            penalize_kneeheight = -100.   #膝盖高度惩罚：用于控制膝盖高度惩罚
            feet_slippage = 3.0   #脚滑动惩罚：用于控制脚滑动惩罚

            # post rewards
            postorientation = 3.0   #姿态奖励：用于控制姿态奖励
            postangvel = 3.0   #角速度奖励：用于控制角速度奖励
            postupperdofpos = 1.0   #上肢关节位置奖励：用于控制上肢关节位置奖励
            #postwaistdofpos = 1.0   #腰部关节位置奖励：用于控制腰部关节位置奖励
            postlinvel = 1.0   #线速度奖励：用于控制线速度奖励


            # reg rewards
            ang_vel_xy = -0.1   #xy轴角速度惩罚：用于控制xy轴角速度惩罚
            dof_acc = -2.5e-7
            smoothness = -0.1   #平滑度惩罚：用于控制平滑度惩罚

            torques = -1e-5   #力矩惩罚：用于控制力矩惩罚
            dof_vel = -5e-4   #关节速度惩罚：用于控制关节速度惩罚

            dof_pos_limits = -3.0   #关节位置极限惩罚：用于控制关节位置极限惩罚
            dof_vel_limits = -2.0   #关节速度极限惩罚：用于控制关节速度极限惩罚
            torque_limits = -3.0   #力矩极限惩罚：用于控制力矩极限惩罚

            deviation_waist_pitch_joint = -0.001   #腰部关节pitch偏差惩罚：用于控制腰部关节pitch偏差惩罚


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)

        catch_th = 0.5   #抓取阈值：用于控制抓取阈值        
        handheight_th = 1.0   #手高度阈值：用于控制手高度阈值
        reach_th = 0.2   #到达阈值：用于控制到达阈值
        strict_th = 0.15   #严格阈值：用于控制严格阈值

        target_dof_pos_sigma = -20
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        catch_sigma = 5.0

        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        max_contact_force = 1000. # forces above this value are penalized


    class dataset: #TODO: 这个类是用于定义数据集
        folder = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper"
        joint_mapping = "{LEGGED_GYM_ROOT_DIR}/resources/datasets/goalkeeper/joint_id.txt"
        frame_rate = 30
        min_time = 0.1 # sec

    class amp: #TODO: 这个类是用于定义amp

        obs_type = 'dof'
        num_obs = 29 * 2  # (old and new)
        amp_coef = 0.4
        num_steps = 2

class K122CfgPPO( LeggedRobotCfgPPO ): #TODO: 这个类是用于创建PPO算法
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01   #熵系数：用于控制策略的探索程度，越大越鼓励探索，越小策略越保守
    class runner( LeggedRobotCfgPPO.runner ):#TODO: 这个类是用于创建运行器
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'     #HIMPPO：预测模型+环境推断模型+增强PPO算法
        num_steps_per_env = 100 # 每次迭代中环境交互步数
        max_iterations = 200000 # 最大迭代次数

        # logging
        save_interval = 200 # 200次迭代保存一次模型
        run_name = 'goalkeepper'
        experiment_name = 'g1'
        wandb_project = "goalkeepper"
        logger = 'wandb'
        
        # load and resume
        resume = False  #是否从上次保存的模型继续训练
        load_run = -1 # -1 = last run  #从上次保存的模型继续训练
        checkpoint = -1 # -1 = last saved model  #从上次保存的模型继续训练
        resume_path = None # updated from load_run and chkpt  #从上次保存的模型继续训练
    
    amp = K122Cfg.amp #TODO: 这个类是用于定义amp