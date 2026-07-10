# `k1_loco_amp` 代码阅读导图

这份文档把 `k1_loco_amp` 任务的核心代码链路压成一份适合复习的阅读笔记，重点是：

- 先看哪个文件
- 每个文件在整个训练闭环里的位置
- 关键函数负责什么
- 数据是怎么一步一步流动的
- 这一部分最容易出什么问题

如果你想看观测和奖励的表格版整理，请配合阅读：

- `docs/k1_loco_amp_obs_reward.md`

---

## 0. 一句话总览

`k1_loco_amp` 是一个 **速度命令驱动的下肢 locomotion PPO 任务**，同时用 **command-conditioned AMP** 约束动作风格，让机器人既能按命令走，又尽量走得像 expert 数据集。

最核心的数据流可以先记成：

`train.py`
`-> task_registry`
`-> LeggedRobotK1LocoAmp`
`-> actor 输出 12 维 action`
`-> PD 控制成 22 维 torque`
`-> env.step()`
`-> raw reward + amp reward`
`-> rollout storage`
`-> GAE`
`-> PPO + AMP update`

---

## 1. 任务注册与训练入口

### 关键文件

- `legged_gym/legged_gym/scripts/train.py`
- `legged_gym/legged_gym/envs/__init__.py`
- `legged_gym/legged_gym/utils/task_registry.py`

### 关键作用

这一层负责回答两个问题：

1. 命令行里写了 `--task k1_loco_amp`，程序怎么找到对应环境类？
2. 环境和训练器分别是怎么实例化出来的？

### 关键链路

1. `train.py::train()`
   - 调 `task_registry.make_env(name=args.task, args=args)`
   - 调 `task_registry.make_alg_runner(...)`
   - 最后执行 `ppo_runner.learn(...)`

2. `envs/__init__.py`
   - 用 `task_registry.register("k1_loco_amp", LeggedRobotK1LocoAmp, K1LocoAmpCfg(), K1LocoAmpCfgPPO())`
   - 把任务名和环境类、配置绑定

3. `task_registry.py`
   - `make_env()`：根据任务名创建环境
   - `make_alg_runner()`：根据任务名创建 `HIMOnPolicyRunner`

### 最容易出 bug 的地方

- `args.task` 拼错，任务根本没注册
- 改了配置类但忘了在 `__init__.py` 重新注册
- CLI 参数覆盖了默认配置，导致运行行为和静态代码看起来不一致

---

## 2. 环境初始化与 step 主循环

### 关键文件

- `legged_gym/legged_gym/envs/base/base_task.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `legged_gym/legged_gym/envs/base/legged_robot_move_amp_2d.py`
- `legged_gym/legged_gym/envs/base/legged_robot_k1_loco_amp.py`

### 继承关系

`LeggedRobotK1LocoAmp`
`-> LeggedRobotMoveAmp2D`
`-> LeggedRobot`
`-> BaseTask`

### 各层职责

- `BaseTask`
  - Isaac Gym 仿真句柄
  - 通用 buffer
  - viewer / render

- `LeggedRobot`
  - 通用机器人环境逻辑
  - 仿真创建、buffer 初始化、基础 reward 框架
  - 通用 `step()` 主循环

- `LeggedRobotMoveAmp2D`
  - 下肢 12 维动作映射
  - PD 控制
  - lower-body AMP 结构

- `LeggedRobotK1LocoAmp`
  - `k1_loco_amp` 专用命令采样
  - 专用 actor/critic obs
  - 专用 reward
  - command-conditioned AMP motion map

### `step()` 执行顺序

1. actor 给出 `actions`
2. `LeggedRobot.step()` 先 clip action
3. `_compute_torques()` 把 12 维动作变成 22 维 torque
4. Isaac Gym 仿真若干 `decimation` 子步
5. `post_physics_step()`
6. `compute_reward()`
7. `check_termination()`
8. `reset_idx()` 重置 done 环境
9. `compute_observations()` 刷新下一步观测

### 最容易出 bug 的地方

- 误以为 `compute_observations()` 在 `compute_reward()` 前执行
- 忽略 `decimation`，误以为一次 policy step 只模拟一个物理步
- 不清楚子类重写了哪些函数，顺着基类读到一半就“走错类”

---

## 3. 命令采样与动作控制

### 关键文件

- `legged_gym/legged_gym/envs/k1/k1_loco_amp_config.py`
- `legged_gym/legged_gym/envs/base/legged_robot_k1_loco_amp.py`
- `legged_gym/legged_gym/envs/base/legged_robot_move_amp_2d.py`

### 命令采样

环境会在 `standing / forward / backward / leftstep / rightstep / diagonal_left / diagonal_right` 之间按概率采样命令类型，然后为每类命令采样：

- `vx`
- `vy`
- `wz`

当前配置里：

- `train_yaw_command = False`

所以 `wz` 实际上固定为 `0`，当前任务主要训练的是 **平面速度命令跟踪**。

### 动作控制

actor 输出 `12` 维动作，只控制左右腿的 `12` 个关节。

控制链路：

1. `_init_lower_body_action_mapping()`
   - 建立 12 个下肢关节在总 22 个 DOF 里的索引映射

2. `_compute_torques()`
   - `action * action_scale`
   - 加到默认下肢姿态上，形成 `joint_pos_target`
   - 上肢保持默认姿态
   - 用 PD 控制算 torque

### 关键理解

- policy 输出的不是 torque
- policy 输出的是 **关节目标位置偏移**
- 最终 Isaac Gym 执行的是 22 维 torque

### 最容易出 bug 的地方

- 12 维动作顺序和 22 个总 DOF 顺序对不上
- 把 action 当成绝对关节角，而不是相对默认姿态的偏移
- `action_scale` 太大或太小，导致控制极不稳定或动作幅度不足

---

## 4. 观测构造

### 关键文件

- `legged_gym/legged_gym/envs/k1/k1_loco_amp_config.py`
- `legged_gym/legged_gym/envs/base/legged_robot_k1_loco_amp.py`
- `legged_gym/legged_gym/envs/g1/g1_utils.py`
- `rsl_rl/rsl_rl/modules/actor_critic.py`

### 三类观测

1. actor observation
2. privileged critic observation
3. AMP observation

### 维度总表

| 类别 | 维度 | 说明 |
| --- | ---: | --- |
| actor 单步观测 | 65 | 命令 + 角速度 + 重力投影 + 22 维关节位置 + 22 维关节速度 + 12 维动作 |
| actor 历史观测 | 650 | 最近 `10 x 65` 步拼接 |
| critic 特权观测 | 68 | actor 单步观测 `65` + `base_lin_vel(3)` |
| AMP 单步观测 | 32 | 下肢关节位置/速度 + base 速度/角速度 + 重力投影 |
| AMP 判别器输入 | 64 | 连续两步 AMP 单步观测拼接 |

### 为什么 actor 用历史，critic 用特权观测

- actor 用历史：
  - 步态有明显相位信息
  - 单帧观测不足以稳定恢复时序状态
  - 用最近 10 步历史更容易学到 gait

- critic 用特权观测：
  - value 估计希望更准确
  - 多看真实 `base_lin_vel` 更容易估计未来累计回报

### 最容易出 bug 的地方

- 改了观测内容但没同步改 config 里的维度
- 把 actor 的 `650` 维历史观测误当成单步观测
- AMP 观测定义改了，但 `num_obs_per_step / num_obs` 没同步

---

## 5. 奖励函数

### 关键文件

- `legged_gym/legged_gym/envs/k1/k1_loco_amp_config.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `legged_gym/legged_gym/envs/base/legged_robot_k1_loco_amp.py`
- `legged_gym/legged_gym/envs/base/legged_robot_move_amp_2d.py`

### 三类奖励

1. 任务奖励
2. AMP 奖励
3. 正则化奖励

### 任务奖励核心项

- `tracking_lin_vel`
- `tracking_ang_vel`
- `yaw_stability`
- `stand_still`
- `upright`
- `height`

它们主要负责：

- 按命令速度走
- 身体别歪
- yaw 别漂
- standing 命令下别乱抖

### 正则化项核心项

- `feet_slip`
- `ang_vel_xy`
- `dof_acc`
- `smoothness`
- `torques`
- `dof_vel`
- `dof_pos_limits`
- `dof_vel_limits`
- `torque_limits`

它们主要负责：

- 减少滑脚
- 减少抖动
- 抑制暴力控制
- 避免关节/速度/力矩打到极限

### AMP 奖励

AMP 奖励不是在环境的 `_reward_*` 里定义，而是在 runner 里由判别器输出，再与环境 reward 合成。

当前配置下：

- `reward_mode = "additive"`
- `total_reward = raw_reward + amp_scale * amp_reward`

### 最容易出 bug 的地方

- 只看 `scale` 不看公式，误判各项真实作用
- 忘了 `_prepare_reward_function()` 会把权重乘以 `dt`
- 把 AMP 奖励误以为是环境内部算出来的

---

## 6. AMP 数据与判别器

### 关键文件

- `legged_gym/resources/datasets/goalkeeper_from_pkl_k1_locomotion`
- `legged_gym/resources/datasets/goalkeeper_from_pkl_k1_locomotion/summary.json`
- `legged_gym/legged_gym/envs/g1/g1_utils.py`
- `legged_gym/legged_gym/envs/base/legged_robot_k1_loco_amp.py`
- `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
- `rsl_rl/rsl_rl/modules/amp.py`

### expert 数据怎么进系统

1. `load_imitation_dataset()`
   - 读取数据集目录下所有 `.pt`
   - 解析 joint mapping

2. `LeggedRobotK1LocoAmp._reinit_amp_motions_for_lower_body()`
   - 按 `motion_weights` 过滤 motion
   - 为每个 motion 建立一个 `MotionLib`

3. `MotionLib`
   - 把 expert 轨迹整理成可随机采样的下肢状态缓存

### command-conditioned AMP

环境会记录：

- `command_type_ids`

并通过：

- `get_amp_motion_ids()`

把它交给 runner。

runner 再通过：

- `_MultiMotionBuffer`

根据当前命令类别，从对应的 expert motion 中采样 expert AMP obs。

例如：

- `forward -> forward`
- `backward -> backward`
- `leftstep -> leftstep`
- `rightstep -> rightstep`
- `diagonal_left/right -> diagonal`

### AMP 判别器输入输出

- 输入：`64` 维，两步 AMP 状态拼接
- 输出：`1` 维判别分数
- 目标：
  - expert 接近 `+1`
  - policy 接近 `-1`

### 最容易出 bug 的地方

- motion 名称和 command map 对不上
- 条件标签 `amp_motion_ids` 和 agent 样本错位
- expert AMP obs 和 agent AMP obs 不是同一种特征定义

---

## 7. ActorCritic 网络

### 关键文件

- `rsl_rl/rsl_rl/modules/actor_critic.py`
- `rsl_rl/rsl_rl/modules/amp.py`

### `k1_loco_amp` 的真实维度

| 模块 | 输入维度 | 输出维度 |
| --- | ---: | ---: |
| history encoder | 650 | 16 |
| actor 当前 one-step obs | 65 | - |
| actor 最终输入 | 81 | 12 |
| critic | 68 | 1 |
| AMP discriminator | 64 | 1 |

### actor

输入构成：

- 当前 one-step obs：`65`
- history latent：`16`
- `estimate_ball_dim = 0`

所以 actor 输入是：

- `65 + 16 = 81`

actor 输出：

- Gaussian policy 的 `mean`
- `std` 是可训练参数

训练时：

- 从高斯分布采样动作

推理时：

- 使用 `mean action`

### critic

critic 直接吃 `68` 维 privileged observation，输出 `1` 维 value。

### 最容易出 bug 的地方

- 改了观测维度但没同步改网络输入维度
- 忘了 `estimate_ball_dim = 0`，误以为 actor 输入还要额外拼东西
- 推理时还在 sample action，导致策略表现随机抖动

---

## 8. Rollout Storage

### 关键文件

- `rsl_rl/rsl_rl/storage/him_rollout_storage.py`
- `rsl_rl/rsl_rl/algorithms/him_ppo.py`
- `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`

### storage 在整个系统里的位置

它是环境、网络和 PPO 更新之间的缓存桥梁。

一次 rollout 中会把这些数据都存起来：

- actor obs
- critic obs
- next critic obs
- sampled actions
- rewards
- dones
- values
- log prob
- old mean/std
- AMP obs
- AMP motion ids

### 关键对象

- `Transition`
  - 单步临时容器

- `HIMRolloutStorage`
  - 整轮 rollout 的大 buffer

### `num_steps_per_env`

表示：

- 每个并行环境先连续采样多少步

如果：

- `num_envs = N`
- `num_steps_per_env = T`

那么一次 rollout 样本数就是：

- `N x T`

### 最容易出 bug 的地方

- 当前 obs、next obs、value、reward 对齐关系错位
- `amp_motion_ids` 没一起存下来，conditioned AMP 更新会失效
- rollout 打平时搞混 `[T, N, ...]` 和 `[T*N, ...]`

---

## 9. PPO + AMP 更新

### 关键文件

- `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
- `rsl_rl/rsl_rl/algorithms/him_ppo.py`
- `rsl_rl/rsl_rl/storage/him_rollout_storage.py`

### 一轮更新分两阶段

1. rollout 收集
2. 参数更新

### rollout 收集阶段

1. actor 采样 action
2. critic 估 value
3. `env.step(action)`
4. AMP 判别器算 `amp_reward`
5. 合成：
   - `total_reward = raw_reward + amp_scale * amp_reward`
6. 写入 storage

### return / advantage

storage 用 GAE 计算：

- `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`
- `A_t = delta_t + gamma * lam * A_{t+1}`
- `R_t = A_t + V(s_t)`

### PPO 更新核心项

- clipped surrogate loss
- value loss
- entropy bonus
- KL adaptive learning rate

### AMP 更新核心项

- 从 storage 取 `amp_obs_batch`
- 根据 `amp_motion_id_batch` 采对应 expert obs
- 判别器输出：
  - `expert_loss`
  - `policy_loss`
  - `amp_loss`

### smoothness loss

除了环境里的动作平滑 reward 外，`HIMPPO.update()` 里还额外加了基于相邻状态插值的网络平滑正则，约束 actor 和 critic 输出不要随状态微小变化而剧烈跳变。

### 最容易出 bug 的地方

- old log prob / mu / sigma 没保存好，PPO ratio 计算就错
- termination critic obs 没正确替换，GAE 会串台
- 把 AMP reward 和 AMP loss 混在一起理解

---

## 10. 最终闭环总结

### 最简流程图

`obs(650), critic_obs(68), amp_obs(32)`
`-> actor/critic`
`-> action(12), value(1)`
`-> PD 控制`
`-> torque(22)`
`-> env.step()`
`-> raw reward + next obs`
`-> AMP 两步状态(64)`
`-> discriminator -> amp_reward`
`-> total_reward`
`-> storage`
`-> GAE`
`-> PPO + AMP update`

### 一句话理解

`k1_loco_amp` 不是单纯的 imitation，也不是单纯的 command tracking，而是：

- 用 PPO 学会 **按命令稳定走**
- 用 AMP 学会 **走得像 expert**
- 用各种正则项把行为限制在 **更稳定、更自然、更物理合理** 的范围内

---

## 附：建议复习顺序

如果以后要重新看一遍代码，建议按下面顺序：

1. `train.py`
2. `envs/__init__.py`
3. `task_registry.py`
4. `legged_robot_k1_loco_amp.py`
5. `legged_robot_move_amp_2d.py`
6. `k1_loco_amp_config.py`
7. `g1_utils.py`
8. `actor_critic.py`
9. `amp.py`
10. `him_rollout_storage.py`
11. `him_ppo.py`
12. `him_on_policy_runner.py`

如果只想抓主干，优先读：

- `legged_robot_k1_loco_amp.py`
- `k1_loco_amp_config.py`
- `g1_utils.py`
- `actor_critic.py`
- `him_on_policy_runner.py`
- `him_ppo.py`

---

## 附：最值得记住的 10 个关键词

- `task_registry`
- `LeggedRobotK1LocoAmp`
- `command_type_ids`
- `lower_body_dof_indices`
- `65 / 650 / 68 / 64`
- `history_encoder`
- `MotionLib`
- `HIMRolloutStorage`
- `GAE`
- `PPO + AMP`
