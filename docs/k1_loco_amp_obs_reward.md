# `k1_loco_amp` 观测与奖励整理

这份说明对应任务 `k1_loco_amp`，实现主线如下：

- 配置: `legged_gym/legged_gym/envs/k1/k1_loco_amp_config.py`
- 环境: `legged_gym/legged_gym/envs/base/legged_robot_k1_loco_amp.py`
- 父类补充: `legged_gym/legged_gym/envs/base/legged_robot_move_amp_2d.py`
- 基类正则项: `legged_gym/legged_gym/envs/base/legged_robot.py`
- AMP 奖励混合: `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`

## 1. 观测

### 1.1 普通观测

`k1_loco_amp` 的单步普通观测由 `LeggedRobotK1LocoAmp._build_actor_one_step_obs()` 构造。

对应代码:

- `legged_robot_k1_loco_amp.py::_build_actor_one_step_obs`
- `k1_loco_amp_config.py::env.num_one_step_observations`

单步维度一共是 `65` 维:

| 模块 | 维度 | 内容 |
| --- | ---: | --- |
| 命令观测 | 3 | `commands * command_scale`，即 `vx_cmd, vy_cmd, wz_cmd` |
| 基座角速度 | 3 | `base_ang_vel` |
| 重力投影 | 3 | `projected_gravity` |
| 关节位置 | 22 | `(dof_pos - default_dof_pos)` |
| 关节速度 | 22 | `dof_vel` |
| 上一时刻动作 | 12 | `actions`，这里只保留下肢 12 维动作 |
| 总计 | 65 | `3 + 3 + 3 + 22 + 22 + 12` |

这里有两个要点:

- 机器人总 DOF 还是 `22`，所以关节位置和速度观测都是 `22` 维。
- 策略输出动作只有下肢 `12` 维，所以最后一段 action 观测是 `12` 维，不是 `22` 维。

最终 actor 输入不是单步 `65` 维，而是 history 堆叠后的:

- `num_actor_history = 10`
- `num_observations = 10 x 65 = 650`

也就是说，策略实际输入是最近 `10` 步普通观测拼接后的 `650` 维向量。

### 1.2 特权观测

`k1_loco_amp` 的特权观测由 `LeggedRobotK1LocoAmp._build_privileged_obs()` 构造。

对应代码:

- `legged_robot_k1_loco_amp.py::_build_privileged_obs`
- `k1_loco_amp_config.py::env.num_privileged_obs`

特权观测是在普通观测基础上额外拼接 `base_lin_vel`:

| 模块 | 维度 |
| --- | ---: |
| 普通观测 actor obs | 65 |
| 基座线速度 `base_lin_vel` | 3 |
| 总计 | 68 |

所以 critic 使用的特权观测是 `68` 维。

### 1.3 AMP 观测

AMP 观测不属于 actor/critic 普通输入，它是单独提供给判别器的模仿状态。

对应代码:

- `legged_robot_k1_loco_amp.py::get_amp_observations`
- `g1_utils.py::build_lower_body_amp_step_obs`
- `k1_loco_amp_config.py::amp`

单步 AMP 观测是 `32` 维:

| 模块 | 维度 | 内容 |
| --- | ---: | --- |
| 下肢关节位置 | 12 | `q_leg` |
| 下肢关节速度 | 12 | `dq_leg` |
| 基座线速度 XY | 2 | `base_lin_vel[:, :2]` |
| 基座角速度 | 3 | `base_ang_vel` |
| 重力投影 | 3 | `projected_gravity` |
| 总计 | 32 | `12 + 12 + 2 + 3 + 3` |

配置中:

- `num_steps = 2`
- `num_obs_per_step = 32`
- `num_obs = 64`

训练时 runner 会把连续两步 AMP 状态拼起来送给判别器，所以判别器最终输入是 `64` 维。

## 2. 奖励

`k1_loco_amp` 的总奖励可以分成三部分:

1. 环境原始奖励
2. AMP 奖励
3. 正则化奖励

其中环境里的 `compute_reward()` 只负责原始环境奖励和正则化奖励；AMP 奖励是在 runner 里和环境奖励合成。

---

### 2.1 任务奖励

这些项主要负责“按命令稳定行走”。

| 奖励名称 | 公式 |
| --- | --- |
| `tracking_lin_vel` | `exp(-sum((commands_xy - base_lin_vel_xy)^2) / tracking_sigma)` |
| `tracking_ang_vel` | `exp(-((commands_wz - base_ang_vel_z)^2) / yaw_rate_sigma)` |
| `yaw_stability` | `exp(-(yaw_error^2) / yaw_sigma)` |
| `stand_still` | `1[||commands_xy|| < 0.05] * exp(-sum(dof_vel_lower_body^2) * stand_still_sigma)` |
| `upright` | `exp(-3 * sum(projected_gravity_xy^2))` |
| `height` | `exp(-((torso_z - target_base_z)^2) / height_sigma)` |

补充说明:

- `yaw_error = wrap_to_pi(yaw - reset_yaw)`
- 当前配置里 `train_yaw_command = False`，所以 `tracking_ang_vel` 里的 `commands_wz` 实际上是 `0`

---

### 2.2 AMP 奖励

AMP 奖励来自判别器，不在环境的 `_reward_*` 函数里定义。

对应代码:

- `him_on_policy_runner.py`
- `legged_robot_move_amp_2d.py::record_amp_reward`

关键配置:

- `enable_discriminator = True`
- `reward_mode = "additive"`
- `amp_scale = 2.0`
- `amp_coef = 0.45`
- `adaptive_amp_scale = True`

| 奖励名称 | 公式 |
| --- | --- |
| `amp_reward` | `0.5 * discriminator_predict_reward(amp_state_two_steps)` |
| `total_reward` | `raw_rewards + amp_scale * amp_reward` |

补充说明:

- `raw_rewards` 是环境里算出来的任务奖励和正则化奖励之和
- `amp_scale` 会在 `adaptive_amp_scale=True` 时动态调整
- `amp_coef = 0.45` 会传给 AMP 模块本身，但当前 `reward_mode="additive"` 时最终混合公式用的是 `amp_scale`

---

### 2.3 正则化奖励

这些项主要用于抑制暴力控制、极限动作和不稳定行为。

| 奖励名称 | 公式 |
| --- | --- |
| `feet_slip` | `sum((||foot_vel_xy||^2) * 1[foot_contact])` |
| `ang_vel_xy` | `sum(base_ang_vel_xy^2)` |
| `dof_acc` | `sum(((last_dof_vel - dof_vel) / dt)^2)` |
| `smoothness` | `sum((actions - 2 * last_actions + last_last_actions)^2)` |
| `torques` | `sum((torques / p_gains)^2)` |
| `dof_vel` | `sum(dof_vel^2)` |
| `dof_pos_limits` | `sum(-(dof_pos - lower_limit).clip(max=0) + (dof_pos - upper_limit).clip(min=0))` |
| `dof_vel_limits` | `sum((abs(dof_vel) - dof_vel_limits * soft_dof_vel_limit).clip(min=0))` |
| `torque_limits` | `sum((abs(torques) - torque_limits * soft_torque_limit).clip(min=0))` |

## 3. 最终整理

### 3.1 观测分类

| 类别 | 维度 | 内容 |
| --- | ---: | --- |
| 普通观测 | 65 | 命令 + 角速度 + 重力投影 + 22 维关节位置 + 22 维关节速度 + 12 维动作 |
| 普通观测历史堆叠 | 650 | `10 x 65` |
| 特权观测 | 68 | 普通观测 + `base_lin_vel(3)` |
| AMP 单步观测 | 32 | 下肢状态 + 基座速度/角速度 + 重力投影 |
| AMP 判别器输入 | 64 | 连续两步 AMP 观测拼接 |

### 3.2 奖励分类

| 类别 | 奖励项 |
| --- | --- |
| 任务奖励 | `tracking_lin_vel`, `tracking_ang_vel`, `yaw_stability`, `stand_still`, `upright`, `height` |
| AMP 奖励 | 判别器输出的 `amp_reward` |
| 正则化奖励 | `feet_slip`, `ang_vel_xy`, `dof_acc`, `smoothness`, `torques`, `dof_vel`, `dof_pos_limits`, `dof_vel_limits`, `torque_limits` |

## 4. 一句话理解

`k1_loco_amp` 本质上是在做:

- 用普通观测和特权观测学会“按速度命令稳定走”
- 用 AMP 奖励把动作风格拉向数据集
- 用正则项压住打滑、抖动、暴力输出和越限

所以它不是单纯的 imitation，也不是单纯的 command tracking，而是两者叠加的 locomotion AMP 任务。
