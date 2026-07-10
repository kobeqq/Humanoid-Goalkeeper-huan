# k1_loco_amp：8600 iteration 后 AMP 步态失败分析

## 结论

当前问题不是 gait phase reward 太小，而是 AMP 风格通道本身存在实现和数据表达缺陷。`model_8600.pt` 的前向策略已经能跟踪速度，但它通过明显的双脚腾空和躯干上下弹跳完成任务。优先级如下：

1. AMP normalizer 用已归一化数据更新统计量，判别器输入尺度长期漂移。
2. checkpoint 没有保存 AMP discriminator、AMP normalizer 和 AMP scale；恢复训练时 actor 被保留，但风格教师随机重置。
3. AMP obs 不含 base height、base vertical velocity、foot position/velocity/contact，判别器无法直接识别 hopping。
4. expert 数据量和方向质量不足；forward 只有 2 段共 3.07 s，rightstep 只有 1 段，diagonal_right 错用只有正 y 的 diagonal 数据。
5. task tracking 信号比当前 AMP 有效贡献更强，而且允许用冲击式推地获得水平速度；当前没有 feet-air-time、双脚腾空或 base vertical motion 约束。
6. 前向约 25.1% 的采样步至少有一个关节达到 torque limit，低阻尼 PD 会放大冲击，但它更像放大器，不是第一根因。

因此不建议继续增加 gait phase reward。应先修 AMP 实现、checkpoint、数据和观测，再让 AMP 承担风格学习。

## 1. Reward 进入 PPO 的真实路径

当前配置：

```text
reward_mode = mixture
amp_coef = 0.20
final_reward = 0.80 * raw_task_reward + 0.20 * raw_amp_reward
```

环境中的所有 task reward scale 会先乘 policy `dt=0.02`。日志中的 `Episode/rew_*` 又除以 `dt`，因此 `tracking_lin_vel≈4.17` 是每秒 reward rate；它对应每步约 `4.17*0.02=0.0834`。进入 PPO 后 tracking 部分约为 `0.8*0.0834=0.0667/step`。`amp_reward≈0.166` 已是每步 raw AMP reward，进入 PPO 后约为 `0.2*0.166=0.0332/step`。

仅比较这两项，tracking 是 AMP 的约 2.0 倍。8600 checkpoint 的前向 clean-eval raw task reward 为 `0.11754/step`；若用日志中的 `0.166` 作为同期 AMP 近似，则：

| 项目 | 进入 PPO 的每步贡献 | 绝对贡献占比 |
|---|---:|---:|
| Task | 0.80 × 0.11754 = 0.09403 | 73.9% |
| AMP | 0.20 × 0.166 = 0.03320 | 26.1% |

这说明 AMP 不是完全没有权重，但不足以主导风格，而且其实现缺陷使这 26% 未必代表可靠的“像人”信号。历史 0~8600 的精确占比不能从 `model_8600.pt` 还原，因为 checkpoint 不含 discriminator/normalizer，工作区也没有对应 TensorBoard event。runner 已新增每个 rollout 的 raw task、raw AMP、加权贡献和 AMP 绝对占比日志，后续训练可以精确回答该问题。

## 2. AMP normalizer 与 checkpoint

已确认原实现顺序错误：

```text
raw obs -> normalize -> discriminator
normalized obs -> normalizer.update()   # 错误
```

已改成：

```text
raw obs -> normalizer.update()
raw obs -> normalize -> discriminator
```

同时 checkpoint 现在保存/恢复：

- actor-critic；
- AMP discriminator；
- AMP normalizer 的 mean/var/count；
- AMP scale；
- optimizer。

旧 checkpoint 缺失 AMP 状态时会明确警告，并跳过不再匹配的 optimizer restore。`model_8600.pt` 只能安全用于 actor playback 或 actor warm-start，不能视为可无损恢复的 AMP 训练 checkpoint。

## 3. command 到 expert motion 的映射

运行时映射如下：

| command | motion_id | motion_name | 判断 |
|---|---:|---|---|
| standing | 0 | standing | 正确 |
| forward | 1 | forward | 名称正确 |
| backward | 2 | backward | 名称正确，但数据方向混杂 |
| leftstep | 3 | leftstep | 名称正确，只有 2 段 |
| rightstep | 4 | rightstep | 名称正确，只有 1 段 |
| diagonal_left | 5 | diagonal | 正确 |
| diagonal_right | 6 | diagonal | 错误：同一批 expert 全是正 y diagonal |

forward 确实取 forward expert，不是映射错位。但 forward expert 只有 2 段、总计 3.07 s，其中一段方向角约 -16.6°、yaw range 约 19.0°，覆盖不足。当前 diagonal 8 段的方向角均约 `+27°~+35°`，不能直接监督 `diagonal_right`。

建议先将 `diagonal_right` 从训练分布移除，直到生成经过左右镜像且 joint/base/foot 坐标全部一致的 right-diagonal expert；不要让它回退到 unconditional expert。

## 4. AMP obs 是否足够

当前单帧 32 维：

```text
12 q + 12 dq + base_vxy(2) + base angular velocity(3) + gravity(3)
```

runner 拼两帧得到 64 维。它缺少：

- base height 和 base vertical velocity；
- 两脚相对 pelvis/torso 的位置和速度；
- foot height；
- contact state。

因此垂直弹跳对 discriminator 几乎不可见：`base_lin_vel` 明确只取 xy。两帧 q/dq 能间接表达腿部周期，但无法可靠区分“交替支撑的人形走路”和“相似关节摆动下的双脚起跳”。

建议的修改顺序不是一次全加：

1. 先加入 `base_z relative to nominal`、`base_vz`、左右脚相对 base 的 position/velocity；
2. 确认 expert 与 policy 使用完全相同的 body frame 和时间间隔；
3. 只有在 expert contact 可稳定重建后再加入 contact bit；
4. 修正 expert base angular velocity 当前 world/body frame 不一致，以及 expert 两帧间隔随机而 policy 间隔固定的问题。

## 5. model_8600 clean-eval 实测

设置：固定命令、4 个并行 episode、每个 6 s、去掉前 1 s、contact threshold 1 N、关闭训练随机化。

| command | tracking error | 双脚接触 | 双脚离地 | base z std | base vz RMS | vertical acc RMS | 任一关节饱和的步比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| forward | 0.0564 m/s | 59.1% | **16.6%** | **0.0319 m** | **0.3700 m/s** | **6.541 m/s²** | **25.1%** |
| backward | 0.0561 | 77.2% | 0.4% | 0.0096 | 0.1093 | 3.557 | 0.4% |
| leftstep | 0.0518 | 67.0% | 4.8% | 0.0070 | 0.1552 | 5.716 | 21.4% |
| rightstep | 0.0745 | 57.7% | 0.3% | 0.0067 | 0.1021 | 3.630 | 29.6% |
| diag left | 0.0545 | 67.3% | 1.5% | 0.0072 | 0.1223 | 3.550 | 29.7% |
| diag right | 0.0552 | 49.0% | 1.3% | 0.0061 | 0.1054 | 3.034 | 31.4% |

前向左右脚接触比例为 77.8%/64.7%；平均 stance 为 0.486/0.404 s，平均 swing 为 0.159/0.252 s，左右明显不对称。双脚离地 16.6% 不是 shuffle，而是明确 hopping。forward expert 的 base z std、vz RMS、vertical acceleration RMS 分别约为 0.0078 m、0.0518 m/s、0.756 m/s²；policy 分别约为 expert 的 4.1、7.1、8.7 倍。

前向 torque：3.63% 的 joint-samples 超过 90% limit，1.48% 精确饱和，但 25.1% 的 control steps 至少有一个关节饱和。主要高负载关节是双侧 hip pitch/roll。当前下肢 stiffness 80、damping 2，属于偏低阻尼；建议先把 hip/knee damping 做小范围 2→3~4 的控制实验，同时保持 policy/reward 不变，判断 PD 放大占比。

## 6. Feet air time 与 yaw

当前任务没有 `feet_air_time`、`step_time` 或 contact-aware swing reward。可以新增，但不能使用“腾空越久越好”的无上限 air-time reward，否则会继续鼓励 hopping。更合理的是：

- 只在单脚 touchdown 时奖励位于目标区间的上一段 swing time；
- stand command 时关闭；
- 对双脚同时离地单独惩罚；
- 不用固定 gait clock 指定哪只脚必须摆动。

前向 heading 与实际 velocity direction 的平均误差约 9.6°，yaw 绝对均值约 3.0°，所以前向 hopping 不是靠旋转身体完成的。但 backward/left/right/diagonal 的 yaw 绝对均值约 44.3°/31.2°/16.7°/52°，说明全向任务确实利用了自由 body rotation。不要恢复 yaw termination；应加入 reset heading 的 soft dead-zone penalty，并单独记录 heading-command/world-velocity 关系。

## 7. 新的 reward 架构

建议总结构：

```text
R = R_task + lambda_amp * R_amp + R_regularization
```

### Task：只负责完成命令

| reward | 建议 |
|---|---|
| tracking_lin_vel | 4.5 降到约 2.5~3.0 |
| tracking_ang_vel | 保留 0.1；无 yaw command 时只抑制 wz |
| stand_still | 保留 0.3，只对 stand 生效 |
| gait_phase | 删除/设为 0 |

### AMP：负责风格

建议改为 additive + adaptive scale，先以 AMP 的绝对贡献占总 reward magnitude 的 35%~45% 为目标，而不是固定 `amp_coef=0.2`。起始建议：

```text
reward_mode = additive
adaptive_amp_scale = true
amp_target_fraction = 0.40
amp_scale_min = 0.20
amp_scale_max = 1.00
amp_scale_ema_alpha = 0.02
```

这只应在 normalizer、checkpoint、obs frame、direction mapping 和 expert 数据修好后启用。

### Regularization：只负责稳定与物理合理性

| reward/penalty | 建议 |
|---|---|
| upright | 1.0 降到 0.5~0.8，避免正奖励总量继续压 AMP |
| height | 0.5 降到 0.2~0.3，并新增 base_vz penalty |
| feet_slip | -0.05 增强到约 -0.1 |
| torques | 保留或从 -1e-5 小幅增到 -2e-5 |
| torque_limits | 暂保留 -0.2；先改 PD/动作冲击，不因日志偏高而降低保护 |
| smoothness/dof_acc | 保留，观察与 AMP 的冲突后再调 |
| bilateral_flight | 新增轻度惩罚，直接针对双脚离地 |
| bounded feet air time | 新增小权重、touchdown 触发、目标区间型 |
| base_vz | 新增平方惩罚 |
| vertical acceleration | 可新增很小权重，避免把接触冲击噪声放大 |
| soft heading | 新增 10°~15° dead zone 外的二次惩罚，不 termination |

## 8. 建议加入 TensorBoard/WandB 的统计量

- `raw_task_reward_mean_per_step`、`raw_amp_reward_mean_per_step`；
- `weighted_task_contribution`、`weighted_amp_contribution`、`amp_abs_fraction`；
- 每个 reward term 的 raw、scaled、进入 PPO 后贡献；
- discriminator expert/policy logits、分类准确率、gradient penalty；
- AMP normalizer mean/std/min/max 及 clip ratio；
- command、motion_id、motion_name 的采样计数；
- left/right/both/no-contact ratio；
- left/right stance/swing duration；
- base height std、base vz RMS、vertical acceleration RMS；
- torque >90% ratio、exact saturation ratio、steps-with-any-saturation；
- yaw from reset、heading-vs-world-velocity angle、heading-vs-command angle；
- 按 direction 分组的上述全部指标，不能只报全局平均。

原始评估数据位于 `logs/diagnostics/model_8600_amp_failure_analysis/fixed_commands/`。
