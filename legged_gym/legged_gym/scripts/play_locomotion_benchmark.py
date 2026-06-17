import json
import os

import isaacgym  # noqa: F401
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


BENCHMARKS = [
    ("forward", [1.0, 0.0, 0.0, 0.0], 4.0, False),
    ("backward", [-0.8, 0.0, 0.0, 0.0], 4.0, False),
    ("left_lateral", [0.0, 1.2, 0.0, 0.0], 4.0, False),
    ("right_lateral", [0.0, -1.2, 0.0, 0.0], 4.0, False),
    ("diagonal", [0.8, 1.0, 0.0, 0.0], 4.0, False),
    ("rotation", [0.0, 0.0, 0.8, 0.0], 4.0, False),
    ("push_recovery", [0.0, 1.0, 0.0, 0.0], 5.0, True),
    ("vertical_jump", [0.0, 0.0, 0.0, 0.25], 4.0, False),
    ("lateral_jump", [0.0, 1.0, 0.0, 0.25], 4.0, False),
]


def set_command(env, command):
    command_tensor = torch.tensor(command, dtype=torch.float, device=env.device)
    env.commands[:] = command_tensor
    env.command_time_left[:] = 1e9
    if hasattr(env, "_fill_command_history"):
        env._fill_command_history(torch.arange(env.num_envs, device=env.device))


def run_case(env, policy, obs, name, command, duration_s, push):
    set_command(env, command)
    start_height = env.torso_pos[:, 2].clone()
    start_y = env.torso_pos[:, 1].clone()
    peak_height = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    air_time = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    tracking_vx = []
    tracking_vy = []
    tracking_yaw = []
    heading_start = env.yaw.clone()
    pushed = False

    num_steps = int(duration_s / env.dt)
    for step in range(num_steps):
        if push and not pushed and step > num_steps // 4 and hasattr(env, "_push_robots"):
            env._push_robots(torch.arange(env.num_envs, device=env.device))
            pushed = True

        actions = policy(obs.detach())
        obs, _, _, _, _, _, _ = env.step(actions.detach())

        peak_height = torch.maximum(peak_height, torch.clip(env.torso_pos[:, 2] - start_height, min=0.0))
        contact_z = env.contact_forces[:, env.contact_feet_indices, 2]
        airborne = torch.sum(contact_z > 1.0, dim=-1) == 0
        air_time += airborne.float() * env.dt
        tracking_vx.append(torch.abs(env.base_lin_vel[:, 0] - command[0]))
        tracking_vy.append(torch.abs(env.base_lin_vel[:, 1] - command[1]))
        tracking_yaw.append(torch.abs(env.base_ang_vel[:, 2] - command[2]))

    tracking_vx = torch.stack(tracking_vx)
    tracking_vy = torch.stack(tracking_vy)
    tracking_yaw = torch.stack(tracking_yaw)
    lateral_distance = torch.abs(env.torso_pos[:, 1] - start_y)
    heading_drift = torch.abs(torch.atan2(torch.sin(env.yaw - heading_start), torch.cos(env.yaw - heading_start)))

    result = {
        "name": name,
        "command": {
            "vx": command[0],
            "vy": command[1],
            "yaw": command[2],
            "jump_height": command[3],
        },
        "tracking_error_vx": tracking_vx.mean().item(),
        "tracking_error_vy": tracking_vy.mean().item(),
        "tracking_error_yaw": tracking_yaw.mean().item(),
        "heading_drift_rad": heading_drift.mean().item(),
        "jump_peak_height": peak_height.mean().item(),
        "air_time": air_time.mean().item(),
        "lateral_jump_distance": lateral_distance.mean().item(),
        "fall_rate": env.reset_buf.float().mean().item(),
    }
    return obs, result


def run_direction_switch(env, policy, obs):
    sequence = [1.5, -1.5, 1.5, -1.5]
    errors = []
    switch_times = []

    for vy in sequence:
        set_command(env, [0.0, vy, 0.0, 0.0])
        reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        elapsed = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        for _ in range(int(1.0 / env.dt)):
            actions = policy(obs.detach())
            obs, _, _, _, _, _, _ = env.step(actions.detach())
            elapsed += (~reached).float() * env.dt
            reached |= torch.sign(torch.tensor(vy, device=env.device)) * env.base_lin_vel[:, 1] >= 0.9 * abs(vy)
            errors.append(torch.abs(env.base_lin_vel[:, 1] - vy))
        switch_times.append(elapsed)

    return obs, {
        "name": "direction_switch",
        "command_sequence_vy": sequence,
        "tracking_error_vy": torch.stack(errors).mean().item(),
        "switch_time": torch.stack(switch_times).mean().item(),
        "fall_rate": env.reset_buf.float().mean().item(),
    }


def play_locomotion_benchmark(args):
    if args.task == "29":
        args.task = "k1_move_amp"
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    results = []
    for name, command, duration_s, push in BENCHMARKS[:4]:
        obs, result = run_case(env, policy, obs, name, command, duration_s, push)
        results.append(result)

    obs, switch_result = run_direction_switch(env, policy, obs)
    results.append(switch_result)

    for name, command, duration_s, push in BENCHMARKS[4:]:
        obs, result = run_case(env, policy, obs, name, command, duration_s, push)
        results.append(result)

    output_path = os.path.join(LEGGED_GYM_ROOT_DIR, "benchmark.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "results": results}, f, indent=2)
    print(f"Wrote locomotion benchmark to {output_path}")


if __name__ == "__main__":
    play_locomotion_benchmark(get_args())
