#!/usr/bin/env python3
"""Fixed-command viewer/debug script for k1_goalkeeper_lower_loco.

Place this file under legged_gym/legged_gym/scripts/ or run it from the repo root
where the legged_gym package is importable.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import isaacgym  # noqa: F401; must be imported before torch/legged_gym env creation
import torch
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.envs.base.legged_robot_goalkeeper_lower_loco import (
    CMD_VX,
    CMD_VY,
    MODE_LOCO,
    MODE_STAND,
)
from legged_gym.utils import task_registry


def parse_args() -> Any:
    default_ckpt = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "k1_goalkeeper_lower_loco",
        "lower_loco_auto_s0_to_s4",
        "model_4600.pt",
    )
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_goalkeeper_lower_loco"},
        {"name": "--checkpoint_path", "type": str, "default": default_ckpt},
        {"name": "--num_envs", "type": int, "default": 1},
        {"name": "--duration_s", "type": float, "default": 20.0},
        {"name": "--vx", "type": float, "default": 0.60},
        {"name": "--vy", "type": float, "default": 0.00},
        {"name": "--mode", "type": str, "default": "loco", "help": "loco or stand"},
        {"name": "--disable_disturbance", "action": "store_true", "default": True},
        {"name": "--print_every", "type": int, "default": 50},
        {"name": "--headless", "action": "store_true", "default": False},
    ]
    args = gymutil.parse_arguments(
        description="Play k1_goalkeeper_lower_loco with a fixed command.",
        custom_parameters=custom_parameters,
    )

    # Make this script compatible with task_registry helpers.
    for name, value in (
        ("sim_device", getattr(args, "rl_device", "cuda:0")),
        ("rl_device", getattr(args, "sim_device", "cuda:0")),
        ("physics_engine", gymapi.SIM_PHYSX),
        ("use_gpu", True),
        ("use_gpu_pipeline", True),
        ("num_threads", 0),
        ("subscenes", 0),
        ("seed", 1),
        ("resume", False),
        ("load_run", None),
        ("checkpoint", None),
        ("experiment_name", None),
        ("run_name", None),
        ("exptid", "debug_play"),
        ("resumeid", None),
        ("max_iterations", None),
        ("max_curriculum_stage", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def disable_play_randomization(env_cfg: Any, num_envs: int, duration_s: float) -> None:
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = max(float(duration_s) + 2.0, 10.0)
    env_cfg.env.play = True
    env_cfg.env.play_fixed_command = True
    env_cfg.noise.add_noise = False

    if hasattr(env_cfg, "auto_curriculum"):
        env_cfg.auto_curriculum.enable = False
    if hasattr(env_cfg, "upper_body_disturbance"):
        env_cfg.upper_body_disturbance.enable = False

    for name in (
        "push_robots",
        "randomize_friction",
        "randomize_joint_injection",
        "randomize_actuation_offset",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_link_mass",
        "randomize_restitution",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "continue_keep",
        "delay",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def rebuild_obs_history(env: Any) -> torch.Tensor:
    """Recompute current one-step observation and fill all history slots with it."""
    env.compute_observations()
    current = env.obs_buf[:, -env.num_one_step_obs :].clone()
    env.obs_buf[:] = current.repeat(1, env.actor_history_length)
    return env.get_observations()


def inject_fixed_command(env: Any, mode: int, vx: float, vy: float, duration_s: float) -> torch.Tensor:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.begin_eval_command(
        env_ids,
        mode=mode,
        vx=float(vx),
        vy=float(vy),
        goal_z=None,
        duration_s=float(duration_s),
        disturbance_strength=0.0,
    )
    return rebuild_obs_history(env)


def main() -> None:
    args = parse_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_play_randomization(env_cfg, args.num_envs, args.duration_s)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    runner.load(args.checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    mode = MODE_STAND if args.mode.lower() == "stand" else MODE_LOCO
    if mode == MODE_STAND:
        args.vx = 0.0
        args.vy = 0.0

    obs = inject_fixed_command(env, mode, args.vx, args.vy, args.duration_s)
    num_steps = max(1, int(args.duration_s / env.dt))

    with torch.no_grad():
        for i in range(num_steps):
            actions = policy(obs.detach())
            step_ret = env.step(actions.detach())
            obs = step_ret[0]
            dones = step_ret[3]

            reset_ids = dones.nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                env.begin_eval_command(
                    reset_ids,
                    mode=mode,
                    vx=float(args.vx),
                    vy=float(args.vy),
                    goal_z=None,
                    duration_s=float(args.duration_s),
                    disturbance_strength=0.0,
                )
                obs = rebuild_obs_history(env)

            if args.print_every > 0 and i % args.print_every == 0:
                cmd_vxy = env.commands[0, CMD_VX:CMD_VY + 1]
                base_vxy = env.base_lin_vel[0, :2]
                cmd_norm = torch.norm(cmd_vxy).clamp(min=1e-6)
                proj = torch.sum(base_vxy * (cmd_vxy / cmd_norm))
                progress_ratio = proj / cmd_norm
                cfg = env.cfg.rewards
                min_required_speed = max(
                    float(getattr(cfg, "active_still_speed_abs", 0.08)),
                    float(getattr(cfg, "active_still_speed_ratio", 0.35)) * cmd_norm.item(),
                )
                active_still = (
                    int(env.command_mode[0].item()) == MODE_LOCO
                    and env.command_phase[0].item()
                    > float(getattr(cfg, "active_still_grace_phase", 0.15))
                    and proj.item() < min_required_speed
                )
                print(
                    f"step={i:05d} "
                    f"stage={int(env.env_curriculum_stage[0].item())} "
                    f"mode={int(env.command_mode[0].item())} "
                    f"cmd_vxy={cmd_vxy.detach().cpu().numpy()} "
                    f"base_vxy={base_vxy.detach().cpu().numpy()} "
                    f"progress_ratio={progress_ratio.item():.3f} "
                    f"active_still={int(active_still)} "
                    f"action_norm={actions[0].norm().item():.3f} "
                    f"torque_norm={env.torques[0].norm().item():.3f}"
                )


if __name__ == "__main__":
    main()
