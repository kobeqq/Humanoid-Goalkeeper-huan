#!/usr/bin/env python3
"""Play/debug fixed MOVE-STOP-MOVE-STOP commands for k1_goalkeeper_lateral_amp_loco."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import isaacgym  # noqa: F401
import torch
from isaacgym import gymapi, gymutil

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.legged_robot_goalkeeper_lateral_amp_loco import (
    CMD_VX,
    CMD_VY,
    MODE_MOVE,
)
from legged_gym.utils import task_registry


def parse_args() -> Any:
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_goalkeeper_lateral_amp_loco"},
        {"name": "--checkpoint_path", "type": str, "default": ""},
        {"name": "--num_envs", "type": int, "default": 1},
        {"name": "--stage", "type": int, "default": 0},
        {"name": "--sequence", "action": "store_true", "default": True},
        {"name": "--vx", "type": float, "default": 0.0},
        {"name": "--vy", "type": float, "default": 0.25},
        {"name": "--move_duration", "type": float, "default": 3.0},
        {"name": "--stop_duration", "type": float, "default": 1.2},
        {"name": "--duration_s", "type": float, "default": 20.0},
        {"name": "--print_every", "type": int, "default": 50},
        {"name": "--headless", "action": "store_true", "default": False},
    ]
    args = gymutil.parse_arguments(
        description="Play K1 lateral AMP loco with a fixed sequence.",
        custom_parameters=custom_parameters,
    )
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


def disable_play_randomization(env_cfg: Any, args: Any) -> None:
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = max(float(args.duration_s) + 2.0, 16.0)
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "auto_curriculum"):
        env_cfg.auto_curriculum.enable = False
    if hasattr(env_cfg, "amp"):
        env_cfg.amp.enable_discriminator = False
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
    env.compute_observations()
    current = env.obs_buf[:, -env.num_one_step_obs :].clone()
    env.obs_buf[:] = current.repeat(1, env.actor_history_length)
    return env.get_observations()


def main() -> None:
    args = parse_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_play_randomization(env_cfg, args)
    if hasattr(train_cfg, "amp"):
        train_cfg.amp.enable_discriminator = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    if not args.checkpoint_path.strip():
        raise ValueError("--checkpoint_path is required for play.")
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    runner.load(args.checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.begin_eval_sequence(
        env_ids,
        stage_idx=args.stage,
        vx=args.vx,
        vy=args.vy,
        move_duration=args.move_duration,
        stop_duration=args.stop_duration,
    )
    obs = rebuild_obs_history(env)
    num_steps = max(1, int(args.duration_s / env.dt))

    with torch.no_grad():
        for step in range(num_steps):
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
            reset_ids = dones.nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                env.begin_eval_sequence(
                    reset_ids,
                    stage_idx=args.stage,
                    vx=args.vx,
                    vy=args.vy,
                    move_duration=args.move_duration,
                    stop_duration=args.stop_duration,
                )
                obs = rebuild_obs_history(env)

            if args.print_every > 0 and step % args.print_every == 0:
                cmd_vxy = env.commands[0, CMD_VX : CMD_VY + 1]
                base_vxy = env.base_lin_vel[0, :2]
                cmd_norm = torch.norm(cmd_vxy).clamp(min=1e-6)
                proj = torch.sum(base_vxy * (cmd_vxy / cmd_norm))
                progress_ratio = proj / cmd_norm
                tracking_error = torch.norm(base_vxy - cmd_vxy)
                stop_speed = torch.norm(base_vxy)
                active_still = (
                    int(env.command_mode[0].item()) == MODE_MOVE
                    and env.segment_progress[0].item() > 0.20
                    and proj.item()
                    < max(0.08, 0.35 * cmd_norm.item())
                )
                seq_count = max(float(env.episode_sequence_count[0].item()), 1.0)
                seq_rate = float(env.episode_sequence_success_sum[0].item()) / seq_count
                print(
                    f"step={step:05d} "
                    f"stage={int(env.env_curriculum_stage[0].item())} "
                    f"segment_idx={int(env.sequence_segment_idx[0].item())} "
                    f"mode={int(env.command_mode[0].item())} "
                    f"cmd_vxy={cmd_vxy.detach().cpu().numpy()} "
                    f"base_vxy={base_vxy.detach().cpu().numpy()} "
                    f"tracking_error={tracking_error.item():.3f} "
                    f"progress_ratio={progress_ratio.item():.3f} "
                    f"active_still={int(active_still)} "
                    f"stop_speed={stop_speed.item():.3f} "
                    f"sequence_success_rate={seq_rate:.3f} "
                    f"action_norm={actions[0].norm().item():.3f} "
                    f"torque_norm={env.torques[0].norm().item():.3f}"
                )


if __name__ == "__main__":
    main()
