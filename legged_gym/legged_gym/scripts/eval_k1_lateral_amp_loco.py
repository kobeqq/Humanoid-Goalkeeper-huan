#!/usr/bin/env python3
"""Evaluate k1_goalkeeper_lateral_amp_loco by stage on move-stop sequences."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi, gymutil

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry


def parse_args() -> Any:
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_goalkeeper_lateral_amp_loco"},
        {"name": "--checkpoint_path", "type": str, "default": ""},
        {"name": "--num_envs", "type": int, "default": 256},
        {"name": "--num_sequences", "type": int, "default": 1000},
        {"name": "--duration_s", "type": float, "default": 12.0},
        {"name": "--output_dir", "type": str, "default": ""},
        {"name": "--exptid", "type": str, "default": "lateral_amp_eval"},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--headless", "action": "store_true", "default": True},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate K1 lateral AMP loco.",
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
        ("resume", False),
        ("load_run", None),
        ("checkpoint", None),
        ("experiment_name", None),
        ("run_name", None),
        ("max_iterations", None),
        ("max_curriculum_stage", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def disable_randomization(env_cfg: Any, args: Any) -> None:
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


def make_env_policy(args: Any):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_randomization(env_cfg, args)
    if hasattr(train_cfg, "amp"):
        train_cfg.amp.enable_discriminator = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not args.checkpoint_path.strip():
        return env, lambda obs: torch.zeros(env.num_envs, env.num_actions, device=env.device)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(args.checkpoint_path.strip(), load_optimizer=False)
    return env, runner.get_inference_policy(device=env.device)


def stage_default_command(stage: int):
    if stage == 0:
        return 0.0, 0.25
    if stage == 1:
        return 0.0, 0.60
    if stage == 2:
        return 1.0, 0.0
    if stage == 3:
        return -0.7, 0.0
    return 0.5, 0.5


def run_stage(env: Any, policy, stage: int, args: Any) -> Dict[str, float]:
    env_ids = torch.arange(env.num_envs, device=env.device)
    vx, vy = stage_default_command(stage)
    env.reset_idx(env_ids)
    env.begin_eval_sequence(env_ids, stage_idx=stage, vx=vx, vy=vy)
    obs = rebuild_obs_history(env)
    target_sequences = int(args.num_sequences)
    last_count = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    rows: List[Dict[str, float]] = []

    with torch.no_grad():
        while len(rows) < target_sequences:
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
            completed = env.episode_sequence_count > last_count
            if torch.any(completed):
                ids = completed.nonzero(as_tuple=False).flatten()
                for env_id in ids.tolist():
                    seq_count = max(float(env.episode_sequence_count[env_id].item()), 1.0)
                    move_count = max(float(env.episode_move_segment_count[env_id].item()), 1.0)
                    stop_count = max(float(env.episode_stop_segment_count[env_id].item()), 1.0)
                    active_steps = max(float(env.episode_active_tracking_steps[env_id].item()), 1.0)
                    rows.append(
                        {
                            "stage": float(stage),
                            "sequence_success_rate": float(
                                env.episode_sequence_success_sum[env_id].item() / seq_count
                            ),
                            "move_success_rate": float(
                                env.episode_move_success_sum[env_id].item() / move_count
                            ),
                            "stop_success_rate": float(
                                env.episode_stop_success_sum[env_id].item() / stop_count
                            ),
                            "fall_rate": float(env.episode_fall[env_id].item()),
                            "tracking_error_vxy": float(
                                env.episode_tracking_error_sum[env_id].item() / active_steps
                            ),
                            "progress_ratio": float(
                                env.episode_progress_ratio_sum[env_id].item() / active_steps
                            ),
                            "active_still_rate": float(
                                env.episode_active_still_sum[env_id].item() / active_steps
                            ),
                        }
                    )
                    if len(rows) >= target_sequences:
                        break
                last_count[ids] = env.episode_sequence_count[ids]

            reset_ids = dones.nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                env.begin_eval_sequence(reset_ids, stage_idx=stage, vx=vx, vy=vy)
                obs = rebuild_obs_history(env)
                last_count[reset_ids] = env.episode_sequence_count[reset_ids]

    keys = [k for k in rows[0].keys() if k != "stage"]
    return {"stage": float(stage), **{k: sum(row[k] for row in rows) / len(rows) for k in keys}}


def main() -> None:
    args = parse_args()
    env, policy = make_env_policy(args)
    rows = [run_stage(env, policy, stage, args) for stage in range(5)]
    summary = {"task": args.task, "checkpoint_path": args.checkpoint_path, "stages": rows}
    output_dir = Path(
        args.output_dir
        or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "k1_goalkeeper_lateral_amp_loco", args.exptid)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
