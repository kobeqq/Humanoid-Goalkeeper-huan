#!/usr/bin/env python3
"""Evaluate k1_goalkeeper_lower_loco on fixed velocity commands (V2 API)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi, gymutil

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.legged_robot_goalkeeper_lower_loco import (
    CMD_VX,
    CMD_VY,
    MODE_LOCO,
    MODE_STAND,
)
from legged_gym.utils import task_registry


Policy = Callable[[torch.Tensor], torch.Tensor]

EVAL_CASES = [
    ("stand", MODE_STAND, 0.0, 0.0),
    ("forward", MODE_LOCO, 0.3, 0.0),
    ("backward", MODE_LOCO, -0.2, 0.0),
    ("left", MODE_LOCO, 0.0, 0.3),
    ("right", MODE_LOCO, 0.0, -0.3),
    ("diag_left", MODE_LOCO, 0.3, 0.3),
    ("diag_right", MODE_LOCO, 0.3, -0.3),
    ("fast_left", MODE_LOCO, 0.0, 0.5),
    ("fast_right", MODE_LOCO, 0.0, -0.5),
]


def parse_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_goalkeeper_lower_loco"},
        {"name": "--checkpoint_path", "type": str, "default": ""},
        {"name": "--exptid", "type": str, "default": "k1_goalkeeper_lower_loco"},
        {"name": "--output_dir", "type": str, "default": ""},
        {"name": "--num_episodes", "type": int, "default": 0},
        {"name": "--num_envs", "type": int, "default": 1},
        {"name": "--duration_s", "type": float, "default": 0.0},
        {"name": "--disturbance", "type": str, "default": "both"},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--headless", "action": "store_true", "default": True},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate K1 lower-body locomotion policy.",
        custom_parameters=custom_parameters,
    )
    if not hasattr(args, "sim_device"):
        args.sim_device = args.rl_device
    if not hasattr(args, "rl_device"):
        args.rl_device = args.sim_device
    for name, value in (
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


def disable_randomization(env_cfg, args) -> None:
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "auto_curriculum"):
        env_cfg.auto_curriculum.enable = False
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


def make_env_and_policy(args) -> Tuple[Any, Policy, torch.Tensor]:
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_randomization(env_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if not args.checkpoint_path.strip():
        return env, lambda obs_tensor: torch.zeros(env.num_envs, env.num_actions, device=env.device), obs

    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(args.checkpoint_path.strip(), load_optimizer=False)
    return env, runner.get_inference_policy(device=env.device), obs


def reset_eval_state(env) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()


def set_fixed_command(env, mode: int, vx: float, vy: float, duration_s: float, disturbance: bool) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    strength = 1.0 if disturbance else 0.0
    env.begin_eval_command(env_ids, mode, vx, vy, goal_z=None, duration_s=duration_s, disturbance_strength=strength)


def step_metrics(env) -> Dict[str, float]:
    leg_ids = env.leg_joint_indices
    contact = env.contact_forces[:, env.contact_feet_indices, 2] > 1.0
    foot_vel = env.rigid_body_states[:, env.contact_feet_indices, 7:10]
    foot_slip = torch.sum(
        torch.square(torch.norm(foot_vel[:, :, :2], dim=-1)) * contact.float(),
        dim=1,
    )
    torque_ratio = torch.abs(env.torques[:, leg_ids]) / env.torque_limits[leg_ids].clamp(min=1e-6)
    dof_pos = env.dof_pos[:, leg_ids]
    limits = env.dof_pos_limits[leg_ids]
    if hasattr(env, "_active_loco_progress"):
        active, cmd_norm, proj, progress_ratio = env._active_loco_progress()
        active_count = active.float().sum().clamp(min=1.0)
        active_progress_ratio = torch.sum(progress_ratio * active.float()) / active_count
        cfg = env.cfg.rewards
        min_required_speed = torch.maximum(
            torch.full_like(cmd_norm, float(getattr(cfg, "active_still_speed_abs", 0.08))),
            float(getattr(cfg, "active_still_speed_ratio", 0.35)) * cmd_norm,
        )
        still = active & (
            env.command_phase > float(getattr(cfg, "active_still_grace_phase", 0.15))
        ) & (proj < min_required_speed)
        active_still_rate = torch.sum(still.float()) / active_count
    else:
        active_progress_ratio = torch.tensor(0.0, device=env.device)
        active_still_rate = torch.tensor(0.0, device=env.device)
    return {
        "progress_ratio": active_progress_ratio.item(),
        "active_still_rate": active_still_rate.item(),
        "curriculum_stage": env.env_curriculum_stage.float().mean().item(),
        "roll_sq": torch.square(env.roll).mean().item(),
        "pitch_sq": torch.square(env.pitch).mean().item(),
        "base_height_error": torch.abs(env.torso_pos[:, 2] - env.standing_goal_z).mean().item(),
        "torque_saturation_fraction": (torque_ratio > 0.98).float().mean().item(),
        "dof_limit_fraction": ((dof_pos < limits[:, 0]) | (dof_pos > limits[:, 1])).float().mean().item(),
        "foot_slip_mean": foot_slip.mean().item(),
    }


def run_episode(
    env,
    policy: Policy,
    obs: torch.Tensor,
    case_name: str,
    mode: int,
    vx: float,
    vy: float,
    duration_s: float,
    disturbance: bool,
):
    reset_eval_state(env)
    set_fixed_command(env, mode, vx, vy, duration_s, disturbance)
    obs = env.get_observations()
    tracking_errors = []
    metric_rows: List[Dict[str, float]] = []
    fall_fraction = 0.0
    elapsed_s = 0.0

    with torch.no_grad():
        for _ in range(max(1, int(duration_s / env.dt))):
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
            elapsed_s += env.dt
            cmd_vxy = env.commands[:, CMD_VX : CMD_VY + 1]
            tracking_errors.append(
                torch.norm(env.base_lin_vel[:, :2] - cmd_vxy, dim=-1).mean().item()
            )
            metric_rows.append(step_metrics(env))
            fall_fraction = max(fall_fraction, env.fall_buf.float().mean().item())
            if torch.any(dones > 0) and torch.any(env.fall_buf):
                break

    n = max(len(metric_rows), 1)
    result = {
        "case": case_name,
        "mode": int(mode),
        "vx": vx,
        "vy": vy,
        "disturbance": float(disturbance),
        "duration_s": elapsed_s,
        "fall_rate": fall_fraction,
        "no_fall_rate": 1.0 - fall_fraction,
        "tracking_error_vxy": sum(tracking_errors) / max(len(tracking_errors), 1),
    }
    result["progress_ratio"] = sum(row["progress_ratio"] for row in metric_rows) / n
    result["active_still_rate"] = sum(row["active_still_rate"] for row in metric_rows) / n
    result["curriculum_stage"] = sum(row["curriculum_stage"] for row in metric_rows) / n
    result["roll_rms"] = (sum(row["roll_sq"] for row in metric_rows) / n) ** 0.5
    result["pitch_rms"] = (sum(row["pitch_sq"] for row in metric_rows) / n) ** 0.5
    for key in ("base_height_error", "torque_saturation_fraction", "dof_limit_fraction", "foot_slip_mean"):
        result[key] = sum(row[key] for row in metric_rows) / n
    return obs, result


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    n = max(len(rows), 1)
    keys = (
        "no_fall_rate",
        "tracking_error_vxy",
        "progress_ratio",
        "active_still_rate",
        "curriculum_stage",
        "roll_rms",
        "pitch_rms",
        "base_height_error",
        "torque_saturation_fraction",
        "dof_limit_fraction",
        "foot_slip_mean",
        "duration_s",
    )
    return {"num_episodes": len(rows), **{key: sum(row[key] for row in rows) / n for key in keys}}


def disturbance_groups(mode: str) -> List[Tuple[str, bool]]:
    if mode == "off":
        return [("no_disturbance", False)]
    if mode == "on":
        return [("disturbance", True)]
    if mode == "both":
        return [("no_disturbance", False), ("disturbance", True)]
    raise ValueError("--disturbance must be one of: off, on, both")


def main() -> None:
    args = parse_args()
    env, policy, obs = make_env_and_policy(args)
    duration_s = args.duration_s or float(env.cfg.env.episode_length_s)
    num_episodes = args.num_episodes or max(50, len(EVAL_CASES))
    output_dir = Path(
        args.output_dir
        or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "lower_loco_eval", args.exptid)
    )

    rows: List[Dict[str, float]] = []
    for group_name, disturbance in disturbance_groups(args.disturbance):
        for ep in range(num_episodes):
            case_name, mode, vx, vy = EVAL_CASES[ep % len(EVAL_CASES)]
            obs, row = run_episode(env, policy, obs, case_name, mode, vx, vy, duration_s, disturbance)
            row["episode"] = ep
            row["group"] = group_name
            rows.append(row)

    summary = {
        "task": args.task,
        "checkpoint_path": args.checkpoint_path,
        "duration_s": duration_s,
        "num_envs": args.num_envs,
        "overall": aggregate(rows),
        "groups": {
            group: aggregate([row for row in rows if row["group"] == group])
            for group, _ in disturbance_groups(args.disturbance)
        },
        "cases": {
            case: aggregate([row for row in rows if row["case"] == case])
            for case, _, _, _ in EVAL_CASES
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes.json").write_text(json.dumps({"episodes": rows}, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved evaluation to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
