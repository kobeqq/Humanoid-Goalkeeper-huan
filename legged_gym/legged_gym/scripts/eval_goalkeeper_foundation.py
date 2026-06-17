#!/usr/bin/env python3
"""Evaluation for k1_goalkeeper_foundation (velocity + height + recover FSM)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi, gymutil

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.legged_robot_goalkeeper_foundation import (
    PHASE_EXECUTE,
    PHASE_RECOVER,
    PHASE_STABLE,
)
from legged_gym.utils import task_registry


Policy = Callable[[torch.Tensor], torch.Tensor]

# (name, stage_idx, vx, vy, goal_z_offset, T_execute)
# goal_z_offset=None -> use standing_goal_z at reset
EVAL_CASES = [
    ("stage0_forward", 0, 0.6, 0.0, None, 4.0),
    ("stage1_lateral", 1, 0.0, 0.6, None, 4.0),
    ("stage2_omni", 2, 0.5, 0.5, None, 4.0),
    ("stage3_fast_omni", 3, 1.0, 0.8, None, 3.0),
    ("stage4_slow_height", 4, 0.2, 0.0, 0.65, 4.0),
    ("stage5_omni_height", 5, 0.8, 0.6, 0.70, 4.0),
]


def parse_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_goalkeeper_foundation"},
        {"name": "--checkpoint_path", "type": str, "default": ""},
        {"name": "--exptid", "type": str, "default": "k1_goalkeeper_foundation"},
        {"name": "--resumeid", "type": str, "default": ""},
        {"name": "--num_episodes", "type": int, "default": 0},
        {"name": "--curriculum_stage", "type": int, "default": -1},
        {"name": "--output_dir", "type": str, "default": ""},
        {"name": "--promote", "action": "store_true", "default": False},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--headless", "action": "store_true", "default": True},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate goalkeeper foundation locomotion policy.",
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
        ("headless", True),
        ("num_envs", 1),
        ("resume", False),
        ("load_run", None),
        ("checkpoint", None),
        ("experiment_name", None),
        ("run_name", None),
        ("max_iterations", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def disable_randomization(env_cfg, episode_length_s: float) -> None:
    env_cfg.env.num_envs = 1
    env_cfg.env.play = True
    env_cfg.env.episode_length_s = episode_length_s
    env_cfg.noise.add_noise = False
    dr = env_cfg.domain_rand
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
        if hasattr(dr, name):
            setattr(dr, name, False)


def make_env_and_policy(args) -> Tuple[Any, Policy, torch.Tensor, Any]:
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_randomization(env_cfg, env_cfg.env.episode_length_s)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    def zero_policy(obs_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.num_actions, device=env.device)

    checkpoint_path = args.checkpoint_path.strip()
    if not checkpoint_path:
        return env, zero_policy, obs, train_cfg

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    if checkpoint_path:
        runner.load(checkpoint_path, load_optimizer=False)
    runner_policy = runner.get_inference_policy(device=env.device)
    policy = lambda obs_tensor: runner_policy(obs_tensor)
    return env, policy, obs, train_cfg


def reset_eval_state(env) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()


def run_episode(
    env,
    policy: Policy,
    obs: torch.Tensor,
    case_name: str,
    stage_idx: int,
    vx: float,
    vy: float,
    goal_z: float,
    t_execute: float,
    eval_cfg,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = env.device
    reset_eval_state(env)
    env_ids = torch.arange(env.num_envs, device=device)
    env.set_max_curriculum_stage(stage_idx)
    env.begin_eval_execute(env_ids, stage_idx, vx, vy, goal_z, t_execute)
    obs = env.get_observations()

    sigma_v = getattr(env.cfg.rewards, "tracking_sigma_v", 0.25)
    sigma_h = getattr(env.cfg.rewards, "tracking_sigma_h", 0.05)
    hold_steps = int(getattr(eval_cfg, "execute_hold_steps", 25))
    recover_hold_steps = int(getattr(eval_cfg, "recover_hold_steps", 25))

    fell = False
    execute_ok_steps = 0
    recover_ok_steps = 0
    saw_execute = False
    saw_recover = False
    execute_success = False
    recover_success = False
    tracking_vxy_sum = 0.0
    tracking_z_sum = 0.0
    tracking_steps = 0
    duration_s = 0.0

    max_steps = int(env.max_episode_length) + 1
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
            duration_s += env.dt

            if bool(dones[0].item()):
                fell = True
                break

            phase = int(env.episode_phase[0].item())
            if phase == PHASE_EXECUTE:
                saw_execute = True
                vxy_err = torch.norm(
                    env.base_lin_vel[0, :2] - env.commands[0, :2]
                ).item()
                z_err = abs(env.torso_pos[0, 2].item() - env.commands[0, 2].item())
                tracking_vxy_sum += vxy_err
                tracking_z_sum += z_err
                tracking_steps += 1
                if vxy_err < 2.0 * sigma_v and z_err < 2.0 * sigma_h:
                    execute_ok_steps += 1
            elif phase == PHASE_RECOVER:
                saw_recover = True
                if bool(env._is_stable()[0].item()):
                    recover_ok_steps += 1

        if execute_ok_steps >= hold_steps:
            execute_success = True
        if recover_ok_steps >= recover_hold_steps:
            recover_success = True

    tracking_vxy = tracking_vxy_sum / max(tracking_steps, 1)
    tracking_z = tracking_z_sum / max(tracking_steps, 1)

    result = {
        "case": case_name,
        "stage": stage_idx,
        "vx": vx,
        "vy": vy,
        "goal_z": goal_z,
        "T_execute": t_execute,
        "duration_s": duration_s,
        "fall": float(fell),
        "saw_execute": float(saw_execute),
        "saw_recover": float(saw_recover),
        "execute_success": float(execute_success),
        "recover_success": float(recover_success),
        "tracking_error_vxy": tracking_vxy,
        "tracking_error_z": tracking_z,
    }
    return obs, result


def aggregate_results(rows: List[Dict[str, float]], eval_cfg) -> Dict[str, Any]:
    n = max(len(rows), 1)
    no_fall_rate = 1.0 - sum(r["fall"] for r in rows) / n
    task_success_rate = sum(r["execute_success"] for r in rows) / n
    recover_success_rate = sum(r["recover_success"] for r in rows) / n
    mean_duration = sum(r["duration_s"] for r in rows) / n
    mean_tracking_vxy = sum(r["tracking_error_vxy"] for r in rows) / n
    mean_tracking_z = sum(r["tracking_error_z"] for r in rows) / n

    summary = {
        "num_episodes": len(rows),
        "no_fall_rate": no_fall_rate,
        "task_success_rate": task_success_rate,
        "recover_success_rate": recover_success_rate,
        "mean_duration_s": mean_duration,
        "mean_tracking_error_vxy": mean_tracking_vxy,
        "mean_tracking_error_z": mean_tracking_z,
    }

    summary["promotion_pass"] = (
        no_fall_rate >= eval_cfg.min_no_fall_rate
        and task_success_rate >= eval_cfg.min_task_success_rate
        and recover_success_rate >= eval_cfg.min_recover_success_rate
        and mean_duration >= eval_cfg.min_episode_duration_s
        and mean_tracking_vxy <= eval_cfg.max_tracking_error_vxy
        and mean_tracking_z <= eval_cfg.max_tracking_error_z
    )
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    env, policy, obs, train_cfg = make_env_and_policy(args)
    eval_cfg = env.cfg.eval
    num_episodes = args.num_episodes or eval_cfg.num_episodes

    if args.curriculum_stage >= 0:
        env.set_max_curriculum_stage(args.curriculum_stage)
        cases = [c for c in EVAL_CASES if c[1] == args.curriculum_stage]
        if not cases:
            raise ValueError(f"No eval case for curriculum stage {args.curriculum_stage}")
    else:
        cases = [c for c in EVAL_CASES if c[1] <= env.max_curriculum_stage]

    output_dir = Path(
        args.output_dir
        or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "foundation_eval", args.exptid)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for ep in range(num_episodes):
        case = cases[ep % len(cases)]
        name, stage_idx, vx, vy, goal_z_offset, t_execute = case
        reset_eval_state(env)
        standing_z = float(env.standing_goal_z[0].item())
        goal_z = standing_z if goal_z_offset is None else float(goal_z_offset)
        obs, row = run_episode(
            env,
            policy,
            obs,
            name,
            stage_idx,
            vx,
            vy,
            goal_z,
            t_execute,
            eval_cfg,
        )
        row["episode"] = ep
        rows.append(row)

    summary = aggregate_results(rows, eval_cfg)
    summary["max_curriculum_stage"] = int(env.max_curriculum_stage)
    summary["cases"] = [c[0] for c in cases]

    write_json(output_dir / "episodes.json", {"episodes": rows})
    write_json(output_dir / "summary.json", summary)

    if args.promote and summary["promotion_pass"]:
        next_stage = min(env.max_curriculum_stage + 1, 5)
        stage_path = output_dir / "curriculum_stage.json"
        write_json(
            stage_path,
            {
                "max_curriculum_stage": next_stage,
                "previous_stage": int(env.max_curriculum_stage),
                "summary": summary,
            },
        )
        print(f"Promotion: stage {env.max_curriculum_stage} -> {next_stage}")
        print(f"Wrote {stage_path}")

    print(f"Saved evaluation to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
