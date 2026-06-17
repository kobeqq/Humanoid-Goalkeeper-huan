#!/usr/bin/env python3
"""Goal-target viewer/demo script for k1_goalkeeper_lower_loco.

Unlike velocity-command play scripts, this task uses fixed body-frame goal targets
(front/back/left/right/stand/diagonal). Run from repo root or legged_gym/ with the
package on PYTHONPATH.
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
from legged_gym.envs.base.legged_robot_goalkeeper_lower_loco import DIRECTION_NAME_TO_ID
from legged_gym.utils import get_load_path, task_registry

# Preset demos: (direction, x_mag, y_mag) in body frame (x forward, y left).
DEMO_TARGETS = {
    "stand": ("stand", 0.0, 0.0),
    "front": ("front", 1.00, 0.0),
    "back": ("back", 1.00, 0.0),
    "left": ("left", 0.0, 1.00),
    "right": ("right", 0.0, 1.00),
    "front_left": ("front_left", 0.80, 0.80),
    "front_right": ("front_right", 0.80, 0.80),
    "back_left": ("back_left", 0.80, 0.80),
    "back_right": ("back_right", 0.80, 0.80),
}

SEQUENCE_ORDER = (
    "stand",
    "front",
    "left",
    "right",
    "back",
    "front_left",
    "front_right",
)


def parse_args() -> Any:
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_goalkeeper_lower_loco"},
        {
            "name": "--checkpoint_path",
            "type": str,
            "default": "",
            "help": "Full path to model_*.pt; if empty, resolve from --exptid.",
        },
        {"name": "--exptid", "type": str, "default": "debug"},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "-1 = latest"},
        {"name": "--num_envs", "type": int, "default": 1},
        {
            "name": "--target",
            "type": str,
            "default": "front",
            "help": f"One of {sorted(DEMO_TARGETS.keys()) + ['sequence']}",
        },
        {"name": "--local_x", "type": float, "default": -1.0, "help": "Override x magnitude (<0 = use preset)"},
        {"name": "--local_y", "type": float, "default": -1.0, "help": "Override y magnitude (<0 = use preset)"},
        {"name": "--deadline_s", "type": float, "default": -1.0, "help": "Target deadline (<0 = auto from stage)"},
        {"name": "--stage", "type": int, "default": 0, "help": "Curriculum stage for reach/yaw thresholds"},
        {"name": "--duration_s", "type": float, "default": 60.0, "help": "Viewer run length (seconds)"},
        {"name": "--sequence_interval_s", "type": float, "default": 8.0, "help": "Cycle interval when --target=sequence"},
        {"name": "--max_steps", "type": int, "default": 0, "help": "If >0, stop after this many steps (headless smoke)"},
        {"name": "--print_every", "type": int, "default": 50},
        {"name": "--headless", "action": "store_true", "default": False},
    ]
    args = gymutil.parse_arguments(
        description="Play k1_goalkeeper_lower_loco with fixed goal targets.",
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
        ("experiment_name", None),
        ("run_name", None),
        ("resumeid", None),
        ("max_iterations", None),
        ("max_curriculum_stage", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def resolve_checkpoint(args: Any, experiment_name: str) -> str:
    if args.checkpoint_path:
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        return args.checkpoint_path
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", experiment_name, args.exptid)
    return get_load_path(log_root, checkpoint=args.checkpoint)


def disable_play_randomization(env_cfg: Any, num_envs: int, duration_s: float) -> None:
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = max(float(duration_s) + 5.0, 15.0)
    env_cfg.env.play = True
    env_cfg.env.play_fixed_command = True
    env_cfg.noise.add_noise = False

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


def direction_name(direction_id: int) -> str:
    for name, idx in DIRECTION_NAME_TO_ID.items():
        if idx == int(direction_id):
            return name
    return str(int(direction_id))


def inject_goal_target(
    env: Any,
    direction: str,
    x_mag: float,
    y_mag: float,
    deadline_s: float | None,
    stage: int,
) -> torch.Tensor:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.begin_eval_goal_target(
        env_ids,
        direction=direction,
        x_mag=x_mag,
        y_mag=y_mag,
        deadline_s=deadline_s,
        stage=stage,
        disturbance_strength=0.0,
    )
    return rebuild_obs_history(env)


def resolve_target_spec(args: Any, sequence_index: int = 0) -> tuple[str, float, float]:
    if args.target.lower() == "sequence":
        key = SEQUENCE_ORDER[sequence_index % len(SEQUENCE_ORDER)]
    else:
        key = args.target.lower()
    if key not in DEMO_TARGETS:
        valid = sorted(DEMO_TARGETS.keys()) + ["sequence"]
        raise ValueError(f"Unknown --target {args.target!r}; choose one of {valid}")

    direction, x_mag, y_mag = DEMO_TARGETS[key]
    if args.local_x >= 0.0:
        x_mag = float(args.local_x)
    if args.local_y >= 0.0:
        y_mag = float(args.local_y)
    return direction, x_mag, y_mag


def print_status(env: Any, step: int, label: str) -> None:
    dist = env._target_xy_distance()[0].item()
    yaw_err = env._yaw_error_to_target()[0].item()
    yaw_deg = abs(yaw_err) * 180.0 / 3.14159265
    height = env.root_states[0, 2].item()
    reached = int(env.target_reached[0].item() > 0.5)
    success = int(env.episode_target_success[0].item() > 0.5)
    time_left = env.target_time_left[0].item()
    direction = direction_name(int(env.target_direction[0].item()))
    print(
        f"step={step:05d} target={label} dir={direction} "
        f"dist={dist:.3f}m yaw_err={yaw_deg:.1f}deg height={height:.3f}m "
        f"reached={reached} success={success} time_left={time_left:.1f}s"
    )


def main() -> None:
    args = parse_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_play_randomization(env_cfg, args.num_envs, args.duration_s)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    checkpoint_path = resolve_checkpoint(args, train_cfg.runner.experiment_name)
    print(f"[play] loading checkpoint: {checkpoint_path}")

    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    deadline_s = None if args.deadline_s < 0.0 else float(args.deadline_s)
    direction, x_mag, y_mag = resolve_target_spec(args)
    target_label = args.target.lower()
    obs = inject_goal_target(env, direction, x_mag, y_mag, deadline_s, args.stage)
    print(
        f"[play] goal target={target_label} direction={direction} "
        f"local=({x_mag:.2f}, {y_mag:.2f}) stage={args.stage} "
        f"deadline={'auto' if deadline_s is None else f'{deadline_s:.1f}s'}"
    )
    print_status(env, 0, target_label)

    num_steps = args.max_steps if args.max_steps > 0 else max(1, int(args.duration_s / env.dt))
    sequence_index = 0
    steps_since_switch = 0
    switch_interval = max(1, int(args.sequence_interval_s / env.dt))

    with torch.no_grad():
        for i in range(1, num_steps + 1):
            actions = policy(obs.detach())
            step_ret = env.step(actions.detach())
            obs = step_ret[0]
            dones = step_ret[3]

            reset_ids = dones.nonzero(as_tuple=False).flatten()
            if len(reset_ids) > 0:
                direction, x_mag, y_mag = resolve_target_spec(args, sequence_index)
                env.begin_eval_goal_target(
                    reset_ids,
                    direction=direction,
                    x_mag=x_mag,
                    y_mag=y_mag,
                    deadline_s=deadline_s,
                    stage=args.stage,
                    disturbance_strength=0.0,
                )
                obs = rebuild_obs_history(env)

            if args.target.lower() == "sequence":
                steps_since_switch += 1
                if steps_since_switch >= switch_interval:
                    sequence_index += 1
                    steps_since_switch = 0
                    direction, x_mag, y_mag = resolve_target_spec(args, sequence_index)
                    target_label = SEQUENCE_ORDER[sequence_index % len(SEQUENCE_ORDER)]
                    env_ids = torch.arange(env.num_envs, device=env.device)
                    env.begin_eval_goal_target(
                        env_ids,
                        direction=direction,
                        x_mag=x_mag,
                        y_mag=y_mag,
                        deadline_s=deadline_s,
                        stage=args.stage,
                        disturbance_strength=0.0,
                    )
                    obs = rebuild_obs_history(env)
                    print(f"[play] switched to target={target_label} direction={direction}")
                    print_status(env, i, target_label)

            if args.print_every > 0 and i % args.print_every == 0:
                print_status(env, i, target_label)


if __name__ == "__main__":
    main()
