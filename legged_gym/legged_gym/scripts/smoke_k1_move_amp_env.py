"""Lightweight smoke test for env startup and optional single PPO update."""
import argparse
import os
import sys
import tempfile

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import isaacgym  # noqa: F401 — must import before torch
from legged_gym.envs import *  # noqa: F401, F403
import torch

from legged_gym.utils import get_args, set_seed, task_registry
from legged_gym.utils.helpers import update_cfg_from_args


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--task",
        type=str,
        default="k1_move_amp",
        choices=["k1_move_amp", "k1_loco_amp", "k1_goalkeeper_foundation", "k1_goalkeeper_lower_loco", "k1"],
    )
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--runner-update", action="store_true")
    return parser.parse_known_args()


def main():
    cli, _ = parse_args()
    sys.argv = [sys.argv[0]]
    args = get_args()
    args.task = cli.task
    args.num_envs = cli.num_envs
    args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    set_seed(env_cfg.seed)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    env.reset()
    for _ in range(cli.steps):
        obs, priv_obs, rew, done, extras, _, _ = env.step(actions)
        assert obs.shape[-1] == env.num_obs, (
            f"obs dim {obs.shape[-1]} != num_obs {env.num_obs}"
        )
        if priv_obs is not None:
            assert priv_obs.shape[-1] == env.num_privileged_obs, (
                f"privileged obs dim {priv_obs.shape[-1]} != {env.num_privileged_obs}"
            )

    print(
        f"OK: task={args.task} num_envs={env.num_envs} steps={cli.steps} "
        f"obs={env.num_obs} priv={env.num_privileged_obs} amp={env.num_amp_obs} "
        f"max_curriculum_stage={getattr(env, 'max_curriculum_stage', 'n/a')}"
    )
    if hasattr(env, "curriculum_stage_specs"):
        unlocked = min(env.max_curriculum_stage + 1, getattr(env, "num_curriculum_stages", 6))
        print("Curriculum stages unlocked:", unlocked)
        for i in range(unlocked):
            spec = env.curriculum_stage_specs[i]
            duration = spec.get("duration", spec.get("T_execute", [0, 0]))
            print(
                f"  stage{i}: vx={spec['vx']} vy={spec['vy']} "
                f"goal_z_mode={spec['goal_z_mode']} duration={duration} "
                f"modes={spec.get('mode_probs', {})}"
            )
    if hasattr(env, "reward_names"):
        print("Active rewards:", env.reward_names)

    if cli.runner_update:
        train_cfg.runner.resume = False
        train_cfg.runner.max_iterations = 1
        train_cfg.runner.save_interval = 1000
        train_cfg.runner.log_interval = 1
        train_cfg.runner.logger = "tensorboard"
        train_cfg.algorithm.num_learning_epochs = 1
        train_cfg.algorithm.num_mini_batches = 1
        train_cfg.runner.num_steps_per_env = min(int(cli.steps), 8)
        log_root = tempfile.mkdtemp(prefix="k1_amp_smoke_", dir="/tmp")
        os.makedirs(log_root, exist_ok=True)
        runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
            log_root=log_root,
        )
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        print(f"Runner update OK: log_root={log_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
