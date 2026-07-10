# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math
import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_LEGGED_GYM_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_ROOT = os.path.dirname(_LEGGED_GYM_PACKAGE_ROOT)
for _path in (_LEGGED_GYM_PACKAGE_ROOT, os.path.join(_REPO_ROOT, "rsl_rl")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import isaacgym
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_jit_to_onnx,
    load_onnx_policy,
    task_registry,
)
import torch
import faulthandler

EXPORT_POLICY = False

LOWER_LOCO_PLAY_VX = 0.3
LOWER_LOCO_PLAY_VY = 0.0
LOWER_LOCO_CMD_DURATION_S = 60.0
OMNI_PLAY_RADIUS = 2.0
OMNI_DIRECTION_NAMES = ("front", "left", "back", "right")
OMNI_DIRECTION_CENTERS = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
PLAY_CAMERA_OFFSET = np.array([2.0, -2.0, 1.2], dtype=np.float32)
PLAY_CAMERA_TARGET_HEIGHT = 0.9


def configure_play_env(env_cfg, task_name: str, num_envs_override=None):
    env_cfg.env.num_envs = 1 if num_envs_override is None else int(num_envs_override)
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    if task_name == "k1_goalkeeper_lower_loco":
        env_cfg.env.episode_length_s = 8
        env_cfg.env.play_fixed_command = True
        if hasattr(env_cfg, "auto_curriculum"):
            env_cfg.auto_curriculum.enable = False
        if hasattr(env_cfg, "upper_body_disturbance"):
            env_cfg.upper_body_disturbance.enable = False
    elif task_name == "k1_omni_move_amp":
        env_cfg.env.episode_length_s = 6
        env_cfg.commands.curriculum_enable = False
        env_cfg.commands.reset_on_success = False
        env_cfg.commands.stage_allowed_bins = [[0, 1, 2, 3]]
        env_cfg.commands.stage_radius_ranges = [[OMNI_PLAY_RADIUS, OMNI_PLAY_RADIUS]]
        env_cfg.commands.yaw_ref_world = True
        env_cfg.commands.yaw_ref = 0.0
    elif task_name == "k1_loco_amp":
        env_cfg.env.episode_length_s = 6
        env_cfg.domain_rand.randomize_joint_injection = False
        env_cfg.domain_rand.randomize_actuation_offset = False
        env_cfg.domain_rand.randomize_payload_mass = False
        env_cfg.domain_rand.randomize_com_displacement = False
        env_cfg.domain_rand.randomize_link_mass = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.randomize_kp = False
        env_cfg.domain_rand.randomize_kd = False
        env_cfg.domain_rand.delay = False
        env_cfg.domain_rand.push_robots = False
    else:
        env_cfg.env.episode_length_s = 3
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.push_interval_s = 6


def _get_camera_target(env):
    if hasattr(env, "rigid_body_states") and hasattr(env, "torso_index"):
        target = env.rigid_body_states[0, env.torso_index, 0:3].detach().cpu().numpy()
    elif hasattr(env, "root_states"):
        target = env.root_states[0, 0:3].detach().cpu().numpy()
    else:
        target = np.zeros(3, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).copy()
    target[2] = max(float(target[2]), PLAY_CAMERA_TARGET_HEIGHT)
    return target


def update_play_camera(env):
    if not getattr(env, "viewer", None):
        return
    target = _get_camera_target(env)
    camera_pos = target + PLAY_CAMERA_OFFSET
    env.gym.viewer_camera_look_at(
        env.viewer,
        None,
        gymapi.Vec3(*camera_pos.tolist()),
        gymapi.Vec3(*target.tolist()),
    )


def sync_play_obs(env):
    """Rebuild obs_buf so injected commands appear in actor history."""
    env.compute_observations()
    one_step = env.obs_buf[:, -env.num_one_step_obs :].clone()
    env.obs_buf = one_step.repeat(1, env.actor_history_length)
    return env.get_observations()


def get_fixed_play_command(args, default=None):
    values = [
        getattr(args, "play_vx", None),
        getattr(args, "play_vy", None),
        getattr(args, "play_wz", None),
    ]
    if all(value is None for value in values):
        return default
    return tuple(0.0 if value is None else float(value) for value in values)


def inject_lower_loco_command(env, env_ids, vx=LOWER_LOCO_PLAY_VX, vy=LOWER_LOCO_PLAY_VY):
    if not hasattr(env, "begin_eval_command"):
        return
    try:
        from legged_gym.envs.base.legged_robot_goalkeeper_lower_loco import MODE_LOCO
    except ModuleNotFoundError:
        MODE_LOCO = 1
    env.begin_eval_command(
        env_ids,
        MODE_LOCO,
        vx,
        vy,
        goal_z=None,
        duration_s=LOWER_LOCO_CMD_DURATION_S,
        disturbance_strength=0.0,
    )


def infer_loco_amp_command_name(command):
    vx, vy, wz = [float(value) for value in command]
    if max(abs(vx), abs(vy), abs(wz)) < 0.05:
        return "standing"
    if vx > 0.05 and abs(vy) > 0.05:
        return "diagonal_left" if vy > 0.0 else "diagonal_right"
    if abs(vx) >= abs(vy):
        return "forward" if vx >= 0.0 else "backward"
    return "leftstep" if vy >= 0.0 else "rightstep"


def set_loco_amp_command_type(env, env_ids, command):
    if not hasattr(env, "command_type_ids"):
        return
    command_names = getattr(env, "amp_command_names", None)
    if command_names is None and hasattr(env.cfg.commands, "motion_commands"):
        command_names = list(env.cfg.commands.motion_commands.keys())
    if not command_names:
        return
    command_name = infer_loco_amp_command_name(command)
    if command_name in command_names:
        env.command_type_ids[env_ids] = int(command_names.index(command_name))


def inject_loco_amp_commands(env, env_ids=None, command=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    if command is None:
        base_cmds = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [-0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, -0.25, 0.0],
                [0.25, 0.20, 0.0],
            ],
            dtype=torch.float,
            device=env.device,
        )
        repeat = (len(env_ids) + base_cmds.shape[0] - 1) // base_cmds.shape[0]
        cmds = base_cmds.repeat(repeat, 1)[: len(env_ids)]
    else:
        cmd = torch.tensor(command, dtype=torch.float, device=env.device).view(1, 3)
        cmds = cmd.repeat(len(env_ids), 1)
        set_loco_amp_command_type(env, env_ids, command)
    env.commands[env_ids] = cmds
    env.command_time_left[env_ids] = 999.0


def omni_direction_bins(env):
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long) % len(OMNI_DIRECTION_NAMES)


def inject_omni_targets(env, env_ids, all_direction_bins, radius=OMNI_PLAY_RADIUS):
    if len(env_ids) == 0 or not hasattr(env, "target_pos"):
        return

    if hasattr(env, "rigid_body_states") and hasattr(env, "torso_index"):
        env.torso_pos = env.rigid_body_states[:, env.torso_index, 0:3]

    direction_bins = all_direction_bins[env_ids]
    centers = torch.tensor(OMNI_DIRECTION_CENTERS, dtype=torch.float, device=env.device)
    headings = centers[direction_bins]
    target_xy = torch.stack(
        (radius * torch.cos(headings), radius * torch.sin(headings)),
        dim=-1,
    )

    if hasattr(env, "yaw_ref"):
        env.yaw_ref[env_ids] = 0.0
    if hasattr(env, "target_heading"):
        env.target_heading[env_ids] = headings
    if hasattr(env, "target_dir_bin"):
        env.target_dir_bin[env_ids] = direction_bins

    env.target_pos[env_ids, :2] = env.torso_pos[env_ids, :2] + target_xy
    if hasattr(env, "_uses_target_z") and not env._uses_target_z():
        env.target_pos[env_ids, 2] = env.torso_pos[env_ids, 2]


def play(args):
    faulthandler.enable()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_play_env(env_cfg, args.task, num_envs_override=args.num_envs)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    fixed_play_command = get_fixed_play_command(args)

    if args.task == "k1_goalkeeper_lower_loco":
        env_ids = torch.arange(env.num_envs, device=env.device)
        lower_loco_command = get_fixed_play_command(
            args,
            default=(LOWER_LOCO_PLAY_VX, LOWER_LOCO_PLAY_VY, 0.0),
        )
        inject_lower_loco_command(env, env_ids, vx=lower_loco_command[0], vy=lower_loco_command[1])
        obs = sync_play_obs(env)
        print(
            f"[play] fixed command vx={lower_loco_command[0]}, vy={lower_loco_command[1]}, "
            f"mode=LOCO, num_envs={env.num_envs}"
        )
    elif args.task == "k1_omni_move_amp":
        env_ids = torch.arange(env.num_envs, device=env.device)
        omni_bins = omni_direction_bins(env)
        inject_omni_targets(env, env_ids, omni_bins)
        obs = sync_play_obs(env)
        labels = [OMNI_DIRECTION_NAMES[int(i)] for i in omni_bins.detach().cpu().tolist()]
        print(f"[play] k1_omni_move_amp fixed targets radius={OMNI_PLAY_RADIUS}, dirs={labels}")
    elif args.task == "k1_loco_amp":
        inject_loco_amp_commands(env, command=fixed_play_command)
        obs = sync_play_obs(env)
        if fixed_play_command is None:
            print("[play] k1_loco_amp fixed commands:", env.commands[: min(env.num_envs, 10)].tolist())
        else:
            print(f"[play] k1_loco_amp uniform command: {fixed_play_command}, num_envs={env.num_envs}")
    else:
        obs = env.get_observations()

    update_play_camera(env)

    if EXPORT_POLICY:
        policy_name = args.task
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, policy_name)
        print("Exported policy as jit script to: ", path)

        jit_path = os.path.join(path, f"{policy_name}.pt")
        jit_model = torch.jit.load(jit_path)
        dummy_input = torch.randn(1, obs.shape[1], device="cpu")
        onnx_path = os.path.join(path, f"{policy_name}.onnx")
        export_jit_to_onnx(jit_model, onnx_path, dummy_input)
        policy = load_onnx_policy(onnx_path)

    for step_i in range(3000):
        actions = policy(obs.detach())
        if args.task == "k1_goalkeeper_lower_loco" and step_i == 0:
            print(
                f"[play] action norm={actions.norm(dim=-1).mean().item():.4f}, "
                f"cmd vxy={env.commands[0, :2].tolist()}"
            )
        step_ret = env.step(actions.detach())
        obs = step_ret[0]
        dones = step_ret[3]
        update_play_camera(env)

        if args.task == "k1_goalkeeper_lower_loco" and torch.any(dones):
            reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            inject_lower_loco_command(env, reset_ids, vx=lower_loco_command[0], vy=lower_loco_command[1])
            obs = sync_play_obs(env)
        elif args.task == "k1_omni_move_amp" and torch.any(dones):
            reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            inject_omni_targets(env, reset_ids, omni_bins)
            obs = sync_play_obs(env)
        elif args.task == "k1_loco_amp" and torch.any(dones):
            reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            inject_loco_amp_commands(env, reset_ids, command=fixed_play_command)
            obs = sync_play_obs(env)


if __name__ == "__main__":
    args = get_args()
    play(args)
