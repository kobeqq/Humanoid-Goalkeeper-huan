#!/usr/bin/env python3
"""Policy playback diagnostics for velocity-command locomotion tasks.

This script runs trained policies in clean eval conditions, forces fixed
commands every step, records rollout state, and writes raw data plus aggregate
metrics. It does not train or modify task configs on disk.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_LEGGED_GYM_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_ROOT = os.path.dirname(_LEGGED_GYM_PACKAGE_ROOT)
for _path in (_LEGGED_GYM_PACKAGE_ROOT, os.path.join(_REPO_ROOT, "rsl_rl")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import isaacgym  # noqa: F401
import torch
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.legged_robot_move_amp_2d import euler_from_quaternion
from legged_gym.utils import get_load_path, task_registry
from legged_gym.utils.math import quat_rotate_inverse


FIXED_COMMANDS: List[Tuple[str, Tuple[float, float, float]]] = [
    ("stand", (0.00, 0.00, 0.00)),
    ("forward", (0.25, 0.00, 0.00)),
    ("backward", (-0.25, 0.00, 0.00)),
    ("leftstep", (0.00, 0.25, 0.00)),
    ("rightstep", (0.00, -0.25, 0.00)),
    ("diag_left", (0.18, 0.18, 0.00)),
    ("diag_right", (0.18, -0.18, 0.00)),
]

SPEED_SCAN_DIRECTIONS = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "leftstep": (0.0, 1.0, 0.0),
    "rightstep": (0.0, -1.0, 0.0),
}

SWITCH_SEQUENCE: List[Tuple[float, float, str, Tuple[float, float, float]]] = [
    (0.0, 3.0, "stand", (0.00, 0.00, 0.00)),
    (3.0, 9.0, "forward", (0.25, 0.00, 0.00)),
    (9.0, 15.0, "backward", (-0.25, 0.00, 0.00)),
    (15.0, 21.0, "leftstep", (0.00, 0.25, 0.00)),
    (21.0, 27.0, "rightstep", (0.00, -0.25, 0.00)),
    (27.0, 33.0, "diag_left", (0.18, 0.18, 0.00)),
    (33.0, 36.0, "stand", (0.00, 0.00, 0.00)),
]

SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def parse_args() -> Any:
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_loco_amp"},
        {"name": "--checkpoint", "type": str, "default": "model_1400"},
        {"name": "--checkpoint-path", "type": str, "default": ""},
        {"name": "--run", "type": str, "default": ""},
        {"name": "--device", "type": str, "default": "cuda:0"},
        {"name": "--num-envs", "type": int, "default": 0},
        {"name": "--episodes-per-command", "type": int, "default": 10},
        {"name": "--episodes-per-speed", "type": int, "default": 5},
        {"name": "--episode-length-s", "type": float, "default": 6.0},
        {"name": "--warmup-s", "type": float, "default": 1.0},
        {"name": "--contact-threshold", "type": float, "default": 1.0},
        {"name": "--output-dir", "type": str, "default": os.path.join("logs", "diagnostics", "model_1400")},
        {"name": "--experiment", "type": str, "default": "all", "choices": ["fixed_commands", "speed_scan", "command_switch", "all"]},
        {"name": "--plot", "action": "store_true", "default": False},
        {"name": "--headless", "action": "store_true", "default": True},
    ]
    args = gymutil.parse_arguments(
        description="Run deterministic policy playback diagnostics.",
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
        ("exptid", args.run or "diagnostics"),
        ("resumeid", None),
        ("experiment_name", None),
        ("run_name", None),
        ("max_iterations", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    if getattr(args, "run", ""):
        args.exptid = args.run
    if getattr(args, "device", ""):
        args.sim_device = args.device
        args.rl_device = args.device
    return args


def warn(metadata: Dict[str, Any], message: str) -> None:
    print(f"[diagnostics] WARNING: {message}")
    metadata.setdefault("warnings", []).append(message)


def safe_set(obj: Any, path: str, value: Any, metadata: Dict[str, Any]) -> bool:
    cur = obj
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(cur, part):
            warn(metadata, f"missing config field {path}; skipped")
            return False
        cur = getattr(cur, part)
    if not hasattr(cur, parts[-1]):
        warn(metadata, f"missing config field {path}; skipped")
        return False
    setattr(cur, parts[-1], value)
    return True


def disable_randomization(env_cfg: Any, num_envs: int, duration_s: float, metadata: Dict[str, Any]) -> None:
    safe_set(env_cfg, "env.num_envs", int(num_envs), metadata)
    safe_set(env_cfg, "env.play", True, metadata)
    safe_set(env_cfg, "env.episode_length_s", float(duration_s), metadata)
    safe_set(env_cfg, "noise.add_noise", False, metadata)

    for path, value in (
        ("terrain.mesh_type", "plane"),
        ("terrain.curriculum", False),
        ("terrain.measure_heights", False),
        ("commands.resampling_time", 1.0e9),
        ("commands.curriculum", False),
        ("commands.curriculum_enable", False),
        ("init_state.root_vel_init_range", 0.0),
        ("init_state.ref_init_prob", 0.0),
        ("domain_rand.push_robots", False),
        ("domain_rand.randomize_friction", False),
        ("domain_rand.randomize_base_mass", False),
        ("domain_rand.randomize_payload_mass", False),
        ("domain_rand.randomize_base_com", False),
        ("domain_rand.randomize_com_displacement", False),
        ("domain_rand.randomize_link_mass", False),
        ("domain_rand.randomize_kp", False),
        ("domain_rand.randomize_kd", False),
        ("domain_rand.randomize_motor_strength", False),
        ("domain_rand.randomize_joint_injection", False),
        ("domain_rand.randomize_actuation_offset", False),
        ("domain_rand.randomize_initial_joint_pos", False),
        ("domain_rand.randomize_restitution", False),
        ("domain_rand.randomize_rigid_props_on_reset", False),
        ("domain_rand.delay", False),
        ("domain_rand.continue_keep", False),
    ):
        safe_set(env_cfg, path, value, metadata)


def checkpoint_label(value: str) -> str:
    stem = Path(value).stem if value else "model"
    return stem if stem.startswith("model_") else value.replace(".pt", "")


def resolve_checkpoint(args: Any, train_cfg: Any) -> str:
    if args.checkpoint_path:
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        return args.checkpoint_path
    ckpt = args.checkpoint
    if ckpt.startswith("model_") and ckpt.endswith(".pt"):
        checkpoint = ckpt
    elif ckpt.startswith("model_"):
        checkpoint = f"{ckpt}.pt"
    else:
        try:
            checkpoint = int(ckpt)
        except ValueError:
            checkpoint = ckpt
    run = args.run or getattr(train_cfg.runner, "load_run", -1)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    if run not in (None, "", -1, "-1"):
        log_root = os.path.join(log_root, str(run))
    return get_load_path(log_root, checkpoint=checkpoint)


def make_env_and_policy(args: Any, metadata: Dict[str, Any]) -> Tuple[Any, Any, torch.Tensor, str, Any]:
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    max_duration = max(float(args.episode_length_s), SWITCH_SEQUENCE[-1][1] if args.experiment in ("command_switch", "all") else 0.0)
    if args.experiment == "fixed_commands":
        needed_envs = args.episodes_per_command
    elif args.experiment == "speed_scan":
        needed_envs = args.episodes_per_speed
    elif args.experiment == "command_switch":
        needed_envs = 1
    else:
        needed_envs = max(args.episodes_per_command, args.episodes_per_speed, 1)
    num_envs = int(args.num_envs) if int(args.num_envs) > 0 else needed_envs
    args.num_envs = num_envs
    disable_randomization(env_cfg, num_envs, max_duration + 0.5, metadata)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    checkpoint_path = resolve_checkpoint(args, train_cfg)
    print(f"[diagnostics] loading checkpoint: {checkpoint_path}")

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
    obs = rebuild_obs_history(env)
    return env, policy, obs, checkpoint_path, train_cfg


def rebuild_obs_history(env: Any) -> torch.Tensor:
    env.compute_observations()
    if hasattr(env, "num_one_step_obs") and hasattr(env, "actor_history_length"):
        current = env.obs_buf[:, -env.num_one_step_obs :].clone()
        env.obs_buf[:] = current.repeat(1, env.actor_history_length)
    return env.get_observations()


def infer_command_type_name(command_name: str, command: Sequence[float]) -> str:
    if command_name == "stand":
        return "standing"
    if command_name == "diag_left":
        return "diagonal_left"
    if command_name == "diag_right":
        return "diagonal_right"
    vx, vy, wz = command
    if max(abs(vx), abs(vy), abs(wz)) < 0.05:
        return "standing"
    return command_name


def force_command(env: Any, command: Sequence[float], command_name: str, env_ids: Optional[torch.Tensor] = None) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "commands"):
        return
    cmd = torch.tensor(command, dtype=torch.float, device=env.device).view(1, 3)
    env.commands[env_ids, :3] = cmd.repeat(len(env_ids), 1)
    if hasattr(env, "command_time_left"):
        env.command_time_left[env_ids] = 999.0
    if hasattr(env, "command_type_ids"):
        names = getattr(env, "amp_command_names", None)
        if names is None and hasattr(env.cfg.commands, "motion_commands"):
            names = list(env.cfg.commands.motion_commands.keys())
        type_name = infer_command_type_name(command_name, command)
        if names and type_name in names:
            env.command_type_ids[env_ids] = int(names.index(type_name))


def reset_envs_for_rollout(env: Any, command: Sequence[float], command_name: str, active_envs: int) -> torch.Tensor:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    if hasattr(env, "episode_length_buf"):
        env.episode_length_buf[:] = 0
    if hasattr(env, "reset_buf"):
        env.reset_buf[:] = 0
    if hasattr(env, "time_out_buf"):
        env.time_out_buf[:] = False
    force_command(env, command, command_name, env_ids)
    obs = rebuild_obs_history(env)
    return obs


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def find_foot_indices(env: Any, metadata: Dict[str, Any]) -> Optional[List[int]]:
    candidates: List[int] = []
    source = ""
    for attr in ("feet_indices", "contact_feet_indices"):
        if hasattr(env, attr):
            arr = getattr(env, attr)
            vals = to_numpy(arr).astype(int).reshape(-1).tolist()
            if vals:
                candidates = vals
                source = attr
                break
    if not candidates and hasattr(env, "body_names"):
        tokens = ("foot", "ankle", "ankle_roll", "toe")
        for i, name in enumerate(env.body_names):
            if any(token in str(name).lower() for token in tokens):
                candidates.append(i)
        source = "body_name_search"
    if len(candidates) < 2:
        warn(metadata, "could not determine two foot body indices; foot/contact plots may be skipped")
        metadata.setdefault("missing_fields", []).append("foot_indices")
        return None
    body_names = list(getattr(env, "body_names", []))
    left = [idx for idx in candidates if idx < len(body_names) and ("left" in body_names[idx].lower() or body_names[idx].lower().startswith("l_"))]
    right = [idx for idx in candidates if idx < len(body_names) and ("right" in body_names[idx].lower() or body_names[idx].lower().startswith("r_"))]
    if left and right:
        foot_indices = [left[0], right[0]]
    else:
        foot_indices = candidates[:2]
    metadata["foot_indices"] = foot_indices
    metadata["foot_index_source"] = source
    metadata["foot_body_names"] = [body_names[i] if i < len(body_names) else str(i) for i in foot_indices]
    return foot_indices


def get_base_quat(env: Any) -> torch.Tensor:
    if hasattr(env, "rigid_body_states") and hasattr(env, "upper_body_index"):
        return env.rigid_body_states[:, env.upper_body_index, 3:7]
    if hasattr(env, "base_quat"):
        return env.base_quat
    return env.root_states[:, 3:7]


def get_base_pos(env: Any) -> torch.Tensor:
    if hasattr(env, "rigid_body_states") and hasattr(env, "upper_body_index"):
        return env.rigid_body_states[:, env.upper_body_index, 0:3]
    if hasattr(env, "torso_pos"):
        return env.torso_pos
    return env.root_states[:, 0:3]


def get_body_frame_velocity(env: Any, metadata: Dict[str, Any]) -> torch.Tensor:
    if hasattr(env, "base_lin_vel"):
        metadata["base_velocity_frame"] = "body"
        return env.base_lin_vel
    if hasattr(env, "root_states"):
        metadata["base_velocity_frame"] = "root_world_converted_to_body"
        return quat_rotate_inverse(env.root_states[:, 3:7], env.root_states[:, 7:10])
    metadata.setdefault("missing_fields", []).append("base_lin_vel")
    return torch.full((env.num_envs, 3), float("nan"), device=env.device)


def get_body_frame_ang_velocity(env: Any) -> torch.Tensor:
    if hasattr(env, "base_ang_vel"):
        return env.base_ang_vel
    if hasattr(env, "root_states"):
        return quat_rotate_inverse(env.root_states[:, 3:7], env.root_states[:, 10:13])
    return torch.full((env.num_envs, 3), float("nan"), device=env.device)


def collect_step(
    env: Any,
    actions: torch.Tensor,
    experiment: str,
    command_name: str,
    command: Sequence[float],
    step_i: int,
    active_envs: int,
    episode_offset: int,
    contact_threshold: float,
    foot_indices: Optional[List[int]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    n = active_envs
    base_pos = get_base_pos(env)[:n]
    base_vel = get_body_frame_velocity(env, metadata)[:n]
    base_ang = get_body_frame_ang_velocity(env)[:n]
    quat = get_base_quat(env)[:n]
    roll, pitch, yaw = euler_from_quaternion(quat)

    row: Dict[str, Any] = {
        "time": np.full(n, step_i * float(env.dt), dtype=np.float32),
        "episode_id": np.arange(episode_offset, episode_offset + n, dtype=np.int32),
        "env_id": np.arange(n, dtype=np.int32),
        "experiment": np.array([experiment] * n),
        "command_name": np.array([command_name] * n),
        "cmd_vx": np.full(n, command[0], dtype=np.float32),
        "cmd_vy": np.full(n, command[1], dtype=np.float32),
        "cmd_yaw_rate": np.full(n, command[2], dtype=np.float32),
        "base_pos_x": to_numpy(base_pos[:, 0]),
        "base_pos_y": to_numpy(base_pos[:, 1]),
        "base_pos_z": to_numpy(base_pos[:, 2]),
        "base_vx": to_numpy(base_vel[:, 0]),
        "base_vy": to_numpy(base_vel[:, 1]),
        "base_vz": to_numpy(base_vel[:, 2]),
        "base_ang_vel_x": to_numpy(base_ang[:, 0]),
        "base_ang_vel_y": to_numpy(base_ang[:, 1]),
        "base_ang_vel_z": to_numpy(base_ang[:, 2]),
        "roll": to_numpy(roll[:n]),
        "pitch": to_numpy(pitch[:n]),
        "yaw": to_numpy(yaw[:n]),
        "actions": to_numpy(actions[:n]),
    }

    for attr in ("dof_pos", "dof_vel", "torques"):
        if hasattr(env, attr):
            row[attr] = to_numpy(getattr(env, attr)[:n])
        else:
            metadata.setdefault("missing_fields", []).append(attr)

    if hasattr(env, "contact_forces") and foot_indices is not None:
        forces = env.contact_forces[:n, foot_indices, 2]
        left_force = forces[:, 0]
        right_force = forces[:, 1]
        left_contact = left_force > contact_threshold
        right_contact = right_force > contact_threshold
        row.update(
            {
                "left_contact": to_numpy(left_contact).astype(np.float32),
                "right_contact": to_numpy(right_contact).astype(np.float32),
                "left_contact_force_z": to_numpy(left_force),
                "right_contact_force_z": to_numpy(right_force),
                "num_feet_contact": to_numpy(left_contact.float() + right_contact.float()),
            }
        )
    else:
        metadata.setdefault("missing_fields", []).append("contact_forces")

    if hasattr(env, "rigid_body_states") and foot_indices is not None:
        foot_world = env.rigid_body_states[:n, foot_indices, 0:3]
        rel = foot_world - base_pos[:, None, :]
        foot_base = quat_rotate_inverse(quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4), rel.reshape(-1, 3)).view(n, 2, 3)
        row.update(
            {
                "left_foot_x_base": to_numpy(foot_base[:, 0, 0]),
                "left_foot_y_base": to_numpy(foot_base[:, 0, 1]),
                "left_foot_z_base": to_numpy(foot_base[:, 0, 2]),
                "right_foot_x_base": to_numpy(foot_base[:, 1, 0]),
                "right_foot_y_base": to_numpy(foot_base[:, 1, 1]),
                "right_foot_z_base": to_numpy(foot_base[:, 1, 2]),
            }
        )
        metadata["foot_position_frame"] = "base"
    else:
        metadata.setdefault("missing_fields", []).append("rigid_body_states")

    return row


def append_rows(storage: Dict[str, List[np.ndarray]], row: Dict[str, Any]) -> None:
    for key, value in row.items():
        storage.setdefault(key, []).append(np.asarray(value))


def finalize_raw(storage: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, chunks in storage.items():
        try:
            out[key] = np.concatenate(chunks, axis=0)
        except ValueError:
            out[key] = np.array(chunks, dtype=object)
    return out


def write_raw_npz(path: Path, raw: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **raw)


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def finite_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanstd(arr))


def metric_range(raw: Dict[str, np.ndarray], mask: np.ndarray, key: str) -> float:
    if key not in raw:
        return float("nan")
    vals = np.asarray(raw[key][mask], dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.nanmax(vals) - np.nanmin(vals))


def compute_episode_metrics(raw: Dict[str, np.ndarray], warmup_s: float, extra_keys: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if "episode_id" not in raw:
        return []
    rows: List[Dict[str, Any]] = []
    episode_ids = sorted(set(np.asarray(raw["episode_id"]).astype(int).tolist()))
    for ep in episode_ids:
        base_mask = np.asarray(raw["episode_id"]).astype(int) == ep
        stat_mask = base_mask & (np.asarray(raw["time"], dtype=float) >= warmup_s)
        if not stat_mask.any():
            continue
        cmd_vx = finite_mean(raw["cmd_vx"][stat_mask])
        cmd_vy = finite_mean(raw["cmd_vy"][stat_mask])
        cmd_speed = math.sqrt(cmd_vx * cmd_vx + cmd_vy * cmd_vy)
        vx = np.asarray(raw["base_vx"][stat_mask], dtype=float)
        vy = np.asarray(raw["base_vy"][stat_mask], dtype=float)
        vz = np.asarray(raw["base_vz"][stat_mask], dtype=float)
        speed_norm_series = np.sqrt(vx * vx + vy * vy)
        tracking_error = np.sqrt((vx - cmd_vx) ** 2 + (vy - cmd_vy) ** 2)
        along = (vx * cmd_vx + vy * cmd_vy) / max(cmd_speed, 1.0e-6)
        denom = np.maximum(speed_norm_series * cmd_speed, 1.0e-6)
        direction_cosine = (vx * cmd_vx + vy * cmd_vy) / denom
        row: Dict[str, Any] = {
            "episode_id": ep,
            "experiment": str(raw["experiment"][base_mask][0]),
            "command_name": str(raw["command_name"][base_mask][0]),
            "cmd_vx": cmd_vx,
            "cmd_vy": cmd_vy,
            "cmd_yaw_rate": finite_mean(raw["cmd_yaw_rate"][stat_mask]),
            "command_speed_norm": cmd_speed,
            "base_vx": finite_mean(vx),
            "base_vy": finite_mean(vy),
            "mean_tracking_error_xy": finite_mean(tracking_error),
            "actual_speed_norm": finite_mean(speed_norm_series),
            "speed_ratio": finite_mean(speed_norm_series) / max(cmd_speed, 1.0e-6),
            "actual_speed_along_command": finite_mean(along),
            "direction_cosine": finite_mean(direction_cosine),
            "stand_drift_speed": finite_mean(speed_norm_series) if cmd_speed < 1.0e-6 else float("nan"),
            "base_height_mean": finite_mean(raw["base_pos_z"][stat_mask]),
            "base_height_std": finite_std(raw["base_pos_z"][stat_mask]),
            "base_z_vel_rms": math.sqrt(finite_mean(vz * vz)),
            "roll_pitch_rms": math.sqrt(finite_mean(np.asarray(raw["roll"][stat_mask], dtype=float) ** 2 + np.asarray(raw["pitch"][stat_mask], dtype=float) ** 2)),
        }
        if "left_contact" in raw and "right_contact" in raw:
            lc = np.asarray(raw["left_contact"][stat_mask], dtype=float) > 0.5
            rc = np.asarray(raw["right_contact"][stat_mask], dtype=float) > 0.5
            state = lc.astype(int) * 2 + rc.astype(int)
            duration = max(float(np.asarray(raw["time"][stat_mask])[-1] - np.asarray(raw["time"][stat_mask])[0]), 1.0e-6)
            row.update(
                {
                    "left_contact_ratio": finite_mean(lc.astype(float)),
                    "right_contact_ratio": finite_mean(rc.astype(float)),
                    "both_contact_ratio": finite_mean((lc & rc).astype(float)),
                    "no_contact_ratio": finite_mean((~lc & ~rc).astype(float)),
                    "flight_ratio": finite_mean((~lc & ~rc).astype(float)),
                    "left_only_ratio": finite_mean((lc & ~rc).astype(float)),
                    "right_only_ratio": finite_mean((~lc & rc).astype(float)),
                    "contact_switch_count": float(np.count_nonzero(np.diff(state)) / duration),
                }
            )
        for prefix in ("left", "right"):
            for axis in ("x", "y", "z"):
                row[f"{prefix}_foot_{axis}_range"] = metric_range(raw, stat_mask, f"{prefix}_foot_{axis}_base")
        if np.isfinite(row.get("left_foot_x_range", np.nan)):
            left_xy = math.sqrt(row["left_foot_x_range"] ** 2 + row["left_foot_y_range"] ** 2)
            right_xy = math.sqrt(row["right_foot_x_range"] ** 2 + row["right_foot_y_range"] ** 2)
            row["foot_xy_range_mean"] = finite_mean(np.array([left_xy, right_xy]))
            row["foot_clearance_mean"] = finite_mean(np.array([row["left_foot_z_range"], row["right_foot_z_range"]]))
        for key, out_key in (("actions", "action"), ("dof_pos", "joint_pos"), ("dof_vel", "joint_vel"), ("torques", "torque")):
            if key in raw:
                arr = np.asarray(raw[key][stat_mask], dtype=float)
                if out_key == "action":
                    row["action_rms"] = math.sqrt(finite_mean(arr * arr))
                    row["action_abs_mean"] = finite_mean(np.abs(arr))
                elif out_key == "joint_pos":
                    row["joint_pos_range_mean"] = finite_mean(np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0))
                elif out_key == "joint_vel":
                    row["joint_vel_rms"] = math.sqrt(finite_mean(arr * arr))
                elif out_key == "torque":
                    row["torque_rms"] = math.sqrt(finite_mean(arr * arr))
        if "torques" in raw and extra_keys and extra_keys.get("torque_limits") is not None:
            torques = np.asarray(raw["torques"][stat_mask], dtype=float)
            limits = np.asarray(extra_keys["torque_limits"], dtype=float).reshape(1, -1)
            row["torque_limit_ratio"] = finite_mean((np.abs(torques) > 0.9 * limits).astype(float))
        rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key, items in groups.items():
        agg = {k: v for k, v in zip(group_keys, key)}
        numeric_keys = sorted(k for item in items for k, v in item.items() if isinstance(v, (int, float, np.floating)) and k not in group_keys)
        for nk in numeric_keys:
            vals = np.array([float(item.get(nk, np.nan)) for item in items], dtype=float)
            agg[f"{nk}_mean"] = finite_mean(vals)
            agg[f"{nk}_std"] = finite_std(vals)
        out.append(agg)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_rollout_group(
    env: Any,
    policy: Any,
    obs: torch.Tensor,
    experiment: str,
    command_name: str,
    command: Sequence[float],
    duration_s: float,
    active_envs: int,
    episode_offset: int,
    contact_threshold: float,
    foot_indices: Optional[List[int]],
    metadata: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    storage: Dict[str, List[np.ndarray]] = {}
    obs = reset_envs_for_rollout(env, command, command_name, active_envs)
    steps = int(round(duration_s / float(env.dt)))
    active_ids = torch.arange(active_envs, device=env.device)
    for step_i in range(steps):
        force_command(env, command, command_name, active_ids)
        with torch.no_grad():
            actions = policy(obs.detach())
        step_ret = env.step(actions.detach())
        obs = step_ret[0]
        dones = step_ret[3]
        force_command(env, command, command_name, active_ids)
        append_rows(
            storage,
            collect_step(
                env,
                actions,
                experiment,
                command_name,
                command,
                step_i,
                active_envs,
                episode_offset,
                contact_threshold,
                foot_indices,
                metadata,
            ),
        )
        if torch.any(dones[:active_envs]):
            reset_ids = (dones[:active_envs] > 0).nonzero(as_tuple=False).flatten()
            force_command(env, command, command_name, reset_ids)
            obs = rebuild_obs_history(env)
    return obs, finalize_raw(storage)


def run_fixed_commands(env: Any, policy: Any, obs: torch.Tensor, args: Any, root: Path, foot_indices: Optional[List[int]], metadata: Dict[str, Any], extra: Dict[str, Any]) -> torch.Tensor:
    exp_dir = root / "fixed_commands"
    all_raw: Dict[str, List[np.ndarray]] = {}
    active_envs = min(env.num_envs, int(args.episodes_per_command))
    episode_offset = 0
    for command_name, command in FIXED_COMMANDS:
        print(f"[diagnostics] fixed_commands: {command_name} {command}")
        obs, raw = run_rollout_group(env, policy, obs, "fixed_commands", command_name, command, args.episode_length_s, active_envs, episode_offset, args.contact_threshold, foot_indices, metadata)
        for key, value in raw.items():
            all_raw.setdefault(key, []).append(value)
        episode_offset += active_envs
    raw_all = finalize_raw(all_raw)
    write_raw_npz(exp_dir / "raw_rollouts.npz", raw_all)
    rows = compute_episode_metrics(raw_all, args.warmup_s, extra)
    write_csv(exp_dir / "metrics_per_episode.csv", rows)
    write_csv(exp_dir / "metrics_by_command.csv", aggregate(rows, ["command_name"]))
    write_metadata(exp_dir / "metadata.json", metadata, args, extra)
    return obs


def run_speed_scan(env: Any, policy: Any, obs: torch.Tensor, args: Any, root: Path, foot_indices: Optional[List[int]], metadata: Dict[str, Any], extra: Dict[str, Any]) -> torch.Tensor:
    exp_dir = root / "speed_scan"
    all_raw: Dict[str, List[np.ndarray]] = {}
    active_envs = min(env.num_envs, int(args.episodes_per_speed))
    episode_offset = 0
    for direction, unit in SPEED_SCAN_DIRECTIONS.items():
        for speed in SPEEDS:
            command = (unit[0] * speed, unit[1] * speed, 0.0)
            print(f"[diagnostics] speed_scan: {direction} speed={speed:.2f}")
            obs, raw = run_rollout_group(env, policy, obs, "speed_scan", direction, command, args.episode_length_s, active_envs, episode_offset, args.contact_threshold, foot_indices, metadata)
            for key, value in raw.items():
                all_raw.setdefault(key, []).append(value)
            episode_offset += active_envs
    raw_all = finalize_raw(all_raw)
    write_raw_npz(exp_dir / "raw_rollouts.npz", raw_all)
    rows = compute_episode_metrics(raw_all, args.warmup_s, extra)
    for row in rows:
        row["command_speed"] = row.get("command_speed_norm", float("nan"))
    write_csv(exp_dir / "metrics_per_episode.csv", rows)
    write_csv(exp_dir / "metrics_by_command_speed.csv", aggregate(rows, ["command_name", "command_speed"]))
    write_metadata(exp_dir / "metadata.json", metadata, args, extra)
    return obs


def switch_command_at(t: float) -> Tuple[str, Tuple[float, float, float]]:
    for start, end, name, command in SWITCH_SEQUENCE:
        if start <= t < end:
            return name, command
    return SWITCH_SEQUENCE[-1][2], SWITCH_SEQUENCE[-1][3]


def run_command_switch(env: Any, policy: Any, obs: torch.Tensor, args: Any, root: Path, foot_indices: Optional[List[int]], metadata: Dict[str, Any], extra: Dict[str, Any]) -> torch.Tensor:
    exp_dir = root / "command_switch"
    storage: Dict[str, List[np.ndarray]] = {}
    active_envs = 1
    command_name, command = switch_command_at(0.0)
    obs = reset_envs_for_rollout(env, command, command_name, active_envs)
    steps = int(round(SWITCH_SEQUENCE[-1][1] / float(env.dt)))
    active_ids = torch.arange(active_envs, device=env.device)
    last_name = command_name
    for step_i in range(steps):
        t = step_i * float(env.dt)
        command_name, command = switch_command_at(t)
        force_command(env, command, command_name, active_ids)
        if command_name != last_name:
            obs = rebuild_obs_history(env)
            last_name = command_name
        with torch.no_grad():
            actions = policy(obs.detach())
        step_ret = env.step(actions.detach())
        obs = step_ret[0]
        dones = step_ret[3]
        force_command(env, command, command_name, active_ids)
        append_rows(
            storage,
            collect_step(
                env,
                actions,
                "command_switch",
                command_name,
                command,
                step_i,
                active_envs,
                0,
                args.contact_threshold,
                foot_indices,
                metadata,
            ),
        )
        if torch.any(dones[:active_envs]):
            force_command(env, command, command_name, active_ids)
            obs = rebuild_obs_history(env)
    raw = finalize_raw(storage)
    write_raw_npz(exp_dir / "raw_rollouts.npz", raw)
    scalar_keys = [k for k, v in raw.items() if np.asarray(v).ndim == 1]
    rows = [{key: raw[key][i] for key in scalar_keys} for i in range(len(raw["time"]))]
    write_csv(exp_dir / "metrics_timeseries.csv", rows)
    write_metadata(exp_dir / "metadata.json", metadata, args, extra)
    return obs


def write_metadata(path: Path, metadata: Dict[str, Any], args: Any, extra: Dict[str, Any]) -> None:
    data = dict(metadata)
    data["task"] = args.task
    data["checkpoint_arg"] = args.checkpoint
    data["run"] = args.run
    data["episode_length_s"] = args.episode_length_s
    data["warmup_s"] = args.warmup_s
    data["contact_threshold"] = args.contact_threshold
    data["command_table"] = {name: cmd for name, cmd in FIXED_COMMANDS}
    data["switch_sequence"] = SWITCH_SEQUENCE
    data["available_fields"] = sorted(extra.get("available_fields", []))
    data["dof_names"] = extra.get("dof_names", [])
    data["body_names"] = extra.get("body_names", [])
    data["checkpoint_path"] = extra.get("checkpoint_path", "")
    data["randomization_overrides_attempted"] = True
    data["missing_fields"] = sorted(set(data.get("missing_fields", [])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {"warnings": [], "missing_fields": []}
    env, policy, obs, checkpoint_path, train_cfg = make_env_and_policy(args, metadata)
    foot_indices = find_foot_indices(env, metadata)
    extra = {
        "checkpoint_path": checkpoint_path,
        "torque_limits": to_numpy(getattr(env, "torque_limits", None)) if hasattr(env, "torque_limits") else None,
        "dof_names": list(getattr(env, "dof_names", [])),
        "body_names": list(getattr(env, "body_names", [])),
        "available_fields": [name for name in (
            "root_states",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "torques",
            "actions",
            "contact_forces",
            "contact_feet_indices",
            "rigid_body_states",
            "torque_limits",
        ) if hasattr(env, name)],
    }
    metadata["runner_experiment_name"] = train_cfg.runner.experiment_name

    if args.experiment in ("fixed_commands", "all"):
        obs = run_fixed_commands(env, policy, obs, args, root, foot_indices, metadata, extra)
    if args.experiment in ("speed_scan", "all"):
        obs = run_speed_scan(env, policy, obs, args, root, foot_indices, metadata, extra)
    if args.experiment in ("command_switch", "all"):
        obs = run_command_switch(env, policy, obs, args, root, foot_indices, metadata, extra)

    write_metadata(root / "metadata.json", metadata, args, extra)
    if args.plot:
        from plot_policy_diagnostics import main as plot_main

        plot_main(["--input-dir", str(root), "--output-dir", str(root / "figures")])
    print(f"[diagnostics] done: {root}")


if __name__ == "__main__":
    main()
