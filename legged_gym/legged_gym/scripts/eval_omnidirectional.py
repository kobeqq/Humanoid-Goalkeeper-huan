#!/usr/bin/env python3
"""Eight-direction evaluation for goal-conditioned omnidirectional locomotion."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil

# IsaacGym Preview 4 still references deprecated NumPy aliases.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry


DIRECTIONS = [
    ("Front", 0.0),
    ("FrontLeft", 45.0),
    ("Left", 90.0),
    ("BackLeft", 135.0),
    ("Back", 180.0),
    ("BackRight", 225.0),
    ("Right", 270.0),
    ("FrontRight", 315.0),
]


def parse_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "k1_move_amp"},
        {"name": "--checkpoint_path", "type": str, "default": ""},
        {"name": "--exptid", "type": str, "default": "debug_k1_move_amp_robust"},
        {"name": "--resumeid", "type": str, "default": ""},
        {"name": "--num_trials", "type": int, "default": 20},
        {"name": "--duration_s", "type": float, "default": 5.0},
        {"name": "--goal_radius", "type": float, "default": 2.0},
        {"name": "--output_dir", "type": str, "default": os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "omni_eval")},
        {"name": "--seed", "type": int, "default": 1},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate turn-and-forward vs true omnidirectional locomotion.",
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
        ("num_envs", None),
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


def disable_randomization(env_cfg, num_envs: int, duration_s: float) -> None:
    env_cfg.env.num_envs = num_envs
    env_cfg.env.play = True
    env_cfg.env.episode_length_s = duration_s + 1.0
    env_cfg.noise.add_noise = False
    if hasattr(env_cfg.terrain, "use_bumpy_ground"):
        env_cfg.terrain.use_bumpy_ground = False

    dr = env_cfg.domain_rand
    for name in (
        "push_robots",
        "randomize_friction",
        "arm_disturbance_enable",
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

    if hasattr(env_cfg.rewards, "resample_on_success"):
        env_cfg.rewards.resample_on_success = False
    if hasattr(env_cfg.rewards, "terminate_on_success_before_dynamic"):
        env_cfg.rewards.terminate_on_success_before_dynamic = False
    if hasattr(env_cfg.rewards, "dynamic_goal_stage"):
        env_cfg.rewards.dynamic_goal_stage = 99


Policy = Callable[[torch.Tensor], torch.Tensor]


def make_env_and_policy(args) -> Tuple[Any, Policy, torch.Tensor]:
    num_envs = len(DIRECTIONS) * args.num_trials
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_randomization(env_cfg, num_envs, args.duration_s)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    checkpoint_path = args.checkpoint_path.strip()
    train_cfg.runner.resume = not bool(checkpoint_path)
    log_root = "default" if not checkpoint_path else None
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=log_root,  # type: ignore[arg-type]
    )

    if checkpoint_path:
        try:
            runner.load(checkpoint_path, load_optimizer=False)
            runner_policy = runner.get_inference_policy(device=env.device)
            policy = lambda obs_tensor: runner_policy(obs_tensor)
        except Exception:
            jit_policy = torch.jit.load(checkpoint_path, map_location=env.device)
            jit_policy.eval()

            def policy(obs_tensor: torch.Tensor) -> torch.Tensor:
                return jit_policy(obs_tensor)
    else:
        runner_policy = runner.get_inference_policy(device=env.device)
        policy = lambda obs_tensor: runner_policy(obs_tensor)
    return env, policy, obs


def reset_fixed_state(env, goal_radius: float, target_angles_deg: torch.Tensor) -> None:
    device = env.device
    env_ids = torch.arange(env.num_envs, device=device)
    env.reset_idx(env_ids)

    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins
    env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    env.root_states[:, 7:13] = 0.0

    if getattr(env, "use_ball_actor", False) and getattr(env, "ball_states", None) is not None:
        env.ball_states[:] = env.base_init_state
        env.ball_states[:, :3] = env.env_origins
        env.ball_states[:, 2] = -10.0
        env.ball_states[:, 7:13] = 0.0
        all_states = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
    else:
        all_states = env.root_states.contiguous()
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_states))

    dof_state = env.dof_state.view(env.num_envs, env.num_dof, 2)
    if hasattr(env, "standpos"):
        standpos = env.standpos
        if standpos.dim() == 1:
            dof_pos = standpos.unsqueeze(0).repeat(env.num_envs, 1)
        else:
            dof_pos = standpos[:1].repeat(env.num_envs, 1)
    else:
        dof_pos = env.default_dof_pos.clone()
    dof_state[:, :, 0] = dof_pos
    dof_state[:, :, 1] = 0.0
    env.dof_pos[:] = dof_pos
    env.dof_vel[:] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))

    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    env.torso_pos = env.rigid_body_states[:, env.torso_index, 0:3]
    theta = torch.deg2rad(target_angles_deg.to(device))
    target_xy = torch.stack((goal_radius * torch.cos(theta), goal_radius * torch.sin(theta)), dim=-1)
    env.target_pos[:, :2] = env.torso_pos[:, :2] + target_xy
    if not env._uses_target_z():
        env.target_pos[:, 2] = env.torso_pos[:, 2]

    env.episode_length_buf[:] = 0
    env.reset_buf[:] = 0
    env.time_out_buf[:] = False
    if hasattr(env, "goal_reached_this_step"):
        env.goal_reached_this_step[:] = 0.0
    if hasattr(env, "last_target_dist"):
        env.last_target_dist[:] = env._target_xy_distance()
    for name in ("last_actions", "last_last_actions", "actions", "last_dof_vel", "last_torques", "last_root_vel"):
        if hasattr(env, name):
            getattr(env, name)[:] = 0.0

    env.compute_observations()
    current_obs = env.obs_buf[:, -env.num_one_step_obs :].clone()
    history = env.obs_buf.shape[1] // env.num_one_step_obs
    env.obs_buf[:] = current_obs.repeat(1, history)


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def classify(direction: str, lateral_ratio: float, yaw_change: float, mean_forward: float) -> str:
    yaw_abs = abs(yaw_change)
    if direction in ("Left", "Right"):
        if lateral_ratio > 0.6 and yaw_abs < 20.0:
            return "TRUE SIDE WALK"
        if yaw_abs > 60.0 and lateral_ratio < 0.2:
            return "TURN + FORWARD"
    if direction == "Back":
        if mean_forward < 0.0 and yaw_abs < 30.0:
            return "TRUE BACKWARD WALK"
        if yaw_abs > 120.0:
            return "TURN AROUND + FORWARD"
    return "UNCLASSIFIED"


def aggregate_episode(rows: List[Dict[str, float]], direction: str, target_angle_deg: float, success: bool, fall: bool) -> Dict[str, Any]:
    if not rows:
        return {}
    n = len(rows)
    mean_forward = sum(r["base_lin_vel_body_x"] for r in rows) / n
    mean_lateral = sum(r["base_lin_vel_body_y"] for r in rows) / n
    mean_abs_yaw_rate = sum(abs(r["base_ang_vel_z"]) for r in rows) / n
    yaw_change_deg = wrap_degrees(math.degrees(rows[-1]["base_yaw"] - rows[0]["base_yaw"]))
    dx = rows[-1]["base_position_x"] - rows[0]["base_position_x"]
    dy = rows[-1]["base_position_y"] - rows[0]["base_position_y"]
    move_angle = math.degrees(math.atan2(dy, dx)) % 360.0 if abs(dx) + abs(dy) > 1e-8 else 0.0
    heading_error = wrap_degrees(target_angle_deg - move_angle)
    lateral_ratio = abs(mean_lateral) / (abs(mean_forward) + abs(mean_lateral) + 1e-6)
    return {
        "direction": direction,
        "target_angle_deg": target_angle_deg,
        "mean_forward_vel": mean_forward,
        "mean_lateral_vel": mean_lateral,
        "mean_abs_yaw_rate": mean_abs_yaw_rate,
        "yaw_change_deg": yaw_change_deg,
        "dx": dx,
        "dy": dy,
        "move_angle_deg": move_angle,
        "heading_error_deg": heading_error,
        "lateral_ratio": lateral_ratio,
        "success": float(success),
        "fall": float(fall),
        "classification": classify(direction, lateral_ratio, yaw_change_deg, mean_forward),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, summary_rows: List[Dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    labels = [r["Direction"] for r in summary_rows]
    target = [r["Target Angle"] for r in summary_rows]
    move = [r["Move Angle"] for r in summary_rows]
    yaw = [abs(r["Yaw Change"]) for r in summary_rows]
    lateral = [r["Lateral Ratio"] for r in summary_rows]

    plt.figure(figsize=(7, 5))
    plt.plot(target, target, "k--", label="ideal")
    plt.scatter(target, move)
    for x, y, label in zip(target, move, labels):
        plt.annotate(label, (x, y), fontsize=8)
    plt.xlabel("Target Angle (deg)")
    plt.ylabel("Move Angle (deg)")
    plt.title("Figure 1: Target Angle vs Move Angle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_target_vs_move_angle.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(labels, yaw)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Yaw Change (deg)")
    plt.title("Figure 2: Direction vs Yaw Change")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_direction_vs_yaw_change.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(labels, lateral)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Lateral Ratio")
    plt.ylim(0.0, 1.0)
    plt.title("Figure 3: Direction vs Lateral Ratio")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_direction_vs_lateral_ratio.png", dpi=160)
    plt.close()


def write_report(output_dir: Path, summary_rows: List[Dict[str, float]]) -> None:
    lines = [
        "# Eight-Direction Omnidirectional Locomotion Evaluation",
        "",
        "| Direction | Forward Vel | Lateral Vel | Lateral Ratio | Mean Yaw Rate | Yaw Change | Move Angle | Success Rate | Fall Rate | Classification |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['Direction']} | {r['Forward Vel']:.3f} | {r['Lateral Vel']:.3f} | "
            f"{r['Lateral Ratio']:.3f} | {r['Mean Yaw Rate']:.3f} | {r['Yaw Change']:.1f} | "
            f"{r['Move Angle']:.1f} | {r['Success Rate']:.2f} | {r['Fall Rate']:.2f} | {r['Classification']} |"
        )
    lines.extend([
        "",
        "Figures:",
        "- `figure1_target_vs_move_angle.png`",
        "- `figure2_direction_vs_yaw_change.png`",
        "- `figure3_direction_vs_lateral_ratio.png`",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env, policy, _ = make_env_and_policy(args)
    device = env.device
    num_envs = len(DIRECTIONS) * args.num_trials
    direction_ids = torch.arange(num_envs, device=device) // args.num_trials
    target_angles = torch.tensor([DIRECTIONS[int(i)][1] for i in direction_ids.cpu()], device=device)
    direction_names = [DIRECTIONS[int(i)][0] for i in direction_ids.cpu()]

    reset_fixed_state(env, args.goal_radius, target_angles)
    obs = env.get_observations()
    steps = int(round(args.duration_s / env.dt))
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    per_env_rows: List[List[Dict[str, float]]] = [[] for _ in range(num_envs)]
    timeseries_rows: List[Dict[str, float]] = []

    with torch.no_grad():
        for step in range(steps):
            target_local = env._target_local_obs()
            dist = env._target_xy_distance()
            succeeded |= dist < float(env.cfg.rewards.target_reach_threshold)
            data = {
                "base_lin_vel_body_x": env.base_lin_vel[:, 0].detach().cpu(),
                "base_lin_vel_body_y": env.base_lin_vel[:, 1].detach().cpu(),
                "base_ang_vel_z": env.base_ang_vel[:, 2].detach().cpu(),
                "base_position_x": (env.torso_pos[:, 0] - env.env_origins[:, 0]).detach().cpu(),
                "base_position_y": (env.torso_pos[:, 1] - env.env_origins[:, 1]).detach().cpu(),
                "base_yaw": env.yaw.detach().cpu(),
                "target_local_x": target_local[:, 0].detach().cpu(),
                "target_local_y": target_local[:, 1].detach().cpu(),
            }
            active_cpu = active.detach().cpu()
            for env_id in range(num_envs):
                if not active_cpu[env_id]:
                    continue
                row = {
                    "step": step,
                    "time_s": step * env.dt,
                    "env_id": env_id,
                    "trial": env_id % args.num_trials,
                    "direction": direction_names[env_id],
                    "target_angle_deg": float(target_angles[env_id].item()),
                }
                for key, value in data.items():
                    row[key] = float(value[env_id].item())
                per_env_rows[env_id].append(row)
                timeseries_rows.append(row)

            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
            done_bool = dones.bool()
            fell |= done_bool & active
            active &= ~done_bool

    episode_rows = []
    for env_id in range(num_envs):
        row = aggregate_episode(
            per_env_rows[env_id],
            direction_names[env_id],
            float(target_angles[env_id].item()),
            bool(succeeded[env_id].item()),
            bool(fell[env_id].item()),
        )
        if row:
            row["env_id"] = env_id
            row["trial"] = env_id % args.num_trials
            episode_rows.append(row)

    summary_rows = []
    for direction, angle in DIRECTIONS:
        rows = [r for r in episode_rows if r["direction"] == direction]
        n = max(len(rows), 1)
        summary = {
            "Direction": direction,
            "Target Angle": angle,
            "Forward Vel": sum(r["mean_forward_vel"] for r in rows) / n,
            "Lateral Vel": sum(r["mean_lateral_vel"] for r in rows) / n,
            "Lateral Ratio": sum(r["lateral_ratio"] for r in rows) / n,
            "Mean Yaw Rate": sum(r["mean_abs_yaw_rate"] for r in rows) / n,
            "Yaw Change": sum(abs(r["yaw_change_deg"]) for r in rows) / n,
            "Move Angle": sum(r["move_angle_deg"] for r in rows) / n,
            "Success Rate": sum(r["success"] for r in rows) / n,
            "Fall Rate": sum(r["fall"] for r in rows) / n,
        }
        summary["Classification"] = classify(
            direction,
            summary["Lateral Ratio"],
            summary["Yaw Change"],
            summary["Forward Vel"],
        )
        summary_rows.append(summary)

    write_csv(output_dir / "timeseries.csv", timeseries_rows)
    write_csv(output_dir / "episodes.csv", episode_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    make_plots(output_dir, summary_rows)
    write_report(output_dir, summary_rows)

    print(f"Saved evaluation outputs to: {output_dir}")
    print((output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
