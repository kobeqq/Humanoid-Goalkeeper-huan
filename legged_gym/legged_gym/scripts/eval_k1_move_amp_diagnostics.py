import argparse
import csv
import importlib.util
import importlib.machinery
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "legged_gym"
RSL_RL_ROOT = REPO_ROOT / "rsl_rl"
for path in (PACKAGE_ROOT, RSL_RL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
if "onnxruntime" not in sys.modules:
    onnx_stub = types.ModuleType("onnxruntime")
    onnx_stub.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", loader=None)
    sys.modules["onnxruntime"] = onnx_stub
if "git" not in sys.modules:
    git_stub = types.ModuleType("git")
    git_stub.__spec__ = importlib.machinery.ModuleSpec("git", loader=None)

    class _MissingGitRepo:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GitPython is not installed in this environment.")

    git_stub.Repo = _MissingGitRepo
    sys.modules["git"] = git_stub
if "pydelatin" not in sys.modules:
    pydelatin_stub = types.ModuleType("pydelatin")
    pydelatin_stub.__spec__ = importlib.machinery.ModuleSpec("pydelatin", loader=None)

    class _MissingDelatin:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("pydelatin is not installed in this environment.")

    pydelatin_stub.Delatin = _MissingDelatin
    sys.modules["pydelatin"] = pydelatin_stub
if "pyfqmr" not in sys.modules:
    pyfqmr_stub = types.ModuleType("pyfqmr")
    pyfqmr_stub.__spec__ = importlib.machinery.ModuleSpec("pyfqmr", loader=None)
    sys.modules["pyfqmr"] = pyfqmr_stub
LEGACY_INIT = PACKAGE_ROOT / "legged_gym" / "__init__.py"
if "legged_gym" not in sys.modules:
    spec = importlib.util.spec_from_file_location("legged_gym", LEGACY_INIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legged_gym"] = module
    spec.loader.exec_module(module)

import isaacgym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaacgym import gymapi

if not hasattr(np, "float"):
    np.float = float

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_rotate_inverse
from legged_gym.utils.task_registry import task_registry


DEFAULT_CHECKPOINT = (
    f"{LEGGED_GYM_ROOT_DIR}/logs/k1_move_amp/debug_k1_move_amp/model_3200.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate k1_move_amp checkpoint diagnostics.")
    parser.add_argument("--task", type=str, default="k1_move_amp")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/diagnostics/model_3200_k1_move_amp",
    )
    parser.add_argument("--rl-device", type=str, default="cpu")
    parser.add_argument("--sim-device", type=str, default="cpu")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--target-x", type=float, default=5.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def make_runtime_args(cli_args):
    sim_on_gpu = "cuda" in cli_args.sim_device
    return SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=cli_args.sim_device,
        rl_device=cli_args.rl_device,
        headless=cli_args.headless,
        num_threads=0,
        use_gpu=sim_on_gpu,
        use_gpu_pipeline=sim_on_gpu,
        subscenes=0,
        device="cpu",
        num_envs=cli_args.num_envs,
        seed=cli_args.seed,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        exptid="diagnostics",
        resumeid=None,
        horovod=False,
    )


def scenario_table(target_x, target_y):
    return [
        {"name": "target_x5", "offset_xy": (target_x, target_y)},
    ]


def configure_env(env_cfg, cli_args, num_scenarios):
    env_cfg.env.num_envs = min(cli_args.num_envs, num_scenarios)
    env_cfg.env.episode_length_s = max(cli_args.duration_s + 1.0, env_cfg.env.episode_length_s)
    env_cfg.env.play = False
    env_cfg.noise.add_noise = False

    env_cfg.commands.fix_target = True
    env_cfg.commands.fixed_target_x = cli_args.target_x
    env_cfg.commands.target_use_y = abs(float(cli_args.target_y)) > 1e-6
    env_cfg.commands.target_use_z = False
    if env_cfg.commands.target_use_y:
        env_cfg.commands.target_y = [cli_args.target_y, cli_args.target_y]

    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg.domain_rand, "randomize_base_mass"):
        env_cfg.domain_rand.randomize_base_mass = False
    if hasattr(env_cfg.domain_rand, "randomize_base_com"):
        env_cfg.domain_rand.randomize_base_com = False
    if hasattr(env_cfg.domain_rand, "randomize_motor_strength"):
        env_cfg.domain_rand.randomize_motor_strength = False


def set_fixed_targets(env, scenarios):
    count = min(env.num_envs, len(scenarios))
    env.target_pos[:count, :2] = env.env_origins[:count, :2]
    for idx, scenario in enumerate(scenarios[:count]):
        dx, dy = scenario["offset_xy"]
        env.target_pos[idx, 0] += dx
        env.target_pos[idx, 1] += dy
    env.target_pos[:count, 2] = env.torso_pos[:count, 2]


def get_upper_body_quat(env):
    return env.rigid_body_states[:, env.upper_body_index, 3:7]


def collect_step(env, active_mask, scenarios, step_idx):
    upper_quat = get_upper_body_quat(env)
    torso_world_vel = env.rigid_body_states[:, env.torso_index, 7:10]
    foot_world = env.rigid_body_states[:, env.contact_feet_indices, 0:3]
    foot_base = quat_rotate_inverse(
        upper_quat.unsqueeze(1).expand(-1, foot_world.shape[1], -1).reshape(-1, 4),
        (foot_world - env.torso_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, foot_world.shape[1], 3)

    record = {
        "time": np.full(env.num_envs, step_idx * env.dt, dtype=np.float32),
        "env_id": np.arange(env.num_envs, dtype=np.int32),
        "scenario": np.array([scenario["name"] for scenario in scenarios], dtype="<U16"),
        "active": active_mask.detach().cpu().numpy().astype(np.int32),
        "base_pos_x": env.torso_pos[:, 0].detach().cpu().numpy(),
        "base_pos_y": env.torso_pos[:, 1].detach().cpu().numpy(),
        "base_pos_z": env.torso_pos[:, 2].detach().cpu().numpy(),
        "target_pos_x": env.target_pos[:, 0].detach().cpu().numpy(),
        "target_pos_y": env.target_pos[:, 1].detach().cpu().numpy(),
        "base_vx_body": env.base_lin_vel[:, 0].detach().cpu().numpy(),
        "base_vy_body": env.base_lin_vel[:, 1].detach().cpu().numpy(),
        "base_vz_body": env.base_lin_vel[:, 2].detach().cpu().numpy(),
        "torso_vx_world": torso_world_vel[:, 0].detach().cpu().numpy(),
        "torso_vy_world": torso_world_vel[:, 1].detach().cpu().numpy(),
        "torso_vz_world": torso_world_vel[:, 2].detach().cpu().numpy(),
        "roll": env.roll.detach().cpu().numpy(),
        "pitch": env.pitch.detach().cpu().numpy(),
        "yaw": env.yaw.detach().cpu().numpy(),
        "base_ang_vel_x": env.base_ang_vel[:, 0].detach().cpu().numpy(),
        "base_ang_vel_y": env.base_ang_vel[:, 1].detach().cpu().numpy(),
        "base_ang_vel_z": env.base_ang_vel[:, 2].detach().cpu().numpy(),
        "left_contact": (env.contact_forces[:, env.contact_feet_indices[0], 2] > 1.0).float().detach().cpu().numpy(),
        "right_contact": (env.contact_forces[:, env.contact_feet_indices[1], 2] > 1.0).float().detach().cpu().numpy(),
        "left_contact_force_z": env.contact_forces[:, env.contact_feet_indices[0], 2].detach().cpu().numpy(),
        "right_contact_force_z": env.contact_forces[:, env.contact_feet_indices[1], 2].detach().cpu().numpy(),
        "left_foot_x_base": foot_base[:, 0, 0].detach().cpu().numpy(),
        "left_foot_y_base": foot_base[:, 0, 1].detach().cpu().numpy(),
        "left_foot_z_base": foot_base[:, 0, 2].detach().cpu().numpy(),
        "right_foot_x_base": foot_base[:, 1, 0].detach().cpu().numpy(),
        "right_foot_y_base": foot_base[:, 1, 1].detach().cpu().numpy(),
        "right_foot_z_base": foot_base[:, 1, 2].detach().cpu().numpy(),
    }
    return record


def flatten_records(records):
    merged = {}
    for key in records[0]:
        merged[key] = np.concatenate([record[key] for record in records], axis=0)
    return merged


def save_npz(data, path):
    np.savez(path, **data)


def rising_edges(binary_signal):
    signal = np.asarray(binary_signal, dtype=np.int32)
    return np.where((signal[1:] == 1) & (signal[:-1] == 0))[0] + 1


def wrapped_phase_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def dominant_phase_metrics(time_s, left_z, right_z, min_hz=0.3, max_hz=5.0):
    if len(time_s) < 8:
        return math.nan, math.nan
    dt = float(np.median(np.diff(time_s)))
    left = np.asarray(left_z) - np.mean(left_z)
    right = np.asarray(right_z) - np.mean(right_z)
    freq = np.fft.rfftfreq(len(time_s), dt)
    left_fft = np.fft.rfft(left)
    right_fft = np.fft.rfft(right)
    valid = (freq >= min_hz) & (freq <= max_hz)
    if not np.any(valid):
        return math.nan, math.nan
    power = np.abs(left_fft) ** 2 + np.abs(right_fft) ** 2
    valid_idx = np.where(valid)[0]
    dom_idx = valid_idx[np.argmax(power[valid])]
    phase_deg = math.degrees(np.angle(right_fft[dom_idx]) - np.angle(left_fft[dom_idx]))
    return float(freq[dom_idx]), float(wrapped_phase_deg(phase_deg))


def scenario_metrics(flat, scenarios, warmup_s):
    metrics = []
    for env_id, scenario in enumerate(scenarios):
        mask = (flat["env_id"] == env_id) & (flat["active"] > 0) & (flat["time"] >= warmup_s)
        if not np.any(mask):
            continue

        time_s = flat["time"][mask]
        base_x = flat["base_pos_x"][mask]
        base_y = flat["base_pos_y"][mask]
        target_x = flat["target_pos_x"][mask]
        target_y = flat["target_pos_y"][mask]
        torso_vx = flat["torso_vx_world"][mask]
        torso_vy = flat["torso_vy_world"][mask]
        torso_vz = flat["torso_vz_world"][mask]
        speed_norm = np.sqrt(torso_vx ** 2 + torso_vy ** 2)
        target_dx = target_x - base_x
        target_dy = target_y - base_y
        target_dist = np.sqrt(target_dx ** 2 + target_dy ** 2)
        target_dir_x = target_dx / np.clip(target_dist, 1e-6, None)
        target_dir_y = target_dy / np.clip(target_dist, 1e-6, None)
        speed_along = torso_vx * target_dir_x + torso_vy * target_dir_y

        left_contact = flat["left_contact"][mask] > 0.5
        right_contact = flat["right_contact"][mask] > 0.5
        left_z = flat["left_foot_z_base"][mask]
        right_z = flat["right_foot_z_base"][mask]
        dom_freq, phase_lag_deg = dominant_phase_metrics(time_s, left_z, right_z)

        left_edges = rising_edges(left_contact)
        right_edges = rising_edges(right_contact)
        step_intervals = []
        if len(left_edges) >= 2:
            step_intervals.extend(np.diff(time_s[left_edges]).tolist())
        if len(right_edges) >= 2:
            step_intervals.extend(np.diff(time_s[right_edges]).tolist())
        step_frequency_hz = float(1.0 / np.mean(step_intervals)) if step_intervals else math.nan

        roll = np.degrees(flat["roll"][mask])
        pitch = np.degrees(flat["pitch"][mask])
        base_ang_vel_xy = np.sqrt(
            flat["base_ang_vel_x"][mask] ** 2 + flat["base_ang_vel_y"][mask] ** 2
        )

        metrics.append(
            {
                "scenario": scenario["name"],
                "target_x": float(scenario["offset_xy"][0]),
                "target_y": float(scenario["offset_xy"][1]),
                "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0,
                "target_dist_start": float(target_dist[0]),
                "target_dist_end": float(target_dist[-1]),
                "target_progress": float(target_dist[0] - target_dist[-1]),
                "target_progress_ratio": float((target_dist[0] - target_dist[-1]) / max(target_dist[0], 1e-6)),
                "mean_speed_norm": float(np.mean(speed_norm)),
                "std_speed_norm": float(np.std(speed_norm)),
                "mean_approach_speed": float(np.mean(speed_along)),
                "std_approach_speed": float(np.std(speed_along)),
                "max_speed_norm": float(np.max(speed_norm)),
                "left_duty_factor": float(np.mean(left_contact)),
                "right_duty_factor": float(np.mean(right_contact)),
                "double_support_ratio": float(np.mean(left_contact & right_contact)),
                "flight_ratio": float(np.mean((~left_contact) & (~right_contact))),
                "contact_alternation_ratio": float(np.mean(left_contact ^ right_contact)),
                "step_frequency_hz": step_frequency_hz,
                "gait_phase_lag_deg": phase_lag_deg,
                "gait_dominant_frequency_hz": dom_freq,
                "roll_rms_deg": float(np.sqrt(np.mean(roll ** 2))),
                "pitch_rms_deg": float(np.sqrt(np.mean(pitch ** 2))),
                "roll_pitch_rms_deg": float(np.sqrt(np.mean(roll ** 2 + pitch ** 2))),
                "base_height_mean": float(np.mean(flat["base_pos_z"][mask])),
                "base_height_std": float(np.std(flat["base_pos_z"][mask])),
                "base_vz_rms": float(np.sqrt(np.mean(torso_vz ** 2))),
                "base_ang_vel_xy_rms": float(np.sqrt(np.mean(base_ang_vel_xy ** 2))),
            }
        )
    return metrics


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_speed(flat, scenarios, warmup_s, output_path):
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.4 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for env_id, (ax, scenario) in enumerate(zip(axes, scenarios)):
        mask = (flat["env_id"] == env_id) & (flat["active"] > 0) & (flat["time"] >= warmup_s)
        if not np.any(mask):
            ax.set_title(f"{scenario['name']} (no active samples after warmup)")
            ax.grid(alpha=0.3)
            continue
        t = flat["time"][mask]
        vx = flat["torso_vx_world"][mask]
        vy = flat["torso_vy_world"][mask]
        speed = np.sqrt(vx ** 2 + vy ** 2)
        ax.plot(t, vx, label="vx world")
        ax.plot(t, vy, label="vy world")
        ax.plot(t, speed, label="speed norm", linewidth=2.0)
        ax.set_title(scenario["name"])
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("k1_move_amp checkpoint 3200 speed diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gait_contacts(flat, scenarios, warmup_s, output_path):
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.6 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for env_id, (ax, scenario) in enumerate(zip(axes, scenarios)):
        mask = (flat["env_id"] == env_id) & (flat["active"] > 0) & (flat["time"] >= warmup_s)
        if not np.any(mask):
            ax.set_title(f"{scenario['name']} (no active samples after warmup)")
            ax.grid(alpha=0.3)
            continue
        t = flat["time"][mask]
        left_z = flat["left_foot_z_base"][mask]
        right_z = flat["right_foot_z_base"][mask]
        left_c = flat["left_contact"][mask] > 0.5
        right_c = flat["right_contact"][mask] > 0.5
        ax.plot(t, left_z, label="left foot z")
        ax.plot(t, right_z, label="right foot z")
        ax.fill_between(t, left_z.min() - 0.01, left_z.min() - 0.005, where=left_c, alpha=0.35, step="pre", label="left contact")
        ax.fill_between(t, left_z.min() - 0.025, left_z.min() - 0.02, where=right_c, alpha=0.35, step="pre", label="right contact")
        ax.set_title(scenario["name"])
        ax.set_ylabel("base-frame z [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Foot height and contact timing", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trunk_stability(flat, scenarios, warmup_s, output_path):
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.6 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for env_id, (ax, scenario) in enumerate(zip(axes, scenarios)):
        mask = (flat["env_id"] == env_id) & (flat["active"] > 0) & (flat["time"] >= warmup_s)
        if not np.any(mask):
            ax.set_title(f"{scenario['name']} (no active samples after warmup)")
            ax.grid(alpha=0.3)
            continue
        t = flat["time"][mask]
        roll = np.degrees(flat["roll"][mask])
        pitch = np.degrees(flat["pitch"][mask])
        z = flat["base_pos_z"][mask]
        ax.plot(t, roll, label="roll [deg]")
        ax.plot(t, pitch, label="pitch [deg]")
        ax.plot(t, (z - np.mean(z)) * 100.0, label="base z centered [cm]")
        ax.set_title(scenario["name"])
        ax.set_ylabel("stability")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Trunk stability timeseries", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_bars(metrics, output_path):
    scenarios = [row["scenario"] for row in metrics]
    x = np.arange(len(scenarios))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    axes[0].bar(x, [row["target_progress_ratio"] for row in metrics])
    axes[0].set_title("Target progress ratio")
    axes[0].set_ylabel("ratio")

    axes[1].bar(x, [row["step_frequency_hz"] for row in metrics])
    axes[1].set_title("Step frequency")
    axes[1].set_ylabel("Hz")

    axes[2].bar(x, [row["gait_phase_lag_deg"] for row in metrics])
    axes[2].set_title("Left-right phase lag")
    axes[2].set_ylabel("deg")

    axes[3].bar(x - 0.18, [row["roll_rms_deg"] for row in metrics], width=0.36, label="roll rms")
    axes[3].bar(x + 0.18, [row["pitch_rms_deg"] for row in metrics], width=0.36, label="pitch rms")
    axes[3].set_title("Trunk roll/pitch RMS")
    axes[3].set_ylabel("deg")
    axes[3].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20)
        ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metrics, output_path, checkpoint_path, cli_args):
    lines = [
        "# k1_move_amp Checkpoint Diagnostics",
        "",
        f"- task: `{cli_args.task}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- duration_s: `{cli_args.duration_s}`",
        f"- warmup_s: `{cli_args.warmup_s}`",
        "",
        "## Summary Metrics",
        "",
        "| scenario | target_progress | target_progress_ratio | mean_approach_speed | step_frequency_hz | gait_phase_lag_deg | roll_rms_deg | pitch_rms_deg | base_vz_rms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {scenario} | {target_progress:.4f} | {target_progress_ratio:.4f} | {mean_approach_speed:.4f} | {step_frequency_hz:.4f} | {gait_phase_lag_deg:.2f} | {roll_rms_deg:.3f} | {pitch_rms_deg:.3f} | {base_vz_rms:.4f} |".format(
                **row
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = os.path.abspath(cli_args.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    scenarios = scenario_table(cli_args.target_x, cli_args.target_y)
    runtime_args = make_runtime_args(cli_args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli_args.task)
    configure_env(env_cfg, cli_args, len(scenarios))
    env, _ = task_registry.make_env(name=cli_args.task, args=runtime_args, env_cfg=env_cfg)

    train_cfg.runner.resume = False
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=cli_args.task,
        args=runtime_args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    set_fixed_targets(env, scenarios)
    env.compute_observations()
    obs = env.get_observations()

    max_steps = int(round(cli_args.duration_s / env.dt))
    warmup_steps = int(round(cli_args.warmup_s / env.dt))
    active_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    records = []

    with torch.no_grad():
        for step_idx in range(max_steps):
            set_fixed_targets(env, scenarios)
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _ = env.step(actions.detach())
            if step_idx >= warmup_steps:
                records.append(collect_step(env, active_mask, scenarios[: env.num_envs], step_idx))
            newly_done = dones > 0
            active_mask = active_mask & (~newly_done)
            if not bool(active_mask.any()):
                break

    if not records:
        raise RuntimeError("No rollout records collected.")

    flat = flatten_records(records)
    metrics = scenario_metrics(flat, scenarios[: env.num_envs], cli_args.warmup_s)

    npz_path = output_dir / "raw_rollouts.npz"
    save_npz(flat, npz_path)
    write_csv(metrics, output_dir / "metrics_summary.csv")

    foot_indices = env.contact_feet_indices.detach().cpu().tolist()
    body_names = getattr(env, "body_names", None)
    if body_names is not None:
        foot_body_names = [body_names[i] for i in foot_indices]
    else:
        foot_body_names = [f"body_{i}" for i in foot_indices]

    metadata = {
        "task": cli_args.task,
        "checkpoint_path": checkpoint_path,
        "duration_s": cli_args.duration_s,
        "warmup_s": cli_args.warmup_s,
        "dt": float(env.dt),
        "num_envs": env.num_envs,
        "scenarios": scenarios[: env.num_envs],
        "foot_indices": foot_indices,
        "foot_body_names": foot_body_names,
        "available_fields": sorted(flat.keys()),
        "env_cfg": {
            "episode_length_s": env_cfg.env.episode_length_s,
            "randomization_disabled": True,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plot_speed(flat, scenarios[: env.num_envs], cli_args.warmup_s, figures_dir / "fig_speed_timeseries.png")
    plot_gait_contacts(flat, scenarios[: env.num_envs], cli_args.warmup_s, figures_dir / "fig_gait_contacts.png")
    plot_trunk_stability(flat, scenarios[: env.num_envs], cli_args.warmup_s, figures_dir / "fig_trunk_stability.png")
    plot_summary_bars(metrics, figures_dir / "fig_summary_metrics.png")
    write_report(metrics, output_dir / "diagnostics_report.md", checkpoint_path, cli_args)

    print(f"Saved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
