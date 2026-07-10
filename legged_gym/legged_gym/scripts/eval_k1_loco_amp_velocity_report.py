#!/usr/bin/env python3
"""Chinese velocity, trunk stability, and gait-phase report for k1_loco_amp."""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "legged_gym"
RSL_RL_ROOT = REPO_ROOT / "rsl_rl"
for path in (PACKAGE_ROOT, RSL_RL_ROOT, Path(__file__).resolve().parent):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

for module_name in ("onnxruntime", "git", "pydelatin", "pyfqmr"):
    if module_name not in sys.modules:
        stub = types.ModuleType(module_name)
        stub.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
        if module_name == "pydelatin":
            class _MissingDelatin:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("pydelatin is not installed in this environment.")
            stub.Delatin = _MissingDelatin
        if module_name == "git":
            class _MissingGitRepo:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("GitPython is not installed in this environment.")
            stub.Repo = _MissingGitRepo
        sys.modules[module_name] = stub

LEGACY_INIT = PACKAGE_ROOT / "legged_gym" / "__init__.py"
if "legged_gym" not in sys.modules:
    spec = importlib.util.spec_from_file_location("legged_gym", LEGACY_INIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legged_gym"] = module
    spec.loader.exec_module(module)

import isaacgym  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaacgym import gymapi
from matplotlib import font_manager
from matplotlib.text import Text

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry

import play_policy_diagnostics as diag

CJK_FONT = None


DIRS: List[Tuple[str, Tuple[float, float]]] = [
    ("前", (1.0, 0.0)),
    ("前左", (1.0, 1.0)),
    ("左", (0.0, 1.0)),
    ("后左", (-1.0, 1.0)),
    ("后", (-1.0, 0.0)),
    ("后右", (-1.0, -1.0)),
    ("右", (0.0, -1.0)),
    ("前右", (1.0, -1.0)),
]

AMP_NAMES = {
    "前": "forward",
    "前左": "diagonal_left",
    "左": "leftstep",
    "后左": "backward",
    "后": "backward",
    "后右": "backward",
    "右": "rightstep",
    "前右": "diagonal_right",
}


def setup_chinese_font() -> None:
    global CJK_FONT
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/todesk/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    for path in candidates:
        if path.exists():
            CJK_FONT = font_manager.FontProperties(fname=str(path))
            break
    plt.rcParams["axes.unicode_minus"] = False


def apply_cjk_font(fig: plt.Figure) -> None:
    if CJK_FONT is None:
        return
    for text in fig.findobj(Text):
        text.set_fontproperties(CJK_FONT)


def parse_args() -> argparse.Namespace:
    default_checkpoint = (
        Path(LEGGED_GYM_ROOT_DIR) / "logs" / "k1_loco_amp" / "debug" / "model_2800.pt"
    )
    parser = argparse.ArgumentParser(description="Evaluate k1_loco_amp checkpoint 2800.")
    parser.add_argument("--task", type=str, default="k1_loco_amp")
    parser.add_argument("--exptid", type=str, default="debug")
    parser.add_argument("--checkpoint", type=int, default=2800)
    parser.add_argument("--checkpoint-path", type=str, default=str(default_checkpoint))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(LEGGED_GYM_ROOT_DIR) / "logs" / "diagnostics" / "model_2800_k1_loco_amp_velocity_eval"),
    )
    parser.add_argument("--rl-device", type=str, default="cpu")
    parser.add_argument("--sim-device", type=str, default="cpu")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--speed-episodes", type=int, default=24)
    parser.add_argument("--direction-episodes", type=int, default=24)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=0.70)
    parser.add_argument("--speed-step", type=float, default=0.05)
    parser.add_argument("--direction-component-speed", type=float, default=0.30)
    parser.add_argument("--plot-only", action="store_true", default=False)
    return parser.parse_args()


def make_runtime_args(args: argparse.Namespace, num_envs: int) -> SimpleNamespace:
    sim_on_gpu = "cuda" in args.sim_device
    return SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        headless=args.headless,
        num_threads=0,
        use_gpu=sim_on_gpu,
        use_gpu_pipeline=sim_on_gpu,
        subscenes=0,
        device=args.rl_device,
        num_envs=num_envs,
        seed=args.seed,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=args.checkpoint,
        exptid=args.exptid,
        resumeid=None,
        horovod=False,
    )


def safe_set(obj: Any, path: str, value: Any) -> None:
    cur = obj
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(cur, part):
            return
        cur = getattr(cur, part)
    if hasattr(cur, parts[-1]):
        setattr(cur, parts[-1], value)


def configure_env(env_cfg: Any, num_envs: int, duration_s: float) -> None:
    safe_set(env_cfg, "env.num_envs", int(num_envs))
    safe_set(env_cfg, "env.play", True)
    safe_set(env_cfg, "env.episode_length_s", float(duration_s) + 0.5)
    safe_set(env_cfg, "noise.add_noise", False)
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
        safe_set(env_cfg, path, value)


def make_env_policy(args: argparse.Namespace, num_envs: int):
    runtime_args = make_runtime_args(args, num_envs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_env(env_cfg, num_envs, args.duration_s)
    env, _ = task_registry.make_env(name=args.task, args=runtime_args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=runtime_args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(args.checkpoint_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs = diag.rebuild_obs_history(env)
    return env, policy, obs, train_cfg


def command_from_unit(unit_xy: Tuple[float, float], speed: float) -> Tuple[float, float, float]:
    x, y = unit_xy
    norm = math.sqrt(x * x + y * y)
    if norm < 1.0e-6:
        return (0.0, 0.0, 0.0)
    return (speed * x / norm, speed * y / norm, 0.0)


def component_command(unit_xy: Tuple[float, float], component_speed: float) -> Tuple[float, float, float]:
    x, y = unit_xy
    return (
        float(np.sign(x) * component_speed) if abs(x) > 1.0e-6 else 0.0,
        float(np.sign(y) * component_speed) if abs(y) > 1.0e-6 else 0.0,
        0.0,
    )


def set_command(env: Any, command: Sequence[float], amp_name: str, env_ids: torch.Tensor) -> None:
    diag.force_command(env, command, amp_name, env_ids)


def reset_for_command(env: Any, command: Sequence[float], amp_name: str) -> torch.Tensor:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    if hasattr(env, "episode_length_buf"):
        env.episode_length_buf[:] = 0
    if hasattr(env, "reset_buf"):
        env.reset_buf[:] = 0
    if hasattr(env, "time_out_buf"):
        env.time_out_buf[:] = False
    set_command(env, command, amp_name, env_ids)
    return diag.rebuild_obs_history(env)


def run_group(
    env: Any,
    policy: Any,
    label: str,
    command: Sequence[float],
    amp_name: str,
    duration_s: float,
    episode_offset: int,
    foot_indices: Sequence[int],
    contact_threshold: float,
) -> Dict[str, np.ndarray]:
    storage: Dict[str, List[np.ndarray]] = {}
    obs = reset_for_command(env, command, amp_name)
    active_ids = torch.arange(env.num_envs, device=env.device)
    metadata: Dict[str, Any] = {"missing_fields": []}
    steps = int(round(duration_s / float(env.dt)))
    for step_i in range(steps):
        set_command(env, command, amp_name, active_ids)
        with torch.no_grad():
            actions = policy(obs.detach())
        step_ret = env.step(actions.detach())
        obs = step_ret[0]
        dones = step_ret[3]
        set_command(env, command, amp_name, active_ids)
        diag.append_rows(
            storage,
            diag.collect_step(
                env,
                actions,
                "k1_loco_amp_velocity_eval",
                label,
                command,
                step_i,
                env.num_envs,
                episode_offset,
                contact_threshold,
                list(foot_indices),
                metadata,
            ),
        )
        if torch.any(dones):
            reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            set_command(env, command, amp_name, reset_ids)
            obs = diag.rebuild_obs_history(env)
    return diag.finalize_raw(storage)


def merge_raw(chunks: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    storage: Dict[str, List[np.ndarray]] = {}
    for raw in chunks:
        for key, value in raw.items():
            storage.setdefault(key, []).append(value)
    return diag.finalize_raw(storage)


def aggregate_mean(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for group_key, items in groups.items():
        row = {key: value for key, value in zip(keys, group_key)}
        numeric = sorted({k for item in items for k, v in item.items() if isinstance(v, (int, float, np.floating))})
        for key in numeric:
            vals = np.asarray([float(item.get(key, np.nan)) for item in items], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
            row[f"{key}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else math.nan
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def select_mask(raw: Dict[str, np.ndarray], direction: str, episode_id: int | None = None) -> np.ndarray:
    mask = raw["command_name"].astype(str) == direction
    if episode_id is not None:
        mask = mask & (raw["episode_id"].astype(int) == int(episode_id))
    return mask


def first_episode(raw: Dict[str, np.ndarray], direction: str) -> int | None:
    mask = raw["command_name"].astype(str) == direction
    eps = sorted(set(raw["episode_id"][mask].astype(int).tolist()))
    return eps[0] if eps else None


def direction_metrics_from_raw(raw: Dict[str, np.ndarray], direction_rows: List[Dict[str, Any]], warmup_s: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in direction_rows:
        direction = str(row["command_name"])
        episode_id = first_episode(raw, direction)
        if episode_id is None:
            continue
        mask = select_mask(raw, direction, episode_id) & (raw["time"].astype(float) >= warmup_s)
        if not np.any(mask):
            continue
        left_z = raw["left_foot_z_base"][mask]
        right_z = raw["right_foot_z_base"][mask]
        time = raw["time"][mask]
        freq, phase = diag_metrics_phase(time, left_z, right_z)
        out[direction] = {
            "roll_rms_deg": math.degrees(float(row.get("roll_pitch_rms_mean", np.nan))),
            "base_z_vel_rms": float(row.get("base_z_vel_rms_mean", np.nan)),
            "step_frequency_hz": contact_step_frequency(time, raw["left_contact"][mask], raw["right_contact"][mask]),
            "phase_lag_deg": phase,
            "dominant_frequency_hz": freq,
        }
    return out


def contact_step_frequency(time: np.ndarray, left_contact: np.ndarray, right_contact: np.ndarray) -> float:
    intervals: List[float] = []
    for signal in (left_contact > 0.5, right_contact > 0.5):
        edges = np.where((signal[1:] == 1) & (signal[:-1] == 0))[0] + 1
        if len(edges) >= 2:
            intervals.extend(np.diff(time[edges]).astype(float).tolist())
    return float(1.0 / np.mean(intervals)) if intervals else math.nan


def wrap_deg(angle: float) -> float:
    return float(((angle + 180.0) % 360.0) - 180.0)


def diag_metrics_phase(time: np.ndarray, left_z: np.ndarray, right_z: np.ndarray) -> Tuple[float, float]:
    if len(time) < 8:
        return math.nan, math.nan
    dt = float(np.median(np.diff(time)))
    if not math.isfinite(dt) or dt <= 0.0:
        return math.nan, math.nan
    left = left_z - np.mean(left_z)
    right = right_z - np.mean(right_z)
    freq = np.fft.rfftfreq(len(time), dt)
    left_fft = np.fft.rfft(left)
    right_fft = np.fft.rfft(right)
    valid = (freq >= 0.3) & (freq <= 5.0)
    if not np.any(valid):
        return math.nan, math.nan
    power = np.abs(left_fft) ** 2 + np.abs(right_fft) ** 2
    valid_idx = np.where(valid)[0]
    dom_idx = valid_idx[int(np.argmax(power[valid]))]
    phase = math.degrees(np.angle(right_fft[dom_idx]) - np.angle(left_fft[dom_idx]))
    return float(freq[dom_idx]), wrap_deg(phase)


def plot_speed_scan(rows: List[Dict[str, Any]], output_path: Path) -> None:
    directions = [name for name, _ in DIRS]
    max_speed: Dict[str, float] = {}
    max_error: Dict[str, float] = {}
    command_at_max: Dict[str, float] = {}
    best_rows: Dict[str, Dict[str, Any]] = {}
    for direction in directions:
        group = [r for r in rows if r["command_name"] == direction]
        group = sorted(group, key=lambda r: float(r["command_speed"]))
        y = np.asarray([float(r["actual_speed_along_command_mean"]) for r in group])
        err = np.asarray([float(r["mean_tracking_error_xy_mean"]) for r in group])
        cmd = np.asarray([float(r["command_speed"]) for r in group])
        if len(y) == 0 or not np.isfinite(y).any():
            continue
        best_idx = int(np.nanargmax(y))
        best_row = group[best_idx]
        cmd_vx = float(best_row.get("cmd_vx_mean", 0.0))
        cmd_vy = float(best_row.get("cmd_vy_mean", 0.0))
        cmd_wz = float(best_row.get("cmd_yaw_rate_mean", 0.0))
        base_vx = float(best_row.get("base_vx_mean", 0.0))
        base_vy = float(best_row.get("base_vy_mean", 0.0))
        base_wz = float(best_row.get("base_ang_vel_z_mean", 0.0))
        max_speed[direction] = float(y[best_idx])
        max_error[direction] = float(
            math.sqrt((base_vx - cmd_vx) ** 2 + (base_vy - cmd_vy) ** 2 + (base_wz - cmd_wz) ** 2)
        )
        command_at_max[direction] = float(cmd[best_idx])
        best_rows[direction] = best_row

    fig = plt.figure(figsize=(18.0, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    table_ax = fig.add_subplot(gs[0, 1])

    max_abs = 0.0
    for direction in directions:
        row = best_rows.get(direction)
        if row is None:
            continue
        cmd_vx = float(row.get("cmd_vx_mean", 0.0))
        cmd_vy = float(row.get("cmd_vy_mean", 0.0))
        base_vx = float(row.get("base_vx_mean", 0.0))
        base_vy = float(row.get("base_vy_mean", 0.0))
        max_abs = max(max_abs, abs(cmd_vx), abs(cmd_vy), abs(base_vx), abs(base_vy))

    lim = max(0.35, math.ceil((max_abs + 0.08) * 10.0) / 10.0)
    ax.axhline(0.0, color="0.75", linewidth=0.9)
    ax.axvline(0.0, color="0.75", linewidth=0.9)
    for direction in directions:
        row = best_rows.get(direction)
        if row is None:
            continue
        cmd_vx = float(row.get("cmd_vx_mean", 0.0))
        cmd_vy = float(row.get("cmd_vy_mean", 0.0))
        cmd_wz = float(row.get("cmd_yaw_rate_mean", 0.0))
        base_vx = float(row.get("base_vx_mean", 0.0))
        base_vy = float(row.get("base_vy_mean", 0.0))
        base_wz = float(row.get("base_ang_vel_z_mean", 0.0))
        ax.arrow(
            0.0,
            0.0,
            cmd_vx,
            cmd_vy,
            linestyle="--",
            width=0.002,
            head_width=0.018,
            head_length=0.025,
            length_includes_head=True,
            color="0.68",
            alpha=0.85,
        )
        ax.arrow(
            0.0,
            0.0,
            base_vx,
            base_vy,
            width=0.004,
            head_width=0.024,
            head_length=0.032,
            length_includes_head=True,
            color="tab:blue",
            alpha=0.92,
        )
        label_x = cmd_vx * 1.10 + (0.022 if cmd_vx >= 0 else -0.022)
        label_y = cmd_vy * 1.10 + (0.022 if cmd_vy >= 0 else -0.022)
        ax.text(label_x, label_y, direction, fontsize=10, ha="center", va="center")
        ax.plot([cmd_vx, base_vx], [cmd_vy, base_vy], color="tab:red", linewidth=1.2, alpha=0.55)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("vx [m/s]")
    ax.set_ylabel("vy [m/s]")
    ax.grid(alpha=0.25)
    ax.set_title("命令速度向量 vs 实际平均速度向量", fontsize=15, pad=14)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="0.68", edgecolor="0.68", alpha=0.55, label="command"),
            Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.85, label="actual"),
        ],
        loc="upper right",
        fontsize=10,
    )

    table_ax.axis("off")
    table_rows = [
        [
            d,
            "({:.2f}, {:.2f}, {:.2f})".format(
                float(best_rows[d].get("cmd_vx_mean", 0.0)),
                float(best_rows[d].get("cmd_vy_mean", 0.0)),
                float(best_rows[d].get("cmd_yaw_rate_mean", 0.0)),
            ) if d in best_rows else "",
            "({:.2f}, {:.2f}, {:.2f})".format(
                float(best_rows[d].get("base_vx_mean", 0.0)),
                float(best_rows[d].get("base_vy_mean", 0.0)),
                float(best_rows[d].get("base_ang_vel_z_mean", 0.0)),
            ) if d in best_rows else "",
            f"{max_speed.get(d, math.nan):.3f}",
            f"{max_error.get(d, math.nan):.3f}",
        ]
        for d in directions
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["方向", "命令(vx,vy,wz)", "实际(vx,vy,wz)", "最大实际速度", "3D偏移"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.05, 3.0)
    if CJK_FONT is not None:
        for cell in table.get_celld().values():
            cell.get_text().set_fontproperties(CJK_FONT)
    table_ax.set_title("速度跟随数值汇总", fontsize=13, pad=14)
    fig.suptitle("速度跟随测试：箭头显示 vx/vy，标注与表格显示三维命令", fontsize=16)
    apply_cjk_font(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_trunk(raw: Dict[str, np.ndarray], metrics: Dict[str, Dict[str, float]], warmup_s: float, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), sharex=False)
    for ax, (direction, _) in zip(axes.flat, DIRS):
        ep = first_episode(raw, direction)
        if ep is None:
            ax.set_title(f"{direction}: 无数据")
            continue
        mask = select_mask(raw, direction, ep) & (raw["time"].astype(float) >= warmup_s)
        t = raw["time"][mask] - warmup_s
        roll = np.degrees(raw["roll"][mask])
        pitch = np.degrees(raw["pitch"][mask])
        z_cm = (raw["base_pos_z"][mask] - np.mean(raw["base_pos_z"][mask])) * 100.0
        ax.plot(t, roll, label="roll", linewidth=1.25)
        ax.plot(t, pitch, label="pitch", linewidth=1.25)
        ax.plot(t, z_cm, label="躯干高度波动(cm)", linewidth=1.25)
        m = metrics.get(direction, {})
        ax.set_title(f"{direction} | z_v RMS={m.get('base_z_vel_rms', math.nan):.3f}", fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_xlabel("时间 [s]")
        ax.set_ylabel("角度/高度")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("躯干稳定性测试：八方向，vx/vy 分量使用 ±0.3", fontsize=15)
    apply_cjk_font(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gait(raw: Dict[str, np.ndarray], metrics: Dict[str, Dict[str, float]], warmup_s: float, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), sharex=False)
    for ax, (direction, _) in zip(axes.flat, DIRS):
        ep = first_episode(raw, direction)
        if ep is None:
            ax.set_title(f"{direction}: 无数据")
            continue
        mask = select_mask(raw, direction, ep) & (raw["time"].astype(float) >= warmup_s)
        t = raw["time"][mask] - warmup_s
        left_z = raw["left_foot_z_base"][mask]
        right_z = raw["right_foot_z_base"][mask]
        left_c = raw["left_contact"][mask] > 0.5
        right_c = raw["right_contact"][mask] > 0.5
        low = min(float(np.min(left_z)), float(np.min(right_z)))
        ax.plot(t, left_z, label="左脚高度", linewidth=1.25)
        ax.plot(t, right_z, label="右脚高度", linewidth=1.25)
        ax.fill_between(t, low - 0.015, low - 0.006, where=left_c, step="pre", alpha=0.32, label="左脚接触")
        ax.fill_between(t, low - 0.030, low - 0.021, where=right_c, step="pre", alpha=0.32, label="右脚接触")
        m = metrics.get(direction, {})
        ax.set_title(
            f"{direction} | 相位差={m.get('phase_lag_deg', math.nan):.0f}°, 频率={m.get('step_frequency_hz', math.nan):.2f}Hz",
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        ax.set_xlabel("时间 [s]")
        ax.set_ylabel("脚高度 [m]")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("步态相位测试：八方向，vx/vy 分量使用 ±0.3", fontsize=15)
    apply_cjk_font(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    args: argparse.Namespace,
    speed_rows: List[Dict[str, Any]],
    direction_rows: List[Dict[str, Any]],
    gait_metrics: Dict[str, Dict[str, float]],
    figures: Sequence[Path],
) -> None:
    best_by_dir = {}
    for direction, _ in DIRS:
        group = [r for r in speed_rows if r["command_name"] == direction]
        if not group:
            continue
        best = max(group, key=lambda r: float(r["actual_speed_along_command_mean"]))
        best_by_dir[direction] = best

    lines = [
        "# k1_loco_amp 速度与步态评估报告",
        "",
        "## 评估设置",
        "",
        f"- 任务：`{args.task}`",
        f"- 实验：`--exptid={args.exptid}`",
        f"- checkpoint：`{args.checkpoint_path}`",
        f"- 速度扫描样本：每个方向、每个速度 `{args.speed_episodes}` 条 rollout",
        f"- 八方向稳定性/步态样本：每个方向 `{args.direction_episodes}` 条 rollout",
        f"- 单条 rollout 时长：`{args.duration_s:.2f}s`，统计跳过前 `{args.warmup_s:.2f}s`",
        f"- 八方向测试命令：vx/vy 分量使用 `±{args.direction_component_speed:.2f}`",
        "",
        "## 生成图片",
        "",
    ]
    lines.extend(f"- `{figure}`" for figure in figures)
    lines.extend([
        "",
        "## 速度测试结论",
        "",
        "| 方向 | 命令(vx,vy,wz) | 实际(vx,vy,wz) | 最大实际速度 | 3D偏移 |",
        "| --- | --- | --- | ---: | ---: |",
    ])
    for direction, _ in DIRS:
        best = best_by_dir.get(direction)
        if not best:
            continue
        cmd_vx = float(best.get("cmd_vx_mean", 0.0))
        cmd_vy = float(best.get("cmd_vy_mean", 0.0))
        cmd_wz = float(best.get("cmd_yaw_rate_mean", 0.0))
        base_vx = float(best.get("base_vx_mean", 0.0))
        base_vy = float(best.get("base_vy_mean", 0.0))
        base_wz = float(best.get("base_ang_vel_z_mean", 0.0))
        err_3d = math.sqrt((base_vx - cmd_vx) ** 2 + (base_vy - cmd_vy) ** 2 + (base_wz - cmd_wz) ** 2)
        lines.append(
            "| {direction} | ({cmd_vx:.2f}, {cmd_vy:.2f}, {cmd_wz:.2f}) | ({base_vx:.2f}, {base_vy:.2f}, {base_wz:.2f}) | {actual:.3f} | {err:.3f} |".format(
                direction=direction,
                cmd_vx=cmd_vx,
                cmd_vy=cmd_vy,
                cmd_wz=cmd_wz,
                base_vx=base_vx,
                base_vy=base_vy,
                base_wz=base_wz,
                actual=float(best["actual_speed_along_command_mean"]),
                err=err_3d,
            )
        )

    lines.extend([
        "",
        "## 躯干稳定性与步态相位",
        "",
        "| 方向 | roll/pitch RMS | 躯干z速度RMS | 步频(Hz) | 左右脚相位差(deg) | 主频(Hz) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    by_direction = {str(row["command_name"]): row for row in direction_rows}
    for direction, _ in DIRS:
        row = by_direction.get(direction, {})
        gm = gait_metrics.get(direction, {})
        lines.append(
            "| {direction} | {rp:.3f} | {zv:.3f} | {sf:.3f} | {ph:.1f} | {df:.3f} |".format(
                direction=direction,
                rp=float(row.get("roll_pitch_rms_mean", math.nan)),
                zv=float(row.get("base_z_vel_rms_mean", math.nan)),
                sf=float(gm.get("step_frequency_hz", math.nan)),
                ph=float(gm.get("phase_lag_deg", math.nan)),
                df=float(gm.get("dominant_frequency_hz", math.nan)),
            )
        )

    lines.extend([
        "",
        "## 说明",
        "",
        "- 速度图使用命令/实际速度向量图：灰色虚线箭头和蓝色箭头画在 vx/vy 平面；标注与右侧表格显示完整三维 (vx, vy, wz)，3D偏移同时包含 yaw-rate 误差。",
        "- 躯干稳定性图中，roll/pitch 使用角度，躯干高度波动用厘米显示，便于和角度曲线放在同一子图里观察。",
        "- 步态相位图中，脚高度曲线用于观察摆动幅度，底部色带表示左右脚接触地面区间；标题里的相位差来自左右脚高度主频的相位差。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_chinese_font()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    speed_dir = output_dir / "speed_scan"
    direction_dir = output_dir / "direction_pm03"
    for directory in (figures_dir, speed_dir, direction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    figures = [
        figures_dir / "fig_1_speed_following_max_speed.png",
        figures_dir / "fig_2_trunk_stability_8dir.png",
        figures_dir / "fig_3_gait_phase_8dir.png",
    ]

    if args.plot_only:
        speed_agg = read_csv_rows(speed_dir / "metrics_by_direction_speed.csv")
        direction_agg = read_csv_rows(direction_dir / "metrics_by_direction.csv")
        direction_raw = dict(np.load(direction_dir / "raw_rollouts.npz", allow_pickle=True))
        gait_metrics = direction_metrics_from_raw(direction_raw, direction_agg, args.warmup_s)
        plot_speed_scan(speed_agg, figures[0])
        plot_trunk(direction_raw, gait_metrics, args.warmup_s, figures[1])
        plot_gait(direction_raw, gait_metrics, args.warmup_s, figures[2])
        metadata = {
            "task": args.task,
            "exptid": args.exptid,
            "checkpoint": args.checkpoint,
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "duration_s": args.duration_s,
            "warmup_s": args.warmup_s,
            "speed_episodes": args.speed_episodes,
            "direction_episodes": args.direction_episodes,
            "direction_component_speed": args.direction_component_speed,
            "directions": [{"name": name, "unit": unit} for name, unit in DIRS],
            "plot_only": True,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(output_dir / "diagnostics_report.md", args, speed_agg, direction_agg, gait_metrics, figures)
        print(f"[eval] plot-only done: {output_dir}")
        return

    if not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    num_envs = max(int(args.speed_episodes), int(args.direction_episodes))
    env, policy, obs, train_cfg = make_env_policy(args, num_envs)
    metadata: Dict[str, Any] = {"warnings": [], "missing_fields": []}
    foot_indices = diag.find_foot_indices(env, metadata)
    if foot_indices is None:
        raise RuntimeError("Could not determine foot indices for gait diagnostics.")
    extra = {
        "torque_limits": diag.to_numpy(getattr(env, "torque_limits", None)) if hasattr(env, "torque_limits") else None,
    }

    speed_levels = np.arange(args.speed_step, args.max_speed + args.speed_step * 0.5, args.speed_step)
    speed_chunks: List[Dict[str, np.ndarray]] = []
    speed_episode_rows: List[Dict[str, Any]] = []
    episode_offset = 0
    for direction, unit in DIRS:
        for speed in speed_levels:
            command = command_from_unit(unit, float(speed))
            print(f"[eval] speed_scan {direction} command_speed={speed:.2f} command={command}")
            raw = run_group(
                env,
                policy,
                direction,
                command,
                AMP_NAMES[direction],
                args.duration_s,
                episode_offset,
                foot_indices,
                args.contact_threshold,
            )
            speed_chunks.append(raw)
            rows = diag.compute_episode_metrics(raw, args.warmup_s, extra)
            for row in rows:
                row["command_speed"] = float(speed)
            speed_episode_rows.extend(rows)
            episode_offset += env.num_envs

    speed_raw = merge_raw(speed_chunks)
    np.savez_compressed(speed_dir / "raw_rollouts.npz", **speed_raw)
    speed_agg = aggregate_mean(speed_episode_rows, ["command_name", "command_speed"])
    write_csv(speed_dir / "metrics_per_episode.csv", speed_episode_rows)
    write_csv(speed_dir / "metrics_by_direction_speed.csv", speed_agg)

    direction_chunks: List[Dict[str, np.ndarray]] = []
    direction_episode_rows: List[Dict[str, Any]] = []
    episode_offset = 0
    for direction, unit in DIRS:
        command = component_command(unit, args.direction_component_speed)
        print(f"[eval] direction_pm03 {direction} command={command}")
        raw = run_group(
            env,
            policy,
            direction,
            command,
            AMP_NAMES[direction],
            args.duration_s,
            episode_offset,
            foot_indices,
            args.contact_threshold,
        )
        direction_chunks.append(raw)
        direction_episode_rows.extend(diag.compute_episode_metrics(raw, args.warmup_s, extra))
        episode_offset += env.num_envs

    direction_raw = merge_raw(direction_chunks)
    np.savez_compressed(direction_dir / "raw_rollouts.npz", **direction_raw)
    direction_agg = aggregate_mean(direction_episode_rows, ["command_name"])
    write_csv(direction_dir / "metrics_per_episode.csv", direction_episode_rows)
    write_csv(direction_dir / "metrics_by_direction.csv", direction_agg)

    gait_metrics = direction_metrics_from_raw(direction_raw, direction_agg, args.warmup_s)
    plot_speed_scan(speed_agg, figures[0])
    plot_trunk(direction_raw, gait_metrics, args.warmup_s, figures[1])
    plot_gait(direction_raw, gait_metrics, args.warmup_s, figures[2])

    metadata.update(
        {
            "task": args.task,
            "exptid": args.exptid,
            "checkpoint": args.checkpoint,
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "runner_experiment_name": train_cfg.runner.experiment_name,
            "duration_s": args.duration_s,
            "warmup_s": args.warmup_s,
            "speed_episodes": args.speed_episodes,
            "direction_episodes": args.direction_episodes,
            "speed_levels": [float(x) for x in speed_levels],
            "direction_component_speed": args.direction_component_speed,
            "directions": [{"name": name, "unit": unit} for name, unit in DIRS],
            "foot_indices": foot_indices,
            "body_names": list(getattr(env, "body_names", [])),
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "diagnostics_report.md", args, speed_agg, direction_agg, gait_metrics, figures)
    print(f"[eval] done: {output_dir}")


if __name__ == "__main__":
    main()
