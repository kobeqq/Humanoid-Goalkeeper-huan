#!/usr/bin/env python3
"""Plot policy playback diagnostics produced by play_policy_diagnostics.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIXED_ORDER = ["stand", "forward", "backward", "leftstep", "rightstep", "diag_left", "diag_right"]
SWITCH_BOUNDARIES = [0, 3, 9, 15, 21, 27, 33, 36]
SWITCH_NAMES = ["stand", "forward", "backward", "leftstep", "rightstep", "diag_left", "stand"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot policy diagnostic figures.")
    parser.add_argument("--input-dir", default="logs/diagnostics/model_1400")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args(argv)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def read_metadata(input_dir: Path) -> Dict:
    for path in (input_dir / "metadata.json", input_dir / "fixed_commands" / "metadata.json"):
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return {}


def ordered(df: pd.DataFrame, key: str = "command_name") -> pd.DataFrame:
    if df.empty or key not in df.columns:
        return df
    order = {name: i for i, name in enumerate(FIXED_ORDER)}
    return df.assign(_order=df[key].map(order).fillna(999)).sort_values(["_order", key]).drop(columns=["_order"])


def col(df: pd.DataFrame, base: str) -> Optional[str]:
    if base in df.columns:
        return base
    mean = f"{base}_mean"
    if mean in df.columns:
        return mean
    return None


def err_col(df: pd.DataFrame, base: str) -> Optional[str]:
    std = f"{base}_std"
    return std if std in df.columns else None


def save(fig: plt.Figure, output_dir: Path, name: str, made: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=180)
    plt.close(fig)
    made.append(str(output_dir / name))


def plot_velocity_vectors(metrics: pd.DataFrame, output_dir: Path, made: List[str], skipped: List[str]) -> None:
    req = ["cmd_vx_mean", "cmd_vy_mean", "base_vx", "base_vy", "actual_speed_along_command_mean"]
    vx_col = col(metrics, "cmd_vx")
    vy_col = col(metrics, "cmd_vy")
    if metrics.empty or vx_col is None or vy_col is None:
        skipped.append("Figure A: missing command velocity columns")
        return
    actual_vx = col(metrics, "base_vx")
    actual_vy = col(metrics, "base_vy")
    if actual_vx is None or actual_vy is None:
        # Per-command metrics do not directly include mean base_vx/base_vy. Use
        # along-command speed to draw a conservative projection.
        along_col = col(metrics, "actual_speed_along_command")
        if along_col is None:
            skipped.append("Figure A: missing actual velocity columns")
            return
        cmd_norm = np.sqrt(metrics[vx_col] ** 2 + metrics[vy_col] ** 2).replace(0.0, np.nan)
        ux = metrics[vx_col] / cmd_norm
        uy = metrics[vy_col] / cmd_norm
        ax_vals = (ux * metrics[along_col]).fillna(0.0)
        ay_vals = (uy * metrics[along_col]).fillna(0.0)
    else:
        ax_vals = metrics[actual_vx]
        ay_vals = metrics[actual_vy]

    df = ordered(metrics)
    ax_vals = ax_vals.loc[df.index]
    ay_vals = ay_vals.loc[df.index]
    fig, ax = plt.subplots(figsize=(7, 7))
    for idx, row in df.iterrows():
        name = row["command_name"]
        ax.arrow(0, 0, row[vx_col], row[vy_col], linestyle="--", width=0.002, alpha=0.35, length_includes_head=True, color="tab:gray")
        ax.arrow(0, 0, ax_vals.loc[idx], ay_vals.loc[idx], width=0.004, alpha=0.9, length_includes_head=True, color="tab:blue")
        ax.text(row[vx_col] * 1.08 + 0.01, row[vy_col] * 1.08 + 0.01, str(name), fontsize=9)
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.axvline(0, color="0.8", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    lim = max(0.35, float(np.nanmax(np.abs([df[vx_col], df[vy_col], ax_vals, ay_vals])) + 0.08))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("vx [m/s]")
    ax.set_ylabel("vy [m/s]")
    ax.set_title("Commanded vs actual mean velocity")
    ax.legend(["command", "actual"], loc="upper right")
    save(fig, output_dir, "fig_A_velocity_vectors.png", made)


def bar_metric(metrics: pd.DataFrame, output_dir: Path, made: List[str], skipped: List[str], metric: str, ylabel: str, title: str, filename: str) -> None:
    y = col(metrics, metric)
    if metrics.empty or y is None:
        skipped.append(f"{filename}: missing {metric}")
        return
    e = err_col(metrics, metric)
    df = ordered(metrics)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(df["command_name"], df[y], yerr=df[e] if e else None, capsize=3, color="tab:blue", alpha=0.82)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    save(fig, output_dir, filename, made)


def representative(raw: Dict[str, np.ndarray], command: str) -> np.ndarray:
    if not raw or "command_name" not in raw or "episode_id" not in raw:
        return np.array([], dtype=bool)
    names = raw["command_name"].astype(str)
    eps = raw["episode_id"].astype(int)
    choices = sorted(set(eps[names == command].tolist()))
    if not choices:
        return np.array([], dtype=bool)
    return (names == command) & (eps == choices[0])


def plot_base_height(raw: Dict[str, np.ndarray], output_dir: Path, made: List[str], skipped: List[str]) -> None:
    if not raw or "base_pos_z" not in raw:
        skipped.append("Figure C: missing raw base_pos_z")
        return
    commands = ["backward", "forward", "leftstep", "rightstep"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for command in commands:
        mask = representative(raw, command)
        if mask.size and mask.any():
            ax.plot(raw["time"][mask], raw["base_pos_z"][mask], label=command)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("base height [m]")
    ax.set_title("Base height during policy playback")
    ax.legend()
    save(fig, output_dir, "fig_C_base_height_timeseries.png", made)


def plot_contact_raster(raw: Dict[str, np.ndarray], output_dir: Path, made: List[str], skipped: List[str]) -> None:
    if not raw or "left_contact" not in raw or "right_contact" not in raw:
        skipped.append("Figure D: missing contact columns")
        return
    commands = ["backward", "forward", "leftstep", "rightstep"]
    fig, axes = plt.subplots(len(commands), 1, figsize=(9, 6), sharex=True)
    for ax, command in zip(axes, commands):
        mask = representative(raw, command)
        if mask.size and mask.any():
            t = raw["time"][mask]
            left = raw["left_contact"][mask].astype(float)
            right = raw["right_contact"][mask].astype(float)
            ax.step(t, left + 1.0, where="post", label="left foot")
            ax.step(t, right, where="post", label="right foot")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["right foot", "left foot"])
        ax.set_title(command)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="upper right")
    save(fig, output_dir, "fig_D_contact_raster.png", made)


def plot_foot_trajectory(raw: Dict[str, np.ndarray], output_dir: Path, made: List[str], skipped: List[str]) -> None:
    required = ["left_foot_x_base", "left_foot_y_base", "left_foot_z_base", "right_foot_x_base", "right_foot_y_base", "right_foot_z_base"]
    if not raw or any(k not in raw for k in required):
        skipped.append("Figure E: missing base-frame foot position columns")
        return
    commands = ["forward", "backward", "leftstep", "rightstep"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, command in zip(axes.flat, commands):
        mask = representative(raw, command)
        if mask.size and mask.any():
            horiz = "x" if command in ("forward", "backward") else "y"
            ax.plot(raw[f"left_foot_{horiz}_base"][mask], raw["left_foot_z_base"][mask], label="left foot")
            ax.plot(raw[f"right_foot_{horiz}_base"][mask], raw["right_foot_z_base"][mask], label="right foot")
            ax.set_xlabel(f"foot {horiz} in base [m]")
            ax.set_ylabel("foot z in base [m]")
        ax.set_title(command)
    axes.flat[0].legend()
    save(fig, output_dir, "fig_E_foot_trajectory.png", made)


def plot_speed_scan(scan: pd.DataFrame, output_dir: Path, made: List[str], skipped: List[str]) -> None:
    x = col(scan, "command_speed")
    y = col(scan, "actual_speed_along_command")
    if scan.empty or x is None or y is None:
        skipped.append("Figure G: missing speed scan velocity metrics")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        for direction, group in scan.groupby("command_name"):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], marker="o", label=direction)
        max_x = max(0.3, float(scan[x].max()))
        ax.plot([0, max_x], [0, max_x], "--", color="0.5", label="ideal")
        ax.set_xlabel("command_speed [m/s]")
        ax.set_ylabel("actual_speed_along_command [m/s]")
        ax.set_title("Speed scan actual speed vs command speed")
        ax.legend()
        save(fig, output_dir, "fig_G_speed_scan_actual_vs_command.png", made)

    z = col(scan, "base_z_vel_rms")
    if scan.empty or x is None or z is None:
        skipped.append("Figure H: missing speed scan base_z_vel_rms")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        for direction, group in scan.groupby("command_name"):
            group = group.sort_values(x)
            ax.plot(group[x], group[z], marker="o", label=direction)
        ax.set_xlabel("command_speed [m/s]")
        ax.set_ylabel("base_z_vel_rms")
        ax.set_title("Speed scan vertical bounce")
        ax.legend()
        save(fig, output_dir, "fig_H_speed_scan_base_z_vel_rms.png", made)


def plot_switch(raw: Dict[str, np.ndarray], output_dir: Path, made: List[str], skipped: List[str]) -> None:
    required = ["time", "cmd_vx", "cmd_vy", "base_vx", "base_vy", "base_pos_z"]
    if not raw or any(k not in raw for k in required):
        skipped.append("Figure I: missing command switch raw columns")
        return
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    t = raw["time"]
    axes[0].plot(t, raw["cmd_vx"], "--", label="cmd_vx")
    axes[0].plot(t, raw["base_vx"], label="actual base_vx")
    axes[0].set_ylabel("vx [m/s]")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, raw["cmd_vy"], "--", label="cmd_vy")
    axes[1].plot(t, raw["base_vy"], label="actual base_vy")
    axes[1].set_ylabel("vy [m/s]")
    axes[1].legend(loc="upper right")
    axes[2].plot(t, raw["base_pos_z"], label="base height")
    axes[2].set_ylabel("height [m]")
    if "left_contact" in raw and "right_contact" in raw:
        axes[3].step(t, raw["left_contact"].astype(float) + 1.0, where="post", label="left foot")
        axes[3].step(t, raw["right_contact"].astype(float), where="post", label="right foot")
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(["right foot", "left foot"])
        axes[3].legend(loc="upper right")
    axes[3].set_xlabel("time [s]")
    for ax in axes:
        for boundary in SWITCH_BOUNDARIES:
            ax.axvline(boundary, color="0.75", linestyle="--", linewidth=0.8)
        for start, end, name in zip(SWITCH_BOUNDARIES[:-1], SWITCH_BOUNDARIES[1:], SWITCH_NAMES):
            ax.axvspan(start, end, color="0.93", alpha=0.25 if name == "stand" else 0.08)
    ymin, ymax = axes[0].get_ylim()
    for start, end, name in zip(SWITCH_BOUNDARIES[:-1], SWITCH_BOUNDARIES[1:], SWITCH_NAMES):
        axes[0].text((start + end) * 0.5, ymax, name, ha="center", va="top", fontsize=8)
    axes[0].set_title("Command switch response")
    save(fig, output_dir, "fig_I_command_switch_response.png", made)


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No data._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        vals = []
        for c in cols:
            value = row[c]
            if isinstance(value, float):
                vals.append(f"{value:.5g}" if math.isfinite(value) else "nan")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def auto_diagnosis(fixed: pd.DataFrame) -> List[str]:
    notes: List[str] = []
    if fixed.empty or "command_name" not in fixed.columns:
        return notes
    speed = col(fixed, "speed_ratio")
    bounce = col(fixed, "base_z_vel_rms")
    foot = col(fixed, "foot_xy_range_mean")
    no_contact = col(fixed, "no_contact_ratio")
    both = col(fixed, "both_contact_ratio")
    torque_limit = col(fixed, "torque_limit_ratio")
    joint_range = col(fixed, "joint_pos_range_mean")
    by_name = fixed.set_index("command_name")
    if speed and "backward" in by_name.index:
        back = by_name.loc["backward", speed]
        others = [name for name in ("forward", "leftstep", "rightstep") if name in by_name.index]
        if others and all(back > by_name.loc[name, speed] + 0.10 for name in others):
            notes.append("Backward tracking is better than other directions; this suggests the visual asymmetry is measurable in speed_ratio.")
    if bounce and "backward" in by_name.index:
        back = by_name.loc["backward", bounce]
        bouncier = [name for name in ("forward", "leftstep", "rightstep") if name in by_name.index and by_name.loc[name, bounce] > back * 1.2]
        if bouncier:
            notes.append("Forward or lateral commands show stronger vertical bouncing than backward, consistent with pogo-like playback.")
    if foot and "backward" in by_name.index:
        back = by_name.loc["backward", foot]
        smaller = [name for name in ("forward", "leftstep", "rightstep") if name in by_name.index and by_name.loc[name, foot] < back * 0.8]
        if smaller:
            notes.append("Forward or lateral commands have smaller foot swing amplitude, which may indicate small shuffling steps.")
    if no_contact and both:
        abnormal = fixed[(fixed[no_contact] > 0.25) | (fixed[both] > 0.75)]
        if not abnormal.empty:
            notes.append("Contact pattern may be pogo-like or non-alternating for commands with high no_contact or both_contact ratios.")
    if torque_limit and joint_range:
        suspect = fixed[(fixed[torque_limit] > 0.10) & (fixed[joint_range] < fixed[joint_range].median())]
        if not suspect.empty:
            notes.append("The policy may be using high torques without producing large joint motion for some commands.")
    if not notes:
        notes.append("The automatic rules did not find a strong directional diagnosis; inspect the figures before drawing conclusions.")
    return notes


def write_report(input_dir: Path, output_dir: Path, fixed: pd.DataFrame, scan: pd.DataFrame, metadata: Dict, made: List[str], skipped: List[str]) -> None:
    report = input_dir / "diagnostics_report.md"
    lines = [
        "# Policy Diagnostics Report",
        "",
        "## Experiment Configuration",
        "",
        f"- task: `{metadata.get('task', 'unknown')}`",
        f"- checkpoint: `{metadata.get('checkpoint_path', metadata.get('checkpoint_arg', 'unknown'))}`",
        f"- warmup_s: `{metadata.get('warmup_s', 'unknown')}`",
        f"- episode_length_s: `{metadata.get('episode_length_s', 'unknown')}`",
        f"- randomization overrides attempted: `{metadata.get('randomization_overrides_attempted', False)}`",
        f"- base_velocity_frame: `{metadata.get('base_velocity_frame', 'unknown')}`",
        "",
        "## Command Table",
        "",
        "```json",
        json.dumps(metadata.get("command_table", {}), indent=2),
        "```",
        "",
        "## Fixed Commands Metrics",
        "",
        markdown_table(fixed),
        "",
        "## Speed Scan Metrics",
        "",
        markdown_table(scan),
        "",
        "## Automatic Diagnosis",
        "",
    ]
    lines.extend(f"- {note}" for note in auto_diagnosis(fixed))
    lines.extend(["", "## Generated Figures", ""])
    lines.extend(f"- `{path}`" for path in made)
    if skipped:
        lines.extend(["", "## Skipped Outputs", ""])
        lines.extend(f"- {item}" for item in skipped)
    missing = metadata.get("missing_fields", [])
    if missing:
        lines.extend(["", "## Missing Fields", ""])
        lines.extend(f"- `{item}`" for item in missing)
    report.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    made: List[str] = []
    skipped: List[str] = []
    metadata = read_metadata(input_dir)

    fixed = ordered(read_csv(input_dir / "fixed_commands" / "metrics_by_command.csv"))
    fixed_raw = read_npz(input_dir / "fixed_commands" / "raw_rollouts.npz")
    scan = read_csv(input_dir / "speed_scan" / "metrics_by_command_speed.csv")
    switch_raw = read_npz(input_dir / "command_switch" / "raw_rollouts.npz")

    plot_velocity_vectors(fixed, output_dir, made, skipped)
    bar_metric(fixed, output_dir, made, skipped, "mean_tracking_error_xy", "mean_tracking_error_xy [m/s]", "Tracking error by command", "fig_B_tracking_error_by_command.png")
    bar_metric(fixed, output_dir, made, skipped, "speed_ratio", "speed_ratio", "Speed ratio by command", "fig_B2_speed_ratio_by_command.png")
    plot_base_height(fixed_raw, output_dir, made, skipped)
    bar_metric(fixed, output_dir, made, skipped, "base_z_vel_rms", "base_z_vel_rms", "Vertical velocity RMS by command", "fig_C2_base_z_vel_rms_by_command.png")
    plot_contact_raster(fixed_raw, output_dir, made, skipped)
    mode_cols = [col(fixed, name) for name in ("no_contact_ratio", "left_only_ratio", "right_only_ratio", "both_contact_ratio")]
    if fixed is not None and not fixed.empty and all(mode_cols):
        df = ordered(fixed)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bottom = np.zeros(len(df))
        labels = ["no_contact", "left_only", "right_only", "both_contact"]
        colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
        for label, c, color in zip(labels, mode_cols, colors):
            vals = df[c].to_numpy(dtype=float)
            ax.bar(df["command_name"], vals, bottom=bottom, label=label, color=color, alpha=0.82)
            bottom += vals
        ax.set_ylabel("ratio")
        ax.set_title("Contact mode ratio by command")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="upper right")
        save(fig, output_dir, "fig_D2_contact_mode_ratio_by_command.png", made)
    else:
        skipped.append("fig_D2_contact_mode_ratio_by_command.png: missing contact mode ratio columns")
    plot_foot_trajectory(fixed_raw, output_dir, made, skipped)
    bar_metric(fixed, output_dir, made, skipped, "foot_xy_range_mean", "foot_xy_range_mean [m]", "Foot swing amplitude by command", "fig_E2_foot_xy_range_by_command.png")
    bar_metric(fixed, output_dir, made, skipped, "joint_pos_range_mean", "joint_pos_range_mean [rad]", "Joint range by command", "fig_F_joint_range_by_command.png")
    if col(fixed, "torque_limit_ratio"):
        bar_metric(fixed, output_dir, made, skipped, "torque_limit_ratio", "torque_limit_ratio", "Torque limit ratio by command", "fig_F2_torque_limit_ratio_by_command.png")
    else:
        bar_metric(fixed, output_dir, made, skipped, "torque_rms", "torque_rms", "Torque RMS by command", "fig_F2_torque_rms_by_command.png")
    plot_speed_scan(scan, output_dir, made, skipped)
    plot_switch(switch_raw, output_dir, made, skipped)
    write_report(input_dir, output_dir, fixed, scan, metadata, made, skipped)

    print(f"[plot] wrote {len(made)} figures to {output_dir}")
    if skipped:
        print("[plot] skipped:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
