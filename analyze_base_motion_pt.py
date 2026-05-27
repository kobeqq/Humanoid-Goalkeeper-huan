#!/usr/bin/env python3
"""
Analyze robot base trajectory from a motion .pt file (LeggedGym / MotionLib format).

Expected dict keys (any subset ok):
  - base_position: (T, 3) or (1, T, 3) — world-frame xyz [m]
  - base_pose: (T, 4) quaternion (optional, for heading/yaw plots)
  - fps: scalar or in dict (optional)

Usage:
  python3 analyze_base_motion_pt.py --pt path/to/motion.pt
  python3 analyze_base_motion_pt.py --pt path/to/motion.pt --no-show --save base_traj.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _load_pt(path: str):
    import torch

    # Motion dicts are trusted local files; avoid weights_only for broad torch versions.
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=False)
    except TypeError:
        return torch.load(path, **kw)


def _to_numpy(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _squeeze_time_pos(pos: np.ndarray) -> np.ndarray:
    """Return (T, 3)."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 3 and pos.shape[0] == 1:
        pos = pos[0]
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"base_position must be (T,3) or (1,T,3), got shape {pos.shape}")
    return pos


def euler_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """q: (T,4) x,y,z,w (same as MotionLib / g1_utils) -> roll, pitch, yaw (T,3)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.stack([roll, pitch, yaw], axis=1)


def main():
    ap = argparse.ArgumentParser(description="Analyze base_position in motion .pt")
    ap.add_argument("--pt", type=str, required=True, help="Path to .pt motion file")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS for time axis (default: from file or 30)")
    ap.add_argument("--save", type=str, default=None, help="Save figure to this path (png/pdf)")
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (useful on headless servers)",
    )
    args = ap.parse_args()

    path = os.path.abspath(args.pt)
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = _load_pt(path)
    if not isinstance(data, dict):
        print(f"Expected dict at top level, got {type(data)}", file=sys.stderr)
        sys.exit(1)

    if "base_position" not in data:
        print("Keys in file:", list(data.keys()), file=sys.stderr)
        print("Missing required key: base_position", file=sys.stderr)
        sys.exit(1)

    pos = _squeeze_time_pos(_to_numpy(data["base_position"]))
    T = pos.shape[0]

    fps = args.fps
    if fps is None:
        if "fps" in data:
            fps = float(_to_numpy(data["fps"]).reshape(-1)[0])
        else:
            fps = 30.0
    t = np.arange(T, dtype=np.float64) / fps

    # Stats
    delta = np.diff(pos, axis=0)
    step_dist = np.linalg.norm(delta, axis=1)
    total_path = float(np.sum(step_dist))
    disp = float(np.linalg.norm(pos[-1] - pos[0]))
    print(f"File: {path}")
    print(f"Frames T = {T}, fps = {fps:.3f}, duration = {t[-1]:.3f} s")
    print(f"base_position xyz range:")
    for i, name in enumerate("xyz"):
        print(f"  {name}: min={pos[:, i].min():.6f}  max={pos[:, i].max():.6f}")
    print(f"displacement |p_end - p_start| = {disp:.6f} m")
    print(f"path length (sum ||Δp||)      = {total_path:.6f} m")
    print(f"mean step |Δp|                = {float(np.mean(step_dist)):.6f} m")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots. pip install matplotlib", file=sys.stderr)
        return

    has_quat = "base_pose" in data
    quat = _to_numpy(data["base_pose"]) if has_quat else None
    if has_quat:
        if quat.ndim == 3 and quat.shape[0] == 1:
            quat = quat[0]
        if quat.shape != (T, 4):
            print(
                f"Warning: base_pose shape {quat.shape} != ({T},4); skipping yaw plot",
                file=sys.stderr,
            )
            has_quat = False

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, pos[:, 0], label="x")
    ax1.plot(t, pos[:, 1], label="y")
    ax1.plot(t, pos[:, 2], label="z")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("position (m)")
    ax1.set_title("Base position vs time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(pos[:, 0], pos[:, 1], lw=0.8)
    ax2.scatter(pos[0, 0], pos[0, 1], c="g", s=40, zorder=5, label="start")
    ax2.scatter(pos[-1, 0], pos[-1, 1], c="r", s=40, zorder=5, label="end")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Base trajectory (top view, XY)")
    ax2.axis("equal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=0.8)
    ax3.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c="g", s=30)
    ax3.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c="r", s=30)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title("Base trajectory (3D)")

    ax4 = fig.add_subplot(2, 2, 4)
    if has_quat:
        rpy = euler_from_quat_xyzw(quat.astype(np.float64))
        yaw_deg = np.degrees(rpy[:, 2])
        ax4.plot(t, yaw_deg, color="C0")
        ax4.set_xlabel("time (s)")
        ax4.set_ylabel("yaw (deg)")
        ax4.set_title("Base yaw (from base_pose quaternion)")
    else:
        ax4.plot(t[1:], step_dist, color="C1")
        ax4.set_xlabel("time (s)")
        ax4.set_ylabel("|Δp| (m)")
        ax4.set_title("Per-frame base displacement")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(os.path.basename(path))
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure: {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
