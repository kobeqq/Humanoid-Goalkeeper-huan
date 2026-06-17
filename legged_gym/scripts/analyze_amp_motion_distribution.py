#!/usr/bin/env python3
"""Report directional distribution for AMP motion .pt files."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch


def load_motion(path: Path) -> dict:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a dict, got {type(data)}")
    return data


def as_tensor(data: dict, key: str) -> torch.Tensor:
    if key not in data:
        raise KeyError(f"Missing {key}")
    value = data[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float)


def yaw_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def finite_difference(values: torch.Tensor, fps: float) -> torch.Tensor:
    vel = torch.zeros_like(values)
    if values.shape[0] > 1:
        vel[:-1] = (values[1:] - values[:-1]) * fps
        vel[-1] = vel[-2]
    return vel


def body_xy_velocity(pos: torch.Tensor, yaw: torch.Tensor, fps: float) -> torch.Tensor:
    world_vel = finite_difference(pos[:, :2], fps)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    body_x = cos_yaw * world_vel[:, 0] + sin_yaw * world_vel[:, 1]
    body_y = -sin_yaw * world_vel[:, 0] + cos_yaw * world_vel[:, 1]
    return torch.stack((body_x, body_y), dim=-1)


def yaw_rate(yaw: torch.Tensor, fps: float) -> torch.Tensor:
    unwrapped = torch.from_numpy(np.unwrap(yaw.numpy())).float()
    return finite_difference(unwrapped.unsqueeze(-1), fps).squeeze(-1)


def classify_motion(body_vel: torch.Tensor, wz: torch.Tensor, speed_threshold: float, yaw_rate_threshold: float, axis_half_angle_deg: float) -> dict:
    speed = torch.norm(body_vel, dim=-1)
    moving = speed >= speed_threshold
    turning_any = torch.abs(wz) >= yaw_rate_threshold
    stationary = ~moving & ~turning_any
    turning_primary = ~moving & turning_any

    angle = torch.atan2(body_vel[:, 1], body_vel[:, 0])
    half = math.radians(axis_half_angle_deg)
    abs_angle = torch.abs(angle)
    forward = moving & (abs_angle <= half)
    backward = moving & ((math.pi - abs_angle) <= half)
    left = moving & (torch.abs(angle - math.pi / 2.0) <= half)
    right = moving & (torch.abs(angle + math.pi / 2.0) <= half)
    diagonal = moving & ~(forward | backward | left | right)

    return {
        "forward": forward,
        "backward": backward,
        "left": left,
        "right": right,
        "diagonal": diagonal,
        "turning_primary": turning_primary,
        "turning_any": turning_any,
        "stationary": stationary,
    }


def pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * count / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="AMP dataset folder containing .pt motion files")
    parser.add_argument("--speed-threshold", type=float, default=0.05, help="m/s below which XY motion is stationary")
    parser.add_argument("--yaw-rate-threshold", type=float, default=0.3, help="rad/s for turning frame detection")
    parser.add_argument("--axis-half-angle-deg", type=float, default=22.5, help="forward/back/left/right cone half angle")
    args = parser.parse_args()

    files = sorted(path for path in args.folder.iterdir() if path.suffix == ".pt")
    if not files:
        raise FileNotFoundError(f"No .pt files found in {args.folder}")

    totals = {
        "forward": 0,
        "backward": 0,
        "left": 0,
        "right": 0,
        "diagonal": 0,
        "turning_primary": 0,
        "turning_any": 0,
        "stationary": 0,
    }
    total_frames = 0
    clip_rows = []

    for path in files:
        data = load_motion(path)
        pos = as_tensor(data, "base_position")
        quat = as_tensor(data, "base_pose")
        fps = float(data.get("fps", 30.0))
        yaw = yaw_from_quat_xyzw(quat)
        body_vel = body_xy_velocity(pos, yaw, fps)
        wz = yaw_rate(yaw, fps)
        classes = classify_motion(
            body_vel,
            wz,
            args.speed_threshold,
            args.yaw_rate_threshold,
            args.axis_half_angle_deg,
        )
        frames = int(pos.shape[0])
        total_frames += frames
        row = {"clip": path.name, "frames": frames}
        for key, mask in classes.items():
            count = int(mask.sum().item())
            totals[key] += count
            row[key] = count
        clip_rows.append(row)

    print("Motion Distribution Report")
    print(f"Dataset folder: {os.fspath(args.folder)}")
    print(f"Clips: {len(files)}")
    print(f"Frames: {total_frames}")
    print(f"Speed threshold: {args.speed_threshold:.3f} m/s")
    print(f"Yaw-rate threshold: {args.yaw_rate_threshold:.3f} rad/s")
    print("")
    print(f"Forward Motion %: {pct(totals['forward'], total_frames):.2f}")
    print(f"Backward Motion %: {pct(totals['backward'], total_frames):.2f}")
    print(f"Left Motion %: {pct(totals['left'], total_frames):.2f}")
    print(f"Right Motion %: {pct(totals['right'], total_frames):.2f}")
    print(f"Diagonal Motion %: {pct(totals['diagonal'], total_frames):.2f}")
    print(f"Turning Motion %: {pct(totals['turning_any'], total_frames):.2f}")
    print(f"Stationary/Other %: {pct(totals['stationary'], total_frames):.2f}")
    print("")
    print("Per-clip primary direction counts:")
    for row in clip_rows:
        print(
            f"- {row['clip']}: frames={row['frames']} "
            f"forward={row['forward']} backward={row['backward']} "
            f"left={row['left']} right={row['right']} diagonal={row['diagonal']} "
            f"turning_primary={row['turning_primary']} turning_any={row['turning_any']} "
            f"stationary={row['stationary']}"
        )


if __name__ == "__main__":
    main()
