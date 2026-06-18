#!/usr/bin/env python3
"""将 K1 motion 数据按动作类型拆分为多个 .pt 文件，供 AMP / 2D locomotion 使用。

支持两种输入：
1. 单个长轨迹 .pt（如 goalkeeper_from_pkl_k1.pt）：按 base 运动自动切段；
2. 目录下多个 clip .pt：按 clip 统计量分类。

每个输出类别保存为一个 .pt，内容是 clip dict 的 list，与 load_imitation_dataset 兼容。

用法：
python legged_gym/scripts/split_k1_motion_dataset.py \\
  --src legged_gym/resources/datasets/goalkeeper_from_pkl_k1/goalkeeper_from_pkl_k1.pt \\
  --dst legged_gym/resources/datasets/goalkeeper_from_pkl_k1_locomotion
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

CLIP_KEYS = ("base_position", "base_pose", "joint_position", "joint_velocity", "fps")
ROOT_POS_KEYS = ("base_position", "root_position", "pelvis_position")


def _yaw_from_quat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _smooth_1d(values, window=15):
    if len(values) == 0:
        return values
    pad = np.concatenate([[values[0]], values])
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.convolve(pad, kernel, mode="same")
    return out[: len(values)]


def _contiguous_segments(mask, min_len):
    segments = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask)))
    return segments


def _extract_clip(traj, start, end):
    """从长轨迹中截取 [start, end) 帧，保留 MotionLib 所需字段。"""
    clip = {}
    for key in CLIP_KEYS:
        if key not in traj:
            continue
        value = traj[key]
        if key == "fps":
            clip[key] = value
            continue
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        clip[key] = tensor[start:end].clone()
    return clip


def _clip_summary(name, clip, fps):
    bp = clip["base_position"]
    if torch.is_tensor(bp):
        bp = bp.numpy()
    jp = clip["joint_position"]
    if torch.is_tensor(jp):
        jp = jp.numpy()
    quat = clip["base_pose"]
    if torch.is_tensor(quat):
        quat = quat.numpy()

    num_frames = int(bp.shape[0])
    duration = num_frames / fps
    disp = bp[-1, :2] - bp[0, :2]
    speed = float(np.linalg.norm(disp))
    direction = float(math.degrees(math.atan2(disp[1], disp[0])))
    yaw = _yaw_from_quat(quat)
    yaw_range = float(np.degrees(np.abs(yaw[-1] - yaw[0])))
    min_height = float(bp[:, 2].min()) if bp.shape[1] >= 3 else None

    dt = 1.0 / fps
    vel = np.diff(bp[:, :2], axis=0) / dt
    mean_velocity = float(np.linalg.norm(vel, axis=1).mean()) if len(vel) else 0.0

    return {
        "name": name,
        "frames": num_frames,
        "duration": duration,
        "mean_velocity": mean_velocity,
        "speed": speed,
        "direction_angle": direction,
        "yaw_range": yaw_range,
        "min_height": min_height,
    }


def classify_motion_segment(start, end, base_pos, vel, height, yaw):
    """根据 base xy 速度/位移分类单个运动片段。"""
    disp = base_pos[end, :2] - base_pos[start, :2]
    dist = float(np.linalg.norm(disp))
    direction = math.degrees(math.atan2(disp[1], disp[0]))
    seg_vel = vel[start:end].mean(axis=0)
    seg_speed = float(np.linalg.norm(seg_vel))
    vel_dir = math.degrees(math.atan2(seg_vel[1], seg_vel[0]))
    h_min = float(height[start : end + 1].min())
    yaw_delta = float(np.degrees(np.abs(yaw[end] - yaw[start])))

    if h_min < 0.42 and (seg_speed > 0.18 or dist > 0.12):
        return "dive_jump"

    if seg_speed < 0.12 or dist < 0.08:
        return "standing"

    abs_fwd = abs(float(seg_vel[0]))
    abs_lat = abs(float(seg_vel[1]))

    # 横向速度主导：leftstep / rightstep
    if abs_lat > abs_fwd * 0.85 and abs_lat > 0.07:
        return "leftstep" if seg_vel[1] > 0.0 else "rightstep"

    # 守门员数据里常见斜向逼近步（约 25°~75°），单独归为 diagonal
    for angle in (direction, vel_dir):
        if 25.0 <= angle <= 75.0 or -115.0 <= angle <= -65.0:
            return "diagonal"
        if -75.0 <= angle <= -25.0 or 65.0 <= angle <= 115.0:
            return "diagonal"

    if abs(direction) < 25.0 or abs(vel_dir) < 25.0:
        return "forward"
    if abs(abs(direction) - 180.0) < 35.0 or abs(abs(vel_dir) - 180.0) < 35.0:
        return "backward"
    if yaw_delta > 40.0:
        return "turn_or_yaw_heavy"
    return "diagonal"


def segment_long_trajectory(traj, fps=30.0, min_move_frames=18, min_stand_frames=30):
    """把单条长轨迹切成 standing / locomotion 片段。"""
    base_pos = traj["base_position"]
    if torch.is_tensor(base_pos):
        base_pos = base_pos.numpy()
    quat = traj["base_pose"]
    if torch.is_tensor(quat):
        quat = quat.numpy()

    fps = float(traj.get("fps", fps))
    dt = 1.0 / fps
    vel = np.diff(base_pos[:, :2], axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    speed_smooth = _smooth_1d(speed)
    height = base_pos[:, 2]
    yaw = _yaw_from_quat(quat)

    moving = speed_smooth > 0.14
    move_segments = _contiguous_segments(moving, min_move_frames)
    stand_segments = _contiguous_segments(~moving, min_stand_frames)

    labeled = []
    for start, end in move_segments:
        label = classify_motion_segment(start, end, base_pos, vel, height, yaw)
        labeled.append((start, end, label))

    for start, end in stand_segments:
        # 避免与 moving 片段重叠（moving 优先）
        overlap = any(not (end <= ms or start >= me) for ms, me, _ in labeled)
        if not overlap:
            labeled.append((start, end, "standing"))

    labeled.sort(key=lambda x: x[0])
    return labeled, fps


def split_long_trajectory_file(src_path):
    data = torch.load(src_path, map_location="cpu")
    if not isinstance(data, dict) or "joint_position" not in data:
        raise ValueError(f"{src_path} 不是包含 joint_position 的单条轨迹 dict。")
    segments, fps = segment_long_trajectory(data)
    buckets = {}
    summaries = []
    for idx, (start, end, label) in enumerate(segments):
        clip = _extract_clip(data, start, end)
        clip_name = f"{src_path.stem}_{label}_{idx:03d}"
        buckets.setdefault(label, []).append(clip)
        item = _clip_summary(clip_name, clip, fps)
        item["label"] = label
        item["start"] = start
        item["end"] = end
        summaries.append(item)
    return buckets, summaries, fps


def _iter_source_files(src: Path):
    if src.is_file() and src.suffix == ".pt":
        yield src
        return
    for path in sorted(src.glob("*.pt")):
        if path.is_file():
            yield path


def main():
    parser = argparse.ArgumentParser(description="Split K1 motion data into locomotion buckets.")
    parser.add_argument("--src", required=True, type=Path, help="Source .pt file or directory.")
    parser.add_argument("--dst", required=True, type=Path, help="Output directory.")
    parser.add_argument("--joint-mapping", type=Path, default=None, help="Optional joint_id_k1.txt to copy.")
    parser.add_argument("--dry-run", action="store_true", help="Only print summary.")
    args = parser.parse_args()

    all_buckets = {}
    all_summaries = []
    for src_path in _iter_source_files(args.src):
        data = torch.load(src_path, map_location="cpu")
        if isinstance(data, dict) and "joint_position" in data and len(data["joint_position"].shape) == 2:
            buckets, summaries, _ = split_long_trajectory_file(src_path)
        else:
            raise ValueError(f"暂不支持的文件结构: {src_path}")

        for label, clips in buckets.items():
            all_buckets.setdefault(label, []).extend(clips)
        all_summaries.extend(summaries)

    counts = {label: len(clips) for label, clips in all_buckets.items()}
    print("=== 拆分统计 ===")
    print(json.dumps(counts, indent=2, ensure_ascii=False))

    if args.dry_run:
        print(json.dumps(all_summaries[:20], indent=2, ensure_ascii=False))
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    for label, clips in sorted(all_buckets.items()):
        out_path = args.dst / f"{label}.pt"
        torch.save(clips, out_path)
        print(f"写入 {out_path} ({len(clips)} clips)")

    mapping_src = args.joint_mapping
    if mapping_src is None:
        candidate = args.src.parent / "joint_id_k1.txt"
        if candidate.is_file():
            mapping_src = candidate
    if mapping_src is not None and mapping_src.is_file():
        shutil.copy2(mapping_src, args.dst / mapping_src.name)

    summary = {
        "source": str(args.src),
        "destination": str(args.dst),
        "counts": counts,
        "clips": all_summaries,
    }
    summary_path = args.dst / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
