#!/usr/bin/env python3
"""
Convert G1 goalkeeper .pt motion files (21 joint columns) to K1 layout (22 joints).

This is **heuristic retargeting** (same semantic joints copied, head/wrist split guessed).
It is NOT a full IK / body-scale retarget; quality may be poor for head and arms.

G1 column order: resources/datasets/goalkeeper/joint_id.txt (21 joints)
K1 column order: resources/datasets/goalkeeper_k1/joint_id.txt (22 joints)

Usage:
  python3 convert_g1_motion_pt_to_k1.py \\
    --src_dir ../resources/datasets/goalkeeper \\
    --dst_dir ../resources/datasets/goalkeeper_k1

  # Optional: overwrite
  python3 convert_g1_motion_pt_to_k1.py --src_dir ... --dst_dir ... --force
"""

from __future__ import annotations

import argparse
import os
import torch

# G1 joint_id.txt indices (0..20)
G1 = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "left_shoulder_pitch_joint": 13,
    "left_shoulder_roll_joint": 14,
    "left_shoulder_yaw_joint": 15,
    "left_elbow_joint": 16,
    "right_shoulder_pitch_joint": 17,
    "right_shoulder_roll_joint": 18,
    "right_shoulder_yaw_joint": 19,
    "right_elbow_joint": 20,
}

# K1 joint order (0..21) — must match goalkeeper_k1/joint_id.txt
K1_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def g1_to_k1_tensor(j21: torch.Tensor) -> torch.Tensor:
    """
    j21: shape (T, 21) or (21,) for joint angles or velocities.
    Returns: (T, 22) or (22,)
    """
    single = j21.dim() == 1
    if single:
        j21 = j21.unsqueeze(0)
    g = j21
    T = g.shape[0]
    k = torch.zeros(T, 22, dtype=g.dtype, device=g.device)

    def col(name: str) -> torch.Tensor:
        return g[:, G1[name]]

    # Legs: direct
    k[:, 10] = col("left_hip_pitch_joint")
    k[:, 11] = col("left_hip_roll_joint")
    k[:, 12] = col("left_hip_yaw_joint")
    k[:, 13] = col("left_knee_joint")
    k[:, 14] = col("left_ankle_pitch_joint")
    k[:, 15] = col("left_ankle_roll_joint")
    k[:, 16] = col("right_hip_pitch_joint")
    k[:, 17] = col("right_hip_roll_joint")
    k[:, 18] = col("right_hip_yaw_joint")
    k[:, 19] = col("right_knee_joint")
    k[:, 20] = col("right_ankle_pitch_joint")
    k[:, 21] = col("right_ankle_roll_joint")

    # Head: G1 has no separate head; map waist yaw to head yaw (tunable)
    k[:, 0] = col("waist_yaw_joint")
    k[:, 1] = 0.0

    # Arms: G1 has shoulder yaw + single elbow; K1 has elbow pitch + yaw
    k[:, 2] = col("left_shoulder_pitch_joint")
    k[:, 3] = col("left_shoulder_roll_joint")
    k[:, 4] = col("left_elbow_joint")
    k[:, 5] = col("left_shoulder_yaw_joint")

    k[:, 6] = col("right_shoulder_pitch_joint")
    k[:, 7] = col("right_shoulder_roll_joint")
    k[:, 8] = col("right_elbow_joint")
    k[:, 9] = col("right_shoulder_yaw_joint")

    if single:
        return k.squeeze(0)
    return k


def convert_one(src_path: str, dst_path: str, force: bool) -> None:
    if os.path.exists(dst_path) and not force:
        print(f"skip (exists): {dst_path}")
        return
    data = torch.load(src_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {src_path}, got {type(data)}")

    out = dict(data)
    for key in ("joint_position", "joint_velocity"):
        if key not in out:
            raise KeyError(f"{src_path} missing {key}")
        out[key] = g1_to_k1_tensor(out[key])

    # K1 motion lib uses no keyframe bodies when keyframe_name matches nothing;
    # keep link_* from G1 only if shapes still match AMP — they are 17 keyframes.
    # MotionLib does not read link_* when len(keyframe_names)==0 for K1.
    # Drop bulky / unused keys to avoid confusion (optional).
    # Uncomment to strip:
    # for k in ("link_position", "link_orientation", "link_velocity", "link_angular_velocity"):
    #     out.pop(k, None)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    torch.save(out, dst_path)
    print(f"written {dst_path}  joint shape {tuple(out['joint_position'].shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "resources", "datasets", "goalkeeper"
        ),
        help="Directory containing G1 *.pt files",
    )
    ap.add_argument(
        "--dst_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "resources", "datasets", "goalkeeper_k1"
        ),
        help="Output directory for K1 *.pt files",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    dst_dir = os.path.abspath(args.dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    names = [
        "lefthand.pt",
        "righthand.pt",
        "leftjump.pt",
        "rightjump.pt",
        "leftstep.pt",
        "rightstep.pt",
    ]
    for n in names:
        p = os.path.join(src_dir, n)
        if not os.path.isfile(p):
            print(f"warning: missing {p}")
            continue
        convert_one(p, os.path.join(dst_dir, n), args.force)

if __name__ == "__main__":
    main()
