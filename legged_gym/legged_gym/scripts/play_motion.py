#!/usr/bin/env python3
"""Kinematically replay a MotionLib-style .pt file in IsaacGym.

Example:
python legged_gym/legged_gym/scripts/play_motion.py \
    --task k1_move_amp \
    --motion legged_gym/resources/datasets/goalkeeper_from_pkl_k1/goalkeeper_from_pkl_k1.pt

If the robot looks folded / missing limbs, try: --base-quat-order wxyz
Foot snap (--foot-snap-margin, default 0) adjusts root height so the lowest
foot-link world z matches the target (0 = on ground plane). It moves both
down and up; use a small negative value (e.g. -0.02) if soles still look
floating because the link frame sits above the mesh sole.
Tune overall height with --z-offset.

World-axis mismatches (lateral looks reversed vs base_position): try
--flip-base-y, --flip-base-x, or --horiz-flip-180 (negate xy + yaw +180°).
With --flip-base-y, try --swap-lr-dofs so left/right legs match the mirror.

IsaacGym/PhysX usually needs one simulate() step for articulated FK to reach
rigid_body_states; without it, foot heights stay stale (wrong snap / wrong draw).
Use --keep-gravity to keep URDF gravity on during replay.
"""

from __future__ import annotations

import os
import sys
import time

import isaacgym  # noqa: F401
import torch
from isaacgym import gymapi, gymtorch, gymutil

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry


DEFAULT_MOTION = (
    "legged_gym/resources/datasets/goalkeeper_from_pkl_k1/"
    "goalkeeper_from_pkl_k1.pt"
)


def parse_args():

    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "k1_move_amp",
        },
        {
            "name": "--motion",
            "type": str,
            "default": DEFAULT_MOTION,
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 1,
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
        },
        {
            "name": "--seed",
            "type": int,
            "default": 1,
        },
        {
            "name": "--start",
            "type": int,
            "default": 0,
        },
        {
            "name": "--end",
            "type": int,
            "default": -1,
        },
        {
            "name": "--stride",
            "type": int,
            "default": 1,
        },
        {
            "name": "--speed",
            "type": float,
            "default": 1.0,
        },
        {
            "name": "--loop",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--no-recenter",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--z-offset",
            "type": float,
            "default": 0.0,
        },
        {
            "name": "--simulate",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--spawn-z",
            "type": str,
            "default": "from_motion",
        },
        {
            "name": "--base-quat-order",
            "type": str,
            "default": "xyzw",
        },
        {
            "name": "--no-foot-snap",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--foot-snap-margin",
            "type": float,
            "default": 0.0,
        },
        {
            "name": "--flip-visuals",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--keep-gravity",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--flip-base-x",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--flip-base-y",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--horiz-flip-180",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--swap-lr-dofs",
            "action": "store_true",
            "default": False,
        },
    ]

    args = gymutil.parse_arguments(
        description="Replay motion .pt in IsaacGym",
        custom_parameters=custom_parameters,
    )

    # ===== legged_gym compatibility =====

    args.sim_device = args.rl_device

    if not hasattr(args, "physics_engine"):
        args.physics_engine = gymapi.SIM_PHYSX

    if not hasattr(args, "use_gpu"):
        args.use_gpu = True

    if not hasattr(args, "use_gpu_pipeline"):
        args.use_gpu_pipeline = True

    if not hasattr(args, "sim_device_type"):
        args.sim_device_type = "cuda"

    if not hasattr(args, "compute_device_id"):
        args.compute_device_id = 0

    if not hasattr(args, "graphics_device_id"):
        args.graphics_device_id = 0

    return args


def load_motion(path: str):

    path = os.path.abspath(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict motion file, got {type(data)}")

    return data


def as_time_tensor(data, key, dims):

    if key not in data:
        raise KeyError(f"Missing key: {key}")

    value = data[key]

    tensor = (
        value.detach().cpu().float()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value).float()
    )

    if tensor.ndim == dims + 1 and tensor.shape[0] == 1:
        tensor = tensor[0]

    if tensor.ndim != dims:
        raise ValueError(
            f"{key} must have {dims} dims, got {tuple(tensor.shape)}"
        )

    return tensor


def normalize_quat_xyzw(quat):
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_conjugate_xyzw(quat):
    out = quat.clone()
    out[..., :3] *= -1.0
    return out


def quat_mul_xyzw(a, b):

    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)

    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def finite_difference(values, fps):

    vel = torch.zeros_like(values)

    if values.shape[0] > 1:
        vel[:-1] = (values[1:] - values[:-1]) * fps
        vel[-1] = vel[-2]

    return vel


def estimate_angular_velocity_xyzw(quat, fps):

    ang_vel = torch.zeros(quat.shape[0], 3, dtype=quat.dtype)

    if quat.shape[0] <= 1:
        return ang_vel

    delta = quat_mul_xyzw(
        quat[1:],
        quat_conjugate_xyzw(quat[:-1]),
    )

    delta = normalize_quat_xyzw(delta)

    delta[delta[:, 3] < 0.0] *= -1.0

    angle = 2.0 * torch.atan2(
        delta[:, :3].norm(dim=-1),
        delta[:, 3].clamp(-1.0, 1.0),
    )

    axis = delta[:, :3] / delta[:, :3].norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    ang_vel[:-1] = axis * angle.unsqueeze(-1) * fps
    ang_vel[-1] = ang_vel[-2]

    return ang_vel


def build_motion_tensors(data, num_dof, device, base_quat_order="xyzw"):

    root_pos = as_time_tensor(data, "base_position", 2)

    root_quat_raw = as_time_tensor(data, "base_pose", 2)
    if base_quat_order == "xyzw":
        pass
    elif base_quat_order == "wxyz":
        root_quat_raw = torch.stack(
            (
                root_quat_raw[..., 1],
                root_quat_raw[..., 2],
                root_quat_raw[..., 3],
                root_quat_raw[..., 0],
            ),
            dim=-1,
        )
    else:
        raise ValueError(
            f"Unknown base_quat_order {base_quat_order!r}; use xyzw or wxyz"
        )

    root_quat = normalize_quat_xyzw(root_quat_raw)

    joint_pos = as_time_tensor(data, "joint_position", 2)
    if joint_pos.shape[-1] != num_dof:
        raise ValueError(
            f"Motion joint_position has {joint_pos.shape[-1]} DOFs, "
            f"but the loaded asset has {num_dof} DOFs"
        )

    fps = float(data.get("fps", 30.0))

    if "joint_velocity" in data:
        joint_vel = as_time_tensor(data, "joint_velocity", 2)
    else:
        joint_vel = finite_difference(joint_pos, fps)

    root_lin_vel = finite_difference(root_pos, fps)

    root_ang_vel = estimate_angular_velocity_xyzw(
        root_quat,
        fps,
    )

    T = min(
        root_pos.shape[0],
        root_quat.shape[0],
        joint_pos.shape[0],
        joint_vel.shape[0],
    )

    return {
        "root_pos": root_pos[:T].to(device),
        "root_quat": root_quat[:T].to(device),
        "root_lin_vel": root_lin_vel[:T].to(device),
        "root_ang_vel": root_ang_vel[:T].to(device),
        "joint_pos": joint_pos[:T].to(device),
        "joint_vel": joint_vel[:T].to(device),
        "fps": fps,
        "num_frames": T,
    }


def apply_world_motion_corrections(
    motion,
    fps,
    *,
    horiz_flip_180,
    flip_base_x,
    flip_base_y,
):
    rp = motion["root_pos"]
    rq = motion["root_quat"]

    if horiz_flip_180:
        rp = rp.clone()
        rp[:, :2] *= -1.0
        motion["root_pos"] = rp
        t = rp.shape[0]
        q180 = rp.new_tensor([0.0, 0.0, 1.0, 0.0]).expand(t, 4)
        motion["root_quat"] = normalize_quat_xyzw(quat_mul_xyzw(q180, rq))
    elif flip_base_x or flip_base_y:
        rp = rp.clone()
        if flip_base_x:
            rp[:, 0] *= -1.0
        if flip_base_y:
            rp[:, 1] *= -1.0
        motion["root_pos"] = rp

    motion["root_lin_vel"] = finite_difference(motion["root_pos"], fps)
    motion["root_ang_vel"] = estimate_angular_velocity_xyzw(
        motion["root_quat"],
        fps,
    )


def swap_lr_joint_channels_k1_22(joint_pos, joint_vel):
    """Swap L/R arm (4+4) and L/R leg (6+6) channels for K1 22-DOF order."""
    jp = joint_pos.clone()
    jv = joint_vel.clone()
    la, ra = slice(2, 6), slice(6, 10)
    ll, rl = slice(10, 16), slice(16, 22)
    tmp_p, tmp_v = jp[:, la].clone(), jv[:, la].clone()
    jp[:, la], jp[:, ra] = jp[:, ra].clone(), tmp_p
    jv[:, la], jv[:, ra] = jv[:, ra].clone(), tmp_v
    tmp_p, tmp_v = jp[:, ll].clone(), jv[:, ll].clone()
    jp[:, ll], jp[:, rl] = jp[:, rl].clone(), tmp_p
    jv[:, ll], jv[:, rl] = jv[:, rl].clone(), tmp_v
    return jp, jv


def prepare_motion_placement(motion, env, start, recenter, z_offset, spawn_z):
    root_offset = torch.zeros(3, dtype=torch.float, device=env.device)

    if recenter:
        root_offset[:2] = -motion["root_pos"][start, :2]

    if spawn_z == "match_cfg":
        root_offset[2] = (
            float(env.cfg.init_state.pos[2])
            - float(motion["root_pos"][start, 2])
            + float(z_offset)
        )
    elif spawn_z == "from_motion":
        root_offset[2] = float(z_offset)
    else:
        raise ValueError(
            f"Unknown --spawn-z {spawn_z!r}; use from_motion or match_cfg"
        )

    motion["root_offset"] = root_offset


def snap_feet_to_ground_plane(env, motion, frame, target_min_foot_z, max_iters=24):
    """Shift root z so min(foot link world z) -> target_min_foot_z (bidirectional)."""
    if not hasattr(env, "contact_feet_indices"):
        return
    if env.contact_feet_indices.numel() == 0:
        return

    eps = 2e-4
    target = float(target_min_foot_z)

    for _ in range(max_iters):
        apply_frame(env, motion, frame, physics_sync=True)
        if str(env.device).startswith("cuda"):
            torch.cuda.synchronize()

        feet_z = env.rigid_body_states[0, env.contact_feet_indices, 2]
        min_z = float(feet_z.min().item())
        err = min_z - target
        if abs(err) < eps:
            return

        motion["root_offset"][2] = motion["root_offset"][2] - err


def configure_env(args):

    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.play = True

    env_cfg.noise.add_noise = False

    if hasattr(env_cfg.env, "use_ball_actor"):
        env_cfg.env.use_ball_actor = False

    if hasattr(env_cfg.domain_rand, "push_robots"):
        env_cfg.domain_rand.push_robots = False

    if getattr(args, "flip_visuals", False):
        env_cfg.asset.flip_visual_attachments = True

    if not getattr(args, "keep_gravity", False) and hasattr(
        env_cfg.asset,
        "disable_gravity",
    ):
        env_cfg.asset.disable_gravity = True

    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    return env


def set_viewer_camera(env):

    if not env.viewer:
        return

    cam_pos = gymapi.Vec3(2.5, -3.0, 1.6)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.8)

    env.gym.viewer_camera_look_at(
        env.viewer,
        None,
        cam_pos,
        cam_target,
    )


def draw_frame(env):

    if not env.viewer:
        return

    if env.gym.query_viewer_has_closed(env.viewer):
        sys.exit(0)

    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)


def apply_frame(env, motion, frame, physics_sync=True):

    joint_pos = motion["joint_pos"][frame]
    joint_vel = motion["joint_vel"][frame]

    root_pos = motion["root_pos"][frame].unsqueeze(0).clone()
    root_pos += motion["root_offset"].unsqueeze(0)
    root_pos += env.env_origins

    env.root_states[:, :3] = root_pos

    env.root_states[:, 3:7] = motion["root_quat"][frame].clone()

    env.root_states[:, 7:10] = motion["root_lin_vel"][frame].clone()

    env.root_states[:, 10:13] = motion["root_ang_vel"][frame].clone()

    env.dof_state[:, 0] = joint_pos
    env.dof_state[:, 1] = joint_vel

    env.dof_pos[:] = joint_pos
    env.dof_vel[:] = joint_vel

    env.gym.set_actor_root_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(
            env.root_states.view(-1, 13)
        ),
    )

    env.gym.set_dof_state_tensor(
        env.sim,
        gymtorch.unwrap_tensor(
            env.dof_state.view(-1, 2)
        ),
    )

    env.gym.refresh_actor_root_state_tensor(env.sim)

    env.gym.refresh_dof_state_tensor(env.sim)

    env.gym.refresh_rigid_body_state_tensor(env.sim)

    if physics_sync:
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)


def main():

    args = parse_args()

    env = configure_env(args)

    set_viewer_camera(env)

    data = load_motion(args.motion)

    motion = build_motion_tensors(
        data,
        env.num_dof,
        env.device,
        base_quat_order=args.base_quat_order,
    )

    apply_world_motion_corrections(
        motion,
        motion["fps"],
        horiz_flip_180=args.horiz_flip_180,
        flip_base_x=args.flip_base_x,
        flip_base_y=args.flip_base_y,
    )

    if args.swap_lr_dofs:
        if env.num_dof != 22:
            raise ValueError(
                f"--swap-lr-dofs is for K1 22-DOF ordering only; got {env.num_dof}"
            )
        jp, jv = swap_lr_joint_channels_k1_22(
            motion["joint_pos"],
            motion["joint_vel"],
        )
        motion["joint_pos"] = jp
        motion["joint_vel"] = jv

    start = max(0, args.start)

    end = (
        motion["num_frames"]
        if args.end < 0
        else min(args.end, motion["num_frames"])
    )

    stride = max(1, args.stride)

    frame_ids = list(range(start, end, stride))
    prepare_motion_placement(
        motion,
        env,
        start,
        recenter=not args.no_recenter,
        z_offset=args.z_offset,
        spawn_z=args.spawn_z,
    )

    if not args.no_foot_snap:
        snap_feet_to_ground_plane(
            env,
            motion,
            start,
            args.foot_snap_margin,
        )

    frame_dt = stride / motion["fps"]

    print(f"Loaded motion: {args.motion}")
    print(f"Frames: {motion['num_frames']}")
    print(f"spawn_z={args.spawn_z} base_quat_order={args.base_quat_order}")
    if (
        args.horiz_flip_180
        or args.flip_base_x
        or args.flip_base_y
        or args.swap_lr_dofs
    ):
        print(
            "world_corr="
            f"horiz180={args.horiz_flip_180} flip_x={args.flip_base_x} "
            f"flip_y={args.flip_base_y} swap_lr={args.swap_lr_dofs}"
        )
    print(f"Root offset applied: {motion['root_offset'].detach().cpu().tolist()}")

    apply_frame(env, motion, start, physics_sync=not args.simulate)
    if hasattr(env, "contact_feet_indices") and env.contact_feet_indices.numel() > 0:
        fz = env.rigid_body_states[0, env.contact_feet_indices, 2].detach().cpu().tolist()
        print(f"Foot link z (env 0, after setup): {fz}")

    while True:

        for frame in frame_ids:

            t0 = time.time()

            apply_frame(env, motion, frame, physics_sync=not args.simulate)

            if args.simulate:
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)

            draw_frame(env)

            sleep_t = frame_dt - (time.time() - t0)

            if sleep_t > 0:
                time.sleep(sleep_t)

        if not args.loop:
            break


if __name__ == "__main__":
    main()