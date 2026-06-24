import os
import math
import torch
import random
import pickle
from tqdm import tqdm
from legged_gym.utils.math import (
    euler_xyz_to_quat,
    quat_apply_yaw,
    quat_apply_yaw_inverse,
    quat_mul, quat_conjugate,
    quat_mul_yaw_inverse,
    quat_mul_yaw,
    quat_mul,
    quat_apply,
    quat_rotate_inverse,
)
# from isaacgym.torch_utils import quat_apply, normalize
import copy
import torch


def build_lower_body_amp_step_obs(q_leg, dq_leg, base_lin_vel, base_ang_vel, projected_gravity):
    """Build one-step lower-body AMP obs.

    Shape:
        q_leg:             [N, 12]
        dq_leg:            [N, 12]
        base_ang_vel:      [N, 3]
        projected_gravity: [N, 3]

    Output:
        [N, 30] = 12 q + 12 dq + 3 base angular velocity + 3 projected gravity

    base_lin_vel is intentionally accepted for API compatibility but not used
    in this first version.
    """
    del base_lin_vel
    return torch.cat((q_leg, dq_leg, base_ang_vel, projected_gravity), dim=-1)


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    quat_angle = torch.as_tensor(quat_angle, dtype=torch.float32)
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return torch.cat([roll_x.view(-1, 1), pitch_y.view(-1, 1), yaw_z.view(-1, 1)], dim=1)


def load_imitation_dataset(folder, mapping="joint_id.txt", suffix=".pt"):
    # List all files with the given suffix (e.g., .pt)
    filenames = [name for name in os.listdir(folder) if name.endswith(suffix)]
    
    multidataset = {}
    for filename in tqdm(filenames):
        try:
            data = torch.load(os.path.join(folder, filename))
            if isinstance(data, dict):
                dataset_list = [data]
            elif isinstance(data, list):
                dataset_list = [traj for traj in data if isinstance(traj, dict)]
            else:
                raise TypeError(f"Unsupported motion file payload type: {type(data).__name__}")

            random.shuffle(dataset_list)
            multidataset[filename[:-len(suffix)]] = dataset_list

        except Exception as e:
            print(f"{filename} load failed!!! Error: {e}")
            continue


    # Read and process the joint_id mapping
    with open(mapping, "r") as file:
        lines = file.readlines() #读取文件中的每一行

    # Process the joint ID mapping into a dictionary
    lines = [line.strip().split(" ") for line in lines] #将每一行中的空格分割成列表
    joint_id_dict = {k: int(v) for v, k in lines}

    return multidataset, joint_id_dict


class MotionLib:
    def __init__(
        self,
        datasets,
        mapping,
        dof_names,
        keyframe_names,
        fps=30,
        min_dt=0.1,
        device="cuda:0",
        output_device=None,
        amp_obs_type="keyframe",
        num_steps=2,
        include_dof_vel=False,
    ):
        if isinstance(device, str) and device.strip().lower() == "gpu":
            device = "cuda:0"
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        if output_device is not None:
            if not isinstance(output_device, torch.device):
                output_device = torch.device(output_device)
        else:
            output_device = device
        self.storage_device = device
        self.device = output_device
        self.fps = fps
        self.env_fps = 50
        self.num_steps = num_steps
        self.include_dof_vel = include_dof_vel
        datasets = self._normalize_datasets(datasets)
        get_len = lambda x: x["base_position"].shape[0]

        datasets = [data for data in datasets if get_len(data) > max(math.ceil(min_dt * fps), 3)]

        self.motion_len = torch.tensor([get_len(data) for data in datasets], dtype=torch.long, device=self.device)
        self.num_motion, self.tot_len = self.motion_len.shape[0], self.motion_len.sum()
        self.motion_sampling_prob = torch.ones(self.num_motion, dtype=torch.float, device=self.device)

        # import ipdb; ipdb.set_trace()
        self.motion_end_ids = torch.cumsum(self.motion_len, dim=0)
        self.motion_start_ids = torch.nn.functional.pad(self.motion_end_ids, (1, -1), "constant", 0)
        
        sd = self.storage_device
        self.motion_base_rpy = torch.zeros(self.tot_len, 3, dtype=torch.float, device=sd)
        self.motion_base_pos = torch.zeros(self.tot_len, 3, dtype=torch.float, device=sd)
        self.motion_base_lin_vel = torch.zeros(self.tot_len, 3, dtype=torch.float, device=sd)
        self.motion_base_ang_vel = torch.zeros(self.tot_len, 3, dtype=torch.float, device=sd)
        self.motion_dof_pos = torch.zeros(self.tot_len, len(dof_names), dtype=torch.float, device=sd)
        self.motion_dof_vel = torch.zeros(self.tot_len, len(dof_names), dtype=torch.float, device=sd)
        self.motion_keyframe_pos = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=sd)
        self.motion_keyframe_rpy = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=sd)
        self.motion_keyframe_lin_vel = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=sd)
        self.motion_keyframe_ang_vel = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=sd)
        
        self.motion_keyframe_pos_local = torch.zeros(self.tot_len, len(keyframe_names), 3, dtype=torch.float, device=sd)
        self.motion_keyframe_quat_local = torch.zeros(self.tot_len, len(keyframe_names), 4, dtype=torch.float, device=sd)

        for i, traj in enumerate(tqdm(datasets)):
            start, end = self.motion_start_ids[i], self.motion_end_ids[i]

            self.motion_base_pos[start:end] = torch.tensor(traj["base_position"], dtype=torch.float, device=sd)
            #! Note: Quat to RPY, not sure the correctness
            self.motion_base_rpy[start:end] = torch.tensor(euler_from_quaternion(traj["base_pose"]), dtype=torch.float, device=sd)   
            self.motion_base_lin_vel[start:end-1] = (self.motion_base_pos[start+1:end] - self.motion_base_pos[start:end-1]) * self.fps
            self.motion_base_ang_vel[start:end-1] = (self.motion_base_rpy[start+1:end] - self.motion_base_rpy[start:end-1]) * self.fps
            self.motion_base_lin_vel[end-1:end] = self.motion_base_lin_vel[end-2:end-1]
            self.motion_base_ang_vel[end-1:end] = self.motion_base_ang_vel[end-2:end-1]
            
            dof_pos = torch.tensor(traj["joint_position"], dtype=torch.float, device=sd)
            dof_vel = torch.tensor(traj["joint_velocity"], dtype=torch.float, device=sd)
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.motion_dof_pos[start:end, j] = dof_pos[:, mapping[name]]
                    self.motion_dof_vel[start:end, j] = dof_vel[:, mapping[name]]

            for k, name in enumerate(keyframe_names):
                # import ipdb; ipdb.set_trace()
                self.motion_keyframe_pos[start:end, k] = torch.tensor(traj["link_position"][:, k], dtype=torch.float, device=sd)
                self.motion_keyframe_rpy[start:end, k] = torch.tensor(euler_from_quaternion(traj["link_oritentation"][:, k]), dtype=torch.float, device=sd)
                self.motion_keyframe_lin_vel[start:end, k] = torch.tensor(traj["lin_velocity"][:, k], dtype=torch.float, device=sd)
                self.motion_keyframe_ang_vel[start:end, k] = torch.tensor(traj["link_angular_velocity"][:, k], dtype=torch.float, device=sd)
            
            self.motion_keyframe_pos[start:end, :, 0:2] -= self.motion_base_pos[start:start+1, None, 0:2]
            self.motion_base_pos[start:end, 0:2] -= self.motion_base_pos[start:start+1, 0:2].clone()

            self.motion_keyframe_pos[start:end, :, 2] -= -0.2
            self.motion_base_pos[start:end, 2] -= -0.2

            # !note: the yaw maybe inaccurate
            local_rotation = euler_xyz_to_quat(self.motion_base_rpy[start:end])[:, None]
            self.motion_keyframe_pos_local[start:end] = quat_apply_yaw_inverse(local_rotation.clone(), self.motion_keyframe_pos[start:end] - self.motion_base_pos[start:end][:, None]) 
            # import ipdb; ipdb.set_trace()
            self.motion_keyframe_quat_local[start:end] = quat_mul_yaw_inverse(local_rotation.clone(), euler_xyz_to_quat(self.motion_keyframe_rpy[start:end]))

        self.amp_obs_type = amp_obs_type

    @staticmethod
    def _normalize_datasets(datasets):
        normalized = []
        for data in datasets:
            if isinstance(data, dict):
                normalized.append(data)
            elif isinstance(data, list):
                normalized.extend([traj for traj in data if isinstance(traj, dict)])
            else:
                raise TypeError(f"Unsupported motion entry type: {type(data).__name__}")
        return normalized


    @staticmethod    
    def calc_blend(motion, time0, time1, w0, w1):
        motion0, motion1 = motion[time0], motion[time1]
        new_w0 = w0.reshape(w0.shape + (1,) * (motion0.dim() - w0.dim()))
        new_w1 = w1.reshape(w1.shape + (1,) * (motion1.dim() - w1.dim()))
        return new_w0 * motion0 + new_w1 * motion1
    

    def get_expert_obs(self, batch_size):
        motion_ids = torch.randint(0, self.num_motion, (batch_size,), device=self.device)
        start_ids = self.motion_start_ids[motion_ids]
        end_ids = self.motion_end_ids[motion_ids]
        motion_len = self.motion_len[motion_ids]

        time_in_proportion = torch.rand(batch_size, device=self.device)
        clip_tail_proportion = (self.num_steps / motion_len)
        
        # Fix: Convert 0 to a tensor with the same device/dtype as clip_tail_proportion
        min_val = torch.zeros_like(clip_tail_proportion)
        time_in_proportion = time_in_proportion.clamp(min_val, 1 - clip_tail_proportion)

        motion_ids = start_ids + torch.floor(time_in_proportion * (end_ids - start_ids)).long()
        motion_dof = self._get_amp_obs(motion_ids).view(batch_size, -1)

        ratio = self.fps / self.env_fps
        ratio *= torch.rand(batch_size, device=self.device) * 1.0 + 0.25  # Random ratio per sample

        for i in range(1, self.num_steps):
            next_pos = motion_ids + i * ratio
            floor = torch.floor(next_pos).long()
            ceil = floor + 1
            
            max_idx = self.motion_dof_pos.shape[0] - 1
            floor = torch.clamp(floor, 0, max_idx)
            ceil = torch.clamp(ceil, 0, max_idx)

            linear_ratio = (next_pos - floor).unsqueeze(-1)
            floor_idx = floor.to(self.storage_device)
            ceil_idx = ceil.to(self.storage_device)
            lr = linear_ratio.to(self.storage_device)
            motion_dof_next = self._get_amp_obs_blend(floor_idx, ceil_idx, lr)
            motion_dof = torch.cat([motion_dof, motion_dof_next], dim=-1).view(batch_size, -1)

        return motion_dof.to(self.device, non_blocking=True)

    def _get_projected_gravity_from_rpy(self, rpy):
        quat = euler_xyz_to_quat(rpy)
        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=rpy.device).repeat(rpy.shape[0], 1)
        return quat_rotate_inverse(quat, gravity)

    def _project_gravity(self, frame_ids):
        idx = frame_ids.to(self.storage_device)
        return self._get_projected_gravity_from_rpy(self.motion_base_rpy[idx]).to(self.device)

    def _get_amp_obs_blend(self, floor_idx, ceil_idx, linear_ratio):
        dof_pos = (
            self.motion_dof_pos[floor_idx] * (1 - linear_ratio) + self.motion_dof_pos[ceil_idx] * linear_ratio
        ).to(self.device)
        if self.include_dof_vel:
            dof_vel = (
                self.motion_dof_vel[floor_idx] * (1 - linear_ratio) + self.motion_dof_vel[ceil_idx] * linear_ratio
            ).to(self.device)
        else:
            dof_vel = torch.zeros_like(dof_pos)

        if self.amp_obs_type == "lower_body_state":
            base_ang_vel = (
                self.motion_base_ang_vel[floor_idx] * (1 - linear_ratio)
                + self.motion_base_ang_vel[ceil_idx] * linear_ratio
            ).to(self.device)
            gravity = self._project_gravity(floor_idx)
            return build_lower_body_amp_step_obs(
                dof_pos,
                dof_vel,
                torch.zeros_like(base_ang_vel),
                base_ang_vel,
                gravity,
            )

        if self.include_dof_vel:
            return torch.cat((dof_pos, dof_vel), dim=-1)
        return dof_pos

    def _get_amp_dof_obs(self, frame_ids):
        idx = frame_ids.to(self.storage_device)
        dof_pos = self.motion_dof_pos[idx]
        if not self.include_dof_vel:
            return dof_pos.to(self.device)
        dof_vel = self.motion_dof_vel[idx]
        return torch.cat((dof_pos, dof_vel), dim=-1).to(self.device)

    def _get_amp_lower_body_state_obs(self, frame_ids):
        idx = frame_ids.to(self.storage_device)
        q_leg = self.motion_dof_pos[idx].to(self.device)
        dq_leg = self.motion_dof_vel[idx].to(self.device) if self.include_dof_vel else torch.zeros_like(q_leg)
        base_ang_vel = self.motion_base_ang_vel[idx].to(self.device)
        projected_gravity = self._get_projected_gravity_from_rpy(self.motion_base_rpy[idx]).to(self.device)
        return build_lower_body_amp_step_obs(
            q_leg,
            dq_leg,
            torch.zeros_like(base_ang_vel),
            base_ang_vel,
            projected_gravity,
        )

    def _get_amp_obs(self, frame_ids):
        if self.amp_obs_type == "lower_body_state":
            return self._get_amp_lower_body_state_obs(frame_ids)
        return self._get_amp_dof_obs(frame_ids)

    def sample_reference_state(self, batch_size, random_time=True):
        motion_ids = torch.randint(0, self.num_motion, (batch_size,), device=self.device)
        start_ids = self.motion_start_ids[motion_ids]
        end_ids = self.motion_end_ids[motion_ids]
        if random_time:
            frame_ids = start_ids + torch.floor(
                torch.rand(batch_size, device=self.device) * (end_ids - start_ids).float()
            ).long()
        else:
            frame_ids = start_ids
        idx = frame_ids.to(self.storage_device)
        return {
            "dof_pos": self.motion_dof_pos[idx].to(self.device),
            "dof_vel": self.motion_dof_vel[idx].to(self.device),
            "base_z": self.motion_base_pos[idx, 2].to(self.device),
        }
