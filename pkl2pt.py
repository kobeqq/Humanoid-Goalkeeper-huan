import os
import pickle
import torch
import numpy as np

def convert_pkl_to_pt(input_pkl_path, output_pt_path):
    # 1. 加载原始 pkl 数据
    print(f"Loading data from {input_pkl_path}...")
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    # 提取字段并处理 fps
    root_pos = data['base_position']
    root_rot = data['base_pose']
    dof_pos = data['joint_position']
    dof_vel = data['joint_velocity']
    fps = data.get('fps', 30.0)  # 如果没有提供 fps，默认兜底设为 30
    dt = 1.0 / fps

    # 2. 转换为 PyTorch FloatTensor
    base_position = torch.tensor(root_pos).float()   # shape: (T, 3)
    base_pose = torch.tensor(root_rot).float()       # shape: (T, 4)
    joint_position = torch.tensor(dof_pos).float()   # shape: (T, 29)
    joint_velocity = torch.tensor(dof_vel).float()   # shape: (T, 29)
    
    T, num_joints = joint_position.shape
    
    # 3. 计算 joint_velocity
    # 初始化全 0 tensor
    # joint_velocity = torch.zeros_like(joint_position)
    
    # if T > 1:
    #     # 使用差分计算前 T-1 帧的速度: v[t] = (p[t+1] - p[t]) / dt
    #     joint_velocity[:-1] = (joint_position[1:] - joint_position[:-1]) / dt
    #     # 末帧复制前一帧的速度以对齐 shape
    #     joint_velocity[-1] = joint_velocity[-2]
    
    # 4. 组装 MotionLib 需要的字典格式
    motion_dict = {
        'base_position': base_position,
        'base_pose': base_pose,
        'joint_position': joint_position,
        'joint_velocity': joint_velocity,
        'fps': fps  # 保留 fps 方便后续读取对齐
    }
    
    # 5. 确保目标文件夹存在并保存
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save(motion_dict, output_pt_path)
    
    # 打印信息用于验证
    print(f"Successfully saved to: {output_pt_path}")
    print("--- Tensor Shapes ---")
    print(f"base_position:  {motion_dict['base_position'].shape}")
    print(f"base_pose:      {motion_dict['base_pose'].shape}")
    print(f"joint_position: {motion_dict['joint_position'].shape}")
    print(f"joint_velocity: {motion_dict['joint_velocity'].shape}")

if __name__ == "__main__":
    # 根据你的实际相对/绝对路径修改这里的 input 路径
    INPUT_PKL = "/home/huan/GMR/out_puts/k1/goalkeeper_k1.pkl" 
    OUTPUT_PT = "/home/huan/Humanoid-Goalkeeper-huan/legged_gym/resources/datasets/goalkeeper_from_pkl_k1/goalkeeper_from_pkl_k1.pt"
    
    convert_pkl_to_pt(INPUT_PKL, OUTPUT_PT)