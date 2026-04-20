from isaacgym import gymapi

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
sim_params.use_gpu_pipeline = False
sim_params.physx.use_gpu = False

asset_root = "./"
asset_file = "legged_gym/resources/robots/k1/urdf/K1_22dof.urdf"

asset_options = gymapi.AssetOptions()
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 获取 joint 数量
num_joints = gym.get_asset_joint_count(asset)

print("=== Isaac Gym Joint List ===")

joint_names = []
for i in range(num_joints):
    name = gym.get_asset_joint_name(asset, i)
    joint_type = gym.get_asset_joint_type(asset, i)

    joint_names.append(name)

    print(f"{i:02d} | {name:20s} | type={joint_type}")

# 建立 name -> index 映射（非常关键）
joint_name_to_idx = {
    name: i for i, name in enumerate(joint_names)
}


print("\n=== Joint Mapping ===")
print(joint_name_to_idx)

num_dofs = gym.get_asset_dof_count(asset)

print("\n=== DOF List ===")
for i in range(num_dofs):
    dof_name = gym.get_asset_dof_name(asset, i)
    print(f"{i:02d} | {dof_name}")



