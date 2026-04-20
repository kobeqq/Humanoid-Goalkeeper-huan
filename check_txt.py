import json
import numpy as np
import matplotlib.pyplot as plt

# ========= 1. 读取数据 =========
file_path = "/home/huan/下载/k1_motion_amp/back_step_002_0500_2100_back_step.scale.txt"

with open(file_path, "r") as f:
    data = json.load(f)

frames = np.array(data["Frames"])   # (T, D)
print("数据形状:", frames.shape)

# ========= 2. 提取 root =========
root_pos = frames[:, 0:3]   # x, y, z

# ========= 3. 画 root 曲线 =========
plt.figure()
plt.plot(root_pos[:, 0], label="x (横向)")
plt.plot(root_pos[:, 1], label="y (前后)")
plt.plot(root_pos[:, 2], label="z (高度)")
plt.legend()
plt.title("Root Position")
plt.show()

# ========= 4. 画轨迹（俯视图） =========
plt.figure()
plt.plot(root_pos[:, 0], root_pos[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory (Top View)")
plt.axis("equal")
plt.grid()
plt.show()

# ========= 5. 关节是否在动 =========
plt.figure()
plt.plot(frames[:, 10:20])   # 随便看10个关节
plt.title("Joint Signals")
plt.show()

# ========= 6. 动画显示轨迹 =========
plt.figure()
for i in range(0, len(root_pos), 10):  # 每10帧播放
    plt.cla()
    plt.plot(root_pos[:i, 0], root_pos[:i, 1], 'b-')
    plt.scatter(root_pos[i, 0], root_pos[i, 1], c='r')
    
    plt.xlim(np.min(root_pos[:,0])-0.5, np.max(root_pos[:,0])+0.5)
    plt.ylim(np.min(root_pos[:,1])-0.5, np.max(root_pos[:,1])+0.5)
    
    plt.title(f"Frame {i}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pause(0.01)

plt.show()

# ========= 7. 判断是否横移 =========
dx = root_pos[-1, 0] - root_pos[0, 0]
dy = root_pos[-1, 1] - root_pos[0, 1]

print("\n===== 运动分析 =====")
print("横向位移 x:", dx)
print("前向位移 y:", dy)

if abs(dx) > abs(dy):
    print("👉 这是【横移为主】的动作")
else:
    print("👉 这是【前进/后退为主】的动作")