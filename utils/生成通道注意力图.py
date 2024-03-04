import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

# 随机生成一些数据来表示通道和它们的注意力权重
channel_count = 10
attention_weights = np.random.rand(channel_count)
channels = np.arange(channel_count)

# 创建一个3D图
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="3d")

# 生成条形图的数据
xpos = channels
ypos = np.zeros(channel_count)
zpos = np.zeros(channel_count)
dx = np.ones(channel_count) * 0.5  # 条形的宽度
dy = np.ones(channel_count) * 0.5  # 条形的深度
dz = attention_weights  # 条形的高度，即权重

# 为每个条形设置不同的颜色
colors = plt.cm.Reds(dz / max(dz))

# 绘制条形图
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

# 设置坐标轴标签
ax.set_xlabel("Channel")
ax.set_zlabel("Attention Weight")

# 隐藏Y轴标签
ax.get_yaxis().set_visible(False)

# 展示图表
plt.savefig("attention_weights.png")
