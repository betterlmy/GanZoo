import numpy as np
import matplotlib.pyplot as plt

# 生成一个6x6的空间注意力图，这里简单使用随机数来模拟注意力权重
attention_map = np.random.rand(6, 6)

# 使用matplotlib来绘制注意力图
plt.figure(figsize=(6, 6))
plt.imshow(attention_map, cmap="Reds")
plt.colorbar()
# plt.show()
plt.savefig("attention_map.png")
