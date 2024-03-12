import numpy as np
import matplotlib.pyplot as plt

random_confidence_map = np.random.rand(1,1)

# 使用Matplotlib进行可视化
plt.imshow(random_confidence_map, cmap="Greens", interpolation="nearest")
# 在每个像素点上标出具体的数据值
for i in range(random_confidence_map.shape[0]):
    for j in range(random_confidence_map.shape[1]):
        plt.text(
            j,
            i,
            f"{random_confidence_map[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=50,
        )
plt.axis("off")  # 取消横纵轴的显示
# 注意，这里不再调用plt.colorbar()来避免显示颜色条
plt.savefig("1x1置信图.svg")
