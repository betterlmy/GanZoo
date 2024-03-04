# 假设有一个序列，序列长度为6，随机生成一个自注意力权重矩阵来模拟自注意力机制
import numpy as np
import matplotlib.pyplot as plt

sequence_length = 5
attention_weights = np.random.rand(sequence_length, sequence_length)

# 使用热力图来表示这些权重
plt.figure(figsize=(4, 4))
plt.imshow(attention_weights, cmap="Reds", interpolation="none")
plt.colorbar()
plt.xlabel("Query Position")
plt.ylabel("Key Position")

# 添加每个单元格的权重值
for i in range(sequence_length):
    for j in range(sequence_length):
        text = plt.text(
            j,
            i,
            f"{attention_weights[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
        )

plt.savefig("self_attention_weights.png")
