# CycleGAN
Pytorch实现版本

## 模型介绍
论文：Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

论文地址：https://arxiv.org/abs/1703.10593

![img.png](assets/img.png)

CycleGAN(循环生成对抗网络)是一种特殊的GAN，用于解决无监督学习中的图像到图像的翻译问题。训练时不需要成对的配对样本，只需要源域和目标域的图像。训练后网络就能实现对图像源域到目标域的迁移。解决了模型需要成对数据进行训练的困难。

**特点：**

1. **双向训练**：CycleGAN包含两个生成器和两个判别器。一个生成器负责将图像从域A翻译到域B，另一个则相反。对应地，每个域都有一个判别器。
2. **循环一致性损失（Cycle Consistency Loss）**：这是CycleGAN的核心创新。它确保如果一个图像从域A翻译到域B，然后再翻译回域A，最终得到的图像应该与原始图像尽可能相似。这样的损失函数帮助模型学习在没有成对数据的情况下进行有效的翻译。
3. **无需成对数据**：与需要成对数据的传统图像到图像的翻译方法不同，CycleGAN不需要成对的训练样本。这大大增加了其应用范围，因为在许多情况下，获取精确配对的训练数据是不切实际的。

区别于pix2pixGAN：p2p需要成对的数据集

**模型架构图**

![img](assets/68873220482849de9624f9deb6a9b80b.png)

**(a)图**展示了cycleGAN的基本架构： 展示了两个域X和Y之间的基本CycleGAN结构，包括两个生成器（G和F）和两个判别器（D_X和D_Y）。

- **生成器G**: 负责将域X的图像转换到域Y。
- **生成器F**: 负责将域Y的图像转换回域X。
- **判别器$D_X$**: 判别域X中的图像是否为真实图像。
- **判别器$D_Y$**: 判别域Y中的图像是否为真实图像。

生成器和判别器的损失函数都是由对抗损失和循环一致性损失组成的。

**部分(b) 和 (c)** 展示了循环一致性损失的概念：在CycleGAN中，不仅要求生成器能将图像从一个域转换到另一个域，还要求这种转换是可逆的. 

即如果我们先通过G将图像从X转换到Y得到$\hat{Y}$，再通过F将$\hat{Y}$转换回X域得到$\hat{X}$，这个$\hat{X}$应该与原始的X非常接近。同理，这个$\hat{Y}$也应该与原始的Y非常接近。

这就是所谓的循环一致性损失，它帮助模型在没有成对样本的情况下学习到有意义的图像转换。

## 代码文件夹结构

* cyclegan
    * train.py --- 训练代码。
    * datasets.py --- 数据加载和预处理。
    * models.py   --- CycleGAN中使用的神经网络架构，包括生成器和判别器模型。
    * utils.py

###  models.py

cyclegan模型实现的细节，包含了一个生成器类(GeneratorResNet)和判别器类(Discriminator)。以及一个残差块(ResidualBlock)和权重初始化的方法(weights_init_normal)

生成器包含了：

1. 初始卷积块 (outchannels = 64)
2. 两个下采样块 (outchannels \*=2  卷积stride = 2)
3. n个残差块  (outchannels = outchannels)
4. 两个上采样块  (outchannels //=2  卷积stride = 1 Upsample方式='nearest')
5. 输出层

判别器包含了4个鉴别器块

CycleGAN 的鉴别器基于 PatchGAN 的概念设计。PatchGAN 通过对输入图像的不同区域（补丁）分别进行真假判断，而不是对整个图像作为一个整体进行判断。这有助于模型更加细致地区分图像的局部特征，使其能够更有效地区分真实图像和生成图像的局部差异。

`self.output_shape` 在 CycleGAN 的鉴别器中表示输出的形状。这个形状是基于输入形状和网络的结构计算出来的。在这个特定的实现中，它是根据网络中的下采样步骤来计算的。

### datasets.py

